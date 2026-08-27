#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-7B: compare correct-vs-wrong Direction geometry and optionally test
causality of the learned Direction subspace.

Expected existing outputs:
  <direction-dir>/vectors.npz
  <direction-dir>/sample_split_and_generation.csv
  <group-root>/restricted_direction_groups.csv

Run from AdaptVis/llava16 repository root.

Offline only:
python analyze_direction_geometry_and_causality_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --group-root output/qwen7b_direction_conditioned_failure_v1 \
  --layers 14,16,18 \
  --output-dir output/qwen7b_direction_geometry_causal_v1

Causal run:
CUDA_VISIBLE_DEVICES=0 python analyze_direction_geometry_and_causality_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --group-root output/qwen7b_direction_conditioned_failure_v1 \
  --layers 14,16,18 \
  --model qwen-7b \
  --device cuda:0 \
  --run-causal \
  --output-dir output/qwen7b_direction_geometry_causal_v1 \
  --overwrite-causal
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as dirscan
import analyze_text_stream_visual_causal_transfer_v1 as causal


RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--group-root", required=True)
    p.add_argument("--layers", default="14,16,18")
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--run-causal", action="store_true")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager","sdpa","flash_attention_2","none"])
    p.add_argument("--prompt-template", default=(
        "Determine the spatial relation of the {subject} to the {reference} "
        "in the image. Answer with left, right, above, or below."
    ))
    p.add_argument(
        "--groups",
        default="restricted_correct_repr_strong,restricted_wrong_repr_strong"
    )
    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument("--overwrite-causal", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str,str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_layers(s):
    out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("No layers selected")
    return list(dict.fromkeys(out))


def mean(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def fraction(xs):
    xs = [bool(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def fit_codebook(X_train, y_train):
    center = X_train.mean(axis=0)
    Xc = X_train - center
    protos = []
    for rel in RELATIONS:
        m = (y_train == rel)
        if not np.any(m):
            raise RuntimeError(f"No train samples for {rel}")
        v = Xc[m].mean(axis=0)
        v = v / max(float(np.linalg.norm(v)), EPS)
        protos.append(v)
    return center.astype(np.float32), np.stack(protos).astype(np.float32)


def spatial_basis_from_prototypes(protos):
    x = protos[REL2ID["right"]] - protos[REL2ID["left"]]
    y = protos[REL2ID["above"]] - protos[REL2ID["below"]]
    A = np.stack([x,y], axis=1).astype(np.float64)
    Q, R = np.linalg.qr(A)
    keep = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, keep]
    if Q.shape[1] == 0:
        raise RuntimeError("Spatial basis rank is 0")
    return Q.astype(np.float32)


def bootstrap_diff(a, b, n_boot, rng):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    obs = float(a.mean() - b.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = aa.mean() - bb.mean()
    lo, hi = np.percentile(boots, [2.5,97.5])
    return obs, float(lo), float(hi)


def load_assets(direction_dir, group_root, layers):
    with np.load(direction_dir/"vectors.npz", allow_pickle=True) as z:
        arr = {k:z[k] for k in z.files}
    split = read_csv(direction_dir/"sample_split_and_generation.csv")
    groups = read_csv(group_root/"restricted_direction_groups.csv")

    sids = arr["sample_index"].astype(np.int64)
    labels = np.asarray([dirscan.norm_relation(x) for x in arr["relation"]])
    residual = np.asarray(arr["residual"], dtype=np.float32)

    idx_by_sid = {int(s):i for i,s in enumerate(sids.tolist())}
    train_sids = {
        int(r["sample_index"]) for r in split if r["split"].strip()=="train"
    }
    train_idx = np.asarray(
        [idx_by_sid[s] for s in train_sids if s in idx_by_sid], dtype=np.int64
    )

    codebooks = {}
    for li in layers:
        center, protos = fit_codebook(residual[train_idx,li,:], labels[train_idx])
        basis = spatial_basis_from_prototypes(protos)
        codebooks[li] = dict(center=center, protos=protos, basis=basis)

    return arr, groups, idx_by_sid, labels, residual, codebooks


def score_one(x, y, cb):
    center, protos, basis = cb["center"], cb["protos"], cb["basis"]
    xc = x-center
    n = max(float(np.linalg.norm(xc)), EPS)
    xn = xc/n
    scores = xn @ protos.T
    gi = REL2ID[str(y)]
    wrong = [float(scores[j]) for j in range(4) if j != gi]
    proj = xc @ basis
    proj_norm = float(np.linalg.norm(proj))
    return {
        "probe_correct": int(int(np.argmax(scores))==gi),
        "gt_cosine": float(scores[gi]),
        "best_wrong_cosine": max(wrong),
        "direction_margin": float(scores[gi]-max(wrong)),
        "residual_norm": float(np.linalg.norm(x)),
        "centered_norm": float(n),
        "spatial_projection_norm": proj_norm,
        "spatial_energy_fraction": float((proj_norm**2)/max(n**2,EPS)),
    }


def run_offline(groups, idx_by_sid, labels, residual, codebooks,
                layers, selected_groups, out_dir, bootstrap, seed):
    g_by_sid = {int(r["sid"]):r["restricted_group"] for r in groups}
    rows = []
    for sid,g in g_by_sid.items():
        if g not in selected_groups and g != "restricted_wrong_repr_weak":
            continue
        if sid not in idx_by_sid:
            continue
        idx = idx_by_sid[sid]
        for li in layers:
            rows.append({
                "sid":sid,
                "restricted_group":g,
                "relation":labels[idx],
                "layer":li,
                **score_one(residual[idx,li,:], labels[idx], codebooks[li]),
            })
    write_csv(out_dir/"direction_geometry_per_sample.csv", rows)

    metrics = [
        "probe_correct","gt_cosine","best_wrong_cosine","direction_margin",
        "residual_norm","centered_norm","spatial_projection_norm",
        "spatial_energy_fraction"
    ]

    summary = []
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["restricted_group"],int(r["layer"]))].append(r)

    for (g,li),rs in sorted(buckets.items()):
        item = {"restricted_group":g,"layer":li,"n":len(rs)}
        for m in metrics:
            vals = np.asarray([float(r[m]) for r in rs])
            item[f"{m}_mean"] = float(vals.mean())
            item[f"{m}_median"] = float(np.median(vals))
            item[f"{m}_std"] = float(vals.std())
        summary.append(item)
    write_csv(out_dir/"direction_geometry_group_summary.csv", summary)

    Aname = "restricted_correct_repr_strong"
    Bname = "restricted_wrong_repr_strong"
    rng = np.random.default_rng(seed)
    comp = []
    for li in layers:
        A = [r for r in rows if r["restricted_group"]==Aname and r["layer"]==li]
        B = [r for r in rows if r["restricted_group"]==Bname and r["layer"]==li]
        if not A or not B:
            continue
        for m in metrics:
            a = [float(r[m]) for r in A]
            b = [float(r[m]) for r in B]
            diff,lo,hi = bootstrap_diff(a,b,bootstrap,rng)
            comp.append({
                "layer":li,"metric":m,
                "n_correct_strong":len(a),"n_wrong_strong":len(b),
                "correct_strong_mean":mean(a),
                "wrong_strong_mean":mean(b),
                "correct_minus_wrong":diff,
                "bootstrap95_lo":lo,
                "bootstrap95_hi":hi,
            })
    write_csv(out_dir/"correct_strong_vs_wrong_strong_geometry.csv", comp)

    relsum = []
    rb = defaultdict(list)
    for r in rows:
        rb[(r["restricted_group"],int(r["layer"]),r["relation"])].append(r)
    for (g,li,rel),rs in sorted(rb.items()):
        relsum.append({
            "restricted_group":g,"layer":li,"relation":rel,"n":len(rs),
            "probe_acc":mean(r["probe_correct"] for r in rs),
            "mean_margin":mean(r["direction_margin"] for r in rs),
            "mean_gt_cosine":mean(r["gt_cosine"] for r in rs),
            "mean_spatial_energy_fraction":mean(
                r["spatial_energy_fraction"] for r in rs
            ),
        })
    write_csv(out_dir/"direction_geometry_by_relation.csv", relsum)

    print("\n"+"="*110)
    print("OFFLINE DIRECTION GEOMETRY")
    print("="*110)
    print("layer  group                            N   probeAcc   margin    GTcos   spatialFrac")
    for li in layers:
        for g in [Aname,Bname,"restricted_wrong_repr_weak"]:
            rs = [r for r in rows if r["layer"]==li and r["restricted_group"]==g]
            if rs:
                print(
                    f"L{li:02d}   {g:<31s} {len(rs):3d}   "
                    f"{mean(r['probe_correct'] for r in rs):.3f}      "
                    f"{mean(r['direction_margin'] for r in rs):+.4f}   "
                    f"{mean(r['gt_cosine'] for r in rs):+.4f}   "
                    f"{mean(r['spatial_energy_fraction'] for r in rs):.4f}"
                )

    print("\nCorrect-strong minus wrong-strong:")
    for r in comp:
        if r["metric"] in {
            "direction_margin","gt_cosine",
            "spatial_projection_norm","spatial_energy_fraction"
        }:
            print(
                f"L{int(r['layer']):02d} {r['metric']:<28s} "
                f"{float(r['correct_minus_wrong']):+.5f} "
                f"[{float(r['bootstrap95_lo']):+.5f}, "
                f"{float(r['bootstrap95_hi']):+.5f}]"
            )


def pool_hidden(x, positions):
    pos = [int(p) for p in positions if 0 <= int(p) < int(x.shape[1])]
    idx = torch.as_tensor(pos, device=x.device, dtype=torch.long)
    return x[0].index_select(0,idx).mean(dim=0)


class RemoveDirectionSubspace:
    def __init__(self, module, layer, subj_pos, ref_pos, noimg_diff_cpu,
                 center_cpu, basis_cpu, mode, random_basis_cpu=None):
        self.layer = int(layer)
        self.subj = list(map(int,subj_pos))
        self.ref = list(map(int,ref_pos))
        self.noimg_diff_cpu = noimg_diff_cpu
        self.center_cpu = center_cpu
        self.basis_cpu = basis_cpu
        self.random_basis_cpu = random_basis_cpu
        self.mode = mode
        self.applied = 0
        self.delta_norm = float("nan")
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self,_m,_a,output):
        if self.applied:
            return output
        x = causal.first_tensor(output)
        hs = pool_hidden(x,self.subj)
        hr = pool_hidden(x,self.ref)
        noimg = self.noimg_diff_cpu.to(x.device,dtype=x.dtype)
        center = self.center_cpu.to(x.device,dtype=x.dtype)
        B = self.basis_cpu.to(x.device,dtype=x.dtype)

        q = (hs-hr) - noimg - center
        spatial = B @ (B.T @ q)

        if self.mode=="spatial":
            delta = spatial
        elif self.mode=="random_matched":
            R = self.random_basis_cpu.to(x.device,dtype=x.dtype)
            dr = R @ (R.T @ q)
            nr = dr.norm()
            ns = spatial.norm()
            delta = torch.zeros_like(spatial) if float(nr.detach().cpu()) <= EPS else dr*(ns/nr)
        else:
            raise ValueError(self.mode)

        self.delta_norm = float(delta.detach().float().norm().cpu())
        y = x.clone()
        d = delta/2.0
        y[0,self.subj,:] -= d
        y[0,self.ref,:] += d
        self.applied += 1
        return causal.replace_first_tensor(output,y)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def make_random_basis(hidden, rank, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((hidden,rank))
    Q,_ = np.linalg.qr(A)
    return torch.from_numpy(Q[:,:rank].astype(np.float32))


def capture_noimg_diffs(model,processor,device,decoder_layers,layers,
                        question,subject,reference):
    rendered = dirscan.build_chat_prompt(processor,question,False)
    batch = dirscan.process_inputs(processor,rendered,None,device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    spos = dirscan.locate_phrase_positions(processor.tokenizer,ids,subject)
    rpos = dirscan.locate_phrase_positions(processor.tokenizer,ids,reference)

    with causal.LayerOutputCollector(decoder_layers,layers) as col:
        with torch.inference_mode():
            model(**batch,output_attentions=False,output_hidden_states=False,
                  use_cache=False,return_dict=True)
        col.validate()

    out = {}
    for li in layers:
        h = col.states[li].unsqueeze(0)
        out[li] = (pool_hidden(h,spos)-pool_hidden(h,rpos)).detach().float().cpu()
    del batch
    return out


def summarize_causal(rows,out_path):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["restricted_group"],int(r["layer"]))].append(r)
    out = []
    for (g,li),rs in sorted(buckets.items()):
        out.append({
            "restricted_group":g,"layer":li,"n":len(rs),
            "baseline_acc":fraction(int(r["baseline_correct"])==1 for r in rs),
            "baseline_margin":mean(r["baseline_margin"] for r in rs),
            "spatial_remove_acc":fraction(int(r["spatial_correct"])==1 for r in rs),
            "spatial_margin_drop":mean(r["spatial_margin_drop"] for r in rs),
            "spatial_pred_change":fraction(int(r["spatial_pred_changed"])==1 for r in rs),
            "random_remove_acc":fraction(int(r["random_correct"])==1 for r in rs),
            "random_margin_drop":mean(r["random_margin_drop"] for r in rs),
            "spatial_minus_random_margin_drop":
                mean(r["spatial_margin_drop"] for r in rs) -
                mean(r["random_margin_drop"] for r in rs),
        })
    write_csv(out_path,out)

    print("\n"+"="*120)
    print("CAUSAL DIRECTION-SUBSPACE REMOVAL")
    print("="*120)
    print("group                              layer N  baseAcc spatialAcc randomAcc spDrop randDrop sp-rand")
    for g in ["restricted_correct_repr_strong","restricted_wrong_repr_strong"]:
        for r in out:
            if r["restricted_group"]==g:
                print(
                    f"{g:<34s} L{int(r['layer']):02d} {int(r['n']):3d} "
                    f"{r['baseline_acc']:.3f}   {r['spatial_remove_acc']:.3f}     "
                    f"{r['random_remove_acc']:.3f}    "
                    f"{r['spatial_margin_drop']:+.4f} "
                    f"{r['random_margin_drop']:+.4f} "
                    f"{r['spatial_minus_random_margin_drop']:+.4f}"
                )


def run_causal(args, groups, codebooks, layers, selected_groups, out_dir):
    cdir = out_dir/"causal"
    if args.overwrite_causal and cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    per_path = cdir/"per_sample.csv"

    selected = [r for r in groups if r["restricted_group"] in selected_groups]
    if args.max_per_group is not None:
        rng = random.Random(args.seed)
        tmp = []
        for g in selected_groups:
            rr = [r for r in selected if r["restricted_group"]==g]
            rng.shuffle(rr)
            tmp.extend(rr[:args.max_per_group])
        selected = tmp

    keep = {int(r["sid"]) for r in selected}
    group_by_sid = {int(r["sid"]):r["restricted_group"] for r in selected}

    records,_ = base.load_records(args.dataset,Path(args.data_root),None)
    records = [r for r in records if int(r.sid) in keep]

    spec = base.SPECS[args.model]
    cls = getattr(transformers,spec.model_class)
    kw = {
        "dtype":base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage":True,
        "trust_remote_code":spec.trust_remote_code,
        "device_map":{"":args.device},
    }
    if args.attn_impl!="none":
        kw["attn_implementation"] = args.attn_impl

    model = cls.from_pretrained(spec.repo_id,**kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model,processor)
    device = torch.device(args.device)

    decoder_layers,_ = causal.direction_base.resolve_decoder_layers(model)
    token_map = causal.relation_token_variants(processor.tokenizer)

    random_bases = {}
    for li in layers:
        B = codebooks[li]["basis"]
        random_bases[li] = make_random_basis(
            B.shape[0],B.shape[1],args.seed+10007*li
        )

    rows = []
    for rec in tqdm(records,desc="direction causal"):
        img = None
        batch = None
        try:
            sid = int(rec.sid)
            gt = dirscan.norm_relation(rec.relation)
            question = args.prompt_template.format(
                subject=rec.subject,reference=rec.reference
            )
            img = Image.open(rec.image_path).convert("RGB")
            batch,_,subj_pos,ref_pos = causal.build_batch(
                processor,rec,question,img,device
            )
            baseline = causal.score_forward(model,batch,token_map,gt)

            noimg = capture_noimg_diffs(
                model,processor,device,decoder_layers,layers,
                question,str(rec.subject),str(rec.reference)
            )

            for li in layers:
                cb = codebooks[li]
                result = {}
                dnorm = {}
                for mode in ["spatial","random_matched"]:
                    patch = RemoveDirectionSubspace(
                        decoder_layers[li],li,subj_pos,ref_pos,noimg[li],
                        torch.from_numpy(cb["center"]),
                        torch.from_numpy(cb["basis"]),
                        mode,random_bases[li]
                    )
                    try:
                        with patch:
                            res = causal.score_forward(model,batch,token_map,gt)
                    finally:
                        patch.close()
                    result[mode] = res
                    dnorm[mode] = patch.delta_norm

                sp = result["spatial"]
                rd = result["random_matched"]
                rows.append({
                    "sid":sid,
                    "restricted_group":group_by_sid[sid],
                    "relation":gt,
                    "layer":li,
                    "baseline_pred":baseline["pred"],
                    "baseline_correct":int(baseline["correct"]),
                    "baseline_margin":baseline["margin"],
                    "spatial_pred":sp["pred"],
                    "spatial_correct":int(sp["correct"]),
                    "spatial_margin":sp["margin"],
                    "spatial_margin_drop":baseline["margin"]-sp["margin"],
                    "spatial_pred_changed":int(sp["pred"]!=baseline["pred"]),
                    "random_pred":rd["pred"],
                    "random_correct":int(rd["correct"]),
                    "random_margin":rd["margin"],
                    "random_margin_drop":baseline["margin"]-rd["margin"],
                    "random_pred_changed":int(rd["pred"]!=baseline["pred"]),
                    "spatial_delta_norm":dnorm["spatial"],
                    "random_delta_norm":dnorm["random_matched"],
                })

            if len(rows)%20==0:
                write_csv(per_path,rows)

        finally:
            if img is not None:
                img.close()
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(per_path,rows)
    summarize_causal(rows,cdir/"summary.csv")


def main():
    args = parse_args()
    direction_dir = Path(args.direction_dir)
    group_root = Path(args.group_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    selected_groups = [x.strip() for x in args.groups.split(",") if x.strip()]

    arr,groups,idx_by_sid,labels,residual,codebooks = load_assets(
        direction_dir,group_root,layers
    )

    run_offline(
        groups,idx_by_sid,labels,residual,codebooks,layers,
        selected_groups,out_dir,args.bootstrap,args.seed
    )

    if args.run_causal:
        run_causal(
            args,groups,codebooks,layers,selected_groups,out_dir
        )

    (out_dir/"summary.json").write_text(
        json.dumps({
            "layers":layers,
            "groups":selected_groups,
            "run_causal":bool(args.run_causal),
            "causal_intervention":
                "remove centered image-noimage subject-reference projection "
                "on learned LR/AB spatial basis"
        },indent=2),
        encoding="utf-8"
    )

    print("\nSaved to",out_dir)


if __name__ == "__main__":
    main()
