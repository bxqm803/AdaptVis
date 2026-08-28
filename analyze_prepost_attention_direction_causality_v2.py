
# -*- coding: utf-8 -*-

"""
Pre-/post-Attention Semantic Direction causal scan.

Fix over the previous block-output experiment:
  * semantic Direction codebooks are fitted directly at the ACTUAL computation
    points, so there is no cache-layer indexing ambiguity;
  * PRE_ATTN intervention is made at decoder-block input, before Attention reads
    subject/reference tokens;
  * POST_ATTN intervention is made by editing Attention output, which is exactly
    equivalent to editing the post-Attention residual before both the MLP and
    residual branches;
  * gradient C is checked with very small +/- finite differences.

For stage s in {pre_attn, post_attn}:

  r_l,s = (h_sub-h_ref)^img - (h_sub-h_ref)^noimg
  d_l,s(g,c) = unit(mu_g - mu_c)

Representation availability:

  R = (r_l,s - center_l,s) dot d_l,s(g,c)

Local causal utilization:

  C = d/d eps [logit_g-logit_c]

under the pair-preserving edit

  h_sub += eps/2 * d ; h_ref -= eps/2 * d

Interpretation:
  R low, C normal/high -> information insufficiency candidate.
  R high, C low       -> information exists but is weakly utilized.
  R < 0, |C| high     -> wrong-side state on a causally relevant axis.
  C < 0               -> semantic / causal direction mismatch.

The script uses cached ACTUAL model.generate() correct/wrong grouping, but does
NOT reuse cached hidden states for the new codebooks.

Recommended smoke test:

CUDA_VISIBLE_DEVICES=0 python analyze_prepost_attention_direction_causality_v2.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \
  --layers 14,16,19 --max-train 32 --max-eval 20 \
  --fd-layers 14,16,19 --fd-max-samples 8 \
  --output-dir output/qwen7b_prepost_direction_causality_smoke --overwrite

Recommended full run:

CUDA_VISIBLE_DEVICES=0 python analyze_prepost_attention_direction_causality_v2.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \
  --layers auto --eval-split test \
  --fd-layers auto --fd-top-k 4 --fd-max-samples 30 \
  --output-dir output/qwen7b_prepost_direction_causality_v2 --overwrite
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as direction

RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
STAGES = ("pre_attn", "post_attn")
EPS = 1e-10


def args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--failure-dir", default=None)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--prompt-template", default=(
        "Determine the spatial relation of the {subject} to the {reference} "
        "in the image. Answer with left, right, above, or below."))
    p.add_argument("--layers", default="auto")
    p.add_argument("--min-role-gap", type=float, default=0.5)
    p.add_argument("--eval-split", default="test", choices=["train", "test", "all"])
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--control-quantile", type=float, default=0.25)
    p.add_argument("--random-controls", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fd-layers", default="auto")
    p.add_argument("--fd-top-k", type=int, default=4)
    p.add_argument("--fd-max-samples", type=int, default=30)
    p.add_argument("--fd-eps-scales", default="0.01,0.025,0.05,0.1,0.25")
    p.add_argument("--fd-random-control", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def safe_mean(xs):
    a = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    return float(a.mean()) if len(a) else float("nan")


def safe_frac(xs):
    x = list(xs)
    return float(np.mean(x)) if x else float("nan")


def parse_layers(text, n):
    t = str(text).strip().lower()
    if t == "all": return list(range(n))
    out = []
    for z in str(text).split(","):
        z = z.strip()
        if not z: continue
        if "-" in z:
            a, b = map(int, z.split("-", 1)); step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else: out.append(int(z))
    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n]
    if bad: raise ValueError(f"invalid layers {bad}; valid 0..{n-1}")
    return out


def parse_floats(text):
    x = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not x or any(v <= 0 for v in x): raise ValueError("eps scales must be positive")
    return x


def load_meta(direction_dir):
    dd = Path(direction_dir)
    with np.load(dd / "vectors.npz", allow_pickle=True) as z:
        sids = z["sample_index"].astype(np.int64)
        labels = np.asarray([direction.norm_relation(x) for x in z["relation"]], dtype=object)
    gt = {int(s): str(labels[i]) for i, s in enumerate(sids.tolist())}
    split, gen = {}, {}
    for r in read_csv(dd / "sample_split_and_generation.csv"):
        sid = int(r["sample_index"]); split[sid] = str(r.get("split", "")).strip()
        pred = direction.norm_relation(r.get("generation_pred", ""))
        group = str(r.get("generation_group", "")).strip().lower()
        if group not in ("correct", "wrong") and gt.get(sid) in REL2ID and pred in REL2ID:
            group = "correct" if pred == gt[sid] else "wrong"
        gen[sid] = {"generation_group": group, "generation_pred": pred,
                    "generation_text": str(r.get("generation_text", ""))}
    return {"sids": [int(x) for x in sids], "gt": gt, "split": split, "generation": gen}


def choose_layers(text, n_layers, failure_dir, min_gap):
    if str(text).strip().lower() != "auto":
        ls = parse_layers(text, n_layers)
        return ls, [{"layer": l, "selected": 1, "reason": "explicit"} for l in ls]
    if not failure_dir: raise ValueError("--failure-dir required for --layers auto")
    rows = read_csv(Path(failure_dir) / "top_candidate_layers.csv")
    out, audit = [], []
    for r in rows:
        l = int(r["layer"])
        sig = int(float(r.get("pairwise_deficit_ci_positive", 0)))
        gap = float(r["wrong_minus_correct_gt_minus_maxnonGT_gap"])
        yes = sig == 1 and gap <= -float(min_gap)
        audit.append({"layer": l, "pairwise_deficit_ci_positive": sig,
                      "role_gap": gap, "selected": int(yes)})
        if yes: out.append(l)
    out = sorted(set(out))
    if not out: raise RuntimeError("auto layer selection returned none")
    return out, audit


def get_attr_path(obj, path):
    for p in path.split("."): obj = getattr(obj, p)
    return obj


def decoder_layers(model):
    for path in ["model.language_model.layers", "language_model.layers",
                 "model.model.layers", "model.layers", "language_model.model.layers"]:
        try:
            x = get_attr_path(model, path)
            if len(x) and hasattr(x[0], "self_attn") and hasattr(x[0], "post_attention_layernorm"):
                return x, path
        except Exception: pass
    raise RuntimeError("cannot resolve decoder layers")


def first_tensor(x):
    if torch.is_tensor(x): return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y): return y
    raise RuntimeError(f"no tensor in {type(x)}")


def replace_first_tensor(output, new):
    if torch.is_tensor(output): return new
    if isinstance(output, tuple):
        q = list(output)
        for i, x in enumerate(q):
            if torch.is_tensor(x): q[i] = new; return tuple(q)
    if isinstance(output, list):
        q = list(output)
        for i, x in enumerate(q):
            if torch.is_tensor(x): q[i] = new; return q
    raise RuntimeError(f"cannot replace tensor in {type(output)}")


def pool(x, pos):
    idx = torch.as_tensor([int(p) for p in pos if 0 <= int(p) < x.shape[0]],
                          device=x.device, dtype=torch.long)
    if idx.numel() == 0: raise RuntimeError("empty phrase positions")
    return x.index_select(0, idx).mean(0)


class StageCapture:
    def __init__(self, layers, selected):
        self.data = defaultdict(dict); self.handles = []
        for l in selected:
            b = layers[l]
            def block_pre(_m, args, li=l):
                if args: self.data[li]["pre_attn"] = first_tensor(args)
            def post_pre(_m, args, li=l):
                if args: self.data[li]["post_attn"] = first_tensor(args)
            self.handles.append(b.register_forward_pre_hook(block_pre))
            self.handles.append(b.post_attention_layernorm.register_forward_pre_hook(post_pre))
    def validate(self, selected):
        miss = [(l, s) for l in selected for s in STAGES if s not in self.data.get(l, {})]
        if miss: raise RuntimeError(f"missing stage captures {miss}")
    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception): h.remove()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def build_batch(processor, rec, question, image, device):
    prompt = direction.build_chat_prompt(processor, question, image is not None)
    batch = direction.process_inputs(processor, prompt, image, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    sp = direction.locate_phrase_positions(processor.tokenizer, ids, str(rec.subject))
    rp = direction.locate_phrase_positions(processor.tokenizer, ids, str(rec.reference))
    return batch, sp, rp


def pooled_capture(cap, selected, sp, rp):
    out = {}
    for l in selected:
        out[l] = {}
        for stage in STAGES:
            x = cap.data[l][stage][0]
            out[l][stage] = (pool(x, sp) - pool(x, rp)).detach().float().cpu().numpy().astype(np.float32)
    return out


def capture_inference(model, processor, layers, selected, rec, image, device, prompt_template):
    q = prompt_template.format(subject=rec.subject, reference=rec.reference)
    batch, sp, rp = build_batch(processor, rec, q, image, device)
    with StageCapture(layers, selected) as cap:
        with torch.inference_mode():
            model(**batch, output_attentions=False, output_hidden_states=False,
                  use_cache=False, return_dict=True)
        cap.validate(selected)
        out = pooled_capture(cap, selected, sp, rp)
    del batch
    return out


def unit(v):
    v = np.asarray(v, dtype=np.float64); n = np.linalg.norm(v)
    return (v / max(float(n), EPS)).astype(np.float32)


def basis(vs):
    A = np.stack(vs, 1).astype(np.float64); u, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > 1e-8 * max(float(s.max()), 1.0)
    if not keep.any(): raise RuntimeError("degenerate spatial basis")
    return u[:, keep].astype(np.float32)


def fit_cb(X, y):
    center = X.mean(0).astype(np.float32); Xc = X - center
    means, protos = {}, {}
    for r in RELATIONS:
        m = Xc[y == r].mean(0).astype(np.float32); means[r] = m; protos[r] = unit(m)
    parr = np.stack([protos[r] for r in RELATIONS])
    B = basis([means["right"] - means["left"], means["above"] - means["below"]])
    std = {}
    for g in RELATIONS:
        for c in RELATIONS:
            if c == g: continue
            d = unit(protos[g] - protos[c]); std[(g, c)] = float(np.std(Xc @ d))
    return {"center": center, "means": means, "protos": protos,
            "proto_arr": parr, "basis": B, "axis_std": std}


def sem_axis(cb, g, c): return unit(cb["protos"][g] - cb["protos"][c])


def rep_metrics(v, cb, g, c):
    q = v - cb["center"]; d = sem_axis(cb, g, c)
    sc = q @ cb["proto_arr"].T
    return {"R_raw": float(q @ d),
            "R_cos": float(q @ d / max(float(np.linalg.norm(q)), EPS)),
            "direction_pred": RELATIONS[int(np.argmax(sc))], "axis": d}


def fit_actual_point_codebooks(model, processor, layers, selected, train_sids,
                                records, meta, device, prompt_template, outdir):
    vecs = defaultdict(list); ys = []; kept = []; errors = []
    for sid in tqdm(train_sids, desc="train actual-point codebooks"):
        img = None
        try:
            rec = records[sid]; img = Image.open(rec.image_path).convert("RGB")
            real = capture_inference(model, processor, layers, selected, rec, img,
                                     device, prompt_template)
            no = capture_inference(model, processor, layers, selected, rec, None,
                                   device, prompt_template)
            for l in selected:
                for s in STAGES: vecs[(l, s)].append(real[l][s] - no[l][s])
            ys.append(meta["gt"][sid]); kept.append(sid)
        except Exception as e:
            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})
        finally:
            if img is not None: img.close()
            gc.collect();
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    y = np.asarray(ys, dtype=object)
    cbs, diag = {}, []
    arrays = {"sid": np.asarray(kept), "relation": y,
              "layers": np.asarray(selected), "stages": np.asarray(STAGES, dtype=object)}
    for l in selected:
        for s in STAGES:
            X = np.stack(vecs[(l, s)]).astype(np.float32); arrays[f"L{l}_{s}"] = X
            cb = fit_cb(X, y); cbs[(l, s)] = cb
            pred = np.asarray([RELATIONS[i] for i in np.argmax((X-cb["center"]) @ cb["proto_arr"].T, axis=1)])
            diag.append({"layer": l, "stage": s, "n": len(X),
                         "train_direction_acc": float(np.mean(pred == y)),
                         "mean_norm": float(np.linalg.norm(X, axis=1).mean())})
    np.savez_compressed(Path(outdir)/"train_stage_vectors.npz", **arrays)
    write_csv(Path(outdir)/"stage_codebook_diagnostics.csv", diag)
    Path(outdir, "train_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    return cbs, diag


def tokenizer_ids(tok, text):
    try: return [int(x) for x in tok.encode(text, add_special_tokens=False)]
    except Exception:
        o = tok(text, add_special_tokens=False); x = o["input_ids"] if isinstance(o, dict) else o.input_ids
        if x and isinstance(x[0], (list, tuple)): x = x[0]
        return [int(v) for v in x]


def relation_tokens(tok):
    out = {}; unk = getattr(tok, "unk_token_id", None)
    for r in RELATIONS:
        ids = []
        for t in (r, " "+r, "\n"+r, r.capitalize(), " "+r.capitalize()):
            x = tokenizer_ids(tok, t)
            if len(x) == 1 and (unk is None or x[0] != unk): ids.append(x[0])
        ids = list(dict.fromkeys(ids))
        if not ids: raise RuntimeError(f"no one-token variant for {r}")
        out[r] = ids
    return out


def logits_of(out):
    for x in [getattr(out, "logits", None),
              getattr(getattr(out, "language_model_outputs", None), "logits", None),
              getattr(getattr(out, "text_model_output", None), "logits", None)]:
        if torch.is_tensor(x): return x
    if isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x) and x.ndim == 3: return x
    raise RuntimeError("cannot find logits")


def rel_scores(vec, token_map):
    vals = []
    for r in RELATIONS:
        ids = torch.as_tensor(token_map[r], device=vec.device, dtype=torch.long)
        vals.append(vec.index_select(0, ids).max())
    return torch.stack(vals)


def rand_orth(dim, B, n, seed):
    rng = np.random.default_rng(seed); B = B.astype(np.float64); out = []
    for _ in range(max(20, 20*n)):
        if len(out) >= n: break
        v = rng.standard_normal(dim); v -= B @ (B.T @ v); nv = np.linalg.norm(v)
        if nv > 1e-8: out.append((v/nv).astype(np.float32))
    return out


def scan_sample(model, processor, token_map, layers, selected, cbs, rec, sid,
                meta, device, prompt_template, random_controls, seed):
    gt = meta["gt"][sid]; gen = meta["generation"][sid]; group = gen["generation_group"]
    q = prompt_template.format(subject=rec.subject, reference=rec.reference)
    no = capture_inference(model, processor, layers, selected, rec, None, device, prompt_template)
    img = Image.open(rec.image_path).convert("RGB")
    try:
        batch, sp, rp = build_batch(processor, rec, q, img, device)
        with StageCapture(layers, selected) as cap:
            out = model(**batch, output_attentions=False, output_hidden_states=False,
                        use_cache=False, return_dict=True)
            cap.validate(selected)
            rs = rel_scores(logits_of(out)[0, -1], token_map); rnp = rs.detach().float().cpu().numpy()
            best = max([r for r in RELATIONS if r != gt], key=lambda r: rnp[REL2ID[r]])
            primary = gen["generation_pred"] if group == "wrong" else best
            inputs, keys = [], []
            for l in selected:
                for s in STAGES: inputs.append(cap.data[l][s]); keys.append((l, s))
            grads = {}
            for i, r in enumerate(RELATIONS):
                gg = torch.autograd.grad(rs[i], inputs, retain_graph=i < 3, allow_unused=True)
                grads[r] = {k: g for k, g in zip(keys, gg)}
            real = pooled_capture(cap, selected, sp, rp)
            rows = []
            for l in selected:
                for s in STAGES:
                    rv = real[l][s] - no[l][s]; cb = cbs[(l, s)]
                    for comp in RELATIONS:
                        if comp == gt: continue
                        rm = rep_metrics(rv, cb, gt, comp); d = rm["axis"]
                        gg = grads[gt][(l,s)]; cg = grads[comp][(l,s)]
                        if gg is None or cg is None: continue
                        gp = .5*(pool(gg[0].float(), sp)-pool(gg[0].float(), rp))
                        cp = .5*(pool(cg[0].float(), sp)-pool(cg[0].float(), rp))
                        mg = (gp-cp).detach().cpu().numpy().astype(np.float32)
                        C = float(mg @ d); B = cb["basis"]
                        proj = B @ (B.T @ mg); pn = np.linalg.norm(proj); gn = np.linalg.norm(mg)
                        align = float(proj @ d / pn) if pn > EPS else float("nan")
                        rnd = rand_orth(len(d), B, random_controls,
                                        seed + sid*100003 + l*1009 + (0 if s=="pre_attn" else 300007) + REL2ID[comp]*17)
                        ra = safe_mean(abs(float(mg @ x)) for x in rnd)
                        rows.append({"sid": sid, "layer": l, "stage": s, "gt": gt,
                                     "competitor": comp, "generation_group": group,
                                     "generation_pred": gen["generation_pred"],
                                     "is_primary_foil": int(comp == primary),
                                     "primary_foil": primary,
                                     "baseline_firststep_pred": RELATIONS[int(np.argmax(rnp))],
                                     "baseline_margin": float(rnp[REL2ID[gt]]-rnp[REL2ID[comp]]),
                                     "R_raw": rm["R_raw"], "R_cos": rm["R_cos"],
                                     "direction_pred": rm["direction_pred"],
                                     "direction_correct": int(rm["direction_pred"] == gt),
                                     "C_margin": C, "C_positive": int(C > 0),
                                     "alignment": align,
                                     "spatial_grad_fraction": float(pn/max(float(gn), EPS)),
                                     "random_abs_mean": ra,
                                     "specificity": float(abs(C)/ra) if math.isfinite(ra) and ra > EPS else float("nan")})
        del batch
        return rows
    finally: img.close()


def boot_gap(a, b, n, rng):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b): return (float("nan"),)*3
    obs=float(a.mean()-b.mean()); z=np.empty(n)
    for i in range(n): z[i]=a[rng.integers(0,len(a),len(a))].mean()-b[rng.integers(0,len(b),len(b))].mean()
    lo,hi=np.percentile(z,[2.5,97.5]); return obs,float(lo),float(hi)


def summarize(rows, selected, bootstrap, seed):
    rng=np.random.default_rng(seed); prim=[r for r in rows if r["is_primary_foil"]==1]; out=[]
    for l in selected:
        for s in STAGES:
            rr=[r for r in prim if r["layer"]==l and r["stage"]==s]
            C=[r for r in rr if r["generation_group"]=="correct"]; W=[r for r in rr if r["generation_group"]=="wrong"]
            z={"layer":l,"stage":s,"n_correct":len(C),"n_wrong":len(W)}
            for m in ("R_raw","C_margin","alignment","spatial_grad_fraction","specificity"):
                cv=[r[m] for r in C]; wv=[r[m] for r in W]; gap,lo,hi=boot_gap(cv,wv,bootstrap,rng)
                z[f"{m}_correct"]=safe_mean(cv); z[f"{m}_wrong"]=safe_mean(wv)
                z[f"{m}_gap_CminusW"]=gap; z[f"{m}_gap_ci95_lo"]=lo; z[f"{m}_gap_ci95_hi"]=hi
            z["Cpos_correct"]=safe_frac(r["C_margin"]>0 for r in C); z["Cpos_wrong"]=safe_frac(r["C_margin"]>0 for r in W)
            z["Rpos_Cnonpos_correct"]=safe_frac(r["R_raw"]>0 and r["C_margin"]<=0 for r in C)
            z["Rpos_Cnonpos_wrong"]=safe_frac(r["R_raw"]>0 and r["C_margin"]<=0 for r in W)
            out.append(z)
    return out


def thresholds_and_failures(rows, q):
    buckets=defaultdict(list)
    for r in rows:
        if r["generation_group"]=="correct": buckets[(r["layer"],r["stage"],r["gt"],r["competitor"])].append(r)
    th={}; throws=[]
    for k,rr in buckets.items():
        R=np.asarray([r["R_raw"] for r in rr]); C=np.asarray([r["C_margin"] for r in rr])
        th[k]={"Rq":float(np.quantile(R,q)),"Rm":float(np.median(R)),"Cq":float(np.quantile(C,q)),"Cm":float(np.median(C)),"n":len(rr)}
        l,s,g,c=k; throws.append({"layer":l,"stage":s,"gt":g,"competitor":c,**th[k]})
    fmap=[]
    for r in rows:
        if r["generation_group"]!="wrong" or not r["is_primary_foil"]: continue
        k=(r["layer"],r["stage"],r["gt"],r["competitor"])
        if k not in th: continue
        t=th[k]; rd=r["R_raw"]<t["Rq"]; cd=r["C_margin"]<t["Cq"]
        typ="both" if rd and cd else "representation_only" if rd else "utilization_only" if cd else "neither"
        fmap.append({"sid":r["sid"],"layer":r["layer"],"stage":r["stage"],"gt":r["gt"],"final_wrong":r["competitor"],
                     "R":r["R_raw"],"C":r["C_margin"],"representation_deficit":int(rd),"utilization_deficit":int(cd),
                     "failure_type":typ,"R_deficit_from_median":t["Rm"]-r["R_raw"],"C_deficit_from_median":t["Cm"]-r["C_margin"]})
    fsum=[]
    for l in sorted(set(r["layer"] for r in fmap)):
        for s in STAGES:
            rr=[r for r in fmap if r["layer"]==l and r["stage"]==s]; n=len(rr); c=Counter(r["failure_type"] for r in rr)
            fsum.append({"layer":l,"stage":s,"n_wrong":n,
                         "RepDef":safe_frac(r["representation_deficit"] for r in rr),
                         "UtilDef":safe_frac(r["utilization_deficit"] for r in rr),
                         "Both":c["both"]/n if n else float("nan"),"RepOnly":c["representation_only"]/n if n else float("nan"),
                         "UtilOnly":c["utilization_only"]/n if n else float("nan"),"Neither":c["neither"]/n if n else float("nan")})
    return throws,fmap,fsum


def transition(summary):
    d={(r["layer"],r["stage"]):r for r in summary}; out=[]
    for l in sorted(set(r["layer"] for r in summary)):
        a,b=d[(l,"pre_attn")],d[(l,"post_attn")]
        out.append({"layer":l,"pre_Rgap":a["R_raw_gap_CminusW"],"post_Rgap":b["R_raw_gap_CminusW"],
                    "attention_added_Rgap":b["R_raw_gap_CminusW"]-a["R_raw_gap_CminusW"],
                    "pre_Cgap":a["C_margin_gap_CminusW"],"post_Cgap":b["C_margin_gap_CminusW"],
                    "attention_changed_Cgap":b["C_margin_gap_CminusW"]-a["C_margin_gap_CminusW"]})
    return out


class PreIntervention:
    def __init__(self, block, sp, rp, delta):
        self.sp=list(map(int,sp)); self.rp=list(map(int,rp)); self.delta=torch.from_numpy(delta.astype(np.float32)); self.applied=False
        self.h=block.register_forward_pre_hook(self.hook)
    def hook(self,_m,args):
        if self.applied or not args:return None
        x=first_tensor(args); y=x.clone(); half=.5*self.delta.to(y.device,y.dtype)
        y[0,self.sp,:]+=half; y[0,self.rp,:]-=half; q=list(args)
        for i,z in enumerate(q):
            if torch.is_tensor(z): q[i]=y; break
        self.applied=True; return tuple(q)
    def close(self):
        with contextlib.suppress(Exception): self.h.remove()


class PostIntervention:
    def __init__(self, attn, sp, rp, delta):
        self.sp=list(map(int,sp)); self.rp=list(map(int,rp)); self.delta=torch.from_numpy(delta.astype(np.float32)); self.applied=False
        self.h=attn.register_forward_hook(self.hook)
    def hook(self,_m,_a,out):
        if self.applied:return out
        x=first_tensor(out); y=x.clone(); half=.5*self.delta.to(y.device,y.dtype)
        y[0,self.sp,:]+=half; y[0,self.rp,:]-=half; self.applied=True; return replace_first_tensor(out,y)
    def close(self):
        with contextlib.suppress(Exception): self.h.remove()


def score(model,batch,token_map):
    with torch.inference_mode(): rs=rel_scores(logits_of(model(**batch,output_attentions=False,output_hidden_states=False,use_cache=False,return_dict=True))[0,-1],token_map)
    a=rs.detach().float().cpu().numpy(); return {r:float(a[REL2ID[r]]) for r in RELATIONS}


def score_edit(model,batch,token_map,layers,l,stage,sp,rp,delta):
    h=PreIntervention(layers[l],sp,rp,delta) if stage=="pre_attn" else PostIntervention(layers[l].self_attn,sp,rp,delta)
    try:
        x=score(model,batch,token_map)
        if not h.applied: raise RuntimeError(f"{stage} hook not applied L{l}")
        return x
    finally:h.close()


def choose_fd(text, summary, n_layers, topk):
    if text.lower()=="none":return []
    if text.lower()!="auto":return parse_layers(text,n_layers)
    by=defaultdict(dict)
    for r in summary:by[r["layer"]][r["stage"]]=r
    x=[]
    for l,z in by.items():
        if len(z)<2:continue
        score=max(abs(z[s]["C_margin_gap_CminusW"]) for s in STAGES)
        x.append((score,l))
    return sorted([l for _,l in sorted(x,reverse=True)[:topk]])


def finite_diff(model,processor,token_map,layers,cbs,fd_layers,records,meta,gradrows,device,prompt_template,maxsamples,scales,random_control,seed):
    prim=[r for r in gradrows if r["is_primary_foil"] and r["layer"] in fd_layers]
    sids=sorted(set(r["sid"] for r in prim)); rng=random.Random(seed+99)
    if maxsamples and len(sids)>maxsamples:
        w=[s for s in sids if meta["generation"][s]["generation_group"]=="wrong"]; c=[s for s in sids if s not in set(w)]
        rng.shuffle(w);rng.shuffle(c); sids=sorted(set(w[:maxsamples//2]+c[:maxsamples-maxsamples//2]))
    look={(r["sid"],r["layer"],r["stage"]):r for r in prim}; rows=[]
    for sid in tqdm(sids,desc="small-epsilon FD"):
        rec=records[sid]; img=Image.open(rec.image_path).convert("RGB")
        try:
            q=prompt_template.format(subject=rec.subject,reference=rec.reference);batch,sp,rp=build_batch(processor,rec,q,img,device);base_sc=score(model,batch,token_map)
            for l in fd_layers:
                for stage in STAGES:
                    k=(sid,l,stage)
                    if k not in look:continue
                    g=look[k];gt=g["gt"];comp=g["competitor"];cb=cbs[(l,stage)];d=sem_axis(cb,gt,comp);sigma=cb["axis_std"][(gt,comp)] or 1.0
                    B=cb["basis"];rnd=rand_orth(len(d),B,1,seed+sid*911+l*17+(0 if stage=="pre_attn" else 3));rd=rnd[0] if rnd else None
                    bm=base_sc[gt]-base_sc[comp]
                    for scale in scales:
                        eps=scale*sigma
                        p=score_edit(model,batch,token_map,layers,l,stage,sp,rp,eps*d);m=score_edit(model,batch,token_map,layers,l,stage,sp,rp,-eps*d)
                        pm=p[gt]-p[comp];mm=m[gt]-m[comp];fd=(pm-mm)/(2*eps)
                        row={"sid":sid,"layer":l,"stage":stage,"group":g["generation_group"],"gt":gt,"competitor":comp,"eps_scale":scale,
                             "gradient_C":g["C_margin"],"fd_slope":fd,"same_sign":int(np.sign(fd)==np.sign(g["C_margin"]) and fd!=0 and g["C_margin"]!=0),
                             "monotonic":int(pm>bm>mm)}
                        if random_control and rd is not None:
                            rp1=score_edit(model,batch,token_map,layers,l,stage,sp,rp,eps*rd);rm1=score_edit(model,batch,token_map,layers,l,stage,sp,rp,-eps*rd)
                            rfd=((rp1[gt]-rp1[comp])-(rm1[gt]-rm1[comp]))/(2*eps);row["random_fd_slope"]=rfd;row["semantic_gt_random"]=int(abs(fd)>abs(rfd))
                        else:row["random_fd_slope"]=float("nan");row["semantic_gt_random"]=0
                        rows.append(row)
            del batch
        finally:img.close();gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
    sm=[]
    buckets=defaultdict(list)
    for r in rows:buckets[(r["layer"],r["stage"],r["eps_scale"],r["group"])].append(r)
    for k,rr in sorted(buckets.items()):
        l,s,e,g=k;gv=np.asarray([r["gradient_C"] for r in rr]);fv=np.asarray([r["fd_slope"] for r in rr]);rv=np.asarray([r["random_fd_slope"] for r in rr])
        corr=float(np.corrcoef(gv,fv)[0,1]) if len(rr)>1 and np.std(gv)>0 and np.std(fv)>0 else float("nan")
        sm.append({"layer":l,"stage":s,"eps_scale":e,"group":g,"n":len(rr),"gradC":float(gv.mean()),"fdSlope":float(fv.mean()),"corr":corr,
                   "sameSign":safe_frac(r["same_sign"] for r in rr),"monotonic":safe_frac(r["monotonic"] for r in rr),
                   "semantic_gt_random":safe_frac(r["semantic_gt_random"] for r in rr),"absFD":float(np.mean(np.abs(fv))),"absRandom":float(np.nanmean(np.abs(rv)))})
    return rows,sm


def print_results(summary,trans,fsum,fd):
    print("\n"+"="*160);print("PRE/POST-ATTENTION DIRECTION CAUSAL MAP");print("="*160)
    print("layer stage      R cor/wr gap | C cor/wr gap [95%CI] | align cor/wr | C>0 cor/wr")
    for r in summary:
        print(f"L{r['layer']:02d} {r['stage']:9s} {r['R_raw_correct']:+7.3f}/{r['R_raw_wrong']:+7.3f} {r['R_raw_gap_CminusW']:+7.3f} | "
              f"{r['C_margin_correct']:+8.5f}/{r['C_margin_wrong']:+8.5f} {r['C_margin_gap_CminusW']:+8.5f} "
              f"[{r['C_margin_gap_ci95_lo']:+7.5f},{r['C_margin_gap_ci95_hi']:+7.5f}] | "
              f"{r['alignment_correct']:+6.3f}/{r['alignment_wrong']:+6.3f} | {r['Cpos_correct']:.3f}/{r['Cpos_wrong']:.3f}")
    print("\nATTENTION TRANSITION (positive added_Rgap = attention enlarges representation gap)")
    for r in trans:print(f"L{r['layer']:02d}: Rgap {r['pre_Rgap']:+.3f}->{r['post_Rgap']:+.3f} add={r['attention_added_Rgap']:+.3f} | Cgap {r['pre_Cgap']:+.5f}->{r['post_Cgap']:+.5f}")
    print("\nWRONG FAILURE TYPES")
    for r in fsum:print(f"L{r['layer']:02d} {r['stage']:9s}: RepDef={r['RepDef']:.3f} UtilDef={r['UtilDef']:.3f} Both={r['Both']:.3f} RepOnly={r['RepOnly']:.3f} UtilOnly={r['UtilOnly']:.3f}")
    if fd:
        print("\nFINITE-DIFFERENCE SANITY")
        for r in fd:print(f"L{r['layer']:02d} {r['stage']:9s} eps={r['eps_scale']:.3f} {r['group']:7s} N={r['n']:2d} grad={r['gradC']:+.5f} fd={r['fdSlope']:+.5f} corr={r['corr']:+.3f} sign={r['sameSign']:.3f} mono={r['monotonic']:.3f} sem>rand={r['semantic_gt_random']:.3f}")


def main():
    a=args_parser();out=Path(a.output_dir)
    if a.overwrite and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    meta=load_meta(a.direction_dir);records,_=base.load_records(a.dataset,Path(a.data_root),None);records={int(r.sid):r for r in records}
    spec=base.SPECS[a.model];cls=getattr(transformers,spec.model_class);dtype=base.resolve_dtype(spec.dtype_name)
    kw={"low_cpu_mem_usage":True,"trust_remote_code":spec.trust_remote_code,"device_map":{"":a.device}}
    if a.attn_impl!="none":kw["attn_implementation"]=a.attn_impl
    try:model=cls.from_pretrained(spec.repo_id,dtype=dtype,**kw)
    except TypeError:model=cls.from_pretrained(spec.repo_id,torch_dtype=dtype,**kw)
    model.eval();processor=AutoProcessor.from_pretrained(spec.repo_id,trust_remote_code=spec.trust_remote_code);base.configure_processor(model,processor)
    layers,path=decoder_layers(model);selected,audit=choose_layers(a.layers,len(layers),a.failure_dir,a.min_role_gap);write_csv(out/"selected_problem_layers.csv",audit)
    print(f"[decoder] {path}; selected={selected}")
    train=[s for s in meta["sids"] if meta["split"].get(s)=="train" and meta["gt"].get(s) in REL2ID and s in records]
    if a.max_train and len(train)>a.max_train:
        rng=random.Random(a.seed);by=defaultdict(list)
        for s in train:by[meta["gt"][s]].append(s)
        for r in by:rng.shuffle(by[r])
        q=[]
        while len(q)<a.max_train and any(by.values()):
            for r in RELATIONS:
                if by[r] and len(q)<a.max_train:q.append(by[r].pop())
        train=sorted(q)
    cbs,diag=fit_actual_point_codebooks(model,processor,layers,selected,train,records,meta,torch.device(a.device),a.prompt_template,out)
    print("[codebooks]",[(r["layer"],r["stage"],round(r["train_direction_acc"],3)) for r in diag])
    ev=[]
    for s in meta["sids"]:
        if a.eval_split!="all" and meta["split"].get(s)!=a.eval_split:continue
        if s not in records or meta["gt"].get(s) not in REL2ID:continue
        g=meta["generation"].get(s,{});group=g.get("generation_group");pred=g.get("generation_pred")
        if group not in ("correct","wrong"):continue
        if group=="wrong" and (pred not in REL2ID or pred==meta["gt"][s]):continue
        ev.append(s)
    if a.max_eval and len(ev)>a.max_eval:
        rng=random.Random(a.seed+1);w=[s for s in ev if meta["generation"][s]["generation_group"]=="wrong"];c=[s for s in ev if s not in set(w)];rng.shuffle(w);rng.shuffle(c);ev=sorted(set(w[:a.max_eval//2]+c[:a.max_eval-a.max_eval//2]))
    token_map=relation_tokens(processor.tokenizer);rows=[];errors=[]
    for i,sid in enumerate(tqdm(ev,desc="pre/post causal scan"),1):
        try:rows+=scan_sample(model,processor,token_map,layers,selected,cbs,records[sid],sid,meta,torch.device(a.device),a.prompt_template,a.random_controls,a.seed)
        except Exception as e:errors.append({"sid":sid,"error_type":type(e).__name__,"error":str(e)});tqdm.write(f"[ERR {sid}] {e}")
        finally:model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if a.save_every and i%a.save_every==0:write_csv(out/"per_sample_stage_axis.csv",rows)
    write_csv(out/"per_sample_stage_axis.csv",rows);write_csv(out/"errors.csv",errors)
    summ=summarize(rows,selected,a.bootstrap,a.seed);trans=transition(summ);th,fmap,fsum=thresholds_and_failures(rows,a.control_quantile)
    write_csv(out/"primary_foil_stage_summary.csv",summ);write_csv(out/"attention_transition_summary.csv",trans);write_csv(out/"correct_stage_axis_thresholds.csv",th);write_csv(out/"wrong_stage_failure_map.csv",fmap);write_csv(out/"wrong_stage_failure_type_summary.csv",fsum)
    fd_layers=choose_fd(a.fd_layers,summ,len(layers),a.fd_top_k);print("[fd layers]",fd_layers);fdrows,fd=[] ,[]
    if fd_layers:
        fdrows,fd=finite_diff(model,processor,token_map,layers,cbs,fd_layers,records,meta,rows,torch.device(a.device),a.prompt_template,a.fd_max_samples,parse_floats(a.fd_eps_scales),a.fd_random_control,a.seed)
        write_csv(out/"finite_difference_per_sample.csv",fdrows);write_csv(out/"finite_difference_summary.csv",fd)
    print_results(summ,trans,fsum,fd)
    (out/"summary.json").write_text(json.dumps({"selected_layers":selected,"n_train":len(train),"n_eval":len(ev),"fd_layers":fd_layers,
        "R":"actual-point img-noimg Direction availability","C":"local derivative of first-step GT-vs-competitor margin at actual pre/post Attention point",
        "important":"accept C only when small-epsilon finite difference agrees"},indent=2),encoding="utf-8")

if __name__=="__main__":main()
