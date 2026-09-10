#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_qwen_spatial_heads_vs_causal_core_v1.py

Purpose
-------
Systematically scan MANY attention heads and ask:

    Which spatial heads consume the same token states that our writer-guided
    causal ranking identifies as the strongest causal core?

This directly links:

    causal state at decoder output L
        ->
    attention head at layer L+1

because block-output h_L is the input to the next attention block.

The script scans BOTH spatial-head definitions already used in this project:

1) Direction head
   Head output at subject/reference positions:
       r_h = z_h(subject) - z_h(reference)

   We use aligned Real-Gray:
       delta_r_h = r_h^real - r_h^gray

   A relation-specific direction code is learned on a calibration split.
   Held-out 4-way spatial accuracy is reported for EVERY scanned head.

   Token-source decomposition:
       delta_r_h
         = sum_p [
             (A_real(sub,p)-A_real(ref,p)) V_real_h(p)
             -
             (A_gray(sub,p)-A_gray(ref,p)) V_gray_h(p)
           ]

   Each source token gets:
       S_dir(h,p) = < contribution_h,p , d_h,GT >

   Therefore we can rank ALL source tokens by how much spatial-direction
   information they supply to that head.

2) Centroid head
   Subject/reference queries attend to visual tokens. Their attention centroids
   determine LEFT/RIGHT/ABOVE/BELOW.

   Held-out centroid accuracy is reported for EVERY scanned head.

   Each visual source token gets an exact leave-one-out contribution:
       S_cent(h,p)
         = GT_axis_score(full centroid)
           - GT_axis_score(centroid without p)

Then we compare those token rankings with an EXISTING causal ranking CSV:

    ranked_k36_tokens.csv

For each causal core K1 / K3 / K5 / K7 / ... and every scanned head, we report:

- causal states eligible for this head (same aligned input layer)
- mean spatial percentile of those causal states
    1.0 = strongest spatial source
    0.5 = random expectation
- fraction in the head's spatial Top10% / Top5%
- matched-budget micro recall
- enrichment over random
- spatial-head held-out accuracy

This answers:
    "Do the strongest causal tokens line up with specific spatial heads?"

Recommended exploratory scan
----------------------------
Scan heads in attention layers 21..27, whose inputs are causal source layers
20..26:

python scan_qwen_spatial_heads_vs_causal_core_v1.py \
  --model qwen-3b \
  --head-layers 21,22,23,24,25,26,27 \
  --ranked-causal outputs/oracle_rank_marginal_v1_r10/ranked_k36_tokens.csv \
  --cores 1,3,5,7,10,20,36 \
  --eval-scope test \
  --eval-max-samples 80 \
  --output-dir outputs/spatial_head_fullscan_vs_causal_core_v1 \
  --overwrite

Important interpretation
------------------------
A high-scoring pair means:
    the causal token state is a strong DIRECT SOURCE for that next-layer head's
    spatial computation.

It does NOT yet prove mediation to the final answer. After scanning, the strongest
candidate heads should be tested by intervention / blocking.

This is an exploratory head scan. Selecting the best head and claiming a final
effect on the SAME 80 samples would be selection-biased. Re-test selected heads
on a fresh split for paper-level evidence.
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
DISPLAY = {"left":"left", "right":"right", "above":"on", "below":"under"}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b","qwen-7b"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager",
        choices=["eager","sdpa","flash_attention_2","none"],
    )

    p.add_argument(
        "--head-layers",
        default="21,22,23,24,25,26,27",
        help=(
            "Attention layers to scan. With offset=1, these correspond to "
            "causal block-output source layers 20..26."
        ),
    )
    p.add_argument(
        "--head-input-offset", type=int, default=1,
        help="head layer = causal source layer + offset."
    )
    p.add_argument("--direction-pool", choices=["mean","last"], default="mean")

    p.add_argument("--ranked-causal", required=True,
                   help="ranked_k36_tokens.csv from the causal-rank experiment.")
    p.add_argument("--cores", default="1,3,5,7,10,20,36")

    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-scope", choices=["test","all_data"], default="test")
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument(
        "--min-spatial-acc", type=float, default=0.60,
        help=(
            "Threshold only for the compact 'strong heads' output. "
            "ALL heads are still scanned and saved."
        ),
    )
    p.add_argument("--top-heads", type=int, default=30,
                   help="How many heads to print in compact leaderboards.")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return sorted({
        int(x.strip().upper().replace("L",""))
        for x in str(s).split(",") if x.strip()
    })


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_median(xs):
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v/n).astype(np.float32)


def hname(L,h):
    return f"L{int(L)}H{int(h):02d}"


def get_text_config(model):
    cfg = getattr(model,"config",None)
    for c in (
        getattr(cfg,"text_config",None),
        getattr(cfg,"language_config",None),
        cfg,
    ):
        if c is not None and getattr(c,"num_attention_heads",None) is not None:
            return c
    raise RuntimeError("Could not resolve text config")


def resolve_attn(layer):
    for n in ("self_attn","attention","attn"):
        x = getattr(layer,n,None)
        if x is not None:
            return x
    raise RuntimeError("Could not resolve attention module")


def resolve_o_proj(attn):
    for n in ("o_proj","out_proj","proj"):
        x = getattr(attn,n,None)
        if isinstance(x,torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve o_proj")


def resolve_v_proj(attn):
    for n in ("v_proj","value","value_proj"):
        x = getattr(attn,n,None)
        if isinstance(x,torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve v_proj")


def extract_attentions(outputs):
    for x in (
        getattr(outputs,"attentions",None),
        getattr(getattr(outputs,"language_model_outputs",None),"attentions",None),
        getattr(getattr(outputs,"text_model_output",None),"attentions",None),
    ):
        if isinstance(x,(list,tuple)) and len(x):
            return tuple(x)
    raise RuntimeError(
        "No attentions returned. Use --attn-impl eager for this scan."
    )


def norm_attn(x):
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected attention shape {tuple(x.shape)}")
    return x.detach().float().cpu().numpy().astype(np.float32)


def make_gray(real,value):
    from PIL import Image
    v = max(0,min(255,int(value)))
    return Image.new("RGB",real.size,(v,v,v))


def span_positions(span):
    return list(range(int(span[0]),int(span[1])+1))


def object_positions(processor,ids,subject,reference):
    try:
        ss,rr = base.locate_object_spans(
            processor.tokenizer,ids,subject,reference
        )
        return span_positions(ss),span_positions(rr)
    except Exception:
        return (
            dyn.find_text_positions(processor.tokenizer,ids,subject),
            dyn.find_text_positions(processor.tokenizer,ids,reference),
        )


def pool_rows(A,positions,mode):
    valid = [int(p) for p in positions if 0 <= int(p) < A.shape[0]]
    if not valid:
        raise RuntimeError("No valid object token positions")
    if mode == "last":
        return A[valid[-1]]
    return A[valid].mean(axis=0)


def pool_states(H,positions,mode):
    valid = [int(p) for p in positions if 0 <= int(p) < H.shape[0]]
    if not valid:
        raise RuntimeError("No valid object token positions")
    if mode == "last":
        return H[valid[-1]]
    return H[valid].mean(axis=0)


class LayerCapture:
    """
    Capture pre-o_proj head outputs and v_proj outputs for ALL heads of selected
    layers in one forward. Attention matrices are taken from model outputs.
    """
    def __init__(self,model,decoder_layers,head_layers):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg,"num_key_value_heads",self.n_heads))
        self.hidden_size = int(getattr(cfg,"hidden_size",0) or 0)
        if self.hidden_size <= 0:
            op = resolve_o_proj(resolve_attn(decoder_layers[head_layers[0]]))
            self.hidden_size = int(op.in_features)
        self.head_dim = self.hidden_size // self.n_heads

        self.pre_o = {}
        self.v = {}
        self.handles = []

        for L in head_layers:
            attn = resolve_attn(decoder_layers[L])
            op = resolve_o_proj(attn)
            vp = resolve_v_proj(attn)

            def make_op_hook(L):
                def hook(_m,inputs):
                    self.pre_o[L] = (
                        inputs[0].detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            def make_v_hook(L):
                def hook(_m,_inp,out):
                    self.v[L] = (
                        out.detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            self.handles.append(op.register_forward_pre_hook(make_op_hook(L)))
            self.handles.append(vp.register_forward_hook(make_v_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def run_capture(model,decoder_layers,batch,head_layers,need_attn=True):
    cap = LayerCapture(model,decoder_layers,head_layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = bool(need_attn)
        out = model(**kw)
        att = {}
        if need_attn:
            aa = extract_attentions(out)
            for L in head_layers:
                att[L] = norm_attn(aa[L])
        return {
            "pre_o":cap.pre_o,
            "v":cap.v,
            "attn":att,
            "n_heads":cap.n_heads,
            "n_kv_heads":cap.n_kv_heads,
            "head_dim":cap.head_dim,
        }
    finally:
        cap.close()


def all_head_pre_o(pre_o,L,n_heads,head_dim):
    x = pre_o[L][0]  # [seq, hidden]
    return x.reshape(x.shape[0],n_heads,head_dim)


def all_head_v(vout,n_heads,n_kv_heads,head_dim):
    """
    v_proj output -> [seq, n_heads, head_dim], expanding grouped KV heads.
    """
    x = vout[0]
    if x.shape[-1] % head_dim != 0:
        raise RuntimeError(
            f"v_proj width={x.shape[-1]} incompatible with head_dim={head_dim}"
        )
    nkv = x.shape[-1] // head_dim
    x = x.reshape(x.shape[0],nkv,head_dim)
    if nkv == n_heads:
        return x
    if n_heads % nkv != 0:
        raise RuntimeError(f"Cannot expand KV heads: H={n_heads}, KVH={nkv}")
    repeat = n_heads // nkv
    return np.repeat(x,repeat,axis=1)


def fit_direction_codes(cal_store,n_heads,head_dim,head_layers):
    """
    cal_store[(L,h)][relation] -> list of head-delta vectors.
    """
    codes = {}
    for L in head_layers:
        for h in range(n_heads):
            means = {}
            allx = []
            ok = True
            for r in REL:
                xs = cal_store[(L,h)][r]
                if not xs:
                    ok = False
                    break
                arr = np.stack(xs).astype(np.float32)
                means[r] = arr.mean(axis=0).astype(np.float32)
                allx.extend(xs)
            if not ok:
                continue
            center = np.mean(np.stack(allx),axis=0).astype(np.float32)
            dirs = {
                r:normalize_np(means[r]-center) for r in REL
            }
            codes[(L,h)] = {"center":center,"dirs":dirs}
    return codes


def predict_dir(x,code):
    z = normalize_np(np.asarray(x,np.float32)-code["center"])
    sc = {r:float(np.dot(z,code["dirs"][r])) for r in REL}
    return max(REL,key=lambda r:sc[r]),sc


def direction_scores_all_heads(real_cap,gray_cap,L,spos,rpos,pool,dirs_gt):
    """
    Returns:
      scores [H,K] : per source-token spatial contribution
      head_delta [H,D] : captured Real-Gray subject-reference head residual
      recon_cos [H]
    """
    Ar = real_cap["attn"][L]   # [H,Q,K]
    Ag = gray_cap["attn"][L]
    H = real_cap["n_heads"]
    D = real_cap["head_dim"]

    Vr = all_head_v(
        real_cap["v"][L],H,real_cap["n_kv_heads"],D
    ) # [K,H,D]
    Vg = all_head_v(
        gray_cap["v"][L],H,gray_cap["n_kv_heads"],D
    )

    # pooled attention rows: [H,K]
    if pool == "last":
        qs,qr = int(spos[-1]),int(rpos[-1])
        cr = Ar[:,qs,:] - Ar[:,qr,:]
        cg = Ag[:,qs,:] - Ag[:,qr,:]
    else:
        ss = [int(x) for x in spos]
        rr = [int(x) for x in rpos]
        cr = Ar[:,ss,:].mean(axis=1) - Ar[:,rr,:].mean(axis=1)
        cg = Ag[:,ss,:].mean(axis=1) - Ag[:,rr,:].mean(axis=1)

    n = min(cr.shape[1],cg.shape[1],Vr.shape[0],Vg.shape[0])
    cr,cg = cr[:,:n],cg[:,:n]
    Vr,Vg = Vr[:n],Vg[:n]

    # [H,K,D]
    vec = (
        cr[:,:,None] * np.transpose(Vr,(1,0,2))
        -
        cg[:,:,None] * np.transpose(Vg,(1,0,2))
    ).astype(np.float32)

    # dirs_gt [H,D]
    scores = np.einsum("hkd,hd->hk",vec,dirs_gt).astype(np.float32)

    Hr = all_head_pre_o(real_cap["pre_o"],L,H,D) # [seq,H,D]
    Hg = all_head_pre_o(gray_cap["pre_o"],L,H,D)

    if pool == "last":
        hd = (
            Hr[int(spos[-1])] - Hr[int(rpos[-1])]
            - Hg[int(spos[-1])] + Hg[int(rpos[-1])]
        )
    else:
        hd = (
            Hr[spos].mean(axis=0) - Hr[rpos].mean(axis=0)
            - Hg[spos].mean(axis=0) + Hg[rpos].mean(axis=0)
        )
    recon = vec.sum(axis=1)

    recon_cos = np.full(H,np.nan,dtype=np.float32)
    for h in range(H):
        a,b = hd[h],recon[h]
        na,nb = np.linalg.norm(a),np.linalg.norm(b)
        if na > EPS and nb > EPS:
            recon_cos[h] = float(np.dot(a,b)/(na*nb))

    return scores,hd.astype(np.float32),recon_cos


def relation_from_centroids_local(dx,dy):
    # Match existing centroid logic: choose dominant normalized axis.
    # base.relation_from_centroids is used if available.
    try:
        return base.relation_from_centroids(dx,dy)
    except Exception:
        if abs(dx) >= abs(dy):
            return ("right" if dx > 0 else "left"), abs(dx)
        return ("below" if dy > 0 else "above"), abs(dy)


def centroid_all_heads(attn,visual_idx,coords,spos,rpos,gt):
    """
    For one layer:
      predictions per head
      per-head per-visual-token leave-one-out GT contribution

    Returns:
      pred list length H
      axis_conf list length H
      influence [H,V]
    """
    H = attn.shape[0]
    vi = np.asarray(visual_idx,dtype=np.int64)
    C = np.asarray(coords,dtype=np.float64)
    qs,qr = int(spos[-1]),int(rpos[-1])

    As = np.asarray(attn[:,qs,vi],dtype=np.float64)  # [H,V]
    Ar = np.asarray(attn[:,qr,vi],dtype=np.float64)

    As = As / np.maximum(As.sum(axis=1,keepdims=True),1e-12)
    Ar = Ar / np.maximum(Ar.sum(axis=1,keepdims=True),1e-12)

    cs = As @ C  # [H,2]
    cr = Ar @ C

    dx = cs[:,0]-cr[:,0]
    dy = cs[:,1]-cr[:,1]

    preds=[]
    confs=[]
    for h in range(H):
        pred,conf = relation_from_centroids_local(float(dx[h]),float(dy[h]))
        preds.append(pred)
        confs.append(float(conf))

    # full GT axis score [H]
    if gt=="left":
        full = -dx
    elif gt=="right":
        full = dx
    elif gt=="above":
        full = -dy
    else:
        full = dy

    # Exact leave-one-out for each head x visual token.
    # cs_minus[h,v,:] = (cs[h] - w[h,v]*coord[v]) / (1-w[h,v])
    den_s = np.maximum(1.0-As,1e-12)
    den_r = np.maximum(1.0-Ar,1e-12)

    cs_minus = (
        cs[:,None,:] - As[:,:,None]*C[None,:,:]
    ) / den_s[:,:,None]
    cr_minus = (
        cr[:,None,:] - Ar[:,:,None]*C[None,:,:]
    ) / den_r[:,:,None]

    mdx = cs_minus[:,:,0]-cr_minus[:,:,0]
    mdy = cs_minus[:,:,1]-cr_minus[:,:,1]

    if gt=="left":
        minus = -mdx
    elif gt=="right":
        minus = mdx
    elif gt=="above":
        minus = -mdy
    else:
        minus = mdy

    influence = (full[:,None]-minus).astype(np.float32)
    return preds,np.asarray(confs,np.float32),influence


def percentile_info(score_by_pos):
    items = sorted(
        score_by_pos.items(),
        key=lambda kv:(-float(kv[1]),int(kv[0]))
    )
    n=len(items)
    if n==0:
        return {},[],set(),set()

    ranks={}
    j=0
    while j<n:
        k=j+1
        while k<n and float(items[k][1])==float(items[j][1]):
            k+=1
        avg=0.5*((j+1)+k)
        for t in range(j,k):
            ranks[int(items[t][0])] = avg
        j=k

    pct={
        p:(1.0 if n==1 else float((n-r)/(n-1)))
        for p,r in ranks.items()
    }
    ordered=[int(p) for p,_ in items]
    n10=max(1,int(math.ceil(0.10*n)))
    n05=max(1,int(math.ceil(0.05*n)))
    return pct,ordered,set(ordered[:n10]),set(ordered[:n05])


def load_causal(path):
    raw=read_csv(path)
    by_sid=defaultdict(list)
    for r in raw:
        by_sid[int(float(r["sid"]))].append({
            "rank":int(float(r["rank"])),
            "source_layer":int(float(r["source_layer"])),
            "position":int(float(r["position"])),
            "token":r.get("token",""),
            "category":r.get("category",""),
            "broad_category":r.get("broad_category",""),
            "mediation":float(r["mediation"]),
        })
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x:x["rank"])
    return by_sid


def main():
    a=parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    head_layers=parse_ints(a.head_layers)
    cores=parse_ints(a.cores)
    causal_by_sid=load_causal(a.ranked_causal)
    causal_sids=set(causal_by_sid)

    outdir=Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)

    two=base.import_two_object_module()
    prompts=base.load_standard_prompts(Path(a.prompt_jsonl))
    records,_audit=two.load_records("coco_two",Path(a.data_root),None)
    rec_by_sid={int(r.sid):r for r in records}

    meta=[]
    for rec in records:
        sid=int(rec.sid)
        if sid not in prompts:
            continue
        p=prompts[sid]
        gt=traj.normalize_relation(base,p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid":sid,
            "gt":gt,
            "subject":str(p["subject"]),
            "reference":str(p["reference"]),
            "question_text":str(p["question_text"]),
        })

    meta=traj.stratified_cap(meta,a.max_samples,a.seed)
    train,heldout=traj.stratified_split(meta,a.train_ratio,a.seed)
    test=list(meta) if a.eval_scope=="all_data" else list(heldout)
    if int(a.eval_max_samples)>0:
        test=traj.stratified_cap(test,a.eval_max_samples,a.seed+1)

    # Only evaluate SIDs for which causal ranking exists.
    test=[m for m in test if int(m["sid"]) in causal_sids]
    if not test:
        raise RuntimeError("No overlap between eval split and ranked-causal CSV")

    specs=base.merged_model_specs(two)
    spec=specs[a.model]
    cls=getattr(transformers,spec.model_class)

    kw=dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"":a.device},
    )
    if a.attn_impl!="none":
        kw["attn_implementation"]=a.attn_impl

    model=processor=None
    try:
        print(f"Loading {spec.repo_id}",flush=True)
        model=cls.from_pretrained(spec.repo_id,**kw)
        model.eval()
        processor=AutoProcessor.from_pretrained(
            spec.repo_id,trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model,processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers,decoder_path=base.resolve_decoder_layers(model)
        cfg=get_text_config(model)
        n_heads=int(cfg.num_attention_heads)
        head_dim=int(getattr(cfg,"hidden_size"))//n_heads

        for L in head_layers:
            if not (0<=L<len(decoder_layers)):
                raise ValueError(f"Bad head layer L{L}")

        print("="*150)
        print("FULL SPATIAL-HEAD SCAN vs WRITER-CAUSAL CORE")
        print("="*150)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"heads/layer={n_heads} | head_dim={head_dim}")
        print(f"scan attention layers={head_layers}")
        print(
            "aligned causal source layers="
            f"{[L-a.head_input_offset for L in head_layers]}"
        )
        print(f"direction heads scanned={len(head_layers)*n_heads}")
        print(f"centroid heads scanned={len(head_layers)*n_heads}")
        print(f"direction calibration N={len(train)}")
        print(f"evaluation/common causal N={len(test)}")
        print(f"cores={cores}")
        print()

        # -------------------------------------------------------------
        # 1) CALIBRATE direction codes for EVERY head.
        # -------------------------------------------------------------
        cal_store=defaultdict(lambda:defaultdict(list))

        for m in tqdm(train,desc="CALIBRATE all direction heads"):
            sid=int(m["sid"])
            real=gray=rb=gb=None
            try:
                real=base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real=real.convert("RGB")
                gray=make_gray(real,a.gray_value)

                rb=base.make_question_batch(
                    processor=processor,image=real,
                    question_text=m["question_text"],device=torch.device(a.device)
                )
                gb=base.make_question_batch(
                    processor=processor,image=gray,
                    question_text=m["question_text"],device=torch.device(a.device)
                )
                ids=rb["input_ids"][0].detach().cpu().tolist()
                spos,rpos=object_positions(
                    processor,ids,m["subject"],m["reference"]
                )

                rc=run_capture(
                    model,decoder_layers,rb,head_layers,need_attn=False
                )
                gc_=run_capture(
                    model,decoder_layers,gb,head_layers,need_attn=False
                )

                H=rc["n_heads"]
                D=rc["head_dim"]

                for L in head_layers:
                    Hr=all_head_pre_o(rc["pre_o"],L,H,D)
                    Hg=all_head_pre_o(gc_["pre_o"],L,H,D)

                    if a.direction_pool=="last":
                        delta=(
                            Hr[int(spos[-1])] - Hr[int(rpos[-1])]
                            - Hg[int(spos[-1])] + Hg[int(rpos[-1])]
                        ) # [H,D]
                    else:
                        delta=(
                            Hr[spos].mean(axis=0)-Hr[rpos].mean(axis=0)
                            -Hg[spos].mean(axis=0)+Hg[rpos].mean(axis=0)
                        )

                    for h in range(H):
                        cal_store[(L,h)][m["gt"]].append(
                            delta[h].astype(np.float32)
                        )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()

        dir_codes=fit_direction_codes(
            cal_store,n_heads,head_dim,head_layers
        )

        # -------------------------------------------------------------
        # 2) EVALUATE all heads and compare causal states online.
        # -------------------------------------------------------------
        dir_correct=defaultdict(list)
        dir_recon=defaultdict(list)
        cent_correct=defaultdict(list)
        cent_conf=defaultdict(list)

        # agg[(family,L,h,K)] -> lists/counts
        agg=defaultdict(lambda:{
            "eligible":0,
            "overlap":0,
            "expected_overlap":0.0,
            "percentiles":[],
            "top10_hits":0,
            "top05_hits":0,
            "expected_top10":0.0,
            "expected_top05":0.0,
            "samples_with_eligible":0,
            "sample_recalls":[],
        })

        # direct rank details for strongest ranks
        rank_agg=defaultdict(lambda:{
            "N":0,
            "pct":[],
            "top10":0,
            "top05":0,
        })

        # Save only causal-state/head matches, not every token of every head.
        causal_detail=[]

        max_core=max(cores)

        for m in tqdm(test,desc="EVAL all heads vs causal core"):
            sid=int(m["sid"])
            gt=m["gt"]

            real=gray=rb=gb=None
            try:
                real=base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real=real.convert("RGB")
                gray=make_gray(real,a.gray_value)

                rb=base.make_question_batch(
                    processor=processor,image=real,
                    question_text=m["question_text"],device=torch.device(a.device)
                )
                gb=base.make_question_batch(
                    processor=processor,image=gray,
                    question_text=m["question_text"],device=torch.device(a.device)
                )

                ids=rb["input_ids"][0].detach().cpu().tolist()
                spos,rpos=object_positions(
                    processor,ids,m["subject"],m["reference"]
                )
                visual_idx=list(map(int,base.resolve_visual_indices(
                    model,processor,rb,ids
                )))
                coords_t=base.visual_coordinates(
                    model,rb,len(visual_idx),torch.device(a.device)
                )
                if coords_t is None:
                    raise RuntimeError("Could not resolve visual coordinates")
                coords=coords_t.detach().float().cpu().numpy().astype(np.float32)

                rc=run_capture(
                    model,decoder_layers,rb,head_layers,need_attn=True
                )
                gc_=run_capture(
                    model,decoder_layers,gb,head_layers,need_attn=True
                )

                causal=causal_by_sid[sid]

                for L in head_layers:
                    source_layer=L-int(a.head_input_offset)

                    # ---------------- DIRECTION ----------------
                    dirs_gt=np.stack([
                        dir_codes[(L,h)]["dirs"][gt]
                        for h in range(n_heads)
                    ]).astype(np.float32)

                    dscores,head_delta,recon=direction_scores_all_heads(
                        rc,gc_,L,spos,rpos,a.direction_pool,dirs_gt
                    ) # [H,K]

                    # spatial accuracy for all direction heads
                    for h in range(n_heads):
                        pred,_=predict_dir(
                            head_delta[h],dir_codes[(L,h)]
                        )
                        dir_correct[(L,h)].append(pred==gt)
                        dir_recon[(L,h)].append(float(recon[h]))

                    # compare each head with causal core
                    for h in range(n_heads):
                        npos=min(dscores.shape[1],len(ids)-1)
                        score_by_pos={
                            p:float(dscores[h,p])
                            for p in range(max(0,npos))
                        }
                        pct,ordered,top10,top05=percentile_info(score_by_pos)
                        N=len(ordered)
                        if N==0:
                            continue
                        top10_frac=len(top10)/N
                        top05_frac=len(top05)/N

                        # direct rank relation
                        for state in causal:
                            rnk=int(state["rank"])
                            if rnk>max_core:
                                break
                            if int(state["source_layer"])!=source_layer:
                                continue
                            pos=int(state["position"])
                            if pos not in pct:
                                continue
                            z=rank_agg[("direction",L,h,rnk)]
                            z["N"]+=1
                            z["pct"].append(float(pct[pos]))
                            z["top10"]+=int(pos in top10)
                            z["top05"]+=int(pos in top05)

                        for K in cores:
                            states=[
                                x for x in causal
                                if x["rank"]<=K
                                and x["source_layer"]==source_layer
                                and x["position"] in pct
                            ]
                            c=len(states)
                            if c==0:
                                continue

                            spatial_top=set(ordered[:min(c,N)])
                            hits=sum(x["position"] in spatial_top for x in states)

                            z=agg[("direction",L,h,K)]
                            z["eligible"]+=c
                            z["overlap"]+=hits
                            z["expected_overlap"]+=c*min(c,N)/N
                            z["samples_with_eligible"]+=1
                            z["sample_recalls"].append(hits/c)

                            for x in states:
                                pos=int(x["position"])
                                z["percentiles"].append(float(pct[pos]))
                                z["top10_hits"]+=int(pos in top10)
                                z["top05_hits"]+=int(pos in top05)
                                z["expected_top10"]+=top10_frac
                                z["expected_top05"]+=top05_frac

                                if K in (1,3,5,7) and x["rank"]<=K:
                                    causal_detail.append({
                                        "sid":sid,
                                        "gt":DISPLAY[gt],
                                        "family":"direction",
                                        "head_name":hname(L,h),
                                        "head_layer":L,
                                        "aligned_source_layer":source_layer,
                                        "core_k":K,
                                        "causal_rank":x["rank"],
                                        "causal_M":x["mediation"],
                                        "position":pos,
                                        "token":x["token"],
                                        "category":x["category"],
                                        "spatial_source_score":score_by_pos[pos],
                                        "spatial_percentile":pct[pos],
                                        "spatial_top10":pos in top10,
                                        "spatial_top05":pos in top05,
                                    })

                    # ---------------- CENTROID ----------------
                    preds,confs,cinfl=centroid_all_heads(
                        rc["attn"][L],visual_idx,coords,spos,rpos,gt
                    ) # [H,V]

                    for h in range(n_heads):
                        cent_correct[(L,h)].append(preds[h]==gt)
                        cent_conf[(L,h)].append(float(confs[h]))

                    p_to_i={int(p):j for j,p in enumerate(visual_idx)}

                    for h in range(n_heads):
                        score_by_pos={
                            int(p):float(cinfl[h,j])
                            for j,p in enumerate(visual_idx)
                            if int(p)<len(ids)-1
                        }
                        pct,ordered,top10,top05=percentile_info(score_by_pos)
                        N=len(ordered)
                        if N==0:
                            continue
                        top10_frac=len(top10)/N
                        top05_frac=len(top05)/N

                        for state in causal:
                            rnk=int(state["rank"])
                            if rnk>max_core:
                                break
                            if int(state["source_layer"])!=source_layer:
                                continue
                            pos=int(state["position"])
                            if pos not in pct:
                                continue
                            z=rank_agg[("centroid",L,h,rnk)]
                            z["N"]+=1
                            z["pct"].append(float(pct[pos]))
                            z["top10"]+=int(pos in top10)
                            z["top05"]+=int(pos in top05)

                        for K in cores:
                            states=[
                                x for x in causal
                                if x["rank"]<=K
                                and x["source_layer"]==source_layer
                                and x["position"] in pct
                            ]
                            c=len(states)
                            if c==0:
                                continue

                            spatial_top=set(ordered[:min(c,N)])
                            hits=sum(x["position"] in spatial_top for x in states)

                            z=agg[("centroid",L,h,K)]
                            z["eligible"]+=c
                            z["overlap"]+=hits
                            z["expected_overlap"]+=c*min(c,N)/N
                            z["samples_with_eligible"]+=1
                            z["sample_recalls"].append(hits/c)

                            for x in states:
                                pos=int(x["position"])
                                z["percentiles"].append(float(pct[pos]))
                                z["top10_hits"]+=int(pos in top10)
                                z["top05_hits"]+=int(pos in top05)
                                z["expected_top10"]+=top10_frac
                                z["expected_top05"]+=top05_frac

                                if K in (1,3,5,7) and x["rank"]<=K:
                                    causal_detail.append({
                                        "sid":sid,
                                        "gt":DISPLAY[gt],
                                        "family":"centroid",
                                        "head_name":hname(L,h),
                                        "head_layer":L,
                                        "aligned_source_layer":source_layer,
                                        "core_k":K,
                                        "causal_rank":x["rank"],
                                        "causal_M":x["mediation"],
                                        "position":pos,
                                        "token":x["token"],
                                        "category":x["category"],
                                        "spatial_source_score":score_by_pos[pos],
                                        "spatial_percentile":pct[pos],
                                        "spatial_top10":pos in top10,
                                        "spatial_top05":pos in top05,
                                    })

            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # 3) Spatial accuracy table for ALL heads.
        # -------------------------------------------------------------
        accuracy_rows=[]
        for L in head_layers:
            for h in range(n_heads):
                accuracy_rows.append({
                    "family":"direction",
                    "head_name":hname(L,h),
                    "head_layer":L,
                    "head":h,
                    "aligned_source_layer":L-a.head_input_offset,
                    "N":len(dir_correct[(L,h)]),
                    "spatial_accuracy":safe_mean(
                        float(x) for x in dir_correct[(L,h)]
                    ),
                    "mean_reconstruction_cosine":safe_mean(
                        dir_recon[(L,h)]
                    ),
                })
                accuracy_rows.append({
                    "family":"centroid",
                    "head_name":hname(L,h),
                    "head_layer":L,
                    "head":h,
                    "aligned_source_layer":L-a.head_input_offset,
                    "N":len(cent_correct[(L,h)]),
                    "spatial_accuracy":safe_mean(
                        float(x) for x in cent_correct[(L,h)]
                    ),
                    "mean_axis_confidence":safe_mean(
                        cent_conf[(L,h)]
                    ),
                })

        # -------------------------------------------------------------
        # 4) Head x causal core table.
        # -------------------------------------------------------------
        overlap_rows=[]
        for family in ("direction","centroid"):
            for L in head_layers:
                for h in range(n_heads):
                    acc=next(
                        r["spatial_accuracy"] for r in accuracy_rows
                        if r["family"]==family
                        and r["head_layer"]==L and r["head"]==h
                    )
                    for K in cores:
                        z=agg[(family,L,h,K)]
                        elig=z["eligible"]

                        micro=(
                            z["overlap"]/elig if elig else float("nan")
                        )
                        random_rec=(
                            z["expected_overlap"]/elig
                            if elig else float("nan")
                        )
                        enrich=(
                            z["overlap"]/z["expected_overlap"]
                            if z["expected_overlap"]>EPS else float("nan")
                        )
                        top10=(
                            z["top10_hits"]/elig if elig else float("nan")
                        )
                        top05=(
                            z["top05_hits"]/elig if elig else float("nan")
                        )

                        overlap_rows.append({
                            "family":family,
                            "head_name":hname(L,h),
                            "head_layer":L,
                            "head":h,
                            "aligned_source_layer":L-a.head_input_offset,
                            "core_k":K,
                            "spatial_accuracy":acc,
                            "eligible_causal_states":elig,
                            "samples_with_eligible":z["samples_with_eligible"],
                            "micro_recall":micro,
                            "expected_random_recall":random_rec,
                            "enrichment_over_random":enrich,
                            "mean_spatial_percentile":safe_mean(z["percentiles"]),
                            "median_spatial_percentile":safe_median(z["percentiles"]),
                            "spatial_top10_rate":top10,
                            "top10_enrichment":(
                                z["top10_hits"]/z["expected_top10"]
                                if z["expected_top10"]>EPS else float("nan")
                            ),
                            "spatial_top05_rate":top05,
                            "top05_enrichment":(
                                z["top05_hits"]/z["expected_top05"]
                                if z["expected_top05"]>EPS else float("nan")
                            ),
                            "mean_sample_recall":safe_mean(z["sample_recalls"]),
                        })

        # Direct causal rank table.
        rank_rows=[]
        for family in ("direction","centroid"):
            for L in head_layers:
                for h in range(n_heads):
                    acc=next(
                        r["spatial_accuracy"] for r in accuracy_rows
                        if r["family"]==family
                        and r["head_layer"]==L and r["head"]==h
                    )
                    for rank in range(1,max_core+1):
                        z=rank_agg[(family,L,h,rank)]
                        N=z["N"]
                        rank_rows.append({
                            "family":family,
                            "head_name":hname(L,h),
                            "head_layer":L,
                            "head":h,
                            "aligned_source_layer":L-a.head_input_offset,
                            "causal_rank":rank,
                            "spatial_accuracy":acc,
                            "eligible_N":N,
                            "mean_spatial_percentile":safe_mean(z["pct"]),
                            "spatial_top10_rate":(
                                z["top10"]/N if N else float("nan")
                            ),
                            "spatial_top05_rate":(
                                z["top05"]/N if N else float("nan")
                            ),
                        })

        # -------------------------------------------------------------
        # 5) Leaderboards.
        # -------------------------------------------------------------
        strong_rows=[
            r for r in overlap_rows
            if np.isfinite(float(r["spatial_accuracy"]))
            and float(r["spatial_accuracy"])>=a.min_spatial_acc
            and int(r["eligible_causal_states"])>0
        ]

        # A conservative joint score:
        # - head must itself be spatial
        # - causal tokens should rank high in its sources
        # We do NOT use enrichment because tiny expected counts explode.
        for r in strong_rows:
            r["joint_score"] = (
                float(r["spatial_accuracy"])
                *
                float(r["mean_spatial_percentile"])
                *
                float(r["spatial_top10_rate"])
            )

        leaderboard=[]
        for K in cores:
            rr=[r.copy() for r in strong_rows if int(r["core_k"])==K]
            rr.sort(
                key=lambda r:(
                    float(r["joint_score"]),
                    int(r["eligible_causal_states"])
                ),
                reverse=True,
            )
            for rank,r in enumerate(rr[:a.top_heads],1):
                r["leaderboard_rank"]=rank
                leaderboard.append(r)

        write_csv(outdir/"all_head_spatial_accuracy.csv",accuracy_rows)
        write_csv(outdir/"all_head_causal_core_overlap.csv",overlap_rows)
        write_csv(outdir/"all_head_causal_rank_spatialness.csv",rank_rows)
        write_csv(outdir/"strong_head_leaderboard.csv",leaderboard)
        write_csv(outdir/"causal_state_head_details.csv",causal_detail)

        metadata={
            "model":a.model,
            "repo_id":spec.repo_id,
            "scan_head_layers":head_layers,
            "n_heads_per_layer":n_heads,
            "direction_heads_scanned":len(head_layers)*n_heads,
            "centroid_heads_scanned":len(head_layers)*n_heads,
            "calibration_N":len(train),
            "evaluation_N":len(test),
            "causal_csv":str(Path(a.ranked_causal)),
            "cores":cores,
            "head_input_offset":a.head_input_offset,
            "min_spatial_acc_for_leaderboard":a.min_spatial_acc,
            "selection_bias_warning":(
                "This is an exploratory all-head scan. Any selected best heads "
                "should be re-tested on fresh samples."
            ),
        }
        (outdir/"metadata.json").write_text(
            json.dumps(metadata,indent=2),encoding="utf-8"
        )

        # -------------------------------------------------------------
        # 6) Console.
        # -------------------------------------------------------------
        print("\n"+"="*150)
        print("TOP SPATIAL HEADS BY HELD-OUT ACCURACY")
        print("="*150)
        for family in ("direction","centroid"):
            rr=[r for r in accuracy_rows if r["family"]==family]
            rr.sort(key=lambda r:float(r["spatial_accuracy"]),reverse=True)
            print(f"\n{family.upper()}:")
            for r in rr[:min(a.top_heads,len(rr))]:
                print(
                    f"  {r['head_name']:<8s} "
                    f"input<-L{int(r['aligned_source_layer']):02d} "
                    f"acc={float(r['spatial_accuracy']):.4f}"
                )

        for K in [k for k in cores if k in (1,3,5,7)]:
            rr=[
                r for r in leaderboard
                if int(r["core_k"])==K
            ]
            print("\n"+"="*150)
            print(f"TOP SPATIAL HEADS ALIGNED WITH CAUSAL K{K}")
            print(
                "joint = spatial_acc * mean_spatial_percentile * Top10_rate "
                "(exploratory ranking only)"
            )
            print("="*150)
            print(
                f"{'#':>3s} | {'family':<9s} | {'head':<8s} | {'src':>4s} | "
                f"{'acc':>6s} | {'elig':>5s} | {'spPct':>6s} | "
                f"{'top10':>6s} | {'micro':>6s} | {'joint':>7s}"
            )
            print("-"*105)
            for r in rr[:a.top_heads]:
                print(
                    f"{int(r['leaderboard_rank']):3d} | "
                    f"{r['family']:<9s} | "
                    f"{r['head_name']:<8s} | "
                    f"L{int(r['aligned_source_layer']):02d} | "
                    f"{float(r['spatial_accuracy']):6.3f} | "
                    f"{int(r['eligible_causal_states']):5d} | "
                    f"{float(r['mean_spatial_percentile']):6.3f} | "
                    f"{float(r['spatial_top10_rate']):6.3f} | "
                    f"{float(r['micro_recall']):6.3f} | "
                    f"{float(r['joint_score']):7.3f}"
                )

        print("\nHow to read:")
        print(
            "  - First require spatial_accuracy to be high: otherwise a head "
            "aligning with causal tokens is not evidence for a spatial mechanism."
        )
        print(
            "  - Then inspect spPct / Top10 for K1/K3. If near 1, the strongest "
            "causal tokens are also that head's strongest spatial sources."
        )
        print(
            "  - Compare layers: causal L -> spatial head L+1. This can reveal "
            "where the causal core enters a known spatial computation."
        )
        print(
            "  - Centroid is visual-only; Direction can use visual or text source tokens."
        )
        print(
            "  - Do not claim mediation from this scan alone. Re-test selected "
            "heads on fresh samples, then do causal blocking."
        )
        print("\nSaved:",outdir)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__=="__main__":
    main()
