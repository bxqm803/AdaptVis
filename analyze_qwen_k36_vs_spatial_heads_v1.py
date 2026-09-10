#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_qwen_k36_vs_spatial_heads_v1.py

Purpose
-------
Compare the current writer-guided K36 middle states with TWO already-established
spatial-head mechanisms in this repo:

1) Direction heads
   Existing repo method:
       analyze_coco_head_object_residual_direction_probe_v1.py

   A head is spatial when its subject-reference head-output residual predicts
   LEFT/RIGHT/ABOVE/BELOW well.

   Here we go one step deeper and decompose that head-level spatial relation
   back onto source token positions:

       z_h(sub) - z_h(ref)
         = sum_p [A_h(sub,p)-A_h(ref,p)] V_h(p)

   Using aligned Real-Gray runs, token p's image-conditioned contribution is:

       c_h,p =
         [A_real(sub,p)-A_real(ref,p)] V_real_h(p)
         -
         [A_gray(sub,p)-A_gray(ref,p)] V_gray_h(p)

   We learn the GT relation direction d_h,r on a disjoint calibration split and
   score each token:

       S_dir(h,p) = < c_h,p , d_h,GT >

   This answers:
       "Which source tokens actually supply spatial relation content to this
        known direction head?"

2) Centroid heads
   Existing repo method:
       analyze_coco_centroid_generation_step1_v4.py

   A head is spatial when subject/reference queries attend to visual tokens whose
   attention centroids reproduce the object relation.

   For each visual source token p, we compute exact leave-one-token-out influence
   on the GT centroid-axis score:

       S_centroid(h,p)
         = score_full(h,GT) - score_without_visual_token_p(h,GT)

   Positive means that visual token helps this head's centroid geometry express
   the correct relation.

Then, for every spatial head, we compare its Top-N source positions with the
K36 states at the decoder state layer that FEEDS that attention head.

Important layer convention
--------------------------
K36 edits are registered on decoder block OUTPUTS:
    h_after_block[L]

The next attention block consumes that state. Therefore by default:

    K36 source L22  -> attention head L23
    K36 source L23  -> attention head L24
    K36 source L25  -> attention head L26
    K36 source L26  -> attention head L27

This is controlled by:
    --head-input-offset 1

Default heads are the strong Qwen-3B heads already found in this project that
align with source layers L20-L26:

Direction:
    L23H01, L23H05, L26H03

Centroid:
    L24H05, L27H10

The older L19H13 / L20H05 and later L28H08 / L31H07 can also be passed, but
their immediate input layers fall outside the default K36 L20-L26 range.

Main outputs
------------
direction_head_accuracy.csv
    Held-out accuracy of the re-estimated Real-Gray direction-head code under
    the SAME standard prompt used for K36.

centroid_head_accuracy.csv
    Held-out centroid relation accuracy for configured centroid heads.

token_scores.csv
    Per sample / head / source token:
      writer M
      whether token is in K36
      direction-head token spatial contribution
      centroid-head token spatial contribution
      token category

head_overlap_summary.csv
    Per-head K36-vs-spatial-source overlap, enrichment, and rank correlation.

k36_coverage_summary.csv
    Fraction of K36 states on head-aligned layers captured by direction heads,
    centroid heads, or either family.

Interpretation
--------------
High overlap/enrichment:
    K36 states substantially coincide with the token sources used by known
    spatial heads.

Low overlap but spatial heads remain accurate:
    known explicit spatial-head sources and writer-causal states are largely
    different sets.

Direction and centroid families disagree:
    different spatial mechanisms consume different source tokens.

This script is ANALYSIS ONLY. It does not claim that attention weight alone is
causal; direction score uses A*V decomposition, centroid score uses exact
leave-one-out geometry. A later path-blocking experiment is needed for causal
mediation.
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
from typing import Any, Dict, List, Sequence, Tuple

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
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager","sdpa","flash_attention_2","none"])

    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--writer-target-layers", default="32,34,35")
    p.add_argument("--k36", type=int, default=36)

    p.add_argument(
        "--direction-heads",
        default="23:1,23:5,26:3",
        help='Comma-separated "layer:head", e.g. 23:1,23:5,26:3',
    )
    p.add_argument(
        "--centroid-heads",
        default="24:5,27:10",
        help='Comma-separated "layer:head".',
    )
    p.add_argument(
        "--head-input-offset",
        type=int,
        default=1,
        help=(
            "head layer = K36 source layer + offset. "
            "For decoder block-output K36 states, offset=1 is the intended mapping."
        ),
    )
    p.add_argument(
        "--direction-pool",
        choices=["mean","last"],
        default="mean",
        help="Pooling over multi-token subject/reference queries for direction heads.",
    )
    p.add_argument(
        "--spatial-top-mode",
        choices=["same_budget","fixed"],
        default="same_budget",
        help=(
            "same_budget: spatial-head Top-N uses number of K36 states available "
            "at that head's aligned source layer; fixed: use --spatial-top-k."
        ),
    )
    p.add_argument("--spatial-top-k", type=int, default=8)

    p.add_argument("--writer-mode", choices=["centered","raw"], default="centered")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--eval-scope", choices=["test","all_data"], default="test")
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({
        int(x.strip().upper().replace("L",""))
        for x in str(s).split(",") if x.strip()
    })


def parse_heads(s: str) -> List[Tuple[int,int]]:
    out = []
    seen = set()
    for raw in str(s).split(","):
        raw = raw.strip()
        if not raw:
            continue
        raw = raw.upper().replace("L","").replace("H",":")
        parts = [x for x in raw.split(":") if x != ""]
        if len(parts) != 2:
            raise ValueError(f"Bad head spec: {raw!r}")
        item = (int(parts[0]), int(parts[1]))
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def hname(L,h):
    return f"L{int(L)}H{int(h):02d}"


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_corr(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def rankdata(a):
    """Average ranks for ties; no scipy dependency."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def safe_spearman(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    return safe_corr(rankdata(x), rankdata(y))


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v/n).astype(np.float32)


def cosine_np(a,b):
    a,b = np.asarray(a,np.float32), np.asarray(b,np.float32)
    na,nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a,b)/(na*nb))


def make_gray(real, value):
    from PIL import Image
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real.size, (v,v,v))


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
    raise RuntimeError("Could not resolve self-attention module")


def resolve_o_proj(attn):
    for n in ("o_proj","out_proj","proj"):
        x = getattr(attn,n,None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve o_proj")


def resolve_v_proj(attn):
    for n in ("v_proj","value","value_proj"):
        x = getattr(attn,n,None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve v_proj")


def extract_attentions(outputs):
    candidates = [
        getattr(outputs,"attentions",None),
        getattr(getattr(outputs,"language_model_outputs",None),"attentions",None),
        getattr(getattr(outputs,"text_model_output",None),"attentions",None),
    ]
    for x in candidates:
        if isinstance(x,(list,tuple)) and len(x):
            return tuple(x)
    raise RuntimeError("No attentions returned; use --attn-impl eager")


def norm_attn(x):
    # -> [heads, q, k]
    if x is None:
        raise RuntimeError("Attention tensor is None")
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected attention shape {tuple(x.shape)}")
    return x.detach().float().cpu().numpy().astype(np.float32)


def span_positions(span):
    return list(range(int(span[0]), int(span[1])+1))


def object_positions(processor, ids, subject, reference):
    try:
        ss, rr = base.locate_object_spans(
            processor.tokenizer, ids, subject, reference
        )
        return span_positions(ss), span_positions(rr)
    except Exception:
        return (
            dyn.find_text_positions(processor.tokenizer, ids, subject),
            dyn.find_text_positions(processor.tokenizer, ids, reference),
        )


def pool_query_rows(A, positions, mode):
    valid = [int(p) for p in positions if 0 <= int(p) < A.shape[0]]
    if not valid:
        raise RuntimeError("No valid query positions")
    if mode == "last":
        return A[valid[-1]]
    return A[valid].mean(axis=0)


def pool_head_states(H, positions, mode):
    valid = [int(p) for p in positions if 0 <= int(p) < H.shape[0]]
    if not valid:
        raise RuntimeError("No valid object positions")
    if mode == "last":
        return H[valid[-1]]
    return H[valid].mean(axis=0)


class MultiCapture:
    """
    During one prompt forward:
      - captures selected decoder block outputs (optional)
      - captures pre-o_proj head outputs for configured direction heads
      - captures v_proj outputs for all configured spatial-head layers
      - returns model attentions via forward output
    """
    def __init__(self, model, decoder_layers, direction_heads, all_head_layers,
                 state_layers=()):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg,"num_key_value_heads",self.n_heads))
        self.hidden_size = int(getattr(cfg,"hidden_size",0) or 0)
        if self.hidden_size <= 0:
            op = resolve_o_proj(resolve_attn(decoder_layers[0]))
            self.hidden_size = int(op.in_features)
        self.head_dim = self.hidden_size // self.n_heads

        self.dir_by_layer = defaultdict(list)
        for L,h in direction_heads:
            self.dir_by_layer[int(L)].append(int(h))

        self.pre_o = {}
        self.v = {}
        self.states = {}
        self.handles = []

        for L in sorted(set(all_head_layers)):
            attn = resolve_attn(decoder_layers[L])
            op = resolve_o_proj(attn)
            vp = resolve_v_proj(attn)

            def make_op_hook(L):
                def hook(_m, inputs):
                    x = inputs[0]
                    self.pre_o[L] = x.detach().float().cpu().numpy().astype(np.float32)
                return hook

            def make_v_hook(L):
                def hook(_m, _inp, out):
                    self.v[L] = out.detach().float().cpu().numpy().astype(np.float32)
                return hook

            self.handles.append(op.register_forward_pre_hook(make_op_hook(L)))
            self.handles.append(vp.register_forward_hook(make_v_hook(L)))

        for L in sorted(set(state_layers)):
            def make_state_hook(L):
                def hook(_m,_inp,out):
                    x = traj.first_tensor(out)
                    self.states[L] = x.detach().float().cpu().numpy().astype(np.float32)
                    return out
                return hook
            self.handles.append(decoder_layers[L].register_forward_hook(make_state_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_capture(model, decoder_layers, batch, direction_heads, all_head_layers,
                state_layers=(), need_attn=True):
    cap = MultiCapture(
        model, decoder_layers, direction_heads, all_head_layers, state_layers
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = bool(need_attn)
        out = model(**kw)
        attentions = extract_attentions(out) if need_attn else None
        attn_sel = {}
        if need_attn:
            for L in all_head_layers:
                attn_sel[L] = norm_attn(attentions[L])
        return {
            "pre_o": cap.pre_o,
            "v": cap.v,
            "states": cap.states,
            "attn": attn_sel,
            "n_heads": cap.n_heads,
            "n_kv_heads": cap.n_kv_heads,
            "head_dim": cap.head_dim,
        }
    finally:
        cap.close()


def get_head_pre_o(pre_o, L, h, n_heads, head_dim):
    x = pre_o[L][0]  # [seq, hidden]
    y = x.reshape(x.shape[0], n_heads, head_dim)
    return y[:,h,:]


def get_v_for_attention_head(vout, h, n_heads, n_kv_heads, head_dim):
    x = vout[0]  # [seq, kv_width]
    if x.shape[-1] % head_dim != 0:
        raise RuntimeError(
            f"v_proj width {x.shape[-1]} incompatible with head_dim={head_dim}"
        )
    nkv = x.shape[-1] // head_dim
    x = x.reshape(x.shape[0], nkv, head_dim)

    # Standard grouped-query mapping: consecutive query heads share one KV head.
    if nkv == n_heads:
        kvh = h
    else:
        repeat = n_heads // nkv
        if repeat * nkv != n_heads:
            raise RuntimeError(f"Cannot map H={n_heads} to KVH={nkv}")
        kvh = h // repeat
    return x[:,kvh,:]


def fit_relation_dirs(X_by_rel):
    means = {}
    allx = []
    for r in REL:
        xs = X_by_rel[r]
        if not xs:
            raise RuntimeError(f"No calibration vectors for relation {r}")
        means[r] = np.mean(np.stack(xs),axis=0).astype(np.float32)
        allx.extend(xs)
    center = np.mean(np.stack(allx),axis=0).astype(np.float32)
    dirs = {r: normalize_np(means[r]-center) for r in REL}
    return center, dirs


def predict_relation(x, center, dirs):
    z = normalize_np(np.asarray(x,np.float32)-center)
    scores = {r: float(np.dot(z,dirs[r])) for r in REL}
    pred = max(REL, key=lambda r:scores[r])
    return pred, scores


def direction_token_contrib(real_cap, gray_cap, L, h, subpos, refpos, pool, gt_dir):
    """
    Exact A*V decomposition of the Real-Gray subject-reference head output.
    Returns one vector/scalar per key position.
    """
    Ar = real_cap["attn"][L][h]  # [q,k]
    Ag = gray_cap["attn"][L][h]
    Vr = get_v_for_attention_head(
        real_cap["v"][L], h,
        real_cap["n_heads"], real_cap["n_kv_heads"], real_cap["head_dim"]
    )
    Vg = get_v_for_attention_head(
        gray_cap["v"][L], h,
        gray_cap["n_heads"], gray_cap["n_kv_heads"], gray_cap["head_dim"]
    )

    ar_s = pool_query_rows(Ar, subpos, pool)
    ar_r = pool_query_rows(Ar, refpos, pool)
    ag_s = pool_query_rows(Ag, subpos, pool)
    ag_r = pool_query_rows(Ag, refpos, pool)

    n = min(len(ar_s), len(ag_s), Vr.shape[0], Vg.shape[0])
    coeff_r = (ar_s[:n] - ar_r[:n]).astype(np.float32)
    coeff_g = (ag_s[:n] - ag_r[:n]).astype(np.float32)

    vec = coeff_r[:,None]*Vr[:n] - coeff_g[:,None]*Vg[:n]
    score = vec @ normalize_np(gt_dir)
    norm = np.linalg.norm(vec,axis=1)

    # Validate against captured pre-o_proj head residual.
    Hr = get_head_pre_o(
        real_cap["pre_o"], L,h,real_cap["n_heads"],real_cap["head_dim"]
    )
    Hg = get_head_pre_o(
        gray_cap["pre_o"], L,h,gray_cap["n_heads"],gray_cap["head_dim"]
    )
    head_delta = (
        pool_head_states(Hr,subpos,pool)-pool_head_states(Hr,refpos,pool)
        -
        pool_head_states(Hg,subpos,pool)+pool_head_states(Hg,refpos,pool)
    )
    reconstructed = vec.sum(axis=0)
    recon_cos = cosine_np(head_delta,reconstructed)
    recon_relerr = float(
        np.linalg.norm(head_delta-reconstructed) /
        max(np.linalg.norm(head_delta), EPS)
    )
    return score.astype(np.float32), norm.astype(np.float32), recon_cos, recon_relerr, head_delta


def gt_axis_score(cent_sub, cent_ref, gt):
    dx = float(cent_sub[0]-cent_ref[0])
    dy = float(cent_sub[1]-cent_ref[1])
    if gt == "left":
        return -dx
    if gt == "right":
        return dx
    if gt == "above":
        return -dy
    if gt == "below":
        return dy
    raise ValueError(gt)


def centroid_token_influence(attn_head, visual_idx, coords, subpos, refpos, gt):
    """
    Existing centroid mechanism uses LAST object token as query.
    Return per visual-token leave-one-out contribution to GT axis score.
    """
    if not subpos or not refpos:
        raise RuntimeError("Missing object positions")
    qs, qr = int(subpos[-1]), int(refpos[-1])
    vi = np.asarray(visual_idx,dtype=np.int64)
    C = np.asarray(coords,dtype=np.float32)
    A = attn_head

    ws = np.asarray(A[qs,vi],np.float64)
    wr = np.asarray(A[qr,vi],np.float64)
    ws = ws / max(float(ws.sum()),EPS)
    wr = wr / max(float(wr.sum()),EPS)

    cs = (ws[:,None]*C).sum(axis=0)
    cr = (wr[:,None]*C).sum(axis=0)
    full = gt_axis_score(cs,cr,gt)

    # Exact leave-one-out centroids, vectorized.
    ds = np.maximum(1.0-ws, 1e-12)
    dr = np.maximum(1.0-wr, 1e-12)
    cs_minus = (cs[None,:] - ws[:,None]*C) / ds[:,None]
    cr_minus = (cr[None,:] - wr[:,None]*C) / dr[:,None]

    if gt == "left":
        scores_minus = -(cs_minus[:,0]-cr_minus[:,0])
    elif gt == "right":
        scores_minus = (cs_minus[:,0]-cr_minus[:,0])
    elif gt == "above":
        scores_minus = -(cs_minus[:,1]-cr_minus[:,1])
    else:
        scores_minus = (cs_minus[:,1]-cr_minus[:,1])

    influence = full - scores_minus

    dx = float(cs[0]-cr[0])
    dy = float(cs[1]-cr[1])
    pred, axis_conf = base.relation_from_centroids(dx,dy)

    # Also return normalized direct attention mass as a simpler descriptive quantity.
    mean_mass = 0.5*(ws+wr)
    return (
        influence.astype(np.float32),
        mean_mass.astype(np.float32),
        pred,
        float(axis_conf),
        float(full),
    )


def overlap_stats(k36_positions, spatial_ranked_positions, eligible_n, budget):
    k36 = set(map(int,k36_positions))
    spatial = set(map(int,spatial_ranked_positions[:budget]))
    inter = k36 & spatial
    union = k36 | spatial

    expected = (
        len(k36)*len(spatial)/float(eligible_n)
        if eligible_n > 0 else float("nan")
    )
    enrich = (
        len(inter)/expected
        if np.isfinite(expected) and expected > 0 else float("nan")
    )
    return {
        "k36_count":len(k36),
        "spatial_top_count":len(spatial),
        "intersection":len(inter),
        "recall_k36":len(inter)/len(k36) if k36 else float("nan"),
        "precision_spatial":len(inter)/len(spatial) if spatial else float("nan"),
        "jaccard":len(inter)/len(union) if union else float("nan"),
        "expected_random_overlap":expected,
        "enrichment_over_random":enrich,
    }, spatial


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    writer_targets = parse_ints(a.writer_target_layers)
    direction_heads = parse_heads(a.direction_heads)
    centroid_heads = parse_heads(a.centroid_heads)
    all_heads = direction_heads + centroid_heads
    head_layers = sorted(set(L for L,_ in all_heads))

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two",Path(a.data_root),None)
    rec_by_sid = {int(r.sid):r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base,p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid":sid,
            "gt":gt,
            "subject":str(p["subject"]),
            "reference":str(p["reference"]),
            "question_text":str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta,a.max_samples,a.seed)
    train, heldout = traj.stratified_split(meta,a.train_ratio,a.seed)
    test = list(meta) if a.eval_scope == "all_data" else list(heldout)
    test = traj.stratified_cap(test,a.eval_max_samples,a.seed+1)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers,spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"":a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None
    try:
        print(f"Loading {spec.repo_id}",flush=True)
        model = cls.from_pretrained(spec.repo_id,**kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model,processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)
        cfg = get_text_config(model)
        n_heads = int(cfg.num_attention_heads)

        for L,h in all_heads:
            if not (0 <= L < n_layers):
                raise ValueError(f"Bad layer {L}")
            if not (0 <= h < n_heads):
                raise ValueError(f"Bad head {h}; n_heads={n_heads}")

        mapped = []
        for family, heads in (("direction",direction_heads),("centroid",centroid_heads)):
            for L,h in heads:
                S = L - int(a.head_input_offset)
                mapped.append({
                    "family":family,
                    "head_layer":L,
                    "head":h,
                    "head_name":hname(L,h),
                    "aligned_k36_source_layer":S,
                    "source_in_k36_range":S in source_layers,
                })
        write_csv(outdir/"head_layer_mapping.csv",mapped)

        print("="*132)
        print("K36 vs KNOWN SPATIAL HEAD TOKEN SOURCES")
        print("="*132)
        print(f"model={a.model} decoder={decoder_path} layers={n_layers} heads={n_heads}")
        print(f"calibration N={len(train)} eval N={len(test)} scope={a.eval_scope}")
        print(f"K36 source layers={source_layers}, writer targets={writer_targets}, K={a.k36}")
        print(f"direction heads={[hname(*x) for x in direction_heads]}")
        print(f"centroid heads={[hname(*x) for x in centroid_heads]}")
        print(f"head input mapping: source L -> attention L+{a.head_input_offset}")
        print()

        # -------------------------------------------------------------
        # 1) Calibrate writer vectors AND direction-head relation codebooks.
        # -------------------------------------------------------------
        q_by_sid = {}
        dir_cal = {
            (L,h):{r:[] for r in REL}
            for L,h in direction_heads
        }

        for m in tqdm(train,desc="CALIBRATE writer + direction-head code"):
            sid = int(m["sid"])
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real = real.convert("RGB")
                gray = make_gray(real,a.gray_value)
                rb = base.make_question_batch(
                    processor=processor,image=real,
                    question_text=m["question_text"],device=torch.device(a.device)
                )
                gb = base.make_question_batch(
                    processor=processor,image=gray,
                    question_text=m["question_text"],device=torch.device(a.device)
                )
                ids = rb["input_ids"][0].detach().cpu().tolist()
                spos,rpos = object_positions(
                    processor,ids,m["subject"],m["reference"]
                )

                # Capture late residuals for writer.
                hr = dyn.capture_cpu(model,decoder_layers,rb,writer_targets)
                hg = dyn.capture_cpu(model,decoder_layers,gb,writer_targets)
                q_by_sid[sid] = {
                    T:(hr[T][0,-1]-hg[T][0,-1]).astype(np.float32)
                    for T in writer_targets
                }

                # Direction heads: same head-output residual principle as existing
                # repo probe, but Real-Gray so token positions remain aligned.
                rc = run_capture(
                    model,decoder_layers,rb,direction_heads,head_layers,
                    state_layers=(),need_attn=False
                )
                gc_ = run_capture(
                    model,decoder_layers,gb,direction_heads,head_layers,
                    state_layers=(),need_attn=False
                )
                for L,h in direction_heads:
                    Hr = get_head_pre_o(
                        rc["pre_o"],L,h,rc["n_heads"],rc["head_dim"]
                    )
                    Hg = get_head_pre_o(
                        gc_["pre_o"],L,h,gc_["n_heads"],gc_["head_dim"]
                    )
                    x = (
                        pool_head_states(Hr,spos,a.direction_pool)
                        - pool_head_states(Hr,rpos,a.direction_pool)
                        - pool_head_states(Hg,spos,a.direction_pool)
                        + pool_head_states(Hg,rpos,a.direction_pool)
                    ).astype(np.float32)
                    dir_cal[(L,h)][m["gt"]].append(x)

            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()

        writers, writer_geom = dyn.learn_writers(
            train,q_by_sid,writer_targets,a.writer_mode
        )
        write_csv(outdir/"writer_geometry.csv",writer_geom)

        dir_codes = {}
        for key,byrel in dir_cal.items():
            center,dirs = fit_relation_dirs(byrel)
            dir_codes[key] = {"center":center,"dirs":dirs}

        # -------------------------------------------------------------
        # 2) Eval:
        #    - K36
        #    - direction-head token A*V contribution
        #    - centroid-head visual leave-one-out contribution
        # -------------------------------------------------------------
        token_rows = []
        overlap_rows = []
        dir_acc_rows_raw = []
        cent_acc_rows_raw = []
        coverage_rows = []

        for m in tqdm(test,desc="EVAL K36 vs spatial heads"):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = rb = gb = cap = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real = real.convert("RGB")
                gray = make_gray(real,a.gray_value)
                rb = base.make_question_batch(
                    processor=processor,image=real,
                    question_text=m["question_text"],device=torch.device(a.device)
                )
                gb = base.make_question_batch(
                    processor=processor,image=gray,
                    question_text=m["question_text"],device=torch.device(a.device)
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                spos,rpos = object_positions(
                    processor,ids,m["subject"],m["reference"]
                )
                cats,toks = dyn.build_categories(
                    model,processor,rb,ids,m["subject"],m["reference"]
                )
                visual_idx = list(map(int,base.resolve_visual_indices(
                    model,processor,rb,ids
                )))
                coords_t = base.visual_coordinates(
                    model,rb,len(visual_idx),torch.device(a.device)
                )
                if coords_t is None:
                    raise RuntimeError("Could not build visual coordinates")
                coords = coords_t.detach().float().cpu().numpy().astype(np.float32)

                # Gray run also captures source states for K36 delta.
                gray_cap = run_capture(
                    model,decoder_layers,gb,direction_heads,head_layers,
                    state_layers=source_layers,need_attn=True
                )
                real_head_cap = run_capture(
                    model,decoder_layers,rb,direction_heads,head_layers,
                    state_layers=(),need_attn=True
                )

                # K36 writer-guided selection.
                graph_layers = sorted(set(source_layers+writer_targets))
                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model,decoder_layers,rb,graph_layers,min(source_layers)
                    )
                    terms = []
                    for T in writer_targets:
                        shat = torch.as_tensor(
                            normalize_np(writers[T][gt]),
                            device=cap.states[T].device,dtype=torch.float32
                        )
                        terms.append(torch.dot(cap.states[T][0,-1].float(),shat))
                    J = torch.stack(terms).sum()
                    grads = torch.autograd.grad(
                        J,[cap.states[S] for S in source_layers],
                        retain_graph=False,create_graph=False,allow_unused=False
                    )

                    rows_by_layer = {}
                    lookup_writer = {}
                    for S,g in zip(source_layers,grads):
                        Hr = cap.states[S][0].detach().float().cpu().numpy().astype(np.float32)
                        Hg = gray_cap["states"][S][0].astype(np.float32)
                        G = g[0].detach().float().cpu().numpy().astype(np.float32)
                        npos = min(len(ids),len(cats),len(toks),Hr.shape[0],Hg.shape[0],G.shape[0])
                        rr = []
                        for p in range(max(0,npos-1)):
                            delta = (Hr[p]-Hg[p]).astype(np.float32)
                            row = {
                                "position":p,
                                "token":str(toks[p]).replace("\n","\\n"),
                                "category":cats[p],
                                "broad_category":dyn.broad_category(cats[p]),
                                "mediation":float(np.dot(delta,G[p])),
                                "delta_h":delta,
                                "delta_h_norm":float(np.linalg.norm(delta)),
                                "grad_norm":float(np.linalg.norm(G[p])),
                            }
                            rr.append(row)
                            lookup_writer[(S,p)] = row
                        rr.sort(key=lambda x:x["mediation"],reverse=True)
                        rows_by_layer[S] = rr

                cap.close()
                cap = None

                _specs,k36_export = dyn.make_specs(
                    tuple(source_layers),rows_by_layer,"positive",a.k36,
                    sid,a.seed,strategy="global_unique"
                )
                k36_keys = {
                    (int(r["source_layer"]),int(r["position"]))
                    for r in k36_export
                }

                # Per sample maps of which K36 states were captured by at least one
                # configured head family.
                captured_dir = set()
                captured_cent = set()

                # ---------------------------------------------
                # Direction heads.
                # ---------------------------------------------
                for L,h in direction_heads:
                    S = L - int(a.head_input_offset)
                    if S not in source_layers:
                        continue

                    scores,norms,recon_cos,recon_relerr,head_delta = direction_token_contrib(
                        real_head_cap,gray_cap,L,h,spos,rpos,a.direction_pool,
                        dir_codes[(L,h)]["dirs"][gt]
                    )
                    pred, pred_scores = predict_relation(
                        head_delta,dir_codes[(L,h)]["center"],dir_codes[(L,h)]["dirs"]
                    )
                    dir_acc_rows_raw.append({
                        "sid":sid,"gt":DISPLAY[gt],"head_name":hname(L,h),
                        "head_layer":L,"head":h,"source_layer":S,
                        "prediction":DISPLAY[pred],
                        "correct":pred==gt,
                        "reconstruction_cosine":recon_cos,
                        "reconstruction_relative_error":recon_relerr,
                        "gt_direction_score":pred_scores[gt],
                    })

                    eligible_pos = list(range(min(len(scores),len(ids)-1)))
                    writer_vals = [
                        lookup_writer[(S,p)]["mediation"]
                        if (S,p) in lookup_writer else float("nan")
                        for p in eligible_pos
                    ]
                    spatial_vals = [float(scores[p]) for p in eligible_pos]

                    k36_at_S = sorted(
                        p for SS,p in k36_keys if SS == S and p in eligible_pos
                    )
                    budget = (
                        len(k36_at_S)
                        if a.spatial_top_mode=="same_budget"
                        else int(a.spatial_top_k)
                    )
                    ranked = sorted(
                        eligible_pos,key=lambda p:float(scores[p]),reverse=True
                    )
                    stats,spatial_top = overlap_stats(
                        k36_at_S,ranked,len(eligible_pos),budget
                    )
                    for p in k36_at_S:
                        if p in spatial_top:
                            captured_dir.add((S,p))

                    overlap_rows.append({
                        "sid":sid,"gt":DISPLAY[gt],"family":"direction",
                        "head_name":hname(L,h),"head_layer":L,"head":h,
                        "aligned_source_layer":S,"eligible_tokens":len(eligible_pos),
                        "pearson_writerM_vs_spatial":safe_corr(writer_vals,spatial_vals),
                        "spearman_writerM_vs_spatial":safe_spearman(writer_vals,spatial_vals),
                        "head_spatial_prediction":DISPLAY[pred],
                        "head_spatial_correct":pred==gt,
                        "reconstruction_cosine":recon_cos,
                        **stats,
                    })

                    for p in eligible_pos:
                        wr = lookup_writer.get((S,p),{})
                        token_rows.append({
                            "sid":sid,"gt":DISPLAY[gt],"family":"direction",
                            "head_name":hname(L,h),"head_layer":L,"head":h,
                            "aligned_source_layer":S,
                            "position":p,
                            "token":str(toks[p]).replace("\n","\\n"),
                            "category":cats[p],
                            "broad_category":dyn.broad_category(cats[p]),
                            "writer_m":wr.get("mediation",float("nan")),
                            "in_k36":(S,p) in k36_keys,
                            "spatial_source_score":float(scores[p]),
                            "spatial_source_norm":float(norms[p]),
                            "in_spatial_top":p in spatial_top,
                        })

                # ---------------------------------------------
                # Centroid heads.
                # ---------------------------------------------
                for L,h in centroid_heads:
                    S = L - int(a.head_input_offset)
                    if S not in source_layers:
                        continue

                    A = real_head_cap["attn"][L][h]
                    infl,mass,pred,axis_conf,full_score = centroid_token_influence(
                        A,visual_idx,coords,spos,rpos,gt
                    )
                    cent_acc_rows_raw.append({
                        "sid":sid,"gt":DISPLAY[gt],"head_name":hname(L,h),
                        "head_layer":L,"head":h,"source_layer":S,
                        "prediction":DISPLAY.get(pred,pred),
                        "correct":pred==gt,
                        "axis_confidence":axis_conf,
                        "gt_axis_score":full_score,
                    })

                    visual_pos = [
                        int(p) for p in visual_idx if int(p) < len(ids)-1
                    ]
                    p_to_i = {int(p):i for i,p in enumerate(visual_idx)}
                    eligible_pos = [p for p in visual_pos if p in p_to_i]

                    writer_vals = [
                        lookup_writer[(S,p)]["mediation"]
                        if (S,p) in lookup_writer else float("nan")
                        for p in eligible_pos
                    ]
                    spatial_vals = [
                        float(infl[p_to_i[p]]) for p in eligible_pos
                    ]

                    k36_at_S_visual = sorted(
                        p for SS,p in k36_keys
                        if SS == S and p in set(eligible_pos)
                    )
                    budget = (
                        len(k36_at_S_visual)
                        if a.spatial_top_mode=="same_budget"
                        else int(a.spatial_top_k)
                    )
                    ranked = sorted(
                        eligible_pos,
                        key=lambda p:float(infl[p_to_i[p]]),
                        reverse=True
                    )
                    stats,spatial_top = overlap_stats(
                        k36_at_S_visual,ranked,len(eligible_pos),budget
                    )
                    for p in k36_at_S_visual:
                        if p in spatial_top:
                            captured_cent.add((S,p))

                    k36_all_at_S = [p for SS,p in k36_keys if SS==S]
                    overlap_rows.append({
                        "sid":sid,"gt":DISPLAY[gt],"family":"centroid",
                        "head_name":hname(L,h),"head_layer":L,"head":h,
                        "aligned_source_layer":S,"eligible_tokens":len(eligible_pos),
                        "k36_all_count_at_source":len(k36_all_at_S),
                        "k36_visual_fraction_at_source":(
                            len(k36_at_S_visual)/len(k36_all_at_S)
                            if k36_all_at_S else float("nan")
                        ),
                        "pearson_writerM_vs_spatial":safe_corr(writer_vals,spatial_vals),
                        "spearman_writerM_vs_spatial":safe_spearman(writer_vals,spatial_vals),
                        "head_spatial_prediction":DISPLAY.get(pred,pred),
                        "head_spatial_correct":pred==gt,
                        "axis_confidence":axis_conf,
                        **stats,
                    })

                    for p in eligible_pos:
                        i = p_to_i[p]
                        wr = lookup_writer.get((S,p),{})
                        token_rows.append({
                            "sid":sid,"gt":DISPLAY[gt],"family":"centroid",
                            "head_name":hname(L,h),"head_layer":L,"head":h,
                            "aligned_source_layer":S,
                            "position":p,
                            "token":str(toks[p]).replace("\n","\\n"),
                            "category":cats[p],
                            "broad_category":dyn.broad_category(cats[p]),
                            "writer_m":wr.get("mediation",float("nan")),
                            "in_k36":(S,p) in k36_keys,
                            "spatial_source_score":float(infl[i]),
                            "centroid_attention_mass":float(mass[i]),
                            "in_spatial_top":p in spatial_top,
                        })

                # Coverage only over K36 states whose source layer has at least one
                # configured immediately-downstream spatial head.
                dir_source_layers = {
                    L-a.head_input_offset for L,_ in direction_heads
                    if (L-a.head_input_offset) in source_layers
                }
                cent_source_layers = {
                    L-a.head_input_offset for L,_ in centroid_heads
                    if (L-a.head_input_offset) in source_layers
                }
                aligned_keys = {
                    x for x in k36_keys
                    if x[0] in (dir_source_layers | cent_source_layers)
                }
                coverage_rows.append({
                    "sid":sid,"gt":DISPLAY[gt],
                    "k36_total":len(k36_keys),
                    "k36_on_head_aligned_source_layers":len(aligned_keys),
                    "direction_top_capture":len(aligned_keys & captured_dir),
                    "centroid_top_capture":len(aligned_keys & captured_cent),
                    "either_top_capture":len(aligned_keys & (captured_dir|captured_cent)),
                    "direction_coverage":(
                        len(aligned_keys & captured_dir)/len(aligned_keys)
                        if aligned_keys else float("nan")
                    ),
                    "centroid_coverage":(
                        len(aligned_keys & captured_cent)/len(aligned_keys)
                        if aligned_keys else float("nan")
                    ),
                    "either_coverage":(
                        len(aligned_keys & (captured_dir|captured_cent))/len(aligned_keys)
                        if aligned_keys else float("nan")
                    ),
                })

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # 3) Aggregate.
        # -------------------------------------------------------------
        dir_summary = []
        for L,h in direction_heads:
            rows = [r for r in dir_acc_rows_raw if r["head_layer"]==L and r["head"]==h]
            if not rows:
                continue
            dir_summary.append({
                "head_name":hname(L,h),"head_layer":L,"head":h,
                "aligned_source_layer":L-a.head_input_offset,
                "N":len(rows),
                "heldout_accuracy":safe_mean(float(r["correct"]) for r in rows),
                "mean_reconstruction_cosine":safe_mean(r["reconstruction_cosine"] for r in rows),
                "mean_reconstruction_relative_error":safe_mean(r["reconstruction_relative_error"] for r in rows),
            })

        cent_summary = []
        for L,h in centroid_heads:
            rows = [r for r in cent_acc_rows_raw if r["head_layer"]==L and r["head"]==h]
            if not rows:
                continue
            cent_summary.append({
                "head_name":hname(L,h),"head_layer":L,"head":h,
                "aligned_source_layer":L-a.head_input_offset,
                "N":len(rows),
                "heldout_accuracy":safe_mean(float(r["correct"]) for r in rows),
                "mean_axis_confidence":safe_mean(r["axis_confidence"] for r in rows),
            })

        head_overlap_summary = []
        keys = sorted(set(
            (r["family"],r["head_layer"],r["head"],r["head_name"],r["aligned_source_layer"])
            for r in overlap_rows
        ))
        for fam,L,h,name,S in keys:
            rr = [
                r for r in overlap_rows
                if r["family"]==fam and r["head_layer"]==L and r["head"]==h
            ]
            head_overlap_summary.append({
                "family":fam,"head_name":name,"head_layer":L,"head":h,
                "aligned_source_layer":S,"N":len(rr),
                "head_spatial_accuracy":safe_mean(float(r["head_spatial_correct"]) for r in rr),
                "mean_k36_count":safe_mean(r["k36_count"] for r in rr),
                "mean_spatial_top_count":safe_mean(r["spatial_top_count"] for r in rr),
                "mean_overlap":safe_mean(r["intersection"] for r in rr),
                "mean_k36_recall":safe_mean(r["recall_k36"] for r in rr),
                "mean_jaccard":safe_mean(r["jaccard"] for r in rr),
                "mean_enrichment_over_random":safe_mean(r["enrichment_over_random"] for r in rr),
                "mean_pearson_writerM_vs_spatial":safe_mean(r["pearson_writerM_vs_spatial"] for r in rr),
                "mean_spearman_writerM_vs_spatial":safe_mean(r["spearman_writerM_vs_spatial"] for r in rr),
            })

        coverage_summary = [{
            "N":len(coverage_rows),
            "mean_k36_total":safe_mean(r["k36_total"] for r in coverage_rows),
            "mean_k36_on_head_aligned_layers":safe_mean(
                r["k36_on_head_aligned_source_layers"] for r in coverage_rows
            ),
            "direction_coverage":safe_mean(r["direction_coverage"] for r in coverage_rows),
            "centroid_coverage":safe_mean(r["centroid_coverage"] for r in coverage_rows),
            "either_coverage":safe_mean(r["either_coverage"] for r in coverage_rows),
        }]

        # Token-category enrichment among overlapping K36/spatial-top tokens.
        cat_summary = []
        for fam in ("direction","centroid"):
            rr = [r for r in token_rows if r["family"]==fam and r["in_k36"]]
            for cat in ("visual","subject","reference","relation_words","other_text"):
                denom = [r for r in rr if r["broad_category"]==cat]
                if not denom:
                    continue
                cat_summary.append({
                    "family":fam,"broad_category":cat,
                    "k36_tokens_in_category":len(denom),
                    "fraction_also_spatial_top":safe_mean(
                        float(r["in_spatial_top"]) for r in denom
                    ),
                    "mean_spatial_source_score":safe_mean(
                        r["spatial_source_score"] for r in denom
                    ),
                    "mean_writer_m":safe_mean(r["writer_m"] for r in denom),
                })

        write_csv(outdir/"direction_head_accuracy.csv",dir_summary)
        write_csv(outdir/"direction_head_accuracy_per_sample.csv",dir_acc_rows_raw)
        write_csv(outdir/"centroid_head_accuracy.csv",cent_summary)
        write_csv(outdir/"centroid_head_accuracy_per_sample.csv",cent_acc_rows_raw)
        write_csv(outdir/"token_scores.csv",token_rows)
        write_csv(outdir/"head_overlap_per_sample.csv",overlap_rows)
        write_csv(outdir/"head_overlap_summary.csv",head_overlap_summary)
        write_csv(outdir/"k36_coverage_per_sample.csv",coverage_rows)
        write_csv(outdir/"k36_coverage_summary.csv",coverage_summary)
        write_csv(outdir/"k36_category_overlap.csv",cat_summary)

        meta_out = {
            "model":a.model,
            "repo_id":spec.repo_id,
            "calibration_N":len(train),
            "eval_N":len(test),
            "eval_scope":a.eval_scope,
            "source_layers":source_layers,
            "writer_targets":writer_targets,
            "k36":a.k36,
            "direction_heads":[hname(*x) for x in direction_heads],
            "centroid_heads":[hname(*x) for x in centroid_heads],
            "head_input_offset":a.head_input_offset,
            "direction_definition":(
                "Per-token exact A*V decomposition of Real-Gray "
                "subject-reference pre-o_proj head residual, projected onto "
                "calibrated relation direction."
            ),
            "centroid_definition":(
                "Per-visual-token exact leave-one-out change in GT centroid-axis score."
            ),
            "important_note":(
                "Direction calibration uses the existing head-output residual idea "
                "but Real-Gray rather than historical Image-NoImage, to preserve "
                "position alignment with K36. Held-out head accuracy is always reported."
            ),
        }
        (outdir/"metadata.json").write_text(
            json.dumps(meta_out,indent=2),encoding="utf-8"
        )

        print("\n"+"="*132)
        print("DIRECTION HEADS: HELD-OUT SPATIAL ACCURACY")
        print("="*132)
        for r in dir_summary:
            print(
                f"{r['head_name']:<8s} input<-L{r['aligned_source_layer']:02d} "
                f"acc={r['heldout_accuracy']:.4f} "
                f"recon_cos={r['mean_reconstruction_cosine']:.4f} "
                f"recon_relerr={r['mean_reconstruction_relative_error']:.4f}"
            )

        print("\n"+"="*132)
        print("CENTROID HEADS: HELD-OUT SPATIAL ACCURACY")
        print("="*132)
        for r in cent_summary:
            print(
                f"{r['head_name']:<8s} input<-L{r['aligned_source_layer']:02d} "
                f"acc={r['heldout_accuracy']:.4f} "
                f"axis_conf={r['mean_axis_confidence']:.4f}"
            )

        print("\n"+"="*132)
        print("K36 vs SPATIAL-HEAD SOURCE TOKENS")
        print("="*132)
        for r in head_overlap_summary:
            print(
                f"{r['family']:<9s} {r['head_name']:<8s} "
                f"K36@L{r['aligned_source_layer']:02d}={r['mean_k36_count']:.2f} | "
                f"recall={r['mean_k36_recall']:.3f} "
                f"jaccard={r['mean_jaccard']:.3f} "
                f"enrich={r['mean_enrichment_over_random']:.2f}x | "
                f"rho={r['mean_spearman_writerM_vs_spatial']:+.3f}"
            )

        c = coverage_summary[0]
        print("\nConfigured-head union coverage of K36 states on aligned source layers:")
        print(
            f"  direction={c['direction_coverage']:.3f} | "
            f"centroid={c['centroid_coverage']:.3f} | "
            f"either={c['either_coverage']:.3f}"
        )

        print("\nInterpretation:")
        print("  - First confirm each configured head still has strong held-out spatial accuracy.")
        print("  - enrichment >> 1 but recall modest: K36 contains a real spatial-head subset, but is broader.")
        print("  - recall/enrichment near random despite strong head accuracy: known spatial-head sources differ from K36.")
        print("  - direction/centroid disagreement means the two spatial mechanisms consume different token sources.")
        print("  - direction reconstruction cosine should be ~1; otherwise inspect attention/value conventions before interpreting overlap.")
        print("\nSaved:",outdir)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
