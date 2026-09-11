#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_three_spatial_head_token_selectors_all440_v1.py

Three spatial-head -> causal-token recovery strategies requested by the current
AdaptVis experiment.

Shared setup
============
Synthetic-400 is used ONLY to construct, for every (layer, head), four frozen
relation directions:

    d_h,left, d_h,right, d_h,above, d_h,below

plus the source center mu_h.

COCO-440's existing Direction-Head residual cache supplies the sample/head
representation

    q_h(x)

which is scored against the four frozen Synthetic directions:

    s_h,r(x) = cos(q_h(x) - mu_h, d_h,r)

IMPORTANT:
- Synthetic gives the FOUR DIRECTION VECTORS.
- COCO labels are allowed to choose the MACRO high-accuracy heads in method 1.
- Methods 2/3 choose heads sample-by-sample from COCO activations.
- Oracle Core50/Top10 causal states are used ONLY for recovery evaluation.

The three requested methods
===========================

METHOD 1: macro_coco_top10
--------------------------
1) Apply the frozen Synthetic relation directions to all COCO-440 q_h.
2) For every head, classify relation with argmax_r s_h,r.
3) Compute that head's ACC on COCO-440.
4) Select the globally highest-ACC Top10 heads from the requested middle layers.
5) For each sample, each of these fixed heads uses its OWN argmax relation r_h(x).
6) Token source score for that head:

       c_h,p = [A_h(sub,p)-A_h(ref,p)] V_h(p)
       token_score(h,p) = <c_h,p, d_h,r_h(x)>

7) Aggregate fixed-head token rankings and compare with oracle Top10/Core50.

This method asks:
    "Do the heads that are globally strongest on COCO spatial decoding point to
     the causal tokens?"

METHOD 2: micro_head_argmax
---------------------------
For EACH SAMPLE:
1) For every candidate middle-layer head:
       r_h(x) = argmax_r s_h,r(x)
       sensitivity_h(x) = max_r s_h,r(x)
2) Select the sample's Top-M heads by sensitivity_h(x).
3) Each selected head uses its OWN r_h(x) to score tokens:
       <c_h,p, d_h,r_h(x)>
4) Aggregate and compare with oracle Top10/Core50.

This asks:
    "For this input, which heads are maximally sensitive to ANY spatial
     direction, and which tokens feed those heads?"

METHOD 3: micro_vote_then_sensitive
-----------------------------------
For EACH SAMPLE:
1) Fixed Macro COCO-Top10 heads first vote for a single relation:
       r_vote(x)
   Majority vote over each head's argmax; ties use summed cosine support.
2) Now inspect ALL candidate middle-layer heads.
3) For each head:
       sensitivity_h(x) = s_h,r_vote(x)
4) Select Top-M heads most sensitive to this ONE voted direction.
5) Every selected head scores tokens only along d_h,r_vote(x):
       <c_h,p, d_h,r_vote(x)>
6) Aggregate and compare with oracle Top10/Core50.

This asks:
    "Once the head population chooses one direction, which heads are most
     responsive to that direction on this exact sample, and which tokens feed
     those heads?"

Token side
==========
The token contribution uses REAL IMAGE only:

    c_h,p = [A_real,h(sub,p)-A_real,h(ref,p)] V_real,h(p)

NO gray image is used for selecting tokens.
NO writer gradient is used for selecting tokens.
NO answer logit is used for selecting tokens.
NO GT relation is used inside methods 2/3 token selection.

Because score scales differ across heads, each head's token projection is
converted to a percentile over eligible TEXT tokens.  Selected heads at the
same input state layer are combined with method-specific head weights.

Layer alignment:
    attention head H reads decoder block-output state H-1.

The final candidate ranking mirrors the oracle global_unique convention:
if the same token position appears at multiple source layers, keep only the
source layer with the largest spatial-head consensus score.

Inputs
======
--source-cache
    Synthetic-400 NPZ:
      sample_index, relation, residual [N,L,H,D]

--coco-direction-cache
    COCO-440 NPZ:
      sample_index, relation, residual [N,L,H,D]

--oracle-bank-dir
    Output of build_oracle_core50_top10_candidate_bank_all440_v1.py:
      top10_candidates_all440.csv
      core50_candidates_all440.csv

Typical
=======
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_three_spatial_head_token_selectors_all440_v1.py \
  --model qwen-3b \
  --source-cache \
    output/qwen3b_direction_selector_syn400_to_coco440/synthetic_direction_head_relation_vectors.npz \
  --coco-direction-cache \
    output/qwen3b_head_object_residual_direction/relation_vectors.npz \
  --oracle-bank-dir \
    output/qwen3b_oracle_core50_top10_bank_all440 \
  --head-layers 21,22,23,24,25,26,27 \
  --macro-top-n 10 \
  --micro-head-budgets 3,5,10,20 \
  --select-ks 3,5,7,10 \
  --require-n 440 \
  --output-dir output/qwen3b_three_spatial_head_token_selectors_all440 \
  --overwrite

Notes
=====
This is a mechanistic discovery experiment.  Method 1 deliberately uses
COCO-440 GT labels to identify the globally most accurate heads, per the user's
requested macro analysis.  Therefore Macro COCO-Top10 itself is NOT a
label-free deployment selector.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--source-cache",
        required=True,
        help="Synthetic-400 relation-vector residual NPZ.",
    )
    p.add_argument(
        "--coco-direction-cache",
        required=True,
        help="COCO-440 relation_vectors.npz.",
    )
    p.add_argument(
        "--oracle-bank-dir",
        required=True,
        help="Top10/Core50 causal candidate bank directory.",
    )

    p.add_argument(
        "--head-layers",
        default="21,22,23,24,25,26,27",
        help="Middle attention-head layers H; candidate state layer is H-1.",
    )
    p.add_argument(
        "--head-input-offset",
        type=int,
        default=1,
        help="source state layer = attention head layer - offset.",
    )
    p.add_argument(
        "--macro-top-n",
        type=int,
        default=10,
        help="Number of globally highest COCO-accuracy heads for method 1 and voting.",
    )
    p.add_argument(
        "--micro-head-budgets",
        default="3,5,10,20",
        help="Sample-specific head counts for methods 2 and 3.",
    )
    p.add_argument(
        "--select-ks",
        default="3,5,7,10",
        help="Final global_unique exact-state candidate counts.",
    )
    p.add_argument(
        "--direction-pool",
        choices=["mean", "last"],
        default="mean",
    )
    p.add_argument(
        "--sensitivity-score",
        choices=["topcos", "margin"],
        default="topcos",
        help=(
            "Method 2 head sensitivity: topcos=max cosine; "
            "margin=top1 cosine - top2 cosine."
        ),
    )
    p.add_argument(
        "--head-weight-mode",
        choices=["equal", "sensitivity"],
        default="sensitivity",
        help=(
            "Within-state-layer head aggregation. Macro sensitivity weight uses "
            "COCO head accuracy; micro methods use sample-specific cosine sensitivity."
        ),
    )
    p.add_argument(
        "--positive-token-only",
        action="store_true",
        help=(
            "If set, negative direction-projection tokens receive score 0 before "
            "percentile ranking. Default ranks signed projections directly."
        ),
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--require-n",
        type=int,
        default=440,
        help="Require this many overlapping COCO/bank SIDs; 0 disables.",
    )

    p.add_argument(
        "--cache-token-contrib",
        action="store_true",
        help=(
            "Cache per-sample REAL-image directional token projections. "
            "This is larger on disk but allows post-processing without another forward."
        ),
    )
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-dir", required=True)

    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "bottom": "below",
    }.get(s, s)


def hname(L, h):
    return f"L{int(L)}H{int(h):02d}"


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def rankdata_average(a):
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


def percentiles(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) <= 1:
        return np.ones(len(a), dtype=np.float32)
    r = rankdata_average(a)
    return ((r - 1.0) / float(len(a) - 1)).astype(np.float32)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


# =============================================================================
# Direction caches
# =============================================================================

def resolve_npz(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    q = p / "relation_vectors.npz"
    if q.exists():
        return q
    raise FileNotFoundError(path_or_dir)


def load_direction_cache(path_or_dir):
    p = resolve_npz(path_or_dir)
    z = np.load(p, allow_pickle=True)

    required = {"relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{p} missing arrays: {sorted(missing)}")

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray([canon_rel(v) for v in z["relation"]], dtype=object)

    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    elif "sid" in z.files:
        sid = np.asarray(z["sid"]).astype(int)
    else:
        sid = np.arange(len(y), dtype=int)

    if X.ndim != 4:
        raise RuntimeError(f"{p}: expected [N,L,H,D], got {X.shape}")

    valid = np.isin(y, np.asarray(REL, dtype=object))
    return p, sid[valid], y[valid], X[valid]


def fit_synthetic_directions(X, y):
    """
    Same relation-direction idea used by the selector:
      mu_h = E[q_h]
      d_h,r = normalize(E[q_h | r] - mu_h)
    """
    center = X.mean(axis=0).astype(np.float32)
    dirs = np.zeros(
        (X.shape[1], X.shape[2], len(REL), X.shape[3]),
        dtype=np.float32,
    )
    for ri, r in enumerate(REL):
        m = y == r
        if not np.any(m):
            raise RuntimeError(f"Synthetic source missing relation={r}")
        dirs[:, :, ri, :] = normalize(
            X[m].mean(axis=0) - center,
            axis=-1,
        )
    return center, dirs


def score_against_synthetic_dirs(X, center, dirs):
    """
    s[i,L,H,r] = cosine(q_i,L,H - source_center_L,H, d_L,H,r)
    """
    q = normalize(X - center[None, :, :, :], axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", q, dirs, optimize=True)


def compute_coco_head_accuracy(coco_scores, coco_y, head_layers):
    yi = np.asarray([RID[r] for r in coco_y], dtype=int)
    pred = np.argmax(coco_scores, axis=-1)

    rows = []
    for L in head_layers:
        for h in range(coco_scores.shape[2]):
            correct = pred[:, L, h] == yi
            row = {
                "head_layer": int(L),
                "head": int(h),
                "head_name": hname(L, h),
                "coco_accuracy": float(correct.mean()),
                "N": int(len(correct)),
            }
            for ri, r in enumerate(REL):
                m = yi == ri
                row[f"acc_{r}"] = (
                    float(correct[m].mean()) if np.any(m) else np.nan
                )
                row[f"N_{r}"] = int(m.sum())
            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["coco_accuracy", "head_layer", "head"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


# =============================================================================
# Oracle bank
# =============================================================================

def load_oracle_bank(bank_dir):
    root = Path(bank_dir)
    top_path = root / "top10_candidates_all440.csv"
    core_path = root / "core50_candidates_all440.csv"

    if not top_path.exists():
        raise FileNotFoundError(top_path)
    if not core_path.exists():
        raise FileNotFoundError(core_path)

    def load_one(path):
        df = pd.read_csv(path)
        req = {"sid", "source_layer", "position", "category", "broad_category"}
        miss = req - set(df.columns)
        if miss:
            raise RuntimeError(f"{path} missing: {sorted(miss)}")

        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
        df["source_layer"] = pd.to_numeric(
            df["source_layer"], errors="raise"
        ).astype(int)
        df["position"] = pd.to_numeric(
            df["position"], errors="raise"
        ).astype(int)

        if "is_text" in df.columns:
            df["is_text"] = (
                df["is_text"].astype(str).str.lower().isin(
                    {"true", "1", "yes", "t"}
                )
            )
        else:
            df["is_text"] = (
                ~df["category"].astype(str).eq("visual")
                & ~df["broad_category"].astype(str).eq("visual")
            )
        return df

    return load_one(top_path), load_one(core_path)


def target_lookup(df):
    out = {}
    for sid, g in df.groupby("sid"):
        q = g[g["is_text"]]
        states = {
            (int(r.source_layer), int(r.position))
            for r in q.itertuples()
        }
        out[int(sid)] = {
            "states": states,
            "positions": {p for _, p in states},
        }
    return out


# =============================================================================
# Model capture
# =============================================================================

def get_text_config(model):
    cfg = getattr(model, "config", None)
    for c in (
        getattr(cfg, "text_config", None),
        getattr(cfg, "language_config", None),
        cfg,
    ):
        if c is not None and getattr(c, "num_attention_heads", None) is not None:
            return c
    raise RuntimeError("Could not resolve text config")


def resolve_attn(layer):
    for n in ("self_attn", "attention", "attn"):
        x = getattr(layer, n, None)
        if x is not None:
            return x
    raise RuntimeError("Could not resolve self-attention module")


def resolve_o_proj(attn):
    for n in ("o_proj", "out_proj", "proj"):
        x = getattr(attn, n, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve o_proj")


def resolve_v_proj(attn):
    for n in ("v_proj", "value", "value_proj"):
        x = getattr(attn, n, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve v_proj")


def extract_attentions(outputs):
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "attentions",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "attentions",
            None,
        ),
    ]
    for x in candidates:
        if isinstance(x, (list, tuple)) and len(x):
            return tuple(x)
    raise RuntimeError(
        "No attentions returned. Use --attn-impl eager for this experiment."
    )


def norm_attn(x):
    if x is None:
        raise RuntimeError("Attention tensor is None")
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected attention shape {tuple(x.shape)}")
    return x.detach().float().cpu().numpy().astype(np.float32)


class Capture:
    def __init__(self, model, decoder_layers, head_layers):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(
            getattr(cfg, "num_key_value_heads", self.n_heads)
        )
        self.hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        if self.hidden_size <= 0:
            self.hidden_size = int(
                resolve_o_proj(resolve_attn(decoder_layers[0])).in_features
            )
        self.head_dim = self.hidden_size // self.n_heads

        self.pre_o = {}
        self.v = {}
        self.handles = []

        for L in head_layers:
            attn = resolve_attn(decoder_layers[L])
            op = resolve_o_proj(attn)
            vp = resolve_v_proj(attn)

            def make_op_hook(layer_idx):
                def hook(_m, inputs):
                    x = inputs[0]
                    self.pre_o[layer_idx] = (
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            def make_v_hook(layer_idx):
                def hook(_m, _inp, out):
                    self.v[layer_idx] = (
                        out.detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_op_hook(L))
            )
            self.handles.append(
                vp.register_forward_hook(make_v_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_capture(model, decoder_layers, batch, head_layers):
    cap = Capture(model, decoder_layers, head_layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = True
        out = model(**kw)
        all_attn = extract_attentions(out)
        attn = {L: norm_attn(all_attn[L]) for L in head_layers}
        return {
            "attn": attn,
            "v": cap.v,
            "pre_o": cap.pre_o,
            "n_heads": cap.n_heads,
            "n_kv_heads": cap.n_kv_heads,
            "head_dim": cap.head_dim,
        }
    finally:
        cap.close()


def get_v_for_attention_head(vout, h, n_heads, n_kv_heads, head_dim):
    x = vout[0]
    if x.shape[-1] % head_dim != 0:
        raise RuntimeError(
            f"v_proj width {x.shape[-1]} incompatible with head_dim={head_dim}"
        )
    nkv = x.shape[-1] // head_dim
    x = x.reshape(x.shape[0], nkv, head_dim)

    if nkv == n_heads:
        kvh = h
    else:
        repeat = n_heads // nkv
        if repeat * nkv != n_heads:
            raise RuntimeError(
                f"Cannot map query heads={n_heads} to kv heads={nkv}"
            )
        kvh = h // repeat

    return x[:, kvh, :]


def span_positions(span):
    return list(range(int(span[0]), int(span[1]) + 1))


def object_positions(processor, ids, subject, reference):
    try:
        ss, rr = base.locate_object_spans(
            processor.tokenizer,
            ids,
            subject,
            reference,
        )
        return span_positions(ss), span_positions(rr)
    except Exception:
        return (
            dyn.find_text_positions(
                processor.tokenizer, ids, subject
            ),
            dyn.find_text_positions(
                processor.tokenizer, ids, reference
            ),
        )


def pool_rows(A, positions, mode):
    valid = [int(p) for p in positions if 0 <= int(p) < A.shape[0]]
    if not valid:
        raise RuntimeError("No valid object query positions")
    if mode == "last":
        return A[valid[-1]]
    return A[valid].mean(axis=0)


def real_token_projection_all_relations(
    cap,
    head_layer,
    head,
    subpos,
    refpos,
    direction_pool,
    dirs_4xd,
    text_positions,
):
    """
    REAL-image source decomposition:
        c_p = [A(sub,p)-A(ref,p)] V(p)

    Return token projection onto each Synthetic direction:
        proj[p,r] = <c_p, d_r>

    Output shape [P_text, 4].
    """
    A = cap["attn"][head_layer][head]
    V = get_v_for_attention_head(
        cap["v"][head_layer],
        head,
        cap["n_heads"],
        cap["n_kv_heads"],
        cap["head_dim"],
    )

    a_sub = pool_rows(A, subpos, direction_pool)
    a_ref = pool_rows(A, refpos, direction_pool)

    n = min(len(a_sub), len(a_ref), V.shape[0])
    coeff = (a_sub[:n] - a_ref[:n]).astype(np.float32)
    c = coeff[:, None] * V[:n]

    valid = [p for p in text_positions if 0 <= int(p) < n]
    if valid != text_positions:
        raise RuntimeError(
            "Some text positions fall outside captured attention/value length."
        )

    C = c[np.asarray(valid, dtype=np.int64)]
    D = np.asarray(dirs_4xd, dtype=np.float32)
    proj = C @ D.T

    return proj.astype(np.float32)


# =============================================================================
# Three head-selection strategies
# =============================================================================

def head_argmax_rel(score4):
    score4 = np.asarray(score4, dtype=np.float64)
    rid = int(np.argmax(score4))
    return REL[rid], rid


def topcos_and_margin(score4):
    s = np.sort(np.asarray(score4, dtype=np.float64))
    top = float(s[-1])
    margin = float(s[-1] - s[-2]) if len(s) >= 2 else float("nan")
    return top, margin


def majority_vote_relation(heads, sample_scores):
    """
    Literal head vote:
      each head votes argmax relation;
      ties -> largest summed cosine support among tied relations.
    """
    votes = Counter()
    support = defaultdict(float)

    for H, h in heads:
        s4 = sample_scores[H, h]
        rid = int(np.argmax(s4))
        votes[rid] += 1
        for r in range(4):
            support[r] += float(s4[r])

    if not votes:
        raise RuntimeError("No heads available for vote")

    max_vote = max(votes.values())
    tied = [r for r, n in votes.items() if n == max_vote]
    if len(tied) == 1:
        rid = tied[0]
    else:
        rid = max(tied, key=lambda r: support[r])

    return REL[rid], rid, {
        REL[r]: int(votes.get(r, 0))
        for r in range(4)
    }


def candidate_head_pool(head_layers, n_heads):
    return [(H, h) for H in head_layers for h in range(n_heads)]


def method_macro_heads(macro_heads, sample_scores, coco_acc_lookup):
    """
    Fixed Macro COCO-TopN heads.
    Each head uses its own sample-specific argmax direction.
    """
    rows = []
    for H, h in macro_heads:
        s4 = sample_scores[H, h]
        rel, rid = head_argmax_rel(s4)
        topcos, margin = topcos_and_margin(s4)

        rows.append({
            "head_layer": H,
            "head": h,
            "head_name": hname(H, h),
            "relation": rel,
            "relation_id": rid,
            "sensitivity": topcos,
            "margin": margin,
            "head_weight": max(
                float(coco_acc_lookup[(H, h)]) - 0.25,
                EPS,
            ),
            "selection_reason": "fixed_macro_coco_top_accuracy",
        })
    return rows


def method_micro_argmax_heads(
    all_heads,
    sample_scores,
    budget,
    sensitivity_mode,
):
    """
    Every head chooses its own best relation; select sample-specific heads by
    their maximum relation sensitivity (or top1-top2 margin).
    """
    rows = []
    for H, h in all_heads:
        s4 = sample_scores[H, h]
        rel, rid = head_argmax_rel(s4)
        topcos, margin = topcos_and_margin(s4)
        sensitivity = topcos if sensitivity_mode == "topcos" else margin

        rows.append({
            "head_layer": H,
            "head": h,
            "head_name": hname(H, h),
            "relation": rel,
            "relation_id": rid,
            "sensitivity": float(sensitivity),
            "topcos": topcos,
            "margin": margin,
            "head_weight": max(float(sensitivity), EPS),
            "selection_reason": f"sample_top_{sensitivity_mode}",
        })

    rows.sort(
        key=lambda r: (
            float(r["sensitivity"]),
            -int(r["head_layer"]),
            -int(r["head"]),
        ),
        reverse=True,
    )
    return rows[: int(budget)]


def method_micro_vote_sensitive_heads(
    all_heads,
    sample_scores,
    vote_heads,
    budget,
):
    """
    Macro heads first vote one relation; then ALL heads are ranked only by
    similarity to that relation.
    """
    vote_rel, vote_rid, vote_counts = majority_vote_relation(
        vote_heads,
        sample_scores,
    )

    rows = []
    for H, h in all_heads:
        s = float(sample_scores[H, h, vote_rid])
        rows.append({
            "head_layer": H,
            "head": h,
            "head_name": hname(H, h),
            "relation": vote_rel,
            "relation_id": vote_rid,
            "sensitivity": s,
            "head_weight": max(s, EPS),
            "selection_reason": "sample_sensitive_to_voted_relation",
        })

    rows.sort(
        key=lambda r: (
            float(r["sensitivity"]),
            -int(r["head_layer"]),
            -int(r["head"]),
        ),
        reverse=True,
    )
    return rows[: int(budget)], vote_rel, vote_rid, vote_counts


# =============================================================================
# Head -> token aggregation
# =============================================================================

def selected_heads_to_state_scores(
    selected_heads,
    token_proj_by_head,
    text_positions,
    head_input_offset,
    head_weight_mode,
    positive_token_only,
):
    """
    Each selected head already specifies which relation direction it uses.

    1) read that relation column from its [P,4] token projection;
    2) optionally zero negatives;
    3) convert to within-head percentile;
    4) combine heads that feed the same source state layer H-offset.

    Returns rows for every exact candidate state (source_layer, position).
    """
    by_source = defaultdict(list)

    for hr in selected_heads:
        H = int(hr["head_layer"])
        h = int(hr["head"])
        rid = int(hr["relation_id"])
        S = H - int(head_input_offset)

        proj = np.asarray(
            token_proj_by_head[(H, h)][:, rid],
            dtype=np.float64,
        )
        if positive_token_only:
            proj = np.maximum(proj, 0.0)

        pct = percentiles(proj)

        if head_weight_mode == "equal":
            w = 1.0
        else:
            w = max(float(hr["head_weight"]), EPS)

        by_source[S].append({
            "head_layer": H,
            "head": h,
            "head_name": hr["head_name"],
            "relation": hr["relation"],
            "raw_projection": proj,
            "percentile": pct,
            "weight": w,
            "sensitivity": float(hr["sensitivity"]),
        })

    rows = []
    for S, heads in by_source.items():
        W = np.asarray([x["weight"] for x in heads], dtype=np.float64)
        W = W / max(float(W.sum()), EPS)
        pct_mat = np.stack([x["percentile"] for x in heads], axis=0)
        raw_mat = np.stack([x["raw_projection"] for x in heads], axis=0)

        score = np.average(pct_mat, axis=0, weights=W)
        raw = np.average(raw_mat, axis=0, weights=W)

        head_names = ",".join(x["head_name"] for x in heads)
        head_rels = ",".join(
            f"{x['head_name']}:{x['relation']}" for x in heads
        )

        for j, pos in enumerate(text_positions):
            rows.append({
                "source_layer": int(S),
                "position": int(pos),
                "score": float(score[j]),
                "weighted_raw_projection": float(raw[j]),
                "n_heads_at_layer": int(len(heads)),
                "heads": head_names,
                "head_relations": head_rels,
            })

    return rows


def global_unique_rank(rows):
    """
    Oracle bank was built with global_unique:
      same token position across layers -> keep strongest layer.

    Do the same here.
    """
    best = {}
    for r in rows:
        pos = int(r["position"])
        if (
            pos not in best
            or float(r["score"]) > float(best[pos]["score"])
        ):
            best[pos] = dict(r)

    ranked = sorted(
        best.values(),
        key=lambda r: (
            float(r["score"]),
            float(r["weighted_raw_projection"]),
            -int(r["source_layer"]),
            -int(r["position"]),
        ),
        reverse=True,
    )
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank
    return ranked


# =============================================================================
# Recovery
# =============================================================================

def recovery(chosen, target):
    pred_states = {
        (int(r["source_layer"]), int(r["position"]))
        for r in chosen
    }
    pred_pos = {int(r["position"]) for r in chosen}

    tar_states = target["states"]
    tar_pos = target["positions"]

    exact = len(pred_states & tar_states)
    pos = len(pred_pos & tar_pos)

    return {
        "selected_N": len(pred_states),
        "target_state_N": len(tar_states),
        "target_position_N": len(tar_pos),
        "exact_hits": exact,
        "position_hits": pos,
        "exact_recall": safe_div(exact, len(tar_states)),
        "position_recall": safe_div(pos, len(tar_pos)),
        "exact_precision": safe_div(exact, len(pred_states)),
        "position_precision": safe_div(pos, len(pred_pos)),
    }


def random_expectation(chosen, target, text_positions):
    U = len(text_positions)
    if U <= 0:
        return np.nan, np.nan

    sel_by_layer = Counter(
        int(r["source_layer"]) for r in chosen
    )
    tar_by_layer = Counter(
        int(L) for L, _ in target["states"]
    )

    ex_exact = 0.0
    for L, nsel in sel_by_layer.items():
        ex_exact += (
            float(nsel)
            * float(tar_by_layer.get(L, 0))
            / float(U)
        )

    kpos = len({int(r["position"]) for r in chosen})
    ex_pos = (
        float(kpos)
        * float(len(target["positions"]))
        / float(U)
    )
    return ex_exact, ex_pos


# =============================================================================
# Token contribution cache
# =============================================================================

def token_cache_path(cache_dir, sid):
    return Path(cache_dir) / f"sid_{int(sid):06d}.npz"


def save_token_cache(
    path,
    sid,
    head_layers,
    text_positions,
    token_ids,
    tokens,
    categories,
    broad_categories,
    proj,
):
    # proj [n_head_layers, n_heads, n_text, 4]
    np.savez_compressed(
        path,
        sid=np.asarray([sid], dtype=np.int64),
        head_layers=np.asarray(head_layers, dtype=np.int64),
        positions=np.asarray(text_positions, dtype=np.int64),
        token_ids=np.asarray(token_ids, dtype=np.int64),
        tokens=np.asarray(tokens, dtype=object),
        categories=np.asarray(categories, dtype=object),
        broad_categories=np.asarray(broad_categories, dtype=object),
        projection=proj.astype(np.float16),
    )


def load_token_cache(path):
    z = np.load(path, allow_pickle=True)
    return {
        "sid": int(np.asarray(z["sid"]).reshape(-1)[0]),
        "head_layers": np.asarray(z["head_layers"], dtype=int),
        "positions": np.asarray(z["positions"], dtype=int),
        "token_ids": np.asarray(z["token_ids"], dtype=int),
        "tokens": np.asarray(z["tokens"], dtype=object),
        "categories": np.asarray(z["categories"], dtype=object),
        "broad_categories": np.asarray(
            z["broad_categories"], dtype=object
        ),
        "projection": np.asarray(
            z["projection"], dtype=np.float32
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    head_layers = parse_ints(a.head_layers)
    micro_budgets = parse_ints(a.micro_head_budgets)
    select_ks = parse_ints(a.select_ks)

    if not head_layers:
        raise ValueError("--head-layers is empty")
    if not micro_budgets:
        raise ValueError("--micro-head-budgets is empty")
    if not select_ks:
        raise ValueError("--select-ks is empty")

    outdir = Path(a.output_dir)
    cache_dir = outdir / "token_projection_cache"

    if a.overwrite and outdir.exists():
        for p in list(outdir.iterdir()):
            if p.name == "token_projection_cache":
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if a.overwrite_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # A. Synthetic-400 ONLY -> four directions.
    # -------------------------------------------------------------------------
    syn_path, syn_sid, syn_y, syn_X = load_direction_cache(
        a.source_cache
    )
    syn_center, syn_dirs = fit_synthetic_directions(
        syn_X, syn_y
    )

    # -------------------------------------------------------------------------
    # B. Existing COCO q_h cache -> per-head/sample four cosine scores.
    #    COCO GT is explicitly allowed for MACRO head accuracy.
    # -------------------------------------------------------------------------
    coco_path, coco_sid, coco_y, coco_X = load_direction_cache(
        a.coco_direction_cache
    )

    if syn_X.shape[1:] != coco_X.shape[1:]:
        raise RuntimeError(
            f"Synthetic/COCO direction geometry mismatch: "
            f"{syn_X.shape} vs {coco_X.shape}"
        )

    if max(head_layers) >= coco_X.shape[1]:
        raise RuntimeError(
            f"Requested head layer {max(head_layers)} but cache has "
            f"{coco_X.shape[1]} layers"
        )

    coco_scores = score_against_synthetic_dirs(
        coco_X,
        syn_center,
        syn_dirs,
    )

    coco_head_df = compute_coco_head_accuracy(
        coco_scores,
        coco_y,
        head_layers,
    )
    coco_head_df.to_csv(
        outdir / "coco_head_accuracy_from_synthetic_directions.csv",
        index=False,
    )

    macro_df = coco_head_df.head(int(a.macro_top_n)).copy()
    macro_df["macro_rank"] = np.arange(1, len(macro_df) + 1)
    macro_df.to_csv(
        outdir / "macro_coco_top_heads.csv",
        index=False,
    )

    macro_heads = [
        (int(r.head_layer), int(r.head))
        for r in macro_df.itertuples()
    ]
    coco_acc_lookup = {
        (int(r.head_layer), int(r.head)): float(r.coco_accuracy)
        for r in coco_head_df.itertuples()
    }

    all_heads = candidate_head_pool(
        head_layers,
        coco_X.shape[2],
    )

    # -------------------------------------------------------------------------
    # C. Oracle candidate bank.
    # -------------------------------------------------------------------------
    top_bank, core_bank = load_oracle_bank(
        a.oracle_bank_dir
    )
    top_lookup = target_lookup(top_bank)
    core_lookup = target_lookup(core_bank)

    coco_idx = {int(s): i for i, s in enumerate(coco_sid)}
    bank_sids = sorted(
        set(top_lookup)
        & set(core_lookup)
        & set(coco_idx)
    )

    if a.require_n and len(bank_sids) != int(a.require_n):
        raise RuntimeError(
            f"Expected N={a.require_n} overlapping bank/COCO SIDs; "
            f"got {len(bank_sids)}"
        )

    if a.max_samples and a.max_samples > 0:
        bank_sids = bank_sids[: int(a.max_samples)]

    # -------------------------------------------------------------------------
    # D. COCO metadata.
    # -------------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    meta = {}
    for sid in bank_sids:
        if sid not in prompts or sid not in rec_by_sid:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta[sid] = {
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        }

    missing = sorted(set(bank_sids) - set(meta))
    if missing:
        raise RuntimeError(
            f"Missing prompt/record metadata for {len(missing)} bank SIDs; "
            f"first={missing[:20]}"
        )

    # -------------------------------------------------------------------------
    # E. Determine which real-image token projection caches are missing.
    # -------------------------------------------------------------------------
    if a.cache_token_contrib:
        need_forward = [
            sid for sid in bank_sids
            if not token_cache_path(cache_dir, sid).exists()
        ]
    else:
        # We still need one forward per sample this run; no persistent cache.
        need_forward = list(bank_sids)

    # -------------------------------------------------------------------------
    # F. Load model and run REAL-image forward(s).
    # -------------------------------------------------------------------------
    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None

    # Outputs built during either fresh forward or cached post-processing.
    recovery_rows = []
    selected_state_rows = []
    selected_head_rows = []
    sample_vote_rows = []

    # We need model if any forward is required.
    if need_forward:
        print(
            f"Loading {spec.repo_id}; REAL-image forwards={len(need_forward)}",
            flush=True,
        )
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        cfg = get_text_config(model)
        n_heads_model = int(cfg.num_attention_heads)
        head_dim_model = int(
            (getattr(cfg, "hidden_size", 0) or
             resolve_o_proj(resolve_attn(decoder_layers[0])).in_features)
            // n_heads_model
        )

        if n_heads_model != coco_X.shape[2]:
            raise RuntimeError(
                f"Model heads={n_heads_model}, direction cache heads={coco_X.shape[2]}"
            )
        if head_dim_model != syn_X.shape[3]:
            raise RuntimeError(
                f"Model head_dim={head_dim_model}, direction dim={syn_X.shape[3]}"
            )

        print(
            f"decoder={decoder_path} | head_layers={head_layers} | "
            f"macro heads={[hname(*x) for x in macro_heads]}"
        )
    else:
        print("All token projection caches exist; skipping model load.")

    def process_one_sample(sid, sample_token_data):
        """
        Run all three methods from one sample's cached/fresh token projections.
        """
        ci = coco_idx[sid]
        sample_scores = coco_scores[ci]  # [L,H,4]
        gt = meta[sid]["gt"]

        head_layers_local = list(
            map(int, sample_token_data["head_layers"])
        )
        if head_layers_local != head_layers:
            raise RuntimeError(
                f"Cached head layers mismatch sid={sid}: "
                f"{head_layers_local} vs {head_layers}"
            )

        positions = list(
            map(int, sample_token_data["positions"])
        )
        projection = np.asarray(
            sample_token_data["projection"],
            dtype=np.float32,
        )  # [nLayers,H,P,4]

        layer_to_i = {
            H: i for i, H in enumerate(head_layers)
        }

        token_proj_by_head = {}
        for H in head_layers:
            li = layer_to_i[H]
            for h in range(projection.shape[1]):
                token_proj_by_head[(H, h)] = projection[li, h]

        # ---- relation vote diagnostic from fixed Macro TopN ----
        vote_rel, vote_rid, vote_counts = majority_vote_relation(
            macro_heads,
            sample_scores,
        )
        sample_vote_rows.append({
            "sid": sid,
            "gt": gt,
            "vote_relation": vote_rel,
            "vote_correct": vote_rel == gt,
            **{f"votes_{r}": int(vote_counts[r]) for r in REL},
        })

        # ------------------------------------------------------------------
        # METHOD 1: fixed Macro COCO-TopN.
        # ------------------------------------------------------------------
        macro_selected = method_macro_heads(
            macro_heads,
            sample_scores,
            coco_acc_lookup,
        )
        run_method(
            sid=sid,
            gt=gt,
            method="macro_coco_top10",
            head_budget=len(macro_selected),
            selected_heads=macro_selected,
            vote_relation=None,
            vote_correct=None,
            token_proj_by_head=token_proj_by_head,
            positions=positions,
            sample_token_data=sample_token_data,
        )

        # ------------------------------------------------------------------
        # METHODS 2 / 3: sample-specific head budgets.
        # ------------------------------------------------------------------
        for budget in micro_budgets:
            m2 = method_micro_argmax_heads(
                all_heads,
                sample_scores,
                budget,
                a.sensitivity_score,
            )
            run_method(
                sid=sid,
                gt=gt,
                method="micro_head_argmax",
                head_budget=budget,
                selected_heads=m2,
                vote_relation=None,
                vote_correct=None,
                token_proj_by_head=token_proj_by_head,
                positions=positions,
                sample_token_data=sample_token_data,
            )

            m3, vr, vrid, vc = method_micro_vote_sensitive_heads(
                all_heads,
                sample_scores,
                macro_heads,
                budget,
            )
            run_method(
                sid=sid,
                gt=gt,
                method="micro_vote_then_sensitive",
                head_budget=budget,
                selected_heads=m3,
                vote_relation=vr,
                vote_correct=(vr == gt),
                token_proj_by_head=token_proj_by_head,
                positions=positions,
                sample_token_data=sample_token_data,
            )

    def run_method(
        sid,
        gt,
        method,
        head_budget,
        selected_heads,
        vote_relation,
        vote_correct,
        token_proj_by_head,
        positions,
        sample_token_data,
    ):
        # Save selected heads.
        for rank, hr in enumerate(selected_heads, 1):
            selected_head_rows.append({
                "sid": sid,
                "gt": gt,
                "method": method,
                "head_budget": head_budget,
                "head_rank": rank,
                "head_layer": int(hr["head_layer"]),
                "head": int(hr["head"]),
                "head_name": hr["head_name"],
                "aligned_source_layer": (
                    int(hr["head_layer"]) - int(a.head_input_offset)
                ),
                "relation_used": hr["relation"],
                "sensitivity": float(hr["sensitivity"]),
                "head_weight": float(hr["head_weight"]),
                "vote_relation": vote_relation,
                "vote_correct": vote_correct,
            })

        state_rows = selected_heads_to_state_scores(
            selected_heads,
            token_proj_by_head,
            positions,
            a.head_input_offset,
            a.head_weight_mode,
            a.positive_token_only,
        )
        ranked = global_unique_rank(state_rows)

        max_k = max(select_ks)
        for rr in ranked[:max_k]:
            pos = int(rr["position"])
            try:
                j = positions.index(pos)
                tok = str(sample_token_data["tokens"][j])
                cat = str(sample_token_data["categories"][j])
                broad = str(
                    sample_token_data["broad_categories"][j]
                )
            except Exception:
                tok = cat = broad = ""

            selected_state_rows.append({
                "sid": sid,
                "gt": gt,
                "method": method,
                "head_budget": head_budget,
                "rank": int(rr["rank"]),
                "source_layer": int(rr["source_layer"]),
                "position": pos,
                "token": tok,
                "category": cat,
                "broad_category": broad,
                "score": float(rr["score"]),
                "weighted_raw_projection": float(
                    rr["weighted_raw_projection"]
                ),
                "n_heads_at_layer": int(rr["n_heads_at_layer"]),
                "heads": rr["heads"],
                "head_relations": rr["head_relations"],
                "vote_relation": vote_relation,
                "in_top10_exact": (
                    (int(rr["source_layer"]), pos)
                    in top_lookup[sid]["states"]
                ),
                "in_core50_exact": (
                    (int(rr["source_layer"]), pos)
                    in core_lookup[sid]["states"]
                ),
                "in_top10_position": (
                    pos in top_lookup[sid]["positions"]
                ),
                "in_core50_position": (
                    pos in core_lookup[sid]["positions"]
                ),
            })

        for K in select_ks:
            chosen = ranked[:K]
            for target_name, target in (
                ("top10", top_lookup[sid]),
                ("core50", core_lookup[sid]),
            ):
                met = recovery(chosen, target)
                ex_exact, ex_pos = random_expectation(
                    chosen,
                    target,
                    positions,
                )
                recovery_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "method": method,
                    "head_budget": head_budget,
                    "K": K,
                    "target": target_name,
                    "vote_relation": vote_relation,
                    "vote_correct": vote_correct,
                    **met,
                    "expected_random_exact_hits": ex_exact,
                    "expected_random_position_hits": ex_pos,
                    "exact_enrichment_over_random": (
                        met["exact_hits"] / ex_exact
                        if np.isfinite(ex_exact) and ex_exact > 0
                        else np.nan
                    ),
                    "position_enrichment_over_random": (
                        met["position_hits"] / ex_pos
                        if np.isfinite(ex_pos) and ex_pos > 0
                        else np.nan
                    ),
                })

    try:
        # Fresh forward samples.
        for sid in tqdm(
            bank_sids,
            desc="THREE spatial-head token selectors",
        ):
            cp = token_cache_path(cache_dir, sid)

            if a.cache_token_contrib and cp.exists():
                sample_token_data = load_token_cache(cp)
                process_one_sample(sid, sample_token_data)
                continue

            if model is None or processor is None:
                raise RuntimeError(
                    f"Missing token cache for sid={sid} but model was not loaded."
                )

            m = meta[sid]
            image = rb = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=torch.device(a.device),
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                spos, rpos = object_positions(
                    processor,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                cats, toks = dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                npos = min(len(ids), len(cats), len(toks))
                text_positions = [
                    p for p in range(max(0, npos - 1))
                    if dyn.broad_category(cats[p]) != "visual"
                ]
                if not text_positions:
                    raise RuntimeError(
                        f"No eligible text positions sid={sid}"
                    )

                cap = run_capture(
                    model,
                    decoder_layers,
                    rb,
                    head_layers,
                )

                nL = len(head_layers)
                nH = cap["n_heads"]
                nP = len(text_positions)
                proj = np.full(
                    (nL, nH, nP, 4),
                    np.nan,
                    dtype=np.float32,
                )

                for li, H in enumerate(head_layers):
                    for h in range(nH):
                        proj[li, h] = (
                            real_token_projection_all_relations(
                                cap=cap,
                                head_layer=H,
                                head=h,
                                subpos=spos,
                                refpos=rpos,
                                direction_pool=a.direction_pool,
                                dirs_4xd=syn_dirs[H, h],
                                text_positions=text_positions,
                            )
                        )

                sample_token_data = {
                    "sid": sid,
                    "head_layers": np.asarray(
                        head_layers, dtype=int
                    ),
                    "positions": np.asarray(
                        text_positions, dtype=int
                    ),
                    "token_ids": np.asarray(
                        [int(ids[p]) for p in text_positions],
                        dtype=int,
                    ),
                    "tokens": np.asarray(
                        [
                            str(toks[p]).replace("\n", "\\n")
                            for p in text_positions
                        ],
                        dtype=object,
                    ),
                    "categories": np.asarray(
                        [str(cats[p]) for p in text_positions],
                        dtype=object,
                    ),
                    "broad_categories": np.asarray(
                        [
                            dyn.broad_category(cats[p])
                            for p in text_positions
                        ],
                        dtype=object,
                    ),
                    "projection": proj,
                }

                if a.cache_token_contrib:
                    save_token_cache(
                        cp,
                        sid=sid,
                        head_layers=head_layers,
                        text_positions=text_positions,
                        token_ids=sample_token_data["token_ids"],
                        tokens=sample_token_data["tokens"],
                        categories=sample_token_data["categories"],
                        broad_categories=sample_token_data[
                            "broad_categories"
                        ],
                        proj=proj,
                    )

                process_one_sample(
                    sid,
                    sample_token_data,
                )

                del cap
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                if rb is not None:
                    del rb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # G. Save per-sample detail.
    # -------------------------------------------------------------------------
    rec_df = pd.DataFrame(recovery_rows)
    states_df = pd.DataFrame(selected_state_rows)
    heads_df = pd.DataFrame(selected_head_rows)
    votes_df = pd.DataFrame(sample_vote_rows)

    rec_df.to_csv(
        outdir / "recovery_per_sample.csv",
        index=False,
    )
    states_df.to_csv(
        outdir / "selected_states_per_sample.csv",
        index=False,
    )
    heads_df.to_csv(
        outdir / "selected_heads_per_sample.csv",
        index=False,
    )
    votes_df.to_csv(
        outdir / "macro_head_vote_per_sample.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # H. Aggregate recovery.
    # -------------------------------------------------------------------------
    summary_rows = []
    group_cols = [
        "method",
        "head_budget",
        "K",
        "target",
    ]

    for key, g in rec_df.groupby(group_cols):
        method, hb, K, target = key

        total_exact = int(g["exact_hits"].sum())
        total_tar = int(g["target_state_N"].sum())
        total_pos = int(g["position_hits"].sum())
        total_tar_pos = int(g["target_position_N"].sum())
        total_selected = int(g["selected_N"].sum())

        ex_exact = float(
            g["expected_random_exact_hits"].sum()
        )
        ex_pos = float(
            g["expected_random_position_hits"].sum()
        )

        summary_rows.append({
            "method": method,
            "head_budget": hb,
            "K": K,
            "target": target,
            "N": int(g["sid"].nunique()),
            "micro_exact_recall": safe_div(
                total_exact, total_tar
            ),
            "macro_exact_recall": safe_mean(
                g["exact_recall"]
            ),
            "micro_position_recall": safe_div(
                total_pos, total_tar_pos
            ),
            "macro_position_recall": safe_mean(
                g["position_recall"]
            ),
            "micro_exact_precision": safe_div(
                total_exact, total_selected
            ),
            "micro_position_precision": safe_div(
                total_pos, total_selected
            ),
            "exact_enrichment_over_random": (
                total_exact / ex_exact
                if ex_exact > 0 else np.nan
            ),
            "position_enrichment_over_random": (
                total_pos / ex_pos
                if ex_pos > 0 else np.nan
            ),
            "mean_exact_hits": safe_mean(
                g["exact_hits"]
            ),
            "mean_position_hits": safe_mean(
                g["position_hits"]
            ),
            "vote_accuracy": (
                safe_mean(
                    g.loc[
                        g["vote_correct"].notna(),
                        "vote_correct",
                    ].astype(float)
                )
                if g["vote_correct"].notna().any()
                else np.nan
            ),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "target",
            "micro_exact_recall",
            "micro_position_recall",
        ],
        ascending=[True, False, False],
    )
    summary.to_csv(
        outdir / "recovery_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # I. Macro head list + vote diagnostics + report.
    # -------------------------------------------------------------------------
    vote_acc = (
        float(votes_df["vote_correct"].mean())
        if len(votes_df)
        else np.nan
    )

    # Relation breakdown for macro vote.
    vote_rel_rows = []
    if len(votes_df):
        for r in REL:
            g = votes_df[votes_df["gt"] == r]
            if len(g):
                vote_rel_rows.append({
                    "relation": r,
                    "N": len(g),
                    "vote_accuracy": float(
                        g["vote_correct"].mean()
                    ),
                })
    pd.DataFrame(vote_rel_rows).to_csv(
        outdir / "macro_vote_relation_accuracy.csv",
        index=False,
    )

    lines = []
    lines.append("=" * 176)
    lines.append(
        "THREE SPATIAL-HEAD -> CAUSAL-TOKEN SELECTION METHODS"
    )
    lines.append("=" * 176)
    lines.append(
        f"N={len(bank_sids)} | Synthetic supplies directions only | "
        f"Macro head ranking uses COCO GT | "
        f"token source decomposition uses REAL image only"
    )
    lines.append("")

    lines.append("METHOD 1 — MACRO COCO TOP HEADS")
    lines.append("-" * 176)
    show_macro = macro_df[
        [
            "macro_rank",
            "head_name",
            "head_layer",
            "head",
            "coco_accuracy",
            "acc_left",
            "acc_right",
            "acc_above",
            "acc_below",
        ]
    ]
    lines.append(
        show_macro.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    lines.append("")
    lines.append(
        f"Macro Top{a.macro_top_n} majority-vote relation accuracy "
        f"on evaluated COCO samples = {vote_acc:.4f}"
    )
    lines.append("")

    lines.append("BEST CORE50 RECOVERY")
    lines.append("-" * 176)
    q = summary[
        summary["target"] == "core50"
    ].head(30)
    show = [
        "method",
        "head_budget",
        "K",
        "micro_exact_recall",
        "micro_position_recall",
        "exact_enrichment_over_random",
        "position_enrichment_over_random",
        "vote_accuracy",
    ]
    lines.append(
        q[show].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    lines.append("")

    lines.append("BEST TOP10 RECOVERY")
    lines.append("-" * 176)
    q = summary[
        summary["target"] == "top10"
    ].head(30)
    lines.append(
        q[show].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    report = "\n".join(lines) + "\n"
    (outdir / "analysis_summary.txt").write_text(
        report,
        encoding="utf-8",
    )
    print(report)

    metadata = {
        "model": a.model,
        "synthetic_direction_cache": str(syn_path),
        "coco_direction_cache": str(coco_path),
        "oracle_bank_dir": str(Path(a.oracle_bank_dir)),
        "N": len(bank_sids),
        "head_layers": head_layers,
        "source_layers": [
            H - int(a.head_input_offset)
            for H in head_layers
        ],
        "macro_top_n": int(a.macro_top_n),
        "macro_heads": [
            hname(H, h) for H, h in macro_heads
        ],
        "micro_head_budgets": micro_budgets,
        "select_ks": select_ks,
        "direction_pool": a.direction_pool,
        "sensitivity_score_method2": a.sensitivity_score,
        "head_weight_mode": a.head_weight_mode,
        "positive_token_only": bool(
            a.positive_token_only
        ),
        "method_definitions": {
            "macro_coco_top10": (
                "Synthetic directions -> COCO per-head classification ACC -> "
                "fixed global TopN heads; each head uses its own sample argmax relation."
            ),
            "micro_head_argmax": (
                "For each sample, every candidate head argmaxes four Synthetic "
                "directions; sample Top-M heads selected by max cosine or margin; "
                "each head uses its own argmax direction."
            ),
            "micro_vote_then_sensitive": (
                "Macro COCO-TopN heads majority-vote one relation; then all "
                "candidate heads are ranked by cosine to that voted direction; "
                "Top-M heads all use that same voted direction."
            ),
        },
        "token_score": (
            "real-image c_h,p=[A_h(sub,p)-A_h(ref,p)]V_h(p), projected onto "
            "the relation direction assigned by each method; per-head token "
            "projection converted to within-head percentile."
        ),
        "forbidden_from_token_selector": [
            "gray image",
            "writer vector",
            "writer gradient",
            "answer logit",
            "oracle causal score",
        ],
        "oracle_bank_usage": "evaluation only",
        "important_caveat": (
            "Method 1 uses COCO GT labels to rank heads by accuracy and is a "
            "mechanistic macro analysis, not a label-free deployment method."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Saved:", outdir)


if __name__ == "__main__":
    main()
