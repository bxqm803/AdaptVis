#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spatial_heads_recover_oracle_text_tokens_all440_v1.py

Question
--------
Can high-accuracy spatial attention heads identify, for EACH SAMPLE, the text
token states that belong to the oracle writer-guided Top10 / Core50 causal bank?

This script is deliberately NOT a repair/generation experiment.

Spatial-head side uses:
    REAL IMAGE ONLY
    NO gray image
    NO writer
    NO writer gradient
    NO GT relation
    NO target answer/logit

Oracle Top10/Core50 is used ONLY as an evaluation target after spatial-head
token rankings have already been computed.

Source-only head selection
--------------------------
A Synthetic-400 Direction-Head cache is used to:
  1) compute 5-fold OOF four-way spatial accuracy for every head;
  2) rank heads within each head layer;
  3) fit a four-relation spatial subspace for each head.

The COCO sample itself supplies only the real-image attention computation.

Spatial token scores
--------------------
For a head h at layer H and source token p,

    c_{h,p} = [A_h(sub,p) - A_h(ref,p)] V_h(p)

and

    q_h = sum_p c_{h,p}.

Four scores are tested in the SAME real-image forward pass:

  attention_abs
      |A_h(sub,p) - A_h(ref,p)|

  av_norm
      ||c_{h,p}||

  self_align
      <c_{h,p}, normalize(q_h)>

  spatial_self_align
      <P_h c_{h,p}, normalize(P_h q_h)>

where P_h projects onto the source-learned spatial subspace

    span{d_h,left, d_h,right, d_h,above, d_h,below}.

No hard relation prediction is needed for self_align or spatial_self_align.

Layer alignment
---------------
Causal candidate state at decoder block OUTPUT L is consumed by attention block
H=L+1, matching the project's existing convention.

Consensus
---------
For each head:
  - convert token scores to within-head percentiles;
  - combine high-accuracy heads at the same head layer;
  - optionally weight by source OOF accuracy and current-sample spatialness;
  - map head layer H -> candidate state layer H-1;
  - enforce global_unique by token position across source layers;
  - take Top-K exact (layer, position) states.

Outputs
-------
source_oof_head_accuracy.csv
selected_heads_by_layer.csv
per_head_recovery_summary.csv
head_accuracy_recovery_correlation.csv
consensus_recovery_summary.csv
consensus_recovery_per_sample.csv
spatial_selected_topmax.csv
analysis_summary.txt
metadata.json

Per-sample real-image spatial scores are cached under:
    <output-dir>/sample_score_cache/sid_XXXXXX.npz

Therefore changing head budgets / consensus weights / recovery K does NOT need
another model forward. Run once, then reuse the cache.

Typical
-------
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_spatial_heads_recover_oracle_text_tokens_all440_v1.py \
  --model qwen-3b \
  --source-cache \
    output/qwen3b_direction_selector_syn400_to_coco440/synthetic_direction_head_relation_vectors.npz \
  --oracle-bank-dir output/qwen3b_oracle_core50_top10_bank_all440 \
  --head-layers 21,22,23,24,25,26,27 \
  --head-budgets 1,2,3,5 \
  --select-ks 3,5,7,10 \
  --weight-modes equal,acc,acc_spatialness \
  --require-n 440 \
  --output-dir output/qwen3b_spatial_head_token_recovery_all440 \
  --overwrite
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
REL_TO_ID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12
SCORE_METHODS = (
    "attention_abs",
    "av_norm",
    "self_align",
    "spatial_self_align",
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help=(
            "Synthetic-400 direction-head residual NPZ produced by the v2 selector, "
            "containing sample_index/relation/residual."
        ),
    )
    p.add_argument(
        "--oracle-bank-dir",
        required=True,
        help=(
            "Directory produced by build_oracle_core50_top10_candidate_bank_all440_v1.py"
        ),
    )

    p.add_argument(
        "--head-layers",
        default="21,22,23,24,25,26,27",
        help="Attention-head layers. Their input states are H-1.",
    )
    p.add_argument(
        "--head-budgets",
        default="1,2,3,5",
        help="Top-M source-OOF heads PER head layer.",
    )
    p.add_argument(
        "--select-ks",
        default="3,5,7,10",
        help="Global-unique spatial states selected per sample.",
    )
    p.add_argument(
        "--weight-modes",
        default="equal,acc,acc_spatialness",
        help="Subset of equal,acc,acc_spatialness.",
    )
    p.add_argument(
        "--direction-pool",
        choices=["mean", "last"],
        default="mean",
        help="Pool subject/reference query tokens for head decomposition.",
    )
    p.add_argument("--head-input-offset", type=int, default=1)

    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=17)
    p.add_argument(
        "--min-head-acc",
        type=float,
        default=0.0,
        help=(
            "Optional source OOF accuracy threshold after per-layer ranking. "
            "0 keeps all; head budget still chooses the highest-accuracy heads."
        ),
    )
    p.add_argument(
        "--per-head-k",
        type=int,
        default=10,
        help="Top text positions per individual head for head-level recovery analysis.",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--require-n",
        type=int,
        default=440,
        help="Require this many bank SIDs; 0 disables.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite summary outputs, but preserve sample forward cache.",
    )
    p.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Also redo all real-image model forwards.",
    )
    p.add_argument("--output-dir", required=True)

    return p.parse_args()


# =============================================================================
# Generic
# =============================================================================

def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def parse_strs(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


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


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def rankdata_average(a):
    """Average ranks for ties, 1-based."""
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


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    return safe_corr(rankdata_average(x), rankdata_average(y))


# =============================================================================
# Source-only Direction-Head codebook / OOF accuracy
# =============================================================================

def load_source_cache(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    required = {"relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{p} missing arrays: {sorted(missing)}")

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)
    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    else:
        sid = np.arange(len(y), dtype=int)

    valid = np.isin(y, np.asarray(REL, dtype=object))
    return p, sid[valid], y[valid], X[valid]


def fit_source_codebook(X, y):
    center = X.mean(axis=0).astype(np.float32)
    dirs = np.zeros(
        (X.shape[1], X.shape[2], len(REL), X.shape[3]),
        dtype=np.float32,
    )
    for ri, r in enumerate(REL):
        m = y == r
        if not np.any(m):
            raise RuntimeError(f"No source samples for relation={r}")
        dirs[:, :, ri, :] = normalize(X[m].mean(axis=0) - center, axis=-1)
    return center, dirs


def score_source(X, center, dirs):
    Xn = normalize(X - center[None, :, :, :], axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", Xn, dirs, optimize=True)


def stratified_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_folds)]
    for r in REL:
        idx = np.where(y == r)[0].copy()
        rng.shuffle(idx)
        pieces = np.array_split(idx, n_folds)
        for f, piece in enumerate(pieces):
            buckets[f].extend(piece.tolist())
    return [np.asarray(sorted(x), dtype=int) for x in buckets]


def source_oof_head_accuracy(X, y, folds, seed):
    N = len(y)
    yi = np.asarray([REL_TO_ID[r] for r in y], dtype=int)
    split = stratified_folds(y, folds, seed)
    correct = np.zeros((X.shape[1], X.shape[2]), dtype=np.int64)
    all_idx = np.arange(N)

    for te in split:
        keep = np.ones(N, dtype=bool)
        keep[te] = False
        tr = all_idx[keep]
        center, dirs = fit_source_codebook(X[tr], y[tr])
        sc = score_source(X[te], center, dirs)
        pred = np.argmax(sc, axis=-1)
        correct += (pred == yi[te, None, None]).sum(axis=0)

    return correct.astype(np.float32) / float(N)


def spatial_basis(dirs_4xd):
    """
    Orthonormal basis for span of the four source relation directions.
    Centered 4-class directions typically have rank <=3.
    """
    D = np.asarray(dirs_4xd, dtype=np.float64)
    if not np.isfinite(D).all():
        return np.zeros((D.shape[1], 0), dtype=np.float32)
    _u, s, vt = np.linalg.svd(D, full_matrices=False)
    if not len(s):
        return np.zeros((D.shape[1], 0), dtype=np.float32)
    tol = max(D.shape) * np.finfo(np.float64).eps * float(s[0])
    rank = int(np.sum(s > max(tol, 1e-8)))
    if rank <= 0:
        return np.zeros((D.shape[1], 0), dtype=np.float32)
    return vt[:rank].T.astype(np.float32)  # [D, rank]


def project_to_basis(x, B):
    x = np.asarray(x, dtype=np.float32)
    if B.shape[1] == 0:
        return np.zeros_like(x)
    return (x @ B @ B.T).astype(np.float32)


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

    top = pd.read_csv(top_path)
    core = pd.read_csv(core_path)

    req = {"sid", "source_layer", "position", "category", "broad_category"}
    for name, df in [("top10", top), ("core50", core)]:
        missing = req - set(df.columns)
        if missing:
            raise RuntimeError(f"{name} bank missing: {sorted(missing)}")
        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
        df["source_layer"] = pd.to_numeric(
            df["source_layer"], errors="raise"
        ).astype(int)
        df["position"] = pd.to_numeric(
            df["position"], errors="raise"
        ).astype(int)

        if "is_text" in df.columns:
            df["is_text"] = df["is_text"].astype(str).str.lower().isin(
                {"true", "1", "yes", "t"}
            )
        else:
            df["is_text"] = (
                ~df["broad_category"].astype(str).eq("visual")
                & ~df["category"].astype(str).eq("visual")
            )

    return top, core


def build_target_lookup(df):
    out = {}
    for sid, g in df.groupby("sid"):
        all_states = {
            (int(r.source_layer), int(r.position))
            for r in g.itertuples()
        }
        text_g = g[g["is_text"]]
        text_states = {
            (int(r.source_layer), int(r.position))
            for r in text_g.itertuples()
        }
        out[int(sid)] = {
            "all_states": all_states,
            "text_states": text_states,
            "all_positions": {p for _, p in all_states},
            "text_positions": {p for _, p in text_states},
        }
    return out


# =============================================================================
# Model capture: REAL image only
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


class RealSpatialCapture:
    def __init__(self, decoder_layers, head_layers):
        cfg = get_text_config_from_layers_or_fail(decoder_layers)
        # overwritten by caller after model config if necessary
        self.n_heads = int(cfg["n_heads"])
        self.n_kv_heads = int(cfg["n_kv_heads"])
        self.hidden_size = int(cfg["hidden_size"])
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


# Global shape populated before RealSpatialCapture is used.
_CAPTURE_SHAPE = None


def get_text_config_from_layers_or_fail(_decoder_layers):
    if _CAPTURE_SHAPE is None:
        raise RuntimeError("Internal capture shape has not been initialized")
    return _CAPTURE_SHAPE


@torch.inference_mode()
def run_real_capture(model, decoder_layers, batch, head_layers):
    cap = RealSpatialCapture(decoder_layers, head_layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = True
        out = model(**kw)

        att_all = extract_attentions(out)
        att = {L: norm_attn(att_all[L]) for L in head_layers}

        return {
            "pre_o": cap.pre_o,
            "v": cap.v,
            "attn": att,
            "n_heads": cap.n_heads,
            "n_kv_heads": cap.n_kv_heads,
            "head_dim": cap.head_dim,
        }
    finally:
        cap.close()


def get_head_pre_o(pre_o, L, h, n_heads, head_dim):
    x = pre_o[L][0]
    y = x.reshape(x.shape[0], n_heads, head_dim)
    return y[:, h, :]


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
            raise RuntimeError(f"Cannot map query heads={n_heads} to kv heads={nkv}")
        kvh = h // repeat

    return x[:, kvh, :]


def span_positions(span):
    return list(range(int(span[0]), int(span[1]) + 1))


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
        raise RuntimeError("No valid object query positions")
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


def token_scores_one_head(
    cap,
    L,
    h,
    subpos,
    refpos,
    pool,
    spatial_B,
    text_positions,
):
    """
    REAL IMAGE ONLY.

    c_p = (A_sub,p - A_ref,p) * V_p
    q   = sum_p c_p

    Returns score vectors aligned to text_positions plus diagnostics.
    """
    A = cap["attn"][L][h]  # [q,k]
    V = get_v_for_attention_head(
        cap["v"][L],
        h,
        cap["n_heads"],
        cap["n_kv_heads"],
        cap["head_dim"],
    )

    a_s = pool_query_rows(A, subpos, pool)
    a_r = pool_query_rows(A, refpos, pool)

    n = min(len(a_s), len(a_r), V.shape[0])
    coeff = (a_s[:n] - a_r[:n]).astype(np.float32)
    vec = coeff[:, None] * V[:n]  # [key, d]
    q = vec.sum(axis=0).astype(np.float32)

    # Validate A*V reconstruction against captured pre-o_proj head output.
    H = get_head_pre_o(
        cap["pre_o"], L, h, cap["n_heads"], cap["head_dim"]
    )
    q_capture = (
        pool_head_states(H, subpos, pool)
        - pool_head_states(H, refpos, pool)
    ).astype(np.float32)
    recon_cos = cosine(q, q_capture)
    recon_relerr = float(
        np.linalg.norm(q - q_capture)
        / max(float(np.linalg.norm(q_capture)), EPS)
    )

    qhat = normalize(q)
    proj_q = project_to_basis(q, spatial_B)
    proj_q_norm = float(np.linalg.norm(proj_q))
    q_norm = float(np.linalg.norm(q))
    spatialness = proj_q_norm / max(q_norm, EPS)
    proj_q_hat = normalize(proj_q)

    valid_pos = [p for p in text_positions if 0 <= int(p) < n]
    if not valid_pos:
        raise RuntimeError("No eligible text source positions")

    idx = np.asarray(valid_pos, dtype=np.int64)
    c = vec[idx]
    c_sp = project_to_basis(c, spatial_B)

    scores = {
        "attention_abs": np.abs(coeff[idx]).astype(np.float32),
        "av_norm": np.linalg.norm(c, axis=1).astype(np.float32),
        "self_align": (c @ qhat).astype(np.float32),
        "spatial_self_align": (c_sp @ proj_q_hat).astype(np.float32),
    }

    return valid_pos, scores, {
        "q_norm": q_norm,
        "spatial_q_norm": proj_q_norm,
        "spatialness": spatialness,
        "reconstruction_cosine": recon_cos,
        "reconstruction_relative_error": recon_relerr,
    }


# =============================================================================
# Per-sample score cache
# =============================================================================

def cache_path(cache_dir, sid):
    return Path(cache_dir) / f"sid_{int(sid):06d}.npz"


def save_sample_cache(
    path,
    sid,
    head_layers,
    positions,
    token_ids,
    tokens,
    categories,
    broad_categories,
    scores,
    spatialness,
    recon_cos,
    recon_relerr,
):
    np.savez_compressed(
        path,
        sid=np.asarray([sid], dtype=np.int64),
        head_layers=np.asarray(head_layers, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.int64),
        token_ids=np.asarray(token_ids, dtype=np.int64),
        tokens=np.asarray(tokens, dtype=object),
        categories=np.asarray(categories, dtype=object),
        broad_categories=np.asarray(broad_categories, dtype=object),
        attention_abs=scores["attention_abs"].astype(np.float16),
        av_norm=scores["av_norm"].astype(np.float16),
        self_align=scores["self_align"].astype(np.float16),
        spatial_self_align=scores["spatial_self_align"].astype(np.float16),
        spatialness=spatialness.astype(np.float32),
        reconstruction_cosine=recon_cos.astype(np.float32),
        reconstruction_relative_error=recon_relerr.astype(np.float32),
    )


def load_sample_cache(path):
    z = np.load(path, allow_pickle=True)
    out = {
        "sid": int(np.asarray(z["sid"]).reshape(-1)[0]),
        "head_layers": np.asarray(z["head_layers"], dtype=int),
        "positions": np.asarray(z["positions"], dtype=int),
        "token_ids": np.asarray(z["token_ids"], dtype=int),
        "tokens": np.asarray(z["tokens"], dtype=object),
        "categories": np.asarray(z["categories"], dtype=object),
        "broad_categories": np.asarray(z["broad_categories"], dtype=object),
        "spatialness": np.asarray(z["spatialness"], dtype=np.float32),
        "reconstruction_cosine": np.asarray(
            z["reconstruction_cosine"], dtype=np.float32
        ),
        "reconstruction_relative_error": np.asarray(
            z["reconstruction_relative_error"], dtype=np.float32
        ),
    }
    for m in SCORE_METHODS:
        out[m] = np.asarray(z[m], dtype=np.float32)
    return out


# =============================================================================
# Recovery helpers
# =============================================================================

def selected_heads_per_layer(head_layers, oof_acc, budget, min_acc):
    out = {}
    for L in head_layers:
        rows = [
            (float(oof_acc[L, h]), int(h))
            for h in range(oof_acc.shape[1])
            if float(oof_acc[L, h]) >= float(min_acc)
        ]
        rows.sort(key=lambda x: x[0], reverse=True)
        out[L] = rows[: int(budget)]
    return out


def consensus_state_scores(
    sample,
    method,
    selected_by_layer,
    oof_acc,
    weight_mode,
    head_input_offset,
):
    """
    Returns one row per (source_layer, text position), prior to global_unique.
    score is a weighted mean of within-head percentiles.
    """
    head_layers = list(map(int, sample["head_layers"]))
    layer_to_i = {L: i for i, L in enumerate(head_layers)}
    positions = sample["positions"]
    rows = []

    for H in head_layers:
        selected = selected_by_layer.get(H, [])
        if not selected:
            continue

        li = layer_to_i[H]
        per_head_pct = []
        weights = []
        head_names = []

        for acc, h in selected:
            raw = sample[method][li, h, :]
            pct = percentiles(raw)

            if weight_mode == "equal":
                w = 1.0
            elif weight_mode == "acc":
                w = max(float(acc) - 0.25, 0.0)
            elif weight_mode == "acc_spatialness":
                sp = max(float(sample["spatialness"][li, h]), 0.0)
                w = max(float(acc) - 0.25, 0.0) * sp
            else:
                raise ValueError(weight_mode)

            if not np.isfinite(w) or w <= 0:
                continue

            per_head_pct.append(pct)
            weights.append(w)
            head_names.append(hname(H, h))

        if not per_head_pct:
            continue

        arr = np.stack(per_head_pct, axis=0)
        ww = np.asarray(weights, dtype=np.float64)
        score = np.average(arr, axis=0, weights=ww)

        S = int(H) - int(head_input_offset)

        for j, p in enumerate(positions):
            rows.append(
                {
                    "source_layer": S,
                    "head_layer": H,
                    "position": int(p),
                    "score": float(score[j]),
                    "n_heads": int(len(weights)),
                    "heads": ",".join(head_names),
                }
            )

    return rows


def global_unique_rank(rows):
    """
    Match oracle global_unique logic:
    same token position may appear at multiple source layers; keep the layer with
    strongest spatial consensus score.
    """
    best = {}
    for r in rows:
        p = int(r["position"])
        if p not in best or float(r["score"]) > float(best[p]["score"]):
            best[p] = r
    ranked = sorted(
        best.values(),
        key=lambda r: (
            float(r["score"]),
            -int(r["source_layer"]),
            -int(r["position"]),
        ),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r["spatial_rank"] = i
    return ranked


def recovery_metrics(selected, target):
    selected_states = {
        (int(r["source_layer"]), int(r["position"]))
        for r in selected
    }
    selected_positions = {
        int(r["position"]) for r in selected
    }

    target_states = set(target["text_states"])
    target_positions = set(target["text_positions"])

    exact_hit = len(selected_states & target_states)
    pos_hit = len(selected_positions & target_positions)

    return {
        "selected_K": len(selected_states),
        "target_text_state_N": len(target_states),
        "target_text_position_N": len(target_positions),
        "exact_hits": exact_hit,
        "position_hits": pos_hit,
        "exact_recall": safe_div(exact_hit, len(target_states)),
        "position_recall": safe_div(pos_hit, len(target_positions)),
        "exact_precision": safe_div(exact_hit, len(selected_states)),
        "position_precision": safe_div(pos_hit, len(selected_positions)),
    }


def random_expectation(selected, target, universe_positions):
    """
    Deterministic matched-layer expectation.

    Exact-state expectation:
      sum_L n_selected(L) * n_target(L) / |text positions|

    Position expectation:
      K_unique * n_target_positions / |text positions|

    This is the expectation under uniform text-position selection and keeps the
    spatial selector's observed layer budget for exact-state matching.
    """
    U = len(universe_positions)
    if U <= 0:
        return {
            "expected_exact_hits": np.nan,
            "expected_position_hits": np.nan,
        }

    sel_by_L = Counter(int(r["source_layer"]) for r in selected)
    tar_by_L = Counter(int(L) for L, _ in target["text_states"])

    ex_exact = 0.0
    for L, nsel in sel_by_L.items():
        ex_exact += float(nsel) * float(tar_by_L.get(L, 0)) / float(U)

    kpos = len({int(r["position"]) for r in selected})
    ex_pos = (
        float(kpos) * float(len(target["text_positions"])) / float(U)
    )

    return {
        "expected_exact_hits": ex_exact,
        "expected_position_hits": ex_pos,
    }


def per_head_recovery(
    sample,
    source_oof_acc,
    targets,
    per_head_k,
    head_input_offset,
):
    rows = []
    positions = sample["positions"]
    U = len(positions)

    for li, H in enumerate(sample["head_layers"]):
        H = int(H)
        S = H - int(head_input_offset)

        for h in range(source_oof_acc.shape[1]):
            acc = float(source_oof_acc[H, h])

            for method in SCORE_METHODS:
                raw = sample[method][li, h, :]
                order = np.argsort(-raw, kind="mergesort")
                take = order[: min(int(per_head_k), len(order))]
                chosen_pos = {int(positions[j]) for j in take}

                for target_name, target in targets.items():
                    tar = {
                        p for L, p in target["text_states"]
                        if int(L) == int(S)
                    }
                    if not tar:
                        continue

                    hit = len(chosen_pos & tar)
                    expected = (
                        len(chosen_pos) * len(tar) / float(U)
                        if U > 0 else np.nan
                    )

                    rows.append(
                        {
                            "head_name": hname(H, h),
                            "head_layer": H,
                            "head": h,
                            "aligned_source_layer": S,
                            "source_oof_accuracy": acc,
                            "method": method,
                            "target": target_name,
                            "target_states_at_layer": len(tar),
                            "selected_positions": len(chosen_pos),
                            "hits": hit,
                            "recall": safe_div(hit, len(tar)),
                            "expected_random_hits": expected,
                            "enrichment_over_random": (
                                hit / expected
                                if np.isfinite(expected) and expected > 0
                                else np.nan
                            ),
                            "spatialness": float(sample["spatialness"][li, h]),
                            "reconstruction_cosine": float(
                                sample["reconstruction_cosine"][li, h]
                            ),
                        }
                    )
    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    global _CAPTURE_SHAPE

    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    head_layers = parse_ints(a.head_layers)
    head_budgets = parse_ints(a.head_budgets)
    select_ks = parse_ints(a.select_ks)
    weight_modes = parse_strs(a.weight_modes)

    bad_weights = [
        x for x in weight_modes
        if x not in {"equal", "acc", "acc_spatialness"}
    ]
    if bad_weights:
        raise ValueError(f"Unknown weight modes: {bad_weights}")

    if not head_layers:
        raise ValueError("--head-layers is empty")
    if not head_budgets:
        raise ValueError("--head-budgets is empty")
    if not select_ks:
        raise ValueError("--select-ks is empty")

    outdir = Path(a.output_dir)
    cache_dir = outdir / "sample_score_cache"

    if a.overwrite and outdir.exists():
        # Preserve expensive per-sample forward cache unless explicitly asked
        # to overwrite it.
        for p in outdir.iterdir():
            if p.name == "sample_score_cache":
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if a.overwrite_cache:
        for p in cache_dir.glob("sid_*.npz"):
            p.unlink()

    # -------------------------------------------------------------------------
    # 1) Source-only head accuracy and spatial subspaces.
    # -------------------------------------------------------------------------
    source_path, source_sid, source_y, source_X = load_source_cache(
        a.source_cache
    )

    if max(head_layers) >= source_X.shape[1]:
        raise RuntimeError(
            f"Head layer {max(head_layers)} exceeds source cache layers "
            f"{source_X.shape[1]}"
        )

    print("[source] computing source-only OOF head accuracy...", flush=True)
    source_oof_acc = source_oof_head_accuracy(
        source_X,
        source_y,
        a.cv_folds,
        a.cv_seed,
    )
    source_center, source_dirs = fit_source_codebook(source_X, source_y)

    head_rows = []
    selected_union = set()
    max_budget = max(head_budgets)

    for H in head_layers:
        ranked = sorted(
            [
                (float(source_oof_acc[H, h]), h)
                for h in range(source_oof_acc.shape[1])
                if float(source_oof_acc[H, h]) >= float(a.min_head_acc)
            ],
            reverse=True,
        )
        for rank, (acc, h) in enumerate(ranked, 1):
            head_rows.append(
                {
                    "head_layer": H,
                    "head": h,
                    "head_name": hname(H, h),
                    "aligned_source_layer": H - a.head_input_offset,
                    "source_oof_accuracy": acc,
                    "rank_within_layer": rank,
                    "in_max_budget": rank <= max_budget,
                }
            )
            if rank <= max_budget:
                selected_union.add((H, h))

    head_df = pd.DataFrame(head_rows)
    head_df.to_csv(outdir / "source_oof_head_accuracy.csv", index=False)

    sel_head_df = head_df[head_df["in_max_budget"]].copy()
    sel_head_df.to_csv(outdir / "selected_heads_by_layer.csv", index=False)

    if not len(sel_head_df):
        raise RuntimeError("No heads survive --min-head-acc")

    # Spatial basis per head, learned entirely on Synthetic source.
    basis = {}
    for H in head_layers:
        for h in range(source_X.shape[2]):
            basis[(H, h)] = spatial_basis(source_dirs[H, h])

    # -------------------------------------------------------------------------
    # 2) Load oracle bank; GT is used only here as recovery target.
    # -------------------------------------------------------------------------
    top_bank, core_bank = load_oracle_bank(a.oracle_bank_dir)
    top_lookup = build_target_lookup(top_bank)
    core_lookup = build_target_lookup(core_bank)

    bank_sids = sorted(set(top_lookup) & set(core_lookup))
    if a.require_n and len(bank_sids) != int(a.require_n):
        raise RuntimeError(
            f"Expected {a.require_n} bank SIDs, got {len(bank_sids)}"
        )
    if a.max_samples and a.max_samples > 0:
        bank_sids = bank_sids[: int(a.max_samples)]

    # -------------------------------------------------------------------------
    # 3) Build COCO metadata for exactly those SIDs.
    # -------------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta_by_sid = {}
    for sid in bank_sids:
        if sid not in rec_by_sid or sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta_by_sid[sid] = {
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        }

    missing_meta = sorted(set(bank_sids) - set(meta_by_sid))
    if missing_meta:
        raise RuntimeError(
            f"Missing COCO metadata for {len(missing_meta)} bank SIDs; "
            f"first={missing_meta[:20]}"
        )

    # -------------------------------------------------------------------------
    # 4) Forward only samples not already cached.
    # -------------------------------------------------------------------------
    need_forward = [
        sid for sid in bank_sids
        if not cache_path(cache_dir, sid).exists()
    ]

    model = processor = None
    if need_forward:
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

        print(
            f"[model] loading {spec.repo_id}; "
            f"real-image forwards needed={len(need_forward)}",
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
        n_heads = int(cfg.num_attention_heads)
        n_kv_heads = int(
            getattr(cfg, "num_key_value_heads", n_heads)
        )
        hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            hidden_size = int(
                resolve_o_proj(resolve_attn(decoder_layers[0])).in_features
            )
        _CAPTURE_SHAPE = {
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "hidden_size": hidden_size,
        }

        if source_X.shape[2] != n_heads:
            raise RuntimeError(
                f"Source cache heads={source_X.shape[2]} but model heads={n_heads}"
            )
        if source_X.shape[3] != hidden_size // n_heads:
            raise RuntimeError(
                f"Source head_dim={source_X.shape[3]} but model head_dim="
                f"{hidden_size // n_heads}"
            )

        print(
            f"[model] decoder={decoder_path} | layers={len(decoder_layers)} "
            f"| heads={n_heads} | selected head layers={head_layers}"
        )

        try:
            for sid in tqdm(need_forward, desc="REAL spatial-head token scores"):
                m = meta_by_sid[sid]
                image = None
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

                    # Text only, exclude prompt-last because oracle mediation
                    # bank also excludes the source last token.
                    text_positions = [
                        p for p in range(min(len(ids), len(cats), len(toks)) - 1)
                        if dyn.broad_category(cats[p]) != "visual"
                    ]
                    if not text_positions:
                        raise RuntimeError("No text positions")

                    cap = run_real_capture(
                        model,
                        decoder_layers,
                        rb,
                        head_layers,
                    )

                    nH = len(head_layers)
                    nHeads = cap["n_heads"]
                    nP = len(text_positions)

                    score_arrays = {
                        method: np.full(
                            (nH, nHeads, nP),
                            np.nan,
                            dtype=np.float32,
                        )
                        for method in SCORE_METHODS
                    }
                    spatialness = np.full(
                        (nH, nHeads), np.nan, dtype=np.float32
                    )
                    recon_cos = np.full(
                        (nH, nHeads), np.nan, dtype=np.float32
                    )
                    recon_relerr = np.full(
                        (nH, nHeads), np.nan, dtype=np.float32
                    )

                    for li, H in enumerate(head_layers):
                        for h in range(nHeads):
                            pos, sc, diag = token_scores_one_head(
                                cap,
                                H,
                                h,
                                spos,
                                rpos,
                                a.direction_pool,
                                basis[(H, h)],
                                text_positions,
                            )
                            if pos != text_positions:
                                raise RuntimeError(
                                    f"Token-position mismatch sid={sid} H={H} h={h}"
                                )
                            for method in SCORE_METHODS:
                                score_arrays[method][li, h, :] = sc[method]
                            spatialness[li, h] = diag["spatialness"]
                            recon_cos[li, h] = diag["reconstruction_cosine"]
                            recon_relerr[li, h] = diag[
                                "reconstruction_relative_error"
                            ]

                    token_ids = [int(ids[p]) for p in text_positions]
                    token_strs = [
                        str(toks[p]).replace("\n", "\\n")
                        for p in text_positions
                    ]
                    token_cats = [str(cats[p]) for p in text_positions]
                    token_broad = [
                        dyn.broad_category(cats[p])
                        for p in text_positions
                    ]

                    save_sample_cache(
                        cache_path(cache_dir, sid),
                        sid,
                        head_layers,
                        text_positions,
                        token_ids,
                        token_strs,
                        token_cats,
                        token_broad,
                        score_arrays,
                        spatialness,
                        recon_cos,
                        recon_relerr,
                    )

                    del rb, cap
                finally:
                    if image is not None:
                        with contextlib.suppress(Exception):
                            image.close()
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

    else:
        print(
            f"[cache] all {len(bank_sids)} sample score caches already exist; "
            "no model forward needed."
        )

    # -------------------------------------------------------------------------
    # 5) Post-process cached scores: per-head relation and consensus recovery.
    # -------------------------------------------------------------------------
    per_head_rows = []
    consensus_rows = []
    selected_rows = []

    max_select_k = max(select_ks)

    # Precompute head selections for all requested budgets.
    heads_by_budget = {}
    for budget in head_budgets:
        heads_by_budget[budget] = selected_heads_per_layer(
            head_layers,
            source_oof_acc,
            budget,
            a.min_head_acc,
        )

    for sid in tqdm(bank_sids, desc="RECOVERY post-processing"):
        sample = load_sample_cache(cache_path(cache_dir, sid))

        targets = {
            "top10": top_lookup[sid],
            "core50": core_lookup[sid],
        }

        # Individual-head analysis.
        per_head_rows.extend(
            {
                "sid": sid,
                **r,
            }
            for r in per_head_recovery(
                sample,
                source_oof_acc,
                targets,
                a.per_head_k,
                a.head_input_offset,
            )
        )

        for method in SCORE_METHODS:
            for budget in head_budgets:
                sel_by_layer = heads_by_budget[budget]

                for weight_mode in weight_modes:
                    state_rows = consensus_state_scores(
                        sample,
                        method,
                        sel_by_layer,
                        source_oof_acc,
                        weight_mode,
                        a.head_input_offset,
                    )
                    ranked = global_unique_rank(state_rows)
                    ranked = ranked[:max_select_k]

                    # Save the max-K prefix once; smaller K are prefixes.
                    for rr in ranked:
                        p = int(rr["position"])
                        try:
                            j = int(np.where(sample["positions"] == p)[0][0])
                            tok = str(sample["tokens"][j])
                            cat = str(sample["categories"][j])
                            broad = str(sample["broad_categories"][j])
                        except Exception:
                            tok, cat, broad = "", "", ""

                        selected_rows.append(
                            {
                                "sid": sid,
                                "method": method,
                                "head_budget_per_layer": budget,
                                "weight_mode": weight_mode,
                                "spatial_rank": int(rr["spatial_rank"]),
                                "score": float(rr["score"]),
                                "source_layer": int(rr["source_layer"]),
                                "head_layer": int(rr["head_layer"]),
                                "position": p,
                                "token": tok,
                                "category": cat,
                                "broad_category": broad,
                                "heads": rr["heads"],
                                "in_oracle_top10_exact": (
                                    (int(rr["source_layer"]), p)
                                    in top_lookup[sid]["text_states"]
                                ),
                                "in_oracle_core50_exact": (
                                    (int(rr["source_layer"]), p)
                                    in core_lookup[sid]["text_states"]
                                ),
                                "in_oracle_top10_position": (
                                    p in top_lookup[sid]["text_positions"]
                                ),
                                "in_oracle_core50_position": (
                                    p in core_lookup[sid]["text_positions"]
                                ),
                            }
                        )

                    for K in select_ks:
                        chosen = ranked[:K]

                        for target_name, target in targets.items():
                            met = recovery_metrics(chosen, target)
                            rnd = random_expectation(
                                chosen,
                                target,
                                sample["positions"],
                            )

                            exact_enrich = (
                                met["exact_hits"] / rnd["expected_exact_hits"]
                                if np.isfinite(rnd["expected_exact_hits"])
                                and rnd["expected_exact_hits"] > 0
                                else np.nan
                            )
                            pos_enrich = (
                                met["position_hits"]
                                / rnd["expected_position_hits"]
                                if np.isfinite(rnd["expected_position_hits"])
                                and rnd["expected_position_hits"] > 0
                                else np.nan
                            )

                            consensus_rows.append(
                                {
                                    "sid": sid,
                                    "gt": meta_by_sid[sid]["gt"],
                                    "method": method,
                                    "head_budget_per_layer": budget,
                                    "weight_mode": weight_mode,
                                    "K": K,
                                    "target": target_name,
                                    **met,
                                    **rnd,
                                    "exact_enrichment_over_random": exact_enrich,
                                    "position_enrichment_over_random": pos_enrich,
                                }
                            )

    per_head = pd.DataFrame(per_head_rows)
    per_sample = pd.DataFrame(consensus_rows)
    selected_df = pd.DataFrame(selected_rows)

    per_head.to_csv(
        outdir / "per_head_recovery_per_sample.csv",
        index=False,
    )
    per_sample.to_csv(
        outdir / "consensus_recovery_per_sample.csv",
        index=False,
    )
    selected_df.to_csv(
        outdir / "spatial_selected_topmax.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 6) Aggregate per-head recovery and accuracy-vs-recovery relationship.
    # -------------------------------------------------------------------------
    head_summary_rows = []
    if len(per_head):
        keys = [
            "head_name",
            "head_layer",
            "head",
            "aligned_source_layer",
            "source_oof_accuracy",
            "method",
            "target",
        ]
        for key, g in per_head.groupby(keys, dropna=False):
            (
                name, H, h, S, acc, method, target
            ) = key
            head_summary_rows.append(
                {
                    "head_name": name,
                    "head_layer": H,
                    "head": h,
                    "aligned_source_layer": S,
                    "source_oof_accuracy": acc,
                    "method": method,
                    "target": target,
                    "N_samples_with_target_at_layer": int(g["sid"].nunique()),
                    "mean_recall": safe_mean(g["recall"]),
                    "mean_enrichment_over_random": safe_mean(
                        g["enrichment_over_random"]
                    ),
                    "mean_hits": safe_mean(g["hits"]),
                    "mean_spatialness": safe_mean(g["spatialness"]),
                    "mean_reconstruction_cosine": safe_mean(
                        g["reconstruction_cosine"]
                    ),
                }
            )

    head_summary = pd.DataFrame(head_summary_rows)
    if len(head_summary):
        head_summary = head_summary.sort_values(
            [
                "method",
                "target",
                "mean_recall",
                "source_oof_accuracy",
            ],
            ascending=[True, True, False, False],
        )
    head_summary.to_csv(
        outdir / "per_head_recovery_summary.csv",
        index=False,
    )

    corr_rows = []
    if len(head_summary):
        for (method, target), g in head_summary.groupby(
            ["method", "target"]
        ):
            corr_rows.append(
                {
                    "method": method,
                    "target": target,
                    "N_heads": len(g),
                    "pearson_oofAcc_vs_recall": safe_corr(
                        g["source_oof_accuracy"],
                        g["mean_recall"],
                    ),
                    "spearman_oofAcc_vs_recall": safe_spearman(
                        g["source_oof_accuracy"],
                        g["mean_recall"],
                    ),
                    "pearson_oofAcc_vs_enrichment": safe_corr(
                        g["source_oof_accuracy"],
                        g["mean_enrichment_over_random"],
                    ),
                    "spearman_oofAcc_vs_enrichment": safe_spearman(
                        g["source_oof_accuracy"],
                        g["mean_enrichment_over_random"],
                    ),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        outdir / "head_accuracy_recovery_correlation.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7) Aggregate consensus recovery.
    # -------------------------------------------------------------------------
    summary_rows = []
    if len(per_sample):
        group_cols = [
            "method",
            "head_budget_per_layer",
            "weight_mode",
            "K",
            "target",
        ]
        for key, g in per_sample.groupby(group_cols):
            method, budget, weight_mode, K, target = key

            total_hits = int(g["exact_hits"].sum())
            total_target = int(g["target_text_state_N"].sum())
            total_pos_hits = int(g["position_hits"].sum())
            total_pos_target = int(g["target_text_position_N"].sum())
            total_exp_exact = float(g["expected_exact_hits"].sum())
            total_exp_pos = float(g["expected_position_hits"].sum())

            summary_rows.append(
                {
                    "method": method,
                    "head_budget_per_layer": budget,
                    "weight_mode": weight_mode,
                    "K": K,
                    "target": target,
                    "N_samples": int(g["sid"].nunique()),
                    "micro_exact_recall": safe_div(
                        total_hits, total_target
                    ),
                    "macro_exact_recall": safe_mean(g["exact_recall"]),
                    "micro_position_recall": safe_div(
                        total_pos_hits, total_pos_target
                    ),
                    "macro_position_recall": safe_mean(
                        g["position_recall"]
                    ),
                    "micro_exact_precision": safe_div(
                        total_hits, int(g["selected_K"].sum())
                    ),
                    "micro_position_precision": safe_div(
                        total_pos_hits, int(g["selected_K"].sum())
                    ),
                    "exact_enrichment_over_random": (
                        total_hits / total_exp_exact
                        if total_exp_exact > 0 else np.nan
                    ),
                    "position_enrichment_over_random": (
                        total_pos_hits / total_exp_pos
                        if total_exp_pos > 0 else np.nan
                    ),
                    "mean_exact_hits": safe_mean(g["exact_hits"]),
                    "mean_position_hits": safe_mean(g["position_hits"]),
                    "mean_target_text_states": safe_mean(
                        g["target_text_state_N"]
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary = summary.sort_values(
            [
                "target",
                "micro_exact_recall",
                "micro_position_recall",
                "exact_enrichment_over_random",
            ],
            ascending=[True, False, False, False],
        )
    summary.to_csv(
        outdir / "consensus_recovery_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 8) Text report.
    # -------------------------------------------------------------------------
    lines = []
    lines.append("=" * 172)
    lines.append("SPATIAL HEAD -> ORACLE TEXT CAUSAL TOKEN RECOVERY")
    lines.append("=" * 172)
    lines.append(
        f"N={len(bank_sids)} | model={a.model} | REAL IMAGE ONLY | "
        "no gray / no writer / no gradient / no GT relation in selector"
    )
    lines.append(
        f"head layers={head_layers} -> source layers="
        f"{[H-a.head_input_offset for H in head_layers]}"
    )
    lines.append(
        f"head budgets/layer={head_budgets} | selected global-unique K={select_ks}"
    )
    lines.append("")

    lines.append("SOURCE-ONLY HIGH-ACCURACY HEADS")
    lines.append("-" * 172)
    for H in head_layers:
        g = head_df[
            (head_df["head_layer"] == H)
            & (head_df["rank_within_layer"] <= max_budget)
        ].sort_values("rank_within_layer")
        if not len(g):
            continue
        vals = ", ".join(
            f"{r.head_name}={r.source_oof_accuracy:.4f}"
            for r in g.itertuples()
        )
        lines.append(
            f"head L{H:02d} -> state L{H-a.head_input_offset:02d}: {vals}"
        )
    lines.append("")

    lines.append("HEAD ACCURACY vs CAUSAL-TOKEN RECOVERY")
    lines.append("-" * 172)
    if len(corr_df):
        lines.append(
            corr_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    else:
        lines.append("(no rows)")
    lines.append("")

    lines.append("BEST CONSENSUS CONFIGS: CORE50 TEXT EXACT-STATE")
    lines.append("-" * 172)
    if len(summary):
        q = summary[summary["target"] == "core50"].head(30)
        show = [
            "method",
            "head_budget_per_layer",
            "weight_mode",
            "K",
            "micro_exact_recall",
            "micro_position_recall",
            "exact_enrichment_over_random",
            "position_enrichment_over_random",
        ]
        lines.append(
            q[show].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    lines.append("")

    lines.append("BEST CONSENSUS CONFIGS: TOP10 TEXT EXACT-STATE")
    lines.append("-" * 172)
    if len(summary):
        q = summary[summary["target"] == "top10"].head(30)
        show = [
            "method",
            "head_budget_per_layer",
            "weight_mode",
            "K",
            "micro_exact_recall",
            "micro_position_recall",
            "exact_enrichment_over_random",
            "position_enrichment_over_random",
        ]
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
        "source_cache": str(source_path),
        "oracle_bank_dir": str(Path(a.oracle_bank_dir)),
        "N_samples": len(bank_sids),
        "head_layers": head_layers,
        "source_layers": [
            H - int(a.head_input_offset) for H in head_layers
        ],
        "head_budgets_per_layer": head_budgets,
        "select_ks": select_ks,
        "weight_modes": weight_modes,
        "source_cv_folds": int(a.cv_folds),
        "source_cv_seed": int(a.cv_seed),
        "direction_pool": a.direction_pool,
        "selector_inputs": (
            "real-image attention A, value V, and source-only Synthetic spatial "
            "head accuracy/subspace only"
        ),
        "forbidden_from_selector": [
            "gray image",
            "Real-Gray token displacement",
            "writer vector",
            "writer gradient",
            "GT relation",
            "target answer logit",
        ],
        "score_definitions": {
            "attention_abs": "|A(sub,p)-A(ref,p)|",
            "av_norm": "||(A(sub,p)-A(ref,p))*V(p)||",
            "self_align": "<c_hp, normalize(sum_p c_hp)>",
            "spatial_self_align": (
                "<P_h c_hp, normalize(P_h sum_p c_hp)>, where P_h is "
                "Synthetic-source spatial subspace"
            ),
        },
        "layer_mapping": (
            "attention head layer H consumes decoder block-output state H-1"
        ),
        "candidate_domain": "text only; source last token excluded",
        "global_unique": (
            "same token position across layers keeps the source layer with the "
            "largest spatial consensus score"
        ),
        "oracle_usage": "evaluation target only",
        "sample_score_cache": str(cache_dir),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Saved:", outdir)


if __name__ == "__main__":
    main()
