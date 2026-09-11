#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_four_k7_vs_direction_head_sources_v1.py

Purpose
-------
Validate the ORIGINAL cross-mechanism convergence:

  decision-side causal ranking
      M_{r,L,p} = (h_real - h_gray)^T dJ_r/dh_{L,p}

  versus

  Direction-Head source-token spatial ranking
      c^RG_{h,p}
        = [(A^R_sub,p - A^R_ref,p) V^R_p]
        - [(A^G_sub,p - A^G_ref,p) V^G_p]

      S^dir_{h,p}(r) = < c^RG_{h,p}, d^h_r >

with the important layer alignment

      causal block-output L  --->  Direction Head at L+1.

This script DOES NOT approximate the Direction-Head side with generic
last-token attention.  It reconstructs the source-token contribution to the
subject-reference head output, projects it onto the head's relation direction,
and independently ranks source tokens.

It then asks:

1) For known / all Direction Heads:
   if a causal Rank-k state is at layer L, where does its token position rank
   among spatial sources of heads at L+1?

2) Rank-wise convergence:
   do causal Rank1/3/5/7 states have higher spatial-source percentile than
   weaker causal states?

3) Matched-budget overlap:
   if Top-K contains m states at source layer L, how much do those m positions
   overlap the Direction Head's own spatial Top-m sources at L+1?

4) Head-level validation:
   across heads in the SAME head layer, is Direction-Head decoding accuracy
   positively associated with causal/spatial-source convergence?

5) Non-oracle candidate selection:
   every sample has K^left, K^right, K^on, K^under.
   For each candidate r, score how well K^r agrees with the independently
   computed Direction-Head source ranking under relation hypothesis r.
   No test GT is used to compute these four candidate scores.

Calibration / leakage note
--------------------------
Relation directions d^h_r are fit from relation_vectors.npz produced by
analyze_coco_head_object_residual_direction_probe_v1.py.  By default this
script EXCLUDES the current N evaluation SIDs from that direction fit.

Head selection for the candidate selector uses head_results.csv.  If that CSV
was produced from a probe evaluation containing these same SIDs, the identity
of "top heads" is not strictly OOF.  The per-sample candidate score itself is
still GT-free.  For a publication-grade selector, fit/select heads on a
disjoint calibration set.

Expected existing files
-----------------------
FOURWAY_DIR/
  all_four_writer_mediation.pkl.gz
  per_sample_direction_scores.csv

DIRECTION_DIR/
  relation_vectors.npz
  head_results.csv

Repo helpers
------------
  analyze_coco_centroid_generation_step1_v4.py
  analyze_coco_head_object_residual_direction_probe_v1.py

Typical run
-----------
CUDA_VISIBLE_DEVICES=0 python -u validate_four_k7_vs_direction_head_sources_v1.py \
  --fourway-dir output/qwen3b_four_writer_k7_N80 \
  --direction-dir output/qwen3b_head_object_residual_direction \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --ks 1,3,5,7,14,24,36 \
  --top-heads-per-layer 3 \
  --focus-heads L26H03,L23H01,L23H05 \
  --max-eval-samples 80 \
  --output-dir output/qwen3b_k7_vs_direction_source_rank_N80 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import analyze_coco_head_object_residual_direction_probe_v1 as hprobe


REL = ("left", "right", "on", "under")
REL_TO_PROBE = {"left": "left", "right": "right", "on": "above", "under": "below"}
PROBE_TO_REL = {"left": "left", "right": "right", "above": "on", "below": "under",
                "on": "on", "under": "under"}
EPS = 1e-12


# =============================================================================
# Generic utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--ks", default="1,3,5,7,14,24,36")
    p.add_argument("--top-heads-per-layer", type=int, default=3)
    p.add_argument("--focus-heads", default="L26H03,L23H01,L23H05")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
        help="Use eager for exact manual attention reconstruction.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-eval-samples", type=int, default=80)
    p.add_argument(
        "--direction-fit",
        default="exclude_eval",
        choices=["exclude_eval", "all"],
        help="Fit per-head relation directions with or without evaluation SIDs.",
    )
    p.add_argument(
        "--selector-k",
        type=int,
        default=7,
        help="Primary K used for the four-way candidate selector report.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return sorted({int(x.strip().upper().replace("L", ""))
                   for x in str(s).split(",") if x.strip()})


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    mp = {
        "left": "left", "left_of": "left", "l": "left",
        "right": "right", "right_of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under", "bottom": "under",
    }
    return mp.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def parse_head_name(s):
    m = re.fullmatch(r"\s*L(\d+)H(\d+)\s*", str(s), flags=re.I)
    if not m:
        raise ValueError(f"Bad head name: {s!r}; expected like L26H03")
    return int(m.group(1)), int(m.group(2))


def head_name(L, H):
    return f"L{int(L)}H{int(H):02d}"


def safe_mean(xs):
    x = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def safe_std(xs):
    x = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan


def safe_div(a, b):
    return float(a / b) if abs(float(b)) > EPS else np.nan


def spearman(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce")
    b = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def make_gray_image(img, value):
    v = int(np.clip(value, 0, 255))
    return Image.new("RGB", img.size, (v, v, v))


def write_csv(path, df_or_rows):
    p = Path(path)
    if isinstance(df_or_rows, pd.DataFrame):
        df_or_rows.to_csv(p, index=False)
    else:
        pd.DataFrame(df_or_rows).to_csv(p, index=False)


# =============================================================================
# Causal rankings
# =============================================================================

def global_unique_rank(g, max_k):
    q = g.copy()
    q["mediation"] = pd.to_numeric(q["mediation"], errors="coerce")
    q = q[np.isfinite(q["mediation"]) & (q["mediation"] > 0)].copy()
    if not len(q):
        return q

    # Same token position can only be selected once; keep layer with largest M.
    idx = q.groupby("position")["mediation"].idxmax()
    q = q.loc[idx].sort_values("mediation", ascending=False).head(max_k).copy()
    q["causal_rank"] = np.arange(1, len(q) + 1)
    return q


def build_all_causal_ranks(med, max_k):
    chunks = []
    for (sid, cand), g in med.groupby(["sid", "candidate_relation"]):
        q = global_unique_rank(g, max_k)
        if len(q):
            chunks.append(q)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# =============================================================================
# Fit relation directions d^h_r from independent head-probe cache
# =============================================================================

def load_direction_probe(direction_dir, eval_sids, mode):
    ddir = Path(direction_dir)
    npz_path = ddir / "relation_vectors.npz"
    csv_path = ddir / "head_results.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    z = np.load(npz_path, allow_pickle=True)
    required = {"sample_index", "relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{npz_path} missing arrays: {sorted(missing)}")

    sids = np.asarray(z["sample_index"]).astype(int)
    labels = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)
    residual = np.asarray(z["residual"], dtype=np.float32)
    # [N, layers, heads, head_dim]
    if residual.ndim != 4:
        raise RuntimeError(f"Expected residual [N,L,H,D], got {residual.shape}")

    if mode == "exclude_eval":
        keep = ~np.isin(sids, np.asarray(sorted(eval_sids), dtype=int))
    else:
        keep = np.ones(len(sids), dtype=bool)

    # Keep only valid four relations.
    keep &= np.isin(labels, np.asarray(REL, dtype=object))
    if keep.sum() < 8:
        raise RuntimeError(
            f"Only {int(keep.sum())} direction-fit samples remain; cannot fit directions."
        )

    X = residual[keep]
    y = labels[keep]
    L, H, D = X.shape[1:]

    dirs = np.zeros((L, H, len(REL), D), dtype=np.float32)
    counts = {}
    for li in range(L):
        for hi in range(H):
            x = X[:, li, hi, :]
            center = x.mean(axis=0)
            for ri, rel in enumerate(REL):
                m = (y == rel)
                counts[rel] = int(m.sum())
                if not m.any():
                    raise RuntimeError(f"No calibration samples for relation={rel}")
                d = (x[m] - center).mean(axis=0)
                dirs[li, hi, ri] = normalize(d)

    heads = pd.read_csv(csv_path)
    heads["layer"] = pd.to_numeric(heads["layer"], errors="raise").astype(int)
    heads["head"] = pd.to_numeric(heads["head"], errors="raise").astype(int)

    print(
        f"Direction fit: mode={mode} N={int(keep.sum())} "
        f"counts={{" + ", ".join(f"{r}:{int((y==r).sum())}" for r in REL) + "}}"
    )
    return dirs, heads, {
        "npz": str(npz_path),
        "csv": str(csv_path),
        "fit_n": int(keep.sum()),
        "fit_counts": {r: int((y == r).sum()) for r in REL},
    }


# =============================================================================
# Exact attention/value source contribution capture
# =============================================================================

def rotate_half_np(x):
    d = x.shape[-1]
    if d % 2:
        raise RuntimeError(f"Odd head_dim={d}; cannot rotate_half")
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


class LayerQKVRecorder:
    """
    Capture just enough to reconstruct attention rows for subject/reference
    queries at selected decoder layers.

    q_proj: keep only query positions
    k_proj/v_proj: keep all source positions, KV heads only
    position_embeddings: keep cos/sin
    attention_mask: keep only query rows
    """

    def __init__(self, layer, query_positions):
        self.layer = layer
        self.attn = hprobe.resolve_self_attention(layer)
        self.query_positions = [int(x) for x in query_positions]
        self.handles = []
        self.q = None
        self.k = None
        self.v = None
        self.cos = None
        self.sin = None
        self.mask = None
        self.mask_is_bool = False

        for name in ("q_proj", "k_proj", "v_proj"):
            if not hasattr(self.attn, name):
                raise RuntimeError(
                    f"{type(self.attn).__name__} lacks {name}; this script targets Qwen2/2.5-style attention."
                )

        self.handles.append(
            self.attn.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self.handles.append(self.attn.q_proj.register_forward_hook(self._q_hook))
        self.handles.append(self.attn.k_proj.register_forward_hook(self._k_hook))
        self.handles.append(self.attn.v_proj.register_forward_hook(self._v_hook))

    def _pre(self, module, args, kwargs):
        pe = kwargs.get("position_embeddings", None)
        if pe is None or len(pe) != 2:
            raise RuntimeError("Attention position_embeddings not available in forward kwargs")
        cos, sin = pe
        # [B,S,D]
        self.cos = cos[0].detach().float().cpu().numpy()
        self.sin = sin[0].detach().float().cpu().numpy()

        am = kwargs.get("attention_mask", None)
        self.mask = None
        self.mask_is_bool = False
        if torch.is_tensor(am):
            self.mask_is_bool = (am.dtype == torch.bool)
            # Common Qwen eager mask: [B,1,Q,K] or [B,H,Q,K]
            if am.ndim == 4:
                self.mask = (
                    am[0, :, self.query_positions, :]
                    .detach().float().cpu().numpy()
                )  # [M,Q,S]
            elif am.ndim == 3:
                self.mask = (
                    am[0, self.query_positions, :]
                    .detach().float().cpu().numpy()
                )  # [Q,S]
            elif am.ndim == 2:
                # Usually padding mask, not full causal mask.
                self.mask = am[0].detach().float().cpu().numpy()[None, :]

    def _q_hook(self, module, inp, out):
        # [B,S,H*D] -> only subject/ref queries
        self.q = out[0, self.query_positions].detach().float().cpu().numpy()

    def _k_hook(self, module, inp, out):
        self.k = out[0].detach().float().cpu().numpy()

    def _v_hook(self, module, inp, out):
        self.v = out[0].detach().float().cpu().numpy()

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def capture_condition_contributions(
    model,
    decoder_layers,
    batch,
    target_head_layers,
    subject_positions,
    reference_positions,
):
    qpos = sorted(set(map(int, subject_positions + reference_positions)))
    qlocal = {p: i for i, p in enumerate(qpos)}
    sub_local = [qlocal[p] for p in subject_positions if p in qlocal]
    ref_local = [qlocal[p] for p in reference_positions if p in qlocal]
    if not sub_local or not ref_local:
        raise RuntimeError("Could not map subject/reference query positions")

    recs = {}
    try:
        for HL in target_head_layers:
            recs[HL] = LayerQKVRecorder(decoder_layers[HL], qpos)

        with torch.inference_mode():
            kw = dict(batch)
            kw["use_cache"] = False
            kw["output_attentions"] = False
            kw["output_hidden_states"] = False
            _ = model(**kw)

        out = {}
        for HL, rec in recs.items():
            attn = rec.attn
            if rec.q is None or rec.k is None or rec.v is None:
                raise RuntimeError(f"L{HL}: missing q/k/v capture")
            if rec.cos is None or rec.sin is None:
                raise RuntimeError(f"L{HL}: missing position embeddings")

            H = int(getattr(attn, "num_heads", getattr(attn, "num_attention_heads", 0)))
            Hkv = int(getattr(attn, "num_key_value_heads", H))
            if H <= 0:
                raise RuntimeError(f"L{HL}: cannot infer num_heads")

            D = int(getattr(attn, "head_dim", rec.q.shape[-1] // H))
            if rec.q.shape[-1] != H * D:
                raise RuntimeError(
                    f"L{HL}: q width {rec.q.shape[-1]} != H*D={H*D}"
                )
            if rec.k.shape[-1] != Hkv * D or rec.v.shape[-1] != Hkv * D:
                raise RuntimeError(
                    f"L{HL}: k/v widths incompatible with Hkv={Hkv}, D={D}"
                )

            S = min(rec.k.shape[0], rec.v.shape[0], rec.cos.shape[0], rec.sin.shape[0])
            Q = len(qpos)

            q = rec.q.reshape(Q, H, D)
            k = rec.k[:S].reshape(S, Hkv, D)
            v = rec.v[:S].reshape(S, Hkv, D)

            cos_q = rec.cos[np.asarray(qpos), :D]
            sin_q = rec.sin[np.asarray(qpos), :D]
            cos_k = rec.cos[:S, :D]
            sin_k = rec.sin[:S, :D]

            qrot = q * cos_q[:, None, :] + rotate_half_np(q) * sin_q[:, None, :]
            krot = k * cos_k[:, None, :] + rotate_half_np(k) * sin_k[:, None, :]

            groups = H // Hkv
            if H % Hkv:
                raise RuntimeError(f"L{HL}: H={H} not divisible by Hkv={Hkv}")
            krep = np.repeat(krot, groups, axis=1)  # [S,H,D]
            vrep = np.repeat(v, groups, axis=1)     # [S,H,D]

            # [H,Q,S]
            scores = np.einsum("qhd,shd->hqs", qrot, krep, optimize=True)
            scores *= float(D) ** -0.5

            if rec.mask is not None:
                m = rec.mask
                if m.ndim == 3:
                    # [M,Q,S]
                    m = m[:, :, :S]
                    if m.shape[0] == 1:
                        scores += m[0][None, :, :]
                    elif m.shape[0] == H:
                        scores += m
                    else:
                        # Conservative fallback to first mask head.
                        scores += m[0][None, :, :]
                elif m.ndim == 2 and m.shape[0] == Q:
                    scores += m[:, :S][None, :, :]
                elif m.ndim == 2 and m.shape[0] == 1:
                    # Padding-like mask. If values are binary, convert 0 to -inf.
                    mm = m[:, :S]
                    if np.all(np.isin(np.unique(mm), [0.0, 1.0])):
                        bad = (mm <= 0)
                        scores[:, :, bad[0]] = -1e30
            else:
                # Exact causal fallback.
                for qi, abs_q in enumerate(qpos):
                    if abs_q + 1 < S:
                        scores[:, qi, abs_q + 1:] = -1e30

            A = softmax_np(scores, axis=-1)  # [H,Q,S]
            A_sub = A[:, sub_local, :].mean(axis=1)  # [H,S]
            A_ref = A[:, ref_local, :].mean(axis=1)  # [H,S]
            deltaA = A_sub - A_ref

            # [H,S,D]: exact additive source contribution to z_sub-z_ref pre-W_O.
            contrib = deltaA[:, :, None] * np.transpose(vrep, (1, 0, 2))

            # Visible source universe: any token causally available to at least one
            # subject/reference query. Tokens after max query cannot contribute.
            universe_end = min(S, max(qpos) + 1)

            out[HL] = {
                "contrib": contrib[:, :universe_end].astype(np.float32),
                "deltaA": deltaA[:, :universe_end].astype(np.float32),
                "seq_len": int(S),
                "universe_end": int(universe_end),
                "num_heads": int(H),
                "head_dim": int(D),
            }
        return out
    finally:
        for rec in recs.values():
            rec.close()


# =============================================================================
# Spatial score + percentile
# =============================================================================

def percentile_arrays(scores):
    """
    scores: [H,R,U]. Return ascending percentile in (0,1], so 1 = strongest.

    Ties are broken by argsort order; exact ties are rare for projected
    contribution scores and do not affect the intended ranking diagnostic.
    """
    H, R, U = scores.shape
    order = np.argsort(scores, axis=-1)
    ranks = np.empty_like(order)
    base_rank = np.broadcast_to(np.arange(U, dtype=np.int64), (H, R, U))
    np.put_along_axis(ranks, order, base_rank, axis=-1)
    return (ranks.astype(np.float32) + 1.0) / float(U)


def compute_spatial_scores(real_caps, gray_caps, directions, target_layers):
    """
    For each head layer HL:
      c_RG [H,U,D] = c_real - c_gray
      score [H,R,U] = dot(c_RG, d^h_r)
      percentile [H,R,U]
    """
    out = {}
    for HL in target_layers:
        if HL not in real_caps or HL not in gray_caps:
            continue

        cr = real_caps[HL]["contrib"]
        cg = gray_caps[HL]["contrib"]
        U = min(cr.shape[1], cg.shape[1])
        H = min(cr.shape[0], cg.shape[0], directions.shape[1])
        D = min(cr.shape[2], cg.shape[2], directions.shape[3])

        c = cr[:H, :U, :D] - cg[:H, :U, :D]
        dirs = directions[HL, :H, :, :D]  # [H,R,D]
        # [H,R,U]
        scores = np.einsum("hud,hrd->hru", c, dirs, optimize=True).astype(np.float32)
        pct = percentile_arrays(scores)

        out[HL] = {
            "scores": scores,
            "percentiles": pct,
            "universe": int(U),
        }
    return out


# =============================================================================
# Direction-head selection helpers
# =============================================================================

def relation_acc_col(rel):
    return {
        "left": "residual_left_accuracy",
        "right": "residual_right_accuracy",
        "on": "residual_on_accuracy",
        "under": "residual_under_accuracy",
    }[rel]


def select_heads(head_results, head_layer, n, rel=None):
    g = head_results[head_results["layer"] == int(head_layer)].copy()
    if not len(g):
        return []
    col = "residual_accuracy_mean"
    if rel is not None:
        rc = relation_acc_col(rel)
        if rc in g.columns:
            col = rc
    g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=[col]).sort_values(col, ascending=False)
    return g.head(int(n))["head"].astype(int).tolist()


def head_accuracy(head_results, L, H):
    g = head_results[
        (head_results.layer == int(L)) & (head_results["head"] == int(H))
    ]
    if not len(g):
        return np.nan
    return float(g["residual_accuracy_mean"].iloc[0])


# =============================================================================
# Per-sample analysis
# =============================================================================

def matched_overlap_for_head(scores_1d, causal_positions):
    """
    Causal budget m positions vs the head's own spatial Top-m sources.
    """
    pos = sorted(set(int(p) for p in causal_positions))
    m = len(pos)
    U = len(scores_1d)
    pos = [p for p in pos if 0 <= p < U]
    m = len(pos)
    if m == 0 or U == 0:
        return np.nan, np.nan, 0, U

    if m >= U:
        top = set(range(U))
    else:
        top = set(np.argpartition(scores_1d, -m)[-m:].tolist())
    hit = len(set(pos) & top)
    recall = hit / m
    expected = m / U
    return float(recall), float(expected), int(hit), int(U)


def analyze_sample(
    sid,
    ranked_sid,
    spatial,
    head_results,
    ks,
    top_heads_per_layer,
):
    """
    Returns:
      oracle_state_head_rows
      oracle_overlap_rows
      candidate_score_rows
    """
    state_rows = []
    overlap_rows = []
    cand_rows = []

    # -------------------------------------------------------------------------
    # A. Oracle relation: ALL heads, state-level percentile + matched overlap.
    # -------------------------------------------------------------------------
    gt = str(ranked_sid["gt"].iloc[0])
    baseline_prediction = str(ranked_sid["baseline_prediction"].iloc[0])
    baseline_correct = bool(ranked_sid["baseline_correct"].iloc[0])
    gt_idx = REL.index(gt)

    oracle = ranked_sid[ranked_sid["candidate_relation"] == gt].copy()

    for _, st in oracle.iterrows():
        L = int(st["source_layer"])
        HL = L + 1
        p = int(st["position"])
        if HL not in spatial:
            continue
        U = spatial[HL]["universe"]
        if not (0 <= p < U):
            continue

        pct = spatial[HL]["percentiles"][:, gt_idx, p]
        scores = spatial[HL]["scores"][:, gt_idx, p]
        for h in range(len(pct)):
            state_rows.append({
                "sid": sid,
                "gt": gt,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": baseline_correct,
                "causal_source_layer": L,
                "head_layer": HL,
                "head": h,
                "head_name": head_name(HL, h),
                "causal_position": p,
                "causal_rank": int(st["causal_rank"]),
                "causal_M": float(st["mediation"]),
                "spatial_percentile": float(pct[h]),
                "spatial_score": float(scores[h]),
                "top10_hit": bool(pct[h] >= 0.90),
                "top05_hit": bool(pct[h] >= 0.95),
                "top01_exact": bool(pct[h] >= (1.0 - 0.5 / max(U, 1))),
                "universe": U,
                "direction_accuracy": head_accuracy(head_results, HL, h),
            })

    for K in ks:
        ok = oracle[oracle["causal_rank"] <= K]
        for L, lg in ok.groupby("source_layer"):
            HL = int(L) + 1
            if HL not in spatial:
                continue
            positions = lg["position"].astype(int).tolist()
            U = spatial[HL]["universe"]
            for h in range(spatial[HL]["scores"].shape[0]):
                recall, expected, hit, _ = matched_overlap_for_head(
                    spatial[HL]["scores"][h, gt_idx, :U],
                    positions,
                )
                if not np.isfinite(recall):
                    continue
                overlap_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "baseline_prediction": baseline_prediction,
                    "baseline_correct": baseline_correct,
                    "K": int(K),
                    "causal_source_layer": int(L),
                    "head_layer": HL,
                    "head": h,
                    "head_name": head_name(HL, h),
                    "budget_m": len([p for p in positions if 0 <= p < U]),
                    "intersection": hit,
                    "matched_recall": recall,
                    "random_expected_recall": expected,
                    "enrichment_ratio": safe_div(recall, expected),
                    "direction_accuracy": head_accuracy(head_results, HL, h),
                })

    # -------------------------------------------------------------------------
    # B. Four candidate K sets, no GT in candidate scoring.
    # -------------------------------------------------------------------------
    for K in ks:
        for cand in REL:
            cg = ranked_sid[
                (ranked_sid["candidate_relation"] == cand)
                & (ranked_sid["causal_rank"] <= K)
            ].copy()
            if not len(cg):
                continue

            ridx = REL.index(cand)
            state_scores_overall = []
            state_scores_relheads = []
            state_ranks = []
            overlap_num_overall = 0.0
            overlap_den_overall = 0.0
            overlap_num_rel = 0.0
            overlap_den_rel = 0.0
            covered_states = 0

            # State-wise percentile compatibility.
            for _, st in cg.iterrows():
                L = int(st["source_layer"])
                HL = L + 1
                p = int(st["position"])
                rank = int(st["causal_rank"])
                if HL not in spatial:
                    continue
                U = spatial[HL]["universe"]
                if not (0 <= p < U):
                    continue

                all_pct = spatial[HL]["percentiles"][:, ridx, p]

                hs_overall = select_heads(
                    head_results, HL, top_heads_per_layer, rel=None
                )
                hs_rel = select_heads(
                    head_results, HL, top_heads_per_layer, rel=cand
                )
                hs_overall = [h for h in hs_overall if h < len(all_pct)]
                hs_rel = [h for h in hs_rel if h < len(all_pct)]

                if hs_overall:
                    state_scores_overall.append(float(np.mean(all_pct[hs_overall])))
                else:
                    state_scores_overall.append(np.nan)

                if hs_rel:
                    state_scores_relheads.append(float(np.mean(all_pct[hs_rel])))
                else:
                    state_scores_relheads.append(np.nan)

                state_ranks.append(rank)
                covered_states += 1

            # Matched-budget overlap compatibility per source layer.
            for L, lg in cg.groupby("source_layer"):
                HL = int(L) + 1
                if HL not in spatial:
                    continue
                U = spatial[HL]["universe"]
                positions = [
                    int(p) for p in lg["position"].tolist()
                    if 0 <= int(p) < U
                ]
                if not positions:
                    continue

                hs_overall = select_heads(
                    head_results, HL, top_heads_per_layer, rel=None
                )
                hs_rel = select_heads(
                    head_results, HL, top_heads_per_layer, rel=cand
                )

                vals = []
                for h in hs_overall:
                    if h >= spatial[HL]["scores"].shape[0]:
                        continue
                    rec, _, _, _ = matched_overlap_for_head(
                        spatial[HL]["scores"][h, ridx, :U], positions
                    )
                    if np.isfinite(rec):
                        vals.append(rec)
                if vals:
                    overlap_num_overall += float(np.mean(vals)) * len(positions)
                    overlap_den_overall += len(positions)

                vals = []
                for h in hs_rel:
                    if h >= spatial[HL]["scores"].shape[0]:
                        continue
                    rec, _, _, _ = matched_overlap_for_head(
                        spatial[HL]["scores"][h, ridx, :U], positions
                    )
                    if np.isfinite(rec):
                        vals.append(rec)
                if vals:
                    overlap_num_rel += float(np.mean(vals)) * len(positions)
                    overlap_den_rel += len(positions)

            def finite_pair(vals, ranks):
                vv = np.asarray(vals, float)
                rr = np.asarray(ranks, float)
                good = np.isfinite(vv) & np.isfinite(rr)
                return vv[good], rr[good]

            vo, ro = finite_pair(state_scores_overall, state_ranks)
            vr, rr = finite_pair(state_scores_relheads, state_ranks)

            def rwmean(v, r):
                if not len(v):
                    return np.nan
                w = 1.0 / np.maximum(r, 1.0)
                return float(np.average(v, weights=w))

            cand_rows.append({
                "sid": sid,
                "gt": gt,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": baseline_correct,
                "candidate_relation": cand,
                "K": int(K),
                "n_causal_states": int(len(cg)),
                "n_covered_states": int(covered_states),
                "coverage": safe_div(covered_states, len(cg)),
                "overall_heads_mean_percentile": float(np.mean(vo)) if len(vo) else np.nan,
                "overall_heads_rankweighted_percentile": rwmean(vo, ro),
                "relation_heads_mean_percentile": float(np.mean(vr)) if len(vr) else np.nan,
                "relation_heads_rankweighted_percentile": rwmean(vr, rr),
                "overall_heads_matched_overlap": safe_div(
                    overlap_num_overall, overlap_den_overall
                ),
                "relation_heads_matched_overlap": safe_div(
                    overlap_num_rel, overlap_den_rel
                ),
            })

    return state_rows, overlap_rows, cand_rows


# =============================================================================
# Summaries
# =============================================================================

def summarize_oracle_states(state_df, ks):
    rows = []
    for K in ks:
        q = state_df[state_df.causal_rank <= K]
        for (HL, h, bc), g in q.groupby(
            ["head_layer", "head", "baseline_correct"]
        ):
            rows.append({
                "K": int(K),
                "head_layer": int(HL),
                "head": int(h),
                "head_name": head_name(HL, h),
                "baseline_group": "correct" if bc else "wrong",
                "N_samples": int(g.sid.nunique()),
                "N_states": int(len(g)),
                "direction_accuracy": safe_mean(g.direction_accuracy),
                "mean_spatial_percentile": safe_mean(g.spatial_percentile),
                "top10_rate": float(g.top10_hit.mean()),
                "top05_rate": float(g.top05_hit.mean()),
                "top01_rate": float(g.top01_exact.mean()),
            })

        for (HL, h), g in q.groupby(["head_layer", "head"]):
            rows.append({
                "K": int(K),
                "head_layer": int(HL),
                "head": int(h),
                "head_name": head_name(HL, h),
                "baseline_group": "all",
                "N_samples": int(g.sid.nunique()),
                "N_states": int(len(g)),
                "direction_accuracy": safe_mean(g.direction_accuracy),
                "mean_spatial_percentile": safe_mean(g.spatial_percentile),
                "top10_rate": float(g.top10_hit.mean()),
                "top05_rate": float(g.top05_hit.mean()),
                "top01_rate": float(g.top01_exact.mean()),
            })
    return pd.DataFrame(rows)


def summarize_overlap(overlap_df):
    rows = []
    if not len(overlap_df):
        return pd.DataFrame()
    for (K, HL, h, group), g in overlap_df.assign(
        baseline_group=np.where(
            overlap_df.baseline_correct, "correct", "wrong"
        )
    ).groupby(["K", "head_layer", "head", "baseline_group"]):
        denom = g.budget_m.sum()
        micro = safe_div(g.intersection.sum(), denom)
        expected = safe_div(
            np.sum(g.random_expected_recall * g.budget_m), denom
        )
        rows.append({
            "K": int(K),
            "head_layer": int(HL),
            "head": int(h),
            "head_name": head_name(HL, h),
            "baseline_group": group,
            "N_samples": int(g.sid.nunique()),
            "total_budget": int(denom),
            "micro_recall": micro,
            "random_expected_recall": expected,
            "enrichment_ratio": safe_div(micro, expected),
            "direction_accuracy": safe_mean(g.direction_accuracy),
        })

    for (K, HL, h), g in overlap_df.groupby(["K", "head_layer", "head"]):
        denom = g.budget_m.sum()
        micro = safe_div(g.intersection.sum(), denom)
        expected = safe_div(
            np.sum(g.random_expected_recall * g.budget_m), denom
        )
        rows.append({
            "K": int(K),
            "head_layer": int(HL),
            "head": int(h),
            "head_name": head_name(HL, h),
            "baseline_group": "all",
            "N_samples": int(g.sid.nunique()),
            "total_budget": int(denom),
            "micro_recall": micro,
            "random_expected_recall": expected,
            "enrichment_ratio": safe_div(micro, expected),
            "direction_accuracy": safe_mean(g.direction_accuracy),
        })
    return pd.DataFrame(rows)


def head_accuracy_alignment_correlations(state_summary):
    rows = []
    allg = state_summary[state_summary.baseline_group == "all"].copy()
    for (K, HL), g in allg.groupby(["K", "head_layer"]):
        rows.append({
            "K": int(K),
            "head_layer": int(HL),
            "n_heads": int(len(g)),
            "rho_directionAcc_vs_percentile": spearman(
                g.direction_accuracy, g.mean_spatial_percentile
            ),
            "rho_directionAcc_vs_top10": spearman(
                g.direction_accuracy, g.top10_rate
            ),
            "rho_directionAcc_vs_top01": spearman(
                g.direction_accuracy, g.top01_rate
            ),
        })

    # Within-layer pooled head-rank correlation, deconfounded for layer.
    for K, g in allg.groupby("K"):
        z = g.copy()
        z["acc_rank"] = z.groupby("head_layer")["direction_accuracy"].rank(pct=True)
        z["pct_rank"] = z.groupby("head_layer")["mean_spatial_percentile"].rank(pct=True)
        z["top10_rank"] = z.groupby("head_layer")["top10_rate"].rank(pct=True)
        rows.append({
            "K": int(K),
            "head_layer": -1,
            "n_heads": int(len(z)),
            "rho_directionAcc_vs_percentile": spearman(z.acc_rank, z.pct_rank),
            "rho_directionAcc_vs_top10": spearman(z.acc_rank, z.top10_rank),
            "rho_directionAcc_vs_top01": np.nan,
        })
    return pd.DataFrame(rows)


def selector_summary(candidate_df):
    score_cols = [
        "overall_heads_mean_percentile",
        "overall_heads_rankweighted_percentile",
        "relation_heads_mean_percentile",
        "relation_heads_rankweighted_percentile",
        "overall_heads_matched_overlap",
        "relation_heads_matched_overlap",
    ]
    rows = []
    details = []
    for (K, sid), g in candidate_df.groupby(["K", "sid"]):
        if set(g.candidate_relation) != set(REL):
            continue
        gt = str(g.gt.iloc[0])
        bp = str(g.baseline_prediction.iloc[0])
        bc = bool(g.baseline_correct.iloc[0])

        for score_col in score_cols:
            gg = g[["candidate_relation", score_col]].dropna()
            if len(gg) < 4:
                continue
            winner = gg.loc[gg[score_col].idxmax()]
            pred = str(winner.candidate_relation)
            vals = gg.sort_values(score_col, ascending=False)[score_col].to_numpy(float)
            detail = {
                "sid": int(sid),
                "K": int(K),
                "score": score_col,
                "gt": gt,
                "baseline_prediction": bp,
                "baseline_correct": bc,
                "pred": pred,
                "pred_is_gt": pred == gt,
                "pred_matches_baseline": pred == bp,
                "winner_margin": float(vals[0] - vals[1]),
            }
            for r in REL:
                hit = gg[gg.candidate_relation == r]
                detail[f"score_{r}"] = (
                    float(hit[score_col].iloc[0]) if len(hit) else np.nan
                )
            details.append(detail)

    d = pd.DataFrame(details)
    if not len(d):
        return pd.DataFrame(), d

    for (K, score), g in d.groupby(["K", "score"]):
        c = g.baseline_correct
        w = ~c
        rows.append({
            "K": int(K),
            "score": score,
            "N": len(g),
            "GTacc_all": float(g.pred_is_gt.mean()),
            "GTacc_correct": float(g.loc[c, "pred_is_gt"].mean()) if c.any() else np.nan,
            "GTacc_wrong": float(g.loc[w, "pred_is_gt"].mean()) if w.any() else np.nan,
            "matchBaseline_all": float(g.pred_matches_baseline.mean()),
            "matchBaseline_wrong": (
                float(g.loc[w, "pred_matches_baseline"].mean()) if w.any() else np.nan
            ),
            "margin_correct": safe_mean(g.loc[c, "winner_margin"]),
            "margin_wrong": safe_mean(g.loc[w, "winner_margin"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["GTacc_wrong", "GTacc_all"], ascending=False
    ), d


def focus_head_report(state_summary, overlap_summary, focus_heads, ks):
    rows = []
    for L, H in focus_heads:
        hn = head_name(L, H)
        for K in ks:
            s = state_summary[
                (state_summary.head_layer == L)
                & (state_summary["head"] == H)
                & (state_summary.K == K)
                & (state_summary.baseline_group == "all")
            ]
            o = overlap_summary[
                (overlap_summary.head_layer == L)
                & (overlap_summary["head"] == H)
                & (overlap_summary.K == K)
                & (overlap_summary.baseline_group == "all")
            ] if len(overlap_summary) else pd.DataFrame()

            if not len(s) and not len(o):
                continue
            rows.append({
                "head_name": hn,
                "K": K,
                "N_samples": int(s.N_samples.iloc[0]) if len(s) else (
                    int(o.N_samples.iloc[0]) if len(o) else 0
                ),
                "N_states": int(s.N_states.iloc[0]) if len(s) else 0,
                "direction_accuracy": float(s.direction_accuracy.iloc[0]) if len(s) else (
                    float(o.direction_accuracy.iloc[0]) if len(o) else np.nan
                ),
                "mean_spatial_percentile": float(s.mean_spatial_percentile.iloc[0]) if len(s) else np.nan,
                "top10_rate": float(s.top10_rate.iloc[0]) if len(s) else np.nan,
                "top05_rate": float(s.top05_rate.iloc[0]) if len(s) else np.nan,
                "top01_rate": float(s.top01_rate.iloc[0]) if len(s) else np.nan,
                "matched_micro_recall": float(o.micro_recall.iloc[0]) if len(o) else np.nan,
                "random_expected_recall": float(o.random_expected_recall.iloc[0]) if len(o) else np.nan,
                "overlap_enrichment": float(o.enrichment_ratio.iloc[0]) if len(o) else np.nan,
            })
    return pd.DataFrame(rows)


def render_summary(
    args,
    focus_df,
    corr_df,
    selector_df,
    state_df,
    direction_meta,
):
    lines = []
    lines.append("=" * 166)
    lines.append("CAUSAL TOKEN RANKING vs DIRECTION-HEAD SPATIAL-SOURCE RANKING")
    lines.append("=" * 166)
    lines.append(
        f"N={state_df.sid.nunique() if len(state_df) else 0} | "
        f"source={args.source_layers} | Ks={args.ks} | "
        f"direction_fit={args.direction_fit} (N={direction_meta['fit_n']})"
    )
    lines.append(
        "Layer alignment is exact: causal block-output L -> Direction Head L+1."
    )
    lines.append(
        "Spatial score uses c_RG=[(A_sub-A_ref)V]_Real - [(A_sub-A_ref)V]_Gray, "
        "projected onto independently fit per-head relation direction."
    )
    lines.append("")

    lines.append("1) FOCUS HEADS: RANK-WISE CONVERGENCE")
    lines.append("-" * 166)
    if len(focus_df):
        for hn in [head_name(*parse_head_name(x)) for x in args.focus_heads.split(",") if x.strip()]:
            g = focus_df[focus_df.head_name == hn].sort_values("K")
            if not len(g):
                continue
            lines.append(f"{hn}:")
            for r in g.itertuples():
                lines.append(
                    f"  K={int(r.K):2d} Nsample={int(r.N_samples):2d} Nstate={int(r.N_states):3d} | "
                    f"pct={r.mean_spatial_percentile:.3f} "
                    f"Top10={r.top10_rate:.3f} Top5={r.top05_rate:.3f} Top1={r.top01_rate:.3f} | "
                    f"matchedRecall={r.matched_micro_recall:.3f} "
                    f"rand={r.random_expected_recall:.3f} enrich={r.overlap_enrichment:.2f}x"
                )
    else:
        lines.append("No focus-head rows (check focus head names / layer coverage).")
    lines.append("")

    lines.append("2) DOES HIGHER DIRECTION-HEAD ACCURACY PREDICT STRONGER CAUSAL/SPATIAL CONVERGENCE?")
    lines.append("-" * 166)
    if len(corr_df):
        z = corr_df[corr_df.K.isin([1, 3, 5, 7, 36])].copy()
        for r in z.itertuples():
            layer = "POOLED-within-layer" if int(r.head_layer) == -1 else f"L{int(r.head_layer):02d}"
            lines.append(
                f"K={int(r.K):2d} {layer:<20s} nHeads={int(r.n_heads):3d} | "
                f"rho(acc,pct)={r.rho_directionAcc_vs_percentile:+.3f} "
                f"rho(acc,Top10)={r.rho_directionAcc_vs_top10:+.3f}"
            )
    lines.append("")

    lines.append("3) NON-ORACLE FOUR-WAY CANDIDATE-K SELECTION BY CROSS-MECHANISM AGREEMENT")
    lines.append("-" * 166)
    if len(selector_df):
        z = selector_df[
            selector_df.K.isin(sorted(set([args.selector_k, 3, 5, 7])))
        ].sort_values(["GTacc_wrong", "GTacc_all"], ascending=False)
        for r in z.head(30).itertuples():
            lines.append(
                f"K={int(r.K):2d} {r.score:<42s} | "
                f"GTacc all={r.GTacc_all:.3f} correct={r.GTacc_correct:.3f} "
                f"wrong={r.GTacc_wrong:.3f} | "
                f"matchBaselineWrong={r.matchBaseline_wrong:.3f}"
            )
    else:
        lines.append("No complete four-way selector rows.")
    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 166)
    lines.append(
        "The original convergence hypothesis is supported if focus Direction Heads show high "
        "spatial percentile for the strongest causal ranks, the percentile/overlap decays as K grows, "
        "and higher direction-decoding heads outperform weaker heads within the same layer."
    )
    lines.append(
        "The selector is a separate question: even a real causal/spatial convergence does not guarantee "
        "that cross-mechanism agreement can identify the correct one among four candidate K sets."
    )
    lines.append(
        "For selector usefulness, the decisive column is GTacc_wrong. Candidate scores themselves use "
        "no test GT; GT appears only in evaluation."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.attn_impl != "eager":
        print(
            "[WARN] Exact reconstruction is validated against Qwen eager attention. "
            "For this experiment, --attn-impl eager is strongly recommended."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_layers = parse_ints(args.source_layers)
    ks = parse_ints(args.ks)
    if args.selector_k not in ks:
        ks = sorted(set(ks + [args.selector_k]))
    max_k = max(ks)
    target_head_layers = sorted({L + 1 for L in source_layers})
    focus_heads = [
        parse_head_name(x)
        for x in args.focus_heads.split(",")
        if x.strip()
    ]

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dir = outdir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    fourway = Path(args.fourway_dir)
    med_path = fourway / "all_four_writer_mediation.pkl.gz"
    ps_path = fourway / "per_sample_direction_scores.csv"
    if not med_path.exists():
        raise FileNotFoundError(med_path)
    if not ps_path.exists():
        raise FileNotFoundError(ps_path)

    med = pd.read_pickle(med_path, compression="gzip")
    ps = pd.read_csv(ps_path)

    med["sid"] = pd.to_numeric(med["sid"], errors="raise").astype(int)
    med["source_layer"] = pd.to_numeric(
        med["source_layer"], errors="raise"
    ).astype(int)
    med["position"] = pd.to_numeric(med["position"], errors="raise").astype(int)
    med["candidate_relation"] = med["candidate_relation"].map(canon_rel)
    med["gt"] = med["gt"].map(canon_rel)
    med["baseline_prediction"] = med["baseline_prediction"].map(canon_rel)
    med["baseline_correct"] = med["baseline_correct"].map(boolify)
    med = med[med.source_layer.isin(source_layers)].copy()

    ps["sid"] = pd.to_numeric(ps["sid"], errors="raise").astype(int)
    ps["gt"] = ps["gt"].map(canon_rel)
    ps["baseline_prediction"] = ps["baseline_prediction"].map(canon_rel)
    ps["baseline_correct"] = ps["baseline_correct"].map(boolify)

    eval_sids = sorted(set(ps.sid) & set(med.sid))
    if args.max_eval_samples and args.max_eval_samples > 0:
        # Keep same deterministic N80 subset if the fourway run itself has N80.
        # If more are present, sample deterministically.
        if len(eval_sids) > args.max_eval_samples:
            rng = np.random.default_rng(args.seed)
            eval_sids = sorted(
                rng.choice(
                    eval_sids,
                    size=args.max_eval_samples,
                    replace=False,
                ).tolist()
            )

    med = med[med.sid.isin(eval_sids)].copy()
    ps = ps[ps.sid.isin(eval_sids)].copy()

    ranked = build_all_causal_ranks(med, max_k=max_k)
    if not len(ranked):
        raise RuntimeError("No positive causal rankings found.")

    dirs, head_results, direction_meta = load_direction_probe(
        args.direction_dir, set(eval_sids), args.direction_fit
    )

    # Validate layers exist in direction cache.
    max_dir_layer = dirs.shape[0] - 1
    target_head_layers = [L for L in target_head_layers if L <= max_dir_layer]
    if not target_head_layers:
        raise RuntimeError("No aligned target head layers available in direction cache.")

    # Load model/data using repo-native helpers.
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _ = two.load_records("coco_two", Path(args.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    missing = [
        sid for sid in eval_sids
        if sid not in prompts or sid not in rec_by_sid
    ]
    if missing:
        raise RuntimeError(f"Missing prompt/record for SIDs: {missing[:20]}")

    specs = base.merged_model_specs(two)
    spec = specs[args.model]
    cls = getattr(transformers, spec.model_class)
    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"Loading {spec.repo_id}", flush=True)
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model, processor)
    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = hprobe.resolve_decoder_layers(model)
    target_head_layers = [
        L for L in target_head_layers if 0 <= L < len(decoder_layers)
    ]
    print(
        f"decoder={decoder_path} | N={len(eval_sids)} | "
        f"causal source={source_layers} | aligned head layers={target_head_layers}"
    )

    all_state_rows = []
    all_overlap_rows = []
    all_candidate_rows = []
    errors = []

    try:
        for sid in tqdm(eval_sids, desc="source-ranking"):
            cache = chunk_dir / f"sid_{sid}.pkl.gz"
            if cache.exists():
                obj = pd.read_pickle(cache, compression="gzip")
                all_state_rows.extend(obj["state"])
                all_overlap_rows.extend(obj["overlap"])
                all_candidate_rows.extend(obj["candidate"])
                continue

            real = gray = rb = gb = None
            try:
                meta = prompts[sid]
                subject = str(meta["subject"])
                reference = str(meta["reference"])
                question_text = str(meta["question_text"])

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, args.gray_value)

                device = torch.device(args.device)
                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=question_text,
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=question_text,
                    device=device,
                )

                ids_r = rb["input_ids"][0].detach().cpu().tolist()
                ids_g = gb["input_ids"][0].detach().cpu().tolist()
                if ids_r != ids_g:
                    raise RuntimeError(
                        "Real/Gray tokenization mismatch; cannot align source positions."
                    )

                subpos = hprobe.locate_phrase_positions(
                    processor.tokenizer, ids_r, subject
                )
                refpos = hprobe.locate_phrase_positions(
                    processor.tokenizer, ids_r, reference
                )

                real_caps = capture_condition_contributions(
                    model, decoder_layers, rb,
                    target_head_layers, subpos, refpos
                )
                gray_caps = capture_condition_contributions(
                    model, decoder_layers, gb,
                    target_head_layers, subpos, refpos
                )

                spatial = compute_spatial_scores(
                    real_caps, gray_caps, dirs, target_head_layers
                )

                rsid = ranked[ranked.sid == sid].copy()
                if not len(rsid):
                    raise RuntimeError("No causal ranking rows for SID")

                sr, ov, cr = analyze_sample(
                    sid=sid,
                    ranked_sid=rsid,
                    spatial=spatial,
                    head_results=head_results,
                    ks=ks,
                    top_heads_per_layer=args.top_heads_per_layer,
                )

                pd.to_pickle(
                    {"state": sr, "overlap": ov, "candidate": cr},
                    cache,
                    compression="gzip",
                )
                all_state_rows.extend(sr)
                all_overlap_rows.extend(ov)
                all_candidate_rows.extend(cr)

            except Exception as e:
                import traceback
                errors.append({
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                })
                tqdm.write(f"[ERROR] sid={sid}: {type(e).__name__}: {e}")
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    state_df = pd.DataFrame(all_state_rows)
    overlap_df = pd.DataFrame(all_overlap_rows)
    candidate_df = pd.DataFrame(all_candidate_rows)

    if not len(state_df):
        raise RuntimeError(
            "No state-level results were produced. Check errors.json / prompt alignment."
        )

    state_summary = summarize_oracle_states(state_df, ks)
    overlap_summary = summarize_overlap(overlap_df)
    corr_df = head_accuracy_alignment_correlations(state_summary)
    selector_df, selector_detail = selector_summary(candidate_df)
    focus_df = focus_head_report(
        state_summary, overlap_summary, focus_heads, ks
    )

    # Save compact result tables.
    state_df.to_csv(outdir / "oracle_causal_states_vs_head_source_rank.csv", index=False)
    state_summary.to_csv(outdir / "oracle_head_rank_convergence_summary.csv", index=False)
    overlap_df.to_csv(outdir / "oracle_matched_budget_overlap_detail.csv", index=False)
    overlap_summary.to_csv(outdir / "oracle_matched_budget_overlap_summary.csv", index=False)
    corr_df.to_csv(outdir / "direction_accuracy_vs_alignment_correlation.csv", index=False)
    candidate_df.to_csv(outdir / "fourway_candidate_crossmechanism_scores.csv", index=False)
    selector_df.to_csv(outdir / "fourway_selector_summary.csv", index=False)
    selector_detail.to_csv(outdir / "fourway_selector_detail.csv", index=False)
    focus_df.to_csv(outdir / "focus_heads_rankwise_convergence.csv", index=False)

    (outdir / "errors.json").write_text(
        json.dumps(errors, indent=2), encoding="utf-8"
    )

    metadata = {
        "N_requested": len(eval_sids),
        "N_success": int(state_df.sid.nunique()),
        "source_layers": source_layers,
        "aligned_head_layers": target_head_layers,
        "ks": ks,
        "selector_k": args.selector_k,
        "top_heads_per_layer": args.top_heads_per_layer,
        "focus_heads": [head_name(*x) for x in focus_heads],
        "gray_value": args.gray_value,
        "direction_fit": args.direction_fit,
        "direction_probe": direction_meta,
        "definition": {
            "causal": "M=(h_real-h_gray)^T grad J_r",
            "source_contribution": (
                "c_RG=[(A_sub-A_ref)V]_Real - [(A_sub-A_ref)V]_Gray"
            ),
            "spatial_source_score": "dot(c_RG, per-head relation direction d_r)",
            "layer_alignment": "causal block-output L -> attention head L+1",
            "percentile": "ascending percentile among causally visible source tokens; 1=highest",
        },
        "selector_note": (
            "Candidate scores use no test GT. Per-head relation directions are calibrated "
            "from probe relation labels; evaluation GT is used only after all four candidate scores are computed."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    text = render_summary(
        args, focus_df, corr_df, selector_df, state_df, direction_meta
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    print()
    print(text)
    if errors:
        print(f"[WARN] {len(errors)} samples failed; see {outdir/'errors.json'}")
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
