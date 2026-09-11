#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_reverse_spatial_to_causal_core_all440_v1.py

Goal
====
Reverse the previously observed spatial-head <-> writer-causal-token association
WITHOUT changing the spatial score that established the association.

The original association used, for Direction Head h at attention layer H:

    c_RG(h,p)
      = [A_real(sub,p)-A_real(ref,p)] V_real(h,p)
        -
        [A_gray(sub,p)-A_gray(ref,p)] V_gray(h,p)

and, for relation r,

    S_spatial(h,p;r) = < c_RG(h,p), d_h,r >

with strict layer alignment:

    causal block-output state L  <->  attention head layer H=L+1.

This script keeps exactly that structure and asks whether spatial source ranking
can be REVERSED into causal-state selection.

Alignment improvements in this version
======================================

1) Synthetic-400 directions are re-extracted as REAL-GRAY head residuals,
   NOT the older Real-NoImage source residual:

       q_RG(h)
         = [z_real(sub)-z_real(ref)]
           -
           [z_gray(sub)-z_gray(ref)]

   Synthetic uses the exact COCO Direction-probe prompt wording by default.

2) COCO token source scores use the same REAL-GRAY A*V decomposition as the
   original association experiment.

3) Relation vectors and token-source vectors therefore live in the SAME
   PRE-W_O per-head feature space and use the SAME real-vs-gray contrast.

4) Layer mapping is kept conditional:
       head H scores causal state layer H-1.
   We do NOT infer layer by comparing unrelated raw scales.

5) Head selection is from COCO calibration accuracy, PER LAYER, so every
   causal source layer 20..26 remains represented.

6) A clean COCO-calibrated Real-Gray direction bank is also fit on the
   calibration split as a sanity bank.  This lets us separate:
       cross-domain direction mismatch
   from
       failure of the reverse spatial->causal implication.

Main experimental lines
=======================

A. cocoRG + GT relation + ORACLE per-layer budget
   ------------------------------------------------
   Strongest sanity ceiling.
   This is the direct inverse of the original matched-budget association:
   for each causal source layer L, if the oracle target contains m states,
   take the spatial ranking Top-m at the aligned head layer L+1.

   If this is NOT strong, the old association does not transfer to the current
   440/Core50 bank definition and we should stop before building a selector.

B. synRG + GT relation + ORACLE per-layer budget
   ----------------------------------------------
   Tests whether Synthetic-400 Real-Gray directions preserve the same inverse
   association across domains.

C. synRG + PREDICTED relation + ORACLE per-layer budget
   -----------------------------------------------------
   Isolates relation-routing error while keeping the layer budget fixed.

D. synRG + PREDICTED relation + CALIBRATION layer prior
   -----------------------------------------------------
   Practical reverse selector:
       relation = spatial-head vote
       head identities = COCO calibration high-ACC heads per layer
       token ranking = original Real-Gray spatial score
       layer budget = causal layer prior learned only from calibration SIDs

E. synRG + PREDICTED relation + GLOBAL Top-K
   ------------------------------------------
   No sample-specific layer budget. Exact states from all aligned layers are
   ranked by within-layer spatial percentile; global_unique removes duplicate
   token positions across layers.

Direction banks
===============
syn_rg:
    fitted only from synthetic_shapes_4dir_400 Real-Gray head residuals.

coco_calib_rg:
    fitted only from COCO calibration-split Real-Gray head residuals.
    Used as a sanity / upper-reference bank, not the desired cross-domain method.

Evaluation bank
===============
Oracle Top10/Core50 text states come from:
    build_oracle_core50_top10_candidate_bank_all440_v1.py

Oracle causal labels are NEVER used to compute token spatial scores. They are
used only for:
    - evaluation;
    - explicit diagnostic oracle layer budget;
    - calibration-only layer priors for the practical selector.

Typical full-440 diagnostic
===========================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_reverse_spatial_to_causal_core_all440_v1.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --oracle-bank-dir output/qwen3b_oracle_core50_top10_bank_all440 \
  --head-layers 21,22,23,24,25,26,27 \
  --heads-per-layer 1,3,5 \
  --vote-top-n 10 \
  --select-ks 3,5,7,10 \
  --eval-scope all_data \
  --require-eval-n 440 \
  --cache-token-projections \
  --output-dir output/qwen3b_reverse_spatial_to_causal_all440_v1 \
  --overwrite

Publication-style heldout check
===============================
Use:
    --eval-scope heldout
    --calib-frac 0.30
and omit --require-eval-n 440.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn

# Reuse the exact Real-Gray A*V Direction-head machinery from the original
# association scan rather than reimplementing a subtly different score.
import scan_qwen_spatial_heads_vs_causal_core_v1 as oldscan


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12

PROBE_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

SYN_REL_MAP = {
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
}


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default="")
    p.add_argument("--synthetic-max-samples", type=int, default=0)

    p.add_argument(
        "--oracle-bank-dir",
        required=True,
        help="Output of build_oracle_core50_top10_candidate_bank_all440_v1.py",
    )

    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--direction-pool", choices=["mean", "last"], default="mean")

    p.add_argument(
        "--head-layers",
        default="21,22,23,24,25,26,27",
        help="Attention head layers; state source layer = head layer - 1.",
    )
    p.add_argument("--head-input-offset", type=int, default=1)
    p.add_argument(
        "--heads-per-layer",
        default="1,3,5",
        help="High-COCO-accuracy Direction heads retained PER aligned layer.",
    )
    p.add_argument(
        "--vote-top-n",
        type=int,
        default=10,
        help="Global high-accuracy heads used for sample relation voting.",
    )
    p.add_argument(
        "--select-ks",
        default="3,5,7,10",
        help="Final candidate counts for prior-budget/global selection.",
    )
    p.add_argument(
        "--head-weight",
        choices=["equal", "accuracy"],
        default="accuracy",
    )
    p.add_argument(
        "--positive-only",
        action="store_true",
        help="Clamp negative spatial token projections to zero before ranking.",
    )

    p.add_argument("--calib-frac", type=float, default=0.30)
    p.add_argument("--split-seed", type=int, default=17)
    p.add_argument(
        "--eval-scope",
        choices=["all_data", "heldout"],
        default="all_data",
        help=(
            "all_data evaluates all 440 (diagnostic; calibration SIDs are included); "
            "heldout evaluates only the disjoint heldout split."
        ),
    )
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument(
        "--require-eval-n",
        type=int,
        default=0,
        help="Fail unless eval size equals this value; 0 disables.",
    )

    p.add_argument(
        "--cache-token-projections",
        action="store_true",
        help="Cache expensive COCO Real-Gray token projections for both direction banks.",
    )
    p.add_argument("--overwrite-synthetic-cache", action="store_true")
    p.add_argument("--overwrite-token-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)

    return p.parse_args()


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return SYN_REL_MAP.get(s, s)


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def hname(L, h):
    return f"L{int(L)}H{int(h):02d}"


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def make_gray(real, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real.size, (v, v, v))


def rank_percentiles(scores):
    """
    Higher score -> percentile closer to 1.
    Average rank for exact ties.
    """
    x = np.asarray(scores, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    order = np.argsort(-x, kind="mergesort")
    rank = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        rank[order[i:j]] = avg
        i = j

    return ((n - rank) / float(n - 1)).astype(np.float32)


# =============================================================================
# Synthetic loader
# =============================================================================

def load_synthetic_rows(a):
    root = Path(a.synthetic_dir)
    labels_path = (
        Path(a.synthetic_labels)
        if a.synthetic_labels
        else root / "labels.jsonl"
    )
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            raw = str(item["relation"]).strip().lower()
            if raw not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: bad relation={raw!r}"
                )

            image_value = Path(str(item["image"]))
            image_path = (
                image_value
                if image_value.is_absolute()
                else root / image_value
            )
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()

            rows.append({
                "sid": int(item.get("id", len(rows))),
                "image_path": str(image_path),
                "subject": subject,
                "reference": reference,
                "relation": SYN_REL_MAP[raw],
                # Exact same probe prompt wording as COCO Direction probe.
                "question_text": PROBE_PROMPT.format(
                    subject=subject,
                    reference=reference,
                ),
            })

    rows.sort(key=lambda r: r["sid"])
    if a.synthetic_max_samples > 0:
        rows = rows[: int(a.synthetic_max_samples)]

    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in REL if counts[r] == 0]
    if missing:
        raise RuntimeError(
            f"Synthetic source missing={missing}; counts={dict(counts)}"
        )

    return rows, labels_path


# =============================================================================
# Oracle causal bank
# =============================================================================

def load_oracle_bank(root):
    root = Path(root)
    p_top = root / "top10_candidates_all440.csv"
    p_core = root / "core50_candidates_all440.csv"

    if not p_top.exists():
        raise FileNotFoundError(p_top)
    if not p_core.exists():
        raise FileNotFoundError(p_core)

    def one(path):
        df = pd.read_csv(path)
        req = {"sid", "source_layer", "position", "category", "broad_category"}
        miss = req - set(df.columns)
        if miss:
            raise RuntimeError(f"{path} missing {sorted(miss)}")

        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
        df["source_layer"] = pd.to_numeric(
            df["source_layer"], errors="raise"
        ).astype(int)
        df["position"] = pd.to_numeric(
            df["position"], errors="raise"
        ).astype(int)

        if "is_text" in df.columns:
            df["is_text"] = df["is_text"].map(boolify)
        else:
            df["is_text"] = (
                ~df["broad_category"].astype(str).eq("visual")
                & ~df["category"].astype(str).eq("visual")
            )

        return df[df["is_text"]].copy()

    return one(p_top), one(p_core)


def target_lookup(df):
    out = {}
    for sid, g in df.groupby("sid"):
        states = {
            (int(r.source_layer), int(r.position))
            for r in g.itertuples()
        }
        by_layer = defaultdict(set)
        for L, p in states:
            by_layer[L].add(p)

        out[int(sid)] = {
            "states": states,
            "positions": {p for _, p in states},
            "by_layer": {L: set(v) for L, v in by_layer.items()},
        }
    return out


def layer_prior_from_calibration(df, calib_sids, source_layers):
    q = df[df["sid"].isin(set(calib_sids))].copy()
    counts = Counter(int(x) for x in q["source_layer"].tolist())
    arr = np.asarray(
        [float(counts.get(L, 0)) for L in source_layers],
        dtype=np.float64,
    )
    if arr.sum() <= 0:
        arr[:] = 1.0
    arr /= arr.sum()
    return {L: float(arr[i]) for i, L in enumerate(source_layers)}


def allocate_largest_remainder(prior, K, source_layers):
    probs = np.asarray([prior.get(L, 0.0) for L in source_layers], float)
    if probs.sum() <= 0:
        probs[:] = 1.0
    probs /= probs.sum()

    raw = probs * int(K)
    base_n = np.floor(raw).astype(int)
    left = int(K) - int(base_n.sum())

    order = np.argsort(-(raw - base_n), kind="mergesort")
    for i in order[:left]:
        base_n[i] += 1

    return {L: int(base_n[i]) for i, L in enumerate(source_layers)}


# =============================================================================
# Direction banks
# =============================================================================

def fit_codebook(X, y):
    """
    X [N, n_head_layers, H, D]
    center [n_head_layers,H,D]
    dirs   [n_head_layers,H,4,D]
    """
    center = X.mean(axis=0).astype(np.float32)
    dirs = np.zeros(
        (X.shape[1], X.shape[2], len(REL), X.shape[3]),
        dtype=np.float32,
    )
    for ri, rel in enumerate(REL):
        m = y == rel
        if not np.any(m):
            raise RuntimeError(f"No samples for relation={rel}")
        dirs[:, :, ri, :] = normalize(
            X[m].mean(axis=0) - center,
            axis=-1,
        )
    return center, dirs


def codebook_scores(X, center, dirs):
    q = normalize(X - center[None, :, :, :], axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", q, dirs, optimize=True)


def head_accuracy(scores, y, head_layers):
    yi = np.asarray([RID[r] for r in y], dtype=int)
    pred = np.argmax(scores, axis=-1)

    rows = []
    for li, H in enumerate(head_layers):
        for h in range(scores.shape[2]):
            ok = pred[:, li, h] == yi
            row = {
                "head_layer": H,
                "head": h,
                "head_name": hname(H, h),
                "accuracy": float(ok.mean()),
                "N": len(ok),
            }
            for ri, r in enumerate(REL):
                m = yi == ri
                row[f"acc_{r}"] = (
                    float(ok[m].mean()) if np.any(m) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def compare_direction_banks(
    syn_dirs,
    coco_dirs,
    head_layers,
):
    rows = []
    for li, H in enumerate(head_layers):
        for h in range(syn_dirs.shape[1]):
            for ri, r in enumerate(REL):
                a = syn_dirs[li, h, ri]
                b = coco_dirs[li, h, ri]
                rows.append({
                    "head_layer": H,
                    "head": h,
                    "head_name": hname(H, h),
                    "relation": r,
                    "cos_synRG_vs_cocoCalibRG": float(
                        np.dot(normalize(a), normalize(b))
                    ),
                })
    return pd.DataFrame(rows)


# =============================================================================
# Model extraction helpers
# =============================================================================

def all_head_delta_from_caps(
    real_cap,
    gray_cap,
    Hlayer,
    spos,
    rpos,
    pool,
):
    H = real_cap["n_heads"]
    D = real_cap["head_dim"]

    Hr = oldscan.all_head_pre_o(
        real_cap["pre_o"], Hlayer, H, D
    )
    Hg = oldscan.all_head_pre_o(
        gray_cap["pre_o"], Hlayer, H, D
    )

    if pool == "last":
        s = int(spos[-1])
        r = int(rpos[-1])
        delta = Hr[s] - Hr[r] - Hg[s] + Hg[r]
    else:
        ss = [int(x) for x in spos]
        rr = [int(x) for x in rpos]
        delta = (
            Hr[ss].mean(axis=0)
            - Hr[rr].mean(axis=0)
            - Hg[ss].mean(axis=0)
            + Hg[rr].mean(axis=0)
        )

    return delta.astype(np.float32)


def direction_token_projection_all_relations(
    real_cap,
    gray_cap,
    Hlayer,
    spos,
    rpos,
    pool,
    dirs_h4d,
    text_positions,
):
    """
    EXACT original Real-Gray A*V source decomposition, but project onto all
    four relation directions at once.

    returns:
        proj [H, P_text, 4]
        head_delta [H,D]
        recon_cos [H]
    """
    Ar = real_cap["attn"][Hlayer]  # [H,Q,K]
    Ag = gray_cap["attn"][Hlayer]

    H = real_cap["n_heads"]
    D = real_cap["head_dim"]

    Vr = oldscan.all_head_v(
        real_cap["v"][Hlayer],
        H,
        real_cap["n_kv_heads"],
        D,
    )
    Vg = oldscan.all_head_v(
        gray_cap["v"][Hlayer],
        H,
        gray_cap["n_kv_heads"],
        D,
    )

    if pool == "last":
        qs = int(spos[-1])
        qr = int(rpos[-1])
        cr = Ar[:, qs, :] - Ar[:, qr, :]
        cg = Ag[:, qs, :] - Ag[:, qr, :]
    else:
        ss = [int(x) for x in spos]
        rr = [int(x) for x in rpos]
        cr = Ar[:, ss, :].mean(axis=1) - Ar[:, rr, :].mean(axis=1)
        cg = Ag[:, ss, :].mean(axis=1) - Ag[:, rr, :].mean(axis=1)

    n = min(cr.shape[1], cg.shape[1], Vr.shape[0], Vg.shape[0])
    cr = cr[:, :n]
    cg = cg[:, :n]
    Vr = Vr[:n]
    Vg = Vg[:n]

    # [H,K,D] -- exact old spatial source vector.
    vec = (
        cr[:, :, None] * np.transpose(Vr, (1, 0, 2))
        -
        cg[:, :, None] * np.transpose(Vg, (1, 0, 2))
    ).astype(np.float32)

    valid_pos = [int(p) for p in text_positions if 0 <= int(p) < n]
    if valid_pos != list(map(int, text_positions)):
        raise RuntimeError("Some text positions fall outside A/V sequence.")

    C = vec[:, np.asarray(valid_pos, dtype=np.int64), :]  # [H,P,D]
    Dirs = np.asarray(dirs_h4d, dtype=np.float32)         # [H,4,D]
    proj = np.einsum("hpd,hrd->hpr", C, Dirs, optimize=True)

    head_delta = all_head_delta_from_caps(
        real_cap, gray_cap, Hlayer, spos, rpos, pool
    )

    recon = vec.sum(axis=1)
    recon_cos = np.full(H, np.nan, dtype=np.float32)
    for h in range(H):
        a = head_delta[h]
        b = recon[h]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > EPS and nb > EPS:
            recon_cos[h] = float(np.dot(a, b) / (na * nb))

    return proj.astype(np.float32), head_delta, recon_cos


# =============================================================================
# Cache IO
# =============================================================================

def save_residual_cache(path, sids, labels, X, head_layers, note):
    np.savez_compressed(
        path,
        sample_index=np.asarray(sids, dtype=np.int64),
        relation=np.asarray(labels, dtype=object),
        residual=np.asarray(X, dtype=np.float16),
        head_layers=np.asarray(head_layers, dtype=np.int32),
        note=np.asarray([note], dtype=object),
    )


def load_residual_cache(path):
    z = np.load(path, allow_pickle=True)
    return (
        np.asarray(z["sample_index"]).astype(int),
        np.asarray([canon_rel(x) for x in z["relation"]], dtype=object),
        np.asarray(z["residual"], dtype=np.float32),
        np.asarray(z["head_layers"]).astype(int).tolist(),
    )


def token_cache_path(root, sid):
    return Path(root) / f"sid_{int(sid):06d}.npz"


def save_token_projection_cache(
    path,
    sid,
    head_layers,
    positions,
    token_ids,
    tokens,
    categories,
    broad_categories,
    head_delta,
    relation_scores_syn,
    relation_scores_coco,
    proj_syn,
    proj_coco,
    recon_cos,
):
    np.savez_compressed(
        path,
        sid=np.asarray([sid], dtype=np.int64),
        head_layers=np.asarray(head_layers, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.int32),
        token_ids=np.asarray(token_ids, dtype=np.int32),
        tokens=np.asarray(tokens, dtype=object),
        categories=np.asarray(categories, dtype=object),
        broad_categories=np.asarray(broad_categories, dtype=object),
        head_delta=head_delta.astype(np.float16),
        relation_scores_syn=relation_scores_syn.astype(np.float16),
        relation_scores_coco=relation_scores_coco.astype(np.float16),
        proj_syn=proj_syn.astype(np.float16),
        proj_coco=proj_coco.astype(np.float16),
        recon_cos=recon_cos.astype(np.float16),
    )


def load_token_projection_cache(path):
    z = np.load(path, allow_pickle=True)
    return {
        "sid": int(np.asarray(z["sid"]).reshape(-1)[0]),
        "head_layers": np.asarray(z["head_layers"]).astype(int).tolist(),
        "positions": np.asarray(z["positions"]).astype(int),
        "token_ids": np.asarray(z["token_ids"]).astype(int),
        "tokens": np.asarray(z["tokens"], dtype=object),
        "categories": np.asarray(z["categories"], dtype=object),
        "broad_categories": np.asarray(z["broad_categories"], dtype=object),
        "head_delta": np.asarray(z["head_delta"], dtype=np.float32),
        "relation_scores_syn": np.asarray(
            z["relation_scores_syn"], dtype=np.float32
        ),
        "relation_scores_coco": np.asarray(
            z["relation_scores_coco"], dtype=np.float32
        ),
        "proj_syn": np.asarray(z["proj_syn"], dtype=np.float32),
        "proj_coco": np.asarray(z["proj_coco"], dtype=np.float32),
        "recon_cos": np.asarray(z["recon_cos"], dtype=np.float32),
    }


# =============================================================================
# Split / head selection / voting
# =============================================================================

def stratified_split(meta, frac, seed):
    """
    Use repository helper if available; otherwise deterministic fallback.
    """
    try:
        return traj.stratified_split(meta, frac, seed)
    except Exception:
        rng = np.random.default_rng(seed)
        train, test = [], []
        for r in REL:
            idx = [i for i, m in enumerate(meta) if m["gt"] == r]
            rng.shuffle(idx)
            n = max(1, int(round(len(idx) * frac)))
            train.extend(meta[i] for i in idx[:n])
            test.extend(meta[i] for i in idx[n:])
        train.sort(key=lambda x: x["sid"])
        test.sort(key=lambda x: x["sid"])
        return train, test


def select_heads_per_layer(acc_df, head_layers, n):
    out = {}
    for H in head_layers:
        q = (
            acc_df[acc_df["head_layer"] == H]
            .sort_values(["accuracy", "head"], ascending=[False, True])
            .head(int(n))
        )
        out[H] = [
            (int(r.head), float(r.accuracy))
            for r in q.itertuples()
        ]
    return out


def global_top_heads(acc_df, n):
    q = acc_df.sort_values(
        ["accuracy", "head_layer", "head"],
        ascending=[False, True, True],
    ).head(int(n))
    return [
        (int(r.head_layer), int(r.head), float(r.accuracy))
        for r in q.itertuples()
    ]


def vote_relation(sample_relation_scores, vote_heads, head_layers):
    """
    Majority vote over per-head argmax. Tie -> sum cosine support.
    sample_relation_scores [nL,H,4]
    """
    li = {H: i for i, H in enumerate(head_layers)}
    votes = Counter()
    support = np.zeros(4, dtype=np.float64)

    for H, h, _acc in vote_heads:
        s4 = np.asarray(sample_relation_scores[li[H], h], dtype=np.float64)
        r = int(np.argmax(s4))
        votes[r] += 1
        support += s4

    maxv = max(votes.values())
    tied = [r for r, c in votes.items() if c == maxv]
    rid = tied[0] if len(tied) == 1 else max(tied, key=lambda r: support[r])

    return REL[rid], rid, {REL[r]: int(votes.get(r, 0)) for r in range(4)}


# =============================================================================
# Spatial ranking -> reverse causal selection
# =============================================================================

def aggregate_layer_rankings(
    sample,
    bank_name,
    relation,
    selected_heads,
    head_layers,
    head_input_offset,
    head_weight_mode,
    positive_only,
):
    """
    Returns:
      layer_rankings[source_layer] = ordered rows
      all_rows = rows across exact states

    Scores are converted to within-head percentiles BEFORE head aggregation.
    Therefore layers remain scale-comparable only as percentiles, not raw logits.
    """
    rid = RID[relation]
    positions = list(map(int, sample["positions"]))
    proj = sample["proj_syn"] if bank_name == "syn_rg" else sample["proj_coco"]

    li = {H: i for i, H in enumerate(head_layers)}
    layer_rankings = {}
    all_rows = []

    for H in head_layers:
        heads = selected_heads.get(H, [])
        if not heads:
            continue

        head_pct = []
        head_raw = []
        weights = []
        names = []

        for h, acc in heads:
            raw = np.asarray(proj[li[H], h, :, rid], dtype=np.float64)
            if positive_only:
                raw = np.maximum(raw, 0.0)

            pct = rank_percentiles(raw)

            w = 1.0
            if head_weight_mode == "accuracy":
                w = max(float(acc) - 0.25, EPS)

            head_pct.append(pct)
            head_raw.append(raw)
            weights.append(w)
            names.append(hname(H, h))

        P = np.stack(head_pct, axis=0)
        R = np.stack(head_raw, axis=0)
        W = np.asarray(weights, dtype=np.float64)

        score = np.average(P, axis=0, weights=W)
        raw_score = np.average(R, axis=0, weights=W)

        S = H - int(head_input_offset)
        rows = []
        for j, p in enumerate(positions):
            row = {
                "source_layer": S,
                "head_layer": H,
                "position": int(p),
                "score": float(score[j]),
                "raw_score": float(raw_score[j]),
                "heads": ",".join(names),
            }
            rows.append(row)
            all_rows.append(dict(row))

        rows.sort(
            key=lambda x: (
                float(x["score"]),
                float(x["raw_score"]),
                -int(x["position"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, 1):
            row["layer_rank"] = rank
        layer_rankings[S] = rows

    return layer_rankings, all_rows


def select_oracle_layer_budget(layer_rankings, target):
    """
    Direct inverse of matched-budget association:
    if target has m causal states at L, take spatial Top-m at L.
    """
    out = []
    for L, target_pos in target["by_layer"].items():
        m = len(target_pos)
        ranked = layer_rankings.get(int(L), [])
        out.extend(dict(r) for r in ranked[:m])
    return out


def select_fixed_layer_budget(layer_rankings, allocation):
    out = []
    for L, m in allocation.items():
        ranked = layer_rankings.get(int(L), [])
        out.extend(dict(r) for r in ranked[: int(m)])
    return out


def select_global_unique(all_rows, K):
    """
    Scores are already within-layer percentile aggregates.
    Same token position across layers: retain the strongest exact-state score.
    """
    best = {}
    for r in all_rows:
        p = int(r["position"])
        if (
            p not in best
            or (float(r["score"]), float(r["raw_score"]))
            > (float(best[p]["score"]), float(best[p]["raw_score"]))
        ):
            best[p] = dict(r)

    ranked = sorted(
        best.values(),
        key=lambda x: (
            float(x["score"]),
            float(x["raw_score"]),
            -int(x["source_layer"]),
            -int(x["position"]),
        ),
        reverse=True,
    )
    return ranked[: int(K)]


def recovery_metrics(selected, target):
    ps = {
        (int(r["source_layer"]), int(r["position"]))
        for r in selected
    }
    pp = {int(r["position"]) for r in selected}
    ts = target["states"]
    tp = target["positions"]

    ex = len(ps & ts)
    po = len(pp & tp)

    return {
        "selected_N": len(ps),
        "target_N": len(ts),
        "target_position_N": len(tp),
        "exact_hits": ex,
        "position_hits": po,
        "exact_recall": safe_div(ex, len(ts)),
        "exact_precision": safe_div(ex, len(ps)),
        "position_recall": safe_div(po, len(tp)),
        "position_precision": safe_div(po, len(pp)),
    }


def association_metrics(layer_rankings, target):
    """
    Original direction of the association:
    where do known causal states rank in the spatial source ranking?
    """
    pcts = []
    top10 = []
    top05 = []

    for L, target_pos in target["by_layer"].items():
        rows = layer_rankings.get(int(L), [])
        if not rows:
            continue
        n = len(rows)
        score_by_pos = {int(r["position"]): float(r["score"]) for r in rows}

        ordered = [
            int(r["position"])
            for r in sorted(
                rows,
                key=lambda x: (
                    float(x["score"]),
                    float(x["raw_score"]),
                ),
                reverse=True,
            )
        ]
        rank = {p: i for i, p in enumerate(ordered)}
        n10 = max(1, int(math.ceil(0.10 * n)))
        n05 = max(1, int(math.ceil(0.05 * n)))
        s10 = set(ordered[:n10])
        s05 = set(ordered[:n05])

        for p in target_pos:
            if p not in rank:
                continue
            i = rank[p]
            pct = 1.0 if n == 1 else float((n - 1 - i) / (n - 1))
            pcts.append(pct)
            top10.append(p in s10)
            top05.append(p in s05)

    return {
        "N_target_states_ranked": len(pcts),
        "mean_spatial_percentile": safe_mean(pcts),
        "fraction_spatial_top10pct": safe_mean(float(x) for x in top10),
        "fraction_spatial_top05pct": safe_mean(float(x) for x in top05),
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
    heads_per_layer_values = parse_ints(a.heads_per_layer)
    select_ks = parse_ints(a.select_ks)

    if not head_layers:
        raise ValueError("--head-layers is empty")
    if not heads_per_layer_values:
        raise ValueError("--heads-per-layer is empty")
    if not select_ks:
        raise ValueError("--select-ks is empty")

    source_layers = [
        H - int(a.head_input_offset)
        for H in head_layers
    ]
    if int(a.head_input_offset) != 1:
        print(
            f"[warning] old association used offset=1; requested offset={a.head_input_offset}"
        )

    outdir = Path(a.output_dir)
    synth_cache = outdir / "synthetic_realgray_head_residuals.npz"
    token_cache_dir = outdir / "coco_realgray_token_projection_cache"

    if a.overwrite and outdir.exists():
        # Keep expensive caches unless explicit cache-overwrite flags are supplied.
        for p in list(outdir.iterdir()):
            if (
                p.name == synth_cache.name
                and not a.overwrite_synthetic_cache
            ):
                continue
            if (
                p.name == token_cache_dir.name
                and not a.overwrite_token_cache
            ):
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    outdir.mkdir(parents=True, exist_ok=True)
    token_cache_dir.mkdir(parents=True, exist_ok=True)

    if a.overwrite_synthetic_cache and synth_cache.exists():
        synth_cache.unlink()

    if a.overwrite_token_cache and token_cache_dir.exists():
        shutil.rmtree(token_cache_dir)
        token_cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0) Data / metadata / oracle targets.
    # -------------------------------------------------------------------------
    top_df, core_df = load_oracle_bank(a.oracle_bank_dir)
    top_lookup = target_lookup(top_df)
    core_lookup = target_lookup(core_df)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        if sid not in top_lookup or sid not in core_lookup:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta.sort(key=lambda x: x["sid"])
    calib, heldout = stratified_split(meta, a.calib_frac, a.split_seed)
    eval_meta = list(meta) if a.eval_scope == "all_data" else list(heldout)

    if a.max_eval_samples > 0:
        try:
            eval_meta = traj.stratified_cap(
                eval_meta,
                a.max_eval_samples,
                a.split_seed + 1,
            )
        except Exception:
            eval_meta = eval_meta[: int(a.max_eval_samples)]

    if a.require_eval_n and len(eval_meta) != int(a.require_eval_n):
        raise RuntimeError(
            f"Expected eval N={a.require_eval_n}, got {len(eval_meta)}"
        )

    calib_sids = [int(x["sid"]) for x in calib]
    eval_sids = [int(x["sid"]) for x in eval_meta]
    meta_by_sid = {int(x["sid"]): x for x in meta}

    # Calibration-only causal layer priors.
    layer_prior = {
        "top10": layer_prior_from_calibration(
            top_df, calib_sids, source_layers
        ),
        "core50": layer_prior_from_calibration(
            core_df, calib_sids, source_layers
        ),
    }

    prior_rows = []
    for target_name in ("top10", "core50"):
        for L in source_layers:
            prior_rows.append({
                "target": target_name,
                "source_layer": L,
                "calibration_prior": layer_prior[target_name][L],
            })
    pd.DataFrame(prior_rows).to_csv(
        outdir / "calibration_causal_layer_prior.csv",
        index=False,
    )

    synthetic_rows, synthetic_labels_path = load_synthetic_rows(a)

    # -------------------------------------------------------------------------
    # 1) Load model.
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

    try:
        print(f"Loading {spec.repo_id}", flush=True)
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
        cfg = oldscan.get_text_config(model)
        n_heads = int(cfg.num_attention_heads)
        hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            hidden_size = int(
                oldscan.resolve_o_proj(
                    oldscan.resolve_attn(decoder_layers[0])
                ).in_features
            )
        head_dim = hidden_size // n_heads

        for H in head_layers:
            if not (0 <= H < len(decoder_layers)):
                raise ValueError(f"Bad head layer={H}")

        print("=" * 170)
        print("REVERSE ORIGINAL SPATIAL-HEAD ASSOCIATION -> CAUSAL TOKEN")
        print("=" * 170)
        print(
            f"model={a.model} | decoder={decoder_path} | heads={n_heads} "
            f"| head_dim={head_dim}"
        )
        print(
            f"Synthetic Real-Gray source N={len(synthetic_rows)} | "
            f"COCO calibration N={len(calib)} | eval N={len(eval_meta)}"
        )
        print(
            f"head layers={head_layers} -> causal source layers={source_layers}"
        )
        print(
            "Spatial score is EXACT old Real-Gray A*V projection; "
            "only its direction bank / relation routing are varied."
        )
        print()

        # ---------------------------------------------------------------------
        # 2) Synthetic-400 REAL-GRAY head residuals -> frozen directions.
        # ---------------------------------------------------------------------
        if synth_cache.exists():
            syn_sid, syn_y, syn_X, cached_layers = load_residual_cache(
                synth_cache
            )
            if cached_layers != head_layers:
                raise RuntimeError(
                    f"Synthetic cache head layers={cached_layers}, "
                    f"requested={head_layers}. Use --overwrite-synthetic-cache."
                )
            print(
                f"[synthetic] reusing Real-Gray cache {synth_cache} "
                f"N={len(syn_y)} shape={syn_X.shape}"
            )
        else:
            syn_sids = []
            syn_labels = []
            syn_resid = []
            syn_errors = []

            for rec in tqdm(
                synthetic_rows,
                desc="SYNTHETIC Real-Gray head residuals",
            ):
                real = gray = rb = gb = None
                try:
                    real = Image.open(rec["image_path"]).convert("RGB")
                    gray = make_gray(real, a.gray_value)

                    rb = base.make_question_batch(
                        processor=processor,
                        image=real,
                        question_text=rec["question_text"],
                        device=torch.device(a.device),
                    )
                    gb = base.make_question_batch(
                        processor=processor,
                        image=gray,
                        question_text=rec["question_text"],
                        device=torch.device(a.device),
                    )

                    ids = rb["input_ids"][0].detach().cpu().tolist()
                    spos, rpos = oldscan.object_positions(
                        processor,
                        ids,
                        rec["subject"],
                        rec["reference"],
                    )

                    rc = oldscan.run_capture(
                        model,
                        decoder_layers,
                        rb,
                        head_layers,
                        need_attn=False,
                    )
                    gc_ = oldscan.run_capture(
                        model,
                        decoder_layers,
                        gb,
                        head_layers,
                        need_attn=False,
                    )

                    arr = []
                    for H in head_layers:
                        arr.append(
                            all_head_delta_from_caps(
                                rc, gc_, H, spos, rpos, a.direction_pool
                            )
                        )

                    syn_sids.append(int(rec["sid"]))
                    syn_labels.append(rec["relation"])
                    syn_resid.append(np.stack(arr, axis=0))

                except Exception as exc:
                    syn_errors.append({
                        "sid": int(rec["sid"]),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-10:],
                    })
                    tqdm.write(
                        f"[synthetic ERROR] sid={rec['sid']} "
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    if real is not None:
                        with contextlib.suppress(Exception):
                            real.close()
                    if gray is not None:
                        with contextlib.suppress(Exception):
                            gray.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not syn_resid:
                raise RuntimeError("Synthetic Real-Gray extraction produced zero rows.")

            syn_sid = np.asarray(syn_sids, dtype=int)
            syn_y = np.asarray(syn_labels, dtype=object)
            syn_X = np.stack(syn_resid).astype(np.float32)

            save_residual_cache(
                synth_cache,
                syn_sid,
                syn_y,
                syn_X,
                head_layers,
                note=(
                    "Synthetic-400 PRE-W_O subject-reference REAL-GRAY "
                    "head residuals using exact COCO probe prompt."
                ),
            )
            (outdir / "synthetic_extraction_errors.json").write_text(
                json.dumps(syn_errors, indent=2),
                encoding="utf-8",
            )

        syn_center, syn_dirs = fit_codebook(syn_X, syn_y)

        # ---------------------------------------------------------------------
        # 3) COCO calibration REAL-GRAY head residuals.
        #    First pass requires no attention matrices.
        # ---------------------------------------------------------------------
        calib_resid_path = outdir / "coco_calibration_realgray_head_residuals.npz"

        if calib_resid_path.exists() and not a.overwrite_token_cache:
            cal_sid, cal_y, cal_X, cached_layers = load_residual_cache(
                calib_resid_path
            )
            if cached_layers != head_layers:
                raise RuntimeError(
                    f"Calibration cache layers={cached_layers}; "
                    "use --overwrite-token-cache."
                )
            print(
                f"[COCO calib] reusing residual cache N={len(cal_y)} "
                f"shape={cal_X.shape}"
            )
        else:
            cal_sids = []
            cal_labels = []
            cal_resid = []

            for m in tqdm(calib, desc="COCO CALIB Real-Gray residuals"):
                sid = int(m["sid"])
                real = gray = rb = gb = None
                try:
                    real = base.record_image(rec_by_sid[sid])
                    if hasattr(real, "convert"):
                        real = real.convert("RGB")
                    gray = make_gray(real, a.gray_value)

                    rb = base.make_question_batch(
                        processor=processor,
                        image=real,
                        question_text=m["question_text"],
                        device=torch.device(a.device),
                    )
                    gb = base.make_question_batch(
                        processor=processor,
                        image=gray,
                        question_text=m["question_text"],
                        device=torch.device(a.device),
                    )

                    ids = rb["input_ids"][0].detach().cpu().tolist()
                    spos, rpos = oldscan.object_positions(
                        processor,
                        ids,
                        m["subject"],
                        m["reference"],
                    )

                    rc = oldscan.run_capture(
                        model, decoder_layers, rb, head_layers, need_attn=False
                    )
                    gc_ = oldscan.run_capture(
                        model, decoder_layers, gb, head_layers, need_attn=False
                    )

                    arr = []
                    for H in head_layers:
                        arr.append(
                            all_head_delta_from_caps(
                                rc, gc_, H, spos, rpos, a.direction_pool
                            )
                        )

                    cal_sids.append(sid)
                    cal_labels.append(m["gt"])
                    cal_resid.append(np.stack(arr, axis=0))

                finally:
                    if real is not None:
                        with contextlib.suppress(Exception):
                            real.close()
                    if gray is not None:
                        with contextlib.suppress(Exception):
                            gray.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            cal_sid = np.asarray(cal_sids, dtype=int)
            cal_y = np.asarray(cal_labels, dtype=object)
            cal_X = np.stack(cal_resid).astype(np.float32)

            save_residual_cache(
                calib_resid_path,
                cal_sid,
                cal_y,
                cal_X,
                head_layers,
                note="COCO calibration PRE-W_O Real-Gray head residuals.",
            )

        coco_center, coco_dirs = fit_codebook(cal_X, cal_y)

        # ---------------------------------------------------------------------
        # 4) Direction alignment + calibration head accuracy for BOTH banks.
        # ---------------------------------------------------------------------
        align_df = compare_direction_banks(
            syn_dirs,
            coco_dirs,
            head_layers,
        )
        align_df.to_csv(
            outdir / "direction_alignment_synRG_vs_cocoCalibRG.csv",
            index=False,
        )

        syn_cal_scores = codebook_scores(
            cal_X,
            syn_center,
            syn_dirs,
        )
        coco_cal_scores = codebook_scores(
            cal_X,
            coco_center,
            coco_dirs,
        )

        syn_acc_df = head_accuracy(
            syn_cal_scores,
            cal_y,
            head_layers,
        )
        syn_acc_df["bank"] = "syn_rg"

        coco_acc_df = head_accuracy(
            coco_cal_scores,
            cal_y,
            head_layers,
        )
        coco_acc_df["bank"] = "coco_calib_rg"

        acc_df = pd.concat(
            [syn_acc_df, coco_acc_df],
            ignore_index=True,
        )
        acc_df.to_csv(
            outdir / "coco_calibration_head_accuracy.csv",
            index=False,
        )

        selected_heads = {}
        vote_heads = {}

        selected_rows = []
        vote_rows = []

        for bank_name, df in (
            ("syn_rg", syn_acc_df),
            ("coco_calib_rg", coco_acc_df),
        ):
            selected_heads[bank_name] = {}
            for n in heads_per_layer_values:
                x = select_heads_per_layer(df, head_layers, n)
                selected_heads[bank_name][n] = x
                for H, hs in x.items():
                    for rank, (h, acc) in enumerate(hs, 1):
                        selected_rows.append({
                            "bank": bank_name,
                            "heads_per_layer": n,
                            "head_layer": H,
                            "source_layer": H - a.head_input_offset,
                            "rank_within_layer": rank,
                            "head": h,
                            "head_name": hname(H, h),
                            "calibration_accuracy": acc,
                        })

            vote_heads[bank_name] = global_top_heads(
                df,
                a.vote_top_n,
            )
            for rank, (H, h, acc) in enumerate(
                vote_heads[bank_name], 1
            ):
                vote_rows.append({
                    "bank": bank_name,
                    "vote_rank": rank,
                    "head_layer": H,
                    "head": h,
                    "head_name": hname(H, h),
                    "calibration_accuracy": acc,
                })

        pd.DataFrame(selected_rows).to_csv(
            outdir / "selected_heads_per_layer.csv",
            index=False,
        )
        pd.DataFrame(vote_rows).to_csv(
            outdir / "relation_vote_heads.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # 5) COCO evaluation Real-Gray A*V token projections.
        # ---------------------------------------------------------------------
        relation_rows = []
        reverse_rows = []
        assoc_rows = []
        selected_state_rows = []
        recon_rows = []

        for m in tqdm(eval_meta, desc="COCO reverse spatial -> causal"):
            sid = int(m["sid"])
            cache_p = token_cache_path(token_cache_dir, sid)

            if a.cache_token_projections and cache_p.exists():
                sample = load_token_projection_cache(cache_p)
                if sample["head_layers"] != head_layers:
                    raise RuntimeError(
                        f"Token cache layer mismatch sid={sid}; "
                        "use --overwrite-token-cache."
                    )
            else:
                real = gray = rb = gb = None
                try:
                    real = base.record_image(rec_by_sid[sid])
                    if hasattr(real, "convert"):
                        real = real.convert("RGB")
                    gray = make_gray(real, a.gray_value)

                    rb = base.make_question_batch(
                        processor=processor,
                        image=real,
                        question_text=m["question_text"],
                        device=torch.device(a.device),
                    )
                    gb = base.make_question_batch(
                        processor=processor,
                        image=gray,
                        question_text=m["question_text"],
                        device=torch.device(a.device),
                    )

                    ids = rb["input_ids"][0].detach().cpu().tolist()
                    spos, rpos = oldscan.object_positions(
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
                        p
                        for p in range(max(0, npos - 1))
                        if dyn.broad_category(cats[p]) != "visual"
                    ]
                    if not text_positions:
                        raise RuntimeError(
                            f"No eligible text tokens sid={sid}"
                        )

                    rc = oldscan.run_capture(
                        model,
                        decoder_layers,
                        rb,
                        head_layers,
                        need_attn=True,
                    )
                    gc_ = oldscan.run_capture(
                        model,
                        decoder_layers,
                        gb,
                        head_layers,
                        need_attn=True,
                    )

                    nL = len(head_layers)
                    nP = len(text_positions)

                    head_delta = np.full(
                        (nL, n_heads, head_dim),
                        np.nan,
                        dtype=np.float32,
                    )
                    relation_scores_syn = np.full(
                        (nL, n_heads, 4),
                        np.nan,
                        dtype=np.float32,
                    )
                    relation_scores_coco = np.full_like(
                        relation_scores_syn,
                        np.nan,
                    )
                    proj_syn = np.full(
                        (nL, n_heads, nP, 4),
                        np.nan,
                        dtype=np.float32,
                    )
                    proj_coco = np.full_like(proj_syn, np.nan)
                    recon_cos = np.full(
                        (nL, n_heads),
                        np.nan,
                        dtype=np.float32,
                    )

                    for li, H in enumerate(head_layers):
                        ps, hd, rcoss = (
                            direction_token_projection_all_relations(
                                real_cap=rc,
                                gray_cap=gc_,
                                Hlayer=H,
                                spos=spos,
                                rpos=rpos,
                                pool=a.direction_pool,
                                dirs_h4d=syn_dirs[li],
                                text_positions=text_positions,
                            )
                        )
                        pc, hd2, _ = (
                            direction_token_projection_all_relations(
                                real_cap=rc,
                                gray_cap=gc_,
                                Hlayer=H,
                                spos=spos,
                                rpos=rpos,
                                pool=a.direction_pool,
                                dirs_h4d=coco_dirs[li],
                                text_positions=text_positions,
                            )
                        )

                        if not np.allclose(hd, hd2, atol=1e-5, rtol=1e-4):
                            raise RuntimeError(
                                f"Head delta mismatch sid={sid} H={H}"
                            )

                        head_delta[li] = hd
                        recon_cos[li] = rcoss
                        proj_syn[li] = ps
                        proj_coco[li] = pc

                        qsyn = normalize(
                            hd - syn_center[li],
                            axis=-1,
                        )
                        qcoco = normalize(
                            hd - coco_center[li],
                            axis=-1,
                        )

                        relation_scores_syn[li] = np.einsum(
                            "hd,hrd->hr",
                            qsyn,
                            syn_dirs[li],
                            optimize=True,
                        )
                        relation_scores_coco[li] = np.einsum(
                            "hd,hrd->hr",
                            qcoco,
                            coco_dirs[li],
                            optimize=True,
                        )

                    sample = {
                        "sid": sid,
                        "head_layers": list(head_layers),
                        "positions": np.asarray(text_positions, dtype=int),
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
                        "head_delta": head_delta,
                        "relation_scores_syn": relation_scores_syn,
                        "relation_scores_coco": relation_scores_coco,
                        "proj_syn": proj_syn,
                        "proj_coco": proj_coco,
                        "recon_cos": recon_cos,
                    }

                    if a.cache_token_projections:
                        save_token_projection_cache(
                            cache_p,
                            sid=sid,
                            head_layers=head_layers,
                            positions=sample["positions"],
                            token_ids=sample["token_ids"],
                            tokens=sample["tokens"],
                            categories=sample["categories"],
                            broad_categories=sample["broad_categories"],
                            head_delta=head_delta,
                            relation_scores_syn=relation_scores_syn,
                            relation_scores_coco=relation_scores_coco,
                            proj_syn=proj_syn,
                            proj_coco=proj_coco,
                            recon_cos=recon_cos,
                        )

                finally:
                    if real is not None:
                        with contextlib.suppress(Exception):
                            real.close()
                    if gray is not None:
                        with contextlib.suppress(Exception):
                            gray.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Reconstruction audit.
            for li, H in enumerate(head_layers):
                recon_rows.append({
                    "sid": sid,
                    "head_layer": H,
                    "mean_reconstruction_cosine": safe_mean(
                        sample["recon_cos"][li]
                    ),
                    "min_reconstruction_cosine": float(
                        np.nanmin(sample["recon_cos"][li])
                    ),
                })

            # Relation votes for both direction banks.
            predicted = {}
            for bank_name in ("syn_rg", "coco_calib_rg"):
                rs = (
                    sample["relation_scores_syn"]
                    if bank_name == "syn_rg"
                    else sample["relation_scores_coco"]
                )
                pred, pred_rid, votes = vote_relation(
                    rs,
                    vote_heads[bank_name],
                    head_layers,
                )
                predicted[bank_name] = pred

                relation_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "bank": bank_name,
                    "prediction": pred,
                    "correct": pred == m["gt"],
                    **{f"votes_{r}": votes[r] for r in REL},
                })

            # -------------------------------------------------------------
            # Reverse experiments.
            # -------------------------------------------------------------
            for bank_name in ("coco_calib_rg", "syn_rg"):
                for hpL in heads_per_layer_values:
                    chosen_heads = selected_heads[bank_name][hpL]

                    for relation_mode in ("oracle_gt", "pred_vote"):
                        rel = (
                            m["gt"]
                            if relation_mode == "oracle_gt"
                            else predicted[bank_name]
                        )

                        layer_rankings, all_rows = aggregate_layer_rankings(
                            sample=sample,
                            bank_name=bank_name,
                            relation=rel,
                            selected_heads=chosen_heads,
                            head_layers=head_layers,
                            head_input_offset=a.head_input_offset,
                            head_weight_mode=a.head_weight,
                            positive_only=a.positive_only,
                        )

                        for target_name, target in (
                            ("top10", top_lookup[sid]),
                            ("core50", core_lookup[sid]),
                        ):
                            # Original association metric itself.
                            assoc = association_metrics(
                                layer_rankings,
                                target,
                            )
                            assoc_rows.append({
                                "sid": sid,
                                "gt": m["gt"],
                                "bank": bank_name,
                                "heads_per_layer": hpL,
                                "relation_mode": relation_mode,
                                "relation_used": rel,
                                "relation_correct": rel == m["gt"],
                                "target": target_name,
                                **assoc,
                            })

                            # A/C: exact per-sample oracle layer budget.
                            sel = select_oracle_layer_budget(
                                layer_rankings,
                                target,
                            )
                            met = recovery_metrics(sel, target)
                            reverse_rows.append({
                                "sid": sid,
                                "gt": m["gt"],
                                "bank": bank_name,
                                "heads_per_layer": hpL,
                                "relation_mode": relation_mode,
                                "relation_used": rel,
                                "relation_correct": rel == m["gt"],
                                "selection_mode": "oracle_layer_budget",
                                "K": len(sel),
                                "target": target_name,
                                **met,
                            })

                            # D: calibration-only layer prior.
                            for K in select_ks:
                                alloc = allocate_largest_remainder(
                                    layer_prior[target_name],
                                    K,
                                    source_layers,
                                )
                                sel = select_fixed_layer_budget(
                                    layer_rankings,
                                    alloc,
                                )
                                met = recovery_metrics(sel, target)
                                reverse_rows.append({
                                    "sid": sid,
                                    "gt": m["gt"],
                                    "bank": bank_name,
                                    "heads_per_layer": hpL,
                                    "relation_mode": relation_mode,
                                    "relation_used": rel,
                                    "relation_correct": rel == m["gt"],
                                    "selection_mode": "calib_layer_prior",
                                    "K": K,
                                    "target": target_name,
                                    **met,
                                })

                                # E: no target layer prior.
                                sel = select_global_unique(
                                    all_rows,
                                    K,
                                )
                                met = recovery_metrics(sel, target)
                                reverse_rows.append({
                                    "sid": sid,
                                    "gt": m["gt"],
                                    "bank": bank_name,
                                    "heads_per_layer": hpL,
                                    "relation_mode": relation_mode,
                                    "relation_used": rel,
                                    "relation_correct": rel == m["gt"],
                                    "selection_mode": "global_unique",
                                    "K": K,
                                    "target": target_name,
                                    **met,
                                })

                            # Save one compact candidate list for strongest
                            # practical config family: calibration prior K10.
                            if (
                                bank_name == "syn_rg"
                                and relation_mode == "pred_vote"
                                and hpL == max(heads_per_layer_values)
                            ):
                                alloc = allocate_largest_remainder(
                                    layer_prior[target_name],
                                    10,
                                    source_layers,
                                )
                                selected10 = select_fixed_layer_budget(
                                    layer_rankings,
                                    alloc,
                                )
                                for rank, rr in enumerate(
                                    sorted(
                                        selected10,
                                        key=lambda x: (
                                            x["score"], x["raw_score"]
                                        ),
                                        reverse=True,
                                    ),
                                    1,
                                ):
                                    p = int(rr["position"])
                                    try:
                                        j = list(sample["positions"]).index(p)
                                        tok = str(sample["tokens"][j])
                                        cat = str(sample["categories"][j])
                                        broad = str(
                                            sample["broad_categories"][j]
                                        )
                                    except Exception:
                                        tok = cat = broad = ""
                                    selected_state_rows.append({
                                        "sid": sid,
                                        "target_bank": target_name,
                                        "rank": rank,
                                        "source_layer": int(rr["source_layer"]),
                                        "head_layer": int(rr["head_layer"]),
                                        "position": p,
                                        "token": tok,
                                        "category": cat,
                                        "broad_category": broad,
                                        "spatial_percentile_score": float(
                                            rr["score"]
                                        ),
                                        "raw_spatial_score": float(
                                            rr["raw_score"]
                                        ),
                                        "relation_used": rel,
                                        "relation_correct": rel == m["gt"],
                                        "heads": rr["heads"],
                                        "is_exact_oracle_state": (
                                            (int(rr["source_layer"]), p)
                                            in target["states"]
                                        ),
                                        "is_oracle_position": (
                                            p in target["positions"]
                                        ),
                                    })

        # ---------------------------------------------------------------------
        # 6) Aggregate.
        # ---------------------------------------------------------------------
        relation_df = pd.DataFrame(relation_rows)
        reverse_df = pd.DataFrame(reverse_rows)
        assoc_df = pd.DataFrame(assoc_rows)
        recon_df = pd.DataFrame(recon_rows)
        selected_df = pd.DataFrame(selected_state_rows)

        relation_df.to_csv(
            outdir / "relation_vote_per_sample.csv",
            index=False,
        )
        reverse_df.to_csv(
            outdir / "reverse_recovery_per_sample.csv",
            index=False,
        )
        assoc_df.to_csv(
            outdir / "association_sanity_per_sample.csv",
            index=False,
        )
        recon_df.to_csv(
            outdir / "reconstruction_audit.csv",
            index=False,
        )
        selected_df.to_csv(
            outdir / "practical_synRG_predicted_selected_states.csv",
            index=False,
        )

        # Relation summary.
        relation_summary = (
            relation_df.groupby("bank", as_index=False)
            .agg(
                N=("sid", "nunique"),
                vote_accuracy=("correct", "mean"),
            )
        )
        relation_summary.to_csv(
            outdir / "relation_vote_summary.csv",
            index=False,
        )

        # Association summary.
        assoc_summary_rows = []
        keys = [
            "bank",
            "heads_per_layer",
            "relation_mode",
            "target",
        ]
        for key, g in assoc_df.groupby(keys):
            bank, hpL, rm, target = key
            assoc_summary_rows.append({
                "bank": bank,
                "heads_per_layer": hpL,
                "relation_mode": rm,
                "target": target,
                "N": int(g["sid"].nunique()),
                "relation_accuracy": float(
                    g["relation_correct"].mean()
                ),
                "mean_spatial_percentile": safe_mean(
                    g["mean_spatial_percentile"]
                ),
                "fraction_spatial_top10pct": safe_mean(
                    g["fraction_spatial_top10pct"]
                ),
                "fraction_spatial_top05pct": safe_mean(
                    g["fraction_spatial_top05pct"]
                ),
            })
        assoc_summary = pd.DataFrame(assoc_summary_rows).sort_values(
            [
                "target",
                "mean_spatial_percentile",
            ],
            ascending=[True, False],
        )
        assoc_summary.to_csv(
            outdir / "association_sanity_summary.csv",
            index=False,
        )

        # Reverse recovery summary.
        summary_rows = []
        keys = [
            "bank",
            "heads_per_layer",
            "relation_mode",
            "selection_mode",
            "K",
            "target",
        ]
        for key, g in reverse_df.groupby(keys):
            bank, hpL, rm, sm, K, target = key
            exact_hits = int(g["exact_hits"].sum())
            target_n = int(g["target_N"].sum())
            pos_hits = int(g["position_hits"].sum())
            target_pos_n = int(g["target_position_N"].sum())
            selected_n = int(g["selected_N"].sum())

            summary_rows.append({
                "bank": bank,
                "heads_per_layer": hpL,
                "relation_mode": rm,
                "selection_mode": sm,
                "K": K,
                "target": target,
                "N": int(g["sid"].nunique()),
                "relation_accuracy": float(
                    g["relation_correct"].mean()
                ),
                "micro_exact_recall": safe_div(
                    exact_hits, target_n
                ),
                "macro_exact_recall": safe_mean(
                    g["exact_recall"]
                ),
                "micro_exact_precision": safe_div(
                    exact_hits, selected_n
                ),
                "micro_position_recall": safe_div(
                    pos_hits, target_pos_n
                ),
                "macro_position_recall": safe_mean(
                    g["position_recall"]
                ),
                "micro_position_precision": safe_div(
                    pos_hits, selected_n
                ),
                "mean_exact_hits": safe_mean(
                    g["exact_hits"]
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
            outdir / "reverse_recovery_summary.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # 7) Human-readable report: order by the causal chain of ablations.
        # ---------------------------------------------------------------------
        lines = []
        lines.append("=" * 190)
        lines.append(
            "REVERSE ORIGINAL REAL-GRAY SPATIAL ASSOCIATION -> CAUSAL TOKENS"
        )
        lines.append("=" * 190)
        lines.append(
            f"eval_scope={a.eval_scope} | calibration N={len(calib)} | "
            f"eval N={len(eval_meta)}"
        )
        lines.append(
            "Synthetic directions: Real-Gray PRE-W_O object residual, exact COCO probe prompt."
        )
        lines.append(
            "Token spatial score: ORIGINAL Real-Gray A*V source contribution projected "
            "onto relation direction."
        )
        lines.append(
            f"head layers={head_layers} -> state layers={source_layers}"
        )
        lines.append("")

        lines.append("DIRECTION BANK ALIGNMENT")
        lines.append("-" * 190)
        for H in head_layers:
            q = align_df[align_df["head_layer"] == H]
            lines.append(
                f"H{H:02d}: mean cos(synRG,cocoCalibRG)="
                f"{q['cos_synRG_vs_cocoCalibRG'].mean():.4f}"
            )
        lines.append(
            f"ALL: mean cos={align_df['cos_synRG_vs_cocoCalibRG'].mean():.4f}"
        )
        lines.append("")

        lines.append("RELATION VOTE")
        lines.append("-" * 190)
        lines.append(
            relation_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")

        lines.append(
            "SANITY: WHERE DO KNOWN CAUSAL STATES RANK IN SPATIAL SOURCE RANKING?"
        )
        lines.append("-" * 190)
        lines.append(
            assoc_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")

        lines.append(
            "DIRECT INVERSION CEILING: ORACLE RELATION + ORACLE PER-LAYER BUDGET"
        )
        lines.append("-" * 190)
        q = summary[
            (summary["relation_mode"] == "oracle_gt")
            & (summary["selection_mode"] == "oracle_layer_budget")
        ].sort_values(
            ["target", "micro_exact_recall"],
            ascending=[True, False],
        )
        show = [
            "bank",
            "heads_per_layer",
            "target",
            "N",
            "micro_exact_recall",
            "micro_exact_precision",
            "micro_position_recall",
            "mean_exact_hits",
        ]
        lines.append(
            q[show].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")

        lines.append(
            "ISOLATE ROUTING ERROR: PREDICTED RELATION + ORACLE PER-LAYER BUDGET"
        )
        lines.append("-" * 190)
        q = summary[
            (summary["relation_mode"] == "pred_vote")
            & (summary["selection_mode"] == "oracle_layer_budget")
        ].sort_values(
            ["target", "micro_exact_recall"],
            ascending=[True, False],
        )
        show2 = [
            "bank",
            "heads_per_layer",
            "target",
            "relation_accuracy",
            "micro_exact_recall",
            "micro_exact_precision",
            "micro_position_recall",
        ]
        lines.append(
            q[show2].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")

        lines.append(
            "PRACTICAL REVERSE: synRG + PREDICTED RELATION + CALIBRATION LAYER PRIOR"
        )
        lines.append("-" * 190)
        q = summary[
            (summary["bank"] == "syn_rg")
            & (summary["relation_mode"] == "pred_vote")
            & (summary["selection_mode"] == "calib_layer_prior")
        ].sort_values(
            ["target", "micro_exact_recall"],
            ascending=[True, False],
        )
        show3 = [
            "heads_per_layer",
            "K",
            "target",
            "relation_accuracy",
            "micro_exact_recall",
            "micro_exact_precision",
            "micro_position_recall",
            "micro_position_precision",
        ]
        lines.append(
            q[show3].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")

        lines.append(
            "NO LAYER PRIOR: synRG + PREDICTED RELATION + GLOBAL_UNIQUE"
        )
        lines.append("-" * 190)
        q = summary[
            (summary["bank"] == "syn_rg")
            & (summary["relation_mode"] == "pred_vote")
            & (summary["selection_mode"] == "global_unique")
        ].sort_values(
            ["target", "micro_exact_recall"],
            ascending=[True, False],
        )
        lines.append(
            q[show3].to_string(
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

        # ---------------------------------------------------------------------
        # 8) Metadata.
        # ---------------------------------------------------------------------
        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "synthetic_labels": str(synthetic_labels_path),
            "synthetic_N": int(len(syn_y)),
            "calibration_N": int(len(calib)),
            "eval_N": int(len(eval_meta)),
            "eval_scope": a.eval_scope,
            "head_layers": head_layers,
            "source_layers": source_layers,
            "heads_per_layer": heads_per_layer_values,
            "vote_top_n": int(a.vote_top_n),
            "select_ks": select_ks,
            "gray_value": int(a.gray_value),
            "direction_pool": a.direction_pool,
            "spatial_score": (
                "EXACT original Direction-head source score: "
                "c_RG=[A_real(sub)-A_real(ref)]V_real - "
                "[A_gray(sub)-A_gray(ref)]V_gray; "
                "S=<c_RG,d_relation>."
            ),
            "synthetic_direction_definition": (
                "PRE-W_O subject-reference Real-Gray residual, centered over "
                "Synthetic-400 and relation mean minus global center."
            ),
            "layer_alignment": (
                "causal decoder block-output L -> attention head H=L+1"
            ),
            "oracle_layer_budget_meaning": (
                "Diagnostic direct inverse of the original matched-budget "
                "association. Uses target per-layer counts but never target positions."
            ),
            "calib_layer_prior_meaning": (
                "Layer allocation probabilities estimated only from calibration "
                "SIDs' oracle causal bank."
            ),
            "important_caveat": (
                "all_data is a mechanistic diagnostic because calibration SIDs "
                "are included in evaluation. Use heldout for disjoint evaluation."
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("Saved:", outdir)

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
