#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_nonoracle_how_policy_sweep_v1_1.py

Purpose
=======
We keep WHERE fixed and optimize only HOW.

The fixed WHERE template is the global canonical Top-K scaffold discovered by
eval_global_fixed_l26_top10_trajectory_gating_v1.py.  Importantly, this script
can apply that SAME template to samples that were NOT present in that run, so a
Top10 template discovered on N=80 can be evaluated on all COCO_two N=440.

For every sample and every resolved global slot p, for L20..L26:

    a[L,p] = h_REAL[L,p] - h_REAL[L-1,p]

For each candidate relation r:

    q_r[L,p] = < a[L,p], d S_r / d h[L,p] >

These four q_r are GT-free.  They are computed once and reused by all policies.

Direction-Head posterior
========================
This script reconstructs the full 4-way posterior P(r) from the existing
Synthetic-400 -> COCO Direction-Head selector, rather than using only its argmax.

It loads:
    <direction-selector-dir>/synthetic_fitted_direction_codebook.npz
    <direction-selector-dir>/source_oof_head_reliability.csv
and the existing COCO Direction-Head target residual cache.

Default selector is exactly Top10 equal-weight, non-layer-balanced, T=1,
matching `top10_equal`.

Policies tested
===============

A. Existing hard-route method
-----------------------------
    r_hat = argmax_r P(r)

For each relation hypothesis r:
    foil(r) = strongest CLEAN sequence-score candidate != r
    B_r     = q_r - q_foil(r)

Existing method:
    B_hard = B_{r_hat}

Variants:
    hard_all
    hard_disagree                      # only if r_hat != baseline answer
    hard conflict-margin triggers
    alpha sweep
    positive-only
    negative-cancel-only
    update-level |B_hard| thresholds
    per-sample Top-M |B_hard| updates

B. Soft relation posterior
---------------------------
Instead of forcing a single relation:

    B_soft = sum_r (P(r) - 1/4) q_r

Uniform P gives zero automatically.

C. Posterior expected relation-margin
-------------------------------------
    B_post = sum_r P(r) B_r

This averages relation-conditioned decision effects directly.

D. Relation-sign consensus
---------------------------
Weighted sign vote:
    B_vote = sum_r P(r) sign(B_r)

Top-2 consensus:
    take the two highest posterior relations;
    if sign(B_r1) == sign(B_r2), use that sign;
    otherwise ABSTAIN.

This asks the actual question we care about:
    do plausible relations AGREE on what to do with this update,
rather than requiring exact relation classification.

E. Conflict margin
------------------
For repair triggering, use:

    C = P(r_hat) - P(r_baseline)

only when r_hat != r_baseline.

This is more directly relevant than ordinary top1-top2 confidence:
"Is the independent spatial belief strong enough to overturn the current answer?"

F. Update-level confidence / Top-M
----------------------------------
Use |B| directly:
    |B| > tau
or keep only the Top-M largest |B| updates per sample.

G. Scale sweep
--------------
Default:
    alpha = 0.1, 0.25, 0.35, 0.5, 0.75

Oracle ceiling
==============
For analysis only:

    B_oracle = B_GT

This uses GT and is NEVER used by non-oracle policies.

Full-440 generalization diagnostic
==================================
If the global template was originally discovered on N=80, this script reads its
discovery SIDs and reports generation separately for:

    template_seen       (the original discovery samples)
    template_unseen     (all other samples)

Thus a 440 run directly tells us whether the fixed WHERE scaffold generalizes.

Expected repository files
=========================
Run from AdaptVis/llava16 root and keep these helper scripts available:
    eval_real_causal_token_update_gating_v1.py
    eval_l26_horizontal_top7_real_update_trajectory_v1.py
    eval_global_fixed_l26_top10_trajectory_gating_v1.py
    eval_direction_head_selector_synthetic400_to_coco440_v2.py

Recommended broad N80 sweep
===========================
CUDA_VISIBLE_DEVICES=0 python -u eval_nonoracle_how_policy_sweep_v1_1.py \
  --model qwen-3b \
  --global-template-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --direction-selector-dir output/qwen3b_direction_selector_syn400_to_coco440 \
  --target-direction-dir output/qwen3b_head_object_residual_direction \
  --selector-k 10 \
  --update-layers 20-26 \
  --alphas 0.1,0.25,0.35,0.5,0.75 \
  --conflict-thresholds 0.02,0.05,0.10 \
  --update-thresholds 0.005,0.01,0.02,0.05 \
  --top-ms 10,20,30,45 \
  --eval-max-samples 80 \
  --policy-set all \
  --output-dir output/qwen3b_nonoracle_how_sweep_n80_v1 \
  --overwrite

Full 440
========
Same command, but:
    --eval-max-samples 0
    --output-dir output/qwen3b_nonoracle_how_sweep_all440_v1

Use --policy-set core for a somewhat cheaper 440 run while still covering every
optimization family.
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
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26
import eval_global_fixed_l26_top10_trajectory_gating_v1 as gfix


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# Standalone Direction-Head selector helpers
# =============================================================================
#
# These functions are copied/reimplemented from
# eval_direction_head_selector_synthetic400_to_coco440_v2.py so this sweep
# does NOT require that Python file to exist in the repository.  It only needs
# the selector OUTPUT files:
#
#   synthetic_fitted_direction_codebook.npz
#   source_oof_head_reliability.csv
#
# plus the target Direction-Head cache relation_vectors.npz.


def _dh_resolve_npz(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p

    candidates = [
        p / "relation_vectors.npz",
        p / "direction_vectors.npz",
    ]
    for q in candidates:
        if q.exists():
            return q

    npzs = sorted(p.glob("*.npz")) if p.exists() else []
    if len(npzs) == 1:
        return npzs[0]

    raise FileNotFoundError(
        f"Could not resolve target Direction-Head NPZ from: {p}. "
        f"Tried {[str(x) for x in candidates]}"
    )


def _dh_load_target_cache(path_or_dir):
    p = _dh_resolve_npz(path_or_dir)
    z = np.load(p, allow_pickle=True)

    required = {"relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{p} missing arrays: {sorted(missing)}; keys={list(z.files)}"
        )

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray(
        [canon_rel(x) for x in z["relation"]],
        dtype=object,
    )

    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    else:
        sid = np.arange(len(y), dtype=int)

    valid = np.isin(y, np.asarray(REL, dtype=object))
    return p, sid[valid], y[valid], X[valid]


def _dh_normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def _dh_softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    x = x / max(float(temperature), 1e-8)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(
        e.sum(axis=axis, keepdims=True),
        EPS,
    )


def _dh_score_all_heads(X, center, dirs):
    """
    X      : [N,L,H,D]
    center : [L,H,D]
    dirs   : [L,H,R,D]
    return : [N,L,H,R]
    """
    Xc = X - center[None, :, :, :]
    Xn = _dh_normalize(Xc, axis=-1)
    return np.einsum(
        "nlhd,lhrd->nlhr",
        Xn,
        dirs,
        optimize=True,
    )


def _dh_ensemble_probs(
    scores,
    heads,
    reliability,
    temperature,
    weighted,
    layer_balanced,
):
    """
    Reimplementation of the selector's ensemble_probs().

    scores: [N,L,H,R], per-head cosine relation scores.

    First convert each head to a relation probability via softmax.
    Non-layer-balanced:
        weighted/equal average directly over selected heads.
    Layer-balanced:
        average selected heads inside each layer first, then average layers
        equally so a layer with more selected heads cannot dominate.
    """
    probs = _dh_softmax(
        scores,
        axis=-1,
        temperature=temperature,
    )
    N = scores.shape[0]

    if not layer_balanced:
        out = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0

        for l, h in heads:
            w = (
                max(float(reliability[l, h]) - 0.25, 0.0)
                if weighted
                else 1.0
            )
            if w <= 0:
                continue

            out += w * probs[:, l, h, :]
            denom += w

        if denom <= EPS:
            return np.mean(
                np.stack(
                    [probs[:, l, h, :] for l, h in heads],
                    axis=0,
                ),
                axis=0,
            )

        return out / denom

    # Layer-balanced branch.
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(int(l), []).append(int(h))

    layer_outputs = []

    for l, hs in sorted(by_layer.items()):
        layer_out = np.zeros(
            (N, len(REL)),
            dtype=np.float64,
        )
        denom = 0.0

        for h in hs:
            w = (
                max(float(reliability[l, h]) - 0.25, 0.0)
                if weighted
                else 1.0
            )
            if w <= 0:
                continue

            layer_out += w * probs[:, l, h, :]
            denom += w

        if denom <= EPS:
            layer_out = np.mean(
                np.stack(
                    [probs[:, l, h, :] for h in hs],
                    axis=0,
                ),
                axis=0,
            )
        else:
            layer_out /= denom

        layer_outputs.append(layer_out)

    if not layer_outputs:
        raise RuntimeError("No selected Direction Heads for ensemble.")

    return np.mean(
        np.stack(layer_outputs, axis=0),
        axis=0,
    )


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument(
        "--global-template-dir",
        required=True,
        help=(
            "Completed global fixed scaffold run. Only global_topk_slots.csv is "
            "required for applying the SAME scaffold to new/full-440 samples."
        ),
    )
    p.add_argument("--prior-real-update-dir", required=True)

    p.add_argument(
        "--direction-selector-dir",
        required=True,
        help=(
            "Existing Synthetic-400 Direction-Head selector output containing "
            "synthetic_fitted_direction_codebook.npz and "
            "source_oof_head_reliability.csv."
        ),
    )
    p.add_argument(
        "--target-direction-dir",
        required=True,
        help="Existing COCO Direction-Head relation_vectors.npz or its directory.",
    )

    p.add_argument("--selector-k", type=int, default=10)
    p.add_argument(
        "--selector-weighted",
        action="store_true",
        help="Use source-reliability weighting instead of equal Top-K heads.",
    )
    p.add_argument(
        "--selector-layer-balanced",
        action="store_true",
    )
    p.add_argument("--selector-temperature", type=float, default=1.0)

    p.add_argument("--update-layers", default="20-26")

    p.add_argument("--alphas", default="0.1,0.25,0.35,0.5,0.75")
    p.add_argument(
        "--conflict-thresholds",
        default="0.02,0.05,0.10",
    )
    p.add_argument(
        "--update-thresholds",
        default="0.005,0.01,0.02,0.05",
    )
    p.add_argument("--top-ms", default="10,20,30,45")

    p.add_argument(
        "--policy-set",
        default="all",
        choices=["core", "all"],
        help=(
            "core = every optimization family with fewer duplicate variants; "
            "all = broader sweep."
        ),
    )

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all prior-run samples (normally 440).",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    out = set()
    for x in str(text).split(","):
        x = x.strip().upper().replace("L", "")
        if not x:
            continue
        if "-" in x:
            a, b = x.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(x))
    return sorted(out)


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


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
        "bottom": "below",
    }.get(s, s)


def tag_float(x: float) -> str:
    return (
        f"{float(x):.6f}"
        .rstrip("0")
        .rstrip(".")
        .replace("-", "m")
        .replace(".", "p")
    )


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def weighted_accuracy(correct, weights):
    c = np.asarray(correct, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(c) & np.isfinite(w) & (w >= 0)
    if not np.any(ok):
        return float("nan")
    den = float(w[ok].sum())
    if den <= EPS:
        return float("nan")
    return float(np.sum(c[ok] * w[ok]) / den)


# =============================================================================
# Policy representation
# =============================================================================

@dataclass(frozen=True)
class Policy:
    name: str
    score_kind: str
    trigger_kind: str = "all"
    trigger_value: float = 0.0
    action: str = "signed"
    scale: float = 0.5
    abs_threshold: float = 0.0
    top_m: int = 0
    oracle: bool = False


def build_policies(
    *,
    alphas,
    conflict_thresholds,
    update_thresholds,
    top_ms,
    policy_set,
):
    policies = []

    def add(
        name,
        score_kind,
        trigger_kind="all",
        trigger_value=0.0,
        action="signed",
        scale=0.5,
        abs_threshold=0.0,
        top_m=0,
        oracle=False,
    ):
        policies.append(
            Policy(
                name=name,
                score_kind=score_kind,
                trigger_kind=trigger_kind,
                trigger_value=float(trigger_value),
                action=action,
                scale=float(scale),
                abs_threshold=float(abs_threshold),
                top_m=int(top_m),
                oracle=bool(oracle),
            )
        )

    # -------------------------------------------------------------
    # A) Existing hard-route baseline.
    # -------------------------------------------------------------
    add("hard_all_a0p5", "hard", scale=0.5)

    for alpha in alphas:
        add(
            f"hard_disagree_a{tag_float(alpha)}",
            "hard",
            trigger_kind="disagree",
            scale=alpha,
        )

    # -------------------------------------------------------------
    # B) Conflict-margin trigger against baseline answer.
    # -------------------------------------------------------------
    for t in conflict_thresholds:
        add(
            f"hard_conflict_{tag_float(t)}_a0p5",
            "hard",
            trigger_kind="conflict",
            trigger_value=t,
            scale=0.5,
        )

    # -------------------------------------------------------------
    # C) Update-level |B| threshold and Top-M.
    # -------------------------------------------------------------
    for t in update_thresholds:
        add(
            f"hard_disagree_abs_{tag_float(t)}_a0p5",
            "hard",
            trigger_kind="disagree",
            scale=0.5,
            abs_threshold=t,
        )

    for m in top_ms:
        add(
            f"hard_disagree_top{int(m)}_a0p5",
            "hard",
            trigger_kind="disagree",
            scale=0.5,
            top_m=m,
        )

    # -------------------------------------------------------------
    # D) Asymmetric positive / negative intervention.
    # -------------------------------------------------------------
    add(
        "hard_disagree_positive_only_a0p5",
        "hard",
        trigger_kind="disagree",
        action="positive",
        scale=0.5,
    )
    add(
        "hard_disagree_negative_cancel_a0p5",
        "hard",
        trigger_kind="disagree",
        action="negative",
        scale=0.5,
    )

    # -------------------------------------------------------------
    # E) Soft posterior / expected B / sign consensus.
    # -------------------------------------------------------------
    add("soft_all_a0p5", "soft", scale=0.5)
    add(
        "soft_disagree_a0p5",
        "soft",
        trigger_kind="disagree",
        scale=0.5,
    )

    add("postB_all_a0p5", "postB", scale=0.5)
    add(
        "postB_disagree_a0p5",
        "postB",
        trigger_kind="disagree",
        scale=0.5,
    )

    add("vote_all_a0p5", "vote", scale=0.5)
    add(
        "vote_disagree_a0p5",
        "vote",
        trigger_kind="disagree",
        scale=0.5,
    )

    add("top2cons_all_a0p5", "top2", scale=0.5)
    add(
        "top2cons_disagree_a0p5",
        "top2",
        trigger_kind="disagree",
        scale=0.5,
    )

    # A few confidence-limited posterior variants.
    for kind in ("soft", "postB"):
        for m in (20, 30):
            add(
                f"{kind}_disagree_top{m}_a0p5",
                kind,
                trigger_kind="disagree",
                scale=0.5,
                top_m=m,
            )

    if policy_set == "all":
        # Apply conflict trigger to alternative posterior methods.
        for t in conflict_thresholds:
            for kind in ("soft", "postB", "vote", "top2"):
                add(
                    f"{kind}_conflict_{tag_float(t)}_a0p5",
                    kind,
                    trigger_kind="conflict",
                    trigger_value=t,
                    scale=0.5,
                )

        # Small alpha sweep for the two strongest soft-style alternatives.
        for alpha in alphas:
            if abs(float(alpha) - 0.5) < 1e-12:
                continue
            add(
                f"soft_disagree_a{tag_float(alpha)}",
                "soft",
                trigger_kind="disagree",
                scale=alpha,
            )
            add(
                f"postB_disagree_a{tag_float(alpha)}",
                "postB",
                trigger_kind="disagree",
                scale=alpha,
            )

    # Oracle ceiling. Same fixed WHERE, GT only for sign.
    add(
        "oracle_signed_reference_a0p5",
        "oracle",
        scale=0.5,
        oracle=True,
    )

    # Dedupe names.
    out = []
    seen = set()
    for p in policies:
        if p.name not in seen:
            out.append(p)
            seen.add(p.name)
    return out


# =============================================================================
# Fixed global template
# =============================================================================

def load_global_template(template_dir: Path):
    p = template_dir / "global_topk_slots.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    top = pd.read_csv(p)
    if "global_rank" not in top.columns:
        top = top.copy()
        top["global_rank"] = np.arange(1, len(top) + 1)

    top = top.sort_values("global_rank").copy()

    discovery_sids = set()
    disc = template_dir / "discovery_sample_l26_topk.csv"
    if disc.exists():
        d = pd.read_csv(disc)
        if "sid" in d.columns:
            discovery_sids = set(
                pd.to_numeric(d["sid"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

    return top, discovery_sids


# =============================================================================
# Direction posterior reconstruction
# =============================================================================

def load_direction_posterior(
    *,
    selector_dir: Path,
    target_direction_dir: Path,
    selector_k: int,
    weighted: bool,
    layer_balanced: bool,
    temperature: float,
):
    codebook_path = selector_dir / "synthetic_fitted_direction_codebook.npz"
    reliability_path = selector_dir / "source_oof_head_reliability.csv"

    if not codebook_path.exists():
        raise FileNotFoundError(codebook_path)
    if not reliability_path.exists():
        raise FileNotFoundError(reliability_path)

    z = np.load(codebook_path, allow_pickle=True)
    center = np.asarray(z["center"], dtype=np.float32)
    directions = np.asarray(z["directions"], dtype=np.float32)

    target_path, target_sid, target_y, target_X = _dh_load_target_cache(
        target_direction_dir
    )

    if target_X.shape[1:] != center.shape:
        raise RuntimeError(
            f"Target residual shape {target_X.shape[1:]} != "
            f"source center shape {center.shape}"
        )

    target_scores = _dh_score_all_heads(
        target_X,
        center,
        directions,
    )

    rel_df = pd.read_csv(reliability_path)
    for c in ("rank", "layer", "head"):
        rel_df[c] = pd.to_numeric(rel_df[c], errors="raise").astype(int)
    rel_df["source_oof_accuracy"] = pd.to_numeric(
        rel_df["source_oof_accuracy"], errors="raise"
    ).astype(float)
    rel_df = rel_df.sort_values("rank")

    selected_rows = rel_df.head(int(selector_k))
    if len(selected_rows) < int(selector_k):
        raise RuntimeError(
            f"Requested selector Top{selector_k}, only {len(selected_rows)} heads."
        )

    heads = [
        (int(r.layer), int(r.head))
        for r in selected_rows.itertuples()
    ]

    reliability = np.full(
        center.shape[:2],
        0.25,
        dtype=np.float32,
    )
    for r in rel_df.itertuples():
        l, h = int(r.layer), int(r.head)
        if 0 <= l < reliability.shape[0] and 0 <= h < reliability.shape[1]:
            reliability[l, h] = float(r.source_oof_accuracy)

    probs = _dh_ensemble_probs(
        scores=target_scores,
        heads=heads,
        reliability=reliability,
        temperature=float(temperature),
        weighted=bool(weighted),
        layer_balanced=bool(layer_balanced),
    )

    if probs.shape != (len(target_sid), len(REL)):
        raise RuntimeError(f"Unexpected posterior shape: {probs.shape}")

    posterior = {}
    rows = []

    for i, sid in enumerate(target_sid):
        sid = int(sid)
        pp = np.asarray(probs[i], dtype=np.float64)
        route = REL[int(np.argmax(pp))]
        order = np.argsort(-pp)
        margin = float(pp[order[0]] - pp[order[1]])

        posterior[sid] = {
            "prob": {r: float(pp[RID[r]]) for r in REL},
            "route": route,
            "confidence": float(pp.max()),
            "margin": margin,
            "target_gt_for_check": canon_rel(target_y[i]),
        }

        rows.append(
            {
                "sid": sid,
                "target_gt_for_check": canon_rel(target_y[i]),
                "route": route,
                "confidence": float(pp.max()),
                "margin": margin,
                **{f"prob_{r}": float(pp[RID[r]]) for r in REL},
            }
        )

    # Optional sanity check against old saved predictions.
    pred_csv = selector_dir / "synthetic400_to_coco440_predictions.csv"
    sanity = {}
    if pred_csv.exists():
        old = pd.read_csv(pred_csv)
        if "sid" in old.columns:
            old["sid"] = pd.to_numeric(old["sid"], errors="raise").astype(int)

            method = (
                f"top{int(selector_k)}_"
                + ("weighted" if weighted else "equal")
                + ("_layerbalanced" if layer_balanced else "")
            )
            col = f"pred_{method}"

            if col in old.columns:
                old_map = dict(
                    zip(
                        old["sid"],
                        old[col].map(canon_rel),
                    )
                )
                comparable = [
                    sid for sid in posterior
                    if sid in old_map
                ]
                if comparable:
                    sanity["saved_prediction_column"] = col
                    sanity["N"] = len(comparable)
                    sanity["agreement"] = float(
                        np.mean(
                            [
                                posterior[sid]["route"] == old_map[sid]
                                for sid in comparable
                            ]
                        )
                    )

    return (
        posterior,
        pd.DataFrame(rows),
        selected_rows.copy(),
        target_path,
        codebook_path,
        reliability_path,
        sanity,
    )


# =============================================================================
# q_r and candidate-conditioned B_r
# =============================================================================

def strongest_other(scores: Dict[str, float], r: str) -> str:
    return max(
        (x for x in REL if x != r),
        key=lambda x: float(scores[x]),
    )


def q_for_entry(entry, grads, r):
    L = int(entry["update_layer"])
    p = int(entry["real_position"])
    g = grads[r].get(L, None)
    if g is None:
        return float("nan")
    if not (0 <= p < g.shape[1]):
        return float("nan")
    a = np.asarray(entry["_real_update"], np.float32)
    return float(
        np.dot(
            a,
            g[0, p].astype(np.float32),
        )
    )


def compute_update_policy_scores(
    *,
    entries,
    scores,
    grads,
    probs,
    route,
    gt,
    spec_meta,
):
    foils = {
        r: strongest_other(scores, r)
        for r in REL
    }

    out = []

    # Posterior order is sample-level.
    pvec = np.asarray([probs[r] for r in REL], dtype=np.float64)
    order = np.argsort(-pvec)
    top1 = REL[int(order[0])]
    top2 = REL[int(order[1])]

    for e in entries:
        q = {
            r: q_for_entry(e, grads, r)
            for r in REL
        }
        if not all(np.isfinite(q[r]) for r in REL):
            continue

        Br = {
            r: float(q[r] - q[foils[r]])
            for r in REL
        }

        hard = float(Br[route])

        soft = float(
            sum(
                (float(probs[r]) - 0.25) * float(q[r])
                for r in REL
            )
        )

        postB = float(
            sum(
                float(probs[r]) * float(Br[r])
                for r in REL
            )
        )

        vote = float(
            sum(
                float(probs[r]) * float(np.sign(Br[r]))
                for r in REL
            )
        )

        s1 = int(np.sign(Br[top1]))
        s2 = int(np.sign(Br[top2]))
        if s1 != 0 and s1 == s2:
            top2_score = float(
                s1 * min(abs(Br[top1]), abs(Br[top2]))
            )
        else:
            top2_score = 0.0

        oracle = float(Br[gt])

        sm = spec_meta.get(
            int(e["real_position"]),
            {},
        )

        ee = dict(e)
        ee["_score_hard"] = hard
        ee["_score_soft"] = soft
        ee["_score_postB"] = postB
        ee["_score_vote"] = vote
        ee["_score_top2"] = top2_score
        ee["_score_oracle"] = oracle

        ee["_q"] = q
        ee["_Br"] = Br
        ee["_foils"] = foils

        ee["canonical_slot"] = str(
            sm.get("canonical_slot", "")
        )
        ee["global_rank"] = int(
            sm.get("global_rank", sm.get("best_rank", -1))
        )

        out.append(ee)

    return out


SCORE_KEYS = {
    "hard": "_score_hard",
    "soft": "_score_soft",
    "postB": "_score_postB",
    "vote": "_score_vote",
    "top2": "_score_top2",
    "oracle": "_score_oracle",
}


# =============================================================================
# Policy patching
# =============================================================================

def sample_trigger(policy: Policy, *, route, baseline, conflict_margin):
    if policy.oracle:
        return True

    if policy.trigger_kind == "all":
        return True

    if policy.trigger_kind == "disagree":
        return (
            route in REL
            and baseline in REL
            and route != baseline
        )

    if policy.trigger_kind == "conflict":
        return (
            route in REL
            and baseline in REL
            and route != baseline
            and float(conflict_margin) >= float(policy.trigger_value)
        )

    raise ValueError(policy.trigger_kind)


def choose_policy_entries(scored, policy: Policy):
    key = SCORE_KEYS[policy.score_kind]

    rows = []
    for e in scored:
        ee = dict(e)
        ee["real_update_decision_score"] = float(e[key])
        rows.append(ee)

    # Top-M update selection by |policy score|, sample-wide.
    if int(policy.top_m) > 0 and len(rows) > int(policy.top_m):
        order = np.argsort(
            -np.asarray(
                [
                    abs(float(e["real_update_decision_score"]))
                    for e in rows
                ],
                dtype=float,
            )
        )
        keep = set(
            int(i) for i in order[: int(policy.top_m)]
        )
        rows = [
            e for i, e in enumerate(rows)
            if i in keep
        ]

    return rows


def patch_for_policy(scored, policy: Policy):
    rows = choose_policy_entries(scored, policy)

    if policy.action == "signed":
        condition = "real_signed"
    elif policy.action == "positive":
        condition = "real_positive"
    elif policy.action == "negative":
        condition = "real_negative_cancel"
    else:
        raise ValueError(policy.action)

    return gate.build_real_update_patch_map(
        rows,
        condition,
        float(policy.scale),
        float(policy.abs_threshold),
    )


# =============================================================================
# Summaries
# =============================================================================

def summarize_custom(gen_df):
    """
    Same style as gate.summarize_generation, but preserve arbitrary policy names.
    """
    if not len(gen_df):
        return pd.DataFrame()

    base = (
        gen_df[gen_df["condition"] == "baseline"]
        .set_index("sid")
    )

    rows = []

    for cond, g in gen_df[
        gen_df["condition"] != "baseline"
    ].groupby("condition"):
        g = g.copy()
        sids = sorted(
            set(g["sid"].astype(int))
            & set(base.index.astype(int))
        )
        if not sids:
            continue

        bb = base.loc[sids]
        gg = g.set_index("sid").loc[sids]

        bcor = bb["correct"].astype(bool).to_numpy()
        pcor = gg["correct"].astype(bool).to_numpy()
        changed = (
            bb["prediction"].astype(str).to_numpy()
            != gg["prediction"].astype(str).to_numpy()
        )

        w2c = int(np.sum((~bcor) & pcor))
        c2w = int(np.sum(bcor & (~pcor)))

        rows.append(
            {
                "condition": cond,
                "N": len(sids),
                "baseline_accuracy": float(np.mean(bcor)),
                "patched_accuracy": float(np.mean(pcor)),
                "gain": float(np.mean(pcor) - np.mean(bcor)),
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net": w2c - c2w,
                "changed": int(np.sum(changed)),
                "repair_rate_on_wrong": (
                    w2c / max(int(np.sum(~bcor)), 1)
                ),
                "preserve_rate_on_correct": (
                    1.0 - c2w / max(int(np.sum(bcor)), 1)
                ),
                "trigger_rate": float(
                    gg["triggered"].astype(bool).mean()
                ),
                "mean_patched_updates": float(
                    gg["n_patched_updates"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "patched_accuracy",
            "correct_to_wrong",
            "wrong_to_correct",
        ],
        ascending=[False, True, False],
    )


def summarize_relation(gen_df):
    rows = []

    for gt, x in gen_df.groupby("gt"):
        s = summarize_custom(x)
        if len(s):
            s.insert(1, "relation", DISPLAY.get(gt, gt))
            rows.append(s)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_seen_unseen(gen_df):
    if "template_split" not in gen_df.columns:
        return pd.DataFrame()

    rows = []
    for split, x in gen_df.groupby("template_split"):
        s = summarize_custom(x)
        if len(s):
            s.insert(1, "template_split", split)
            rows.append(s)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sign_summary(update_df):
    rows = []

    for method, col in [
        ("hard", "score_hard"),
        ("soft", "score_soft"),
        ("postB", "score_postB"),
        ("vote", "score_vote"),
        ("top2", "score_top2"),
    ]:
        for cohort_name, g in [
            ("all", update_df),
            (
                "baseline_wrong",
                update_df[~update_df["baseline_correct"]],
            ),
            (
                "baseline_correct",
                update_df[update_df["baseline_correct"]],
            ),
        ]:
            v = g[
                (g["oracle_sign"] != 0)
                & (np.sign(g[col]) != 0)
            ].copy()

            if len(v):
                pred = np.sign(v[col].to_numpy(float)).astype(int)
                oracle = v["oracle_sign"].to_numpy(int)
                correct = pred == oracle

                rows.append(
                    {
                        "method": method,
                        "cohort": cohort_name,
                        "N_updates": int(len(v)),
                        "N_samples": int(v["sid"].nunique()),
                        "sign_accuracy": float(np.mean(correct)),
                        "weighted_sign_accuracy": weighted_accuracy(
                            correct,
                            v["oracle_abs_B"].to_numpy(float),
                        ),
                        "coverage_of_nonzero_oracle_updates": (
                            len(v)
                            / max(
                                int(
                                    np.sum(
                                        g["oracle_sign"].to_numpy(int) != 0
                                    )
                                ),
                                1,
                            )
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "method": method,
                        "cohort": cohort_name,
                        "N_updates": 0,
                        "N_samples": 0,
                        "sign_accuracy": np.nan,
                        "weighted_sign_accuracy": np.nan,
                        "coverage_of_nonzero_oracle_updates": 0.0,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    update_layers = parse_ints(a.update_layers)
    alphas = parse_floats(a.alphas)
    conflict_thresholds = parse_floats(a.conflict_thresholds)
    update_thresholds = parse_floats(a.update_thresholds)
    top_ms = parse_ints(a.top_ms)

    if not update_layers:
        raise ValueError("No update layers")

    policies = build_policies(
        alphas=alphas,
        conflict_thresholds=conflict_thresholds,
        update_thresholds=update_thresholds,
        top_ms=top_ms,
        policy_set=a.policy_set,
    )

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    template_dir = Path(a.global_template_dir)
    prior_dir = Path(a.prior_real_update_dir)
    selector_dir = Path(a.direction_selector_dir)

    global_top, discovery_sids = load_global_template(
        template_dir
    )

    (
        posterior,
        posterior_df,
        selected_heads,
        target_cache_path,
        codebook_path,
        reliability_path,
        selector_sanity,
    ) = load_direction_posterior(
        selector_dir=selector_dir,
        target_direction_dir=Path(a.target_direction_dir),
        selector_k=int(a.selector_k),
        weighted=bool(a.selector_weighted),
        layer_balanced=bool(a.selector_layer_balanced),
        temperature=float(a.selector_temperature),
    )

    posterior_df.to_csv(
        outdir / "direction_selector_posterior.csv",
        index=False,
    )
    selected_heads.to_csv(
        outdir / "direction_selector_selected_heads.csv",
        index=False,
    )

    # Full prior cohort / baseline.
    prior_cohort, _, prior_metadata, _ = l26.load_prior_run(
        prior_dir
    )

    prior_cohort = prior_cohort[
        prior_cohort["sid"].isin(set(posterior))
    ].copy()

    if int(a.eval_max_samples) > 0:
        prior_cohort = l26.stratified_cap_df(
            prior_cohort,
            int(a.eval_max_samples),
            int(a.seed) + 411,
        )

    two, eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        prior_cohort,
    )

    eval_meta = [
        m for m in eval_meta
        if int(m["sid"]) in posterior
    ]

    texts = prior_metadata.get(
        "candidate_texts",
        {
            "left": "left",
            "right": "right",
            "above": "above",
            "below": "below",
        },
    )
    texts = {
        canon_rel(k): str(v)
        for k, v in texts.items()
    }
    for r in REL:
        if r not in texts:
            raise RuntimeError(
                f"candidate_texts missing {r}: {texts}"
            )

    reduction = str(
        prior_metadata.get(
            "sequence_score_reduction",
            "mean",
        )
    )
    if reduction not in ("mean", "sum"):
        reduction = "mean"

    model = processor = None
    generation_rows = []
    update_rows = []
    resolution_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = (
            l26.load_model(a, two)
        )
        device = torch.device(a.device)
        candidate_ids = gate.encode_candidate_ids(
            processor,
            texts,
        )

        print("=" * 210)
        print("NON-ORACLE HOW POLICY SWEEP")
        print("=" * 210)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N requested={len(eval_meta)}")
        print(f"fixed template K={len(global_top)} from {template_dir}")
        print(
            f"template discovery SIDs={len(discovery_sids)} "
            f"(used only for seen/unseen reporting)"
        )
        print(f"update_layers={update_layers}")
        print(
            f"Direction selector Top{a.selector_k} | "
            f"weighted={a.selector_weighted} | "
            f"layer_balanced={a.selector_layer_balanced}"
        )
        print(f"selector sanity={selector_sanity}")
        print(f"policy_set={a.policy_set} | N policies={len(policies)}")
        print()

        for p in policies:
            print(
                f"  {p.name:42s} "
                f"score={p.score_kind:6s} "
                f"trigger={p.trigger_kind:8s} "
                f"scale={p.scale:.3f} "
                f"abs={p.abs_threshold:.4f} "
                f"topM={p.top_m:2d} "
                f"action={p.action}"
            )
        print()

        for m in tqdm(
            eval_meta,
            desc="HOW sweep",
        ):
            sid = int(m["sid"])
            real = None

            split = (
                "template_seen"
                if sid in discovery_sids
                else "template_unseen"
            )

            pp = posterior[sid]
            probs = pp["prob"]
            route = canon_rel(pp["route"])
            baseline_pred = canon_rel(
                m["baseline_prediction"]
            )

            conflict_margin = (
                float(probs[route] - probs.get(baseline_pred, 0.0))
                if (
                    route in REL
                    and baseline_pred in REL
                    and route != baseline_pred
                )
                else 0.0
            )

            generation_rows.append(
                {
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "prediction": baseline_pred,
                    "correct": bool(m["baseline_correct"]),
                    "triggered": False,
                    "route": route,
                    "route_correct": route == m["gt"],
                    "baseline_prediction": baseline_pred,
                    "route_disagrees_baseline": route != baseline_pred,
                    "conflict_margin": conflict_margin,
                    "template_split": split,
                    "resolved_global_positions": 0,
                    "n_patched_updates": 0,
                    "n_positive_updates": 0,
                    "n_negative_updates": 0,
                    "n_neutral_updates": 0,
                    "text": "",
                }
            )

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = gfix.dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                mapping, canon_seq, sig = gfix.canonical_mapping(
                    model=model,
                    processor=processor,
                    batch=rb,
                    ids=ids,
                    subject=m["subject"],
                    reference=m["reference"],
                )

                specs, resolved = gfix.make_specs_from_global_slots(
                    global_top=global_top,
                    mapping=mapping,
                    ids=ids,
                    cats=cats,
                    toks=toks,
                    selector_layer=max(update_layers),
                )

                for rr in resolved:
                    resolution_rows.append(
                        {
                            "sid": sid,
                            "gt": m["gt"],
                            "template_split": split,
                            "signature_hash": sig,
                            **rr,
                        }
                    )

                if not specs:
                    raise RuntimeError(
                        "No global template positions resolved"
                    )

                spec_meta = {
                    int(s["real_position"]): s
                    for s in specs
                }

                capture_layers = sorted(
                    set(
                        update_layers
                        + [L - 1 for L in update_layers]
                    )
                )

                real_states = gate.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                entries = gate.build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=bool(m["baseline_correct"]),
                    specs=specs,
                    r2n={},
                    real_states=real_states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=False,
                )

                if not entries:
                    raise RuntimeError(
                        "No REAL updates on fixed global scaffold"
                    )

                grad_layers = sorted(
                    set(
                        int(e["update_layer"])
                        for e in entries
                    )
                )

                # ---------------------------------------------------------
                # Four candidate sequence scores + gradients ONCE.
                # ---------------------------------------------------------
                scores = {}
                grads = {}

                for r in REL:
                    score, grad = gate.sequence_score_and_grads(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        answer_ids=candidate_ids[r],
                        reduction=reduction,
                        grad_layers=grad_layers,
                    )
                    scores[r] = float(score)
                    grads[r] = grad

                scored = compute_update_policy_scores(
                    entries=entries,
                    scores=scores,
                    grads=grads,
                    probs=probs,
                    route=route,
                    gt=canon_rel(m["gt"]),
                    spec_meta=spec_meta,
                )

                if not scored:
                    raise RuntimeError(
                        "No update got all four q_r scores"
                    )

                # ---------------------------------------------------------
                # Export update-level diagnostics.
                # ---------------------------------------------------------
                for e in scored:
                    q = e["_q"]
                    Br = e["_Br"]
                    foils = e["_foils"]

                    row = {
                        "sid": sid,
                        "gt": canon_rel(m["gt"]),
                        "baseline_correct": bool(m["baseline_correct"]),
                        "baseline_prediction": baseline_pred,
                        "route": route,
                        "route_correct": route == canon_rel(m["gt"]),
                        "route_disagrees_baseline": route != baseline_pred,
                        "conflict_margin": conflict_margin,
                        "template_split": split,
                        "canonical_slot": e["canonical_slot"],
                        "global_rank": e["global_rank"],
                        "update_layer": int(e["update_layer"]),
                        "position": int(e["real_position"]),
                        "token": str(e["token"]),
                        "category": str(e["category"]),
                        "real_update_norm": float(e["real_update_norm"]),
                        "score_hard": float(e["_score_hard"]),
                        "score_soft": float(e["_score_soft"]),
                        "score_postB": float(e["_score_postB"]),
                        "score_vote": float(e["_score_vote"]),
                        "score_top2": float(e["_score_top2"]),
                        "score_oracle": float(e["_score_oracle"]),
                        "oracle_sign": int(np.sign(e["_score_oracle"])),
                        "oracle_abs_B": abs(float(e["_score_oracle"])),
                        **{
                            f"prob_{r}": float(probs[r])
                            for r in REL
                        },
                        **{
                            f"sequence_score_{r}": float(scores[r])
                            for r in REL
                        },
                        **{
                            f"q_{r}": float(q[r])
                            for r in REL
                        },
                        **{
                            f"B_{r}": float(Br[r])
                            for r in REL
                        },
                        **{
                            f"foil_{r}": str(foils[r])
                            for r in REL
                        },
                    }
                    update_rows.append(row)

                # ---------------------------------------------------------
                # Actual generation for every policy.
                # ---------------------------------------------------------
                for policy in policies:
                    active = sample_trigger(
                        policy,
                        route=route,
                        baseline=baseline_pred,
                        conflict_margin=conflict_margin,
                    )

                    if active:
                        patch_map, counts = patch_for_policy(
                            scored,
                            policy,
                        )
                    else:
                        patch_map = {}
                        counts = {
                            "patched": 0,
                            "positive": 0,
                            "negative": 0,
                            "neutral": len(scored),
                        }

                    if not active or not patch_map:
                        pred = baseline_pred
                        text = ""
                    else:
                        pred, text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            max_new_tokens=a.max_new_tokens,
                        )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": m["gt"],
                            "condition": policy.name,
                            "prediction": pred,
                            "correct": pred == m["gt"],
                            "triggered": bool(active),
                            "route": route,
                            "route_correct": route == m["gt"],
                            "baseline_prediction": baseline_pred,
                            "route_disagrees_baseline": route != baseline_pred,
                            "conflict_margin": conflict_margin,
                            "template_split": split,
                            "resolved_global_positions": len(specs),
                            "n_patched_updates": int(
                                counts["patched"]
                            ),
                            "n_positive_updates": int(
                                counts["positive"]
                            ),
                            "n_negative_updates": int(
                                counts["negative"]
                            ),
                            "n_neutral_updates": int(
                                counts["neutral"]
                            ),
                            "text": text,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": (
                            traceback.format_exc().splitlines()[-80:]
                        ),
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =================================================================
        # Save raw outputs
        # =================================================================
        gen_df = pd.DataFrame(generation_rows)
        upd_df = pd.DataFrame(update_rows)
        res_df = pd.DataFrame(resolution_rows)

        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )
        upd_df.to_csv(
            outdir / "per_update_policy_scores.csv",
            index=False,
        )
        res_df.to_csv(
            outdir / "resolved_global_slots_all_eval_samples.csv",
            index=False,
        )

        # =================================================================
        # Summaries
        # =================================================================
        gen_summary = summarize_custom(gen_df)
        gen_relation = summarize_relation(gen_df)
        gen_split = summarize_seen_unseen(gen_df)
        sign_df = sign_summary(upd_df)

        gen_summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )
        gen_relation.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )
        gen_split.to_csv(
            outdir / "generation_template_seen_vs_unseen.csv",
            index=False,
        )
        sign_df.to_csv(
            outdir / "sign_summary.csv",
            index=False,
        )

        # Route summary.
        sample_routes = (
            gen_df[gen_df["condition"] == "baseline"][
                [
                    "sid",
                    "gt",
                    "correct",
                    "route",
                    "route_correct",
                    "baseline_prediction",
                    "route_disagrees_baseline",
                    "conflict_margin",
                    "template_split",
                ]
            ]
            .drop_duplicates("sid")
            .rename(columns={"correct": "baseline_correct"})
        )

        route_rows = []
        for cohort_name, g in [
            ("all", sample_routes),
            (
                "baseline_wrong",
                sample_routes[~sample_routes["baseline_correct"]],
            ),
            (
                "baseline_correct",
                sample_routes[sample_routes["baseline_correct"]],
            ),
            (
                "template_unseen",
                sample_routes[
                    sample_routes["template_split"]
                    == "template_unseen"
                ],
            ),
        ]:
            if not len(g):
                continue
            route_rows.append(
                {
                    "cohort": cohort_name,
                    "N": int(len(g)),
                    "direction_relation_accuracy": float(
                        g["route_correct"].mean()
                    ),
                    "disagreement_rate_with_baseline": float(
                        g["route_disagrees_baseline"].mean()
                    ),
                    "mean_conflict_margin": float(
                        g["conflict_margin"].mean()
                    ),
                }
            )

        route_summary = pd.DataFrame(route_rows)
        route_summary.to_csv(
            outdir / "route_summary.csv",
            index=False,
        )

        # Best policy tables.
        nonoracle = gen_summary[
            ~gen_summary["condition"].str.startswith("oracle_")
        ].copy()

        best = nonoracle.head(15).copy()
        best.to_csv(
            outdir / "best_nonoracle_policies.csv",
            index=False,
        )

        # Policy definition table.
        policy_df = pd.DataFrame(
            [
                {
                    "condition": p.name,
                    "score_kind": p.score_kind,
                    "trigger_kind": p.trigger_kind,
                    "trigger_value": p.trigger_value,
                    "action": p.action,
                    "scale": p.scale,
                    "abs_threshold": p.abs_threshold,
                    "top_m": p.top_m,
                    "oracle": p.oracle,
                }
                for p in policies
            ]
        )
        policy_df.to_csv(
            outdir / "policy_definitions.csv",
            index=False,
        )

        print("=" * 210)
        print("NON-ORACLE HOW SWEEP RESULTS")
        print("=" * 210)
        print(
            f"N successful={gen_df['sid'].nunique()} | "
            f"N update rows={len(upd_df)}"
        )
        print(f"selector sanity={selector_sanity}")

        print("\nROUTE SUMMARY")
        print("-" * 210)
        print(
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nSIGN PREDICTION")
        print("-" * 210)
        print(
            sign_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nTOP NON-ORACLE GENERATION POLICIES")
        print("-" * 210)
        print(
            best.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        if len(gen_split):
            print("\nTEMPLATE SEEN vs UNSEEN")
            print("-" * 210)
            top_names = set(best["condition"].head(8))
            view = gen_split[
                gen_split["condition"].isin(top_names)
            ]
            print(
                view.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

        report = [
            "=" * 210,
            "NON-ORACLE HOW POLICY SWEEP",
            "=" * 210,
            f"model={a.model} repo={spec.repo_id}",
            f"N successful={gen_df['sid'].nunique()}",
            f"fixed template K={len(global_top)}",
            f"template discovery N={len(discovery_sids)}",
            f"update layers={update_layers}",
            f"selector TopK={a.selector_k}",
            f"selector sanity={selector_sanity}",
            f"N policies={len(policies)}",
            "",
            "ROUTE SUMMARY",
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "SIGN PREDICTION",
            sign_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "TOP NON-ORACLE GENERATION POLICIES",
            best.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "Interpretation reminders:",
            "  hard     = current top1 Direction route -> candidate-answer gradient.",
            "  soft     = sum_r (P(r)-1/4) q_r.",
            "  postB    = sum_r P(r) B_r.",
            "  vote     = posterior-weighted vote over sign(B_r).",
            "  top2     = intervene only when the two most likely relations agree",
            "             on update sign.",
            "  conflict = trigger only when Direction belief disagrees with baseline",
            "             AND exceeds baseline probability by the requested margin.",
            "  abs_*    = update-level |B_hard| abstention.",
            "  topM_*   = only strongest M update decisions in that sample.",
            "",
            "For a 440 run, inspect generation_template_seen_vs_unseen.csv.",
            "If unseen samples retain the gain, the fixed WHERE scaffold generalizes",
            "beyond the samples that originally created the template.",
        ]

        report_text = "\n".join(report) + "\n"
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        metadata = {
            "script": "eval_nonoracle_how_policy_sweep_v1_1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "global_template_dir": str(template_dir),
            "prior_real_update_dir": str(prior_dir),
            "direction_selector_dir": str(selector_dir),
            "target_direction_cache": str(target_cache_path),
            "synthetic_codebook": str(codebook_path),
            "source_reliability": str(reliability_path),
            "selector_sanity": selector_sanity,
            "selector_k": int(a.selector_k),
            "selector_weighted": bool(a.selector_weighted),
            "selector_layer_balanced": bool(a.selector_layer_balanced),
            "selector_temperature": float(a.selector_temperature),
            "update_layers": update_layers,
            "candidate_texts": texts,
            "sequence_score_reduction": reduction,
            "alphas": alphas,
            "conflict_thresholds": conflict_thresholds,
            "update_thresholds": update_thresholds,
            "top_ms": top_ms,
            "policy_set": a.policy_set,
            "N_policies": len(policies),
            "N_template_discovery_sids": len(discovery_sids),
            "nonoracle_definition": (
                "All non-oracle policies use fixed global WHERE + frozen "
                "Synthetic Direction-Head posterior + candidate-answer gradients. "
                "GT is used only for evaluation."
            ),
            "oracle_reference": (
                "oracle_signed_reference_a0p5 uses B_GT only as a diagnostic ceiling."
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
