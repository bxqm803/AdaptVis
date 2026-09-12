#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_pseudogt_centered_relation_signed_gating_v1.py

Purpose
=======
Previous pseudo-GT experiment used

    q_r(i,L,p) = a_{i,L,p}^T grad_h S_r

and found that, on GT=LEFT / baseline=LEFT samples, RIGHT / ABOVE / BELOW
effects were strongly correlated.  That suggests q_r may contain a large
candidate-common decision component rather than a relation-specific component.

This script removes that common mode before assigning positive/negative signs.

For every realized CLEAN block update

    a_{i,L,p} = h_out - h_in

compute the four candidate effects

    q_left, q_right, q_above, q_below

with

    q_r = a^T grad S_r.

Then compute the per-update common mode

    q_common = (q_left + q_right + q_above + q_below) / 4

and the centered relation-specific effect

    q_centered_r = q_r - q_common.

Because dot products are linear, this is EXACTLY

    q_centered_r
      = a^T [ grad S_r - mean_k grad S_k ].

No LEFT-vs-RIGHT opposition is assumed.

For pseudo target RIGHT:

    B_RIGHT := q_centered_right.

Positive / negative signed gating is the same mild intervention used in the
previous oracle repair:

    B_target > 0 : add +alpha * a
    B_target < 0 : add -alpha * a

So alpha=0.5 gives approximately

    positive effective update:  a -> 1.5 a
    negative effective update:  a -> 0.5 a

and NEVER reverses the entire update.

Controls
========
1) RAW control:
       uses q_target without common-mode removal on the exact same cohort.

2) CENTERED sign-shuffle control:
       within each layer, preserve the number of + / - / 0 assignments but
       randomly permute which update gets which sign.

3) Single-layer CENTERED gating:
       localizes where centered relation-conditioned utilization has leverage.

Important caution
=================
Centering forces

    q_centered_left + q_centered_right
      + q_centered_above + q_centered_below = 0

for every update.  Therefore a reduction in pairwise same-sign rates after
centering is partly mathematical and is NOT evidence by itself.

The meaningful evidence is CAUSAL TARGET SPECIFICITY:
does centered pseudo-RIGHT gating selectively increase RIGHT generation more
than RAW gating and sign-shuffle, and do different pseudo targets selectively
drive different relations?

Recommended LEFT -> pseudo RIGHT run
====================================
CUDA_VISIBLE_DEVICES=0 python -u eval_pseudogt_centered_relation_signed_gating_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --source-rel left \
  --pseudo-targets right \
  --update-layers 20-26 \
  --scales 0.5 \
  --max-samples 40 \
  --output-dir output/qwen3b_left_pseudogt_right_centered_n40_v1 \
  --overwrite

Stronger all-four pseudo-target test
===================================
CUDA_VISIBLE_DEVICES=0 python -u eval_pseudogt_centered_relation_signed_gating_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --source-rel left \
  --pseudo-targets all \
  --update-layers 20-26 \
  --scales 0.5 \
  --max-samples 40 \
  --output-dir output/qwen3b_left_pseudogt_all4_centered_n40_v1 \
  --overwrite

Main outputs
============
per_update_relation_effects.csv
common_mode_by_layer.csv
relation_pair_summary_raw.csv
relation_pair_summary_centered.csv
pseudo_polarity_by_layer.csv
generation_per_sample.csv
generation_summary.csv
all_layer_comparison.csv
controllability_matrix_centered.csv
single_layer_centered_summary.csv
analysis_summary.txt
metadata.json
errors.jsonl
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
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import analyze_real_update_module_head_sources_v1 as src
    import eval_real_causal_token_update_gating_v1 as upd
except Exception as exc:
    raise SystemExit(
        "Run this script from the AdaptVis repository root beside:\n"
        "  analyze_real_update_module_head_sources_v1.py\n"
        "  eval_real_causal_token_update_gating_v1.py\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
EPS = 1e-12


# =============================================================================
# CLI / helpers
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
    p.add_argument("--real-update-dir", required=True)

    p.add_argument("--source-rel", default="left", choices=REL)
    p.add_argument(
        "--pseudo-targets",
        default="right",
        help="'all' or comma-separated relations.",
    )

    p.add_argument("--update-layers", default="20-26")
    p.add_argument(
        "--exclude-target-layer",
        action="store_true",
        help="Analyze only L < latest selected causal target layer for a token.",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=1e-8,
    )
    p.add_argument(
        "--scales",
        default="0.5",
        help="Comma-separated alpha values for signed gating.",
    )

    p.add_argument(
        "--run-raw-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run raw q_target signed gating on the exact same cohort.",
    )
    p.add_argument(
        "--single-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run centered signed gating one layer at a time.",
    )
    p.add_argument(
        "--all-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--shuffle-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Within each layer, preserve centered +/-/0 counts but randomly "
            "permute sign assignments across updates."
        ),
    )
    p.add_argument("--shuffle-repeats", type=int, default=1)

    p.add_argument(
        "--max-samples",
        type=int,
        default=40,
        help="0 = all qualifying samples.",
    )
    p.add_argument(
        "--include-nonsource-baseline",
        action="store_true",
        help=(
            "Default: GT=source-rel AND baseline generation=source-rel. "
            "Enable to include all GT=source-rel samples."
        ),
    )
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument(
        "--score-margins",
        action="store_true",
        help=(
            "Teacher-force pseudo-target minus strongest clean other candidate "
            "after each intervention. Adds forward cost."
        ),
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_scales(text: str) -> List[float]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        raise ValueError("--scales is empty")
    return out


def parse_targets(text: str) -> List[str]:
    s = str(text).strip().lower()
    if s == "all":
        return list(REL)

    out = []
    for x in s.split(","):
        r = src.normalize_rel(x.strip())
        if r not in REL:
            raise ValueError(f"Unknown pseudo target: {x!r}")
        if r not in out:
            out.append(r)

    if not out:
        raise ValueError("--pseudo-targets is empty")
    return out


def safe_mean(xs):
    x = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


# =============================================================================
# Cohort
# =============================================================================

def select_meta(a, meta_all):
    source = src.normalize_rel(a.source_rel)
    rows = []

    for m in meta_all:
        gt = src.normalize_rel(m["gt"])
        pred = src.normalize_rel(m["baseline_prediction"])

        if gt != source:
            continue
        if not a.include_nonsource_baseline and pred != source:
            continue

        z = dict(m)
        z["gt"] = gt
        z["baseline_prediction"] = pred
        rows.append(z)

    if not rows:
        raise RuntimeError(
            f"No qualifying samples for source={source}; "
            f"include_nonsource_baseline={a.include_nonsource_baseline}"
        )

    rows = sorted(rows, key=lambda z: int(z["sid"]))

    if a.max_samples > 0 and len(rows) > a.max_samples:
        rng = np.random.default_rng(a.seed)
        keep = sorted(
            rng.choice(
                np.arange(len(rows)),
                size=int(a.max_samples),
                replace=False,
            ).tolist()
        )
        rows = [rows[i] for i in keep]

    return rows


# =============================================================================
# Clean realized updates
# =============================================================================

def build_clean_updates(
    *,
    specs,
    real_states,
    update_layers,
    exclude_target_layer,
):
    out = []

    for s in specs:
        p = int(s["real_position"])
        cmax = int(s["max_target_layer"])

        for L in update_layers:
            L = int(L)
            if L < 1:
                continue

            if exclude_target_layer:
                if L >= cmax:
                    continue
            else:
                if L > cmax:
                    continue

            if L not in real_states or (L - 1) not in real_states:
                continue
            if not (
                0 <= p < real_states[L].shape[1]
                and 0 <= p < real_states[L - 1].shape[1]
            ):
                continue

            a_real = (
                real_states[L][0, p].astype(np.float32)
                - real_states[L - 1][0, p].astype(np.float32)
            )

            out.append(
                {
                    "update_layer": L,
                    "real_position": p,
                    "token": str(s["token"]),
                    "category": str(s["category"]),
                    "broad_category": str(s["broad_category"]),
                    "max_target_layer": cmax,
                    "_real_update": a_real,
                }
            )

    return out


# =============================================================================
# Four candidate gradients -> raw and centered relation effects
# =============================================================================

def compute_candidate_scores_and_grads(
    *,
    model,
    decoder_layers,
    batch,
    candidate_ids,
    reduction,
    grad_layers,
):
    scores = {}
    grads = {}

    for r in REL:
        score, grad = upd.sequence_score_and_grads(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
            grad_layers=grad_layers,
        )
        scores[r] = float(score)
        grads[r] = grad

    return scores, grads


def attach_relation_effects(
    entries,
    *,
    scores,
    grads,
    threshold,
):
    """
    For each clean update a:

        q_r = a . grad S_r

        q_common = mean_r q_r

        centered_q_r = q_r - q_common
                     = a . (grad S_r - mean_k grad S_k)

    The second equality is exact by linearity.
    """
    out = []

    best_other = {
        r: max(
            (rr for rr in REL if rr != r),
            key=lambda rr: scores[rr],
        )
        for r in REL
    }

    for e in entries:
        L = int(e["update_layer"])
        p = int(e["real_position"])
        a_real = np.asarray(e["_real_update"], np.float32)

        q = {}
        valid = True

        for r in REL:
            gr = grads[r].get(L, None)
            if gr is None or not (0 <= p < gr.shape[1]):
                valid = False
                break
            q[r] = float(
                np.dot(
                    a_real,
                    gr[0, p].astype(np.float32),
                )
            )

        if not valid:
            continue

        q_values = np.asarray([q[r] for r in REL], dtype=np.float64)
        q_common = float(q_values.mean())
        centered = {
            r: float(q[r] - q_common)
            for r in REL
        }

        raw_abs_mean = float(np.mean(np.abs(q_values)))
        centered_abs_mean = float(
            np.mean(np.abs([centered[r] for r in REL]))
        )

        z = dict(e)
        z["q_common"] = q_common
        z["raw_abs_mean_across_relations"] = raw_abs_mean
        z["centered_abs_mean_across_relations"] = centered_abs_mean
        z["common_abs_over_raw_abs_mean"] = (
            abs(q_common) / max(raw_abs_mean, EPS)
        )
        z["common_abs_over_centered_abs_mean"] = (
            abs(q_common) / max(centered_abs_mean, EPS)
        )
        z["raw_relation_std"] = float(np.std(q_values))
        z["raw_relation_range"] = float(np.max(q_values) - np.min(q_values))

        for r in REL:
            qr = float(q[r])
            cr = float(centered[r])
            comp = best_other[r]

            z[f"q_{r}"] = qr
            z[f"raw_sign_{r}"] = int(
                1 if qr > threshold
                else -1 if qr < -threshold
                else 0
            )

            z[f"centered_q_{r}"] = cr
            z[f"centered_sign_{r}"] = int(
                1 if cr > threshold
                else -1 if cr < -threshold
                else 0
            )

            z[f"best_other_{r}"] = comp
            z[f"margin_B_{r}"] = float(qr - q[comp])

        # Numerical check: centered effects must sum to ~0.
        z["centered_sum_check"] = float(
            sum(centered[r] for r in REL)
        )
        z["real_update_norm"] = float(np.linalg.norm(a_real))
        out.append(z)

    return out


def effect_value(e, target: str, mode: str) -> float:
    if mode == "centered":
        return float(e[f"centered_q_{target}"])
    if mode == "raw":
        return float(e[f"q_{target}"])
    raise ValueError(mode)


# =============================================================================
# Signed-gating patch maps
# =============================================================================

def build_signed_patch(
    entries,
    *,
    target,
    mode,
    layers,
    scale,
    threshold,
):
    allowed = set(map(int, layers))
    pmap = {}
    counts = {
        "n_positive": 0,
        "n_negative": 0,
        "n_neutral": 0,
        "n_patched": 0,
    }

    for e in entries:
        L = int(e["update_layer"])
        if L not in allowed:
            continue

        b = effect_value(e, target, mode)
        a_real = np.asarray(e["_real_update"], np.float32)

        vec = None
        if b > threshold:
            counts["n_positive"] += 1
            vec = float(scale) * a_real
        elif b < -threshold:
            counts["n_negative"] += 1
            vec = -float(scale) * a_real
        else:
            counts["n_neutral"] += 1

        if vec is not None:
            upd.add_patch(
                pmap,
                L,
                int(e["real_position"]),
                vec,
            )
            counts["n_patched"] += 1

    return pmap, counts


def build_shuffled_centered_patch(
    entries,
    *,
    target,
    layers,
    scale,
    threshold,
    rng,
):
    """
    Preserve + / - / 0 centered-sign counts independently in each layer,
    but randomly reassign the signs to updates in that layer.
    """
    allowed = set(map(int, layers))
    pmap = {}
    counts = {
        "n_positive": 0,
        "n_negative": 0,
        "n_neutral": 0,
        "n_patched": 0,
    }

    for L in sorted(allowed):
        rows = [
            e for e in entries
            if int(e["update_layer"]) == L
        ]
        if not rows:
            continue

        signs = []
        for e in rows:
            b = float(e[f"centered_q_{target}"])
            signs.append(
                1 if b > threshold
                else -1 if b < -threshold
                else 0
            )

        signs = np.asarray(signs, dtype=np.int8)
        rng.shuffle(signs)

        for e, sgn in zip(rows, signs.tolist()):
            if sgn > 0:
                counts["n_positive"] += 1
                vec = float(scale) * np.asarray(
                    e["_real_update"], np.float32
                )
            elif sgn < 0:
                counts["n_negative"] += 1
                vec = -float(scale) * np.asarray(
                    e["_real_update"], np.float32
                )
            else:
                counts["n_neutral"] += 1
                vec = None

            if vec is not None:
                upd.add_patch(
                    pmap,
                    L,
                    int(e["real_position"]),
                    vec,
                )
                counts["n_patched"] += 1

    return pmap, counts


# =============================================================================
# Evaluate an intervention
# =============================================================================

def evaluate_patch(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    patch_map,
    candidate_ids,
    reduction,
    target,
    competitor,
    max_new_tokens,
    score_margin,
):
    pred, text = upd.generate_with_patch(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        batch=batch,
        patch_map=patch_map,
        max_new_tokens=max_new_tokens,
    )
    pred = src.normalize_rel(pred)

    margin = float("nan")
    if score_margin:
        margin = upd.sequence_margin_with_patch(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            candidate_ids=candidate_ids,
            reduction=reduction,
            gt=target,
            competitor=competitor,
            patch_map=patch_map,
        )

    return {
        "prediction": pred,
        "text": text,
        "pseudo_target_hit": pred == target,
        "pseudo_target_margin": margin,
    }


# =============================================================================
# Diagnostics / summaries
# =============================================================================

def relation_pair_summary(
    update_df: pd.DataFrame,
    *,
    prefix: str,
    label: str,
) -> pd.DataFrame:
    rows = []

    for i, r1 in enumerate(REL):
        for r2 in REL[i + 1:]:
            x = pd.to_numeric(
                update_df[f"{prefix}{r1}"],
                errors="coerce",
            ).to_numpy(float)
            y = pd.to_numeric(
                update_df[f"{prefix}{r2}"],
                errors="coerce",
            ).to_numpy(float)

            good = np.isfinite(x) & np.isfinite(y)
            x = x[good]
            y = y[good]
            if len(x) == 0:
                continue

            sx = np.sign(x)
            sy = np.sign(y)

            pearson = (
                float(np.corrcoef(x, y)[0, 1])
                if len(x) >= 2
                and np.std(x) > EPS
                and np.std(y) > EPS
                else float("nan")
            )

            rows.append(
                {
                    "effect_space": label,
                    "relation_1": r1,
                    "relation_2": r2,
                    "N_updates": int(len(x)),
                    "pearson": pearson,
                    "same_sign_fraction": float(
                        np.mean(sx == sy)
                    ),
                    "opposite_sign_fraction": float(
                        np.mean(sx == -sy)
                    ),
                    "both_positive_fraction": float(
                        np.mean((sx > 0) & (sy > 0))
                    ),
                    "both_negative_fraction": float(
                        np.mean((sx < 0) & (sy < 0))
                    ),
                    "r1_pos_r2_neg_fraction": float(
                        np.mean((sx > 0) & (sy < 0))
                    ),
                    "r1_neg_r2_pos_fraction": float(
                        np.mean((sx < 0) & (sy > 0))
                    ),
                }
            )

    return pd.DataFrame(rows)


def common_mode_by_layer(update_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for L, g in update_df.groupby(
        "update_layer",
        sort=True,
    ):
        q = g[[f"q_{r}" for r in REL]].to_numpy(float)
        c = g[[f"centered_q_{r}" for r in REL]].to_numpy(float)

        raw_sign = np.sign(q)
        all_same_nonzero = (
            (np.all(raw_sign > 0, axis=1))
            | (np.all(raw_sign < 0, axis=1))
        )

        rows.append(
            {
                "update_layer": int(L),
                "N_updates": int(len(g)),
                "mean_abs_q_common": float(
                    np.mean(np.abs(g["q_common"].to_numpy(float)))
                ),
                "mean_raw_abs_across_relations": float(
                    np.mean(np.abs(q))
                ),
                "mean_centered_abs_across_relations": float(
                    np.mean(np.abs(c))
                ),
                "mean_common_abs_over_raw_abs_mean": safe_mean(
                    g["common_abs_over_raw_abs_mean"]
                ),
                "mean_common_abs_over_centered_abs_mean": safe_mean(
                    g["common_abs_over_centered_abs_mean"]
                ),
                "raw_all4_same_sign_fraction": float(
                    np.mean(all_same_nonzero)
                ),
                "mean_relation_std_raw": float(
                    np.mean(np.std(q, axis=1))
                ),
                "mean_relation_range_raw": float(
                    np.mean(np.max(q, axis=1) - np.min(q, axis=1))
                ),
                "max_abs_centered_sum_check": float(
                    np.max(
                        np.abs(
                            g["centered_sum_check"].to_numpy(float)
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def polarity_by_layer(
    update_df: pd.DataFrame,
    targets,
    threshold,
) -> pd.DataFrame:
    rows = []

    for target in targets:
        for mode, col in (
            ("raw", f"q_{target}"),
            ("centered", f"centered_q_{target}"),
        ):
            for L, g in update_df.groupby(
                "update_layer",
                sort=True,
            ):
                x = pd.to_numeric(
                    g[col],
                    errors="coerce",
                ).to_numpy(float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue

                rows.append(
                    {
                        "pseudo_target": target,
                        "mode": mode,
                        "update_layer": int(L),
                        "N_updates": int(len(x)),
                        "positive_fraction": float(
                            np.mean(x > threshold)
                        ),
                        "negative_fraction": float(
                            np.mean(x < -threshold)
                        ),
                        "mean_effect": float(np.mean(x)),
                        "mean_abs_effect": float(
                            np.mean(np.abs(x))
                        ),
                    }
                )

    return pd.DataFrame(rows)


def generation_summary(gdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "condition",
        "pseudo_target",
        "scale",
        "intervention_layer",
        "shuffle_repeat",
    ]

    for keys, g in gdf.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):
        cond, target, scale, layer, shuf = keys

        row = {
            "condition": cond,
            "pseudo_target": target,
            "scale": scale,
            "intervention_layer": layer,
            "shuffle_repeat": shuf,
            "N": int(len(g)),
            "pseudo_target_hit_rate": float(
                g["pseudo_target_hit"].mean()
            ),
            "changed_from_baseline_rate": float(
                (
                    g["prediction"]
                    != g["baseline_prediction"]
                ).mean()
            ),
            "mean_n_positive": safe_mean(
                g["n_positive"]
            ),
            "mean_n_negative": safe_mean(
                g["n_negative"]
            ),
            "mean_n_patched": safe_mean(
                g["n_patched"]
            ),
            "mean_pseudo_target_margin": safe_mean(
                g["pseudo_target_margin"]
            ),
        }

        for r in REL:
            row[f"pred_{r}_fraction"] = float(
                np.mean(g["prediction"] == r)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source = src.normalize_rel(a.source_rel)
    targets = parse_targets(a.pseudo_targets)
    update_layers = src.parse_layers(a.update_layers)
    scales = parse_scales(a.scales)

    if not update_layers:
        raise ValueError("No update layers.")

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    run_dir = Path(a.real_update_dir)

    (
        cohort,
        selected_by_sid,
        _prior_update,
        prior_metadata,
    ) = src.load_prior_run(
        run_dir,
        0,
        a.seed,
    )

    two, meta_all, rec_by_sid = src.load_dataset_for_sids(
        a,
        cohort,
    )
    meta = select_meta(a, meta_all)

    generation_rows = []
    update_rows = []

    model = processor = None

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        ) = src.load_model(a, two)

        n_layers = len(decoder_layers)
        capture_layers = sorted(
            set(update_layers + [L - 1 for L in update_layers])
        )
        bad = [
            L for L in capture_layers
            if not 0 <= L < n_layers
        ]
        if bad:
            raise ValueError(
                f"capture layers outside 0..{n_layers-1}: {bad}"
            )

        candidate_texts = src.get_candidate_texts(
            prior_metadata
        )
        candidate_ids = src.encode_candidate_ids(
            processor,
            candidate_texts,
        )

        reduction = str(
            prior_metadata.get(
                "sequence_score_reduction",
                "mean",
            )
        )
        if reduction not in ("mean", "sum"):
            reduction = "mean"

        device = torch.device(a.device)

        print("=" * 190)
        print("PSEUDO-GT CENTERED RELATION SIGNED GATING")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"true source cohort={source}")
        print(f"pseudo targets={targets}")
        print(f"N samples={len(meta)}")
        print(f"update_layers={update_layers}")
        print(f"scales={scales}")
        print(f"run_raw_control={a.run_raw_control}")
        print(
            "CENTERED definition: centered_q_r = q_r - mean(q_left,q_right,q_above,q_below)"
        )
        print(
            "                    = a dot [grad S_r - mean_k grad S_k]"
        )
        print()

        for m in tqdm(
            meta,
            desc=f"GT={source} CENTERED PSEUDO={','.join(targets)}",
        ):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            image = None

            try:
                image = src.base.record_image(
                    rec_by_sid[sid]
                )
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                rb = src.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )

                base_text = src.base.generate_text(
                    model,
                    processor,
                    rb,
                    max_new_tokens=a.max_new_tokens,
                )
                base_pred = src.normalize_rel(
                    src.traj.normalize_relation(
                        src.base,
                        base_text,
                    )
                )

                real_states = upd.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                specs = src.causal_position_specs(
                    selected_by_sid[sid]
                )
                entries = build_clean_updates(
                    specs=specs,
                    real_states=real_states,
                    update_layers=update_layers,
                    exclude_target_layer=bool(
                        a.exclude_target_layer
                    ),
                )
                if not entries:
                    raise RuntimeError(
                        "No valid clean updates."
                    )

                grad_layers = sorted(
                    set(
                        int(e["update_layer"])
                        for e in entries
                    )
                )

                scores, grads = compute_candidate_scores_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    candidate_ids=candidate_ids,
                    reduction=reduction,
                    grad_layers=grad_layers,
                )

                scored = attach_relation_effects(
                    entries,
                    scores=scores,
                    grads=grads,
                    threshold=float(
                        a.decision_threshold
                    ),
                )
                if not scored:
                    raise RuntimeError(
                        "No update received candidate gradients."
                    )

                for e in scored:
                    row = {
                        "sid": sid,
                        "true_gt": source,
                        "baseline_prediction": base_pred,
                        "update_layer": int(
                            e["update_layer"]
                        ),
                        "real_position": int(
                            e["real_position"]
                        ),
                        "token": str(e["token"]),
                        "category": str(e["category"]),
                        "broad_category": str(
                            e["broad_category"]
                        ),
                        "max_target_layer": int(
                            e["max_target_layer"]
                        ),
                        "real_update_norm": float(
                            e["real_update_norm"]
                        ),
                        "q_common": float(
                            e["q_common"]
                        ),
                        "raw_abs_mean_across_relations": float(
                            e["raw_abs_mean_across_relations"]
                        ),
                        "centered_abs_mean_across_relations": float(
                            e["centered_abs_mean_across_relations"]
                        ),
                        "common_abs_over_raw_abs_mean": float(
                            e["common_abs_over_raw_abs_mean"]
                        ),
                        "common_abs_over_centered_abs_mean": float(
                            e["common_abs_over_centered_abs_mean"]
                        ),
                        "raw_relation_std": float(
                            e["raw_relation_std"]
                        ),
                        "raw_relation_range": float(
                            e["raw_relation_range"]
                        ),
                        "centered_sum_check": float(
                            e["centered_sum_check"]
                        ),
                    }

                    for r in REL:
                        row[f"score_{r}"] = float(
                            scores[r]
                        )
                        row[f"q_{r}"] = float(
                            e[f"q_{r}"]
                        )
                        row[f"raw_sign_{r}"] = int(
                            e[f"raw_sign_{r}"]
                        )
                        row[f"centered_q_{r}"] = float(
                            e[f"centered_q_{r}"]
                        )
                        row[f"centered_sign_{r}"] = int(
                            e[f"centered_sign_{r}"]
                        )
                        row[f"best_other_{r}"] = str(
                            e[f"best_other_{r}"]
                        )
                        row[f"margin_B_{r}"] = float(
                            e[f"margin_B_{r}"]
                        )

                    update_rows.append(row)

                available_layers = sorted(
                    set(
                        int(e["update_layer"])
                        for e in scored
                    )
                )

                # Baseline rows.
                for target in targets:
                    competitor = str(
                        scored[0][f"best_other_{target}"]
                    )
                    clean_margin = float(
                        scores[target] - scores[competitor]
                    )
                    generation_rows.append(
                        {
                            "sid": sid,
                            "true_gt": source,
                            "baseline_prediction": base_pred,
                            "condition": "baseline",
                            "pseudo_target": target,
                            "scale": 0.0,
                            "intervention_layer": np.nan,
                            "shuffle_repeat": np.nan,
                            "prediction": base_pred,
                            "pseudo_target_hit": (
                                base_pred == target
                            ),
                            "pseudo_target_margin": clean_margin,
                            "n_positive": 0,
                            "n_negative": 0,
                            "n_neutral": 0,
                            "n_patched": 0,
                            "text": base_text,
                        }
                    )

                # Target-conditioned interventions.
                for target in targets:
                    competitor = str(
                        scored[0][f"best_other_{target}"]
                    )

                    for scale in scales:
                        if a.all_layers:
                            # CENTERED primary.
                            pmap, counts = build_signed_patch(
                                scored,
                                target=target,
                                mode="centered",
                                layers=available_layers,
                                scale=float(scale),
                                threshold=float(
                                    a.decision_threshold
                                ),
                            )
                            res = evaluate_patch(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                patch_map=pmap,
                                candidate_ids=candidate_ids,
                                reduction=reduction,
                                target=target,
                                competitor=competitor,
                                max_new_tokens=int(
                                    a.max_new_tokens
                                ),
                                score_margin=bool(
                                    a.score_margins
                                ),
                            )
                            generation_rows.append(
                                {
                                    "sid": sid,
                                    "true_gt": source,
                                    "baseline_prediction": base_pred,
                                    "condition": "centered_signed_all",
                                    "pseudo_target": target,
                                    "scale": float(scale),
                                    "intervention_layer": np.nan,
                                    "shuffle_repeat": np.nan,
                                    **res,
                                    **counts,
                                }
                            )

                            # RAW same-cohort control.
                            if a.run_raw_control:
                                rpmap, rcounts = build_signed_patch(
                                    scored,
                                    target=target,
                                    mode="raw",
                                    layers=available_layers,
                                    scale=float(scale),
                                    threshold=float(
                                        a.decision_threshold
                                    ),
                                )
                                rres = evaluate_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=rb,
                                    patch_map=rpmap,
                                    candidate_ids=candidate_ids,
                                    reduction=reduction,
                                    target=target,
                                    competitor=competitor,
                                    max_new_tokens=int(
                                        a.max_new_tokens
                                    ),
                                    score_margin=bool(
                                        a.score_margins
                                    ),
                                )
                                generation_rows.append(
                                    {
                                        "sid": sid,
                                        "true_gt": source,
                                        "baseline_prediction": base_pred,
                                        "condition": "raw_signed_all",
                                        "pseudo_target": target,
                                        "scale": float(scale),
                                        "intervention_layer": np.nan,
                                        "shuffle_repeat": np.nan,
                                        **rres,
                                        **rcounts,
                                    }
                                )

                            # CENTERED sign-shuffle control.
                            if a.shuffle_control:
                                for rr in range(
                                    int(a.shuffle_repeats)
                                ):
                                    rng = np.random.default_rng(
                                        int(a.seed)
                                        + 1000003 * sid
                                        + 101 * REL.index(target)
                                        + 17 * rr
                                    )
                                    spmap, scounts = (
                                        build_shuffled_centered_patch(
                                            scored,
                                            target=target,
                                            layers=available_layers,
                                            scale=float(scale),
                                            threshold=float(
                                                a.decision_threshold
                                            ),
                                            rng=rng,
                                        )
                                    )
                                    sres = evaluate_patch(
                                        model=model,
                                        processor=processor,
                                        decoder_layers=decoder_layers,
                                        batch=rb,
                                        patch_map=spmap,
                                        candidate_ids=candidate_ids,
                                        reduction=reduction,
                                        target=target,
                                        competitor=competitor,
                                        max_new_tokens=int(
                                            a.max_new_tokens
                                        ),
                                        score_margin=bool(
                                            a.score_margins
                                        ),
                                    )
                                    generation_rows.append(
                                        {
                                            "sid": sid,
                                            "true_gt": source,
                                            "baseline_prediction": base_pred,
                                            "condition": "centered_sign_shuffle_all",
                                            "pseudo_target": target,
                                            "scale": float(scale),
                                            "intervention_layer": np.nan,
                                            "shuffle_repeat": int(rr),
                                            **sres,
                                            **scounts,
                                        }
                                    )

                        # CENTERED one layer at a time.
                        if a.single_layer:
                            for L in available_layers:
                                pmap, counts = build_signed_patch(
                                    scored,
                                    target=target,
                                    mode="centered",
                                    layers=[L],
                                    scale=float(scale),
                                    threshold=float(
                                        a.decision_threshold
                                    ),
                                )
                                res = evaluate_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=rb,
                                    patch_map=pmap,
                                    candidate_ids=candidate_ids,
                                    reduction=reduction,
                                    target=target,
                                    competitor=competitor,
                                    max_new_tokens=int(
                                        a.max_new_tokens
                                    ),
                                    score_margin=bool(
                                        a.score_margins
                                    ),
                                )
                                generation_rows.append(
                                    {
                                        "sid": sid,
                                        "true_gt": source,
                                        "baseline_prediction": base_pred,
                                        "condition": "centered_signed_single",
                                        "pseudo_target": target,
                                        "scale": float(scale),
                                        "intervention_layer": int(L),
                                        "shuffle_repeat": np.nan,
                                        **res,
                                        **counts,
                                    }
                                )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "traceback_tail": (
                            traceback.format_exc()
                            .splitlines()[-80:]
                        ),
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                with contextlib.suppress(Exception):
                    if image is not None:
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

    if not update_rows:
        raise RuntimeError(
            "No per-update relation effects produced."
        )
    if not generation_rows:
        raise RuntimeError(
            "No generation results produced."
        )

    udf = pd.DataFrame(update_rows)
    gdf = pd.DataFrame(generation_rows)

    udf.to_csv(
        outdir / "per_update_relation_effects.csv",
        index=False,
    )
    gdf.to_csv(
        outdir / "generation_per_sample.csv",
        index=False,
    )

    common_df = common_mode_by_layer(udf)
    common_df.to_csv(
        outdir / "common_mode_by_layer.csv",
        index=False,
    )

    raw_pairs = relation_pair_summary(
        udf,
        prefix="q_",
        label="raw",
    )
    centered_pairs = relation_pair_summary(
        udf,
        prefix="centered_q_",
        label="centered",
    )
    raw_pairs.to_csv(
        outdir / "relation_pair_summary_raw.csv",
        index=False,
    )
    centered_pairs.to_csv(
        outdir / "relation_pair_summary_centered.csv",
        index=False,
    )

    pol_df = polarity_by_layer(
        udf,
        targets,
        float(a.decision_threshold),
    )
    pol_df.to_csv(
        outdir / "pseudo_polarity_by_layer.csv",
        index=False,
    )

    gsum = generation_summary(gdf)
    gsum.to_csv(
        outdir / "generation_summary.csv",
        index=False,
    )

    all_compare = gsum[
        gsum["condition"].isin(
            [
                "baseline",
                "raw_signed_all",
                "centered_signed_all",
                "centered_sign_shuffle_all",
            ]
        )
    ].copy()
    all_compare.to_csv(
        outdir / "all_layer_comparison.csv",
        index=False,
    )

    centered_matrix = gsum[
        gsum["condition"] == "centered_signed_all"
    ][
        [
            "pseudo_target",
            "scale",
            "N",
            "pseudo_target_hit_rate",
            "pred_left_fraction",
            "pred_right_fraction",
            "pred_above_fraction",
            "pred_below_fraction",
        ]
    ].copy()
    centered_matrix.to_csv(
        outdir / "controllability_matrix_centered.csv",
        index=False,
    )

    single = gsum[
        gsum["condition"] == "centered_signed_single"
    ].copy()
    single.to_csv(
        outdir / "single_layer_centered_summary.csv",
        index=False,
    )

    report = [
        "=" * 190,
        "PSEUDO-GT CENTERED RELATION SIGNED GATING",
        "=" * 190,
        f"true source cohort={source}",
        f"pseudo targets={targets}",
        f"N samples={gdf['sid'].nunique()}",
        f"update_layers={update_layers}",
        f"scales={scales}",
        "",
        "A. COMMON-MODE MAGNITUDE BY LAYER",
        "-" * 190,
        common_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "B. RAW RELATION EFFECT PAIRS",
        "-" * 190,
        raw_pairs.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "C. CENTERED RELATION EFFECT PAIRS",
        "-" * 190,
        centered_pairs.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "NOTE: centered effects sum to zero by construction; pairwise sign",
        "differences after centering are NOT causal evidence by themselves.",
        "",
        "D. RAW vs CENTERED vs SHUFFLE — ALL LAYERS",
        "-" * 190,
        all_compare.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "E. CENTERED CONTROLLABILITY MATRIX",
        "-" * 190,
        centered_matrix.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "F. CENTERED SINGLE-LAYER GATING",
        "-" * 190,
        (
            single.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(single)
            else "DISABLED / EMPTY"
        ),
        "",
        "What would support a relation-specific utilization interpretation:",
        "  1) centered_signed_all drives the chosen pseudo target substantially",
        "     more than raw_signed_all and centered_sign_shuffle_all;",
        "  2) with pseudo-targets=all, RIGHT gating preferentially raises RIGHT,",
        "     ABOVE gating preferentially raises ABOVE, etc.;",
        "  3) the effect localizes to specific layers under centered_signed_single.",
        "",
        "What would weaken the hypothesis:",
        "  centered gating mainly destroys the original answer, produces arbitrary",
        "  other relations, or is no better than sign shuffle.",
        "",
        "Caveat:",
        "  Candidate identity is still chosen externally and the prior causal-token",
        "  scaffold is oracle-derived. This is a mechanism test, not non-oracle repair.",
    ]

    report_text = "\n".join(report) + "\n"
    print(report_text)
    (outdir / "analysis_summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script": "eval_pseudogt_centered_relation_signed_gating_v1.py",
        "model": a.model,
        "real_update_dir": str(run_dir),
        "true_source_relation": source,
        "pseudo_targets": targets,
        "N_samples": int(gdf["sid"].nunique()),
        "update_layers": update_layers,
        "scales": scales,
        "decision_threshold": float(
            a.decision_threshold
        ),
        "raw_effect": "q_r = a_real dot grad(S_r)",
        "common_mode": "q_common = mean_r q_r",
        "centered_effect": (
            "centered_q_r = q_r - q_common = "
            "a_real dot (grad S_r - mean_k grad S_k)"
        ),
        "signed_gating": (
            "positive -> +alpha*a_real; negative -> -alpha*a_real"
        ),
        "run_raw_control": bool(a.run_raw_control),
        "shuffle_control": bool(a.shuffle_control),
        "single_layer": bool(a.single_layer),
        "uses_true_gt_in_centered_sign": False,
        "uses_true_gt_for_cohort_selection": True,
        "prior_causal_position_selection_is_oracle": True,
        "important_centering_caveat": (
            "sum_r centered_q_r = 0 by construction, so pairwise sign "
            "separation after centering is partly mathematical; causal "
            "generation target specificity is the primary evidence."
        ),
    }

    (outdir / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
