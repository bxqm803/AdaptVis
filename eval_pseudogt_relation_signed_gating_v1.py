#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_pseudogt_relation_signed_gating_v1.py

Goal
====
Take samples whose TRUE GT is e.g. LEFT, keep the original image/question and
the original clean decoder trajectory unchanged, but COUNTERFACTUALLY treat
another spatial relation (e.g. RIGHT) as the desired pseudo-GT.

Crucially, the primary polarity definition DOES NOT compare RIGHT against LEFT.

For every realized clean block update

    a_{i,L,p} = h_out - h_in

and every candidate spatial relation r, compute

    q_r(i,L,p) = a_{i,L,p}^T grad_h S_r

where S_r is the teacher-forced sequence score of candidate r.

Primary pseudo-GT polarity
==========================
If pseudo-GT is RIGHT:

    B_RIGHT(i,L,p) := q_RIGHT(i,L,p)

Then:

    B_RIGHT > 0  -> this realized update locally INCREASES RIGHT score
    B_RIGHT < 0  -> this realized update locally DECREASES RIGHT score

No LEFT vector / LEFT score is used in this primary sign definition.

The intervention reuses the same signed-gating logic that was effective in the
previous oracle repair experiment:

    B_target > 0 : add +alpha * a
    B_target < 0 : add -alpha * a

So with alpha=0.5:

    positive clean update: effective a -> 1.5 a
    negative clean update: effective a -> 0.5 a

We do NOT reverse the whole update vector.

Why this experiment matters
===========================
If the SAME clean updates can be re-labeled by different pseudo spatial goals,
and pseudo-target-specific signed gating selectively drives generation toward
the chosen target, that supports the interpretation that "positive/negative"
is relation-conditioned utilization polarity rather than an intrinsic
good/bad property of an update.

This script also computes, for every update:

    q_left, q_right, q_above, q_below

so we can directly test whether LEFT and RIGHT effects are actually opposite.
They are NOT assumed to be.

Optional margin control
=======================
For comparison only, the script also computes

    B_margin(r) = q_r - q_best_other

where best_other is the strongest clean candidate other than r.

Use:
    --polarity-mode target_score   (DEFAULT; exactly the user's proposal)
or
    --polarity-mode target_margin  (control)

Recommended first run: LEFT samples, pseudo-GT RIGHT
====================================================
CUDA_VISIBLE_DEVICES=0 python -u eval_pseudogt_relation_signed_gating_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --source-rel left \
  --pseudo-targets right \
  --update-layers 20-26 \
  --scales 0.5 \
  --max-samples 40 \
  --polarity-mode target_score \
  --output-dir output/qwen3b_left_pseudogt_right_signed_n40_v1 \
  --overwrite

Stronger controllability test: same LEFT cohort, all four pseudo targets
=======================================================================
CUDA_VISIBLE_DEVICES=0 python -u eval_pseudogt_relation_signed_gating_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --source-rel left \
  --pseudo-targets all \
  --update-layers 20-26 \
  --scales 0.5 \
  --max-samples 40 \
  --polarity-mode target_score \
  --output-dir output/qwen3b_left_pseudogt_all4_signed_n40_v1 \
  --overwrite

Outputs
=======
per_update_relation_effects.csv
relation_effect_pair_summary.csv
generation_per_sample.csv
generation_summary.csv
controllability_matrix.csv
single_layer_summary.csv
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
from typing import Dict, Iterable, List

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
        help="'all' or comma-separated relations, e.g. right or left,right,above,below",
    )

    p.add_argument("--update-layers", default="20-26")
    p.add_argument(
        "--exclude-target-layer",
        action="store_true",
        help="Use only L < latest selected causal target layer for each token.",
    )

    p.add_argument(
        "--polarity-mode",
        default="target_score",
        choices=["target_score", "target_margin"],
        help=(
            "target_score: B_r = a dot grad S_r (does not use source relation). "
            "target_margin: B_r = a dot grad(S_r-S_best_other), control only."
        ),
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
        "--single-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run pseudo-target signed gating one layer at a time.",
    )
    p.add_argument(
        "--all-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run signed gating across all requested available layers.",
    )
    p.add_argument(
        "--shuffle-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Randomly permute +/-/0 signs WITHIN each layer while preserving the "
            "number of positive/negative/neutral updates."
        ),
    )
    p.add_argument(
        "--shuffle-repeats",
        type=int,
        default=1,
    )

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
            "Default cohort requires GT=source-rel AND baseline generation=source-rel. "
            "Set this to include all GT=source-rel samples."
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
            "Also teacher-force pseudo-target minus best-other score after each "
            "intervention. Adds substantial forward cost."
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
    vals = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError("--scales is empty")
    return vals


def parse_targets(text: str) -> List[str]:
    s = str(text).strip().lower()
    if s == "all":
        return list(REL)

    out = []
    for x in s.split(","):
        r = src.normalize_rel(x.strip())
        if not r:
            continue
        if r not in REL:
            raise ValueError(f"Unknown pseudo target {x!r}")
        if r not in out:
            out.append(r)

    if not out:
        raise ValueError("--pseudo-targets is empty")
    return out


def safe_mean(x):
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


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

    if a.max_samples > 0 and len(rows) > a.max_samples:
        rng = np.random.default_rng(a.seed)
        keep_idx = sorted(
            rng.choice(
                np.arange(len(rows)),
                size=int(a.max_samples),
                replace=False,
            ).tolist()
        )
        rows = [rows[i] for i in keep_idx]

    rows = sorted(rows, key=lambda z: int(z["sid"]))
    return rows


# =============================================================================
# Clean actual updates
# =============================================================================

def build_clean_updates(
    *,
    specs,
    real_states,
    update_layers,
    exclude_target_layer,
):
    rows = []

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

            rows.append(
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

    return rows


# =============================================================================
# Candidate score gradients and per-relation effects
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
        s, g = upd.sequence_score_and_grads(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
            grad_layers=grad_layers,
        )
        scores[r] = float(s)
        grads[r] = g

    return scores, grads


def attach_relation_effects(
    entries,
    *,
    scores,
    grads,
    threshold,
):
    """
    For every actual clean update a, compute:

        q_r = a dot grad S_r             for all r

    Also compute the optional margin-control quantity:

        m_r = q_r - q_best_other(r)

    No source/GT relation is used in q_r.
    """
    out = []

    competitors = {
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

            v = gr[0, p].astype(np.float32)
            q[r] = float(np.dot(a_real, v))

        if not valid:
            continue

        z = dict(e)

        for r in REL:
            comp = competitors[r]
            margin_b = float(q[r] - q[comp])

            z[f"q_{r}"] = float(q[r])
            z[f"sign_{r}"] = int(
                1 if q[r] > threshold
                else -1 if q[r] < -threshold
                else 0
            )
            z[f"best_other_{r}"] = comp
            z[f"margin_B_{r}"] = margin_b
            z[f"margin_sign_{r}"] = int(
                1 if margin_b > threshold
                else -1 if margin_b < -threshold
                else 0
            )

        z["real_update_norm"] = float(np.linalg.norm(a_real))
        out.append(z)

    return out


def polarity_value(e, target: str, mode: str) -> float:
    if mode == "target_score":
        return float(e[f"q_{target}"])
    if mode == "target_margin":
        return float(e[f"margin_B_{target}"])
    raise ValueError(mode)


# =============================================================================
# Patch construction
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
    pmap = {}
    allowed = set(map(int, layers))

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

        b = polarity_value(e, target, mode)
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


def build_shuffled_sign_patch(
    entries,
    *,
    target,
    mode,
    layers,
    scale,
    threshold,
    rng,
):
    """
    Preserve the exact number of + / - / 0 signs in each layer, but randomly
    assign those signs to update vectors in that layer.
    """
    pmap = {}
    allowed = set(map(int, layers))

    total_pos = total_neg = total_neu = total_patched = 0

    for L in sorted(allowed):
        rows = [
            e for e in entries
            if int(e["update_layer"]) == L
        ]
        if not rows:
            continue

        signs = []
        for e in rows:
            b = polarity_value(e, target, mode)
            if b > threshold:
                signs.append(1)
            elif b < -threshold:
                signs.append(-1)
            else:
                signs.append(0)

        signs = np.asarray(signs, dtype=np.int8)
        rng.shuffle(signs)

        for e, sgn in zip(rows, signs.tolist()):
            if sgn > 0:
                total_pos += 1
                vec = float(scale) * np.asarray(
                    e["_real_update"], np.float32
                )
            elif sgn < 0:
                total_neg += 1
                vec = -float(scale) * np.asarray(
                    e["_real_update"], np.float32
                )
            else:
                total_neu += 1
                vec = None

            if vec is not None:
                upd.add_patch(
                    pmap,
                    L,
                    int(e["real_position"]),
                    vec,
                )
                total_patched += 1

    return pmap, {
        "n_positive": total_pos,
        "n_negative": total_neg,
        "n_neutral": total_neu,
        "n_patched": total_patched,
    }


# =============================================================================
# Generation / score evaluation
# =============================================================================

def relation_margin_with_patch(
    *,
    model,
    decoder_layers,
    batch,
    candidate_ids,
    reduction,
    target,
    competitor,
    patch_map,
):
    return upd.sequence_margin_with_patch(
        model=model,
        decoder_layers=decoder_layers,
        batch=batch,
        candidate_ids=candidate_ids,
        reduction=reduction,
        gt=target,
        competitor=competitor,
        patch_map=patch_map,
    )


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
        margin = relation_margin_with_patch(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            candidate_ids=candidate_ids,
            reduction=reduction,
            target=target,
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
# Diagnostics: are relation effects actually opposites?
# =============================================================================

def relation_pair_summary(update_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i, r1 in enumerate(REL):
        for r2 in REL[i + 1:]:
            x = pd.to_numeric(
                update_df[f"q_{r1}"], errors="coerce"
            ).to_numpy(float)
            y = pd.to_numeric(
                update_df[f"q_{r2}"], errors="coerce"
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
                    "relation_1": r1,
                    "relation_2": r2,
                    "N_updates": int(len(x)),
                    "pearson_q": pearson,
                    "same_sign_fraction": float(np.mean(sx == sy)),
                    "opposite_sign_fraction": float(np.mean(sx == -sy)),
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


# =============================================================================
# Summaries
# =============================================================================

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
                (g["prediction"] != g["baseline_prediction"]).mean()
            ),
            "mean_n_positive": safe_mean(g["n_positive"]),
            "mean_n_negative": safe_mean(g["n_negative"]),
            "mean_n_patched": safe_mean(g["n_patched"]),
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


def controllability_matrix(gsum: pd.DataFrame) -> pd.DataFrame:
    """
    Wide relation-distribution matrix for all-layer signed gating.
    """
    x = gsum[
        gsum["condition"] == "pseudo_signed_all"
    ].copy()

    keep = [
        "pseudo_target",
        "scale",
        "N",
        "pseudo_target_hit_rate",
        *[f"pred_{r}_fraction" for r in REL],
    ]
    return x[keep].sort_values(
        ["scale", "pseudo_target"]
    ).reset_index(drop=True)


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

    model = processor = None
    generation_rows = []
    update_rows = []

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

        print("=" * 190)
        print("PSEUDO-GT RELATION-CONDITIONED SIGNED GATING")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source cohort GT={source}")
        print(f"pseudo targets={targets}")
        print(f"N samples={len(meta)}")
        print(f"update_layers={update_layers}")
        print(f"polarity_mode={a.polarity_mode}")
        print(f"scales={scales}")
        print(
            "cohort filter="
            + (
                "GT=source AND baseline_generation=source"
                if not a.include_nonsource_baseline
                else "GT=source only"
            )
        )
        print()
        print(
            "PRIMARY target_score definition: "
            "B_r = a_real dot grad(S_r); no source/LEFT score enters the sign."
        )
        print()

        device = torch.device(a.device)

        for m in tqdm(
            meta,
            desc=f"GT={source} PSEUDO={','.join(targets)}",
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

                # Self-contained baseline generation.
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

                # Clean realized block updates.
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

                # Four candidate gradients once. This lets us ask whether
                # LEFT/RIGHT are actually opposite without assuming it.
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
                        "No update received relation effects."
                    )

                # Save one WIDE row per update with q for all four relations.
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
                    }

                    for r in REL:
                        row[f"score_{r}"] = float(
                            scores[r]
                        )
                        row[f"q_{r}"] = float(
                            e[f"q_{r}"]
                        )
                        row[f"sign_{r}"] = int(
                            e[f"sign_{r}"]
                        )
                        row[f"best_other_{r}"] = str(
                            e[f"best_other_{r}"]
                        )
                        row[f"margin_B_{r}"] = float(
                            e[f"margin_B_{r}"]
                        )
                        row[f"margin_sign_{r}"] = int(
                            e[f"margin_sign_{r}"]
                        )

                    update_rows.append(row)

                available_layers = sorted(
                    set(
                        int(e["update_layer"])
                        for e in scored
                    )
                )

                # -------------------------------------------------------------
                # Baseline rows for each pseudo target.
                # -------------------------------------------------------------
                for target in targets:
                    comp = str(
                        scored[0][f"best_other_{target}"]
                    )
                    clean_margin = float(
                        scores[target] - scores[comp]
                    )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "true_gt": source,
                            "baseline_prediction": base_pred,
                            "condition": "baseline",
                            "pseudo_target": target,
                            "polarity_mode": a.polarity_mode,
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

                # -------------------------------------------------------------
                # Pseudo-target-conditioned interventions.
                # -------------------------------------------------------------
                for target in targets:
                    competitor = str(
                        scored[0][f"best_other_{target}"]
                    )

                    for scale in scales:
                        # All-layer signed gating.
                        if a.all_layers:
                            pmap, counts = build_signed_patch(
                                scored,
                                target=target,
                                mode=a.polarity_mode,
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
                                    "condition": "pseudo_signed_all",
                                    "pseudo_target": target,
                                    "polarity_mode": a.polarity_mode,
                                    "scale": float(scale),
                                    "intervention_layer": np.nan,
                                    "shuffle_repeat": np.nan,
                                    **res,
                                    **counts,
                                }
                            )

                            # Sign-shuffle control, preserving sign counts
                            # independently within each layer.
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
                                    rpmap, rcounts = (
                                        build_shuffled_sign_patch(
                                            scored,
                                            target=target,
                                            mode=a.polarity_mode,
                                            layers=available_layers,
                                            scale=float(scale),
                                            threshold=float(
                                                a.decision_threshold
                                            ),
                                            rng=rng,
                                        )
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
                                            "condition": "sign_shuffle_all",
                                            "pseudo_target": target,
                                            "polarity_mode": a.polarity_mode,
                                            "scale": float(scale),
                                            "intervention_layer": np.nan,
                                            "shuffle_repeat": int(rr),
                                            **rres,
                                            **rcounts,
                                        }
                                    )

                        # One layer at a time.
                        if a.single_layer:
                            for L in available_layers:
                                pmap, counts = build_signed_patch(
                                    scored,
                                    target=target,
                                    mode=a.polarity_mode,
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
                                        "condition": "pseudo_signed_single",
                                        "pseudo_target": target,
                                        "polarity_mode": a.polarity_mode,
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
            "No per-update relation effects were produced."
        )
    if not generation_rows:
        raise RuntimeError(
            "No generation results were produced."
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

    pair_df = relation_pair_summary(udf)
    pair_df.to_csv(
        outdir / "relation_effect_pair_summary.csv",
        index=False,
    )

    gsum = generation_summary(gdf)
    gsum.to_csv(
        outdir / "generation_summary.csv",
        index=False,
    )

    cm = controllability_matrix(gsum)
    cm.to_csv(
        outdir / "controllability_matrix.csv",
        index=False,
    )

    single = gsum[
        gsum["condition"] == "pseudo_signed_single"
    ].copy()
    single.to_csv(
        outdir / "single_layer_summary.csv",
        index=False,
    )

    # Per-layer polarity composition for every pseudo target.
    layer_rows = []
    for target in targets:
        col = (
            f"q_{target}"
            if a.polarity_mode == "target_score"
            else f"margin_B_{target}"
        )

        for L, g in udf.groupby(
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

            layer_rows.append(
                {
                    "pseudo_target": target,
                    "polarity_mode": a.polarity_mode,
                    "update_layer": int(L),
                    "N_updates": int(len(x)),
                    "positive_fraction": float(
                        np.mean(
                            x > a.decision_threshold
                        )
                    ),
                    "negative_fraction": float(
                        np.mean(
                            x < -a.decision_threshold
                        )
                    ),
                    "mean_effect": float(
                        np.mean(x)
                    ),
                    "mean_abs_effect": float(
                        np.mean(np.abs(x))
                    ),
                }
            )

    layer_polarity = pd.DataFrame(layer_rows)
    layer_polarity.to_csv(
        outdir / "pseudo_polarity_by_layer.csv",
        index=False,
    )

    # Report.
    report = [
        "=" * 190,
        "PSEUDO-GT RELATION-CONDITIONED SIGNED GATING",
        "=" * 190,
        f"true source cohort={source}",
        f"pseudo targets={targets}",
        f"N samples={gdf['sid'].nunique()}",
        f"update_layers={update_layers}",
        f"polarity_mode={a.polarity_mode}",
        f"scales={scales}",
        "",
        "A. RELATION EFFECT PAIRS ON THE SAME CLEAN UPDATES",
        "-" * 190,
        (
            pair_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(pair_df)
            else "EMPTY"
        ),
        "",
        "Interpretation of A:",
        "  q_r = a_real dot grad(S_r).",
        "  same_sign and both_positive can be nonzero: relation effects are NOT",
        "  assumed to be mutually opposite. LEFT/RIGHT opposition is an empirical",
        "  question here, not a construction.",
        "",
        "B. PSEUDO-TARGET POLARITY BY LAYER",
        "-" * 190,
        (
            layer_polarity.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(layer_polarity)
            else "EMPTY"
        ),
        "",
        "C. ALL-LAYER CONTROLLABILITY MATRIX",
        "-" * 190,
        (
            cm.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(cm)
            else "EMPTY"
        ),
        "",
        "D. SINGLE-LAYER PSEUDO-TARGET SIGNED GATING",
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
        "What would be strong evidence:",
        "  1) On the SAME original cohort and SAME clean updates, changing only the",
        "     pseudo target changes the +/- assignment substantially.",
        "  2) Signed gating using pseudo target RIGHT increases RIGHT generation;",
        "     pseudo target ABOVE increases ABOVE generation; etc.",
        "  3) Target-specific gating is much stronger than sign_shuffle_all, which",
        "     preserves how many + and - updates exist but destroys which update",
        "     receives which sign.",
        "  4) A layer-specific target effect localizes where relation-conditioned",
        "     utilization has causal leverage.",
        "",
        "Caveat:",
        "  grad(S_r) is still an oracle-chosen candidate score direction used for",
        "  mechanism diagnosis. This experiment tests whether update polarity is",
        "  relation-conditioned and causally steerable; it does not yet provide a",
        "  training-free non-oracle selector.",
    ]

    report_text = "\n".join(report) + "\n"
    print(report_text)
    (outdir / "analysis_summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script": "eval_pseudogt_relation_signed_gating_v1.py",
        "model": a.model,
        "real_update_dir": str(run_dir),
        "true_source_relation": source,
        "pseudo_targets": targets,
        "N_samples": int(gdf["sid"].nunique()),
        "update_layers": update_layers,
        "polarity_mode": a.polarity_mode,
        "scales": scales,
        "decision_threshold": float(
            a.decision_threshold
        ),
        "primary_definition": (
            "q_r = a_real dot grad(S_r); target_score mode uses sign(q_r) "
            "without subtracting source/GT relation score"
        ),
        "margin_control_definition": (
            "margin_B_r = q_r - q_best_other(r)"
        ),
        "signed_gating_definition": (
            "positive -> add +alpha*a_real; "
            "negative -> add -alpha*a_real"
        ),
        "require_baseline_source": (
            not bool(a.include_nonsource_baseline)
        ),
        "single_layer": bool(a.single_layer),
        "all_layers": bool(a.all_layers),
        "shuffle_control": bool(
            a.shuffle_control
        ),
        "uses_pseudo_target_candidate_identity": True,
        "uses_true_gt_in_polarity": False,
        "uses_true_gt_for_cohort_selection": True,
        "prior_causal_position_selection_is_oracle": True,
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
