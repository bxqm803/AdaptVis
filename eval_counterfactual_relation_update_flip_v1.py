#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_counterfactual_relation_update_flip_v1.py

Question
========
For a sample whose true relation is LEFT, suppose we counterfactually declare
the desired relation to be RIGHT.  Can we re-orient the REAL block updates,
layer by layer, so that generation flips toward RIGHT?

This is an oracle MECHANISM experiment, not a deployable method.

Key idea
========
For source relation s and counterfactual target t, define the target-vs-source
local decision axis at every causal-token update:

    g_{t<-s,L,p} = d(S_t - S_s) / d h_{L,p}

For the actual clean update

    a_{L,p} = h_{L,p}^{out} - h_{L,p}^{in}

define

    C_{L,p}^{t<-s} = a_{L,p}^T g_{t<-s,L,p}

Interpretation:
    C > 0 : this clean update locally favors the COUNTERFACTUAL target t
    C < 0 : this clean update locally favors the original source s

IMPORTANT:
The sign inversion itself is mathematically built into the definition:
    grad(S_t-S_s) = -grad(S_s-S_t).
So merely observing that LEFT-positive becomes RIGHT-negative proves nothing.

The meaningful test is CAUSAL GENERATION:

For every update with C < 0, reverse its realized vector at the block output:

    clean effective update:       +a
    intervention added:           -2a
    resulting effective update:   -a

Under the local linear model, its target-axis contribution changes from

    C < 0  --->  -C > 0.

Updates already favoring the target (C > 0) are left untouched.

We test:

  1) SINGLE-LAYER flip:
       reverse source-supporting updates only at one layer L.
       This is the cleanest causal localization.

  2) CUMULATIVE flip:
       reverse source-supporting updates from the first tested layer through L.
       This is an open-loop multi-layer intervention because later clean updates
       were measured on the unedited trajectory.

  3) ALL-LAYER flip:
       reverse every source-supporting update across all tested layers.

  4) MATCHED RANDOM control:
       at each layer, reverse the same NUMBER of updates, but choose positions
       randomly rather than by target-axis sign.

If target-directed flipping drives LEFT samples specifically toward RIGHT much
more than matched random flipping, this is evidence that the realized updates
participate in a relation-directed decision process.

Recommended first run
=====================
Use samples whose baseline generation is actually LEFT; otherwise "flip to
RIGHT" is not a clean transition.

CUDA_VISIBLE_DEVICES=0 python -u eval_counterfactual_relation_update_flip_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --source-rel left \
  --target-rel right \
  --update-layers 20-26 \
  --max-samples 40 \
  --flip-scale 1.0 \
  --output-dir output/qwen3b_left_to_right_update_flip_n40_v1 \
  --overwrite

Notes on flip-scale
===================
The patch added for a source-supporting clean update is

    -2 * flip_scale * a

Thus:
    flip_scale=0.5  => effective update approximately 0      (cancel)
    flip_scale=1.0  => effective update approximately -a     (full reversal)

Outputs
=======
per_update_counterfactual_axis.csv
generation_per_sample.csv
generation_summary.csv
generation_by_layer.csv
cumulative_by_layer.csv
axis_score_by_layer.csv
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

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import analyze_real_update_module_head_sources_v1 as src
    import eval_real_causal_token_update_gating_v1 as upd
except Exception as exc:
    raise SystemExit(
        "This script must run from the AdaptVis repository root beside:\n"
        "  analyze_real_update_module_head_sources_v1.py\n"
        "  eval_real_causal_token_update_gating_v1.py\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-12


# =============================================================================
# CLI
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
    p.add_argument("--target-rel", default="right", choices=REL)
    p.add_argument(
        "--allow-nonopposite",
        action="store_true",
        help="Permit target other than the geometrical opposite of source.",
    )

    p.add_argument("--update-layers", default="20-26")
    p.add_argument(
        "--exclude-target-layer",
        action="store_true",
        help="Use only L < latest selected causal target layer for a token.",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=1e-8,
    )
    p.add_argument(
        "--flip-scale",
        type=float,
        default=1.0,
        help=(
            "Patch source-supporting updates by -2*flip_scale*a. "
            "0.5 cancels; 1.0 fully reverses the clean update."
        ),
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
            "By default only samples whose baseline generation equals source-rel "
            "are included. Set this flag to include every GT=source-rel sample."
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
        "--single-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--cumulative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--matched-random",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--random-repeats",
        type=int,
        default=1,
        help="Matched-random generation repeats per single layer.",
    )

    p.add_argument(
        "--score-margins",
        action="store_true",
        help=(
            "Also teacher-force S_target-S_source under every intervention. "
            "This roughly triples forward cost; generation is always run."
        ),
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_rel(x):
    return src.normalize_rel(x)


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# Cohort
# =============================================================================

def choose_cohort(a, cohort, meta):
    mdf = pd.DataFrame(meta)
    if not len(mdf):
        raise RuntimeError("No metadata rows.")

    source = normalize_rel(a.source_rel)
    target = normalize_rel(a.target_rel)

    mdf["gt"] = mdf["gt"].map(normalize_rel)
    mdf["baseline_prediction"] = mdf["baseline_prediction"].map(normalize_rel)

    mdf = mdf[mdf["gt"] == source].copy()

    if not a.include_nonsource_baseline:
        mdf = mdf[mdf["baseline_prediction"] == source].copy()

    if not len(mdf):
        raise RuntimeError(
            f"No qualifying samples for source={source}; "
            f"include_nonsource_baseline={a.include_nonsource_baseline}"
        )

    if a.max_samples > 0 and len(mdf) > a.max_samples:
        rng = np.random.default_rng(a.seed)
        ids = mdf["sid"].to_numpy(int)
        keep = set(
            rng.choice(
                ids,
                size=int(a.max_samples),
                replace=False,
            ).tolist()
        )
        mdf = mdf[mdf["sid"].isin(keep)].copy()

    mdf = mdf.sort_values("sid").reset_index(drop=True)
    keep_sids = set(mdf["sid"].astype(int))

    return mdf, keep_sids


# =============================================================================
# Clean actual updates and target-vs-source axis
# =============================================================================

def valid_entries(
    *,
    specs,
    real_states,
    update_layers,
    exclude_target_layer,
):
    entries = []

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

            entries.append(
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

    return entries


def attach_axis_scores(
    entries,
    *,
    grad_target,
    grad_source,
    threshold,
):
    out = []

    for e in entries:
        L = int(e["update_layer"])
        p = int(e["real_position"])

        gtarg = grad_target.get(L, None)
        gsrc = grad_source.get(L, None)
        if gtarg is None or gsrc is None:
            continue
        if not (
            0 <= p < gtarg.shape[1]
            and 0 <= p < gsrc.shape[1]
        ):
            continue

        g_axis = (
            gtarg[0, p].astype(np.float32)
            - gsrc[0, p].astype(np.float32)
        )
        a_real = np.asarray(e["_real_update"], np.float32)

        score = float(np.dot(a_real, g_axis))
        if score > threshold:
            polarity = "toward_target"
            sign = 1
        elif score < -threshold:
            polarity = "toward_source"
            sign = -1
        else:
            polarity = "neutral"
            sign = 0

        z = dict(e)
        z.update(
            {
                "target_axis_score": score,
                "target_axis_sign": sign,
                "target_axis_polarity": polarity,
                "target_axis_grad_norm": float(np.linalg.norm(g_axis)),
                "real_update_norm": float(np.linalg.norm(a_real)),
                "target_axis_cosine": (
                    float(
                        np.dot(a_real, g_axis)
                        / max(
                            float(np.linalg.norm(a_real))
                            * float(np.linalg.norm(g_axis)),
                            EPS,
                        )
                    )
                ),
                "_target_axis_grad": g_axis,
            }
        )
        out.append(z)

    return out


# =============================================================================
# Patches
# =============================================================================

def target_flip_patch(
    entries,
    *,
    layers,
    flip_scale,
    threshold,
):
    """
    Reverse every CLEAN update that points against target axis.

        effective = a + (-2*scale*a)

    scale .5 -> 0
    scale 1  -> -a
    """
    selected_layers = set(map(int, layers))
    pmap = {}
    n_toward_target = 0
    n_toward_source = 0
    n_neutral = 0
    n_flipped = 0

    for e in entries:
        L = int(e["update_layer"])
        if L not in selected_layers:
            continue

        s = float(e["target_axis_score"])
        a_real = np.asarray(e["_real_update"], np.float32)

        if s > threshold:
            n_toward_target += 1
        elif s < -threshold:
            n_toward_source += 1
            upd.add_patch(
                pmap,
                L,
                int(e["real_position"]),
                -2.0 * float(flip_scale) * a_real,
            )
            n_flipped += 1
        else:
            n_neutral += 1

    counts = {
        "n_toward_target": n_toward_target,
        "n_toward_source": n_toward_source,
        "n_neutral": n_neutral,
        "n_flipped": n_flipped,
    }
    return pmap, counts


def matched_random_patch(
    entries,
    *,
    layer,
    flip_scale,
    threshold,
    rng,
):
    """
    At one layer, flip the same number of clean updates as target-directed flip,
    but choose rows uniformly without looking at axis sign.
    """
    L = int(layer)
    rows = [e for e in entries if int(e["update_layer"]) == L]
    n_needed = sum(
        float(e["target_axis_score"]) < -threshold
        for e in rows
    )

    if n_needed <= 0 or len(rows) == 0:
        return {}, {
            "n_random_flipped": 0,
            "n_target_selected_reference": int(n_needed),
        }

    n_needed = min(int(n_needed), len(rows))
    choice = rng.choice(
        np.arange(len(rows)),
        size=n_needed,
        replace=False,
    )

    pmap = {}
    for j in choice:
        e = rows[int(j)]
        upd.add_patch(
            pmap,
            L,
            int(e["real_position"]),
            -2.0 * float(flip_scale)
            * np.asarray(e["_real_update"], np.float32),
        )

    return pmap, {
        "n_random_flipped": int(n_needed),
        "n_target_selected_reference": int(n_needed),
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_patch(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    patch_map,
    candidate_ids,
    source,
    target,
    reduction,
    max_new_tokens,
    score_margins,
):
    pred, text = upd.generate_with_patch(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        batch=batch,
        patch_map=patch_map,
        max_new_tokens=max_new_tokens,
    )

    margin = float("nan")
    if score_margins:
        margin = upd.sequence_margin_with_patch(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            candidate_ids=candidate_ids,
            reduction=reduction,
            gt=target,
            competitor=source,
            patch_map=patch_map,
        )

    return {
        "prediction": normalize_rel(pred),
        "text": text,
        "target_hit": normalize_rel(pred) == target,
        "source_hit": normalize_rel(pred) == source,
        "target_minus_source_margin": margin,
    }


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(gdf):
    rows = []

    group_cols = ["condition", "intervention_layer", "cumulative_end_layer"]
    for keys, g in gdf.groupby(group_cols, dropna=False, sort=False):
        cond, layer, cum = keys
        rows.append(
            {
                "condition": cond,
                "intervention_layer": layer,
                "cumulative_end_layer": cum,
                "N": int(len(g)),
                "target_hit_rate": float(g["target_hit"].mean()),
                "source_hit_rate": float(g["source_hit"].mean()),
                "changed_from_baseline_rate": float(
                    (g["prediction"] != g["baseline_prediction"]).mean()
                ),
                "mean_n_flipped": float(
                    pd.to_numeric(g["n_flipped"], errors="coerce").mean()
                ),
                "mean_target_minus_source_margin": float(
                    pd.to_numeric(
                        g["target_minus_source_margin"],
                        errors="coerce",
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def axis_layer_summary(udf):
    rows = []
    for L, g in udf.groupby("update_layer", sort=True):
        s = g["target_axis_score"].to_numpy(float)
        abs_mass = np.abs(s).sum()
        toward_t = np.maximum(s, 0).sum()
        toward_s = np.maximum(-s, 0).sum()

        rows.append(
            {
                "update_layer": int(L),
                "N_updates": int(len(g)),
                "toward_target_fraction": float(np.mean(s > 0)),
                "toward_source_fraction": float(np.mean(s < 0)),
                "mean_target_axis_score": float(np.mean(s)),
                "target_mass_fraction": (
                    float(toward_t / abs_mass)
                    if abs_mass > EPS else np.nan
                ),
                "source_mass_fraction": (
                    float(toward_s / abs_mass)
                    if abs_mass > EPS else np.nan
                ),
                "mean_abs_target_axis_score": float(np.mean(np.abs(s))),
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

    source = normalize_rel(a.source_rel)
    target = normalize_rel(a.target_rel)

    if source == target:
        raise ValueError("source-rel and target-rel must differ.")

    if not a.allow_nonopposite and OPPOSITE[source] != target:
        raise ValueError(
            f"{source}->{target} is not an opposite-axis pair. "
            f"Expected target={OPPOSITE[source]!r}. "
            "Use --allow-nonopposite if intentional."
        )

    update_layers = src.parse_layers(a.update_layers)
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
        0,  # choose source-relation cohort ourselves below
        a.seed,
    )

    two, meta_all, rec_by_sid = src.load_dataset_for_sids(
        a,
        cohort,
    )

    cohort_df, keep_sids = choose_cohort(
        a,
        cohort,
        meta_all,
    )
    meta_by_sid = {
        int(m["sid"]): m
        for m in meta_all
        if int(m["sid"]) in keep_sids
    }
    meta = [meta_by_sid[int(s)] for s in cohort_df["sid"].astype(int)]

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
        bad = [L for L in capture_layers if not 0 <= L < n_layers]
        if bad:
            raise ValueError(
                f"capture layers outside 0..{n_layers-1}: {bad}"
            )

        candidate_texts = src.get_candidate_texts(prior_metadata)
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
        print("COUNTERFACTUAL RELATION UPDATE FLIP")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source={source} -> target={target}")
        print(f"N={len(meta)}")
        print(f"update_layers={update_layers}")
        print(f"flip_scale={a.flip_scale}")
        print(
            "baseline filter="
            + (
                "GT=source AND baseline=source"
                if not a.include_nonsource_baseline
                else "GT=source only"
            )
        )
        print(
            "NOTE: sign inversion under target-vs-source gradient is definitional; "
            "generation flip is the actual test."
        )
        print()

        device = torch.device(a.device)

        for m in tqdm(meta, desc=f"{source.upper()} -> {target.upper()}"):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            image = None
            try:
                image = src.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                rb = src.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )

                # Baseline generation is recorded again so this run is self-contained.
                base_text = src.base.generate_text(
                    model,
                    processor,
                    rb,
                    max_new_tokens=a.max_new_tokens,
                )
                base_pred = normalize_rel(
                    src.traj.normalize_relation(src.base, base_text)
                )

                # Clean prompt trajectory -> realized block updates.
                real_states = upd.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                specs = src.causal_position_specs(
                    selected_by_sid[sid]
                )
                entries = valid_entries(
                    specs=specs,
                    real_states=real_states,
                    update_layers=update_layers,
                    exclude_target_layer=bool(a.exclude_target_layer),
                )
                if not entries:
                    raise RuntimeError("No valid clean updates.")

                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                target_score, grad_target = upd.sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    answer_ids=candidate_ids[target],
                    reduction=reduction,
                    grad_layers=grad_layers,
                )
                source_score, grad_source = upd.sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    answer_ids=candidate_ids[source],
                    reduction=reduction,
                    grad_layers=grad_layers,
                )
                clean_axis_margin = float(target_score - source_score)

                scored = attach_axis_scores(
                    entries,
                    grad_target=grad_target,
                    grad_source=grad_source,
                    threshold=float(a.decision_threshold),
                )
                if not scored:
                    raise RuntimeError("No updates received target-axis score.")

                for e in scored:
                    update_rows.append(
                        {
                            "sid": sid,
                            "source_relation": source,
                            "target_relation": target,
                            "baseline_prediction": base_pred,
                            "baseline_target_hit": base_pred == target,
                            "clean_target_minus_source_margin": clean_axis_margin,
                            "update_layer": int(e["update_layer"]),
                            "real_position": int(e["real_position"]),
                            "token": str(e["token"]),
                            "category": str(e["category"]),
                            "broad_category": str(e["broad_category"]),
                            "max_target_layer": int(e["max_target_layer"]),
                            "target_axis_score": float(e["target_axis_score"]),
                            "target_axis_sign": int(e["target_axis_sign"]),
                            "target_axis_polarity": str(e["target_axis_polarity"]),
                            "target_axis_cosine": float(e["target_axis_cosine"]),
                            "real_update_norm": float(e["real_update_norm"]),
                            "target_axis_grad_norm": float(
                                e["target_axis_grad_norm"]
                            ),
                        }
                    )

                # Baseline row.
                baseline_margin = clean_axis_margin
                generation_rows.append(
                    {
                        "sid": sid,
                        "source_relation": source,
                        "target_relation": target,
                        "baseline_prediction": base_pred,
                        "condition": "baseline",
                        "intervention_layer": np.nan,
                        "cumulative_end_layer": np.nan,
                        "prediction": base_pred,
                        "target_hit": base_pred == target,
                        "source_hit": base_pred == source,
                        "target_minus_source_margin": baseline_margin,
                        "n_flipped": 0,
                        "n_toward_target": int(
                            sum(e["target_axis_score"] > a.decision_threshold for e in scored)
                        ),
                        "n_toward_source": int(
                            sum(e["target_axis_score"] < -a.decision_threshold for e in scored)
                        ),
                        "text": base_text,
                    }
                )

                available_layers = sorted(
                    set(int(e["update_layer"]) for e in scored)
                )

                # -------------------------------------------------------------
                # Single-layer target-directed reversal.
                # -------------------------------------------------------------
                if a.single_layer:
                    for L in available_layers:
                        pmap, counts = target_flip_patch(
                            scored,
                            layers=[L],
                            flip_scale=float(a.flip_scale),
                            threshold=float(a.decision_threshold),
                        )
                        res = evaluate_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=pmap,
                            candidate_ids=candidate_ids,
                            source=source,
                            target=target,
                            reduction=reduction,
                            max_new_tokens=int(a.max_new_tokens),
                            score_margins=bool(a.score_margins),
                        )
                        generation_rows.append(
                            {
                                "sid": sid,
                                "source_relation": source,
                                "target_relation": target,
                                "baseline_prediction": base_pred,
                                "condition": "target_flip_single",
                                "intervention_layer": int(L),
                                "cumulative_end_layer": np.nan,
                                **res,
                                **counts,
                            }
                        )

                        # Matched random control at the SAME layer and with the
                        # SAME number of reversed update positions.
                        if a.matched_random:
                            for rr in range(int(a.random_repeats)):
                                rng = np.random.default_rng(
                                    int(a.seed)
                                    + 1000003 * sid
                                    + 1009 * int(L)
                                    + 17 * rr
                                )
                                rpmap, rcounts = matched_random_patch(
                                    scored,
                                    layer=L,
                                    flip_scale=float(a.flip_scale),
                                    threshold=float(a.decision_threshold),
                                    rng=rng,
                                )
                                rres = evaluate_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=rb,
                                    patch_map=rpmap,
                                    candidate_ids=candidate_ids,
                                    source=source,
                                    target=target,
                                    reduction=reduction,
                                    max_new_tokens=int(a.max_new_tokens),
                                    score_margins=bool(a.score_margins),
                                )
                                generation_rows.append(
                                    {
                                        "sid": sid,
                                        "source_relation": source,
                                        "target_relation": target,
                                        "baseline_prediction": base_pred,
                                        "condition": "matched_random_single",
                                        "intervention_layer": int(L),
                                        "cumulative_end_layer": np.nan,
                                        "random_repeat": int(rr),
                                        **rres,
                                        "n_flipped": int(
                                            rcounts["n_random_flipped"]
                                        ),
                                        "n_toward_target": np.nan,
                                        "n_toward_source": int(
                                            rcounts[
                                                "n_target_selected_reference"
                                            ]
                                        ),
                                    }
                                )

                # -------------------------------------------------------------
                # Cumulative open-loop reversal.
                # -------------------------------------------------------------
                if a.cumulative and available_layers:
                    prefix = []
                    for L in available_layers:
                        prefix.append(int(L))
                        pmap, counts = target_flip_patch(
                            scored,
                            layers=prefix,
                            flip_scale=float(a.flip_scale),
                            threshold=float(a.decision_threshold),
                        )
                        res = evaluate_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=pmap,
                            candidate_ids=candidate_ids,
                            source=source,
                            target=target,
                            reduction=reduction,
                            max_new_tokens=int(a.max_new_tokens),
                            score_margins=bool(a.score_margins),
                        )
                        generation_rows.append(
                            {
                                "sid": sid,
                                "source_relation": source,
                                "target_relation": target,
                                "baseline_prediction": base_pred,
                                "condition": "target_flip_cumulative",
                                "intervention_layer": np.nan,
                                "cumulative_end_layer": int(L),
                                **res,
                                **counts,
                            }
                        )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-80:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
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

    if not generation_rows:
        raise RuntimeError("No generation results.")

    gdf = pd.DataFrame(generation_rows)
    udf = pd.DataFrame(update_rows)

    gdf.to_csv(
        outdir / "generation_per_sample.csv",
        index=False,
    )
    udf.to_csv(
        outdir / "per_update_counterfactual_axis.csv",
        index=False,
    )

    gsum = summarize_generation(gdf)
    gsum.to_csv(
        outdir / "generation_summary.csv",
        index=False,
    )

    layer_summary = gsum[
        gsum["condition"].isin(
            ["target_flip_single", "matched_random_single"]
        )
    ].copy()
    layer_summary.to_csv(
        outdir / "generation_by_layer.csv",
        index=False,
    )

    cumulative_summary = gsum[
        gsum["condition"] == "target_flip_cumulative"
    ].copy()
    cumulative_summary.to_csv(
        outdir / "cumulative_by_layer.csv",
        index=False,
    )

    axis_summary = axis_layer_summary(udf)
    axis_summary.to_csv(
        outdir / "axis_score_by_layer.csv",
        index=False,
    )

    # Compact report.
    baseline = gsum[gsum["condition"] == "baseline"]
    single = gsum[gsum["condition"] == "target_flip_single"]
    rand = gsum[gsum["condition"] == "matched_random_single"]
    cumul = gsum[gsum["condition"] == "target_flip_cumulative"]

    report = [
        "=" * 190,
        "COUNTERFACTUAL RELATION UPDATE FLIP",
        "=" * 190,
        f"source={source} -> target={target}",
        f"N samples={gdf['sid'].nunique()}",
        f"update_layers={update_layers}",
        f"flip_scale={a.flip_scale}",
        "",
        "A. CLEAN TARGET-vs-SOURCE AXIS BY LAYER",
        "-" * 190,
        axis_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "B. BASELINE",
        "-" * 190,
        (
            baseline.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(baseline) else "EMPTY"
        ),
        "",
        "C. SINGLE-LAYER TARGET-DIRECTED FLIP",
        "-" * 190,
        (
            single.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(single) else "EMPTY"
        ),
        "",
        "D. MATCHED-RANDOM SINGLE-LAYER CONTROL",
        "-" * 190,
        (
            rand.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(rand) else "DISABLED / EMPTY"
        ),
        "",
        "E. CUMULATIVE TARGET-DIRECTED FLIP",
        "-" * 190,
        (
            cumul.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(cumul) else "DISABLED / EMPTY"
        ),
        "",
        "Interpretation:",
        "  * Do NOT treat target-axis sign inversion as evidence; that follows from",
        "    changing S_source-S_target into S_target-S_source.",
        "  * The real evidence is whether target-directed reversal causes generation",
        "    to move specifically from source relation to target relation.",
        "  * A layer with high target_hit_rate under target_flip_single but low",
        "    target_hit_rate under matched_random_single is a candidate causal",
        "    relation-to-decision transformation layer.",
        "  * Cumulative intervention is open-loop: later patches use clean-trajectory",
        "    update vectors even though earlier layers have already been edited.",
    ]
    report_text = "\n".join(report) + "\n"
    print(report_text)
    (outdir / "analysis_summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script": "eval_counterfactual_relation_update_flip_v1.py",
        "model": a.model,
        "real_update_dir": str(run_dir),
        "source_relation": source,
        "target_relation": target,
        "opposite_pair": OPPOSITE[source] == target,
        "N_samples": int(gdf["sid"].nunique()),
        "update_layers": update_layers,
        "flip_scale": float(a.flip_scale),
        "decision_threshold": float(a.decision_threshold),
        "require_baseline_source": not bool(a.include_nonsource_baseline),
        "single_layer": bool(a.single_layer),
        "cumulative": bool(a.cumulative),
        "matched_random": bool(a.matched_random),
        "random_repeats": int(a.random_repeats),
        "score_margins": bool(a.score_margins),
        "axis_definition": (
            "C = a_real dot grad(S_target - S_source)"
        ),
        "flip_definition": (
            "if C<0 add -2*flip_scale*a_real at block output; "
            "flip_scale=1 makes clean effective +a become approximately -a"
        ),
        "causal_caveat": (
            "Oracle counterfactual target gradient and prior oracle causal-token "
            "selection are used. This tests relation-directed causal steerability, "
            "not a deployable non-oracle selector."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
