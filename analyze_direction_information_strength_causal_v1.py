#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direction Evidence Strength Diagnostic + Causal Validation
==========================================================

Goal
----
Test whether generation-wrong samples fail because their spatial information is
actually weaker, rather than merely because the model "has the information but
does not use it."

This script deliberately separates three things at the REAL computation points
used in analyze_prepost_attention_direction_causality_v2.py:

    pre_attn  : decoder block input, before Attention_l
    post_attn : residual stream after Attention_l and before MLP_l

For every selected layer/stage, define the image-grounded subject-reference
residual:

    r = (h_sub - h_ref)^img - (h_sub - h_ref)^noimg

and center it with TRAIN statistics:

    q = r - center

A TRAIN-derived 2-D spatial subspace is:

    S = span(mu_right - mu_left, mu_above - mu_below)

with orthonormal basis B.

We then decompose q into:

    q_spatial = B B^T q
    q_other   = q - q_spatial

and measure:

1) Spatial magnitude / "how much spatial signal"
       S_abs  = ||q_spatial||
       S_frac = ||q_spatial|| / ||q||

2) Spatial orientation / "where the spatial signal points"
       A_GT = cos(q_spatial, projected GT prototype)

3) Relation-coordinate evidence
       s_left, s_right, s_above, s_below
       s_GT
       s_finalWrong
       max_nonGT
       GT-minus-maxNonGT

This separates:

    weak magnitude:
        wrong has lower S_abs / S_frac

    wrong orientation:
        wrong has similar S_abs but lower A_GT

    weak GT component:
        wrong has lower s_GT

    excessive competitor:
        wrong has higher s_finalWrong / max_nonGT

Attention transition
--------------------
For each sample and layer we explicitly compare POST_ATTN - PRE_ATTN:

    Delta S_abs
    Delta A_GT
    Delta s_GT
    Delta max_nonGT

so we can ask whether Attention accumulates less spatial signal, rotates the
signal away from GT, or amplifies competitors.

Causal validation
-----------------
Correlation is not enough. For selected layers/stages we perform four targeted
pair-preserving interventions on the REAL hidden state:

A) magnitude_only
   Keep the sample's own spatial direction fixed, but raise only the magnitude
   of q_spatial to the median generation-correct control magnitude for the same
   GT relation.

   This does NOT inject the oracle GT direction.

   If this improves wrong samples whose Direction prediction is already correct,
   it supports "correct spatial evidence exists but is too weak."

B) gt_boost_hold_foil
   Raise the GT prototype coordinate toward the correct-control median while
   holding the foil coordinate fixed (minimum-L2 edit).

C) foil_suppress_hold_gt
   Reduce the foil coordinate toward the correct-control median while holding
   the GT coordinate fixed.

D) both
   Apply GT boost + foil suppression simultaneously.

Every targeted edit can be compared with a norm-matched random direction
orthogonal to the spatial subspace.

Evaluation
----------
1) All selected problem layers:
       first-step four-relation margin and restricted argmax.

2) Optional top layer/stage candidates:
       fresh full model.generate(), reporting W->C / C->W.

Important interpretation
------------------------
The strongest evidence for "spatial information itself is weak" is NOT merely
that R or s_GT is lower.

The stronger causal pattern is:

    - wrong sample has a Direction-correct spatial state;
    - S_abs is below correct-control level;
    - magnitude_only strengthens the sample's EXISTING spatial component;
    - final relation margin / generation improves;
    - norm-matched non-spatial random edit does not.

That directly tests spatial-evidence strength without supplying a new GT
direction.

Dependency
----------
Place this script in the same directory as:

    analyze_prepost_attention_direction_causality_v2.py
    extract_two_object_relation_states.py
    analyze_layerwise_direction_failure_scan_v1.py

Recommended full run after your existing v2 experiment
--------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python analyze_direction_information_strength_causal_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers auto \
  --eval-split test \
  --causal-layers all_selected \
  --causal-max-wrong 40 \
  --causal-max-correct 20 \
  --generation-top-k 2 \
  --generation-max-samples 30 \
  --output-dir output/qwen7b_direction_information_strength_causal_v1 \
  --overwrite

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python analyze_direction_information_strength_causal_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14,16,19 \
  --max-eval 20 \
  --causal-layers 14,16,19 \
  --causal-max-wrong 8 \
  --causal-max-correct 4 \
  --generation-top-k 1 \
  --generation-max-samples 8 \
  --output-dir output/qwen7b_direction_information_strength_causal_smoke \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import analyze_prepost_attention_direction_causality_v2 as core


RELATIONS = core.RELATIONS
REL2ID = core.REL2ID
STAGES = core.STAGES
EPS = 1e-10


# =============================================================================
# CLI / generic utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--failure-dir", default=None)
    p.add_argument(
        "--prepost-dir",
        default=None,
        help=(
            "Existing v2 output containing train_stage_vectors.npz. "
            "Strongly recommended to avoid refitting TRAIN stage codebooks."
        ),
    )
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument("--layers", default="auto")
    p.add_argument("--min-role-gap", type=float, default=0.5)
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=3000)

    p.add_argument(
        "--weak-quantile",
        type=float,
        default=0.25,
        help="Lower correct-control quantile used to call S / GT / alignment weak.",
    )
    p.add_argument(
        "--excess-quantile",
        type=float,
        default=0.75,
        help="Upper correct-control quantile used to call competitor evidence excessive.",
    )
    p.add_argument(
        "--target-stat",
        default="median",
        choices=["median", "mean"],
        help="Correct-control target for causal edits.",
    )

    p.add_argument(
        "--causal-layers",
        default="all_selected",
        help="none, all_selected, auto, or explicit layer list.",
    )
    p.add_argument(
        "--causal-stages",
        default="pre_attn,post_attn",
        help="Comma-separated subset of pre_attn,post_attn.",
    )
    p.add_argument("--causal-max-wrong", type=int, default=40)
    p.add_argument("--causal-max-correct", type=int, default=20)
    p.add_argument(
        "--causal-modes",
        default="magnitude_only,gt_boost_hold_foil,foil_suppress_hold_gt,both",
    )
    p.add_argument(
        "--random-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional edit norm cap; <=0 disables clipping.",
    )

    p.add_argument(
        "--generation-top-k",
        type=int,
        default=2,
        help="Top layer-stage pairs from first-step causal scan to validate with generate(). 0 disables.",
    )
    p.add_argument(
        "--generation-max-samples",
        type=int,
        default=30,
    )
    p.add_argument(
        "--generation-modes",
        default="magnitude_only,gt_boost_hold_foil,foil_suppress_hold_gt,both",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(vals.mean()) if len(vals) else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(np.median(vals)) if len(vals) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_csv_words(text: str) -> List[str]:
    return [
        x.strip()
        for x in str(text).split(",")
        if x.strip()
    ]


def parse_stages(text: str) -> List[str]:
    vals = parse_csv_words(text)
    bad = [x for x in vals if x not in STAGES]
    if bad:
        raise ValueError(f"Unknown stages: {bad}")
    if not vals:
        raise ValueError("No stages selected.")
    return vals


def target_stat(vals: Sequence[float], which: str) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    return float(np.median(x) if which == "median" else np.mean(x))


def bootstrap_gap(
    correct_vals: Sequence[float],
    wrong_vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(correct_vals, dtype=np.float64)
    b = np.asarray(wrong_vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")

    obs = float(a.mean() - b.mean())
    boots = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = aa.mean() - bb.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


# =============================================================================
# Stage codebooks: reuse v2 TRAIN activations if possible
# =============================================================================

def load_codebooks_from_prepost(
    prepost_dir: Path,
    selected_layers: Sequence[int],
):
    path = prepost_dir / "train_stage_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        files = set(z.files)
        y = np.asarray(z["relation"], dtype=object)

        cbs = {}
        diagnostics = []

        for li in selected_layers:
            for stage in STAGES:
                key = f"L{li}_{stage}"
                if key not in files:
                    raise KeyError(
                        f"{path} missing {key}; rerun v2 with this layer "
                        "or omit --prepost-dir."
                    )

                X = np.asarray(z[key], dtype=np.float32)
                cb = core.fit_cb(X, y)
                cbs[(li, stage)] = cb

                pred_idx = np.argmax(
                    (X - cb["center"]) @ cb["proto_arr"].T,
                    axis=1,
                )
                pred = np.asarray(
                    [RELATIONS[int(i)] for i in pred_idx],
                    dtype=object,
                )

                diagnostics.append({
                    "layer": li,
                    "stage": stage,
                    "n_train": len(X),
                    "train_direction_acc":
                        float(np.mean(pred == y)),
                    "mean_residual_norm":
                        float(np.linalg.norm(X, axis=1).mean()),
                    "source": str(path),
                })

    return cbs, diagnostics


# =============================================================================
# Direction evidence decomposition
# =============================================================================

def project_spatial(q: np.ndarray, B: np.ndarray) -> np.ndarray:
    B64 = np.asarray(B, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    return (B64 @ (B64.T @ q64)).astype(np.float32)


def projected_relation_direction(cb, rel: str) -> np.ndarray:
    p = np.asarray(cb["protos"][rel], dtype=np.float32)
    p_sp = project_spatial(p, cb["basis"])
    return core.unit(p_sp)


def decompose_vector(
    residual_vec: np.ndarray,
    cb,
    gt: str,
):
    q = (
        np.asarray(residual_vec, dtype=np.float32)
        - np.asarray(cb["center"], dtype=np.float32)
    )
    q_sp = project_spatial(q, cb["basis"])

    q_norm = float(np.linalg.norm(q))
    S_abs = float(np.linalg.norm(q_sp))
    S_frac = S_abs / q_norm if q_norm > EPS else float("nan")

    sp_dirs = {
        rel: projected_relation_direction(cb, rel)
        for rel in RELATIONS
    }

    if S_abs > EPS:
        q_sp_unit = q_sp / S_abs
        align = {
            rel: float(q_sp_unit @ sp_dirs[rel])
            for rel in RELATIONS
        }
    else:
        align = {rel: float("nan") for rel in RELATIONS}

    # Keep the original normalized prototype coordinates too.
    proto_scores = {
        rel: float(q @ cb["protos"][rel])
        for rel in RELATIONS
    }

    spatial_scores = {
        rel: float(q_sp @ sp_dirs[rel])
        for rel in RELATIONS
    }

    non_gt = [r for r in RELATIONS if r != gt]
    max_non_gt = max(
        non_gt,
        key=lambda r: proto_scores[r],
    )

    direction_pred = max(
        RELATIONS,
        key=lambda r: proto_scores[r],
    )

    return {
        "q": q,
        "q_sp": q_sp,
        "q_norm": q_norm,
        "S_abs": S_abs,
        "S_frac": S_frac,
        "A_GT": align[gt],
        "A_max_nonGT": max(
            align[r] for r in non_gt
            if math.isfinite(align[r])
        ) if any(math.isfinite(align[r]) for r in non_gt)
        else float("nan"),
        "s_GT": proto_scores[gt],
        "s_max_nonGT": proto_scores[max_non_gt],
        "max_nonGT_relation": max_non_gt,
        "GT_minus_maxNonGT":
            proto_scores[gt] - proto_scores[max_non_gt],
        "direction_pred": direction_pred,
        "direction_correct": int(direction_pred == gt),
        "proto_scores": proto_scores,
        "spatial_scores": spatial_scores,
        "alignments": align,
    }


def select_eval_sids(
    meta,
    records,
    split,
    max_eval,
    seed,
):
    sids = []

    for sid in meta["idx_by_sid"]:
        if split != "all" and meta["split"].get(sid, "") != split:
            continue
        if sid not in records:
            continue

        gt = meta["gt"].get(sid, "")
        gen = meta["generation"].get(sid, {})
        group = gen.get("generation_group", "")
        pred = gen.get("generation_pred", "")

        if gt not in REL2ID:
            continue
        if group not in ("correct", "wrong"):
            continue
        if group == "wrong" and (
            pred not in REL2ID or pred == gt
        ):
            continue

        sids.append(int(sid))

    if max_eval is None or len(sids) <= max_eval:
        return sorted(sids)

    rng = random.Random(seed)

    wrong = [
        sid for sid in sids
        if meta["generation"][sid]["generation_group"] == "wrong"
    ]
    correct = [
        sid for sid in sids
        if meta["generation"][sid]["generation_group"] == "correct"
    ]

    rng.shuffle(wrong)
    rng.shuffle(correct)

    nw = min(len(wrong), max(1, max_eval // 2))
    nc = min(len(correct), max_eval - nw)

    chosen = wrong[:nw] + correct[:nc]
    if len(chosen) < max_eval:
        left = [s for s in sids if s not in set(chosen)]
        rng.shuffle(left)
        chosen += left[: max_eval - len(chosen)]

    return sorted(set(chosen))


def collect_eval_decomposition(
    *,
    model,
    processor,
    decoder_layers,
    selected_layers,
    codebooks,
    records,
    meta,
    eval_sids,
    device,
    prompt_template,
    out_dir,
    save_every,
):
    rows = []
    errors = []
    vectors: Dict[int, Dict[Tuple[int, str], np.ndarray]] = {}

    for i, sid in enumerate(
        tqdm(eval_sids, desc="Direction strength decomposition"),
        1,
    ):
        rec = records[sid]
        image = None

        try:
            image = Image.open(rec.image_path).convert("RGB")

            real = core.capture_inference(
                model,
                processor,
                decoder_layers,
                selected_layers,
                rec,
                image,
                device,
                prompt_template,
            )
            noimg = core.capture_inference(
                model,
                processor,
                decoder_layers,
                selected_layers,
                rec,
                None,
                device,
                prompt_template,
            )

            gt = meta["gt"][sid]
            gen = meta["generation"][sid]
            group = gen["generation_group"]
            gen_pred = gen["generation_pred"]

            vectors[sid] = {}

            for li in selected_layers:
                for stage in STAGES:
                    residual = (
                        real[li][stage] - noimg[li][stage]
                    ).astype(np.float32)
                    vectors[sid][(li, stage)] = residual

                    d = decompose_vector(
                        residual,
                        codebooks[(li, stage)],
                        gt,
                    )

                    row = {
                        "sid": sid,
                        "layer": li,
                        "stage": stage,
                        "gt": gt,
                        "generation_group": group,
                        "generation_pred": gen_pred,

                        "S_abs": d["S_abs"],
                        "S_frac": d["S_frac"],
                        "q_norm": d["q_norm"],

                        "A_GT": d["A_GT"],
                        "A_max_nonGT": d["A_max_nonGT"],

                        "s_GT": d["s_GT"],
                        "s_max_nonGT": d["s_max_nonGT"],
                        "max_nonGT_relation":
                            d["max_nonGT_relation"],
                        "GT_minus_maxNonGT":
                            d["GT_minus_maxNonGT"],

                        "direction_pred":
                            d["direction_pred"],
                        "direction_correct":
                            d["direction_correct"],
                    }

                    for rel in RELATIONS:
                        row[f"s_{rel}"] = d["proto_scores"][rel]
                        row[f"spatial_s_{rel}"] = d["spatial_scores"][rel]
                        row[f"align_{rel}"] = d["alignments"][rel]

                    if (
                        group == "wrong"
                        and gen_pred in REL2ID
                        and gen_pred != gt
                    ):
                        row["s_finalWrong"] = d["proto_scores"][gen_pred]
                        row["A_finalWrong"] = d["alignments"][gen_pred]
                        row["GT_minus_finalWrong"] = (
                            d["proto_scores"][gt]
                            - d["proto_scores"][gen_pred]
                        )
                    else:
                        row["s_finalWrong"] = float("nan")
                        row["A_finalWrong"] = float("nan")
                        row["GT_minus_finalWrong"] = float("nan")

                    rows.append(row)

            if save_every > 0 and i % save_every == 0:
                write_csv(
                    out_dir / "per_sample_direction_strength.csv",
                    rows,
                )

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[ERROR sid={sid}] {type(e).__name__}: {e}"
            )

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(
        out_dir / "per_sample_direction_strength.csv",
        rows,
    )
    write_csv(
        out_dir / "decomposition_errors.csv",
        errors,
    )

    # Save vectors for reproducibility / downstream intervention analysis.
    good_sids = sorted(vectors)
    arrays = {
        "sid": np.asarray(good_sids, dtype=np.int64),
        "layers": np.asarray(selected_layers, dtype=np.int64),
        "stages": np.asarray(STAGES, dtype=object),
    }
    for li in selected_layers:
        for stage in STAGES:
            vals = [
                vectors[sid][(li, stage)]
                for sid in good_sids
                if (li, stage) in vectors[sid]
            ]
            if len(vals) == len(good_sids):
                arrays[f"L{li}_{stage}"] = np.stack(vals)

    np.savez_compressed(
        out_dir / "eval_stage_residual_vectors.npz",
        **arrays,
    )

    return rows, vectors, errors


# =============================================================================
# Descriptive summaries and matched correct controls
# =============================================================================

def summarize_correct_wrong(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)
    out = []

    metrics = (
        "S_abs",
        "S_frac",
        "A_GT",
        "s_GT",
        "s_max_nonGT",
        "GT_minus_maxNonGT",
    )

    for li in selected_layers:
        for stage in STAGES:
            rr = [
                r for r in rows
                if int(r["layer"]) == li
                and r["stage"] == stage
            ]
            cor = [
                r for r in rr
                if r["generation_group"] == "correct"
            ]
            wr = [
                r for r in rr
                if r["generation_group"] == "wrong"
            ]

            row = {
                "layer": li,
                "stage": stage,
                "n_correct": len(cor),
                "n_wrong": len(wr),
                "direction_acc_correct": safe_frac(
                    int(r["direction_correct"]) == 1
                    for r in cor
                ),
                "direction_acc_wrong": safe_frac(
                    int(r["direction_correct"]) == 1
                    for r in wr
                ),
            }

            for metric in metrics:
                cv = [float(r[metric]) for r in cor]
                wv = [float(r[metric]) for r in wr]
                gap, lo, hi = bootstrap_gap(
                    cv,
                    wv,
                    bootstrap,
                    rng,
                )

                row[f"{metric}_correct"] = safe_mean(cv)
                row[f"{metric}_wrong"] = safe_mean(wv)
                row[f"{metric}_gap_CminusW"] = gap
                row[f"{metric}_gap_ci95_lo"] = lo
                row[f"{metric}_gap_ci95_hi"] = hi

            out.append(row)

    return out


def build_control_targets(
    rows,
    weak_q,
    excess_q,
    which_stat,
):
    """
    Correct controls matched by layer/stage/GT.

    Foil-specific target is computed later from the stored per-relation columns,
    so the same correct control set can answer any wrong GT->foil pair.
    """
    buckets = defaultdict(list)

    for r in rows:
        if r["generation_group"] != "correct":
            continue
        key = (
            int(r["layer"]),
            str(r["stage"]),
            str(r["gt"]),
        )
        buckets[key].append(r)

    targets = {}
    output = []

    for key, rr in sorted(buckets.items()):
        li, stage, gt = key

        def vals(col):
            return np.asarray(
                [float(r[col]) for r in rr],
                dtype=np.float64,
            )

        S = vals("S_abs")
        Sf = vals("S_frac")
        A = vals("A_GT")
        G = vals("s_GT")

        t = {
            "n": len(rr),

            "S_target": target_stat(S, which_stat),
            "S_q_low": float(np.quantile(S, weak_q)),
            "Sfrac_target": target_stat(Sf, which_stat),
            "Sfrac_q_low": float(np.quantile(Sf, weak_q)),

            "A_target": target_stat(A, which_stat),
            "A_q_low": float(np.quantile(A, weak_q)),

            "GT_target": target_stat(G, which_stat),
            "GT_q_low": float(np.quantile(G, weak_q)),
        }

        for foil in RELATIONS:
            if foil == gt:
                continue
            F = vals(f"s_{foil}")
            P = G - F

            t[f"foil_{foil}_target"] = target_stat(F, which_stat)
            t[f"foil_{foil}_q_high"] = float(
                np.quantile(F, excess_q)
            )
            t[f"pair_{foil}_target"] = target_stat(P, which_stat)
            t[f"pair_{foil}_q_low"] = float(
                np.quantile(P, weak_q)
            )

        targets[key] = t

        row = {
            "layer": li,
            "stage": stage,
            "gt": gt,
            "weak_quantile": weak_q,
            "excess_quantile": excess_q,
            "target_stat": which_stat,
            **t,
        }
        output.append(row)

    return targets, output


def build_wrong_diagnosis(
    rows,
    targets,
):
    out = []

    for r in rows:
        if r["generation_group"] != "wrong":
            continue

        gt = str(r["gt"])
        foil = str(r["generation_pred"])

        if foil not in REL2ID or foil == gt:
            continue

        key = (
            int(r["layer"]),
            str(r["stage"]),
            gt,
        )
        if key not in targets:
            continue

        t = targets[key]

        S = float(r["S_abs"])
        A = float(r["A_GT"])
        sgt = float(r["s_GT"])
        sfoil = float(r[f"s_{foil}"])
        pair = sgt - sfoil

        magnitude_weak = S < float(t["S_q_low"])
        orientation_weak = A < float(t["A_q_low"])
        gt_weak = sgt < float(t["GT_q_low"])
        foil_excess = (
            sfoil > float(t[f"foil_{foil}_q_high"])
        )
        pair_weak = (
            pair < float(t[f"pair_{foil}_q_low"])
        )

        if magnitude_weak and orientation_weak:
            highlevel = "magnitude_and_orientation_weak"
        elif magnitude_weak:
            highlevel = "magnitude_weak_only"
        elif orientation_weak:
            highlevel = "orientation_weak_only"
        else:
            highlevel = "neither_magnitude_nor_orientation_weak"

        out.append({
            "sid": int(r["sid"]),
            "layer": int(r["layer"]),
            "stage": r["stage"],
            "gt": gt,
            "final_wrong": foil,

            "direction_correct":
                int(r["direction_correct"]),

            "S_abs": S,
            "S_target": t["S_target"],
            "S_deficit":
                float(t["S_target"]) - S,
            "magnitude_weak":
                int(magnitude_weak),

            "A_GT": A,
            "A_target": t["A_target"],
            "A_deficit":
                float(t["A_target"]) - A,
            "orientation_weak":
                int(orientation_weak),

            "s_GT": sgt,
            "GT_target": t["GT_target"],
            "GT_deficit":
                float(t["GT_target"]) - sgt,
            "GT_component_weak":
                int(gt_weak),

            "s_finalWrong": sfoil,
            "foil_target":
                t[f"foil_{foil}_target"],
            "foil_excess":
                sfoil - float(t[f"foil_{foil}_target"]),
            "foil_component_excess":
                int(foil_excess),

            "GT_minus_finalWrong": pair,
            "pair_target":
                t[f"pair_{foil}_target"],
            "pair_deficit":
                float(t[f"pair_{foil}_target"]) - pair,
            "pair_weak":
                int(pair_weak),

            "highlevel_strength_type":
                highlevel,
        })

    return out


def summarize_wrong_diagnosis(
    rows,
    selected_layers,
):
    out = []

    for li in selected_layers:
        for stage in STAGES:
            rr = [
                r for r in rows
                if int(r["layer"]) == li
                and r["stage"] == stage
            ]

            out.append({
                "layer": li,
                "stage": stage,
                "n_wrong": len(rr),

                "direction_correct_rate": safe_frac(
                    int(r["direction_correct"]) == 1
                    for r in rr
                ),

                "magnitude_weak_rate": safe_frac(
                    int(r["magnitude_weak"]) == 1
                    for r in rr
                ),
                "orientation_weak_rate": safe_frac(
                    int(r["orientation_weak"]) == 1
                    for r in rr
                ),
                "GT_component_weak_rate": safe_frac(
                    int(r["GT_component_weak"]) == 1
                    for r in rr
                ),
                "foil_component_excess_rate": safe_frac(
                    int(r["foil_component_excess"]) == 1
                    for r in rr
                ),
                "pair_weak_rate": safe_frac(
                    int(r["pair_weak"]) == 1
                    for r in rr
                ),

                "magnitude_and_orientation_weak_rate": safe_frac(
                    r["highlevel_strength_type"]
                    == "magnitude_and_orientation_weak"
                    for r in rr
                ),
                "magnitude_weak_only_rate": safe_frac(
                    r["highlevel_strength_type"]
                    == "magnitude_weak_only"
                    for r in rr
                ),
                "orientation_weak_only_rate": safe_frac(
                    r["highlevel_strength_type"]
                    == "orientation_weak_only"
                    for r in rr
                ),

                "mean_S_deficit": safe_mean(
                    r["S_deficit"] for r in rr
                ),
                "mean_A_deficit": safe_mean(
                    r["A_deficit"] for r in rr
                ),
                "mean_GT_deficit": safe_mean(
                    r["GT_deficit"] for r in rr
                ),
                "mean_foil_excess": safe_mean(
                    r["foil_excess"] for r in rr
                ),
                "mean_pair_deficit": safe_mean(
                    r["pair_deficit"] for r in rr
                ),
            })

    return out


def attention_transition_rows(rows):
    lookup = {
        (int(r["sid"]), int(r["layer"]), str(r["stage"])): r
        for r in rows
    }

    sids = sorted({int(r["sid"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})

    out = []

    for sid in sids:
        for li in layers:
            a = lookup.get((sid, li, "pre_attn"))
            b = lookup.get((sid, li, "post_attn"))
            if a is None or b is None:
                continue

            out.append({
                "sid": sid,
                "layer": li,
                "gt": a["gt"],
                "generation_group":
                    a["generation_group"],
                "generation_pred":
                    a["generation_pred"],
                "direction_correct_pre":
                    a["direction_correct"],
                "direction_correct_post":
                    b["direction_correct"],

                "delta_S_abs":
                    float(b["S_abs"]) - float(a["S_abs"]),
                "delta_S_frac":
                    float(b["S_frac"]) - float(a["S_frac"]),
                "delta_A_GT":
                    float(b["A_GT"]) - float(a["A_GT"]),
                "delta_s_GT":
                    float(b["s_GT"]) - float(a["s_GT"]),
                "delta_s_max_nonGT":
                    float(b["s_max_nonGT"])
                    - float(a["s_max_nonGT"]),
                "delta_GT_minus_maxNonGT":
                    float(b["GT_minus_maxNonGT"])
                    - float(a["GT_minus_maxNonGT"]),
            })

    return out


def summarize_attention_transition(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed + 500)
    out = []

    metrics = (
        "delta_S_abs",
        "delta_S_frac",
        "delta_A_GT",
        "delta_s_GT",
        "delta_s_max_nonGT",
        "delta_GT_minus_maxNonGT",
    )

    for li in selected_layers:
        rr = [r for r in rows if int(r["layer"]) == li]
        cor = [
            r for r in rr
            if r["generation_group"] == "correct"
        ]
        wr = [
            r for r in rr
            if r["generation_group"] == "wrong"
        ]

        row = {
            "layer": li,
            "n_correct": len(cor),
            "n_wrong": len(wr),
        }

        for metric in metrics:
            cv = [float(r[metric]) for r in cor]
            wv = [float(r[metric]) for r in wr]
            gap, lo, hi = bootstrap_gap(
                cv, wv, bootstrap, rng
            )

            row[f"{metric}_correct"] = safe_mean(cv)
            row[f"{metric}_wrong"] = safe_mean(wv)
            row[f"{metric}_gap_CminusW"] = gap
            row[f"{metric}_gap_ci95_lo"] = lo
            row[f"{metric}_gap_ci95_hi"] = hi

        out.append(row)

    return out


# =============================================================================
# Causal edit construction
# =============================================================================

def min_l2_two_constraint(
    v1: np.ndarray,
    v2: np.ndarray,
    c1: float,
    c2: float,
) -> np.ndarray:
    """
    Minimum L2 delta satisfying:
        delta dot v1 = c1
        delta dot v2 = c2
    using a stable pseudo-inverse in FP64.
    """
    A = np.stack(
        [
            np.asarray(v1, dtype=np.float64),
            np.asarray(v2, dtype=np.float64),
        ],
        axis=1,
    )  # [D,2]
    c = np.asarray([c1, c2], dtype=np.float64)
    gram = A.T @ A
    delta = A @ (np.linalg.pinv(gram, rcond=1e-10) @ c)
    return delta.astype(np.float32)


def clip_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return delta
    n = float(np.linalg.norm(delta))
    if n <= max_norm or n <= EPS:
        return delta
    return (delta * (max_norm / n)).astype(np.float32)


def random_orthogonal_delta(
    delta_norm: float,
    B: np.ndarray,
    dim: int,
    seed: int,
):
    if delta_norm <= EPS:
        return np.zeros(dim, dtype=np.float32)

    rng = np.random.default_rng(seed)
    B64 = np.asarray(B, dtype=np.float64)

    for _ in range(100):
        v = rng.standard_normal(dim)
        v = v - B64 @ (B64.T @ v)
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            return (
                v / n * delta_norm
            ).astype(np.float32)

    raise RuntimeError("Could not sample random orthogonal delta.")


def build_edit(
    *,
    mode: str,
    decomposition: Mapping[str, Any],
    cb,
    gt: str,
    foil: str,
    target: Mapping[str, Any],
    max_edit_norm: float,
):
    q = np.asarray(decomposition["q"], dtype=np.float32)
    q_sp = np.asarray(
        decomposition["q_sp"],
        dtype=np.float32,
    )

    p_gt = np.asarray(cb["protos"][gt], dtype=np.float32)
    p_foil = np.asarray(cb["protos"][foil], dtype=np.float32)

    s_gt = float(q @ p_gt)
    s_foil = float(q @ p_foil)

    if mode == "magnitude_only":
        S = float(np.linalg.norm(q_sp))
        S_target = float(target["S_target"])

        if S <= EPS or S >= S_target:
            delta = np.zeros_like(q_sp)
        else:
            delta = q_sp * (S_target / S - 1.0)

    elif mode == "gt_boost_hold_foil":
        gt_def = max(
            0.0,
            float(target["GT_target"]) - s_gt,
        )
        delta = min_l2_two_constraint(
            p_gt,
            p_foil,
            gt_def,
            0.0,
        )

    elif mode == "foil_suppress_hold_gt":
        foil_excess = max(
            0.0,
            s_foil - float(target[f"foil_{foil}_target"]),
        )
        delta = min_l2_two_constraint(
            p_gt,
            p_foil,
            0.0,
            -foil_excess,
        )

    elif mode == "both":
        gt_def = max(
            0.0,
            float(target["GT_target"]) - s_gt,
        )
        foil_excess = max(
            0.0,
            s_foil - float(target[f"foil_{foil}_target"]),
        )
        delta = min_l2_two_constraint(
            p_gt,
            p_foil,
            gt_def,
            -foil_excess,
        )

    else:
        raise ValueError(f"Unknown causal mode: {mode}")

    delta = clip_delta(
        np.asarray(delta, dtype=np.float32),
        max_edit_norm,
    )

    return delta


# =============================================================================
# First-step causal evaluation
# =============================================================================

def select_causal_layers(
    text: str,
    selected_layers: Sequence[int],
    summary_rows,
    n_layers: int,
):
    t = str(text).strip().lower()

    if t == "none":
        return []
    if t == "all_selected":
        return list(selected_layers)
    if t != "auto":
        return core.parse_layers(text, n_layers)

    # Rank layers by representation gap in the diagnostic itself.
    by_layer = defaultdict(list)
    for r in summary_rows:
        by_layer[int(r["layer"])].append(r)

    scored = []
    for li, rr in by_layer.items():
        score = max(
            max(
                0.0,
                float(r["S_abs_gap_CminusW"]),
            )
            + max(
                0.0,
                float(r["GT_minus_maxNonGT_gap_CminusW"]),
            )
            for r in rr
        )
        scored.append((score, li))

    scored.sort(reverse=True)
    return sorted(
        li for score, li in scored if score > 0
    )


def choose_causal_sids(
    meta,
    eval_sids,
    max_wrong,
    max_correct,
    seed,
):
    rng = random.Random(seed + 707)

    wrong = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "wrong"
    ]
    correct = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "correct"
    ]

    rng.shuffle(wrong)
    rng.shuffle(correct)

    if max_wrong is not None:
        wrong = wrong[:max_wrong]
    if max_correct is not None:
        correct = correct[:max_correct]

    return sorted(wrong + correct)


def baseline_foil_from_scores(scores, gt):
    return max(
        [r for r in RELATIONS if r != gt],
        key=lambda r: scores[r],
    )


def restricted_pred(scores):
    return max(RELATIONS, key=lambda r: scores[r])


def causal_firststep_scan(
    *,
    model,
    processor,
    token_map,
    decoder_layers,
    causal_layers,
    causal_stages,
    causal_modes,
    causal_sids,
    records,
    meta,
    vectors,
    codebooks,
    targets,
    device,
    prompt_template,
    random_controls,
    max_edit_norm,
    seed,
):
    rows = []
    errors = []

    for sid in tqdm(
        causal_sids,
        desc="causal strength validation (first-step)",
    ):
        rec = records[sid]
        image = None

        try:
            image = Image.open(rec.image_path).convert("RGB")
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            batch, sp, rp = core.build_batch(
                processor,
                rec,
                question,
                image,
                device,
            )

            base_scores = core.score(
                model,
                batch,
                token_map,
            )
            base_pred = restricted_pred(base_scores)

            gt = meta["gt"][sid]
            group = meta["generation"][sid]["generation_group"]
            gen_pred = meta["generation"][sid]["generation_pred"]

            if (
                group == "wrong"
                and gen_pred in REL2ID
                and gen_pred != gt
            ):
                foil = gen_pred
                foil_kind = "actual_generated_wrong"
            else:
                foil = baseline_foil_from_scores(
                    base_scores,
                    gt,
                )
                foil_kind = "baseline_firststep_best_nonGT"

            base_margin = (
                base_scores[gt] - base_scores[foil]
            )
            base_correct = int(base_pred == gt)

            for li in causal_layers:
                for stage in causal_stages:
                    residual = vectors[sid][(li, stage)]
                    cb = codebooks[(li, stage)]

                    decomp = decompose_vector(
                        residual,
                        cb,
                        gt,
                    )

                    key = (li, stage, gt)
                    if key not in targets:
                        continue
                    t = targets[key]

                    subgroup = (
                        "wrong_direction_correct"
                        if (
                            group == "wrong"
                            and decomp["direction_correct"] == 1
                        )
                        else
                        "wrong_direction_incorrect"
                        if group == "wrong"
                        else "correct"
                    )

                    for mode in causal_modes:
                        delta = build_edit(
                            mode=mode,
                            decomposition=decomp,
                            cb=cb,
                            gt=gt,
                            foil=foil,
                            target=t,
                            max_edit_norm=max_edit_norm,
                        )
                        dnorm = float(np.linalg.norm(delta))
                        triggered = dnorm > EPS

                        edited = core.score_edit(
                            model,
                            batch,
                            token_map,
                            decoder_layers,
                            li,
                            stage,
                            sp,
                            rp,
                            delta,
                        )

                        pred = restricted_pred(edited)
                        margin = edited[gt] - edited[foil]

                        row = {
                            "sid": sid,
                            "layer": li,
                            "stage": stage,
                            "generation_group": group,
                            "subgroup": subgroup,
                            "gt": gt,
                            "foil": foil,
                            "foil_kind": foil_kind,
                            "mode": mode,
                            "is_random_control": 0,

                            "direction_pred":
                                decomp["direction_pred"],
                            "direction_correct":
                                decomp["direction_correct"],

                            "S_abs": decomp["S_abs"],
                            "A_GT": decomp["A_GT"],
                            "s_GT": decomp["s_GT"],
                            "s_foil":
                                decomp["proto_scores"][foil],

                            "triggered": int(triggered),
                            "edit_norm": dnorm,

                            "baseline_pred": base_pred,
                            "baseline_correct": base_correct,
                            "baseline_margin": base_margin,

                            "edited_pred": pred,
                            "edited_correct": int(pred == gt),
                            "edited_margin": margin,

                            "margin_gain":
                                margin - base_margin,
                            "W2C": int(
                                base_correct == 0
                                and pred == gt
                            ),
                            "C2W": int(
                                base_correct == 1
                                and pred != gt
                            ),
                        }
                        rows.append(row)

                        if random_controls and triggered:
                            rd = random_orthogonal_delta(
                                dnorm,
                                cb["basis"],
                                len(delta),
                                seed=(
                                    seed
                                    + sid * 100003
                                    + li * 1009
                                    + (0 if stage == "pre_attn" else 3001)
                                    + hash(mode) % 1000
                                ),
                            )

                            random_scores = core.score_edit(
                                model,
                                batch,
                                token_map,
                                decoder_layers,
                                li,
                                stage,
                                sp,
                                rp,
                                rd,
                            )
                            rpred = restricted_pred(
                                random_scores
                            )
                            rmargin = (
                                random_scores[gt]
                                - random_scores[foil]
                            )

                            rr = dict(row)
                            rr.update({
                                "mode": f"random_{mode}",
                                "is_random_control": 1,
                                "edited_pred": rpred,
                                "edited_correct":
                                    int(rpred == gt),
                                "edited_margin": rmargin,
                                "margin_gain":
                                    rmargin - base_margin,
                                "W2C": int(
                                    base_correct == 0
                                    and rpred == gt
                                ),
                                "C2W": int(
                                    base_correct == 1
                                    and rpred != gt
                                ),
                            })
                            rows.append(rr)

            del batch

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[CAUSAL ERROR sid={sid}] "
                f"{type(e).__name__}: {e}"
            )

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows, errors


def summarize_causal_firststep(rows):
    out = []

    buckets = defaultdict(list)

    for r in rows:
        keys = [
            (
                int(r["layer"]),
                str(r["stage"]),
                str(r["mode"]),
                str(r["subgroup"]),
            ),
        ]
        # Also aggregate all wrong samples.
        if str(r["generation_group"]) == "wrong":
            keys.append(
                (
                    int(r["layer"]),
                    str(r["stage"]),
                    str(r["mode"]),
                    "wrong_all",
                )
            )

        for key in keys:
            buckets[key].append(r)

    for key, rr in sorted(buckets.items()):
        li, stage, mode, subgroup = key

        targeted = not mode.startswith("random_")

        # Specificity against matched random mode, if present.
        specific_gain = float("nan")
        base_mode = (
            mode[len("random_"):]
            if mode.startswith("random_")
            else mode
        )

        if targeted:
            rand_key = (
                li,
                stage,
                f"random_{base_mode}",
                subgroup,
            )
            rand_rr = buckets.get(rand_key, [])
            if rand_rr:
                specific_gain = (
                    safe_mean(r["margin_gain"] for r in rr)
                    - safe_mean(r["margin_gain"] for r in rand_rr)
                )

        out.append({
            "layer": li,
            "stage": stage,
            "mode": mode,
            "subgroup": subgroup,
            "n": len(rr),

            "trigger_rate": safe_frac(
                int(r["triggered"]) == 1
                for r in rr
            ),
            "mean_edit_norm": safe_mean(
                r["edit_norm"] for r in rr
            ),

            "baseline_acc": safe_mean(
                r["baseline_correct"] for r in rr
            ),
            "edited_acc": safe_mean(
                r["edited_correct"] for r in rr
            ),
            "acc_gain": (
                safe_mean(r["edited_correct"] for r in rr)
                - safe_mean(r["baseline_correct"] for r in rr)
            ),

            "mean_margin_gain": safe_mean(
                r["margin_gain"] for r in rr
            ),
            "specific_margin_gain_vs_random":
                specific_gain,

            "W2C": int(sum(int(r["W2C"]) for r in rr)),
            "C2W": int(sum(int(r["C2W"]) for r in rr)),
            "net": int(
                sum(int(r["W2C"]) for r in rr)
                - sum(int(r["C2W"]) for r in rr)
            ),
        })

    return out


# =============================================================================
# Full generation validation on top causal candidates
# =============================================================================

def parse_generated_relation(text: str) -> Optional[str]:
    s = text.strip().lower()
    pats = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bunder(?:neath)?\b"),
    ]

    hits = []
    for rel, pat in pats:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))

    if not hits:
        return None

    hits.sort()
    return hits[0][1]


def generate_from_batch(
    model,
    processor,
    batch,
    max_new_tokens,
):
    input_len = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    suffix = generated[0, input_len:]
    text = processor.tokenizer.decode(
        suffix,
        skip_special_tokens=True,
    ).strip()

    del generated
    return text, parse_generated_relation(text)


def generate_with_stage_edit(
    *,
    model,
    processor,
    batch,
    decoder_layers,
    layer,
    stage,
    sp,
    rp,
    delta,
    max_new_tokens,
):
    if stage == "pre_attn":
        hook = core.PreIntervention(
            decoder_layers[layer],
            sp,
            rp,
            delta,
        )
    elif stage == "post_attn":
        hook = core.PostIntervention(
            decoder_layers[layer].self_attn,
            sp,
            rp,
            delta,
        )
    else:
        raise ValueError(stage)

    try:
        text, pred = generate_from_batch(
            model,
            processor,
            batch,
            max_new_tokens,
        )
        if not hook.applied:
            raise RuntimeError(
                f"generation hook not applied: L{layer} {stage}"
            )
        return text, pred
    finally:
        hook.close()


def choose_generation_candidates(
    causal_summary,
    top_k,
):
    if top_k <= 0:
        return []

    # Magnitude-only is the direct test of "spatial information strength".
    candidates = []

    for r in causal_summary:
        if r["mode"] != "magnitude_only":
            continue
        if r["subgroup"] != "wrong_direction_correct":
            continue

        spec = float(r["specific_margin_gain_vs_random"])
        gain = float(r["mean_margin_gain"])
        score = (
            spec if math.isfinite(spec)
            else gain
        )

        candidates.append(
            (
                score,
                int(r["layer"]),
                str(r["stage"]),
            )
        )

    candidates.sort(reverse=True)

    picked = []
    seen = set()

    for score, li, stage in candidates:
        key = (li, stage)
        if key in seen:
            continue
        seen.add(key)
        picked.append(key)
        if len(picked) >= top_k:
            break

    return picked


def generation_validation(
    *,
    model,
    processor,
    decoder_layers,
    candidates,
    modes,
    eval_sids,
    max_samples,
    records,
    meta,
    vectors,
    codebooks,
    targets,
    device,
    prompt_template,
    max_new_tokens,
    max_edit_norm,
    seed,
):
    if not candidates or max_samples <= 0:
        return [], []

    rng = random.Random(seed + 9001)

    wrong = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "wrong"
    ]
    correct = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "correct"
    ]

    rng.shuffle(wrong)
    rng.shuffle(correct)

    nw = min(len(wrong), max_samples // 2)
    nc = min(len(correct), max_samples - nw)

    selected = wrong[:nw] + correct[:nc]
    if len(selected) < max_samples:
        remain = [
            sid for sid in eval_sids
            if sid not in set(selected)
        ]
        rng.shuffle(remain)
        selected += remain[
            : max_samples - len(selected)
        ]

    selected = sorted(set(selected))

    rows = []

    for sid in tqdm(
        selected,
        desc="full generation causal validation",
    ):
        rec = records[sid]
        image = Image.open(rec.image_path).convert("RGB")

        try:
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            batch, sp, rp = core.build_batch(
                processor,
                rec,
                question,
                image,
                device,
            )

            gt = meta["gt"][sid]
            cached_group = (
                meta["generation"][sid]["generation_group"]
            )
            cached_pred = (
                meta["generation"][sid]["generation_pred"]
            )

            # Fresh baseline generation, so W2C/C2W compares exactly the same
            # runtime/model with and without intervention.
            baseline_text, baseline_pred = generate_from_batch(
                model,
                processor,
                batch,
                max_new_tokens,
            )
            baseline_correct = int(
                baseline_pred == gt
            )

            # Define foil for edit construction.
            if (
                cached_group == "wrong"
                and cached_pred in REL2ID
                and cached_pred != gt
            ):
                foil = cached_pred
            else:
                # Use Direction strongest competitor at the candidate point.
                li0, stage0 = candidates[0]
                d0 = decompose_vector(
                    vectors[sid][(li0, stage0)],
                    codebooks[(li0, stage0)],
                    gt,
                )
                foil = d0["max_nonGT_relation"]

            for li, stage in candidates:
                cb = codebooks[(li, stage)]
                decomp = decompose_vector(
                    vectors[sid][(li, stage)],
                    cb,
                    gt,
                )
                t = targets[(li, stage, gt)]

                subgroup = (
                    "wrong_direction_correct"
                    if (
                        cached_group == "wrong"
                        and decomp["direction_correct"] == 1
                    )
                    else
                    "wrong_direction_incorrect"
                    if cached_group == "wrong"
                    else "correct"
                )

                for mode in modes:
                    delta = build_edit(
                        mode=mode,
                        decomposition=decomp,
                        cb=cb,
                        gt=gt,
                        foil=foil,
                        target=t,
                        max_edit_norm=max_edit_norm,
                    )

                    text, pred = generate_with_stage_edit(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        layer=li,
                        stage=stage,
                        sp=sp,
                        rp=rp,
                        delta=delta,
                        max_new_tokens=max_new_tokens,
                    )

                    edited_correct = int(pred == gt)

                    rows.append({
                        "sid": sid,
                        "layer": li,
                        "stage": stage,
                        "mode": mode,
                        "subgroup": subgroup,
                        "gt": gt,
                        "foil": foil,

                        "cached_generation_group":
                            cached_group,
                        "cached_generation_pred":
                            cached_pred,

                        "fresh_baseline_text":
                            baseline_text,
                        "fresh_baseline_pred":
                            baseline_pred,
                        "fresh_baseline_correct":
                            baseline_correct,

                        "edited_text": text,
                        "edited_pred": pred,
                        "edited_correct": edited_correct,

                        "edit_norm":
                            float(np.linalg.norm(delta)),
                        "triggered":
                            int(np.linalg.norm(delta) > EPS),

                        "W2C": int(
                            baseline_correct == 0
                            and edited_correct == 1
                        ),
                        "C2W": int(
                            baseline_correct == 1
                            and edited_correct == 0
                        ),
                    })

            del batch

        finally:
            image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    buckets = defaultdict(list)
    for r in rows:
        buckets[
            (
                int(r["layer"]),
                str(r["stage"]),
                str(r["mode"]),
                str(r["subgroup"]),
            )
        ].append(r)

        if str(r["cached_generation_group"]) == "wrong":
            buckets[
                (
                    int(r["layer"]),
                    str(r["stage"]),
                    str(r["mode"]),
                    "wrong_all",
                )
            ].append(r)

    summary = []

    for key, rr in sorted(buckets.items()):
        li, stage, mode, subgroup = key

        bacc = safe_mean(
            r["fresh_baseline_correct"] for r in rr
        )
        eacc = safe_mean(
            r["edited_correct"] for r in rr
        )

        summary.append({
            "layer": li,
            "stage": stage,
            "mode": mode,
            "subgroup": subgroup,
            "n": len(rr),

            "trigger_rate": safe_frac(
                int(r["triggered"]) == 1
                for r in rr
            ),
            "mean_edit_norm": safe_mean(
                r["edit_norm"] for r in rr
            ),

            "fresh_baseline_acc": bacc,
            "edited_generation_acc": eacc,
            "generation_acc_gain":
                eacc - bacc,

            "W2C":
                int(sum(int(r["W2C"]) for r in rr)),
            "C2W":
                int(sum(int(r["C2W"]) for r in rr)),
            "net": int(
                sum(int(r["W2C"]) for r in rr)
                - sum(int(r["C2W"]) for r in rr)
            ),
        })

    return rows, summary


# =============================================================================
# Console output
# =============================================================================

def print_decomposition_summary(rows):
    print("\n" + "=" * 178)
    print("DIRECTION EVIDENCE STRENGTH — CORRECT vs WRONG")
    print("=" * 178)
    print(
        "layer stage | Sabs cor/wr gap | Sfrac cor/wr gap | "
        "A_GT cor/wr gap | sGT cor/wr gap | maxNonGT cor/wr gap | pairGap"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} | "
            f"{float(r['S_abs_correct']):+7.3f}/"
            f"{float(r['S_abs_wrong']):+7.3f} "
            f"{float(r['S_abs_gap_CminusW']):+7.3f} | "
            f"{float(r['S_frac_correct']):.3f}/"
            f"{float(r['S_frac_wrong']):.3f} "
            f"{float(r['S_frac_gap_CminusW']):+6.3f} | "
            f"{float(r['A_GT_correct']):+6.3f}/"
            f"{float(r['A_GT_wrong']):+6.3f} "
            f"{float(r['A_GT_gap_CminusW']):+6.3f} | "
            f"{float(r['s_GT_correct']):+7.3f}/"
            f"{float(r['s_GT_wrong']):+7.3f} "
            f"{float(r['s_GT_gap_CminusW']):+7.3f} | "
            f"{float(r['s_max_nonGT_correct']):+7.3f}/"
            f"{float(r['s_max_nonGT_wrong']):+7.3f} "
            f"{float(r['s_max_nonGT_gap_CminusW']):+7.3f} | "
            f"{float(r['GT_minus_maxNonGT_gap_CminusW']):+7.3f}"
        )


def print_wrong_diagnosis(rows):
    print("\n" + "=" * 158)
    print("GENERATION-WRONG: WHAT KIND OF SPATIAL WEAKNESS?")
    print("=" * 158)
    print(
        "layer stage | dirCorrect magWeak orientWeak GTweak foilEx pairWeak | "
        "mean Sdef Adef GTdef foilEx pairDef"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} | "
            f"{float(r['direction_correct_rate']):.3f} "
            f"{float(r['magnitude_weak_rate']):.3f} "
            f"{float(r['orientation_weak_rate']):.3f} "
            f"{float(r['GT_component_weak_rate']):.3f} "
            f"{float(r['foil_component_excess_rate']):.3f} "
            f"{float(r['pair_weak_rate']):.3f} | "
            f"{float(r['mean_S_deficit']):+7.3f} "
            f"{float(r['mean_A_deficit']):+6.3f} "
            f"{float(r['mean_GT_deficit']):+7.3f} "
            f"{float(r['mean_foil_excess']):+7.3f} "
            f"{float(r['mean_pair_deficit']):+7.3f}"
        )


def print_attention_summary(rows):
    print("\n" + "=" * 158)
    print("ATTENTION ACCUMULATION — POST minus PRE")
    print("=" * 158)
    print(
        "layer | dS cor/wr gap | dA cor/wr gap | dsGT cor/wr gap | "
        "dMaxNonGT cor/wr gap | dPair cor/wr gap"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} | "
            f"{float(r['delta_S_abs_correct']):+7.3f}/"
            f"{float(r['delta_S_abs_wrong']):+7.3f} "
            f"{float(r['delta_S_abs_gap_CminusW']):+7.3f} | "
            f"{float(r['delta_A_GT_correct']):+6.3f}/"
            f"{float(r['delta_A_GT_wrong']):+6.3f} "
            f"{float(r['delta_A_GT_gap_CminusW']):+6.3f} | "
            f"{float(r['delta_s_GT_correct']):+7.3f}/"
            f"{float(r['delta_s_GT_wrong']):+7.3f} "
            f"{float(r['delta_s_GT_gap_CminusW']):+7.3f} | "
            f"{float(r['delta_s_max_nonGT_correct']):+7.3f}/"
            f"{float(r['delta_s_max_nonGT_wrong']):+7.3f} "
            f"{float(r['delta_s_max_nonGT_gap_CminusW']):+7.3f} | "
            f"{float(r['delta_GT_minus_maxNonGT_correct']):+7.3f}/"
            f"{float(r['delta_GT_minus_maxNonGT_wrong']):+7.3f} "
            f"{float(r['delta_GT_minus_maxNonGT_gap_CminusW']):+7.3f}"
        )


def print_causal_summary(rows):
    print("\n" + "=" * 178)
    print("CAUSAL VALIDATION — FIRST-STEP RELATION DECISION")
    print("=" * 178)
    print(
        "layer stage mode subgroup N | trigger editNorm | "
        "acc base->edit gain | marginGain specificVsRandom | W2C C2W net"
    )

    for r in rows:
        if r["mode"].startswith("random_"):
            continue
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} "
            f"{str(r['mode']):25s} "
            f"{str(r['subgroup']):24s} "
            f"{int(r['n']):3d} | "
            f"{float(r['trigger_rate']):.3f} "
            f"{float(r['mean_edit_norm']):6.3f} | "
            f"{float(r['baseline_acc']):.3f}->"
            f"{float(r['edited_acc']):.3f} "
            f"{float(r['acc_gain']):+6.3f} | "
            f"{float(r['mean_margin_gain']):+8.4f} "
            f"{float(r['specific_margin_gain_vs_random']):+8.4f} | "
            f"{int(r['W2C']):3d} "
            f"{int(r['C2W']):3d} "
            f"{int(r['net']):+4d}"
        )


def print_generation_summary(rows):
    if not rows:
        return

    print("\n" + "=" * 150)
    print("FULL model.generate() VALIDATION")
    print("=" * 150)
    print(
        "layer stage mode subgroup N | trigger | "
        "generation acc base->edit gain | W2C C2W net"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} "
            f"{str(r['mode']):25s} "
            f"{str(r['subgroup']):24s} "
            f"{int(r['n']):3d} | "
            f"{float(r['trigger_rate']):.3f} | "
            f"{float(r['fresh_baseline_acc']):.3f}->"
            f"{float(r['edited_generation_acc']):.3f} "
            f"{float(r['generation_acc_gain']):+6.3f} | "
            f"{int(r['W2C']):3d} "
            f"{int(r['C2W']):3d} "
            f"{int(r['net']):+4d}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_dir = Path(args.direction_dir)
    failure_dir = (
        Path(args.failure_dir)
        if args.failure_dir is not None
        else None
    )

    meta = core.load_meta(direction_dir)

    records_list, _audit = core.base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records = {
        int(r.sid): r
        for r in records_list
    }

    # Model.
    spec = core.base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    dtype = core.base.resolve_dtype(spec.dtype_name)

    kw: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")

    try:
        model = cls.from_pretrained(
            spec.repo_id,
            dtype=dtype,
            **kw,
        )
    except TypeError:
        model = cls.from_pretrained(
            spec.repo_id,
            torch_dtype=dtype,
            **kw,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.base.configure_processor(
        model,
        processor,
    )

    device = torch.device(args.device)

    decoder_layers, layer_path = core.decoder_layers(
        model
    )
    n_layers = len(decoder_layers)

    print(
        f"[decoder] {layer_path}; n_layers={n_layers}"
    )

    selected_layers, selection_audit = core.choose_layers(
        args.layers,
        n_layers,
        failure_dir,
        args.min_role_gap,
    )
    print("[selected problem layers]", selected_layers)

    write_csv(
        out_dir / "selected_problem_layers.csv",
        selection_audit,
    )

    # -------------------------------------------------------------------------
    # TRAIN stage codebooks.
    # -------------------------------------------------------------------------
    if args.prepost_dir is not None:
        codebooks, codebook_diag = load_codebooks_from_prepost(
            Path(args.prepost_dir),
            selected_layers,
        )
        print(
            "[codebook] reused",
            Path(args.prepost_dir) / "train_stage_vectors.npz",
        )
    else:
        train_sids = [
            sid
            for sid in meta["idx_by_sid"]
            if meta["split"].get(sid, "") == "train"
            and sid in records
            and meta["gt"].get(sid, "") in REL2ID
        ]

        if args.max_train is not None:
            rng = random.Random(args.seed)
            rng.shuffle(train_sids)
            train_sids = train_sids[: args.max_train]

        codebooks, codebook_diag = (
            core.fit_actual_point_codebooks(
                model,
                processor,
                decoder_layers,
                selected_layers,
                train_sids,
                records,
                meta,
                device,
                args.prompt_template,
                out_dir,
            )
        )

    write_csv(
        out_dir / "stage_codebook_diagnostics.csv",
        codebook_diag,
    )

    # -------------------------------------------------------------------------
    # Eval decomposition.
    # -------------------------------------------------------------------------
    eval_sids = select_eval_sids(
        meta,
        records,
        args.eval_split,
        args.max_eval,
        args.seed,
    )

    counts = Counter(
        meta["generation"][sid]["generation_group"]
        for sid in eval_sids
    )
    print(
        f"[eval] N={len(eval_sids)} groups={dict(counts)}"
    )

    decomposition_rows, vectors, errors = (
        collect_eval_decomposition(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            selected_layers=selected_layers,
            codebooks=codebooks,
            records=records,
            meta=meta,
            eval_sids=eval_sids,
            device=device,
            prompt_template=args.prompt_template,
            out_dir=out_dir,
            save_every=args.save_every,
        )
    )

    if not decomposition_rows:
        raise RuntimeError(
            "No decomposition rows were produced."
        )

    correct_wrong_summary = summarize_correct_wrong(
        decomposition_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "correct_wrong_strength_summary.csv",
        correct_wrong_summary,
    )

    targets, target_rows = build_control_targets(
        decomposition_rows,
        args.weak_quantile,
        args.excess_quantile,
        args.target_stat,
    )
    write_csv(
        out_dir / "correct_strength_targets.csv",
        target_rows,
    )

    wrong_diag = build_wrong_diagnosis(
        decomposition_rows,
        targets,
    )
    write_csv(
        out_dir / "wrong_strength_diagnosis.csv",
        wrong_diag,
    )

    wrong_diag_summary = summarize_wrong_diagnosis(
        wrong_diag,
        selected_layers,
    )
    write_csv(
        out_dir / "wrong_strength_diagnosis_summary.csv",
        wrong_diag_summary,
    )

    trans_rows = attention_transition_rows(
        decomposition_rows
    )
    write_csv(
        out_dir / "per_sample_attention_strength_transition.csv",
        trans_rows,
    )

    trans_summary = summarize_attention_transition(
        trans_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "attention_strength_transition_summary.csv",
        trans_summary,
    )

    print_decomposition_summary(
        correct_wrong_summary
    )
    print_wrong_diagnosis(
        wrong_diag_summary
    )
    print_attention_summary(
        trans_summary
    )

    # -------------------------------------------------------------------------
    # First-step causal validation.
    # -------------------------------------------------------------------------
    causal_layers = select_causal_layers(
        args.causal_layers,
        selected_layers,
        correct_wrong_summary,
        n_layers,
    )
    causal_stages = parse_stages(
        args.causal_stages
    )
    causal_modes = parse_csv_words(
        args.causal_modes
    )

    valid_modes = {
        "magnitude_only",
        "gt_boost_hold_foil",
        "foil_suppress_hold_gt",
        "both",
    }
    bad = [
        x for x in causal_modes
        if x not in valid_modes
    ]
    if bad:
        raise ValueError(
            f"Unknown causal modes: {bad}"
        )

    print(
        f"\n[causal] layers={causal_layers}, "
        f"stages={causal_stages}, modes={causal_modes}"
    )

    causal_rows = []
    causal_summary = []

    if causal_layers:
        causal_sids = choose_causal_sids(
            meta,
            eval_sids,
            args.causal_max_wrong,
            args.causal_max_correct,
            args.seed,
        )

        token_map = core.relation_tokens(
            processor.tokenizer
        )

        causal_rows, causal_errors = causal_firststep_scan(
            model=model,
            processor=processor,
            token_map=token_map,
            decoder_layers=decoder_layers,
            causal_layers=causal_layers,
            causal_stages=causal_stages,
            causal_modes=causal_modes,
            causal_sids=causal_sids,
            records=records,
            meta=meta,
            vectors=vectors,
            codebooks=codebooks,
            targets=targets,
            device=device,
            prompt_template=args.prompt_template,
            random_controls=args.random_controls,
            max_edit_norm=args.max_edit_norm,
            seed=args.seed,
        )

        write_csv(
            out_dir / "causal_firststep_per_sample.csv",
            causal_rows,
        )
        write_csv(
            out_dir / "causal_errors.csv",
            causal_errors,
        )

        causal_summary = summarize_causal_firststep(
            causal_rows
        )
        write_csv(
            out_dir / "causal_firststep_summary.csv",
            causal_summary,
        )

        print_causal_summary(
            causal_summary
        )

    # -------------------------------------------------------------------------
    # Full generation validation on top layer-stage pairs.
    # -------------------------------------------------------------------------
    generation_candidates = choose_generation_candidates(
        causal_summary,
        args.generation_top_k,
    )

    print(
        "\n[generation candidates]",
        generation_candidates,
    )

    generation_rows = []
    generation_summary = []

    if generation_candidates:
        generation_modes = parse_csv_words(
            args.generation_modes
        )
        bad = [
            x for x in generation_modes
            if x not in valid_modes
        ]
        if bad:
            raise ValueError(
                f"Unknown generation modes: {bad}"
            )

        generation_rows, generation_summary = generation_validation(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            candidates=generation_candidates,
            modes=generation_modes,
            eval_sids=eval_sids,
            max_samples=args.generation_max_samples,
            records=records,
            meta=meta,
            vectors=vectors,
            codebooks=codebooks,
            targets=targets,
            device=device,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            max_edit_norm=args.max_edit_norm,
            seed=args.seed,
        )

        write_csv(
            out_dir / "generation_validation_per_sample.csv",
            generation_rows,
        )
        write_csv(
            out_dir / "generation_validation_summary.csv",
            generation_summary,
        )

        print_generation_summary(
            generation_summary
        )

    meta_out = {
        "experiment":
            "Direction evidence strength decomposition + causal validation",

        "selected_layers": selected_layers,
        "causal_layers": causal_layers,
        "causal_stages": causal_stages,
        "generation_candidates":
            generation_candidates,

        "n_eval": len(eval_sids),
        "eval_generation_groups": dict(counts),

        "definitions": {
            "S_abs":
                "norm of centered residual projected into TRAIN-derived "
                "2D spatial subspace",
            "S_frac":
                "S_abs / norm(centered residual)",
            "A_GT":
                "cosine between projected spatial component and projected "
                "GT relation prototype",
            "s_GT":
                "centered residual dot normalized GT prototype",
            "magnitude_only":
                "scale only the sample's existing spatial projection to "
                "correct-control target magnitude; orientation unchanged",
            "gt_boost_hold_foil":
                "minimum-L2 edit raising GT prototype score while holding "
                "foil prototype score fixed",
            "foil_suppress_hold_gt":
                "minimum-L2 edit reducing foil prototype score while holding "
                "GT prototype score fixed",
        },

        "strongest_information_weakness_test":
            "magnitude_only improves wrong_direction_correct samples more "
            "than norm-matched non-spatial random controls",

        "warning":
            "GT/foil coordinate interventions are oracle diagnostic tests. "
            "Magnitude-only is less oracle because it preserves each sample's "
            "own spatial direction, but the target magnitude still comes from "
            "same-GT correct controls.",
    }

    (out_dir / "summary.json").write_text(
        json.dumps(
            meta_out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "selected_problem_layers.csv",
        "stage_codebook_diagnostics.csv",
        "per_sample_direction_strength.csv",
        "eval_stage_residual_vectors.npz",
        "correct_wrong_strength_summary.csv",
        "correct_strength_targets.csv",
        "wrong_strength_diagnosis.csv",
        "wrong_strength_diagnosis_summary.csv",
        "per_sample_attention_strength_transition.csv",
        "attention_strength_transition_summary.csv",
        "causal_firststep_per_sample.csv",
        "causal_firststep_summary.csv",
        "generation_validation_per_sample.csv",
        "generation_validation_summary.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
