#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four-way layer-update decomposition + correct-control deficit/excess scan.

CPU-only. Reuses the cached layer-wise Direction vectors and cached ACTUAL
model.generate() predictions.

Purpose
-------
For each transition L(l-1) -> L(l), do NOT compress the update into only one
GT-vs-wrong number. Instead decompose the same layer update onto ALL FOUR
relation prototypes:

    delta_r_l = r_l - r_{l-1}

    u_left  = delta_r_l dot mu_left,l
    u_right = delta_r_l dot mu_right,l
    u_above = delta_r_l dot mu_above,l
    u_below = delta_r_l dot mu_below,l

where r_l is the cached image-minus-noimage subject-reference residual
Direction vector.

For a sample with GT=g, define:
    u_GT
    u_finalWrong             (generation-wrong samples only)
    max_nonGT_update
    GT_minus_maxNonGT_update = u_GT - max_{c != GT} u_c

Correct-control comparison
--------------------------
Targets are learned from TRAIN samples whose cached ACTUAL model.generate()
result is correct, matched by GT relation.

For a generation-wrong test sample:

    GT deficit:
        target(u_GT | GT) - observed(u_GT)

    final-wrong excess:
        observed(u_finalWrong) - target(u_finalWrong | GT)

    other-competitor excess:
        max over remaining non-GT relations of
        observed(u_c) - target(u_c | GT)

    pairwise update deficit:
        target(u_GT - u_finalWrong | GT, same foil axis)
        - observed(u_GT - u_finalWrong)

Positive GT deficit = missing normal GT-supporting update.
Positive wrong excess = extra update along a wrong relation coordinate.
Positive pairwise deficit = the layer failed to separate GT from the actual
final generated wrong relation as much as correct controls normally do.

IMPORTANT
---------
The four relation prototypes are not orthogonal, so the four u_r values are
coordinates/projections, not independent causal "amounts of information".
For this reason the script also saves pairwise GT-vs-competitor updates and
prototype drift between adjacent layers.

Correctness groups come from:
    <direction-dir>/sample_split_and_generation.csv

Required cached files:
    <direction-dir>/vectors.npz
    <direction-dir>/sample_split_and_generation.csv

Recommended:
    python analyze_fourway_layer_update_failure_v1.py \
      --direction-dir output/qwen7b_layer_direction_scan_v1 \
      --eval-split test \
      --control-split train \
      --target-stat median \
      --output-dir output/qwen7b_fourway_layer_update_failure_v1 \
      --overwrite

Useful restricted window:
    --layers 12-20

Main outputs
------------
per_sample_fourway_update.csv
    All selected parseable generation-correct/wrong samples x layers.

wrong_sample_deficit_excess.csv
    Wrong samples with GT deficit / final-wrong excess / other excess.

fourway_role_summary.csv
    Test generation-correct vs generation-wrong comparison.

wrong_deficit_excess_summary.csv
    Layer-wise average deficits/excesses on wrong samples.

failure_type_summary.csv
    GT-deficit only / wrong-excess only / both / neither rates.

summary_by_gt_finalwrong.csv
    Relation-confusion-specific results.

worst_failure_layer_frequency.csv
    Which layers most often have the largest GT deficit, wrong excess,
    pairwise deficit, or other-competitor excess.

prototype_drift.csv
    Adjacent-layer prototype cosine, useful for checking basis changes.

top_candidate_layers.csv
    Candidate failure layers ranked primarily by pairwise update deficit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


# =============================================================================
# CLI / I/O
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--control-split",
        default="train",
        choices=["train", "test"],
        help="Generation-correct samples used to define normal update targets.",
    )
    p.add_argument(
        "--layers",
        default="all",
        help="Target layers. Example: all, 12-20, or 12,14,16,18,20.",
    )
    p.add_argument(
        "--target-stat",
        default="median",
        choices=["mean", "median", "q25", "q75"],
    )
    p.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help=(
            "Deficit/excess must exceed this value to count as present in "
            "failure-type summaries."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
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


def norm_relation(x: Any) -> str:
    t = str(x).strip().lower()
    aliases = {
        "left of": "left",
        "right of": "right",
        "on": "above",
        "over": "above",
        "up": "above",
        "under": "below",
        "beneath": "below",
        "down": "below",
    }
    if t in REL2ID:
        return t
    if t in aliases:
        return aliases[t]
    tokens = t.replace(",", " ").replace(".", " ").split()
    for rel in RELATIONS:
        if rel in tokens:
            return rel
    return ""


def safe_mean(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(a.mean()) if len(a) else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(np.median(a)) if len(a) else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(a.std()) if len(a) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def stat_value(vals: Sequence[float], kind: str) -> float:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    if kind == "mean":
        return float(np.mean(a))
    if kind == "median":
        return float(np.median(a))
    if kind == "q25":
        return float(np.quantile(a, 0.25))
    if kind == "q75":
        return float(np.quantile(a, 0.75))
    raise ValueError(kind)


def parse_layers(text: str, n_layers: int) -> List[int]:
    if str(text).strip().lower() == "all":
        return list(range(1, n_layers))

    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(chunk))

    out = sorted(set(out))
    out = [x for x in out if x != 0]
    bad = [x for x in out if not 1 <= x < n_layers]
    if bad:
        raise ValueError(
            f"Invalid target layers {bad}; valid update targets are "
            f"L1..L{n_layers - 1}"
        )
    if not out:
        raise ValueError("No valid update layers.")
    return out


# =============================================================================
# Assets / codebooks
# =============================================================================

def fit_codebook(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    center = X_train.mean(axis=0)
    Xc = X_train - center

    protos = []
    for rel in RELATIONS:
        mask = y_train == rel
        if not np.any(mask):
            raise RuntimeError(f"No training examples for {rel}")
        p = Xc[mask].mean(axis=0)
        n = float(np.linalg.norm(p))
        if n <= EPS:
            raise RuntimeError(f"Degenerate prototype for {rel}")
        protos.append(p / n)

    return (
        center.astype(np.float32),
        np.stack(protos).astype(np.float32),
    )


def load_assets(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    gen_path = direction_dir / "sample_split_and_generation.csv"

    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    with np.load(vec_path, allow_pickle=True) as z:
        arr = {k: z[k] for k in z.files}

    required = {"sample_index", "relation", "residual"}
    missing = required - set(arr)
    if missing:
        raise KeyError(
            f"{vec_path} missing required arrays: {sorted(missing)}"
        )

    rows = read_csv(gen_path)

    sids = arr["sample_index"].astype(np.int64)
    labels = np.asarray(
        [norm_relation(x) for x in arr["relation"]],
        dtype=object,
    )
    residual = np.asarray(arr["residual"], dtype=np.float32)

    if residual.ndim != 3:
        raise ValueError(
            f"Expected residual [N,L,D], got {residual.shape}"
        )

    idx_by_sid = {int(s): i for i, s in enumerate(sids.tolist())}

    split_by_sid = {}
    gen_by_sid = {}
    for r in rows:
        sid = int(r["sample_index"])
        split = str(r.get("split", "")).strip()
        pred = norm_relation(r.get("generation_pred", ""))
        group = str(r.get("generation_group", "")).strip().lower()

        idx = idx_by_sid.get(sid)
        gt = labels[idx] if idx is not None else ""

        if group not in ("correct", "wrong"):
            if pred in REL2ID and gt in REL2ID:
                group = "correct" if pred == gt else "wrong"

        split_by_sid[sid] = split
        gen_by_sid[sid] = {
            "generation_group": group,
            "generation_pred": pred,
            "generation_text": str(r.get("generation_text", "")),
        }

    # IMPORTANT: Direction prototypes keep the exact original training split.
    train_idx = np.asarray(
        [
            idx_by_sid[sid]
            for sid, split in split_by_sid.items()
            if split == "train" and sid in idx_by_sid
        ],
        dtype=np.int64,
    )
    if len(train_idx) == 0:
        raise RuntimeError("No train split rows for Direction codebook.")

    n_layers = residual.shape[1]
    codebooks = {}

    for li in range(n_layers):
        center, protos = fit_codebook(
            residual[train_idx, li, :],
            labels[train_idx],
        )
        codebooks[li] = {
            "center": center,
            "protos": protos,
        }

    return {
        "sids": sids,
        "labels": labels,
        "residual": residual,
        "idx_by_sid": idx_by_sid,
        "split": split_by_sid,
        "generation": gen_by_sid,
        "codebooks": codebooks,
        "n_layers": n_layers,
    }


# =============================================================================
# Per-layer update decomposition
# =============================================================================

def fourway_projection(
    delta_r: np.ndarray,
    protos: np.ndarray,
) -> np.ndarray:
    """
    Raw directional coordinates:
        [delta_r·mu_left, ..., delta_r·mu_below]
    Prototypes are unit-normalized.
    """
    return (delta_r @ protos.T).astype(np.float64)


def state_scores(
    r: np.ndarray,
    center: np.ndarray,
    protos: np.ndarray,
) -> np.ndarray:
    return ((r - center) @ protos.T).astype(np.float64)


def best_non_gt(scores: np.ndarray, gt: str) -> str:
    x = scores.copy()
    x[REL2ID[gt]] = -np.inf
    return RELATIONS[int(np.argmax(x))]


def build_per_sample_updates(
    assets,
    selected_layers,
    eval_split,
):
    rows = []

    for sid, idx in assets["idx_by_sid"].items():
        split = assets["split"].get(sid, "")
        if eval_split != "all" and split != eval_split:
            continue

        gen = assets["generation"].get(sid, {})
        group = gen.get("generation_group", "")
        pred = gen.get("generation_pred", "")
        gt = assets["labels"][idx]

        # Need parseable actual generation for correct/wrong grouping.
        if group not in ("correct", "wrong"):
            continue
        if gt not in REL2ID:
            continue
        if group == "wrong" and (
            pred not in REL2ID or pred == gt
        ):
            continue

        residual = assets["residual"][idx]

        for li in selected_layers:
            prev = li - 1
            delta_r = residual[li] - residual[prev]

            cur_cb = assets["codebooks"][li]
            prev_cb = assets["codebooks"][prev]

            u_cur = fourway_projection(
                delta_r,
                cur_cb["protos"],
            )
            u_prev = fourway_projection(
                delta_r,
                prev_cb["protos"],
            )

            # Current state competitor is useful for all samples, including
            # generation-correct samples.
            cur_state = state_scores(
                residual[li],
                cur_cb["center"],
                cur_cb["protos"],
            )
            current_comp = best_non_gt(cur_state, gt)

            gt_i = REL2ID[gt]
            non_gt_ids = [
                i for i in range(4) if i != gt_i
            ]
            max_non_gt_i = max(
                non_gt_ids,
                key=lambda i: float(u_cur[i]),
            )
            max_non_gt_rel = RELATIONS[max_non_gt_i]

            row = {
                "sid": sid,
                "split": split,
                "layer": li,
                "previous_layer": prev,
                "gt": gt,
                "generation_group": group,
                "generation_pred": pred,
                "current_state_best_competitor": current_comp,

                "update_norm": float(np.linalg.norm(delta_r)),

                "u_left": float(u_cur[REL2ID["left"]]),
                "u_right": float(u_cur[REL2ID["right"]]),
                "u_above": float(u_cur[REL2ID["above"]]),
                "u_below": float(u_cur[REL2ID["below"]]),

                "u_prevaxis_left": float(u_prev[REL2ID["left"]]),
                "u_prevaxis_right": float(u_prev[REL2ID["right"]]),
                "u_prevaxis_above": float(u_prev[REL2ID["above"]]),
                "u_prevaxis_below": float(u_prev[REL2ID["below"]]),

                "u_gt": float(u_cur[gt_i]),
                "u_prevaxis_gt": float(u_prev[gt_i]),

                "max_non_gt_relation": max_non_gt_rel,
                "max_non_gt_update": float(u_cur[max_non_gt_i]),
                "gt_minus_max_non_gt_update":
                    float(u_cur[gt_i] - u_cur[max_non_gt_i]),
            }

            # Save all three pairwise GT-vs-competitor update margins.
            for comp in RELATIONS:
                if comp == gt:
                    continue
                ci = REL2ID[comp]
                row[f"pair_gt_vs_{comp}"] = float(
                    u_cur[gt_i] - u_cur[ci]
                )
                row[f"pair_prevaxis_gt_vs_{comp}"] = float(
                    u_prev[gt_i] - u_prev[ci]
                )

            if group == "wrong":
                wi = REL2ID[pred]
                other_ids = [
                    i for i in range(4)
                    if i not in (gt_i, wi)
                ]
                other_rel = max(
                    other_ids,
                    key=lambda i: float(u_cur[i]),
                ) if other_ids else None

                row.update({
                    "u_final_wrong": float(u_cur[wi]),
                    "gt_minus_final_wrong_update":
                        float(u_cur[gt_i] - u_cur[wi]),
                    "u_prevaxis_final_wrong":
                        float(u_prev[wi]),
                    "prevaxis_gt_minus_final_wrong_update":
                        float(u_prev[gt_i] - u_prev[wi]),
                    "largest_other_wrong_relation":
                        RELATIONS[other_rel] if other_rel is not None else "",
                    "largest_other_wrong_update":
                        float(u_cur[other_rel])
                        if other_rel is not None else float("nan"),
                })
            else:
                row.update({
                    "u_final_wrong": float("nan"),
                    "gt_minus_final_wrong_update": float("nan"),
                    "u_prevaxis_final_wrong": float("nan"),
                    "prevaxis_gt_minus_final_wrong_update": float("nan"),
                    "largest_other_wrong_relation": "",
                    "largest_other_wrong_update": float("nan"),
                })

            rows.append(row)

    return rows


# =============================================================================
# Correct-control targets
# =============================================================================

def build_control_targets(
    assets,
    selected_layers,
    control_split,
    target_kind,
):
    """
    Build targets directly from TRAIN/selected-control generation-correct
    samples' layer updates, matched by GT.

    Targets are stored per (layer, GT, direction-coordinate), plus pairwise
    GT-vs-competitor margins.
    """
    controls = defaultdict(list)

    for sid, idx in assets["idx_by_sid"].items():
        if assets["split"].get(sid, "") != control_split:
            continue

        gen = assets["generation"].get(sid, {})
        if gen.get("generation_group", "") != "correct":
            continue

        gt = assets["labels"][idx]
        if gt not in REL2ID:
            continue

        residual = assets["residual"][idx]

        for li in selected_layers:
            delta_r = residual[li] - residual[li - 1]
            u = fourway_projection(
                delta_r,
                assets["codebooks"][li]["protos"],
            )
            controls[(li, gt)].append(u)

    targets = {}
    rows = []

    for li in selected_layers:
        for gt in RELATIONS:
            vals = controls.get((li, gt), [])
            if not vals:
                raise RuntimeError(
                    f"No generation-correct controls for L{li}, GT={gt}, "
                    f"split={control_split}"
                )

            A = np.stack(vals).astype(np.float64)
            entry = {
                "n": len(A),
                "coord": {},
                "coord_mean": {},
                "coord_median": {},
                "coord_std": {},
                "pair": {},
                "pair_mean": {},
                "pair_median": {},
                "pair_std": {},
            }

            for rel in RELATIONS:
                col = A[:, REL2ID[rel]]
                entry["coord"][rel] = stat_value(col, target_kind)
                entry["coord_mean"][rel] = float(np.mean(col))
                entry["coord_median"][rel] = float(np.median(col))
                entry["coord_std"][rel] = float(np.std(col))

            gi = REL2ID[gt]
            for comp in RELATIONS:
                if comp == gt:
                    continue
                ci = REL2ID[comp]
                pair = A[:, gi] - A[:, ci]
                entry["pair"][comp] = stat_value(pair, target_kind)
                entry["pair_mean"][comp] = float(np.mean(pair))
                entry["pair_median"][comp] = float(np.median(pair))
                entry["pair_std"][comp] = float(np.std(pair))

            targets[(li, gt)] = entry

            for rel in RELATIONS:
                rows.append({
                    "layer": li,
                    "gt": gt,
                    "target_type": "coordinate",
                    "relation_or_competitor": rel,
                    "n_controls": entry["n"],
                    "target_stat": target_kind,
                    "target": entry["coord"][rel],
                    "mean": entry["coord_mean"][rel],
                    "median": entry["coord_median"][rel],
                    "std": entry["coord_std"][rel],
                })

            for comp in RELATIONS:
                if comp == gt:
                    continue
                rows.append({
                    "layer": li,
                    "gt": gt,
                    "target_type": "pairwise_gt_minus_competitor",
                    "relation_or_competitor": comp,
                    "n_controls": entry["n"],
                    "target_stat": target_kind,
                    "target": entry["pair"][comp],
                    "mean": entry["pair_mean"][comp],
                    "median": entry["pair_median"][comp],
                    "std": entry["pair_std"][comp],
                })

    return targets, rows


# =============================================================================
# Wrong-sample deficit / excess
# =============================================================================

def build_wrong_deficit_excess(
    per_rows,
    targets,
    positive_threshold,
):
    out = []

    for r in per_rows:
        if r["generation_group"] != "wrong":
            continue

        li = int(r["layer"])
        gt = str(r["gt"])
        wrong = str(r["generation_pred"])
        t = targets[(li, gt)]

        observed = {
            rel: float(r[f"u_{rel}"])
            for rel in RELATIONS
        }

        # Positive = missing expected GT update.
        gt_deficit = (
            float(t["coord"][gt]) - observed[gt]
        )

        # Positive = more wrong-coordinate update than correct controls.
        final_wrong_excess = (
            observed[wrong] - float(t["coord"][wrong])
        )

        other_rels = [
            rel for rel in RELATIONS
            if rel not in (gt, wrong)
        ]
        other_excesses = {
            rel: observed[rel] - float(t["coord"][rel])
            for rel in other_rels
        }
        if other_excesses:
            largest_other_rel = max(
                other_excesses,
                key=other_excesses.get,
            )
            largest_other_excess = float(
                other_excesses[largest_other_rel]
            )
        else:
            largest_other_rel = ""
            largest_other_excess = float("nan")

        observed_pair = (
            observed[gt] - observed[wrong]
        )
        target_pair = float(t["pair"][wrong])

        # Positive = less GT-vs-finalwrong separation than normal.
        pairwise_deficit = target_pair - observed_pair

        # Standardized diagnostics relative to correct-control spread.
        gt_std = max(float(t["coord_std"][gt]), EPS)
        wrong_std = max(float(t["coord_std"][wrong]), EPS)
        pair_std = max(float(t["pair_std"][wrong]), EPS)

        gt_def_present = gt_deficit > positive_threshold
        wrong_ex_present = final_wrong_excess > positive_threshold
        other_ex_present = (
            math.isfinite(largest_other_excess)
            and largest_other_excess > positive_threshold
        )

        if gt_def_present and wrong_ex_present:
            failure_type = "both_gt_deficit_and_finalwrong_excess"
        elif gt_def_present:
            failure_type = "gt_deficit_only"
        elif wrong_ex_present:
            failure_type = "finalwrong_excess_only"
        else:
            failure_type = "neither_gtdef_nor_finalwrongexcess"

        if other_ex_present:
            failure_type_with_other = (
                failure_type + "+other_competitor_excess"
            )
        else:
            failure_type_with_other = failure_type

        row = {
            "sid": int(r["sid"]),
            "layer": li,
            "gt": gt,
            "generation_pred": wrong,

            "u_gt": observed[gt],
            "target_u_gt": float(t["coord"][gt]),
            "gt_deficit": gt_deficit,
            "gt_deficit_z_by_control_std": gt_deficit / gt_std,

            "u_final_wrong": observed[wrong],
            "target_u_final_wrong": float(t["coord"][wrong]),
            "final_wrong_excess": final_wrong_excess,
            "final_wrong_excess_z_by_control_std":
                final_wrong_excess / wrong_std,

            "observed_gt_minus_finalwrong_update": observed_pair,
            "target_gt_minus_finalwrong_update": target_pair,
            "pairwise_update_deficit": pairwise_deficit,
            "pairwise_deficit_z_by_control_std":
                pairwise_deficit / pair_std,

            "largest_other_wrong_relation": largest_other_rel,
            "largest_other_wrong_excess": largest_other_excess,

            "gt_deficit_present": int(gt_def_present),
            "final_wrong_excess_present": int(wrong_ex_present),
            "other_competitor_excess_present": int(other_ex_present),

            "failure_type": failure_type,
            "failure_type_with_other": failure_type_with_other,
        }

        for rel in RELATIONS:
            row[f"observed_u_{rel}"] = observed[rel]
            row[f"target_u_{rel}"] = float(t["coord"][rel])
            row[f"excess_{rel}"] = (
                observed[rel] - float(t["coord"][rel])
            )

        out.append(row)

    return out


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_mean_ci(
    vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")

    obs = float(a.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        x = a[rng.integers(0, len(a), len(a))]
        boots[i] = x.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_group_gap(
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

    obs = float(b.mean() - a.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = bb.mean() - aa.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


# =============================================================================
# Summaries
# =============================================================================

def summarize_fourway_roles(
    per_rows,
    selected_layers,
    bootstrap,
    seed,
):
    """
    Canonical-role comparison:
      u_GT
      max non-GT update
      GT-minus-maxNonGT update

    This is available for both generation-correct and generation-wrong.
    """
    rng = np.random.default_rng(seed)
    out = []

    for li in selected_layers:
        rr = [r for r in per_rows if int(r["layer"]) == li]
        correct = [
            r for r in rr if r["generation_group"] == "correct"
        ]
        wrong = [
            r for r in rr if r["generation_group"] == "wrong"
        ]

        metrics = {
            "u_gt": (
                [float(r["u_gt"]) for r in correct],
                [float(r["u_gt"]) for r in wrong],
            ),
            "max_non_gt_update": (
                [float(r["max_non_gt_update"]) for r in correct],
                [float(r["max_non_gt_update"]) for r in wrong],
            ),
            "gt_minus_max_non_gt_update": (
                [
                    float(r["gt_minus_max_non_gt_update"])
                    for r in correct
                ],
                [
                    float(r["gt_minus_max_non_gt_update"])
                    for r in wrong
                ],
            ),
        }

        row = {
            "layer": li,
            "n_correct": len(correct),
            "n_wrong": len(wrong),
        }

        for name, (cvals, wvals) in metrics.items():
            gap, lo, hi = bootstrap_group_gap(
                cvals, wvals, bootstrap, rng
            )
            row[f"{name}_correct"] = safe_mean(cvals)
            row[f"{name}_wrong"] = safe_mean(wvals)
            row[f"{name}_wrong_minus_correct"] = gap
            row[f"{name}_gap_ci95_lo"] = lo
            row[f"{name}_gap_ci95_hi"] = hi

        out.append(row)

    return out


def summarize_wrong_deficit_excess(
    wrong_rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed + 1000)
    out = []

    for li in selected_layers:
        rr = [r for r in wrong_rows if int(r["layer"]) == li]

        row = {
            "layer": li,
            "n_wrong": len(rr),
        }

        for metric in (
            "gt_deficit",
            "final_wrong_excess",
            "largest_other_wrong_excess",
            "pairwise_update_deficit",
        ):
            vals = [float(r[metric]) for r in rr]
            m, lo, hi = bootstrap_mean_ci(
                vals, bootstrap, rng
            )
            row[f"mean_{metric}"] = m
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
            row[f"{metric}_positive_rate"] = safe_frac(
                float(v) > 0 for v in vals
            )

        row["gt_deficit_present_rate"] = safe_frac(
            int(r["gt_deficit_present"]) == 1 for r in rr
        )
        row["final_wrong_excess_present_rate"] = safe_frac(
            int(r["final_wrong_excess_present"]) == 1 for r in rr
        )
        row["other_competitor_excess_present_rate"] = safe_frac(
            int(r["other_competitor_excess_present"]) == 1
            for r in rr
        )

        out.append(row)

    return out


def summarize_failure_types(
    wrong_rows,
    selected_layers,
):
    out = []

    base_types = (
        "gt_deficit_only",
        "finalwrong_excess_only",
        "both_gt_deficit_and_finalwrong_excess",
        "neither_gtdef_nor_finalwrongexcess",
    )

    for li in selected_layers:
        rr = [r for r in wrong_rows if int(r["layer"]) == li]
        counts = Counter(r["failure_type"] for r in rr)
        n = len(rr)

        row = {"layer": li, "n_wrong": n}
        for typ in base_types:
            row[f"{typ}_count"] = counts[typ]
            row[f"{typ}_rate"] = (
                counts[typ] / n if n else float("nan")
            )

        row["other_competitor_excess_rate"] = safe_frac(
            int(r["other_competitor_excess_present"]) == 1
            for r in rr
        )
        out.append(row)

    return out


def summarize_by_gt_finalwrong(
    wrong_rows,
):
    buckets = defaultdict(list)
    for r in wrong_rows:
        buckets[
            (
                int(r["layer"]),
                str(r["gt"]),
                str(r["generation_pred"]),
            )
        ].append(r)

    out = []
    for (li, gt, wrong), rr in sorted(buckets.items()):
        out.append({
            "layer": li,
            "gt": gt,
            "final_wrong": wrong,
            "n": len(rr),
            "mean_gt_deficit": safe_mean(
                r["gt_deficit"] for r in rr
            ),
            "mean_final_wrong_excess": safe_mean(
                r["final_wrong_excess"] for r in rr
            ),
            "mean_other_competitor_excess": safe_mean(
                r["largest_other_wrong_excess"] for r in rr
            ),
            "mean_pairwise_update_deficit": safe_mean(
                r["pairwise_update_deficit"] for r in rr
            ),
            "gt_deficit_rate": safe_frac(
                int(r["gt_deficit_present"]) == 1
                for r in rr
            ),
            "final_wrong_excess_rate": safe_frac(
                int(r["final_wrong_excess_present"]) == 1
                for r in rr
            ),
            "both_rate": safe_frac(
                r["failure_type"]
                == "both_gt_deficit_and_finalwrong_excess"
                for r in rr
            ),
        })

    return out


def worst_failure_layers(
    wrong_rows,
    selected_layers,
):
    by_sid = defaultdict(list)
    for r in wrong_rows:
        by_sid[int(r["sid"])].append(r)

    counters = {
        "gt_deficit": Counter(),
        "final_wrong_excess": Counter(),
        "largest_other_wrong_excess": Counter(),
        "pairwise_update_deficit": Counter(),
    }

    per_sample = []

    for sid, rr in by_sid.items():
        rr = sorted(rr, key=lambda x: int(x["layer"]))
        row = {
            "sid": sid,
            "gt": rr[0]["gt"],
            "generation_pred": rr[0]["generation_pred"],
        }

        for metric in counters:
            best = max(
                rr,
                key=lambda x: float(x[metric]),
            )
            li = int(best["layer"])
            counters[metric][li] += 1
            row[f"max_{metric}_layer"] = li
            row[f"max_{metric}_value"] = float(best[metric])

        per_sample.append(row)

    n = len(by_sid)
    freq = []
    for li in selected_layers:
        row = {"layer": li, "n_wrong_samples": n}
        for metric, counter in counters.items():
            row[f"{metric}_worst_count"] = counter[li]
            row[f"{metric}_worst_rate"] = (
                counter[li] / n if n else float("nan")
            )
        freq.append(row)

    return per_sample, freq


def prototype_drift_rows(
    assets,
    selected_layers,
):
    rows = []
    for li in selected_layers:
        prev = li - 1
        p0 = assets["codebooks"][prev]["protos"]
        p1 = assets["codebooks"][li]["protos"]

        for rel in RELATIONS:
            a = p0[REL2ID[rel]]
            b = p1[REL2ID[rel]]
            cos = float(
                (a @ b)
                / max(
                    float(np.linalg.norm(a) * np.linalg.norm(b)),
                    EPS,
                )
            )
            rows.append({
                "layer": li,
                "previous_layer": prev,
                "relation": rel,
                "prototype_cosine_prev_to_current": cos,
            })

    return rows


def rank_candidate_layers(
    deficit_summary,
    role_summary,
    worst_freq,
):
    role = {int(r["layer"]): r for r in role_summary}
    worst = {int(r["layer"]): r for r in worst_freq}

    rows = []
    for r in deficit_summary:
        li = int(r["layer"])
        rr = role[li]
        ww = worst[li]

        rows.append({
            "layer": li,

            "mean_pairwise_update_deficit":
                r["mean_pairwise_update_deficit"],
            "pairwise_deficit_ci95_lo":
                r["pairwise_update_deficit_ci95_lo"],
            "pairwise_deficit_ci95_hi":
                r["pairwise_update_deficit_ci95_hi"],
            "pairwise_deficit_positive_rate":
                r["pairwise_update_deficit_positive_rate"],

            "mean_gt_deficit":
                r["mean_gt_deficit"],
            "gt_deficit_ci95_lo":
                r["gt_deficit_ci95_lo"],
            "gt_deficit_ci95_hi":
                r["gt_deficit_ci95_hi"],

            "mean_final_wrong_excess":
                r["mean_final_wrong_excess"],
            "final_wrong_excess_ci95_lo":
                r["final_wrong_excess_ci95_lo"],
            "final_wrong_excess_ci95_hi":
                r["final_wrong_excess_ci95_hi"],

            "mean_other_competitor_excess":
                r["mean_largest_other_wrong_excess"],

            # If GT-minus-maxNonGT is much lower on wrong samples, this is
            # another relation-agnostic sign of missing separation.
            "wrong_minus_correct_gt_minus_maxnonGT_gap":
                rr["gt_minus_max_non_gt_update_wrong_minus_correct"],
            "role_gap_ci95_lo":
                rr["gt_minus_max_non_gt_update_gap_ci95_lo"],
            "role_gap_ci95_hi":
                rr["gt_minus_max_non_gt_update_gap_ci95_hi"],

            "pairwise_deficit_worst_layer_rate":
                ww["pairwise_update_deficit_worst_rate"],
            "gt_deficit_worst_layer_rate":
                ww["gt_deficit_worst_rate"],
            "final_wrong_excess_worst_layer_rate":
                ww["final_wrong_excess_worst_rate"],
        })

    # Primary rank: how much GT-vs-actual-final-wrong update is missing
    # relative to correct controls.
    rows.sort(
        key=lambda x: float(
            x["mean_pairwise_update_deficit"]
        ),
        reverse=True,
    )

    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["pairwise_deficit_ci_positive"] = int(
            math.isfinite(float(r["pairwise_deficit_ci95_lo"]))
            and float(r["pairwise_deficit_ci95_lo"]) > 0
        )
        r["gt_deficit_ci_positive"] = int(
            math.isfinite(float(r["gt_deficit_ci95_lo"]))
            and float(r["gt_deficit_ci95_lo"]) > 0
        )
        r["final_wrong_excess_ci_positive"] = int(
            math.isfinite(float(r["final_wrong_excess_ci95_lo"]))
            and float(r["final_wrong_excess_ci95_lo"]) > 0
        )

    return rows


# =============================================================================
# Console
# =============================================================================

def print_role_summary(rows):
    print("\n" + "=" * 158)
    print("FOUR-WAY UPDATE ROLE SUMMARY — ACTUAL GENERATION CORRECT vs WRONG")
    print("=" * 158)
    print(
        "layer Ncor Nwr | GTupdate cor/wr  gap(W-C) | "
        "maxNonGT cor/wr gap(W-C) | GT-maxNonGT cor/wr gap(W-C)"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{int(r['n_correct']):3d} {int(r['n_wrong']):3d} | "
            f"{float(r['u_gt_correct']):+7.3f}/"
            f"{float(r['u_gt_wrong']):+7.3f} "
            f"{float(r['u_gt_wrong_minus_correct']):+7.3f} | "
            f"{float(r['max_non_gt_update_correct']):+7.3f}/"
            f"{float(r['max_non_gt_update_wrong']):+7.3f} "
            f"{float(r['max_non_gt_update_wrong_minus_correct']):+7.3f} | "
            f"{float(r['gt_minus_max_non_gt_update_correct']):+7.3f}/"
            f"{float(r['gt_minus_max_non_gt_update_wrong']):+7.3f} "
            f"{float(r['gt_minus_max_non_gt_update_wrong_minus_correct']):+7.3f}"
        )


def print_deficit_summary(rows):
    print("\n" + "=" * 166)
    print("GENERATION-WRONG: UPDATE DEFICIT / EXCESS VS TRAIN GENERATION-CORRECT CONTROLS")
    print("=" * 166)
    print(
        "layer N | GTdeficit       95%CI        pos% | "
        "finalWrongExcess    95%CI        pos% | "
        "otherExcess | pairDeficit      95%CI        pos%"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} {int(r['n_wrong']):3d} | "
            f"{float(r['mean_gt_deficit']):+7.3f} "
            f"[{float(r['gt_deficit_ci95_lo']):+6.3f},"
            f"{float(r['gt_deficit_ci95_hi']):+6.3f}] "
            f"{float(r['gt_deficit_positive_rate']):.3f} | "
            f"{float(r['mean_final_wrong_excess']):+7.3f} "
            f"[{float(r['final_wrong_excess_ci95_lo']):+6.3f},"
            f"{float(r['final_wrong_excess_ci95_hi']):+6.3f}] "
            f"{float(r['final_wrong_excess_positive_rate']):.3f} | "
            f"{float(r['mean_largest_other_wrong_excess']):+7.3f} | "
            f"{float(r['mean_pairwise_update_deficit']):+7.3f} "
            f"[{float(r['pairwise_update_deficit_ci95_lo']):+6.3f},"
            f"{float(r['pairwise_update_deficit_ci95_hi']):+6.3f}] "
            f"{float(r['pairwise_update_deficit_positive_rate']):.3f}"
        )


def print_failure_types(rows):
    print("\n" + "=" * 126)
    print("FAILURE TYPE RATES ON GENERATION-WRONG SAMPLES")
    print("=" * 126)
    print(
        "layer N | GTdef-only  WrongEx-only  BOTH  neither | otherCompetitorExcess"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} {int(r['n_wrong']):3d} | "
            f"{float(r['gt_deficit_only_rate']):.3f}       "
            f"{float(r['finalwrong_excess_only_rate']):.3f}        "
            f"{float(r['both_gt_deficit_and_finalwrong_excess_rate']):.3f}  "
            f"{float(r['neither_gtdef_nor_finalwrongexcess_rate']):.3f}   | "
            f"{float(r['other_competitor_excess_rate']):.3f}"
        )


def print_top(rows, n=12):
    print("\n" + "=" * 168)
    print("TOP CANDIDATE LAYERS — MISSING GT-vs-FINAL-WRONG UPDATE")
    print("=" * 168)
    print(
        "rank layer pairDef      95%CI       GTdef       95%CI      "
        "finalWrongEx     95%CI      otherEx  pairWorst%  GTworst%  WrongExWorst%"
    )

    for r in rows[:n]:
        print(
            f"{int(r['rank']):>3d}  L{int(r['layer']):02d} "
            f"{float(r['mean_pairwise_update_deficit']):+7.3f} "
            f"[{float(r['pairwise_deficit_ci95_lo']):+6.3f},"
            f"{float(r['pairwise_deficit_ci95_hi']):+6.3f}] "
            f"{float(r['mean_gt_deficit']):+7.3f} "
            f"[{float(r['gt_deficit_ci95_lo']):+6.3f},"
            f"{float(r['gt_deficit_ci95_hi']):+6.3f}] "
            f"{float(r['mean_final_wrong_excess']):+7.3f} "
            f"[{float(r['final_wrong_excess_ci95_lo']):+6.3f},"
            f"{float(r['final_wrong_excess_ci95_hi']):+6.3f}] "
            f"{float(r['mean_other_competitor_excess']):+7.3f} "
            f"{float(r['pairwise_deficit_worst_layer_rate']):.3f}      "
            f"{float(r['gt_deficit_worst_layer_rate']):.3f}     "
            f"{float(r['final_wrong_excess_worst_layer_rate']):.3f}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    direction_dir = Path(args.direction_dir)
    out_dir = Path(args.output_dir)

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_assets(direction_dir)
    selected_layers = parse_layers(
        args.layers,
        assets["n_layers"],
    )

    per_rows = build_per_sample_updates(
        assets=assets,
        selected_layers=selected_layers,
        eval_split=args.eval_split,
    )

    if not per_rows:
        raise RuntimeError("No parseable correct/wrong evaluation samples.")

    sample_ids = sorted({int(r["sid"]) for r in per_rows})
    group_counts = Counter(
        next(
            r["generation_group"]
            for r in per_rows
            if int(r["sid"]) == sid
        )
        for sid in sample_ids
    )

    targets, target_rows = build_control_targets(
        assets=assets,
        selected_layers=selected_layers,
        control_split=args.control_split,
        target_kind=args.target_stat,
    )

    wrong_rows = build_wrong_deficit_excess(
        per_rows=per_rows,
        targets=targets,
        positive_threshold=args.positive_threshold,
    )

    role_summary = summarize_fourway_roles(
        per_rows=per_rows,
        selected_layers=selected_layers,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    deficit_summary = summarize_wrong_deficit_excess(
        wrong_rows=wrong_rows,
        selected_layers=selected_layers,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    type_summary = summarize_failure_types(
        wrong_rows=wrong_rows,
        selected_layers=selected_layers,
    )

    by_relation = summarize_by_gt_finalwrong(
        wrong_rows=wrong_rows,
    )

    per_wrong_worst, worst_freq = worst_failure_layers(
        wrong_rows=wrong_rows,
        selected_layers=selected_layers,
    )

    drift = prototype_drift_rows(
        assets=assets,
        selected_layers=selected_layers,
    )

    ranked = rank_candidate_layers(
        deficit_summary=deficit_summary,
        role_summary=role_summary,
        worst_freq=worst_freq,
    )

    write_csv(
        out_dir / "per_sample_fourway_update.csv",
        per_rows,
    )
    write_csv(
        out_dir / "correct_control_update_targets.csv",
        target_rows,
    )
    write_csv(
        out_dir / "wrong_sample_deficit_excess.csv",
        wrong_rows,
    )
    write_csv(
        out_dir / "fourway_role_summary.csv",
        role_summary,
    )
    write_csv(
        out_dir / "wrong_deficit_excess_summary.csv",
        deficit_summary,
    )
    write_csv(
        out_dir / "failure_type_summary.csv",
        type_summary,
    )
    write_csv(
        out_dir / "summary_by_gt_finalwrong.csv",
        by_relation,
    )
    write_csv(
        out_dir / "per_wrong_sample_worst_failure_layer.csv",
        per_wrong_worst,
    )
    write_csv(
        out_dir / "worst_failure_layer_frequency.csv",
        worst_freq,
    )
    write_csv(
        out_dir / "prototype_drift.csv",
        drift,
    )
    write_csv(
        out_dir / "top_candidate_layers.csv",
        ranked,
    )

    print(
        f"[data] eval split={args.eval_split}; "
        f"parseable samples={len(sample_ids)}; "
        f"groups={dict(group_counts)}"
    )
    print(
        f"[controls] split={args.control_split}; "
        f"actual-generation correct only; target={args.target_stat}"
    )
    print(f"[layers] {selected_layers}")

    print_role_summary(role_summary)
    print_deficit_summary(deficit_summary)
    print_failure_types(type_summary)
    print_top(ranked)

    meta = {
        "experiment":
            "four-way layer update decomposition + correct-control deficit/excess",
        "direction_dir": str(direction_dir),
        "eval_split": args.eval_split,
        "control_split": args.control_split,
        "target_stat": args.target_stat,
        "positive_threshold": args.positive_threshold,
        "selected_layers": selected_layers,
        "n_eval_parseable": len(sample_ids),
        "generation_group_counts": dict(group_counts),
        "correctness_definition":
            "cached actual model.generate() prediction",
        "direction_update_definition":
            "delta_r_l = residual_l - residual_(l-1)",
        "fourway_coordinate_definition":
            "u_relation = delta_r_l dot mu_relation,l",
        "gt_deficit":
            "target_correct_control_u_GT - observed_wrong_u_GT",
        "final_wrong_excess":
            "observed_wrong_u_finalGeneratedWrong - "
            "target_correct_control_u_sameRelation",
        "pairwise_update_deficit":
            "target_correct_control_(u_GT-u_finalWrong) - "
            "observed_wrong_(u_GT-u_finalWrong)",
        "warning":
            "Relation prototypes are non-orthogonal. Four-way projections are "
            "diagnostic coordinates, not independent causal quantities.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "per_sample_fourway_update.csv",
        "correct_control_update_targets.csv",
        "wrong_sample_deficit_excess.csv",
        "fourway_role_summary.csv",
        "wrong_deficit_excess_summary.csv",
        "failure_type_summary.csv",
        "summary_by_gt_finalwrong.csv",
        "per_wrong_sample_worst_failure_layer.csv",
        "worst_failure_layer_frequency.csv",
        "prototype_drift.csv",
        "top_candidate_layers.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
