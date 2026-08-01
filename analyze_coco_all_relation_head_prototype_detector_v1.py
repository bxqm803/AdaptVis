#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-test-GT error detector from four relation prototypes in every attention head.

This script reuses vectors extracted by:

    analyze_coco_head_swap_error_detector_v1_20260801.py

For every outer CV training fold and every head h, it learns four prototypes from
TRAINING samples that the model generated correctly:

    p[h,left], p[h,right], p[h,above], p[h,below]

Three separate prototype banks are learned from:

    original object-pair state: A - B under "A relative to B?"
    swapped object-pair state:  A - B under "B relative to A?"
    stable object-pair state:   (original + swapped) / 2

At TEST time the script never selects a prototype using test GT.  It compares
one test activation with ALL FOUR prototypes in every head.  Legal no-test-GT
features include:

    * cosine to all four prototypes;
    * top-vs-second prototype margin and prototype entropy;
    * support for the model's own generated relation;
    * agreement between the internally preferred prototype and model output;
    * original/swapped prototype-score consistency;
    * aggregate relation scores and head-vote distributions.

GT and correctness are used only inside the OUTER TRAINING fold to build the
correct-relation prototypes and train the error detector.  Test GT is used only
for final evaluation and optional relation-balanced fold assignment.

Input directory must contain:

    config.json
    head_order.json
    swap_cells.jsonl
    vectors/sid_XXXXXX.npz

Main outputs:

    model_performance.csv
    repeat_performance.csv
    relation_performance.csv
    oof_predictions.csv
    selected_features_by_fold.csv
    selected_feature_stability.csv
    head_importance_summary.csv
    internal_relation_oof.csv
    summary.json

The target label is baseline free-generation error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required: {exc}")


SCRIPT_VERSION = "coco-all-relation-head-prototype-detector-v1"
RELATIONS: Tuple[str, ...] = ("left", "right", "above", "below")
RELATION_TO_INDEX = {relation: index for index, relation in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-8
STATES: Tuple[str, ...] = ("orig", "swap", "stable")
MODEL_GROUPS: Tuple[str, ...] = (
    "confidence",
    "prototype_global",
    "confidence_plus_prototype_global",
    "prototype_head_scores",
    "prototype_head",
    "confidence_plus_prototype_head",
)


# -----------------------------------------------------------------------------
# CLI and I/O
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Output directory from analyze_coco_head_swap_error_detector_v1_20260801.py.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument(
        "--stratify",
        choices=("relation_error", "error"),
        default="relation_error",
        help="Fold balancing only; test GT is never used to construct predictor features.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=512,
        help="Fold-local ANOVA feature selection; 0 keeps all features.",
    )
    parser.add_argument("--logreg-c", type=float, default=0.10)
    parser.add_argument("--l1-ratio", type=float, default=0.50)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--review-fraction", type=float, default=0.20)
    parser.add_argument(
        "--model-groups",
        default=",".join(MODEL_GROUPS),
        help="Comma-separated subset of supported feature groups.",
    )
    parser.add_argument(
        "--model-prediction-source",
        choices=("baseline", "original_closed"),
        default="baseline",
        help="Prediction available at test time and used for prototype/output agreement features.",
    )
    parser.add_argument(
        "--prototype-training",
        choices=("correct_only", "all_train"),
        default="correct_only",
        help="correct_only is the main normative model; all_train is a useful ablation.",
    )
    parser.add_argument(
        "--write-long-prototype-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write repeated-fold per-sample/per-head prototype scores; very large.",
    )
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ")
    aliases = {
        "left": "left",
        "left of": "left",
        "right": "right",
        "right of": "right",
        "on": "above",
        "over": "above",
        "above": "above",
        "under": "below",
        "below": "below",
    }
    return aliases.get(text)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------


def row_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def softmax_last(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=-1, keepdims=True), EPS)


def entropy_last(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return -np.sum(probability * np.log(np.maximum(probability, EPS)), axis=-1)


def top_margin(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64), axis=-1)
    return sorted_values[..., -1] - sorted_values[..., -2]


def score_vector_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denominator, EPS)


def gather_relation_score(scores: np.ndarray, relation_indices: np.ndarray) -> np.ndarray:
    """scores [N,H,R], relation_indices [N] -> [N,H]."""
    n_samples, n_heads, _ = scores.shape
    sample_index = np.arange(n_samples)[:, None]
    head_index = np.arange(n_heads)[None, :]
    relation_index = relation_indices[:, None]
    return scores[sample_index, head_index, relation_index]


def relation_margin_against_best_other(
    scores: np.ndarray, relation_indices: np.ndarray
) -> np.ndarray:
    chosen = gather_relation_score(scores, relation_indices)
    masked = np.array(scores, dtype=np.float64, copy=True)
    for sample_index, relation_index in enumerate(relation_indices.tolist()):
        masked[sample_index, :, int(relation_index)] = -np.inf
    return chosen - np.max(masked, axis=-1)


def parse_head_layer(head_name: str) -> int:
    match = re.fullmatch(r"L(\d+)H(\d+)", str(head_name))
    if not match:
        raise ValueError(f"Invalid head name {head_name!r}")
    return int(match.group(1))


# -----------------------------------------------------------------------------
# Load extracted head vectors
# -----------------------------------------------------------------------------


def deduplicate_rows(rows: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    output: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        output[int(row[key])] = dict(row)
    return [output[index] for index in sorted(output)]


def load_dataset(input_dir: Path, prediction_source: str) -> Dict[str, Any]:
    config_path = input_dir / "config.json"
    head_order_path = input_dir / "head_order.json"
    cells_path = input_dir / "swap_cells.jsonl"
    vector_dir = input_dir / "vectors"
    for path in (config_path, head_order_path, cells_path, vector_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    config = read_json(config_path)
    head_order = sorted(read_json(head_order_path), key=lambda row: int(row["index"]))
    head_names = [str(row["name"]) for row in head_order]
    head_layers = np.asarray([int(row["layer"]) for row in head_order], dtype=np.int64)

    rows = deduplicate_rows(read_jsonl(cells_path), "sid")
    selected_rows: List[Dict[str, Any]] = []
    originals: List[np.ndarray] = []
    swaps: List[np.ndarray] = []
    expected_shape: Optional[Tuple[int, int, int]] = None

    for row in rows:
        relation = normalize_relation(row.get("gt"))
        if relation not in RELATION_TO_INDEX:
            continue
        if prediction_source == "baseline":
            prediction = normalize_relation(row.get("baseline_prediction"))
        else:
            prediction = normalize_relation(row.get("original_closed_prediction"))
        if prediction not in RELATION_TO_INDEX:
            continue
        sid = int(row["sid"])
        vector_path = vector_dir / f"sid_{sid:06d}.npz"
        if not vector_path.exists():
            continue
        with np.load(vector_path, allow_pickle=False) as data:
            original = np.asarray(data["original_heads"], dtype=np.float32)
            swapped = np.asarray(data["swapped_heads"], dtype=np.float32)
        if original.shape != swapped.shape:
            raise RuntimeError(f"SID {sid}: original/swap mismatch {original.shape} vs {swapped.shape}")
        if expected_shape is None:
            expected_shape = tuple(original.shape)
        if tuple(original.shape) != expected_shape:
            raise RuntimeError(f"SID {sid}: expected {expected_shape}, got {original.shape}")
        if original.shape[0] != len(head_names) or original.shape[1] != 2:
            raise RuntimeError(
                f"SID {sid}: vector shape {original.shape} incompatible with heads={len(head_names)}"
            )
        selected_rows.append({**row, "gt": relation, "model_prediction": prediction})
        originals.append(original)
        swaps.append(swapped)

    if not selected_rows:
        raise RuntimeError("No usable extracted samples")

    original_heads = np.stack(originals, axis=0)
    swapped_heads = np.stack(swaps, axis=0)
    original_pair = original_heads[:, :, 0, :] - original_heads[:, :, 1, :]
    swapped_pair = swapped_heads[:, :, 0, :] - swapped_heads[:, :, 1, :]
    stable_pair = 0.5 * (original_pair + swapped_pair)

    relations = np.asarray([row["gt"] for row in selected_rows], dtype=object)
    relation_indices = np.asarray([RELATION_TO_INDEX[str(value)] for value in relations], dtype=np.int64)
    errors = np.asarray(
        [0 if bool_value(row.get("baseline_correct", False)) else 1 for row in selected_rows],
        dtype=np.int64,
    )
    model_predictions = np.asarray([row["model_prediction"] for row in selected_rows], dtype=object)
    model_prediction_indices = np.asarray(
        [RELATION_TO_INDEX[str(value)] for value in model_predictions], dtype=np.int64
    )
    sids = np.asarray([int(row["sid"]) for row in selected_rows], dtype=np.int64)

    return {
        "config": config,
        "rows": selected_rows,
        "head_names": head_names,
        "head_layers": head_layers,
        "sids": sids,
        "relations": relations,
        "relation_indices": relation_indices,
        "errors": errors,
        "model_predictions": model_predictions,
        "model_prediction_indices": model_prediction_indices,
        "vectors": {
            "orig": row_unit(original_pair).astype(np.float32),
            "swap": row_unit(swapped_pair).astype(np.float32),
            "stable": row_unit(stable_pair).astype(np.float32),
        },
    }


# -----------------------------------------------------------------------------
# Confidence baseline
# -----------------------------------------------------------------------------


def score_map(row: Mapping[str, Any], key: str) -> Dict[str, float]:
    raw = row.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Missing score mapping {key} for sid={row.get('sid')}")
    output: Dict[str, float] = {}
    for raw_relation, value in raw.items():
        relation = normalize_relation(raw_relation)
        if relation in RELATION_TO_INDEX:
            output[relation] = float(value)
    if set(output) != set(RELATIONS):
        raise ValueError(f"Incomplete score mapping {key} for sid={row.get('sid')}: {output}")
    return output


def confidence_features(rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    feature_rows: List[np.ndarray] = []
    names = [
        "confidence::orig_top_margin",
        "confidence::orig_max_probability",
        "confidence::orig_entropy",
        "confidence::swap_top_margin",
        "confidence::swap_max_probability",
        "confidence::swap_entropy",
        "confidence::prediction_pair_opposite",
        "confidence::top_margin_gap",
    ]
    for row in rows:
        original = score_map(row, "original_closed_scores")
        swapped = score_map(row, "swapped_closed_scores")
        original_values = np.asarray([original[r] for r in RELATIONS], dtype=np.float64)
        swapped_values = np.asarray([swapped[r] for r in RELATIONS], dtype=np.float64)
        original_probability = softmax_last(original_values)
        swapped_probability = softmax_last(swapped_values)
        original_margin = float(top_margin(original_values))
        swapped_margin = float(top_margin(swapped_values))
        original_prediction = normalize_relation(row.get("original_closed_prediction"))
        swapped_prediction = normalize_relation(row.get("swapped_closed_prediction"))
        pair_opposite = float(
            original_prediction in OPPOSITE
            and swapped_prediction == OPPOSITE.get(original_prediction)
        )
        feature_rows.append(
            np.asarray(
                [
                    original_margin,
                    float(np.max(original_probability)),
                    float(entropy_last(original_probability)),
                    swapped_margin,
                    float(np.max(swapped_probability)),
                    float(entropy_last(swapped_probability)),
                    pair_opposite,
                    abs(original_margin - swapped_margin),
                ],
                dtype=np.float64,
            )
        )
    return np.stack(feature_rows, axis=0), names


# -----------------------------------------------------------------------------
# Fold-local prototype banks
# -----------------------------------------------------------------------------


def build_prototype_sums(
    *,
    vectors: np.ndarray,  # [N,H,D] unit vectors
    train_indices: np.ndarray,
    relations: np.ndarray,
    errors: np.ndarray,
    correct_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    n_relations = len(RELATIONS)
    n_heads = vectors.shape[1]
    head_dim = vectors.shape[2]
    sums = np.zeros((n_relations, n_heads, head_dim), dtype=np.float64)
    counts = np.zeros(n_relations, dtype=np.int64)
    for relation_index, relation in enumerate(RELATIONS):
        mask = relations[train_indices] == relation
        if correct_only:
            mask = mask & (errors[train_indices] == 0)
        members = train_indices[mask]
        if members.size == 0:
            raise RuntimeError(
                f"Training fold has no prototype members for relation={relation}; "
                f"correct_only={correct_only}"
            )
        sums[relation_index] = np.asarray(vectors[members], dtype=np.float64).sum(axis=0)
        counts[relation_index] = int(members.size)
    return sums, counts


def score_all_prototypes(
    *,
    vectors: np.ndarray,  # [N,H,D], unit
    target_indices: np.ndarray,
    prototype_sums: np.ndarray,  # [R,H,D]
    prototype_counts: np.ndarray,  # [R]
    loo_relation_indices: Optional[np.ndarray] = None,
    loo_member_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return [T,H,R] similarities to ALL relation prototypes.

    Test calls pass no LOO labels or masks.  Therefore test GT cannot affect
    these features.  Training calls may subtract a correct sample from its own
    relation prototype to avoid self-inclusion.
    """
    prototypes = row_unit(
        prototype_sums / np.maximum(prototype_counts[:, None, None], 1)
    )
    current = np.asarray(vectors[target_indices], dtype=np.float64)
    scores = np.einsum("thd,rhd->thr", current, prototypes, optimize=True)

    if loo_relation_indices is None or loo_member_mask is None:
        return scores.astype(np.float32)
    if len(loo_relation_indices) != len(target_indices) or len(loo_member_mask) != len(target_indices):
        raise ValueError("LOO metadata length mismatch")

    for relation_index in range(len(RELATIONS)):
        local = np.where(
            np.asarray(loo_member_mask, dtype=bool)
            & (np.asarray(loo_relation_indices, dtype=np.int64) == relation_index)
        )[0]
        if local.size == 0:
            continue
        count = int(prototype_counts[relation_index])
        if count <= 1:
            continue
        sample_indices = target_indices[local]
        loo_sum = prototype_sums[relation_index][None, :, :] - np.asarray(
            vectors[sample_indices], dtype=np.float64
        )
        loo_proto = row_unit(loo_sum / float(count - 1))
        loo_score = np.sum(
            np.asarray(vectors[sample_indices], dtype=np.float64) * loo_proto,
            axis=-1,
        )
        scores[local, :, relation_index] = loo_score
    return scores.astype(np.float32)


def build_fold_scores(
    *,
    vectors_by_state: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    relations: np.ndarray,
    relation_indices: np.ndarray,
    errors: np.ndarray,
    target_is_training: bool,
    correct_only: bool,
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    train_set = set(map(int, train_indices.tolist()))
    if target_is_training:
        member_mask = np.asarray(
            [
                (int(sample_index) in train_set)
                and ((not correct_only) or errors[int(sample_index)] == 0)
                for sample_index in target_indices
            ],
            dtype=bool,
        )
        loo_relations: Optional[np.ndarray] = relation_indices[target_indices]
    else:
        # Explicitly no target relation is passed to score_all_prototypes.
        member_mask = None
        loo_relations = None

    for state in STATES:
        sums, counts = build_prototype_sums(
            vectors=vectors_by_state[state],
            train_indices=train_indices,
            relations=relations,
            errors=errors,
            correct_only=correct_only,
        )
        output[state] = score_all_prototypes(
            vectors=vectors_by_state[state],
            target_indices=target_indices,
            prototype_sums=sums,
            prototype_counts=counts,
            loo_relation_indices=loo_relations,
            loo_member_mask=member_mask,
        )
    return output


# -----------------------------------------------------------------------------
# No-test-GT feature construction
# -----------------------------------------------------------------------------


def flatten_head_metrics(
    values: np.ndarray, head_names: Sequence[str], metric_names: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    n_samples, n_heads, n_metrics = values.shape
    if n_heads != len(head_names) or n_metrics != len(metric_names):
        raise ValueError("Head metric shape/name mismatch")
    names = [f"{head}::{metric}" for head in head_names for metric in metric_names]
    return values.reshape(n_samples, n_heads * n_metrics).astype(np.float64), names


def build_head_features(
    *,
    scores_by_state: Mapping[str, np.ndarray],
    model_prediction_indices: np.ndarray,
    head_names: Sequence[str],
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Return scores-only and full per-head no-test-GT features."""
    raw_arrays: List[np.ndarray] = []
    raw_names: List[str] = []
    derived_arrays: List[np.ndarray] = []
    derived_names: List[str] = []

    for state in STATES:
        scores = np.asarray(scores_by_state[state], dtype=np.float64)
        for relation_index, relation in enumerate(RELATIONS):
            raw_arrays.append(scores[:, :, relation_index])
            raw_names.append(f"proto_{state}_cos_{relation}")

        probability = softmax_last(scores)
        argmax_relation = np.argmax(scores, axis=-1)
        predicted_score = gather_relation_score(scores, model_prediction_indices)
        predicted_margin = relation_margin_against_best_other(scores, model_prediction_indices)
        predicted_is_top = (
            argmax_relation == model_prediction_indices[:, None]
        ).astype(np.float64)
        state_map = {
            f"proto_{state}_top_score": np.max(scores, axis=-1),
            f"proto_{state}_top_margin": top_margin(scores),
            f"proto_{state}_entropy": entropy_last(probability),
            f"proto_{state}_model_pred_score": predicted_score,
            f"proto_{state}_model_pred_margin": predicted_margin,
            f"proto_{state}_model_pred_is_top": predicted_is_top,
        }
        for name, values in state_map.items():
            derived_names.append(name)
            derived_arrays.append(values)

    original = np.asarray(scores_by_state["orig"], dtype=np.float64)
    swapped = np.asarray(scores_by_state["swap"], dtype=np.float64)
    stable = np.asarray(scores_by_state["stable"], dtype=np.float64)
    cross_map = {
        "proto_orig_swap_score_cos": score_vector_cosine(original, swapped),
        "proto_orig_swap_score_l1": np.mean(np.abs(original - swapped), axis=-1),
        "proto_orig_swap_top_agree": (
            np.argmax(original, axis=-1) == np.argmax(swapped, axis=-1)
        ).astype(np.float64),
        "proto_stable_orig_score_cos": score_vector_cosine(stable, original),
        "proto_stable_swap_score_cos": score_vector_cosine(stable, swapped),
    }
    for name, values in cross_map.items():
        derived_names.append(name)
        derived_arrays.append(values)

    raw_values = np.stack(raw_arrays, axis=-1)
    derived_values = np.stack(derived_arrays, axis=-1)
    full_values = np.concatenate([raw_values, derived_values], axis=-1)
    full_names = raw_names + derived_names
    raw_flat, raw_flat_names = flatten_head_metrics(raw_values, head_names, raw_names)
    full_flat, full_flat_names = flatten_head_metrics(full_values, head_names, full_names)
    return raw_flat, raw_flat_names, full_flat, full_flat_names


def aggregate_block_features(
    *,
    prefix: str,
    scores: np.ndarray,  # [N,H,R]
    model_prediction_indices: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    mean_scores = np.mean(scores, axis=1)
    probability = softmax_last(mean_scores)
    internal_prediction = np.argmax(mean_scores, axis=-1)
    sample_index = np.arange(len(mean_scores))
    predicted_score = mean_scores[sample_index, model_prediction_indices]
    masked = np.array(mean_scores, copy=True)
    masked[sample_index, model_prediction_indices] = -np.inf
    predicted_margin = predicted_score - np.max(masked, axis=-1)

    votes = np.argmax(scores, axis=-1)
    vote_fraction = np.stack(
        [np.mean(votes == relation_index, axis=1) for relation_index in range(len(RELATIONS))],
        axis=-1,
    )
    vote_entropy = entropy_last(np.maximum(vote_fraction, EPS))
    model_vote_fraction = vote_fraction[sample_index, model_prediction_indices]

    arrays: List[np.ndarray] = []
    names: List[str] = []
    for relation_index, relation in enumerate(RELATIONS):
        arrays.append(mean_scores[:, relation_index])
        names.append(f"{prefix}::mean_cos_{relation}")
    arrays.extend(
        [
            np.max(mean_scores, axis=-1),
            top_margin(mean_scores),
            entropy_last(probability),
            predicted_score,
            predicted_margin,
            (internal_prediction == model_prediction_indices).astype(np.float64),
        ]
    )
    names.extend(
        [
            f"{prefix}::top_score",
            f"{prefix}::top_margin",
            f"{prefix}::entropy",
            f"{prefix}::model_pred_score",
            f"{prefix}::model_pred_margin",
            f"{prefix}::internal_agrees_model",
        ]
    )
    for relation_index, relation in enumerate(RELATIONS):
        arrays.append(vote_fraction[:, relation_index])
        names.append(f"{prefix}::vote_fraction_{relation}")
    arrays.extend([vote_entropy, model_vote_fraction])
    names.extend([f"{prefix}::vote_entropy", f"{prefix}::model_vote_fraction"])
    return np.stack(arrays, axis=-1), names


def build_global_features(
    *,
    scores_by_state: Mapping[str, np.ndarray],
    model_prediction_indices: np.ndarray,
    head_layers: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    for state in STATES:
        values, names = aggregate_block_features(
            prefix=f"global::{state}",
            scores=np.asarray(scores_by_state[state], dtype=np.float64),
            model_prediction_indices=model_prediction_indices,
        )
        feature_blocks.append(values)
        feature_names.extend(names)

    stable = np.asarray(scores_by_state["stable"], dtype=np.float64)
    for layer in sorted(set(map(int, head_layers.tolist()))):
        mask = head_layers == layer
        values, names = aggregate_block_features(
            prefix=f"layer_L{layer}::stable",
            scores=stable[:, mask, :],
            model_prediction_indices=model_prediction_indices,
        )
        feature_blocks.append(values)
        feature_names.extend(names)

    original_mean = np.mean(np.asarray(scores_by_state["orig"], dtype=np.float64), axis=1)
    swapped_mean = np.mean(np.asarray(scores_by_state["swap"], dtype=np.float64), axis=1)
    stable_mean = np.mean(stable, axis=1)
    cross = np.stack(
        [
            score_vector_cosine(original_mean, swapped_mean),
            np.mean(np.abs(original_mean - swapped_mean), axis=-1),
            (np.argmax(original_mean, axis=-1) == np.argmax(swapped_mean, axis=-1)).astype(np.float64),
            score_vector_cosine(stable_mean, original_mean),
            score_vector_cosine(stable_mean, swapped_mean),
        ],
        axis=-1,
    )
    cross_names = [
        "global::orig_swap_score_cos",
        "global::orig_swap_score_l1",
        "global::orig_swap_top_agree",
        "global::stable_orig_score_cos",
        "global::stable_swap_score_cos",
    ]
    feature_blocks.append(cross)
    feature_names.extend(cross_names)

    internal_prediction = np.argmax(stable_mean, axis=-1)
    internal_margin = top_margin(stable_mean)
    return (
        np.concatenate(feature_blocks, axis=1),
        feature_names,
        internal_prediction.astype(np.int64),
        internal_margin.astype(np.float64),
    )


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------


def choose_group(
    *,
    group: str,
    confidence_train: np.ndarray,
    confidence_test: np.ndarray,
    confidence_names: Sequence[str],
    global_train: np.ndarray,
    global_test: np.ndarray,
    global_names: Sequence[str],
    score_train: np.ndarray,
    score_test: np.ndarray,
    score_names: Sequence[str],
    head_train: np.ndarray,
    head_test: np.ndarray,
    head_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if group == "confidence":
        return confidence_train, confidence_test, list(confidence_names)
    if group == "prototype_global":
        return global_train, global_test, list(global_names)
    if group == "confidence_plus_prototype_global":
        return (
            np.concatenate([confidence_train, global_train], axis=1),
            np.concatenate([confidence_test, global_test], axis=1),
            list(confidence_names) + list(global_names),
        )
    if group == "prototype_head_scores":
        return score_train, score_test, list(score_names)
    if group == "prototype_head":
        return head_train, head_test, list(head_names)
    if group == "confidence_plus_prototype_head":
        return (
            np.concatenate([confidence_train, head_train], axis=1),
            np.concatenate([confidence_test, head_test], axis=1),
            list(confidence_names) + list(head_names),
        )
    raise ValueError(group)


def fit_predict_fold(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(x_train)
    test_imputed = imputer.transform(x_test)
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    n_features = train_scaled.shape[1]
    k = n_features if int(args.top_k_features) <= 0 else min(int(args.top_k_features), n_features)
    selected_indices = np.arange(n_features)
    if k < n_features:
        selector = SelectKBest(score_func=f_classif, k=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            train_scaled = selector.fit_transform(train_scaled, y_train)
        test_scaled = selector.transform(test_scaled)
        selected_indices = selector.get_support(indices=True)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=float(args.logreg_c),
        l1_ratio=float(args.l1_ratio),
        class_weight="balanced",
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train_scaled, y_train)
    probability = model.predict_proba(test_scaled)[:, 1]
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    selected_rows: List[Dict[str, Any]] = []
    for local_index, original_index in enumerate(selected_indices.tolist()):
        coefficient = float(coefficients[local_index])
        selected_rows.append(
            {
                "feature": str(feature_names[original_index]),
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
                "nonzero": bool(abs(coefficient) > 1e-10),
            }
        )
    return probability, selected_rows


def best_f1_threshold(y: np.ndarray, probability: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if thresholds.size == 0:
        return 0.5, float(f1_score(y, probability >= 0.5, zero_division=0))
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], EPS
    )
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), float(f1[index])


def metric_row(
    *, model_name: str, y: np.ndarray, probability: np.ndarray, review_fraction: float
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    threshold, best_f1 = best_f1_threshold(y, probability)
    review_count = max(1, int(math.ceil(len(y) * review_fraction)))
    order = np.argsort(-probability)
    reviewed = order[:review_count]
    retained = order[review_count:]
    total_errors = max(int(y.sum()), 1)
    return {
        "model": model_name,
        "N": len(y),
        "errors": int(y.sum()),
        "error_prevalence": float(y.mean()),
        "AUROC": float(roc_auc_score(y, probability)),
        "AUPRC": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)),
        "F1_at_0.5": float(f1_score(y, probability >= 0.5, zero_division=0)),
        "best_F1_exploratory": best_f1,
        "best_F1_threshold_exploratory": threshold,
        "review_fraction": review_fraction,
        "review_count": review_count,
        "error_recall_at_review_fraction": float(y[reviewed].sum() / total_errors),
        "review_precision": float(y[reviewed].mean()),
        "retained_accuracy": float(1.0 - y[retained].mean()) if retained.size else float("nan"),
    }


def parse_feature_head(feature: str) -> Tuple[Optional[str], str]:
    parts = str(feature).split("::")
    if parts and re.fullmatch(r"L\d+H\d+", parts[0]):
        return parts[0], "::".join(parts[1:])
    return None, str(feature)


def run_detector(args: argparse.Namespace, data: Mapping[str, Any], output_dir: Path) -> None:
    groups = [item.strip() for item in str(args.model_groups).split(",") if item.strip()]
    invalid = sorted(set(groups) - set(MODEL_GROUPS))
    if invalid:
        raise ValueError(f"Unknown model groups: {invalid}")

    sids = np.asarray(data["sids"], dtype=np.int64)
    relations = np.asarray(data["relations"], dtype=object)
    relation_indices = np.asarray(data["relation_indices"], dtype=np.int64)
    errors = np.asarray(data["errors"], dtype=np.int64)
    model_prediction_indices = np.asarray(data["model_prediction_indices"], dtype=np.int64)
    model_predictions = np.asarray(data["model_predictions"], dtype=object)
    vectors_by_state = data["vectors"]
    head_names = list(data["head_names"])
    head_layers = np.asarray(data["head_layers"], dtype=np.int64)
    rows = list(data["rows"])
    confidence, confidence_names = confidence_features(rows)

    if args.stratify == "relation_error":
        strata = np.asarray(
            [f"{relations[i]}__{int(errors[i])}" for i in range(len(errors))], dtype=object
        )
    else:
        strata = errors.astype(str)
    counts = Counter(map(str, strata.tolist()))
    too_small = {key: value for key, value in counts.items() if value < int(args.outer_folds)}
    if too_small:
        raise RuntimeError(
            f"Strata need at least --outer-folds samples: {too_small}. Reduce folds."
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=int(args.outer_folds),
        n_repeats=int(args.outer_repeats),
        random_state=int(args.seed),
    )
    probability_sum = {group: np.zeros(len(errors), dtype=np.float64) for group in groups}
    probability_count = {group: np.zeros(len(errors), dtype=np.int64) for group in groups}
    repeat_probability = {
        group: np.full((int(args.outer_repeats), len(errors)), np.nan, dtype=np.float64)
        for group in groups
    }
    internal_score_sum = np.zeros((len(errors), len(RELATIONS)), dtype=np.float64)
    internal_score_count = np.zeros(len(errors), dtype=np.int64)
    internal_margin_sum = np.zeros(len(errors), dtype=np.float64)
    selection_rows: List[Dict[str, Any]] = []
    repeat_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    correct_only = args.prototype_training == "correct_only"

    for split_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(errors)), strata)
    ):
        repeat_index = split_index // int(args.outer_folds)
        fold_index = split_index % int(args.outer_folds)

        train_scores = build_fold_scores(
            vectors_by_state=vectors_by_state,
            train_indices=train_indices,
            target_indices=train_indices,
            relations=relations,
            relation_indices=relation_indices,
            errors=errors,
            target_is_training=True,
            correct_only=correct_only,
        )
        test_scores = build_fold_scores(
            vectors_by_state=vectors_by_state,
            train_indices=train_indices,
            target_indices=test_indices,
            relations=relations,
            relation_indices=relation_indices,
            errors=errors,
            target_is_training=False,
            correct_only=correct_only,
        )

        score_train, score_names, head_train, head_feature_names = build_head_features(
            scores_by_state=train_scores,
            model_prediction_indices=model_prediction_indices[train_indices],
            head_names=head_names,
        )
        score_test, score_test_names, head_test, head_test_names = build_head_features(
            scores_by_state=test_scores,
            model_prediction_indices=model_prediction_indices[test_indices],
            head_names=head_names,
        )
        if score_names != score_test_names or head_feature_names != head_test_names:
            raise RuntimeError("Train/test head feature names differ")

        global_train, global_names, _, _ = build_global_features(
            scores_by_state=train_scores,
            model_prediction_indices=model_prediction_indices[train_indices],
            head_layers=head_layers,
        )
        global_test, global_test_names, internal_prediction, internal_margin = build_global_features(
            scores_by_state=test_scores,
            model_prediction_indices=model_prediction_indices[test_indices],
            head_layers=head_layers,
        )
        if global_names != global_test_names:
            raise RuntimeError("Train/test global feature names differ")

        stable_mean = np.mean(np.asarray(test_scores["stable"], dtype=np.float64), axis=1)
        internal_score_sum[test_indices] += stable_mean
        internal_score_count[test_indices] += 1
        internal_margin_sum[test_indices] += internal_margin

        if args.write_long_prototype_scores:
            for local_index, sample_index in enumerate(test_indices.tolist()):
                for head_index, head in enumerate(head_names):
                    row: Dict[str, Any] = {
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "sid": int(sids[sample_index]),
                        "head": head,
                    }
                    for state in STATES:
                        for relation_index, relation in enumerate(RELATIONS):
                            row[f"{state}_cos_{relation}"] = float(
                                test_scores[state][local_index, head_index, relation_index]
                            )
                    long_rows.append(row)

        for group in groups:
            x_train, x_test, names = choose_group(
                group=group,
                confidence_train=confidence[train_indices],
                confidence_test=confidence[test_indices],
                confidence_names=confidence_names,
                global_train=global_train,
                global_test=global_test,
                global_names=global_names,
                score_train=score_train,
                score_test=score_test,
                score_names=score_names,
                head_train=head_train,
                head_test=head_test,
                head_names=head_feature_names,
            )
            probability, selected = fit_predict_fold(
                x_train=x_train,
                y_train=errors[train_indices],
                x_test=x_test,
                feature_names=names,
                args=args,
            )
            probability_sum[group][test_indices] += probability
            probability_count[group][test_indices] += 1
            repeat_probability[group][repeat_index, test_indices] = probability
            for item in selected:
                head, metric = parse_feature_head(str(item["feature"]))
                selection_rows.append(
                    {
                        "model": group,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "feature": item["feature"],
                        "head": head,
                        "metric": metric,
                        "coefficient": item["coefficient"],
                        "abs_coefficient": item["abs_coefficient"],
                        "nonzero": item["nonzero"],
                    }
                )

        print(
            f"[repeat {repeat_index + 1}/{args.outer_repeats} "
            f"fold {fold_index + 1}/{args.outer_folds}] "
            f"train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )

    performance_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    averaged_probability: Dict[str, np.ndarray] = {}
    for group in groups:
        if not np.all(probability_count[group] == int(args.outer_repeats)):
            raise RuntimeError(f"OOF count mismatch for {group}")
        probability = probability_sum[group] / np.maximum(probability_count[group], 1)
        averaged_probability[group] = probability
        performance_rows.append(
            metric_row(
                model_name=group,
                y=errors,
                probability=probability,
                review_fraction=float(args.review_fraction),
            )
        )
        for repeat_index in range(int(args.outer_repeats)):
            values = repeat_probability[group][repeat_index]
            if np.isnan(values).any():
                raise RuntimeError(f"Missing repeated OOF values for {group}, repeat={repeat_index}")
            repeat_rows.append(
                {
                    "repeat": repeat_index,
                    **metric_row(
                        model_name=group,
                        y=errors,
                        probability=values,
                        review_fraction=float(args.review_fraction),
                    ),
                }
            )
        for relation in RELATIONS:
            mask = relations == relation
            if mask.sum() < 2 or len(np.unique(errors[mask])) < 2:
                continue
            relation_rows.append(
                {
                    "relation": relation,
                    **metric_row(
                        model_name=group,
                        y=errors[mask],
                        probability=probability[mask],
                        review_fraction=float(args.review_fraction),
                    ),
                }
            )

    average_internal_scores = internal_score_sum / np.maximum(
        internal_score_count[:, None], 1
    )
    internal_prediction_indices = np.argmax(average_internal_scores, axis=-1)
    internal_predictions = np.asarray(
        [RELATIONS[index] for index in internal_prediction_indices], dtype=object
    )
    average_internal_margin = internal_margin_sum / np.maximum(internal_score_count, 1)
    internal_rows: List[Dict[str, Any]] = []
    for index, sid in enumerate(sids.tolist()):
        row = {
            "sid": int(sid),
            "gt": str(relations[index]),
            "model_prediction": str(model_predictions[index]),
            "baseline_correct": bool(errors[index] == 0),
            "internal_prediction_no_test_gt": str(internal_predictions[index]),
            "internal_prediction_correct": bool(internal_predictions[index] == relations[index]),
            "internal_agrees_model": bool(internal_predictions[index] == model_predictions[index]),
            "internal_margin": float(average_internal_margin[index]),
        }
        for relation_index, relation in enumerate(RELATIONS):
            row[f"internal_score_{relation}"] = float(
                average_internal_scores[index, relation_index]
            )
        internal_rows.append(row)

        oof = {
            "sid": int(sid),
            "gt": str(relations[index]),
            "model_prediction": str(model_predictions[index]),
            "baseline_correct": bool(errors[index] == 0),
            "error_label": int(errors[index]),
            "internal_prediction_no_test_gt": str(internal_predictions[index]),
            "internal_agrees_model": bool(internal_predictions[index] == model_predictions[index]),
        }
        for group in groups:
            oof[f"error_probability__{group}"] = float(averaged_probability[group][index])
        oof_rows.append(oof)

    write_csv(output_dir / "model_performance.csv", performance_rows)
    write_csv(output_dir / "repeat_performance.csv", repeat_rows)
    write_csv(output_dir / "relation_performance.csv", relation_rows)
    write_csv(output_dir / "oof_predictions.csv", oof_rows)
    write_csv(output_dir / "internal_relation_oof.csv", internal_rows)
    write_csv(output_dir / "selected_features_by_fold.csv", selection_rows)
    if long_rows:
        write_csv(output_dir / "prototype_scores_long.csv", long_rows)

    total_folds = int(args.outer_folds) * int(args.outer_repeats)
    grouped_features: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in selection_rows:
        if bool_value(row["nonzero"]):
            grouped_features[(str(row["model"]), str(row["feature"]))].append(row)
    stability_rows: List[Dict[str, Any]] = []
    for (model_name, feature), items in grouped_features.items():
        head, metric = parse_feature_head(feature)
        coefficients = np.asarray([float(item["coefficient"]) for item in items], dtype=np.float64)
        stability_rows.append(
            {
                "model": model_name,
                "feature": feature,
                "head": head,
                "metric": metric,
                "selected_nonzero_folds": len(items),
                "total_folds": total_folds,
                "nonzero_fold_fraction": len(items) / float(total_folds),
                "mean_coefficient_when_selected": float(coefficients.mean()),
                "mean_abs_coefficient_when_selected": float(np.abs(coefficients).mean()),
            }
        )
    stability_rows.sort(
        key=lambda row: (
            str(row["model"]),
            -float(row["nonzero_fold_fraction"]),
            -float(row["mean_abs_coefficient_when_selected"]),
        )
    )
    write_csv(output_dir / "selected_feature_stability.csv", stability_rows)

    head_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        if row.get("head"):
            head_groups[(str(row["model"]), str(row["head"]))].append(row)
    head_rows: List[Dict[str, Any]] = []
    for (model_name, head), items in head_groups.items():
        best = max(
            items,
            key=lambda row: (
                float(row["nonzero_fold_fraction"]),
                float(row["mean_abs_coefficient_when_selected"]),
            ),
        )
        head_rows.append(
            {
                "model": model_name,
                "head": head,
                "max_nonzero_fold_fraction": max(
                    float(row["nonzero_fold_fraction"]) for row in items
                ),
                "mean_nonzero_fold_fraction": safe_mean(
                    float(row["nonzero_fold_fraction"]) for row in items
                ),
                "max_mean_abs_coefficient": max(
                    float(row["mean_abs_coefficient_when_selected"]) for row in items
                ),
                "dominant_feature": best["feature"],
                "dominant_metric": best["metric"],
                "dominant_mean_coefficient": best["mean_coefficient_when_selected"],
            }
        )
    head_rows.sort(
        key=lambda row: (
            str(row["model"]),
            -float(row["max_nonzero_fold_fraction"]),
            -float(row["max_mean_abs_coefficient"]),
        )
    )
    write_csv(output_dir / "head_importance_summary.csv", head_rows)

    internal_accuracy = float(accuracy_score(relations, internal_predictions))
    internal_model_agreement = float(np.mean(internal_predictions == model_predictions))
    internal_accuracy_on_wrong_generation = float(
        np.mean(internal_predictions[errors == 1] == relations[errors == 1])
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "input_dir": str(args.input_dir),
        "samples": len(sids),
        "correct": int((errors == 0).sum()),
        "wrong": int(errors.sum()),
        "heads": len(head_names),
        "relations": list(RELATIONS),
        "prototype_training": args.prototype_training,
        "model_prediction_source": args.model_prediction_source,
        "test_feature_uses_gt": False,
        "training_prototype_uses_gt": True,
        "training_prototype_uses_correctness": bool(correct_only),
        "model_performance": performance_rows,
        "internal_relation_prediction": {
            "accuracy": internal_accuracy,
            "agreement_with_model_output": internal_model_agreement,
            "accuracy_among_generation_wrong": internal_accuracy_on_wrong_generation,
        },
        "interpretation_limits": [
            "Test activations are compared with all four relation prototypes; no test GT selects a prototype.",
            "GT may be used for relation-balanced fold assignment and final evaluation only.",
            "Correct-only prototype construction is supervised normative modeling.",
            "Predictive head features do not establish that selected heads cause the errors.",
        ],
    }
    write_json(output_dir / "summary.json", summary)

    print("\n" + "=" * 168)
    print("COCO ALL-RELATION PER-HEAD PROTOTYPE ERROR DETECTOR — NO TEST GT FEATURES")
    print("=" * 168)
    print(
        f"Samples={len(sids)} | correct={(errors == 0).sum()} | wrong={errors.sum()} | "
        f"heads={len(head_names)} | repeats={args.outer_repeats} folds={args.outer_folds}"
    )
    print(
        f"Prototype training={args.prototype_training} | "
        f"model prediction source={args.model_prediction_source} | test_feature_uses_gt=False"
    )
    print("\nOOF PERFORMANCE")
    for row in sorted(performance_rows, key=lambda item: float(item["AUPRC"]), reverse=True):
        print(
            f"{str(row['model']):44s} AUROC={float(row['AUROC']):.4f} "
            f"AUPRC={float(row['AUPRC']):.4f} Brier={float(row['Brier']):.4f} "
            f"top{int(round(100 * float(args.review_fraction)))}%-error-recall="
            f"{float(row['error_recall_at_review_fraction']):.4f}"
        )
    print("\nINTERNAL RELATION PREDICTION FROM STABLE PROTOTYPES")
    print(
        f"accuracy={internal_accuracy:.4f} | agrees_model={internal_model_agreement:.4f} | "
        f"accuracy_among_generation_wrong={internal_accuracy_on_wrong_generation:.4f}"
    )
    for group in ("prototype_head", "confidence_plus_prototype_head"):
        relevant = [row for row in head_rows if row["model"] == group][:10]
        if not relevant:
            continue
        print(f"\nTOP STABLE HEADS: {group}")
        for row in relevant:
            print(
                f"{str(row['head']):8s} stable={float(row['max_nonzero_fold_fraction']):.3f} "
                f"|coef|={float(row['max_mean_abs_coefficient']):.4f} "
                f"metric={row['dominant_metric']}"
            )
    print(f"\nSaved outputs to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if int(args.outer_folds) < 2 or int(args.outer_repeats) < 1:
        raise ValueError("--outer-folds must be >=2 and --outer-repeats >=1")
    if not 0.0 < float(args.review_fraction) < 1.0:
        raise ValueError("--review-fraction must be in (0,1)")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(input_dir, args.model_prediction_source)
    write_json(
        output_dir / "config.json",
        {
            "script_version": SCRIPT_VERSION,
            "arguments": vars(args),
            "source_config": data["config"],
            "samples": len(data["sids"]),
            "heads": len(data["head_names"]),
            "test_feature_uses_gt": False,
        },
    )
    run_detector(args, data, output_dir)


if __name__ == "__main__":
    main()
