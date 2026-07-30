#!/usr/bin/env python3
"""
Offline circuit-based error detector for the COCO two-object / Qwen2.5-VL
spatial-relation experiment.

Purpose
-------
The validated P_POS7 -> L26VH0 path is a strong relation-information transport
path, but direct amplification / deletion / sign-flip gives only modest free-
generation accuracy gains.  This script tests a different use:

    Can internal channel signals identify which normal free-generation answers
    are likely to be wrong?

The script does NOT modify or load the VLM.  It reuses outputs already produced
by:

  1. output/spatial_storage_transport_utilization/coco/qwen-3b/extraction.jsonl
  2. qwen-3b_head_misrouting_pos7_neg5/baseline_generation.jsonl
  3. qwen-3b_head_misrouting_pos7_neg5/head_path_ablation_generation.jsonl

All detector features are constructed without using GT.  GT / correctness is
used only as the supervised error label during cross-validation.

Feature families
----------------
confidence
    Original and swapped prompt-last four-relation score margins, entropy,
    relation-token confidence, and generation-vs-closed prediction agreement.

pair
    Original/swapped inverse-relation consistency.  Swapped scores are aligned
    to the original relation axis by left<->right and above<->below.

circuit_aggregate
    P_POS7 and P_NEG5 aggregate contribution vectors, head-vote disagreement,
    positive/negative competition, receiver-delta magnitudes.

circuit_static
    Aggregate circuit features plus per-head path contribution features.

circuit_active
    Static circuit features plus whether each path-specific head ablation
    changes the complete free-generation prediction.  This is a more expensive
    diagnostic detector because it assumes the 12 ablation generations exist.

Evaluation
----------
Repeated stratified out-of-fold logistic regression.  Main metrics:
AUROC, AUPRC, F1, Brier score, error recall at fixed review fractions, and
selective accuracy after rejecting the highest-risk samples.

Important leakage rule
----------------------
The following existing fields are deliberately NEVER used as detector inputs:
GT, correct/fixed/broken, E_GT, wrong_over_gt_contribution,
misleading_for_generation_error, strict_misleading_for_generation_error, and
generation_pair_status.  They may appear only as labels or descriptive metadata.

Outputs
-------
  error_detector_features.csv
  detector_oof_predictions.csv
  detector_summary.csv
  detector_fold_metrics.csv
  detector_risk_coverage.csv
  detector_feature_coefficients.csv
  detector_univariate_features.csv
  detector_group_risk_summary.csv
  summary.json
  config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn and pandas. "
        f"Import failed: {type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "coco-circuit-error-detector-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
STATUS_ALIASES = {
    "both_correct": "CC",
    "original_only": "CW",
    "swapped_only": "WC",
    "both_wrong": "WW",
}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-output-dir",
        required=True,
        help="Directory containing config.json and extraction.jsonl.",
    )
    p.add_argument(
        "--head-output-dir",
        required=True,
        help=(
            "Directory produced by analyze_coco_head_misrouting_generation_v1.py "
            "and containing baseline_generation.jsonl plus "
            "head_path_ablation_generation.jsonl."
        ),
    )
    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument("--positive-bundle", default="P_POS7")
    p.add_argument("--negative-bundle", default="P_NEG5")
    p.add_argument("--model", default="qwen-3b")

    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=10)
    p.add_argument("--logistic-c", type=float, default=1.0)
    p.add_argument("--bootstrap-repeats", type=int, default=3000)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument(
        "--review-fractions",
        default="0.05,0.10,0.20,0.30,0.40",
        help="Fractions of highest-risk answers sent for review/rejection.",
    )
    p.add_argument(
        "--feature-sets",
        default=(
            "confidence,pair,circuit_aggregate,circuit_static,"
            "circuit_active,confidence_pair,confidence_circuit,all"
        ),
    )
    p.add_argument(
        "--sample-max-samples",
        type=int,
        default=0,
        help="0 means all common SIDs. Nonzero is only for a quick smoke test.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(dict(payload))
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
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_csv_tokens(text: str) -> List[str]:
    result: List[str] = []
    for item in str(text).split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result


def parse_float_list(text: str) -> List[float]:
    result: List[float] = []
    for item in parse_csv_tokens(text):
        value = float(item)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Review fraction must be in [0,1): {value}")
        result.append(value)
    return sorted(set(result))


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "left_of": "left",
        "right_of": "right",
        "on": "above",
        "under": "below",
    }
    if text in RELATIONS:
        return text
    return aliases.get(text)


def relation_scores(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, Mapping):
        return None
    result: Dict[str, float] = {}
    for key, item in value.items():
        relation = normalize_relation(key)
        if relation is None:
            continue
        try:
            result[relation] = float(item)
        except (TypeError, ValueError):
            continue
    if not all(relation in result for relation in RELATIONS):
        return None
    return {relation: float(result[relation]) for relation in RELATIONS}


def score_vector(scores: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(scores[relation]) for relation in RELATIONS], dtype=np.float64)


def stable_softmax(vector: Sequence[float]) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64)
    x = x - np.nanmax(x)
    exp = np.exp(np.clip(x, -80.0, 80.0))
    denominator = float(np.sum(exp))
    if not math.isfinite(denominator) or denominator <= 0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / denominator


def entropy_from_probabilities(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def top_relation(vector: Sequence[float]) -> str:
    return RELATIONS[int(np.nanargmax(np.asarray(vector, dtype=np.float64)))]


def top_margin(vector: Sequence[float]) -> float:
    x = np.sort(np.asarray(vector, dtype=np.float64))
    return float(x[-1] - x[-2])


def target_margin(vector: Sequence[float], target: Optional[str]) -> float:
    target = normalize_relation(target)
    if target is None:
        return float("nan")
    x = np.asarray(vector, dtype=np.float64)
    index = RELATIONS.index(target)
    others = np.delete(x, index)
    return float(x[index] - np.max(others))


def target_rank(vector: Sequence[float], target: Optional[str]) -> float:
    target = normalize_relation(target)
    if target is None:
        return float("nan")
    x = np.asarray(vector, dtype=np.float64)
    order = np.argsort(-x, kind="mergesort")
    return float(np.where(order == RELATIONS.index(target))[0][0] + 1)


def centered_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denominator)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(float(denominator)), 1e-12))


def sanitize_name(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text)


def parse_head_text(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("L") and "H" in text:
        layer, head = text[1:].split("H", 1)
        return f"L{int(layer)}H{int(head)}"
    if ":" in text:
        layer, head = text.split(":", 1)
        return f"L{int(layer)}H{int(head)}"
    raise ValueError(f"Invalid head specification: {value!r}")


def load_bundles(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if not isinstance(source, Mapping):
        raise ValueError("Bundle JSON must contain an object")
    result: Dict[str, List[str]] = {}
    for name, values in source.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Bundle {name} must be a list")
        parsed = [parse_head_text(value) for value in values]
        if len(parsed) != len(set(parsed)):
            raise ValueError(f"Duplicate head in bundle {name}")
        result[str(name)] = parsed
    return result


def source_original_correct(row: Mapping[str, Any]) -> Optional[bool]:
    status = str(row.get("generation_pair_status", ""))
    if status in {"both_correct", "original_only"}:
        return True
    if status in {"swapped_only", "both_wrong"}:
        return False
    for key in (
        "original_generation_correct",
        "generation_original_correct",
        "original_correct",
    ):
        if key in row:
            return bool(row[key])
    return None


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------


def add_vector_features(
    features: Dict[str, float],
    prefix: str,
    vector: Sequence[float],
    *,
    generation_prediction: Optional[str] = None,
    closed_prediction: Optional[str] = None,
) -> None:
    x = np.asarray(vector, dtype=np.float64)
    probabilities = stable_softmax(x)
    centered = x - np.mean(x)
    preferred = top_relation(x)
    features[f"{prefix}top_margin"] = top_margin(x)
    features[f"{prefix}entropy"] = entropy_from_probabilities(probabilities)
    features[f"{prefix}max_probability"] = float(np.max(probabilities))
    features[f"{prefix}range"] = float(np.max(x) - np.min(x))
    features[f"{prefix}centered_l2"] = float(np.linalg.norm(centered))
    features[f"{prefix}abs_mean"] = float(np.mean(np.abs(x)))
    if generation_prediction is not None:
        features[f"{prefix}generation_margin"] = target_margin(
            x, generation_prediction
        )
        features[f"{prefix}generation_rank"] = target_rank(
            x, generation_prediction
        )
        features[f"{prefix}agrees_generation"] = float(
            preferred == generation_prediction
        )
    if closed_prediction is not None:
        features[f"{prefix}closed_margin"] = target_margin(x, closed_prediction)
        features[f"{prefix}agrees_closed"] = float(preferred == closed_prediction)


def head_vote_features(
    features: Dict[str, float],
    prefix: str,
    vectors: Sequence[np.ndarray],
    *,
    generation_prediction: Optional[str],
    closed_prediction: Optional[str],
) -> None:
    if not vectors:
        return
    votes = [top_relation(vector) for vector in vectors]
    counts = Counter(votes)
    probabilities = np.asarray(
        [counts.get(relation, 0) / len(votes) for relation in RELATIONS],
        dtype=np.float64,
    )
    features[f"{prefix}vote_entropy"] = entropy_from_probabilities(probabilities)
    features[f"{prefix}vote_max_fraction"] = float(np.max(probabilities))
    features[f"{prefix}vote_unique_relations"] = float(len(counts))
    if generation_prediction is not None:
        features[f"{prefix}vote_generation_fraction"] = float(
            counts.get(generation_prediction, 0) / len(votes)
        )
    if closed_prediction is not None:
        features[f"{prefix}vote_closed_fraction"] = float(
            counts.get(closed_prediction, 0) / len(votes)
        )


def build_sample_features(
    *,
    source_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    head_rows: Mapping[str, Mapping[str, Any]],
    positive_heads: Sequence[str],
    negative_heads: Sequence[str],
) -> Dict[str, Any]:
    sid = int(baseline_row["sid"])
    generation_prediction = normalize_relation(baseline_row.get("prediction"))
    original_scores = relation_scores(
        baseline_row.get("closed_scores")
        or baseline_row.get("baseline_closed_scores")
    )
    swapped_scores = relation_scores(source_row.get("swapped_relation_logits"))
    if original_scores is None:
        raise RuntimeError(f"SID {sid}: missing original closed scores")
    if swapped_scores is None:
        raise RuntimeError(f"SID {sid}: missing swapped_relation_logits")

    original_vector = score_vector(original_scores)
    swapped_vector = score_vector(swapped_scores)
    original_closed_prediction = top_relation(original_vector)
    swapped_closed_prediction = top_relation(swapped_vector)
    aligned_swapped_vector = np.asarray(
        [swapped_scores[OPPOSITE[relation]] for relation in RELATIONS],
        dtype=np.float64,
    )

    original_correct = bool(baseline_row.get("correct"))
    cached_correct = source_original_correct(source_row)
    if cached_correct is not None and bool(cached_correct) != original_correct:
        raise RuntimeError(
            f"SID {sid}: baseline correctness mismatch between existing outputs"
        )

    result: Dict[str, Any] = {
        "sid": sid,
        "label_wrong": int(not original_correct),
        "baseline_correct": int(original_correct),
        "gt": normalize_relation(baseline_row.get("gt") or source_row.get("gt")),
        "generation_prediction": generation_prediction,
        "closed_prediction": original_closed_prediction,
        "swapped_closed_prediction": swapped_closed_prediction,
        "source_generation_pair_status": str(
            source_row.get("generation_pair_status", "unknown")
        ),
        "status_alias": STATUS_ALIASES.get(
            str(source_row.get("generation_pair_status", "")),
            str(source_row.get("generation_pair_status", "unknown")),
        ),
    }
    features: Dict[str, float] = {}

    # Output / next-token confidence baseline.
    add_vector_features(
        features,
        "conf_original_",
        original_vector,
        generation_prediction=generation_prediction,
        closed_prediction=original_closed_prediction,
    )
    add_vector_features(
        features,
        "conf_swapped_",
        swapped_vector,
        generation_prediction=(
            OPPOSITE[generation_prediction]
            if generation_prediction in OPPOSITE
            else None
        ),
        closed_prediction=swapped_closed_prediction,
    )
    features["conf_generation_closed_agreement"] = float(
        generation_prediction is not None
        and generation_prediction == original_closed_prediction
    )
    features["conf_generation_parsed"] = float(generation_prediction is not None)
    features["conf_generation_token_count"] = float(
        baseline_row.get("new_token_count", 0) or 0
    )

    # Query-swap consistency without GT.
    pair_inverse_consistent = bool(
        swapped_closed_prediction == OPPOSITE[original_closed_prediction]
    )
    features["pair_closed_inverse_consistent"] = float(pair_inverse_consistent)
    features["pair_closed_inverse_inconsistent"] = float(not pair_inverse_consistent)
    features["pair_closed_same_relation"] = float(
        swapped_closed_prediction == original_closed_prediction
    )
    features["pair_aligned_centered_cosine"] = centered_cosine(
        original_vector, aligned_swapped_vector
    )
    orig_centered = original_vector - np.mean(original_vector)
    swap_centered = aligned_swapped_vector - np.mean(aligned_swapped_vector)
    features["pair_aligned_centered_l2"] = float(
        np.linalg.norm(orig_centered - swap_centered)
    )
    features["pair_aligned_centered_l2_relative"] = safe_ratio(
        np.linalg.norm(orig_centered - swap_centered),
        np.linalg.norm(orig_centered) + np.linalg.norm(swap_centered),
    )
    features["pair_top_margin_min"] = float(
        min(top_margin(original_vector), top_margin(swapped_vector))
    )
    features["pair_top_margin_abs_difference"] = float(
        abs(top_margin(original_vector) - top_margin(swapped_vector))
    )
    features["pair_entropy_abs_difference"] = float(
        abs(
            entropy_from_probabilities(stable_softmax(original_vector))
            - entropy_from_probabilities(stable_softmax(swapped_vector))
        )
    )
    if generation_prediction is not None:
        features["pair_generation_vs_swapped_closed_consistent"] = float(
            swapped_closed_prediction == OPPOSITE[generation_prediction]
        )

    all_required_heads = list(dict.fromkeys(list(positive_heads) + list(negative_heads)))
    missing = [head for head in all_required_heads if head not in head_rows]
    if missing:
        raise RuntimeError(f"SID {sid}: missing head rows: {missing}")

    vectors_by_head: Dict[str, np.ndarray] = {}
    changed_predictions: List[Optional[str]] = []
    for head in all_required_heads:
        row = head_rows[head]
        contribution = relation_scores(row.get("head_logit_contribution"))
        if contribution is None:
            raise RuntimeError(f"SID {sid} head {head}: missing contribution vector")
        vector = score_vector(contribution)
        vectors_by_head[head] = vector
        safe_head = sanitize_name(head)
        prefix = f"head_{safe_head}_"
        add_vector_features(
            features,
            prefix,
            vector,
            generation_prediction=generation_prediction,
            closed_prediction=original_closed_prediction,
        )
        features[f"{prefix}receiver_delta_norm"] = float(
            row.get("receiver_delta_norm", float("nan"))
        )
        features[f"{prefix}receiver_delta_ratio"] = float(
            row.get("receiver_delta_ratio", float("nan"))
        )

        changed = bool(row.get("generation_prediction_changed", False))
        ablated_prediction = normalize_relation(
            row.get("ablated_generation_prediction")
        )
        features[f"probe_{safe_head}_generation_changed"] = float(changed)
        features[f"probe_{safe_head}_ablated_parsed"] = float(
            ablated_prediction is not None
        )
        if generation_prediction is not None:
            features[f"probe_{safe_head}_ablated_equals_original_generation"] = float(
                ablated_prediction == generation_prediction
            )
        changed_predictions.append(ablated_prediction)

    def aggregate_group(name: str, heads: Sequence[str]) -> np.ndarray:
        vectors = [vectors_by_head[head] for head in heads]
        matrix = np.stack(vectors, axis=0)
        summed = np.sum(matrix, axis=0)
        prefix = f"circ_{name}_"
        add_vector_features(
            features,
            prefix,
            summed,
            generation_prediction=generation_prediction,
            closed_prediction=original_closed_prediction,
        )
        features[f"{prefix}head_count"] = float(len(heads))
        features[f"{prefix}mean_head_norm"] = float(
            np.mean(np.linalg.norm(matrix - matrix.mean(axis=1, keepdims=True), axis=1))
        )
        features[f"{prefix}relation_variance_mean"] = float(
            np.mean(np.var(matrix, axis=0))
        )
        features[f"{prefix}head_pair_disagreement"] = float(
            np.mean(
                [
                    top_relation(vectors[i]) != top_relation(vectors[j])
                    for i in range(len(vectors))
                    for j in range(i + 1, len(vectors))
                ]
            )
            if len(vectors) >= 2
            else 0.0
        )
        features[f"{prefix}mean_receiver_delta_ratio"] = float(
            np.mean(
                [
                    float(head_rows[head].get("receiver_delta_ratio", float("nan")))
                    for head in heads
                ]
            )
        )
        head_vote_features(
            features,
            prefix,
            vectors,
            generation_prediction=generation_prediction,
            closed_prediction=original_closed_prediction,
        )
        return summed

    positive_sum = aggregate_group("positive", positive_heads)
    negative_sum = aggregate_group("negative", negative_heads)
    features["circ_pos_neg_centered_cosine"] = centered_cosine(
        positive_sum, negative_sum
    )
    features["circ_pos_neg_norm_ratio"] = safe_ratio(
        np.linalg.norm(positive_sum - np.mean(positive_sum)),
        np.linalg.norm(negative_sum - np.mean(negative_sum)),
    )
    features["circ_pos_neg_top_same"] = float(
        top_relation(positive_sum) == top_relation(negative_sum)
    )
    features["circ_pos_neg_top_opposite"] = float(
        top_relation(negative_sum) == OPPOSITE[top_relation(positive_sum)]
    )
    if generation_prediction is not None:
        features["circ_pos_minus_neg_generation_margin"] = float(
            target_margin(positive_sum, generation_prediction)
            - target_margin(negative_sum, generation_prediction)
        )
        features["circ_positive_negative_generation_agreement"] = float(
            top_relation(positive_sum) == generation_prediction
            and top_relation(negative_sum) == generation_prediction
        )

    parsed_changed = [value for value in changed_predictions if value is not None]
    features["probe_changed_head_count"] = float(
        sum(
            bool(head_rows[head].get("generation_prediction_changed", False))
            for head in all_required_heads
        )
    )
    features["probe_changed_head_fraction"] = safe_ratio(
        features["probe_changed_head_count"], len(all_required_heads)
    )
    features["probe_unique_ablated_predictions"] = float(len(set(parsed_changed)))
    features["probe_ablated_parse_fraction"] = safe_ratio(
        len(parsed_changed), len(all_required_heads)
    )
    if generation_prediction is not None:
        features["probe_ablations_disagree_generation_fraction"] = safe_ratio(
            sum(value != generation_prediction for value in parsed_changed),
            len(all_required_heads),
        )

    result.update(features)
    return result


# -----------------------------------------------------------------------------
# Statistics and cross-validation
# -----------------------------------------------------------------------------


def feature_columns_for_set(
    all_columns: Sequence[str],
    feature_set: str,
) -> List[str]:
    columns = list(all_columns)
    confidence = [column for column in columns if column.startswith("conf_")]
    pair = [column for column in columns if column.startswith("pair_")]
    aggregate = [column for column in columns if column.startswith("circ_")]
    per_head = [column for column in columns if column.startswith("head_")]
    probe = [column for column in columns if column.startswith("probe_")]
    mapping = {
        "confidence": confidence,
        "pair": pair,
        "circuit_aggregate": aggregate,
        "circuit_static": aggregate + per_head,
        "circuit_active": aggregate + per_head + probe,
        "confidence_pair": confidence + pair,
        "confidence_circuit": confidence + aggregate + per_head,
        "all": confidence + pair + aggregate + per_head + probe,
    }
    if feature_set not in mapping:
        raise ValueError(
            f"Unknown feature set {feature_set}; available={sorted(mapping)}"
        )
    return list(dict.fromkeys(mapping[feature_set]))


def metric_payload(y_true: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y_true, probability)),
        "AUPRC": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "accuracy_at_0.5": float(np.mean(prediction == y_true)),
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "precision_wrong_at_0.5": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall_wrong_at_0.5": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "F1_wrong_at_0.5": float(f1_score(y_true, prediction, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    metric: str,
    repeats: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: List[float] = []
    n = len(y_true)
    for _ in range(int(repeats)):
        index = rng.integers(0, n, size=n)
        y = y_true[index]
        if len(np.unique(y)) < 2:
            continue
        p = probability[index]
        if metric == "AUROC":
            values.append(float(roc_auc_score(y, p)))
        elif metric == "AUPRC":
            values.append(float(average_precision_score(y, p)))
        else:
            raise ValueError(metric)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(np.asarray(values), [0.025, 0.975]).tolist())


def repeated_oof_logistic(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    y: np.ndarray,
    folds: int,
    repeats: int,
    c_value: float,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    x = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    prediction_sum = np.zeros(len(frame), dtype=np.float64)
    prediction_count = np.zeros(len(frame), dtype=np.int64)
    fold_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []

    for repeat in range(int(repeats)):
        splitter = StratifiedKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(seed) + 1009 * repeat,
        )
        for fold, (train_index, test_index) in enumerate(splitter.split(x, y)):
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=float(c_value),
                            class_weight="balanced",
                            max_iter=5000,
                            solver="liblinear",
                            random_state=int(seed) + repeat,
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(x.iloc[train_index], y[train_index])
            probability = pipeline.predict_proba(x.iloc[test_index])[:, 1]
            prediction_sum[test_index] += probability
            prediction_count[test_index] += 1

            fold_metric = metric_payload(y[test_index], probability)
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_N": len(train_index),
                    "test_N": len(test_index),
                    **fold_metric,
                }
            )
            coefficients = pipeline.named_steps["classifier"].coef_[0]
            for feature, coefficient in zip(feature_columns, coefficients):
                coefficient_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "feature": feature,
                        "coefficient": float(coefficient),
                    }
                )

    if np.any(prediction_count != int(repeats)):
        raise RuntimeError(
            "Each sample must receive exactly one out-of-fold prediction per repeat"
        )
    return prediction_sum / prediction_count, fold_rows, coefficient_rows


def risk_coverage_rows(
    *,
    y: np.ndarray,
    probability: np.ndarray,
    feature_set: str,
    review_fractions: Sequence[float],
) -> List[Dict[str, Any]]:
    order = np.argsort(-probability, kind="mergesort")
    n = len(y)
    total_errors = int(np.sum(y))
    rows: List[Dict[str, Any]] = []
    for fraction in review_fractions:
        review_n = int(round(float(fraction) * n))
        review_n = min(max(review_n, 0), n - 1)
        reviewed = order[:review_n]
        retained = order[review_n:]
        reviewed_errors = int(np.sum(y[reviewed])) if review_n else 0
        retained_errors = int(np.sum(y[retained]))
        rows.append(
            {
                "feature_set": feature_set,
                "review_fraction": float(fraction),
                "review_N": review_n,
                "coverage": float(len(retained) / n),
                "review_error_precision": float(
                    reviewed_errors / review_n if review_n else float("nan")
                ),
                "review_error_recall": float(
                    reviewed_errors / total_errors if total_errors else float("nan")
                ),
                "retained_accuracy": float(1.0 - retained_errors / len(retained)),
                "retained_errors": retained_errors,
                "reviewed_errors": reviewed_errors,
            }
        )
    return rows


def univariate_summary(frame: pd.DataFrame, feature_columns: Sequence[str]) -> List[Dict[str, Any]]:
    y = frame["label_wrong"].to_numpy(dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    for feature in feature_columns:
        x = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(x)
        if np.sum(finite) < 20 or len(np.unique(x[finite])) < 2:
            continue
        median = float(np.nanmedian(x))
        x = np.where(np.isfinite(x), x, median)
        raw_auc = float(roc_auc_score(y, x))
        direction = 1.0 if raw_auc >= 0.5 else -1.0
        discriminative_auc = max(raw_auc, 1.0 - raw_auc)
        correct = x[y == 0]
        wrong = x[y == 1]
        pooled = math.sqrt(
            max(
                (
                    (len(correct) - 1) * np.var(correct, ddof=1)
                    + (len(wrong) - 1) * np.var(wrong, ddof=1)
                )
                / max(len(correct) + len(wrong) - 2, 1),
                1e-24,
            )
        )
        cohen_d_wrong_minus_correct = float(
            (np.mean(wrong) - np.mean(correct)) / pooled
        )
        rows.append(
            {
                "feature": feature,
                "N": int(len(x)),
                "mean_correct": float(np.mean(correct)),
                "mean_wrong": float(np.mean(wrong)),
                "wrong_minus_correct": float(np.mean(wrong) - np.mean(correct)),
                "cohen_d_wrong_minus_correct": cohen_d_wrong_minus_correct,
                "raw_AUROC_wrong_high": raw_auc,
                "best_direction_AUROC": discriminative_auc,
                "risk_direction": "high_is_wrong" if direction > 0 else "low_is_wrong",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["best_direction_AUROC"]),
            -abs(float(row["cohen_d_wrong_minus_correct"])),
        ),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if args.cv_repeats < 1:
        raise ValueError("--cv-repeats must be >= 1")
    if args.logistic_c <= 0:
        raise ValueError("--logistic-c must be positive")

    source_dir = Path(args.source_output_dir)
    head_dir = Path(args.head_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = source_dir / "config.json"
    source_path = source_dir / "extraction.jsonl"
    baseline_path = head_dir / "baseline_generation.jsonl"
    head_effect_path = head_dir / "head_path_ablation_generation.jsonl"
    for path in (source_config_path, source_path, baseline_path, head_effect_path):
        if not path.exists():
            raise FileNotFoundError(path)

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if str(source_config.get("model")) != str(args.model):
        raise RuntimeError(
            f"Source model={source_config.get('model')} but --model={args.model}"
        )

    bundles = load_bundles(Path(args.bundle_json))
    if args.positive_bundle not in bundles:
        raise KeyError(f"Missing bundle {args.positive_bundle}")
    if args.negative_bundle not in bundles:
        raise KeyError(f"Missing bundle {args.negative_bundle}")
    positive_heads = bundles[args.positive_bundle]
    negative_heads = bundles[args.negative_bundle]
    required_heads = set(positive_heads) | set(negative_heads)

    source_rows = {int(row["sid"]): row for row in read_jsonl(source_path)}
    baseline_rows = {int(row["sid"]): row for row in read_jsonl(baseline_path)}
    head_rows_by_sid: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(head_effect_path):
        sid = int(row["sid"])
        head = parse_head_text(row["head"])
        if head in required_heads:
            head_rows_by_sid[sid][head] = row

    common_sids = sorted(set(source_rows) & set(baseline_rows) & set(head_rows_by_sid))
    complete_sids = [
        sid
        for sid in common_sids
        if required_heads.issubset(set(head_rows_by_sid[sid]))
    ]
    if args.sample_max_samples > 0:
        complete_sids = complete_sids[: int(args.sample_max_samples)]
    if not complete_sids:
        raise RuntimeError("No complete common samples")

    feature_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for sid in complete_sids:
        try:
            feature_rows.append(
                build_sample_features(
                    source_row=source_rows[sid],
                    baseline_row=baseline_rows[sid],
                    head_rows=head_rows_by_sid[sid],
                    positive_heads=positive_heads,
                    negative_heads=negative_heads,
                )
            )
        except Exception as exc:
            skipped.append(
                {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if skipped:
        write_json(output_dir / "skipped_samples.json", skipped)
    if not feature_rows:
        raise RuntimeError("Feature extraction produced no samples")

    frame = pd.DataFrame(feature_rows).sort_values("sid").reset_index(drop=True)
    frame.to_csv(output_dir / "error_detector_features.csv", index=False)

    y = frame["label_wrong"].to_numpy(dtype=np.int64)
    wrong_count = int(np.sum(y))
    correct_count = int(len(y) - wrong_count)
    if min(wrong_count, correct_count) < args.cv_folds:
        raise RuntimeError(
            f"Smallest class has {min(wrong_count, correct_count)} samples, "
            f"less than cv_folds={args.cv_folds}"
        )

    metadata_columns = {
        "sid",
        "label_wrong",
        "baseline_correct",
        "gt",
        "generation_prediction",
        "closed_prediction",
        "swapped_closed_prediction",
        "source_generation_pair_status",
        "status_alias",
    }
    numeric_candidates = [
        column
        for column in frame.columns
        if column not in metadata_columns
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]

    requested_feature_sets = parse_csv_tokens(args.feature_sets)
    valid_feature_sets = {
        "confidence",
        "pair",
        "circuit_aggregate",
        "circuit_static",
        "circuit_active",
        "confidence_pair",
        "confidence_circuit",
        "all",
    }
    unknown = set(requested_feature_sets) - valid_feature_sets
    if unknown:
        raise ValueError(f"Unknown feature sets: {sorted(unknown)}")

    review_fractions = parse_float_list(args.review_fractions)
    oof_frame = frame[
        [
            "sid",
            "label_wrong",
            "baseline_correct",
            "gt",
            "generation_prediction",
            "source_generation_pair_status",
            "status_alias",
        ]
    ].copy()
    detector_summary: List[Dict[str, Any]] = []
    all_fold_rows: List[Dict[str, Any]] = []
    all_coefficient_rows: List[Dict[str, Any]] = []
    all_risk_rows: List[Dict[str, Any]] = []
    predictions_by_set: Dict[str, np.ndarray] = {}

    for set_index, feature_set in enumerate(requested_feature_sets):
        columns = feature_columns_for_set(numeric_candidates, feature_set)
        if not columns:
            raise RuntimeError(f"Feature set {feature_set} is empty")
        probability, fold_rows, coefficient_rows = repeated_oof_logistic(
            frame=frame,
            feature_columns=columns,
            y=y,
            folds=int(args.cv_folds),
            repeats=int(args.cv_repeats),
            c_value=float(args.logistic_c),
            seed=int(args.seed) + 10000 * set_index,
        )
        predictions_by_set[feature_set] = probability
        oof_frame[f"risk_{feature_set}"] = probability
        metrics = metric_payload(y, probability)
        auroc_ci = bootstrap_metric_ci(
            y,
            probability,
            metric="AUROC",
            repeats=int(args.bootstrap_repeats),
            seed=int(args.seed) + 101 * set_index,
        )
        auprc_ci = bootstrap_metric_ci(
            y,
            probability,
            metric="AUPRC",
            repeats=int(args.bootstrap_repeats),
            seed=int(args.seed) + 201 * set_index,
        )
        detector_summary.append(
            {
                "feature_set": feature_set,
                "feature_count": len(columns),
                "N": len(frame),
                "wrong_N": wrong_count,
                "wrong_prevalence": float(wrong_count / len(frame)),
                **metrics,
                "AUROC_ci_low": auroc_ci[0],
                "AUROC_ci_high": auroc_ci[1],
                "AUPRC_ci_low": auprc_ci[0],
                "AUPRC_ci_high": auprc_ci[1],
            }
        )
        for row in fold_rows:
            all_fold_rows.append({"feature_set": feature_set, **row})
        for row in coefficient_rows:
            all_coefficient_rows.append({"feature_set": feature_set, **row})
        all_risk_rows.extend(
            risk_coverage_rows(
                y=y,
                probability=probability,
                feature_set=feature_set,
                review_fractions=review_fractions,
            )
        )

    oof_frame.to_csv(output_dir / "detector_oof_predictions.csv", index=False)
    summary_frame = pd.DataFrame(detector_summary).sort_values(
        ["AUROC", "AUPRC"], ascending=[False, False]
    )
    summary_frame.to_csv(output_dir / "detector_summary.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(
        output_dir / "detector_fold_metrics.csv", index=False
    )
    pd.DataFrame(all_risk_rows).to_csv(
        output_dir / "detector_risk_coverage.csv", index=False
    )

    coefficient_frame = pd.DataFrame(all_coefficient_rows)
    if not coefficient_frame.empty:
        coefficient_summary = (
            coefficient_frame.groupby(["feature_set", "feature"], as_index=False)
            .agg(
                coefficient_mean=("coefficient", "mean"),
                coefficient_std=("coefficient", "std"),
                mean_abs_coefficient=("coefficient", lambda x: np.mean(np.abs(x))),
                positive_fold_fraction=("coefficient", lambda x: np.mean(np.asarray(x) > 0)),
            )
            .sort_values(
                ["feature_set", "mean_abs_coefficient"],
                ascending=[True, False],
            )
        )
        coefficient_summary.to_csv(
            output_dir / "detector_feature_coefficients.csv", index=False
        )
    else:
        (output_dir / "detector_feature_coefficients.csv").write_text(
            "", encoding="utf-8"
        )

    univariate = univariate_summary(frame, numeric_candidates)
    write_csv(output_dir / "detector_univariate_features.csv", univariate)

    best_feature_set = str(summary_frame.iloc[0]["feature_set"])
    best_probability = predictions_by_set[best_feature_set]
    group_rows: List[Dict[str, Any]] = []
    for grouping, key_values in (
        ("status", frame["status_alias"]),
        ("gt", frame["gt"]),
        ("generation_prediction", frame["generation_prediction"]),
    ):
        for value in sorted(set(map(str, key_values))):
            mask = np.asarray([str(item) == value for item in key_values])
            part_y = y[mask]
            part_p = best_probability[mask]
            group_rows.append(
                {
                    "grouping": grouping,
                    "group": value,
                    "N": int(np.sum(mask)),
                    "wrong_N": int(np.sum(part_y)),
                    "wrong_rate": float(np.mean(part_y)),
                    "mean_predicted_risk": float(np.mean(part_p)),
                    "median_predicted_risk": float(np.median(part_p)),
                }
            )
    write_csv(output_dir / "detector_group_risk_summary.csv", group_rows)

    comparison: Dict[str, Any] = {}
    if "all" in predictions_by_set and "confidence" in predictions_by_set:
        rng = np.random.default_rng(int(args.seed) + 999)
        differences: List[float] = []
        n = len(y)
        for _ in range(int(args.bootstrap_repeats)):
            index = rng.integers(0, n, size=n)
            if len(np.unique(y[index])) < 2:
                continue
            differences.append(
                float(
                    roc_auc_score(y[index], predictions_by_set["all"][index])
                    - roc_auc_score(
                        y[index], predictions_by_set["confidence"][index]
                    )
                )
            )
        comparison = {
            "all_minus_confidence_AUROC": float(
                roc_auc_score(y, predictions_by_set["all"])
                - roc_auc_score(y, predictions_by_set["confidence"])
            ),
            "bootstrap_ci_low": float(np.quantile(differences, 0.025))
            if differences
            else float("nan"),
            "bootstrap_ci_high": float(np.quantile(differences, 0.975))
            if differences
            else float("nan"),
            "bootstrap_probability_difference_positive": float(
                np.mean(np.asarray(differences) > 0)
            )
            if differences
            else float("nan"),
        }

    pair_rule = frame["pair_closed_inverse_inconsistent"].to_numpy(dtype=float) >= 0.5
    pair_rule_metrics = {
        "flagged_N": int(np.sum(pair_rule)),
        "precision_wrong": float(precision_score(y, pair_rule, zero_division=0)),
        "recall_wrong": float(recall_score(y, pair_rule, zero_division=0)),
        "F1_wrong": float(f1_score(y, pair_rule, zero_division=0)),
    }

    config = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "source_output_dir": str(source_dir),
        "head_output_dir": str(head_dir),
        "bundle_json": str(args.bundle_json),
        "positive_bundle": args.positive_bundle,
        "negative_bundle": args.negative_bundle,
        "positive_heads": positive_heads,
        "negative_heads": negative_heads,
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats,
        "logistic_c": args.logistic_c,
        "bootstrap_repeats": args.bootstrap_repeats,
        "feature_sets": requested_feature_sets,
        "review_fractions": review_fractions,
        "leakage_policy": (
            "GT/correct/fixed/broken/E_GT/misleading/status are excluded from all "
            "detector features; correctness is used only as label"
        ),
    }
    write_json(output_dir / "config.json", config)

    final_summary = {
        "script_version": SCRIPT_VERSION,
        "N": len(frame),
        "correct_N": correct_count,
        "wrong_N": wrong_count,
        "baseline_generation_accuracy": float(correct_count / len(frame)),
        "wrong_prevalence": float(wrong_count / len(frame)),
        "common_complete_sids": len(complete_sids),
        "skipped_samples": len(skipped),
        "best_feature_set": best_feature_set,
        "best_detector": summary_frame.iloc[0].to_dict(),
        "all_minus_confidence_comparison": comparison,
        "pair_inverse_inconsistency_rule": pair_rule_metrics,
    }
    write_json(output_dir / "summary.json", final_summary)

    print("\n" + "=" * 152)
    print("CIRCUIT ERROR-DETECTION RESULT")
    print("=" * 152)
    print(
        f"Samples={len(frame)} | correct={correct_count} | wrong={wrong_count} | "
        f"baseline generation accuracy={correct_count/len(frame):.4f}"
    )
    print(
        "All metrics below use repeated out-of-fold predictions; "
        "GT is used only as the error label."
    )
    print("\nFEATURE-SET COMPARISON")
    print(
        f"{'feature_set':>22} {'features':>8} {'AUROC':>8} {'AUPRC':>8} "
        f"{'F1':>8} {'Brier':>8} {'AUROC 95% CI':>23}"
    )
    for _, row in summary_frame.iterrows():
        print(
            f"{str(row['feature_set']):>22} {int(row['feature_count']):8d} "
            f"{float(row['AUROC']):8.4f} {float(row['AUPRC']):8.4f} "
            f"{float(row['F1_wrong_at_0.5']):8.4f} {float(row['brier']):8.4f} "
            f"[{float(row['AUROC_ci_low']):.4f}, {float(row['AUROC_ci_high']):.4f}]"
        )

    print("\nPAIR-INVERSE INCONSISTENCY RULE")
    print(
        f"flagged={pair_rule_metrics['flagged_N']} | "
        f"precision={pair_rule_metrics['precision_wrong']:.4f} | "
        f"recall={pair_rule_metrics['recall_wrong']:.4f} | "
        f"F1={pair_rule_metrics['F1_wrong']:.4f}"
    )

    if comparison:
        print("\nINCREMENTAL VALUE OVER CONFIDENCE")
        print(
            "all - confidence AUROC = "
            f"{comparison['all_minus_confidence_AUROC']:+.4f} "
            f"CI=[{comparison['bootstrap_ci_low']:+.4f}, "
            f"{comparison['bootstrap_ci_high']:+.4f}]"
        )

    best_risk = pd.DataFrame(all_risk_rows)
    best_risk = best_risk[best_risk["feature_set"] == best_feature_set]
    print(f"\nRISK-COVERAGE FOR BEST SET: {best_feature_set}")
    print(
        f"{'review':>8} {'coverage':>10} {'errRecall':>10} "
        f"{'errPrecision':>12} {'retainedAcc':>12}"
    )
    for _, row in best_risk.iterrows():
        print(
            f"{float(row['review_fraction']):8.2f} "
            f"{float(row['coverage']):10.4f} "
            f"{float(row['review_error_recall']):10.4f} "
            f"{float(row['review_error_precision']):12.4f} "
            f"{float(row['retained_accuracy']):12.4f}"
        )

    print("\nTOP UNIVARIATE CIRCUIT FEATURES")
    circuit_univariate = [
        row
        for row in univariate
        if str(row["feature"]).startswith(("circ_", "head_", "probe_"))
    ][:15]
    print(
        f"{'feature':>58} {'AUROC*':>8} {'direction':>14} {'d(w-c)':>10}"
    )
    for row in circuit_univariate:
        print(
            f"{str(row['feature']):>58} "
            f"{float(row['best_direction_AUROC']):8.4f} "
            f"{str(row['risk_direction']):>14} "
            f"{float(row['cohen_d_wrong_minus_correct']):+10.4f}"
        )

    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
