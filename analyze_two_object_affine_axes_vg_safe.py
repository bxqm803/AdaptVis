#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-validated opposing-affine-axis probe for COCO_two / VG_two states.

Input is the .npz produced by extract_two_object_relation_states.py. For each
stored decoder depth and each available opposing relation pair, this script
fits on training folds only:

  d_a = normalize(mean(r | positive_a) - mean(r | negative_a))
  c_a = 0.5 * (mean(r | positive_a) + mean(r | negative_a))

where r = h(subject) - h(reference). At test time it reports:
  * conditional sign accuracy for each opposing axis;
  * a multi-affine routing accuracy across all available axes;
  * axis-direction cosine geometry;
  * a train-label shuffle control.

No VLM parameters are updated. The labels are only used inside each training
fold to estimate the axis centres and directions.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn with StratifiedGroupKFold is required: pip install -U scikit-learn") from exc


AXIS_CANDIDATES = [
    ("horizontal", "left", "right"),
    ("vertical", "below", "above"),
    ("depth", "behind", "front"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label-shuffle-repeats", type=int, default=20)
    parser.add_argument(
        "--min-axis-class-count",
        type=int,
        default=20,
        help=(
            "Only analyse an opposing axis when both of its labels have at least this many samples. "
            "This prevents sparse VG front/behind or top/bottom labels from destabilising the main CV result. "
            "Use 5 for an explicitly exploratory sparse-axis run."
        ),
    )
    parser.add_argument("--no-label-shuffle", action="store_true")
    parser.add_argument("--output", required=True, help="Output JSON path (or prefix without suffix).")
    return parser.parse_args()


def safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray) -> np.ndarray:
    norm = safe_norm(x)
    if not np.isfinite(norm) or norm <= 1e-10:
        raise RuntimeError("Degenerate opposing direction with near-zero norm.")
    return x / norm


def json_float(x: Any) -> Any:
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return [json_float(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): json_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_float(v) for v in x]
    return x


def available_axes(labels: Sequence[str]) -> List[Tuple[str, str, str]]:
    values = set(labels)
    return [axis for axis in AXIS_CANDIDATES if axis[1] in values and axis[2] in values]


def select_evaluable_axes(
    labels: Sequence[str], min_axis_class_count: int
) -> Tuple[List[Tuple[str, str, str]], List[Dict[str, Any]]]:
    """Keep only complete, sufficiently supported opposing axes.

    Some VG_two releases are strongly imbalanced: an axis can be absent
    altogether (e.g. no bottom labels) or have only a handful of samples.
    Such labels must not make a valid horizontal evaluation fail.
    """
    if min_axis_class_count < 1:
        raise ValueError("--min-axis-class-count must be >= 1")
    counts = Counter(labels)
    selected: List[Tuple[str, str, str]] = []
    diagnostics: List[Dict[str, Any]] = []
    for name, negative, positive in AXIS_CANDIDATES:
        n_neg = int(counts.get(negative, 0))
        n_pos = int(counts.get(positive, 0))
        complete = n_neg > 0 and n_pos > 0
        sufficient = min(n_neg, n_pos) >= min_axis_class_count
        diagnostics.append(
            {
                "name": name,
                "negative": negative,
                "positive": positive,
                "negative_count": n_neg,
                "positive_count": n_pos,
                "complete": complete,
                "selected": bool(complete and sufficient),
                "reason": (
                    "selected"
                    if complete and sufficient
                    else ("missing_opposite_label" if not complete else "below_min_axis_class_count")
                ),
            }
        )
        if complete and sufficient:
            selected.append((name, negative, positive))
    return selected, diagnostics


def choose_n_splits(labels: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """Bound CV folds by per-label sample and distinct-image support."""
    if requested < 2:
        raise ValueError("--cv-folds must be at least 2")
    values = sorted(set(labels.tolist()))
    if not values:
        raise RuntimeError("No labels remain after axis filtering.")
    per_label_samples = [int(np.sum(labels == value)) for value in values]
    per_label_groups = [len(set(groups[labels == value].tolist())) for value in values]
    usable = min([requested, *per_label_samples, *per_label_groups])
    if usable < 2:
        details = {value: {"samples": int(np.sum(labels == value)), "groups": len(set(groups[labels == value].tolist()))} for value in values}
        raise RuntimeError(f"Insufficient distinct-image support for grouped CV: {details}")
    return int(usable)

def fit_axes(vectors: np.ndarray, labels: np.ndarray, axes: Sequence[Tuple[str, str, str]]) -> Dict[str, Dict[str, Any]]:
    fitted: Dict[str, Dict[str, Any]] = {}
    for name, negative, positive in axes:
        neg = vectors[labels == negative]
        pos = vectors[labels == positive]
        if len(neg) == 0 or len(pos) == 0:
            raise RuntimeError(f"Training split is missing {negative}/{positive} for axis {name}.")
        mean_neg = np.mean(neg, axis=0)
        mean_pos = np.mean(pos, axis=0)
        direction = unit(mean_pos - mean_neg)
        centre = 0.5 * (mean_pos + mean_neg)
        train_delta = vectors[np.isin(labels, [negative, positive])] - centre
        projection = train_delta @ direction
        residual = train_delta - projection[:, None] * direction[None, :]
        # A family-specific residual scale makes axis routing less sensitive to
        # differences in intrinsic spread across horizontal / vertical / depth.
        residual_scale = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        residual_scale = max(residual_scale, 1e-6)
        fitted[name] = {
            "negative": negative,
            "positive": positive,
            "direction": direction,
            "centre": centre,
            "residual_scale": residual_scale,
        }
    return fitted


def predict(vectors: np.ndarray, fitted: Mapping[str, Mapping[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    names = list(fitted)
    n = len(vectors)
    family_dist: Dict[str, np.ndarray] = {}
    projections: Dict[str, np.ndarray] = {}
    per_axis_relation: Dict[str, np.ndarray] = {}
    for name in names:
        params = fitted[name]
        centre = params["centre"]
        direction = params["direction"]
        delta = vectors - centre[None, :]
        proj = delta @ direction
        residual = delta - proj[:, None] * direction[None, :]
        family_dist[name] = np.sqrt(np.sum(residual * residual, axis=1)) / float(params["residual_scale"])
        projections[name] = proj
        per_axis_relation[name] = np.where(proj >= 0.0, params["positive"], params["negative"])
    dist_matrix = np.stack([family_dist[name] for name in names], axis=1)
    picked = np.argmin(dist_matrix, axis=1)
    predicted = np.asarray([per_axis_relation[names[p]][i] for i, p in enumerate(picked)], dtype=object)
    return predicted, projections, family_dist


def confusion(true: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> List[List[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(true, pred):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix.tolist()


def fold_splits(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    counts = Counter(labels.tolist())
    min_count = min(counts.values())
    if min_count < n_splits:
        raise RuntimeError(f"Smallest relation class has {min_count} samples; cannot use {n_splits}-fold stratification.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(labels), dtype=np.int8)
    return list(splitter.split(dummy, labels, groups))


def evaluate_cv(
    vectors: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    axes: Sequence[Tuple[str, str, str]],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    train_label_permutation: np.random.Generator | None = None,
) -> Dict[str, Any]:
    all_true: List[str] = []
    all_pred: List[str] = []
    sign_correct: Dict[str, List[bool]] = {name: [] for name, _, _ in axes}
    sign_scores: Dict[str, List[float]] = {name: [] for name, _, _ in axes}
    geometry: List[np.ndarray] = []
    centre_distances: List[Dict[str, float]] = []

    for train_idx, test_idx in splits:
        train_vectors = vectors[train_idx]
        train_labels = labels[train_idx].copy()
        if train_label_permutation is not None:
            train_labels = train_label_permutation.permutation(train_labels)
        fitted = fit_axes(train_vectors, train_labels, axes)
        pred, projections, _ = predict(vectors[test_idx], fitted)
        true = labels[test_idx]
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())
        for name, negative, positive in axes:
            mask = np.isin(true, [negative, positive])
            if not np.any(mask):
                continue
            expected_positive = true[mask] == positive
            projected_positive = projections[name][mask] >= 0.0
            sign_correct[name].extend((expected_positive == projected_positive).tolist())
            sign_scores[name].extend(projections[name][mask].tolist())

        directions = [fitted[name]["direction"] for name, _, _ in axes]
        geometry.append(np.asarray(directions) @ np.asarray(directions).T)
        fold_dist: Dict[str, float] = {}
        for a_idx, (a_name, _, _) in enumerate(axes):
            for b_name, _, _ in axes[a_idx + 1 :]:
                ca = fitted[a_name]["centre"]
                cb = fitted[b_name]["centre"]
                sa = fitted[a_name]["residual_scale"]
                sb = fitted[b_name]["residual_scale"]
                fold_dist[f"{a_name}__{b_name}"] = float(np.linalg.norm(ca - cb) / max(1e-6, 0.5 * (sa + sb)))
        centre_distances.append(fold_dist)

    labels_order = sorted(set(labels.tolist()))
    per_axis = {}
    for name, negative, positive in axes:
        values = sign_correct[name]
        per_axis[name] = {
            "negative": negative,
            "positive": positive,
            "n": len(values),
            "sign_accuracy": float(np.mean(values)) if values else None,
            "mean_signed_projection": float(np.mean(sign_scores[name])) if sign_scores[name] else None,
        }
    centre_summary = {}
    for key in sorted({key for item in centre_distances for key in item}):
        vals = [item[key] for item in centre_distances if key in item]
        centre_summary[key] = float(np.mean(vals)) if vals else None

    result = {
        "n": int(len(all_true)),
        "affine_routing_accuracy": float(np.mean(np.asarray(all_true, dtype=object) == np.asarray(all_pred, dtype=object))),
        "per_axis_sign_accuracy": per_axis,
        "relation_order": labels_order,
        "confusion_matrix": confusion(all_true, all_pred, labels_order),
        "mean_axis_cosine_gram": np.mean(np.stack(geometry, axis=0), axis=0).tolist(),
        "axis_order": [name for name, _, _ in axes],
        "mean_family_centre_distance": centre_summary,
    }
    return result


def full_diagnostic(vectors: np.ndarray, labels: np.ndarray, axes: Sequence[Tuple[str, str, str]]) -> Dict[str, Any]:
    fitted = fit_axes(vectors, labels, axes)
    pred, projections, _ = predict(vectors, fitted)
    per_axis = {}
    for name, negative, positive in axes:
        mask = np.isin(labels, [negative, positive])
        expected_positive = labels[mask] == positive
        predicted_positive = projections[name][mask] >= 0.0
        per_axis[name] = float(np.mean(expected_positive == predicted_positive))
    dirs = np.asarray([fitted[name]["direction"] for name, _, _ in axes])
    return {
        "affine_routing_accuracy": float(np.mean(pred == labels)),
        "per_axis_sign_accuracy": per_axis,
        "axis_cosine_gram": (dirs @ dirs.T).tolist(),
        "axis_order": [name for name, _, _ in axes],
    }


def main() -> None:
    args = parse_args()
    source = Path(args.input_npz)
    with np.load(source, allow_pickle=True) as loaded:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        labels = np.asarray([str(x) for x in loaded["relation"].tolist()], dtype=object)
        groups = np.asarray([str(x) for x in loaded["image_id"].tolist()], dtype=object)
        vectors = loaded["relation_vectors"].astype(np.float32)
        block_ids = [int(v) for v in loaded["decoder_block_index"].tolist()]

    if vectors.ndim != 3:
        raise RuntimeError(f"Expected relation_vectors [N,L,H], got {vectors.shape}")

    raw_counts = Counter(labels.tolist())
    axes, axis_diagnostics = select_evaluable_axes(labels.tolist(), args.min_axis_class_count)
    if not axes:
        raise RuntimeError(
            "No opposing axis has enough support after filtering. "
            f"Parsed labels: {dict(raw_counts)}; min_axis_class_count={args.min_axis_class_count}; "
            f"axis_screening={axis_diagnostics}"
        )

    used_label_set = {label for _, negative, positive in axes for label in (negative, positive)}
    keep = np.isin(labels, sorted(used_label_set))
    excluded_counts = Counter(labels[~keep].tolist())
    vectors = vectors[keep]
    labels = labels[keep]
    groups = groups[keep]

    effective_cv_folds = choose_n_splits(labels, groups, args.cv_folds)
    splits = fold_splits(labels, groups, effective_cv_folds, args.seed)
    print(
        f"{metadata.get('dataset')} / {metadata.get('model_alias')}: n={len(labels)} "
        f"(raw={int(sum(raw_counts.values()))}), groups={len(set(groups.tolist()))}, "
        f"used_labels={dict(Counter(labels.tolist()))}, excluded={dict(excluded_counts)}, "
        f"axes={[x[0] for x in axes]}, cv_folds={effective_cv_folds}"
    )

    summary: Dict[str, Any] = {
        "input_npz": str(source),
        "metadata": metadata,
        "requested_cv_folds": args.cv_folds,
        "effective_cv_folds": effective_cv_folds,
        "group_unit": "image_id",
        "min_axis_class_count": args.min_axis_class_count,
        "raw_relation_counts": dict(raw_counts),
        "used_relation_counts": dict(Counter(labels.tolist())),
        "excluded_relation_counts": dict(excluded_counts),
        "axis_screening": axis_diagnostics,
        "axes": [{"name": name, "negative": neg, "positive": pos} for name, neg, pos in axes],
        "layers": {},
    }

    rng = np.random.default_rng(args.seed + 913)
    for layer_index, block in enumerate(block_ids):
        layer_vectors = vectors[:, layer_index, :]
        cv = evaluate_cv(layer_vectors, labels, groups, axes, splits)
        full = full_diagnostic(layer_vectors, labels, axes)
        layer_result: Dict[str, Any] = {
            "decoder_block_index": int(block),
            "pair_or_image_group_cv": cv,
            "full_data_diagnostic": full,
        }
        if not args.no_label_shuffle and args.label_shuffle_repeats > 0:
            scores = []
            for _ in range(args.label_shuffle_repeats):
                shuffled = evaluate_cv(layer_vectors, labels, groups, axes, splits, train_label_permutation=rng)
                scores.append(shuffled["affine_routing_accuracy"])
            layer_result["train_label_shuffle_control"] = {
                "repeats": int(args.label_shuffle_repeats),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "chance": float(1.0 / len(set(labels.tolist()))),
            }

        summary["layers"][str(block)] = layer_result
        axis_text = " | ".join(
            f"{name}={result['sign_accuracy']:.3f}"
            for name, result in cv["per_axis_sign_accuracy"].items()
            if result["sign_accuracy"] is not None
        )
        shuffle_text = ""
        if "train_label_shuffle_control" in layer_result:
            control = layer_result["train_label_shuffle_control"]
            shuffle_text = f" | shuffle={control['mean']:.3f}±{control['std']:.3f}"
        print(
            f"L{block}: affine-routing={cv['affine_routing_accuracy']:.3f} | {axis_text}{shuffle_text}"
        )

    out = Path(args.output)
    if out.suffix != ".json":
        out = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_float(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
