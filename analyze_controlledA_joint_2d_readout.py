#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint 2D cardinal readout for saved Controlled_Images_A LLaVA states.

This is a lightweight post-hoc analysis script: it reads the .npz emitted by
run_llava15_controlledA_spatial_id_probe_v6_affine.py and does NOT run or
fine-tune LLaVA again.

For each requested layer, it:
  1. builds a shared per-object baseline from all four relations;
  2. forms a high-dimensional relation residual
         r = [h(subject)-mu(subject)] - [h(reference)-mu(reference)];
  3. fits ONE affine linear map r -> z in R^2 against shared cardinal targets
         left=(-1,0), right=(+1,0), under=(0,-1), on=(0,+1);
  4. decodes by nearest cardinal target in the learned 2D space.

The VLM is frozen.  The only fitted object is a small linear readout.  With
--fit-all, the result is a descriptive full-data alignment score.  With
--cv-folds > 1, the script also reports grouped cross-validation accuracy,
which is the meaningful generalization estimate for this readout.

Run from the AdaptVis repository root, for example:

python3 analyze_controlledA_joint_2d_readout.py \
  --input-npz output/llava15_controlledA_spatial_id_affine_L13_L16_L31.npz \
  --layers 13,16,31 \
  --fit-all \
  --cv-folds 5 \
  --split-unit pair \
  --output output/llava15_controlledA_joint2d_linear_L13_L16_L31
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


LABELS: Tuple[str, ...] = ("left", "right", "under", "on")
TARGETS: Dict[str, np.ndarray] = {
    "left": np.array([-1.0, 0.0], dtype=np.float64),
    "right": np.array([1.0, 0.0], dtype=np.float64),
    "under": np.array([0.0, -1.0], dtype=np.float64),
    "on": np.array([0.0, 1.0], dtype=np.float64),
}


@dataclass(frozen=True)
class StateRecord:
    sid: int
    relation: str
    subject: str
    reference: str
    subject_state: np.ndarray
    reference_state: np.ndarray

    @property
    def group(self) -> str:
        return " || ".join(sorted((self.subject, self.reference)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a single supervised 2D cardinal readout to saved Controlled_A LLaVA residuals."
    )
    parser.add_argument(
        "--input-npz",
        required=True,
        help=".npz saved by run_llava15_controlledA_spatial_id_probe_v6_affine.py",
    )
    parser.add_argument(
        "--layers",
        default="13,16,31",
        help="Comma-separated zero-based decoder block indices present in the input .npz.",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Also fit/evaluate on all samples. This is descriptive, not held-out generalization.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of grouped CV folds. Set 1 to skip CV. Default: 5.",
    )
    parser.add_argument(
        "--split-unit",
        choices=["pair", "sample"],
        default="pair",
        help="CV grouping unit. 'pair' keeps an unordered subject/reference pair in one fold.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help=(
            "Relative ridge strength. The effective value is ridge times the mean diagonal "
            "of the train Gram matrix. Default: 1e-3."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        default="output/llava15_controlledA_joint2d_linear",
        help="Output base path; .json and .npz are appended.",
    )
    return parser.parse_args()


def parse_layers(text: str) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or min(values) < 0:
        raise ValueError(f"Invalid --layers: {text!r}")
    return values


def load_records(npz_path: Path, layer: int) -> List[StateRecord]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input .npz not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        required = [
            "sid",
            "relation",
            "subject_name",
            "reference_name",
            f"layer_{layer}_subject",
            f"layer_{layer}_reference",
        ]
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(
                f"The input .npz does not contain the requested layer {layer}. Missing keys: {missing}"
            )

        sid = np.asarray(data["sid"], dtype=np.int64)
        relation = np.asarray(data["relation"], dtype=object)
        subject = np.asarray(data["subject_name"], dtype=object)
        reference = np.asarray(data["reference_name"], dtype=object)
        subject_states = np.asarray(data[f"layer_{layer}_subject"], dtype=np.float64)
        reference_states = np.asarray(data[f"layer_{layer}_reference"], dtype=np.float64)

    n = len(sid)
    if not all(len(x) == n for x in (relation, subject, reference, subject_states, reference_states)):
        raise RuntimeError(f"Layer {layer}: inconsistent record counts in the input .npz.")

    records: List[StateRecord] = []
    for i in range(n):
        label = str(relation[i]).strip().lower()
        if label not in TARGETS:
            continue
        records.append(
            StateRecord(
                sid=int(sid[i]),
                relation=label,
                subject=str(subject[i]),
                reference=str(reference[i]),
                subject_state=subject_states[i].copy(),
                reference_state=reference_states[i].copy(),
            )
        )

    counts = Counter(record.relation for record in records)
    missing_labels = [label for label in LABELS if counts[label] == 0]
    if missing_labels:
        raise RuntimeError(f"Layer {layer}: missing relation labels: {missing_labels}")
    return records


def fit_object_means(records: Sequence[StateRecord]) -> Dict[str, np.ndarray]:
    by_object: Dict[str, List[np.ndarray]] = defaultdict(list)
    for record in records:
        by_object[record.subject].append(record.subject_state)
        by_object[record.reference].append(record.reference_state)
    return {
        name: np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float64)
        for name, vectors in by_object.items()
        if vectors
    }


def relation_residuals(
    records: Sequence[StateRecord],
    object_means: Mapping[str, np.ndarray],
) -> Tuple[List[StateRecord], np.ndarray, np.ndarray]:
    kept: List[StateRecord] = []
    residuals: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for record in records:
        subject_mean = object_means.get(record.subject)
        reference_mean = object_means.get(record.reference)
        if subject_mean is None or reference_mean is None:
            continue
        residual = (
            (record.subject_state - subject_mean)
            - (record.reference_state - reference_mean)
        ).astype(np.float64)
        kept.append(record)
        residuals.append(residual)
        targets.append(TARGETS[record.relation])

    if not kept:
        raise RuntimeError("No usable records remain after object-centering.")
    return kept, np.stack(residuals, axis=0), np.stack(targets, axis=0)


def fit_ridge_affine_2d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge_relative: float,
) -> Dict[str, np.ndarray | float]:
    """Fit Y ~= X W + b using dual ridge regression.

    The dual form avoids inverting a hidden_dim x hidden_dim matrix.  Ridge is
    scaled relative to the mean Gram diagonal, so its magnitude is stable
    across layers with different residual norms.
    """
    if x_train.ndim != 2 or y_train.ndim != 2 or y_train.shape[1] != 2:
        raise ValueError("Expected X=[n,d] and Y=[n,2].")
    if len(x_train) != len(y_train):
        raise ValueError("X/Y sample counts do not match.")
    if len(x_train) < 8:
        raise RuntimeError("Need at least 8 train examples for a stable 2D readout.")

    x_mean = np.mean(x_train, axis=0, keepdims=True)
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean

    gram = x_centered @ x_centered.T
    mean_diag = float(np.mean(np.diag(gram)))
    lambda_effective = max(1e-8, float(ridge_relative) * max(mean_diag, 1e-8))
    system = gram + lambda_effective * np.eye(len(x_train), dtype=np.float64)

    try:
        alpha = np.linalg.solve(system, y_centered)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(system) @ y_centered

    weight = x_centered.T @ alpha  # [hidden, 2]
    bias = (y_mean - x_mean @ weight).reshape(2)
    return {
        "weight": weight,
        "bias": bias,
        "x_mean": x_mean.reshape(-1),
        "y_mean": y_mean.reshape(2),
        "lambda_effective": lambda_effective,
    }


def predict_affine_2d(x: np.ndarray, fit: Mapping[str, np.ndarray | float]) -> np.ndarray:
    weight = np.asarray(fit["weight"], dtype=np.float64)
    bias = np.asarray(fit["bias"], dtype=np.float64)
    return x @ weight + bias


def decode_cardinal(z: np.ndarray) -> Tuple[List[str], np.ndarray]:
    target_matrix = np.stack([TARGETS[label] for label in LABELS], axis=0)
    squared_distances = np.sum((z[:, None, :] - target_matrix[None, :, :]) ** 2, axis=2)
    indices = np.argmin(squared_distances, axis=1)
    labels = [LABELS[int(index)] for index in indices]
    return labels, squared_distances


def metrics_from_predictions(
    records: Sequence[StateRecord],
    predicted_coordinates: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    labels_true = [record.relation for record in records]
    labels_pred, squared_distances = decode_cardinal(predicted_coordinates)

    correct = np.array([true == pred for true, pred in zip(labels_true, labels_pred)], dtype=np.bool_)
    target = np.stack([TARGETS[label] for label in labels_true], axis=0)
    mse = float(np.mean((predicted_coordinates - target) ** 2))

    pred_norm = np.linalg.norm(predicted_coordinates, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = np.sum(predicted_coordinates * target, axis=1) / np.maximum(pred_norm * target_norm, 1e-12)

    confusion: Dict[str, Dict[str, int]] = {
        true: {pred: 0 for pred in LABELS}
        for true in LABELS
    }
    for true, pred in zip(labels_true, labels_pred):
        confusion[true][pred] += 1

    per_relation: Dict[str, Dict[str, float | int | None]] = {}
    target_centres: Dict[str, List[float]] = {}
    for label in LABELS:
        indices = [i for i, true in enumerate(labels_true) if true == label]
        per_relation[label] = {
            "n": len(indices),
            "accuracy": float(np.mean(correct[indices])) if indices else None,
        }
        target_centres[label] = (
            np.mean(predicted_coordinates[indices], axis=0).tolist() if indices else [float("nan"), float("nan")]
        )

    summary: Dict[str, object] = {
        "accuracy": float(np.mean(correct)),
        "evaluation_n": int(len(records)),
        "coordinate_mse": mse,
        "mean_target_cosine": float(np.mean(cosine)),
        "per_relation": per_relation,
        "predicted_coordinate_class_means": target_centres,
        "confusion_matrix": confusion,
    }
    artifacts = {
        "sid": np.array([record.sid for record in records], dtype=np.int64),
        "ground_truth": np.array(labels_true, dtype=object),
        "prediction": np.array(labels_pred, dtype=object),
        "coordinates": predicted_coordinates.astype(np.float32),
        "target_coordinates": target.astype(np.float32),
        "correct": correct,
        "squared_distances_to_targets": squared_distances.astype(np.float32),
    }
    return summary, artifacts


def grouped_folds(
    records: Sequence[StateRecord],
    *,
    n_folds: int,
    split_unit: str,
    seed: int,
) -> List[List[int]]:
    if n_folds < 2:
        return []

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = record.group if split_unit == "pair" else str(record.sid)
        groups[key].append(idx)
    if len(groups) < n_folds:
        raise RuntimeError(f"Need at least {n_folds} groups for {n_folds}-fold CV; got {len(groups)}.")

    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    fold_counts: List[Counter[str]] = [Counter() for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]

    total_counts = Counter(record.relation for record in records)
    target_per_fold = {label: total_counts[label] / n_folds for label in LABELS}
    total_target = len(records) / n_folds

    for _, indices in items:
        group_counts = Counter(records[index].relation for index in indices)

        def score(fold: int) -> Tuple[float, float, int]:
            # Score the *entire* post-assignment partition.  Looking only at
            # the candidate fold would send every equally sized first group to
            # fold 0 because all folds initially have the same local score.
            relation_error = 0.0
            size_error = 0.0
            for candidate in range(n_folds):
                candidate_size = fold_sizes[candidate] + (len(indices) if candidate == fold else 0)
                size_error += abs(candidate_size - total_target)
                for label in LABELS:
                    candidate_count = fold_counts[candidate][label] + (
                        group_counts[label] if candidate == fold else 0
                    )
                    relation_error += abs(candidate_count - target_per_fold[label])
            return relation_error, size_error, fold_sizes[fold]

        chosen = min(range(n_folds), key=score)
        fold_indices[chosen].extend(indices)
        fold_counts[chosen].update(group_counts)
        fold_sizes[chosen] += len(indices)

    for fold, indices in enumerate(fold_indices):
        label_counts = Counter(records[index].relation for index in indices)
        if any(label_counts[label] == 0 for label in LABELS):
            raise RuntimeError(
                f"Fold {fold} lacks one or more labels: {dict(label_counts)}. "
                "Use fewer folds or --split-unit sample."
            )
    return fold_indices


def cross_validated_2d_readout(
    records: Sequence[StateRecord],
    *,
    n_folds: int,
    split_unit: str,
    seed: int,
    ridge_relative: float,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    folds = grouped_folds(records, n_folds=n_folds, split_unit=split_unit, seed=seed)
    all_indices = set(range(len(records)))

    prediction_by_index: Dict[int, np.ndarray] = {}
    fold_metrics: List[Dict[str, object]] = []
    dropped_by_fold: List[Dict[str, int]] = []
    lambda_values: List[float] = []

    for fold_index, test_indices_list in enumerate(folds):
        test_indices = set(test_indices_list)
        train_indices = sorted(all_indices - test_indices)
        train_records = [records[index] for index in train_indices]
        raw_test_records = [records[index] for index in sorted(test_indices)]

        object_means = fit_object_means(train_records)
        train_kept, x_train, y_train = relation_residuals(train_records, object_means)
        test_kept, x_test, _ = relation_residuals(raw_test_records, object_means)
        if not test_kept:
            raise RuntimeError(f"Fold {fold_index}: all held-out samples were dropped due to unseen objects.")

        fit = fit_ridge_affine_2d(x_train, y_train, ridge_relative=ridge_relative)
        z_test = predict_affine_2d(x_test, fit)
        metric, _ = metrics_from_predictions(test_kept, z_test)
        metric = dict(metric)
        metric.update(
            {
                "fold": int(fold_index),
                "train_n": int(len(train_kept)),
                "test_n": int(len(test_kept)),
                "dropped_test_n": int(len(raw_test_records) - len(test_kept)),
                "objects_in_train_baseline": int(len(object_means)),
                "lambda_effective": float(fit["lambda_effective"]),
            }
        )
        fold_metrics.append(metric)
        lambda_values.append(float(fit["lambda_effective"]))
        dropped_by_fold.append({"fold": fold_index, "dropped_test_n": len(raw_test_records) - len(test_kept)})

        by_sid = {record.sid: coordinate for record, coordinate in zip(test_kept, z_test)}
        for index in test_indices:
            record = records[index]
            if record.sid in by_sid:
                prediction_by_index[index] = by_sid[record.sid]

    ordered_indices = sorted(prediction_by_index)
    kept_records = [records[index] for index in ordered_indices]
    kept_coordinates = np.stack([prediction_by_index[index] for index in ordered_indices], axis=0)
    aggregate, artifacts = metrics_from_predictions(kept_records, kept_coordinates)
    aggregate = dict(aggregate)
    aggregate.update(
        {
            "decoder": "grouped_cross_validated_affine_ridge_to_cardinal_2d",
            "description": (
                "A frozen-state, supervised linear readout. For each fold, shared object baselines "
                "and the affine ridge map are fitted only on the other folds; held-out samples are then "
                "mapped to the common targets left=(-1,0), right=(1,0), under=(0,-1), on=(0,1)."
            ),
            "cv_folds": int(n_folds),
            "split_unit": split_unit,
            "fold_metrics": fold_metrics,
            "mean_lambda_effective": float(np.mean(lambda_values)),
            "dropped_by_fold": dropped_by_fold,
        }
    )
    return aggregate, artifacts


def full_fit_2d_readout(
    records: Sequence[StateRecord],
    *,
    ridge_relative: float,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray | float]]:
    object_means = fit_object_means(records)
    kept, x_all, y_all = relation_residuals(records, object_means)
    fit = fit_ridge_affine_2d(x_all, y_all, ridge_relative=ridge_relative)
    coordinates = predict_affine_2d(x_all, fit)
    summary, artifacts = metrics_from_predictions(kept, coordinates)
    summary = dict(summary)
    summary.update(
        {
            "decoder": "full_data_affine_ridge_to_cardinal_2d",
            "description": (
                "A frozen-state supervised linear alignment from high-dimensional relation residuals to the "
                "single shared cardinal 2D target system left=(-1,0), right=(1,0), under=(0,-1), on=(0,1). "
                "This is a full-data descriptive score, not a held-out estimate."
            ),
            "objects_in_shared_baseline": int(len(object_means)),
            "lambda_effective": float(fit["lambda_effective"]),
        }
    )
    extras: Dict[str, np.ndarray | float] = {
        "object_names": np.array(sorted(object_means), dtype=object),
        "object_means": np.stack([object_means[name] for name in sorted(object_means)], axis=0).astype(np.float32),
        "weight": np.asarray(fit["weight"], dtype=np.float32),
        "bias": np.asarray(fit["bias"], dtype=np.float32),
        "feature_mean": np.asarray(fit["x_mean"], dtype=np.float32),
        "target_mean": np.asarray(fit["y_mean"], dtype=np.float32),
        "lambda_effective": float(fit["lambda_effective"]),
    }
    return summary, artifacts, extras


def main() -> None:
    args = parse_args()
    if args.cv_folds < 1:
        raise ValueError("--cv-folds must be >= 1.")
    if args.ridge < 0.0:
        raise ValueError("--ridge must be non-negative.")

    layers = parse_layers(args.layers)
    input_path = Path(args.input_npz)
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "metadata": {
            "input_npz": str(input_path),
            "layers_zero_based": layers,
            "targets": {label: TARGETS[label].tolist() for label in LABELS},
            "ridge_relative": float(args.ridge),
            "fit_all": bool(args.fit_all),
            "cv_folds": int(args.cv_folds),
            "split_unit": args.split_unit,
            "seed": int(args.seed),
            "important_note": (
                "No LLaVA weights are trained or modified. The procedure fits a supervised affine "
                "linear readout over frozen relation residuals. High full-fit accuracy means a common "
                "2D cardinal coordinate is linearly recoverable; grouped CV is the generalization estimate."
            ),
        },
        "results": {},
    }
    arrays: Dict[str, np.ndarray] = {}

    print("Joint 2D cardinal readout")
    print(f"Input states: {input_path}")
    print(f"Layers (zero-based): {layers}")
    print(f"Ridge relative strength: {args.ridge:g}")

    for layer in layers:
        records = load_records(input_path, layer)
        layer_result: Dict[str, object] = {
            "input_samples": int(len(records)),
            "relation_counts": dict(Counter(record.relation for record in records)),
            "unique_unordered_pairs": int(len({record.group for record in records})),
            "unique_objects": int(len({record.subject for record in records} | {record.reference for record in records})),
        }
        print(f"\nL{layer}: n={len(records)}, pairs={layer_result['unique_unordered_pairs']}, objects={layer_result['unique_objects']}")

        if args.fit_all:
            full_summary, full_artifacts, full_extras = full_fit_2d_readout(records, ridge_relative=args.ridge)
            layer_result["full_data_linear_2d"] = full_summary
            print(
                "  full-fit learned-2D acc={acc:.3f} | coordinate MSE={mse:.3f} | "
                "mean target cosine={cos:.3f} (n={n}; lambda={lam:.3e})".format(
                    acc=float(full_summary["accuracy"]),
                    mse=float(full_summary["coordinate_mse"]),
                    cos=float(full_summary["mean_target_cosine"]),
                    n=int(full_summary["evaluation_n"]),
                    lam=float(full_summary["lambda_effective"]),
                )
            )
            for name, value in full_artifacts.items():
                arrays[f"layer_{layer}_full_{name}"] = value
            for name, value in full_extras.items():
                if isinstance(value, np.ndarray):
                    arrays[f"layer_{layer}_full_{name}"] = value
                else:
                    arrays[f"layer_{layer}_full_{name}"] = np.array(value, dtype=np.float64)

        if args.cv_folds >= 2:
            cv_summary, cv_artifacts = cross_validated_2d_readout(
                records,
                n_folds=args.cv_folds,
                split_unit=args.split_unit,
                seed=args.seed,
                ridge_relative=args.ridge,
            )
            layer_result["grouped_cv_linear_2d"] = cv_summary
            print(
                "  {folds}-fold {unit}-CV learned-2D acc={acc:.3f} | coordinate MSE={mse:.3f} | "
                "mean target cosine={cos:.3f} (n={n})".format(
                    folds=args.cv_folds,
                    unit=args.split_unit,
                    acc=float(cv_summary["accuracy"]),
                    mse=float(cv_summary["coordinate_mse"]),
                    cos=float(cv_summary["mean_target_cosine"]),
                    n=int(cv_summary["evaluation_n"]),
                )
            )
            for name, value in cv_artifacts.items():
                arrays[f"layer_{layer}_cv_{name}"] = value

        payload["results"][str(layer)] = layer_result

    json_path = output_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    npz_path = output_base.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    print(f"\nSaved summary: {json_path}")
    print(f"Saved learned 2D coordinates/readouts: {npz_path}")


if __name__ == "__main__":
    main()
