#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled_A raw relation-direction probe over saved VLM hidden states.

This script intentionally does NOT assume a Cartesian coordinate system:
  * left/right are not required to be opposite;
  * on/under are not required to be opposite;
  * no relation is forced onto a 1D or 2D axis.

For each frozen decoder layer and each grouped-CV fold, it builds one direction
for every answer relation from training data only:

    r_i = h_L(subject_i) - h_L(reference_i)
    g   = mean_{train}(r_i)
    d_c = normalize(mean_{train, y_i=c}(r_i - g))

A held-out example is classified with:

    y_hat = argmax_c cosine(r_i - g, d_c)

Thus this is a lightweight supervised relation-codebook probe. It uses relation
labels only to form the TRAIN-FOLD directions. The VLM is never fine-tuned, and
test labels are used only after prediction to compute accuracy.

It reads the NPZ emitted by:
  run_llava15_controlledA_spatial_id_probe_v6_affine.py

Run from the AdaptVis repository root:

python3 analyze_controlledA_relation_direction_probe.py \
  --input-npz output/llava15_controlledA_spatial_id_affine_L13_L16_L31.npz \
  --layers 13,16,31 \
  --cv-folds 5 \
  --split-unit pair \
  --label-shuffle-repeats 50 \
  --output output/llava15_controlledA_relation_direction
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


LABELS: Tuple[str, ...] = ("left", "right", "on", "under")
EPS = 1e-12


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
        # Do not let the four layouts of one unordered object pair cross folds.
        return " || ".join(sorted((self.subject, self.reference)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grouped-CV raw relation-direction probe for Controlled_A saved VLM states."
    )
    parser.add_argument(
        "--input-npz",
        required=True,
        help="NPZ emitted by run_llava15_controlledA_spatial_id_probe_v6_affine.py",
    )
    parser.add_argument(
        "--layers",
        default="13,16,31",
        help="Comma-separated zero-based decoder blocks stored in the NPZ.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Grouped CV folds. Default: 5.",
    )
    parser.add_argument(
        "--split-unit",
        choices=("pair", "sample"),
        default="pair",
        help="Use pair to keep every layout of the same unordered object pair in one fold.",
    )
    parser.add_argument(
        "--feature-centering",
        choices=("global", "none"),
        default="global",
        help=(
            "Subtract the train-fold global mean from all relation residuals before cosine scoring. "
            "Default: global. This is label-free and does not impose pairwise opposition."
        ),
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Also compute an in-sample descriptive result. Do not use it as a generalization score.",
    )
    parser.add_argument(
        "--label-shuffle-repeats",
        type=int,
        default=50,
        help=(
            "Number of training-label shuffle controls. In each CV fold the train labels are permuted, "
            "directions are rebuilt, and real held-out labels are scored. Set 0 to skip. Default: 50."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        default="output/llava15_controlledA_relation_direction",
        help="Output base path; .json and .npz are appended.",
    )
    return parser.parse_args()


def parse_layers(text: str) -> List[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or min(values) < 0:
        raise ValueError(f"Invalid --layers: {text!r}")
    return values


def load_records(npz_path: Path, layer: int) -> List[StateRecord]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        required = (
            "sid",
            "relation",
            "subject_name",
            "reference_name",
            f"layer_{layer}_subject",
            f"layer_{layer}_reference",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Layer {layer} unavailable; missing keys: {missing}")

        sid = np.asarray(data["sid"], dtype=np.int64)
        relation = np.asarray(data["relation"], dtype=object)
        subject = np.asarray(data["subject_name"], dtype=object)
        reference = np.asarray(data["reference_name"], dtype=object)
        subject_state = np.asarray(data[f"layer_{layer}_subject"], dtype=np.float64)
        reference_state = np.asarray(data[f"layer_{layer}_reference"], dtype=np.float64)

    n = len(sid)
    if not all(len(x) == n for x in (relation, subject, reference, subject_state, reference_state)):
        raise RuntimeError(f"Layer {layer}: inconsistent NPZ array sizes.")

    rows: List[StateRecord] = []
    for i in range(n):
        label = str(relation[i]).strip().lower()
        if label not in LABELS:
            continue
        rows.append(
            StateRecord(
                sid=int(sid[i]),
                relation=label,
                subject=str(subject[i]),
                reference=str(reference[i]),
                subject_state=subject_state[i].copy(),
                reference_state=reference_state[i].copy(),
            )
        )
    counts = Counter(row.relation for row in rows)
    missing_labels = [label for label in LABELS if counts[label] == 0]
    if missing_labels:
        raise RuntimeError(f"Layer {layer}: missing labels {missing_labels}")
    return rows


def raw_relation_features(records: Sequence[StateRecord]) -> np.ndarray:
    # No per-object baseline is used. This keeps all held-out samples evaluable,
    # including pairs whose object categories never occurred in a train fold.
    return np.stack(
        [(row.subject_state - row.reference_state).astype(np.float64) for row in records],
        axis=0,
    )


def norm_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def grouped_folds(
    records: Sequence[StateRecord],
    *,
    n_folds: int,
    split_unit: str,
    seed: int,
) -> List[List[int]]:
    if n_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        key = row.group if split_unit == "pair" else str(row.sid)
        groups[key].append(index)
    if len(groups) < n_folds:
        raise RuntimeError(f"Need >= {n_folds} groups; got {len(groups)}")

    # Greedy balanced assignment by per-class counts. With Controlled_A each
    # object pair normally contributes one example for each of the four labels.
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    totals = Counter(row.relation for row in records)
    wanted = {label: totals[label] / n_folds for label in LABELS}
    wanted_size = len(records) / n_folds
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    fold_counts: List[Counter[str]] = [Counter() for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]

    for _, indices in items:
        group_counts = Counter(records[i].relation for i in indices)

        def assignment_score(chosen_fold: int) -> Tuple[float, float, int]:
            relation_error = 0.0
            size_error = 0.0
            for candidate in range(n_folds):
                candidate_size = fold_sizes[candidate] + (len(indices) if candidate == chosen_fold else 0)
                size_error += abs(candidate_size - wanted_size)
                for label in LABELS:
                    candidate_count = fold_counts[candidate][label] + (
                        group_counts[label] if candidate == chosen_fold else 0
                    )
                    relation_error += abs(candidate_count - wanted[label])
            return relation_error, size_error, fold_sizes[chosen_fold]

        target = min(range(n_folds), key=assignment_score)
        folds[target].extend(indices)
        fold_counts[target].update(group_counts)
        fold_sizes[target] += len(indices)

    for fold_id, indices in enumerate(folds):
        counts = Counter(records[i].relation for i in indices)
        missing = [label for label in LABELS if counts[label] == 0]
        if missing:
            raise RuntimeError(f"Fold {fold_id} has no {missing}; reduce --cv-folds.")
    return folds


def fit_direction_codebook(
    x_train: np.ndarray,
    y_train: Sequence[str],
    *,
    feature_centering: str,
) -> Dict[str, object]:
    if feature_centering == "global":
        center = np.mean(x_train, axis=0, keepdims=True)
    elif feature_centering == "none":
        center = np.zeros((1, x_train.shape[1]), dtype=np.float64)
    else:
        raise ValueError(f"Unknown centering: {feature_centering}")

    centered = x_train - center
    directions: List[np.ndarray] = []
    relation_means: List[np.ndarray] = []
    for label in LABELS:
        idx = [i for i, value in enumerate(y_train) if value == label]
        if not idx:
            raise RuntimeError(f"Training fold lacks relation {label}")
        mean = np.mean(centered[idx], axis=0)
        relation_means.append(mean)
        directions.append(mean / max(float(np.linalg.norm(mean)), EPS))

    return {
        "center": center.reshape(-1),
        "directions": np.stack(directions, axis=0),
        "relation_means": np.stack(relation_means, axis=0),
    }


def score_direction_codebook(x: np.ndarray, fit: Mapping[str, object]) -> Tuple[List[str], np.ndarray]:
    center = np.asarray(fit["center"], dtype=np.float64)
    directions = np.asarray(fit["directions"], dtype=np.float64)
    query = norm_rows(x - center[None, :])
    scores = query @ directions.T
    predicted = [LABELS[int(i)] for i in np.argmax(scores, axis=1)]
    return predicted, scores


def metrics(
    records: Sequence[StateRecord],
    predicted: Sequence[str],
    scores: np.ndarray,
) -> Dict[str, object]:
    truth = [row.relation for row in records]
    correct = np.asarray([a == b for a, b in zip(truth, predicted)], dtype=bool)

    confusion: Dict[str, Dict[str, int]] = {
        true: {pred: 0 for pred in LABELS} for true in LABELS
    }
    for true, pred in zip(truth, predicted):
        confusion[true][pred] += 1

    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    per_relation = {}
    for label in LABELS:
        idx = np.asarray([i for i, value in enumerate(truth) if value == label], dtype=np.int64)
        per_relation[label] = {
            "n": int(len(idx)),
            "accuracy": float(np.mean(correct[idx])) if len(idx) else None,
            "mean_margin": float(np.mean(margins[idx])) if len(idx) else None,
        }

    return {
        "accuracy": float(np.mean(correct)),
        "evaluation_n": int(len(records)),
        "mean_top1_margin": float(np.mean(margins)),
        "per_relation": per_relation,
        "confusion_matrix": confusion,
    }


def aggregate_predictions(
    records: Sequence[StateRecord],
    predictions_by_index: Mapping[int, str],
    scores_by_index: Mapping[int, np.ndarray],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    ordered = sorted(predictions_by_index)
    kept = [records[i] for i in ordered]
    predicted = [predictions_by_index[i] for i in ordered]
    scores = np.stack([scores_by_index[i] for i in ordered], axis=0)
    summary = metrics(kept, predicted, scores)
    artifacts = {
        "sid": np.asarray([row.sid for row in kept], dtype=np.int64),
        "group": np.asarray([row.group for row in kept], dtype=object),
        "ground_truth": np.asarray([row.relation for row in kept], dtype=object),
        "prediction": np.asarray(predicted, dtype=object),
        "scores": scores.astype(np.float32),
        "correct": np.asarray([row.relation == pred for row, pred in zip(kept, predicted)], dtype=bool),
    }
    return summary, artifacts


def run_grouped_cv(
    records: Sequence[StateRecord],
    x: np.ndarray,
    *,
    folds: Sequence[Sequence[int]],
    feature_centering: str,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], List[Dict[str, object]]]:
    all_indices = set(range(len(records)))
    prediction_by_index: Dict[int, str] = {}
    scores_by_index: Dict[int, np.ndarray] = {}
    fold_rows: List[Dict[str, object]] = []

    for fold_id, test_indices_raw in enumerate(folds):
        test_indices = sorted(test_indices_raw)
        train_indices = sorted(all_indices - set(test_indices))
        x_train = x[train_indices]
        y_train = [records[i].relation for i in train_indices]
        x_test = x[test_indices]

        fit = fit_direction_codebook(x_train, y_train, feature_centering=feature_centering)
        predicted, scores = score_direction_codebook(x_test, fit)
        fold_metric = metrics([records[i] for i in test_indices], predicted, scores)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "train_n": int(len(train_indices)),
                "test_n": int(len(test_indices)),
                **fold_metric,
            }
        )
        for index, pred, score in zip(test_indices, predicted, scores):
            prediction_by_index[index] = pred
            scores_by_index[index] = score

    summary, artifacts = aggregate_predictions(records, prediction_by_index, scores_by_index)
    summary = dict(summary)
    summary.update(
        {
            "decoder": "raw_pair_difference_cosine_relation_direction_codebook",
            "description": (
                "For each train fold, fit one independent cosine direction per answer relation from "
                "r=h(subject)-h(reference). No Cartesian coordinates, opposition, per-object baseline, "
                "or learned projection are imposed."
            ),
            "feature_centering": feature_centering,
            "fold_metrics": fold_rows,
        }
    )
    return summary, artifacts, fold_rows


def shuffled_label_control(
    records: Sequence[StateRecord],
    x: np.ndarray,
    *,
    folds: Sequence[Sequence[int]],
    feature_centering: str,
    repeats: int,
    seed: int,
) -> Dict[str, object]:
    if repeats <= 0:
        return {"repeats": 0}

    all_indices = set(range(len(records)))
    accuracies: List[float] = []
    for repeat in range(repeats):
        prediction_by_index: Dict[int, str] = {}
        scores_by_index: Dict[int, np.ndarray] = {}
        for fold_id, test_indices_raw in enumerate(folds):
            test_indices = sorted(test_indices_raw)
            train_indices = sorted(all_indices - set(test_indices))
            y_train = [records[i].relation for i in train_indices]
            shuffled = list(y_train)
            rng = random.Random(seed + 100_003 * repeat + 997 * fold_id)
            rng.shuffle(shuffled)
            fit = fit_direction_codebook(
                x[train_indices],
                shuffled,
                feature_centering=feature_centering,
            )
            predicted, scores = score_direction_codebook(x[test_indices], fit)
            for index, pred, score in zip(test_indices, predicted, scores):
                prediction_by_index[index] = pred
                scores_by_index[index] = score
        metric, _ = aggregate_predictions(records, prediction_by_index, scores_by_index)
        accuracies.append(float(metric["accuracy"]))

    values = np.asarray(accuracies, dtype=np.float64)
    return {
        "repeats": int(repeats),
        "chance_accuracy": 1.0 / len(LABELS),
        "mean_accuracy": float(np.mean(values)),
        "std_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min_accuracy": float(np.min(values)),
        "max_accuracy": float(np.max(values)),
        "all_accuracies": values.tolist(),
        "description": (
            "Negative control: train-fold relation labels are randomly permuted before directions are fitted. "
            "Held-out labels remain intact and are used only for scoring."
        ),
    }


def full_data_summary(
    records: Sequence[StateRecord],
    x: np.ndarray,
    *,
    feature_centering: str,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, np.ndarray]]:
    y = [row.relation for row in records]
    fit = fit_direction_codebook(x, y, feature_centering=feature_centering)
    predicted, scores = score_direction_codebook(x, fit)
    summary = metrics(records, predicted, scores)
    summary = dict(summary)
    summary.update(
        {
            "description": "In-sample diagnostic only; directions and predictions use the same records.",
            "feature_centering": feature_centering,
        }
    )
    directions = np.asarray(fit["directions"], dtype=np.float64)
    gram = directions @ directions.T
    direction_norms = np.linalg.norm(np.asarray(fit["relation_means"]), axis=1)
    geometry = {
        "labels": list(LABELS),
        "direction_cosine_gram": gram.tolist(),
        "uncentred_relation_mean_norms": direction_norms.tolist(),
        "interpretation": (
            "This Gram matrix is descriptive only. Cosine near -1 indicates a naturally opposite pair; "
            "cosine near 0 indicates weak alignment; neither relation is imposed by this probe."
        ),
    }
    artifacts = {
        "direction_labels": np.asarray(LABELS, dtype=object),
        "directions": directions.astype(np.float32),
        "direction_cosine_gram": gram.astype(np.float32),
    }
    return summary, geometry, artifacts


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if args.label_shuffle_repeats < 0:
        raise ValueError("--label-shuffle-repeats must be >= 0")

    input_path = Path(args.input_npz)
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    payload: Dict[str, object] = {
        "metadata": {
            "input_npz": str(input_path),
            "layers_zero_based": layers,
            "relations": list(LABELS),
            "cv_folds": int(args.cv_folds),
            "split_unit": args.split_unit,
            "feature_centering": args.feature_centering,
            "label_shuffle_repeats": int(args.label_shuffle_repeats),
            "seed": int(args.seed),
            "important_note": (
                "The VLM is frozen. This is a supervised train-fold relation-direction codebook probe, "
                "not native generation accuracy. It does not assume left/right or on/under are opposites."
            ),
        },
        "results": {},
    }
    arrays: Dict[str, np.ndarray] = {}

    print("Raw relation-direction probe (no Cartesian or opposition constraints)")
    print(f"Input states: {input_path}")
    print(f"Layers (zero-based): {layers}")
    print(f"CV: {args.cv_folds}-fold {args.split_unit}-grouped | centering={args.feature_centering}")

    for layer in layers:
        records = load_records(input_path, layer)
        x = raw_relation_features(records)
        folds = grouped_folds(records, n_folds=args.cv_folds, split_unit=args.split_unit, seed=args.seed)
        cv_summary, cv_artifacts, _ = run_grouped_cv(
            records,
            x,
            folds=folds,
            feature_centering=args.feature_centering,
        )
        shuffle = shuffled_label_control(
            records,
            x,
            folds=folds,
            feature_centering=args.feature_centering,
            repeats=args.label_shuffle_repeats,
            seed=args.seed,
        )

        layer_result: Dict[str, object] = {
            "input_samples": int(len(records)),
            "relation_counts": dict(Counter(row.relation for row in records)),
            "unique_unordered_pairs": int(len({row.group for row in records})),
            "unique_objects": int(len({row.subject for row in records} | {row.reference for row in records})),
            "pair_cv_relation_direction": cv_summary,
            "train_label_shuffle_control": shuffle,
        }

        print(f"\nL{layer}: n={len(records)}, pairs={layer_result['unique_unordered_pairs']}, objects={layer_result['unique_objects']}")
        print(
            "  pair-CV relation-direction acc={acc:.3f} | mean margin={margin:.3f} | n={n}".format(
                acc=float(cv_summary["accuracy"]),
                margin=float(cv_summary["mean_top1_margin"]),
                n=int(cv_summary["evaluation_n"]),
            )
        )
        if args.label_shuffle_repeats > 0:
            print(
                "  train-label shuffle control={mean:.3f} ± {std:.3f} "
                "(chance={chance:.3f}; repeats={repeats})".format(
                    mean=float(shuffle["mean_accuracy"]),
                    std=float(shuffle["std_accuracy"]),
                    chance=float(shuffle["chance_accuracy"]),
                    repeats=int(shuffle["repeats"]),
                )
            )

        if args.fit_all:
            full_summary, geometry, full_artifacts = full_data_summary(
                records,
                x,
                feature_centering=args.feature_centering,
            )
            layer_result["full_data_diagnostic"] = full_summary
            layer_result["full_data_direction_geometry"] = geometry
            print(
                "  full-data diagnostic acc={acc:.3f} (do not use as held-out result)".format(
                    acc=float(full_summary["accuracy"])
                )
            )
            print("  relation direction cosine Gram matrix (label order: left,right,on,under):")
            print(np.array2string(np.asarray(geometry["direction_cosine_gram"]), precision=3, suppress_small=True))
            for name, value in full_artifacts.items():
                arrays[f"layer_{layer}_full_{name}"] = value

        for name, value in cv_artifacts.items():
            arrays[f"layer_{layer}_cv_{name}"] = value
        payload["results"][f"layer_{layer}"] = layer_result

    json_path = output_base.with_suffix(".json")
    npz_path = output_base.with_suffix(".npz")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(npz_path, **arrays)
    print(f"\nSaved summary: {json_path}")
    print(f"Saved CV predictions / directions: {npz_path}")


if __name__ == "__main__":
    main()
