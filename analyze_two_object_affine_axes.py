#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/test probe for spatial relation vectors saved by
extract_two_object_relation_states.py.

Default mode is the four-direction cosine codebook required by the custom
VG 800-sample dataset:

    r = h(subject) - h(reference)

For every selected decoder layer and every random split:
  1. Stratify each relation independently.
  2. Use --train-ratio of each relation for training (default: 0.30).
  3. Compute one normalized direction vector for each relation from training.
  4. Classify each test vector by maximum cosine similarity.

With 200 samples per relation and --train-ratio 0.30, every split contains:
  train: 60 x 4 = 240
  test:  140 x 4 = 560

The old opposing-affine-axis analysis remains available through:

    --probe-mode affine_axes

Expected NPZ fields:
  relation
  relation_vectors          [N, L, H]
  decoder_block_index       [L]
Optional fields:
  image_id, subject, reference, metadata_json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

EPS = 1e-12

AXIS_CANDIDATES: List[Tuple[str, str, str]] = [
    ("horizontal", "left", "right"),
    ("vertical", "under", "on"),
    ("depth", "behind", "in_front"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument(
        "--probe-mode",
        choices=("four_direction", "affine_axes"),
        default="four_direction",
        help="four_direction: one cosine prototype per relation; "
        "affine_axes: legacy opposing-axis probe.",
    )
    parser.add_argument(
        "--relations",
        default="left,right,on,under",
        help="Comma-separated relation order used by four_direction mode.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.30,
        help="Fraction of every relation class used for training.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of independent stratified 30/70 splits.",
    )
    parser.add_argument(
        "--feature-centering",
        choices=("global", "none"),
        default="global",
        help="Subtract the training-set global mean before fitting directions.",
    )
    parser.add_argument(
        "--layers",
        default="auto",
        help="auto/all or comma-separated decoder block indices, e.g. 6,9,12.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--label-shuffle-repeats",
        type=int,
        default=50,
        help="Training-label shuffle repetitions per layer and split.",
    )
    parser.add_argument("--no-label-shuffle", action="store_true")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path or output prefix without .json.",
    )
    return parser.parse_args()


def normalize_relation(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._,")

    aliases = {
        "left_of": "left",
        "to_left_of": "left",
        "to_the_left_of": "left",
        "right_of": "right",
        "to_right_of": "right",
        "to_the_right_of": "right",
        "above": "on",
        "above_of": "on",
        "top": "on",
        "top_of": "on",
        "on_top": "on",
        "on_top_of": "on",
        "below": "under",
        "below_of": "under",
        "bottom": "under",
        "beneath": "under",
        "underneath": "under",
        "infront": "in_front",
        "front": "in_front",
        "front_of": "in_front",
        "in_front_of": "in_front",
        "back": "behind",
        "back_of": "behind",
    }
    return aliases.get(text, text)


def parse_layers(text: str, available: Sequence[int]) -> List[int]:
    if text.strip().lower() in {"auto", "all"}:
        return list(available)

    requested = sorted(
        {
            int(token.strip().lstrip("Ll"))
            for token in text.split(",")
            if token.strip()
        }
    )
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise RuntimeError(
            f"Requested decoder blocks are absent: {missing}; available={list(available)}"
        )
    return requested


def json_value(value: Any) -> Any:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPS)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= EPS:
        raise RuntimeError("Encountered a near-zero direction vector.")
    return vector / norm


def load_input(path: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[int]]:
    with np.load(path, allow_pickle=True) as loaded:
        keys = set(loaded.files)
        required = {"relation", "relation_vectors", "decoder_block_index"}
        missing = sorted(required - keys)
        if missing:
            raise RuntimeError(f"Missing NPZ fields {missing}; available={sorted(keys)}")

        labels = np.asarray(
            [normalize_relation(x) for x in loaded["relation"].tolist()],
            dtype=object,
        )
        vectors = np.asarray(loaded["relation_vectors"], dtype=np.float32)
        blocks = [int(x) for x in loaded["decoder_block_index"].tolist()]

        metadata: Dict[str, Any] = {}
        if "metadata_json" in keys:
            try:
                metadata = json.loads(str(loaded["metadata_json"].item()))
            except Exception:
                metadata = {"metadata_json_parse_error": True}

    if vectors.ndim != 3:
        raise RuntimeError(
            f"Expected relation_vectors with shape [N,L,H], got {vectors.shape}"
        )
    if vectors.shape[0] != len(labels):
        raise RuntimeError(
            f"Sample count mismatch: labels={len(labels)}, vectors={vectors.shape[0]}"
        )
    if vectors.shape[1] != len(blocks):
        raise RuntimeError(
            f"Layer count mismatch: vectors L={vectors.shape[1]}, blocks={len(blocks)}"
        )
    if not np.all(np.isfinite(vectors)):
        bad = int(np.size(vectors) - np.isfinite(vectors).sum())
        raise RuntimeError(f"relation_vectors contains {bad} non-finite values.")

    return metadata, labels, vectors, blocks


def stratified_split(
    labels: np.ndarray,
    relation_order: Sequence[str],
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """Exact per-class stratified sample split.

    For 200 samples and train_ratio=0.30 this chooses exactly 60 train samples
    and 140 test samples for that class.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train-ratio must be between 0 and 1, got {train_ratio}")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []
    train_counts: Dict[str, int] = {}
    test_counts: Dict[str, int] = {}

    for relation in relation_order:
        indices = np.flatnonzero(labels == relation)
        if len(indices) < 2:
            raise RuntimeError(
                f"Relation {relation!r} has only {len(indices)} sample(s); need >=2."
            )

        indices = rng.permutation(indices)
        n_train = int(round(len(indices) * train_ratio))
        n_train = min(max(n_train, 1), len(indices) - 1)

        train_part = indices[:n_train]
        test_part = indices[n_train:]

        train_indices.extend(train_part.tolist())
        test_indices.extend(test_part.tolist())
        train_counts[relation] = int(len(train_part))
        test_counts[relation] = int(len(test_part))

    train_array = np.asarray(rng.permutation(train_indices), dtype=np.int64)
    test_array = np.asarray(rng.permutation(test_indices), dtype=np.int64)

    overlap = np.intersect1d(train_array, test_array)
    if len(overlap):
        raise RuntimeError(f"Internal split error: {len(overlap)} overlapping samples.")

    return train_array, test_array, train_counts, test_counts


def confusion_matrix(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    n_classes: int,
) -> List[List[int]]:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_value, pred_value in zip(true_indices.tolist(), predicted_indices.tolist()):
        matrix[int(true_value), int(pred_value)] += 1
    return matrix.tolist()


def fit_direction_codebook(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    relation_order: Sequence[str],
    feature_centering: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_centering == "global":
        center = np.mean(train_vectors, axis=0)
    else:
        center = np.zeros(train_vectors.shape[1], dtype=np.float32)

    centered = train_vectors - center[None, :]
    directions: List[np.ndarray] = []
    raw_means: List[np.ndarray] = []

    for relation in relation_order:
        mask = train_labels == relation
        if not np.any(mask):
            raise RuntimeError(f"Training split is missing relation {relation!r}.")
        mean_vector = np.mean(centered[mask], axis=0)
        raw_means.append(mean_vector)
        directions.append(unit(mean_vector))

    return center, np.stack(directions), np.stack(raw_means)


def evaluate_direction_codebook(
    test_vectors: np.ndarray,
    test_labels: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
    relation_order: Sequence[str],
) -> Dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(relation_order)}
    true = np.asarray([label_to_index[x] for x in test_labels], dtype=np.int64)

    scores = normalize_rows(test_vectors - center[None, :]) @ directions.T
    predicted = np.argmax(scores, axis=1)
    correct = predicted == true

    true_scores = scores[np.arange(len(true)), true]
    other_scores = scores.copy()
    other_scores[np.arange(len(true)), true] = -np.inf
    signed_margin = true_scores - np.max(other_scores, axis=1)

    sorted_scores = np.sort(scores, axis=1)
    top1_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    per_class: Dict[str, Any] = {}
    for relation, class_index in label_to_index.items():
        mask = true == class_index
        per_class[relation] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(predicted[mask] == class_index)),
            "mean_true_cosine": float(np.mean(true_scores[mask])),
            "mean_signed_margin": float(np.mean(signed_margin[mask])),
        }

    return {
        "accuracy": float(np.mean(correct)),
        "n": int(len(true)),
        "mean_signed_margin": float(np.mean(signed_margin)),
        "mean_top1_margin": float(np.mean(top1_margin)),
        "per_class": per_class,
        "relation_order": list(relation_order),
        "confusion_matrix": confusion_matrix(true, predicted, len(relation_order)),
        "direction_cosine_gram": (directions @ directions.T).tolist(),
    }


def available_axes(labels: Sequence[str]) -> List[Tuple[str, str, str]]:
    values = set(labels)
    return [
        axis
        for axis in AXIS_CANDIDATES
        if axis[1] in values and axis[2] in values
    ]


def fit_affine_axes(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    axes: Sequence[Tuple[str, str, str]],
) -> Dict[str, Dict[str, Any]]:
    fitted: Dict[str, Dict[str, Any]] = {}

    for name, negative, positive in axes:
        negative_vectors = train_vectors[train_labels == negative]
        positive_vectors = train_vectors[train_labels == positive]
        if len(negative_vectors) == 0 or len(positive_vectors) == 0:
            raise RuntimeError(
                f"Training split is missing {negative}/{positive} for {name}."
            )

        mean_negative = np.mean(negative_vectors, axis=0)
        mean_positive = np.mean(positive_vectors, axis=0)
        direction = unit(mean_positive - mean_negative)
        center = 0.5 * (mean_positive + mean_negative)

        axis_train = train_vectors[
            np.isin(train_labels, [negative, positive])
        ]
        delta = axis_train - center[None, :]
        projection = delta @ direction
        residual = delta - projection[:, None] * direction[None, :]
        residual_scale = float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
        )

        fitted[name] = {
            "negative": negative,
            "positive": positive,
            "direction": direction,
            "center": center,
            "residual_scale": max(residual_scale, 1e-6),
        }

    return fitted


def evaluate_affine_axes(
    test_vectors: np.ndarray,
    test_labels: np.ndarray,
    fitted: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    axis_names = list(fitted)
    family_distances: List[np.ndarray] = []
    family_relations: List[np.ndarray] = []
    projections: Dict[str, np.ndarray] = {}

    for name in axis_names:
        params = fitted[name]
        delta = test_vectors - params["center"][None, :]
        projection = delta @ params["direction"]
        residual = delta - projection[:, None] * params["direction"][None, :]
        distance = (
            np.sqrt(np.sum(residual * residual, axis=1))
            / float(params["residual_scale"])
        )
        relation = np.where(
            projection >= 0.0,
            params["positive"],
            params["negative"],
        )
        family_distances.append(distance)
        family_relations.append(relation)
        projections[name] = projection

    distance_matrix = np.stack(family_distances, axis=1)
    selected_axis = np.argmin(distance_matrix, axis=1)
    predicted = np.asarray(
        [family_relations[axis_index][i] for i, axis_index in enumerate(selected_axis)],
        dtype=object,
    )

    relation_order = sorted(
        {
            str(params[side])
            for params in fitted.values()
            for side in ("negative", "positive")
        }
    )
    label_to_index = {label: i for i, label in enumerate(relation_order)}
    true_index = np.asarray([label_to_index[x] for x in test_labels], dtype=np.int64)
    pred_index = np.asarray([label_to_index[x] for x in predicted], dtype=np.int64)

    per_axis: Dict[str, Any] = {}
    for name, params in fitted.items():
        negative = str(params["negative"])
        positive = str(params["positive"])
        mask = np.isin(test_labels, [negative, positive])
        expected_positive = test_labels[mask] == positive
        predicted_positive = projections[name][mask] >= 0.0
        per_axis[name] = {
            "negative": negative,
            "positive": positive,
            "n": int(mask.sum()),
            "sign_accuracy": float(np.mean(expected_positive == predicted_positive)),
        }

    directions = np.stack([fitted[name]["direction"] for name in axis_names])
    return {
        "accuracy": float(np.mean(predicted == test_labels)),
        "n": int(len(test_labels)),
        "per_axis_sign_accuracy": per_axis,
        "relation_order": relation_order,
        "confusion_matrix": confusion_matrix(
            true_index, pred_index, len(relation_order)
        ),
        "axis_order": axis_names,
        "axis_cosine_gram": (directions @ directions.T).tolist(),
    }


def summarize_scalar(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.input_npz)
    if not source.exists():
        raise SystemExit(f"Missing input NPZ: {source}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    metadata, all_labels, all_vectors, block_ids = load_input(source)
    selected_blocks = parse_layers(args.layers, block_ids)

    requested_relations = [
        normalize_relation(token)
        for token in args.relations.split(",")
        if token.strip()
    ]
    if len(set(requested_relations)) != len(requested_relations):
        raise SystemExit(
            f"Duplicate normalized relations in --relations: {requested_relations}"
        )

    if args.probe_mode == "four_direction":
        relation_order = requested_relations
    else:
        axes = available_axes(all_labels.tolist())
        if not axes:
            raise SystemExit(
                "No complete opposing relation pair found. "
                f"Parsed labels={dict(Counter(all_labels.tolist()))}"
            )
        relation_order = []
        for _, negative, positive in axes:
            relation_order.extend([negative, positive])
        relation_order = list(dict.fromkeys(relation_order))

    keep_mask = np.isin(all_labels, relation_order)
    labels = all_labels[keep_mask]
    vectors = all_vectors[keep_mask]

    counts = Counter(labels.tolist())
    missing_relations = [
        relation for relation in relation_order if counts.get(relation, 0) == 0
    ]
    if missing_relations:
        raise SystemExit(
            f"Missing requested relations {missing_relations}; counts={dict(counts)}"
        )

    # Generate splits once and reuse them for every decoder layer.
    splits: List[Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]] = []
    for repeat in range(args.repeats):
        splits.append(
            stratified_split(
                labels,
                relation_order,
                args.train_ratio,
                args.seed + repeat * 1009,
            )
        )

    print(
        f"Input={source} | mode={args.probe_mode} | n={len(labels)} | "
        f"counts={dict(counts)}"
    )
    print(
        f"Split: train_ratio={args.train_ratio:.2f}, "
        f"test_ratio={1.0 - args.train_ratio:.2f}, repeats={args.repeats}"
    )
    print(f"Decoder blocks: {selected_blocks}")
    first_split = splits[0]
    print(
        f"First split: train={len(first_split[0])} {first_split[2]} | "
        f"test={len(first_split[1])} {first_split[3]}"
    )

    summary: Dict[str, Any] = {
        "input_npz": str(source),
        "metadata": metadata,
        "probe_mode": args.probe_mode,
        "relations": relation_order,
        "relation_counts": dict(counts),
        "train_ratio": float(args.train_ratio),
        "test_ratio": float(1.0 - args.train_ratio),
        "repeats": int(args.repeats),
        "feature_centering": args.feature_centering,
        "seed": int(args.seed),
        "layers": {},
    }

    if args.probe_mode == "affine_axes":
        summary["axes"] = [
            {"name": name, "negative": negative, "positive": positive}
            for name, negative, positive in axes
        ]

    block_to_position = {block: i for i, block in enumerate(block_ids)}
    shuffle_rng = np.random.default_rng(args.seed + 99991)

    for block in selected_blocks:
        layer_vectors = vectors[:, block_to_position[block], :]
        repeat_results: List[Dict[str, Any]] = []

        for repeat, (train_idx, test_idx, train_counts, test_counts) in enumerate(splits):
            train_vectors = layer_vectors[train_idx]
            test_vectors = layer_vectors[test_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            if args.probe_mode == "four_direction":
                center, directions, raw_means = fit_direction_codebook(
                    train_vectors,
                    train_labels,
                    relation_order,
                    args.feature_centering,
                )
                result = evaluate_direction_codebook(
                    test_vectors,
                    test_labels,
                    center,
                    directions,
                    relation_order,
                )
                result["raw_class_mean_norms"] = {
                    relation: float(np.linalg.norm(raw_means[i]))
                    for i, relation in enumerate(relation_order)
                }
            else:
                fitted = fit_affine_axes(train_vectors, train_labels, axes)
                result = evaluate_affine_axes(
                    test_vectors,
                    test_labels,
                    fitted,
                )

            result.update(
                {
                    "repeat": int(repeat),
                    "split_seed": int(args.seed + repeat * 1009),
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_counts": train_counts,
                    "test_counts": test_counts,
                }
            )

            if not args.no_label_shuffle and args.label_shuffle_repeats > 0:
                shuffle_scores: List[float] = []
                for _ in range(args.label_shuffle_repeats):
                    shuffled_labels = shuffle_rng.permutation(train_labels)
                    if args.probe_mode == "four_direction":
                        shuffled_center, shuffled_directions, _ = fit_direction_codebook(
                            train_vectors,
                            shuffled_labels,
                            relation_order,
                            args.feature_centering,
                        )
                        shuffled_result = evaluate_direction_codebook(
                            test_vectors,
                            test_labels,
                            shuffled_center,
                            shuffled_directions,
                            relation_order,
                        )
                    else:
                        shuffled_fitted = fit_affine_axes(
                            train_vectors,
                            shuffled_labels,
                            axes,
                        )
                        shuffled_result = evaluate_affine_axes(
                            test_vectors,
                            test_labels,
                            shuffled_fitted,
                        )
                    shuffle_scores.append(float(shuffled_result["accuracy"]))

                result["train_label_shuffle_control"] = {
                    "repeats": int(args.label_shuffle_repeats),
                    "accuracy": summarize_scalar(shuffle_scores),
                    "chance": float(1.0 / len(relation_order)),
                }

            repeat_results.append(result)

            extra = ""
            if "mean_signed_margin" in result:
                extra = f" | margin={result['mean_signed_margin']:.3f}"
            print(
                f"L{block} rep{repeat}: train={len(train_idx)} test={len(test_idx)} "
                f"acc={result['accuracy']:.3f}{extra}"
            )

        accuracies = [float(item["accuracy"]) for item in repeat_results]
        layer_summary: Dict[str, Any] = {
            "decoder_block_index": int(block),
            "accuracy": summarize_scalar(accuracies),
            "repeats": repeat_results,
        }

        if args.probe_mode == "four_direction":
            layer_summary["mean_signed_margin"] = summarize_scalar(
                [float(item["mean_signed_margin"]) for item in repeat_results]
            )
            layer_summary["mean_top1_margin"] = summarize_scalar(
                [float(item["mean_top1_margin"]) for item in repeat_results]
            )

        summary["layers"][str(block)] = layer_summary
        print(
            f"L{block} summary: "
            f"acc={layer_summary['accuracy']['mean']:.3f}"
            f"±{layer_summary['accuracy']['std']:.3f}"
        )

    output = Path(args.output)
    if output.suffix != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_value(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved summary: {output}")

    tsv_path = output.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "layer\taccuracy_mean\taccuracy_std\taccuracy_min\taccuracy_max\n"
        )
        for block, layer_info in summary["layers"].items():
            accuracy = layer_info["accuracy"]
            handle.write(
                f"L{block}\t{accuracy['mean']:.8f}\t{accuracy['std']:.8f}\t"
                f"{accuracy['min']:.8f}\t{accuracy['max']:.8f}\n"
            )
    print(f"Saved TSV: {tsv_path}")


if __name__ == "__main__":
    main()
