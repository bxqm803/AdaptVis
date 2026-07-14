#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a COCO multi-model attention-centroid versus generation benchmark.

The script evaluates these model keys by default:

    qwen2-2b
    qwen-3b
    qwen-7b
    internvl-1b
    internvl-2b
    internvl-8b
    llava-7b
    llava-13b

For every model it:

1. Runs analyze_coco_attention_flow_swap_step1_v1.py on COCO-two.
2. Reads the saved per-sample attention centroids.
3. Evaluates:
   - normal original generation;
   - normal swapped generation aligned to the original relation;
   - best single attention-centroid head (oracle diagnostic);
   - top-k oracle-selected centroid ensemble (diagnostic upper bound);
   - top-k label-free centroid ensemble;
   - 30%-train-selected top-k centroid ensemble on a held-out 70% test split;
   - best hidden-similarity centroid layer;
   - best all-head attention-map-average layer.
4. Compares centroid and generation through:
   - accuracy gap;
   - centroid accuracy on generation-wrong samples;
   - centroid/generation quadrants;
   - generation-or-centroid union accuracy;
   - per-relation accuracy.
5. Writes one combined CSV/JSON report across all models.

The model is never trained or fine-tuned. Ground-truth labels are used for:
- evaluation;
- oracle head selection, which is explicitly marked as diagnostic;
- the 30% head-selection split, whose result is reported only on held-out 70%.

The label-free top-k method uses only:
- original/swap same-object map stability;
- subject/reference separation;
- visual attention mass.

Required repository files:
- analyze_coco_attention_flow_swap_step1_v1.py
- extract_two_object_relation_states.py
- prompts/COCO_QA_two_obj_with_answer_four_options.jsonl
- the COCO-two dataset under --data-root

This is a sequential runner. Use one sufficiently large GPU; the 8B and 13B
models are run after the smaller models and are unloaded between subprocesses.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "coco-multimodel-centroid-benchmark-v1"

DEFAULT_MODELS = [
    "qwen2-2b",
    "qwen-3b",
    "qwen-7b",
    "internvl-1b",
    "internvl-2b",
    "internvl-8b",
    "llava-7b",
    "llava-13b",
]

RELATIONS = ("left", "right", "above", "below")
RELATION_TO_CODE = {
    relation: index for index, relation in enumerate(RELATIONS)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys from extract_two_object_relation_states.SPECS.",
    )
    parser.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--step1-script",
        default="analyze_coco_attention_flow_swap_step1_v1.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the per-model Step 1 run.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=["eager"])
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers passed to the Step 1 script. Use all for head discovery.",
    )
    parser.add_argument("--report-layer", default="auto")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--output-root",
        default="output/coco_multimodel_centroid_benchmark",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Do not launch models; analyze existing Step 1 directories only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rerun existing per-model Step 1 outputs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the full suite when one model fails.",
    )
    return parser.parse_args()


def parse_models(value: str) -> List[str]:
    result: List[str] = []
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    if not result:
        raise ValueError("--models resolved to an empty list")
    return result


def normalize_relation(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    aliases = {
        "left of": "left",
        "right of": "right",
        "over": "above",
        "under": "below",
        "beneath": "below",
    }
    if text in RELATIONS:
        return text
    return aliases.get(text)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_specs() -> Dict[str, Any]:
    module = importlib.import_module("extract_two_object_relation_states")
    specs = getattr(module, "SPECS", None)
    if not isinstance(specs, dict):
        raise RuntimeError(
            "extract_two_object_relation_states.SPECS is unavailable"
        )
    return specs


def run_command(command: Sequence[str], command_log: Path) -> None:
    command_log.parent.mkdir(parents=True, exist_ok=True)
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def build_step1_command(
    args: argparse.Namespace,
    model: str,
    output_dir: Path,
) -> List[str]:
    command = [
        args.python,
        args.step1_script,
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--prompt-jsonl",
        args.prompt_jsonl,
        "--model",
        model,
        "--device",
        args.device,
        "--attn-impl",
        args.attn_impl,
        "--layers",
        args.layers,
        "--report-layer",
        args.report_layer,
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--print-every",
        str(args.print_every),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def stratified_split(
    samples: Sequence[Dict[str, Any]],
    *,
    train_fraction: float,
    seed: int,
) -> Tuple[set[int], set[int]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--train-fraction must be in (0, 1)")
    grouped: Dict[str, List[int]] = defaultdict(list)
    for row in samples:
        relation = normalize_relation(row.get("gt"))
        if relation in RELATIONS:
            grouped[relation].append(int(row["sid"]))

    train: set[int] = set()
    test: set[int] = set()
    for relation_index, relation in enumerate(RELATIONS):
        values = sorted(grouped[relation])
        rng = random.Random(seed + 1009 * relation_index)
        rng.shuffle(values)
        n_train = int(round(len(values) * train_fraction))
        n_train = max(1, min(len(values) - 1, n_train))
        train.update(values[:n_train])
        test.update(values[n_train:])
    return train, test


def relation_prediction_from_centroids(
    centroids: np.ndarray,
) -> np.ndarray:
    """Predict relation from [..., object=2, xy=2] centroids."""
    dx = centroids[..., 0, 0] - centroids[..., 1, 0]
    dy = centroids[..., 0, 1] - centroids[..., 1, 1]
    horizontal = np.abs(dx) >= np.abs(dy)
    prediction = np.empty(dx.shape, dtype=np.int8)
    prediction[horizontal & (dx < 0)] = RELATION_TO_CODE["left"]
    prediction[horizontal & (dx >= 0)] = RELATION_TO_CODE["right"]
    prediction[(~horizontal) & (dy < 0)] = RELATION_TO_CODE["above"]
    prediction[(~horizontal) & (dy >= 0)] = RELATION_TO_CODE["below"]
    return prediction


def load_sample_arrays(
    step1_dir: Path,
    row: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    sid = int(row["sid"])
    candidates = [
        step1_dir / "sample_arrays" / f"{sid}.npz",
    ]
    array_file = row.get("array_file")
    if array_file:
        supplied = Path(str(array_file))
        candidates.extend([
            supplied,
            step1_dir / supplied,
        ])
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing sample array for sid={sid}; checked={candidates}"
        )
    with np.load(path, allow_pickle=False) as data:
        required = [
            "original_object_centroids",
            "swapped_object_centroids_role_order",
            "same_object_map_cosine",
            "original_object_separation",
            "swapped_object_separation",
            "original_prompt_visual_mass",
            "swapped_prompt_visual_mass",
        ]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise RuntimeError(
                f"{path} is missing arrays {missing}; available={data.files}"
            )
        return {
            name: data[name].astype(np.float64)
            for name in required
        }


def sample_geometry(
    arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    original = arrays["original_object_centroids"]
    swapped_role = arrays["swapped_object_centroids_role_order"]
    swapped_aligned = swapped_role[:, :, [1, 0], :]
    average = 0.5 * (original + swapped_aligned)

    stability = np.clip(
        arrays["same_object_map_cosine"].mean(axis=-1),
        0.0,
        1.0,
    )
    separation = np.clip(
        np.sqrt(np.maximum(
            0.0,
            arrays["original_object_separation"]
            * arrays["swapped_object_separation"],
        )),
        0.0,
        1.0,
    )
    original_mass = arrays["original_prompt_visual_mass"]
    swapped_mass = arrays["swapped_prompt_visual_mass"]
    visual_mass = np.clip(
        0.25 * (
            original_mass[:, :, 0]
            + original_mass[:, :, 1]
            + swapped_mass[:, :, 0]
            + swapped_mass[:, :, 1]
        ),
        0.0,
        1.0,
    )
    local_quality = (
        stability
        * np.sqrt(np.maximum(0.0, separation * visual_mass))
    )
    return average, local_quality


def flatten_head_order(
    score: np.ndarray,
    top_k: int,
) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for layer_position in range(score.shape[0]):
        for head in range(score.shape[1]):
            value = float(score[layer_position, head])
            if np.isfinite(value):
                candidates.append((value, layer_position, head))
    candidates.sort(reverse=True)
    return [
        (layer_position, head, value)
        for value, layer_position, head in candidates[:top_k]
    ]


def train_selected_head_scores(
    step1_dir: Path,
    samples: Sequence[Dict[str, Any]],
    train_sids: set[int],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, int]:
    correct = np.zeros(shape, dtype=np.float64)
    count = 0
    for row in samples:
        sid = int(row["sid"])
        if sid not in train_sids:
            continue
        gt = normalize_relation(row.get("gt"))
        if gt not in RELATIONS:
            continue
        arrays = load_sample_arrays(step1_dir, row)
        average, _ = sample_geometry(arrays)
        predictions = relation_prediction_from_centroids(average)
        correct += predictions == RELATION_TO_CODE[gt]
        count += 1
    if count == 0:
        raise RuntimeError("The 30% training split contains no usable samples")
    return correct / count, count


def selected_head_description(
    selected: Sequence[Tuple[int, int, float]],
    layer_indices: np.ndarray,
) -> str:
    return ",".join(
        f"L{int(layer_indices[layer_position])}H{head}"
        for layer_position, head, _ in selected
    )


def ensemble_prediction(
    average_centroids: np.ndarray,
    local_quality: np.ndarray,
    selected: Sequence[Tuple[int, int, float]],
) -> str:
    centroids: List[np.ndarray] = []
    weights: List[float] = []
    for layer_position, head, global_score in selected:
        centroid = average_centroids[layer_position, head]
        local_score = float(local_quality[layer_position, head])
        weight = max(0.0, float(global_score)) * max(0.0, local_score)
        centroids.append(centroid)
        weights.append(weight)

    centroid_stack = np.stack(centroids, axis=0)
    weight_array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weight_array).all() or float(weight_array.sum()) <= 1e-12:
        weight_array = np.ones(len(centroids), dtype=np.float64)
    weight_array /= weight_array.sum()
    ensemble = np.sum(
        weight_array[:, None, None] * centroid_stack,
        axis=0,
    )
    code = int(relation_prediction_from_centroids(ensemble))
    return RELATIONS[code]


def summarize_method(
    *,
    method: str,
    rows: Sequence[Dict[str, Any]],
    prediction_by_sid: Dict[int, str],
    evaluation_sids: Optional[set[int]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    usable: List[Tuple[Dict[str, Any], str, str, Optional[str]]] = []
    for row in rows:
        sid = int(row["sid"])
        if evaluation_sids is not None and sid not in evaluation_sids:
            continue
        centroid_prediction = prediction_by_sid.get(sid)
        gt = normalize_relation(row.get("gt"))
        generation_prediction = normalize_relation(row.get("original_prediction"))
        if centroid_prediction not in RELATIONS or gt not in RELATIONS:
            continue
        usable.append((row, centroid_prediction, gt, generation_prediction))

    if not usable:
        return {
            "method": method,
            "n": 0,
        }, []

    centroid_correct = np.asarray([
        prediction == gt
        for _, prediction, gt, _ in usable
    ], dtype=bool)
    generation_valid = np.asarray([
        generation in RELATIONS
        for _, _, _, generation in usable
    ], dtype=bool)
    generation_correct = np.asarray([
        generation == gt
        for _, _, gt, generation in usable
    ], dtype=bool)

    generation_wrong_mask = generation_valid & (~generation_correct)
    generation_correct_mask = generation_valid & generation_correct

    quadrants = Counter()
    for centroid_ok, generation_ok, generation_is_valid in zip(
        centroid_correct,
        generation_correct,
        generation_valid,
    ):
        if not generation_is_valid:
            continue
        quadrants[
            ("centroid_correct" if centroid_ok else "centroid_wrong")
            + "__"
            + ("generation_correct" if generation_ok else "generation_wrong")
        ] += 1

    valid_joint = generation_valid
    summary = {
        "method": method,
        "n": len(usable),
        "n_generation_valid": int(generation_valid.sum()),
        "centroid_accuracy": float(centroid_correct.mean()),
        "generation_accuracy_same_subset": (
            float(generation_correct[generation_valid].mean())
            if generation_valid.any() else None
        ),
        "centroid_minus_generation": (
            float(centroid_correct[generation_valid].mean())
            - float(generation_correct[generation_valid].mean())
            if generation_valid.any() else None
        ),
        "centroid_accuracy_on_generation_wrong": (
            float(centroid_correct[generation_wrong_mask].mean())
            if generation_wrong_mask.any() else None
        ),
        "n_generation_wrong": int(generation_wrong_mask.sum()),
        "centroid_accuracy_on_generation_correct": (
            float(centroid_correct[generation_correct_mask].mean())
            if generation_correct_mask.any() else None
        ),
        "generation_or_centroid_union_accuracy": (
            float(np.mean(
                generation_correct[valid_joint]
                | centroid_correct[valid_joint]
            ))
            if valid_joint.any() else None
        ),
        "generation_centroid_agreement": (
            float(np.mean([
                centroid_prediction == generation_prediction
                for _, centroid_prediction, _, generation_prediction in usable
                if generation_prediction in RELATIONS
            ]))
            if generation_valid.any() else None
        ),
        **{
            key: int(quadrants.get(key, 0))
            for key in [
                "centroid_correct__generation_correct",
                "centroid_correct__generation_wrong",
                "centroid_wrong__generation_correct",
                "centroid_wrong__generation_wrong",
            ]
        },
    }

    per_relation_rows: List[Dict[str, Any]] = []
    for relation in RELATIONS:
        relation_items = [
            item for item in usable if item[2] == relation
        ]
        if not relation_items:
            continue
        centroid_relation_correct = [
            prediction == gt
            for _, prediction, gt, _ in relation_items
        ]
        generation_relation_valid = [
            generation in RELATIONS
            for _, _, _, generation in relation_items
        ]
        generation_relation_correct = [
            generation == gt
            for _, _, gt, generation in relation_items
            if generation in RELATIONS
        ]
        per_relation_rows.append({
            "method": method,
            "relation": relation,
            "n": len(relation_items),
            "centroid_accuracy": float(np.mean(centroid_relation_correct)),
            "generation_accuracy_same_subset": (
                float(np.mean(generation_relation_correct))
                if generation_relation_correct else None
            ),
            "n_generation_valid": int(sum(generation_relation_valid)),
        })
    return summary, per_relation_rows


def evaluate_step1_model(
    *,
    model: str,
    step1_dir: Path,
    top_k: int,
    train_fraction: float,
    split_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    summary_path = step1_dir / "summary.json"
    aggregate_path = step1_dir / "aggregate_metrics.npz"
    samples_path = step1_dir / "samples.jsonl"
    for path in (summary_path, aggregate_path, samples_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 1 output: {path}")

    step1_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = read_jsonl(samples_path)
    if not samples:
        raise RuntimeError(f"No samples in {samples_path}")

    with np.load(aggregate_path, allow_pickle=False) as aggregate:
        required = [
            "layer_indices",
            "attention_average_accuracy",
            "unsupervised_head_score",
            "headmean_average_accuracy",
            "similarity_average_accuracy",
        ]
        missing = [name for name in required if name not in aggregate.files]
        if missing:
            raise RuntimeError(
                f"{aggregate_path} is missing arrays {missing}"
            )
        layer_indices = aggregate["layer_indices"].astype(np.int64)
        oracle_score = aggregate[
            "attention_average_accuracy"
        ].astype(np.float64)
        unsupervised_score = aggregate[
            "unsupervised_head_score"
        ].astype(np.float64)
        headmean_average_accuracy = aggregate[
            "headmean_average_accuracy"
        ].astype(np.float64)
        similarity_average_accuracy = aggregate[
            "similarity_average_accuracy"
        ].astype(np.float64)

    if oracle_score.ndim != 2:
        raise RuntimeError(
            f"Expected [layer,head] oracle score, got {oracle_score.shape}"
        )

    train_sids, test_sids = stratified_split(
        samples,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    train_score, n_train = train_selected_head_scores(
        step1_dir,
        samples,
        train_sids,
        oracle_score.shape,
    )

    selections = {
        "oracle_topk_full": flatten_head_order(oracle_score, top_k),
        "unsupervised_topk_full": flatten_head_order(
            unsupervised_score, top_k
        ),
        "train30_topk_test70": flatten_head_order(train_score, top_k),
    }

    predictions: Dict[str, Dict[int, str]] = {
        method: {} for method in selections
    }
    for row in samples:
        sid = int(row["sid"])
        arrays = load_sample_arrays(step1_dir, row)
        average, local_quality = sample_geometry(arrays)
        for method, selected in selections.items():
            predictions[method][sid] = ensemble_prediction(
                average,
                local_quality,
                selected,
            )

    method_summaries: Dict[str, Dict[str, Any]] = {}
    per_relation_rows: List[Dict[str, Any]] = []
    for method, prediction_by_sid in predictions.items():
        evaluation_sids = (
            test_sids if method == "train30_topk_test70" else None
        )
        method_summary, relation_rows = summarize_method(
            method=method,
            rows=samples,
            prediction_by_sid=prediction_by_sid,
            evaluation_sids=evaluation_sids,
        )
        method_summary["selected_heads"] = selected_head_description(
            selections[method],
            layer_indices,
        )
        method_summary["selection_uses_full_gt"] = (
            method == "oracle_topk_full"
        )
        method_summary["selection_is_label_free"] = (
            method == "unsupervised_topk_full"
        )
        method_summary["selection_train_n"] = (
            n_train if method == "train30_topk_test70" else None
        )
        method_summary["evaluation_test_n"] = (
            len(test_sids)
            if method == "train30_topk_test70" else None
        )
        method_summaries[method] = method_summary
        for row in relation_rows:
            row["model"] = model
            per_relation_rows.append(row)

    best_single_position = np.unravel_index(
        int(np.nanargmax(oracle_score)),
        oracle_score.shape,
    )
    best_single_layer_position, best_single_head = best_single_position
    best_head = {
        "layer": int(layer_indices[best_single_layer_position]),
        "head": int(best_single_head),
        "accuracy": float(
            oracle_score[best_single_layer_position, best_single_head]
        ),
    }
    best_headmean_layer_position = int(
        np.nanargmax(headmean_average_accuracy)
    )
    best_similarity_layer_position = int(
        np.nanargmax(similarity_average_accuracy)
    )

    generation_accuracy = float(
        step1_summary["generation_original_accuracy"]
    )
    headline = {
        "model": model,
        "n_samples": int(step1_summary["n_samples"]),
        "generation_original_accuracy": generation_accuracy,
        "generation_swapped_aligned_accuracy": float(
            step1_summary["generation_swapped_aligned_accuracy"]
        ),
        "answer_swap_consistency": float(
            step1_summary["answer_swap_consistency"]
        ),
        "best_single_centroid_accuracy_oracle": best_head["accuracy"],
        "best_single_centroid_head": (
            f"L{best_head['layer']}H{best_head['head']}"
        ),
        "best_all_head_map_average_accuracy": float(
            headmean_average_accuracy[best_headmean_layer_position]
        ),
        "best_all_head_map_average_layer": int(
            layer_indices[best_headmean_layer_position]
        ),
        "best_hidden_similarity_centroid_accuracy": float(
            similarity_average_accuracy[best_similarity_layer_position]
        ),
        "best_hidden_similarity_centroid_layer": int(
            layer_indices[best_similarity_layer_position]
        ),
        "oracle_top5_centroid_accuracy": method_summaries[
            "oracle_topk_full"
        ]["centroid_accuracy"],
        "oracle_top5_minus_generation": method_summaries[
            "oracle_topk_full"
        ]["centroid_minus_generation"],
        "oracle_top5_on_generation_wrong": method_summaries[
            "oracle_topk_full"
        ]["centroid_accuracy_on_generation_wrong"],
        "oracle_top5_union_accuracy": method_summaries[
            "oracle_topk_full"
        ]["generation_or_centroid_union_accuracy"],
        "oracle_top5_heads": method_summaries[
            "oracle_topk_full"
        ]["selected_heads"],
        "unsup_top5_centroid_accuracy": method_summaries[
            "unsupervised_topk_full"
        ]["centroid_accuracy"],
        "unsup_top5_minus_generation": method_summaries[
            "unsupervised_topk_full"
        ]["centroid_minus_generation"],
        "unsup_top5_on_generation_wrong": method_summaries[
            "unsupervised_topk_full"
        ]["centroid_accuracy_on_generation_wrong"],
        "unsup_top5_union_accuracy": method_summaries[
            "unsupervised_topk_full"
        ]["generation_or_centroid_union_accuracy"],
        "unsup_top5_heads": method_summaries[
            "unsupervised_topk_full"
        ]["selected_heads"],
        "train30_test70_generation_accuracy": method_summaries[
            "train30_topk_test70"
        ]["generation_accuracy_same_subset"],
        "train30_test70_centroid_accuracy": method_summaries[
            "train30_topk_test70"
        ]["centroid_accuracy"],
        "train30_test70_minus_generation": method_summaries[
            "train30_topk_test70"
        ]["centroid_minus_generation"],
        "train30_test70_on_generation_wrong": method_summaries[
            "train30_topk_test70"
        ]["centroid_accuracy_on_generation_wrong"],
        "train30_test70_union_accuracy": method_summaries[
            "train30_topk_test70"
        ]["generation_or_centroid_union_accuracy"],
        "train30_selected_heads": method_summaries[
            "train30_topk_test70"
        ]["selected_heads"],
        "train30_n": n_train,
        "test70_n": len(test_sids),
    }

    detailed = {
        "model": model,
        "step1_summary": step1_summary,
        "best_single_head": best_head,
        "best_all_head_map_average": {
            "layer": int(layer_indices[best_headmean_layer_position]),
            "accuracy": float(
                headmean_average_accuracy[best_headmean_layer_position]
            ),
        },
        "best_hidden_similarity_centroid": {
            "layer": int(layer_indices[best_similarity_layer_position]),
            "accuracy": float(
                similarity_average_accuracy[best_similarity_layer_position]
            ),
        },
        "methods": method_summaries,
        "split": {
            "seed": split_seed,
            "train_fraction": train_fraction,
            "train_sids": sorted(train_sids),
            "test_sids": sorted(test_sids),
        },
    }
    return headline, per_relation_rows, detailed


def print_headline(row: Dict[str, Any]) -> None:
    print("\n" + "=" * 112)
    print(f"MODEL: {row['model']}")
    print("=" * 112)
    print(
        f"generation={row['generation_original_accuracy']:.4f} | "
        f"swap-generation={row['generation_swapped_aligned_accuracy']:.4f}"
    )
    print(
        f"best-single-oracle={row['best_single_centroid_accuracy_oracle']:.4f} "
        f"({row['best_single_centroid_head']})"
    )
    print(
        f"oracle-top5={row['oracle_top5_centroid_accuracy']:.4f} | "
        f"gap={row['oracle_top5_minus_generation']:+.4f} | "
        f"on-gen-wrong={row['oracle_top5_on_generation_wrong']:.4f} | "
        f"union={row['oracle_top5_union_accuracy']:.4f}"
    )
    print(
        f"unsup-top5={row['unsup_top5_centroid_accuracy']:.4f} | "
        f"gap={row['unsup_top5_minus_generation']:+.4f} | "
        f"on-gen-wrong={row['unsup_top5_on_generation_wrong']:.4f} | "
        f"union={row['unsup_top5_union_accuracy']:.4f}"
    )
    print(
        f"train30/test70 centroid={row['train30_test70_centroid_accuracy']:.4f} | "
        f"generation={row['train30_test70_generation_accuracy']:.4f} | "
        f"gap={row['train30_test70_minus_generation']:+.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    models = parse_models(args.models)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    step1_root = output_root / "step1"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    command_log = output_root / "commands.log"
    errors_path = reports_root / "errors.jsonl"
    if args.overwrite:
        for path in (
            reports_root / "model_comparison.csv",
            reports_root / "model_comparison.json",
            reports_root / "per_relation.csv",
            reports_root / "detailed_results.json",
            errors_path,
        ):
            if path.exists():
                path.unlink()

    specs = import_specs()
    unsupported = [model for model in models if model not in specs]
    if unsupported:
        print(
            "WARNING: model keys absent from SPECS and will be skipped: "
            + ", ".join(unsupported)
        )

    headline_rows: List[Dict[str, Any]] = []
    per_relation_rows: List[Dict[str, Any]] = []
    detailed_results: Dict[str, Any] = {}

    for model in models:
        if model not in specs:
            append_jsonl(errors_path, {
                "model": model,
                "stage": "model_validation",
                "error": "Model key is absent from extract_two_object_relation_states.SPECS",
            })
            continue

        model_step1_dir = step1_root / model
        try:
            if not args.skip_step1:
                complete = (
                    (model_step1_dir / "summary.json").exists()
                    and (model_step1_dir / "aggregate_metrics.npz").exists()
                    and (model_step1_dir / "samples.jsonl").exists()
                )
                if args.overwrite and model_step1_dir.exists():
                    shutil.rmtree(model_step1_dir)
                    complete = False
                if complete:
                    print(
                        f"\n[{model}] complete Step 1 output already exists; skipping extraction."
                    )
                else:
                    if model_step1_dir.exists():
                        raise RuntimeError(
                            f"Partial Step 1 directory exists: {model_step1_dir}. "
                            "Delete it or pass --overwrite."
                        )
                    run_command(
                        build_step1_command(
                            args,
                            model,
                            model_step1_dir,
                        ),
                        command_log,
                    )

            headline, relation_rows, detailed = evaluate_step1_model(
                model=model,
                step1_dir=model_step1_dir,
                top_k=args.top_k,
                train_fraction=args.train_fraction,
                split_seed=args.split_seed,
            )
            headline_rows.append(headline)
            per_relation_rows.extend(relation_rows)
            detailed_results[model] = detailed
            print_headline(headline)

            # Save after each model so long suites are interruption-safe.
            write_csv(
                reports_root / "model_comparison.csv",
                headline_rows,
            )
            write_csv(
                reports_root / "per_relation.csv",
                per_relation_rows,
            )
            (reports_root / "model_comparison.json").write_text(
                json.dumps(
                    headline_rows,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (reports_root / "detailed_results.json").write_text(
                json.dumps(
                    detailed_results,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except Exception as exc:
            error = {
                "model": model,
                "stage": "step1_or_analysis",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-24:],
            }
            append_jsonl(errors_path, error)
            print(
                f"\n[ERROR] model={model}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                raise

    if not headline_rows:
        raise RuntimeError(
            f"No model completed successfully. See {errors_path}"
        )

    # A compact matrix convenient for paper-table preparation.
    matrix_rows: List[Dict[str, Any]] = []
    for row in headline_rows:
        matrix_rows.append({
            "model": row["model"],
            "generation": row["generation_original_accuracy"],
            "best_single_oracle": row[
                "best_single_centroid_accuracy_oracle"
            ],
            "oracle_top5": row["oracle_top5_centroid_accuracy"],
            "unsup_top5": row["unsup_top5_centroid_accuracy"],
            "train30_test70_generation": row[
                "train30_test70_generation_accuracy"
            ],
            "train30_test70_centroid": row[
                "train30_test70_centroid_accuracy"
            ],
            "train30_test70_gap": row[
                "train30_test70_minus_generation"
            ],
            "hidden_similarity_best": row[
                "best_hidden_similarity_centroid_accuracy"
            ],
            "all_head_map_average_best": row[
                "best_all_head_map_average_accuracy"
            ],
        })
    write_csv(reports_root / "paper_matrix.csv", matrix_rows)

    print("\n" + "=" * 112)
    print("MULTI-MODEL BENCHMARK COMPLETE")
    print("=" * 112)
    print(f"Successful models: {len(headline_rows)}/{len(models)}")
    print(f"Main comparison:   {reports_root / 'model_comparison.csv'}")
    print(f"Paper matrix:      {reports_root / 'paper_matrix.csv'}")
    print(f"Per relation:      {reports_root / 'per_relation.csv'}")
    print(f"Detailed JSON:     {reports_root / 'detailed_results.json'}")
    if errors_path.exists():
        print(f"Errors/skips:      {errors_path}")


if __name__ == "__main__":
    main()
