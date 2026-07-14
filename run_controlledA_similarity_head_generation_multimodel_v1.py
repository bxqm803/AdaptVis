#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Controlled-A centroid-versus-generation comparison.

Default models:
    qwen2-2b
    qwen-3b
    qwen-7b
    internvl-1b
    internvl-2b
    internvl-8b
    llava-7b
    llava-13b

For each model, this script runs:
    analyze_controlledA_similarity_head_generation_step1_v1.py

It reports exactly three headline results:

1. Best hidden-similarity centroid accuracy and decoder layer.
2. Best single attention-head centroid accuracy and layer/head.
3. Normal model generation accuracy.

Controlled-A labels are:
    left, right, on, under

For centroid geometry:
    on    = subject center is above reference center
    under = subject center is below reference center

Every normal generation is printed with:
    question, GT, parsed prediction, current ACC, raw generation text.

There is no train30/test70 split, top-k ensemble, training, or fine-tuning.
The best layer/head are selected with full-set GT and are diagnostic oracle
statistics, not label-free deployment results.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "controlledA-similarity-head-generation-multimodel-v1"

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

RELATIONS = ("left", "right", "on", "under")
CODE_TO_RELATION = {
    index: relation for index, relation in enumerate(RELATIONS)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys from the Controlled-A extractor SPECS.",
    )
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--controlled-module",
        default="",
        help=(
            "Optional explicit Controlled-A extractor module. Otherwise the "
            "runner tries extract_controlled_relation_states_standalone and "
            "extract_controlledA_relation_states_standalone."
        ),
    )
    parser.add_argument("--dataset-key", default="Controlled_Images_A")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--step1-script",
        default="analyze_controlledA_similarity_head_generation_step1_v1.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for each model subprocess.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=["eager"])
    parser.add_argument(
        "--layers",
        default="all",
        help="Use all to discover the best similarity layer and attention head.",
    )
    parser.add_argument(
        "--report-layer",
        default="auto",
        help="Only affects auxiliary Step 1 summaries.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature used by hidden-similarity visual grounding.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full Controlled-A set.",
    )
    parser.add_argument(
        "--output-root",
        default="output/controlledA_similarity_head_generation_multimodel",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Analyze already completed per-model Step 1 outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rerun existing per-model outputs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the suite when one model fails.",
    )
    return parser.parse_args()

def parse_models(value: str) -> List[str]:
    models: List[str] = []
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in models:
            models.append(item)
    if not models:
        raise ValueError("--models resolved to an empty list")
    return models


def normalize_relation(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "left": "left",
        "left of": "left",
        "right": "right",
        "right of": "right",
        "on": "on",
        "above": "on",
        "over": "on",
        "on top of": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
    }
    return aliases.get(" ".join(text.split()))

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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def import_controlled_module(explicit: str = ""):
    names: List[str] = []
    if explicit:
        names.append(explicit)
    names.extend([
        "extract_controlled_relation_states_standalone",
        "extract_controlledA_relation_states_standalone",
    ])
    errors: List[str] = []
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "Could not import a Controlled-A extractor module. Tried:\n  "
        + "\n  ".join(errors)
    )


def import_specs(explicit: str = "") -> Tuple[Any, Dict[str, Any]]:
    module = import_controlled_module(explicit)
    specs = getattr(module, "SPECS", None)
    if not isinstance(specs, dict):
        raise RuntimeError(f"{module.__name__}.SPECS is unavailable")
    return module, specs

def run_command(command: Sequence[str], command_log: Path) -> None:
    command_log.parent.mkdir(parents=True, exist_ok=True)
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def build_step1_command(
    args: argparse.Namespace,
    *,
    model: str,
    output_dir: Path,
) -> List[str]:
    command = [
        args.python,
        args.step1_script,
        "--dataset",
        "controlled_A",
        "--prompt-jsonl",
        args.prompt_jsonl,
        "--dataset-key",
        args.dataset_key,
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
        "1",
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    if args.controlled_module:
        command.extend(["--controlled-module", args.controlled_module])
    if args.download:
        command.append("--download")
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.overwrite:
        command.append("--overwrite")
    return command

def resolve_array_path(step1_dir: Path, row: Dict[str, Any]) -> Path:
    sid = int(row["sid"])
    candidates = [
        step1_dir / "sample_arrays" / f"{sid}.npz",
    ]
    supplied = row.get("array_file")
    if supplied:
        supplied_path = Path(str(supplied))
        candidates.extend([
            supplied_path,
            step1_dir / supplied_path,
            step1_dir / supplied_path.name,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find sample arrays for sid={sid}; checked={candidates}"
    )


def relation_from_code(value: Any) -> Optional[str]:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    return CODE_TO_RELATION.get(code)


def load_predictions_for_selected_positions(
    *,
    step1_dir: Path,
    sample_rows: Sequence[Dict[str, Any]],
    similarity_layer_position: int,
    attention_layer_position: int,
    attention_head: int,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    similarity_predictions: Dict[int, str] = {}
    attention_predictions: Dict[int, str] = {}

    for row in sample_rows:
        sid = int(row["sid"])
        array_path = resolve_array_path(step1_dir, row)
        with np.load(array_path, allow_pickle=False) as data:
            if "similarity_average_prediction" not in data.files:
                raise RuntimeError(
                    f"{array_path} lacks similarity_average_prediction"
                )
            if "attention_average_prediction" not in data.files:
                raise RuntimeError(
                    f"{array_path} lacks attention_average_prediction"
                )

            similarity_array = data[
                "similarity_average_prediction"
            ]
            attention_array = data[
                "attention_average_prediction"
            ]

            similarity_predictions[sid] = relation_from_code(
                similarity_array[similarity_layer_position]
            )
            attention_predictions[sid] = relation_from_code(
                attention_array[
                    attention_layer_position,
                    attention_head,
                ]
            )

    return similarity_predictions, attention_predictions


def accuracy(
    predictions: Sequence[Optional[str]],
    ground_truth: Sequence[str],
) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError("Prediction and GT lengths differ")
    if not ground_truth:
        return float("nan")
    return float(np.mean([
        prediction == gt
        for prediction, gt in zip(predictions, ground_truth)
    ]))


def method_on_generation_wrong(
    method_predictions: Sequence[Optional[str]],
    generation_predictions: Sequence[Optional[str]],
    ground_truth: Sequence[str],
) -> Tuple[Optional[float], int]:
    mask = [
        generation != gt
        for generation, gt in zip(generation_predictions, ground_truth)
    ]
    count = int(sum(mask))
    if count == 0:
        return None, 0
    value = float(np.mean([
        method == gt
        for method, generation, gt in zip(
            method_predictions,
            generation_predictions,
            ground_truth,
        )
        if generation != gt
    ]))
    return value, count


def evaluate_model(
    *,
    model: str,
    step1_dir: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    summary_path = step1_dir / "summary.json"
    aggregate_path = step1_dir / "aggregate_metrics.npz"
    samples_path = step1_dir / "samples.jsonl"

    for path in (summary_path, aggregate_path, samples_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 1 output: {path}")

    step1_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    sample_rows = read_jsonl(samples_path)
    if not sample_rows:
        raise RuntimeError(f"No sample rows in {samples_path}")

    with np.load(aggregate_path, allow_pickle=False) as aggregate:
        required = [
            "layer_indices",
            "similarity_average_accuracy",
            "attention_average_accuracy",
        ]
        missing = [
            name for name in required if name not in aggregate.files
        ]
        if missing:
            raise RuntimeError(
                f"{aggregate_path} is missing {missing}; "
                f"available={aggregate.files}"
            )

        layer_indices = aggregate["layer_indices"].astype(np.int64)
        similarity_accuracy_by_layer = aggregate[
            "similarity_average_accuracy"
        ].astype(np.float64)
        attention_accuracy_by_head = aggregate[
            "attention_average_accuracy"
        ].astype(np.float64)

    if similarity_accuracy_by_layer.ndim != 1:
        raise RuntimeError(
            "similarity_average_accuracy must have shape [layer], "
            f"got {similarity_accuracy_by_layer.shape}"
        )
    if attention_accuracy_by_head.ndim != 2:
        raise RuntimeError(
            "attention_average_accuracy must have shape [layer,head], "
            f"got {attention_accuracy_by_head.shape}"
        )

    best_similarity_position = int(
        np.nanargmax(similarity_accuracy_by_layer)
    )
    best_similarity_layer = int(
        layer_indices[best_similarity_position]
    )

    best_attention_position = np.unravel_index(
        int(np.nanargmax(attention_accuracy_by_head)),
        attention_accuracy_by_head.shape,
    )
    best_attention_layer_position = int(
        best_attention_position[0]
    )
    best_attention_head = int(best_attention_position[1])
    best_attention_layer = int(
        layer_indices[best_attention_layer_position]
    )

    similarity_predictions, attention_predictions = (
        load_predictions_for_selected_positions(
            step1_dir=step1_dir,
            sample_rows=sample_rows,
            similarity_layer_position=best_similarity_position,
            attention_layer_position=best_attention_layer_position,
            attention_head=best_attention_head,
        )
    )

    ground_truth: List[str] = []
    generation_predictions: List[Optional[str]] = []
    similarity_prediction_list: List[Optional[str]] = []
    attention_prediction_list: List[Optional[str]] = []
    sample_comparison_rows: List[Dict[str, Any]] = []

    running_generation_correct = 0
    for index, row in enumerate(sample_rows, 1):
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        if gt not in RELATIONS:
            raise RuntimeError(f"Invalid GT for sid={sid}: {row.get('gt')!r}")

        generation_prediction = normalize_relation(
            row.get("original_prediction")
        )
        similarity_prediction = similarity_predictions.get(sid)
        attention_prediction = attention_predictions.get(sid)

        generation_correct = generation_prediction == gt
        similarity_correct = similarity_prediction == gt
        attention_correct = attention_prediction == gt

        running_generation_correct += int(generation_correct)

        ground_truth.append(gt)
        generation_predictions.append(generation_prediction)
        similarity_prediction_list.append(similarity_prediction)
        attention_prediction_list.append(attention_prediction)

        sample_comparison_rows.append({
            "model": model,
            "index": index,
            "sid": sid,
            "question": row.get("question"),
            "gt": gt,
            "generation_text": row.get("original_generated_text"),
            "generation_prediction": generation_prediction,
            "generation_correct": generation_correct,
            "generation_running_accuracy": (
                running_generation_correct / index
            ),
            "similarity_centroid_layer": best_similarity_layer,
            "similarity_centroid_prediction": similarity_prediction,
            "similarity_centroid_correct": similarity_correct,
            "best_attention_head_layer": best_attention_layer,
            "best_attention_head": best_attention_head,
            "best_attention_head_prediction": attention_prediction,
            "best_attention_head_correct": attention_correct,
        })

    n = len(ground_truth)
    generation_accuracy = accuracy(
        generation_predictions,
        ground_truth,
    )
    similarity_accuracy = accuracy(
        similarity_prediction_list,
        ground_truth,
    )
    attention_accuracy = accuracy(
        attention_prediction_list,
        ground_truth,
    )
    generation_valid_rate = float(np.mean([
        prediction in RELATIONS
        for prediction in generation_predictions
    ]))

    similarity_on_wrong, n_generation_wrong = (
        method_on_generation_wrong(
            similarity_prediction_list,
            generation_predictions,
            ground_truth,
        )
    )
    attention_on_wrong, _ = method_on_generation_wrong(
        attention_prediction_list,
        generation_predictions,
        ground_truth,
    )

    headline = {
        "model": model,
        "n": n,
        "similarity_centroid_accuracy": similarity_accuracy,
        "similarity_centroid_best_layer": best_similarity_layer,
        "best_attention_head_centroid_accuracy": attention_accuracy,
        "best_attention_head_layer": best_attention_layer,
        "best_attention_head": best_attention_head,
        "normal_generation_accuracy": generation_accuracy,
        "generation_valid_rate": generation_valid_rate,
        "similarity_minus_generation": (
            similarity_accuracy - generation_accuracy
        ),
        "best_head_minus_generation": (
            attention_accuracy - generation_accuracy
        ),
        "n_generation_wrong": n_generation_wrong,
        "similarity_accuracy_on_generation_wrong": (
            similarity_on_wrong
        ),
        "best_head_accuracy_on_generation_wrong": (
            attention_on_wrong
        ),
        "selection_note": (
            "best similarity layer and best attention head selected "
            "with full-set GT; diagnostic oracle selection"
        ),
    }

    per_relation_rows: List[Dict[str, Any]] = []
    for relation in RELATIONS:
        indices = [
            index for index, gt in enumerate(ground_truth)
            if gt == relation
        ]
        if not indices:
            continue
        per_relation_rows.append({
            "model": model,
            "relation": relation,
            "n": len(indices),
            "similarity_centroid_accuracy": float(np.mean([
                similarity_prediction_list[index] == relation
                for index in indices
            ])),
            "best_attention_head_centroid_accuracy": float(np.mean([
                attention_prediction_list[index] == relation
                for index in indices
            ])),
            "normal_generation_accuracy": float(np.mean([
                generation_predictions[index] == relation
                for index in indices
            ])),
        })

    return headline, per_relation_rows, sample_comparison_rows


def print_model_summary(row: Dict[str, Any]) -> None:
    print("\n" + "=" * 112)
    print(f"MODEL: {row['model']} | n={row['n']}")
    print("=" * 112)
    print(
        "1. Similarity centroid:       "
        f"{row['similarity_centroid_accuracy']:.4f} "
        f"at L{row['similarity_centroid_best_layer']}"
    )
    print(
        "2. Best attention-head centroid: "
        f"{row['best_attention_head_centroid_accuracy']:.4f} "
        f"at L{row['best_attention_head_layer']}"
        f"H{row['best_attention_head']}"
    )
    print(
        "3. Normal generation:         "
        f"{row['normal_generation_accuracy']:.4f} "
        f"(valid rate={row['generation_valid_rate']:.4f})"
    )
    print(
        "Gaps versus generation:       "
        f"similarity={row['similarity_minus_generation']:+.4f} | "
        f"best-head={row['best_head_minus_generation']:+.4f}"
    )
    print(
        "Centroid accuracy on generation-wrong samples: "
        f"similarity={row['similarity_accuracy_on_generation_wrong']} | "
        f"best-head={row['best_head_accuracy_on_generation_wrong']}"
    )


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)

    output_root = Path(args.output_root)
    step1_root = output_root / "step1"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    errors_path = reports_root / "errors.jsonl"
    command_log = output_root / "commands.log"

    if args.overwrite:
        for path in (
            reports_root / "model_comparison.csv",
            reports_root / "model_comparison.json",
            reports_root / "per_relation.csv",
            reports_root / "sample_comparison.csv",
            errors_path,
        ):
            if path.exists():
                path.unlink()

    controlled_module, specs = import_specs(args.controlled_module)
    print(f"Controlled-A module: {controlled_module.__name__}")
    missing_models = [
        model for model in models if model not in specs
    ]
    if missing_models:
        print(
            "WARNING: these model keys are absent from "
            "the Controlled-A extractor SPECS and will be skipped: "
            + ", ".join(missing_models)
        )

    headline_rows: List[Dict[str, Any]] = []
    per_relation_rows: List[Dict[str, Any]] = []
    all_sample_rows: List[Dict[str, Any]] = []

    for model in models:
        if model not in specs:
            append_jsonl(errors_path, {
                "model": model,
                "stage": "model_validation",
                "error": (
                    "Model key absent from "
                    f"{controlled_module.__name__}.SPECS"
                ),
            })
            continue

        model_output = step1_root / model
        try:
            if not args.skip_step1:
                complete = all(
                    (model_output / filename).exists()
                    for filename in (
                        "summary.json",
                        "aggregate_metrics.npz",
                        "samples.jsonl",
                    )
                )

                if args.overwrite and model_output.exists():
                    shutil.rmtree(model_output)
                    complete = False

                if complete:
                    print(
                        f"\n[{model}] completed Step 1 already exists; "
                        "skipping model forward."
                    )
                else:
                    if model_output.exists():
                        raise RuntimeError(
                            f"Partial output directory exists: {model_output}. "
                            "Delete it or pass --overwrite."
                        )
                    run_command(
                        build_step1_command(
                            args,
                            model=model,
                            output_dir=model_output,
                        ),
                        command_log,
                    )

            headline, relation_rows, sample_rows = evaluate_model(
                model=model,
                step1_dir=model_output,
            )
            headline_rows.append(headline)
            per_relation_rows.extend(relation_rows)
            all_sample_rows.extend(sample_rows)
            print_model_summary(headline)

            # Save incrementally after every model.
            write_csv(
                reports_root / "model_comparison.csv",
                headline_rows,
            )
            write_csv(
                reports_root / "per_relation.csv",
                per_relation_rows,
            )
            write_csv(
                reports_root / "sample_comparison.csv",
                all_sample_rows,
            )
            (reports_root / "model_comparison.json").write_text(
                json.dumps(
                    headline_rows,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except Exception as exc:
            append_jsonl(errors_path, {
                "model": model,
                "stage": "step1_or_analysis",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-24:],
            })
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

    print("\n" + "=" * 112)
    print("ALL MODELS COMPLETE")
    print("=" * 112)
    print(f"Successful models: {len(headline_rows)}/{len(models)}")
    print(f"Main table:   {reports_root / 'model_comparison.csv'}")
    print(f"Per relation: {reports_root / 'per_relation.csv'}")
    print(f"Per sample:   {reports_root / 'sample_comparison.csv'}")
    if errors_path.exists():
        print(f"Errors/skips: {errors_path}")


if __name__ == "__main__":
    main()
