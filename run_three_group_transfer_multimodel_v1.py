#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end three-group spatial transfer analysis for multiple VLMs.

This driver DOES NOT require an existing prior directory.

For each model it performs:

Pass 1
------
Run analyze_coco_centroid_generation_step1_v4.py from scratch to obtain:
- normal greedy generation;
- hidden-similarity centroids at every decoder layer;
- attention centroids and auxiliary routing arrays.

It selects the best hidden-similarity average-centroid layer and divides samples into:
A: centroid correct + generation correct
B: centroid correct + generation wrong
C: centroid wrong   + generation correct
D: centroid wrong   + generation wrong (optional in the downstream trace)

Pass 2
------
Create a compatibility cache from the freshly produced Pass-1 files, then run
compare_centroid_generation_transfer_groups_v1_1.py to trace A/B/C through
the frozen model and compare:
- last -> subject/reference/visual attention;
- isolated A*V and o_proj routed contributions;
- layerwise GT-vs-opposite logit margins;
- attention/MLP/block margin gains;
- matched A-vs-B, A-vs-C, and B-vs-C comparisons.

The compatibility cache is generated inside the current output directory and is
not an external prerequisite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


SCRIPT_VERSION = "run-three-group-transfer-multimodel-v1"
RELATIONS = ("left", "right", "above", "below")
CODE_TO_RELATION = {i: relation for i, relation in enumerate(RELATIONS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models",
        default="llava-7b,qwen-3b,qwen-7b",
        help="Comma-separated model aliases.",
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--step1-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument(
        "--compare-script",
        default="compare_centroid_generation_transfer_groups_v1_1.py",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layers", default="all")
    p.add_argument("--report-layer", default="auto")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional dataset cap used in Pass 1.",
    )
    p.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional A/B/C per-group cap used in Pass 2.",
    )
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--print-every-step1", type=int, default=20)
    p.add_argument("--print-every-trace", type=int, default=10)
    p.add_argument(
        "--output-root",
        default="output/three_group_transfer_fresh/coco",
    )
    p.add_argument(
        "--centroid-source",
        default="similarity_average",
        choices=[
            "similarity_average",
            "similarity_original",
            "headmean_average",
            "best_attention_head",
        ],
        help=(
            "Centroid used to form groups. similarity_average is the default "
            "object-token/visual-token hidden-similarity centroid averaged over "
            "original and swapped prompts."
        ),
    )
    p.add_argument(
        "--selected-head-mode",
        default="all_at_centroid_layer",
        choices=["all_at_centroid_layer", "top_attention_heads"],
        help="Heads retained for the detailed selected-head trace in Pass 2.",
    )
    p.add_argument("--top-heads", type=int, default=20)
    p.add_argument("--include-group-d", action="store_true")
    p.add_argument("--make-plots", action="store_true")
    p.add_argument(
        "--reuse-step1",
        action="store_true",
        help="Reuse complete Pass-1 output. Default is to rerun it.",
    )
    p.add_argument(
        "--reuse-trace",
        action="store_true",
        help="Reuse an existing trace directory instead of rerunning Pass 2.",
    )
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


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
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    aliases = {
        "left": "left",
        "left of": "left",
        "to the left": "left",
        "to the left of": "left",
        "right": "right",
        "right of": "right",
        "to the right": "right",
        "to the right of": "right",
        "above": "above",
        "over": "above",
        "on": "above",
        "on top of": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
    }
    if text in aliases:
        return aliases[text]
    for token, relation in (
        ("left", "left"),
        ("right", "right"),
        ("below", "below"),
        ("under", "below"),
        ("above", "above"),
        ("over", "above"),
    ):
        if token in text.split():
            return relation
    return None


def relation_from_code(value: Any) -> Optional[str]:
    try:
        return CODE_TO_RELATION.get(int(value))
    except (TypeError, ValueError):
        return None


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
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in command)
    print("\n$ " + printable, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + printable + "\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            env=dict(env) if env is not None else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, list(command))


def resolve_array_path(step1_dir: Path, row: Mapping[str, Any]) -> Path:
    sid = int(row["sid"])
    candidates = [step1_dir / "sample_arrays" / f"{sid}.npz"]
    supplied = row.get("array_file")
    if supplied:
        supplied_path = Path(str(supplied))
        candidates.extend(
            [
                supplied_path,
                step1_dir / supplied_path,
                step1_dir / supplied_path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find per-sample arrays for sid={sid}; checked={candidates}"
    )


def safe_scalar(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def axis_confidence_from_centroids(centroids: np.ndarray) -> float:
    """
    centroids: [2, 2], ordered [subject, reference], coordinate [x, y].
    """
    array = np.asarray(centroids, dtype=np.float64)
    if array.shape != (2, 2):
        return float("nan")
    dx = float(array[0, 0] - array[1, 0])
    dy = float(array[0, 1] - array[1, 1])
    ax, ay = abs(dx), abs(dy)
    return abs(ax - ay) / (ax + ay + 1e-8)


def centroid_delta_from_centroids(centroids: np.ndarray) -> tuple[float, float]:
    array = np.asarray(centroids, dtype=np.float64)
    if array.shape != (2, 2):
        return float("nan"), float("nan")
    return (
        float(array[0, 0] - array[1, 0]),
        float(array[0, 1] - array[1, 1]),
    )


def select_centroid_position(
    aggregate_path: Path,
    source: str,
) -> Dict[str, Any]:
    with np.load(aggregate_path, allow_pickle=False) as data:
        layers = data["layer_indices"].astype(np.int64)

        if source == "similarity_average":
            accuracy = data["similarity_average_accuracy"].astype(np.float64)
            position = int(np.nanargmax(accuracy))
            return {
                "source": source,
                "layer_position": position,
                "layer": int(layers[position]),
                "head": None,
                "accuracy": float(accuracy[position]),
            }

        if source == "similarity_original":
            accuracy = data["similarity_original_accuracy"].astype(np.float64)
            position = int(np.nanargmax(accuracy))
            return {
                "source": source,
                "layer_position": position,
                "layer": int(layers[position]),
                "head": None,
                "accuracy": float(accuracy[position]),
            }

        if source == "headmean_average":
            accuracy = data["headmean_average_accuracy"].astype(np.float64)
            position = int(np.nanargmax(accuracy))
            return {
                "source": source,
                "layer_position": position,
                "layer": int(layers[position]),
                "head": None,
                "accuracy": float(accuracy[position]),
            }

        if source == "best_attention_head":
            accuracy = data["attention_average_accuracy"].astype(np.float64)
            flat = int(np.nanargmax(accuracy))
            layer_position, head = np.unravel_index(flat, accuracy.shape)
            return {
                "source": source,
                "layer_position": int(layer_position),
                "layer": int(layers[layer_position]),
                "head": int(head),
                "accuracy": float(accuracy[layer_position, head]),
            }

    raise ValueError(f"Unsupported centroid source: {source}")


def selected_heads_from_step1(
    *,
    aggregate_path: Path,
    summary_path: Path,
    centroid_selection: Mapping[str, Any],
    mode: str,
    top_heads: int,
) -> List[Dict[str, int]]:
    with np.load(aggregate_path, allow_pickle=False) as data:
        attention_accuracy = data["attention_average_accuracy"]
        n_heads = int(attention_accuracy.shape[1])

    if mode == "all_at_centroid_layer":
        layer = int(centroid_selection["layer"])
        return [{"layer": layer, "head": head} for head in range(n_heads)]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get("top_attention_heads_by_accuracy", [])
    selected: List[Dict[str, int]] = []
    for row in source[: max(1, int(top_heads))]:
        key = (int(row["layer"]), int(row["head"]))
        if key not in {(x["layer"], x["head"]) for x in selected}:
            selected.append({"layer": key[0], "head": key[1]})
    if selected:
        return selected

    # Conservative fallback.
    accuracy = np.asarray(attention_accuracy, dtype=np.float64)
    with np.load(aggregate_path, allow_pickle=False) as data:
        layers = data["layer_indices"].astype(np.int64)
    order = np.argsort(accuracy.reshape(-1))[::-1]
    for flat in order[: max(1, int(top_heads))]:
        layer_pos, head = np.unravel_index(int(flat), accuracy.shape)
        selected.append({"layer": int(layers[layer_pos]), "head": int(head)})
    return selected


def extract_sample_centroid(
    *,
    data: Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    source = str(selection["source"])
    pos = int(selection["layer_position"])
    head = selection.get("head")

    if source == "similarity_average":
        prediction = relation_from_code(data["similarity_average_prediction"][pos])

        original_centroids = np.asarray(
            data.get("original_similarity_centroids", np.empty((0,))),
        )
        swapped_centroids = np.asarray(
            data.get("swapped_similarity_centroids", np.empty((0,))),
        )
        # Step-1 file versions differ in key naming. Use direct average-centroid
        # storage when available; otherwise retain NaN diagnostics.
        average_candidates = (
            "average_similarity_centroids",
            "similarity_average_centroids",
        )
        centroids = None
        for key in average_candidates:
            if key in data:
                centroids = np.asarray(data[key][pos], dtype=np.float64)
                break
        if centroids is None and (
            original_centroids.ndim >= 3 and swapped_centroids.ndim >= 3
        ):
            # swapped prompt objects are reversed; align back to subject/reference.
            centroids = 0.5 * (
                original_centroids[pos]
                + swapped_centroids[pos][[1, 0]]
            )

        axis = (
            axis_confidence_from_centroids(centroids)
            if centroids is not None
            else float("nan")
        )
        dx, dy = (
            centroid_delta_from_centroids(centroids)
            if centroids is not None
            else (float("nan"), float("nan"))
        )
        separation = float("nan")
        for key in (
            "original_similarity_separation",
            "similarity_separation_original",
        ):
            if key in data:
                separation = safe_scalar(data[key][pos])
                break
        return {
            "prediction": prediction,
            "axis_confidence": axis,
            "delta_x": dx,
            "delta_y": dy,
            "mean_separation": separation,
        }

    if source == "similarity_original":
        prediction = relation_from_code(data["original_similarity_prediction"][pos])
        axis = float("nan")
        for key in (
            "original_similarity_axis_confidence",
            "similarity_axis_confidence_original",
        ):
            if key in data:
                axis = safe_scalar(data[key][pos])
                break
        centroids = None
        for key in (
            "original_similarity_centroids",
            "similarity_centroids_original",
        ):
            if key in data:
                centroids = np.asarray(data[key][pos], dtype=np.float64)
                break
        dx, dy = (
            centroid_delta_from_centroids(centroids)
            if centroids is not None
            else (float("nan"), float("nan"))
        )
        separation = float("nan")
        for key in (
            "original_similarity_separation",
            "similarity_separation_original",
        ):
            if key in data:
                separation = safe_scalar(data[key][pos])
                break
        return {
            "prediction": prediction,
            "axis_confidence": axis,
            "delta_x": dx,
            "delta_y": dy,
            "mean_separation": separation,
        }

    if source == "headmean_average":
        prediction = relation_from_code(data["headmean_average_prediction"][pos])
        return {
            "prediction": prediction,
            "axis_confidence": float("nan"),
            "delta_x": float("nan"),
            "delta_y": float("nan"),
            "mean_separation": float("nan"),
        }

    if source == "best_attention_head":
        if head is None:
            raise RuntimeError("best_attention_head selection has no head")
        prediction = relation_from_code(
            data["attention_average_prediction"][pos, int(head)]
        )
        return {
            "prediction": prediction,
            "axis_confidence": float("nan"),
            "delta_x": float("nan"),
            "delta_y": float("nan"),
            "mean_separation": float("nan"),
        }

    raise ValueError(source)


def build_compatibility_prior(
    *,
    step1_dir: Path,
    prior_dir: Path,
    centroid_source: str,
    selected_head_mode: str,
    top_heads: int,
) -> Dict[str, Any]:
    aggregate_path = step1_dir / "aggregate_metrics.npz"
    samples_path = step1_dir / "samples.jsonl"
    summary_path = step1_dir / "summary.json"
    for path in (aggregate_path, samples_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing fresh Pass-1 output: {path}")

    selection = select_centroid_position(aggregate_path, centroid_source)
    selected_heads = selected_heads_from_step1(
        aggregate_path=aggregate_path,
        summary_path=summary_path,
        centroid_selection=selection,
        mode=selected_head_mode,
        top_heads=top_heads,
    )

    sample_rows = read_jsonl(samples_path)
    centroid_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    group_counts = {
        "A_centroid_correct_generation_correct": 0,
        "B_centroid_correct_generation_wrong": 0,
        "C_centroid_wrong_generation_correct": 0,
        "D_centroid_wrong_generation_wrong": 0,
    }

    for row in sample_rows:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        if gt not in RELATIONS:
            continue

        array_path = resolve_array_path(step1_dir, row)
        with np.load(array_path, allow_pickle=False) as sample_data:
            centroid = extract_sample_centroid(
                data=sample_data,
                selection=selection,
            )

        centroid_prediction = normalize_relation(centroid["prediction"])
        generation_prediction = normalize_relation(
            row.get("original_prediction")
        )
        centroid_correct = centroid_prediction == gt
        generation_correct = generation_prediction == gt

        if centroid_correct and generation_correct:
            group_counts["A_centroid_correct_generation_correct"] += 1
        elif centroid_correct and not generation_correct:
            group_counts["B_centroid_correct_generation_wrong"] += 1
        elif not centroid_correct and generation_correct:
            group_counts["C_centroid_wrong_generation_correct"] += 1
        else:
            group_counts["D_centroid_wrong_generation_wrong"] += 1

        centroid_rows.append(
            {
                "sid": sid,
                "gt": gt,
                "centroid_source": centroid_source,
                "centroid_layer": int(selection["layer"]),
                "centroid_head": selection.get("head"),
                "centroid_prediction": centroid_prediction,
                "centroid_correct": centroid_correct,
                # Use axis confidence as the primary continuous confidence.
                "centroid_confidence": safe_scalar(
                    centroid.get("axis_confidence")
                ),
                "axis_confidence": safe_scalar(
                    centroid.get("axis_confidence")
                ),
                "head_agreement": float("nan"),
                "swap_stability": (
                    1.0 if bool(row.get("answer_swap_consistent")) else 0.0
                ),
                "mean_separation": safe_scalar(
                    centroid.get("mean_separation")
                ),
                "mean_visual_mass": safe_scalar(
                    row.get("report_first_answer_visual_mass_mean")
                ),
                "delta_x": safe_scalar(centroid.get("delta_x")),
                "delta_y": safe_scalar(centroid.get("delta_y")),
                "head_rows": selected_heads,
            }
        )
        generation_rows.append(
            {
                "sid": sid,
                "gt": gt,
                "baseline_prediction": generation_prediction,
                "baseline_correct": generation_correct,
                "generation_text": row.get("original_generated_text"),
                "lm_relation_margin": safe_scalar(
                    row.get("original_relation_gt_margin")
                ),
                "lm_relation_top": generation_prediction,
            }
        )

    if not centroid_rows:
        raise RuntimeError(f"No valid samples found in {samples_path}")

    if prior_dir.exists():
        shutil.rmtree(prior_dir)
    prior_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(prior_dir / "centroid_analysis.jsonl", centroid_rows)
    write_jsonl(prior_dir / "generation.jsonl", generation_rows)
    config = {
        "script_version": SCRIPT_VERSION,
        "generated_from_fresh_step1": str(step1_dir),
        "centroid_selection": selection,
        "selected_head_mode": selected_head_mode,
        "selected_heads": selected_heads,
        "group_counts": group_counts,
    }
    (prior_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def step1_complete(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in ("summary.json", "aggregate_metrics.npz", "samples.jsonl")
    ) and (path / "sample_arrays").is_dir()


def trace_complete(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in (
            "summary.json",
            "trace_arrays.npz",
            "sample_metadata.jsonl",
        )
    )


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)

    step1_script = Path(args.step1_script)
    compare_script = Path(args.compare_script)
    if not step1_script.exists():
        raise FileNotFoundError(step1_script)
    if not compare_script.exists():
        raise FileNotFoundError(compare_script)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    errors_path = output_root / "errors.jsonl"
    commands_path = output_root / "commands.log"
    model_summary_rows: List[Dict[str, Any]] = []

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for model in models:
        model_root = output_root / model
        step1_dir = model_root / "pass1_centroid_generation"
        prior_dir = model_root / "fresh_group_cache"
        trace_dir = model_root / "pass2_transfer_trace"
        step1_log = model_root / "pass1.log"
        trace_log = model_root / "pass2.log"

        try:
            print("\n" + "=" * 112)
            print(f"MODEL: {model}")
            print("=" * 112)

            if args.reuse_step1 and step1_complete(step1_dir):
                print(f"[Pass 1] Reusing complete output: {step1_dir}")
            else:
                if step1_dir.exists():
                    shutil.rmtree(step1_dir)

                command = [
                    args.python,
                    str(step1_script),
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
                    "eager",
                    "--layers",
                    args.layers,
                    "--report-layer",
                    args.report_layer,
                    "--temperature",
                    str(args.temperature),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--print-every",
                    str(args.print_every_step1),
                    "--output-dir",
                    str(step1_dir),
                    "--overwrite",
                ]
                if args.max_samples is not None:
                    command.extend(["--max-samples", str(args.max_samples)])

                with commands_path.open("a", encoding="utf-8") as handle:
                    handle.write("$ " + " ".join(command) + "\n")
                run_command(command, log_path=step1_log, env=env)

            config = build_compatibility_prior(
                step1_dir=step1_dir,
                prior_dir=prior_dir,
                centroid_source=args.centroid_source,
                selected_head_mode=args.selected_head_mode,
                top_heads=args.top_heads,
            )

            print(
                "[Groups] "
                + ", ".join(
                    f"{name.split('_')[0]}={count}"
                    for name, count in config["group_counts"].items()
                )
            )
            print(
                "[Centroid] "
                f"{config['centroid_selection']['source']} "
                f"L{config['centroid_selection']['layer']} "
                f"acc={config['centroid_selection']['accuracy']:.4f}"
            )

            if args.reuse_trace and trace_complete(trace_dir):
                print(f"[Pass 2] Reusing complete output: {trace_dir}")
            else:
                if trace_dir.exists():
                    shutil.rmtree(trace_dir)

                command = [
                    args.python,
                    str(compare_script),
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
                    "eager",
                    "--prior-dir",
                    str(prior_dir),
                    "--output-dir",
                    str(trace_dir),
                    "--bootstrap-samples",
                    str(args.bootstrap_samples),
                    "--print-every",
                    str(args.print_every_trace),
                    "--overwrite",
                ]
                if args.max_per_group is not None:
                    command.extend(
                        ["--max-per-group", str(args.max_per_group)]
                    )
                if args.include_group_d:
                    command.append("--include-group-d")
                if args.make_plots:
                    command.append("--make-plots")

                with commands_path.open("a", encoding="utf-8") as handle:
                    handle.write("$ " + " ".join(command) + "\n")
                run_command(command, log_path=trace_log, env=env)

            model_summary_rows.append(
                {
                    "model": model,
                    "centroid_source": config["centroid_selection"]["source"],
                    "centroid_layer": config["centroid_selection"]["layer"],
                    "centroid_head": config["centroid_selection"].get("head"),
                    "centroid_accuracy": config["centroid_selection"]["accuracy"],
                    **config["group_counts"],
                    "step1_dir": str(step1_dir),
                    "trace_dir": str(trace_dir),
                    "status": "complete",
                }
            )
            write_csv(output_root / "model_summary.csv", model_summary_rows)
            print(f"[DONE] {model}")

        except Exception as exc:
            error = {
                "model": model,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-24:],
            }
            append_jsonl(errors_path, error)
            model_summary_rows.append(
                {
                    "model": model,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            write_csv(output_root / "model_summary.csv", model_summary_rows)
            print(
                f"[ERROR] {model}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                raise

    completed = [row for row in model_summary_rows if row["status"] == "complete"]
    if not completed:
        raise RuntimeError(f"No model completed. Inspect {errors_path}")

    print("\n" + "=" * 112)
    print("FRESH THREE-GROUP TRANSFER ANALYSIS COMPLETE")
    print("=" * 112)
    print(f"Completed: {len(completed)}/{len(models)}")
    print(f"Summary: {output_root / 'model_summary.csv'}")
    if errors_path.exists() and errors_path.stat().st_size:
        print(f"Errors:  {errors_path}")


if __name__ == "__main__":
    main()
