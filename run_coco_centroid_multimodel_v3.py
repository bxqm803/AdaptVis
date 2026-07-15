#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COCO-two centroid comparison across models.

Companion Step-1 script:
    analyze_coco_centroid_step1_v3.py

Default models:
    qwen2-2b
    qwen2-vl-7b
    qwen-3b
    qwen-7b
    internvl-1b
    internvl-2b
    internvl-8b
    llava-7b
    llava-13b

For each model, report:

1. Best hidden-similarity centroid layer.
2. Best all-head-mean attention centroid layer.
3. Best single attention-head centroid layer/head.

All metrics use the same original/swap-aligned average protocol. Models are
frozen. The best layer/head is selected using full-set GT, so these are
diagnostic oracle statistics.

Generation from Step 1 is intentionally not reported. Step 1 forces at least
two decoding steps to expose answer-token routing and is therefore not the
normal free-generation baseline.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "coco-centroid-multimodel-v3"

DEFAULT_MODELS = [
    "qwen2-2b",
    "qwen2-vl-7b",
    "qwen-3b",
    "qwen-7b",
    "internvl-1b",
    "internvl-2b",
    "internvl-8b",
    "llava-7b",
    "llava-13b",
]

RELATIONS = ("left", "right", "above", "below")
CODE_TO_RELATION = {
    index: relation for index, relation in enumerate(RELATIONS)
}

EXTRA_MODEL_SPECS = {
    "qwen2-vl-7b": SimpleNamespace(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        model_class="Qwen2VLForConditionalGeneration",
        dtype_name="bfloat16",
        trust_remote_code=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model aliases.",
    )
    parser.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--step1-script",
        default="analyze_coco_centroid_step1_v3.py",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--report-layer", default="auto")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default="output/coco_centroid_multimodel_v3",
    )
    parser.add_argument("--skip-step1", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    aliases = {
        "left": "left",
        "left of": "left",
        "right": "right",
        "right of": "right",
        "above": "above",
        "over": "above",
        "on": "above",
        "on top of": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
    }
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


def import_two_object_module() -> Any:
    return importlib.import_module("extract_two_object_relation_states")


def merged_model_specs(module: Any) -> Dict[str, Any]:
    specs = dict(getattr(module, "SPECS", {}) or {})
    specs.update(EXTRA_MODEL_SPECS)
    return specs


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
        "20",
        "--output-dir",
        str(output_dir),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def resolve_array_path(
    step1_dir: Path,
    row: Mapping[str, Any],
) -> Path:
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
        f"Could not find arrays for sid={sid}; checked={candidates}"
    )


def relation_from_code(value: Any) -> Optional[str]:
    try:
        return CODE_TO_RELATION.get(int(value))
    except (TypeError, ValueError):
        return None


def select_best_positions(
    aggregate_path: Path,
) -> Dict[str, Any]:
    with np.load(aggregate_path, allow_pickle=False) as aggregate:
        required = [
            "layer_indices",
            "similarity_average_accuracy",
            "headmean_average_accuracy",
            "attention_average_accuracy",
        ]
        missing = [
            key for key in required if key not in aggregate.files
        ]
        if missing:
            raise RuntimeError(
                f"{aggregate_path} is missing {missing}; "
                f"available={aggregate.files}"
            )

        layers = aggregate["layer_indices"].astype(np.int64)
        similarity = aggregate[
            "similarity_average_accuracy"
        ].astype(np.float64)
        headmean = aggregate[
            "headmean_average_accuracy"
        ].astype(np.float64)
        per_head = aggregate[
            "attention_average_accuracy"
        ].astype(np.float64)

    if similarity.ndim != 1:
        raise RuntimeError(
            f"similarity_average_accuracy shape={similarity.shape}"
        )
    if headmean.ndim != 1:
        raise RuntimeError(
            f"headmean_average_accuracy shape={headmean.shape}"
        )
    if per_head.ndim != 2:
        raise RuntimeError(
            f"attention_average_accuracy shape={per_head.shape}"
        )

    similarity_position = int(np.nanargmax(similarity))
    headmean_position = int(np.nanargmax(headmean))
    flat_position = int(np.nanargmax(per_head))
    head_layer_position, head = np.unravel_index(
        flat_position,
        per_head.shape,
    )

    return {
        "similarity_position": similarity_position,
        "similarity_layer": int(layers[similarity_position]),
        "similarity_accuracy": float(similarity[similarity_position]),
        "headmean_position": headmean_position,
        "headmean_layer": int(layers[headmean_position]),
        "headmean_accuracy": float(headmean[headmean_position]),
        "best_head_layer_position": int(head_layer_position),
        "best_head_layer": int(layers[head_layer_position]),
        "best_head": int(head),
        "best_head_accuracy": float(
            per_head[head_layer_position, head]
        ),
    }


def load_selected_predictions(
    *,
    step1_dir: Path,
    sample_rows: Sequence[Mapping[str, Any]],
    positions: Mapping[str, Any],
) -> Dict[int, Dict[str, Optional[str]]]:
    result: Dict[int, Dict[str, Optional[str]]] = {}

    for row in sample_rows:
        sid = int(row["sid"])
        path = resolve_array_path(step1_dir, row)
        with np.load(path, allow_pickle=False) as data:
            required = [
                "similarity_average_prediction",
                "headmean_average_prediction",
                "attention_average_prediction",
            ]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise RuntimeError(
                    f"{path} is missing {missing}. "
                    "The output must be produced by COCO Step-1 v3."
                )

            similarity_code = data[
                "similarity_average_prediction"
            ][positions["similarity_position"]]
            headmean_code = data[
                "headmean_average_prediction"
            ][positions["headmean_position"]]
            best_head_code = data[
                "attention_average_prediction"
            ][
                positions["best_head_layer_position"],
                positions["best_head"],
            ]

        result[sid] = {
            "similarity": relation_from_code(similarity_code),
            "headmean": relation_from_code(headmean_code),
            "best_head": relation_from_code(best_head_code),
        }

    return result


def evaluate_model(
    *,
    model: str,
    step1_dir: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    aggregate_path = step1_dir / "aggregate_metrics.npz"
    samples_path = step1_dir / "samples.jsonl"
    summary_path = step1_dir / "summary.json"

    for path in (aggregate_path, samples_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Step-1 output: {path}")

    sample_rows = read_jsonl(samples_path)
    if not sample_rows:
        raise RuntimeError(f"No samples in {samples_path}")

    positions = select_best_positions(aggregate_path)
    predictions = load_selected_predictions(
        step1_dir=step1_dir,
        sample_rows=sample_rows,
        positions=positions,
    )

    method_names = ("similarity", "headmean", "best_head")
    counts = {name: 0 for name in method_names}
    relation_counts: Dict[str, Dict[str, int]] = {
        relation: {
            "n": 0,
            **{name: 0 for name in method_names},
        }
        for relation in RELATIONS
    }
    sample_output: List[Dict[str, Any]] = []

    for index, row in enumerate(sample_rows, 1):
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        if gt not in RELATIONS:
            raise RuntimeError(
                f"Invalid GT for sid={sid}: {row.get('gt')!r}"
            )

        pred = predictions[sid]
        correct = {
            name: pred[name] == gt
            for name in method_names
        }
        for name in method_names:
            counts[name] += int(correct[name])

        relation_counts[gt]["n"] += 1
        for name in method_names:
            relation_counts[gt][name] += int(correct[name])

        sample_output.append({
            "model": model,
            "index": index,
            "sid": sid,
            "question": row.get("question"),
            "gt": gt,
            "similarity_layer": positions["similarity_layer"],
            "similarity_prediction": pred["similarity"],
            "similarity_correct": correct["similarity"],
            "headmean_attention_layer": positions["headmean_layer"],
            "headmean_attention_prediction": pred["headmean"],
            "headmean_attention_correct": correct["headmean"],
            "best_attention_head_layer": positions[
                "best_head_layer"
            ],
            "best_attention_head": positions["best_head"],
            "best_attention_head_prediction": pred["best_head"],
            "best_attention_head_correct": correct["best_head"],
        })

    n = len(sample_rows)
    headline = {
        "model": model,
        "n": n,
        "similarity_centroid_accuracy": counts["similarity"] / n,
        "similarity_centroid_best_layer": positions[
            "similarity_layer"
        ],
        "headmean_attention_centroid_accuracy": (
            counts["headmean"] / n
        ),
        "headmean_attention_best_layer": positions[
            "headmean_layer"
        ],
        "best_attention_head_centroid_accuracy": (
            counts["best_head"] / n
        ),
        "best_attention_head_layer": positions[
            "best_head_layer"
        ],
        "best_attention_head": positions["best_head"],
        "headmean_minus_similarity": (
            counts["headmean"] - counts["similarity"]
        ) / n,
        "best_head_minus_headmean": (
            counts["best_head"] - counts["headmean"]
        ) / n,
        "selection_note": (
            "best similarity layer, head-mean layer, and single head "
            "selected using full-set GT; diagnostic oracle statistics"
        ),
    }

    per_relation: List[Dict[str, Any]] = []
    for relation in RELATIONS:
        stats = relation_counts[relation]
        relation_n = stats["n"]
        if relation_n == 0:
            continue
        per_relation.append({
            "model": model,
            "relation": relation,
            "n": relation_n,
            "similarity_centroid_accuracy": (
                stats["similarity"] / relation_n
            ),
            "headmean_attention_centroid_accuracy": (
                stats["headmean"] / relation_n
            ),
            "best_attention_head_centroid_accuracy": (
                stats["best_head"] / relation_n
            ),
        })

    return headline, per_relation, sample_output


def print_model_summary(row: Mapping[str, Any]) -> None:
    print("\n" + "=" * 112)
    print(f"MODEL: {row['model']} | n={row['n']}")
    print("=" * 112)
    print(
        "1. Similarity centroid:              "
        f"{row['similarity_centroid_accuracy']:.4f} "
        f"at L{row['similarity_centroid_best_layer']}"
    )
    print(
        "2. All-head mean attention centroid: "
        f"{row['headmean_attention_centroid_accuracy']:.4f} "
        f"at L{row['headmean_attention_best_layer']}"
    )
    print(
        "3. Best single attention head:       "
        f"{row['best_attention_head_centroid_accuracy']:.4f} "
        f"at L{row['best_attention_head_layer']}"
        f"H{row['best_attention_head']}"
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

    module = import_two_object_module()
    specs = merged_model_specs(module)
    print(f"Dataset module: {module.__name__}")

    missing = [model for model in models if model not in specs]
    if missing:
        print(
            "WARNING: unsupported model aliases will be skipped: "
            + ", ".join(missing)
        )

    headline_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for model in models:
        if model not in specs:
            append_jsonl(errors_path, {
                "model": model,
                "stage": "model_validation",
                "error": "Model alias absent from merged model specs",
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
                        f"\n[{model}] complete Step-1 output exists; "
                        "skipping model forward."
                    )
                else:
                    if model_output.exists():
                        raise RuntimeError(
                            f"Partial output exists: {model_output}. "
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

            headline, relation_part, sample_part = evaluate_model(
                model=model,
                step1_dir=model_output,
            )
            headline_rows.append(headline)
            relation_rows.extend(relation_part)
            sample_rows.extend(sample_part)

            print_model_summary(headline)

            write_csv(
                reports_root / "model_comparison.csv",
                headline_rows,
            )
            write_csv(
                reports_root / "per_relation.csv",
                relation_rows,
            )
            write_csv(
                reports_root / "sample_comparison.csv",
                sample_rows,
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
            f"No model completed. See {errors_path}"
        )

    print("\n" + "=" * 112)
    print("COCO-TWO CENTROID SUITE COMPLETE")
    print("=" * 112)
    print(f"Successful models: {len(headline_rows)}/{len(models)}")
    print(f"Main table:   {reports_root / 'model_comparison.csv'}")
    print(f"Per relation: {reports_root / 'per_relation.csv'}")
    print(f"Per sample:   {reports_root / 'sample_comparison.csv'}")
    if errors_path.exists() and errors_path.stat().st_size:
        print(f"Errors/skips: {errors_path}")


if __name__ == "__main__":
    main()
