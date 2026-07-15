#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run normal autoregressive generation on Controlled-A.

This script does NOT recompute similarity centroids or attention centroids.
It only re-runs normal model generation with:

    do_sample=False
    max_new_tokens=<user value>
    no min_new_tokens

Therefore the model may emit EOS immediately after a one-token answer.  The
generated text used for accuracy is not taken from the previous attention
tracing run.

Default model keys:
    qwen2-2b
    qwen-3b
    qwen-7b
    internvl-1b
    internvl-2b
    internvl-8b
    llava-7b
    llava-13b

For every sample it prints:
    Question
    GT
    Pred
    Current ACC
    raw Generation

Outputs:
    <output-root>/<model>/samples.jsonl
    <output-root>/<model>/summary.json
    <output-root>/reports/model_comparison.csv
    <output-root>/reports/per_relation.csv
    <output-root>/reports/errors.jsonl

Optionally, when the previous centroid model_comparison.csv exists, it also
writes:
    <output-root>/reports/centroid_generation_combined.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import random
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "controlledA-normal-generation-only-v1"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys from the Controlled-A extractor SPECS.",
    )
    parser.add_argument(
        "--helper-module",
        default="analyze_controlledA_similarity_head_generation_step1_v1",
        help=(
            "Module containing the already-tested Controlled-A prompt/image/model "
            "helper functions."
        ),
    )
    parser.add_argument(
        "--controlled-module",
        default="",
        help=(
            "Optional explicit Controlled-A extractor module. Otherwise the "
            "helper tries the known Controlled-A extractor module names."
        ),
    )
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument("--dataset-key", default="Controlled_Images_A")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default="output/controlledA_normal_generation_multimodel",
    )
    parser.add_argument(
        "--centroid-report",
        default=(
            "output/controlledA_similarity_head_generation_multimodel/"
            "reports/model_comparison.csv"
        ),
        help=(
            "Optional previous centroid summary to merge with the corrected "
            "normal generation results."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing generation-only output and run from scratch.",
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
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    exact = {
        "left": "left",
        "left of": "left",
        "to the left": "left",
        "to the left of": "left",
        "right": "right",
        "right of": "right",
        "to the right": "right",
        "to the right of": "right",
        "on": "on",
        "above": "on",
        "over": "on",
        "on top": "on",
        "on top of": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
    }
    if text in exact:
        return exact[text]

    import re

    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(under|below|beneath)\b", "under"),
        (r"\b(on top of|on top|above|over)\b", "on"),
        (r"\bon\b", "on"),
    ]
    hits: List[Tuple[int, str]] = []
    for pattern, label in patterns:
        match = re.search(pattern, text)
        if match:
            hits.append((match.start(), label))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    return hits[0][1]


def extract_all_relations(value: Any) -> List[str]:
    """Return all distinct relation labels in textual order."""
    if value is None:
        return []
    import re

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(under|below|beneath)\b", "under"),
        (r"\b(on top of|on top|above|over)\b", "on"),
        (r"\bon\b", "on"),
    ]
    matches: List[Tuple[int, str]] = []
    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), label))
    matches.sort(key=lambda item: item[0])

    result: List[str] = []
    for _, label in matches:
        if label not in result:
            result.append(label)
    return result


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def one_line(value: Any) -> str:
    return " ".join(str(value).split())


def load_helper(module_name: str):
    module = importlib.import_module(module_name)
    required = [
        "import_controlled_module",
        "load_standard_prompts",
        "record_image",
        "make_question_batch",
        "resolve_dtype",
        "configure_processor",
        "generate_text",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"Helper module {module_name!r} is missing functions: {missing}"
        )
    return module


def safe_remove_batch(batch: Optional[Dict[str, Any]]) -> None:
    if not batch:
        return
    for value in batch.values():
        if torch.is_tensor(value):
            del value
    batch.clear()


def load_model_and_processor(
    *,
    helper: Any,
    spec: Any,
    model_key: str,
    device: str,
) -> Tuple[Any, Any]:
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no class {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": helper.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": device},
    }

    print(f"\nLoading {model_key}: {spec.repo_id}", flush=True)
    model = model_cls.from_pretrained(
        spec.repo_id,
        **load_kwargs,
    )
    model.eval()

    # Greedy decoding does not use sampling controls.  Some checkpoints store
    # non-default sampling values and transformers emits a warning even with
    # do_sample=False.  Clearing them does not alter greedy decoding.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    helper.configure_processor(model, processor)
    return model, processor


def compute_model_summary(
    *,
    model_key: str,
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    valid_rows = [
        row for row in rows
        if row.get("status") == "ok"
    ]
    if not valid_rows:
        raise RuntimeError(f"No successful rows for model={model_key}")

    n = len(valid_rows)
    correct = sum(bool(row.get("correct")) for row in valid_rows)
    parse_valid = sum(
        row.get("prediction") in RELATIONS
        for row in valid_rows
    )
    multi_relation = sum(
        len(row.get("all_relations", [])) > 1
        for row in valid_rows
    )
    strict_single_correct = sum(
        len(row.get("all_relations", [])) == 1
        and row.get("all_relations", [None])[0] == row.get("gt")
        for row in valid_rows
    )
    contains_gt = sum(
        row.get("gt") in row.get("all_relations", [])
        for row in valid_rows
    )

    summary = {
        "model": model_key,
        "n": n,
        "normal_generation_accuracy": correct / n,
        "parse_valid_rate": parse_valid / n,
        "strict_single_label_accuracy": strict_single_correct / n,
        "contains_gt_accuracy": contains_gt / n,
        "multi_relation_rate": multi_relation / n,
        "n_correct": correct,
        "n_parse_valid": parse_valid,
        "n_multi_relation": multi_relation,
        "parser": "first relation in generated text",
        "generation_mode": (
            "greedy; do_sample=False; no min_new_tokens; "
            "model may stop at EOS naturally"
        ),
    }

    relation_rows: List[Dict[str, Any]] = []
    for relation in RELATIONS:
        selected = [
            row for row in valid_rows
            if row.get("gt") == relation
        ]
        if not selected:
            continue
        relation_rows.append({
            "model": model_key,
            "relation": relation,
            "n": len(selected),
            "normal_generation_accuracy": sum(
                bool(row.get("correct"))
                for row in selected
            ) / len(selected),
            "strict_single_label_accuracy": sum(
                len(row.get("all_relations", [])) == 1
                and row.get("all_relations", [None])[0] == relation
                for row in selected
            ) / len(selected),
            "contains_gt_accuracy": sum(
                relation in row.get("all_relations", [])
                for row in selected
            ) / len(selected),
            "multi_relation_rate": sum(
                len(row.get("all_relations", [])) > 1
                for row in selected
            ) / len(selected),
        })

    return summary, relation_rows


def merge_centroid_report(
    *,
    centroid_path: Path,
    generation_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not centroid_path.exists():
        return

    with centroid_path.open("r", encoding="utf-8") as handle:
        centroid_rows = list(csv.DictReader(handle))

    generation_by_model = {
        row["model"]: row for row in generation_rows
    }
    combined: List[Dict[str, Any]] = []

    for centroid_row in centroid_rows:
        model = centroid_row.get("model")
        generation_row = generation_by_model.get(model)
        if generation_row is None:
            continue

        combined.append({
            "model": model,
            "n": generation_row["n"],
            "similarity_centroid_accuracy": centroid_row.get(
                "similarity_centroid_accuracy"
            ),
            "similarity_centroid_best_layer": centroid_row.get(
                "similarity_centroid_best_layer"
            ),
            "best_attention_head_centroid_accuracy": centroid_row.get(
                "best_attention_head_centroid_accuracy"
            ),
            "best_attention_head_layer": centroid_row.get(
                "best_attention_head_layer"
            ),
            "best_attention_head": centroid_row.get(
                "best_attention_head"
            ),
            "corrected_normal_generation_accuracy": generation_row[
                "normal_generation_accuracy"
            ],
            "strict_single_label_accuracy": generation_row[
                "strict_single_label_accuracy"
            ],
            "contains_gt_accuracy": generation_row[
                "contains_gt_accuracy"
            ],
            "multi_relation_rate": generation_row[
                "multi_relation_rate"
            ],
            "similarity_minus_corrected_generation": (
                float(centroid_row["similarity_centroid_accuracy"])
                - float(generation_row["normal_generation_accuracy"])
            ),
            "best_head_minus_corrected_generation": (
                float(
                    centroid_row[
                        "best_attention_head_centroid_accuracy"
                    ]
                )
                - float(generation_row["normal_generation_accuracy"])
            ),
        })

    write_csv(output_path, combined)


def run_one_model(
    *,
    args: argparse.Namespace,
    helper: Any,
    controlled_module: Any,
    records: Sequence[Any],
    prompt_rows: Dict[int, Dict[str, Any]],
    model_key: str,
    model_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if model_key not in controlled_module.SPECS:
        raise ValueError(
            f"Model {model_key!r} not found in "
            f"{controlled_module.__name__}.SPECS"
        )

    if args.overwrite and model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples_path = model_dir / "samples.jsonl"
    errors_path = model_dir / "errors.jsonl"
    summary_path = model_dir / "summary.json"
    config_path = model_dir / "config.json"

    existing_rows = [
        row for row in read_jsonl(samples_path)
        if row.get("status") == "ok"
    ]
    done_sids = {
        int(row["sid"]) for row in existing_rows
    }

    spec = controlled_module.SPECS[model_key]
    config = {
        "script_version": SCRIPT_VERSION,
        "model": model_key,
        "repo_id": spec.repo_id,
        "prompt_jsonl": args.prompt_jsonl,
        "dataset_key": args.dataset_key,
        "controlled_module": controlled_module.__name__,
        "helper_module": args.helper_module,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "min_new_tokens": None,
        "normal_generation": True,
        "n_records": len(records),
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if len(done_sids) == len(records) and summary_path.exists():
        print(
            f"\n[{model_key}] complete generation output exists; "
            "reusing it."
        )
        rows = read_jsonl(samples_path)
        summary, relation_rows = compute_model_summary(
            model_key=model_key,
            rows=rows,
        )
        return summary, relation_rows

    model = None
    processor = None
    model, processor = load_model_and_processor(
        helper=helper,
        spec=spec,
        model_key=model_key,
        device=args.device,
    )
    device = torch.device(args.device)

    running_rows = {
        int(row["sid"]): row for row in existing_rows
    }
    running_correct = sum(
        bool(row.get("correct"))
        for row in existing_rows
    )
    running_total = len(existing_rows)
    started = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"normal-generation:controlled_A:{model_key}",
        ):
            sid = int(record.sid)
            if sid in done_sids:
                continue

            image = None
            batch = None
            try:
                prompt_row = prompt_rows[sid]
                question = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATIONS:
                    raise ValueError(
                        f"Unsupported GT for sid={sid}: "
                        f"{prompt_row['answer_raw']!r}"
                    )

                image = helper.record_image(record)
                batch = helper.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )

                # This is the corrected normal generation call:
                # no min_new_tokens and no attention-tracing requirement.
                generated_text = helper.generate_text(
                    model=model,
                    processor=processor,
                    batch=batch,
                    max_new_tokens=args.max_new_tokens,
                )

                prediction = normalize_relation(generated_text)
                all_relations = extract_all_relations(generated_text)
                correct = prediction == gt

                running_total += 1
                running_correct += int(correct)
                running_accuracy = running_correct / running_total

                row = {
                    "status": "ok",
                    "model": model_key,
                    "sid": sid,
                    "question": question,
                    "gt": gt,
                    "prediction": prediction,
                    "all_relations": all_relations,
                    "correct": correct,
                    "strict_single_correct": (
                        len(all_relations) == 1
                        and all_relations[0] == gt
                    ),
                    "contains_gt": gt in all_relations,
                    "multi_relation": len(all_relations) > 1,
                    "generated_text": generated_text,
                    "running_correct": running_correct,
                    "running_total": running_total,
                    "running_accuracy": running_accuracy,
                }
                append_jsonl(samples_path, row)
                running_rows[sid] = row

                tqdm.write(
                    f"\n[GEN {running_total}/{len(records)}] "
                    f"model={model_key} | sid={sid}\n"
                    f"  Question: {one_line(question)}\n"
                    f"  GT: {gt}\n"
                    f"  Pred: {prediction or '<invalid>'}\n"
                    f"  Current ACC: "
                    f"{running_correct}/{running_total} "
                    f"= {running_accuracy:.4f}\n"
                    f"  Generation: {generated_text!r}"
                )

            except Exception as exc:
                error = {
                    "status": "error",
                    "model": model_key,
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                }
                append_jsonl(errors_path, error)
                print(
                    f"\n[ERROR] model={model_key} sid={sid}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if args.stop_on_error:
                    raise
            finally:
                safe_remove_batch(batch)
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        elapsed = time.time() - started
        print(
            f"\n[{model_key}] elapsed={elapsed / 60.0:.2f} min",
            flush=True,
        )
        del processor
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = list(running_rows.values())
    rows.sort(key=lambda row: int(row["sid"]))
    summary, relation_rows = compute_model_summary(
        model_key=model_key,
        rows=rows,
    )
    summary["elapsed_seconds"] = elapsed
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(f"MODEL: {model_key} | n={summary['n']}")
    print("=" * 100)
    print(
        f"Normal generation ACC:     "
        f"{summary['normal_generation_accuracy']:.4f}"
    )
    print(
        f"Parse valid rate:          "
        f"{summary['parse_valid_rate']:.4f}"
    )
    print(
        f"Strict single-label ACC:   "
        f"{summary['strict_single_label_accuracy']:.4f}"
    )
    print(
        f"Contains-GT rate:          "
        f"{summary['contains_gt_accuracy']:.4f}"
    )
    print(
        f"Multi-relation rate:       "
        f"{summary['multi_relation_rate']:.4f}"
    )

    return summary, relation_rows


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = parse_models(args.models)
    helper = load_helper(args.helper_module)
    controlled_module = helper.import_controlled_module(
        args.controlled_module
    )

    prompt_path = Path(args.prompt_jsonl)
    records, audit = controlled_module.load_records(
        prompt_path,
        dataset_key=args.dataset_key,
        keep_relations=list(RELATIONS),
        download=args.download,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    if not records:
        raise RuntimeError("No usable Controlled-A records")

    prompt_rows = helper.load_standard_prompts(prompt_path)
    missing_sids = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing_sids:
        raise RuntimeError(
            f"Prompt file is missing {len(missing_sids)} record IDs; "
            f"first={missing_sids[:10]}"
        )

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    suite_errors_path = reports_dir / "errors.jsonl"

    if args.overwrite and suite_errors_path.exists():
        suite_errors_path.unlink()

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Controlled module: {controlled_module.__name__}")
    print(f"Records: {len(records)}")
    print(
        "Generation: greedy, do_sample=False, "
        f"max_new_tokens={args.max_new_tokens}, "
        "no min_new_tokens"
    )

    model_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []

    for model_key in models:
        try:
            summary, model_relation_rows = run_one_model(
                args=args,
                helper=helper,
                controlled_module=controlled_module,
                records=records,
                prompt_rows=prompt_rows,
                model_key=model_key,
                model_dir=output_root / model_key,
            )
            model_rows.append(summary)
            relation_rows.extend(model_relation_rows)

            # Save after every completed model.
            write_csv(
                reports_dir / "model_comparison.csv",
                model_rows,
            )
            write_csv(
                reports_dir / "per_relation.csv",
                relation_rows,
            )
            (reports_dir / "model_comparison.json").write_text(
                json.dumps(
                    model_rows,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            centroid_path = Path(args.centroid_report)
            merge_centroid_report(
                centroid_path=centroid_path,
                generation_rows=model_rows,
                output_path=(
                    reports_dir / "centroid_generation_combined.csv"
                ),
            )

        except Exception as exc:
            append_jsonl(suite_errors_path, {
                "model": model_key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-24:],
            })
            print(
                f"\n[MODEL ERROR] {model_key}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                raise

    if not model_rows:
        raise RuntimeError(
            f"No model completed. See {suite_errors_path}"
        )

    print("\n" + "=" * 112)
    print("CORRECTED CONTROLLED-A NORMAL GENERATION COMPLETE")
    print("=" * 112)
    print(f"Successful models: {len(model_rows)}/{len(models)}")
    print(
        f"Generation table: "
        f"{reports_dir / 'model_comparison.csv'}"
    )
    print(
        f"Per relation:     "
        f"{reports_dir / 'per_relation.csv'}"
    )
    combined_path = reports_dir / "centroid_generation_combined.csv"
    if combined_path.exists():
        print(f"Combined table:   {combined_path}")
    if suite_errors_path.exists() and suite_errors_path.stat().st_size:
        print(f"Errors/skips:     {suite_errors_path}")


if __name__ == "__main__":
    main()
