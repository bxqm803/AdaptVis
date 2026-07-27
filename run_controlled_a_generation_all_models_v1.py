#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation-only benchmark on the original Controlled_Images_A dataset.

This script does NOT extract hidden states and does NOT compute centroid probes.
It only:

1. Loads Controlled-A through the original repository path:
       extract_controlledA_relation_states_standalone.py::load_records
       -> dataset_zoo.get_dataset("Controlled_Images_A")
2. Loads the standard Controlled-A questions:
       prompts/Controlled_Images_A_with_answer_four_options.jsonl
3. Runs deterministic autoregressive generation.
4. Prints and saves the complete generated text for every sample.
5. Records:
       pred, question, gt, generation, acc

Generation length
-----------------
The old combined benchmark used max_new_tokens=6. This standalone script uses
128 by default. Generation still stops early at EOS. A finite safety cap is
necessary to prevent a malformed model from generating indefinitely.

If a sample reaches the cap without EOS, hit_token_limit=1 is saved and a
warning is printed. Increase --max-new-tokens to 256 or 512 if needed.

Models
------
    qwen-3b
    qwen-7b
    qwen2-2b
    llava15-7b
    llava15-13b
    internvl-1b
    internvl-2b
    internvl-8b

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u run_controlled_a_generation_all_models_v1.py \
  --models all \
  --device cuda:0 \
  --max-new-tokens 128 \
  --output-dir output/controlled_a_generation_all_models
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "controlled-a-generation-all-models-v1"

REQUESTED_MODEL_ALIASES = (
    "qwen-3b",
    "qwen-7b",
    "qwen2-2b",
    "llava15-7b",
    "llava15-13b",
    "internvl-1b",
    "internvl-2b",
    "internvl-8b",
)

MODEL_KEY_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "qwen-3b": (
        "qwen-3b",
        "qwen2.5-3b",
        "qwen2_5-3b",
        "qwen25-3b",
        "qwen2.5-vl-3b",
        "qwen2_5_vl_3b",
    ),
    "qwen-7b": (
        "qwen-7b",
        "qwen2.5-7b",
        "qwen2_5-7b",
        "qwen25-7b",
        "qwen2.5-vl-7b",
        "qwen2_5_vl_7b",
    ),
    "qwen2-2b": (
        "qwen2-2b",
        "qwen2vl-2b",
        "qwen2-vl-2b",
        "qwen2_vl_2b",
        "qwen2-2b-instruct",
    ),
    "llava15-7b": (
        "llava-7b",
        "llava15-7b",
        "llava1.5-7b",
        "llava-1.5-7b",
    ),
    "llava15-13b": (
        "llava-13b",
        "llava15-13b",
        "llava1.5-13b",
        "llava-1.5-13b",
    ),
    "internvl-1b": (
        "internvl-1b",
        "internvl2-1b",
        "internvl2.5-1b",
        "internvl25-1b",
    ),
    "internvl-2b": (
        "internvl-2b",
        "internvl2-2b",
        "internvl2.5-2b",
        "internvl25-2b",
    ),
    "internvl-8b": (
        "internvl-8b",
        "internvl2-8b",
        "internvl2.5-8b",
        "internvl25-8b",
    ),
}


@dataclass
class Sample:
    uid: str
    sid: int
    subject: str
    reference: str
    question: str
    gt: str
    record: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default="all",
        help="'all' or comma-separated aliases/registry keys.",
    )
    parser.add_argument(
        "--controlled-script",
        default="extract_controlledA_relation_states_standalone.py",
    )
    parser.add_argument(
        "--controlled-prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    parser.add_argument(
        "--two-object-script",
        default="extract_two_object_relation_states.py",
        help="Used only to obtain the merged repository model registry.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--dtype", default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help=(
            "Safety cap. Generation stops earlier at EOS. Increase this when "
            "hit_token_limit=1 appears."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_model_key(requested: str, specs: Mapping[str, Any]) -> str:
    if requested in specs:
        return requested

    candidates = list(MODEL_KEY_CANDIDATES.get(requested, (requested,)))
    for candidate in candidates:
        if candidate in specs:
            return candidate

    normalized_available = {normalize_key(key): key for key in specs}
    for candidate in candidates:
        normalized = normalize_key(candidate)
        if normalized in normalized_available:
            return normalized_available[normalized]

    matches: List[str] = []
    for candidate in candidates:
        normalized = normalize_key(candidate)
        for available in specs:
            available_normalized = normalize_key(available)
            if normalized == available_normalized:
                matches.append(available)
            elif normalized in available_normalized or available_normalized in normalized:
                matches.append(available)

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Cannot resolve model alias {requested!r}. "
        f"Candidates={candidates}. Available={sorted(specs)}"
    )


def sanitize_uid(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if text:
        return text[:160]
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:20]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def normalize_gt_native(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"left", "right", "on", "under"}:
        return text
    if text in {"above", "over", "top"}:
        return "on"
    if text in {"below", "beneath", "bottom"}:
        return "under"
    return ""


def parse_generation_relation(text: Any) -> str:
    """Parse a Controlled-A answer while avoiding 'on the left' -> on.

    Priority:
      left/right/under/below/above first;
      standalone on only when none of those appears.
    """
    value = str(text).strip().lower()
    value = re.sub(r"[_/|-]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    candidates: List[Tuple[int, str]] = []
    patterns = (
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bbelow\b", "under"),
        (r"\bbeneath\b", "under"),
        (r"\babove\b", "on"),
        (r"\bover\b", "on"),
        (r"\bon top of\b", "on"),
    )
    for pattern, relation in patterns:
        match = re.search(pattern, value)
        if match:
            candidates.append((match.start(), relation))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    if re.search(r"\bon\b", value):
        return "on"
    return ""


def eos_token_ids(model: Any, processor: Any) -> set[int]:
    result: set[int] = set()

    generation_config = getattr(model, "generation_config", None)
    for source in (
        getattr(generation_config, "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(processor.tokenizer, "eos_token_id", None),
    ):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            result.update(int(value) for value in source if value is not None)
        else:
            result.add(int(source))
    return result


def generate_complete_text(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[str, int, bool, List[int]]:
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    input_length = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )

    generated_ids = output_ids[0, input_length:].detach().cpu()
    generated_token_ids = [int(value) for value in generated_ids.tolist()]
    text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    eos_ids = eos_token_ids(model, processor)
    ended_with_eos = bool(
        generated_token_ids
        and generated_token_ids[-1] in eos_ids
    )
    hit_token_limit = bool(
        len(generated_token_ids) >= int(max_new_tokens)
        and not ended_with_eos
    )

    del output_ids, generated_ids
    return text, len(generated_token_ids), hit_token_limit, generated_token_ids


def load_model_and_processor(
    *,
    alias: str,
    resolved_key: str,
    spec: Any,
    base: Any,
    args: argparse.Namespace,
) -> Tuple[Any, Any]:
    model_class = getattr(transformers, spec.model_class, None)
    if model_class is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} lacks {spec.model_class}"
        )

    dtype = (
        base.resolve_dtype(args.dtype)
        if args.dtype
        else base.resolve_dtype(spec.dtype_name)
    )
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl

    print(
        f"\nLoading alias={alias} registry_key={resolved_key} "
        f"repo={spec.repo_id}",
        flush=True,
    )
    try:
        model = model_class.from_pretrained(spec.repo_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = model_class.from_pretrained(spec.repo_id, **kwargs)

    model.eval()
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        # Deterministic greedy decoding.
        if hasattr(generation_config, "do_sample"):
            generation_config.do_sample = False
        for field_name in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field_name):
                setattr(generation_config, field_name, None)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    return model, processor


def load_samples(
    *,
    prompt_path: Path,
    controlled: Any,
    base: Any,
    max_samples: Optional[int],
    num_workers: int,
    download: bool,
) -> List[Sample]:
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    records, audit = controlled.load_records(
        prompt_path,
        download=bool(download),
        max_samples=max_samples,
        num_workers=int(num_workers),
    )
    prompts = base.load_standard_prompts(prompt_path)

    samples: List[Sample] = []
    for record in records:
        sid = int(record.sid)
        prompt = prompts.get(sid)
        if prompt is None:
            continue

        gt = normalize_gt_native(getattr(record, "relation", ""))
        if not gt:
            continue

        samples.append(
            Sample(
                uid=sanitize_uid(sid),
                sid=sid,
                subject=str(prompt["subject"]),
                reference=str(prompt["reference"]),
                question=str(prompt["question_text"]),
                gt=gt,
                record=record,
            )
        )

    if not samples:
        raise RuntimeError("No usable Controlled-A samples")

    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.gt] = counts.get(sample.gt, 0) + 1

    print(
        f"Controlled-A samples={len(samples)} "
        f"relations={counts} audit={len(audit)}",
        flush=True,
    )
    return samples


def run_model(
    *,
    alias: str,
    resolved_key: str,
    spec: Any,
    samples: Sequence[Sample],
    model: Any,
    processor: Any,
    base: Any,
    args: argparse.Namespace,
    root: Path,
) -> Dict[str, Any]:
    model_dir = root / alias
    if args.overwrite and model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = model_dir / "generation_samples.jsonl"
    csv_path = model_dir / "generation_samples.csv"
    errors_path = model_dir / "errors.jsonl"

    completed: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        completed = {
            str(row["uid"]): row
            for row in read_jsonl(jsonl_path)
        }

    config = {
        "script_version": SCRIPT_VERSION,
        "model": alias,
        "resolved_model_key": resolved_key,
        "repo_id": spec.repo_id,
        "dataset": "controlled_a",
        "N_requested": len(samples),
        "max_new_tokens": int(args.max_new_tokens),
        "decoding": "greedy; do_sample=False; EOS early stopping",
    }
    write_json(model_dir / "config.json", config)

    successful = len(completed)

    for sample in tqdm(
        samples,
        desc=f"{alias}:controlled_a_generation",
        total=len(samples),
    ):
        if sample.uid in completed:
            continue

        image = None
        batch = None
        try:
            image = base.record_image(sample.record)
            batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=sample.question,
                device=torch.device(args.device),
            )

            generation, token_count, hit_limit, token_ids = generate_complete_text(
                model=model,
                processor=processor,
                batch=dict(batch),
                max_new_tokens=args.max_new_tokens,
            )
            pred = parse_generation_relation(generation)
            acc = int(pred == sample.gt)

            row = {
                "model": alias,
                "resolved_model_key": resolved_key,
                "dataset": "controlled_a",
                "uid": sample.uid,
                "sid": sample.sid,
                "subject": sample.subject,
                "reference": sample.reference,
                "pred": pred,
                "question": sample.question,
                "gt": sample.gt,
                "generation": generation,
                "acc": acc,
                "parse_ok": int(bool(pred)),
                "generated_token_count": int(token_count),
                "hit_token_limit": int(hit_limit),
                "generated_token_ids": token_ids,
            }
            append_jsonl(jsonl_path, row)
            completed[sample.uid] = row
            successful += 1

            if args.print_every > 0 and successful % args.print_every == 0:
                print("\n" + "=" * 100, flush=True)
                print(
                    f"model={alias} sid={sample.sid} "
                    f"pred={pred or '<unparsed>'} gt={sample.gt} acc={acc} "
                    f"tokens={token_count} hit_limit={int(hit_limit)}",
                    flush=True,
                )
                print(f"question:\n{sample.question}", flush=True)
                print("generation:", flush=True)
                print(generation, flush=True)
                print("=" * 100, flush=True)

            if hit_limit:
                print(
                    f"[WARNING {alias} sid={sample.sid}] generation reached "
                    f"--max-new-tokens={args.max_new_tokens}; increase the cap.",
                    flush=True,
                )

        except Exception as exc:
            error = {
                "model": alias,
                "sid": sample.sid,
                "uid": sample.uid,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            print(
                f"\n[ERROR {alias} sid={sample.sid}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            if batch is not None:
                del batch
            gc.collect()
            if torch.cuda.is_available() and (
                args.empty_cache_every <= 1
                or max(successful, 1) % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    rows = list(completed.values())
    rows.sort(key=lambda row: int(row["sid"]))

    # Rewrite deduplicated output after resume.
    jsonl_path.unlink(missing_ok=True)
    for row in rows:
        append_jsonl(jsonl_path, row)
    write_csv(csv_path, rows)

    parsed = [row for row in rows if int(row["parse_ok"]) == 1]
    summary = {
        "model": alias,
        "resolved_model_key": resolved_key,
        "repo_id": spec.repo_id,
        "dataset": "controlled_a",
        "N": len(rows),
        "generation_acc": (
            float(np.mean([int(row["acc"]) for row in rows]))
            if rows else float("nan")
        ),
        "parse_rate": (
            float(np.mean([int(row["parse_ok"]) for row in rows]))
            if rows else float("nan")
        ),
        "accuracy_among_parsed": (
            float(np.mean([int(row["acc"]) for row in parsed]))
            if parsed else float("nan")
        ),
        "mean_generated_token_count": (
            float(np.mean([int(row["generated_token_count"]) for row in rows]))
            if rows else float("nan")
        ),
        "max_generated_token_count": (
            int(max(int(row["generated_token_count"]) for row in rows))
            if rows else 0
        ),
        "hit_token_limit_count": int(
            sum(int(row["hit_token_limit"]) for row in rows)
        ),
        "errors": len(read_jsonl(errors_path)),
    }
    write_json(model_dir / "summary.json", summary)

    print(
        f"\nSUMMARY {alias}: N={summary['N']} "
        f"acc={summary['generation_acc']:.4f} "
        f"parse={summary['parse_rate']:.4f} "
        f"mean_tokens={summary['mean_generated_token_count']:.2f} "
        f"max_tokens={summary['max_generated_token_count']} "
        f"hit_limit={summary['hit_token_limit_count']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    models = parse_csv_list(args.models)
    if models == ["all"]:
        models = list(REQUESTED_MODEL_ALIASES)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    base = import_file(Path(args.base_script), "controlled_generation_base")
    two_object = import_file(
        Path(args.two_object_script),
        "controlled_generation_two_object",
    )
    controlled = import_file(
        Path(args.controlled_script),
        "controlled_generation_dataset",
    )

    if hasattr(base, "merged_model_specs"):
        specs = base.merged_model_specs(two_object)
    else:
        specs = getattr(two_object, "MODEL_SPECS")

    resolved: Dict[str, str] = {
        alias: resolve_model_key(alias, specs)
        for alias in models
    }

    print("Resolved models:", flush=True)
    for alias, key in resolved.items():
        print(f"  {alias:14s} -> {key:16s} -> {specs[key].repo_id}", flush=True)

    samples = load_samples(
        prompt_path=Path(args.controlled_prompt_jsonl),
        controlled=controlled,
        base=base,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        download=args.download,
    )

    summaries: List[Dict[str, Any]] = []
    top_errors = root / "model_errors.jsonl"

    for alias in models:
        key = resolved[alias]
        spec = specs[key]
        model = None
        processor = None
        try:
            model, processor = load_model_and_processor(
                alias=alias,
                resolved_key=key,
                spec=spec,
                base=base,
                args=args,
            )
            summaries.append(
                run_model(
                    alias=alias,
                    resolved_key=key,
                    spec=spec,
                    samples=samples,
                    model=model,
                    processor=processor,
                    base=base,
                    args=args,
                    root=root,
                )
            )
        except Exception as exc:
            append_jsonl(
                top_errors,
                {
                    "model": alias,
                    "resolved_model_key": key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"\n[FATAL MODEL ERROR {alias}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(root / "generation_summary.csv", summaries)
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        "",
    ]
    for row in summaries:
        lines.append(
            f"{row['model']:14s} N={row['N']:4d} "
            f"acc={row['generation_acc']:.4f} "
            f"parse={row['parse_rate']:.4f} "
            f"mean_tokens={row['mean_generated_token_count']:.2f} "
            f"max_tokens={row['max_generated_token_count']:3d} "
            f"hit_limit={row['hit_token_limit_count']}"
        )
    (root / "generation_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nSaved outputs to {root}", flush=True)


if __name__ == "__main__":
    main()
