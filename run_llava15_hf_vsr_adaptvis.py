#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaVA-1.5 HF + RMSNorm-epsilon + ScalingVis / AdaptVis evaluation on VSR.

This is a VSR adapter around the repository's validated LLaVA implementation:
  run_llava15_hf_rmsnorm_eps_ablation_original.py

It reuses the original script's:
  - checkpoint-faithful LLaVA loading
  - RMSNorm epsilon override
  - merged image-token mask capture
  - pre-softmax LLaMA attention intervention
  - confidence-routed AdaptVis logic

VSR prompt:
  USER: <image>
  Does the statement accurately describe the image?
  Reply with exactly one word: yes or no. Do not explain.
  Statement: ...
  ASSISTANT:

Examples
--------
# Standard checkpoint control:
CUDA_VISIBLE_DEVICES=0 python run_llava15_hf_vsr_adaptvis.py \
  --method base --rms-norm-eps 1e-5 --max-layers 4

# Full LLaVA eps + AdaptVis condition:
CUDA_VISIBLE_DEVICES=1 python run_llava15_hf_vsr_adaptvis.py \
  --method adapt_vis --rms-norm-eps 1e-6 \
  --weight1 0.5 --weight2 1.5 --threshold 0.4 --max-layers 4

The script writes one JSONL file incrementally and can resume safely.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Reuse the exact LLaVA attention patch / epsilon implementation that produced
# the previous Controlled-A results.
import run_llava15_hf_rmsnorm_eps_ablation_original as core


DEFAULT_VSR_ANN = "data/benchmarks/vsr/repo/data/splits/zeroshot/test.jsonl"
DEFAULT_VSR_IMAGE_ROOT = "data/benchmarks/vsr/images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HF LLaVA-1.5 ScalingVis / AdaptVis on VSR."
    )

    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision", default="a272c74")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16", "auto"],
        help=(
            "Use float32 to match the previous LLaVA HF / custom comparison. "
            "Use bfloat16 only if memory is constrained."
        ),
    )

    parser.add_argument(
        "--method",
        default="adapt_vis",
        choices=["base", "scaling_vis", "adapt_vis"],
    )
    parser.add_argument("--rms-norm-eps", default=1e-5, type=float)
    parser.add_argument("--weight", default=0.5, type=float)
    parser.add_argument("--weight1", default=0.5, type=float)
    parser.add_argument("--weight2", default=1.5, type=float)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument(
        "--max-layers",
        default=4,
        type=int,
        help="Apply ScalingVis / AdaptVis to decoder layers [0, max_layers).",
    )
    parser.add_argument("--max-new-tokens", default=4, type=int)

    parser.add_argument("--vsr-ann", default=DEFAULT_VSR_ANN)
    parser.add_argument("--vsr-image-root", default=DEFAULT_VSR_IMAGE_ROOT)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-first", default=5, type=int)
    parser.add_argument("--output-dir", default="output/llava15_vsr_adaptvis")

    # Fields required by the imported core.load_hf_model implementation.
    parser.add_argument("--ignore-mismatched-sizes", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_vsr_label(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if int(value) else "no"

    text = normalize_space(value).lower()
    aliases = {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "correct": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
        "incorrect": "no",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(f"Unsupported VSR label: {value!r}")


def parse_yes_no(generation: str) -> str:
    match = re.search(r"\b(yes|no)\b", normalize_space(generation).lower())
    return match.group(1) if match else ""


def make_prompt(caption: str) -> str:
    return (
        "USER: <image>\n"
        "Does the statement accurately describe the image? "
        "Reply with exactly one word: yes or no. Do not explain.\n"
        f"Statement: {caption}\n"
        "ASSISTANT:"
    )


def load_vsr_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ann_path = Path(args.vsr_ann)
    image_root = Path(args.vsr_image_root)

    if not ann_path.is_file():
        raise FileNotFoundError(f"VSR JSONL not found: {ann_path}")

    records: List[Dict[str, Any]] = []
    with ann_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)

            required = {"caption", "label", "image_link"}
            missing = required - set(item)
            if missing:
                raise KeyError(f"VSR row {index} lacks keys: {sorted(missing)}")

            image_name = Path(str(item["image_link"]).split("?", 1)[0]).name
            if not image_name:
                raise ValueError(f"VSR row {index}: bad image_link={item['image_link']!r}")

            caption = normalize_space(item["caption"])
            records.append(
                {
                    "sid": str(index),
                    "caption": caption,
                    "gold": normalize_vsr_label(item["label"]),
                    "relation": item.get("relation"),
                    "image_link": item["image_link"],
                    "image_path": str(image_root / image_name),
                    "prompt": make_prompt(caption),
                }
            )

    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    records = records[args.start :]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No VSR records selected.")
    return records


def model_dtype(model: Any) -> torch.dtype:
    return next(model.parameters()).dtype


def prepare_inputs(
    processor: Any,
    image: Image.Image,
    prompt: str,
    device: str,
    dtype: torch.dtype,
) -> Any:
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            device=device,
            dtype=dtype,
        )
    return inputs


def job_tag(args: argparse.Namespace) -> str:
    eps = f"{args.rms_norm_eps:.0e}".replace("-", "m")
    method = args.method
    if method == "scaling_vis":
        method += f"_w{args.weight:g}"
    elif method == "adapt_vis":
        method += (
            f"_w1{args.weight1:g}_w2{args.weight2:g}"
            f"_thr{args.threshold:g}"
        )
    return f"llava15_vsr_{method}_eps{eps}_l{args.max_layers}"


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tag = job_tag(args)
    return root / f"{tag}.jsonl", root / f"{tag}.summary.json"


def resume_map(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["sid"])] = record
    return records


def generation_metadata(diag: Any) -> Dict[str, Any]:
    return {
        "requested_weight": float(diag.requested_weight),
        "modified_calls": int(diag.modified_calls),
        "image_token_count": int(diag.image_token_count),
        "merged_sequence_length": int(diag.merged_sequence_length),
        "image_start": diag.image_start,
        "image_end": diag.image_end,
    }


def selected_weight_and_probe(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    controller: Any,
    inputs: Any,
    prompt_length: int,
) -> Tuple[float, Optional[str], Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    if args.method == "base":
        return 1.0, None, None, None, None

    if args.method == "scaling_vis":
        return float(args.weight), None, None, None, None

    probe_output, probe_diag = core.generate_once(
        model=model,
        inputs=inputs,
        controller=controller,
        weight=1.0,
        max_new_tokens=args.max_new_tokens,
    )
    probe_generation = core.decode_generated(processor, probe_output, prompt_length)
    probe_confidence = float(core.first_step_confidence(probe_output))
    rounded_confidence = float(round(probe_confidence, 2))
    selected_weight = (
        float(args.weight1)
        if rounded_confidence < float(args.threshold)
        else float(args.weight2)
    )

    return (
        selected_weight,
        probe_generation,
        probe_confidence,
        rounded_confidence,
        generation_metadata(probe_diag),
    )


def relation_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("relation") or "unknown")
        groups[key].append(record)

    result: Dict[str, Dict[str, Any]] = {}
    for relation, rows in sorted(groups.items()):
        correct = sum(bool(row["correct"]) for row in rows)
        result[relation] = {
            "num_samples": len(rows),
            "num_correct": correct,
            "accuracy": correct / max(len(rows), 1),
        }
    return result


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if args.rms_norm_eps <= 0:
        raise ValueError("--rms-norm-eps must be positive.")

    vsr = load_vsr_records(args)
    jsonl_path, summary_path = output_paths(args)
    completed = resume_map(jsonl_path) if args.resume else {}

    # The imported loader expects these core args and installs exactly the same
    # pre-softmax LLaMA intervention used in prior LLaVA runs.
    model, processor, _, controller = core.load_hf_model(args)
    active_dtype = model_dtype(model)

    existing = list(completed.values())
    correct_count = sum(bool(row.get("correct")) for row in existing)
    evaluated = len(existing)
    skipped_missing = 0
    relation_counter: Counter[str] = Counter()

    mode = "a" if args.resume else "w"
    with jsonl_path.open(mode, encoding="utf-8") as output:
        bar = tqdm(
            vsr,
            desc=(
                f"LLaVA-1.5/VSR/{args.method}/"
                f"eps={args.rms_norm_eps:g}/layers={args.max_layers}"
            ),
        )
        for index, sample in enumerate(bar):
            if sample["sid"] in completed:
                continue

            image_path = Path(sample["image_path"])
            if not image_path.is_file():
                message = f"Missing VSR image sid={sample['sid']}: {image_path}"
                if args.skip_missing_images:
                    print("WARNING:", message)
                    skipped_missing += 1
                    continue
                raise FileNotFoundError(message)

            image = Image.open(image_path).convert("RGB")
            inputs = prepare_inputs(
                processor=processor,
                image=image,
                prompt=sample["prompt"],
                device=args.device,
                dtype=active_dtype,
            )
            prompt_length = int(inputs["input_ids"].shape[-1])

            (
                selected_weight,
                probe_generation,
                probe_confidence,
                rounded_confidence,
                probe_diag,
            ) = selected_weight_and_probe(
                args=args,
                model=model,
                processor=processor,
                controller=controller,
                inputs=inputs,
                prompt_length=prompt_length,
            )

            final_output, final_diag = core.generate_once(
                model=model,
                inputs=inputs,
                controller=controller,
                weight=selected_weight,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = core.decode_generated(processor, final_output, prompt_length)
            prediction_normalized = parse_yes_no(prediction)
            correct = prediction_normalized == sample["gold"]

            final_diag_dict = generation_metadata(final_diag)
            if args.method in {"scaling_vis", "adapt_vis"} and selected_weight != 1.0:
                if final_diag_dict["modified_calls"] <= 0:
                    raise RuntimeError(
                        "ScalingVis/AdaptVis requested a non-unit weight, but no "
                        "attention module was modified. Refusing invalid results."
                    )

            record = {
                "sid": sample["sid"],
                "dataset": "vsr",
                "image_path": sample["image_path"],
                "image_link": sample["image_link"],
                "caption": sample["caption"],
                "relation": sample["relation"],
                "prompt": sample["prompt"],
                "gold": sample["gold"],
                "prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "correct": bool(correct),
                "method": args.method,
                "selected_weight": float(selected_weight),
                "probe_generation": probe_generation,
                "probe_confidence": probe_confidence,
                "rounded_probe_confidence": rounded_confidence,
                "probe_diagnostics": probe_diag,
                "final_diagnostics": final_diag_dict,
            }

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

            correct_count += int(correct)
            evaluated += 1
            relation_counter[str(sample["relation"] or "unknown")] += 1

            if index < args.print_first:
                print("-" * 100)
                print(
                    f"[{sample['sid']}] relation={sample['relation']!r} "
                    f"gold={sample['gold']!r} pred={prediction!r} "
                    f"norm={prediction_normalized!r} correct={correct}"
                )
                print(
                    f"weight={selected_weight:g} "
                    f"modified_calls={final_diag_dict['modified_calls']} "
                    f"image_tokens={final_diag_dict['image_token_count']}"
                )

            bar.set_postfix(acc=f"{correct_count / max(evaluated, 1):.4f}")

    all_records = list(resume_map(jsonl_path).values())
    low_count = (
        sum(row["selected_weight"] == float(args.weight1) for row in all_records)
        if args.method == "adapt_vis"
        else None
    )
    high_count = (
        sum(row["selected_weight"] == float(args.weight2) for row in all_records)
        if args.method == "adapt_vis"
        else None
    )

    summary = {
        "model": args.model,
        "revision": args.revision,
        "dataset": "VSR",
        "vsr_ann": str(args.vsr_ann),
        "vsr_image_root": str(args.vsr_image_root),
        "method": args.method,
        "rms_norm_eps": float(args.rms_norm_eps),
        "weight": float(args.weight),
        "weight1": float(args.weight1),
        "weight2": float(args.weight2),
        "threshold": float(args.threshold),
        "max_layers": int(args.max_layers),
        "max_new_tokens": int(args.max_new_tokens),
        "intervention": (
            "Initial multimodal prefill only: multiply raw pre-softmax "
            "attention logits from the final prompt query to merged "
            "image-token keys in selected LLaMA decoder layers."
        ),
        "num_selected": len(vsr),
        "num_evaluated": len(all_records),
        "num_correct": sum(bool(row["correct"]) for row in all_records),
        "accuracy": (
            sum(bool(row["correct"]) for row in all_records)
            / max(len(all_records), 1)
        ),
        "skipped_missing_images_this_run": skipped_missing,
        "low_branch_count": low_count,
        "high_branch_count": high_count,
        "per_relation": relation_summary(all_records),
        "records_jsonl": str(jsonl_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        f"RESULT LLaVA-1.5 VSR | method={args.method} "
        f"eps={args.rms_norm_eps:g} | "
        f"{summary['num_correct']}/{summary['num_evaluated']} = "
        f"{summary['accuracy']:.6f}"
    )
    if args.method == "adapt_vis":
        print(
            f"branches: w1={args.weight1:g}: {low_count}, "
            f"w2={args.weight2:g}: {high_count}"
        )
    print(f"records: {jsonl_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
