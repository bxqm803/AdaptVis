#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate all Qwen2-VL Instruct sizes on Controlled-A with two protocols.

Official Qwen2-VL Instruct checkpoints evaluated by default:
    qwen2-vl-2b   -> Qwen/Qwen2-VL-2B-Instruct
    qwen2-vl-7b   -> Qwen/Qwen2-VL-7B-Instruct
    qwen2-vl-72b  -> Qwen/Qwen2-VL-72B-Instruct

For every image/question, the script evaluates:

1. Free greedy generation
   - model.generate(...)
   - do_sample=False
   - no min_new_tokens
   - the model may stop at EOS naturally

2. Direct four-candidate likelihood scoring
   - candidates: left, right, on, under
   - teacher-forced conditional log likelihood
   - reports both:
       a) mean token log probability (headline candidate result)
       b) summed token log probability
   - no label is used to select a candidate

The script prints every sample with:
    question
    GT
    raw generation
    generation prediction and running ACC
    candidate scores
    candidate prediction and running ACC

This script does not train or fine-tune the model.

It reuses dataset/image/prompt helper functions from:
    analyze_controlledA_similarity_head_generation_step1_v1.py

Default outputs:
    output/qwen2vl_controlledA_generation_vs_candidates/
        qwen2-vl-2b/
        qwen2-vl-7b/
        qwen2-vl-72b/
        reports/model_comparison.csv
        reports/per_relation.csv
        reports/confusion.csv
        reports/errors.jsonl
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )
except Exception as exc:
    raise SystemExit(
        "Unable to import Qwen2-VL classes. "
        f"Installed transformers error: {type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "qwen2vl-controlledA-generation-vs-candidates-v1"

MODEL_REPOS = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
}

DEFAULT_MODELS = list(MODEL_REPOS)
RELATIONS = ("left", "right", "on", "under")
CANDIDATE_TEXT = {
    "left": "left",
    "right": "right",
    "on": "on",
    "under": "under",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated keys: qwen2-vl-2b,qwen2-vl-7b,qwen2-vl-72b"
        ),
    )
    parser.add_argument(
        "--helper-module",
        default="analyze_controlledA_similarity_head_generation_step1_v1",
        help="Existing Controlled-A helper module without the .py suffix.",
    )
    parser.add_argument(
        "--controlled-module",
        default="",
        help=(
            "Optional explicit Controlled-A extractor module. If omitted, "
            "the helper module tries its known extractor module names."
        ),
    )
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument("--dataset-key", default="Controlled_Images_A")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Transformers device_map. Use auto for multi-GPU/sharded loading, "
            "or cuda:0 for one GPU."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
    )
    parser.add_argument(
        "--quantize-72b",
        default="4bit",
        choices=["none", "4bit", "8bit"],
        help=(
            "Quantization used only for qwen2-vl-72b. Default 4bit. "
            "2B and 7B remain in --dtype precision."
        ),
    )
    parser.add_argument(
        "--max-memory",
        default="",
        help=(
            "Optional JSON mapping for accelerate device_map, for example "
            """'{"0":"78GiB","1":"78GiB","cpu":"200GiB"}'."""
        ),
    )
    parser.add_argument(
        "--offload-folder",
        default="",
        help="Optional disk-offload directory for device_map=auto.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--candidate-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Score the four candidates in one batch. If batched scoring fails, "
            "the script automatically falls back to sequential scoring."
        ),
    )
    parser.add_argument(
        "--candidate-prefix",
        default="",
        help=(
            "Text prepended to each candidate before tokenization. "
            "The default empty prefix scores exactly left/right/on/under."
        ),
    )
    parser.add_argument(
        "--score-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also include EOS probability after each candidate.",
    )

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default="output/qwen2vl_controlledA_generation_vs_candidates",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for each requested model.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the complete suite on the first sample/model error.",
    )
    return parser.parse_args()


def parse_models(value: str) -> List[str]:
    result: List[str] = []
    for item in str(value).split(","):
        key = item.strip().lower()
        if key and key not in result:
            result.append(key)
    if not result:
        raise ValueError("--models resolved to an empty list")
    unknown = [key for key in result if key not in MODEL_REPOS]
    if unknown:
        raise ValueError(
            f"Unknown model keys: {unknown}; available={list(MODEL_REPOS)}"
        )
    return result


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None

    import re

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

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
    if value is None:
        return []

    import re

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(under|below|beneath)\b", "under"),
        (r"\b(on top of|on top|above|over)\b", "on"),
        (r"\bon\b", "on"),
    ]
    hits: List[Tuple[int, str]] = []
    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            hits.append((match.start(), label))
    hits.sort(key=lambda item: item[0])

    labels: List[str] = []
    for _, label in hits:
        if label not in labels:
            labels.append(label)
    return labels


def one_line(value: Any) -> str:
    return " ".join(str(value).split())


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
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


def load_helper(name: str) -> Any:
    module = importlib.import_module(name)
    required = [
        "import_controlled_module",
        "load_standard_prompts",
        "record_image",
        "make_question_batch",
        "configure_processor",
    ]
    missing = [key for key in required if not hasattr(module, key)]
    if missing:
        raise RuntimeError(
            f"Helper module {name!r} lacks required functions: {missing}"
        )
    return module


def parse_max_memory(value: str) -> Optional[Dict[Any, str]]:
    if not value.strip():
        return None
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("--max-memory must decode to a JSON object")

    result: Dict[Any, str] = {}
    for key, memory in raw.items():
        parsed_key: Any = int(key) if str(key).isdigit() else str(key)
        result[parsed_key] = str(memory)
    return result


def build_quantization_config(
    model_key: str,
    args: argparse.Namespace,
) -> Optional[BitsAndBytesConfig]:
    if model_key != "qwen2-vl-72b":
        return None
    if args.quantize_72b == "none":
        return None
    if args.quantize_72b == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    compute_dtype = (
        torch.bfloat16
        if args.dtype in ("auto", "bfloat16")
        else resolve_dtype(args.dtype)
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def model_input_device(model: Any) -> torch.device:
    # With accelerate sharding, model.device normally points to the input
    # embedding device. Fall back to the first non-meta parameter.
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine model input device")


def load_model(
    model_key: str,
    args: argparse.Namespace,
    helper: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    repo_id = MODEL_REPOS[model_key]
    quantization_config = build_quantization_config(model_key, args)

    kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    if args.offload_folder:
        offload_folder = Path(args.offload_folder) / model_key
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_folder)
        kwargs["offload_state_dict"] = True

    print("\n" + "=" * 110)
    print(f"Loading {model_key}: {repo_id}")
    print(f"dtype={args.dtype} | device_map={args.device_map}")
    print(
        "quantization="
        + (
            args.quantize_72b
            if model_key == "qwen2-vl-72b"
            else "none"
        )
    )
    print("=" * 110, flush=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        repo_id,
        **kwargs,
    )
    model.eval()

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    processor = AutoProcessor.from_pretrained(repo_id)
    helper.configure_processor(model, processor)

    metadata = {
        "model": model_key,
        "repo_id": repo_id,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_impl,
        "quantization": (
            args.quantize_72b
            if model_key == "qwen2-vl-72b"
            else "none"
        ),
        "input_device": str(model_input_device(model)),
    }
    return model, processor, metadata


def decode_generated(
    processor: Any,
    output_ids: torch.Tensor,
    prompt_length: int,
) -> str:
    new_ids = output_ids[0, prompt_length:]
    return processor.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()


def run_generation(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    max_new_tokens: int,
) -> str:
    prompt_length = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    text = decode_generated(processor, output_ids, prompt_length)
    del output_ids
    return text


def candidate_token_ids(
    tokenizer: Any,
    candidate_prefix: str,
    score_eos: bool,
) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for relation in RELATIONS:
        text = candidate_prefix + CANDIDATE_TEXT[relation]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(value) for value in ids]
        if not ids:
            raise RuntimeError(
                f"Candidate {relation!r} tokenized to an empty sequence"
            )
        if score_eos:
            if eos_id is None:
                raise RuntimeError(
                    "--score-eos requested but tokenizer has no eos_token_id"
                )
            ids.append(int(eos_id))
        result[relation] = ids
    return result


def duplicate_multimodal_value(
    key: str,
    value: Any,
    repeats: int,
) -> Any:
    if not torch.is_tensor(value):
        return value

    if key in ("pixel_values", "pixel_values_videos"):
        return torch.cat([value] * repeats, dim=0)

    if key in (
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    ):
        if value.ndim == 1:
            return value.repeat(repeats)
        repeat_shape = [repeats] + [1] * (value.ndim - 1)
        return value.repeat(*repeat_shape)

    return value


def make_batched_candidate_inputs(
    batch: Dict[str, Any],
    candidate_ids_by_label: Mapping[str, Sequence[int]],
    pad_token_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    labels = list(RELATIONS)
    prompt_ids = batch["input_ids"]
    prompt_mask = batch.get("attention_mask")

    if prompt_ids.shape[0] != 1:
        raise ValueError(
            f"Expected one prompt row, got {tuple(prompt_ids.shape)}"
        )
    prompt_length = int(prompt_ids.shape[1])
    max_candidate_length = max(
        len(candidate_ids_by_label[label]) for label in labels
    )
    full_length = prompt_length + max_candidate_length

    input_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    metadata: Dict[str, Dict[str, int]] = {}

    for label in labels:
        candidate_ids = list(candidate_ids_by_label[label])
        candidate_length = len(candidate_ids)
        padding_length = max_candidate_length - candidate_length

        candidate_tensor = torch.tensor(
            candidate_ids,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        ).unsqueeze(0)
        padding = torch.full(
            (1, padding_length),
            int(pad_token_id),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        row_ids = torch.cat(
            [prompt_ids, candidate_tensor, padding],
            dim=1,
        )
        input_rows.append(row_ids)

        if prompt_mask is None:
            prompt_row_mask = torch.ones(
                (1, prompt_length),
                dtype=torch.long,
                device=prompt_ids.device,
            )
        else:
            prompt_row_mask = prompt_mask

        continuation_mask = torch.cat([
            torch.ones(
                (1, candidate_length),
                dtype=prompt_row_mask.dtype,
                device=prompt_row_mask.device,
            ),
            torch.zeros(
                (1, padding_length),
                dtype=prompt_row_mask.dtype,
                device=prompt_row_mask.device,
            ),
        ], dim=1)
        mask_rows.append(
            torch.cat([prompt_row_mask, continuation_mask], dim=1)
        )

        metadata[label] = {
            "row": len(input_rows) - 1,
            "prompt_length": prompt_length,
            "candidate_length": candidate_length,
            "full_length": full_length,
        }

    full_batch: Dict[str, Any] = {
        "input_ids": torch.cat(input_rows, dim=0),
        "attention_mask": torch.cat(mask_rows, dim=0),
    }

    for key, value in batch.items():
        if key in ("input_ids", "attention_mask"):
            continue

        if key in ("token_type_ids", "mm_token_type_ids"):
            if not torch.is_tensor(value) or value.shape[0] != 1:
                continue
            extension = torch.zeros(
                (1, max_candidate_length),
                dtype=value.dtype,
                device=value.device,
            )
            one_row = torch.cat([value, extension], dim=1)
            full_batch[key] = one_row.repeat(len(labels), 1)
            continue

        full_batch[key] = duplicate_multimodal_value(
            key,
            value,
            len(labels),
        )

    return full_batch, metadata


def score_candidates_batched(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    candidate_ids_by_label: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, Any]]:
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")

    full_batch, metadata = make_batched_candidate_inputs(
        batch,
        candidate_ids_by_label,
        int(pad_token_id),
    )

    with torch.inference_mode():
        outputs = model(
            **full_batch,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits.float()

    result: Dict[str, Dict[str, Any]] = {}
    for label in RELATIONS:
        info = metadata[label]
        row = info["row"]
        prompt_length = info["prompt_length"]
        candidate_length = info["candidate_length"]
        candidate_ids = torch.tensor(
            candidate_ids_by_label[label],
            dtype=torch.long,
            device=logits.device,
        )

        # Token t is predicted by the logit at position t-1.
        candidate_logits = logits[
            row,
            prompt_length - 1:
                prompt_length + candidate_length - 1,
            :,
        ]
        log_probs = F.log_softmax(candidate_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=candidate_ids.unsqueeze(-1),
        ).squeeze(-1)

        result[label] = {
            "sum_logprob": float(token_log_probs.sum().item()),
            "mean_logprob": float(token_log_probs.mean().item()),
            "token_logprobs": [
                float(value) for value in token_log_probs.tolist()
            ],
            "token_ids": [
                int(value) for value in candidate_ids.tolist()
            ],
        }

    del outputs
    del logits
    del full_batch
    return result


def make_single_candidate_inputs(
    batch: Dict[str, Any],
    candidate_ids: Sequence[int],
) -> Tuple[Dict[str, Any], int, int]:
    prompt_ids = batch["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    candidate_tensor = torch.tensor(
        list(candidate_ids),
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    ).unsqueeze(0)

    result = dict(batch)
    result["input_ids"] = torch.cat(
        [prompt_ids, candidate_tensor],
        dim=1,
    )

    prompt_mask = batch.get("attention_mask")
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
    extension = torch.ones(
        (1, len(candidate_ids)),
        dtype=prompt_mask.dtype,
        device=prompt_mask.device,
    )
    result["attention_mask"] = torch.cat(
        [prompt_mask, extension],
        dim=1,
    )

    for key in ("token_type_ids", "mm_token_type_ids"):
        value = batch.get(key)
        if torch.is_tensor(value):
            extension = torch.zeros(
                (1, len(candidate_ids)),
                dtype=value.dtype,
                device=value.device,
            )
            result[key] = torch.cat([value, extension], dim=1)

    return result, prompt_length, len(candidate_ids)


def score_candidates_sequential(
    model: Any,
    batch: Dict[str, Any],
    candidate_ids_by_label: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for label in RELATIONS:
        candidate_ids = list(candidate_ids_by_label[label])
        full_batch, prompt_length, candidate_length = (
            make_single_candidate_inputs(batch, candidate_ids)
        )

        with torch.inference_mode():
            outputs = model(
                **full_batch,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits.float()

        candidate_logits = logits[
            0,
            prompt_length - 1:
                prompt_length + candidate_length - 1,
            :,
        ]
        target = torch.tensor(
            candidate_ids,
            dtype=torch.long,
            device=logits.device,
        )
        log_probs = F.log_softmax(candidate_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target.unsqueeze(-1),
        ).squeeze(-1)

        result[label] = {
            "sum_logprob": float(token_log_probs.sum().item()),
            "mean_logprob": float(token_log_probs.mean().item()),
            "token_logprobs": [
                float(value) for value in token_log_probs.tolist()
            ],
            "token_ids": [int(value) for value in candidate_ids],
        }

        del outputs
        del logits
        del full_batch

    return result


def score_candidates(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    candidate_ids_by_label: Mapping[str, Sequence[int]],
    prefer_batch: bool,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    if prefer_batch:
        try:
            return (
                score_candidates_batched(
                    model,
                    processor,
                    batch,
                    candidate_ids_by_label,
                ),
                "batched",
            )
        except Exception as exc:
            print(
                "\n[WARN] Batched candidate scoring failed; "
                "falling back to sequential scoring.\n"
                f"       {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return (
        score_candidates_sequential(
            model,
            batch,
            candidate_ids_by_label,
        ),
        "sequential",
    )


def argmax_score(
    scores: Mapping[str, Mapping[str, Any]],
    field: str,
) -> str:
    return max(
        RELATIONS,
        key=lambda label: float(scores[label][field]),
    )


def softmax_over_candidate_scores(
    scores: Mapping[str, Mapping[str, Any]],
    field: str,
) -> Dict[str, float]:
    values = torch.tensor(
        [float(scores[label][field]) for label in RELATIONS],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(values, dim=0).tolist()
    return {
        label: float(probability)
        for label, probability in zip(RELATIONS, probabilities)
    }


def safe_cleanup_batch(batch: Optional[Dict[str, Any]]) -> None:
    if batch is not None:
        batch.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def completed_rows(samples_path: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(samples_path)
    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "ok":
            result[int(row["sid"])] = row
    return result


def summarize_model(
    model_key: str,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    usable = [row for row in rows if row.get("status") == "ok"]
    if not usable:
        raise RuntimeError(f"No successful rows for {model_key}")

    n = len(usable)
    generation_correct = sum(
        bool(row["generation_correct"]) for row in usable
    )
    candidate_mean_correct = sum(
        bool(row["candidate_mean_correct"]) for row in usable
    )
    candidate_sum_correct = sum(
        bool(row["candidate_sum_correct"]) for row in usable
    )
    strict_generation_correct = sum(
        bool(row["generation_strict_single_correct"])
        for row in usable
    )
    generation_valid = sum(
        row.get("generation_prediction") in RELATIONS
        for row in usable
    )
    generation_multi = sum(
        bool(row["generation_multi_relation"])
        for row in usable
    )

    headline = {
        "model": model_key,
        "repo_id": metadata["repo_id"],
        "n": n,
        "generation_accuracy": generation_correct / n,
        "generation_strict_single_accuracy": (
            strict_generation_correct / n
        ),
        "generation_valid_rate": generation_valid / n,
        "generation_multi_relation_rate": generation_multi / n,
        "candidate_mean_logprob_accuracy": (
            candidate_mean_correct / n
        ),
        "candidate_sum_logprob_accuracy": (
            candidate_sum_correct / n
        ),
        "candidate_mean_minus_generation": (
            candidate_mean_correct / n - generation_correct / n
        ),
        "candidate_sum_minus_generation": (
            candidate_sum_correct / n - generation_correct / n
        ),
        "dtype": metadata["dtype"],
        "quantization": metadata["quantization"],
        "device_map": metadata["device_map"],
        "candidate_protocol": (
            "teacher-forced conditional log likelihood over "
            "left/right/on/under"
        ),
    }

    per_relation: List[Dict[str, Any]] = []
    for relation in RELATIONS:
        selected = [row for row in usable if row["gt"] == relation]
        if not selected:
            continue
        per_relation.append({
            "model": model_key,
            "relation": relation,
            "n": len(selected),
            "generation_accuracy": sum(
                row["generation_prediction"] == relation
                for row in selected
            ) / len(selected),
            "candidate_mean_logprob_accuracy": sum(
                row["candidate_mean_prediction"] == relation
                for row in selected
            ) / len(selected),
            "candidate_sum_logprob_accuracy": sum(
                row["candidate_sum_prediction"] == relation
                for row in selected
            ) / len(selected),
        })

    confusion_rows: List[Dict[str, Any]] = []
    for method, prediction_field in [
        ("generation", "generation_prediction"),
        ("candidate_mean", "candidate_mean_prediction"),
        ("candidate_sum", "candidate_sum_prediction"),
    ]:
        for gt in RELATIONS:
            selected = [row for row in usable if row["gt"] == gt]
            counts = Counter(
                row.get(prediction_field) for row in selected
            )
            confusion_rows.append({
                "model": model_key,
                "method": method,
                "gt": gt,
                **{
                    f"pred_{relation}": counts.get(relation, 0)
                    for relation in RELATIONS
                },
                "pred_invalid": counts.get(None, 0),
                "n": len(selected),
            })

    return headline, per_relation, confusion_rows


def run_one_model(
    *,
    args: argparse.Namespace,
    helper: Any,
    records: Sequence[Any],
    prompts: Mapping[int, Mapping[str, Any]],
    model_key: str,
    model_dir: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if args.overwrite and model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples_path = model_dir / "samples.jsonl"
    errors_path = model_dir / "errors.jsonl"
    summary_path = model_dir / "summary.json"
    config_path = model_dir / "config.json"

    done = completed_rows(samples_path)
    done_sids = set(done)

    model = None
    processor = None
    model, processor, metadata = load_model(
        model_key,
        args,
        helper,
    )
    input_device = model_input_device(model)
    candidate_ids = candidate_token_ids(
        processor.tokenizer,
        args.candidate_prefix,
        args.score_eos,
    )

    metadata["candidate_tokenization"] = {
        label: {
            "text": args.candidate_prefix + CANDIDATE_TEXT[label],
            "token_ids": ids,
            "tokens": processor.tokenizer.convert_ids_to_tokens(ids),
        }
        for label, ids in candidate_ids.items()
    }
    metadata["score_eos"] = bool(args.score_eos)
    metadata["script_version"] = SCRIPT_VERSION
    metadata["max_new_tokens"] = args.max_new_tokens
    metadata["n_records"] = len(records)

    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counters = {
        "total": len(done),
        "generation_correct": sum(
            bool(row["generation_correct"]) for row in done.values()
        ),
        "candidate_mean_correct": sum(
            bool(row["candidate_mean_correct"]) for row in done.values()
        ),
        "candidate_sum_correct": sum(
            bool(row["candidate_sum_correct"]) for row in done.values()
        ),
    }

    started = time.time()
    try:
        for record in tqdm(
            records,
            desc=f"qwen2vl-controlledA:{model_key}",
        ):
            sid = int(record.sid)
            if sid in done_sids:
                continue

            image = None
            batch: Optional[Dict[str, Any]] = None
            try:
                prompt_row = prompts[sid]
                question = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATIONS:
                    raise ValueError(
                        f"Unsupported GT at sid={sid}: "
                        f"{prompt_row['answer_raw']!r}"
                    )

                image = helper.record_image(record)
                batch = helper.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=input_device,
                )

                generated_text = run_generation(
                    model,
                    processor,
                    batch,
                    args.max_new_tokens,
                )
                generation_prediction = normalize_relation(
                    generated_text
                )
                generation_relations = extract_all_relations(
                    generated_text
                )

                scores, scoring_mode = score_candidates(
                    model,
                    processor,
                    batch,
                    candidate_ids,
                    args.candidate_batch,
                )
                candidate_mean_prediction = argmax_score(
                    scores,
                    "mean_logprob",
                )
                candidate_sum_prediction = argmax_score(
                    scores,
                    "sum_logprob",
                )
                candidate_mean_probabilities = (
                    softmax_over_candidate_scores(
                        scores,
                        "mean_logprob",
                    )
                )
                candidate_sum_probabilities = (
                    softmax_over_candidate_scores(
                        scores,
                        "sum_logprob",
                    )
                )

                generation_correct = generation_prediction == gt
                candidate_mean_correct = (
                    candidate_mean_prediction == gt
                )
                candidate_sum_correct = (
                    candidate_sum_prediction == gt
                )

                counters["total"] += 1
                counters["generation_correct"] += int(
                    generation_correct
                )
                counters["candidate_mean_correct"] += int(
                    candidate_mean_correct
                )
                counters["candidate_sum_correct"] += int(
                    candidate_sum_correct
                )

                total = counters["total"]
                generation_acc = (
                    counters["generation_correct"] / total
                )
                candidate_mean_acc = (
                    counters["candidate_mean_correct"] / total
                )
                candidate_sum_acc = (
                    counters["candidate_sum_correct"] / total
                )

                row = {
                    "status": "ok",
                    "model": model_key,
                    "repo_id": MODEL_REPOS[model_key],
                    "sid": sid,
                    "question": question,
                    "gt": gt,
                    "generation_text": generated_text,
                    "generation_prediction": generation_prediction,
                    "generation_relations": generation_relations,
                    "generation_correct": generation_correct,
                    "generation_strict_single_correct": (
                        len(generation_relations) == 1
                        and generation_relations[0] == gt
                    ),
                    "generation_multi_relation": (
                        len(generation_relations) > 1
                    ),
                    "candidate_mean_prediction": (
                        candidate_mean_prediction
                    ),
                    "candidate_mean_correct": (
                        candidate_mean_correct
                    ),
                    "candidate_sum_prediction": (
                        candidate_sum_prediction
                    ),
                    "candidate_sum_correct": (
                        candidate_sum_correct
                    ),
                    "candidate_scores": scores,
                    "candidate_mean_probabilities": (
                        candidate_mean_probabilities
                    ),
                    "candidate_sum_probabilities": (
                        candidate_sum_probabilities
                    ),
                    "candidate_scoring_mode": scoring_mode,
                    "running_n": total,
                    "running_generation_accuracy": generation_acc,
                    "running_candidate_mean_accuracy": (
                        candidate_mean_acc
                    ),
                    "running_candidate_sum_accuracy": (
                        candidate_sum_acc
                    ),
                }
                append_jsonl(samples_path, row)
                done[sid] = row
                done_sids.add(sid)

                mean_score_text = " | ".join(
                    f"{label}="
                    f"{scores[label]['mean_logprob']:.4f}"
                    for label in RELATIONS
                )
                sum_score_text = " | ".join(
                    f"{label}="
                    f"{scores[label]['sum_logprob']:.4f}"
                    for label in RELATIONS
                )

                tqdm.write(
                    f"\n[{total}/{len(records)}] "
                    f"model={model_key} | sid={sid}\n"
                    f"  Question: {one_line(question)}\n"
                    f"  GT: {gt}\n"
                    f"  Generation: {generated_text!r}\n"
                    f"  Gen Pred: {generation_prediction or '<invalid>'} "
                    f"| Current ACC: "
                    f"{counters['generation_correct']}/{total} "
                    f"= {generation_acc:.4f}\n"
                    f"  Candidate(mean): "
                    f"{candidate_mean_prediction} "
                    f"| Current ACC: "
                    f"{counters['candidate_mean_correct']}/{total} "
                    f"= {candidate_mean_acc:.4f}\n"
                    f"    mean logP: {mean_score_text}\n"
                    f"  Candidate(sum):  "
                    f"{candidate_sum_prediction} "
                    f"| Current ACC: "
                    f"{counters['candidate_sum_correct']}/{total} "
                    f"= {candidate_sum_acc:.4f}\n"
                    f"    sum logP:  {sum_score_text}"
                )

            except Exception as exc:
                error = {
                    "status": "error",
                    "model": model_key,
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[
                        -30:
                    ],
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
                safe_cleanup_batch(batch)
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass

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

    rows = list(done.values())
    rows.sort(key=lambda row: int(row["sid"]))
    headline, per_relation, confusion = summarize_model(
        model_key,
        rows,
        metadata,
    )
    headline["elapsed_seconds"] = elapsed

    summary_path.write_text(
        json.dumps(
            {
                "headline": headline,
                "per_relation": per_relation,
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print(f"MODEL SUMMARY: {model_key} | n={headline['n']}")
    print("=" * 110)
    print(
        f"Free generation ACC:       "
        f"{headline['generation_accuracy']:.4f}"
    )
    print(
        f"Candidate mean-logP ACC:   "
        f"{headline['candidate_mean_logprob_accuracy']:.4f}"
    )
    print(
        f"Candidate sum-logP ACC:    "
        f"{headline['candidate_sum_logprob_accuracy']:.4f}"
    )
    print(
        f"Candidate(mean)-generation:"
        f"{headline['candidate_mean_minus_generation']:+.4f}"
    )
    print(
        f"Multi-relation generation: "
        f"{headline['generation_multi_relation_rate']:.4f}"
    )

    return headline, per_relation, confusion


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    prompts = helper.load_standard_prompts(prompt_path)
    missing = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompts
    ]
    if missing:
        raise RuntimeError(
            f"Prompt JSONL lacks {len(missing)} IDs; "
            f"first missing IDs={missing[:10]}"
        )

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    suite_errors_path = reports_dir / "errors.jsonl"

    if args.overwrite and suite_errors_path.exists():
        suite_errors_path.unlink()

    print(f"Script: {SCRIPT_VERSION}")
    print(f"Controlled module: {controlled_module.__name__}")
    print(f"Prompt file: {prompt_path}")
    print(f"Samples: {len(records)}")
    print(f"Models: {models}")
    print(
        "Protocols: free greedy generation + "
        "four-candidate conditional likelihood"
    )

    headline_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    for model_key in models:
        try:
            headline, per_relation, confusion = run_one_model(
                args=args,
                helper=helper,
                records=records,
                prompts=prompts,
                model_key=model_key,
                model_dir=output_root / model_key,
            )
            headline_rows.append(headline)
            relation_rows.extend(per_relation)
            confusion_rows.extend(confusion)

            write_csv(
                reports_dir / "model_comparison.csv",
                headline_rows,
            )
            write_csv(
                reports_dir / "per_relation.csv",
                relation_rows,
            )
            write_csv(
                reports_dir / "confusion.csv",
                confusion_rows,
            )
            (reports_dir / "model_comparison.json").write_text(
                json.dumps(
                    headline_rows,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except Exception as exc:
            append_jsonl(suite_errors_path, {
                "model": model_key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-30:],
            })
            print(
                f"\n[MODEL ERROR] {model_key}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                raise

    if not headline_rows:
        raise RuntimeError(
            f"No model completed successfully. See {suite_errors_path}"
        )

    print("\n" + "=" * 118)
    print("QWEN2-VL CONTROLLED-A EVALUATION COMPLETE")
    print("=" * 118)
    print(f"Successful models: {len(headline_rows)}/{len(models)}")
    print(
        f"Main table:   "
        f"{reports_dir / 'model_comparison.csv'}"
    )
    print(
        f"Per relation: "
        f"{reports_dir / 'per_relation.csv'}"
    )
    print(
        f"Confusion:    "
        f"{reports_dir / 'confusion.csv'}"
    )
    if suite_errors_path.exists() and suite_errors_path.stat().st_size:
        print(f"Errors:       {suite_errors_path}")


if __name__ == "__main__":
    main()
