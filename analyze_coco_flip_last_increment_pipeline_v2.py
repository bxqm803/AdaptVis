#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified batch pipeline for object-token -> prompt-last counterfactual transfer.

This single script replaces the earlier chain of analysis scripts. It loads the
model once, runs every eligible COCO left/right sample together with its
horizontal flip, and separates results into:

    both_correct
    original_only_correct
    flipped_only_correct
    both_wrong

For each pair it records:

1. Original-minus-flip state differences at subject, reference, and prompt-last.
2. Exact prompt-last block decomposition:

       Delta output = Delta input + Delta attention + Delta MLP

3. Exact source-group/head attention contributions into prompt-last:

       C_{S->last}^{l,h}
         = sum_{s in S} A_{last,s}^{l,h} V_s^{l,h} W_O^{l,h}

       Delta C = C_original - C_flipped

4. Exact routing/content decomposition:

       Delta C = routing + content

       routing = (A_o - A_f) mean(V_o, V_f) W_O
       content = mean(A_o, A_f) (V_o - V_f) W_O

5. Alignment of each source/head contribution with:
   - the same-layer prompt-last attention difference;
   - the same-layer NEW prompt-last block increment;
   - the final-layer prompt-last difference.

The script then compares every error group with both_correct, so that one can
ask whether errors are associated with:

- weak position-sensitive object-token states;
- weak object-value writes into prompt-last;
- more opposing/canceling object paths;
- weak prompt-last increments;
- weak final prompt-last counterfactual differences.

No centroid and no trained probe are used. These are descriptive local
computational decompositions. A causal claim requires a later edge
replacement/restoration experiment.

Repository dependency
---------------------
The file should be placed in AdaptVis next to the repository's existing
`extract_two_object_relation_states.py`, which supplies the dataset loader and
model specifications. It does NOT import any of the previous analysis scripts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import inspect
import json
import math
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


VERSION = "coco-flip-last-increment-pipeline-v2"
RELATIONS = ("left", "right", "above", "below")
HORIZONTAL_RELATIONS = ("left", "right")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
PAIR_STATUSES = (
    "both_correct",
    "original_only_correct",
    "flipped_only_correct",
    "both_wrong",
)
AGGREGATE_STATUSES = ("all",) + PAIR_STATUSES
STATE_GROUPS = ("subject", "reference", "prompt_last")
SOURCE_GROUPS = (
    "visual_all",
    "subject",
    "reference",
    "relation",
    "options",
    "query_words",
    "instruction_other",
    "question_other",
    "chat_prefix",
    "chat_suffix",
    "other_text",
    "self",
)

DEFAULT_PROMPT_FILES = {
    "coco_two": Path(
        "prompts/COCO_QA_two_obj_with_answer_four_options.jsonl"
    ),
}

# Adds the original Qwen2-VL checkpoint without editing the repository loader.
EXTRA_MODEL_SPECS = {
    "qwen2-vl-7b": SimpleNamespace(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        model_class="Qwen2VLForConditionalGeneration",
        dtype_name="bfloat16",
        trust_remote_code=False,
    ),
}

STATE_METRICS = (
    "input_delta_norm",
    "attention_delta_norm",
    "mlp_delta_norm",
    "increment_delta_norm",
    "output_delta_norm",
    "output_delta_to_final_last_projection_fraction",
    "attention_fraction_of_increment",
    "mlp_fraction_of_increment",
)
SOURCE_METRICS = (
    "delta_norm",
    "routing_norm",
    "content_norm",
    "routing_share_by_norm",
    "delta_to_last_attention_projection_fraction",
    "delta_to_last_increment_projection_fraction",
    "content_to_last_increment_projection_fraction",
    "routing_to_last_increment_projection_fraction",
    "delta_to_final_last_projection_fraction",
    "content_to_final_last_projection_fraction",
)
HEAD_METRICS = (
    "delta_norm",
    "routing_norm",
    "content_norm",
    "routing_share_by_norm",
    "delta_to_last_increment_projection_fraction",
    "content_to_last_increment_projection_fraction",
    "delta_to_final_last_projection_fraction",
    "content_to_final_last_projection_fraction",
)
DIAGNOSTIC_METRICS = (
    "subject_peak_input_delta_norm",
    "reference_peak_input_delta_norm",
    "object_state_peak_norm",
    "prompt_last_peak_increment_delta_norm",
    "prompt_last_peak_output_delta_norm",
    "final_last_delta_norm",
    "subject_content_transfer_peak",
    "reference_content_transfer_peak",
    "object_content_transfer_peak",
    "object_content_transfer_mean",
    "object_content_negative_rate",
    "visual_transfer_peak",
    "options_transfer_peak",
)

STANDARD_OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?\s*Answer\s+with",
    flags=re.IGNORECASE | re.DOTALL,
)


# -----------------------------------------------------------------------------
# Arguments and repository/model utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="coco_two",
        choices=["coco_two"],
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default=None,
        help=(
            "Defaults to prompts/COCO_QA_two_obj_with_answer_four_options.jsonl."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
        help="Use eager for this analysis.",
    )
    parser.add_argument(
        "--state-layers",
        default="all",
        help=(
            "Layers for subject/reference/prompt-last trajectories. "
            "Use all, auto:N, or comma-separated zero-based indices."
        ),
    )
    parser.add_argument(
        "--edge-layers",
        default="19,20,21,22,23,24,25,26,27,28,29,30",
        help=(
            "Layers for source/head -> prompt-last decomposition. "
            "Use all, auto:N, or comma-separated zero-based indices."
        ),
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCE_GROUPS),
        help=f"Comma-separated source groups; allowed={SOURCE_GROUPS}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit before left/right filtering.",
    )
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument(
        "--save-head-details",
        action="store_true",
        help="Save every per-sample head row. This can be large.",
    )
    parser.add_argument("--top-k-report", type=int, default=25)
    parser.add_argument(
        "--replay-tolerance",
        type=float,
        default=5e-3,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def import_two_object_module() -> Any:
    return importlib.import_module("extract_two_object_relation_states")


def merged_model_specs(module: Any) -> Dict[str, Any]:
    specs = dict(getattr(module, "SPECS", {}) or {})
    specs.update(EXTRA_MODEL_SPECS)
    return specs


def resolve_prompt_path(args: argparse.Namespace) -> Path:
    path = (
        Path(args.prompt_jsonl)
        if args.prompt_jsonl
        else DEFAULT_PROMPT_FILES[args.dataset]
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing standard prompt file: {path}. "
            "Pass it explicitly with --prompt-jsonl."
        )
    return path


def extract_standard_user_text(raw_question: str) -> str:
    text = str(raw_question).strip()
    text = re.sub(r"^\s*<image>\s*", "", text, flags=re.IGNORECASE)
    match = re.search(
        r"\bUSER\s*:\s*(.*?)(?:\s*\bASSISTANT\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = match.group(1)
    return text.strip()


def standard_answer_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def parse_standard_objects(question_text: str) -> Tuple[str, str]:
    compact = re.sub(r"\s+", " ", str(question_text)).strip()
    match = STANDARD_OBJECT_RE.search(compact)
    if not match:
        raise ValueError(
            "Could not parse subject/reference from standard question: "
            f"{compact!r}"
        )
    subject = match.group(1).strip()
    reference = match.group(2).strip()
    if not subject or not reference:
        raise ValueError(f"Empty subject/reference in question: {compact!r}")
    return subject, reference


def load_standard_prompts(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            required = {"id", "question", "answer"}
            if not required.issubset(row):
                raise ValueError(
                    f"{path}:{line_no} must contain id/question/answer; "
                    f"keys={sorted(row.keys())}"
                )
            sid = int(row["id"])
            if sid in rows:
                raise ValueError(f"Duplicate prompt id={sid} in {path}")
            raw_question = str(row["question"])
            question_text = extract_standard_user_text(raw_question)
            subject, reference = parse_standard_objects(question_text)
            rows[sid] = {
                "id": sid,
                "raw_question": raw_question,
                "question_text": question_text,
                "answer_raw": standard_answer_value(row["answer"]),
                "subject": subject,
                "reference": reference,
            }
    if not rows:
        raise RuntimeError(f"No standard questions loaded from {path}")
    return rows


def build_prompt(processor: Any, question_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question_text},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question_text


def configure_processor(model: Any, processor: Any) -> None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if (
        vision_config is not None
        and hasattr(processor, "patch_size")
        and hasattr(vision_config, "patch_size")
    ):
        processor.patch_size = int(vision_config.patch_size)

    strategy = getattr(config, "vision_feature_select_strategy", None)
    if (
        strategy is not None
        and hasattr(processor, "vision_feature_select_strategy")
    ):
        processor.vision_feature_select_strategy = str(strategy)

    if (
        getattr(config, "model_type", "") == "llava"
        and hasattr(processor, "num_additional_image_tokens")
    ):
        processor.num_additional_image_tokens = 1


def move_batch(batch: Any, device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    output = tokenizer(text, add_special_tokens=False)
    ids = output.input_ids
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(value) for value in ids]


def find_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
) -> List[int]:
    if not needle:
        return []
    width = len(needle)
    return [
        start
        for start in range(len(haystack) - width + 1)
        if list(haystack[start : start + width]) == list(needle)
    ]


def phrase_surface_variants(phrase: str) -> List[str]:
    raw = str(phrase).strip()
    candidates = [
        raw,
        raw.lower(),
        raw.upper(),
        raw.title(),
        raw.capitalize(),
    ]
    return list(dict.fromkeys(value for value in candidates if value))


def find_phrase_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    phrase: str,
    *,
    include_article_variants: bool = False,
) -> List[Tuple[int, int]]:
    variants: List[str] = []
    for surface in phrase_surface_variants(phrase):
        variants.extend([surface, " " + surface])
        if include_article_variants:
            variants.extend(["the " + surface, " the " + surface])

    seen_ids = set()
    spans: List[Tuple[int, int]] = []
    for variant in variants:
        ids = tokenizer_ids(tokenizer, variant)
        key = tuple(ids)
        if not ids or key in seen_ids:
            continue
        seen_ids.add(key)
        for start in find_subsequence(input_ids, ids):
            spans.append((start, start + len(ids) - 1))
    return sorted(set(spans))


def locate_object_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    subject: str,
    reference: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    subject_spans = find_phrase_spans(
        tokenizer,
        input_ids,
        subject,
        include_article_variants=True,
    )
    reference_spans = find_phrase_spans(
        tokenizer,
        input_ids,
        reference,
        include_article_variants=True,
    )
    valid = [
        (subject_span, reference_span)
        for subject_span in subject_spans
        for reference_span in reference_spans
        if subject_span[1] < reference_span[0]
    ]
    if not valid:
        raise ValueError(
            "Could not locate ordered object spans: "
            f"subject={subject!r}, reference={reference!r}, "
            f"subject_spans={subject_spans}, "
            f"reference_spans={reference_spans}"
        )
    return max(
        valid,
        key=lambda pair: (pair[1][0], pair[0][0]),
    )


def record_image(record: Any) -> Image.Image:
    if hasattr(record, "image"):
        return record.image.copy().convert("RGB")
    if hasattr(record, "image_path"):
        return Image.open(record.image_path).convert("RGB")
    raise TypeError("Record has neither image nor image_path")


def get_attr_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def resolve_decoder_layers(model: Any) -> Tuple[Any, str]:
    preferred = [
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in preferred:
        value = get_attr_path(model, path)
        if (
            isinstance(value, (torch.nn.ModuleList, list, tuple))
            and len(value) >= 4
        ):
            return value, path

    candidates: List[Tuple[str, Any]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) >= 4:
            candidates.append(
                (f"{name}.layers" if name else "layers", layers)
            )
    candidates.sort(
        key=lambda item: (
            0
            if any(
                keyword in item[0].lower()
                for keyword in ("language", "text")
            )
            else 1,
            1
            if any(
                keyword in item[0].lower()
                for keyword in ("visual", "vision")
            )
            else 0,
            -len(item[1]),
        )
    )
    if candidates:
        return candidates[0][1], candidates[0][0]
    raise RuntimeError("Could not resolve language-model decoder layers")


def candidate_token_id(tokenizer: Any, token: str) -> Optional[int]:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    try:
        token_id = int(token_id)
    except Exception:
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    if unknown is not None and token_id == int(unknown):
        return None
    return token_id


def resolve_visual_indices(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    input_ids: Sequence[int],
) -> List[int]:
    mm_type_ids = batch.get("mm_token_type_ids")
    if torch.is_tensor(mm_type_ids) and mm_type_ids.ndim == 2:
        direct = (
            torch.nonzero(mm_type_ids[0] == 1, as_tuple=False)
            .flatten()
            .tolist()
        )
        if direct:
            return [int(value) for value in direct]

    token_type_ids = batch.get("token_type_ids")
    if torch.is_tensor(token_type_ids) and token_type_ids.ndim == 2:
        unique = set(
            int(value)
            for value in token_type_ids[0].detach().cpu().tolist()
        )
        if 1 in unique:
            direct = (
                torch.nonzero(token_type_ids[0] == 1, as_tuple=False)
                .flatten()
                .tolist()
            )
            if direct:
                return [int(value) for value in direct]

    token_ids = set()
    objects = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        processor,
        getattr(processor, "tokenizer", None),
    ]
    for obj in objects:
        if obj is None:
            continue
        for name in ("image_token_id", "image_token_index"):
            value = getattr(obj, name, None)
            if isinstance(value, (int, np.integer)) and int(value) >= 0:
                token_ids.add(int(value))

    tokenizer = processor.tokenizer
    for token in (
        "<|image_pad|>",
        "<image>",
        "<image_token>",
        "<IMG_CONTEXT>",
    ):
        token_id = candidate_token_id(tokenizer, token)
        if token_id is not None:
            token_ids.add(token_id)

    indices = [
        index
        for index, token_id in enumerate(input_ids)
        if int(token_id) in token_ids
    ]
    if indices:
        return indices

    start_ids = {
        token_id
        for token in ("<|vision_start|>", "<image_start>", "<img>")
        if (token_id := candidate_token_id(tokenizer, token)) is not None
    }
    end_ids = {
        token_id
        for token in ("<|vision_end|>", "<image_end>", "</img>")
        if (token_id := candidate_token_id(tokenizer, token)) is not None
    }
    starts = [
        index
        for index, value in enumerate(input_ids)
        if int(value) in start_ids
    ]
    ends = [
        index
        for index, value in enumerate(input_ids)
        if int(value) in end_ids
    ]
    spans = [(start, end) for start in starts for end in ends if start < end]
    if spans:
        start, end = min(spans, key=lambda pair: pair[1] - pair[0])
        fallback = list(range(start + 1, end))
        if fallback:
            return fallback

    raise ValueError(
        "Could not identify visual-token positions. "
        f"Candidate image token IDs were {sorted(token_ids)}"
    )


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )
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
        "above": "above",
        "over": "above",
        "on": "above",
        "on top of": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
        "bottom": "below",
    }
    if text in exact:
        return exact[text]
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(below|under|beneath|bottom)\b", "below"),
        (r"\b(above|over|on top|top)\b", "above"),
        (r"\bon\b", "above"),
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


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    unknown = getattr(tokenizer, "unk_token_id", None)
    for relation in RELATIONS:
        surfaces = [
            relation,
            " " + relation,
            "\n" + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ]
        token_ids: List[int] = []
        for surface in surfaces:
            ids = tokenizer_ids(tokenizer, surface)
            if len(ids) != 1:
                continue
            token_id = int(ids[0])
            if unknown is not None and token_id == int(unknown):
                continue
            token_ids.append(token_id)
        token_ids = list(dict.fromkeys(token_ids))
        if not token_ids:
            raise RuntimeError(
                f"No one-token generation variant found for {relation!r}"
            )
        result[relation] = token_ids
    return result


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text.startswith("auto:"):
        stride = int(text.split(":", 1)[1])
        if stride <= 0:
            raise ValueError("auto stride must be positive")
        layers = list(range(stride - 1, n_layers, stride))
        if not layers or layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)
        return sorted(set(layers))

    result: List[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        layer = int(raw)
        if layer < 0:
            layer += n_layers
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"Layer {layer} outside valid range 0..{n_layers - 1}"
            )
        if layer not in result:
            result.append(layer)
    if not result:
        raise ValueError("No layers selected")
    return result


def parse_subset(
    value: str,
    allowed: Sequence[str],
    label: str,
) -> List[str]:
    allowed_set = set(allowed)
    result: List[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported {label}: {item}; allowed={sorted(allowed_set)}"
            )
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"No {label} selected")
    return result


# -----------------------------------------------------------------------------
# Token-role location
# -----------------------------------------------------------------------------


def find_subsequence_starts(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> List[int]:
    sequence_values = list(map(int, sequence))
    pattern_values = list(map(int, pattern))
    if not pattern_values or len(pattern_values) > len(sequence_values):
        return []
    width = len(pattern_values)
    return [
        start
        for start in range(len(sequence_values) - width + 1)
        if sequence_values[start : start + width] == pattern_values
    ]


def tokenize_without_specials(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = (
        encoded["input_ids"]
        if isinstance(encoded, Mapping)
        else encoded.input_ids
    )
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(value) for value in ids]


def locate_phrase_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    phrase: str,
) -> List[Tuple[int, int]]:
    raw = str(phrase)
    variants: List[str] = []
    for candidate in (
        raw,
        " " + raw,
        raw.strip(),
        " " + raw.strip(),
    ):
        if candidate and candidate not in variants:
            variants.append(candidate)

    spans = set()
    for candidate in variants:
        token_ids = tokenize_without_specials(tokenizer, candidate)
        for start in find_subsequence_starts(input_ids, token_ids):
            spans.add((start, start + len(token_ids) - 1))
    return sorted(spans)


def choose_span(
    spans: Sequence[Tuple[int, int]],
    *,
    min_start: Optional[int] = None,
    max_end: Optional[int] = None,
    prefer: str = "last",
) -> Optional[Tuple[int, int]]:
    valid = []
    for start, end in spans:
        if min_start is not None and start < min_start:
            continue
        if max_end is not None and end > max_end:
            continue
        valid.append((int(start), int(end)))
    if not valid:
        return None
    if prefer == "first":
        return min(valid, key=lambda item: (item[0], item[1]))
    return max(valid, key=lambda item: (item[0], item[1]))


def span_to_positions(
    span: Optional[Tuple[int, int]],
) -> List[int]:
    if span is None:
        return []
    return list(range(int(span[0]), int(span[1]) + 1))


def decode_single_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return str(text).replace("\n", "\\n")


def normalize_token_text(text: str) -> str:
    value = str(text).replace("\n", "\\n").replace("\t", "\\t")
    value = value.strip()
    return value if value else "<blank>"


def locate_semantic_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    question: str,
    subject_span: Tuple[int, int],
    reference_span: Tuple[int, int],
    text_positions: Sequence[int],
) -> Dict[str, List[int]]:
    subject = span_to_positions(subject_span)
    reference = span_to_positions(reference_span)
    subject_start, subject_end = map(int, subject_span)
    reference_start, reference_end = map(int, reference_span)
    text = sorted(set(map(int, text_positions)))
    if not text:
        raise RuntimeError("No text-token positions were identified")

    question_span = choose_span(
        locate_phrase_spans(tokenizer, input_ids, question),
        prefer="last",
    )
    question_start = question_span[0] if question_span else min(text)
    question_end = question_span[1] if question_span else max(text)

    where_span = choose_span(
        locate_phrase_spans(tokenizer, input_ids, "Where"),
        max_end=subject_start - 1,
        prefer="last",
    )

    copula_span = None
    for phrase in ("is", "are"):
        candidate = choose_span(
            locate_phrase_spans(tokenizer, input_ids, phrase),
            max_end=subject_start - 1,
            prefer="last",
        )
        if candidate is not None and (
            copula_span is None or candidate[0] > copula_span[0]
        ):
            copula_span = candidate

    connector_span = None
    connector_phrase = None
    for phrase in (
        "in relation to",
        "relative to",
        "with respect to",
    ):
        candidate = choose_span(
            locate_phrase_spans(tokenizer, input_ids, phrase),
            min_start=subject_end + 1,
            max_end=reference_start - 1,
            prefer="last",
        )
        if candidate is not None:
            connector_span = candidate
            connector_phrase = phrase
            break

    relation_keyword_span = None
    if connector_phrase is not None:
        keyword = (
            "relative" if "relative" in connector_phrase else "relation"
        )
        relation_keyword_span = choose_span(
            locate_phrase_spans(tokenizer, input_ids, keyword),
            min_start=subject_end + 1,
            max_end=reference_start - 1,
            prefer="last",
        )

    connector_to_span = choose_span(
        locate_phrase_spans(tokenizer, input_ids, "to"),
        min_start=(
            relation_keyword_span[1] + 1
            if relation_keyword_span
            else subject_end + 1
        ),
        max_end=reference_start - 1,
        prefer="first",
    )

    question_text = str(question)
    question_mark_index = question_text.find("?")
    instruction_text = (
        question_text[question_mark_index + 1 :].strip()
        if question_mark_index >= 0
        else ""
    )
    instruction_span = (
        choose_span(
            locate_phrase_spans(tokenizer, input_ids, instruction_text),
            min_start=reference_end + 1,
            max_end=question_end,
            prefer="last",
        )
        if instruction_text
        else None
    )

    def after_reference_word(
        word: str,
        prefer: str = "first",
    ) -> Optional[Tuple[int, int]]:
        return choose_span(
            locate_phrase_spans(tokenizer, input_ids, word),
            min_start=reference_end + 1,
            max_end=question_end,
            prefer=prefer,
        )

    answer_span = after_reference_word("Answer")
    with_span = after_reference_word("with")
    one_span = after_reference_word("one")
    spatial_span = after_reference_word("spatial")
    answer_relation_span = after_reference_word("relation", prefer="last")
    option_left_span = after_reference_word("left")
    option_right_span = after_reference_word("right")
    option_above_span = after_reference_word("above")
    option_below_span = after_reference_word("below")
    option_all = sorted(
        set(
            span_to_positions(option_left_span)
            + span_to_positions(option_right_span)
            + span_to_positions(option_above_span)
            + span_to_positions(option_below_span)
        )
    )

    semantic: Dict[str, List[int]] = {
        "subject": subject,
        "reference": reference,
        "both": sorted(set(subject + reference)),
        "where": span_to_positions(where_span),
        "copula": span_to_positions(copula_span),
        "relation_connector": span_to_positions(connector_span),
        "relation_keyword": span_to_positions(relation_keyword_span),
        "connector_to": span_to_positions(connector_to_span),
        "answer_instruction": span_to_positions(instruction_span),
        "answer": span_to_positions(answer_span),
        "with": span_to_positions(with_span),
        "one": span_to_positions(one_span),
        "spatial": span_to_positions(spatial_span),
        "answer_relation": span_to_positions(answer_relation_span),
        "option_left": span_to_positions(option_left_span),
        "option_right": span_to_positions(option_right_span),
        "option_above": span_to_positions(option_above_span),
        "option_below": span_to_positions(option_below_span),
        "option_all": option_all,
        "question_last": [int(question_end)],
        "chat_prefix": [position for position in text if position < question_start],
        "chat_suffix": [position for position in text if position > question_end],
        "prompt_last": [max(text)],
        "all_text": text,
    }

    known_question = set()
    for key in (
        "subject",
        "reference",
        "where",
        "copula",
        "relation_connector",
        "answer_instruction",
    ):
        known_question.update(semantic.get(key, []))
    semantic["question_other"] = [
        position
        for position in text
        if question_start <= position <= question_end
        and position not in known_question
    ]
    semantic["_question_span"] = [
        int(question_start),
        int(question_end),
    ]
    return semantic


def token_role(
    position: int,
    semantic: Mapping[str, Sequence[int]],
) -> str:
    priority = (
        "subject",
        "reference",
        "relation_keyword",
        "connector_to",
        "relation_connector",
        "where",
        "copula",
        "answer",
        "with",
        "one",
        "spatial",
        "option_left",
        "option_right",
        "option_above",
        "option_below",
        "answer_relation",
        "answer_instruction",
        "chat_prefix",
        "chat_suffix",
        "question_other",
    )
    for role in priority:
        if int(position) in set(map(int, semantic.get(role, []))):
            return role
    return "other_text"


def build_token_manifest(
    tokenizer: Any,
    input_ids: Sequence[int],
    text_positions: Sequence[int],
    semantic: Mapping[str, Sequence[int]],
) -> List[Dict[str, Any]]:
    question_bounds = list(semantic.get("_question_span", []))
    question_start = question_bounds[0] if len(question_bounds) == 2 else None
    question_end = question_bounds[1] if len(question_bounds) == 2 else None

    manifest: List[Dict[str, Any]] = []
    for text_rank, position in enumerate(
        sorted(set(map(int, text_positions)))
    ):
        token_id = int(input_ids[position])
        decoded = decode_single_token(tokenizer, token_id)
        role = token_role(position, semantic)
        role_positions = sorted(set(map(int, semantic.get(role, []))))
        role_rank = (
            role_positions.index(position)
            if position in role_positions
            else -1
        )
        manifest.append(
            {
                "position": position,
                "text_rank": text_rank,
                "token_id": token_id,
                "token_text": decoded,
                "token_text_norm": normalize_token_text(decoded),
                "token_role": role,
                "role_rank": role_rank,
                "inside_question": bool(
                    question_start is not None
                    and question_end is not None
                    and question_start <= position <= question_end
                ),
            }
        )
    return manifest


def broad_role(token_role_name: str) -> str:
    role = str(token_role_name)
    if role == "subject":
        return "subject"
    if role == "reference":
        return "reference"
    if role in {
        "relation_keyword",
        "connector_to",
        "relation_connector",
    }:
        return "relation"
    if role in {
        "option_left",
        "option_right",
        "option_above",
        "option_below",
    }:
        return "options"
    if role in {"where", "copula"}:
        return "query_words"
    if role in {
        "answer",
        "with",
        "one",
        "spatial",
        "answer_relation",
        "answer_instruction",
    }:
        return "instruction_other"
    if role == "question_other":
        return "question_other"
    if role == "chat_prefix":
        return "chat_prefix"
    if role == "chat_suffix":
        return "chat_suffix"
    return "other_text"


def build_source_groups(
    *,
    sequence_length: int,
    visual_indices: Sequence[int],
    token_manifest: Sequence[Mapping[str, Any]],
    target_positions: Sequence[int],
) -> Dict[str, List[int]]:
    target_set = set(map(int, target_positions))
    visual_set = {
        int(position)
        for position in visual_indices
        if 0 <= int(position) < sequence_length
    }
    manifest_by_position = {
        int(item["position"]): item
        for item in token_manifest
    }
    groups: Dict[str, List[int]] = {
        name: [] for name in SOURCE_GROUPS
    }

    for position in range(sequence_length):
        if position in target_set:
            groups["self"].append(position)
            continue
        if position in visual_set:
            groups["visual_all"].append(position)
            continue
        item = manifest_by_position.get(position)
        if item is None:
            groups["other_text"].append(position)
            continue
        groups[broad_role(str(item["token_role"]))].append(position)

    for name in groups:
        groups[name] = sorted(set(groups[name]))

    covered = [
        position
        for positions in groups.values()
        for position in positions
    ]
    counts = Counter(covered)
    missing = [
        position
        for position in range(sequence_length)
        if counts[position] == 0
    ]
    duplicates = [
        position
        for position, count in counts.items()
        if count > 1
    ]
    if missing or duplicates:
        raise RuntimeError(
            f"Invalid source partition: missing={missing[:20]}, "
            f"duplicates={duplicates[:20]}"
        )
    return groups


def state_group_positions(
    semantic: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for name in STATE_GROUPS:
        positions = sorted(set(map(int, semantic.get(name, []))))
        if not positions:
            raise RuntimeError(f"State group {name} has no positions")
        groups[name] = positions
    return groups


# -----------------------------------------------------------------------------
# Attention/state capture and exact contribution decomposition
# -----------------------------------------------------------------------------


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(
        f"Cannot locate attention output tensor in {type(output).__name__}"
    )


def find_attention_weights(output: Any) -> torch.Tensor:
    candidates: List[torch.Tensor] = []
    if isinstance(output, (tuple, list)):
        candidates.extend(
            item for item in output if torch.is_tensor(item)
        )
    elif isinstance(output, Mapping):
        candidates.extend(
            value for value in output.values() if torch.is_tensor(value)
        )
    else:
        for name in ("attn_weights", "attention_weights", "attentions"):
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                candidates.append(value)
    four_dimensional = [
        tensor for tensor in candidates if tensor.ndim == 4
    ]
    if not four_dimensional:
        raise RuntimeError(
            "Standalone eager attention replay did not return a 4D "
            "attention-weight tensor"
        )
    return four_dimensional[0]


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "logits",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "logits",
            None,
        ),
    ]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("No language-model logits found")


def score_relations(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for relation in RELATIONS:
        ids = [
            int(token_id)
            for token_id in token_map[relation]
            if 0 <= int(token_id) < logits.numel()
        ]
        if not ids:
            raise RuntimeError(f"No token variants found for {relation}")
        index = torch.tensor(
            ids,
            device=logits.device,
            dtype=torch.long,
        )
        scores[relation] = float(
            logits.index_select(0, index).max().detach().cpu()
        )
    return scores


def nested_detach(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(nested_detach(item) for item in value)
    if isinstance(value, list):
        return [nested_detach(item) for item in value]
    if isinstance(value, dict):
        return {
            key: nested_detach(item)
            for key, item in value.items()
        }
    return value


def locate_hidden_states(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    value = kwargs.get("hidden_states")
    if torch.is_tensor(value):
        return value
    for item in args:
        if torch.is_tensor(item) and item.ndim == 3:
            return item
    raise RuntimeError("Cannot locate hidden_states in module inputs")


def resolve_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise RuntimeError(
        f"Cannot locate self-attention inside {type(layer).__name__}"
    )


@dataclass
class AttentionCall:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    actual_target_output: torch.Tensor


class CaptureAttentionCalls:
    def __init__(
        self,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
        target_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer_indices = list(layer_indices)
        self.target_positions = sorted(set(map(int, target_positions)))
        self.handles: List[Any] = []
        self.pre_inputs: Dict[
            int,
            Tuple[Tuple[Any, ...], Dict[str, Any]],
        ] = {}
        self.actual_outputs: Dict[int, torch.Tensor] = {}
        self.events: Counter = Counter()

    def __enter__(self) -> "CaptureAttentionCalls":
        for layer_index in self.layer_indices:
            attention = resolve_self_attention(
                self.decoder_layers[layer_index]
            )

            def make_pre_hook(index: int):
                def pre_hook(
                    _module: Any,
                    args: Tuple[Any, ...],
                    kwargs: Dict[str, Any],
                ) -> None:
                    self.pre_inputs[index] = (
                        tuple(nested_detach(args)),
                        dict(nested_detach(kwargs)),
                    )
                    self.events[(index, "pre")] += 1

                return pre_hook

            def make_post_hook(index: int):
                def post_hook(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    _kwargs: Dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = first_tensor(output)
                    position_index = torch.tensor(
                        self.target_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    self.actual_outputs[index] = (
                        hidden[0]
                        .index_select(0, position_index)
                        .detach()
                        .float()
                        .cpu()
                    )
                    self.events[(index, "post")] += 1

                return post_hook

            self.handles.append(
                attention.register_forward_pre_hook(
                    make_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    make_post_hook(layer_index),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def materialize(self) -> Dict[int, AttentionCall]:
        result: Dict[int, AttentionCall] = {}
        for layer_index in self.layer_indices:
            pre_count = int(self.events[(layer_index, "pre")])
            post_count = int(self.events[(layer_index, "post")])
            if pre_count != 1 or post_count != 1:
                raise RuntimeError(
                    f"Layer {layer_index}: expected one attention call; "
                    f"pre={pre_count}, post={post_count}"
                )
            args, kwargs = self.pre_inputs[layer_index]
            result[layer_index] = AttentionCall(
                args=args,
                kwargs=kwargs,
                actual_target_output=self.actual_outputs[layer_index],
            )
        return result


@dataclass
class StateCapture:
    positions: List[int]
    block_inputs: Dict[int, torch.Tensor]
    attention_outputs: Dict[int, torch.Tensor]
    block_outputs: Dict[int, torch.Tensor]


class CaptureStateComponents:
    def __init__(
        self,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
        positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer_indices = list(layer_indices)
        self.positions = sorted(set(map(int, positions)))
        self.handles: List[Any] = []
        self.block_inputs: Dict[int, torch.Tensor] = {}
        self.attention_outputs: Dict[int, torch.Tensor] = {}
        self.block_outputs: Dict[int, torch.Tensor] = {}
        self.events: Counter = Counter()

    def _select(self, hidden: torch.Tensor) -> torch.Tensor:
        position_index = torch.tensor(
            self.positions,
            device=hidden.device,
            dtype=torch.long,
        )
        return (
            hidden[0]
            .index_select(0, position_index)
            .detach()
            .float()
            .cpu()
        )

    def __enter__(self) -> "CaptureStateComponents":
        for layer_index in self.layer_indices:
            layer = self.decoder_layers[layer_index]
            attention = resolve_self_attention(layer)

            def make_block_pre(index: int):
                def hook(
                    _module: Any,
                    args: Tuple[Any, ...],
                    kwargs: Dict[str, Any],
                ) -> None:
                    hidden = locate_hidden_states(args, kwargs)
                    self.block_inputs[index] = self._select(hidden)
                    self.events[(index, "block_pre")] += 1

                return hook

            def make_attention_post(index: int):
                def hook(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    _kwargs: Dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = first_tensor(output)
                    self.attention_outputs[index] = self._select(hidden)
                    self.events[(index, "attention_post")] += 1

                return hook

            def make_block_post(index: int):
                def hook(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    _kwargs: Dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = first_tensor(output)
                    self.block_outputs[index] = self._select(hidden)
                    self.events[(index, "block_post")] += 1

                return hook

            self.handles.append(
                layer.register_forward_pre_hook(
                    make_block_pre(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    make_attention_post(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    make_block_post(layer_index),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def materialize(self) -> StateCapture:
        for layer_index in self.layer_indices:
            for event_name, storage in (
                ("block_pre", self.block_inputs),
                ("attention_post", self.attention_outputs),
                ("block_post", self.block_outputs),
            ):
                count = int(self.events[(layer_index, event_name)])
                if count != 1 or layer_index not in storage:
                    raise RuntimeError(
                        f"Layer {layer_index}: {event_name} count={count}, "
                        "expected exactly one captured tensor"
                    )
        return StateCapture(
            positions=list(self.positions),
            block_inputs=dict(self.block_inputs),
            attention_outputs=dict(self.attention_outputs),
            block_outputs=dict(self.block_outputs),
        )


@dataclass
class LayerTrace:
    target_positions: List[int]
    attention_weights: torch.Tensor  # [H,T,S], CPU float32
    value_states: torch.Tensor  # [H,S,Dh], CPU float32
    attention_output: torch.Tensor  # [T,D], CPU float32
    o_proj_weight: torch.Tensor  # [D,H,Dh], CPU float32
    replay_max_abs_error: float
    replay_relative_error: float


@dataclass
class RunTrace:
    scores: Dict[str, float]
    prediction: str
    state: StateCapture
    edges: Dict[int, LayerTrace]


def accepts_keyword(module: Any, name: str) -> bool:
    signature = inspect.signature(module.forward)
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def repeat_key_value(
    value_states: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    n_kv_heads = int(value_states.shape[1])
    if n_kv_heads == n_heads:
        return value_states
    if n_heads % n_kv_heads != 0:
        raise RuntimeError(
            f"Cannot repeat {n_kv_heads} KV heads to {n_heads} query heads"
        )
    repeat = n_heads // n_kv_heads
    return value_states.repeat_interleave(repeat, dim=1)


def project_value_states(
    attention: Any,
    hidden_states: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    value_projection = getattr(attention, "v_proj", None)
    output_projection = getattr(attention, "o_proj", None)
    if value_projection is None or output_projection is None:
        raise RuntimeError(
            f"{type(attention).__name__} must expose v_proj and o_proj"
        )

    values = value_projection(hidden_states)
    if values.ndim != 3:
        raise RuntimeError(
            f"v_proj returned {tuple(values.shape)}, expected [B,S,D]"
        )

    n_kv_heads = getattr(attention, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(
            getattr(attention, "config", None),
            "num_key_value_heads",
            None,
        )
    if n_kv_heads is None:
        n_kv_heads = n_heads
    n_kv_heads = int(n_kv_heads)

    if values.shape[-1] % n_kv_heads != 0:
        raise RuntimeError(
            f"v_proj dimension {values.shape[-1]} is not divisible by "
            f"num_key_value_heads={n_kv_heads}"
        )
    head_dim = int(values.shape[-1] // n_kv_heads)
    values = (
        values.view(
            values.shape[0],
            values.shape[1],
            n_kv_heads,
            head_dim,
        )
        .transpose(1, 2)
        .contiguous()
    )
    return repeat_key_value(values, n_heads)


def reshape_o_projection(
    attention: Any,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    output_projection = getattr(attention, "o_proj", None)
    if output_projection is None or not hasattr(output_projection, "weight"):
        raise RuntimeError(
            f"{type(attention).__name__} has no o_proj.weight"
        )
    weight = output_projection.weight.detach().float()
    if weight.ndim != 2:
        raise RuntimeError(
            f"o_proj.weight must be 2D, got {tuple(weight.shape)}"
        )
    if weight.shape[1] != n_heads * head_dim:
        raise RuntimeError(
            f"o_proj input dimension {weight.shape[1]} != "
            f"{n_heads} * {head_dim}"
        )
    return weight.view(weight.shape[0], n_heads, head_dim)


def replay_attention_layer(
    attention: Any,
    call: AttentionCall,
    target_positions: Sequence[int],
) -> LayerTrace:
    args = call.args
    kwargs = dict(call.kwargs)
    hidden_states = locate_hidden_states(args, kwargs)

    if not accepts_keyword(attention, "output_attentions"):
        raise RuntimeError(
            f"{type(attention).__name__}.forward does not accept "
            "output_attentions"
        )
    kwargs["output_attentions"] = True
    if accepts_keyword(attention, "use_cache"):
        kwargs["use_cache"] = False

    config = getattr(attention, "config", None)
    old_implementation = None
    if config is not None and hasattr(config, "_attn_implementation"):
        old_implementation = config._attn_implementation
        config._attn_implementation = "eager"

    try:
        with torch.inference_mode():
            replay_output = attention(*args, **kwargs)
    finally:
        if (
            config is not None
            and old_implementation is not None
            and hasattr(config, "_attn_implementation")
        ):
            config._attn_implementation = old_implementation

    replay_hidden = first_tensor(replay_output)
    weights = find_attention_weights(replay_output)
    if weights.shape[0] != 1:
        raise RuntimeError(
            f"Expected batch size 1, got weights {tuple(weights.shape)}"
        )

    positions = sorted(set(map(int, target_positions)))
    target_index_gpu = torch.tensor(
        positions,
        device=weights.device,
        dtype=torch.long,
    )
    replay_target = (
        replay_hidden[0]
        .index_select(0, target_index_gpu)
        .detach()
        .float()
        .cpu()
    )
    actual_target = call.actual_target_output
    if replay_target.shape != actual_target.shape:
        raise RuntimeError(
            "Replay/actual attention target shapes differ: "
            f"{tuple(replay_target.shape)} vs {tuple(actual_target.shape)}"
        )
    replay_difference = replay_target - actual_target
    replay_max_abs_error = float(replay_difference.abs().max())
    replay_relative_error = float(
        replay_difference.norm()
        / actual_target.norm().clamp_min(1e-12)
    )

    target_weights = (
        weights[0]
        .index_select(1, target_index_gpu)
        .detach()
        .float()
        .cpu()
    )
    n_heads = int(target_weights.shape[0])

    with torch.inference_mode():
        values = project_value_states(
            attention,
            hidden_states,
            n_heads=n_heads,
        )
    value_states = values[0].detach().float().cpu()
    head_dim = int(value_states.shape[-1])
    o_proj_weight = (
        reshape_o_projection(
            attention,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        .detach()
        .float()
        .cpu()
    )

    if target_weights.shape[-1] != value_states.shape[1]:
        raise RuntimeError(
            "Attention key length and value sequence length differ: "
            f"{target_weights.shape[-1]} vs {value_states.shape[1]}"
        )

    return LayerTrace(
        target_positions=positions,
        attention_weights=target_weights,
        value_states=value_states,
        attention_output=replay_target,
        o_proj_weight=o_proj_weight,
        replay_max_abs_error=replay_max_abs_error,
        replay_relative_error=replay_relative_error,
    )


def run_trace(
    *,
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    state_layers: Sequence[int],
    edge_layers: Sequence[int],
    state_positions: Sequence[int],
    last_positions: Sequence[int],
) -> RunTrace:
    with torch.inference_mode():
        with CaptureStateComponents(
            decoder_layers,
            state_layers,
            state_positions,
        ) as state_capture, CaptureAttentionCalls(
            decoder_layers,
            edge_layers,
            last_positions,
        ) as attention_capture:
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

    logits = extract_logits(outputs)[0, -1, :]
    scores = score_relations(logits, token_map)
    prediction = max(RELATIONS, key=lambda relation: scores[relation])
    state = state_capture.materialize()
    attention_calls = attention_capture.materialize()

    edge_traces: Dict[int, LayerTrace] = {}
    for layer_index in edge_layers:
        attention = resolve_self_attention(decoder_layers[layer_index])
        edge_traces[layer_index] = replay_attention_layer(
            attention,
            attention_calls[layer_index],
            last_positions,
        )

    del outputs, logits, attention_calls
    return RunTrace(
        scores=scores,
        prediction=prediction,
        state=state,
        edges=edge_traces,
    )


def project_heads(
    head_vectors: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    if head_vectors.ndim != 2 or o_proj_weight.ndim != 3:
        raise RuntimeError(
            f"Bad projection shapes: {tuple(head_vectors.shape)}, "
            f"{tuple(o_proj_weight.shape)}"
        )
    if head_vectors.shape[0] != o_proj_weight.shape[1]:
        raise RuntimeError("Head count mismatch in output projection")
    if head_vectors.shape[1] != o_proj_weight.shape[2]:
        raise RuntimeError("Head dimension mismatch in output projection")
    return torch.einsum("hd,ohd->ho", head_vectors, o_proj_weight)


def compute_group_head_vectors(
    *,
    original: LayerTrace,
    flipped: LayerTrace,
    target_positions: Sequence[int],
    source_positions: Sequence[int],
) -> Dict[str, torch.Tensor]:
    if original.target_positions != flipped.target_positions:
        raise RuntimeError("Original/flip target positions differ")
    if original.attention_weights.shape != flipped.attention_weights.shape:
        raise RuntimeError("Original/flip attention shapes differ")
    if original.value_states.shape != flipped.value_states.shape:
        raise RuntimeError("Original/flip value-state shapes differ")

    target_lookup = {
        int(position): index
        for index, position in enumerate(original.target_positions)
    }
    target_local = [
        target_lookup[int(position)]
        for position in target_positions
        if int(position) in target_lookup
    ]
    if not target_local:
        raise RuntimeError("Target group has no traced positions")
    if not source_positions:
        raise RuntimeError("Source group is empty")

    target_index = torch.tensor(target_local, dtype=torch.long)
    source_index = torch.tensor(
        sorted(set(map(int, source_positions))),
        dtype=torch.long,
    )

    attention_original = (
        original.attention_weights
        .index_select(1, target_index)
        .index_select(2, source_index)
    )
    attention_flipped = (
        flipped.attention_weights
        .index_select(1, target_index)
        .index_select(2, source_index)
    )
    values_original = original.value_states.index_select(1, source_index)
    values_flipped = flipped.value_states.index_select(1, source_index)

    target_count = float(len(target_local))
    original_head = torch.einsum(
        "hts,hsd->hd",
        attention_original,
        values_original,
    ) / target_count
    flipped_head = torch.einsum(
        "hts,hsd->hd",
        attention_flipped,
        values_flipped,
    ) / target_count
    routing_head = torch.einsum(
        "hts,hsd->hd",
        attention_original - attention_flipped,
        0.5 * (values_original + values_flipped),
    ) / target_count
    content_head = torch.einsum(
        "hts,hsd->hd",
        0.5 * (attention_original + attention_flipped),
        values_original - values_flipped,
    ) / target_count

    original_residual = project_heads(
        original_head,
        original.o_proj_weight,
    )
    flipped_residual = project_heads(
        flipped_head,
        flipped.o_proj_weight,
    )
    routing_residual = project_heads(
        routing_head,
        original.o_proj_weight,
    )
    content_residual = project_heads(
        content_head,
        original.o_proj_weight,
    )

    delta = original_residual - flipped_residual
    decomposition_error = delta - routing_residual - content_residual
    return {
        "delta_heads": delta,
        "routing_heads": routing_residual,
        "content_heads": content_residual,
        "error_heads": decomposition_error,
        "delta_total": delta.sum(dim=0),
        "routing_total": routing_residual.sum(dim=0),
        "content_total": content_residual.sum(dim=0),
        "error_total": decomposition_error.sum(dim=0),
    }


# -----------------------------------------------------------------------------
# Vector/state metrics
# -----------------------------------------------------------------------------


def select_rows(
    tensor: torch.Tensor,
    all_positions: Sequence[int],
    requested_positions: Sequence[int],
) -> torch.Tensor:
    lookup = {
        int(position): index
        for index, position in enumerate(all_positions)
    }
    local = [
        lookup[int(position)]
        for position in requested_positions
        if int(position) in lookup
    ]
    if not local:
        raise RuntimeError(
            f"No requested positions found: requested={requested_positions}, "
            f"available={all_positions}"
        )
    index = torch.tensor(local, dtype=torch.long)
    return tensor.index_select(0, index)


def mean_rows(
    tensor: torch.Tensor,
    all_positions: Sequence[int],
    requested_positions: Sequence[int],
) -> torch.Tensor:
    return select_rows(
        tensor,
        all_positions,
        requested_positions,
    ).mean(dim=0)


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def vector_norm(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().norm())


def safe_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    denominator = left_cpu.norm() * right_cpu.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(left_cpu, right_cpu) / denominator)


def projection_fraction(
    vector: torch.Tensor,
    target: torch.Tensor,
) -> float:
    vector_cpu = vector.detach().float().cpu()
    target_cpu = target.detach().float().cpu()
    denominator = float(target_cpu.pow(2).sum())
    if denominator <= 1e-12:
        return float("nan")
    return float(torch.dot(vector_cpu, target_cpu) / denominator)


def vector_stats(
    vector: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    return {
        f"{prefix}_cosine": safe_cosine(vector, target),
        f"{prefix}_projection_fraction": projection_fraction(
            vector,
            target,
        ),
    }


def state_component_vectors(
    *,
    original: RunTrace,
    flipped: RunTrace,
    layer: int,
    positions: Sequence[int],
) -> Dict[str, torch.Tensor]:
    original_input = mean_rows(
        original.state.block_inputs[layer],
        original.state.positions,
        positions,
    )
    flipped_input = mean_rows(
        flipped.state.block_inputs[layer],
        flipped.state.positions,
        positions,
    )
    original_attention = mean_rows(
        original.state.attention_outputs[layer],
        original.state.positions,
        positions,
    )
    flipped_attention = mean_rows(
        flipped.state.attention_outputs[layer],
        flipped.state.positions,
        positions,
    )
    original_output = mean_rows(
        original.state.block_outputs[layer],
        original.state.positions,
        positions,
    )
    flipped_output = mean_rows(
        flipped.state.block_outputs[layer],
        flipped.state.positions,
        positions,
    )

    input_delta = original_input - flipped_input
    attention_delta = original_attention - flipped_attention
    output_delta = original_output - flipped_output
    increment_delta = output_delta - input_delta
    mlp_delta = increment_delta - attention_delta
    decomposition_error = (
        output_delta - input_delta - attention_delta - mlp_delta
    )
    return {
        "input_delta": input_delta,
        "attention_delta": attention_delta,
        "mlp_delta": mlp_delta,
        "increment_delta": increment_delta,
        "output_delta": output_delta,
        "decomposition_error": decomposition_error,
    }


def make_state_rows(
    *,
    original: RunTrace,
    flipped: RunTrace,
    layers: Sequence[int],
    group_positions: Mapping[str, Sequence[int]],
    final_last_delta: torch.Tensor,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[Tuple[int, str], Dict[str, torch.Tensor]],
]:
    rows: List[Dict[str, Any]] = []
    vectors: Dict[
        Tuple[int, str],
        Dict[str, torch.Tensor],
    ] = {}

    for layer in layers:
        for group, positions in group_positions.items():
            components = state_component_vectors(
                original=original,
                flipped=flipped,
                layer=layer,
                positions=positions,
            )
            vectors[(layer, group)] = components
            output_norm = max(
                vector_norm(components["output_delta"]),
                1e-12,
            )
            row: Dict[str, Any] = {
                "layer": int(layer),
                "token_group": group,
                "input_delta_norm": vector_norm(
                    components["input_delta"]
                ),
                "attention_delta_norm": vector_norm(
                    components["attention_delta"]
                ),
                "mlp_delta_norm": vector_norm(
                    components["mlp_delta"]
                ),
                "increment_delta_norm": vector_norm(
                    components["increment_delta"]
                ),
                "output_delta_norm": vector_norm(
                    components["output_delta"]
                ),
                "decomposition_relative_error": (
                    vector_norm(components["decomposition_error"])
                    / output_norm
                ),
                "attention_fraction_of_increment": projection_fraction(
                    components["attention_delta"],
                    components["increment_delta"],
                ),
                "mlp_fraction_of_increment": projection_fraction(
                    components["mlp_delta"],
                    components["increment_delta"],
                ),
                "input_fraction_of_output": projection_fraction(
                    components["input_delta"],
                    components["output_delta"],
                ),
                "increment_fraction_of_output": projection_fraction(
                    components["increment_delta"],
                    components["output_delta"],
                ),
            }
            for name in (
                "input_delta",
                "attention_delta",
                "mlp_delta",
                "increment_delta",
                "output_delta",
            ):
                row.update(
                    vector_stats(
                        components[name],
                        final_last_delta,
                        f"{name}_to_final_last",
                    )
                )
            rows.append(row)
    return rows, vectors


def make_edge_rows(
    *,
    original: RunTrace,
    flipped: RunTrace,
    edge_layers: Sequence[int],
    source_groups: Sequence[str],
    source_positions: Mapping[str, Sequence[int]],
    last_positions: Sequence[int],
    last_state_vectors: Mapping[
        Tuple[int, str],
        Mapping[str, torch.Tensor],
    ],
    final_last_delta: torch.Tensor,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    group_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    reconstruction_rows: List[Dict[str, Any]] = []

    for layer in edge_layers:
        local = last_state_vectors[(layer, "prompt_last")]
        attention_delta = local["attention_delta"]
        increment_delta = local["increment_delta"]
        output_delta = local["output_delta"]

        all_selected_delta = torch.zeros_like(attention_delta)
        all_selected_routing = torch.zeros_like(attention_delta)
        all_selected_content = torch.zeros_like(attention_delta)

        for source_group in source_groups:
            positions = list(source_positions[source_group])
            if not positions:
                continue
            vectors = compute_group_head_vectors(
                original=original.edges[layer],
                flipped=flipped.edges[layer],
                target_positions=last_positions,
                source_positions=positions,
            )
            all_selected_delta += vectors["delta_total"]
            all_selected_routing += vectors["routing_total"]
            all_selected_content += vectors["content_total"]

            routing_norm = vector_norm(vectors["routing_total"])
            content_norm = vector_norm(vectors["content_total"])
            norm_denominator = routing_norm + content_norm
            group_row: Dict[str, Any] = {
                "layer": int(layer),
                "source_group": source_group,
                "n_source_positions": len(positions),
                "n_heads": int(vectors["delta_heads"].shape[0]),
                "delta_norm": vector_norm(vectors["delta_total"]),
                "routing_norm": routing_norm,
                "content_norm": content_norm,
                "routing_share_by_norm": (
                    routing_norm / norm_denominator
                    if norm_denominator > 1e-12
                    else float("nan")
                ),
                "decomposition_relative_error": (
                    vector_norm(vectors["error_total"])
                    / max(
                        vector_norm(vectors["delta_total"]),
                        1e-12,
                    )
                ),
            }
            for target_name, target_vector in (
                ("last_attention", attention_delta),
                ("last_increment", increment_delta),
                ("last_output", output_delta),
                ("final_last", final_last_delta),
            ):
                group_row.update(
                    vector_stats(
                        vectors["delta_total"],
                        target_vector,
                        f"delta_to_{target_name}",
                    )
                )
                group_row.update(
                    vector_stats(
                        vectors["routing_total"],
                        target_vector,
                        f"routing_to_{target_name}",
                    )
                )
                group_row.update(
                    vector_stats(
                        vectors["content_total"],
                        target_vector,
                        f"content_to_{target_name}",
                    )
                )
            group_rows.append(group_row)

            n_heads = int(vectors["delta_heads"].shape[0])
            for head in range(n_heads):
                delta = vectors["delta_heads"][head]
                routing = vectors["routing_heads"][head]
                content = vectors["content_heads"][head]
                error = vectors["error_heads"][head]
                routing_head_norm = vector_norm(routing)
                content_head_norm = vector_norm(content)
                head_denominator = routing_head_norm + content_head_norm
                head_row: Dict[str, Any] = {
                    "layer": int(layer),
                    "head": int(head),
                    "source_group": source_group,
                    "n_source_positions": len(positions),
                    "delta_norm": vector_norm(delta),
                    "routing_norm": routing_head_norm,
                    "content_norm": content_head_norm,
                    "routing_share_by_norm": (
                        routing_head_norm / head_denominator
                        if head_denominator > 1e-12
                        else float("nan")
                    ),
                    "decomposition_relative_error": (
                        vector_norm(error)
                        / max(vector_norm(delta), 1e-12)
                    ),
                }
                for target_name, target_vector in (
                    ("last_attention", attention_delta),
                    ("last_increment", increment_delta),
                    ("last_output", output_delta),
                    ("final_last", final_last_delta),
                ):
                    head_row.update(
                        vector_stats(
                            delta,
                            target_vector,
                            f"delta_to_{target_name}",
                        )
                    )
                    head_row.update(
                        vector_stats(
                            content,
                            target_vector,
                            f"content_to_{target_name}",
                        )
                    )
                head_rows.append(head_row)

        attention_reconstruction_error = (
            all_selected_delta - attention_delta
        )
        routing_content_error = (
            all_selected_delta
            - all_selected_routing
            - all_selected_content
        )
        block_decomposition_error = (
            local["output_delta"]
            - local["input_delta"]
            - local["attention_delta"]
            - local["mlp_delta"]
        )
        reconstruction_rows.append(
            {
                "layer": int(layer),
                "selected_sources": ",".join(source_groups),
                "last_attention_delta_norm": vector_norm(
                    attention_delta
                ),
                "selected_edge_sum_norm": vector_norm(
                    all_selected_delta
                ),
                "attention_reconstruction_relative_error": (
                    vector_norm(attention_reconstruction_error)
                    / max(vector_norm(attention_delta), 1e-12)
                ),
                "routing_content_relative_error": (
                    vector_norm(routing_content_error)
                    / max(vector_norm(all_selected_delta), 1e-12)
                ),
                "block_decomposition_relative_error": (
                    vector_norm(block_decomposition_error)
                    / max(
                        vector_norm(local["output_delta"]),
                        1e-12,
                    )
                ),
                "replay_max_abs_error": max(
                    original.edges[layer].replay_max_abs_error,
                    flipped.edges[layer].replay_max_abs_error,
                ),
                "replay_relative_error": max(
                    original.edges[layer].replay_relative_error,
                    flipped.edges[layer].replay_relative_error,
                ),
            }
        )
    return group_rows, head_rows, reconstruction_rows


# -----------------------------------------------------------------------------
# Streaming output and aggregation
# -----------------------------------------------------------------------------


class StreamingCSV:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.writer = None
        self.fields: Optional[List[str]] = None

    def write(self, row: Mapping[str, Any]) -> None:
        if self.file is None:
            self.fields = list(row.keys())
            self.file = self.path.open(
                "w",
                encoding="utf-8",
                newline="",
            )
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=self.fields,
            )
            self.writer.writeheader()
        assert self.writer is not None
        assert self.fields is not None
        extra = set(row.keys()) - set(self.fields)
        if extra:
            raise RuntimeError(
                f"{self.path.name}: fields changed after header: "
                f"{sorted(extra)}"
            )
        self.writer.writerow(
            {field: row.get(field) for field in self.fields}
        )

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None


@dataclass
class RunningMetric:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    positive: int = 0
    negative: int = 0
    zero: int = 0

    def update(self, value: Any) -> None:
        if not finite(value):
            return
        number = float(value)
        self.n += 1
        self.total += number
        self.total_sq += number * number
        if number > 0:
            self.positive += 1
        elif number < 0:
            self.negative += 1
        else:
            self.zero += 1

    def summary(self, prefix: str) -> Dict[str, Any]:
        if self.n == 0:
            return {
                f"n_{prefix}": 0,
                f"mean_{prefix}": None,
                f"std_{prefix}": None,
                f"positive_rate_{prefix}": None,
                f"negative_rate_{prefix}": None,
            }
        mean = self.total / self.n
        variance = max(
            0.0,
            self.total_sq / self.n - mean * mean,
        )
        return {
            f"n_{prefix}": self.n,
            f"mean_{prefix}": mean,
            f"std_{prefix}": math.sqrt(variance),
            f"positive_rate_{prefix}": self.positive / self.n,
            f"negative_rate_{prefix}": self.negative / self.n,
        }


class GroupAggregator:
    def __init__(
        self,
        key_names: Sequence[str],
        metric_names: Sequence[str],
    ) -> None:
        self.key_names = list(key_names)
        self.metric_names = list(metric_names)
        self.groups: Dict[
            Tuple[Any, ...],
            Dict[str, RunningMetric],
        ] = {}

    def update(
        self,
        key_values: Sequence[Any],
        row: Mapping[str, Any],
    ) -> None:
        key = tuple(key_values)
        metrics = self.groups.get(key)
        if metrics is None:
            metrics = {
                metric: RunningMetric()
                for metric in self.metric_names
            }
            self.groups[key] = metrics
        for metric in self.metric_names:
            metrics[metric].update(row.get(metric))

    def rows(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for key, metrics in self.groups.items():
            row = {
                name: value
                for name, value in zip(self.key_names, key)
            }
            for metric_name, metric in metrics.items():
                row.update(metric.summary(metric_name))
            result.append(row)
        return sorted(
            result,
            key=lambda row: tuple(
                str(row[name]) for name in self.key_names
            ),
        )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def pair_status(
    original_correct: bool,
    flipped_correct: bool,
) -> str:
    if original_correct and flipped_correct:
        return "both_correct"
    if original_correct:
        return "original_only_correct"
    if flipped_correct:
        return "flipped_only_correct"
    return "both_wrong"


def add_metadata(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    output = {
        "sid": metadata["sid"],
        "pair_status": metadata["pair_status"],
        "subject": metadata["subject"],
        "reference": metadata["reference"],
        "original_relation": metadata["original_relation"],
        "flipped_relation": metadata["flipped_relation"],
        "original_prediction": metadata["original_prediction"],
        "flipped_prediction": metadata["flipped_prediction"],
    }
    output.update(dict(row))
    return output


def peak_row(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Optional[Mapping[str, Any]]:
    eligible = [row for row in rows if finite(row.get(metric))]
    if not eligible:
        return None
    return max(eligible, key=lambda row: float(row[metric]))


def build_sample_diagnostics(
    *,
    metadata: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    final_last_delta_norm: float,
) -> Dict[str, Any]:
    state_by_group = {
        group: [
            row for row in state_rows if row["token_group"] == group
        ]
        for group in STATE_GROUPS
    }
    source_by_group = {
        group: [
            row for row in source_rows if row["source_group"] == group
        ]
        for group in ("subject", "reference", "visual_all", "options")
    }

    subject_peak = peak_row(
        state_by_group["subject"],
        "input_delta_norm",
    )
    reference_peak = peak_row(
        state_by_group["reference"],
        "input_delta_norm",
    )
    last_increment_peak = peak_row(
        state_by_group["prompt_last"],
        "increment_delta_norm",
    )
    last_output_peak = peak_row(
        state_by_group["prompt_last"],
        "output_delta_norm",
    )
    subject_transfer_peak = peak_row(
        source_by_group["subject"],
        "content_to_last_increment_projection_fraction",
    )
    reference_transfer_peak = peak_row(
        source_by_group["reference"],
        "content_to_last_increment_projection_fraction",
    )
    visual_transfer_peak = peak_row(
        source_by_group["visual_all"],
        "delta_to_last_increment_projection_fraction",
    )
    options_transfer_peak = peak_row(
        source_by_group["options"],
        "delta_to_last_increment_projection_fraction",
    )

    object_by_layer: Dict[int, float] = defaultdict(float)
    object_rows = source_by_group["subject"] + source_by_group["reference"]
    for row in object_rows:
        metric = row.get(
            "content_to_last_increment_projection_fraction"
        )
        if finite(metric):
            object_by_layer[int(row["layer"])] += float(metric)

    if object_by_layer:
        object_peak_layer, object_peak_value = max(
            object_by_layer.items(),
            key=lambda item: item[1],
        )
        object_mean_value = float(
            np.mean(list(object_by_layer.values()))
        )
        object_negative = (
            sum(value < 0 for value in object_by_layer.values())
            / len(object_by_layer)
        )
    else:
        object_peak_layer = None
        object_peak_value = float("nan")
        object_mean_value = float("nan")
        object_negative = float("nan")

    def row_value(
        row: Optional[Mapping[str, Any]],
        metric: str,
    ) -> float:
        if row is None or not finite(row.get(metric)):
            return float("nan")
        return float(row[metric])

    def row_layer(row: Optional[Mapping[str, Any]]) -> Optional[int]:
        return int(row["layer"]) if row is not None else None

    subject_norm = row_value(subject_peak, "input_delta_norm")
    reference_norm = row_value(reference_peak, "input_delta_norm")
    object_state_values = [
        value
        for value in (subject_norm, reference_norm)
        if finite(value)
    ]
    object_state_peak = (
        max(object_state_values)
        if object_state_values
        else float("nan")
    )

    return {
        "sid": metadata["sid"],
        "pair_status": metadata["pair_status"],
        "subject": metadata["subject"],
        "reference": metadata["reference"],
        "original_relation": metadata["original_relation"],
        "flipped_relation": metadata["flipped_relation"],
        "original_prediction": metadata["original_prediction"],
        "flipped_prediction": metadata["flipped_prediction"],
        "original_correct": metadata["original_correct"],
        "flipped_correct": metadata["flipped_correct"],
        "subject_peak_input_delta_norm": subject_norm,
        "subject_peak_input_layer": row_layer(subject_peak),
        "reference_peak_input_delta_norm": reference_norm,
        "reference_peak_input_layer": row_layer(reference_peak),
        "object_state_peak_norm": object_state_peak,
        "prompt_last_peak_increment_delta_norm": row_value(
            last_increment_peak,
            "increment_delta_norm",
        ),
        "prompt_last_peak_increment_layer": row_layer(
            last_increment_peak
        ),
        "prompt_last_peak_output_delta_norm": row_value(
            last_output_peak,
            "output_delta_norm",
        ),
        "prompt_last_peak_output_layer": row_layer(last_output_peak),
        "final_last_delta_norm": float(final_last_delta_norm),
        "subject_content_transfer_peak": row_value(
            subject_transfer_peak,
            "content_to_last_increment_projection_fraction",
        ),
        "subject_content_transfer_peak_layer": row_layer(
            subject_transfer_peak
        ),
        "reference_content_transfer_peak": row_value(
            reference_transfer_peak,
            "content_to_last_increment_projection_fraction",
        ),
        "reference_content_transfer_peak_layer": row_layer(
            reference_transfer_peak
        ),
        "object_content_transfer_peak": object_peak_value,
        "object_content_transfer_peak_layer": object_peak_layer,
        "object_content_transfer_mean": object_mean_value,
        "object_content_negative_rate": object_negative,
        "visual_transfer_peak": row_value(
            visual_transfer_peak,
            "delta_to_last_increment_projection_fraction",
        ),
        "visual_transfer_peak_layer": row_layer(visual_transfer_peak),
        "options_transfer_peak": row_value(
            options_transfer_peak,
            "delta_to_last_increment_projection_fraction",
        ),
        "options_transfer_peak_layer": row_layer(options_transfer_peak),
    }


def standardized_difference(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float:
    if n_a <= 1 or n_b <= 1:
        return float("nan")
    pooled_numerator = (
        (n_a - 1) * std_a * std_a
        + (n_b - 1) * std_b * std_b
    )
    pooled_denominator = n_a + n_b - 2
    if pooled_denominator <= 0:
        return float("nan")
    pooled_std = math.sqrt(
        max(0.0, pooled_numerator / pooled_denominator)
    )
    if pooled_std <= 1e-12:
        return float("nan")
    return (mean_a - mean_b) / pooled_std


def build_contrasts(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    key_names: Sequence[str],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    by_key = {
        tuple(row[name] for name in key_names): row
        for row in summary_rows
    }
    non_status_names = [
        name for name in key_names if name != "pair_status"
    ]
    identities = sorted(
        set(
            tuple(row[name] for name in non_status_names)
            for row in summary_rows
        ),
        key=lambda value: tuple(map(str, value)),
    )
    result: List[Dict[str, Any]] = []

    for identity in identities:
        identity_map = dict(zip(non_status_names, identity))
        clean_key = tuple(
            "both_correct"
            if name == "pair_status"
            else identity_map[name]
            for name in key_names
        )
        clean = by_key.get(clean_key)
        if clean is None:
            continue

        for status in (
            "original_only_correct",
            "flipped_only_correct",
            "both_wrong",
        ):
            error_key = tuple(
                status
                if name == "pair_status"
                else identity_map[name]
                for name in key_names
            )
            error = by_key.get(error_key)
            if error is None:
                continue

            for metric in metric_names:
                clean_mean = clean.get(f"mean_{metric}")
                error_mean = error.get(f"mean_{metric}")
                clean_std = clean.get(f"std_{metric}")
                error_std = error.get(f"std_{metric}")
                clean_n = clean.get(f"n_{metric}")
                error_n = error.get(f"n_{metric}")
                if not all(
                    finite(value)
                    for value in (
                        clean_mean,
                        error_mean,
                        clean_std,
                        error_std,
                        clean_n,
                        error_n,
                    )
                ):
                    continue

                output: Dict[str, Any] = {
                    "pair_status": status,
                    **identity_map,
                    "metric": metric,
                    "error_n": int(error_n),
                    "both_correct_n": int(clean_n),
                    "error_mean": float(error_mean),
                    "both_correct_mean": float(clean_mean),
                    "mean_difference_error_minus_correct": (
                        float(error_mean) - float(clean_mean)
                    ),
                    "standardized_difference": standardized_difference(
                        float(error_mean),
                        float(error_std),
                        int(error_n),
                        float(clean_mean),
                        float(clean_std),
                        int(clean_n),
                    ),
                }
                error_positive = error.get(f"positive_rate_{metric}")
                clean_positive = clean.get(f"positive_rate_{metric}")
                error_negative = error.get(f"negative_rate_{metric}")
                clean_negative = clean.get(f"negative_rate_{metric}")
                output["positive_rate_difference"] = (
                    float(error_positive) - float(clean_positive)
                    if finite(error_positive) and finite(clean_positive)
                    else None
                )
                output["negative_rate_difference"] = (
                    float(error_negative) - float(clean_negative)
                    if finite(error_negative) and finite(clean_negative)
                    else None
                )
                result.append(output)
    return result


def summarize_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    aggregator = GroupAggregator(
        key_names=("pair_status",),
        metric_names=DIAGNOSTIC_METRICS,
    )
    for row in rows:
        for status in ("all", str(row["pair_status"])):
            aggregator.update((status,), row)
    return aggregator.rows()


def quantile(
    rows: Sequence[Mapping[str, Any]],
    status: str,
    metric: str,
    probability: float,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["pair_status"] == status and finite(row.get(metric))
    ]
    if not values:
        return float("nan")
    return float(np.quantile(values, probability))


def build_failure_signatures(
    diagnostics: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    thresholds = {
        "object_state_q25": quantile(
            diagnostics,
            "both_correct",
            "object_state_peak_norm",
            0.25,
        ),
        "object_transfer_q25": quantile(
            diagnostics,
            "both_correct",
            "object_content_transfer_peak",
            0.25,
        ),
        "object_negative_q75": quantile(
            diagnostics,
            "both_correct",
            "object_content_negative_rate",
            0.75,
        ),
        "last_increment_q25": quantile(
            diagnostics,
            "both_correct",
            "prompt_last_peak_increment_delta_norm",
            0.25,
        ),
        "final_last_q25": quantile(
            diagnostics,
            "both_correct",
            "final_last_delta_norm",
            0.25,
        ),
    }

    result: List[Dict[str, Any]] = []
    for row in diagnostics:
        if row["pair_status"] == "both_correct":
            continue
        reasons: List[str] = []

        if (
            finite(row.get("object_state_peak_norm"))
            and finite(thresholds["object_state_q25"])
            and float(row["object_state_peak_norm"])
            < thresholds["object_state_q25"]
        ):
            reasons.append("weak_object_position_sensitive_state")

        if (
            finite(row.get("object_content_negative_rate"))
            and finite(thresholds["object_negative_q75"])
            and float(row["object_content_negative_rate"])
            > thresholds["object_negative_q75"]
        ):
            reasons.append("more_opposing_object_to_last_layers")

        if (
            finite(row.get("object_content_transfer_peak"))
            and finite(thresholds["object_transfer_q25"])
            and float(row["object_content_transfer_peak"])
            < thresholds["object_transfer_q25"]
        ):
            reasons.append("weak_object_to_last_content_write")

        if (
            finite(row.get("prompt_last_peak_increment_delta_norm"))
            and finite(thresholds["last_increment_q25"])
            and float(row["prompt_last_peak_increment_delta_norm"])
            < thresholds["last_increment_q25"]
        ):
            reasons.append("weak_prompt_last_increment")

        if (
            finite(row.get("final_last_delta_norm"))
            and finite(thresholds["final_last_q25"])
            and float(row["final_last_delta_norm"])
            < thresholds["final_last_q25"]
        ):
            reasons.append("weak_final_last_counterfactual_difference")

        if not reasons:
            reasons.append(
                "difference_present_but_not_explained_by_simple_missing_signal"
            )

        result.append(
            {
                **dict(row),
                **thresholds,
                "provisional_failure_signature": ";".join(reasons),
                "signature_is_heuristic": True,
            }
        )
    return result


def top_contrast_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    status: str,
    metric: str,
    ascending: bool,
    top_k: int,
    allowed_sources: Optional[Sequence[str]] = None,
) -> List[Mapping[str, Any]]:
    allowed = set(allowed_sources) if allowed_sources is not None else None
    selected = [
        row
        for row in rows
        if row["pair_status"] == status
        and row["metric"] == metric
        and finite(row.get("mean_difference_error_minus_correct"))
        and (
            allowed is None
            or row.get("source_group") in allowed
        )
    ]
    return sorted(
        selected,
        key=lambda row: float(
            row["mean_difference_error_minus_correct"]
        ),
        reverse=not ascending,
    )[:top_k]


def format_number(value: Any, width: int = 12) -> str:
    if not finite(value):
        return f"{'nan':>{width}}"
    return f"{float(value):>{width}.6f}"


def build_report(
    *,
    args: argparse.Namespace,
    counts: Mapping[str, int],
    diagnostic_summary: Sequence[Mapping[str, Any]],
    source_contrasts: Sequence[Mapping[str, Any]],
    head_contrasts: Sequence[Mapping[str, Any]],
    reconstruction_summary: Mapping[str, Any],
) -> str:
    diagnostic_by_status = {
        row["pair_status"]: row
        for row in diagnostic_summary
    }

    lines = [
        "=" * 160,
        "BATCH OBJECT-TO-LAST COUNTERFACTUAL TRANSFER BY CORRECTNESS CATEGORY",
        f"model={args.model}",
        "counts: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(counts.items())
        ),
        "=" * 160,
        "",
        "Sample-level diagnostic means",
        (
            f"{'Status':>25}{'N':>7}{'ObjState':>13}"
            f"{'ObjWrite':>13}{'ObjNeg%':>11}"
            f"{'LastIncr':>13}{'FinalLast':>13}"
            f"{'VisualPeak':>13}{'OptionPeak':>13}"
        ),
        "-" * 125,
    ]

    for status in AGGREGATE_STATUSES:
        row = diagnostic_by_status.get(status)
        if row is None:
            continue
        sample_count = int(row.get("n_object_state_peak_norm", 0))
        object_state = row.get("mean_object_state_peak_norm")
        object_write = row.get("mean_object_content_transfer_peak")
        object_negative = row.get("mean_object_content_negative_rate")
        object_negative_percent = (
            100.0 * float(object_negative)
            if finite(object_negative)
            else None
        )
        last_increment = row.get(
            "mean_prompt_last_peak_increment_delta_norm"
        )
        final_last = row.get("mean_final_last_delta_norm")
        visual_peak = row.get("mean_visual_transfer_peak")
        option_peak = row.get("mean_options_transfer_peak")
        line = (
            f"{status:>25}"
            f"{sample_count:>7}"
            f"{format_number(object_state, 13)}"
            f"{format_number(object_write, 13)}"
            f"{format_number(object_negative_percent, 11)}"
            f"{format_number(last_increment, 13)}"
            f"{format_number(final_last, 13)}"
            f"{format_number(visual_peak, 13)}"
            f"{format_number(option_peak, 13)}"
        )
        lines.append(line)

    for status in (
        "original_only_correct",
        "flipped_only_correct",
        "both_wrong",
    ):
        lines.extend(
            [
                "",
                f"[{status}] object-source content-write deficits vs both_correct",
                (
                    f"{'Rank':>5}{'Layer':>7}{'Source':>18}"
                    f"{'ErrMean':>13}{'Correct':>13}"
                    f"{'Diff':>13}{'StdDiff':>11}"
                    f"{'NegRateD':>12}"
                ),
                "-" * 100,
            ]
        )
        deficits = top_contrast_rows(
            source_contrasts,
            status=status,
            metric="content_to_last_increment_projection_fraction",
            ascending=True,
            top_k=args.top_k_report,
            allowed_sources=("subject", "reference"),
        )
        for rank, row in enumerate(deficits, start=1):
            line = (
                f"{rank:>5}"
                f"{int(row['layer']):>7}"
                f"{str(row['source_group']):>18}"
                f"{format_number(row['error_mean'], 13)}"
                f"{format_number(row['both_correct_mean'], 13)}"
                f"{format_number(row['mean_difference_error_minus_correct'], 13)}"
                f"{format_number(row['standardized_difference'], 11)}"
                f"{format_number(row.get('negative_rate_difference'), 12)}"
            )
            lines.append(line)

        lines.extend(
            [
                "",
                f"[{status}] head-level object content-write deficits vs both_correct",
                (
                    f"{'Rank':>5}{'Layer':>7}{'Head':>7}"
                    f"{'Source':>18}{'ErrMean':>13}"
                    f"{'Correct':>13}{'Diff':>13}"
                    f"{'StdDiff':>11}"
                ),
                "-" * 100,
            ]
        )
        head_deficits = top_contrast_rows(
            head_contrasts,
            status=status,
            metric="content_to_last_increment_projection_fraction",
            ascending=True,
            top_k=args.top_k_report,
            allowed_sources=("subject", "reference"),
        )
        for rank, row in enumerate(head_deficits, start=1):
            line = (
                f"{rank:>5}"
                f"{int(row['layer']):>7}"
                f"{int(row['head']):>7}"
                f"{str(row['source_group']):>18}"
                f"{format_number(row['error_mean'], 13)}"
                f"{format_number(row['both_correct_mean'], 13)}"
                f"{format_number(row['mean_difference_error_minus_correct'], 13)}"
                f"{format_number(row['standardized_difference'], 11)}"
            )
            lines.append(line)

    lines.extend(
        [
            "",
            "Reliability maxima",
            json.dumps(
                dict(reconstruction_summary),
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Interpretation constraints:",
            "- Lower object-state norms suggest weaker position-sensitive object states.",
            "- Normal object state plus lower object content-write suggests a transfer deficit.",
            "- A higher negative object-write rate suggests opposing/canceling paths.",
            "- Normal transfer and last increment in an error group means simple information absence is insufficient.",
            "- both_wrong is usually small, so its averages are unstable.",
            "- These are descriptive comparisons; validate candidate paths by exact edge restoration/replacement.",
        ]
    )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main batch pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    sources = parse_subset(args.sources, SOURCE_GROUPS, "source group")
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    error_path = output_dir / "errors.jsonl"
    baseline_writer = StreamingCSV(output_dir / "baseline_pairs.csv")
    state_writer = StreamingCSV(output_dir / "per_sample_state.csv")
    source_writer = StreamingCSV(output_dir / "per_sample_source.csv")
    head_writer = StreamingCSV(output_dir / "per_sample_head.csv")
    reconstruction_writer = StreamingCSV(
        output_dir / "reconstruction_checks.csv"
    )

    data_module = import_two_object_module()
    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    prompt_rows = load_standard_prompts(resolve_prompt_path(args))

    specs = merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model}; available={sorted(specs)}"
        )
    spec = specs[args.model]
    model_class = getattr(transformers, spec.model_class, None)
    if model_class is None:
        raise RuntimeError(f"transformers lacks {spec.model_class}")

    model_kwargs: Dict[str, Any] = {
        "dtype": resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        model_kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
    model = model_class.from_pretrained(spec.repo_id, **model_kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    requested_state_layers = parse_layers(args.state_layers, n_layers)
    edge_layers = parse_layers(args.edge_layers, n_layers)
    final_layer = n_layers - 1
    state_layers = sorted(
        set(requested_state_layers + edge_layers + [final_layer])
    )
    token_map = relation_token_variants(processor.tokenizer)

    for layer_index in edge_layers:
        attention = resolve_self_attention(decoder_layers[layer_index])
        for required_name in ("v_proj", "o_proj"):
            if not hasattr(attention, required_name):
                raise RuntimeError(
                    f"Layer {layer_index} attention "
                    f"{type(attention).__name__} lacks {required_name}"
                )

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "relations": list(HORIZONTAL_RELATIONS),
        "state_layers": state_layers,
        "edge_layers": edge_layers,
        "final_layer": final_layer,
        "sources": sources,
        "decoder_path": decoder_path,
        "n_decoder_layers": n_layers,
        "max_samples": args.max_samples,
        "save_head_details": args.save_head_details,
        "audit": audit,
        "pair_statuses": list(PAIR_STATUSES),
        "uses_centroid": False,
        "uses_trained_probe": False,
        "updates_model_weights": False,
        "analysis_dependencies": [
            "extract_two_object_relation_states.py"
        ],
    }
    (output_dir / "config.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    state_aggregator = GroupAggregator(
        key_names=("pair_status", "layer", "token_group"),
        metric_names=STATE_METRICS,
    )
    source_aggregator = GroupAggregator(
        key_names=("pair_status", "layer", "source_group"),
        metric_names=SOURCE_METRICS,
    )
    head_aggregator = GroupAggregator(
        key_names=("pair_status", "layer", "head", "source_group"),
        metric_names=HEAD_METRICS,
    )

    diagnostics: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    reconstruction_max = {
        "attention_reconstruction_relative_error": 0.0,
        "routing_content_relative_error": 0.0,
        "block_decomposition_relative_error": 0.0,
        "replay_max_abs_error": 0.0,
        "replay_relative_error": 0.0,
    }
    analyzed = 0
    start_time = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"last-increment:{args.model}",
        ):
            sid = int(record.sid)
            counts["seen"] += 1
            original_image = None
            flipped_image = None
            original_batch = None
            flipped_batch = None
            original_trace = None
            flipped_trace = None

            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                original_relation = normalize_relation(
                    prompt_row["answer_raw"]
                )
                if original_relation not in HORIZONTAL_RELATIONS:
                    continue
                flipped_relation = OPPOSITE[original_relation]
                counts["eligible_relation_seen"] += 1

                original_image = record_image(record).convert("RGB")
                flipped_image = original_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
                rendered = build_prompt(processor, question)

                original_batch = move_batch(
                    processor(
                        text=[rendered],
                        images=[original_image],
                        return_tensors="pt",
                    ),
                    device,
                )
                flipped_batch = move_batch(
                    processor(
                        text=[rendered],
                        images=[flipped_image],
                        return_tensors="pt",
                    ),
                    device,
                )

                original_ids = (
                    original_batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                flipped_ids = (
                    flipped_batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                if original_ids != flipped_ids:
                    raise RuntimeError(
                        "Original and flipped tokenizations differ"
                    )

                subject_span, reference_span = locate_object_spans(
                    processor.tokenizer,
                    original_ids,
                    subject,
                    reference,
                )
                visual_indices = resolve_visual_indices(
                    model,
                    processor,
                    original_batch,
                    original_ids,
                )
                visual_set = set(map(int, visual_indices))
                text_positions = [
                    position
                    for position in range(len(original_ids))
                    if position not in visual_set
                ]
                semantic = locate_semantic_spans(
                    processor.tokenizer,
                    original_ids,
                    question,
                    subject_span,
                    reference_span,
                    text_positions,
                )
                token_manifest = build_token_manifest(
                    processor.tokenizer,
                    original_ids,
                    text_positions,
                    semantic,
                )
                group_positions = state_group_positions(semantic)
                state_positions = sorted(
                    set(
                        position
                        for positions in group_positions.values()
                        for position in positions
                    )
                )
                last_positions = group_positions["prompt_last"]
                source_position_map = build_source_groups(
                    sequence_length=len(original_ids),
                    visual_indices=visual_indices,
                    token_manifest=token_manifest,
                    target_positions=last_positions,
                )

                original_trace = run_trace(
                    model=model,
                    batch=original_batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    state_layers=state_layers,
                    edge_layers=edge_layers,
                    state_positions=state_positions,
                    last_positions=last_positions,
                )
                flipped_trace = run_trace(
                    model=model,
                    batch=flipped_batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    state_layers=state_layers,
                    edge_layers=edge_layers,
                    state_positions=state_positions,
                    last_positions=last_positions,
                )

                original_prediction = original_trace.prediction
                flipped_prediction = flipped_trace.prediction
                original_correct = (
                    original_prediction == original_relation
                )
                flipped_correct = (
                    flipped_prediction == flipped_relation
                )
                status = pair_status(original_correct, flipped_correct)
                counts[status] += 1
                counts["original_correct"] += int(original_correct)
                counts["flipped_correct"] += int(flipped_correct)
                counts["predictions_opposite"] += int(
                    OPPOSITE.get(original_prediction)
                    == flipped_prediction
                )

                metadata = {
                    "sid": sid,
                    "pair_status": status,
                    "subject": subject,
                    "reference": reference,
                    "original_relation": original_relation,
                    "flipped_relation": flipped_relation,
                    "original_prediction": original_prediction,
                    "flipped_prediction": flipped_prediction,
                    "original_correct": bool(original_correct),
                    "flipped_correct": bool(flipped_correct),
                }
                baseline_writer.write(
                    {
                        **metadata,
                        "original_scores": json.dumps(
                            original_trace.scores,
                            ensure_ascii=False,
                        ),
                        "flipped_scores": json.dumps(
                            flipped_trace.scores,
                            ensure_ascii=False,
                        ),
                    }
                )

                final_last_original = mean_rows(
                    original_trace.state.block_outputs[final_layer],
                    original_trace.state.positions,
                    last_positions,
                )
                final_last_flipped = mean_rows(
                    flipped_trace.state.block_outputs[final_layer],
                    flipped_trace.state.positions,
                    last_positions,
                )
                final_last_delta = (
                    final_last_original - final_last_flipped
                )

                state_rows, state_vectors = make_state_rows(
                    original=original_trace,
                    flipped=flipped_trace,
                    layers=state_layers,
                    group_positions=group_positions,
                    final_last_delta=final_last_delta,
                )
                source_rows, head_rows, reconstruction_rows = make_edge_rows(
                    original=original_trace,
                    flipped=flipped_trace,
                    edge_layers=edge_layers,
                    source_groups=sources,
                    source_positions=source_position_map,
                    last_positions=last_positions,
                    last_state_vectors=state_vectors,
                    final_last_delta=final_last_delta,
                )

                for row in state_rows:
                    state_writer.write(add_metadata(row, metadata))
                    for aggregate_status in ("all", status):
                        state_aggregator.update(
                            (
                                aggregate_status,
                                int(row["layer"]),
                                str(row["token_group"]),
                            ),
                            row,
                        )

                for row in source_rows:
                    source_writer.write(add_metadata(row, metadata))
                    for aggregate_status in ("all", status):
                        source_aggregator.update(
                            (
                                aggregate_status,
                                int(row["layer"]),
                                str(row["source_group"]),
                            ),
                            row,
                        )

                for row in head_rows:
                    if args.save_head_details:
                        head_writer.write(add_metadata(row, metadata))
                    for aggregate_status in ("all", status):
                        head_aggregator.update(
                            (
                                aggregate_status,
                                int(row["layer"]),
                                int(row["head"]),
                                str(row["source_group"]),
                            ),
                            row,
                        )

                for row in reconstruction_rows:
                    reconstruction_writer.write(
                        add_metadata(row, metadata)
                    )
                    for metric in reconstruction_max:
                        if finite(row.get(metric)):
                            reconstruction_max[metric] = max(
                                reconstruction_max[metric],
                                float(row[metric]),
                            )

                diagnostics.append(
                    build_sample_diagnostics(
                        metadata=metadata,
                        state_rows=state_rows,
                        source_rows=source_rows,
                        final_last_delta_norm=vector_norm(
                            final_last_delta
                        ),
                    )
                )

                analyzed += 1
                if (
                    args.print_every > 0
                    and analyzed % args.print_every == 0
                ):
                    print(
                        f"[{analyzed}] sid={sid} status={status} "
                        f"orig={original_prediction} "
                        f"flip={flipped_prediction}",
                        flush=True,
                    )

            except Exception as error:
                counts["errors"] += 1
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"[ERROR] sid={sid}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

            finally:
                for image in (original_image, flipped_image):
                    if image is not None:
                        try:
                            image.close()
                        except Exception:
                            pass
                del (
                    original_batch,
                    flipped_batch,
                    original_trace,
                    flipped_trace,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        baseline_writer.close()
        state_writer.close()
        source_writer.close()
        head_writer.close()
        reconstruction_writer.close()

    if not diagnostics:
        raise RuntimeError(
            "No eligible samples were analyzed. Inspect errors.jsonl"
        )

    state_summary = state_aggregator.rows()
    source_summary = source_aggregator.rows()
    head_summary = head_aggregator.rows()
    diagnostic_summary = summarize_diagnostics(diagnostics)

    state_contrasts = build_contrasts(
        state_summary,
        key_names=("pair_status", "layer", "token_group"),
        metric_names=STATE_METRICS,
    )
    source_contrasts = build_contrasts(
        source_summary,
        key_names=("pair_status", "layer", "source_group"),
        metric_names=SOURCE_METRICS,
    )
    head_contrasts = build_contrasts(
        head_summary,
        key_names=("pair_status", "layer", "head", "source_group"),
        metric_names=HEAD_METRICS,
    )
    failure_signatures = build_failure_signatures(diagnostics)

    write_csv(output_dir / "sample_diagnostics.csv", diagnostics)
    write_csv(
        output_dir / "diagnostic_summary_by_pair_status.csv",
        diagnostic_summary,
    )
    write_csv(
        output_dir / "state_summary_by_pair_status.csv",
        state_summary,
    )
    write_csv(
        output_dir / "source_summary_by_pair_status.csv",
        source_summary,
    )
    write_csv(
        output_dir / "head_summary_by_pair_status.csv",
        head_summary,
    )
    write_csv(
        output_dir / "state_contrasts_vs_both_correct.csv",
        state_contrasts,
    )
    write_csv(
        output_dir / "source_contrasts_vs_both_correct.csv",
        source_contrasts,
    )
    write_csv(
        output_dir / "head_contrasts_vs_both_correct.csv",
        head_contrasts,
    )
    write_csv(
        output_dir / "provisional_failure_signatures.csv",
        failure_signatures,
    )

    report = build_report(
        args=args,
        counts=counts,
        diagnostic_summary=diagnostic_summary,
        source_contrasts=source_contrasts,
        head_contrasts=head_contrasts,
        reconstruction_summary=reconstruction_max,
    )
    (output_dir / "report.txt").write_text(
        report,
        encoding="utf-8",
    )
    print("\n" + report)

    summary = {
        "version": VERSION,
        "model": args.model,
        "counts": dict(counts),
        "analyzed": analyzed,
        "elapsed_minutes": (time.time() - start_time) / 60.0,
        "reconstruction_max": reconstruction_max,
        "output_files": [
            "config.json",
            "baseline_pairs.csv",
            "per_sample_state.csv",
            "per_sample_source.csv",
            "per_sample_head.csv (only with --save-head-details)",
            "reconstruction_checks.csv",
            "sample_diagnostics.csv",
            "diagnostic_summary_by_pair_status.csv",
            "state_summary_by_pair_status.csv",
            "source_summary_by_pair_status.csv",
            "head_summary_by_pair_status.csv",
            "state_contrasts_vs_both_correct.csv",
            "source_contrasts_vs_both_correct.csv",
            "head_contrasts_vs_both_correct.csv",
            "provisional_failure_signatures.csv",
            "report.txt",
            "summary.json",
            "errors.jsonl",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
