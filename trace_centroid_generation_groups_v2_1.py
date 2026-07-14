#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace why correct top-k attention centroids do not become correct generation.

The primary comparison is:

A) top-k centroid correct + baseline generation correct
B) top-k centroid correct + baseline generation wrong

Version 2 fixes the projected A*V measurements and adds stronger controls:

- robust v_proj/o_proj discovery, including nested projection modules;
- keyword-aware forward pre-hooks;
- exact isolated-head o_proj calls instead of fragile weight reshaping;
- raw and projected A*V coverage diagnostics;
- attention, MLP, and whole-block GT-margin gains;
- layer-0-normalized logit-lens trajectories;
- same-relation nearest-neighbour matching on centroid quality;
- paired bootstrap comparisons after matching;
- matched per-head routing comparisons;
- optional same-image original/swap paired tracing.

The script performs frozen-model diagnostic forwards only. It does not train,
fine-tune, alter model weights, replace model answers, or apply an intervention.

Required prior files:
- <prior-dir>/config.json
- <prior-dir>/centroid_analysis.jsonl
- <prior-dir>/generation.jsonl

Optional swap-pair analysis additionally uses:
- <step1-dir>/samples.jsonl

The prior directory should be produced by:
    eval_topk_attention_centroid_generation_v1.py

The optional Step 1 directory should be produced by:
    analyze_coco_attention_flow_swap_step1_v1.py
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
import re
import shutil
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "trace-centroid-generation-groups-v2.2"

DEFAULT_PROMPT_FILES = {
    "coco_two": Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl"),
    "vg_two": Path("prompts/VG_QA_two_obj_with_answer_four_options.jsonl"),
}

RELATIONS = ("left", "right", "above", "below")

AUTO_LAYERS = {
    "llava-7b": 12,
    "llava-13b": 16,
    "qwen-3b": 24,
    "qwen-7b": 19,
}




def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def import_two_object_module():
    return importlib.import_module("extract_two_object_relation_states")


def resolve_prompt_path(args: argparse.Namespace) -> Path:
    if args.prompt_jsonl:
        path = Path(args.prompt_jsonl)
    else:
        path = DEFAULT_PROMPT_FILES[args.dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing standard prompt file: {path}. "
            "Pass it explicitly with --prompt-jsonl."
        )
    return path


def extract_standard_user_text(raw_question: str) -> str:
    """Extract the exact USER text from a stored <image>/USER/ASSISTANT prompt."""
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


STANDARD_OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?\s*Answer\s+with",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_standard_objects(question_text: str) -> Tuple[str, str]:
    """Parse exact subject/reference surface strings from the standard question."""
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
    """Render the dataset standard question with the current model chat template."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question_text},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question_text


def one_line(value: Any) -> str:
    return " ".join(str(value).split())


def should_print_sample(done_count: int, print_every: int) -> bool:
    return print_every > 0 and done_count % print_every == 0


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
    if strategy is not None and hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = str(strategy)

    if getattr(config, "model_type", "") == "llava" and hasattr(
        processor, "num_additional_image_tokens"
    ):
        processor.num_additional_image_tokens = 1


def move_batch(batch: Any, device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out.input_ids
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    width = len(needle)
    return [
        start
        for start in range(len(haystack) - width + 1)
        if list(haystack[start:start + width]) == list(needle)
    ]


def phrase_surface_variants(phrase: str) -> List[str]:
    """Return conservative case variants for token-span lookup.

    Some chat templates/tokenizers normalize acronym-like object names, for
    example the standard question may contain "TV" while the rendered prompt
    tokenizes it as "tv".  These variants affect only span lookup; the standard
    dataset question itself is kept unchanged.
    """
    raw = str(phrase).strip()
    candidates = [
        raw,
        raw.lower(),
        raw.upper(),
        raw.title(),
        raw.capitalize(),
    ]
    return list(dict.fromkeys(x for x in candidates if x))


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
        (s, r)
        for s in subject_spans
        for r in reference_spans
        if s[1] < r[0]
    ]
    if not valid:
        raise ValueError(
            "Could not locate ordered object spans: "
            f"subject={subject!r}, reference={reference!r}, "
            f"subject_spans={subject_spans}, reference_spans={reference_spans}"
        )
    return max(valid, key=lambda pair: (pair[1][0], pair[0][0]))


def record_image(record: Any) -> Image.Image:
    if hasattr(record, "image"):
        return record.image.copy().convert("RGB")
    if hasattr(record, "image_path"):
        return Image.open(record.image_path).convert("RGB")
    raise TypeError("Record has neither image nor image_path.")


def hidden_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for states in candidates:
        if isinstance(states, (tuple, list)) and states and torch.is_tensor(states[-1]):
            return tuple(states)
    raise RuntimeError("No decoder hidden states returned by model backend.")


def get_attr_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_decoder_layers(model: Any) -> Tuple[Any, str]:
    preferred = [
        "model.language_model.layers",             # Qwen2.5-VL recent HF
        "model.model.language_model.layers",
        "language_model.model.layers",             # LLaVA
        "language_model.layers",
        "model.language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in preferred:
        value = get_attr_path(model, path)
        if isinstance(value, (torch.nn.ModuleList, list, tuple)) and len(value) >= 4:
            return value, path

    candidates: List[Tuple[str, Any]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) >= 4:
            candidates.append((f"{name}.layers" if name else "layers", layers))

    # Prefer a language/text path and avoid visual transformer blocks.
    candidates.sort(
        key=lambda item: (
            0 if any(k in item[0].lower() for k in ("language", "text")) else 1,
            1 if any(k in item[0].lower() for k in ("visual", "vision")) else 0,
            -len(item[1]),
        )
    )
    if candidates:
        return candidates[0][1], candidates[0][0]
    raise RuntimeError("Could not resolve language-model decoder layers.")


def candidate_token_id(tokenizer: Any, token: str) -> Optional[int]:
    try:
        idx = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if idx is None:
        return None
    try:
        idx = int(idx)
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and idx == int(unk):
        return None
    return idx


def resolve_visual_indices(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    input_ids: Sequence[int],
) -> List[int]:
    # Recent Qwen2.5-VL processors provide modality IDs directly:
    # text=0, image=1, video=2. Prefer this over special-token heuristics.
    mm_type_ids = batch.get("mm_token_type_ids")
    if torch.is_tensor(mm_type_ids) and mm_type_ids.ndim == 2:
        direct = torch.nonzero(mm_type_ids[0] == 1, as_tuple=False).flatten().tolist()
        if direct:
            return [int(x) for x in direct]

    token_type_ids = batch.get("token_type_ids")
    if torch.is_tensor(token_type_ids) and token_type_ids.ndim == 2:
        unique = set(int(x) for x in token_type_ids[0].detach().cpu().tolist())
        if 1 in unique:
            direct = torch.nonzero(token_type_ids[0] == 1, as_tuple=False).flatten().tolist()
            if direct:
                return [int(x) for x in direct]

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
        idx = candidate_token_id(tokenizer, token)
        if idx is not None:
            token_ids.add(idx)

    indices = [i for i, token_id in enumerate(input_ids) if int(token_id) in token_ids]
    if indices:
        return indices

    # Fallback: tokens strictly between a vision-start and vision-end token.
    start_ids = {
        idx
        for token in ("<|vision_start|>", "<image_start>", "<img>")
        if (idx := candidate_token_id(tokenizer, token)) is not None
    }
    end_ids = {
        idx
        for token in ("<|vision_end|>", "<image_end>", "</img>")
        if (idx := candidate_token_id(tokenizer, token)) is not None
    }
    starts = [i for i, x in enumerate(input_ids) if int(x) in start_ids]
    ends = [i for i, x in enumerate(input_ids) if int(x) in end_ids]
    spans = [(s, e) for s in starts for e in ends if s < e]
    if spans:
        s, e = min(spans, key=lambda pair: pair[1] - pair[0])
        fallback = list(range(s + 1, e))
        if fallback:
            return fallback

    raise ValueError(
        "Could not identify visual-token positions. "
        f"Candidate image token IDs were {sorted(token_ids)}."
    )


def visual_coordinates(
    model: Any,
    batch: Dict[str, Any],
    n_visual_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return normalized (x, y) coordinates in visual-token flatten order."""
    grid = batch.get("image_grid_thw")
    if torch.is_tensor(grid) and grid.numel() >= 3:
        values = grid.detach().cpu().reshape(-1, 3)[0].tolist()
        t, h, w = [int(x) for x in values]
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        merge = int(getattr(vision_config, "spatial_merge_size", 1) or 1)
        temporal_merge = int(getattr(vision_config, "temporal_patch_size", 1) or 1)

        candidates = []
        for tm in (1, temporal_merge):
            tt = max(1, t // max(1, tm))
            hh = max(1, h // max(1, merge))
            ww = max(1, w // max(1, merge))
            candidates.append((tt, hh, ww))
        # Some processor versions already report the post-merge grid.
        candidates.append((max(1, t), max(1, h), max(1, w)))

        for tt, hh, ww in candidates:
            if tt * hh * ww != n_visual_tokens:
                continue
            ys = torch.linspace(0.0, 1.0, hh, device=device)
            xs = torch.linspace(0.0, 1.0, ww, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            if tt > 1:
                xy = xy.repeat(tt, 1)
            return xy

    side = int(round(math.sqrt(n_visual_tokens)))
    if side * side == n_visual_tokens:
        ys = torch.linspace(0.0, 1.0, side, device=device)
        xs = torch.linspace(0.0, 1.0, side, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return None


def relation_from_centroids(delta_x: float, delta_y: float) -> Tuple[str, float]:
    ax = abs(delta_x)
    ay = abs(delta_y)
    axis_conf = abs(ax - ay) / (ax + ay + 1e-8)
    if ax >= ay:
        return ("left" if delta_x < 0 else "right"), float(axis_conf)
    return ("above" if delta_y < 0 else "below"), float(axis_conf)


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
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
    hits.sort(key=lambda x: x[0])
    return hits[0][1]


def decode_new_tokens(
    processor: Any,
    output_ids: torch.Tensor,
    input_length: int,
) -> str:
    generated_ids = output_ids[0, input_length:]
    tokenizer = processor.tokenizer
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_text(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    max_new_tokens: int,
) -> str:
    input_length = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = decode_new_tokens(processor, output_ids, input_length)
    del output_ids
    return text

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_batch(
    processor: Any,
    record: Any,
    prompt_row: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], str, str, str, str, Any, Image.Image]:
    # Use exact object surface forms parsed from the dataset standard question.
    # This is required for reliable token-span lookup (for example, "TV" vs "tv").
    subject = str(prompt_row["subject"])
    reference = str(prompt_row["reference"])
    question_text = str(prompt_row["question_text"])
    raw_question = str(prompt_row["raw_question"])
    answer_raw = prompt_row["answer_raw"]

    rendered = build_prompt(processor, question_text)
    image = record_image(record)
    batch = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
    )
    batch = move_batch(batch, device)
    return (
        batch,
        subject,
        reference,
        question_text,
        raw_question,
        answer_raw,
        image,
    )



def label_token_id_variants(tokenizer: Any) -> Dict[str, List[int]]:
    """Resolve one-token surface variants for the four answer labels."""
    mapping: Dict[str, List[int]] = {}
    unk = getattr(tokenizer, "unk_token_id", None)

    for label in RELATIONS:
        surfaces = [
            label,
            " " + label,
            label.capitalize(),
            " " + label.capitalize(),
            "\n" + label,
            "\n" + label.capitalize(),
        ]
        ids: List[int] = []
        for surface in surfaces:
            token_ids = tokenizer_ids(tokenizer, surface)
            if len(token_ids) != 1:
                continue
            token_id = int(token_ids[0])
            if unk is not None and token_id == int(unk):
                continue
            ids.append(token_id)
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise RuntimeError(
                f"No one-token generation variant found for relation {label!r}. "
                "This first implementation guides only the first answer token."
            )
        mapping[label] = ids
    return mapping





GROUP_CORRECT = "centroid_correct_generation_correct"
GROUP_WRONG = "centroid_correct_generation_wrong"
GROUPS = (GROUP_CORRECT, GROUP_WRONG)

RELATION_TO_INDEX = {
    relation: index for index, relation in enumerate(RELATIONS)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager"],
        help="This trace requires eager attention weights.",
    )
    p.add_argument(
        "--prior-dir",
        required=True,
        help=(
            "Output directory from eval_topk_attention_centroid_generation_v1.py."
        ),
    )
    p.add_argument(
        "--centroid-analysis-jsonl",
        default=None,
        help="Override <prior-dir>/centroid_analysis.jsonl.",
    )
    p.add_argument(
        "--generation-jsonl",
        default=None,
        help="Override <prior-dir>/generation.jsonl.",
    )
    p.add_argument(
        "--step1-dir",
        default=None,
        help=(
            "Optional Step 1 directory containing samples.jsonl. Required only "
            "when --run-swap-pairs is enabled."
        ),
    )
    p.add_argument(
        "--run-swap-pairs",
        action="store_true",
        help=(
            "Trace same-image original/swap prompt pairs where one generation "
            "is correct and the aligned counterpart is wrong."
        ),
    )
    p.add_argument(
        "--max-swap-pairs",
        type=int,
        default=None,
        help="Optional random cap for the original/swap paired experiment.",
    )
    p.add_argument(
        "--relations",
        default="left,right,above,below",
        help="Comma-separated GT relations to include.",
    )
    p.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional random cap per main comparison group for a smoke run.",
    )
    p.add_argument(
        "--matching-mode",
        default="reuse",
        choices=["reuse", "unique"],
        help=(
            "Same-relation nearest-neighbour matching. reuse allows the same "
            "generation-correct control to match multiple wrong samples; unique "
            "uses each control at most once."
        ),
    )
    p.add_argument(
        "--matching-features",
        default=(
            "centroid_confidence,axis_confidence,head_agreement,"
            "swap_stability,mean_separation,mean_visual_mass,prompt_length"
        ),
        help="Comma-separated scalar features used for within-relation matching.",
    )
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap repetitions for unpaired and paired intervals.",
    )
    p.add_argument(
        "--persistent-layers",
        type=int,
        default=2,
        help="Consecutive layers required for a persistent divergence.",
    )
    p.add_argument(
        "--minimum-effect",
        type=float,
        default=0.30,
        help="Minimum absolute Cohen effect size for divergence detection.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def parse_relations(value: str) -> List[str]:
    relations = []
    for item in str(value).split(","):
        relation = normalize_relation(item)
        if relation is None or relation not in RELATIONS:
            raise ValueError(f"Unsupported relation in --relations: {item!r}")
        if relation not in relations:
            relations.append(relation)
    if not relations:
        raise ValueError("--relations resolved to an empty list")
    return relations


def first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    for name in ("last_hidden_state", "hidden_states", "attn_output"):
        item = getattr(value, name, None)
        if torch.is_tensor(item):
            return item
    raise TypeError(f"Could not extract tensor from {type(value)}")


def attention_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(getattr(outputs, "language_model_outputs", None), "attentions", None),
        getattr(getattr(outputs, "text_model_output", None), "attentions", None),
    ]
    for attentions in candidates:
        if isinstance(attentions, (tuple, list)) and attentions:
            tensors = tuple(x for x in attentions if torch.is_tensor(x))
            if tensors:
                return tensors
    raise RuntimeError(
        "No decoder attentions returned. Load with --attn-impl eager and "
        "request output_attentions=True."
    )


def resolve_final_norm(model: Any) -> Tuple[Optional[torch.nn.Module], str]:
    preferred = [
        "model.language_model.norm",
        "model.model.language_model.norm",
        "language_model.model.norm",
        "language_model.norm",
        "model.language_model.model.norm",
        "model.model.norm",
        "model.norm",
    ]
    for path in preferred:
        module = get_attr_path(model, path)
        if isinstance(module, torch.nn.Module):
            return module, path

    candidates: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if (
            lowered.endswith(".norm")
            and "vision" not in lowered
            and "visual" not in lowered
            and any(key in lowered for key in ("language", "model", "text"))
        ):
            candidates.append((name, module))
    if candidates:
        candidates.sort(key=lambda pair: len(pair[0]))
        return candidates[-1][1], candidates[-1][0]
    return None, "<identity>"


def resolve_self_attention(layer: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module, name
    for name, module in layer.named_children():
        if "attn" in name.lower() or "attention" in name.lower():
            return module, name
    raise RuntimeError(f"Could not resolve self-attention in {type(layer)}")


def resolve_projection(
    attention: torch.nn.Module,
    names: Sequence[str],
) -> Optional[torch.nn.Module]:
    module, _ = resolve_projection_with_path(attention, names)
    return module


def resolve_projection_with_path(
    attention: torch.nn.Module,
    names: Sequence[str],
) -> Tuple[Optional[torch.nn.Module], str]:
    wanted = {str(name).lower() for name in names}

    for name in names:
        module = getattr(attention, name, None)
        if isinstance(module, torch.nn.Module):
            return module, str(name)

    # Some implementations wrap projections one level deeper.
    for child_name, module in attention.named_modules():
        if not child_name:
            continue
        leaf = child_name.split(".")[-1].lower()
        if leaf in wanted and isinstance(module, torch.nn.Module):
            return module, child_name

    # Conservative semantic fallback.
    semantic_tokens = {
        "v_proj": ("v_proj", "value_proj", "value"),
        "value_proj": ("v_proj", "value_proj", "value"),
        "o_proj": ("o_proj", "out_proj", "output_proj", "dense"),
        "out_proj": ("o_proj", "out_proj", "output_proj", "dense"),
        "output_proj": ("o_proj", "out_proj", "output_proj", "dense"),
        "dense": ("o_proj", "out_proj", "output_proj", "dense"),
    }
    candidates: List[Tuple[int, str, torch.nn.Module]] = []
    expanded: set[str] = set()
    for name in wanted:
        expanded.update(semantic_tokens.get(name, (name,)))
    for child_name, module in attention.named_modules():
        if not child_name or not isinstance(module, torch.nn.Module):
            continue
        lowered = child_name.lower()
        if any(token in lowered for token in expanded):
            candidates.append((len(child_name.split(".")), child_name, module))
    if candidates:
        candidates.sort(key=lambda row: (row[0], len(row[1])))
        _, path, module = candidates[0]
        return module, path
    return None, "<missing>"

def relation_token_rows(
    model: Any,
    label_token_ids: Dict[str, List[int]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[List[int]]]:
    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("Model output embedding has no accessible weight")
    all_ids: List[int] = []
    relation_positions: List[List[int]] = []
    for relation in RELATIONS:
        positions: List[int] = []
        for token_id in label_token_ids[relation]:
            if token_id not in all_ids:
                all_ids.append(token_id)
            positions.append(all_ids.index(token_id))
        relation_positions.append(positions)
    ids_tensor = torch.tensor(
        all_ids,
        dtype=torch.long,
        device=lm_head.weight.device,
    )
    weight = lm_head.weight.index_select(0, ids_tensor)
    bias = None
    lm_bias = getattr(lm_head, "bias", None)
    if torch.is_tensor(lm_bias):
        bias = lm_bias.index_select(0, ids_tensor)
    return weight, bias, relation_positions


def relation_scores_from_states(
    states: torch.Tensor,
    *,
    final_norm: Optional[torch.nn.Module],
    token_weight: torch.Tensor,
    token_bias: Optional[torch.Tensor],
    relation_positions: Sequence[Sequence[int]],
) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError(f"Expected [N,D] states, got {tuple(states.shape)}")
    device = token_weight.device
    dtype = token_weight.dtype
    work = states.to(device=device, dtype=dtype)
    with torch.inference_mode():
        if final_norm is not None:
            work = final_norm(work)
        token_scores = F.linear(
            work.float(),
            token_weight.float(),
            token_bias.float() if token_bias is not None else None,
        )
        relation_scores = torch.stack([
            token_scores[:, list(positions)].max(dim=-1).values
            for positions in relation_positions
        ], dim=-1)
    return relation_scores.detach().cpu().numpy().astype(np.float32)


def relation_diagnostics(
    scores: np.ndarray,
    gt_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.ndim != 2 or scores.shape[1] != len(RELATIONS):
        raise ValueError(f"Invalid relation score shape: {scores.shape}")
    gt = scores[:, gt_index]
    other = scores.copy()
    other[:, gt_index] = -np.inf
    margin = gt - np.max(other, axis=1)
    rank = 1 + np.sum(scores > gt[:, None], axis=1)
    prediction = np.argmax(scores, axis=1)
    return (
        margin.astype(np.float32),
        rank.astype(np.int16),
        prediction.astype(np.int16),
    )


def safe_cosine_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    cosine = F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)
    return cosine.detach().cpu().numpy().astype(np.float32)


def span_indices(span: Tuple[int, int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def reshape_projected_values(
    flat_values: torch.Tensor,
    *,
    n_attention_heads: int,
    attention_module: torch.nn.Module,
) -> torch.Tensor:
    """Return [token, query_head, head_dim], expanding GQA KV heads."""
    if flat_values.ndim == 3 and flat_values.shape[0] == 1:
        flat_values = flat_values[0]
    if flat_values.ndim != 2:
        raise ValueError(
            f"Expected projected values [T,Dv], got {tuple(flat_values.shape)}"
        )
    value_dim = int(flat_values.shape[-1])
    configured_head_dim = getattr(attention_module, "head_dim", None)
    configured_kv = getattr(attention_module, "num_key_value_heads", None)

    head_dim = int(configured_head_dim) if configured_head_dim is not None else None
    n_kv = int(configured_kv) if configured_kv is not None else None

    if head_dim is None and n_kv and value_dim % n_kv == 0:
        head_dim = value_dim // n_kv
    if n_kv is None and head_dim and value_dim % head_dim == 0:
        n_kv = value_dim // head_dim
    if head_dim is None:
        if value_dim % n_attention_heads != 0:
            raise RuntimeError(
                f"Cannot infer value heads: Dv={value_dim}, H={n_attention_heads}"
            )
        head_dim = value_dim // n_attention_heads
        n_kv = n_attention_heads
    if n_kv is None:
        n_kv = value_dim // head_dim
    if n_kv * head_dim != value_dim:
        raise RuntimeError(
            f"Invalid value projection shape: n_kv={n_kv}, "
            f"head_dim={head_dim}, value_dim={value_dim}"
        )

    values = flat_values.reshape(flat_values.shape[0], n_kv, head_dim)
    if n_kv == n_attention_heads:
        return values
    if n_attention_heads % n_kv != 0:
        raise RuntimeError(
            f"Query heads {n_attention_heads} not divisible by KV heads {n_kv}"
        )
    return values.repeat_interleave(n_attention_heads // n_kv, dim=1)


def module_device_dtype(
    module: torch.nn.Module,
    fallback: torch.Tensor,
) -> Tuple[torch.device, torch.dtype]:
    for parameter in module.parameters(recurse=True):
        return parameter.device, parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.device, buffer.dtype
    return fallback.device, fallback.dtype


def call_projection_module(
    module: torch.nn.Module,
    packed: torch.Tensor,
) -> torch.Tensor:
    """Call a linear-like projection and remove any bias exactly."""
    device, dtype = module_device_dtype(module, packed)
    work = packed.to(device=device, dtype=dtype)
    zeros = torch.zeros_like(work)

    def call(value: torch.Tensor) -> torch.Tensor:
        try:
            output = module(value)
        except Exception:
            output = module(value.unsqueeze(0))
            output = first_tensor(output)
            if output.ndim == 3 and output.shape[0] == 1:
                output = output[0]
            return output
        return first_tensor(output)

    with torch.inference_mode():
        projected = call(work)
        baseline = call(zeros)
    if projected.ndim == 3 and projected.shape[0] == 1:
        projected = projected[0]
    if baseline.ndim == 3 and baseline.shape[0] == 1:
        baseline = baseline[0]
    if projected.ndim != 2:
        raise RuntimeError(
            f"Output projection returned unexpected shape {tuple(projected.shape)}"
        )
    return projected - baseline


def project_all_heads_isolated(
    head_vectors: torch.Tensor,
    o_proj: Optional[torch.nn.Module],
) -> Optional[torch.Tensor]:
    """Project [H,Dh] with one active head per row, returning [H,Dmodel]."""
    if o_proj is None or head_vectors.ndim != 2:
        return None
    n_heads, head_dim = [int(x) for x in head_vectors.shape]
    packed = torch.zeros(
        (n_heads, n_heads * head_dim),
        dtype=head_vectors.dtype,
        device=head_vectors.device,
    )
    for head in range(n_heads):
        start = head * head_dim
        packed[head, start:start + head_dim] = head_vectors[head]
    try:
        return call_projection_module(o_proj, packed)
    except Exception:
        # Weight-slice fallback for standard/transposed linear modules.
        weight = getattr(o_proj, "weight", None)
        if not torch.is_tensor(weight):
            return None
        target = head_vectors.to(device=weight.device, dtype=weight.dtype)
        if weight.shape[1] == n_heads * head_dim:
            reshaped = weight.reshape(weight.shape[0], n_heads, head_dim)
            return torch.einsum("hd,ohd->ho", target, reshaped)
        if weight.shape[0] == n_heads * head_dim:
            reshaped = weight.reshape(n_heads, head_dim, weight.shape[1])
            return torch.einsum("hd,hdo->ho", target, reshaped)
        return None


def project_one_head_isolated(
    vectors: torch.Tensor,
    *,
    o_proj: Optional[torch.nn.Module],
    head_index: int,
    total_heads: int,
) -> Optional[torch.Tensor]:
    """Project [N,Dh] through one head's isolated o_proj input slice."""
    if o_proj is None or vectors.ndim != 2:
        return None
    n_rows, head_dim = [int(x) for x in vectors.shape]
    packed = torch.zeros(
        (n_rows, total_heads * head_dim),
        dtype=vectors.dtype,
        device=vectors.device,
    )
    start = int(head_index) * head_dim
    packed[:, start:start + head_dim] = vectors
    try:
        return call_projection_module(o_proj, packed)
    except Exception:
        weight = getattr(o_proj, "weight", None)
        if not torch.is_tensor(weight):
            return None
        target = vectors.to(device=weight.device, dtype=weight.dtype)
        if weight.shape[1] == total_heads * head_dim:
            head_weight = weight[:, start:start + head_dim]
            return F.linear(target, head_weight, None)
        if weight.shape[0] == total_heads * head_dim:
            head_weight = weight[start:start + head_dim, :]
            return target @ head_weight
        return None


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")

class LayerTraceCollector:
    """Capture token states and projected values needed by the trace."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        selected_spatial_layers: Sequence[int],
    ) -> None:
        self.layers = list(layers)
        self.selected_spatial_layers = {
            int(x) for x in selected_spatial_layers
        }
        self.attention_modules: List[torch.nn.Module] = []
        self.v_projections: List[Optional[torch.nn.Module]] = []
        self.o_projections: List[Optional[torch.nn.Module]] = []
        self.v_projection_paths: List[str] = []
        self.o_projection_paths: List[str] = []
        self.handles: List[Any] = []
        self.active = False
        self.subject_indices: List[int] = []
        self.reference_indices: List[int] = []
        self.visual_indices: List[int] = []
        self.last_index = -1
        self.reset_storage()

        for layer_index, layer in enumerate(self.layers):
            attention, _ = resolve_self_attention(layer)
            v_proj, v_path = resolve_projection_with_path(
                attention, ("v_proj", "value_proj")
            )
            o_proj, o_path = resolve_projection_with_path(
                attention, ("o_proj", "out_proj", "output_proj", "dense")
            )
            self.attention_modules.append(attention)
            self.v_projections.append(v_proj)
            self.o_projections.append(o_proj)
            self.v_projection_paths.append(v_path)
            self.o_projection_paths.append(o_path)

            self.handles.append(
                layer.register_forward_pre_hook(
                    self._make_layer_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_pre_hook(
                    self._make_attention_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._make_attention_hook(layer_index)
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    self._make_layer_hook(layer_index)
                )
            )

    def reset_storage(self) -> None:
        n = len(getattr(self, "layers", []))
        self.layer_input_last: List[Optional[torch.Tensor]] = [None] * n
        self.layer_input_subject: List[Optional[torch.Tensor]] = [None] * n
        self.layer_input_reference: List[Optional[torch.Tensor]] = [None] * n
        self.attention_output_last: List[Optional[torch.Tensor]] = [None] * n
        self.layer_output_last: List[Optional[torch.Tensor]] = [None] * n
        self.layer_output_subject: List[Optional[torch.Tensor]] = [None] * n
        self.layer_output_reference: List[Optional[torch.Tensor]] = [None] * n
        self.object_values: List[Optional[torch.Tensor]] = [None] * n
        self.visual_values: List[Optional[torch.Tensor]] = [None] * n

    def set_sample(
        self,
        *,
        subject_indices: Sequence[int],
        reference_indices: Sequence[int],
        visual_indices: Sequence[int],
        last_index: int,
    ) -> None:
        self.subject_indices = [int(x) for x in subject_indices]
        self.reference_indices = [int(x) for x in reference_indices]
        self.visual_indices = [int(x) for x in visual_indices]
        self.last_index = int(last_index)
        self.reset_storage()
        self.active = True

    def _hidden_input(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        kwargs = kwargs or {}
        for key in ("hidden_states", "x", "inputs_embeds"):
            value = kwargs.get(key)
            if torch.is_tensor(value) and value.ndim == 3:
                return value
        for value in args:
            if torch.is_tensor(value) and value.ndim == 3:
                return value
            try:
                tensor = first_tensor(value)
            except Exception:
                continue
            if tensor.ndim == 3:
                return tensor
        return None

    def _make_layer_pre_hook(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> None:
            if not self.active:
                return
            hidden = self._hidden_input(args, kwargs)
            if hidden is None:
                return
            self.layer_input_last[layer_index] = (
                hidden[0, self.last_index].detach().float().cpu()
            )
            self.layer_input_subject[layer_index] = (
                hidden[0, self.subject_indices].mean(dim=0)
                .detach().float().cpu()
            )
            self.layer_input_reference[layer_index] = (
                hidden[0, self.reference_indices].mean(dim=0)
                .detach().float().cpu()
            )
        return hook

    def _make_attention_pre_hook(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> None:
            if not self.active:
                return
            hidden = self._hidden_input(args, kwargs)
            v_proj = self.v_projections[layer_index]
            if hidden is None or v_proj is None:
                return
            object_indices = self.subject_indices + self.reference_indices
            # Qwen may pass float32 normalized hidden states into an
            # attention module whose projection weights are bfloat16. The
            # model's own attention forward handles that conversion internally,
            # but this diagnostic hook calls v_proj directly, so it must cast
            # the captured states to the projection module's device and dtype.
            projection_device, projection_dtype = module_device_dtype(
                v_proj,
                hidden,
            )
            object_input = hidden[:, object_indices, :].to(
                device=projection_device,
                dtype=projection_dtype,
            )
            with torch.inference_mode():
                object_values = v_proj(object_input)
                object_tensor = first_tensor(object_values)
                if object_tensor.ndim == 2:
                    object_tensor = object_tensor.unsqueeze(0)
                self.object_values[layer_index] = (
                    object_tensor[0].detach().cpu()
                )

                if layer_index in self.selected_spatial_layers:
                    visual_input = hidden[:, self.visual_indices, :].to(
                        device=projection_device,
                        dtype=projection_dtype,
                    )
                    visual_values = v_proj(visual_input)
                    visual_tensor = first_tensor(visual_values)
                    if visual_tensor.ndim == 2:
                        visual_tensor = visual_tensor.unsqueeze(0)
                    self.visual_values[layer_index] = (
                        visual_tensor[0].detach().cpu()
                    )
        return hook

    def _make_attention_hook(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            _args: Tuple[Any, ...],
            output: Any,
        ) -> None:
            if not self.active:
                return
            try:
                tensor = first_tensor(output)
            except Exception:
                return
            if tensor.ndim == 3:
                self.attention_output_last[layer_index] = (
                    tensor[0, self.last_index].detach().float().cpu()
                )
        return hook

    def _make_layer_hook(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            _args: Tuple[Any, ...],
            output: Any,
        ) -> None:
            if not self.active:
                return
            try:
                hidden = first_tensor(output)
            except Exception:
                return
            if hidden.ndim != 3:
                return
            self.layer_output_last[layer_index] = (
                hidden[0, self.last_index].detach().float().cpu()
            )
            self.layer_output_subject[layer_index] = (
                hidden[0, self.subject_indices].mean(dim=0)
                .detach().float().cpu()
            )
            self.layer_output_reference[layer_index] = (
                hidden[0, self.reference_indices].mean(dim=0)
                .detach().float().cpu()
            )
        return hook

    def projection_diagnostics(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for layer, (attention, v_proj, o_proj) in enumerate(zip(
            self.attention_modules,
            self.v_projections,
            self.o_projections,
        )):
            def shape_of(module: Optional[torch.nn.Module]) -> Optional[List[int]]:
                if module is None:
                    return None
                weight = getattr(module, "weight", None)
                return (
                    [int(x) for x in weight.shape]
                    if torch.is_tensor(weight)
                    else None
                )
            rows.append({
                "layer": layer,
                "attention_class": type(attention).__name__,
                "v_proj_found": v_proj is not None,
                "v_proj_path": self.v_projection_paths[layer],
                "v_proj_class": type(v_proj).__name__ if v_proj is not None else None,
                "v_proj_weight_shape": shape_of(v_proj),
                "o_proj_found": o_proj is not None,
                "o_proj_path": self.o_projection_paths[layer],
                "o_proj_class": type(o_proj).__name__ if o_proj is not None else None,
                "o_proj_weight_shape": shape_of(o_proj),
            })
        return rows

    def close(self) -> None:
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

def stack_required(
    values: Sequence[Optional[torch.Tensor]],
    name: str,
) -> torch.Tensor:
    missing = [i for i, value in enumerate(values) if value is None]
    if missing:
        raise RuntimeError(f"Missing {name} for layers {missing[:10]}")
    return torch.stack([value for value in values if value is not None], dim=0)


def build_prior_groups(
    analysis_rows: Sequence[Dict[str, Any]],
    generation_rows: Sequence[Dict[str, Any]],
    allowed_relations: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    analysis_by_sid = {
        int(row["sid"]): row for row in analysis_rows
    }
    merged: Dict[int, Dict[str, Any]] = {}
    for generation in generation_rows:
        sid = int(generation["sid"])
        analysis = analysis_by_sid.get(sid)
        if analysis is None:
            continue
        gt = normalize_relation(generation.get("gt"))
        if gt not in allowed_relations:
            continue
        if not bool(analysis.get("centroid_correct")):
            continue
        baseline_prediction = normalize_relation(
            generation.get("baseline_prediction")
        )
        if baseline_prediction is None:
            continue
        baseline_correct = bool(
            generation.get(
                "baseline_correct",
                baseline_prediction == gt,
            )
        )
        group = GROUP_CORRECT if baseline_correct else GROUP_WRONG
        merged[sid] = {
            "sid": sid,
            "group": group,
            "gt": gt,
            "baseline_prediction": baseline_prediction,
            "baseline_correct": baseline_correct,
            "centroid_prediction": normalize_relation(
                analysis.get("centroid_prediction")
            ),
            "centroid_confidence": float(
                analysis.get("centroid_confidence", np.nan)
            ),
            "axis_confidence": float(
                analysis.get("axis_confidence", np.nan)
            ),
            "head_agreement": float(
                analysis.get("head_agreement", np.nan)
            ),
            "swap_stability": float(
                analysis.get("swap_stability", np.nan)
            ),
            "mean_separation": float(
                analysis.get("mean_separation", np.nan)
            ),
            "mean_visual_mass": float(
                analysis.get("mean_visual_mass", np.nan)
            ),
            "centroid_delta_x": float(
                analysis.get("delta_x", np.nan)
            ),
            "centroid_delta_y": float(
                analysis.get("delta_y", np.nan)
            ),
            "prior_lm_margin": float(
                generation.get("lm_relation_margin", np.nan)
            ),
            "prior_lm_top": normalize_relation(
                generation.get("lm_relation_top")
            ),
        }
    return merged


def cap_groups(
    rows_by_sid: Dict[int, Dict[str, Any]],
    max_per_group: Optional[int],
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    if max_per_group is None:
        return rows_by_sid
    if max_per_group <= 0:
        raise ValueError("--max-per-group must be positive")
    rng = random.Random(seed)
    selected: Dict[int, Dict[str, Any]] = {}
    for group in GROUPS:
        rows = [
            row for row in rows_by_sid.values()
            if row["group"] == group
        ]
        rng.shuffle(rows)
        for row in rows[:max_per_group]:
            selected[int(row["sid"])] = row
    return selected


def load_selected_heads(prior_dir: Path) -> List[Dict[str, Any]]:
    config_path = prior_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        heads = config.get("selected_heads")
        if isinstance(heads, list) and heads:
            return [
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in ("layer", "head")
                    },
                }
                for row in heads
            ]
    analysis_rows = read_jsonl(prior_dir / "centroid_analysis.jsonl")
    for row in analysis_rows:
        head_rows = row.get("head_rows")
        if isinstance(head_rows, list) and head_rows:
            return [
                {
                    "layer": int(item["layer"]),
                    "head": int(item["head"]),
                }
                for item in head_rows
            ]
    raise RuntimeError(
        "Could not recover selected heads from prior config or analysis rows"
    )


def attention_and_av_metrics(
    *,
    attentions: Sequence[torch.Tensor],
    collector: LayerTraceCollector,
    subject_indices: Sequence[int],
    reference_indices: Sequence[int],
    visual_indices: Sequence[int],
    selected_heads: Sequence[Dict[str, Any]],
    last_index: int,
) -> Dict[str, np.ndarray]:
    n_layers = len(attentions)
    n_heads = int(attentions[0].shape[1])

    last_subject = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    last_reference = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    last_visual = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    routing_balance = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    routing_av_norm = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    routing_av_projected_norm = np.full(
        (n_layers, n_heads), np.nan, dtype=np.float32
    )
    routing_projection_available = np.zeros(
        (n_layers, n_heads), dtype=np.int8
    )

    selected_count = len(selected_heads)
    spatial_subject_visual_mass = np.full(selected_count, np.nan, dtype=np.float32)
    spatial_reference_visual_mass = np.full(selected_count, np.nan, dtype=np.float32)
    spatial_subject_av_norm = np.full(selected_count, np.nan, dtype=np.float32)
    spatial_reference_av_norm = np.full(selected_count, np.nan, dtype=np.float32)
    spatial_subject_av_projected_norm = np.full(
        selected_count, np.nan, dtype=np.float32
    )
    spatial_reference_av_projected_norm = np.full(
        selected_count, np.nan, dtype=np.float32
    )
    spatial_av_difference_norm = np.full(selected_count, np.nan, dtype=np.float32)
    spatial_projection_available = np.zeros(selected_count, dtype=np.int8)

    selected_lookup = {
        (int(row["layer"]), int(row["head"])): index
        for index, row in enumerate(selected_heads)
    }

    subj = torch.tensor(subject_indices, dtype=torch.long)
    ref = torch.tensor(reference_indices, dtype=torch.long)
    vis = torch.tensor(visual_indices, dtype=torch.long)

    for layer_index, attention_tensor in enumerate(attentions):
        attention = attention_tensor[0].detach().float().cpu()
        if attention.ndim != 3:
            raise RuntimeError(
                f"Layer {layer_index} attention shape={tuple(attention.shape)}"
            )
        if int(attention.shape[0]) != n_heads:
            raise RuntimeError("Head count changed across layers")

        last_row = attention[:, last_index, :]
        last_subject_t = last_row.index_select(-1, subj).sum(dim=-1)
        last_reference_t = last_row.index_select(-1, ref).sum(dim=-1)
        last_visual_t = last_row.index_select(-1, vis).sum(dim=-1)
        denominator = last_subject_t + last_reference_t
        balance_t = (
            2.0 * torch.minimum(last_subject_t, last_reference_t)
            / (denominator + 1e-8)
        )

        last_subject[layer_index] = last_subject_t.numpy()
        last_reference[layer_index] = last_reference_t.numpy()
        last_visual[layer_index] = last_visual_t.numpy()
        routing_balance[layer_index] = balance_t.numpy()

        object_values_flat = collector.object_values[layer_index]
        if object_values_flat is not None:
            values = reshape_projected_values(
                object_values_flat,
                n_attention_heads=n_heads,
                attention_module=collector.attention_modules[layer_index],
            ).float()
            n_subject = len(subject_indices)
            values_subject = values[:n_subject].permute(1, 0, 2)
            values_reference = values[n_subject:].permute(1, 0, 2)
            weights_subject = last_row.index_select(-1, subj)
            weights_reference = last_row.index_select(-1, ref)
            contribution = (
                torch.einsum("ht,htd->hd", weights_subject, values_subject)
                + torch.einsum("ht,htd->hd", weights_reference, values_reference)
            )
            routing_av_norm[layer_index] = (
                contribution.norm(dim=-1).numpy().astype(np.float32)
            )
            projected = project_all_heads_isolated(
                contribution,
                collector.o_projections[layer_index],
            )
            if projected is not None:
                projected_norm = (
                    projected.detach().float().cpu().norm(dim=-1).numpy()
                    .astype(np.float32)
                )
                if len(projected_norm) == n_heads:
                    routing_av_projected_norm[layer_index] = projected_norm
                    routing_projection_available[layer_index] = 1

        visual_values_flat = collector.visual_values[layer_index]
        if visual_values_flat is None:
            continue
        visual_values = reshape_projected_values(
            visual_values_flat,
            n_attention_heads=n_heads,
            attention_module=collector.attention_modules[layer_index],
        ).float().permute(1, 0, 2)

        for head in range(n_heads):
            selected_index = selected_lookup.get((layer_index, head))
            if selected_index is None:
                continue

            subject_weights = (
                attention[head].index_select(0, subj).index_select(1, vis)
            )
            reference_weights = (
                attention[head].index_select(0, ref).index_select(1, vis)
            )
            spatial_subject_visual_mass[selected_index] = float(
                subject_weights.sum(dim=-1).mean().item()
            )
            spatial_reference_visual_mass[selected_index] = float(
                reference_weights.sum(dim=-1).mean().item()
            )

            value_head = visual_values[head]
            subject_av = torch.einsum(
                "qv,vd->qd", subject_weights, value_head
            ).mean(dim=0)
            reference_av = torch.einsum(
                "qv,vd->qd", reference_weights, value_head
            ).mean(dim=0)

            spatial_subject_av_norm[selected_index] = float(subject_av.norm().item())
            spatial_reference_av_norm[selected_index] = float(reference_av.norm().item())
            spatial_av_difference_norm[selected_index] = float(
                (subject_av - reference_av).norm().item()
            )

            projected = project_one_head_isolated(
                torch.stack([subject_av, reference_av], dim=0),
                o_proj=collector.o_projections[layer_index],
                head_index=head,
                total_heads=n_heads,
            )
            if projected is not None:
                projected = projected.detach().float().cpu()
                spatial_subject_av_projected_norm[selected_index] = float(
                    projected[0].norm().item()
                )
                spatial_reference_av_projected_norm[selected_index] = float(
                    projected[1].norm().item()
                )
                spatial_projection_available[selected_index] = 1

    return {
        "last_to_subject_mass": last_subject,
        "last_to_reference_mass": last_reference,
        "last_to_visual_mass": last_visual,
        "routing_balance": routing_balance,
        "routing_av_norm": routing_av_norm,
        "routing_av_projected_norm": routing_av_projected_norm,
        "routing_projection_available": routing_projection_available,
        "spatial_subject_visual_mass": spatial_subject_visual_mass,
        "spatial_reference_visual_mass": spatial_reference_visual_mass,
        "spatial_subject_av_norm": spatial_subject_av_norm,
        "spatial_reference_av_norm": spatial_reference_av_norm,
        "spatial_subject_av_projected_norm": spatial_subject_av_projected_norm,
        "spatial_reference_av_projected_norm": spatial_reference_av_projected_norm,
        "spatial_av_difference_norm": spatial_av_difference_norm,
        "spatial_projection_available": spatial_projection_available,
    }

def bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or repetitions <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample_a = a[rng.integers(0, len(a), len(a))]
        sample_b = b[rng.integers(0, len(b), len(b))]
        differences[index] = sample_a.mean() - sample_b.mean()
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(
        (
            (len(a) - 1) * float(np.var(a, ddof=1))
            + (len(b) - 1) * float(np.var(b, ddof=1))
        )
        / max(1, len(a) + len(b) - 2)
    )
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def relation_macro_difference(
    values: np.ndarray,
    group_codes: np.ndarray,
    relation_codes: np.ndarray,
) -> float:
    differences = []
    for relation_index in range(len(RELATIONS)):
        mask_relation = relation_codes == relation_index
        a = values[
            mask_relation & (group_codes == 0)
        ]
        b = values[
            mask_relation & (group_codes == 1)
        ]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) and len(b):
            differences.append(float(np.mean(a) - np.mean(b)))
    return (
        float(np.mean(differences))
        if differences else float("nan")
    )


def comparison_row(
    *,
    metric: str,
    values: np.ndarray,
    group_codes: np.ndarray,
    relation_codes: np.ndarray,
    bootstrap_samples: int,
    seed: int,
    relation: str = "all",
) -> Dict[str, Any]:
    if relation == "all":
        mask = np.ones(len(values), dtype=bool)
    else:
        mask = relation_codes == RELATION_TO_INDEX[relation]
    a = values[mask & (group_codes == 0)]
    b = values[mask & (group_codes == 1)]
    a_valid = a[np.isfinite(a)]
    b_valid = b[np.isfinite(b)]
    ci_low, ci_high = bootstrap_difference(
        a_valid,
        b_valid,
        repetitions=bootstrap_samples,
        seed=seed,
    )
    return {
        "metric": metric,
        "relation": relation,
        "n_generation_correct": int(len(a_valid)),
        "n_generation_wrong": int(len(b_valid)),
        "mean_generation_correct": (
            float(np.mean(a_valid)) if len(a_valid) else float("nan")
        ),
        "mean_generation_wrong": (
            float(np.mean(b_valid)) if len(b_valid) else float("nan")
        ),
        "difference_correct_minus_wrong": (
            float(np.mean(a_valid) - np.mean(b_valid))
            if len(a_valid) and len(b_valid)
            else float("nan")
        ),
        "cohen_d_correct_minus_wrong": effect_size(a_valid, b_valid),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "relation_macro_difference": (
            relation_macro_difference(
                values,
                group_codes,
                relation_codes,
            )
            if relation == "all"
            else float("nan")
        ),
    }


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


def earliest_persistent_divergence(
    rows: Sequence[Dict[str, Any]],
    *,
    stage: str,
    minimum_effect: float = 0.30,
    consecutive: int = 2,
) -> Optional[int]:
    candidates = sorted(
        [
            row for row in rows
            if row.get("stage") == stage
            and row.get("relation") == "all"
        ],
        key=lambda row: int(row["layer"]),
    )
    for start in range(0, max(0, len(candidates) - consecutive + 1)):
        window = candidates[start:start + consecutive]
        valid = True
        for row in window:
            d = float(row["cohen_d_correct_minus_wrong"])
            low = float(row["bootstrap_ci_low"])
            high = float(row["bootstrap_ci_high"])
            if (
                not np.isfinite(d)
                or abs(d) < minimum_effect
                or not np.isfinite(low)
                or not np.isfinite(high)
                or low <= 0.0 <= high
            ):
                valid = False
                break
        if valid:
            return int(window[0]["layer"])
    return None



def invert_relation(value: Optional[str]) -> Optional[str]:
    mapping = {
        "left": "right",
        "right": "left",
        "above": "below",
        "below": "above",
    }
    normalized = normalize_relation(value)
    return mapping.get(normalized)


def build_swapped_question(subject: str, reference: str) -> str:
    return (
        f"Where is the {reference} in relation to the {subject}? "
        "Answer with left, right, above, or below."
    )


def make_question_batch(
    *,
    processor: Any,
    image: Image.Image,
    question_text: str,
    device: torch.device,
) -> Dict[str, Any]:
    rendered = build_prompt(processor, question_text)
    batch = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
    )
    return move_batch(batch, device)


def parse_matching_features(value: str) -> List[str]:
    features = [item.strip() for item in str(value).split(",") if item.strip()]
    if not features:
        raise ValueError("--matching-features resolved to an empty list")
    return features


def trace_one_prompt(
    *,
    model: Any,
    processor: Any,
    collector: LayerTraceCollector,
    batch: Dict[str, Any],
    subject: str,
    reference: str,
    gt: str,
    selected_heads: Sequence[Dict[str, Any]],
    final_norm: Optional[torch.nn.Module],
    token_weight: torch.Tensor,
    token_bias: Optional[torch.Tensor],
    relation_positions: Sequence[Sequence[int]],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    input_ids = [
        int(x) for x in batch["input_ids"][0].detach().cpu().tolist()
    ]
    subject_span, reference_span = locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_indices = span_indices(subject_span)
    reference_indices = span_indices(reference_span)
    visual_indices = resolve_visual_indices(
        model,
        processor,
        batch,
        input_ids,
    )
    last_index = len(input_ids) - 1

    collector.set_sample(
        subject_indices=subject_indices,
        reference_indices=reference_indices,
        visual_indices=visual_indices,
        last_index=last_index,
    )
    trace_stage = "model_forward"
    try:
        with torch.inference_mode():
            outputs = model(
                **batch,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        collector.active = False

    trace_stage = "extract_attentions"
    attentions = attention_tuple(outputs)
    n_layers = len(collector.layers)
    if len(attentions) != n_layers:
        raise RuntimeError(
            f"Expected {n_layers} attentions, got {len(attentions)}"
        )

    input_last = stack_required(
        collector.layer_input_last, "layer input last states"
    )
    attention_last = stack_required(
        collector.attention_output_last, "attention output last states"
    )
    after_attention_last = input_last + attention_last
    layer_output_last = stack_required(
        collector.layer_output_last, "layer output last states"
    )
    input_subject = stack_required(
        collector.layer_input_subject, "layer input subject states"
    )
    input_reference = stack_required(
        collector.layer_input_reference, "layer input reference states"
    )
    output_subject = stack_required(
        collector.layer_output_subject, "layer output subject states"
    )
    output_reference = stack_required(
        collector.layer_output_reference, "layer output reference states"
    )

    arrays: Dict[str, np.ndarray] = {}
    trace_stage = "logit_lens"
    stage_states = {
        "input": input_last,
        "after_attention": after_attention_last,
        "after_layer": layer_output_last,
    }
    gt_index = RELATION_TO_INDEX[gt]
    stage_margins: Dict[str, np.ndarray] = {}

    for stage, states in stage_states.items():
        relation_scores = relation_scores_from_states(
            states,
            final_norm=final_norm,
            token_weight=token_weight,
            token_bias=token_bias,
            relation_positions=relation_positions,
        )
        margin, rank, prediction = relation_diagnostics(
            relation_scores,
            gt_index,
        )
        arrays[f"logit_scores_{stage}"] = relation_scores
        arrays[f"gt_margin_{stage}"] = margin
        arrays[f"gt_rank_{stage}"] = rank
        arrays[f"prediction_{stage}"] = prediction
        stage_margins[stage] = margin

    arrays["gt_margin_gain_attention"] = (
        stage_margins["after_attention"] - stage_margins["input"]
    ).astype(np.float32)
    arrays["gt_margin_gain_mlp"] = (
        stage_margins["after_layer"] - stage_margins["after_attention"]
    ).astype(np.float32)
    arrays["gt_margin_gain_block"] = (
        stage_margins["after_layer"] - stage_margins["input"]
    ).astype(np.float32)

    layer0_baseline = float(stage_margins["input"][0])
    for stage in ("input", "after_attention", "after_layer"):
        arrays[f"gt_margin_layer0_normalized_{stage}"] = (
            stage_margins[stage] - layer0_baseline
        ).astype(np.float32)

    input_delta = input_subject - input_reference
    output_delta = output_subject - output_reference
    arrays["object_delta_norm_input"] = (
        input_delta.norm(dim=-1).numpy().astype(np.float32)
    )
    arrays["object_delta_norm_after_layer"] = (
        output_delta.norm(dim=-1).numpy().astype(np.float32)
    )
    arrays["object_cosine_input"] = safe_cosine_rows(
        input_subject, input_reference
    )
    arrays["object_cosine_after_layer"] = safe_cosine_rows(
        output_subject, output_reference
    )
    arrays["last_subject_cosine_input"] = safe_cosine_rows(
        input_last, input_subject
    )
    arrays["last_reference_cosine_input"] = safe_cosine_rows(
        input_last, input_reference
    )
    arrays["last_subject_cosine_after_layer"] = safe_cosine_rows(
        layer_output_last, output_subject
    )
    arrays["last_reference_cosine_after_layer"] = safe_cosine_rows(
        layer_output_last, output_reference
    )

    trace_stage = "attention_av_metrics"
    attention_metrics = attention_and_av_metrics(
        attentions=attentions,
        collector=collector,
        subject_indices=subject_indices,
        reference_indices=reference_indices,
        visual_indices=visual_indices,
        selected_heads=selected_heads,
        last_index=last_index,
    )
    arrays.update(attention_metrics)

    after_layer_margin = stage_margins["after_layer"]
    positive_layers = np.flatnonzero(after_layer_margin > 0.0)
    first_positive = int(positive_layers[0]) if len(positive_layers) else -1
    peak_layer = int(np.argmax(after_layer_margin))
    peak_margin = float(after_layer_margin[peak_layer])
    final_margin = float(after_layer_margin[-1])
    late_drop = peak_margin - final_margin

    routing_mass = (
        attention_metrics["last_to_subject_mass"]
        + attention_metrics["last_to_reference_mass"]
    )
    late_start = max(0, n_layers - max(4, n_layers // 4))

    raw_routing_av = attention_metrics["routing_av_norm"][late_start:]
    projected_routing_av = attention_metrics[
        "routing_av_projected_norm"
    ][late_start:]
    raw_spatial_av = 0.5 * (
        attention_metrics["spatial_subject_av_norm"]
        + attention_metrics["spatial_reference_av_norm"]
    )
    projected_spatial_av = 0.5 * (
        attention_metrics["spatial_subject_av_projected_norm"]
        + attention_metrics["spatial_reference_av_projected_norm"]
    )

    metadata = {
        "subject_span": list(subject_span),
        "reference_span": list(reference_span),
        "subject_token_count": len(subject_indices),
        "reference_token_count": len(reference_indices),
        "n_visual_tokens": len(visual_indices),
        "prompt_length": len(input_ids),
        "first_positive_layer_after_layer": first_positive,
        "peak_gt_margin_layer_after_layer": peak_layer,
        "peak_gt_margin_after_layer": peak_margin,
        "final_gt_margin_after_layer": final_margin,
        "late_margin_drop": late_drop,
        "late_last_to_object_mass_mean": finite_mean(
            routing_mass[late_start:]
        ),
        "late_routing_av_raw_norm_mean": finite_mean(raw_routing_av),
        "late_routing_av_projected_norm_mean": finite_mean(
            projected_routing_av
        ),
        "selected_spatial_visual_mass_mean": finite_mean(
            0.5 * (
                attention_metrics["spatial_subject_visual_mass"]
                + attention_metrics["spatial_reference_visual_mass"]
            )
        ),
        "selected_spatial_av_raw_norm_mean": finite_mean(raw_spatial_av),
        "selected_spatial_av_projected_norm_mean": finite_mean(
            projected_spatial_av
        ),
        "routing_projection_coverage": float(np.mean(
            attention_metrics["routing_projection_available"]
        )),
        "spatial_projection_coverage": float(np.mean(
            attention_metrics["spatial_projection_available"]
        )),
    }

    del outputs, attentions
    return metadata, arrays


def build_matched_pairs(
    metadata_rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    mode: str,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for relation in RELATIONS:
        correct_indices = [
            index for index, row in enumerate(metadata_rows)
            if row["group"] == GROUP_CORRECT and row["gt"] == relation
        ]
        wrong_indices = [
            index for index, row in enumerate(metadata_rows)
            if row["group"] == GROUP_WRONG and row["gt"] == relation
        ]
        if not correct_indices or not wrong_indices:
            continue

        all_indices = correct_indices + wrong_indices
        matrix = np.asarray([
            [
                float(metadata_rows[index].get(feature, np.nan))
                for feature in feature_names
            ]
            for index in all_indices
        ], dtype=np.float64)

        # Median-impute and robust-standardize within relation.
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            values[~np.isfinite(values)] = median
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            scale = 1.4826 * mad
            if scale <= 1e-8:
                scale = float(np.std(values))
            if scale <= 1e-8:
                scale = 1.0
            matrix[:, column] = (values - center) / scale

        correct_matrix = matrix[:len(correct_indices)]
        wrong_matrix = matrix[len(correct_indices):]
        available = set(range(len(correct_indices)))

        order = sorted(
            range(len(wrong_indices)),
            key=lambda local: int(metadata_rows[wrong_indices[local]]["sid"]),
        )
        for wrong_local in order:
            if mode == "unique":
                if not available:
                    break
                candidates = sorted(available)
            else:
                candidates = list(range(len(correct_indices)))
            if not candidates:
                break
            distances = [
                float(np.linalg.norm(
                    correct_matrix[candidate] - wrong_matrix[wrong_local]
                ))
                for candidate in candidates
            ]
            best_position = int(np.argmin(distances))
            best_local = candidates[best_position]
            if mode == "unique":
                available.discard(best_local)

            correct_index = correct_indices[best_local]
            wrong_index = wrong_indices[wrong_local]
            pairs.append({
                "relation": relation,
                "correct_index": correct_index,
                "wrong_index": wrong_index,
                "correct_sid": int(metadata_rows[correct_index]["sid"]),
                "wrong_sid": int(metadata_rows[wrong_index]["sid"]),
                "distance": float(distances[best_position]),
            })
    return pairs


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or repetitions <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = values[rng.integers(0, len(values), len(values))]
        means[index] = float(np.mean(sample))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison_row(
    *,
    metric: str,
    values: np.ndarray,
    pairs: Sequence[Dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    layer: Optional[int] = None,
    component: Optional[str] = None,
) -> Dict[str, Any]:
    correct_values = np.asarray([
        values[int(pair["correct_index"])] for pair in pairs
    ], dtype=np.float64)
    wrong_values = np.asarray([
        values[int(pair["wrong_index"])] for pair in pairs
    ], dtype=np.float64)
    differences = correct_values - wrong_values
    finite = np.isfinite(differences)
    differences = differences[finite]
    correct_values = correct_values[finite]
    wrong_values = wrong_values[finite]
    low, high = paired_bootstrap_interval(
        differences,
        repetitions=bootstrap_samples,
        seed=seed,
    )
    std = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    row: Dict[str, Any] = {
        "metric": metric,
        "n_pairs": int(len(differences)),
        "mean_correct": (
            float(np.mean(correct_values)) if len(correct_values) else float("nan")
        ),
        "mean_wrong": (
            float(np.mean(wrong_values)) if len(wrong_values) else float("nan")
        ),
        "mean_paired_difference": (
            float(np.mean(differences)) if len(differences) else float("nan")
        ),
        "paired_cohen_dz": (
            float(np.mean(differences) / std)
            if len(differences) > 1 and std > 1e-12
            else float("nan")
        ),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }
    if layer is not None:
        row["layer"] = int(layer)
    if component is not None:
        row["component"] = component
    return row


def earliest_persistent_paired_divergence(
    rows: Sequence[Dict[str, Any]],
    *,
    component: str,
    minimum_effect: float,
    consecutive: int,
) -> Optional[int]:
    candidates = sorted(
        [
            row for row in rows
            if row.get("component") == component
        ],
        key=lambda row: int(row["layer"]),
    )
    for start in range(max(0, len(candidates) - consecutive + 1)):
        window = candidates[start:start + consecutive]
        if len(window) < consecutive:
            continue
        valid = True
        for row in window:
            effect = float(row["paired_cohen_dz"])
            low = float(row["bootstrap_ci_low"])
            high = float(row["bootstrap_ci_high"])
            if (
                not np.isfinite(effect)
                or abs(effect) < minimum_effect
                or not np.isfinite(low)
                or not np.isfinite(high)
                or low <= 0.0 <= high
            ):
                valid = False
                break
        if valid:
            return int(window[0]["layer"])
    return None


def load_step1_samples(step1_dir: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(step1_dir / "samples.jsonl")
    return {int(row["sid"]): row for row in rows}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be >= 0")
    if args.persistent_layers <= 0:
        raise ValueError("--persistent-layers must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prior_dir = Path(args.prior_dir)
    analysis_path = (
        Path(args.centroid_analysis_jsonl)
        if args.centroid_analysis_jsonl
        else prior_dir / "centroid_analysis.jsonl"
    )
    generation_path = (
        Path(args.generation_jsonl)
        if args.generation_jsonl
        else prior_dir / "generation.jsonl"
    )
    if not analysis_path.exists():
        raise FileNotFoundError(analysis_path)
    if not generation_path.exists():
        raise FileNotFoundError(generation_path)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_relations = parse_relations(args.relations)
    matching_features = parse_matching_features(args.matching_features)
    analysis_rows = read_jsonl(analysis_path)
    generation_rows = read_jsonl(generation_path)
    prior_rows = build_prior_groups(
        analysis_rows,
        generation_rows,
        allowed_relations,
    )
    prior_rows = cap_groups(
        prior_rows,
        args.max_per_group,
        args.seed,
    )
    if not prior_rows:
        raise RuntimeError("No centroid-correct group samples were selected")

    selected_heads = load_selected_heads(prior_dir)
    selected_layers = sorted({
        int(row["layer"]) for row in selected_heads
    })

    print("\nSelected spatial centroid heads:")
    for rank, row in enumerate(selected_heads, 1):
        print(f"  {rank:2d}. L{int(row['layer']):02d}H{int(row['head']):02d}")

    group_counter = Counter(row["group"] for row in prior_rows.values())
    print("\nSelected comparison groups:")
    for group in GROUPS:
        print(f"  {group}: {group_counter[group]}")

    step1_samples: Dict[int, Dict[str, Any]] = {}
    swap_pair_sids: List[int] = []
    if args.run_swap_pairs:
        if not args.step1_dir:
            raise ValueError("--run-swap-pairs requires --step1-dir")
        step1_samples = load_step1_samples(Path(args.step1_dir))
        swap_pair_sids = [
            sid for sid, prior in prior_rows.items()
            if sid in step1_samples
            and bool(step1_samples[sid].get("original_correct"))
            != bool(step1_samples[sid].get("swapped_aligned_correct"))
        ]
        rng = random.Random(args.seed)
        rng.shuffle(swap_pair_sids)
        if args.max_swap_pairs is not None:
            swap_pair_sids = swap_pair_sids[:args.max_swap_pairs]
        swap_pair_sids.sort()
        print(f"  original/swap divergent pairs: {len(swap_pair_sids)}")

    module = import_two_object_module()
    records, audit = module.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {int(record.sid): record for record in records}
    prompt_path = resolve_prompt_path(args)
    prompt_rows = load_standard_prompts(prompt_path)

    required_sids = set(prior_rows) | set(swap_pair_sids)
    missing = [
        sid for sid in required_sids
        if sid not in record_by_sid or sid not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} selected records/prompts; first={missing[:10]}"
        )

    if args.model not in module.SPECS:
        raise ValueError(f"Unknown model: {args.model}")
    spec = module.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": "eager",
    }
    print(f"\nLoading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = resolve_decoder_layers(model)
    n_layers = len(layers)
    if selected_layers and max(selected_layers) >= n_layers:
        raise RuntimeError(
            f"Selected layer {max(selected_layers)} outside model with {n_layers} layers"
        )
    final_norm, final_norm_path = resolve_final_norm(model)
    label_token_ids = label_token_id_variants(processor.tokenizer)
    token_weight, token_bias, relation_positions = relation_token_rows(
        model,
        label_token_ids,
    )

    collector = LayerTraceCollector(layers, selected_layers)
    projection_diagnostics = collector.projection_diagnostics()
    (output_dir / "projection_diagnostics.json").write_text(
        json.dumps(projection_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Decoder layers: {layers_path} ({n_layers})")
    print(f"Final norm:     {final_norm_path}")
    print(
        "Projection discovery: "
        f"v_proj={sum(int(row['v_proj_found']) for row in projection_diagnostics)}/{n_layers}, "
        f"o_proj={sum(int(row['o_proj_found']) for row in projection_diagnostics)}/{n_layers}"
    )

    metadata_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, List[np.ndarray]] = defaultdict(list)
    original_trace_by_sid: Dict[int, Dict[str, Any]] = {}
    errors_path = output_dir / "errors.jsonl"
    metadata_path = output_dir / "sample_metadata.jsonl"
    for path in (metadata_path, errors_path):
        if path.exists():
            path.unlink()

    started = time.time()
    completed = 0

    try:
        for sid in tqdm(sorted(prior_rows), desc=f"group-trace:{args.model}"):
            batch = None
            image = None
            try:
                failure_stage = "sample_setup"
                prior = prior_rows[sid]
                record = record_by_sid[sid]
                prompt_row = prompt_rows[sid]
                image = record_image(record)
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt != prior["gt"]:
                    raise RuntimeError(
                        f"GT mismatch sid={sid}: prompt={gt}, prior={prior['gt']}"
                    )
                batch = make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                failure_stage = "trace_one_prompt"
                trace_meta, trace_arrays = trace_one_prompt(
                    model=model,
                    processor=processor,
                    collector=collector,
                    batch=batch,
                    subject=subject,
                    reference=reference,
                    gt=gt,
                    selected_heads=selected_heads,
                    final_norm=final_norm,
                    token_weight=token_weight,
                    token_bias=token_bias,
                    relation_positions=relation_positions,
                )
                failure_stage = "post_trace_metadata"
                metadata = {
                    **prior,
                    **trace_meta,
                    "subject": subject,
                    "reference": reference,
                    "question": question,
                }
                row_index = len(metadata_rows)
                metadata["row_index"] = row_index
                metadata_rows.append(metadata)
                append_jsonl(metadata_path, metadata)
                for key, value in trace_arrays.items():
                    arrays[key].append(value)
                original_trace_by_sid[sid] = {
                    "metadata": metadata,
                    "arrays": trace_arrays,
                }

                completed += 1
                if should_print_sample(completed, args.print_every):
                    tqdm.write(
                        f"\n[{completed}/{len(prior_rows)}] sid={sid} | "
                        f"{prior['group']} | GT={gt} | base={prior['baseline_prediction']}\n"
                        f"  conf={prior['centroid_confidence']:.4f} | "
                        f"peak=L{trace_meta['peak_gt_margin_layer_after_layer']} "
                        f"{trace_meta['peak_gt_margin_after_layer']:+.3f} | "
                        f"final={trace_meta['final_gt_margin_after_layer']:+.3f} | "
                        f"proj spatial/routing="
                        f"{trace_meta['spatial_projection_coverage']:.2f}/"
                        f"{trace_meta['routing_projection_coverage']:.2f}"
                    )
            except Exception as exc:
                append_jsonl(errors_path, {
                    "stage": locals().get("failure_stage", "group_trace_setup"),
                    "sid": sid,
                    "group": prior_rows.get(sid, {}).get("group"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
                tqdm.write(f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}")
            finally:
                if batch is not None:
                    del batch
                if image is not None:
                    del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not metadata_rows:
            raise RuntimeError("No group samples were traced successfully")

        stacked = {
            key: np.stack(values, axis=0)
            for key, values in arrays.items()
        }
        sids = np.asarray(
            [int(row["sid"]) for row in metadata_rows], dtype=np.int64
        )
        group_codes = np.asarray(
            [0 if row["group"] == GROUP_CORRECT else 1 for row in metadata_rows],
            dtype=np.int8,
        )
        relation_codes = np.asarray(
            [RELATION_TO_INDEX[row["gt"]] for row in metadata_rows],
            dtype=np.int8,
        )
        stacked.update({
            "sids": sids,
            "group_codes": group_codes,
            "relation_codes": relation_codes,
            "layer_indices": np.arange(n_layers, dtype=np.int16),
            "selected_spatial_layers": np.asarray(
                [int(row["layer"]) for row in selected_heads], dtype=np.int16
            ),
            "selected_spatial_heads": np.asarray(
                [int(row["head"]) for row in selected_heads], dtype=np.int16
            ),
        })
        np.savez_compressed(output_dir / "trace_arrays.npz", **stacked)

        scalar_metrics = {
            name: np.asarray(
                [float(row.get(name, np.nan)) for row in metadata_rows],
                dtype=np.float32,
            )
            for name in [
                "centroid_confidence",
                "axis_confidence",
                "head_agreement",
                "swap_stability",
                "mean_separation",
                "mean_visual_mass",
                "prompt_length",
                "subject_token_count",
                "reference_token_count",
                "prior_lm_margin",
                "peak_gt_margin_after_layer",
                "final_gt_margin_after_layer",
                "late_margin_drop",
                "late_last_to_object_mass_mean",
                "late_routing_av_raw_norm_mean",
                "late_routing_av_projected_norm_mean",
                "selected_spatial_visual_mass_mean",
                "selected_spatial_av_raw_norm_mean",
                "selected_spatial_av_projected_norm_mean",
                "routing_projection_coverage",
                "spatial_projection_coverage",
            ]
        }

        scalar_rows: List[Dict[str, Any]] = []
        for metric, values in scalar_metrics.items():
            for relation in ("all", *allowed_relations):
                scalar_rows.append(comparison_row(
                    metric=metric,
                    values=values,
                    group_codes=group_codes,
                    relation_codes=relation_codes,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + len(scalar_rows),
                    relation=relation,
                ))
        write_csv(output_dir / "group_scalar_comparison.csv", scalar_rows)

        absolute_rows: List[Dict[str, Any]] = []
        for stage in ("input", "after_attention", "after_layer"):
            matrix = stacked[f"gt_margin_{stage}"]
            for layer_index in range(n_layers):
                for relation in ("all", *allowed_relations):
                    row = comparison_row(
                        metric="gt_relation_margin",
                        values=matrix[:, layer_index],
                        group_codes=group_codes,
                        relation_codes=relation_codes,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 1000 + len(absolute_rows),
                        relation=relation,
                    )
                    row.update({"stage": stage, "layer": layer_index})
                    absolute_rows.append(row)
        write_csv(output_dir / "layerwise_logit_lens_absolute.csv", absolute_rows)

        normalized_rows: List[Dict[str, Any]] = []
        for stage in ("input", "after_attention", "after_layer"):
            matrix = stacked[f"gt_margin_layer0_normalized_{stage}"]
            for layer_index in range(n_layers):
                for relation in ("all", *allowed_relations):
                    row = comparison_row(
                        metric="layer0_normalized_gt_margin",
                        values=matrix[:, layer_index],
                        group_codes=group_codes,
                        relation_codes=relation_codes,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 3000 + len(normalized_rows),
                        relation=relation,
                    )
                    row.update({"stage": stage, "layer": layer_index})
                    normalized_rows.append(row)
        write_csv(
            output_dir / "layerwise_logit_lens_layer0_normalized.csv",
            normalized_rows,
        )

        gain_rows: List[Dict[str, Any]] = []
        gain_components = {
            "attention": stacked["gt_margin_gain_attention"],
            "mlp": stacked["gt_margin_gain_mlp"],
            "block": stacked["gt_margin_gain_block"],
        }
        for component, matrix in gain_components.items():
            for layer_index in range(n_layers):
                for relation in ("all", *allowed_relations):
                    row = comparison_row(
                        metric="gt_margin_gain",
                        values=matrix[:, layer_index],
                        group_codes=group_codes,
                        relation_codes=relation_codes,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 5000 + len(gain_rows),
                        relation=relation,
                    )
                    row.update({"component": component, "layer": layer_index})
                    gain_rows.append(row)
        write_csv(output_dir / "layerwise_module_gains.csv", gain_rows)

        hidden_routing_metrics = {
            "object_delta_norm_input": stacked["object_delta_norm_input"],
            "object_delta_norm_after_layer": stacked[
                "object_delta_norm_after_layer"
            ],
            "object_cosine_input": stacked["object_cosine_input"],
            "object_cosine_after_layer": stacked[
                "object_cosine_after_layer"
            ],
            "last_to_object_mass_head_mean": np.nanmean(
                stacked["last_to_subject_mass"]
                + stacked["last_to_reference_mass"],
                axis=-1,
            ),
            "routing_balance_head_mean": np.nanmean(
                stacked["routing_balance"], axis=-1
            ),
            "last_to_visual_mass_head_mean": np.nanmean(
                stacked["last_to_visual_mass"], axis=-1
            ),
            "routing_av_raw_norm_head_mean": np.nanmean(
                stacked["routing_av_norm"], axis=-1
            ),
            "routing_av_projected_norm_head_mean": np.nanmean(
                stacked["routing_av_projected_norm"], axis=-1
            ),
        }
        hidden_rows: List[Dict[str, Any]] = []
        for metric, matrix in hidden_routing_metrics.items():
            for layer_index in range(n_layers):
                for relation in ("all", *allowed_relations):
                    row = comparison_row(
                        metric=metric,
                        values=matrix[:, layer_index],
                        group_codes=group_codes,
                        relation_codes=relation_codes,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 8000 + len(hidden_rows),
                        relation=relation,
                    )
                    row["layer"] = layer_index
                    hidden_rows.append(row)
        write_csv(output_dir / "layerwise_hidden_routing.csv", hidden_rows)

        # Same-relation, centroid-quality matched controls.
        matched_pairs = build_matched_pairs(
            metadata_rows,
            feature_names=matching_features,
            mode=args.matching_mode,
        )
        write_csv(output_dir / "matched_pairs.csv", [
            {
                **pair,
                **{
                    f"correct_{feature}": metadata_rows[
                        int(pair["correct_index"])
                    ].get(feature)
                    for feature in matching_features
                },
                **{
                    f"wrong_{feature}": metadata_rows[
                        int(pair["wrong_index"])
                    ].get(feature)
                    for feature in matching_features
                },
            }
            for pair in matched_pairs
        ])

        matched_scalar_rows = [
            paired_comparison_row(
                metric=metric,
                values=values,
                pairs=matched_pairs,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 11000 + index,
            )
            for index, (metric, values) in enumerate(scalar_metrics.items())
        ]
        write_csv(
            output_dir / "matched_scalar_comparison.csv",
            matched_scalar_rows,
        )

        matched_gain_rows: List[Dict[str, Any]] = []
        for component, matrix in gain_components.items():
            for layer_index in range(n_layers):
                matched_gain_rows.append(paired_comparison_row(
                    metric="gt_margin_gain",
                    values=matrix[:, layer_index],
                    pairs=matched_pairs,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 13000 + len(matched_gain_rows),
                    layer=layer_index,
                    component=component,
                ))
        write_csv(
            output_dir / "matched_layerwise_module_gains.csv",
            matched_gain_rows,
        )

        matched_normalized_rows: List[Dict[str, Any]] = []
        for stage in ("input", "after_attention", "after_layer"):
            matrix = stacked[f"gt_margin_layer0_normalized_{stage}"]
            for layer_index in range(n_layers):
                matched_normalized_rows.append(paired_comparison_row(
                    metric="layer0_normalized_gt_margin",
                    values=matrix[:, layer_index],
                    pairs=matched_pairs,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 15000 + len(matched_normalized_rows),
                    layer=layer_index,
                    component=stage,
                ))
        write_csv(
            output_dir / "matched_layerwise_normalized_margin.csv",
            matched_normalized_rows,
        )

        matched_hidden_rows: List[Dict[str, Any]] = []
        for metric, matrix in hidden_routing_metrics.items():
            for layer_index in range(n_layers):
                matched_hidden_rows.append(paired_comparison_row(
                    metric=metric,
                    values=matrix[:, layer_index],
                    pairs=matched_pairs,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 17000 + len(matched_hidden_rows),
                    layer=layer_index,
                ))
        write_csv(
            output_dir / "matched_layerwise_hidden_routing.csv",
            matched_hidden_rows,
        )

        # Per-head unmatched and matched routing analyses.
        routing_metric_arrays = {
            "last_to_object_mass": (
                stacked["last_to_subject_mass"]
                + stacked["last_to_reference_mass"]
            ),
            "routing_balance": stacked["routing_balance"],
            "routing_av_raw_norm": stacked["routing_av_norm"],
            "routing_av_projected_norm": stacked[
                "routing_av_projected_norm"
            ],
        }
        n_heads = routing_metric_arrays["last_to_object_mass"].shape[-1]
        routing_head_rows: List[Dict[str, Any]] = []
        matched_routing_head_rows: List[Dict[str, Any]] = []
        for layer_index in range(n_layers):
            for head in range(n_heads):
                unmatched_row: Dict[str, Any] = {
                    "layer": layer_index,
                    "head": head,
                }
                matched_row: Dict[str, Any] = {
                    "layer": layer_index,
                    "head": head,
                }
                for metric, cube in routing_metric_arrays.items():
                    values = cube[:, layer_index, head]
                    comparison = comparison_row(
                        metric=metric,
                        values=values,
                        group_codes=group_codes,
                        relation_codes=relation_codes,
                        bootstrap_samples=0,
                        seed=args.seed,
                        relation="all",
                    )
                    unmatched_row[f"{metric}_difference"] = comparison[
                        "difference_correct_minus_wrong"
                    ]
                    unmatched_row[f"{metric}_cohen_d"] = comparison[
                        "cohen_d_correct_minus_wrong"
                    ]
                    unmatched_row[
                        f"{metric}_relation_macro_difference"
                    ] = comparison["relation_macro_difference"]

                    paired = paired_comparison_row(
                        metric=metric,
                        values=values,
                        pairs=matched_pairs,
                        bootstrap_samples=0,
                        seed=args.seed,
                    )
                    matched_row[f"{metric}_paired_difference"] = paired[
                        "mean_paired_difference"
                    ]
                    matched_row[f"{metric}_paired_dz"] = paired[
                        "paired_cohen_dz"
                    ]
                routing_head_rows.append(unmatched_row)
                matched_routing_head_rows.append(matched_row)

        routing_head_rows.sort(
            key=lambda row: -abs(float(
                row.get("routing_av_projected_norm_cohen_d", 0.0)
            )) if np.isfinite(float(
                row.get("routing_av_projected_norm_cohen_d", np.nan)
            )) else float("inf")
        )
        matched_routing_head_rows.sort(
            key=lambda row: -abs(float(
                row.get("routing_av_projected_norm_paired_dz", 0.0)
            )) if np.isfinite(float(
                row.get("routing_av_projected_norm_paired_dz", np.nan)
            )) else float("inf")
        )
        write_csv(
            output_dir / "routing_head_differences.csv",
            routing_head_rows,
        )
        write_csv(
            output_dir / "matched_routing_head_differences.csv",
            matched_routing_head_rows,
        )

        spatial_metrics = {
            "subject_visual_mass": stacked["spatial_subject_visual_mass"],
            "reference_visual_mass": stacked[
                "spatial_reference_visual_mass"
            ],
            "subject_av_raw_norm": stacked["spatial_subject_av_norm"],
            "reference_av_raw_norm": stacked["spatial_reference_av_norm"],
            "subject_av_projected_norm": stacked[
                "spatial_subject_av_projected_norm"
            ],
            "reference_av_projected_norm": stacked[
                "spatial_reference_av_projected_norm"
            ],
            "av_difference_norm": stacked["spatial_av_difference_norm"],
        }
        spatial_head_rows: List[Dict[str, Any]] = []
        matched_spatial_head_rows: List[Dict[str, Any]] = []
        for selected_index, selected in enumerate(selected_heads):
            for metric, matrix in spatial_metrics.items():
                values = matrix[:, selected_index]
                comparison = comparison_row(
                    metric=metric,
                    values=values,
                    group_codes=group_codes,
                    relation_codes=relation_codes,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 21000 + len(spatial_head_rows),
                    relation="all",
                )
                comparison.update({
                    "selected_rank": selected_index + 1,
                    "layer": int(selected["layer"]),
                    "head": int(selected["head"]),
                })
                spatial_head_rows.append(comparison)

                paired = paired_comparison_row(
                    metric=metric,
                    values=values,
                    pairs=matched_pairs,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 23000 + len(matched_spatial_head_rows),
                )
                paired.update({
                    "selected_rank": selected_index + 1,
                    "layer": int(selected["layer"]),
                    "head": int(selected["head"]),
                })
                matched_spatial_head_rows.append(paired)
        write_csv(
            output_dir / "spatial_head_differences.csv",
            spatial_head_rows,
        )
        write_csv(
            output_dir / "matched_spatial_head_differences.csv",
            matched_spatial_head_rows,
        )

        # Candidate breakpoint ranking from matched module gains.
        candidate_rows: List[Dict[str, Any]] = []
        matched_gain_lookup = {
            (row["component"], int(row["layer"])): row
            for row in matched_gain_rows
        }
        matched_hidden_lookup = {
            (row["metric"], int(row["layer"])): row
            for row in matched_hidden_rows
        }
        for layer_index in range(n_layers):
            row = {"layer": layer_index}
            score_terms = []
            for component in ("attention", "mlp", "block"):
                gain = matched_gain_lookup[(component, layer_index)]
                effect = float(gain["paired_cohen_dz"])
                row[f"{component}_gain_difference"] = gain[
                    "mean_paired_difference"
                ]
                row[f"{component}_gain_dz"] = effect
                row[f"{component}_gain_ci_low"] = gain[
                    "bootstrap_ci_low"
                ]
                row[f"{component}_gain_ci_high"] = gain[
                    "bootstrap_ci_high"
                ]
                if np.isfinite(effect):
                    score_terms.append(abs(effect))
            for metric in (
                "last_to_object_mass_head_mean",
                "routing_av_raw_norm_head_mean",
                "routing_av_projected_norm_head_mean",
            ):
                hidden = matched_hidden_lookup[(metric, layer_index)]
                row[f"{metric}_difference"] = hidden[
                    "mean_paired_difference"
                ]
                row[f"{metric}_dz"] = hidden["paired_cohen_dz"]
            row["candidate_score"] = (
                float(max(score_terms)) if score_terms else float("nan")
            )
            candidate_rows.append(row)
        candidate_rows.sort(
            key=lambda row: -float(row["candidate_score"])
            if np.isfinite(float(row["candidate_score"]))
            else float("inf")
        )
        write_csv(output_dir / "candidate_breakpoint_layers.csv", candidate_rows)

        # Optional same-image original/swap paired experiment.
        swap_pair_rows: List[Dict[str, Any]] = []
        swap_gain_rows: List[Dict[str, Any]] = []
        swap_hidden_rows: List[Dict[str, Any]] = []
        if args.run_swap_pairs:
            for sid in tqdm(swap_pair_sids, desc="swap-pair-trace"):
                batch = None
                image = None
                try:
                    original_trace = original_trace_by_sid.get(sid)
                    if original_trace is None:
                        continue
                    step1 = step1_samples[sid]
                    record = record_by_sid[sid]
                    prompt_row = prompt_rows[sid]
                    image = record_image(record)
                    original_subject = str(prompt_row["subject"])
                    original_reference = str(prompt_row["reference"])
                    original_gt = normalize_relation(prompt_row["answer_raw"])
                    swapped_gt = invert_relation(original_gt)
                    swapped_question = build_swapped_question(
                        original_subject,
                        original_reference,
                    )
                    batch = make_question_batch(
                        processor=processor,
                        image=image,
                        question_text=swapped_question,
                        device=device,
                    )
                    swap_meta, swap_arrays = trace_one_prompt(
                        model=model,
                        processor=processor,
                        collector=collector,
                        batch=batch,
                        subject=original_reference,
                        reference=original_subject,
                        gt=swapped_gt,
                        selected_heads=selected_heads,
                        final_norm=final_norm,
                        token_weight=token_weight,
                        token_bias=token_bias,
                        relation_positions=relation_positions,
                    )

                    original_correct = bool(step1.get("original_correct"))
                    swapped_correct = bool(step1.get("swapped_aligned_correct"))
                    if original_correct == swapped_correct:
                        continue
                    correct_side = "original" if original_correct else "swapped"
                    wrong_side = "swapped" if original_correct else "original"

                    original_arrays = original_trace["arrays"]
                    correct_arrays = (
                        original_arrays if original_correct else swap_arrays
                    )
                    wrong_arrays = (
                        swap_arrays if original_correct else original_arrays
                    )
                    original_meta = original_trace["metadata"]
                    correct_meta = (
                        original_meta if original_correct else swap_meta
                    )
                    wrong_meta = (
                        swap_meta if original_correct else original_meta
                    )

                    swap_pair_rows.append({
                        "sid": sid,
                        "gt_original": original_gt,
                        "correct_side": correct_side,
                        "wrong_side": wrong_side,
                        "original_prediction": step1.get("original_prediction"),
                        "swapped_prediction_aligned": step1.get(
                            "swapped_prediction_aligned"
                        ),
                        "original_centroid_confidence": original_meta.get(
                            "centroid_confidence"
                        ),
                        "correct_final_margin": correct_meta[
                            "final_gt_margin_after_layer"
                        ],
                        "wrong_final_margin": wrong_meta[
                            "final_gt_margin_after_layer"
                        ],
                    })

                    for component, key in (
                        ("attention", "gt_margin_gain_attention"),
                        ("mlp", "gt_margin_gain_mlp"),
                        ("block", "gt_margin_gain_block"),
                    ):
                        difference = (
                            correct_arrays[key] - wrong_arrays[key]
                        )
                        for layer_index, value in enumerate(difference):
                            swap_gain_rows.append({
                                "sid": sid,
                                "component": component,
                                "layer": layer_index,
                                "correct_minus_wrong": float(value),
                            })

                    swap_hidden_metrics = {
                        "last_to_object_mass_head_mean": (
                            np.nanmean(
                                correct_arrays["last_to_subject_mass"]
                                + correct_arrays["last_to_reference_mass"],
                                axis=-1,
                            )
                            - np.nanmean(
                                wrong_arrays["last_to_subject_mass"]
                                + wrong_arrays["last_to_reference_mass"],
                                axis=-1,
                            )
                        ),
                        "routing_av_raw_norm_head_mean": (
                            np.nanmean(
                                correct_arrays["routing_av_norm"], axis=-1
                            )
                            - np.nanmean(
                                wrong_arrays["routing_av_norm"], axis=-1
                            )
                        ),
                        "routing_av_projected_norm_head_mean": (
                            np.nanmean(
                                correct_arrays["routing_av_projected_norm"],
                                axis=-1,
                            )
                            - np.nanmean(
                                wrong_arrays["routing_av_projected_norm"],
                                axis=-1,
                            )
                        ),
                    }
                    for metric, difference in swap_hidden_metrics.items():
                        for layer_index, value in enumerate(difference):
                            swap_hidden_rows.append({
                                "sid": sid,
                                "metric": metric,
                                "layer": layer_index,
                                "correct_minus_wrong": float(value),
                            })
                except Exception as exc:
                    append_jsonl(errors_path, {
                        "stage": "swap_pair_trace",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-20:],
                    })
                finally:
                    if batch is not None:
                        del batch
                    if image is not None:
                        del image
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            write_csv(output_dir / "swap_pair_metadata.csv", swap_pair_rows)

            swap_gain_summary: List[Dict[str, Any]] = []
            for component in ("attention", "mlp", "block"):
                for layer_index in range(n_layers):
                    values = np.asarray([
                        row["correct_minus_wrong"]
                        for row in swap_gain_rows
                        if row["component"] == component
                        and int(row["layer"]) == layer_index
                    ], dtype=np.float64)
                    low, high = paired_bootstrap_interval(
                        values,
                        repetitions=args.bootstrap_samples,
                        seed=args.seed + 26000 + len(swap_gain_summary),
                    )
                    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    swap_gain_summary.append({
                        "component": component,
                        "layer": layer_index,
                        "n_pairs": len(values),
                        "mean_correct_minus_wrong": (
                            float(np.mean(values)) if len(values) else float("nan")
                        ),
                        "paired_dz": (
                            float(np.mean(values) / std)
                            if len(values) > 1 and std > 1e-12
                            else float("nan")
                        ),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    })
            write_csv(
                output_dir / "swap_paired_module_gains.csv",
                swap_gain_summary,
            )

            swap_hidden_summary: List[Dict[str, Any]] = []
            for metric in (
                "last_to_object_mass_head_mean",
                "routing_av_raw_norm_head_mean",
                "routing_av_projected_norm_head_mean",
            ):
                for layer_index in range(n_layers):
                    values = np.asarray([
                        row["correct_minus_wrong"]
                        for row in swap_hidden_rows
                        if row["metric"] == metric
                        and int(row["layer"]) == layer_index
                    ], dtype=np.float64)
                    low, high = paired_bootstrap_interval(
                        values,
                        repetitions=args.bootstrap_samples,
                        seed=args.seed + 28000 + len(swap_hidden_summary),
                    )
                    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    swap_hidden_summary.append({
                        "metric": metric,
                        "layer": layer_index,
                        "n_pairs": len(values),
                        "mean_correct_minus_wrong": (
                            float(np.mean(values)) if len(values) else float("nan")
                        ),
                        "paired_dz": (
                            float(np.mean(values) / std)
                            if len(values) > 1 and std > 1e-12
                            else float("nan")
                        ),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    })
            write_csv(
                output_dir / "swap_paired_hidden_routing.csv",
                swap_hidden_summary,
            )

        group_counts = {
            group: int(np.sum(
                group_codes == (0 if group == GROUP_CORRECT else 1)
            ))
            for group in GROUPS
        }
        earliest_matched = {
            component: earliest_persistent_paired_divergence(
                matched_gain_rows,
                component=component,
                minimum_effect=args.minimum_effect,
                consecutive=args.persistent_layers,
            )
            for component in ("attention", "mlp", "block")
        }

        projection_summary = {
            "routing_projected_finite_fraction": float(np.mean(
                np.isfinite(stacked["routing_av_projected_norm"])
            )),
            "spatial_projected_finite_fraction": float(np.mean(
                np.isfinite(stacked["spatial_subject_av_projected_norm"])
                & np.isfinite(stacked["spatial_reference_av_projected_norm"])
            )),
            "routing_raw_finite_fraction": float(np.mean(
                np.isfinite(stacked["routing_av_norm"])
            )),
            "spatial_raw_finite_fraction": float(np.mean(
                np.isfinite(stacked["spatial_subject_av_norm"])
                & np.isfinite(stacked["spatial_reference_av_norm"])
            )),
        }

        summary = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "dataset": args.dataset,
            "prior_dir": str(prior_dir),
            "step1_dir": args.step1_dir,
            "n_successful": len(metadata_rows),
            "n_requested": len(prior_rows),
            "elapsed_minutes": (time.time() - started) / 60.0,
            "group_counts": group_counts,
            "matched_pair_count": len(matched_pairs),
            "matching_mode": args.matching_mode,
            "matching_features": matching_features,
            "selected_spatial_heads": selected_heads,
            "decoder_layers_path": layers_path,
            "final_norm_path": final_norm_path,
            "n_layers": n_layers,
            "projection_coverage": projection_summary,
            "earliest_persistent_matched_module_gain_divergence":
                earliest_matched,
            "swap_pair_count": len(swap_pair_rows),
            "interpretation_note": (
                "Absolute logit-lens differences are descriptive. Prefer "
                "matched module gains and same-image swap-paired differences "
                "when choosing causal intervention layers."
            ),
            "audit": audit,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 108)
        print("CENTROID-CORRECT GROUP TRACE V2")
        print("=" * 108)
        print(
            f"generation correct: {group_counts[GROUP_CORRECT]} | "
            f"generation wrong: {group_counts[GROUP_WRONG]} | "
            f"matched pairs: {len(matched_pairs)}"
        )
        print("\nProjection coverage:")
        for key, value in projection_summary.items():
            print(f"  {key:42s}: {value:.4f}")
        print("\nEarliest persistent matched module-gain divergence:")
        for component, layer_index in earliest_matched.items():
            print(f"  {component:10s}: {layer_index}")
        if args.run_swap_pairs:
            print(f"\nSame-image original/swap divergent pairs: {len(swap_pair_rows)}")

        print("\nSaved outputs:")
        for filename in [
            "projection_diagnostics.json",
            "sample_metadata.jsonl",
            "trace_arrays.npz",
            "group_scalar_comparison.csv",
            "layerwise_logit_lens_absolute.csv",
            "layerwise_logit_lens_layer0_normalized.csv",
            "layerwise_module_gains.csv",
            "layerwise_hidden_routing.csv",
            "matched_pairs.csv",
            "matched_scalar_comparison.csv",
            "matched_layerwise_module_gains.csv",
            "matched_layerwise_normalized_margin.csv",
            "matched_layerwise_hidden_routing.csv",
            "routing_head_differences.csv",
            "matched_routing_head_differences.csv",
            "spatial_head_differences.csv",
            "matched_spatial_head_differences.csv",
            "candidate_breakpoint_layers.csv",
            "summary.json",
        ]:
            print(f"  {output_dir / filename}")
        if args.run_swap_pairs:
            for filename in [
                "swap_pair_metadata.csv",
                "swap_paired_module_gains.csv",
                "swap_paired_hidden_routing.csv",
            ]:
                print(f"  {output_dir / filename}")
        if errors_path.exists():
            print(f"  {errors_path}")

    finally:
        collector.close()
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
