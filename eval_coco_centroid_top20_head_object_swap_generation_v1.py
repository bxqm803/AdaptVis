#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO Top-Centroid Head -> Single-Head A/B Output-Swap Generation Experiment
===========================================================================

Supported models
----------------
- llava-7b : llava-hf/llava-1.5-7b-hf
- qwen-3b  : Qwen/Qwen2.5-VL-3B-Instruct
- qwen-7b  : Qwen/Qwen2.5-VL-7B-Instruct

Dataset
-------
Uses the same COCO-two-object extraction convention as
extract_two_object_relation_states.py:

    data/coco_qa_two_obj.json
    data/val2017/{image_id:012d}.jpg

and the same standard question file used by the centroid V4 experiment:

    prompts/COCO_QA_two_obj_with_answer_four_options.jsonl

Experiment
----------
Phase 1: scan every decoder attention head.

For each COCO sample, run:

    Q_AB: Where is A in relation to B?
    Q_BA: Where is B in relation to A?

For every layer/head:
  1) obtain A/B object-token -> visual-token attention centroids;
  2) align Q_BA role order [B, A] back to semantic order [A, B];
  3) average original and aligned-swapped centroids;
  4) predict left/right/above/below from the averaged centroid pair;
  5) accumulate centroid accuracy.

Rank ALL heads by centroid accuracy and keep Top-K (default 20).

Phase 2: causal generation intervention.

For the flipped question Q_BA, whose object-token role order is [B, A],
take ONE Top-K head at a time and swap ONLY that head's pre-W_O output
slice at the two object-token positions:

    [head_output(B), head_output(A)]
        ->
    [head_output(A), head_output(B)]

Everything else is unchanged.  The patch is applied ONLY during prefill
(query_length == prompt_length); autoregressive decode steps are untouched.
Then greedy generation continues normally.

Outputs
-------
<output-dir>/
    config.json
    centroid_scan.npz
    top_heads.json
    scan_errors.jsonl
    interventions.jsonl
    intervention_errors.jsonl
    summary.json

The intervention result reports, per head:
- centroid accuracy/rank
- patched Q_BA accuracy
- prediction-change rate
- strict semantic reversal:
      baseline Q_BA is correct AND patched prediction == original Q_AB GT
- opposite-transition rate
- fixed / broken
- mean first-step relation-logit margin shift toward the ORIGINAL Q_AB relation

Notes
-----
1) This script is self-contained: it does NOT import the repository's other
   analysis scripts.
2) The head swap is at the input of self_attn.o_proj, i.e. the concatenated
   per-head attention output before W_O.  Only one head slice is changed.
3) The script loads attention with eager implementation because Phase 1 needs
   complete prompt attention probabilities.
4) By default an existing centroid_scan.npz is reused.  intervention rows are
   resumed by sid.  Use --overwrite to restart everything.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import re
import shutil
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-top-centroid-head-object-swap-generation-v1"

RELATIONS: Tuple[str, ...] = ("left", "right", "above", "below")
RELATION_TO_INDEX = {name: idx for idx, name in enumerate(RELATIONS)}
INVERSE_RELATION = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


# =============================================================================
# Model specs
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    model_class: str
    dtype_name: str
    trust_remote_code: bool = False


MODEL_SPECS: Dict[str, ModelSpec] = {
    "llava-7b": ModelSpec(
        alias="llava-7b",
        repo_id="llava-hf/llava-1.5-7b-hf",
        model_class="LlavaForConditionalGeneration",
        dtype_name="float16",
        trust_remote_code=False,
    ),
    "qwen-3b": ModelSpec(
        alias="qwen-3b",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_class="Qwen2_5_VLForConditionalGeneration",
        dtype_name="bfloat16",
        trust_remote_code=False,
    ),
    "qwen-7b": ModelSpec(
        alias="qwen-7b",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_class="Qwen2_5_VLForConditionalGeneration",
        dtype_name="bfloat16",
        trust_remote_code=False,
    ),
}


# =============================================================================
# Dataset
# =============================================================================

@dataclass(frozen=True)
class Record:
    sid: int
    image_id: str
    image_path: Path
    caption: str
    opposite_caption: str
    subject: str
    reference: str
    relation: str


REL_ALIAS = {
    "left": "left",
    "right": "right",
    "above": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "top": "above",
    "bottom": "below",
}


def canonical_phrase(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[\s\.,;:!?]+$", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_relation_caption(caption: str) -> Optional[Tuple[str, str, str]]:
    """Same relation-caption parser convention as the existing COCO extractor."""
    text = str(caption).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,!?:;\n\t")
    text = re.sub(
        r"^(?:a|an|the)?\s*(?:photo|picture|image)\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    copula = r"(?:(?:is|are)\s+)?"
    patterns: List[Tuple[str, str]] = [
        (
            rf"^(?P<s>.+?)\s+{copula}(?:to\s+the\s+)?(?P<r>left|right)\s+of\s+(?P<o>.+)$",
            "lr",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}(?:on\s+)?top\s+of\s+(?P<o>.+)$",
            "top",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}at\s+the\s+(?P<r>top|bottom)\s+of\s+(?P<o>.+)$",
            "tb",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}(?P<r>above|below|under|underneath)\s+(?P<o>.+)$",
            "vertical",
        ),
    ]
    for pattern, fixed_relation in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        subject = canonical_phrase(match.group("s"))
        reference = canonical_phrase(match.group("o"))
        if not subject or not reference or subject == reference:
            return None
        raw = match.groupdict().get("r")
        relation = fixed_relation if raw is None else str(raw).lower()
        if relation == "lr":
            relation = str(raw).lower()
        elif relation == "tb":
            relation = str(raw).lower()
        if relation not in REL_ALIAS:
            return None
        return subject, reference, REL_ALIAS[relation]
    return None


def load_coco_records(
    data_root: Path,
    max_samples: Optional[int],
) -> Tuple[List[Record], List[Dict[str, Any]]]:
    annotation_path = data_root / "coco_qa_two_obj.json"
    image_dir = data_root / "val2017"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotation_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    records: List[Record] = []
    audit: List[Dict[str, Any]] = []

    for sid, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            audit.append({"sid": sid, "reason": "invalid_row", "row": row})
            continue

        image_id, caption, opposite_caption = row[0], str(row[1]), str(row[2])
        parsed = parse_relation_caption(caption)
        if parsed is None:
            audit.append({
                "sid": sid,
                "reason": "caption_parse_failed",
                "caption": caption,
                "opposite_caption": opposite_caption,
            })
            continue

        subject, reference, relation = parsed
        image_path = image_dir / f"{int(image_id):012d}.jpg"
        if not image_path.exists():
            audit.append({
                "sid": sid,
                "reason": "image_missing",
                "image_id": str(image_id),
                "image_path": str(image_path),
            })
            continue

        records.append(
            Record(
                sid=sid,
                image_id=str(image_id),
                image_path=image_path,
                caption=caption,
                opposite_caption=opposite_caption,
                subject=subject,
                reference=reference,
                relation=relation,
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break

    return records, audit


# =============================================================================
# Standard prompts
# =============================================================================

STANDARD_OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?\s*Answer\s+with",
    flags=re.IGNORECASE | re.DOTALL,
)


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


# =============================================================================
# Generic helpers
# =============================================================================

def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


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


def move_batch(batch: Any, device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def configure_processor(model: Any, processor: Any) -> None:
    """LLaVA placeholder compatibility; no-op for Qwen where irrelevant."""
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


def clear_sampling_defaults(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    for field in ("temperature", "top_p", "top_k"):
        if hasattr(generation_config, field):
            setattr(generation_config, field, None)


def build_prompt(processor: Any, question_text: str) -> str:
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


def build_swapped_question(subject: str, reference: str) -> str:
    return (
        f"Where is the {reference} in relation to the {subject}? "
        "Answer with left, right, above, or below."
    )


def make_batch(
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


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out.input_ids
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


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
        if list(haystack[start:start + width]) == list(needle)
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
    """
    Locate the ordered pair that occurs in the user question.

    Q_AB -> subject=A, reference=B
    Q_BA -> subject=B, reference=A
    """
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

    # Prefer the latest ordered pair in the rendered user question.
    return max(valid, key=lambda pair: (pair[1][0], pair[0][0]))


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


def invert_relation(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return INVERSE_RELATION.get(value)


def get_attr_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_decoder_layers(model: Any) -> Tuple[Sequence[Any], str]:
    preferred = [
        "model.language_model.layers",          # Qwen2.5-VL recent HF
        "model.model.language_model.layers",
        "language_model.model.layers",          # LLaVA
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
            0 if any(k in item[0].lower() for k in ("language", "text")) else 1,
            1 if any(k in item[0].lower() for k in ("visual", "vision")) else 0,
            -len(item[1]),
        )
    )
    if candidates:
        return candidates[0][1], candidates[0][0]
    raise RuntimeError("Could not resolve language-model decoder layers.")


def find_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise AttributeError(
        f"Could not find self-attention inside {type(layer).__name__}"
    )


# =============================================================================
# Visual-token handling
# =============================================================================

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
    # Qwen2.5-VL: prefer modality IDs when available.
    mm_type_ids = batch.get("mm_token_type_ids")
    if torch.is_tensor(mm_type_ids) and mm_type_ids.ndim == 2:
        direct = (
            torch.nonzero(mm_type_ids[0] == 1, as_tuple=False)
            .flatten()
            .tolist()
        )
        if direct:
            return [int(x) for x in direct]

    token_type_ids = batch.get("token_type_ids")
    if torch.is_tensor(token_type_ids) and token_type_ids.ndim == 2:
        unique = set(
            int(x)
            for x in token_type_ids[0].detach().cpu().tolist()
        )
        if 1 in unique:
            direct = (
                torch.nonzero(token_type_ids[0] == 1, as_tuple=False)
                .flatten()
                .tolist()
            )
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

    indices = [
        i
        for i, token_id in enumerate(input_ids)
        if int(token_id) in token_ids
    ]
    if indices:
        return indices

    # Fallback: positions between explicit vision boundary tokens.
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
        f"Candidate image token IDs={sorted(token_ids)}"
    )


def visual_coordinates(
    model: Any,
    batch: Dict[str, Any],
    n_visual_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Return normalized (x,y) coordinates in flattened visual-token order.

    Qwen uses image_grid_thw and spatial_merge_size when available.
    LLaVA normally falls through to a square token grid.
    """
    grid = batch.get("image_grid_thw")
    if torch.is_tensor(grid) and grid.numel() >= 3:
        values = grid.detach().cpu().reshape(-1, 3)[0].tolist()
        t, h, w = [int(x) for x in values]
        vision_config = getattr(
            getattr(model, "config", None),
            "vision_config",
            None,
        )
        merge = int(
            getattr(vision_config, "spatial_merge_size", 1) or 1
        )
        temporal_merge = int(
            getattr(vision_config, "temporal_patch_size", 1) or 1
        )
        candidates = []
        for tm in (1, temporal_merge):
            tt = max(1, t // max(1, tm))
            hh = max(1, h // max(1, merge))
            ww = max(1, w // max(1, merge))
            candidates.append((tt, hh, ww))

        # Some processor versions already return post-merge dimensions.
        candidates.append((max(1, t), max(1, h), max(1, w)))

        for tt, hh, ww in candidates:
            if tt * hh * ww != n_visual_tokens:
                continue
            ys = torch.linspace(0.0, 1.0, hh, device=device)
            xs = torch.linspace(0.0, 1.0, ww, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xy = torch.stack(
                [xx.reshape(-1), yy.reshape(-1)],
                dim=-1,
            )
            if tt > 1:
                xy = xy.repeat(tt, 1)
            return xy

    side = int(round(math.sqrt(n_visual_tokens)))
    if side * side == n_visual_tokens:
        ys = torch.linspace(0.0, 1.0, side, device=device)
        xs = torch.linspace(0.0, 1.0, side, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack(
            [xx.reshape(-1), yy.reshape(-1)],
            dim=-1,
        )

    return None


# =============================================================================
# Attention / centroid scan
# =============================================================================

def generation_steps(output: Any, name: str) -> Tuple[Any, ...]:
    candidates = [
        getattr(output, name, None),
        getattr(output, f"decoder_{name}", None),
    ]
    for value in candidates:
        if isinstance(value, (tuple, list)) and len(value) > 0:
            return tuple(value)
    raise RuntimeError(
        f"Generation did not return {name}. "
        "Use eager attention and a compatible transformers version."
    )


def step_layers(step_value: Any) -> Tuple[Any, ...]:
    if isinstance(step_value, (tuple, list)):
        return tuple(step_value)
    raise RuntimeError(
        f"Expected tuple/list of per-layer values, got {type(step_value)}"
    )


def normalize_attention_tensor(
    value: torch.Tensor,
    expected_query_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Return [heads, query, key] for batch-size-one attention.
    """
    if not torch.is_tensor(value):
        raise TypeError(f"Attention is not tensor: {type(value)}")
    tensor = value
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise RuntimeError(
            f"Expected attention rank 3/4, got {tuple(value.shape)}"
        )

    if (
        expected_query_length is not None
        and tensor.shape[1] != expected_query_length
        and tensor.shape[0] == expected_query_length
    ):
        tensor = tensor.permute(1, 0, 2)

    return tensor.float()


def prompt_attention_layers(
    model: Any,
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    Run one natural greedy decoding step only to retrieve prefill attentions.
    """
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    attention_steps = generation_steps(generated, "attentions")
    prompt_attentions = step_layers(attention_steps[0])
    del generated
    return prompt_attentions


def object_centroids_all_layers(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    subject: str,
    reference: str,
    n_layers: int,
) -> np.ndarray:
    """
    Return:
        centroids [layer, head, 2 objects, 2 xy]

    Object order always follows the current prompt role order:
        [subject, reference]
    """
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    input_length = len(input_ids)

    subject_span, reference_span = locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])

    visual_indices = resolve_visual_indices(
        model,
        processor,
        batch,
        input_ids,
    )
    coords = visual_coordinates(
        model,
        batch,
        len(visual_indices),
        batch["input_ids"].device,
    )
    if coords is None:
        raise RuntimeError(
            f"Could not construct coordinates for "
            f"{len(visual_indices)} visual tokens"
        )

    attentions = prompt_attention_layers(model, batch)
    if len(attentions) < n_layers:
        raise RuntimeError(
            f"Attention layers returned={len(attentions)} "
            f"< decoder layers={n_layers}"
        )

    visual_index_tensor = torch.as_tensor(
        visual_indices,
        device=batch["input_ids"].device,
        dtype=torch.long,
    )

    all_centers: List[np.ndarray] = []
    n_heads_ref: Optional[int] = None

    for layer in range(n_layers):
        attn = normalize_attention_tensor(
            attentions[layer],
            expected_query_length=input_length,
        )
        n_heads = int(attn.shape[0])
        if n_heads_ref is None:
            n_heads_ref = n_heads
        elif n_heads != n_heads_ref:
            raise RuntimeError(
                f"Head count changed across layers: "
                f"{n_heads_ref} -> {n_heads} at layer {layer}"
            )

        # [heads, 2, key]
        rows = attn[:, [subject_index, reference_index], :]
        # [heads, 2, visual]
        visual = rows.index_select(-1, visual_index_tensor)
        visual_mass = visual.sum(dim=-1, keepdim=True)
        weights = visual / visual_mass.clamp_min(1e-12)

        # [heads, 2, 2]
        centers = torch.einsum(
            "hqv,vd->hqd",
            weights,
            coords.float(),
        )
        all_centers.append(
            centers.detach().cpu().numpy().astype(np.float32)
        )

    del attentions
    return np.stack(all_centers, axis=0)


def relation_codes_from_centroids(
    centroids: np.ndarray,
) -> np.ndarray:
    """
    centroids [..., 2 objects, 2 coords]
    returns int code in {0:left,1:right,2:above,3:below}
    """
    dx = centroids[..., 0, 0] - centroids[..., 1, 0]
    dy = centroids[..., 0, 1] - centroids[..., 1, 1]
    ax = np.abs(dx)
    ay = np.abs(dy)
    horizontal = ax >= ay
    return np.where(
        horizontal,
        np.where(dx < 0.0, 0, 1),
        np.where(dy < 0.0, 2, 3),
    ).astype(np.int8)


def scan_all_heads(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    records: Sequence[Record],
    prompt_rows: Mapping[int, Dict[str, Any]],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    scan_path = output_dir / "centroid_scan.npz"
    top_path = output_dir / "top_heads.json"
    error_path = output_dir / "scan_errors.jsonl"

    if scan_path.exists():
        with np.load(scan_path, allow_pickle=False) as z:
            accuracy = z["attention_average_accuracy"].astype(np.float64)
            correct = z["correct_count"].astype(np.int64)
            valid = z["valid_count"].astype(np.int64)
            layers = z["layer_indices"].astype(np.int64)
        return {
            "accuracy": accuracy,
            "correct": correct,
            "valid": valid,
            "layers": layers,
            "scan_path": str(scan_path),
            "top_path": str(top_path),
        }

    n_layers = len(decoder_layers)
    correct: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None

    good_samples = 0

    for record in tqdm(
        records,
        desc="phase1-centroid-scan",
    ):
        image = None
        batch_ab = None
        batch_ba = None
        try:
            sid = int(record.sid)
            row = prompt_rows[sid]

            subject = str(row["subject"])
            reference = str(row["reference"])
            question_ab = str(row["question_text"])
            gt = normalize_relation(row["answer_raw"])
            if gt not in RELATION_TO_INDEX:
                raise ValueError(f"Unsupported GT={gt!r} for sid={sid}")

            image = Image.open(record.image_path).convert("RGB")

            batch_ab = make_batch(
                processor,
                image,
                question_ab,
                device,
            )
            question_ba = build_swapped_question(subject, reference)
            batch_ba = make_batch(
                processor,
                image,
                question_ba,
                device,
            )

            # Q_AB role order: [A,B]
            c_ab = object_centroids_all_layers(
                model=model,
                processor=processor,
                batch=batch_ab,
                subject=subject,
                reference=reference,
                n_layers=n_layers,
            )

            # Q_BA role order: [B,A]
            c_ba_role = object_centroids_all_layers(
                model=model,
                processor=processor,
                batch=batch_ba,
                subject=reference,
                reference=subject,
                n_layers=n_layers,
            )

            if c_ab.shape != c_ba_role.shape:
                raise RuntimeError(
                    f"Centroid shape mismatch: "
                    f"{c_ab.shape} vs {c_ba_role.shape}"
                )

            # Semantic alignment:
            # Q_BA [B,A] -> [A,B]
            c_ba_aligned = c_ba_role[:, :, [1, 0], :]

            # Same metric as centroid V4 top-head ranking.
            c_avg = 0.5 * (c_ab + c_ba_aligned)
            pred = relation_codes_from_centroids(c_avg)
            gt_code = RELATION_TO_INDEX[gt]

            if correct is None:
                correct = np.zeros(pred.shape, dtype=np.int64)
                valid = np.zeros(pred.shape, dtype=np.int64)

            if pred.shape != correct.shape:
                raise RuntimeError(
                    f"Head shape changed: {pred.shape} vs {correct.shape}"
                )

            correct += (pred == gt_code).astype(np.int64)
            assert valid is not None
            valid += 1
            good_samples += 1

        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    "sid": int(record.sid),
                    "phase": "scan",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-14:],
                },
            )
            tqdm.write(
                f"[scan error] sid={record.sid}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            for obj in (batch_ab, batch_ba, image):
                if obj is not None:
                    del obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if correct is None or valid is None or good_samples == 0:
        raise RuntimeError(
            "No centroid-scan samples completed. Inspect scan_errors.jsonl"
        )

    accuracy = np.divide(
        correct,
        np.maximum(valid, 1),
        dtype=np.float64,
    )
    layers = np.arange(n_layers, dtype=np.int32)

    np.savez_compressed(
        scan_path,
        layer_indices=layers,
        attention_average_accuracy=accuracy.astype(np.float32),
        correct_count=correct,
        valid_count=valid,
        n_good_samples=np.asarray(good_samples, dtype=np.int64),
    )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "valid": valid,
        "layers": layers,
        "scan_path": str(scan_path),
        "top_path": str(top_path),
    }


def rank_heads(
    scan: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    accuracy = np.asarray(scan["accuracy"])
    layers = np.asarray(scan["layers"])
    if accuracy.ndim != 2:
        raise RuntimeError(
            f"Expected [layer,head] accuracy, got {accuracy.shape}"
        )

    rows: List[Dict[str, Any]] = []
    for layer_pos, layer in enumerate(layers.tolist()):
        for head in range(accuracy.shape[1]):
            score = float(accuracy[layer_pos, head])
            if not np.isfinite(score):
                continue
            rows.append({
                "layer": int(layer),
                "head": int(head),
                "centroid_accuracy": score,
                "correct_count": int(scan["correct"][layer_pos, head]),
                "valid_count": int(scan["valid"][layer_pos, head]),
            })

    rows.sort(
        key=lambda x: (
            x["centroid_accuracy"],
            x["correct_count"],
            -x["layer"],
            -x["head"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    return rows[:top_k]


# =============================================================================
# Relation logits / generation
# =============================================================================

def relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    unk = getattr(tokenizer, "unk_token_id", None)

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
            if unk is not None and token_id == int(unk):
                continue
            token_ids.append(token_id)

        token_ids = list(dict.fromkeys(token_ids))
        if not token_ids:
            raise RuntimeError(
                f"No one-token generation variant for {relation!r}"
            )
        result[relation] = token_ids

    return result


def relation_score_map(
    score_vector: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for relation in RELATIONS:
        ids = torch.as_tensor(
            list(token_map[relation]),
            device=score_vector.device,
            dtype=torch.long,
        )
        result[relation] = float(
            score_vector.index_select(0, ids).float().max().item()
        )
    return result


def decode_new_tokens(
    processor: Any,
    sequences: torch.Tensor,
    input_length: int,
) -> str:
    generated_ids = sequences[0, input_length:]
    return processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def generate_with_scores(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    max_new_tokens: int,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    input_length = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = generated.sequences
    text = decode_new_tokens(
        processor,
        sequences,
        input_length,
    )
    prediction = normalize_relation(text)

    scores = tuple(getattr(generated, "scores", ()) or ())
    if not scores:
        raise RuntimeError("Generation returned no next-token scores")

    first_scores = relation_score_map(
        scores[0][0],
        relation_token_map,
    )

    out = {
        "text": text,
        "prediction": prediction,
        "relation_scores": first_scores,
        "first_token_id": int(sequences[0, input_length].item()),
    }
    del generated, sequences
    return out


# =============================================================================
# Single-head pre-W_O A/B swap intervention
# =============================================================================

def resolve_head_geometry(attention: Any) -> Tuple[int, int, int]:
    """
    Returns:
        n_heads, head_dim, o_proj_in_features

    o_proj input is expected to be concatenated query-head output:
        hidden = n_heads * head_dim
    """
    o_proj = getattr(attention, "o_proj", None)
    if o_proj is None:
        raise AttributeError(
            f"{type(attention).__name__} exposes no o_proj"
        )

    in_features = getattr(o_proj, "in_features", None)
    if in_features is None:
        weight = getattr(o_proj, "weight", None)
        if weight is None or weight.ndim != 2:
            raise RuntimeError("Could not infer o_proj input width")
        in_features = int(weight.shape[1])
    else:
        in_features = int(in_features)

    head_dim = getattr(attention, "head_dim", None)
    if head_dim is not None:
        head_dim = int(head_dim)
        if head_dim > 0 and in_features % head_dim == 0:
            n_heads = in_features // head_dim
            return int(n_heads), int(head_dim), int(in_features)

    for name in ("num_heads", "num_attention_heads"):
        n_heads = getattr(attention, name, None)
        if n_heads is not None:
            n_heads = int(n_heads)
            if n_heads > 0 and in_features % n_heads == 0:
                return (
                    n_heads,
                    in_features // n_heads,
                    in_features,
                )

    config = getattr(attention, "config", None)
    if config is not None:
        for name in ("num_attention_heads", "num_heads"):
            n_heads = getattr(config, name, None)
            if n_heads is not None:
                n_heads = int(n_heads)
                if n_heads > 0 and in_features % n_heads == 0:
                    return (
                        n_heads,
                        in_features // n_heads,
                        in_features,
                    )

    raise RuntimeError(
        f"Could not infer head geometry for "
        f"{type(attention).__name__}; o_proj_in={in_features}"
    )


class SingleHeadObjectSwap:
    """
    Swap ONE attention head's pre-W_O output at the two object-token
    positions during PREFILL only.

    On Q_BA:
        subject_index   = token(B)
        reference_index = token(A)

    Patch:
        [B_slice, A_slice] -> [A_slice, B_slice]
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        layer: int,
        head: int,
        subject_index: int,
        reference_index: int,
        prompt_length: int,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer = int(layer)
        self.head = int(head)
        self.subject_index = int(subject_index)
        self.reference_index = int(reference_index)
        self.prompt_length = int(prompt_length)

        self.handle = None
        self.patch_events = 0
        self.n_heads: Optional[int] = None
        self.head_dim: Optional[int] = None

    def __enter__(self) -> "SingleHeadObjectSwap":
        if not (0 <= self.layer < len(self.decoder_layers)):
            raise ValueError(
                f"Layer {self.layer} outside [0,{len(self.decoder_layers)-1}]"
            )

        attention = find_self_attention(
            self.decoder_layers[self.layer]
        )
        o_proj = getattr(attention, "o_proj", None)
        if o_proj is None:
            raise AttributeError(
                f"L{self.layer} attention exposes no o_proj"
            )

        n_heads, head_dim, hidden = resolve_head_geometry(attention)
        self.n_heads = n_heads
        self.head_dim = head_dim

        if not (0 <= self.head < n_heads):
            raise ValueError(
                f"L{self.layer} H{self.head} outside [0,{n_heads-1}]"
            )

        start = self.head * head_dim
        stop = (self.head + 1) * head_dim

        def pre_hook(_module: Any, inputs: Tuple[Any, ...]):
            if not inputs:
                return None

            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim != 3:
                return None

            # Only patch full-prompt prefill. Decode steps have q_len=1.
            if int(x.shape[1]) != self.prompt_length:
                return None

            if int(x.shape[-1]) != hidden:
                raise RuntimeError(
                    f"o_proj input width changed: "
                    f"{x.shape[-1]} vs expected {hidden}"
                )

            if not (
                0 <= self.subject_index < x.shape[1]
                and 0 <= self.reference_index < x.shape[1]
            ):
                raise IndexError(
                    f"Object indices outside prefill length "
                    f"{x.shape[1]}: subject={self.subject_index}, "
                    f"reference={self.reference_index}"
                )

            # Clone to avoid mutating a tensor that may still be referenced
            # by the attention implementation.
            y = x.clone()

            subject_slice = y[
                :,
                self.subject_index,
                start:stop,
            ].clone()
            reference_slice = y[
                :,
                self.reference_index,
                start:stop,
            ].clone()

            y[
                :,
                self.subject_index,
                start:stop,
            ] = reference_slice
            y[
                :,
                self.reference_index,
                start:stop,
            ] = subject_slice

            self.patch_events += 1
            return (y,) + tuple(inputs[1:])

        self.handle = o_proj.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        tb: Any,
    ) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Intervention evaluation
# =============================================================================

def toward_original_margin(
    scores: Mapping[str, float],
    original_gt: str,
    swapped_gt: str,
) -> float:
    return float(scores[original_gt] - scores[swapped_gt])


def evaluate_interventions(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    records: Sequence[Record],
    prompt_rows: Mapping[int, Dict[str, Any]],
    top_heads: Sequence[Dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    rows_path = output_dir / "interventions.jsonl"
    errors_path = output_dir / "intervention_errors.jsonl"

    previous = read_jsonl(rows_path)
    done_sids = {int(row["sid"]) for row in previous}
    relation_token_map = relation_token_variants(processor.tokenizer)

    for record in tqdm(
        records,
        desc="phase2-single-head-swap",
    ):
        sid = int(record.sid)
        if sid in done_sids:
            continue

        image = None
        batch = None
        try:
            row = prompt_rows[sid]
            subject_a = str(row["subject"])
            reference_b = str(row["reference"])
            original_gt = normalize_relation(row["answer_raw"])
            if original_gt not in RELATIONS:
                raise ValueError(
                    f"Unsupported original GT={original_gt!r}"
                )

            swapped_gt = invert_relation(original_gt)
            assert swapped_gt is not None

            # Q_BA: subject=B, reference=A
            question_ba = build_swapped_question(
                subject_a,
                reference_b,
            )

            image = Image.open(record.image_path).convert("RGB")
            batch = make_batch(
                processor,
                image,
                question_ba,
                device,
            )

            input_ids = (
                batch["input_ids"][0]
                .detach()
                .cpu()
                .tolist()
            )

            # Current Q_BA role order is [B,A].
            span_b, span_a = locate_object_spans(
                processor.tokenizer,
                input_ids,
                reference_b,   # subject in Q_BA = B
                subject_a,     # reference in Q_BA = A
            )
            index_b = int(span_b[1])
            index_a = int(span_a[1])
            prompt_length = len(input_ids)

            baseline = generate_with_scores(
                model=model,
                processor=processor,
                batch=batch,
                max_new_tokens=max_new_tokens,
                relation_token_map=relation_token_map,
            )
            baseline_prediction = baseline["prediction"]
            baseline_correct = (
                baseline_prediction is not None
                and baseline_prediction == swapped_gt
            )
            base_orig_margin = toward_original_margin(
                baseline["relation_scores"],
                original_gt,
                swapped_gt,
            )

            head_results: List[Dict[str, Any]] = []

            for head_info in top_heads:
                layer = int(head_info["layer"])
                head = int(head_info["head"])

                with SingleHeadObjectSwap(
                    decoder_layers=decoder_layers,
                    layer=layer,
                    head=head,
                    subject_index=index_b,
                    reference_index=index_a,
                    prompt_length=prompt_length,
                ) as patch:
                    patched = generate_with_scores(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=max_new_tokens,
                        relation_token_map=relation_token_map,
                    )

                if patch.patch_events == 0:
                    raise RuntimeError(
                        f"L{layer}H{head} registered zero patch events"
                    )

                pred = patched["prediction"]
                patched_correct = (
                    pred is not None and pred == swapped_gt
                )
                prediction_changed = (
                    baseline_prediction is not None
                    and pred is not None
                    and pred != baseline_prediction
                )

                # Strongest expected result:
                # baseline answers correct inverse relation for Q_BA,
                # but after [B,A]->[A,B] head swap it answers original Q_AB GT.
                strict_semantic_reversal = bool(
                    baseline_prediction == swapped_gt
                    and pred == original_gt
                )

                opposite_transition = bool(
                    baseline_prediction in RELATIONS
                    and pred
                    == INVERSE_RELATION.get(baseline_prediction)
                )

                patched_orig_margin = toward_original_margin(
                    patched["relation_scores"],
                    original_gt,
                    swapped_gt,
                )

                head_results.append({
                    "rank": int(head_info["rank"]),
                    "layer": layer,
                    "head": head,
                    "centroid_accuracy": float(
                        head_info["centroid_accuracy"]
                    ),
                    "prediction": pred,
                    "generated_text": patched["text"],
                    "first_token_id": patched["first_token_id"],
                    "relation_scores": patched["relation_scores"],
                    "correct_for_Q_BA": bool(patched_correct),
                    "prediction_changed": bool(prediction_changed),
                    "strict_semantic_reversal": strict_semantic_reversal,
                    "opposite_transition": opposite_transition,
                    "fixed": bool(
                        (not baseline_correct) and patched_correct
                    ),
                    "broken": bool(
                        baseline_correct and (not patched_correct)
                    ),
                    "toward_original_margin": patched_orig_margin,
                    "delta_toward_original_margin": float(
                        patched_orig_margin - base_orig_margin
                    ),
                    "patch_events": int(patch.patch_events),
                    "n_heads_at_layer": patch.n_heads,
                    "head_dim": patch.head_dim,
                })

            sample_row = {
                "sid": sid,
                "image_id": record.image_id,
                "subject_A": subject_a,
                "reference_B": reference_b,
                "original_Q_AB_gt": original_gt,
                "swapped_Q_BA_gt": swapped_gt,
                "swapped_question": question_ba,
                "swapped_role_order_before_patch": ["B", "A"],
                "swapped_role_order_after_patch_for_target_head": ["A", "B"],
                "object_token_indices_Q_BA": {
                    "B_subject_index": index_b,
                    "A_reference_index": index_a,
                },
                "baseline_Q_BA": {
                    "prediction": baseline_prediction,
                    "correct": bool(baseline_correct),
                    "generated_text": baseline["text"],
                    "first_token_id": baseline["first_token_id"],
                    "relation_scores": baseline["relation_scores"],
                    "toward_original_margin": base_orig_margin,
                },
                "heads": head_results,
            }
            append_jsonl(rows_path, sample_row)

            changed_heads = sum(
                int(x["prediction_changed"])
                for x in head_results
            )
            reversal_heads = sum(
                int(x["strict_semantic_reversal"])
                for x in head_results
            )
            tqdm.write(
                f"sid={sid:4d} | "
                f"Q_BA base={baseline_prediction} gt={swapped_gt} | "
                f"changed_heads={changed_heads}/{len(head_results)} | "
                f"strict_reversal={reversal_heads}"
            )

        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "sid": sid,
                    "phase": "intervention",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-18:],
                },
            )
            tqdm.write(
                f"[intervention error] sid={sid}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if batch is not None:
                del batch
            if image is not None:
                del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return read_jsonl(rows_path)


def summarize_interventions(
    rows: Sequence[Dict[str, Any]],
    top_heads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    per_head: Dict[Tuple[int, int], List[Dict[str, Any]]] = {
        (int(h["layer"]), int(h["head"])): []
        for h in top_heads
    }

    baseline_valid = 0
    baseline_correct = 0

    for row in rows:
        base_pred = row["baseline_Q_BA"]["prediction"]
        if base_pred in RELATIONS:
            baseline_valid += 1
            baseline_correct += int(
                bool(row["baseline_Q_BA"]["correct"])
            )

        for result in row.get("heads", []):
            key = (int(result["layer"]), int(result["head"]))
            if key in per_head:
                per_head[key].append(result)

    head_summaries: List[Dict[str, Any]] = []

    lookup = {
        (int(h["layer"]), int(h["head"])): h
        for h in top_heads
    }

    for key, values in per_head.items():
        info = lookup[key]
        n = len(values)

        valid_pairs = [
            x
            for x in values
            if x["prediction"] in RELATIONS
        ]
        nv = len(valid_pairs)

        baseline_correct_subset = [
            x
            for x in valid_pairs
            if not x.get("_dummy", False)
        ]

        # strict reversal denominator: samples for which Q_BA baseline was
        # correct.  Reconstruct from each sample row below.
        strict_num = sum(
            int(x["strict_semantic_reversal"])
            for x in valid_pairs
        )

        # Count baseline-correct denominator from row association.
        bc_denom = 0
        for row in rows:
            if not bool(row["baseline_Q_BA"]["correct"]):
                continue
            match = next(
                (
                    x
                    for x in row.get("heads", [])
                    if int(x["layer"]) == key[0]
                    and int(x["head"]) == key[1]
                ),
                None,
            )
            if match is not None and match["prediction"] in RELATIONS:
                bc_denom += 1

        head_summaries.append({
            "rank": int(info["rank"]),
            "layer": key[0],
            "head": key[1],
            "centroid_accuracy": float(
                info["centroid_accuracy"]
            ),
            "centroid_correct_count": int(
                info["correct_count"]
            ),
            "centroid_valid_count": int(
                info["valid_count"]
            ),
            "n": n,
            "n_valid_generation": nv,
            "patched_Q_BA_accuracy": (
                float(np.mean([
                    bool(x["correct_for_Q_BA"])
                    for x in valid_pairs
                ]))
                if valid_pairs else None
            ),
            "prediction_changed_count": sum(
                int(x["prediction_changed"])
                for x in valid_pairs
            ),
            "prediction_changed_rate": (
                float(np.mean([
                    bool(x["prediction_changed"])
                    for x in valid_pairs
                ]))
                if valid_pairs else None
            ),
            "strict_semantic_reversal_count": strict_num,
            "strict_semantic_reversal_rate_given_baseline_correct": (
                strict_num / bc_denom
                if bc_denom else None
            ),
            "baseline_correct_denom_for_reversal": bc_denom,
            "opposite_transition_count": sum(
                int(x["opposite_transition"])
                for x in valid_pairs
            ),
            "opposite_transition_rate": (
                float(np.mean([
                    bool(x["opposite_transition"])
                    for x in valid_pairs
                ]))
                if valid_pairs else None
            ),
            "fixed": sum(
                int(x["fixed"])
                for x in valid_pairs
            ),
            "broken": sum(
                int(x["broken"])
                for x in valid_pairs
            ),
            "mean_delta_toward_original_margin": (
                float(np.mean([
                    float(x["delta_toward_original_margin"])
                    for x in valid_pairs
                ]))
                if valid_pairs else None
            ),
            "median_delta_toward_original_margin": (
                float(np.median([
                    float(x["delta_toward_original_margin"])
                    for x in valid_pairs
                ]))
                if valid_pairs else None
            ),
        })

    head_summaries.sort(key=lambda x: x["rank"])

    return {
        "n_sample_rows": len(rows),
        "baseline_Q_BA_valid": baseline_valid,
        "baseline_Q_BA_accuracy": (
            baseline_correct / baseline_valid
            if baseline_valid else None
        ),
        "heads": head_summaries,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 132)
    print("TOP-CENTROID SINGLE-HEAD [B,A] -> [A,B] OUTPUT-SWAP GENERATION")
    print("=" * 132)

    baseline = summary.get("baseline_Q_BA_accuracy")
    if baseline is not None:
        print(
            f"Q_BA baseline accuracy = {baseline:.4f} "
            f"(n={summary.get('baseline_Q_BA_valid')})"
        )

    print(
        "\n"
        "rank  head      cent_acc   patch_acc   changed   strict_rev*   "
        "opp_flip   mean_dMargin(original)"
    )
    print("-" * 132)

    for x in summary.get("heads", []):
        patch_acc = x["patched_Q_BA_accuracy"]
        changed = x["prediction_changed_rate"]
        strict = x["strict_semantic_reversal_rate_given_baseline_correct"]
        opp = x["opposite_transition_rate"]
        dm = x["mean_delta_toward_original_margin"]

        print(
            f"{x['rank']:>4d}  "
            f"L{x['layer']:02d}H{x['head']:02d}   "
            f"{x['centroid_accuracy']:.4f}     "
            f"{patch_acc if patch_acc is not None else float('nan'):.4f}     "
            f"{changed if changed is not None else float('nan'):.4f}     "
            f"{strict if strict is not None else float('nan'):.4f}       "
            f"{opp if opp is not None else float('nan'):.4f}     "
            f"{dm if dm is not None else float('nan'):+.4f}"
        )

    print(
        "\n* strict_rev: among samples where normal Q_BA is correct, "
        "patched Q_BA becomes the ORIGINAL Q_AB ground-truth relation."
    )


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_SPECS),
    )
    p.add_argument(
        "--data-root",
        default="data",
        help=(
            "COCO root containing coco_qa_two_obj.json and val2017/. "
            "Default: data"
        ),
    )
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug cap. Default uses all usable COCO-two records.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
    )
    p.add_argument(
        "--phase",
        choices=["all", "scan", "intervene"],
        default="all",
        help=(
            "all: scan then intervene; "
            "scan: only find Top-K; "
            "intervene: reuse existing centroid_scan.npz."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    return p.parse_args()


def load_model_and_processor(
    args: argparse.Namespace,
) -> Tuple[Any, Any, Sequence[Any], str]:
    spec = MODEL_SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no "
            f"{spec.model_class}. "
            "Use the same transformers environment that already runs "
            "your centroid V4 experiments."
        )

    print(f"Loading {args.model}: {spec.repo_id}")
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": "eager",
    }

    model = model_cls.from_pretrained(
        spec.repo_id,
        **load_kwargs,
    )
    model.eval()
    clear_sampling_defaults(model)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    configure_processor(model, processor)

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    return model, processor, decoder_layers, decoder_path


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    prompt_path = Path(args.prompt_jsonl)

    records, audit = load_coco_records(
        data_root,
        args.max_samples,
    )
    if not records:
        raise RuntimeError("No usable COCO-two records")

    prompt_rows = load_standard_prompts(prompt_path)
    missing = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Prompt file missing {len(missing)} record IDs; "
            f"first={missing[:10]}"
        )

    relation_counts = Counter(
        normalize_relation(prompt_rows[int(r.sid)]["answer_raw"])
        for r in records
    )
    print(
        f"COCO-two usable={len(records)} | "
        f"relations={dict(relation_counts)} | audit={len(audit)}"
    )

    model = processor = None
    started = time.time()

    try:
        model, processor, decoder_layers, decoder_path = (
            load_model_and_processor(args)
        )
        device = torch.device(args.device)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": MODEL_SPECS[args.model].repo_id,
            "transformers_version": transformers.__version__,
            "data_root": str(data_root),
            "prompt_jsonl": str(prompt_path),
            "n_records": len(records),
            "relation_counts": dict(relation_counts),
            "dataset_audit_count": len(audit),
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "phase": args.phase,
            "patch": {
                "question": "Q_BA",
                "location": "self_attn.o_proj forward-pre-hook",
                "representation": "pre-W_O concatenated per-head output",
                "source_role_order": ["B", "A"],
                "target_role_order": ["A", "B"],
                "prefill_only": True,
                "single_head_at_a_time": True,
            },
        }
        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Decoder path={decoder_path} | "
            f"layers={len(decoder_layers)}"
        )

        scan_path = output_dir / "centroid_scan.npz"

        if args.phase == "intervene" and not scan_path.exists():
            raise FileNotFoundError(
                f"--phase intervene requires existing {scan_path}"
            )

        scan = scan_all_heads(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            records=records,
            prompt_rows=prompt_rows,
            device=device,
            output_dir=output_dir,
        )

        top_heads = rank_heads(scan, args.top_k)
        (output_dir / "top_heads.json").write_text(
            json.dumps(top_heads, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nTop centroid heads:")
        for row in top_heads:
            print(
                f"  #{row['rank']:02d} "
                f"L{row['layer']:02d}H{row['head']:02d} | "
                f"acc={row['centroid_accuracy']:.4f} "
                f"({row['correct_count']}/{row['valid_count']})"
            )

        if args.phase == "scan":
            print(
                f"\nSaved scan: {output_dir / 'centroid_scan.npz'}"
            )
            print(
                f"Saved Top-{args.top_k}: "
                f"{output_dir / 'top_heads.json'}"
            )
            return

        rows = evaluate_interventions(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            records=records,
            prompt_rows=prompt_rows,
            top_heads=top_heads,
            device=device,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
        )

        summary = summarize_interventions(
            rows,
            top_heads,
        )
        summary["config"] = config
        summary["elapsed_minutes"] = (
            time.time() - started
        ) / 60.0

        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print_summary(summary)

        print(f"\nSaved:")
        print(f"  {output_dir / 'centroid_scan.npz'}")
        print(f"  {output_dir / 'top_heads.json'}")
        print(f"  {output_dir / 'interventions.jsonl'}")
        print(f"  {output_dir / 'summary.json'}")

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
