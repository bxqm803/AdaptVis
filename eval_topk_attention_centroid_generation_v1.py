#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-k attention-centroid analysis and guided generation.

This script consumes the complete Step 1 output produced by:

    analyze_coco_attention_flow_swap_step1_v1.py

It does not recompute attention maps. Instead, it selects a small set of
attention heads and reads their saved subject/reference visual centroids.

For every sample it computes:

1) Original top-k centroid ensemble.
2) Swap-aligned top-k centroid ensemble.
3) Original/swap averaged top-k centroid ensemble.
4) Normal frozen-model generation.
5) Optional centroid-guided normal autoregressive generation.

The central diagnostic is the relationship between centroid and generation:

- centroid correct, generation correct;
- centroid correct, generation wrong;
- centroid wrong, generation correct;
- centroid wrong, generation wrong.

Guidance uses the continuous top-k geometry only at the first generated answer
token. It does not replace the answer, modify model weights, train a probe, or
change the prompt. Ground truth is used only for reporting. When
--head-selection oracle_accuracy is used, GT-derived aggregate head accuracy
from Step 1 selects the heads, so that mode is an oracle diagnostic rather than
a deployable method. The default unsupervised selection uses only Step 1's
label-free stability score.

Required Step 1 files:
- aggregate_metrics.npz
- sample_arrays/<sid>.npz

Typical use:
- first run --only analyze to inspect top-k centroid quality;
- then run --only both to compare baseline and guided generation.
"""
from __future__ import annotations

import argparse
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


SCRIPT_VERSION = "topk-attention-centroid-generation-v1"

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




RELATION_TO_INDEX = {
    "left": 0,
    "right": 1,
    "above": 2,
    "below": 3,
}
INDEX_TO_RELATION = np.asarray(RELATIONS, dtype="<U8")


@dataclass
class TopKCentroidEvidence:
    sid: int
    selected_heads: List[Dict[str, Any]]
    source: str
    ensemble_mode: str
    weight_mode: str
    prediction: str
    delta_x: float
    delta_y: float
    subject_x: float
    subject_y: float
    reference_x: float
    reference_y: float
    axis_confidence: float
    head_agreement: float
    swap_stability: float
    mean_separation: float
    mean_visual_mass: float
    confidence: float
    geometry_scores: Dict[str, float]
    head_rows: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default=None,
        help=(
            "Standard question file. For coco_two, the default is "
            "prompts/COCO_QA_two_obj_with_answer_four_options.jsonl."
        ),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
        help="Generation backend. Attention is loaded from Step 1 arrays.",
    )
    p.add_argument(
        "--step1-dir",
        required=True,
        help=(
            "Step 1 output directory containing aggregate_metrics.npz and "
            "sample_arrays/."
        ),
    )
    p.add_argument(
        "--head-selection",
        default="unsupervised",
        choices=["unsupervised", "oracle_accuracy", "manual"],
        help=(
            "unsupervised: rank by label-free Step 1 score; "
            "oracle_accuracy: rank by GT-derived centroid accuracy, diagnostic only; "
            "manual: use --manual-heads."
        ),
    )
    p.add_argument(
        "--manual-heads",
        default="",
        help="Comma-separated zero-based layer:head pairs, e.g. 13:6,17:10.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--centroid-source",
        default="average",
        choices=["original", "swap_aligned", "average"],
        help="Centroid source used for analysis and guidance.",
    )
    p.add_argument(
        "--ensemble-mode",
        default="weighted_centroid",
        choices=["weighted_centroid", "mean_centroid", "weighted_vote"],
    )
    p.add_argument(
        "--weight-mode",
        default="global_sample",
        choices=["uniform", "global", "sample", "global_sample"],
        help=(
            "How selected heads are weighted. global uses the aggregate ranking "
            "score; sample uses per-sample stability/separation/visual mass."
        ),
    )
    p.add_argument(
        "--confidence-mode",
        default="axis_agreement_swap",
        choices=[
            "none",
            "axis",
            "agreement",
            "axis_agreement",
            "axis_agreement_swap",
            "combined",
        ],
        help="Label-free confidence used to gate and scale guidance.",
    )
    p.add_argument(
        "--guide-strengths",
        default="1.0,2.0,4.0",
        help="Comma-separated additive logit strengths. Use 0 for a no-op control.",
    )
    p.add_argument(
        "--min-guide-confidence",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--max-lm-margin",
        type=float,
        default=float("inf"),
        help=(
            "Guide only when the baseline first-token relation logit margin is "
            "at most this value. Default inf disables this gate."
        ),
    )
    p.add_argument(
        "--guide-mode",
        default="soft",
        choices=["soft", "constrained"],
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument(
        "--only",
        default="both",
        choices=["analyze", "generation", "both"],
        help=(
            "analyze: read Step 1 arrays only; generation: load saved centroid "
            "analysis and run generation; both: analyze then run generation."
        ),
    )
    p.add_argument(
        "--analysis-jsonl",
        default=None,
        help=(
            "Existing centroid analysis JSONL used by --only generation. "
            "Defaults to <output-dir>/centroid_analysis.jsonl."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()
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




def parse_float_list(value: str) -> List[float]:
    values: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if number < 0:
            raise ValueError("--guide-strengths values must be nonnegative")
        if number not in values:
            values.append(number)
    if not values:
        raise ValueError("--guide-strengths resolved to an empty list")
    return values


def parse_manual_heads(value: str) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)\s*[:/]\s*(\d+)", item)
        if not match:
            raise ValueError(
                f"Invalid manual head {item!r}; expected layer:head"
            )
        pair = (int(match.group(1)), int(match.group(2)))
        if pair not in result:
            result.append(pair)
    return result


def load_step1_head_table(step1_dir: Path) -> Dict[str, Any]:
    path = step1_dir / "aggregate_metrics.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 1 aggregate: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = [
            "layer_indices",
            "unsupervised_head_score",
            "attention_average_accuracy",
        ]
        missing = [name for name in required if name not in z.files]
        if missing:
            raise RuntimeError(
                f"{path} is missing {missing}; available={z.files}"
            )
        layers = z["layer_indices"].astype(np.int64)
        unsupervised = z["unsupervised_head_score"].astype(np.float64)
        oracle = z["attention_average_accuracy"].astype(np.float64)
    if unsupervised.ndim != 2 or oracle.shape != unsupervised.shape:
        raise RuntimeError(
            "Expected Step 1 scores with shape [selected_layer, head]"
        )
    if len(layers) != unsupervised.shape[0]:
        raise RuntimeError("layer_indices does not match head-score arrays")
    return {
        "path": str(path),
        "layers": layers,
        "unsupervised": unsupervised,
        "oracle": oracle,
        "n_heads": int(unsupervised.shape[1]),
    }


def select_heads(
    table: Dict[str, Any],
    mode: str,
    top_k: int,
    manual_heads: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    layers = table["layers"]
    layer_to_pos = {
        int(layer): position
        for position, layer in enumerate(layers.tolist())
    }
    unsupervised = table["unsupervised"]
    oracle = table["oracle"]

    if mode == "manual":
        if not manual_heads:
            raise ValueError(
                "--head-selection manual requires --manual-heads"
            )
        pairs = list(manual_heads)
    else:
        score = unsupervised if mode == "unsupervised" else oracle
        candidates: List[Tuple[float, int, int]] = []
        for layer_pos, layer in enumerate(layers.tolist()):
            for head in range(score.shape[1]):
                value = float(score[layer_pos, head])
                if np.isfinite(value):
                    candidates.append((value, int(layer), int(head)))
        candidates.sort(reverse=True)
        pairs = [
            (layer, head)
            for _, layer, head in candidates[:top_k]
        ]

    if len(pairs) < top_k:
        raise RuntimeError(
            f"Selected only {len(pairs)} heads, but top-k={top_k}"
        )
    pairs = pairs[:top_k]

    rows: List[Dict[str, Any]] = []
    for layer, head in pairs:
        if layer not in layer_to_pos:
            raise ValueError(
                f"Layer {layer} is not present in Step 1 layer_indices"
            )
        layer_pos = layer_to_pos[layer]
        if not (0 <= head < table["n_heads"]):
            raise ValueError(
                f"Head {head} outside [0, {table['n_heads'] - 1}]"
            )
        global_score = (
            float(unsupervised[layer_pos, head])
            if mode != "oracle_accuracy"
            else float(oracle[layer_pos, head])
        )
        rows.append({
            "layer": int(layer),
            "head": int(head),
            "layer_position": int(layer_pos),
            "selection_score": global_score,
            "unsupervised_score": float(
                unsupervised[layer_pos, head]
            ),
            "oracle_centroid_accuracy": float(
                oracle[layer_pos, head]
            ),
        })
    return rows


def relation_from_delta(dx: float, dy: float) -> Tuple[str, float]:
    magnitude = abs(dx) + abs(dy)
    axis_confidence = (
        abs(abs(dx) - abs(dy)) / magnitude
        if magnitude > 1e-12 else 0.0
    )
    if abs(dx) >= abs(dy):
        return ("left" if dx < 0 else "right"), axis_confidence
    return ("above" if dy < 0 else "below"), axis_confidence


def geometry_scores_from_delta(
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw = np.asarray([-dx, dx, -dy, dy], dtype=np.float64)
    scale = float(np.max(np.abs(raw)))
    normalized = (
        raw / scale
        if scale > 1e-12
        else np.zeros(4, dtype=np.float64)
    )
    return normalized, {
        relation: float(normalized[index])
        for index, relation in enumerate(RELATIONS)
    }


def evidence_confidence(
    *,
    mode: str,
    axis: float,
    agreement: float,
    swap: float,
    separation: float,
    visual_mass: float,
) -> float:
    axis = float(np.clip(axis, 0.0, 1.0))
    agreement = float(np.clip(agreement, 0.0, 1.0))
    swap = float(np.clip(swap, 0.0, 1.0))
    separation = float(np.clip(separation, 0.0, 1.0))
    visual_mass = float(np.clip(visual_mass, 0.0, 1.0))

    if mode == "none":
        return 1.0
    if mode == "axis":
        return axis
    if mode == "agreement":
        return agreement
    if mode == "axis_agreement":
        return axis * agreement
    if mode == "axis_agreement_swap":
        return axis * agreement * swap
    if mode == "combined":
        return (
            axis
            * agreement
            * swap
            * math.sqrt(max(0.0, separation * visual_mass))
        )
    raise ValueError(f"Unsupported confidence mode: {mode}")


def load_topk_evidence(
    *,
    sample_path: Path,
    selected_heads: Sequence[Dict[str, Any]],
    centroid_source: str,
    ensemble_mode: str,
    weight_mode: str,
    confidence_mode: str,
) -> TopKCentroidEvidence:
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Missing Step 1 sample array: {sample_path}"
        )

    with np.load(sample_path, allow_pickle=False) as z:
        required = [
            "sid",
            "layer_indices",
            "original_object_centroids",
            "swapped_object_centroids_role_order",
            "same_object_map_cosine",
            "original_object_separation",
            "swapped_object_separation",
            "original_prompt_visual_mass",
            "swapped_prompt_visual_mass",
        ]
        missing = [name for name in required if name not in z.files]
        if missing:
            raise RuntimeError(
                f"{sample_path} is missing arrays {missing}"
            )

        sid = int(z["sid"].item())
        layers = z["layer_indices"].astype(np.int64)
        layer_to_pos = {
            int(layer): position
            for position, layer in enumerate(layers.tolist())
        }

        original = z[
            "original_object_centroids"
        ].astype(np.float64)
        swapped_role = z[
            "swapped_object_centroids_role_order"
        ].astype(np.float64)
        swapped_aligned = swapped_role[:, :, [1, 0], :]
        average = 0.5 * (original + swapped_aligned)

        map_cosine = z[
            "same_object_map_cosine"
        ].astype(np.float64)
        original_separation = z[
            "original_object_separation"
        ].astype(np.float64)
        swapped_separation = z[
            "swapped_object_separation"
        ].astype(np.float64)
        original_visual_mass = z[
            "original_prompt_visual_mass"
        ].astype(np.float64)
        swapped_visual_mass = z[
            "swapped_prompt_visual_mass"
        ].astype(np.float64)

    source_array = {
        "original": original,
        "swap_aligned": swapped_aligned,
        "average": average,
    }[centroid_source]

    head_rows: List[Dict[str, Any]] = []
    centroid_stack: List[np.ndarray] = []
    weights: List[float] = []
    local_qualities: List[float] = []
    stabilities: List[float] = []
    separations: List[float] = []
    visual_masses: List[float] = []

    for selected in selected_heads:
        layer = int(selected["layer"])
        head = int(selected["head"])
        if layer not in layer_to_pos:
            raise RuntimeError(
                f"Layer {layer} missing from sample {sample_path}"
            )
        layer_pos = layer_to_pos[layer]

        centroids = source_array[layer_pos, head]
        stability = float(np.clip(
            np.mean(map_cosine[layer_pos, head]),
            0.0,
            1.0,
        ))
        if centroid_source == "original":
            separation = float(
                np.clip(original_separation[layer_pos, head], 0.0, 1.0)
            )
            visual_mass = float(np.clip(
                0.5 * (
                    original_visual_mass[layer_pos, head, 0]
                    + original_visual_mass[layer_pos, head, 1]
                ),
                0.0,
                1.0,
            ))
        elif centroid_source == "swap_aligned":
            separation = float(
                np.clip(swapped_separation[layer_pos, head], 0.0, 1.0)
            )
            visual_mass = float(np.clip(
                0.5 * (
                    swapped_visual_mass[layer_pos, head, 0]
                    + swapped_visual_mass[layer_pos, head, 1]
                ),
                0.0,
                1.0,
            ))
        else:
            separation = float(np.clip(
                math.sqrt(max(
                    0.0,
                    float(original_separation[layer_pos, head])
                    * float(swapped_separation[layer_pos, head]),
                )),
                0.0,
                1.0,
            ))
            visual_mass = float(np.clip(
                0.25 * (
                    original_visual_mass[layer_pos, head, 0]
                    + original_visual_mass[layer_pos, head, 1]
                    + swapped_visual_mass[layer_pos, head, 0]
                    + swapped_visual_mass[layer_pos, head, 1]
                ),
                0.0,
                1.0,
            ))

        local_quality = (
            stability
            * math.sqrt(max(0.0, separation * visual_mass))
        )
        global_quality = max(
            float(selected["selection_score"]),
            0.0,
        )

        if weight_mode == "uniform":
            weight = 1.0
        elif weight_mode == "global":
            weight = global_quality
        elif weight_mode == "sample":
            weight = local_quality
        elif weight_mode == "global_sample":
            weight = global_quality * local_quality
        else:
            raise ValueError(
                f"Unsupported weight mode: {weight_mode}"
            )
        weight = max(float(weight), 1e-8)

        dx = float(centroids[0, 0] - centroids[1, 0])
        dy = float(centroids[0, 1] - centroids[1, 1])
        relation, axis = relation_from_delta(dx, dy)

        centroid_stack.append(centroids)
        weights.append(weight)
        local_qualities.append(local_quality)
        stabilities.append(stability)
        separations.append(separation)
        visual_masses.append(visual_mass)
        head_rows.append({
            "layer": layer,
            "head": head,
            "selection_score": float(selected["selection_score"]),
            "unsupervised_score": float(
                selected["unsupervised_score"]
            ),
            "oracle_centroid_accuracy": float(
                selected["oracle_centroid_accuracy"]
            ),
            "sample_weight": weight,
            "sample_local_quality": local_quality,
            "swap_stability": stability,
            "separation": separation,
            "visual_mass": visual_mass,
            "subject_x": float(centroids[0, 0]),
            "subject_y": float(centroids[0, 1]),
            "reference_x": float(centroids[1, 0]),
            "reference_y": float(centroids[1, 1]),
            "delta_x": dx,
            "delta_y": dy,
            "prediction": relation,
            "axis_confidence": axis,
        })

    centroids_np = np.stack(centroid_stack, axis=0)
    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np /= weights_np.sum()

    # Continuous geometry is always retained for guidance.
    ensemble_centroids = np.sum(
        weights_np[:, None, None] * centroids_np,
        axis=0,
    )
    dx = float(
        ensemble_centroids[0, 0] - ensemble_centroids[1, 0]
    )
    dy = float(
        ensemble_centroids[0, 1] - ensemble_centroids[1, 1]
    )
    geometric_prediction, axis_confidence = relation_from_delta(
        dx,
        dy,
    )

    if ensemble_mode == "weighted_vote":
        vote = np.zeros(len(RELATIONS), dtype=np.float64)
        for index, row in enumerate(head_rows):
            vote[RELATION_TO_INDEX[row["prediction"]]] += weights_np[index]
        prediction = RELATIONS[int(np.argmax(vote))]
    elif ensemble_mode in ("weighted_centroid", "mean_centroid"):
        if ensemble_mode == "mean_centroid":
            ensemble_centroids = np.mean(centroids_np, axis=0)
            dx = float(
                ensemble_centroids[0, 0]
                - ensemble_centroids[1, 0]
            )
            dy = float(
                ensemble_centroids[0, 1]
                - ensemble_centroids[1, 1]
            )
            geometric_prediction, axis_confidence = relation_from_delta(
                dx,
                dy,
            )
        prediction = geometric_prediction
    else:
        raise ValueError(
            f"Unsupported ensemble mode: {ensemble_mode}"
        )

    if ensemble_mode == "weighted_vote":
        agreement = float(np.sum([
            weights_np[index]
            for index, row in enumerate(head_rows)
            if row["prediction"] == prediction
        ]))
    else:
        agreement = float(np.sum([
            weights_np[index]
            for index, row in enumerate(head_rows)
            if row["prediction"] == prediction
        ]))

    swap_stability = float(np.sum(
        weights_np * np.asarray(stabilities, dtype=np.float64)
    ))
    mean_separation = float(np.sum(
        weights_np * np.asarray(separations, dtype=np.float64)
    ))
    mean_visual_mass = float(np.sum(
        weights_np * np.asarray(visual_masses, dtype=np.float64)
    ))
    confidence = evidence_confidence(
        mode=confidence_mode,
        axis=axis_confidence,
        agreement=agreement,
        swap=swap_stability,
        separation=mean_separation,
        visual_mass=mean_visual_mass,
    )
    _, geometry_scores = geometry_scores_from_delta(dx, dy)

    return TopKCentroidEvidence(
        sid=sid,
        selected_heads=[dict(row) for row in selected_heads],
        source=centroid_source,
        ensemble_mode=ensemble_mode,
        weight_mode=weight_mode,
        prediction=prediction,
        delta_x=dx,
        delta_y=dy,
        subject_x=float(ensemble_centroids[0, 0]),
        subject_y=float(ensemble_centroids[0, 1]),
        reference_x=float(ensemble_centroids[1, 0]),
        reference_y=float(ensemble_centroids[1, 1]),
        axis_confidence=float(axis_confidence),
        head_agreement=agreement,
        swap_stability=swap_stability,
        mean_separation=mean_separation,
        mean_visual_mass=mean_visual_mass,
        confidence=float(confidence),
        geometry_scores=geometry_scores,
        head_rows=head_rows,
    )


def evidence_to_dict(
    evidence: TopKCentroidEvidence,
) -> Dict[str, Any]:
    return {
        "sid": evidence.sid,
        "source": evidence.source,
        "ensemble_mode": evidence.ensemble_mode,
        "weight_mode": evidence.weight_mode,
        "centroid_prediction": evidence.prediction,
        "delta_x": evidence.delta_x,
        "delta_y": evidence.delta_y,
        "subject_x": evidence.subject_x,
        "subject_y": evidence.subject_y,
        "reference_x": evidence.reference_x,
        "reference_y": evidence.reference_y,
        "axis_confidence": evidence.axis_confidence,
        "head_agreement": evidence.head_agreement,
        "swap_stability": evidence.swap_stability,
        "mean_separation": evidence.mean_separation,
        "mean_visual_mass": evidence.mean_visual_mass,
        "centroid_confidence": evidence.confidence,
        "geometry_scores": evidence.geometry_scores,
        "head_rows": evidence.head_rows,
    }


def evidence_from_row(row: Dict[str, Any]) -> TopKCentroidEvidence:
    return TopKCentroidEvidence(
        sid=int(row["sid"]),
        selected_heads=[],
        source=str(row["source"]),
        ensemble_mode=str(row["ensemble_mode"]),
        weight_mode=str(row["weight_mode"]),
        prediction=str(row["centroid_prediction"]),
        delta_x=float(row["delta_x"]),
        delta_y=float(row["delta_y"]),
        subject_x=float(row["subject_x"]),
        subject_y=float(row["subject_y"]),
        reference_x=float(row["reference_x"]),
        reference_y=float(row["reference_y"]),
        axis_confidence=float(row["axis_confidence"]),
        head_agreement=float(row["head_agreement"]),
        swap_stability=float(row["swap_stability"]),
        mean_separation=float(row["mean_separation"]),
        mean_visual_mass=float(row["mean_visual_mass"]),
        confidence=float(row["centroid_confidence"]),
        geometry_scores={
            relation: float(row["geometry_scores"][relation])
            for relation in RELATIONS
        },
        head_rows=list(row.get("head_rows", [])),
    )


def next_token_relation_scores(
    *,
    model: Any,
    batch: Dict[str, Any],
    label_token_ids: Dict[str, List[int]],
) -> Dict[str, Any]:
    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
            return_dict=True,
        )
    logits = outputs.logits[0, -1].float()
    relation_scores = np.asarray([
        max(
            float(logits[token_id].item())
            for token_id in label_token_ids[relation]
        )
        for relation in RELATIONS
    ], dtype=np.float64)
    relation_probs = torch.softmax(
        torch.from_numpy(relation_scores),
        dim=0,
    ).numpy()
    order = np.argsort(relation_scores)[::-1]
    top_relation = RELATIONS[int(order[0])]
    margin = float(
        relation_scores[order[0]] - relation_scores[order[1]]
    )
    result = {
        "relation_scores": {
            relation: float(relation_scores[index])
            for index, relation in enumerate(RELATIONS)
        },
        "relation_probs": {
            relation: float(relation_probs[index])
            for index, relation in enumerate(RELATIONS)
        },
        "top_relation": top_relation,
        "margin": margin,
    }
    del outputs, logits
    return result


class TopKCentroidLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        prompt_length: int,
        label_token_ids: Dict[str, List[int]],
        label_bias: Dict[str, float],
        mode: str,
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.label_token_ids = {
            relation: [int(token_id) for token_id in ids]
            for relation, ids in label_token_ids.items()
        }
        self.label_bias = {
            relation: float(label_bias[relation])
            for relation in RELATIONS
        }
        self.mode = str(mode)
        self.applied = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.applied:
            return scores
        if int(input_ids.shape[1]) != self.prompt_length:
            return scores

        updated = scores.clone()
        if self.mode == "constrained":
            masked = torch.full_like(updated, float("-inf"))
            for relation in RELATIONS:
                for token_id in self.label_token_ids[relation]:
                    masked[:, token_id] = (
                        updated[:, token_id]
                        + self.label_bias[relation]
                    )
            updated = masked
        elif self.mode == "soft":
            for relation in RELATIONS:
                for token_id in self.label_token_ids[relation]:
                    updated[:, token_id] = (
                        updated[:, token_id]
                        + self.label_bias[relation]
                    )
        else:
            raise ValueError(
                f"Unsupported guide mode: {self.mode}"
            )
        self.applied = True
        return updated


def guided_generate(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    evidence: TopKCentroidEvidence,
    label_token_ids: Dict[str, List[int]],
    lm_margin: float,
    guide_strength: float,
    min_confidence: float,
    max_lm_margin: float,
    guide_mode: str,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    prompt_length = int(batch["input_ids"].shape[1])
    geometry_vector = np.asarray([
        evidence.geometry_scores[relation]
        for relation in RELATIONS
    ], dtype=np.float64)

    enabled = (
        evidence.confidence >= min_confidence
        and lm_margin <= max_lm_margin
        and guide_strength > 0.0
        and float(np.max(np.abs(geometry_vector))) > 0.0
    )
    effective_strength = (
        float(guide_strength) * float(evidence.confidence)
        if enabled else 0.0
    )
    label_bias = {
        relation: (
            effective_strength * float(geometry_vector[index])
            if enabled else 0.0
        )
        for index, relation in enumerate(RELATIONS)
    }

    spatial_processor = TopKCentroidLogitsProcessor(
        prompt_length=prompt_length,
        label_token_ids=label_token_ids,
        label_bias=label_bias,
        mode=guide_mode,
    )
    processors = LogitsProcessorList([spatial_processor])

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            logits_processor=processors,
        )
    text = decode_new_tokens(
        processor,
        output_ids,
        prompt_length,
    )
    del output_ids

    if not spatial_processor.applied:
        raise RuntimeError(
            "Top-k centroid logits processor was never applied"
        )

    return text, {
        "guidance_enabled": bool(enabled),
        "guidance_applied": True,
        "guide_strength": float(guide_strength),
        "effective_guide_strength": effective_strength,
        "centroid_confidence": float(evidence.confidence),
        "lm_margin": float(lm_margin),
        "max_lm_margin": float(max_lm_margin),
        "label_bias": label_bias,
        "geometry_scores": evidence.geometry_scores,
    }


def summarize_analysis(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("centroid_prediction")
        and row.get("gt") in RELATIONS
    ]
    if not valid:
        return {"n_valid": 0}

    per_relation: Dict[str, Any] = {}
    for relation in RELATIONS:
        relation_rows = [
            row for row in valid if row["gt"] == relation
        ]
        per_relation[relation] = {
            "n": len(relation_rows),
            "centroid_accuracy": (
                float(np.mean([
                    bool(row["centroid_correct"])
                    for row in relation_rows
                ]))
                if relation_rows else None
            ),
            "mean_confidence": (
                float(np.mean([
                    float(row["centroid_confidence"])
                    for row in relation_rows
                ]))
                if relation_rows else None
            ),
        }

    return {
        "n_valid": len(valid),
        "centroid_accuracy": float(np.mean([
            bool(row["centroid_correct"]) for row in valid
        ])),
        "mean_centroid_confidence": float(np.mean([
            float(row["centroid_confidence"]) for row in valid
        ])),
        "mean_axis_confidence": float(np.mean([
            float(row["axis_confidence"]) for row in valid
        ])),
        "mean_head_agreement": float(np.mean([
            float(row["head_agreement"]) for row in valid
        ])),
        "mean_swap_stability": float(np.mean([
            float(row["swap_stability"]) for row in valid
        ])),
        "per_relation": per_relation,
    }


def summarize_generation(
    *,
    analysis_rows: Sequence[Dict[str, Any]],
    generation_rows: Sequence[Dict[str, Any]],
    strengths: Sequence[float],
) -> Dict[str, Any]:
    analysis_by_sid = {
        int(row["sid"]): row for row in analysis_rows
    }
    valid = [
        row for row in generation_rows
        if row.get("baseline_prediction")
        and int(row["sid"]) in analysis_by_sid
    ]
    if not valid:
        return {"n_valid": 0}

    quadrants = Counter()
    centroid_on_baseline_wrong: List[bool] = []
    centroid_on_baseline_correct: List[bool] = []
    margins_by_quadrant: Dict[str, List[float]] = defaultdict(list)

    per_relation: Dict[str, Dict[str, Any]] = {}
    for relation in RELATIONS:
        per_relation[relation] = {
            "n": 0,
            "baseline_correct": 0,
            "centroid_correct": 0,
            "guided_correct": {
                str(strength): 0 for strength in strengths
            },
        }

    for row in valid:
        analysis = analysis_by_sid[int(row["sid"])]
        baseline_correct = bool(row["baseline_correct"])
        centroid_correct = bool(analysis["centroid_correct"])
        key = (
            ("centroid_correct" if centroid_correct else "centroid_wrong")
            + "__"
            + ("generation_correct" if baseline_correct else "generation_wrong")
        )
        quadrants[key] += 1
        margins_by_quadrant[key].append(
            float(row["lm_relation_margin"])
        )
        if baseline_correct:
            centroid_on_baseline_correct.append(centroid_correct)
        else:
            centroid_on_baseline_wrong.append(centroid_correct)

        relation = row["gt"]
        stats = per_relation[relation]
        stats["n"] += 1
        stats["baseline_correct"] += int(baseline_correct)
        stats["centroid_correct"] += int(centroid_correct)
        for strength in strengths:
            condition = row["guided"][str(strength)]
            stats["guided_correct"][str(strength)] += int(
                bool(condition["correct"])
            )

    conditions: Dict[str, Any] = {}
    for strength in strengths:
        key = str(strength)
        fixed = broken = changed = enabled = 0
        condition_valid = []
        for row in valid:
            result = row["guided"].get(key)
            if not result or not result.get("prediction"):
                continue
            condition_valid.append(row)
            baseline_correct = bool(row["baseline_correct"])
            guided_correct = bool(result["correct"])
            fixed += int((not baseline_correct) and guided_correct)
            broken += int(baseline_correct and (not guided_correct))
            changed += int(
                row["baseline_prediction"] != result["prediction"]
            )
            enabled += int(bool(result["guidance_enabled"]))

        baseline_accuracy = float(np.mean([
            bool(row["baseline_correct"])
            for row in condition_valid
        ])) if condition_valid else None
        guided_accuracy = float(np.mean([
            bool(row["guided"][key]["correct"])
            for row in condition_valid
        ])) if condition_valid else None

        conditions[key] = {
            "n_valid": len(condition_valid),
            "baseline_accuracy": baseline_accuracy,
            "guided_accuracy": guided_accuracy,
            "absolute_change": (
                guided_accuracy - baseline_accuracy
                if baseline_accuracy is not None
                and guided_accuracy is not None
                else None
            ),
            "fixed": fixed,
            "broken": broken,
            "net_fixed_minus_broken": fixed - broken,
            "changed_predictions": changed,
            "guidance_enabled": enabled,
        }

    per_relation_output: Dict[str, Any] = {}
    for relation, stats in per_relation.items():
        n = stats["n"]
        per_relation_output[relation] = {
            "n": n,
            "baseline_accuracy": (
                stats["baseline_correct"] / n if n else None
            ),
            "centroid_accuracy": (
                stats["centroid_correct"] / n if n else None
            ),
            "guided_accuracy": {
                key: (value / n if n else None)
                for key, value in stats["guided_correct"].items()
            },
        }

    return {
        "n_valid": len(valid),
        "baseline_accuracy": float(np.mean([
            bool(row["baseline_correct"]) for row in valid
        ])),
        "centroid_accuracy": float(np.mean([
            bool(analysis_by_sid[int(row["sid"])]["centroid_correct"])
            for row in valid
        ])),
        "centroid_accuracy_on_baseline_wrong": (
            float(np.mean(centroid_on_baseline_wrong))
            if centroid_on_baseline_wrong else None
        ),
        "n_baseline_wrong": len(centroid_on_baseline_wrong),
        "centroid_accuracy_on_baseline_correct": (
            float(np.mean(centroid_on_baseline_correct))
            if centroid_on_baseline_correct else None
        ),
        "n_baseline_correct": len(centroid_on_baseline_correct),
        "baseline_or_centroid_oracle_accuracy": float(np.mean([
            bool(row["baseline_correct"])
            or bool(
                analysis_by_sid[int(row["sid"])]["centroid_correct"]
            )
            for row in valid
        ])),
        "baseline_centroid_agreement": float(np.mean([
            row["baseline_prediction"]
            == analysis_by_sid[int(row["sid"])]["centroid_prediction"]
            for row in valid
        ])),
        "quadrants": dict(quadrants),
        "mean_lm_margin_by_quadrant": {
            key: float(np.mean(values))
            for key, values in margins_by_quadrant.items()
            if values
        },
        "conditions": conditions,
        "per_relation": per_relation_output,
    }


def print_selected_heads(
    selected_heads: Sequence[Dict[str, Any]],
    mode: str,
) -> None:
    print("\nSelected heads:")
    for rank, row in enumerate(selected_heads, 1):
        print(
            f"  {rank:2d}. L{row['layer']:02d} H{row['head']:02d} | "
            f"selection={row['selection_score']:.6f} | "
            f"unsup={row['unsupervised_score']:.6f} | "
            f"oracle_acc={row['oracle_centroid_accuracy']:.4f}"
        )
    if mode == "oracle_accuracy":
        print(
            "  NOTE: oracle_accuracy uses GT-derived aggregate head accuracy "
            "and is diagnostic only."
        )


def print_analysis_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 102)
    print("TOP-K ATTENTION-CENTROID ANALYSIS")
    print("=" * 102)
    print(
        f"centroid accuracy:       {summary['centroid_accuracy']:.4f} "
        f"(n={summary['n_valid']})"
    )
    print(
        f"mean confidence:         "
        f"{summary['mean_centroid_confidence']:.4f}"
    )
    print(
        f"mean axis confidence:    "
        f"{summary['mean_axis_confidence']:.4f}"
    )
    print(
        f"mean head agreement:     "
        f"{summary['mean_head_agreement']:.4f}"
    )
    print(
        f"mean swap stability:     "
        f"{summary['mean_swap_stability']:.4f}"
    )
    print("\nPer relation:")
    for relation, stats in summary["per_relation"].items():
        print(
            f"  {relation:6s} n={stats['n']:4d} | "
            f"centroid={stats['centroid_accuracy']:.4f} | "
            f"confidence={stats['mean_confidence']:.4f}"
        )


def print_generation_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 108)
    print("TOP-K CENTROID VS GENERATION")
    print("=" * 108)
    print(
        f"baseline generation:     {summary['baseline_accuracy']:.4f}"
    )
    print(
        f"top-k centroid:          {summary['centroid_accuracy']:.4f}"
    )
    print(
        f"centroid on base-wrong:  "
        f"{summary['centroid_accuracy_on_baseline_wrong']:.4f} "
        f"(n={summary['n_baseline_wrong']})"
    )
    print(
        f"centroid on base-correct:"
        f"{summary['centroid_accuracy_on_baseline_correct']:.4f} "
        f"(n={summary['n_baseline_correct']})"
    )
    print(
        f"baseline ∪ centroid oracle:"
        f"{summary['baseline_or_centroid_oracle_accuracy']:.4f}"
    )
    print(
        f"baseline/centroid agreement:"
        f"{summary['baseline_centroid_agreement']:.4f}"
    )

    print("\nMechanism quadrants:")
    for key in [
        "centroid_correct__generation_correct",
        "centroid_correct__generation_wrong",
        "centroid_wrong__generation_correct",
        "centroid_wrong__generation_wrong",
    ]:
        print(
            f"  {key:40s}: "
            f"{summary['quadrants'].get(key, 0)}"
        )

    print("\nGuided generation:")
    for strength, stats in summary["conditions"].items():
        print(
            f"  strength={strength:>5s} | "
            f"guided={stats['guided_accuracy']:.4f} | "
            f"delta={stats['absolute_change']:+.4f} | "
            f"fixed={stats['fixed']} | broken={stats['broken']} | "
            f"net={stats['net_fixed_minus_broken']:+d} | "
            f"changed={stats['changed_predictions']} | "
            f"enabled={stats['guidance_enabled']}"
        )

    print("\nPer relation:")
    for relation, stats in summary["per_relation"].items():
        guided = " | ".join(
            f"g@{strength}={accuracy:.4f}"
            for strength, accuracy in stats["guided_accuracy"].items()
        )
        print(
            f"  {relation:6s} n={stats['n']:4d} | "
            f"base={stats['baseline_accuracy']:.4f} | "
            f"centroid={stats['centroid_accuracy']:.4f} | "
            f"{guided}"
        )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if args.only != "analyze":
            raise RuntimeError("CUDA requested but unavailable")
    if not (0.0 <= args.min_guide_confidence <= 1.0):
        raise ValueError(
            "--min-guide-confidence must be in [0, 1]"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0")

    strengths = parse_float_list(args.guide_strengths)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    step1_dir = Path(args.step1_dir)
    head_table = load_step1_head_table(step1_dir)
    selected_heads = select_heads(
        head_table,
        args.head_selection,
        args.top_k,
        parse_manual_heads(args.manual_heads),
    )

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = (
        Path(args.analysis_jsonl)
        if args.analysis_jsonl
        else output_dir / "centroid_analysis.jsonl"
    )
    generation_path = output_dir / "generation.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "model": args.model,
        "step1_dir": str(step1_dir),
        "head_selection": args.head_selection,
        "top_k": args.top_k,
        "selected_heads": selected_heads,
        "centroid_source": args.centroid_source,
        "ensemble_mode": args.ensemble_mode,
        "weight_mode": args.weight_mode,
        "confidence_mode": args.confidence_mode,
        "guide_strengths": strengths,
        "min_guide_confidence": args.min_guide_confidence,
        "max_lm_margin": args.max_lm_margin,
        "guide_mode": args.guide_mode,
        "uses_gt_for_guidance": (
            args.head_selection == "oracle_accuracy"
        ),
        "updates_model_weights": False,
        "changes_prompt": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_selected_heads(
        selected_heads,
        args.head_selection,
    )

    module = import_two_object_module()
    records, audit = module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    if not records:
        raise RuntimeError("No usable records")
    prompt_path = resolve_prompt_path(args)
    prompt_rows = load_standard_prompts(prompt_path)
    missing_ids = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing_ids:
        raise RuntimeError(
            f"Prompt file is missing {len(missing_ids)} IDs; "
            f"first={missing_ids[:10]}"
        )

    if args.only in ("analyze", "both"):
        if analysis_path.exists():
            analysis_path.unlink()
        analysis_rows: List[Dict[str, Any]] = []
        for record in tqdm(records, desc="topk-centroid-analysis"):
            sid = int(record.sid)
            try:
                prompt_row = prompt_rows[sid]
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATIONS:
                    raise ValueError(
                        f"Unsupported GT for sid={sid}: {gt!r}"
                    )
                evidence = load_topk_evidence(
                    sample_path=step1_dir / "sample_arrays" / f"{sid}.npz",
                    selected_heads=selected_heads,
                    centroid_source=args.centroid_source,
                    ensemble_mode=args.ensemble_mode,
                    weight_mode=args.weight_mode,
                    confidence_mode=args.confidence_mode,
                )
                row = evidence_to_dict(evidence)
                row.update({
                    "subject": str(prompt_row["subject"]),
                    "reference": str(prompt_row["reference"]),
                    "gt": gt,
                    "centroid_correct": evidence.prediction == gt,
                })
                append_jsonl(analysis_path, row)
                analysis_rows.append(row)
            except Exception as exc:
                append_jsonl(errors_path, {
                    "stage": "analysis",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": (
                        traceback.format_exc().splitlines()[-16:]
                    ),
                })

        analysis_summary = summarize_analysis(analysis_rows)
        print_analysis_summary(analysis_summary)
    else:
        analysis_rows = read_jsonl(analysis_path)
        if not analysis_rows:
            raise RuntimeError(
                f"No analysis rows found in {analysis_path}"
            )
        analysis_summary = summarize_analysis(analysis_rows)

    if args.only == "analyze":
        summary_path.write_text(
            json.dumps({
                "config": config,
                "audit": audit,
                "analysis": analysis_summary,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved analysis: {analysis_path}")
        print(f"Saved summary:  {summary_path}")
        return

    if args.model not in module.SPECS:
        raise ValueError(
            f"Model {args.model!r} not found in model specs"
        )
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
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    print(f"\nLoading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        **load_kwargs,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    configure_processor(model, processor)
    device = torch.device(args.device)
    label_token_ids = label_token_id_variants(
        processor.tokenizer
    )

    analysis_by_sid = {
        int(row["sid"]): row for row in analysis_rows
    }
    if generation_path.exists():
        generation_path.unlink()

    completed = 0
    started = time.time()
    try:
        for record in tqdm(
            records,
            desc=f"topk-centroid-generation:{args.model}",
        ):
            sid = int(record.sid)
            batch = None
            image = None
            try:
                if sid not in analysis_by_sid:
                    raise RuntimeError(
                        f"Missing centroid analysis for sid={sid}"
                    )
                prompt_row = prompt_rows[sid]
                (
                    batch,
                    subject,
                    reference,
                    question_text,
                    _raw_question,
                    answer_raw,
                    image,
                ) = make_batch(
                    processor,
                    record,
                    prompt_row,
                    device,
                )
                gt = normalize_relation(answer_raw)
                evidence = evidence_from_row(
                    analysis_by_sid[sid]
                )

                lm_info = next_token_relation_scores(
                    model=model,
                    batch=batch,
                    label_token_ids=label_token_ids,
                )
                baseline_text = generate_text(
                    model,
                    processor,
                    batch,
                    args.max_new_tokens,
                )
                baseline_prediction = normalize_relation(
                    baseline_text
                )
                baseline_correct = bool(
                    baseline_prediction is not None
                    and baseline_prediction == gt
                )

                guided_results: Dict[str, Any] = {}
                for strength in strengths:
                    text, metadata = guided_generate(
                        model=model,
                        processor=processor,
                        batch=batch,
                        evidence=evidence,
                        label_token_ids=label_token_ids,
                        lm_margin=float(lm_info["margin"]),
                        guide_strength=float(strength),
                        min_confidence=float(
                            args.min_guide_confidence
                        ),
                        max_lm_margin=float(args.max_lm_margin),
                        guide_mode=args.guide_mode,
                        max_new_tokens=args.max_new_tokens,
                    )
                    prediction = normalize_relation(text)
                    guided_results[str(strength)] = {
                        **metadata,
                        "generated_text": text,
                        "prediction": prediction,
                        "correct": bool(
                            prediction is not None
                            and prediction == gt
                        ),
                    }

                row = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "gt": gt,
                    "question": question_text,
                    "centroid_prediction": evidence.prediction,
                    "centroid_correct": evidence.prediction == gt,
                    "centroid_confidence": evidence.confidence,
                    "delta_x": evidence.delta_x,
                    "delta_y": evidence.delta_y,
                    "head_agreement": evidence.head_agreement,
                    "swap_stability": evidence.swap_stability,
                    "baseline_generated_text": baseline_text,
                    "baseline_prediction": baseline_prediction,
                    "baseline_correct": baseline_correct,
                    "lm_relation_scores": lm_info[
                        "relation_scores"
                    ],
                    "lm_relation_probs": lm_info[
                        "relation_probs"
                    ],
                    "lm_relation_top": lm_info[
                        "top_relation"
                    ],
                    "lm_relation_margin": lm_info["margin"],
                    "guided": guided_results,
                }
                append_jsonl(generation_path, row)
                completed += 1

                if should_print_sample(
                    completed,
                    args.print_every,
                ):
                    lines = [
                        f"\n[{completed}/{len(records)}] sid={sid} | "
                        f"{subject} -> {reference}",
                        f"  GT/base/centroid: {gt} / "
                        f"{baseline_prediction} / {evidence.prediction}",
                        f"  centroid conf={evidence.confidence:.4f} | "
                        f"agreement={evidence.head_agreement:.4f} | "
                        f"swap={evidence.swap_stability:.4f} | "
                        f"LM margin={lm_info['margin']:.4f}",
                    ]
                    for strength in strengths:
                        result = guided_results[str(strength)]
                        status = "SAME"
                        if (
                            not baseline_correct
                            and result["correct"]
                        ):
                            status = "FIXED"
                        elif (
                            baseline_correct
                            and not result["correct"]
                        ):
                            status = "BROKEN"
                        elif (
                            baseline_prediction
                            != result["prediction"]
                        ):
                            status = "CHANGED"
                        lines.append(
                            f"  guide@{strength:<5g}: "
                            f"{result['prediction']} | {status} | "
                            f"enabled={int(result['guidance_enabled'])} | "
                            f"eff={result['effective_guide_strength']:.4f}"
                        )
                    tqdm.write("\n".join(lines))

                del guided_results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(errors_path, {
                    "stage": "generation",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": (
                        traceback.format_exc().splitlines()[-16:]
                    ),
                })
                tqdm.write(
                    f"\n[ERROR] sid={sid} | "
                    f"{type(exc).__name__}: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if batch is not None:
                    del batch
                if image is not None:
                    del image

        generation_rows = read_jsonl(generation_path)
        generation_summary = summarize_generation(
            analysis_rows=analysis_rows,
            generation_rows=generation_rows,
            strengths=strengths,
        )
        print_generation_summary(generation_summary)

        summary = {
            "config": config,
            "audit": audit,
            "elapsed_minutes": (
                time.time() - started
            ) / 60.0,
            "analysis": analysis_summary,
            "generation": generation_summary,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved analysis:   {analysis_path}")
        print(f"Saved generation: {generation_path}")
        print(f"Saved summary:    {summary_path}")
        if errors_path.exists():
            print(f"Saved errors:     {errors_path}")

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
