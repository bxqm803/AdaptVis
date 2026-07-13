#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free swap-consistent object-token generation.

For each image/question pair, the script runs the same frozen VLM on:

    Q_AB: Where is object A in relation to object B?
    Q_BA: Where is object B in relation to object A?

At a selected decoder layer it extracts the two object-word hidden states.

Let

    z_AB = h_A(Q_AB) - h_B(Q_AB)
    z_BA = h_B(Q_BA) - h_A(Q_BA)

The component that reverses under object-order exchange is

    z_consistent = 0.5 * (z_AB - z_BA)

while

    z_role_bias = 0.5 * (z_AB + z_BA)

is the component that does not reverse under the exchange and can contain
subject/reference-role or prompt-order bias.

During the final pass, the original question Q_AB is used unchanged.  At the
selected decoder layer, only its two object-token states are modified.  Their
shared center is preserved, while their difference is interpolated toward
z_consistent.  All later layers and the LM head then generate normally.

No model parameters or probes are trained.  No answer labels or direction words
are used to construct the intervention.  Ground-truth labels are read only
after generation for evaluation.

The script also reports centroid grounding from:
- the original prompt,
- the swapped prompt aligned back to A relative to B,
- the average of the two aligned centroid estimates.
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
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "swap-consistent-token-generation-v1"

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


@dataclass
class SwapData:
    sid: int
    layer: int
    subject_index: int
    reference_index: int
    input_length: int

    z_original: np.ndarray
    z_swapped_role: np.ndarray
    z_consistent: np.ndarray
    z_role_bias: np.ndarray

    swap_antisymmetry_cosine: float
    original_norm: float
    swapped_role_norm: float
    consistent_norm: float
    role_bias_norm: float
    role_bias_ratio: float

    original_map_separation: float
    swapped_map_separation: float
    original_entropy_confidence: float
    swapped_entropy_confidence: float

    original_subject_x: float
    original_subject_y: float
    original_reference_x: float
    original_reference_y: float

    swapped_subject_x: float
    swapped_subject_y: float
    swapped_reference_x: float
    swapped_reference_y: float

    average_subject_x: float
    average_subject_y: float
    average_reference_x: float
    average_reference_y: float

    original_axis_confidence: float
    swapped_axis_confidence: float
    average_axis_confidence: float

    original_grounding_prediction: str
    swapped_grounding_prediction: str
    average_grounding_prediction: str


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
    )
    p.add_argument(
        "--layer",
        default="auto",
        help="Zero-based decoder block index, or 'auto'.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for object-token/visual-token centroid maps.",
    )
    p.add_argument(
        "--swap-strength",
        type=float,
        default=0.5,
        help=(
            "Interpolation strength from the original object-token difference "
            "toward the swap-consistent difference. 0 is a no-op; 1 removes the "
            "entire swap-symmetric role-bias component."
        ),
    )
    p.add_argument(
        "--target-norm",
        default="preserve_original",
        choices=["raw", "preserve_original"],
        help=(
            "Whether to use the raw swap-consistent vector norm or match it to "
            "the current original object-difference norm before interpolation."
        ),
    )
    p.add_argument(
        "--patch-confidence-mode",
        default="none",
        choices=[
            "none",
            "grounding",
            "swap_consistency",
            "grounding_swap",
        ],
        help=(
            "Optional label-free scaling of --swap-strength. 'grounding' uses "
            "the geometric mean of original/swapped map separation; "
            "'swap_consistency' uses cosine agreement between z_AB and -z_BA."
        ),
    )
    p.add_argument(
        "--min-patch-confidence",
        type=float,
        default=0.0,
        help="Disable the intervention below this confidence.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one detailed sample every N completed samples; 0 disables.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--only",
        choices=["both", "baseline", "patched"],
        default="both",
        help=(
            "Run both stages, only baseline/swap capture, or only patched "
            "generation from saved swap data."
        ),
    )
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


def entropy_confidence(weights: torch.Tensor) -> torch.Tensor:
    n = int(weights.shape[-1])
    if n <= 1:
        return torch.ones(weights.shape[:-1], device=weights.device, dtype=weights.dtype)
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
    return (1.0 - entropy / math.log(n)).clamp(0.0, 1.0)


def confidence_value(
    mode: str,
    separation: float,
    subject_entropy: float,
    reference_entropy: float,
) -> float:
    if mode == "none":
        return 1.0
    entropy_pair = math.sqrt(max(0.0, subject_entropy * reference_entropy))
    if mode == "separation":
        return float(np.clip(separation, 0.0, 1.0))
    if mode == "entropy":
        return float(np.clip(entropy_pair, 0.0, 1.0))
    if mode == "combined":
        return float(np.clip(math.sqrt(max(0.0, separation * entropy_pair)), 0.0, 1.0))
    raise ValueError(f"Unsupported confidence mode: {mode}")



@dataclass
class PromptState:
    first_index: int
    second_index: int
    first_hidden: torch.Tensor
    second_hidden: torch.Tensor
    first_x: float
    first_y: float
    second_x: float
    second_y: float
    map_separation: float
    entropy_confidence: float
    axis_confidence: float
    grounding_prediction: str
    n_visual_tokens: int


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


def extract_prompt_state(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    first_object: str,
    second_object: str,
    layer: int,
    temperature: float,
) -> PromptState:
    if temperature <= 0:
        raise ValueError("--temperature must be positive")

    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    first_span, second_span = locate_object_spans(
        processor.tokenizer,
        input_ids,
        first_object,
        second_object,
    )
    first_index = int(first_span[1])
    second_index = int(second_span[1])
    visual_indices = resolve_visual_indices(
        model,
        processor,
        batch,
        input_ids,
    )

    with torch.inference_mode():
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    states = hidden_tuple(outputs)
    n_blocks = len(states) - 1
    if not (0 <= layer < n_blocks):
        raise ValueError(
            f"Requested layer={layer}, model has {n_blocks} decoder blocks"
        )

    hidden = states[layer + 1][0].float()
    if int(hidden.shape[0]) != len(input_ids):
        raise RuntimeError(
            f"Token/hidden mismatch: input={len(input_ids)}, "
            f"hidden={int(hidden.shape[0])}"
        )

    first_hidden = hidden[first_index]
    second_hidden = hidden[second_index]
    visual = hidden[
        torch.as_tensor(
            visual_indices,
            device=hidden.device,
            dtype=torch.long,
        )
    ]

    visual_norm = F.normalize(visual, dim=-1)
    first_logits = torch.matmul(
        visual_norm,
        F.normalize(first_hidden, dim=-1),
    ) / float(temperature)
    second_logits = torch.matmul(
        visual_norm,
        F.normalize(second_hidden, dim=-1),
    ) / float(temperature)

    first_weights = torch.softmax(first_logits, dim=0)
    second_weights = torch.softmax(second_logits, dim=0)

    separation = 0.5 * torch.sum(
        torch.abs(first_weights - second_weights)
    )
    first_entropy = entropy_confidence(first_weights)
    second_entropy = entropy_confidence(second_weights)
    entropy_pair = math.sqrt(
        max(
            0.0,
            float(first_entropy.item())
            * float(second_entropy.item()),
        )
    )

    coords = visual_coordinates(
        model,
        batch,
        len(visual_indices),
        hidden.device,
    )
    if coords is None or int(coords.shape[0]) != len(visual_indices):
        raise RuntimeError(
            f"Could not construct coordinates for "
            f"{len(visual_indices)} visual tokens"
        )

    first_center = torch.sum(
        first_weights[:, None] * coords,
        dim=0,
    )
    second_center = torch.sum(
        second_weights[:, None] * coords,
        dim=0,
    )
    first_x, first_y = [
        float(value) for value in first_center.tolist()
    ]
    second_x, second_y = [
        float(value) for value in second_center.tolist()
    ]
    prediction, axis_confidence = relation_from_centroids(
        first_x - second_x,
        first_y - second_y,
    )

    result = PromptState(
        first_index=first_index,
        second_index=second_index,
        first_hidden=first_hidden.detach().cpu(),
        second_hidden=second_hidden.detach().cpu(),
        first_x=first_x,
        first_y=first_y,
        second_x=second_x,
        second_y=second_y,
        map_separation=float(separation.item()),
        entropy_confidence=float(entropy_pair),
        axis_confidence=float(axis_confidence),
        grounding_prediction=prediction,
        n_visual_tokens=len(visual_indices),
    )

    del (
        outputs,
        states,
        hidden,
        visual,
        first_weights,
        second_weights,
    )
    return result


def capture_swap_data(
    *,
    model: Any,
    processor: Any,
    original_batch: Dict[str, Any],
    swapped_batch: Dict[str, Any],
    sid: int,
    subject: str,
    reference: str,
    layer: int,
    temperature: float,
) -> SwapData:
    # Original role order: A(subject) - B(reference).
    original = extract_prompt_state(
        model=model,
        processor=processor,
        batch=original_batch,
        first_object=subject,
        second_object=reference,
        layer=layer,
        temperature=temperature,
    )

    # Swapped role order: B(subject) - A(reference).
    swapped = extract_prompt_state(
        model=model,
        processor=processor,
        batch=swapped_batch,
        first_object=reference,
        second_object=subject,
        layer=layer,
        temperature=temperature,
    )

    h_a_original = original.first_hidden.float()
    h_b_original = original.second_hidden.float()
    h_b_swapped = swapped.first_hidden.float()
    h_a_swapped = swapped.second_hidden.float()

    z_original = h_a_original - h_b_original
    z_swapped_role = h_b_swapped - h_a_swapped

    # Antisymmetric under A/B exchange: relation/content component.
    z_consistent = 0.5 * (z_original - z_swapped_role)
    # Symmetric under A/B exchange: subject/reference role or order component.
    z_role_bias = 0.5 * (z_original + z_swapped_role)

    original_norm = float(z_original.norm().item())
    swapped_role_norm = float(z_swapped_role.norm().item())
    consistent_norm = float(z_consistent.norm().item())
    role_bias_norm = float(z_role_bias.norm().item())

    swap_antisymmetry_cosine = float(
        F.cosine_similarity(
            z_original[None],
            (-z_swapped_role)[None],
            dim=-1,
        ).item()
    )
    role_bias_ratio = role_bias_norm / max(original_norm, 1e-8)

    # Swapped prompt aligned back to the original question A relative to B:
    # A is the reference token in Q_BA, B is the subject token.
    swapped_subject_x = swapped.second_x
    swapped_subject_y = swapped.second_y
    swapped_reference_x = swapped.first_x
    swapped_reference_y = swapped.first_y
    swapped_prediction, swapped_axis_confidence = relation_from_centroids(
        swapped_subject_x - swapped_reference_x,
        swapped_subject_y - swapped_reference_y,
    )

    average_subject_x = 0.5 * (
        original.first_x + swapped_subject_x
    )
    average_subject_y = 0.5 * (
        original.first_y + swapped_subject_y
    )
    average_reference_x = 0.5 * (
        original.second_x + swapped_reference_x
    )
    average_reference_y = 0.5 * (
        original.second_y + swapped_reference_y
    )
    average_prediction, average_axis_confidence = relation_from_centroids(
        average_subject_x - average_reference_x,
        average_subject_y - average_reference_y,
    )

    return SwapData(
        sid=sid,
        layer=layer,
        subject_index=original.first_index,
        reference_index=original.second_index,
        input_length=int(original_batch["input_ids"].shape[1]),

        z_original=z_original.numpy().astype(np.float16),
        z_swapped_role=z_swapped_role.numpy().astype(np.float16),
        z_consistent=z_consistent.numpy().astype(np.float16),
        z_role_bias=z_role_bias.numpy().astype(np.float16),

        swap_antisymmetry_cosine=swap_antisymmetry_cosine,
        original_norm=original_norm,
        swapped_role_norm=swapped_role_norm,
        consistent_norm=consistent_norm,
        role_bias_norm=role_bias_norm,
        role_bias_ratio=role_bias_ratio,

        original_map_separation=original.map_separation,
        swapped_map_separation=swapped.map_separation,
        original_entropy_confidence=original.entropy_confidence,
        swapped_entropy_confidence=swapped.entropy_confidence,

        original_subject_x=original.first_x,
        original_subject_y=original.first_y,
        original_reference_x=original.second_x,
        original_reference_y=original.second_y,

        swapped_subject_x=swapped_subject_x,
        swapped_subject_y=swapped_subject_y,
        swapped_reference_x=swapped_reference_x,
        swapped_reference_y=swapped_reference_y,

        average_subject_x=average_subject_x,
        average_subject_y=average_subject_y,
        average_reference_x=average_reference_x,
        average_reference_y=average_reference_y,

        original_axis_confidence=original.axis_confidence,
        swapped_axis_confidence=swapped_axis_confidence,
        average_axis_confidence=average_axis_confidence,

        original_grounding_prediction=original.grounding_prediction,
        swapped_grounding_prediction=swapped_prediction,
        average_grounding_prediction=average_prediction,
    )


def save_swap_data(path: Path, data: SwapData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            sid=np.asarray(data.sid, dtype=np.int64),
            layer=np.asarray(data.layer, dtype=np.int32),
            subject_index=np.asarray(
                data.subject_index,
                dtype=np.int32,
            ),
            reference_index=np.asarray(
                data.reference_index,
                dtype=np.int32,
            ),
            input_length=np.asarray(
                data.input_length,
                dtype=np.int32,
            ),
            z_original=data.z_original,
            z_swapped_role=data.z_swapped_role,
            z_consistent=data.z_consistent,
            z_role_bias=data.z_role_bias,
            swap_antisymmetry_cosine=np.asarray(
                data.swap_antisymmetry_cosine,
                dtype=np.float32,
            ),
            original_norm=np.asarray(
                data.original_norm,
                dtype=np.float32,
            ),
            swapped_role_norm=np.asarray(
                data.swapped_role_norm,
                dtype=np.float32,
            ),
            consistent_norm=np.asarray(
                data.consistent_norm,
                dtype=np.float32,
            ),
            role_bias_norm=np.asarray(
                data.role_bias_norm,
                dtype=np.float32,
            ),
            role_bias_ratio=np.asarray(
                data.role_bias_ratio,
                dtype=np.float32,
            ),
            original_map_separation=np.asarray(
                data.original_map_separation,
                dtype=np.float32,
            ),
            swapped_map_separation=np.asarray(
                data.swapped_map_separation,
                dtype=np.float32,
            ),
            original_entropy_confidence=np.asarray(
                data.original_entropy_confidence,
                dtype=np.float32,
            ),
            swapped_entropy_confidence=np.asarray(
                data.swapped_entropy_confidence,
                dtype=np.float32,
            ),
            original_subject_x=np.asarray(
                data.original_subject_x,
                dtype=np.float32,
            ),
            original_subject_y=np.asarray(
                data.original_subject_y,
                dtype=np.float32,
            ),
            original_reference_x=np.asarray(
                data.original_reference_x,
                dtype=np.float32,
            ),
            original_reference_y=np.asarray(
                data.original_reference_y,
                dtype=np.float32,
            ),
            swapped_subject_x=np.asarray(
                data.swapped_subject_x,
                dtype=np.float32,
            ),
            swapped_subject_y=np.asarray(
                data.swapped_subject_y,
                dtype=np.float32,
            ),
            swapped_reference_x=np.asarray(
                data.swapped_reference_x,
                dtype=np.float32,
            ),
            swapped_reference_y=np.asarray(
                data.swapped_reference_y,
                dtype=np.float32,
            ),
            average_subject_x=np.asarray(
                data.average_subject_x,
                dtype=np.float32,
            ),
            average_subject_y=np.asarray(
                data.average_subject_y,
                dtype=np.float32,
            ),
            average_reference_x=np.asarray(
                data.average_reference_x,
                dtype=np.float32,
            ),
            average_reference_y=np.asarray(
                data.average_reference_y,
                dtype=np.float32,
            ),
            original_axis_confidence=np.asarray(
                data.original_axis_confidence,
                dtype=np.float32,
            ),
            swapped_axis_confidence=np.asarray(
                data.swapped_axis_confidence,
                dtype=np.float32,
            ),
            average_axis_confidence=np.asarray(
                data.average_axis_confidence,
                dtype=np.float32,
            ),
            original_grounding_prediction=np.asarray(
                data.original_grounding_prediction,
                dtype="<U8",
            ),
            swapped_grounding_prediction=np.asarray(
                data.swapped_grounding_prediction,
                dtype="<U8",
            ),
            average_grounding_prediction=np.asarray(
                data.average_grounding_prediction,
                dtype="<U8",
            ),
        )
    os.replace(tmp, path)


def load_swap_data(path: Path) -> SwapData:
    with np.load(path, allow_pickle=False) as z:
        return SwapData(
            sid=int(z["sid"].item()),
            layer=int(z["layer"].item()),
            subject_index=int(z["subject_index"].item()),
            reference_index=int(z["reference_index"].item()),
            input_length=int(z["input_length"].item()),

            z_original=z["z_original"],
            z_swapped_role=z["z_swapped_role"],
            z_consistent=z["z_consistent"],
            z_role_bias=z["z_role_bias"],

            swap_antisymmetry_cosine=float(
                z["swap_antisymmetry_cosine"].item()
            ),
            original_norm=float(z["original_norm"].item()),
            swapped_role_norm=float(
                z["swapped_role_norm"].item()
            ),
            consistent_norm=float(z["consistent_norm"].item()),
            role_bias_norm=float(z["role_bias_norm"].item()),
            role_bias_ratio=float(z["role_bias_ratio"].item()),

            original_map_separation=float(
                z["original_map_separation"].item()
            ),
            swapped_map_separation=float(
                z["swapped_map_separation"].item()
            ),
            original_entropy_confidence=float(
                z["original_entropy_confidence"].item()
            ),
            swapped_entropy_confidence=float(
                z["swapped_entropy_confidence"].item()
            ),

            original_subject_x=float(
                z["original_subject_x"].item()
            ),
            original_subject_y=float(
                z["original_subject_y"].item()
            ),
            original_reference_x=float(
                z["original_reference_x"].item()
            ),
            original_reference_y=float(
                z["original_reference_y"].item()
            ),

            swapped_subject_x=float(
                z["swapped_subject_x"].item()
            ),
            swapped_subject_y=float(
                z["swapped_subject_y"].item()
            ),
            swapped_reference_x=float(
                z["swapped_reference_x"].item()
            ),
            swapped_reference_y=float(
                z["swapped_reference_y"].item()
            ),

            average_subject_x=float(
                z["average_subject_x"].item()
            ),
            average_subject_y=float(
                z["average_subject_y"].item()
            ),
            average_reference_x=float(
                z["average_reference_x"].item()
            ),
            average_reference_y=float(
                z["average_reference_y"].item()
            ),

            original_axis_confidence=float(
                z["original_axis_confidence"].item()
            ),
            swapped_axis_confidence=float(
                z["swapped_axis_confidence"].item()
            ),
            average_axis_confidence=float(
                z["average_axis_confidence"].item()
            ),

            original_grounding_prediction=str(
                z["original_grounding_prediction"].item()
            ),
            swapped_grounding_prediction=str(
                z["swapped_grounding_prediction"].item()
            ),
            average_grounding_prediction=str(
                z["average_grounding_prediction"].item()
            ),
        )


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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
) -> Tuple[
    Dict[str, Any],
    str,
    str,
    str,
    str,
    Any,
    Image.Image,
]:
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


def patch_confidence(
    data: SwapData,
    mode: str,
) -> float:
    grounding = math.sqrt(
        max(
            0.0,
            float(data.original_map_separation)
            * float(data.swapped_map_separation),
        )
    )
    swap_consistency = float(
        np.clip(
            0.5 * (data.swap_antisymmetry_cosine + 1.0),
            0.0,
            1.0,
        )
    )

    if mode == "none":
        return 1.0
    if mode == "grounding":
        return float(np.clip(grounding, 0.0, 1.0))
    if mode == "swap_consistency":
        return swap_consistency
    if mode == "grounding_swap":
        return float(
            np.clip(grounding * swap_consistency, 0.0, 1.0)
        )
    raise ValueError(f"Unsupported patch confidence mode: {mode}")


def swap_patched_generate(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    decoder_layer: Any,
    data: SwapData,
    swap_strength: float,
    target_norm: str,
    confidence_mode: str,
    min_confidence: float,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    current_length = int(batch["input_ids"].shape[1])
    if current_length != data.input_length:
        raise RuntimeError(
            f"Original prompt token length changed: "
            f"saved={data.input_length}, current={current_length}"
        )

    confidence = patch_confidence(data, confidence_mode)
    effective_strength = float(swap_strength) * float(confidence)
    enabled = bool(
        confidence >= float(min_confidence)
        and abs(effective_strength) > 0.0
        and data.consistent_norm > 1e-8
    )

    target_saved = torch.from_numpy(
        data.z_consistent.astype(np.float32)
    ).to(batch["input_ids"].device)

    state: Dict[str, Any] = {
        "patched": False,
        "current_difference_norm": None,
        "target_difference_norm": None,
        "new_difference_norm": None,
        "applied_delta_norm": None,
        "current_target_cosine": None,
    }

    def hook(
        _module: Any,
        _inputs: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        if not enabled or state["patched"]:
            return output

        hidden = (
            output[0]
            if isinstance(output, (tuple, list))
            else output
        )
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output
        if int(hidden.shape[1]) <= max(
            data.subject_index,
            data.reference_index,
        ):
            # Decode steps after prefill usually have sequence length 1.
            return output

        hidden_new = hidden.clone()
        hs = hidden_new[:, data.subject_index, :].float()
        hr = hidden_new[:, data.reference_index, :].float()
        center = 0.5 * (hs + hr)
        z_current = hs - hr

        target = target_saved.to(
            device=hidden.device,
            dtype=torch.float32,
        )[None, :]

        if target_norm == "preserve_original":
            target = (
                target
                / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                * z_current.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
            )
        elif target_norm != "raw":
            raise ValueError(
                f"Unsupported target norm mode: {target_norm}"
            )

        z_new = (
            (1.0 - effective_strength) * z_current
            + effective_strength * target
        )
        hs_new = center + 0.5 * z_new
        hr_new = center - 0.5 * z_new

        hidden_new[:, data.subject_index, :] = hs_new.to(
            hidden.dtype
        )
        hidden_new[:, data.reference_index, :] = hr_new.to(
            hidden.dtype
        )

        state["patched"] = True
        state["current_difference_norm"] = float(
            z_current.norm(dim=-1).mean().item()
        )
        state["target_difference_norm"] = float(
            target.norm(dim=-1).mean().item()
        )
        state["new_difference_norm"] = float(
            z_new.norm(dim=-1).mean().item()
        )
        state["applied_delta_norm"] = float(
            (z_new - z_current).norm(dim=-1).mean().item()
        )
        state["current_target_cosine"] = float(
            F.cosine_similarity(
                z_current,
                target,
                dim=-1,
            ).mean().item()
        )

        if isinstance(output, tuple):
            return (hidden_new,) + output[1:]
        if isinstance(output, list):
            return [hidden_new] + list(output[1:])
        return hidden_new

    handle = decoder_layer.register_forward_hook(hook)
    try:
        text = generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
    finally:
        handle.remove()

    if enabled and not state["patched"]:
        raise RuntimeError(
            "Swap-consistent hook was registered but did not patch prefill"
        )

    metadata = {
        "patch_enabled": enabled,
        "patch_applied": bool(state["patched"]),
        "patch_confidence_mode": confidence_mode,
        "patch_confidence": float(confidence),
        "swap_strength": float(swap_strength),
        "effective_swap_strength": (
            float(effective_strength) if enabled else 0.0
        ),
        "target_norm": target_norm,
        **state,
    }
    return text, metadata


def accuracy(
    rows: Iterable[Dict[str, Any]],
    key: str = "correct",
) -> Optional[float]:
    rows = list(rows)
    if not rows:
        return None
    return float(
        np.mean([bool(row.get(key, False)) for row in rows])
    )


def paired_comparison(
    baseline_by_sid: Dict[int, Dict[str, Any]],
    patched_by_sid: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    paired = [
        sid
        for sid in sorted(
            set(baseline_by_sid) & set(patched_by_sid)
        )
        if baseline_by_sid[sid].get("prediction")
        and patched_by_sid[sid].get("prediction")
    ]

    fixed = broken = changed = both_correct = both_wrong = 0
    for sid in paired:
        baseline = baseline_by_sid[sid]
        patched = patched_by_sid[sid]
        changed += int(
            baseline.get("prediction")
            != patched.get("prediction")
        )
        if (
            not baseline.get("correct")
            and patched.get("correct")
        ):
            fixed += 1
        elif (
            baseline.get("correct")
            and not patched.get("correct")
        ):
            broken += 1
        elif (
            baseline.get("correct")
            and patched.get("correct")
        ):
            both_correct += 1
        else:
            both_wrong += 1

    baseline_accuracy = accuracy(
        baseline_by_sid[sid] for sid in paired
    )
    patched_accuracy = accuracy(
        patched_by_sid[sid] for sid in paired
    )

    return {
        "n": len(paired),
        "baseline_accuracy": baseline_accuracy,
        "patched_accuracy": patched_accuracy,
        "absolute_change": (
            patched_accuracy - baseline_accuracy
            if baseline_accuracy is not None
            and patched_accuracy is not None
            else None
        ),
        "fixed": fixed,
        "broken": broken,
        "net_fixed_minus_broken": fixed - broken,
        "changed_predictions": changed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def summarize(
    baseline_rows: List[Dict[str, Any]],
    patched_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_by_sid = {
        int(row["sid"]): row
        for row in baseline_rows
        if "sid" in row and "error" not in row
    }
    patched_by_sid = {
        int(row["sid"]): row
        for row in patched_rows
        if "sid" in row and "error" not in row
    }

    baseline_valid = [
        row
        for row in baseline_by_sid.values()
        if row.get("prediction")
    ]
    patched_valid = [
        row
        for row in patched_by_sid.values()
        if row.get("prediction")
    ]
    original_ground_valid = [
        row
        for row in baseline_by_sid.values()
        if row.get("original_grounding_prediction")
    ]
    swapped_ground_valid = [
        row
        for row in baseline_by_sid.values()
        if row.get("swapped_grounding_prediction")
    ]
    average_ground_valid = [
        row
        for row in baseline_by_sid.values()
        if row.get("average_grounding_prediction")
    ]

    pair_stats = paired_comparison(
        baseline_by_sid,
        patched_by_sid,
    )

    baseline_wrong = [
        row
        for row in original_ground_valid
        if row.get("prediction")
        and not bool(row.get("correct"))
    ]

    common = [
        sid
        for sid in sorted(
            set(baseline_by_sid) & set(patched_by_sid)
        )
        if baseline_by_sid[sid].get("prediction")
        and patched_by_sid[sid].get("prediction")
    ]
    relation_groups: Dict[str, List[int]] = defaultdict(list)
    for sid in common:
        relation_groups[
            str(baseline_by_sid[sid].get("gt"))
        ].append(sid)

    per_relation: Dict[str, Dict[str, Any]] = {}
    for relation, sids in sorted(relation_groups.items()):
        per_relation[relation] = {
            "n": len(sids),
            "baseline_accuracy": accuracy(
                baseline_by_sid[sid] for sid in sids
            ),
            "original_grounding_accuracy": accuracy(
                (
                    baseline_by_sid[sid]
                    for sid in sids
                    if baseline_by_sid[sid].get(
                        "original_grounding_prediction"
                    )
                ),
                key="original_grounding_correct",
            ),
            "swapped_grounding_accuracy": accuracy(
                (
                    baseline_by_sid[sid]
                    for sid in sids
                    if baseline_by_sid[sid].get(
                        "swapped_grounding_prediction"
                    )
                ),
                key="swapped_grounding_correct",
            ),
            "average_grounding_accuracy": accuracy(
                (
                    baseline_by_sid[sid]
                    for sid in sids
                    if baseline_by_sid[sid].get(
                        "average_grounding_prediction"
                    )
                ),
                key="average_grounding_correct",
            ),
            "patched_accuracy": accuracy(
                patched_by_sid[sid] for sid in sids
            ),
        }

    return {
        "n_baseline_valid": len(baseline_valid),
        "n_patched_valid": len(patched_valid),
        "n_original_grounding_valid": len(
            original_ground_valid
        ),
        "n_swapped_grounding_valid": len(
            swapped_ground_valid
        ),
        "n_average_grounding_valid": len(
            average_ground_valid
        ),

        "baseline_accuracy": accuracy(baseline_valid),
        "original_grounding_accuracy": accuracy(
            original_ground_valid,
            key="original_grounding_correct",
        ),
        "swapped_grounding_accuracy": accuracy(
            swapped_ground_valid,
            key="swapped_grounding_correct",
        ),
        "average_grounding_accuracy": accuracy(
            average_ground_valid,
            key="average_grounding_correct",
        ),
        "patched_accuracy": accuracy(patched_valid),

        "original_grounding_on_baseline_wrong": accuracy(
            baseline_wrong,
            key="original_grounding_correct",
        ),
        "n_original_grounding_on_baseline_wrong": len(
            baseline_wrong
        ),

        "paired": pair_stats,
        "baseline_parse_failures": (
            len(baseline_by_sid) - len(baseline_valid)
        ),
        "patched_parse_failures": (
            len(patched_by_sid) - len(patched_valid)
        ),

        "mean_swap_antisymmetry_cosine": (
            float(np.mean([
                float(row["swap_antisymmetry_cosine"])
                for row in baseline_valid
            ]))
            if baseline_valid else None
        ),
        "mean_role_bias_ratio": (
            float(np.mean([
                float(row["role_bias_ratio"])
                for row in baseline_valid
            ]))
            if baseline_valid else None
        ),
        "mean_patch_confidence": (
            float(np.mean([
                float(row.get("patch_confidence", 0.0))
                for row in patched_valid
            ]))
            if patched_valid else None
        ),
        "mean_effective_swap_strength": (
            float(np.mean([
                float(
                    row.get(
                        "effective_swap_strength",
                        0.0,
                    )
                )
                for row in patched_valid
            ]))
            if patched_valid else None
        ),
        "per_relation": per_relation,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print(
        "BASELINE VS SWAPPED-TEXT-TOKEN CONSISTENT GENERATION"
    )
    print("=" * 96)

    metrics = [
        (
            "baseline generation",
            "baseline_accuracy",
            "n_baseline_valid",
        ),
        (
            "original centroid",
            "original_grounding_accuracy",
            "n_original_grounding_valid",
        ),
        (
            "swapped centroid",
            "swapped_grounding_accuracy",
            "n_swapped_grounding_valid",
        ),
        (
            "averaged centroid",
            "average_grounding_accuracy",
            "n_average_grounding_valid",
        ),
        (
            "swap-token patched",
            "patched_accuracy",
            "n_patched_valid",
        ),
    ]
    for index, (name, key, n_key) in enumerate(metrics, 1):
        value = summary.get(key)
        if value is None:
            print(f"{index}) {name:23s}: n/a")
        else:
            print(
                f"{index}) {name:23s}: {value:.4f} "
                f"(n={summary.get(n_key)})"
            )

    wrong_acc = summary.get(
        "original_grounding_on_baseline_wrong"
    )
    print("\nOriginal centroid complementarity:")
    if wrong_acc is not None:
        print(
            f"grounding on baseline-wrong:   "
            f"{wrong_acc:.4f} "
            f"(n={summary.get('n_original_grounding_on_baseline_wrong')})"
        )
    else:
        print("grounding on baseline-wrong:   n/a")

    paired = summary["paired"]
    print("\nPaired baseline/patched comparison:")
    print(f"paired valid:                 {paired.get('n')}")
    print(
        f"paired baseline:              "
        f"{paired['baseline_accuracy']:.4f}"
        if paired.get("baseline_accuracy") is not None
        else "paired baseline:              n/a"
    )
    print(
        f"paired patched:               "
        f"{paired['patched_accuracy']:.4f}"
        if paired.get("patched_accuracy") is not None
        else "paired patched:               n/a"
    )
    print(
        f"paired absolute change:       "
        f"{paired['absolute_change']:+.4f}"
        if paired.get("absolute_change") is not None
        else "paired absolute change:       n/a"
    )
    print(f"fixed:                        {paired.get('fixed')}")
    print(f"broken:                       {paired.get('broken')}")
    print(
        f"net fixed-broken:             "
        f"{paired.get('net_fixed_minus_broken'):+d}"
    )
    print(
        f"prediction changed:           "
        f"{paired.get('changed_predictions')}"
    )
    print(
        f"baseline parse fail:          "
        f"{summary.get('baseline_parse_failures')}"
    )
    print(
        f"patched parse fail:           "
        f"{summary.get('patched_parse_failures')}"
    )
    print(
        f"mean swap antisymmetry cosine:"
        f"{summary.get('mean_swap_antisymmetry_cosine')}"
    )
    print(
        f"mean role-bias ratio:         "
        f"{summary.get('mean_role_bias_ratio')}"
    )
    print(
        f"mean patch confidence:        "
        f"{summary.get('mean_patch_confidence')}"
    )
    print(
        f"mean effective swap strength: "
        f"{summary.get('mean_effective_swap_strength')}"
    )

    print("\nPer relation on paired samples:")
    for relation, stats in summary.get(
        "per_relation",
        {},
    ).items():
        print(
            f"  {relation:6s} n={stats['n']:4d} | "
            f"base={stats['baseline_accuracy']:.4f} | "
            f"orig-cent={stats['original_grounding_accuracy']:.4f} | "
            f"swap-cent={stats['swapped_grounding_accuracy']:.4f} | "
            f"avg-cent={stats['average_grounding_accuracy']:.4f} | "
            f"patched={stats['patched_accuracy']:.4f}"
        )


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.swap_strength < 0.0 or args.swap_strength > 2.0:
        raise ValueError("--swap-strength must be in [0, 2]")
    if not (0.0 <= args.min_patch_confidence <= 1.0):
        raise ValueError(
            "--min-patch-confidence must be in [0, 1]"
        )
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
            f"Prompt file {prompt_path} is missing "
            f"{len(missing_ids)} record IDs; first={missing_ids[:10]}"
        )

    if args.model not in module.SPECS:
        raise ValueError(
            f"Model {args.model!r} not found in "
            "extract_two_object_relation_states.SPECS"
        )
    spec = module.SPECS[args.model]

    if args.layer == "auto":
        if args.model not in AUTO_LAYERS:
            raise ValueError(
                f"No auto layer for {args.model!r}; "
                "pass --layer explicitly"
            )
        layer = AUTO_LAYERS[args.model]
    else:
        layer = int(args.layer)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "swap_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.jsonl"
    patched_path = output_dir / "patched.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    run_config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "model": args.model,
        "repo_id": spec.repo_id,
        "transformers_version": transformers.__version__,
        "prompt_jsonl": str(prompt_path),
        "layer": layer,
        "temperature": args.temperature,
        "swap_strength": args.swap_strength,
        "target_norm": args.target_norm,
        "patch_confidence_mode": args.patch_confidence_mode,
        "min_patch_confidence": args.min_patch_confidence,
        "max_new_tokens": args.max_new_tokens,
        "n_records": len(records),
        "audit": audit,
        "uses_gt_for_intervention": False,
        "changes_original_question": False,
        "edits_output_logits": False,
        "updates_model_weights": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(
            run_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )
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

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
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

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    if not (0 <= layer < len(decoder_layers)):
        raise ValueError(
            f"Requested layer={layer}, decoder has "
            f"{len(decoder_layers)} layers"
        )

    print(
        f"Standard prompt file: {prompt_path} "
        f"(rows={len(prompt_rows)})"
    )
    print(
        f"Resolved decoder layers: {decoder_path}, "
        f"n={len(decoder_layers)}, layer={layer}"
    )
    print(
        f"Swap patch: strength={args.swap_strength}, "
        f"target_norm={args.target_norm}, "
        f"confidence={args.patch_confidence_mode}, "
        f"threshold={args.min_patch_confidence}"
    )

    baseline_existing = [
        row
        for row in read_jsonl(baseline_path)
        if "sid" in row and "error" not in row
    ]
    patched_existing = [
        row
        for row in read_jsonl(patched_path)
        if "sid" in row and "error" not in row
    ]
    baseline_done = {
        int(row["sid"]) for row in baseline_existing
    }
    patched_done = {
        int(row["sid"]) for row in patched_existing
    }

    baseline_seen = len(baseline_existing)
    baseline_correct_count = sum(
        bool(row.get("correct"))
        for row in baseline_existing
    )
    patched_seen = len(patched_existing)
    patched_correct_count = sum(
        bool(row.get("correct"))
        for row in patched_existing
    )

    started = time.time()

    try:
        if args.only in ("both", "baseline"):
            print(
                "\nPASS 1/2: baseline generation + original/swapped "
                "text-token capture"
            )
            for record in tqdm(
                records,
                desc=f"baseline:{args.dataset}:{args.model}",
            ):
                sid = int(record.sid)
                data_path = data_dir / f"{sid}.npz"
                if sid in baseline_done and data_path.exists():
                    continue

                original_batch = None
                swapped_batch = None
                image = None
                try:
                    prompt_row = prompt_rows[sid]
                    (
                        original_batch,
                        subject,
                        reference,
                        question_text,
                        raw_question,
                        answer_raw,
                        image,
                    ) = make_batch(
                        processor,
                        record,
                        prompt_row,
                        device,
                    )

                    baseline_text = generate_text(
                        model,
                        processor,
                        original_batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                    baseline_prediction = normalize_relation(
                        baseline_text
                    )
                    gt = normalize_relation(answer_raw)

                    swapped_question = build_swapped_question(
                        subject,
                        reference,
                    )
                    swapped_batch = make_question_batch(
                        processor=processor,
                        image=image,
                        question_text=swapped_question,
                        device=device,
                    )

                    data = capture_swap_data(
                        model=model,
                        processor=processor,
                        original_batch=original_batch,
                        swapped_batch=swapped_batch,
                        sid=sid,
                        subject=subject,
                        reference=reference,
                        layer=layer,
                        temperature=args.temperature,
                    )
                    save_swap_data(data_path, data)

                    row = {
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "gt": gt,
                        "prediction": baseline_prediction,
                        "correct": bool(
                            gt is not None
                            and baseline_prediction is not None
                            and gt == baseline_prediction
                        ),
                        "generated_text": baseline_text,
                        "question": question_text,
                        "swapped_question": swapped_question,
                        "raw_question": raw_question,
                        "standard_answer_raw": answer_raw,
                        "layer": layer,

                        "original_grounding_prediction": (
                            data.original_grounding_prediction
                            or None
                        ),
                        "original_grounding_correct": bool(
                            gt is not None
                            and data.original_grounding_prediction
                            and gt
                            == data.original_grounding_prediction
                        ),
                        "swapped_grounding_prediction": (
                            data.swapped_grounding_prediction
                            or None
                        ),
                        "swapped_grounding_correct": bool(
                            gt is not None
                            and data.swapped_grounding_prediction
                            and gt
                            == data.swapped_grounding_prediction
                        ),
                        "average_grounding_prediction": (
                            data.average_grounding_prediction
                            or None
                        ),
                        "average_grounding_correct": bool(
                            gt is not None
                            and data.average_grounding_prediction
                            and gt
                            == data.average_grounding_prediction
                        ),

                        "original_subject_centroid": [
                            data.original_subject_x,
                            data.original_subject_y,
                        ],
                        "original_reference_centroid": [
                            data.original_reference_x,
                            data.original_reference_y,
                        ],
                        "swapped_subject_centroid": [
                            data.swapped_subject_x,
                            data.swapped_subject_y,
                        ],
                        "swapped_reference_centroid": [
                            data.swapped_reference_x,
                            data.swapped_reference_y,
                        ],
                        "average_subject_centroid": [
                            data.average_subject_x,
                            data.average_subject_y,
                        ],
                        "average_reference_centroid": [
                            data.average_reference_x,
                            data.average_reference_y,
                        ],

                        "original_map_separation": (
                            data.original_map_separation
                        ),
                        "swapped_map_separation": (
                            data.swapped_map_separation
                        ),
                        "swap_antisymmetry_cosine": (
                            data.swap_antisymmetry_cosine
                        ),
                        "original_norm": data.original_norm,
                        "swapped_role_norm": (
                            data.swapped_role_norm
                        ),
                        "consistent_norm": data.consistent_norm,
                        "role_bias_norm": data.role_bias_norm,
                        "role_bias_ratio": data.role_bias_ratio,
                    }
                    append_jsonl(baseline_path, row)
                    baseline_done.add(sid)
                    baseline_seen += 1
                    baseline_correct_count += int(
                        row["correct"]
                    )

                    if should_print_sample(
                        baseline_seen,
                        args.print_every,
                    ):
                        tqdm.write(
                            f"\n[BASE {baseline_seen}/{len(records)}] "
                            f"sid={sid} | {subject} -> {reference}\n"
                            f"  original   : "
                            f"{one_line(question_text)}\n"
                            f"  swapped    : "
                            f"{one_line(swapped_question)}\n"
                            f"  gt         : {gt}\n"
                            f"  generation : {baseline_text!r}\n"
                            f"  pred       : {baseline_prediction}\n"
                            f"  correct    : {int(row['correct'])}\n"
                            f"  acc        : "
                            f"{baseline_correct_count}/{baseline_seen}="
                            f"{baseline_correct_count / baseline_seen:.4f}\n"
                            f"  centroids  : "
                            f"orig={data.original_grounding_prediction}, "
                            f"swap={data.swapped_grounding_prediction}, "
                            f"avg={data.average_grounding_prediction}\n"
                            f"  swap_cos={data.swap_antisymmetry_cosine:+.3f} | "
                            f"role_bias_ratio={data.role_bias_ratio:.3f}"
                        )

                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "baseline",
                            "sid": sid,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback_tail": (
                                traceback.format_exc().splitlines()[
                                    -12:
                                ]
                            ),
                        },
                    )
                    tqdm.write(
                        f"\n[BASE ERROR] sid={sid} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    if original_batch is not None:
                        del original_batch
                    if swapped_batch is not None:
                        del swapped_batch
                    if image is not None:
                        del image

        if args.only in ("both", "patched"):
            print(
                "\nPASS 2/2: original-prompt generation with "
                "swap-consistent object-token patch"
            )
            baseline_by_sid = {
                int(row["sid"]): row
                for row in read_jsonl(baseline_path)
                if "sid" in row and "error" not in row
            }

            fixed_running = 0
            broken_running = 0
            for row in patched_existing:
                baseline = baseline_by_sid.get(
                    int(row["sid"])
                )
                if baseline is None:
                    continue
                if (
                    not baseline.get("correct")
                    and row.get("correct")
                ):
                    fixed_running += 1
                elif (
                    baseline.get("correct")
                    and not row.get("correct")
                ):
                    broken_running += 1

            for record in tqdm(
                records,
                desc=f"patched:{args.dataset}:{args.model}",
            ):
                sid = int(record.sid)
                if sid in patched_done:
                    continue

                data_path = data_dir / f"{sid}.npz"
                if not data_path.exists():
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "patched",
                            "sid": sid,
                            "error_type": "FileNotFoundError",
                            "error": (
                                f"Missing swap data: {data_path}"
                            ),
                        },
                    )
                    continue

                batch = None
                image = None
                try:
                    data = load_swap_data(data_path)
                    prompt_row = prompt_rows[sid]
                    (
                        batch,
                        subject,
                        reference,
                        question_text,
                        raw_question,
                        answer_raw,
                        image,
                    ) = make_batch(
                        processor,
                        record,
                        prompt_row,
                        device,
                    )

                    patched_text, metadata = (
                        swap_patched_generate(
                            model=model,
                            processor=processor,
                            batch=batch,
                            decoder_layer=decoder_layers[layer],
                            data=data,
                            swap_strength=args.swap_strength,
                            target_norm=args.target_norm,
                            confidence_mode=(
                                args.patch_confidence_mode
                            ),
                            min_confidence=(
                                args.min_patch_confidence
                            ),
                            max_new_tokens=(
                                args.max_new_tokens
                            ),
                        )
                    )
                    patched_prediction = normalize_relation(
                        patched_text
                    )
                    gt = normalize_relation(answer_raw)

                    row = {
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "gt": gt,
                        "prediction": patched_prediction,
                        "correct": bool(
                            gt is not None
                            and patched_prediction is not None
                            and gt == patched_prediction
                        ),
                        "generated_text": patched_text,
                        "question": question_text,
                        "raw_question": raw_question,
                        "grounding_prediction": (
                            data.original_grounding_prediction
                            or None
                        ),
                        "swap_antisymmetry_cosine": (
                            data.swap_antisymmetry_cosine
                        ),
                        "role_bias_ratio": (
                            data.role_bias_ratio
                        ),
                        **metadata,
                    }
                    append_jsonl(patched_path, row)
                    patched_done.add(sid)
                    patched_seen += 1
                    patched_correct_count += int(
                        row["correct"]
                    )

                    baseline = baseline_by_sid.get(sid)
                    baseline_prediction = (
                        baseline.get("prediction")
                        if baseline is not None
                        else None
                    )
                    status = "NO_BASELINE"
                    if baseline is not None:
                        if (
                            not baseline.get("correct")
                            and row.get("correct")
                        ):
                            status = "FIXED"
                            fixed_running += 1
                        elif (
                            baseline.get("correct")
                            and not row.get("correct")
                        ):
                            status = "BROKEN"
                            broken_running += 1
                        elif (
                            baseline.get("correct")
                            and row.get("correct")
                        ):
                            status = "UNCHANGED_CORRECT"
                        else:
                            status = "UNCHANGED_WRONG"

                    if should_print_sample(
                        patched_seen,
                        args.print_every,
                    ):
                        tqdm.write(
                            f"\n[PATCH {patched_seen}/{len(records)}] "
                            f"sid={sid} | {subject} -> {reference}\n"
                            f"  question   : "
                            f"{one_line(question_text)}\n"
                            f"  gt         : {gt}\n"
                            f"  generation : {patched_text!r}\n"
                            f"  pred       : {patched_prediction}\n"
                            f"  correct    : {int(row['correct'])}\n"
                            f"  acc        : "
                            f"{patched_correct_count}/{patched_seen}="
                            f"{patched_correct_count / patched_seen:.4f}\n"
                            f"  base={baseline_prediction} -> "
                            f"patched={patched_prediction} | "
                            f"{status} | fixed={fixed_running} | "
                            f"broken={broken_running} | "
                            f"net={fixed_running - broken_running}\n"
                            f"  swap_cos="
                            f"{data.swap_antisymmetry_cosine:+.3f} | "
                            f"role_bias_ratio="
                            f"{data.role_bias_ratio:.3f} | "
                            f"confidence="
                            f"{metadata['patch_confidence']:.3f} | "
                            f"effective_strength="
                            f"{metadata['effective_swap_strength']:.3f}"
                        )

                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "patched",
                            "sid": sid,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback_tail": (
                                traceback.format_exc().splitlines()[
                                    -12:
                                ]
                            ),
                        },
                    )
                    tqdm.write(
                        f"\n[PATCH ERROR] sid={sid} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    if batch is not None:
                        del batch
                    if image is not None:
                        del image

        baseline_rows = read_jsonl(baseline_path)
        patched_rows = read_jsonl(patched_path)

        if baseline_rows and patched_rows:
            summary = summarize(
                baseline_rows,
                patched_rows,
            )
            summary["config"] = run_config
            summary["elapsed_minutes"] = (
                time.time() - started
            ) / 60.0
            summary_path.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print_summary(summary)
            print(f"\nSaved summary: {summary_path}")
        else:
            print(
                "Completed requested stage. "
                f"baseline_rows={len(baseline_rows)}, "
                f"patched_rows={len(patched_rows)}"
            )

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
