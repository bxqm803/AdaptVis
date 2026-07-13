#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free latent spatial-carrier generation.

This script evaluates four outputs on the same standard two-object questions:

1) Baseline generation
   The frozen VLM answers the dataset question normally.

2) Centroid grounding
   Object-word hidden states are matched to visual-token hidden states by cosine
   similarity. Their weighted visual centroids yield dx/dy and a direct spatial
   diagnosis.

3) Carrier control
   The same model generates from a prompt with a neutral trailing marker
   ("Spatial evidence:") but receives no hidden-state intervention. This isolates
   the effect of merely changing the prompt.

4) Spatial-carrier generation
   At the selected decoder layer, the visual-token hidden states are regressed
   against their known x/y grid coordinates, producing two per-image latent axes:
   a_x for increasing x and a_y for increasing y. The centroid displacement
   (dx, dy) is converted into a latent relation vector dx*a_x + dy*a_y and added
   only to the neutral carrier token. Later decoder layers and the LM head then
   generate normally.

No model parameters or probes are trained. No per-sample GT label is used in
centroid extraction, basis construction, carrier injection, or generation.
Ground truth is read only after generation for evaluation.
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


SCRIPT_VERSION = "latent-spatial-carrier-generation-v1"

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
class CarrierData:
    sid: int
    layer: int
    subject_index: int
    reference_index: int
    baseline_input_length: int
    spatial_vector: np.ndarray
    map_separation: float
    subject_entropy_confidence: float
    reference_entropy_confidence: float
    subject_peak_weight: float
    reference_peak_weight: float
    n_visual_tokens: int
    subject_x: float
    subject_y: float
    reference_x: float
    reference_y: float
    delta_x: float
    delta_y: float
    axis_confidence: float
    grounding_prediction: str
    basis_r2: float
    basis_axis_cosine: float
    basis_x_norm: float
    basis_y_norm: float
    spatial_vector_norm: float


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
        help="Zero-based decoder block index for centroid/basis extraction.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for object-token/visual-token grounding.",
    )
    p.add_argument(
        "--basis-ridge",
        type=float,
        default=1e-4,
        help="Ridge term for the per-image x/y latent-basis regression.",
    )
    p.add_argument(
        "--carrier-marker",
        default="Spatial evidence:",
        help="Neutral text marker whose final token receives the spatial vector.",
    )
    p.add_argument(
        "--carrier-strength",
        type=float,
        default=0.5,
        help=(
            "Maximum injected-vector norm as a fraction of the carrier-token "
            "hidden-state norm before label-free confidence scaling."
        ),
    )
    p.add_argument(
        "--carrier-confidence-mode",
        default="separation_axis",
        choices=["none", "separation", "axis", "separation_axis"],
        help="Label-free confidence used to scale carrier injection.",
    )
    p.add_argument(
        "--min-carrier-confidence",
        type=float,
        default=0.0,
        help="Disable hidden-state injection below this confidence.",
    )
    p.add_argument(
        "--renorm-carrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve the carrier token's original L2 norm after injection.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one detailed record every N completed records; 0 disables.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--only",
        choices=["both", "baseline", "carrier"],
        default="both",
        help=(
            "Run both stages, only baseline/latent capture, or only carrier "
            "control+guided generation from saved carrier data."
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


def build_carrier_data(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    sid: int,
    subject: str,
    reference: str,
    layer: int,
    temperature: float,
    basis_ridge: float,
) -> CarrierData:
    if temperature <= 0:
        raise ValueError("--temperature must be positive")
    if basis_ridge < 0:
        raise ValueError("--basis-ridge must be non-negative")

    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])
    visual_indices = resolve_visual_indices(model, processor, batch, input_ids)

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
            f"Requested layer={layer}, but model has {n_blocks} decoder blocks"
        )

    hidden = states[layer + 1][0].float()
    if int(hidden.shape[0]) != len(input_ids):
        raise RuntimeError(
            f"Token/hidden mismatch: input={len(input_ids)}, "
            f"hidden={int(hidden.shape[0])}"
        )

    index_tensor = torch.as_tensor(
        visual_indices,
        device=hidden.device,
        dtype=torch.long,
    )
    visual = hidden[index_tensor]
    hs = hidden[subject_index]
    hr = hidden[reference_index]

    # Object-token -> visual-token grounding.
    visual_norm = F.normalize(visual, dim=-1)
    logits_s = torch.matmul(visual_norm, F.normalize(hs, dim=-1))
    logits_r = torch.matmul(visual_norm, F.normalize(hr, dim=-1))
    weights_s = torch.softmax(logits_s / float(temperature), dim=0)
    weights_r = torch.softmax(logits_r / float(temperature), dim=0)

    separation = 0.5 * torch.sum(torch.abs(weights_s - weights_r))
    ent_s = entropy_confidence(weights_s)
    ent_r = entropy_confidence(weights_r)

    coords = visual_coordinates(
        model,
        batch,
        len(visual_indices),
        hidden.device,
    )
    if coords is None or int(coords.shape[0]) != len(visual_indices):
        raise RuntimeError(
            f"Could not construct visual coordinates for {len(visual_indices)} tokens"
        )
    coords = coords.float()

    center_s = torch.sum(weights_s[:, None] * coords, dim=0)
    center_r = torch.sum(weights_r[:, None] * coords, dim=0)
    subject_x, subject_y = [float(x) for x in center_s.tolist()]
    reference_x, reference_y = [float(x) for x in center_r.tolist()]
    delta_x = subject_x - reference_x
    delta_y = subject_y - reference_y
    grounding_prediction, axis_confidence = relation_from_centroids(
        delta_x,
        delta_y,
    )

    # Per-image latent position basis:
    # visual_j ~= mean_visual + (x_j-mean_x)*a_x + (y_j-mean_y)*a_y.
    coord_centered = coords - coords.mean(dim=0, keepdim=True)
    visual_centered = visual - visual.mean(dim=0, keepdim=True)
    gram = coord_centered.T @ coord_centered
    gram = gram + float(basis_ridge) * torch.eye(
        2,
        device=gram.device,
        dtype=gram.dtype,
    )
    rhs = coord_centered.T @ visual_centered
    basis = torch.linalg.solve(gram, rhs)  # [2, hidden_dim]
    axis_x = basis[0]
    axis_y = basis[1]

    spatial_vector = (
        float(delta_x) * axis_x
        + float(delta_y) * axis_y
    )

    fitted = coord_centered @ basis
    residual_ss = torch.sum((visual_centered - fitted) ** 2)
    total_ss = torch.sum(visual_centered ** 2).clamp_min(1e-12)
    basis_r2 = float((1.0 - residual_ss / total_ss).item())
    basis_axis_cosine = float(
        F.cosine_similarity(axis_x[None], axis_y[None], dim=-1).item()
    )
    basis_x_norm = float(axis_x.norm().item())
    basis_y_norm = float(axis_y.norm().item())
    spatial_vector_norm = float(spatial_vector.norm().item())

    data = CarrierData(
        sid=sid,
        layer=layer,
        subject_index=subject_index,
        reference_index=reference_index,
        baseline_input_length=len(input_ids),
        spatial_vector=(
            spatial_vector.detach().cpu().numpy().astype(np.float16)
        ),
        map_separation=float(separation.item()),
        subject_entropy_confidence=float(ent_s.item()),
        reference_entropy_confidence=float(ent_r.item()),
        subject_peak_weight=float(weights_s.max().item()),
        reference_peak_weight=float(weights_r.max().item()),
        n_visual_tokens=len(visual_indices),
        subject_x=subject_x,
        subject_y=subject_y,
        reference_x=reference_x,
        reference_y=reference_y,
        delta_x=delta_x,
        delta_y=delta_y,
        axis_confidence=axis_confidence,
        grounding_prediction=grounding_prediction,
        basis_r2=basis_r2,
        basis_axis_cosine=basis_axis_cosine,
        basis_x_norm=basis_x_norm,
        basis_y_norm=basis_y_norm,
        spatial_vector_norm=spatial_vector_norm,
    )

    del outputs, states, hidden, visual, weights_s, weights_r
    return data

def save_carrier_data(path: Path, data: CarrierData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            sid=np.asarray(data.sid, dtype=np.int64),
            layer=np.asarray(data.layer, dtype=np.int32),
            subject_index=np.asarray(data.subject_index, dtype=np.int32),
            reference_index=np.asarray(data.reference_index, dtype=np.int32),
            baseline_input_length=np.asarray(
                data.baseline_input_length,
                dtype=np.int32,
            ),
            spatial_vector=data.spatial_vector,
            map_separation=np.asarray(data.map_separation, dtype=np.float32),
            subject_entropy_confidence=np.asarray(
                data.subject_entropy_confidence,
                dtype=np.float32,
            ),
            reference_entropy_confidence=np.asarray(
                data.reference_entropy_confidence,
                dtype=np.float32,
            ),
            subject_peak_weight=np.asarray(
                data.subject_peak_weight,
                dtype=np.float32,
            ),
            reference_peak_weight=np.asarray(
                data.reference_peak_weight,
                dtype=np.float32,
            ),
            n_visual_tokens=np.asarray(data.n_visual_tokens, dtype=np.int32),
            subject_x=np.asarray(data.subject_x, dtype=np.float32),
            subject_y=np.asarray(data.subject_y, dtype=np.float32),
            reference_x=np.asarray(data.reference_x, dtype=np.float32),
            reference_y=np.asarray(data.reference_y, dtype=np.float32),
            delta_x=np.asarray(data.delta_x, dtype=np.float32),
            delta_y=np.asarray(data.delta_y, dtype=np.float32),
            axis_confidence=np.asarray(
                data.axis_confidence,
                dtype=np.float32,
            ),
            grounding_prediction=np.asarray(
                data.grounding_prediction,
                dtype="<U8",
            ),
            basis_r2=np.asarray(data.basis_r2, dtype=np.float32),
            basis_axis_cosine=np.asarray(
                data.basis_axis_cosine,
                dtype=np.float32,
            ),
            basis_x_norm=np.asarray(data.basis_x_norm, dtype=np.float32),
            basis_y_norm=np.asarray(data.basis_y_norm, dtype=np.float32),
            spatial_vector_norm=np.asarray(
                data.spatial_vector_norm,
                dtype=np.float32,
            ),
        )
    os.replace(tmp, path)

def load_carrier_data(path: Path) -> CarrierData:
    with np.load(path, allow_pickle=False) as z:
        return CarrierData(
            sid=int(z["sid"].item()),
            layer=int(z["layer"].item()),
            subject_index=int(z["subject_index"].item()),
            reference_index=int(z["reference_index"].item()),
            baseline_input_length=int(z["baseline_input_length"].item()),
            spatial_vector=z["spatial_vector"],
            map_separation=float(z["map_separation"].item()),
            subject_entropy_confidence=float(
                z["subject_entropy_confidence"].item()
            ),
            reference_entropy_confidence=float(
                z["reference_entropy_confidence"].item()
            ),
            subject_peak_weight=float(z["subject_peak_weight"].item()),
            reference_peak_weight=float(z["reference_peak_weight"].item()),
            n_visual_tokens=int(z["n_visual_tokens"].item()),
            subject_x=float(z["subject_x"].item()),
            subject_y=float(z["subject_y"].item()),
            reference_x=float(z["reference_x"].item()),
            reference_y=float(z["reference_y"].item()),
            delta_x=float(z["delta_x"].item()),
            delta_y=float(z["delta_y"].item()),
            axis_confidence=float(z["axis_confidence"].item()),
            grounding_prediction=str(z["grounding_prediction"].item()),
            basis_r2=float(z["basis_r2"].item()),
            basis_axis_cosine=float(z["basis_axis_cosine"].item()),
            basis_x_norm=float(z["basis_x_norm"].item()),
            basis_y_norm=float(z["basis_y_norm"].item()),
            spatial_vector_norm=float(z["spatial_vector_norm"].item()),
        )

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





def carrier_question(
    standard_question: str,
    marker: str,
) -> str:
    marker = str(marker).strip()
    if not marker:
        raise ValueError("--carrier-marker must not be empty")
    return f"{str(standard_question).strip()}\n{marker}"


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


def locate_carrier_index(
    tokenizer: Any,
    input_ids: Sequence[int],
    marker: str,
) -> Tuple[int, Tuple[int, int]]:
    marker = str(marker).strip()
    variants = [
        marker,
        " " + marker,
        "\n" + marker,
    ]
    spans: List[Tuple[int, int]] = []
    seen = set()
    for variant in variants:
        ids = tokenizer_ids(tokenizer, variant)
        key = tuple(ids)
        if not ids or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, ids):
            spans.append((start, start + len(ids) - 1))
    spans = sorted(set(spans))
    if not spans:
        raise ValueError(
            f"Could not locate carrier marker {marker!r} in tokenized prompt"
        )
    span = max(spans, key=lambda item: item[1])
    return int(span[1]), span


def carrier_confidence(
    data: CarrierData,
    mode: str,
) -> float:
    separation = float(np.clip(data.map_separation, 0.0, 1.0))
    axis = float(data.axis_confidence)
    axis = float(np.clip(axis, 0.0, 1.0)) if np.isfinite(axis) else 0.0

    if mode == "none":
        return 1.0
    if mode == "separation":
        return separation
    if mode == "axis":
        return axis
    if mode == "separation_axis":
        return separation * axis
    raise ValueError(f"Unsupported carrier confidence mode: {mode}")


def carrier_guided_generate(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    decoder_layer: Any,
    data: CarrierData,
    carrier_index: int,
    carrier_strength: float,
    confidence_mode: str,
    min_confidence: float,
    renorm_carrier: bool,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    confidence = carrier_confidence(data, confidence_mode)
    enabled = bool(
        confidence >= float(min_confidence)
        and float(carrier_strength) != 0.0
        and float(data.spatial_vector_norm) > 1e-8
    )
    effective_strength = (
        float(carrier_strength) * float(confidence)
        if enabled else 0.0
    )

    vector = torch.from_numpy(
        data.spatial_vector.astype(np.float32)
    ).to(batch["input_ids"].device)

    state: Dict[str, Any] = {
        "patched": False,
        "carrier_norm_before": None,
        "carrier_norm_after": None,
        "delta_norm": None,
    }

    def hook(
        _module: Any,
        _inputs: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        if not enabled or state["patched"]:
            return output

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output
        if int(hidden.shape[1]) <= int(carrier_index):
            # Decode steps usually have sequence length 1; patch prefill only.
            return output

        hidden_new = hidden.clone()
        carrier = hidden_new[:, carrier_index, :]
        carrier_float = carrier.float()
        carrier_norm = carrier_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        direction = vector.to(
            device=hidden.device,
            dtype=torch.float32,
        )
        direction = direction / direction.norm().clamp_min(1e-8)
        delta = (
            effective_strength
            * carrier_norm
            * direction[None, :]
        )
        updated = carrier_float + delta

        if renorm_carrier:
            updated = (
                updated
                / updated.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                * carrier_norm
            )

        hidden_new[:, carrier_index, :] = updated.to(hidden.dtype)

        state["patched"] = True
        state["carrier_norm_before"] = float(
            carrier_norm.mean().item()
        )
        state["carrier_norm_after"] = float(
            updated.norm(dim=-1).mean().item()
        )
        state["delta_norm"] = float(
            delta.norm(dim=-1).mean().item()
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
            "Carrier hook was registered but did not patch the prefill state"
        )

    metadata = {
        "carrier_enabled": enabled,
        "carrier_patched": bool(state["patched"]),
        "carrier_confidence": float(confidence),
        "carrier_confidence_mode": confidence_mode,
        "carrier_strength": float(carrier_strength),
        "effective_carrier_strength": float(effective_strength),
        "renorm_carrier": bool(renorm_carrier),
        "carrier_norm_before": state["carrier_norm_before"],
        "carrier_norm_after": state["carrier_norm_after"],
        "delta_norm": state["delta_norm"],
    }
    return text, metadata


def accuracy(
    rows: Iterable[Dict[str, Any]],
    key: str = "correct",
) -> Optional[float]:
    rows = list(rows)
    if not rows:
        return None
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def paired_comparison(
    reference_by_sid: Dict[int, Dict[str, Any]],
    candidate_by_sid: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    paired = [
        sid
        for sid in sorted(set(reference_by_sid) & set(candidate_by_sid))
        if reference_by_sid[sid].get("prediction")
        and candidate_by_sid[sid].get("prediction")
    ]
    fixed = broken = changed = both_correct = both_wrong = 0
    for sid in paired:
        reference = reference_by_sid[sid]
        candidate = candidate_by_sid[sid]
        changed += int(
            reference.get("prediction") != candidate.get("prediction")
        )
        if (not reference.get("correct")) and candidate.get("correct"):
            fixed += 1
        elif reference.get("correct") and (not candidate.get("correct")):
            broken += 1
        elif reference.get("correct") and candidate.get("correct"):
            both_correct += 1
        else:
            both_wrong += 1

    ref_acc = accuracy(reference_by_sid[sid] for sid in paired)
    cand_acc = accuracy(candidate_by_sid[sid] for sid in paired)
    return {
        "n": len(paired),
        "reference_accuracy": ref_acc,
        "candidate_accuracy": cand_acc,
        "absolute_change": (
            cand_acc - ref_acc
            if ref_acc is not None and cand_acc is not None
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
    control_rows: List[Dict[str, Any]],
    guided_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_by_sid = {
        int(row["sid"]): row
        for row in baseline_rows
        if "sid" in row and "error" not in row
    }
    control_by_sid = {
        int(row["sid"]): row
        for row in control_rows
        if "sid" in row and "error" not in row
    }
    guided_by_sid = {
        int(row["sid"]): row
        for row in guided_rows
        if "sid" in row and "error" not in row
    }

    baseline_valid = [
        row for row in baseline_by_sid.values() if row.get("prediction")
    ]
    grounding_valid = [
        row for row in baseline_by_sid.values()
        if row.get("grounding_prediction")
    ]
    control_valid = [
        row for row in control_by_sid.values() if row.get("prediction")
    ]
    guided_valid = [
        row for row in guided_by_sid.values() if row.get("prediction")
    ]

    grounding_on_baseline_wrong = [
        row for row in grounding_valid
        if row.get("prediction") and not bool(row.get("correct"))
    ]
    grounding_on_baseline_correct = [
        row for row in grounding_valid
        if row.get("prediction") and bool(row.get("correct"))
    ]
    agreement_rows = [
        row for row in grounding_valid if row.get("prediction")
    ]

    oracle = (
        float(np.mean([
            bool(row.get("correct")) or bool(row.get("grounding_correct"))
            for row in grounding_valid
        ]))
        if grounding_valid else None
    )
    agreement = (
        float(np.mean([
            row.get("prediction") == row.get("grounding_prediction")
            for row in agreement_rows
        ]))
        if agreement_rows else None
    )

    baseline_control = paired_comparison(
        baseline_by_sid,
        control_by_sid,
    )
    baseline_guided = paired_comparison(
        baseline_by_sid,
        guided_by_sid,
    )
    control_guided = paired_comparison(
        control_by_sid,
        guided_by_sid,
    )

    common = sorted(
        set(baseline_by_sid)
        & set(control_by_sid)
        & set(guided_by_sid)
    )
    per_relation: Dict[str, Dict[str, Any]] = {}
    relation_groups: Dict[str, List[int]] = defaultdict(list)
    for sid in common:
        if (
            baseline_by_sid[sid].get("prediction")
            and control_by_sid[sid].get("prediction")
            and guided_by_sid[sid].get("prediction")
        ):
            relation_groups[str(baseline_by_sid[sid].get("gt"))].append(sid)

    for relation, sids in sorted(relation_groups.items()):
        ground_rows = [
            baseline_by_sid[sid]
            for sid in sids
            if baseline_by_sid[sid].get("grounding_prediction")
        ]
        per_relation[relation] = {
            "n": len(sids),
            "baseline_accuracy": accuracy(
                baseline_by_sid[sid] for sid in sids
            ),
            "grounding_accuracy": accuracy(
                ground_rows,
                key="grounding_correct",
            ),
            "grounding_n": len(ground_rows),
            "control_accuracy": accuracy(
                control_by_sid[sid] for sid in sids
            ),
            "guided_accuracy": accuracy(
                guided_by_sid[sid] for sid in sids
            ),
        }

    return {
        "n_baseline_valid": len(baseline_valid),
        "n_grounding_valid": len(grounding_valid),
        "n_control_valid": len(control_valid),
        "n_guided_valid": len(guided_valid),
        "baseline_accuracy": accuracy(baseline_valid),
        "grounding_accuracy": accuracy(
            grounding_valid,
            key="grounding_correct",
        ),
        "control_accuracy": accuracy(control_valid),
        "guided_accuracy": accuracy(guided_valid),
        "grounding_accuracy_on_baseline_wrong": accuracy(
            grounding_on_baseline_wrong,
            key="grounding_correct",
        ),
        "n_grounding_on_baseline_wrong": len(
            grounding_on_baseline_wrong
        ),
        "grounding_accuracy_on_baseline_correct": accuracy(
            grounding_on_baseline_correct,
            key="grounding_correct",
        ),
        "n_grounding_on_baseline_correct": len(
            grounding_on_baseline_correct
        ),
        "baseline_or_grounding_oracle_accuracy": oracle,
        "baseline_grounding_agreement": agreement,
        "baseline_vs_control": baseline_control,
        "baseline_vs_guided": baseline_guided,
        "control_vs_guided": control_guided,
        "baseline_parse_failures": (
            len(baseline_by_sid) - len(baseline_valid)
        ),
        "control_parse_failures": len(control_by_sid) - len(control_valid),
        "guided_parse_failures": len(guided_by_sid) - len(guided_valid),
        "mean_carrier_confidence": (
            float(np.mean([
                float(row.get("carrier_confidence", 0.0))
                for row in guided_valid
            ]))
            if guided_valid else None
        ),
        "mean_effective_carrier_strength": (
            float(np.mean([
                float(row.get("effective_carrier_strength", 0.0))
                for row in guided_valid
            ]))
            if guided_valid else None
        ),
        "mean_basis_r2": (
            float(np.mean([
                float(row.get("basis_r2", 0.0))
                for row in baseline_valid
            ]))
            if baseline_valid else None
        ),
        "per_relation": per_relation,
    }


def print_pair(
    title: str,
    stats: Dict[str, Any],
) -> None:
    print(f"\n{title}:")
    print(f"paired valid:             {stats.get('n')}")
    ref = stats.get("reference_accuracy")
    cand = stats.get("candidate_accuracy")
    delta = stats.get("absolute_change")
    print(
        f"reference accuracy:       {ref:.4f}"
        if ref is not None else "reference accuracy:       n/a"
    )
    print(
        f"candidate accuracy:       {cand:.4f}"
        if cand is not None else "candidate accuracy:       n/a"
    )
    print(
        f"absolute change:          {delta:+.4f}"
        if delta is not None else "absolute change:          n/a"
    )
    print(f"fixed:                    {stats.get('fixed')}")
    print(f"broken:                   {stats.get('broken')}")
    print(
        f"net fixed-broken:         "
        f"{stats.get('net_fixed_minus_broken'):+d}"
    )
    print(
        f"prediction changed:       "
        f"{stats.get('changed_predictions')}"
    )


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print("BASELINE VS CENTROID VS LATENT SPATIAL-CARRIER GENERATION")
    print("=" * 96)

    metrics = [
        ("baseline generation", "baseline_accuracy", "n_baseline_valid"),
        ("centroid grounding", "grounding_accuracy", "n_grounding_valid"),
        ("carrier control", "control_accuracy", "n_control_valid"),
        ("carrier guided", "guided_accuracy", "n_guided_valid"),
    ]
    for index, (name, key, n_key) in enumerate(metrics, 1):
        value = summary.get(key)
        if value is None:
            print(f"{index}) {name:22s}: n/a")
        else:
            print(
                f"{index}) {name:22s}: {value:.4f} "
                f"(n={summary.get(n_key)})"
            )

    print("\nDoes centroid grounding add information?")
    gw = summary.get("grounding_accuracy_on_baseline_wrong")
    gc = summary.get("grounding_accuracy_on_baseline_correct")
    oracle = summary.get("baseline_or_grounding_oracle_accuracy")
    agreement = summary.get("baseline_grounding_agreement")
    print(
        f"grounding on baseline-wrong:   {gw:.4f} "
        f"(n={summary.get('n_grounding_on_baseline_wrong')})"
        if gw is not None else
        "grounding on baseline-wrong:   n/a"
    )
    print(
        f"grounding on baseline-correct: {gc:.4f} "
        f"(n={summary.get('n_grounding_on_baseline_correct')})"
        if gc is not None else
        "grounding on baseline-correct: n/a"
    )
    print(
        f"baseline union grounding oracle:{oracle:.4f}"
        if oracle is not None else
        "baseline union grounding oracle:n/a"
    )
    print(
        f"baseline/grounding agreement:  {agreement:.4f}"
        if agreement is not None else
        "baseline/grounding agreement:  n/a"
    )

    print_pair(
        "Baseline -> carrier control (prompt-only effect)",
        summary["baseline_vs_control"],
    )
    print_pair(
        "Baseline -> carrier guided (total effect)",
        summary["baseline_vs_guided"],
    )
    print_pair(
        "Carrier control -> carrier guided (latent-vector effect)",
        summary["control_vs_guided"],
    )

    print(
        f"\nbaseline parse fail:            "
        f"{summary.get('baseline_parse_failures')}"
    )
    print(
        f"carrier-control parse fail:     "
        f"{summary.get('control_parse_failures')}"
    )
    print(
        f"carrier-guided parse fail:      "
        f"{summary.get('guided_parse_failures')}"
    )
    print(
        f"mean carrier confidence:        "
        f"{summary.get('mean_carrier_confidence')}"
    )
    print(
        f"mean effective carrier strength:"
        f"{summary.get('mean_effective_carrier_strength')}"
    )
    print(f"mean latent basis R2:           {summary.get('mean_basis_r2')}")

    print("\nPer relation on common valid samples:")
    for relation, stats in summary.get("per_relation", {}).items():
        grounding = (
            f"{stats['grounding_accuracy']:.4f}"
            if stats.get("grounding_accuracy") is not None
            else "n/a"
        )
        print(
            f"  {relation:6s} n={stats['n']:4d} | "
            f"base={stats['baseline_accuracy']:.4f} | "
            f"ground={grounding} (n={stats['grounding_n']}) | "
            f"control={stats['control_accuracy']:.4f} | "
            f"guided={stats['guided_accuracy']:.4f}"
        )


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.basis_ridge < 0:
        raise ValueError("--basis-ridge must be non-negative")
    if args.carrier_strength < 0:
        raise ValueError("--carrier-strength must be non-negative")
    if not (0.0 <= args.min_carrier_confidence <= 1.0):
        raise ValueError("--min-carrier-confidence must be in [0, 1]")
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
    missing_prompt_ids = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing_prompt_ids:
        raise RuntimeError(
            f"Standard prompt file {prompt_path} is missing "
            f"{len(missing_prompt_ids)} IDs; first={missing_prompt_ids[:10]}"
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
                f"No auto layer for {args.model!r}; pass --layer explicitly"
            )
        layer = AUTO_LAYERS[args.model]
    else:
        layer = int(args.layer)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "carrier_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.jsonl"
    control_path = output_dir / "carrier_control.jsonl"
    guided_path = output_dir / "carrier_guided.jsonl"
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
        "basis_ridge": args.basis_ridge,
        "carrier_marker": args.carrier_marker,
        "carrier_strength": args.carrier_strength,
        "carrier_confidence_mode": args.carrier_confidence_mode,
        "min_carrier_confidence": args.min_carrier_confidence,
        "renorm_carrier": args.renorm_carrier,
        "max_new_tokens": args.max_new_tokens,
        "n_records": len(records),
        "audit": audit,
        "uses_gt_for_intervention": False,
        "updates_model_weights": False,
        "edits_output_logits": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no "
            f"{spec.model_class}"
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
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
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
            f"Requested layer={layer}, decoder has {len(decoder_layers)} layers"
        )

    print(f"Standard prompt file: {prompt_path} (rows={len(prompt_rows)})")
    print(
        f"Resolved decoder layers: {decoder_path}, n={len(decoder_layers)}, "
        f"carrier layer={layer}"
    )
    print(f"Carrier marker: {args.carrier_marker!r}")
    print(
        f"Carrier injection: strength={args.carrier_strength}, "
        f"confidence={args.carrier_confidence_mode}, "
        f"threshold={args.min_carrier_confidence}, "
        f"renorm={args.renorm_carrier}"
    )

    baseline_existing = [
        row for row in read_jsonl(baseline_path)
        if "sid" in row and "error" not in row
    ]
    control_existing = [
        row for row in read_jsonl(control_path)
        if "sid" in row and "error" not in row
    ]
    guided_existing = [
        row for row in read_jsonl(guided_path)
        if "sid" in row and "error" not in row
    ]

    baseline_done = {int(row["sid"]) for row in baseline_existing}
    control_done = {int(row["sid"]) for row in control_existing}
    guided_done = {int(row["sid"]) for row in guided_existing}

    baseline_seen = len(baseline_existing)
    baseline_correct_count = sum(
        bool(row.get("correct")) for row in baseline_existing
    )
    control_seen = len(control_existing)
    control_correct_count = sum(
        bool(row.get("correct")) for row in control_existing
    )
    guided_seen = len(guided_existing)
    guided_correct_count = sum(
        bool(row.get("correct")) for row in guided_existing
    )

    started = time.time()

    try:
        if args.only in ("both", "baseline"):
            print(
                "\nPASS 1/2: baseline generation + centroid + latent basis capture"
            )
            for record in tqdm(
                records,
                desc=f"baseline:{args.dataset}:{args.model}",
            ):
                sid = int(record.sid)
                data_path = data_dir / f"{sid}.npz"
                if sid in baseline_done and data_path.exists():
                    continue

                batch = None
                image = None
                try:
                    prompt_row = prompt_rows[sid]
                    (
                        batch,
                        subject,
                        reference,
                        standard_question,
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
                        batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                    baseline_prediction = normalize_relation(
                        baseline_text
                    )
                    gt = normalize_relation(answer_raw)

                    data = build_carrier_data(
                        model=model,
                        processor=processor,
                        batch=batch,
                        sid=sid,
                        subject=subject,
                        reference=reference,
                        layer=layer,
                        temperature=args.temperature,
                        basis_ridge=args.basis_ridge,
                    )
                    save_carrier_data(data_path, data)

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
                        "question": standard_question,
                        "raw_question": raw_question,
                        "standard_answer_raw": answer_raw,
                        "layer": layer,
                        "grounding_prediction": (
                            data.grounding_prediction or None
                        ),
                        "grounding_correct": bool(
                            gt is not None
                            and data.grounding_prediction
                            and gt == data.grounding_prediction
                        ),
                        "subject_centroid": [
                            data.subject_x,
                            data.subject_y,
                        ],
                        "reference_centroid": [
                            data.reference_x,
                            data.reference_y,
                        ],
                        "delta_xy": [
                            data.delta_x,
                            data.delta_y,
                        ],
                        "map_separation": data.map_separation,
                        "axis_confidence": data.axis_confidence,
                        "basis_r2": data.basis_r2,
                        "basis_axis_cosine": data.basis_axis_cosine,
                        "basis_x_norm": data.basis_x_norm,
                        "basis_y_norm": data.basis_y_norm,
                        "spatial_vector_norm": data.spatial_vector_norm,
                    }
                    append_jsonl(baseline_path, row)
                    baseline_done.add(sid)
                    baseline_seen += 1
                    baseline_correct_count += int(row["correct"])

                    if should_print_sample(
                        baseline_seen,
                        args.print_every,
                    ):
                        tqdm.write(
                            f"\n[BASE {baseline_seen}/{len(records)}] "
                            f"sid={sid} | {subject} -> {reference}\n"
                            f"  question   : {one_line(standard_question)}\n"
                            f"  gt         : {gt}\n"
                            f"  generation : {baseline_text!r}\n"
                            f"  pred       : {baseline_prediction}\n"
                            f"  correct    : {int(row['correct'])}\n"
                            f"  acc        : "
                            f"{baseline_correct_count}/{baseline_seen}="
                            f"{baseline_correct_count / baseline_seen:.4f}\n"
                            f"  grounding={data.grounding_prediction} | "
                            f"dx={data.delta_x:+.4f} dy={data.delta_y:+.4f} | "
                            f"sep={data.map_separation:.3f} | "
                            f"axis={data.axis_confidence:.3f} | "
                            f"basis_R2={data.basis_r2:.4f}"
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
                                traceback.format_exc().splitlines()[-12:]
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
                    if batch is not None:
                        del batch
                    if image is not None:
                        del image

        if args.only in ("both", "carrier"):
            print(
                "\nPASS 2/2: carrier-prompt control + latent spatial-carrier generation"
            )
            baseline_by_sid = {
                int(row["sid"]): row
                for row in read_jsonl(baseline_path)
                if "sid" in row and "error" not in row
            }

            for record in tqdm(
                records,
                desc=f"carrier:{args.dataset}:{args.model}",
            ):
                sid = int(record.sid)
                need_control = sid not in control_done
                need_guided = sid not in guided_done
                if not need_control and not need_guided:
                    continue

                data_path = data_dir / f"{sid}.npz"
                if not data_path.exists():
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "carrier",
                            "sid": sid,
                            "error_type": "FileNotFoundError",
                            "error": f"Missing carrier data: {data_path}",
                        },
                    )
                    continue

                batch = None
                image = None
                try:
                    data = load_carrier_data(data_path)
                    prompt_row = prompt_rows[sid]
                    subject = str(prompt_row["subject"])
                    reference = str(prompt_row["reference"])
                    standard_question = str(prompt_row["question_text"])
                    raw_question = str(prompt_row["raw_question"])
                    answer_raw = prompt_row["answer_raw"]
                    gt = normalize_relation(answer_raw)

                    question_with_carrier = carrier_question(
                        standard_question,
                        args.carrier_marker,
                    )
                    image = record_image(record)
                    batch = make_question_batch(
                        processor=processor,
                        image=image,
                        question_text=question_with_carrier,
                        device=device,
                    )
                    token_ids = batch["input_ids"][0].detach().cpu().tolist()
                    carrier_index, carrier_span = locate_carrier_index(
                        processor.tokenizer,
                        token_ids,
                        args.carrier_marker,
                    )

                    if need_control:
                        control_text = generate_text(
                            model,
                            processor,
                            batch,
                            max_new_tokens=args.max_new_tokens,
                        )
                        control_prediction = normalize_relation(control_text)
                        control_row = {
                            "sid": sid,
                            "subject": subject,
                            "reference": reference,
                            "gt": gt,
                            "prediction": control_prediction,
                            "correct": bool(
                                gt is not None
                                and control_prediction is not None
                                and gt == control_prediction
                            ),
                            "generated_text": control_text,
                            "standard_question": standard_question,
                            "carrier_question": question_with_carrier,
                            "raw_question": raw_question,
                            "carrier_marker": args.carrier_marker,
                            "carrier_index": carrier_index,
                            "carrier_span": list(carrier_span),
                        }
                        append_jsonl(control_path, control_row)
                        control_done.add(sid)
                        control_seen += 1
                        control_correct_count += int(control_row["correct"])

                        if should_print_sample(
                            control_seen,
                            args.print_every,
                        ):
                            tqdm.write(
                                f"\n[CONTROL {control_seen}/{len(records)}] "
                                f"sid={sid} | {subject} -> {reference}\n"
                                f"  question   : {one_line(question_with_carrier)}\n"
                                f"  gt         : {gt}\n"
                                f"  generation : {control_text!r}\n"
                                f"  pred       : {control_prediction}\n"
                                f"  correct    : {int(control_row['correct'])}\n"
                                f"  acc        : "
                                f"{control_correct_count}/{control_seen}="
                                f"{control_correct_count / control_seen:.4f}"
                            )

                    if need_guided:
                        guided_text, metadata = carrier_guided_generate(
                            model=model,
                            processor=processor,
                            batch=batch,
                            decoder_layer=decoder_layers[layer],
                            data=data,
                            carrier_index=carrier_index,
                            carrier_strength=args.carrier_strength,
                            confidence_mode=args.carrier_confidence_mode,
                            min_confidence=args.min_carrier_confidence,
                            renorm_carrier=args.renorm_carrier,
                            max_new_tokens=args.max_new_tokens,
                        )
                        guided_prediction = normalize_relation(guided_text)
                        guided_row = {
                            "sid": sid,
                            "subject": subject,
                            "reference": reference,
                            "gt": gt,
                            "prediction": guided_prediction,
                            "correct": bool(
                                gt is not None
                                and guided_prediction is not None
                                and gt == guided_prediction
                            ),
                            "generated_text": guided_text,
                            "standard_question": standard_question,
                            "carrier_question": question_with_carrier,
                            "raw_question": raw_question,
                            "carrier_marker": args.carrier_marker,
                            "carrier_index": carrier_index,
                            "carrier_span": list(carrier_span),
                            "grounding_prediction": (
                                data.grounding_prediction or None
                            ),
                            "delta_xy": [
                                data.delta_x,
                                data.delta_y,
                            ],
                            "map_separation": data.map_separation,
                            "axis_confidence": data.axis_confidence,
                            "basis_r2": data.basis_r2,
                            "basis_axis_cosine": data.basis_axis_cosine,
                            "spatial_vector_norm": data.spatial_vector_norm,
                            **metadata,
                        }
                        append_jsonl(guided_path, guided_row)
                        guided_done.add(sid)
                        guided_seen += 1
                        guided_correct_count += int(guided_row["correct"])

                        baseline = baseline_by_sid.get(sid)
                        base_prediction = (
                            baseline.get("prediction")
                            if baseline is not None else None
                        )
                        status = "NO_BASELINE"
                        if baseline is not None:
                            if (
                                not baseline.get("correct")
                                and guided_row.get("correct")
                            ):
                                status = "FIXED"
                            elif (
                                baseline.get("correct")
                                and not guided_row.get("correct")
                            ):
                                status = "BROKEN"
                            elif (
                                baseline.get("correct")
                                and guided_row.get("correct")
                            ):
                                status = "UNCHANGED_CORRECT"
                            else:
                                status = "UNCHANGED_WRONG"

                        if should_print_sample(
                            guided_seen,
                            args.print_every,
                        ):
                            tqdm.write(
                                f"\n[GUIDED {guided_seen}/{len(records)}] "
                                f"sid={sid} | {subject} -> {reference}\n"
                                f"  gt         : {gt}\n"
                                f"  generation : {guided_text!r}\n"
                                f"  pred       : {guided_prediction}\n"
                                f"  correct    : {int(guided_row['correct'])}\n"
                                f"  acc        : "
                                f"{guided_correct_count}/{guided_seen}="
                                f"{guided_correct_count / guided_seen:.4f}\n"
                                f"  base={base_prediction} -> "
                                f"guided={guided_prediction} | {status}\n"
                                f"  grounding={data.grounding_prediction} | "
                                f"carrier_conf={metadata['carrier_confidence']:.3f} | "
                                f"effective_strength="
                                f"{metadata['effective_carrier_strength']:.3f} | "
                                f"delta_norm={metadata['delta_norm']}"
                            )

                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "carrier",
                            "sid": sid,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback_tail": (
                                traceback.format_exc().splitlines()[-12:]
                            ),
                        },
                    )
                    tqdm.write(
                        f"\n[CARRIER ERROR] sid={sid} | "
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
        control_rows = read_jsonl(control_path)
        guided_rows = read_jsonl(guided_path)
        if baseline_rows and control_rows and guided_rows:
            summary = summarize(
                baseline_rows,
                control_rows,
                guided_rows,
            )
            summary["config"] = run_config
            summary["elapsed_minutes"] = (
                time.time() - started
            ) / 60.0
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print_summary(summary)
            print(f"\nSaved summary: {summary_path}")
        else:
            print(
                "Completed requested stage. "
                f"baseline_rows={len(baseline_rows)}, "
                f"control_rows={len(control_rows)}, "
                f"guided_rows={len(guided_rows)}"
            )

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
