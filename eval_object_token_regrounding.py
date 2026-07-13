#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline vs. training-free object-token visual re-grounding.

Purpose
-------
Evaluate whether visual information already bound to the two object-word tokens
can improve spatial-relation generation without fine-tuning and without using
per-sample GT labels for intervention.

The script runs two passes over the same records:

1) Baseline pass
   - Greedy generation with the frozen VLM.
   - A separate frozen forward captures a chosen decoder layer.
   - Each object token is matched to image tokens by cosine similarity.
   - The two grounded visual summaries are stored for the modified pass.

2) Modified pass
   - At the same decoder layer, a forward hook changes only the final sub-token
     of the subject and reference object spans.
   - The shared center of the two object states is preserved.
   - Their difference is moved toward the difference between their grounded
     visual summaries.
   - The remaining layers and LM head generate normally.

No trainable parameters are introduced. Ground-truth relations are read only
when computing evaluation metrics after generation; they are never used to
construct the patch.

This first implementation deliberately uses object-token/image-token cosine
similarity rather than output_attentions=True. Returning all decoder attention
matrices can be prohibitively expensive for Qwen2.5-VL image sequences. The
similarity grounding is model-agnostic and directly tests the core hypothesis.
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


PROMPT_TEMPLATE = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with one spatial relation."
)

RELATIONS = ("left", "right", "on", "under")

AUTO_LAYERS = {
    "llava-7b": 12,
    "llava-13b": 16,
    "qwen-3b": 24,
    "qwen-7b": 19,
}


@dataclass
class PatchData:
    sid: int
    subject_index: int
    reference_index: int
    input_length: int
    layer: int
    z_text: np.ndarray
    z_visual: np.ndarray
    confidence: float
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
        help="No decoder attention matrices are requested; sdpa is supported.",
    )
    p.add_argument(
        "--layer",
        default="auto",
        help="Zero-based decoder block index, or 'auto'.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Maximum interpolation strength toward grounded visual difference.",
    )
    p.add_argument(
        "--confidence-mode",
        default="separation",
        choices=["none", "separation", "entropy", "combined"],
        help=(
            "How to scale gamma without labels. 'separation' uses total variation "
            "between the two object-to-image similarity maps."
        ),
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Do not patch samples below this label-free confidence.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--only",
        choices=["both", "baseline", "modified"],
        default="both",
        help="Run both passes, only baseline/capture, or only modified from saved patches.",
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


def build_prompt(processor: Any, subject: str, reference: str) -> Tuple[str, str]:
    prompt_text = PROMPT_TEMPLATE.format(subject=subject, reference=reference)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt_text
    return rendered, prompt_text


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


def find_phrase_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    phrase: str,
    *,
    include_article_variants: bool = False,
) -> List[Tuple[int, int]]:
    variants = [phrase, " " + phrase]
    if include_article_variants:
        variants.extend(["the " + phrase, " the " + phrase])

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
    return ("on" if delta_y < 0 else "under"), float(axis_conf)


def normalize_relation(value: Any) -> Optional[str]:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    exact = {
        "left": "left",
        "left of": "left",
        "to the left": "left",
        "right": "right",
        "right of": "right",
        "to the right": "right",
        "on": "on",
        "above": "on",
        "over": "on",
        "on top of": "on",
        "top": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
    }
    if text in exact:
        return exact[text]

    # Prefer explicit spatial phrases. Word boundaries avoid matching "on" inside words.
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(under|below|beneath|bottom)\b", "under"),
        (r"\b(above|over|on top|top)\b", "on"),
        (r"\bon\b", "on"),
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


def build_patch_data(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    sid: int,
    subject: str,
    reference: str,
    layer: int,
    temperature: float,
    confidence_mode: str,
) -> PatchData:
    if temperature <= 0:
        raise ValueError("--temperature must be positive")

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
        raise ValueError(f"Requested layer={layer}, but model has {n_blocks} decoder blocks")

    hidden = states[layer + 1][0].float()
    if int(hidden.shape[0]) != len(input_ids):
        raise RuntimeError(
            f"Token/hidden mismatch: input={len(input_ids)}, hidden={int(hidden.shape[0])}"
        )

    hs = hidden[subject_index]
    hr = hidden[reference_index]
    visual = hidden[torch.as_tensor(visual_indices, device=hidden.device, dtype=torch.long)]

    # Object-token -> image-token grounding by normalized similarity.
    visual_norm = F.normalize(visual, dim=-1)
    hs_norm = F.normalize(hs, dim=-1)
    hr_norm = F.normalize(hr, dim=-1)
    logits_s = torch.matmul(visual_norm, hs_norm) / float(temperature)
    logits_r = torch.matmul(visual_norm, hr_norm) / float(temperature)
    weights_s = torch.softmax(logits_s, dim=0)
    weights_r = torch.softmax(logits_r, dim=0)

    gs = torch.sum(weights_s[:, None] * visual, dim=0)
    gr = torch.sum(weights_r[:, None] * visual, dim=0)

    z_text = hs - hr
    z_visual = gs - gr

    # Match only the magnitude. The modified pass decides interpolation strength.
    z_text_norm = z_text.norm().clamp_min(1e-8)
    z_visual_norm = z_visual.norm().clamp_min(1e-8)
    z_visual_matched = z_visual / z_visual_norm * z_text_norm

    # Total variation distance between the two grounding maps, in [0, 1].
    separation = 0.5 * torch.sum(torch.abs(weights_s - weights_r))
    ent_s = entropy_confidence(weights_s)
    ent_r = entropy_confidence(weights_r)
    conf = confidence_value(
        confidence_mode,
        float(separation.item()),
        float(ent_s.item()),
        float(ent_r.item()),
    )

    coords = visual_coordinates(model, batch, len(visual_indices), hidden.device)
    if coords is not None and int(coords.shape[0]) == len(visual_indices):
        center_s = torch.sum(weights_s[:, None] * coords, dim=0)
        center_r = torch.sum(weights_r[:, None] * coords, dim=0)
        subject_x, subject_y = [float(x) for x in center_s.tolist()]
        reference_x, reference_y = [float(x) for x in center_r.tolist()]
        delta_x = subject_x - reference_x
        delta_y = subject_y - reference_y
        grounding_prediction, axis_confidence = relation_from_centroids(delta_x, delta_y)
    else:
        subject_x = subject_y = reference_x = reference_y = float("nan")
        delta_x = delta_y = float("nan")
        axis_confidence = float("nan")
        grounding_prediction = ""

    patch = PatchData(
        sid=sid,
        subject_index=subject_index,
        reference_index=reference_index,
        input_length=len(input_ids),
        layer=layer,
        z_text=z_text.detach().cpu().numpy().astype(np.float16),
        z_visual=z_visual_matched.detach().cpu().numpy().astype(np.float16),
        confidence=conf,
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
    )

    del outputs, states, hidden, visual, weights_s, weights_r, gs, gr
    return patch


def save_patch(path: Path, patch: PatchData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            sid=np.asarray(patch.sid, dtype=np.int64),
            subject_index=np.asarray(patch.subject_index, dtype=np.int32),
            reference_index=np.asarray(patch.reference_index, dtype=np.int32),
            input_length=np.asarray(patch.input_length, dtype=np.int32),
            layer=np.asarray(patch.layer, dtype=np.int32),
            z_text=patch.z_text,
            z_visual=patch.z_visual,
            confidence=np.asarray(patch.confidence, dtype=np.float32),
            map_separation=np.asarray(patch.map_separation, dtype=np.float32),
            subject_entropy_confidence=np.asarray(
                patch.subject_entropy_confidence, dtype=np.float32
            ),
            reference_entropy_confidence=np.asarray(
                patch.reference_entropy_confidence, dtype=np.float32
            ),
            subject_peak_weight=np.asarray(patch.subject_peak_weight, dtype=np.float32),
            reference_peak_weight=np.asarray(patch.reference_peak_weight, dtype=np.float32),
            n_visual_tokens=np.asarray(patch.n_visual_tokens, dtype=np.int32),
            subject_x=np.asarray(patch.subject_x, dtype=np.float32),
            subject_y=np.asarray(patch.subject_y, dtype=np.float32),
            reference_x=np.asarray(patch.reference_x, dtype=np.float32),
            reference_y=np.asarray(patch.reference_y, dtype=np.float32),
            delta_x=np.asarray(patch.delta_x, dtype=np.float32),
            delta_y=np.asarray(patch.delta_y, dtype=np.float32),
            axis_confidence=np.asarray(patch.axis_confidence, dtype=np.float32),
            grounding_prediction=np.asarray(patch.grounding_prediction, dtype="<U8"),
        )
    os.replace(tmp, path)


def load_patch(path: Path) -> PatchData:
    with np.load(path, allow_pickle=False) as z:
        return PatchData(
            sid=int(z["sid"].item()),
            subject_index=int(z["subject_index"].item()),
            reference_index=int(z["reference_index"].item()),
            input_length=int(z["input_length"].item()),
            layer=int(z["layer"].item()),
            z_text=z["z_text"],
            z_visual=z["z_visual"],
            confidence=float(z["confidence"].item()),
            map_separation=float(z["map_separation"].item()),
            subject_entropy_confidence=float(z["subject_entropy_confidence"].item()),
            reference_entropy_confidence=float(z["reference_entropy_confidence"].item()),
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
    device: torch.device,
) -> Tuple[Dict[str, Any], str, str, str, Image.Image]:
    subject = str(record.subject)
    reference = str(record.reference)
    rendered, prompt_text = build_prompt(processor, subject, reference)
    image = record_image(record)
    batch = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
    )
    batch = move_batch(batch, device)
    return batch, subject, reference, prompt_text, image


def modified_generate(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    decoder_layer: Any,
    patch: PatchData,
    gamma: float,
    min_confidence: float,
    max_new_tokens: int,
) -> Tuple[str, float, bool]:
    if int(batch["input_ids"].shape[1]) != patch.input_length:
        raise RuntimeError(
            f"Input length changed: saved={patch.input_length}, "
            f"current={int(batch['input_ids'].shape[1])}"
        )

    gamma_effective = float(gamma) * float(patch.confidence)
    enabled = patch.confidence >= min_confidence and abs(gamma_effective) > 0.0

    z_text = torch.from_numpy(patch.z_text.astype(np.float32)).to(
        device=batch["input_ids"].device
    )
    z_visual = torch.from_numpy(patch.z_visual.astype(np.float32)).to(
        device=batch["input_ids"].device
    )
    delta_float = gamma_effective * (z_visual - z_text)

    state = {"patched": False}

    def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not enabled or state["patched"]:
            return output

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output
        if hidden.shape[1] <= max(patch.subject_index, patch.reference_index):
            # Autoregressive decode step after prefill generally has sequence length 1.
            return output

        hidden_new = hidden.clone()
        delta = delta_float.to(device=hidden.device, dtype=hidden.dtype)
        hidden_new[:, patch.subject_index, :] += 0.5 * delta
        hidden_new[:, patch.reference_index, :] -= 0.5 * delta
        state["patched"] = True

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
        raise RuntimeError("Patch hook was registered but never modified the prefill hidden states")
    return text, gamma_effective, bool(state["patched"])


def summarize(
    baseline_rows: List[Dict[str, Any]],
    modified_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base_by_sid = {int(row["sid"]): row for row in baseline_rows if "error" not in row}
    mod_by_sid = {int(row["sid"]): row for row in modified_rows if "error" not in row}
    common = sorted(set(base_by_sid) & set(mod_by_sid))

    baseline_valid = [base_by_sid[s] for s in common if base_by_sid[s].get("prediction")]
    modified_valid = [mod_by_sid[s] for s in common if mod_by_sid[s].get("prediction")]
    paired_valid = [
        s
        for s in common
        if base_by_sid[s].get("prediction") and mod_by_sid[s].get("prediction")
    ]

    def accuracy(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
        rows = list(rows)
        if not rows:
            return None
        return float(np.mean([bool(row.get("correct")) for row in rows]))

    fixed = 0
    broken = 0
    changed = 0
    both_correct = 0
    both_wrong = 0
    per_relation: Dict[str, Dict[str, Any]] = {}

    rel_groups: Dict[str, List[int]] = defaultdict(list)
    for sid in paired_valid:
        b = base_by_sid[sid]
        m = mod_by_sid[sid]
        if b["prediction"] != m["prediction"]:
            changed += 1
        if (not b["correct"]) and m["correct"]:
            fixed += 1
        elif b["correct"] and (not m["correct"]):
            broken += 1
        elif b["correct"] and m["correct"]:
            both_correct += 1
        else:
            both_wrong += 1
        rel_groups[str(b["gt"])].append(sid)

    for rel, sids in sorted(rel_groups.items()):
        per_relation[rel] = {
            "n": len(sids),
            "baseline_accuracy": accuracy(base_by_sid[s] for s in sids),
            "modified_accuracy": accuracy(mod_by_sid[s] for s in sids),
        }

    grounding_rows = [
        base_by_sid[s]
        for s in paired_valid
        if base_by_sid[s].get("grounding_prediction")
    ]
    grounding_accuracy = accuracy(
        {"correct": row.get("grounding_correct", False)} for row in grounding_rows
    )

    confidence_values = [
        float(mod_by_sid[s].get("confidence", 0.0)) for s in paired_valid
    ]
    gamma_values = [
        float(mod_by_sid[s].get("gamma_effective", 0.0)) for s in paired_valid
    ]

    return {
        "n_baseline_rows": len(baseline_rows),
        "n_modified_rows": len(modified_rows),
        "n_common": len(common),
        "n_paired_valid": len(paired_valid),
        "baseline_parse_failures": len(common) - len(baseline_valid),
        "modified_parse_failures": len(common) - len(modified_valid),
        "baseline_accuracy": accuracy(base_by_sid[s] for s in paired_valid),
        "modified_accuracy": accuracy(mod_by_sid[s] for s in paired_valid),
        "grounding_accuracy": grounding_accuracy,
        "n_grounding_valid": len(grounding_rows),
        "fixed": fixed,
        "broken": broken,
        "net_fixed_minus_broken": fixed - broken,
        "changed_predictions": changed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mean_confidence": float(np.mean(confidence_values)) if confidence_values else None,
        "mean_gamma_effective": float(np.mean(gamma_values)) if gamma_values else None,
        "per_relation": per_relation,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("BASELINE VS OBJECT-TOKEN RE-GROUNDING")
    print("=" * 80)
    print(f"paired valid:       {summary.get('n_paired_valid')}")
    bacc = summary.get("baseline_accuracy")
    macc = summary.get("modified_accuracy")
    print(f"baseline accuracy:  {bacc:.4f}" if bacc is not None else "baseline accuracy:  n/a")
    print(f"modified accuracy:  {macc:.4f}" if macc is not None else "modified accuracy:  n/a")
    if bacc is not None and macc is not None:
        print(f"absolute change:    {macc - bacc:+.4f}")
    gacc = summary.get("grounding_accuracy")
    if gacc is not None:
        print(f"centroid grounding: {gacc:.4f} (n={summary.get('n_grounding_valid')})")
    print(f"fixed:              {summary.get('fixed')}")
    print(f"broken:             {summary.get('broken')}")
    print(f"net fixed-broken:   {summary.get('net_fixed_minus_broken'):+d}")
    print(f"prediction changed: {summary.get('changed_predictions')}")
    print(f"baseline parse fail:{summary.get('baseline_parse_failures')}")
    print(f"modified parse fail:{summary.get('modified_parse_failures')}")
    print(f"mean confidence:    {summary.get('mean_confidence')}")
    print(f"mean effective gamma:{summary.get('mean_gamma_effective')}")
    print("\nPer relation:")
    for rel, stats in summary.get("per_relation", {}).items():
        print(
            f"  {rel:6s} n={stats['n']:4d} | "
            f"base={stats['baseline_accuracy']:.4f} | "
            f"modified={stats['modified_accuracy']:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (0.0 <= args.gamma <= 2.0):
        raise ValueError("Use --gamma in [0, 2] for the initial experiment")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("--min-confidence must be in [0, 1]")

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
    if args.model not in module.SPECS:
        raise ValueError(
            f"Model {args.model!r} not found in extract_two_object_relation_states.SPECS"
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
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.jsonl"
    modified_path = output_dir / "modified.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    run_config = {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "model": args.model,
        "repo_id": spec.repo_id,
        "transformers_version": transformers.__version__,
        "prompt_template": PROMPT_TEMPLATE,
        "layer": layer,
        "temperature": args.temperature,
        "gamma": args.gamma,
        "confidence_mode": args.confidence_mode,
        "min_confidence": args.min_confidence,
        "max_new_tokens": args.max_new_tokens,
        "n_records": len(records),
        "audit": audit,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

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
    print(
        f"Resolved decoder layers: {decoder_path}, n={len(decoder_layers)}, "
        f"target layer={layer}"
    )
    if not (0 <= layer < len(decoder_layers)):
        raise ValueError(
            f"Requested layer={layer}, decoder has {len(decoder_layers)} layers"
        )

    baseline_done = {
        int(row["sid"])
        for row in read_jsonl(baseline_path)
        if "sid" in row and "error" not in row
    }
    modified_done = {
        int(row["sid"])
        for row in read_jsonl(modified_path)
        if "sid" in row and "error" not in row
    }

    started = time.time()

    try:
        if args.only in ("both", "baseline"):
            print("\nPASS 1/2: baseline generation + label-free patch capture")
            for record in tqdm(records, desc=f"baseline:{args.dataset}:{args.model}"):
                sid = int(record.sid)
                patch_path = patch_dir / f"{sid}.npz"
                if sid in baseline_done and patch_path.exists():
                    continue

                batch = None
                image = None
                try:
                    batch, subject, reference, prompt_text, image = make_batch(
                        processor,
                        record,
                        device,
                    )
                    baseline_text = generate_text(
                        model,
                        processor,
                        batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                    baseline_prediction = normalize_relation(baseline_text)
                    gt = normalize_relation(record.relation)

                    patch = build_patch_data(
                        model=model,
                        processor=processor,
                        batch=batch,
                        sid=sid,
                        subject=subject,
                        reference=reference,
                        layer=layer,
                        temperature=args.temperature,
                        confidence_mode=args.confidence_mode,
                    )
                    save_patch(patch_path, patch)

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
                        "prompt": prompt_text,
                        "layer": layer,
                        "confidence": patch.confidence,
                        "map_separation": patch.map_separation,
                        "subject_entropy_confidence": patch.subject_entropy_confidence,
                        "reference_entropy_confidence": patch.reference_entropy_confidence,
                        "subject_peak_weight": patch.subject_peak_weight,
                        "reference_peak_weight": patch.reference_peak_weight,
                        "n_visual_tokens": patch.n_visual_tokens,
                        "grounding_prediction": patch.grounding_prediction or None,
                        "grounding_correct": bool(
                            gt is not None
                            and patch.grounding_prediction
                            and gt == patch.grounding_prediction
                        ),
                        "subject_centroid": [patch.subject_x, patch.subject_y],
                        "reference_centroid": [patch.reference_x, patch.reference_y],
                        "delta_xy": [patch.delta_x, patch.delta_y],
                        "axis_confidence": patch.axis_confidence,
                    }
                    append_jsonl(baseline_path, row)
                    baseline_done.add(sid)

                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "baseline",
                            "sid": sid,
                            "subject": str(getattr(record, "subject", "")),
                            "reference": str(getattr(record, "reference", "")),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback_tail": traceback.format_exc().splitlines()[-12:],
                        },
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    if batch is not None:
                        del batch
                    if image is not None:
                        del image

        if args.only in ("both", "modified"):
            print("\nPASS 2/2: modified generation with object-token re-grounding")
            for record in tqdm(records, desc=f"modified:{args.dataset}:{args.model}"):
                sid = int(record.sid)
                if sid in modified_done:
                    continue
                patch_path = patch_dir / f"{sid}.npz"
                if not patch_path.exists():
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "modified",
                            "sid": sid,
                            "error_type": "FileNotFoundError",
                            "error": f"Missing patch file: {patch_path}",
                        },
                    )
                    continue

                batch = None
                image = None
                try:
                    patch = load_patch(patch_path)
                    batch, subject, reference, prompt_text, image = make_batch(
                        processor,
                        record,
                        device,
                    )
                    modified_text, gamma_effective, actually_patched = modified_generate(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layer=decoder_layers[layer],
                        patch=patch,
                        gamma=args.gamma,
                        min_confidence=args.min_confidence,
                        max_new_tokens=args.max_new_tokens,
                    )
                    modified_prediction = normalize_relation(modified_text)
                    gt = normalize_relation(record.relation)
                    row = {
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "gt": gt,
                        "prediction": modified_prediction,
                        "correct": bool(
                            gt is not None
                            and modified_prediction is not None
                            and gt == modified_prediction
                        ),
                        "generated_text": modified_text,
                        "prompt": prompt_text,
                        "layer": layer,
                        "gamma": args.gamma,
                        "confidence": patch.confidence,
                        "gamma_effective": gamma_effective,
                        "actually_patched": actually_patched,
                        "map_separation": patch.map_separation,
                        "subject_entropy_confidence": patch.subject_entropy_confidence,
                        "reference_entropy_confidence": patch.reference_entropy_confidence,
                        "n_visual_tokens": patch.n_visual_tokens,
                        "grounding_prediction": patch.grounding_prediction or None,
                        "axis_confidence": patch.axis_confidence,
                    }
                    append_jsonl(modified_path, row)
                    modified_done.add(sid)

                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "pass": "modified",
                            "sid": sid,
                            "subject": str(getattr(record, "subject", "")),
                            "reference": str(getattr(record, "reference", "")),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback_tail": traceback.format_exc().splitlines()[-12:],
                        },
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    if batch is not None:
                        del batch
                    if image is not None:
                        del image

        baseline_rows = read_jsonl(baseline_path)
        modified_rows = read_jsonl(modified_path)
        if baseline_rows and modified_rows:
            summary = summarize(baseline_rows, modified_rows)
            summary["config"] = run_config
            summary["elapsed_minutes"] = (time.time() - started) / 60.0
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print_summary(summary)
            print(f"\nSaved summary: {summary_path}")
        else:
            print(
                f"Completed requested pass. baseline_rows={len(baseline_rows)}, "
                f"modified_rows={len(modified_rows)}"
            )

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
