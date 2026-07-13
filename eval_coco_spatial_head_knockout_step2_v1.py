#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: causal knockout of sparse spatial-attention heads.

This script uses the layer/head rankings produced by Step 1:

    analyze_coco_attention_flow_swap_step1_v1.py

It tests whether the identified spatial heads causally contribute to normal
autoregressive generation.  The original question and image are kept unchanged.

For selected decoder layers/heads and only at the subject/reference object-token
query positions, three intervention modes are supported:

1) visual_value_zero
   Subtract only the selected head's visual-token value contribution:

       sum_{j in visual} A(q,j) V(j) -> 0

   Attention weights are left unchanged.  This asks whether the visual values
   carried through the spatial head matter for generation.

2) weight_zero_renorm
   Set attention weights from the object query to visual keys to zero, renormalize
   the remaining text-key weights, reconstruct the selected head output, and
   continue generation.

3) head_output_zero
   Remove the selected head's entire output at the two object-token positions.
   This is a stronger control and is less specific to the visual pathway.

The intervention is implemented after eager attention has produced its
probabilities but before the attention block output is returned.  The script
reconstructs the selected per-head output using the returned attention weights
and the captured value projection, then applies the original output projection.

The script compares:
- top_unsupervised: heads selected by Step 1's label-free stability score;
- random_matched: random heads in the same layers;
- bottom_matched: lowest-score heads in the same layers;
- top_oracle: heads selected by centroid accuracy, for analysis only.

No model parameters are trained.  Ground truth is used only after generation to
compute accuracy, fixed/broken counts, and per-relation statistics.
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


SCRIPT_VERSION = "spatial-head-knockout-step2-v1"

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






HEAD_GROUPS = (
    "top_unsupervised",
    "random_matched",
    "bottom_matched",
    "top_oracle",
)


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
        default="eager",
        choices=["eager"],
        help="Complete attention probabilities are required.",
    )
    p.add_argument(
        "--step1-dir",
        required=True,
        help=(
            "Output directory from Step 1 containing aggregate_metrics.npz, "
            "for example output/coco_attention_flow_swap_full/qwen-3b."
        ),
    )
    p.add_argument(
        "--knockout-mode",
        default="visual_value_zero",
        choices=[
            "visual_value_zero",
            "weight_zero_renorm",
            "head_output_zero",
        ],
    )
    p.add_argument(
        "--condition-groups",
        default="top_unsupervised,random_matched,bottom_matched,top_oracle",
        help="Comma-separated subset of: " + ",".join(HEAD_GROUPS),
    )
    p.add_argument(
        "--top-k",
        default="1,2,5",
        help="Comma-separated numbers of heads to intervene on.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=17,
        help="Seed for same-layer random control heads.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--print-every", type=int, default=1)
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





def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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


def parse_csv_names(value: str, allowed: Sequence[str]) -> List[str]:
    result: List[str] = []
    allowed_set = set(allowed)
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported value {item!r}; allowed={sorted(allowed_set)}"
            )
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("Resolved empty name list")
    return result


def parse_positive_ints(value: str) -> List[int]:
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise ValueError("All --top-k values must be positive")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError("--top-k resolved to an empty list")
    return sorted(result)


def load_step1_rankings(
    step1_dir: Path,
) -> Dict[str, Any]:
    path = step1_dir / "aggregate_metrics.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Step 1 aggregate metrics: {path}"
        )
    with np.load(path, allow_pickle=False) as z:
        required = [
            "layer_indices",
            "unsupervised_head_score",
            "attention_average_accuracy",
        ]
        missing = [name for name in required if name not in z.files]
        if missing:
            raise RuntimeError(
                f"{path} is missing arrays: {missing}; files={z.files}"
            )
        layers = z["layer_indices"].astype(np.int64)
        unsupervised = z["unsupervised_head_score"].astype(np.float64)
        oracle = z["attention_average_accuracy"].astype(np.float64)

    if unsupervised.ndim != 2 or oracle.shape != unsupervised.shape:
        raise RuntimeError(
            "Step 1 ranking arrays must both have shape [layer, head]"
        )
    if len(layers) != unsupervised.shape[0]:
        raise RuntimeError(
            "layer_indices length does not match ranking arrays"
        )

    return {
        "path": str(path),
        "layers": layers,
        "unsupervised": unsupervised,
        "oracle": oracle,
        "n_heads": int(unsupervised.shape[1]),
    }


def sorted_head_pairs(
    layers: np.ndarray,
    score: np.ndarray,
    descending: bool,
) -> List[Tuple[int, int, float]]:
    rows: List[Tuple[int, int, float]] = []
    for layer_pos, layer in enumerate(layers.tolist()):
        for head in range(score.shape[1]):
            value = float(score[layer_pos, head])
            if not np.isfinite(value):
                continue
            rows.append((int(layer), int(head), value))
    rows.sort(key=lambda row: row[2], reverse=descending)
    return rows


def matched_heads(
    *,
    reference: Sequence[Tuple[int, int, float]],
    layers: np.ndarray,
    score: np.ndarray,
    mode: str,
    seed: int,
) -> List[Tuple[int, int, float]]:
    layer_to_pos = {
        int(layer): pos for pos, layer in enumerate(layers.tolist())
    }
    rng = random.Random(seed)
    used: set[Tuple[int, int]] = set()
    result: List[Tuple[int, int, float]] = []

    for rank, (layer, reference_head, _) in enumerate(reference):
        layer_pos = layer_to_pos[layer]
        candidates = list(range(score.shape[1]))
        candidates = [
            head
            for head in candidates
            if head != reference_head and (layer, head) not in used
        ]
        if not candidates:
            candidates = [
                head
                for head in range(score.shape[1])
                if (layer, head) not in used
            ]
        if not candidates:
            raise RuntimeError(
                f"Could not find a unique matched control in layer {layer}"
            )

        if mode == "random":
            rng.shuffle(candidates)
            chosen = candidates[0]
        elif mode == "bottom":
            chosen = min(
                candidates,
                key=lambda head: float(score[layer_pos, head]),
            )
        else:
            raise ValueError(f"Unsupported matched-head mode: {mode}")

        used.add((layer, chosen))
        result.append(
            (layer, int(chosen), float(score[layer_pos, chosen]))
        )
    return result


def build_conditions(
    ranking: Dict[str, Any],
    groups: Sequence[str],
    top_k_values: Sequence[int],
    seed: int,
) -> Dict[str, List[Tuple[int, int, float]]]:
    max_k = max(top_k_values)
    layers = ranking["layers"]
    unsupervised = ranking["unsupervised"]
    oracle = ranking["oracle"]

    top_unsup_full = sorted_head_pairs(
        layers,
        unsupervised,
        descending=True,
    )[:max_k]
    top_oracle_full = sorted_head_pairs(
        layers,
        oracle,
        descending=True,
    )[:max_k]

    if len(top_unsup_full) < max_k or len(top_oracle_full) < max_k:
        raise RuntimeError(
            f"Not enough ranked heads for max top-k={max_k}"
        )

    random_full = matched_heads(
        reference=top_unsup_full,
        layers=layers,
        score=unsupervised,
        mode="random",
        seed=seed,
    )
    bottom_full = matched_heads(
        reference=top_unsup_full,
        layers=layers,
        score=unsupervised,
        mode="bottom",
        seed=seed,
    )

    source = {
        "top_unsupervised": top_unsup_full,
        "random_matched": random_full,
        "bottom_matched": bottom_full,
        "top_oracle": top_oracle_full,
    }

    conditions: Dict[str, List[Tuple[int, int, float]]] = {}
    for group in groups:
        for k in top_k_values:
            conditions[f"{group}_k{k}"] = list(source[group][:k])
    return conditions


def find_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise AttributeError(
        f"Could not find self-attention module inside {type(layer).__name__}"
    )


def find_attention_weights(output: Any) -> Optional[torch.Tensor]:
    candidates: List[Any]
    if isinstance(output, (tuple, list)):
        candidates = list(output[1:])
    else:
        candidates = []

    for value in candidates:
        if (
            torch.is_tensor(value)
            and value.ndim == 4
            and value.shape[-1] >= value.shape[-2]
        ):
            return value
    return None


def replace_first_output(output: Any, first: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (first,) + output[1:]
    if isinstance(output, list):
        return [first] + list(output[1:])
    if torch.is_tensor(output):
        return first
    raise TypeError(
        f"Unsupported self-attention output type: {type(output)}"
    )


def project_without_bias(o_proj: Any, value: torch.Tensor) -> torch.Tensor:
    weight = getattr(o_proj, "weight", None)
    if weight is None:
        raise AttributeError(
            f"Output projection {type(o_proj).__name__} has no weight"
        )
    return F.linear(
        value.to(dtype=weight.dtype),
        weight,
        bias=None,
    )


class SpatialHeadKnockout:
    """Context manager that alters selected head outputs during prefill only."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        selected_heads: Sequence[Tuple[int, int, float]],
        subject_index: int,
        reference_index: int,
        visual_indices: Sequence[int],
        prompt_length: int,
        mode: str,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.selected_heads = list(selected_heads)
        self.subject_index = int(subject_index)
        self.reference_index = int(reference_index)
        self.visual_indices = [int(index) for index in visual_indices]
        self.prompt_length = int(prompt_length)
        self.mode = mode

        self.handles: List[Any] = []
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.patched_layers: Counter = Counter()
        self.patch_events = 0

    def __enter__(self) -> "SpatialHeadKnockout":
        by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head, _ in self.selected_heads:
            by_layer[int(layer)].append(int(head))

        for layer_index, heads in by_layer.items():
            if not (0 <= layer_index < len(self.decoder_layers)):
                raise ValueError(
                    f"Selected layer {layer_index} outside decoder range"
                )
            attention = find_self_attention(
                self.decoder_layers[layer_index]
            )
            v_proj = getattr(attention, "v_proj", None)
            o_proj = getattr(attention, "o_proj", None)
            if v_proj is None or o_proj is None:
                raise AttributeError(
                    f"Layer {layer_index} attention does not expose "
                    "v_proj/o_proj; this script currently expects Qwen/LLaVA-"
                    "style projections."
                )

            def make_v_hook(layer_id: int):
                def v_hook(_module: Any, _inputs: Any, output: Any) -> None:
                    value = output[0] if isinstance(output, tuple) else output
                    if torch.is_tensor(value):
                        self.value_cache[layer_id] = value
                return v_hook

            def make_attention_hook(
                layer_id: int,
                selected_layer_heads: List[int],
                output_projection: Any,
            ):
                def attention_hook(
                    _module: Any,
                    _inputs: Any,
                    output: Any,
                ) -> Any:
                    attention_output = (
                        output[0]
                        if isinstance(output, (tuple, list))
                        else output
                    )
                    weights = find_attention_weights(output)
                    values_raw = self.value_cache.pop(layer_id, None)

                    if (
                        not torch.is_tensor(attention_output)
                        or weights is None
                        or values_raw is None
                    ):
                        return output
                    if attention_output.ndim != 3:
                        return output

                    batch, query_length, hidden_size = (
                        attention_output.shape
                    )
                    if query_length != self.prompt_length:
                        # Decode steps have query length 1. Patch prefill only.
                        return output

                    weights = weights.float()
                    if weights.shape[0] != batch:
                        return output
                    n_heads = int(weights.shape[1])
                    key_length = int(weights.shape[-1])
                    if (
                        int(weights.shape[-2]) != query_length
                        or key_length != self.prompt_length
                    ):
                        return output
                    if hidden_size % n_heads != 0:
                        raise RuntimeError(
                            f"Hidden size {hidden_size} not divisible by "
                            f"attention heads {n_heads}"
                        )

                    head_dim = hidden_size // n_heads
                    if values_raw.ndim != 3:
                        raise RuntimeError(
                            f"Unexpected v_proj output shape: "
                            f"{tuple(values_raw.shape)}"
                        )
                    if (
                        values_raw.shape[0] != batch
                        or values_raw.shape[1] != key_length
                        or values_raw.shape[-1] % head_dim != 0
                    ):
                        raise RuntimeError(
                            "v_proj output is incompatible with returned "
                            f"attention: values={tuple(values_raw.shape)}, "
                            f"weights={tuple(weights.shape)}, "
                            f"head_dim={head_dim}"
                        )

                    n_kv_heads = int(
                        values_raw.shape[-1] // head_dim
                    )
                    if n_heads % n_kv_heads != 0:
                        raise RuntimeError(
                            f"n_heads={n_heads} is not divisible by "
                            f"n_kv_heads={n_kv_heads}"
                        )

                    values = values_raw.float().reshape(
                        batch,
                        key_length,
                        n_kv_heads,
                        head_dim,
                    )
                    values = values.permute(0, 2, 1, 3)
                    repeat = n_heads // n_kv_heads
                    if repeat > 1:
                        values = values.repeat_interleave(
                            repeat,
                            dim=1,
                        )

                    valid_heads = sorted(set(selected_layer_heads))
                    for head in valid_heads:
                        if not (0 <= head < n_heads):
                            raise ValueError(
                                f"Layer {layer_id} head {head} outside "
                                f"[0, {n_heads - 1}]"
                            )
                    query_positions = [
                        self.subject_index,
                        self.reference_index,
                    ]
                    for query in query_positions:
                        if not (0 <= query < query_length):
                            raise ValueError(
                                f"Object query index {query} outside "
                                f"[0, {query_length - 1}]"
                            )
                    visual = [
                        index
                        for index in self.visual_indices
                        if 0 <= index < key_length
                    ]
                    if not visual:
                        raise RuntimeError(
                            "No valid visual key indices for knockout"
                        )

                    head_index = torch.as_tensor(
                        valid_heads,
                        device=weights.device,
                        dtype=torch.long,
                    )
                    query_index = torch.as_tensor(
                        query_positions,
                        device=weights.device,
                        dtype=torch.long,
                    )
                    visual_index = torch.as_tensor(
                        visual,
                        device=weights.device,
                        dtype=torch.long,
                    )

                    # [B, selected_heads, object_queries, K]
                    selected_weights = weights.index_select(
                        1,
                        head_index,
                    ).index_select(
                        2,
                        query_index,
                    )
                    selected_values = values.index_select(
                        1,
                        head_index,
                    )

                    old_output = torch.einsum(
                        "bhqk,bhkd->bhqd",
                        selected_weights,
                        selected_values,
                    )

                    if self.mode == "visual_value_zero":
                        visual_weights = selected_weights.index_select(
                            -1,
                            visual_index,
                        )
                        visual_values = selected_values.index_select(
                            2,
                            visual_index,
                        )
                        removed = torch.einsum(
                            "bhqv,bhvd->bhqd",
                            visual_weights,
                            visual_values,
                        )
                        delta = -removed

                    elif self.mode == "weight_zero_renorm":
                        new_weights = selected_weights.clone()
                        new_weights[..., visual_index] = 0.0
                        denominator = new_weights.sum(
                            dim=-1,
                            keepdim=True,
                        )
                        if torch.any(denominator <= 1e-12):
                            raise RuntimeError(
                                "Removing visual attention left no non-visual "
                                "attention mass for at least one query/head"
                            )
                        new_weights = new_weights / denominator
                        new_output = torch.einsum(
                            "bhqk,bhkd->bhqd",
                            new_weights,
                            selected_values,
                        )
                        delta = new_output - old_output

                    elif self.mode == "head_output_zero":
                        delta = -old_output

                    else:
                        raise ValueError(
                            f"Unsupported knockout mode: {self.mode}"
                        )

                    delta_heads = torch.zeros(
                        (
                            batch,
                            n_heads,
                            query_length,
                            head_dim,
                        ),
                        device=attention_output.device,
                        dtype=torch.float32,
                    )
                    for local_head, head in enumerate(valid_heads):
                        for local_query, query in enumerate(
                            query_positions
                        ):
                            delta_heads[
                                :,
                                head,
                                query,
                                :,
                            ] = delta[
                                :,
                                local_head,
                                local_query,
                                :,
                            ].to(delta_heads.device)

                    delta_concat = (
                        delta_heads.permute(0, 2, 1, 3)
                        .contiguous()
                        .reshape(batch, query_length, hidden_size)
                    )
                    projected_delta = project_without_bias(
                        output_projection,
                        delta_concat,
                    )
                    new_attention_output = (
                        attention_output
                        + projected_delta.to(
                            dtype=attention_output.dtype,
                            device=attention_output.device,
                        )
                    )

                    self.patch_events += 1
                    self.patched_layers[layer_id] += 1
                    return replace_first_output(
                        output,
                        new_attention_output,
                    )

                return attention_hook

            self.handles.append(
                v_proj.register_forward_hook(
                    make_v_hook(layer_index)
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    make_attention_hook(
                        layer_index,
                        list(heads),
                        o_proj,
                    )
                )
            )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback_value: Any,
    ) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.value_cache.clear()


def generate_for_analysis(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    max_new_tokens: int,
) -> str:
    input_length = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        sequences = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=False,
        )
    return decode_new_tokens(
        processor,
        sequences,
        input_length,
    )


def condition_summary(
    rows: Sequence[Dict[str, Any]],
    condition_names: Sequence[str],
) -> Dict[str, Any]:
    baseline_valid = [
        row for row in rows if row.get("baseline_prediction")
    ]
    summary: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_baseline_valid": len(baseline_valid),
        "baseline_accuracy": (
            float(np.mean([
                bool(row.get("baseline_correct"))
                for row in baseline_valid
            ]))
            if baseline_valid else None
        ),
        "conditions": {},
    }

    for condition in condition_names:
        valid = [
            row
            for row in rows
            if row.get("baseline_prediction")
            and row.get("conditions", {})
            .get(condition, {})
            .get("prediction")
        ]
        fixed = broken = changed = 0
        per_relation: Dict[str, Dict[str, int]] = {
            relation: {
                "n": 0,
                "baseline_correct": 0,
                "condition_correct": 0,
                "fixed": 0,
                "broken": 0,
            }
            for relation in RELATIONS
        }

        for row in valid:
            result = row["conditions"][condition]
            base_correct = bool(row["baseline_correct"])
            new_correct = bool(result["correct"])
            changed += int(
                row["baseline_prediction"] != result["prediction"]
            )
            fixed += int((not base_correct) and new_correct)
            broken += int(base_correct and (not new_correct))

            relation = row["gt"]
            stats = per_relation[relation]
            stats["n"] += 1
            stats["baseline_correct"] += int(base_correct)
            stats["condition_correct"] += int(new_correct)
            stats["fixed"] += int(
                (not base_correct) and new_correct
            )
            stats["broken"] += int(
                base_correct and (not new_correct)
            )

        baseline_accuracy = (
            float(np.mean([
                bool(row["baseline_correct"]) for row in valid
            ]))
            if valid else None
        )
        condition_accuracy = (
            float(np.mean([
                bool(
                    row["conditions"][condition]["correct"]
                )
                for row in valid
            ]))
            if valid else None
        )

        relation_output: Dict[str, Any] = {}
        for relation, stats in per_relation.items():
            n = stats["n"]
            relation_output[relation] = {
                "n": n,
                "baseline_accuracy": (
                    stats["baseline_correct"] / n if n else None
                ),
                "condition_accuracy": (
                    stats["condition_correct"] / n if n else None
                ),
                "fixed": stats["fixed"],
                "broken": stats["broken"],
            }

        summary["conditions"][condition] = {
            "n_valid": len(valid),
            "baseline_accuracy": baseline_accuracy,
            "condition_accuracy": condition_accuracy,
            "absolute_change": (
                condition_accuracy - baseline_accuracy
                if baseline_accuracy is not None
                and condition_accuracy is not None
                else None
            ),
            "fixed": fixed,
            "broken": broken,
            "net_fixed_minus_broken": fixed - broken,
            "prediction_changed": changed,
            "per_relation": relation_output,
        }

    return summary


def print_summary(
    summary: Dict[str, Any],
    condition_heads: Dict[str, List[Tuple[int, int, float]]],
) -> None:
    print("\n" + "=" * 106)
    print("STEP 2: CAUSAL SPATIAL-HEAD KNOCKOUT")
    print("=" * 106)
    baseline = summary.get("baseline_accuracy")
    print(
        f"baseline generation: "
        f"{baseline:.4f} (n={summary.get('n_baseline_valid')})"
        if baseline is not None else
        "baseline generation: n/a"
    )

    for name, stats in summary["conditions"].items():
        heads = ", ".join(
            f"L{layer}H{head}"
            for layer, head, _ in condition_heads[name]
        )
        accuracy = stats["condition_accuracy"]
        delta = stats["absolute_change"]
        print(
            f"\n{name}\n"
            f"  heads: {heads}\n"
            f"  accuracy: "
            f"{accuracy:.4f} "
            f"(delta={delta:+.4f}, n={stats['n_valid']})\n"
            f"  fixed={stats['fixed']} | broken={stats['broken']} | "
            f"net={stats['net_fixed_minus_broken']:+d} | "
            f"changed={stats['prediction_changed']}"
        )
        for relation, relation_stats in stats["per_relation"].items():
            if relation_stats["n"] == 0:
                continue
            print(
                f"    {relation:6s} n={relation_stats['n']:4d} | "
                f"base={relation_stats['baseline_accuracy']:.4f} | "
                f"knockout={relation_stats['condition_accuracy']:.4f} | "
                f"fixed={relation_stats['fixed']} | "
                f"broken={relation_stats['broken']}"
            )


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0")

    groups = parse_csv_names(
        args.condition_groups,
        HEAD_GROUPS,
    )
    top_k_values = parse_positive_ints(args.top_k)
    step1_dir = Path(args.step1_dir)
    ranking = load_step1_rankings(step1_dir)
    conditions = build_conditions(
        ranking,
        groups,
        top_k_values,
        args.random_seed,
    )
    condition_names = list(conditions)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

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
            f"{len(missing_ids)} IDs; first={missing_ids[:10]}"
        )

    if args.model not in module.SPECS:
        raise ValueError(
            f"Model {args.model!r} not found in "
            "extract_two_object_relation_states.SPECS"
        )
    spec = module.SPECS[args.model]

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite or a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

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
    for condition, heads in conditions.items():
        for layer, head, _ in heads:
            if not (0 <= layer < len(decoder_layers)):
                raise ValueError(
                    f"Step 1 selected layer {layer}, but current model "
                    f"has {len(decoder_layers)} decoder layers"
                )

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "prompt_jsonl": str(prompt_path),
        "model": args.model,
        "repo_id": spec.repo_id,
        "transformers_version": transformers.__version__,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "step1_dir": str(step1_dir),
        "ranking_file": ranking["path"],
        "knockout_mode": args.knockout_mode,
        "groups": groups,
        "top_k": top_k_values,
        "conditions": {
            name: [
                {
                    "layer": layer,
                    "head": head,
                    "ranking_score": score,
                }
                for layer, head, score in heads
            ]
            for name, heads in conditions.items()
        },
        "max_new_tokens": args.max_new_tokens,
        "n_records": len(records),
        "audit": audit,
        "uses_gt_for_intervention": False,
        "updates_model_weights": False,
        "changes_prompt": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Decoder: {decoder_path}, n={len(decoder_layers)}"
    )
    print(f"Knockout mode: {args.knockout_mode}")
    for name, heads in conditions.items():
        print(
            f"{name}: "
            + ", ".join(
                f"L{layer}H{head}({score:.6f})"
                for layer, head, score in heads
            )
        )

    completed = 0
    started = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"step2-knockout:{args.model}",
        ):
            sid = int(record.sid)
            batch = None
            image = None
            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question_text = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATIONS:
                    raise ValueError(
                        f"Unsupported GT for sid={sid}: {gt!r}"
                    )

                image = record_image(record)
                rendered = build_prompt(
                    processor,
                    question_text,
                )
                batch = processor(
                    text=[rendered],
                    images=[image],
                    return_tensors="pt",
                )
                batch = move_batch(batch, device)

                input_ids = (
                    batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
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
                prompt_length = len(input_ids)

                baseline_text = generate_for_analysis(
                    model=model,
                    processor=processor,
                    batch=batch,
                    max_new_tokens=args.max_new_tokens,
                )
                baseline_prediction = normalize_relation(
                    baseline_text
                )
                baseline_correct = bool(
                    baseline_prediction is not None
                    and baseline_prediction == gt
                )

                condition_results: Dict[str, Any] = {}
                for condition, heads in conditions.items():
                    with SpatialHeadKnockout(
                        decoder_layers=decoder_layers,
                        selected_heads=heads,
                        subject_index=subject_index,
                        reference_index=reference_index,
                        visual_indices=visual_indices,
                        prompt_length=prompt_length,
                        mode=args.knockout_mode,
                    ) as intervention:
                        generated_text = generate_for_analysis(
                            model=model,
                            processor=processor,
                            batch=batch,
                            max_new_tokens=args.max_new_tokens,
                        )

                    prediction = normalize_relation(
                        generated_text
                    )
                    correct = bool(
                        prediction is not None
                        and prediction == gt
                    )
                    if intervention.patch_events == 0:
                        raise RuntimeError(
                            f"Condition {condition} registered no knockout "
                            "events. Confirm eager attention returns weights."
                        )

                    condition_results[condition] = {
                        "prediction": prediction,
                        "correct": correct,
                        "generated_text": generated_text,
                        "patch_events": intervention.patch_events,
                        "patched_layers": {
                            str(layer): count
                            for layer, count
                            in intervention.patched_layers.items()
                        },
                    }

                row = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "gt": gt,
                    "question": question_text,
                    "baseline_prediction": baseline_prediction,
                    "baseline_correct": baseline_correct,
                    "baseline_generated_text": baseline_text,
                    "conditions": condition_results,
                }
                append_jsonl(rows_path, row)
                completed += 1

                if should_print_sample(
                    completed,
                    args.print_every,
                ):
                    lines = [
                        f"\n[{completed}/{len(records)}] sid={sid} | "
                        f"{subject} -> {reference}",
                        f"  GT/base: {gt} / {baseline_prediction} "
                        f"(correct={int(baseline_correct)})",
                    ]
                    for condition, result in condition_results.items():
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
                            f"  {condition:28s}: "
                            f"{result['prediction']} | {status} | "
                            f"events={result['patch_events']}"
                        )
                    tqdm.write("\n".join(lines))

                del condition_results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": (
                            traceback.format_exc().splitlines()[-18:]
                        ),
                    },
                )
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

        rows = read_jsonl(rows_path)
        if not rows:
            raise RuntimeError(
                "No samples completed; inspect errors.jsonl"
            )
        summary = condition_summary(
            rows,
            condition_names,
        )
        summary["config"] = config
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

        print_summary(summary, conditions)
        print(f"\nSaved rows:    {rows_path}")
        print(f"Saved summary: {summary_path}")
        if errors_path.exists():
            print(f"Saved errors:  {errors_path}")

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
