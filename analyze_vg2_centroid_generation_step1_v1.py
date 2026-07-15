#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VG2 Step 1: centroid and natural-generation comparison.

For every image, this script evaluates the original question

    Q_AB: Where is object A in relation to object B?

and the swapped question

    Q_BA: Where is object B in relation to object A?

The model is not modified. The script uses natural greedy generation with eager attention
to expose prompt and answer-token attention while it records, for every
selected decoder layer and attention head:

- subject/reference object-token -> visual-token attention mass and centroid;
- question-last-token -> visual-token attention and object-token routing;
- prompt-last-token -> visual-token attention and object-token routing;
- first-generated-answer-token -> visual-token attention and object-token routing
  when generation naturally reaches a second decoding step;
- subject/reference attention-map separation;
- original/swap same-object map cosine and centroid distance;
- original/swap attention-centroid relation consistency;
- hidden-state similarity centroids for comparison with attention centroids;
- original/swapped answer distributions and relation-logit margins.

The swapped prompt is aligned back to the original semantic order A relative to
B before comparison.  Ground-truth labels are used only for reporting analysis
metrics, never for generation or attention extraction.

Outputs:
- samples.jsonl: readable per-sample summary;
- sample_arrays/<sid>.npz: full layer/head arrays;
- aggregate_metrics.npz: dataset-level layer/head arrays;
- summary.json: headline metrics and top heads;
- errors.jsonl: failed records.

Use --attn-impl eager.  SDPA/FlashAttention generally do not return complete
attention probabilities.
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
from types import SimpleNamespace
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


SCRIPT_VERSION = "vg2-centroid-generation-step1-v1"

DEFAULT_PROMPT_FILES = {
    "vg_two": Path("prompts/VG_QA_two_obj_with_answer_four_options.jsonl"),
}

RELATIONS = ("left", "right", "above", "below")

AUTO_LAYERS = {
    "llava-7b": 12,
    "llava-13b": 16,
    "qwen-3b": 24,
    "qwen-7b": 19,
    # Used only for readable auxiliary summaries. With --layers all, every
    # decoder layer is still evaluated.
    "qwen2-vl-7b": 16,
}

# Add the original Qwen2-VL-7B checkpoint locally, without requiring an edit
# to extract_two_object_relation_states.py.
EXTRA_MODEL_SPECS = {
    "qwen2-vl-7b": SimpleNamespace(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        model_class="Qwen2VLForConditionalGeneration",
        dtype_name="bfloat16",
        trust_remote_code=False,
    ),
}




QUERY_NAMES = (
    "subject",
    "reference",
    "question_last",
    "prompt_last",
)
RELATION_TO_INDEX = {
    "left": 0,
    "right": 1,
    "above": 2,
    "below": 3,
}
INDEX_TO_RELATION = np.asarray(
    ["left", "right", "above", "below"],
    dtype="<U8",
)
INVERSE_RELATION = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="vg_two", choices=["vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--vg-data-json",
        default="data/vg_qa_two_obj_four_options.json",
        help=(
            "Filtered VG2 data JSON aligned one-to-one with the four-option "
            "prompt JSONL. Each item is expected to start with image_id."
        ),
    )
    p.add_argument(
        "--vg-image-root",
        default="data/vg/images",
        help="Directory containing Visual Genome images such as 2354705.jpg.",
    )
    p.add_argument(
        "--prompt-jsonl",
        default=None,
        help=(
            "Four-option VG2 question file. Default: "
            "prompts/VG_QA_two_obj_with_answer_four_options.jsonl."
        ),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--layers",
        default="all",
        help=(
            "Comma-separated zero-based decoder layers, for example 12,18,24,30, "
            "or 'all'."
        ),
    )
    p.add_argument(
        "--report-layer",
        default="auto",
        help="Layer used for readable per-sample summaries, or 'auto'.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help=(
            "Maximum number of newly generated tokens. No minimum generation "
            "length is forced; generation may stop naturally at EOS."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for hidden-state similarity centroid maps.",
    )
    p.add_argument(
        "--save-object-maps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save normalized subject/reference visual attention maps for every "
            "layer/head. This substantially increases disk use."
        ),
    )
    p.add_argument(
        "--array-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Floating dtype used in per-sample NPZ files.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one sample summary every N completed samples; 0 disables.",
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


def merged_model_specs(module: Any) -> Dict[str, Any]:
    specs = dict(getattr(module, "SPECS", {}) or {})
    specs.update(EXTRA_MODEL_SPECS)
    return specs


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def find_vg_image(image_root: Path, image_id: Any) -> Optional[Path]:
    """Resolve one VG image without assuming a single filename extension."""
    stem = str(image_id).strip()
    for suffix in IMAGE_SUFFIXES:
        candidate = image_root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    # Common Visual Genome directory layouts.
    for subdir in ("VG_100K", "VG_100K_2", "images"):
        base = image_root / subdir
        for suffix in IMAGE_SUFFIXES:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def load_vg2_records(
    data_json: Path,
    image_root: Path,
    max_samples: Optional[int],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Load filtered VG2 triples while preserving their new sequential IDs."""
    if not data_json.exists():
        raise FileNotFoundError(f"Missing filtered VG2 JSON: {data_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing VG image root: {image_root}")

    with data_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(
            f"{data_json} must contain a top-level list, got "
            f"{type(data).__name__}"
        )

    records: List[Any] = []
    missing_images: List[Dict[str, Any]] = []
    malformed: List[Dict[str, Any]] = []

    for sid, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            malformed.append({"sid": sid, "item": repr(item)[:500]})
            continue

        image_id = item[0]
        image_path = find_vg_image(image_root, image_id)
        if image_path is None:
            missing_images.append({"sid": sid, "image_id": str(image_id)})
            continue

        records.append(SimpleNamespace(
            sid=sid,
            image_id=str(image_id),
            image_path=image_path,
            original_caption=(str(item[1]) if len(item) > 1 else None),
            swapped_caption=(str(item[2]) if len(item) > 2 else None),
        ))
        if max_samples is not None and len(records) >= int(max_samples):
            break

    audit = {
        "data_json": str(data_json),
        "image_root": str(image_root),
        "source_rows": len(data),
        "usable_records": len(records),
        "missing_image_count": len(missing_images),
        "malformed_count": len(malformed),
        "missing_image_examples": missing_images[:20],
        "malformed_examples": malformed[:20],
    }
    return records, audit


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


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    result: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0:
            layer += n_layers
        if not (0 <= layer < n_layers):
            raise ValueError(
                f"Layer {layer} is outside decoder range [0, {n_layers - 1}]"
            )
        result.append(layer)
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("--layers resolved to an empty list")
    return result


def resolve_report_layer(
    value: str,
    model_name: str,
    selected_layers: Sequence[int],
) -> int:
    text = str(value).strip().lower()
    if text == "auto":
        preferred = AUTO_LAYERS.get(model_name)
        if preferred in selected_layers:
            return int(preferred)
        return int(selected_layers[len(selected_layers) // 2])
    layer = int(text)
    if layer not in selected_layers:
        raise ValueError(
            f"--report-layer {layer} is not present in --layers {list(selected_layers)}"
        )
    return layer


def locate_question_last_token(
    tokenizer: Any,
    input_ids: Sequence[int],
    question_text: str,
) -> int:
    variants = [
        str(question_text).strip(),
        " " + str(question_text).strip(),
        "\n" + str(question_text).strip(),
        "Answer with left, right, above, or below.",
        " Answer with left, right, above, or below.",
        "below.",
        " below.",
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
        if spans and len(ids) > 4:
            break
    if spans:
        return int(max(spans, key=lambda item: item[1])[1])

    # Conservative fallback: the last non-special token before the assistant
    # generation suffix. This is recorded in the sample summary.
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for index in range(len(input_ids) - 1, -1, -1):
        if int(input_ids[index]) not in special_ids:
            return int(index)
    return int(len(input_ids) - 1)


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
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
                f"No one-token generation variant found for relation {relation!r}"
            )
        result[relation] = token_ids
    return result


def relation_scores(
    score_vector: torch.Tensor,
    token_map: Dict[str, List[int]],
    gt: Optional[str],
) -> Dict[str, Any]:
    logits = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            token_map[relation],
            device=score_vector.device,
            dtype=torch.long,
        )
        logits.append(score_vector[ids].float().max())
    logits_tensor = torch.stack(logits)
    probs = torch.softmax(logits_tensor, dim=0)
    order = torch.argsort(logits_tensor, descending=True)
    top1 = int(order[0].item())
    top2 = int(order[1].item())
    top1_margin = float(
        (logits_tensor[top1] - logits_tensor[top2]).item()
    )

    gt_margin = None
    if gt in RELATION_TO_INDEX:
        gt_index = RELATION_TO_INDEX[gt]
        other = torch.cat([
            logits_tensor[:gt_index],
            logits_tensor[gt_index + 1:],
        ])
        gt_margin = float(
            (logits_tensor[gt_index] - other.max()).item()
        )

    return {
        "logits": logits_tensor.detach().cpu().numpy().astype(np.float32),
        "probs": probs.detach().cpu().numpy().astype(np.float32),
        "prediction": str(RELATIONS[top1]),
        "top1_margin": top1_margin,
        "gt_margin": gt_margin,
    }


def invert_relation(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return INVERSE_RELATION.get(value)


def aligned_swap_vector(values: np.ndarray) -> np.ndarray:
    """Align B-relative-to-A relation scores back to A-relative-to-B."""
    return values[[1, 0, 3, 2]]


def normalize_attention_tensor(
    value: torch.Tensor,
    expected_query_length: Optional[int] = None,
) -> torch.Tensor:
    """Return [heads, query, key] for a batch-size-one attention tensor."""
    if not torch.is_tensor(value):
        raise TypeError(f"Attention value is not a tensor: {type(value)}")
    tensor = value
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise RuntimeError(
            f"Expected attention with 3/4 dimensions, got {tuple(value.shape)}"
        )

    # Standard HF layout is [heads, query, key].  A defensive transpose handles
    # uncommon [query, heads, key] outputs.
    if (
        expected_query_length is not None
        and tensor.shape[1] != expected_query_length
        and tensor.shape[0] == expected_query_length
    ):
        tensor = tensor.permute(1, 0, 2)
    return tensor.float()


def generation_steps(
    output: Any,
    name: str,
) -> Tuple[Any, ...]:
    candidates = [
        getattr(output, name, None),
        getattr(output, f"decoder_{name}", None),
    ]
    for value in candidates:
        if isinstance(value, (tuple, list)) and len(value) > 0:
            return tuple(value)
    raise RuntimeError(
        f"Generation output did not return {name}. "
        "Use --attn-impl eager and a transformers/model version that supports "
        "output_attentions/output_hidden_states during generate()."
    )


def step_layers(step_value: Any) -> Tuple[Any, ...]:
    if isinstance(step_value, (tuple, list)):
        return tuple(step_value)
    raise RuntimeError(
        f"Expected a tuple/list of layer tensors, got {type(step_value)}"
    )


def query_attention_metrics(
    rows: torch.Tensor,
    visual_indices: Sequence[int],
    coords: torch.Tensor,
    subject_index: int,
    reference_index: int,
) -> Dict[str, torch.Tensor]:
    """rows: [heads, queries, key]."""
    visual_index_tensor = torch.as_tensor(
        visual_indices,
        device=rows.device,
        dtype=torch.long,
    )
    visual = rows.index_select(-1, visual_index_tensor)
    visual_mass = visual.sum(dim=-1)
    normalized = visual / visual_mass[..., None].clamp_min(1e-12)
    centroids = torch.einsum("hqv,vd->hqd", normalized, coords.float())
    entropy = entropy_confidence(normalized)
    peak = normalized.max(dim=-1).values

    subject_attention = rows[..., subject_index]
    reference_attention = rows[..., reference_index]
    routing_sum = subject_attention + reference_attention
    routing_balance = (
        torch.minimum(subject_attention, reference_attention)
        / torch.maximum(subject_attention, reference_attention).clamp_min(1e-12)
    )
    return {
        "visual_mass": visual_mass,
        "visual_maps": normalized,
        "centroids": centroids,
        "entropy_confidence": entropy,
        "peak": peak,
        "to_subject": subject_attention,
        "to_reference": reference_attention,
        "routing_sum": routing_sum,
        "routing_balance": routing_balance,
    }


def relation_codes_from_centroids(
    centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """centroids [..., 2 objects, 2 coordinates]."""
    dx = centroids[..., 0, 0] - centroids[..., 1, 0]
    dy = centroids[..., 0, 1] - centroids[..., 1, 1]
    ax = np.abs(dx)
    ay = np.abs(dy)
    horizontal = ax >= ay
    codes = np.where(
        horizontal,
        np.where(dx < 0.0, 0, 1),
        np.where(dy < 0.0, 2, 3),
    ).astype(np.int8)
    axis_confidence = (
        np.abs(ax - ay) / (ax + ay + 1e-8)
    ).astype(np.float32)
    return codes, axis_confidence


def similarity_layer_metrics(
    *,
    hidden_layers: Sequence[torch.Tensor],
    selected_layers: Sequence[int],
    subject_index: int,
    reference_index: int,
    visual_indices: Sequence[int],
    coords: torch.Tensor,
    temperature: float,
) -> Dict[str, np.ndarray]:
    n_selected = len(selected_layers)
    n_visual = len(visual_indices)
    centroids = np.zeros((n_selected, 2, 2), dtype=np.float32)
    maps = np.zeros((n_selected, 2, n_visual), dtype=np.float32)
    separation = np.zeros(n_selected, dtype=np.float32)
    entropy = np.zeros((n_selected, 2), dtype=np.float32)

    visual_index_tensor = torch.as_tensor(
        visual_indices,
        device=coords.device,
        dtype=torch.long,
    )

    for out_index, layer in enumerate(selected_layers):
        hidden = hidden_layers[layer + 1][0].float()
        visual = hidden.index_select(0, visual_index_tensor)
        visual_norm = F.normalize(visual, dim=-1)

        object_states = torch.stack([
            hidden[subject_index],
            hidden[reference_index],
        ])
        logits = torch.matmul(
            F.normalize(object_states, dim=-1),
            visual_norm.T,
        ) / float(temperature)
        weights = torch.softmax(logits, dim=-1)
        centers = torch.matmul(weights, coords.float())

        maps[out_index] = (
            weights.detach().cpu().numpy().astype(np.float32)
        )
        centroids[out_index] = (
            centers.detach().cpu().numpy().astype(np.float32)
        )
        separation[out_index] = float(
            (0.5 * torch.sum(torch.abs(weights[0] - weights[1]))).item()
        )
        entropy[out_index] = (
            entropy_confidence(weights)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    prediction, axis_confidence = relation_codes_from_centroids(centroids)
    return {
        "maps": maps,
        "centroids": centroids,
        "separation": separation,
        "entropy_confidence": entropy,
        "prediction": prediction,
        "axis_confidence": axis_confidence,
    }


def analyze_prompt(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    question_text: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    temperature: float,
    max_new_tokens: int,
    relation_token_map: Dict[str, List[int]],
    gt: Optional[str],
) -> Dict[str, Any]:
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
    question_last_index = locate_question_last_token(
        processor.tokenizer,
        input_ids,
        question_text,
    )
    prompt_last_index = input_length - 1

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
            f"Could not construct coordinates for {len(visual_indices)} visual tokens"
        )

    generation_length = int(max_new_tokens)
    if generation_length < 1:
        raise ValueError("max_new_tokens must be positive")
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=generation_length,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
        )

    sequences = generated.sequences
    generated_text = decode_new_tokens(
        processor,
        sequences,
        input_length,
    )
    prediction = normalize_relation(generated_text)
    first_token_id = int(sequences[0, input_length].item())
    first_token_text = processor.tokenizer.decode(
        [first_token_id],
        skip_special_tokens=False,
    )

    scores = tuple(getattr(generated, "scores", ()) or ())
    if not scores:
        raise RuntimeError("Generation output did not include next-token scores")
    relation_score_data = relation_scores(
        scores[0][0],
        relation_token_map,
        gt,
    )

    attention_steps = generation_steps(generated, "attentions")
    hidden_steps = generation_steps(generated, "hidden_states")

    prompt_attentions = step_layers(attention_steps[0])
    prompt_hidden_layers = step_layers(hidden_steps[0])

    # A generated token can act as an attention query only at the following
    # autoregressive step. Do not force that following step. If generation
    # naturally stops after one token, answer-query routing is unavailable and
    # the corresponding arrays are saved as NaN without affecting prompt-based
    # similarity or attention-centroid metrics.
    answer_query_available = len(attention_steps) >= 2
    answer_attentions = (
        step_layers(attention_steps[1])
        if answer_query_available
        else None
    )

    n_heads = int(
        normalize_attention_tensor(
            prompt_attentions[selected_layers[0]],
            expected_query_length=input_length,
        ).shape[0]
    )
    n_selected = len(selected_layers)
    n_queries = len(QUERY_NAMES)
    n_visual = len(visual_indices)

    prompt_visual_mass = np.zeros(
        (n_selected, n_heads, n_queries),
        dtype=np.float32,
    )
    prompt_centroids = np.zeros(
        (n_selected, n_heads, n_queries, 2),
        dtype=np.float32,
    )
    prompt_entropy = np.zeros_like(prompt_visual_mass)
    prompt_peak = np.zeros_like(prompt_visual_mass)
    prompt_to_subject = np.zeros_like(prompt_visual_mass)
    prompt_to_reference = np.zeros_like(prompt_visual_mass)
    prompt_routing_sum = np.zeros_like(prompt_visual_mass)
    prompt_routing_balance = np.zeros_like(prompt_visual_mass)

    answer_visual_mass = np.full(
        (n_selected, n_heads),
        np.nan,
        dtype=np.float32,
    )
    answer_centroids = np.full(
        (n_selected, n_heads, 2),
        np.nan,
        dtype=np.float32,
    )
    answer_entropy = np.full_like(answer_visual_mass, np.nan)
    answer_peak = np.full_like(answer_visual_mass, np.nan)
    answer_to_subject = np.full_like(answer_visual_mass, np.nan)
    answer_to_reference = np.full_like(answer_visual_mass, np.nan)
    answer_routing_sum = np.full_like(answer_visual_mass, np.nan)
    answer_routing_balance = np.full_like(answer_visual_mass, np.nan)

    object_maps = np.zeros(
        (n_selected, n_heads, 2, n_visual),
        dtype=np.float32,
    )
    object_centroids = np.zeros(
        (n_selected, n_heads, 2, 2),
        dtype=np.float32,
    )
    object_separation = np.zeros(
        (n_selected, n_heads),
        dtype=np.float32,
    )

    query_indices = [
        subject_index,
        reference_index,
        question_last_index,
        prompt_last_index,
    ]

    for out_index, layer in enumerate(selected_layers):
        prompt_tensor = normalize_attention_tensor(
            prompt_attentions[layer],
            expected_query_length=input_length,
        )
        if int(prompt_tensor.shape[0]) != n_heads:
            raise RuntimeError(
                f"Head count changed at layer {layer}: "
                f"{prompt_tensor.shape[0]} vs {n_heads}"
            )
        rows = prompt_tensor[:, query_indices, :]
        metrics = query_attention_metrics(
            rows,
            visual_indices,
            coords,
            subject_index,
            reference_index,
        )

        prompt_visual_mass[out_index] = (
            metrics["visual_mass"].detach().cpu().numpy()
        )
        prompt_centroids[out_index] = (
            metrics["centroids"].detach().cpu().numpy()
        )
        prompt_entropy[out_index] = (
            metrics["entropy_confidence"].detach().cpu().numpy()
        )
        prompt_peak[out_index] = (
            metrics["peak"].detach().cpu().numpy()
        )
        prompt_to_subject[out_index] = (
            metrics["to_subject"].detach().cpu().numpy()
        )
        prompt_to_reference[out_index] = (
            metrics["to_reference"].detach().cpu().numpy()
        )
        prompt_routing_sum[out_index] = (
            metrics["routing_sum"].detach().cpu().numpy()
        )
        prompt_routing_balance[out_index] = (
            metrics["routing_balance"].detach().cpu().numpy()
        )

        maps = metrics["visual_maps"][:, :2, :]
        centers = metrics["centroids"][:, :2, :]
        object_maps[out_index] = maps.detach().cpu().numpy()
        object_centroids[out_index] = centers.detach().cpu().numpy()
        object_separation[out_index] = (
            0.5
            * torch.sum(torch.abs(maps[:, 0, :] - maps[:, 1, :]), dim=-1)
        ).detach().cpu().numpy()

        if answer_query_available:
            assert answer_attentions is not None
            answer_tensor = normalize_attention_tensor(
                answer_attentions[layer],
                expected_query_length=1,
            )
            answer_rows = answer_tensor[:, -1:, :]
            answer_metrics = query_attention_metrics(
                answer_rows,
                visual_indices,
                coords,
                subject_index,
                reference_index,
            )
            answer_visual_mass[out_index] = (
                answer_metrics["visual_mass"][:, 0].detach().cpu().numpy()
            )
            answer_centroids[out_index] = (
                answer_metrics["centroids"][:, 0, :].detach().cpu().numpy()
            )
            answer_entropy[out_index] = (
                answer_metrics["entropy_confidence"][:, 0]
                .detach()
                .cpu()
                .numpy()
            )
            answer_peak[out_index] = (
                answer_metrics["peak"][:, 0].detach().cpu().numpy()
            )
            answer_to_subject[out_index] = (
                answer_metrics["to_subject"][:, 0].detach().cpu().numpy()
            )
            answer_to_reference[out_index] = (
                answer_metrics["to_reference"][:, 0].detach().cpu().numpy()
            )
            answer_routing_sum[out_index] = (
                answer_metrics["routing_sum"][:, 0].detach().cpu().numpy()
            )
            answer_routing_balance[out_index] = (
                answer_metrics["routing_balance"][:, 0]
                .detach()
                .cpu()
                .numpy()
            )

    object_prediction, object_axis_confidence = (
        relation_codes_from_centroids(object_centroids)
    )

    similarity = similarity_layer_metrics(
        hidden_layers=prompt_hidden_layers,
        selected_layers=selected_layers,
        subject_index=subject_index,
        reference_index=reference_index,
        visual_indices=visual_indices,
        coords=coords,
        temperature=temperature,
    )

    result = {
        "input_length": input_length,
        "subject_index": subject_index,
        "reference_index": reference_index,
        "question_last_index": question_last_index,
        "prompt_last_index": prompt_last_index,
        "visual_indices": np.asarray(visual_indices, dtype=np.int32),
        "visual_coordinates": coords.detach().cpu().numpy().astype(np.float32),
        "n_heads": n_heads,
        "generated_text": generated_text,
        "prediction": prediction,
        "first_token_id": first_token_id,
        "first_token_text": first_token_text,
        "answer_query_available": bool(answer_query_available),
        "generation_steps_returned": int(len(attention_steps)),
        "relation_logits": relation_score_data["logits"],
        "relation_probs": relation_score_data["probs"],
        "relation_score_prediction": relation_score_data["prediction"],
        "relation_top1_margin": relation_score_data["top1_margin"],
        "relation_gt_margin": relation_score_data["gt_margin"],

        "prompt_visual_mass": prompt_visual_mass,
        "prompt_centroids": prompt_centroids,
        "prompt_entropy_confidence": prompt_entropy,
        "prompt_peak": prompt_peak,
        "prompt_to_subject": prompt_to_subject,
        "prompt_to_reference": prompt_to_reference,
        "prompt_routing_sum": prompt_routing_sum,
        "prompt_routing_balance": prompt_routing_balance,

        "answer_visual_mass": answer_visual_mass,
        "answer_centroids": answer_centroids,
        "answer_entropy_confidence": answer_entropy,
        "answer_peak": answer_peak,
        "answer_to_subject": answer_to_subject,
        "answer_to_reference": answer_to_reference,
        "answer_routing_sum": answer_routing_sum,
        "answer_routing_balance": answer_routing_balance,

        "object_maps": object_maps,
        "object_centroids": object_centroids,
        "object_separation": object_separation,
        "object_prediction": object_prediction,
        "object_axis_confidence": object_axis_confidence,

        "similarity_maps": similarity["maps"],
        "similarity_centroids": similarity["centroids"],
        "similarity_separation": similarity["separation"],
        "similarity_entropy_confidence": similarity["entropy_confidence"],
        "similarity_prediction": similarity["prediction"],
        "similarity_axis_confidence": similarity["axis_confidence"],
    }

    del generated, sequences
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def map_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = (
        np.linalg.norm(a, axis=-1)
        * np.linalg.norm(b, axis=-1)
        + 1e-12
    )
    return (numerator / denominator).astype(np.float32)


def map_js_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (a + b)
    a_safe = np.clip(a, 1e-12, None)
    b_safe = np.clip(b, 1e-12, None)
    m_safe = np.clip(midpoint, 1e-12, None)
    kl_a = np.sum(a_safe * (np.log(a_safe) - np.log(m_safe)), axis=-1)
    kl_b = np.sum(b_safe * (np.log(b_safe) - np.log(m_safe)), axis=-1)
    return (0.5 * (kl_a + kl_b)).astype(np.float32)


def head_mean_relation(
    maps: np.ndarray,
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """maps: [layers, heads, 2 objects, visual]."""
    mean_maps = maps.mean(axis=1)
    centroids = np.einsum("lov,vd->lod", mean_maps, coords)
    prediction, axis_confidence = relation_codes_from_centroids(centroids)
    return prediction, axis_confidence


def to_relation_name(code: int) -> str:
    if 0 <= int(code) < len(INDEX_TO_RELATION):
        return str(INDEX_TO_RELATION[int(code)])
    return "unknown"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def save_sample_arrays(
    path: Path,
    *,
    sid: int,
    selected_layers: Sequence[int],
    original: Dict[str, Any],
    swapped: Dict[str, Any],
    cross: Dict[str, np.ndarray],
    save_object_maps: bool,
    array_dtype: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    float_dtype = np.float16 if array_dtype == "float16" else np.float32

    arrays: Dict[str, Any] = {
        "sid": np.asarray(sid, dtype=np.int64),
        "layer_indices": np.asarray(selected_layers, dtype=np.int32),
        "query_names": np.asarray(QUERY_NAMES, dtype="<U24"),
        "relation_names": INDEX_TO_RELATION,
        "original_answer_query_available": np.asarray(
            original["answer_query_available"], dtype=np.bool_
        ),
        "swapped_answer_query_available": np.asarray(
            swapped["answer_query_available"], dtype=np.bool_
        ),

        "original_prompt_visual_mass": original["prompt_visual_mass"].astype(float_dtype),
        "original_prompt_centroids": original["prompt_centroids"].astype(float_dtype),
        "original_prompt_entropy_confidence": original[
            "prompt_entropy_confidence"
        ].astype(float_dtype),
        "original_prompt_peak": original["prompt_peak"].astype(float_dtype),
        "original_prompt_to_subject": original["prompt_to_subject"].astype(float_dtype),
        "original_prompt_to_reference": original[
            "prompt_to_reference"
        ].astype(float_dtype),
        "original_prompt_routing_sum": original["prompt_routing_sum"].astype(float_dtype),
        "original_prompt_routing_balance": original[
            "prompt_routing_balance"
        ].astype(float_dtype),

        "original_answer_visual_mass": original["answer_visual_mass"].astype(float_dtype),
        "original_answer_centroids": original["answer_centroids"].astype(float_dtype),
        "original_answer_entropy_confidence": original[
            "answer_entropy_confidence"
        ].astype(float_dtype),
        "original_answer_peak": original["answer_peak"].astype(float_dtype),
        "original_answer_to_subject": original["answer_to_subject"].astype(float_dtype),
        "original_answer_to_reference": original[
            "answer_to_reference"
        ].astype(float_dtype),
        "original_answer_routing_sum": original["answer_routing_sum"].astype(float_dtype),
        "original_answer_routing_balance": original[
            "answer_routing_balance"
        ].astype(float_dtype),

        "original_object_centroids": original["object_centroids"].astype(float_dtype),
        "original_object_separation": original["object_separation"].astype(float_dtype),
        "original_object_prediction": original["object_prediction"].astype(np.int8),
        "original_object_axis_confidence": original[
            "object_axis_confidence"
        ].astype(float_dtype),

        "swapped_prompt_visual_mass": swapped["prompt_visual_mass"].astype(float_dtype),
        "swapped_prompt_centroids": swapped["prompt_centroids"].astype(float_dtype),
        "swapped_prompt_entropy_confidence": swapped[
            "prompt_entropy_confidence"
        ].astype(float_dtype),
        "swapped_prompt_peak": swapped["prompt_peak"].astype(float_dtype),
        "swapped_prompt_to_subject": swapped["prompt_to_subject"].astype(float_dtype),
        "swapped_prompt_to_reference": swapped[
            "prompt_to_reference"
        ].astype(float_dtype),
        "swapped_prompt_routing_sum": swapped["prompt_routing_sum"].astype(float_dtype),
        "swapped_prompt_routing_balance": swapped[
            "prompt_routing_balance"
        ].astype(float_dtype),

        "swapped_answer_visual_mass": swapped["answer_visual_mass"].astype(float_dtype),
        "swapped_answer_centroids": swapped["answer_centroids"].astype(float_dtype),
        "swapped_answer_entropy_confidence": swapped[
            "answer_entropy_confidence"
        ].astype(float_dtype),
        "swapped_answer_peak": swapped["answer_peak"].astype(float_dtype),
        "swapped_answer_to_subject": swapped["answer_to_subject"].astype(float_dtype),
        "swapped_answer_to_reference": swapped[
            "answer_to_reference"
        ].astype(float_dtype),
        "swapped_answer_routing_sum": swapped["answer_routing_sum"].astype(float_dtype),
        "swapped_answer_routing_balance": swapped[
            "answer_routing_balance"
        ].astype(float_dtype),

        "swapped_object_centroids_role_order": swapped[
            "object_centroids"
        ].astype(float_dtype),
        "swapped_object_separation": swapped["object_separation"].astype(float_dtype),
        "swapped_object_prediction_role_order": swapped[
            "object_prediction"
        ].astype(np.int8),
        "swapped_object_axis_confidence_role_order": swapped[
            "object_axis_confidence"
        ].astype(float_dtype),

        "original_similarity_centroids": original[
            "similarity_centroids"
        ].astype(float_dtype),
        "original_similarity_separation": original[
            "similarity_separation"
        ].astype(float_dtype),
        "original_similarity_prediction": original[
            "similarity_prediction"
        ].astype(np.int8),
        "swapped_similarity_centroids_role_order": swapped[
            "similarity_centroids"
        ].astype(float_dtype),
        "swapped_similarity_separation": swapped[
            "similarity_separation"
        ].astype(float_dtype),
        "swapped_similarity_prediction_role_order": swapped[
            "similarity_prediction"
        ].astype(np.int8),

        "same_object_map_cosine": cross["same_object_map_cosine"].astype(float_dtype),
        "same_object_map_js": cross["same_object_map_js"].astype(float_dtype),
        "same_object_centroid_distance": cross[
            "same_object_centroid_distance"
        ].astype(float_dtype),
        "attention_swapped_aligned_prediction": cross[
            "attention_swapped_aligned_prediction"
        ].astype(np.int8),
        "attention_average_prediction": cross[
            "attention_average_prediction"
        ].astype(np.int8),
        "attention_relation_consistency": cross[
            "attention_relation_consistency"
        ].astype(np.int8),
        "similarity_swapped_aligned_prediction": cross[
            "similarity_swapped_aligned_prediction"
        ].astype(np.int8),
        "similarity_average_prediction": cross[
            "similarity_average_prediction"
        ].astype(np.int8),
        "similarity_relation_consistency": cross[
            "similarity_relation_consistency"
        ].astype(np.int8),

        # Overall attention: average the normalized visual-attention maps over
        # all heads within each decoder layer, then compute object centroids.
        "headmean_original_prediction": cross[
            "headmean_original_prediction"
        ].astype(np.int8),
        "headmean_swapped_aligned_prediction": cross[
            "headmean_swapped_aligned_prediction"
        ].astype(np.int8),
        "headmean_average_prediction": cross[
            "headmean_average_prediction"
        ].astype(np.int8),
    }

    if save_object_maps:
        arrays.update({
            "original_object_maps": original["object_maps"].astype(float_dtype),
            "swapped_object_maps_role_order": swapped[
                "object_maps"
            ].astype(float_dtype),
            "original_similarity_maps": original[
                "similarity_maps"
            ].astype(float_dtype),
            "swapped_similarity_maps_role_order": swapped[
                "similarity_maps"
            ].astype(float_dtype),
        })

    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def init_aggregate(
    n_layers: int,
    n_heads: int,
    n_queries: int,
) -> Dict[str, Any]:
    shape_lh = (n_layers, n_heads)
    shape_lhq = (n_layers, n_heads, n_queries)
    zeros_lh = lambda: np.zeros(shape_lh, dtype=np.float64)
    zeros_lhq = lambda: np.zeros(shape_lhq, dtype=np.float64)

    return {
        "n": 0,
        "generation_original_correct": 0,
        "generation_swapped_aligned_correct": 0,
        "generation_answer_swap_consistent": 0,
        "generation_original_valid": 0,
        "generation_swapped_valid": 0,
        "answer_distribution_cosine_sum": 0.0,
        "answer_distribution_js_sum": 0.0,

        "attention_original_correct": zeros_lh(),
        "attention_swapped_correct": zeros_lh(),
        "attention_average_correct": zeros_lh(),
        "attention_relation_consistency": zeros_lh(),
        "attention_count": zeros_lh(),

        "similarity_original_correct": np.zeros(n_layers, dtype=np.float64),
        "similarity_swapped_correct": np.zeros(n_layers, dtype=np.float64),
        "similarity_average_correct": np.zeros(n_layers, dtype=np.float64),
        "similarity_relation_consistency": np.zeros(n_layers, dtype=np.float64),
        "similarity_count": np.zeros(n_layers, dtype=np.float64),

        "same_object_map_cosine_sum": np.zeros(
            (n_layers, n_heads, 2),
            dtype=np.float64,
        ),
        "same_object_map_js_sum": np.zeros(
            (n_layers, n_heads, 2),
            dtype=np.float64,
        ),
        "same_object_centroid_distance_sum": np.zeros(
            (n_layers, n_heads, 2),
            dtype=np.float64,
        ),
        "original_object_separation_sum": zeros_lh(),
        "swapped_object_separation_sum": zeros_lh(),

        "original_prompt_visual_mass_sum": zeros_lhq(),
        "swapped_prompt_visual_mass_sum": zeros_lhq(),
        "original_prompt_to_subject_sum": zeros_lhq(),
        "original_prompt_to_reference_sum": zeros_lhq(),
        "swapped_prompt_to_subject_sum": zeros_lhq(),
        "swapped_prompt_to_reference_sum": zeros_lhq(),

        "original_answer_visual_mass_sum": zeros_lh(),
        "swapped_answer_visual_mass_sum": zeros_lh(),
        "original_answer_to_subject_sum": zeros_lh(),
        "original_answer_to_reference_sum": zeros_lh(),
        "swapped_answer_to_subject_sum": zeros_lh(),
        "swapped_answer_to_reference_sum": zeros_lh(),
        "original_answer_routing_balance_sum": zeros_lh(),
        "swapped_answer_routing_balance_sum": zeros_lh(),
        "original_answer_query_count": 0,
        "swapped_answer_query_count": 0,

        "correct_group_count": 0,
        "wrong_group_count": 0,
        "correct_answer_query_count": 0,
        "wrong_answer_query_count": 0,
        "answer_to_subject_correct_sum": zeros_lh(),
        "answer_to_reference_correct_sum": zeros_lh(),
        "answer_routing_balance_correct_sum": zeros_lh(),
        "answer_visual_mass_correct_sum": zeros_lh(),
        "answer_to_subject_wrong_sum": zeros_lh(),
        "answer_to_reference_wrong_sum": zeros_lh(),
        "answer_routing_balance_wrong_sum": zeros_lh(),
        "answer_visual_mass_wrong_sum": zeros_lh(),

        "headmean_original_correct": np.zeros(n_layers, dtype=np.float64),
        "headmean_swapped_correct": np.zeros(n_layers, dtype=np.float64),
        "headmean_average_correct": np.zeros(n_layers, dtype=np.float64),

        "per_relation": {
            relation: {
                "n": 0,
                "generation_original_correct": 0,
                "generation_swapped_aligned_correct": 0,
                "headmean_original_correct": np.zeros(n_layers, dtype=np.float64),
                "headmean_swapped_correct": np.zeros(n_layers, dtype=np.float64),
                "headmean_average_correct": np.zeros(n_layers, dtype=np.float64),
                "similarity_original_correct": np.zeros(n_layers, dtype=np.float64),
                "similarity_swapped_correct": np.zeros(n_layers, dtype=np.float64),
                "similarity_average_correct": np.zeros(n_layers, dtype=np.float64),
            }
            for relation in RELATIONS
        },
    }


def update_aggregate(
    aggregate: Dict[str, Any],
    *,
    gt: str,
    original: Dict[str, Any],
    swapped: Dict[str, Any],
    cross: Dict[str, np.ndarray],
    headmean_original: np.ndarray,
    headmean_swapped: np.ndarray,
    headmean_average: np.ndarray,
    original_correct: bool,
) -> None:
    gt_code = RELATION_TO_INDEX[gt]
    aggregate["n"] += 1

    original_prediction = original["prediction"]
    swapped_aligned_prediction = invert_relation(swapped["prediction"])

    if original_prediction is not None:
        aggregate["generation_original_valid"] += 1
        aggregate["generation_original_correct"] += int(
            original_prediction == gt
        )
    if swapped_aligned_prediction is not None:
        aggregate["generation_swapped_valid"] += 1
        aggregate["generation_swapped_aligned_correct"] += int(
            swapped_aligned_prediction == gt
        )
    if original_prediction is not None and swapped["prediction"] is not None:
        aggregate["generation_answer_swap_consistent"] += int(
            original_prediction == swapped_aligned_prediction
        )

    original_probs = original["relation_probs"]
    swap_probs_aligned = aligned_swap_vector(swapped["relation_probs"])
    aggregate["answer_distribution_cosine_sum"] += float(
        np.dot(original_probs, swap_probs_aligned)
        / (
            np.linalg.norm(original_probs)
            * np.linalg.norm(swap_probs_aligned)
            + 1e-12
        )
    )
    aggregate["answer_distribution_js_sum"] += float(
        map_js_divergence(
            original_probs[None, :],
            swap_probs_aligned[None, :],
        )[0]
    )

    aggregate["attention_original_correct"] += (
        original["object_prediction"] == gt_code
    )
    aggregate["attention_swapped_correct"] += (
        cross["attention_swapped_aligned_prediction"] == gt_code
    )
    aggregate["attention_average_correct"] += (
        cross["attention_average_prediction"] == gt_code
    )
    aggregate["attention_relation_consistency"] += (
        cross["attention_relation_consistency"]
    )
    aggregate["attention_count"] += 1.0

    aggregate["similarity_original_correct"] += (
        original["similarity_prediction"] == gt_code
    )
    aggregate["similarity_swapped_correct"] += (
        cross["similarity_swapped_aligned_prediction"] == gt_code
    )
    aggregate["similarity_average_correct"] += (
        cross["similarity_average_prediction"] == gt_code
    )
    aggregate["similarity_relation_consistency"] += (
        cross["similarity_relation_consistency"]
    )
    aggregate["similarity_count"] += 1.0

    aggregate["same_object_map_cosine_sum"] += (
        cross["same_object_map_cosine"]
    )
    aggregate["same_object_map_js_sum"] += cross["same_object_map_js"]
    aggregate["same_object_centroid_distance_sum"] += (
        cross["same_object_centroid_distance"]
    )
    aggregate["original_object_separation_sum"] += (
        original["object_separation"]
    )
    aggregate["swapped_object_separation_sum"] += (
        swapped["object_separation"]
    )

    aggregate["original_prompt_visual_mass_sum"] += (
        original["prompt_visual_mass"]
    )
    aggregate["swapped_prompt_visual_mass_sum"] += (
        swapped["prompt_visual_mass"]
    )
    aggregate["original_prompt_to_subject_sum"] += (
        original["prompt_to_subject"]
    )
    aggregate["original_prompt_to_reference_sum"] += (
        original["prompt_to_reference"]
    )
    aggregate["swapped_prompt_to_subject_sum"] += (
        swapped["prompt_to_subject"]
    )
    aggregate["swapped_prompt_to_reference_sum"] += (
        swapped["prompt_to_reference"]
    )

    if original["answer_query_available"]:
        aggregate["original_answer_query_count"] += 1
        aggregate["original_answer_visual_mass_sum"] += (
            original["answer_visual_mass"]
        )
        aggregate["original_answer_to_subject_sum"] += (
            original["answer_to_subject"]
        )
        aggregate["original_answer_to_reference_sum"] += (
            original["answer_to_reference"]
        )
        aggregate["original_answer_routing_balance_sum"] += (
            original["answer_routing_balance"]
        )

    if swapped["answer_query_available"]:
        aggregate["swapped_answer_query_count"] += 1
        aggregate["swapped_answer_visual_mass_sum"] += (
            swapped["answer_visual_mass"]
        )
        aggregate["swapped_answer_to_subject_sum"] += (
            swapped["answer_to_subject"]
        )
        aggregate["swapped_answer_to_reference_sum"] += (
            swapped["answer_to_reference"]
        )
        aggregate["swapped_answer_routing_balance_sum"] += (
            swapped["answer_routing_balance"]
        )

    if original_correct:
        aggregate["correct_group_count"] += 1
        if original["answer_query_available"]:
            aggregate["correct_answer_query_count"] += 1
            aggregate["answer_to_subject_correct_sum"] += (
                original["answer_to_subject"]
            )
            aggregate["answer_to_reference_correct_sum"] += (
                original["answer_to_reference"]
            )
            aggregate["answer_routing_balance_correct_sum"] += (
                original["answer_routing_balance"]
            )
            aggregate["answer_visual_mass_correct_sum"] += (
                original["answer_visual_mass"]
            )
    else:
        aggregate["wrong_group_count"] += 1
        if original["answer_query_available"]:
            aggregate["wrong_answer_query_count"] += 1
            aggregate["answer_to_subject_wrong_sum"] += (
                original["answer_to_subject"]
            )
            aggregate["answer_to_reference_wrong_sum"] += (
                original["answer_to_reference"]
            )
            aggregate["answer_routing_balance_wrong_sum"] += (
                original["answer_routing_balance"]
            )
            aggregate["answer_visual_mass_wrong_sum"] += (
                original["answer_visual_mass"]
            )

    aggregate["headmean_original_correct"] += (
        headmean_original == gt_code
    )
    aggregate["headmean_swapped_correct"] += (
        headmean_swapped == gt_code
    )
    aggregate["headmean_average_correct"] += (
        headmean_average == gt_code
    )

    relation_stats = aggregate["per_relation"][gt]
    relation_stats["n"] += 1
    relation_stats["generation_original_correct"] += int(
        original_prediction == gt
    )
    relation_stats["generation_swapped_aligned_correct"] += int(
        swapped_aligned_prediction == gt
    )
    relation_stats["headmean_original_correct"] += (
        headmean_original == gt_code
    )
    relation_stats["headmean_swapped_correct"] += (
        headmean_swapped == gt_code
    )
    relation_stats["headmean_average_correct"] += (
        headmean_average == gt_code
    )
    relation_stats["similarity_original_correct"] += (
        original["similarity_prediction"] == gt_code
    )
    relation_stats["similarity_swapped_correct"] += (
        cross["similarity_swapped_aligned_prediction"] == gt_code
    )
    relation_stats["similarity_average_correct"] += (
        cross["similarity_average_prediction"] == gt_code
    )


def finalize_aggregate(
    aggregate: Dict[str, Any],
    selected_layers: Sequence[int],
    report_layer_position: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    n = max(1, int(aggregate["n"]))
    count_lh = np.maximum(aggregate["attention_count"], 1.0)
    count_l = np.maximum(aggregate["similarity_count"], 1.0)

    def available_mean(total: np.ndarray, count: int) -> np.ndarray:
        if int(count) <= 0:
            return np.full_like(total, np.nan, dtype=np.float32)
        return (total / int(count)).astype(np.float32)

    original_answer_n = int(aggregate["original_answer_query_count"])
    swapped_answer_n = int(aggregate["swapped_answer_query_count"])

    metrics: Dict[str, np.ndarray] = {
        "layer_indices": np.asarray(selected_layers, dtype=np.int32),
        "query_names": np.asarray(QUERY_NAMES, dtype="<U24"),

        "attention_original_accuracy": (
            aggregate["attention_original_correct"] / count_lh
        ).astype(np.float32),
        "attention_swapped_accuracy": (
            aggregate["attention_swapped_correct"] / count_lh
        ).astype(np.float32),
        "attention_average_accuracy": (
            aggregate["attention_average_correct"] / count_lh
        ).astype(np.float32),
        "attention_relation_consistency": (
            aggregate["attention_relation_consistency"] / count_lh
        ).astype(np.float32),

        "similarity_original_accuracy": (
            aggregate["similarity_original_correct"] / count_l
        ).astype(np.float32),
        "similarity_swapped_accuracy": (
            aggregate["similarity_swapped_correct"] / count_l
        ).astype(np.float32),
        "similarity_average_accuracy": (
            aggregate["similarity_average_correct"] / count_l
        ).astype(np.float32),
        "similarity_relation_consistency": (
            aggregate["similarity_relation_consistency"] / count_l
        ).astype(np.float32),

        "same_object_map_cosine": (
            aggregate["same_object_map_cosine_sum"] / n
        ).astype(np.float32),
        "same_object_map_js": (
            aggregate["same_object_map_js_sum"] / n
        ).astype(np.float32),
        "same_object_centroid_distance": (
            aggregate["same_object_centroid_distance_sum"] / n
        ).astype(np.float32),
        "original_object_separation": (
            aggregate["original_object_separation_sum"] / n
        ).astype(np.float32),
        "swapped_object_separation": (
            aggregate["swapped_object_separation_sum"] / n
        ).astype(np.float32),

        "original_prompt_visual_mass": (
            aggregate["original_prompt_visual_mass_sum"] / n
        ).astype(np.float32),
        "swapped_prompt_visual_mass": (
            aggregate["swapped_prompt_visual_mass_sum"] / n
        ).astype(np.float32),
        "original_prompt_to_subject": (
            aggregate["original_prompt_to_subject_sum"] / n
        ).astype(np.float32),
        "original_prompt_to_reference": (
            aggregate["original_prompt_to_reference_sum"] / n
        ).astype(np.float32),
        "swapped_prompt_to_subject": (
            aggregate["swapped_prompt_to_subject_sum"] / n
        ).astype(np.float32),
        "swapped_prompt_to_reference": (
            aggregate["swapped_prompt_to_reference_sum"] / n
        ).astype(np.float32),

        "original_answer_visual_mass": available_mean(
            aggregate["original_answer_visual_mass_sum"],
            original_answer_n,
        ),
        "swapped_answer_visual_mass": available_mean(
            aggregate["swapped_answer_visual_mass_sum"],
            swapped_answer_n,
        ),
        "original_answer_to_subject": available_mean(
            aggregate["original_answer_to_subject_sum"],
            original_answer_n,
        ),
        "original_answer_to_reference": available_mean(
            aggregate["original_answer_to_reference_sum"],
            original_answer_n,
        ),
        "swapped_answer_to_subject": available_mean(
            aggregate["swapped_answer_to_subject_sum"],
            swapped_answer_n,
        ),
        "swapped_answer_to_reference": available_mean(
            aggregate["swapped_answer_to_reference_sum"],
            swapped_answer_n,
        ),
        "original_answer_routing_balance": available_mean(
            aggregate["original_answer_routing_balance_sum"],
            original_answer_n,
        ),
        "swapped_answer_routing_balance": available_mean(
            aggregate["swapped_answer_routing_balance_sum"],
            swapped_answer_n,
        ),

        "headmean_original_accuracy": (
            aggregate["headmean_original_correct"] / n
        ).astype(np.float32),
        "headmean_swapped_accuracy": (
            aggregate["headmean_swapped_correct"] / n
        ).astype(np.float32),
        "headmean_average_accuracy": (
            aggregate["headmean_average_correct"] / n
        ).astype(np.float32),
    }

    correct_answer_n = int(aggregate["correct_answer_query_count"])
    wrong_answer_n = int(aggregate["wrong_answer_query_count"])
    metrics.update({
        "answer_to_subject_correct": available_mean(
            aggregate["answer_to_subject_correct_sum"], correct_answer_n
        ),
        "answer_to_reference_correct": available_mean(
            aggregate["answer_to_reference_correct_sum"], correct_answer_n
        ),
        "answer_routing_balance_correct": available_mean(
            aggregate["answer_routing_balance_correct_sum"], correct_answer_n
        ),
        "answer_visual_mass_correct": available_mean(
            aggregate["answer_visual_mass_correct_sum"], correct_answer_n
        ),
        "answer_to_subject_wrong": available_mean(
            aggregate["answer_to_subject_wrong_sum"], wrong_answer_n
        ),
        "answer_to_reference_wrong": available_mean(
            aggregate["answer_to_reference_wrong_sum"], wrong_answer_n
        ),
        "answer_routing_balance_wrong": available_mean(
            aggregate["answer_routing_balance_wrong_sum"], wrong_answer_n
        ),
        "answer_visual_mass_wrong": available_mean(
            aggregate["answer_visual_mass_wrong_sum"], wrong_answer_n
        ),
    })

    attention_accuracy = metrics["attention_average_accuracy"]
    map_stability = metrics["same_object_map_cosine"].mean(axis=-1)
    separation = 0.5 * (
        metrics["original_object_separation"]
        + metrics["swapped_object_separation"]
    )
    object_visual_mass = 0.25 * (
        metrics["original_prompt_visual_mass"][:, :, 0]
        + metrics["original_prompt_visual_mass"][:, :, 1]
        + metrics["swapped_prompt_visual_mass"][:, :, 0]
        + metrics["swapped_prompt_visual_mass"][:, :, 1]
    )
    unsupervised_head_score = (
        map_stability * separation * object_visual_mass
    )
    metrics["unsupervised_head_score"] = (
        unsupervised_head_score.astype(np.float32)
    )

    def top_heads(
        values: np.ndarray,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        flat_order = np.argsort(values.reshape(-1))[::-1][:limit]
        rows = []
        for flat_index in flat_order:
            layer_pos, head = np.unravel_index(
                int(flat_index),
                values.shape,
            )
            rows.append({
                "layer": int(selected_layers[layer_pos]),
                "head": int(head),
                "score": float(values[layer_pos, head]),
                "attention_original_accuracy": float(
                    metrics["attention_original_accuracy"][layer_pos, head]
                ),
                "attention_swapped_accuracy": float(
                    metrics["attention_swapped_accuracy"][layer_pos, head]
                ),
                "attention_average_accuracy": float(
                    metrics["attention_average_accuracy"][layer_pos, head]
                ),
                "map_cosine": float(
                    map_stability[layer_pos, head]
                ),
                "object_separation": float(
                    separation[layer_pos, head]
                ),
                "object_visual_mass": float(
                    object_visual_mass[layer_pos, head]
                ),
            })
        return rows

    report_layer = int(selected_layers[report_layer_position])
    report_attention = metrics["attention_average_accuracy"][
        report_layer_position
    ]
    best_report_head = int(np.argmax(report_attention))

    summary: Dict[str, Any] = {
        "n_samples": int(aggregate["n"]),
        "answer_query_availability": {
            "original": int(aggregate["original_answer_query_count"]),
            "swapped": int(aggregate["swapped_answer_query_count"]),
        },
        "generation_original_accuracy": (
            aggregate["generation_original_correct"]
            / max(1, aggregate["generation_original_valid"])
        ),
        "generation_original_accuracy_all_samples": (
            aggregate["generation_original_correct"]
            / max(1, n)
        ),
        "generation_original_valid_rate": (
            aggregate["generation_original_valid"] / max(1, n)
        ),
        "generation_swapped_aligned_accuracy": (
            aggregate["generation_swapped_aligned_correct"]
            / max(1, aggregate["generation_swapped_valid"])
        ),
        "answer_swap_consistency": (
            aggregate["generation_answer_swap_consistent"] / n
        ),
        "mean_answer_distribution_cosine": (
            aggregate["answer_distribution_cosine_sum"] / n
        ),
        "mean_answer_distribution_js": (
            aggregate["answer_distribution_js_sum"] / n
        ),
        "correct_group_count": int(aggregate["correct_group_count"]),
        "wrong_group_count": int(aggregate["wrong_group_count"]),
        "report_layer": report_layer,
        "report_layer_metrics": {
            "similarity_original_accuracy": float(
                metrics["similarity_original_accuracy"][
                    report_layer_position
                ]
            ),
            "similarity_swapped_accuracy": float(
                metrics["similarity_swapped_accuracy"][
                    report_layer_position
                ]
            ),
            "similarity_average_accuracy": float(
                metrics["similarity_average_accuracy"][
                    report_layer_position
                ]
            ),
            "attention_headmean_original_accuracy": float(
                metrics["headmean_original_accuracy"][
                    report_layer_position
                ]
            ),
            "attention_headmean_swapped_accuracy": float(
                metrics["headmean_swapped_accuracy"][
                    report_layer_position
                ]
            ),
            "attention_headmean_average_accuracy": float(
                metrics["headmean_average_accuracy"][
                    report_layer_position
                ]
            ),
            "attention_best_head": best_report_head,
            "attention_best_head_average_accuracy": float(
                report_attention[best_report_head]
            ),
            "attention_mean_head_average_accuracy": float(
                report_attention.mean()
            ),
            "answer_to_subject_correct_mean_over_heads": float(
                metrics["answer_to_subject_correct"][
                    report_layer_position
                ].mean()
            ),
            "answer_to_subject_wrong_mean_over_heads": float(
                metrics["answer_to_subject_wrong"][
                    report_layer_position
                ].mean()
            ),
            "answer_to_reference_correct_mean_over_heads": float(
                metrics["answer_to_reference_correct"][
                    report_layer_position
                ].mean()
            ),
            "answer_to_reference_wrong_mean_over_heads": float(
                metrics["answer_to_reference_wrong"][
                    report_layer_position
                ].mean()
            ),
            "answer_routing_balance_correct_mean_over_heads": float(
                metrics["answer_routing_balance_correct"][
                    report_layer_position
                ].mean()
            ),
            "answer_routing_balance_wrong_mean_over_heads": float(
                metrics["answer_routing_balance_wrong"][
                    report_layer_position
                ].mean()
            ),
            "answer_visual_mass_correct_mean_over_heads": float(
                metrics["answer_visual_mass_correct"][
                    report_layer_position
                ].mean()
            ),
            "answer_visual_mass_wrong_mean_over_heads": float(
                metrics["answer_visual_mass_wrong"][
                    report_layer_position
                ].mean()
            ),
        },
        "best_similarity_layers": {
            "original": {
                "layer": int(
                    selected_layers[
                        int(np.argmax(metrics["similarity_original_accuracy"]))
                    ]
                ),
                "accuracy": float(
                    metrics["similarity_original_accuracy"].max()
                ),
            },
            "swapped": {
                "layer": int(
                    selected_layers[
                        int(np.argmax(metrics["similarity_swapped_accuracy"]))
                    ]
                ),
                "accuracy": float(
                    metrics["similarity_swapped_accuracy"].max()
                ),
            },
            "average": {
                "layer": int(
                    selected_layers[
                        int(np.argmax(metrics["similarity_average_accuracy"]))
                    ]
                ),
                "accuracy": float(
                    metrics["similarity_average_accuracy"].max()
                ),
            },
        },
        "top_attention_heads_by_accuracy": top_heads(
            attention_accuracy,
            limit=20,
        ),
        "top_attention_heads_unsupervised": top_heads(
            unsupervised_head_score,
            limit=20,
        ),
        "per_relation": {},
    }

    for relation, stats in aggregate["per_relation"].items():
        relation_n = max(1, int(stats["n"]))
        summary["per_relation"][relation] = {
            "n": int(stats["n"]),
            "generation_original_accuracy": (
                stats["generation_original_correct"] / relation_n
            ),
            "generation_swapped_aligned_accuracy": (
                stats["generation_swapped_aligned_correct"] / relation_n
            ),
            "report_layer_headmean_original_accuracy": float(
                stats["headmean_original_correct"][
                    report_layer_position
                ] / relation_n
            ),
            "report_layer_headmean_swapped_accuracy": float(
                stats["headmean_swapped_correct"][
                    report_layer_position
                ] / relation_n
            ),
            "report_layer_headmean_average_accuracy": float(
                stats["headmean_average_correct"][
                    report_layer_position
                ] / relation_n
            ),
            "report_layer_similarity_original_accuracy": float(
                stats["similarity_original_correct"][
                    report_layer_position
                ] / relation_n
            ),
            "report_layer_similarity_swapped_accuracy": float(
                stats["similarity_swapped_correct"][
                    report_layer_position
                ] / relation_n
            ),
            "report_layer_similarity_average_accuracy": float(
                stats["similarity_average_correct"][
                    report_layer_position
                ] / relation_n
            ),
        }

    return summary, metrics


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("VG2: CENTROID AND NATURAL GENERATION COMPARISON")
    print("=" * 100)
    print(
        f"original generation (all): "
        f"{summary['generation_original_accuracy_all_samples']:.4f}"
    )
    print(
        f"original generation valid: "
        f"{summary['generation_original_accuracy']:.4f} "
        f"(valid rate={summary['generation_original_valid_rate']:.4f})"
    )
    print(
        f"swapped generation aligned:"
        f"{summary['generation_swapped_aligned_accuracy']:.4f}"
    )
    print(
        f"answer swap consistency:   "
        f"{summary['answer_swap_consistency']:.4f}"
    )
    print(
        f"answer distribution cosine:"
        f"{summary['mean_answer_distribution_cosine']:.4f}"
    )
    print(
        f"answer distribution JS:    "
        f"{summary['mean_answer_distribution_js']:.4f}"
    )

    layer = summary["report_layer"]
    report = summary["report_layer_metrics"]
    print(f"\nReport layer L{layer}:")
    print(
        f"similarity centroid original/swap/avg: "
        f"{report['similarity_original_accuracy']:.4f} / "
        f"{report['similarity_swapped_accuracy']:.4f} / "
        f"{report['similarity_average_accuracy']:.4f}"
    )
    print(
        f"attention head-mean original/swap/avg: "
        f"{report['attention_headmean_original_accuracy']:.4f} / "
        f"{report['attention_headmean_swapped_accuracy']:.4f} / "
        f"{report['attention_headmean_average_accuracy']:.4f}"
    )
    print(
        f"best attention head: H{report['attention_best_head']} "
        f"acc={report['attention_best_head_average_accuracy']:.4f}; "
        f"mean-head acc={report['attention_mean_head_average_accuracy']:.4f}"
    )
    print(
        "first-answer routing, correct vs wrong:\n"
        f"  to subject : "
        f"{report['answer_to_subject_correct_mean_over_heads']:.6f} vs "
        f"{report['answer_to_subject_wrong_mean_over_heads']:.6f}\n"
        f"  to reference: "
        f"{report['answer_to_reference_correct_mean_over_heads']:.6f} vs "
        f"{report['answer_to_reference_wrong_mean_over_heads']:.6f}\n"
        f"  balance    : "
        f"{report['answer_routing_balance_correct_mean_over_heads']:.4f} vs "
        f"{report['answer_routing_balance_wrong_mean_over_heads']:.4f}\n"
        f"  visual mass: "
        f"{report['answer_visual_mass_correct_mean_over_heads']:.4f} vs "
        f"{report['answer_visual_mass_wrong_mean_over_heads']:.4f}"
    )

    print("\nPer relation at report layer:")
    for relation, stats in summary["per_relation"].items():
        print(
            f"  {relation:6s} n={stats['n']:4d} | "
            f"gen={stats['generation_original_accuracy']:.4f} | "
            f"swap-gen={stats['generation_swapped_aligned_accuracy']:.4f} | "
            f"sim={stats['report_layer_similarity_original_accuracy']:.4f} | "
            f"swap-sim={stats['report_layer_similarity_swapped_accuracy']:.4f} | "
            f"attn={stats['report_layer_headmean_original_accuracy']:.4f} | "
            f"swap-attn={stats['report_layer_headmean_swapped_accuracy']:.4f}"
        )

    print("\nTop 5 attention heads by centroid accuracy:")
    for row in summary["top_attention_heads_by_accuracy"][:5]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} | "
            f"acc={row['score']:.4f} | "
            f"map_cos={row['map_cosine']:.4f} | "
            f"sep={row['object_separation']:.4f} | "
            f"vis_mass={row['object_visual_mass']:.4f}"
        )

    print("\nTop 5 attention heads by label-free stability score:")
    for row in summary["top_attention_heads_unsupervised"][:5]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} | "
            f"score={row['score']:.6f} | "
            f"acc={row['attention_average_accuracy']:.4f} | "
            f"map_cos={row['map_cosine']:.4f} | "
            f"sep={row['object_separation']:.4f}"
        )


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.attn_impl != "eager":
        raise ValueError(
            "Step 1 requires --attn-impl eager so complete attention "
            "probabilities are returned."
        )
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    module = import_two_object_module()

    prompt_path = resolve_prompt_path(args)
    prompt_rows = load_standard_prompts(prompt_path)
    records, audit = load_vg2_records(
        Path(args.vg_data_json),
        Path(args.vg_image_root),
        args.max_samples,
    )
    if not records:
        raise RuntimeError(
            "No usable VG2 records. Inspect the data JSON, image root, and audit."
        )
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

    specs = merged_model_specs(module)
    if args.model not in specs:
        raise ValueError(
            f"Model {args.model!r} not found in the merged model specs. "
            f"Available={sorted(specs)}"
        )
    spec = specs[args.model]

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "sample_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    aggregate_path = output_dir / "aggregate_metrics.npz"

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

    # We use deterministic greedy decoding. Some checkpoints ship sampling
    # defaults (temperature/top_p/top_k) in generation_config, which causes a
    # warning even though do_sample=False. Clearing them does not change greedy
    # generation; it only removes the irrelevant warning.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    selected_layers = parse_layers(args.layers, len(decoder_layers))
    report_layer = resolve_report_layer(
        args.report_layer,
        args.model,
        selected_layers,
    )
    report_layer_position = selected_layers.index(report_layer)
    relation_token_map = relation_token_variants(
        processor.tokenizer
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "vg_data_json": str(args.vg_data_json),
        "vg_image_root": str(args.vg_image_root),
        "prompt_jsonl": str(prompt_path),
        "model": args.model,
        "repo_id": spec.repo_id,
        "transformers_version": transformers.__version__,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "selected_layers": selected_layers,
        "report_layer": report_layer,
        "attn_implementation": "eager",
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "save_object_maps": args.save_object_maps,
        "array_dtype": args.array_dtype,
        "n_records": len(records),
        "audit": audit,
        "uses_gt_for_generation": False,
        "modifies_model": False,
        "generation_is_trace_only": False,
        "generation_forces_minimum_tokens": False,
        "generation_note": (
            "Generation uses the requested max_new_tokens as an upper bound "
            "and may stop naturally at EOS. First-answer-token routing is "
            "reported only when a second decoding step naturally exists."
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Decoder: {decoder_path}, n={len(decoder_layers)}; "
        f"layers={selected_layers}; report=L{report_layer}"
    )
    print(
        "Relation-token variants:\n"
        + json.dumps(
            {
                relation: [
                    {
                        "id": token_id,
                        "decoded": processor.tokenizer.decode([token_id]),
                    }
                    for token_id in ids
                ]
                for relation, ids in relation_token_map.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    existing_rows = [
        row for row in read_jsonl(samples_path)
        if "sid" in row and "error" not in row
    ]
    done = {
        int(row["sid"])
        for row in existing_rows
        if (arrays_dir / f"{int(row['sid'])}.npz").exists()
    }

    # Rebuild aggregate from this run. Resume is intentionally conservative:
    # already completed samples are skipped, so use --overwrite for a fresh,
    # self-contained aggregate.
    if done and not args.overwrite:
        raise RuntimeError(
            "Existing sample files were found. Use --overwrite for Step 1 so "
            "aggregate_metrics.npz is computed from one complete run."
        )

    aggregate: Optional[Dict[str, Any]] = None
    completed = 0
    started = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"attention-step1:{args.dataset}:{args.model}",
        ):
            sid = int(record.sid)
            image = None
            original_batch = None
            swapped_batch = None

            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question_text = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATION_TO_INDEX:
                    raise ValueError(
                        f"Unsupported GT relation for sid={sid}: {gt!r}"
                    )

                image = record_image(record)
                original_batch = make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question_text,
                    device=device,
                )
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

                original = analyze_prompt(
                    model=model,
                    processor=processor,
                    batch=original_batch,
                    question_text=question_text,
                    subject=subject,
                    reference=reference,
                    selected_layers=selected_layers,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    relation_token_map=relation_token_map,
                    gt=gt,
                )
                swapped = analyze_prompt(
                    model=model,
                    processor=processor,
                    batch=swapped_batch,
                    question_text=swapped_question,
                    subject=reference,
                    reference=subject,
                    selected_layers=selected_layers,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    relation_token_map=relation_token_map,
                    gt=invert_relation(gt),
                )

                if original["n_heads"] != swapped["n_heads"]:
                    raise RuntimeError(
                        f"Original/swap head count mismatch: "
                        f"{original['n_heads']} vs {swapped['n_heads']}"
                    )
                if (
                    original["object_maps"].shape[-1]
                    != swapped["object_maps"].shape[-1]
                ):
                    raise RuntimeError(
                        "Original/swap visual-token count mismatch"
                    )

                # Align swapped role order [B, A] back to semantic order [A, B].
                swapped_maps_aligned = swapped["object_maps"][:, :, [1, 0], :]
                swapped_centroids_aligned = swapped[
                    "object_centroids"
                ][:, :, [1, 0], :]
                attention_swapped_pred, _ = relation_codes_from_centroids(
                    swapped_centroids_aligned
                )
                average_centroids = 0.5 * (
                    original["object_centroids"]
                    + swapped_centroids_aligned
                )
                attention_average_pred, _ = relation_codes_from_centroids(
                    average_centroids
                )

                same_object_map_cosine = map_cosine(
                    original["object_maps"],
                    swapped_maps_aligned,
                )
                same_object_map_js = map_js_divergence(
                    original["object_maps"],
                    swapped_maps_aligned,
                )
                same_object_centroid_distance = np.linalg.norm(
                    original["object_centroids"]
                    - swapped_centroids_aligned,
                    axis=-1,
                ).astype(np.float32)
                attention_relation_consistency = (
                    original["object_prediction"]
                    == attention_swapped_pred
                ).astype(np.int8)

                swapped_similarity_centroids_aligned = swapped[
                    "similarity_centroids"
                ][:, [1, 0], :]
                similarity_swapped_pred, _ = relation_codes_from_centroids(
                    swapped_similarity_centroids_aligned
                )
                average_similarity_centroids = 0.5 * (
                    original["similarity_centroids"]
                    + swapped_similarity_centroids_aligned
                )
                similarity_average_pred, _ = relation_codes_from_centroids(
                    average_similarity_centroids
                )
                similarity_relation_consistency = (
                    original["similarity_prediction"]
                    == similarity_swapped_pred
                ).astype(np.int8)

                cross = {
                    "same_object_map_cosine": same_object_map_cosine,
                    "same_object_map_js": same_object_map_js,
                    "same_object_centroid_distance": (
                        same_object_centroid_distance
                    ),
                    "attention_swapped_aligned_prediction": (
                        attention_swapped_pred
                    ),
                    "attention_average_prediction": (
                        attention_average_pred
                    ),
                    "attention_relation_consistency": (
                        attention_relation_consistency
                    ),
                    "similarity_swapped_aligned_prediction": (
                        similarity_swapped_pred
                    ),
                    "similarity_average_prediction": (
                        similarity_average_pred
                    ),
                    "similarity_relation_consistency": (
                        similarity_relation_consistency
                    ),
                }

                headmean_original, _ = head_mean_relation(
                    original["object_maps"],
                    original["visual_coordinates"],
                )
                headmean_swapped, _ = head_mean_relation(
                    swapped_maps_aligned,
                    original["visual_coordinates"],
                )
                headmean_average, _ = head_mean_relation(
                    0.5 * (
                        original["object_maps"]
                        + swapped_maps_aligned
                    ),
                    original["visual_coordinates"],
                )
                cross.update({
                    "headmean_original_prediction": headmean_original,
                    "headmean_swapped_aligned_prediction": headmean_swapped,
                    "headmean_average_prediction": headmean_average,
                })

                if aggregate is None:
                    aggregate = init_aggregate(
                        len(selected_layers),
                        original["n_heads"],
                        len(QUERY_NAMES),
                    )

                original_correct = bool(
                    original["prediction"] == gt
                )
                update_aggregate(
                    aggregate,
                    gt=gt,
                    original=original,
                    swapped=swapped,
                    cross=cross,
                    headmean_original=headmean_original,
                    headmean_swapped=headmean_swapped,
                    headmean_average=headmean_average,
                    original_correct=original_correct,
                )

                array_path = arrays_dir / f"{sid}.npz"
                save_sample_arrays(
                    array_path,
                    sid=sid,
                    selected_layers=selected_layers,
                    original=original,
                    swapped=swapped,
                    cross=cross,
                    save_object_maps=args.save_object_maps,
                    array_dtype=args.array_dtype,
                )

                report = report_layer_position
                swapped_aligned_generation = invert_relation(
                    swapped["prediction"]
                )
                answer_swap_consistent = bool(
                    original["prediction"] is not None
                    and swapped_aligned_generation is not None
                    and original["prediction"]
                    == swapped_aligned_generation
                )

                original_probs = original["relation_probs"]
                swapped_probs_aligned = aligned_swap_vector(
                    swapped["relation_probs"]
                )
                distribution_cosine = float(
                    np.dot(original_probs, swapped_probs_aligned)
                    / (
                        np.linalg.norm(original_probs)
                        * np.linalg.norm(swapped_probs_aligned)
                        + 1e-12
                    )
                )
                distribution_js = float(
                    map_js_divergence(
                        original_probs[None, :],
                        swapped_probs_aligned[None, :],
                    )[0]
                )

                row = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "gt": gt,
                    "question": question_text,
                    "swapped_question": swapped_question,

                    "original_generated_text": original[
                        "generated_text"
                    ],
                    "original_prediction": original["prediction"],
                    "original_correct": original_correct,
                    "original_answer_query_available": bool(
                        original["answer_query_available"]
                    ),
                    "swapped_answer_query_available": bool(
                        swapped["answer_query_available"]
                    ),
                    "swapped_generated_text": swapped[
                        "generated_text"
                    ],
                    "swapped_prediction_raw": swapped["prediction"],
                    "swapped_prediction_aligned": (
                        swapped_aligned_generation
                    ),
                    "swapped_aligned_correct": bool(
                        swapped_aligned_generation == gt
                    ),
                    "answer_swap_consistent": (
                        answer_swap_consistent
                    ),

                    "original_relation_logits": {
                        relation: float(
                            original["relation_logits"][index]
                        )
                        for index, relation in enumerate(RELATIONS)
                    },
                    "original_relation_probs": {
                        relation: float(
                            original["relation_probs"][index]
                        )
                        for index, relation in enumerate(RELATIONS)
                    },
                    "swapped_relation_probs_aligned": {
                        relation: float(
                            swapped_probs_aligned[index]
                        )
                        for index, relation in enumerate(RELATIONS)
                    },
                    "original_relation_top1_margin": safe_float(
                        original["relation_top1_margin"]
                    ),
                    "original_relation_gt_margin": safe_float(
                        original["relation_gt_margin"]
                    ),
                    "swapped_relation_top1_margin": safe_float(
                        swapped["relation_top1_margin"]
                    ),
                    "swapped_relation_gt_margin": safe_float(
                        swapped["relation_gt_margin"]
                    ),
                    "answer_distribution_cosine": (
                        distribution_cosine
                    ),
                    "answer_distribution_js": distribution_js,

                    "report_layer": report_layer,
                    "report_similarity_original": to_relation_name(
                        original["similarity_prediction"][report]
                    ),
                    "report_similarity_swapped_aligned": (
                        to_relation_name(
                            similarity_swapped_pred[report]
                        )
                    ),
                    "report_similarity_average": (
                        to_relation_name(
                            similarity_average_pred[report]
                        )
                    ),
                    "report_attention_headmean_original": (
                        to_relation_name(
                            headmean_original[report]
                        )
                    ),
                    "report_attention_headmean_swapped_aligned": (
                        to_relation_name(
                            headmean_swapped[report]
                        )
                    ),
                    "report_attention_headmean_average": (
                        to_relation_name(
                            headmean_average[report]
                        )
                    ),
                    "report_same_object_map_cosine_mean": float(
                        same_object_map_cosine[report].mean()
                    ),
                    "report_same_object_centroid_distance_mean": float(
                        same_object_centroid_distance[report].mean()
                    ),
                    "report_object_separation_original_mean": float(
                        original["object_separation"][report].mean()
                    ),
                    "report_object_separation_swapped_mean": float(
                        swapped["object_separation"][report].mean()
                    ),
                    "report_first_answer_to_subject_mean": float(
                        original["answer_to_subject"][report].mean()
                    ),
                    "report_first_answer_to_reference_mean": float(
                        original["answer_to_reference"][report].mean()
                    ),
                    "report_first_answer_routing_balance_mean": float(
                        original["answer_routing_balance"][report].mean()
                    ),
                    "report_first_answer_visual_mass_mean": float(
                        original["answer_visual_mass"][report].mean()
                    ),
                    "report_question_last_to_subject_mean": float(
                        original["prompt_to_subject"][
                            report,
                            :,
                            QUERY_NAMES.index("question_last"),
                        ].mean()
                    ),
                    "report_question_last_to_reference_mean": float(
                        original["prompt_to_reference"][
                            report,
                            :,
                            QUERY_NAMES.index("question_last"),
                        ].mean()
                    ),
                    "array_file": str(array_path),
                }
                append_jsonl(samples_path, row)
                completed += 1

                # Print every normal-generation result. Invalid/unparsed
                # generations count as wrong in the running all-sample ACC.
                running_correct = int(
                    aggregate["generation_original_correct"]
                )
                running_total = int(completed)
                running_accuracy = (
                    running_correct / running_total
                    if running_total else 0.0
                )
                prediction_text = (
                    original["prediction"]
                    if original["prediction"] is not None
                    else "<invalid>"
                )
                raw_generation = one_line(
                    original["generated_text"]
                )
                tqdm.write(
                    f"\n[GEN {completed}/{len(records)}] "
                    f"model={args.model} | sid={sid}\n"
                    f"  Question: {one_line(question_text)}\n"
                    f"  GT: {gt}\n"
                    f"  Pred: {prediction_text}\n"
                    f"  Current ACC: "
                    f"{running_correct}/{running_total} "
                    f"= {running_accuracy:.4f}\n"
                    f"  Generation: {raw_generation!r}"
                )

                del original, swapped, cross
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
                            traceback.format_exc().splitlines()[-16:]
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
                if original_batch is not None:
                    del original_batch
                if swapped_batch is not None:
                    del swapped_batch
                if image is not None:
                    del image

        if aggregate is None:
            raise RuntimeError(
                "No samples completed; inspect errors.jsonl"
            )

        summary, aggregate_metrics = finalize_aggregate(
            aggregate,
            selected_layers,
            report_layer_position,
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

        with aggregate_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **aggregate_metrics,
            )

        print_summary(summary)
        print(f"\nSaved samples:   {samples_path}")
        print(f"Saved arrays:    {arrays_dir}")
        print(f"Saved aggregate: {aggregate_path}")
        print(f"Saved summary:   {summary_path}")

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
