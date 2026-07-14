#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover and suppress conflict/noise attention heads.

This script consumes the outputs of Step 1:

    analyze_coco_attention_flow_swap_step1_v1.py

and performs two stages.

Stage A: label-free conflict-head discovery
-------------------------------------------
A small set of high-confidence spatial heads is selected from Step 1's
unsupervised score. For every saved sample, their original/swap-aligned object
centroids are combined into a spatial consensus. Every other head is scored by:

- visual attention mass;
- subject/reference map separation;
- disagreement with the positive-head centroid consensus;
- original/swap instability.

A high conflict score therefore means that a head strongly reads visual tokens,
separates the two objects, but points away from the reliable spatial consensus
or changes substantially under subject/reference swapping.

Ground-truth labels are not used to rank conflict heads. Accuracy stored by
Step 1 is included only as a diagnostic column.

Stage B: causal suppression
---------------------------
The original prompt and image are kept unchanged. For selected conflict heads,
only the visual-token value contribution written into the subject/reference
object-token positions is scaled:

    o_visual <- scale * o_visual

where scale=0 removes the contribution and 0<scale<1 softly suppresses it.

The script compares:
- conflict: highest discovered conflict scores;
- random_matched: random heads in the same layers;
- low_conflict_matched: lowest-conflict heads in the same layers.

Optionally, it can also amplify trusted positive spatial heads with scale>1.

No model parameters are trained. Ground truth is used only after generation for
accuracy, fixed/broken counts, and per-relation reporting.

Required Step 1 files:
- aggregate_metrics.npz
- sample_arrays/*.npz
- samples.jsonl
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


SCRIPT_VERSION = "discover-suppress-conflict-heads-v1"

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








GROUPS = (
    "conflict",
    "random_matched",
    "low_conflict_matched",
    "positive_amplify",
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
    p.add_argument("--attn-impl", default="eager", choices=["eager"])
    p.add_argument(
        "--step1-dir",
        required=True,
        help=(
            "Step 1 output directory containing aggregate_metrics.npz, "
            "samples.jsonl and sample_arrays/."
        ),
    )
    p.add_argument(
        "--only",
        default="both",
        choices=["discover", "intervene", "both"],
        help="Run discovery only, intervention only, or both.",
    )
    p.add_argument(
        "--ranking-json",
        default=None,
        help=(
            "Existing ranking JSON for --only intervene. By default reads "
            "<output-dir>/conflict_head_ranking.json."
        ),
    )

    p.add_argument(
        "--positive-consensus-k",
        type=int,
        default=5,
        help="Number of trusted unsupervised spatial heads used for consensus.",
    )
    p.add_argument(
        "--discovery-max-samples",
        type=int,
        default=None,
        help="Optional cap on Step 1 sample arrays used for discovery.",
    )
    p.add_argument(
        "--min-consensus-confidence",
        type=float,
        default=0.05,
        help="Ignore samples whose positive-head consensus is weaker than this.",
    )
    p.add_argument(
        "--consensus-conflict-weight",
        type=float,
        default=0.75,
        help="Weight on disagreement with the positive spatial consensus.",
    )
    p.add_argument(
        "--swap-noise-weight",
        type=float,
        default=0.25,
        help="Weight on original/swap map instability.",
    )
    p.add_argument(
        "--exclude-positive-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude positive consensus heads from conflict candidates.",
    )

    p.add_argument(
        "--condition-groups",
        default="conflict,random_matched,low_conflict_matched",
        help="Comma-separated subset of: " + ",".join(GROUPS),
    )
    p.add_argument(
        "--top-k",
        default="1,2,5",
        help="Comma-separated numbers of heads per intervention.",
    )
    p.add_argument(
        "--suppression-scale",
        type=float,
        default=0.0,
        help="Visual contribution scale for conflict/control heads; 0 removes it.",
    )
    p.add_argument(
        "--positive-amplify-scale",
        type=float,
        default=1.25,
        help="Visual contribution scale used by positive_amplify.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=17,
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
    allowed_set = set(allowed)
    result: List[str] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported group {item!r}; allowed={sorted(allowed_set)}"
            )
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("No condition group selected")
    return result


def parse_positive_ints(value: str) -> List[int]:
    result: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise ValueError("--top-k values must be positive")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError("--top-k resolved to an empty list")
    return sorted(result)


def relation_code_from_centroids(centroids: np.ndarray) -> np.ndarray:
    """centroids [..., 2 objects, 2 coordinates] -> relation code 0..3."""
    dx = centroids[..., 0, 0] - centroids[..., 1, 0]
    dy = centroids[..., 0, 1] - centroids[..., 1, 1]
    horizontal = np.abs(dx) >= np.abs(dy)
    return np.where(
        horizontal,
        np.where(dx < 0.0, 0, 1),
        np.where(dy < 0.0, 2, 3),
    ).astype(np.int8)


def load_step1_aggregate(step1_dir: Path) -> Dict[str, np.ndarray]:
    path = step1_dir / "aggregate_metrics.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 1 aggregate: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = [
            "layer_indices",
            "unsupervised_head_score",
            "attention_average_accuracy",
            "same_object_map_cosine",
            "original_object_separation",
            "swapped_object_separation",
            "original_prompt_visual_mass",
            "swapped_prompt_visual_mass",
        ]
        missing = [name for name in required if name not in z.files]
        if missing:
            raise RuntimeError(
                f"Step 1 aggregate is missing {missing}; available={z.files}"
            )
        return {
            name: z[name].copy()
            for name in required
        }


def sorted_global_heads(
    layers: np.ndarray,
    score: np.ndarray,
    descending: bool = True,
) -> List[Tuple[int, int, float]]:
    rows: List[Tuple[int, int, float]] = []
    for layer_pos, layer in enumerate(layers.tolist()):
        for head in range(score.shape[1]):
            value = float(score[layer_pos, head])
            if np.isfinite(value):
                rows.append((int(layer), int(head), value))
    rows.sort(key=lambda row: row[2], reverse=descending)
    return rows


def discover_conflict_heads(
    *,
    step1_dir: Path,
    positive_k: int,
    max_samples: Optional[int],
    min_consensus_confidence: float,
    consensus_conflict_weight: float,
    swap_noise_weight: float,
    exclude_positive_heads: bool,
) -> Dict[str, Any]:
    aggregate = load_step1_aggregate(step1_dir)
    layers = aggregate["layer_indices"].astype(np.int64)
    unsupervised = aggregate["unsupervised_head_score"].astype(np.float64)
    oracle_accuracy = aggregate["attention_average_accuracy"].astype(np.float64)

    if unsupervised.ndim != 2:
        raise RuntimeError(
            f"Expected unsupervised_head_score [layer, head], got "
            f"{unsupervised.shape}"
        )
    n_layers, n_heads = unsupervised.shape
    if positive_k <= 0 or positive_k > n_layers * n_heads:
        raise ValueError(
            f"--positive-consensus-k must be in [1, {n_layers * n_heads}]"
        )

    positive_heads = sorted_global_heads(
        layers,
        unsupervised,
        descending=True,
    )[:positive_k]
    positive_keys = {
        (int(layer), int(head))
        for layer, head, _ in positive_heads
    }
    layer_to_pos = {
        int(layer): pos for pos, layer in enumerate(layers.tolist())
    }
    positive_positions = [
        (layer_to_pos[layer], head, max(float(score), 1e-8))
        for layer, head, score in positive_heads
    ]
    positive_global_weights = np.asarray(
        [score for _, _, score in positive_positions],
        dtype=np.float64,
    )
    positive_global_weights /= positive_global_weights.sum()

    sample_dir = step1_dir / "sample_arrays"
    paths = sorted(
        sample_dir.glob("*.npz"),
        key=lambda path: int(path.stem),
    )
    if max_samples is not None:
        paths = paths[: int(max_samples)]
    if not paths:
        raise FileNotFoundError(
            f"No Step 1 sample arrays found in {sample_dir}"
        )

    total_weight = np.zeros((n_layers, n_heads), dtype=np.float64)
    conflict_sum = np.zeros_like(total_weight)
    consensus_conflict_sum = np.zeros_like(total_weight)
    swap_noise_sum = np.zeros_like(total_weight)
    signal_sum = np.zeros_like(total_weight)
    relation_disagreement_sum = np.zeros_like(total_weight)
    centroid_distance_sum = np.zeros_like(total_weight)
    stability_sum = np.zeros_like(total_weight)

    used_samples = 0
    skipped_samples = 0
    consensus_relation_counts = np.zeros(4, dtype=np.int64)

    for path in tqdm(paths, desc="discover-conflict-heads"):
        with np.load(path, allow_pickle=False) as z:
            required = [
                "layer_indices",
                "original_object_centroids",
                "swapped_object_centroids_role_order",
                "original_object_separation",
                "swapped_object_separation",
                "same_object_map_cosine",
                "original_prompt_visual_mass",
                "swapped_prompt_visual_mass",
            ]
            missing = [name for name in required if name not in z.files]
            if missing:
                raise RuntimeError(
                    f"{path} is missing {missing}. Re-run Step 1 with the "
                    "current script version."
                )

            sample_layers = z["layer_indices"].astype(np.int64)
            if not np.array_equal(sample_layers, layers):
                raise RuntimeError(
                    f"Layer mismatch in {path}: "
                    f"{sample_layers.tolist()} vs {layers.tolist()}"
                )

            original_centroids = z[
                "original_object_centroids"
            ].astype(np.float64)
            swapped_centroids = z[
                "swapped_object_centroids_role_order"
            ].astype(np.float64)
            swapped_aligned = swapped_centroids[:, :, [1, 0], :]
            average_centroids = 0.5 * (
                original_centroids + swapped_aligned
            )

            original_separation = z[
                "original_object_separation"
            ].astype(np.float64)
            swapped_separation = z[
                "swapped_object_separation"
            ].astype(np.float64)
            separation = np.sqrt(
                np.clip(
                    original_separation * swapped_separation,
                    0.0,
                    None,
                )
            )

            map_cosine = z[
                "same_object_map_cosine"
            ].astype(np.float64)
            stability = np.clip(
                map_cosine.mean(axis=-1),
                0.0,
                1.0,
            )

            original_visual_mass = z[
                "original_prompt_visual_mass"
            ].astype(np.float64)
            swapped_visual_mass = z[
                "swapped_prompt_visual_mass"
            ].astype(np.float64)
            # Query dimension 0/1 = subject/reference object tokens.
            visual_mass = 0.25 * (
                original_visual_mass[:, :, 0]
                + original_visual_mass[:, :, 1]
                + swapped_visual_mass[:, :, 0]
                + swapped_visual_mass[:, :, 1]
            )
            signal = np.sqrt(
                np.clip(visual_mass * separation, 0.0, None)
            )

            positive_centroids = np.stack(
                [
                    average_centroids[layer_pos, head]
                    for layer_pos, head, _ in positive_positions
                ],
                axis=0,
            )
            consensus_centroids = np.sum(
                positive_global_weights[:, None, None]
                * positive_centroids,
                axis=0,
            )
            positive_relations = relation_code_from_centroids(
                positive_centroids
            )
            consensus_relation = int(
                relation_code_from_centroids(
                    consensus_centroids[None, ...]
                )[0]
            )
            consensus_relation_counts[consensus_relation] += 1

            relation_agreement = float(
                np.mean(positive_relations == consensus_relation)
            )
            positive_stability = float(np.sum(
                positive_global_weights
                * np.asarray([
                    stability[layer_pos, head]
                    for layer_pos, head, _ in positive_positions
                ])
            ))
            positive_separation = float(np.sum(
                positive_global_weights
                * np.asarray([
                    separation[layer_pos, head]
                    for layer_pos, head, _ in positive_positions
                ])
            ))
            positive_visual_mass = float(np.sum(
                positive_global_weights
                * np.asarray([
                    visual_mass[layer_pos, head]
                    for layer_pos, head, _ in positive_positions
                ])
            ))
            consensus_confidence = (
                relation_agreement
                * positive_stability
                * math.sqrt(
                    max(0.0, positive_separation * positive_visual_mass)
                )
            )

            if consensus_confidence < min_consensus_confidence:
                skipped_samples += 1
                continue

            head_relations = relation_code_from_centroids(
                average_centroids
            )
            relation_disagreement = (
                head_relations != consensus_relation
            ).astype(np.float64)

            centroid_distance = (
                np.linalg.norm(
                    average_centroids
                    - consensus_centroids[None, None, :, :],
                    axis=-1,
                ).mean(axis=-1)
                / math.sqrt(2.0)
            )
            centroid_distance = np.clip(
                centroid_distance,
                0.0,
                1.0,
            )

            # Stable-but-opposing heads can be actively conflicting, while
            # unstable heads are treated as swap noise.
            consensus_conflict = (
                signal
                * (
                    0.5 * relation_disagreement
                    + 0.5 * centroid_distance
                )
                * (0.5 + 0.5 * stability)
            )
            swap_noise = signal * (1.0 - stability)
            combined = (
                consensus_conflict_weight * consensus_conflict
                + swap_noise_weight * swap_noise
            )

            weight = float(consensus_confidence)
            total_weight += weight
            conflict_sum += weight * combined
            consensus_conflict_sum += weight * consensus_conflict
            swap_noise_sum += weight * swap_noise
            signal_sum += weight * signal
            relation_disagreement_sum += weight * relation_disagreement
            centroid_distance_sum += weight * centroid_distance
            stability_sum += weight * stability
            used_samples += 1

    if used_samples == 0:
        raise RuntimeError(
            "No samples passed --min-consensus-confidence. Lower the threshold."
        )

    denominator = np.maximum(total_weight, 1e-12)
    conflict_score = conflict_sum / denominator
    consensus_conflict_score = consensus_conflict_sum / denominator
    swap_noise_score = swap_noise_sum / denominator
    mean_signal = signal_sum / denominator
    mean_relation_disagreement = (
        relation_disagreement_sum / denominator
    )
    mean_centroid_distance = centroid_distance_sum / denominator
    mean_stability = stability_sum / denominator

    if exclude_positive_heads:
        for layer, head in positive_keys:
            layer_pos = layer_to_pos[layer]
            conflict_score[layer_pos, head] = -np.inf

    ranked = []
    for layer_pos, layer in enumerate(layers.tolist()):
        for head in range(n_heads):
            score = float(conflict_score[layer_pos, head])
            if not np.isfinite(score):
                continue
            ranked.append({
                "layer": int(layer),
                "head": int(head),
                "conflict_score": score,
                "consensus_conflict_score": float(
                    consensus_conflict_score[layer_pos, head]
                ),
                "swap_noise_score": float(
                    swap_noise_score[layer_pos, head]
                ),
                "mean_signal": float(mean_signal[layer_pos, head]),
                "relation_disagreement_rate": float(
                    mean_relation_disagreement[layer_pos, head]
                ),
                "mean_centroid_distance": float(
                    mean_centroid_distance[layer_pos, head]
                ),
                "mean_swap_stability": float(
                    mean_stability[layer_pos, head]
                ),
                "unsupervised_positive_score": float(
                    unsupervised[layer_pos, head]
                ),
                "oracle_centroid_accuracy": float(
                    oracle_accuracy[layer_pos, head]
                ),
            })
    ranked.sort(
        key=lambda row: row["conflict_score"],
        reverse=True,
    )

    low_ranked = sorted(
        ranked,
        key=lambda row: row["conflict_score"],
    )

    return {
        "script_version": SCRIPT_VERSION,
        "step1_dir": str(step1_dir),
        "used_samples": used_samples,
        "skipped_samples": skipped_samples,
        "positive_consensus_k": positive_k,
        "positive_heads": [
            {
                "layer": int(layer),
                "head": int(head),
                "unsupervised_score": float(score),
                "oracle_centroid_accuracy": float(
                    oracle_accuracy[layer_to_pos[layer], head]
                ),
            }
            for layer, head, score in positive_heads
        ],
        "consensus_relation_counts": {
            relation: int(consensus_relation_counts[index])
            for index, relation in enumerate(RELATIONS)
        },
        "score_config": {
            "min_consensus_confidence": min_consensus_confidence,
            "consensus_conflict_weight": consensus_conflict_weight,
            "swap_noise_weight": swap_noise_weight,
            "exclude_positive_heads": exclude_positive_heads,
        },
        "ranked_conflict_heads": ranked,
        "ranked_low_conflict_heads": low_ranked,
        "layer_indices": [int(value) for value in layers.tolist()],
        "n_heads": int(n_heads),
    }


def write_ranking_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    columns = [
        "rank",
        "layer",
        "head",
        "conflict_score",
        "consensus_conflict_score",
        "swap_noise_score",
        "mean_signal",
        "relation_disagreement_rate",
        "mean_centroid_distance",
        "mean_swap_stability",
        "unsupervised_positive_score",
        "oracle_centroid_accuracy",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for rank, row in enumerate(rows, 1):
            values = {
                **row,
                "rank": rank,
            }
            handle.write(
                ",".join(str(values[column]) for column in columns)
                + "\n"
            )


def matched_heads(
    *,
    reference: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    used: set[Tuple[int, int]] = set()
    result: List[Dict[str, Any]] = []

    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_layer[int(row["layer"])].append(row)

    for reference_row in reference:
        layer = int(reference_row["layer"])
        head = int(reference_row["head"])
        pool = [
            row
            for row in by_layer[layer]
            if int(row["head"]) != head
            and (layer, int(row["head"])) not in used
        ]
        if not pool:
            raise RuntimeError(
                f"No unused matched head is available in layer {layer}"
            )

        if mode == "random":
            chosen = rng.choice(pool)
        elif mode == "low":
            chosen = min(
                pool,
                key=lambda row: float(row["conflict_score"]),
            )
        else:
            raise ValueError(f"Unsupported matched mode: {mode}")

        used.add((layer, int(chosen["head"])))
        result.append(chosen)
    return result


def build_conditions(
    *,
    ranking: Dict[str, Any],
    groups: Sequence[str],
    top_k_values: Sequence[int],
    suppression_scale: float,
    positive_amplify_scale: float,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    max_k = max(top_k_values)
    conflict_full = ranking["ranked_conflict_heads"][:max_k]
    if len(conflict_full) < max_k:
        raise RuntimeError(
            f"Only {len(conflict_full)} conflict heads found; need {max_k}"
        )

    random_full = matched_heads(
        reference=conflict_full,
        candidates=ranking["ranked_conflict_heads"],
        mode="random",
        seed=seed,
    )
    low_full = matched_heads(
        reference=conflict_full,
        candidates=ranking["ranked_low_conflict_heads"],
        mode="low",
        seed=seed,
    )
    positive_full = ranking["positive_heads"][:max_k]
    if "positive_amplify" in groups and len(positive_full) < max_k:
        raise RuntimeError(
            f"positive_consensus_k={len(positive_full)} is smaller than "
            f"requested top-k={max_k}"
        )

    sources = {
        "conflict": (
            conflict_full,
            suppression_scale,
        ),
        "random_matched": (
            random_full,
            suppression_scale,
        ),
        "low_conflict_matched": (
            low_full,
            suppression_scale,
        ),
        "positive_amplify": (
            positive_full,
            positive_amplify_scale,
        ),
    }

    conditions: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        source_rows, scale = sources[group]
        for k in top_k_values:
            selected = source_rows[:k]
            conditions[f"{group}_k{k}"] = {
                "group": group,
                "scale": float(scale),
                "heads": [
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "score": float(
                            row.get(
                                "conflict_score",
                                row.get("unsupervised_score", 0.0),
                            )
                        ),
                    }
                    for row in selected
                ],
            }
    return conditions


def find_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise AttributeError(
        f"Could not find self-attention inside {type(layer).__name__}"
    )


def find_attention_weights(output: Any) -> Optional[torch.Tensor]:
    if not isinstance(output, (tuple, list)):
        return None
    for value in output[1:]:
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
        f"Unsupported attention output type: {type(output)}"
    )


def project_without_bias(o_proj: Any, value: torch.Tensor) -> torch.Tensor:
    weight = getattr(o_proj, "weight", None)
    if weight is None:
        raise AttributeError(
            f"{type(o_proj).__name__} has no projection weight"
        )
    return F.linear(
        value.to(dtype=weight.dtype),
        weight,
        bias=None,
    )


class VisualContributionScaler:
    """Scale selected heads' visual contribution at object-token positions."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        selected_heads: Sequence[Dict[str, Any]],
        subject_index: int,
        reference_index: int,
        visual_indices: Sequence[int],
        prompt_length: int,
        scale: float,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.selected_heads = list(selected_heads)
        self.subject_index = int(subject_index)
        self.reference_index = int(reference_index)
        self.visual_indices = [int(value) for value in visual_indices]
        self.prompt_length = int(prompt_length)
        self.scale = float(scale)

        self.handles: List[Any] = []
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.patch_events = 0
        self.patched_layers: Counter = Counter()

    def __enter__(self) -> "VisualContributionScaler":
        by_layer: Dict[int, List[int]] = defaultdict(list)
        for row in self.selected_heads:
            by_layer[int(row["layer"])].append(int(row["head"]))

        for layer_index, heads in by_layer.items():
            if not (0 <= layer_index < len(self.decoder_layers)):
                raise ValueError(
                    f"Layer {layer_index} outside decoder range"
                )
            attention = find_self_attention(
                self.decoder_layers[layer_index]
            )
            v_proj = getattr(attention, "v_proj", None)
            o_proj = getattr(attention, "o_proj", None)
            if v_proj is None or o_proj is None:
                raise AttributeError(
                    f"Layer {layer_index} does not expose v_proj/o_proj"
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
                        or attention_output.ndim != 3
                    ):
                        return output

                    batch, query_length, hidden_size = attention_output.shape
                    if query_length != self.prompt_length:
                        return output

                    weights = weights.float()
                    n_heads = int(weights.shape[1])
                    key_length = int(weights.shape[-1])
                    if (
                        int(weights.shape[-2]) != query_length
                        or key_length != self.prompt_length
                        or hidden_size % n_heads != 0
                    ):
                        return output

                    head_dim = hidden_size // n_heads
                    if (
                        values_raw.ndim != 3
                        or values_raw.shape[0] != batch
                        or values_raw.shape[1] != key_length
                        or values_raw.shape[-1] % head_dim != 0
                    ):
                        raise RuntimeError(
                            "v_proj output is incompatible with attention output"
                        )

                    n_kv_heads = int(
                        values_raw.shape[-1] // head_dim
                    )
                    if n_heads % n_kv_heads != 0:
                        raise RuntimeError(
                            f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
                        )

                    values = values_raw.float().reshape(
                        batch,
                        key_length,
                        n_kv_heads,
                        head_dim,
                    ).permute(0, 2, 1, 3)
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
                    visual = [
                        index
                        for index in self.visual_indices
                        if 0 <= index < key_length
                    ]
                    if not visual:
                        raise RuntimeError("No visual key indices found")

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

                    selected_weights = weights.index_select(
                        1,
                        head_index,
                    ).index_select(
                        2,
                        query_index,
                    ).index_select(
                        -1,
                        visual_index,
                    )
                    selected_values = values.index_select(
                        1,
                        head_index,
                    ).index_select(
                        2,
                        visual_index,
                    )
                    visual_output = torch.einsum(
                        "bhqv,bhvd->bhqd",
                        selected_weights,
                        selected_values,
                    )
                    delta = (self.scale - 1.0) * visual_output

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
                            delta_heads[:, head, query, :] = delta[
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
                            device=attention_output.device,
                            dtype=attention_output.dtype,
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


def summarize_results(
    rows: Sequence[Dict[str, Any]],
    condition_names: Sequence[str],
) -> Dict[str, Any]:
    baseline_valid = [
        row for row in rows if row.get("baseline_prediction")
    ]
    output: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_baseline_valid": len(baseline_valid),
        "baseline_accuracy": (
            float(np.mean([
                bool(row["baseline_correct"])
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
        per_relation = {
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
            baseline_correct = bool(row["baseline_correct"])
            condition_correct = bool(result["correct"])
            fixed += int((not baseline_correct) and condition_correct)
            broken += int(baseline_correct and (not condition_correct))
            changed += int(
                row["baseline_prediction"] != result["prediction"]
            )

            stats = per_relation[row["gt"]]
            stats["n"] += 1
            stats["baseline_correct"] += int(baseline_correct)
            stats["condition_correct"] += int(condition_correct)
            stats["fixed"] += int(
                (not baseline_correct) and condition_correct
            )
            stats["broken"] += int(
                baseline_correct and (not condition_correct)
            )

        baseline_accuracy = (
            float(np.mean([
                bool(row["baseline_correct"]) for row in valid
            ]))
            if valid else None
        )
        condition_accuracy = (
            float(np.mean([
                bool(row["conditions"][condition]["correct"])
                for row in valid
            ]))
            if valid else None
        )

        per_relation_output = {}
        for relation, stats in per_relation.items():
            n = stats["n"]
            per_relation_output[relation] = {
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

        output["conditions"][condition] = {
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
            "per_relation": per_relation_output,
        }

    return output


def print_discovery(ranking: Dict[str, Any], limit: int = 10) -> None:
    print("\n" + "=" * 104)
    print("CONFLICT-HEAD DISCOVERY")
    print("=" * 104)
    print(
        f"used samples={ranking['used_samples']}, "
        f"skipped={ranking['skipped_samples']}"
    )
    print("\nPositive consensus heads:")
    for row in ranking["positive_heads"]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} | "
            f"unsup={row['unsupervised_score']:.6f} | "
            f"oracle_acc={row['oracle_centroid_accuracy']:.4f}"
        )

    print(f"\nTop {limit} conflict candidates:")
    for rank, row in enumerate(
        ranking["ranked_conflict_heads"][:limit],
        1,
    ):
        print(
            f"  {rank:2d}. L{row['layer']:02d} H{row['head']:02d} | "
            f"score={row['conflict_score']:.6f} | "
            f"signal={row['mean_signal']:.4f} | "
            f"rel_dis={row['relation_disagreement_rate']:.4f} | "
            f"dist={row['mean_centroid_distance']:.4f} | "
            f"swap_stab={row['mean_swap_stability']:.4f} | "
            f"oracle_acc={row['oracle_centroid_accuracy']:.4f}"
        )


def print_summary(
    summary: Dict[str, Any],
    conditions: Dict[str, Dict[str, Any]],
) -> None:
    print("\n" + "=" * 108)
    print("CONFLICT-HEAD SUPPRESSION RESULTS")
    print("=" * 108)
    baseline = summary["baseline_accuracy"]
    print(
        f"baseline generation: {baseline:.4f} "
        f"(n={summary['n_baseline_valid']})"
        if baseline is not None
        else "baseline generation: n/a"
    )

    for name, stats in summary["conditions"].items():
        condition = conditions[name]
        heads = ", ".join(
            f"L{row['layer']}H{row['head']}"
            for row in condition["heads"]
        )
        print(
            f"\n{name} | scale={condition['scale']}\n"
            f"  heads: {heads}\n"
            f"  accuracy={stats['condition_accuracy']:.4f} | "
            f"delta={stats['absolute_change']:+.4f} | "
            f"fixed={stats['fixed']} | broken={stats['broken']} | "
            f"net={stats['net_fixed_minus_broken']:+d} | "
            f"changed={stats['prediction_changed']}"
        )
        for relation, relation_stats in stats["per_relation"].items():
            if relation_stats["n"] == 0:
                continue
            print(
                f"    {relation:6s} n={relation_stats['n']:4d} | "
                f"base={relation_stats['baseline_accuracy']:.4f} | "
                f"new={relation_stats['condition_accuracy']:.4f} | "
                f"fixed={relation_stats['fixed']} | "
                f"broken={relation_stats['broken']}"
            )


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (0.0 <= args.suppression_scale <= 1.0):
        raise ValueError("--suppression-scale must be in [0, 1]")
    if args.positive_amplify_scale < 1.0:
        raise ValueError("--positive-amplify-scale must be >= 1")
    if args.positive_consensus_k <= 0:
        raise ValueError("--positive-consensus-k must be positive")
    if not (0.0 <= args.min_consensus_confidence <= 1.0):
        raise ValueError(
            "--min-consensus-confidence must be in [0, 1]"
        )
    if args.consensus_conflict_weight < 0 or args.swap_noise_weight < 0:
        raise ValueError("Discovery weights must be nonnegative")
    if (
        args.consensus_conflict_weight
        + args.swap_noise_weight
        <= 0
    ):
        raise ValueError("At least one discovery weight must be positive")

    groups = parse_csv_names(
        args.condition_groups,
        GROUPS,
    )
    top_k_values = parse_positive_ints(args.top_k)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = (
        Path(args.ranking_json)
        if args.ranking_json
        else output_dir / "conflict_head_ranking.json"
    )
    ranking_csv_path = output_dir / "conflict_head_ranking.csv"

    if args.only in ("discover", "both"):
        ranking = discover_conflict_heads(
            step1_dir=Path(args.step1_dir),
            positive_k=args.positive_consensus_k,
            max_samples=args.discovery_max_samples,
            min_consensus_confidence=args.min_consensus_confidence,
            consensus_conflict_weight=args.consensus_conflict_weight,
            swap_noise_weight=args.swap_noise_weight,
            exclude_positive_heads=args.exclude_positive_heads,
        )
        ranking_path.write_text(
            json.dumps(ranking, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_ranking_csv(
            ranking_csv_path,
            ranking["ranked_conflict_heads"],
        )
        print_discovery(ranking)
        print(f"\nSaved ranking JSON: {ranking_path}")
        print(f"Saved ranking CSV:  {ranking_csv_path}")
    else:
        if not ranking_path.exists():
            raise FileNotFoundError(
                f"Missing ranking JSON for intervention: {ranking_path}"
            )
        ranking = json.loads(
            ranking_path.read_text(encoding="utf-8")
        )

    if args.only == "discover":
        return

    conditions = build_conditions(
        ranking=ranking,
        groups=groups,
        top_k_values=top_k_values,
        suppression_scale=args.suppression_scale,
        positive_amplify_scale=args.positive_amplify_scale,
        seed=args.random_seed,
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
            f"Prompt file is missing {len(missing_ids)} IDs; "
            f"first={missing_ids[:10]}"
        )

    if args.model not in module.SPECS:
        raise ValueError(
            f"Model {args.model!r} not found in model specs"
        )
    spec = module.SPECS[args.model]

    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    if results_path.exists():
        raise RuntimeError(
            f"{results_path} already exists. Use --overwrite or a new directory."
        )

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {spec.model_class}"
        )

    print(f"\nLoading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    for condition in conditions.values():
        for row in condition["heads"]:
            layer = int(row["layer"])
            if not (0 <= layer < len(decoder_layers)):
                raise ValueError(
                    f"Selected layer {layer} outside decoder range "
                    f"[0, {len(decoder_layers) - 1}]"
                )

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "model": args.model,
        "repo_id": spec.repo_id,
        "transformers_version": transformers.__version__,
        "step1_dir": args.step1_dir,
        "ranking_json": str(ranking_path),
        "decoder_path": decoder_path,
        "condition_groups": groups,
        "top_k": top_k_values,
        "suppression_scale": args.suppression_scale,
        "positive_amplify_scale": args.positive_amplify_scale,
        "conditions": conditions,
        "n_records": len(records),
        "audit": audit,
        "uses_gt_for_head_selection": False,
        "updates_model_weights": False,
        "changes_prompt": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Decoder: {decoder_path}, n={len(decoder_layers)}")
    for name, condition in conditions.items():
        print(
            f"{name}: scale={condition['scale']} | "
            + ", ".join(
                f"L{row['layer']}H{row['head']}"
                for row in condition["heads"]
            )
        )

    completed = 0
    started = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"conflict-suppression:{args.model}",
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
                for name, condition in conditions.items():
                    with VisualContributionScaler(
                        decoder_layers=decoder_layers,
                        selected_heads=condition["heads"],
                        subject_index=subject_index,
                        reference_index=reference_index,
                        visual_indices=visual_indices,
                        prompt_length=prompt_length,
                        scale=float(condition["scale"]),
                    ) as intervention:
                        generated_text = generate_for_analysis(
                            model=model,
                            processor=processor,
                            batch=batch,
                            max_new_tokens=args.max_new_tokens,
                        )

                    if intervention.patch_events == 0:
                        raise RuntimeError(
                            f"Condition {name} registered no intervention events"
                        )

                    prediction = normalize_relation(generated_text)
                    correct = bool(
                        prediction is not None
                        and prediction == gt
                    )
                    condition_results[name] = {
                        "prediction": prediction,
                        "correct": correct,
                        "generated_text": generated_text,
                        "scale": float(condition["scale"]),
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
                append_jsonl(results_path, row)
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
                    for name, result in condition_results.items():
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
                            f"  {name:28s}: "
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

        rows = read_jsonl(results_path)
        if not rows:
            raise RuntimeError(
                "No intervention samples completed; inspect errors.jsonl"
            )
        summary = summarize_results(
            rows,
            condition_names,
        )
        summary["config"] = config
        summary["elapsed_minutes"] = (
            time.time() - started
        ) / 60.0
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print_summary(summary, conditions)
        print(f"\nSaved results: {results_path}")
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
