#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-model single-head natural object->prompt-last receiver scaling scan.

Models / default scan windows
=============================
llava-7b:
    L20-L31, all query heads
    expected language decoder: 32 layers, 32 query heads/layer

qwen-7b:
    L18-L27, all query heads
    expected language decoder: 28 layers, 28 query heads/layer

Sampling
========
Randomly sample 200 COCO-two records by SID using --seed (default 17).
With the same dataset + seed, llava-7b and qwen-7b use the SAME 200 SIDs.

Intervention
============
For every selected sample and every scanned query head (L,h), first obtain the
CLEAN sample-specific natural object-text -> prompt-last post-W_O contribution:

    c_{L,h}
      = W_O^{L,h} sum_{s in subject/reference text tokens}
                    A_{L,h}[last,s] V_{L,h}[s]

Then run full greedy model.generate() and patch only that layer/head message at
prompt-last during PREFILL:

    attention_out'_L[last]
      = attention_out_L[last] + scale * c_{L,h}

Default scale=6, matching the Qwen-3B discovery scan.

This does NOT:
    * modify attention probabilities A,
    * construct a class / GT direction,
    * use GT to choose the intervention,
    * use probe ACC as the final metric.

Primary metric is REAL full-generation ACC:
    patched ACC
    W->C
    C->W
    net = W->C - C->W
    generation changed rate

LLaVA merged-position handling
==============================
For LLaVA, the raw prompt may contain an image placeholder that expands into
many visual embeddings before the language decoder. Therefore raw input_ids
positions can differ from decoder positions.

This script:
  1) captures the actual language-decoder PREFILL sequence length during clean
     model.generate(),
  2) compares it with raw input_ids length,
  3) maps subject/reference positions into decoder coordinates,
  4) uses prompt_last = merged_decoder_length - 1,
  5) patches only when the attention q_len equals that merged prefill length.

For Qwen2.5-VL the raw/decoder lengths are normally already aligned; then the
mapping is identity.

Dependencies in AdaptVis root / PYTHONPATH
==========================================
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py

Recommended parallel runs on two GPUs
=====================================

# GPU 0: LLaVA-1.5-7B
CUDA_VISIBLE_DEVICES=0 python -u \
  scan_coco_receiver_scaling_multimodel_200_v1.py \
  --model llava-7b \
  --num-samples 200 \
  --scale 6 \
  --device cuda:0 \
  --output-dir output/llava7b_receiver_scan_L20_L31_n200_s6_v1 \
  --overwrite

# GPU 1: Qwen2.5-VL-7B
CUDA_VISIBLE_DEVICES=1 python -u \
  scan_coco_receiver_scaling_multimodel_200_v1.py \
  --model qwen-7b \
  --num-samples 200 \
  --scale 6 \
  --device cuda:0 \
  --output-dir output/qwen7b_receiver_scan_L18_L27_n200_s6_v1 \
  --overwrite

Note:
When CUDA_VISIBLE_DEVICES=1 is set, that physical GPU becomes cuda:0 inside the
process, hence --device cuda:0 is intentional.

Outputs
=======
selected_sids.json
baseline_eval.jsonl
baseline_eval.csv
patch_results.jsonl
patch_results.csv
single_head_summary.csv
layer_summary.csv
relation_summary.csv
config.json
report.txt
errors.jsonl

Resume
======
The script appends after every baseline / patched generation. If interrupted,
rerun the same command WITHOUT --overwrite. Completed conditions are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import re
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-receiver-scaling-multimodel-200-v1"

RELATIONS = ("left", "right", "above", "below")

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

MODEL_WINDOWS: Dict[str, Dict[str, Any]] = {
    "llava-7b": {
        "layers": list(range(20, 32)),
        "expected_decoder_layers": 32,
        "expected_query_heads": 32,
        "description": "LLaVA-1.5-7B",
    },
    "qwen-7b": {
        "layers": list(range(18, 28)),
        "expected_decoder_layers": 28,
        "expected_query_heads": 28,
        "description": "Qwen2.5-VL-7B",
    },
}


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_WINDOWS),
    )
    p.add_argument(
        "--dataset",
        default="coco_two",
        choices=("coco_two",),
    )
    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Randomly selected records. 0 means all records.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Same seed => same selected SIDs across models.",
    )

    p.add_argument(
        "--layers",
        default="",
        help=(
            "Optional override, e.g. 20-31. Empty uses model defaults: "
            "llava-7b=L20-L31; qwen-7b=L18-L27."
        ),
    )
    p.add_argument(
        "--heads",
        default="",
        help=(
            "Optional explicit list, e.g. 26:4,26:6. "
            "Empty scans ALL query heads in selected layers."
        ),
    )

    p.add_argument(
        "--scale",
        type=float,
        default=6.0,
        help=(
            "Added gain. scale=6 means attention_out += 6*c_Lh; "
            "the selected natural message has effective coefficient ~7."
        ),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )
    p.add_argument(
        "--device",
        default="cuda:0",
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager",),
    )
    p.add_argument(
        "--trace-chunk-size",
        type=int,
        default=0,
        help=(
            "0 traces all selected layers in one clean forward. "
            "Positive value splits layer replay into chunks."
        ),
    )

    p.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument(
        "--empty-cache-every",
        type=int,
        default=5,
    )
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


def parse_int_ranges(text: str) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()

    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            a_text, b_text = chunk.split("-", 1)
            a = int(a_text)
            b = int(b_text)
            step = 1 if b >= a else -1
            values = range(a, b + step, step)
        else:
            values = [int(chunk)]

        for value in values:
            if value not in seen:
                out.append(value)
                seen.add(value)

    return out


def parse_head(text: str) -> Tuple[int, int]:
    value = str(text).strip().upper()
    value = value.replace("L", "").replace("H", ":")
    while "::" in value:
        value = value.replace("::", ":")

    if ":" not in value:
        raise ValueError(f"Bad head specification: {text!r}")

    a, b = value.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()

    for item in str(text).split(","):
        if not item.strip():
            continue
        head = parse_head(item)
        if head not in seen:
            out.append(head)
            seen.add(head)

    return out


def hname(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip().lower()

    aliases = {
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "over": "above",
        "on": "above",
    }
    if text in RELATIONS:
        return text
    if text in aliases:
        return aliases[text]

    hits: List[Tuple[int, str]] = []

    for relation in RELATIONS:
        match = re.search(
            rf"\b{re.escape(relation)}\b",
            text,
        )
        if match:
            hits.append((match.start(), relation))

    for pattern, relation in (
        (r"\bunder(?:neath)?\b|\bbeneath\b", "below"),
        (r"\bover\b|\bon top\b", "above"),
    ):
        match = re.search(pattern, text)
        if match:
            hits.append((match.start(), relation))

    if not hits:
        return None

    hits.sort()
    return hits[0][1]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "t"
    }


def safe_mean(values: Iterable[Any]) -> float:
    xs: List[float] = []

    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)

    return float(np.mean(xs)) if xs else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    xs: List[float] = []

    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)

    return float(np.median(xs)) if xs else float("nan")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(dict(row))


def append_jsonl(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(
        model,
        "generation_config",
        None,
    )

    if cfg is None:
        return

    for name in (
        "temperature",
        "top_p",
        "top_k",
    ):
        if hasattr(cfg, name):
            setattr(cfg, name, None)


def relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    """
    Used only by attention_helper.run_and_trace() to compute its clean
    first-step relation diagnostic. The causal scan itself is ranked by full
    generation, not by this score.
    """
    result: Dict[str, List[int]] = {}

    for relation in RELATIONS:
        ids = set()

        for candidate in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            token_ids = tokenizer.encode(
                candidate,
                add_special_tokens=False,
            )
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))

        if not ids:
            token_ids = tokenizer.encode(
                " " + relation,
                add_special_tokens=False,
            )
            if not token_ids:
                raise RuntimeError(
                    f"No token IDs found for relation {relation!r}"
                )
            ids.add(int(token_ids[-1]))

        result[relation] = sorted(ids)

    return result


# =============================================================================
# Sampling
# =============================================================================

def select_random_records(
    records: Sequence[Any],
    *,
    num_samples: int,
    seed: int,
) -> List[Any]:
    ordered = sorted(
        records,
        key=lambda record: int(record.sid),
    )

    if num_samples <= 0 or num_samples >= len(ordered):
        return ordered

    rng = random.Random(seed)
    selected = rng.sample(
        ordered,
        int(num_samples),
    )

    selected.sort(
        key=lambda record: int(record.sid)
    )
    return selected


# =============================================================================
# Attention-module output helpers
# =============================================================================

def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected attention output [B,S,D], got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise RuntimeError(
        "Could not locate 3D attention output tensor."
    )


def replace_first_3d(
    output: Any,
    replacement: torch.Tensor,
) -> Any:
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        items = list(output)

        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return tuple(items)

    if isinstance(output, list):
        items = list(output)

        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return items

    raise RuntimeError(
        "Could not replace 3D attention output tensor."
    )


def locate_hidden_3d(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    candidate = kwargs.get("hidden_states")

    if torch.is_tensor(candidate) and candidate.ndim == 3:
        return candidate

    for value in args:
        if torch.is_tensor(value) and value.ndim == 3:
            return value

    raise RuntimeError(
        "Could not locate 3D hidden_states in attention call."
    )


class CaptureDecoderPrefillLength:
    """
    Capture the actual language-decoder self-attention q_len during generate().

    This is required for LLaVA because raw input_ids can use an image placeholder
    while the decoder sees the expanded multimodal sequence.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layer: int,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layer = int(layer)
        self.handle = None
        self.lengths: List[int] = []

    def __enter__(
        self,
    ) -> "CaptureDecoderPrefillLength":
        attention = self.attention_helper.resolve_self_attention(
            self.decoder_layers[self.layer]
        )

        def pre_hook(
            _module: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> None:
            hidden = locate_hidden_3d(
                args,
                kwargs,
            )
            self.lengths.append(
                int(hidden.shape[1])
            )

        self.handle = attention.register_forward_pre_hook(
            pre_hook,
            with_kwargs=True,
        )

        return self

    def prefill_length(self) -> int:
        if not self.lengths:
            raise RuntimeError(
                f"L{self.layer}: decoder prefill-length hook never fired."
            )

        # Full prefill has the largest q_len; cached decode steps are usually 1.
        return max(self.lengths)

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()

        self.handle = None


class PromptLastLayerDelta:
    """
    Add one fixed CLEAN sample/head residual-space vector to one language
    self-attention layer at decoder prompt-last during PREFILL only.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layer: int,
        merged_prompt_length: int,
        merged_prompt_last: int,
        delta: np.ndarray,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper

        self.layer = int(layer)
        self.merged_prompt_length = int(
            merged_prompt_length
        )
        self.merged_prompt_last = int(
            merged_prompt_last
        )
        self.delta = np.asarray(
            delta,
            dtype=np.float32,
        )

        self.handle = None
        self.applications = 0

    def __enter__(
        self,
    ) -> "PromptLastLayerDelta":
        attention = self.attention_helper.resolve_self_attention(
            self.decoder_layers[self.layer]
        )

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> Any:
            hidden = first_3d(output)

            # Only the full multimodal prompt prefill.
            if int(hidden.shape[1]) != self.merged_prompt_length:
                return None

            if not (
                0
                <= self.merged_prompt_last
                < int(hidden.shape[1])
            ):
                raise RuntimeError(
                    f"L{self.layer}: merged prompt-last "
                    f"{self.merged_prompt_last} outside q_len={hidden.shape[1]}"
                )

            if int(hidden.shape[-1]) != int(self.delta.shape[0]):
                raise RuntimeError(
                    f"L{self.layer}: delta dim={self.delta.shape[0]} "
                    f"but attention output dim={hidden.shape[-1]}"
                )

            modified = hidden.clone()

            modified[
                0,
                self.merged_prompt_last,
            ] += torch.as_tensor(
                self.delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            self.applications += 1

            return replace_first_3d(
                output,
                modified,
            )

        self.handle = attention.register_forward_hook(
            hook
        )

        return self

    def validate(self) -> None:
        if self.applications != 1:
            raise RuntimeError(
                f"L{self.layer}: expected exactly one prefill patch, "
                f"got {self.applications}."
            )

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()

        self.handle = None


# =============================================================================
# Raw-token -> merged decoder-position mapping
# =============================================================================

def candidate_token_id(
    tokenizer: Any,
    token: str,
) -> Optional[int]:
    try:
        value = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None

    if value is None:
        return None

    try:
        value = int(value)
    except Exception:
        return None

    unk = getattr(
        tokenizer,
        "unk_token_id",
        None,
    )

    if unk is not None and value == int(unk):
        return None

    return value


def image_placeholder_positions(
    *,
    model: Any,
    processor: Any,
    input_ids: Sequence[int],
) -> List[int]:
    token_ids: Set[int] = set()

    objects = [
        getattr(model, "config", None),
        getattr(
            getattr(model, "config", None),
            "text_config",
            None,
        ),
        processor,
        getattr(processor, "tokenizer", None),
    ]

    for obj in objects:
        if obj is None:
            continue

        for name in (
            "image_token_id",
            "image_token_index",
        ):
            value = getattr(
                obj,
                name,
                None,
            )

            if isinstance(
                value,
                (int, np.integer),
            ):
                token_ids.add(int(value))

    tokenizer = processor.tokenizer

    for token in (
        "<image>",
        "<|image_pad|>",
        "<image_token>",
        "<IMG_CONTEXT>",
    ):
        value = candidate_token_id(
            tokenizer,
            token,
        )
        if value is not None:
            token_ids.add(value)

    return [
        index
        for index, token_id in enumerate(input_ids)
        if int(token_id) in token_ids
    ]


def map_text_positions_to_decoder(
    *,
    model: Any,
    processor: Any,
    input_ids: Sequence[int],
    raw_positions: Sequence[int],
    merged_length: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Qwen case:
        merged_length == raw_length -> identity.

    LLaVA placeholder-expansion case:
        the image is inserted before the semantic question. If decoder length is
        larger than raw input_ids length, all question positions AFTER the image
        placeholder shift by delta = merged_length - raw_length.

    This intentionally refuses ambiguous mappings instead of silently using the
    wrong object tokens.
    """
    raw_length = len(input_ids)
    raw_positions = sorted(
        set(map(int, raw_positions))
    )

    if not raw_positions:
        raise RuntimeError(
            "No raw text positions to map."
        )

    if merged_length == raw_length:
        return raw_positions, {
            "mapping_mode": "identity",
            "raw_length": raw_length,
            "merged_length": merged_length,
            "shift": 0,
            "image_placeholders": image_placeholder_positions(
                model=model,
                processor=processor,
                input_ids=input_ids,
            ),
        }

    if merged_length < raw_length:
        raise RuntimeError(
            "Decoder sequence is shorter than raw input_ids: "
            f"raw={raw_length}, merged={merged_length}. "
            "No safe mapper is implemented for this backend."
        )

    placeholders = image_placeholder_positions(
        model=model,
        processor=processor,
        input_ids=input_ids,
    )

    shift = int(
        merged_length - raw_length
    )

    # In the standard LLaVA chat prompt, image placeholder(s) precede the text
    # question. We require that all object text positions lie after them.
    if placeholders:
        last_image_position = max(placeholders)

        if any(
            position <= last_image_position
            for position in raw_positions
        ):
            raise RuntimeError(
                "At least one target text token is not after the image "
                "placeholder; refusing simple LLaVA shift mapping."
            )
    else:
        # Some processor/model combinations consume/rewrite the image marker
        # before the raw tokenizer output exposes an obvious special ID.
        # The prompt construction used here always inserts the image before the
        # question, so a positive length expansion still shifts all semantic
        # question tokens equally. Make this explicit in diagnostics.
        last_image_position = None

    mapped = [
        int(position + shift)
        for position in raw_positions
    ]

    if any(
        position < 0 or position >= merged_length
        for position in mapped
    ):
        raise RuntimeError(
            f"Mapped positions {mapped} outside merged length {merged_length}."
        )

    return mapped, {
        "mapping_mode": (
            "image_prefix_shift"
            if placeholders
            else "image_prefix_shift_no_marker"
        ),
        "raw_length": raw_length,
        "merged_length": merged_length,
        "shift": shift,
        "image_placeholders": placeholders,
        "last_image_placeholder": last_image_position,
    }


# =============================================================================
# Clean per-head natural message
# =============================================================================

def trace_target_index(
    trace: Any,
    merged_prompt_last: int,
) -> int:
    lookup = {
        int(global_position): local
        for local, global_position in enumerate(
            trace.target_positions
        )
    }

    if int(merged_prompt_last) not in lookup:
        raise RuntimeError(
            f"prompt_last={merged_prompt_last} missing from trace target "
            f"positions {trace.target_positions}."
        )

    return int(
        lookup[int(merged_prompt_last)]
    )


def all_head_object_writes(
    *,
    trace: Any,
    merged_prompt_last: int,
    object_positions: Sequence[int],
) -> np.ndarray:
    """
    Return post-W_O object-text -> prompt-last natural message for every QUERY
    head in this layer.

    shape:
        [n_query_heads, hidden_size]
    """
    object_positions = sorted(
        set(map(int, object_positions))
    )

    if not object_positions:
        raise RuntimeError(
            "No object source positions."
        )

    local_target = trace_target_index(
        trace,
        merged_prompt_last,
    )

    source = torch.as_tensor(
        object_positions,
        dtype=torch.long,
    )

    if int(source.max()) >= int(
        trace.value_states.shape[1]
    ):
        raise RuntimeError(
            f"Object source max={int(source.max())} exceeds "
            f"value sequence length={trace.value_states.shape[1]}."
        )

    weights = (
        trace.attention_weights[
            :,
            local_target,
            :,
        ]
        .index_select(
            1,
            source,
        )
        .float()
    )  # [Hq,Sobj]

    values = (
        trace.value_states
        .index_select(
            1,
            source,
        )
        .float()
    )  # [Hq,Sobj,Dh], with GQA values repeated to query heads if required

    pre = torch.einsum(
        "hs,hsd->hd",
        weights,
        values,
    )  # [Hq,Dh]

    post = torch.einsum(
        "hd,ohd->ho",
        pre,
        trace.o_proj_weight.float(),
    )  # [Hq,Dmodel]

    return (
        post.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def extract_clean_messages(
    *,
    attention_helper: Any,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    merged_prompt_last: int,
    object_positions: Sequence[int],
    chunk_size: int,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, float],
]:
    messages: Dict[int, np.ndarray] = {}
    replay_errors: Dict[int, float] = {}

    layers = list(map(int, layers))

    if chunk_size <= 0:
        chunks = [layers]
    else:
        chunks = [
            layers[
                start:
                start + int(chunk_size)
            ]
            for start in range(
                0,
                len(layers),
                int(chunk_size),
            )
        ]

    for chunk in chunks:
        _, traces = attention_helper.run_and_trace(
            model=model,
            batch=batch,
            token_map=relation_token_map,
            decoder_layers=decoder_layers,
            layer_indices=chunk,
            target_positions=[
                int(merged_prompt_last)
            ],
        )

        for layer in chunk:
            trace = traces[int(layer)]

            messages[int(layer)] = all_head_object_writes(
                trace=trace,
                merged_prompt_last=merged_prompt_last,
                object_positions=object_positions,
            )

            replay_errors[int(layer)] = float(
                trace.replay_relative_error
            )

        del traces

    return messages, replay_errors


# =============================================================================
# Prompt / generation
# =============================================================================

def build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Any:
    rendered = probe.build_chat_prompt(
        processor,
        question,
        True,
    )

    return probe.process_inputs(
        processor,
        rendered,
        image,
        device,
    )


@torch.inference_mode()
def clean_generation_with_geometry(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    geometry_layer: int,
    max_new_tokens: int,
) -> Tuple[
    Optional[str],
    str,
    int,
]:
    """
    Return:
        prediction,
        decoded generated continuation,
        actual language-decoder merged prefill length.
    """
    raw_prompt_length = int(
        batch["input_ids"].shape[1]
    )

    with CaptureDecoderPrefillLength(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layer=geometry_layer,
    ) as capture:
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        merged_prompt_length = (
            capture.prefill_length()
        )

    generated_text = processor.tokenizer.decode(
        generated[
            0,
            raw_prompt_length:,
        ],
        skip_special_tokens=True,
    ).strip()

    del generated

    return (
        normalize_relation(generated_text),
        generated_text,
        int(merged_prompt_length),
    )


@torch.inference_mode()
def patched_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    layer: int,
    merged_prompt_length: int,
    merged_prompt_last: int,
    delta: np.ndarray,
    max_new_tokens: int,
) -> Tuple[
    Optional[str],
    str,
]:
    raw_prompt_length = int(
        batch["input_ids"].shape[1]
    )

    with PromptLastLayerDelta(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layer=layer,
        merged_prompt_length=merged_prompt_length,
        merged_prompt_last=merged_prompt_last,
        delta=delta,
    ) as patch:
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        patch.validate()

    generated_text = processor.tokenizer.decode(
        generated[
            0,
            raw_prompt_length:,
        ],
        skip_special_tokens=True,
    ).strip()

    del generated

    return (
        normalize_relation(generated_text),
        generated_text,
    )


# =============================================================================
# Head shape
# =============================================================================

def infer_num_query_heads(
    *,
    attention: Any,
    model: Any,
) -> int:
    for obj in (
        attention,
        getattr(attention, "config", None),
        getattr(model, "config", None),
        getattr(
            getattr(model, "config", None),
            "text_config",
            None,
        ),
    ):
        if obj is None:
            continue

        for name in (
            "num_heads",
            "num_attention_heads",
            "n_heads",
        ):
            value = getattr(
                obj,
                name,
                None,
            )

            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass

    q_proj = getattr(
        attention,
        "q_proj",
        None,
    )
    head_dim = getattr(
        attention,
        "head_dim",
        None,
    )

    if (
        q_proj is not None
        and head_dim is not None
        and hasattr(q_proj, "out_features")
    ):
        return (
            int(q_proj.out_features)
            // int(head_dim)
        )

    raise RuntimeError(
        f"Could not infer query-head count for {type(attention).__name__}."
    )


def infer_head_shape(
    *,
    attention: Any,
    model: Any,
) -> Tuple[int, int, int]:
    n_heads = infer_num_query_heads(
        attention=attention,
        model=model,
    )

    o_proj = getattr(
        attention,
        "o_proj",
        None,
    )

    if o_proj is None:
        raise RuntimeError(
            f"{type(attention).__name__} has no o_proj."
        )

    in_features = getattr(
        o_proj,
        "in_features",
        None,
    )
    out_features = getattr(
        o_proj,
        "out_features",
        None,
    )

    if (
        in_features is None
        and hasattr(o_proj, "weight")
    ):
        in_features = int(
            o_proj.weight.shape[1]
        )

    if (
        out_features is None
        and hasattr(o_proj, "weight")
    ):
        out_features = int(
            o_proj.weight.shape[0]
        )

    if in_features is None or out_features is None:
        raise RuntimeError(
            "Could not infer o_proj shape."
        )

    in_features = int(in_features)
    out_features = int(out_features)

    if in_features % n_heads != 0:
        raise RuntimeError(
            f"o_proj in_features={in_features} not divisible "
            f"by query_heads={n_heads}."
        )

    head_dim = int(
        in_features // n_heads
    )

    return n_heads, head_dim, out_features


# =============================================================================
# Resume keys
# =============================================================================

def patch_key(
    *,
    sid: int,
    layer: int,
    head: int,
    scale: float,
) -> Tuple[
    int,
    int,
    int,
    float,
]:
    return (
        int(sid),
        int(layer),
        int(head),
        round(
            float(scale),
            9,
        ),
    )


# =============================================================================
# Summaries
# =============================================================================

def summarize_heads(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_sid = {
        int(row["sid"]): row
        for row in baseline_rows
    }

    grouped: Dict[
        Tuple[int, int, float],
        List[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in patch_rows:
        grouped[
            (
                int(row["layer"]),
                int(row["head"]),
                float(row["scale"]),
            )
        ].append(row)

    summary: List[Dict[str, Any]] = []

    for (
        layer,
        head,
        scale,
    ), rows in grouped.items():
        patched_by_sid = {
            int(row["sid"]): normalize_relation(
                row["patched_generation_prediction"]
            )
            for row in rows
        }

        covered = [
            sid
            for sid in baseline_by_sid
            if sid in patched_by_sid
        ]

        if not covered:
            continue

        base_correct = []
        new_correct = []

        w2c = 0
        c2w = 0
        changed = 0

        for sid in covered:
            base = baseline_by_sid[sid]
            gt = normalize_relation(
                base["gt"]
            )
            base_prediction = normalize_relation(
                base["generation_prediction"]
            )
            patched_prediction = patched_by_sid[sid]

            base_ok = parse_bool(
                base["generation_correct"]
            )
            new_ok = (
                patched_prediction == gt
            )

            base_correct.append(
                float(base_ok)
            )
            new_correct.append(
                float(new_ok)
            )

            w2c += int(
                (not base_ok)
                and new_ok
            )
            c2w += int(
                base_ok
                and (not new_ok)
            )
            changed += int(
                patched_prediction
                != base_prediction
            )

        base_acc = safe_mean(
            base_correct
        )
        patched_acc = safe_mean(
            new_correct
        )

        summary.append({
            "layer": layer,
            "head": head,
            "head_name": hname(
                layer,
                head,
            ),
            "scale": scale,
            "N_expected": len(
                baseline_by_sid
            ),
            "N_completed": len(
                covered
            ),
            "complete": (
                len(covered)
                == len(baseline_by_sid)
            ),
            "baseline_acc": base_acc,
            "patched_acc": patched_acc,
            "delta_acc": (
                patched_acc
                - base_acc
            ),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net_repairs": (
                w2c
                - c2w
            ),
            "generation_changed": changed,
            "generation_changed_rate": (
                changed
                / max(
                    len(covered),
                    1,
                )
            ),
            "mean_message_norm": safe_mean(
                row["message_norm"]
                for row in rows
            ),
            "mean_delta_norm": safe_mean(
                row["delta_norm"]
                for row in rows
            ),
            "mean_replay_relative_error": safe_mean(
                row["replay_relative_error"]
                for row in rows
            ),
        })

    summary.sort(
        key=lambda row: (
            0
            if bool(row["complete"])
            else 1,
            -int(row["net_repairs"]),
            -float(row["patched_acc"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )

    return summary


def summarize_layers(
    head_summary: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        int,
        List[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in head_summary:
        grouped[
            int(row["layer"])
        ].append(row)

    output: List[Dict[str, Any]] = []

    for layer in sorted(grouped):
        rows = grouped[layer]

        complete = [
            row
            for row in rows
            if bool(row["complete"])
        ]

        use = complete if complete else rows

        if not use:
            continue

        ranked = sorted(
            use,
            key=lambda row: (
                -int(row["net_repairs"]),
                -float(row["patched_acc"]),
            ),
        )

        best = ranked[0]

        output.append({
            "layer": layer,
            "N_heads": len(rows),
            "N_complete_heads": len(complete),
            "best_head": best["head_name"],
            "best_patched_acc": best["patched_acc"],
            "best_delta_acc": best["delta_acc"],
            "best_wrong_to_correct": best["wrong_to_correct"],
            "best_correct_to_wrong": best["correct_to_wrong"],
            "best_net_repairs": best["net_repairs"],
            "positive_net_heads": sum(
                int(row["net_repairs"]) > 0
                for row in use
            ),
            "zero_net_heads": sum(
                int(row["net_repairs"]) == 0
                for row in use
            ),
            "negative_net_heads": sum(
                int(row["net_repairs"]) < 0
                for row in use
            ),
            "mean_net_repairs": safe_mean(
                row["net_repairs"]
                for row in use
            ),
            "median_net_repairs": safe_median(
                row["net_repairs"]
                for row in use
            ),
        })

    return output


def relation_summary(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    top_heads: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_sid = {
        int(row["sid"]): row
        for row in baseline_rows
    }

    patch_lookup: Dict[
        Tuple[int, int, float],
        Dict[int, Mapping[str, Any]],
    ] = defaultdict(dict)

    for row in patch_rows:
        key = (
            int(row["layer"]),
            int(row["head"]),
            float(row["scale"]),
        )
        patch_lookup[key][
            int(row["sid"])
        ] = row

    output: List[Dict[str, Any]] = []

    for condition in top_heads:
        key = (
            int(condition["layer"]),
            int(condition["head"]),
            float(condition["scale"]),
        )

        rows_by_sid = patch_lookup.get(
            key,
            {},
        )

        for relation in RELATIONS:
            sids = [
                sid
                for sid, base in baseline_by_sid.items()
                if normalize_relation(
                    base["gt"]
                )
                == relation
                and sid in rows_by_sid
            ]

            if not sids:
                continue

            base_acc = safe_mean(
                float(
                    normalize_relation(
                        baseline_by_sid[
                            sid
                        ][
                            "generation_prediction"
                        ]
                    )
                    == relation
                )
                for sid in sids
            )

            patched_acc = safe_mean(
                float(
                    normalize_relation(
                        rows_by_sid[
                            sid
                        ][
                            "patched_generation_prediction"
                        ]
                    )
                    == relation
                )
                for sid in sids
            )

            output.append({
                "head_name": condition["head_name"],
                "layer": condition["layer"],
                "head": condition["head"],
                "scale": condition["scale"],
                "relation": relation,
                "N": len(sids),
                "baseline_acc": base_acc,
                "patched_acc": patched_acc,
                "delta_acc": (
                    patched_acc
                    - base_acc
                ),
            })

    return output


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.scale < 0:
        raise ValueError(
            "--scale must be non-negative."
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_defaults = MODEL_WINDOWS[
        args.model
    ]

    layers = (
        parse_int_ranges(args.layers)
        if args.layers.strip()
        else list(
            model_defaults["layers"]
        )
    )

    if not layers:
        raise ValueError(
            "No layers selected."
        )

    explicit_heads = parse_heads(
        args.heads
    )

    output_dir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and output_dir.exists()
    ):
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_jsonl = (
        output_dir
        / "baseline_eval.jsonl"
    )
    baseline_csv = (
        output_dir
        / "baseline_eval.csv"
    )
    patch_jsonl = (
        output_dir
        / "patch_results.jsonl"
    )
    errors_path = (
        output_dir
        / "errors.jsonl"
    )

    if (
        not args.overwrite
        and not args.resume
        and any(
            output_dir.iterdir()
        )
    ):
        raise RuntimeError(
            f"{output_dir} is not empty. "
            "Use --overwrite or --resume."
        )

    probe = importlib.import_module(
        args.probe_module
    )
    attention_helper = importlib.import_module(
        args.attention_helper_module
    )
    base = probe.base

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )

    selected_records = select_random_records(
        records,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    selected_sids = [
        int(record.sid)
        for record in selected_records
    ]

    (
        output_dir
        / "selected_sids.json"
    ).write_text(
        json.dumps(
            {
                "seed": args.seed,
                "num_samples": len(
                    selected_sids
                ),
                "sids": selected_sids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    record_by_sid = {
        int(record.sid): record
        for record in selected_records
    }

    spec = base.SPECS[
        args.model
    ]

    model_class = getattr(
        transformers,
        spec.model_class,
    )

    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(
            spec.dtype_name
        ),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {
            "": args.device
        },
        "attn_implementation": args.attn_impl,
    }

    model = None
    processor = None

    try:
        print(
            f"Loading {args.model}: {spec.repo_id}",
            flush=True,
        )

        model = model_class.from_pretrained(
            spec.repo_id,
            **load_kwargs,
        )
        model.eval()

        clear_sampling_defaults(
            model
        )

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )

        base.configure_processor(
            model,
            processor,
        )

        device = torch.device(
            args.device
        )

        decoder_layers, decoder_path = (
            probe.resolve_decoder_layers(
                model
            )
        )

        actual_decoder_layers = len(
            decoder_layers
        )

        expected_decoder_layers = int(
            model_defaults[
                "expected_decoder_layers"
            ]
        )

        if (
            actual_decoder_layers
            != expected_decoder_layers
        ):
            raise RuntimeError(
                f"{args.model}: expected "
                f"{expected_decoder_layers} language decoder layers, "
                f"found {actual_decoder_layers} at {decoder_path}. "
                "Refusing to silently scan a different architecture."
            )

        for layer in layers:
            if not (
                0
                <= int(layer)
                < actual_decoder_layers
            ):
                raise ValueError(
                    f"L{layer} outside decoder range "
                    f"L0-L{actual_decoder_layers - 1}."
                )

        n_heads_by_layer: Dict[
            int,
            int,
        ] = {}
        head_dim_by_layer: Dict[
            int,
            int,
        ] = {}
        hidden_by_layer: Dict[
            int,
            int,
        ] = {}

        for layer in layers:
            attention = (
                attention_helper.resolve_self_attention(
                    decoder_layers[
                        int(layer)
                    ]
                )
            )

            (
                n_heads,
                head_dim,
                hidden_size,
            ) = infer_head_shape(
                attention=attention,
                model=model,
            )

            n_heads_by_layer[
                int(layer)
            ] = int(n_heads)
            head_dim_by_layer[
                int(layer)
            ] = int(head_dim)
            hidden_by_layer[
                int(layer)
            ] = int(hidden_size)

        unique_shapes = sorted(
            {
                (
                    n_heads_by_layer[layer],
                    head_dim_by_layer[layer],
                    hidden_by_layer[layer],
                )
                for layer in layers
            }
        )

        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Non-uniform scanned-layer head shapes: {unique_shapes}"
            )

        (
            actual_query_heads,
            actual_head_dim,
            actual_hidden_size,
        ) = unique_shapes[0]

        expected_query_heads = int(
            model_defaults[
                "expected_query_heads"
            ]
        )

        if (
            actual_query_heads
            != expected_query_heads
        ):
            raise RuntimeError(
                f"{args.model}: expected "
                f"{expected_query_heads} query heads/layer, "
                f"found {actual_query_heads}. "
                "Refusing to scan with an architecture mismatch."
            )

        if explicit_heads:
            scan_heads = []

            for layer, head in explicit_heads:
                if layer not in layers:
                    raise ValueError(
                        f"{hname(layer, head)} outside --layers."
                    )

                if not (
                    0
                    <= head
                    < n_heads_by_layer[layer]
                ):
                    raise ValueError(
                        f"{hname(layer, head)} outside H0-H"
                        f"{n_heads_by_layer[layer] - 1}."
                    )

                scan_heads.append(
                    (
                        int(layer),
                        int(head),
                    )
                )
        else:
            scan_heads = [
                (
                    int(layer),
                    int(head),
                )
                for layer in layers
                for head in range(
                    n_heads_by_layer[layer]
                )
            ]

        relation_token_map = (
            relation_token_variants(
                processor.tokenizer
            )
        )

        print(
            "\n"
            + "=" * 170
        )
        print(
            "MULTI-MODEL NATURAL RECEIVER HEAD SCAN"
        )
        print(
            "=" * 170
        )
        print(
            "model              :",
            args.model,
            f"({model_defaults['description']})",
        )
        print(
            "repo               :",
            spec.repo_id,
        )
        print(
            "decoder path       :",
            decoder_path,
        )
        print(
            "decoder layers     :",
            actual_decoder_layers,
        )
        print(
            "query heads/layer  :",
            actual_query_heads,
        )
        print(
            "head dim           :",
            actual_head_dim,
        )
        print(
            "hidden size        :",
            actual_hidden_size,
        )
        print(
            "scan layers        :",
            f"L{min(layers)}-L{max(layers)}",
            f"({len(layers)} layers)",
        )
        print(
            "N scan heads       :",
            len(scan_heads),
        )
        print(
            "random samples     :",
            len(selected_records),
        )
        print(
            "sample seed        :",
            args.seed,
        )
        print(
            "scale              :",
            args.scale,
        )
        print(
            "prompt             :",
            args.prompt_template,
        )
        print(
            "=" * 170,
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Phase A: clean full-generation baseline + actual decoder geometry.
        # ---------------------------------------------------------------------

        existing_baselines = (
            load_jsonl(
                baseline_jsonl
            )
            if args.resume
            else []
        )

        baseline_by_sid: Dict[
            int,
            Dict[str, Any],
        ] = {
            int(row["sid"]): row
            for row in existing_baselines
        }

        geometry_layer = int(
            layers[0]
        )

        for sample_index, record in enumerate(
            tqdm(
                selected_records,
                desc=f"baseline:{args.model}",
            ),
            start=1,
        ):
            sid = int(
                record.sid
            )

            if sid in baseline_by_sid:
                continue

            image = None
            batch = None

            try:
                gt = normalize_relation(
                    record.relation
                )

                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Unsupported GT relation {record.relation!r}"
                    )

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(
                    record.image_path
                ).convert("RGB")

                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(x)
                    for x in (
                        batch[
                            "input_ids"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ]

                raw_subject_positions = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(record.subject),
                    )
                )
                raw_reference_positions = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(record.reference),
                    )
                )

                (
                    generation_prediction,
                    generation_text,
                    merged_prompt_length,
                ) = clean_generation_with_geometry(
                    model=model,
                    processor=processor,
                    batch=batch,
                    decoder_layers=decoder_layers,
                    attention_helper=attention_helper,
                    geometry_layer=geometry_layer,
                    max_new_tokens=args.max_new_tokens,
                )

                (
                    merged_subject_positions,
                    subject_map,
                ) = map_text_positions_to_decoder(
                    model=model,
                    processor=processor,
                    input_ids=input_ids,
                    raw_positions=raw_subject_positions,
                    merged_length=merged_prompt_length,
                )

                (
                    merged_reference_positions,
                    reference_map,
                ) = map_text_positions_to_decoder(
                    model=model,
                    processor=processor,
                    input_ids=input_ids,
                    raw_positions=raw_reference_positions,
                    merged_length=merged_prompt_length,
                )

                if (
                    subject_map["mapping_mode"]
                    != reference_map["mapping_mode"]
                ):
                    raise RuntimeError(
                        "Subject/reference mapping modes disagree."
                    )

                merged_prompt_last = int(
                    merged_prompt_length - 1
                )

                generation_correct = (
                    generation_prediction == gt
                )

                row = {
                    "sid": sid,
                    "image_id": str(
                        getattr(
                            record,
                            "image_id",
                            "",
                        )
                    ),
                    "subject": str(
                        record.subject
                    ),
                    "reference": str(
                        record.reference
                    ),
                    "gt": gt,
                    "generation_prediction": (
                        generation_prediction
                    ),
                    "generation_text": (
                        generation_text
                    ),
                    "generation_correct": (
                        generation_correct
                    ),
                    "raw_prompt_length": len(
                        input_ids
                    ),
                    "merged_prompt_length": (
                        merged_prompt_length
                    ),
                    "merged_prompt_last": (
                        merged_prompt_last
                    ),
                    "position_mapping_mode": (
                        subject_map[
                            "mapping_mode"
                        ]
                    ),
                    "position_shift": int(
                        subject_map[
                            "shift"
                        ]
                    ),
                    "raw_subject_positions": (
                        raw_subject_positions
                    ),
                    "raw_reference_positions": (
                        raw_reference_positions
                    ),
                    "merged_subject_positions": (
                        merged_subject_positions
                    ),
                    "merged_reference_positions": (
                        merged_reference_positions
                    ),
                    "image_placeholder_positions": (
                        subject_map[
                            "image_placeholders"
                        ]
                    ),
                }

                append_jsonl(
                    baseline_jsonl,
                    row,
                )

                baseline_by_sid[
                    sid
                ] = row

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "baseline",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                del batch
                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index
                    % args.empty_cache_every
                    == 0
                ):
                    torch.cuda.empty_cache()

        baseline_rows = [
            baseline_by_sid[sid]
            for sid in selected_sids
            if sid in baseline_by_sid
        ]

        write_csv(
            baseline_csv,
            baseline_rows,
        )

        if len(
            baseline_rows
        ) != len(
            selected_sids
        ):
            raise RuntimeError(
                f"Baseline completed for {len(baseline_rows)}/"
                f"{len(selected_sids)} selected samples."
            )

        baseline_acc = safe_mean(
            float(
                parse_bool(
                    row[
                        "generation_correct"
                    ]
                )
            )
            for row in baseline_rows
        )

        mapping_counts: Dict[
            str,
            int,
        ] = defaultdict(int)

        for row in baseline_rows:
            mapping_counts[
                str(
                    row[
                        "position_mapping_mode"
                    ]
                )
            ] += 1

        shift_values = [
            int(
                row[
                    "position_shift"
                ]
            )
            for row in baseline_rows
        ]

        print(
            "\nBaseline generation:"
        )
        print(
            f"  ACC = {100 * baseline_acc:.2f}% "
            f"({sum(parse_bool(r['generation_correct']) for r in baseline_rows)}/"
            f"{len(baseline_rows)})"
        )
        print(
            "  position mappings =",
            dict(mapping_counts),
        )
        print(
            "  position shift range =",
            (
                min(shift_values),
                max(shift_values),
            ),
        )

        # ---------------------------------------------------------------------
        # Phase B: sample-major clean trace -> independent single-head patches.
        # ---------------------------------------------------------------------

        prior_patch_rows = (
            load_jsonl(
                patch_jsonl
            )
            if args.resume
            else []
        )

        completed = {
            patch_key(
                sid=int(row["sid"]),
                layer=int(row["layer"]),
                head=int(row["head"]),
                scale=float(row["scale"]),
            )
            for row in prior_patch_rows
        }

        total_conditions = (
            len(baseline_rows)
            * len(scan_heads)
        )

        print(
            "\nHead scan:"
        )
        print(
            f"  expected patched generations = {total_conditions}"
        )
        print(
            f"  already completed = {len(completed)}"
        )

        for sample_index, base_row in enumerate(
            tqdm(
                baseline_rows,
                desc=f"receiver-scan:{args.model}",
            ),
            start=1,
        ):
            sid = int(
                base_row["sid"]
            )

            image = None
            batch = None

            try:
                needed_heads = [
                    (
                        layer,
                        head,
                    )
                    for (
                        layer,
                        head,
                    ) in scan_heads
                    if patch_key(
                        sid=sid,
                        layer=layer,
                        head=head,
                        scale=args.scale,
                    )
                    not in completed
                ]

                if not needed_heads:
                    continue

                record = record_by_sid[
                    sid
                ]

                gt = normalize_relation(
                    base_row["gt"]
                )
                baseline_prediction = (
                    normalize_relation(
                        base_row[
                            "generation_prediction"
                        ]
                    )
                )
                baseline_correct = parse_bool(
                    base_row[
                        "generation_correct"
                    ]
                )

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(
                    record.image_path
                ).convert("RGB")

                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(x)
                    for x in (
                        batch[
                            "input_ids"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ]

                raw_subject_positions = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(record.subject),
                    )
                )
                raw_reference_positions = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(record.reference),
                    )
                )

                merged_prompt_length = int(
                    base_row[
                        "merged_prompt_length"
                    ]
                )
                merged_prompt_last = int(
                    base_row[
                        "merged_prompt_last"
                    ]
                )

                (
                    merged_subject_positions,
                    subject_map,
                ) = map_text_positions_to_decoder(
                    model=model,
                    processor=processor,
                    input_ids=input_ids,
                    raw_positions=raw_subject_positions,
                    merged_length=merged_prompt_length,
                )

                (
                    merged_reference_positions,
                    reference_map,
                ) = map_text_positions_to_decoder(
                    model=model,
                    processor=processor,
                    input_ids=input_ids,
                    raw_positions=raw_reference_positions,
                    merged_length=merged_prompt_length,
                )

                object_positions = sorted(
                    set(
                        map(
                            int,
                            merged_subject_positions
                            + merged_reference_positions,
                        )
                    )
                )

                needed_layers = sorted(
                    {
                        int(layer)
                        for layer, _ in needed_heads
                    }
                )

                (
                    clean_messages,
                    replay_errors,
                ) = extract_clean_messages(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layers=needed_layers,
                    merged_prompt_last=merged_prompt_last,
                    object_positions=object_positions,
                    chunk_size=args.trace_chunk_size,
                )

                for layer in needed_layers:
                    observed_heads = int(
                        clean_messages[
                            layer
                        ].shape[0]
                    )

                    expected_heads = int(
                        n_heads_by_layer[
                            layer
                        ]
                    )

                    if (
                        observed_heads
                        != expected_heads
                    ):
                        raise RuntimeError(
                            f"L{layer}: trace returned {observed_heads} "
                            f"query heads, expected {expected_heads}."
                        )

                for (
                    layer,
                    head,
                ) in needed_heads:
                    message = np.asarray(
                        clean_messages[
                            layer
                        ][
                            head
                        ],
                        dtype=np.float32,
                    )

                    delta = (
                        float(args.scale)
                        * message
                    )

                    (
                        patched_prediction,
                        patched_text,
                    ) = patched_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        layer=layer,
                        merged_prompt_length=merged_prompt_length,
                        merged_prompt_last=merged_prompt_last,
                        delta=delta,
                        max_new_tokens=args.max_new_tokens,
                    )

                    patched_correct = (
                        patched_prediction
                        == gt
                    )

                    row = {
                        "sid": sid,
                        "layer": int(
                            layer
                        ),
                        "head": int(
                            head
                        ),
                        "head_name": hname(
                            layer,
                            head,
                        ),
                        "scale": float(
                            args.scale
                        ),
                        "gt": gt,
                        "baseline_generation_prediction": (
                            baseline_prediction
                        ),
                        "baseline_generation_correct": (
                            baseline_correct
                        ),
                        "patched_generation_prediction": (
                            patched_prediction
                        ),
                        "patched_generation_text": (
                            patched_text
                        ),
                        "patched_generation_correct": (
                            patched_correct
                        ),
                        "wrong_to_correct": (
                            (not baseline_correct)
                            and patched_correct
                        ),
                        "correct_to_wrong": (
                            baseline_correct
                            and (
                                not patched_correct
                            )
                        ),
                        "generation_changed": (
                            patched_prediction
                            != baseline_prediction
                        ),
                        "message_norm": float(
                            np.linalg.norm(
                                message
                            )
                        ),
                        "delta_norm": float(
                            np.linalg.norm(
                                delta
                            )
                        ),
                        "replay_relative_error": float(
                            replay_errors[
                                layer
                            ]
                        ),
                        "raw_prompt_length": len(
                            input_ids
                        ),
                        "merged_prompt_length": (
                            merged_prompt_length
                        ),
                        "position_mapping_mode": (
                            subject_map[
                                "mapping_mode"
                            ]
                        ),
                        "position_shift": (
                            subject_map[
                                "shift"
                            ]
                        ),
                        "N_object_positions": len(
                            object_positions
                        ),
                    }

                    append_jsonl(
                        patch_jsonl,
                        row,
                    )

                    prior_patch_rows.append(
                        row
                    )

                    completed.add(
                        patch_key(
                            sid=sid,
                            layer=layer,
                            head=head,
                            scale=args.scale,
                        )
                    )

                del clean_messages

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "head_scan",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                del batch

                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index
                    % args.empty_cache_every
                    == 0
                ):
                    torch.cuda.empty_cache()

        patch_rows = load_jsonl(
            patch_jsonl
        )

        write_csv(
            output_dir
            / "patch_results.csv",
            patch_rows,
        )

        head_summary = summarize_heads(
            baseline_rows=baseline_rows,
            patch_rows=patch_rows,
        )

        write_csv(
            output_dir
            / "single_head_summary.csv",
            head_summary,
        )

        layer_summary = summarize_layers(
            head_summary
        )

        write_csv(
            output_dir
            / "layer_summary.csv",
            layer_summary,
        )

        complete_heads = [
            row
            for row in head_summary
            if bool(
                row["complete"]
            )
        ]

        relation_rows = relation_summary(
            baseline_rows=baseline_rows,
            patch_rows=patch_rows,
            top_heads=complete_heads[
                :30
            ],
        )

        write_csv(
            output_dir
            / "relation_summary.csv",
            relation_rows,
        )

        print(
            "\n"
            + "=" * 176
        )
        print(
            f"{args.model.upper()} RECEIVER HEAD SCAN SUMMARY"
        )
        print(
            "=" * 176
        )
        print(
            f"Baseline full-generation ACC: "
            f"{100 * baseline_acc:.2f}% | N={len(baseline_rows)}"
        )
        print()
        print(
            f"  {'rank':>4s} "
            f"{'head':<9s} "
            f"{'scale':>5s} "
            f"{'N':>7s} "
            f"{'ACC':>8s} "
            f"{'delta':>9s} "
            f"{'W->C':>5s} "
            f"{'C->W':>5s} "
            f"{'net':>5s} "
            f"{'chg':>7s} "
            f"{'replay':>9s}"
        )

        for rank, row in enumerate(
            head_summary[:40],
            start=1,
        ):
            print(
                f"  {rank:>4d} "
                f"{str(row['head_name']):<9s} "
                f"{float(row['scale']):>5.1f} "
                f"{int(row['N_completed']):>3d}/"
                f"{int(row['N_expected']):<3d} "
                f"{100 * float(row['patched_acc']):>7.2f}% "
                f"{100 * float(row['delta_acc']):>+8.2f} "
                f"{int(row['wrong_to_correct']):>5d} "
                f"{int(row['correct_to_wrong']):>5d} "
                f"{int(row['net_repairs']):>+5d} "
                f"{100 * float(row['generation_changed_rate']):>6.2f}% "
                f"{float(row['mean_replay_relative_error']):>9.3e}"
            )

        print(
            "\nBest head per layer:"
        )

        for row in layer_summary:
            print(
                f"  L{int(row['layer']):02d} "
                f"best={str(row['best_head']):<9s} "
                f"ACC={100 * float(row['best_patched_acc']):6.2f}% "
                f"delta={100 * float(row['best_delta_acc']):+6.2f}pp "
                f"W->C={int(row['best_wrong_to_correct']):3d} "
                f"C->W={int(row['best_correct_to_wrong']):3d} "
                f"net={int(row['best_net_repairs']):+3d} "
                f"positive={int(row['positive_net_heads']):2d}/"
                f"{int(row['N_heads']):2d}"
            )

        print(
            "=" * 176
        )

        config = {
            "script_version": SCRIPT_VERSION,
            "model_alias": args.model,
            "model_description": model_defaults[
                "description"
            ],
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "data_root": args.data_root,
            "sample_seed": args.seed,
            "num_samples": len(
                selected_sids
            ),
            "selected_sids": selected_sids,
            "decoder_path": str(
                decoder_path
            ),
            "decoder_layers": (
                actual_decoder_layers
            ),
            "query_heads_per_layer": (
                actual_query_heads
            ),
            "head_dim": (
                actual_head_dim
            ),
            "hidden_size": (
                actual_hidden_size
            ),
            "scan_layers": layers,
            "scan_head_count": len(
                scan_heads
            ),
            "scan_heads": [
                hname(
                    layer,
                    head,
                )
                for layer, head in scan_heads
            ],
            "scale": args.scale,
            "prompt_template": (
                args.prompt_template
            ),
            "baseline_acc": (
                baseline_acc
            ),
            "mapping_counts": dict(
                mapping_counts
            ),
            "position_shift_min": min(
                shift_values
            ),
            "position_shift_max": max(
                shift_values
            ),
            "uses_gt_for_intervention": False,
            "metric": (
                "full greedy model.generate() "
                "relation accuracy"
            ),
            "patch_formula": (
                "attention_out_L[decoder_prompt_last] += "
                "scale * clean_sample_specific_object_to_last_post_WO_head_message"
            ),
            "dataset_audit": audit,
        }

        (
            output_dir
            / "config.json"
        ).write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        report_lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"repo: {spec.repo_id}",
            f"decoder_layers: {actual_decoder_layers}",
            f"query_heads_per_layer: {actual_query_heads}",
            f"head_dim: {actual_head_dim}",
            f"hidden_size: {actual_hidden_size}",
            f"scan_layers: {layers}",
            f"N random samples: {len(baseline_rows)}",
            f"seed: {args.seed}",
            f"scale: {args.scale}",
            f"baseline full-generation ACC: {100 * baseline_acc:.2f}%",
            f"position mappings: {dict(mapping_counts)}",
            "",
            "TOP SINGLE HEADS",
        ]

        for rank, row in enumerate(
            head_summary[:40],
            start=1,
        ):
            report_lines.append(
                f"{rank:02d} {row['head_name']} "
                f"ACC={100 * float(row['patched_acc']):.2f}% "
                f"delta={100 * float(row['delta_acc']):+.2f}pp "
                f"W->C={int(row['wrong_to_correct'])} "
                f"C->W={int(row['correct_to_wrong'])} "
                f"net={int(row['net_repairs']):+d} "
                f"complete={bool(row['complete'])}"
            )

        report_lines += [
            "",
            "INTERPRETATION",
            (
                "A strong positive head means that increasing this head's own "
                "clean sample-specific object-text -> prompt-last message causally "
                "improves final autoregressive relation generation on this sample set."
            ),
            (
                "The head identity is discovered independently for this model; "
                "Qwen-3B L26H04/H06 are not assumed or reused."
            ),
            (
                "This is an exploratory 200-sample head-selection scan. Freeze "
                "selected heads/scales and test on fresh held-out samples before "
                "reporting an unbiased final ACC improvement."
            ),
        ]

        (
            output_dir
            / "report.txt"
        ).write_text(
            "\n".join(
                report_lines
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "\nSaved:"
        )

        for filename in (
            "selected_sids.json",
            "baseline_eval.jsonl",
            "baseline_eval.csv",
            "patch_results.jsonl",
            "patch_results.csv",
            "single_head_summary.csv",
            "layer_summary.csv",
            "relation_summary.csv",
            "config.json",
            "report.txt",
        ):
            print(
                " ",
                output_dir
                / filename,
            )

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
