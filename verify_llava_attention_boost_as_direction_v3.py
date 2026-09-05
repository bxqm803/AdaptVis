#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_llava_attention_boost_as_direction_v3.py

Verify whether LLaVA early last-token -> visual-token attention enhancement
acts primarily as a model-generated spatial steering vector.

This script is designed for:
    https://github.com/bxqm803/AdaptVis/tree/llava16

and reuses the repository's own LLaVA ScalingVis / AdaptVis implementation:
    model_zoo/llava/modeling_llava_scal.py
    model_zoo/llama/modeling_llama_add_attn.py

========================================================================
HYPOTHESIS
========================================================================

The existing attention intervention modifies the LAST query row's attention
to visual tokens. Therefore, at each decoder layer l it changes the last-token
residual state.

Define:

    delta_boost[i,l]
        = h_boost[i,last,l] - h_base[i,last,l]

If the attention intervention is effectively producing a spatial steering
vector from the current image, then:

(1) delta_boost itself should become relation-decodable:
        left / right / above / below

(2) spatial information in the last-token visual residual should appear
    earlier / become stronger under boost:

        base_residual[i,l]
          = h_base_real[i,last,l] - h_noimage[i,last,l]

        boost_residual[i,l]
          = h_boost_real[i,last,l] - h_noimage[i,last,l]

(3) If boost is restricted to layers [0 ... B], then after block B the only
    changed causal state should be the last-token residual (under the current
    AdaptVis implementation, only the last query row is edited).

    Therefore, replaying the SAME sample-specific boundary delta:

        h_base[last,B] += delta_boost[i,B]

    while disabling attention boost should reproduce the boosted behavior.

(4) Stronger diagnostic:
    Fit relation-specific mean boost deltas on TRAIN only:

        mu_r[B] = E_train[delta_boost[i,B] | y_i=r]

    On TEST, use GT only as an ORACLE diagnostic:

        h_base[last,B] += mu_GT[B]

    If this also improves strongly, then the boost delta contains a stable
    relation-specific causal component, rather than being only arbitrary
    sample-specific visual content.

========================================================================
OUTPUT
========================================================================

direction_probe_summary.csv
    Layer-wise TRAIN/TEST direction readout for:
      - base_residual
      - boost_residual
      - boost_delta

    Main columns:
      representation
      layer
      accuracy_mean / std
      L/R/A/B accuracy

generation_summary.csv                 [when --run-generation]
generation_details.csv
    baseline vs attention boost vs exact-delta replay.

oracle_mean_delta_summary.csv          [when --run-oracle-mean-replay]
oracle_mean_delta_details.csv
    TRAIN mean relation delta, GT-routed on TEST only.

vectors.npz                            [optional --save-vectors]

========================================================================
IMPORTANT
========================================================================

This is a mechanism diagnostic.

- The relation probes use TRAIN labels only to fit relation means.
- TEST labels are never used to construct the probe.
- The exact-sample replay is not a deployable method; it tests mechanistic
  equivalence between attention modification and additive activation steering.
- The GT mean-delta replay is explicitly an oracle diagnostic.
- The same attention implementation used by AdaptVis is preserved.

========================================================================
EXAMPLE: LLaVA-1.5-7B / COCO-two
========================================================================

First run the representation diagnostic only:

CUDA_VISIBLE_DEVICES=0 python verify_llava_attention_boost_as_direction_v3.py \
  --model llava-7b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --boost-layers 0-7 \
  --boost-weight 1.2 \
  --attention-variant mul_img \
  --train-ratio 0.30 \
  --repeats 5 \
  --output-dir output/llava7b_boost_as_direction_v1 \
  --overwrite

Then causal replay:

CUDA_VISIBLE_DEVICES=0 python verify_llava_attention_boost_as_direction_v3.py \
  --model llava-7b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --boost-layers 0-7 \
  --boost-weight 1.2 \
  --attention-variant mul_img \
  --train-ratio 0.30 \
  --repeats 5 \
  --run-generation \
  --run-oracle-mean-replay \
  --max-new-tokens 8 \
  --output-dir output/llava7b_boost_as_direction_v1 \
  --overwrite

If your successful 65 -> 80 experiment used another AdaptVis variant,
match it exactly, e.g.:

    --attention-variant prob_img --boost-weight 1.5

or:

    --attention-variant add_img --boost-weight 0.5

Do NOT compare different boost definitions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import extract_two_object_relation_states as data_helpers

from model_zoo.llava.modeling_llava_scal import (
    LlavaForConditionalGenerationScal,
)


RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12

MODEL_REPOS = {
    "llava-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-13b": "llava-hf/llava-1.5-13b-hf",
}


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        default="llava-7b",
        choices=sorted(MODEL_REPOS),
    )
    p.add_argument(
        "--dataset",
        default="coco_two",
        choices=["coco_two", "vg_two"],
    )
    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
        help=(
            "Use the exact prompt file from the successful attention-boost run. "
            "Rows are matched by id/sid."
        ),
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--revision",
        default="a272c74",
        help=(
            "Repository llava15.py uses a272c74. "
            "Pass empty string to use the model repo default revision."
        ),
    )

    p.add_argument(
        "--boost-layers",
        default="0-7",
        help="Decoder blocks whose last-query visual attention is modified.",
    )
    p.add_argument(
        "--boost-weight",
        type=float,
        default=1.2,
    )
    p.add_argument(
        "--attention-variant",
        default="mul_img",
        choices=[
            "mul_img",
            "add_img",
            "center_img",
            "prob_img",
            "clip_img",
            "tanh_img",
            "softsign_img",
        ],
    )

    p.add_argument(
        "--probe-layers",
        default="all",
        help=(
            "Layers at which last-token states are saved and probed. "
            "'all', 'auto', '0-12', or comma list."
        ),
    )

    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke-test cap after record loading.",
    )

    p.add_argument(
        "--run-generation",
        action="store_true",
        help=(
            "Run baseline, boosted attention, and exact boundary-delta replay "
            "with actual model.generate()."
        ),
    )
    p.add_argument(
        "--run-oracle-mean-replay",
        action="store_true",
        help=(
            "On split-0 TEST only: inject TRAIN relation-mean boost delta "
            "selected by TEST GT. Oracle diagnostic."
        ),
    )

    p.add_argument(
        "--generation-max-samples",
        type=int,
        default=None,
        help="Optional cap for expensive generation replay.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument(
        "--mean-delta-scale",
        type=float,
        default=1.0,
        help="Scale for oracle TRAIN mean boost-delta replay.",
    )

    p.add_argument("--save-vectors", action="store_true")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


# ============================================================================
# Generic utilities
# ============================================================================

def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(values: Iterable[float]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: Iterable[float]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.std(vals)) if vals else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_relation(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z]+", " ", text)

    tokens = text.split()

    for token in tokens[:12]:
        if token in REL_TO_ID:
            return token

    # Handle natural variants conservatively.
    if "to the left" in text or " left " in f" {text} ":
        return "left"
    if "to the right" in text or " right " in f" {text} ":
        return "right"
    if "above" in text or "top" in tokens:
        return "above"
    if (
        "below" in text
        or "under" in tokens
        or "underneath" in tokens
        or "bottom" in tokens
    ):
        return "below"

    return None


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    raw = str(text).strip().lower()

    if raw == "all":
        return list(range(n_layers))

    if raw == "auto":
        candidates = [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 10, 12, 16, 20, 24, 28,
            n_layers - 1,
        ]
        return sorted({
            x for x in candidates
            if 0 <= x < n_layers
        })

    result: List[int] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            result.extend(range(a, b + step, step))
        else:
            result.append(int(part))

    result = list(dict.fromkeys(result))

    bad = [
        x for x in result
        if x < 0 or x >= n_layers
    ]
    if bad:
        raise ValueError(
            f"Invalid layer indices={bad}; decoder has 0..{n_layers-1}"
        )

    if not result:
        raise ValueError("No layers selected.")

    return result


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(denom, EPS)


def replace_first_tensor(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden

    if isinstance(output, tuple):
        values = list(output)
        for i, item in enumerate(values):
            if torch.is_tensor(item) and item.ndim == 3:
                values[i] = hidden
                return tuple(values)
        raise RuntimeError("No rank-3 tensor in tuple layer output.")

    if isinstance(output, list):
        values = list(output)
        for i, item in enumerate(values):
            if torch.is_tensor(item) and item.ndim == 3:
                values[i] = hidden
                return values
        raise RuntimeError("No rank-3 tensor in list layer output.")

    raise TypeError(
        f"Unsupported layer output type={type(output)}"
    )


def first_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise RuntimeError(
        f"Could not extract layer hidden state from {type(output)}"
    )


# ============================================================================
# Prompt loading
# ============================================================================

def load_prompt_map(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    result: Dict[int, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            sid = int(
                row.get("id", line_idx)
            )

            result[sid] = row

    return result


def prompt_for_record(
    rec: Any,
    prompt_map: Mapping[int, Mapping[str, Any]],
) -> str:
    row = prompt_map.get(
        int(rec.sid)
    )

    if row is None:
        raise KeyError(
            f"sid={rec.sid} missing from prompt JSONL."
        )

    question = str(
        row["question"]
    )

    return question


def noimage_prompt(prompt: str) -> str:
    # Remove only the image placeholder. Keep the same user/assistant wording.
    text = prompt.replace("<image>", "")
    text = re.sub(r"^\s*\n", "", text)
    return text


# ============================================================================
# Model resolution / loading
# ============================================================================

def resolve_decoder_layers(model: Any) -> Sequence[torch.nn.Module]:
    candidates = [
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
    ]

    for path in candidates:
        obj = model

        okay = True

        for part in path.split("."):
            if not hasattr(obj, part):
                okay = False
                break
            obj = getattr(obj, part)

        if okay and isinstance(
            obj,
            (torch.nn.ModuleList, list, tuple),
        ):
            return obj

    raise RuntimeError(
        "Could not resolve LLaVA decoder layers."
    )


def load_model_and_processor(
    args: argparse.Namespace,
):
    repo_id = MODEL_REPOS[
        args.model
    ]

    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "ignore_mismatched_sizes": True,
    }

    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    if str(args.revision).strip():
        kwargs["revision"] = str(args.revision).strip()

    model = (
        LlavaForConditionalGenerationScal
        .from_pretrained(
            repo_id,
            **kwargs,
        )
        .eval()
        .to(args.device)
    )

    processor_kwargs: Dict[str, Any] = {}

    if args.cache_dir:
        processor_kwargs[
            "cache_dir"
        ] = args.cache_dir

    if str(args.revision).strip():
        processor_kwargs[
            "revision"
        ] = str(args.revision).strip()

    processor = AutoProcessor.from_pretrained(
        repo_id,
        **processor_kwargs,
    )

    # IMPORTANT:
    # Do NOT call data_helpers.configure_processor() here.
    #
    # This repository's custom LlavaForConditionalGenerationScal uses the
    # legacy merge path:
    #     ONE <image> placeholder -> model internally expands to 576 patches.
    #
    # Recent LlavaProcessor versions may pre-expand <image> into ~576
    # placeholders when patch_size / vision_feature_select_strategy are set.
    # That is incompatible with the custom legacy merge.
    #
    # We therefore never use processor(text=..., images=...) below. Text and
    # image preprocessing are performed separately with processor.tokenizer
    # and processor.image_processor.
    return model, processor


# ============================================================================
# Restrict repository AdaptVis intervention to chosen layer window
# ============================================================================

class RestrictBoostLayers:
    """
    The repository's custom attention receives:
        weight, idx, keys, adjust_method

    and by default applies AdaptVis when idx < 32.

    This pre-hook keeps the implementation intact, but forces weight=None for
    layers outside --boost-layers.
    """

    def __init__(
        self,
        decoder_layers: Sequence[torch.nn.Module],
        enabled_layers: Sequence[int],
    ):
        self.enabled = set(
            int(x)
            for x in enabled_layers
        )

        self.handles = []

        for layer in decoder_layers:
            attn = getattr(
                layer,
                "self_attn",
                None,
            )

            if attn is None:
                raise RuntimeError(
                    "Decoder layer has no self_attn."
                )

            self.handles.append(
                attn.register_forward_pre_hook(
                    self._hook,
                    with_kwargs=True,
                )
            )

    def _hook(
        self,
        module,
        args,
        kwargs,
    ):
        idx = kwargs.get(
            "idx",
            None,
        )

        if idx is not None and int(idx) not in self.enabled:
            kwargs = dict(kwargs)
            kwargs["weight"] = None

        return args, kwargs

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ============================================================================
# Batch construction / forwards
# ============================================================================

def move_batch(
    batch: Mapping[str, Any],
    device: str,
) -> Dict[str, Any]:
    return {
        key: (
            value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def _tokenize_text_only(
    processor: Any,
    text: str,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize WITHOUT invoking LlavaProcessor.__call__.

    This is intentional: recent LlavaProcessor versions can expand one
    <image> marker into hundreds of image placeholders, while this repository's
    custom LlavaForConditionalGenerationScal expects exactly ONE placeholder.
    """
    tok = processor.tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )

    result: Dict[str, torch.Tensor] = {
        "input_ids": tok["input_ids"],
    }

    if "attention_mask" in tok:
        result["attention_mask"] = tok["attention_mask"]
    else:
        result["attention_mask"] = torch.ones_like(
            tok["input_ids"],
            dtype=torch.long,
        )

    return result


def _process_image_only(
    processor: Any,
    image: Image.Image,
) -> Dict[str, torch.Tensor]:
    """
    Image preprocessing WITHOUT any text/image placeholder manipulation.
    """
    image_processor = getattr(
        processor,
        "image_processor",
        None,
    )

    if image_processor is None:
        raise RuntimeError(
            "AutoProcessor has no image_processor; cannot build legacy "
            "LLaVA input safely."
        )

    img = image_processor(
        images=image,
        return_tensors="pt",
    )

    if "pixel_values" not in img:
        raise RuntimeError(
            f"image_processor returned keys={list(img.keys())}, "
            "but no pixel_values."
        )

    return {
        "pixel_values": img["pixel_values"],
    }


def build_real_batch(
    processor: Any,
    prompt: str,
    image: Image.Image,
    device: str,
    image_token_index: int,
) -> Dict[str, Any]:
    """
    Legacy-compatible input for this repository's custom LLaVA model.

    Critical invariant:
        exactly ONE image-token placeholder must be present in input_ids.

    The custom model then internally expands that single token into the 576
    projected visual tokens.
    """
    text_batch = _tokenize_text_only(
        processor,
        prompt,
    )

    image_batch = _process_image_only(
        processor,
        image,
    )

    input_ids = text_batch["input_ids"]

    count = int(
        (input_ids == int(image_token_index))
        .sum()
        .item()
    )

    if count != 1:
        # Helpful diagnostics for the exact failure that caused v1/v2 to break.
        image_token = getattr(
            processor,
            "image_token",
            "<image>",
        )

        raw_marker_count = str(prompt).count(
            str(image_token)
        )

        raise RuntimeError(
            "Legacy LlavaForConditionalGenerationScal requires exactly ONE "
            f"image placeholder token before model-side merge, but got {count}. "
            f"model.config.image_token_index={image_token_index}; "
            f"raw prompt marker count={raw_marker_count}; "
            f"input_len={input_ids.shape[1]}. "
            "Do not pass this prompt through LlavaProcessor multimodal "
            "placeholder expansion."
        )

    batch: Dict[str, Any] = {
        **text_batch,
        **image_batch,
    }

    return move_batch(
        batch,
        device,
    )


def build_noimage_batch(
    processor: Any,
    prompt: str,
    device: str,
) -> Dict[str, Any]:
    text = noimage_prompt(
        prompt
    )

    batch = _tokenize_text_only(
        processor,
        text,
    )

    # Sanity check: no-image branch must contain no image placeholder.
    image_token_id = getattr(
        processor.tokenizer,
        "image_token_id",
        None,
    )

    if image_token_id is None:
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                "<image>"
            )
        except Exception:
            image_token_id = None

    if image_token_id is not None:
        count = int(
            (
                batch["input_ids"]
                == int(image_token_id)
            )
            .sum()
            .item()
        )

        if count != 0:
            raise RuntimeError(
                f"NoImage branch still contains {count} image tokens."
            )

    return move_batch(
        batch,
        device,
    )


def hidden_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    states = getattr(
        outputs,
        "hidden_states",
        None,
    )

    if (
        not isinstance(
            states,
            (tuple, list),
        )
        or not states
    ):
        raise RuntimeError(
            "Model did not return hidden_states."
        )

    return tuple(states)


def forward_real(
    model: Any,
    batch: Mapping[str, Any],
    *,
    weight: Optional[float],
    variant: str,
) -> Any:
    with torch.inference_mode():
        return model(
            **batch,
            weight=weight,
            adjust_method=variant,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )


def forward_noimage(
    model: Any,
    batch: Mapping[str, Any],
) -> Any:
    with torch.inference_mode():
        return model(
            **batch,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )


def last_token_by_layer(
    outputs: Any,
    selected_layers: Sequence[int],
) -> Dict[int, np.ndarray]:
    states = hidden_tuple(
        outputs
    )

    # Custom LLaMA output convention:
    # states[0] = embedding input
    # states[k+1] = output of block k for early/intermediate k.
    # We avoid treating the final normalized state as raw last-block output
    # differently; for the diagnostic that distinction is okay if consistent.
    result: Dict[int, np.ndarray] = {}

    for layer in selected_layers:
        idx = layer + 1

        if idx >= len(states):
            raise RuntimeError(
                f"L{layer}: hidden_states has only {len(states)} entries."
            )

        h = states[idx]

        if h.ndim != 3 or int(h.shape[0]) != 1:
            raise RuntimeError(
                f"L{layer}: bad hidden shape={tuple(h.shape)}"
            )

        result[layer] = (
            h[0, -1]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return result


# ============================================================================
# Relation-direction probe
# ============================================================================

def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    train: List[int] = []
    test: List[int] = []

    for relation in RELATIONS:
        indices = np.flatnonzero(
            labels == relation
        ).tolist()

        rng.shuffle(indices)

        if len(indices) < 2:
            raise RuntimeError(
                f"Need >=2 samples for relation={relation}"
            )

        n_train = int(
            round(
                len(indices)
                * train_ratio
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(indices) - 1,
            ),
        )

        train.extend(
            indices[:n_train]
        )

        test.extend(
            indices[n_train:]
        )

    rng.shuffle(train)
    rng.shuffle(test)

    return (
        np.asarray(
            train,
            dtype=np.int64,
        ),
        np.asarray(
            test,
            dtype=np.int64,
        ),
    )


def fit_relation_directions(
    X: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same simple centered relation-direction codebook:
        center = mean(X_train)
        d_r = normalize(mean(X_train[y=r] - center))
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    center = X.mean(
        axis=0
    )

    Xc = X - center

    directions = []

    for relation in RELATIONS:
        mask = labels == relation

        if not bool(
            mask.any()
        ):
            raise RuntimeError(
                f"No TRAIN samples for {relation}"
            )

        direction = Xc[
            mask
        ].mean(
            axis=0
        )

        norm = float(
            np.linalg.norm(
                direction
            )
        )

        direction = (
            direction
            / max(norm, EPS)
        )

        directions.append(
            direction
        )

    return (
        center,
        np.stack(
            directions,
            axis=0,
        ).astype(np.float32),
    )


def evaluate_direction_probe(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    center, directions = (
        fit_relation_directions(
            X[train_idx],
            labels[train_idx],
        )
    )

    Xt = normalize_rows(
        X[test_idx]
        - center
    )

    scores = Xt @ directions.T

    pred = np.argmax(
        scores,
        axis=1,
    )

    gt = np.asarray(
        [
            REL_TO_ID[
                str(x)
            ]
            for x in labels[
                test_idx
            ]
        ],
        dtype=np.int64,
    )

    result: Dict[str, Any] = {
        "accuracy": float(
            np.mean(
                pred == gt
            )
        ),
        "n_test": len(
            test_idx
        ),
    }

    for relation, rid in REL_TO_ID.items():
        mask = gt == rid

        result[
            f"{relation}_accuracy"
        ] = (
            float(
                np.mean(
                    pred[mask]
                    == gt[mask]
                )
            )
            if bool(mask.any())
            else float("nan")
        )

    return result


def run_all_direction_probes(
    *,
    base_residual: np.ndarray,
    boost_residual: np.ndarray,
    boost_delta: np.ndarray,
    labels: np.ndarray,
    layers: Sequence[int],
    train_ratio: float,
    repeats: int,
    seed: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Inputs shape:
        [N, L, D]
    """

    representations = {
        "base_residual": base_residual,
        "boost_residual": boost_residual,
        "boost_delta": boost_delta,
    }

    repeat_rows: List[
        Dict[str, Any]
    ] = []

    for rep in range(repeats):
        train_idx, test_idx = (
            stratified_split(
                labels,
                train_ratio,
                seed + rep,
            )
        )

        for rep_name, tensor in representations.items():
            for li, layer in enumerate(layers):
                metrics = (
                    evaluate_direction_probe(
                        tensor[:, li, :],
                        labels,
                        train_idx,
                        test_idx,
                    )
                )

                repeat_rows.append({
                    "repeat": rep,
                    "representation": rep_name,
                    "layer": layer,
                    "train_N": len(train_idx),
                    "test_N": len(test_idx),
                    **metrics,
                })

    summary_rows: List[
        Dict[str, Any]
    ] = []

    for rep_name in representations:
        for layer in layers:
            rows = [
                row
                for row in repeat_rows
                if (
                    row["representation"]
                    == rep_name
                    and int(row["layer"])
                    == int(layer)
                )
            ]

            summary_rows.append({
                "representation": rep_name,
                "layer": layer,
                "accuracy_mean": safe_mean(
                    row["accuracy"]
                    for row in rows
                ),
                "accuracy_std": safe_std(
                    row["accuracy"]
                    for row in rows
                ),
                "left_accuracy": safe_mean(
                    row["left_accuracy"]
                    for row in rows
                ),
                "right_accuracy": safe_mean(
                    row["right_accuracy"]
                    for row in rows
                ),
                "above_accuracy": safe_mean(
                    row["above_accuracy"]
                    for row in rows
                ),
                "below_accuracy": safe_mean(
                    row["below_accuracy"]
                    for row in rows
                ),
            })

    return repeat_rows, summary_rows


# ============================================================================
# Generation patch from repository
# ============================================================================

def install_repository_weighted_generation():
    """
    The current AdaptVis llava15 wrapper monkey-patches GenerationMixin so that
    the custom generation path forwards `weight`, `adjust_method`, and `pos`
    to LlavaForConditionalGenerationScal.

    Reuse it instead of inventing a different generation implementation.
    """
    from model_zoo.llava15 import (
        change_greedy_to_add_weight,
    )

    change_greedy_to_add_weight()


def decode_generated(
    processor: Any,
    output: Any,
    prompt_len: int,
) -> str:
    sequences = getattr(
        output,
        "sequences",
        None,
    )

    if sequences is None and isinstance(
        output,
        Mapping,
    ):
        sequences = output.get(
            "sequences",
            None,
        )

    if sequences is None:
        if torch.is_tensor(output):
            sequences = output
        else:
            raise RuntimeError(
                "Generation returned no sequences."
            )

    return (
        processor.decode(
            sequences[
                0,
                int(prompt_len):,
            ],
            skip_special_tokens=True,
        )
        .strip()
    )


def generate_once(
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    boost: bool,
) -> Tuple[str, Optional[str]]:
    kwargs: Dict[str, Any] = {
        **batch,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    if boost:
        kwargs["weight"] = args.boost_weight
        kwargs[
            "adjust_method"
        ] = args.attention_variant

    with torch.inference_mode():
        output = model.generate(
            **kwargs
        )

    text = decode_generated(
        processor,
        output,
        prompt_len=int(
            batch["input_ids"].shape[1]
        ),
    )

    return (
        text,
        normalize_relation(text),
    )


# ============================================================================
# Boundary replay hook
# ============================================================================

class AddLastTokenDelta:
    """
    Add one vector to the last token at decoder block output.

    Apply only once, on the initial full-prompt pass (sequence length > 1).
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        delta: np.ndarray,
        scale: float = 1.0,
    ):
        self.delta = np.asarray(
            delta,
            dtype=np.float32,
        )

        self.scale = float(
            scale
        )

        self.applied = False

        self.handle = (
            layer.register_forward_hook(
                self._hook
            )
        )

    def _hook(
        self,
        module,
        inputs,
        output,
    ):
        if self.applied:
            return output

        hidden = first_hidden(
            output
        )

        # Only the initial full prompt; skip incremental q_len=1 generation.
        if (
            hidden.ndim != 3
            or int(hidden.shape[1]) <= 1
        ):
            return output

        edited = hidden.clone()

        delta = torch.as_tensor(
            self.delta,
            device=edited.device,
            dtype=edited.dtype,
        )

        if int(
            delta.numel()
        ) != int(
            edited.shape[-1]
        ):
            raise RuntimeError(
                f"Replay delta dim={delta.numel()} "
                f"!= hidden={edited.shape[-1]}"
            )

        edited[
            0,
            -1,
            :
        ] = (
            edited[
                0,
                -1,
                :
            ]
            + self.scale
            * delta
        )

        self.applied = True

        return replace_first_tensor(
            output,
            edited,
        )

    def close(self):
        with contextlib.suppress(
            Exception
        ):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ============================================================================
# Extract baseline / boost / no-image last-token states
# ============================================================================

def extract_vectors(
    *,
    model: Any,
    processor: Any,
    records: Sequence[Any],
    prompt_map: Mapping[int, Mapping[str, Any]],
    probe_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
    List[str],
    List[Dict[str, Any]],
]:
    """
    Returns:
        base_last      [N,L,D]
        boost_last     [N,L,D]
        noimage_last   [N,L,D]
        boost_delta    [N,L,D]
        saved_sids
        saved_labels
        extraction_rows
    """

    base_rows: List[
        np.ndarray
    ] = []

    boost_rows: List[
        np.ndarray
    ] = []

    noimage_rows: List[
        np.ndarray
    ] = []

    saved_sids: List[int] = []
    saved_labels: List[str] = []

    diagnostics: List[
        Dict[str, Any]
    ] = []

    for rec in tqdm(
        records,
        desc="extract base/boost/noimage last states",
    ):
        image = None
        real_batch = None
        no_batch = None

        try:
            relation = normalize_relation(
                rec.relation
            )

            if relation not in REL_TO_ID:
                continue

            prompt = prompt_for_record(
                rec,
                prompt_map,
            )

            image = Image.open(
                rec.image_path
            ).convert("RGB")

            real_batch = build_real_batch(
                processor,
                prompt,
                image,
                args.device,
                int(model.config.image_token_index),
            )

            no_batch = build_noimage_batch(
                processor,
                prompt,
                args.device,
            )

            base_out = forward_real(
                model,
                real_batch,
                weight=None,
                variant=args.attention_variant,
            )

            base_state = (
                last_token_by_layer(
                    base_out,
                    probe_layers,
                )
            )

            boost_out = forward_real(
                model,
                real_batch,
                weight=args.boost_weight,
                variant=args.attention_variant,
            )

            boost_state = (
                last_token_by_layer(
                    boost_out,
                    probe_layers,
                )
            )

            no_out = forward_noimage(
                model,
                no_batch,
            )

            no_state = (
                last_token_by_layer(
                    no_out,
                    probe_layers,
                )
            )

            base_arr = np.stack(
                [
                    base_state[layer]
                    for layer in probe_layers
                ],
                axis=0,
            )

            boost_arr = np.stack(
                [
                    boost_state[layer]
                    for layer in probe_layers
                ],
                axis=0,
            )

            no_arr = np.stack(
                [
                    no_state[layer]
                    for layer in probe_layers
                ],
                axis=0,
            )

            delta_arr = (
                boost_arr
                - base_arr
            )

            base_rows.append(
                base_arr.astype(
                    np.float32
                )
            )

            boost_rows.append(
                boost_arr.astype(
                    np.float32
                )
            )

            noimage_rows.append(
                no_arr.astype(
                    np.float32
                )
            )

            saved_sids.append(
                int(rec.sid)
            )

            saved_labels.append(
                relation
            )

            diagnostics.append({
                "sid": int(rec.sid),
                "relation": relation,
                "base_last_norm_final_probe": float(
                    np.linalg.norm(
                        base_arr[-1]
                    )
                ),
                "boost_last_norm_final_probe": float(
                    np.linalg.norm(
                        boost_arr[-1]
                    )
                ),
                "boost_delta_norm_boundary_candidate": float(
                    np.linalg.norm(
                        delta_arr[-1]
                    )
                ),
            })

            del base_out
            del boost_out
            del no_out

        except Exception as exc:
            tqdm.write(
                f"[ERROR sid={getattr(rec, 'sid', '?')}] "
                f"{type(exc).__name__}: {exc}"
            )

            diagnostics.append({
                "sid": int(
                    getattr(
                        rec,
                        "sid",
                        -1,
                    )
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": " | ".join(
                    traceback.format_exc()
                    .splitlines()[-6:]
                ),
            })

        finally:
            if image is not None:
                image.close()

            if real_batch is not None:
                del real_batch

            if no_batch is not None:
                del no_batch

            cleanup()

    if not base_rows:
        raise RuntimeError(
            "No samples extracted."
        )

    base_last = np.stack(
        base_rows,
        axis=0,
    ).astype(np.float32)

    boost_last = np.stack(
        boost_rows,
        axis=0,
    ).astype(np.float32)

    noimage_last = np.stack(
        noimage_rows,
        axis=0,
    ).astype(np.float32)

    boost_delta = (
        boost_last
        - base_last
    ).astype(np.float32)

    return (
        base_last,
        boost_last,
        noimage_last,
        boost_delta,
        saved_sids,
        saved_labels,
        diagnostics,
    )


# ============================================================================
# Generation replay diagnostics
# ============================================================================

def run_generation_replay(
    *,
    model: Any,
    processor: Any,
    records_by_sid: Mapping[int, Any],
    prompt_map: Mapping[int, Mapping[str, Any]],
    decoder_layers: Sequence[torch.nn.Module],
    sids: Sequence[int],
    labels_by_sid: Mapping[int, str],
    boundary_delta_by_sid: Mapping[int, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    install_repository_weighted_generation()

    rows: List[
        Dict[str, Any]
    ] = []

    selected_sids = list(
        sids
    )

    if args.generation_max_samples is not None:
        selected_sids = selected_sids[
            : int(
                args.generation_max_samples
            )
        ]

    boundary = max(
        parse_layer_spec(
            args.boost_layers,
            len(decoder_layers),
        )
    )

    for sid in tqdm(
        selected_sids,
        desc="generate base / boost / exact-delta replay",
    ):
        rec = records_by_sid[
            int(sid)
        ]

        gt = labels_by_sid[
            int(sid)
        ]

        prompt = prompt_for_record(
            rec,
            prompt_map,
        )

        image = None
        batch = None

        try:
            image = Image.open(
                rec.image_path
            ).convert("RGB")

            batch = build_real_batch(
                processor,
                prompt,
                image,
                args.device,
                int(model.config.image_token_index),
            )

            base_text, base_pred = (
                generate_once(
                    model,
                    processor,
                    batch,
                    args,
                    boost=False,
                )
            )

            boost_text, boost_pred = (
                generate_once(
                    model,
                    processor,
                    batch,
                    args,
                    boost=True,
                )
            )

            delta = (
                boundary_delta_by_sid[
                    int(sid)
                ]
            )

            with AddLastTokenDelta(
                decoder_layers[
                    boundary
                ],
                delta,
                scale=1.0,
            ):
                replay_text, replay_pred = (
                    generate_once(
                        model,
                        processor,
                        batch,
                        args,
                        boost=False,
                    )
                )

            rows.append({
                "sid": int(sid),
                "gt": gt,
                "boundary_layer": boundary,
                "base_pred": base_pred or "",
                "boost_pred": boost_pred or "",
                "replay_pred": replay_pred or "",
                "base_correct": int(
                    base_pred == gt
                ),
                "boost_correct": int(
                    boost_pred == gt
                ),
                "replay_correct": int(
                    replay_pred == gt
                ),
                "boost_equals_replay_pred": int(
                    boost_pred == replay_pred
                ),
                "base_text": base_text,
                "boost_text": boost_text,
                "replay_text": replay_text,
            })

        finally:
            if image is not None:
                image.close()

            if batch is not None:
                del batch

            cleanup()

    summary = {
        "N": len(rows),
        "boundary_layer": boundary,
        "base_acc": safe_mean(
            row["base_correct"]
            for row in rows
        ),
        "boost_acc": safe_mean(
            row["boost_correct"]
            for row in rows
        ),
        "exact_delta_replay_acc": safe_mean(
            row["replay_correct"]
            for row in rows
        ),
        "boost_replay_prediction_agreement": safe_mean(
            row["boost_equals_replay_pred"]
            for row in rows
        ),
    }

    return summary, rows


def run_oracle_mean_delta_replay(
    *,
    model: Any,
    processor: Any,
    records_by_sid: Mapping[int, Any],
    prompt_map: Mapping[int, Mapping[str, Any]],
    decoder_layers: Sequence[torch.nn.Module],
    all_sids: Sequence[int],
    labels: np.ndarray,
    boundary_deltas: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """
    Split-0 only.

    Fit mu_r from TRAIN boost deltas.
    TEST receives mu_GT as an oracle diagnostic.
    """

    install_repository_weighted_generation()

    train_idx, test_idx = stratified_split(
        labels,
        args.train_ratio,
        args.seed,
    )

    relation_means: Dict[
        str,
        np.ndarray
    ] = {}

    for relation in RELATIONS:
        mask_idx = [
            idx
            for idx in train_idx.tolist()
            if labels[idx] == relation
        ]

        relation_means[
            relation
        ] = (
            boundary_deltas[
                mask_idx
            ]
            .mean(
                axis=0
            )
            .astype(
                np.float32
            )
        )

    boundary = max(
        parse_layer_spec(
            args.boost_layers,
            len(decoder_layers),
        )
    )

    test_indices = (
        test_idx.tolist()
    )

    if args.generation_max_samples is not None:
        test_indices = test_indices[
            : int(
                args.generation_max_samples
            )
        ]

    rows: List[
        Dict[str, Any]
    ] = []

    for array_idx in tqdm(
        test_indices,
        desc="oracle TRAIN-mean delta replay",
    ):
        sid = int(
            all_sids[
                array_idx
            ]
        )

        gt = str(
            labels[
                array_idx
            ]
        )

        rec = records_by_sid[
            sid
        ]

        prompt = prompt_for_record(
            rec,
            prompt_map,
        )

        image = None
        batch = None

        try:
            image = Image.open(
                rec.image_path
            ).convert(
                "RGB"
            )

            batch = build_real_batch(
                processor,
                prompt,
                image,
                args.device,
                int(model.config.image_token_index),
            )

            base_text, base_pred = (
                generate_once(
                    model,
                    processor,
                    batch,
                    args,
                    boost=False,
                )
            )

            delta = relation_means[
                gt
            ]

            with AddLastTokenDelta(
                decoder_layers[
                    boundary
                ],
                delta,
                scale=args.mean_delta_scale,
            ):
                edit_text, edit_pred = (
                    generate_once(
                        model,
                        processor,
                        batch,
                        args,
                        boost=False,
                    )
                )

            rows.append({
                "sid": sid,
                "gt": gt,
                "boundary_layer": boundary,
                "base_pred": base_pred or "",
                "oracle_mean_delta_pred": edit_pred or "",
                "base_correct": int(
                    base_pred == gt
                ),
                "edit_correct": int(
                    edit_pred == gt
                ),
                "W2C": int(
                    base_pred != gt
                    and edit_pred == gt
                ),
                "C2W": int(
                    base_pred == gt
                    and edit_pred != gt
                ),
                "base_text": base_text,
                "edit_text": edit_text,
            })

        finally:
            if image is not None:
                image.close()

            if batch is not None:
                del batch

            cleanup()

    base_acc = safe_mean(
        row["base_correct"]
        for row in rows
    )

    edit_acc = safe_mean(
        row["edit_correct"]
        for row in rows
    )

    w2c = sum(
        int(row["W2C"])
        for row in rows
    )

    c2w = sum(
        int(row["C2W"])
        for row in rows
    )

    summary = {
        "N": len(rows),
        "boundary_layer": boundary,
        "mean_delta_scale": args.mean_delta_scale,
        "base_acc": base_acc,
        "oracle_mean_delta_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
        "note": (
            "GT-routed TEST diagnostic; relation means fit on TRAIN only."
        ),
    }

    return summary, rows


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    if (
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if not (
        0.0
        < args.train_ratio
        < 1.0
    ):
        raise ValueError(
            "--train-ratio must be in (0,1)."
        )

    random.seed(
        args.seed
    )
    np.random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )

    outdir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and outdir.exists()
    ):
        shutil.rmtree(
            outdir
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, audit = (
        data_helpers.load_records(
            args.dataset,
            Path(args.data_root),
            args.max_samples,
        )
    )

    records = [
        rec
        for rec in records
        if normalize_relation(
            rec.relation
        ) in REL_TO_ID
    ]

    if not records:
        raise RuntimeError(
            "No four-way records found."
        )

    prompt_map = load_prompt_map(
        Path(
            args.prompt_jsonl
        )
    )

    model, processor = (
        load_model_and_processor(
            args
        )
    )

    decoder_layers = (
        resolve_decoder_layers(
            model
        )
    )

    n_layers = len(
        decoder_layers
    )

    boost_layers = parse_layer_spec(
        args.boost_layers,
        n_layers,
    )

    probe_layers = parse_layer_spec(
        args.probe_layers,
        n_layers,
    )

    boundary_layer = max(
        boost_layers
    )

    if boundary_layer not in probe_layers:
        probe_layers = sorted(
            set(
                probe_layers
                + [boundary_layer]
            )
        )

    print(
        "\n"
        + "=" * 154
    )
    print(
        "VERIFY LLaVA ATTENTION BOOST AS A MODEL-GENERATED SPATIAL DIRECTION"
    )
    print(
        "=" * 154
    )
    print(
        f"model={args.model} | repo={MODEL_REPOS[args.model]}"
    )
    print(
        f"dataset={args.dataset} | N={len(records)}"
    )
    print(
        f"decoder_layers={n_layers}"
    )
    print(
        f"boost_layers={boost_layers}"
    )
    print(
        f"boundary_layer=L{boundary_layer}"
    )
    print(
        f"boost_weight={args.boost_weight}"
    )
    print(
        f"attention_variant={args.attention_variant}"
    )
    print(
        f"probe_layers={probe_layers}"
    )
    print(
        f"split={args.train_ratio:.2f}/{1.0-args.train_ratio:.2f} "
        f"repeats={args.repeats}"
    )
    print(
        "input_builder=legacy_manual "
        "(tokenizer + image_processor; exactly one <image> token)"
    )
    print(
        f"image_token_index={int(model.config.image_token_index)}"
    )
    print(
        "=" * 154
    )

    # Keep existing AdaptVis attention code unchanged except its layer window.
    with RestrictBoostLayers(
        decoder_layers,
        boost_layers,
    ):
        (
            base_last,
            boost_last,
            noimage_last,
            boost_delta,
            saved_sids,
            saved_labels,
            extraction_rows,
        ) = extract_vectors(
            model=model,
            processor=processor,
            records=records,
            prompt_map=prompt_map,
            probe_layers=probe_layers,
            args=args,
        )

        labels = np.asarray(
            saved_labels,
            dtype=object,
        )

        base_residual = (
            base_last
            - noimage_last
        ).astype(
            np.float32
        )

        boost_residual = (
            boost_last
            - noimage_last
        ).astype(
            np.float32
        )

        (
            probe_repeat_rows,
            probe_summary_rows,
        ) = run_all_direction_probes(
            base_residual=base_residual,
            boost_residual=boost_residual,
            boost_delta=boost_delta,
            labels=labels,
            layers=probe_layers,
            train_ratio=args.train_ratio,
            repeats=args.repeats,
            seed=args.seed,
        )

        write_csv(
            outdir
            / "direction_probe_repeats.csv",
            probe_repeat_rows,
        )

        write_csv(
            outdir
            / "direction_probe_summary.csv",
            probe_summary_rows,
        )

        write_csv(
            outdir
            / "extraction_diagnostics.csv",
            extraction_rows,
        )

        if args.save_vectors:
            np.savez_compressed(
                outdir
                / "vectors.npz",
                sample_index=np.asarray(
                    saved_sids,
                    dtype=np.int64,
                ),
                relation=labels,
                layers=np.asarray(
                    probe_layers,
                    dtype=np.int32,
                ),
                base_last=base_last.astype(
                    np.float16
                ),
                boost_last=boost_last.astype(
                    np.float16
                ),
                noimage_last=noimage_last.astype(
                    np.float16
                ),
                base_residual=base_residual.astype(
                    np.float16
                ),
                boost_residual=boost_residual.astype(
                    np.float16
                ),
                boost_delta=boost_delta.astype(
                    np.float16
                ),
            )

        # ---------------------------------------------------------------
        # Console direction summary
        # ---------------------------------------------------------------
        print(
            "\n"
            + "=" * 154
        )
        print(
            "DIRECTION READOUT: WHAT INFORMATION ENTERS THE LAST TOKEN?"
        )
        print(
            "=" * 154
        )

        summary_lookup = {
            (
                row["representation"],
                int(row["layer"]),
            ): row
            for row in probe_summary_rows
        }

        print(
            f"{'layer':>5s} | "
            f"{'base Real-NoImg':>17s} | "
            f"{'boost Real-NoImg':>18s} | "
            f"{'boost-base delta':>17s} | "
            f"{'delta gain':>10s}"
        )

        print(
            "-" * 154
        )

        for layer in probe_layers:
            base_row = summary_lookup[
                (
                    "base_residual",
                    layer,
                )
            ]

            boost_row = summary_lookup[
                (
                    "boost_residual",
                    layer,
                )
            ]

            delta_row = summary_lookup[
                (
                    "boost_delta",
                    layer,
                )
            ]

            gain = (
                boost_row[
                    "accuracy_mean"
                ]
                -
                base_row[
                    "accuracy_mean"
                ]
            )

            print(
                f"L{layer:02d}   | "
                f"{base_row['accuracy_mean']:.4f}"
                f"±{base_row['accuracy_std']:.3f} | "
                f"{boost_row['accuracy_mean']:.4f}"
                f"±{boost_row['accuracy_std']:.3f} | "
                f"{delta_row['accuracy_mean']:.4f}"
                f"±{delta_row['accuracy_std']:.3f} | "
                f"{gain:+.4f}"
            )

        print(
            "=" * 154
        )

        # ---------------------------------------------------------------
        # Boundary delta map for causal replay.
        # ---------------------------------------------------------------
        boundary_li = probe_layers.index(
            boundary_layer
        )

        boundary_deltas = (
            boost_delta[
                :,
                boundary_li,
                :,
            ]
        )

        boundary_delta_by_sid = {
            int(sid): boundary_deltas[
                i
            ]
            for i, sid in enumerate(
                saved_sids
            )
        }

        records_by_sid = {
            int(rec.sid): rec
            for rec in records
        }

        labels_by_sid = {
            int(sid): str(labels[i])
            for i, sid in enumerate(
                saved_sids
            )
        }

        if args.run_generation:
            generation_summary, generation_rows = (
                run_generation_replay(
                    model=model,
                    processor=processor,
                    records_by_sid=records_by_sid,
                    prompt_map=prompt_map,
                    decoder_layers=decoder_layers,
                    sids=saved_sids,
                    labels_by_sid=labels_by_sid,
                    boundary_delta_by_sid=boundary_delta_by_sid,
                    args=args,
                )
            )

            write_csv(
                outdir
                / "generation_summary.csv",
                [generation_summary],
            )

            write_csv(
                outdir
                / "generation_details.csv",
                generation_rows,
            )

            print(
                "\n"
                + "=" * 154
            )
            print(
                "CAUSAL TEST 1 — EXACT SAMPLE-SPECIFIC DELTA REPLAY"
            )
            print(
                "=" * 154
            )

            print(
                f"N={generation_summary['N']} | "
                f"boundary=L{generation_summary['boundary_layer']}"
            )

            print(
                f"baseline             = "
                f"{generation_summary['base_acc']:.4f}"
            )

            print(
                f"attention boost      = "
                f"{generation_summary['boost_acc']:.4f}"
            )

            print(
                f"exact delta replay   = "
                f"{generation_summary['exact_delta_replay_acc']:.4f}"
            )

            print(
                f"boost/replay pred agreement = "
                f"{generation_summary['boost_replay_prediction_agreement']:.4f}"
            )

            print(
                "=" * 154
            )

        if args.run_oracle_mean_replay:
            (
                oracle_summary,
                oracle_rows,
            ) = run_oracle_mean_delta_replay(
                model=model,
                processor=processor,
                records_by_sid=records_by_sid,
                prompt_map=prompt_map,
                decoder_layers=decoder_layers,
                all_sids=saved_sids,
                labels=labels,
                boundary_deltas=boundary_deltas,
                args=args,
            )

            write_csv(
                outdir
                / "oracle_mean_delta_summary.csv",
                [oracle_summary],
            )

            write_csv(
                outdir
                / "oracle_mean_delta_details.csv",
                oracle_rows,
            )

            print(
                "\n"
                + "=" * 154
            )
            print(
                "CAUSAL TEST 2 — TRAIN RELATION-MEAN BOOST DELTA, GT ROUTING"
            )
            print(
                "=" * 154
            )

            print(
                f"N={oracle_summary['N']} | "
                f"boundary=L{oracle_summary['boundary_layer']} | "
                f"scale={oracle_summary['mean_delta_scale']}"
            )

            print(
                f"acc "
                f"{oracle_summary['base_acc']:.4f}"
                f" -> "
                f"{oracle_summary['oracle_mean_delta_acc']:.4f} "
                f"({oracle_summary['gain']:+.4f}) | "
                f"W2C={oracle_summary['W2C']} "
                f"C2W={oracle_summary['C2W']} "
                f"net={oracle_summary['net']:+d}"
            )

            print(
                "NOTE: this is an oracle TEST diagnostic; "
                "GT selects a TRAIN-fitted relation mean."
            )

            print(
                "=" * 154
            )

    config = {
        "model": args.model,
        "repo_id": MODEL_REPOS[
            args.model
        ],
        "dataset": args.dataset,
        "N_requested": len(records),
        "N_saved": len(saved_sids),
        "boost_layers": boost_layers,
        "boundary_layer": boundary_layer,
        "boost_weight": args.boost_weight,
        "attention_variant": args.attention_variant,
        "probe_layers": probe_layers,
        "train_ratio": args.train_ratio,
        "repeats": args.repeats,
        "seed": args.seed,
        "prompt_jsonl": args.prompt_jsonl,
        "run_generation": args.run_generation,
        "run_oracle_mean_replay": args.run_oracle_mean_replay,
    }

    (
        outdir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[saved] "
        f"{outdir / 'direction_probe_summary.csv'}"
    )

    if args.run_generation:
        print(
            f"[saved] "
            f"{outdir / 'generation_summary.csv'}"
        )

    if args.run_oracle_mean_replay:
        print(
            f"[saved] "
            f"{outdir / 'oracle_mean_delta_summary.csv'}"
        )


if __name__ == "__main__":
    main()
