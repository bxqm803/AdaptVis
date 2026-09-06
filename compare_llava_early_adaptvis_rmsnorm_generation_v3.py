#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_llava_early_adaptvis_rmsnorm_generation_v3.py

Compare LLaVA LAST TOKEN under three conditions: original RMSNorm epsilon,
RMSNorm epsilon 1e-6 alone, and epsilon 1e-6 plus early AdaptVis.

Default experiment:
    baseline: no AdaptVis scaling
    adaptvis:  L0-L3 last-query -> visual-token PRE-SOFTMAX logits scaled by 0.5

For mul_img the repository applies:
    s_img_new = weight * s_img
BEFORE softmax. When visual logits are negative, weight=0.5 shrinks them
toward zero and therefore can INCREASE their post-softmax attention mass.

The script measures, at every decoder layer:

    h_base[l]       = baseline last-token hidden state
    h_boost[l]      = boosted last-token hidden state
    delta[l]        = h_boost[l] - h_base[l]

Geometry:
    ||h_base||
    ||h_boost||
    ||delta||
    ||delta|| / ||h_base||
    cosine(h_base, h_boost)
    cosine(delta[l], delta[boundary])
    cosine(delta[l], delta[l-1])

The boundary is the final boosted layer (default L3).

It also evaluates relation information using TRAIN-only direction prototypes:

    base_residual[l]  = h_base_real[l]  - h_noimage[l]
    boost_residual[l] = h_boost_real[l] - h_noimage[l]
    boost_delta[l]    = h_boost_real[l] - h_base_real[l]

Thus the output tells us BOTH:

1) how much / in what direction the last-token state changes;
2) whether that change becomes increasingly relation-specific downstream.

Interpretation
--------------
If boost is only L0-L3 but:

    ||delta|| remains non-zero after L3,
    cosine(delta_l, delta_L3) gradually falls,
    while relation accuracy of delta_l rises,

then the early attention intervention is NOT simply copying one fixed vector
to the end. It creates an early sample-specific perturbation that subsequent
layers transform into a more spatially structured signal.

If instead:

    cosine(delta_l, delta_L3) stays near 1

then the perturbation is mostly propagated unchanged.

The script uses the repository's custom:
    model_zoo.llava.modeling_llava_scal.LlavaForConditionalGenerationScal

and preserves its LEGACY input contract:
    exactly ONE <image> token in input_ids,
    with image patches expanded inside the model.

Example
-------
CUDA_VISIBLE_DEVICES=0 python compare_llava_early_adaptvis_rmsnorm_generation_v3.py \
  --model llava-7b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --boost-layers 0-3 \
  --boost-weight 0.5 \
  --attention-variant mul_img \
  --train-ratio 0.30 \
  --repeats 5 \
  --save-vectors \
  --output-dir output/llava7b_early4_adaptvis_rmsnorm_v3 \
  --overwrite

Smoke test:
CUDA_VISIBLE_DEVICES=0 python compare_llava_early_adaptvis_rmsnorm_generation_v3.py \
  --model llava-7b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --boost-layers 0-3 \
  --boost-weight 0.5 \
  --attention-variant mul_img \
  --max-samples 20 \
  --repeats 2 \
  --output-dir output/llava7b_early4_lasttoken_compare_smoke \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
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


# =============================================================================
# CLI
# =============================================================================

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
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--revision",
        default="a272c74",
        help="Same revision used by the repository LLaVA wrapper.",
    )

    p.add_argument(
        "--boost-layers",
        default="0-3",
        help="Layers whose last-query visual attention is enhanced.",
    )
    p.add_argument(
        "--boost-weight",
        type=float,
        default=0.5,
        help=(
            "For mul_img this multiplies PRE-SOFTMAX visual attention logits. "
            "With predominantly negative visual logits, 0<weight<1 shrinks "
            "them toward zero and can increase post-softmax visual attention."
        ),
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
        help="'all', 'auto', range like 0-12, or comma list.",
    )

    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--base-rms-eps",
        type=float,
        default=1e-5,
        help="Original/before RMSNorm epsilon.",
    )
    p.add_argument(
        "--enhanced-rms-eps",
        type=float,
        default=1e-6,
        help="Enhanced/after RMSNorm epsilon.",
    )

    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip actual model.generate() A/B/C accuracy verification.",
    )
    p.add_argument(
        "--generation-max-samples",
        type=int,
        default=None,
        help="Optional cap for generation verification only.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum generated tokens for accuracy verification.",
    )

    p.add_argument("--save-vectors", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


# =============================================================================
# General utilities
# =============================================================================

def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(values: Iterable[float]) -> float:
    vals: List[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: Iterable[float]) -> float:
    vals: List[float] = []
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
            8, 10, 12, 14, 16, 18, 20,
            22, 24, 26, 28, 30,
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
            f"Invalid layer indices={bad}; valid=0..{n_layers-1}"
        )

    if not result:
        raise ValueError("No layers selected.")

    return result


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= EPS or nb <= EPS:
        return float("nan")

    return float(
        np.dot(a, b) / (na * nb)
    )


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(denom, EPS)


# =============================================================================
# Prompt loading
# =============================================================================

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
            sid = int(row.get("id", line_idx))
            result[sid] = row

    return result


def prompt_for_record(
    rec: Any,
    prompt_map: Mapping[int, Mapping[str, Any]],
) -> str:
    row = prompt_map.get(int(rec.sid))

    if row is None:
        raise KeyError(
            f"sid={rec.sid} missing from prompt JSONL."
        )

    return str(row["question"])


def noimage_prompt(prompt: str) -> str:
    text = prompt.replace("<image>", "")
    text = re.sub(r"^\s*\n", "", text)
    return text


# =============================================================================
# Model and legacy-compatible LLaVA inputs
# =============================================================================

def resolve_decoder_layers(
    model: Any,
) -> Sequence[torch.nn.Module]:
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

        if (
            okay
            and isinstance(
                obj,
                (torch.nn.ModuleList, list, tuple),
            )
        ):
            return obj

    raise RuntimeError(
        "Could not resolve LLaVA decoder layers."
    )


def load_model_and_processor(
    args: argparse.Namespace,
):
    repo_id = MODEL_REPOS[args.model]

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
        processor_kwargs["cache_dir"] = args.cache_dir

    if str(args.revision).strip():
        processor_kwargs["revision"] = str(args.revision).strip()

    processor = AutoProcessor.from_pretrained(
        repo_id,
        **processor_kwargs,
    )

    # DO NOT call configure_processor().
    # The repository's custom LLaVA model expects ONE <image> placeholder
    # and expands it internally.
    return model, processor


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


def tokenize_text_only(
    processor: Any,
    text: str,
) -> Dict[str, torch.Tensor]:
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


def process_image_only(
    processor: Any,
    image: Image.Image,
) -> Dict[str, torch.Tensor]:
    image_processor = getattr(
        processor,
        "image_processor",
        None,
    )

    if image_processor is None:
        raise RuntimeError(
            "Processor has no image_processor."
        )

    img = image_processor(
        images=image,
        return_tensors="pt",
    )

    if "pixel_values" not in img:
        raise RuntimeError(
            "image_processor returned no pixel_values."
        )

    return {
        "pixel_values": img["pixel_values"],
    }


def build_real_batch(
    model: Any,
    processor: Any,
    prompt: str,
    image: Image.Image,
    device: str,
) -> Dict[str, Any]:
    text_batch = tokenize_text_only(
        processor,
        prompt,
    )

    image_batch = process_image_only(
        processor,
        image,
    )

    image_token_index = int(
        model.config.image_token_index
    )

    count = int(
        (
            text_batch["input_ids"]
            == image_token_index
        )
        .sum()
        .item()
    )

    if count != 1:
        raise RuntimeError(
            "Legacy LlavaForConditionalGenerationScal expects exactly ONE "
            f"<image> token, got {count}. "
            f"image_token_index={image_token_index}."
        )

    return move_batch(
        {
            **text_batch,
            **image_batch,
        },
        device,
    )


def build_noimage_batch(
    processor: Any,
    prompt: str,
    device: str,
) -> Dict[str, Any]:
    batch = tokenize_text_only(
        processor,
        noimage_prompt(prompt),
    )

    return move_batch(
        batch,
        device,
    )



# =============================================================================
# RMSNorm epsilon control
# =============================================================================

def get_rmsnorm_eps_values(model: Any) -> Dict[str, float]:
    """
    Return runtime epsilon values from RMSNorm modules.

    This repository's custom RMSNorm stores epsilon in `variance_epsilon`.
    We also support modules using `.eps` for robustness.
    """
    values: Dict[str, float] = {}

    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()

        if "rmsnorm" not in class_name:
            continue

        if hasattr(module, "variance_epsilon"):
            values[name] = float(module.variance_epsilon)
        elif hasattr(module, "eps"):
            values[name] = float(module.eps)

    return values


def set_rmsnorm_eps(
    model: Any,
    eps: float,
) -> int:
    """
    Change RMSNorm epsilon at runtime.

    Important:
        Changing model.config alone is NOT sufficient after model creation,
        because the actual RMSNorm modules already hold their own epsilon.

    Therefore this function updates both configs and every instantiated
    RMSNorm module.
    """
    eps = float(eps)

    # Update all obvious config objects.
    config_candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(
            getattr(
                getattr(model, "language_model", None),
                "model",
                None,
            ),
            "config",
            None,
        ),
    ]

    for cfg in config_candidates:
        if cfg is not None and hasattr(cfg, "rms_norm_eps"):
            cfg.rms_norm_eps = eps

    changed = 0

    for _, module in model.named_modules():
        class_name = module.__class__.__name__.lower()

        if "rmsnorm" not in class_name:
            continue

        touched = False

        if hasattr(module, "variance_epsilon"):
            module.variance_epsilon = eps
            touched = True

        if hasattr(module, "eps"):
            module.eps = eps
            touched = True

        changed += int(touched)

    if changed == 0:
        raise RuntimeError(
            "No RMSNorm modules were found. "
            "Cannot verify epsilon intervention."
        )

    return changed


# =============================================================================
# Actual generation verification
# =============================================================================

def install_repository_weighted_generation() -> None:
    """
    Reuse the branch's own generation patch so `weight` and `adjust_method`
    are passed into LlavaForConditionalGenerationScal during generation.
    """
    from model_zoo.llava15 import change_greedy_to_add_weight

    change_greedy_to_add_weight()


def decode_generated(
    processor: Any,
    output: Any,
    prompt_len: int,
) -> str:
    sequences = getattr(output, "sequences", None)

    if sequences is None and isinstance(output, Mapping):
        sequences = output.get("sequences", None)

    if sequences is None:
        if torch.is_tensor(output):
            sequences = output
        else:
            raise RuntimeError("Generation returned no sequences.")

    return (
        processor.decode(
            sequences[0, int(prompt_len):],
            skip_special_tokens=True,
        )
        .strip()
    )


def generate_relation_once(
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    *,
    weight: Optional[float],
    attention_variant: str,
    max_new_tokens: int,
) -> Tuple[str, Optional[str]]:
    kwargs: Dict[str, Any] = {
        **batch,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    if weight is not None:
        kwargs["weight"] = float(weight)
        kwargs["adjust_method"] = attention_variant

    with torch.inference_mode():
        output = model.generate(**kwargs)

    text = decode_generated(
        processor,
        output,
        prompt_len=int(batch["input_ids"].shape[1]),
    )

    return text, normalize_relation(text)


def evaluate_generation_conditions(
    *,
    model: Any,
    processor: Any,
    records_by_sid: Mapping[int, Any],
    sids: Sequence[int],
    labels_by_sid: Mapping[int, str],
    prompt_map: Mapping[int, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate exactly three conditions on the SAME samples:

      A: base_rms_eps, no AdaptVis
      B: enhanced_rms_eps, no AdaptVis
      C: enhanced_rms_eps, AdaptVis

    This verifies that the hidden-state analysis is attached to a real
    generation improvement.
    """
    install_repository_weighted_generation()

    selected_sids = list(sids)

    if args.generation_max_samples is not None:
        selected_sids = selected_sids[
            : int(args.generation_max_samples)
        ]

    rows: List[Dict[str, Any]] = []

    for sid in tqdm(
        selected_sids,
        desc="generation A/B/C",
    ):
        sid = int(sid)
        rec = records_by_sid[sid]
        gt = str(labels_by_sid[sid])

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
                model,
                processor,
                prompt,
                image,
                args.device,
            )

            # A: original epsilon, no AdaptVis.
            set_rmsnorm_eps(
                model,
                args.base_rms_eps,
            )

            base_text, base_pred = generate_relation_once(
                model,
                processor,
                batch,
                weight=None,
                attention_variant=args.attention_variant,
                max_new_tokens=args.max_new_tokens,
            )

            # B: changed epsilon only.
            set_rmsnorm_eps(
                model,
                args.enhanced_rms_eps,
            )

            eps_text, eps_pred = generate_relation_once(
                model,
                processor,
                batch,
                weight=None,
                attention_variant=args.attention_variant,
                max_new_tokens=args.max_new_tokens,
            )

            # C: changed epsilon + early AdaptVis.
            full_text, full_pred = generate_relation_once(
                model,
                processor,
                batch,
                weight=args.boost_weight,
                attention_variant=args.attention_variant,
                max_new_tokens=args.max_new_tokens,
            )

            rows.append({
                "sid": sid,
                "gt": gt,

                "base_pred": base_pred or "",
                "eps_only_pred": eps_pred or "",
                "full_pred": full_pred or "",

                "base_correct": int(base_pred == gt),
                "eps_only_correct": int(eps_pred == gt),
                "full_correct": int(full_pred == gt),

                "base_to_full_W2C": int(
                    base_pred != gt
                    and full_pred == gt
                ),
                "base_to_full_C2W": int(
                    base_pred == gt
                    and full_pred != gt
                ),

                "base_text": base_text,
                "eps_only_text": eps_text,
                "full_text": full_text,
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

    eps_acc = safe_mean(
        row["eps_only_correct"]
        for row in rows
    )

    full_acc = safe_mean(
        row["full_correct"]
        for row in rows
    )

    w2c = sum(
        int(row["base_to_full_W2C"])
        for row in rows
    )

    c2w = sum(
        int(row["base_to_full_C2W"])
        for row in rows
    )

    summary = {
        "N": len(rows),

        "base_rms_eps": args.base_rms_eps,
        "enhanced_rms_eps": args.enhanced_rms_eps,

        "boost_layers": args.boost_layers,
        "attention_variant": args.attention_variant,
        "boost_weight": args.boost_weight,

        "base_acc": base_acc,
        "eps_only_acc": eps_acc,
        "full_acc": full_acc,

        "eps_gain_vs_base": eps_acc - base_acc,
        "adaptvis_gain_over_eps": full_acc - eps_acc,
        "total_gain_vs_base": full_acc - base_acc,

        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
    }

    return summary, rows


# =============================================================================
# Generic direction probe for arbitrary representation sets
# =============================================================================

def direction_probe_many(
    *,
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    layers: Sequence[int],
    train_ratio: float,
    repeats: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    repeat_rows: List[Dict[str, Any]] = []

    for rep in range(repeats):
        train_idx, test_idx = stratified_split(
            labels,
            train_ratio,
            seed + rep,
        )

        for name, tensor in representations.items():
            for li, layer in enumerate(layers):
                metrics = evaluate_probe(
                    tensor[:, li, :],
                    labels,
                    train_idx,
                    test_idx,
                )

                repeat_rows.append({
                    "repeat": rep,
                    "representation": name,
                    "layer": layer,
                    **metrics,
                })

    summary_rows: List[Dict[str, Any]] = []

    for name in representations:
        for layer in layers:
            rows = [
                row
                for row in repeat_rows
                if (
                    row["representation"] == name
                    and int(row["layer"]) == int(layer)
                )
            ]

            summary_rows.append({
                "representation": name,
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


# =============================================================================
# Restrict existing AdaptVis attention boost to chosen layers
# =============================================================================

class RestrictBoostLayers:
    """
    Existing repository attention receives:
        weight, idx, keys, adjust_method

    Outside the requested window, set weight=None.
    """

    def __init__(
        self,
        decoder_layers: Sequence[torch.nn.Module],
        enabled_layers: Sequence[int],
    ):
        self.enabled = {
            int(x)
            for x in enabled_layers
        }

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
        idx = kwargs.get("idx", None)

        if (
            idx is not None
            and int(idx) not in self.enabled
        ):
            kwargs = dict(kwargs)
            kwargs["weight"] = None

        return args, kwargs

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Forward and hidden-state extraction
# =============================================================================

def hidden_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    states = getattr(
        outputs,
        "hidden_states",
        None,
    )

    if (
        not isinstance(states, (tuple, list))
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
    boost_weight: Optional[float],
    attention_variant: str,
):
    with torch.inference_mode():
        return model(
            **batch,
            weight=boost_weight,
            adjust_method=attention_variant,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )


def forward_noimage(
    model: Any,
    batch: Mapping[str, Any],
):
    with torch.inference_mode():
        return model(
            **batch,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )


def last_token_trajectory(
    outputs: Any,
    selected_layers: Sequence[int],
) -> Dict[int, np.ndarray]:
    states = hidden_tuple(outputs)

    result: Dict[int, np.ndarray] = {}

    for layer in selected_layers:
        state_index = layer + 1

        if state_index >= len(states):
            raise RuntimeError(
                f"L{layer}: hidden_states has only {len(states)} entries."
            )

        state = states[state_index]

        if state.ndim != 3 or int(state.shape[0]) != 1:
            raise RuntimeError(
                f"L{layer}: invalid hidden shape={tuple(state.shape)}"
            )

        result[layer] = (
            state[0, -1]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return result


# =============================================================================
# TRAIN/TEST direction readout
# =============================================================================

def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    train: List[int] = []
    test: List[int] = []

    for relation in RELATIONS:
        ids = np.flatnonzero(
            labels == relation
        ).tolist()

        rng.shuffle(ids)

        if len(ids) < 2:
            raise RuntimeError(
                f"Need >=2 samples for relation={relation}"
            )

        n_train = int(
            round(
                len(ids) * train_ratio
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(ids) - 1,
            ),
        )

        train.extend(ids[:n_train])
        test.extend(ids[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)

    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def fit_direction_codebook(
    X: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    center = X.mean(axis=0)
    Xc = X - center

    directions = []

    for relation in RELATIONS:
        mask = labels == relation

        if not bool(mask.any()):
            raise RuntimeError(
                f"No TRAIN samples for relation={relation}"
            )

        direction = Xc[mask].mean(axis=0)

        direction = (
            direction
            / max(
                float(np.linalg.norm(direction)),
                EPS,
            )
        )

        directions.append(direction)

    return (
        center.astype(np.float32),
        np.stack(
            directions,
            axis=0,
        ).astype(np.float32),
    )


def evaluate_probe(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    center, directions = fit_direction_codebook(
        X[train_idx],
        labels[train_idx],
    )

    Xt = normalize_rows(
        X[test_idx] - center
    )

    scores = Xt @ directions.T

    pred = np.argmax(
        scores,
        axis=1,
    )

    gt = np.asarray(
        [
            REL_TO_ID[str(x)]
            for x in labels[test_idx]
        ],
        dtype=np.int64,
    )

    result: Dict[str, Any] = {
        "accuracy": float(
            np.mean(pred == gt)
        ),
        "n_test": int(len(test_idx)),
    }

    for relation, rid in REL_TO_ID.items():
        mask = gt == rid
        result[f"{relation}_accuracy"] = (
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


def direction_probe_all(
    base_residual: np.ndarray,
    boost_residual: np.ndarray,
    delta: np.ndarray,
    labels: np.ndarray,
    layers: Sequence[int],
    train_ratio: float,
    repeats: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    representations = {
        "base_residual": base_residual,
        "boost_residual": boost_residual,
        "boost_delta": delta,
    }

    repeat_rows: List[Dict[str, Any]] = []

    for rep in range(repeats):
        train_idx, test_idx = stratified_split(
            labels,
            train_ratio,
            seed + rep,
        )

        for name, tensor in representations.items():
            for li, layer in enumerate(layers):
                metrics = evaluate_probe(
                    tensor[:, li, :],
                    labels,
                    train_idx,
                    test_idx,
                )

                repeat_rows.append({
                    "repeat": rep,
                    "representation": name,
                    "layer": layer,
                    **metrics,
                })

    summary_rows: List[Dict[str, Any]] = []

    for name in representations:
        for layer in layers:
            rows = [
                row
                for row in repeat_rows
                if (
                    row["representation"] == name
                    and int(row["layer"]) == int(layer)
                )
            ]

            summary_rows.append({
                "representation": name,
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


# =============================================================================
# Main extraction
# =============================================================================

def main() -> None:
    args = parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(
            "--train-ratio must be in (0,1)."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, audit = data_helpers.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )

    records = [
        rec
        for rec in records
        if normalize_relation(rec.relation)
        in REL_TO_ID
    ]

    if not records:
        raise RuntimeError(
            "No usable four-way records."
        )

    prompt_map = load_prompt_map(
        Path(args.prompt_jsonl)
    )

    model, processor = load_model_and_processor(
        args
    )

    decoder_layers = resolve_decoder_layers(
        model
    )

    n_layers = len(decoder_layers)

    boost_layers = parse_layer_spec(
        args.boost_layers,
        n_layers,
    )

    probe_layers = parse_layer_spec(
        args.probe_layers,
        n_layers,
    )

    boundary_layer = max(boost_layers)

    if boundary_layer not in probe_layers:
        probe_layers = sorted(
            set(
                probe_layers
                + [boundary_layer]
            )
        )

    original_eps_values = get_rmsnorm_eps_values(
        model
    )

    original_unique_eps = sorted({
        float(v)
        for v in original_eps_values.values()
    })

    # Verify that runtime mutation reaches actual modules.
    n_rms = set_rmsnorm_eps(
        model,
        args.base_rms_eps,
    )

    base_eps_after_set = sorted({
        float(v)
        for v in get_rmsnorm_eps_values(model).values()
    })

    set_rmsnorm_eps(
        model,
        args.enhanced_rms_eps,
    )

    enhanced_eps_after_set = sorted({
        float(v)
        for v in get_rmsnorm_eps_values(model).values()
    })

    # Restore A before extraction begins.
    set_rmsnorm_eps(
        model,
        args.base_rms_eps,
    )

    if args.attention_variant == "mul_img":
        if 0.0 < args.boost_weight < 1.0:
            print(
                f"[AdaptVis] mul_img weight={args.boost_weight}: "
                "pre-softmax visual logits are shrunk toward zero. "
                "For negative logits this can increase post-softmax "
                "visual attention."
            )

    print("\n" + "=" * 166)
    print(
        "LLaVA THREE-CONDITION TEST: RMSNorm epsilon + EARLY AdaptVis"
    )
    print("=" * 166)

    print(
        f"model={args.model} | dataset={args.dataset} | N={len(records)}"
    )

    print(
        f"A ORIGINAL : rms_eps={args.base_rms_eps:g}, no AdaptVis"
    )

    print(
        f"B EPS ONLY : rms_eps={args.enhanced_rms_eps:g}, no AdaptVis"
    )

    print(
        f"C FULL     : rms_eps={args.enhanced_rms_eps:g}, "
        f"{args.attention_variant} weight={args.boost_weight} "
        f"on layers={boost_layers}"
    )

    print(
        f"RMSNorm modules changed={n_rms} | "
        f"loaded eps values={original_unique_eps}"
    )

    print(
        f"verified A eps={base_eps_after_set} | "
        f"verified B/C eps={enhanced_eps_after_set}"
    )

    print(
        "legacy_input=True | exactly one <image> placeholder"
    )

    print("=" * 166)

    # ---------------------------------------------------------------------
    # Arrays:
    #
    # A = base eps, no AdaptVis
    # B = enhanced eps, no AdaptVis
    # C = enhanced eps + AdaptVis
    # ---------------------------------------------------------------------
    A_real_rows: List[np.ndarray] = []
    A_no_rows: List[np.ndarray] = []

    B_real_rows: List[np.ndarray] = []
    B_no_rows: List[np.ndarray] = []

    C_real_rows: List[np.ndarray] = []

    saved_sids: List[int] = []
    saved_labels: List[str] = []

    per_sample_geometry: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with RestrictBoostLayers(
        decoder_layers,
        boost_layers,
    ):
        for rec in tqdm(
            records,
            desc="A(base eps) / B(eps only) / C(eps+AdaptVis)",
        ):
            image = None
            real_batch = None
            no_batch = None

            try:
                relation = normalize_relation(
                    rec.relation
                )

                prompt = prompt_for_record(
                    rec,
                    prompt_map,
                )

                image = Image.open(
                    rec.image_path
                ).convert("RGB")

                real_batch = build_real_batch(
                    model,
                    processor,
                    prompt,
                    image,
                    args.device,
                )

                no_batch = build_noimage_batch(
                    processor,
                    prompt,
                    args.device,
                )

                # =========================================================
                # A: original epsilon, no AdaptVis.
                # =========================================================
                set_rmsnorm_eps(
                    model,
                    args.base_rms_eps,
                )

                A_real_out = forward_real(
                    model,
                    real_batch,
                    boost_weight=None,
                    attention_variant=args.attention_variant,
                )

                A_no_out = forward_noimage(
                    model,
                    no_batch,
                )

                # =========================================================
                # B: enhanced epsilon only.
                # =========================================================
                set_rmsnorm_eps(
                    model,
                    args.enhanced_rms_eps,
                )

                B_real_out = forward_real(
                    model,
                    real_batch,
                    boost_weight=None,
                    attention_variant=args.attention_variant,
                )

                B_no_out = forward_noimage(
                    model,
                    no_batch,
                )

                # =========================================================
                # C: enhanced epsilon + early AdaptVis.
                # =========================================================
                C_real_out = forward_real(
                    model,
                    real_batch,
                    boost_weight=args.boost_weight,
                    attention_variant=args.attention_variant,
                )

                A_real_map = last_token_trajectory(
                    A_real_out,
                    probe_layers,
                )

                A_no_map = last_token_trajectory(
                    A_no_out,
                    probe_layers,
                )

                B_real_map = last_token_trajectory(
                    B_real_out,
                    probe_layers,
                )

                B_no_map = last_token_trajectory(
                    B_no_out,
                    probe_layers,
                )

                C_real_map = last_token_trajectory(
                    C_real_out,
                    probe_layers,
                )

                def stack_map(m):
                    return np.stack(
                        [
                            m[layer]
                            for layer in probe_layers
                        ],
                        axis=0,
                    ).astype(np.float32)

                A_real = stack_map(A_real_map)
                A_no = stack_map(A_no_map)

                B_real = stack_map(B_real_map)
                B_no = stack_map(B_no_map)

                C_real = stack_map(C_real_map)

                d_eps = (
                    B_real
                    - A_real
                ).astype(np.float32)

                d_adapt = (
                    C_real
                    - B_real
                ).astype(np.float32)

                d_total = (
                    C_real
                    - A_real
                ).astype(np.float32)

                boundary_li = probe_layers.index(
                    boundary_layer
                )

                d_adapt_boundary = d_adapt[
                    boundary_li
                ]

                d_total_boundary = d_total[
                    boundary_li
                ]

                for li, layer in enumerate(probe_layers):
                    hA = A_real[li]
                    hB = B_real[li]
                    hC = C_real[li]

                    de = d_eps[li]
                    da = d_adapt[li]
                    dt = d_total[li]

                    A_norm = float(
                        np.linalg.norm(hA)
                    )

                    per_sample_geometry.append({
                        "sid": int(rec.sid),
                        "relation": relation,
                        "layer": layer,

                        "is_adaptvis_layer": int(
                            layer in set(boost_layers)
                        ),

                        "A_base_norm": A_norm,
                        "B_eps_only_norm": float(
                            np.linalg.norm(hB)
                        ),
                        "C_full_norm": float(
                            np.linalg.norm(hC)
                        ),

                        "delta_eps_norm": float(
                            np.linalg.norm(de)
                        ),
                        "delta_adaptvis_norm": float(
                            np.linalg.norm(da)
                        ),
                        "delta_total_norm": float(
                            np.linalg.norm(dt)
                        ),

                        "relative_total_delta_norm": (
                            float(np.linalg.norm(dt))
                            / max(A_norm, EPS)
                        ),

                        "cos_A_C": cosine_np(
                            hA,
                            hC,
                        ),

                        "cos_adapt_delta_boundary": cosine_np(
                            da,
                            d_adapt_boundary,
                        ),

                        "cos_total_delta_boundary": cosine_np(
                            dt,
                            d_total_boundary,
                        ),
                    })

                A_real_rows.append(A_real)
                A_no_rows.append(A_no)

                B_real_rows.append(B_real)
                B_no_rows.append(B_no)

                C_real_rows.append(C_real)

                saved_sids.append(
                    int(rec.sid)
                )

                saved_labels.append(
                    str(relation)
                )

                del A_real_out
                del A_no_out
                del B_real_out
                del B_no_out
                del C_real_out

            except Exception as exc:
                errors.append({
                    "sid": int(
                        getattr(rec, "sid", -1)
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": " | ".join(
                        traceback.format_exc()
                        .splitlines()[-8:]
                    ),
                })

                tqdm.write(
                    f"[ERROR sid={getattr(rec, 'sid', '?')}] "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if image is not None:
                    image.close()

                if real_batch is not None:
                    del real_batch

                if no_batch is not None:
                    del no_batch

                cleanup()

    if not A_real_rows:
        raise RuntimeError(
            "No samples were extracted successfully."
        )

    A_real = np.stack(
        A_real_rows,
        axis=0,
    ).astype(np.float32)

    A_no = np.stack(
        A_no_rows,
        axis=0,
    ).astype(np.float32)

    B_real = np.stack(
        B_real_rows,
        axis=0,
    ).astype(np.float32)

    B_no = np.stack(
        B_no_rows,
        axis=0,
    ).astype(np.float32)

    C_real = np.stack(
        C_real_rows,
        axis=0,
    ).astype(np.float32)

    d_eps = (
        B_real
        - A_real
    ).astype(np.float32)

    d_adapt = (
        C_real
        - B_real
    ).astype(np.float32)

    d_total = (
        C_real
        - A_real
    ).astype(np.float32)

    A_residual = (
        A_real
        - A_no
    ).astype(np.float32)

    B_residual = (
        B_real
        - B_no
    ).astype(np.float32)

    # No-image has no visual tokens, hence condition C's corresponding
    # no-image branch is identical to B: same epsilon, no AdaptVis applicable.
    C_residual = (
        C_real
        - B_no
    ).astype(np.float32)

    labels = np.asarray(
        saved_labels,
        dtype=object,
    )

    # ---------------------------------------------------------------------
    # Aggregate geometric state change.
    # ---------------------------------------------------------------------
    geometry_summary: List[Dict[str, Any]] = []

    for layer in probe_layers:
        rows = [
            row
            for row in per_sample_geometry
            if int(row["layer"]) == int(layer)
        ]

        geometry_summary.append({
            "layer": layer,
            "is_adaptvis_layer": int(
                layer in set(boost_layers)
            ),
            "N": len(rows),

            "A_base_norm": safe_mean(
                row["A_base_norm"]
                for row in rows
            ),

            "B_eps_only_norm": safe_mean(
                row["B_eps_only_norm"]
                for row in rows
            ),

            "C_full_norm": safe_mean(
                row["C_full_norm"]
                for row in rows
            ),

            "delta_eps_norm": safe_mean(
                row["delta_eps_norm"]
                for row in rows
            ),

            "delta_adaptvis_norm": safe_mean(
                row["delta_adaptvis_norm"]
                for row in rows
            ),

            "delta_total_norm": safe_mean(
                row["delta_total_norm"]
                for row in rows
            ),

            "relative_total_delta_norm": safe_mean(
                row["relative_total_delta_norm"]
                for row in rows
            ),

            "cos_A_C": safe_mean(
                row["cos_A_C"]
                for row in rows
            ),

            "cos_adapt_delta_boundary": safe_mean(
                row["cos_adapt_delta_boundary"]
                for row in rows
            ),

            "cos_total_delta_boundary": safe_mean(
                row["cos_total_delta_boundary"]
                for row in rows
            ),
        })

    # ---------------------------------------------------------------------
    # Direction readout for states and decomposed deltas.
    # ---------------------------------------------------------------------
    representations = {
        "A_base_residual": A_residual,
        "B_eps_only_residual": B_residual,
        "C_full_residual": C_residual,

        "delta_eps": d_eps,
        "delta_adaptvis": d_adapt,
        "delta_total": d_total,
    }

    probe_repeat_rows, probe_summary_rows = (
        direction_probe_many(
            representations=representations,
            labels=labels,
            layers=probe_layers,
            train_ratio=args.train_ratio,
            repeats=args.repeats,
            seed=args.seed,
        )
    )

    probe_lookup = {
        (
            row["representation"],
            int(row["layer"]),
        ): row
        for row in probe_summary_rows
    }

    combined_summary: List[Dict[str, Any]] = []

    for row in geometry_summary:
        layer = int(row["layer"])

        combined_summary.append({
            **row,

            "A_base_spatial_acc": probe_lookup[
                ("A_base_residual", layer)
            ]["accuracy_mean"],

            "B_eps_only_spatial_acc": probe_lookup[
                ("B_eps_only_residual", layer)
            ]["accuracy_mean"],

            "C_full_spatial_acc": probe_lookup[
                ("C_full_residual", layer)
            ]["accuracy_mean"],

            "delta_eps_spatial_acc": probe_lookup[
                ("delta_eps", layer)
            ]["accuracy_mean"],

            "delta_adaptvis_spatial_acc": probe_lookup[
                ("delta_adaptvis", layer)
            ]["accuracy_mean"],

            "delta_total_spatial_acc": probe_lookup[
                ("delta_total", layer)
            ]["accuracy_mean"],
        })

    # ---------------------------------------------------------------------
    # Save hidden-state analysis.
    # ---------------------------------------------------------------------
    write_csv(
        outdir / "lasttoken_geometry_per_sample.csv",
        per_sample_geometry,
    )

    write_csv(
        outdir / "lasttoken_geometry_summary.csv",
        geometry_summary,
    )

    write_csv(
        outdir / "direction_probe_repeats.csv",
        probe_repeat_rows,
    )

    write_csv(
        outdir / "direction_probe_summary.csv",
        probe_summary_rows,
    )

    write_csv(
        outdir / "combined_layer_summary.csv",
        combined_summary,
    )

    (
        outdir / "errors.json"
    ).write_text(
        json.dumps(
            errors,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.save_vectors:
        np.savez_compressed(
            outdir / "lasttoken_vectors_ABC.npz",

            sample_index=np.asarray(
                saved_sids,
                dtype=np.int64,
            ),

            relation=labels,

            layers=np.asarray(
                probe_layers,
                dtype=np.int32,
            ),

            boost_layers=np.asarray(
                boost_layers,
                dtype=np.int32,
            ),

            A_real=A_real.astype(np.float16),
            A_noimage=A_no.astype(np.float16),

            B_real=B_real.astype(np.float16),
            B_noimage=B_no.astype(np.float16),

            C_real=C_real.astype(np.float16),

            A_residual=A_residual.astype(np.float16),
            B_residual=B_residual.astype(np.float16),
            C_residual=C_residual.astype(np.float16),

            delta_eps=d_eps.astype(np.float16),
            delta_adaptvis=d_adapt.astype(np.float16),
            delta_total=d_total.astype(np.float16),
        )

    # ---------------------------------------------------------------------
    # Console table 1: actual hidden-state geometry.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 184)
    print(
        "LAST-TOKEN GEOMETRY: "
        "A(eps1e-5) -> B(eps1e-6) -> C(eps1e-6 + AdaptVis)"
    )
    print("=" * 184)

    print(
        f"{'layer':>5s} | "
        f"{'||d_eps||':>10s} | "
        f"{'||d_adapt||':>11s} | "
        f"{'||d_total||':>11s} | "
        f"{'||dt||/||A||':>12s} | "
        f"{'cos(A,C)':>10s} | "
        f"{'cos(dA,dA@L'+str(boundary_layer)+')':>15s}"
    )

    print("-" * 184)

    for row in combined_summary:
        layer = int(row["layer"])
        marker = "*" if int(row["is_adaptvis_layer"]) else " "

        print(
            f"{marker}L{layer:02d} | "
            f"{row['delta_eps_norm']:10.4f} | "
            f"{row['delta_adaptvis_norm']:11.4f} | "
            f"{row['delta_total_norm']:11.4f} | "
            f"{row['relative_total_delta_norm']:12.5f} | "
            f"{row['cos_A_C']:10.6f} | "
            f"{row['cos_adapt_delta_boundary']:15.6f}"
        )

    print("=" * 184)
    print(
        "* = AdaptVis active on this layer"
    )

    # ---------------------------------------------------------------------
    # Console table 2: spatial information.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 184)
    print(
        "DIRECTION READOUT: SPATIAL INFORMATION IN STATE AND EACH CHANGE COMPONENT"
    )
    print("=" * 184)

    print(
        f"{'layer':>5s} | "
        f"{'A_base':>8s} | "
        f"{'B_eps':>8s} | "
        f"{'C_full':>8s} | "
        f"{'d_eps':>8s} | "
        f"{'d_adapt':>8s} | "
        f"{'d_total':>8s}"
    )

    print("-" * 184)

    for row in combined_summary:
        layer = int(row["layer"])
        marker = "*" if int(row["is_adaptvis_layer"]) else " "

        print(
            f"{marker}L{layer:02d} | "
            f"{row['A_base_spatial_acc']:8.4f} | "
            f"{row['B_eps_only_spatial_acc']:8.4f} | "
            f"{row['C_full_spatial_acc']:8.4f} | "
            f"{row['delta_eps_spatial_acc']:8.4f} | "
            f"{row['delta_adaptvis_spatial_acc']:8.4f} | "
            f"{row['delta_total_spatial_acc']:8.4f}"
        )

    print("=" * 184)

    # ---------------------------------------------------------------------
    # Actual model.generate() verification.
    # ---------------------------------------------------------------------
    generation_summary = None
    generation_rows = None

    if not args.skip_generation:
        records_by_sid = {
            int(rec.sid): rec
            for rec in records
        }

        labels_by_sid = {
            int(sid): str(labels[i])
            for i, sid in enumerate(saved_sids)
        }

        # RestrictBoostLayers must remain active for the full condition.
        with RestrictBoostLayers(
            decoder_layers,
            boost_layers,
        ):
            generation_summary, generation_rows = (
                evaluate_generation_conditions(
                    model=model,
                    processor=processor,
                    records_by_sid=records_by_sid,
                    sids=saved_sids,
                    labels_by_sid=labels_by_sid,
                    prompt_map=prompt_map,
                    args=args,
                )
            )

        write_csv(
            outdir / "generation_summary.csv",
            [generation_summary],
        )

        write_csv(
            outdir / "generation_details.csv",
            generation_rows,
        )

        print("\n" + "=" * 166)
        print(
            "ACTUAL model.generate() ACCURACY CHECK"
        )
        print("=" * 166)

        print(
            f"N={generation_summary['N']}"
        )

        print(
            f"A original   eps={args.base_rms_eps:g}, no AdaptVis"
            f"                         : "
            f"{generation_summary['base_acc']:.4f}"
        )

        print(
            f"B eps-only   eps={args.enhanced_rms_eps:g}, no AdaptVis"
            f"                         : "
            f"{generation_summary['eps_only_acc']:.4f} "
            f"({generation_summary['eps_gain_vs_base']:+.4f})"
        )

        print(
            f"C full       eps={args.enhanced_rms_eps:g}, "
            f"{args.attention_variant}={args.boost_weight}, L{boost_layers[0]}-L{boost_layers[-1]}"
            f" : "
            f"{generation_summary['full_acc']:.4f} "
            f"({generation_summary['total_gain_vs_base']:+.4f} vs A)"
        )

        print(
            f"AdaptVis incremental gain over eps-only: "
            f"{generation_summary['adaptvis_gain_over_eps']:+.4f}"
        )

        print(
            f"A -> C: W2C={generation_summary['W2C']} "
            f"C2W={generation_summary['C2W']} "
            f"net={generation_summary['net']:+d}"
        )

        print("=" * 166)

    # Restore enhanced epsilon because that is the final condition under study.
    set_rmsnorm_eps(
        model,
        args.enhanced_rms_eps,
    )

    config = {
        "model": args.model,
        "repo_id": MODEL_REPOS[args.model],
        "dataset": args.dataset,

        "N_requested": len(records),
        "N_success": len(saved_sids),
        "N_errors": len(errors),

        "base_rms_eps": args.base_rms_eps,
        "enhanced_rms_eps": args.enhanced_rms_eps,
        "n_rmsnorm_modules": n_rms,

        "boost_layers": boost_layers,
        "boundary_layer": boundary_layer,
        "boost_weight": args.boost_weight,
        "attention_variant": args.attention_variant,

        "probe_layers": probe_layers,
        "train_ratio": args.train_ratio,
        "repeats": args.repeats,
        "seed": args.seed,

        "generation_enabled": not args.skip_generation,
        "generation_max_samples": args.generation_max_samples,
        "max_new_tokens": args.max_new_tokens,

        "prompt_jsonl": args.prompt_jsonl,
    }

    (
        outdir / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[saved] {outdir / 'combined_layer_summary.csv'}"
    )

    print(
        f"[saved] {outdir / 'direction_probe_summary.csv'}"
    )

    if not args.skip_generation:
        print(
            f"[saved] {outdir / 'generation_summary.csv'}"
        )

    if args.save_vectors:
        print(
            f"[saved] {outdir / 'lasttoken_vectors_ABC.npz'}"
        )


if __name__ == "__main__":
    main()
