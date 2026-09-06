#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_llava_early_attention_lasttoken_v1.py

Compare how the LLaVA LAST TOKEN changes when its attention to VISUAL TOKENS
is multiplied in only the first few decoder layers.

Default experiment:
    baseline: no attention scaling
    boost:    L0-L3 last-query -> visual-token attention scaled by 1.2

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
CUDA_VISIBLE_DEVICES=0 python compare_llava_early_attention_lasttoken_v1.py \
  --model llava-7b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --boost-layers 0-3 \
  --boost-weight 1.2 \
  --attention-variant mul_img \
  --train-ratio 0.30 \
  --repeats 5 \
  --save-vectors \
  --output-dir output/llava7b_early4_lasttoken_compare_v1 \
  --overwrite

Smoke test:
CUDA_VISIBLE_DEVICES=0 python compare_llava_early_attention_lasttoken_v1.py \
  --model llava-7b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --boost-layers 0-3 \
  --boost-weight 1.2 \
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
        default=1.2,
        help="Multiplicative coefficient (>1 for mul_img).",
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

    if (
        args.attention_variant == "mul_img"
        and args.boost_weight <= 1.0
    ):
        print(
            f"[warning] mul_img with weight={args.boost_weight} "
            "is not an enhancement (>1 expected)."
        )

    print("\n" + "=" * 156)
    print(
        "LLaVA LAST-TOKEN TRAJECTORY: BASELINE vs EARLY LAST->VISUAL ATTENTION BOOST"
    )
    print("=" * 156)
    print(
        f"model={args.model} | dataset={args.dataset} | N={len(records)}"
    )
    print(
        f"boost_layers={boost_layers} | boundary=L{boundary_layer}"
    )
    print(
        f"attention_variant={args.attention_variant} | "
        f"boost_weight={args.boost_weight}"
    )
    print(
        f"probe_layers={probe_layers}"
    )
    print(
        "legacy_input=True | exactly one <image> placeholder"
    )
    print("=" * 156)

    base_rows: List[np.ndarray] = []
    boost_rows: List[np.ndarray] = []
    noimage_rows: List[np.ndarray] = []

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
            desc="baseline / boost / noimage",
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

                base_out = forward_real(
                    model,
                    real_batch,
                    boost_weight=None,
                    attention_variant=args.attention_variant,
                )

                boost_out = forward_real(
                    model,
                    real_batch,
                    boost_weight=args.boost_weight,
                    attention_variant=args.attention_variant,
                )

                no_out = forward_noimage(
                    model,
                    no_batch,
                )

                base_map = last_token_trajectory(
                    base_out,
                    probe_layers,
                )

                boost_map = last_token_trajectory(
                    boost_out,
                    probe_layers,
                )

                no_map = last_token_trajectory(
                    no_out,
                    probe_layers,
                )

                base_arr = np.stack(
                    [
                        base_map[layer]
                        for layer in probe_layers
                    ],
                    axis=0,
                ).astype(np.float32)

                boost_arr = np.stack(
                    [
                        boost_map[layer]
                        for layer in probe_layers
                    ],
                    axis=0,
                ).astype(np.float32)

                no_arr = np.stack(
                    [
                        no_map[layer]
                        for layer in probe_layers
                    ],
                    axis=0,
                ).astype(np.float32)

                delta_arr = (
                    boost_arr
                    - base_arr
                ).astype(np.float32)

                boundary_li = probe_layers.index(
                    boundary_layer
                )

                boundary_delta = delta_arr[
                    boundary_li
                ]

                previous_delta: Optional[np.ndarray] = None

                for li, layer in enumerate(probe_layers):
                    hb = base_arr[li]
                    hs = boost_arr[li]
                    d = delta_arr[li]

                    base_norm = float(
                        np.linalg.norm(hb)
                    )

                    boost_norm = float(
                        np.linalg.norm(hs)
                    )

                    delta_norm = float(
                        np.linalg.norm(d)
                    )

                    row = {
                        "sid": int(rec.sid),
                        "relation": relation,
                        "layer": layer,
                        "is_boosted_layer": int(
                            layer in set(boost_layers)
                        ),
                        "base_norm": base_norm,
                        "boost_norm": boost_norm,
                        "boost_minus_base_norm": (
                            boost_norm
                            - base_norm
                        ),
                        "delta_norm": delta_norm,
                        "relative_delta_norm": (
                            delta_norm
                            / max(base_norm, EPS)
                        ),
                        "cos_base_boost": cosine_np(
                            hb,
                            hs,
                        ),
                        "cos_delta_boundary": cosine_np(
                            d,
                            boundary_delta,
                        ),
                        "cos_delta_prev_probe": (
                            cosine_np(
                                d,
                                previous_delta,
                            )
                            if previous_delta is not None
                            else float("nan")
                        ),
                    }

                    per_sample_geometry.append(
                        row
                    )

                    previous_delta = d

                base_rows.append(base_arr)
                boost_rows.append(boost_arr)
                noimage_rows.append(no_arr)

                saved_sids.append(
                    int(rec.sid)
                )

                saved_labels.append(
                    str(relation)
                )

                del base_out
                del boost_out
                del no_out

            except Exception as exc:
                errors.append({
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

    if not base_rows:
        raise RuntimeError(
            "No samples were extracted successfully."
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

    delta = (
        boost_last
        - base_last
    ).astype(np.float32)

    base_residual = (
        base_last
        - noimage_last
    ).astype(np.float32)

    boost_residual = (
        boost_last
        - noimage_last
    ).astype(np.float32)

    labels = np.asarray(
        saved_labels,
        dtype=object,
    )

    # -------------------------------------------------------------------------
    # Aggregate geometry.
    # -------------------------------------------------------------------------
    geometry_summary: List[Dict[str, Any]] = []

    for layer in probe_layers:
        rows = [
            row
            for row in per_sample_geometry
            if int(row["layer"]) == int(layer)
        ]

        geometry_summary.append({
            "layer": layer,
            "is_boosted_layer": int(
                layer in set(boost_layers)
            ),
            "N": len(rows),
            "base_norm": safe_mean(
                row["base_norm"]
                for row in rows
            ),
            "boost_norm": safe_mean(
                row["boost_norm"]
                for row in rows
            ),
            "boost_minus_base_norm": safe_mean(
                row["boost_minus_base_norm"]
                for row in rows
            ),
            "delta_norm": safe_mean(
                row["delta_norm"]
                for row in rows
            ),
            "delta_norm_std": safe_std(
                row["delta_norm"]
                for row in rows
            ),
            "relative_delta_norm": safe_mean(
                row["relative_delta_norm"]
                for row in rows
            ),
            "cos_base_boost": safe_mean(
                row["cos_base_boost"]
                for row in rows
            ),
            "cos_delta_boundary": safe_mean(
                row["cos_delta_boundary"]
                for row in rows
            ),
            "cos_delta_prev_probe": safe_mean(
                row["cos_delta_prev_probe"]
                for row in rows
            ),
        })

    # -------------------------------------------------------------------------
    # Direction readout.
    # -------------------------------------------------------------------------
    probe_repeat_rows, probe_summary_rows = (
        direction_probe_all(
            base_residual=base_residual,
            boost_residual=boost_residual,
            delta=delta,
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

    # Merge useful direction metrics into geometry summary.
    combined_summary: List[Dict[str, Any]] = []

    for row in geometry_summary:
        layer = int(row["layer"])

        base_probe = probe_lookup[
            ("base_residual", layer)
        ]

        boost_probe = probe_lookup[
            ("boost_residual", layer)
        ]

        delta_probe = probe_lookup[
            ("boost_delta", layer)
        ]

        combined_summary.append({
            **row,
            "base_spatial_acc": base_probe[
                "accuracy_mean"
            ],
            "boost_spatial_acc": boost_probe[
                "accuracy_mean"
            ],
            "boost_minus_base_spatial_acc": (
                boost_probe[
                    "accuracy_mean"
                ]
                -
                base_probe[
                    "accuracy_mean"
                ]
            ),
            "delta_spatial_acc": delta_probe[
                "accuracy_mean"
            ],
            "delta_spatial_acc_std": delta_probe[
                "accuracy_std"
            ],
        })

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------
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
            outdir / "lasttoken_vectors.npz",
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
            boost_weight=np.asarray(
                [args.boost_weight],
                dtype=np.float32,
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
            boost_delta=delta.astype(
                np.float16
            ),
        )

    # -------------------------------------------------------------------------
    # Console.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 176)
    print(
        "HOW DOES THE LAST TOKEN CHANGE?  "
        "BASELINE vs EARLY LAST->VISUAL ATTENTION BOOST"
    )
    print("=" * 176)

    print(
        f"{'layer':>5s} | "
        f"{'||delta||':>10s} | "
        f"{'||d||/||h||':>12s} | "
        f"{'cos(base,boost)':>15s} | "
        f"{'cos(d,d@L'+str(boundary_layer)+')':>13s} | "
        f"{'base_sp':>8s} | "
        f"{'boost_sp':>8s} | "
        f"{'delta_sp':>8s}"
    )

    print("-" * 176)

    for row in combined_summary:
        layer = int(row["layer"])
        marker = "*" if int(row["is_boosted_layer"]) else " "

        print(
            f"{marker}L{layer:02d} | "
            f"{row['delta_norm']:10.4f} | "
            f"{row['relative_delta_norm']:12.5f} | "
            f"{row['cos_base_boost']:15.6f} | "
            f"{row['cos_delta_boundary']:13.6f} | "
            f"{row['base_spatial_acc']:8.4f} | "
            f"{row['boost_spatial_acc']:8.4f} | "
            f"{row['delta_spatial_acc']:8.4f}"
        )

    print("=" * 176)
    print(
        f"* = boosted layer | boost={args.attention_variant} "
        f"weight={args.boost_weight} on {boost_layers}"
    )

    print(
        "\nInterpretation keys:"
    )
    print(
        "  cos(base,boost) ~ 1 : state direction barely rotates."
    )
    print(
        "  cos(delta,delta@boundary) high after boundary : "
        "early perturbation is preserved."
    )
    print(
        "  cos(delta,delta@boundary) drops while delta_sp rises : "
        "downstream layers transform the early visual perturbation "
        "into a more relation-specific representation."
    )
    print(
        "  boost_sp - base_sp : whether total last-token spatial "
        "decodability itself improves."
    )

    config = {
        "model": args.model,
        "repo_id": MODEL_REPOS[args.model],
        "dataset": args.dataset,
        "N_requested": len(records),
        "N_success": len(saved_sids),
        "N_errors": len(errors),
        "boost_layers": boost_layers,
        "boundary_layer": boundary_layer,
        "boost_weight": args.boost_weight,
        "attention_variant": args.attention_variant,
        "probe_layers": probe_layers,
        "train_ratio": args.train_ratio,
        "repeats": args.repeats,
        "seed": args.seed,
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
        f"[saved] {outdir / 'lasttoken_geometry_per_sample.csv'}"
    )
    print(
        f"[saved] {outdir / 'direction_probe_summary.csv'}"
    )

    if args.save_vectors:
        print(
            f"[saved] {outdir / 'lasttoken_vectors.npz'}"
        )


if __name__ == "__main__":
    main()

