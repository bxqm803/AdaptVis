#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure-visual horizontal-flip causal structure scan for COCO left/right.

This script is designed to separate VISUAL GEOMETRY from QUERY ROLE.

Counterfactual
==============
Target:
    prompt  = "Where is A relative to B?"
    image   = original
    answer  = LEFT (example)

Source:
    prompt  = EXACTLY THE SAME
    image   = horizontal flip of original
    answer  = RIGHT

Thus:
    object identities          fixed
    subject/reference roles    fixed
    text/token positions       fixed
    prompt template            fixed
    image size                 fixed
    only horizontal visual geometry changes systematically

Two experiments
===============

A. FULL-FLIP RESIDUAL SCAN
--------------------------
At decoder layer L, capture source/flipped object residuals:

    s_A^L, s_B^L

Run the ORIGINAL image/prompt and replace only the object-token residual states:

    h_A^L <- s_A^L
    h_B^L <- s_B^L

then let all later layers run live.

Report source/opposite follow rate (LEFT->RIGHT or RIGHT->LEFT) for each L.

Interpretation:
    This is a causal SUFFICIENCY / precursor scan.
    It tells us when the object residual contains enough flipped-image
    information to make downstream computation produce the opposite answer.

    IMPORTANT:
    A high FULL-FLIP effect at an early layer does NOT alone prove that an
    abstract LEFT/RIGHT variable is already represented there.  It may transfer
    lower-level visual information that downstream layers later convert into
    relation.

B. FLIP-SPECIFIC DAS
--------------------
For selected layers, learn a low-dimensional orthonormal subspace U such that

    h' = h + U U^T (s_flip - h)

for the SAME object identity:

    source A -> target A
    source B -> target B

Model weights are frozen; only U is learned.

Training objective:
    change final next-token relation from clean GT to opposite(GT).

Held-out evaluation:
    clean
    learned_flip
    random_flip
    full_flip

A convincing pure-visual relation subspace should show:

    learned_flip IIA >> random_flip IIA
    held-out IIA is high
    modest D (4/8/16) is sufficient
    a clear layerwise emergence/peak
    full_flip is high enough to provide a causal upper reference

Inputs
======
Requires the output from:

    eval_coco_horizontal_flip_generation_v3.py

specifically:

    flip_generation_pairs.jsonl

The default training set is only flip BOTH-CORRECT samples:
    clean_correct == True
    flip_correct_aligned == True

This ensures the source image actually induces the intended opposite relation.

Recommended first run
=====================

# Coarse structural scan + coarse DAS:
CUDA_VISIBLE_DEVICES=0 python -u \
validate_coco_horizontal_flip_relation_das_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --flip-pairs-jsonl \
    output/qwen3b_coco_horizontal_flip_generation_v1/flip_generation_pairs.jsonl \
  --full-scan-layers 0-30 \
  --das-layers 0,4,8,12,16,20,24,28,30 \
  --subspace-dims 8,16 \
  --train-size 48 \
  --epochs 5 \
  --eval-max-samples 0 \
  --device cuda:0 \
  --output-dir output/qwen3b_horizontal_flip_relation_das_v1 \
  --overwrite

Smoke
=====

CUDA_VISIBLE_DEVICES=0 python -u \
validate_coco_horizontal_flip_relation_das_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --flip-pairs-jsonl \
    output/qwen3b_coco_horizontal_flip_generation_v1/flip_generation_pairs.jsonl \
  --full-scan-layers 0,4,8,12,16,20,24,28 \
  --das-layers 8,16,24 \
  --subspace-dims 8 \
  --train-size 24 \
  --epochs 2 \
  --eval-max-samples 64 \
  --device cuda:0 \
  --output-dir output/qwen3b_horizontal_flip_relation_das_smoke \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import json
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


VERSION = "coco-horizontal-flip-relation-das-v1"
RELATIONS = ("left", "right", "above", "below")
LR = ("left", "right")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--object-state", default="mean", choices=("mean", "last"))

    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
        help="Existing extraction.jsonl/config.json from the COCO pipeline.",
    )
    p.add_argument(
        "--flip-pairs-jsonl",
        required=True,
        help="flip_generation_pairs.jsonl from eval_coco_horizontal_flip_generation_v3.py",
    )

    p.add_argument(
        "--full-scan-layers",
        default="0-30",
        help="Layer spec, e.g. 0-30 or 0,2,4,... Full residual flip scan.",
    )
    p.add_argument(
        "--das-layers",
        default="0,4,8,12,16,20,24,28,30",
        help="Layers where a learned low-D flip subspace is trained.",
    )
    p.add_argument(
        "--subspace-dims",
        default="8,16",
    )

    p.add_argument(
        "--train-flip-status",
        default="both_correct",
        choices=("both_correct", "clean_correct", "flip_correct", "all"),
    )
    p.add_argument("--train-size", type=int, default=48)
    p.add_argument("--train-ratio", type=float, default=0.25)
    p.add_argument("--eval-max-samples", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=13)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--logit-temperature", type=float, default=1.0)

    p.add_argument("--random-repeats", type=int, default=1)
    p.add_argument(
        "--source-cache-dtype",
        default="float16",
        choices=("float16", "float32"),
    )

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--empty-cache-every", type=int, default=8)

    p.add_argument(
        "--das-helper",
        default="validate_coco_relation_das_v1.py",
        help="Reuses tested residual hooks / relation-logit helpers.",
    )
    p.add_argument(
        "--ioi-script",
        default="analyze_coco_ioi_backward_circuit_v1.py",
    )
    p.add_argument(
        "--producer-script",
        default="analyze_coco_producer_qk_ov_v1.py",
    )
    p.add_argument(
        "--receiver-script",
        default="analyze_coco_receiver_qkv_v1.py",
    )
    p.add_argument(
        "--v3-script",
        default="analyze_spatial_storage_transport_utilization_v3.py",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================

def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_int_list(text: str) -> List[int]:
    vals = []
    seen = set()
    for piece in str(text).split(","):
        piece = piece.strip().upper().replace("L", "")
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            aa, bb = int(a), int(b)
            seq = range(min(aa, bb), max(aa, bb) + 1)
        else:
            seq = [int(piece)]
        for v in seq:
            if v not in seen:
                seen.add(v)
                vals.append(v)
    if not vals:
        raise ValueError(f"Empty layer list: {text}")
    return sorted(vals)


def safe_mean(xs: Iterable[float]) -> float:
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std()) if a.size else float("nan")


def stratified_take(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in rows]
    if n <= 0 or n >= len(rows):
        return rows

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["gt"]), []).append(r)
    for g in groups.values():
        rng.shuffle(g)

    keys = sorted(groups)
    out = []
    idx = {k: 0 for k in keys}

    while len(out) < n:
        progressed = False
        for k in keys:
            i = idx[k]
            if i < len(groups[k]) and len(out) < n:
                out.append(groups[k][i])
                idx[k] += 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(out)
    return out


# =============================================================================
# Flip image / batch helpers
# =============================================================================

def try_get_attr(obj: Any, names: Sequence[str]) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def try_get_mapping(mp: Mapping[str, Any], names: Sequence[str]) -> Any:
    for n in names:
        if n in mp and mp[n] is not None:
            return mp[n]
    return None


def infer_image(
    row: Mapping[str, Any],
    records_by_sid: Mapping[int, Any],
    pair: Any,
) -> Image.Image:
    sid = int(row["sid"])
    rec = records_by_sid.get(sid, {})

    cand = try_get_attr(
        pair,
        ["image", "original_image", "image_path", "original_image_path", "img_path"],
    )

    if isinstance(cand, Image.Image):
        return cand.convert("RGB")

    if cand is None:
        cand = try_get_mapping(
            row,
            ["image", "image_path", "img_path", "file_name", "filename"],
        )

    if isinstance(cand, Image.Image):
        return cand.convert("RGB")

    if cand is None and isinstance(rec, Mapping):
        cand = try_get_mapping(
            rec,
            ["image", "image_path", "img_path", "file_name", "filename"],
        )

    if isinstance(cand, Image.Image):
        return cand.convert("RGB")

    if cand is None:
        raise KeyError(
            f"Could not infer image for sid={sid}; "
            f"pair attrs={list(getattr(pair, '__dict__', {}).keys())}"
        )

    p = Path(str(cand))
    if not p.exists():
        raise FileNotFoundError(p)
    return Image.open(p).convert("RGB")


def horizontal_flip(img: Image.Image) -> Image.Image:
    try:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except AttributeError:
        return img.transpose(Image.FLIP_LEFT_RIGHT)


def image_only_processor(processor: Any, image: Image.Image) -> Mapping[str, Any]:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        raise RuntimeError("processor.image_processor missing")

    errors = []
    for kwargs in (
        {"images": [image], "return_tensors": "pt"},
        {"images": image, "return_tensors": "pt"},
    ):
        try:
            return ip(**kwargs)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    raise RuntimeError(
        "image_processor failed:\n" + "\n".join(errors)
    )


def build_flipped_batch(
    *,
    processor: Any,
    original_batch: Mapping[str, Any],
    flipped_image: Image.Image,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Preserve the exact original prompt/chat-template tokens and replace only
    visual tensors.
    """
    batch: Dict[str, Any] = {}
    for k, v in original_batch.items():
        batch[k] = v.clone() if torch.is_tensor(v) else v

    visual = image_only_processor(processor, flipped_image)

    visual_names = {
        "pixel_values",
        "pixel_values_images",
        "image_grid_thw",
        "image_sizes",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
    }

    replaced = []
    for k, v in visual.items():
        if k in batch or k in visual_names:
            batch[k] = v.to(device) if torch.is_tensor(v) else v
            replaced.append(k)

    if not replaced:
        raise RuntimeError(
            f"No visual tensors replaced. "
            f"original={list(original_batch.keys())}, visual={list(visual.keys())}"
        )

    # Exact prompt preservation.
    for k in ("input_ids", "attention_mask"):
        if k in original_batch and k in batch:
            if torch.is_tensor(original_batch[k]) and torch.is_tensor(batch[k]):
                if not torch.equal(original_batch[k], batch[k]):
                    raise RuntimeError(f"{k} changed in flipped batch")

    # Horizontal flip preserves input resolution, therefore Qwen image grid
    # should stay exactly the same.
    if "image_grid_thw" in original_batch and "image_grid_thw" in batch:
        a = original_batch["image_grid_thw"]
        b = batch["image_grid_thw"]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a.to(b.device), b):
                raise RuntimeError(
                    "image_grid_thw changed after horizontal flip; "
                    "cannot guarantee identical text/image-token layout."
                )

    return batch


# =============================================================================
# Dataset merge / split
# =============================================================================

def load_flip_rows(
    *,
    extraction_jsonl: Path,
    flip_pairs_jsonl: Path,
) -> List[Dict[str, Any]]:
    extraction = {
        int(r["sid"]): dict(r)
        for r in read_jsonl(extraction_jsonl)
        if str(r.get("gt")) in LR
    }
    flips = {
        int(r["sid"]): dict(r)
        for r in read_jsonl(flip_pairs_jsonl)
    }

    rows = []
    for sid in sorted(set(extraction) & set(flips)):
        r = dict(extraction[sid])
        f = flips[sid]

        # Safety: flip-generation file and extraction must agree on clean GT.
        if str(f.get("gt")) != str(r.get("gt")):
            raise RuntimeError(
                f"GT mismatch sid={sid}: extraction={r.get('gt')} flip={f.get('gt')}"
            )

        r["flip_clean_prediction"] = str(f.get("clean_prediction", ""))
        r["flip_prediction"] = str(f.get("flip_prediction", ""))
        r["flip_clean_correct"] = bool(f.get("clean_correct", False))
        r["flip_correct_aligned"] = bool(f.get("flip_correct_aligned", False))
        r["flip_prediction_opposes"] = bool(f.get("prediction_opposes", False))
        r["flip_both_correct"] = (
            r["flip_clean_correct"] and r["flip_correct_aligned"]
        )
        rows.append(r)

    if not rows:
        raise RuntimeError(
            "No common left/right SIDs between extraction and flip-pairs JSONL"
        )
    return rows


def eligible_train(row: Mapping[str, Any], status: str) -> bool:
    if status == "all":
        return True
    if status == "both_correct":
        return bool(row["flip_both_correct"])
    if status == "clean_correct":
        return bool(row["flip_clean_correct"])
    if status == "flip_correct":
        return bool(row["flip_correct_aligned"])
    raise ValueError(status)


def build_split(
    *,
    rows: Sequence[Mapping[str, Any]],
    train_status: str,
    train_size: int,
    train_ratio: float,
    eval_max_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eligible = [dict(r) for r in rows if eligible_train(r, train_status)]
    if not eligible:
        raise RuntimeError(f"No rows for train status={train_status}")

    if train_size > 0:
        n_train = min(train_size, len(eligible))
    else:
        n_train = max(4, int(round(len(eligible) * train_ratio)))

    train = stratified_take(eligible, n_train, seed)
    train_sids = {int(r["sid"]) for r in train}

    eval_pool = [dict(r) for r in rows if int(r["sid"]) not in train_sids]
    eval_rows = stratified_take(eval_pool, eval_max_samples, seed + 701)

    return train, eval_rows


# =============================================================================
# Source cache: SAME prompt, flipped image, SAME A/B token positions
# =============================================================================

def build_flip_source_cache(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    das: Any,
    storage_dtype: np.dtype,
    error_path: Path,
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:

    device = torch.device(args.device)
    cache: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    print("\nCaching H-FLIP source object residuals...", flush=True)

    for i, row in enumerate(
        tqdm(rows, desc="flip-source-cache"),
        start=1,
    ):
        pair = None
        try:
            pair = receiver.prepare_pair(
                args=args,
                row=row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                processor=processor,
                device=device,
            )

            img = infer_image(row, records_by_sid, pair)
            flip_img = horizontal_flip(img)

            flip_batch = build_flipped_batch(
                processor=processor,
                original_batch=pair.original_batch,
                flipped_image=flip_img,
                device=device,
            )

            # Prompt is identical, so object token positions are exactly the
            # ORIGINAL A/B positions.  No B->A role alignment is used.
            values = das.capture_source_layers(
                model=model,
                swapped_batch=flip_batch,
                decoder_layers=decoder_layers,
                layers=layers,
                swapped_a_positions=pair.original_a_positions,
                swapped_b_positions=pair.original_b_positions,
                storage_dtype=storage_dtype,
            )
            cache[int(pair.sid)] = values

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "flip_source_cache",
                "sid": int(row["sid"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            if args.fail_fast:
                raise
        finally:
            if pair is not None:
                receiver.release_pair(pair)

            if (
                torch.cuda.is_available()
                and args.empty_cache_every > 0
                and i % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    return cache


# =============================================================================
# Intervened forwards
# =============================================================================

def forward_flip_train(
    *,
    model: Any,
    batch: Mapping[str, Any],
    decoder_layer: Any,
    target_a_positions: Sequence[int],
    target_b_positions: Sequence[int],
    source_state: Mapping[str, np.ndarray],
    basis_parameter: torch.Tensor,
    source_label_id: int,
    relation_token_map: Mapping[str, Sequence[int]],
    temperature: float,
    das: Any,
) -> Tuple[torch.Tensor, str, np.ndarray]:

    # "identity" here means source A->target A, source B->target B.
    # This is the MAIN pure-visual counterfactual, not a control.
    with das.ResidualSubspaceIntervention(
        decoder_layer=decoder_layer,
        target_a_positions=target_a_positions,
        target_b_positions=target_b_positions,
        source_state=source_state,
        alignment="identity",
        basis=basis_parameter,
        full_replace=False,
    ):
        out = model(
            **batch,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    scores = das.relation_class_logits(
        out.logits[0, -1],
        relation_token_map,
    )
    target = torch.tensor(
        [int(source_label_id)],
        device=scores.device,
        dtype=torch.long,
    )
    loss = F.cross_entropy(
        (scores / float(temperature))[None, :],
        target,
    )

    pred_id = int(scores.detach().argmax().item())
    pred = ID_TO_REL[pred_id]
    arr = scores.detach().cpu().numpy().astype(np.float32)
    del out
    return loss, pred, arr


@torch.inference_mode()
def forward_flip_eval(
    *,
    model: Any,
    batch: Mapping[str, Any],
    decoder_layer: Any,
    target_a_positions: Sequence[int],
    target_b_positions: Sequence[int],
    source_state: Mapping[str, np.ndarray],
    relation_token_map: Mapping[str, Sequence[int]],
    condition: str,
    learned_basis: Optional[torch.Tensor],
    random_basis: Optional[torch.Tensor],
    das: Any,
) -> Dict[str, Any]:

    if condition == "clean":
        return das.forward_clean(
            model=model,
            batch=batch,
            relation_token_map=relation_token_map,
        )

    if condition == "learned_flip":
        basis = learned_basis
        full = False
    elif condition == "random_flip":
        basis = random_basis
        full = False
    elif condition == "full_flip":
        basis = None
        full = True
    else:
        raise ValueError(condition)

    with das.ResidualSubspaceIntervention(
        decoder_layer=decoder_layer,
        target_a_positions=target_a_positions,
        target_b_positions=target_b_positions,
        source_state=source_state,
        alignment="identity",  # SAME object identity; prompt role fixed.
        basis=basis,
        full_replace=full,
    ):
        out = model(
            **batch,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    pred, scores = das.relation_prediction(
        out.logits[0, -1],
        relation_token_map,
    )
    del out
    return {
        "prediction": pred,
        "scores": scores,
    }


# =============================================================================
# Metrics
# =============================================================================

def summarize_predictions(
    *,
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    scores: Sequence[np.ndarray],
    clean_predictions: Sequence[str],
) -> Dict[str, Any]:

    n = len(rows)
    target_hits = 0
    source_hits = 0
    other_hits = 0
    changes = 0
    margins = []
    clean_correct_to_source = 0
    clean_correct_n = 0

    for row, pred, sc, clean_pred in zip(
        rows, predictions, scores, clean_predictions
    ):
        gt = str(row["gt"])
        src = OPPOSITE[gt]

        target_hits += int(pred == gt)
        source_hits += int(pred == src)
        other_hits += int(pred not in (gt, src))
        changes += int(pred != clean_pred)

        s = np.asarray(sc, dtype=np.float64)
        margins.append(
            float(s[REL_TO_ID[src]] - s[REL_TO_ID[gt]])
        )

        if clean_pred == gt:
            clean_correct_n += 1
            clean_correct_to_source += int(pred == src)

    return {
        "condition": condition,
        "N": n,
        "target_accuracy": target_hits / n if n else float("nan"),
        "source_follow_iia": source_hits / n if n else float("nan"),
        "other_rate": other_hits / n if n else float("nan"),
        "prediction_change_vs_clean": changes / n if n else float("nan"),
        "clean_correct_to_source_rate": (
            clean_correct_to_source / clean_correct_n
            if clean_correct_n else float("nan")
        ),
        "source_minus_target_margin_mean": safe_mean(margins),
        "source_minus_target_margin_std": safe_std(margins),
    }


# =============================================================================
# FULL-FLIP scan
# =============================================================================

def run_full_scan(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    source_cache: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    das: Any,
    error_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    device = torch.device(args.device)
    summary_rows = []
    sample_rows = []

    for layer in layers:
        clean_preds = []
        clean_scores = []
        full_preds = []
        full_scores = []
        successful = []

        print(f"\n>>> FULL-FLIP L{layer}", flush=True)

        for i, row in enumerate(
            tqdm(rows, desc=f"full-flip:L{layer}"),
            start=1,
        ):
            sid = int(row["sid"])
            if sid not in source_cache:
                continue

            pair = None
            try:
                pair = receiver.prepare_pair(
                    args=args,
                    row=row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=device,
                )

                clean = forward_flip_eval(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[sid][layer],
                    relation_token_map=relation_token_map,
                    condition="clean",
                    learned_basis=None,
                    random_basis=None,
                    das=das,
                )

                full = forward_flip_eval(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[sid][layer],
                    relation_token_map=relation_token_map,
                    condition="full_flip",
                    learned_basis=None,
                    random_basis=None,
                    das=das,
                )

                clean_preds.append(str(clean["prediction"]))
                clean_scores.append(np.asarray(clean["scores"], dtype=np.float32))
                full_preds.append(str(full["prediction"]))
                full_scores.append(np.asarray(full["scores"], dtype=np.float32))
                successful.append(dict(row))

                sample_rows.append({
                    "layer": layer,
                    "sid": sid,
                    "gt": str(row["gt"]),
                    "source_gt": OPPOSITE[str(row["gt"])],
                    "flip_both_correct": bool(row["flip_both_correct"]),
                    "clean_prediction": str(clean["prediction"]),
                    "full_flip_prediction": str(full["prediction"]),
                    "full_flip_source_follow": (
                        str(full["prediction"]) == OPPOSITE[str(row["gt"])]
                    ),
                })

            except Exception as exc:
                append_jsonl(error_path, {
                    "phase": "full_scan",
                    "layer": layer,
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver.release_pair(pair)

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and i % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        subset_specs = [
            ("heldout_all", list(range(len(successful)))),
            (
                "heldout_flip_both_correct",
                [
                    i for i, r in enumerate(successful)
                    if bool(r["flip_both_correct"])
                ],
            ),
        ]

        for subset, indices in subset_specs:
            if not indices:
                continue
            rs = [successful[i] for i in indices]
            cp = [clean_preds[i] for i in indices]
            cs = [clean_scores[i] for i in indices]
            fp = [full_preds[i] for i in indices]
            fs = [full_scores[i] for i in indices]

            clean_summary = summarize_predictions(
                condition="clean",
                rows=rs,
                predictions=cp,
                scores=cs,
                clean_predictions=cp,
            )
            clean_summary.update({
                "version": VERSION,
                "layer": layer,
                "eval_subset": subset,
            })
            summary_rows.append(clean_summary)

            full_summary = summarize_predictions(
                condition="full_flip",
                rows=rs,
                predictions=fp,
                scores=fs,
                clean_predictions=cp,
            )
            full_summary.update({
                "version": VERSION,
                "layer": layer,
                "eval_subset": subset,
            })
            summary_rows.append(full_summary)

    return summary_rows, sample_rows


# =============================================================================
# DAS training
# =============================================================================

@dataclass
class TrainResult:
    q_basis_cpu: np.ndarray
    history: List[Dict[str, Any]]
    train_source_follow: float
    train_loss: float


def train_das(
    *,
    args: argparse.Namespace,
    layer: int,
    dim: int,
    train_rows: Sequence[Mapping[str, Any]],
    source_cache: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    das: Any,
    d_model: int,
    error_path: Path,
) -> TrainResult:

    device = torch.device(args.device)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed + 10007 * layer + 131 * dim)
    init = torch.randn(d_model, dim, generator=g, dtype=torch.float32)
    init, _ = torch.linalg.qr(init, mode="reduced")
    raw_basis = torch.nn.Parameter(init.to(device=device))

    optimizer = torch.optim.AdamW(
        [raw_basis],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history = []
    rng = random.Random(args.seed + 9001 * layer + 17 * dim)

    for epoch in range(1, args.epochs + 1):
        order = [
            dict(r)
            for r in train_rows
            if int(r["sid"]) in source_cache
        ]
        rng.shuffle(order)

        optimizer.zero_grad(set_to_none=True)
        losses = []
        source_hits = 0
        target_hits = 0
        n = 0

        for step, row in enumerate(
            tqdm(
                order,
                desc=f"train-flip:L{layer}:D{dim}:E{epoch}",
                leave=False,
            ),
            start=1,
        ):
            pair = None
            try:
                pair = receiver.prepare_pair(
                    args=args,
                    row=row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=device,
                )

                gt = str(pair.gt)
                src = OPPOSITE[gt]

                loss, pred, _ = forward_flip_train(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[int(pair.sid)][layer],
                    basis_parameter=raw_basis,
                    source_label_id=REL_TO_ID[src],
                    relation_token_map=relation_token_map,
                    temperature=args.logit_temperature,
                    das=das,
                )

                (loss / max(1, args.grad_accum)).backward()
                losses.append(float(loss.detach().item()))
                source_hits += int(pred == src)
                target_hits += int(pred == gt)
                n += 1

                if (
                    step % max(1, args.grad_accum) == 0
                    or step == len(order)
                ):
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [raw_basis],
                            float(args.max_grad_norm),
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            except Exception as exc:
                append_jsonl(error_path, {
                    "phase": "train_das",
                    "layer": layer,
                    "dim": dim,
                    "epoch": epoch,
                    "sid": int(row["sid"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver.release_pair(pair)

        erow = {
            "layer": layer,
            "dim": dim,
            "epoch": epoch,
            "N": n,
            "loss": safe_mean(losses),
            "train_source_follow": source_hits / n if n else float("nan"),
            "train_target_accuracy": target_hits / n if n else float("nan"),
        }
        history.append(erow)

        print(
            f"[flip-DAS] L{layer} D{dim} E{epoch} "
            f"N={n} loss={erow['loss']:.4f} "
            f"src={100*erow['train_source_follow']:.2f}% "
            f"tgt={100*erow['train_target_accuracy']:.2f}%",
            flush=True,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with torch.no_grad():
        q = das.orthonormal_basis(raw_basis).detach().cpu().numpy().astype(np.float32)

    last = history[-1]
    return TrainResult(
        q_basis_cpu=q,
        history=history,
        train_source_follow=float(last["train_source_follow"]),
        train_loss=float(last["loss"]),
    )


# =============================================================================
# DAS held-out evaluation
# =============================================================================

def evaluate_das(
    *,
    args: argparse.Namespace,
    layer: int,
    dim: int,
    q_basis_cpu: np.ndarray,
    eval_rows: Sequence[Mapping[str, Any]],
    source_cache: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    das: Any,
    error_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    device = torch.device(args.device)
    learned = torch.as_tensor(
        q_basis_cpu,
        device=device,
        dtype=torch.float32,
    )

    random_bases = [
        das.random_orthonormal_basis(
            d_model=int(q_basis_cpu.shape[0]),
            k=dim,
            seed=args.seed + 17011 * layer + 733 * dim + rr,
            device=device,
        )
        for rr in range(max(1, args.random_repeats))
    ]

    conditions = ("clean", "learned_flip", "random_flip", "full_flip")

    preds = {c: [] for c in conditions}
    scores = {c: [] for c in conditions}
    successful = []
    samples = []

    for i, row in enumerate(
        tqdm(eval_rows, desc=f"eval-flip-DAS:L{layer}:D{dim}"),
        start=1,
    ):
        sid = int(row["sid"])
        if sid not in source_cache:
            continue

        pair = None
        try:
            pair = receiver.prepare_pair(
                args=args,
                row=row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                processor=processor,
                device=device,
            )

            result_by_condition = {}
            for cond in conditions:
                result = forward_flip_eval(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[sid][layer],
                    relation_token_map=relation_token_map,
                    condition=cond,
                    learned_basis=learned if cond == "learned_flip" else None,
                    random_basis=random_bases[0] if cond == "random_flip" else None,
                    das=das,
                )
                preds[cond].append(str(result["prediction"]))
                scores[cond].append(np.asarray(result["scores"], dtype=np.float32))
                result_by_condition[cond] = str(result["prediction"])

            successful.append(dict(row))

            samples.append({
                "sid": sid,
                "layer": layer,
                "dim": dim,
                "gt": str(row["gt"]),
                "source_gt": OPPOSITE[str(row["gt"])],
                "flip_both_correct": bool(row["flip_both_correct"]),
                **{f"pred_{k}": v for k, v in result_by_condition.items()},
            })

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "eval_das",
                "layer": layer,
                "dim": dim,
                "sid": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            if args.fail_fast:
                raise
        finally:
            if pair is not None:
                receiver.release_pair(pair)

            if (
                torch.cuda.is_available()
                and args.empty_cache_every > 0
                and i % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    if not successful:
        raise RuntimeError(f"No successful eval rows L{layer} D{dim}")

    clean_preds = preds["clean"]

    subset_specs = [
        ("heldout_all", list(range(len(successful)))),
        (
            "heldout_flip_both_correct",
            [
                i for i, r in enumerate(successful)
                if bool(r["flip_both_correct"])
            ],
        ),
    ]

    summary = []

    for subset, indices in subset_specs:
        if not indices:
            continue

        rs = [successful[i] for i in indices]
        clean = [clean_preds[i] for i in indices]

        for cond in conditions:
            row = summarize_predictions(
                condition=cond,
                rows=rs,
                predictions=[preds[cond][i] for i in indices],
                scores=[scores[cond][i] for i in indices],
                clean_predictions=clean,
            )
            row.update({
                "version": VERSION,
                "layer": layer,
                "dim": dim,
                "eval_subset": subset,
                "random_repeat": 0 if cond == "random_flip" else "",
            })
            summary.append(row)

    # Optional extra random subspaces.
    for rr in range(1, len(random_bases)):
        rp = []
        rscores = []

        for row in tqdm(
            successful,
            desc=f"random{rr}:L{layer}:D{dim}",
            leave=False,
        ):
            sid = int(row["sid"])
            pair = None
            try:
                pair = receiver.prepare_pair(
                    args=args,
                    row=row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=device,
                )

                result = forward_flip_eval(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[sid][layer],
                    relation_token_map=relation_token_map,
                    condition="random_flip",
                    learned_basis=None,
                    random_basis=random_bases[rr],
                    das=das,
                )
                rp.append(str(result["prediction"]))
                rscores.append(np.asarray(result["scores"], dtype=np.float32))
            finally:
                if pair is not None:
                    receiver.release_pair(pair)

        for subset, indices in subset_specs:
            if not indices:
                continue
            subset_rows = [successful[i] for i in indices]
            clean = [clean_preds[i] for i in indices]
            row = summarize_predictions(
                condition="random_flip",
                rows=subset_rows,
                predictions=[rp[i] for i in indices],
                scores=[rscores[i] for i in indices],
                clean_predictions=clean,
            )
            row.update({
                "version": VERSION,
                "layer": layer,
                "dim": dim,
                "eval_subset": subset,
                "random_repeat": rr,
            })
            summary.append(row)

    return summary, samples


# =============================================================================
# Reporting
# =============================================================================

def print_full_scan(rows: Sequence[Mapping[str, Any]]) -> None:
    vals = [
        r for r in rows
        if r["condition"] == "full_flip"
        and r["eval_subset"] == "heldout_flip_both_correct"
    ]

    print("\n" + "=" * 116)
    print("PURE-VISUAL FULL-FLIP RESIDUAL CAUSAL SCAN")
    print("=" * 116)
    print(
        f"{'L':>4s} {'N':>5s} {'target':>9s} {'SOURCE/IIA':>11s} "
        f"{'change':>9s} {'cleanGT->src':>13s} {'src-tgt margin':>14s}"
    )
    print("-" * 116)

    for r in sorted(vals, key=lambda x: int(x["layer"])):
        print(
            f"L{int(r['layer']):<3d} "
            f"{int(r['N']):>5d} "
            f"{100*float(r['target_accuracy']):>8.2f}% "
            f"{100*float(r['source_follow_iia']):>10.2f}% "
            f"{100*float(r['prediction_change_vs_clean']):>8.2f}% "
            f"{100*float(r['clean_correct_to_source_rate']):>12.2f}% "
            f"{float(r['source_minus_target_margin_mean']):>+13.4f}"
        )


def print_das_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    vals = [
        r for r in rows
        if r["eval_subset"] == "heldout_flip_both_correct"
        and str(r.get("random_repeat", "")) in ("", "0")
    ]

    print("\n" + "=" * 136)
    print("PURE-VISUAL H-FLIP DAS — HELD-OUT BOTH-CORRECT")
    print("=" * 136)
    print(
        f"{'L':>4s} {'D':>4s} {'condition':<15s} {'N':>5s} "
        f"{'target':>9s} {'SOURCE/IIA':>11s} {'change':>9s} "
        f"{'cleanGT->src':>13s} {'margin':>11s}"
    )
    print("-" * 136)

    for r in sorted(
        vals,
        key=lambda x: (
            int(x["layer"]),
            int(x["dim"]),
            str(x["condition"]),
        ),
    ):
        print(
            f"L{int(r['layer']):<3d} "
            f"{int(r['dim']):>4d} "
            f"{str(r['condition']):<15s} "
            f"{int(r['N']):>5d} "
            f"{100*float(r['target_accuracy']):>8.2f}% "
            f"{100*float(r['source_follow_iia']):>10.2f}% "
            f"{100*float(r['prediction_change_vs_clean']):>8.2f}% "
            f"{100*float(r['clean_correct_to_source_rate']):>12.2f}% "
            f"{float(r['source_minus_target_margin_mean']):>+10.4f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.train_size == 0 and not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.epochs < 1:
        raise ValueError("--epochs must be >=1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >=1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output dir not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    error_path = out_dir / "errors.jsonl"

    das = import_file(Path(args.das_helper), "_hflip_das_helper")
    ioi = import_file(Path(args.ioi_script), "_hflip_ioi")
    producer = import_file(Path(args.producer_script), "_hflip_producer")
    receiver = import_file(Path(args.receiver_script), "_hflip_receiver")
    v3 = import_file(Path(args.v3_script), "_hflip_v3")
    base = import_file(Path(args.base_script), "_hflip_base")

    source_dir = Path(args.source_output_dir)
    extraction_path = source_dir / "extraction.jsonl"
    config_path = source_dir / "config.json"

    if not extraction_path.exists():
        raise FileNotFoundError(extraction_path)

    flip_path = Path(args.flip_pairs_jsonl)
    if not flip_path.exists():
        raise FileNotFoundError(flip_path)

    rows = load_flip_rows(
        extraction_jsonl=extraction_path,
        flip_pairs_jsonl=flip_path,
    )

    train_rows, eval_rows = build_split(
        rows=rows,
        train_status=args.train_flip_status,
        train_size=args.train_size,
        train_ratio=args.train_ratio,
        eval_max_samples=args.eval_max_samples,
        seed=args.split_seed,
    )

    # Full scan is held-out only: no train leakage in the reported main curve.
    full_scan_rows = list(eval_rows)

    union = {}
    for r in list(train_rows) + list(eval_rows):
        union[int(r["sid"])] = dict(r)
    source_rows = sorted(union.values(), key=lambda r: int(r["sid"]))

    full_layers = parse_int_list(args.full_scan_layers)
    das_layers = parse_int_list(args.das_layers)
    cache_layers = sorted(set(full_layers + das_layers))
    dims = parse_int_list(args.subspace_dims)

    model = processor = None

    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )

        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        for layer in cache_layers:
            if not 0 <= layer < len(decoder_layers):
                raise ValueError(
                    f"L{layer} outside decoder range 0..{len(decoder_layers)-1}"
                )

        saved_max = getattr(args, "max_samples", None)
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(
                args,
                base,
            )
        finally:
            args.max_samples = saved_max

        dtype = np.float16 if args.source_cache_dtype == "float16" else np.float32

        source_cache = build_flip_source_cache(
            args=args,
            rows=source_rows,
            layers=cache_layers,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            records_by_sid=records_by_sid,
            prompt_rows=prompt_rows,
            base=base,
            v3=v3,
            receiver=receiver,
            das=das,
            storage_dtype=dtype,
            error_path=error_path,
        )

        train_rows = [r for r in train_rows if int(r["sid"]) in source_cache]
        eval_rows = [r for r in eval_rows if int(r["sid"]) in source_cache]
        full_scan_rows = [r for r in full_scan_rows if int(r["sid"]) in source_cache]

        if not train_rows or not eval_rows:
            raise RuntimeError("Train/eval empty after flip source-cache failures")

        first_sid = int(train_rows[0]["sid"])
        first_layer = cache_layers[0]
        d_model = int(np.asarray(source_cache[first_sid][first_layer]["A"]).shape[0])

        config = {
            "version": VERSION,
            "model": args.model,
            "repo_id": getattr(spec, "repo_id", ""),
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "d_model": d_model,
            "counterfactual": "same prompt + horizontally flipped image",
            "object_mapping": "source flip A->target A; source flip B->target B",
            "role_swap": False,
            "relations": ["left", "right"],
            "full_scan_layers": full_layers,
            "das_layers": das_layers,
            "subspace_dims": dims,
            "train_flip_status": args.train_flip_status,
            "N_flip_pairs_total": len(rows),
            "N_train": len(train_rows),
            "N_eval": len(eval_rows),
            "N_eval_flip_both_correct": sum(bool(r["flip_both_correct"]) for r in eval_rows),
            "train_sids": [int(r["sid"]) for r in train_rows],
            "eval_sids": [int(r["sid"]) for r in eval_rows],
            "intervention": "h' = h + QQ^T(s_flip-h), A->A, B->B",
            "audit": audit,
        }
        write_json(out_dir / "config.json", config)

        print("\n" + "=" * 126)
        print("PURE-VISUAL HORIZONTAL-FLIP CAUSAL STRUCTURE")
        print("=" * 126)
        print("model                    :", args.model)
        print("N matched flip pairs     :", len(rows))
        print("N train                  :", len(train_rows))
        print("N heldout                :", len(eval_rows))
        print("heldout flip-both-correct:", sum(bool(r["flip_both_correct"]) for r in eval_rows))
        print("full scan layers         :", full_layers)
        print("DAS layers               :", das_layers)
        print("DAS dims                 :", dims)
        print("mapping                  : flip A->A, flip B->B")
        print("query role swap          : NO")
        print("=" * 126, flush=True)

        # ------------------------------------------------------------------
        # A. Full residual causal precursor scan
        # ------------------------------------------------------------------
        full_summary, full_samples = run_full_scan(
            args=args,
            rows=full_scan_rows,
            layers=full_layers,
            source_cache=source_cache,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            relation_token_map=relation_token_map,
            records_by_sid=records_by_sid,
            prompt_rows=prompt_rows,
            base=base,
            v3=v3,
            receiver=receiver,
            das=das,
            error_path=error_path,
        )

        write_csv(out_dir / "full_scan.csv", full_summary)
        for r in full_samples:
            append_jsonl(out_dir / "full_scan_samples.jsonl", r)

        print_full_scan(full_summary)

        # ------------------------------------------------------------------
        # B. Low-dimensional pure-visual DAS
        # ------------------------------------------------------------------
        all_das_summary = []
        all_history = []

        for layer in das_layers:
            for dim in dims:
                if dim > d_model:
                    raise ValueError(f"D{dim} > hidden size {d_model}")

                print(f"\n>>> TRAIN PURE-VISUAL DAS L{layer} D{dim}", flush=True)

                result = train_das(
                    args=args,
                    layer=layer,
                    dim=dim,
                    train_rows=train_rows,
                    source_cache=source_cache,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    receiver=receiver,
                    das=das,
                    d_model=d_model,
                    error_path=error_path,
                )

                all_history.extend(result.history)

                np.savez_compressed(
                    out_dir / f"basis_L{layer}_D{dim}.npz",
                    Q=result.q_basis_cpu,
                    layer=np.asarray([layer], dtype=np.int32),
                    dim=np.asarray([dim], dtype=np.int32),
                    counterfactual=np.asarray(["horizontal_flip_same_prompt"]),
                )

                print(f">>> EVAL PURE-VISUAL DAS L{layer} D{dim}", flush=True)

                summary, samples = evaluate_das(
                    args=args,
                    layer=layer,
                    dim=dim,
                    q_basis_cpu=result.q_basis_cpu,
                    eval_rows=eval_rows,
                    source_cache=source_cache,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    receiver=receiver,
                    das=das,
                    error_path=error_path,
                )

                for r in summary:
                    r["train_source_follow_last_epoch"] = result.train_source_follow
                    r["train_loss_last_epoch"] = result.train_loss

                all_das_summary.extend(summary)

                for r in samples:
                    append_jsonl(
                        out_dir / f"samples_L{layer}_D{dim}.jsonl",
                        r,
                    )

                write_csv(out_dir / "das_summary.csv", all_das_summary)
                write_csv(out_dir / "train_history.csv", all_history)

                print_das_summary(summary)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print_das_summary(all_das_summary)

        # Best learned pure-visual settings.
        best = [
            dict(r)
            for r in all_das_summary
            if r["condition"] == "learned_flip"
            and r["eval_subset"] == "heldout_flip_both_correct"
            and str(r.get("random_repeat", "")) in ("", "0")
        ]
        best.sort(
            key=lambda r: float(r["source_follow_iia"]),
            reverse=True,
        )

        print("\n" + "=" * 110)
        print("BEST HELD-OUT PURE-VISUAL DAS SETTINGS")
        print("=" * 110)

        for rank, r in enumerate(best[:12], start=1):
            print(
                f"{rank:02d}. "
                f"L{int(r['layer'])} D{int(r['dim'])} "
                f"IIA={100*float(r['source_follow_iia']):.2f}% "
                f"change={100*float(r['prediction_change_vs_clean']):.2f}% "
                f"margin={float(r['source_minus_target_margin_mean']):+.4f}"
            )

        report = [
            f"version: {VERSION}",
            f"model: {args.model}",
            f"N_train: {len(train_rows)}",
            f"N_eval: {len(eval_rows)}",
            f"N_eval_flip_both_correct: {sum(bool(r['flip_both_correct']) for r in eval_rows)}",
            "",
            "COUNTERFACTUAL",
            "same prompt; horizontally flipped image; A->A and B->B",
            "no query-role swap",
            "",
            "HOW TO READ FULL SCAN",
            "full_flip source-follow asks when the object residual becomes causally",
            "sufficient for the downstream model to produce the flipped relation.",
            "This can be an early causal precursor; do not automatically call it an",
            "abstract LEFT/RIGHT representation.",
            "",
            "HOW TO READ DAS",
            "learned_flip >> random_flip on held-out flip-both-correct examples",
            "with modest D is evidence for a compact causal visual-relation state.",
            "The earliest layer where this rises sharply is the main candidate",
            "visual-relation emergence window.",
            "",
            "NEXT",
            "Once the emergence window is localized, decompose each block into",
            "incoming residual / attention output / MLP output using the SAME",
            "horizontal-flip source. Then revisit candidate heads.",
        ]

        if best:
            report += ["", "TOP PURE-VISUAL DAS"]
            for r in best[:12]:
                report.append(
                    f"L{int(r['layer'])} D{int(r['dim'])}: "
                    f"IIA={100*float(r['source_follow_iia']):.2f}%"
                )

        (out_dir / "report.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "config.json",
            "full_scan.csv",
            "full_scan_samples.jsonl",
            "das_summary.csv",
            "train_history.csv",
            "report.txt",
        ):
            print(" ", out_dir / name)

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
