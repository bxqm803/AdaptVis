#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_direction_head_to_centroid_causal_v1.py

Question
========
Do established middle-layer DIRECTION / spatial heads causally influence a
later ATTENTION-CENTROID head?

This is an internal mechanistic diagnostic only.  It does NOT generate answers,
optimize accuracy, or perform correction.

Default causal pairs (Qwen2.5-VL-3B)
====================================
    L23H01(direction) -> L24H05(centroid)
    L23H05(direction) -> L24H05(centroid)
    L26H03(direction) -> L27H10(centroid)

Why cross-layer only?
=====================
Heads inside the SAME transformer attention layer are computed in parallel.
Therefore a claim such as L24H05 -> L24H10 is not a valid within-layer causal
ordering.  Every tested source head here is strictly earlier than its target
centroid head.

Source-head spatial axes
========================
On a DISJOINT calibration split, capture the source head's pre-W_O output at
subject/reference text-token positions on the real image and a gray-image
control.  For head h at layer L:

    g_i = [z_h(sub)-z_h(ref)]_REAL - [z_h(sub)-z_h(ref)]_GRAY

where z_h is the head slice immediately before W_O.  Per relation, compute
class means mu_left/right/above/below and define:

    u_H = unit(mu_right - mu_left)
    u_V = unit(mu_above - mu_below)

with natural half-gap amplitudes:

    a_H = 0.5 * ||mu_right - mu_left||
    a_V = 0.5 * ||mu_above - mu_below||.

The calibration split is never used for causal-effect reporting.

Causal intervention
===================
On each evaluation sample, modify ONLY the selected source head's pre-W_O
slice at object-token positions.  To change the pooled subject-reference head
vector by +strength*a_H*u_H, add +delta/2 to subject and -delta/2 to reference;
for the negative intervention use the opposite sign.  The same construction is
used for V.

All model weights, visual tokens, source-head attention weights, other heads,
and all other token positions remain unchanged.

The later target head is then run normally.  We read its subject/reference
visual-attention centroids:

    c_H = x_sub - x_ref        (+ = right)
    c_V = -(y_sub - y_ref)     (+ = above)

and estimate the finite-difference causal transport:

                    [ dc_H / dg_H    dc_H / dg_V ]
    J_D->C =        [                            ]
                    [ dc_V / dg_H    dc_V / dg_V ]

Controls
========
1) same_head_orthogonal
   Perturb the SAME source head by an equal-norm vector orthogonal to both u_H
   and u_V.  This asks whether any same-head perturbation moves the centroid.

2) random_head
   Perturb a SAME-LAYER non-source head by an equal-norm fixed random vector.
   This asks whether any same-layer head perturbation moves the centroid.

Interpretation
==============
Evidence that a direction head causally influences a later centroid head:

  * spatial-axis finite differences are reproducibly non-zero;
  * effect magnitude > same-head-orthogonal and random-head controls;
  * ideally H intervention preferentially changes c_H and V changes c_V;
  * source layer < target layer by construction.

Axis-specific positive diagonal is stronger evidence of geometrically aligned
transport, but NONZERO effects can still establish causal influence even if the
mapping rotates or changes sign.

Recommended pilot (80 calibration + 80 held-out evaluation)
============================================================
CUDA_VISIBLE_DEVICES=0 python -u diagnose_direction_head_to_centroid_causal_v1.py \
  --pairs '23:1@24:5,23:5@24:5,26:3@27:10' \
  --fit-per-relation 20 \
  --eval-per-relation 20 \
  --strength 1.0 \
  --direction-pool mean \
  --device cuda:0 \
  --output-dir output/qwen3b_direction_to_centroid_causal_n80_v1 \
  --overwrite

For a faster smoke test use --fit-per-relation 8 --eval-per-relation 8.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import eval_coco_centroid_select_late_direction_qwen25_v1 as centexp
    import analyze_coco_centroid_generation_step1_v4 as cent
except Exception as exc:
    raise SystemExit(
        "Run this script from the AdaptVis llava16 repository root next to "
        "eval_coco_centroid_select_late_direction_qwen25_v1.py and "
        "analyze_coco_centroid_generation_step1_v4.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
EPS = 1e-12


# =============================================================================
# CLI / basic helpers
# =============================================================================

@dataclass(frozen=True)
class PairSpec:
    source_layer: int
    source_head: int
    target_layer: int
    target_head: int

    @property
    def name(self) -> str:
        return (
            f"L{self.source_layer}H{self.source_head:02d}"
            f"_to_L{self.target_layer}H{self.target_head:02d}"
        )


def parse_pairs(text: str) -> List[PairSpec]:
    out: List[PairSpec] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "@" not in raw:
            raise ValueError(
                f"Bad pair {raw!r}; use source_layer:head@target_layer:head, "
                "e.g. 23:1@24:5"
            )
        left, right = raw.split("@", 1)
        sl, sh = [int(x.strip().lower().replace("l", "").replace("h", "")) for x in left.split(":", 1)]
        tl, th = [int(x.strip().lower().replace("l", "").replace("h", "")) for x in right.split(":", 1)]
        if sl >= tl:
            raise ValueError(
                f"Invalid causal pair {raw!r}: source layer must be strictly earlier than target layer"
            )
        spec = PairSpec(sl, sh, tl, th)
        if spec not in out:
            out.append(spec)
    if not out:
        raise ValueError("No causal pairs parsed")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--pairs",
        default="23:1@24:5,23:5@24:5,26:3@27:10",
        help="Comma-separated source_layer:head@target_layer:head specs.",
    )
    p.add_argument(
        "--control-head",
        default="auto",
        help="Same-layer random-head control. 'auto' chooses the highest valid non-source head.",
    )
    p.add_argument("--fit-per-relation", type=int, default=20)
    p.add_argument(
        "--eval-per-relation",
        type=int,
        default=20,
        help="0 = use all remaining samples after calibration.",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Intervention amplitude in calibration half-gap units.",
    )
    p.add_argument(
        "--direction-pool",
        choices=["mean", "last"],
        default="mean",
        help="Pooling of source-head object token positions, matching direction-head definition.",
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.pairs_parsed = parse_pairs(args.pairs)
    if args.fit_per_relation <= 0:
        p.error("--fit-per-relation must be > 0")
    if args.eval_per_relation < 0:
        p.error("--eval-per-relation must be >= 0")
    if args.strength <= 0:
        p.error("--strength must be > 0")
    # centexp.load_coco_records expects max_samples to exist.
    args.max_samples = None
    return args


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def norm_rel(x: Any) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    mapping = {
        "on": "above",
        "over": "above",
        "on top of": "above",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
    }
    return mapping.get(s, s)


def unit_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if n <= EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return (x / n).astype(np.float32)


def safe_mean(xs: Iterable[float]) -> float:
    vals = np.asarray([float(x) for x in xs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    vals = np.asarray([float(x) for x in xs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# =============================================================================
# Model internals
# =============================================================================

def get_text_config(model: Any) -> Any:
    cfg = getattr(model, "config", None)
    for c in (
        getattr(cfg, "text_config", None),
        getattr(cfg, "language_config", None),
        cfg,
    ):
        if c is not None and getattr(c, "num_attention_heads", None) is not None:
            return c
    raise RuntimeError("Could not resolve text config / num_attention_heads")


def resolve_attn(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise RuntimeError(f"Could not resolve attention module in {type(layer).__name__}")


def resolve_o_proj(attn: Any) -> torch.nn.Module:
    for name in ("o_proj", "out_proj", "proj"):
        module = getattr(attn, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise RuntimeError(f"Could not resolve output projection in {type(attn).__name__}")


def span_positions(span: Sequence[int]) -> List[int]:
    if len(span) < 2:
        raise RuntimeError(f"Bad span {span}")
    a, b = int(span[0]), int(span[1])
    if b < a:
        a, b = b, a
    return list(range(a, b + 1))


def pool_head_rows(
    head_tensor: torch.Tensor,  # [S,H,Dh]
    positions: Sequence[int],
    head: int,
    mode: str,
) -> torch.Tensor:
    valid = [int(p) for p in positions if 0 <= int(p) < head_tensor.shape[0]]
    if not valid:
        raise RuntimeError("No valid object-token positions")
    rows = head_tensor[valid, int(head), :]
    return rows[-1] if mode == "last" else rows.mean(dim=0)


class CapturePreWO:
    """Capture concatenated pre-W_O head outputs at selected decoder layers."""

    def __init__(self, decoder_layers: Sequence[Any], layers: Sequence[int], n_heads: int):
        self.decoder_layers = decoder_layers
        self.layers = sorted(set(int(x) for x in layers))
        self.n_heads = int(n_heads)
        self.handles: List[Any] = []
        self.data: Dict[int, torch.Tensor] = {}
        self.events: Dict[int, int] = defaultdict(int)

    def __enter__(self):
        for L in self.layers:
            op = resolve_o_proj(resolve_attn(self.decoder_layers[L]))

            def make_hook(layer_index: int):
                def hook(_module, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError("o_proj pre-hook did not receive tensor")
                    x = inputs[0]
                    if x.ndim != 3:
                        raise RuntimeError(f"pre-W_O shape={tuple(x.shape)}, expected [B,S,D]")
                    if x.shape[-1] % self.n_heads != 0:
                        raise RuntimeError(
                            f"pre-W_O dim={x.shape[-1]} not divisible by n_heads={self.n_heads}"
                        )
                    dh = int(x.shape[-1] // self.n_heads)
                    # Clone outside inference_mode concerns; all callers use no_grad.
                    self.data[layer_index] = (
                        x.detach().clone()[0].reshape(x.shape[1], self.n_heads, dh).float().cpu()
                    )
                    self.events[layer_index] += 1
                return hook

            self.handles.append(op.register_forward_pre_hook(make_hook(L)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class PatchHeadPreWO:
    """Add deltas to one head slice at selected sequence positions before W_O."""

    def __init__(
        self,
        attention: Any,
        head: int,
        deltas_by_position: Mapping[int, torch.Tensor],
        n_heads: int,
    ):
        self.attention = attention
        self.head = int(head)
        self.deltas = {int(k): v.detach().clone() for k, v in deltas_by_position.items()}
        self.n_heads = int(n_heads)
        self.handle = None
        self.events = 0

    def __enter__(self):
        op = resolve_o_proj(self.attention)

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("o_proj patch hook did not receive tensor")
            x = inputs[0]
            if x.ndim != 3:
                raise RuntimeError(f"o_proj input shape={tuple(x.shape)}, expected [B,S,D]")
            if x.shape[-1] % self.n_heads != 0:
                raise RuntimeError("o_proj width not divisible by number of heads")
            dh = int(x.shape[-1] // self.n_heads)
            if not (0 <= self.head < self.n_heads):
                raise RuntimeError(f"head {self.head} outside 0..{self.n_heads-1}")
            start, end = self.head * dh, (self.head + 1) * dh
            y = x.clone()
            for pos, delta in self.deltas.items():
                if not (0 <= pos < y.shape[1]):
                    raise RuntimeError(f"patch position {pos} outside sequence length {y.shape[1]}")
                d = delta.to(device=y.device, dtype=y.dtype).reshape(-1)
                if d.numel() != dh:
                    raise RuntimeError(f"delta dim={d.numel()} != head_dim={dh}")
                y[0, pos, start:end] = y[0, pos, start:end] + d
            self.events += 1
            return (y,) + tuple(inputs[1:])

        self.handle = op.register_forward_pre_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =============================================================================
# Dataset / splits
# =============================================================================

def gray_image_like(image: Image.Image, value: int) -> Image.Image:
    v = max(0, min(255, int(value)))
    return Image.new("RGB", image.size, (v, v, v))


def stratified_disjoint_split(
    records: Sequence[Mapping[str, Any]],
    fit_per_relation: int,
    eval_per_relation: int,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    rng = random.Random(int(seed))
    by_rel: Dict[str, List[Mapping[str, Any]]] = {r: [] for r in REL}
    for record in records:
        r = norm_rel(record.get("relation", ""))
        if r in by_rel:
            by_rel[r].append(record)
    fit: List[Mapping[str, Any]] = []
    ev: List[Mapping[str, Any]] = []
    for r in REL:
        xs = list(by_rel[r])
        rng.shuffle(xs)
        if len(xs) < fit_per_relation + (0 if eval_per_relation == 0 else eval_per_relation):
            raise RuntimeError(
                f"Relation {r}: only {len(xs)} samples, need fit={fit_per_relation} "
                f"+ eval={eval_per_relation if eval_per_relation else 'remaining'}"
            )
        fit.extend(xs[:fit_per_relation])
        remain = xs[fit_per_relation:]
        if eval_per_relation == 0:
            ev.extend(remain)
        else:
            ev.extend(remain[:eval_per_relation])
    rng.shuffle(fit)
    rng.shuffle(ev)
    fit_sids = {int(x["sid"]) for x in fit}
    eval_sids = {int(x["sid"]) for x in ev}
    if fit_sids & eval_sids:
        raise RuntimeError("Calibration/evaluation split overlap")
    return fit, ev


# =============================================================================
# Forward/capture helpers
# =============================================================================

def locate_sample_geometry(
    model: Any,
    processor: Any,
    batch: Mapping[str, torch.Tensor],
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    input_length = len(input_ids)
    sub_span, ref_span = cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        record["subject"],
        record["reference"],
    )
    subject_positions = span_positions(sub_span)
    reference_positions = span_positions(ref_span)
    subject_index = int(sub_span[1])
    reference_index = int(ref_span[1])
    visual_indices = cent.resolve_visual_indices(model, processor, batch, input_ids)
    coords = cent.visual_coordinates(
        model,
        batch,
        len(visual_indices),
        batch["input_ids"].device,
    )
    if coords is None:
        raise RuntimeError("Could not construct visual coordinates")
    return {
        "input_length": int(input_length),
        "subject_positions": subject_positions,
        "reference_positions": reference_positions,
        "subject_index": subject_index,
        "reference_index": reference_index,
        "visual_indices": list(map(int, visual_indices)),
        "coords": coords.detach().float(),
    }


def run_capture_prewo(
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, torch.Tensor],
    source_layers: Sequence[int],
    n_heads: int,
    output_attentions: bool,
) -> Tuple[Dict[int, torch.Tensor], Optional[Sequence[torch.Tensor]], Any]:
    with CapturePreWO(decoder_layers, source_layers, n_heads) as cap:
        with torch.no_grad():
            outputs = model(
                **batch,
                use_cache=False,
                output_attentions=bool(output_attentions),
                output_hidden_states=False,
                return_dict=True,
            )
    for L in source_layers:
        if cap.events.get(int(L), 0) != 1 or int(L) not in cap.data:
            raise RuntimeError(f"pre-W_O capture L{L} events={cap.events.get(int(L), 0)}")
    attentions = centexp.resolve_attention_tuple(outputs) if output_attentions else None
    return dict(cap.data), attentions, outputs


def source_feature(
    real_pre: torch.Tensor,
    gray_pre: torch.Tensor,
    source_head: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    pool: str,
) -> np.ndarray:
    rs = pool_head_rows(real_pre, subject_positions, source_head, pool)
    rr = pool_head_rows(real_pre, reference_positions, source_head, pool)
    gs = pool_head_rows(gray_pre, subject_positions, source_head, pool)
    gr = pool_head_rows(gray_pre, reference_positions, source_head, pool)
    return (rs - rr - gs + gr).numpy().astype(np.float32)


def centroid_from_attentions(
    attentions: Sequence[torch.Tensor],
    target_layer: int,
    target_head: int,
    sample_geometry: Mapping[str, Any],
) -> Dict[str, float]:
    if not (0 <= target_layer < len(attentions)):
        raise RuntimeError(f"Target L{target_layer} unavailable; attentions={len(attentions)}")
    prompt = cent.normalize_attention_tensor(
        attentions[int(target_layer)],
        expected_query_length=int(sample_geometry["input_length"]),
    )
    if not (0 <= target_head < prompt.shape[0]):
        raise RuntimeError(f"Target head H{target_head} outside 0..{prompt.shape[0]-1}")
    rows = prompt[:, [sample_geometry["subject_index"], sample_geometry["reference_index"]], :]
    metrics = cent.query_attention_metrics(
        rows,
        sample_geometry["visual_indices"],
        sample_geometry["coords"],
        sample_geometry["subject_index"],
        sample_geometry["reference_index"],
    )
    maps = metrics["visual_maps"].detach().float()
    m = maps[int(target_head)]  # [2,V]
    coords = sample_geometry["coords"].to(device=m.device, dtype=torch.float32)
    centers = torch.matmul(m, coords)
    sx, sy = [float(x) for x in centers[0].detach().cpu().tolist()]
    rx, ry = [float(x) for x in centers[1].detach().cpu().tolist()]
    dx = sx - rx
    dy = sy - ry
    pred, axis_conf = cent.relation_from_centroids(dx, dy)
    visual_mass = metrics["visual_mass"].detach().float()
    # query_attention_metrics returns visual_mass indexed [head, object] in repo code.
    try:
        smass = float(visual_mass[int(target_head), 0].item())
        rmass = float(visual_mass[int(target_head), 1].item())
    except Exception:
        smass = float("nan")
        rmass = float("nan")
    return {
        "c_H": float(dx),
        "c_V": float(-dy),
        "subject_x": sx,
        "subject_y": sy,
        "reference_x": rx,
        "reference_y": ry,
        "centroid_pred": norm_rel(pred),
        "axis_confidence": float(axis_conf),
        "subject_visual_mass": smass,
        "reference_visual_mass": rmass,
    }


# =============================================================================
# Calibration
# =============================================================================

def fit_axis_code(
    features_by_relation: Mapping[str, Sequence[np.ndarray]],
) -> Dict[str, Any]:
    means: Dict[str, np.ndarray] = {}
    all_rows: List[np.ndarray] = []
    for r in REL:
        xs = list(features_by_relation[r])
        if not xs:
            raise RuntimeError(f"No calibration features for relation={r}")
        arr = np.stack(xs).astype(np.float32)
        means[r] = arr.mean(axis=0).astype(np.float32)
        all_rows.extend(xs)
    center = np.mean(np.stack(all_rows), axis=0).astype(np.float32)
    relation_dirs = {r: unit_np(means[r] - center) for r in REL}

    diff_h = (means["right"] - means["left"]).astype(np.float32)
    diff_v = (means["above"] - means["below"]).astype(np.float32)
    gap_h = float(np.linalg.norm(diff_h))
    gap_v = float(np.linalg.norm(diff_v))
    u_h = unit_np(diff_h)
    u_v = unit_np(diff_v)

    return {
        "center": center,
        "means": means,
        "relation_dirs": relation_dirs,
        "u_H": u_h,
        "u_V": u_v,
        "half_gap_H": max(gap_h / 2.0, EPS),
        "half_gap_V": max(gap_v / 2.0, EPS),
        "axis_cosine": float(np.dot(u_h, u_v)),
    }


def predict_relation(feature: np.ndarray, code: Mapping[str, Any]) -> str:
    z = np.asarray(feature, dtype=np.float32) - np.asarray(code["center"], dtype=np.float32)
    n = float(np.linalg.norm(z))
    if n <= EPS:
        return REL[0]
    z = z / n
    scores = {r: float(np.dot(z, code["relation_dirs"][r])) for r in REL}
    return max(REL, key=lambda r: scores[r])


def make_orthogonal_vector(u_h: np.ndarray, u_v: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    d = len(u_h)
    for _ in range(100):
        x = rng.standard_normal(d).astype(np.float64)
        # Project away span{u_h, u_v}; use least squares for non-orthogonal axes.
        B = np.stack([u_h, u_v], axis=1).astype(np.float64)
        coeff, *_ = np.linalg.lstsq(B, x, rcond=None)
        x = x - B @ coeff
        n = float(np.linalg.norm(x))
        if n > 1e-6:
            return (x / n).astype(np.float32)
    raise RuntimeError("Failed to construct orthogonal control vector")


def make_random_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal(int(dim)).astype(np.float64)
    return unit_np(x)


# =============================================================================
# Intervention construction / execution
# =============================================================================

def add_delta(mapping: Dict[int, torch.Tensor], pos: int, delta: torch.Tensor) -> None:
    pos = int(pos)
    if pos in mapping:
        mapping[pos] = mapping[pos] + delta
    else:
        mapping[pos] = delta.clone()


def build_pair_difference_patch(
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    pair_delta: torch.Tensor,
    pool: str,
) -> Dict[int, torch.Tensor]:
    """
    Construct token-level deltas so the pooled subject-reference head vector
    changes exactly by pair_delta.
    """
    d = pair_delta.reshape(-1) * 0.5
    mapping: Dict[int, torch.Tensor] = {}
    if pool == "last":
        add_delta(mapping, int(subject_positions[-1]), +d)
        add_delta(mapping, int(reference_positions[-1]), -d)
    else:
        # Same +d/-d on every member preserves the mean-pool pair change exactly.
        for p in subject_positions:
            add_delta(mapping, int(p), +d)
        for p in reference_positions:
            add_delta(mapping, int(p), -d)
    return mapping


def run_patched_centroid(
    *,
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, torch.Tensor],
    sample_geometry: Mapping[str, Any],
    source_layer: int,
    patch_head: int,
    deltas: Mapping[int, torch.Tensor],
    n_heads: int,
    target_layer: int,
    target_head: int,
) -> Dict[str, float]:
    attn = resolve_attn(decoder_layers[int(source_layer)])
    with PatchHeadPreWO(attn, patch_head, deltas, n_heads) as patch:
        with torch.no_grad():
            outputs = model(
                **batch,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
    if patch.events != 1:
        raise RuntimeError(f"Patch fired {patch.events} times, expected 1")
    attentions = centexp.resolve_attention_tuple(outputs)
    result = centroid_from_attentions(
        attentions,
        target_layer,
        target_head,
        sample_geometry,
    )
    del outputs, attentions
    return result


# =============================================================================
# Aggregation
# =============================================================================

def finite_difference_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["pair", "control_type", "axis"]
    for keys, sub in df.groupby(group_cols, sort=False):
        pair, control_type, axis = keys
        plus = sub[sub.sign == +1].set_index("sid")
        minus = sub[sub.sign == -1].set_index("sid")
        common = sorted(set(plus.index) & set(minus.index))
        for sid in common:
            p = plus.loc[sid]
            m = minus.loc[sid]
            amp = float(p.injected_pair_delta_norm)
            if amp <= EPS:
                continue
            fd_h = 0.5 * (float(p.c_H) - float(m.c_H))
            fd_v = 0.5 * (float(p.c_V) - float(m.c_V))
            rows.append(
                {
                    "sid": int(sid),
                    "gt": str(p.gt),
                    "pair": pair,
                    "control_type": control_type,
                    "axis": axis,
                    "amplitude": amp,
                    "fd_delta_cH": fd_h,
                    "fd_delta_cV": fd_v,
                    "deriv_cH": fd_h / amp,
                    "deriv_cV": fd_v / amp,
                    "fd_effect_norm": float(math.hypot(fd_h, fd_v)),
                    "same_axis_effect": fd_h if axis == "H" else fd_v,
                    "cross_axis_effect": fd_v if axis == "H" else fd_h,
                }
            )
    return pd.DataFrame(rows)


def summarize_matrix(fd: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (pair, control), sub in fd.groupby(["pair", "control_type"], sort=False):
        h = sub[sub.axis == "H"]
        v = sub[sub.axis == "V"]
        if h.empty or v.empty:
            continue
        J_HH = safe_mean(h.deriv_cH)
        J_VH = safe_mean(h.deriv_cV)  # output V from source H
        J_HV = safe_mean(v.deriv_cH)  # output H from source V
        J_VV = safe_mean(v.deriv_cV)
        abs_diag = safe_mean(list(np.abs(h.deriv_cH)) + list(np.abs(v.deriv_cV)))
        abs_cross = safe_mean(list(np.abs(h.deriv_cV)) + list(np.abs(v.deriv_cH)))
        rows.append(
            {
                "pair": pair,
                "control_type": control,
                "N_H": int(len(h)),
                "N_V": int(len(v)),
                "J_HH": J_HH,
                "J_HV": J_HV,
                "J_VH": J_VH,
                "J_VV": J_VV,
                "median_J_HH": safe_median(h.deriv_cH),
                "median_J_HV": safe_median(v.deriv_cH),
                "median_J_VH": safe_median(h.deriv_cV),
                "median_J_VV": safe_median(v.deriv_cV),
                "H_same_axis_positive_rate": float((h.same_axis_effect > 0).mean()),
                "V_same_axis_positive_rate": float((v.same_axis_effect > 0).mean()),
                "mean_abs_diag": abs_diag,
                "mean_abs_cross": abs_cross,
                "diag_over_cross": float(abs_diag / max(abs_cross, EPS)),
                "mean_fd_effect_norm_H": safe_mean(h.fd_effect_norm),
                "mean_fd_effect_norm_V": safe_mean(v.fd_effect_norm),
                "corr_H_amplitude_effect": safe_corr(h.amplitude, h.same_axis_effect),
                "corr_V_amplitude_effect": safe_corr(v.amplitude, v.same_axis_effect),
            }
        )
    return pd.DataFrame(rows)


def summarize_controls(fd: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pair, sub in fd.groupby("pair", sort=False):
        for axis in ("H", "V"):
            ax = sub[sub.axis == axis]
            stats = {}
            for control in ("spatial_axis", "same_head_orthogonal", "random_head"):
                z = ax[ax.control_type == control]
                stats[control] = {
                    "norm": safe_mean(z.fd_effect_norm),
                    "same": safe_mean(np.abs(z.same_axis_effect)),
                    "cross": safe_mean(np.abs(z.cross_axis_effect)),
                }
            s = stats["spatial_axis"]
            o = stats["same_head_orthogonal"]
            r = stats["random_head"]
            rows.append(
                {
                    "pair": pair,
                    "axis": axis,
                    "spatial_effect_norm": s["norm"],
                    "orthogonal_effect_norm": o["norm"],
                    "random_head_effect_norm": r["norm"],
                    "spatial_over_orthogonal": float(s["norm"] / max(o["norm"], EPS)),
                    "spatial_over_random_head": float(s["norm"] / max(r["norm"], EPS)),
                    "spatial_same_axis_abs": s["same"],
                    "spatial_cross_axis_abs": s["cross"],
                    "spatial_axis_specificity": float(s["same"] / max(s["cross"], EPS)),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_gt(fd: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    target = fd[fd.control_type == "spatial_axis"]
    for (pair, axis, gt), sub in target.groupby(["pair", "axis", "gt"], sort=False):
        rows.append(
            {
                "pair": pair,
                "axis": axis,
                "gt": gt,
                "N": int(len(sub)),
                "mean_same_axis_effect": safe_mean(sub.same_axis_effect),
                "mean_cross_axis_effect": safe_mean(sub.cross_axis_effect),
                "same_axis_positive_rate": float((sub.same_axis_effect > 0).mean()),
                "mean_effect_norm": safe_mean(sub.fd_effect_norm),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records, audit = centexp.load_coco_records(args)
    # Normalize relation field defensively.
    records = [dict(r) for r in records]
    for r in records:
        r["relation"] = norm_rel(r.get("relation", ""))
    fit_records, eval_records = stratified_disjoint_split(
        records,
        args.fit_per_relation,
        args.eval_per_relation,
        args.seed,
    )

    model, processor, decoder_layers, decoder_path, model_spec = centexp.load_model_and_processor(args)
    text_cfg = get_text_config(model)
    n_heads = int(text_cfg.num_attention_heads)
    hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        op0 = resolve_o_proj(resolve_attn(decoder_layers[args.pairs_parsed[0].source_layer]))
        hidden_size = int(op0.in_features)
    if hidden_size % n_heads != 0:
        raise RuntimeError(f"hidden_size={hidden_size} not divisible by n_heads={n_heads}")
    head_dim = hidden_size // n_heads

    for p in args.pairs_parsed:
        if p.target_layer >= len(decoder_layers):
            raise RuntimeError(f"{p.name}: target layer outside decoder")
        if p.source_head >= n_heads or p.target_head >= n_heads:
            raise RuntimeError(f"{p.name}: head index outside 0..{n_heads-1}")

    if str(args.control_head).strip().lower() == "auto":
        control_head_global = n_heads - 1
    else:
        control_head_global = int(args.control_head)
        if not (0 <= control_head_global < n_heads):
            raise RuntimeError(f"control head {control_head_global} outside 0..{n_heads-1}")

    source_layers = sorted(set(p.source_layer for p in args.pairs_parsed))

    print("\n" + "=" * 180)
    print("DIRECTION/SPATIAL HEAD -> LATER CENTROID HEAD: CAUSAL TEST")
    print("=" * 180)
    print(f"model={args.model} decoder={decoder_path} heads={n_heads} head_dim={head_dim}")
    print("pairs=" + ", ".join(p.name for p in args.pairs_parsed))
    print(
        f"calibration={len(fit_records)} ({args.fit_per_relation}/relation) | "
        f"evaluation={len(eval_records)} "
        f"({'all remaining' if args.eval_per_relation == 0 else str(args.eval_per_relation) + '/relation'})"
    )
    print(f"direction definition=Real-Gray pre-W_O object-pair | pool={args.direction_pool}")
    print(f"strength={args.strength} half-gap units | random control head default=H{control_head_global:02d}")
    print("No generation. Evaluation GT is used only for reporting direction-head accuracy / by-GT summaries.")
    print("=" * 180, flush=True)

    # -------------------------------------------------------------------------
    # Calibration: fit source-head H/V axes on disjoint samples.
    # -------------------------------------------------------------------------
    cal_store: Dict[str, Dict[str, List[np.ndarray]]] = {
        p.name: {r: [] for r in REL} for p in args.pairs_parsed
    }
    errors: List[Dict[str, Any]] = []

    for record in tqdm(fit_records, desc="CALIBRATE direction axes"):
        sid = int(record["sid"])
        real = gray = None
        try:
            real = centexp.open_record_image(record)
            gray = gray_image_like(real, args.gray_value)
            device = torch.device(args.device)
            real_batch = centexp.make_batch(processor, real, record, device)
            gray_batch = centexp.make_batch(processor, gray, record, device)
            geom = locate_sample_geometry(model, processor, real_batch, record)

            real_pre, _, real_out = run_capture_prewo(
                model, decoder_layers, real_batch, source_layers, n_heads, False
            )
            gray_pre, _, gray_out = run_capture_prewo(
                model, decoder_layers, gray_batch, source_layers, n_heads, False
            )
            gt = norm_rel(record["relation"])
            for pair in args.pairs_parsed:
                feat = source_feature(
                    real_pre[pair.source_layer],
                    gray_pre[pair.source_layer],
                    pair.source_head,
                    geom["subject_positions"],
                    geom["reference_positions"],
                    args.direction_pool,
                )
                cal_store[pair.name][gt].append(feat)
            del real_out, gray_out, real_pre, gray_pre, real_batch, gray_batch
        except Exception as exc:
            errors.append({
                "phase": "calibration",
                "sid": sid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": " | ".join(traceback.format_exc().splitlines()[-10:]),
            })
            tqdm.write(f"[CAL ERROR sid={sid}] {type(exc).__name__}: {exc}")
        finally:
            for image in (real, gray):
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
            cleanup()

    codes: Dict[str, Dict[str, Any]] = {}
    cal_rows: List[Dict[str, Any]] = []
    axes_npz: Dict[str, np.ndarray] = {}
    control_heads: Dict[str, int] = {}

    for i, pair in enumerate(args.pairs_parsed):
        code = fit_axis_code(cal_store[pair.name])
        codes[pair.name] = code
        ch = control_head_global
        if ch == pair.source_head:
            ch = n_heads - 2 if n_heads >= 2 else 0
        if ch == pair.source_head:
            raise RuntimeError(f"Could not choose control head for {pair.name}")
        control_heads[pair.name] = int(ch)

        orth = make_orthogonal_vector(
            code["u_H"], code["u_V"], args.seed + 1000 + i * 13
        )
        rnd = make_random_unit(head_dim, args.seed + 2000 + i * 17)
        code["u_orth"] = orth
        code["u_random"] = rnd

        cal_rows.append({
            "pair": pair.name,
            "source_layer": pair.source_layer,
            "source_head": pair.source_head,
            "target_layer": pair.target_layer,
            "target_head": pair.target_head,
            "control_head": int(ch),
            "N_calibration": int(sum(len(v) for v in cal_store[pair.name].values())),
            "half_gap_H": float(code["half_gap_H"]),
            "half_gap_V": float(code["half_gap_V"]),
            "cos_uH_uV": float(code["axis_cosine"]),
            "orth_dot_uH": float(np.dot(orth, code["u_H"])),
            "orth_dot_uV": float(np.dot(orth, code["u_V"])),
        })
        safe = pair.name.replace("-", "_")
        axes_npz[f"{safe}__u_H"] = code["u_H"]
        axes_npz[f"{safe}__u_V"] = code["u_V"]
        axes_npz[f"{safe}__u_orth"] = orth
        axes_npz[f"{safe}__u_random"] = rnd
        axes_npz[f"{safe}__center"] = code["center"]
        for r in REL:
            axes_npz[f"{safe}__mu_{r}"] = code["means"][r]
            axes_npz[f"{safe}__dir_{r}"] = code["relation_dirs"][r]

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(outdir / "calibration_summary.csv", index=False)
    np.savez_compressed(outdir / "direction_axes.npz", **axes_npz)

    print("\nCALIBRATION AXES")
    print("-" * 180)
    print(cal_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # -------------------------------------------------------------------------
    # Held-out causal evaluation.
    # -------------------------------------------------------------------------
    sample_rows: List[Dict[str, Any]] = []
    direction_eval_rows: List[Dict[str, Any]] = []
    clean_centroid_rows: List[Dict[str, Any]] = []

    for record in tqdm(eval_records, desc="EVAL direction->centroid causal"):
        sid = int(record["sid"])
        real = gray = None
        try:
            real = centexp.open_record_image(record)
            gray = gray_image_like(real, args.gray_value)
            device = torch.device(args.device)
            real_batch = centexp.make_batch(processor, real, record, device)
            gray_batch = centexp.make_batch(processor, gray, record, device)
            geom = locate_sample_geometry(model, processor, real_batch, record)

            # One clean REAL forward captures all source head outputs and all target attentions.
            real_pre, real_attn, real_out = run_capture_prewo(
                model, decoder_layers, real_batch, source_layers, n_heads, True
            )
            assert real_attn is not None
            # One gray forward lets us independently verify direction-head accuracy on eval.
            gray_pre, _, gray_out = run_capture_prewo(
                model, decoder_layers, gray_batch, source_layers, n_heads, False
            )

            gt = norm_rel(record["relation"])

            for pair_index, pair in enumerate(args.pairs_parsed):
                code = codes[pair.name]
                clean_cent = centroid_from_attentions(
                    real_attn,
                    pair.target_layer,
                    pair.target_head,
                    geom,
                )
                clean_centroid_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "pair": pair.name,
                    "centroid_pred": clean_cent["centroid_pred"],
                    "centroid_correct": bool(clean_cent["centroid_pred"] == gt),
                    "clean_cH": clean_cent["c_H"],
                    "clean_cV": clean_cent["c_V"],
                })

                feat = source_feature(
                    real_pre[pair.source_layer],
                    gray_pre[pair.source_layer],
                    pair.source_head,
                    geom["subject_positions"],
                    geom["reference_positions"],
                    args.direction_pool,
                )
                dpred = predict_relation(feat, code)
                direction_eval_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "pair": pair.name,
                    "direction_pred": dpred,
                    "direction_correct": bool(dpred == gt),
                })

                for axis in ("H", "V"):
                    amp = float(args.strength) * float(code[f"half_gap_{axis}"])
                    spatial_vec = np.asarray(code[f"u_{axis}"], dtype=np.float32)
                    orth_vec = np.asarray(code["u_orth"], dtype=np.float32)
                    random_vec = np.asarray(code["u_random"], dtype=np.float32)

                    control_defs = [
                        ("spatial_axis", pair.source_head, spatial_vec),
                        ("same_head_orthogonal", pair.source_head, orth_vec),
                        ("random_head", control_heads[pair.name], random_vec),
                    ]

                    for control_type, patch_head, vec in control_defs:
                        for sign in (+1, -1):
                            pair_delta = torch.as_tensor(
                                float(sign) * amp * vec,
                                dtype=torch.float32,
                            )
                            deltas = build_pair_difference_patch(
                                geom["subject_positions"],
                                geom["reference_positions"],
                                pair_delta,
                                args.direction_pool,
                            )
                            cf = run_patched_centroid(
                                model=model,
                                decoder_layers=decoder_layers,
                                batch=real_batch,
                                sample_geometry=geom,
                                source_layer=pair.source_layer,
                                patch_head=int(patch_head),
                                deltas=deltas,
                                n_heads=n_heads,
                                target_layer=pair.target_layer,
                                target_head=pair.target_head,
                            )
                            sample_rows.append({
                                "sid": sid,
                                "gt": gt,
                                "pair": pair.name,
                                "source_layer": pair.source_layer,
                                "source_head": pair.source_head,
                                "target_layer": pair.target_layer,
                                "target_head": pair.target_head,
                                "control_type": control_type,
                                "patch_head": int(patch_head),
                                "axis": axis,
                                "sign": int(sign),
                                "strength": float(args.strength),
                                "injected_pair_delta_norm": float(amp),
                                "clean_c_H": float(clean_cent["c_H"]),
                                "clean_c_V": float(clean_cent["c_V"]),
                                "c_H": float(cf["c_H"]),
                                "c_V": float(cf["c_V"]),
                                "delta_c_H": float(cf["c_H"] - clean_cent["c_H"]),
                                "delta_c_V": float(cf["c_V"] - clean_cent["c_V"]),
                                "clean_centroid_pred": clean_cent["centroid_pred"],
                                "cf_centroid_pred": cf["centroid_pred"],
                                "direction_pred": dpred,
                                "direction_correct": bool(dpred == gt),
                            })

            del real_out, gray_out, real_pre, gray_pre, real_attn, real_batch, gray_batch

            # Incremental persistence for long cluster runs.
            if len(sample_rows) and len(sample_rows) % 200 == 0:
                pd.DataFrame(sample_rows).to_csv(outdir / "sample_interventions.csv", index=False)

        except Exception as exc:
            errors.append({
                "phase": "evaluation",
                "sid": sid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": " | ".join(traceback.format_exc().splitlines()[-12:]),
            })
            tqdm.write(f"[EVAL ERROR sid={sid}] {type(exc).__name__}: {exc}")
        finally:
            for image in (real, gray):
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
            cleanup()

    if not sample_rows:
        raise RuntimeError("No successful causal evaluation rows")

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(outdir / "sample_interventions.csv", index=False)
    dir_df = pd.DataFrame(direction_eval_rows)
    dir_df.to_csv(outdir / "direction_head_eval.csv", index=False)
    cent_df = pd.DataFrame(clean_centroid_rows)
    cent_df.to_csv(outdir / "clean_centroid_eval.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "errors.csv", index=False)

    fd = finite_difference_rows(sample_df)
    fd.to_csv(outdir / "per_sample_finite_difference.csv", index=False)
    matrix = summarize_matrix(fd)
    matrix.to_csv(outdir / "causal_matrix_summary.csv", index=False)
    controls = summarize_controls(fd)
    controls.to_csv(outdir / "control_comparison.csv", index=False)
    by_gt = summarize_by_gt(fd)
    by_gt.to_csv(outdir / "spatial_axis_by_gt.csv", index=False)

    direction_summary = (
        dir_df.groupby("pair", as_index=False)
        .agg(N=("sid", "count"), direction_accuracy=("direction_correct", "mean"))
    )
    centroid_summary = (
        cent_df.groupby("pair", as_index=False)
        .agg(N=("sid", "count"), centroid_accuracy=("centroid_correct", "mean"))
    )
    sanity = direction_summary.merge(centroid_summary, on=["pair", "N"], how="outer")
    sanity.to_csv(outdir / "clean_readout_sanity.csv", index=False)

    print("\n" + "=" * 180)
    print("CLEAN HELD-OUT READOUT SANITY")
    print("=" * 180)
    print(sanity.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 180)
    print("SPATIAL/DIRECTION HEAD -> LATER CENTROID: CAUSAL FINITE-DIFFERENCE MATRIX")
    print("=" * 180)
    print(matrix.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nCONTROL COMPARISON")
    print("-" * 180)
    print(controls.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nSPATIAL-AXIS EFFECT BY GT RELATION")
    print("-" * 180)
    print(by_gt.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    metadata = {
        "script": "diagnose_direction_head_to_centroid_causal_v1.py",
        "model": args.model,
        "repo_id": getattr(model_spec, "repo_id", ""),
        "decoder_path": decoder_path,
        "pairs": [p.__dict__ for p in args.pairs_parsed],
        "control_heads": control_heads,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "fit_per_relation": args.fit_per_relation,
        "eval_per_relation": args.eval_per_relation,
        "n_fit": len(fit_records),
        "n_eval_requested": len(eval_records),
        "direction_pool": args.direction_pool,
        "gray_value": args.gray_value,
        "strength": args.strength,
        "seed": args.seed,
        "dataset_audit": audit,
        "errors": len(errors),
        "intervention": {
            "source_feature": "Real-Gray subject-reference pre-W_O head output",
            "spatial_axes": "u_H=unit(mu_right-mu_left), u_V=unit(mu_above-mu_below) from disjoint calibration",
            "patch": "equal/opposite source-head pre-W_O object-token deltas",
            "target_readout": "later head subject/reference visual-attention centroid",
            "same_head_control": "equal-norm vector orthogonal to fitted H/V span",
            "random_head_control": "equal-norm fixed random vector on same-layer non-source head",
        },
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n[saved] {outdir / 'sample_interventions.csv'}")
    print(f"[saved] {outdir / 'causal_matrix_summary.csv'}")
    print(f"[saved] {outdir / 'control_comparison.csv'}")
    print(f"[saved] {outdir / 'spatial_axis_by_gt.csv'}")
    print(f"[success] eval_samples={sample_df.sid.nunique()} errors={len(errors)}")


if __name__ == "__main__":
    main()
