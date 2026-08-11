#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-stage causal specialization test for COCO two-object spatial reasoning.

Working hypothesis
==================

Stage I  (L19-L23): RELATION REPRESENTATION
    Direction-high heads write relation-bearing information into object-token states.

Stage II (L20-L24): SPATIAL ROUTING / GEOMETRY
    Early centroid-high heads organize/route object information so downstream
    receiver heads can read the relevant object tokens.

Stage III (L24-L26+): RECEIVER / LAST-TOKEN INTEGRATION
    Receiver heads read relation-bearing object values and write them into the
    prompt-last residual, which is then read by the LM head.

The experiment DOES NOT infer these functions from Top-K overlap.  It perturbs
each proposed stage separately and measures five internal consequences:

    M1 relation_state_damage
       Downstream direction representation at held-out sentinel heads.

    M2 receiver_routing_damage
       Change in receiver attention from prompt-last to subject/reference tokens.

    M3 receiver_object_write_damage
       Change in the selected receivers' object->prompt-last post-W_O writes.

    M4 last_token_damage
       Change in the final decoder prompt-last state.

    M5 answer_margin_damage
       Change in GT-vs-opposite next-token relation margin.

The desired "functional specialization" pattern is approximately:

                         M1 relation   M2 routing   M3 write   M4 last   M5 answer
    Stage I Direction       HIGH        lower       HIGH       HIGH      HIGH
    Stage II Centroid       lower       HIGH        HIGH       HIGH      HIGH
    Stage III Receiver      ~0          ~0          HIGH       HIGH      HIGH
    matched random          low         low         low        low       low

Interventions
=============

Stage I / Stage II:
    Zero selected attention-head pre-W_O slices ONLY at object-token positions.
    The rest of the head and all model weights remain untouched.

Stage III:
    Do NOT zero the whole receiver head.
    On a clean original run, compute for each selected receiver head:

        c_h = W_O^h sum_{s in object} A_h[last,s] V_h[s]

    Then subtract sum_h c_h from that layer's attention output at prompt-last.
    Thus the receiver can still process other tokens, but its selected
    object->last write is removed.

Matched controls
================

For every stage a same-layer random bundle with the same number of heads is
constructed automatically, excluding all hypothesized heads/readout heads.

Internal normalization
======================

For vector-valued metrics x, the original-vs-swapped clean pair defines a
sample-specific counterfactual axis:

    delta = x_swap - x_clean

For intervention x_int:

    axis_shift = <x_int-x_clean, delta> / ||delta||^2
    rel_l2     = ||x_int-x_clean|| / ||delta||

Interpretation:
    axis_shift ~ 0 : no movement toward swapped computation
    axis_shift ~ 1 : roughly full clean->swapped movement
    rel_l2     ~ 0 : little internal damage
    rel_l2     ~ 1 : intervention size comparable to full semantic swap

For the answer margin:

    normalized_answer_damage =
        (margin_clean - margin_int) / (margin_clean - margin_swap)

Positive values mean the intervention removes support for the original relation.

Dependencies
============

Place this script in the AdaptVis repository root next to:

    analyze_coco_ioi_backward_circuit_v1.py
    analyze_coco_producer_qk_ov_v1.py
    analyze_coco_receiver_qkv_v1.py
    analyze_spatial_storage_transport_utilization_v3.py
    analyze_coco_centroid_generation_step1_v4.py
    analyze_coco_flip_attention_spatial_vectors_v1.py

It also requires the v3 source cache:

    output/spatial_storage_transport_utilization/coco/qwen-3b/
        config.json
        extraction.jsonl
        cache/<sid>.npz

Recommended first run
=====================

CUDA_VISIBLE_DEVICES=0 python -u validate_three_stage_spatial_circuit_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --sample-status both_correct \
  --sample-max-samples 30 \
  --object-state mean \
  --device cuda:0 \
  --output-dir output/qwen3b_three_stage_spatial_circuit_v1 \
  --overwrite

Then validate on more samples:

CUDA_VISIBLE_DEVICES=0 python -u validate_three_stage_spatial_circuit_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --sample-status both_correct \
  --sample-max-samples 100 \
  --object-state mean \
  --device cuda:0 \
  --output-dir output/qwen3b_three_stage_spatial_circuit_v1_n100 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_VERSION = "three-stage-spatial-circuit-v1"

RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

# ---------------------------------------------------------------------------
# Default disjoint functional bundles.
#
# Stage I:
#   top raw-image direction heads, excluding centroid overlap and L26 sentinel.
#
# Stage II:
#   early/mid centroid Top10 heads that are NOT Direction Top10.
#
# Stage III:
#   receiver candidates, excluding L24H5 because it is used in Stage II.
#
# Direction sentinels:
#   later L26 direction-readable heads not directly intervened in Stage III.
# ---------------------------------------------------------------------------

DEFAULT_STAGE1 = "23:5,23:1,19:13,23:0,20:8"
DEFAULT_STAGE2 = "24:5,20:5,22:13,22:0,21:1"
DEFAULT_STAGE3 = "26:4,26:2,26:6,26:0,24:4"
DEFAULT_DIRECTION_READOUT = "26:3,26:1,26:7"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument(
        "--object-state",
        choices=("last", "mean"),
        default="mean",
        help="Use mean to match the current head-direction experiment.",
    )

    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
    )

    p.add_argument("--stage1-heads", default=DEFAULT_STAGE1)
    p.add_argument("--stage2-heads", default=DEFAULT_STAGE2)
    p.add_argument("--stage3-heads", default=DEFAULT_STAGE3)
    p.add_argument(
        "--direction-readout-heads",
        default=DEFAULT_DIRECTION_READOUT,
        help="Downstream relation-state sentinels; should not overlap intervention bundles.",
    )

    p.add_argument(
        "--sample-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
    )
    p.add_argument(
        "--sample-max-samples",
        type=int,
        default=30,
        help="0 means all eligible samples.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--trace-layer-chunk", type=int, default=8)
    p.add_argument("--min-margin-denominator", type=float, default=1e-4)
    p.add_argument(
        "--require-margin-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require clean original margin >0 and swapped margin <0.",
    )
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)

    # Companion scripts.
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
    p.add_argument(
        "--attention-helper",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Compatibility names consumed by imported helpers.
    p.add_argument("--max-samples", type=int, default=None)

    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def import_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_head(text: str) -> Tuple[int, int]:
    value = str(text).strip().upper()
    value = value.replace("L", "").replace("H", ":")
    while "::" in value:
        value = value.replace("::", ":")
    if ":" not in value:
        raise ValueError(f"Bad head spec {text!r}; expected L23H5 or 23:5")
    layer, head = value.split(":", 1)
    return int(layer), int(head)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        head = parse_head(item)
        if head not in seen:
            seen.add(head)
            out.append(head)
    if not out:
        raise ValueError("No heads selected")
    return out


def hname(head: Tuple[int, int]) -> str:
    return f"L{int(head[0])}H{int(head[1]):02d}"


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def finite(values: Iterable[Any]) -> np.ndarray:
    x = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    return x[np.isfinite(x)]


def safe_mean(values: Iterable[Any]) -> float:
    x = finite(values)
    return float(x.mean()) if x.size else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    x = finite(values)
    return float(np.median(x)) if x.size else float("nan")


def safe_std(values: Iterable[Any]) -> float:
    x = finite(values)
    return float(x.std()) if x.size else float("nan")


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def axis_metrics(
    clean: np.ndarray,
    swapped: np.ndarray,
    intervention: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    swapped = np.asarray(swapped, dtype=np.float64).reshape(-1)
    intervention = np.asarray(intervention, dtype=np.float64).reshape(-1)

    if clean.shape != swapped.shape or clean.shape != intervention.shape:
        raise RuntimeError(
            f"Axis vectors differ: clean={clean.shape}, swap={swapped.shape}, "
            f"intervention={intervention.shape}"
        )

    delta = swapped - clean
    moved = intervention - clean
    denom2 = float(np.dot(delta, delta))
    denom = math.sqrt(max(denom2, 0.0))

    return {
        "axis_shift": (
            float(np.dot(moved, delta) / denom2)
            if denom2 > eps else float("nan")
        ),
        "relative_l2": (
            float(np.linalg.norm(moved) / denom)
            if denom > eps else float("nan")
        ),
        "cosine_to_clean": cosine(intervention, clean),
        "cosine_to_swapped": cosine(intervention, swapped),
        "clean_swap_distance": denom,
        "absolute_l2": float(np.linalg.norm(moved)),
    }


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stratified_limit(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda r: int(r["sid"]))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    labels = [r for r in RELATIONS if r in groups]
    cursors = {label: 0 for label in labels}
    selected: List[Dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for label in labels:
            i = cursors[label]
            if i < len(groups[label]) and len(selected) < limit:
                selected.append(groups[label][i])
                cursors[label] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda r: int(r["sid"]))


def relation_margin_from_scores(scores: Mapping[str, float], gt: str) -> float:
    return float(scores[gt] - scores[OPPOSITE[gt]])


# =============================================================================
# Model structural helpers
# =============================================================================

def resolve_attention(attention_helper: Any, decoder_layers: Sequence[Any], layer: int) -> Any:
    return attention_helper.resolve_self_attention(decoder_layers[int(layer)])


def infer_o_head_dim(attention: Any, receiver_module: Any) -> Tuple[int, int]:
    shape = receiver_module.resolve_attention_shape(attention)
    o_proj = getattr(attention, "o_proj", None)
    if o_proj is None:
        raise RuntimeError(f"{type(attention).__name__} lacks o_proj")
    in_features = getattr(o_proj, "in_features", None)
    if in_features is None and hasattr(o_proj, "weight"):
        in_features = int(o_proj.weight.shape[1])
    if in_features is None:
        raise RuntimeError("Cannot infer o_proj input width")
    in_features = int(in_features)
    if in_features % shape.n_query_heads != 0:
        raise RuntimeError(
            f"o_proj width={in_features} not divisible by query heads={shape.n_query_heads}"
        )
    return shape.n_query_heads, in_features // shape.n_query_heads


def validate_heads(
    *,
    heads: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    label: str,
) -> None:
    for layer, head in heads:
        if not 0 <= int(layer) < len(decoder_layers):
            raise ValueError(f"{label}: layer {layer} outside decoder")
        attention = resolve_attention(attention_helper, decoder_layers, layer)
        n_heads, _ = infer_o_head_dim(attention, receiver_module)
        if not 0 <= int(head) < n_heads:
            raise ValueError(
                f"{label}: {hname((layer, head))} outside 0..{n_heads-1}"
            )


def matched_random_heads(
    *,
    target: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    excluded: set[Tuple[int, int]],
    seed: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    output: List[Tuple[int, int]] = []
    used = set(excluded)

    for layer, _ in target:
        attention = resolve_attention(attention_helper, decoder_layers, layer)
        n_heads, _ = infer_o_head_dim(attention, receiver_module)
        candidates = [
            (int(layer), h)
            for h in range(n_heads)
            if (int(layer), h) not in used
        ]
        if not candidates:
            raise RuntimeError(
                f"No random head available at layer {layer}; exclusions too broad"
            )
        pick = rng.choice(candidates)
        output.append(pick)
        used.add(pick)
    return output


# =============================================================================
# Interventions / captures
# =============================================================================

class PreWOObjectController:
    """
    One pre-hook per relevant o_proj.

    It first zeros selected head slices at object positions, then captures
    selected downstream readout-head vectors AFTER the zero intervention.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        zero_heads: Sequence[Tuple[int, int]],
        capture_heads: Sequence[Tuple[int, int]],
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver_module = receiver_module
        self.zero_by_layer: Dict[int, List[int]] = defaultdict(list)
        self.capture_by_layer: Dict[int, List[int]] = defaultdict(list)

        for layer, head in zero_heads:
            self.zero_by_layer[int(layer)].append(int(head))
        for layer, head in capture_heads:
            self.capture_by_layer[int(layer)].append(int(head))

        self.subject_positions = sorted(set(map(int, subject_positions)))
        self.reference_positions = sorted(set(map(int, reference_positions)))
        self.handles: List[Any] = []
        self.captured: Dict[Tuple[int, int], np.ndarray] = {}
        self.events: Dict[int, int] = defaultdict(int)

    def __enter__(self) -> "PreWOObjectController":
        layers = sorted(set(self.zero_by_layer) | set(self.capture_by_layer))
        for layer in layers:
            attention = resolve_attention(
                self.attention_helper, self.decoder_layers, layer
            )
            o_proj = getattr(attention, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"L{layer} attention lacks o_proj")
            n_heads, head_dim = infer_o_head_dim(
                attention, self.receiver_module
            )

            def make_hook(
                layer_index: int,
                query_heads: int,
                dim: int,
            ):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(
                            f"L{layer_index} o_proj pre-hook expected tensor input"
                        )
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        raise RuntimeError(
                            f"L{layer_index} o_proj input shape={tuple(x.shape)}"
                        )
                    if int(x.shape[-1]) != query_heads * dim:
                        raise RuntimeError(
                            f"L{layer_index}: width={int(x.shape[-1])}, "
                            f"expected={query_heads*dim}"
                        )

                    modified = x
                    zeros = self.zero_by_layer.get(layer_index, [])
                    if zeros:
                        modified = x.clone()
                        object_positions = sorted(
                            set(self.subject_positions + self.reference_positions)
                        )
                        for position in object_positions:
                            if not 0 <= position < int(modified.shape[1]):
                                raise RuntimeError(
                                    f"L{layer_index}: object position {position} "
                                    f"outside S={int(modified.shape[1])}"
                                )
                            for head in zeros:
                                start = int(head) * dim
                                stop = start + dim
                                modified[0, position, start:stop] = 0

                    # Capture role-ordered subject-reference vector after patch.
                    for head in self.capture_by_layer.get(layer_index, []):
                        start = int(head) * dim
                        stop = start + dim
                        sp = torch.as_tensor(
                            self.subject_positions,
                            device=modified.device,
                            dtype=torch.long,
                        )
                        rp = torch.as_tensor(
                            self.reference_positions,
                            device=modified.device,
                            dtype=torch.long,
                        )
                        subject = (
                            modified[0]
                            .index_select(0, sp)[:, start:stop]
                            .mean(dim=0)
                        )
                        reference = (
                            modified[0]
                            .index_select(0, rp)[:, start:stop]
                            .mean(dim=0)
                        )
                        self.captured[(layer_index, int(head))] = (
                            (subject - reference)
                            .detach().float().cpu().numpy()
                        )

                    self.events[layer_index] += 1
                    if modified is x:
                        return None
                    return (modified, *inputs[1:])
                return hook

            self.handles.append(
                o_proj.register_forward_pre_hook(
                    make_hook(layer, n_heads, head_dim)
                )
            )
        return self

    def direction_vector(
        self,
        ordered_heads: Sequence[Tuple[int, int]],
    ) -> np.ndarray:
        vectors: List[np.ndarray] = []
        missing = []
        for head in ordered_heads:
            if head not in self.captured:
                missing.append(hname(head))
            else:
                vectors.append(np.asarray(self.captured[head], dtype=np.float32))
        if missing:
            raise RuntimeError(f"Missing direction readout captures: {missing}")
        return np.concatenate(vectors, axis=0)

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def replace_first_3d_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return tuple(items)
        raise RuntimeError("Attention tuple output has no 3D hidden tensor")

    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return items
        raise RuntimeError("Attention list output has no 3D hidden tensor")

    raise RuntimeError(f"Unsupported attention output type {type(output).__name__}")


def first_3d_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(f"Expected 3D output, got {tuple(output.shape)}")
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not locate attention hidden output")


class RemovePromptLastWrite:
    """
    Subtract an already-computed post-W_O residual vector from selected layers'
    attention output at prompt-last.

    The vector is the sum of selected heads' CLEAN object-only contributions.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        prompt_last: int,
        remove_by_layer: Mapping[int, np.ndarray],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.prompt_last = int(prompt_last)
        self.remove_by_layer = {
            int(layer): np.asarray(value, dtype=np.float32)
            for layer, value in remove_by_layer.items()
        }
        self.handles: List[Any] = []
        self.events: Dict[int, int] = defaultdict(int)

    def __enter__(self) -> "RemovePromptLastWrite":
        for layer, vector in self.remove_by_layer.items():
            attention = resolve_attention(
                self.attention_helper, self.decoder_layers, layer
            )

            def make_hook(layer_index: int, residual: np.ndarray):
                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    hidden = first_3d_tensor(output)
                    if not 0 <= self.prompt_last < int(hidden.shape[1]):
                        raise RuntimeError(
                            f"L{layer_index}: prompt-last {self.prompt_last} "
                            f"outside S={int(hidden.shape[1])}"
                        )
                    if residual.shape[0] != int(hidden.shape[-1]):
                        raise RuntimeError(
                            f"L{layer_index}: removal D={residual.shape[0]} "
                            f"!= hidden D={int(hidden.shape[-1])}"
                        )
                    modified = hidden.clone()
                    delta = torch.as_tensor(
                        residual,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    modified[0, self.prompt_last] -= delta
                    self.events[layer_index] += 1
                    return replace_first_3d_tensor(output, modified)
                return hook

            self.handles.append(
                attention.register_forward_hook(make_hook(layer, vector))
            )
        return self

    def validate(self) -> None:
        missing = [
            layer for layer in self.remove_by_layer
            if self.events[layer] < 1
        ]
        if missing:
            raise RuntimeError(
                f"Prompt-last write removal did not fire at layers {missing}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class LastTokenCapture:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        prompt_last: int,
    ) -> None:
        self.layer = decoder_layers[-1]
        self.prompt_last = int(prompt_last)
        self.handle: Any = None
        self.value: Optional[np.ndarray] = None
        self.events = 0

    def __enter__(self) -> "LastTokenCapture":
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = first_3d_tensor(output)
            if not 0 <= self.prompt_last < int(hidden.shape[1]):
                raise RuntimeError("prompt-last outside final-layer output")
            self.value = (
                hidden[0, self.prompt_last]
                .detach().float().cpu().numpy()
            )
            self.events += 1

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def get(self) -> np.ndarray:
        if self.value is None:
            raise RuntimeError("Final prompt-last state was not captured")
        return np.asarray(self.value, dtype=np.float32)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
            self.handle = None


# =============================================================================
# Readout extraction from attention traces
# =============================================================================

def trace_target_local(trace: Any, position: int) -> int:
    lookup = {
        int(global_position): int(local)
        for local, global_position in enumerate(trace.target_positions)
    }
    if int(position) not in lookup:
        raise KeyError(
            f"Target position {position} absent from {trace.target_positions}"
        )
    return lookup[int(position)]


def receiver_routing_vector(
    *,
    traces: Mapping[int, Any],
    receiver_heads: Sequence[Tuple[int, int]],
    prompt_last: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors: List[float] = []
    details: List[Dict[str, Any]] = []

    subj_index = torch.as_tensor(
        sorted(set(map(int, subject_positions))),
        dtype=torch.long,
    )
    ref_index = torch.as_tensor(
        sorted(set(map(int, reference_positions))),
        dtype=torch.long,
    )

    for layer, head in receiver_heads:
        trace = traces[int(layer)]
        local = trace_target_local(trace, prompt_last)
        weights = trace.attention_weights[int(head), local].float()

        subj_mass = float(weights.index_select(0, subj_index).sum())
        ref_mass = float(weights.index_select(0, ref_index).sum())
        total = subj_mass + ref_mass
        balance = (
            min(subj_mass, ref_mass) / max(subj_mass, ref_mass)
            if max(subj_mass, ref_mass) > 1e-12
            else float("nan")
        )

        # Keep both absolute object mass and role balance in the vector.
        vectors.extend([subj_mass, ref_mass])
        details.append({
            "head": hname((layer, head)),
            "subject_mass": subj_mass,
            "reference_mass": ref_mass,
            "object_mass": total,
            "balance": balance,
        })

    return np.asarray(vectors, dtype=np.float32), details


def receiver_write_vectors(
    *,
    traces: Mapping[int, Any],
    receiver_heads: Sequence[Tuple[int, int]],
    receiver_module: Any,
    prompt_last: int,
    object_positions: Sequence[int],
    effective_removed_heads: set[Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray], List[Dict[str, Any]]]:
    concatenated: List[np.ndarray] = []
    by_head: Dict[Tuple[int, int], np.ndarray] = {}
    details: List[Dict[str, Any]] = []

    for layer, head in receiver_heads:
        trace = traces[int(layer)]
        write, mass = receiver_module.object_head_write(
            trace=trace,
            head=int(head),
            target_position=int(prompt_last),
            source_positions=object_positions,
        )
        vector = write.detach().float().cpu().numpy().astype(np.float32)
        raw_norm = float(np.linalg.norm(vector))

        if (int(layer), int(head)) in effective_removed_heads:
            effective = np.zeros_like(vector)
        else:
            effective = vector

        by_head[(int(layer), int(head))] = vector
        concatenated.append(effective)
        details.append({
            "head": hname((layer, head)),
            "object_attention_mass": float(mass),
            "raw_write_norm": raw_norm,
            "effective_write_norm": float(np.linalg.norm(effective)),
            "removed": bool((int(layer), int(head)) in effective_removed_heads),
        })

    return np.concatenate(concatenated, axis=0), by_head, details


def removal_vectors_from_clean(
    *,
    clean_write_by_head: Mapping[Tuple[int, int], np.ndarray],
    heads_to_remove: Sequence[Tuple[int, int]],
) -> Dict[int, np.ndarray]:
    by_layer: Dict[int, np.ndarray] = {}
    for head in heads_to_remove:
        if head not in clean_write_by_head:
            raise RuntimeError(
                f"Clean object-write vector missing for {hname(head)}"
            )
        layer = int(head[0])
        value = np.asarray(clean_write_by_head[head], dtype=np.float32)
        if layer not in by_layer:
            by_layer[layer] = np.zeros_like(value)
        by_layer[layer] += value
    return by_layer


# =============================================================================
# One condition
# =============================================================================

@dataclass
class ConditionResult:
    name: str
    scores: Dict[str, float]
    prediction: str
    margin: float
    direction_vector: np.ndarray
    routing_vector: np.ndarray
    receiver_write_vector: np.ndarray
    last_token_vector: np.ndarray
    routing_details: List[Dict[str, Any]]
    write_details: List[Dict[str, Any]]
    write_by_head: Dict[Tuple[int, int], np.ndarray]


@torch.inference_mode()
def run_condition(
    *,
    name: str,
    model: Any,
    batch: Mapping[str, Any],
    gt: str,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    attention_helper: Any,
    receiver_module: Any,
    v3: Any,
    zero_heads: Sequence[Tuple[int, int]],
    direction_readout_heads: Sequence[Tuple[int, int]],
    receiver_readout_heads: Sequence[Tuple[int, int]],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
    trace_layers: Sequence[int],
    trace_layer_chunk: int,
    remove_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    effective_removed_heads: Optional[set[Tuple[int, int]]] = None,
) -> ConditionResult:
    remove_by_layer = dict(remove_by_layer or {})
    effective_removed_heads = set(effective_removed_heads or set())

    controller = PreWOObjectController(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        zero_heads=zero_heads,
        capture_heads=direction_readout_heads,
        subject_positions=subject_positions,
        reference_positions=reference_positions,
    )
    remover = RemovePromptLastWrite(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        prompt_last=prompt_last,
        remove_by_layer=remove_by_layer,
    )
    last_capture = LastTokenCapture(
        decoder_layers=decoder_layers,
        prompt_last=prompt_last,
    )

    target_positions = sorted(
        set(
            map(int, subject_positions)
        )
        | set(map(int, reference_positions))
        | {int(prompt_last)}
    )

    try:
        with controller, remover, last_capture:
            result, traces = v3.trace_prompt_chunks(
                attention_helper=attention_helper,
                model=model,
                batch=batch,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                layers=trace_layers,
                target_positions=target_positions,
                chunk_size=trace_layer_chunk,
            )
        if remove_by_layer:
            remover.validate()

        direction_vector = controller.direction_vector(direction_readout_heads)
        routing_vector, routing_details = receiver_routing_vector(
            traces=traces,
            receiver_heads=receiver_readout_heads,
            prompt_last=prompt_last,
            subject_positions=subject_positions,
            reference_positions=reference_positions,
        )
        object_positions = sorted(
            set(map(int, subject_positions))
            | set(map(int, reference_positions))
        )
        write_vector, write_by_head, write_details = receiver_write_vectors(
            traces=traces,
            receiver_heads=receiver_readout_heads,
            receiver_module=receiver_module,
            prompt_last=prompt_last,
            object_positions=object_positions,
            effective_removed_heads=effective_removed_heads,
        )

        scores = {
            relation: float(result["scores"][relation])
            for relation in RELATIONS
        }
        margin = relation_margin_from_scores(scores, gt)

        return ConditionResult(
            name=name,
            scores=scores,
            prediction=str(result["prediction"]),
            margin=float(margin),
            direction_vector=direction_vector,
            routing_vector=routing_vector,
            receiver_write_vector=write_vector,
            last_token_vector=last_capture.get(),
            routing_details=routing_details,
            write_details=write_details,
            write_by_head=write_by_head,
        )
    finally:
        controller.close()
        remover.close()


# =============================================================================
# Sample metrics and aggregation
# =============================================================================

def compare_condition(
    *,
    sid: int,
    gt: str,
    status: str,
    clean: ConditionResult,
    swapped: ConditionResult,
    intervention: ConditionResult,
    matched_control: str,
) -> Dict[str, Any]:
    direction = axis_metrics(
        clean.direction_vector,
        swapped.direction_vector,
        intervention.direction_vector,
    )
    routing = axis_metrics(
        clean.routing_vector,
        swapped.routing_vector,
        intervention.routing_vector,
    )
    write = axis_metrics(
        clean.receiver_write_vector,
        swapped.receiver_write_vector,
        intervention.receiver_write_vector,
    )
    last = axis_metrics(
        clean.last_token_vector,
        swapped.last_token_vector,
        intervention.last_token_vector,
    )

    denominator = clean.margin - swapped.margin
    answer_norm = (
        (clean.margin - intervention.margin) / denominator
        if abs(denominator) > 1e-12
        else float("nan")
    )
    answer_abs_norm = (
        abs(clean.margin - intervention.margin) / abs(denominator)
        if abs(denominator) > 1e-12
        else float("nan")
    )

    return {
        "script_version": SCRIPT_VERSION,
        "sid": int(sid),
        "gt": gt,
        "pair_status": status,
        "condition": intervention.name,
        "matched_control": matched_control,

        "clean_prediction": clean.prediction,
        "swapped_prediction": swapped.prediction,
        "intervention_prediction": intervention.prediction,
        "prediction_changed": bool(intervention.prediction != clean.prediction),

        "clean_margin": clean.margin,
        "swapped_margin": swapped.margin,
        "intervention_margin": intervention.margin,
        "margin_denominator": denominator,
        "normalized_answer_damage": answer_norm,
        "absolute_normalized_answer_damage": answer_abs_norm,
        "crossed_decision_boundary": bool(clean.margin > 0 >= intervention.margin),

        "relation_axis_shift": direction["axis_shift"],
        "relation_relative_l2": direction["relative_l2"],
        "relation_cosine_to_clean": direction["cosine_to_clean"],

        "routing_axis_shift": routing["axis_shift"],
        "routing_relative_l2": routing["relative_l2"],
        "routing_cosine_to_clean": routing["cosine_to_clean"],

        "write_axis_shift": write["axis_shift"],
        "write_relative_l2": write["relative_l2"],
        "write_cosine_to_clean": write["cosine_to_clean"],

        "last_axis_shift": last["axis_shift"],
        "last_relative_l2": last["relative_l2"],
        "last_cosine_to_clean": last["cosine_to_clean"],

        "clean_receiver_object_mass_mean": safe_mean(
            d["object_mass"] for d in clean.routing_details
        ),
        "intervention_receiver_object_mass_mean": safe_mean(
            d["object_mass"] for d in intervention.routing_details
        ),
        "clean_receiver_balance_mean": safe_mean(
            d["balance"] for d in clean.routing_details
        ),
        "intervention_receiver_balance_mean": safe_mean(
            d["balance"] for d in intervention.routing_details
        ),
    }


def aggregate_conditions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append(row)

    metrics = (
        "relation_relative_l2",
        "routing_relative_l2",
        "write_relative_l2",
        "last_relative_l2",
        "normalized_answer_damage",
        "absolute_normalized_answer_damage",
        "relation_axis_shift",
        "routing_axis_shift",
        "write_axis_shift",
        "last_axis_shift",
    )

    result: List[Dict[str, Any]] = []
    for condition, values in groups.items():
        row: Dict[str, Any] = {
            "condition": condition,
            "matched_control": str(values[0]["matched_control"]),
            "N": len(values),
            "prediction_change_rate": safe_mean(
                int(bool(v["prediction_changed"])) for v in values
            ),
            "boundary_cross_rate": safe_mean(
                int(bool(v["crossed_decision_boundary"])) for v in values
            ),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = safe_mean(v[metric] for v in values)
            row[f"median_{metric}"] = safe_median(v[metric] for v in values)
            row[f"std_{metric}"] = safe_std(v[metric] for v in values)
        result.append(row)

    preferred = [
        "stage1_direction",
        "control_stage1",
        "stage2_centroid",
        "control_stage2",
        "stage3_receiver",
        "control_stage3",
    ]
    order = {name: i for i, name in enumerate(preferred)}
    result.sort(key=lambda r: order.get(str(r["condition"]), 999))
    return result


def dependency_matrix(summary: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {str(row["condition"]): row for row in summary}
    pairs = [
        ("stage1_direction", "control_stage1"),
        ("stage2_centroid", "control_stage2"),
        ("stage3_receiver", "control_stage3"),
    ]

    output: List[Dict[str, Any]] = []
    for stage, control in pairs:
        s = by_name.get(stage)
        c = by_name.get(control)
        if s is None or c is None:
            continue
        output.append({
            "stage": stage,
            "control": control,
            "N": int(s["N"]),

            # Raw normalized damage.
            "relation_damage": safe_float(s["mean_relation_relative_l2"]),
            "routing_damage": safe_float(s["mean_routing_relative_l2"]),
            "receiver_write_damage": safe_float(s["mean_write_relative_l2"]),
            "last_token_damage": safe_float(s["mean_last_relative_l2"]),
            "answer_damage": safe_float(s["mean_absolute_normalized_answer_damage"]),

            # Matched-control excess is the main specialization statistic.
            "relation_excess_vs_control": (
                safe_float(s["mean_relation_relative_l2"])
                - safe_float(c["mean_relation_relative_l2"])
            ),
            "routing_excess_vs_control": (
                safe_float(s["mean_routing_relative_l2"])
                - safe_float(c["mean_routing_relative_l2"])
            ),
            "receiver_write_excess_vs_control": (
                safe_float(s["mean_write_relative_l2"])
                - safe_float(c["mean_write_relative_l2"])
            ),
            "last_token_excess_vs_control": (
                safe_float(s["mean_last_relative_l2"])
                - safe_float(c["mean_last_relative_l2"])
            ),
            "answer_excess_vs_control": (
                safe_float(s["mean_absolute_normalized_answer_damage"])
                - safe_float(c["mean_absolute_normalized_answer_damage"])
            ),

            "signed_answer_effect": safe_float(s["mean_normalized_answer_damage"]),
            "prediction_change_rate": safe_float(s["prediction_change_rate"]),
            "control_prediction_change_rate": safe_float(c["prediction_change_rate"]),
        })
    return output


def print_matrix(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("THREE-STAGE DEPENDENCY MATRIX")
    print("=" * 132)
    print(
        "stage              relation    routing     write      last     answer  "
        "| excess vs matched control: relation routing write last answer"
    )
    print("-" * 132)
    for row in rows:
        print(
            f"{str(row['stage']):<18s} "
            f"{safe_float(row['relation_damage']):>9.4f} "
            f"{safe_float(row['routing_damage']):>10.4f} "
            f"{safe_float(row['receiver_write_damage']):>9.4f} "
            f"{safe_float(row['last_token_damage']):>9.4f} "
            f"{safe_float(row['answer_damage']):>9.4f} | "
            f"{safe_float(row['relation_excess_vs_control']):>8.4f} "
            f"{safe_float(row['routing_excess_vs_control']):>7.4f} "
            f"{safe_float(row['receiver_write_excess_vs_control']):>7.4f} "
            f"{safe_float(row['last_token_excess_vs_control']):>7.4f} "
            f"{safe_float(row['answer_excess_vs_control']):>7.4f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import companion modules.
    ioi = import_file(Path(args.ioi_script), "_three_stage_ioi")
    producer = import_file(Path(args.producer_script), "_three_stage_producer")
    receiver = import_file(Path(args.receiver_script), "_three_stage_receiver")
    v3 = import_file(Path(args.v3_script), "_three_stage_v3")
    base = import_file(Path(args.base_script), "_three_stage_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "_three_stage_attention",
    )

    # Source rows.
    source_dir = Path(args.source_output_dir)
    config_path = source_dir / "config.json"
    extraction_path = source_dir / "extraction.jsonl"
    if not config_path.exists() or not extraction_path.exists():
        raise FileNotFoundError(
            f"Missing v3 source cache at {source_dir}. Need config.json + extraction.jsonl."
        )
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_rows = read_jsonl(extraction_path)

    if str(source_config.get("model")) != args.model:
        raise RuntimeError(
            f"source model={source_config.get('model')} vs --model={args.model}"
        )

    # Filter samples.
    eligible: List[Dict[str, Any]] = []
    for row in source_rows:
        status = str(row.get("generation_pair_status", ""))
        if args.sample_status != "all" and status != args.sample_status:
            continue
        gt = str(row["gt"])
        clean_margin = float(row["baseline_lm_margin"])
        swapped_logits = np.asarray(row["swapped_relation_logits"], dtype=np.float64)
        gt_i = RELATIONS.index(gt)
        op_i = RELATIONS.index(OPPOSITE[gt])
        swapped_margin = float(swapped_logits[gt_i] - swapped_logits[op_i])
        denom = clean_margin - swapped_margin
        if abs(denom) < args.min_margin_denominator:
            continue
        if args.require_margin_sign and not (
            clean_margin > 0 and swapped_margin < 0
        ):
            continue
        eligible.append(dict(row))

    selected_rows = stratified_limit(
        eligible,
        args.sample_max_samples,
        args.seed,
    )
    if not selected_rows:
        raise RuntimeError("No eligible samples")

    # Load model / data.
    model = None
    processor = None
    try:
        (
            model,
            processor,
            spec,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer.load_model_bundle(args=args, base=base)

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        stage1 = parse_heads(args.stage1_heads)
        stage2 = parse_heads(args.stage2_heads)
        stage3 = parse_heads(args.stage3_heads)
        direction_readout = parse_heads(args.direction_readout_heads)

        all_named = set(stage1) | set(stage2) | set(stage3) | set(direction_readout)
        if len(stage1) != len(set(stage1)):
            raise RuntimeError("Stage I duplicates")
        if len(stage2) != len(set(stage2)):
            raise RuntimeError("Stage II duplicates")
        if len(stage3) != len(set(stage3)):
            raise RuntimeError("Stage III duplicates")

        # Intervention groups should be disjoint for the first clean test.
        overlaps = {
            "stage1_stage2": sorted(set(stage1) & set(stage2)),
            "stage1_stage3": sorted(set(stage1) & set(stage3)),
            "stage2_stage3": sorted(set(stage2) & set(stage3)),
        }
        if any(overlaps.values()):
            raise RuntimeError(
                "Default specialization test requires disjoint stage bundles; "
                f"overlaps={overlaps}"
            )

        for label, heads in (
            ("stage1", stage1),
            ("stage2", stage2),
            ("stage3", stage3),
            ("direction_readout", direction_readout),
        ):
            validate_heads(
                heads=heads,
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver_module=receiver,
                label=label,
            )

        # One layer-matched random bundle per intervention.
        excluded = set(all_named)
        random1 = matched_random_heads(
            target=stage1,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            excluded=excluded,
            seed=args.seed + 101,
        )
        excluded |= set(random1)

        random2 = matched_random_heads(
            target=stage2,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            excluded=excluded,
            seed=args.seed + 202,
        )
        excluded |= set(random2)

        random3 = matched_random_heads(
            target=stage3,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            excluded=excluded,
            seed=args.seed + 303,
        )

        # Receiver readout is the hypothesized Stage III bundle.
        receiver_readout = list(stage3)

        # Trace all layers needed for Stage III and its random same-layer controls.
        trace_layers = sorted(
            set(layer for layer, _ in receiver_readout + random3)
        )

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": getattr(spec, "repo_id", ""),
            "dataset": args.dataset,
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "object_state": args.object_state,
            "stage1_direction_heads": [hname(h) for h in stage1],
            "stage2_centroid_heads": [hname(h) for h in stage2],
            "stage3_receiver_heads": [hname(h) for h in stage3],
            "direction_readout_heads": [hname(h) for h in direction_readout],
            "control_stage1_heads": [hname(h) for h in random1],
            "control_stage2_heads": [hname(h) for h in random2],
            "control_stage3_heads": [hname(h) for h in random3],
            "trace_layers": trace_layers,
            "sample_status": args.sample_status,
            "sample_max_samples": args.sample_max_samples,
            "selected_samples": len(selected_rows),
            "seed": args.seed,
            "interventions": {
                "stage1_direction": "zero pre-WO head slices at object tokens",
                "stage2_centroid": "zero pre-WO head slices at object tokens",
                "stage3_receiver": (
                    "subtract clean selected object->prompt-last post-WO head writes "
                    "from attention output at prompt-last"
                ),
            },
            "readouts": {
                "M1": "role-ordered subject-reference pre-WO vector at held-out L26 direction heads",
                "M2": "Stage III receiver [attention(last,subject), attention(last,reference)]",
                "M3": "concatenated Stage III receiver object->last post-WO writes",
                "M4": "final decoder prompt-last state",
                "M5": "GT-vs-opposite next-token relation margin",
            },
            "audit": audit,
        }
        write_json(output_dir / "config.json", config)

        print("\n" + "=" * 120)
        print("THREE-STAGE CAUSAL SPECIALIZATION TEST")
        print("=" * 120)
        print("Stage I Direction :", ", ".join(hname(h) for h in stage1))
        print("Stage II Centroid :", ", ".join(hname(h) for h in stage2))
        print("Stage III Receiver:", ", ".join(hname(h) for h in stage3))
        print("Direction readout :", ", ".join(hname(h) for h in direction_readout))
        print("Control I         :", ", ".join(hname(h) for h in random1))
        print("Control II        :", ", ".join(hname(h) for h in random2))
        print("Control III       :", ", ".join(hname(h) for h in random3))
        print("Trace layers      :", trace_layers)
        print("Samples           :", len(selected_rows))
        print("=" * 120, flush=True)

        sample_path = output_dir / "sample_metrics.jsonl"
        detail_path = output_dir / "sample_internal_details.jsonl"
        error_path = output_dir / "errors.jsonl"

        all_rows: List[Dict[str, Any]] = []

        for sample_index, source_row in enumerate(
            tqdm(selected_rows, desc=f"three-stage:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                pair = receiver.prepare_pair(
                    args=args,
                    row=source_row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=torch.device(args.device),
                )

                gt = str(pair.gt)
                status = str(source_row["generation_pair_status"])

                # -------------------------------------------------------------
                # Clean original and clean swapped baselines.
                # -------------------------------------------------------------
                clean = run_condition(
                    name="clean",
                    model=model,
                    batch=pair.original_batch,
                    gt=gt,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    attention_helper=attention_helper,
                    receiver_module=receiver,
                    v3=v3,
                    zero_heads=[],
                    direction_readout_heads=direction_readout,
                    receiver_readout_heads=receiver_readout,
                    subject_positions=pair.original_a_positions,
                    reference_positions=pair.original_b_positions,
                    prompt_last=pair.original_prompt_last,
                    trace_layers=trace_layers,
                    trace_layer_chunk=args.trace_layer_chunk,
                )

                # Swapped prompt role order is B(subject) - A(reference).
                swapped_gt = OPPOSITE[gt]
                swapped = run_condition(
                    name="swapped",
                    model=model,
                    batch=pair.swapped_batch,
                    gt=swapped_gt,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    attention_helper=attention_helper,
                    receiver_module=receiver,
                    v3=v3,
                    zero_heads=[],
                    direction_readout_heads=direction_readout,
                    receiver_readout_heads=receiver_readout,
                    subject_positions=pair.swapped_b_positions,
                    reference_positions=pair.swapped_a_positions,
                    prompt_last=pair.swapped_prompt_last,
                    trace_layers=trace_layers,
                    trace_layer_chunk=args.trace_layer_chunk,
                )

                # Convert swapped margin to original fixed axis.
                # swapped.margin is opposite-GT vs original-GT; negate it.
                swapped_fixed = ConditionResult(
                    name=swapped.name,
                    scores=swapped.scores,
                    prediction=swapped.prediction,
                    margin=-float(swapped.margin),
                    direction_vector=swapped.direction_vector,
                    routing_vector=swapped.routing_vector,
                    receiver_write_vector=swapped.receiver_write_vector,
                    last_token_vector=swapped.last_token_vector,
                    routing_details=swapped.routing_details,
                    write_details=swapped.write_details,
                    write_by_head=swapped.write_by_head,
                )

                # Clean object writes for Stage III and matched control.
                # Need traces for random3 heads too.  Reconstruct them from a
                # small auxiliary clean trace only when random3 contains heads
                # outside receiver_readout.
                all_write_heads = list(dict.fromkeys(receiver_readout + random3))

                # We need clean write vectors for random3. Run a lightweight
                # clean condition with write readout expanded; other outputs ignored.
                clean_allwrites = run_condition(
                    name="clean_allwrites",
                    model=model,
                    batch=pair.original_batch,
                    gt=gt,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    attention_helper=attention_helper,
                    receiver_module=receiver,
                    v3=v3,
                    zero_heads=[],
                    direction_readout_heads=direction_readout,
                    receiver_readout_heads=all_write_heads,
                    subject_positions=pair.original_a_positions,
                    reference_positions=pair.original_b_positions,
                    prompt_last=pair.original_prompt_last,
                    trace_layers=trace_layers,
                    trace_layer_chunk=args.trace_layer_chunk,
                )

                stage3_remove = removal_vectors_from_clean(
                    clean_write_by_head=clean_allwrites.write_by_head,
                    heads_to_remove=stage3,
                )
                control3_remove = removal_vectors_from_clean(
                    clean_write_by_head=clean_allwrites.write_by_head,
                    heads_to_remove=random3,
                )

                # -------------------------------------------------------------
                # Six interventions: three hypotheses + three matched controls.
                # -------------------------------------------------------------
                intervention_specs = [
                    (
                        "stage1_direction",
                        stage1,
                        {},
                        set(),
                        "control_stage1",
                    ),
                    (
                        "control_stage1",
                        random1,
                        {},
                        set(),
                        "control_stage1",
                    ),
                    (
                        "stage2_centroid",
                        stage2,
                        {},
                        set(),
                        "control_stage2",
                    ),
                    (
                        "control_stage2",
                        random2,
                        {},
                        set(),
                        "control_stage2",
                    ),
                    (
                        "stage3_receiver",
                        [],
                        stage3_remove,
                        set(stage3),
                        "control_stage3",
                    ),
                    (
                        "control_stage3",
                        [],
                        control3_remove,
                        set(),
                        "control_stage3",
                    ),
                ]

                sample_details: Dict[str, Any] = {
                    "sid": int(pair.sid),
                    "gt": gt,
                    "status": status,
                    "clean_prediction": clean.prediction,
                    "swapped_prediction": swapped.prediction,
                    "clean_margin": clean.margin,
                    "swapped_margin_fixed_axis": swapped_fixed.margin,
                    "clean_routing": clean.routing_details,
                    "clean_writes": clean.write_details,
                }

                for (
                    condition_name,
                    zero_heads,
                    remove_by_layer,
                    effective_removed,
                    matched_control,
                ) in intervention_specs:
                    intervention = run_condition(
                        name=condition_name,
                        model=model,
                        batch=pair.original_batch,
                        gt=gt,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        attention_helper=attention_helper,
                        receiver_module=receiver,
                        v3=v3,
                        zero_heads=zero_heads,
                        direction_readout_heads=direction_readout,
                        receiver_readout_heads=receiver_readout,
                        subject_positions=pair.original_a_positions,
                        reference_positions=pair.original_b_positions,
                        prompt_last=pair.original_prompt_last,
                        trace_layers=trace_layers,
                        trace_layer_chunk=args.trace_layer_chunk,
                        remove_by_layer=remove_by_layer,
                        effective_removed_heads=effective_removed,
                    )

                    row = compare_condition(
                        sid=pair.sid,
                        gt=gt,
                        status=status,
                        clean=clean,
                        swapped=swapped_fixed,
                        intervention=intervention,
                        matched_control=matched_control,
                    )
                    append_jsonl(sample_path, row)
                    all_rows.append(row)

                    sample_details[condition_name] = {
                        "prediction": intervention.prediction,
                        "margin": intervention.margin,
                        "routing": intervention.routing_details,
                        "writes": intervention.write_details,
                    }

                append_jsonl(detail_path, sample_details)

                if args.print_every > 0 and sample_index % args.print_every == 0:
                    current = [
                        r for r in all_rows
                        if int(r["sid"]) == int(pair.sid)
                    ]
                    compact = " | ".join(
                        f"{r['condition']}:"
                        f" rel={safe_float(r['relation_relative_l2']):.2f}"
                        f" route={safe_float(r['routing_relative_l2']):.2f}"
                        f" write={safe_float(r['write_relative_l2']):.2f}"
                        f" last={safe_float(r['last_relative_l2']):.2f}"
                        f" ans={safe_float(r['normalized_answer_damage']):+.2f}"
                        for r in current
                    )
                    tqdm.write(
                        f"sid={pair.sid} gt={gt} clean={clean.prediction} | {compact}"
                    )

            except Exception as exc:
                error = {
                    "sid": int(source_row["sid"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, error)
                print(
                    f"\n[ERROR sid={source_row['sid']}] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and (
                    args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        if not all_rows:
            raise RuntimeError("No sample metrics generated")

        summary = aggregate_conditions(all_rows)
        matrix = dependency_matrix(summary)

        write_csv(output_dir / "condition_summary.csv", summary)
        write_csv(output_dir / "dependency_matrix.csv", matrix)

        print_matrix(matrix)

        report_lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"N samples: {len(set(int(r['sid']) for r in all_rows))}",
            "",
            "HOW TO READ dependency_matrix.csv",
            "",
            "Each stage row reports normalized internal damage, where clean->swapped",
            "semantic change is approximately one unit.  More important are the",
            "*_excess_vs_control columns, which subtract same-layer random-head damage.",
            "",
            "Evidence FOR the three-stage specialization hypothesis:",
            "",
            "1. Stage I Direction:",
            "   relation_excess_vs_control > 0 and downstream write/last/answer effects > 0.",
            "   Interpretation: removing relation-bearing object states propagates downstream.",
            "",
            "2. Stage II Centroid:",
            "   routing_excess_vs_control is clearly positive, ideally larger than its",
            "   relation-specific excess.  Interpretation: these heads disproportionately",
            "   alter how Stage III receivers read subject/reference tokens.",
            "",
            "3. Stage III Receiver:",
            "   receiver_write_excess_vs_control, last_token_excess_vs_control and",
            "   answer_excess_vs_control are positive, while relation/routing excess stays",
            "   near zero.  This is the cleanest evidence that the receiver's job is",
            "   object->prompt-last integration rather than upstream relation encoding.",
            "",
            "Evidence AGAINST a clean three-stage story:",
            "",
            "- All three interventions damage the same internal metrics equally.",
            "- Stage II centroid bundle does not affect receiver routing beyond controls.",
            "- Stage III intervention substantially changes upstream relation/routing",
            "  readouts (it should not, because the intervention is post-W_O at last).",
            "- Random controls are as strong as the named bundles.",
            "",
            "Important limitation:",
            "Stage I and II interventions zero the selected head output at object tokens;",
            "Stage II is therefore a head-function ablation, not a pure QK-only routing",
            "intervention.  If Stage II shows promising routing selectivity, the next test",
            "should hold V fixed and perturb only its QK/attention pattern.",
        ]
        (output_dir / "report.txt").write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        print(f"  {output_dir / 'sample_metrics.jsonl'}")
        print(f"  {output_dir / 'sample_internal_details.jsonl'}")
        print(f"  {output_dir / 'condition_summary.csv'}")
        print(f"  {output_dir / 'dependency_matrix.csv'}")
        print(f"  {output_dir / 'report.txt'}")

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
