#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Writer -> DAS-U causal validation.

Purpose
=======
We already learned low-dimensional causal relation subspaces U with
validate_coco_relation_das_v1.py.  This script asks a different question:

    Which candidate Stage-I attention heads causally WRITE into those
    relation subspaces?

The experiment does NOT ablate heads.

For each candidate writer head h:
  1) Same image, two prompt branches:
       target/original: A relative to B = r
       source/swapped : B relative to A = opposite(r)

  2) Capture the REAL source/swapped pre-W_O object states of head h.

  3) Run target/original again and replace only that head's object states:
       source subject B   -> target subject A
       source reference A -> target reference B

  4) Let the downstream network run normally.

  5) At several learned DAS relation subspaces U (default L22/L23/L24/L25,
     D16), measure whether the resulting U coordinates move from the target
     state toward the source/opposite state.

This directly tests:
    writer head state -> downstream causal relation subspace

Candidate heads
===============
By default candidates are the raw-image Direction Top20, ranked by
`img_accuracy_mean`, restricted to Stage-I-ish layers L18-L23.

A writer at layer Lw is only evaluated against U layers Lu >= Lw.
Upstream U layers are marked invalid/skipped.

Main U-space metrics
====================
For each sample and writer/U pair:

    t = target/original U coordinates
    s = source/swapped ROLE-aligned U coordinates
    p = U coordinates after writer natural-state patch

where each state is the concatenation of subject and reference projected
coordinates:

    [U^T h_subject ; U^T h_reference]

Define d = s - t, m = p - t.

source_progress:
    dot(m, d) / ||d||^2

Interpretation:
    0 = stayed at target
    1 = reached source along the target->source U direction
   >1 = overshot source direction
   <0 = moved away from source

source_side:
    ||p-s|| < ||p-t||

fraction_distance_closed:
    1 - ||p-s|| / ||t-s||

off_axis_ratio:
    ||m - source_progress*d|| / ||d||

A true relation writer should show positive source_progress / distance closure
in downstream U, preferably across multiple U layers.

Controls
========
Phase 1:
    Rank all Top20 candidate writers using ROLE natural-state patch.

Phase 2:
    For only the strongest `--confirm-top-n` writers (default 5), run:
      * identity patch:
            source A -> target A
            source B -> target B
      * same-layer matched-random head ROLE patch

This keeps the full Top20 scan tractable while giving strong controls on the
best candidates.

Evaluation samples
==================
Default --sample-scope das_eval reuses only eval_sids from the DAS run's
config.json, so the Writer->U test stays separate from DAS basis training.

Default --pair-status both_correct further requires both original and swapped
branches to be correct, making the source/opposite state semantically clean.

Recommended run
===============

CUDA_VISIBLE_DEVICES=0 python -u validate_writer_to_das_u_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --direction-results output/qwen3b_coco_head_object_residual_direction_probe/head_results.csv \
  --das-dir output/qwen3b_relation_das_full \
  --writer-layers 18-23 \
  --top-k 20 \
  --u-layers 22,23,24,25 \
  --u-dim 16 \
  --sample-scope das_eval \
  --pair-status both_correct \
  --max-samples 0 \
  --confirm-top-n 5 \
  --replacement-mode tokenwise_resample \
  --device cuda:0 \
  --output-dir output/qwen3b_writer_to_das_u_top20 \
  --overwrite

Smoke test
==========

CUDA_VISIBLE_DEVICES=0 python -u validate_writer_to_das_u_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --direction-results output/qwen3b_coco_head_object_residual_direction_probe/head_results.csv \
  --das-dir output/qwen3b_relation_das_full \
  --writer-layers 18-23 \
  --top-k 20 \
  --u-layers 22,23,24,25 \
  --u-dim 16 \
  --sample-scope das_eval \
  --pair-status both_correct \
  --max-samples 80 \
  --confirm-top-n 3 \
  --device cuda:0 \
  --output-dir output/qwen3b_writer_to_das_u_smoke \
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
import re
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


VERSION = "writer-to-das-u-v1"

RELATIONS = ("left", "right", "above", "below")
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
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", default="mean", choices=("mean", "last"))

    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
    )
    p.add_argument(
        "--direction-results",
        default="output/qwen3b_coco_head_object_residual_direction_probe/head_results.csv",
    )
    p.add_argument(
        "--rank-metric",
        default="img_accuracy_mean",
        help="Direction ranking metric.",
    )
    p.add_argument(
        "--writer-layers",
        default="18-23",
        help="Allowed writer candidate layer range/list, e.g. 18-23 or 19,20,21,22,23.",
    )
    p.add_argument("--top-k", type=int, default=20)

    p.add_argument(
        "--das-dir",
        default="output/qwen3b_relation_das_full",
        help="Directory containing basis_L*_D*.npz + config.json from DAS experiment.",
    )
    p.add_argument(
        "--u-layers",
        default="22,23,24,25",
        help="Relation-U layers to inspect simultaneously.",
    )
    p.add_argument(
        "--u-dim",
        type=int,
        default=16,
        help="Fixed DAS subspace dimension loaded for every --u-layers layer.",
    )

    p.add_argument(
        "--sample-scope",
        default="das_eval",
        choices=("das_eval", "all"),
        help="das_eval uses only held-out eval_sids from DAS config.json.",
    )
    p.add_argument(
        "--pair-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)

    p.add_argument(
        "--confirm-top-n",
        type=int,
        default=5,
        help="After TopK role scan, run identity/random controls only for strongest N writers. 0 disables.",
    )
    p.add_argument(
        "--replacement-mode",
        default="tokenwise_resample",
        choices=("tokenwise_resample", "pooled_broadcast", "mean_shift"),
    )

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=5)

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
    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def first_3d_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output) and output.ndim == 3:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not find 3D hidden tensor")


def parse_int_spec(text: str) -> List[int]:
    result: List[int] = []
    seen = set()
    for piece in str(text).split(","):
        piece = piece.strip().upper().replace("L", "")
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            aa, bb = int(a), int(b)
            values = range(min(aa, bb), max(aa, bb) + 1)
        else:
            values = [int(piece)]
        for v in values:
            if v not in seen:
                seen.add(v)
                result.append(v)
    if not result:
        raise ValueError(f"Empty integer specification: {text!r}")
    return sorted(result)


def hname(head: Tuple[int, int]) -> str:
    return f"L{int(head[0])}H{int(head[1]):02d}"


def safe_mean(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in rows]
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda r: int(r["sid"]))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    cursors = {r: 0 for r in RELATIONS}
    chosen: List[Dict[str, Any]] = []
    while len(chosen) < limit:
        moved = False
        for rel in RELATIONS:
            group = groups.get(rel, [])
            i = cursors[rel]
            if i < len(group) and len(chosen) < limit:
                chosen.append(group[i])
                cursors[rel] += 1
                moved = True
        if not moved:
            break
    return sorted(chosen, key=lambda r: int(r["sid"]))


# =============================================================================
# Direction candidate ranking
# =============================================================================

def load_writer_candidates(
    *,
    path: Path,
    metric: str,
    allowed_layers: Sequence[int],
    top_k: int,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    if not path.exists():
        candidates = sorted(Path("output").glob("**/head_results.csv"))
        msg = [f"Direction results not found: {path}"]
        if candidates:
            msg.append("Candidates:")
            msg.extend(f"  {p}" for p in candidates[:30])
        raise FileNotFoundError("\n".join(msg))

    allowed = set(map(int, allowed_layers))
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty direction results: {path}")
    if metric not in rows[0]:
        raise KeyError(
            f"Metric {metric!r} missing. Columns={list(rows[0].keys())}"
        )

    parsed: List[Tuple[float, Tuple[int, int]]] = []
    for row in rows:
        try:
            layer = int(row["layer"])
            head = int(row["head"])
            score = float(row[metric])
        except Exception:
            continue
        if layer not in allowed or not np.isfinite(score):
            continue
        parsed.append((score, (layer, head)))

    parsed.sort(key=lambda x: x[0], reverse=True)
    if len(parsed) < top_k:
        raise RuntimeError(
            f"Need Top{top_k} Stage-I candidates but only {len(parsed)} valid rows "
            f"in layers={sorted(allowed)}"
        )

    heads = [h for _, h in parsed[:top_k]]
    scores = {h: float(s) for s, h in parsed[:top_k]}
    return heads, scores


def validate_heads(
    heads: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver: Any,
) -> None:
    for layer, head in heads:
        if not 0 <= layer < len(decoder_layers):
            raise ValueError(f"{hname((layer, head))}: invalid layer")
        attn = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver.resolve_attention_shape(attn)
        nh = int(shape.n_query_heads)
        if not 0 <= head < nh:
            raise ValueError(f"{hname((layer, head))}: invalid head for n={nh}")


def matched_random_heads(
    *,
    target_heads: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver: Any,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    excluded = set(target_heads)
    used = set(target_heads)
    out: List[Tuple[int, int]] = []
    for layer, _head in target_heads:
        attn = attention_helper.resolve_self_attention(decoder_layers[layer])
        nh = int(receiver.resolve_attention_shape(attn).n_query_heads)
        choices = [
            (layer, h)
            for h in range(nh)
            if (layer, h) not in used
        ]
        if not choices:
            raise RuntimeError(f"No same-layer random control available at L{layer}")
        pick = rng.choice(choices)
        out.append(pick)
        used.add(pick)
    return out


# =============================================================================
# DAS bases
# =============================================================================

def load_das_bases(
    *,
    das_dir: Path,
    u_layers: Sequence[int],
    u_dim: int,
) -> Dict[int, np.ndarray]:
    bases: Dict[int, np.ndarray] = {}
    missing: List[Path] = []

    for layer in u_layers:
        path = das_dir / f"basis_L{layer}_D{u_dim}.npz"
        if not path.exists():
            missing.append(path)
            continue
        data = np.load(path)
        if "Q" not in data:
            raise KeyError(f"{path} lacks Q")
        q = np.asarray(data["Q"], dtype=np.float32)
        if q.ndim != 2 or int(q.shape[1]) != int(u_dim):
            raise RuntimeError(
                f"Bad Q shape in {path}: {q.shape}, expected [d_model,{u_dim}]"
            )
        # Defensive orthonormalization.
        q, _ = np.linalg.qr(q)
        bases[int(layer)] = q[:, :u_dim].astype(np.float32)

    if missing:
        available = sorted(das_dir.glob("basis_L*_D*.npz"))
        msg = ["Missing DAS bases:"]
        msg.extend(f"  {p}" for p in missing)
        if available:
            msg.append("Available:")
            msg.extend(f"  {p.name}" for p in available[:50])
        raise FileNotFoundError("\n".join(msg))

    return bases


# =============================================================================
# Source head + residual capture
# =============================================================================

class SourceEverythingCapture:
    """
    One swapped/source forward captures:
      * all candidate/random pre-W_O head object-token states
      * object residual A/B states at all requested DAS-U layers
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver: Any,
        heads: Sequence[Tuple[int, int]],
        u_layers: Sequence[int],
        source_a_positions: Sequence[int],
        source_b_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver = receiver
        self.by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head in heads:
            self.by_layer[int(layer)].append(int(head))
        self.u_layers = set(map(int, u_layers))
        self.a_positions = sorted(set(map(int, source_a_positions)))
        self.b_positions = sorted(set(map(int, source_b_positions)))
        self.handles: List[Any] = []
        self.head_states: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.residual_states: Dict[int, Dict[str, np.ndarray]] = {}

    def __enter__(self) -> "SourceEverythingCapture":
        for layer, heads in self.by_layer.items():
            attn = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"L{layer} attention lacks o_proj")
            nh = int(self.receiver.resolve_attention_shape(attn).n_query_heads)
            width = int(o_proj.weight.shape[1])
            if width % nh != 0:
                raise RuntimeError(f"L{layer}: o_proj input width not divisible by heads")
            hd = width // nh

            def make_head_hook(layer_index: int, head_ids: Sequence[int], head_dim: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None

                    ap = torch.as_tensor(
                        self.a_positions, device=x.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=x.device, dtype=torch.long
                    )
                    for head in head_ids:
                        lo = int(head) * head_dim
                        hi = lo + head_dim
                        a = x[0].index_select(0, ap)[:, lo:hi]
                        b = x[0].index_select(0, bp)[:, lo:hi]
                        self.head_states[(layer_index, int(head))] = {
                            "A_tokens": a.detach().float().cpu().numpy().astype(np.float32),
                            "B_tokens": b.detach().float().cpu().numpy().astype(np.float32),
                        }
                    return None
                return hook

            self.handles.append(
                o_proj.register_forward_pre_hook(
                    make_head_hook(layer, heads, hd)
                )
            )

        for layer in sorted(self.u_layers):
            block = self.decoder_layers[layer]

            def make_resid_hook(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if int(hidden.shape[0]) != 1:
                        return
                    ap = torch.as_tensor(
                        self.a_positions, device=hidden.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=hidden.device, dtype=torch.long
                    )
                    a = hidden[0].index_select(0, ap).mean(dim=0)
                    b = hidden[0].index_select(0, bp).mean(dim=0)
                    self.residual_states[layer_index] = {
                        "A": a.detach().float().cpu().numpy().astype(np.float32),
                        "B": b.detach().float().cpu().numpy().astype(np.float32),
                    }
                return hook

            self.handles.append(block.register_forward_hook(make_resid_hook(layer)))

        return self

    def validate(
        self,
        heads: Sequence[Tuple[int, int]],
        u_layers: Sequence[int],
    ) -> None:
        missing_h = [hname(h) for h in heads if h not in self.head_states]
        missing_u = [l for l in u_layers if l not in self.residual_states]
        if missing_h or missing_u:
            raise RuntimeError(
                f"Missing source captures heads={missing_h} U_layers={missing_u}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


# =============================================================================
# Target natural writer patch + U capture
# =============================================================================

def map_source_rows_to_target(
    *,
    source_rows: torch.Tensor,
    target_rows: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, str]:
    ns, ds = int(source_rows.shape[0]), int(source_rows.shape[1])
    nt, dt = int(target_rows.shape[0]), int(target_rows.shape[1])
    if ns < 1 or nt < 1 or ds != dt:
        raise ValueError(
            f"Bad rows source={tuple(source_rows.shape)} target={tuple(target_rows.shape)}"
        )

    if mode == "tokenwise_resample":
        if ns == nt:
            return source_rows, "exact_tokenwise"
        idx = torch.linspace(
            0, ns - 1, steps=nt,
            device=source_rows.device,
            dtype=torch.float32,
        ).round().long()
        return source_rows.index_select(0, idx), "nearest_source_row_resample"

    if mode == "pooled_broadcast":
        return (
            source_rows.mean(dim=0, keepdim=True).expand(nt, -1),
            "source_pooled_broadcast",
        )

    if mode == "mean_shift":
        sm = source_rows.mean(dim=0, keepdim=True)
        tm = target_rows.mean(dim=0, keepdim=True)
        return target_rows + (sm - tm), "target_cloud_shift_to_source_mean"

    raise ValueError(mode)


class TargetWriterPatchAndCapture:
    """
    Patch exactly one target pre-W_O head and capture downstream U-layer
    residual object states plus final next-token relation logits.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver: Any,
        writer_head: Tuple[int, int],
        source_head_state: Mapping[str, np.ndarray],
        alignment: str,
        target_a_positions: Sequence[int],
        target_b_positions: Sequence[int],
        capture_u_layers: Sequence[int],
        replacement_mode: str,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver = receiver
        self.writer_head = (int(writer_head[0]), int(writer_head[1]))
        self.source_head_state = source_head_state
        self.alignment = str(alignment)
        self.a_positions = sorted(set(map(int, target_a_positions)))
        self.b_positions = sorted(set(map(int, target_b_positions)))
        self.capture_u_layers = sorted(set(map(int, capture_u_layers)))
        self.replacement_mode = str(replacement_mode)

        self.handles: List[Any] = []
        self.residual_states: Dict[int, Dict[str, np.ndarray]] = {}
        self.patch_count = 0
        self.patch_stats: Dict[str, Any] = {}

    def __enter__(self) -> "TargetWriterPatchAndCapture":
        layer, head = self.writer_head
        attn = self.attention_helper.resolve_self_attention(
            self.decoder_layers[layer]
        )
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            raise RuntimeError(f"L{layer} attention lacks o_proj")
        nh = int(self.receiver.resolve_attention_shape(attn).n_query_heads)
        width = int(o_proj.weight.shape[1])
        if width % nh != 0:
            raise RuntimeError(f"L{layer}: bad o_proj input width")
        hd = width // nh

        if self.alignment == "role":
            # source/swapped subject B -> target subject A
            # source/swapped reference A -> target reference B
            src_for_a_np = np.asarray(
                self.source_head_state["B_tokens"], dtype=np.float32
            )
            src_for_b_np = np.asarray(
                self.source_head_state["A_tokens"], dtype=np.float32
            )
        elif self.alignment == "identity":
            src_for_a_np = np.asarray(
                self.source_head_state["A_tokens"], dtype=np.float32
            )
            src_for_b_np = np.asarray(
                self.source_head_state["B_tokens"], dtype=np.float32
            )
        else:
            raise ValueError(self.alignment)

        def patch_hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            x = inputs[0]
            if x.ndim != 3 or int(x.shape[0]) != 1:
                return None

            modified = x.clone()
            ap = torch.as_tensor(
                self.a_positions, device=x.device, dtype=torch.long
            )
            bp = torch.as_tensor(
                self.b_positions, device=x.device, dtype=torch.long
            )
            lo = int(head) * hd
            hi = lo + hd

            ta = modified[0].index_select(0, ap)[:, lo:hi]
            tb = modified[0].index_select(0, bp)[:, lo:hi]
            sa = torch.as_tensor(src_for_a_np, device=x.device, dtype=x.dtype)
            sb = torch.as_tensor(src_for_b_np, device=x.device, dtype=x.dtype)

            ra, map_a = map_source_rows_to_target(
                source_rows=sa,
                target_rows=ta,
                mode=self.replacement_mode,
            )
            rb, map_b = map_source_rows_to_target(
                source_rows=sb,
                target_rows=tb,
                mode=self.replacement_mode,
            )

            modified[0, ap, lo:hi] = ra
            modified[0, bp, lo:hi] = rb
            self.patch_count += 1
            self.patch_stats = {
                "writer": hname(self.writer_head),
                "alignment": self.alignment,
                "A_mapping": map_a,
                "B_mapping": map_b,
                "target_A_tokens": int(ta.shape[0]),
                "target_B_tokens": int(tb.shape[0]),
                "source_A_tokens": int(sa.shape[0]),
                "source_B_tokens": int(sb.shape[0]),
            }
            return (modified, *inputs[1:])

        self.handles.append(o_proj.register_forward_pre_hook(patch_hook))

        for u_layer in self.capture_u_layers:
            block = self.decoder_layers[u_layer]

            def make_capture(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if int(hidden.shape[0]) != 1:
                        return
                    ap = torch.as_tensor(
                        self.a_positions, device=hidden.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=hidden.device, dtype=torch.long
                    )
                    a = hidden[0].index_select(0, ap).mean(dim=0)
                    b = hidden[0].index_select(0, bp).mean(dim=0)
                    self.residual_states[layer_index] = {
                        "A": a.detach().float().cpu().numpy().astype(np.float32),
                        "B": b.detach().float().cpu().numpy().astype(np.float32),
                    }
                return hook

            self.handles.append(
                block.register_forward_hook(make_capture(u_layer))
            )

        return self

    def validate(self) -> None:
        if self.patch_count < 1:
            raise RuntimeError(
                f"Writer patch did not fire for {hname(self.writer_head)}"
            )
        missing = [
            l for l in self.capture_u_layers
            if l not in self.residual_states
        ]
        if missing:
            raise RuntimeError(
                f"Missing U-layer captures after writer patch: {missing}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


class CleanTargetCapture:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        u_layers: Sequence[int],
        target_a_positions: Sequence[int],
        target_b_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.u_layers = sorted(set(map(int, u_layers)))
        self.a_positions = sorted(set(map(int, target_a_positions)))
        self.b_positions = sorted(set(map(int, target_b_positions)))
        self.handles: List[Any] = []
        self.residual_states: Dict[int, Dict[str, np.ndarray]] = {}

    def __enter__(self) -> "CleanTargetCapture":
        for layer in self.u_layers:
            block = self.decoder_layers[layer]

            def make_capture(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if int(hidden.shape[0]) != 1:
                        return
                    ap = torch.as_tensor(
                        self.a_positions, device=hidden.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=hidden.device, dtype=torch.long
                    )
                    a = hidden[0].index_select(0, ap).mean(dim=0)
                    b = hidden[0].index_select(0, bp).mean(dim=0)
                    self.residual_states[layer_index] = {
                        "A": a.detach().float().cpu().numpy().astype(np.float32),
                        "B": b.detach().float().cpu().numpy().astype(np.float32),
                    }
                return hook

            self.handles.append(
                block.register_forward_hook(make_capture(layer))
            )
        return self

    def validate(self) -> None:
        missing = [l for l in self.u_layers if l not in self.residual_states]
        if missing:
            raise RuntimeError(f"Missing clean U-layer captures: {missing}")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


# =============================================================================
# Relation logits
# =============================================================================

def relation_scores(
    logits: torch.Tensor,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[str, np.ndarray]:
    scores: List[float] = []
    for rel in RELATIONS:
        ids = torch.as_tensor(
            list(relation_token_map[rel]),
            device=logits.device,
            dtype=torch.long,
        )
        score = logits.index_select(0, ids).float().max()
        scores.append(float(score.item()))
    arr = np.asarray(scores, dtype=np.float32)
    return ID_TO_REL[int(arr.argmax())], arr


# =============================================================================
# U-space geometry
# =============================================================================

def role_pair_coords(
    state: Mapping[str, np.ndarray],
    q: np.ndarray,
    *,
    source_branch: bool,
) -> np.ndarray:
    """
    Target/original role: subject=A, reference=B.
    Source/swapped role: subject=B, reference=A.
    """
    if source_branch:
        sub = np.asarray(state["B"], dtype=np.float32)
        ref = np.asarray(state["A"], dtype=np.float32)
    else:
        sub = np.asarray(state["A"], dtype=np.float32)
        ref = np.asarray(state["B"], dtype=np.float32)

    qq = np.asarray(q, dtype=np.float32)
    return np.concatenate(
        [sub @ qq, ref @ qq],
        axis=0,
    ).astype(np.float32)


def identity_pair_coords(
    state: Mapping[str, np.ndarray],
    q: np.ndarray,
) -> np.ndarray:
    a = np.asarray(state["A"], dtype=np.float32)
    b = np.asarray(state["B"], dtype=np.float32)
    qq = np.asarray(q, dtype=np.float32)
    return np.concatenate([a @ qq, b @ qq], axis=0).astype(np.float32)


def u_geometry(
    target: np.ndarray,
    source: np.ndarray,
    patched: np.ndarray,
) -> Dict[str, float]:
    t = np.asarray(target, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    p = np.asarray(patched, dtype=np.float64)

    d = s - t
    m = p - t
    dd = float(np.dot(d, d))
    eps = 1e-12

    if dd <= eps:
        return {
            "source_progress": float("nan"),
            "source_side": float("nan"),
            "fraction_distance_closed": float("nan"),
            "off_axis_ratio": float("nan"),
            "move_norm_ratio": float("nan"),
            "target_source_distance": 0.0,
        }

    progress = float(np.dot(m, d) / dd)
    d_ts = math.sqrt(dd)
    d_ps = float(np.linalg.norm(p - s))
    d_pt = float(np.linalg.norm(p - t))
    off = m - progress * d

    return {
        "source_progress": progress,
        "source_side": float(d_ps < d_pt),
        "fraction_distance_closed": float(1.0 - d_ps / (d_ts + eps)),
        "off_axis_ratio": float(np.linalg.norm(off) / (d_ts + eps)),
        "move_norm_ratio": float(np.linalg.norm(m) / (d_ts + eps)),
        "target_source_distance": d_ts,
    }


# =============================================================================
# Cache preparation
# =============================================================================

def prepare_source_and_clean_cache(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    all_heads: Sequence[Tuple[int, int]],
    u_layers: Sequence[int],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    attention_helper: Any,
    error_path: Path,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    cache: Dict[int, Dict[str, Any]] = {}
    successful_rows: List[Dict[str, Any]] = []

    print("\nCaching source head states + source/clean DAS-U states...", flush=True)

    for i, row in enumerate(
        tqdm(rows, desc="writer-cache"),
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
                device=torch.device(args.device),
            )

            src_cap = SourceEverythingCapture(
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver=receiver,
                heads=all_heads,
                u_layers=u_layers,
                source_a_positions=pair.swapped_a_positions,
                source_b_positions=pair.swapped_b_positions,
            )
            with src_cap:
                out_src = model(
                    **pair.swapped_batch,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            src_cap.validate(all_heads, u_layers)
            src_pred, src_scores = relation_scores(
                out_src.logits[0, -1],
                relation_token_map,
            )
            del out_src

            clean_cap = CleanTargetCapture(
                decoder_layers=decoder_layers,
                u_layers=u_layers,
                target_a_positions=pair.original_a_positions,
                target_b_positions=pair.original_b_positions,
            )
            with clean_cap:
                out_clean = model(
                    **pair.original_batch,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            clean_cap.validate()
            clean_pred, clean_scores = relation_scores(
                out_clean.logits[0, -1],
                relation_token_map,
            )
            del out_clean

            cache[int(pair.sid)] = {
                "head_states": src_cap.head_states,
                "source_u_states": src_cap.residual_states,
                "clean_u_states": clean_cap.residual_states,
                "clean_prediction": clean_pred,
                "clean_scores": clean_scores,
                "source_prediction": src_pred,
                "source_scores": src_scores,
            }
            successful_rows.append(dict(row))

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "cache",
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

    return cache, successful_rows


# =============================================================================
# Writer scan
# =============================================================================

def run_writer_condition(
    *,
    args: argparse.Namespace,
    condition_name: str,
    writer_head: Tuple[int, int],
    source_head_key: Tuple[int, int],
    alignment: str,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[int, Mapping[str, Any]],
    u_layers: Sequence[int],
    bases: Mapping[int, np.ndarray],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    attention_helper: Any,
    error_path: Path,
    sample_log_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    writer_head = target head being patched.
    source_head_key = which source head states to inject. Normally same as writer_head;
                      for matched-random control it is the random head itself.
    """
    valid_u_layers = [
        l for l in u_layers if int(l) >= int(writer_head[0])
    ]
    if not valid_u_layers:
        raise RuntimeError(
            f"{hname(writer_head)} has no downstream U layer among {u_layers}"
        )

    per_layer: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    final_source_hits = 0
    final_target_hits = 0
    final_changes = 0
    final_margins: List[float] = []
    n = 0

    for i, row in enumerate(
        tqdm(
            rows,
            desc=f"{condition_name}:{hname(writer_head)}",
            leave=False,
        ),
        start=1,
    ):
        sid = int(row["sid"])
        if sid not in cache:
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
                device=torch.device(args.device),
            )

            source_head_state = cache[sid]["head_states"][source_head_key]

            cap = TargetWriterPatchAndCapture(
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver=receiver,
                writer_head=writer_head,
                source_head_state=source_head_state,
                alignment=alignment,
                target_a_positions=pair.original_a_positions,
                target_b_positions=pair.original_b_positions,
                capture_u_layers=valid_u_layers,
                replacement_mode=args.replacement_mode,
            )
            with cap:
                out = model(
                    **pair.original_batch,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            cap.validate()
            pred, scores = relation_scores(
                out.logits[0, -1],
                relation_token_map,
            )
            del out

            gt = str(pair.gt)
            source_gt = OPPOSITE[gt]
            clean_pred = str(cache[sid]["clean_prediction"])

            final_source_hits += int(pred == source_gt)
            final_target_hits += int(pred == gt)
            final_changes += int(pred != clean_pred)
            final_margins.append(
                float(scores[REL_TO_ID[source_gt]] - scores[REL_TO_ID[gt]])
            )
            n += 1

            sample_row: Dict[str, Any] = {
                "sid": sid,
                "condition": condition_name,
                "writer": hname(writer_head),
                "source_head": hname(source_head_key),
                "alignment": alignment,
                "gt_target": gt,
                "gt_source": source_gt,
                "clean_prediction": clean_pred,
                "patched_prediction": pred,
            }

            for u_layer in valid_u_layers:
                q = bases[u_layer]

                target_coords = role_pair_coords(
                    cache[sid]["clean_u_states"][u_layer],
                    q,
                    source_branch=False,
                )
                source_coords = role_pair_coords(
                    cache[sid]["source_u_states"][u_layer],
                    q,
                    source_branch=True,
                )
                patched_coords = role_pair_coords(
                    cap.residual_states[u_layer],
                    q,
                    source_branch=False,
                )

                geom = u_geometry(
                    target_coords,
                    source_coords,
                    patched_coords,
                )
                per_layer[u_layer].append(geom)

                sample_row[f"U{u_layer}_source_progress"] = geom["source_progress"]
                sample_row[f"U{u_layer}_source_side"] = geom["source_side"]
                sample_row[f"U{u_layer}_fraction_closed"] = geom[
                    "fraction_distance_closed"
                ]
                sample_row[f"U{u_layer}_off_axis_ratio"] = geom["off_axis_ratio"]

            if sample_log_path is not None:
                append_jsonl(sample_log_path, sample_row)

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "writer_scan",
                "condition": condition_name,
                "writer": hname(writer_head),
                "source_head": hname(source_head_key),
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

    layer_rows: List[Dict[str, Any]] = []
    for u_layer in valid_u_layers:
        values = per_layer[u_layer]
        row = {
            "condition": condition_name,
            "writer": hname(writer_head),
            "writer_layer": int(writer_head[0]),
            "writer_head": int(writer_head[1]),
            "source_head": hname(source_head_key),
            "u_layer": int(u_layer),
            "N": len(values),
            "source_progress_mean": safe_mean(v["source_progress"] for v in values),
            "source_progress_std": safe_std(v["source_progress"] for v in values),
            "source_side_rate": safe_mean(v["source_side"] for v in values),
            "fraction_distance_closed_mean": safe_mean(
                v["fraction_distance_closed"] for v in values
            ),
            "off_axis_ratio_mean": safe_mean(v["off_axis_ratio"] for v in values),
            "move_norm_ratio_mean": safe_mean(v["move_norm_ratio"] for v in values),
            "target_source_distance_mean": safe_mean(
                v["target_source_distance"] for v in values
            ),
        }
        layer_rows.append(row)

    final = {
        "condition": condition_name,
        "writer": hname(writer_head),
        "source_head": hname(source_head_key),
        "N": n,
        "next_token_source_follow": final_source_hits / n if n else float("nan"),
        "next_token_target_accuracy": final_target_hits / n if n else float("nan"),
        "next_token_change_vs_clean": final_changes / n if n else float("nan"),
        "source_minus_target_margin_mean": safe_mean(final_margins),
    }
    return layer_rows, final


# =============================================================================
# Ranking / reporting
# =============================================================================

def build_writer_summary(
    *,
    role_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    direction_scores: Mapping[Tuple[int, int], float],
) -> List[Dict[str, Any]]:
    by_writer: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in role_rows:
        if str(row["condition"]) == "role":
            by_writer[str(row["writer"])].append(row)

    final_map = {
        str(r["writer"]): r
        for r in final_rows
        if str(r["condition"]) == "role"
    }

    score_by_name = {
        hname(h): float(s) for h, s in direction_scores.items()
    }

    out: List[Dict[str, Any]] = []
    for writer, rows in by_writer.items():
        valid = [
            r for r in rows
            if np.isfinite(float(r["source_progress_mean"]))
        ]
        if not valid:
            continue

        best = max(
            valid,
            key=lambda r: float(r["source_progress_mean"]),
        )
        mean_progress = safe_mean(
            float(r["source_progress_mean"]) for r in valid
        )
        mean_closed = safe_mean(
            float(r["fraction_distance_closed_mean"]) for r in valid
        )
        mean_side = safe_mean(
            float(r["source_side_rate"]) for r in valid
        )

        fin = final_map.get(writer, {})
        out.append({
            "writer": writer,
            "writer_layer": int(best["writer_layer"]),
            "direction_img_accuracy": score_by_name.get(writer, float("nan")),
            "best_u_layer": int(best["u_layer"]),
            "best_source_progress": float(best["source_progress_mean"]),
            "best_source_side_rate": float(best["source_side_rate"]),
            "best_fraction_distance_closed": float(
                best["fraction_distance_closed_mean"]
            ),
            "mean_source_progress_across_downstream_u": mean_progress,
            "mean_source_side_across_downstream_u": mean_side,
            "mean_fraction_closed_across_downstream_u": mean_closed,
            "next_token_source_follow": float(
                fin.get("next_token_source_follow", float("nan"))
            ),
            "next_token_target_accuracy": float(
                fin.get("next_token_target_accuracy", float("nan"))
            ),
            "next_token_change_vs_clean": float(
                fin.get("next_token_change_vs_clean", float("nan"))
            ),
            "source_minus_target_margin_mean": float(
                fin.get("source_minus_target_margin_mean", float("nan"))
            ),
        })

    out.sort(
        key=lambda r: (
            float(r["best_source_progress"]),
            float(r["mean_source_progress_across_downstream_u"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(out, start=1):
        row["writer_rank_by_U"] = rank
    return out


def print_writer_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 150)
    print("WRITER -> DAS-U RANKING")
    print("=" * 150)
    print(
        f"{'rank':>4s} {'writer':<9s} {'dirACC':>8s} {'bestU':>6s} "
        f"{'bestProg':>10s} {'srcSide':>9s} {'closed':>9s} "
        f"{'meanProg':>10s} {'nextSRC':>9s} {'change':>9s}"
    )
    print("-" * 150)
    for row in rows:
        print(
            f"{int(row['writer_rank_by_U']):>4d} "
            f"{str(row['writer']):<9s} "
            f"{100*float(row['direction_img_accuracy']):>7.2f}% "
            f"L{int(row['best_u_layer']):<5d} "
            f"{float(row['best_source_progress']):>+9.3f} "
            f"{100*float(row['best_source_side_rate']):>8.2f}% "
            f"{100*float(row['best_fraction_distance_closed']):>8.2f}% "
            f"{float(row['mean_source_progress_across_downstream_u']):>+9.3f} "
            f"{100*float(row['next_token_source_follow']):>8.2f}% "
            f"{100*float(row['next_token_change_vs_clean']):>8.2f}%"
        )


def build_control_comparison(
    *,
    role_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    random_map: Mapping[str, str],
) -> List[Dict[str, Any]]:
    role = {
        (str(r["writer"]), int(r["u_layer"])): r
        for r in role_rows
        if str(r["condition"]) == "role"
    }
    identity = {
        (str(r["writer"]), int(r["u_layer"])): r
        for r in identity_rows
    }
    random_c = {
        (str(r["writer"]), int(r["u_layer"])): r
        for r in random_rows
    }

    rows: List[Dict[str, Any]] = []
    for key, rr in role.items():
        if key not in identity or key not in random_c:
            continue
        ii = identity[key]
        rnd = random_c[key]
        writer, u_layer = key
        rows.append({
            "writer": writer,
            "matched_random": random_map.get(writer, ""),
            "u_layer": u_layer,
            "role_progress": float(rr["source_progress_mean"]),
            "identity_progress": float(ii["source_progress_mean"]),
            "random_progress": float(rnd["source_progress_mean"]),
            "role_minus_identity": (
                float(rr["source_progress_mean"])
                - float(ii["source_progress_mean"])
            ),
            "role_minus_random": (
                float(rr["source_progress_mean"])
                - float(rnd["source_progress_mean"])
            ),
            "role_source_side": float(rr["source_side_rate"]),
            "identity_source_side": float(ii["source_side_rate"]),
            "random_source_side": float(rnd["source_side_rate"]),
            "role_fraction_closed": float(
                rr["fraction_distance_closed_mean"]
            ),
            "identity_fraction_closed": float(
                ii["fraction_distance_closed_mean"]
            ),
            "random_fraction_closed": float(
                rnd["fraction_distance_closed_mean"]
            ),
        })

    rows.sort(
        key=lambda r: float(r["role_minus_random"]),
        reverse=True,
    )
    return rows


def print_control_table(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    print("\n" + "=" * 150)
    print("TOP WRITER CONTROL CONFIRMATION")
    print("=" * 150)
    print(
        f"{'writer':<9s} {'random':<9s} {'U':>4s} "
        f"{'ROLE':>9s} {'identity':>10s} {'random':>9s} "
        f"{'role-id':>10s} {'role-rnd':>10s}"
    )
    print("-" * 150)
    for row in rows:
        print(
            f"{str(row['writer']):<9s} "
            f"{str(row['matched_random']):<9s} "
            f"L{int(row['u_layer']):<3d} "
            f"{float(row['role_progress']):>+8.3f} "
            f"{float(row['identity_progress']):>+9.3f} "
            f"{float(row['random_progress']):>+8.3f} "
            f"{float(row['role_minus_identity']):>+9.3f} "
            f"{float(row['role_minus_random']):>+9.3f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.top_k < 1:
        raise ValueError("--top-k must be >=1")
    if args.confirm_top_n < 0:
        raise ValueError("--confirm-top-n must be >=0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / "errors.jsonl"

    ioi = import_file(Path(args.ioi_script), "_w2u_ioi")
    producer = import_file(Path(args.producer_script), "_w2u_producer")
    receiver = import_file(Path(args.receiver_script), "_w2u_receiver")
    v3 = import_file(Path(args.v3_script), "_w2u_v3")
    base = import_file(Path(args.base_script), "_w2u_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "_w2u_attention",
    )

    source_dir = Path(args.source_output_dir)
    extraction_path = source_dir / "extraction.jsonl"
    source_config_path = source_dir / "config.json"
    if not extraction_path.exists() or not source_config_path.exists():
        raise FileNotFoundError(
            f"Need extraction.jsonl + config.json in {source_dir}"
        )

    rows = [
        r for r in read_jsonl(extraction_path)
        if str(r.get("gt")) in RELATIONS
    ]

    das_dir = Path(args.das_dir)
    das_config_path = das_dir / "config.json"
    if not das_config_path.exists():
        available = sorted(Path("output").glob("**/basis_L*_D*.npz"))
        msg = [f"DAS config not found: {das_config_path}"]
        if available:
            msg.append("Available DAS basis examples:")
            msg.extend(f"  {p}" for p in available[:30])
        raise FileNotFoundError("\n".join(msg))

    das_config = json.loads(das_config_path.read_text(encoding="utf-8"))

    if args.sample_scope == "das_eval":
        eval_sids = set(map(int, das_config.get("eval_sids", [])))
        if not eval_sids:
            raise RuntimeError(
                f"{das_config_path} has no eval_sids; use --sample-scope all if intentional"
            )
        rows = [r for r in rows if int(r["sid"]) in eval_sids]

    if args.pair_status != "all":
        rows = [
            r for r in rows
            if str(r.get("generation_pair_status", "")) == args.pair_status
        ]

    rows = stratified_subset(
        rows,
        args.max_samples,
        args.sample_seed,
    )
    if not rows:
        raise RuntimeError("No samples after scope/status filtering")

    writer_layers = parse_int_spec(args.writer_layers)
    u_layers = parse_int_spec(args.u_layers)

    candidate_heads, direction_scores = load_writer_candidates(
        path=Path(args.direction_results),
        metric=args.rank_metric,
        allowed_layers=writer_layers,
        top_k=args.top_k,
    )

    bases = load_das_bases(
        das_dir=das_dir,
        u_layers=u_layers,
        u_dim=args.u_dim,
    )

    model = processor = None
    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )
        model.eval()

        saved_max = getattr(args, "max_samples", None)
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(
                args, base
            )
        finally:
            args.max_samples = saved_max

        validate_heads(
            candidate_heads,
            decoder_layers,
            attention_helper,
            receiver,
        )

        random_heads = matched_random_heads(
            target_heads=candidate_heads,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver=receiver,
            seed=args.seed + 991,
        )
        validate_heads(
            random_heads,
            decoder_layers,
            attention_helper,
            receiver,
        )

        random_map = {
            hname(candidate_heads[i]): hname(random_heads[i])
            for i in range(len(candidate_heads))
        }

        all_source_heads = list(
            dict.fromkeys(candidate_heads + random_heads)
        )

        print("\n" + "=" * 130)
        print("WRITER -> DAS-U CAUSAL SCAN")
        print("=" * 130)
        print("model          :", args.model)
        print("N samples      :", len(rows))
        print("sample scope   :", args.sample_scope)
        print("pair status    :", args.pair_status)
        print("writer layers  :", writer_layers)
        print("Direction TopK :", args.top_k)
        print("U layers       :", u_layers)
        print("U dim          :", args.u_dim)
        print("confirm top N  :", args.confirm_top_n)
        print("\nCandidate writers:")
        for rank, h in enumerate(candidate_heads, start=1):
            print(
                f"  {rank:02d}. {hname(h)} "
                f"dir_acc={100*direction_scores[h]:.2f}%"
            )
        print("=" * 130, flush=True)

        config = {
            "version": VERSION,
            "model": args.model,
            "repo_id": getattr(spec, "repo_id", ""),
            "decoder_path": decoder_path,
            "N_samples": len(rows),
            "sample_scope": args.sample_scope,
            "pair_status": args.pair_status,
            "sample_sids": [int(r["sid"]) for r in rows],
            "direction_results": str(args.direction_results),
            "rank_metric": args.rank_metric,
            "writer_layers": writer_layers,
            "candidate_heads": [
                {
                    "rank": i + 1,
                    "head": hname(h),
                    "direction_score": float(direction_scores[h]),
                    "matched_random": hname(random_heads[i]),
                }
                for i, h in enumerate(candidate_heads)
            ],
            "das_dir": str(das_dir),
            "u_layers": u_layers,
            "u_dim": args.u_dim,
            "confirm_top_n": args.confirm_top_n,
            "replacement_mode": args.replacement_mode,
            "audit": audit,
        }
        write_json(out_dir / "config.json", config)

        cache, rows_ok = prepare_source_and_clean_cache(
            args=args,
            rows=rows,
            all_heads=all_source_heads,
            u_layers=u_layers,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            relation_token_map=relation_token_map,
            records_by_sid=records_by_sid,
            prompt_rows=prompt_rows,
            base=base,
            v3=v3,
            receiver=receiver,
            attention_helper=attention_helper,
            error_path=error_path,
        )
        rows = rows_ok
        if not rows:
            raise RuntimeError("No successful rows after cache phase")

        # ---------------------------------------------------------------------
        # Phase 1: Top20 role scan
        # ---------------------------------------------------------------------
        role_layer_rows: List[Dict[str, Any]] = []
        role_final_rows: List[Dict[str, Any]] = []

        sample_log = out_dir / "writer_role_samples.jsonl"

        for idx, head in enumerate(candidate_heads, start=1):
            # A writer later than every U layer cannot be evaluated.
            if min(
                [u for u in u_layers if u >= head[0]],
                default=None,
            ) is None:
                print(
                    f"[skip] {hname(head)} has no downstream requested U layer",
                    flush=True,
                )
                continue

            print(
                f"\n[ROLE {idx:02d}/{len(candidate_heads)}] {hname(head)}",
                flush=True,
            )
            layer_rows, final_row = run_writer_condition(
                args=args,
                condition_name="role",
                writer_head=head,
                source_head_key=head,
                alignment="role",
                rows=rows,
                cache=cache,
                u_layers=u_layers,
                bases=bases,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                relation_token_map=relation_token_map,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                receiver=receiver,
                attention_helper=attention_helper,
                error_path=error_path,
                sample_log_path=sample_log,
            )
            role_layer_rows.extend(layer_rows)
            role_final_rows.append(final_row)

            write_csv(
                out_dir / "writer_u_effects_role.csv",
                role_layer_rows,
            )

        writer_summary = build_writer_summary(
            role_rows=role_layer_rows,
            final_rows=role_final_rows,
            direction_scores=direction_scores,
        )
        write_csv(
            out_dir / "writer_summary.csv",
            writer_summary,
        )
        print_writer_table(writer_summary)

        # ---------------------------------------------------------------------
        # Phase 2: controls for strongest N
        # ---------------------------------------------------------------------
        identity_rows: List[Dict[str, Any]] = []
        random_rows: List[Dict[str, Any]] = []
        identity_final: List[Dict[str, Any]] = []
        random_final: List[Dict[str, Any]] = []

        confirm_n = min(
            int(args.confirm_top_n),
            len(writer_summary),
        )

        if confirm_n > 0:
            name_to_head = {hname(h): h for h in candidate_heads}
            head_to_random = {
                candidate_heads[i]: random_heads[i]
                for i in range(len(candidate_heads))
            }

            for rank_row in writer_summary[:confirm_n]:
                writer = name_to_head[str(rank_row["writer"])]
                rnd = head_to_random[writer]

                print(
                    f"\n[CONTROL] writer={hname(writer)} random={hname(rnd)}",
                    flush=True,
                )

                lr, fr = run_writer_condition(
                    args=args,
                    condition_name="identity",
                    writer_head=writer,
                    source_head_key=writer,
                    alignment="identity",
                    rows=rows,
                    cache=cache,
                    u_layers=u_layers,
                    bases=bases,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    receiver=receiver,
                    attention_helper=attention_helper,
                    error_path=error_path,
                    sample_log_path=None,
                )
                identity_rows.extend(lr)
                identity_final.append(fr)

                # Random control patches the RANDOM target head with its own
                # swapped/source role state.
                lr, fr = run_writer_condition(
                    args=args,
                    condition_name="random_role",
                    writer_head=rnd,
                    source_head_key=rnd,
                    alignment="role",
                    rows=rows,
                    cache=cache,
                    u_layers=u_layers,
                    bases=bases,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    receiver=receiver,
                    attention_helper=attention_helper,
                    error_path=error_path,
                    sample_log_path=None,
                )

                # Rename random control rows back to the candidate writer so
                # comparisons align, while preserving random source_head.
                for r in lr:
                    r["random_target_head"] = r["writer"]
                    r["writer"] = hname(writer)
                fr["random_target_head"] = fr["writer"]
                fr["writer"] = hname(writer)

                random_rows.extend(lr)
                random_final.append(fr)

            write_csv(
                out_dir / "writer_u_effects_identity.csv",
                identity_rows,
            )
            write_csv(
                out_dir / "writer_u_effects_random.csv",
                random_rows,
            )

            control_rows = build_control_comparison(
                role_rows=role_layer_rows,
                identity_rows=identity_rows,
                random_rows=random_rows,
                random_map=random_map,
            )
            write_csv(
                out_dir / "writer_control_comparison.csv",
                control_rows,
            )
            print_control_table(control_rows)
        else:
            control_rows = []

        # ---------------------------------------------------------------------
        # Relationship between Direction decoding and Writer->U causality
        # ---------------------------------------------------------------------
        if len(writer_summary) >= 3:
            x = np.asarray(
                [float(r["direction_img_accuracy"]) for r in writer_summary],
                dtype=np.float64,
            )
            y = np.asarray(
                [float(r["best_source_progress"]) for r in writer_summary],
                dtype=np.float64,
            )
            pearson = float(np.corrcoef(x, y)[0, 1])

            xr = np.argsort(np.argsort(x))
            yr = np.argsort(np.argsort(y))
            spearman = float(np.corrcoef(xr, yr)[0, 1])
        else:
            pearson = spearman = float("nan")

        summary_json = {
            "version": VERSION,
            "N": len(rows),
            "pearson_direction_acc_vs_best_U_progress": pearson,
            "spearman_direction_acc_vs_best_U_progress": spearman,
            "top_writers": writer_summary[:10],
            "control_rows": control_rows[:30],
        }
        write_json(
            out_dir / "summary.json",
            summary_json,
        )

        print("\n" + "=" * 110)
        print("DIRECTION DECODING vs WRITER->U CAUSALITY")
        print("=" * 110)
        print(
            f"Pearson  direction_acc vs best_U_progress: {pearson:+.4f}"
        )
        print(
            f"Spearman direction_acc vs best_U_progress: {spearman:+.4f}"
        )

        report = [
            f"version: {VERSION}",
            f"model: {args.model}",
            f"N successful: {len(rows)}",
            f"writer candidates: Top{args.top_k} Direction in layers {writer_layers}",
            f"DAS-U layers: {u_layers}, dim={args.u_dim}",
            "",
            "PRIMARY INTERPRETATION",
            "A candidate head is a plausible relation-subspace WRITER if its",
            "role-aligned natural state replacement causes downstream DAS-U",
            "coordinates to move toward the source/opposite state:",
            "",
            "  source_progress > 0",
            "  fraction_distance_closed > 0",
            "  source_side_rate elevated",
            "",
            "Stronger evidence requires the best candidates to satisfy:",
            "  role progress >> identity progress",
            "  role progress >> same-layer random-head progress",
            "",
            "Do not interpret a high Direction probe ACC alone as writer evidence.",
            "writer_summary.csv ranks by actual downstream U movement.",
            "",
            "FILES",
            "writer_summary.csv",
            "writer_u_effects_role.csv",
            "writer_u_effects_identity.csv",
            "writer_u_effects_random.csv",
            "writer_control_comparison.csv",
            "writer_role_samples.jsonl",
            "summary.json",
            "config.json",
        ]
        (out_dir / "report.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "writer_summary.csv",
            "writer_u_effects_role.csv",
            "writer_u_effects_identity.csv",
            "writer_u_effects_random.csv",
            "writer_control_comparison.csv",
            "writer_role_samples.jsonl",
            "summary.json",
            "config.json",
            "report.txt",
        ):
            path = out_dir / name
            if path.exists():
                print(" ", path)

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
