#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAS-style causal spatial subspace search for COCO 4-way relations.

Goal
====
Previous head-level interventions suggest that spatial relation information may
be distributed across multiple heads / residual dimensions rather than living
in one "relation head".

This script therefore changes the unit of analysis:

    head  ->  low-dimensional subspace of a decoder residual representation

For one decoder layer l, let target/original object hidden state be h and the
same-image swapped/source object hidden state be s.  We learn an orthonormal
basis U in R^(d_model x k) and perform an interchange intervention only inside
that subspace:

    h' = h + U U^T (s - h)

Everything orthogonal to U is left exactly as the target/original run.

Target/source construction
==========================
Same image, same two objects:

    target/original: A relative to B -> r
    source/swapped : B relative to A -> opposite(r)

The MAIN ROLE intervention uses:

    source subject B   -> target subject A
    source reference A -> target reference B

Identity control uses:

    source A -> target A
    source B -> target B

Training
========
Model weights are frozen. Only U is optimized.

The basis is trained on a small subset of BOTH-CORRECT pairs to maximize the
source/opposite relation at the model's final next-token relation logits.

The basis is orthonormalized by differentiable QR on every intervention:

    Q = qr(raw_basis).Q

and Q is the actual learned subspace.

Evaluation
==========
Evaluation is fully held-out from basis training.

For every (layer, subspace_dim), report:

    clean
    learned_role
    learned_identity
    random_role
    full_role        # whole residual state replacement upper reference
    full_identity

Metrics:
    target_acc
    source_follow_iia      # main causal abstraction / interchange metric
    other_rate
    prediction_change
    source_minus_target_margin

Results are reported separately for:
    heldout_all
    heldout_both_correct

Important interpretation
========================
A compelling distributed causal relation representation looks like:

    learned_role IIA >> random_role IIA
    learned_role IIA >> learned_identity IIA
    heldout IIA stays high (not just train)
    modest k (e.g. 4/8/16) is enough
    one or more layers show a clear peak

If full_role is high but learned_role is low:
    optimization or chosen k is insufficient.

If even full_role is low:
    object residual state at that layer is probably not the causal relation
    variable we are looking for.

Smoke test
==========

CUDA_VISIBLE_DEVICES=0 python -u validate_coco_relation_das_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --layers 19,23,26 \
  --subspace-dims 8 \
  --train-size 48 \
  --epochs 3 \
  --eval-max-samples 160 \
  --train-pair-status both_correct \
  --device cuda:0 \
  --output-dir output/qwen3b_relation_das_smoke \
  --overwrite

Stronger run
============

CUDA_VISIBLE_DEVICES=0 python -u validate_coco_relation_das_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --layers 18,19,20,21,22,23,24,25,26,27 \
  --subspace-dims 4,8,16 \
  --train-size 64 \
  --epochs 6 \
  --eval-max-samples 0 \
  --train-pair-status both_correct \
  --device cuda:0 \
  --output-dir output/qwen3b_relation_das_full \
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
import torch.nn.functional as F
from tqdm import tqdm


VERSION = "coco-relation-das-v1"

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
        help="Existing extraction.jsonl + config.json produced by the earlier pipeline.",
    )

    p.add_argument(
        "--layers",
        default="19,23,26",
        help="Decoder layers at whose OUTPUT residual stream DAS is learned.",
    )
    p.add_argument(
        "--subspace-dims",
        default="8",
        help="Comma-separated learned subspace dimensions, e.g. 4,8,16.",
    )

    p.add_argument(
        "--train-pair-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
        help="Training sources should preferably be both_correct so source relation is clean.",
    )
    p.add_argument(
        "--train-size",
        type=int,
        default=48,
        help="Number of training pairs. 0 uses --train-ratio instead.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.15,
        help="Used only when --train-size 0.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=160,
        help="0 = all held-out samples.",
    )
    p.add_argument("--split-seed", type=int, default=13)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument(
        "--logit-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to 4 relation class logits during DAS training.",
    )

    p.add_argument(
        "--random-repeats",
        type=int,
        default=1,
        help="Number of random orthonormal subspaces used in held-out random_role control.",
    )
    p.add_argument(
        "--eval-full-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate full role/identity residual replacement as an upper reference.",
    )

    p.add_argument(
        "--source-cache-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="CPU dtype for cached source object residual states.",
    )
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=8)
    p.add_argument("--empty-cache-every", type=int, default=8)

    # Same repository helper scripts used by the previous experiments.
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
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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


def parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v not in seen:
            vals.append(v)
            seen.add(v)
    if not vals:
        raise ValueError("Empty integer list")
    return vals


def safe_mean(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def first_3d_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output) and output.ndim == 3:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not find 3D hidden tensor in decoder output")


def replace_first_3d_tensor(output: Any, new_hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output) and output.ndim == 3:
        return new_hidden
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = new_hidden
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = new_hidden
                return items
    raise RuntimeError("Could not replace 3D hidden tensor in decoder output")


def relation_class_logits(
    final_vocab_logits: torch.Tensor,
    relation_token_map: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    """
    Differentiable 4-class score. Mirrors the previous experiment's readout:
    for each relation, take max over its allowed token ids.
    """
    scores: List[torch.Tensor] = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            list(relation_token_map[relation]),
            device=final_vocab_logits.device,
            dtype=torch.long,
        )
        scores.append(
            final_vocab_logits.index_select(0, ids).float().max()
        )
    return torch.stack(scores, dim=0)


def relation_prediction(
    final_vocab_logits: torch.Tensor,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[str, np.ndarray]:
    with torch.no_grad():
        scores = relation_class_logits(
            final_vocab_logits,
            relation_token_map,
        )
        pred_id = int(scores.argmax().item())
        return ID_TO_REL[pred_id], scores.detach().cpu().numpy().astype(np.float32)


def stratified_take(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in rows]
    if n <= 0 or len(rows) <= n:
        return sorted(rows, key=lambda x: int(x["sid"]))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: List[Dict[str, Any]] = []
    idx = {r: 0 for r in RELATIONS}
    while len(selected) < n:
        moved = False
        for relation in RELATIONS:
            group = groups.get(relation, [])
            i = idx[relation]
            if i < len(group) and len(selected) < n:
                selected.append(group[i])
                idx[relation] += 1
                moved = True
        if not moved:
            break
    return sorted(selected, key=lambda x: int(x["sid"]))


def filter_pair_status(
    rows: Sequence[Mapping[str, Any]],
    status: str,
) -> List[Dict[str, Any]]:
    if status == "all":
        return [dict(r) for r in rows]
    return [
        dict(r)
        for r in rows
        if str(r.get("generation_pair_status", "")) == status
    ]


# =============================================================================
# Source residual caching
# =============================================================================

class MultiLayerObjectResidualCapture:
    """
    Capture source/swapped decoder-layer OUTPUT residual object states.

    Cache identity-aligned A/B pooled vectors.  Since swapped prompt asks
    B relative A:
        source role subject   = B
        source role reference = A
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        layers: Sequence[int],
        a_positions: Sequence[int],
        b_positions: Sequence[int],
        storage_dtype: np.dtype,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layers = sorted(set(map(int, layers)))
        self.a_positions = sorted(set(map(int, a_positions)))
        self.b_positions = sorted(set(map(int, b_positions)))
        self.storage_dtype = storage_dtype
        self.handles: List[Any] = []
        self.values: Dict[int, Dict[str, np.ndarray]] = {}

    def __enter__(self) -> "MultiLayerObjectResidualCapture":
        for layer in self.layers:
            block = self.decoder_layers[layer]

            def make_hook(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if int(hidden.shape[0]) != 1:
                        return

                    ap = torch.as_tensor(
                        self.a_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    bp = torch.as_tensor(
                        self.b_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    a = hidden[0].index_select(0, ap).mean(dim=0)
                    b = hidden[0].index_select(0, bp).mean(dim=0)

                    self.values[layer_index] = {
                        "A": a.detach().float().cpu().numpy().astype(self.storage_dtype),
                        "B": b.detach().float().cpu().numpy().astype(self.storage_dtype),
                    }
                return hook

            self.handles.append(
                block.register_forward_hook(make_hook(layer))
            )
        return self

    def validate(self) -> None:
        missing = [l for l in self.layers if l not in self.values]
        if missing:
            raise RuntimeError(f"Missing source residual captures: {missing}")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()


@torch.inference_mode()
def capture_source_layers(
    *,
    model: Any,
    swapped_batch: Mapping[str, Any],
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    swapped_a_positions: Sequence[int],
    swapped_b_positions: Sequence[int],
    storage_dtype: np.dtype,
) -> Dict[int, Dict[str, np.ndarray]]:
    cap = MultiLayerObjectResidualCapture(
        decoder_layers=decoder_layers,
        layers=layers,
        a_positions=swapped_a_positions,
        b_positions=swapped_b_positions,
        storage_dtype=storage_dtype,
    )
    with cap:
        out = model(
            **swapped_batch,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    cap.validate()
    del out
    return cap.values


# =============================================================================
# DAS intervention
# =============================================================================

def orthonormal_basis(raw_basis: torch.Tensor) -> torch.Tensor:
    # Reduced QR: [D,K] -> [D,K], differentiable.
    q, _ = torch.linalg.qr(raw_basis.float(), mode="reduced")
    return q


def random_orthonormal_basis(
    d_model: int,
    k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    raw = torch.randn(d_model, k, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.to(device=device)


class ResidualSubspaceIntervention:
    """
    Modify output residual hidden state at one decoder layer.

    learned/random mode:
        h' = h + P(s-h), P = QQ^T

    full mode:
        h' = s

    Target A = subject, target B = reference.

    alignment="role":
        target A <- source B
        target B <- source A

    alignment="identity":
        target A <- source A
        target B <- source B
    """

    def __init__(
        self,
        *,
        decoder_layer: Any,
        target_a_positions: Sequence[int],
        target_b_positions: Sequence[int],
        source_state: Mapping[str, np.ndarray],
        alignment: str,
        basis: Optional[torch.Tensor],
        full_replace: bool,
    ) -> None:
        self.decoder_layer = decoder_layer
        self.a_positions = sorted(set(map(int, target_a_positions)))
        self.b_positions = sorted(set(map(int, target_b_positions)))
        self.source_state = source_state
        self.alignment = str(alignment)
        self.basis = basis
        self.full_replace = bool(full_replace)
        self.handle: Any = None
        self.applications = 0

    def __enter__(self) -> "ResidualSubspaceIntervention":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_3d_tensor(output)
            if int(hidden.shape[0]) != 1:
                return None

            if self.alignment == "role":
                src_a_np = self.source_state["B"]  # source subject
                src_b_np = self.source_state["A"]  # source reference
            elif self.alignment == "identity":
                src_a_np = self.source_state["A"]
                src_b_np = self.source_state["B"]
            else:
                raise ValueError(self.alignment)

            ap = torch.as_tensor(
                self.a_positions,
                device=hidden.device,
                dtype=torch.long,
            )
            bp = torch.as_tensor(
                self.b_positions,
                device=hidden.device,
                dtype=torch.long,
            )

            src_a = torch.as_tensor(
                np.asarray(src_a_np, dtype=np.float32),
                device=hidden.device,
                dtype=torch.float32,
            )
            src_b = torch.as_tensor(
                np.asarray(src_b_np, dtype=np.float32),
                device=hidden.device,
                dtype=torch.float32,
            )

            # Keep intervention arithmetic in fp32, cast back to model dtype.
            y = hidden.float().clone()

            if self.full_replace:
                y[0, ap, :] = src_a[None, :]
                y[0, bp, :] = src_b[None, :]
            else:
                if self.basis is None:
                    raise RuntimeError("Subspace intervention requires basis")

                q = orthonormal_basis(self.basis)
                if int(q.shape[0]) != int(y.shape[-1]):
                    raise RuntimeError(
                        f"Basis width {q.shape[0]} != hidden width {y.shape[-1]}"
                    )

                # Each target token keeps its orthogonal complement.
                # Only its projection onto learned Q is replaced by the
                # corresponding source role/identity pooled state projection.
                ta = y[0].index_select(0, ap)
                tb = y[0].index_select(0, bp)

                delta_a = (src_a[None, :] - ta) @ q
                delta_b = (src_b[None, :] - tb) @ q

                ta_new = ta + delta_a @ q.T
                tb_new = tb + delta_b @ q.T

                y[0, ap, :] = ta_new
                y[0, bp, :] = tb_new

            self.applications += 1
            return replace_first_3d_tensor(output, y.to(dtype=hidden.dtype))

        self.handle = self.decoder_layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Forward passes
# =============================================================================

@torch.inference_mode()
def forward_clean(
    *,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    out = model(
        **batch,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    pred, scores = relation_prediction(
        out.logits[0, -1],
        relation_token_map,
    )
    del out
    return {
        "prediction": pred,
        "scores": scores,
    }


def forward_intervened_train(
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
) -> Tuple[torch.Tensor, str, np.ndarray]:
    """
    Gradient flows only to basis_parameter because model parameters are frozen.
    """
    with ResidualSubspaceIntervention(
        decoder_layer=decoder_layer,
        target_a_positions=target_a_positions,
        target_b_positions=target_b_positions,
        source_state=source_state,
        alignment="role",
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

    class_scores = relation_class_logits(
        out.logits[0, -1],
        relation_token_map,
    )
    target = torch.tensor(
        [int(source_label_id)],
        device=class_scores.device,
        dtype=torch.long,
    )
    loss = F.cross_entropy(
        (class_scores / float(temperature))[None, :],
        target,
    )

    pred_id = int(class_scores.detach().argmax().item())
    pred = ID_TO_REL[pred_id]
    scores = class_scores.detach().cpu().numpy().astype(np.float32)
    del out
    return loss, pred, scores


@torch.inference_mode()
def forward_intervened_eval(
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
) -> Dict[str, Any]:
    if condition == "clean":
        return forward_clean(
            model=model,
            batch=batch,
            relation_token_map=relation_token_map,
        )

    if condition == "learned_role":
        alignment = "role"
        basis = learned_basis
        full = False
    elif condition == "learned_identity":
        alignment = "identity"
        basis = learned_basis
        full = False
    elif condition == "random_role":
        alignment = "role"
        basis = random_basis
        full = False
    elif condition == "full_role":
        alignment = "role"
        basis = None
        full = True
    elif condition == "full_identity":
        alignment = "identity"
        basis = None
        full = True
    else:
        raise ValueError(condition)

    with ResidualSubspaceIntervention(
        decoder_layer=decoder_layer,
        target_a_positions=target_a_positions,
        target_b_positions=target_b_positions,
        source_state=source_state,
        alignment=alignment,
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

    pred, scores = relation_prediction(
        out.logits[0, -1],
        relation_token_map,
    )
    del out
    return {
        "prediction": pred,
        "scores": scores,
    }


# =============================================================================
# Data split / caches
# =============================================================================

def build_train_eval_split(
    *,
    rows: Sequence[Mapping[str, Any]],
    train_status: str,
    train_size: int,
    train_ratio: float,
    eval_max_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eligible = filter_pair_status(rows, train_status)
    if not eligible:
        raise RuntimeError(f"No rows with train status={train_status}")

    if train_size > 0:
        n_train = min(train_size, len(eligible))
    else:
        n_train = max(4, int(round(len(eligible) * float(train_ratio))))

    train_rows = stratified_take(
        eligible,
        n_train,
        seed,
    )
    train_sids = {int(r["sid"]) for r in train_rows}

    eval_pool = [
        dict(r)
        for r in rows
        if int(r["sid"]) not in train_sids
    ]
    eval_rows = stratified_take(
        eval_pool,
        eval_max_samples,
        seed + 701,
    )

    return train_rows, eval_rows


def build_source_cache(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    receiver: Any,
    storage_dtype: np.dtype,
    error_path: Path,
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    cache: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    print("\nCaching swapped/source object residual states...", flush=True)
    for i, row in enumerate(
        tqdm(rows, desc="source-cache"),
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

            values = capture_source_layers(
                model=model,
                swapped_batch=pair.swapped_batch,
                decoder_layers=decoder_layers,
                layers=layers,
                swapped_a_positions=pair.swapped_a_positions,
                swapped_b_positions=pair.swapped_b_positions,
                storage_dtype=storage_dtype,
            )
            cache[int(pair.sid)] = values

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "source_cache",
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
# DAS training
# =============================================================================

@dataclass
class TrainResult:
    q_basis_cpu: np.ndarray
    history: List[Dict[str, Any]]
    train_source_follow: float
    train_loss: float


def train_one_das(
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
    d_model: int,
    error_path: Path,
) -> TrainResult:
    if dim <= 0 or dim > d_model:
        raise ValueError(f"Bad subspace dim={dim} for d_model={d_model}")

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

    history: List[Dict[str, Any]] = []
    rng = random.Random(args.seed + 9001 * layer + 17 * dim)

    model.eval()

    for epoch in range(1, int(args.epochs) + 1):
        order = [dict(r) for r in train_rows if int(r["sid"]) in source_cache]
        rng.shuffle(order)
        if not order:
            raise RuntimeError("No train rows survived source cache")

        optimizer.zero_grad(set_to_none=True)
        running_loss: List[float] = []
        source_hits = 0
        target_hits = 0
        processed = 0

        for step_idx, row in enumerate(
            tqdm(
                order,
                desc=f"train:L{layer}:D{dim}:E{epoch}",
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
                source_gt = OPPOSITE[gt]

                loss, pred, _scores = forward_intervened_train(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[int(pair.sid)][layer],
                    basis_parameter=raw_basis,
                    source_label_id=REL_TO_ID[source_gt],
                    relation_token_map=relation_token_map,
                    temperature=args.logit_temperature,
                )

                scaled = loss / float(max(1, args.grad_accum))
                scaled.backward()

                running_loss.append(float(loss.detach().item()))
                source_hits += int(pred == source_gt)
                target_hits += int(pred == gt)
                processed += 1

                if (
                    step_idx % max(1, args.grad_accum) == 0
                    or step_idx == len(order)
                ):
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [raw_basis],
                            max_norm=float(args.max_grad_norm),
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            except Exception as exc:
                append_jsonl(error_path, {
                    "phase": "train",
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

        epoch_row = {
            "layer": layer,
            "dim": dim,
            "epoch": epoch,
            "N": processed,
            "loss": safe_mean(running_loss),
            "train_source_follow": (
                float(source_hits / processed) if processed else float("nan")
            ),
            "train_target_accuracy": (
                float(target_hits / processed) if processed else float("nan")
            ),
        }
        history.append(epoch_row)
        print(
            f"[DAS train] L{layer} D{dim} epoch={epoch} "
            f"N={processed} loss={epoch_row['loss']:.4f} "
            f"source-follow={100*epoch_row['train_source_follow']:.2f}% "
            f"target={100*epoch_row['train_target_accuracy']:.2f}%",
            flush=True,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with torch.no_grad():
        q = orthonormal_basis(raw_basis).detach().cpu().numpy().astype(np.float32)

    last = history[-1]
    return TrainResult(
        q_basis_cpu=q,
        history=history,
        train_source_follow=float(last["train_source_follow"]),
        train_loss=float(last["loss"]),
    )


# =============================================================================
# Held-out evaluation
# =============================================================================

def summarize_condition(
    *,
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    scores: Sequence[np.ndarray],
    clean_predictions: Sequence[str],
) -> Dict[str, Any]:
    n = len(rows)
    if not (len(predictions) == len(scores) == len(clean_predictions) == n):
        raise RuntimeError("Summary length mismatch")

    target_hits = 0
    source_hits = 0
    other_hits = 0
    changes = 0
    margins: List[float] = []
    clean_correct_to_source = 0
    clean_correct_n = 0

    for row, pred, score, clean_pred in zip(
        rows,
        predictions,
        scores,
        clean_predictions,
    ):
        gt = str(row["gt"])
        source_gt = OPPOSITE[gt]

        target_hits += int(pred == gt)
        source_hits += int(pred == source_gt)
        other_hits += int(pred not in (gt, source_gt))
        changes += int(pred != clean_pred)

        s = np.asarray(score, dtype=np.float64)
        margins.append(
            float(s[REL_TO_ID[source_gt]] - s[REL_TO_ID[gt]])
        )

        if clean_pred == gt:
            clean_correct_n += 1
            clean_correct_to_source += int(pred == source_gt)

    return {
        "condition": condition,
        "N": n,
        "target_accuracy": target_hits / n if n else float("nan"),
        "source_follow_iia": source_hits / n if n else float("nan"),
        "other_rate": other_hits / n if n else float("nan"),
        "prediction_change_vs_clean": changes / n if n else float("nan"),
        "source_minus_target_margin_mean": safe_mean(margins),
        "source_minus_target_margin_std": safe_std(margins),
        "clean_correct_to_source_N": clean_correct_to_source,
        "clean_correct_N": clean_correct_n,
        "clean_correct_to_source_rate": (
            clean_correct_to_source / clean_correct_n
            if clean_correct_n else float("nan")
        ),
    }


def evaluate_one_setting(
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
    error_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    device = torch.device(args.device)
    learned_basis = torch.as_tensor(
        q_basis_cpu,
        device=device,
        dtype=torch.float32,
    )

    conditions = [
        "clean",
        "learned_role",
        "learned_identity",
        "random_role",
    ]
    if args.eval_full_residual:
        conditions += ["full_role", "full_identity"]

    per_sample: List[Dict[str, Any]] = []
    by_condition_predictions: Dict[str, List[str]] = {
        c: [] for c in conditions
    }
    by_condition_scores: Dict[str, List[np.ndarray]] = {
        c: [] for c in conditions
    }
    successful_rows: List[Dict[str, Any]] = []

    # One random subspace per repeat. If repeats > 1, random_role per-sample
    # prediction is taken from repeat 0 for per-sample logs, while summary later
    # averages separate repeat summaries.
    random_bases = [
        random_orthonormal_basis(
            d_model=int(q_basis_cpu.shape[0]),
            k=dim,
            seed=args.seed + 17011 * layer + 733 * dim + rr,
            device=device,
        )
        for rr in range(max(1, int(args.random_repeats)))
    ]

    # For clean / learned / full conditions evaluate once.
    base_conditions = [c for c in conditions if c != "random_role"]

    for i, row in enumerate(
        tqdm(
            eval_rows,
            desc=f"eval:L{layer}:D{dim}",
        ),
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

            sample: Dict[str, Any] = {
                "sid": sid,
                "gt_target": str(pair.gt),
                "gt_source": OPPOSITE[str(pair.gt)],
                "pair_status": str(row.get("generation_pair_status", "")),
            }

            for condition in base_conditions:
                result = forward_intervened_eval(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layer=decoder_layers[layer],
                    target_a_positions=pair.original_a_positions,
                    target_b_positions=pair.original_b_positions,
                    source_state=source_cache[sid][layer],
                    relation_token_map=relation_token_map,
                    condition=condition,
                    learned_basis=learned_basis,
                    random_basis=None,
                )
                by_condition_predictions[condition].append(
                    str(result["prediction"])
                )
                by_condition_scores[condition].append(
                    np.asarray(result["scores"], dtype=np.float32)
                )
                sample[f"{condition}__prediction"] = str(result["prediction"])

            # Random repeat 0 for canonical sample log / initial summary.
            rand0 = forward_intervened_eval(
                model=model,
                batch=pair.original_batch,
                decoder_layer=decoder_layers[layer],
                target_a_positions=pair.original_a_positions,
                target_b_positions=pair.original_b_positions,
                source_state=source_cache[sid][layer],
                relation_token_map=relation_token_map,
                condition="random_role",
                learned_basis=None,
                random_basis=random_bases[0],
            )
            by_condition_predictions["random_role"].append(
                str(rand0["prediction"])
            )
            by_condition_scores["random_role"].append(
                np.asarray(rand0["scores"], dtype=np.float32)
            )
            sample["random_role__prediction"] = str(rand0["prediction"])

            per_sample.append(sample)
            successful_rows.append(dict(row))

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "eval",
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

    if not successful_rows:
        raise RuntimeError(f"No successful eval rows L{layer} D{dim}")

    clean_preds = by_condition_predictions["clean"]

    summary_rows: List[Dict[str, Any]] = []

    # Two evaluation subsets:
    subset_specs = [
        ("heldout_all", list(range(len(successful_rows)))),
        (
            "heldout_both_correct",
            [
                i for i, row in enumerate(successful_rows)
                if str(row.get("generation_pair_status", "")) == "both_correct"
            ],
        ),
    ]

    for subset_name, indices in subset_specs:
        if not indices:
            continue

        subset_rows = [successful_rows[i] for i in indices]
        subset_clean = [clean_preds[i] for i in indices]

        for condition in base_conditions + ["random_role"]:
            preds = [
                by_condition_predictions[condition][i]
                for i in indices
            ]
            scs = [
                by_condition_scores[condition][i]
                for i in indices
            ]
            row = summarize_condition(
                condition=condition,
                rows=subset_rows,
                predictions=preds,
                scores=scs,
                clean_predictions=subset_clean,
            )
            row.update({
                "version": VERSION,
                "layer": layer,
                "dim": dim,
                "eval_subset": subset_name,
                "random_repeat": 0 if condition == "random_role" else "",
            })
            summary_rows.append(row)

    # Extra random repeats only need random_role summaries.
    if len(random_bases) > 1:
        for rr in range(1, len(random_bases)):
            rand_preds: List[str] = []
            rand_scores: List[np.ndarray] = []

            for row in tqdm(
                successful_rows,
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
                    result = forward_intervened_eval(
                        model=model,
                        batch=pair.original_batch,
                        decoder_layer=decoder_layers[layer],
                        target_a_positions=pair.original_a_positions,
                        target_b_positions=pair.original_b_positions,
                        source_state=source_cache[sid][layer],
                        relation_token_map=relation_token_map,
                        condition="random_role",
                        learned_basis=None,
                        random_basis=random_bases[rr],
                    )
                    rand_preds.append(str(result["prediction"]))
                    rand_scores.append(
                        np.asarray(result["scores"], dtype=np.float32)
                    )
                finally:
                    if pair is not None:
                        receiver.release_pair(pair)

            for subset_name, indices in subset_specs:
                if not indices:
                    continue
                subset_rows = [successful_rows[i] for i in indices]
                subset_clean = [clean_preds[i] for i in indices]
                preds = [rand_preds[i] for i in indices]
                scs = [rand_scores[i] for i in indices]
                row = summarize_condition(
                    condition="random_role",
                    rows=subset_rows,
                    predictions=preds,
                    scores=scs,
                    clean_predictions=subset_clean,
                )
                row.update({
                    "version": VERSION,
                    "layer": layer,
                    "dim": dim,
                    "eval_subset": subset_name,
                    "random_repeat": rr,
                })
                summary_rows.append(row)

    return summary_rows, per_sample


# =============================================================================
# Reporting
# =============================================================================

def print_summary_table(rows: Sequence[Mapping[str, Any]]) -> None:
    # Main rows: random repeat 0 only.
    display = [
        r for r in rows
        if str(r.get("random_repeat", "")) in ("", "0")
    ]
    print("\n" + "=" * 138)
    print("DAS HELD-OUT INTERCHANGE RESULTS")
    print("=" * 138)
    print(
        f"{'layer':>6s} {'dim':>4s} {'subset':<22s} {'condition':<18s} "
        f"{'N':>5s} {'target':>9s} {'SOURCE/IIA':>11s} {'other':>8s} "
        f"{'change':>9s} {'cleanGT->src':>13s} {'src-tgt margin':>14s}"
    )
    print("-" * 138)
    for r in display:
        print(
            f"L{int(r['layer']):<5d} "
            f"{int(r['dim']):>4d} "
            f"{str(r['eval_subset']):<22s} "
            f"{str(r['condition']):<18s} "
            f"{int(r['N']):>5d} "
            f"{100*float(r['target_accuracy']):>8.2f}% "
            f"{100*float(r['source_follow_iia']):>10.2f}% "
            f"{100*float(r['other_rate']):>7.2f}% "
            f"{100*float(r['prediction_change_vs_clean']):>8.2f}% "
            f"{100*float(r['clean_correct_to_source_rate']):>12.2f}% "
            f"{float(r['source_minus_target_margin_mean']):>+13.4f}"
        )


def best_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        dict(r)
        for r in rows
        if str(r["condition"]) == "learned_role"
        and str(r["eval_subset"]) == "heldout_both_correct"
        and str(r.get("random_repeat", "")) in ("", "0")
    ]
    candidates.sort(
        key=lambda r: float(r["source_follow_iia"]),
        reverse=True,
    )
    return candidates


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.epochs < 1:
        raise ValueError("--epochs must be >=1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >=1")
    if args.train_size == 0 and not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be in (0,1)")
    if args.logit_temperature <= 0:
        raise ValueError("--logit-temperature must be >0")

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

    ioi = import_file(Path(args.ioi_script), "_das_ioi")
    producer = import_file(Path(args.producer_script), "_das_producer")
    receiver = import_file(Path(args.receiver_script), "_das_receiver")
    v3 = import_file(Path(args.v3_script), "_das_v3")
    base = import_file(Path(args.base_script), "_das_base")

    source_dir = Path(args.source_output_dir)
    extraction = source_dir / "extraction.jsonl"
    config_path = source_dir / "config.json"
    if not extraction.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Need config.json + extraction.jsonl in {source_dir}"
        )

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(source_config.get("model")) != args.model:
        raise RuntimeError(
            f"Source cache model={source_config.get('model')} != --model={args.model}"
        )

    rows = [
        r for r in read_jsonl(extraction)
        if str(r.get("gt")) in RELATIONS
    ]
    if not rows:
        raise RuntimeError("No extraction rows")

    train_rows, eval_rows = build_train_eval_split(
        rows=rows,
        train_status=args.train_pair_status,
        train_size=args.train_size,
        train_ratio=args.train_ratio,
        eval_max_samples=args.eval_max_samples,
        seed=args.split_seed,
    )

    union_by_sid: Dict[int, Dict[str, Any]] = {}
    for r in list(train_rows) + list(eval_rows):
        union_by_sid[int(r["sid"])] = dict(r)
    source_rows = sorted(
        union_by_sid.values(),
        key=lambda r: int(r["sid"]),
    )

    model = processor = None
    all_summary: List[Dict[str, Any]] = []
    all_history: List[Dict[str, Any]] = []

    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )

        # Freeze model. Autograd after intervention only needs basis gradients.
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        saved_max = getattr(args, "max_samples", None)
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(
                args,
                base,
            )
        finally:
            args.max_samples = saved_max

        layers = parse_int_list(args.layers)
        dims = parse_int_list(args.subspace_dims)

        for l in layers:
            if not 0 <= l < len(decoder_layers):
                raise ValueError(
                    f"Layer {l} outside 0..{len(decoder_layers)-1}"
                )

        # Get hidden size robustly from a source capture instead of assuming config nesting.
        cache_dtype = (
            np.float16
            if args.source_cache_dtype == "float16"
            else np.float32
        )

        source_cache = build_source_cache(
            args=args,
            rows=source_rows,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            layers=layers,
            records_by_sid=records_by_sid,
            prompt_rows=prompt_rows,
            base=base,
            v3=v3,
            receiver=receiver,
            storage_dtype=cache_dtype,
            error_path=error_path,
        )

        train_rows = [
            r for r in train_rows if int(r["sid"]) in source_cache
        ]
        eval_rows = [
            r for r in eval_rows if int(r["sid"]) in source_cache
        ]
        if not train_rows or not eval_rows:
            raise RuntimeError("Train/eval empty after source-cache failures")

        first_sid = int(train_rows[0]["sid"])
        first_layer = layers[0]
        d_model = int(
            np.asarray(source_cache[first_sid][first_layer]["A"]).shape[0]
        )

        for d in dims:
            if d > d_model:
                raise ValueError(
                    f"Subspace dim {d} > hidden size {d_model}"
                )

        config = {
            "version": VERSION,
            "model": args.model,
            "repo_id": getattr(spec, "repo_id", ""),
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "d_model": d_model,
            "layers": layers,
            "subspace_dims": dims,
            "train_pair_status": args.train_pair_status,
            "train_sids": [int(r["sid"]) for r in train_rows],
            "eval_sids": [int(r["sid"]) for r in eval_rows],
            "N_train": len(train_rows),
            "N_eval": len(eval_rows),
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "random_repeats": args.random_repeats,
            "eval_full_residual": args.eval_full_residual,
            "intervention": "h' = h + QQ^T(s-h), Q=qr(raw_basis).Q",
            "role_mapping": "source B->target A; source A->target B",
            "identity_mapping": "source A->target A; source B->target B",
            "audit": audit,
        }
        write_json(out_dir / "config.json", config)

        print("\n" + "=" * 120)
        print("DAS-STYLE DISTRIBUTED SPATIAL RELATION SEARCH")
        print("=" * 120)
        print("model             :", args.model)
        print("hidden size       :", d_model)
        print("layers            :", layers)
        print("subspace dims     :", dims)
        print("train status      :", args.train_pair_status)
        print("N train           :", len(train_rows))
        print("N heldout eval    :", len(eval_rows))
        print("both-correct eval :", sum(
            str(r.get("generation_pair_status", "")) == "both_correct"
            for r in eval_rows
        ))
        print("=" * 120, flush=True)

        for layer in layers:
            for dim in dims:
                print(
                    f"\n>>> TRAIN DAS L{layer} D{dim}",
                    flush=True,
                )

                result = train_one_das(
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
                    d_model=d_model,
                    error_path=error_path,
                )

                all_history.extend(result.history)

                basis_path = out_dir / f"basis_L{layer}_D{dim}.npz"
                np.savez_compressed(
                    basis_path,
                    Q=result.q_basis_cpu,
                    layer=np.asarray([layer], dtype=np.int32),
                    dim=np.asarray([dim], dtype=np.int32),
                )

                print(
                    f">>> EVAL DAS L{layer} D{dim}",
                    flush=True,
                )
                summary_rows, sample_rows = evaluate_one_setting(
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
                    error_path=error_path,
                )

                for row in summary_rows:
                    row["train_source_follow_last_epoch"] = result.train_source_follow
                    row["train_loss_last_epoch"] = result.train_loss
                all_summary.extend(summary_rows)

                sample_path = out_dir / f"samples_L{layer}_D{dim}.jsonl"
                for row in sample_rows:
                    append_jsonl(sample_path, row)

                write_csv(
                    out_dir / "summary.csv",
                    all_summary,
                )
                write_csv(
                    out_dir / "train_history.csv",
                    all_history,
                )

                print_summary_table(summary_rows)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print_summary_table(all_summary)

        best = best_rows(all_summary)
        print("\n" + "=" * 100)
        print("BEST HELD-OUT BOTH-CORRECT LEARNED ROLE IIA")
        print("=" * 100)
        for rank, row in enumerate(best[:10], start=1):
            print(
                f"{rank:02d}. L{int(row['layer'])} D{int(row['dim'])} "
                f"IIA={100*float(row['source_follow_iia']):.2f}% "
                f"target={100*float(row['target_accuracy']):.2f}% "
                f"change={100*float(row['prediction_change_vs_clean']):.2f}% "
                f"margin={float(row['source_minus_target_margin_mean']):+.4f}"
            )

        report_lines = [
            f"version: {VERSION}",
            f"model: {args.model}",
            f"hidden_size: {d_model}",
            f"layers: {layers}",
            f"subspace_dims: {dims}",
            f"N_train: {len(train_rows)}",
            f"N_eval: {len(eval_rows)}",
            "",
            "PRIMARY METRIC",
            "source_follow_iia = fraction of held-out target runs whose next-token",
            "relation prediction becomes the source/swapped opposite relation after",
            "only the learned residual subspace is interchanged.",
            "",
            "HOW TO READ",
            "Strong evidence:",
            "  learned_role >> random_role",
            "  learned_role >> learned_identity",
            "  heldout_both_correct IIA is high",
            "  a modest subspace dimension is sufficient",
            "",
            "Diagnostics:",
            "  full_role high but learned_role low -> optimizer/k may be insufficient",
            "  full_role low -> this layer's object residual is not a strong causal carrier",
            "  train IIA high but heldout low -> overfitting / adversarial subspace",
            "",
            "FILES",
            "summary.csv",
            "train_history.csv",
            "basis_L*_D*.npz",
            "samples_L*_D*.jsonl",
            "config.json",
        ]

        if best:
            report_lines += [
                "",
                "TOP HELDOUT BOTH-CORRECT SETTINGS",
            ]
            for row in best[:10]:
                report_lines.append(
                    f"L{int(row['layer'])} D{int(row['dim'])}: "
                    f"IIA={100*float(row['source_follow_iia']):.2f}%"
                )

        (out_dir / "report.txt").write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        print(" ", out_dir / "summary.csv")
        print(" ", out_dir / "train_history.csv")
        print(" ", out_dir / "report.txt")
        print(" ", out_dir / "config.json")

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
