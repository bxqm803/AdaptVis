#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy-based causal validation for Stage I and Stage III of the COCO spatial circuit.

Goal
====
Replace hard-to-interpret normalized L2 / CE values with directly readable ACC.

Working hypotheses
------------------
Stage I (Direction / relation-bearing senders):
    Removing Stage-I head output at subject/reference tokens should reduce the
    spatial-relation information decodable from LATER object-token head states,
    then reduce prompt-last relation information and answer accuracy.

Stage III (Receiver / object->last integration):
    Removing only selected receivers' CLEAN object->prompt-last writes should
    leave object-token relation ACC essentially unchanged, but reduce
    prompt-last relation ACC and answer accuracy.

Main outputs
------------
1) object_probe_accuracy.csv
   Frozen CLEAN-trained 4-way probe on the concatenated pre-W_O attention-head
   state difference z(A)-z(B) for each layer.

2) last_probe_accuracy.csv
   Frozen CLEAN-trained 4-way probe on prompt-last decoder-block hidden state
   for each layer.

3) answer_accuracy.csv
   Four-way teacher-forced next-token relation ACC for:
       clean
       stage1_ablation
       stage1_random
       stage3_cut
       stage3_random

4) generation_accuracy.csv  (unless --no-run-generation)
   Full greedy autoregressive generation ACC, parse rate, Correct->Wrong,
   Wrong->Correct, and net changes relative to clean.

5) accuracy_report.txt
   Compact human-readable summary.

Probe protocol
--------------
The probe is NEVER retrained after intervention.
For each repeat:
    * make a stratified CLEAN 15% train / 85% test split;
    * fit the old AdaptVis centered class-mean direction probe on CLEAN train;
    * freeze center + class directions;
    * evaluate exactly the same held-out samples under all intervention conditions.

Thus an intervention-induced ACC drop means that the spatial code learned from
clean states is no longer preserved in the intervened states.

Interventions
-------------
Stage I:
    Zero selected pre-W_O head slices ONLY at object-token positions.

Stage III:
    For each selected receiver h, compute from a CLEAN pass:

        c_h = W_O^h sum_{s in object} A_h[last,s] V_h[s]

    and subtract sum_h c_h from that layer's attention output at prompt-last.
    The receiver head is not deleted; only its clean object->last write is cut.

Generation
----------
During model.generate(use_cache=True), interventions are applied only on the
PREFILL forward pass. This is deliberate:
    * Stage I modifies object states / the prefill cache.
    * Stage III cuts the object->original-prompt-last write that produces the
      first answer-token state and propagates through downstream prefill states.
The script reports this explicitly as "prefill intervention generation".

Repository dependencies
-----------------------
Place next to:
    validate_three_stage_spatial_circuit_v1.py
    analyze_coco_ioi_backward_circuit_v1.py
    analyze_coco_producer_qk_ov_v1.py
    analyze_coco_receiver_qkv_v1.py
    analyze_spatial_storage_transport_utilization_v3.py
    analyze_coco_centroid_generation_step1_v4.py
    analyze_coco_flip_attention_spatial_vectors_v1.py

Requires v3 cache:
    output/spatial_storage_transport_utilization/coco/qwen-3b/

Smoke test (40 samples, generation on those 40)
------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python -u validate_stage1_stage3_accuracy_circuit_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --max-samples 40 \
  --generation-max-samples 40 \
  --device cuda:0 \
  --output-dir output/qwen3b_stage1_stage3_accuracy_smoke \
  --overwrite

Full 440
--------
CUDA_VISIBLE_DEVICES=0 python -u validate_stage1_stage3_accuracy_circuit_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --max-samples 0 \
  --generation-max-samples 0 \
  --probe-train-ratio 0.15 \
  --probe-repeats 5 \
  --probe-seed 1 \
  --device cuda:0 \
  --output-dir output/qwen3b_stage1_stage3_accuracy_v1 \
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


VERSION = "stage1-stage3-accuracy-circuit-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}

DEFAULT_STAGE1 = "23:5,23:1,19:13,23:0,20:8"
DEFAULT_STAGE3 = "26:4,26:2,26:6,26:0,24:4"
DEFAULT_LAYERS = "18-35"

REL_PATTERNS: Dict[str, Sequence[str]] = {
    "left": (r"\bleft\s+of\b", r"\bto\s+the\s+left\b", r"\bleft\b"),
    "right": (r"\bright\s+of\b", r"\bto\s+the\s+right\b", r"\bright\b"),
    "above": (r"\bon\s+top\s+of\b", r"\batop\b", r"\babove\b", r"\bover\b"),
    "below": (r"\bunderneath\b", r"\bbeneath\b", r"\bbelow\b", r"\bunder\b"),
}


# =============================================================================
# CLI / generic
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

    p.add_argument("--stage1-heads", default=DEFAULT_STAGE1)
    p.add_argument("--stage3-heads", default=DEFAULT_STAGE3)
    p.add_argument("--object-probe-layers", default=DEFAULT_LAYERS)
    p.add_argument("--last-probe-layers", default=DEFAULT_LAYERS)

    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means all source samples. Use 40 for smoke test.",
    )
    p.add_argument("--sample-seed", type=int, default=17)

    p.add_argument("--probe-train-ratio", type=float, default=0.15)
    p.add_argument("--probe-repeats", type=int, default=5)
    p.add_argument("--probe-seed", type=int, default=1)

    p.add_argument(
        "--run-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--generation-max-samples",
        type=int,
        default=0,
        help="0 means all analyzed samples. Stratified subset if >0.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--min-new-tokens", type=int, default=1)

    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=29)

    p.add_argument(
        "--three-stage-script",
        default="validate_three_stage_spatial_circuit_v1.py",
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([dict(r) for r in rows])


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_head(text: str) -> Tuple[int, int]:
    x = str(text).strip().upper().replace("L", "").replace("H", ":")
    while "::" in x:
        x = x.replace("::", ":")
    if ":" not in x:
        raise ValueError(f"Bad head {text!r}")
    a, b = x.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        h = parse_head(item)
        if h not in seen:
            seen.add(h)
            out.append(h)
    if not out:
        raise ValueError("No heads selected")
    return out


def hname(h: Tuple[int, int]) -> str:
    return f"L{h[0]}H{h[1]:02d}"


def parse_layers(text: str, n_layers: int) -> List[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(n_layers))
    out: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(chunk))
    out = sorted(set(out))
    bad = [x for x in out if not 0 <= x < n_layers]
    if bad:
        raise ValueError(f"Layers outside 0..{n_layers-1}: {bad}")
    return out


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def one_line(text: Any) -> str:
    return " ".join(str(text).split())


def parse_generation(text: str) -> Optional[str]:
    text = one_line(text).lower()
    hits: List[Tuple[int, int, str]] = []
    for rel, patterns in REL_PATTERNS.items():
        for pri, pat in enumerate(patterns):
            m = re.search(pat, text)
            if m:
                hits.append((m.start(), pri, rel))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(x) for x in rows]
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda x: int(x["sid"]))
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)
    rng = random.Random(seed)
    for v in groups.values():
        rng.shuffle(v)
    labels = [r for r in RELATIONS if r in groups]
    cur = {r: 0 for r in labels}
    out: List[Dict[str, Any]] = []
    while len(out) < limit:
        moved = False
        for r in labels:
            i = cur[r]
            if i < len(groups[r]) and len(out) < limit:
                out.append(groups[r][i])
                cur[r] += 1
                moved = True
        if not moved:
            break
    return sorted(out, key=lambda x: int(x["sid"]))


# =============================================================================
# Probe: exactly a frozen clean-trained centered class-direction classifier
# =============================================================================

def stratified_train_indices(labels: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls in range(len(RELATIONS)):
        idx = np.flatnonzero(labels == cls)
        idx = idx.copy()
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * ratio)))
        n = min(n, max(1, len(idx) - 1)) if len(idx) > 1 else 1
        selected.extend(map(int, idx[:n]))
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def fit_direction_probe(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    center = x.mean(axis=0)
    centered = x - center[None, :]
    dirs = []
    for cls in range(len(RELATIONS)):
        values = centered[y == cls]
        if len(values) == 0:
            raise RuntimeError(f"No train examples for class {RELATIONS[cls]}")
        d = values.mean(axis=0)
        norm = np.linalg.norm(d)
        if norm <= 1e-12:
            raise RuntimeError(f"Zero direction prototype for {RELATIONS[cls]}")
        dirs.append(d / norm)
    return center.astype(np.float32), np.asarray(dirs, dtype=np.float32)


def predict_direction_probe(x: np.ndarray, center: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norm, 1e-12, None)
    scores = x @ np.asarray(dirs, dtype=np.float64).T
    return scores.argmax(axis=1).astype(np.int64)


def repeated_frozen_probe_accuracy(
    *,
    clean_x: np.ndarray,
    condition_x: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_ratio: float,
    repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    per_condition: Dict[str, List[float]] = defaultdict(list)
    per_disagree: Dict[str, List[float]] = defaultdict(list)

    n = len(labels)
    all_idx = np.arange(n, dtype=np.int64)
    for repeat in range(repeats):
        train = stratified_train_indices(labels, train_ratio, seed + repeat)
        mask = np.ones(n, dtype=bool)
        mask[train] = False
        test = all_idx[mask]
        center, dirs = fit_direction_probe(clean_x[train], labels[train])
        clean_pred = predict_direction_probe(clean_x[test], center, dirs)
        for name, x in condition_x.items():
            pred = predict_direction_probe(x[test], center, dirs)
            per_condition[name].append(float(np.mean(pred == labels[test])))
            per_disagree[name].append(float(np.mean(pred != clean_pred)))

    out: List[Dict[str, Any]] = []
    for name in condition_x:
        vals = np.asarray(per_condition[name], dtype=np.float64)
        dis = np.asarray(per_disagree[name], dtype=np.float64)
        out.append({
            "condition": name,
            "accuracy_mean": float(vals.mean()),
            "accuracy_std": float(vals.std()),
            "prediction_disagreement_vs_clean_mean": float(dis.mean()),
            "prediction_disagreement_vs_clean_std": float(dis.std()),
            "repeats": int(repeats),
            "train_ratio": float(train_ratio),
        })
    return out


# =============================================================================
# Feature/intervention hooks
# =============================================================================

class FeatureInterventionCapture:
    """Zero Stage-I head slices at object positions and capture layer features."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        object_layers: Sequence[int],
        last_layers: Sequence[int],
        zero_heads: Sequence[Tuple[int, int]],
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        prompt_last: int,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver_module = receiver_module
        self.object_layers = set(map(int, object_layers))
        self.last_layers = set(map(int, last_layers))
        self.zero_by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head in zero_heads:
            self.zero_by_layer[int(layer)].append(int(head))
        self.subject_positions = sorted(set(map(int, subject_positions)))
        self.reference_positions = sorted(set(map(int, reference_positions)))
        self.prompt_last = int(prompt_last)
        self.handles: List[Any] = []
        self.object_features: Dict[int, np.ndarray] = {}
        self.last_features: Dict[int, np.ndarray] = {}

    def __enter__(self) -> "FeatureInterventionCapture":
        o_layers = sorted(self.object_layers | set(self.zero_by_layer))
        for layer in o_layers:
            attention = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            o_proj = getattr(attention, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"L{layer} has no o_proj")
            shape = self.receiver_module.resolve_attention_shape(attention)
            n_heads = int(shape.n_query_heads)
            width = int(o_proj.weight.shape[1])
            if width % n_heads != 0:
                raise RuntimeError(f"L{layer} o_proj width/head mismatch")
            head_dim = width // n_heads

            def make_pre_hook(layer_index: int, nh: int, hd: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None
                    modified = x
                    zeros = self.zero_by_layer.get(layer_index, [])
                    if zeros:
                        modified = x.clone()
                        positions = sorted(set(self.subject_positions + self.reference_positions))
                        for pos in positions:
                            if 0 <= pos < int(modified.shape[1]):
                                for head in zeros:
                                    a = head * hd
                                    b = a + hd
                                    modified[0, pos, a:b] = 0

                    if layer_index in self.object_layers:
                        sp = torch.as_tensor(
                            self.subject_positions, device=modified.device, dtype=torch.long
                        )
                        rp = torch.as_tensor(
                            self.reference_positions, device=modified.device, dtype=torch.long
                        )
                        if int(sp.max()) < int(modified.shape[1]) and int(rp.max()) < int(modified.shape[1]):
                            a = modified[0].index_select(0, sp).mean(dim=0)
                            b = modified[0].index_select(0, rp).mean(dim=0)
                            self.object_features[layer_index] = (
                                (a - b).detach().float().cpu().numpy().astype(np.float32)
                            )

                    if modified is x:
                        return None
                    return (modified, *inputs[1:])
                return hook

            self.handles.append(o_proj.register_forward_pre_hook(make_pre_hook(layer, n_heads, head_dim)))

        for layer in sorted(self.last_layers):
            block = self.decoder_layers[layer]

            def make_block_hook(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if 0 <= self.prompt_last < int(hidden.shape[1]):
                        self.last_features[layer_index] = (
                            hidden[0, self.prompt_last]
                            .detach().float().cpu().numpy().astype(np.float32)
                        )
                return hook

            self.handles.append(block.register_forward_hook(make_block_hook(layer)))
        return self

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class PrefillObjectZero:
    """Generation-safe Stage-I hook: only acts when full prompt is present."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        heads: Sequence[Tuple[int, int]],
        object_positions: Sequence[int],
        prompt_length: int,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver_module = receiver_module
        self.by_layer: Dict[int, List[int]] = defaultdict(list)
        for l, h in heads:
            self.by_layer[int(l)].append(int(h))
        self.object_positions = sorted(set(map(int, object_positions)))
        self.prompt_length = int(prompt_length)
        self.handles: List[Any] = []
        self.applications = 0

    def __enter__(self) -> "PrefillObjectZero":
        for layer, heads in self.by_layer.items():
            attention = self.attention_helper.resolve_self_attention(self.decoder_layers[layer])
            o_proj = attention.o_proj
            shape = self.receiver_module.resolve_attention_shape(attention)
            nh = int(shape.n_query_heads)
            width = int(o_proj.weight.shape[1])
            hd = width // nh

            def make_hook(layer_index: int, head_ids: Sequence[int], head_dim: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None
                    # use_cache decode steps usually have S=1; only prefill has prompt length
                    if int(x.shape[1]) < self.prompt_length:
                        return None
                    modified = x.clone()
                    for pos in self.object_positions:
                        if 0 <= pos < int(modified.shape[1]):
                            for head in head_ids:
                                a = int(head) * head_dim
                                modified[0, pos, a:a+head_dim] = 0
                    self.applications += 1
                    return (modified, *inputs[1:])
                return hook
            self.handles.append(o_proj.register_forward_pre_hook(make_hook(layer, heads, hd)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


class PromptLastWriteCut:
    """Subtract fixed clean object->prompt-last vectors; safe for forward/generation."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        remove_by_layer: Mapping[int, np.ndarray],
        prompt_last: int,
        prompt_length: int,
        prefill_only: bool,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.remove_by_layer = {int(k): np.asarray(v, dtype=np.float32) for k, v in remove_by_layer.items()}
        self.prompt_last = int(prompt_last)
        self.prompt_length = int(prompt_length)
        self.prefill_only = bool(prefill_only)
        self.handles: List[Any] = []
        self.applications = 0

    def __enter__(self) -> "PromptLastWriteCut":
        for layer, vec in self.remove_by_layer.items():
            attention = self.attention_helper.resolve_self_attention(self.decoder_layers[layer])

            def make_hook(layer_index: int, residual: np.ndarray):
                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    hidden = first_3d_tensor(output)
                    if self.prefill_only and int(hidden.shape[1]) < self.prompt_length:
                        return None
                    if not 0 <= self.prompt_last < int(hidden.shape[1]):
                        return None
                    modified = hidden.clone()
                    delta = torch.as_tensor(residual, device=hidden.device, dtype=hidden.dtype)
                    modified[0, self.prompt_last] -= delta
                    self.applications += 1
                    return replace_first_3d_tensor(output, modified)
                return hook

            self.handles.append(attention.register_forward_hook(make_hook(layer, vec)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def first_3d_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim == 3:
            return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("No 3D hidden tensor in module output")


def replace_first_3d_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return items
    raise RuntimeError("Cannot replace attention hidden output")


# =============================================================================
# Clean receiver-write extraction
# =============================================================================

@torch.inference_mode()
def clean_object_writes(
    *,
    model: Any,
    batch: Mapping[str, Any],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    v3: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    heads: Sequence[Tuple[int, int]],
    object_positions: Sequence[int],
    prompt_last: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    layers = sorted(set(int(l) for l, _ in heads))
    _, traces = v3.trace_prompt_chunks(
        attention_helper=attention_helper,
        model=model,
        batch=batch,
        relation_token_map=relation_token_map,
        decoder_layers=decoder_layers,
        layers=layers,
        target_positions=sorted(set(map(int, object_positions)) | {int(prompt_last)}),
        chunk_size=max(1, len(layers)),
    )
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for head in heads:
        layer, h = head
        write, _mass = receiver_module.object_head_write(
            trace=traces[int(layer)],
            head=int(h),
            target_position=int(prompt_last),
            source_positions=object_positions,
        )
        out[head] = write.detach().float().cpu().numpy().astype(np.float32)
    return out


def sum_writes_by_layer(
    write_by_head: Mapping[Tuple[int, int], np.ndarray],
    heads: Sequence[Tuple[int, int]],
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for head in heads:
        layer = int(head[0])
        vec = np.asarray(write_by_head[head], dtype=np.float32)
        if layer not in out:
            out[layer] = np.zeros_like(vec)
        out[layer] += vec
    return out


# =============================================================================
# One forward / one generation
# =============================================================================

@torch.inference_mode()
def run_feature_forward(
    *,
    condition: str,
    model: Any,
    batch: Mapping[str, Any],
    base: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    object_layers: Sequence[int],
    last_layers: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
    zero_heads: Sequence[Tuple[int, int]],
    remove_by_layer: Optional[Mapping[int, np.ndarray]],
) -> Dict[str, Any]:
    capture = FeatureInterventionCapture(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        object_layers=object_layers,
        last_layers=last_layers,
        zero_heads=zero_heads,
        subject_positions=subject_positions,
        reference_positions=reference_positions,
        prompt_last=prompt_last,
    )
    cutter = PromptLastWriteCut(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        remove_by_layer=remove_by_layer or {},
        prompt_last=prompt_last,
        prompt_length=int(batch["input_ids"].shape[1]),
        prefill_only=False,
    )
    with capture, cutter:
        outputs = model(**batch, use_cache=False, return_dict=True)
    relation = base.relation_scores(
        outputs.logits[0, -1], dict(relation_token_map), gt=None
    )
    pred = str(relation["prediction"])
    logits = np.asarray(relation["logits"], dtype=np.float32)
    del outputs

    missing_obj = [l for l in object_layers if l not in capture.object_features]
    missing_last = [l for l in last_layers if l not in capture.last_features]
    if missing_obj or missing_last:
        raise RuntimeError(
            f"Missing captures condition={condition} object={missing_obj} last={missing_last}"
        )

    return {
        "condition": condition,
        "prediction": pred,
        "relation_logits": logits,
        "object": capture.object_features,
        "last": capture.last_features,
    }


def generation_kwargs(processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    tok = processor.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kw: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "pad_token_id": pad,
    }
    if tok.eos_token_id is not None:
        kw["eos_token_id"] = tok.eos_token_id
    return kw


@torch.inference_mode()
def run_generation(
    *,
    condition: str,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    object_positions: Sequence[int],
    prompt_last: int,
    zero_heads: Sequence[Tuple[int, int]],
    remove_by_layer: Optional[Mapping[int, np.ndarray]],
    gen_kw: Mapping[str, Any],
) -> Dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    zero = PrefillObjectZero(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        heads=zero_heads,
        object_positions=object_positions,
        prompt_length=prompt_length,
    )
    cut = PromptLastWriteCut(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        remove_by_layer=remove_by_layer or {},
        prompt_last=prompt_last,
        prompt_length=prompt_length,
        prefill_only=True,
    )
    with zero, cut:
        out = model.generate(**batch, **dict(gen_kw))
    seq = out.sequences[0]
    new = seq[prompt_length:]
    ids = [int(x) for x in new.detach().cpu().tolist()]
    text = processor.tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    pred = parse_generation(text)
    del out
    return {
        "condition": condition,
        "prediction": pred,
        "text": one_line(text),
        "token_ids": ids,
    }


# =============================================================================
# Accuracy summaries
# =============================================================================

def answer_accuracy_rows(
    predictions: Mapping[str, Sequence[Optional[str]]],
    labels_text: Sequence[str],
) -> List[Dict[str, Any]]:
    labels = list(labels_text)
    out: List[Dict[str, Any]] = []
    clean = list(predictions["clean"])
    for condition, preds_seq in predictions.items():
        preds = list(preds_seq)
        correct = [p == g for p, g in zip(preds, labels)]
        parse = [p in RELATIONS for p in preds]
        row: Dict[str, Any] = {
            "condition": condition,
            "N": len(labels),
            "accuracy": float(np.mean(correct)),
            "parse_rate": float(np.mean(parse)),
        }
        if condition != "clean":
            c2w = sum((c == g) and (p != g) for c, p, g in zip(clean, preds, labels))
            w2c = sum((c != g) and (p == g) for c, p, g in zip(clean, preds, labels))
            row.update({
                "correct_to_wrong": int(c2w),
                "wrong_to_correct": int(w2c),
                "net_correct_change": int(w2c - c2w),
                "accuracy_delta_vs_clean": float(np.mean(correct) - np.mean([c == g for c, g in zip(clean, labels)])),
                "prediction_change_rate_vs_clean": float(np.mean([p != c for p, c in zip(preds, clean)])),
            })
        else:
            row.update({
                "correct_to_wrong": 0,
                "wrong_to_correct": 0,
                "net_correct_change": 0,
                "accuracy_delta_vs_clean": 0.0,
                "prediction_change_rate_vs_clean": 0.0,
            })
        out.append(row)
    return out


def print_answer_table(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(f"{'condition':<20s} {'N':>5s} {'ACC':>9s} {'ΔACC':>9s} {'C->W':>7s} {'W->C':>7s} {'net':>7s} {'change':>9s} {'parse':>8s}")
    print("-" * 104)
    for r in rows:
        print(
            f"{str(r['condition']):<20s} {int(r['N']):>5d} "
            f"{100*float(r['accuracy']):>8.2f}% {100*float(r['accuracy_delta_vs_clean']):>+8.2f}% "
            f"{int(r['correct_to_wrong']):>7d} {int(r['wrong_to_correct']):>7d} "
            f"{int(r['net_correct_change']):>+7d} "
            f"{100*float(r['prediction_change_rate_vs_clean']):>8.2f}% "
            f"{100*float(r['parse_rate']):>7.2f}%"
        )


def print_probe_snapshot(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    layers_to_show: Sequence[int],
) -> None:
    by = {(int(r["layer"]), str(r["condition"])): r for r in rows}
    conditions = ["clean", "stage1_random", "stage1_ablation", "stage3_random", "stage3_cut"]
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'layer':>6s} " + " ".join(f"{c:>17s}" for c in conditions))
    print("-" * 100)
    for layer in layers_to_show:
        vals = []
        for c in conditions:
            r = by.get((int(layer), c))
            if r is None:
                vals.append("-".rjust(17))
            else:
                vals.append(f"{100*float(r['accuracy_mean']):6.2f}±{100*float(r['accuracy_std']):4.2f}%".rjust(17))
        print(f"L{layer:<4d} " + " ".join(vals))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0 < args.probe_train_ratio < 1:
        raise ValueError("--probe-train-ratio must be in (0,1)")
    if args.probe_repeats < 1:
        raise ValueError("--probe-repeats must be >=1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import the same tested helpers used by the previous experiment.
    three = import_file(Path(args.three_stage_script), "_acc_three")
    ioi = import_file(Path(args.ioi_script), "_acc_ioi")
    producer = import_file(Path(args.producer_script), "_acc_producer")
    receiver = import_file(Path(args.receiver_script), "_acc_receiver")
    v3 = import_file(Path(args.v3_script), "_acc_v3")
    base = import_file(Path(args.base_script), "_acc_base")
    attention_helper = import_file(Path(args.attention_helper), "_acc_attention")

    source_dir = Path(args.source_output_dir)
    extraction = source_dir / "extraction.jsonl"
    config_path = source_dir / "config.json"
    if not extraction.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Need v3 config.json + extraction.jsonl in {source_dir}"
        )
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(source_config.get("model")) != args.model:
        raise RuntimeError(
            f"Source cache model={source_config.get('model')} != {args.model}"
        )
    rows = read_jsonl(extraction)
    rows = [r for r in rows if str(r.get("gt")) in RELATIONS]
    rows = stratified_subset(rows, args.max_samples, args.sample_seed)
    if not rows:
        raise RuntimeError("No source rows")

    model = processor = None
    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )
        # prepare_data_helpers() interprets args.max_samples as a DATASET truncation.
        # Here --max-samples means an analysis subset selected from the full source
        # cache, so always load the full dataset and restore the CLI value after.
        _analysis_max_samples = args.max_samples
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)
        finally:
            args.max_samples = _analysis_max_samples

        stage1 = parse_heads(args.stage1_heads)
        stage3 = parse_heads(args.stage3_heads)
        if set(stage1) & set(stage3):
            raise RuntimeError("Stage1 and Stage3 heads must be disjoint")

        object_layers = parse_layers(args.object_probe_layers, len(decoder_layers))
        last_layers = parse_layers(args.last_probe_layers, len(decoder_layers))

        for label, heads in (("stage1", stage1), ("stage3", stage3)):
            three.validate_heads(
                heads=heads,
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver_module=receiver,
                label=label,
            )

        excluded = set(stage1) | set(stage3)
        stage1_random = three.matched_random_heads(
            target=stage1,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            excluded=excluded,
            seed=args.seed + 101,
        )
        excluded |= set(stage1_random)
        stage3_random = three.matched_random_heads(
            target=stage3,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            excluded=excluded,
            seed=args.seed + 303,
        )

        conditions = (
            "clean",
            "stage1_ablation",
            "stage1_random",
            "stage3_cut",
            "stage3_random",
        )

        config = {
            "version": VERSION,
            "model": args.model,
            "repo_id": getattr(spec, "repo_id", ""),
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "N": len(rows),
            "stage1_heads": [hname(h) for h in stage1],
            "stage1_random": [hname(h) for h in stage1_random],
            "stage3_heads": [hname(h) for h in stage3],
            "stage3_random": [hname(h) for h in stage3_random],
            "object_probe_layers": object_layers,
            "last_probe_layers": last_layers,
            "probe_train_ratio": args.probe_train_ratio,
            "probe_repeats": args.probe_repeats,
            "probe_seed": args.probe_seed,
            "run_generation": args.run_generation,
            "generation_max_samples": args.generation_max_samples,
            "generation_intervention_scope": "prefill only",
            "audit": audit,
        }
        write_json(out_dir / "config.json", config)

        print("\n" + "=" * 120)
        print("ACCURACY-BASED STAGE-I / STAGE-III VALIDATION")
        print("=" * 120)
        print("Stage I       :", ", ".join(hname(h) for h in stage1))
        print("Stage I random:", ", ".join(hname(h) for h in stage1_random))
        print("Stage III     :", ", ".join(hname(h) for h in stage3))
        print("Stage III rand:", ", ".join(hname(h) for h in stage3_random))
        print("Object layers :", object_layers)
        print("Last layers   :", last_layers)
        print("Samples       :", len(rows))
        print("=" * 120, flush=True)

        # sid -> condition -> layer -> feature
        object_features: Dict[str, Dict[int, List[np.ndarray]]] = {
            c: {l: [] for l in object_layers} for c in conditions
        }
        last_features: Dict[str, Dict[int, List[np.ndarray]]] = {
            c: {l: [] for l in last_layers} for c in conditions
        }
        labels_text: List[str] = []
        sids: List[int] = []
        next_token_predictions: Dict[str, List[Optional[str]]] = {c: [] for c in conditions}

        # Generation rows are a stratified subset of analyzed rows.
        gen_rows = stratified_subset(rows, args.generation_max_samples, args.sample_seed + 999)
        gen_sid_set = {int(r["sid"]) for r in gen_rows} if args.run_generation else set()
        generation_records: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        gen_kw = generation_kwargs(processor, args)

        sample_metrics_path = out_dir / "sample_predictions.jsonl"
        errors_path = out_dir / "errors.jsonl"

        all_receiver_heads = list(dict.fromkeys(stage3 + stage3_random))

        for index, row in enumerate(tqdm(rows, desc=f"accuracy:{args.model}"), start=1):
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
                gt = str(pair.gt)
                object_positions = pair.original_object_positions

                # One clean trace gives exact Stage-III object->last writes for both
                # named and matched-random receiver bundles.
                clean_writes = clean_object_writes(
                    model=model,
                    batch=pair.original_batch,
                    decoder_layers=decoder_layers,
                    attention_helper=attention_helper,
                    receiver_module=receiver,
                    v3=v3,
                    relation_token_map=relation_token_map,
                    heads=all_receiver_heads,
                    object_positions=object_positions,
                    prompt_last=pair.original_prompt_last,
                )
                stage3_remove = sum_writes_by_layer(clean_writes, stage3)
                stage3_random_remove = sum_writes_by_layer(clean_writes, stage3_random)

                specs = [
                    ("clean", [], {}),
                    ("stage1_ablation", stage1, {}),
                    ("stage1_random", stage1_random, {}),
                    ("stage3_cut", [], stage3_remove),
                    ("stage3_random", [], stage3_random_remove),
                ]

                sample_row: Dict[str, Any] = {"sid": int(pair.sid), "gt": gt}
                for condition, zero_heads, remove_by_layer in specs:
                    result = run_feature_forward(
                        condition=condition,
                        model=model,
                        batch=pair.original_batch,
                        base=base,
                        relation_token_map=relation_token_map,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        receiver_module=receiver,
                        object_layers=object_layers,
                        last_layers=last_layers,
                        subject_positions=pair.original_a_positions,
                        reference_positions=pair.original_b_positions,
                        prompt_last=pair.original_prompt_last,
                        zero_heads=zero_heads,
                        remove_by_layer=remove_by_layer,
                    )
                    for layer in object_layers:
                        object_features[condition][layer].append(result["object"][layer])
                    for layer in last_layers:
                        last_features[condition][layer].append(result["last"][layer])
                    next_token_predictions[condition].append(result["prediction"])
                    sample_row[f"{condition}_next_token_prediction"] = result["prediction"]

                if args.run_generation and int(pair.sid) in gen_sid_set:
                    for condition, zero_heads, remove_by_layer in specs:
                        g = run_generation(
                            condition=condition,
                            model=model,
                            processor=processor,
                            batch=pair.original_batch,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            receiver_module=receiver,
                            object_positions=object_positions,
                            prompt_last=pair.original_prompt_last,
                            zero_heads=zero_heads,
                            remove_by_layer=remove_by_layer,
                            gen_kw=gen_kw,
                        )
                        generation_records[int(pair.sid)][condition] = g
                        sample_row[f"{condition}_generation_prediction"] = g["prediction"]
                        sample_row[f"{condition}_generation_text"] = g["text"]

                labels_text.append(gt)
                sids.append(int(pair.sid))
                append_jsonl(sample_metrics_path, sample_row)

                if args.print_every > 0 and index % args.print_every == 0:
                    compact = " ".join(
                        f"{c}={next_token_predictions[c][-1]}"
                        for c in conditions
                    )
                    tqdm.write(f"sid={pair.sid} gt={gt} | next-token {compact}")

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": int(row["sid"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"\n[ERROR sid={row['sid']}] {type(exc).__name__}: {exc}", flush=True)
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache_every > 0 and index % args.empty_cache_every == 0:
                    torch.cuda.empty_cache()

        if not labels_text:
            raise RuntimeError("No successful samples")
        labels = np.asarray([REL_TO_ID[x] for x in labels_text], dtype=np.int64)
        n = len(labels)
        if any(len(next_token_predictions[c]) != n for c in conditions):
            raise RuntimeError("Condition sample count mismatch")

        # ---------------------------------------------------------------------
        # Frozen clean-trained object probes layer-by-layer.
        # ---------------------------------------------------------------------
        object_probe_rows: List[Dict[str, Any]] = []
        for layer in object_layers:
            clean_x = np.stack(object_features["clean"][layer], axis=0)
            condition_x = {
                c: np.stack(object_features[c][layer], axis=0)
                for c in conditions
            }
            rows_layer = repeated_frozen_probe_accuracy(
                clean_x=clean_x,
                condition_x=condition_x,
                labels=labels,
                train_ratio=args.probe_train_ratio,
                repeats=args.probe_repeats,
                seed=args.probe_seed,
            )
            clean_acc = next(r["accuracy_mean"] for r in rows_layer if r["condition"] == "clean")
            for r in rows_layer:
                r["layer"] = int(layer)
                r["accuracy_delta_vs_clean"] = float(r["accuracy_mean"] - clean_acc)
                object_probe_rows.append(r)

        # ---------------------------------------------------------------------
        # Frozen clean-trained prompt-last probes layer-by-layer.
        # ---------------------------------------------------------------------
        last_probe_rows: List[Dict[str, Any]] = []
        for layer in last_layers:
            clean_x = np.stack(last_features["clean"][layer], axis=0)
            condition_x = {
                c: np.stack(last_features[c][layer], axis=0)
                for c in conditions
            }
            rows_layer = repeated_frozen_probe_accuracy(
                clean_x=clean_x,
                condition_x=condition_x,
                labels=labels,
                train_ratio=args.probe_train_ratio,
                repeats=args.probe_repeats,
                seed=args.probe_seed,
            )
            clean_acc = next(r["accuracy_mean"] for r in rows_layer if r["condition"] == "clean")
            for r in rows_layer:
                r["layer"] = int(layer)
                r["accuracy_delta_vs_clean"] = float(r["accuracy_mean"] - clean_acc)
                last_probe_rows.append(r)

        write_csv(out_dir / "object_probe_accuracy.csv", object_probe_rows)
        write_csv(out_dir / "last_probe_accuracy.csv", last_probe_rows)

        # Teacher-forced 4-way next-token accuracy on all successfully analyzed samples.
        answer_rows = answer_accuracy_rows(next_token_predictions, labels_text)
        write_csv(out_dir / "answer_accuracy.csv", answer_rows)
        print_answer_table("FOUR-WAY NEXT-TOKEN RELATION ACCURACY", answer_rows)

        # Full generation accuracy on selected generation SIDs.
        generation_rows: List[Dict[str, Any]] = []
        if args.run_generation:
            ordered_gen_sids = [sid for sid in sids if sid in generation_records]
            gen_labels = [labels_text[sids.index(sid)] for sid in ordered_gen_sids]
            gen_preds: Dict[str, List[Optional[str]]] = {
                c: [generation_records[sid][c]["prediction"] for sid in ordered_gen_sids]
                for c in conditions
            }
            generation_rows = answer_accuracy_rows(gen_preds, gen_labels)
            write_csv(out_dir / "generation_accuracy.csv", generation_rows)
            print_answer_table("FULL GREEDY GENERATION ACCURACY (PREFILL INTERVENTION)", generation_rows)

        # Show a compact layer snapshot including intervention boundaries.
        key_layers = sorted(set(
            [object_layers[0], object_layers[-1]]
            + [l for l, _ in stage1]
            + [l+1 for l, _ in stage1 if l+1 in object_layers]
            + [l for l, _ in stage3]
            + [l+1 for l, _ in stage3 if l+1 in object_layers]
        ))
        key_layers = [l for l in key_layers if l in object_layers]
        print_probe_snapshot("OBJECT A-B RELATION PROBE ACC (CLEAN-TRAINED, FROZEN)", object_probe_rows, key_layers)
        key_last = [l for l in key_layers if l in last_layers]
        print_probe_snapshot("PROMPT-LAST RELATION PROBE ACC (CLEAN-TRAINED, FROZEN)", last_probe_rows, key_last)

        # Extra intuitive checks.
        object_by = {(r["layer"], r["condition"]): r for r in object_probe_rows}
        last_by = {(r["layer"], r["condition"]): r for r in last_probe_rows}
        late_object_layers = [l for l in object_layers if l > max(l for l, _ in stage1)]
        post_stage3_last_layers = [l for l in last_layers if l >= min(l for l, _ in stage3)]

        # Positive "excess drop" means the named intervention lowers ACC more
        # than its matched random control.
        stage1_object_drop = safe_mean(
            object_by[(l, "stage1_random")]["accuracy_delta_vs_clean"]
            - object_by[(l, "stage1_ablation")]["accuracy_delta_vs_clean"]
            for l in late_object_layers
        ) if late_object_layers else float("nan")
        stage3_object_drop = safe_mean(
            object_by[(l, "stage3_random")]["accuracy_delta_vs_clean"]
            - object_by[(l, "stage3_cut")]["accuracy_delta_vs_clean"]
            for l in object_layers
        )
        stage3_last_drop = safe_mean(
            last_by[(l, "stage3_random")]["accuracy_delta_vs_clean"]
            - last_by[(l, "stage3_cut")]["accuracy_delta_vs_clean"]
            for l in post_stage3_last_layers
        ) if post_stage3_last_layers else float("nan")

        report = []
        report.append(f"version: {VERSION}")
        report.append(f"model: {args.model}")
        report.append(f"N successful: {n}")
        report.append("")
        report.append("MAIN INTUITIVE CLAIMS")
        report.append("---------------------")
        report.append(
            "Stage I support is strongest if later OBJECT relation-probe ACC drops "
            "substantially more than same-layer random ablation, followed by lower "
            "last-token and answer ACC."
        )
        report.append(
            "Stage III support is strongest if OBJECT relation-probe ACC stays near "
            "clean/random, while prompt-LAST probe ACC and answer/generation ACC drop."
        )
        report.append("")
        report.append(f"mean Stage-I late-object ACC excess drop vs random: {100*stage1_object_drop:+.2f} percentage points")
        report.append(f"mean Stage-III object ACC excess drop vs random: {100*stage3_object_drop:+.2f} percentage points")
        report.append(f"mean Stage-III post-receiver last-token ACC excess drop vs random: {100*stage3_last_drop:+.2f} percentage points")
        report.append("")
        report.append("NEXT-TOKEN ACC")
        for r in answer_rows:
            report.append(
                f"{r['condition']:<18s} ACC={100*r['accuracy']:.2f}% "
                f"delta={100*r['accuracy_delta_vs_clean']:+.2f}pp "
                f"C->W={r['correct_to_wrong']} W->C={r['wrong_to_correct']} net={r['net_correct_change']:+d}"
            )
        if generation_rows:
            report.append("")
            report.append("FULL GENERATION ACC (prefill intervention)")
            for r in generation_rows:
                report.append(
                    f"{r['condition']:<18s} ACC={100*r['accuracy']:.2f}% "
                    f"delta={100*r['accuracy_delta_vs_clean']:+.2f}pp "
                    f"C->W={r['correct_to_wrong']} W->C={r['wrong_to_correct']} net={r['net_correct_change']:+d} "
                    f"parse={100*r['parse_rate']:.2f}%"
                )
        report.append("")
        report.append("FILES")
        report.append("object_probe_accuracy.csv : object-token head-space ACC vs layer")
        report.append("last_probe_accuracy.csv   : prompt-last hidden-state ACC vs layer")
        report.append("answer_accuracy.csv       : four-way next-token ACC + transitions")
        if generation_rows:
            report.append("generation_accuracy.csv   : full greedy generation ACC + transitions")
        report.append("sample_predictions.jsonl  : per-sample predictions/text")

        (out_dir / "accuracy_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n" + "\n".join(report), flush=True)
        print("\nSaved to", out_dir)

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
