#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-head functional mismatch analysis for COCO two-object spatial reasoning.

Purpose
-------
This script asks a more specific question than a plain correct-vs-wrong head
activation comparison:

    Do wrong samples deviate from the normal FUNCTIONAL behavior of the same
    heads observed on correct samples?

It reuses the 2x2 counterfactual design from
`analyze_coco_reasoning_vs_relay_factorial_v2.py`:

    00: original image + original query
    01: original image + role-swapped query
    10: horizontal-flipped image + original query
    11: horizontal-flipped image + role-swapped query

For each selected attention head, it extracts:

1. Object-pair pre-WO output (for producer heads)
   - used to measure visual-layout V, query Q, and interaction I factors.

2. Prompt-last pre-WO head output
   - used to measure per-head relation-composition interaction.

3. Prompt-last attention mass to identity A and identity B
   - used to test whether routing follows subject/reference roles after query swap.

4. Layer input / full-attention output / block output
   - used to construct a correct-sample normative relation direction at each layer.

5. Approximate direct-logit contribution
   - pre-WO head output is mapped through that head's W_O slice and projected onto
     the final left-vs-right unembedding direction, with a sample-specific RMSNorm
     scaling approximation.

The script then learns "normal" directions/distributions ONLY from baseline-correct
samples and measures how every baseline-wrong sample deviates.

Main outputs
------------
head_function_sample_scores.csv
    One row per sample/head with layout, role-routing, composition, maintenance,
    opponent, and writer-related scores.

head_function_summary.csv
    Correct-vs-wrong aggregate comparison for every head, plus candidate function
    labels. Labels are candidates, not causal proof.

wrong_sample_top_mismatches.csv
    Top anomalous head/function pairs for each wrong sample relative to the correct
    distribution of the same head.

wrong_sample_stage_summary.csv
    The dominant candidate failure stage for each wrong sample.

function_failure_counts.csv
    Counts of dominant mismatch types across wrong samples.

Important limits
----------------
* This is a functional mismatch screen, not final causal attribution.
* A head is named only as a candidate when multiple diagnostics agree.
* Direct logit attribution is approximate because final normalization is nonlinear
  and later layers can transform earlier contributions.
* Role routing is inferred from prompt-last attention to identity-aligned object
  token spans; attention weight alone is not treated as sufficient evidence.
* The default experiment is left/right only because horizontal flip supplies a
  valid controlled visual intervention.

Required companion files in repository root
-------------------------------------------
analyze_coco_reasoning_vs_relay_factorial_v2.py
analyze_coco_ioi_backward_circuit_v1.py
analyze_coco_producer_qk_ov_v1.py
analyze_coco_receiver_qkv_v1.py
analyze_spatial_storage_transport_utilization_v3.py
analyze_coco_centroid_generation_step1_v4.py
analyze_coco_flip_attention_spatial_vectors_v1.py
coco_ioi_role_bundles_v1.json
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_VERSION = "coco-head-function-mismatch-v1"
CELL_NAMES = ("00", "01", "10", "11")
HORIZONTAL = ("left", "right")
EPS = 1e-12


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=("extract", "analyze", "all"), default="all")
    p.add_argument("--model", required=True)
    p.add_argument("--source-output-dir", required=True)
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", choices=("last", "mean"), default="last")
    p.add_argument(
        "--attention-object-state",
        choices=("last", "mean"),
        default="mean",
        help="How prompt-last attention to a multi-token object span is pooled.",
    )

    p.add_argument(
        "--baseline-generation-jsonl",
        default="",
        help="Optional original-query free-generation rows. Empty uses <head-output-dir>/baseline_generation.jsonl.",
    )
    p.add_argument(
        "--head-output-dir",
        default="output/coco_ioi_backward/qwen-3b_head_misrouting_pos7_neg5",
    )

    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument("--bundle-name", default="P_POS7")
    p.add_argument(
        "--scan-layers",
        default="19,20,21,22,23,26,27,28,29,30,31",
        help="Decoder layers whose query heads are scanned.",
    )
    p.add_argument(
        "--heads",
        default="",
        help="Optional explicit heads such as L19H13,L26H0. Empty scans all query heads in --scan-layers.",
    )
    p.add_argument(
        "--producer-pair-heads",
        default="bundle",
        help="Heads for object-pair layout analysis: bundle, all, or an explicit head list.",
    )
    p.add_argument("--probe-folds", type=int, default=5)
    p.add_argument("--bootstrap-repeats", type=int, default=1000)
    p.add_argument("--top-mismatches-per-sample", type=int, default=10)

    # Candidate naming thresholds. These do not alter raw scores.
    p.add_argument("--direction-acc-min", type=float, default=0.70)
    p.add_argument("--layout-share-min", type=float, default=0.50)
    p.add_argument("--composition-share-min", type=float, default=0.20)
    p.add_argument("--role-flip-min", type=float, default=0.60)
    p.add_argument("--candidate-percentile", type=float, default=0.80)
    p.add_argument("--writer-min-layer", type=int, default=28)

    p.add_argument("--sample-max-samples", type=int, default=0)
    p.add_argument("--include-sids-file", default="")
    p.add_argument("--exclude-sids-from", default="")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument(
        "--factorial-script",
        default="analyze_coco_reasoning_vs_relay_factorial_v2.py",
    )
    p.add_argument("--ioi-script", default="analyze_coco_ioi_backward_circuit_v1.py")
    p.add_argument("--producer-script", default="analyze_coco_producer_qk_ov_v1.py")
    p.add_argument("--receiver-script", default="analyze_coco_receiver_qkv_v1.py")
    p.add_argument("--v3-script", default="analyze_spatial_storage_transport_utilization_v3.py")
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--attention-helper", default="analyze_coco_flip_attention_spatial_vectors_v1.py")

    # Compatibility with imported helpers.
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def import_file(path: Path, module_name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def deduplicate_rows(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    table: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    order: List[Tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        if key not in table:
            order.append(key)
        table[key] = dict(row)
    return [table[key] for key in order]


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(array.mean()) if array.size else float("nan")


def safe_median(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(np.median(array)) if array.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(array.std(ddof=1)) if array.size >= 2 else float("nan")


def safe_divide(numerator: float, denominator: float, default: float = float("nan")) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return default
    if abs(float(denominator)) <= EPS:
        return default
    return float(numerator) / float(denominator)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "left of": "left",
        "right of": "right",
        "above": "above",
        "below": "below",
        "on": "above",
        "under": "below",
    }
    return aliases.get(text, text if text in {"left", "right", "above", "below"} else None)


def relation_sign(relation: str) -> float:
    relation = str(relation)
    if relation == "left":
        return 1.0
    if relation == "right":
        return -1.0
    raise ValueError(f"Horizontal relation required, got {relation!r}")


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    output: List[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            output.extend(range(int(a), int(b) + 1))
        else:
            output.append(int(item))
    output = sorted(set(output))
    invalid = [layer for layer in output if layer < 0 or layer >= n_layers]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; model has {n_layers} decoder layers")
    return output


def parse_head(value: Any) -> Tuple[int, int]:
    text = str(value).strip().upper()
    if text.startswith("L") and "H" in text:
        layer, head = text[1:].split("H", 1)
        return int(layer), int(head)
    if ":" in text:
        layer, head = text.split(":", 1)
        return int(layer), int(head)
    raise ValueError(f"Invalid head {value!r}")


def parse_heads(text: str) -> List[Tuple[int, int]]:
    return sorted(set(parse_head(item) for item in str(text).split(",") if item.strip()))


def head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def extract_sids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    output: set[int] = set()
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for row in read_jsonl(path):
            if row.get("sid") is not None:
                output.add(int(row["sid"]))
        return output
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and item.get("sid") is not None:
                    output.add(int(item["sid"]))
                else:
                    output.add(int(item))
            return output
        if isinstance(value, Mapping):
            for key in ("selected_sids", "sids", "sample_ids"):
                if key in value:
                    output.update(int(item) for item in value[key])
                    return output
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("sid") not in (None, ""):
                    output.add(int(row["sid"]))
        return output
    for token in path.read_text(encoding="utf-8").replace(",", " ").split():
        output.add(int(token))
    return output


def stratified_limit(rows: Sequence[Mapping[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("gt"))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    output: List[Dict[str, Any]] = []
    keys = sorted(groups)
    while len(output) < limit and any(groups[key] for key in keys):
        for key in keys:
            if groups[key] and len(output) < limit:
                output.append(groups[key].pop())
    return sorted(output, key=lambda row: int(row["sid"]))


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector)
    return vector / norm


def percentile_rank(values: Mapping[str, float], higher: bool = True) -> Dict[str, float]:
    finite = [(key, float(value)) for key, value in values.items() if math.isfinite(float(value))]
    if not finite:
        return {key: float("nan") for key in values}
    finite.sort(key=lambda item: item[1], reverse=not higher)
    n = len(finite)
    ranks: Dict[str, float] = {}
    for index, (key, _value) in enumerate(finite):
        ranks[key] = 1.0 if n == 1 else index / float(n - 1)
    return {key: ranks.get(key, float("nan")) for key in values}


def robust_center_scale(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(array.std(ddof=1)) if array.size >= 2 else 1.0
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return center, scale


def robust_z(value: float, center: float, scale: float) -> float:
    if not all(math.isfinite(float(v)) for v in (value, center, scale)) or abs(float(scale)) <= EPS:
        return float("nan")
    return (float(value) - float(center)) / float(scale)


def cohen_d(correct: Sequence[float], wrong: Sequence[float]) -> float:
    a = np.asarray([float(v) for v in correct if math.isfinite(float(v))], dtype=np.float64)
    b = np.asarray([float(v) for v in wrong if math.isfinite(float(v))], dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled_num = (a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)
    pooled_den = a.size + b.size - 2
    if pooled_den <= 0:
        return float("nan")
    pooled = math.sqrt(max(float(pooled_num / pooled_den), 0.0))
    return safe_divide(float(b.mean() - a.mean()), pooled)


def auc_rank(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    positives = [s for y, s in pairs if y == 1]
    negatives = [s for y, s in pairs if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(positives) * len(negatives))


def bootstrap_mean_ci(values: Sequence[float], repeats: int, seed: int) -> Tuple[float, float]:
    array = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if repeats <= 0:
        mean = float(array.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        samples[index] = rng.choice(array, size=array.size, replace=True).mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


# -----------------------------------------------------------------------------
# Head/capture specification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int
    head_dim: int
    start: int
    stop: int

    @property
    def name(self) -> str:
        return head_name(self.layer, self.head)


@dataclass
class ScanSpec:
    heads: List[HeadSpec]
    heads_by_layer: Dict[int, List[HeadSpec]]
    scan_layers: List[int]
    producer_pair_names: set[str]


def build_scan_spec(
    *,
    args: argparse.Namespace,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver: Any,
    bundle_heads: Sequence[Tuple[int, int]],
) -> ScanSpec:
    if str(args.heads).strip():
        selected = parse_heads(args.heads)
        scan_layers = sorted(set(layer for layer, _head in selected))
    else:
        scan_layers = parse_layer_spec(args.scan_layers, len(decoder_layers))
        selected = []
        for layer in scan_layers:
            attention = attention_helper.resolve_self_attention(decoder_layers[layer])
            shape = receiver.resolve_attention_shape(attention)
            selected.extend((layer, head) for head in range(int(shape.n_query_heads)))

    specs: List[HeadSpec] = []
    for layer, head in sorted(set(selected)):
        attention = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver.resolve_attention_shape(attention)
        n_heads = int(shape.n_query_heads)
        head_dim = int(shape.query_head_dim)
        if not 0 <= head < n_heads:
            raise ValueError(f"{head_name(layer, head)} outside n_query_heads={n_heads}")
        specs.append(HeadSpec(layer, head, head_dim, head * head_dim, (head + 1) * head_dim))

    pair_raw = str(args.producer_pair_heads).strip()
    if pair_raw.lower() == "bundle":
        producer_pair = {head_name(layer, head) for layer, head in bundle_heads}
    elif pair_raw.lower() == "all":
        producer_pair = {spec.name for spec in specs}
    else:
        producer_pair = {head_name(layer, head) for layer, head in parse_heads(pair_raw)}

    selected_names = {spec.name for spec in specs}
    missing = sorted(producer_pair - selected_names)
    if missing:
        raise ValueError(f"Producer-pair heads are not included in scan: {missing}")

    by_layer: Dict[int, List[HeadSpec]] = defaultdict(list)
    for spec in specs:
        by_layer[spec.layer].append(spec)
    return ScanSpec(
        heads=specs,
        heads_by_layer={layer: sorted(values, key=lambda item: item.head) for layer, values in by_layer.items()},
        scan_layers=sorted(by_layer),
        producer_pair_names=producer_pair,
    )


class HeadFunctionalCapture:
    def __init__(
        self,
        *,
        cell: Any,
        spec: ScanSpec,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        ioi: Any,
        final_norm: torch.nn.Module,
    ) -> None:
        self.cell = cell
        self.spec = spec
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.ioi = ioi
        self.final_norm = final_norm
        self.arrays: Dict[str, torch.Tensor] = {}
        self.handles: List[Any] = []
        self.events: Counter[str] = Counter()

    def _prompt(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"{name}: expected [1,S,H], got {tuple(tensor.shape)}")
        position = int(self.cell.prompt_last)
        if position >= int(tensor.shape[1]):
            raise RuntimeError(f"{name}: prompt position {position} outside S={tensor.shape[1]}")
        self.arrays[name] = tensor[0, position].detach().float().cpu()
        self.events[name] += 1

    def __enter__(self) -> "HeadFunctionalCapture":
        for layer_index in self.spec.scan_layers:
            layer = self.decoder_layers[layer_index]
            attention = self.attention_helper.resolve_self_attention(layer)
            o_proj = self.ioi.output_projection_module(attention)
            head_specs = list(self.spec.heads_by_layer[layer_index])

            def make_layer_pre(index: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{index} layer pre-hook missing hidden state")
                    self._prompt(f"L{index}_input", inputs[0])
                return hook

            self.handles.append(layer.register_forward_pre_hook(make_layer_pre(layer_index)))

            def make_o_pre(index: int, local_heads: List[HeadSpec]):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{index} o_proj pre-hook missing tensor")
                    tensor = inputs[0]
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError(f"L{index} o_proj input must be [1,S,D]")
                    p = int(self.cell.prompt_last)
                    a_pos = int(self.cell.a_positions[-1])
                    b_pos = int(self.cell.b_positions[-1])
                    for item in local_heads:
                        prompt = tensor[0, p, item.start:item.stop]
                        self.arrays[f"{item.name}__pre_prompt"] = prompt.detach().float().cpu()
                        self.events[f"{item.name}__pre_prompt"] += 1
                        if item.name in self.spec.producer_pair_names:
                            a = tensor[0, a_pos, item.start:item.stop]
                            b = tensor[0, b_pos, item.start:item.stop]
                            self.arrays[f"{item.name}__pair"] = (a - b).detach().float().cpu()
                            self.arrays[f"{item.name}__pair_mean"] = (0.5 * (a + b)).detach().float().cpu()
                            self.events[f"{item.name}__pair"] += 1
                return hook

            self.handles.append(o_proj.register_forward_pre_hook(make_o_pre(layer_index, head_specs)))

            def make_attn_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._prompt(f"L{index}_attn", tensor_from_output(output))
                    return output
                return hook

            self.handles.append(attention.register_forward_hook(make_attn_hook(layer_index)))

            def make_layer_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._prompt(f"L{index}_block", tensor_from_output(output))
                    return output
                return hook

            self.handles.append(layer.register_forward_hook(make_layer_hook(layer_index)))

        def final_pre(_module: Any, inputs: Tuple[Any, ...]) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("final norm pre-hook missing tensor")
            self._prompt("final_norm_input", inputs[0])

        def final_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            self._prompt("final_norm", tensor_from_output(output))
            return output

        self.handles.append(self.final_norm.register_forward_pre_hook(final_pre))
        self.handles.append(self.final_norm.register_forward_hook(final_hook))
        return self

    def finalize(self) -> Dict[str, np.ndarray]:
        required = ["final_norm_input", "final_norm"]
        for layer in self.spec.scan_layers:
            required.extend([f"L{layer}_input", f"L{layer}_attn", f"L{layer}_block"])
        for head in self.spec.heads:
            required.append(f"{head.name}__pre_prompt")
            if head.name in self.spec.producer_pair_names:
                required.append(f"{head.name}__pair")
        missing = [name for name in required if name not in self.arrays]
        if missing:
            raise RuntimeError(f"Missing capture arrays: {missing[:20]}")
        return {
            key: value.detach().float().cpu().numpy().astype(np.float32)
            for key, value in self.arrays.items()
        }

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"Unable to extract tensor from {type(output)!r}")


def replace_none_with_nan(value: Optional[float]) -> float:
    return float(value) if value is not None else float("nan")


def pool_attention(attention_row: torch.Tensor, positions: Sequence[int], mode: str) -> float:
    valid = [int(position) for position in positions if 0 <= int(position) < int(attention_row.shape[-1])]
    if not valid:
        return float("nan")
    if mode == "last":
        return float(attention_row[valid[-1]].detach().float().cpu().item())
    values = attention_row[torch.as_tensor(valid, device=attention_row.device, dtype=torch.long)]
    return float(values.detach().float().mean().cpu().item())


def extract_attention_scalars(
    *,
    outputs: Any,
    cell: Any,
    spec: ScanSpec,
    mode: str,
) -> Dict[str, np.ndarray]:
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        raise RuntimeError(
            "Model returned no attentions. Use --attn-impl eager and a Transformers version that supports output_attentions."
        )
    arrays: Dict[str, np.ndarray] = {}
    for layer in spec.scan_layers:
        if layer >= len(attentions) or attentions[layer] is None:
            raise RuntimeError(f"Missing attention tensor for layer {layer}")
        tensor = attentions[layer]
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"L{layer} attention expected [1,H,Q,K], got {tuple(tensor.shape)}")
        query_index = int(cell.prompt_last)
        if query_index >= int(tensor.shape[-2]):
            query_index = int(tensor.shape[-2]) - 1
        for head in spec.heads_by_layer[layer]:
            if head.head >= int(tensor.shape[1]):
                raise RuntimeError(f"{head.name}: attention has only H={tensor.shape[1]}")
            row = tensor[0, head.head, query_index]
            a = pool_attention(row, cell.a_positions, mode)
            b = pool_attention(row, cell.b_positions, mode)
            arrays[f"{head.name}__attn_a"] = np.asarray(a, dtype=np.float32)
            arrays[f"{head.name}__attn_b"] = np.asarray(b, dtype=np.float32)
            arrays[f"{head.name}__attn_object_mass"] = np.asarray(a + b, dtype=np.float32)
    return arrays


# -----------------------------------------------------------------------------
# Factor and normative-direction analysis
# -----------------------------------------------------------------------------


def factorial_terms(cell_vectors: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    z00 = np.asarray(cell_vectors["00"], dtype=np.float64)
    z01 = np.asarray(cell_vectors["01"], dtype=np.float64)
    z10 = np.asarray(cell_vectors["10"], dtype=np.float64)
    z11 = np.asarray(cell_vectors["11"], dtype=np.float64)
    return {
        "G": 0.25 * (z00 + z01 + z10 + z11),
        "V": 0.25 * (-z00 - z01 + z10 + z11),
        "Q": 0.25 * (-z00 + z01 - z10 + z11),
        "I": 0.25 * (z00 - z01 - z10 + z11),
    }


def factor_metrics(terms: Mapping[str, np.ndarray], prefix: str = "") -> Dict[str, float]:
    norms = {key: float(np.linalg.norm(np.asarray(terms[key], dtype=np.float64))) for key in ("V", "Q", "I")}
    energies = {key: value * value for key, value in norms.items()}
    total = float(sum(energies.values()))
    output: Dict[str, float] = {}
    for key in ("V", "Q", "I"):
        output[f"{prefix}{key}_norm"] = norms[key]
        output[f"{prefix}{key}_share"] = safe_divide(energies[key], total, 0.0)
    output[f"{prefix}factor_total_energy"] = total
    return output


def stable_fold(sid: int, folds: int, seed: int) -> int:
    # Stable and dependency-free; SID-level grouping is sufficient here because
    # every SID contributes all four cells together.
    value = (int(sid) * 1103515245 + int(seed) * 12345 + 0x9E3779B9) & 0x7FFFFFFF
    return int(value % int(folds))


def normative_direction_scores(
    *,
    vectors_by_sid: Mapping[int, np.ndarray],
    gt_by_sid: Mapping[int, str],
    correct_by_sid: Mapping[int, bool],
    fold_by_sid: Mapping[int, int],
    folds: int,
) -> Tuple[Dict[int, float], Dict[int, str], Dict[int, np.ndarray], float, float]:
    correct_sids = [sid for sid in vectors_by_sid if bool(correct_by_sid.get(sid, False))]
    if len(correct_sids) < 2:
        raise RuntimeError("At least two baseline-correct samples are required for normative directions")

    all_oriented = [relation_sign(gt_by_sid[sid]) * np.asarray(vectors_by_sid[sid], dtype=np.float64) for sid in correct_sids]
    all_direction = unit_vector(np.mean(np.stack(all_oriented, axis=0), axis=0))

    scores: Dict[int, float] = {}
    predictions: Dict[int, str] = {}
    directions: Dict[int, np.ndarray] = {}
    for sid, vector in vectors_by_sid.items():
        if bool(correct_by_sid.get(sid, False)):
            train = [
                other
                for other in correct_sids
                if other != sid and fold_by_sid[other] != fold_by_sid[sid]
            ]
            if not train:
                train = [other for other in correct_sids if other != sid]
            if not train:
                train = correct_sids
            oriented = [relation_sign(gt_by_sid[other]) * np.asarray(vectors_by_sid[other], dtype=np.float64) for other in train]
            direction = unit_vector(np.mean(np.stack(oriented, axis=0), axis=0))
        else:
            direction = all_direction
        raw = float(np.dot(np.asarray(vector, dtype=np.float64), direction))
        scores[sid] = relation_sign(gt_by_sid[sid]) * raw
        predictions[sid] = "left" if raw >= 0 else "right"
        directions[sid] = direction

    correct_accuracy = safe_mean(predictions[sid] == gt_by_sid[sid] for sid in correct_sids)
    wrong_sids = [sid for sid in vectors_by_sid if not bool(correct_by_sid.get(sid, False))]
    wrong_accuracy = safe_mean(predictions[sid] == gt_by_sid[sid] for sid in wrong_sids)
    return scores, predictions, directions, correct_accuracy, wrong_accuracy


def resolve_output_embedding(model: Any) -> torch.nn.Module:
    if hasattr(model, "get_output_embeddings"):
        module = model.get_output_embeddings()
        if module is not None and hasattr(module, "weight"):
            return module
    for path in ("lm_head", "model.lm_head", "language_model.lm_head"):
        current = model
        ok = True
        for name in path.split("."):
            if not hasattr(current, name):
                ok = False
                break
            current = getattr(current, name)
        if ok and hasattr(current, "weight"):
            return current
    raise RuntimeError("Unable to resolve output embedding / lm_head")


def relation_unembedding_direction(
    *,
    model: Any,
    relation_token_map: Mapping[str, Sequence[int]],
) -> np.ndarray:
    module = resolve_output_embedding(model)
    weight = module.weight.detach()
    left_ids = torch.as_tensor(
        [int(token) for token in relation_token_map["left"]],
        device=weight.device,
        dtype=torch.long,
    )
    right_ids = torch.as_tensor(
        [int(token) for token in relation_token_map["right"]],
        device=weight.device,
        dtype=torch.long,
    )
    left = weight.index_select(0, left_ids).float().mean(dim=0)
    right = weight.index_select(0, right_ids).float().mean(dim=0)
    return (left - right).cpu().numpy().astype(np.float64)


def final_norm_scale(final_norm: torch.nn.Module, final_input: np.ndarray) -> np.ndarray:
    x = np.asarray(final_input, dtype=np.float64)
    weight = getattr(final_norm, "weight", None)
    if weight is None:
        gain = np.ones_like(x)
    else:
        gain = weight.detach().float().cpu().numpy().astype(np.float64)
    eps = None
    for name in ("variance_epsilon", "eps", "epsilon"):
        if hasattr(final_norm, name):
            eps = float(getattr(final_norm, name))
            break
    if eps is None:
        eps = 1e-6
    rms = math.sqrt(float(np.mean(x * x)) + eps)
    return gain / max(rms, 1e-12)


def o_proj_weight_slices(
    *,
    scan_spec: ScanSpec,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    ioi: Any,
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for layer, heads in scan_spec.heads_by_layer.items():
        attention = attention_helper.resolve_self_attention(decoder_layers[layer])
        module = ioi.output_projection_module(attention)
        if not hasattr(module, "weight"):
            raise RuntimeError(f"L{layer} output projection has no weight")
        weight = module.weight.detach().float().cpu().numpy().astype(np.float64)
        for head in heads:
            if weight.shape[1] < head.stop:
                raise RuntimeError(f"{head.name}: W_O input width mismatch {weight.shape}")
            output[head.name] = np.asarray(weight[:, head.start:head.stop], dtype=np.float64)
    return output


def load_sample_arrays(vector_dir: Path, sids: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    output: Dict[int, Dict[str, np.ndarray]] = {}
    for sid in sids:
        path = vector_dir / f"sid_{sid:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            output[sid] = {key: np.asarray(data[key]) for key in data.files}
    return output


def get_cell_array(sample: Mapping[str, np.ndarray], cell: str, key: str) -> np.ndarray:
    full = f"{cell}__{key}"
    if full not in sample:
        raise KeyError(full)
    return np.asarray(sample[full], dtype=np.float64)


def summarize_metric(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Dict[str, float]:
    correct = [float(row[metric]) for row in rows if bool_value(row["baseline_correct"]) and math.isfinite(float(row[metric]))]
    wrong = [float(row[metric]) for row in rows if not bool_value(row["baseline_correct"]) and math.isfinite(float(row[metric]))]
    labels = [0 if bool_value(row["baseline_correct"]) else 1 for row in rows if math.isfinite(float(row[metric]))]
    scores = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
    return {
        f"{metric}_correct_mean": safe_mean(correct),
        f"{metric}_wrong_mean": safe_mean(wrong),
        f"{metric}_wrong_minus_correct": safe_mean(wrong) - safe_mean(correct),
        f"{metric}_cohen_d_wrong_minus_correct": cohen_d(correct, wrong),
        f"{metric}_wrong_AUROC": auc_rank(labels, scores),
    }


def candidate_labels(
    *,
    row: Mapping[str, Any],
    percentile_maps: Mapping[str, Mapping[str, float]],
    args: argparse.Namespace,
) -> List[str]:
    name = str(row["head"])
    labels: List[str] = []
    if bool_value(row.get("is_producer_pair_head", False)):
        if (
            float(row.get("pair_V_share_correct_mean", float("nan"))) >= float(args.layout_share_min)
            and float(row.get("layout_correct_cv_accuracy", float("nan"))) >= float(args.direction_acc_min)
        ):
            labels.append("layout_producer")

    role_rank = float(percentile_maps["role"].get(name, float("nan")))
    if (
        math.isfinite(role_rank)
        and role_rank >= float(args.candidate_percentile)
        and float(row.get("role_flip_consistency_correct_mean", 0.0)) >= float(args.role_flip_min)
    ):
        labels.append("role_routing")

    composition_rank = float(percentile_maps["composition"].get(name, float("nan")))
    if (
        math.isfinite(composition_rank)
        and composition_rank >= float(args.candidate_percentile)
        and float(row.get("pre_I_share_correct_mean", 0.0)) >= float(args.composition_share_min)
        and float(row.get("composition_correct_cv_accuracy", 0.0)) >= float(args.direction_acc_min)
    ):
        labels.append("relation_composition")

    maintenance_rank = float(percentile_maps["maintenance"].get(name, float("nan")))
    if (
        math.isfinite(maintenance_rank)
        and maintenance_rank >= float(args.candidate_percentile)
        and float(row.get("layer_input_relation_correct_cv_accuracy", 0.0)) >= float(args.direction_acc_min)
    ):
        labels.append("relation_maintenance_amplification")

    opponent_rank = float(percentile_maps["opponent"].get(name, float("nan")))
    if math.isfinite(opponent_rank) and opponent_rank >= float(args.candidate_percentile):
        labels.append("opponent_or_suppressor")

    writer_rank = float(percentile_maps["writer"].get(name, float("nan")))
    if (
        int(row["layer"]) >= int(args.writer_min_layer)
        and math.isfinite(writer_rank)
        and writer_rank >= float(args.candidate_percentile)
        and float(row.get("writer_dla_oriented_correct_mean", 0.0)) > 0.0
    ):
        labels.append("answer_writer")

    return labels or ["mixed_or_unclassified"]


def mismatch_for_row(
    *,
    row: Mapping[str, Any],
    labels: Sequence[str],
    stats: Mapping[Tuple[str, str], Tuple[float, float]],
) -> Tuple[str, float, str]:
    head = str(row["head"])
    candidates: List[Tuple[str, float, str]] = []

    def z(metric: str) -> float:
        center, scale = stats.get((head, metric), (float("nan"), float("nan")))
        return robust_z(float(row.get(metric, float("nan"))), center, scale)

    for label in labels:
        if label == "layout_producer":
            options = [
                (max(0.0, -z("layout_oriented_score")), "low_layout_direction"),
                (max(0.0, z("pair_Q_share")), "query_contamination"),
                (max(0.0, z("pair_I_share")), "unexpected_interaction_in_producer"),
            ]
        elif label == "role_routing":
            options = [
                (max(0.0, -z("role_oriented_score")), "role_swap_contrast_lost"),
                (max(0.0, -z("role_flip_consistency")), "attention_does_not_swap_roles"),
                (max(0.0, -z("object_attention_mass")), "object_attention_mass_low"),
            ]
        elif label == "relation_composition":
            options = [
                (max(0.0, -z("composition_oriented_score")), "interaction_direction_wrong_or_weak"),
                (max(0.0, -z("pre_I_share")), "interaction_share_low"),
            ]
        elif label == "relation_maintenance_amplification":
            options = [
                (max(0.0, -z("maintenance_oriented_score")), "relation_support_not_maintained"),
            ]
        elif label == "opponent_or_suppressor":
            options = [
                (max(0.0, z("opponent_pressure")), "excess_opponent_pressure"),
            ]
        elif label == "answer_writer":
            options = [
                (max(0.0, -z("writer_dla_oriented")), "GT_answer_write_missing_or_reversed"),
            ]
        else:
            options = [
                (max(0.0, -z("composition_oriented_score")), "mixed_composition_deviation"),
                (max(0.0, -z("maintenance_oriented_score")), "mixed_maintenance_deviation"),
                (max(0.0, -z("writer_dla_oriented")), "mixed_writer_deviation"),
                (max(0.0, z("opponent_pressure")), "mixed_opponent_deviation"),
            ]
        score, detail = max(options, key=lambda item: item[0])
        candidates.append((label, score, detail))

    return max(candidates, key=lambda item: item[1])


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def analyze(
    *,
    args: argparse.Namespace,
    factor: Any,
    scan_spec: ScanSpec,
    cell_rows: Sequence[Mapping[str, Any]],
    arrays_by_sid: Mapping[int, Mapping[str, np.ndarray]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    ioi: Any,
    final_norm: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
    model: Any,
    output_dir: Path,
) -> None:
    meta_by_sid = {int(row["sid"]): dict(row) for row in cell_rows}
    usable_sids = [
        sid
        for sid, row in sorted(meta_by_sid.items())
        if row.get("baseline_generation_correct") is not None
        and normalize_relation(row.get("gt")) in HORIZONTAL
    ]
    if not usable_sids:
        raise RuntimeError("No samples with baseline generation correctness")

    gt_by_sid = {sid: str(normalize_relation(meta_by_sid[sid]["gt"])) for sid in usable_sids}
    correct_by_sid = {sid: bool_value(meta_by_sid[sid]["baseline_generation_correct"]) for sid in usable_sids}
    fold_by_sid = {sid: stable_fold(sid, int(args.probe_folds), int(args.seed)) for sid in usable_sids}
    correct_count = sum(correct_by_sid.values())
    wrong_count = len(usable_sids) - correct_count
    if correct_count < max(2, int(args.probe_folds)):
        raise RuntimeError(f"Too few correct samples for normative cross-fitting: {correct_count}")
    if wrong_count < 1:
        raise RuntimeError("No baseline-wrong samples to compare")

    # Layer-level normative relation axes.
    layer_input_vectors: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    layer_block_vectors: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for sid in usable_sids:
        sample = arrays_by_sid[sid]
        for layer in scan_spec.scan_layers:
            layer_input_vectors[layer][sid] = factorial_terms(
                {cell: get_cell_array(sample, cell, f"L{layer}_input") for cell in CELL_NAMES}
            )["I"]
            layer_block_vectors[layer][sid] = factorial_terms(
                {cell: get_cell_array(sample, cell, f"L{layer}_block") for cell in CELL_NAMES}
            )["I"]

    layer_input_scores: Dict[int, Dict[int, float]] = {}
    layer_input_acc: Dict[int, float] = {}
    layer_block_directions: Dict[int, Dict[int, np.ndarray]] = {}
    layer_block_acc: Dict[int, float] = {}
    for layer in scan_spec.scan_layers:
        input_scores, _pred, _directions, correct_acc, _wrong_acc = normative_direction_scores(
            vectors_by_sid=layer_input_vectors[layer],
            gt_by_sid=gt_by_sid,
            correct_by_sid=correct_by_sid,
            fold_by_sid=fold_by_sid,
            folds=int(args.probe_folds),
        )
        layer_input_scores[layer] = input_scores
        layer_input_acc[layer] = correct_acc
        _scores, _pred, directions, correct_acc, _wrong_acc = normative_direction_scores(
            vectors_by_sid=layer_block_vectors[layer],
            gt_by_sid=gt_by_sid,
            correct_by_sid=correct_by_sid,
            fold_by_sid=fold_by_sid,
            folds=int(args.probe_folds),
        )
        layer_block_directions[layer] = directions
        layer_block_acc[layer] = correct_acc

    wo_slices = o_proj_weight_slices(
        scan_spec=scan_spec,
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        ioi=ioi,
    )
    unembed_lr = relation_unembedding_direction(model=model, relation_token_map=relation_token_map)

    # Per-head factorial vectors first, so normative correct-only directions can be learned.
    pair_terms: Dict[str, Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)
    pre_terms: Dict[str, Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)
    for head in scan_spec.heads:
        for sid in usable_sids:
            sample = arrays_by_sid[sid]
            pre_terms[head.name][sid] = factorial_terms(
                {cell: get_cell_array(sample, cell, f"{head.name}__pre_prompt") for cell in CELL_NAMES}
            )
            if head.name in scan_spec.producer_pair_names:
                pair_terms[head.name][sid] = factorial_terms(
                    {cell: get_cell_array(sample, cell, f"{head.name}__pair") for cell in CELL_NAMES}
                )

    layout_scores: Dict[str, Dict[int, float]] = {}
    layout_predictions: Dict[str, Dict[int, str]] = {}
    layout_correct_acc: Dict[str, float] = {}
    layout_wrong_acc: Dict[str, float] = {}
    composition_scores: Dict[str, Dict[int, float]] = {}
    composition_predictions: Dict[str, Dict[int, str]] = {}
    composition_correct_acc: Dict[str, float] = {}
    composition_wrong_acc: Dict[str, float] = {}

    for head in scan_spec.heads:
        if head.name in scan_spec.producer_pair_names:
            scores, preds, _dirs, c_acc, w_acc = normative_direction_scores(
                vectors_by_sid={sid: pair_terms[head.name][sid]["V"] for sid in usable_sids},
                gt_by_sid=gt_by_sid,
                correct_by_sid=correct_by_sid,
                fold_by_sid=fold_by_sid,
                folds=int(args.probe_folds),
            )
            layout_scores[head.name] = scores
            layout_predictions[head.name] = preds
            layout_correct_acc[head.name] = c_acc
            layout_wrong_acc[head.name] = w_acc

        scores, preds, _dirs, c_acc, w_acc = normative_direction_scores(
            vectors_by_sid={sid: pre_terms[head.name][sid]["I"] for sid in usable_sids},
            gt_by_sid=gt_by_sid,
            correct_by_sid=correct_by_sid,
            fold_by_sid=fold_by_sid,
            folds=int(args.probe_folds),
        )
        composition_scores[head.name] = scores
        composition_predictions[head.name] = preds
        composition_correct_acc[head.name] = c_acc
        composition_wrong_acc[head.name] = w_acc

    # Correct-sample role polarity: a role-routing head may follow subject (+) or reference (-).
    role_contrast_raw: Dict[str, Dict[int, float]] = defaultdict(dict)
    role_polarity: Dict[str, float] = {}
    for head in scan_spec.heads:
        for sid in usable_sids:
            sample = arrays_by_sid[sid]
            d: Dict[str, float] = {}
            for cell in CELL_NAMES:
                a = float(get_cell_array(sample, cell, f"{head.name}__attn_a"))
                b = float(get_cell_array(sample, cell, f"{head.name}__attn_b"))
                d[cell] = a - b
            role_contrast_raw[head.name][sid] = 0.25 * (d["00"] - d["01"] + d["10"] - d["11"])
        mean_correct = safe_mean(
            role_contrast_raw[head.name][sid]
            for sid in usable_sids
            if correct_by_sid[sid]
        )
        role_polarity[head.name] = 1.0 if not math.isfinite(mean_correct) or mean_correct >= 0 else -1.0

    sample_rows: List[Dict[str, Any]] = []
    for head in scan_spec.heads:
        wo = wo_slices[head.name]  # [hidden, head_dim]
        for sid in usable_sids:
            sample = arrays_by_sid[sid]
            meta = meta_by_sid[sid]
            correct = correct_by_sid[sid]
            gt = gt_by_sid[sid]
            pre = pre_terms[head.name][sid]
            metrics = factor_metrics(pre, prefix="pre_")

            pair_metrics: Dict[str, float] = {
                "pair_V_norm": float("nan"),
                "pair_Q_norm": float("nan"),
                "pair_I_norm": float("nan"),
                "pair_V_share": float("nan"),
                "pair_Q_share": float("nan"),
                "pair_I_share": float("nan"),
                "pair_factor_total_energy": float("nan"),
            }
            if head.name in scan_spec.producer_pair_names:
                pair_metrics = factor_metrics(pair_terms[head.name][sid], prefix="pair_")

            attn_a: Dict[str, float] = {}
            attn_b: Dict[str, float] = {}
            object_mass: Dict[str, float] = {}
            identity_diff: Dict[str, float] = {}
            for cell in CELL_NAMES:
                attn_a[cell] = float(get_cell_array(sample, cell, f"{head.name}__attn_a"))
                attn_b[cell] = float(get_cell_array(sample, cell, f"{head.name}__attn_b"))
                object_mass[cell] = float(get_cell_array(sample, cell, f"{head.name}__attn_object_mass"))
                identity_diff[cell] = attn_a[cell] - attn_b[cell]
            role_contrast = role_contrast_raw[head.name][sid]
            role_oriented = role_polarity[head.name] * role_contrast
            role_flip = 0.5 * (
                float(identity_diff["00"] * identity_diff["01"] < 0.0)
                + float(identity_diff["10"] * identity_diff["11"] < 0.0)
            )

            x00 = get_cell_array(sample, "00", f"{head.name}__pre_prompt")
            post_contribution = wo @ np.asarray(x00, dtype=np.float64)
            relation_direction = layer_block_directions[head.layer][sid]
            maintenance = relation_sign(gt) * float(np.dot(post_contribution, relation_direction))

            final_input = get_cell_array(sample, "00", "final_norm_input")
            scale = final_norm_scale(final_norm, final_input)
            effective_unembed = scale * unembed_lr
            writer_dla = relation_sign(gt) * float(np.dot(post_contribution, effective_unembed))
            opponent_pressure = max(0.0, -writer_dla)

            row = {
                "sid": sid,
                "gt": gt,
                "baseline_prediction": normalize_relation(meta.get("baseline_generation_prediction")),
                "baseline_correct": correct,
                "layer": head.layer,
                "head_index": head.head,
                "head": head.name,
                "is_producer_pair_head": head.name in scan_spec.producer_pair_names,
                "fold": fold_by_sid[sid],
                **metrics,
                **pair_metrics,
                "layout_oriented_score": layout_scores.get(head.name, {}).get(sid, float("nan")),
                "layout_prediction": layout_predictions.get(head.name, {}).get(sid),
                "layout_prediction_correct": (
                    layout_predictions.get(head.name, {}).get(sid) == gt
                    if head.name in layout_predictions
                    else None
                ),
                "role_swap_contrast_raw": role_contrast,
                "role_polarity_from_correct": role_polarity[head.name],
                "role_oriented_score": role_oriented,
                "role_flip_consistency": role_flip,
                "object_attention_mass": safe_mean(object_mass.values()),
                "attn_identity_diff_00": identity_diff["00"],
                "attn_identity_diff_01": identity_diff["01"],
                "attn_identity_diff_10": identity_diff["10"],
                "attn_identity_diff_11": identity_diff["11"],
                "composition_oriented_score": composition_scores[head.name][sid],
                "composition_prediction": composition_predictions[head.name][sid],
                "composition_prediction_correct": composition_predictions[head.name][sid] == gt,
                "layer_input_relation_oriented_score": layer_input_scores[head.layer][sid],
                "maintenance_oriented_score": maintenance,
                "writer_dla_oriented": writer_dla,
                "opponent_pressure": opponent_pressure,
            }
            sample_rows.append(row)

    write_csv(output_dir / "head_function_sample_scores.csv", sample_rows)

    rows_by_head: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        rows_by_head[str(row["head"])].append(row)

    metric_names = [
        "pair_V_share",
        "pair_Q_share",
        "pair_I_share",
        "layout_oriented_score",
        "role_oriented_score",
        "role_flip_consistency",
        "object_attention_mass",
        "pre_I_share",
        "composition_oriented_score",
        "maintenance_oriented_score",
        "writer_dla_oriented",
        "opponent_pressure",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for head in scan_spec.heads:
        rows = rows_by_head[head.name]
        summary: Dict[str, Any] = {
            "layer": head.layer,
            "head_index": head.head,
            "head": head.name,
            "N": len(rows),
            "correct_N": sum(bool_value(row["baseline_correct"]) for row in rows),
            "wrong_N": sum(not bool_value(row["baseline_correct"]) for row in rows),
            "is_producer_pair_head": head.name in scan_spec.producer_pair_names,
            "layout_correct_cv_accuracy": layout_correct_acc.get(head.name, float("nan")),
            "layout_wrong_accuracy": layout_wrong_acc.get(head.name, float("nan")),
            "composition_correct_cv_accuracy": composition_correct_acc[head.name],
            "composition_wrong_accuracy": composition_wrong_acc[head.name],
            "layer_input_relation_correct_cv_accuracy": layer_input_acc[head.layer],
            "layer_block_relation_correct_cv_accuracy": layer_block_acc[head.layer],
        }
        for metric in metric_names:
            summary.update(summarize_metric(rows, metric))
        summary_rows.append(summary)

    role_strength = {
        row["head"]: abs(float(row.get("role_oriented_score_correct_mean", float("nan"))))
        * max(float(row.get("object_attention_mass_correct_mean", 0.0)), 0.0)
        for row in summary_rows
    }
    composition_strength = {
        row["head"]: max(float(row.get("composition_oriented_score_correct_mean", float("nan"))), 0.0)
        * max(float(row.get("pre_I_share_correct_mean", 0.0)), 0.0)
        for row in summary_rows
    }
    maintenance_strength = {
        row["head"]: float(row.get("maintenance_oriented_score_correct_mean", float("nan")))
        for row in summary_rows
    }
    writer_strength = {
        row["head"]: float(row.get("writer_dla_oriented_correct_mean", float("nan")))
        for row in summary_rows
    }
    opponent_strength = {
        row["head"]: float(row.get("opponent_pressure_correct_mean", float("nan")))
        for row in summary_rows
    }
    percentile_maps = {
        "role": percentile_rank(role_strength),
        "composition": percentile_rank(composition_strength),
        "maintenance": percentile_rank(maintenance_strength),
        "writer": percentile_rank(writer_strength),
        "opponent": percentile_rank(opponent_strength),
    }

    label_by_head: Dict[str, List[str]] = {}
    for row in summary_rows:
        labels = candidate_labels(row=row, percentile_maps=percentile_maps, args=args)
        label_by_head[str(row["head"])] = labels
        row["candidate_functions"] = ";".join(labels)
        row["role_strength_percentile"] = percentile_maps["role"].get(str(row["head"]), float("nan"))
        row["composition_strength_percentile"] = percentile_maps["composition"].get(str(row["head"]), float("nan"))
        row["maintenance_strength_percentile"] = percentile_maps["maintenance"].get(str(row["head"]), float("nan"))
        row["writer_strength_percentile"] = percentile_maps["writer"].get(str(row["head"]), float("nan"))
        row["opponent_strength_percentile"] = percentile_maps["opponent"].get(str(row["head"]), float("nan"))

        # One concise priority score for correct-vs-wrong divergence.
        relevant_effects = []
        for metric in metric_names:
            value = float(row.get(f"{metric}_cohen_d_wrong_minus_correct", float("nan")))
            if math.isfinite(value):
                relevant_effects.append(abs(value))
        row["max_abs_correct_wrong_cohen_d"] = max(relevant_effects) if relevant_effects else float("nan")

    summary_rows.sort(
        key=lambda row: (
            -float(row.get("max_abs_correct_wrong_cohen_d", -1.0))
            if math.isfinite(float(row.get("max_abs_correct_wrong_cohen_d", float("nan"))))
            else 1.0,
            int(row["layer"]),
            int(row["head_index"]),
        )
    )
    write_csv(output_dir / "head_function_summary.csv", summary_rows)

    # Correct distributions for per-wrong-sample robust z-scores.
    stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for head, rows in rows_by_head.items():
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in rows
                if bool_value(row["baseline_correct"]) and math.isfinite(float(row[metric]))
            ]
            stats[(head, metric)] = robust_center_scale(values)

    wrong_top_rows: List[Dict[str, Any]] = []
    wrong_stage_rows: List[Dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    sample_rows_by_sid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        sample_rows_by_sid[int(row["sid"])].append(row)

    for sid in usable_sids:
        if correct_by_sid[sid]:
            continue
        ranked: List[Tuple[str, float, str, Dict[str, Any]]] = []
        for row in sample_rows_by_sid[sid]:
            function, score, detail = mismatch_for_row(
                row=row,
                labels=label_by_head[str(row["head"])],
                stats=stats,
            )
            ranked.append((function, score, detail, row))
        ranked.sort(key=lambda item: item[1], reverse=True)
        top = ranked[: max(1, int(args.top_mismatches_per_sample))]
        for rank, (function, score, detail, row) in enumerate(top, start=1):
            wrong_top_rows.append(
                {
                    "sid": sid,
                    "gt": gt_by_sid[sid],
                    "baseline_prediction": meta_by_sid[sid].get("baseline_generation_prediction"),
                    "rank": rank,
                    "head": row["head"],
                    "layer": row["layer"],
                    "head_index": row["head_index"],
                    "candidate_function": function,
                    "mismatch_score_robust_z": score,
                    "mismatch_detail": detail,
                    "all_candidate_functions": ";".join(label_by_head[str(row["head"])]),
                    "layout_oriented_score": row["layout_oriented_score"],
                    "role_oriented_score": row["role_oriented_score"],
                    "role_flip_consistency": row["role_flip_consistency"],
                    "composition_oriented_score": row["composition_oriented_score"],
                    "maintenance_oriented_score": row["maintenance_oriented_score"],
                    "writer_dla_oriented": row["writer_dla_oriented"],
                    "opponent_pressure": row["opponent_pressure"],
                }
            )
        dominant_function, dominant_score, dominant_detail, dominant_row = ranked[0]
        failure_counts[dominant_function] += 1
        wrong_stage_rows.append(
            {
                "sid": sid,
                "gt": gt_by_sid[sid],
                "baseline_prediction": meta_by_sid[sid].get("baseline_generation_prediction"),
                "dominant_function_mismatch": dominant_function,
                "dominant_head": dominant_row["head"],
                "dominant_mismatch_score_robust_z": dominant_score,
                "dominant_detail": dominant_detail,
                "second_function_mismatch": ranked[1][0] if len(ranked) > 1 else None,
                "second_head": ranked[1][3]["head"] if len(ranked) > 1 else None,
                "second_mismatch_score_robust_z": ranked[1][1] if len(ranked) > 1 else None,
            }
        )

    write_csv(output_dir / "wrong_sample_top_mismatches.csv", wrong_top_rows)
    write_csv(output_dir / "wrong_sample_stage_summary.csv", wrong_stage_rows)
    write_csv(
        output_dir / "function_failure_counts.csv",
        [
            {
                "candidate_function": function,
                "wrong_sample_count": count,
                "wrong_sample_fraction": count / float(wrong_count),
            }
            for function, count in failure_counts.most_common()
        ],
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "samples": len(usable_sids),
        "correct": correct_count,
        "wrong": wrong_count,
        "heads": len(scan_spec.heads),
        "scan_layers": scan_spec.scan_layers,
        "producer_pair_heads": sorted(scan_spec.producer_pair_names),
        "candidate_function_counts": dict(
            Counter(label for labels in label_by_head.values() for label in labels)
        ),
        "dominant_wrong_sample_failure_counts": dict(failure_counts),
        "limits": [
            "Candidate labels combine descriptive diagnostics; they are not causal proof.",
            "Normative directions and robust distributions are learned only from baseline-correct samples.",
            "Wrong-sample scores are compared to the same head's correct-sample distribution.",
            "Direct logit attribution uses a sample-specific RMSNorm scaling approximation and ignores later nonlinear transformations.",
            "Horizontal left/right only.",
        ],
    }
    write_json(output_dir / "summary.json", summary)

    print("\n" + "=" * 148)
    print("COCO PER-HEAD FUNCTIONAL MISMATCH RESULT")
    print("=" * 148)
    print(
        f"Samples={len(usable_sids)} | correct={correct_count} | wrong={wrong_count} | "
        f"heads={len(scan_spec.heads)} | layers={','.join(map(str, scan_spec.scan_layers))}"
    )
    print("\nCANDIDATE HEAD FUNCTIONS")
    for name, count in Counter(label for labels in label_by_head.values() for label in labels).most_common():
        print(f"{name:42s} {count:4d}")
    print("\nDOMINANT MISMATCH ACROSS WRONG SAMPLES")
    for name, count in failure_counts.most_common():
        print(f"{name:42s} {count:4d}  ({count / float(wrong_count):.3f})")
    print("\nTOP CORRECT-vs-WRONG HEAD DIFFERENCES")
    for row in summary_rows[:20]:
        print(
            f"{str(row['head']):8s} functions={str(row['candidate_functions']):48s} "
            f"max|d|={float(row['max_abs_correct_wrong_cohen_d']):8.3f} "
            f"Iacc(C/W)={float(row['composition_correct_cv_accuracy']):.3f}/"
            f"{float(row['composition_wrong_accuracy']):.3f}"
        )
    print(f"\nSaved outputs to {output_dir}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.probe_folds < 2:
        raise ValueError("--probe-folds must be >=2")
    if not 0.0 <= args.candidate_percentile <= 1.0:
        raise ValueError("--candidate-percentile must be in [0,1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    factor = import_file(Path(args.factorial_script), "head_function_factorial")
    ioi = import_file(Path(args.ioi_script), "head_function_ioi")
    producer = import_file(Path(args.producer_script), "head_function_producer")
    receiver = import_file(Path(args.receiver_script), "head_function_receiver")
    v3 = import_file(Path(args.v3_script), "head_function_v3")
    base = import_file(Path(args.base_script), "head_function_base")
    attention_helper = import_file(Path(args.attention_helper), "head_function_attention")

    source_config, source_rows = ioi.load_source_rows(args)
    baseline_path = factor.resolve_baseline_path(args)
    baseline_by_sid = factor.load_baseline_rows(baseline_path)

    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(str(args.include_sids_file).strip()))
    excluded: set[int] = set()
    for raw in str(args.exclude_sids_from).split(","):
        item = raw.strip()
        if item:
            excluded.update(extract_sids(Path(item)))

    selected_rows: List[Dict[str, Any]] = []
    for row in source_rows:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        if gt not in HORIZONTAL:
            continue
        if sid in excluded or (included is not None and sid not in included):
            continue
        selected_rows.append({**dict(row), "sid": sid, "gt": gt})
    selected_rows = stratified_limit(selected_rows, int(args.sample_max_samples), int(args.seed))
    if not selected_rows:
        raise RuntimeError("No horizontal COCO samples selected")
    selected_sids = [int(row["sid"]) for row in selected_rows]

    model = None
    processor = None
    try:
        (
            model,
            processor,
            spec_model,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer.load_model_bundle(args=args, base=base)
        final_norm = factor.resolve_final_norm(model, decoder_path)
        bundle_heads = factor.load_bundle(Path(args.bundle_json), args.bundle_name)
        scan_spec = build_scan_spec(
            args=args,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver=receiver,
            bundle_heads=bundle_heads,
        )
        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec_model.repo_id,
            "dataset": "coco_two_horizontal_only",
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "baseline_generation_jsonl": str(baseline_path),
            "decoder_path": decoder_path,
            "n_layers": len(decoder_layers),
            "scan_layers": scan_spec.scan_layers,
            "heads": [head.name for head in scan_spec.heads],
            "producer_pair_heads": sorted(scan_spec.producer_pair_names),
            "four_cells": {
                "00": "original image + original query",
                "01": "original image + role-swapped query",
                "10": "horizontal-flipped image + original query",
                "11": "horizontal-flipped image + role-swapped query",
            },
            "selected_samples": len(selected_rows),
            "selected_sids": selected_sids,
            "phase": args.phase,
            "audit": audit,
            "limits": [
                "Left/right only.",
                "Per-head function labels are candidate labels, not final causal names.",
                "Correct samples define the normative directions and robust distributions.",
            ],
        }
        write_json(output_dir / "config.json", config)

        cells_path = output_dir / "factorial_head_cells.jsonl"
        existing = deduplicate_rows(read_jsonl(cells_path), ("sid",)) if args.resume else []
        done = {
            int(row["sid"])
            for row in existing
            if (vector_dir / f"sid_{int(row['sid']):06d}.npz").exists()
        }

        if args.phase in ("extract", "all"):
            pending = [row for row in selected_rows if int(row["sid"]) not in done]
            print(
                f"Per-head four-cell extraction: samples={len(selected_rows)} pending={len(pending)} "
                f"heads={len(scan_spec.heads)} forwards={len(pending) * 4}",
                flush=True,
            )
            for index, source_row in enumerate(
                tqdm(pending, desc=f"head-function-extract:{args.model}"), start=1
            ):
                sample = None
                try:
                    sid = int(source_row["sid"])
                    sample = factor.prepare_four_cells(
                        args=args,
                        source_row=source_row,
                        records_by_sid=records_by_sid,
                        prompt_rows=prompt_rows,
                        base=base,
                        v3=v3,
                        receiver=receiver,
                        processor=processor,
                        device=torch.device(args.device),
                    )
                    arrays: Dict[str, np.ndarray] = {}
                    cells_meta: Dict[str, Dict[str, Any]] = {}
                    for cell_name in CELL_NAMES:
                        cell = sample.cells[cell_name]
                        capture = HeadFunctionalCapture(
                            cell=cell,
                            spec=scan_spec,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            ioi=ioi,
                            final_norm=final_norm,
                        )
                        with capture:
                            outputs = model(
                                **dict(cell.batch),
                                use_cache=False,
                                return_dict=True,
                                output_attentions=True,
                            )
                        captured = capture.finalize()
                        attention_arrays = extract_attention_scalars(
                            outputs=outputs,
                            cell=cell,
                            spec=scan_spec,
                            mode=str(args.attention_object_state),
                        )
                        scores = factor.relation_score_map(
                            base.relation_scores(outputs.logits[0, -1], relation_token_map, gt=None)
                        )
                        prediction, top_margin = factor.score_prediction(scores)
                        for key, value in {**captured, **attention_arrays}.items():
                            arrays[f"{cell_name}__{key}"] = np.asarray(value)
                        arrays[f"{cell_name}__relation_logits"] = np.asarray(
                            [scores["left"], scores["right"]], dtype=np.float32
                        )
                        cells_meta[cell_name] = {
                            "visual_flip": int(cell.visual_flip),
                            "query_swap": int(cell.query_swap),
                            "expected_relation": cell.expected_relation,
                            "closed_scores": scores,
                            "closed_prediction": prediction,
                            "closed_top_margin": top_margin,
                            "closed_correct": bool(prediction == cell.expected_relation),
                            "a_positions": [int(v) for v in cell.a_positions],
                            "b_positions": [int(v) for v in cell.b_positions],
                            "prompt_last": int(cell.prompt_last),
                        }
                        del outputs
                    vector_path = vector_dir / f"sid_{sid:06d}.npz"
                    np.savez_compressed(vector_path, **arrays)
                    baseline = baseline_by_sid.get(sid, {})
                    row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": sample.gt,
                        "subject": sample.subject,
                        "reference": sample.reference,
                        "baseline_generation_prediction": normalize_relation(baseline.get("prediction")),
                        "baseline_generation_correct": (
                            bool(baseline.get("correct", False)) if baseline else None
                        ),
                        "cells": cells_meta,
                        "vector_file": str(vector_path),
                    }
                    append_jsonl(cells_path, row)
                    if args.print_every > 0 and index % args.print_every == 0:
                        pattern = "/".join(cells_meta[cell]["closed_prediction"] for cell in CELL_NAMES)
                        print(
                            f"[extract {index}/{len(pending)} sid={sid}] "
                            f"baseline={row['baseline_generation_prediction']} correct={row['baseline_generation_correct']} "
                            f"closed={pattern}",
                            flush=True,
                        )
                except Exception as exc:
                    error = {
                        "phase": "extract",
                        "sid": int(source_row.get("sid", -1)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(errors_path, error)
                    print(
                        f"[ERROR extract sid={source_row.get('sid')}] {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                finally:
                    factor.release_four_cells(sample, receiver)
                    gc.collect()
                    if (
                        args.device.startswith("cuda")
                        and args.empty_cache_every > 0
                        and index % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

        if args.phase == "extract":
            complete = deduplicate_rows(read_jsonl(cells_path), ("sid",))
            complete_sids = {int(row["sid"]) for row in complete}
            missing = [sid for sid in selected_sids if sid not in complete_sids]
            print(
                f"Extraction completed: built={len(complete_sids)}/{len(selected_sids)} "
                f"failed={len(missing)} output={output_dir}",
                flush=True,
            )
            if missing:
                raise RuntimeError(f"Incomplete extraction SIDs {missing[:20]}; inspect {errors_path}")
            return

        cell_rows = deduplicate_rows(read_jsonl(cells_path), ("sid",))
        available_sids = [
            int(row["sid"])
            for row in cell_rows
            if (vector_dir / f"sid_{int(row['sid']):06d}.npz").exists()
        ]
        expected = set(selected_sids)
        missing = sorted(expected - set(available_sids))
        if missing:
            raise RuntimeError(f"Missing extracted SIDs {missing[:20]}; run --phase extract first")
        arrays_by_sid = load_sample_arrays(vector_dir, selected_sids)
        analyze(
            args=args,
            factor=factor,
            scan_spec=scan_spec,
            cell_rows=cell_rows,
            arrays_by_sid=arrays_by_sid,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            ioi=ioi,
            final_norm=final_norm,
            relation_token_map=relation_token_map,
            model=model,
            output_dir=output_dir,
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
