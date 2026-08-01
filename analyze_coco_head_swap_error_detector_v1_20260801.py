#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO two-object per-head object-swap error detector.

Research question
-----------------
For every COCO two-object sample, run:

    original:  A relative to B?
    swapped:   B relative to A?

At every selected decoder attention head, capture the pre-W_O head output at
identity-aligned object-token positions A and B.  The script then measures how
well the head respects the object/query swap and trains an out-of-fold detector
for free-generation correctness.

The four canonical relations are:

    left <-> right
    above/on <-> below/under

The detector is evaluated with repeated stratified out-of-fold prediction.  All
supervised feature selection, relation-conditioned correct-sample prototypes,
and relation-conditioned normalization are fitted only on the outer training
fold.  Correct training samples use leave-one-out prototypes/statistics when
constructing their own training features.

Feature groups
--------------
confidence
    Closed-set score confidence/equivariance from the original and swapped runs.
    Does not use GT margins.

swap_raw
    Per-head object-swap scalar features that do not use GT prototypes:
    identity-aligned pair cosine/residual, identity-vs-role token matching,
    stable/mismatch energy share, and query-induced token shift.

confidence_plus_swap_raw
    Concatenation of confidence and swap_raw.

gt_conditioned_head
    GT-relation-conditioned diagnostic features.  For each outer fold and each
    relation/head, correct training samples define normal object-pair prototypes
    and normal scalar distributions.  Test samples are compared only with the
    corresponding GT relation's training prototypes.

confidence_plus_gt_conditioned_head
    Concatenation of confidence and GT-conditioned head features.

Important interpretation limits
--------------------------------
* The prediction label is baseline free-generation correctness.
* GT-conditioned models are mechanistic diagnostics, not deployable detectors.
* A detector above chance shows a stable association, not causal attribution.
* The script compares the same head under original/swapped queries; it does not
  patch one sample's activation into another sample.
* Head outputs are captured before W_O at identity-aligned object-token spans.

Main outputs
------------
swap_cells.jsonl
vectors/sid_XXXXXX.npz
    Original/swapped per-head A/B vectors.

sample_swap_scalar_features.csv
    Descriptive per-sample/per-head swap features.

oof_predictions.csv
model_performance.csv
relation_performance.csv
repeat_performance.csv
    Repeated out-of-fold detector results.

selected_feature_stability.csv
head_importance_summary.csv
    Fold-stable selected features and heads.

config.json, summary.json, errors.jsonl

Required companion files in repository root
-------------------------------------------
analyze_coco_reasoning_vs_relay_factorial_v2.py
analyze_coco_ioi_backward_circuit_v1.py
analyze_coco_producer_qk_ov_v1.py
analyze_coco_receiver_qkv_v1.py
analyze_spatial_storage_transport_utilization_v3.py
analyze_coco_centroid_generation_step1_v4.py
analyze_coco_flip_attention_spatial_vectors_v1.py
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
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required: {exc}")


SCRIPT_VERSION = "coco-head-swap-error-detector-v1-20260801"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-8
RAW_METRICS = (
    "pair_cos",
    "pair_residual",
    "pair_norm_logratio",
    "identity_token_cos",
    "role_token_cos",
    "identity_role_gap",
    "identity_token_residual",
    "role_token_residual",
    "query_shift",
    "stable_share",
    "mismatch_share",
    "mean_state_cos",
    "mean_state_residual",
)
NORMATIVE_METRICS = (
    "pair_cos",
    "pair_residual",
    "identity_role_gap",
    "query_shift",
    "stable_share",
    "mean_state_residual",
)
MODEL_GROUPS = (
    "confidence",
    "swap_raw",
    "confidence_plus_swap_raw",
    "gt_conditioned_head",
    "confidence_plus_gt_conditioned_head",
)


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
        "--capture-pool",
        choices=("last", "mean"),
        default="mean",
        help="How captured head outputs are pooled over a multi-token object span.",
    )
    p.add_argument(
        "--baseline-generation-jsonl",
        default="",
        help="Empty uses <head-output-dir>/baseline_generation.jsonl.",
    )
    p.add_argument(
        "--head-output-dir",
        default="output/coco_ioi_backward/qwen-3b_head_misrouting_pos7_neg5",
        help="Used to infer baseline_generation.jsonl when the explicit path is empty.",
    )

    p.add_argument(
        "--scan-layers",
        default="all",
        help="all, comma list, or inclusive ranges such as 0-31,19,21-23.",
    )
    p.add_argument(
        "--heads",
        default="",
        help="Optional explicit heads such as L19H13,L26H0. Overrides --scan-layers.",
    )
    p.add_argument(
        "--save-dtype",
        choices=("float16", "float32"),
        default="float16",
    )

    p.add_argument("--sample-max-samples", type=int, default=0)
    p.add_argument("--include-sids-file", default="")
    p.add_argument("--exclude-sids-from", default="")

    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=5)
    p.add_argument(
        "--top-k-features",
        type=int,
        default=512,
        help="Fold-local ANOVA feature selection; 0 keeps all features.",
    )
    p.add_argument("--logreg-c", type=float, default=0.10)
    p.add_argument("--l1-ratio", type=float, default=0.50)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--review-fraction", type=float, default=0.20)
    p.add_argument(
        "--model-groups",
        default=",".join(MODEL_GROUPS),
        help="Comma-separated subset of supported feature groups.",
    )
    p.add_argument(
        "--write-long-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample/per-head descriptive scalar CSV.",
    )

    p.add_argument("--seed", type=int, default=83)
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

    # Compatibility with imported repository helpers.
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
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def deduplicate_rows(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> List[Dict[str, Any]]:
    output: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        output[tuple(row.get(key) for key in keys)] = dict(row)
    return list(output.values())


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ")
    aliases = {
        "left of": "left",
        "right of": "right",
        "on": "above",
        "over": "above",
        "above": "above",
        "under": "below",
        "below": "below",
    }
    return aliases.get(text, text if text in RELATIONS else None)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_head(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"[Ll](\d+)[Hh](\d+)", str(value).strip())
    if not match:
        raise ValueError(f"Invalid head specification: {value!r}")
    return int(match.group(1)), int(match.group(2))


def parse_heads(text: str) -> List[Tuple[int, int]]:
    return [parse_head(item) for item in str(text).split(",") if item.strip()]


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    raw = str(text).strip().lower()
    if raw in {"all", "*"}:
        return list(range(n_layers))
    output: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                start, stop = stop, start
            output.update(range(start, stop + 1))
        else:
            output.add(int(token))
    invalid = sorted(layer for layer in output if layer < 0 or layer >= n_layers)
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; model has {n_layers} decoder layers")
    return sorted(output)


def head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def extract_sids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    output: set[int] = set()
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        if suffix == ".jsonl":
            values: Any = read_jsonl(path)
        else:
            values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, Mapping):
            values = values.get("selected_sids", values.get("sids", values.get("rows", [])))
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                if isinstance(value, Mapping):
                    for key in ("sid", "sample_id", "id"):
                        if key in value:
                            output.add(int(value[key]))
                            break
                else:
                    output.add(int(value))
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("sid", "sample_id", "id"):
                    if row.get(key) not in (None, ""):
                        output.add(int(row[key]))
                        break
    else:
        for token in re.split(r"[\s,]+", path.read_text(encoding="utf-8")):
            if token.strip():
                output.add(int(token))
    if not output:
        raise RuntimeError(f"No SIDs found in {path}")
    return output


def stratified_limit(
    rows: Sequence[Mapping[str, Any]], limit: int, seed: int
) -> List[Dict[str, Any]]:
    values = [dict(row) for row in rows]
    if limit <= 0 or limit >= len(values):
        return values
    groups: Dict[Tuple[str, bool], List[Dict[str, Any]]] = defaultdict(list)
    for row in values:
        groups[(str(row["gt"]), bool(row["baseline_correct"]))].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: List[Dict[str, Any]] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        group = groups[key]
        if group:
            selected.append(group.pop())
        keys = [item for item in keys if groups[item]]
        cursor += 1
    return sorted(selected, key=lambda row: int(row["sid"]))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if math.isfinite(float(b)) and abs(float(b)) > EPS else float(default)


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def vector_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denominator, EPS)


def normalized_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1), EPS
    )


def row_unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def softmax_np(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.maximum(e.sum(), EPS)


# -----------------------------------------------------------------------------
# Head capture
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int
    head_dim: int
    start: int
    stop: int
    global_index: int

    @property
    def name(self) -> str:
        return head_name(self.layer, self.head)


@dataclass
class ScanSpec:
    heads: List[HeadSpec]
    heads_by_layer: Dict[int, List[HeadSpec]]
    scan_layers: List[int]
    head_dim: int


def build_scan_spec(
    *,
    args: argparse.Namespace,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver: Any,
) -> ScanSpec:
    if str(args.heads).strip():
        selected = sorted(set(parse_heads(args.heads)))
    else:
        selected = []
        for layer in parse_layer_spec(args.scan_layers, len(decoder_layers)):
            attention = attention_helper.resolve_self_attention(decoder_layers[layer])
            shape = receiver.resolve_attention_shape(attention)
            selected.extend((layer, head) for head in range(int(shape.n_query_heads)))

    specs: List[HeadSpec] = []
    dimensions: set[int] = set()
    for global_index, (layer, head) in enumerate(selected):
        attention = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver.resolve_attention_shape(attention)
        n_heads = int(shape.n_query_heads)
        head_dim = int(shape.query_head_dim)
        if not 0 <= head < n_heads:
            raise ValueError(f"{head_name(layer, head)} outside n_query_heads={n_heads}")
        dimensions.add(head_dim)
        specs.append(
            HeadSpec(
                layer=layer,
                head=head,
                head_dim=head_dim,
                start=head * head_dim,
                stop=(head + 1) * head_dim,
                global_index=global_index,
            )
        )
    if not specs:
        raise RuntimeError("No heads selected")
    if len(dimensions) != 1:
        raise RuntimeError(
            f"Selected heads have non-uniform dimensions {sorted(dimensions)}; "
            "this script expects a uniform decoder head dimension."
        )
    by_layer: Dict[int, List[HeadSpec]] = defaultdict(list)
    for item in specs:
        by_layer[item.layer].append(item)
    return ScanSpec(
        heads=specs,
        heads_by_layer={layer: sorted(items, key=lambda item: item.head) for layer, items in by_layer.items()},
        scan_layers=sorted(by_layer),
        head_dim=next(iter(dimensions)),
    )


def pool_positions(tensor: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    valid = [int(position) for position in positions if 0 <= int(position) < int(tensor.shape[1])]
    if not valid:
        raise RuntimeError("No valid object positions for capture")
    if mode == "last":
        return tensor[0, valid[-1]]
    index = torch.as_tensor(valid, dtype=torch.long, device=tensor.device)
    return tensor[0].index_select(0, index).mean(dim=0)


class HeadObjectCapture:
    def __init__(
        self,
        *,
        spec: ScanSpec,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        ioi: Any,
        a_positions: Sequence[int],
        b_positions: Sequence[int],
        pool_mode: str,
    ) -> None:
        self.spec = spec
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.ioi = ioi
        self.a_positions = list(map(int, a_positions))
        self.b_positions = list(map(int, b_positions))
        self.pool_mode = str(pool_mode)
        self.output = torch.empty(
            (len(spec.heads), 2, spec.head_dim), dtype=torch.float32, device="cpu"
        )
        self.seen: set[int] = set()
        self.handles: List[Any] = []

    def __enter__(self) -> "HeadObjectCapture":
        for layer_index in self.spec.scan_layers:
            attention = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer_index]
            )
            o_proj = self.ioi.output_projection_module(attention)
            local_heads = list(self.spec.heads_by_layer[layer_index])

            def make_hook(items: List[HeadSpec], layer: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{layer} o_proj pre-hook missing tensor")
                    tensor = inputs[0]
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError(
                            f"L{layer} o_proj input must be [1,S,D], got {tuple(tensor.shape)}"
                        )
                    a_full = pool_positions(tensor, self.a_positions, self.pool_mode)
                    b_full = pool_positions(tensor, self.b_positions, self.pool_mode)
                    for item in items:
                        self.output[item.global_index, 0] = (
                            a_full[item.start:item.stop].detach().float().cpu()
                        )
                        self.output[item.global_index, 1] = (
                            b_full[item.start:item.stop].detach().float().cpu()
                        )
                        self.seen.add(item.global_index)
                return hook

            self.handles.append(o_proj.register_forward_pre_hook(make_hook(local_heads, layer_index)))
        return self

    def finalize(self) -> np.ndarray:
        missing = [item.name for item in self.spec.heads if item.global_index not in self.seen]
        if missing:
            raise RuntimeError(f"Missing head captures: {missing[:20]}")
        return self.output.numpy().astype(np.float32)

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# -----------------------------------------------------------------------------
# Extraction helpers
# -----------------------------------------------------------------------------


def relation_score_dict(
    *, outputs: Any, relation_token_map: Mapping[str, Sequence[int]], base: Any, factor: Any
) -> Dict[str, float]:
    raw = base.relation_scores(outputs.logits[0, -1], relation_token_map, gt=None)
    scores = factor.relation_score_map(raw)
    return {relation: float(scores[relation]) for relation in RELATIONS}


def prediction_and_margin(scores: Mapping[str, float]) -> Tuple[str, float]:
    ranked = sorted(((float(value), relation) for relation, value in scores.items()), reverse=True)
    prediction = ranked[0][1]
    margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else float("nan")
    return prediction, float(margin)


def confidence_features_from_scores(
    *,
    original: Mapping[str, float],
    swapped: Mapping[str, float],
    original_prediction: str,
    swapped_prediction: str,
) -> Tuple[np.ndarray, List[str]]:
    orig_values = [float(original[relation]) for relation in RELATIONS]
    swap_values = [float(swapped[relation]) for relation in RELATIONS]
    po = softmax_np(orig_values)
    ps = softmax_np(swap_values)
    orig_sorted = np.sort(np.asarray(orig_values, dtype=np.float64))[::-1]
    swap_sorted = np.sort(np.asarray(swap_values, dtype=np.float64))[::-1]
    orig_entropy = float(-np.sum(po * np.log(np.maximum(po, EPS))))
    swap_entropy = float(-np.sum(ps * np.log(np.maximum(ps, EPS))))
    pair_opposite = float(swapped_prediction == OPPOSITE.get(original_prediction))
    values = np.asarray(
        [
            orig_sorted[0] - orig_sorted[1],
            float(po.max()),
            orig_entropy,
            swap_sorted[0] - swap_sorted[1],
            float(ps.max()),
            swap_entropy,
            pair_opposite,
            abs((orig_sorted[0] - orig_sorted[1]) - (swap_sorted[0] - swap_sorted[1])),
        ],
        dtype=np.float64,
    )
    names = [
        "confidence::orig_top_margin",
        "confidence::orig_max_probability",
        "confidence::orig_entropy",
        "confidence::swap_top_margin",
        "confidence::swap_max_probability",
        "confidence::swap_entropy",
        "confidence::prediction_pair_opposite",
        "confidence::top_margin_gap",
    ]
    return values, names


# -----------------------------------------------------------------------------
# Vector loading and scalar swap features
# -----------------------------------------------------------------------------


def load_vectors(
    *, vector_dir: Path, sids: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    original: List[np.ndarray] = []
    swapped: List[np.ndarray] = []
    expected_shape: Optional[Tuple[int, int, int]] = None
    for sid in sids:
        path = vector_dir / f"sid_{int(sid):06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            orig = np.asarray(data["original_heads"], dtype=np.float32)
            swap = np.asarray(data["swapped_heads"], dtype=np.float32)
        if orig.shape != swap.shape:
            raise RuntimeError(f"SID {sid}: original/swap shape mismatch {orig.shape} vs {swap.shape}")
        if expected_shape is None:
            expected_shape = tuple(orig.shape)
        if tuple(orig.shape) != expected_shape:
            raise RuntimeError(f"SID {sid}: expected shape {expected_shape}, got {orig.shape}")
        original.append(orig)
        swapped.append(swap)
    return np.stack(original, axis=0), np.stack(swapped, axis=0)


def compute_swap_scalar_features(
    original_heads: np.ndarray,
    swapped_heads: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Return [N,H,M] scalar features from identity-aligned A/B vectors."""
    a0 = np.asarray(original_heads[:, :, 0, :], dtype=np.float64)
    b0 = np.asarray(original_heads[:, :, 1, :], dtype=np.float64)
    a1 = np.asarray(swapped_heads[:, :, 0, :], dtype=np.float64)
    b1 = np.asarray(swapped_heads[:, :, 1, :], dtype=np.float64)

    d0 = a0 - b0
    d1 = a1 - b1
    m0 = 0.5 * (a0 + b0)
    m1 = 0.5 * (a1 + b1)
    stable = 0.5 * (d0 + d1)
    mismatch = 0.5 * (d0 - d1)

    stable_energy = np.sum(stable * stable, axis=-1)
    mismatch_energy = np.sum(mismatch * mismatch, axis=-1)
    factor_energy = np.maximum(stable_energy + mismatch_energy, EPS)

    identity_token_cos = 0.5 * (vector_cosine(a0, a1) + vector_cosine(b0, b1))
    role_token_cos = 0.5 * (vector_cosine(a0, b1) + vector_cosine(b0, a1))
    identity_token_residual = 0.5 * (
        normalized_residual(a0, a1) + normalized_residual(b0, b1)
    )
    role_token_residual = 0.5 * (
        normalized_residual(a0, b1) + normalized_residual(b0, a1)
    )
    query_shift = 0.5 * (
        np.linalg.norm(a0 - a1, axis=-1) / np.maximum(np.linalg.norm(a0, axis=-1) + np.linalg.norm(a1, axis=-1), EPS)
        + np.linalg.norm(b0 - b1, axis=-1) / np.maximum(np.linalg.norm(b0, axis=-1) + np.linalg.norm(b1, axis=-1), EPS)
    )

    metric_map = {
        "pair_cos": vector_cosine(d0, d1),
        "pair_residual": normalized_residual(d0, d1),
        "pair_norm_logratio": np.log(
            (np.linalg.norm(d0, axis=-1) + EPS) / (np.linalg.norm(d1, axis=-1) + EPS)
        ),
        "identity_token_cos": identity_token_cos,
        "role_token_cos": role_token_cos,
        "identity_role_gap": identity_token_cos - role_token_cos,
        "identity_token_residual": identity_token_residual,
        "role_token_residual": role_token_residual,
        "query_shift": query_shift,
        "stable_share": stable_energy / factor_energy,
        "mismatch_share": mismatch_energy / factor_energy,
        "mean_state_cos": vector_cosine(m0, m1),
        "mean_state_residual": normalized_residual(m0, m1),
    }
    matrix = np.stack([metric_map[name] for name in RAW_METRICS], axis=-1)
    return matrix.astype(np.float32), list(RAW_METRICS)


def flatten_head_metrics(
    values: np.ndarray,
    head_names: Sequence[str],
    metric_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    n, h, m = values.shape
    if h != len(head_names) or m != len(metric_names):
        raise ValueError("Feature shape does not match head/metric names")
    names = [f"{head}::{metric}" for head in head_names for metric in metric_names]
    return values.reshape(n, h * m).astype(np.float64), names


# -----------------------------------------------------------------------------
# Fold-local GT-conditioned features
# -----------------------------------------------------------------------------


def leave_one_out_prototype_scores(
    *,
    vectors: np.ndarray,  # [N,H,D], already unit-normalized
    relations: np.ndarray,
    errors: np.ndarray,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    leave_one_out_train: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """GT prototype cosine and GT-vs-opposite prototype margin [T,H]."""
    n_heads = vectors.shape[1]
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for relation in RELATIONS:
        members = train_indices[(relations[train_indices] == relation) & (errors[train_indices] == 0)]
        if members.size == 0:
            raise RuntimeError(f"Outer training fold has no correct samples for relation={relation}")
        sums[relation] = vectors[members].sum(axis=0)
        counts[relation] = int(members.size)

    gt_scores = np.empty((len(target_indices), n_heads), dtype=np.float64)
    margins = np.empty_like(gt_scores)
    train_set = set(map(int, train_indices.tolist()))
    for out_index, sample_index in enumerate(target_indices):
        relation = str(relations[sample_index])
        opposite = OPPOSITE[relation]
        own_correct = bool(errors[sample_index] == 0 and int(sample_index) in train_set)

        gt_sum = sums[relation]
        gt_count = counts[relation]
        if leave_one_out_train and own_correct:
            gt_sum = gt_sum - vectors[sample_index]
            gt_count -= 1
        if gt_count <= 0:
            gt_sum = sums[relation]
            gt_count = counts[relation]
        gt_proto = row_unit(gt_sum / float(gt_count))
        opp_proto = row_unit(sums[opposite] / float(counts[opposite]))
        current = vectors[sample_index]
        gt = np.sum(current * gt_proto, axis=-1)
        opp = np.sum(current * opp_proto, axis=-1)
        gt_scores[out_index] = gt
        margins[out_index] = gt - opp
    return gt_scores, margins


def leave_one_out_normative_z(
    *,
    scalar_values: np.ndarray,  # [N,H,M]
    metric_names: Sequence[str],
    relations: np.ndarray,
    errors: np.ndarray,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    leave_one_out_train: bool,
) -> Tuple[np.ndarray, List[str]]:
    metric_indices = [metric_names.index(name) for name in NORMATIVE_METRICS]
    selected = np.asarray(scalar_values[:, :, metric_indices], dtype=np.float64)
    n_heads = selected.shape[1]
    n_metrics = selected.shape[2]

    sums: Dict[str, np.ndarray] = {}
    sums_sq: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for relation in RELATIONS:
        members = train_indices[(relations[train_indices] == relation) & (errors[train_indices] == 0)]
        if members.size == 0:
            raise RuntimeError(f"Outer training fold has no correct samples for relation={relation}")
        values = selected[members]
        sums[relation] = values.sum(axis=0)
        sums_sq[relation] = (values * values).sum(axis=0)
        counts[relation] = int(members.size)

    z_rows = np.empty((len(target_indices), n_heads, n_metrics), dtype=np.float64)
    train_set = set(map(int, train_indices.tolist()))
    for out_index, sample_index in enumerate(target_indices):
        relation = str(relations[sample_index])
        total = sums[relation]
        total_sq = sums_sq[relation]
        count = counts[relation]
        own_correct = bool(errors[sample_index] == 0 and int(sample_index) in train_set)
        if leave_one_out_train and own_correct and count > 1:
            total = total - selected[sample_index]
            total_sq = total_sq - selected[sample_index] ** 2
            count -= 1
        mean = total / float(max(count, 1))
        variance = np.maximum(total_sq / float(max(count, 1)) - mean * mean, 1e-6)
        std = np.sqrt(variance)
        z_rows[out_index] = (selected[sample_index] - mean) / std

    output = np.concatenate([z_rows, np.abs(z_rows)], axis=-1)
    names = [f"norm_z::{name}" for name in NORMATIVE_METRICS] + [
        f"norm_abs_z::{name}" for name in NORMATIVE_METRICS
    ]
    return output, names


def build_gt_conditioned_features(
    *,
    original_pair_unit: np.ndarray,
    swapped_pair_unit: np.ndarray,
    stable_pair_unit: np.ndarray,
    scalar_values: np.ndarray,
    scalar_metric_names: Sequence[str],
    relations: np.ndarray,
    errors: np.ndarray,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    leave_one_out_train: bool,
    head_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    orig_gt, orig_margin = leave_one_out_prototype_scores(
        vectors=original_pair_unit,
        relations=relations,
        errors=errors,
        train_indices=train_indices,
        target_indices=target_indices,
        leave_one_out_train=leave_one_out_train,
    )
    swap_gt, swap_margin = leave_one_out_prototype_scores(
        vectors=swapped_pair_unit,
        relations=relations,
        errors=errors,
        train_indices=train_indices,
        target_indices=target_indices,
        leave_one_out_train=leave_one_out_train,
    )
    stable_gt, stable_margin = leave_one_out_prototype_scores(
        vectors=stable_pair_unit,
        relations=relations,
        errors=errors,
        train_indices=train_indices,
        target_indices=target_indices,
        leave_one_out_train=leave_one_out_train,
    )
    gt_mean = 0.5 * (orig_gt + swap_gt)
    gt_min = np.minimum(orig_gt, swap_gt)
    gt_gap = np.abs(orig_gt - swap_gt)
    margin_mean = 0.5 * (orig_margin + swap_margin)
    margin_min = np.minimum(orig_margin, swap_margin)
    margin_gap = np.abs(orig_margin - swap_margin)

    proto_names = [
        "proto_orig_gt_cos",
        "proto_swap_gt_cos",
        "proto_stable_gt_cos",
        "proto_gt_cos_mean",
        "proto_gt_cos_min",
        "proto_gt_cos_gap",
        "proto_orig_gt_vs_opp_margin",
        "proto_swap_gt_vs_opp_margin",
        "proto_stable_gt_vs_opp_margin",
        "proto_margin_mean",
        "proto_margin_min",
        "proto_margin_gap",
    ]
    proto_values = np.stack(
        [
            orig_gt,
            swap_gt,
            stable_gt,
            gt_mean,
            gt_min,
            gt_gap,
            orig_margin,
            swap_margin,
            stable_margin,
            margin_mean,
            margin_min,
            margin_gap,
        ],
        axis=-1,
    )

    z_values, z_names = leave_one_out_normative_z(
        scalar_values=scalar_values,
        metric_names=scalar_metric_names,
        relations=relations,
        errors=errors,
        train_indices=train_indices,
        target_indices=target_indices,
        leave_one_out_train=leave_one_out_train,
    )
    combined = np.concatenate([proto_values, z_values], axis=-1)
    combined_names = proto_names + z_names
    return flatten_head_metrics(combined, head_names, combined_names)


# -----------------------------------------------------------------------------
# Detector fitting/evaluation
# -----------------------------------------------------------------------------


def choose_feature_group(
    *,
    group: str,
    confidence_train: np.ndarray,
    confidence_test: np.ndarray,
    swap_train: np.ndarray,
    swap_test: np.ndarray,
    gt_train: np.ndarray,
    gt_test: np.ndarray,
    confidence_names: Sequence[str],
    swap_names: Sequence[str],
    gt_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if group == "confidence":
        return confidence_train, confidence_test, list(confidence_names)
    if group == "swap_raw":
        return swap_train, swap_test, list(swap_names)
    if group == "confidence_plus_swap_raw":
        return (
            np.concatenate([confidence_train, swap_train], axis=1),
            np.concatenate([confidence_test, swap_test], axis=1),
            list(confidence_names) + list(swap_names),
        )
    if group == "gt_conditioned_head":
        return gt_train, gt_test, list(gt_names)
    if group == "confidence_plus_gt_conditioned_head":
        return (
            np.concatenate([confidence_train, gt_train], axis=1),
            np.concatenate([confidence_test, gt_test], axis=1),
            list(confidence_names) + list(gt_names),
        )
    raise ValueError(f"Unknown feature group {group!r}")


def fit_predict_fold(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_i = imputer.fit_transform(x_train)
    x_test_i = imputer.transform(x_test)
    x_train_s = scaler.fit_transform(x_train_i)
    x_test_s = scaler.transform(x_test_i)

    n_features = x_train_s.shape[1]
    k = n_features if int(args.top_k_features) <= 0 else min(int(args.top_k_features), n_features)
    selected_indices = np.arange(n_features)
    if k < n_features:
        selector = SelectKBest(score_func=f_classif, k=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_train_s = selector.fit_transform(x_train_s, y_train)
        x_test_s = selector.transform(x_test_s)
        selected_indices = selector.get_support(indices=True)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=float(args.logreg_c),
        l1_ratio=float(args.l1_ratio),
        class_weight="balanced",
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_train_s, y_train)
    probability = model.predict_proba(x_test_s)[:, 1]
    coefficient = np.asarray(model.coef_[0], dtype=np.float64)
    selected_rows: List[Dict[str, Any]] = []
    for local_index, original_index in enumerate(selected_indices):
        selected_rows.append(
            {
                "feature": str(feature_names[int(original_index)]),
                "coefficient": float(coefficient[local_index]),
                "abs_coefficient": float(abs(coefficient[local_index])),
                "nonzero": bool(abs(coefficient[local_index]) > 1e-10),
            }
        )
    return probability, selected_rows


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return 0.5, float(f1_score(y, p >= 0.5, zero_division=0))
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], EPS)
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), float(f1[index])


def metric_row(
    *, model_name: str, y: np.ndarray, probability: np.ndarray, review_fraction: float
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    threshold, best_f1 = best_f1_threshold(y, p)
    n_review = max(1, int(math.ceil(len(y) * float(review_fraction))))
    order = np.argsort(-p)
    reviewed = order[:n_review]
    retained = order[n_review:]
    error_total = max(int(y.sum()), 1)
    return {
        "model": model_name,
        "N": len(y),
        "errors": int(y.sum()),
        "error_prevalence": float(y.mean()),
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "F1_at_0.5": float(f1_score(y, p >= 0.5, zero_division=0)),
        "best_F1_exploratory": best_f1,
        "best_F1_threshold_exploratory": threshold,
        "review_fraction": float(review_fraction),
        "review_count": n_review,
        "error_recall_at_review_fraction": float(y[reviewed].sum() / error_total),
        "review_precision": float(y[reviewed].mean()),
        "retained_accuracy": float(1.0 - y[retained].mean()) if retained.size else float("nan"),
    }


def parse_feature_head(feature: str) -> Tuple[Optional[str], str]:
    parts = str(feature).split("::")
    if parts and re.fullmatch(r"L\d+H\d+", parts[0]):
        return parts[0], "::".join(parts[1:])
    return None, str(feature)


def run_detector(
    *,
    args: argparse.Namespace,
    sids: np.ndarray,
    relations: np.ndarray,
    errors: np.ndarray,
    confidence: np.ndarray,
    confidence_names: Sequence[str],
    scalar_values: np.ndarray,
    scalar_metric_names: Sequence[str],
    swap_raw: np.ndarray,
    swap_raw_names: Sequence[str],
    original_pair_unit: np.ndarray,
    swapped_pair_unit: np.ndarray,
    stable_pair_unit: np.ndarray,
    head_names: Sequence[str],
    metadata_by_sid: Mapping[int, Mapping[str, Any]],
    output_dir: Path,
) -> None:
    groups = [item.strip() for item in str(args.model_groups).split(",") if item.strip()]
    invalid = sorted(set(groups) - set(MODEL_GROUPS))
    if invalid:
        raise ValueError(f"Unknown --model-groups entries: {invalid}")

    strata = np.asarray(
        [f"{relations[i]}__{int(errors[i])}" for i in range(len(errors))], dtype=object
    )
    stratum_counts = Counter(map(str, strata.tolist()))
    too_small = {key: value for key, value in stratum_counts.items() if value < int(args.outer_folds)}
    if too_small:
        raise RuntimeError(
            f"Each relation/correctness stratum needs at least --outer-folds samples; "
            f"too small: {too_small}. Use the full dataset or reduce --outer-folds."
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=int(args.outer_folds),
        n_repeats=int(args.outer_repeats),
        random_state=int(args.seed),
    )

    probability_sum = {group: np.zeros(len(errors), dtype=np.float64) for group in groups}
    probability_count = {group: np.zeros(len(errors), dtype=np.int64) for group in groups}
    repeat_probabilities = {
        group: np.full((int(args.outer_repeats), len(errors)), np.nan, dtype=np.float64)
        for group in groups
    }
    selection_records: List[Dict[str, Any]] = []
    repeat_rows: List[Dict[str, Any]] = []

    for split_index, (train_indices, test_indices) in enumerate(splitter.split(np.zeros(len(errors)), strata)):
        repeat_index = split_index // int(args.outer_folds)
        fold_index = split_index % int(args.outer_folds)
        gt_train, gt_names = build_gt_conditioned_features(
            original_pair_unit=original_pair_unit,
            swapped_pair_unit=swapped_pair_unit,
            stable_pair_unit=stable_pair_unit,
            scalar_values=scalar_values,
            scalar_metric_names=scalar_metric_names,
            relations=relations,
            errors=errors,
            train_indices=train_indices,
            target_indices=train_indices,
            leave_one_out_train=True,
            head_names=head_names,
        )
        gt_test, gt_test_names = build_gt_conditioned_features(
            original_pair_unit=original_pair_unit,
            swapped_pair_unit=swapped_pair_unit,
            stable_pair_unit=stable_pair_unit,
            scalar_values=scalar_values,
            scalar_metric_names=scalar_metric_names,
            relations=relations,
            errors=errors,
            train_indices=train_indices,
            target_indices=test_indices,
            leave_one_out_train=False,
            head_names=head_names,
        )
        if gt_names != gt_test_names:
            raise RuntimeError("GT-conditioned train/test feature names differ")

        for group in groups:
            x_train, x_test, names = choose_feature_group(
                group=group,
                confidence_train=confidence[train_indices],
                confidence_test=confidence[test_indices],
                swap_train=swap_raw[train_indices],
                swap_test=swap_raw[test_indices],
                gt_train=gt_train,
                gt_test=gt_test,
                confidence_names=confidence_names,
                swap_names=swap_raw_names,
                gt_names=gt_names,
            )
            probability, selected = fit_predict_fold(
                x_train=x_train,
                y_train=errors[train_indices],
                x_test=x_test,
                feature_names=names,
                args=args,
            )
            probability_sum[group][test_indices] += probability
            probability_count[group][test_indices] += 1
            repeat_probabilities[group][repeat_index, test_indices] = probability
            for row in selected:
                head, metric = parse_feature_head(str(row["feature"]))
                selection_records.append(
                    {
                        "model": group,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "feature": row["feature"],
                        "head": head,
                        "metric": metric,
                        "coefficient": row["coefficient"],
                        "abs_coefficient": row["abs_coefficient"],
                        "nonzero": row["nonzero"],
                    }
                )
        print(
            f"[detector repeat={repeat_index + 1}/{args.outer_repeats} "
            f"fold={fold_index + 1}/{args.outer_folds}] train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )

    performance_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    averaged_probability: Dict[str, np.ndarray] = {}
    for group in groups:
        if not np.all(probability_count[group] == int(args.outer_repeats)):
            raise RuntimeError(
                f"Model {group}: OOF count mismatch {np.unique(probability_count[group], return_counts=True)}"
            )
        avg = probability_sum[group] / np.maximum(probability_count[group], 1)
        averaged_probability[group] = avg
        performance_rows.append(
            metric_row(
                model_name=group,
                y=errors,
                probability=avg,
                review_fraction=float(args.review_fraction),
            )
        )
        for repeat_index in range(int(args.outer_repeats)):
            p = repeat_probabilities[group][repeat_index]
            if np.isnan(p).any():
                raise RuntimeError(f"Model {group}, repeat {repeat_index}: missing OOF probabilities")
            repeat_rows.append(
                {
                    "repeat": repeat_index,
                    **metric_row(
                        model_name=group,
                        y=errors,
                        probability=p,
                        review_fraction=float(args.review_fraction),
                    ),
                }
            )
        for relation in RELATIONS:
            mask = relations == relation
            if mask.sum() < 2 or len(np.unique(errors[mask])) < 2:
                continue
            relation_rows.append(
                {
                    "relation": relation,
                    **metric_row(
                        model_name=group,
                        y=errors[mask],
                        probability=avg[mask],
                        review_fraction=float(args.review_fraction),
                    ),
                }
            )

    for i, sid in enumerate(sids):
        meta = metadata_by_sid[int(sid)]
        row = {
            "sid": int(sid),
            "gt": str(relations[i]),
            "gt_raw": meta.get("gt_raw"),
            "baseline_prediction": meta.get("baseline_prediction"),
            "baseline_correct": bool(errors[i] == 0),
            "error_label": int(errors[i]),
            "original_closed_prediction": meta.get("original_closed_prediction"),
            "swapped_closed_prediction": meta.get("swapped_closed_prediction"),
        }
        for group in groups:
            row[f"error_probability__{group}"] = float(averaged_probability[group][i])
        oof_rows.append(row)

    write_csv(output_dir / "oof_predictions.csv", oof_rows)
    write_csv(output_dir / "model_performance.csv", performance_rows)
    write_csv(output_dir / "repeat_performance.csv", repeat_rows)
    write_csv(output_dir / "relation_performance.csv", relation_rows)
    write_csv(output_dir / "selected_features_by_fold.csv", selection_records)

    # Feature stability.
    total_folds = int(args.outer_folds) * int(args.outer_repeats)
    grouped_features: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in selection_records:
        grouped_features[(str(row["model"]), str(row["feature"]))].append(row)
    stability_rows: List[Dict[str, Any]] = []
    for (model_name, feature), rows in grouped_features.items():
        head, metric = parse_feature_head(feature)
        nonzero_rows = [row for row in rows if bool(row["nonzero"])]
        stability_rows.append(
            {
                "model": model_name,
                "feature": feature,
                "head": head,
                "metric": metric,
                "selected_fold_count": len(rows),
                "selected_fold_fraction": len(rows) / float(total_folds),
                "nonzero_fold_count": len(nonzero_rows),
                "nonzero_fold_fraction": len(nonzero_rows) / float(total_folds),
                "mean_coefficient_when_selected": safe_mean(float(row["coefficient"]) for row in rows),
                "mean_abs_coefficient_when_selected": safe_mean(float(row["abs_coefficient"]) for row in rows),
            }
        )
    stability_rows.sort(
        key=lambda row: (
            str(row["model"]),
            -float(row["nonzero_fold_fraction"]),
            -float(row["mean_abs_coefficient_when_selected"]),
        )
    )
    write_csv(output_dir / "selected_feature_stability.csv", stability_rows)

    # Aggregate stable importance by head.
    head_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        if row.get("head"):
            head_groups[(str(row["model"]), str(row["head"]))].append(row)
    head_rows: List[Dict[str, Any]] = []
    for (model_name, head), rows in head_groups.items():
        best = max(
            rows,
            key=lambda row: (
                float(row["nonzero_fold_fraction"]),
                float(row["mean_abs_coefficient_when_selected"]),
            ),
        )
        head_rows.append(
            {
                "model": model_name,
                "head": head,
                "max_nonzero_fold_fraction": max(float(row["nonzero_fold_fraction"]) for row in rows),
                "mean_nonzero_fold_fraction": safe_mean(float(row["nonzero_fold_fraction"]) for row in rows),
                "max_mean_abs_coefficient": max(float(row["mean_abs_coefficient_when_selected"]) for row in rows),
                "dominant_feature": best["feature"],
                "dominant_metric": best["metric"],
                "dominant_mean_coefficient": best["mean_coefficient_when_selected"],
            }
        )
    head_rows.sort(
        key=lambda row: (
            str(row["model"]),
            -float(row["max_nonzero_fold_fraction"]),
            -float(row["max_mean_abs_coefficient"]),
        )
    )
    write_csv(output_dir / "head_importance_summary.csv", head_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "samples": len(sids),
        "correct": int((errors == 0).sum()),
        "wrong": int(errors.sum()),
        "error_prevalence": float(errors.mean()),
        "model_performance": performance_rows,
        "top_heads": {
            group: [
                row
                for row in head_rows
                if row["model"] == group
            ][:20]
            for group in groups
        },
        "limits": [
            "GT-conditioned models use relation labels and training-fold correct prototypes.",
            "Free-generation correctness is the target label.",
            "Closed-set confidence is measured from original/swapped prefill relation scores.",
            "Predictive association is not causal attribution.",
        ],
    }
    write_json(output_dir / "summary.json", summary)

    print("\n" + "=" * 160)
    print("COCO PER-HEAD OBJECT-SWAP ERROR DETECTOR")
    print("=" * 160)
    print(
        f"Samples={len(sids)} | correct={(errors == 0).sum()} | wrong={errors.sum()} | "
        f"heads={len(head_names)} | repeats={args.outer_repeats} folds={args.outer_folds}"
    )
    print("\nOOF PERFORMANCE")
    for row in sorted(performance_rows, key=lambda item: float(item["AUPRC"]), reverse=True):
        print(
            f"{str(row['model']):42s} AUROC={float(row['AUROC']):.4f} "
            f"AUPRC={float(row['AUPRC']):.4f} Brier={float(row['Brier']):.4f} "
            f"top{int(round(100 * float(args.review_fraction)))}%-error-recall="
            f"{float(row['error_recall_at_review_fraction']):.4f}"
        )
    for group in groups:
        relevant = [row for row in head_rows if row["model"] == group][:10]
        if not relevant:
            continue
        print(f"\nTOP STABLE HEADS: {group}")
        for row in relevant:
            print(
                f"{str(row['head']):8s} stable={float(row['max_nonzero_fold_fraction']):.3f} "
                f"|coef|={float(row['max_mean_abs_coefficient']):.4f} "
                f"metric={row['dominant_metric']}"
            )
    print(f"\nSaved outputs to {output_dir}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if int(args.outer_folds) < 2 or int(args.outer_repeats) < 1:
        raise ValueError("--outer-folds must be >=2 and --outer-repeats >=1")
    if not 0.0 < float(args.review_fraction) < 1.0:
        raise ValueError("--review-fraction must be in (0,1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists() and args.phase in ("extract", "all"):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    cells_path = output_dir / "swap_cells.jsonl"

    factor = import_file(Path(args.factorial_script), "head_swap_factorial")
    ioi = import_file(Path(args.ioi_script), "head_swap_ioi")
    producer = import_file(Path(args.producer_script), "head_swap_producer")
    receiver = import_file(Path(args.receiver_script), "head_swap_receiver")
    v3 = import_file(Path(args.v3_script), "head_swap_v3")
    base = import_file(Path(args.base_script), "head_swap_base")
    attention_helper = import_file(Path(args.attention_helper), "head_swap_attention")

    source_config, source_rows = ioi.load_source_rows(args)
    baseline_path = factor.resolve_baseline_path(args)
    baseline_by_sid = factor.load_baseline_rows(baseline_path)
    if not baseline_by_sid:
        raise RuntimeError(f"No baseline generation rows found at {baseline_path}")

    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(str(args.include_sids_file).strip()))
    excluded: set[int] = set()
    for item in str(args.exclude_sids_from).split(","):
        if item.strip():
            excluded.update(extract_sids(Path(item.strip())))

    selected_rows: List[Dict[str, Any]] = []
    missing_baseline: List[int] = []
    for row in source_rows:
        sid = int(row["sid"])
        gt_raw = row.get("gt")
        gt = normalize_relation(gt_raw)
        if gt not in RELATIONS:
            continue
        if sid in excluded or (included is not None and sid not in included):
            continue
        baseline = baseline_by_sid.get(sid)
        if baseline is None:
            missing_baseline.append(sid)
            continue
        selected_rows.append(
            {
                **dict(row),
                "sid": sid,
                "gt": gt,
                "gt_raw": gt_raw,
                "baseline_prediction": normalize_relation(baseline.get("prediction")),
                "baseline_correct": bool(baseline.get("correct", False)),
            }
        )
    selected_rows = stratified_limit(selected_rows, int(args.sample_max_samples), int(args.seed))
    if not selected_rows:
        raise RuntimeError("No COCO samples selected")
    selected_sids = [int(row["sid"]) for row in selected_rows]

    model = None
    processor = None
    scan_spec: Optional[ScanSpec] = None
    try:
        if args.phase in ("extract", "all"):
            (
                model,
                processor,
                spec_model,
                decoder_layers,
                decoder_path,
                relation_token_map,
            ) = producer.load_model_bundle(args=args, base=base)
            scan_spec = build_scan_spec(
                args=args,
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver=receiver,
            )
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

            config = {
                "script_version": SCRIPT_VERSION,
                "model": args.model,
                "repo_id": spec_model.repo_id,
                "dataset": "coco_two_all_four_relations",
                "source_output_dir": args.source_output_dir,
                "source_script_version": source_config.get("script_version"),
                "baseline_generation_jsonl": str(baseline_path),
                "decoder_path": decoder_path,
                "n_layers": len(decoder_layers),
                "scan_layers": scan_spec.scan_layers,
                "head_dim": scan_spec.head_dim,
                "heads": [item.name for item in scan_spec.heads],
                "relations": list(RELATIONS),
                "opposites": OPPOSITE,
                "selected_samples": len(selected_rows),
                "selected_sids": selected_sids,
                "missing_baseline_sids": missing_baseline,
                "capture_pool": args.capture_pool,
                "save_dtype": args.save_dtype,
                "audit": audit,
                "feature_definition": (
                    "Identity A/B object-token pre-W_O head outputs under original and role-swapped queries."
                ),
            }
            write_json(output_dir / "config.json", config)
            write_json(
                output_dir / "head_order.json",
                [
                    {
                        "index": item.global_index,
                        "name": item.name,
                        "layer": item.layer,
                        "head": item.head,
                        "head_dim": item.head_dim,
                    }
                    for item in scan_spec.heads
                ],
            )

            existing = deduplicate_rows(read_jsonl(cells_path), ("sid",)) if args.resume else []
            done = {
                int(row["sid"])
                for row in existing
                if (vector_dir / f"sid_{int(row['sid']):06d}.npz").exists()
            }
            pending = [row for row in selected_rows if int(row["sid"]) not in done]
            print(
                f"Object-swap extraction: samples={len(selected_rows)} pending={len(pending)} "
                f"heads={len(scan_spec.heads)} forwards={2 * len(pending)}",
                flush=True,
            )
            save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
            for index, source_row in enumerate(
                tqdm(pending, desc=f"head-swap-extract:{args.model}"), start=1
            ):
                pair = None
                try:
                    sid = int(source_row["sid"])
                    gt = str(source_row["gt"])
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
                    original_capture = HeadObjectCapture(
                        spec=scan_spec,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        ioi=ioi,
                        a_positions=pair.original_a_positions,
                        b_positions=pair.original_b_positions,
                        pool_mode=args.capture_pool,
                    )
                    with original_capture:
                        original_outputs = model(
                            **dict(pair.original_batch),
                            use_cache=False,
                            return_dict=True,
                        )
                    original_heads = original_capture.finalize()
                    original_scores = relation_score_dict(
                        outputs=original_outputs,
                        relation_token_map=relation_token_map,
                        base=base,
                        factor=factor,
                    )
                    original_prediction, original_margin = prediction_and_margin(original_scores)
                    del original_outputs

                    swapped_capture = HeadObjectCapture(
                        spec=scan_spec,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        ioi=ioi,
                        a_positions=pair.swapped_a_positions,
                        b_positions=pair.swapped_b_positions,
                        pool_mode=args.capture_pool,
                    )
                    with swapped_capture:
                        swapped_outputs = model(
                            **dict(pair.swapped_batch),
                            use_cache=False,
                            return_dict=True,
                        )
                    swapped_heads = swapped_capture.finalize()
                    swapped_scores = relation_score_dict(
                        outputs=swapped_outputs,
                        relation_token_map=relation_token_map,
                        base=base,
                        factor=factor,
                    )
                    swapped_prediction, swapped_margin = prediction_and_margin(swapped_scores)
                    del swapped_outputs

                    vector_path = vector_dir / f"sid_{sid:06d}.npz"
                    np.savez_compressed(
                        vector_path,
                        original_heads=original_heads.astype(save_dtype),
                        swapped_heads=swapped_heads.astype(save_dtype),
                    )
                    baseline = baseline_by_sid[sid]
                    row = {
                        "script_version": SCRIPT_VERSION,
                        "sid": sid,
                        "gt": gt,
                        "gt_raw": source_row.get("gt_raw"),
                        "opposite_gt": OPPOSITE[gt],
                        "baseline_prediction": normalize_relation(baseline.get("prediction")),
                        "baseline_correct": bool(baseline.get("correct", False)),
                        "subject": prompt_rows[sid].get("subject"),
                        "reference": prompt_rows[sid].get("reference"),
                        "original_closed_scores": original_scores,
                        "original_closed_prediction": original_prediction,
                        "original_closed_top_margin": original_margin,
                        "original_closed_correct": bool(original_prediction == gt),
                        "swapped_closed_scores": swapped_scores,
                        "swapped_closed_prediction": swapped_prediction,
                        "swapped_closed_top_margin": swapped_margin,
                        "swapped_closed_correct": bool(swapped_prediction == OPPOSITE[gt]),
                        "prediction_pair_opposite": bool(swapped_prediction == OPPOSITE.get(original_prediction)),
                        "original_a_positions": list(map(int, pair.original_a_positions)),
                        "original_b_positions": list(map(int, pair.original_b_positions)),
                        "swapped_a_positions": list(map(int, pair.swapped_a_positions)),
                        "swapped_b_positions": list(map(int, pair.swapped_b_positions)),
                        "vector_file": str(vector_path),
                    }
                    append_jsonl(cells_path, row)
                    if args.print_every > 0 and index % int(args.print_every) == 0:
                        print(
                            f"[extract {index}/{len(pending)} sid={sid}] gt={gt} "
                            f"free={row['baseline_prediction']} correct={row['baseline_correct']} "
                            f"closed={original_prediction}/{swapped_prediction}",
                            flush=True,
                        )
                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "phase": "extract",
                            "sid": int(source_row.get("sid", -1)),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(
                        f"[ERROR extract sid={source_row.get('sid')}] {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                finally:
                    if pair is not None:
                        receiver.release_pair(pair)
                    gc.collect()
                    if (
                        args.device.startswith("cuda")
                        and args.empty_cache_every > 0
                        and index % int(args.empty_cache_every) == 0
                    ):
                        torch.cuda.empty_cache()

            if args.phase == "extract":
                rows = deduplicate_rows(read_jsonl(cells_path), ("sid",))
                built = {
                    int(row["sid"])
                    for row in rows
                    if (vector_dir / f"sid_{int(row['sid']):06d}.npz").exists()
                }
                missing = sorted(set(selected_sids) - built)
                print(
                    f"Extraction completed: built={len(built)}/{len(selected_sids)} failed={len(missing)}",
                    flush=True,
                )
                if missing:
                    raise RuntimeError(f"Incomplete extraction SIDs {missing[:20]}; inspect {errors_path}")
                return

        # Analyze phase: no model needed.  Load scan specification from saved config.
        if args.phase == "analyze":
            config_path = output_dir / "config.json"
            head_order_path = output_dir / "head_order.json"
            if not config_path.exists() or not head_order_path.exists():
                raise FileNotFoundError(
                    f"Analyze phase requires {config_path} and {head_order_path}; run extraction first"
                )
        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        head_order = json.loads((output_dir / "head_order.json").read_text(encoding="utf-8"))
        head_names = [str(row["name"]) for row in head_order]

        cell_rows = deduplicate_rows(read_jsonl(cells_path), ("sid",))
        metadata_by_sid = {int(row["sid"]): dict(row) for row in cell_rows}
        available_sids = [
            sid for sid in selected_sids
            if sid in metadata_by_sid and (vector_dir / f"sid_{sid:06d}.npz").exists()
        ]
        missing = sorted(set(selected_sids) - set(available_sids))
        if missing:
            raise RuntimeError(f"Missing extracted SIDs {missing[:20]}; run --phase extract first")

        original_heads, swapped_heads = load_vectors(vector_dir=vector_dir, sids=available_sids)
        scalar_values, scalar_metric_names = compute_swap_scalar_features(
            original_heads, swapped_heads
        )
        swap_raw, swap_raw_names = flatten_head_metrics(
            scalar_values, head_names, scalar_metric_names
        )

        relations = np.asarray([metadata_by_sid[sid]["gt"] for sid in available_sids], dtype=object)
        errors = np.asarray(
            [0 if bool(metadata_by_sid[sid]["baseline_correct"]) else 1 for sid in available_sids],
            dtype=np.int64,
        )
        sids_array = np.asarray(available_sids, dtype=np.int64)

        confidence_rows: List[np.ndarray] = []
        confidence_names: Optional[List[str]] = None
        for sid in available_sids:
            meta = metadata_by_sid[sid]
            values, names = confidence_features_from_scores(
                original=meta["original_closed_scores"],
                swapped=meta["swapped_closed_scores"],
                original_prediction=str(meta["original_closed_prediction"]),
                swapped_prediction=str(meta["swapped_closed_prediction"]),
            )
            confidence_rows.append(values)
            confidence_names = names
        confidence = np.stack(confidence_rows, axis=0)
        assert confidence_names is not None

        # Precompute unit object-pair vectors once.  This avoids rebuilding
        # several hundred MB of pair tensors inside every outer fold.
        original_pair = (
            np.asarray(original_heads[:, :, 0, :], dtype=np.float32)
            - np.asarray(original_heads[:, :, 1, :], dtype=np.float32)
        )
        swapped_pair = (
            np.asarray(swapped_heads[:, :, 0, :], dtype=np.float32)
            - np.asarray(swapped_heads[:, :, 1, :], dtype=np.float32)
        )
        stable_pair = 0.5 * (original_pair + swapped_pair)
        original_pair_unit = row_unit(original_pair).astype(np.float32, copy=False)
        swapped_pair_unit = row_unit(swapped_pair).astype(np.float32, copy=False)
        stable_pair_unit = row_unit(stable_pair).astype(np.float32, copy=False)
        del original_pair, swapped_pair, stable_pair

        if args.write_long_features:
            long_rows: List[Dict[str, Any]] = []
            for sample_index, sid in enumerate(available_sids):
                meta = metadata_by_sid[sid]
                for head_index, head in enumerate(head_names):
                    row = {
                        "sid": sid,
                        "gt": relations[sample_index],
                        "gt_raw": meta.get("gt_raw"),
                        "baseline_prediction": meta.get("baseline_prediction"),
                        "baseline_correct": bool(errors[sample_index] == 0),
                        "error_label": int(errors[sample_index]),
                        "head": head,
                    }
                    for metric_index, metric in enumerate(scalar_metric_names):
                        row[metric] = float(scalar_values[sample_index, head_index, metric_index])
                    long_rows.append(row)
            write_csv(output_dir / "sample_swap_scalar_features.csv", long_rows)

        run_detector(
            args=args,
            sids=sids_array,
            relations=relations,
            errors=errors,
            confidence=confidence,
            confidence_names=confidence_names,
            scalar_values=scalar_values,
            scalar_metric_names=scalar_metric_names,
            swap_raw=swap_raw,
            swap_raw_names=swap_raw_names,
            original_pair_unit=original_pair_unit,
            swapped_pair_unit=swapped_pair_unit,
            stable_pair_unit=stable_pair_unit,
            head_names=head_names,
            metadata_by_sid=metadata_by_sid,
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
