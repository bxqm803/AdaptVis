#!/usr/bin/env python3
"""
Signed routing-shift analysis for the validated COCO/Qwen2.5-VL spatial circuit.

Purpose
-------
This script does not try to improve generation accuracy. It asks:

1. Do wrong samples still use the canonical P_POS7 -> L26VH0 route, but with
   the relation direction reversed?
2. Do heads that are weak on correct samples become strong on wrong samples?
3. Does relation evidence become diffuse across many heads?
4. Is the L26VH0 relation state already wrong, or does the final generation
   disagree with an L26 channel state that still favors GT?

Core intervention
-----------------
For every selected sender head h in L19-L23:

    h at original object-token positions
        -> residual / token-wise MLP path
        -> L26 shared KV-head-0 Value channel

The script performs an IOI-style C-pass:

- all intermediate attention outputs at relevant positions are frozen to clean;
- only h is removed at the selected sender object positions;
- the resulting L26VH0 Value state is captured;
- no free generation is rerun for each head.

The head's receiver-state contribution is:

    delta_h = V26_base - V26_without_h

At the two role positions, the relation-bearing contribution is:

    role_delta_h = delta_h(subject) - delta_h(reference)

Cross-fitted relation directions are learned only from clean receiver states of
baseline-correct samples:

    horizontal axis = centroid(left) - centroid(right)
    vertical axis   = centroid(above) - centroid(below)

Every head is then assigned:

- transport norm;
- signed projection toward GT;
- signed projection toward the generated prediction relative to GT;
- rank among all scanned heads;
- canonical/off-circuit membership;
- weak-on-correct / emergent-on-wrong flags.

This is a GT-supervised mechanism-analysis script, not a deployable detector.
GT is used to orient relation axes and to classify failure mechanisms.

Default scan
------------
All query heads in decoder layers 19,20,21,22,23.

Main outputs
------------
raw_head_routing.jsonl
    Scalar C-pass measurements before relation-axis projection.

vectors/sid_XXXXXX.npz
    Clean L26VH0 role state and per-head subject/reference/role deltas.

signed_head_routing.csv
    Final per-sample, per-head signed routing table.

head_shift_summary.csv
    Correct-vs-wrong transport/routing shifts for every head.

sample_routing_summary.csv
    Canonical share, off-circuit share, routing entropy, top-k concentration,
    weak-head emergence, channel prediction, and heuristic failure class.

status_routing_summary.csv
    Aggregate metrics for all/correct/wrong and CC/CW/WC/WW groups.

weak_head_emergence.csv
    Detailed weak-on-correct heads that emerge on wrong samples.

summary.json, config.json, errors.jsonl

Required companion scripts in repository root
---------------------------------------------
analyze_coco_circuit_failure_repair_v1.py
analyze_coco_ioi_backward_circuit_v1.py
analyze_coco_producer_qk_ov_v1.py
analyze_coco_receiver_qkv_v1.py
analyze_spatial_storage_transport_utilization_v3.py
analyze_coco_centroid_generation_step1_v4.py
analyze_coco_flip_attention_spatial_vectors_v1.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-signed-routing-shift-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
STATUS_ALIASES = {
    "both_correct": "CC",
    "original_only": "CW",
    "swapped_only": "WC",
    "both_wrong": "WW",
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
        "--baseline-generation-jsonl",
        default="",
        help="Previously reproduced free-generation baseline rows. When empty, "
        "uses <head-output-dir>/baseline_generation.jsonl.",
    )
    p.add_argument(
        "--head-output-dir",
        default=(
            "output/coco_ioi_backward/"
            "qwen-3b_head_misrouting_pos7_neg5"
        ),
        help="Used only to infer baseline_generation.jsonl.",
    )

    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument("--canonical-bundle", default="P_POS7")
    p.add_argument(
        "--scan-layers",
        default="19,20,21,22,23",
        help="Comma/range layer specification, e.g. 19-23.",
    )
    p.add_argument(
        "--scan-heads",
        default="",
        help="Optional explicit heads, e.g. L19H8,L21H14 or 19:8,21:14. "
        "Overrides --scan-layers.",
    )

    p.add_argument("--receiver-layer", type=int, default=26)
    p.add_argument("--receiver-query-head", type=int, default=0)
    p.add_argument("--receiver-channel", choices=("v",), default="v")
    p.add_argument(
        "--sender-position-mode",
        choices=("joint", "subject", "reference"),
        default="joint",
        help="joint removes a head at both role positions in the same C-pass.",
    )

    p.add_argument(
        "--sample-status",
        choices=(
            "all",
            "both_correct",
            "original_only",
            "swapped_only",
            "both_wrong",
        ),
        default="all",
    )
    p.add_argument(
        "--sample-max-samples",
        type=int,
        default=0,
        help="0 means all eligible samples.",
    )
    p.add_argument(
        "--include-sids-file",
        default="",
        help="Optional JSONL/CSV/text file containing the only SIDs to run.",
    )
    p.add_argument(
        "--exclude-sids-from",
        default="",
        help="Optional comma-separated JSONL/CSV/text files containing SIDs.",
    )

    p.add_argument(
        "--axis-folds",
        type=int,
        default=5,
        help="Cross-fitting folds for relation axes and class centroids.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k concentration and emergent-head cutoff.",
    )
    p.add_argument(
        "--weak-rank-fraction",
        type=float,
        default=0.50,
        help="A head is weak on correct samples when its median rank is below "
        "this fraction of the head list, e.g. 0.50 means bottom half.",
    )
    p.add_argument(
        "--emergent-q",
        type=float,
        default=0.95,
        help="Correct-sample quantile used for unexpected wrong-sample strength.",
    )
    p.add_argument(
        "--effect-epsilon",
        type=float,
        default=1e-6,
    )
    p.add_argument(
        "--exact-score-effects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also patch every ablated L26VH0 state into a scoring pass. "
        "This approximately doubles runtime.",
    )

    p.add_argument(
        "--canonical-share-threshold",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--canonical-wrong-fraction-threshold",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--off-takeover-threshold",
        type=float,
        default=0.65,
    )
    p.add_argument(
        "--emergent-wrong-share-threshold",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--diffuse-entropy-threshold",
        type=float,
        default=0.80,
    )
    p.add_argument(
        "--diffuse-topk-threshold",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--cancellation-threshold",
        type=float,
        default=0.25,
    )

    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument(
        "--failure-script",
        default="analyze_coco_circuit_failure_repair_v1.py",
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

    # Compatibility with imported helpers.
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Basic utilities
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_mean(values: Iterable[Any]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def safe_quantile(values: Iterable[Any], q: float) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def safe_std(values: Iterable[Any]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if x.size >= 2 else float("nan")


def cohens_d(correct: Sequence[float], wrong: Sequence[float]) -> float:
    a = np.asarray(correct, dtype=np.float64)
    b = np.asarray(wrong, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_num = (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled = math.sqrt(max(float(pooled_num / pooled_den), 0.0))
    if pooled <= 0:
        return float("nan")
    return float((b.mean() - a.mean()) / pooled)


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "left_of": "left",
        "right_of": "right",
        "on": "above",
        "under": "below",
    }
    if text in RELATIONS:
        return text
    return aliases.get(text)


def relation_margin(scores: Mapping[str, float], target: Optional[str]) -> float:
    target = normalize_relation(target)
    if target is None or target not in scores:
        return float("nan")
    return float(scores[target]) - float(scores[OPPOSITE[target]])


def parse_head(value: Any) -> Tuple[int, int]:
    text = str(value).strip()
    if text.startswith("L") and "H" in text:
        layer, head = text[1:].split("H", 1)
        return int(layer), int(head)
    if ":" in text:
        layer, head = text.split(":", 1)
        return int(layer), int(head)
    raise ValueError(f"Invalid head {value!r}")


def head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    result: List[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(item)
            result.extend(range(start, stop + 1))
        else:
            result.append(int(item))
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("No layers selected")
    for layer in result:
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside 0..{n_layers - 1}")
    return result


def stable_fold(sid: int, folds: int, seed: int) -> int:
    payload = f"{seed}:{int(sid)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % int(folds)


def unit_vector(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    if not math.isfinite(norm) or norm <= eps:
        return np.zeros_like(x, dtype=np.float64)
    return x / norm


def normalized_entropy(weights: Sequence[float]) -> float:
    x = np.asarray(weights, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if total <= 0 or len(x) <= 1:
        return 0.0
    p = x / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(x)))


def rank_desc(values: Sequence[float]) -> np.ndarray:
    """1-based descending ranks; average rank for ties."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(-x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def extract_sids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    result: set[int] = set()
    if path.suffix.lower() == ".jsonl":
        for row in read_jsonl(path):
            if "sid" in row:
                result.add(int(row["sid"]))
        return result
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("sid", "")).strip():
                    result.add(int(row["sid"]))
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                if isinstance(payload, Mapping) and "sid" in payload:
                    result.add(int(payload["sid"]))
                else:
                    result.add(int(payload))
            except json.JSONDecodeError:
                result.add(int(text.split(",", 1)[0]))
    return result


# -----------------------------------------------------------------------------
# Bundle/head selection
# -----------------------------------------------------------------------------


def load_bundles(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if not isinstance(source, Mapping):
        raise ValueError("Bundle JSON must contain an object")
    result: Dict[str, List[Tuple[int, int]]] = {}
    for name, values in source.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Bundle {name} must be a list")
        parsed = [parse_head(value) for value in values]
        result[str(name)] = parsed
    return result


@dataclass(frozen=True)
class ScanHead:
    layer: int
    head: int
    node: Any
    canonical: bool

    @property
    def name(self) -> str:
        return head_name(self.layer, self.head)


def build_scan_heads(
    *,
    args: argparse.Namespace,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    ioi: Any,
    canonical_heads: set[Tuple[int, int]],
) -> List[ScanHead]:
    selected: List[Tuple[int, int]] = []
    if str(args.scan_heads).strip():
        selected = [
            parse_head(item)
            for item in str(args.scan_heads).split(",")
            if item.strip()
        ]
    else:
        layers = parse_layer_spec(args.scan_layers, len(decoder_layers))
        for layer in layers:
            attention = attention_helper.resolve_self_attention(decoder_layers[layer])
            shape = receiver_module.resolve_attention_shape(attention)
            selected.extend(
                (int(layer), int(head))
                for head in range(int(shape.n_query_heads))
            )

    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError("No sender heads selected")

    result: List[ScanHead] = []
    for layer, head in selected:
        if layer >= int(args.receiver_layer):
            raise ValueError(
                f"Sender {head_name(layer, head)} must be earlier than "
                f"receiver layer {args.receiver_layer}"
            )
        attention = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver_module.resolve_attention_shape(attention)
        if not 0 <= head < int(shape.n_query_heads):
            raise ValueError(
                f"{head_name(layer, head)} outside query-head range "
                f"0..{int(shape.n_query_heads)-1}"
            )
        result.append(
            ScanHead(
                layer=int(layer),
                head=int(head),
                node=ioi.SenderNode("attention", int(layer), int(head)),
                canonical=(int(layer), int(head)) in canonical_heads,
            )
        )
    return result


# -----------------------------------------------------------------------------
# Baseline and vector helpers
# -----------------------------------------------------------------------------


def resolve_baseline_path(args: argparse.Namespace) -> Path:
    if str(args.baseline_generation_jsonl).strip():
        return Path(str(args.baseline_generation_jsonl).strip())
    return Path(args.head_output_dir) / "baseline_generation.jsonl"


def deduplicate_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    table: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        table[key] = dict(row)
    return list(table.values())


def load_baseline_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = deduplicate_rows(read_jsonl(path), ("sid",))
    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        prediction = normalize_relation(
            row.get("prediction", row.get("baseline_generation_prediction"))
        )
        if gt is None:
            raise RuntimeError(f"SID {sid}: missing/invalid GT in baseline file")
        correct = row.get("correct")
        if correct is None:
            correct = prediction == gt
        result[sid] = {
            **dict(row),
            "sid": sid,
            "gt": gt,
            "prediction": prediction,
            "correct": bool(correct),
            "parsed": prediction is not None,
        }
    return result


def state_head_slice(
    tensor: torch.Tensor,
    *,
    unit_head: int,
    head_dim: int,
) -> np.ndarray:
    start = int(unit_head) * int(head_dim)
    stop = start + int(head_dim)
    value = tensor[start:stop].detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def source_status(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    value = row.get(
        "generation_pair_status",
        baseline.get(
            "source_generation_pair_status",
            baseline.get("generation_pair_status", "unknown"),
        ),
    )
    return str(value)


# -----------------------------------------------------------------------------
# Cross-fitted axes
# -----------------------------------------------------------------------------


@dataclass
class AxisModel:
    center: np.ndarray
    class_centroids: Dict[str, np.ndarray]
    class_directions: Dict[str, np.ndarray]
    horizontal: np.ndarray
    vertical: np.ndarray


def fit_axis_model(
    records: Sequence[Mapping[str, Any]],
    baseline_vectors: Mapping[int, np.ndarray],
) -> AxisModel:
    by_relation: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_vectors: List[np.ndarray] = []
    for row in records:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        if gt is None or sid not in baseline_vectors:
            continue
        vector = np.asarray(baseline_vectors[sid], dtype=np.float64)
        by_relation[gt].append(vector)
        all_vectors.append(vector)
    missing = [relation for relation in RELATIONS if not by_relation[relation]]
    if missing:
        raise RuntimeError(
            f"Axis-training split lacks correct samples for relations {missing}"
        )
    center = np.mean(np.stack(all_vectors, axis=0), axis=0)
    centroids = {
        relation: np.mean(np.stack(by_relation[relation], axis=0), axis=0)
        for relation in RELATIONS
    }
    directions = {
        relation: unit_vector(centroids[relation] - center)
        for relation in RELATIONS
    }
    horizontal = unit_vector(centroids["left"] - centroids["right"])
    vertical = unit_vector(centroids["above"] - centroids["below"])
    return AxisModel(
        center=np.asarray(center, dtype=np.float64),
        class_centroids=centroids,
        class_directions=directions,
        horizontal=horizontal,
        vertical=vertical,
    )


def gt_axis_direction(model: AxisModel, relation: str) -> np.ndarray:
    if relation == "left":
        return model.horizontal
    if relation == "right":
        return -model.horizontal
    if relation == "above":
        return model.vertical
    if relation == "below":
        return -model.vertical
    raise KeyError(relation)


def relation_contrast_direction(
    model: AxisModel,
    positive: str,
    negative: str,
) -> np.ndarray:
    return unit_vector(
        model.class_centroids[positive] - model.class_centroids[negative]
    )


def channel_scores(model: AxisModel, vector: np.ndarray) -> Dict[str, float]:
    centered = np.asarray(vector, dtype=np.float64) - model.center
    centered_unit = unit_vector(centered)
    return {
        relation: float(np.dot(centered_unit, model.class_directions[relation]))
        for relation in RELATIONS
    }


def channel_prediction_and_margin(
    scores: Mapping[str, float],
) -> Tuple[str, float]:
    ranked = sorted(
        RELATIONS,
        key=lambda relation: float(scores[relation]),
        reverse=True,
    )
    return ranked[0], float(scores[ranked[0]] - scores[ranked[1]])


# -----------------------------------------------------------------------------
# Summarization
# -----------------------------------------------------------------------------


def build_projected_rows(
    *,
    args: argparse.Namespace,
    selected_rows: Sequence[Mapping[str, Any]],
    baseline_by_sid: Mapping[int, Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    vector_dir: Path,
    scan_heads: Sequence[ScanHead],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, Dict[str, Any]],
    Dict[str, Dict[str, float]],
]:
    selected_sids = {int(row["sid"]) for row in selected_rows}
    raw_unique = deduplicate_rows(raw_rows, ("sid", "head"))
    raw_by_key = {
        (int(row["sid"]), str(row["head"])): dict(row)
        for row in raw_unique
        if int(row["sid"]) in selected_sids
    }

    baseline_vectors: Dict[int, np.ndarray] = {}
    delta_vectors: Dict[int, np.ndarray] = {}
    vector_head_names: Dict[int, List[str]] = {}
    for sid in sorted(selected_sids):
        path = vector_dir / f"sid_{sid:06d}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing vector file for SID {sid}: {path}")
        with np.load(path, allow_pickle=False) as data:
            baseline_vectors[sid] = np.asarray(
                data["baseline_role"], dtype=np.float64
            )
            delta_vectors[sid] = np.asarray(data["delta_role"], dtype=np.float64)
            vector_head_names[sid] = [str(x) for x in data["head_names"].tolist()]

    correct_records = [
        {
            "sid": int(row["sid"]),
            "gt": baseline_by_sid[int(row["sid"])]["gt"],
        }
        for row in selected_rows
        if bool(baseline_by_sid[int(row["sid"])]["correct"])
    ]
    if len(correct_records) < max(8, int(args.axis_folds) * 2):
        raise RuntimeError(
            f"Only {len(correct_records)} correct samples; insufficient for "
            f"{args.axis_folds}-fold relation-axis fitting"
        )

    folds = {
        sid: stable_fold(sid, int(args.axis_folds), int(args.seed))
        for sid in selected_sids
    }
    axis_models: Dict[int, AxisModel] = {}
    for fold in range(int(args.axis_folds)):
        train = [
            row
            for row in correct_records
            if folds[int(row["sid"])] != fold
        ]
        axis_models[fold] = fit_axis_model(train, baseline_vectors)

    canonical_names = {head.name for head in scan_heads if head.canonical}
    projected: List[Dict[str, Any]] = []
    sample_meta: Dict[int, Dict[str, Any]] = {}

    for source_row in selected_rows:
        sid = int(source_row["sid"])
        baseline = baseline_by_sid[sid]
        gt = str(baseline["gt"])
        prediction = normalize_relation(baseline.get("prediction"))
        model = axis_models[folds[sid]]

        base_scores = channel_scores(model, baseline_vectors[sid])
        channel_prediction, channel_margin = channel_prediction_and_margin(base_scores)
        sample_meta[sid] = {
            "sid": sid,
            "gt": gt,
            "generation_prediction": prediction,
            "baseline_correct": bool(baseline["correct"]),
            "baseline_parsed": bool(baseline.get("parsed", prediction is not None)),
            "status": source_status(source_row, baseline),
            "status_alias": STATUS_ALIASES.get(
                source_status(source_row, baseline),
                source_status(source_row, baseline),
            ),
            "channel_prediction": channel_prediction,
            "channel_margin": float(channel_margin),
            "channel_correct": bool(channel_prediction == gt),
            "channel_agrees_generation": bool(
                prediction is not None and channel_prediction == prediction
            ),
            **{
                f"channel_score_{relation}": float(base_scores[relation])
                for relation in RELATIONS
            },
        }

        names = vector_head_names[sid]
        vectors = delta_vectors[sid]
        if vectors.shape[0] != len(names):
            raise RuntimeError(f"SID {sid}: vector/head count mismatch")
        name_to_index = {name: index for index, name in enumerate(names)}
        missing = [head.name for head in scan_heads if head.name not in name_to_index]
        if missing:
            raise RuntimeError(f"SID {sid}: vector file lacks heads {missing[:8]}")

        gt_direction = gt_axis_direction(model, gt)
        pred_direction = None
        if prediction is not None and prediction != gt:
            pred_direction = relation_contrast_direction(model, prediction, gt)

        for head in scan_heads:
            key = (sid, head.name)
            if key not in raw_by_key:
                raise RuntimeError(f"Missing raw row SID={sid} head={head.name}")
            raw = dict(raw_by_key[key])
            delta = np.asarray(
                vectors[name_to_index[head.name]], dtype=np.float64
            )
            transport_norm = float(np.linalg.norm(delta))
            gt_projection = float(np.dot(delta, gt_direction))
            gt_cosine = (
                gt_projection / transport_norm
                if transport_norm > 1e-12
                else 0.0
            )
            if pred_direction is None:
                pred_over_gt_projection = 0.0
                pred_over_gt_cosine = 0.0
            else:
                pred_over_gt_projection = float(np.dot(delta, pred_direction))
                pred_over_gt_cosine = (
                    pred_over_gt_projection / transport_norm
                    if transport_norm > 1e-12
                    else 0.0
                )

            row = {
                **raw,
                "canonical": bool(head.name in canonical_names),
                "axis_fold": int(folds[sid]),
                "state_transport_norm": transport_norm,
                "state_GT_axis_projection": gt_projection,
                "state_GT_axis_cosine": float(gt_cosine),
                "state_abs_GT_axis_projection": abs(gt_projection),
                "state_pred_over_GT_projection": pred_over_gt_projection,
                "state_pred_over_GT_cosine": float(pred_over_gt_cosine),
                "state_supports_GT": bool(
                    gt_projection > float(args.effect_epsilon)
                ),
                "state_opposes_GT": bool(
                    gt_projection < -float(args.effect_epsilon)
                ),
                "state_supports_wrong_prediction_over_GT": bool(
                    prediction is not None
                    and prediction != gt
                    and pred_over_gt_projection > float(args.effect_epsilon)
                ),
                "baseline_correct": bool(baseline["correct"]),
                "generation_prediction": prediction,
                "channel_prediction": channel_prediction,
                "channel_correct": bool(channel_prediction == gt),
                "channel_agrees_generation": bool(
                    prediction is not None and channel_prediction == prediction
                ),
                "channel_margin": float(channel_margin),
                "status": sample_meta[sid]["status"],
                "status_alias": sample_meta[sid]["status_alias"],
            }
            projected.append(row)

    # Sample-wise ranks.
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in projected:
        grouped[int(row["sid"])].append(row)
    for sid, values in grouped.items():
        transport_ranks = rank_desc(
            [float(row["state_transport_norm"]) for row in values]
        )
        relation_ranks = rank_desc(
            [float(row["state_abs_GT_axis_projection"]) for row in values]
        )
        wrong_ranks = rank_desc(
            [
                max(float(row["state_pred_over_GT_projection"]), 0.0)
                for row in values
            ]
        )
        for index, row in enumerate(values):
            row["transport_rank"] = float(transport_ranks[index])
            row["relation_rank"] = float(relation_ranks[index])
            row["wrong_support_rank"] = float(wrong_ranks[index])

    # Correct-sample reference distribution per head.
    by_head_correct: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in projected:
        if bool(row["baseline_correct"]):
            by_head_correct[str(row["head"])].append(row)

    head_reference: Dict[str, Dict[str, float]] = {}
    n_heads = len(scan_heads)
    weak_cutoff = float(args.weak_rank_fraction) * float(n_heads)
    for head in scan_heads:
        values = by_head_correct.get(head.name, [])
        if not values:
            raise RuntimeError(f"No correct reference rows for {head.name}")
        transport = [float(row["state_transport_norm"]) for row in values]
        relation_abs = [
            float(row["state_abs_GT_axis_projection"]) for row in values
        ]
        transport_ranks = [float(row["transport_rank"]) for row in values]
        relation_ranks = [float(row["relation_rank"]) for row in values]
        transport_median_rank = safe_median(transport_ranks)
        relation_median_rank = safe_median(relation_ranks)
        head_reference[head.name] = {
            "correct_transport_median": safe_median(transport),
            "correct_transport_q": safe_quantile(transport, float(args.emergent_q)),
            "correct_relation_abs_median": safe_median(relation_abs),
            "correct_relation_abs_q": safe_quantile(
                relation_abs, float(args.emergent_q)
            ),
            "correct_transport_median_rank": transport_median_rank,
            "correct_relation_median_rank": relation_median_rank,
            "weak_transport_on_correct": float(
                math.isfinite(transport_median_rank)
                and transport_median_rank > weak_cutoff
            ),
            "weak_relation_on_correct": float(
                math.isfinite(relation_median_rank)
                and relation_median_rank > weak_cutoff
            ),
        }

    for row in projected:
        ref = head_reference[str(row["head"])]
        transport_q = max(float(ref["correct_transport_q"]), 1e-12)
        relation_q = max(float(ref["correct_relation_abs_q"]), 1e-12)
        row["correct_transport_reference_median"] = float(
            ref["correct_transport_median"]
        )
        row["correct_transport_reference_q"] = float(ref["correct_transport_q"])
        row["correct_relation_reference_median"] = float(
            ref["correct_relation_abs_median"]
        )
        row["correct_relation_reference_q"] = float(
            ref["correct_relation_abs_q"]
        )
        row["weak_transport_on_correct"] = bool(
            ref["weak_transport_on_correct"] > 0.5
        )
        row["weak_relation_on_correct"] = bool(
            ref["weak_relation_on_correct"] > 0.5
        )
        row["transport_strength_ratio_to_correct_q"] = float(
            float(row["state_transport_norm"]) / transport_q
        )
        row["relation_strength_ratio_to_correct_q"] = float(
            float(row["state_abs_GT_axis_projection"]) / relation_q
        )

        wrong = not bool(row["baseline_correct"])
        top_k = int(args.top_k)
        row["emergent_transport_head"] = bool(
            wrong
            and bool(row["weak_transport_on_correct"])
            and (
                float(row["transport_rank"]) <= top_k
                or float(row["state_transport_norm"])
                > float(ref["correct_transport_q"])
            )
        )
        row["emergent_wrong_direction_head"] = bool(
            wrong
            and bool(row["weak_relation_on_correct"])
            and bool(row["state_supports_wrong_prediction_over_GT"])
            and (
                float(row["wrong_support_rank"]) <= top_k
                or float(row["state_pred_over_GT_projection"])
                > float(ref["correct_relation_abs_q"])
            )
        )

    return projected, sample_meta, head_reference


def summarize_heads(
    projected: Sequence[Mapping[str, Any]],
    scan_heads: Sequence[ScanHead],
) -> List[Dict[str, Any]]:
    by_head: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in projected:
        by_head[str(row["head"])].append(row)

    output: List[Dict[str, Any]] = []
    for head in scan_heads:
        values = by_head[head.name]
        correct = [row for row in values if bool(row["baseline_correct"])]
        wrong = [row for row in values if not bool(row["baseline_correct"])]

        correct_transport = [
            float(row["state_transport_norm"]) for row in correct
        ]
        wrong_transport = [
            float(row["state_transport_norm"]) for row in wrong
        ]
        correct_relation = [
            float(row["state_abs_GT_axis_projection"]) for row in correct
        ]
        wrong_relation = [
            float(row["state_abs_GT_axis_projection"]) for row in wrong
        ]

        output.append(
            {
                "head": head.name,
                "layer": int(head.layer),
                "head_index": int(head.head),
                "canonical": bool(head.canonical),
                "N": len(values),
                "correct_N": len(correct),
                "wrong_N": len(wrong),
                "correct_transport_mean": safe_mean(correct_transport),
                "wrong_transport_mean": safe_mean(wrong_transport),
                "wrong_over_correct_transport_ratio": (
                    safe_mean(wrong_transport)
                    / max(safe_mean(correct_transport), 1e-12)
                ),
                "transport_shift_cohens_d_wrong_minus_correct": cohens_d(
                    correct_transport, wrong_transport
                ),
                "correct_transport_median_rank": safe_median(
                    float(row["transport_rank"]) for row in correct
                ),
                "wrong_transport_median_rank": safe_median(
                    float(row["transport_rank"]) for row in wrong
                ),
                "correct_relation_abs_mean": safe_mean(correct_relation),
                "wrong_relation_abs_mean": safe_mean(wrong_relation),
                "wrong_over_correct_relation_abs_ratio": (
                    safe_mean(wrong_relation)
                    / max(safe_mean(correct_relation), 1e-12)
                ),
                "relation_shift_cohens_d_wrong_minus_correct": cohens_d(
                    correct_relation, wrong_relation
                ),
                "correct_relation_median_rank": safe_median(
                    float(row["relation_rank"]) for row in correct
                ),
                "wrong_relation_median_rank": safe_median(
                    float(row["relation_rank"]) for row in wrong
                ),
                "correct_supports_GT_rate": safe_mean(
                    int(bool(row["state_supports_GT"])) for row in correct
                ),
                "wrong_opposes_GT_rate": safe_mean(
                    int(bool(row["state_opposes_GT"])) for row in wrong
                ),
                "wrong_supports_generated_over_GT_rate": safe_mean(
                    int(bool(row["state_supports_wrong_prediction_over_GT"]))
                    for row in wrong
                ),
                "weak_transport_on_correct": bool(
                    values[0]["weak_transport_on_correct"]
                ),
                "weak_relation_on_correct": bool(
                    values[0]["weak_relation_on_correct"]
                ),
                "emergent_transport_wrong_count": sum(
                    int(bool(row["emergent_transport_head"])) for row in wrong
                ),
                "emergent_transport_wrong_rate": safe_mean(
                    int(bool(row["emergent_transport_head"])) for row in wrong
                ),
                "emergent_wrong_direction_count": sum(
                    int(bool(row["emergent_wrong_direction_head"]))
                    for row in wrong
                ),
                "emergent_wrong_direction_rate": safe_mean(
                    int(bool(row["emergent_wrong_direction_head"]))
                    for row in wrong
                ),
                "wrong_mean_pred_over_GT_projection": safe_mean(
                    float(row["state_pred_over_GT_projection"]) for row in wrong
                ),
                "wrong_mean_GT_axis_projection": safe_mean(
                    float(row["state_GT_axis_projection"]) for row in wrong
                ),
            }
        )
    return output


def classify_failure(
    *,
    args: argparse.Namespace,
    baseline_correct: bool,
    channel_correct: bool,
    channel_agrees_generation: bool,
    canonical_relation_share: float,
    canonical_wrong_fraction: float,
    off_wrong_share: float,
    emergent_wrong_share: float,
    entropy: float,
    topk_mass: float,
    cancellation: float,
    total_relation_mass: float,
    low_evidence_threshold: float,
) -> str:
    if baseline_correct:
        return "correct"
    if channel_correct:
        return "downstream_or_writer_flip"
    if channel_agrees_generation:
        if (
            canonical_relation_share >= float(args.canonical_share_threshold)
            and canonical_wrong_fraction
            >= float(args.canonical_wrong_fraction_threshold)
        ):
            return "canonical_route_sign_reversal"
        if (
            off_wrong_share >= float(args.off_takeover_threshold)
            and emergent_wrong_share
            >= float(args.emergent_wrong_share_threshold)
        ):
            return "off_circuit_weak_head_takeover"
        if (
            entropy >= float(args.diffuse_entropy_threshold)
            and topk_mass <= float(args.diffuse_topk_threshold)
            and cancellation <= float(args.cancellation_threshold)
        ):
            return "diffuse_conflict"
        if total_relation_mass <= low_evidence_threshold:
            return "low_relation_evidence"
        return "upstream_wrong_mixed_route"
    if (
        entropy >= float(args.diffuse_entropy_threshold)
        and topk_mass <= float(args.diffuse_topk_threshold)
    ):
        return "channel_generation_disagreement_diffuse"
    return "channel_generation_disagreement"


def summarize_samples(
    *,
    args: argparse.Namespace,
    projected: Sequence[Mapping[str, Any]],
    sample_meta: Mapping[int, Mapping[str, Any]],
    scan_heads: Sequence[ScanHead],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_sid: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in projected:
        by_sid[int(row["sid"])].append(row)

    canonical_names = {head.name for head in scan_heads if head.canonical}
    preliminary: List[Dict[str, Any]] = []
    emergence_rows: List[Dict[str, Any]] = []

    for sid, values in sorted(by_sid.items()):
        meta = dict(sample_meta[sid])
        abs_relation = np.asarray(
            [float(row["state_abs_GT_axis_projection"]) for row in values],
            dtype=np.float64,
        )
        signed_relation = np.asarray(
            [float(row["state_GT_axis_projection"]) for row in values],
            dtype=np.float64,
        )
        transport = np.asarray(
            [float(row["state_transport_norm"]) for row in values],
            dtype=np.float64,
        )
        wrong_support = np.asarray(
            [
                max(float(row["state_pred_over_GT_projection"]), 0.0)
                for row in values
            ],
            dtype=np.float64,
        )
        canonical_mask = np.asarray(
            [str(row["head"]) in canonical_names for row in values],
            dtype=bool,
        )

        relation_total = float(abs_relation.sum())
        transport_total = float(transport.sum())
        wrong_total = float(wrong_support.sum())

        canonical_relation_mass = float(abs_relation[canonical_mask].sum())
        off_relation_mass = float(abs_relation[~canonical_mask].sum())
        canonical_transport_mass = float(transport[canonical_mask].sum())
        off_transport_mass = float(transport[~canonical_mask].sum())
        canonical_wrong_mass = float(wrong_support[canonical_mask].sum())
        off_wrong_mass = float(wrong_support[~canonical_mask].sum())

        canonical_negative_mass = float(
            np.clip(-signed_relation[canonical_mask], 0.0, None).sum()
        )
        canonical_signed_abs = float(abs_relation[canonical_mask].sum())
        canonical_wrong_fraction = (
            canonical_negative_mass / canonical_signed_abs
            if canonical_signed_abs > 0
            else 0.0
        )

        sorted_relation = np.sort(abs_relation)[::-1]
        k = min(int(args.top_k), len(sorted_relation))
        topk_relation_mass = (
            float(sorted_relation[:k].sum() / relation_total)
            if relation_total > 0
            else 0.0
        )
        entropy = normalized_entropy(abs_relation)
        cancellation = (
            abs(float(signed_relation.sum())) / relation_total
            if relation_total > 0
            else 0.0
        )

        emergent_transport = [
            row for row in values if bool(row["emergent_transport_head"])
        ]
        emergent_wrong = [
            row for row in values if bool(row["emergent_wrong_direction_head"])
        ]
        emergent_wrong_mass = sum(
            max(float(row["state_pred_over_GT_projection"]), 0.0)
            for row in emergent_wrong
        )
        emergent_wrong_share = (
            float(emergent_wrong_mass / wrong_total)
            if wrong_total > 0
            else 0.0
        )

        top_relation_heads = sorted(
            values,
            key=lambda row: float(row["state_abs_GT_axis_projection"]),
            reverse=True,
        )[:k]
        top_transport_heads = sorted(
            values,
            key=lambda row: float(row["state_transport_norm"]),
            reverse=True,
        )[:k]
        top_wrong_heads = sorted(
            values,
            key=lambda row: float(row["state_pred_over_GT_projection"]),
            reverse=True,
        )[:k]

        row = {
            **meta,
            "head_count": len(values),
            "relation_total_abs_mass": relation_total,
            "relation_signed_sum": float(signed_relation.sum()),
            "relation_support_GT_mass": float(
                np.clip(signed_relation, 0.0, None).sum()
            ),
            "relation_oppose_GT_mass": float(
                np.clip(-signed_relation, 0.0, None).sum()
            ),
            "transport_total_mass": transport_total,
            "canonical_relation_share": (
                canonical_relation_mass / relation_total
                if relation_total > 0
                else 0.0
            ),
            "off_circuit_relation_share": (
                off_relation_mass / relation_total
                if relation_total > 0
                else 0.0
            ),
            "canonical_transport_share": (
                canonical_transport_mass / transport_total
                if transport_total > 0
                else 0.0
            ),
            "off_circuit_transport_share": (
                off_transport_mass / transport_total
                if transport_total > 0
                else 0.0
            ),
            "canonical_wrong_support_share": (
                canonical_wrong_mass / wrong_total if wrong_total > 0 else 0.0
            ),
            "off_circuit_wrong_support_share": (
                off_wrong_mass / wrong_total if wrong_total > 0 else 0.0
            ),
            "canonical_wrong_direction_fraction": canonical_wrong_fraction,
            "canonical_sign_flip_head_fraction": safe_mean(
                int(bool(row["state_opposes_GT"]))
                for row in values
                if bool(row["canonical"])
            ),
            "routing_topk_mass": topk_relation_mass,
            "routing_entropy": entropy,
            "routing_cancellation_ratio": cancellation,
            "emergent_transport_head_count": len(emergent_transport),
            "emergent_wrong_direction_head_count": len(emergent_wrong),
            "emergent_wrong_direction_mass_share": emergent_wrong_share,
            "top_relation_heads": ",".join(
                str(row["head"]) for row in top_relation_heads
            ),
            "top_transport_heads": ",".join(
                str(row["head"]) for row in top_transport_heads
            ),
            "top_wrong_support_heads": ",".join(
                str(row["head"]) for row in top_wrong_heads
            ),
            "emergent_transport_heads": ",".join(
                str(row["head"]) for row in emergent_transport
            ),
            "emergent_wrong_direction_heads": ",".join(
                str(row["head"]) for row in emergent_wrong
            ),
        }
        preliminary.append(row)

        for head_row in emergent_wrong:
            emergence_rows.append(
                {
                    "sid": sid,
                    "gt": meta["gt"],
                    "generation_prediction": meta["generation_prediction"],
                    "status_alias": meta["status_alias"],
                    "head": head_row["head"],
                    "layer": head_row["head_layer"],
                    "head_index": head_row["head_index"],
                    "canonical": head_row["canonical"],
                    "state_transport_norm": head_row["state_transport_norm"],
                    "transport_rank": head_row["transport_rank"],
                    "state_GT_axis_projection": head_row[
                        "state_GT_axis_projection"
                    ],
                    "state_pred_over_GT_projection": head_row[
                        "state_pred_over_GT_projection"
                    ],
                    "wrong_support_rank": head_row["wrong_support_rank"],
                    "correct_relation_reference_q": head_row[
                        "correct_relation_reference_q"
                    ],
                    "relation_strength_ratio_to_correct_q": head_row[
                        "relation_strength_ratio_to_correct_q"
                    ],
                }
            )

    correct_masses = [
        float(row["relation_total_abs_mass"])
        for row in preliminary
        if bool(row["baseline_correct"])
    ]
    low_threshold = safe_quantile(correct_masses, 0.25)

    for row in preliminary:
        row["heuristic_failure_class"] = classify_failure(
            args=args,
            baseline_correct=bool(row["baseline_correct"]),
            channel_correct=bool(row["channel_correct"]),
            channel_agrees_generation=bool(row["channel_agrees_generation"]),
            canonical_relation_share=float(row["canonical_relation_share"]),
            canonical_wrong_fraction=float(
                row["canonical_wrong_direction_fraction"]
            ),
            off_wrong_share=float(row["off_circuit_wrong_support_share"]),
            emergent_wrong_share=float(
                row["emergent_wrong_direction_mass_share"]
            ),
            entropy=float(row["routing_entropy"]),
            topk_mass=float(row["routing_topk_mass"]),
            cancellation=float(row["routing_cancellation_ratio"]),
            total_relation_mass=float(row["relation_total_abs_mass"]),
            low_evidence_threshold=float(low_threshold),
        )
        row["low_evidence_threshold_correct_q25"] = float(low_threshold)

    return preliminary, emergence_rows


def summarize_groups(
    sample_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {
        "all": list(sample_rows),
        "correct": [
            row for row in sample_rows if bool(row["baseline_correct"])
        ],
        "wrong": [
            row for row in sample_rows if not bool(row["baseline_correct"])
        ],
    }
    for alias in ("CC", "CW", "WC", "WW"):
        groups[alias] = [
            row for row in sample_rows if str(row["status_alias"]) == alias
        ]

    metrics = (
        "channel_correct",
        "channel_agrees_generation",
        "channel_margin",
        "relation_total_abs_mass",
        "relation_signed_sum",
        "relation_support_GT_mass",
        "relation_oppose_GT_mass",
        "canonical_relation_share",
        "off_circuit_relation_share",
        "canonical_transport_share",
        "off_circuit_transport_share",
        "canonical_wrong_support_share",
        "off_circuit_wrong_support_share",
        "canonical_wrong_direction_fraction",
        "canonical_sign_flip_head_fraction",
        "routing_topk_mass",
        "routing_entropy",
        "routing_cancellation_ratio",
        "emergent_transport_head_count",
        "emergent_wrong_direction_head_count",
        "emergent_wrong_direction_mass_share",
    )

    output: List[Dict[str, Any]] = []
    for name, values in groups.items():
        if not values:
            continue
        row: Dict[str, Any] = {
            "group": name,
            "N": len(values),
            "baseline_accuracy": safe_mean(
                int(bool(value["baseline_correct"])) for value in values
            ),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = safe_mean(
                float(value[metric]) for value in values
            )
            row[f"{metric}_median"] = safe_median(
                float(value[metric]) for value in values
            )
        output.append(row)
    return output


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.axis_folds < 2:
        raise ValueError("--axis-folds must be >= 2")
    if not 0.0 < args.weak_rank_fraction < 1.0:
        raise ValueError("--weak-rank-fraction must be in (0,1)")
    if not 0.5 < args.emergent_q < 1.0:
        raise ValueError("--emergent-q must be in (0.5,1)")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    failure = import_file(Path(args.failure_script), "signed_routing_failure")
    ioi = import_file(Path(args.ioi_script), "signed_routing_ioi")
    producer = import_file(Path(args.producer_script), "signed_routing_producer")
    receiver = import_file(Path(args.receiver_script), "signed_routing_receiver")
    v3 = import_file(Path(args.v3_script), "signed_routing_v3")
    base = import_file(Path(args.base_script), "signed_routing_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "signed_routing_attention",
    )

    source_config, source_rows = ioi.load_source_rows(args)
    baseline_path = resolve_baseline_path(args)
    baseline_by_sid = load_baseline_rows(baseline_path)

    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(str(args.include_sids_file).strip()))
    excluded: set[int] = set()
    for raw in str(args.exclude_sids_from).split(","):
        item = raw.strip()
        if item:
            excluded.update(extract_sids(Path(item)))

    selected_rows = [
        dict(row)
        for row in source_rows
        if (
            args.sample_status == "all"
            or str(row.get("generation_pair_status")) == args.sample_status
        )
        and int(row["sid"]) in baseline_by_sid
        and int(row["sid"]) not in excluded
        and (included is None or int(row["sid"]) in included)
    ]
    selected_rows = sorted(selected_rows, key=lambda row: int(row["sid"]))
    if args.sample_max_samples > 0:
        rng = random.Random(args.seed)
        by_status: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in selected_rows:
            sid = int(row["sid"])
            label = (
                f"{source_status(row, baseline_by_sid[sid])}:"
                f"{int(bool(baseline_by_sid[sid]['correct']))}"
            )
            by_status[label].append(row)
        for values in by_status.values():
            rng.shuffle(values)
        limited: List[Dict[str, Any]] = []
        labels = sorted(by_status)
        cursors = {label: 0 for label in labels}
        while len(limited) < int(args.sample_max_samples):
            progressed = False
            for label in labels:
                index = cursors[label]
                if index < len(by_status[label]):
                    limited.append(by_status[label][index])
                    cursors[label] += 1
                    progressed = True
                    if len(limited) >= int(args.sample_max_samples):
                        break
            if not progressed:
                break
        selected_rows = sorted(limited, key=lambda row: int(row["sid"]))

    if not selected_rows:
        raise RuntimeError("No eligible samples after baseline/source intersection")

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

        bundles = load_bundles(Path(args.bundle_json))
        if args.canonical_bundle not in bundles:
            raise KeyError(
                f"Missing canonical bundle {args.canonical_bundle}; "
                f"available={sorted(bundles)}"
            )
        canonical_heads = set(bundles[args.canonical_bundle])

        scan_heads = build_scan_heads(
            args=args,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            ioi=ioi,
            canonical_heads=canonical_heads,
        )
        head_names = [head.name for head in scan_heads]

        writer = ioi.WriterNode(
            "attention",
            int(args.receiver_layer),
            int(args.receiver_query_head),
        )
        units = ioi.build_receiver_units(
            writers=[writer],
            channels=[str(args.receiver_channel)],
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
        )
        if len(units) != 1:
            raise RuntimeError(f"Expected one receiver unit, got {len(units)}")
        unit = units[0]

        attention = attention_helper.resolve_self_attention(
            decoder_layers[int(unit.layer)]
        )
        shape = receiver.resolve_attention_shape(attention)
        head_dim = int(shape.kv_head_dim)

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "baseline_generation_jsonl": str(baseline_path),
            "decoder_path": decoder_path,
            "scan_layers": args.scan_layers,
            "scan_heads": head_names,
            "scan_head_count": len(scan_heads),
            "canonical_bundle": args.canonical_bundle,
            "canonical_heads": sorted(
                head_name(layer, head) for layer, head in canonical_heads
            ),
            "receiver": {
                "layer": int(unit.layer),
                "channel": str(unit.channel),
                "kv_head": int(unit.kv_head),
                "unit_head": int(unit.unit_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "head_dim": head_dim,
            },
            "sender_position_mode": args.sender_position_mode,
            "axis_folds": args.axis_folds,
            "top_k": args.top_k,
            "weak_rank_fraction": args.weak_rank_fraction,
            "emergent_q": args.emergent_q,
            "exact_score_effects": args.exact_score_effects,
            "selected_samples": len(selected_rows),
            "selected_sids": [int(row["sid"]) for row in selected_rows],
            "audit": audit,
            "transformers_version": transformers.__version__,
            "interpretation": (
                "C-pass state deltas measure path-specific transport into "
                "L26VH0. Cross-fitted GT-oriented receiver axes measure signed "
                "relation direction. Failure classes are descriptive heuristics."
            ),
        }
        write_json(output_dir / "config.json", config)

        raw_path = output_dir / "raw_head_routing.jsonl"
        clean_path = output_dir / "clean_sample_states.jsonl"
        errors_path = output_dir / "errors.jsonl"

        raw_rows = read_jsonl(raw_path) if args.resume else []
        clean_rows = read_jsonl(clean_path) if args.resume else []
        raw_unique = deduplicate_rows(raw_rows, ("sid", "head"))
        clean_unique = deduplicate_rows(clean_rows, ("sid",))
        raw_count_by_sid = Counter(int(row["sid"]) for row in raw_unique)
        clean_done = {int(row["sid"]) for row in clean_unique}

        selected_sids = [int(row["sid"]) for row in selected_rows]
        done_sids = {
            sid
            for sid in selected_sids
            if raw_count_by_sid[sid] == len(scan_heads)
            and sid in clean_done
            and (vector_dir / f"sid_{sid:06d}.npz").exists()
        }
        pending_rows = [
            row for row in selected_rows if int(row["sid"]) not in done_sids
        ]

        print(
            "Signed routing scan: "
            f"samples={len(selected_rows)}, pending={len(pending_rows)}, "
            f"heads={len(scan_heads)}, C-passes="
            f"{len(selected_rows) * len(scan_heads)}, "
            f"exact_score_effects={args.exact_score_effects}",
            flush=True,
        )

        capture_layers = list(range(int(args.receiver_layer) + 1))

        for sample_index, source_row in enumerate(
            tqdm(pending_rows, desc=f"signed-routing:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                sid = int(source_row["sid"])
                baseline = baseline_by_sid[sid]
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

                if not pair.original_a_positions or not pair.original_b_positions:
                    raise RuntimeError("Missing original subject/reference positions")
                subject_position = int(pair.original_a_positions[-1])
                reference_position = int(pair.original_b_positions[-1])

                if args.sender_position_mode == "joint":
                    sender_positions = sorted(
                        {subject_position, reference_position}
                    )
                elif args.sender_position_mode == "subject":
                    sender_positions = [subject_position]
                else:
                    sender_positions = [reference_position]

                receiver_positions = sorted(
                    set(map(int, pair.original_object_positions))
                    | {subject_position, reference_position}
                )
                attention_positions = sorted(
                    set(sender_positions) | set(receiver_positions)
                )

                clean_scores, clean_capture, baseline_states = (
                    failure.capture_clean_original(
                        pair=pair,
                        capture_layers=capture_layers,
                        attention_positions=attention_positions,
                        receiver_positions=receiver_positions,
                        unit=unit,
                        model=model,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        base=base,
                        receiver_module=receiver,
                        attention_helper=attention_helper,
                        ioi=ioi,
                    )
                )
                baseline_v = baseline_states[int(unit.layer)][unit.channel]
                baseline_subject = state_head_slice(
                    baseline_v[subject_position],
                    unit_head=int(unit.unit_head),
                    head_dim=head_dim,
                )
                baseline_reference = state_head_slice(
                    baseline_v[reference_position],
                    unit_head=int(unit.unit_head),
                    head_dim=head_dim,
                )
                baseline_role = baseline_subject - baseline_reference

                sample_raw_rows: List[Dict[str, Any]] = []
                delta_subject_values: List[np.ndarray] = []
                delta_reference_values: List[np.ndarray] = []
                delta_role_values: List[np.ndarray] = []

                clean_score_map = {
                    relation: float(clean_scores["scores"][relation])
                    for relation in RELATIONS
                }

                for scan_head in scan_heads:
                    bundle = failure.HeadBundle(
                        name=scan_head.name,
                        heads=(scan_head.node,),
                    )
                    ablated_states = failure.run_bundle_removal_c_pass(
                        bundle=bundle,
                        sender_positions=sender_positions,
                        receiver_positions=receiver_positions,
                        unit=unit,
                        pair=pair,
                        clean_capture=clean_capture,
                        model=model,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        base=base,
                        receiver_module=receiver,
                        attention_helper=attention_helper,
                        ioi=ioi,
                    )
                    ablated_v = ablated_states[int(unit.layer)][unit.channel]
                    ablated_subject = state_head_slice(
                        ablated_v[subject_position],
                        unit_head=int(unit.unit_head),
                        head_dim=head_dim,
                    )
                    ablated_reference = state_head_slice(
                        ablated_v[reference_position],
                        unit_head=int(unit.unit_head),
                        head_dim=head_dim,
                    )
                    delta_subject = baseline_subject - ablated_subject
                    delta_reference = baseline_reference - ablated_reference
                    delta_role = delta_subject - delta_reference

                    delta_subject_values.append(delta_subject)
                    delta_reference_values.append(delta_reference)
                    delta_role_values.append(delta_role)

                    row: Dict[str, Any] = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": baseline["gt"],
                        "generation_prediction": baseline["prediction"],
                        "baseline_correct": bool(baseline["correct"]),
                        "head": scan_head.name,
                        "head_layer": int(scan_head.layer),
                        "head_index": int(scan_head.head),
                        "canonical": bool(scan_head.canonical),
                        "sender_position_mode": args.sender_position_mode,
                        "subject_position": subject_position,
                        "reference_position": reference_position,
                        "sender_positions": sender_positions,
                        "receiver_positions": receiver_positions,
                        "receiver_unit": unit.unit,
                        "delta_subject_norm": float(
                            np.linalg.norm(delta_subject)
                        ),
                        "delta_reference_norm": float(
                            np.linalg.norm(delta_reference)
                        ),
                        "delta_role_norm": float(np.linalg.norm(delta_role)),
                        "clean_closed_prediction": normalize_relation(
                            clean_scores.get("prediction")
                        ),
                        "clean_closed_scores": clean_score_map,
                        "clean_GT_axis_margin": relation_margin(
                            clean_score_map, baseline["gt"]
                        ),
                        "status": source_status(source_row, baseline),
                        "status_alias": STATUS_ALIASES.get(
                            source_status(source_row, baseline),
                            source_status(source_row, baseline),
                        ),
                        "path_definition": (
                            "sender head at selected object positions -> "
                            "residual/token-wise MLP -> L26VH0"
                        ),
                    }

                    if args.exact_score_effects:
                        ablated_scores = failure.run_receiver_state(
                            unit=unit,
                            full_states_by_position=ablated_v,
                            pair=pair,
                            model=model,
                            decoder_layers=decoder_layers,
                            relation_token_map=relation_token_map,
                            base=base,
                            receiver_module=receiver,
                            attention_helper=attention_helper,
                        )
                        ablated_score_map = {
                            relation: float(ablated_scores["scores"][relation])
                            for relation in RELATIONS
                        }
                        contribution = {
                            relation: (
                                clean_score_map[relation]
                                - ablated_score_map[relation]
                            )
                            for relation in RELATIONS
                        }
                        gt = str(baseline["gt"])
                        prediction = normalize_relation(baseline["prediction"])
                        row.update(
                            {
                                "ablated_closed_prediction": normalize_relation(
                                    ablated_scores.get("prediction")
                                ),
                                "ablated_closed_scores": ablated_score_map,
                                "exact_score_contribution": contribution,
                                "exact_GT_axis_contribution": (
                                    contribution[gt]
                                    - contribution[OPPOSITE[gt]]
                                ),
                                "exact_prediction_over_GT_contribution": (
                                    float("nan")
                                    if prediction is None
                                    else contribution[prediction]
                                    - contribution[gt]
                                ),
                            }
                        )

                    sample_raw_rows.append(row)

                vector_path = vector_dir / f"sid_{sid:06d}.npz"
                np.savez_compressed(
                    vector_path,
                    head_names=np.asarray(head_names, dtype="U32"),
                    baseline_subject=np.asarray(
                        baseline_subject, dtype=np.float32
                    ),
                    baseline_reference=np.asarray(
                        baseline_reference, dtype=np.float32
                    ),
                    baseline_role=np.asarray(baseline_role, dtype=np.float32),
                    delta_subject=np.stack(
                        delta_subject_values, axis=0
                    ).astype(np.float32),
                    delta_reference=np.stack(
                        delta_reference_values, axis=0
                    ).astype(np.float32),
                    delta_role=np.stack(
                        delta_role_values, axis=0
                    ).astype(np.float32),
                )

                clean_row = {
                    "script_version": SCRIPT_VERSION,
                    "model": args.model,
                    "sid": sid,
                    "gt": baseline["gt"],
                    "generation_prediction": baseline["prediction"],
                    "baseline_correct": bool(baseline["correct"]),
                    "clean_closed_prediction": normalize_relation(
                        clean_scores.get("prediction")
                    ),
                    "clean_closed_scores": clean_score_map,
                    "subject_position": subject_position,
                    "reference_position": reference_position,
                    "sender_positions": sender_positions,
                    "receiver_positions": receiver_positions,
                    "vector_file": str(vector_path),
                    "status": source_status(source_row, baseline),
                    "status_alias": STATUS_ALIASES.get(
                        source_status(source_row, baseline),
                        source_status(source_row, baseline),
                    ),
                }
                append_jsonl(clean_path, clean_row)
                for row in sample_raw_rows:
                    append_jsonl(raw_path, row)

                raw_rows.extend(sample_raw_rows)
                clean_rows.append(clean_row)

                if args.print_every > 0 and sample_index % args.print_every == 0:
                    print(
                        f"[sample {sample_index}/{len(pending_rows)} sid={sid}] "
                        f"saved_heads={len(sample_raw_rows)}",
                        flush=True,
                    )

            except Exception as exc:
                error = {
                    "script_version": SCRIPT_VERSION,
                    "source_sid": int(source_row.get("sid", -1)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(errors_path, error)
                print(
                    f"[ERROR sid={source_row.get('sid')}] "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    release = getattr(receiver, "release_pair", None)
                    if callable(release):
                        try:
                            release(pair)
                        except Exception:
                            pass
                    del pair
                gc.collect()
                if (
                    args.device.startswith("cuda")
                    and args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        # Re-read complete outputs after possible resume.
        raw_rows = deduplicate_rows(read_jsonl(raw_path), ("sid", "head"))
        selected_sid_set = {int(row["sid"]) for row in selected_rows}
        complete_counts = Counter(
            int(row["sid"])
            for row in raw_rows
            if int(row["sid"]) in selected_sid_set
        )
        incomplete = [
            sid
            for sid in sorted(selected_sid_set)
            if complete_counts[sid] != len(scan_heads)
            or not (vector_dir / f"sid_{sid:06d}.npz").exists()
        ]
        if incomplete:
            raise RuntimeError(
                f"Incomplete samples after scan: {incomplete[:20]} "
                f"(total={len(incomplete)})"
            )

        projected, sample_meta, _ = build_projected_rows(
            args=args,
            selected_rows=selected_rows,
            baseline_by_sid=baseline_by_sid,
            raw_rows=raw_rows,
            vector_dir=vector_dir,
            scan_heads=scan_heads,
        )
        head_summary = summarize_heads(projected, scan_heads)
        sample_summary, emergence_rows = summarize_samples(
            args=args,
            projected=projected,
            sample_meta=sample_meta,
            scan_heads=scan_heads,
        )
        status_summary = summarize_groups(sample_summary)

        write_csv(output_dir / "signed_head_routing.csv", projected)
        write_csv(output_dir / "head_shift_summary.csv", head_summary)
        write_csv(output_dir / "sample_routing_summary.csv", sample_summary)
        write_csv(output_dir / "status_routing_summary.csv", status_summary)
        write_csv(output_dir / "weak_head_emergence.csv", emergence_rows)

        wrong_samples = [
            row for row in sample_summary if not bool(row["baseline_correct"])
        ]
        correct_samples = [
            row for row in sample_summary if bool(row["baseline_correct"])
        ]
        failure_counts = Counter(
            str(row["heuristic_failure_class"]) for row in sample_summary
        )
        channel_correct_all = safe_mean(
            int(bool(row["channel_correct"])) for row in sample_summary
        )
        channel_correct_wrong = safe_mean(
            int(bool(row["channel_correct"])) for row in wrong_samples
        )
        channel_agrees_gen_wrong = safe_mean(
            int(bool(row["channel_agrees_generation"])) for row in wrong_samples
        )

        ranked_emergent = sorted(
            head_summary,
            key=lambda row: (
                -float(row["emergent_wrong_direction_rate"]),
                -float(row["wrong_over_correct_relation_abs_ratio"]),
                float(row["wrong_relation_median_rank"]),
            ),
        )
        ranked_transport = sorted(
            head_summary,
            key=lambda row: (
                -float(row["wrong_over_correct_transport_ratio"]),
                float(row["wrong_transport_median_rank"]),
            ),
        )

        summary = {
            "script_version": SCRIPT_VERSION,
            "N": len(sample_summary),
            "baseline_correct": len(correct_samples),
            "baseline_wrong": len(wrong_samples),
            "baseline_accuracy": safe_mean(
                int(bool(row["baseline_correct"])) for row in sample_summary
            ),
            "scan_head_count": len(scan_heads),
            "expected_c_passes": len(sample_summary) * len(scan_heads),
            "sender_position_mode": args.sender_position_mode,
            "channel_centroid_accuracy_all": channel_correct_all,
            "wrong_samples_channel_still_GT_rate": channel_correct_wrong,
            "wrong_samples_channel_agrees_generation_rate": channel_agrees_gen_wrong,
            "correct_canonical_relation_share_mean": safe_mean(
                float(row["canonical_relation_share"])
                for row in correct_samples
            ),
            "wrong_canonical_relation_share_mean": safe_mean(
                float(row["canonical_relation_share"])
                for row in wrong_samples
            ),
            "correct_off_circuit_relation_share_mean": safe_mean(
                float(row["off_circuit_relation_share"])
                for row in correct_samples
            ),
            "wrong_off_circuit_relation_share_mean": safe_mean(
                float(row["off_circuit_relation_share"])
                for row in wrong_samples
            ),
            "correct_routing_entropy_mean": safe_mean(
                float(row["routing_entropy"]) for row in correct_samples
            ),
            "wrong_routing_entropy_mean": safe_mean(
                float(row["routing_entropy"]) for row in wrong_samples
            ),
            "wrong_canonical_wrong_direction_fraction_mean": safe_mean(
                float(row["canonical_wrong_direction_fraction"])
                for row in wrong_samples
            ),
            "wrong_emergent_transport_head_count_mean": safe_mean(
                float(row["emergent_transport_head_count"])
                for row in wrong_samples
            ),
            "wrong_emergent_wrong_direction_head_count_mean": safe_mean(
                float(row["emergent_wrong_direction_head_count"])
                for row in wrong_samples
            ),
            "heuristic_failure_class_counts": dict(failure_counts),
            "top_emergent_wrong_direction_heads": [
                {
                    "head": row["head"],
                    "canonical": row["canonical"],
                    "emergent_wrong_direction_rate": row[
                        "emergent_wrong_direction_rate"
                    ],
                    "wrong_over_correct_relation_abs_ratio": row[
                        "wrong_over_correct_relation_abs_ratio"
                    ],
                    "wrong_supports_generated_over_GT_rate": row[
                        "wrong_supports_generated_over_GT_rate"
                    ],
                }
                for row in ranked_emergent[:20]
            ],
            "top_wrong_transport_shift_heads": [
                {
                    "head": row["head"],
                    "canonical": row["canonical"],
                    "wrong_over_correct_transport_ratio": row[
                        "wrong_over_correct_transport_ratio"
                    ],
                    "transport_shift_cohens_d_wrong_minus_correct": row[
                        "transport_shift_cohens_d_wrong_minus_correct"
                    ],
                    "wrong_transport_median_rank": row[
                        "wrong_transport_median_rank"
                    ],
                }
                for row in ranked_transport[:20]
            ],
            "limitations": [
                "Relation axes are GT-oriented and used for mechanism analysis.",
                "C-pass deltas isolate the sender-to-L26VH0 residual/MLP route.",
                "Heuristic failure classes are descriptive, not ground-truth causes.",
                "Exact output-score effects are absent unless --exact-score-effects is enabled.",
            ],
        }
        write_json(output_dir / "summary.json", summary)

        print("\n" + "=" * 148)
        print("SIGNED ROUTING-SHIFT RESULT")
        print("=" * 148)
        print(
            f"Samples={summary['N']} | correct={summary['baseline_correct']} | "
            f"wrong={summary['baseline_wrong']} | heads={summary['scan_head_count']} | "
            f"baseline_acc={summary['baseline_accuracy']:.4f}"
        )
        print(
            "L26VH0 channel centroid: "
            f"all accuracy={summary['channel_centroid_accuracy_all']:.4f} | "
            f"among generation-wrong, channel still predicts GT="
            f"{summary['wrong_samples_channel_still_GT_rate']:.4f} | "
            f"channel agrees with wrong generation="
            f"{summary['wrong_samples_channel_agrees_generation_rate']:.4f}"
        )
        print("\nCORRECT VS WRONG ROUTING")
        print(
            "canonical relation share: "
            f"correct={summary['correct_canonical_relation_share_mean']:.4f} "
            f"wrong={summary['wrong_canonical_relation_share_mean']:.4f}"
        )
        print(
            "off-circuit relation share: "
            f"correct={summary['correct_off_circuit_relation_share_mean']:.4f} "
            f"wrong={summary['wrong_off_circuit_relation_share_mean']:.4f}"
        )
        print(
            "routing entropy: "
            f"correct={summary['correct_routing_entropy_mean']:.4f} "
            f"wrong={summary['wrong_routing_entropy_mean']:.4f}"
        )
        print(
            "wrong canonical wrong-direction fraction: "
            f"{summary['wrong_canonical_wrong_direction_fraction_mean']:.4f}"
        )
        print(
            "wrong emergent heads/sample: "
            f"transport={summary['wrong_emergent_transport_head_count_mean']:.3f} "
            f"wrong-direction="
            f"{summary['wrong_emergent_wrong_direction_head_count_mean']:.3f}"
        )

        print("\nHEURISTIC FAILURE CLASSES")
        for name, count in failure_counts.most_common():
            print(f"{name:42s} {count:4d}")

        print("\nTOP WEAK-ON-CORRECT HEADS EMERGING TOWARD WRONG PREDICTIONS")
        print(
            f"{'head':>8} {'canon':>6} {'emerge':>9} {'wrong/corr':>11} "
            f"{'wrongSupport':>13}"
        )
        for row in ranked_emergent[:20]:
            print(
                f"{str(row['head']):>8} "
                f"{str(bool(row['canonical'])):>6} "
                f"{float(row['emergent_wrong_direction_rate']):9.4f} "
                f"{float(row['wrong_over_correct_relation_abs_ratio']):11.4f} "
                f"{float(row['wrong_supports_generated_over_GT_rate']):13.4f}"
            )

        print("\nTOP HEADS BY WRONG/CORRECT TRANSPORT INCREASE")
        print(
            f"{'head':>8} {'canon':>6} {'ratio':>10} {'d':>10} "
            f"{'wrongRank':>10}"
        )
        for row in ranked_transport[:20]:
            print(
                f"{str(row['head']):>8} "
                f"{str(bool(row['canonical'])):>6} "
                f"{float(row['wrong_over_correct_transport_ratio']):10.4f} "
                f"{float(row['transport_shift_cohens_d_wrong_minus_correct']):+10.4f} "
                f"{float(row['wrong_transport_median_rank']):10.2f}"
            )

        print(f"\nSaved outputs to {output_dir}", flush=True)

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
