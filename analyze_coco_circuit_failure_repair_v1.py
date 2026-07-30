#!/usr/bin/env python3
"""
Diagnose and repair the validated P_POS7/P_NEG5 -> L26VH0 spatial-relation
circuit in Qwen2.5-VL on COCO two-object query-swap data.

The script does two related experiments on the ORIGINAL question only.

1. Failure diagnosis
   It estimates the path-specific contribution of a sender bundle by zeroing
   that bundle's pre-W_O head vectors at the object-token positions, freezing
   all other attention outputs at those positions to the clean run, recomputing
   token-wise residual/MLP propagation, capturing the resulting L26 V state,
   and patching only L26 KV-head 0's V slice back into a normal original run.

   For every sample:

       positive_support = M_base - M_without_POS7
       negative_burden  = M_without_NEG5 - M_base
       circuit_balance  = positive_support - negative_burden

   M is the GT-vs-opposite next-token relation-logit margin. The primary
   analysis compares these quantities between high-confidence correct,
   low-confidence correct, and wrong samples.

2. No-training circuit repair
   Let V be clean L26VH0, V_-pos the state after path-specific P_POS7 removal,
   and V_-neg the state after path-specific P_NEG5 removal. The repaired state
   is

       V' = V + alpha * (V - V_-pos) - beta * (V - V_-neg)

   alpha amplifies the model's own positive circuit contribution; beta removes
   part of the model's own negative circuit contribution. This intervention
   uses no external centroid classifier and no ground-truth relation at
   inference time. Ground truth is used only to evaluate accuracy.

The script evaluates an alpha/beta grid, creates a deterministic stratified
TUNE/TEST split, selects one global pair on TUNE, and reports TEST accuracy.

Expected existing files in the repository root:
  analyze_coco_ioi_backward_circuit_v1.py
  analyze_coco_producer_qk_ov_v1.py
  analyze_coco_receiver_qkv_v1.py
  analyze_spatial_storage_transport_utilization_v3.py
  analyze_coco_centroid_generation_step1_v4.py
  analyze_coco_flip_attention_spatial_vectors_v1.py
  coco_ioi_role_bundles_v1.json

Outputs:
  circuit_sample_metrics.jsonl
  repair_grid_effect.jsonl
  diagnostic_group_summary.csv
  diagnostic_feature_summary.csv
  repair_grid_summary.csv
  best_repair.json
  config.json
  tokenization.json
  errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
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
from collections import defaultdict
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


SCRIPT_VERSION = "coco-circuit-failure-repair-v1"
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
PHASES = ("diagnose", "repair", "all")


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--phase", choices=PHASES, default="all")
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
        "--require-single-token-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument("--positive-bundle", default="P_POS7")
    p.add_argument("--negative-bundle", default="P_NEG5")
    p.add_argument("--receiver-layer", type=int, default=26)
    p.add_argument("--receiver-query-head", type=int, default=0)
    p.add_argument("--receiver-channel", choices=("v",), default="v")
    p.add_argument("--receiver-kv-scope", choices=("objects",), default="objects")
    p.add_argument(
        "--sender-object-positions",
        choices=("last", "all"),
        default="last",
        help="Object positions where sender heads are path-ablated.",
    )

    p.add_argument("--sample-status", choices=STATUSES, default="all")
    p.add_argument(
        "--sample-max-samples",
        type=int,
        default=0,
        help="0 means all eligible source rows.",
    )
    p.add_argument(
        "--exclude-sids-from",
        default="",
        help="Comma-separated JSONL/CSV files whose sid values are excluded.",
    )
    p.add_argument(
        "--include-sids-file",
        default="",
        help="Optional text/JSONL/CSV file limiting the run to listed sids.",
    )

    p.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--beta-grid", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--tune-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=1)
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

    # Kept for compatibility with imported helpers.
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


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


def parse_float_grid(value: str, label: str) -> List[float]:
    values: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{label} values must be finite and >= 0: {item}")
        values.append(number)
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError(f"No values in {label}")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
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
                fields.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_head(value: Any) -> Tuple[int, int]:
    text = str(value).strip()
    if text.startswith("L") and "H" in text:
        layer, head = text[1:].split("H", 1)
        return int(layer), int(head)
    if ":" in text:
        layer, head = text.split(":", 1)
        return int(layer), int(head)
    raise ValueError(f"Invalid head: {value!r}")


@dataclass(frozen=True)
class HeadBundle:
    name: str
    heads: Tuple[Any, ...]

    @property
    def head_names(self) -> Tuple[str, ...]:
        return tuple(str(head.node) for head in self.heads)


def load_named_bundles(
    path: Path,
    names: Sequence[str],
    ioi: Any,
) -> Dict[str, HeadBundle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if not isinstance(source, Mapping):
        raise ValueError("Bundle JSON must contain an object")
    result: Dict[str, HeadBundle] = {}
    for name in names:
        if name not in source:
            raise KeyError(f"Missing bundle {name}; available={list(source)}")
        parsed = [parse_head(item) for item in source[name]]
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError(f"Invalid or duplicate heads in bundle {name}")
        nodes = tuple(
            ioi.SenderNode("attention", int(layer), int(head))
            for layer, head in parsed
        )
        result[name] = HeadBundle(name, nodes)
    return result


def status_matches(row: Mapping[str, Any], status: str, ioi: Any) -> bool:
    if status == "all":
        return True
    return bool(ioi.status_matches(row, status))


def extract_sids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    result: set[int] = set()
    if suffix == ".jsonl":
        for row in read_jsonl(path):
            if "sid" in row:
                result.add(int(row["sid"]))
        return result
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("sid", "").strip():
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


def deterministic_stratified_limit(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if limit <= 0 or len(rows) <= limit:
        return rows
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('gt','?')}::{row.get('generation_pair_status','?')}"
        groups[key].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: List[Dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


# -----------------------------------------------------------------------------
# Path-specific sender removal and receiver manipulation
# -----------------------------------------------------------------------------


class FreezeAttentionBundleZeroAtPositions:
    """Freeze attention pre-WO vectors to clean, then zero selected heads."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        output_projection_fn: Any,
        clean_attention: Mapping[int, Mapping[int, torch.Tensor]],
        positions_by_layer: Mapping[int, Sequence[int]],
        bundle: HeadBundle,
        zero_positions: Sequence[int],
    ) -> None:
        self.handles: List[Any] = []
        self.layer_events: Dict[int, int] = defaultdict(int)
        self.sender_events: Dict[str, int] = defaultdict(int)
        zero_set = set(map(int, zero_positions))
        by_layer: Dict[int, List[Any]] = defaultdict(list)
        for sender in bundle.heads:
            by_layer[int(sender.layer)].append(sender)

        for layer_index, positions in sorted(positions_by_layer.items()):
            layer_index = int(layer_index)
            layer = decoder_layers[layer_index]
            attention = attention_helper.resolve_self_attention(layer)
            shape = receiver_module.resolve_attention_shape(attention)
            module = output_projection_fn(attention)
            selected_positions = tuple(sorted(set(map(int, positions))))
            selected_senders = tuple(by_layer.get(layer_index, ()))

            def make_hook(
                index: int,
                layer_shape: Any,
                layer_positions: Sequence[int],
                layer_senders: Sequence[Any],
            ):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    if not inputs:
                        raise RuntimeError("W_O pre-hook received no input")
                    tensor = inputs[0]
                    if not torch.is_tensor(tensor) or tensor.ndim != 3:
                        raise RuntimeError("W_O input must be [B,S,D]")
                    if int(tensor.shape[0]) != 1:
                        raise RuntimeError("Expected batch size 1")
                    modified = tensor.clone()
                    for position in layer_positions:
                        if not 0 <= int(position) < int(tensor.shape[1]):
                            raise RuntimeError(
                                f"Position {position} outside sequence length {tensor.shape[1]}"
                            )
                        clean = clean_attention[index][int(position)].to(
                            device=tensor.device,
                            dtype=tensor.dtype,
                        )
                        modified[0, int(position)] = clean
                    self.layer_events[index] += 1

                    for sender in layer_senders:
                        start = int(sender.head) * int(layer_shape.query_head_dim)
                        stop = start + int(layer_shape.query_head_dim)
                        for position in layer_positions:
                            if int(position) not in zero_set:
                                continue
                            modified[0, int(position), start:stop] = 0
                            self.sender_events[str(sender.node)] += 1
                    return (modified, *inputs[1:])

                return hook

            self.handles.append(
                module.register_forward_pre_hook(
                    make_hook(
                        layer_index,
                        shape,
                        selected_positions,
                        selected_senders,
                    )
                )
            )

    def validate(self, bundle: HeadBundle) -> None:
        if not self.layer_events:
            raise RuntimeError("Freeze hooks did not fire")
        missing = [
            sender.node
            for sender in bundle.heads
            if self.sender_events[str(sender.node)] < 1
        ]
        if missing:
            raise RuntimeError(f"Bundle zero patches did not fire: {missing}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


@torch.inference_mode()
def capture_clean_original(
    *,
    pair: Any,
    capture_layers: Sequence[int],
    attention_positions: Sequence[int],
    receiver_positions: Sequence[int],
    unit: Any,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    ioi: Any,
) -> Tuple[Dict[str, Any], Any, Dict[int, Dict[str, Dict[int, torch.Tensor]]]]:
    projection_capture = ioi.CaptureProjectionAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        positions_by_layer_channel={
            (int(unit.layer), str(unit.channel)): list(receiver_positions)
        },
    )
    try:
        with ioi.CaptureWriterActivations(
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            layers=capture_layers,
            positions=attention_positions,
            capture_attention=True,
            capture_mlp=False,
        ) as clean_capture:
            result = receiver_module.run_scores(
                model=model,
                batch=pair.original_batch,
                base=base,
                relation_token_map=relation_token_map,
            )
        clean_capture.validate()
        projection_capture.validate()
        states = {
            int(layer): {
                str(channel): dict(position_map)
                for channel, position_map in channel_map.items()
            }
            for layer, channel_map in projection_capture.states.items()
        }
        return result, clean_capture, states
    finally:
        projection_capture.close()


@torch.inference_mode()
def run_bundle_removal_c_pass(
    *,
    bundle: HeadBundle,
    sender_positions: Sequence[int],
    receiver_positions: Sequence[int],
    unit: Any,
    pair: Any,
    clean_capture: Any,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    ioi: Any,
) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
    freeze_positions = sorted(
        set(map(int, sender_positions)) | set(map(int, receiver_positions))
    )
    positions_by_layer = {
        int(layer): freeze_positions for layer in clean_capture.layers
    }
    freeze = FreezeAttentionBundleZeroAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        output_projection_fn=ioi.output_projection_module,
        clean_attention=clean_capture.attention,
        positions_by_layer=positions_by_layer,
        bundle=bundle,
        zero_positions=sender_positions,
    )
    projection_capture = ioi.CaptureProjectionAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        positions_by_layer_channel={
            (int(unit.layer), str(unit.channel)): list(receiver_positions)
        },
    )
    try:
        receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        freeze.validate(bundle)
        projection_capture.validate()
        return {
            int(layer): {
                str(channel): dict(position_map)
                for channel, position_map in channel_map.items()
            }
            for layer, channel_map in projection_capture.states.items()
        }
    finally:
        projection_capture.close()
        freeze.close()


@torch.inference_mode()
def run_receiver_state(
    *,
    unit: Any,
    full_states_by_position: Mapping[int, torch.Tensor],
    pair: Any,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
) -> Dict[str, Any]:
    attention = attention_helper.resolve_self_attention(
        decoder_layers[int(unit.layer)]
    )
    shape = receiver_module.resolve_attention_shape(attention)
    module = receiver_module.projection_module(attention, unit.channel)
    head_dim = int(shape.kv_head_dim)
    patch = receiver_module.ProjectionHeadPatch(
        module=module,
        head=int(unit.unit_head),
        head_dim=head_dim,
        target_to_source={
            int(position): tensor
            for position, tensor in full_states_by_position.items()
        },
    )
    try:
        result = receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        if not patch.applied:
            raise RuntimeError("Receiver projection patch did not fire")
        return result
    finally:
        patch.close()


def combine_receiver_states(
    *,
    baseline: Mapping[int, torch.Tensor],
    without_positive: Mapping[int, torch.Tensor],
    without_negative: Mapping[int, torch.Tensor],
    alpha: float,
    beta: float,
) -> Dict[int, torch.Tensor]:
    result: Dict[int, torch.Tensor] = {}
    positions = sorted(set(baseline) & set(without_positive) & set(without_negative))
    if set(positions) != set(baseline):
        raise RuntimeError("Receiver state positions do not match")
    for position in positions:
        base_state = baseline[position].float()
        positive_delta = base_state - without_positive[position].float()
        negative_delta = base_state - without_negative[position].float()
        result[int(position)] = (
            base_state
            + float(alpha) * positive_delta
            - float(beta) * negative_delta
        )
    return result


def state_delta_norms(
    baseline: Mapping[int, torch.Tensor],
    ablated: Mapping[int, torch.Tensor],
    unit: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> Tuple[float, float]:
    attention = attention_helper.resolve_self_attention(
        decoder_layers[int(unit.layer)]
    )
    shape = receiver_module.resolve_attention_shape(attention)
    head_dim = int(shape.kv_head_dim)
    start = int(unit.unit_head) * head_dim
    stop = start + head_dim
    deltas = []
    ratios = []
    for position in sorted(baseline):
        b = baseline[position][start:stop].float()
        a = ablated[position][start:stop].float()
        delta = torch.linalg.vector_norm(b - a).item()
        base_norm = torch.linalg.vector_norm(b).item()
        deltas.append(delta)
        ratios.append(delta / max(base_norm, 1e-12))
    return float(np.mean(deltas)), float(np.mean(ratios))


# -----------------------------------------------------------------------------
# Statistical summaries
# -----------------------------------------------------------------------------


def rankdata(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    mask = np.isfinite(s)
    s = s[mask]
    y = y[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    rank_sum_pos = float(ranks[y].sum())
    return float(
        (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def cohens_d(correct: Sequence[float], wrong: Sequence[float]) -> float:
    a = np.asarray(correct, dtype=np.float64)
    b = np.asarray(wrong, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_num = (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def assign_confidence_groups(rows: List[Dict[str, Any]]) -> None:
    correct_margins = sorted(
        float(row["baseline_margin"])
        for row in rows
        if bool(row["baseline_correct"])
    )
    if not correct_margins:
        for row in rows:
            row["confidence_group"] = "wrong"
        return
    q25 = float(np.quantile(correct_margins, 0.25))
    q75 = float(np.quantile(correct_margins, 0.75))
    for row in rows:
        if not bool(row["baseline_correct"]):
            group = "wrong"
        elif float(row["baseline_margin"]) <= q25:
            group = "low_correct"
        elif float(row["baseline_margin"]) >= q75:
            group = "high_correct"
        else:
            group = "mid_correct"
        row["confidence_group"] = group
        row["correct_margin_q25"] = q25
        row["correct_margin_q75"] = q75


def build_diagnostic_summaries(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    assign_confidence_groups(rows)
    group_rows: List[Dict[str, Any]] = []
    group_order = ["high_correct", "mid_correct", "low_correct", "wrong"]
    features = [
        "baseline_margin",
        "positive_support",
        "negative_burden",
        "circuit_balance",
        "positive_delta_norm",
        "negative_delta_norm",
        "positive_delta_ratio",
        "negative_delta_ratio",
    ]
    for group in group_order:
        values = [row for row in rows if row["confidence_group"] == group]
        if not values:
            continue
        summary: Dict[str, Any] = {
            "confidence_group": group,
            "N": len(values),
            "baseline_accuracy": safe_mean(int(row["baseline_correct"]) for row in values),
        }
        for feature in features:
            summary[f"mean_{feature}"] = safe_mean(row[feature] for row in values)
            summary[f"median_{feature}"] = safe_median(row[feature] for row in values)
        group_rows.append(summary)

    labels = [bool(row["baseline_correct"]) for row in rows]
    margins = [float(row["baseline_margin"]) for row in rows]
    feature_specs = {
        "baseline_margin": [float(row["baseline_margin"]) for row in rows],
        "positive_support": [float(row["positive_support"]) for row in rows],
        "negative_burden_inverse": [-float(row["negative_burden"]) for row in rows],
        "circuit_balance": [float(row["circuit_balance"]) for row in rows],
        "positive_delta_ratio": [float(row["positive_delta_ratio"]) for row in rows],
        "negative_delta_ratio_inverse": [-float(row["negative_delta_ratio"]) for row in rows],
    }
    feature_rows: List[Dict[str, Any]] = []
    for name, values in feature_specs.items():
        correct_values = [value for value, label in zip(values, labels) if label]
        wrong_values = [value for value, label in zip(values, labels) if not label]
        feature_rows.append(
            {
                "feature": name,
                "N": len(values),
                "correct_mean": safe_mean(correct_values),
                "wrong_mean": safe_mean(wrong_values),
                "cohens_d_correct_minus_wrong": cohens_d(correct_values, wrong_values),
                "auroc_predict_correct": auroc(values, labels),
                "spearman_with_baseline_margin": spearman(values, margins),
            }
        )
    return group_rows, feature_rows


def deterministic_split(
    metric_rows: Sequence[Mapping[str, Any]],
    tune_fraction: float,
    seed: int,
) -> Dict[int, str]:
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("--tune-fraction must be between 0 and 1")
    groups: Dict[Tuple[str, bool], List[int]] = defaultdict(list)
    for row in metric_rows:
        groups[(str(row["gt"]), bool(row["baseline_correct"]))].append(int(row["sid"]))
    result: Dict[int, str] = {}
    for key, sids in sorted(groups.items()):
        local_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8],
            "big",
        )
        rng = random.Random(local_seed)
        sids = sorted(set(sids))
        rng.shuffle(sids)
        if len(sids) == 1:
            n_tune = 1 if tune_fraction >= 0.5 else 0
        else:
            n_tune = min(
                len(sids) - 1,
                max(1, int(round(len(sids) * tune_fraction))),
            )
        for index, sid in enumerate(sids):
            result[int(sid)] = "tune" if index < n_tune else "test"
    return result


def summarize_repair_grid(
    metric_rows: Sequence[Mapping[str, Any]],
    repair_rows: Sequence[Mapping[str, Any]],
    tune_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = deterministic_split(metric_rows, tune_fraction, seed)
    baseline_by_sid = {
        int(row["sid"]): {
            "correct": bool(row["baseline_correct"]),
            "margin": float(row["baseline_margin"]),
        }
        for row in metric_rows
    }
    groups: Dict[Tuple[str, float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in repair_rows:
        sid = int(row["sid"])
        groups[(split[sid], float(row["alpha"]), float(row["beta"]))].append(row)

    summary: List[Dict[str, Any]] = []
    for (split_name, alpha, beta), values in sorted(groups.items()):
        fixes = 0
        breaks = 0
        baseline_correct = 0
        repaired_correct = 0
        margin_deltas = []
        for row in values:
            sid = int(row["sid"])
            before = baseline_by_sid[sid]["correct"]
            after = bool(row["repaired_correct"])
            baseline_correct += int(before)
            repaired_correct += int(after)
            fixes += int((not before) and after)
            breaks += int(before and (not after))
            margin_deltas.append(
                float(row["repaired_margin"]) - baseline_by_sid[sid]["margin"]
            )
        n = len(values)
        summary.append(
            {
                "split": split_name,
                "alpha": alpha,
                "beta": beta,
                "N": n,
                "baseline_accuracy": baseline_correct / n,
                "repaired_accuracy": repaired_correct / n,
                "accuracy_delta": (repaired_correct - baseline_correct) / n,
                "fixes": fixes,
                "breaks": breaks,
                "net_fixes": fixes - breaks,
                "mean_margin_delta": safe_mean(margin_deltas),
                "median_margin_delta": safe_median(margin_deltas),
            }
        )

    tune = [row for row in summary if row["split"] == "tune"]
    if not tune:
        raise RuntimeError("No tune rows")
    tune_sorted = sorted(
        tune,
        key=lambda row: (
            -float(row["repaired_accuracy"]),
            -int(row["net_fixes"]),
            int(row["breaks"]),
            float(row["alpha"]) + float(row["beta"]),
            float(row["alpha"]),
            float(row["beta"]),
        ),
    )
    selected = tune_sorted[0]
    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    test_match = [
        row
        for row in summary
        if row["split"] == "test"
        and float(row["alpha"]) == alpha
        and float(row["beta"]) == beta
    ]
    if len(test_match) != 1:
        raise RuntimeError("Could not resolve selected grid point on test")

    all_rows = [row for row in repair_rows if float(row["alpha"]) == alpha and float(row["beta"]) == beta]
    all_baseline = sum(int(baseline_by_sid[int(row["sid"])]["correct"]) for row in all_rows)
    all_repaired = sum(int(bool(row["repaired_correct"])) for row in all_rows)
    all_fixes = sum(
        int((not baseline_by_sid[int(row["sid"])]["correct"]) and bool(row["repaired_correct"]))
        for row in all_rows
    )
    all_breaks = sum(
        int(baseline_by_sid[int(row["sid"])]["correct"] and (not bool(row["repaired_correct"])))
        for row in all_rows
    )

    best = {
        "selection_rule": "maximize tune accuracy, then net fixes, then minimize breaks and intervention magnitude",
        "selected_alpha": alpha,
        "selected_beta": beta,
        "tune": dict(selected),
        "test": dict(test_match[0]),
        "all_descriptive_only": {
            "N": len(all_rows),
            "baseline_accuracy": all_baseline / len(all_rows),
            "repaired_accuracy": all_repaired / len(all_rows),
            "accuracy_delta": (all_repaired - all_baseline) / len(all_rows),
            "fixes": all_fixes,
            "breaks": all_breaks,
            "net_fixes": all_fixes - all_breaks,
        },
        "split_by_sid": {str(sid): value for sid, value in sorted(split.items())},
    }
    return summary, best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    alphas = parse_float_grid(args.alpha_grid, "alpha grid")
    betas = parse_float_grid(args.beta_grid, "beta grid")
    run_repair = args.phase in {"repair", "all"}

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ioi = import_file(Path(args.ioi_script), "failure_repair_ioi")
    producer_module = import_file(Path(args.producer_script), "failure_repair_producer")
    receiver_module = import_file(Path(args.receiver_script), "failure_repair_receiver")
    v3 = import_file(Path(args.v3_script), "failure_repair_v3")
    base = import_file(Path(args.base_script), "failure_repair_base")
    attention_helper = import_file(Path(args.attention_helper), "failure_repair_attention")

    source_config, source_rows = ioi.load_source_rows(args)

    excluded: set[int] = set()
    for item in str(args.exclude_sids_from).split(","):
        item = item.strip()
        if item:
            excluded.update(extract_sids(Path(item)))
    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(args.include_sids_file.strip()))

    selected_rows = [
        dict(row)
        for row in source_rows
        if status_matches(row, args.sample_status, ioi)
        and int(row["sid"]) not in excluded
        and (included is None or int(row["sid"]) in included)
    ]
    selected_rows = deterministic_stratified_limit(
        selected_rows,
        int(args.sample_max_samples),
        int(args.seed),
    )
    if not selected_rows:
        raise RuntimeError("No eligible samples after status/include/exclude filters")

    bundles = load_named_bundles(
        Path(args.bundle_json),
        [args.positive_bundle, args.negative_bundle],
        ioi,
    )
    positive_bundle = bundles[args.positive_bundle]
    negative_bundle = bundles[args.negative_bundle]

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
        ) = producer_module.load_model_bundle(args=args, base=base)

        token_report = ioi.tokenization_report(processor.tokenizer, relation_token_map)
        write_json(output_dir / "tokenization.json", token_report)
        if (
            args.require_single_token_labels
            and not token_report["all_relations_have_single_token_continuation"]
        ):
            raise RuntimeError("At least one relation lacks a one-token continuation")

        receiver_layer = int(args.receiver_layer)
        writer = ioi.WriterNode("attention", receiver_layer, int(args.receiver_query_head))
        units = ioi.build_receiver_units(
            writers=[writer],
            channels=[str(args.receiver_channel)],
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver_module,
        )
        if len(units) != 1:
            raise RuntimeError(f"Expected one receiver unit, got {len(units)}")
        unit = units[0]

        for bundle in (positive_bundle, negative_bundle):
            for sender in bundle.heads:
                if int(sender.layer) >= receiver_layer:
                    raise ValueError(
                        f"{bundle.name}: {sender.node} must be earlier than L{receiver_layer}"
                    )
                attention = attention_helper.resolve_self_attention(
                    decoder_layers[int(sender.layer)]
                )
                shape = receiver_module.resolve_attention_shape(attention)
                if not 0 <= int(sender.head) < int(shape.n_query_heads):
                    raise ValueError(f"Invalid head {sender.node}")

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)
        config = {
            "script_version": SCRIPT_VERSION,
            "phase": args.phase,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "decoder_path": decoder_path,
            "n_layers": len(decoder_layers),
            "positive_bundle": {
                "name": positive_bundle.name,
                "heads": list(positive_bundle.head_names),
            },
            "negative_bundle": {
                "name": negative_bundle.name,
                "heads": list(negative_bundle.head_names),
            },
            "receiver": {
                "unit": unit.unit,
                "layer": int(unit.layer),
                "query_head": int(unit.query_head),
                "unit_head": int(unit.unit_head),
                "kv_head": int(unit.kv_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "channel": unit.channel,
                "kv_scope": args.receiver_kv_scope,
            },
            "sender_object_positions": args.sender_object_positions,
            "ablation": "zero selected pre-WO head slices; freeze all other attention at object positions; recompute MLPs",
            "repair_formula": "V' = V + alpha*(V - V_without_positive) - beta*(V - V_without_negative)",
            "alpha_grid": alphas,
            "beta_grid": betas,
            "sample_status": args.sample_status,
            "selected_samples": len(selected_rows),
            "excluded_sids": sorted(excluded),
            "included_sid_count": None if included is None else len(included),
            "tune_fraction": args.tune_fraction,
            "seed": args.seed,
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        metric_path = output_dir / "circuit_sample_metrics.jsonl"
        repair_path = output_dir / "repair_grid_effect.jsonl"
        errors_path = output_dir / "errors.jsonl"
        existing_metrics = read_jsonl(metric_path) if args.resume else []
        existing_repairs = read_jsonl(repair_path) if args.resume else []
        metric_done = {int(row["sid"]) for row in existing_metrics}
        repair_done = {
            (int(row["sid"]), float(row["alpha"]), float(row["beta"]))
            for row in existing_repairs
        }
        expected_grid = {(float(a), float(b)) for a in alphas for b in betas}

        pending_rows = []
        for row in selected_rows:
            sid = int(row["sid"])
            needs_metric = sid not in metric_done
            needs_repair = run_repair and any(
                (sid, alpha, beta) not in repair_done
                for alpha, beta in expected_grid
            )
            if needs_metric or needs_repair:
                pending_rows.append(row)

        print(
            "Circuit failure/repair scan: "
            f"requested_N={len(selected_rows)}, pending_N={len(pending_rows)}, "
            f"existing_metrics={len(existing_metrics)}, "
            f"existing_repairs={len(existing_repairs)}, "
            f"grid={len(alphas)}x{len(betas)}, receiver={unit.unit}",
            flush=True,
        )

        capture_layers = list(range(receiver_layer + 1))

        for sample_index, source_row in enumerate(
            tqdm(pending_rows, desc=f"circuit-repair:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                pair = receiver_module.prepare_pair(
                    args=args,
                    row=source_row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=torch.device(args.device),
                )

                if args.sender_object_positions == "last":
                    if not pair.original_a_positions or not pair.original_b_positions:
                        raise RuntimeError("Missing original object positions")
                    sender_positions = sorted(
                        {
                            int(pair.original_a_positions[-1]),
                            int(pair.original_b_positions[-1]),
                        }
                    )
                else:
                    sender_positions = list(map(int, pair.original_object_positions))
                receiver_positions = list(map(int, pair.original_object_positions))
                attention_positions = sorted(
                    set(sender_positions) | set(receiver_positions)
                )

                baseline_result, clean_capture, baseline_states = capture_clean_original(
                    pair=pair,
                    capture_layers=capture_layers,
                    attention_positions=attention_positions,
                    receiver_positions=receiver_positions,
                    unit=unit,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                    ioi=ioi,
                )

                positive_removed_states = run_bundle_removal_c_pass(
                    bundle=positive_bundle,
                    sender_positions=sender_positions,
                    receiver_positions=receiver_positions,
                    unit=unit,
                    pair=pair,
                    clean_capture=clean_capture,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                    ioi=ioi,
                )
                negative_removed_states = run_bundle_removal_c_pass(
                    bundle=negative_bundle,
                    sender_positions=sender_positions,
                    receiver_positions=receiver_positions,
                    unit=unit,
                    pair=pair,
                    clean_capture=clean_capture,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                    ioi=ioi,
                )

                baseline_v = baseline_states[int(unit.layer)][unit.channel]
                positive_removed_v = positive_removed_states[int(unit.layer)][unit.channel]
                negative_removed_v = negative_removed_states[int(unit.layer)][unit.channel]

                without_positive_result = run_receiver_state(
                    unit=unit,
                    full_states_by_position=positive_removed_v,
                    pair=pair,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                )
                without_negative_result = run_receiver_state(
                    unit=unit,
                    full_states_by_position=negative_removed_v,
                    pair=pair,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                )

                gt = str(pair.gt)
                baseline_margin = float(ioi.relation_margin(baseline_result["logits"], gt))
                without_positive_margin = float(
                    ioi.relation_margin(without_positive_result["logits"], gt)
                )
                without_negative_margin = float(
                    ioi.relation_margin(without_negative_result["logits"], gt)
                )
                positive_support = baseline_margin - without_positive_margin
                negative_burden = without_negative_margin - baseline_margin
                circuit_balance = positive_support - negative_burden
                positive_delta_norm, positive_delta_ratio = state_delta_norms(
                    baseline_v,
                    positive_removed_v,
                    unit,
                    decoder_layers,
                    attention_helper,
                    receiver_module,
                )
                negative_delta_norm, negative_delta_ratio = state_delta_norms(
                    baseline_v,
                    negative_removed_v,
                    unit,
                    decoder_layers,
                    attention_helper,
                    receiver_module,
                )

                metric_row = {
                    "script_version": SCRIPT_VERSION,
                    "model": args.model,
                    "sid": int(pair.sid),
                    "gt": gt,
                    "source_generation_pair_status": source_row.get(
                        "generation_pair_status", "unknown"
                    ),
                    "baseline_prediction": baseline_result["prediction"],
                    "baseline_correct": bool(baseline_result["prediction"] == gt),
                    "baseline_margin": baseline_margin,
                    "without_positive_prediction": without_positive_result["prediction"],
                    "without_positive_correct": bool(
                        without_positive_result["prediction"] == gt
                    ),
                    "without_positive_margin": without_positive_margin,
                    "positive_support": positive_support,
                    "without_negative_prediction": without_negative_result["prediction"],
                    "without_negative_correct": bool(
                        without_negative_result["prediction"] == gt
                    ),
                    "without_negative_margin": without_negative_margin,
                    "negative_burden": negative_burden,
                    "circuit_balance": circuit_balance,
                    "positive_delta_norm": positive_delta_norm,
                    "negative_delta_norm": negative_delta_norm,
                    "positive_delta_ratio": positive_delta_ratio,
                    "negative_delta_ratio": negative_delta_ratio,
                    "positive_removal_flipped_correct_to_wrong": bool(
                        baseline_result["prediction"] == gt
                        and without_positive_result["prediction"] != gt
                    ),
                    "negative_removal_fixed_wrong_to_correct": bool(
                        baseline_result["prediction"] != gt
                        and without_negative_result["prediction"] == gt
                    ),
                    "sender_positions": sender_positions,
                    "receiver_positions": receiver_positions,
                    "receiver_unit": unit.unit,
                    "path_definition": "bundle removal to L26VH0 through residual and recomputed MLPs; intermediate attention frozen",
                }
                if int(pair.sid) not in metric_done:
                    append_jsonl(metric_path, metric_row)
                    existing_metrics.append(metric_row)
                    metric_done.add(int(pair.sid))

                if run_repair:
                    for alpha in alphas:
                        for beta in betas:
                            key = (int(pair.sid), float(alpha), float(beta))
                            if key in repair_done:
                                continue
                            if float(alpha) == 0.0 and float(beta) == 0.0:
                                repaired_result = baseline_result
                            else:
                                repaired_states = combine_receiver_states(
                                    baseline=baseline_v,
                                    without_positive=positive_removed_v,
                                    without_negative=negative_removed_v,
                                    alpha=float(alpha),
                                    beta=float(beta),
                                )
                                repaired_result = run_receiver_state(
                                    unit=unit,
                                    full_states_by_position=repaired_states,
                                    pair=pair,
                                    model=model,
                                    decoder_layers=decoder_layers,
                                    relation_token_map=relation_token_map,
                                    base=base,
                                    receiver_module=receiver_module,
                                    attention_helper=attention_helper,
                                )
                            repaired_margin = float(
                                ioi.relation_margin(repaired_result["logits"], gt)
                            )
                            repair_row = {
                                "script_version": SCRIPT_VERSION,
                                "model": args.model,
                                "sid": int(pair.sid),
                                "gt": gt,
                                "source_generation_pair_status": source_row.get(
                                    "generation_pair_status", "unknown"
                                ),
                                "alpha": float(alpha),
                                "beta": float(beta),
                                "baseline_prediction": baseline_result["prediction"],
                                "baseline_correct": bool(
                                    baseline_result["prediction"] == gt
                                ),
                                "baseline_margin": baseline_margin,
                                "repaired_prediction": repaired_result["prediction"],
                                "repaired_correct": bool(
                                    repaired_result["prediction"] == gt
                                ),
                                "repaired_margin": repaired_margin,
                                "margin_delta": repaired_margin - baseline_margin,
                                "fixed": bool(
                                    baseline_result["prediction"] != gt
                                    and repaired_result["prediction"] == gt
                                ),
                                "broken": bool(
                                    baseline_result["prediction"] == gt
                                    and repaired_result["prediction"] != gt
                                ),
                                "positive_support": positive_support,
                                "negative_burden": negative_burden,
                                "circuit_balance": circuit_balance,
                                "receiver_unit": unit.unit,
                            }
                            append_jsonl(repair_path, repair_row)
                            existing_repairs.append(repair_row)
                            repair_done.add(key)

            except Exception as exc:
                error = {
                    "script_version": SCRIPT_VERSION,
                    "sid": int(source_row["sid"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(errors_path, error)
                print(
                    f"\n[ERROR sid={source_row['sid']}] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver_module.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and (
                    args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

            if args.print_every > 0 and sample_index % args.print_every == 0:
                print(
                    f"[sample {sample_index}/{len(pending_rows)}] "
                    f"metrics={len(existing_metrics)} repairs={len(existing_repairs)}",
                    flush=True,
                )

        # Restrict summaries to selected SIDs in case the output directory was
        # previously used with a broader source set.
        selected_sids = {int(row["sid"]) for row in selected_rows}
        final_metrics = [
            dict(row) for row in read_jsonl(metric_path)
            if int(row["sid"]) in selected_sids
        ]
        group_summary, feature_summary = build_diagnostic_summaries(final_metrics)
        write_csv(output_dir / "diagnostic_group_summary.csv", group_summary)
        write_csv(output_dir / "diagnostic_feature_summary.csv", feature_summary)

        baseline_accuracy = safe_mean(
            int(bool(row["baseline_correct"])) for row in final_metrics
        )
        diagnostics = {
            "N": len(final_metrics),
            "baseline_accuracy": baseline_accuracy,
            "positive_removal_break_rate": safe_mean(
                int(bool(row["positive_removal_flipped_correct_to_wrong"]))
                for row in final_metrics
            ),
            "negative_removal_fix_rate": safe_mean(
                int(bool(row["negative_removal_fixed_wrong_to_correct"]))
                for row in final_metrics
            ),
            "mean_positive_support": safe_mean(
                float(row["positive_support"]) for row in final_metrics
            ),
            "mean_negative_burden": safe_mean(
                float(row["negative_burden"]) for row in final_metrics
            ),
            "mean_circuit_balance": safe_mean(
                float(row["circuit_balance"]) for row in final_metrics
            ),
            "group_summary_file": "diagnostic_group_summary.csv",
            "feature_summary_file": "diagnostic_feature_summary.csv",
        }
        write_json(output_dir / "diagnostic_overview.json", diagnostics)

        if run_repair:
            final_repairs = [
                dict(row) for row in read_jsonl(repair_path)
                if int(row["sid"]) in selected_sids
                and float(row["alpha"]) in set(alphas)
                and float(row["beta"]) in set(betas)
            ]
            expected_count = len(final_metrics) * len(alphas) * len(betas)
            if len(final_repairs) != expected_count:
                raise RuntimeError(
                    f"Repair rows incomplete: got {len(final_repairs)}, "
                    f"expected {expected_count}"
                )
            repair_summary, best = summarize_repair_grid(
                final_metrics,
                final_repairs,
                args.tune_fraction,
                args.seed,
            )
            write_csv(output_dir / "repair_grid_summary.csv", repair_summary)
            write_json(output_dir / "best_repair.json", best)
            print(
                "\nBest global repair selected on TUNE:\n"
                f"  alpha={best['selected_alpha']:.4f} "
                f"beta={best['selected_beta']:.4f}\n"
                f"  TUNE: baseline={best['tune']['baseline_accuracy']:.4f} "
                f"repaired={best['tune']['repaired_accuracy']:.4f} "
                f"delta={best['tune']['accuracy_delta']:+.4f} "
                f"fixes={best['tune']['fixes']} breaks={best['tune']['breaks']}\n"
                f"  TEST: baseline={best['test']['baseline_accuracy']:.4f} "
                f"repaired={best['test']['repaired_accuracy']:.4f} "
                f"delta={best['test']['accuracy_delta']:+.4f} "
                f"fixes={best['test']['fixes']} breaks={best['test']['breaks']}",
                flush=True,
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
