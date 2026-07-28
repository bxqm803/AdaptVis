#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receiver-head Q/K/V channel analysis for the COCO two-object spatial task.

This script is the next stage after:

    analyze_coco_producer_qk_ov_v1.py

The producer script asks which heads read visual tokens and write relation-bearing
information into object-token residual states. This script asks the next, narrower
question:

    Which later attention components causally use the clean object states, and do
    they use them through the receiver Q, K, or V channel?

The script intentionally has two stages.

PHASE 1: receiver scan
----------------------
For every later layer/head, measure the object->prompt-last contribution:

    c_h = W_O^h sum_{s in object tokens} A[last,s]^h V_s^h

for the original and subject/reference-swapped questions. Candidate receivers are
ranked by how much their clean-vs-swapped object write explains the clean-vs-
swapped prompt-last block difference.

This phase is descriptive. It only selects candidates.

PHASE 2: receiver-channel causal patch
--------------------------------------
For the top receiver candidates, independently patch pre-RoPE projection outputs:

    Q restore:
        original prompt-last q_proj head slice -> swapped prompt-last

    K restore:
        original identity-aligned object k_proj KV-head slice -> swapped objects

    V restore:
        original identity-aligned object v_proj KV-head slice -> swapped objects

The reverse clean-side corruption is also supported.

Important interpretation:

* Q patch tests whether the receiver's destination query channel is causally
  sensitive to the clean computation.
* K patch tests whether clean object states change source addressability. The
  patch is made at k_proj output before rotary position embedding, so clean
  content is transplanted while the target run keeps its own token position.
* V patch tests whether clean object states provide relation-bearing transported
  content.
* This is a receiver-channel intervention, not yet an exact producer->receiver
  path patch. A later script must freeze the corrupt run and allow only one
  producer write to affect one receiver Q/K/V channel.
* In grouped-query attention, K/V projections are shared. K/V results are
  therefore reported per KV head together with the query heads that share it.

Required existing outputs
-------------------------
1. Source v3 directory:

       config.json
       extraction.jsonl
       cache/<sid>.npz

2. Producer directory:

       producer_top_heads.json

Outputs
-------
    config.json
    receiver_head_scan.csv
    receiver_top_heads.json
    receiver_channel_causal.jsonl
    receiver_channel_summary.csv
    report.txt
    errors.jsonl

Example: Qwen-3B
----------------
CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_receiver_qkv_v1.py \
  --model qwen-3b \
  --source-output-dir \
    output/spatial_storage_transport_utilization/coco/qwen-3b \
  --producer-output-dir \
    output/coco_producer_qk_ov/qwen-3b \
  --phase all \
  --receiver-layers after_producer \
  --scan-status both_correct \
  --scan-max-samples 100 \
  --rank-metric mean_projection_fraction_block \
  --top-receiver-heads 16 \
  --causal-status both_correct \
  --causal-max-samples 60 \
  --channels q,k,v \
  --conditions restore_on_swapped,corrupt_on_original \
  --device cuda:0 \
  --output-dir output/coco_receiver_qkv/qwen-3b \
  --overwrite

Example: LLaVA-7B
-----------------
CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_receiver_qkv_v1.py \
  --model llava-7b \
  --source-output-dir \
    output/spatial_storage_transport_utilization/coco/llava-7b \
  --producer-output-dir \
    output/coco_producer_qk_ov/llava-7b \
  --phase all \
  --receiver-layers after_producer \
  --scan-status both_correct \
  --scan-max-samples 100 \
  --top-receiver-heads 16 \
  --causal-status both_correct \
  --causal-max-samples 60 \
  --device cuda:0 \
  --output-dir output/coco_receiver_qkv/llava-7b \
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
from PIL import Image
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-receiver-qkv-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
CHANNELS = ("q", "k", "v")
CONDITIONS = ("restore_on_swapped", "corrupt_on_original")


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source-output-dir",
        required=True,
        help="Existing analyze_spatial_storage_transport_utilization_v3.py output.",
    )
    parser.add_argument(
        "--producer-output-dir",
        required=True,
        help="Existing analyze_coco_producer_qk_ov_v1.py output.",
    )
    parser.add_argument("--phase", choices=("all", "scan", "causal"), default="all")
    parser.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=("eager",))
    parser.add_argument(
        "--object-state",
        choices=("last", "mean"),
        default="last",
        help="Must match the source v3 and producer settings.",
    )
    parser.add_argument(
        "--producer-heads",
        default="top",
        help="'top' reads producer_top_heads.json, or use '19:13,20:6'.",
    )
    parser.add_argument(
        "--receiver-layers",
        default="after_producer",
        help=(
            "'after_producer' scans all layers after the earliest selected producer; "
            "'all', or comma-separated zero-based layers."
        ),
    )
    parser.add_argument("--scan-status", choices=STATUSES, default="both_correct")
    parser.add_argument(
        "--scan-max-samples",
        type=int,
        default=100,
        help="0 or negative means all eligible samples.",
    )
    parser.add_argument("--trace-layer-chunk", type=int, default=4)
    parser.add_argument(
        "--rank-metric",
        choices=(
            "mean_projection_fraction_block",
            "mean_abs_projection_fraction_block",
            "mean_delta_norm",
            "mean_object_attention_mass",
            "positive_projection_rate",
        ),
        default="mean_projection_fraction_block",
    )
    parser.add_argument("--top-receiver-heads", type=int, default=16)
    parser.add_argument(
        "--save-scan-per-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--causal-status", choices=STATUSES, default="both_correct")
    parser.add_argument(
        "--causal-max-samples",
        type=int,
        default=60,
        help="0 or negative means all eligible samples.",
    )
    parser.add_argument(
        "--channels",
        default="q,k,v",
        help="Comma-separated subset of q,k,v.",
    )
    parser.add_argument(
        "--conditions",
        default="restore_on_swapped,corrupt_on_original",
        help="Comma-separated receiver-channel intervention conditions.",
    )
    parser.add_argument(
        "--causal-heads",
        default="top",
        help=(
            "'top' reads receiver_top_heads.json, or use query-head entries "
            "such as '24:3,27:11'."
        ),
    )
    parser.add_argument("--min-margin-denominator", type=float, default=1e-4)
    parser.add_argument(
        "--causal-require-margin-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument(
        "--producer-script",
        default="analyze_coco_producer_qk_ov_v1.py",
    )
    parser.add_argument(
        "--v3-script",
        default="analyze_spatial_storage_transport_utilization_v3.py",
    )
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    parser.add_argument(
        "--attention-helper",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
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


def safe_mean(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def relation_margin(logits: Sequence[float], gt: str) -> float:
    values = np.asarray(logits, dtype=np.float64)
    return float(values[REL_TO_ID[gt]] - values[REL_TO_ID[OPPOSITE[gt]]])


def status_matches(row: Mapping[str, Any], status: str) -> bool:
    return status == "all" or str(row["generation_pair_status"]) == status


def stratified_limit(
    rows: Sequence[Mapping[str, Any]],
    limit: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if limit is None or limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda item: int(item["sid"]))
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected: List[Dict[str, Any]] = []
    labels = [relation for relation in RELATIONS if relation in groups]
    cursors = {label: 0 for label in labels}
    while len(selected) < limit:
        progressed = False
        for label in labels:
            cursor = cursors[label]
            if cursor < len(groups[label]) and len(selected) < limit:
                selected.append(groups[label][cursor])
                cursors[label] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: int(item["sid"]))


def parse_subset(value: str, allowed: Sequence[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    result: List[str] = []
    for raw in str(value).split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(f"Unsupported {label}={item}; allowed={sorted(allowed_set)}")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"No {label} selected")
    return result


def parse_head_list(value: str, top_path: Path) -> List[Tuple[int, int]]:
    text = str(value).strip().lower()
    if text == "top":
        if not top_path.exists():
            raise FileNotFoundError(top_path)
        rows = json.loads(top_path.read_text(encoding="utf-8"))
        return [(int(row["layer"]), int(row["head"])) for row in rows]
    result: List[Tuple[int, int]] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        layer_text, head_text = item.split(":", 1)
        result.append((int(layer_text), int(head_text)))
    if not result:
        raise ValueError("No heads selected")
    return list(dict.fromkeys(result))


def parse_receiver_layers(
    value: str,
    n_layers: int,
    producer_heads: Sequence[Tuple[int, int]],
) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text == "after_producer":
        if not producer_heads:
            raise ValueError("after_producer requires producer heads")
        earliest = min(layer for layer, _ in producer_heads)
        layers = list(range(earliest + 1, n_layers))
        if not layers:
            raise ValueError(f"No decoder layer exists after producer layer {earliest}")
        return layers
    result: List[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        layer = int(item)
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside [0,{n_layers - 1}]")
        result.append(layer)
    if not result:
        raise ValueError("No receiver layers selected")
    return sorted(set(result))


def aligned_position_pairs(
    original_positions: Sequence[int],
    swapped_positions: Sequence[int],
) -> List[Tuple[int, int]]:
    original = list(map(int, original_positions))
    swapped = list(map(int, swapped_positions))
    if len(original) != len(swapped):
        raise RuntimeError(
            "Identity-aligned token lengths differ: "
            f"{len(original)} vs {len(swapped)}"
        )
    return list(zip(original, swapped))


# -----------------------------------------------------------------------------
# Attention architecture and projection helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionShape:
    n_query_heads: int
    n_kv_heads: int
    query_head_dim: int
    kv_head_dim: int

    @property
    def query_per_kv(self) -> int:
        if self.n_query_heads % self.n_kv_heads != 0:
            raise RuntimeError(
                f"Query heads {self.n_query_heads} are not divisible by "
                f"KV heads {self.n_kv_heads}"
            )
        return self.n_query_heads // self.n_kv_heads

    def kv_head_for_query(self, query_head: int) -> int:
        return int(query_head) // self.query_per_kv

    def shared_query_heads(self, kv_head: int) -> List[int]:
        start = int(kv_head) * self.query_per_kv
        return list(range(start, start + self.query_per_kv))


def resolve_attention_shape(attention: Any) -> AttentionShape:
    q_proj = getattr(attention, "q_proj", None)
    k_proj = getattr(attention, "k_proj", None)
    v_proj = getattr(attention, "v_proj", None)
    if q_proj is None or k_proj is None or v_proj is None:
        raise RuntimeError(
            f"{type(attention).__name__} must expose q_proj/k_proj/v_proj"
        )
    q_out = int(q_proj.out_features)
    k_out = int(k_proj.out_features)
    v_out = int(v_proj.out_features)
    if k_out != v_out:
        raise RuntimeError(f"k_proj.out_features={k_out} != v_proj.out_features={v_out}")

    config = getattr(attention, "config", None)
    n_query = getattr(attention, "num_heads", None)
    if n_query is None:
        n_query = getattr(attention, "num_attention_heads", None)
    if n_query is None and config is not None:
        n_query = getattr(config, "num_attention_heads", None)
    if n_query is None:
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            raise RuntimeError("Cannot infer number of query heads")
        n_query = q_out // int(head_dim)
    n_query = int(n_query)

    n_kv = getattr(attention, "num_key_value_heads", None)
    if n_kv is None and config is not None:
        n_kv = getattr(config, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = n_query
    n_kv = int(n_kv)

    if q_out % n_query != 0 or k_out % n_kv != 0:
        raise RuntimeError(
            f"Projection dimensions cannot be split: q={q_out}/{n_query}, "
            f"kv={k_out}/{n_kv}"
        )
    shape = AttentionShape(
        n_query_heads=n_query,
        n_kv_heads=n_kv,
        query_head_dim=q_out // n_query,
        kv_head_dim=k_out // n_kv,
    )
    _ = shape.query_per_kv
    return shape


def head_slice(head: int, head_dim: int) -> slice:
    start = int(head) * int(head_dim)
    return slice(start, start + int(head_dim))


def projection_module(attention: Any, channel: str) -> torch.nn.Module:
    module = getattr(attention, f"{channel}_proj", None)
    if module is None:
        raise RuntimeError(f"{type(attention).__name__} lacks {channel}_proj")
    return module


class CaptureProjectionPositions:
    """Capture q/k/v projection outputs at selected sequence positions."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layers: Sequence[int],
        q_positions: Sequence[int],
        kv_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layers = list(map(int, layers))
        self.q_positions = sorted(set(map(int, q_positions)))
        self.kv_positions = sorted(set(map(int, kv_positions)))
        self.handles: List[Any] = []
        self.states: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.events: Dict[Tuple[int, str], int] = defaultdict(int)

    def __enter__(self) -> "CaptureProjectionPositions":
        for layer in self.layers:
            attention = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            for channel, positions in (
                ("q", self.q_positions),
                ("k", self.kv_positions),
                ("v", self.kv_positions),
            ):
                module = projection_module(attention, channel)

                def make_hook(
                    layer_index: int,
                    channel_name: str,
                    selected_positions: Sequence[int],
                ):
                    def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                        if not torch.is_tensor(output) or output.ndim != 3:
                            raise RuntimeError(
                                f"{channel_name}_proj output must be [B,S,D], got "
                                f"{type(output).__name__}"
                            )
                        if int(output.shape[0]) != 1:
                            raise RuntimeError("Projection capture expects batch size 1")
                        for position in selected_positions:
                            if not 0 <= int(position) < int(output.shape[1]):
                                raise RuntimeError(
                                    f"Position {position} outside sequence length "
                                    f"{int(output.shape[1])}"
                                )
                            self.states[layer_index][channel_name][int(position)] = (
                                output[0, int(position)]
                                .detach()
                                .float()
                                .cpu()
                            )
                        self.events[(layer_index, channel_name)] += 1
                        return output
                    return hook

                self.handles.append(
                    module.register_forward_hook(
                        make_hook(layer, channel, positions)
                    )
                )
        return self

    def validate(self) -> None:
        for layer in self.layers:
            for channel in CHANNELS:
                if self.events[(layer, channel)] < 1:
                    raise RuntimeError(
                        f"Projection capture did not fire for L{layer} {channel.upper()}"
                    )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


class ProjectionHeadPatch:
    """Replace one projection-head slice at selected target positions."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        head: int,
        head_dim: int,
        target_to_source: Mapping[int, torch.Tensor],
    ) -> None:
        self.head = int(head)
        self.head_dim = int(head_dim)
        self.target_to_source = {
            int(position): tensor.detach().float().cpu()
            for position, tensor in target_to_source.items()
        }
        self.applied = False
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output) or output.ndim != 3:
            raise RuntimeError("Projection patch expects a [B,S,D] tensor")
        if int(output.shape[0]) != 1:
            raise RuntimeError("Projection patch expects batch size 1")
        selected = head_slice(self.head, self.head_dim)
        if selected.stop > int(output.shape[-1]):
            raise RuntimeError(
                f"Head slice {selected.start}:{selected.stop} exceeds projection "
                f"dimension {int(output.shape[-1])}"
            )
        modified = output.clone()
        count = 0
        for target_position, source_full in self.target_to_source.items():
            if not 0 <= target_position < int(output.shape[1]):
                raise RuntimeError(
                    f"Patch position {target_position} outside sequence length "
                    f"{int(output.shape[1])}"
                )
            source_slice = source_full[selected].to(
                device=output.device,
                dtype=output.dtype,
            )
            modified[0, target_position, selected] = source_slice
            count += 1
        self.applied = count > 0
        return modified

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self) -> "ProjectionHeadPatch":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# -----------------------------------------------------------------------------
# Prompt preparation
# -----------------------------------------------------------------------------


@dataclass
class PreparedPair:
    sid: int
    gt: str
    subject: str
    reference: str
    image: Image.Image
    original_batch: Dict[str, Any]
    swapped_batch: Dict[str, Any]
    original_ids: List[int]
    swapped_ids: List[int]
    original_a_positions: List[int]
    original_b_positions: List[int]
    swapped_a_positions: List[int]
    swapped_b_positions: List[int]
    original_prompt_last: int
    swapped_prompt_last: int

    @property
    def original_object_positions(self) -> List[int]:
        return sorted(set(self.original_a_positions + self.original_b_positions))

    @property
    def swapped_object_positions(self) -> List[int]:
        return sorted(set(self.swapped_a_positions + self.swapped_b_positions))

    @property
    def identity_pairs(self) -> List[Tuple[int, int]]:
        return (
            aligned_position_pairs(self.original_a_positions, self.swapped_a_positions)
            + aligned_position_pairs(self.original_b_positions, self.swapped_b_positions)
        )


def prepare_pair(
    *,
    args: argparse.Namespace,
    row: Mapping[str, Any],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    base: Any,
    v3: Any,
    processor: Any,
    device: torch.device,
) -> PreparedPair:
    sid = int(row["sid"])
    prompt = prompt_rows[sid]
    subject = str(prompt["subject"])
    reference = str(prompt["reference"])
    original_question = str(prompt["question_text"])
    swapped_question = base.build_swapped_question(subject, reference)
    image = base.record_image(records_by_sid[sid])
    original_batch = base.make_question_batch(
        processor=processor,
        image=image,
        question_text=original_question,
        device=device,
    )
    swapped_batch = base.make_question_batch(
        processor=processor,
        image=image,
        question_text=swapped_question,
        device=device,
    )
    original_ids = original_batch["input_ids"][0].detach().cpu().tolist()
    swapped_ids = swapped_batch["input_ids"][0].detach().cpu().tolist()
    original_a_span, original_b_span = base.locate_object_spans(
        processor.tokenizer,
        original_ids,
        subject,
        reference,
    )
    swapped_b_span, swapped_a_span = base.locate_object_spans(
        processor.tokenizer,
        swapped_ids,
        reference,
        subject,
    )
    original_a_positions = list(
        map(int, v3.span_positions(original_a_span, args.object_state))
    )
    original_b_positions = list(
        map(int, v3.span_positions(original_b_span, args.object_state))
    )
    swapped_a_positions = list(
        map(int, v3.span_positions(swapped_a_span, args.object_state))
    )
    swapped_b_positions = list(
        map(int, v3.span_positions(swapped_b_span, args.object_state))
    )
    pair = PreparedPair(
        sid=sid,
        gt=str(row["gt"]),
        subject=subject,
        reference=reference,
        image=image,
        original_batch=original_batch,
        swapped_batch=swapped_batch,
        original_ids=original_ids,
        swapped_ids=swapped_ids,
        original_a_positions=original_a_positions,
        original_b_positions=original_b_positions,
        swapped_a_positions=swapped_a_positions,
        swapped_b_positions=swapped_b_positions,
        original_prompt_last=len(original_ids) - 1,
        swapped_prompt_last=len(swapped_ids) - 1,
    )
    _ = pair.identity_pairs
    return pair


def release_pair(pair: Optional[PreparedPair]) -> None:
    if pair is None:
        return
    with contextlib.suppress(Exception):
        pair.image.close()
    for name in ("original_batch", "swapped_batch"):
        batch = getattr(pair, name, None)
        if isinstance(batch, dict):
            batch.clear()


# -----------------------------------------------------------------------------
# Baseline scoring and receiver scan
# -----------------------------------------------------------------------------


def scores_to_logits(scores: Mapping[str, float]) -> List[float]:
    return [float(scores[relation]) for relation in RELATIONS]


@torch.inference_mode()
def run_scores(
    *,
    model: Any,
    batch: Mapping[str, Any],
    base: Any,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    outputs = model(
        **batch,
        use_cache=False,
        return_dict=True,
    )
    relation = base.relation_scores(
        outputs.logits[0, -1],
        dict(relation_token_map),
        gt=None,
    )
    logits = np.asarray(relation["logits"], dtype=np.float64)
    result = {
        "logits": logits.tolist(),
        "scores": {
            relation_name: float(logits[index])
            for index, relation_name in enumerate(RELATIONS)
        },
        "prediction": str(relation["prediction"]),
    }
    del outputs
    return result


def object_head_write(
    *,
    trace: Any,
    head: int,
    target_position: int,
    source_positions: Sequence[int],
) -> Tuple[torch.Tensor, float]:
    lookup = {
        int(global_position): local_index
        for local_index, global_position in enumerate(trace.target_positions)
    }
    if int(target_position) not in lookup:
        raise KeyError(
            f"Target {target_position} absent from trace targets {trace.target_positions}"
        )
    local = int(lookup[int(target_position)])
    source = torch.as_tensor(
        sorted(set(map(int, source_positions))),
        dtype=torch.long,
    )
    if source.numel() == 0:
        raise RuntimeError("Object source group is empty")
    weights = trace.attention_weights[int(head), local].index_select(0, source).float()
    values = trace.value_states[int(head)].index_select(0, source).float()
    pre = torch.einsum("s,sd->d", weights, values)
    weight = trace.o_proj_weight[:, int(head), :].float()
    post = torch.einsum("d,od->o", pre, weight)
    return post, float(weights.sum())


def scan_sample_rows(
    *,
    pair: PreparedPair,
    receiver_layers: Sequence[int],
    original_traces: Mapping[int, Any],
    swapped_traces: Mapping[int, Any],
    attention_helper: Any,
    decoder_layers: Sequence[Any],
    producer_heads: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compatible_by_layer = {
        layer: [
            {"layer": int(p_layer), "head": int(p_head)}
            for p_layer, p_head in producer_heads
            if int(p_layer) < int(layer)
        ]
        for layer in receiver_layers
    }
    for layer in receiver_layers:
        original_trace = original_traces[int(layer)]
        swapped_trace = swapped_traces[int(layer)]
        original_block = original_trace.block_output[0].float()
        swapped_block = swapped_trace.block_output[0].float()
        block_delta = original_block - swapped_block
        denominator = float(block_delta.pow(2).sum())
        attention = attention_helper.resolve_self_attention(decoder_layers[int(layer)])
        shape = resolve_attention_shape(attention)
        trace_heads = int(original_trace.attention_weights.shape[0])
        if trace_heads != shape.n_query_heads:
            raise RuntimeError(
                f"L{layer}: trace heads={trace_heads}, architecture heads={shape.n_query_heads}"
            )
        for query_head in range(shape.n_query_heads):
            original_write, original_mass = object_head_write(
                trace=original_trace,
                head=query_head,
                target_position=pair.original_prompt_last,
                source_positions=pair.original_object_positions,
            )
            swapped_write, swapped_mass = object_head_write(
                trace=swapped_trace,
                head=query_head,
                target_position=pair.swapped_prompt_last,
                source_positions=pair.swapped_object_positions,
            )
            delta = original_write - swapped_write
            dot = float(torch.dot(delta, block_delta))
            projection = dot / denominator if denominator > 1e-12 else float("nan")
            rows.append(
                {
                    "sid": pair.sid,
                    "gt": pair.gt,
                    "layer": int(layer),
                    "head": int(query_head),
                    "kv_head": int(shape.kv_head_for_query(query_head)),
                    "shared_query_heads": json.dumps(
                        shape.shared_query_heads(shape.kv_head_for_query(query_head))
                    ),
                    "compatible_producers": json.dumps(compatible_by_layer[int(layer)]),
                    "object_attention_mass_original": original_mass,
                    "object_attention_mass_swapped": swapped_mass,
                    "object_attention_mass_mean": 0.5 * (original_mass + swapped_mass),
                    "object_write_delta_norm": float(delta.norm()),
                    "cosine_to_prompt_last_block_delta": safe_cosine(delta, block_delta),
                    "projection_fraction_block": projection,
                    "abs_projection_fraction_block": abs(projection),
                    "positive_projection": bool(math.isfinite(projection) and projection > 0),
                    "block_delta_norm": float(block_delta.norm()),
                    "replay_relative_error_original": float(
                        original_trace.replay_relative_error
                    ),
                    "replay_relative_error_swapped": float(
                        swapped_trace.replay_relative_error
                    ),
                }
            )
    return rows


def aggregate_scan(
    rows: Sequence[Mapping[str, Any]],
    rank_metric: str,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["layer"]), int(row["head"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (layer, head), values in sorted(groups.items()):
        result.append(
            {
                "layer": layer,
                "head": head,
                "kv_head": int(values[0]["kv_head"]),
                "shared_query_heads": values[0]["shared_query_heads"],
                "compatible_producers": values[0]["compatible_producers"],
                "N": len(values),
                "mean_projection_fraction_block": safe_mean(
                    value["projection_fraction_block"] for value in values
                ),
                "median_projection_fraction_block": safe_median(
                    value["projection_fraction_block"] for value in values
                ),
                "mean_abs_projection_fraction_block": safe_mean(
                    value["abs_projection_fraction_block"] for value in values
                ),
                "mean_delta_norm": safe_mean(
                    value["object_write_delta_norm"] for value in values
                ),
                "mean_object_attention_mass": safe_mean(
                    value["object_attention_mass_mean"] for value in values
                ),
                "mean_cosine_to_block_delta": safe_mean(
                    value["cosine_to_prompt_last_block_delta"] for value in values
                ),
                "positive_projection_rate": safe_mean(
                    int(bool(value["positive_projection"])) for value in values
                ),
                "max_replay_relative_error": max(
                    max(
                        float(value["replay_relative_error_original"]),
                        float(value["replay_relative_error_swapped"]),
                    )
                    for value in values
                ),
            }
        )
    result.sort(
        key=lambda row: (
            -float(row[rank_metric])
            if math.isfinite(float(row[rank_metric]))
            else float("inf"),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
    return result


def run_scan_phase(
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Mapping[str, Any]],
    producer_heads: Sequence[Tuple[int, int]],
    receiver_layers: Sequence[int],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    base: Any,
    v3: Any,
    attention_helper: Any,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eligible = [
        dict(row) for row in source_rows if status_matches(row, args.scan_status)
    ]
    eligible = stratified_limit(eligible, args.scan_max_samples, args.seed)
    if not eligible:
        raise RuntimeError("No samples eligible for receiver scan")
    print(
        f"Receiver scan: N={len(eligible)}, layers={receiver_layers[0]}.."
        f"{receiver_layers[-1]} ({len(receiver_layers)} layers)",
        flush=True,
    )

    per_sample_path = output_dir / "receiver_scan_per_sample.jsonl"
    if args.save_scan_per_sample:
        per_sample_path.unlink(missing_ok=True)
    errors_path = output_dir / "errors.jsonl"
    all_rows: List[Dict[str, Any]] = []
    device = torch.device(args.device)

    for sample_index, row in enumerate(
        tqdm(eligible, desc=f"receiver-scan:{args.model}"),
        start=1,
    ):
        pair: Optional[PreparedPair] = None
        try:
            pair = prepare_pair(
                args=args,
                row=row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                processor=processor,
                device=device,
            )
            original_result, original_traces = v3.trace_prompt_chunks(
                attention_helper=attention_helper,
                model=model,
                batch=pair.original_batch,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                layers=receiver_layers,
                target_positions=[pair.original_prompt_last],
                chunk_size=args.trace_layer_chunk,
            )
            swapped_result, swapped_traces = v3.trace_prompt_chunks(
                attention_helper=attention_helper,
                model=model,
                batch=pair.swapped_batch,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                layers=receiver_layers,
                target_positions=[pair.swapped_prompt_last],
                chunk_size=args.trace_layer_chunk,
            )
            original_margin = relation_margin(
                scores_to_logits(original_result["scores"]), pair.gt
            )
            swapped_margin = relation_margin(
                scores_to_logits(swapped_result["scores"]), pair.gt
            )
            sample_rows = scan_sample_rows(
                pair=pair,
                receiver_layers=receiver_layers,
                original_traces=original_traces,
                swapped_traces=swapped_traces,
                attention_helper=attention_helper,
                decoder_layers=decoder_layers,
                producer_heads=producer_heads,
            )
            for item in sample_rows:
                item.update(
                    {
                        "original_margin": original_margin,
                        "swapped_margin_original_axis": swapped_margin,
                        "margin_denominator": original_margin - swapped_margin,
                        "original_prediction": str(original_result["prediction"]),
                        "swapped_prediction": str(swapped_result["prediction"]),
                    }
                )
                if args.save_scan_per_sample:
                    append_jsonl(per_sample_path, item)
            all_rows.extend(sample_rows)
            del original_traces, swapped_traces
        except Exception as exc:
            error = {
                "phase": "scan",
                "sid": int(row["sid"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            if args.fail_fast:
                raise
        finally:
            release_pair(pair)
            gc.collect()
            if torch.cuda.is_available() and (
                args.empty_cache_every > 0
                and sample_index % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    metrics = aggregate_scan(all_rows, args.rank_metric)
    top = metrics[: int(args.top_receiver_heads)]
    write_csv(output_dir / "receiver_head_scan.csv", metrics)
    write_json(output_dir / "receiver_top_heads.json", top)
    return metrics, top


# -----------------------------------------------------------------------------
# Receiver Q/K/V causal patch
# -----------------------------------------------------------------------------


def causal_units_from_query_heads(
    *,
    query_heads: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    channels: Sequence[str],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    seen = set()
    for layer, query_head in query_heads:
        attention = attention_helper.resolve_self_attention(decoder_layers[int(layer)])
        shape = resolve_attention_shape(attention)
        if not 0 <= int(query_head) < shape.n_query_heads:
            raise ValueError(
                f"L{layer} query head {query_head} outside 0..{shape.n_query_heads - 1}"
            )
        for channel in channels:
            if channel == "q":
                unit_head = int(query_head)
                kv_head = shape.kv_head_for_query(query_head)
                shared = [int(query_head)]
            else:
                kv_head = shape.kv_head_for_query(query_head)
                unit_head = kv_head
                shared = shape.shared_query_heads(kv_head)
            key = (int(layer), channel, int(unit_head))
            if key in seen:
                continue
            seen.add(key)
            units.append(
                {
                    "unit": (
                        f"L{int(layer)}QH{int(unit_head)}"
                        if channel == "q"
                        else f"L{int(layer)}KVH{int(unit_head)}-{channel.upper()}"
                    ),
                    "layer": int(layer),
                    "channel": channel,
                    "unit_head": int(unit_head),
                    "query_head": int(query_head),
                    "kv_head": int(kv_head),
                    "shared_query_heads": shared,
                    "n_query_heads": shape.n_query_heads,
                    "n_kv_heads": shape.n_kv_heads,
                    "head_dim": (
                        shape.query_head_dim if channel == "q" else shape.kv_head_dim
                    ),
                }
            )
    return units


def capture_pair_projections(
    *,
    pair: PreparedPair,
    layers: Sequence[int],
    model: Any,
    base: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[int, Dict[str, Dict[int, torch.Tensor]]],
    Dict[int, Dict[str, Dict[int, torch.Tensor]]],
]:
    with torch.inference_mode():
        with CaptureProjectionPositions(
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            layers=layers,
            q_positions=[pair.original_prompt_last],
            kv_positions=pair.original_object_positions,
        ) as original_capture:
            original_result = run_scores(
                model=model,
                batch=pair.original_batch,
                base=base,
                relation_token_map=relation_token_map,
            )
        original_capture.validate()

        with CaptureProjectionPositions(
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            layers=layers,
            q_positions=[pair.swapped_prompt_last],
            kv_positions=pair.swapped_object_positions,
        ) as swapped_capture:
            swapped_result = run_scores(
                model=model,
                batch=pair.swapped_batch,
                base=base,
                relation_token_map=relation_token_map,
            )
        swapped_capture.validate()

    original_states = {
        layer: {
            channel: dict(position_map)
            for channel, position_map in channel_map.items()
        }
        for layer, channel_map in original_capture.states.items()
    }
    swapped_states = {
        layer: {
            channel: dict(position_map)
            for channel, position_map in channel_map.items()
        }
        for layer, channel_map in swapped_capture.states.items()
    }
    return original_result, swapped_result, original_states, swapped_states


def patch_position_map(
    *,
    pair: PreparedPair,
    channel: str,
    condition: str,
    original_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    swapped_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    layer: int,
) -> Tuple[str, Dict[int, torch.Tensor]]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if channel == "q":
        if condition == "restore_on_swapped":
            return "swapped", {
                pair.swapped_prompt_last: original_states[layer]["q"][
                    pair.original_prompt_last
                ]
            }
        return "original", {
            pair.original_prompt_last: swapped_states[layer]["q"][
                pair.swapped_prompt_last
            ]
        }

    mapping: Dict[int, torch.Tensor] = {}
    for original_position, swapped_position in pair.identity_pairs:
        if condition == "restore_on_swapped":
            mapping[int(swapped_position)] = original_states[layer][channel][
                int(original_position)
            ]
        else:
            mapping[int(original_position)] = swapped_states[layer][channel][
                int(swapped_position)
            ]
    return (
        "swapped" if condition == "restore_on_swapped" else "original",
        mapping,
    )


@torch.inference_mode()
def run_projection_patch(
    *,
    unit: Mapping[str, Any],
    condition: str,
    pair: PreparedPair,
    original_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    swapped_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    model: Any,
    base: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
) -> Dict[str, Any]:
    layer = int(unit["layer"])
    channel = str(unit["channel"])
    attention = attention_helper.resolve_self_attention(decoder_layers[layer])
    module = projection_module(attention, channel)
    target_side, mapping = patch_position_map(
        pair=pair,
        channel=channel,
        condition=condition,
        original_states=original_states,
        swapped_states=swapped_states,
        layer=layer,
    )
    batch = pair.swapped_batch if target_side == "swapped" else pair.original_batch
    intervention = ProjectionHeadPatch(
        module=module,
        head=int(unit["unit_head"]),
        head_dim=int(unit["head_dim"]),
        target_to_source=mapping,
    )
    try:
        result = run_scores(
            model=model,
            batch=batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        if not intervention.applied:
            raise RuntimeError("Projection intervention hook did not fire")
        result["target_side"] = target_side
        result["target_positions"] = sorted(map(int, mapping))
        return result
    finally:
        intervention.close()


def run_causal_phase(
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Mapping[str, Any]],
    receiver_heads: Sequence[Tuple[int, int]],
    producer_heads: Sequence[Tuple[int, int]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    base: Any,
    v3: Any,
    attention_helper: Any,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    channels = parse_subset(args.channels, CHANNELS, "channels")
    conditions = parse_subset(args.conditions, CONDITIONS, "conditions")
    units = causal_units_from_query_heads(
        query_heads=receiver_heads,
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        channels=channels,
    )
    layers = sorted({int(unit["layer"]) for unit in units})
    eligible = [
        dict(row) for row in source_rows if status_matches(row, args.causal_status)
    ]
    filtered: List[Dict[str, Any]] = []
    for row in eligible:
        original_cached = float(row["baseline_lm_margin"])
        swapped_cached = relation_margin(row["swapped_relation_logits"], str(row["gt"]))
        if abs(original_cached - swapped_cached) < args.min_margin_denominator:
            continue
        if args.causal_require_margin_sign and not (
            original_cached > 0 and swapped_cached < 0
        ):
            continue
        filtered.append(row)
    eligible = stratified_limit(filtered, args.causal_max_samples, args.seed)
    if not eligible:
        raise RuntimeError("No samples eligible for receiver-channel causal patch")

    print(
        f"Receiver causal: N={len(eligible)}, units={len(units)}, "
        f"layers={layers}, conditions={conditions}",
        flush=True,
    )
    output_path = output_dir / "receiver_channel_causal.jsonl"
    errors_path = output_dir / "errors.jsonl"
    existing = read_jsonl(output_path) if args.resume else []
    completed = {
        (
            int(row["sid"]),
            int(row["layer"]),
            str(row["channel"]),
            int(row["unit_head"]),
            str(row["condition"]),
        )
        for row in existing
    }
    rows = list(existing)
    device = torch.device(args.device)

    for sample_index, source_row in enumerate(
        tqdm(eligible, desc=f"receiver-causal:{args.model}"),
        start=1,
    ):
        pair: Optional[PreparedPair] = None
        try:
            pair = prepare_pair(
                args=args,
                row=source_row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                processor=processor,
                device=device,
            )
            (
                original_result,
                swapped_result,
                original_states,
                swapped_states,
            ) = capture_pair_projections(
                pair=pair,
                layers=layers,
                model=model,
                base=base,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
            )
            original_margin = relation_margin(original_result["logits"], pair.gt)
            swapped_margin = relation_margin(swapped_result["logits"], pair.gt)
            denominator = float(original_margin - swapped_margin)
            if abs(denominator) < args.min_margin_denominator:
                continue
            if args.causal_require_margin_sign and not (
                original_margin > 0 and swapped_margin < 0
            ):
                continue

            for unit in units:
                compatible_producers = [
                    {"layer": int(layer), "head": int(head)}
                    for layer, head in producer_heads
                    if int(layer) < int(unit["layer"])
                ]
                if not compatible_producers:
                    continue
                for condition in conditions:
                    key = (
                        pair.sid,
                        int(unit["layer"]),
                        str(unit["channel"]),
                        int(unit["unit_head"]),
                        condition,
                    )
                    if key in completed:
                        continue
                    intervention = run_projection_patch(
                        unit=unit,
                        condition=condition,
                        pair=pair,
                        original_states=original_states,
                        swapped_states=swapped_states,
                        model=model,
                        base=base,
                        relation_token_map=relation_token_map,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                    )
                    intervention_margin = relation_margin(
                        intervention["logits"], pair.gt
                    )
                    if condition == "restore_on_swapped":
                        raw_effect = intervention_margin - swapped_margin
                        normalized = raw_effect / denominator
                        crossed = bool(swapped_margin < 0 <= intervention_margin)
                    else:
                        raw_effect = original_margin - intervention_margin
                        normalized = raw_effect / denominator
                        crossed = bool(original_margin > 0 >= intervention_margin)
                    result_row = {
                        "sid": pair.sid,
                        "gt": pair.gt,
                        "subject": pair.subject,
                        "reference": pair.reference,
                        "unit": str(unit["unit"]),
                        "layer": int(unit["layer"]),
                        "channel": str(unit["channel"]),
                        "unit_head": int(unit["unit_head"]),
                        "query_head": int(unit["query_head"]),
                        "kv_head": int(unit["kv_head"]),
                        "shared_query_heads": list(unit["shared_query_heads"]),
                        "condition": condition,
                        "patch_location": "pre_rope_projection_output",
                        "identity_alignment": "A->A,B->B",
                        "compatible_producers": compatible_producers,
                        "original_margin": original_margin,
                        "swapped_margin_original_axis": swapped_margin,
                        "margin_denominator": denominator,
                        "intervention_margin": intervention_margin,
                        "raw_effect": raw_effect,
                        "normalized_effect": normalized,
                        "expected_positive": bool(raw_effect > 0),
                        "crossed_decision_boundary": crossed,
                        "original_prediction": original_result["prediction"],
                        "swapped_prediction": swapped_result["prediction"],
                        "intervention_prediction": intervention["prediction"],
                        "target_side": intervention["target_side"],
                        "target_positions": intervention["target_positions"],
                        "original_prompt_last": pair.original_prompt_last,
                        "swapped_prompt_last": pair.swapped_prompt_last,
                        "original_a_positions": pair.original_a_positions,
                        "original_b_positions": pair.original_b_positions,
                        "swapped_a_positions": pair.swapped_a_positions,
                        "swapped_b_positions": pair.swapped_b_positions,
                    }
                    append_jsonl(output_path, result_row)
                    rows.append(result_row)
                    completed.add(key)
        except Exception as exc:
            error = {
                "phase": "causal",
                "sid": int(source_row["sid"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            if args.fail_fast:
                raise
        finally:
            release_pair(pair)
            gc.collect()
            if torch.cuda.is_available() and (
                args.empty_cache_every > 0
                and sample_index % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()
    return rows


def summarize_causal(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["layer"]),
                str(row["channel"]),
                int(row["unit_head"]),
                str(row["condition"]),
            )
        ].append(row)
    summary: List[Dict[str, Any]] = []
    for (layer, channel, unit_head, condition), values in sorted(groups.items()):
        summary.append(
            {
                "unit": values[0]["unit"],
                "layer": layer,
                "channel": channel,
                "unit_head": unit_head,
                "query_head": int(values[0]["query_head"]),
                "kv_head": int(values[0]["kv_head"]),
                "shared_query_heads": json.dumps(values[0]["shared_query_heads"]),
                "condition": condition,
                "N": len(values),
                "mean_raw_effect": safe_mean(value["raw_effect"] for value in values),
                "median_raw_effect": safe_median(value["raw_effect"] for value in values),
                "mean_normalized_effect": safe_mean(
                    value["normalized_effect"] for value in values
                ),
                "median_normalized_effect": safe_median(
                    value["normalized_effect"] for value in values
                ),
                "positive_effect_rate": safe_mean(
                    int(bool(value["expected_positive"])) for value in values
                ),
                "crossed_decision_boundary_rate": safe_mean(
                    int(bool(value["crossed_decision_boundary"])) for value in values
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            -float(row["mean_normalized_effect"])
            if math.isfinite(float(row["mean_normalized_effect"]))
            else float("inf"),
            int(row["layer"]),
            str(row["channel"]),
            int(row["unit_head"]),
        )
    )
    return summary


# -----------------------------------------------------------------------------
# Report and main
# -----------------------------------------------------------------------------


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    producer_heads: Sequence[Tuple[int, int]],
    scan_rows: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
    causal_summary: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"model: {args.model}",
        f"source_output_dir: {args.source_output_dir}",
        f"producer_output_dir: {args.producer_output_dir}",
        f"producer_heads: {list(producer_heads)}",
        "",
    ]
    if scan_rows:
        lines.extend(
            [
                "TOP RECEIVER CANDIDATES",
                (
                    "The scan ranks later query heads by the clean-vs-swapped "
                    "object->prompt-last post-WO write. It is candidate selection, "
                    "not causal path identification."
                ),
            ]
        )
        for row in top_rows:
            lines.append(
                f"rank={int(row['rank'])} "
                f"L{int(row['layer'])} H{int(row['head'])} "
                f"KVH={int(row['kv_head'])} "
                f"proj={float(row['mean_projection_fraction_block']):+.6f} "
                f"absProj={float(row['mean_abs_projection_fraction_block']):.6f} "
                f"deltaNorm={float(row['mean_delta_norm']):.6f} "
                f"mass={float(row['mean_object_attention_mass']):.6f} "
                f"positive={float(row['positive_projection_rate']):.4f}"
            )
        lines.append("")
    if causal_summary:
        lines.extend(
            [
                "RECEIVER Q/K/V CHANNEL PATCH",
                (
                    "Q is patched at prompt-last. K/V are patched at identity-aligned "
                    "object tokens. All patches are at q_proj/k_proj/v_proj output "
                    "before RoPE."
                ),
            ]
        )
        for row in causal_summary[:80]:
            lines.append(
                f"{row['unit']} {row['condition']} N={int(row['N'])} "
                f"effect={float(row['mean_normalized_effect']):+.6f} "
                f"raw={float(row['mean_raw_effect']):+.6f} "
                f"positive={float(row['positive_effect_rate']):.4f} "
                f"crossed={float(row['crossed_decision_boundary_rate']):.4f}"
            )
        lines.extend(
            [
                "",
                "Interpretation:",
                "  A strong Q effect means the receiver destination query depends on",
                "  the clean computation at prompt-last.",
                "  A strong K effect means clean object states alter whether those",
                "  source tokens are addressable by the receiver.",
                "  A strong V effect means clean object states carry content that the",
                "  receiver transports to prompt-last.",
                "  These are receiver-channel effects. They do not establish which",
                "  producer supplied the causally relevant part of that channel.",
                "  The next stage is exact producer->receiver path patching.",
            ]
        )
    (output_dir / "report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.trace_layer_chunk < 1:
        raise ValueError("--trace-layer-chunk must be >= 1")
    if args.top_receiver_heads < 1:
        raise ValueError("--top-receiver-heads must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_output_dir)
    producer_dir = Path(args.producer_output_dir)
    source_config_path = source_dir / "config.json"
    source_extraction_path = source_dir / "extraction.jsonl"
    producer_config_path = producer_dir / "config.json"
    producer_top_path = producer_dir / "producer_top_heads.json"
    for path in (
        source_config_path,
        source_extraction_path,
        producer_top_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    producer_config = (
        json.loads(producer_config_path.read_text(encoding="utf-8"))
        if producer_config_path.exists()
        else {}
    )
    source_rows = read_jsonl(source_extraction_path)
    if args.max_samples is not None:
        source_rows = source_rows[: int(args.max_samples)]
    if not source_rows:
        raise RuntimeError("Source extraction is empty")
    if str(source_config.get("model")) != args.model:
        raise RuntimeError(
            f"Source model={source_config.get('model')} but --model={args.model}"
        )
    if str(source_config.get("dataset")) != args.dataset:
        raise RuntimeError(
            f"Source dataset={source_config.get('dataset')} but --dataset={args.dataset}"
        )
    source_object_state = str(source_config.get("object_state", args.object_state))
    if source_object_state != args.object_state:
        raise RuntimeError(
            f"Source object_state={source_object_state}, "
            f"but --object-state={args.object_state}"
        )
    if producer_config:
        if str(producer_config.get("model")) != args.model:
            raise RuntimeError(
                f"Producer model={producer_config.get('model')} "
                f"but --model={args.model}"
            )
        producer_object_state = str(
            producer_config.get("object_state", args.object_state)
        )
        if producer_object_state != args.object_state:
            raise RuntimeError(
                f"Producer object_state={producer_object_state}, "
                f"but --object-state={args.object_state}"
            )

    producer = import_file(Path(args.producer_script), "receiver_qkv_producer")
    v3 = import_file(Path(args.v3_script), "receiver_qkv_v3")
    base = import_file(Path(args.base_script), "receiver_qkv_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "receiver_qkv_attention",
    )

    producer_heads = parse_head_list(args.producer_heads, producer_top_path)
    model = None
    processor = None
    scan_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    causal_rows: List[Dict[str, Any]] = []
    try:
        (
            model,
            processor,
            spec,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer.load_model_bundle(args=args, base=base)
        receiver_layers = parse_receiver_layers(
            args.receiver_layers,
            len(decoder_layers),
            producer_heads,
        )
        for layer in receiver_layers:
            attention = attention_helper.resolve_self_attention(decoder_layers[layer])
            _ = resolve_attention_shape(attention)

        two_object = base.import_two_object_module()
        records, audit = two_object.load_records(
            args.dataset,
            Path(args.data_root),
            args.max_samples,
        )
        records_by_sid = {int(record.sid): record for record in records}
        prompt_rows = base.load_standard_prompts(Path(args.prompt_jsonl))

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": str(source_dir),
            "producer_output_dir": str(producer_dir),
            "source_script_version": source_config.get("script_version"),
            "producer_script_version": producer_config.get("script_version"),
            "producer_heads": [
                {"layer": int(layer), "head": int(head)}
                for layer, head in producer_heads
            ],
            "receiver_layers": receiver_layers,
            "decoder_path": decoder_path,
            "object_state": args.object_state,
            "scan_status": args.scan_status,
            "scan_max_samples": args.scan_max_samples,
            "rank_metric": args.rank_metric,
            "top_receiver_heads": args.top_receiver_heads,
            "causal_status": args.causal_status,
            "causal_max_samples": args.causal_max_samples,
            "channels": parse_subset(args.channels, CHANNELS, "channels"),
            "conditions": parse_subset(args.conditions, CONDITIONS, "conditions"),
            "patch_location": "pre_rope_qkv_projection_output",
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        if args.phase in {"all", "scan"}:
            scan_rows, top_rows = run_scan_phase(
                args=args,
                source_rows=source_rows,
                producer_heads=producer_heads,
                receiver_layers=receiver_layers,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                relation_token_map=relation_token_map,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                attention_helper=attention_helper,
                output_dir=output_dir,
            )
        else:
            scan_path = output_dir / "receiver_head_scan.csv"
            top_path = output_dir / "receiver_top_heads.json"
            if not scan_path.exists() or not top_path.exists():
                raise RuntimeError(
                    "--phase causal requires existing receiver scan outputs"
                )
            with scan_path.open("r", encoding="utf-8") as handle:
                scan_rows = list(csv.DictReader(handle))
            top_rows = json.loads(top_path.read_text(encoding="utf-8"))

        if args.phase in {"all", "causal"}:
            receiver_heads = parse_head_list(
                args.causal_heads,
                output_dir / "receiver_top_heads.json",
            )
            causal_rows = run_causal_phase(
                args=args,
                source_rows=source_rows,
                receiver_heads=receiver_heads,
                producer_heads=producer_heads,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                relation_token_map=relation_token_map,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                attention_helper=attention_helper,
                output_dir=output_dir,
            )

        causal_summary = summarize_causal(causal_rows) if causal_rows else []
        if causal_summary:
            write_csv(
                output_dir / "receiver_channel_summary.csv",
                causal_summary,
            )
        write_report(
            output_dir=output_dir,
            args=args,
            producer_heads=producer_heads,
            scan_rows=scan_rows,
            top_rows=top_rows,
            causal_summary=causal_summary,
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
