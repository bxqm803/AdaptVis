#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IOI-style backward circuit discovery for the COCO two-object spatial task.

This script adapts the core discovery logic from the GPT-2 Small IOI circuit:

    1. Start at the answer logits.
    2. Find attention heads / MLPs whose END-position activations directly affect
       the relation logit difference while later attention heads are frozen.
    3. Treat the strongest late attention heads as receiver/writer nodes.
    4. Trace backward by path-patching earlier sender heads into the receiver's
       Q, K, or V inputs, while intermediate attention heads are frozen and MLPs
       are recomputed.

The script does NOT require free autoregressive generation. By default it scores
relation-token logits at the final prompt position:

    logit(GT relation token) - logit(opposite relation token)

This is analogous to IOI, where the model is evaluated at the END position before
it generates the indirect-object name. It is the first answer TOKEN, not the
first character. The script validates relation-token tokenization and can fail
when canonical relation labels are not single-token continuations.

Phases
------
1. token_check
   Save the exact relation token ids / decoded strings used by the existing
   relation scorer. No generation is run.

2. writer_scan
   Scan all requested attention heads and MLPs at prompt-last. For one sender
   component h:

       * run the original prompt;
       * freeze all attention-head outputs at prompt-last to their original
         activations;
       * replace h(prompt-last) with its swapped-prompt activation;
       * recompute MLPs and the final logits.

   This measures direct paths h -> END/logits through residual connections and
   MLPs, excluding paths mediated by later attention heads.

3. writer_validate
   Re-evaluate the selected writer nodes from writer_top.json on more samples.

4. upstream_path
   For each earlier sender h and selected receiver attention head r:

       C pass:
         * run the original prompt;
         * freeze intermediate attention outputs to original activations;
         * patch h from the swapped prompt;
         * recompute MLPs;
         * capture the resulting r.Q / r.K / r.V.

       D pass:
         * run the original prompt normally;
         * patch only r.Q / r.K / r.V with the value captured in C;
         * recompute the receiver and all downstream layers;
         * measure the relation-logit effect.

   This isolates direct sender -> receiver paths through residual connections and
   MLPs, excluding paths containing intermediate attention heads.

Important limits
----------------
* Current upstream receivers are attention heads; MLP writers are scanned at the
  output stage but are not yet used as Q/K/V receivers.
* The script operates on the language decoder. It does not claim that a selected
  subgraph is complete until faithfulness, completeness, and minimality tests are
  run separately.
* K/V units are deduplicated for grouped-query attention.
* writer_scan is expensive: Qwen-3B has 576 query heads plus 36 MLPs. Use a small
  discovery sample first, then writer_validate on selected nodes.

Recommended commands
--------------------
A. Check answer-token scoring

CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_ioi_backward_circuit_v1.py \
  --phase token_check \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --device cuda:0 \
  --output-dir output/coco_ioi_backward/qwen-3b_token_check \
  --overwrite

B. IOI-style direct writer discovery, all heads + MLPs, 20 samples

CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_ioi_backward_circuit_v1.py \
  --phase writer_scan \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --writer-layers all \
  --writer-components attention,mlp \
  --writer-candidates all \
  --causal-status both_correct \
  --causal-max-samples 20 \
  --top-positive-heads 16 \
  --top-negative-heads 8 \
  --top-mlps 8 \
  --device cuda:0 \
  --output-dir output/coco_ioi_backward/qwen-3b_writer_discovery \
  --overwrite

C. Validate selected writers on 100 samples

CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_ioi_backward_circuit_v1.py \
  --phase writer_validate \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --writer-top-file output/coco_ioi_backward/qwen-3b_writer_discovery/writer_top.json \
  --causal-status both_correct \
  --causal-max-samples 100 \
  --device cuda:0 \
  --output-dir output/coco_ioi_backward/qwen-3b_writer_validation \
  --overwrite

D. Trace all earlier attention heads into top writer Q/K/V inputs

CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_ioi_backward_circuit_v1.py \
  --phase upstream_path \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --writer-top-file output/coco_ioi_backward/qwen-3b_writer_validation/writer_top.json \
  --receiver-writers positive_attention \
  --max-receivers 8 \
  --sender-heads all \
  --sender-mlps none \
  --sender-position-scopes all \
  --upstream-channels q,k,v \
  --receiver-kv-scope all \
  --causal-status both_correct \
  --causal-max-samples 10 \
  --device cuda:0 \
  --output-dir output/coco_ioi_backward/qwen-3b_upstream_discovery \
  --overwrite

Outputs
-------
token_check:
    tokenization.json

writer_scan / writer_validate:
    writer_direct_effect.jsonl or writer_validation_effect.jsonl
    writer_direct_summary.csv or writer_validation_summary.csv
    writer_top.json
    report.txt
    errors.jsonl

upstream_path:
    upstream_path_effect.jsonl
    upstream_path_summary.csv
    upstream_top_edges.json
    report.txt
    errors.jsonl
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-ioi-backward-circuit-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
PHASES = ("token_check", "writer_scan", "writer_validate", "upstream_path")
WRITER_COMPONENTS = ("attention", "mlp")
UPSTREAM_CHANNELS = ("q", "k", "v")
SENDER_SCOPES = ("prompt_last", "objects_identity", "objects_role", "all")
KV_SCOPES = ("objects", "all")


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=("eager",))
    parser.add_argument("--object-state", choices=("last", "mean"), default="last")

    parser.add_argument(
        "--require-single-token-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require each canonical relation to have a one-token plain or leading-"
            "space continuation. This guards against interpreting only the first "
            "subtoken of a multi-token answer."
        ),
    )

    parser.add_argument("--writer-layers", default="all")
    parser.add_argument(
        "--writer-components",
        default="attention,mlp",
        help="Comma-separated subset of attention,mlp.",
    )
    parser.add_argument(
        "--writer-candidates",
        default="all",
        help=(
            "'all', 'top' (read --writer-top-file), or explicit comma-separated "
            "nodes such as 31:4,33:7,34:mlp."
        ),
    )
    parser.add_argument(
        "--writer-top-file",
        default=None,
        help="writer_top.json used by writer_validate or upstream_path.",
    )
    parser.add_argument("--top-positive-heads", type=int, default=16)
    parser.add_argument("--top-negative-heads", type=int, default=8)
    parser.add_argument("--top-mlps", type=int, default=8)

    parser.add_argument(
        "--receiver-writers",
        default="positive_attention",
        choices=("positive_attention", "negative_attention", "all_attention"),
    )
    parser.add_argument("--max-receivers", type=int, default=8)
    parser.add_argument(
        "--sender-heads",
        default="all",
        help="'all', 'none', or comma-separated L:H entries.",
    )
    parser.add_argument(
        "--sender-mlps",
        default="none",
        help="'all', 'none', or comma-separated layer ids.",
    )
    parser.add_argument(
        "--sender-layers",
        default="all",
        help="Layer filter for all sender heads/MLPs.",
    )
    parser.add_argument(
        "--sender-position-scopes",
        default="all",
        help=(
            "Comma-separated subset of prompt_last,objects_identity,objects_role,all."
        ),
    )
    parser.add_argument(
        "--upstream-channels",
        default="q,k,v",
        help="Comma-separated subset of q,k,v.",
    )
    parser.add_argument(
        "--receiver-kv-scope",
        default="all",
        choices=KV_SCOPES,
        help="Positions patched for receiver K/V.",
    )
    parser.add_argument("--top-upstream-edges", type=int, default=100)

    parser.add_argument("--causal-status", choices=STATUSES, default="both_correct")
    parser.add_argument("--causal-max-samples", type=int, default=20)
    parser.add_argument("--min-margin-denominator", type=float, default=1e-4)
    parser.add_argument(
        "--causal-require-margin-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--empty-cache-every", type=int, default=5)

    parser.add_argument(
        "--producer-script",
        default="analyze_coco_producer_qk_ov_v1.py",
    )
    parser.add_argument(
        "--receiver-script",
        default="analyze_coco_receiver_qkv_v1.py",
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


def safe_std(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size >= 2 else float("nan")


def parse_subset(value: str, allowed: Sequence[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    output: List[str] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported {label}={item}; allowed={sorted(allowed_set)}"
            )
        if item not in output:
            output.append(item)
    if not output:
        raise ValueError(f"No {label} selected")
    return output


def parse_layer_spec(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    result: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(part)
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    result = list(dict.fromkeys(result))
    for layer in result:
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside 0..{n_layers - 1}")
    return result


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
    labels = [label for label in RELATIONS if label in groups]
    selected: List[Dict[str, Any]] = []
    cursor = {label: 0 for label in labels}
    while len(selected) < limit:
        progressed = False
        for label in labels:
            index = cursor[label]
            if index < len(groups[label]) and len(selected) < limit:
                selected.append(groups[label][index])
                cursor[label] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: int(item["sid"]))


def output_projection_module(attention: Any) -> torch.nn.Module:
    for name in ("o_proj", "out_proj", "dense", "proj", "c_proj"):
        module = getattr(attention, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError(
        f"Unable to locate attention output projection on {type(attention).__name__}"
    )


def resolve_mlp(layer: Any) -> torch.nn.Module:
    for name in ("mlp", "feed_forward", "ffn", "ff"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError(f"Unable to locate MLP on {type(layer).__name__}")


def tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported module output type: {type(output).__name__}")


def replace_tensor_output(original_output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(original_output):
        return tensor
    if isinstance(original_output, tuple):
        return (tensor, *original_output[1:])
    if isinstance(original_output, list):
        return [tensor, *original_output[1:]]
    raise RuntimeError(f"Unsupported module output type: {type(original_output).__name__}")


# -----------------------------------------------------------------------------
# Node descriptions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterNode:
    kind: str
    layer: int
    head: Optional[int] = None

    @property
    def node(self) -> str:
        if self.kind == "attention":
            return f"L{self.layer}H{int(self.head)}"
        return f"L{self.layer}MLP"


@dataclass(frozen=True)
class SenderNode:
    kind: str
    layer: int
    head: Optional[int] = None

    @property
    def node(self) -> str:
        if self.kind == "attention":
            return f"L{self.layer}H{int(self.head)}"
        return f"L{self.layer}MLP"


@dataclass(frozen=True)
class ReceiverUnit:
    layer: int
    channel: str
    unit_head: int
    query_head: int
    kv_head: int
    shared_query_heads: Tuple[int, ...]

    @property
    def unit(self) -> str:
        if self.channel == "q":
            return f"L{self.layer}QH{self.unit_head}"
        return f"L{self.layer}{self.channel.upper()}H{self.unit_head}"


# -----------------------------------------------------------------------------
# Tokenization validation
# -----------------------------------------------------------------------------


def tokenization_report(
    tokenizer: Any,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "score_mode": "next_token_relation_variants_at_prompt_last",
        "free_generation_required": False,
        "relations": {},
    }
    all_have_single = True
    for relation in RELATIONS:
        plain = tokenizer.encode(relation, add_special_tokens=False)
        spaced = tokenizer.encode(" " + relation, add_special_tokens=False)
        variants = [int(token_id) for token_id in relation_token_map[relation]]
        report["relations"][relation] = {
            "plain_ids": list(map(int, plain)),
            "plain_tokens": [tokenizer.decode([int(token_id)]) for token_id in plain],
            "leading_space_ids": list(map(int, spaced)),
            "leading_space_tokens": [
                tokenizer.decode([int(token_id)]) for token_id in spaced
            ],
            "scorer_variant_ids": variants,
            "scorer_variant_tokens": [
                tokenizer.decode([int(token_id)]) for token_id in variants
            ],
            "has_single_token_plain_or_spaced": bool(
                len(plain) == 1 or len(spaced) == 1
            ),
        }
        all_have_single = all_have_single and bool(
            len(plain) == 1 or len(spaced) == 1
        )
    report["all_relations_have_single_token_continuation"] = all_have_single
    return report


# -----------------------------------------------------------------------------
# Activation capture and IOI-style freezing
# -----------------------------------------------------------------------------


class CaptureWriterActivations:
    """Capture attention pre-WO head vectors and MLP output at selected positions."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layers: Sequence[int],
        positions: Sequence[int],
        capture_attention: bool,
        capture_mlp: bool,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layers = list(map(int, layers))
        self.positions = sorted(set(map(int, positions)))
        self.capture_attention = bool(capture_attention)
        self.capture_mlp = bool(capture_mlp)
        self.attention: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        self.mlp: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        self.handles: List[Any] = []
        self.events: Dict[Tuple[str, int], int] = defaultdict(int)

    def __enter__(self) -> "CaptureWriterActivations":
        for layer_index in self.layers:
            layer = self.decoder_layers[layer_index]
            if self.capture_attention:
                attention = self.attention_helper.resolve_self_attention(layer)
                module = output_projection_module(attention)

                def make_pre_hook(index: int):
                    def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                        if not inputs:
                            raise RuntimeError("W_O pre-hook received no input")
                        tensor = inputs[0]
                        if not torch.is_tensor(tensor) or tensor.ndim != 3:
                            raise RuntimeError("W_O input must be [B,S,D]")
                        if int(tensor.shape[0]) != 1:
                            raise RuntimeError("Capture expects batch size 1")
                        for position in self.positions:
                            if not 0 <= position < int(tensor.shape[1]):
                                raise RuntimeError(
                                    f"Position {position} outside sequence length "
                                    f"{int(tensor.shape[1])}"
                                )
                            self.attention[index][position] = (
                                tensor[0, position].detach().float().cpu()
                            )
                        self.events[("attention", index)] += 1
                    return hook

                self.handles.append(
                    module.register_forward_pre_hook(make_pre_hook(layer_index))
                )

            if self.capture_mlp:
                mlp = resolve_mlp(layer)

                def make_mlp_hook(index: int):
                    def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                        tensor = tensor_from_output(output)
                        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                            raise RuntimeError("MLP output must be [1,S,D]")
                        for position in self.positions:
                            if not 0 <= position < int(tensor.shape[1]):
                                raise RuntimeError(
                                    f"Position {position} outside MLP sequence length "
                                    f"{int(tensor.shape[1])}"
                                )
                            self.mlp[index][position] = (
                                tensor[0, position].detach().float().cpu()
                            )
                        self.events[("mlp", index)] += 1
                        return output
                    return hook

                self.handles.append(
                    mlp.register_forward_hook(make_mlp_hook(layer_index))
                )
        return self

    def validate(self) -> None:
        for layer in self.layers:
            if self.capture_attention and self.events[("attention", layer)] < 1:
                raise RuntimeError(f"Attention capture did not fire at L{layer}")
            if self.capture_mlp and self.events[("mlp", layer)] < 1:
                raise RuntimeError(f"MLP capture did not fire at L{layer}")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


class FreezeAttentionAtPositions:
    """
    Freeze all attention-head pre-WO vectors to original activations at selected
    positions. Optionally replace one sender head with swapped activations.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        original: Mapping[int, Mapping[int, torch.Tensor]],
        positions_by_layer: Mapping[int, Sequence[int]],
        sender: Optional[SenderNode] = None,
        sender_mapping: Optional[Mapping[int, int]] = None,
        swapped: Optional[Mapping[int, Mapping[int, torch.Tensor]]] = None,
    ) -> None:
        self.handles: List[Any] = []
        self.applied: Dict[int, int] = defaultdict(int)
        sender_mapping = dict(sender_mapping or {})

        for layer_index, positions in sorted(positions_by_layer.items()):
            layer = decoder_layers[int(layer_index)]
            attention = attention_helper.resolve_self_attention(layer)
            shape = receiver_module.resolve_attention_shape(attention)
            module = output_projection_module(attention)
            selected_positions = sorted(set(map(int, positions)))

            def make_hook(
                index: int,
                layer_shape: Any,
                layer_positions: Sequence[int],
            ):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    if not inputs:
                        raise RuntimeError("W_O pre-hook received no input")
                    tensor = inputs[0]
                    if not torch.is_tensor(tensor) or tensor.ndim != 3:
                        raise RuntimeError("W_O input must be [B,S,D]")
                    if int(tensor.shape[0]) != 1:
                        raise RuntimeError("Freeze patch expects batch size 1")
                    modified = tensor.clone()
                    count = 0
                    for target_position in layer_positions:
                        clean = original[index][int(target_position)].to(
                            device=tensor.device,
                            dtype=tensor.dtype,
                        )
                        if clean.numel() != int(tensor.shape[-1]):
                            raise RuntimeError(
                                f"L{index} clean W_O input dim {clean.numel()} != "
                                f"current {int(tensor.shape[-1])}"
                            )
                        modified[0, int(target_position)] = clean
                        count += 1

                    if (
                        sender is not None
                        and sender.kind == "attention"
                        and int(sender.layer) == index
                    ):
                        if swapped is None:
                            raise RuntimeError("Sender patch requires swapped activations")
                        head = int(sender.head)
                        start = head * int(layer_shape.query_head_dim)
                        stop = start + int(layer_shape.query_head_dim)
                        for target_position, source_position in sender_mapping.items():
                            if int(target_position) not in layer_positions:
                                continue
                            source = swapped[index][int(source_position)][start:stop].to(
                                device=tensor.device,
                                dtype=tensor.dtype,
                            )
                            modified[0, int(target_position), start:stop] = source
                            count += 1
                    self.applied[index] += count
                    return (modified, *inputs[1:])
                return hook

            self.handles.append(
                module.register_forward_pre_hook(
                    make_hook(layer_index, shape, selected_positions)
                )
            )

    def validate(self) -> None:
        if not self.applied:
            raise RuntimeError("Attention freeze hooks did not fire")

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


class PatchMLPAtPositions:
    def __init__(
        self,
        *,
        mlp: torch.nn.Module,
        target_to_source: Mapping[int, torch.Tensor],
    ) -> None:
        self.target_to_source = {
            int(position): tensor.detach().float().cpu()
            for position, tensor in target_to_source.items()
        }
        self.applied = False
        self.handle = mlp.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        tensor = tensor_from_output(output)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError("MLP patch expects [1,S,D]")
        modified = tensor.clone()
        count = 0
        for position, source in self.target_to_source.items():
            if not 0 <= position < int(tensor.shape[1]):
                raise RuntimeError(
                    f"MLP patch position {position} outside {int(tensor.shape[1])}"
                )
            modified[0, position] = source.to(
                device=tensor.device,
                dtype=tensor.dtype,
            )
            count += 1
        self.applied = count > 0
        return replace_tensor_output(output, modified)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


class CaptureProjectionAtPositions:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        positions_by_layer_channel: Mapping[Tuple[int, str], Sequence[int]],
    ) -> None:
        self.states: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.events: Dict[Tuple[int, str], int] = defaultdict(int)
        self.handles: List[Any] = []
        self.spec = {
            (int(layer), str(channel)): sorted(set(map(int, positions)))
            for (layer, channel), positions in positions_by_layer_channel.items()
        }
        for (layer, channel), positions in sorted(self.spec.items()):
            attention = attention_helper.resolve_self_attention(
                decoder_layers[int(layer)]
            )
            module = receiver_module.projection_module(attention, channel)

            def make_hook(index: int, channel_name: str, selected: Sequence[int]):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    if not torch.is_tensor(output) or output.ndim != 3:
                        raise RuntimeError("Projection output must be [B,S,D]")
                    if int(output.shape[0]) != 1:
                        raise RuntimeError("Projection capture expects batch size 1")
                    for position in selected:
                        if not 0 <= position < int(output.shape[1]):
                            raise RuntimeError(
                                f"Projection position {position} outside "
                                f"{int(output.shape[1])}"
                            )
                        self.states[index][channel_name][position] = (
                            output[0, position].detach().float().cpu()
                        )
                    self.events[(index, channel_name)] += 1
                    return output
                return hook

            self.handles.append(
                module.register_forward_hook(
                    make_hook(layer, channel, positions)
                )
            )

    def validate(self) -> None:
        for key in self.spec:
            if self.events[key] < 1:
                raise RuntimeError(
                    f"Projection capture did not fire at L{key[0]} {key[1].upper()}"
                )

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


# -----------------------------------------------------------------------------
# Writer candidates and top files
# -----------------------------------------------------------------------------


def all_writer_nodes(
    *,
    layers: Sequence[int],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    components: Sequence[str],
) -> List[WriterNode]:
    nodes: List[WriterNode] = []
    for layer in layers:
        if "attention" in components:
            attention = attention_helper.resolve_self_attention(
                decoder_layers[int(layer)]
            )
            shape = receiver_module.resolve_attention_shape(attention)
            nodes.extend(
                WriterNode("attention", int(layer), int(head))
                for head in range(shape.n_query_heads)
            )
        if "mlp" in components:
            nodes.append(WriterNode("mlp", int(layer), None))
    return nodes


def parse_writer_node(text: str) -> WriterNode:
    raw = str(text).strip().lower()
    layer_text, component = raw.split(":", 1)
    layer = int(layer_text)
    if component == "mlp":
        return WriterNode("mlp", layer, None)
    return WriterNode("attention", layer, int(component))


def load_writer_top(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("writer_top.json must contain an object")
    return value


def top_file_nodes(value: Mapping[str, Any]) -> List[WriterNode]:
    nodes: List[WriterNode] = []
    for key in ("positive_attention", "negative_attention", "mlps"):
        for row in value.get(key, []):
            kind = "mlp" if key == "mlps" else "attention"
            nodes.append(
                WriterNode(
                    kind=kind,
                    layer=int(row["layer"]),
                    head=(None if kind == "mlp" else int(row["head"])),
                )
            )
    unique: Dict[str, WriterNode] = {node.node: node for node in nodes}
    return list(unique.values())


def choose_writer_nodes(
    *,
    args: argparse.Namespace,
    layers: Sequence[int],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    components: Sequence[str],
) -> List[WriterNode]:
    text = str(args.writer_candidates).strip().lower()
    if text == "all":
        return all_writer_nodes(
            layers=layers,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver_module,
            components=components,
        )
    if text == "top":
        if not args.writer_top_file:
            raise ValueError("--writer-candidates top requires --writer-top-file")
        return top_file_nodes(load_writer_top(Path(args.writer_top_file)))
    nodes = [parse_writer_node(item) for item in text.split(",") if item.strip()]
    layer_set = set(layers)
    for node in nodes:
        if node.layer not in layer_set:
            raise ValueError(f"Writer {node.node} outside selected writer layers")
        if node.kind not in components:
            raise ValueError(f"Writer {node.node} excluded by --writer-components")
    return nodes


# -----------------------------------------------------------------------------
# Baselines and direct writer path patching
# -----------------------------------------------------------------------------


@torch.inference_mode()
def capture_writer_pair(
    *,
    pair: Any,
    layers: Sequence[int],
    components: Sequence[str],
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    CaptureWriterActivations,
    CaptureWriterActivations,
]:
    with CaptureWriterActivations(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layers=layers,
        positions=[pair.original_prompt_last],
        capture_attention=True,
        capture_mlp="mlp" in components,
    ) as original_capture:
        original_result = receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
    original_capture.validate()

    with CaptureWriterActivations(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layers=layers,
        positions=[pair.swapped_prompt_last],
        capture_attention=True,
        capture_mlp="mlp" in components,
    ) as swapped_capture:
        swapped_result = receiver_module.run_scores(
            model=model,
            batch=pair.swapped_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
    swapped_capture.validate()
    return original_result, swapped_result, original_capture, swapped_capture


@torch.inference_mode()
def run_direct_writer_patch(
    *,
    node: WriterNode,
    pair: Any,
    original_capture: CaptureWriterActivations,
    swapped_capture: CaptureWriterActivations,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
) -> Dict[str, Any]:
    positions_by_layer = {
        int(layer): [int(pair.original_prompt_last)]
        for layer in original_capture.layers
    }
    sender_mapping = {
        int(pair.original_prompt_last): int(pair.swapped_prompt_last)
    }
    freeze = FreezeAttentionAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        original=original_capture.attention,
        positions_by_layer=positions_by_layer,
        sender=(SenderNode("attention", node.layer, node.head) if node.kind == "attention" else None),
        sender_mapping=sender_mapping,
        swapped=(swapped_capture.attention if node.kind == "attention" else None),
    )
    mlp_patch: Optional[PatchMLPAtPositions] = None
    try:
        if node.kind == "mlp":
            mlp_patch = PatchMLPAtPositions(
                mlp=resolve_mlp(decoder_layers[int(node.layer)]),
                target_to_source={
                    int(pair.original_prompt_last): swapped_capture.mlp[
                        int(node.layer)
                    ][int(pair.swapped_prompt_last)]
                },
            )
        result = receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        freeze.validate()
        if mlp_patch is not None and not mlp_patch.applied:
            raise RuntimeError("MLP writer patch did not fire")
        return result
    finally:
        if mlp_patch is not None:
            mlp_patch.close()
        freeze.close()


def writer_output_paths(phase: str, output_dir: Path) -> Tuple[Path, Path]:
    if phase == "writer_validate":
        return (
            output_dir / "writer_validation_effect.jsonl",
            output_dir / "writer_validation_summary.csv",
        )
    return (
        output_dir / "writer_direct_effect.jsonl",
        output_dir / "writer_direct_summary.csv",
    )


def summarize_writer_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["kind"]),
                int(row["layer"]),
                int(row.get("head", -1)),
            )
        ].append(row)
    summary: List[Dict[str, Any]] = []
    for (kind, layer, head), values in groups.items():
        summary.append(
            {
                "node": values[0]["node"],
                "kind": kind,
                "layer": layer,
                "head": ("" if head < 0 else head),
                "N": len(values),
                "mean_raw_effect": safe_mean(v["raw_effect"] for v in values),
                "median_raw_effect": safe_median(v["raw_effect"] for v in values),
                "std_raw_effect": safe_std(v["raw_effect"] for v in values),
                "mean_normalized_effect": safe_mean(
                    v["normalized_effect"] for v in values
                ),
                "median_normalized_effect": safe_median(
                    v["normalized_effect"] for v in values
                ),
                "std_normalized_effect": safe_std(
                    v["normalized_effect"] for v in values
                ),
                "positive_effect_rate": safe_mean(
                    int(bool(v["expected_positive"])) for v in values
                ),
                "crossed_decision_boundary_rate": safe_mean(
                    int(bool(v["crossed_decision_boundary"])) for v in values
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            -abs(float(row["mean_normalized_effect"])),
            int(row["layer"]),
            str(row["kind"]),
            str(row["head"]),
        )
    )
    for rank, row in enumerate(summary, start=1):
        row["rank_abs_effect"] = rank
    return summary


def build_writer_top(
    *,
    summary: Sequence[Mapping[str, Any]],
    top_positive_heads: int,
    top_negative_heads: int,
    top_mlps: int,
) -> Dict[str, Any]:
    attention = [row for row in summary if row["kind"] == "attention"]
    mlps = [row for row in summary if row["kind"] == "mlp"]
    positive = sorted(
        [row for row in attention if float(row["mean_normalized_effect"]) > 0],
        key=lambda row: -float(row["mean_normalized_effect"]),
    )[: max(0, int(top_positive_heads))]
    negative = sorted(
        [row for row in attention if float(row["mean_normalized_effect"]) < 0],
        key=lambda row: float(row["mean_normalized_effect"]),
    )[: max(0, int(top_negative_heads))]
    mlp_selected = sorted(
        mlps,
        key=lambda row: -abs(float(row["mean_normalized_effect"])),
    )[: max(0, int(top_mlps))]

    def compact(row: Mapping[str, Any]) -> Dict[str, Any]:
        result = {
            "node": row["node"],
            "kind": row["kind"],
            "layer": int(row["layer"]),
            "mean_normalized_effect": float(row["mean_normalized_effect"]),
            "median_normalized_effect": float(row["median_normalized_effect"]),
            "positive_effect_rate": float(row["positive_effect_rate"]),
            "N": int(row["N"]),
        }
        if row["kind"] == "attention":
            result["head"] = int(row["head"])
        return result

    return {
        "script_version": SCRIPT_VERSION,
        "selection_metric": "mean_normalized_direct_path_effect",
        "positive_attention": [compact(row) for row in positive],
        "negative_attention": [compact(row) for row in negative],
        "mlps": [compact(row) for row in mlp_selected],
    }


# -----------------------------------------------------------------------------
# Upstream path patching
# -----------------------------------------------------------------------------


def sender_position_mapping(
    pair: Any,
    scope: str,
) -> Dict[int, int]:
    if scope == "prompt_last":
        return {
            int(pair.original_prompt_last): int(pair.swapped_prompt_last)
        }
    if scope == "objects_identity":
        if not pair.original_a_positions or not pair.original_b_positions:
            raise RuntimeError("Missing object positions")
        return {
            int(pair.original_a_positions[-1]): int(pair.swapped_a_positions[-1]),
            int(pair.original_b_positions[-1]): int(pair.swapped_b_positions[-1]),
        }
    if scope == "objects_role":
        if not pair.original_a_positions or not pair.original_b_positions:
            raise RuntimeError("Missing object positions")
        return {
            int(pair.original_a_positions[-1]): int(pair.swapped_b_positions[-1]),
            int(pair.original_b_positions[-1]): int(pair.swapped_a_positions[-1]),
        }
    if scope == "all":
        if len(pair.original_ids) != len(pair.swapped_ids):
            raise RuntimeError(
                "sender scope 'all' requires equal original/swapped sequence lengths; "
                "use prompt_last, objects_identity, or objects_role instead"
            )
        return {index: index for index in range(len(pair.original_ids))}
    raise ValueError(scope)


def receiver_channel_positions(
    pair: Any,
    channel: str,
    kv_scope: str,
) -> List[int]:
    if channel == "q":
        return [int(pair.original_prompt_last)]
    if kv_scope == "objects":
        return list(map(int, pair.original_object_positions))
    if kv_scope == "all":
        return list(range(len(pair.original_ids)))
    raise ValueError(kv_scope)


def load_receiver_writer_nodes(
    *,
    path: Path,
    group: str,
    max_receivers: int,
) -> List[WriterNode]:
    value = load_writer_top(path)
    if group == "positive_attention":
        rows = list(value.get("positive_attention", []))
    elif group == "negative_attention":
        rows = list(value.get("negative_attention", []))
    elif group == "all_attention":
        rows = list(value.get("positive_attention", [])) + list(
            value.get("negative_attention", [])
        )
    else:
        raise ValueError(group)
    nodes: List[WriterNode] = []
    seen = set()
    for row in rows:
        node = WriterNode("attention", int(row["layer"]), int(row["head"]))
        if node.node in seen:
            continue
        seen.add(node.node)
        nodes.append(node)
        if max_receivers > 0 and len(nodes) >= max_receivers:
            break
    if not nodes:
        raise RuntimeError(f"No receiver writer heads found in {path}")
    return nodes


def build_receiver_units(
    *,
    writers: Sequence[WriterNode],
    channels: Sequence[str],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> List[ReceiverUnit]:
    units: List[ReceiverUnit] = []
    seen = set()
    for writer in writers:
        attention = attention_helper.resolve_self_attention(
            decoder_layers[int(writer.layer)]
        )
        shape = receiver_module.resolve_attention_shape(attention)
        query_head = int(writer.head)
        kv_head = int(shape.kv_head_for_query(query_head))
        for channel in channels:
            if channel == "q":
                unit_head = query_head
                shared = (query_head,)
            else:
                unit_head = kv_head
                shared = tuple(shape.shared_query_heads(kv_head))
            key = (int(writer.layer), str(channel), int(unit_head))
            if key in seen:
                continue
            seen.add(key)
            units.append(
                ReceiverUnit(
                    layer=int(writer.layer),
                    channel=str(channel),
                    unit_head=int(unit_head),
                    query_head=query_head,
                    kv_head=kv_head,
                    shared_query_heads=shared,
                )
            )
    return units


def parse_sender_heads(
    *,
    value: str,
    layers: Sequence[int],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> List[SenderNode]:
    text = str(value).strip().lower()
    if text == "none":
        return []
    if text == "all":
        nodes: List[SenderNode] = []
        for layer in layers:
            attention = attention_helper.resolve_self_attention(
                decoder_layers[int(layer)]
            )
            shape = receiver_module.resolve_attention_shape(attention)
            nodes.extend(
                SenderNode("attention", int(layer), int(head))
                for head in range(shape.n_query_heads)
            )
        return nodes
    nodes = []
    for item in text.split(","):
        if not item.strip():
            continue
        layer_text, head_text = item.strip().split(":", 1)
        nodes.append(SenderNode("attention", int(layer_text), int(head_text)))
    return nodes


def parse_sender_mlps(value: str, layers: Sequence[int]) -> List[SenderNode]:
    text = str(value).strip().lower()
    if text == "none":
        return []
    if text == "all":
        return [SenderNode("mlp", int(layer), None) for layer in layers]
    return [
        SenderNode("mlp", int(item.strip()), None)
        for item in text.split(",")
        if item.strip()
    ]


@torch.inference_mode()
def capture_upstream_pair(
    *,
    pair: Any,
    layers: Sequence[int],
    original_positions: Sequence[int],
    swapped_positions: Sequence[int],
    sender_mlps_present: bool,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    CaptureWriterActivations,
    CaptureWriterActivations,
]:
    with CaptureWriterActivations(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layers=layers,
        positions=original_positions,
        capture_attention=True,
        capture_mlp=sender_mlps_present,
    ) as original_capture:
        original_result = receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
    original_capture.validate()

    with CaptureWriterActivations(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layers=layers,
        positions=swapped_positions,
        capture_attention=True,
        capture_mlp=sender_mlps_present,
    ) as swapped_capture:
        swapped_result = receiver_module.run_scores(
            model=model,
            batch=pair.swapped_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
    swapped_capture.validate()
    return original_result, swapped_result, original_capture, swapped_capture


@torch.inference_mode()
def run_c_pass_capture_receivers(
    *,
    sender: SenderNode,
    sender_mapping: Mapping[int, int],
    receiver_units: Sequence[ReceiverUnit],
    pair: Any,
    original_capture: CaptureWriterActivations,
    swapped_capture: CaptureWriterActivations,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    kv_scope: str,
) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
    active_units = [unit for unit in receiver_units if sender.layer < unit.layer]
    if not active_units:
        return {}

    positions_by_layer_channel: Dict[Tuple[int, str], List[int]] = {}
    freeze_positions = set(map(int, sender_mapping.keys()))
    for unit in active_units:
        positions = receiver_channel_positions(pair, unit.channel, kv_scope)
        positions_by_layer_channel.setdefault((unit.layer, unit.channel), []).extend(
            positions
        )
        freeze_positions.update(positions)

    positions_by_layer = {
        int(layer): sorted(freeze_positions)
        for layer in original_capture.layers
    }
    freeze = FreezeAttentionAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        original=original_capture.attention,
        positions_by_layer=positions_by_layer,
        sender=(sender if sender.kind == "attention" else None),
        sender_mapping=sender_mapping,
        swapped=(swapped_capture.attention if sender.kind == "attention" else None),
    )
    mlp_patch: Optional[PatchMLPAtPositions] = None
    projection_capture = CaptureProjectionAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        positions_by_layer_channel=positions_by_layer_channel,
    )
    try:
        if sender.kind == "mlp":
            mlp_patch = PatchMLPAtPositions(
                mlp=resolve_mlp(decoder_layers[int(sender.layer)]),
                target_to_source={
                    int(target): swapped_capture.mlp[int(sender.layer)][int(source)]
                    for target, source in sender_mapping.items()
                },
            )
        receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        freeze.validate()
        projection_capture.validate()
        if mlp_patch is not None and not mlp_patch.applied:
            raise RuntimeError("Sender MLP patch did not fire")
        return {
            int(layer): {
                str(channel): dict(position_map)
                for channel, position_map in channel_map.items()
            }
            for layer, channel_map in projection_capture.states.items()
        }
    finally:
        projection_capture.close()
        if mlp_patch is not None:
            mlp_patch.close()
        freeze.close()


@torch.inference_mode()
def run_d_receiver_patch(
    *,
    unit: ReceiverUnit,
    c_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    pair: Any,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    kv_scope: str,
) -> Dict[str, Any]:
    attention = attention_helper.resolve_self_attention(
        decoder_layers[int(unit.layer)]
    )
    shape = receiver_module.resolve_attention_shape(attention)
    module = receiver_module.projection_module(attention, unit.channel)
    if unit.channel == "q":
        head_dim = int(shape.query_head_dim)
    else:
        head_dim = int(shape.kv_head_dim)
    positions = receiver_channel_positions(pair, unit.channel, kv_scope)
    mapping = {
        int(position): c_states[int(unit.layer)][unit.channel][int(position)]
        for position in positions
    }
    patch = receiver_module.ProjectionHeadPatch(
        module=module,
        head=int(unit.unit_head),
        head_dim=head_dim,
        target_to_source=mapping,
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


def summarize_upstream_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["sender"]),
                str(row["receiver_unit"]),
                str(row["channel"]),
                str(row["sender_position_scope"]),
            )
        ].append(row)
    summary: List[Dict[str, Any]] = []
    for (sender, receiver_unit, channel, scope), values in groups.items():
        summary.append(
            {
                "sender": sender,
                "sender_kind": values[0]["sender_kind"],
                "sender_layer": int(values[0]["sender_layer"]),
                "sender_head": values[0]["sender_head"],
                "receiver_unit": receiver_unit,
                "receiver_layer": int(values[0]["receiver_layer"]),
                "receiver_query_head": int(values[0]["receiver_query_head"]),
                "receiver_unit_head": int(values[0]["receiver_unit_head"]),
                "receiver_kv_head": int(values[0]["receiver_kv_head"]),
                "shared_query_heads": json.dumps(values[0]["shared_query_heads"]),
                "channel": channel,
                "sender_position_scope": scope,
                "receiver_kv_scope": values[0]["receiver_kv_scope"],
                "N": len(values),
                "mean_raw_effect": safe_mean(v["raw_effect"] for v in values),
                "median_raw_effect": safe_median(v["raw_effect"] for v in values),
                "std_raw_effect": safe_std(v["raw_effect"] for v in values),
                "mean_normalized_effect": safe_mean(
                    v["normalized_effect"] for v in values
                ),
                "median_normalized_effect": safe_median(
                    v["normalized_effect"] for v in values
                ),
                "std_normalized_effect": safe_std(
                    v["normalized_effect"] for v in values
                ),
                "positive_effect_rate": safe_mean(
                    int(bool(v["expected_positive"])) for v in values
                ),
                "crossed_decision_boundary_rate": safe_mean(
                    int(bool(v["crossed_decision_boundary"])) for v in values
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            -abs(float(row["mean_normalized_effect"])),
            int(row["receiver_layer"]),
            int(row["sender_layer"]),
            str(row["sender"]),
        )
    )
    for rank, row in enumerate(summary, start=1):
        row["rank_abs_effect"] = rank
    return summary


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------


def write_writer_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    summary: Sequence[Mapping[str, Any]],
    top: Mapping[str, Any],
) -> None:
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"phase: {args.phase}",
        f"model: {args.model}",
        "score_mode: next-token relation logits at prompt-last; no free generation",
        "",
        "TOP POSITIVE ATTENTION WRITERS",
    ]
    for row in top.get("positive_attention", []):
        lines.append(
            f"{row['node']} effect={float(row['mean_normalized_effect']):+.6f} "
            f"median={float(row['median_normalized_effect']):+.6f} "
            f"positive={float(row['positive_effect_rate']):.4f} N={int(row['N'])}"
        )
    lines.extend(["", "TOP NEGATIVE ATTENTION WRITERS"])
    for row in top.get("negative_attention", []):
        lines.append(
            f"{row['node']} effect={float(row['mean_normalized_effect']):+.6f} "
            f"median={float(row['median_normalized_effect']):+.6f} "
            f"positive={float(row['positive_effect_rate']):.4f} N={int(row['N'])}"
        )
    lines.extend(["", "TOP MLP WRITERS"])
    for row in top.get("mlps", []):
        lines.append(
            f"{row['node']} effect={float(row['mean_normalized_effect']):+.6f} "
            f"median={float(row['median_normalized_effect']):+.6f} "
            f"positive={float(row['positive_effect_rate']):.4f} N={int(row['N'])}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  Positive effect: swapped activation inserted into the original run",
            "  reduces the original GT-vs-opposite relation margin.",
            "  Negative effect: the component directly opposes the original answer",
            "  axis or behaves as a negative/backup writer.",
            "  All later attention outputs at prompt-last are frozen to original",
            "  activations; MLPs are recomputed.",
        ]
    )
    (output_dir / "report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_upstream_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    summary: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"phase: {args.phase}",
        f"model: {args.model}",
        "",
        "TOP UPSTREAM PATHS",
    ]
    for row in summary[: min(100, len(summary))]:
        lines.append(
            f"{row['sender']} -> {row['receiver_unit']} "
            f"scope={row['sender_position_scope']} "
            f"effect={float(row['mean_normalized_effect']):+.6f} "
            f"median={float(row['median_normalized_effect']):+.6f} "
            f"positive={float(row['positive_effect_rate']):.4f} N={int(row['N'])}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  This is an IOI-style direct sender->receiver path patch.",
            "  Intermediate attention outputs are frozen to clean/original values.",
            "  MLPs are recomputed, the receiver Q/K/V input is captured, and only",
            "  that receiver channel is patched into a normal original run.",
            "  Positive effect means the path carries information supporting the",
            "  original GT-vs-opposite relation margin.",
        ]
    )
    (output_dir / "report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Main phase implementations
# -----------------------------------------------------------------------------


def load_source_rows(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    source_dir = Path(args.source_output_dir)
    config_path = source_dir / "config.json"
    extraction_path = source_dir / "extraction.jsonl"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not extraction_path.exists():
        raise FileNotFoundError(extraction_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("model")) != args.model:
        raise RuntimeError(
            f"Source model={config.get('model')} but --model={args.model}"
        )
    if str(config.get("dataset")) != args.dataset:
        raise RuntimeError(
            f"Source dataset={config.get('dataset')} but --dataset={args.dataset}"
        )
    rows = read_jsonl(extraction_path)
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    return config, rows


def eligible_rows(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected = [dict(row) for row in rows if status_matches(row, args.causal_status)]
    filtered: List[Dict[str, Any]] = []
    for row in selected:
        original_margin = float(row["baseline_lm_margin"])
        swapped_margin = relation_margin(row["swapped_relation_logits"], str(row["gt"]))
        denominator = original_margin - swapped_margin
        if abs(denominator) < args.min_margin_denominator:
            continue
        if args.causal_require_margin_sign and not (
            original_margin > 0 and swapped_margin < 0
        ):
            continue
        filtered.append(row)
    return stratified_limit(filtered, args.causal_max_samples, args.seed)


def prepare_data_helpers(args: argparse.Namespace, base: Any) -> Tuple[Dict[int, Any], Mapping[int, Mapping[str, Any]], Any]:
    two_object = base.import_two_object_module()
    records, audit = two_object.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    records_by_sid = {int(record.sid): record for record in records}
    prompt_rows = base.load_standard_prompts(Path(args.prompt_jsonl))
    return records_by_sid, prompt_rows, audit


def run_writer_phase(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_rows: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    v3: Any,
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    writer_layers = parse_layer_spec(args.writer_layers, len(decoder_layers))
    components = parse_subset(
        args.writer_components,
        WRITER_COMPONENTS,
        "writer components",
    )
    if args.phase == "writer_validate":
        if not args.writer_top_file:
            raise ValueError("writer_validate requires --writer-top-file")
        args.writer_candidates = "top"
    nodes = choose_writer_nodes(
        args=args,
        layers=writer_layers,
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        components=components,
    )
    rows_to_run = eligible_rows(args, source_rows)
    if not rows_to_run:
        raise RuntimeError("No eligible samples for writer scan")

    output_path, summary_path = writer_output_paths(args.phase, output_dir)
    errors_path = output_dir / "errors.jsonl"
    existing = read_jsonl(output_path) if args.resume else []
    completed = {
        (int(row["sid"]), str(row["node"]))
        for row in existing
    }
    all_rows = list(existing)

    print(
        f"Writer path scan: N={len(rows_to_run)}, nodes={len(nodes)}, "
        f"layers={writer_layers[0]}..{writer_layers[-1]}",
        flush=True,
    )
    for sample_index, source_row in enumerate(
        tqdm(rows_to_run, desc=f"writer-path:{args.model}"),
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
            (
                original_result,
                swapped_result,
                original_capture,
                swapped_capture,
            ) = capture_writer_pair(
                pair=pair,
                layers=list(range(len(decoder_layers))),
                components=components,
                model=model,
                decoder_layers=decoder_layers,
                relation_token_map=relation_token_map,
                base=base,
                receiver_module=receiver_module,
                attention_helper=attention_helper,
            )
            gt = str(pair.gt)
            original_margin = relation_margin(original_result["logits"], gt)
            swapped_margin = relation_margin(swapped_result["logits"], gt)
            denominator = float(original_margin - swapped_margin)
            if abs(denominator) < args.min_margin_denominator:
                continue
            if args.causal_require_margin_sign and not (
                original_margin > 0 and swapped_margin < 0
            ):
                continue

            for node in nodes:
                key = (int(pair.sid), node.node)
                if key in completed:
                    continue
                intervention = run_direct_writer_patch(
                    node=node,
                    pair=pair,
                    original_capture=original_capture,
                    swapped_capture=swapped_capture,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                )
                intervention_margin = relation_margin(intervention["logits"], gt)
                raw_effect = float(original_margin - intervention_margin)
                normalized = float(raw_effect / denominator)
                crossed = bool(original_margin > 0 >= intervention_margin)
                row = {
                    "script_version": SCRIPT_VERSION,
                    "phase": args.phase,
                    "model": args.model,
                    "sid": int(pair.sid),
                    "gt": gt,
                    "generation_pair_status": source_row["generation_pair_status"],
                    "node": node.node,
                    "kind": node.kind,
                    "layer": int(node.layer),
                    "head": (-1 if node.head is None else int(node.head)),
                    "position": "prompt_last",
                    "path_target": "end_residual_logits",
                    "intermediate_attention": "frozen_to_original_at_prompt_last",
                    "mlps": "recomputed",
                    "score_mode": "next_token_relation_variants_at_prompt_last",
                    "original_margin": original_margin,
                    "swapped_margin_fixed_axis": swapped_margin,
                    "intervention_margin_fixed_axis": intervention_margin,
                    "margin_denominator": denominator,
                    "raw_effect": raw_effect,
                    "normalized_effect": normalized,
                    "expected_positive": bool(raw_effect > 0),
                    "crossed_decision_boundary": crossed,
                    "original_prediction": original_result["prediction"],
                    "swapped_prediction": swapped_result["prediction"],
                    "intervention_prediction": intervention["prediction"],
                }
                append_jsonl(output_path, row)
                all_rows.append(row)
                completed.add(key)
        except Exception as exc:
            error = {
                "phase": args.phase,
                "sid": int(source_row["sid"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            print(
                f"\n[ERROR sid={source_row['sid']}] {type(exc).__name__}: {exc}",
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
                f"[writer {sample_index}/{len(rows_to_run)}] rows={len(all_rows)}",
                flush=True,
            )

    summary = summarize_writer_rows(all_rows)
    top = build_writer_top(
        summary=summary,
        top_positive_heads=args.top_positive_heads,
        top_negative_heads=args.top_negative_heads,
        top_mlps=args.top_mlps,
    )
    write_csv(summary_path, summary)
    write_json(output_dir / "writer_top.json", top)
    write_writer_report(
        output_dir=output_dir,
        args=args,
        summary=summary,
        top=top,
    )


def run_upstream_phase(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_rows: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    v3: Any,
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    if not args.writer_top_file:
        raise ValueError("upstream_path requires --writer-top-file")
    writer_nodes = load_receiver_writer_nodes(
        path=Path(args.writer_top_file),
        group=args.receiver_writers,
        max_receivers=args.max_receivers,
    )
    channels = parse_subset(
        args.upstream_channels,
        UPSTREAM_CHANNELS,
        "upstream channels",
    )
    sender_scopes = parse_subset(
        args.sender_position_scopes,
        SENDER_SCOPES,
        "sender position scopes",
    )
    receiver_units = build_receiver_units(
        writers=writer_nodes,
        channels=channels,
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
    )
    sender_layers = parse_layer_spec(args.sender_layers, len(decoder_layers))
    senders = parse_sender_heads(
        value=args.sender_heads,
        layers=sender_layers,
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
    ) + parse_sender_mlps(args.sender_mlps, sender_layers)
    if not senders:
        raise RuntimeError("No upstream senders selected")
    # Senders after every receiver can never affect a receiver input.
    latest_receiver = max(unit.layer for unit in receiver_units)
    senders = [sender for sender in senders if sender.layer < latest_receiver]

    rows_to_run = eligible_rows(args, source_rows)
    if not rows_to_run:
        raise RuntimeError("No eligible samples for upstream path scan")

    output_path = output_dir / "upstream_path_effect.jsonl"
    errors_path = output_dir / "errors.jsonl"
    existing = read_jsonl(output_path) if args.resume else []
    completed = {
        (
            int(row["sid"]),
            str(row["sender"]),
            str(row["receiver_unit"]),
            str(row["sender_position_scope"]),
        )
        for row in existing
    }
    all_rows = list(existing)

    all_capture_layers = list(range(len(decoder_layers)))
    print(
        f"Upstream path scan: N={len(rows_to_run)}, senders={len(senders)}, "
        f"receiver_units={len(receiver_units)}, scopes={sender_scopes}",
        flush=True,
    )

    for sample_index, source_row in enumerate(
        tqdm(rows_to_run, desc=f"upstream-path:{args.model}"),
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
            # Capture original target positions separately from swapped source positions.
            original_positions = set()
            swapped_positions = set()
            for scope in sender_scopes:
                scope_mapping = sender_position_mapping(pair, scope)
                original_positions.update(scope_mapping.keys())
                swapped_positions.update(scope_mapping.values())
            for unit in receiver_units:
                original_positions.update(
                    receiver_channel_positions(pair, unit.channel, args.receiver_kv_scope)
                )
            original_result, swapped_result, original_capture, swapped_capture = (
                capture_upstream_pair(
                    pair=pair,
                    layers=all_capture_layers,
                    original_positions=sorted(original_positions),
                    swapped_positions=sorted(swapped_positions),
                    sender_mlps_present=any(sender.kind == "mlp" for sender in senders),
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                )
            )
            gt = str(pair.gt)
            original_margin = relation_margin(original_result["logits"], gt)
            swapped_margin = relation_margin(swapped_result["logits"], gt)
            denominator = float(original_margin - swapped_margin)
            if abs(denominator) < args.min_margin_denominator:
                continue
            if args.causal_require_margin_sign and not (
                original_margin > 0 and swapped_margin < 0
            ):
                continue

            for scope in sender_scopes:
                mapping = sender_position_mapping(pair, scope)
                for sender in senders:
                    active_units = [
                        unit for unit in receiver_units if sender.layer < unit.layer
                    ]
                    if not active_units:
                        continue
                    c_states = run_c_pass_capture_receivers(
                        sender=sender,
                        sender_mapping=mapping,
                        receiver_units=active_units,
                        pair=pair,
                        original_capture=original_capture,
                        swapped_capture=swapped_capture,
                        model=model,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        base=base,
                        receiver_module=receiver_module,
                        attention_helper=attention_helper,
                        kv_scope=args.receiver_kv_scope,
                    )
                    for unit in active_units:
                        key = (
                            int(pair.sid),
                            sender.node,
                            unit.unit,
                            scope,
                        )
                        if key in completed:
                            continue
                        intervention = run_d_receiver_patch(
                            unit=unit,
                            c_states=c_states,
                            pair=pair,
                            model=model,
                            decoder_layers=decoder_layers,
                            relation_token_map=relation_token_map,
                            base=base,
                            receiver_module=receiver_module,
                            attention_helper=attention_helper,
                            kv_scope=args.receiver_kv_scope,
                        )
                        intervention_margin = relation_margin(
                            intervention["logits"], gt
                        )
                        raw_effect = float(original_margin - intervention_margin)
                        normalized = float(raw_effect / denominator)
                        crossed = bool(original_margin > 0 >= intervention_margin)
                        row = {
                            "script_version": SCRIPT_VERSION,
                            "phase": "upstream_path",
                            "model": args.model,
                            "sid": int(pair.sid),
                            "gt": gt,
                            "generation_pair_status": source_row[
                                "generation_pair_status"
                            ],
                            "sender": sender.node,
                            "sender_kind": sender.kind,
                            "sender_layer": int(sender.layer),
                            "sender_head": (
                                "" if sender.head is None else int(sender.head)
                            ),
                            "sender_position_scope": scope,
                            "receiver_unit": unit.unit,
                            "receiver_layer": int(unit.layer),
                            "receiver_query_head": int(unit.query_head),
                            "receiver_unit_head": int(unit.unit_head),
                            "receiver_kv_head": int(unit.kv_head),
                            "shared_query_heads": list(unit.shared_query_heads),
                            "channel": unit.channel,
                            "receiver_kv_scope": args.receiver_kv_scope,
                            "path_definition": (
                                "sender_to_receiver_through_residual_and_mlps_only"
                            ),
                            "intermediate_attention": "frozen_to_original",
                            "mlps": "recomputed",
                            "score_mode": (
                                "next_token_relation_variants_at_prompt_last"
                            ),
                            "original_margin": original_margin,
                            "swapped_margin_fixed_axis": swapped_margin,
                            "intervention_margin_fixed_axis": intervention_margin,
                            "margin_denominator": denominator,
                            "raw_effect": raw_effect,
                            "normalized_effect": normalized,
                            "expected_positive": bool(raw_effect > 0),
                            "crossed_decision_boundary": crossed,
                            "original_prediction": original_result["prediction"],
                            "swapped_prediction": swapped_result["prediction"],
                            "intervention_prediction": intervention["prediction"],
                        }
                        append_jsonl(output_path, row)
                        all_rows.append(row)
                        completed.add(key)
        except Exception as exc:
            error = {
                "phase": "upstream_path",
                "sid": int(source_row["sid"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            print(
                f"\n[ERROR sid={source_row['sid']}] {type(exc).__name__}: {exc}",
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
                f"[upstream {sample_index}/{len(rows_to_run)}] rows={len(all_rows)}",
                flush=True,
            )

    summary = summarize_upstream_rows(all_rows)
    write_csv(output_dir / "upstream_path_summary.csv", summary)
    write_json(
        output_dir / "upstream_top_edges.json",
        {
            "script_version": SCRIPT_VERSION,
            "metric": "absolute_mean_normalized_effect",
            "edges": summary[: max(0, int(args.top_upstream_edges))],
        },
    )
    write_upstream_report(
        output_dir=output_dir,
        args=args,
        summary=summary,
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


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

    source_config, source_rows = load_source_rows(args)
    producer_module = import_file(
        Path(args.producer_script),
        "ioi_backward_producer",
    )
    receiver_module = import_file(
        Path(args.receiver_script),
        "ioi_backward_receiver",
    )
    v3 = import_file(Path(args.v3_script), "ioi_backward_v3")
    base = import_file(Path(args.base_script), "ioi_backward_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "ioi_backward_attention",
    )

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

        token_report = tokenization_report(
            processor.tokenizer,
            relation_token_map,
        )
        write_json(output_dir / "tokenization.json", token_report)
        if (
            args.require_single_token_labels
            and not token_report["all_relations_have_single_token_continuation"]
        ):
            raise RuntimeError(
                "At least one relation label lacks a one-token plain/leading-space "
                "continuation. Use a constrained one-token answer format or rerun with "
                "--no-require-single-token-labels only if first-subtoken scoring is "
                "intentional."
            )

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
            "score_mode": "next_token_relation_variants_at_prompt_last",
            "free_generation_required": False,
            "require_single_token_labels": args.require_single_token_labels,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        if args.phase == "token_check":
            print(json.dumps(token_report, ensure_ascii=False, indent=2))
            print(f"\nSaved outputs to {output_dir}", flush=True)
            return

        records_by_sid, prompt_rows, audit = prepare_data_helpers(args, base)
        config["audit"] = audit
        write_json(output_dir / "config.json", config)

        if args.phase in {"writer_scan", "writer_validate"}:
            run_writer_phase(
                args=args,
                output_dir=output_dir,
                source_rows=source_rows,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                relation_token_map=relation_token_map,
                base=base,
                receiver_module=receiver_module,
                attention_helper=attention_helper,
                v3=v3,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
            )
        elif args.phase == "upstream_path":
            run_upstream_phase(
                args=args,
                output_dir=output_dir,
                source_rows=source_rows,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                relation_token_map=relation_token_map,
                base=base,
                receiver_module=receiver_module,
                attention_helper=attention_helper,
                v3=v3,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
            )
        else:
            raise ValueError(args.phase)

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
