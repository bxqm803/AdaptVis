#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-pair tracing of object-position differences into the last token.

This script analyzes ONE COCO two-object sample at a time. It runs the original
image and its horizontal flip with identical text, then measures:

1. The original-minus-flip state difference at subject, reference, and
   prompt-last positions in every decoder layer.
2. The exact prompt-last block decomposition:

       Delta h_out
       = Delta h_in + Delta attention + Delta MLP

3. At selected layers, the exact attention contribution from each source group
   and head into prompt_last:

       C_{S->last}^{l,h}
       = sum_{s in S} A_{last,s}^{l,h} V_s^{l,h} W_O^{l,h}

   and its original-minus-flip difference:

       Delta C = C_original - C_flipped

4. The exact routing/content decomposition:

       Delta C = routing + content

       routing = (A_o - A_f) mean(V_o, V_f) W_O
       content = mean(A_o, A_f) (V_o - V_f) W_O

5. Whether each edge difference aligns with:
   - the prompt-last attention difference in the same layer;
   - the prompt-last NEW block increment in the same layer;
   - the final decoder-layer prompt-last difference.

No centroid and no trained probe are used. The pair itself supplies the
position-sensitive contrast. The analysis is descriptive/local; edge
replacement is still required for a final causal claim.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


VERSION = "coco-flip-last-increment-trace-v1"

RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

STATE_GROUPS = (
    "subject",
    "reference",
    "prompt_last",
)

SOURCE_GROUPS = (
    "visual_all",
    "subject",
    "reference",
    "relation",
    "options",
    "query_words",
    "instruction_other",
    "question_other",
    "chat_prefix",
    "chat_suffix",
    "other_text",
    "self",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge-script",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
        help="The previously generated edge-vector script.",
    )
    parser.add_argument(
        "--helper-script",
        default="analyze_coco_flip_same_token_similarity_v1.py",
    )
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    parser.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sid",
        required=True,
        type=int,
        help="Analyze exactly this sample id.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    parser.add_argument(
        "--state-layers",
        default="all",
        help="Layers used for subject/reference/prompt-last state trajectories.",
    )
    parser.add_argument(
        "--edge-layers",
        default="19,20,21,22,23,24,25,26,27,28,29,30",
        help="Layers used for source/head -> prompt-last decomposition.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCE_GROUPS),
        help=f"Comma-separated source groups; allowed={SOURCE_GROUPS}",
    )
    parser.add_argument(
        "--replay-tolerance",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top paths printed in report.txt.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_subset(
    value: str,
    allowed: Sequence[str],
    label: str,
) -> List[str]:
    allowed_set = set(allowed)
    result: List[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported {label}: {item}; allowed={sorted(allowed_set)}"
            )
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"No {label} selected")
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def select_rows(
    tensor: torch.Tensor,
    all_positions: Sequence[int],
    requested_positions: Sequence[int],
) -> torch.Tensor:
    lookup = {
        int(position): index
        for index, position in enumerate(all_positions)
    }
    local = [
        lookup[int(position)]
        for position in requested_positions
        if int(position) in lookup
    ]
    if not local:
        raise RuntimeError(
            f"No requested positions found. requested={requested_positions}, "
            f"available={all_positions}"
        )
    index = torch.tensor(local, dtype=torch.long)
    return tensor.index_select(0, index)


def mean_rows(
    tensor: torch.Tensor,
    all_positions: Sequence[int],
    requested_positions: Sequence[int],
) -> torch.Tensor:
    return select_rows(
        tensor,
        all_positions,
        requested_positions,
    ).mean(dim=0)


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def projection_fraction(vector: torch.Tensor, target: torch.Tensor) -> float:
    vector = vector.detach().float().cpu()
    target = target.detach().float().cpu()
    denominator = float(target.pow(2).sum())
    if denominator <= 1e-12:
        return float("nan")
    return float(torch.dot(vector, target) / denominator)


def vector_stats(
    vector: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    return {
        f"{prefix}_cosine": safe_cosine(vector, target),
        f"{prefix}_projection_fraction": projection_fraction(vector, target),
    }


def norm(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().norm())


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def resolve_record(records: Sequence[Any], sid: int) -> Any:
    matches = [
        record
        for record in records
        if int(record.sid) == int(sid)
    ]
    if not matches:
        raise KeyError(f"Sample sid={sid} not found")
    if len(matches) != 1:
        raise RuntimeError(f"sid={sid} matched {len(matches)} records")
    return matches[0]


def nested_detach(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(nested_detach(item) for item in value)
    if isinstance(value, list):
        return [nested_detach(item) for item in value]
    if isinstance(value, dict):
        return {
            key: nested_detach(item)
            for key, item in value.items()
        }
    return value


@dataclass
class StateCapture:
    positions: List[int]
    block_inputs: Dict[int, torch.Tensor]
    attention_outputs: Dict[int, torch.Tensor]
    block_outputs: Dict[int, torch.Tensor]


class CaptureStateComponents:
    """Capture block input, attention residual output, and block output."""

    def __init__(
        self,
        edge_module: Any,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
        positions: Sequence[int],
    ) -> None:
        self.edge = edge_module
        self.decoder_layers = decoder_layers
        self.layer_indices = list(layer_indices)
        self.positions = sorted(set(map(int, positions)))
        self.handles: List[Any] = []
        self.block_inputs: Dict[int, torch.Tensor] = {}
        self.attention_outputs: Dict[int, torch.Tensor] = {}
        self.block_outputs: Dict[int, torch.Tensor] = {}
        self.events: Counter = Counter()

    def _select(self, hidden: torch.Tensor) -> torch.Tensor:
        index = torch.tensor(
            self.positions,
            device=hidden.device,
            dtype=torch.long,
        )
        return (
            hidden[0]
            .index_select(0, index)
            .detach()
            .float()
            .cpu()
        )

    def __enter__(self) -> "CaptureStateComponents":
        for layer_index in self.layer_indices:
            layer = self.decoder_layers[layer_index]
            attention = self.edge.resolve_self_attention(layer)

            def make_block_pre(index: int):
                def hook(
                    _module: Any,
                    args: Tuple[Any, ...],
                    kwargs: Dict[str, Any],
                ) -> None:
                    hidden = self.edge.locate_hidden_states(args, kwargs)
                    self.block_inputs[index] = self._select(hidden)
                    self.events[(index, "block_pre")] += 1
                return hook

            def make_attention_post(index: int):
                def hook(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    _kwargs: Dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = self.edge.first_tensor(output)
                    self.attention_outputs[index] = self._select(hidden)
                    self.events[(index, "attention_post")] += 1
                return hook

            def make_block_post(index: int):
                def hook(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    _kwargs: Dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = self.edge.first_tensor(output)
                    self.block_outputs[index] = self._select(hidden)
                    self.events[(index, "block_post")] += 1
                return hook

            self.handles.append(
                layer.register_forward_pre_hook(
                    make_block_pre(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    make_attention_post(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    make_block_post(layer_index),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def materialize(self) -> StateCapture:
        for layer_index in self.layer_indices:
            for event_name, storage in (
                ("block_pre", self.block_inputs),
                ("attention_post", self.attention_outputs),
                ("block_post", self.block_outputs),
            ):
                count = int(self.events[(layer_index, event_name)])
                if count != 1:
                    raise RuntimeError(
                        f"Layer {layer_index}: {event_name} count={count}, "
                        "expected 1"
                    )
                if layer_index not in storage:
                    raise RuntimeError(
                        f"Layer {layer_index}: missing {event_name} tensor"
                    )
        return StateCapture(
            positions=list(self.positions),
            block_inputs=dict(self.block_inputs),
            attention_outputs=dict(self.attention_outputs),
            block_outputs=dict(self.block_outputs),
        )


@dataclass
class RunTrace:
    scores: Dict[str, float]
    prediction: str
    state: StateCapture
    edges: Dict[int, Any]


def run_trace(
    *,
    edge: Any,
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    state_layers: Sequence[int],
    edge_layers: Sequence[int],
    state_positions: Sequence[int],
    last_positions: Sequence[int],
) -> RunTrace:
    with torch.inference_mode():
        with CaptureStateComponents(
            edge,
            decoder_layers,
            state_layers,
            state_positions,
        ) as state_capture, edge.CaptureAttentionCalls(
            decoder_layers,
            edge_layers,
            last_positions,
        ) as attention_capture:
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

    logits = edge.extract_logits(outputs)[0, -1, :]
    scores = edge.score_relations(logits, token_map)
    prediction = max(RELATIONS, key=lambda relation: scores[relation])

    state = state_capture.materialize()
    attention_calls = attention_capture.materialize()
    edge_traces: Dict[int, Any] = {}

    for layer_index in edge_layers:
        attention = edge.resolve_self_attention(
            decoder_layers[layer_index]
        )
        block_output_last = select_rows(
            state.block_outputs[layer_index],
            state.positions,
            last_positions,
        )
        edge_traces[layer_index] = edge.replay_attention_layer(
            attention,
            attention_calls[layer_index],
            block_output_last,
            last_positions,
        )

    del outputs, logits, attention_calls
    return RunTrace(
        scores=scores,
        prediction=prediction,
        state=state,
        edges=edge_traces,
    )


def broad_role(token_role: str) -> str:
    role = str(token_role)
    if role == "subject":
        return "subject"
    if role == "reference":
        return "reference"
    if role in {
        "relation_keyword",
        "connector_to",
        "relation_connector",
    }:
        return "relation"
    if role in {
        "option_left",
        "option_right",
        "option_above",
        "option_below",
    }:
        return "options"
    if role in {"where", "copula"}:
        return "query_words"
    if role in {
        "answer",
        "with",
        "one",
        "spatial",
        "answer_relation",
        "answer_instruction",
    }:
        return "instruction_other"
    if role == "question_other":
        return "question_other"
    if role == "chat_prefix":
        return "chat_prefix"
    if role == "chat_suffix":
        return "chat_suffix"
    return "other_text"


def build_source_groups(
    *,
    sequence_length: int,
    visual_indices: Sequence[int],
    token_manifest: Sequence[Mapping[str, Any]],
    target_positions: Sequence[int],
) -> Dict[str, List[int]]:
    target_set = set(map(int, target_positions))
    groups: Dict[str, List[int]] = {
        name: []
        for name in SOURCE_GROUPS
    }
    visual_set = {
        int(position)
        for position in visual_indices
        if 0 <= int(position) < sequence_length
    }

    for position in range(sequence_length):
        if position in target_set:
            groups["self"].append(position)
            continue
        if position in visual_set:
            groups["visual_all"].append(position)
            continue

        manifest_by_position = {
            int(item["position"]): item
            for item in token_manifest
        }
        item = manifest_by_position.get(position)
        if item is None:
            groups["other_text"].append(position)
            continue
        groups[broad_role(str(item["token_role"]))].append(position)

    for key in groups:
        groups[key] = sorted(set(groups[key]))

    covered = [
        position
        for positions in groups.values()
        for position in positions
    ]
    counts = Counter(covered)
    missing = [
        position
        for position in range(sequence_length)
        if counts[position] == 0
    ]
    duplicates = [
        position
        for position, count in counts.items()
        if count > 1
    ]
    if missing or duplicates:
        raise RuntimeError(
            f"Invalid source partition: missing={missing[:20]}, "
            f"duplicates={duplicates[:20]}"
        )
    return groups


def state_group_positions(
    semantic: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for name in STATE_GROUPS:
        positions = sorted(set(map(
            int,
            semantic.get(name, []),
        )))
        if not positions:
            raise RuntimeError(
                f"State group {name} has no positions"
            )
        groups[name] = positions
    return groups


def state_component_vectors(
    *,
    original: RunTrace,
    flipped: RunTrace,
    layer: int,
    positions: Sequence[int],
) -> Dict[str, torch.Tensor]:
    original_input = mean_rows(
        original.state.block_inputs[layer],
        original.state.positions,
        positions,
    )
    flipped_input = mean_rows(
        flipped.state.block_inputs[layer],
        flipped.state.positions,
        positions,
    )
    original_attention = mean_rows(
        original.state.attention_outputs[layer],
        original.state.positions,
        positions,
    )
    flipped_attention = mean_rows(
        flipped.state.attention_outputs[layer],
        flipped.state.positions,
        positions,
    )
    original_output = mean_rows(
        original.state.block_outputs[layer],
        original.state.positions,
        positions,
    )
    flipped_output = mean_rows(
        flipped.state.block_outputs[layer],
        flipped.state.positions,
        positions,
    )

    input_delta = original_input - flipped_input
    attention_delta = original_attention - flipped_attention
    output_delta = original_output - flipped_output
    increment_delta = output_delta - input_delta
    mlp_delta = increment_delta - attention_delta

    decomposition_error = (
        output_delta
        - input_delta
        - attention_delta
        - mlp_delta
    )

    return {
        "input_delta": input_delta,
        "attention_delta": attention_delta,
        "mlp_delta": mlp_delta,
        "increment_delta": increment_delta,
        "output_delta": output_delta,
        "decomposition_error": decomposition_error,
    }


def edge_group_vectors(
    *,
    edge: Any,
    original_trace: Any,
    flipped_trace: Any,
    target_positions: Sequence[int],
    source_positions: Sequence[int],
) -> Dict[str, torch.Tensor]:
    per_head = edge.compute_group_head_vectors(
        original=original_trace,
        flipped=flipped_trace,
        target_positions=target_positions,
        source_positions=source_positions,
        sign=1.0,
    )
    return {
        "delta_heads": per_head["delta"],
        "routing_heads": per_head["routing"],
        "content_heads": per_head["content"],
        "error_heads": per_head["decomposition_error"],
        "delta_total": per_head["delta"].sum(dim=0),
        "routing_total": per_head["routing"].sum(dim=0),
        "content_total": per_head["content"].sum(dim=0),
        "error_total": per_head["decomposition_error"].sum(dim=0),
    }


def make_state_rows(
    *,
    original: RunTrace,
    flipped: RunTrace,
    layers: Sequence[int],
    group_positions: Mapping[str, Sequence[int]],
    final_last_delta: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, str], Dict[str, torch.Tensor]]]:
    rows: List[Dict[str, Any]] = []
    vectors: Dict[
        Tuple[int, str],
        Dict[str, torch.Tensor],
    ] = {}

    for layer in layers:
        for group, positions in group_positions.items():
            components = state_component_vectors(
                original=original,
                flipped=flipped,
                layer=layer,
                positions=positions,
            )
            vectors[(layer, group)] = components

            row: Dict[str, Any] = {
                "layer": int(layer),
                "token_group": group,
                "input_delta_norm": norm(components["input_delta"]),
                "attention_delta_norm": norm(components["attention_delta"]),
                "mlp_delta_norm": norm(components["mlp_delta"]),
                "increment_delta_norm": norm(components["increment_delta"]),
                "output_delta_norm": norm(components["output_delta"]),
                "decomposition_relative_error": (
                    norm(components["decomposition_error"])
                    / max(norm(components["output_delta"]), 1e-12)
                ),
            }
            for name in (
                "input_delta",
                "attention_delta",
                "mlp_delta",
                "increment_delta",
                "output_delta",
            ):
                row.update(vector_stats(
                    components[name],
                    final_last_delta,
                    f"{name}_to_final_last",
                ))

            row["attention_fraction_of_increment"] = projection_fraction(
                components["attention_delta"],
                components["increment_delta"],
            )
            row["mlp_fraction_of_increment"] = projection_fraction(
                components["mlp_delta"],
                components["increment_delta"],
            )
            row["input_fraction_of_output"] = projection_fraction(
                components["input_delta"],
                components["output_delta"],
            )
            row["increment_fraction_of_output"] = projection_fraction(
                components["increment_delta"],
                components["output_delta"],
            )
            rows.append(row)

    return rows, vectors


def make_edge_rows(
    *,
    edge: Any,
    original: RunTrace,
    flipped: RunTrace,
    edge_layers: Sequence[int],
    source_groups: Sequence[str],
    source_positions: Mapping[str, Sequence[int]],
    last_positions: Sequence[int],
    last_state_vectors: Mapping[Tuple[int, str], Mapping[str, torch.Tensor]],
    final_last_delta: torch.Tensor,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[Tuple[int, str], Dict[str, torch.Tensor]],
    Dict[Tuple[int, int, str], Dict[str, torch.Tensor]],
]:
    group_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    reconstruction_rows: List[Dict[str, Any]] = []
    group_vectors: Dict[
        Tuple[int, str],
        Dict[str, torch.Tensor],
    ] = {}
    head_vectors: Dict[
        Tuple[int, int, str],
        Dict[str, torch.Tensor],
    ] = {}

    for layer in edge_layers:
        local = last_state_vectors[(layer, "prompt_last")]
        attention_delta = local["attention_delta"]
        increment_delta = local["increment_delta"]
        output_delta = local["output_delta"]

        all_selected_delta = torch.zeros_like(attention_delta)
        all_selected_routing = torch.zeros_like(attention_delta)
        all_selected_content = torch.zeros_like(attention_delta)

        for source_group in source_groups:
            positions = list(source_positions[source_group])
            if not positions:
                continue

            vectors = edge_group_vectors(
                edge=edge,
                original_trace=original.edges[layer],
                flipped_trace=flipped.edges[layer],
                target_positions=last_positions,
                source_positions=positions,
            )
            group_vectors[(layer, source_group)] = vectors
            all_selected_delta += vectors["delta_total"]
            all_selected_routing += vectors["routing_total"]
            all_selected_content += vectors["content_total"]

            routing_norm = norm(vectors["routing_total"])
            content_norm = norm(vectors["content_total"])
            denominator = routing_norm + content_norm

            group_row: Dict[str, Any] = {
                "layer": int(layer),
                "source_group": source_group,
                "n_source_positions": len(positions),
                "n_heads": int(vectors["delta_heads"].shape[0]),
                "delta_norm": norm(vectors["delta_total"]),
                "routing_norm": routing_norm,
                "content_norm": content_norm,
                "routing_share_by_norm": (
                    routing_norm / denominator
                    if denominator > 1e-12
                    else float("nan")
                ),
                "decomposition_relative_error": (
                    norm(vectors["error_total"])
                    / max(norm(vectors["delta_total"]), 1e-12)
                ),
            }
            for target_name, target_vector in (
                ("last_attention", attention_delta),
                ("last_increment", increment_delta),
                ("last_output", output_delta),
                ("final_last", final_last_delta),
            ):
                group_row.update(vector_stats(
                    vectors["delta_total"],
                    target_vector,
                    f"delta_to_{target_name}",
                ))
                group_row.update(vector_stats(
                    vectors["routing_total"],
                    target_vector,
                    f"routing_to_{target_name}",
                ))
                group_row.update(vector_stats(
                    vectors["content_total"],
                    target_vector,
                    f"content_to_{target_name}",
                ))
            group_rows.append(group_row)

            n_heads = int(vectors["delta_heads"].shape[0])
            for head in range(n_heads):
                delta = vectors["delta_heads"][head]
                routing = vectors["routing_heads"][head]
                content = vectors["content_heads"][head]
                error = vectors["error_heads"][head]
                head_vectors[(layer, head, source_group)] = {
                    "delta": delta,
                    "routing": routing,
                    "content": content,
                }

                routing_head_norm = norm(routing)
                content_head_norm = norm(content)
                head_denominator = (
                    routing_head_norm + content_head_norm
                )
                head_row: Dict[str, Any] = {
                    "layer": int(layer),
                    "head": int(head),
                    "source_group": source_group,
                    "n_source_positions": len(positions),
                    "delta_norm": norm(delta),
                    "routing_norm": routing_head_norm,
                    "content_norm": content_head_norm,
                    "routing_share_by_norm": (
                        routing_head_norm / head_denominator
                        if head_denominator > 1e-12
                        else float("nan")
                    ),
                    "decomposition_relative_error": (
                        norm(error)
                        / max(norm(delta), 1e-12)
                    ),
                }
                for target_name, target_vector in (
                    ("last_attention", attention_delta),
                    ("last_increment", increment_delta),
                    ("last_output", output_delta),
                    ("final_last", final_last_delta),
                ):
                    head_row.update(vector_stats(
                        delta,
                        target_vector,
                        f"delta_to_{target_name}",
                    ))
                    head_row.update(vector_stats(
                        content,
                        target_vector,
                        f"content_to_{target_name}",
                    ))
                head_rows.append(head_row)

        attention_reconstruction_error = (
            all_selected_delta - attention_delta
        )
        routing_content_error = (
            all_selected_delta
            - all_selected_routing
            - all_selected_content
        )
        block_decomposition_error = (
            local["output_delta"]
            - local["input_delta"]
            - local["attention_delta"]
            - local["mlp_delta"]
        )

        reconstruction_rows.append({
            "layer": int(layer),
            "selected_sources": ",".join(source_groups),
            "last_attention_delta_norm": norm(attention_delta),
            "selected_edge_sum_norm": norm(all_selected_delta),
            "attention_reconstruction_relative_error": (
                norm(attention_reconstruction_error)
                / max(norm(attention_delta), 1e-12)
            ),
            "routing_content_relative_error": (
                norm(routing_content_error)
                / max(norm(all_selected_delta), 1e-12)
            ),
            "block_decomposition_relative_error": (
                norm(block_decomposition_error)
                / max(norm(local["output_delta"]), 1e-12)
            ),
            "replay_max_abs_error": max(
                original.edges[layer].replay_max_abs_error,
                flipped.edges[layer].replay_max_abs_error,
            ),
            "replay_relative_error": max(
                original.edges[layer].replay_relative_error,
                flipped.edges[layer].replay_relative_error,
            ),
        })

    return (
        group_rows,
        head_rows,
        reconstruction_rows,
        group_vectors,
        head_vectors,
    )


def make_object_comparison_rows(
    *,
    edge_layers: Sequence[int],
    state_vectors: Mapping[Tuple[int, str], Mapping[str, torch.Tensor]],
    edge_group_vectors_map: Mapping[Tuple[int, str], Mapping[str, torch.Tensor]],
    final_last_delta: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer in edge_layers:
        last_increment = state_vectors[
            (layer, "prompt_last")
        ]["increment_delta"]
        last_attention = state_vectors[
            (layer, "prompt_last")
        ]["attention_delta"]

        for object_group in ("subject", "reference"):
            object_state = state_vectors[
                (layer, object_group)
            ]["input_delta"]
            edge_vectors = edge_group_vectors_map.get(
                (layer, object_group)
            )
            if edge_vectors is None:
                continue

            row: Dict[str, Any] = {
                "layer": int(layer),
                "object_group": object_group,
                "object_input_delta_norm": norm(object_state),
                "raw_object_to_last_increment_cosine": safe_cosine(
                    object_state,
                    last_increment,
                ),
                "raw_object_to_final_last_cosine": safe_cosine(
                    object_state,
                    final_last_delta,
                ),
                "edge_delta_norm": norm(edge_vectors["delta_total"]),
                "edge_content_norm": norm(edge_vectors["content_total"]),
                "edge_routing_norm": norm(edge_vectors["routing_total"]),
            }
            row.update(vector_stats(
                edge_vectors["delta_total"],
                last_attention,
                "edge_delta_to_last_attention",
            ))
            row.update(vector_stats(
                edge_vectors["delta_total"],
                last_increment,
                "edge_delta_to_last_increment",
            ))
            row.update(vector_stats(
                edge_vectors["delta_total"],
                final_last_delta,
                "edge_delta_to_final_last",
            ))
            row.update(vector_stats(
                edge_vectors["content_total"],
                last_increment,
                "edge_content_to_last_increment",
            ))
            row.update(vector_stats(
                edge_vectors["content_total"],
                final_last_delta,
                "edge_content_to_final_last",
            ))
            rows.append(row)
    return rows


def top_rows(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    top_k: int,
) -> List[Mapping[str, Any]]:
    eligible = [
        row
        for row in rows
        if finite(row.get(metric))
    ]
    return sorted(
        eligible,
        key=lambda row: float(row[metric]),
        reverse=True,
    )[:max(0, int(top_k))]


def format_float(value: Any, width: int = 12) -> str:
    if not finite(value):
        return f"{'nan':>{width}}"
    return f"{float(value):>{width}.6f}"


def build_report(
    *,
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    reconstruction_rows: Sequence[Mapping[str, Any]],
) -> str:
    last_rows = [
        row
        for row in state_rows
        if row["token_group"] == "prompt_last"
    ]
    last_rows = sorted(last_rows, key=lambda row: int(row["layer"]))

    group_top_increment = top_rows(
        group_rows,
        "delta_to_last_increment_projection_fraction",
        args.top_k,
    )
    group_top_final = top_rows(
        group_rows,
        "delta_to_final_last_projection_fraction",
        args.top_k,
    )
    head_top_increment = top_rows(
        head_rows,
        "delta_to_last_increment_projection_fraction",
        args.top_k,
    )

    lines = [
        "=" * 150,
        "SINGLE-PAIR OBJECT-TO-LAST POSITION-SENSITIVE TRANSFER TRACE",
        (
            f"model={args.model} | sid={args.sid} | "
            f"gt={metadata['original_relation']} -> "
            f"{metadata['flipped_relation']} after horizontal flip"
        ),
        (
            f"prediction: original={metadata['original_prediction']} | "
            f"flipped={metadata['flipped_prediction']}"
        ),
        "=" * 150,
        "",
        "Prompt-last state trajectory",
        (
            f"{'Layer':>7}{'InputNorm':>13}{'AttnNorm':>13}"
            f"{'MLPNorm':>13}{'IncrNorm':>13}{'OutputNorm':>13}"
            f"{'Out->Final':>13}{'Attn/Incr':>13}{'MLP/Incr':>13}"
        ),
        "-" * 115,
    ]

    for row in last_rows:
        lines.append(
            f"{int(row['layer']):>7}"
            f"{format_float(row['input_delta_norm'], 13)}"
            f"{format_float(row['attention_delta_norm'], 13)}"
            f"{format_float(row['mlp_delta_norm'], 13)}"
            f"{format_float(row['increment_delta_norm'], 13)}"
            f"{format_float(row['output_delta_norm'], 13)}"
            f"{format_float(row['output_delta_to_final_last_projection_fraction'], 13)}"
            f"{format_float(row['attention_fraction_of_increment'], 13)}"
            f"{format_float(row['mlp_fraction_of_increment'], 13)}"
        )

    lines += [
        "",
        "Top source-group paths aligned with the same-layer NEW prompt-last increment",
        (
            f"{'Rank':>5}{'Layer':>7}{'Source':>22}"
            f"{'DeltaNorm':>13}{'Route%':>10}"
            f"{'ProjIncr':>13}{'ProjFinal':>13}"
            f"{'ContentIncr':>14}"
        ),
        "-" * 105,
    ]
    for rank, row in enumerate(group_top_increment, start=1):
        lines.append(
            f"{rank:>5}"
            f"{int(row['layer']):>7}"
            f"{str(row['source_group']):>22}"
            f"{format_float(row['delta_norm'], 13)}"
            f"{format_float(100.0 * float(row['routing_share_by_norm']), 10)}"
            f"{format_float(row['delta_to_last_increment_projection_fraction'], 13)}"
            f"{format_float(row['delta_to_final_last_projection_fraction'], 13)}"
            f"{format_float(row['content_to_last_increment_projection_fraction'], 14)}"
        )

    lines += [
        "",
        "Top source-group paths aligned with the final prompt-last difference",
        (
            f"{'Rank':>5}{'Layer':>7}{'Source':>22}"
            f"{'DeltaNorm':>13}{'ProjIncr':>13}{'ProjFinal':>13}"
            f"{'CosFinal':>12}"
        ),
        "-" * 90,
    ]
    for rank, row in enumerate(group_top_final, start=1):
        lines.append(
            f"{rank:>5}"
            f"{int(row['layer']):>7}"
            f"{str(row['source_group']):>22}"
            f"{format_float(row['delta_norm'], 13)}"
            f"{format_float(row['delta_to_last_increment_projection_fraction'], 13)}"
            f"{format_float(row['delta_to_final_last_projection_fraction'], 13)}"
            f"{format_float(row['delta_to_final_last_cosine'], 12)}"
        )

    lines += [
        "",
        "Top individual heads aligned with the same-layer NEW prompt-last increment",
        (
            f"{'Rank':>5}{'Layer':>7}{'Head':>7}{'Source':>22}"
            f"{'DeltaNorm':>13}{'Route%':>10}"
            f"{'ProjIncr':>13}{'ProjFinal':>13}"
        ),
        "-" * 95,
    ]
    for rank, row in enumerate(head_top_increment, start=1):
        lines.append(
            f"{rank:>5}"
            f"{int(row['layer']):>7}"
            f"{int(row['head']):>7}"
            f"{str(row['source_group']):>22}"
            f"{format_float(row['delta_norm'], 13)}"
            f"{format_float(100.0 * float(row['routing_share_by_norm']), 10)}"
            f"{format_float(row['delta_to_last_increment_projection_fraction'], 13)}"
            f"{format_float(row['delta_to_final_last_projection_fraction'], 13)}"
        )

    lines += [
        "",
        "Reliability checks",
        (
            f"{'Layer':>7}{'AttnReconErr':>16}{'Route+Content':>16}"
            f"{'BlockDecomp':>16}{'ReplayMax':>14}{'ReplayRel':>14}"
        ),
        "-" * 90,
    ]
    for row in reconstruction_rows:
        lines.append(
            f"{int(row['layer']):>7}"
            f"{format_float(row['attention_reconstruction_relative_error'], 16)}"
            f"{format_float(row['routing_content_relative_error'], 16)}"
            f"{format_float(row['block_decomposition_relative_error'], 16)}"
            f"{format_float(row['replay_max_abs_error'], 14)}"
            f"{format_float(row['replay_relative_error'], 14)}"
        )

    lines += [
        "",
        "How to read the output:",
        "- input_delta: original-minus-flip state already present at block input.",
        "- attention_delta: original-minus-flip update newly written by attention.",
        "- mlp_delta: original-minus-flip update produced locally by the MLP.",
        "- increment_delta = output_delta - input_delta = attention_delta + mlp_delta.",
        "- source edge delta is compared with increment_delta in the SAME target residual space.",
        "- content measures transferred source-value change; routing measures changed attention allocation.",
        "- alignment with final_last is descriptive persistence/alignment, not proof that the edge survived causally.",
        "- A causal follow-up should replace the selected original edge contribution with its flipped counterpart.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    edge = import_file(
        Path(args.edge_script),
        "_attention_edge_v1",
    )
    helper = import_file(
        Path(args.helper_script),
        "_same_token_helper",
    )
    base = import_file(
        Path(args.base_script),
        "_centroid_base_single_pair",
    )

    sources = parse_subset(
        args.sources,
        SOURCE_GROUPS,
        "source group",
    )

    data_module = base.import_two_object_module()
    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record = resolve_record(records, args.sid)
    prompt_rows = base.load_standard_prompts(
        Path(args.prompt_jsonl)
    )
    prompt_row = prompt_rows[args.sid]

    subject = str(prompt_row["subject"])
    reference = str(prompt_row["reference"])
    question = str(prompt_row["question_text"])
    original_relation = base.normalize_relation(
        prompt_row["answer_raw"]
    )
    if original_relation not in {"left", "right"}:
        raise ValueError(
            f"sid={args.sid} relation={original_relation}; "
            "this script currently requires left/right."
        )
    flipped_relation = OPPOSITE[original_relation]

    specs = base.merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model}; available={sorted(specs)}"
        )
    spec = specs[args.model]

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / "errors.jsonl"

    model_class = getattr(
        transformers,
        spec.model_class,
        None,
    )
    if model_class is None:
        raise RuntimeError(
            f"transformers lacks {spec.model_class}"
        )

    model_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        model_kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
    model = model_class.from_pretrained(
        spec.repo_id,
        **model_kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    requested_state_layers = edge.parse_layers(
        args.state_layers,
        n_layers,
    )
    edge_layers = edge.parse_layers(
        args.edge_layers,
        n_layers,
    )
    final_layer = n_layers - 1
    state_layers = sorted(set(
        requested_state_layers
        + edge_layers
        + [final_layer]
    ))

    token_map = base.relation_token_variants(
        processor.tokenizer
    )

    original_image = None
    flipped_image = None
    original_batch = None
    flipped_batch = None
    original_trace = None
    flipped_trace = None

    try:
        original_image = base.record_image(record).convert("RGB")
        flipped_image = original_image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )
        rendered = base.build_prompt(processor, question)

        original_batch = base.move_batch(
            processor(
                text=[rendered],
                images=[original_image],
                return_tensors="pt",
            ),
            device,
        )
        flipped_batch = base.move_batch(
            processor(
                text=[rendered],
                images=[flipped_image],
                return_tensors="pt",
            ),
            device,
        )

        original_ids = (
            original_batch["input_ids"][0]
            .detach()
            .cpu()
            .tolist()
        )
        flipped_ids = (
            flipped_batch["input_ids"][0]
            .detach()
            .cpu()
            .tolist()
        )
        if original_ids != flipped_ids:
            raise RuntimeError(
                "Original and flipped tokenizations differ"
            )

        subject_span, reference_span = base.locate_object_spans(
            processor.tokenizer,
            original_ids,
            subject,
            reference,
        )
        visual_indices = base.resolve_visual_indices(
            model,
            processor,
            original_batch,
            original_ids,
        )
        visual_set = set(map(int, visual_indices))
        text_positions = [
            position
            for position in range(len(original_ids))
            if position not in visual_set
        ]
        semantic = helper.locate_semantic_spans(
            processor.tokenizer,
            original_ids,
            question,
            subject_span,
            reference_span,
            text_positions,
        )
        token_manifest = helper.build_token_manifest(
            processor.tokenizer,
            original_ids,
            text_positions,
            semantic,
        )

        group_positions = state_group_positions(semantic)
        state_positions = sorted(set(
            position
            for positions in group_positions.values()
            for position in positions
        ))
        last_positions = group_positions["prompt_last"]

        source_position_map = build_source_groups(
            sequence_length=len(original_ids),
            visual_indices=visual_indices,
            token_manifest=token_manifest,
            target_positions=last_positions,
        )

        original_trace = run_trace(
            edge=edge,
            model=model,
            batch=original_batch,
            token_map=token_map,
            decoder_layers=decoder_layers,
            state_layers=state_layers,
            edge_layers=edge_layers,
            state_positions=state_positions,
            last_positions=last_positions,
        )
        flipped_trace = run_trace(
            edge=edge,
            model=model,
            batch=flipped_batch,
            token_map=token_map,
            decoder_layers=decoder_layers,
            state_layers=state_layers,
            edge_layers=edge_layers,
            state_positions=state_positions,
            last_positions=last_positions,
        )

        final_last_original = mean_rows(
            original_trace.state.block_outputs[final_layer],
            original_trace.state.positions,
            last_positions,
        )
        final_last_flipped = mean_rows(
            flipped_trace.state.block_outputs[final_layer],
            flipped_trace.state.positions,
            last_positions,
        )
        final_last_delta = (
            final_last_original - final_last_flipped
        )

        state_rows, state_vectors = make_state_rows(
            original=original_trace,
            flipped=flipped_trace,
            layers=state_layers,
            group_positions=group_positions,
            final_last_delta=final_last_delta,
        )

        (
            group_rows,
            head_rows,
            reconstruction_rows,
            edge_group_vectors_map,
            edge_head_vectors_map,
        ) = make_edge_rows(
            edge=edge,
            original=original_trace,
            flipped=flipped_trace,
            edge_layers=edge_layers,
            source_groups=sources,
            source_positions=source_position_map,
            last_positions=last_positions,
            last_state_vectors=state_vectors,
            final_last_delta=final_last_delta,
        )

        object_comparison_rows = make_object_comparison_rows(
            edge_layers=edge_layers,
            state_vectors=state_vectors,
            edge_group_vectors_map=edge_group_vectors_map,
            final_last_delta=final_last_delta,
        )

        original_prediction = original_trace.prediction
        flipped_prediction = flipped_trace.prediction
        metadata = {
            "version": VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "sid": int(args.sid),
            "subject": subject,
            "reference": reference,
            "question": question,
            "original_relation": original_relation,
            "flipped_relation": flipped_relation,
            "original_prediction": original_prediction,
            "flipped_prediction": flipped_prediction,
            "original_correct": (
                original_prediction == original_relation
            ),
            "flipped_correct": (
                flipped_prediction == flipped_relation
            ),
            "original_scores": original_trace.scores,
            "flipped_scores": flipped_trace.scores,
            "decoder_path": decoder_path,
            "n_decoder_layers": n_layers,
            "state_layers": state_layers,
            "edge_layers": edge_layers,
            "final_layer": final_layer,
            "sources": sources,
            "subject_positions": group_positions["subject"],
            "reference_positions": group_positions["reference"],
            "prompt_last_positions": last_positions,
            "visual_token_count": len(visual_indices),
            "sequence_length": len(original_ids),
            "final_last_delta_norm": norm(final_last_delta),
            "audit": audit,
            "uses_centroid": False,
            "uses_trained_probe": False,
            "updates_model_weights": False,
        }

        (output_dir / "sample_metadata.json").write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        write_csv(
            output_dir / "token_state_trajectory.csv",
            state_rows,
        )
        write_csv(
            output_dir / "last_increment_by_source.csv",
            group_rows,
        )
        write_csv(
            output_dir / "last_increment_by_head.csv",
            head_rows,
        )
        write_csv(
            output_dir / "object_to_last_comparison.csv",
            object_comparison_rows,
        )
        write_csv(
            output_dir / "reconstruction_checks.csv",
            reconstruction_rows,
        )

        vector_payload: Dict[str, Any] = {
            "version": VERSION,
            "metadata": metadata,
            "final_last_delta": final_last_delta,
            "state_vectors": {
                f"L{layer}:{group}": {
                    key: value.detach().float().cpu()
                    for key, value in vectors.items()
                }
                for (layer, group), vectors in state_vectors.items()
            },
            "edge_group_vectors": {
                f"L{layer}:{source}->prompt_last": {
                    key: value.detach().float().cpu()
                    for key, value in vectors.items()
                    if torch.is_tensor(value)
                }
                for (layer, source), vectors
                in edge_group_vectors_map.items()
            },
            "edge_head_vectors": {
                f"L{layer}:H{head}:{source}->prompt_last": {
                    key: value.detach().float().cpu()
                    for key, value in vectors.items()
                }
                for (layer, head, source), vectors
                in edge_head_vectors_map.items()
            },
        }
        torch.save(
            vector_payload,
            output_dir / "sample_transfer_vectors.pt",
        )

        report = build_report(
            args=args,
            metadata=metadata,
            state_rows=state_rows,
            group_rows=group_rows,
            head_rows=head_rows,
            reconstruction_rows=reconstruction_rows,
        )
        (output_dir / "report.txt").write_text(
            report,
            encoding="utf-8",
        )

        summary = {
            "version": VERSION,
            "sid": int(args.sid),
            "n_state_rows": len(state_rows),
            "n_group_rows": len(group_rows),
            "n_head_rows": len(head_rows),
            "n_object_comparison_rows": len(
                object_comparison_rows
            ),
            "n_reconstruction_rows": len(
                reconstruction_rows
            ),
            "output_files": [
                "sample_metadata.json",
                "token_state_trajectory.csv",
                "last_increment_by_source.csv",
                "last_increment_by_head.csv",
                "object_to_last_comparison.csv",
                "reconstruction_checks.csv",
                "sample_transfer_vectors.pt",
                "report.txt",
                "summary.json",
                "errors.jsonl",
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\n" + report)
        print(f"Saved to: {output_dir}")

    except Exception as error:
        append_jsonl(
            error_path,
            {
                "sid": int(args.sid),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    finally:
        for image in (original_image, flipped_image):
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
        del (
            original_batch,
            flipped_batch,
            original_trace,
            flipped_trace,
            model,
            processor,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
