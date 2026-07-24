#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract flip-sensitive spatial vectors carried by attention edges.

For the same left/right question, run the original image and its horizontal
flip. At selected decoder layers, reconstruct each attention module with
output_attentions=True and decompose the target-token attention update into
source-token-group and head contributions.

For one head h, source group S, and target group T:

    C_{S->T}^{l,h}
        = mean_{t in T} sum_{s in S} A_{t,s}^{l,h} V_s^{l,h} W_O^{l,h}

The sign-aligned left-minus-right edge vector is:

    Delta C = y * (C_original - C_flipped)

where y=+1 when the original relation is left and y=-1 when it is right.

The original-vs-flip edge difference is also exactly decomposed as:

    Delta C = routing + content

    routing = (A_original - A_flipped) * mean(V_original, V_flipped) * W_O
    content = mean(A_original, A_flipped) * (V_original - V_flipped) * W_O

This first version is intended for Qwen-style decoder attention modules exposing
q_proj/k_proj/v_proj/o_proj. It does not alter model weights and does not use a
centroid or trained probe. The extracted vectors are local computational
contributions; high-scoring paths should later be validated by edge replacement
or ablation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import math
import shutil
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


VERSION = "coco-flip-attention-spatial-vectors-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
PAIR_STATUSES = (
    "all",
    "both_correct",
    "original_only_correct",
    "flipped_only_correct",
    "both_wrong",
)
TARGET_GROUPS = (
    "subject",
    "reference",
    "both",
    "relation_connector",
    "relation_keyword",
    "connector_to",
    "option_left",
    "option_right",
    "option_all",
    "question_last",
    "prompt_last",
    "chat_suffix",
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--helper-script",
        default="analyze_coco_flip_same_token_similarity_v1.py",
        help="Existing helper script containing prompt/token-span utilities.",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
        help="Use eager for the first run; selected layers are replayed in eager mode.",
    )
    p.add_argument(
        "--relations",
        default="left,right",
        help="This version should normally remain left,right.",
    )
    p.add_argument(
        "--layers",
        default="19,20,21,22,23,24,25,26,27,28,29,30",
        help="'all', 'auto:N', or zero-based decoder-layer indices.",
    )
    p.add_argument(
        "--targets",
        default="prompt_last",
        help=f"Comma-separated target groups; allowed={TARGET_GROUPS}",
    )
    p.add_argument(
        "--sources",
        default=(
            "visual_all,subject,reference,relation,options,query_words,"
            "instruction_other,question_other,chat_prefix,chat_suffix,"
            "other_text,self"
        ),
        help=f"Comma-separated source groups; allowed={SOURCE_GROUPS}",
    )
    p.add_argument(
        "--sample-status",
        default="all",
        choices=PAIR_STATUSES,
        help="Which baseline pair status is included in scalar analysis.",
    )
    p.add_argument(
        "--vector-status",
        default="both_correct",
        choices=PAIR_STATUSES,
        help="Which baseline pair status contributes to canonical saved vectors.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument(
        "--save-top-k-vectors",
        type=int,
        default=100,
        help="Save canonical vectors for the top K layer/head/source/target paths.",
    )
    p.add_argument(
        "--rank-metric",
        default="mean_projection_fraction_block",
        choices=[
            "mean_projection_fraction_attention",
            "mean_projection_fraction_block",
            "mean_delta_norm",
            "mean_abs_projection_fraction_attention",
            "mean_abs_projection_fraction_block",
        ],
    )
    p.add_argument(
        "--replay-tolerance",
        type=float,
        default=5e-3,
        help="Warn when standalone eager replay differs from the in-model attention output.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_subset(
    value: str,
    allowed: Sequence[str],
    label: str,
) -> List[str]:
    allowed_set = set(allowed)
    out: List[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported {label}: {item}; allowed={sorted(allowed_set)}"
            )
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = value.strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text.startswith("auto:"):
        stride = int(text.split(":", 1)[1])
        if stride <= 0:
            raise ValueError("auto stride must be positive")
        layers = list(range(stride - 1, n_layers, stride))
        if not layers or layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)
        return sorted(set(layers))
    out: List[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        layer = int(raw)
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"Layer {layer} outside valid range 0..{n_layers - 1}"
            )
        if layer not in out:
            out.append(layer)
    if not out:
        raise ValueError("No layers selected")
    return out


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(
        f"Cannot locate attention output tensor in {type(output).__name__}"
    )


def find_attention_weights(output: Any) -> torch.Tensor:
    candidates: List[torch.Tensor] = []
    if isinstance(output, (tuple, list)):
        candidates.extend(
            item
            for item in output
            if torch.is_tensor(item)
        )
    elif isinstance(output, Mapping):
        candidates.extend(
            value
            for value in output.values()
            if torch.is_tensor(value)
        )
    else:
        for name in ("attn_weights", "attention_weights", "attentions"):
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                candidates.append(value)

    four_dimensional = [
        tensor
        for tensor in candidates
        if tensor.ndim == 4
    ]
    if not four_dimensional:
        raise RuntimeError(
            "Standalone attention replay did not return a 4D attention "
            "weight tensor. Use a Qwen-style module and eager attention."
        )
    return four_dimensional[0]


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "logits",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "logits",
            None,
        ),
    ]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("No language-model logits found")


def score_relations(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for relation in RELATIONS:
        ids = [
            int(token_id)
            for token_id in token_map[relation]
            if 0 <= int(token_id) < logits.numel()
        ]
        if not ids:
            raise RuntimeError(f"No token variants found for {relation}")
        index = torch.tensor(
            ids,
            device=logits.device,
            dtype=torch.long,
        )
        scores[relation] = float(
            logits.index_select(0, index).max().detach().cpu()
        )
    return scores


def pair_status(
    original_correct: bool,
    flipped_correct: bool,
) -> str:
    if original_correct and flipped_correct:
        return "both_correct"
    if original_correct:
        return "original_only_correct"
    if flipped_correct:
        return "flipped_only_correct"
    return "both_wrong"


def status_selected(selected: str, status: str) -> bool:
    return selected == "all" or selected == status


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


def locate_hidden_states(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    value = kwargs.get("hidden_states")
    if torch.is_tensor(value):
        return value
    for item in args:
        if torch.is_tensor(item) and item.ndim == 3:
            return item
    raise RuntimeError(
        "Cannot locate hidden_states in attention-module inputs."
    )


def resolve_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise RuntimeError(
        f"Cannot locate self-attention module inside {type(layer).__name__}"
    )


@dataclass
class AttentionCall:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    actual_target_output: torch.Tensor
    events: int = 1


class CaptureAttentionCalls:
    """Capture selected attention inputs and actual outputs at target positions."""

    def __init__(
        self,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
        target_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer_indices = list(layer_indices)
        self.target_positions = sorted(set(map(int, target_positions)))
        self.handles: List[Any] = []
        self.pre_inputs: Dict[int, Tuple[Tuple[Any, ...], Dict[str, Any]]] = {}
        self.actual_outputs: Dict[int, torch.Tensor] = {}
        self.event_counts: Counter = Counter()

    def __enter__(self) -> "CaptureAttentionCalls":
        for layer_index in self.layer_indices:
            attention = resolve_self_attention(
                self.decoder_layers[layer_index]
            )

            def make_pre_hook(index: int):
                def pre_hook(
                    _module: Any,
                    args: Tuple[Any, ...],
                    kwargs: Dict[str, Any],
                ) -> None:
                    self.pre_inputs[index] = (
                        tuple(nested_detach(args)),
                        dict(nested_detach(kwargs)),
                    )
                    self.event_counts[(index, "pre")] += 1
                return pre_hook

            def make_post_hook(index: int):
                def post_hook(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    _kwargs: Dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = first_tensor(output)
                    index_tensor = torch.tensor(
                        self.target_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    self.actual_outputs[index] = (
                        hidden[0]
                        .index_select(0, index_tensor)
                        .detach()
                        .float()
                        .cpu()
                    )
                    self.event_counts[(index, "post")] += 1
                return post_hook

            self.handles.append(
                attention.register_forward_pre_hook(
                    make_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    make_post_hook(layer_index),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def materialize(self) -> Dict[int, AttentionCall]:
        result: Dict[int, AttentionCall] = {}
        for layer_index in self.layer_indices:
            if layer_index not in self.pre_inputs:
                raise RuntimeError(
                    f"Layer {layer_index}: attention pre-hook did not fire."
                )
            if layer_index not in self.actual_outputs:
                raise RuntimeError(
                    f"Layer {layer_index}: attention post-hook did not fire."
                )
            pre_events = int(self.event_counts[(layer_index, "pre")])
            post_events = int(self.event_counts[(layer_index, "post")])
            if pre_events != 1 or post_events != 1:
                raise RuntimeError(
                    f"Layer {layer_index}: expected one attention call; "
                    f"pre={pre_events}, post={post_events}"
                )
            args, kwargs = self.pre_inputs[layer_index]
            result[layer_index] = AttentionCall(
                args=args,
                kwargs=kwargs,
                actual_target_output=self.actual_outputs[layer_index],
                events=1,
            )
        return result


class CaptureBlockTargets:
    def __init__(
        self,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
        target_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer_indices = list(layer_indices)
        self.target_positions = sorted(set(map(int, target_positions)))
        self.handles: List[Any] = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "CaptureBlockTargets":
        for layer_index in self.layer_indices:
            layer = self.decoder_layers[layer_index]

            def make_hook(index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_tensor(output)
                    index_tensor = torch.tensor(
                        self.target_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    self.outputs[index] = (
                        hidden[0]
                        .index_select(0, index_tensor)
                        .detach()
                        .float()
                        .cpu()
                    )
                return hook

            self.handles.append(
                layer.register_forward_hook(
                    make_hook(layer_index)
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


@dataclass
class LayerTrace:
    target_positions: List[int]
    attention_weights: torch.Tensor  # [H,T,S], CPU float32
    value_states: torch.Tensor       # [H,S,Dh], CPU float32
    attention_output: torch.Tensor   # [T,D], CPU float32
    block_output: torch.Tensor       # [T,D], CPU float32
    o_proj_weight: torch.Tensor      # [D,H,Dh], CPU float32
    replay_max_abs_error: float
    replay_relative_error: float


def accepts_keyword(module: Any, name: str) -> bool:
    signature = inspect.signature(module.forward)
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def repeat_key_value(
    value_states: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    # value_states: [B, Hkv, S, Dh]
    n_kv_heads = int(value_states.shape[1])
    if n_kv_heads == n_heads:
        return value_states
    if n_heads % n_kv_heads != 0:
        raise RuntimeError(
            f"Cannot repeat {n_kv_heads} KV heads to {n_heads} attention heads."
        )
    repeat = n_heads // n_kv_heads
    return value_states.repeat_interleave(repeat, dim=1)


def project_value_states(
    attention: Any,
    hidden_states: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    v_proj = getattr(attention, "v_proj", None)
    o_proj = getattr(attention, "o_proj", None)
    if v_proj is None or o_proj is None:
        raise RuntimeError(
            f"{type(attention).__name__} must expose v_proj and o_proj."
        )

    values = v_proj(hidden_states)
    if values.ndim != 3:
        raise RuntimeError(
            f"v_proj returned shape {tuple(values.shape)}, expected [B,S,D]."
        )

    n_kv_heads = getattr(
        attention,
        "num_key_value_heads",
        None,
    )
    if n_kv_heads is None:
        n_kv_heads = getattr(
            getattr(attention, "config", None),
            "num_key_value_heads",
            None,
        )
    if n_kv_heads is None:
        n_kv_heads = n_heads
    n_kv_heads = int(n_kv_heads)

    if values.shape[-1] % n_kv_heads != 0:
        raise RuntimeError(
            f"v_proj dimension {values.shape[-1]} is not divisible "
            f"by num_key_value_heads={n_kv_heads}."
        )
    head_dim = int(values.shape[-1] // n_kv_heads)
    values = (
        values.view(
            values.shape[0],
            values.shape[1],
            n_kv_heads,
            head_dim,
        )
        .transpose(1, 2)
        .contiguous()
    )
    values = repeat_key_value(values, n_heads)
    return values


def reshape_o_projection(
    attention: Any,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    o_proj = getattr(attention, "o_proj", None)
    if o_proj is None or not hasattr(o_proj, "weight"):
        raise RuntimeError(
            f"{type(attention).__name__} has no o_proj.weight."
        )
    weight = o_proj.weight.detach().float()
    if weight.ndim != 2:
        raise RuntimeError(
            f"o_proj.weight must be 2D, got {tuple(weight.shape)}"
        )
    if weight.shape[1] != n_heads * head_dim:
        raise RuntimeError(
            f"o_proj input dimension {weight.shape[1]} != "
            f"{n_heads} * {head_dim}"
        )
    # [Dout, H*Dh] -> [Dout,H,Dh]
    return weight.view(
        weight.shape[0],
        n_heads,
        head_dim,
    )


def replay_attention_layer(
    attention: Any,
    call: AttentionCall,
    block_output: torch.Tensor,
    target_positions: Sequence[int],
) -> LayerTrace:
    args = call.args
    kwargs = dict(call.kwargs)
    hidden_states = locate_hidden_states(args, kwargs)

    if accepts_keyword(attention, "output_attentions"):
        kwargs["output_attentions"] = True
    else:
        raise RuntimeError(
            f"{type(attention).__name__}.forward does not accept "
            "output_attentions."
        )
    if accepts_keyword(attention, "use_cache"):
        kwargs["use_cache"] = False

    config = getattr(attention, "config", None)
    old_impl = None
    if config is not None and hasattr(config, "_attn_implementation"):
        old_impl = config._attn_implementation
        config._attn_implementation = "eager"

    try:
        with torch.inference_mode():
            replay_output = attention(*args, **kwargs)
    finally:
        if (
            config is not None
            and old_impl is not None
            and hasattr(config, "_attn_implementation")
        ):
            config._attn_implementation = old_impl

    replay_hidden = first_tensor(replay_output)
    weights = find_attention_weights(replay_output)
    if weights.shape[0] != 1:
        raise RuntimeError(
            f"Expected batch size 1, got attention weights {tuple(weights.shape)}"
        )

    positions = sorted(set(map(int, target_positions)))
    position_tensor_gpu = torch.tensor(
        positions,
        device=weights.device,
        dtype=torch.long,
    )
    replay_target = (
        replay_hidden[0]
        .index_select(0, position_tensor_gpu)
        .detach()
        .float()
        .cpu()
    )
    actual_target = call.actual_target_output
    if replay_target.shape != actual_target.shape:
        raise RuntimeError(
            f"Replay/actual attention target shapes differ: "
            f"{tuple(replay_target.shape)} vs {tuple(actual_target.shape)}"
        )
    replay_diff = replay_target - actual_target
    replay_max_abs_error = float(replay_diff.abs().max())
    replay_relative_error = float(
        replay_diff.norm()
        / actual_target.norm().clamp_min(1e-12)
    )

    target_weights = (
        weights[0]
        .index_select(1, position_tensor_gpu)
        .detach()
        .float()
        .cpu()
    )
    n_heads = int(target_weights.shape[0])

    with torch.inference_mode():
        values = project_value_states(
            attention,
            hidden_states,
            n_heads=n_heads,
        )
    values_cpu = values[0].detach().float().cpu()
    head_dim = int(values_cpu.shape[-1])
    o_proj_weight = (
        reshape_o_projection(
            attention,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        .detach()
        .float()
        .cpu()
    )

    if target_weights.shape[-1] != values_cpu.shape[1]:
        raise RuntimeError(
            "Attention key length and value sequence length differ: "
            f"{target_weights.shape[-1]} vs {values_cpu.shape[1]}"
        )

    return LayerTrace(
        target_positions=positions,
        attention_weights=target_weights,
        value_states=values_cpu,
        attention_output=replay_target,
        block_output=block_output.detach().float().cpu(),
        o_proj_weight=o_proj_weight,
        replay_max_abs_error=replay_max_abs_error,
        replay_relative_error=replay_relative_error,
    )


def run_and_trace(
    *,
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    layer_indices: Sequence[int],
    target_positions: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[int, LayerTrace]]:
    with torch.inference_mode():
        with CaptureAttentionCalls(
            decoder_layers,
            layer_indices,
            target_positions,
        ) as attention_capture, CaptureBlockTargets(
            decoder_layers,
            layer_indices,
            target_positions,
        ) as block_capture:
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

    logits = extract_logits(outputs)[0, -1, :]
    scores = score_relations(logits, token_map)
    prediction = max(RELATIONS, key=lambda relation: scores[relation])
    attention_calls = attention_capture.materialize()

    traces: Dict[int, LayerTrace] = {}
    for layer_index in layer_indices:
        attention = resolve_self_attention(
            decoder_layers[layer_index]
        )
        traces[layer_index] = replay_attention_layer(
            attention,
            attention_calls[layer_index],
            block_capture.outputs[layer_index],
            target_positions,
        )

    del outputs, logits, attention_calls
    return {
        "scores": scores,
        "prediction": prediction,
    }, traces


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

    for position in sorted(visual_set):
        if position in target_set:
            groups["self"].append(position)
        else:
            groups["visual_all"].append(position)

    manifest_by_position = {
        int(item["position"]): item
        for item in token_manifest
    }
    for position in range(sequence_length):
        if position in visual_set:
            continue
        if position in target_set:
            groups["self"].append(position)
            continue
        item = manifest_by_position.get(position)
        if item is None:
            groups["other_text"].append(position)
            continue
        group = broad_role(str(item["token_role"]))
        groups[group].append(position)

    groups["self"] = sorted(set(groups["self"]))
    for name in groups:
        groups[name] = sorted(set(groups[name]))

    covered: List[int] = []
    for name, positions in groups.items():
        if name == "self":
            covered.extend(positions)
        else:
            covered.extend(positions)
    if sorted(covered) != list(range(sequence_length)):
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
        raise RuntimeError(
            f"Source partition invalid: missing={missing[:20]}, "
            f"duplicates={duplicates[:20]}"
        )
    return groups


def safe_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    left = left.float()
    right = right.float()
    denom = left.norm() * right.norm()
    if float(denom) <= eps:
        return float("nan")
    return float(torch.dot(left, right) / denom)


def project_heads(
    head_vectors: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    # head_vectors: [H,Dh]
    # o_proj_weight: [D,H,Dh]
    if head_vectors.ndim != 2 or o_proj_weight.ndim != 3:
        raise RuntimeError(
            f"Bad projection shapes: {tuple(head_vectors.shape)}, "
            f"{tuple(o_proj_weight.shape)}"
        )
    if head_vectors.shape[0] != o_proj_weight.shape[1]:
        raise RuntimeError("Head count mismatch in output projection.")
    if head_vectors.shape[1] != o_proj_weight.shape[2]:
        raise RuntimeError("Head dimension mismatch in output projection.")
    # [H,Dh] x [D,H,Dh] -> [H,D]
    return torch.einsum(
        "hd,ohd->ho",
        head_vectors,
        o_proj_weight,
    )


def compute_group_head_vectors(
    *,
    original: LayerTrace,
    flipped: LayerTrace,
    target_positions: Sequence[int],
    source_positions: Sequence[int],
    sign: float,
) -> Dict[str, torch.Tensor]:
    if original.target_positions != flipped.target_positions:
        raise RuntimeError("Original/flip target-position manifests differ.")
    if original.attention_weights.shape != flipped.attention_weights.shape:
        raise RuntimeError("Original/flip attention-weight shapes differ.")
    if original.value_states.shape != flipped.value_states.shape:
        raise RuntimeError("Original/flip value-state shapes differ.")

    target_lookup = {
        int(position): index
        for index, position in enumerate(original.target_positions)
    }
    target_local = [
        target_lookup[int(position)]
        for position in target_positions
        if int(position) in target_lookup
    ]
    if not target_local:
        raise RuntimeError("Target group has no traced positions.")
    if not source_positions:
        raise RuntimeError("Source group is empty.")

    target_index = torch.tensor(
        target_local,
        dtype=torch.long,
    )
    source_index = torch.tensor(
        sorted(set(map(int, source_positions))),
        dtype=torch.long,
    )

    ao = original.attention_weights.index_select(
        1,
        target_index,
    ).index_select(2, source_index)
    af = flipped.attention_weights.index_select(
        1,
        target_index,
    ).index_select(2, source_index)
    vo = original.value_states.index_select(
        1,
        source_index,
    )
    vf = flipped.value_states.index_select(
        1,
        source_index,
    )

    # Mean over target positions; sum over source positions.
    original_head = torch.einsum(
        "hts,hsd->hd",
        ao,
        vo,
    ) / float(len(target_local))
    flipped_head = torch.einsum(
        "hts,hsd->hd",
        af,
        vf,
    ) / float(len(target_local))

    routing_head = torch.einsum(
        "hts,hsd->hd",
        ao - af,
        0.5 * (vo + vf),
    ) / float(len(target_local))
    content_head = torch.einsum(
        "hts,hsd->hd",
        0.5 * (ao + af),
        vo - vf,
    ) / float(len(target_local))

    original_residual = project_heads(
        original_head,
        original.o_proj_weight,
    )
    flipped_residual = project_heads(
        flipped_head,
        flipped.o_proj_weight,
    )
    routing_residual = project_heads(
        routing_head,
        original.o_proj_weight,
    )
    content_residual = project_heads(
        content_head,
        original.o_proj_weight,
    )

    signed_delta = float(sign) * (
        original_residual - flipped_residual
    )
    signed_routing = float(sign) * routing_residual
    signed_content = float(sign) * content_residual

    decomposition_error = (
        signed_delta
        - signed_routing
        - signed_content
    )

    return {
        "delta": signed_delta,
        "routing": signed_routing,
        "content": signed_content,
        "decomposition_error": decomposition_error,
    }


def target_delta(
    *,
    original: LayerTrace,
    flipped: LayerTrace,
    target_positions: Sequence[int],
    sign: float,
    field: str,
) -> torch.Tensor:
    lookup = {
        int(position): index
        for index, position in enumerate(original.target_positions)
    }
    local = [
        lookup[int(position)]
        for position in target_positions
        if int(position) in lookup
    ]
    if not local:
        raise RuntimeError("Target group has no traced positions.")
    index = torch.tensor(local, dtype=torch.long)
    original_value = getattr(original, field).index_select(0, index).mean(dim=0)
    flipped_value = getattr(flipped, field).index_select(0, index).mean(dim=0)
    return float(sign) * (original_value - flipped_value)


class VectorAccumulator:
    def __init__(self) -> None:
        self.items: Dict[
            Tuple[int, int, str, str],
            Dict[str, Any],
        ] = {}

    def update(
        self,
        key: Tuple[int, int, str, str],
        *,
        delta: torch.Tensor,
        routing: torch.Tensor,
        content: torch.Tensor,
    ) -> None:
        delta = delta.detach().float().cpu()
        routing = routing.detach().float().cpu()
        content = content.detach().float().cpu()
        item = self.items.get(key)
        if item is None:
            item = {
                "count": 0,
                "nonzero_count": 0,
                "sum_delta": torch.zeros_like(delta),
                "sum_unit_delta": torch.zeros_like(delta),
                "sum_routing": torch.zeros_like(routing),
                "sum_content": torch.zeros_like(content),
            }
            self.items[key] = item
        item["count"] += 1
        item["sum_delta"] += delta
        item["sum_routing"] += routing
        item["sum_content"] += content
        norm = float(delta.norm())
        if norm > 1e-12:
            item["nonzero_count"] += 1
            item["sum_unit_delta"] += delta / norm

    def vectors_for_key(
        self,
        key: Tuple[int, int, str, str],
    ) -> Dict[str, torch.Tensor]:
        item = self.items[key]
        count = max(1, int(item["count"]))
        nonzero = max(1, int(item["nonzero_count"]))
        mean_delta = item["sum_delta"] / float(count)
        mean_routing = item["sum_routing"] / float(count)
        mean_content = item["sum_content"] / float(count)
        direction = item["sum_unit_delta"] / float(nonzero)
        direction = direction / direction.norm().clamp_min(1e-12)
        return {
            "mean_delta": mean_delta,
            "canonical_direction": direction,
            "mean_routing": mean_routing,
            "mean_content": mean_content,
        }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_status: bool,
) -> List[Dict[str, Any]]:
    grouped: Dict[
        Tuple[Any, ...],
        List[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        key_values: List[Any] = [
            int(row["layer"]),
            int(row["head"]),
            str(row["source_group"]),
            str(row["target_group"]),
        ]
        if include_status:
            key_values.append(str(row["pair_status"]))
        grouped[tuple(key_values)].append(row)

    metrics = (
        "delta_norm",
        "routing_norm",
        "content_norm",
        "routing_share_by_norm",
        "cosine_to_attention_delta",
        "projection_fraction_attention",
        "cosine_to_block_delta",
        "projection_fraction_block",
        "decomposition_relative_error",
    )

    result: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        out: Dict[str, Any] = {
            "layer": key[0],
            "head": key[1],
            "source_group": key[2],
            "target_group": key[3],
            "n_rows": len(items),
            "n_sids": len({
                int(item["sid"])
                for item in items
            }),
        }
        if include_status:
            out["pair_status"] = key[4]

        for metric in metrics:
            values = finite_values(
                item.get(metric)
                for item in items
            )
            if not values:
                out[f"mean_{metric}"] = None
                out[f"median_{metric}"] = None
                continue
            array = np.asarray(values, dtype=np.float64)
            out[f"mean_{metric}"] = float(array.mean())
            out[f"median_{metric}"] = float(np.median(array))

        for metric in (
            "projection_fraction_attention",
            "projection_fraction_block",
        ):
            values = finite_values(
                abs(float(item[metric]))
                for item in items
                if item.get(metric) is not None
            )
            out[f"mean_abs_{metric}"] = (
                float(np.mean(values))
                if values
                else None
            )
        result.append(out)

    return sorted(
        result,
        key=lambda row: (
            int(row["layer"]),
            int(row["head"]),
            str(row["source_group"]),
            str(row["target_group"]),
            str(row.get("pair_status", "")),
        ),
    )


def rank_value(
    row: Mapping[str, Any],
    metric: str,
) -> float:
    value = row.get(metric)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -float("inf")
    if not math.isfinite(number):
        return -float("inf")
    return number


def save_top_vectors(
    *,
    path: Path,
    accumulator: VectorAccumulator,
    summary_all: Sequence[Mapping[str, Any]],
    top_k: int,
    rank_metric: str,
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    candidates = [
        row
        for row in summary_all
        if (
            int(row["layer"]),
            int(row["head"]),
            str(row["source_group"]),
            str(row["target_group"]),
        ) in accumulator.items
    ]
    candidates = sorted(
        candidates,
        key=lambda row: rank_value(row, rank_metric),
        reverse=True,
    )
    selected = candidates[:max(0, int(top_k))]

    keys: List[Tuple[int, int, str, str]] = []
    vector_records: List[Dict[str, Any]] = []
    mean_delta: List[torch.Tensor] = []
    canonical_direction: List[torch.Tensor] = []
    mean_routing: List[torch.Tensor] = []
    mean_content: List[torch.Tensor] = []

    for rank, row in enumerate(selected, start=1):
        key = (
            int(row["layer"]),
            int(row["head"]),
            str(row["source_group"]),
            str(row["target_group"]),
        )
        vectors = accumulator.vectors_for_key(key)
        keys.append(key)
        mean_delta.append(vectors["mean_delta"])
        canonical_direction.append(vectors["canonical_direction"])
        mean_routing.append(vectors["mean_routing"])
        mean_content.append(vectors["mean_content"])
        vector_records.append({
            "rank": rank,
            "layer": key[0],
            "head": key[1],
            "source_group": key[2],
            "target_group": key[3],
            "rank_metric": rank_metric,
            "rank_value": rank_value(row, rank_metric),
            "n_rows": int(row["n_rows"]),
            "n_sids": int(row["n_sids"]),
        })

    payload: Dict[str, Any] = {
        "version": VERSION,
        "metadata": dict(metadata),
        "rank_metric": rank_metric,
        "records": vector_records,
    }
    if keys:
        payload.update({
            "keys": keys,
            "mean_delta": torch.stack(mean_delta, dim=0),
            "canonical_direction": torch.stack(canonical_direction, dim=0),
            "mean_routing": torch.stack(mean_routing, dim=0),
            "mean_content": torch.stack(mean_content, dim=0),
        })
    else:
        payload.update({
            "keys": [],
            "mean_delta": torch.empty(0),
            "canonical_direction": torch.empty(0),
            "mean_routing": torch.empty(0),
            "mean_content": torch.empty(0),
        })
    torch.save(payload, path)
    return vector_records


def report_text(
    *,
    model: str,
    seen: int,
    analyzed: int,
    counts: Mapping[str, int],
    summary_all: Sequence[Mapping[str, Any]],
    rank_metric: str,
) -> str:
    ranked = sorted(
        summary_all,
        key=lambda row: rank_value(row, rank_metric),
        reverse=True,
    )
    lines = [
        "=" * 150,
        "COCO FLIP-SENSITIVE ATTENTION EDGE SPATIAL VECTORS",
        f"model={model} | seen={seen} | analyzed={analyzed}",
        "baseline: " + ", ".join(
            f"{key}={value}"
            for key, value in sorted(counts.items())
        ),
        f"ranking={rank_metric}",
        "=" * 150,
        "",
        "Top attention paths:",
        (
            f"{'Rank':>5}{'Layer':>7}{'Head':>7}"
            f"{'Source':>22}{'Target':>20}{'Nsid':>7}"
            f"{'DeltaNorm':>13}{'Route%':>10}"
            f"{'ProjAttn':>12}{'ProjBlock':>12}"
        ),
        "-" * 120,
    ]
    for rank, row in enumerate(ranked[:50], start=1):
        lines.append(
            f"{rank:>5}"
            f"{int(row['layer']):>7}"
            f"{int(row['head']):>7}"
            f"{str(row['source_group']):>22}"
            f"{str(row['target_group']):>20}"
            f"{int(row['n_sids']):>7}"
            f"{float(row['mean_delta_norm']):>13.6f}"
            f"{100.0 * float(row['mean_routing_share_by_norm']):>9.2f}%"
            f"{float(row['mean_projection_fraction_attention']):>12.6f}"
            f"{float(row['mean_projection_fraction_block']):>12.6f}"
        )
    lines += [
        "",
        "Interpretation:",
        "- delta is the sign-aligned left-minus-right vector carried by one head and one source group.",
        "- routing is caused by changed attention weights; content is caused by changed value content.",
        "- projection_fraction_attention measures alignment with the target token's full attention-update difference.",
        "- projection_fraction_block measures alignment with the target token's decoder-block-output difference.",
        "- Source groups form a non-overlapping partition for each target; self contains the target position itself.",
        "- These are local additive contributions, not yet a complete causal circuit. Validate top paths by edge replacement.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    helper = import_file(
        Path(args.helper_script),
        "_flip_similarity_helper",
    )
    base = import_file(
        Path(args.base_script),
        "_centroid_base",
    )

    relations = parse_subset(
        args.relations,
        RELATIONS,
        "relation",
    )
    if set(relations) - {"left", "right"}:
        raise ValueError(
            "This v1 script only supports horizontal left/right analysis."
        )
    targets = parse_subset(
        args.targets,
        TARGET_GROUPS,
        "target group",
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
        args.max_samples,
    )
    prompt_rows = base.load_standard_prompts(
        Path(args.prompt_jsonl)
    )

    specs = base.merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model}; available={sorted(specs)}"
        )
    spec = specs[args.model]

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    scalar_path = out_dir / "edge_contributions.jsonl"
    baseline_path = out_dir / "baseline_pairs.jsonl"
    error_path = out_dir / "errors.jsonl"

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )
    if model_cls is None:
        raise RuntimeError(
            f"transformers lacks {spec.model_class}"
        )

    kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        **kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = (
        base.resolve_decoder_layers(model)
    )
    layers = parse_layers(
        args.layers,
        len(decoder_layers),
    )
    token_map = base.relation_token_variants(
        processor.tokenizer
    )

    for layer_index in layers:
        attention = resolve_self_attention(
            decoder_layers[layer_index]
        )
        for name in ("v_proj", "o_proj"):
            if not hasattr(attention, name):
                raise RuntimeError(
                    f"Layer {layer_index} attention "
                    f"{type(attention).__name__} lacks {name}."
                )

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "relations": relations,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "layers": layers,
        "targets": targets,
        "sources": sources,
        "sample_status": args.sample_status,
        "vector_status": args.vector_status,
        "max_samples": args.max_samples,
        "rank_metric": args.rank_metric,
        "save_top_k_vectors": args.save_top_k_vectors,
        "audit": audit,
        "capture": {
            "attention_input": True,
            "attention_output": True,
            "attention_weights_replayed_eager": True,
            "value_states": True,
            "block_output": True,
        },
        "updates_model_weights": False,
        "uses_centroid": False,
        "uses_trained_probe": False,
    }
    (out_dir / "config.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Decoder={decoder_path}, total_layers={len(decoder_layers)}, "
        f"layers={layers}"
    )
    print(f"Targets={targets}")
    print(f"Sources={sources}")
    print(
        f"Scalar status={args.sample_status}; "
        f"vector status={args.vector_status}"
    )

    seen = 0
    analyzed = 0
    counts: Counter = Counter()
    scalar_rows: List[Dict[str, Any]] = []
    accumulator = VectorAccumulator()
    replay_warnings: List[Dict[str, Any]] = []
    start_time = time.time()

    for record in tqdm(
        records,
        desc=f"edge-vectors:{args.model}",
    ):
        sid = int(record.sid)
        seen += 1
        original_image = None
        flipped_image = None
        original_batch = None
        flipped_batch = None
        original_traces = None
        flipped_traces = None

        try:
            prompt_row = prompt_rows[sid]
            subject = str(prompt_row["subject"])
            reference = str(prompt_row["reference"])
            question = str(prompt_row["question_text"])
            original_relation = base.normalize_relation(
                prompt_row["answer_raw"]
            )
            if original_relation not in relations:
                continue
            flipped_relation = OPPOSITE[original_relation]

            original_image = (
                base.record_image(record)
                .convert("RGB")
            )
            flipped_image = original_image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            rendered = base.build_prompt(
                processor,
                question,
            )

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
                    "Original/flip tokenization differs."
                )

            subject_span, reference_span = (
                base.locate_object_spans(
                    processor.tokenizer,
                    original_ids,
                    subject,
                    reference,
                )
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

            target_position_map: Dict[str, List[int]] = {}
            for target in targets:
                positions = sorted(set(map(
                    int,
                    semantic.get(target, []),
                )))
                if positions:
                    target_position_map[target] = positions
            if not target_position_map:
                raise RuntimeError(
                    "No requested target group has token positions."
                )
            trace_target_positions = sorted(set(
                position
                for positions in target_position_map.values()
                for position in positions
            ))

            original_result, original_traces = run_and_trace(
                model=model,
                batch=original_batch,
                token_map=token_map,
                decoder_layers=decoder_layers,
                layer_indices=layers,
                target_positions=trace_target_positions,
            )
            flipped_result, flipped_traces = run_and_trace(
                model=model,
                batch=flipped_batch,
                token_map=token_map,
                decoder_layers=decoder_layers,
                layer_indices=layers,
                target_positions=trace_target_positions,
            )

            original_prediction = original_result["prediction"]
            flipped_prediction = flipped_result["prediction"]
            original_correct = (
                original_prediction == original_relation
            )
            flipped_correct = (
                flipped_prediction == flipped_relation
            )
            status = pair_status(
                original_correct,
                flipped_correct,
            )

            counts["eligible_relation_seen"] += 1
            counts["original_correct"] += int(original_correct)
            counts["flip_correct"] += int(flipped_correct)
            counts["both_correct"] += int(
                original_correct and flipped_correct
            )
            counts["predictions_opposite"] += int(
                OPPOSITE.get(original_prediction)
                == flipped_prediction
            )
            counts[status] += 1

            baseline_row = {
                "sid": sid,
                "subject": subject,
                "reference": reference,
                "original_relation": original_relation,
                "flipped_relation": flipped_relation,
                "original_prediction": original_prediction,
                "flipped_prediction": flipped_prediction,
                "original_correct": bool(original_correct),
                "flipped_correct": bool(flipped_correct),
                "pair_status": status,
                "original_scores": original_result["scores"],
                "flipped_scores": flipped_result["scores"],
                "subject_span": list(subject_span),
                "reference_span": list(reference_span),
                "trace_target_positions": trace_target_positions,
            }
            append_jsonl(baseline_path, baseline_row)

            if not status_selected(args.sample_status, status):
                continue

            sign = 1.0 if original_relation == "left" else -1.0
            sequence_length = len(original_ids)

            for target_group, target_positions in target_position_map.items():
                source_position_map = build_source_groups(
                    sequence_length=sequence_length,
                    visual_indices=visual_indices,
                    token_manifest=token_manifest,
                    target_positions=target_positions,
                )

                for layer_index in layers:
                    original_trace = original_traces[layer_index]
                    flipped_trace = flipped_traces[layer_index]

                    replay_max = max(
                        original_trace.replay_max_abs_error,
                        flipped_trace.replay_max_abs_error,
                    )
                    replay_relative = max(
                        original_trace.replay_relative_error,
                        flipped_trace.replay_relative_error,
                    )
                    if replay_max > args.replay_tolerance:
                        warning = {
                            "sid": sid,
                            "layer": layer_index,
                            "replay_max_abs_error": replay_max,
                            "replay_relative_error": replay_relative,
                        }
                        replay_warnings.append(warning)

                    attention_delta = target_delta(
                        original=original_trace,
                        flipped=flipped_trace,
                        target_positions=target_positions,
                        sign=sign,
                        field="attention_output",
                    )
                    block_delta = target_delta(
                        original=original_trace,
                        flipped=flipped_trace,
                        target_positions=target_positions,
                        sign=sign,
                        field="block_output",
                    )
                    attention_denominator = float(
                        attention_delta.pow(2).sum()
                    )
                    block_denominator = float(
                        block_delta.pow(2).sum()
                    )

                    n_heads = int(
                        original_trace.attention_weights.shape[0]
                    )
                    for source_group in sources:
                        source_positions = source_position_map[
                            source_group
                        ]
                        if not source_positions:
                            continue

                        vectors = compute_group_head_vectors(
                            original=original_trace,
                            flipped=flipped_trace,
                            target_positions=target_positions,
                            source_positions=source_positions,
                            sign=sign,
                        )
                        delta_heads = vectors["delta"]
                        routing_heads = vectors["routing"]
                        content_heads = vectors["content"]
                        error_heads = vectors[
                            "decomposition_error"
                        ]

                        for head in range(n_heads):
                            delta = delta_heads[head]
                            routing = routing_heads[head]
                            content = content_heads[head]
                            decomposition_error = error_heads[head]

                            delta_norm = float(delta.norm())
                            routing_norm = float(routing.norm())
                            content_norm = float(content.norm())
                            norm_sum = routing_norm + content_norm
                            routing_share = (
                                routing_norm / norm_sum
                                if norm_sum > 1e-12
                                else float("nan")
                            )
                            decomposition_relative_error = float(
                                decomposition_error.norm()
                                / delta.norm().clamp_min(1e-12)
                            )

                            attention_dot = float(
                                torch.dot(delta, attention_delta)
                            )
                            block_dot = float(
                                torch.dot(delta, block_delta)
                            )
                            projection_attention = (
                                attention_dot / attention_denominator
                                if attention_denominator > 1e-12
                                else float("nan")
                            )
                            projection_block = (
                                block_dot / block_denominator
                                if block_denominator > 1e-12
                                else float("nan")
                            )

                            row = {
                                "sid": sid,
                                "subject": subject,
                                "reference": reference,
                                "original_relation": original_relation,
                                "original_prediction": original_prediction,
                                "flipped_prediction": flipped_prediction,
                                "original_correct": bool(original_correct),
                                "flipped_correct": bool(flipped_correct),
                                "pair_status": status,
                                "layer": int(layer_index),
                                "head": int(head),
                                "source_group": source_group,
                                "target_group": target_group,
                                "n_source_positions": len(source_positions),
                                "n_target_positions": len(target_positions),
                                "delta_norm": delta_norm,
                                "routing_norm": routing_norm,
                                "content_norm": content_norm,
                                "routing_share_by_norm": routing_share,
                                "cosine_to_attention_delta": safe_cosine(
                                    delta,
                                    attention_delta,
                                ),
                                "projection_fraction_attention": (
                                    projection_attention
                                ),
                                "cosine_to_block_delta": safe_cosine(
                                    delta,
                                    block_delta,
                                ),
                                "projection_fraction_block": (
                                    projection_block
                                ),
                                "decomposition_relative_error": (
                                    decomposition_relative_error
                                ),
                                "target_attention_delta_norm": float(
                                    attention_delta.norm()
                                ),
                                "target_block_delta_norm": float(
                                    block_delta.norm()
                                ),
                                "replay_max_abs_error": replay_max,
                                "replay_relative_error": replay_relative,
                            }
                            append_jsonl(scalar_path, row)
                            scalar_rows.append(row)

                            if status_selected(
                                args.vector_status,
                                status,
                            ):
                                accumulator.update(
                                    (
                                        int(layer_index),
                                        int(head),
                                        source_group,
                                        target_group,
                                    ),
                                    delta=delta,
                                    routing=routing,
                                    content=content,
                                )

            analyzed += 1
            if (
                args.print_every > 0
                and analyzed % args.print_every == 0
            ):
                print(
                    f"[{analyzed}] sid={sid} "
                    f"gt={original_relation} "
                    f"orig={original_prediction} "
                    f"flip={flipped_prediction} "
                    f"status={status}",
                    flush=True,
                )

        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"[ERROR] sid={sid}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

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
                original_traces,
                flipped_traces,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not scalar_rows:
        raise RuntimeError(
            "No edge-contribution rows were generated. "
            "Inspect errors.jsonl."
        )

    summary_by_status = summarize_rows(
        scalar_rows,
        include_status=True,
    )
    summary_all = summarize_rows(
        scalar_rows,
        include_status=False,
    )
    write_csv(
        out_dir / "edge_summary_by_pair_status.csv",
        summary_by_status,
    )
    write_csv(
        out_dir / "edge_summary_all.csv",
        summary_all,
    )

    try:
        import pandas as pd
        pd.DataFrame(scalar_rows).to_csv(
            out_dir / "edge_contributions.csv",
            index=False,
        )
    except Exception as exc:
        print(
            f"[WARN] Could not write full CSV: {exc}",
            flush=True,
        )

    vector_records = save_top_vectors(
        path=out_dir / "top_spatial_edge_vectors.pt",
        accumulator=accumulator,
        summary_all=summary_all,
        top_k=args.save_top_k_vectors,
        rank_metric=args.rank_metric,
        metadata=config,
    )
    write_csv(
        out_dir / "top_spatial_edge_vectors_manifest.csv",
        vector_records,
    )

    if replay_warnings:
        write_csv(
            out_dir / "replay_warnings.csv",
            replay_warnings,
        )
    else:
        (out_dir / "replay_warnings.csv").write_text(
            "",
            encoding="utf-8",
        )

    report = report_text(
        model=args.model,
        seen=seen,
        analyzed=analyzed,
        counts=counts,
        summary_all=summary_all,
        rank_metric=args.rank_metric,
    )
    (out_dir / "report.txt").write_text(
        report,
        encoding="utf-8",
    )
    print("\n" + report)

    summary = {
        "version": VERSION,
        "model": args.model,
        "seen": seen,
        "analyzed": analyzed,
        "counts": dict(counts),
        "n_scalar_rows": len(scalar_rows),
        "n_vector_keys": len(accumulator.items),
        "n_saved_vectors": len(vector_records),
        "n_replay_warnings": len(replay_warnings),
        "elapsed_minutes": (
            time.time() - start_time
        ) / 60.0,
        "output_files": [
            "config.json",
            "baseline_pairs.jsonl",
            "edge_contributions.jsonl",
            "edge_contributions.csv",
            "edge_summary_by_pair_status.csv",
            "edge_summary_all.csv",
            "top_spatial_edge_vectors.pt",
            "top_spatial_edge_vectors_manifest.csv",
            "replay_warnings.csv",
            "report.txt",
            "summary.json",
            "errors.jsonl",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
