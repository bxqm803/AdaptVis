#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two ways of increasing the last token's attention to the two object
text-token spans.

Experiment 1: pre_logit_bias
--------------------------------
Conceptually add a bias b to the subject/reference attention logits before
softmax:

    p'_j = softmax(s_j + b * I[j is object])

The eager attention hook exposes probabilities rather than raw logits. During
inference these are exactly equivalent:

    p'_j = normalize(p_j * exp(b * I[j is object]))

Thus this experiment is an exact pre-softmax logit-bias intervention, not an
approximation.

Experiment 2: post_mass_add
--------------------------------
After softmax, directly add a total attention mass delta to the two object spans:

    M_obj' = min(M_obj + delta, 1 - eps)

The relative distribution within object tokens is preserved, and the relative
distribution within all non-object keys is preserved. Non-object mass is
rescaled to keep every head normalized.

By default only the prefill-stage LAST prompt query is changed. Use
--query-scopes all to also intervene on each autoregressive decoding query.

Requirements
------------
Place this file beside:
    run_spatial_repair_three_experiments_v1.py
    trace_centroid_generation_groups_v2_1.py

The first file supplies the already-tested model/data/token-position utilities.

Outputs
-------
For every configuration:
- B repair rate
- A damage rate
- C damage rate
- weighted overall accuracy delta
- mean object attention mass before/after intervention
- per-sample generations in JSONL
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

try:
    import run_spatial_repair_three_experiments_v1 as base
except Exception as exc:
    raise SystemExit(
        "Unable to import run_spatial_repair_three_experiments_v1.py. "
        "Place that script in the same repository directory.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "object-attention-pre-post-boost-v1"

RELATIONS = base.RELATIONS
GROUP_A = base.GROUP_A
GROUP_B = base.GROUP_B
GROUP_C = base.GROUP_C
GROUP_D = base.GROUP_D

AUTO_WINDOWS = {
    "qwen-3b": "23-32",
    "qwen-7b": "20-23",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="qwen-3b,qwen-7b")
    parser.add_argument("--dataset", default="coco_two")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--input-root",
        default="output/three_group_transfer_fresh/coco",
    )
    parser.add_argument(
        "--output-root",
        default="output/object_attention_pre_post_boost/coco",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=["eager"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--groups", default="A,B,C")
    parser.add_argument("--max-per-group", type=int, default=None)
    parser.add_argument("--sid", type=int, default=None)

    parser.add_argument(
        "--experiments",
        default="pre_logit_bias,post_mass_add",
        help="Subset of pre_logit_bias,post_mass_add.",
    )
    parser.add_argument(
        "--pre-biases",
        default="0.25,0.5,0.75,1.0,1.5",
        help=(
            "Bias b added before softmax. Equivalent object multipliers are "
            "exp(b)."
        ),
    )
    parser.add_argument(
        "--post-masses",
        default="0.01,0.025,0.05,0.1,0.2",
        help=(
            "Total attention mass delta added to subject+reference after "
            "softmax, per selected head."
        ),
    )
    parser.add_argument(
        "--layers",
        default="auto",
        help="Layer spec such as 23-32, 23,25,28, or auto.",
    )
    parser.add_argument(
        "--query-scopes",
        default="prefill",
        help="Comma-separated subset of prefill,all.",
    )
    parser.add_argument(
        "--selected-heads",
        default=None,
        help=(
            "Optional layer:head list, e.g. '23:4,23:9,28:7'. "
            "Omit to intervene on every head in active layers."
        ),
    )

    parser.add_argument(
        "--verify-baseline",
        action="store_true",
        help="Rerun the unmodified baseline generation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument(
        "--core-module",
        default="trace_centroid_generation_groups_v2_1",
    )
    return parser.parse_args()


def parse_experiments(text: str) -> List[str]:
    allowed = {"pre_logit_bias", "post_mass_add"}
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    bad = [item for item in values if item not in allowed]
    if bad:
        raise ValueError(f"Unknown experiments: {bad}")
    if not values:
        raise ValueError("No experiments selected.")
    return values


def parse_query_scopes(text: str) -> List[str]:
    allowed = {"prefill", "all"}
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    bad = [item for item in values if item not in allowed]
    if bad:
        raise ValueError(f"Unknown query scopes: {bad}")
    if not values:
        raise ValueError("No query scope selected.")
    return values


def resolve_layers(model_name: str, requested: str, n_layers: int) -> List[int]:
    if str(requested).strip().lower() != "auto":
        return base.parse_layer_spec(requested, n_layers)
    if model_name in AUTO_WINDOWS:
        return base.parse_layer_spec(AUTO_WINDOWS[model_name], n_layers)
    start = max(0, int(math.floor(0.60 * n_layers)))
    stop = min(n_layers, max(start + 1, int(math.ceil(0.90 * n_layers))))
    return list(range(start, stop))


@dataclass(frozen=True)
class AttentionBoostConfig:
    config_id: str
    experiment: str
    layers: Tuple[int, ...]
    query_scope: str
    pre_bias: float = 0.0
    post_mass: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "experiment": self.experiment,
            "layers": ",".join(str(x) for x in self.layers),
            "query_scope": self.query_scope,
            "pre_bias": self.pre_bias,
            "pre_multiplier": math.exp(self.pre_bias),
            "post_mass": self.post_mass,
        }


def build_configs(
    *,
    experiments: Sequence[str],
    layers: Sequence[int],
    query_scopes: Sequence[str],
    pre_biases: Sequence[float],
    post_masses: Sequence[float],
) -> List[AttentionBoostConfig]:
    configs: List[AttentionBoostConfig] = []
    layer_name = f"L{layers[0]}-{layers[-1]}"

    for scope in query_scopes:
        if "pre_logit_bias" in experiments:
            for bias in pre_biases:
                configs.append(
                    AttentionBoostConfig(
                        config_id=(
                            f"pre_logit_bias__{layer_name}"
                            f"__scope{scope}__bias{bias:g}"
                        ),
                        experiment="pre_logit_bias",
                        layers=tuple(layers),
                        query_scope=scope,
                        pre_bias=float(bias),
                    )
                )

        if "post_mass_add" in experiments:
            for mass in post_masses:
                if not 0.0 < float(mass) < 1.0:
                    raise ValueError(
                        f"post mass must lie in (0,1), got {mass}"
                    )
                configs.append(
                    AttentionBoostConfig(
                        config_id=(
                            f"post_mass_add__{layer_name}"
                            f"__scope{scope}__delta{mass:g}"
                        ),
                        experiment="post_mass_add",
                        layers=tuple(layers),
                        query_scope=scope,
                        post_mass=float(mass),
                    )
                )

    if not configs:
        raise ValueError("No configurations generated.")
    return configs


@dataclass
class AttentionLayerCache:
    hidden_states: Optional[torch.Tensor] = None
    all_values: Optional[torch.Tensor] = None
    positions: Optional[base.ExpandedPositionSpec] = None


class ObjectAttentionBoostManager:
    def __init__(
        self,
        *,
        layers: Sequence[torch.nn.Module],
        active_layers: Sequence[int],
        selected_heads: Optional[Mapping[int, set[int]]],
    ) -> None:
        self.layers = list(layers)
        self.active_layers = sorted(set(int(x) for x in active_layers))
        self.selected_heads = selected_heads

        self.config: Optional[AttentionBoostConfig] = None
        self.prompt_spec: Optional[base.PromptPositionSpec] = None
        self.caches = {
            layer: AttentionLayerCache()
            for layer in self.active_layers
        }
        self.handles: List[Any] = []
        self.mass_before: List[float] = []
        self.mass_after: List[float] = []
        self.mass_delta: List[float] = []
        self.intervention_calls = 0

        for layer_index in self.active_layers:
            attention = getattr(self.layers[layer_index], "self_attn", None)
            if attention is None:
                raise RuntimeError(
                    f"Decoder layer {layer_index} has no self_attn."
                )
            self.handles.append(
                attention.register_forward_pre_hook(
                    self._make_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._make_post_hook(layer_index),
                    with_kwargs=True,
                )
            )

    def close(self) -> None:
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()
        self.reset()

    def reset(self) -> None:
        self.config = None
        self.prompt_spec = None
        self.mass_before.clear()
        self.mass_after.clear()
        self.mass_delta.clear()
        self.intervention_calls = 0
        for cache in self.caches.values():
            cache.hidden_states = None
            cache.all_values = None
            cache.positions = None

    def configure(
        self,
        *,
        config: AttentionBoostConfig,
        prompt_spec: base.PromptPositionSpec,
    ) -> None:
        self.reset()
        self.config = config
        self.prompt_spec = prompt_spec

    def stats(self) -> Dict[str, Any]:
        def finite_mean(values: Sequence[float]) -> float:
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]
            return float(np.mean(array)) if len(array) else float("nan")

        return {
            "object_mass_before": finite_mean(self.mass_before),
            "object_mass_after": finite_mean(self.mass_after),
            "object_mass_delta": finite_mean(self.mass_delta),
            "intervention_calls": self.intervention_calls,
        }

    def _active(self, layer_index: int) -> bool:
        return (
            self.config is not None
            and layer_index in self.config.layers
        )

    def _make_pre_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ):
            if not self._active(layer_index):
                return None

            hidden_states = None
            if args and torch.is_tensor(args[0]):
                hidden_states = args[0]
            elif torch.is_tensor(kwargs.get("hidden_states")):
                hidden_states = kwargs["hidden_states"]

            if hidden_states is None:
                raise RuntimeError(
                    f"Unable to capture attention input at L{layer_index}."
                )
            self.caches[layer_index].hidden_states = hidden_states
            return None

        return hook

    def _selected_head_mask(
        self,
        *,
        layer_index: int,
        number_of_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(
            (1, number_of_heads, 1),
            dtype=torch.bool,
            device=device,
        )
        if self.selected_heads is None:
            return mask

        mask.zero_()
        for head in self.selected_heads.get(layer_index, set()):
            if not 0 <= int(head) < number_of_heads:
                raise RuntimeError(
                    f"L{layer_index}H{head} outside H={number_of_heads}."
                )
            mask[:, int(head), :] = True
        return mask

    @staticmethod
    def _pre_softmax_bias(
        *,
        weights: torch.Tensor,
        object_mask: torch.Tensor,
        selected_heads: torch.Tensor,
        bias: float,
    ) -> torch.Tensor:
        """
        Exact equivalence:
            softmax(logits + bias*object_mask)
          = normalize(softmax(logits) * exp(bias*object_mask)).
        """
        multiplier = torch.ones_like(weights)
        active = selected_heads & object_mask
        multiplier = torch.where(
            active,
            torch.full_like(multiplier, math.exp(float(bias))),
            multiplier,
        )
        modified = weights * multiplier
        return modified / modified.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

    @staticmethod
    def _post_softmax_mass_add(
        *,
        weights: torch.Tensor,
        object_mask: torch.Tensor,
        selected_heads: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        """
        Add delta to TOTAL object attention mass after softmax while preserving:
        - relative proportions among object keys;
        - relative proportions among non-object keys.
        """
        eps = 1e-8
        object_float = object_mask.to(weights.dtype)
        nonobject_float = 1.0 - object_float

        object_weights = weights * object_float
        nonobject_weights = weights * nonobject_float

        old_object_mass = object_weights.sum(dim=-1, keepdim=True)
        old_nonobject_mass = nonobject_weights.sum(dim=-1, keepdim=True)

        target_object_mass = torch.clamp(
            old_object_mass + float(delta),
            min=0.0,
            max=1.0 - 1e-6,
        )
        target_object_mass = torch.where(
            selected_heads,
            target_object_mass,
            old_object_mass,
        )
        target_nonobject_mass = 1.0 - target_object_mass

        object_count = object_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform_object = object_float / object_count

        normalized_object = torch.where(
            old_object_mass > eps,
            object_weights / old_object_mass.clamp_min(eps),
            uniform_object,
        )
        normalized_nonobject = torch.where(
            old_nonobject_mass > eps,
            nonobject_weights / old_nonobject_mass.clamp_min(eps),
            torch.zeros_like(nonobject_weights),
        )

        modified = (
            normalized_object * target_object_mass
            + normalized_nonobject * target_nonobject_mass
        )
        return modified / modified.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

    def _make_post_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            output: Any,
        ):
            if not self._active(layer_index):
                return output

            config = self.config
            prompt_spec = self.prompt_spec
            if config is None or prompt_spec is None:
                return output

            attention_output = base.output_first_tensor(output)
            query_length = int(attention_output.shape[-2])
            attention_weights = base.find_attention_weights(
                output,
                query_length,
            )
            if attention_weights is None:
                raise RuntimeError(
                    f"L{layer_index} did not expose eager attention weights."
                )

            cache = self.caches[layer_index]
            hidden_states = cache.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    f"Missing attention input at L{layer_index}."
                )

            v_proj, o_proj = base.resolve_attention_projections(module)
            current_values = v_proj(hidden_states)

            number_of_heads = int(attention_weights.shape[1])
            head_dim = int(o_proj.weight.shape[1]) // number_of_heads
            if head_dim <= 0 or current_values.shape[-1] % head_dim != 0:
                raise RuntimeError(
                    f"Unable to reshape V states at L{layer_index}: "
                    f"V={tuple(current_values.shape)}, H={number_of_heads}, "
                    f"o_proj_in={o_proj.weight.shape[1]}."
                )

            kv_heads = int(current_values.shape[-1]) // head_dim
            current_values = current_values.view(
                current_values.shape[0],
                current_values.shape[1],
                kv_heads,
                head_dim,
            ).transpose(1, 2)
            current_values = base.repeat_kv(
                current_values,
                number_of_heads,
            )

            is_prefill = int(hidden_states.shape[1]) > 1

            if is_prefill:
                cache.all_values = current_values.detach()
                cache.positions = base.expand_positions(
                    prompt_spec,
                    hidden_length=int(hidden_states.shape[1]),
                )
            else:
                if cache.all_values is None:
                    raise RuntimeError(
                        f"L{layer_index} decode step arrived before prefill."
                    )
                cache.all_values = torch.cat(
                    [
                        cache.all_values,
                        current_values.detach().to(
                            device=cache.all_values.device,
                            dtype=cache.all_values.dtype,
                        ),
                    ],
                    dim=-2,
                )

            if config.query_scope == "prefill" and not is_prefill:
                return output

            if cache.all_values is None or cache.positions is None:
                raise RuntimeError(
                    f"Missing cached positions/V states at L{layer_index}."
                )

            key_length = int(attention_weights.shape[-1])
            all_values = cache.all_values

            if int(all_values.shape[-2]) != key_length:
                raise RuntimeError(
                    f"KV length mismatch at L{layer_index}: "
                    f"cached_values={all_values.shape[-2]}, "
                    f"attention_keys={key_length}. "
                    "A sliding-window or architecture-specific KV mapping "
                    "requires a dedicated implementation."
                )

            original_weights = attention_weights[:, :, -1, :].to(
                dtype=torch.float32,
            )
            all_values = all_values.to(
                device=attention_output.device,
                dtype=attention_output.dtype,
            )

            object_positions = sorted(
                set(cache.positions.subject_positions)
                | set(cache.positions.reference_positions)
            )
            object_positions = [
                int(position)
                for position in object_positions
                if 0 <= int(position) < key_length
            ]
            if not object_positions:
                raise RuntimeError(
                    f"No valid object positions at L{layer_index}."
                )

            object_mask = torch.zeros(
                (1, 1, key_length),
                dtype=torch.bool,
                device=original_weights.device,
            )
            object_index = torch.as_tensor(
                object_positions,
                dtype=torch.long,
                device=original_weights.device,
            )
            object_mask.index_fill_(-1, object_index, True)

            selected_heads = self._selected_head_mask(
                layer_index=layer_index,
                number_of_heads=number_of_heads,
                device=original_weights.device,
            )

            before_mass_per_head = (
                original_weights
                * object_mask.to(original_weights.dtype)
            ).sum(dim=-1)

            if config.experiment == "pre_logit_bias":
                modified_weights = self._pre_softmax_bias(
                    weights=original_weights,
                    object_mask=object_mask,
                    selected_heads=selected_heads,
                    bias=config.pre_bias,
                )
            elif config.experiment == "post_mass_add":
                modified_weights = self._post_softmax_mass_add(
                    weights=original_weights,
                    object_mask=object_mask,
                    selected_heads=selected_heads,
                    delta=config.post_mass,
                )
            else:
                raise RuntimeError(
                    f"Unknown experiment: {config.experiment}"
                )

            after_mass_per_head = (
                modified_weights
                * object_mask.to(modified_weights.dtype)
            ).sum(dim=-1)

            selected_2d = selected_heads.squeeze(-1)
            selected_before = before_mass_per_head[selected_2d]
            selected_after = after_mass_per_head[selected_2d]
            if selected_before.numel():
                before_mean = float(selected_before.mean().detach().cpu())
                after_mean = float(selected_after.mean().detach().cpu())
                self.mass_before.append(before_mean)
                self.mass_after.append(after_mean)
                self.mass_delta.append(after_mean - before_mean)

            modified_heads = torch.einsum(
                "bhk,bhkd->bhd",
                modified_weights.to(dtype=all_values.dtype),
                all_values,
            )
            modified_output = F.linear(
                modified_heads.reshape(
                    modified_heads.shape[0],
                    1,
                    -1,
                ),
                o_proj.weight,
                getattr(o_proj, "bias", None),
            )

            result = attention_output.clone()
            result[:, -1:, :] = modified_output.to(
                device=result.device,
                dtype=result.dtype,
            )
            self.intervention_calls += 1
            return base.replace_first_output(output, result)

        return hook


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def completed_keys(path: Path) -> set[Tuple[int, str]]:
    if not path.exists():
        return set()
    keys: set[Tuple[int, str]] = set()
    for row in base.read_jsonl(path):
        try:
            keys.add((int(row["sid"]), str(row["config_id"])))
        except Exception:
            pass
    return keys


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    configs: Sequence[AttentionBoostConfig],
) -> List[Dict[str, Any]]:
    config_lookup = {config.config_id: config for config in configs}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)

    summaries: List[Dict[str, Any]] = []

    for config_id, candidates in grouped.items():
        config = config_lookup.get(config_id)
        if config is None:
            continue

        n = len(candidates)
        baseline_correct_count = sum(
            str(row.get("group")) in {GROUP_A, GROUP_C}
            for row in candidates
        )
        intervention_correct_count = sum(
            bool(row.get("intervention_correct"))
            for row in candidates
        )

        summary: Dict[str, Any] = {
            **config.as_dict(),
            "n": n,
            "parse_rate": float(
                np.mean([bool(row.get("parsed")) for row in candidates])
            ),
            "baseline_accuracy": (
                baseline_correct_count / n if n else float("nan")
            ),
            "intervention_accuracy": (
                intervention_correct_count / n if n else float("nan")
            ),
            "delta_correct": intervention_correct_count - baseline_correct_count,
            "delta_accuracy": (
                (intervention_correct_count - baseline_correct_count) / n
                if n else float("nan")
            ),
            "object_mass_before": float(
                np.nanmean([
                    float(row.get("object_mass_before", np.nan))
                    for row in candidates
                ])
            ),
            "object_mass_after": float(
                np.nanmean([
                    float(row.get("object_mass_after", np.nan))
                    for row in candidates
                ])
            ),
            "object_mass_delta": float(
                np.nanmean([
                    float(row.get("object_mass_delta", np.nan))
                    for row in candidates
                ])
            ),
        }

        for group, short in (
            (GROUP_A, "A"),
            (GROUP_B, "B"),
            (GROUP_C, "C"),
        ):
            subset = [
                row for row in candidates
                if str(row.get("group")) == group
            ]
            accuracy = (
                float(np.mean([
                    bool(row.get("intervention_correct"))
                    for row in subset
                ]))
                if subset else float("nan")
            )
            summary[f"{short}_n"] = len(subset)
            summary[f"{short}_accuracy"] = accuracy

        summary["B_repair_rate"] = summary["B_accuracy"]
        summary["A_damage_rate"] = (
            1.0 - summary["A_accuracy"]
            if np.isfinite(summary["A_accuracy"])
            else float("nan")
        )
        summary["C_damage_rate"] = (
            1.0 - summary["C_accuracy"]
            if np.isfinite(summary["C_accuracy"])
            else float("nan")
        )

        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            -float(row.get("delta_accuracy", -np.inf)),
            -float(row.get("B_repair_rate", -np.inf)),
        )
    )
    return summaries


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 164)
    print("OBJECT ATTENTION PRE/POST BOOST SUMMARY")
    print("=" * 164)
    header = (
        f"{'Config':67s} {'N':>5s} {'Parse':>7s} "
        f"{'B repair':>9s} {'A damage':>9s} {'C damage':>9s} "
        f"{'ΔCorrect':>8s} {'ΔAcc':>8s} "
        f"{'ObjMass before→after':>22s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mass_text = (
            f"{float(row['object_mass_before']):.4f}"
            f"→{float(row['object_mass_after']):.4f}"
        )
        print(
            f"{str(row['config_id'])[:67]:67s} "
            f"{int(row['n']):5d} "
            f"{float(row['parse_rate']):7.3f} "
            f"{float(row['B_repair_rate']):9.3f} "
            f"{float(row['A_damage_rate']):9.3f} "
            f"{float(row['C_damage_rate']):9.3f} "
            f"{int(row['delta_correct']):8d} "
            f"{float(row['delta_accuracy']):8.3f} "
            f"{mass_text:>22s}"
        )


def run_model(
    *,
    args: argparse.Namespace,
    core: Any,
    backend_module: Any,
    model_name: str,
    selected_groups: Sequence[str],
    experiments: Sequence[str],
    query_scopes: Sequence[str],
    pre_biases: Sequence[float],
    post_masses: Sequence[float],
) -> None:
    metadata_path = (
        Path(args.input_root)
        / model_name
        / "pass2_transfer_trace"
        / "sample_metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    prior_rows = base.cap_rows(
        base.read_jsonl(metadata_path),
        selected_groups=selected_groups,
        max_per_group=args.max_per_group,
        seed=args.seed,
        sid=args.sid,
    )
    if not prior_rows:
        raise RuntimeError(f"No selected rows for {model_name}.")

    records, audit = backend_module.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {int(record.sid): record for record in records}

    prompt_path = core.resolve_prompt_path(args)
    prompt_rows = core.load_standard_prompts(prompt_path)

    missing = [
        int(row["sid"])
        for row in prior_rows
        if (
            int(row["sid"]) not in record_by_sid
            or int(row["sid"]) not in prompt_rows
        )
    ]
    if missing:
        raise RuntimeError(
            f"Missing records/prompts for {len(missing)} samples; "
            f"first={missing[:10]}"
        )

    if model_name not in backend_module.SPECS:
        raise ValueError(
            f"Unknown model alias {model_name!r}; "
            f"available={sorted(backend_module.SPECS)}"
        )
    spec = backend_module.SPECS[model_name]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no "
            f"{spec.model_class}"
        )

    print("\n" + "=" * 164)
    print(f"LOADING {model_name}: {spec.repo_id}")
    print("=" * 164)

    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=core.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation=args.attn_impl,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.configure_processor(model, processor)

    device = torch.device(args.device)
    decoder_layers, layers_path = core.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    active_layers = resolve_layers(
        model_name,
        args.layers,
        n_layers,
    )
    selected_heads = base.parse_selected_heads(
        args.selected_heads,
        n_layers,
    )

    configs = build_configs(
        experiments=experiments,
        layers=active_layers,
        query_scopes=query_scopes,
        pre_biases=pre_biases,
        post_masses=post_masses,
    )

    print(f"decoder={layers_path}, layers={n_layers}")
    print(f"active layers={active_layers}")
    print(f"query scopes={query_scopes}")
    print(f"configs={len(configs)}")

    manager = ObjectAttentionBoostManager(
        layers=decoder_layers,
        active_layers=active_layers,
        selected_heads=selected_heads,
    )

    output_dir = Path(args.output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample_results.jsonl"
    summary_path = output_dir / "summary.csv"
    error_path = output_dir / "errors.jsonl"
    config_path = output_dir / "run_config.json"

    if args.overwrite:
        for path in (sample_path, summary_path, error_path, config_path):
            if path.exists():
                path.unlink()

    if (
        (sample_path.exists() or summary_path.exists())
        and not args.resume
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Results already exist in {output_dir}; "
            "use --resume or --overwrite."
        )

    completed = completed_keys(sample_path) if args.resume else set()

    config_path.write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "model": model_name,
                "repo_id": spec.repo_id,
                "dataset": args.dataset,
                "prompt_jsonl": args.prompt_jsonl,
                "active_layers": active_layers,
                "selected_heads": (
                    {
                        str(layer): sorted(heads)
                        for layer, heads in selected_heads.items()
                    }
                    if selected_heads is not None
                    else None
                ),
                "configs": [config.as_dict() for config in configs],
                "sample_count": len(prior_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pending = sum(
        1
        for row in prior_rows
        for config in configs
        if (int(row["sid"]), config.config_id) not in completed
    )
    progress = tqdm(
        total=pending,
        desc=f"attn-boost:{model_name}",
        unit="generation",
        dynamic_ncols=True,
    )

    sample_counter = 0
    started = time.time()

    try:
        for prior in prior_rows:
            sid = int(prior["sid"])
            record = record_by_sid[sid]
            prompt_row = prompt_rows[sid]

            image = None
            batch = None
            try:
                image = core.record_image(record)
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = base.normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Invalid GT for sid={sid}: "
                        f"{prompt_row['answer_raw']!r}"
                    )

                batch = core.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                batch = base.move_batch_to_device(batch, device)

                prompt_spec = base.build_prompt_position_spec(
                    model=model,
                    tokenizer=processor.tokenizer,
                    input_ids=batch["input_ids"],
                    subject=subject,
                    reference=reference,
                )

                if args.verify_baseline:
                    baseline_text, baseline_prediction = base.generate_relation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=args.max_new_tokens,
                        need_attentions=False,
                    )
                else:
                    baseline_text = str(
                        prior.get("generation_text")
                        or prior.get("baseline_generated_text")
                        or ""
                    )
                    baseline_prediction = base.normalize_relation(
                        prior.get("baseline_prediction")
                    )

                for config in configs:
                    key = (sid, config.config_id)
                    if key in completed:
                        continue

                    manager.configure(
                        config=config,
                        prompt_spec=prompt_spec,
                    )
                    try:
                        text, prediction = base.generate_relation(
                            model=model,
                            processor=processor,
                            batch=batch,
                            max_new_tokens=args.max_new_tokens,
                            need_attentions=True,
                        )
                        mass_stats = manager.stats()

                        result = {
                            "model": model_name,
                            "sid": sid,
                            "group": str(prior["group"]),
                            "group_short": base.group_short(str(prior["group"])),
                            "gt": gt,
                            "subject": subject,
                            "reference": reference,
                            "baseline_prediction": baseline_prediction,
                            "baseline_text": baseline_text,
                            **config.as_dict(),
                            "intervention_prediction": prediction,
                            "intervention_text": text,
                            "parsed": prediction in RELATIONS,
                            "intervention_correct": prediction == gt,
                            "repaired": (
                                str(prior["group"]) == GROUP_B
                                and prediction == gt
                            ),
                            "damaged": (
                                str(prior["group"]) in {GROUP_A, GROUP_C}
                                and prediction != gt
                            ),
                            **mass_stats,
                        }
                        append_jsonl(sample_path, result)
                        completed.add(key)

                    except Exception as exc:
                        append_jsonl(
                            error_path,
                            {
                                "model": model_name,
                                "sid": sid,
                                "group": prior.get("group"),
                                "config_id": config.config_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback_tail": traceback.format_exc().splitlines()[-25:],
                            },
                        )
                        raise
                    finally:
                        manager.reset()

                    progress.update(1)

                sample_counter += 1
                if (
                    args.print_every > 0
                    and sample_counter % args.print_every == 0
                ):
                    elapsed = (time.time() - started) / 60.0
                    tqdm.write(
                        f"[{model_name}] samples={sample_counter}/"
                        f"{len(prior_rows)}, elapsed={elapsed:.1f} min"
                    )

                if (
                    args.empty_cache_every > 0
                    and sample_counter % args.empty_cache_every == 0
                ):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            finally:
                manager.reset()
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

    finally:
        progress.close()
        manager.close()

    rows = base.read_jsonl(sample_path)
    summary = aggregate(rows, configs)
    write_csv(summary_path, summary)
    print_summary(summary)

    print(f"\nSamples: {sample_path}")
    print(f"Summary: {summary_path}")
    if error_path.exists() and error_path.stat().st_size:
        print(f"Errors:  {error_path}")

    del model, processor, decoder_layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = base.parse_models(args.models)
    selected_groups = base.parse_groups(args.groups)
    experiments = parse_experiments(args.experiments)
    query_scopes = parse_query_scopes(args.query_scopes)
    pre_biases = base.parse_float_list(args.pre_biases)
    post_masses = base.parse_float_list(args.post_masses)

    core = base.import_core(args.core_module)
    backend_module = core.import_two_object_module()

    print("=" * 164)
    print("OBJECT-TOKEN ATTENTION BOOST: PRE-SOFTMAX VS POST-SOFTMAX")
    print("=" * 164)
    print(f"models={models}")
    print(f"experiments={experiments}")
    print(f"query_scopes={query_scopes}")
    print(
        "pre_logit_bias is implemented through the exact probability-space "
        "identity p'=normalize(p*exp(b*object_mask))."
    )

    completed_models = 0
    failures: List[Tuple[str, str]] = []

    for model_name in models:
        try:
            run_model(
                args=args,
                core=core,
                backend_module=backend_module,
                model_name=model_name,
                selected_groups=selected_groups,
                experiments=experiments,
                query_scopes=query_scopes,
                pre_biases=pre_biases,
                post_masses=post_masses,
            )
            completed_models += 1
        except Exception as exc:
            failures.append(
                (model_name, f"{type(exc).__name__}: {exc}")
            )
            print(
                f"\n[ERROR] {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    print("\n" + "=" * 164)
    print(f"COMPLETE: {completed_models}/{len(models)}")
    for model_name, error in failures:
        print(f"  failed {model_name}: {error}")

    if completed_models == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
