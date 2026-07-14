#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure the signed directional contribution of object-derived A*V in the
centroid-correct generation groups.

This script is designed to run beside:

    trace_centroid_generation_groups_v2_2.py

It reuses that script's model loading, token-span resolution, v_proj/o_proj
discovery, GQA value-head expansion, and centroid-correct group definitions.

The previous trace measured mainly A*V magnitudes.  This script asks whether
each projected contribution supports the ground-truth relation or its opposite.

Main diagnostics
----------------
1. Answer-routing contribution

   At the final prompt token, isolate the contribution routed from the subject
   and reference object tokens through every attention head:

       u_route(h) = o_proj_h(
           A[last -> subject] V(subject)
         + A[last -> reference] V(reference)
       )

   Two signed diagnostics are reported:

   a) Linear unembedding alignment:
          <u_route(h), W_GT - W_OPPOSITE>

   b) Contextual leave-one-out margin contribution:
          margin(actual after-attention residual)
        - margin(actual residual - u_route(h))

      Positive means the object-derived contribution supports GT over the
      opposite relation in the real residual context.  Negative means it pushes
      toward the opposite relation.

2. Subject/reference role contrast

       u_role(h) = u_subject(h) - u_reference(h)

   Its alignment with W_GT - W_OPPOSITE tests whether a head carries an
   appropriately oriented subject-vs-reference signal rather than merely a
   large amount of object information.

3. Selected spatial-head visual contribution

   For each selected centroid head:

       u_visual_delta(h)
         = o_proj_h(A[subject -> visual]V(visual))
         - o_proj_h(A[reference -> visual]V(visual))

   The script measures its signed GT-vs-opposite alignment and its
   leave-one-out contribution to the actual object-difference state after the
   attention sublayer.

Interpretation
--------------
Positive values support the ground-truth direction.
Negative values support the opposite direction.
Near-zero values carry little directly readable directional evidence.

This is a frozen-model diagnostic.  It does not modify model parameters,
replace answers, or apply an intervention.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SCRIPT_VERSION = "trace-centroid-directional-contribution-v1"

OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


# ---------------------------------------------------------------------------
# CLI and base-module loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--base-module",
        default=None,
        help=(
            "Existing centroid trace module to reuse. When omitted, try "
            "trace_centroid_generation_groups_v2_2 and then "
            "trace_centroid_generation_groups_v2_1."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        help=(
            "Completed centroid group trace directory containing summary.json "
            "and sample_metadata.jsonl."
        ),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--prompt-jsonl", default=None)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument(
        "--relations",
        default="left,right,above,below",
        help="Comma-separated relations to retain.",
    )
    parser.add_argument(
        "--routing-layers",
        default="all",
        help=(
            "Layers for answer-routing directional attribution. Examples: "
            "'all', '23:31', or '20,23:31,34,35'."
        ),
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional random cap per generation-correct/wrong group.",
    )

    parser.add_argument(
        "--match-features",
        default=(
            "centroid_confidence,axis_confidence,head_agreement,"
            "swap_stability,mean_separation,mean_visual_mass,prompt_length,"
            "subject_token_count,reference_token_count"
        ),
        help="Features for relation-stratified unique matching.",
    )
    parser.add_argument(
        "--match-caliper",
        type=float,
        default=0.0,
        help=(
            "Maximum robust-standardized Euclidean matching distance. "
            "0 disables the caliper. Matching is always unique."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_base_module(name: Optional[str]) -> Any:
    candidates = []
    if name:
        candidates.append(name)
    candidates.extend([
        "trace_centroid_generation_groups_v2_2",
        "trace_centroid_generation_groups_v2_1",
    ])

    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            module = importlib.import_module(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue

        required = [
            "RELATIONS",
            "GROUP_CORRECT",
            "GROUP_WRONG",
            "read_jsonl",
            "normalize_relation",
            "resolve_decoder_layers",
            "resolve_final_norm",
            "label_token_id_variants",
            "relation_token_rows",
            "relation_scores_from_states",
            "resolve_visual_indices",
            "locate_object_spans",
            "span_indices",
            "attention_tuple",
            "reshape_projected_values",
            "project_all_heads_isolated",
            "project_one_head_isolated",
            "LayerTraceCollector",
            "load_standard_prompts",
            "resolve_prompt_path",
            "make_question_batch",
            "record_image",
            "configure_processor",
            "resolve_dtype",
            "import_two_object_module",
            "write_csv",
        ]
        missing = [item for item in required if not hasattr(module, item)]
        if missing:
            errors.append(f"{candidate}: missing {missing}")
            continue
        return module

    raise RuntimeError(
        "Could not import a compatible centroid trace module:\n  "
        + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_layer_spec(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))

    layers: List[int] = []
    for item in parse_csv_list(text):
        if ":" in item:
            pieces = item.split(":")
            if len(pieces) != 2:
                raise ValueError(f"Invalid layer range: {item!r}")
            start = int(pieces[0])
            end = int(pieces[1])
            if end < start:
                raise ValueError(f"Descending layer range: {item!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(item))

    layers = sorted(set(layers))
    bad = [layer for layer in layers if not 0 <= layer < n_layers]
    if bad:
        raise ValueError(
            f"Routing layers outside [0,{n_layers - 1}]: {bad}"
        )
    if not layers:
        raise ValueError("--routing-layers resolved to an empty set")
    return layers


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def finite_mean(values: np.ndarray) -> float:
    values = finite_values(values)
    return float(np.mean(values)) if len(values) else float("nan")


def safe_cosine_matrix(
    vectors: torch.Tensor,
    direction: torch.Tensor,
) -> np.ndarray:
    if vectors.ndim != 2 or direction.ndim != 1:
        raise ValueError(
            f"Expected vectors [N,D] and direction [D], got "
            f"{tuple(vectors.shape)}, {tuple(direction.shape)}"
        )
    direction_rows = direction.unsqueeze(0).expand_as(vectors)
    result = F.cosine_similarity(
        vectors.float(),
        direction_rows.float(),
        dim=-1,
        eps=1e-8,
    )
    return result.detach().cpu().numpy().astype(np.float32)


def relation_margin_arrays(
    scores: np.ndarray,
    gt_index: int,
    opposite_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError(f"Invalid relation score shape: {scores.shape}")

    gt = scores[:, gt_index]
    gt_vs_opp = gt - scores[:, opposite_index]

    other = scores.copy()
    other[:, gt_index] = -np.inf
    gt_vs_all = gt - np.max(other, axis=1)
    return gt_vs_opp.astype(np.float32), gt_vs_all.astype(np.float32)


def exact_margins(
    base: Any,
    states: torch.Tensor,
    *,
    gt_index: int,
    opposite_index: int,
    final_norm: Optional[torch.nn.Module],
    token_weight: torch.Tensor,
    token_bias: Optional[torch.Tensor],
    relation_positions: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    scores = base.relation_scores_from_states(
        states,
        final_norm=final_norm,
        token_weight=token_weight,
        token_bias=token_bias,
        relation_positions=relation_positions,
    )
    return relation_margin_arrays(scores, gt_index, opposite_index)


def readout_vectors(
    token_weight: torch.Tensor,
    relation_positions: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Return one mean LM-head row per relation in residual coordinates."""
    weight = token_weight.detach().float().cpu()
    rows = []
    for positions in relation_positions:
        if not positions:
            raise RuntimeError("A relation has no LM-head token rows")
        index = torch.as_tensor(positions, dtype=torch.long)
        rows.append(weight.index_select(0, index).mean(dim=0))
    return torch.stack(rows, dim=0)


def robust_standardize(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        values = result[:, column]
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else 0.0
        values[~np.isfinite(values)] = median
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = 1.4826 * mad
        if scale <= 1e-8:
            scale = float(np.std(values))
        if scale <= 1e-8:
            scale = 1.0
        result[:, column] = (values - center) / scale
    return result


# ---------------------------------------------------------------------------
# Capture subject/reference full attention outputs
# ---------------------------------------------------------------------------

class ObjectAttentionOutputCollector:
    """Capture full attention output at subject/reference query positions."""

    def __init__(self, attention_modules: Sequence[torch.nn.Module], base: Any):
        self.base = base
        self.modules = list(attention_modules)
        self.handles: List[Any] = []
        self.active = False
        self.subject_indices: List[int] = []
        self.reference_indices: List[int] = []
        self.reset()

        for layer_index, module in enumerate(self.modules):
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(layer_index)
                )
            )

    def reset(self) -> None:
        n_layers = len(self.modules)
        self.subject_output: List[Optional[torch.Tensor]] = [None] * n_layers
        self.reference_output: List[Optional[torch.Tensor]] = [None] * n_layers

    def set_sample(
        self,
        subject_indices: Sequence[int],
        reference_indices: Sequence[int],
    ) -> None:
        self.subject_indices = [int(x) for x in subject_indices]
        self.reference_indices = [int(x) for x in reference_indices]
        self.reset()
        self.active = True

    def _make_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if not self.active:
                return
            try:
                tensor = self.base.first_tensor(output)
            except Exception:
                return
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                return
            self.subject_output[layer_index] = (
                tensor[0, self.subject_indices]
                .mean(dim=0)
                .detach()
                .float()
                .cpu()
            )
            self.reference_output[layer_index] = (
                tensor[0, self.reference_indices]
                .mean(dim=0)
                .detach()
                .float()
                .cpu()
            )
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# Per-sample directional attribution
# ---------------------------------------------------------------------------

def allocate_directional_arrays(
    n_layers: int,
    n_heads: int,
    n_spatial_heads: int,
) -> Dict[str, np.ndarray]:
    routing_shape = (n_layers, n_heads)
    spatial_shape = (n_spatial_heads,)

    arrays = {}
    for name in [
        "routing_subject_linear_gt_opp",
        "routing_reference_linear_gt_opp",
        "routing_combined_linear_gt_opp",
        "routing_combined_cosine_gt_opp",
        "routing_combined_linear_gt_all_margin",
        "routing_combined_loo_gt_opp_gain",
        "routing_combined_loo_gt_all_gain",
        "routing_role_contrast_linear_gt_opp",
        "routing_role_contrast_cosine_gt_opp",
    ]:
        arrays[name] = np.full(routing_shape, np.nan, dtype=np.float32)

    arrays["routing_projection_available"] = np.zeros(
        routing_shape, dtype=np.int8
    )

    for name in [
        "spatial_subject_linear_gt_opp",
        "spatial_reference_linear_gt_opp",
        "spatial_delta_linear_gt_opp",
        "spatial_delta_cosine_gt_opp",
        "spatial_delta_linear_gt_all_margin",
        "spatial_delta_loo_gt_opp_gain",
        "spatial_delta_loo_gt_all_gain",
    ]:
        arrays[name] = np.full(spatial_shape, np.nan, dtype=np.float32)

    arrays["spatial_direction_correct"] = np.full(
        spatial_shape, -1, dtype=np.int8
    )
    arrays["spatial_projection_available"] = np.zeros(
        spatial_shape, dtype=np.int8
    )
    return arrays


def compute_directional_attribution(
    *,
    base: Any,
    attentions: Sequence[torch.Tensor],
    collector: Any,
    object_output_collector: ObjectAttentionOutputCollector,
    subject_indices: Sequence[int],
    reference_indices: Sequence[int],
    visual_indices: Sequence[int],
    selected_heads: Sequence[Dict[str, Any]],
    routing_layers: Sequence[int],
    last_index: int,
    gt: str,
    relation_vectors: torch.Tensor,
    final_norm: Optional[torch.nn.Module],
    token_weight: torch.Tensor,
    token_bias: Optional[torch.Tensor],
    relation_positions: Sequence[Sequence[int]],
) -> Dict[str, np.ndarray]:
    n_layers = len(attentions)
    n_heads = int(attentions[0].shape[1])

    output = allocate_directional_arrays(
        n_layers=n_layers,
        n_heads=n_heads,
        n_spatial_heads=len(selected_heads),
    )

    relation_to_index = {
        relation: index for index, relation in enumerate(base.RELATIONS)
    }
    gt_index = relation_to_index[gt]
    opposite_relation = OPPOSITE[gt]
    opposite_index = relation_to_index[opposite_relation]

    gt_vector = relation_vectors[gt_index]
    opposite_vector = relation_vectors[opposite_index]
    direction_gt_opp = gt_vector - opposite_vector
    direction_gt_all = (
        gt_vector
        - torch.stack([
            relation_vectors[index]
            for index in range(len(base.RELATIONS))
            if index != gt_index
        ]).mean(dim=0)
    )

    subj = torch.as_tensor(subject_indices, dtype=torch.long)
    ref = torch.as_tensor(reference_indices, dtype=torch.long)
    vis = torch.as_tensor(visual_indices, dtype=torch.long)

    selected_lookup = {
        (int(row["layer"]), int(row["head"])): selected_index
        for selected_index, row in enumerate(selected_heads)
    }
    routing_layer_set = set(int(x) for x in routing_layers)

    for layer_index, attention_tensor in enumerate(attentions):
        attention = attention_tensor[0].detach().float().cpu()
        if attention.ndim != 3:
            raise RuntimeError(
                f"Layer {layer_index} attention shape={tuple(attention.shape)}"
            )
        if int(attention.shape[0]) != n_heads:
            raise RuntimeError("Attention head count changed across layers")

        # ------------------------------------------------------------------
        # Last-token routing from object tokens
        # ------------------------------------------------------------------
        if layer_index in routing_layer_set:
            object_values_flat = collector.object_values[layer_index]
            input_last = collector.layer_input_last[layer_index]
            full_attention_last = collector.attention_output_last[layer_index]

            if (
                object_values_flat is not None
                and input_last is not None
                and full_attention_last is not None
            ):
                values = base.reshape_projected_values(
                    object_values_flat,
                    n_attention_heads=n_heads,
                    attention_module=collector.attention_modules[layer_index],
                ).float()

                n_subject = len(subject_indices)
                values_subject = values[:n_subject].permute(1, 0, 2)
                values_reference = values[n_subject:].permute(1, 0, 2)

                last_row = attention[:, last_index, :]
                weights_subject = last_row.index_select(-1, subj)
                weights_reference = last_row.index_select(-1, ref)

                subject_head = torch.einsum(
                    "ht,htd->hd",
                    weights_subject,
                    values_subject,
                )
                reference_head = torch.einsum(
                    "ht,htd->hd",
                    weights_reference,
                    values_reference,
                )

                projected_subject = base.project_all_heads_isolated(
                    subject_head,
                    collector.o_projections[layer_index],
                )
                projected_reference = base.project_all_heads_isolated(
                    reference_head,
                    collector.o_projections[layer_index],
                )

                if (
                    projected_subject is not None
                    and projected_reference is not None
                ):
                    projected_subject = (
                        projected_subject.detach().float().cpu()
                    )
                    projected_reference = (
                        projected_reference.detach().float().cpu()
                    )
                    if (
                        projected_subject.shape == (n_heads, len(gt_vector))
                        and projected_reference.shape
                        == (n_heads, len(gt_vector))
                    ):
                        combined = projected_subject + projected_reference
                        role_contrast = (
                            projected_subject - projected_reference
                        )

                        output[
                            "routing_subject_linear_gt_opp"
                        ][layer_index] = (
                            projected_subject @ direction_gt_opp
                        ).numpy().astype(np.float32)
                        output[
                            "routing_reference_linear_gt_opp"
                        ][layer_index] = (
                            projected_reference @ direction_gt_opp
                        ).numpy().astype(np.float32)
                        output[
                            "routing_combined_linear_gt_opp"
                        ][layer_index] = (
                            combined @ direction_gt_opp
                        ).numpy().astype(np.float32)
                        output[
                            "routing_combined_cosine_gt_opp"
                        ][layer_index] = safe_cosine_matrix(
                            combined,
                            direction_gt_opp,
                        )

                        linear_relation_scores = (
                            combined @ relation_vectors.T
                        ).numpy()
                        _, linear_gt_all = relation_margin_arrays(
                            linear_relation_scores,
                            gt_index,
                            opposite_index,
                        )
                        output[
                            "routing_combined_linear_gt_all_margin"
                        ][layer_index] = linear_gt_all

                        output[
                            "routing_role_contrast_linear_gt_opp"
                        ][layer_index] = (
                            role_contrast @ direction_gt_opp
                        ).numpy().astype(np.float32)
                        output[
                            "routing_role_contrast_cosine_gt_opp"
                        ][layer_index] = safe_cosine_matrix(
                            role_contrast,
                            direction_gt_opp,
                        )

                        actual_after_attention = (
                            input_last.float()
                            + full_attention_last.float()
                        )
                        baseline_opp, baseline_all = exact_margins(
                            base,
                            actual_after_attention.unsqueeze(0),
                            gt_index=gt_index,
                            opposite_index=opposite_index,
                            final_norm=final_norm,
                            token_weight=token_weight,
                            token_bias=token_bias,
                            relation_positions=relation_positions,
                        )
                        removed_states = (
                            actual_after_attention.unsqueeze(0) - combined
                        )
                        removed_opp, removed_all = exact_margins(
                            base,
                            removed_states,
                            gt_index=gt_index,
                            opposite_index=opposite_index,
                            final_norm=final_norm,
                            token_weight=token_weight,
                            token_bias=token_bias,
                            relation_positions=relation_positions,
                        )
                        output[
                            "routing_combined_loo_gt_opp_gain"
                        ][layer_index] = (
                            baseline_opp[0] - removed_opp
                        ).astype(np.float32)
                        output[
                            "routing_combined_loo_gt_all_gain"
                        ][layer_index] = (
                            baseline_all[0] - removed_all
                        ).astype(np.float32)
                        output[
                            "routing_projection_available"
                        ][layer_index] = 1

        # ------------------------------------------------------------------
        # Selected centroid heads: subject/reference visual A*V difference
        # ------------------------------------------------------------------
        visual_values_flat = collector.visual_values[layer_index]
        if visual_values_flat is None:
            continue

        visual_values = base.reshape_projected_values(
            visual_values_flat,
            n_attention_heads=n_heads,
            attention_module=collector.attention_modules[layer_index],
        ).float().permute(1, 0, 2)

        input_subject = collector.layer_input_subject[layer_index]
        input_reference = collector.layer_input_reference[layer_index]
        full_subject_attention = (
            object_output_collector.subject_output[layer_index]
        )
        full_reference_attention = (
            object_output_collector.reference_output[layer_index]
        )

        for head in range(n_heads):
            selected_index = selected_lookup.get((layer_index, head))
            if selected_index is None:
                continue

            subject_weights = (
                attention[head]
                .index_select(0, subj)
                .index_select(1, vis)
            )
            reference_weights = (
                attention[head]
                .index_select(0, ref)
                .index_select(1, vis)
            )
            value_head = visual_values[head]

            subject_av = torch.einsum(
                "qv,vd->qd",
                subject_weights,
                value_head,
            ).mean(dim=0)
            reference_av = torch.einsum(
                "qv,vd->qd",
                reference_weights,
                value_head,
            ).mean(dim=0)

            projected = base.project_one_head_isolated(
                torch.stack([subject_av, reference_av], dim=0),
                o_proj=collector.o_projections[layer_index],
                head_index=head,
                total_heads=n_heads,
            )
            if projected is None:
                continue

            projected = projected.detach().float().cpu()
            if projected.ndim != 2 or projected.shape[0] != 2:
                continue

            projected_subject = projected[0]
            projected_reference = projected[1]
            projected_delta = projected_subject - projected_reference

            output[
                "spatial_subject_linear_gt_opp"
            ][selected_index] = float(
                torch.dot(projected_subject, direction_gt_opp).item()
            )
            output[
                "spatial_reference_linear_gt_opp"
            ][selected_index] = float(
                torch.dot(projected_reference, direction_gt_opp).item()
            )
            signed_delta = float(
                torch.dot(projected_delta, direction_gt_opp).item()
            )
            output[
                "spatial_delta_linear_gt_opp"
            ][selected_index] = signed_delta
            output[
                "spatial_delta_cosine_gt_opp"
            ][selected_index] = float(
                F.cosine_similarity(
                    projected_delta.unsqueeze(0),
                    direction_gt_opp.unsqueeze(0),
                    dim=-1,
                    eps=1e-8,
                ).item()
            )

            linear_relation_scores = (
                projected_delta @ relation_vectors.T
            ).unsqueeze(0).numpy()
            _, linear_gt_all = relation_margin_arrays(
                linear_relation_scores,
                gt_index,
                opposite_index,
            )
            output[
                "spatial_delta_linear_gt_all_margin"
            ][selected_index] = float(linear_gt_all[0])
            output[
                "spatial_direction_correct"
            ][selected_index] = int(signed_delta > 0.0)

            if (
                input_subject is not None
                and input_reference is not None
                and full_subject_attention is not None
                and full_reference_attention is not None
            ):
                actual_subject_after_attention = (
                    input_subject.float()
                    + full_subject_attention.float()
                )
                actual_reference_after_attention = (
                    input_reference.float()
                    + full_reference_attention.float()
                )
                actual_delta = (
                    actual_subject_after_attention
                    - actual_reference_after_attention
                )

                baseline_opp, baseline_all = exact_margins(
                    base,
                    actual_delta.unsqueeze(0),
                    gt_index=gt_index,
                    opposite_index=opposite_index,
                    final_norm=final_norm,
                    token_weight=token_weight,
                    token_bias=token_bias,
                    relation_positions=relation_positions,
                )
                removed_opp, removed_all = exact_margins(
                    base,
                    (actual_delta - projected_delta).unsqueeze(0),
                    gt_index=gt_index,
                    opposite_index=opposite_index,
                    final_norm=final_norm,
                    token_weight=token_weight,
                    token_bias=token_bias,
                    relation_positions=relation_positions,
                )
                output[
                    "spatial_delta_loo_gt_opp_gain"
                ][selected_index] = float(
                    baseline_opp[0] - removed_opp[0]
                )
                output[
                    "spatial_delta_loo_gt_all_gain"
                ][selected_index] = float(
                    baseline_all[0] - removed_all[0]
                )

            output[
                "spatial_projection_available"
            ][selected_index] = 1

    return output


# ---------------------------------------------------------------------------
# Matching and statistics
# ---------------------------------------------------------------------------

def build_unique_pairs(
    metadata_rows: Sequence[Dict[str, Any]],
    *,
    relations: Sequence[str],
    group_correct: str,
    group_wrong: str,
    feature_names: Sequence[str],
    caliper: float,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    for relation in relations:
        correct_indices = [
            index
            for index, row in enumerate(metadata_rows)
            if row["group"] == group_correct and row["gt"] == relation
        ]
        wrong_indices = [
            index
            for index, row in enumerate(metadata_rows)
            if row["group"] == group_wrong and row["gt"] == relation
        ]
        if not correct_indices or not wrong_indices:
            continue

        all_indices = wrong_indices + correct_indices
        raw = np.asarray([
            [
                float(metadata_rows[index].get(feature, np.nan))
                for feature in feature_names
            ]
            for index in all_indices
        ], dtype=np.float64)
        standardized = robust_standardize(raw)

        wrong_matrix = standardized[:len(wrong_indices)]
        correct_matrix = standardized[len(wrong_indices):]
        distance = np.linalg.norm(
            wrong_matrix[:, None, :] - correct_matrix[None, :, :],
            axis=-1,
        )

        assignments: List[Tuple[int, int]] = []
        try:
            from scipy.optimize import linear_sum_assignment
            row_ids, column_ids = linear_sum_assignment(distance)
            assignments = list(zip(row_ids.tolist(), column_ids.tolist()))
        except Exception:
            edges = [
                (float(distance[i, j]), i, j)
                for i in range(distance.shape[0])
                for j in range(distance.shape[1])
            ]
            edges.sort()
            used_wrong = set()
            used_correct = set()
            for _, i, j in edges:
                if i in used_wrong or j in used_correct:
                    continue
                assignments.append((i, j))
                used_wrong.add(i)
                used_correct.add(j)
                if len(used_wrong) == min(
                    len(wrong_indices), len(correct_indices)
                ):
                    break

        for local_wrong, local_correct in assignments:
            pair_distance = float(distance[local_wrong, local_correct])
            if caliper > 0.0 and pair_distance > caliper:
                continue
            wrong_index = wrong_indices[local_wrong]
            correct_index = correct_indices[local_correct]
            row = {
                "relation": relation,
                "wrong_index": wrong_index,
                "correct_index": correct_index,
                "wrong_sid": int(metadata_rows[wrong_index]["sid"]),
                "correct_sid": int(metadata_rows[correct_index]["sid"]),
                "distance": pair_distance,
            }
            for feature in feature_names:
                correct_value = float(
                    metadata_rows[correct_index].get(feature, np.nan)
                )
                wrong_value = float(
                    metadata_rows[wrong_index].get(feature, np.nan)
                )
                row[f"{feature}_correct"] = correct_value
                row[f"{feature}_wrong"] = wrong_value
                row[f"{feature}_difference"] = (
                    correct_value - wrong_value
                )
            pairs.append(row)

    return pairs


def bootstrap_unpaired_difference(
    correct: np.ndarray,
    wrong: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    correct = finite_values(correct)
    wrong = finite_values(wrong)
    if not len(correct) or not len(wrong):
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        c = rng.choice(correct, size=len(correct), replace=True)
        w = rng.choice(wrong, size=len(wrong), replace=True)
        estimates[index] = np.mean(c) - np.mean(w)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_paired_difference(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    differences = finite_values(differences)
    if not len(differences):
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )
        estimates[index] = np.mean(draw)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def unpaired_comparison(
    *,
    values: np.ndarray,
    group_codes: np.ndarray,
    relation_codes: np.ndarray,
    relation_index: Optional[int],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    mask = np.isfinite(values)
    if relation_index is not None:
        mask &= relation_codes == relation_index

    correct = values[mask & (group_codes == 0)]
    wrong = values[mask & (group_codes == 1)]

    correct_f = finite_values(correct)
    wrong_f = finite_values(wrong)
    difference = (
        float(np.mean(correct_f) - np.mean(wrong_f))
        if len(correct_f) and len(wrong_f)
        else float("nan")
    )

    if len(correct_f) > 1 and len(wrong_f) > 1:
        pooled_denominator = len(correct_f) + len(wrong_f) - 2
        pooled_variance = (
            (len(correct_f) - 1) * np.var(correct_f, ddof=1)
            + (len(wrong_f) - 1) * np.var(wrong_f, ddof=1)
        ) / max(pooled_denominator, 1)
        pooled_std = math.sqrt(max(float(pooled_variance), 0.0))
        cohen_d = (
            difference / pooled_std
            if pooled_std > 1e-12
            else float("nan")
        )
    else:
        cohen_d = float("nan")

    low, high = bootstrap_unpaired_difference(
        correct_f,
        wrong_f,
        samples=bootstrap_samples,
        seed=seed,
    )

    return {
        "n_correct": int(len(correct_f)),
        "n_wrong": int(len(wrong_f)),
        "mean_correct": finite_mean(correct_f),
        "mean_wrong": finite_mean(wrong_f),
        "mean_correct_minus_wrong": difference,
        "cohen_d": float(cohen_d),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "positive_fraction_correct": (
            float(np.mean(correct_f > 0.0))
            if len(correct_f)
            else float("nan")
        ),
        "positive_fraction_wrong": (
            float(np.mean(wrong_f > 0.0))
            if len(wrong_f)
            else float("nan")
        ),
    }


def paired_comparison(
    *,
    values: np.ndarray,
    pairs: Sequence[Dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    differences = []
    correct_values = []
    wrong_values = []

    for pair in pairs:
        correct = float(values[int(pair["correct_index"])])
        wrong = float(values[int(pair["wrong_index"])])
        if not np.isfinite(correct) or not np.isfinite(wrong):
            continue
        correct_values.append(correct)
        wrong_values.append(wrong)
        differences.append(correct - wrong)

    differences_array = np.asarray(differences, dtype=np.float64)
    std = (
        float(np.std(differences_array, ddof=1))
        if len(differences_array) > 1
        else float("nan")
    )
    dz = (
        float(np.mean(differences_array) / std)
        if len(differences_array) > 1 and std > 1e-12
        else float("nan")
    )
    low, high = bootstrap_paired_difference(
        differences_array,
        samples=bootstrap_samples,
        seed=seed,
    )

    return {
        "n_pairs": int(len(differences_array)),
        "mean_correct": finite_mean(np.asarray(correct_values)),
        "mean_wrong": finite_mean(np.asarray(wrong_values)),
        "mean_paired_difference": finite_mean(differences_array),
        "paired_cohen_dz": dz,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "positive_fraction_correct": (
            float(np.mean(np.asarray(correct_values) > 0.0))
            if correct_values
            else float("nan")
        ),
        "positive_fraction_wrong": (
            float(np.mean(np.asarray(wrong_values) > 0.0))
            if wrong_values
            else float("nan")
        ),
    }


def matching_balance_rows(
    metadata_rows: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    features: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []
    for feature in features:
        differences = []
        correct_values = []
        wrong_values = []
        for pair in pairs:
            correct = float(
                metadata_rows[int(pair["correct_index"])].get(
                    feature, np.nan
                )
            )
            wrong = float(
                metadata_rows[int(pair["wrong_index"])].get(
                    feature, np.nan
                )
            )
            if not np.isfinite(correct) or not np.isfinite(wrong):
                continue
            correct_values.append(correct)
            wrong_values.append(wrong)
            differences.append(correct - wrong)

        differences_array = np.asarray(differences, dtype=np.float64)
        std = (
            float(np.std(differences_array, ddof=1))
            if len(differences_array) > 1
            else float("nan")
        )
        rows.append({
            "feature": feature,
            "n_pairs": len(differences_array),
            "mean_correct": finite_mean(np.asarray(correct_values)),
            "mean_wrong": finite_mean(np.asarray(wrong_values)),
            "mean_paired_difference": finite_mean(differences_array),
            "paired_cohen_dz": (
                finite_mean(differences_array) / std
                if np.isfinite(std) and std > 1e-12
                else float("nan")
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

ROUTING_METRICS = [
    "routing_subject_linear_gt_opp",
    "routing_reference_linear_gt_opp",
    "routing_combined_linear_gt_opp",
    "routing_combined_cosine_gt_opp",
    "routing_combined_linear_gt_all_margin",
    "routing_combined_loo_gt_opp_gain",
    "routing_combined_loo_gt_all_gain",
    "routing_role_contrast_linear_gt_opp",
    "routing_role_contrast_cosine_gt_opp",
]

SPATIAL_METRICS = [
    "spatial_subject_linear_gt_opp",
    "spatial_reference_linear_gt_opp",
    "spatial_delta_linear_gt_opp",
    "spatial_delta_cosine_gt_opp",
    "spatial_delta_linear_gt_all_margin",
    "spatial_delta_loo_gt_opp_gain",
    "spatial_delta_loo_gt_all_gain",
]


def summarize_routing(
    *,
    stacked: Dict[str, np.ndarray],
    routing_layers: Sequence[int],
    n_heads: int,
    group_codes: np.ndarray,
    relation_codes: np.ndarray,
    relations: Sequence[str],
    relation_to_index: Dict[str, int],
    pairs: Sequence[Dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows = []
    paired_rows = []

    for metric_index, metric in enumerate(ROUTING_METRICS):
        matrix = stacked[metric]
        for layer in routing_layers:
            for head in range(n_heads):
                values = matrix[:, layer, head]

                relation_options: List[Tuple[str, Optional[int]]] = [
                    ("all", None)
                ] + [
                    (relation, relation_to_index[relation])
                    for relation in relations
                ]

                for relation_name, relation_index in relation_options:
                    row = unpaired_comparison(
                        values=values,
                        group_codes=group_codes,
                        relation_codes=relation_codes,
                        relation_index=relation_index,
                        bootstrap_samples=bootstrap_samples,
                        seed=(
                            seed
                            + metric_index * 100000
                            + layer * 1000
                            + head * 10
                            + (relation_index or 0)
                        ),
                    )
                    row.update({
                        "metric": metric,
                        "relation": relation_name,
                        "layer": layer,
                        "head": head,
                    })
                    group_rows.append(row)

                paired = paired_comparison(
                    values=values,
                    pairs=pairs,
                    bootstrap_samples=bootstrap_samples,
                    seed=(
                        seed
                        + 900000
                        + metric_index * 10000
                        + layer * 100
                        + head
                    ),
                )
                paired.update({
                    "metric": metric,
                    "relation": "all",
                    "layer": layer,
                    "head": head,
                })
                paired_rows.append(paired)

    return group_rows, paired_rows


def summarize_spatial(
    *,
    stacked: Dict[str, np.ndarray],
    selected_heads: Sequence[Dict[str, Any]],
    group_codes: np.ndarray,
    relation_codes: np.ndarray,
    relations: Sequence[str],
    relation_to_index: Dict[str, int],
    pairs: Sequence[Dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows = []
    paired_rows = []

    for metric_index, metric in enumerate(SPATIAL_METRICS):
        matrix = stacked[metric]
        for selected_index, selected in enumerate(selected_heads):
            values = matrix[:, selected_index]

            relation_options: List[Tuple[str, Optional[int]]] = [
                ("all", None)
            ] + [
                (relation, relation_to_index[relation])
                for relation in relations
            ]

            for relation_name, relation_index in relation_options:
                row = unpaired_comparison(
                    values=values,
                    group_codes=group_codes,
                    relation_codes=relation_codes,
                    relation_index=relation_index,
                    bootstrap_samples=bootstrap_samples,
                    seed=(
                        seed
                        + 1200000
                        + metric_index * 1000
                        + selected_index * 10
                        + (relation_index or 0)
                    ),
                )
                row.update({
                    "metric": metric,
                    "relation": relation_name,
                    "selected_rank": selected_index + 1,
                    "layer": int(selected["layer"]),
                    "head": int(selected["head"]),
                })
                group_rows.append(row)

            paired = paired_comparison(
                values=values,
                pairs=pairs,
                bootstrap_samples=bootstrap_samples,
                seed=(
                    seed
                    + 1500000
                    + metric_index * 100
                    + selected_index
                ),
            )
            paired.update({
                "metric": metric,
                "relation": "all",
                "selected_rank": selected_index + 1,
                "layer": int(selected["layer"]),
                "head": int(selected["head"]),
            })
            paired_rows.append(paired)

    # Add directional sign accuracy as a directly interpretable metric.
    sign_matrix = stacked["spatial_direction_correct"].astype(np.float32)
    sign_matrix[sign_matrix < 0] = np.nan
    for selected_index, selected in enumerate(selected_heads):
        values = sign_matrix[:, selected_index]
        row = unpaired_comparison(
            values=values,
            group_codes=group_codes,
            relation_codes=relation_codes,
            relation_index=None,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1700000 + selected_index,
        )
        row.update({
            "metric": "spatial_direction_correct",
            "relation": "all",
            "selected_rank": selected_index + 1,
            "layer": int(selected["layer"]),
            "head": int(selected["head"]),
        })
        group_rows.append(row)

        paired = paired_comparison(
            values=values,
            pairs=pairs,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1800000 + selected_index,
        )
        paired.update({
            "metric": "spatial_direction_correct",
            "relation": "all",
            "selected_rank": selected_index + 1,
            "layer": int(selected["layer"]),
            "head": int(selected["head"]),
        })
        paired_rows.append(paired)

    return group_rows, paired_rows


def sample_summary_rows(
    metadata_rows: Sequence[Dict[str, Any]],
    stacked: Dict[str, np.ndarray],
    routing_layers: Sequence[int],
) -> List[Dict[str, Any]]:
    rows = []
    for index, metadata in enumerate(metadata_rows):
        row = {
            "row_index": index,
            "sid": int(metadata["sid"]),
            "group": metadata["group"],
            "gt": metadata["gt"],
            "baseline_prediction": metadata.get("baseline_prediction"),
            "centroid_confidence": metadata.get("centroid_confidence"),
        }

        routing_slice = stacked[
            "routing_combined_loo_gt_opp_gain"
        ][index, routing_layers, :]
        role_slice = stacked[
            "routing_role_contrast_linear_gt_opp"
        ][index, routing_layers, :]
        spatial_slice = stacked[
            "spatial_delta_loo_gt_opp_gain"
        ][index]
        spatial_linear = stacked[
            "spatial_delta_linear_gt_opp"
        ][index]

        row.update({
            "routing_loo_gt_opp_mean": finite_mean(routing_slice),
            "routing_loo_gt_opp_positive_fraction": (
                float(np.mean(
                    finite_values(routing_slice) > 0.0
                ))
                if len(finite_values(routing_slice))
                else float("nan")
            ),
            "routing_role_linear_mean": finite_mean(role_slice),
            "spatial_loo_gt_opp_mean": finite_mean(spatial_slice),
            "spatial_linear_gt_opp_mean": finite_mean(spatial_linear),
            "spatial_direction_correct_fraction": (
                float(np.mean(
                    finite_values(spatial_linear) > 0.0
                ))
                if len(finite_values(spatial_linear))
                else float("nan")
            ),
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be >= 0")
    if args.max_per_group is not None and args.max_per_group <= 0:
        raise ValueError("--max-per-group must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = load_base_module(args.base_module)

    trace_dir = Path(args.trace_dir)
    summary_path = trace_dir / "summary.json"
    metadata_path = trace_dir / "sample_metadata.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    trace_summary = read_json(summary_path)
    dataset = args.dataset or str(trace_summary["dataset"])
    model_name = args.model or str(trace_summary["model"])

    if args.dataset and args.dataset != trace_summary.get("dataset"):
        raise ValueError(
            f"--dataset={args.dataset} disagrees with trace summary "
            f"{trace_summary.get('dataset')}"
        )
    if args.model and args.model != trace_summary.get("model"):
        raise ValueError(
            f"--model={args.model} disagrees with trace summary "
            f"{trace_summary.get('model')}"
        )

    allowed_relations = []
    for item in parse_csv_list(args.relations):
        relation = base.normalize_relation(item)
        if relation not in base.RELATIONS:
            raise ValueError(f"Unsupported relation: {item!r}")
        if relation not in allowed_relations:
            allowed_relations.append(relation)

    metadata_rows = [
        row
        for row in base.read_jsonl(metadata_path)
        if row.get("group") in (
            base.GROUP_CORRECT,
            base.GROUP_WRONG,
        )
        and base.normalize_relation(row.get("gt")) in allowed_relations
    ]
    for row in metadata_rows:
        row["gt"] = base.normalize_relation(row["gt"])

    if args.max_per_group is not None:
        rng = random.Random(args.seed)
        selected_rows = []
        for group in (base.GROUP_CORRECT, base.GROUP_WRONG):
            group_rows = [
                row for row in metadata_rows if row["group"] == group
            ]
            rng.shuffle(group_rows)
            selected_rows.extend(group_rows[:args.max_per_group])
        metadata_rows = sorted(
            selected_rows,
            key=lambda row: int(row["sid"]),
        )

    if not metadata_rows:
        raise RuntimeError("No trace metadata rows selected")

    selected_heads = trace_summary.get("selected_spatial_heads")
    if not isinstance(selected_heads, list) or not selected_heads:
        selected_heads = base.load_selected_heads(
            Path(trace_summary.get("prior_dir", trace_dir))
        )
    selected_heads = [
        {
            **row,
            "layer": int(row["layer"]),
            "head": int(row["head"]),
        }
        for row in selected_heads
    ]
    selected_layers = sorted({
        int(row["layer"]) for row in selected_heads
    })

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_metadata_path = output_dir / "sample_metadata.jsonl"
    errors_path = output_dir / "errors.jsonl"
    for path in (copied_metadata_path, errors_path):
        if path.exists():
            path.unlink()

    print("\nSelected centroid heads:")
    for rank, row in enumerate(selected_heads, 1):
        print(
            f"  {rank:2d}. L{int(row['layer']):02d}"
            f"H{int(row['head']):02d}"
        )

    print("\nSelected groups:")
    group_counter = Counter(row["group"] for row in metadata_rows)
    relation_counter = Counter(
        (row["group"], row["gt"]) for row in metadata_rows
    )
    for group in (base.GROUP_CORRECT, base.GROUP_WRONG):
        print(f"  {group}: {group_counter[group]}")
        for relation in allowed_relations:
            print(
                f"    {relation:6s}: "
                f"{relation_counter[(group, relation)]}"
            )

    support = base.import_two_object_module()
    records, audit = support.load_records(
        dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {
        int(record.sid): record for record in records
    }

    prompt_args = argparse.Namespace(
        dataset=dataset,
        prompt_jsonl=args.prompt_jsonl,
    )
    prompt_path = base.resolve_prompt_path(prompt_args)
    prompt_rows = base.load_standard_prompts(prompt_path)

    selected_sids = [int(row["sid"]) for row in metadata_rows]
    missing = [
        sid
        for sid in selected_sids
        if sid not in record_by_sid or sid not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Missing records/prompts for {len(missing)} SIDs; "
            f"first={missing[:10]}"
        )

    if model_name not in support.SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    spec = support.SPECS[model_name]
    model_cls = getattr(base.transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={base.transformers.__version__} "
            f"has no {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": "eager",
    }

    print(f"\nLoading {model_name}: {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()
    processor = base.AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = base.resolve_decoder_layers(model)
    n_layers = len(layers)
    routing_layers = parse_layer_spec(args.routing_layers, n_layers)

    final_norm, final_norm_path = base.resolve_final_norm(model)
    label_token_ids = base.label_token_id_variants(processor.tokenizer)
    token_weight, token_bias, relation_positions = base.relation_token_rows(
        model,
        label_token_ids,
    )
    relation_vectors = readout_vectors(
        token_weight,
        relation_positions,
    )

    collector = base.LayerTraceCollector(layers, selected_layers)
    object_output_collector = ObjectAttentionOutputCollector(
        collector.attention_modules,
        base,
    )

    projection_diagnostics = collector.projection_diagnostics()
    (output_dir / "projection_diagnostics.json").write_text(
        json.dumps(projection_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Decoder layers: {layers_path} ({n_layers})")
    print(f"Final norm:     {final_norm_path}")
    print(f"Routing layers: {routing_layers}")

    arrays: Dict[str, List[np.ndarray]] = defaultdict(list)
    successful_metadata: List[Dict[str, Any]] = []
    n_heads: Optional[int] = None

    started = time.time()
    completed = 0

    try:
        for metadata in tqdm(
            metadata_rows,
            desc=f"directional-trace:{model_name}",
        ):
            sid = int(metadata["sid"])
            batch = None
            image = None
            try:
                prompt_row = prompt_rows[sid]
                record = record_by_sid[sid]
                image = base.record_image(record)

                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = base.normalize_relation(prompt_row["answer_raw"])
                if gt != metadata["gt"]:
                    raise RuntimeError(
                        f"GT mismatch sid={sid}: "
                        f"prompt={gt}, trace={metadata['gt']}"
                    )

                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                input_ids = [
                    int(x)
                    for x in batch["input_ids"][0].detach().cpu().tolist()
                ]
                subject_span, reference_span = base.locate_object_spans(
                    processor.tokenizer,
                    input_ids,
                    subject,
                    reference,
                )
                subject_indices = base.span_indices(subject_span)
                reference_indices = base.span_indices(reference_span)
                visual_indices = base.resolve_visual_indices(
                    model,
                    processor,
                    batch,
                    input_ids,
                )
                last_index = len(input_ids) - 1

                collector.set_sample(
                    subject_indices=subject_indices,
                    reference_indices=reference_indices,
                    visual_indices=visual_indices,
                    last_index=last_index,
                )
                object_output_collector.set_sample(
                    subject_indices,
                    reference_indices,
                )

                try:
                    with torch.inference_mode():
                        outputs = model(
                            **batch,
                            output_attentions=True,
                            output_hidden_states=False,
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    collector.active = False
                    object_output_collector.active = False

                attentions = base.attention_tuple(outputs)
                if len(attentions) != n_layers:
                    raise RuntimeError(
                        f"Expected {n_layers} attentions, "
                        f"got {len(attentions)}"
                    )

                sample_n_heads = int(attentions[0].shape[1])
                if n_heads is None:
                    n_heads = sample_n_heads
                elif n_heads != sample_n_heads:
                    raise RuntimeError(
                        f"Head count changed: {n_heads} -> {sample_n_heads}"
                    )

                sample_arrays = compute_directional_attribution(
                    base=base,
                    attentions=attentions,
                    collector=collector,
                    object_output_collector=object_output_collector,
                    subject_indices=subject_indices,
                    reference_indices=reference_indices,
                    visual_indices=visual_indices,
                    selected_heads=selected_heads,
                    routing_layers=routing_layers,
                    last_index=last_index,
                    gt=gt,
                    relation_vectors=relation_vectors,
                    final_norm=final_norm,
                    token_weight=token_weight,
                    token_bias=token_bias,
                    relation_positions=relation_positions,
                )

                row_index = len(successful_metadata)
                saved_metadata = {
                    **metadata,
                    "row_index": row_index,
                    "subject": subject,
                    "reference": reference,
                    "subject_token_count": len(subject_indices),
                    "reference_token_count": len(reference_indices),
                    "n_visual_tokens": len(visual_indices),
                    "prompt_length": len(input_ids),
                }
                successful_metadata.append(saved_metadata)
                append_jsonl(copied_metadata_path, saved_metadata)
                for key, value in sample_arrays.items():
                    arrays[key].append(value)

                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    spatial_signed = sample_arrays[
                        "spatial_delta_linear_gt_opp"
                    ]
                    routing_signed = sample_arrays[
                        "routing_combined_loo_gt_opp_gain"
                    ][routing_layers]
                    tqdm.write(
                        f"\n[{completed}/{len(metadata_rows)}] sid={sid} | "
                        f"{metadata['group']} | GT={gt}\n"
                        f"  spatial signed mean="
                        f"{finite_mean(spatial_signed):+.4f} | "
                        f"routing LOO mean="
                        f"{finite_mean(routing_signed):+.4f}"
                    )

                del outputs, attentions

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "group": metadata.get("group"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-25:],
                })
                tqdm.write(
                    f"\n[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                collector.active = False
                object_output_collector.active = False
                if batch is not None:
                    del batch
                if image is not None:
                    del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not successful_metadata:
            raise RuntimeError("No samples were traced successfully")
        if n_heads is None:
            raise RuntimeError("Could not determine attention head count")

        stacked = {
            key: np.stack(values, axis=0)
            for key, values in arrays.items()
        }

        sids = np.asarray(
            [int(row["sid"]) for row in successful_metadata],
            dtype=np.int64,
        )
        group_codes = np.asarray([
            0 if row["group"] == base.GROUP_CORRECT else 1
            for row in successful_metadata
        ], dtype=np.int8)
        relation_to_index = {
            relation: index
            for index, relation in enumerate(base.RELATIONS)
        }
        relation_codes = np.asarray([
            relation_to_index[row["gt"]]
            for row in successful_metadata
        ], dtype=np.int8)

        stacked.update({
            "sids": sids,
            "group_codes": group_codes,
            "relation_codes": relation_codes,
            "routing_layers": np.asarray(
                routing_layers, dtype=np.int16
            ),
            "selected_spatial_layers": np.asarray(
                [int(row["layer"]) for row in selected_heads],
                dtype=np.int16,
            ),
            "selected_spatial_heads": np.asarray(
                [int(row["head"]) for row in selected_heads],
                dtype=np.int16,
            ),
            "relation_readout_vectors": (
                relation_vectors.numpy().astype(np.float32)
            ),
        })
        np.savez_compressed(
            output_dir / "directional_arrays.npz",
            **stacked,
        )

        # Unique relation-stratified matching with token-count controls.
        match_features = parse_csv_list(args.match_features)
        pairs = build_unique_pairs(
            successful_metadata,
            relations=allowed_relations,
            group_correct=base.GROUP_CORRECT,
            group_wrong=base.GROUP_WRONG,
            feature_names=match_features,
            caliper=args.match_caliper,
        )
        base.write_csv(output_dir / "matched_pairs_unique.csv", pairs)
        base.write_csv(
            output_dir / "matching_balance.csv",
            matching_balance_rows(
                successful_metadata,
                pairs,
                match_features,
            ),
        )

        routing_group_rows, routing_paired_rows = summarize_routing(
            stacked=stacked,
            routing_layers=routing_layers,
            n_heads=n_heads,
            group_codes=group_codes,
            relation_codes=relation_codes,
            relations=allowed_relations,
            relation_to_index=relation_to_index,
            pairs=pairs,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        base.write_csv(
            output_dir / "routing_directional_group_comparison.csv",
            routing_group_rows,
        )
        base.write_csv(
            output_dir / "routing_directional_matched_comparison.csv",
            routing_paired_rows,
        )

        spatial_group_rows, spatial_paired_rows = summarize_spatial(
            stacked=stacked,
            selected_heads=selected_heads,
            group_codes=group_codes,
            relation_codes=relation_codes,
            relations=allowed_relations,
            relation_to_index=relation_to_index,
            pairs=pairs,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        base.write_csv(
            output_dir / "spatial_directional_group_comparison.csv",
            spatial_group_rows,
        )
        base.write_csv(
            output_dir / "spatial_directional_matched_comparison.csv",
            spatial_paired_rows,
        )

        base.write_csv(
            output_dir / "sample_directional_summary.csv",
            sample_summary_rows(
                successful_metadata,
                stacked,
                routing_layers,
            ),
        )

        # Compact head ranking for the two most diagnostic metrics.
        ranking_rows = [
            row
            for row in routing_paired_rows
            if row["metric"] in (
                "routing_combined_loo_gt_opp_gain",
                "routing_role_contrast_linear_gt_opp",
            )
        ]
        ranking_rows.sort(
            key=lambda row: (
                -abs(float(row["paired_cohen_dz"]))
                if np.isfinite(float(row["paired_cohen_dz"]))
                else float("inf")
            )
        )
        base.write_csv(
            output_dir / "directional_head_ranking.csv",
            ranking_rows,
        )

        projection_summary = {
            "routing_finite_fraction": float(np.mean(
                np.isfinite(
                    stacked["routing_combined_loo_gt_opp_gain"][
                        :, routing_layers, :
                    ]
                )
            )),
            "spatial_finite_fraction": float(np.mean(
                np.isfinite(
                    stacked["spatial_delta_loo_gt_opp_gain"]
                )
            )),
        }

        summary = {
            "script_version": SCRIPT_VERSION,
            "base_module": base.__name__,
            "base_script_version": getattr(
                base, "SCRIPT_VERSION", None
            ),
            "trace_dir": str(trace_dir),
            "dataset": dataset,
            "model": model_name,
            "n_requested": len(metadata_rows),
            "n_successful": len(successful_metadata),
            "n_errors": len(metadata_rows) - len(successful_metadata),
            "group_counts": dict(Counter(
                row["group"] for row in successful_metadata
            )),
            "relation_counts": dict(Counter(
                row["gt"] for row in successful_metadata
            )),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "routing_layers": routing_layers,
            "selected_spatial_heads": selected_heads,
            "matched_pair_count_unique": len(pairs),
            "match_features": match_features,
            "match_caliper": args.match_caliper,
            "projection_coverage": projection_summary,
            "decoder_layers_path": layers_path,
            "final_norm_path": final_norm_path,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "interpretation": {
                "positive": (
                    "The isolated contribution supports GT over the "
                    "opposite relation."
                ),
                "negative": (
                    "The isolated contribution supports the opposite "
                    "relation over GT."
                ),
                "primary_routing_metric": (
                    "routing_combined_loo_gt_opp_gain"
                ),
                "primary_role_metric": (
                    "routing_role_contrast_linear_gt_opp"
                ),
                "primary_spatial_metric": (
                    "spatial_delta_loo_gt_opp_gain"
                ),
            },
            "audit": audit,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 104)
        print("CENTROID DIRECTIONAL CONTRIBUTION TRACE")
        print("=" * 104)
        print(
            f"successful: {len(successful_metadata)} / "
            f"{len(metadata_rows)}"
        )
        print(f"unique matched pairs: {len(pairs)}")
        print(
            "coverage: "
            f"routing={projection_summary['routing_finite_fraction']:.4f}, "
            f"spatial={projection_summary['spatial_finite_fraction']:.4f}"
        )

        print("\nTop matched directional routing effects:")
        for row in ranking_rows[:15]:
            print(
                f"  {row['metric']:42s} "
                f"L{int(row['layer']):02d}H{int(row['head']):02d} | "
                f"correct-wrong={float(row['mean_paired_difference']):+.5f} | "
                f"dz={float(row['paired_cohen_dz']):+.3f} | "
                f"CI=[{float(row['bootstrap_ci_low']):+.5f}, "
                f"{float(row['bootstrap_ci_high']):+.5f}]"
            )

        print("\nSaved outputs:")
        for filename in [
            "projection_diagnostics.json",
            "sample_metadata.jsonl",
            "directional_arrays.npz",
            "matched_pairs_unique.csv",
            "matching_balance.csv",
            "routing_directional_group_comparison.csv",
            "routing_directional_matched_comparison.csv",
            "spatial_directional_group_comparison.csv",
            "spatial_directional_matched_comparison.csv",
            "sample_directional_summary.csv",
            "directional_head_ranking.csv",
            "summary.json",
        ]:
            print(f"  {output_dir / filename}")
        if errors_path.exists():
            print(f"  {errors_path}")

    finally:
        object_output_collector.close()
        collector.close()
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
