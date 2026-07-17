#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose where two-object spatial information is lost inside a frozen VLM on COCO-two.

This is a *diagnostic* script. It does not modify the model.

The script tests the following label-free internal hypothesis:

1. Pair formation
   The two object tokens should exchange information. Because decoder attention
   is causal, one prompt exposes only the later-object -> earlier-object edge.
   Running the swapped prompt exposes the reverse semantic edge.

2. Decision routing
   The prompt-last token, whose hidden state produces the first answer logits,
   should read both object tokens rather than only one of them.

3. Role antisymmetry / preservation
   If the original prompt is "A relative to B" and the swapped prompt is
   "B relative to A", the role-ordered object-state difference should reverse:

       (h_A - h_B)_orig  ~=  -(h_B - h_A)_swap

Ground-truth labels are used only *after extraction* to compare normal-generation
correct and wrong samples. They never enter the attention/hidden-state metrics.

The first version intentionally measures attention weights plus hidden-state
role antisymmetry. It does not yet reconstruct per-head A x V x W_O outputs.
Use it to decide whether pair/routing failure is a plausible mechanism before
implementing an invasive repair.

Expected repository context
---------------------------
Place this file next to:

    analyze_coco_centroid_generation_step1_v4.py

The existing script is imported as a helper module for model/data loading,
prompt construction, token-span location, and attention normalization.

Main outputs
------------
- samples.jsonl
- sample_arrays/<sid>.npz
- per_layer.csv
- top_diagnostics.csv
- failure_groups.csv
- summary.json
- report.txt

Example
-------
CUDA_VISIBLE_DEVICES=0 python3 analyze_object_pair_spatial_failure_v1.py \
  --model qwen-3b \
  --layers all \
  --max-samples 40 \
  --output-dir output/object_pair_failure_smoke/qwen-3b \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-object-pair-spatial-failure-v1"
METRIC_FAMILIES = {
    "pair": (
        "pair_strength_mean",
        "pair_strength_top",
        "pair_balance_top",
    ),
    "routing": (
        "routing_sum_mean",
        "routing_sum_top",
        "routing_balance_mean",
        "routing_balance_top",
        "routing_swap_consistency",
    ),
    "representation": (
        "hidden_role_antisymmetry",
        "carrier_role_antisymmetry",
    ),
}
EXPECTED_HIGHER_IS_BETTER = {
    name: True
    for names in METRIC_FAMILIES.values()
    for name in names
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
        help="Existing COCO centroid script imported for helper functions.",
    )
    parser.add_argument(
        "--dataset",
        default="coco_two",
        choices=["coco_two"],
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument("--model", default="qwen-3b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
        help="Use eager for complete attention probabilities.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated zero-based decoder layers or 'all'.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Upper bound for normal greedy generation; EOS may stop earlier.",
    )
    parser.add_argument(
        "--top-head-fraction",
        type=float,
        default=0.25,
        help=(
            "Within each sample/layer, aggregate the strongest fraction of heads "
            "for pair and routing measures. This selection is label-free."
        ),
    )
    parser.add_argument(
        "--strong-heads",
        default="",
        help=(
            "Optional comma-separated L:H list, e.g. 22:0,24:5,28:8. "
            "These heads are saved as auxiliary metrics but do not determine labels."
        ),
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_python_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing helper script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_head_list(value: str) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    for raw in str(value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(
                f"Invalid --strong-heads item {raw!r}; expected layer:head"
            )
        layer_text, head_text = raw.split(":", 1)
        layer = int(layer_text)
        head = int(head_text)
        result.setdefault(layer, [])
        if head not in result[layer]:
            result[layer].append(head)
    for heads in result.values():
        heads.sort()
    return result


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def finite_mean(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result.astype(np.float32)


def balance_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (
        2.0 * np.minimum(a, b) / (a + b + 1e-12)
    ).astype(np.float32)


def top_fraction_mean(
    values: np.ndarray,
    fraction: float,
    *,
    return_indices: bool = False,
) -> Any:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        if return_indices:
            return float("nan"), np.asarray([], dtype=np.int64)
        return float("nan")
    k = max(1, int(math.ceil(float(fraction) * finite.size)))
    order = finite[np.argsort(values[finite])[-k:]]
    mean = float(np.mean(values[order]))
    if return_indices:
        return mean, order.astype(np.int64)
    return mean


def normalize_relation(base: Any, value: Any) -> Optional[str]:
    relation = base.normalize_relation(value)
    return str(relation) if relation is not None else None


def unwrap_layer_tuple(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        value = tuple(value)
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return tuple(value[0])
        return value
    raise TypeError(f"Expected layer tuple, got {type(value).__name__}")


def attention_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(getattr(outputs, "language_model_outputs", None), "attentions", None),
        getattr(getattr(outputs, "text_model_output", None), "attentions", None),
    ]
    for value in candidates:
        layers = unwrap_layer_tuple(value)
        if layers and torch.is_tensor(layers[-1]):
            return tuple(layers)
    raise RuntimeError(
        "No decoder attention tensors returned. Use --attn-impl eager and "
        "a transformers/model version that exposes output_attentions."
    )


def input_ids_list(batch: Mapping[str, Any]) -> List[int]:
    ids = batch["input_ids"]
    if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"Unexpected input_ids shape: {getattr(ids, 'shape', None)}")
    return [int(x) for x in ids[0].detach().cpu().tolist()]


def later_to_earlier_edge(
    attention: torch.Tensor,
    first_index: int,
    second_index: int,
) -> Tuple[np.ndarray, str]:
    """Return the only directly available causal edge between two token positions."""
    if first_index == second_index:
        raise ValueError("Object token indices are identical")
    if first_index > second_index:
        query_index, key_index = first_index, second_index
        direction = "first<-second"
    else:
        query_index, key_index = second_index, first_index
        direction = "second<-first"
    values = attention[:, query_index, key_index]
    return values.detach().float().cpu().numpy().astype(np.float32), direction


def forward_trace(
    *,
    base: Any,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
) -> Dict[str, Any]:
    ids = input_ids_list(batch)
    input_length = len(ids)
    subject_span, reference_span = base.locate_object_spans(
        processor.tokenizer,
        ids,
        subject,
        reference,
    )
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])
    prompt_last_index = input_length - 1

    with torch.inference_mode():
        output = model(
            **batch,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = attention_tuple(output)
    if hasattr(base, "hidden_tuple"):
        hidden_states = tuple(base.hidden_tuple(output))
    else:
        hidden_states = unwrap_layer_tuple(getattr(output, "hidden_states", None))
    if not hidden_states:
        raise RuntimeError("Forward output did not include decoder hidden states")

    n_selected = len(selected_layers)
    first_attention = base.normalize_attention_tensor(
        attentions[selected_layers[0]],
        expected_query_length=input_length,
    )
    n_heads = int(first_attention.shape[0])

    pair_edge = np.full((n_selected, n_heads), np.nan, dtype=np.float32)
    prompt_to_subject = np.full_like(pair_edge, np.nan)
    prompt_to_reference = np.full_like(pair_edge, np.nan)
    subject_hidden = []
    reference_hidden = []
    prompt_hidden = []
    pair_direction = ""

    for out_index, layer in enumerate(selected_layers):
        attention = base.normalize_attention_tensor(
            attentions[layer],
            expected_query_length=input_length,
        )
        if int(attention.shape[0]) != n_heads:
            raise RuntimeError(
                f"Head count changed at L{layer}: {attention.shape[0]} vs {n_heads}"
            )
        edge, direction = later_to_earlier_edge(
            attention,
            subject_index,
            reference_index,
        )
        pair_edge[out_index] = edge
        pair_direction = direction
        prompt_to_subject[out_index] = (
            attention[:, prompt_last_index, subject_index]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        prompt_to_reference[out_index] = (
            attention[:, prompt_last_index, reference_index]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        # hidden_states[0] is the embedding input; hidden_states[layer + 1]
        # is the residual stream after decoder layer `layer`.
        hidden = hidden_states[layer + 1][0].detach().float().cpu()
        subject_hidden.append(hidden[subject_index].numpy())
        reference_hidden.append(hidden[reference_index].numpy())
        prompt_hidden.append(hidden[prompt_last_index].numpy())

    del output
    return {
        "input_length": input_length,
        "subject_index": subject_index,
        "reference_index": reference_index,
        "prompt_last_index": prompt_last_index,
        "n_heads": n_heads,
        "pair_direction": pair_direction,
        "pair_edge": pair_edge,
        "prompt_to_subject": prompt_to_subject,
        "prompt_to_reference": prompt_to_reference,
        "subject_hidden": np.asarray(subject_hidden, dtype=np.float32),
        "reference_hidden": np.asarray(reference_hidden, dtype=np.float32),
        "prompt_hidden": np.asarray(prompt_hidden, dtype=np.float32),
    }


def build_metrics(
    *,
    original: Mapping[str, Any],
    swapped: Mapping[str, Any],
    top_fraction: float,
    selected_layers: Sequence[int],
    strong_heads: Mapping[int, Sequence[int]],
) -> Dict[str, np.ndarray]:
    if int(original["n_heads"]) != int(swapped["n_heads"]):
        raise RuntimeError("Original/swap head count mismatch")

    orig_pair = np.asarray(original["pair_edge"], dtype=np.float32)
    swap_pair = np.asarray(swapped["pair_edge"], dtype=np.float32)
    pair_mean = 0.5 * (orig_pair + swap_pair)
    pair_balance = balance_score(orig_pair, swap_pair)

    # Original role order is A(subject), B(reference).
    orig_to_a = np.asarray(original["prompt_to_subject"], dtype=np.float32)
    orig_to_b = np.asarray(original["prompt_to_reference"], dtype=np.float32)

    # Swapped role order is B(subject), A(reference). Align to semantic A/B.
    swap_to_a = np.asarray(swapped["prompt_to_reference"], dtype=np.float32)
    swap_to_b = np.asarray(swapped["prompt_to_subject"], dtype=np.float32)

    routing_sum_orig = orig_to_a + orig_to_b
    routing_sum_swap = swap_to_a + swap_to_b
    routing_sum_mean = 0.5 * (routing_sum_orig + routing_sum_swap)
    routing_balance_orig = balance_score(orig_to_a, orig_to_b)
    routing_balance_swap = balance_score(swap_to_a, swap_to_b)
    routing_balance_mean = 0.5 * (
        routing_balance_orig + routing_balance_swap
    )

    # Per-head semantic A/B routing consistency across original and swap.
    routing_vectors_orig = np.stack([orig_to_a, orig_to_b], axis=-1)
    routing_vectors_swap = np.stack([swap_to_a, swap_to_b], axis=-1)
    routing_swap_consistency_head = cosine_rows(
        routing_vectors_orig,
        routing_vectors_swap,
    )

    orig_delta = (
        np.asarray(original["subject_hidden"], dtype=np.float32)
        - np.asarray(original["reference_hidden"], dtype=np.float32)
    )
    swap_role_delta = (
        np.asarray(swapped["subject_hidden"], dtype=np.float32)
        - np.asarray(swapped["reference_hidden"], dtype=np.float32)
    )
    hidden_role_antisymmetry = cosine_rows(orig_delta, -swap_role_delta)

    # A lightweight carrier proxy: prompt-last attention weights multiply the
    # corresponding object residual states. This is not exact A x V x W_O.
    orig_carrier = (
        finite_mean(orig_to_a, axis=1)[:, None]
        * np.asarray(original["subject_hidden"], dtype=np.float32)
        - finite_mean(orig_to_b, axis=1)[:, None]
        * np.asarray(original["reference_hidden"], dtype=np.float32)
    )
    swap_role_carrier = (
        finite_mean(np.asarray(swapped["prompt_to_subject"]), axis=1)[:, None]
        * np.asarray(swapped["subject_hidden"], dtype=np.float32)
        - finite_mean(np.asarray(swapped["prompt_to_reference"]), axis=1)[:, None]
        * np.asarray(swapped["reference_hidden"], dtype=np.float32)
    )
    carrier_role_antisymmetry = cosine_rows(
        orig_carrier,
        -swap_role_carrier,
    )

    n_layers = len(selected_layers)
    layer_metrics: Dict[str, np.ndarray] = {
        "pair_strength_mean": finite_mean(pair_mean, axis=1).astype(np.float32),
        "pair_balance_mean": finite_mean(pair_balance, axis=1).astype(np.float32),
        "routing_sum_mean": finite_mean(routing_sum_mean, axis=1).astype(np.float32),
        "routing_balance_mean": finite_mean(
            routing_balance_mean,
            axis=1,
        ).astype(np.float32),
        "routing_swap_consistency": finite_mean(
            routing_swap_consistency_head,
            axis=1,
        ).astype(np.float32),
        "hidden_role_antisymmetry": hidden_role_antisymmetry.astype(np.float32),
        "carrier_role_antisymmetry": carrier_role_antisymmetry.astype(np.float32),
    }

    pair_top = np.full(n_layers, np.nan, dtype=np.float32)
    pair_balance_top = np.full(n_layers, np.nan, dtype=np.float32)
    routing_top = np.full(n_layers, np.nan, dtype=np.float32)
    routing_balance_top = np.full(n_layers, np.nan, dtype=np.float32)
    strong_pair = np.full(n_layers, np.nan, dtype=np.float32)
    strong_routing = np.full(n_layers, np.nan, dtype=np.float32)

    for layer_pos, layer_id in enumerate(selected_layers):
        pair_top[layer_pos], pair_indices = top_fraction_mean(
            pair_mean[layer_pos],
            top_fraction,
            return_indices=True,
        )
        if pair_indices.size:
            pair_balance_top[layer_pos] = float(
                np.nanmean(pair_balance[layer_pos, pair_indices])
            )

        routing_top[layer_pos], routing_indices = top_fraction_mean(
            routing_sum_mean[layer_pos],
            top_fraction,
            return_indices=True,
        )
        if routing_indices.size:
            routing_balance_top[layer_pos] = float(
                np.nanmean(routing_balance_mean[layer_pos, routing_indices])
            )

        selected = [
            head
            for head in strong_heads.get(int(layer_id), [])
            if 0 <= int(head) < pair_mean.shape[1]
        ]
        if selected:
            strong_pair[layer_pos] = float(
                np.nanmean(pair_mean[layer_pos, selected])
            )
            strong_routing[layer_pos] = float(
                np.nanmean(routing_sum_mean[layer_pos, selected])
            )

    layer_metrics["pair_strength_top"] = pair_top
    layer_metrics["pair_balance_top"] = pair_balance_top
    layer_metrics["routing_sum_top"] = routing_top
    layer_metrics["routing_balance_top"] = routing_balance_top
    layer_metrics["strong_head_pair_strength"] = strong_pair
    layer_metrics["strong_head_routing_sum"] = strong_routing

    arrays = {
        "original_pair_edge": orig_pair,
        "swapped_pair_edge": swap_pair,
        "pair_edge_mean": pair_mean,
        "pair_edge_balance": pair_balance,
        "original_prompt_to_a": orig_to_a,
        "original_prompt_to_b": orig_to_b,
        "swapped_prompt_to_a_aligned": swap_to_a,
        "swapped_prompt_to_b_aligned": swap_to_b,
        "routing_sum_mean_heads": routing_sum_mean,
        "routing_balance_mean_heads": routing_balance_mean,
        "routing_swap_consistency_head": routing_swap_consistency_head,
        "hidden_role_delta_original": orig_delta,
        "hidden_role_delta_swapped": swap_role_delta,
        **layer_metrics,
    }
    return arrays


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[valid]
    labels = labels[valid]
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def cohen_d(correct: np.ndarray, wrong: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=np.float64)
    wrong = np.asarray(wrong, dtype=np.float64)
    correct = correct[np.isfinite(correct)]
    wrong = wrong[np.isfinite(wrong)]
    if correct.size < 2 or wrong.size < 2:
        return float("nan")
    var_correct = float(np.var(correct, ddof=1))
    var_wrong = float(np.var(wrong, ddof=1))
    pooled_num = (correct.size - 1) * var_correct + (wrong.size - 1) * var_wrong
    pooled_den = correct.size + wrong.size - 2
    if pooled_den <= 0:
        return float("nan")
    pooled = math.sqrt(max(0.0, pooled_num / pooled_den))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(correct) - np.mean(wrong)) / pooled)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    tpr = float(np.mean(predictions[positive] == 1))
    tnr = float(np.mean(predictions[negative] == 0))
    return 0.5 * (tpr + tnr)


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = np.sort(np.unique(scores[np.isfinite(scores)]))
    if finite.size == 0:
        return np.asarray([], dtype=np.float64)
    if finite.size == 1:
        return finite.copy()
    mids = 0.5 * (finite[:-1] + finite[1:])
    return np.concatenate([
        np.asarray([finite[0] - 1e-9]),
        mids,
        np.asarray([finite[-1] + 1e-9]),
    ])


def best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    higher_is_better: bool,
) -> Tuple[float, float]:
    best_value = float("nan")
    best_score = -float("inf")
    for threshold in candidate_thresholds(scores):
        prediction = (
            scores >= threshold
            if higher_is_better
            else scores <= threshold
        ).astype(np.int64)
        score = balanced_accuracy(labels, prediction)
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_value = float(threshold)
    return best_value, best_score


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    fold_parts: List[List[int]] = [[] for _ in range(n_folds)]
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold_parts[position % n_folds].append(int(index))
    return [np.asarray(sorted(part), dtype=np.int64) for part in fold_parts]


def cross_validated_threshold_ba(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    higher_is_better: bool,
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[valid]
    labels = labels[valid]
    class_counts = [int(np.sum(labels == label)) for label in (0, 1)]
    possible_folds = min([n_folds] + class_counts)
    if possible_folds < 2:
        return float("nan")
    folds = stratified_folds(labels, possible_folds, seed)
    all_predictions = np.full(labels.shape, -1, dtype=np.int64)
    all_indices = np.arange(labels.size)
    for test_indices in folds:
        if test_indices.size == 0:
            continue
        train_mask = np.ones(labels.size, dtype=bool)
        train_mask[test_indices] = False
        threshold, _ = best_threshold(
            scores[train_mask],
            labels[train_mask],
            higher_is_better,
        )
        if not math.isfinite(threshold):
            continue
        all_predictions[test_indices] = (
            scores[test_indices] >= threshold
            if higher_is_better
            else scores[test_indices] <= threshold
        ).astype(np.int64)
    evaluated = all_predictions >= 0
    if not evaluated.any():
        return float("nan")
    return balanced_accuracy(labels[evaluated], all_predictions[evaluated])


def aggregate_diagnostics(
    sample_rows: Sequence[Mapping[str, Any]],
    selected_layers: Sequence[int],
    cv_folds: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = np.asarray([
        int(bool(row["generation_correct"]))
        for row in sample_rows
    ], dtype=np.int64)
    per_layer_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []

    for metric_name in EXPECTED_HIGHER_IS_BETTER:
        matrix = np.asarray([
            row["layer_metrics"][metric_name]
            for row in sample_rows
        ], dtype=np.float64)
        for layer_pos, layer_id in enumerate(selected_layers):
            scores = matrix[:, layer_pos]
            correct = scores[labels == 1]
            wrong = scores[labels == 0]
            auc = rank_auc(scores, labels)
            effect = cohen_d(correct, wrong)
            cv_ba = cross_validated_threshold_ba(
                scores,
                labels,
                n_folds=cv_folds,
                seed=seed + layer_pos,
                higher_is_better=EXPECTED_HIGHER_IS_BETTER[metric_name],
            )
            row = {
                "metric": metric_name,
                "family": next(
                    family
                    for family, names in METRIC_FAMILIES.items()
                    if metric_name in names
                ),
                "layer": int(layer_id),
                "n_valid": int(np.sum(np.isfinite(scores))),
                "correct_mean": safe_float(np.nanmean(correct)),
                "wrong_mean": safe_float(np.nanmean(wrong)),
                "correct_median": safe_float(np.nanmedian(correct)),
                "wrong_median": safe_float(np.nanmedian(wrong)),
                "difference_correct_minus_wrong": safe_float(
                    np.nanmean(correct) - np.nanmean(wrong)
                ),
                "cohen_d": safe_float(effect),
                "auc_correct": safe_float(auc),
                "cv_balanced_accuracy": safe_float(cv_ba),
            }
            per_layer_rows.append(row)

    for metric_name in EXPECTED_HIGHER_IS_BETTER:
        candidates = [
            row for row in per_layer_rows
            if row["metric"] == metric_name
            and row["cv_balanced_accuracy"] is not None
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: (
                row["cv_balanced_accuracy"],
                row["auc_correct"] if row["auc_correct"] is not None else -1,
            ),
        )
        top_rows.append(dict(best))

    top_rows.sort(
        key=lambda row: (
            -(row["cv_balanced_accuracy"] or -1),
            -(row["auc_correct"] or -1),
        )
    )
    return per_layer_rows, top_rows


def choose_family_best(top_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for family in METRIC_FAMILIES:
        rows = [
            row for row in top_rows
            if row.get("family") == family
            and row.get("cv_balanced_accuracy") is not None
        ]
        if rows:
            result[family] = max(
                rows,
                key=lambda row: row["cv_balanced_accuracy"],
            )
    return result


def correct_quantile_threshold(
    sample_rows: Sequence[Mapping[str, Any]],
    metric: str,
    layer_position: int,
    quantile: float = 0.10,
) -> float:
    values = np.asarray([
        row["layer_metrics"][metric][layer_position]
        for row in sample_rows
        if bool(row["generation_correct"])
    ], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, quantile))


def assign_failure_groups(
    sample_rows: Sequence[Mapping[str, Any]],
    selected_layers: Sequence[int],
    family_best: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    layer_to_position = {
        int(layer): index for index, layer in enumerate(selected_layers)
    }
    thresholds: Dict[str, Dict[str, Any]] = {}
    for family, row in family_best.items():
        metric = str(row["metric"])
        layer = int(row["layer"])
        pos = layer_to_position[layer]
        thresholds[family] = {
            "metric": metric,
            "layer": layer,
            "position": pos,
            "q10_correct": correct_quantile_threshold(
                sample_rows,
                metric,
                pos,
                quantile=0.10,
            ),
        }

    # Late overwrite uses the representation metric chosen by CV. The sample
    # must have had a clearly stronger earlier antisymmetry than at the last
    # selected layer.
    representation = thresholds.get("representation")
    group_rows: List[Dict[str, Any]] = []

    for row in sample_rows:
        if bool(row["generation_correct"]):
            group = "generation_correct"
            reason = "baseline generation is correct"
        else:
            pair_low = False
            routing_low = False
            late_overwrite = False
            reasons: List[str] = []

            pair_info = thresholds.get("pair")
            if pair_info and math.isfinite(pair_info["q10_correct"]):
                value = row["layer_metrics"][pair_info["metric"]][
                    pair_info["position"]
                ]
                pair_low = bool(
                    math.isfinite(float(value))
                    and float(value) < pair_info["q10_correct"]
                )
                if pair_low:
                    reasons.append(
                        f"{pair_info['metric']}@L{pair_info['layer']} below correct-q10"
                    )

            routing_info = thresholds.get("routing")
            if routing_info and math.isfinite(routing_info["q10_correct"]):
                value = row["layer_metrics"][routing_info["metric"]][
                    routing_info["position"]
                ]
                routing_low = bool(
                    math.isfinite(float(value))
                    and float(value) < routing_info["q10_correct"]
                )
                if routing_low:
                    reasons.append(
                        f"{routing_info['metric']}@L{routing_info['layer']} below correct-q10"
                    )

            if representation:
                values = np.asarray(
                    row["layer_metrics"][representation["metric"]],
                    dtype=np.float64,
                )
                finite = np.flatnonzero(np.isfinite(values))
                if finite.size >= 2:
                    peak_pos = int(finite[np.argmax(values[finite])])
                    final_pos = int(finite[-1])
                    drop = float(values[peak_pos] - values[final_pos])
                    late_overwrite = bool(
                        peak_pos < final_pos
                        and drop >= 0.25
                        and values[peak_pos] >= 0.50
                    )
                    if late_overwrite:
                        reasons.append(
                            f"{representation['metric']} drops {drop:.3f} "
                            f"from L{selected_layers[peak_pos]} to L{selected_layers[final_pos]}"
                        )

            if pair_low:
                group = "pair_formation_missing"
            elif routing_low:
                group = "decision_routing_missing"
            elif late_overwrite:
                group = "late_relation_overwrite"
            else:
                group = "unclassified_wrong"
            reason = "; ".join(reasons) if reasons else "no diagnostic rule triggered"

        group_rows.append({
            "sid": row["sid"],
            "gt": row["gt"],
            "generation_prediction": row["generation_prediction"],
            "generation_text": row["generation_text"],
            "generation_correct": row["generation_correct"],
            "failure_group": group,
            "reason": reason,
        })
    return group_rows


def verdict(top_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    family_best = choose_family_best(top_rows)
    scores = {
        family: safe_float(row.get("cv_balanced_accuracy"))
        for family, row in family_best.items()
    }
    strong_families = [
        family for family, score in scores.items()
        if score is not None and score >= 0.57
    ]
    best_score = max(
        [score for score in scores.values() if score is not None] or [float("nan")]
    )
    if math.isfinite(best_score) and best_score >= 0.62 and len(strong_families) >= 2:
        label = "SUPPORTED"
        explanation = (
            "At least two mechanism families separate correct and wrong samples, "
            "and the best single diagnostic reaches CV balanced accuracy >= 0.62."
        )
    elif math.isfinite(best_score) and best_score >= 0.57:
        label = "WEAK_SUPPORT"
        explanation = (
            "At least one internal metric carries a repeatable signal, but the "
            "evidence is not yet strong enough to justify a repair mechanism."
        )
    else:
        label = "NOT_SUPPORTED"
        explanation = (
            "Pair/routing/antisymmetry metrics do not reliably distinguish "
            "generation-correct and generation-wrong samples in this run."
        )
    return {
        "label": label,
        "explanation": explanation,
        "best_cv_balanced_accuracy": safe_float(best_score),
        "family_best_cv_balanced_accuracy": scores,
        "strong_families": strong_families,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (0.0 < args.top_head_fraction <= 1.0):
        raise ValueError("--top-head-fraction must be in (0, 1]")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_path = Path(args.base_script)
    base = import_python_file(base_path, "_coco_centroid_base")
    strong_heads = parse_head_list(args.strong_heads)

    prompt_path = Path(args.prompt_jsonl)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt JSONL: {prompt_path}")

    two_object_module = base.import_two_object_module()
    records, audit = two_object_module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    if not records:
        raise RuntimeError("No usable COCO-two records")

    prompt_rows = base.load_standard_prompts(prompt_path)
    missing_ids = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing_ids:
        raise RuntimeError(
            f"Prompt file {prompt_path} is missing {len(missing_ids)} record IDs; "
            f"first={missing_ids[:10]}"
        )

    specs = base.merged_model_specs(two_object_module)
    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model!r}; available={sorted(specs)}"
        )
    model_spec = specs[args.model]
    model_cls = getattr(transformers, model_spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no "
            f"{model_spec.model_class}"
        )

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "sample_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    errors_path = output_dir / "errors.jsonl"

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Loading {args.model}: {model_spec.repo_id}")
    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(model_spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": model_spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    model = model_cls.from_pretrained(
        model_spec.repo_id,
        **load_kwargs,
    )
    model.eval()
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    processor = AutoProcessor.from_pretrained(
        model_spec.repo_id,
        trust_remote_code=model_spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    selected_layers = base.parse_layers(args.layers, len(decoder_layers))
    for layer, heads in strong_heads.items():
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(f"Strong-head layer L{layer} outside decoder range")
        # Head range is checked after the first sample, when n_heads is known.

    config = {
        "script_version": SCRIPT_VERSION,
        "base_script": str(base_path),
        "model": args.model,
        "repo_id": model_spec.repo_id,
        "transformers_version": transformers.__version__,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "selected_layers": selected_layers,
        "top_head_fraction": args.top_head_fraction,
        "strong_heads": strong_heads,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "prompt_jsonl": str(prompt_path),
        "attn_impl": args.attn_impl,
        "n_records": len(records),
        "audit": audit,
        "normal_generation": True,
        "forces_minimum_generation_tokens": False,
        "uses_gt_for_metric_extraction": False,
        "uses_gt_for_posthoc_diagnostic_evaluation": True,
        "modifies_model": False,
        "attention_note": (
            "Pair and routing measures use attention weights. Hidden role "
            "antisymmetry is also measured. Exact per-head A×V×W_O is not "
            "reconstructed in v1."
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sample_rows: List[Dict[str, Any]] = []
    start = time.time()
    completed = 0

    try:
        for record in tqdm(records, desc=f"pair-failure:{args.dataset}:{args.model}"):
            sid = int(record.sid)
            original_batch = None
            swapped_batch = None
            image = None
            try:
                prompt_row = prompt_rows[sid]
                question_text = str(prompt_row["question_text"])
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                gt = normalize_relation(base, prompt_row["answer_raw"])
                if gt not in base.RELATIONS:
                    raise ValueError(f"Unsupported GT for sid={sid}: {gt!r}")

                image = base.record_image(record)
                original_batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question_text,
                    device=device,
                )
                swapped_question = base.build_swapped_question(subject, reference)
                swapped_batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=swapped_question,
                    device=device,
                )

                generation_text = base.generate_text(
                    model,
                    processor,
                    original_batch,
                    args.max_new_tokens,
                )
                generation_prediction = normalize_relation(base, generation_text)
                generation_correct = bool(generation_prediction == gt)

                original = forward_trace(
                    base=base,
                    model=model,
                    processor=processor,
                    batch=original_batch,
                    subject=subject,
                    reference=reference,
                    selected_layers=selected_layers,
                )
                swapped = forward_trace(
                    base=base,
                    model=model,
                    processor=processor,
                    batch=swapped_batch,
                    subject=reference,
                    reference=subject,
                    selected_layers=selected_layers,
                )
                metrics = build_metrics(
                    original=original,
                    swapped=swapped,
                    top_fraction=args.top_head_fraction,
                    selected_layers=selected_layers,
                    strong_heads=strong_heads,
                )

                if completed == 0:
                    n_heads = int(original["n_heads"])
                    for layer, heads in strong_heads.items():
                        invalid = [head for head in heads if head < 0 or head >= n_heads]
                        if invalid:
                            raise ValueError(
                                f"Strong heads {invalid} invalid for model with {n_heads} heads"
                            )

                array_path = arrays_dir / f"{sid}.npz"
                np.savez_compressed(
                    array_path,
                    layer_indices=np.asarray(selected_layers, dtype=np.int16),
                    **metrics,
                )

                layer_metrics = {
                    name: np.asarray(metrics[name], dtype=np.float32).tolist()
                    for name in EXPECTED_HIGHER_IS_BETTER
                }
                row = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "question": question_text,
                    "swapped_question": swapped_question,
                    "gt": gt,
                    "generation_text": generation_text,
                    "generation_prediction": generation_prediction,
                    "generation_correct": generation_correct,
                    "original_pair_direction": original["pair_direction"],
                    "swapped_pair_direction": swapped["pair_direction"],
                    "layer_metrics": layer_metrics,
                    "array_file": str(array_path),
                }
                append_jsonl(samples_path, row)
                sample_rows.append(row)
                completed += 1

                if args.print_every > 0 and completed % args.print_every == 0:
                    running = float(np.mean([
                        bool(item["generation_correct"]) for item in sample_rows
                    ]))
                    best_pair = float(np.nanmax(metrics["pair_strength_top"]))
                    best_route = float(np.nanmax(metrics["routing_sum_top"]))
                    best_anti = float(np.nanmax(metrics["hidden_role_antisymmetry"]))
                    tqdm.write(
                        f"\n[{completed}/{len(records)}] sid={sid} "
                        f"pred={generation_prediction or '<invalid>'} gt={gt} "
                        f"correct={generation_correct} running_acc={running:.4f}\n"
                        f"  max_pair_top={best_pair:.6f} "
                        f"max_routing_top={best_route:.6f} "
                        f"max_hidden_antisym={best_anti:.4f}"
                    )

                del original, swapped, metrics
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
                tqdm.write(f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if original_batch is not None:
                    del original_batch
                if swapped_batch is not None:
                    del swapped_batch
                if image is not None:
                    del image

        if not sample_rows:
            raise RuntimeError("No samples completed; inspect errors.jsonl")

        per_layer_rows, top_rows = aggregate_diagnostics(
            sample_rows,
            selected_layers,
            args.cv_folds,
            args.seed,
        )
        family_best = choose_family_best(top_rows)
        group_rows = assign_failure_groups(
            sample_rows,
            selected_layers,
            family_best,
        )
        verdict_data = verdict(top_rows)

        generation_accuracy = float(np.mean([
            bool(row["generation_correct"]) for row in sample_rows
        ]))
        group_counts = Counter(row["failure_group"] for row in group_rows)

        summary = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "n_samples": len(sample_rows),
            "generation_accuracy": generation_accuracy,
            "n_correct": int(sum(bool(row["generation_correct"]) for row in sample_rows)),
            "n_wrong": int(sum(not bool(row["generation_correct"]) for row in sample_rows)),
            "selected_layers": selected_layers,
            "top_head_fraction": args.top_head_fraction,
            "family_best": family_best,
            "failure_group_counts": dict(group_counts),
            "verdict": verdict_data,
            "elapsed_minutes": (time.time() - start) / 60.0,
            "limitations": [
                "The v1 pair/routing measures use attention probabilities, not exact A×V×W_O contributions.",
                "Correct-vs-wrong comparisons and layer selection are post-hoc diagnostics and use generation correctness labels.",
                "A positive diagnostic result does not yet prove that repairing the edge will improve generation; causal intervention is the next experiment.",
            ],
        }

        write_csv(output_dir / "per_layer.csv", per_layer_rows)
        write_csv(output_dir / "top_diagnostics.csv", top_rows)
        write_csv(output_dir / "failure_groups.csv", group_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report_lines = [
            "=" * 100,
            "OBJECT-PAIR SPATIAL FAILURE DIAGNOSTIC",
            "=" * 100,
            f"model: {args.model}",
            f"samples: {len(sample_rows)}",
            f"normal generation accuracy: {generation_accuracy:.4f}",
            f"verdict: {verdict_data['label']}",
            verdict_data["explanation"],
            "",
            "BEST DIAGNOSTIC PER FAMILY",
            "-" * 100,
        ]
        for family in ("pair", "routing", "representation"):
            row = family_best.get(family)
            if row is None:
                report_lines.append(f"{family:<16} unavailable")
                continue
            report_lines.append(
                f"{family:<16} {row['metric']}@L{row['layer']} | "
                f"correct={row['correct_mean']} | wrong={row['wrong_mean']} | "
                f"d={row['cohen_d']} | AUC={row['auc_correct']} | "
                f"CV-BA={row['cv_balanced_accuracy']}"
            )

        report_lines.extend([
            "",
            "POST-HOC FAILURE GROUPS",
            "-" * 100,
        ])
        for name, count in group_counts.most_common():
            report_lines.append(f"{name:<28} {count}")

        report_lines.extend([
            "",
            "HOW TO INTERPRET",
            "-" * 100,
            "1. Pair family high: wrong answers tend to have weak bidirectional object-token edges.",
            "2. Routing family high: pair edges exist, but the prompt-last decision token fails to read both objects.",
            "3. Representation family high: original/swap role differences are less antisymmetric in wrong samples.",
            "4. If all CV-BA values remain near 0.5, this attention-path hypothesis is not supported.",
            "5. Only after a positive result should a causal repair script modify the diagnosed edge/window.",
        ])
        report = "\n".join(report_lines) + "\n"
        print("\n" + report)
        (output_dir / "report.txt").write_text(report, encoding="utf-8")

        print("Saved:")
        for name in (
            "config.json",
            "samples.jsonl",
            "sample_arrays/",
            "per_layer.csv",
            "top_diagnostics.csv",
            "failure_groups.csv",
            "summary.json",
            "report.txt",
            "errors.jsonl",
        ):
            print(" ", output_dir / name)

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
