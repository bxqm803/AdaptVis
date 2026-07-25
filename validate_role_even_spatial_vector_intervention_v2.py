#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate a role-even four-direction spatial representation and intervene on every sample.

For each image, run both ordered questions:

    Q_AB: Where is A in relation to B?
    Q_BA: Where is B in relation to A?

At decoder layer l:

    r_AB(l) = h_A,subject(l) - h_B,reference(l)
    r_BA_aligned(l) = -(h_B,subject(l) - h_A,reference(l))
    d_even(l) = 0.5 * (r_AB(l) + r_BA_aligned(l))

The role-even state averages away part of the subject/reference-role effect while
retaining the physical A-B order. It is still a mixed object-pair representation,
not assumed to be pure spatial information.

Main changes in v2
------------------
1. Every valid sample is evaluated and intervened on. Centroids/directions are
   cross-fitted by unordered object-pair fold, so each sample uses models trained
   without its own object-pair group.
2. Original and swapped questions are both greedily generated. Results are
   reported overall and separately for both_correct, original_only, swapped_only,
   and both_wrong.
3. No horizontal/vertical axis is assumed. Each relation has its own one-vs-rest
   direction:

       v_r = normalize(c_r - mean_{q != r} c_q),
       r in {left, right, above, below}.

   The four directions need not be opposite or orthogonal.
4. Causal interventions on Q_AB are symmetric across the two object tokens:

       h_A' = h_A + 0.5 * delta_pair
       h_B' = h_B - 0.5 * delta_pair

   so their mean is preserved and the A-B difference changes by delta_pair.

Interventions
-------------
- remove_gt_direction(alpha): progressively remove the sample's component along
  its ground-truth relation direction.
- replace_with_opposite_direction(alpha): remove the current ground-truth
  directional component and inject the opposite relation's train prototype
  component. alpha interpolates from no change to full replacement.
- random_control: same-norm perturbation orthogonal to the span of all four
  relation directions.

For each intervention the script records object-pair, final-last, native LM, and
natural-generation effects. The key question is whether manipulating a
cross-fitted object-token relation component propagates to last and generation.

Required repository files
-------------------------
Place this script in the AdaptVis repository root next to:

    analyze_coco_centroid_generation_step1_v4.py
    extract_two_object_relation_states.py

Outputs
-------
    config.json
    extraction.jsonl
    state_cache/<sid>.npz
    layer_metrics.csv
    selected_layer.json
    interventions.jsonl
    intervention_summary.csv
    report.txt
    errors.jsonl

Example
-------
python validate_role_even_spatial_vector_intervention_v2.py \
  --dataset coco_two \
  --data-root data \
  --model qwen-3b \
  --device cuda:0 \
  --layers all \
  --folds 5 \
  --alphas 0.25,0.5,0.75,1.0 \
  --output-dir output/role_even_four_direction/qwen-3b \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "role-even-four-direction-intervention-v2"
RELATIONS = ("left", "right", "above", "below")
RELATION_TO_INDEX = {name: index for index, name in enumerate(RELATIONS)}
INDEX_TO_RELATION = {index: name for name, index in RELATION_TO_INDEX.items()}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}



# ---------------------------------------------------------------------------
# CLI and generic utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager"])
    p.add_argument(
        "--core-module",
        default="analyze_coco_centroid_generation_step1_v4",
        help="Existing repository helper module imported by this script.",
    )
    p.add_argument(
        "--layers",
        default="all",
        help="Comma-separated zero-based decoder layers or 'all'.",
    )
    p.add_argument(
        "--intervention-layer",
        default="auto",
        help=(
            "Decoder layer to patch, or 'auto'. Auto selects the highest TRAIN "
            "leave-one-out role-even centroid accuracy, breaking ties toward "
            "the earlier layer."
        ),
    )
    p.add_argument(
        "--object-state",
        default="last",
        choices=["last", "mean"],
        help=(
            "How to represent a multi-token object span. 'last' matches the "
            "existing two-object pipeline; 'mean' averages the full span."
        ),
    )
    p.add_argument(
        "--centroid-metric",
        default="cosine",
        choices=["cosine", "euclidean"],
    )
    p.add_argument(
        "--folds",
        type=int,
        default=5,
        help=(
            "Number of unordered-object-pair folds used for cross-fitting. "
            "Every sample is evaluated with centroids trained on the other folds."
        ),
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--intervention-max-samples",
        type=int,
        default=None,
        help="Optional cap on all-sample causal intervention; default runs every valid sample.",
    )
    p.add_argument(
        "--alphas",
        default="0.25,0.5,0.75,1.0",
        help=(
            "Intervention strengths. For remove_gt_direction, alpha=1 removes "
            "the full current GT-direction component. For opposite replacement, "
            "alpha=1 performs the full directional replacement."
        ),
    )
    p.add_argument(
        "--include-opposite-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Interpolate from the current GT-direction component to the opposite "
            "relation prototype direction."
        ),
    )
    p.add_argument(
        "--include-random-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--random-controls-per-spatial",
        type=int,
        default=1,
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--array-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Reuse a completed state_cache and extraction.jsonl.",
    )
    p.add_argument(
        "--skip-interventions",
        action="store_true",
        help="Run only the swap-consistency and centroid validation stages.",
    )
    return p.parse_args()


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError("--alphas must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("--alphas resolved to an empty list")
    return list(dict.fromkeys(values))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


def stable_group_fold(subject: str, reference: str, folds: int, seed: int) -> int:
    pair = "||".join(sorted([str(subject).strip().lower(), str(reference).strip().lower()]))
    digest = hashlib.sha1(f"{seed}::{pair}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % folds


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(a) * normalize_rows(b), axis=-1)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unable to find first tensor in {type(output).__name__}")


def replace_first_tensor(output: Any, new_first: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_first
    if isinstance(output, tuple):
        return (new_first, *output[1:])
    if isinstance(output, list):
        return [new_first, *output[1:]]
    raise TypeError(f"Unsupported module output type: {type(output).__name__}")


def span_positions(span: Tuple[int, int], mode: str) -> Tuple[int, ...]:
    start, end = int(span[0]), int(span[1])
    if mode == "last":
        return (end,)
    return tuple(range(start, end + 1))


def span_state(hidden: torch.Tensor, span: Tuple[int, int], mode: str) -> torch.Tensor:
    positions = span_positions(span, mode)
    index = torch.as_tensor(positions, device=hidden.device, dtype=torch.long)
    values = hidden[0].index_select(0, index)
    if mode == "last":
        return values[0]
    return values.mean(dim=0)


def output_hidden_layers(outputs: Any, n_layers: int) -> Tuple[torch.Tensor, ...]:
    values = tuple(getattr(outputs, "hidden_states", ()) or ())
    if len(values) == n_layers + 1:
        return values[1:]
    if len(values) == n_layers:
        return values
    raise RuntimeError(
        f"Unexpected hidden_states length={len(values)}, decoder_layers={n_layers}"
    )


# ---------------------------------------------------------------------------
# Centroid classifiers and role-even geometry
# ---------------------------------------------------------------------------


@dataclass
class CentroidModel:
    centroids: np.ndarray  # [4, D]
    counts: np.ndarray     # [4]
    metric: str

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.metric == "cosine":
            scores = normalize_rows(x) @ normalize_rows(self.centroids).T
            pred = np.argmax(scores, axis=1)
            sorted_scores = np.sort(scores, axis=1)
            confidence = sorted_scores[:, -1] - sorted_scores[:, -2]
            return pred.astype(np.int64), confidence.astype(np.float32)
        distances = np.sum((x[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)
        pred = np.argmin(distances, axis=1)
        sorted_distances = np.sort(distances, axis=1)
        confidence = sorted_distances[:, 1] - sorted_distances[:, 0]
        return pred.astype(np.int64), confidence.astype(np.float32)


def fit_centroid_model(x: np.ndarray, y: np.ndarray, metric: str) -> CentroidModel:
    if x.ndim != 2:
        raise ValueError(f"Expected [N,D], got {x.shape}")
    centroids = []
    counts = []
    for relation_index in range(len(RELATIONS)):
        mask = y == relation_index
        count = int(mask.sum())
        if count == 0:
            raise RuntimeError(
                f"Training split has no {INDEX_TO_RELATION[relation_index]} samples"
            )
        centroids.append(x[mask].mean(axis=0))
        counts.append(count)
    return CentroidModel(
        centroids=np.stack(centroids, axis=0).astype(np.float32),
        counts=np.asarray(counts, dtype=np.int64),
        metric=metric,
    )


def loo_centroid_accuracy(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    n, dim = x.shape
    class_sums = np.zeros((len(RELATIONS), dim), dtype=np.float64)
    class_counts = np.zeros((len(RELATIONS),), dtype=np.int64)
    for relation_index in range(len(RELATIONS)):
        mask = y == relation_index
        class_sums[relation_index] = x[mask].sum(axis=0, dtype=np.float64)
        class_counts[relation_index] = int(mask.sum())
    correct = 0
    valid = 0
    for index in range(n):
        label = int(y[index])
        if class_counts[label] <= 1:
            continue
        centroids = class_sums.copy()
        counts = class_counts.copy()
        centroids[label] -= x[index]
        counts[label] -= 1
        centroids = centroids / counts[:, None]
        model = CentroidModel(
            centroids=centroids.astype(np.float32),
            counts=counts,
            metric=metric,
        )
        pred, _ = model.predict(x[index:index + 1])
        correct += int(pred[0] == label)
        valid += 1
    return float(correct / valid) if valid else float("nan")


@dataclass
class DirectionGeometry:
    """Four independent one-vs-rest relation directions."""

    center: np.ndarray                 # [D], mean of four class centroids
    directions: np.ndarray             # [4,D], unit one-vs-rest directions
    class_prototype_amplitudes: np.ndarray  # [4], projection of c_r-center on v_r
    pairwise_cosine: np.ndarray        # [4,4]

    def direction(self, relation: str) -> np.ndarray:
        return self.directions[RELATION_TO_INDEX[relation]]

    def prototype_amplitude(self, relation: str) -> float:
        return float(self.class_prototype_amplitudes[RELATION_TO_INDEX[relation]])


def build_direction_geometry(centroids: np.ndarray) -> DirectionGeometry:
    if centroids.shape[0] != len(RELATIONS):
        raise ValueError(f"Expected four centroids, got {centroids.shape}")
    center = centroids.mean(axis=0).astype(np.float64)
    directions = []
    amplitudes = []
    for relation_index in range(len(RELATIONS)):
        others = np.delete(centroids, relation_index, axis=0).mean(axis=0)
        raw = centroids[relation_index].astype(np.float64) - others.astype(np.float64)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-12:
            raise RuntimeError(
                f"Degenerate one-vs-rest direction for {INDEX_TO_RELATION[relation_index]}"
            )
        direction = raw / norm
        directions.append(direction)
        amplitudes.append(float(np.dot(centroids[relation_index] - center, direction)))
    direction_matrix = np.stack(directions, axis=0)
    pairwise = direction_matrix @ direction_matrix.T
    return DirectionGeometry(
        center=center.astype(np.float32),
        directions=direction_matrix.astype(np.float32),
        class_prototype_amplitudes=np.asarray(amplitudes, dtype=np.float32),
        pairwise_cosine=pairwise.astype(np.float32),
    )


def direction_scores(x: np.ndarray, geometry: DirectionGeometry) -> np.ndarray:
    """Return one score per independent relation direction."""
    x2 = np.atleast_2d(x).astype(np.float32)
    return (x2 - geometry.center[None, :]) @ geometry.directions.T


def gt_direction_score(x: np.ndarray, relation: str, geometry: DirectionGeometry) -> float:
    return float(direction_scores(x, geometry)[0, RELATION_TO_INDEX[relation]])


def gt_vs_opposite_margin(
    x: np.ndarray, relation: str, geometry: DirectionGeometry
) -> float:
    scores = direction_scores(x, geometry)[0]
    return float(
        scores[RELATION_TO_INDEX[relation]]
        - scores[RELATION_TO_INDEX[OPPOSITE[relation]]]
    )


def gt_one_vs_rest_margin(
    x: np.ndarray, relation: str, geometry: DirectionGeometry
) -> float:
    scores = direction_scores(x, geometry)[0]
    gt_index = RELATION_TO_INDEX[relation]
    other = np.delete(scores, gt_index)
    return float(scores[gt_index] - np.max(other))


def orthogonal_random_vector(
    dim: int,
    norm: float,
    geometry: DirectionGeometry,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a same-norm vector orthogonal to the span of all four directions."""
    if norm <= 0:
        return np.zeros((dim,), dtype=np.float32)
    basis = geometry.directions.astype(np.float64).T  # [D,4]
    q, _ = np.linalg.qr(basis)
    rank = int(np.linalg.matrix_rank(basis))
    q = q[:, :rank]
    for _ in range(20):
        vector = rng.normal(size=(dim,)).astype(np.float64)
        if rank:
            vector = vector - q @ (q.T @ vector)
        current = float(np.linalg.norm(vector))
        if current >= 1e-10:
            return (vector / current * float(norm)).astype(np.float32)
    raise RuntimeError("Unable to sample an orthogonal random control vector")


# ---------------------------------------------------------------------------
# Prompt extraction and intervention hooks
# ---------------------------------------------------------------------------


@dataclass
class PromptForward:
    subject_span: Tuple[int, int]
    reference_span: Tuple[int, int]
    prompt_last_index: int
    layer_subject: np.ndarray  # [L,D]
    layer_reference: np.ndarray
    layer_last: np.ndarray
    relation_logits: np.ndarray
    relation_probs: np.ndarray
    relation_prediction: Optional[str]
    generated_text: Optional[str]
    generated_prediction: Optional[str]


@torch.inference_mode()
def forward_prompt(
    *,
    base: Any,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    decoder_layer_count: int,
    object_state: str,
    relation_token_map: Dict[str, List[int]],
    generate: bool,
    max_new_tokens: int,
) -> PromptForward:
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = base.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    prompt_last_index = int(len(input_ids) - 1)

    outputs = model(
        **batch,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_layers = output_hidden_layers(outputs, decoder_layer_count)
    if max(prompt_last_index, subject_span[1], reference_span[1]) >= int(hidden_layers[0].shape[1]):
        raise RuntimeError(
            "Text token index exceeds decoder hidden length. "
            f"input_ids={len(input_ids)}, hidden={hidden_layers[0].shape[1]}, "
            f"subject={subject_span}, reference={reference_span}"
        )

    subject_states = []
    reference_states = []
    last_states = []
    for layer in selected_layers:
        hidden = hidden_layers[int(layer)]
        subject_states.append(span_state(hidden, subject_span, object_state).float().cpu().numpy())
        reference_states.append(span_state(hidden, reference_span, object_state).float().cpu().numpy())
        last_states.append(hidden[0, prompt_last_index].float().cpu().numpy())

    relation_data = base.relation_scores(
        outputs.logits[0, -1],
        relation_token_map,
        gt=None,
    )

    generated_text: Optional[str] = None
    generated_prediction: Optional[str] = None
    if generate:
        generated_text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
        generated_prediction = base.normalize_relation(generated_text)

    result = PromptForward(
        subject_span=subject_span,
        reference_span=reference_span,
        prompt_last_index=prompt_last_index,
        layer_subject=np.stack(subject_states, axis=0).astype(np.float32),
        layer_reference=np.stack(reference_states, axis=0).astype(np.float32),
        layer_last=np.stack(last_states, axis=0).astype(np.float32),
        relation_logits=np.asarray(relation_data["logits"], dtype=np.float32),
        relation_probs=np.asarray(relation_data["probs"], dtype=np.float32),
        relation_prediction=relation_data["prediction"],
        generated_text=generated_text,
        generated_prediction=generated_prediction,
    )
    del outputs, hidden_layers
    return result


class PairStateIntervention:
    """Symmetrically modify A/B object token states at one decoder block output."""

    def __init__(
        self,
        layer_module: torch.nn.Module,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        delta_pair: np.ndarray,
    ) -> None:
        self.layer_module = layer_module
        self.subject_positions = tuple(int(x) for x in subject_positions)
        self.reference_positions = tuple(int(x) for x in reference_positions)
        self.delta_pair = np.asarray(delta_pair, dtype=np.float32)
        self.applied = False
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, args: Tuple[Any, ...], output: Any):
        hidden = first_tensor(output)
        max_position = max(self.subject_positions + self.reference_positions)
        # Prefill/full forward only. Cached decode steps have sequence length 1.
        if int(hidden.shape[1]) <= max_position:
            return output
        delta = torch.as_tensor(
            self.delta_pair,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        modified = hidden.clone()
        for position in self.subject_positions:
            modified[:, position, :] = modified[:, position, :] + 0.5 * delta
        for position in self.reference_positions:
            modified[:, position, :] = modified[:, position, :] - 0.5 * delta
        self.applied = True
        return replace_first_tensor(output, modified)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self) -> "PairStateIntervention":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Extraction cache
# ---------------------------------------------------------------------------


def cache_path(cache_dir: Path, sid: int) -> Path:
    return cache_dir / f"{int(sid)}.npz"


def save_state_cache(
    path: Path,
    *,
    layers: Sequence[int],
    r_ab: np.ndarray,
    r_ba_aligned: np.ndarray,
    d_even: np.ndarray,
    mean_ab: np.ndarray,
    last_ab: np.ndarray,
    dtype: str,
) -> None:
    array_dtype = np.float16 if dtype == "float16" else np.float32
    np.savez_compressed(
        path,
        layers=np.asarray(layers, dtype=np.int16),
        r_ab=r_ab.astype(array_dtype),
        r_ba_aligned=r_ba_aligned.astype(array_dtype),
        d_even=d_even.astype(array_dtype),
        mean_ab=mean_ab.astype(array_dtype),
        last_ab=last_ab.astype(array_dtype),
    )


def load_state_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def extraction_complete(rows: Sequence[Mapping[str, Any]], cache_dir: Path) -> bool:
    if not rows:
        return False
    return all(cache_path(cache_dir, int(row["sid"])).exists() for row in rows)


# ---------------------------------------------------------------------------
# Layer metrics
# ---------------------------------------------------------------------------


def stack_layer(
    rows: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    key: str,
    layer_position: int,
) -> np.ndarray:
    values = []
    for row in rows:
        cached = load_state_cache(cache_path(cache_dir, int(row["sid"])))
        values.append(np.asarray(cached[key][layer_position], dtype=np.float32))
    return np.stack(values, axis=0)


def labels_from_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [RELATION_TO_INDEX[str(row["gt"])] for row in rows],
        dtype=np.int64,
    )


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(pred == labels)) if len(labels) else float("nan")


def pair_correctness_status(original_correct: bool, swapped_correct: bool) -> str:
    if original_correct and swapped_correct:
        return "both_correct"
    if original_correct:
        return "original_only"
    if swapped_correct:
        return "swapped_only"
    return "both_wrong"


def evaluate_crossfit_layer_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    selected_layers: Sequence[int],
    folds: int,
    metric: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, Dict[str, Any]]]]:
    """Cross-fit every layer so every sample is evaluated out of its object-pair fold."""
    labels = labels_from_rows(rows)
    fold_ids = np.asarray([int(row["group_fold"]) for row in rows], dtype=np.int64)
    statuses = np.asarray(
        [str(row.get("generation_pair_status", "unknown")) for row in rows],
        dtype=object,
    )
    layer_rows: List[Dict[str, Any]] = []
    models: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for layer_position, layer in enumerate(selected_layers):
        r_ab = stack_layer(rows, cache_dir, "r_ab", layer_position)
        r_ba = stack_layer(rows, cache_dir, "r_ba_aligned", layer_position)
        d_even = stack_layer(rows, cache_dir, "d_even", layer_position)
        last_ab = stack_layer(rows, cache_dir, "last_ab", layer_position)

        pred_r_ab = np.full((len(rows),), -1, dtype=np.int64)
        pred_r_ba = np.full((len(rows),), -1, dtype=np.int64)
        pred_d = np.full((len(rows),), -1, dtype=np.int64)
        pred_last = np.full((len(rows),), -1, dtype=np.int64)
        d_conf = np.full((len(rows),), np.nan, dtype=np.float32)
        last_conf = np.full((len(rows),), np.nan, dtype=np.float32)
        d_opp_margin = np.full((len(rows),), np.nan, dtype=np.float32)
        d_ovr_margin = np.full((len(rows),), np.nan, dtype=np.float32)
        last_opp_margin = np.full((len(rows),), np.nan, dtype=np.float32)
        pair_cos = cosine_rows(r_ab, r_ba).astype(np.float32)
        fold_loo: List[Tuple[int, float]] = []
        models[int(layer)] = {}

        for fold in range(folds):
            eval_mask = fold_ids == fold
            if not np.any(eval_mask):
                continue
            train_mask = ~eval_mask
            train_labels = labels[train_mask]
            missing = [
                relation for relation in RELATIONS
                if not np.any(train_labels == RELATION_TO_INDEX[relation])
            ]
            if missing:
                raise RuntimeError(
                    f"Fold {fold} train split at L{layer} has no samples for {missing}"
                )

            role_even_model = fit_centroid_model(d_even[train_mask], train_labels, metric)
            r_ab_model = fit_centroid_model(r_ab[train_mask], train_labels, metric)
            r_ba_model = fit_centroid_model(r_ba[train_mask], train_labels, metric)
            last_model = fit_centroid_model(last_ab[train_mask], train_labels, metric)
            role_even_geometry = build_direction_geometry(role_even_model.centroids)
            r_ab_geometry = build_direction_geometry(r_ab_model.centroids)
            r_ba_geometry = build_direction_geometry(r_ba_model.centroids)
            last_geometry = build_direction_geometry(last_model.centroids)

            pred_r_ab[eval_mask], _ = r_ab_model.predict(r_ab[eval_mask])
            pred_r_ba[eval_mask], _ = r_ba_model.predict(r_ba[eval_mask])
            pred_d[eval_mask], d_conf[eval_mask] = role_even_model.predict(d_even[eval_mask])
            pred_last[eval_mask], last_conf[eval_mask] = last_model.predict(last_ab[eval_mask])

            eval_indices = np.flatnonzero(eval_mask)
            for index in eval_indices:
                relation = str(rows[index]["gt"])
                d_opp_margin[index] = gt_vs_opposite_margin(
                    d_even[index], relation, role_even_geometry
                )
                d_ovr_margin[index] = gt_one_vs_rest_margin(
                    d_even[index], relation, role_even_geometry
                )
                last_opp_margin[index] = gt_vs_opposite_margin(
                    last_ab[index], relation, last_geometry
                )

            loo = loo_centroid_accuracy(d_even[train_mask], train_labels, metric)
            fold_loo.append((int(train_mask.sum()), loo))
            models[int(layer)][fold] = {
                "role_even_model": role_even_model,
                "r_ab_model": r_ab_model,
                "r_ba_model": r_ba_model,
                "last_model": last_model,
                "role_even_geometry": role_even_geometry,
                "r_ab_geometry": r_ab_geometry,
                "r_ba_geometry": r_ba_geometry,
                "last_geometry": last_geometry,
                "train_loo": loo,
            }

        valid = pred_d >= 0
        if not np.all(valid):
            missing_indices = np.flatnonzero(~valid)[:10].tolist()
            raise RuntimeError(f"Some samples were not cross-fitted: {missing_indices}")
        weighted_loo = float(
            sum(n * score for n, score in fold_loo if np.isfinite(score))
            / max(1, sum(n for n, score in fold_loo if np.isfinite(score)))
        )
        row: Dict[str, Any] = {
            "layer": int(layer),
            "n": int(len(rows)),
            "crossfit_train_loo_d_even_accuracy": weighted_loo,
            "oof_r_ab_accuracy": accuracy(pred_r_ab, labels),
            "oof_r_ba_aligned_accuracy": accuracy(pred_r_ba, labels),
            "oof_d_even_accuracy": accuracy(pred_d, labels),
            "oof_last_accuracy": accuracy(pred_last, labels),
            "oof_mean_swap_aligned_cosine": float(np.mean(pair_cos)),
            "oof_median_swap_aligned_cosine": float(np.median(pair_cos)),
            "oof_d_last_opposite_margin_correlation": pearson_safe(
                d_opp_margin, last_opp_margin
            ),
            "oof_d_even_mean_confidence": float(np.nanmean(d_conf)),
            "oof_last_mean_confidence": float(np.nanmean(last_conf)),
            "oof_d_even_mean_gt_vs_opposite_margin": float(np.nanmean(d_opp_margin)),
            "oof_d_even_mean_gt_one_vs_rest_margin": float(np.nanmean(d_ovr_margin)),
        }
        for status in ("both_correct", "original_only", "swapped_only", "both_wrong"):
            mask = statuses == status
            row[f"n_{status}"] = int(mask.sum())
            row[f"oof_d_even_accuracy_{status}"] = (
                accuracy(pred_d[mask], labels[mask]) if np.any(mask) else float("nan")
            )
            row[f"oof_last_accuracy_{status}"] = (
                accuracy(pred_last[mask], labels[mask]) if np.any(mask) else float("nan")
            )
            row[f"mean_swap_cosine_{status}"] = (
                float(np.mean(pair_cos[mask])) if np.any(mask) else float("nan")
            )
        layer_rows.append(row)

    return layer_rows, models


def select_intervention_layer(
    requested: str,
    selected_layers: Sequence[int],
    layer_rows: Sequence[Mapping[str, Any]],
) -> int:
    text = str(requested).strip().lower()
    final_layer = max(int(x) for x in selected_layers)
    if text != "auto":
        layer = int(text)
        if layer < 0:
            layer = final_layer + 1 + layer
        if layer not in selected_layers:
            raise ValueError(
                f"Requested intervention layer L{layer} is not in selected layers {selected_layers}"
            )
        if layer == final_layer:
            raise ValueError(
                "Do not intervene after the final decoder block: object-token changes "
                "cannot propagate to prompt_last."
            )
        return layer

    candidates = [
        row for row in layer_rows
        if int(row["layer"]) != final_layer
        and np.isfinite(safe_float(row["oof_d_even_accuracy"]))
    ]
    if not candidates:
        raise RuntimeError("No non-final layer has finite cross-fitted d_even accuracy")
    candidates.sort(
        key=lambda row: (
            -safe_float(row["oof_d_even_accuracy"]),
            -safe_float(row["crossfit_train_loo_d_even_accuracy"]),
            int(row["layer"]),
        )
    )
    return int(candidates[0]["layer"])


# ---------------------------------------------------------------------------
# Intervention evaluation
# ---------------------------------------------------------------------------


def relation_native_margin(
    logits: np.ndarray,
    gt: str,
) -> float:
    gt_index = RELATION_TO_INDEX[gt]
    opposite_index = RELATION_TO_INDEX[OPPOSITE[gt]]
    return float(logits[gt_index] - logits[opposite_index])


def condition_delta(
    *,
    condition: str,
    alpha: Optional[float],
    d_even: np.ndarray,
    r_ab: np.ndarray,
    gt: str,
    geometry: DirectionGeometry,
    random_index: int,
    sid: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return delta on r_AB; d_even changes by exactly 0.5 * delta_pair."""
    gt_direction = geometry.direction(gt)
    opposite = OPPOSITE[gt]
    opposite_direction = geometry.direction(opposite)
    centered = d_even - geometry.center
    gt_amplitude = float(np.dot(centered, gt_direction))
    current_gt_component = gt_amplitude * gt_direction
    opposite_prototype_amplitude = geometry.prototype_amplitude(opposite)
    opposite_prototype_component = opposite_prototype_amplitude * opposite_direction

    if condition == "remove_gt_direction":
        assert alpha is not None
        delta_d_even = -float(alpha) * current_gt_component
        delta_pair = 2.0 * delta_d_even
        return delta_pair.astype(np.float32), {
            "alpha": float(alpha),
            "gt_direction_amplitude": gt_amplitude,
            "opposite_prototype_amplitude": opposite_prototype_amplitude,
            "target_gt_direction_amplitude": float((1.0 - float(alpha)) * gt_amplitude),
            "target_opposite_direction_amplitude": float(
                np.dot(centered + delta_d_even, opposite_direction)
            ),
            "target_relation": None,
            "delta_d_even_norm": float(np.linalg.norm(delta_d_even)),
        }

    if condition == "replace_with_opposite_direction":
        assert alpha is not None
        full_replacement = opposite_prototype_component - current_gt_component
        delta_d_even = float(alpha) * full_replacement
        delta_pair = 2.0 * delta_d_even
        target_centered = centered + delta_d_even
        return delta_pair.astype(np.float32), {
            "alpha": float(alpha),
            "gt_direction_amplitude": gt_amplitude,
            "opposite_prototype_amplitude": opposite_prototype_amplitude,
            "target_gt_direction_amplitude": float(np.dot(target_centered, gt_direction)),
            "target_opposite_direction_amplitude": float(
                np.dot(target_centered, opposite_direction)
            ),
            "target_relation": opposite,
            "delta_d_even_norm": float(np.linalg.norm(delta_d_even)),
        }

    if condition.startswith("random_control_"):
        assert alpha is not None
        if condition == "random_control_remove":
            matched_delta_d_even = -float(alpha) * current_gt_component
        elif condition == "random_control_replace":
            matched_delta_d_even = float(alpha) * (
                opposite_prototype_component - current_gt_component
            )
        else:
            raise ValueError(f"Unsupported random control condition: {condition}")
        rng = np.random.default_rng(
            int(seed) * 1000003
            + int(sid) * 97
            + int(random_index) * 7919
            + int(round(float(alpha) * 1000))
            + (17 if condition.endswith("replace") else 0)
        )
        delta_d_even = orthogonal_random_vector(
            dim=int(d_even.shape[0]),
            norm=float(np.linalg.norm(matched_delta_d_even)),
            geometry=geometry,
            rng=rng,
        )
        delta_pair = 2.0 * delta_d_even
        return delta_pair.astype(np.float32), {
            "alpha": float(alpha),
            "gt_direction_amplitude": gt_amplitude,
            "opposite_prototype_amplitude": opposite_prototype_amplitude,
            "target_gt_direction_amplitude": float("nan"),
            "target_opposite_direction_amplitude": float("nan"),
            "target_relation": None,
            "delta_d_even_norm": float(np.linalg.norm(delta_d_even)),
        }

    raise ValueError(f"Unsupported condition: {condition}")


@torch.inference_mode()
def run_intervened_prompt(
    *,
    base: Any,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[torch.nn.Module],
    decoder_layer_count: int,
    selected_layers: Sequence[int],
    intervention_layer: int,
    batch: Dict[str, Any],
    subject: str,
    reference: str,
    object_state: str,
    delta_pair: np.ndarray,
    relation_token_map: Dict[str, List[int]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = base.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_positions = span_positions(subject_span, object_state)
    reference_positions = span_positions(reference_span, object_state)
    prompt_last_index = len(input_ids) - 1

    with PairStateIntervention(
        decoder_layers[intervention_layer],
        subject_positions,
        reference_positions,
        delta_pair,
    ) as intervention:
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if not intervention.applied:
            raise RuntimeError("Pair-state intervention hook was not applied during forward")
        hidden_layers = output_hidden_layers(outputs, decoder_layer_count)
        relation_data = base.relation_scores(
            outputs.logits[0, -1],
            relation_token_map,
            gt=None,
        )

    # Re-register for generation because the previous context removed the hook.
    with PairStateIntervention(
        decoder_layers[intervention_layer],
        subject_positions,
        reference_positions,
        delta_pair,
    ) as generation_intervention:
        generated_text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
        if not generation_intervention.applied:
            raise RuntimeError("Pair-state intervention hook was not applied during generation")

    intervention_position = list(selected_layers).index(intervention_layer)
    patched_layer_hidden = hidden_layers[intervention_layer]
    patched_subject = span_state(patched_layer_hidden, subject_span, object_state).float().cpu().numpy()
    patched_reference = span_state(patched_layer_hidden, reference_span, object_state).float().cpu().numpy()
    patched_pair = patched_subject - patched_reference

    selected_last = np.stack(
        [hidden_layers[int(layer)][0, prompt_last_index].float().cpu().numpy() for layer in selected_layers],
        axis=0,
    ).astype(np.float32)

    result = {
        "patched_pair": patched_pair.astype(np.float32),
        "selected_last": selected_last,
        "relation_logits": np.asarray(relation_data["logits"], dtype=np.float32),
        "relation_probs": np.asarray(relation_data["probs"], dtype=np.float32),
        "relation_prediction": relation_data["prediction"],
        "generated_text": generated_text,
        "generated_prediction": base.normalize_relation(generated_text),
        "intervention_position": intervention_position,
    }
    del outputs, hidden_layers
    return result


def _summarize_group(
    condition: str,
    alpha: Optional[float],
    group_type: str,
    group_value: str,
    group: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    baseline_generation_correct = np.asarray(
        [bool(row["baseline_generation_correct"]) for row in group], dtype=bool
    )
    intervention_generation_correct = np.asarray(
        [bool(row["intervention_generation_correct"]) for row in group], dtype=bool
    )
    changed_to_opposite = np.asarray(
        [bool(row["generation_changed_to_opposite"]) for row in group], dtype=bool
    )
    identity_cosine = np.asarray([
        min(
            safe_float(row["subject_state_cosine_pre_post"]),
            safe_float(row["reference_state_cosine_pre_post"]),
        )
        for row in group
    ])

    def delta(patched_key: str, baseline_key: str) -> np.ndarray:
        return np.asarray([
            safe_float(row[patched_key]) - safe_float(row[baseline_key])
            for row in group
        ])

    d_role_opp = delta(
        "patched_role_even_gt_vs_opposite_margin",
        "baseline_role_even_gt_vs_opposite_margin",
    )
    d_obj_opp = delta(
        "patched_object_gt_vs_opposite_margin",
        "baseline_object_gt_vs_opposite_margin",
    )
    d_last_opp = delta(
        "patched_final_last_gt_vs_opposite_margin",
        "baseline_final_last_gt_vs_opposite_margin",
    )
    d_lm = delta("patched_lm_gt_margin", "baseline_lm_gt_margin")
    return {
        "condition": condition,
        "alpha": alpha,
        "group_type": group_type,
        "group_value": group_value,
        "n": len(group),
        "baseline_generation_accuracy": float(baseline_generation_correct.mean()),
        "intervention_generation_accuracy": float(intervention_generation_correct.mean()),
        "generation_opposite_rate": float(changed_to_opposite.mean()),
        "mean_role_even_gt_vs_opposite_margin_change": float(np.nanmean(d_role_opp)),
        "mean_object_gt_vs_opposite_margin_change": float(np.nanmean(d_obj_opp)),
        "mean_final_last_gt_vs_opposite_margin_change": float(np.nanmean(d_last_opp)),
        "mean_lm_margin_change": float(np.nanmean(d_lm)),
        "role_even_opposite_margin_sign_flip_rate": float(np.mean(
            [
                np.sign(safe_float(row["patched_role_even_gt_vs_opposite_margin"]))
                != np.sign(safe_float(row["baseline_role_even_gt_vs_opposite_margin"]))
                for row in group
            ]
        )),
        "object_opposite_margin_sign_flip_rate": float(np.mean(
            [
                np.sign(safe_float(row["patched_object_gt_vs_opposite_margin"]))
                != np.sign(safe_float(row["baseline_object_gt_vs_opposite_margin"]))
                for row in group
            ]
        )),
        "final_last_opposite_margin_sign_flip_rate": float(np.mean(
            [
                np.sign(safe_float(row["patched_final_last_gt_vs_opposite_margin"]))
                != np.sign(safe_float(row["baseline_final_last_gt_vs_opposite_margin"]))
                for row in group
            ]
        )),
        "lm_margin_sign_flip_rate": float(np.mean(
            [
                np.sign(safe_float(row["patched_lm_gt_margin"]))
                != np.sign(safe_float(row["baseline_lm_gt_margin"]))
                for row in group
            ]
        )),
        "mean_min_object_state_cosine": float(np.nanmean(identity_cosine)),
    }


def summarize_interventions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    base_groups: Dict[Tuple[str, Optional[float]], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        alpha = row.get("alpha")
        alpha_key = None if alpha is None else float(alpha)
        base_groups[(str(row["condition"]), alpha_key)].append(row)

    summary: List[Dict[str, Any]] = []
    for (condition, alpha), base_group in sorted(
        base_groups.items(),
        key=lambda item: (item[0][0], -1.0 if item[0][1] is None else item[0][1]),
    ):
        summary.append(_summarize_group(condition, alpha, "all", "all", base_group))
        for field, group_type in (
            ("generation_pair_status", "generation_pair_status"),
            ("first_step_pair_status", "first_step_pair_status"),
            ("gt", "relation"),
        ):
            values = sorted({str(row.get(field, "unknown")) for row in base_group})
            for value in values:
                subset = [row for row in base_group if str(row.get(field, "unknown")) == value]
                if subset:
                    summary.append(
                        _summarize_group(
                            condition, alpha, group_type, value, subset
                        )
                    )
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_pct(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def build_report(
    *,
    args: argparse.Namespace,
    extraction_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    intervention_layer: int,
    intervention_summary: Sequence[Mapping[str, Any]],
) -> str:
    selected = next(row for row in layer_rows if int(row["layer"]) == intervention_layer)
    statuses = Counter(str(row.get("generation_pair_status", "unknown")) for row in extraction_rows)
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"model: {args.model}",
        f"dataset: {args.dataset}",
        f"all valid samples: {len(extraction_rows)}",
        f"cross-fit folds by unordered object pair: {args.folds}",
        f"generation pair status: {dict(statuses)}",
        f"selected intervention layer: L{intervention_layer}",
        "",
        "SELECTED-LAYER CROSS-FITTED VALIDATION",
        f"OOF r_AB accuracy: {format_pct(safe_float(selected['oof_r_ab_accuracy']))}",
        f"OOF aligned -r_BA accuracy: {format_pct(safe_float(selected['oof_r_ba_aligned_accuracy']))}",
        f"OOF role-even d accuracy: {format_pct(safe_float(selected['oof_d_even_accuracy']))}",
        f"OOF last-token accuracy: {format_pct(safe_float(selected['oof_last_accuracy']))}",
        f"mean aligned swap cosine: {safe_float(selected['oof_mean_swap_aligned_cosine']):.6f}",
        f"d_even vs last GT/opposite-margin correlation: "
        f"{safe_float(selected['oof_d_last_opposite_margin_correlation']):.6f}",
        "",
        "FOUR-DIRECTION NOTE",
        "left, right, above, and below are separate one-vs-rest directions.",
        "They are not forced to be opposite or orthogonal; see selected_layer.json for the 4x4 cosine matrix.",
        "",
        "INTERVENTION SUMMARY (overall and generation-pair groups)",
    ]
    if not intervention_summary:
        lines.append("(skipped)")
    else:
        display = [
            row for row in intervention_summary
            if row.get("group_type") in {"all", "generation_pair_status"}
        ]
        for row in display:
            alpha = row.get("alpha")
            label = str(row["condition"])
            if alpha is not None:
                label += f" alpha={float(alpha):.3f}"
            label += f" [{row['group_type']}={row['group_value']}]"
            lines.append(
                f"{label}: N={row['n']} "
                f"| opposite_gen={format_pct(safe_float(row['generation_opposite_rate']))} "
                f"| dEvenOpp={safe_float(row['mean_role_even_gt_vs_opposite_margin_change']):+.4f} "
                f"| dObjOpp={safe_float(row['mean_object_gt_vs_opposite_margin_change']):+.4f} "
                f"| dLastOpp={safe_float(row['mean_final_last_gt_vs_opposite_margin_change']):+.4f} "
                f"| dLM={safe_float(row['mean_lm_margin_change']):+.4f} "
                f"| idCos={safe_float(row['mean_min_object_state_cosine']):.4f}"
            )
    lines.extend([
        "",
        "INTERPRETATION RULE",
        "- object changes but final-last/LM do not: the manipulated object-token component is not transmitted or not normally used.",
        "- object and final-last change but LM/generation do not: transmission occurs, but native readout/utilization fails.",
        "- object, final-last, LM, and generation all follow replacement: the cross-fitted direction is a causal downstream-used spatial component.",
        "- random controls match the spatial effect or object-state cosine collapses: the effect is nonspecific or too strong.",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.random_controls_per_spatial < 1:
        raise ValueError("--random-controls-per-spatial must be >= 1")

    alphas = parse_float_list(args.alphas)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "state_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = output_dir / "extraction.jsonl"
    errors_path = output_dir / "errors.jsonl"
    layer_metrics_path = output_dir / "layer_metrics.csv"
    selected_layer_path = output_dir / "selected_layer.json"
    interventions_path = output_dir / "interventions.jsonl"
    intervention_summary_path = output_dir / "intervention_summary.csv"
    report_path = output_dir / "report.txt"

    try:
        base = importlib.import_module(args.core_module)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to import {args.core_module}.py. Place this script next to "
            "analyze_coco_centroid_generation_step1_v4.py.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    module = base.import_two_object_module()
    records, audit = module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    if not records:
        raise RuntimeError("No usable records")
    records_by_sid = {int(record.sid): record for record in records}

    prompt_path = Path(args.prompt_jsonl) if args.prompt_jsonl else base.resolve_prompt_path(args)
    prompt_rows = base.load_standard_prompts(prompt_path)
    missing = [int(record.sid) for record in records if int(record.sid) not in prompt_rows]
    if missing:
        raise RuntimeError(
            f"Prompt file is missing {len(missing)} record IDs; first={missing[:10]}"
        )

    specs = base.merged_model_specs(module)
    if args.model not in specs:
        raise ValueError(f"Unknown model {args.model!r}; available={sorted(specs)}")
    spec = specs[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)
    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    selected_layers = base.parse_layers(args.layers, len(decoder_layers))
    final_layer = len(decoder_layers) - 1
    if final_layer not in selected_layers:
        selected_layers = sorted(set(selected_layers + [final_layer]))
        print(
            f"Added final decoder layer L{final_layer} so final-last transmission can be evaluated."
        )
    relation_token_map = base.relation_token_variants(processor.tokenizer)

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "prompt_jsonl": str(prompt_path),
        "model": args.model,
        "repo_id": spec.repo_id,
        "transformers_version": transformers.__version__,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "selected_layers": selected_layers,
        "object_state": args.object_state,
        "centroid_metric": args.centroid_metric,
        "folds": args.folds,
        "alphas": alphas,
        "intervene_all_samples": args.intervention_max_samples is None,
        "audit": audit,
        "role_even_formula": "0.5*((h_A_subject-h_B_reference)-(h_B_subject-h_A_reference))",
        "direction_formula": "normalize(c_r - mean_{q!=r}(c_q))",
        "pair_patch_formula": "h_A += 0.5*delta_pair; h_B -= 0.5*delta_pair",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # -------------------- Phase 1: extract both ordered questions --------------------
    extraction_rows = read_jsonl(extraction_path)
    if args.skip_extraction:
        if not extraction_complete(extraction_rows, cache_dir):
            raise RuntimeError(
                "--skip-extraction was requested, but extraction.jsonl/state_cache is incomplete"
            )
    else:
        if extraction_rows and not args.overwrite:
            raise RuntimeError(
                "Existing extraction output found. Use --overwrite or --skip-extraction."
            )
        completed = 0
        for record in tqdm(records, desc=f"extract-role-even:{args.model}"):
            sid = int(record.sid)
            image: Optional[Image.Image] = None
            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                gt = base.normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATION_TO_INDEX:
                    raise ValueError(f"Unsupported GT for sid={sid}: {gt!r}")
                swapped_gt = OPPOSITE[gt]
                question_ab = str(prompt_row["question_text"])
                question_ba = base.build_swapped_question(subject, reference)
                image = base.record_image(record)
                batch_ab = base.make_question_batch(
                    processor=processor, image=image, question_text=question_ab, device=device
                )
                batch_ba = base.make_question_batch(
                    processor=processor, image=image, question_text=question_ba, device=device
                )

                original = forward_prompt(
                    base=base,
                    model=model,
                    processor=processor,
                    batch=batch_ab,
                    subject=subject,
                    reference=reference,
                    selected_layers=selected_layers,
                    decoder_layer_count=len(decoder_layers),
                    object_state=args.object_state,
                    relation_token_map=relation_token_map,
                    generate=True,
                    max_new_tokens=args.max_new_tokens,
                )
                swapped = forward_prompt(
                    base=base,
                    model=model,
                    processor=processor,
                    batch=batch_ba,
                    subject=reference,
                    reference=subject,
                    selected_layers=selected_layers,
                    decoder_layer_count=len(decoder_layers),
                    object_state=args.object_state,
                    relation_token_map=relation_token_map,
                    generate=True,
                    max_new_tokens=args.max_new_tokens,
                )

                r_ab = original.layer_subject - original.layer_reference
                r_ba = swapped.layer_subject - swapped.layer_reference
                r_ba_aligned = -r_ba
                d_even = 0.5 * (r_ab + r_ba_aligned)
                mean_ab = 0.5 * (original.layer_subject + original.layer_reference)

                save_state_cache(
                    cache_path(cache_dir, sid),
                    layers=selected_layers,
                    r_ab=r_ab,
                    r_ba_aligned=r_ba_aligned,
                    d_even=d_even,
                    mean_ab=mean_ab,
                    last_ab=original.layer_last,
                    dtype=args.array_dtype,
                )

                fold = stable_group_fold(subject, reference, args.folds, args.seed)
                original_generation_correct = original.generated_prediction == gt
                swapped_generation_correct = swapped.generated_prediction == swapped_gt
                original_first_step_correct = original.relation_prediction == gt
                swapped_first_step_correct = swapped.relation_prediction == swapped_gt
                append_jsonl(extraction_path, {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "gt": gt,
                    "swapped_gt": swapped_gt,
                    "group_fold": fold,
                    "question_ab": question_ab,
                    "question_ba": question_ba,
                    "baseline_relation_prediction": original.relation_prediction,
                    "baseline_first_step_correct": original_first_step_correct,
                    "baseline_generated_text": original.generated_text,
                    "baseline_generation_prediction": original.generated_prediction,
                    "baseline_generation_correct": original_generation_correct,
                    "baseline_lm_logits": original.relation_logits.tolist(),
                    "baseline_lm_probs": original.relation_probs.tolist(),
                    "baseline_lm_gt_margin": relation_native_margin(original.relation_logits, gt),
                    "swapped_relation_prediction": swapped.relation_prediction,
                    "swapped_first_step_correct": swapped_first_step_correct,
                    "swapped_generated_text": swapped.generated_text,
                    "swapped_generation_prediction": swapped.generated_prediction,
                    "swapped_generation_correct": swapped_generation_correct,
                    "swapped_lm_logits": swapped.relation_logits.tolist(),
                    "swapped_lm_probs": swapped.relation_probs.tolist(),
                    "swapped_lm_gt_margin": relation_native_margin(swapped.relation_logits, swapped_gt),
                    "generation_pair_status": pair_correctness_status(
                        original_generation_correct, swapped_generation_correct
                    ),
                    "first_step_pair_status": pair_correctness_status(
                        original_first_step_correct, swapped_first_step_correct
                    ),
                    "subject_span_ab": list(original.subject_span),
                    "reference_span_ab": list(original.reference_span),
                    "subject_span_ba": list(swapped.subject_span),
                    "reference_span_ba": list(swapped.reference_span),
                })
                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    print(
                        f"[extract {completed}] sid={sid} gt={gt} "
                        f"AB={original.generated_prediction} BA={swapped.generated_prediction}"
                    )
                del batch_ab, batch_ba, original, swapped
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "extraction",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"[ERROR extraction sid={sid}] {type(exc).__name__}: {exc}")
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available() and (
                    args.empty_cache_every <= 1
                    or completed % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()
        extraction_rows = read_jsonl(extraction_path)

    if not extraction_rows:
        raise RuntimeError("No completed extraction rows")
    valid_rows = [
        row for row in extraction_rows
        if cache_path(cache_dir, int(row["sid"])).exists()
    ]
    if not valid_rows:
        raise RuntimeError("No valid rows with state cache")
    relation_counts = Counter(str(row["gt"]) for row in valid_rows)
    missing_classes = [relation for relation in RELATIONS if relation_counts[relation] == 0]
    if missing_classes:
        raise RuntimeError(f"No valid samples for relation classes: {missing_classes}")
    print(f"Valid samples: {len(valid_rows)}")
    print(f"Relations: {dict(relation_counts)}")
    print(
        "Generation pair status: "
        f"{dict(Counter(str(row['generation_pair_status']) for row in valid_rows))}"
    )

    # -------------------- Phase 2: all-sample cross-fitted validation --------------------
    layer_rows, layer_models = evaluate_crossfit_layer_metrics(
        rows=valid_rows,
        cache_dir=cache_dir,
        selected_layers=selected_layers,
        folds=args.folds,
        metric=args.centroid_metric,
    )
    write_csv(layer_metrics_path, layer_rows)
    intervention_layer = select_intervention_layer(
        args.intervention_layer, selected_layers, layer_rows
    )
    layer_position = selected_layers.index(intervention_layer)
    selected_row = next(row for row in layer_rows if int(row["layer"]) == intervention_layer)
    fold_payload = []
    for fold, bundle in sorted(layer_models[intervention_layer].items()):
        geometry: DirectionGeometry = bundle["role_even_geometry"]
        fold_payload.append({
            "fold": fold,
            "train_loo_d_even_accuracy": bundle["train_loo"],
            "role_even_centroid_counts": bundle["role_even_model"].counts.tolist(),
            "direction_pairwise_cosine": geometry.pairwise_cosine.tolist(),
            "class_prototype_amplitudes": {
                relation: geometry.prototype_amplitude(relation) for relation in RELATIONS
            },
        })
    selected_payload = {
        "layer": intervention_layer,
        "layer_position": layer_position,
        "oof_d_even_accuracy": selected_row["oof_d_even_accuracy"],
        "oof_r_ab_accuracy": selected_row["oof_r_ab_accuracy"],
        "oof_r_ba_aligned_accuracy": selected_row["oof_r_ba_aligned_accuracy"],
        "direction_definition": "normalize(c_r - mean_{q!=r}(c_q))",
        "fold_models": fold_payload,
    }
    selected_layer_path.write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Selected L{intervention_layer}: OOF role-even accuracy="
        f"{safe_float(selected_row['oof_d_even_accuracy']):.4f}"
    )

    # -------------------- Phase 3: causal intervention on every sample --------------------
    intervention_rows: List[Dict[str, Any]] = []
    if not args.skip_interventions:
        sample_rows = list(valid_rows)
        rng = random.Random(args.seed)
        rng.shuffle(sample_rows)
        if args.intervention_max_samples is not None:
            sample_rows = sample_rows[: max(0, int(args.intervention_max_samples))]

        conditions: List[Tuple[str, Optional[float], int]] = []
        for alpha in alphas:
            conditions.append(("remove_gt_direction", float(alpha), 0))
            if args.include_opposite_replacement:
                conditions.append(("replace_with_opposite_direction", float(alpha), 0))
            if args.include_random_control and alpha > 0:
                for random_index in range(args.random_controls_per_spatial):
                    conditions.append(("random_control_remove", float(alpha), random_index))
                    if args.include_opposite_replacement:
                        conditions.append(("random_control_replace", float(alpha), random_index))

        completed = 0
        for row in tqdm(sample_rows, desc=f"intervene-all:L{intervention_layer}"):
            sid = int(row["sid"])
            image: Optional[Image.Image] = None
            try:
                record = records_by_sid[sid]
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                gt = str(row["gt"])
                fold = int(row["group_fold"])
                selected_bundle = layer_models[intervention_layer][fold]
                final_bundle = layer_models[selected_layers[-1]][fold]
                role_even_model: CentroidModel = selected_bundle["role_even_model"]
                r_ab_model: CentroidModel = selected_bundle["r_ab_model"]
                last_model_final: CentroidModel = final_bundle["last_model"]
                role_even_geometry: DirectionGeometry = selected_bundle["role_even_geometry"]
                r_ab_geometry: DirectionGeometry = selected_bundle["r_ab_geometry"]
                last_geometry_final: DirectionGeometry = final_bundle["last_geometry"]

                image = base.record_image(record)
                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=str(prompt_row["question_text"]),
                    device=device,
                )
                cached = load_state_cache(cache_path(cache_dir, sid))
                d_even = np.asarray(cached["d_even"][layer_position], dtype=np.float32)
                r_ab = np.asarray(cached["r_ab"][layer_position], dtype=np.float32)
                r_ba_aligned = np.asarray(
                    cached["r_ba_aligned"][layer_position], dtype=np.float32
                )
                mean_ab = np.asarray(cached["mean_ab"][layer_position], dtype=np.float32)
                baseline_subject_state = mean_ab + 0.5 * r_ab
                baseline_reference_state = mean_ab - 0.5 * r_ab
                baseline_last_final = np.asarray(cached["last_ab"][-1], dtype=np.float32)

                baseline_role_even_pred, _ = role_even_model.predict(d_even[None, :])
                baseline_object_pred, _ = r_ab_model.predict(r_ab[None, :])
                baseline_last_pred, _ = last_model_final.predict(baseline_last_final[None, :])
                baseline_logits = np.asarray(row["baseline_lm_logits"], dtype=np.float32)
                baseline_lm_margin = relation_native_margin(baseline_logits, gt)
                baseline_generation_prediction = row.get("baseline_generation_prediction")

                baseline_role_even_gt_score = gt_direction_score(
                    d_even, gt, role_even_geometry
                )
                baseline_role_even_opp_margin = gt_vs_opposite_margin(
                    d_even, gt, role_even_geometry
                )
                baseline_role_even_ovr_margin = gt_one_vs_rest_margin(
                    d_even, gt, role_even_geometry
                )
                baseline_object_gt_score = gt_direction_score(r_ab, gt, r_ab_geometry)
                baseline_object_opp_margin = gt_vs_opposite_margin(
                    r_ab, gt, r_ab_geometry
                )
                baseline_object_ovr_margin = gt_one_vs_rest_margin(
                    r_ab, gt, r_ab_geometry
                )
                baseline_last_gt_score = gt_direction_score(
                    baseline_last_final, gt, last_geometry_final
                )
                baseline_last_opp_margin = gt_vs_opposite_margin(
                    baseline_last_final, gt, last_geometry_final
                )
                baseline_last_ovr_margin = gt_one_vs_rest_margin(
                    baseline_last_final, gt, last_geometry_final
                )

                for condition, alpha, random_index in conditions:
                    delta_pair, delta_meta = condition_delta(
                        condition=condition,
                        alpha=alpha,
                        d_even=d_even,
                        r_ab=r_ab,
                        gt=gt,
                        geometry=role_even_geometry,
                        random_index=random_index,
                        sid=sid,
                        seed=args.seed,
                    )
                    result = run_intervened_prompt(
                        base=base,
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        decoder_layer_count=len(decoder_layers),
                        selected_layers=selected_layers,
                        intervention_layer=intervention_layer,
                        batch=batch,
                        subject=subject,
                        reference=reference,
                        object_state=args.object_state,
                        delta_pair=delta_pair,
                        relation_token_map=relation_token_map,
                        max_new_tokens=args.max_new_tokens,
                    )
                    patched_pair = np.asarray(result["patched_pair"], dtype=np.float32)
                    patched_role_even = 0.5 * (patched_pair + r_ba_aligned)
                    patched_last_final = np.asarray(
                        result["selected_last"][-1], dtype=np.float32
                    )
                    patched_role_even_pred, _ = role_even_model.predict(
                        patched_role_even[None, :]
                    )
                    patched_object_pred, _ = r_ab_model.predict(patched_pair[None, :])
                    patched_last_pred, _ = last_model_final.predict(
                        patched_last_final[None, :]
                    )
                    patched_lm_margin = relation_native_margin(result["relation_logits"], gt)
                    generated_prediction = result["generated_prediction"]

                    patched_subject_state = baseline_subject_state + 0.5 * delta_pair
                    patched_reference_state = baseline_reference_state - 0.5 * delta_pair
                    subject_identity_cosine = float(cosine_rows(
                        baseline_subject_state[None, :], patched_subject_state[None, :]
                    )[0])
                    reference_identity_cosine = float(cosine_rows(
                        baseline_reference_state[None, :], patched_reference_state[None, :]
                    )[0])

                    output_row = {
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "gt": gt,
                        "opposite": OPPOSITE[gt],
                        "group_fold": fold,
                        "generation_pair_status": row.get("generation_pair_status"),
                        "first_step_pair_status": row.get("first_step_pair_status"),
                        "intervention_layer": intervention_layer,
                        "condition": condition,
                        "alpha": alpha,
                        "random_index": random_index if condition.startswith("random_control") else None,
                        "target_relation": delta_meta.get("target_relation"),
                        "delta_pair_norm": float(np.linalg.norm(delta_pair)),
                        "delta_d_even_norm": delta_meta.get("delta_d_even_norm"),
                        "subject_state_cosine_pre_post": subject_identity_cosine,
                        "reference_state_cosine_pre_post": reference_identity_cosine,
                        "gt_direction_amplitude": delta_meta.get("gt_direction_amplitude"),
                        "opposite_prototype_amplitude": delta_meta.get("opposite_prototype_amplitude"),
                        "target_gt_direction_amplitude": delta_meta.get("target_gt_direction_amplitude"),
                        "target_opposite_direction_amplitude": delta_meta.get(
                            "target_opposite_direction_amplitude"
                        ),
                        "baseline_role_even_prediction": INDEX_TO_RELATION[int(baseline_role_even_pred[0])],
                        "baseline_role_even_gt_direction_score": baseline_role_even_gt_score,
                        "baseline_role_even_gt_vs_opposite_margin": baseline_role_even_opp_margin,
                        "baseline_role_even_gt_one_vs_rest_margin": baseline_role_even_ovr_margin,
                        "patched_role_even_estimate_prediction": INDEX_TO_RELATION[int(patched_role_even_pred[0])],
                        "patched_role_even_gt_direction_score": gt_direction_score(
                            patched_role_even, gt, role_even_geometry
                        ),
                        "patched_role_even_gt_vs_opposite_margin": gt_vs_opposite_margin(
                            patched_role_even, gt, role_even_geometry
                        ),
                        "patched_role_even_gt_one_vs_rest_margin": gt_one_vs_rest_margin(
                            patched_role_even, gt, role_even_geometry
                        ),
                        "baseline_object_prediction": INDEX_TO_RELATION[int(baseline_object_pred[0])],
                        "baseline_object_gt_direction_score": baseline_object_gt_score,
                        "baseline_object_gt_vs_opposite_margin": baseline_object_opp_margin,
                        "baseline_object_gt_one_vs_rest_margin": baseline_object_ovr_margin,
                        "patched_object_prediction": INDEX_TO_RELATION[int(patched_object_pred[0])],
                        "patched_object_gt_direction_score": gt_direction_score(
                            patched_pair, gt, r_ab_geometry
                        ),
                        "patched_object_gt_vs_opposite_margin": gt_vs_opposite_margin(
                            patched_pair, gt, r_ab_geometry
                        ),
                        "patched_object_gt_one_vs_rest_margin": gt_one_vs_rest_margin(
                            patched_pair, gt, r_ab_geometry
                        ),
                        "baseline_final_last_prediction": INDEX_TO_RELATION[int(baseline_last_pred[0])],
                        "baseline_final_last_gt_direction_score": baseline_last_gt_score,
                        "baseline_final_last_gt_vs_opposite_margin": baseline_last_opp_margin,
                        "baseline_final_last_gt_one_vs_rest_margin": baseline_last_ovr_margin,
                        "patched_final_last_prediction": INDEX_TO_RELATION[int(patched_last_pred[0])],
                        "patched_final_last_gt_direction_score": gt_direction_score(
                            patched_last_final, gt, last_geometry_final
                        ),
                        "patched_final_last_gt_vs_opposite_margin": gt_vs_opposite_margin(
                            patched_last_final, gt, last_geometry_final
                        ),
                        "patched_final_last_gt_one_vs_rest_margin": gt_one_vs_rest_margin(
                            patched_last_final, gt, last_geometry_final
                        ),
                        "baseline_lm_prediction": row.get("baseline_relation_prediction"),
                        "baseline_lm_gt_margin": baseline_lm_margin,
                        "patched_lm_prediction": result["relation_prediction"],
                        "patched_lm_gt_margin": patched_lm_margin,
                        "baseline_generation_prediction": baseline_generation_prediction,
                        "baseline_generation_correct": baseline_generation_prediction == gt,
                        "baseline_swapped_generation_prediction": row.get(
                            "swapped_generation_prediction"
                        ),
                        "baseline_swapped_generation_correct": bool(
                            row.get("swapped_generation_correct", False)
                        ),
                        "patched_generated_text": result["generated_text"],
                        "patched_generation_prediction": generated_prediction,
                        "intervention_generation_correct": generated_prediction == gt,
                        "generation_changed_to_opposite": generated_prediction == OPPOSITE[gt],
                    }
                    append_jsonl(interventions_path, output_row)
                    intervention_rows.append(output_row)

                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    print(
                        f"[intervene {completed}/{len(sample_rows)}] sid={sid} gt={gt} "
                        f"status={row.get('generation_pair_status')} "
                        f"baseline={baseline_generation_prediction}"
                    )
                del batch, cached
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "intervention",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"[ERROR intervention sid={sid}] {type(exc).__name__}: {exc}")
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available() and (
                    args.empty_cache_every <= 1
                    or completed % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()
    else:
        intervention_rows = read_jsonl(interventions_path)

    intervention_summary = summarize_interventions(intervention_rows)
    write_csv(intervention_summary_path, intervention_summary)
    report = build_report(
        args=args,
        extraction_rows=valid_rows,
        layer_rows=layer_rows,
        intervention_layer=intervention_layer,
        intervention_summary=intervention_summary,
    )
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
