#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three mechanistic experiments for the centroid-to-generation spatial gap.

The script keeps the original two-object spatial-relation task and measures three
separate stages:

Experiment 1 -- sparse head storage vs. attention mass
------------------------------------------------------
For every selected decoder layer and attention head, reconstruct the head's
visual contribution to the subject and reference object tokens:

    u_vis->sub^(l,h) = sum_{s in visual} A[sub,s] V_s
    u_vis->ref^(l,h) = sum_{s in visual} A[ref,s] V_s
    r_head^(l,h)     = u_vis->sub^(l,h) - u_vis->ref^(l,h)

The vectors are kept in the head-local pre-o_proj space [Dh]. A cross-fitted
four-class centroid classifier measures how much spatial relation information is
stored in each head. Separately, the script computes each head's visual-attention
centroid accuracy and mean visual-attention mass. High-centroid heads are matched
against low-storage heads from the same layer with similar attention mass.

Experiment 2 -- object-to-last transport
----------------------------------------
For each head, reconstruct the actual contribution from the two object text spans
to prompt_last:

    c_obj->last^(l,h)
        = [sum_{s in subject U reference} A[last,s] V_s] W_O^(h)

The script measures:
  * correlation between the head's stored spatial margin and final-last margin;
  * alignment of c_obj->last with a cross-fitted final-last relation direction;
  * object-pair/last representational similarity (CKA, RSA, margin correlation);
  * causal ablation of top-storage object->last head edges versus
    attention-matched control heads.

Experiment 3 -- object x question utilization at last
------------------------------------------------------
At selected layers, prompt_last attention output is decomposed into:

    object contribution: subject + reference source tokens
    question contribution: one selected token group
    rest: all remaining attention/residual content

For each question group q, four forward states are evaluated by subtracting the
corresponding baseline attention contributions before the block MLP:

    F(full), F(no_object), F(no_q), F(no_object_no_q)

The non-additive interaction is:

    G = F(full) - F(no_object) - F(no_q) + F(no_object_no_q)

G>0 means the question-token contribution helps the model utilize object
evidence; G<0 means it suppresses or interferes with object evidence. The block
MLP and all later layers are recomputed naturally after every intervention.

Important scope
---------------
* All valid samples are extracted, regardless of whether original/swapped
  questions are correct. Results are stratified into both_correct,
  original_only, swapped_only, and both_wrong.
* Centroids are cross-fitted by unordered object-pair folds.
* This script does not assume a pure, role-invariant spatial vector.
* Attention replay currently expects Qwen/Llama-style self-attention modules
  exposing v_proj and o_proj and should be run with eager attention.

Required files in the same repository directory
------------------------------------------------
    analyze_coco_centroid_generation_step1_v4.py
    analyze_coco_flip_same_token_similarity_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py
    extract_two_object_relation_states.py

Main outputs
------------
    config.json
    extraction.jsonl
    cache/<sid>.npz
    head_metrics.csv
    top_heads.json
    top_head_group_metrics.csv
    object_last_similarity.csv
    head_ablation.jsonl
    head_ablation_summary.csv
    factorial.jsonl
    factorial_summary.csv
    report.txt
    errors.jsonl

Example
-------
CUDA_VISIBLE_DEVICES=0 python analyze_spatial_storage_transport_utilization_v2.py \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --model llava-7b \
  --device cuda:0 \
  --layers all \
  --trace-layer-chunk 4 \
  --top-heads 8 \
  --factorial-layers auto \
  --factorial-groups query_words,relation,options,instruction \
  --output-dir output/spatial_storage_transport_utilization/coco/llava-7b \
  --overwrite
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


SCRIPT_VERSION = "spatial-storage-transport-utilization-v2"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
ID_TO_REL = {index: name for name, index in REL_TO_ID.items()}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
PAIR_STATUSES = ("both_correct", "original_only", "swapped_only", "both_wrong")
QUESTION_GROUPS = ("query_words", "relation", "options", "instruction", "question_other")


# -----------------------------------------------------------------------------
# CLI / files / numerics
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "Model key from the merged repository model registry, for example "
            "'llava-7b' or 'qwen-3b'. Run one model per process/output directory."
        ),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager"])
    p.add_argument("--layers", default="all")
    p.add_argument("--trace-layer-chunk", type=int, default=4)
    p.add_argument("--object-state", choices=["last", "mean"], default="last")
    p.add_argument("--centroid-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--array-dtype", choices=["float16", "float32"], default="float16")

    p.add_argument("--top-heads", type=int, default=8)
    p.add_argument(
        "--head-rank-metric",
        choices=["content_centroid_accuracy", "attention_centroid_accuracy"],
        default="content_centroid_accuracy",
    )
    p.add_argument(
        "--head-ablation-max-samples",
        type=int,
        default=None,
        help="Default evaluates every valid sample.",
    )
    p.add_argument(
        "--generate-causal-unions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate only for top-head union and matched-control union conditions.",
    )

    p.add_argument(
        "--factorial-layers",
        default="auto",
        help="'auto', or comma-separated decoder layers.",
    )
    p.add_argument("--factorial-auto-layers", type=int, default=2)
    p.add_argument(
        "--factorial-groups",
        default="query_words,relation,options,instruction",
    )
    p.add_argument(
        "--factorial-max-samples",
        type=int,
        default=None,
        help="Default evaluates every valid sample; use a cap for a smoke test.",
    )
    p.add_argument(
        "--factorial-generate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generation for every factorial condition is expensive; LM margin is always computed.",
    )

    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument(
        "--semantic-helper",
        default="analyze_coco_flip_same_token_similarity_v1.py",
    )
    p.add_argument(
        "--attention-helper",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--skip-head-ablation", action="store_true")
    p.add_argument("--skip-factorial", action="store_true")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=10)
    p.add_argument("--replay-tolerance", type=float, default=5e-3)
    return p.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    """Import a Python file under a stable module name.

    The module must be inserted into ``sys.modules`` before ``exec_module``.
    Python 3.11's ``dataclasses`` implementation looks up
    ``sys.modules[cls.__module__]`` while processing ``@dataclass``.  Without
    this registration, dynamically imported helper files containing
    dataclasses fail with ``AttributeError: 'NoneType' object has no attribute
    '__dict__'``.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")

    # Remove a stale module with the same synthetic name, then register the new
    # object before executing it.  This also makes postponed annotations and
    # dataclass type inspection work correctly.
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if len(array) > 1 else float("nan")


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def macro_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for relation_id in range(len(RELATIONS)):
        mask = labels == relation_id
        if np.any(mask):
            values.append(float(np.mean(pred[mask] == labels[mask])))
    return float(np.mean(values)) if values else float("nan")


def stable_pair_fold(subject: str, reference: str, folds: int, seed: int) -> int:
    pair = "||".join(sorted([str(subject).strip().lower(), str(reference).strip().lower()]))
    digest = hashlib.sha256(f"{seed}::{pair}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(folds)


def pair_status(original_correct: bool, swapped_correct: bool) -> str:
    if original_correct and swapped_correct:
        return "both_correct"
    if original_correct:
        return "original_only"
    if swapped_correct:
        return "swapped_only"
    return "both_wrong"


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def relation_margin_from_logits(logits: Sequence[float], gt: str) -> float:
    array = np.asarray(logits, dtype=np.float64)
    return float(array[REL_TO_ID[gt]] - array[REL_TO_ID[OPPOSITE[gt]]])


def relation_prediction_from_logits(logits: Sequence[float]) -> str:
    return ID_TO_REL[int(np.argmax(np.asarray(logits, dtype=np.float64)))]


def parse_layer_list(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside [0,{n_layers - 1}]")
        out.append(layer)
    if not out:
        raise ValueError("Layer list is empty")
    return sorted(set(out))


def parse_groups(value: str) -> List[str]:
    out: List[str] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if item not in QUESTION_GROUPS:
            raise ValueError(f"Unknown factorial group {item}; allowed={QUESTION_GROUPS}")
        out.append(item)
    if not out:
        raise ValueError("No factorial groups selected")
    return list(dict.fromkeys(out))


def cache_file(cache_dir: Path, sid: int) -> Path:
    return cache_dir / f"{int(sid)}.npz"


def span_positions(span: Sequence[int], mode: str) -> List[int]:
    start, end = int(span[0]), int(span[1])
    if mode == "last":
        return [end]
    return list(range(start, end + 1))


def full_span_positions(span: Sequence[int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


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
    raise TypeError(f"Cannot locate 3D hidden tensor in {type(output).__name__}")


def replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        values = list(output)
        for index, item in enumerate(values):
            if torch.is_tensor(item) and item.ndim == 3:
                values[index] = tensor
                return tuple(values)
    if isinstance(output, list):
        values = list(output)
        for index, item in enumerate(values):
            if torch.is_tensor(item) and item.ndim == 3:
                values[index] = tensor
                return values
    raise TypeError(f"Cannot replace hidden tensor in {type(output).__name__}")


# -----------------------------------------------------------------------------
# Centroid models and cross-fitting
# -----------------------------------------------------------------------------


@dataclass
class CentroidModel:
    centroids: np.ndarray  # [4,D]
    counts: np.ndarray     # [4]
    metric: str

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        c = np.asarray(self.centroids, dtype=np.float64)
        if self.metric == "cosine":
            return normalize_rows(x) @ normalize_rows(c).T
        diff = x[:, None, :] - c[None, :, :]
        return -np.sum(diff * diff, axis=-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(x), axis=1).astype(np.int64)

    def gt_opposite_margin(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        scores = self.scores(x)
        out = np.empty((len(x),), dtype=np.float64)
        for index, label in enumerate(labels):
            gt = ID_TO_REL[int(label)]
            out[index] = scores[index, int(label)] - scores[index, REL_TO_ID[OPPOSITE[gt]]]
        return out

    def sample_direction(self, label: int) -> np.ndarray:
        gt = ID_TO_REL[int(label)]
        vector = self.centroids[int(label)] - self.centroids[REL_TO_ID[OPPOSITE[gt]]]
        norm = float(np.linalg.norm(vector))
        return (vector / max(norm, 1e-12)).astype(np.float32)


def fit_centroids(x: np.ndarray, labels: np.ndarray, metric: str) -> CentroidModel:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centroids = []
    counts = []
    for relation_id in range(len(RELATIONS)):
        mask = labels == relation_id
        if not np.any(mask):
            raise RuntimeError(f"Missing class {ID_TO_REL[relation_id]} in centroid training")
        centroids.append(x[mask].mean(axis=0))
        counts.append(int(mask.sum()))
    return CentroidModel(
        centroids=np.stack(centroids, axis=0).astype(np.float32),
        counts=np.asarray(counts, dtype=np.int64),
        metric=str(metric),
    )


def crossfit_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, CentroidModel]]:
    x = np.asarray(x)
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    pred = np.full((len(x),), -1, dtype=np.int64)
    margin = np.full((len(x),), np.nan, dtype=np.float64)
    models: Dict[int, CentroidModel] = {}
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        model = fit_centroids(x[train], labels[train], metric)
        models[int(fold)] = model
        pred[test] = model.predict(x[test])
        margin[test] = model.gt_opposite_margin(x[test], labels[test])
    return pred, margin, models


def sample_directions_from_models(
    labels: np.ndarray,
    folds: np.ndarray,
    models: Mapping[int, CentroidModel],
) -> np.ndarray:
    vectors = []
    for label, fold in zip(labels.tolist(), folds.tolist()):
        vectors.append(models[int(fold)].sample_direction(int(label)))
    return np.stack(vectors, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Attention contribution calculations
# -----------------------------------------------------------------------------


def target_local_indices(trace: Any, positions: Sequence[int]) -> List[int]:
    lookup = {int(position): index for index, position in enumerate(trace.target_positions)}
    out = [lookup[int(position)] for position in positions if int(position) in lookup]
    if not out:
        raise RuntimeError(f"None of target positions {list(positions)} were traced")
    return out


def average_target_weights(trace: Any, target_positions: Sequence[int]) -> torch.Tensor:
    local = target_local_indices(trace, target_positions)
    return trace.attention_weights[:, local, :].mean(dim=1)  # [H,S]


def average_block_state(trace: Any, positions: Sequence[int]) -> torch.Tensor:
    local = target_local_indices(trace, positions)
    return trace.block_output[local].mean(dim=0)  # [D]


def average_attention_output(trace: Any, positions: Sequence[int]) -> torch.Tensor:
    local = target_local_indices(trace, positions)
    return trace.attention_output[local].mean(dim=0)  # [D]


def head_source_pre_vectors(
    trace: Any,
    target_positions: Sequence[int],
    source_positions: Sequence[int],
) -> torch.Tensor:
    if not source_positions:
        return torch.zeros(
            (trace.attention_weights.shape[0], trace.value_states.shape[-1]),
            dtype=torch.float32,
        )
    weights = average_target_weights(trace, target_positions)  # [H,S]
    source = torch.tensor(sorted(set(map(int, source_positions))), dtype=torch.long)
    selected_weights = weights.index_select(1, source)
    selected_values = trace.value_states.index_select(1, source)
    return torch.einsum("hs,hsd->hd", selected_weights, selected_values)


def projected_head_vectors(attn_helper: Any, trace: Any, pre_vectors: torch.Tensor) -> torch.Tensor:
    return attn_helper.project_heads(pre_vectors.float(), trace.o_proj_weight.float())


def visual_attention_centroids(
    trace: Any,
    target_positions: Sequence[int],
    visual_positions: Sequence[int],
    visual_xy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = average_target_weights(trace, target_positions).float()  # [H,S]
    visual = torch.tensor(list(map(int, visual_positions)), dtype=torch.long)
    values = weights.index_select(1, visual).clamp_min(0.0)  # [H,V]
    mass = values.sum(dim=1)
    normalized = values / mass[:, None].clamp_min(1e-12)
    centroids = normalized @ visual_xy.float()
    return centroids, mass


def relation_from_xy(subject_xy: np.ndarray, reference_xy: np.ndarray) -> str:
    dx = float(subject_xy[0] - reference_xy[0])
    dy = float(subject_xy[1] - reference_xy[1])
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "above" if dy < 0 else "below"


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)


# -----------------------------------------------------------------------------
# Semantic source groups
# -----------------------------------------------------------------------------


def locate_option_synonyms(
    semantic_helper: Any,
    tokenizer: Any,
    input_ids: Sequence[int],
    reference_span: Sequence[int],
    question_end: int,
) -> List[int]:
    positions: List[int] = []
    for word in ("left", "right", "above", "below", "under", "beneath", "on"):
        spans = semantic_helper.locate_phrase_spans(tokenizer, input_ids, word)
        span = semantic_helper.choose_span(
            spans,
            min_start=int(reference_span[1]) + 1,
            max_end=int(question_end),
            prefer="first",
        )
        if span is not None:
            positions.extend(range(int(span[0]), int(span[1]) + 1))
    return sorted(set(positions))


def build_question_groups(
    *,
    semantic_helper: Any,
    tokenizer: Any,
    input_ids: Sequence[int],
    question_text: str,
    subject_span: Sequence[int],
    reference_span: Sequence[int],
    visual_positions: Sequence[int],
) -> Dict[str, List[int]]:
    visual_set = set(map(int, visual_positions))
    text_positions = [index for index in range(len(input_ids)) if index not in visual_set]
    semantic = semantic_helper.locate_semantic_spans(
        tokenizer,
        input_ids,
        question_text,
        (int(subject_span[0]), int(subject_span[1])),
        (int(reference_span[0]), int(reference_span[1])),
        text_positions,
    )
    q_span = list(semantic.get("_question_span", []))
    question_end = int(q_span[1]) if len(q_span) == 2 else max(text_positions)
    options = sorted(set(map(int, semantic.get("option_all", []))))
    options = sorted(set(options + locate_option_synonyms(
        semantic_helper,
        tokenizer,
        input_ids,
        reference_span,
        question_end,
    )))
    relation = sorted(set(
        list(map(int, semantic.get("relation_connector", [])))
        + list(map(int, semantic.get("relation_keyword", [])))
        + list(map(int, semantic.get("connector_to", [])))
    ))
    query_words = sorted(set(
        list(map(int, semantic.get("where", [])))
        + list(map(int, semantic.get("copula", [])))
    ))
    instruction = sorted(set(map(int, semantic.get("answer_instruction", []))) - set(options))
    question_other = sorted(set(map(int, semantic.get("question_other", []))))

    excluded = set(full_span_positions(subject_span)) | set(full_span_positions(reference_span))
    excluded.add(len(input_ids) - 1)
    groups = {
        "query_words": query_words,
        "relation": relation,
        "options": options,
        "instruction": instruction,
        "question_other": question_other,
    }
    return {
        name: sorted(position for position in set(positions) if position not in excluded)
        for name, positions in groups.items()
    }


# -----------------------------------------------------------------------------
# Extraction cache
# -----------------------------------------------------------------------------


def save_sample_cache(path: Path, arrays: Mapping[str, np.ndarray], dtype: str) -> None:
    payload: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float16 if dtype == "float16" else np.float32)
        payload[key] = array
    np.savez_compressed(path, **payload)


def load_sample_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def trace_prompt_chunks(
    *,
    attention_helper: Any,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    target_positions: Sequence[int],
    chunk_size: int,
) -> Tuple[Dict[str, Any], Dict[int, Any]]:
    all_traces: Dict[int, Any] = {}
    baseline: Optional[Dict[str, Any]] = None
    for start in range(0, len(layers), max(1, int(chunk_size))):
        chunk = list(layers[start : start + max(1, int(chunk_size))])
        result, traces = attention_helper.run_and_trace(
            model=model,
            batch=batch,
            token_map=relation_token_map,
            decoder_layers=decoder_layers,
            layer_indices=chunk,
            target_positions=target_positions,
        )
        if baseline is None:
            baseline = result
        all_traces.update(traces)
    if baseline is None:
        raise RuntimeError("No attention trace was produced")
    return baseline, all_traces


def extract_one_sample(
    *,
    base: Any,
    semantic_helper: Any,
    attention_helper: Any,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    relation_token_map: Mapping[str, Sequence[int]],
    batch: Mapping[str, Any],
    question_text: str,
    subject: str,
    reference: str,
    gt: str,
    object_state: str,
    trace_layer_chunk: int,
    replay_tolerance: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = base.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_targets = span_positions(subject_span, object_state)
    reference_targets = span_positions(reference_span, object_state)
    subject_sources = full_span_positions(subject_span)
    reference_sources = full_span_positions(reference_span)
    prompt_last = len(input_ids) - 1
    target_positions = sorted(set(subject_targets + reference_targets + [prompt_last]))
    visual_positions = base.resolve_visual_indices(model, processor, dict(batch), input_ids)
    xy = base.visual_coordinates(
        model,
        dict(batch),
        len(visual_positions),
        batch["input_ids"].device,
    )
    if xy is None:
        raise RuntimeError(f"Could not infer visual grid for {len(visual_positions)} tokens")
    xy_cpu = xy.detach().float().cpu()
    groups = build_question_groups(
        semantic_helper=semantic_helper,
        tokenizer=processor.tokenizer,
        input_ids=input_ids,
        question_text=question_text,
        subject_span=subject_span,
        reference_span=reference_span,
        visual_positions=visual_positions,
    )

    baseline, traces = trace_prompt_chunks(
        attention_helper=attention_helper,
        model=model,
        batch=batch,
        relation_token_map=relation_token_map,
        decoder_layers=decoder_layers,
        layers=layers,
        target_positions=target_positions,
        chunk_size=trace_layer_chunk,
    )

    object_pairs: List[np.ndarray] = []
    last_states: List[np.ndarray] = []
    head_visual_pairs: List[np.ndarray] = []
    head_object_to_last: List[np.ndarray] = []
    attention_predictions: List[np.ndarray] = []
    visual_masses: List[np.ndarray] = []
    visual_centroid_margins: List[np.ndarray] = []
    head_objlast_projected_norm: List[np.ndarray] = []
    head_objlast_total_attention_cos: List[np.ndarray] = []
    object_total_attention_cos: List[float] = []
    object_total_attention_norm_ratio: List[float] = []
    replay_errors: List[float] = []

    for layer in layers:
        trace = traces[int(layer)]
        replay_errors.append(float(trace.replay_relative_error))
        if float(trace.replay_relative_error) > float(replay_tolerance):
            # Kept in metadata; do not silently discard the sample.
            pass

        subject_state = average_block_state(trace, subject_targets)
        reference_state = average_block_state(trace, reference_targets)
        last_state = average_block_state(trace, [prompt_last])
        object_pairs.append((subject_state - reference_state).numpy().astype(np.float32))
        last_states.append(last_state.numpy().astype(np.float32))

        sub_visual_pre = head_source_pre_vectors(trace, subject_targets, visual_positions)
        ref_visual_pre = head_source_pre_vectors(trace, reference_targets, visual_positions)
        visual_pair_pre = sub_visual_pre - ref_visual_pre
        head_visual_pairs.append(visual_pair_pre.numpy().astype(np.float32))

        object_sources = sorted(set(subject_sources + reference_sources))
        object_last_pre = head_source_pre_vectors(trace, [prompt_last], object_sources)
        head_object_to_last.append(object_last_pre.numpy().astype(np.float32))
        projected_object_heads = projected_head_vectors(attention_helper, trace, object_last_pre)
        projected_object_total = projected_object_heads.sum(dim=0)
        total_attention = average_attention_output(trace, [prompt_last])
        head_objlast_projected_norm.append(
            projected_object_heads.norm(dim=-1).numpy().astype(np.float32)
        )
        head_objlast_total_attention_cos.append(
            cosine_rows(projected_object_heads, total_attention[None, :].expand_as(projected_object_heads))
            .numpy()
            .astype(np.float32)
        )
        object_total_attention_cos.append(float(torch.nn.functional.cosine_similarity(
            projected_object_total[None, :], total_attention[None, :], dim=-1, eps=1e-12
        )[0]))
        object_total_attention_norm_ratio.append(float(
            projected_object_total.norm() / total_attention.norm().clamp_min(1e-12)
        ))

        sub_xy, sub_mass = visual_attention_centroids(
            trace, subject_targets, visual_positions, xy_cpu
        )
        ref_xy, ref_mass = visual_attention_centroids(
            trace, reference_targets, visual_positions, xy_cpu
        )
        predictions = []
        margins = []
        for head in range(int(sub_xy.shape[0])):
            subject_point = sub_xy[head].numpy()
            reference_point = ref_xy[head].numpy()
            prediction = relation_from_xy(subject_point, reference_point)
            predictions.append(REL_TO_ID[prediction])
            dx = float(subject_point[0] - reference_point[0])
            dy = float(subject_point[1] - reference_point[1])
            if gt == "left":
                margins.append(-dx)
            elif gt == "right":
                margins.append(dx)
            elif gt == "above":
                margins.append(-dy)
            else:
                margins.append(dy)
        attention_predictions.append(np.asarray(predictions, dtype=np.int8))
        visual_centroid_margins.append(np.asarray(margins, dtype=np.float32))
        visual_masses.append((0.5 * (sub_mass + ref_mass)).numpy().astype(np.float32))

    arrays = {
        "layers": np.asarray(layers, dtype=np.int16),
        "object_pair": np.stack(object_pairs, axis=0),             # [L,D]
        "last_state": np.stack(last_states, axis=0),               # [L,D]
        "head_visual_pair_pre": np.stack(head_visual_pairs, axis=0),  # [L,H,Dh]
        "head_object_to_last_pre": np.stack(head_object_to_last, axis=0),
        "attention_centroid_pred": np.stack(attention_predictions, axis=0),
        "attention_centroid_gt_margin": np.stack(visual_centroid_margins, axis=0),
        "visual_attention_mass": np.stack(visual_masses, axis=0),
        "head_objlast_projected_norm": np.stack(head_objlast_projected_norm, axis=0),
        "head_objlast_total_attention_cos": np.stack(head_objlast_total_attention_cos, axis=0),
        "object_total_attention_cos": np.asarray(object_total_attention_cos, dtype=np.float32),
        "object_total_attention_norm_ratio": np.asarray(object_total_attention_norm_ratio, dtype=np.float32),
        "replay_relative_error": np.asarray(replay_errors, dtype=np.float32),
    }
    metadata = {
        "subject_span": list(map(int, subject_span)),
        "reference_span": list(map(int, reference_span)),
        "prompt_last": int(prompt_last),
        "visual_positions": list(map(int, visual_positions)),
        "question_groups": {key: list(map(int, value)) for key, value in groups.items()},
        "relation_logits": [float(baseline["scores"][relation]) for relation in RELATIONS],
        "relation_prediction": str(baseline["prediction"]),
        "max_replay_relative_error": float(np.max(arrays["replay_relative_error"])),
    }
    return arrays, metadata


# -----------------------------------------------------------------------------
# Shared-representation metrics
# -----------------------------------------------------------------------------


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    numerator = float(np.sum(k * l))
    denominator = math.sqrt(float(np.sum(k * k)) * float(np.sum(l * l)))
    return numerator / max(denominator, 1e-12)


def rsa_cosine_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = normalize_rows(np.asarray(x, dtype=np.float64))
    y = normalize_rows(np.asarray(y, dtype=np.float64))
    kx = x @ x.T
    ky = y @ y.T
    upper = np.triu_indices(len(x), k=1)
    return safe_corr(kx[upper], ky[upper])


# -----------------------------------------------------------------------------
# Causal attention-output subtraction
# -----------------------------------------------------------------------------


class AttentionTargetSubtraction:
    """Subtract a fixed residual-space vector from one target in attention output."""

    def __init__(self, attention_module: torch.nn.Module, target_position: int, vector: np.ndarray):
        self.target_position = int(target_position)
        self.vector = np.asarray(vector, dtype=np.float32)
        self.applied = False
        self.handle = attention_module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        hidden = first_tensor(output)
        if int(hidden.shape[1]) <= self.target_position:
            return output
        vector = torch.as_tensor(self.vector, device=hidden.device, dtype=hidden.dtype)
        modified = hidden.clone()
        modified[:, self.target_position, :] = modified[:, self.target_position, :] - vector
        self.applied = True
        return replace_first_tensor(output, modified)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self) -> "AttentionTargetSubtraction":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@torch.inference_mode()
def run_attention_subtraction(
    *,
    base: Any,
    attention_helper: Any,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    layer: int,
    target_position: int,
    vector: np.ndarray,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    final_layer: int,
    generate: bool,
    max_new_tokens: int,
) -> Dict[str, Any]:
    attention_module = attention_helper.resolve_self_attention(decoder_layers[int(layer)])
    with AttentionTargetSubtraction(attention_module, target_position, vector) as intervention:
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if not intervention.applied:
            raise RuntimeError(f"Attention subtraction did not fire at L{layer}")
        hidden_layers = base.output_hidden_layers(outputs, len(decoder_layers))
        final_last = hidden_layers[int(final_layer)][0, int(target_position)].detach().float().cpu().numpy()
        relation_data = base.relation_scores(
            outputs.logits[0, -1], relation_token_map, gt=None
        )
    generated_text = None
    generated_prediction = None
    if generate:
        with AttentionTargetSubtraction(attention_module, target_position, vector) as generation_intervention:
            generated_text = base.generate_text(
                model,
                processor,
                dict(batch),
                max_new_tokens=max_new_tokens,
            )
            if not generation_intervention.applied:
                raise RuntimeError(f"Generation attention subtraction did not fire at L{layer}")
        generated_prediction = base.normalize_relation(generated_text)
    return {
        "final_last": final_last.astype(np.float32),
        "relation_logits": np.asarray(relation_data["logits"], dtype=np.float32),
        "relation_prediction": relation_data["prediction"],
        "generated_text": generated_text,
        "generated_prediction": generated_prediction,
    }


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_causal_rows(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {field: value for field, value in zip(key_fields, key)}
        record.update({
            "N": len(group),
            "mean_delta_final_last_margin": safe_mean(row["delta_final_last_margin"] for row in group),
            "mean_delta_lm_margin": safe_mean(row["delta_lm_margin"] for row in group),
            "prediction_change_rate": safe_mean(float(bool(row["prediction_changed"])) for row in group),
            "correct_to_wrong_rate": safe_mean(float(bool(row["correct_to_wrong"])) for row in group),
            "correct_to_opposite_rate": safe_mean(float(bool(row["correct_to_opposite"])) for row in group),
            "generation_change_rate": safe_mean(
                float(bool(row.get("generation_changed", False)))
                for row in group if row.get("generation_available", False)
            ),
            "generation_correct_to_opposite_rate": safe_mean(
                float(bool(row.get("generation_correct_to_opposite", False)))
                for row in group if row.get("generation_available", False)
            ),
        })
        output.append(record)
    return output


def summarize_factorial(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["layer"], row["question_group"], "all", "all")
        groups[key].append(row)
        key_status = (row["layer"], row["question_group"], "generation_pair_status", row["generation_pair_status"])
        groups[key_status].append(row)
    out = []
    for (layer, question_group, group_type, group_value), group in sorted(groups.items()):
        out.append({
            "layer": layer,
            "question_group": question_group,
            "group_type": group_type,
            "group_value": group_value,
            "N": len(group),
            "mean_interaction_final_last": safe_mean(row["interaction_final_last_margin"] for row in group),
            "mean_interaction_lm": safe_mean(row["interaction_lm_margin"] for row in group),
            "mean_object_effect_with_question_lm": safe_mean(row["object_effect_with_question_lm"] for row in group),
            "mean_object_effect_without_question_lm": safe_mean(row["object_effect_without_question_lm"] for row in group),
            "mean_question_effect_with_object_lm": safe_mean(row["question_effect_with_object_lm"] for row in group),
            "mean_question_effect_without_object_lm": safe_mean(row["question_effect_without_object_lm"] for row in group),
            "positive_interaction_rate_lm": safe_mean(float(row["interaction_lm_margin"] > 0) for row in group),
            "negative_interaction_rate_lm": safe_mean(float(row["interaction_lm_margin"] < 0) for row in group),
        })
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >=2")
    if args.trace_layer_chunk < 1:
        raise ValueError("--trace-layer-chunk must be >=1")
    if args.top_heads < 1:
        raise ValueError("--top-heads must be >=1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = output_dir / "extraction.jsonl"
    errors_path = output_dir / "errors.jsonl"

    base = import_file(Path(args.base_script), "storage_transport_base")
    semantic_helper = import_file(Path(args.semantic_helper), "storage_transport_semantic")
    attention_helper = import_file(Path(args.attention_helper), "storage_transport_attention")
    two_object = base.import_two_object_module()

    records, audit = two_object.load_records(args.dataset, Path(args.data_root), args.max_samples)
    if not records:
        raise RuntimeError("No usable records")
    prompt_path = Path(args.prompt_jsonl) if args.prompt_jsonl else base.resolve_prompt_path(args)
    prompt_rows = base.load_standard_prompts(prompt_path)
    records_by_sid = {int(record.sid): record for record in records}

    specs = base.merged_model_specs(two_object)
    if args.model not in specs:
        raise ValueError(f"Unknown model {args.model}; available={sorted(specs)}")
    spec = specs[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers {transformers.__version__} lacks {spec.model_class}")
    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }
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
    selected_layers = parse_layer_list(args.layers, len(decoder_layers))
    final_layer = len(decoder_layers) - 1
    if final_layer not in selected_layers:
        selected_layers = sorted(set(selected_layers + [final_layer]))
    relation_token_map = base.relation_token_variants(processor.tokenizer)
    factorial_groups = parse_groups(args.factorial_groups)

    config = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "prompt_jsonl": str(prompt_path),
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "selected_layers": selected_layers,
        "folds": args.folds,
        "centroid_metric": args.centroid_metric,
        "top_heads": args.top_heads,
        "factorial_groups": factorial_groups,
        "audit": audit,
        "transformers_version": transformers.__version__,
    }
    write_json(output_dir / "config.json", config)

    # ------------------------------------------------------------------
    # Phase 1: extract all samples
    # ------------------------------------------------------------------
    extraction_rows = read_jsonl(extraction_path)
    if args.skip_extraction:
        if not extraction_rows:
            raise RuntimeError("--skip-extraction but extraction.jsonl is empty")
        missing = [row["sid"] for row in extraction_rows if not cache_file(cache_dir, int(row["sid"])).exists()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} cache files; first={missing[:10]}")
    else:
        if extraction_rows and not args.overwrite:
            raise RuntimeError("Extraction exists; use --overwrite or --skip-extraction")
        completed = 0
        for record in tqdm(records, desc=f"extract-three-experiments:{args.model}"):
            sid = int(record.sid)
            image: Optional[Image.Image] = None
            try:
                if sid not in prompt_rows:
                    raise KeyError(f"Prompt missing sid={sid}")
                prompt = prompt_rows[sid]
                subject = str(prompt["subject"])
                reference = str(prompt["reference"])
                question = str(prompt["question_text"])
                gt = base.normalize_relation(prompt["answer_raw"])
                if gt not in REL_TO_ID:
                    raise ValueError(f"Unsupported relation {gt!r}")
                swapped_gt = OPPOSITE[gt]
                swapped_question = base.build_swapped_question(subject, reference)
                image = base.record_image(record)
                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                arrays, metadata = extract_one_sample(
                    base=base,
                    semantic_helper=semantic_helper,
                    attention_helper=attention_helper,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    layers=selected_layers,
                    relation_token_map=relation_token_map,
                    batch=batch,
                    question_text=question,
                    subject=subject,
                    reference=reference,
                    gt=gt,
                    object_state=args.object_state,
                    trace_layer_chunk=args.trace_layer_chunk,
                    replay_tolerance=args.replay_tolerance,
                )
                save_sample_cache(cache_file(cache_dir, sid), arrays, args.array_dtype)

                generated_text = base.generate_text(
                    model,
                    processor,
                    dict(batch),
                    max_new_tokens=args.max_new_tokens,
                )
                generated_prediction = base.normalize_relation(generated_text)

                swapped_batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=swapped_question,
                    device=device,
                )
                swapped_outputs = model(
                    **swapped_batch,
                    use_cache=False,
                    return_dict=True,
                )
                swapped_relation = base.relation_scores(
                    swapped_outputs.logits[0, -1], relation_token_map, gt=None
                )
                swapped_generated_text = base.generate_text(
                    model,
                    processor,
                    dict(swapped_batch),
                    max_new_tokens=args.max_new_tokens,
                )
                swapped_generated_prediction = base.normalize_relation(swapped_generated_text)
                fold = stable_pair_fold(subject, reference, args.folds, args.seed)
                original_correct = generated_prediction == gt
                swapped_correct = swapped_generated_prediction == swapped_gt
                row = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "question_text": question,
                    "swapped_question": swapped_question,
                    "gt": gt,
                    "swapped_gt": swapped_gt,
                    "label": REL_TO_ID[gt],
                    "fold": fold,
                    "baseline_relation_logits": metadata["relation_logits"],
                    "baseline_relation_prediction": metadata["relation_prediction"],
                    "baseline_lm_margin": relation_margin_from_logits(metadata["relation_logits"], gt),
                    "baseline_generated_text": generated_text,
                    "baseline_generated_prediction": generated_prediction,
                    "baseline_generation_correct": original_correct,
                    "swapped_relation_logits": [float(x) for x in swapped_relation["logits"]],
                    "swapped_relation_prediction": swapped_relation["prediction"],
                    "swapped_generated_text": swapped_generated_text,
                    "swapped_generated_prediction": swapped_generated_prediction,
                    "swapped_generation_correct": swapped_correct,
                    "generation_pair_status": pair_status(original_correct, swapped_correct),
                    "subject_span": metadata["subject_span"],
                    "reference_span": metadata["reference_span"],
                    "prompt_last": metadata["prompt_last"],
                    "visual_positions": metadata["visual_positions"],
                    "question_groups": metadata["question_groups"],
                    "max_replay_relative_error": metadata["max_replay_relative_error"],
                }
                append_jsonl(extraction_path, row)
                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    print(
                        f"[extract {completed}] sid={sid} gt={gt} "
                        f"pred={generated_prediction} swapped={swapped_generated_prediction}"
                    )
                del batch, swapped_batch, swapped_outputs, arrays
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
                    or max(1, completed) % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()
        extraction_rows = read_jsonl(extraction_path)

    valid_rows = [
        row for row in extraction_rows
        if int(row["sid"]) in records_by_sid and cache_file(cache_dir, int(row["sid"])).exists()
    ]
    if not valid_rows:
        raise RuntimeError("No valid extracted samples")
    valid_rows.sort(key=lambda row: int(row["sid"]))
    print(f"Valid samples: {len(valid_rows)}")
    print("Pair statuses:", dict(Counter(row["generation_pair_status"] for row in valid_rows)))

    # Stack cache arrays. Float16 is retained until an operation needs float32/64.
    sample_cache = [load_sample_cache(cache_file(cache_dir, int(row["sid"]))) for row in valid_rows]
    labels = np.asarray([int(row["label"]) for row in valid_rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in valid_rows], dtype=np.int64)
    statuses = np.asarray([str(row["generation_pair_status"]) for row in valid_rows], dtype=object)
    object_pair = np.stack([cache["object_pair"] for cache in sample_cache], axis=0)       # [N,L,D]
    last_state = np.stack([cache["last_state"] for cache in sample_cache], axis=0)         # [N,L,D]
    head_visual = np.stack([cache["head_visual_pair_pre"] for cache in sample_cache], axis=0)  # [N,L,H,Dh]
    head_objlast = np.stack([cache["head_object_to_last_pre"] for cache in sample_cache], axis=0)
    attn_centroid_pred = np.stack([cache["attention_centroid_pred"] for cache in sample_cache], axis=0)
    attn_centroid_margin = np.stack([cache["attention_centroid_gt_margin"] for cache in sample_cache], axis=0)
    visual_mass = np.stack([cache["visual_attention_mass"] for cache in sample_cache], axis=0)
    edge_norm = np.stack([cache["head_objlast_projected_norm"] for cache in sample_cache], axis=0)
    edge_total_cos = np.stack([cache["head_objlast_total_attention_cos"] for cache in sample_cache], axis=0)
    object_total_cos = np.stack([cache["object_total_attention_cos"] for cache in sample_cache], axis=0)
    object_total_norm_ratio = np.stack([cache["object_total_attention_norm_ratio"] for cache in sample_cache], axis=0)

    n_samples, n_layer_positions, n_heads, head_dim = head_visual.shape
    hidden_dim = object_pair.shape[-1]
    if n_layer_positions != len(selected_layers):
        raise RuntimeError("Cache layer count does not match selected_layers")

    # ------------------------------------------------------------------
    # Phase 2A: cross-fitted object/last metrics and shared geometry
    # ------------------------------------------------------------------
    object_last_rows: List[Dict[str, Any]] = []
    object_margins = np.full((n_samples, n_layer_positions), np.nan, dtype=np.float32)
    last_margins = np.full((n_samples, n_layer_positions), np.nan, dtype=np.float32)
    last_models_by_layer: Dict[int, Dict[int, CentroidModel]] = {}
    for layer_pos, layer in enumerate(selected_layers):
        obj_pred, obj_margin, _ = crossfit_centroids(
            object_pair[:, layer_pos, :], labels, folds, args.centroid_metric
        )
        last_pred, last_margin, last_models = crossfit_centroids(
            last_state[:, layer_pos, :], labels, folds, args.centroid_metric
        )
        object_margins[:, layer_pos] = obj_margin.astype(np.float32)
        last_margins[:, layer_pos] = last_margin.astype(np.float32)
        last_models_by_layer[int(layer)] = last_models
        raw_cos = np.sum(
            normalize_rows(object_pair[:, layer_pos, :])
            * normalize_rows(last_state[:, layer_pos, :]),
            axis=1,
        )
        object_last_rows.append({
            "layer": int(layer),
            "object_centroid_accuracy": float(np.mean(obj_pred == labels)),
            "object_centroid_macro_accuracy": macro_accuracy(obj_pred, labels),
            "last_centroid_accuracy": float(np.mean(last_pred == labels)),
            "last_centroid_macro_accuracy": macro_accuracy(last_pred, labels),
            "object_last_margin_correlation": safe_corr(obj_margin, last_margin),
            "linear_cka": linear_cka(object_pair[:, layer_pos, :], last_state[:, layer_pos, :]),
            "rsa_cosine_correlation": rsa_cosine_correlation(object_pair[:, layer_pos, :], last_state[:, layer_pos, :]),
            "mean_raw_object_last_cosine": safe_mean(raw_cos),
            "mean_object_to_total_attention_cosine": safe_mean(object_total_cos[:, layer_pos]),
            "mean_object_to_total_attention_norm_ratio": safe_mean(object_total_norm_ratio[:, layer_pos]),
        })
    write_csv(output_dir / "object_last_similarity.csv", object_last_rows)

    final_pos = selected_layers.index(final_layer)
    final_last_margin = last_margins[:, final_pos].astype(np.float64)
    final_last_models = last_models_by_layer[final_layer]
    final_last_directions = sample_directions_from_models(labels, folds, final_last_models)

    # ------------------------------------------------------------------
    # Phase 2B: per-head storage/attention/transport metrics
    # ------------------------------------------------------------------
    head_rows: List[Dict[str, Any]] = []
    head_oof_margins = np.full((n_samples, n_layer_positions, n_heads), np.nan, dtype=np.float32)
    for layer_pos, layer in enumerate(selected_layers):
        attention_module = attention_helper.resolve_self_attention(decoder_layers[int(layer)])
        o_weight = attention_helper.reshape_o_projection(
            attention_module,
            n_heads=n_heads,
            head_dim=head_dim,
        ).detach().float().cpu().numpy()  # [D,H,Dh]
        for head in range(n_heads):
            x = head_visual[:, layer_pos, head, :].astype(np.float32)
            pred, margin, _ = crossfit_centroids(x, labels, folds, args.centroid_metric)
            head_oof_margins[:, layer_pos, head] = margin.astype(np.float32)
            direct_pred = attn_centroid_pred[:, layer_pos, head].astype(np.int64)

            # Efficient sample-specific projection of the actual object->last edge
            # onto the held-out final-last GT-vs-opposite direction.
            w_head = o_weight[:, head, :]  # [D,Dh]
            local_direction = final_last_directions @ w_head  # [N,Dh]
            edge_projection = np.sum(
                head_objlast[:, layer_pos, head, :].astype(np.float32) * local_direction,
                axis=1,
            )
            row = {
                "layer": int(layer),
                "head": int(head),
                "content_centroid_accuracy": float(np.mean(pred == labels)),
                "content_centroid_macro_accuracy": macro_accuracy(pred, labels),
                "attention_centroid_accuracy": float(np.mean(direct_pred == labels)),
                "attention_centroid_macro_accuracy": macro_accuracy(direct_pred, labels),
                "mean_attention_centroid_gt_margin": safe_mean(attn_centroid_margin[:, layer_pos, head]),
                "mean_visual_attention_mass": safe_mean(visual_mass[:, layer_pos, head]),
                "mean_object_to_last_edge_norm": safe_mean(edge_norm[:, layer_pos, head]),
                "mean_edge_to_total_attention_cosine": safe_mean(edge_total_cos[:, layer_pos, head]),
                "storage_margin_final_last_margin_correlation": safe_corr(margin, final_last_margin),
                "mean_object_edge_final_relation_projection": safe_mean(edge_projection),
                "storage_margin_edge_projection_correlation": safe_corr(margin, edge_projection),
            }
            head_rows.append(row)
    write_csv(output_dir / "head_metrics.csv", head_rows)
    np.savez_compressed(
        output_dir / "crossfit_scores.npz",
        sids=np.asarray([int(row["sid"]) for row in valid_rows], dtype=np.int64),
        labels=labels.astype(np.int8),
        folds=folds.astype(np.int8),
        layers=np.asarray(selected_layers, dtype=np.int16),
        object_margins=object_margins.astype(np.float32),
        last_margins=last_margins.astype(np.float32),
        head_storage_margins=head_oof_margins.astype(np.float32),
    )

    metric = args.head_rank_metric
    sorted_heads = sorted(
        head_rows,
        key=lambda row: (float(row[metric]), float(row["content_centroid_accuracy"])),
        reverse=True,
    )
    top_heads = sorted_heads[: min(args.top_heads, len(sorted_heads))]
    by_layer_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        by_layer_rows[int(row["layer"])].append(row)
    selected_heads: List[Dict[str, Any]] = []
    used_controls = set()
    for rank, row in enumerate(top_heads, start=1):
        layer = int(row["layer"])
        candidates = by_layer_rows[layer]
        accuracies = np.asarray([float(item["content_centroid_accuracy"]) for item in candidates])
        median_acc = float(np.median(accuracies))
        pool = [
            item for item in candidates
            if int(item["head"]) != int(row["head"])
            and float(item["content_centroid_accuracy"]) <= median_acc
            and (layer, int(item["head"])) not in used_controls
        ]
        if not pool:
            pool = [item for item in candidates if int(item["head"]) != int(row["head"])]
        control = min(
            pool,
            key=lambda item: abs(
                float(item["mean_visual_attention_mass"])
                - float(row["mean_visual_attention_mass"])
            ),
        )
        used_controls.add((layer, int(control["head"])))
        selected_heads.append({
            "rank": rank,
            "layer": layer,
            "head": int(row["head"]),
            "rank_metric": metric,
            "rank_metric_value": float(row[metric]),
            "content_centroid_accuracy": float(row["content_centroid_accuracy"]),
            "attention_centroid_accuracy": float(row["attention_centroid_accuracy"]),
            "mean_visual_attention_mass": float(row["mean_visual_attention_mass"]),
            "matched_control_head": int(control["head"]),
            "matched_control_content_centroid_accuracy": float(control["content_centroid_accuracy"]),
            "matched_control_attention_centroid_accuracy": float(control["attention_centroid_accuracy"]),
            "matched_control_visual_attention_mass": float(control["mean_visual_attention_mass"]),
        })
    write_json(output_dir / "top_heads.json", selected_heads)

    top_group_rows: List[Dict[str, Any]] = []
    for selected in selected_heads:
        layer_pos = selected_layers.index(int(selected["layer"]))
        for role, head in (
            ("top", int(selected["head"])),
            ("matched_control", int(selected["matched_control_head"])),
        ):
            margins = head_oof_margins[:, layer_pos, head]
            for status in ("all",) + PAIR_STATUSES:
                mask = np.ones(n_samples, dtype=bool) if status == "all" else statuses == status
                if not np.any(mask):
                    continue
                top_group_rows.append({
                    "rank": selected["rank"],
                    "role": role,
                    "layer": selected["layer"],
                    "head": head,
                    "generation_pair_status": status,
                    "N": int(mask.sum()),
                    "gt_vs_opposite_positive_rate": float(np.mean(margins[mask] > 0)),
                    "mean_gt_opposite_margin": safe_mean(margins[mask]),
                    "mean_visual_attention_mass": safe_mean(visual_mass[mask, layer_pos, head]),
                    "mean_edge_norm": safe_mean(edge_norm[mask, layer_pos, head]),
                })
    write_csv(output_dir / "top_head_group_metrics.csv", top_group_rows)

    # ------------------------------------------------------------------
    # Phase 3: causal object->last head-edge ablation
    # ------------------------------------------------------------------
    head_ablation_path = output_dir / "head_ablation.jsonl"
    head_ablation_rows: List[Dict[str, Any]] = []
    if not args.skip_head_ablation:
        rows_to_run = list(valid_rows)
        if args.head_ablation_max_samples is not None:
            rows_to_run = rows_to_run[: max(0, int(args.head_ablation_max_samples))]
        heads_by_layer: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: {"top": [], "control": []})
        for item in selected_heads:
            heads_by_layer[int(item["layer"])]["top"].append(int(item["head"]))
            heads_by_layer[int(item["layer"])]["control"].append(int(item["matched_control_head"]))

        for row in tqdm(rows_to_run, desc="head-edge-ablation"):
            sid = int(row["sid"])
            record = records_by_sid[sid]
            image: Optional[Image.Image] = None
            try:
                image = base.record_image(record)
                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=str(row["question_text"]),
                    device=device,
                )
                subject_span = tuple(map(int, row["subject_span"]))
                reference_span = tuple(map(int, row["reference_span"]))
                prompt_last = int(row["prompt_last"])
                object_sources = sorted(set(
                    full_span_positions(subject_span) + full_span_positions(reference_span)
                ))
                relevant_layers = sorted(heads_by_layer)
                _, traces = trace_prompt_chunks(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layers=relevant_layers,
                    target_positions=[prompt_last],
                    chunk_size=args.trace_layer_chunk,
                )
                baseline_final = load_sample_cache(cache_file(cache_dir, sid))["last_state"][final_pos].astype(np.float32)
                fold_model = final_last_models[int(row["fold"])]
                baseline_final_margin = float(fold_model.gt_opposite_margin(
                    baseline_final[None, :], np.asarray([int(row["label"])])
                )[0])
                baseline_lm_margin = float(row["baseline_lm_margin"])
                baseline_pred = str(row["baseline_relation_prediction"])
                baseline_gen = row.get("baseline_generated_prediction")

                conditions: List[Tuple[str, int, List[int], bool]] = []
                for selected in selected_heads:
                    layer = int(selected["layer"])
                    conditions.append((f"top_rank_{selected['rank']}", layer, [int(selected["head"])], False))
                    conditions.append((f"control_rank_{selected['rank']}", layer, [int(selected["matched_control_head"])], False))
                for layer in relevant_layers:
                    conditions.append(("top_union", layer, sorted(set(heads_by_layer[layer]["top"])), args.generate_causal_unions))
                    conditions.append(("control_union", layer, sorted(set(heads_by_layer[layer]["control"])), args.generate_causal_unions))

                for condition, layer, heads, generate in conditions:
                    trace = traces[layer]
                    pre = head_source_pre_vectors(trace, [prompt_last], object_sources)
                    projected = projected_head_vectors(attention_helper, trace, pre)
                    vector = projected[heads].sum(dim=0).numpy().astype(np.float32)
                    patched = run_attention_subtraction(
                        base=base,
                        attention_helper=attention_helper,
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        layer=layer,
                        target_position=prompt_last,
                        vector=vector,
                        batch=batch,
                        relation_token_map=relation_token_map,
                        final_layer=final_layer,
                        generate=bool(generate),
                        max_new_tokens=args.max_new_tokens,
                    )
                    patched_final_margin = float(fold_model.gt_opposite_margin(
                        patched["final_last"][None, :], np.asarray([int(row["label"])])
                    )[0])
                    patched_lm_margin = relation_margin_from_logits(patched["relation_logits"], str(row["gt"]))
                    patched_pred = relation_prediction_from_logits(patched["relation_logits"])
                    patched_gen = patched.get("generated_prediction")
                    output = {
                        "sid": sid,
                        "gt": row["gt"],
                        "generation_pair_status": row["generation_pair_status"],
                        "condition": condition,
                        "layer": layer,
                        "heads": heads,
                        "n_heads": len(heads),
                        "subtracted_vector_norm": float(np.linalg.norm(vector)),
                        "baseline_final_last_margin": baseline_final_margin,
                        "patched_final_last_margin": patched_final_margin,
                        "delta_final_last_margin": patched_final_margin - baseline_final_margin,
                        "baseline_lm_margin": baseline_lm_margin,
                        "patched_lm_margin": patched_lm_margin,
                        "delta_lm_margin": patched_lm_margin - baseline_lm_margin,
                        "baseline_prediction": baseline_pred,
                        "patched_prediction": patched_pred,
                        "prediction_changed": patched_pred != baseline_pred,
                        "correct_to_wrong": baseline_pred == row["gt"] and patched_pred != row["gt"],
                        "correct_to_opposite": baseline_pred == row["gt"] and patched_pred == OPPOSITE[row["gt"]],
                        "generation_available": bool(generate),
                        "baseline_generation_prediction": baseline_gen,
                        "patched_generation_prediction": patched_gen,
                        "generation_changed": bool(generate and patched_gen != baseline_gen),
                        "generation_correct_to_opposite": bool(
                            generate and baseline_gen == row["gt"] and patched_gen == OPPOSITE[row["gt"]]
                        ),
                    }
                    append_jsonl(head_ablation_path, output)
                    head_ablation_rows.append(output)
                del batch, traces
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "head_ablation",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"[ERROR head_ablation sid={sid}] {type(exc).__name__}: {exc}")
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        head_ablation_rows = read_jsonl(head_ablation_path)

    head_summary: List[Dict[str, Any]] = []
    if head_ablation_rows:
        head_summary.extend(summarize_causal_rows(
            head_ablation_rows,
            key_fields=("condition", "layer"),
        ))
        # Add pair-status stratification.
        head_summary.extend(summarize_causal_rows(
            head_ablation_rows,
            key_fields=("condition", "layer", "generation_pair_status"),
        ))
    write_csv(output_dir / "head_ablation_summary.csv", head_summary)

    # ------------------------------------------------------------------
    # Phase 4: object x question factorial utilization
    # ------------------------------------------------------------------
    if args.factorial_layers.strip().lower() == "auto":
        layer_scores: Dict[int, float] = defaultdict(float)
        for item in selected_heads:
            layer_scores[int(item["layer"])] += float(item["rank_metric_value"])
        factorial_layers = [
            layer for layer, _ in sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
        ][: max(1, int(args.factorial_auto_layers))]
    else:
        factorial_layers = parse_layer_list(args.factorial_layers, len(decoder_layers))
    write_json(output_dir / "factorial_layers.json", {"layers": factorial_layers})

    factorial_path = output_dir / "factorial.jsonl"
    factorial_rows: List[Dict[str, Any]] = []
    if not args.skip_factorial:
        rows_to_run = list(valid_rows)
        if args.factorial_max_samples is not None:
            rows_to_run = rows_to_run[: max(0, int(args.factorial_max_samples))]
        for row in tqdm(rows_to_run, desc="object-question-factorial"):
            sid = int(row["sid"])
            record = records_by_sid[sid]
            image: Optional[Image.Image] = None
            try:
                image = base.record_image(record)
                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=str(row["question_text"]),
                    device=device,
                )
                prompt_last = int(row["prompt_last"])
                subject_span = tuple(map(int, row["subject_span"]))
                reference_span = tuple(map(int, row["reference_span"]))
                object_sources = sorted(set(
                    full_span_positions(subject_span) + full_span_positions(reference_span)
                ))
                question_groups = {
                    key: list(map(int, value))
                    for key, value in row["question_groups"].items()
                }
                _, traces = trace_prompt_chunks(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layers=factorial_layers,
                    target_positions=[prompt_last],
                    chunk_size=args.trace_layer_chunk,
                )
                fold_model = final_last_models[int(row["fold"])]
                baseline_final = load_sample_cache(cache_file(cache_dir, sid))["last_state"][final_pos].astype(np.float32)
                f_full_last = float(fold_model.gt_opposite_margin(
                    baseline_final[None, :], np.asarray([int(row["label"])])
                )[0])
                f_full_lm = float(row["baseline_lm_margin"])

                for layer in factorial_layers:
                    trace = traces[layer]
                    object_pre = head_source_pre_vectors(trace, [prompt_last], object_sources)
                    object_vector = projected_head_vectors(attention_helper, trace, object_pre).sum(dim=0).numpy().astype(np.float32)
                    no_object = run_attention_subtraction(
                        base=base,
                        attention_helper=attention_helper,
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        layer=layer,
                        target_position=prompt_last,
                        vector=object_vector,
                        batch=batch,
                        relation_token_map=relation_token_map,
                        final_layer=final_layer,
                        generate=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                    f_no_object_last = float(fold_model.gt_opposite_margin(
                        no_object["final_last"][None, :], np.asarray([int(row["label"])])
                    )[0])
                    f_no_object_lm = relation_margin_from_logits(no_object["relation_logits"], str(row["gt"]))

                    for question_group in factorial_groups:
                        q_sources = question_groups.get(question_group, [])
                        if not q_sources:
                            continue
                        q_pre = head_source_pre_vectors(trace, [prompt_last], q_sources)
                        q_vector = projected_head_vectors(attention_helper, trace, q_pre).sum(dim=0).numpy().astype(np.float32)
                        no_q = run_attention_subtraction(
                            base=base,
                            attention_helper=attention_helper,
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            layer=layer,
                            target_position=prompt_last,
                            vector=q_vector,
                            batch=batch,
                            relation_token_map=relation_token_map,
                            final_layer=final_layer,
                            generate=args.factorial_generate,
                            max_new_tokens=args.max_new_tokens,
                        )
                        no_both = run_attention_subtraction(
                            base=base,
                            attention_helper=attention_helper,
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            layer=layer,
                            target_position=prompt_last,
                            vector=object_vector + q_vector,
                            batch=batch,
                            relation_token_map=relation_token_map,
                            final_layer=final_layer,
                            generate=args.factorial_generate,
                            max_new_tokens=args.max_new_tokens,
                        )
                        f_no_q_last = float(fold_model.gt_opposite_margin(
                            no_q["final_last"][None, :], np.asarray([int(row["label"])])
                        )[0])
                        f_no_both_last = float(fold_model.gt_opposite_margin(
                            no_both["final_last"][None, :], np.asarray([int(row["label"])])
                        )[0])
                        f_no_q_lm = relation_margin_from_logits(no_q["relation_logits"], str(row["gt"]))
                        f_no_both_lm = relation_margin_from_logits(no_both["relation_logits"], str(row["gt"]))

                        interaction_last = f_full_last - f_no_object_last - f_no_q_last + f_no_both_last
                        interaction_lm = f_full_lm - f_no_object_lm - f_no_q_lm + f_no_both_lm
                        result = {
                            "sid": sid,
                            "gt": row["gt"],
                            "generation_pair_status": row["generation_pair_status"],
                            "layer": int(layer),
                            "question_group": question_group,
                            "object_vector_norm": float(np.linalg.norm(object_vector)),
                            "question_vector_norm": float(np.linalg.norm(q_vector)),
                            "f_full_final_last_margin": f_full_last,
                            "f_no_object_final_last_margin": f_no_object_last,
                            "f_no_question_final_last_margin": f_no_q_last,
                            "f_no_both_final_last_margin": f_no_both_last,
                            "interaction_final_last_margin": interaction_last,
                            "f_full_lm_margin": f_full_lm,
                            "f_no_object_lm_margin": f_no_object_lm,
                            "f_no_question_lm_margin": f_no_q_lm,
                            "f_no_both_lm_margin": f_no_both_lm,
                            "interaction_lm_margin": interaction_lm,
                            "object_effect_with_question_lm": f_full_lm - f_no_object_lm,
                            "object_effect_without_question_lm": f_no_q_lm - f_no_both_lm,
                            "question_effect_with_object_lm": f_full_lm - f_no_q_lm,
                            "question_effect_without_object_lm": f_no_object_lm - f_no_both_lm,
                            "no_question_prediction": relation_prediction_from_logits(no_q["relation_logits"]),
                            "no_both_prediction": relation_prediction_from_logits(no_both["relation_logits"]),
                            "generation_available": bool(args.factorial_generate),
                            "no_question_generation_prediction": no_q.get("generated_prediction"),
                            "no_both_generation_prediction": no_both.get("generated_prediction"),
                        }
                        append_jsonl(factorial_path, result)
                        factorial_rows.append(result)
                del batch, traces
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "factorial",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"[ERROR factorial sid={sid}] {type(exc).__name__}: {exc}")
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        factorial_rows = read_jsonl(factorial_path)

    factorial_summary = summarize_factorial(factorial_rows)
    write_csv(output_dir / "factorial_summary.csv", factorial_summary)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"model: {args.model}",
        f"dataset: {args.dataset}",
        f"valid samples: {len(valid_rows)}",
        f"generation pair status: {dict(Counter(row['generation_pair_status'] for row in valid_rows))}",
        f"selected layers: {selected_layers}",
        "",
        "EXPERIMENT 1: TOP STORAGE HEADS",
    ]
    for item in selected_heads:
        report_lines.append(
            f"rank={item['rank']} L{item['layer']} H{item['head']} "
            f"contentAcc={item['content_centroid_accuracy']:.4f} "
            f"attentionCentroidAcc={item['attention_centroid_accuracy']:.4f} "
            f"visualMass={item['mean_visual_attention_mass']:.6f} | "
            f"control=H{item['matched_control_head']} "
            f"controlContentAcc={item['matched_control_content_centroid_accuracy']:.4f} "
            f"controlVisualMass={item['matched_control_visual_attention_mass']:.6f}"
        )
    report_lines.extend(["", "EXPERIMENT 2: HEAD ABLATION SUMMARY"])
    for row in head_summary:
        if "generation_pair_status" in row:
            continue
        report_lines.append(
            f"{row['condition']} L{row['layer']} N={row['N']} "
            f"dLast={row['mean_delta_final_last_margin']:+.6f} "
            f"dLM={row['mean_delta_lm_margin']:+.6f} "
            f"correctToOpp={row['correct_to_opposite_rate']:.4f}"
        )
    report_lines.extend(["", "EXPERIMENT 3: OBJECT x QUESTION INTERACTION"])
    for row in factorial_summary:
        if row["group_type"] != "all":
            continue
        report_lines.append(
            f"L{row['layer']} {row['question_group']} N={row['N']} "
            f"G_last={row['mean_interaction_final_last']:+.6f} "
            f"G_lm={row['mean_interaction_lm']:+.6f} "
            f"positive={row['positive_interaction_rate_lm']:.4f}"
        )
    report = "\n".join(report_lines) + "\n"
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
