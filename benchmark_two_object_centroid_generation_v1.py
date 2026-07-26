#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark four-direction object-pair centroid probes and autoregressive generation
across multiple VLMs and two datasets.

Main centroid definition
------------------------
For sample i at decoder layer l:

    r_i^l = h_subject,i^l - h_reference,i^l

For each held-out fold, fit four class centroids on the training folds:

    mu_k^l = mean_{i: y_i=k} r_i^l,
    k in {left, right, above, below}

Predict each held-out sample by nearest cosine centroid:

    y_hat_i^l = argmax_k cos(r_i^l, mu_k^l)

This is the same four-independent-centroid procedure used by
analyze_spatial_storage_transport_utilization_v3.py. It does not force
right=-left or below=-above.

The default cross-validation split is deterministic by unordered object pair,
so (dog, chair) and (chair, dog) always belong to the same fold.

Requested benchmark model aliases
---------------------------------
    qwen-3b
    qwen-7b
    qwen2-2b
    llava15-7b
    llava15-13b
    internvl-1b
    internvl-2b
    internvl-8b

The aliases are resolved against the model registry exposed by
analyze_coco_centroid_generation_step1_v4.py and
extract_two_object_relation_states.py. If a repository uses a slightly
different key, the script searches a list of candidate keys and prints the
resolved key.

Datasets
--------
1. coco_two
   Uses extract_two_object_relation_states.py::load_records("coco_two", ...)
   and the standard prompt JSONL.

2. controlled_a
   First tries repository-native load_records aliases. If unavailable, it loads
   a generic JSON/JSONL manifest. Use --controlled-json and
   --controlled-image-root when auto-detection cannot identify the local paths.

Per-sample generation output
----------------------------
generation_samples.csv/jsonl contain explicit columns:

    model, dataset, sid, pred, question, gt, generation, acc

Additional columns include subject, reference, first-token LM prediction,
relation logits, fold and image path.

Main outputs
------------
<output-dir>/
    benchmark_summary.csv
    benchmark_report.txt
    <dataset>/<model>/
        config.json
        generation_samples.csv
        generation_samples.jsonl
        states/<sid>.npz
        centroid_predictions.jsonl
        layer_metrics.csv
        summary.json
        errors.jsonl

Example: run all requested models and datasets
----------------------------------------------
CUDA_VISIBLE_DEVICES=0 python -u benchmark_two_object_centroid_generation_v1.py \
  --models all \
  --datasets controlled_a,coco_two \
  --data-root data \
  --coco-prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --controlled-json /path/to/controlled_A_manifest.jsonl \
  --controlled-image-root /path/to/Controlled_Images_A \
  --device cuda:0 \
  --folds 5 \
  --object-states last,mean \
  --output-dir output/two_object_centroid_generation_benchmark

Run one model only
------------------
CUDA_VISIBLE_DEVICES=0 python -u benchmark_two_object_centroid_generation_v1.py \
  --models qwen-3b \
  --datasets coco_two \
  --data-root data \
  --device cuda:0 \
  --output-dir output/two_object_centroid_generation_benchmark
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
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "two-object-centroid-generation-benchmark-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
ID_TO_REL = {index: name for name, index in REL_TO_ID.items()}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

REQUESTED_MODEL_ALIASES = (
    "qwen-3b",
    "qwen-7b",
    "qwen2-2b",
    "llava15-7b",
    "llava15-13b",
    "internvl-1b",
    "internvl-2b",
    "internvl-8b",
)

MODEL_KEY_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "qwen-3b": (
        "qwen-3b",
        "qwen2.5-3b",
        "qwen2_5-3b",
        "qwen25-3b",
        "qwen2.5-vl-3b",
        "qwen2_5_vl_3b",
    ),
    "qwen-7b": (
        "qwen-7b",
        "qwen2.5-7b",
        "qwen2_5-7b",
        "qwen25-7b",
        "qwen2.5-vl-7b",
        "qwen2_5_vl_7b",
    ),
    "qwen2-2b": (
        "qwen2-2b",
        "qwen2vl-2b",
        "qwen2-vl-2b",
        "qwen2_vl_2b",
        "qwen2-2b-instruct",
    ),
    "llava15-7b": (
        "llava-7b",
        "llava15-7b",
        "llava1.5-7b",
        "llava-1.5-7b",
    ),
    "llava15-13b": (
        "llava-13b",
        "llava15-13b",
        "llava1.5-13b",
        "llava-1.5-13b",
    ),
    "internvl-1b": (
        "internvl-1b",
        "internvl2-1b",
        "internvl2.5-1b",
        "internvl25-1b",
    ),
    "internvl-2b": (
        "internvl-2b",
        "internvl2-2b",
        "internvl2.5-2b",
        "internvl25-2b",
    ),
    "internvl-8b": (
        "internvl-8b",
        "internvl2-8b",
        "internvl2.5-8b",
        "internvl25-8b",
    ),
}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkSample:
    uid: str
    sid: Any
    dataset: str
    subject: str
    reference: str
    question: str
    gt: str
    image_path: Optional[str] = None
    native_record: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CentroidModel:
    centroids: np.ndarray
    counts: np.ndarray
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
            relation = ID_TO_REL[int(label)]
            opposite_id = REL_TO_ID[OPPOSITE[relation]]
            out[index] = scores[index, int(label)] - scores[index, opposite_id]
        return out


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default="all",
        help="'all' or comma-separated requested aliases/model-registry keys.",
    )
    parser.add_argument(
        "--datasets",
        default="controlled_a,coco_two",
        help="Comma-separated: controlled_a,coco_two",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--controlled-json",
        default=None,
        help="Controlled-A manifest JSON/JSONL. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--controlled-prompt-jsonl",
        default=None,
        help="Optional Controlled-A prompt JSONL keyed by sid.",
    )
    parser.add_argument(
        "--controlled-image-root",
        default=None,
        help="Root directory for Controlled-A images.",
    )
    parser.add_argument(
        "--controlled-question-template",
        default=(
            "What is the spatial relation of the {subject} relative to the "
            "{reference}? Answer with one relation: left, right, above, or below."
        ),
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--dtype", default=None, help="Override model-registry dtype.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--object-states",
        default="last,mean",
        help="Comma-separated subset of last,mean. 'last' matches the v3 default.",
    )
    parser.add_argument(
        "--centroid-metric",
        choices=("cosine", "euclidean"),
        default="cosine",
    )
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--empty-cache-every", type=int, default=10)

    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    parser.add_argument(
        "--semantic-helper",
        default="analyze_coco_flip_same_token_similarity_v1.py",
    )
    parser.add_argument(
        "--two-object-script",
        default="extract_two_object_relation_states.py",
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
        raise ImportError(f"Unable to import {path}")
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
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


def sanitize_uid(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if text:
        return text[:160]
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:20]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def macro_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    values: List[float] = []
    for relation_id in range(len(RELATIONS)):
        mask = labels == relation_id
        if np.any(mask):
            values.append(float(np.mean(pred[mask] == labels[mask])))
    return float(np.mean(values)) if values else float("nan")


def stable_pair_fold(subject: str, reference: str, folds: int, seed: int) -> int:
    pair = "||".join(
        sorted([str(subject).strip().lower(), str(reference).strip().lower()])
    )
    digest = hashlib.sha256(f"{seed}::{pair}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(folds)


def normalize_relation(value: Any, base: Any = None) -> str:
    if base is not None:
        with contextlib.suppress(Exception):
            normalized = str(base.normalize_relation(value)).strip().lower()
            if normalized in REL_TO_ID:
                return normalized
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z]+", " ", text).strip()
    tokens = set(text.split())
    if "left" in tokens:
        return "left"
    if "right" in tokens:
        return "right"
    if tokens & {"above", "over", "on", "top", "upper"}:
        return "above"
    if tokens & {"below", "under", "beneath", "bottom", "lower"}:
        return "below"
    return ""


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise TypeError(f"Cannot locate [B,T,D] hidden tensor in {type(output).__name__}")


def relation_logits_to_list(relation_data: Mapping[str, Any]) -> List[float]:
    logits = relation_data.get("logits", [])
    return [float(value) for value in logits]


# -----------------------------------------------------------------------------
# Centroid fitting: exact four-independent-centroid procedure
# -----------------------------------------------------------------------------


def fit_centroids(x: np.ndarray, labels: np.ndarray, metric: str) -> CentroidModel:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centroids: List[np.ndarray] = []
    counts: List[int] = []
    for relation_id in range(len(RELATIONS)):
        mask = labels == relation_id
        if not np.any(mask):
            raise RuntimeError(
                f"Centroid training split is missing class {ID_TO_REL[relation_id]}"
            )
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


def in_sample_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model = fit_centroids(x, labels, metric)
    return model.predict(x), model.gt_opposite_margin(x, labels)


# -----------------------------------------------------------------------------
# Model aliases and loading
# -----------------------------------------------------------------------------


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_model_key(
    requested: str,
    specs: Mapping[str, Any],
) -> str:
    if requested in specs:
        return requested

    candidates = list(MODEL_KEY_CANDIDATES.get(requested, (requested,)))
    for candidate in candidates:
        if candidate in specs:
            return candidate

    normalized_available = {normalize_key(key): key for key in specs}
    for candidate in candidates:
        normalized = normalize_key(candidate)
        if normalized in normalized_available:
            return normalized_available[normalized]

    # Conservative fuzzy fallback: exact normalized containment in either direction.
    matches: List[str] = []
    for candidate in candidates:
        normalized = normalize_key(candidate)
        for available in specs:
            available_norm = normalize_key(available)
            if normalized == available_norm:
                matches.append(available)
            elif normalized in available_norm or available_norm in normalized:
                matches.append(available)
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Cannot resolve model alias {requested!r}. "
        f"Candidates={candidates}. Available registry keys={sorted(specs)}"
    )


def load_model_and_processor(
    *,
    requested_alias: str,
    resolved_key: str,
    spec: Any,
    base: Any,
    args: argparse.Namespace,
) -> Tuple[Any, Any]:
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} lacks {spec.model_class}"
        )

    dtype = (
        base.resolve_dtype(args.dtype)
        if args.dtype
        else base.resolve_dtype(spec.dtype_name)
    )
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl

    print(
        f"\nLoading alias={requested_alias} registry_key={resolved_key} "
        f"repo={spec.repo_id}",
        flush=True,
    )
    try:
        model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    except TypeError:
        # Some remote-code models reject attn_implementation.
        kwargs.pop("attn_implementation", None)
        model = model_cls.from_pretrained(spec.repo_id, **kwargs)

    model.eval()
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field_name in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field_name):
                setattr(generation_config, field_name, None)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    return model, processor


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def get_first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, dict):
        for key in ("records", "samples", "data", "items", "annotations"):
            if isinstance(value.get(key), list):
                return [dict(item) for item in value[key]]
        # Mapping from id -> row.
        if all(isinstance(item, Mapping) for item in value.values()):
            rows: List[Dict[str, Any]] = []
            for key, item in value.items():
                row = dict(item)
                row.setdefault("sid", key)
                rows.append(row)
            return rows
    raise ValueError(f"Unsupported manifest structure: {path}")


def controlled_manifest_candidates(data_root: Path) -> List[Path]:
    candidates = [
        data_root / "controlled_a.jsonl",
        data_root / "controlled_A.jsonl",
        data_root / "Controlled_A.jsonl",
        data_root / "Controlled_Images_A.jsonl",
        data_root / "controlled_a.json",
        data_root / "controlled_A.json",
        data_root / "Controlled_A.json",
        data_root / "Controlled_Images_A.json",
        data_root / "controlled_a" / "manifest.jsonl",
        data_root / "controlled_A" / "manifest.jsonl",
        data_root / "Controlled_Images_A" / "manifest.jsonl",
        Path("prompts/Controlled_Images_A_with_answer_four_options.jsonl"),
        Path("prompts/controlled_A_with_answer_four_options.jsonl"),
        Path("prompts/controlled_a_with_answer_four_options.jsonl"),
    ]
    return [path for path in candidates if path.exists()]


def controlled_image_root_candidates(data_root: Path) -> List[Path]:
    candidates = [
        data_root / "controlled_a",
        data_root / "controlled_A",
        data_root / "Controlled_A",
        data_root / "Controlled_Images_A",
        data_root / "controlled_images_A",
        Path("data/controlled_a"),
        Path("data/Controlled_Images_A"),
    ]
    return [path for path in candidates if path.exists()]


def resolve_image_path(
    raw_value: Any,
    *,
    manifest_dir: Path,
    image_root: Optional[Path],
) -> Optional[Path]:
    if raw_value in (None, ""):
        return None
    raw = Path(str(raw_value))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(manifest_dir / raw)
        if image_root is not None:
            candidates.append(image_root / raw)
        candidates.append(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def parse_generic_controlled_manifest(
    *,
    manifest_path: Path,
    image_root: Optional[Path],
    question_template: str,
    base: Any,
    max_samples: Optional[int],
) -> List[BenchmarkSample]:
    rows = load_json_or_jsonl(manifest_path)
    samples: List[BenchmarkSample] = []

    subject_keys = (
        "subject", "subj", "subject_name", "object1", "obj1", "object_a",
        "first_object", "query_object", "target_object",
    )
    reference_keys = (
        "reference", "ref", "reference_name", "object2", "obj2", "object_b",
        "second_object", "anchor_object",
    )
    relation_keys = (
        "gt", "answer", "answer_raw", "relation", "label", "direction",
        "spatial_relation",
    )
    question_keys = (
        "question", "question_text", "prompt", "text", "query",
    )
    image_keys = (
        "image_path", "image", "file_name", "filename", "img", "path",
        "image_file",
    )
    sid_keys = ("sid", "id", "sample_id", "uid", "index")

    for index, row in enumerate(rows):
        subject = get_first(row, subject_keys)
        reference = get_first(row, reference_keys)
        gt = normalize_relation(get_first(row, relation_keys), base)
        if subject is None or reference is None or gt not in REL_TO_ID:
            continue

        sid = get_first(row, sid_keys)
        if sid is None:
            sid = index
        question = get_first(row, question_keys)
        if not question:
            question = question_template.format(
                subject=str(subject),
                reference=str(reference),
            )
        image_value = get_first(row, image_keys)
        image_path = resolve_image_path(
            image_value,
            manifest_dir=manifest_path.parent,
            image_root=image_root,
        )
        if image_path is None:
            continue

        samples.append(
            BenchmarkSample(
                uid=sanitize_uid(sid),
                sid=sid,
                dataset="controlled_a",
                subject=str(subject),
                reference=str(reference),
                question=str(question),
                gt=gt,
                image_path=str(image_path),
                metadata={"manifest": str(manifest_path)},
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError(
            f"No usable Controlled-A samples parsed from {manifest_path}. "
            "Expected subject/reference/relation/image fields."
        )
    return samples


def try_native_controlled_loader(
    *,
    two_object: Any,
    base: Any,
    data_root: Path,
    prompt_path: Optional[Path],
    max_samples: Optional[int],
) -> Optional[List[BenchmarkSample]]:
    aliases = ("controlled_a", "controlled_A", "controlled", "controlled_images_a")
    loaded_records = None
    used_alias = None
    for alias in aliases:
        try:
            records, _audit = two_object.load_records(alias, data_root, max_samples)
            if records:
                loaded_records = records
                used_alias = alias
                break
        except Exception:
            continue
    if not loaded_records:
        return None

    if prompt_path is None or not prompt_path.exists():
        return None
    prompts = base.load_standard_prompts(prompt_path)
    samples: List[BenchmarkSample] = []
    for record in loaded_records:
        sid = int(record.sid)
        if sid not in prompts:
            continue
        prompt = prompts[sid]
        gt = normalize_relation(prompt.get("answer_raw"), base)
        if gt not in REL_TO_ID:
            continue
        samples.append(
            BenchmarkSample(
                uid=sanitize_uid(sid),
                sid=sid,
                dataset="controlled_a",
                subject=str(prompt["subject"]),
                reference=str(prompt["reference"]),
                question=str(prompt["question_text"]),
                gt=gt,
                native_record=record,
                metadata={"native_loader_alias": used_alias},
            )
        )
    return samples or None


def load_controlled_a_samples(
    *,
    args: argparse.Namespace,
    base: Any,
    two_object: Any,
) -> List[BenchmarkSample]:
    data_root = Path(args.data_root)
    prompt_path = (
        Path(args.controlled_prompt_jsonl)
        if args.controlled_prompt_jsonl
        else None
    )
    if prompt_path is None:
        prompt_candidates = controlled_manifest_candidates(data_root)
        prompt_path = prompt_candidates[0] if prompt_candidates else None

    native = try_native_controlled_loader(
        two_object=two_object,
        base=base,
        data_root=data_root,
        prompt_path=prompt_path,
        max_samples=args.max_samples,
    )
    if native is not None:
        print(
            f"Controlled-A: native loader, samples={len(native)}, "
            f"prompt={prompt_path}",
            flush=True,
        )
        return native

    if args.controlled_json:
        manifest_path = Path(args.controlled_json)
    else:
        candidates = controlled_manifest_candidates(data_root)
        if not candidates:
            raise FileNotFoundError(
                "Controlled-A manifest was not auto-detected. Pass "
                "--controlled-json and --controlled-image-root explicitly."
            )
        manifest_path = candidates[0]

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    image_root: Optional[Path]
    if args.controlled_image_root:
        image_root = Path(args.controlled_image_root)
    else:
        roots = controlled_image_root_candidates(data_root)
        image_root = roots[0] if roots else None

    samples = parse_generic_controlled_manifest(
        manifest_path=manifest_path,
        image_root=image_root,
        question_template=args.controlled_question_template,
        base=base,
        max_samples=args.max_samples,
    )
    print(
        f"Controlled-A: manifest={manifest_path}, image_root={image_root}, "
        f"samples={len(samples)}",
        flush=True,
    )
    return samples


def load_coco_two_samples(
    *,
    args: argparse.Namespace,
    base: Any,
    two_object: Any,
) -> List[BenchmarkSample]:
    records, audit = two_object.load_records(
        "coco_two",
        Path(args.data_root),
        args.max_samples,
    )
    prompt_path = Path(args.coco_prompt_jsonl)
    prompts = base.load_standard_prompts(prompt_path)
    samples: List[BenchmarkSample] = []
    for record in records:
        sid = int(record.sid)
        if sid not in prompts:
            continue
        prompt = prompts[sid]
        gt = normalize_relation(prompt.get("answer_raw"), base)
        if gt not in REL_TO_ID:
            continue
        samples.append(
            BenchmarkSample(
                uid=sanitize_uid(sid),
                sid=sid,
                dataset="coco_two",
                subject=str(prompt["subject"]),
                reference=str(prompt["reference"]),
                question=str(prompt["question_text"]),
                gt=gt,
                native_record=record,
                metadata={"audit": audit},
            )
        )
    if not samples:
        raise RuntimeError("No usable COCO-two samples")
    print(
        f"COCO-two: prompt={prompt_path}, samples={len(samples)}",
        flush=True,
    )
    return samples


def open_sample_image(sample: BenchmarkSample, base: Any) -> Image.Image:
    if sample.native_record is not None:
        return base.record_image(sample.native_record)
    if not sample.image_path:
        raise FileNotFoundError(f"No image path for sample {sample.sid}")
    return Image.open(sample.image_path).convert("RGB")


# -----------------------------------------------------------------------------
# Decoder block state capture
# -----------------------------------------------------------------------------


def span_positions(span: Sequence[int], mode: str) -> List[int]:
    start, end = int(span[0]), int(span[1])
    if mode == "last":
        return [end]
    if mode == "mean":
        return list(range(start, end + 1))
    raise ValueError(mode)


class ObjectPairCapture:
    def __init__(
        self,
        decoder_layers: Sequence[torch.nn.Module],
        subject_span: Sequence[int],
        reference_span: Sequence[int],
        modes: Sequence[str],
    ):
        self.decoder_layers = list(decoder_layers)
        self.subject_span = tuple(map(int, subject_span))
        self.reference_span = tuple(map(int, reference_span))
        self.modes = list(modes)
        self.values: Dict[str, Dict[int, torch.Tensor]] = {
            mode: {} for mode in self.modes
        }
        self.handles = [
            layer.register_forward_hook(self._build_hook(index))
            for index, layer in enumerate(self.decoder_layers)
        ]

    def _build_hook(self, layer_index: int):
        def hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            hidden = first_tensor(output)
            if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
                raise RuntimeError(
                    f"Expected block output [1,T,D], got {tuple(hidden.shape)}"
                )
            for mode in self.modes:
                subject_positions = span_positions(self.subject_span, mode)
                reference_positions = span_positions(self.reference_span, mode)
                subject_index = torch.tensor(
                    subject_positions,
                    device=hidden.device,
                    dtype=torch.long,
                )
                reference_index = torch.tensor(
                    reference_positions,
                    device=hidden.device,
                    dtype=torch.long,
                )
                subject_state = hidden[0].index_select(0, subject_index).mean(dim=0)
                reference_state = hidden[0].index_select(0, reference_index).mean(dim=0)
                self.values[mode][layer_index] = (
                    subject_state - reference_state
                ).detach().float().cpu()
            return output

        return hook

    def close(self) -> None:
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles = []

    def stacked(self, mode: str) -> np.ndarray:
        missing = [
            index
            for index in range(len(self.decoder_layers))
            if index not in self.values[mode]
        ]
        if missing:
            raise RuntimeError(f"Missing captured decoder layers: {missing[:10]}")
        return np.stack(
            [self.values[mode][index].numpy() for index in range(len(self.decoder_layers))],
            axis=0,
        ).astype(np.float32)

    def __enter__(self) -> "ObjectPairCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# -----------------------------------------------------------------------------
# Per-model/dataset extraction and analysis
# -----------------------------------------------------------------------------


def state_path(states_dir: Path, uid: str) -> Path:
    return states_dir / f"{sanitize_uid(uid)}.npz"


def load_completed_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["uid"]): row for row in rows}


def extract_model_dataset(
    *,
    requested_alias: str,
    resolved_key: str,
    dataset_name: str,
    samples: Sequence[BenchmarkSample],
    model: Any,
    processor: Any,
    base: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    run_dir = output_dir / dataset_name / requested_alias
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    states_dir = run_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    samples_jsonl = run_dir / "generation_samples.jsonl"
    errors_jsonl = run_dir / "errors.jsonl"

    object_modes = parse_csv_list(args.object_states)
    invalid_modes = sorted(set(object_modes) - {"last", "mean"})
    if invalid_modes:
        raise ValueError(f"Unsupported --object-states: {invalid_modes}")
    if not object_modes:
        raise ValueError("--object-states selected no modes")

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    relation_token_map = base.relation_token_variants(processor.tokenizer)
    device = torch.device(args.device)

    config = {
        "script_version": SCRIPT_VERSION,
        "requested_model_alias": requested_alias,
        "resolved_model_key": resolved_key,
        "dataset": dataset_name,
        "n_samples_requested": len(samples),
        "n_decoder_layers": len(decoder_layers),
        "decoder_path": decoder_path,
        "folds": args.folds,
        "seed": args.seed,
        "object_states": object_modes,
        "centroid_metric": args.centroid_metric,
        "centroid_definition": (
            "four independent class means of subject-minus-reference block-output "
            "vectors; held-out nearest cosine/euclidean centroid"
        ),
    }
    write_json(run_dir / "config.json", config)

    completed = load_completed_rows(samples_jsonl) if args.resume else {}
    successful = 0

    for sample in tqdm(
        samples,
        desc=f"{requested_alias}:{dataset_name}",
        total=len(samples),
    ):
        uid = str(sample.uid)
        cache_path = state_path(states_dir, uid)
        if uid in completed and cache_path.exists():
            successful += 1
            continue

        image: Optional[Image.Image] = None
        batch: Optional[Dict[str, Any]] = None
        try:
            image = open_sample_image(sample, base)
            batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=sample.question,
                device=device,
            )
            input_ids = batch["input_ids"][0].detach().cpu().tolist()
            subject_span, reference_span = base.locate_object_spans(
                processor.tokenizer,
                input_ids,
                sample.subject,
                sample.reference,
            )

            with ObjectPairCapture(
                decoder_layers,
                subject_span,
                reference_span,
                object_modes,
            ) as capture:
                with torch.inference_mode():
                    outputs = model(
                        **batch,
                        use_cache=False,
                        return_dict=True,
                    )

            arrays: Dict[str, np.ndarray] = {
                "layers": np.arange(len(decoder_layers), dtype=np.int16),
            }
            for mode in object_modes:
                arrays[f"object_pair_{mode}"] = capture.stacked(mode).astype(
                    np.float16
                )
            np.savez_compressed(cache_path, **arrays)

            relation_data = base.relation_scores(
                outputs.logits[0, -1],
                relation_token_map,
                gt=None,
            )
            lm_pred = normalize_relation(
                relation_data.get("prediction", ""),
                base,
            )
            generation = base.generate_text(
                model,
                processor,
                dict(batch),
                max_new_tokens=args.max_new_tokens,
            )
            generation_pred = normalize_relation(generation, base)
            correct = bool(generation_pred == sample.gt)
            fold = stable_pair_fold(
                sample.subject,
                sample.reference,
                args.folds,
                args.seed,
            )

            row = {
                "model": requested_alias,
                "resolved_model_key": resolved_key,
                "dataset": dataset_name,
                "uid": uid,
                "sid": sample.sid,
                "subject": sample.subject,
                "reference": sample.reference,
                "pred": generation_pred,
                "question": sample.question,
                "gt": sample.gt,
                "generation": generation,
                "acc": int(correct),
                "generation_correct": correct,
                "lm_pred": lm_pred,
                "lm_relation_logits": relation_logits_to_list(relation_data),
                "fold": int(fold),
                "subject_span": list(map(int, subject_span)),
                "reference_span": list(map(int, reference_span)),
                "image_path": sample.image_path,
            }
            append_jsonl(samples_jsonl, row)
            completed[uid] = row
            successful += 1

            if args.print_every > 0 and successful % args.print_every == 0:
                compact_question = " ".join(sample.question.split())
                compact_generation = " ".join(str(generation).split())
                print(
                    f"\n[{requested_alias}][{dataset_name}] "
                    f"sid={sample.sid} pred={generation_pred or '<unparsed>'} "
                    f"gt={sample.gt} acc={int(correct)} "
                    f"question={compact_question!r} "
                    f"generation={compact_generation!r}",
                    flush=True,
                )

            del outputs, relation_data
        except Exception as exc:
            error_row = {
                "model": requested_alias,
                "dataset": dataset_name,
                "uid": uid,
                "sid": sample.sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_jsonl, error_row)
            print(
                f"\n[ERROR {requested_alias}/{dataset_name} sid={sample.sid}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            if image is not None:
                image.close()
            if batch is not None:
                del batch
            gc.collect()
            if torch.cuda.is_available() and (
                args.empty_cache_every <= 1
                or max(successful, 1) % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    rows = read_jsonl(samples_jsonl)
    rows = [
        row
        for row in rows
        if state_path(states_dir, str(row["uid"])).exists()
    ]
    if not rows:
        raise RuntimeError(
            f"No successful samples for {requested_alias}/{dataset_name}"
        )
    rows.sort(key=lambda row: str(row["uid"]))

    # Deduplicate resume output by uid and rewrite clean JSONL/CSV.
    deduplicated = {str(row["uid"]): row for row in rows}
    rows = list(deduplicated.values())
    rows.sort(key=lambda row: str(row["uid"]))
    samples_jsonl.unlink(missing_ok=True)
    for row in rows:
        append_jsonl(samples_jsonl, row)
    write_csv(run_dir / "generation_samples.csv", rows)

    labels = np.asarray([REL_TO_ID[row["gt"]] for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    generation_correct = np.asarray(
        [bool(row["generation_correct"]) for row in rows],
        dtype=bool,
    )
    generation_acc = float(np.mean(generation_correct))
    generation_macro = macro_accuracy(
        np.asarray(
            [
                REL_TO_ID.get(str(row["pred"]), -1)
                for row in rows
            ],
            dtype=np.int64,
        ),
        labels,
    )

    states = [
        np.load(state_path(states_dir, str(row["uid"])), allow_pickle=False)
        for row in rows
    ]
    layer_metrics: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for mode in object_modes:
        key = f"object_pair_{mode}"
        x = np.stack(
            [np.asarray(state[key], dtype=np.float32) for state in states],
            axis=0,
        )  # [N,L,D]
        n_layers = int(x.shape[1])

        for layer in range(n_layers):
            layer_x = x[:, layer, :]
            cv_pred, cv_margin, cv_models = crossfit_centroids(
                layer_x,
                labels,
                folds,
                args.centroid_metric,
            )
            in_pred, in_margin = in_sample_centroids(
                layer_x,
                labels,
                args.centroid_metric,
            )

            metric_row: Dict[str, Any] = {
                "model": requested_alias,
                "resolved_model_key": resolved_key,
                "dataset": dataset_name,
                "object_state": mode,
                "layer": int(layer),
                "N": int(len(rows)),
                "centroid_acc": float(np.mean(cv_pred == labels)),
                "centroid_macro_acc": macro_accuracy(cv_pred, labels),
                "in_sample_centroid_acc": float(np.mean(in_pred == labels)),
                "in_sample_centroid_macro_acc": macro_accuracy(in_pred, labels),
                "generation_acc": generation_acc,
                "generation_macro_acc": generation_macro,
                "centroid_minus_generation": float(
                    np.mean(cv_pred == labels) - generation_acc
                ),
                "mean_gt_opposite_margin": safe_mean(cv_margin),
            }
            for relation_id, relation in enumerate(RELATIONS):
                mask = labels == relation_id
                metric_row[f"acc_{relation}"] = (
                    float(np.mean(cv_pred[mask] == labels[mask]))
                    if np.any(mask)
                    else float("nan")
                )
                metric_row[f"N_{relation}"] = int(mask.sum())
            layer_metrics.append(metric_row)

            for index, row in enumerate(rows):
                fold_model = cv_models[int(folds[index])]
                scores = fold_model.scores(layer_x[index : index + 1])[0]
                prediction_rows.append(
                    {
                        "model": requested_alias,
                        "dataset": dataset_name,
                        "object_state": mode,
                        "sid": row["sid"],
                        "uid": row["uid"],
                        "layer": int(layer),
                        "gt": row["gt"],
                        "centroid_pred": ID_TO_REL[int(cv_pred[index])],
                        "centroid_correct": int(cv_pred[index] == labels[index]),
                        "gt_opposite_margin": float(cv_margin[index]),
                        "score_left": float(scores[REL_TO_ID["left"]]),
                        "score_right": float(scores[REL_TO_ID["right"]]),
                        "score_above": float(scores[REL_TO_ID["above"]]),
                        "score_below": float(scores[REL_TO_ID["below"]]),
                        "generation_pred": row["pred"],
                        "generation_correct": int(row["generation_correct"]),
                        "question": row["question"],
                        "generation": row["generation"],
                    }
                )

    for state in states:
        state.close()

    write_csv(run_dir / "layer_metrics.csv", layer_metrics)
    centroid_predictions_path = run_dir / "centroid_predictions.jsonl"
    centroid_predictions_path.unlink(missing_ok=True)
    for row in prediction_rows:
        append_jsonl(centroid_predictions_path, row)

    best_rows: List[Dict[str, Any]] = []
    for mode in object_modes:
        mode_rows = [row for row in layer_metrics if row["object_state"] == mode]
        best = max(mode_rows, key=lambda row: float(row["centroid_acc"]))
        best_rows.append(best)

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": requested_alias,
        "resolved_model_key": resolved_key,
        "dataset": dataset_name,
        "N": len(rows),
        "generation_acc": generation_acc,
        "generation_macro_acc": generation_macro,
        "best_by_object_state": best_rows,
        "relation_counts": {
            relation: int(np.sum(labels == relation_id))
            for relation_id, relation in enumerate(RELATIONS)
        },
        "errors": len(read_jsonl(errors_jsonl)),
    }
    write_json(run_dir / "summary.json", summary)

    print(
        f"\nSUMMARY {requested_alias}/{dataset_name}: "
        f"N={len(rows)} generation_acc={generation_acc:.4f}",
        flush=True,
    )
    for best in best_rows:
        print(
            f"  object_state={best['object_state']} "
            f"best_layer=L{best['layer']} "
            f"centroid_acc={best['centroid_acc']:.4f} "
            f"macro={best['centroid_macro_acc']:.4f} "
            f"centroid-generation={best['centroid_minus_generation']:+.4f}",
            flush=True,
        )
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    requested_models = parse_csv_list(args.models)
    if requested_models == ["all"]:
        requested_models = list(REQUESTED_MODEL_ALIASES)
    datasets = parse_csv_list(args.datasets)
    invalid_datasets = sorted(set(datasets) - {"controlled_a", "coco_two"})
    if invalid_datasets:
        raise ValueError(f"Unsupported datasets: {invalid_datasets}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = import_file(Path(args.base_script), "centroid_benchmark_base")
    _semantic_helper = import_file(
        Path(args.semantic_helper),
        "centroid_benchmark_semantic",
    )
    two_object = import_file(
        Path(args.two_object_script),
        "centroid_benchmark_two_object",
    )

    # Some base scripts expose their own import helper and merged model registry.
    if hasattr(base, "merged_model_specs"):
        specs = base.merged_model_specs(two_object)
    else:
        specs = getattr(two_object, "MODEL_SPECS")

    resolved_models: Dict[str, str] = {}
    for requested in requested_models:
        resolved_models[requested] = resolve_model_key(requested, specs)
    print("Resolved model aliases:")
    for requested, resolved in resolved_models.items():
        spec = specs[resolved]
        print(f"  {requested:14s} -> {resolved:16s} -> {spec.repo_id}")

    dataset_samples: Dict[str, List[BenchmarkSample]] = {}
    for dataset in datasets:
        if dataset == "coco_two":
            dataset_samples[dataset] = load_coco_two_samples(
                args=args,
                base=base,
                two_object=two_object,
            )
        else:
            dataset_samples[dataset] = load_controlled_a_samples(
                args=args,
                base=base,
                two_object=two_object,
            )

    all_summaries: List[Dict[str, Any]] = []
    top_level_errors = output_dir / "benchmark_errors.jsonl"

    for requested_alias in requested_models:
        resolved_key = resolved_models[requested_alias]
        spec = specs[resolved_key]
        model = None
        processor = None
        try:
            model, processor = load_model_and_processor(
                requested_alias=requested_alias,
                resolved_key=resolved_key,
                spec=spec,
                base=base,
                args=args,
            )
            for dataset in datasets:
                summary = extract_model_dataset(
                    requested_alias=requested_alias,
                    resolved_key=resolved_key,
                    dataset_name=dataset,
                    samples=dataset_samples[dataset],
                    model=model,
                    processor=processor,
                    base=base,
                    args=args,
                    output_dir=output_dir,
                )
                all_summaries.append(summary)
        except Exception as exc:
            append_jsonl(
                top_level_errors,
                {
                    "model": requested_alias,
                    "resolved_model_key": resolved_key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"\n[FATAL MODEL ERROR {requested_alias}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    benchmark_rows: List[Dict[str, Any]] = []
    report_lines: List[str] = [
        f"script_version: {SCRIPT_VERSION}",
        "",
    ]
    for summary in all_summaries:
        for best in summary["best_by_object_state"]:
            row = {
                "model": summary["model"],
                "resolved_model_key": summary["resolved_model_key"],
                "dataset": summary["dataset"],
                "object_state": best["object_state"],
                "N": summary["N"],
                "generation_acc": summary["generation_acc"],
                "generation_macro_acc": summary["generation_macro_acc"],
                "best_centroid_layer": best["layer"],
                "best_centroid_acc": best["centroid_acc"],
                "best_centroid_macro_acc": best["centroid_macro_acc"],
                "centroid_minus_generation": best["centroid_minus_generation"],
                "errors": summary["errors"],
            }
            benchmark_rows.append(row)
            report_lines.append(
                f"{row['model']:14s} {row['dataset']:12s} "
                f"state={row['object_state']:4s} N={row['N']:4d} "
                f"gen={row['generation_acc']:.4f} "
                f"centroid={row['best_centroid_acc']:.4f}@L{row['best_centroid_layer']} "
                f"gap={row['centroid_minus_generation']:+.4f}"
            )

    write_csv(output_dir / "benchmark_summary.csv", benchmark_rows)
    (output_dir / "benchmark_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print("\n" + "\n".join(report_lines), flush=True)
    print(f"\nSaved benchmark outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
