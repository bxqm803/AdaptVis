#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centroid-only spatial enhancement analysis.

This script deliberately ignores autoregressive generation and the old fixed
A/B/C groups. For every decoder layer, samples are dynamically divided into:

    Centroid-Correct@L
    Centroid-Wrong@L

according to that layer's own centroid relation prediction.

It then tests GT-free map enhancements:

1. Temperature sharpening
2. Subject/reference competitive separation
3. Cross-layer map ensemble
4. Ensemble + sharpening
5. Ensemble + competition
6. Ensemble + sharpening + competition
7. Entropy-gated adaptive sharpening

The enhancement algorithm never uses GT boxes or GT relations. GT relation is
used only after prediction to calculate accuracy.

Required files in the same repository directory:
    run_spatial_repair_three_experiments_v1.py
    analyze_object_visual_attention_layers_v1.py
    trace_centroid_generation_groups_v2_1.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
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
        "Unable to import run_spatial_repair_three_experiments_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import analyze_object_visual_attention_layers_v1 as layer_base
except Exception as exc:
    raise SystemExit(
        "Unable to import analyze_object_visual_attention_layers_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "centroid-only-dynamic-groups-enhancement-v2"

RELATIONS = ("left", "right", "above", "below")
RELATION_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
ID_TO_RELATION = {index: name for name, index in RELATION_TO_ID.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        help=(
            "Used only to recover the same evaluated sample SIDs. "
            "The old A/B/C labels are ignored."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="output/centroid_only_spatial_enhancement/coco",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=["eager"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic total sample cap.",
    )
    parser.add_argument(
        "--sid",
        type=int,
        default=None,
        help="Analyze one sample only.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failed samples and continue instead of stopping.",
    )

    parser.add_argument(
        "--similarity-map",
        choices=["softmax", "relu"],
        default="softmax",
    )
    parser.add_argument(
        "--similarity-temperature",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--peak-source",
        choices=[
            "similarity_macro",
            "similarity_accuracy",
            "attention_macro",
            "attention_accuracy",
        ],
        default="similarity_macro",
    )
    parser.add_argument(
        "--search-start",
        type=float,
        default=0.0,
        help="Inclusive decoder-depth fraction for peak search.",
    )
    parser.add_argument(
        "--search-end",
        type=float,
        default=1.0,
        help="Exclusive decoder-depth fraction for peak search.",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--plateau-tolerance",
        type=float,
        default=0.01,
        help="Layers within peak macro accuracy minus this value form plateau.",
    )

    parser.add_argument(
        "--enhance-sources",
        default="similarity,attention",
        help="Subset of similarity,attention.",
    )
    parser.add_argument(
        "--sharpen-temperatures",
        default="0.5,0.7,0.85",
    )
    parser.add_argument(
        "--competition-rhos",
        default="0.25,0.5,0.75,1.0",
    )
    parser.add_argument(
        "--ensemble-radii",
        default="1,2,3",
    )
    parser.add_argument(
        "--adaptive-entropy-quantiles",
        default="0.5,0.75",
        help=(
            "Unsupervised entropy thresholds derived from the selected "
            "source's peak-layer entropy distribution."
        ),
    )
    parser.add_argument(
        "--adaptive-temperatures",
        default="0.5,0.7",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument(
        "--core-module",
        default="trace_centroid_generation_groups_v2_1",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"Empty integer list: {text!r}")
    return sorted(set(values))


def parse_sources(text: str) -> List[str]:
    allowed = {"similarity", "attention"}
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    bad = [item for item in values if item not in allowed]
    if bad:
        raise ValueError(f"Unknown enhancement sources: {bad}")
    if not values:
        raise ValueError("No enhancement source selected.")
    return values


# ---------------------------------------------------------------------------
# Map and geometry utilities
# ---------------------------------------------------------------------------

def normalize_map(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.maximum(probability, 0.0)
    total = float(probability.sum())
    if not np.isfinite(total) or total <= 1e-12:
        probability = np.ones_like(probability, dtype=np.float64)
        total = float(probability.sum())
    return probability / total


def entropy_np(probability: np.ndarray) -> float:
    probability = normalize_map(probability)
    n = probability.size
    if n <= 1:
        return 0.0
    value = -np.sum(
        probability * np.log(np.clip(probability, 1e-12, None))
    )
    return float(value / math.log(n))


def sharpen_map(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    probability = normalize_map(probability)
    if temperature <= 0.0:
        raise ValueError("Sharpening temperature must be positive.")
    powered = np.power(
        np.clip(probability, 1e-20, None),
        1.0 / float(temperature),
    )
    return normalize_map(powered)


def competitive_maps(
    subject: np.ndarray,
    reference: np.ndarray,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Suppress patches strongly preferred by the other object.

    The opposing map is peak-normalized to [0,1], so rho has a comparable
    interpretation across different visual-token counts.
    """
    subject = normalize_map(subject)
    reference = normalize_map(reference)

    subject_peak = subject / max(float(subject.max()), 1e-12)
    reference_peak = reference / max(float(reference.max()), 1e-12)

    new_subject = subject * np.exp(-float(rho) * reference_peak)
    new_reference = reference * np.exp(-float(rho) * subject_peak)
    return normalize_map(new_subject), normalize_map(new_reference)


def centroid_np(
    probability: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    probability = normalize_map(probability)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return np.sum(probability[:, None] * coordinates, axis=0)


def relation_from_centroids_np(
    subject_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> int:
    dx = float(subject_xy[0] - reference_xy[0])
    dy = float(subject_xy[1] - reference_xy[1])

    if abs(dx) >= abs(dy):
        return (
            RELATION_TO_ID["left"]
            if dx < 0.0
            else RELATION_TO_ID["right"]
        )
    return (
        RELATION_TO_ID["above"]
        if dy < 0.0
        else RELATION_TO_ID["below"]
    )


def axis_confidence(
    subject_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> float:
    dx = abs(float(subject_xy[0] - reference_xy[0]))
    dy = abs(float(subject_xy[1] - reference_xy[1]))
    return abs(dx - dy) / max(dx + dy, 1e-12)


def top_fraction_mass_np(
    probability: np.ndarray,
    fraction: float,
) -> float:
    probability = normalize_map(probability)
    k = max(
        1,
        min(
            probability.size,
            int(math.ceil(probability.size * float(fraction))),
        ),
    )
    return float(np.sort(probability)[-k:].sum())


def compactness_np(
    probability: np.ndarray,
    coordinates: np.ndarray,
    centroid: np.ndarray,
) -> float:
    probability = normalize_map(probability)
    squared_distance = np.sum(
        (coordinates - centroid[None, :]) ** 2,
        axis=-1,
    )
    return float(np.sum(probability * squared_distance) / 2.0)


def overlap_np(
    subject: np.ndarray,
    reference: np.ndarray,
) -> float:
    subject = normalize_map(subject)
    reference = normalize_map(reference)
    return float(np.minimum(subject, reference).sum())


def macro_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    scores: List[float] = []
    for relation_code in range(len(RELATIONS)):
        mask = ground_truth == relation_code
        if np.any(mask):
            scores.append(
                float(np.mean(predictions[mask] == ground_truth[mask]))
            )
    return float(np.mean(scores)) if scores else float("nan")


def cohen_d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]

    if len(left) < 2 or len(right) < 2:
        return float("nan")

    pooled_variance = (
        (len(left) - 1) * np.var(left, ddof=1)
        + (len(right) - 1) * np.var(right, ddof=1)
    ) / max(len(left) + len(right) - 2, 1)

    pooled_std = math.sqrt(max(float(pooled_variance), 0.0))
    if pooled_std <= 1e-12:
        return 0.0
    return float((np.mean(left) - np.mean(right)) / pooled_std)


def coordinates_from_shape(
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    y = (np.arange(grid_height, dtype=np.float64) + 0.5) / grid_height
    x = (np.arange(grid_width, dtype=np.float64) + 0.5) / grid_width
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def pair_metrics(
    subject: np.ndarray,
    reference: np.ndarray,
    coordinates: np.ndarray,
    top_fraction: float,
) -> Dict[str, float]:
    subject = normalize_map(subject)
    reference = normalize_map(reference)
    subject_centroid = centroid_np(subject, coordinates)
    reference_centroid = centroid_np(reference, coordinates)

    return {
        "pair_entropy": 0.5 * (
            entropy_np(subject) + entropy_np(reference)
        ),
        "pair_topk": 0.5 * (
            top_fraction_mass_np(subject, top_fraction)
            + top_fraction_mass_np(reference, top_fraction)
        ),
        "pair_compactness": 0.5 * (
            compactness_np(subject, coordinates, subject_centroid)
            + compactness_np(reference, coordinates, reference_centroid)
        ),
        "overlap": overlap_np(subject, reference),
        "separation": float(
            np.linalg.norm(subject_centroid - reference_centroid)
        ),
        "axis_confidence": axis_confidence(
            subject_centroid,
            reference_centroid,
        ),
    }


# ---------------------------------------------------------------------------
# Sample containers and configurations
# ---------------------------------------------------------------------------

@dataclass
class SampleMaps:
    sid: int
    gt_code: int
    grid_height: int
    grid_width: int

    similarity_subject: np.ndarray   # [L,V]
    similarity_reference: np.ndarray
    attention_subject: np.ndarray
    attention_reference: np.ndarray

    similarity_prediction: np.ndarray  # [L]
    attention_prediction: np.ndarray

    attention_pair_entropy: np.ndarray
    attention_pair_topk: np.ndarray
    attention_pair_compactness: np.ndarray
    attention_overlap: np.ndarray
    attention_separation: np.ndarray
    attention_centroid_drift: np.ndarray
    attention_pair_visual_mass: np.ndarray
    attention_pair_head_agreement: np.ndarray


@dataclass(frozen=True)
class EnhancementConfig:
    config_id: str
    source: str
    center_layer: int
    ensemble_radius: int = 0
    temperature: float = 1.0
    competition_rho: float = 0.0
    adaptive_entropy_threshold: Optional[float] = None
    adaptive_temperature: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "source": self.source,
            "center_layer": self.center_layer,
            "ensemble_radius": self.ensemble_radius,
            "temperature": self.temperature,
            "competition_rho": self.competition_rho,
            "adaptive_entropy_threshold": self.adaptive_entropy_threshold,
            "adaptive_temperature": self.adaptive_temperature,
        }


# ---------------------------------------------------------------------------
# Layer summaries
# ---------------------------------------------------------------------------

def find_peak_layer(
    *,
    samples: Sequence[SampleMaps],
    source_key: str,
    search_start: float,
    search_end: float,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    n_layers = int(samples[0].similarity_prediction.shape[0])
    gt = np.asarray([sample.gt_code for sample in samples], dtype=np.int16)

    rows: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        similarity_prediction = np.asarray(
            [sample.similarity_prediction[layer] for sample in samples],
            dtype=np.int16,
        )
        attention_prediction = np.asarray(
            [sample.attention_prediction[layer] for sample in samples],
            dtype=np.int16,
        )
        rows.append(
            {
                "layer": layer,
                "similarity_accuracy": float(
                    np.mean(similarity_prediction == gt)
                ),
                "similarity_macro_accuracy": macro_accuracy(
                    similarity_prediction,
                    gt,
                ),
                "attention_accuracy": float(
                    np.mean(attention_prediction == gt)
                ),
                "attention_macro_accuracy": macro_accuracy(
                    attention_prediction,
                    gt,
                ),
            }
        )

    key = {
        "similarity_macro": "similarity_macro_accuracy",
        "similarity_accuracy": "similarity_accuracy",
        "attention_macro": "attention_macro_accuracy",
        "attention_accuracy": "attention_accuracy",
    }[source_key]

    start = max(
        0,
        min(n_layers - 1, int(math.floor(search_start * n_layers))),
    )
    end = max(
        start + 1,
        min(n_layers, int(math.ceil(search_end * n_layers))),
    )

    peak_row = max(
        rows[start:end],
        key=lambda row: (
            float(row[key]),
            float(row["similarity_macro_accuracy"]),
            -int(row["layer"]),
        ),
    )
    return int(peak_row["layer"]), start, end, rows


def dynamic_correct_wrong_rows(
    *,
    samples: Sequence[SampleMaps],
    grouping_source: str,
) -> List[Dict[str, Any]]:
    """
    Dynamic grouping at every layer.

    The group at L is determined only by that same layer's centroid prediction.
    """
    n_layers = int(samples[0].similarity_prediction.shape[0])
    rows: List[Dict[str, Any]] = []

    metric_names = (
        "attention_pair_entropy",
        "attention_pair_topk",
        "attention_pair_compactness",
        "attention_overlap",
        "attention_separation",
        "attention_centroid_drift",
        "attention_pair_visual_mass",
        "attention_pair_head_agreement",
    )

    for layer in range(n_layers):
        predictions = np.asarray(
            [
                (
                    sample.similarity_prediction[layer]
                    if grouping_source == "similarity"
                    else sample.attention_prediction[layer]
                )
                for sample in samples
            ],
            dtype=np.int16,
        )
        gt = np.asarray(
            [sample.gt_code for sample in samples],
            dtype=np.int16,
        )
        correct_mask = predictions == gt

        for group_name, mask in (
            ("Centroid-Correct", correct_mask),
            ("Centroid-Wrong", ~correct_mask),
        ):
            row: Dict[str, Any] = {
                "layer": layer,
                "grouping_source": grouping_source,
                "group": group_name,
                "n": int(mask.sum()),
            }

            for metric in metric_names:
                values = np.asarray(
                    [
                        float(getattr(sample, metric)[layer])
                        for sample in samples
                    ],
                    dtype=np.float64,
                )
                selected = values[mask]
                selected = selected[np.isfinite(selected)]
                row[metric] = (
                    float(np.mean(selected))
                    if len(selected)
                    else float("nan")
                )

            rows.append(row)

    return rows


def dynamic_effect_rows(
    *,
    samples: Sequence[SampleMaps],
    grouping_source: str,
) -> List[Dict[str, Any]]:
    n_layers = int(samples[0].similarity_prediction.shape[0])
    rows: List[Dict[str, Any]] = []

    metric_names = (
        "attention_pair_entropy",
        "attention_pair_topk",
        "attention_pair_compactness",
        "attention_overlap",
        "attention_separation",
        "attention_centroid_drift",
        "attention_pair_visual_mass",
        "attention_pair_head_agreement",
    )

    gt = np.asarray([sample.gt_code for sample in samples], dtype=np.int16)

    for layer in range(n_layers):
        predictions = np.asarray(
            [
                (
                    sample.similarity_prediction[layer]
                    if grouping_source == "similarity"
                    else sample.attention_prediction[layer]
                )
                for sample in samples
            ],
            dtype=np.int16,
        )
        correct_mask = predictions == gt
        wrong_mask = ~correct_mask

        for metric in metric_names:
            values = np.asarray(
                [
                    float(getattr(sample, metric)[layer])
                    for sample in samples
                ],
                dtype=np.float64,
            )
            correct_values = values[correct_mask]
            wrong_values = values[wrong_mask]

            rows.append(
                {
                    "layer": layer,
                    "grouping_source": grouping_source,
                    "metric": metric,
                    "correct_mean": (
                        float(np.nanmean(correct_values))
                        if len(correct_values) else float("nan")
                    ),
                    "wrong_mean": (
                        float(np.nanmean(wrong_values))
                        if len(wrong_values) else float("nan")
                    ),
                    "correct_minus_wrong": (
                        float(np.nanmean(correct_values))
                        - float(np.nanmean(wrong_values))
                        if len(correct_values) and len(wrong_values)
                        else float("nan")
                    ),
                    "cohen_d_correct_minus_wrong": cohen_d(
                        correct_values,
                        wrong_values,
                    ),
                    "correct_n": int(correct_mask.sum()),
                    "wrong_n": int(wrong_mask.sum()),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Enhancement construction and evaluation
# ---------------------------------------------------------------------------

def source_maps(
    sample: SampleMaps,
    source: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if source == "similarity":
        return sample.similarity_subject, sample.similarity_reference
    if source == "attention":
        return sample.attention_subject, sample.attention_reference
    raise ValueError(source)


def ensemble_maps(
    maps: np.ndarray,
    center_layer: int,
    radius: int,
) -> np.ndarray:
    n_layers = int(maps.shape[0])
    left = max(0, center_layer - radius)
    right = min(n_layers, center_layer + radius + 1)

    normalized = np.stack(
        [normalize_map(maps[layer]) for layer in range(left, right)],
        axis=0,
    )
    return normalize_map(normalized.mean(axis=0))


def transform_pair(
    *,
    subject: np.ndarray,
    reference: np.ndarray,
    config: EnhancementConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    subject = normalize_map(subject)
    reference = normalize_map(reference)

    if config.adaptive_entropy_threshold is not None:
        subject_entropy = entropy_np(subject)
        reference_entropy = entropy_np(reference)

        if subject_entropy >= config.adaptive_entropy_threshold:
            subject = sharpen_map(
                subject,
                config.adaptive_temperature,
            )
        if reference_entropy >= config.adaptive_entropy_threshold:
            reference = sharpen_map(
                reference,
                config.adaptive_temperature,
            )

    if config.temperature < 1.0:
        subject = sharpen_map(subject, config.temperature)
        reference = sharpen_map(reference, config.temperature)

    if config.competition_rho > 0.0:
        subject, reference = competitive_maps(
            subject,
            reference,
            config.competition_rho,
        )

    return normalize_map(subject), normalize_map(reference)


def build_enhancement_configs(
    *,
    sources: Sequence[str],
    peak_layer: int,
    sharpen_temperatures: Sequence[float],
    competition_rhos: Sequence[float],
    ensemble_radii: Sequence[int],
    adaptive_thresholds_by_source: Mapping[str, Sequence[float]],
    adaptive_temperatures: Sequence[float],
) -> List[EnhancementConfig]:
    configs: List[EnhancementConfig] = []

    for source in sources:
        configs.append(
            EnhancementConfig(
                config_id=f"{source}__single_L{peak_layer}__baseline",
                source=source,
                center_layer=peak_layer,
            )
        )

        for temperature in sharpen_temperatures:
            configs.append(
                EnhancementConfig(
                    config_id=(
                        f"{source}__single_L{peak_layer}"
                        f"__sharpen_tau{temperature:g}"
                    ),
                    source=source,
                    center_layer=peak_layer,
                    temperature=float(temperature),
                )
            )

        for rho in competition_rhos:
            configs.append(
                EnhancementConfig(
                    config_id=(
                        f"{source}__single_L{peak_layer}"
                        f"__competition_rho{rho:g}"
                    ),
                    source=source,
                    center_layer=peak_layer,
                    competition_rho=float(rho),
                )
            )

        for temperature in sharpen_temperatures:
            for rho in competition_rhos:
                configs.append(
                    EnhancementConfig(
                        config_id=(
                            f"{source}__single_L{peak_layer}"
                            f"__tau{temperature:g}_rho{rho:g}"
                        ),
                        source=source,
                        center_layer=peak_layer,
                        temperature=float(temperature),
                        competition_rho=float(rho),
                    )
                )

        for threshold in adaptive_thresholds_by_source[source]:
            for temperature in adaptive_temperatures:
                configs.append(
                    EnhancementConfig(
                        config_id=(
                            f"{source}__single_L{peak_layer}"
                            f"__adaptive_H{threshold:.4f}"
                            f"_tau{temperature:g}"
                        ),
                        source=source,
                        center_layer=peak_layer,
                        adaptive_entropy_threshold=float(threshold),
                        adaptive_temperature=float(temperature),
                    )
                )

        for radius in ensemble_radii:
            configs.append(
                EnhancementConfig(
                    config_id=(
                        f"{source}__ensemble_L{peak_layer}"
                        f"_r{radius}__baseline"
                    ),
                    source=source,
                    center_layer=peak_layer,
                    ensemble_radius=int(radius),
                )
            )

            for temperature in sharpen_temperatures:
                configs.append(
                    EnhancementConfig(
                        config_id=(
                            f"{source}__ensemble_L{peak_layer}"
                            f"_r{radius}__tau{temperature:g}"
                        ),
                        source=source,
                        center_layer=peak_layer,
                        ensemble_radius=int(radius),
                        temperature=float(temperature),
                    )
                )

            for rho in competition_rhos:
                configs.append(
                    EnhancementConfig(
                        config_id=(
                            f"{source}__ensemble_L{peak_layer}"
                            f"_r{radius}__rho{rho:g}"
                        ),
                        source=source,
                        center_layer=peak_layer,
                        ensemble_radius=int(radius),
                        competition_rho=float(rho),
                    )
                )

            for temperature in sharpen_temperatures:
                for rho in competition_rhos:
                    configs.append(
                        EnhancementConfig(
                            config_id=(
                                f"{source}__ensemble_L{peak_layer}"
                                f"_r{radius}__tau{temperature:g}"
                                f"_rho{rho:g}"
                            ),
                            source=source,
                            center_layer=peak_layer,
                            ensemble_radius=int(radius),
                            temperature=float(temperature),
                            competition_rho=float(rho),
                        )
                    )

    unique: Dict[str, EnhancementConfig] = {}
    for config in configs:
        unique[config.config_id] = config
    return list(unique.values())


def predict_with_config(
    sample: SampleMaps,
    config: EnhancementConfig,
) -> Tuple[int, Dict[str, float]]:
    subject_layers, reference_layers = source_maps(
        sample,
        config.source,
    )
    coordinates = coordinates_from_shape(
        sample.grid_height,
        sample.grid_width,
    )

    subject = ensemble_maps(
        subject_layers,
        config.center_layer,
        config.ensemble_radius,
    )
    reference = ensemble_maps(
        reference_layers,
        config.center_layer,
        config.ensemble_radius,
    )

    subject, reference = transform_pair(
        subject=subject,
        reference=reference,
        config=config,
    )

    subject_centroid = centroid_np(subject, coordinates)
    reference_centroid = centroid_np(reference, coordinates)
    prediction = relation_from_centroids_np(
        subject_centroid,
        reference_centroid,
    )

    metrics = pair_metrics(
        subject,
        reference,
        coordinates,
        top_fraction=0.10,
    )
    return prediction, metrics


def evaluate_enhancements(
    *,
    samples: Sequence[SampleMaps],
    configs: Sequence[EnhancementConfig],
    baseline_source: str,
    peak_layer: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gt = np.asarray([sample.gt_code for sample in samples], dtype=np.int16)

    baseline_prediction = np.asarray(
        [
            (
                sample.similarity_prediction[peak_layer]
                if baseline_source == "similarity"
                else sample.attention_prediction[peak_layer]
            )
            for sample in samples
        ],
        dtype=np.int16,
    )
    baseline_correct = baseline_prediction == gt
    baseline_accuracy = float(np.mean(baseline_correct))
    baseline_macro = macro_accuracy(baseline_prediction, gt)

    summary_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for config in tqdm(
        configs,
        desc="offline-centroid-enhancement",
        unit="config",
        dynamic_ncols=True,
    ):
        predictions: List[int] = []
        pair_metric_values: Dict[str, List[float]] = defaultdict(list)

        for sample in samples:
            prediction, metrics = predict_with_config(sample, config)
            predictions.append(prediction)
            for key, value in metrics.items():
                pair_metric_values[key].append(float(value))

        prediction_array = np.asarray(predictions, dtype=np.int16)
        correct = prediction_array == gt

        wrong_to_correct_mask = (~baseline_correct) & correct
        correct_to_wrong_mask = baseline_correct & (~correct)

        accuracy = float(np.mean(correct))
        macro = macro_accuracy(prediction_array, gt)

        row: Dict[str, Any] = {
            **config.as_dict(),
            "n": len(samples),
            "baseline_source": baseline_source,
            "baseline_peak_layer": peak_layer,
            "baseline_accuracy": baseline_accuracy,
            "baseline_macro_accuracy": baseline_macro,
            "enhanced_accuracy": accuracy,
            "enhanced_macro_accuracy": macro,
            "delta_accuracy": accuracy - baseline_accuracy,
            "delta_macro_accuracy": macro - baseline_macro,
            "delta_correct": int(correct.sum() - baseline_correct.sum()),
            "wrong_to_correct": int(wrong_to_correct_mask.sum()),
            "wrong_to_correct_rate": (
                float(wrong_to_correct_mask.sum() / max((~baseline_correct).sum(), 1))
            ),
            "correct_to_wrong": int(correct_to_wrong_mask.sum()),
            "correct_to_wrong_rate": (
                float(correct_to_wrong_mask.sum() / max(baseline_correct.sum(), 1))
            ),
        }

        for metric, values in pair_metric_values.items():
            row[f"enhanced_{metric}"] = float(np.mean(values))

        for relation_code, relation_name in ID_TO_RELATION.items():
            relation_mask = gt == relation_code
            row[f"accuracy_{relation_name}"] = (
                float(np.mean(correct[relation_mask]))
                if np.any(relation_mask)
                else float("nan")
            )

        summary_rows.append(row)

        for index, sample in enumerate(samples):
            sample_rows.append(
                {
                    "config_id": config.config_id,
                    "source": config.source,
                    "sid": sample.sid,
                    "gt": ID_TO_RELATION[sample.gt_code],
                    "baseline_prediction": ID_TO_RELATION[
                        int(baseline_prediction[index])
                    ],
                    "enhanced_prediction": ID_TO_RELATION[
                        int(prediction_array[index])
                    ],
                    "baseline_correct": bool(baseline_correct[index]),
                    "enhanced_correct": bool(correct[index]),
                    "wrong_to_correct": bool(wrong_to_correct_mask[index]),
                    "correct_to_wrong": bool(correct_to_wrong_mask[index]),
                }
            )

    summary_rows.sort(
        key=lambda row: (
            -float(row["delta_accuracy"]),
            -float(row["delta_macro_accuracy"]),
            -float(row["wrong_to_correct_rate"]),
            float(row["correct_to_wrong_rate"]),
        )
    )
    return summary_rows, sample_rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
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


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def plateau_layers(
    layer_rows: Sequence[Mapping[str, Any]],
    peak_layer: int,
    source_key: str,
    tolerance: float,
) -> List[int]:
    key = {
        "similarity_macro": "similarity_macro_accuracy",
        "similarity_accuracy": "similarity_accuracy",
        "attention_macro": "attention_macro_accuracy",
        "attention_accuracy": "attention_accuracy",
    }[source_key]
    peak_value = float(layer_rows[peak_layer][key])
    threshold = peak_value - float(tolerance)

    eligible = {
        int(row["layer"])
        for row in layer_rows
        if float(row[key]) >= threshold
    }

    result = [peak_layer]
    left = peak_layer - 1
    while left in eligible:
        result.insert(0, left)
        left -= 1
    right = peak_layer + 1
    while right in eligible:
        result.append(right)
        right += 1
    return result


def print_report(
    *,
    model_name: str,
    layer_rows: Sequence[Mapping[str, Any]],
    correct_wrong_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    enhancement_rows: Sequence[Mapping[str, Any]],
    peak_layer: int,
    search_start: int,
    search_end: int,
    radius: int,
    plateau: Sequence[int],
) -> str:
    lines: List[str] = []
    lines.append("=" * 166)
    lines.append(f"MODEL: {model_name}")
    lines.append(
        f"Peak search L{search_start}-L{search_end - 1}; "
        f"selected L{peak_layer}; plateau={list(plateau)}"
    )
    lines.append("=" * 166)

    left = max(0, peak_layer - radius)
    right = min(len(layer_rows) - 1, peak_layer + radius)

    lines.append(
        "Layer  SimAcc  SimMacro  AttnAcc  AttnMacro"
    )
    for layer in range(left, right + 1):
        row = layer_rows[layer]
        marker = "*" if layer == peak_layer else " "
        lines.append(
            f"{marker}L{layer:02d}  "
            f"{float(row['similarity_accuracy']):7.3f}  "
            f"{float(row['similarity_macro_accuracy']):8.3f}  "
            f"{float(row['attention_accuracy']):7.3f}  "
            f"{float(row['attention_macro_accuracy']):9.3f}"
        )

    lookup = {
        (int(row["layer"]), str(row["group"])): row
        for row in correct_wrong_rows
        if row["grouping_source"] == "similarity"
    }

    lines.append("")
    lines.append(
        "Dynamic grouping by SAME-LAYER similarity centroid correctness"
    )
    lines.append(
        "Layer Group              N   Entropy  Top10  Compact  "
        "Overlap  Separation  Drift  VisMass  HeadAgree"
    )
    for layer in range(left, right + 1):
        for group in ("Centroid-Correct", "Centroid-Wrong"):
            row = lookup.get((layer, group))
            if row is None:
                continue
            lines.append(
                f"L{layer:02d}  {group:18s} "
                f"{int(row['n']):4d}  "
                f"{float(row['attention_pair_entropy']):7.4f}  "
                f"{float(row['attention_pair_topk']):6.4f}  "
                f"{float(row['attention_pair_compactness']):7.4f}  "
                f"{float(row['attention_overlap']):7.4f}  "
                f"{float(row['attention_separation']):10.4f}  "
                f"{float(row['attention_centroid_drift']):6.4f}  "
                f"{float(row['attention_pair_visual_mass']):7.4f}  "
                f"{float(row['attention_pair_head_agreement']):9.4f}"
            )

    effect_lookup = {
        (int(row["layer"]), str(row["metric"])): row
        for row in effect_rows
        if row["grouping_source"] == "similarity"
    }

    lines.append("")
    lines.append("Correct-Wrong effects at selected peak")
    lines.append(
        "Metric                                  Correct     Wrong      C-W        d"
    )
    for metric in (
        "attention_pair_entropy",
        "attention_pair_topk",
        "attention_pair_compactness",
        "attention_overlap",
        "attention_separation",
        "attention_centroid_drift",
        "attention_pair_visual_mass",
        "attention_pair_head_agreement",
    ):
        row = effect_lookup.get((peak_layer, metric))
        if row is None:
            continue
        lines.append(
            f"{metric:38s} "
            f"{float(row['correct_mean']):8.4f}  "
            f"{float(row['wrong_mean']):8.4f}  "
            f"{float(row['correct_minus_wrong']):8.4f}  "
            f"{float(row['cohen_d_correct_minus_wrong']):7.3f}"
        )

    lines.append("")
    lines.append("Top centroid enhancement configurations")
    lines.append(
        "Config                                                                  "
        "BaseAcc  NewAcc   ΔAcc  ΔMacro  W→C  C→W"
    )
    for row in enhancement_rows[:20]:
        lines.append(
            f"{str(row['config_id'])[:70]:70s} "
            f"{float(row['baseline_accuracy']):7.3f}  "
            f"{float(row['enhanced_accuracy']):6.3f}  "
            f"{float(row['delta_accuracy']):6.3f}  "
            f"{float(row['delta_macro_accuracy']):7.3f}  "
            f"{int(row['wrong_to_correct']):3d}  "
            f"{int(row['correct_to_wrong']):3d}"
        )

    report = "\n".join(lines)
    print("\n" + report)
    return report


# ---------------------------------------------------------------------------
# Model run
# ---------------------------------------------------------------------------

def select_sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_samples: Optional[int],
    seed: int,
    sid: Optional[int],
) -> List[Dict[str, Any]]:
    selected = [dict(row) for row in rows]

    if sid is not None:
        selected = [
            row for row in selected
            if int(row.get("sid", -1)) == sid
        ]

    # The old group field is deliberately ignored. Deduplicate by SID.
    unique: Dict[int, Dict[str, Any]] = {}
    for row in selected:
        unique[int(row["sid"])] = row
    selected = [unique[sid_value] for sid_value in sorted(unique)]

    if max_samples is not None and len(selected) > max_samples:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_samples)
        selected.sort(key=lambda row: int(row["sid"]))

    return selected


def run_model(
    *,
    args: argparse.Namespace,
    core: Any,
    backend_module: Any,
    model_name: str,
    enhance_sources: Sequence[str],
    sharpen_temperatures: Sequence[float],
    competition_rhos: Sequence[float],
    ensemble_radii: Sequence[int],
    adaptive_entropy_quantiles: Sequence[float],
    adaptive_temperatures: Sequence[float],
) -> None:
    metadata_path = (
        Path(args.input_root)
        / model_name
        / "pass2_transfer_trace"
        / "sample_metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    prior_rows = select_sample_rows(
        base.read_jsonl(metadata_path),
        max_samples=args.max_samples,
        seed=args.seed,
        sid=args.sid,
    )
    if not prior_rows:
        raise RuntimeError(f"No samples selected for {model_name}.")

    records, audit = backend_module.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {int(record.sid): record for record in records}

    prompt_path = core.resolve_prompt_path(args)
    prompt_rows = core.load_standard_prompts(prompt_path)

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

    print("\n" + "=" * 166)
    print(f"LOADING {model_name}: {spec.repo_id}")
    print("=" * 166)

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

    output_dir = Path(args.output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = (
        output_dir / "layer_accuracy.csv",
        output_dir / "dynamic_correct_wrong.csv",
        output_dir / "dynamic_correct_wrong_effects.csv",
        output_dir / "enhancement_summary.csv",
        output_dir / "enhancement_sample_results.jsonl",
        output_dir / "report.txt",
        output_dir / "run_config.json",
        output_dir / "errors.jsonl",
    )

    if args.overwrite:
        for path in output_files:
            if path.exists():
                path.unlink()
    elif any(path.exists() for path in output_files[:-1]):
        raise FileExistsError(
            f"Results already exist in {output_dir}; pass --overwrite."
        )

    capture = layer_base.LayerGroundingCapture(
        layers=decoder_layers,
        model=model,
        processor=processor,
        similarity_mode=args.similarity_map,
        similarity_temperature=args.similarity_temperature,
        top_fraction=args.top_fraction,
        save_maps=True,
    )

    samples: List[SampleMaps] = []
    metadata_rows: List[Dict[str, Any]] = []

    progress = tqdm(
        prior_rows,
        desc=f"centroid-only:{model_name}",
        unit="sample",
        dynamic_ncols=True,
    )

    try:
        for sample_index, prior in enumerate(progress, 1):
            sid = int(prior["sid"])
            image = None
            batch = None

            try:
                if sid not in record_by_sid or sid not in prompt_rows:
                    raise RuntimeError(
                        f"Missing record or prompt for sid={sid}."
                    )

                record = record_by_sid[sid]
                prompt_row = prompt_rows[sid]

                image = core.record_image(record)
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = base.normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATION_TO_ID:
                    raise RuntimeError(
                        f"Invalid GT for sid={sid}: "
                        f"{prompt_row['answer_raw']!r}"
                    )
                gt_code = RELATION_TO_ID[gt]

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

                capture.configure(
                    prompt_spec=prompt_spec,
                    batch=batch,
                    image_size=tuple(image.size),
                    gt_code=gt_code,
                )

                with torch.inference_mode():
                    _ = model(
                        **batch,
                        use_cache=False,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                arrays = layer_base.sample_result_to_arrays(
                    capture.results,
                    n_layers=n_layers,
                )

                if capture.grid_shape is None:
                    raise RuntimeError(
                        f"Visual grid not resolved for sid={sid}."
                    )

                sample = SampleMaps(
                    sid=sid,
                    gt_code=gt_code,
                    grid_height=int(capture.grid_shape[0]),
                    grid_width=int(capture.grid_shape[1]),

                    similarity_subject=np.asarray(
                        arrays["similarity_subject_map"],
                        dtype=np.float32,
                    ),
                    similarity_reference=np.asarray(
                        arrays["similarity_reference_map"],
                        dtype=np.float32,
                    ),
                    attention_subject=np.asarray(
                        arrays["attention_subject_map"],
                        dtype=np.float32,
                    ),
                    attention_reference=np.asarray(
                        arrays["attention_reference_map"],
                        dtype=np.float32,
                    ),

                    similarity_prediction=np.asarray(
                        arrays["similarity_prediction"],
                        dtype=np.int16,
                    ),
                    attention_prediction=np.asarray(
                        arrays["attention_prediction"],
                        dtype=np.int16,
                    ),

                    attention_pair_entropy=0.5 * (
                        np.asarray(
                            arrays["attention_subject_entropy"],
                            dtype=np.float32,
                        )
                        + np.asarray(
                            arrays["attention_reference_entropy"],
                            dtype=np.float32,
                        )
                    ),
                    attention_pair_topk=0.5 * (
                        np.asarray(
                            arrays["attention_subject_topk"],
                            dtype=np.float32,
                        )
                        + np.asarray(
                            arrays["attention_reference_topk"],
                            dtype=np.float32,
                        )
                    ),
                    attention_pair_compactness=0.5 * (
                        np.asarray(
                            arrays["attention_subject_compactness"],
                            dtype=np.float32,
                        )
                        + np.asarray(
                            arrays["attention_reference_compactness"],
                            dtype=np.float32,
                        )
                    ),
                    attention_overlap=np.asarray(
                        arrays["attention_overlap"],
                        dtype=np.float32,
                    ),
                    attention_separation=np.asarray(
                        arrays["attention_separation"],
                        dtype=np.float32,
                    ),
                    attention_centroid_drift=np.asarray(
                        arrays["attention_centroid_drift"],
                        dtype=np.float32,
                    ),
                    attention_pair_visual_mass=0.5 * (
                        np.asarray(
                            arrays["attention_subject_visual_mass"],
                            dtype=np.float32,
                        )
                        + np.asarray(
                            arrays["attention_reference_visual_mass"],
                            dtype=np.float32,
                        )
                    ),
                    attention_pair_head_agreement=0.5 * (
                        np.asarray(
                            arrays["attention_subject_head_agreement"],
                            dtype=np.float32,
                        )
                        + np.asarray(
                            arrays["attention_reference_head_agreement"],
                            dtype=np.float32,
                        )
                    ),
                )
                samples.append(sample)
                metadata_rows.append(
                    {
                        "sid": sid,
                        "gt": gt,
                        "subject": subject,
                        "reference": reference,
                        "grid_height": sample.grid_height,
                        "grid_width": sample.grid_width,
                    }
                )

            except Exception as exc:
                with (output_dir / "errors.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        json.dumps(
                            {
                                "model": model_name,
                                "sid": sid,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback_tail": traceback.format_exc().splitlines()[-25:],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if not args.continue_on_error:
                    raise

            finally:
                capture.reset()
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

            if (
                args.print_every > 0
                and sample_index % args.print_every == 0
            ):
                progress.set_postfix_str(
                    f"success={len(samples)}",
                    refresh=False,
                )

            if (
                args.empty_cache_every > 0
                and sample_index % args.empty_cache_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

    finally:
        progress.close()
        capture.close()

    if not samples:
        raise RuntimeError(f"No successful samples for {model_name}.")

    peak_layer, search_start, search_end, layer_rows = find_peak_layer(
        samples=samples,
        source_key=args.peak_source,
        search_start=args.search_start,
        search_end=args.search_end,
    )

    correct_wrong_rows: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    for grouping_source in ("similarity", "attention"):
        correct_wrong_rows.extend(
            dynamic_correct_wrong_rows(
                samples=samples,
                grouping_source=grouping_source,
            )
        )
        effect_rows.extend(
            dynamic_effect_rows(
                samples=samples,
                grouping_source=grouping_source,
            )
        )

    adaptive_thresholds_by_source: Dict[str, List[float]] = {}
    for source in enhance_sources:
        entropy_values: List[float] = []
        for sample in samples:
            subject_maps, reference_maps = source_maps(sample, source)
            entropy_values.extend(
                [
                    entropy_np(subject_maps[peak_layer]),
                    entropy_np(reference_maps[peak_layer]),
                ]
            )

        adaptive_thresholds_by_source[source] = [
            float(np.quantile(entropy_values, quantile))
            for quantile in adaptive_entropy_quantiles
        ]

    configs = build_enhancement_configs(
        sources=enhance_sources,
        peak_layer=peak_layer,
        sharpen_temperatures=sharpen_temperatures,
        competition_rhos=competition_rhos,
        ensemble_radii=ensemble_radii,
        adaptive_thresholds_by_source=adaptive_thresholds_by_source,
        adaptive_temperatures=adaptive_temperatures,
    )

    baseline_source = (
        "similarity"
        if args.peak_source.startswith("similarity")
        else "attention"
    )
    enhancement_rows, enhancement_sample_rows = evaluate_enhancements(
        samples=samples,
        configs=configs,
        baseline_source=baseline_source,
        peak_layer=peak_layer,
    )

    plateau = plateau_layers(
        layer_rows=layer_rows,
        peak_layer=peak_layer,
        source_key=args.peak_source,
        tolerance=args.plateau_tolerance,
    )

    write_csv(output_dir / "layer_accuracy.csv", layer_rows)
    write_csv(
        output_dir / "dynamic_correct_wrong.csv",
        correct_wrong_rows,
    )
    write_csv(
        output_dir / "dynamic_correct_wrong_effects.csv",
        effect_rows,
    )
    write_csv(
        output_dir / "enhancement_summary.csv",
        enhancement_rows,
    )
    write_jsonl(
        output_dir / "enhancement_sample_results.jsonl",
        enhancement_sample_rows,
    )
    write_jsonl(
        output_dir / "sample_metadata.jsonl",
        metadata_rows,
    )

    report = print_report(
        model_name=model_name,
        layer_rows=layer_rows,
        correct_wrong_rows=correct_wrong_rows,
        effect_rows=effect_rows,
        enhancement_rows=enhancement_rows,
        peak_layer=peak_layer,
        search_start=search_start,
        search_end=search_end,
        radius=args.neighbor_radius,
        plateau=plateau,
    )
    (output_dir / "report.txt").write_text(
        report,
        encoding="utf-8",
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "model": model_name,
                "repo_id": spec.repo_id,
                "decoder_path": layers_path,
                "n_layers": n_layers,
                "sample_count": len(samples),
                "old_groups_used_for_analysis": False,
                "dynamic_grouping": (
                    "At every layer, compare that layer's own "
                    "centroid-correct and centroid-wrong samples."
                ),
                "peak_source": args.peak_source,
                "peak_layer": peak_layer,
                "plateau_layers": plateau,
                "plateau_tolerance": args.plateau_tolerance,
                "search_start": args.search_start,
                "search_end": args.search_end,
                "similarity_map": args.similarity_map,
                "similarity_temperature": args.similarity_temperature,
                "top_fraction": args.top_fraction,
                "enhance_sources": list(enhance_sources),
                "sharpen_temperatures": list(sharpen_temperatures),
                "competition_rhos": list(competition_rhos),
                "ensemble_radii": list(ensemble_radii),
                "adaptive_entropy_quantiles": list(
                    adaptive_entropy_quantiles
                ),
                "adaptive_thresholds_by_source": (
                    adaptive_thresholds_by_source
                ),
                "adaptive_temperatures": list(adaptive_temperatures),
                "enhancement_config_count": len(configs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_dir}")
    print(f"  report:                  {output_dir / 'report.txt'}")
    print(f"  layer accuracy:          {output_dir / 'layer_accuracy.csv'}")
    print(f"  dynamic correct/wrong:   {output_dir / 'dynamic_correct_wrong.csv'}")
    print(f"  correct/wrong effects:   {output_dir / 'dynamic_correct_wrong_effects.csv'}")
    print(f"  enhancement summary:     {output_dir / 'enhancement_summary.csv'}")
    print(f"  enhancement samples:     {output_dir / 'enhancement_sample_results.jsonl'}")

    del model, processor, decoder_layers, samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must lie in (0,1].")
    if not 0.0 <= args.search_start < args.search_end <= 1.0:
        raise ValueError(
            "Require 0 <= search-start < search-end <= 1."
        )
    if args.similarity_temperature <= 0.0:
        raise ValueError(
            "--similarity-temperature must be positive."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = base.parse_models(args.models)
    enhance_sources = parse_sources(args.enhance_sources)
    sharpen_temperatures = base.parse_float_list(
        args.sharpen_temperatures
    )
    competition_rhos = base.parse_float_list(
        args.competition_rhos
    )
    ensemble_radii = parse_int_list(args.ensemble_radii)
    adaptive_entropy_quantiles = base.parse_float_list(
        args.adaptive_entropy_quantiles
    )
    adaptive_temperatures = base.parse_float_list(
        args.adaptive_temperatures
    )

    for temperature in (
        list(sharpen_temperatures)
        + list(adaptive_temperatures)
    ):
        if not 0.0 < temperature <= 1.0:
            raise ValueError(
                "Sharpen temperatures must lie in (0,1]."
            )
    for rho in competition_rhos:
        if rho < 0.0:
            raise ValueError(
                "Competition rho must be non-negative."
            )
    for radius in ensemble_radii:
        if radius < 1:
            raise ValueError(
                "Ensemble radii must be positive integers."
            )
    for quantile in adaptive_entropy_quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(
                "Adaptive entropy quantiles must lie in [0,1]."
            )

    core = base.import_core(args.core_module)
    backend_module = core.import_two_object_module()

    print("=" * 166)
    print("CENTROID-ONLY DYNAMIC CORRECT/WRONG ANALYSIS AND GT-FREE ENHANCEMENT")
    print("=" * 166)
    print(f"models={models}")
    print(f"enhancement sources={enhance_sources}")
    print(
        "Old A/B/C labels are ignored. Every layer is grouped dynamically "
        "by that layer's own centroid correctness."
    )
    print(
        "GT boxes and GT relations are not used by enhancement transforms. "
        "GT relation is used only for final evaluation."
    )

    completed = 0
    failures: List[Tuple[str, str]] = []

    for model_name in models:
        try:
            run_model(
                args=args,
                core=core,
                backend_module=backend_module,
                model_name=model_name,
                enhance_sources=enhance_sources,
                sharpen_temperatures=sharpen_temperatures,
                competition_rhos=competition_rhos,
                ensemble_radii=ensemble_radii,
                adaptive_entropy_quantiles=adaptive_entropy_quantiles,
                adaptive_temperatures=adaptive_temperatures,
            )
            completed += 1
        except Exception as exc:
            failures.append(
                (model_name, f"{type(exc).__name__}: {exc}")
            )
            print(
                f"\n[ERROR] {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    print("\n" + "=" * 166)
    print(f"COMPLETE: {completed}/{len(models)} models")
    for model_name, error in failures:
        print(f"  failed {model_name}: {error}")

    if completed == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
