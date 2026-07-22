#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare three spatial-reasoning groups with the same frozen-model tracing pipeline.

Groups
------
A: centroid correct + generation correct
B: centroid correct + generation wrong
C: centroid wrong   + generation correct
Optional:
D: centroid wrong   + generation wrong

This script deliberately does NOT use subject-reference hidden-state subtraction
as an analysis signal. It keeps subject and reference routes separate and studies:

1. last-token attention mass to subject / reference / visual tokens;
2. full routed A*V contribution norms, including isolated o_proj output;
3. layerwise GT-vs-opposite logit-lens margins;
4. attention, MLP, and whole-block margin gains;
5. selected spatial-head visual grounding and routed contributions;
6. raw and nuisance-matched pairwise comparisons.

It reuses model/backend and hook utilities from:
    trace_centroid_generation_groups_v2_1.py

Place this file in the AdaptVis repository root, next to that script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
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


SCRIPT_VERSION = "compare-centroid-generation-transfer-groups-v1.1"

RELATIONS = ("left", "right", "above", "below")
RELATION_TO_INDEX = {name: i for i, name in enumerate(RELATIONS)}

GROUP_A = "A_centroid_correct_generation_correct"
GROUP_B = "B_centroid_correct_generation_wrong"
GROUP_C = "C_centroid_wrong_generation_correct"
GROUP_D = "D_centroid_wrong_generation_wrong"

PRIMARY_GROUPS = (GROUP_A, GROUP_B, GROUP_C)
ALL_GROUPS = (GROUP_A, GROUP_B, GROUP_C, GROUP_D)

PAIR_SPECS = (
    ("A_vs_B", GROUP_A, GROUP_B),
    ("A_vs_C", GROUP_A, GROUP_C),
    ("B_vs_C", GROUP_B, GROUP_C),
)

GROUP_TO_CODE = {
    GROUP_A: 0,
    GROUP_B: 1,
    GROUP_C: 2,
    GROUP_D: 3,
}


# ---------------------------------------------------------------------------
# Generic utilities
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
        "--prior-dir",
        required=True,
        help="Directory containing config.json, centroid_analysis.jsonl, generation.jsonl.",
    )
    p.add_argument("--centroid-analysis-jsonl", default=None)
    p.add_argument("--generation-jsonl", default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--core-module",
        default="trace_centroid_generation_groups_v2_1",
        help="Existing repository trace module reused by this script.",
    )
    p.add_argument(
        "--selected-heads",
        default=None,
        help=(
            "Optional override, e.g. '12:3,15:7,20:1'. "
            "If omitted, read prior-dir/config.json when available; otherwise "
            "automatically trace all heads at the model's reference layer."
        ),
    )

    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument(
        "--include-group-d",
        action="store_true",
        help="Also trace centroid-wrong + generation-wrong samples.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--make-plots", action="store_true")

    p.add_argument(
        "--match-features-ab",
        default=(
            "centroid_confidence,axis_confidence,head_agreement,"
            "mean_separation,mean_visual_mass,prompt_length,"
            "subject_token_count,reference_token_count"
        ),
        help="A/B nuisance-matching features. A and B both have correct centroid.",
    )
    p.add_argument(
        "--match-features-other",
        default=(
            "mean_visual_mass,prompt_length,subject_token_count,"
            "reference_token_count,n_visual_tokens"
        ),
        help="A/C and B/C matching features; excludes centroid correctness signals.",
    )
    p.add_argument(
        "--match-reuse",
        action="store_true",
        help="Allow the comparison-side sample to be reused in nearest-neighbour matching.",
    )
    p.add_argument(
        "--persistent-layers",
        type=int,
        default=2,
        help="Consecutive significant layers required for a breakpoint.",
    )
    p.add_argument(
        "--minimum-effect",
        type=float,
        default=0.30,
        help="Minimum absolute Cohen's d for a persistent breakpoint.",
    )
    return p.parse_args()


def import_core(name: str):
    module = importlib.import_module(name)
    required = [
        "read_jsonl",
        "append_jsonl",
        "import_two_object_module",
        "resolve_prompt_path",
        "load_standard_prompts",
        "record_image",
        "normalize_relation",
        "make_question_batch",
        "resolve_dtype",
        "configure_processor",
        "resolve_decoder_layers",
        "resolve_final_norm",
        "label_token_id_variants",
        "relation_token_rows",
        "LayerTraceCollector",
        "trace_one_prompt",
        "load_selected_heads",
    ]
    missing = [item for item in required if not hasattr(module, item)]
    if missing:
        raise RuntimeError(
            f"Core module {name!r} is missing required symbols: {missing}. "
            "Use the llava16 version of trace_centroid_generation_groups_v2_1.py."
        )
    return module


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "correct"}:
        return True
    if text in {"0", "false", "no", "n", "wrong", "incorrect"}:
        return False
    return default


def first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def parse_feature_list(text: str) -> List[str]:
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    if not items:
        raise ValueError("Matching feature list is empty.")
    return items


def parse_selected_heads(text: Optional[str]) -> Optional[List[Dict[str, int]]]:
    if text is None:
        return None
    rows: List[Dict[str, int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid selected head {item!r}; expected layer:head.")
        layer, head = item.split(":", 1)
        rows.append({"layer": int(layer), "head": int(head)})
    if not rows:
        raise ValueError("--selected-heads produced an empty list.")
    return rows


def resolve_num_attention_heads(model: Any) -> int:
    """Resolve the language decoder attention-head count across LLaVA/Qwen configs."""
    config = getattr(model, "config", None)
    candidates = [
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(config, "llm_config", None),
    ]
    for cfg in candidates:
        if cfg is None:
            continue
        for name in ("num_attention_heads", "num_heads", "n_head"):
            value = getattr(cfg, name, None)
            if value is not None:
                value = int(value)
                if value > 0:
                    return value
    raise RuntimeError(
        "Could not resolve num_attention_heads from the model configuration. "
        "Pass --selected-heads explicitly, for example --selected-heads '12:0,12:1'."
    )


def auto_selected_heads(
    *,
    core: Any,
    model: Any,
    model_name: str,
    n_layers: int,
) -> List[Dict[str, int]]:
    """Use every head at the repository's model-specific reference layer."""
    auto_layers = getattr(core, "AUTO_LAYERS", {})
    layer = int(auto_layers.get(model_name, n_layers // 2))
    layer = max(0, min(layer, n_layers - 1))
    n_heads = resolve_num_attention_heads(model)
    return [{"layer": layer, "head": head} for head in range(n_heads)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")


# ---------------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------------

def build_three_groups(
    *,
    core: Any,
    analysis_rows: Sequence[Dict[str, Any]],
    generation_rows: Sequence[Dict[str, Any]],
    include_group_d: bool,
) -> Dict[int, Dict[str, Any]]:
    analysis_by_sid = {int(row["sid"]): row for row in analysis_rows}
    selected: Dict[int, Dict[str, Any]] = {}

    for generation in generation_rows:
        if "sid" not in generation:
            continue
        sid = int(generation["sid"])
        analysis = analysis_by_sid.get(sid)
        if analysis is None:
            continue

        gt = core.normalize_relation(
            first_present(generation, ("gt", "answer", "ground_truth"))
        )
        if gt not in RELATIONS:
            gt = core.normalize_relation(
                first_present(analysis, ("gt", "answer", "ground_truth"))
            )
        if gt not in RELATIONS:
            continue

        centroid_prediction = core.normalize_relation(
            first_present(
                analysis,
                (
                    "centroid_prediction",
                    "topk_centroid_prediction",
                    "prediction",
                ),
            )
        )
        centroid_correct_default = centroid_prediction == gt
        centroid_correct = as_bool(
            analysis.get("centroid_correct"),
            default=centroid_correct_default,
        )

        baseline_prediction = core.normalize_relation(
            first_present(
                generation,
                (
                    "baseline_prediction",
                    "generation_prediction",
                    "generated_relation",
                    "prediction",
                    "parsed_prediction",
                ),
            )
        )
        if baseline_prediction not in RELATIONS:
            continue
        generation_correct = as_bool(
            first_present(
                generation,
                (
                    "baseline_correct",
                    "generation_correct",
                    "correct",
                ),
            ),
            default=baseline_prediction == gt,
        )

        if centroid_correct and generation_correct:
            group = GROUP_A
        elif centroid_correct and not generation_correct:
            group = GROUP_B
        elif not centroid_correct and generation_correct:
            group = GROUP_C
        else:
            group = GROUP_D
            if not include_group_d:
                continue

        selected[sid] = {
            "sid": sid,
            "group": group,
            "gt": gt,
            "centroid_correct": centroid_correct,
            "generation_correct": generation_correct,
            "centroid_prediction": centroid_prediction,
            "baseline_prediction": baseline_prediction,
            "centroid_confidence": safe_float(
                first_present(
                    analysis,
                    ("centroid_confidence", "topk_confidence", "confidence"),
                )
            ),
            "axis_confidence": safe_float(analysis.get("axis_confidence")),
            "head_agreement": safe_float(analysis.get("head_agreement")),
            "swap_stability": safe_float(analysis.get("swap_stability")),
            "mean_separation": safe_float(analysis.get("mean_separation")),
            "mean_visual_mass": safe_float(analysis.get("mean_visual_mass")),
            "centroid_delta_x": safe_float(
                first_present(analysis, ("delta_x", "centroid_delta_x"))
            ),
            "centroid_delta_y": safe_float(
                first_present(analysis, ("delta_y", "centroid_delta_y"))
            ),
            "prior_lm_margin": safe_float(
                first_present(
                    generation,
                    ("lm_relation_margin", "baseline_margin", "gt_margin"),
                )
            ),
            "prior_lm_top": core.normalize_relation(
                first_present(generation, ("lm_relation_top", "baseline_prediction"))
            ),
        }
    return selected


def cap_groups(
    rows_by_sid: Dict[int, Dict[str, Any]],
    *,
    max_per_group: Optional[int],
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    if max_per_group is None:
        return rows_by_sid
    if max_per_group <= 0:
        raise ValueError("--max-per-group must be positive.")
    rng = random.Random(seed)
    kept: Dict[int, Dict[str, Any]] = {}
    for group in ALL_GROUPS:
        rows = [row for row in rows_by_sid.values() if row["group"] == group]
        rng.shuffle(rows)
        for row in rows[:max_per_group]:
            kept[int(row["sid"])] = row
    return kept


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    denom = max(1, len(a) + len(b) - 2)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / denom)
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_unpaired_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or repetitions <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        out[i] = float(np.mean(aa) - np.mean(bb))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_paired_difference(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Tuple[float, float]:
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2 or repetitions <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        sample = d[rng.integers(0, len(d), len(d))]
        out[i] = float(np.mean(sample))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def pairwise_comparison_row(
    *,
    comparison: str,
    group_left: str,
    group_right: str,
    metric: str,
    values: np.ndarray,
    group_labels: np.ndarray,
    relation_codes: np.ndarray,
    relation: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    if relation == "all":
        relation_mask = np.ones(len(values), dtype=bool)
    else:
        relation_mask = relation_codes == RELATION_TO_INDEX[relation]

    left = np.asarray(
        values[relation_mask & (group_labels == group_left)],
        dtype=np.float64,
    )
    right = np.asarray(
        values[relation_mask & (group_labels == group_right)],
        dtype=np.float64,
    )
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]

    lo, hi = bootstrap_unpaired_difference(
        left,
        right,
        repetitions=bootstrap_samples,
        seed=seed,
    )
    mean_left = float(np.mean(left)) if len(left) else float("nan")
    mean_right = float(np.mean(right)) if len(right) else float("nan")

    return {
        "comparison": comparison,
        "group_left": group_left,
        "group_right": group_right,
        "metric": metric,
        "relation": relation,
        "n_left": int(len(left)),
        "n_right": int(len(right)),
        "mean_left": mean_left,
        "mean_right": mean_right,
        "difference_left_minus_right": (
            mean_left - mean_right
            if np.isfinite(mean_left) and np.isfinite(mean_right)
            else float("nan")
        ),
        "cohen_d_left_minus_right": cohen_d(left, right),
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "ci_excludes_zero": bool(
            np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)
        ),
    }


def paired_comparison_row(
    *,
    comparison: str,
    metric: str,
    values: np.ndarray,
    pairs: Sequence[Dict[str, Any]],
    sid_to_row: Mapping[int, int],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    left_values = []
    right_values = []
    for pair in pairs:
        li = sid_to_row.get(int(pair["left_sid"]))
        ri = sid_to_row.get(int(pair["right_sid"]))
        if li is None or ri is None:
            continue
        lv = float(values[li])
        rv = float(values[ri])
        if np.isfinite(lv) and np.isfinite(rv):
            left_values.append(lv)
            right_values.append(rv)

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    diff = left - right
    lo, hi = bootstrap_paired_difference(
        diff,
        repetitions=bootstrap_samples,
        seed=seed,
    )
    return {
        "comparison": comparison,
        "metric": metric,
        "n_pairs": int(len(diff)),
        "mean_left": finite_mean(left),
        "mean_right": finite_mean(right),
        "mean_paired_difference": finite_mean(diff),
        "paired_difference_std": finite_std(diff),
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "ci_excludes_zero": bool(
            np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)
        ),
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def robust_feature_matrix(
    rows: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(
        [
            [safe_float(rows[i].get(name)) for name in feature_names]
            for i in indices
        ],
        dtype=np.float64,
    )
    if matrix.size == 0:
        return matrix

    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        valid = values[np.isfinite(values)]
        median = float(np.median(valid)) if len(valid) else 0.0
        values[~np.isfinite(values)] = median

        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float(q75 - q25)
        if scale <= 1e-8:
            scale = float(np.std(values))
        if scale <= 1e-8:
            scale = 1.0
        matrix[:, col] = (values - median) / scale
    return matrix


def build_nearest_matches(
    *,
    rows: Sequence[Dict[str, Any]],
    group_left: str,
    group_right: str,
    feature_names: Sequence[str],
    reuse_right: bool,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    used_right: set[int] = set()

    for relation in RELATIONS:
        left_indices = [
            i for i, row in enumerate(rows)
            if row["group"] == group_left and row["gt"] == relation
        ]
        right_indices = [
            i for i, row in enumerate(rows)
            if row["group"] == group_right and row["gt"] == relation
        ]
        if not left_indices or not right_indices:
            continue

        all_indices = left_indices + right_indices
        matrix = robust_feature_matrix(rows, all_indices, feature_names)
        n_left = len(left_indices)
        left_matrix = matrix[:n_left]
        right_matrix = matrix[n_left:]

        # Start with difficult-to-match left samples to reduce greedy-order bias.
        candidate_min_dist = []
        for li in range(len(left_indices)):
            distances = np.linalg.norm(right_matrix - left_matrix[li], axis=1)
            candidate_min_dist.append(float(np.min(distances)))
        left_order = np.argsort(candidate_min_dist)[::-1]

        for local_left in left_order:
            distances = np.linalg.norm(
                right_matrix - left_matrix[local_left],
                axis=1,
            )
            order = np.argsort(distances)
            chosen_local_right = None
            for local_right in order:
                global_right_index = right_indices[int(local_right)]
                if reuse_right or global_right_index not in used_right:
                    chosen_local_right = int(local_right)
                    break
            if chosen_local_right is None:
                continue

            global_left_index = left_indices[int(local_left)]
            global_right_index = right_indices[chosen_local_right]
            if not reuse_right:
                used_right.add(global_right_index)

            pairs.append(
                {
                    "relation": relation,
                    "left_sid": int(rows[global_left_index]["sid"]),
                    "right_sid": int(rows[global_right_index]["sid"]),
                    "left_group": group_left,
                    "right_group": group_right,
                    "distance": float(distances[chosen_local_right]),
                    "features": ",".join(feature_names),
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# Layer/head comparison builders
# ---------------------------------------------------------------------------

def append_pairwise_layer_rows(
    *,
    output: List[Dict[str, Any]],
    metric: str,
    matrix: np.ndarray,
    group_labels: np.ndarray,
    relation_codes: np.ndarray,
    bootstrap_samples: int,
    seed_offset: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    n_layers = int(matrix.shape[1])
    for comparison_index, (name, left, right) in enumerate(PAIR_SPECS):
        for layer in range(n_layers):
            for relation in ("all", *RELATIONS):
                row = pairwise_comparison_row(
                    comparison=name,
                    group_left=left,
                    group_right=right,
                    metric=metric,
                    values=matrix[:, layer],
                    group_labels=group_labels,
                    relation_codes=relation_codes,
                    relation=relation,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed_offset + comparison_index * 100000 + layer * 10 + len(output),
                )
                row["layer"] = layer
                if extra:
                    row.update(extra)
                output.append(row)


def append_pairwise_head_rows(
    *,
    output: List[Dict[str, Any]],
    metric: str,
    tensor: np.ndarray,
    group_labels: np.ndarray,
    relation_codes: np.ndarray,
    bootstrap_samples: int,
    seed_offset: int,
) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{metric} must have shape [N,L,H], got {tensor.shape}")
    n_layers, n_heads = int(tensor.shape[1]), int(tensor.shape[2])
    for comparison_index, (name, left, right) in enumerate(PAIR_SPECS):
        for layer in range(n_layers):
            for head in range(n_heads):
                row = pairwise_comparison_row(
                    comparison=name,
                    group_left=left,
                    group_right=right,
                    metric=metric,
                    values=tensor[:, layer, head],
                    group_labels=group_labels,
                    relation_codes=relation_codes,
                    relation="all",
                    bootstrap_samples=bootstrap_samples,
                    seed=(
                        seed_offset
                        + comparison_index * 1000000
                        + layer * 1000
                        + head
                    ),
                )
                row.update({"layer": layer, "head": head})
                output.append(row)


def earliest_persistent_divergence(
    rows: Sequence[Dict[str, Any]],
    *,
    comparison: str,
    metric: str,
    minimum_effect: float,
    consecutive: int,
    extra_filters: Optional[Mapping[str, Any]] = None,
) -> Optional[int]:
    candidates = []
    for row in rows:
        if row.get("comparison") != comparison:
            continue
        if row.get("metric") != metric:
            continue
        if row.get("relation") != "all":
            continue
        if extra_filters and any(row.get(k) != v for k, v in extra_filters.items()):
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: int(row["layer"]))

    for start in range(max(0, len(candidates) - consecutive + 1)):
        window = candidates[start : start + consecutive]
        if len(window) < consecutive:
            break
        signs = []
        valid = True
        for row in window:
            d = safe_float(row.get("cohen_d_left_minus_right"))
            lo = safe_float(row.get("bootstrap_ci_low"))
            hi = safe_float(row.get("bootstrap_ci_high"))
            if (
                not np.isfinite(d)
                or abs(d) < minimum_effect
                or not np.isfinite(lo)
                or not np.isfinite(hi)
                or lo <= 0.0 <= hi
            ):
                valid = False
                break
            signs.append(np.sign(d))
        if valid and len(set(signs)) == 1:
            return int(window[0]["layer"])
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_group_trajectory_plot(
    *,
    path: Path,
    matrix: np.ndarray,
    group_labels: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    layers = np.arange(matrix.shape[1])
    for group in PRIMARY_GROUPS:
        mask = group_labels == group
        if not np.any(mask):
            continue
        values = matrix[mask]
        mean = np.nanmean(values, axis=0)
        valid_count = np.sum(np.isfinite(values), axis=0)
        std = np.nanstd(values, axis=0)
        sem = std / np.sqrt(np.maximum(valid_count, 1))
        ax.plot(layers, mean, label=group)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.15)

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    core = import_core(args.core_module)
    prior_dir = Path(args.prior_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Pass --overwrite to replace generated files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = (
        Path(args.centroid_analysis_jsonl)
        if args.centroid_analysis_jsonl
        else prior_dir / "centroid_analysis.jsonl"
    )
    generation_path = (
        Path(args.generation_jsonl)
        if args.generation_jsonl
        else prior_dir / "generation.jsonl"
    )
    if not analysis_path.exists():
        raise FileNotFoundError(analysis_path)
    if not generation_path.exists():
        raise FileNotFoundError(generation_path)

    analysis_rows = core.read_jsonl(analysis_path)
    generation_rows = core.read_jsonl(generation_path)
    prior_rows = build_three_groups(
        core=core,
        analysis_rows=analysis_rows,
        generation_rows=generation_rows,
        include_group_d=args.include_group_d,
    )
    prior_rows = cap_groups(
        prior_rows,
        max_per_group=args.max_per_group,
        seed=args.seed,
    )
    if not prior_rows:
        raise RuntimeError("No samples remained after building groups.")

    selected_heads = parse_selected_heads(args.selected_heads)
    selected_layers: List[int] = []

    group_counts = Counter(row["group"] for row in prior_rows.values())
    print("=" * 110)
    print("THREE-GROUP CENTROID / GENERATION TRANSFER TRACE")
    print("=" * 110)
    for group in ALL_GROUPS:
        if group in group_counts or group in PRIMARY_GROUPS:
            print(f"{group:46s}: {group_counts[group]}")
    # Selected heads are resolved after the model is loaded. This makes
    # prior-dir/config.json optional.

    # Dataset and prompt loading.
    backend_module = core.import_two_object_module()
    records, audit = backend_module.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {int(record.sid): record for record in records}

    # Reuse the core resolver by supplying an argparse-like namespace.
    prompt_path = core.resolve_prompt_path(args)
    prompt_rows = core.load_standard_prompts(prompt_path)

    missing = [
        sid
        for sid in prior_rows
        if sid not in record_by_sid or sid not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} selected records/prompts; first={missing[:10]}"
        )

    if args.model not in backend_module.SPECS:
        raise ValueError(f"Unknown model: {args.model}")
    spec = backend_module.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": core.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }
    print(f"\nLoading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = core.resolve_decoder_layers(model)
    n_layers = len(layers)

    if selected_heads is None:
        config_path = prior_dir / "config.json"
        if config_path.exists():
            try:
                selected_heads = core.load_selected_heads(prior_dir)
                print(f"Selected heads loaded from: {config_path}")
            except Exception as exc:
                print(
                    "[WARN] Failed to load selected heads from config.json; "
                    f"using automatic reference-layer heads instead: {exc}"
                )
                selected_heads = auto_selected_heads(
                    core=core,
                    model=model,
                    model_name=args.model,
                    n_layers=n_layers,
                )
        else:
            print(
                f"[WARN] Missing {config_path}; using all heads at the "
                "model-specific reference layer."
            )
            selected_heads = auto_selected_heads(
                core=core,
                model=model,
                model_name=args.model,
                n_layers=n_layers,
            )

    selected_layers = sorted({int(row["layer"]) for row in selected_heads})
    if selected_layers and max(selected_layers) >= n_layers:
        raise RuntimeError(
            f"Selected layer {max(selected_layers)} outside model with {n_layers} layers"
        )

    print(
        "Selected heads:",
        ", ".join(
            f"L{int(row['layer']):02d}H{int(row['head']):02d}"
            for row in selected_heads
        ),
    )

    final_norm, final_norm_path = core.resolve_final_norm(model)
    label_token_ids = core.label_token_id_variants(processor.tokenizer)
    token_weight, token_bias, relation_positions = core.relation_token_rows(
        model,
        label_token_ids,
    )
    collector = core.LayerTraceCollector(layers, selected_layers)
    projection_diagnostics = collector.projection_diagnostics()
    (output_dir / "projection_diagnostics.json").write_text(
        json.dumps(projection_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Decoder layers: {layers_path} ({n_layers})")
    print(f"Final norm: {final_norm_path}")

    metadata_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, List[np.ndarray]] = defaultdict(list)
    metadata_path = output_dir / "sample_metadata.jsonl"
    errors_path = output_dir / "errors.jsonl"
    for path in (metadata_path, errors_path):
        if path.exists():
            path.unlink()

    started = time.time()
    completed = 0

    try:
        for sid in tqdm(sorted(prior_rows), desc=f"three-group-trace:{args.model}"):
            batch = None
            image = None
            try:
                prior = prior_rows[sid]
                record = record_by_sid[sid]
                prompt_row = prompt_rows[sid]

                image = core.record_image(record)
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = core.normalize_relation(prompt_row["answer_raw"])
                if gt != prior["gt"]:
                    raise RuntimeError(
                        f"GT mismatch sid={sid}: prompt={gt}, prior={prior['gt']}"
                    )

                batch = core.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                trace_meta, trace_arrays = core.trace_one_prompt(
                    model=model,
                    processor=processor,
                    collector=collector,
                    batch=batch,
                    subject=subject,
                    reference=reference,
                    gt=gt,
                    selected_heads=selected_heads,
                    final_norm=final_norm,
                    token_weight=token_weight,
                    token_bias=token_bias,
                    relation_positions=relation_positions,
                )

                metadata = {
                    **prior,
                    **trace_meta,
                    "subject": subject,
                    "reference": reference,
                    "question": question,
                    "row_index": len(metadata_rows),
                }
                metadata_rows.append(metadata)
                core.append_jsonl(metadata_path, metadata)

                for key, value in trace_arrays.items():
                    arrays[key].append(value)

                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    tqdm.write(
                        f"[{completed}/{len(prior_rows)}] sid={sid} "
                        f"{prior['group']} GT={gt} "
                        f"base={prior['baseline_prediction']} "
                        f"centroid={prior['centroid_prediction']} "
                        f"peak=L{trace_meta['peak_gt_margin_layer_after_layer']} "
                        f"final={trace_meta['final_gt_margin_after_layer']:+.3f}"
                    )
            except Exception as exc:
                core.append_jsonl(
                    errors_path,
                    {
                        "sid": sid,
                        "group": prior_rows.get(sid, {}).get("group"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-20:],
                    },
                )
                tqdm.write(f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}")
            finally:
                if batch is not None:
                    del batch
                if image is not None:
                    del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not metadata_rows:
            raise RuntimeError("No samples were traced successfully.")

        stacked = {
            key: np.stack(values, axis=0)
            for key, values in arrays.items()
        }
        sids = np.asarray(
            [int(row["sid"]) for row in metadata_rows],
            dtype=np.int64,
        )
        group_labels = np.asarray(
            [str(row["group"]) for row in metadata_rows],
            dtype="U64",
        )
        group_codes = np.asarray(
            [GROUP_TO_CODE[str(row["group"])] for row in metadata_rows],
            dtype=np.int8,
        )
        relation_codes = np.asarray(
            [RELATION_TO_INDEX[str(row["gt"])] for row in metadata_rows],
            dtype=np.int8,
        )
        stacked.update(
            {
                "sids": sids,
                "group_codes": group_codes,
                "group_labels": group_labels,
                "relation_codes": relation_codes,
                "layer_indices": np.arange(n_layers, dtype=np.int16),
                "selected_spatial_layers": np.asarray(
                    [int(row["layer"]) for row in selected_heads],
                    dtype=np.int16,
                ),
                "selected_spatial_heads": np.asarray(
                    [int(row["head"]) for row in selected_heads],
                    dtype=np.int16,
                ),
            }
        )
        np.savez_compressed(output_dir / "trace_arrays.npz", **stacked)

        # Derived path metrics. No subject-reference hidden-state subtraction.
        last_to_object_mass = (
            stacked["last_to_subject_mass"]
            + stacked["last_to_reference_mass"]
        )
        route_denominator = (
            last_to_object_mass
            + stacked["last_to_visual_mass"]
            + 1e-8
        )
        object_route_share = last_to_object_mass / route_denominator

        layer_metrics = {
            "last_to_subject_mass_head_mean": np.nanmean(
                stacked["last_to_subject_mass"], axis=-1
            ),
            "last_to_reference_mass_head_mean": np.nanmean(
                stacked["last_to_reference_mass"], axis=-1
            ),
            "last_to_object_mass_head_mean": np.nanmean(
                last_to_object_mass, axis=-1
            ),
            "last_to_visual_mass_head_mean": np.nanmean(
                stacked["last_to_visual_mass"], axis=-1
            ),
            "object_route_share_head_mean": np.nanmean(
                object_route_share, axis=-1
            ),
            "routing_balance_head_mean": np.nanmean(
                stacked["routing_balance"], axis=-1
            ),
            "routing_av_raw_norm_head_mean": np.nanmean(
                stacked["routing_av_norm"], axis=-1
            ),
            "routing_av_projected_norm_head_mean": np.nanmean(
                stacked["routing_av_projected_norm"], axis=-1
            ),
        }

        # 1) Scalar pairwise comparison.
        scalar_names = [
            "centroid_confidence",
            "axis_confidence",
            "head_agreement",
            "swap_stability",
            "mean_separation",
            "mean_visual_mass",
            "prompt_length",
            "subject_token_count",
            "reference_token_count",
            "n_visual_tokens",
            "prior_lm_margin",
            "peak_gt_margin_after_layer",
            "final_gt_margin_after_layer",
            "late_margin_drop",
            "late_last_to_object_mass_mean",
            "late_routing_av_raw_norm_mean",
            "late_routing_av_projected_norm_mean",
            "selected_spatial_visual_mass_mean",
            "selected_spatial_av_raw_norm_mean",
            "selected_spatial_av_projected_norm_mean",
        ]
        scalar_rows: List[Dict[str, Any]] = []
        for metric_index, metric in enumerate(scalar_names):
            values = np.asarray(
                [safe_float(row.get(metric)) for row in metadata_rows],
                dtype=np.float64,
            )
            for comparison_index, (name, left, right) in enumerate(PAIR_SPECS):
                for relation in ("all", *RELATIONS):
                    scalar_rows.append(
                        pairwise_comparison_row(
                            comparison=name,
                            group_left=left,
                            group_right=right,
                            metric=metric,
                            values=values,
                            group_labels=group_labels,
                            relation_codes=relation_codes,
                            relation=relation,
                            bootstrap_samples=args.bootstrap_samples,
                            seed=(
                                args.seed
                                + metric_index * 10000
                                + comparison_index * 1000
                                + len(scalar_rows)
                            ),
                        )
                    )
        write_csv(output_dir / "scalar_pairwise.csv", scalar_rows)

        # 2) Layerwise logit lens.
        logit_rows: List[Dict[str, Any]] = []
        for stage in ("input", "after_attention", "after_layer"):
            append_pairwise_layer_rows(
                output=logit_rows,
                metric="gt_relation_margin",
                matrix=stacked[f"gt_margin_{stage}"],
                group_labels=group_labels,
                relation_codes=relation_codes,
                bootstrap_samples=args.bootstrap_samples,
                seed_offset=args.seed + 100000,
                extra={"stage": stage},
            )
            append_pairwise_layer_rows(
                output=logit_rows,
                metric="layer0_normalized_gt_margin",
                matrix=stacked[f"gt_margin_layer0_normalized_{stage}"],
                group_labels=group_labels,
                relation_codes=relation_codes,
                bootstrap_samples=args.bootstrap_samples,
                seed_offset=args.seed + 200000,
                extra={"stage": stage},
            )
        write_csv(output_dir / "layerwise_logit_lens_pairwise.csv", logit_rows)

        # 3) Attention/MLP/block contribution gains.
        module_rows: List[Dict[str, Any]] = []
        for component, key in (
            ("attention", "gt_margin_gain_attention"),
            ("mlp", "gt_margin_gain_mlp"),
            ("block", "gt_margin_gain_block"),
        ):
            append_pairwise_layer_rows(
                output=module_rows,
                metric="gt_margin_gain",
                matrix=stacked[key],
                group_labels=group_labels,
                relation_codes=relation_codes,
                bootstrap_samples=args.bootstrap_samples,
                seed_offset=args.seed + 300000,
                extra={"component": component},
            )
        write_csv(output_dir / "layerwise_module_gains_pairwise.csv", module_rows)

        # 4) Layerwise transfer/routing metrics.
        routing_rows: List[Dict[str, Any]] = []
        for metric, matrix in layer_metrics.items():
            append_pairwise_layer_rows(
                output=routing_rows,
                metric=metric,
                matrix=matrix,
                group_labels=group_labels,
                relation_codes=relation_codes,
                bootstrap_samples=args.bootstrap_samples,
                seed_offset=args.seed + 400000,
            )
        write_csv(output_dir / "layerwise_routing_pairwise.csv", routing_rows)

        # 5) Per-head transfer/routing metrics.
        head_rows: List[Dict[str, Any]] = []
        head_tensors = {
            "last_to_subject_mass": stacked["last_to_subject_mass"],
            "last_to_reference_mass": stacked["last_to_reference_mass"],
            "last_to_object_mass": last_to_object_mass,
            "last_to_visual_mass": stacked["last_to_visual_mass"],
            "object_route_share": object_route_share,
            "routing_balance": stacked["routing_balance"],
            "routing_av_raw_norm": stacked["routing_av_norm"],
            "routing_av_projected_norm": stacked["routing_av_projected_norm"],
        }
        for metric, tensor in head_tensors.items():
            append_pairwise_head_rows(
                output=head_rows,
                metric=metric,
                tensor=tensor,
                group_labels=group_labels,
                relation_codes=relation_codes,
                bootstrap_samples=args.bootstrap_samples,
                seed_offset=args.seed + 500000,
            )
        write_csv(output_dir / "headwise_routing_pairwise.csv", head_rows)

        top_head_rows = sorted(
            [
                row for row in head_rows
                if row["ci_excludes_zero"]
                and min(int(row["n_left"]), int(row["n_right"])) >= 5
                and np.isfinite(safe_float(row["cohen_d_left_minus_right"]))
            ],
            key=lambda row: abs(float(row["cohen_d_left_minus_right"])),
            reverse=True,
        )[:300]
        write_csv(output_dir / "top_head_candidates.csv", top_head_rows)

        # 6) Selected spatial-head visual/object tracing.
        selected_rows: List[Dict[str, Any]] = []
        selected_metrics = {
            "subject_visual_mass": stacked["spatial_subject_visual_mass"],
            "reference_visual_mass": stacked["spatial_reference_visual_mass"],
            "subject_av_raw_norm": stacked["spatial_subject_av_norm"],
            "reference_av_raw_norm": stacked["spatial_reference_av_norm"],
            "subject_av_projected_norm": stacked[
                "spatial_subject_av_projected_norm"
            ],
            "reference_av_projected_norm": stacked[
                "spatial_reference_av_projected_norm"
            ],
        }
        for metric_index, (metric, matrix) in enumerate(selected_metrics.items()):
            for selected_index, head_spec in enumerate(selected_heads):
                for comparison_index, (name, left, right) in enumerate(PAIR_SPECS):
                    row = pairwise_comparison_row(
                        comparison=name,
                        group_left=left,
                        group_right=right,
                        metric=metric,
                        values=matrix[:, selected_index],
                        group_labels=group_labels,
                        relation_codes=relation_codes,
                        relation="all",
                        bootstrap_samples=args.bootstrap_samples,
                        seed=(
                            args.seed
                            + 600000
                            + metric_index * 10000
                            + selected_index * 100
                            + comparison_index
                        ),
                    )
                    row.update(
                        {
                            "selected_index": selected_index,
                            "layer": int(head_spec["layer"]),
                            "head": int(head_spec["head"]),
                        }
                    )
                    selected_rows.append(row)
        write_csv(
            output_dir / "selected_spatial_heads_pairwise.csv",
            selected_rows,
        )

        # 7) Nuisance-matched comparisons.
        match_features_ab = parse_feature_list(args.match_features_ab)
        match_features_other = parse_feature_list(args.match_features_other)
        matched_pairs_all: List[Dict[str, Any]] = []
        matched_by_comparison: Dict[str, List[Dict[str, Any]]] = {}

        for name, left, right in PAIR_SPECS:
            features = (
                match_features_ab if name == "A_vs_B" else match_features_other
            )
            pairs = build_nearest_matches(
                rows=metadata_rows,
                group_left=left,
                group_right=right,
                feature_names=features,
                reuse_right=args.match_reuse,
            )
            for pair in pairs:
                pair["comparison"] = name
            matched_by_comparison[name] = pairs
            matched_pairs_all.extend(pairs)
        write_csv(output_dir / "matched_pairs.csv", matched_pairs_all)

        sid_to_row = {
            int(row["sid"]): i for i, row in enumerate(metadata_rows)
        }
        matched_rows: List[Dict[str, Any]] = []

        matched_layer_sources: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        for stage in ("input", "after_attention", "after_layer"):
            matched_layer_sources.append(
                (
                    "gt_relation_margin",
                    stacked[f"gt_margin_{stage}"],
                    {"stage": stage},
                )
            )
        for component, key in (
            ("attention", "gt_margin_gain_attention"),
            ("mlp", "gt_margin_gain_mlp"),
            ("block", "gt_margin_gain_block"),
        ):
            matched_layer_sources.append(
                ("gt_margin_gain", stacked[key], {"component": component})
            )
        for metric, matrix in layer_metrics.items():
            matched_layer_sources.append((metric, matrix, {}))

        for source_index, (metric, matrix, extra) in enumerate(
            matched_layer_sources
        ):
            for name, _, _ in PAIR_SPECS:
                pairs = matched_by_comparison[name]
                for layer in range(n_layers):
                    row = paired_comparison_row(
                        comparison=name,
                        metric=metric,
                        values=matrix[:, layer],
                        pairs=pairs,
                        sid_to_row=sid_to_row,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=(
                            args.seed
                            + 700000
                            + source_index * 10000
                            + layer * 10
                        ),
                    )
                    row["layer"] = layer
                    row.update(extra)
                    matched_rows.append(row)
        write_csv(output_dir / "matched_layerwise_pairwise.csv", matched_rows)

        # 8) Candidate breakpoint summary.
        breakpoint_rows: List[Dict[str, Any]] = []
        breakpoint_specs = [
            ("logit_after_layer", logit_rows, "gt_relation_margin", {"stage": "after_layer"}),
            ("attention_gain", module_rows, "gt_margin_gain", {"component": "attention"}),
            ("mlp_gain", module_rows, "gt_margin_gain", {"component": "mlp"}),
            ("block_gain", module_rows, "gt_margin_gain", {"component": "block"}),
            (
                "last_to_object_mass",
                routing_rows,
                "last_to_object_mass_head_mean",
                {},
            ),
            (
                "last_to_visual_mass",
                routing_rows,
                "last_to_visual_mass_head_mean",
                {},
            ),
            (
                "object_route_share",
                routing_rows,
                "object_route_share_head_mean",
                {},
            ),
            (
                "projected_routing_norm",
                routing_rows,
                "routing_av_projected_norm_head_mean",
                {},
            ),
        ]
        for comparison, _, _ in PAIR_SPECS:
            for name, rows, metric, filters in breakpoint_specs:
                layer = earliest_persistent_divergence(
                    rows,
                    comparison=comparison,
                    metric=metric,
                    minimum_effect=args.minimum_effect,
                    consecutive=args.persistent_layers,
                    extra_filters=filters,
                )
                breakpoint_rows.append(
                    {
                        "comparison": comparison,
                        "signal": name,
                        "earliest_persistent_layer": layer,
                        "minimum_abs_cohen_d": args.minimum_effect,
                        "required_consecutive_layers": args.persistent_layers,
                    }
                )
        write_csv(output_dir / "candidate_breakpoints.csv", breakpoint_rows)

        # 9) Compact mechanism-oriented summary.
        late_start = max(0, n_layers - max(4, n_layers // 4))

        def late_group_mean(matrix: np.ndarray, group: str) -> float:
            mask = group_labels == group
            if not np.any(mask):
                return float("nan")
            return finite_mean(matrix[mask, late_start:])

        mechanism_summary: Dict[str, Any] = {}
        for name, left, right in PAIR_SPECS:
            mechanism_summary[name] = {
                "late_layers_start": late_start,
                "left_group": left,
                "right_group": right,
                "late_mean_gt_margin_left": late_group_mean(
                    stacked["gt_margin_after_layer"], left
                ),
                "late_mean_gt_margin_right": late_group_mean(
                    stacked["gt_margin_after_layer"], right
                ),
                "late_mean_object_route_share_left": late_group_mean(
                    layer_metrics["object_route_share_head_mean"], left
                ),
                "late_mean_object_route_share_right": late_group_mean(
                    layer_metrics["object_route_share_head_mean"], right
                ),
                "late_mean_direct_visual_mass_left": late_group_mean(
                    layer_metrics["last_to_visual_mass_head_mean"], left
                ),
                "late_mean_direct_visual_mass_right": late_group_mean(
                    layer_metrics["last_to_visual_mass_head_mean"], right
                ),
                "late_mean_projected_object_routing_left": late_group_mean(
                    layer_metrics["routing_av_projected_norm_head_mean"], left
                ),
                "late_mean_projected_object_routing_right": late_group_mean(
                    layer_metrics["routing_av_projected_norm_head_mean"], right
                ),
                "late_mean_attention_gain_left": late_group_mean(
                    stacked["gt_margin_gain_attention"], left
                ),
                "late_mean_attention_gain_right": late_group_mean(
                    stacked["gt_margin_gain_attention"], right
                ),
                "late_mean_mlp_gain_left": late_group_mean(
                    stacked["gt_margin_gain_mlp"], left
                ),
                "late_mean_mlp_gain_right": late_group_mean(
                    stacked["gt_margin_gain_mlp"], right
                ),
                "matched_pair_count": len(matched_by_comparison[name]),
            }

        summary = {
            "script_version": SCRIPT_VERSION,
            "core_module": args.core_module,
            "dataset": args.dataset,
            "model": args.model,
            "prior_dir": str(prior_dir),
            "analysis_path": str(analysis_path),
            "generation_path": str(generation_path),
            "output_dir": str(output_dir),
            "group_counts_before_trace": dict(group_counts),
            "group_counts_after_trace": dict(Counter(group_labels.tolist())),
            "n_layers": n_layers,
            "n_heads": int(stacked["last_to_subject_mass"].shape[-1]),
            "selected_heads": selected_heads,
            "completed_samples": completed,
            "elapsed_seconds": time.time() - started,
            "breakpoints": breakpoint_rows,
            "mechanism_summary": mechanism_summary,
            "interpretation_guide": {
                "A_vs_B": (
                    "Both groups have centroid-correct localization. "
                    "Lower object routing/projected A*V in B suggests transfer failure. "
                    "Similar routing but more negative late attention/MLP gains in B "
                    "suggests late overwrite."
                ),
                "A_vs_C": (
                    "Both groups generate correctly. If C has weaker object-token "
                    "routing but stronger direct visual routing, C may use an alternate "
                    "visual-to-last path. Similar internal routing with wrong centroid "
                    "suggests centroid is an incomplete readout rather than absent spatial information."
                ),
                "B_vs_C": (
                    "Contrasts failed generation despite centroid-correct localization "
                    "against successful generation despite centroid-wrong readout. "
                    "This is useful for identifying alternate paths, but language shortcuts "
                    "still require flip/text-only controls before making a causal claim."
                ),
            },
            "audit": audit,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.make_plots:
            make_group_trajectory_plot(
                path=output_dir / "plots" / "gt_margin_after_layer.png",
                matrix=stacked["gt_margin_after_layer"],
                group_labels=group_labels,
                title="GT-vs-opposite margin after each decoder block",
                ylabel="GT relation margin",
            )
            make_group_trajectory_plot(
                path=output_dir / "plots" / "object_route_share.png",
                matrix=layer_metrics["object_route_share_head_mean"],
                group_labels=group_labels,
                title="Last-token route share to subject/reference tokens",
                ylabel="Object route share",
            )
            make_group_trajectory_plot(
                path=output_dir / "plots" / "projected_object_routing_norm.png",
                matrix=layer_metrics[
                    "routing_av_projected_norm_head_mean"
                ],
                group_labels=group_labels,
                title="Projected object-to-last routed contribution",
                ylabel="Projected A*V norm",
            )
            make_group_trajectory_plot(
                path=output_dir / "plots" / "attention_margin_gain.png",
                matrix=stacked["gt_margin_gain_attention"],
                group_labels=group_labels,
                title="Attention contribution to GT relation margin",
                ylabel="Attention margin gain",
            )
            make_group_trajectory_plot(
                path=output_dir / "plots" / "mlp_margin_gain.png",
                matrix=stacked["gt_margin_gain_mlp"],
                group_labels=group_labels,
                title="MLP contribution to GT relation margin",
                ylabel="MLP margin gain",
            )

        print("\n" + "=" * 110)
        print("DONE")
        print("=" * 110)
        print("Group counts after successful trace:")
        for group, count in Counter(group_labels.tolist()).items():
            print(f"  {group:46s}: {count}")
        print("\nMain outputs:")
        for filename in (
            "sample_metadata.jsonl",
            "trace_arrays.npz",
            "scalar_pairwise.csv",
            "layerwise_logit_lens_pairwise.csv",
            "layerwise_module_gains_pairwise.csv",
            "layerwise_routing_pairwise.csv",
            "headwise_routing_pairwise.csv",
            "top_head_candidates.csv",
            "selected_spatial_heads_pairwise.csv",
            "matched_pairs.csv",
            "matched_layerwise_pairwise.csv",
            "candidate_breakpoints.csv",
            "summary.json",
        ):
            print(f"  {output_dir / filename}")
        if errors_path.exists():
            print(f"  {errors_path}")

    finally:
        collector.close()
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
