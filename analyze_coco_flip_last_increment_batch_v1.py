#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch analysis of flip-sensitive object-to-last information transfer.

Runs every eligible COCO left/right pair with one model load and separates the
results into four baseline categories:

    both_correct
    original_only_correct
    flipped_only_correct
    both_wrong

For each original/horizontal-flip pair, the script measures:

1. Subject/reference/prompt-last original-minus-flip state trajectories.
2. Exact prompt-last block decomposition:

       Delta output = Delta input + Delta attention + Delta MLP

3. Source-group and head-level attention contributions into prompt_last:

       Delta C_{S->last}^{l,h}
       = C_original - C_flipped

4. Routing/content decomposition of each attention contribution.
5. Alignment of each path with:
   - the same-layer prompt-last attention difference;
   - the same-layer NEW prompt-last increment;
   - the final-layer prompt-last difference.

It then aggregates every metric separately for the four correctness categories
and reports which object-to-last paths are weaker, more negative, or absent in
error categories relative to both-correct pairs.

This is a comparative mechanism diagnostic. It does not by itself prove that a
missing path causes the error; top differences should be validated with exact
edge replacement or restoration.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


VERSION = "coco-flip-last-increment-batch-v1"

RELATIONS = ("left", "right")
OPPOSITE = {
    "left": "right",
    "right": "left",
}
PAIR_STATUSES = (
    "both_correct",
    "original_only_correct",
    "flipped_only_correct",
    "both_wrong",
)
AGGREGATE_STATUSES = ("all",) + PAIR_STATUSES

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

STATE_METRICS = (
    "input_delta_norm",
    "attention_delta_norm",
    "mlp_delta_norm",
    "increment_delta_norm",
    "output_delta_norm",
    "output_delta_to_final_last_projection_fraction",
    "attention_fraction_of_increment",
    "mlp_fraction_of_increment",
)

SOURCE_METRICS = (
    "delta_norm",
    "routing_norm",
    "content_norm",
    "routing_share_by_norm",
    "delta_to_last_attention_projection_fraction",
    "delta_to_last_increment_projection_fraction",
    "content_to_last_increment_projection_fraction",
    "routing_to_last_increment_projection_fraction",
    "delta_to_final_last_projection_fraction",
    "content_to_final_last_projection_fraction",
)

HEAD_METRICS = (
    "delta_norm",
    "routing_norm",
    "content_norm",
    "routing_share_by_norm",
    "delta_to_last_increment_projection_fraction",
    "content_to_last_increment_projection_fraction",
    "delta_to_final_last_projection_fraction",
    "content_to_final_last_projection_fraction",
)

DIAGNOSTIC_METRICS = (
    "subject_peak_input_delta_norm",
    "reference_peak_input_delta_norm",
    "object_state_peak_norm",
    "prompt_last_peak_increment_delta_norm",
    "prompt_last_peak_output_delta_norm",
    "final_last_delta_norm",
    "subject_content_transfer_peak",
    "reference_content_transfer_peak",
    "object_content_transfer_peak",
    "object_content_transfer_mean",
    "object_content_negative_rate",
    "visual_transfer_peak",
    "options_transfer_peak",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-script",
        default="analyze_coco_flip_last_increment_trace_v1.py",
        help="The single-pair trace script generated previously.",
    )
    parser.add_argument(
        "--edge-script",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    parser.add_argument(
        "--state-layers",
        default="all",
        help="Layers for subject/reference/prompt-last state trajectories.",
    )
    parser.add_argument(
        "--edge-layers",
        default="19,20,21,22,23,24,25,26,27,28,29,30",
        help="Layers for source/head -> prompt-last decomposition.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCE_GROUPS),
        help=f"Comma-separated source groups; allowed={SOURCE_GROUPS}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit before relation filtering.",
    )
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument(
        "--save-head-details",
        action="store_true",
        help="Also save every per-sample head row. This can be large.",
    )
    parser.add_argument(
        "--top-k-report",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--replay-tolerance",
        type=float,
        default=5e-3,
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


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


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


class StreamingCSV:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.writer = None
        self.fields: Optional[List[str]] = None

    def write(self, row: Mapping[str, Any]) -> None:
        if self.file is None:
            self.fields = list(row.keys())
            self.file = self.path.open(
                "w",
                encoding="utf-8",
                newline="",
            )
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=self.fields,
            )
            self.writer.writeheader()
        assert self.writer is not None
        assert self.fields is not None
        extra = set(row.keys()) - set(self.fields)
        if extra:
            raise RuntimeError(
                f"{self.path.name}: new fields appeared after header: {sorted(extra)}"
            )
        self.writer.writerow({
            field: row.get(field)
            for field in self.fields
        })

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None


@dataclass
class RunningMetric:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    positive: int = 0
    negative: int = 0
    zero: int = 0

    def update(self, value: Any) -> None:
        if not finite(value):
            return
        number = float(value)
        self.n += 1
        self.total += number
        self.total_sq += number * number
        if number > 0:
            self.positive += 1
        elif number < 0:
            self.negative += 1
        else:
            self.zero += 1

    def summary(self, prefix: str) -> Dict[str, Any]:
        if self.n == 0:
            return {
                f"n_{prefix}": 0,
                f"mean_{prefix}": None,
                f"std_{prefix}": None,
                f"positive_rate_{prefix}": None,
                f"negative_rate_{prefix}": None,
            }
        mean = self.total / self.n
        variance = max(
            0.0,
            self.total_sq / self.n - mean * mean,
        )
        return {
            f"n_{prefix}": self.n,
            f"mean_{prefix}": mean,
            f"std_{prefix}": math.sqrt(variance),
            f"positive_rate_{prefix}": self.positive / self.n,
            f"negative_rate_{prefix}": self.negative / self.n,
        }


class GroupAggregator:
    def __init__(
        self,
        key_names: Sequence[str],
        metric_names: Sequence[str],
    ) -> None:
        self.key_names = list(key_names)
        self.metric_names = list(metric_names)
        self.groups: Dict[
            Tuple[Any, ...],
            Dict[str, RunningMetric],
        ] = {}

    def update(
        self,
        key_values: Sequence[Any],
        row: Mapping[str, Any],
    ) -> None:
        key = tuple(key_values)
        metrics = self.groups.get(key)
        if metrics is None:
            metrics = {
                metric: RunningMetric()
                for metric in self.metric_names
            }
            self.groups[key] = metrics
        for metric in self.metric_names:
            metrics[metric].update(row.get(metric))

    def rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, metrics in self.groups.items():
            row = {
                name: value
                for name, value in zip(self.key_names, key)
            }
            for metric_name, metric in metrics.items():
                row.update(metric.summary(metric_name))
            rows.append(row)
        return sorted(
            rows,
            key=lambda row: tuple(
                str(row[name])
                for name in self.key_names
            ),
        )


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


def add_metadata(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    out = {
        "sid": metadata["sid"],
        "pair_status": metadata["pair_status"],
        "subject": metadata["subject"],
        "reference": metadata["reference"],
        "original_relation": metadata["original_relation"],
        "flipped_relation": metadata["flipped_relation"],
        "original_prediction": metadata["original_prediction"],
        "flipped_prediction": metadata["flipped_prediction"],
    }
    out.update(dict(row))
    return out


def peak_row(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Optional[Mapping[str, Any]]:
    eligible = [
        row
        for row in rows
        if finite(row.get(metric))
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: float(row[metric]),
    )


def mean_values(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if finite(row.get(metric))
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def negative_rate(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if finite(row.get(metric))
    ]
    if not values:
        return float("nan")
    return sum(value < 0 for value in values) / len(values)


def build_sample_diagnostics(
    *,
    metadata: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    final_last_delta_norm: float,
) -> Dict[str, Any]:
    state_by_group = {
        group: [
            row
            for row in state_rows
            if row["token_group"] == group
        ]
        for group in ("subject", "reference", "prompt_last")
    }
    source_by_group = {
        group: [
            row
            for row in source_rows
            if row["source_group"] == group
        ]
        for group in (
            "subject",
            "reference",
            "visual_all",
            "options",
        )
    }

    subject_peak = peak_row(
        state_by_group["subject"],
        "input_delta_norm",
    )
    reference_peak = peak_row(
        state_by_group["reference"],
        "input_delta_norm",
    )
    last_increment_peak = peak_row(
        state_by_group["prompt_last"],
        "increment_delta_norm",
    )
    last_output_peak = peak_row(
        state_by_group["prompt_last"],
        "output_delta_norm",
    )

    subject_transfer_peak = peak_row(
        source_by_group["subject"],
        "content_to_last_increment_projection_fraction",
    )
    reference_transfer_peak = peak_row(
        source_by_group["reference"],
        "content_to_last_increment_projection_fraction",
    )
    visual_transfer_peak = peak_row(
        source_by_group["visual_all"],
        "delta_to_last_increment_projection_fraction",
    )
    options_transfer_peak = peak_row(
        source_by_group["options"],
        "delta_to_last_increment_projection_fraction",
    )

    object_by_layer: Dict[int, float] = defaultdict(float)
    object_rows = (
        source_by_group["subject"]
        + source_by_group["reference"]
    )
    for row in object_rows:
        if finite(
            row.get(
                "content_to_last_increment_projection_fraction"
            )
        ):
            object_by_layer[int(row["layer"])] += float(
                row[
                    "content_to_last_increment_projection_fraction"
                ]
            )
    if object_by_layer:
        object_peak_layer, object_peak_value = max(
            object_by_layer.items(),
            key=lambda item: item[1],
        )
        object_mean_value = float(
            np.mean(list(object_by_layer.values()))
        )
        object_negative = sum(
            value < 0
            for value in object_by_layer.values()
        ) / len(object_by_layer)
    else:
        object_peak_layer = None
        object_peak_value = float("nan")
        object_mean_value = float("nan")
        object_negative = float("nan")

    subject_norm = (
        float(subject_peak["input_delta_norm"])
        if subject_peak is not None
        else float("nan")
    )
    reference_norm = (
        float(reference_peak["input_delta_norm"])
        if reference_peak is not None
        else float("nan")
    )

    return {
        "sid": metadata["sid"],
        "pair_status": metadata["pair_status"],
        "subject": metadata["subject"],
        "reference": metadata["reference"],
        "original_relation": metadata["original_relation"],
        "flipped_relation": metadata["flipped_relation"],
        "original_prediction": metadata["original_prediction"],
        "flipped_prediction": metadata["flipped_prediction"],
        "original_correct": metadata["original_correct"],
        "flipped_correct": metadata["flipped_correct"],
        "subject_peak_input_delta_norm": subject_norm,
        "subject_peak_input_layer": (
            int(subject_peak["layer"])
            if subject_peak is not None
            else None
        ),
        "reference_peak_input_delta_norm": reference_norm,
        "reference_peak_input_layer": (
            int(reference_peak["layer"])
            if reference_peak is not None
            else None
        ),
        "object_state_peak_norm": (
            max(subject_norm, reference_norm)
            if finite(subject_norm) and finite(reference_norm)
            else (
                subject_norm
                if finite(subject_norm)
                else reference_norm
            )
        ),
        "prompt_last_peak_increment_delta_norm": (
            float(last_increment_peak["increment_delta_norm"])
            if last_increment_peak is not None
            else float("nan")
        ),
        "prompt_last_peak_increment_layer": (
            int(last_increment_peak["layer"])
            if last_increment_peak is not None
            else None
        ),
        "prompt_last_peak_output_delta_norm": (
            float(last_output_peak["output_delta_norm"])
            if last_output_peak is not None
            else float("nan")
        ),
        "prompt_last_peak_output_layer": (
            int(last_output_peak["layer"])
            if last_output_peak is not None
            else None
        ),
        "final_last_delta_norm": float(final_last_delta_norm),
        "subject_content_transfer_peak": (
            float(
                subject_transfer_peak[
                    "content_to_last_increment_projection_fraction"
                ]
            )
            if subject_transfer_peak is not None
            else float("nan")
        ),
        "subject_content_transfer_peak_layer": (
            int(subject_transfer_peak["layer"])
            if subject_transfer_peak is not None
            else None
        ),
        "reference_content_transfer_peak": (
            float(
                reference_transfer_peak[
                    "content_to_last_increment_projection_fraction"
                ]
            )
            if reference_transfer_peak is not None
            else float("nan")
        ),
        "reference_content_transfer_peak_layer": (
            int(reference_transfer_peak["layer"])
            if reference_transfer_peak is not None
            else None
        ),
        "object_content_transfer_peak": object_peak_value,
        "object_content_transfer_peak_layer": object_peak_layer,
        "object_content_transfer_mean": object_mean_value,
        "object_content_negative_rate": object_negative,
        "visual_transfer_peak": (
            float(
                visual_transfer_peak[
                    "delta_to_last_increment_projection_fraction"
                ]
            )
            if visual_transfer_peak is not None
            else float("nan")
        ),
        "visual_transfer_peak_layer": (
            int(visual_transfer_peak["layer"])
            if visual_transfer_peak is not None
            else None
        ),
        "options_transfer_peak": (
            float(
                options_transfer_peak[
                    "delta_to_last_increment_projection_fraction"
                ]
            )
            if options_transfer_peak is not None
            else float("nan")
        ),
        "options_transfer_peak_layer": (
            int(options_transfer_peak["layer"])
            if options_transfer_peak is not None
            else None
        ),
    }


def standardized_difference(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float:
    if n_a <= 1 or n_b <= 1:
        return float("nan")
    pooled_numerator = (
        (n_a - 1) * std_a * std_a
        + (n_b - 1) * std_b * std_b
    )
    pooled_denominator = n_a + n_b - 2
    if pooled_denominator <= 0:
        return float("nan")
    pooled_std = math.sqrt(
        max(0.0, pooled_numerator / pooled_denominator)
    )
    if pooled_std <= 1e-12:
        return float("nan")
    return (mean_a - mean_b) / pooled_std


def build_contrasts(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    key_names: Sequence[str],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    by_key = {
        tuple(row[name] for name in key_names): row
        for row in summary_rows
    }
    result: List[Dict[str, Any]] = []

    non_status_names = [
        name
        for name in key_names
        if name != "pair_status"
    ]
    identities = sorted(set(
        tuple(row[name] for name in non_status_names)
        for row in summary_rows
    ))

    for identity in identities:
        identity_map = dict(zip(non_status_names, identity))
        clean_key = tuple(
            "both_correct"
            if name == "pair_status"
            else identity_map[name]
            for name in key_names
        )
        clean = by_key.get(clean_key)
        if clean is None:
            continue

        for status in (
            "original_only_correct",
            "flipped_only_correct",
            "both_wrong",
        ):
            error_key = tuple(
                status
                if name == "pair_status"
                else identity_map[name]
                for name in key_names
            )
            error = by_key.get(error_key)
            if error is None:
                continue

            for metric in metric_names:
                clean_mean = clean.get(f"mean_{metric}")
                error_mean = error.get(f"mean_{metric}")
                clean_std = clean.get(f"std_{metric}")
                error_std = error.get(f"std_{metric}")
                clean_n = clean.get(f"n_{metric}")
                error_n = error.get(f"n_{metric}")
                if not all(
                    finite(value)
                    for value in (
                        clean_mean,
                        error_mean,
                        clean_std,
                        error_std,
                        clean_n,
                        error_n,
                    )
                ):
                    continue

                row: Dict[str, Any] = {
                    "pair_status": status,
                    **identity_map,
                    "metric": metric,
                    "error_n": int(error_n),
                    "both_correct_n": int(clean_n),
                    "error_mean": float(error_mean),
                    "both_correct_mean": float(clean_mean),
                    "mean_difference_error_minus_correct": (
                        float(error_mean) - float(clean_mean)
                    ),
                    "standardized_difference": standardized_difference(
                        float(error_mean),
                        float(error_std),
                        int(error_n),
                        float(clean_mean),
                        float(clean_std),
                        int(clean_n),
                    ),
                }
                error_positive = error.get(
                    f"positive_rate_{metric}"
                )
                clean_positive = clean.get(
                    f"positive_rate_{metric}"
                )
                error_negative = error.get(
                    f"negative_rate_{metric}"
                )
                clean_negative = clean.get(
                    f"negative_rate_{metric}"
                )
                row["positive_rate_difference"] = (
                    float(error_positive) - float(clean_positive)
                    if finite(error_positive)
                    and finite(clean_positive)
                    else None
                )
                row["negative_rate_difference"] = (
                    float(error_negative) - float(clean_negative)
                    if finite(error_negative)
                    and finite(clean_negative)
                    else None
                )
                result.append(row)
    return result


def summarize_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    aggregator = GroupAggregator(
        key_names=("pair_status",),
        metric_names=DIAGNOSTIC_METRICS,
    )
    for row in rows:
        for status in ("all", str(row["pair_status"])):
            aggregator.update((status,), row)
    return aggregator.rows()


def quantile(
    rows: Sequence[Mapping[str, Any]],
    status: str,
    metric: str,
    q: float,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["pair_status"] == status
        and finite(row.get(metric))
    ]
    if not values:
        return float("nan")
    return float(np.quantile(values, q))


def build_failure_signatures(
    diagnostics: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    thresholds = {
        "object_state_q25": quantile(
            diagnostics,
            "both_correct",
            "object_state_peak_norm",
            0.25,
        ),
        "object_transfer_q25": quantile(
            diagnostics,
            "both_correct",
            "object_content_transfer_peak",
            0.25,
        ),
        "object_negative_q75": quantile(
            diagnostics,
            "both_correct",
            "object_content_negative_rate",
            0.75,
        ),
        "last_increment_q25": quantile(
            diagnostics,
            "both_correct",
            "prompt_last_peak_increment_delta_norm",
            0.25,
        ),
        "final_last_q25": quantile(
            diagnostics,
            "both_correct",
            "final_last_delta_norm",
            0.25,
        ),
    }

    result: List[Dict[str, Any]] = []
    for row in diagnostics:
        if row["pair_status"] == "both_correct":
            continue

        reasons: List[str] = []
        if (
            finite(row.get("object_state_peak_norm"))
            and finite(thresholds["object_state_q25"])
            and float(row["object_state_peak_norm"])
            < thresholds["object_state_q25"]
        ):
            reasons.append("weak_object_position_sensitive_state")

        if (
            finite(row.get("object_content_negative_rate"))
            and finite(thresholds["object_negative_q75"])
            and float(row["object_content_negative_rate"])
            > thresholds["object_negative_q75"]
        ):
            reasons.append("more_opposing_object_to_last_layers")

        if (
            finite(row.get("object_content_transfer_peak"))
            and finite(thresholds["object_transfer_q25"])
            and float(row["object_content_transfer_peak"])
            < thresholds["object_transfer_q25"]
        ):
            reasons.append("weak_object_to_last_content_write")

        if (
            finite(row.get("prompt_last_peak_increment_delta_norm"))
            and finite(thresholds["last_increment_q25"])
            and float(row["prompt_last_peak_increment_delta_norm"])
            < thresholds["last_increment_q25"]
        ):
            reasons.append("weak_prompt_last_increment")

        if (
            finite(row.get("final_last_delta_norm"))
            and finite(thresholds["final_last_q25"])
            and float(row["final_last_delta_norm"])
            < thresholds["final_last_q25"]
        ):
            reasons.append("weak_final_last_counterfactual_difference")

        if not reasons:
            reasons.append(
                "difference_present_but_not_explained_by_simple_missing_signal"
            )

        result.append({
            **dict(row),
            **thresholds,
            "provisional_failure_signature": ";".join(reasons),
            "signature_is_heuristic": True,
        })
    return result


def top_contrast_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    status: str,
    metric: str,
    ascending: bool,
    top_k: int,
) -> List[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["pair_status"] == status
        and row["metric"] == metric
        and finite(
            row.get("mean_difference_error_minus_correct")
        )
    ]
    return sorted(
        selected,
        key=lambda row: float(
            row["mean_difference_error_minus_correct"]
        ),
        reverse=not ascending,
    )[:top_k]


def format_number(value: Any, width: int = 12) -> str:
    if not finite(value):
        return f"{'nan':>{width}}"
    return f"{float(value):>{width}.6f}"


def build_report(
    *,
    args: argparse.Namespace,
    counts: Mapping[str, int],
    diagnostic_summary: Sequence[Mapping[str, Any]],
    source_contrasts: Sequence[Mapping[str, Any]],
    state_contrasts: Sequence[Mapping[str, Any]],
    head_contrasts: Sequence[Mapping[str, Any]],
    reconstruction_summary: Mapping[str, Any],
) -> str:
    diagnostic_by_status = {
        row["pair_status"]: row
        for row in diagnostic_summary
    }

    lines = [
        "=" * 160,
        "BATCH OBJECT-TO-LAST COUNTERFACTUAL TRANSFER BY CORRECTNESS CATEGORY",
        f"model={args.model}",
        "counts: " + ", ".join(
            f"{key}={value}"
            for key, value in sorted(counts.items())
        ),
        "=" * 160,
        "",
        "Sample-level diagnostic means",
        (
            f"{'Status':>25}{'N':>7}{'ObjState':>13}"
            f"{'ObjWrite':>13}{'ObjNeg%':>11}"
            f"{'LastIncr':>13}{'FinalLast':>13}"
            f"{'VisualPeak':>13}{'OptionPeak':>13}"
        ),
        "-" * 125,
    ]

    for status in AGGREGATE_STATUSES:
        row = diagnostic_by_status.get(status)
        if row is None:
            continue
        lines.append(
            f"{status:>25}"
            f"{int(row.get('n_object_state_peak_norm', 0)):>7}"
            f"{format_number(row.get('mean_object_state_peak_norm'), 13)}"
            f"{format_number(row.get('mean_object_content_transfer_peak'), 13)}"
            f"{format_number(
                100.0 * float(row['mean_object_content_negative_rate'])
                if finite(row.get('mean_object_content_negative_rate'))
                else None,
                11,
            )}"
            f"{format_number(
                row.get('mean_prompt_last_peak_increment_delta_norm'),
                13,
            )}"
            f"{format_number(row.get('mean_final_last_delta_norm'), 13)}"
            f"{format_number(row.get('mean_visual_transfer_peak'), 13)}"
            f"{format_number(row.get('mean_options_transfer_peak'), 13)}"
        )

    for status in (
        "original_only_correct",
        "flipped_only_correct",
        "both_wrong",
    ):
        lines += [
            "",
            f"[{status}] largest object/source path deficits versus both_correct",
            (
                f"{'Rank':>5}{'Layer':>7}{'Source':>22}"
                f"{'Metric':>48}{'ErrMean':>13}"
                f"{'Correct':>13}{'Diff':>13}{'StdDiff':>11}"
            ),
            "-" * 135,
        ]
        deficits = top_contrast_rows(
            source_contrasts,
            status=status,
            metric="content_to_last_increment_projection_fraction",
            ascending=True,
            top_k=args.top_k_report,
        )
        for rank, row in enumerate(deficits, start=1):
            lines.append(
                f"{rank:>5}"
                f"{int(row['layer']):>7}"
                f"{str(row['source_group']):>22}"
                f"{str(row['metric']):>48}"
                f"{format_number(row['error_mean'], 13)}"
                f"{format_number(row['both_correct_mean'], 13)}"
                f"{format_number(
                    row['mean_difference_error_minus_correct'],
                    13,
                )}"
                f"{format_number(row['standardized_difference'], 11)}"
            )

        lines += [
            "",
            f"[{status}] head-level object path deficits versus both_correct",
            (
                f"{'Rank':>5}{'Layer':>7}{'Head':>7}{'Source':>18}"
                f"{'ErrMean':>13}{'Correct':>13}"
                f"{'Diff':>13}{'StdDiff':>11}"
            ),
            "-" * 100,
        ]
        head_selected = [
            row
            for row in head_contrasts
            if row["pair_status"] == status
            and row["metric"]
            == "content_to_last_increment_projection_fraction"
            and row.get("source_group")
            in {"subject", "reference"}
            and finite(
                row.get("mean_difference_error_minus_correct")
            )
        ]
        head_selected = sorted(
            head_selected,
            key=lambda row: float(
                row["mean_difference_error_minus_correct"]
            ),
        )[:args.top_k_report]
        for rank, row in enumerate(head_selected, start=1):
            lines.append(
                f"{rank:>5}"
                f"{int(row['layer']):>7}"
                f"{int(row['head']):>7}"
                f"{str(row['source_group']):>18}"
                f"{format_number(row['error_mean'], 13)}"
                f"{format_number(row['both_correct_mean'], 13)}"
                f"{format_number(
                    row['mean_difference_error_minus_correct'],
                    13,
                )}"
                f"{format_number(row['standardized_difference'], 11)}"
            )

    lines += [
        "",
        "Reliability summary",
        json.dumps(
            reconstruction_summary,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Interpretation constraints:",
        "- Lower object-state norms suggest weaker position-sensitive state before transfer.",
        "- Normal object state but lower object content-write projection suggests a transfer deficit.",
        "- More negative object-write layers suggest opposing or canceling paths.",
        "- Normal transfer and normal last increment in an error category means the error is not explained by simple information absence.",
        "- both_wrong has few samples in the current dataset; its means are unstable and must be interpreted cautiously.",
        "- These comparisons are descriptive. Validate candidate missing paths with exact edge restoration/replacement.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    single = import_file(
        Path(args.single_script),
        "_single_last_increment",
    )
    edge = import_file(
        Path(args.edge_script),
        "_batch_attention_edge",
    )
    helper = import_file(
        Path(args.helper_script),
        "_batch_same_token",
    )
    base = import_file(
        Path(args.base_script),
        "_batch_centroid_base",
    )

    sources = parse_subset(
        args.sources,
        SOURCE_GROUPS,
        "source group",
    )

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    error_path = output_dir / "errors.jsonl"
    baseline_writer = StreamingCSV(
        output_dir / "baseline_pairs.csv"
    )
    state_writer = StreamingCSV(
        output_dir / "per_sample_state.csv"
    )
    source_writer = StreamingCSV(
        output_dir / "per_sample_source.csv"
    )
    head_writer = StreamingCSV(
        output_dir / "per_sample_head.csv"
    )
    reconstruction_writer = StreamingCSV(
        output_dir / "reconstruction_checks.csv"
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

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "relations": list(RELATIONS),
        "state_layers": state_layers,
        "edge_layers": edge_layers,
        "final_layer": final_layer,
        "sources": sources,
        "decoder_path": decoder_path,
        "n_decoder_layers": n_layers,
        "max_samples": args.max_samples,
        "save_head_details": args.save_head_details,
        "audit": audit,
        "pair_statuses": list(PAIR_STATUSES),
        "uses_centroid": False,
        "uses_trained_probe": False,
        "updates_model_weights": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    state_agg = GroupAggregator(
        key_names=("pair_status", "layer", "token_group"),
        metric_names=STATE_METRICS,
    )
    source_agg = GroupAggregator(
        key_names=("pair_status", "layer", "source_group"),
        metric_names=SOURCE_METRICS,
    )
    head_agg = GroupAggregator(
        key_names=(
            "pair_status",
            "layer",
            "head",
            "source_group",
        ),
        metric_names=HEAD_METRICS,
    )

    diagnostics: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    reconstruction_max = {
        "attention_reconstruction_relative_error": 0.0,
        "routing_content_relative_error": 0.0,
        "block_decomposition_relative_error": 0.0,
        "replay_max_abs_error": 0.0,
        "replay_relative_error": 0.0,
    }
    start_time = time.time()
    analyzed = 0

    try:
        for record in tqdm(
            records,
            desc=f"batch-last-increment:{args.model}",
        ):
            sid = int(record.sid)
            counts["seen"] += 1
            original_image = None
            flipped_image = None
            original_batch = None
            flipped_batch = None
            original_trace = None
            flipped_trace = None

            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                original_relation = base.normalize_relation(
                    prompt_row["answer_raw"]
                )
                if original_relation not in RELATIONS:
                    continue
                flipped_relation = OPPOSITE[original_relation]
                counts["eligible_relation_seen"] += 1

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
                        "Original and flipped tokenizations differ"
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
                group_positions = single.state_group_positions(
                    semantic
                )
                state_positions = sorted(set(
                    position
                    for positions in group_positions.values()
                    for position in positions
                ))
                last_positions = group_positions["prompt_last"]
                source_position_map = single.build_source_groups(
                    sequence_length=len(original_ids),
                    visual_indices=visual_indices,
                    token_manifest=token_manifest,
                    target_positions=last_positions,
                )

                original_trace = single.run_trace(
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
                flipped_trace = single.run_trace(
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

                original_prediction = original_trace.prediction
                flipped_prediction = flipped_trace.prediction
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
                counts[status] += 1
                counts["original_correct"] += int(original_correct)
                counts["flipped_correct"] += int(flipped_correct)
                counts["predictions_opposite"] += int(
                    OPPOSITE.get(original_prediction)
                    == flipped_prediction
                )

                metadata = {
                    "sid": sid,
                    "pair_status": status,
                    "subject": subject,
                    "reference": reference,
                    "original_relation": original_relation,
                    "flipped_relation": flipped_relation,
                    "original_prediction": original_prediction,
                    "flipped_prediction": flipped_prediction,
                    "original_correct": bool(original_correct),
                    "flipped_correct": bool(flipped_correct),
                }
                baseline_writer.write({
                    **metadata,
                    "original_scores": json.dumps(
                        original_trace.scores,
                        ensure_ascii=False,
                    ),
                    "flipped_scores": json.dumps(
                        flipped_trace.scores,
                        ensure_ascii=False,
                    ),
                })

                final_last_original = single.mean_rows(
                    original_trace.state.block_outputs[final_layer],
                    original_trace.state.positions,
                    last_positions,
                )
                final_last_flipped = single.mean_rows(
                    flipped_trace.state.block_outputs[final_layer],
                    flipped_trace.state.positions,
                    last_positions,
                )
                final_last_delta = (
                    final_last_original - final_last_flipped
                )

                state_rows, state_vectors = single.make_state_rows(
                    original=original_trace,
                    flipped=flipped_trace,
                    layers=state_layers,
                    group_positions=group_positions,
                    final_last_delta=final_last_delta,
                )

                (
                    source_rows,
                    head_rows,
                    reconstruction_rows,
                    _edge_group_vectors,
                    _edge_head_vectors,
                ) = single.make_edge_rows(
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

                for row in state_rows:
                    enriched = add_metadata(row, metadata)
                    state_writer.write(enriched)
                    for aggregate_status in ("all", status):
                        state_agg.update(
                            (
                                aggregate_status,
                                int(row["layer"]),
                                str(row["token_group"]),
                            ),
                            row,
                        )

                for row in source_rows:
                    enriched = add_metadata(row, metadata)
                    source_writer.write(enriched)
                    for aggregate_status in ("all", status):
                        source_agg.update(
                            (
                                aggregate_status,
                                int(row["layer"]),
                                str(row["source_group"]),
                            ),
                            row,
                        )

                for row in head_rows:
                    if args.save_head_details:
                        head_writer.write(
                            add_metadata(row, metadata)
                        )
                    for aggregate_status in ("all", status):
                        head_agg.update(
                            (
                                aggregate_status,
                                int(row["layer"]),
                                int(row["head"]),
                                str(row["source_group"]),
                            ),
                            row,
                        )

                for row in reconstruction_rows:
                    enriched = add_metadata(row, metadata)
                    reconstruction_writer.write(enriched)
                    for metric in reconstruction_max:
                        if finite(row.get(metric)):
                            reconstruction_max[metric] = max(
                                reconstruction_max[metric],
                                float(row[metric]),
                            )

                diagnostic = build_sample_diagnostics(
                    metadata=metadata,
                    state_rows=state_rows,
                    source_rows=source_rows,
                    final_last_delta_norm=single.norm(
                        final_last_delta
                    ),
                )
                diagnostics.append(diagnostic)

                analyzed += 1
                if (
                    args.print_every > 0
                    and analyzed % args.print_every == 0
                ):
                    print(
                        f"[{analyzed}] sid={sid} "
                        f"status={status} "
                        f"orig={original_prediction} "
                        f"flip={flipped_prediction}",
                        flush=True,
                    )

            except Exception as error:
                counts["errors"] += 1
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"[ERROR] sid={sid}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

            finally:
                for image in (
                    original_image,
                    flipped_image,
                ):
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
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        baseline_writer.close()
        state_writer.close()
        source_writer.close()
        head_writer.close()
        reconstruction_writer.close()

    if not diagnostics:
        raise RuntimeError(
            "No eligible samples were analyzed. Inspect errors.jsonl."
        )

    state_summary = state_agg.rows()
    source_summary = source_agg.rows()
    head_summary = head_agg.rows()
    diagnostic_summary = summarize_diagnostics(diagnostics)

    state_contrasts = build_contrasts(
        state_summary,
        key_names=("pair_status", "layer", "token_group"),
        metric_names=STATE_METRICS,
    )
    source_contrasts = build_contrasts(
        source_summary,
        key_names=("pair_status", "layer", "source_group"),
        metric_names=SOURCE_METRICS,
    )
    head_contrasts = build_contrasts(
        head_summary,
        key_names=(
            "pair_status",
            "layer",
            "head",
            "source_group",
        ),
        metric_names=HEAD_METRICS,
    )
    failure_signatures = build_failure_signatures(
        diagnostics
    )

    write_csv(
        output_dir / "sample_diagnostics.csv",
        diagnostics,
    )
    write_csv(
        output_dir / "diagnostic_summary_by_pair_status.csv",
        diagnostic_summary,
    )
    write_csv(
        output_dir / "state_summary_by_pair_status.csv",
        state_summary,
    )
    write_csv(
        output_dir / "source_summary_by_pair_status.csv",
        source_summary,
    )
    write_csv(
        output_dir / "head_summary_by_pair_status.csv",
        head_summary,
    )
    write_csv(
        output_dir / "state_contrasts_vs_both_correct.csv",
        state_contrasts,
    )
    write_csv(
        output_dir / "source_contrasts_vs_both_correct.csv",
        source_contrasts,
    )
    write_csv(
        output_dir / "head_contrasts_vs_both_correct.csv",
        head_contrasts,
    )
    write_csv(
        output_dir / "provisional_failure_signatures.csv",
        failure_signatures,
    )

    report = build_report(
        args=args,
        counts=counts,
        diagnostic_summary=diagnostic_summary,
        source_contrasts=source_contrasts,
        state_contrasts=state_contrasts,
        head_contrasts=head_contrasts,
        reconstruction_summary=reconstruction_max,
    )
    (output_dir / "report.txt").write_text(
        report,
        encoding="utf-8",
    )
    print("\n" + report)

    summary = {
        "version": VERSION,
        "model": args.model,
        "counts": dict(counts),
        "analyzed": analyzed,
        "elapsed_minutes": (
            time.time() - start_time
        ) / 60.0,
        "reconstruction_max": reconstruction_max,
        "output_files": [
            "config.json",
            "baseline_pairs.csv",
            "per_sample_state.csv",
            "per_sample_source.csv",
            "per_sample_head.csv (only with --save-head-details)",
            "reconstruction_checks.csv",
            "sample_diagnostics.csv",
            "diagnostic_summary_by_pair_status.csv",
            "state_summary_by_pair_status.csv",
            "source_summary_by_pair_status.csv",
            "head_summary_by_pair_status.csv",
            "state_contrasts_vs_both_correct.csv",
            "source_contrasts_vs_both_correct.csv",
            "head_contrasts_vs_both_correct.csv",
            "provisional_failure_signatures.csv",
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

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
