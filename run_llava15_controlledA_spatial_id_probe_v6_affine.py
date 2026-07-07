#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-layer spatial-ID probe for HF LLaVA-1.5 on Controlled_Images_A.

This version supports both spatial axes:
  * horizontal: left <-> right
  * vertical:   under/bottom <-> on/top

When --fit-all is used with both axes, it also reports a joint four-way
relation reconstruction analysis.  Unlike the legacy decoder, it uses one
shared per-object baseline for both axes and dual-basis coordinates to remove
x/y cross-talk before testing a cardinal spatial readout.

It can run in two modes:
  * --fit-all: estimate each axis, its object means, and its threshold from all
    available samples of that axis, then score the same full set. This is a
    full-data descriptive alignment / reconstruction score, not an unseen-data
    generalization score.
  * default: estimate on a grouped subset and evaluate on held-out samples.

No LLaVA parameters and no learned classifier parameters are trained. The only
estimated quantities are per-object means, a mean-difference direction, and a
scalar threshold.

Run from the AdaptVis repository root.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"
ALL_RELATIONS = {"left", "right", "on", "under"}

# Positive direction is always the second class in ``classes``.
AXIS_SPECS: Dict[str, Dict[str, Any]] = {
    "horizontal": {
        "relations": {"left", "right"},
        "classes": ("left", "right"),
        "negative": "left",
        "positive": "right",
        "position_map": {
            "left": {"subject": "left", "reference": "right"},
            "right": {"subject": "right", "reference": "left"},
        },
        "pair_positive_relation": "right",
        "description": "right-minus-left object-centred residual direction",
    },
    "vertical": {
        "relations": {"on", "under"},
        "classes": ("bottom", "top"),
        "negative": "bottom",
        "positive": "top",
        "position_map": {
            "on": {"subject": "top", "reference": "bottom"},
            "under": {"subject": "bottom", "reference": "top"},
        },
        "pair_positive_relation": "on",
        "description": "top-minus-bottom object-centred residual direction; this is a vertical relational component, not a pure support-relation proof",
    },
}

QUESTION_RE = re.compile(
    r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
    r"in\s+relation\s+to\s+(?:the\s+)?(?P<reference>.+?)\?\s*Answer",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class PromptMeta:
    subject: str
    reference: str
    verb: str


@dataclass
class SampleRecord:
    sid: int
    relation: str
    subject: str
    reference: str
    prompt: str
    group: str
    states: Dict[int, Dict[str, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-layer horizontal/vertical spatial-ID probe for HF LLaVA-1.5 on Controlled_Images_A."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--layers", default="13,16", help="Zero-based decoder block indices, e.g. 13,16 for the 14th and 17th blocks.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument(
        "--split-unit",
        choices=["pair", "sample"],
        default="pair",
        help="Only used without --fit-all. 'pair' keeps an unordered subject/reference pair in one split.",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Estimate object means, axis, and threshold on all usable samples and evaluate the same full set."
    )
    parser.add_argument(
        "--axes",
        default="horizontal,vertical",
        help="Comma-separated subset of: horizontal,vertical. Default: both.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["clean", "original"],
        default="clean",
        help="'clean' removes explicit answer-label options to reduce answer-word leakage.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after keeping supported spatial relations.")
    parser.add_argument("--output", default="output/llava15_controlledA_spatial_id_probe")
    parser.add_argument("--save-states", action="store_true", help="Also save token states, axes, and object means in .npz.")
    parser.add_argument("--print-first", type=int, default=5)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layers(text: str) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or min(values) < 0:
        raise ValueError(f"Invalid --layers: {text!r}")
    return values


def parse_axes(text: str) -> List[str]:
    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = sorted(set(values) - set(AXIS_SPECS))
    if unknown:
        raise ValueError(f"Unsupported --axes values: {unknown}; choices are {sorted(AXIS_SPECS)}")
    if not values:
        raise ValueError("--axes must contain at least one axis.")
    return list(dict.fromkeys(values))


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_relation(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().lower()


def canonical_object(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_prompt(prompt: str) -> PromptMeta:
    match = QUESTION_RE.search(prompt)
    if match is None:
        raise ValueError(f"Could not parse subject/reference from prompt:\n{prompt}")
    return PromptMeta(
        subject=canonical_object(match.group("subject")),
        reference=canonical_object(match.group("reference")),
        verb=match.group("verb").lower(),
    )


def make_clean_prompt(meta: PromptMeta) -> str:
    return (
        f"<image>\nUSER: Where {meta.verb} the {meta.subject} "
        f"in relation to the {meta.reference}?\nASSISTANT:"
    )


def load_prompt_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            row = json.loads(line)
            if int(row.get("id", expected_id)) != expected_id:
                raise ValueError(
                    "Prompt IDs must be contiguous and aligned with dataset order. "
                    f"Expected {expected_id}, found {row.get('id')}"
                )
            rows.append(row)
    return rows


def extract_images_from_batch(batch: Mapping[str, Any]) -> Iterable[Any]:
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    result: List[int] = []
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            result.append(start)
    return result


def find_phrase_last_token(input_ids: Sequence[int], tokenizer, phrase: str) -> int:
    candidates: List[Tuple[int, int, List[int]]] = []
    seen: set[Tuple[int, ...]] = set()
    for variant in (" " + phrase, phrase):
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids
        key = tuple(int(x) for x in token_ids)
        if not key or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, token_ids):
            candidates.append((start, start + len(token_ids) - 1, list(token_ids)))

    if not candidates:
        raise ValueError(
            f"Could not find phrase {phrase!r} in tokenized prompt. "
            "This normally means the prompt template changed."
        )

    max_len = max(len(ids) for _, _, ids in candidates)
    best = [(start, end) for start, end, ids in candidates if len(ids) == max_len]
    unique = sorted(set(best))
    if len(unique) != 1:
        raise ValueError(
            f"Ambiguous token match for phrase {phrase!r}: {unique}. "
            "Use a more explicit prompt template before running this probe."
        )
    return unique[0][1]


def map_text_index_to_merged_index(text_index: int, image_index: int, image_seq_length: int) -> int:
    if text_index == image_index:
        raise ValueError("The requested object token cannot be the <image> placeholder.")
    if text_index < image_index:
        return int(text_index)
    return int(text_index + image_seq_length - 1)


class SelectedTokenCapture:
    """Capture subject/reference residuals after selected decoder blocks."""

    def __init__(self, language_model, layers: Sequence[int]) -> None:
        self.layers = list(language_model.model.layers)
        if max(layers) >= len(self.layers):
            raise ValueError(
                f"Requested layer {max(layers)} but model has only {len(self.layers)} decoder blocks."
            )
        self.selected = set(int(layer) for layer in layers)
        self.subject_index = -1
        self.reference_index = -1
        self.states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.handles = [
            self.layers[layer].register_forward_hook(self._make_hook(layer))
            for layer in sorted(self.selected)
        ]

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Expected [1, L, H] decoder output, got {tuple(hidden.shape)}")
            if not (0 <= self.subject_index < hidden.shape[1]):
                raise RuntimeError(
                    f"Subject merged index {self.subject_index} is outside sequence length {hidden.shape[1]}"
                )
            if not (0 <= self.reference_index < hidden.shape[1]):
                raise RuntimeError(
                    f"Reference merged index {self.reference_index} is outside sequence length {hidden.shape[1]}"
                )
            self.states[layer_index] = {
                "subject": hidden[0, self.subject_index].detach().float().cpu().clone(),
                "reference": hidden[0, self.reference_index].detach().float().cpu().clone(),
            }
        return hook

    def begin(self, subject_index: int, reference_index: int) -> None:
        self.subject_index = int(subject_index)
        self.reference_index = int(reference_index)
        self.states = {}

    def collect(self) -> Dict[int, Dict[str, torch.Tensor]]:
        missing = sorted(self.selected - set(self.states))
        if missing:
            raise RuntimeError(f"No residuals captured for layers: {missing}")
        return self.states

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def choose_group_split(
    records: Sequence[SampleRecord],
    *,
    test_fraction: float,
    seed: int,
    split_unit: str,
    required_relations: set[str],
) -> Tuple[set[int], set[int]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must lie in (0, 1).")

    if split_unit == "sample":
        group_of = {record.sid: str(record.sid) for record in records}
    else:
        group_of = {record.sid: record.group for record in records}

    group_to_records: Dict[str, List[SampleRecord]] = defaultdict(list)
    for record in records:
        group_to_records[group_of[record.sid]].append(record)
    groups = sorted(group_to_records)
    if len(groups) < 2:
        raise RuntimeError("Need at least two split groups.")

    total = len(records)
    total_by_label = Counter(record.relation for record in records)
    target_test = total * test_fraction
    target_by_label = {label: count * test_fraction for label, count in total_by_label.items()}

    best: Optional[Tuple[float, set[str]]] = None
    for trial in range(512):
        rng = random.Random(seed + 1009 * trial)
        test_groups = {group for group in groups if rng.random() < test_fraction}
        if not test_groups or len(test_groups) == len(groups):
            continue
        test_records = [record for group in test_groups for record in group_to_records[group]]
        test_by_label = Counter(record.relation for record in test_records)
        if any(test_by_label[label] == 0 for label in required_relations):
            continue
        score = abs(len(test_records) - target_test)
        score += 3.0 * sum(abs(test_by_label[label] - target_by_label[label]) for label in required_relations)
        if best is None or score < best[0]:
            best = (score, test_groups)

    if best is None:
        raise RuntimeError("Could not construct a split containing every required spatial relation.")

    test_sids = {record.sid for group in best[1] for record in group_to_records[group]}
    train_sids = {record.sid for record in records} - test_sids
    return train_sids, test_sids


def make_occurrences(
    records: Sequence[SampleRecord],
    *,
    layer: int,
    axis_name: str,
) -> List[Dict[str, Any]]:
    spec = AXIS_SPECS[axis_name]
    position_map = spec["position_map"]
    result: List[Dict[str, Any]] = []
    for record in records:
        if record.relation not in spec["relations"]:
            continue
        for role, name in (("subject", record.subject), ("reference", record.reference)):
            result.append(
                {
                    "sid": record.sid,
                    "role": role,
                    "object": name,
                    "position": position_map[record.relation][role],
                    "vector": record.states[layer][role],
                }
            )
    return result


def train_object_means(
    occurrences: Sequence[Dict[str, Any]],
    *,
    classes: Tuple[str, str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    by_object: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in occurrences:
        by_object[item["object"]].append(item)

    means: Dict[str, np.ndarray] = {}
    dropped: Dict[str, int] = {}
    required = set(classes)
    for name, items in by_object.items():
        positions = {item["position"] for item in items}
        if positions != required:
            dropped[name] = len(items)
            continue
        means[name] = np.mean(np.stack([item["vector"] for item in items], axis=0), axis=0)
    return means, dropped


def center_occurrences(
    occurrences: Sequence[Dict[str, Any]],
    object_means: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in occurrences:
        mean = object_means.get(item["object"])
        if mean is None:
            continue
        copied = dict(item)
        copied["residual"] = item["vector"] - mean
        result.append(copied)
    return result


def safe_accuracy(items: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not items:
        return None
    return float(np.mean([item["correct"] for item in items]))


def object_direction_statistics(
    centered_estimation: Sequence[Dict[str, Any]],
    *,
    negative: str,
    positive: str,
    global_axis: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    by_object_position: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for item in centered_estimation:
        by_object_position[item["object"]][item["position"]].append(item["residual"])

    cosines: Dict[str, float] = {}
    directions: Dict[str, np.ndarray] = {}
    for name, by_position in by_object_position.items():
        if not by_position[negative] or not by_position[positive]:
            continue
        direction = (
            np.mean(np.stack(by_position[positive], axis=0), axis=0)
            - np.mean(np.stack(by_position[negative], axis=0), axis=0)
        )
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        directions[name] = direction.astype(np.float32)
        cosines[name] = float(np.dot(direction / norm, global_axis))

    values = np.array(list(cosines.values()), dtype=np.float32)
    summary = {
        "objects_with_direction": int(len(cosines)),
        "object_direction_cosine_mean": float(np.mean(values)) if len(values) else None,
        "object_direction_cosine_median": float(np.median(values)) if len(values) else None,
        "object_direction_positive_fraction": float(np.mean(values > 0.0)) if len(values) else None,
        "per_object_direction_cosine": {name: float(value) for name, value in sorted(cosines.items())},
    }
    if directions:
        ordered = sorted(directions)
        artifacts = {
            "object_direction_names": np.array(ordered, dtype=object),
            "object_directions": np.stack([directions[name] for name in ordered], axis=0).astype(np.float32),
            "object_direction_cosines": np.array([cosines[name] for name in ordered], dtype=np.float32),
        }
    else:
        artifacts = {}
    return summary, artifacts


def _pair_relation_labels(axis_name: str, spec: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (negative_relation, positive_relation) for a spatial axis."""
    if axis_name == "horizontal":
        return "left", "right"
    if axis_name == "vertical":
        return "under", "on"
    raise ValueError(f"Unsupported axis: {axis_name}")


def _pair_deltas(
    records: Sequence[SampleRecord],
    *,
    layer: int,
    object_means: Mapping[str, np.ndarray],
    axis: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute subject-minus-reference coordinates along one fitted axis."""
    result: List[Dict[str, Any]] = []
    for record in records:
        subject_mean = object_means.get(record.subject)
        reference_mean = object_means.get(record.reference)
        if subject_mean is None or reference_mean is None:
            continue
        subject_residual = record.states[layer]["subject"] - subject_mean
        reference_residual = record.states[layer]["reference"] - reference_mean
        delta = float(np.dot(subject_residual, axis) - np.dot(reference_residual, axis))
        result.append(
            {
                "sid": record.sid,
                "relation": record.relation,
                "delta": delta,
            }
        )
    return result


def score_axis_layer(
    records: Sequence[SampleRecord],
    *,
    layer: int,
    axis_name: str,
    estimation_sids: set[int],
    evaluation_sids: set[int],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    spec = AXIS_SPECS[axis_name]
    negative = str(spec["negative"])
    positive = str(spec["positive"])
    classes = tuple(spec["classes"])
    negative_relation, positive_relation = _pair_relation_labels(axis_name, spec)

    axis_records = [record for record in records if record.relation in spec["relations"]]
    estimation_records = [record for record in axis_records if record.sid in estimation_sids]
    evaluation_records = [record for record in axis_records if record.sid in evaluation_sids]
    estimation_occurrences = make_occurrences(estimation_records, layer=layer, axis_name=axis_name)
    evaluation_occurrences = make_occurrences(evaluation_records, layer=layer, axis_name=axis_name)

    object_means, dropped_objects = train_object_means(estimation_occurrences, classes=classes)
    centered_estimation = center_occurrences(estimation_occurrences, object_means)
    centered_evaluation = center_occurrences(evaluation_occurrences, object_means)
    if not centered_estimation or not centered_evaluation:
        raise RuntimeError(
            f"Layer {layer}, {axis_name}: no usable centred occurrences. "
            "Inspect object coverage or use --fit-all."
        )

    by_position: Dict[str, List[np.ndarray]] = defaultdict(list)
    for item in centered_estimation:
        by_position[item["position"]].append(item["residual"])
    if not by_position[negative] or not by_position[positive]:
        raise RuntimeError(f"Layer {layer}, {axis_name}: one axis endpoint is absent after centering.")

    mean_negative = np.mean(np.stack(by_position[negative], axis=0), axis=0)
    mean_positive = np.mean(np.stack(by_position[positive], axis=0), axis=0)
    axis = mean_positive - mean_negative
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-8:
        raise RuntimeError(f"Layer {layer}, {axis_name}: learned axis has near-zero norm.")
    axis = axis / axis_norm

    estimation_scores: Dict[str, List[float]] = {negative: [], positive: []}
    for item in centered_estimation:
        estimation_scores[item["position"]].append(float(np.dot(item["residual"], axis)))
    threshold = 0.5 * (
        float(np.mean(estimation_scores[negative])) + float(np.mean(estimation_scores[positive]))
    )

    evaluated: List[Dict[str, Any]] = []
    for item in centered_evaluation:
        score = float(np.dot(item["residual"], axis))
        prediction = positive if score > threshold else negative
        enriched = dict(item)
        enriched["score"] = score
        enriched["prediction"] = prediction
        enriched["correct"] = prediction == item["position"]
        evaluated.append(enriched)

    subject_eval = [item for item in evaluated if item["role"] == "subject"]
    reference_eval = [item for item in evaluated if item["role"] == "reference"]

    # A pair coordinate is subject minus reference on this axis.  We calibrate
    # its sign and scale from the estimation data.  Scale normalisation is
    # essential before comparing |horizontal| and |vertical| evidence.
    estimation_pair_deltas = _pair_deltas(
        estimation_records,
        layer=layer,
        object_means=object_means,
        axis=axis,
    )
    pair_by_relation: Dict[str, List[float]] = defaultdict(list)
    for item in estimation_pair_deltas:
        pair_by_relation[item["relation"]].append(float(item["delta"]))
    if not pair_by_relation[negative_relation] or not pair_by_relation[positive_relation]:
        raise RuntimeError(
            f"Layer {layer}, {axis_name}: cannot calibrate pair relation coordinate; "
            "one endpoint has no usable object pairs."
        )
    pair_negative_mean = float(np.mean(pair_by_relation[negative_relation]))
    pair_positive_mean = float(np.mean(pair_by_relation[positive_relation]))
    pair_center = 0.5 * (pair_negative_mean + pair_positive_mean)
    pair_half_margin = 0.5 * abs(pair_positive_mean - pair_negative_mean)
    if not np.isfinite(pair_half_margin) or pair_half_margin < 1e-8:
        raise RuntimeError(f"Layer {layer}, {axis_name}: pair-coordinate margin is near zero.")
    pair_orientation = 1.0 if pair_positive_mean >= pair_negative_mean else -1.0

    evaluation_pair_deltas = _pair_deltas(
        evaluation_records,
        layer=layer,
        object_means=object_means,
        axis=axis,
    )
    pair_predictions: List[bool] = []
    for item in evaluation_pair_deltas:
        normalized_delta = pair_orientation * (float(item["delta"]) - pair_center) / pair_half_margin
        predicted_relation = positive_relation if normalized_delta > 0.0 else negative_relation
        pair_predictions.append(predicted_relation == item["relation"])

    direction_summary, direction_artifacts = object_direction_statistics(
        centered_estimation,
        negative=negative,
        positive=positive,
        global_axis=axis,
    )

    summary: Dict[str, Any] = {
        "axis": axis_name,
        "axis_description": spec["description"],
        "layer": int(layer),
        "estimation_samples": len(estimation_records),
        "evaluation_samples": len(evaluation_records),
        "estimation_occurrences": len(estimation_occurrences),
        "evaluation_occurrences": len(evaluation_occurrences),
        "objects_with_balanced_estimation_positions": len(object_means),
        "objects_dropped_unbalanced_estimation_positions": len(dropped_objects),
        "dropped_objects": dict(sorted(dropped_objects.items())),
        "usable_evaluation_occurrences": len(evaluated),
        "subject_evaluation_n": len(subject_eval),
        "reference_evaluation_n": len(reference_eval),
        "pair_evaluation_n": len(evaluation_pair_deltas),
        "occurrence_accuracy": safe_accuracy(evaluated),
        "subject_position_accuracy": safe_accuracy(subject_eval),
        "reference_position_accuracy": safe_accuracy(reference_eval),
        "pairwise_relation_accuracy": float(np.mean(pair_predictions)) if pair_predictions else None,
        "estimation_axis_margin": float(np.mean(estimation_scores[positive]) - np.mean(estimation_scores[negative])),
        "threshold": float(threshold),
        "axis_norm_before_normalization": axis_norm,
        "pair_negative_relation": negative_relation,
        "pair_positive_relation": positive_relation,
        "pair_negative_mean": pair_negative_mean,
        "pair_positive_mean": pair_positive_mean,
        "pair_center": pair_center,
        "pair_half_margin": pair_half_margin,
        "pair_orientation": pair_orientation,
        "evaluation_score_means": {
            negative: float(np.mean([item["score"] for item in evaluated if item["position"] == negative]))
            if any(item["position"] == negative for item in evaluated)
            else None,
            positive: float(np.mean([item["score"] for item in evaluated if item["position"] == positive]))
            if any(item["position"] == positive for item in evaluated)
            else None,
        },
        **direction_summary,
    }
    artifacts: Dict[str, np.ndarray] = {
        "axis": axis.astype(np.float32),
        "object_names": np.array(sorted(object_means), dtype=object),
        "object_means": np.stack([object_means[name] for name in sorted(object_means)], axis=0).astype(np.float32),
        "threshold": np.array([threshold], dtype=np.float32),
        "pair_center": np.array([pair_center], dtype=np.float32),
        "pair_half_margin": np.array([pair_half_margin], dtype=np.float32),
        "pair_orientation": np.array([pair_orientation], dtype=np.float32),
    }
    artifacts.update(direction_artifacts)
    return summary, artifacts


def _restore_axis_fit(artifacts: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, float, float]:
    names = [str(name) for name in artifacts["object_names"].tolist()]
    means = artifacts["object_means"]
    object_means = {name: means[i] for i, name in enumerate(names)}
    axis = artifacts["axis"].astype(np.float32)
    center = float(artifacts["pair_center"][0])
    half_margin = float(artifacts["pair_half_margin"][0])
    orientation = float(artifacts["pair_orientation"][0])
    return axis, object_means, center, half_margin, orientation


def score_four_way_axis_decoder(
    records: Sequence[SampleRecord],
    *,
    layer: int,
    horizontal_artifacts: Mapping[str, np.ndarray],
    vertical_artifacts: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Decode {left,right,on,under} by competing normalized x/y pair evidence.

    For each image, we form the subject-minus-reference coordinate along each
    fitted axis.  Each coordinate is centred and divided by its own
    estimation-set half-margin, so the horizontal and vertical magnitudes are
    commensurate.  The axis with larger absolute normalised evidence chooses
    the relation family; the sign chooses the relation within that family.
    """
    x_axis, x_means, x_center, x_half_margin, x_orientation = _restore_axis_fit(horizontal_artifacts)
    y_axis, y_means, y_center, y_half_margin, y_orientation = _restore_axis_fit(vertical_artifacts)
    eps = 1e-8

    confusion: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in ("left", "right", "on", "under")}
        for gt in ("left", "right", "on", "under")
    }
    correct: List[bool] = []
    decoded_rows: List[Tuple[int, str, str, float, float]] = []
    selected_axis: Counter[str] = Counter()

    for record in records:
        if record.relation not in ALL_RELATIONS:
            continue
        x_subject_mean = x_means.get(record.subject)
        x_reference_mean = x_means.get(record.reference)
        y_subject_mean = y_means.get(record.subject)
        y_reference_mean = y_means.get(record.reference)
        if any(value is None for value in (x_subject_mean, x_reference_mean, y_subject_mean, y_reference_mean)):
            continue

        x_delta = float(
            np.dot(record.states[layer]["subject"] - x_subject_mean, x_axis)
            - np.dot(record.states[layer]["reference"] - x_reference_mean, x_axis)
        )
        y_delta = float(
            np.dot(record.states[layer]["subject"] - y_subject_mean, y_axis)
            - np.dot(record.states[layer]["reference"] - y_reference_mean, y_axis)
        )
        x_evidence = x_orientation * (x_delta - x_center) / max(x_half_margin, eps)
        y_evidence = y_orientation * (y_delta - y_center) / max(y_half_margin, eps)

        if abs(x_evidence) >= abs(y_evidence):
            predicted = "right" if x_evidence > 0.0 else "left"
            selected_axis["horizontal"] += 1
        else:
            predicted = "on" if y_evidence > 0.0 else "under"
            selected_axis["vertical"] += 1

        confusion[record.relation][predicted] += 1
        correct.append(predicted == record.relation)
        decoded_rows.append((record.sid, record.relation, predicted, x_evidence, y_evidence))

    per_relation: Dict[str, Dict[str, Optional[float]]] = {}
    for relation in ("left", "right", "on", "under"):
        total = int(sum(confusion[relation].values()))
        per_relation[relation] = {
            "n": total,
            "accuracy": (float(confusion[relation][relation] / total) if total else None),
        }

    summary: Dict[str, Any] = {
        "decoder": "max_abs_normalized_pair_evidence",
        "description": (
            "Predict horizontal when |x| >= |y| and vertical otherwise; "
            "within the chosen family, the sign determines the relation. "
            "Each coordinate is normalized by its fitted pair half-margin."
        ),
        "layer": int(layer),
        "evaluation_n": int(len(decoded_rows)),
        "four_way_axis_decoding_accuracy": float(np.mean(correct)) if correct else None,
        "per_relation": per_relation,
        "selected_axis_counts": dict(selected_axis),
        "selected_axis_fraction": {
            "horizontal": float(selected_axis["horizontal"] / len(decoded_rows)) if decoded_rows else None,
            "vertical": float(selected_axis["vertical"] / len(decoded_rows)) if decoded_rows else None,
        },
        "confusion_matrix": confusion,
        "note": (
            "When used with --fit-all, this is a full-data reconstruction / alignment accuracy, "
            "not a held-out generalization estimate."
        ),
    }
    if decoded_rows:
        artifacts = {
            "four_way_sid": np.array([row[0] for row in decoded_rows], dtype=np.int64),
            "four_way_ground_truth": np.array([row[1] for row in decoded_rows], dtype=object),
            "four_way_prediction": np.array([row[2] for row in decoded_rows], dtype=object),
            "four_way_x_evidence": np.array([row[3] for row in decoded_rows], dtype=np.float32),
            "four_way_y_evidence": np.array([row[4] for row in decoded_rows], dtype=np.float32),
        }
    else:
        artifacts = {}
    return summary, artifacts



def _shared_object_means(
    records: Sequence[SampleRecord],
    *,
    layer: int,
) -> Dict[str, np.ndarray]:
    """Fit one object-specific baseline from *all* relation families.

    Unlike the legacy four-way decoder, this uses the same per-object mean for
    x and y.  This is required before cross-axis magnitudes can be compared:
    otherwise horizontal and vertical coordinates are measured relative to two
    incompatible object baselines.
    """
    by_object: Dict[str, List[np.ndarray]] = defaultdict(list)
    for record in records:
        if record.relation not in ALL_RELATIONS:
            continue
        by_object[record.subject].append(record.states[layer]["subject"])
        by_object[record.reference].append(record.states[layer]["reference"])
    return {
        name: np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        for name, vectors in by_object.items()
        if vectors
    }


def _shared_pair_residual(
    record: SampleRecord,
    *,
    layer: int,
    object_means: Mapping[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Return the object-centred subject-minus-reference residual."""
    subject_mean = object_means.get(record.subject)
    reference_mean = object_means.get(record.reference)
    if subject_mean is None or reference_mean is None:
        return None
    return (
        (record.states[layer]["subject"] - subject_mean)
        - (record.states[layer]["reference"] - reference_mean)
    ).astype(np.float32)


def _classification_payload(
    rows: Sequence[Tuple[int, str, str]],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    labels = ("left", "right", "on", "under")
    confusion: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in labels}
        for gt in labels
    }
    correct: List[bool] = []
    for _, gt, pred in rows:
        confusion[gt][pred] += 1
        correct.append(gt == pred)

    per_relation: Dict[str, Dict[str, Optional[float]]] = {}
    for relation in labels:
        total = int(sum(confusion[relation].values()))
        per_relation[relation] = {
            "n": total,
            "accuracy": (float(confusion[relation][relation] / total) if total else None),
        }

    summary = {
        "accuracy": float(np.mean(correct)) if correct else None,
        "evaluation_n": int(len(rows)),
        "per_relation": per_relation,
        "confusion_matrix": confusion,
    }
    artifacts = {
        "sid": np.array([sid for sid, _, _ in rows], dtype=np.int64),
        "ground_truth": np.array([gt for _, gt, _ in rows], dtype=object),
        "prediction": np.array([pred for _, _, pred in rows], dtype=object),
    }
    return summary, artifacts


def score_four_way_joint_coordinate_decoder(
    records: Sequence[SampleRecord],
    *,
    layer: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Evaluate a properly calibrated 2D spatial-coordinate readout.

    This replaces the legacy max(|x|, |y|) decoder whose x/y coordinates were
    fitted against different per-object means.  The procedure is:

      1. Fit one shared object baseline using all four relation families.
      2. Build horizontal and vertical relation directions from centred
         subject-minus-reference residuals.
      3. Use the dual basis (D^T D)^(-1) D^T to remove x/y cross-talk when
         the two directions are not orthogonal.
      4. Standardise the resulting x/y coordinates with endpoint margins.

    The ``geometric`` decoder then chooses the best cardinal direction.  The
    optional ``prototype`` decoder is deliberately reported separately: it
    tells us whether the *joint 2D state* contains four relation clusters,
    but it is not evidence for a clean Cartesian coordinate system.
    """
    usable = [record for record in records if record.relation in ALL_RELATIONS]
    object_means = _shared_object_means(usable, layer=layer)

    rows_raw: List[Tuple[SampleRecord, np.ndarray]] = []
    by_relation: Dict[str, List[np.ndarray]] = defaultdict(list)
    for record in usable:
        delta = _shared_pair_residual(record, layer=layer, object_means=object_means)
        if delta is None:
            continue
        rows_raw.append((record, delta))
        by_relation[record.relation].append(delta)

    missing = [label for label in ("left", "right", "on", "under") if not by_relation[label]]
    if missing:
        raise RuntimeError(f"Layer {layer}: cannot fit joint four-way decoder; missing labels: {missing}")

    relation_means = {
        label: np.mean(np.stack(by_relation[label], axis=0), axis=0).astype(np.float32)
        for label in ("left", "right", "on", "under")
    }

    # Cardinal directions in the hidden-state space.
    x_raw = relation_means["right"] - relation_means["left"]
    y_raw = relation_means["on"] - relation_means["under"]
    x_norm = float(np.linalg.norm(x_raw))
    y_norm = float(np.linalg.norm(y_raw))
    if x_norm < 1e-8 or y_norm < 1e-8:
        raise RuntimeError(f"Layer {layer}: a joint spatial direction has near-zero norm.")
    x_axis = (x_raw / x_norm).astype(np.float32)
    y_axis = (y_raw / y_norm).astype(np.float32)

    # D contains unit-but-not-necessarily-orthogonal directions.  The dual
    # basis gives coefficients in this span instead of raw dot products, which
    # prevents y leakage into x (and vice versa) when axes are correlated.
    D = np.stack([x_axis, y_axis], axis=1).astype(np.float64)  # [hidden, 2]
    gram = D.T @ D
    gram_eig = np.linalg.eigvalsh(gram)
    if float(np.min(gram_eig)) < 1e-6:
        raise RuntimeError(
            f"Layer {layer}: horizontal/vertical directions are nearly collinear "
            f"(min Gram eigenvalue={float(np.min(gram_eig)):.3e})."
        )
    dual = np.linalg.inv(gram) @ D.T  # [2, hidden]
    axis_cosine = float(np.dot(x_axis, y_axis))
    gram_condition = float(np.linalg.cond(gram))

    raw_coordinates: List[Tuple[SampleRecord, np.ndarray]] = []
    coords_by_relation: Dict[str, List[np.ndarray]] = defaultdict(list)
    for record, delta in rows_raw:
        coeff = (dual @ delta.astype(np.float64)).astype(np.float32)
        raw_coordinates.append((record, coeff))
        coords_by_relation[record.relation].append(coeff)

    def _endpoint_calibration(
        negative: str,
        positive: str,
        dim: int,
    ) -> Tuple[float, float, float]:
        neg_mean = float(np.mean([coord[dim] for coord in coords_by_relation[negative]]))
        pos_mean = float(np.mean([coord[dim] for coord in coords_by_relation[positive]]))
        center = 0.5 * (neg_mean + pos_mean)
        half_margin = 0.5 * abs(pos_mean - neg_mean)
        if half_margin < 1e-8:
            raise RuntimeError(f"Layer {layer}: joint coordinate margin is near zero.")
        orientation = 1.0 if pos_mean >= neg_mean else -1.0
        return center, half_margin, orientation

    x_center, x_half_margin, x_orientation = _endpoint_calibration("left", "right", 0)
    y_center, y_half_margin, y_orientation = _endpoint_calibration("under", "on", 1)

    coordinate_rows: List[Tuple[int, str, float, float]] = []
    geometric_rows: List[Tuple[int, str, str]] = []
    route_horizontal: List[bool] = []
    route_vertical: List[bool] = []

    standardized: List[Tuple[SampleRecord, np.ndarray]] = []
    for record, coeff in raw_coordinates:
        zx = x_orientation * (float(coeff[0]) - x_center) / x_half_margin
        zy = y_orientation * (float(coeff[1]) - y_center) / y_half_margin
        z = np.array([zx, zy], dtype=np.float32)
        standardized.append((record, z))
        coordinate_rows.append((record.sid, record.relation, zx, zy))

        scores = {
            "left": -zx,
            "right": zx,
            "under": -zy,
            "on": zy,
        }
        predicted = max(scores, key=scores.get)
        geometric_rows.append((record.sid, record.relation, predicted))

        routed_horizontal = abs(zx) >= abs(zy)
        if record.relation in {"left", "right"}:
            route_horizontal.append(routed_horizontal)
        else:
            route_vertical.append(not routed_horizontal)

    geometric_summary, geometric_artifacts = _classification_payload(geometric_rows)
    geometric_summary.update(
        {
            "decoder": "shared_mean_dual_basis_cardinal_decoder",
            "description": (
                "Shared all-relation object baseline; dual-basis coordinates "
                "remove x/y cross-talk; endpoint-standardised coordinates then "
                "select the cardinal relation with the largest signed score."
            ),
            "horizontal_routing_accuracy": float(np.mean(route_horizontal)) if route_horizontal else None,
            "vertical_routing_accuracy": float(np.mean(route_vertical)) if route_vertical else None,
        }
    )

    # Four class prototypes in the calibrated 2D coordinate space.  This is
    # intentionally a diagnostic supervised readout, not the geometric result.
    z_by_relation: Dict[str, List[np.ndarray]] = defaultdict(list)
    for record, z in standardized:
        z_by_relation[record.relation].append(z)
    prototypes = {
        label: np.mean(np.stack(z_by_relation[label], axis=0), axis=0).astype(np.float64)
        for label in ("left", "right", "on", "under")
    }
    residuals = []
    for record, z in standardized:
        residuals.append(z.astype(np.float64) - prototypes[record.relation])
    covariance = np.cov(np.stack(residuals, axis=0), rowvar=False)
    if covariance.ndim == 0:
        covariance = np.eye(2, dtype=np.float64)
    regularizer = max(1e-5, 1e-3 * float(np.trace(covariance)) / 2.0)
    covariance = covariance + regularizer * np.eye(2, dtype=np.float64)
    covariance_inv = np.linalg.inv(covariance)

    prototype_rows: List[Tuple[int, str, str]] = []
    for record, z in standardized:
        z64 = z.astype(np.float64)
        distances = {
            label: float((z64 - proto).T @ covariance_inv @ (z64 - proto))
            for label, proto in prototypes.items()
        }
        predicted = min(distances, key=distances.get)
        prototype_rows.append((record.sid, record.relation, predicted))
    prototype_summary, prototype_artifacts = _classification_payload(prototype_rows)
    prototype_summary.update(
        {
            "decoder": "four_class_mahalanobis_prototype_readout",
            "description": (
                "A supervised diagnostic readout in calibrated 2D coordinate space. "
                "It allows each relation to have its own prototype and therefore "
                "does not test the stronger dominant-axis Cartesian assumption."
            ),
        }
    )


    # Constrained affine-axis decoder.
    #
    # A strict Cartesian readout assumes all four relation directions emanate
    # from one common origin.  The observed data can instead have two affine
    # relation families:
    #
    #   left/right : c_h + t * u_h
    #   under/on   : c_v + t * u_v
    #
    # where c_h and c_v need not coincide.  This decoder selects the nearer
    # affine line using the same pooled Mahalanobis metric as the prototype
    # diagnostic, then uses signed coordinate along that line for polarity.
    def _line_distance_and_coordinate(
        z: np.ndarray,
        center: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[float, float]:
        v = z.astype(np.float64) - center.astype(np.float64)
        u = direction.astype(np.float64)
        denom = float(u.T @ covariance_inv @ u)
        if denom < 1e-12:
            raise RuntimeError(f"Layer {layer}: affine-line direction has near-zero metric norm.")
        coord = float((v.T @ covariance_inv @ u) / denom)
        dist_sq = float(v.T @ covariance_inv @ v - ((v.T @ covariance_inv @ u) ** 2) / denom)
        return max(0.0, dist_sq), coord

    horizontal_center = 0.5 * (prototypes["left"] + prototypes["right"])
    vertical_center = 0.5 * (prototypes["under"] + prototypes["on"])
    horizontal_direction = prototypes["right"] - prototypes["left"]
    vertical_direction = prototypes["on"] - prototypes["under"]

    affine_rows: List[Tuple[int, str, str]] = []
    affine_route_horizontal: List[bool] = []
    affine_route_vertical: List[bool] = []
    affine_distances: List[Tuple[int, str, float, float, str]] = []

    for record, z in standardized:
        dist_h, coord_h = _line_distance_and_coordinate(z, horizontal_center, horizontal_direction)
        dist_v, coord_v = _line_distance_and_coordinate(z, vertical_center, vertical_direction)

        choose_horizontal = dist_h <= dist_v
        if choose_horizontal:
            predicted = "right" if coord_h >= 0.0 else "left"
        else:
            predicted = "on" if coord_v >= 0.0 else "under"

        affine_rows.append((record.sid, record.relation, predicted))
        affine_distances.append((record.sid, record.relation, dist_h, dist_v, predicted))

        if record.relation in {"left", "right"}:
            affine_route_horizontal.append(choose_horizontal)
        else:
            affine_route_vertical.append(not choose_horizontal)

    affine_summary, affine_artifacts = _classification_payload(affine_rows)
    affine_summary.update(
        {
            "decoder": "two_affine_relation_axes_mahalanobis",
            "description": (
                "Two relation families are modeled as separate affine lines: "
                "left/right share one centre and direction; under/on share another. "
                "Routing chooses the closer line under the pooled Mahalanobis metric, "
                "then polarity is decoded by the signed line coordinate. "
                "This is stronger than a four-prototype classifier but weaker than "
                "a single-origin Cartesian-coordinate claim."
            ),
            "horizontal_routing_accuracy": (
                float(np.mean(affine_route_horizontal)) if affine_route_horizontal else None
            ),
            "vertical_routing_accuracy": (
                float(np.mean(affine_route_vertical)) if affine_route_vertical else None
            ),
            "family_center_mahalanobis_distance": float(
                np.sqrt(
                    max(
                        0.0,
                        float(
                            (horizontal_center - vertical_center).T
                            @ covariance_inv
                            @ (horizontal_center - vertical_center)
                        ),
                    )
                )
            ),
        }
    )

    summary: Dict[str, Any] = {
        "layer": int(layer),
        "evaluation_n": int(len(standardized)),
        "joint_fit": {
            "objects_with_shared_baseline": int(len(object_means)),
            "x_axis_cosine_with_y": axis_cosine,
            "axis_gram_condition_number": gram_condition,
            "x_center": x_center,
            "x_half_margin": x_half_margin,
            "x_orientation": x_orientation,
            "y_center": y_center,
            "y_half_margin": y_half_margin,
            "y_orientation": y_orientation,
        },
        "geometric_cardinal_decoder": geometric_summary,
        "affine_axis_decoder": affine_summary,
        "prototype_diagnostic_decoder": prototype_summary,
        "note": (
            "With --fit-all these are full-data reconstruction scores. "
            "The geometric decoder tests a joint coordinate system; the "
            "prototype decoder is only a diagnostic for relation-cluster separability."
        ),
    }

    artifacts: Dict[str, np.ndarray] = {
        "joint_object_names": np.array(sorted(object_means), dtype=object),
        "joint_object_means": np.stack([object_means[name] for name in sorted(object_means)], axis=0).astype(np.float32),
        "joint_x_axis": x_axis.astype(np.float32),
        "joint_y_axis": y_axis.astype(np.float32),
        "joint_dual_basis": dual.astype(np.float32),
        "joint_gram": gram.astype(np.float32),
        "joint_relation_names": np.array(["left", "right", "on", "under"], dtype=object),
        "joint_relation_means": np.stack([relation_means[name] for name in ("left", "right", "on", "under")], axis=0).astype(np.float32),
        "joint_sid": np.array([sid for sid, _, _, _ in coordinate_rows], dtype=np.int64),
        "joint_ground_truth": np.array([gt for _, gt, _, _ in coordinate_rows], dtype=object),
        "joint_zx": np.array([zx for _, _, zx, _ in coordinate_rows], dtype=np.float32),
        "joint_zy": np.array([zy for _, _, _, zy in coordinate_rows], dtype=np.float32),
        "joint_prototype_names": np.array(["left", "right", "on", "under"], dtype=object),
        "joint_prototypes": np.stack([prototypes[name] for name in ("left", "right", "on", "under")], axis=0).astype(np.float32),
        "joint_prototype_covariance": covariance.astype(np.float32),
    }
    artifacts["joint_affine_horizontal_center"] = horizontal_center.astype(np.float32)
    artifacts["joint_affine_vertical_center"] = vertical_center.astype(np.float32)
    artifacts["joint_affine_horizontal_direction"] = horizontal_direction.astype(np.float32)
    artifacts["joint_affine_vertical_direction"] = vertical_direction.astype(np.float32)
    artifacts["joint_affine_sid"] = np.array([sid for sid, _, _, _, _ in affine_distances], dtype=np.int64)
    artifacts["joint_affine_ground_truth"] = np.array([gt for _, gt, _, _, _ in affine_distances], dtype=object)
    artifacts["joint_affine_distance_horizontal"] = np.array(
        [dist_h for _, _, dist_h, _, _ in affine_distances], dtype=np.float32
    )
    artifacts["joint_affine_distance_vertical"] = np.array(
        [dist_v for _, _, _, dist_v, _ in affine_distances], dtype=np.float32
    )
    artifacts["joint_affine_prediction"] = np.array(
        [pred for _, _, _, _, pred in affine_distances], dtype=object
    )
    for prefix, source in (
        ("geometric", geometric_artifacts),
        ("affine", affine_artifacts),
        ("prototype", prototype_artifacts),
    ):
        for name, value in source.items():
            artifacts[f"{prefix}_{name}"] = value
    return summary, artifacts

def load_model_and_processor(args: argparse.Namespace):
    dtype = resolve_dtype(args.dtype)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    model.eval().to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    # Older LLaVA-1.5 processor checkpoints omit these fields, while current
    # Transformers requires them to expand <image> into 576 CLIP patch tokens.
    vision_config = getattr(model.config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is None:
        raise RuntimeError("Could not recover the vision patch size from model.config.vision_config.")
    processor.patch_size = int(patch_size)
    processor.vision_feature_select_strategy = str(
        getattr(model.config, "vision_feature_select_strategy", "default")
    )
    # CLIP has one CLS token in addition to its 24x24 patch grid. Recent
    # LlavaProcessor subtracts it for the default feature-selection strategy:
    # 24*24 + 1 - 1 = 576, matching LLaVA-1.5 image_seq_length.
    processor.num_additional_image_tokens = 1
    return model, processor


@torch.inference_mode()
def extract_states_for_sample(
    *,
    model: LlavaForConditionalGeneration,
    processor,
    capture: SelectedTokenCapture,
    image: Any,
    prompt: str,
    meta: PromptMeta,
    layers: Sequence[int],
    device: str,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(device)
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=model_dtype)

    input_ids = [int(token) for token in inputs["input_ids"][0].detach().cpu().tolist()]
    image_token_id = int(model.config.image_token_index)
    image_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
    image_seq_length = int(getattr(model.config, "image_seq_length", 0))
    if image_seq_length <= 0:
        raise RuntimeError("model.config.image_seq_length is required to map text tokens after <image>.")

    subject_text_index = find_phrase_last_token(input_ids, processor.tokenizer, meta.subject)
    reference_text_index = find_phrase_last_token(input_ids, processor.tokenizer, meta.reference)

    if len(image_positions) == 1:
        image_layout = "single_placeholder_expanded_in_model"
        image_index = image_positions[0]
        subject_merged_index = map_text_index_to_merged_index(subject_text_index, image_index, image_seq_length)
        reference_merged_index = map_text_index_to_merged_index(reference_text_index, image_index, image_seq_length)
        expected_merged_length = len(input_ids) + image_seq_length - 1
    elif len(image_positions) >= image_seq_length:
        image_layout = "processor_expanded_image_placeholders"
        image_index = image_positions[0]
        subject_merged_index = subject_text_index
        reference_merged_index = reference_text_index
        expected_merged_length = len(input_ids)
    else:
        raise RuntimeError(
            f"Unsupported number of <image> placeholder tokens: {len(image_positions)}. "
            f"Expected 1 or at least image_seq_length={image_seq_length}."
        )

    capture.begin(subject_merged_index, reference_merged_index)
    outputs = model(
        **inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    actual_merged_length = int(outputs.logits.shape[1])
    if actual_merged_length != expected_merged_length:
        raise RuntimeError(
            "Unexpected merged sequence length: "
            f"expected {expected_merged_length}, got {actual_merged_length}. "
            "The image-token mapping may need updating for this Transformers version."
        )

    captured = capture.collect()
    states = {
        int(layer): {
            "subject": captured[layer]["subject"].numpy().astype(np.float32),
            "reference": captured[layer]["reference"].numpy().astype(np.float32),
        }
        for layer in layers
    }
    debug = {
        "input_length": len(input_ids),
        "merged_length": actual_merged_length,
        "image_text_index": image_index,
        "image_placeholder_count": len(image_positions),
        "image_token_layout": image_layout,
        "subject_text_index": subject_text_index,
        "reference_text_index": reference_text_index,
        "subject_merged_index": subject_merged_index,
        "reference_merged_index": reference_merged_index,
        "subject_final_token": processor.tokenizer.decode([input_ids[subject_text_index]], skip_special_tokens=False),
        "reference_final_token": processor.tokenizer.decode([input_ids[reference_text_index]], skip_special_tokens=False),
    }
    return states, debug


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    layers = parse_layers(args.layers)
    axes = parse_axes(args.axes)
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path("prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    prompt_rows = load_prompt_rows(prompt_path)
    model, processor = load_model_and_processor(args)
    print(f"Loaded {args.model}@{args.revision}; decoder blocks={len(model.language_model.model.layers)}")
    print(f"Fixed probe layers (zero-based block indices): {layers}")
    print(f"Axes: {axes}")
    print(f"Prompt mode: {args.prompt_mode}")
    if args.fit_all:
        print("Evaluation mode: FULL-DATA in-sample alignment (axis estimated and scored on the same samples).")
    else:
        print(f"Evaluation mode: held-out ({args.split_unit} split, test_fraction={args.test_fraction:.2f}).")

    dataset = get_dataset("Controlled_Images_A", image_preprocess=None, download=args.download)
    total_available = min(len(dataset), len(prompt_rows))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=repository_default_collate,
    )
    capture = SelectedTokenCapture(model.language_model, layers)
    records: List[SampleRecord] = []
    skipped: Counter[str] = Counter()
    debug_first: List[Dict[str, Any]] = []

    try:
        sid = 0
        progress = tqdm(total=total_available, desc="Extracting Controlled_A token states")
        for batch in loader:
            for image in extract_images_from_batch(batch):
                if sid >= total_available:
                    break
                row = prompt_rows[sid]
                relation = normalize_relation(row["answer"])
                if relation not in ALL_RELATIONS:
                    skipped[relation] += 1
                elif args.limit is not None and len(records) >= args.limit:
                    skipped["limit"] += 1
                else:
                    meta = parse_prompt(str(row["question"]))
                    prompt = make_clean_prompt(meta) if args.prompt_mode == "clean" else str(row["question"])
                    states, debug = extract_states_for_sample(
                        model=model,
                        processor=processor,
                        capture=capture,
                        image=image,
                        prompt=prompt,
                        meta=meta,
                        layers=layers,
                        device=args.device,
                    )
                    records.append(
                        SampleRecord(
                            sid=sid,
                            relation=relation,
                            subject=meta.subject,
                            reference=meta.reference,
                            prompt=prompt,
                            group=" || ".join(sorted((meta.subject, meta.reference))),
                            states=states,
                        )
                    )
                    if len(debug_first) < args.print_first:
                        debug_first.append(
                            {
                                "sid": sid,
                                "relation": relation,
                                "subject": meta.subject,
                                "reference": meta.reference,
                                "prompt": prompt,
                                **debug,
                            }
                        )
                sid += 1
                progress.update(1)
            if sid >= total_available:
                break
        progress.close()
    finally:
        capture.close()

    if len(records) < 20:
        raise RuntimeError(f"Only {len(records)} supported spatial samples were collected; expected many more.")

    per_layer: Dict[str, Dict[str, Any]] = {}
    artifacts: Dict[str, np.ndarray] = {}
    split_payload: Dict[str, Dict[str, List[int]]] = {}
    fitted_axis_artifacts: Dict[int, Dict[str, Dict[str, np.ndarray]]] = defaultdict(dict)
    four_way_results: Dict[str, Dict[str, Any]] = {}

    for axis_name in axes:
        spec = AXIS_SPECS[axis_name]
        axis_records = [record for record in records if record.relation in spec["relations"]]
        if len(axis_records) < 20:
            raise RuntimeError(f"Only {len(axis_records)} {axis_name} samples are available.")

        if args.fit_all:
            estimation_sids = {record.sid for record in axis_records}
            evaluation_sids = set(estimation_sids)
        else:
            estimation_sids, evaluation_sids = choose_group_split(
                axis_records,
                test_fraction=args.test_fraction,
                seed=args.seed,
                split_unit=args.split_unit,
                required_relations=set(spec["relations"]),
            )
        split_payload[axis_name] = {
            "estimation_sids": sorted(estimation_sids),
            "evaluation_sids": sorted(evaluation_sids),
        }

        for layer in layers:
            summary, layer_artifacts = score_axis_layer(
                records,
                layer=layer,
                axis_name=axis_name,
                estimation_sids=estimation_sids,
                evaluation_sids=evaluation_sids,
            )
            per_layer.setdefault(str(layer), {})[axis_name] = summary
            fitted_axis_artifacts[layer][axis_name] = layer_artifacts
            for name, value in layer_artifacts.items():
                artifacts[f"layer_{layer}_{axis_name}_{name}"] = value

    # A joint four-way decoder is only meaningful in full-data mode here,
    # because it needs one common estimation pool for both coordinate axes.
    if args.fit_all and {"horizontal", "vertical"}.issubset(set(axes)):
        for layer in layers:
            four_summary, four_artifacts = score_four_way_joint_coordinate_decoder(
                records,
                layer=layer,
            )
            four_way_results[str(layer)] = four_summary
            for name, value in four_artifacts.items():
                artifacts[f"layer_{layer}_{name}"] = value

    payload = {
        "metadata": {
            "model": args.model,
            "revision": args.revision,
            "dataset": "Controlled_Images_A",
            "prompt_file": str(prompt_path),
            "prompt_mode": args.prompt_mode,
            "relations": sorted(ALL_RELATIONS),
            "axes": axes,
            "layers_zero_based": layers,
            "dtype": args.dtype,
            "seed": args.seed,
            "fit_all": bool(args.fit_all),
            "split_unit": None if args.fit_all else args.split_unit,
            "test_fraction": None if args.fit_all else args.test_fraction,
            "interpretation": (
                "Separate-axis metrics use object-centred residual means within each axis family. "
                "The joint four-way analysis instead uses one all-relation object baseline and "
                "dual-basis coordinates so horizontal/vertical evidence is comparable. "
                "With --fit-all, all reported accuracy is full-data descriptive alignment, not held-out generalization."
            ),
        },
        "collection": {
            "total_prompt_rows": len(prompt_rows),
            "total_dataset_items_seen": total_available,
            "supported_spatial_samples_used": len(records),
            "skipped": dict(skipped),
            "relation_counts": dict(Counter(record.relation for record in records)),
            "unique_unordered_pairs": len({record.group for record in records}),
            "splits": split_payload,
        },
        "debug_first_samples": debug_first,
        "results": per_layer,
        "joint_four_way_coordinate_decoder": four_way_results,
    }

    json_path = output_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(value: Optional[float]) -> str:
        return "NA" if value is None else f"{value:.3f}"

    print("\nResults")
    for layer in layers:
        print(f"  L{layer}:")
        for axis_name in axes:
            result = per_layer[str(layer)][axis_name]
            print(
                f"    {axis_name:<10} occurrence={fmt(result['occurrence_accuracy'])} | "
                f"subject={fmt(result['subject_position_accuracy'])} | "
                f"reference={fmt(result['reference_position_accuracy'])} | "
                f"pairwise={fmt(result['pairwise_relation_accuracy'])} "
                f"(occ n={result['usable_evaluation_occurrences']}, pair n={result['pair_evaluation_n']}, "
                f"objects={result['objects_with_balanced_estimation_positions']}, "
                f"mean object cosine={fmt(result['object_direction_cosine_mean'])})"
            )
        if str(layer) in four_way_results:
            result = four_way_results[str(layer)]
            joint = result["joint_fit"]
            geometric = result["geometric_cardinal_decoder"]
            affine = result["affine_axis_decoder"]
            prototype = result["prototype_diagnostic_decoder"]
            print(
                f"    joint-coordinate geometric acc={fmt(geometric['accuracy'])} "
                f"(n={geometric['evaluation_n']}; route h={fmt(geometric['horizontal_routing_accuracy'])}, "
                f"v={fmt(geometric['vertical_routing_accuracy'])}; "
                f"cos(x,y)={fmt(joint['x_axis_cosine_with_y'])}, "
                f"cond={joint['axis_gram_condition_number']:.2f})"
            )
            print(
                f"    two-affine-axis acc={fmt(affine['accuracy'])} "
                f"(route h={fmt(affine['horizontal_routing_accuracy'])}, "
                f"v={fmt(affine['vertical_routing_accuracy'])}; "
                f"family-centre Mahalanobis dist={affine['family_center_mahalanobis_distance']:.2f})"
            )
            print(
                f"    joint-coordinate prototype acc={fmt(prototype['accuracy'])} "
                f"(diagnostic only; not a Cartesian-coordinate claim)"
            )
    print(f"Saved summary: {json_path}")

    if args.save_states:
        ordered = sorted(records, key=lambda item: item.sid)
        arrays: Dict[str, np.ndarray] = {
            "sid": np.array([record.sid for record in ordered], dtype=np.int64),
            "relation": np.array([record.relation for record in ordered], dtype=object),
            "subject_name": np.array([record.subject for record in ordered], dtype=object),
            "reference_name": np.array([record.reference for record in ordered], dtype=object),
        }
        for axis_name in axes:
            estimate_ids = set(split_payload[axis_name]["estimation_sids"])
            evaluation_ids = set(split_payload[axis_name]["evaluation_sids"])
            arrays[f"{axis_name}_is_estimation"] = np.array(
                [record.sid in estimate_ids for record in ordered], dtype=np.bool_
            )
            arrays[f"{axis_name}_is_evaluation"] = np.array(
                [record.sid in evaluation_ids for record in ordered], dtype=np.bool_
            )
        for layer in layers:
            arrays[f"layer_{layer}_subject"] = np.stack(
                [record.states[layer]["subject"] for record in ordered], axis=0
            ).astype(np.float32)
            arrays[f"layer_{layer}_reference"] = np.stack(
                [record.states[layer]["reference"] for record in ordered], axis=0
            ).astype(np.float32)
        arrays.update(artifacts)
        npz_path = output_base.with_suffix(".npz")
        np.savez_compressed(npz_path, **arrays)
        print(f"Saved states/axes: {npz_path}")


if __name__ == "__main__":
    main()
