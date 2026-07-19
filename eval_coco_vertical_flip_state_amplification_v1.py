#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate counterfactual hidden-state amplification on COCO-two above/below.

This is a performance experiment built on the preceding image-flip activation
patching result.

For every above/below sample, the script runs the same question with:

    O: original image
    F: horizontally flipped image

The text tokens and their sequence positions are identical.  At a selected
decoder-block output and selected TEXT-token positions, it constructs:

    original branch:
        h'_O = h_O + gamma * (h_O - h_F)

    flipped branch:
        h'_F = h_F + gamma * (h_F - h_O)

Thus the original/flip counterfactual difference is amplified without adding a
above/below LM-head direction, using a centroid, training a probe, or changing
model weights.

The script evaluates:

1. original baseline;
2. horizontally flipped prediction mapped back to original coordinates;
3. original + mapped-flip logit ensemble;
4. confidence selection between the two baseline branches;
5. original amplified branch;
6. mapped flipped amplified branch;
7. amplified two-branch ensemble;
8. amplified confidence selection;
9. disagreement-gated amplification.

Important:
- Only above/below records are evaluated.
- Vertical flipping is fixed for the entire run; ground truth is NOT used to
  choose a transformation axis.
- Ground truth is used only for final evaluation metrics.
- Interventions patch block-output residual states at text-token positions such
  as subject/reference/both/prompt_last.  Visual tokens are not replaced.
- This version evaluates first-answer-token relation logits, not free generation.
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


VERSION = "coco-vertical-flip-state-amplification-v1"
RELATIONS = ("left", "right", "above", "below")
VERTICAL_MAP = {
    "left": "left",
    "right": "right",
    "above": "below",
    "below": "above",
}
ALLOWED_GROUPS = ("subject", "reference", "both", "prompt_last")


@dataclass(frozen=True)
class BaseCondition:
    layer: int
    token_group: str


@dataclass
class RuntimeCondition:
    layer: int
    token_group: str
    gamma: float
    control: bool
    positions: List[int]

    @property
    def condition_id(self) -> str:
        suffix = "_random_text" if self.control else ""
        gamma_text = f"{self.gamma:g}"
        return f"L{self.layer:02d}_{self.token_group}{suffix}_g{gamma_text}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
        help="Existing COCO/model helper script.",
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument(
        "--conditions",
        default="12:both,15:prompt_last",
        help=(
            "Comma-separated block-output intervention points, for example "
            "'10:both,12:both,15:prompt_last,16:both'."
        ),
    )
    p.add_argument(
        "--gammas",
        default="0.1,0.25,0.5",
        help="Comma-separated nonnegative amplification coefficients.",
    )
    p.add_argument(
        "--include-random-control",
        action="store_true",
        help=(
            "Add matched-count random non-object text-token controls for every "
            "layer/group/gamma condition."
        ),
    )
    p.add_argument(
        "--condition-batch-size",
        type=int,
        default=6,
        help="Number of intervention conditions evaluated in one model batch.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing helper script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gammas(value: str) -> List[float]:
    out: List[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        gamma = float(raw)
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError(f"Invalid gamma: {raw!r}")
        if gamma not in out:
            out.append(gamma)
    if not out:
        raise ValueError("--gammas produced no values")
    return out


def parse_condition_specs(
    value: str,
    n_layers: int,
) -> List[BaseCondition]:
    out: List[BaseCondition] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(
                f"Invalid condition {raw!r}; expected layer:token_group"
            )
        layer_raw, group = raw.split(":", 1)
        layer = int(layer_raw.strip())
        group = group.strip()
        if not 0 <= layer < n_layers:
            raise ValueError(
                f"Condition layer {layer} outside 0..{n_layers - 1}"
            )
        if group not in ALLOWED_GROUPS:
            raise ValueError(
                f"Unsupported token group {group!r}; "
                f"allowed={ALLOWED_GROUPS}"
            )
        condition = BaseCondition(layer=layer, token_group=group)
        if condition not in out:
            out.append(condition)
    if not out:
        raise ValueError("--conditions produced no conditions")
    return out


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        f"Unsupported decoder-block output: {type(output).__name__}"
    )


def replace_first(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    raise TypeError(type(output).__name__)


class CaptureBlockOutputs:
    def __init__(
        self,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer_indices = sorted(set(map(int, layer_indices)))
        self.handles: List[Any] = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "CaptureBlockOutputs":
        for layer_index in self.layer_indices:
            def make_hook(index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    self.outputs[index] = first_tensor(output).detach()
                return hook

            self.handles.append(
                self.decoder_layers[layer_index].register_forward_hook(
                    make_hook(layer_index)
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


class SymmetricAmplificationHooks:
    """Patch one distinct block output per batch row.

    Each runtime condition identifies one layer, one text-position set, and one
    gamma.  For the original branch:

        target = h_original + gamma * (h_original - h_flip)

    For the flipped branch:

        target = h_flip + gamma * (h_flip - h_original)
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        original_capture: Mapping[int, torch.Tensor],
        flipped_capture: Mapping[int, torch.Tensor],
        conditions: Sequence[RuntimeCondition],
        branch: str,
        sequence_length: int,
    ) -> None:
        if branch not in ("original", "flipped"):
            raise ValueError(branch)
        self.decoder_layers = decoder_layers
        self.original_capture = original_capture
        self.flipped_capture = flipped_capture
        self.conditions = list(conditions)
        self.branch = branch
        self.sequence_length = int(sequence_length)
        self.handles: List[Any] = []
        self.patch_counts = [0 for _ in self.conditions]

    def __enter__(self) -> "SymmetricAmplificationHooks":
        selected_layers = sorted({condition.layer for condition in self.conditions})

        for layer_index in selected_layers:
            def make_hook(index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    hidden = first_tensor(output)
                    if hidden.ndim != 3:
                        raise RuntimeError(
                            f"Expected [B,S,H], got {tuple(hidden.shape)}"
                        )
                    batch, seq_len, hidden_size = hidden.shape
                    if batch != len(self.conditions):
                        raise RuntimeError(
                            f"Condition batch mismatch: {batch} != "
                            f"{len(self.conditions)}"
                        )
                    if seq_len != self.sequence_length:
                        raise RuntimeError(
                            f"Sequence mismatch: {seq_len} != "
                            f"{self.sequence_length}"
                        )

                    original = self.original_capture[index]
                    flipped = self.flipped_capture[index]
                    if tuple(original.shape) != (1, seq_len, hidden_size):
                        raise RuntimeError(
                            f"Original capture shape mismatch at L{index}: "
                            f"{tuple(original.shape)}"
                        )
                    if tuple(flipped.shape) != (1, seq_len, hidden_size):
                        raise RuntimeError(
                            f"Flipped capture shape mismatch at L{index}: "
                            f"{tuple(flipped.shape)}"
                        )

                    original = original.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    flipped = flipped.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    if self.branch == "original":
                        base = original
                        other = flipped
                    else:
                        base = flipped
                        other = original

                    patched = hidden.clone()
                    for row_index, condition in enumerate(self.conditions):
                        if condition.layer != index:
                            continue
                        positions = [
                            pos for pos in condition.positions
                            if 0 <= int(pos) < seq_len
                        ]
                        if not positions:
                            raise RuntimeError(
                                f"Empty positions for {condition.condition_id}"
                            )
                        idx = torch.as_tensor(
                            positions,
                            device=hidden.device,
                            dtype=torch.long,
                        )
                        base_values = base[0].index_select(0, idx)
                        other_values = other[0].index_select(0, idx)
                        target = (
                            base_values
                            + float(condition.gamma)
                            * (base_values - other_values)
                        )
                        patched[row_index].index_copy_(0, idx, target)
                        self.patch_counts[row_index] += 1
                    return replace_first(output, patched)

                return hook

            self.handles.append(
                self.decoder_layers[layer_index].register_forward_hook(
                    make_hook(layer_index)
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "logits",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "logits",
            None,
        ),
    ]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("No language-model logits found")


def relation_scores(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for relation in RELATIONS:
        token_ids = [
            int(token_id)
            for token_id in token_map[relation]
            if 0 <= int(token_id) < int(logits.numel())
        ]
        if not token_ids:
            raise RuntimeError(f"No relation-token IDs for {relation}")
        idx = torch.as_tensor(
            token_ids,
            device=logits.device,
            dtype=torch.long,
        )
        scores[relation] = float(
            logits.index_select(0, idx).max().detach().cpu()
        )
    return scores


def prediction(scores: Mapping[str, float]) -> str:
    return max(RELATIONS, key=lambda relation: float(scores[relation]))


def run_forward(
    *,
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    capture_layers: Sequence[int] = (),
) -> Tuple[List[Dict[str, Any]], Dict[int, torch.Tensor]]:
    with torch.inference_mode():
        if capture_layers:
            with CaptureBlockOutputs(
                decoder_layers,
                capture_layers,
            ) as capture:
                outputs = model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            captured = dict(capture.outputs)
        else:
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            captured = {}

    logits = extract_logits(outputs)[:, -1, :]
    results: List[Dict[str, Any]] = []
    for batch_index in range(int(logits.shape[0])):
        scores = relation_scores(logits[batch_index], token_map)
        results.append({
            "scores": scores,
            "prediction": prediction(scores),
        })
    del outputs, logits
    return results, captured


def map_vertical_scores(
    flipped_scores: Mapping[str, float],
) -> Dict[str, float]:
    return {
        relation: float(flipped_scores[VERTICAL_MAP[relation]])
        for relation in RELATIONS
    }


def mean_scores(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> Dict[str, float]:
    return {
        relation: 0.5 * (
            float(first[relation]) + float(second[relation])
        )
        for relation in RELATIONS
    }


def top_margin(scores: Mapping[str, float]) -> float:
    values = sorted(
        (float(scores[relation]) for relation in RELATIONS),
        reverse=True,
    )
    return values[0] - values[1]


def confidence_select_scores(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> Dict[str, float]:
    return dict(first if top_margin(first) >= top_margin(second) else second)


def gt_margin(
    scores: Mapping[str, float],
    ground_truth: str,
) -> float:
    competitors = [
        float(scores[relation])
        for relation in RELATIONS
        if relation != ground_truth
    ]
    return float(scores[ground_truth]) - max(competitors)


def repeated_batch(
    *,
    processor: Any,
    rendered_prompt: str,
    image: Image.Image,
    repeats: int,
    device: torch.device,
    base: Any,
) -> Dict[str, Any]:
    batch = processor(
        text=[rendered_prompt] * int(repeats),
        images=[image] * int(repeats),
        return_tensors="pt",
        padding=True,
    )
    return base.move_batch(batch, device)


def span_positions(span: Tuple[int, int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def choose_random_positions(
    *,
    candidates: Sequence[int],
    count: int,
    seed: int,
) -> List[int]:
    candidates = sorted(set(map(int, candidates)))
    if count <= 0:
        raise ValueError("Random control requires a positive count")
    if len(candidates) < count:
        raise RuntimeError(
            f"Need {count} random text tokens; found {len(candidates)}"
        )
    rng = random.Random(int(seed))
    return sorted(rng.sample(candidates, count))


def build_runtime_conditions(
    *,
    base_conditions: Sequence[BaseCondition],
    gammas: Sequence[float],
    include_random_control: bool,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
    text_positions: Sequence[int],
    sid: int,
    seed: int,
) -> List[RuntimeCondition]:
    subject = sorted(set(map(int, subject_positions)))
    reference = sorted(set(map(int, reference_positions)))
    both = sorted(set(subject + reference))
    actual_positions = {
        "subject": subject,
        "reference": reference,
        "both": both,
        "prompt_last": [int(prompt_last)],
    }

    excluded = set(both)
    excluded.add(int(prompt_last))
    random_candidates = [
        int(pos)
        for pos in text_positions
        if int(pos) not in excluded
    ]

    conditions: List[RuntimeCondition] = []
    for condition_index, base_condition in enumerate(base_conditions):
        positions = list(actual_positions[base_condition.token_group])
        if not positions:
            raise RuntimeError(
                f"Empty token group {base_condition.token_group}"
            )

        random_positions: Optional[List[int]] = None
        if include_random_control:
            group_code = sum(ord(ch) for ch in base_condition.token_group)
            random_positions = choose_random_positions(
                candidates=random_candidates,
                count=len(positions),
                seed=(
                    int(seed) * 1000003
                    + int(sid) * 1009
                    + int(base_condition.layer) * 97
                    + int(condition_index) * 53
                    + group_code
                ),
            )

        for gamma in gammas:
            conditions.append(RuntimeCondition(
                layer=base_condition.layer,
                token_group=base_condition.token_group,
                gamma=float(gamma),
                control=False,
                positions=list(positions),
            ))
            if random_positions is not None:
                conditions.append(RuntimeCondition(
                    layer=base_condition.layer,
                    token_group=base_condition.token_group,
                    gamma=float(gamma),
                    control=True,
                    positions=list(random_positions),
                ))
    return conditions


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_method_row(
    *,
    path: Path,
    sid: int,
    ground_truth: str,
    baseline_prediction: str,
    method: str,
    scores: Mapping[str, float],
    layer: Optional[int] = None,
    token_group: Optional[str] = None,
    gamma: Optional[float] = None,
    control: bool = False,
    condition_id: str = "baseline",
    baseline_disagreement: bool = False,
    branch_agreement: Optional[bool] = None,
) -> None:
    pred = prediction(scores)
    baseline_correct = baseline_prediction == ground_truth
    correct = pred == ground_truth
    append_jsonl(path, {
        "sid": int(sid),
        "ground_truth": ground_truth,
        "method": method,
        "condition_id": condition_id,
        "layer": layer,
        "token_group": token_group,
        "gamma": gamma,
        "control": bool(control),
        "prediction": pred,
        "correct": bool(correct),
        "baseline_prediction": baseline_prediction,
        "baseline_correct": bool(baseline_correct),
        "repaired": bool((not baseline_correct) and correct),
        "damaged": bool(baseline_correct and (not correct)),
        "changed": bool(pred != baseline_prediction),
        "baseline_disagreement": bool(baseline_disagreement),
        "branch_agreement": branch_agreement,
        "gt_margin": gt_margin(scores, ground_truth),
        "scores": dict(scores),
    })


def summarize_method_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["condition_id"],
            row["method"],
            row.get("layer"),
            row.get("token_group"),
            row.get("gamma"),
            bool(row.get("control", False)),
        )
        grouped[key].append(row)

    summary: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        condition_id, method, layer, token_group, gamma, control = key
        n = len(items)
        repaired = sum(bool(item["repaired"]) for item in items)
        damaged = sum(bool(item["damaged"]) for item in items)
        agreement_values = [
            bool(item["branch_agreement"])
            for item in items
            if item.get("branch_agreement") is not None
        ]
        disagreement_items = [
            item for item in items
            if bool(item.get("baseline_disagreement"))
        ]
        summary.append({
            "condition_id": condition_id,
            "method": method,
            "layer": layer,
            "token_group": token_group,
            "gamma": gamma,
            "control": control,
            "n": n,
            "accuracy": float(np.mean([
                bool(item["correct"]) for item in items
            ])),
            "repaired": repaired,
            "damaged": damaged,
            "net_gain_count": repaired - damaged,
            "changed": sum(bool(item["changed"]) for item in items),
            "mean_gt_margin": float(np.mean([
                float(item["gt_margin"]) for item in items
            ])),
            "branch_agreement_rate": (
                float(np.mean(agreement_values))
                if agreement_values else None
            ),
            "baseline_disagreement_n": len(disagreement_items),
            "accuracy_on_baseline_disagreement": (
                float(np.mean([
                    bool(item["correct"]) for item in disagreement_items
                ]))
                if disagreement_items else None
            ),
        })
    return sorted(
        summary,
        key=lambda row: (
            bool(row["control"]),
            str(row["condition_id"]),
            str(row["method"]),
        ),
    )


def report_text(
    *,
    model_name: str,
    n_samples: int,
    baseline_disagreement_count: int,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    baseline_order = {
        "baseline_original": 0,
        "baseline_flip_mapped": 1,
        "baseline_flip_ensemble": 2,
        "baseline_confidence_select": 3,
    }
    baseline = [
        row for row in rows
        if row["condition_id"] == "baseline"
    ]
    baseline.sort(key=lambda row: baseline_order.get(row["method"], 99))

    interventions = [
        row for row in rows
        if row["condition_id"] != "baseline"
        and not bool(row["control"])
        and row["method"] in (
            "original_amplified",
            "amplified_ensemble",
            "amplified_confidence_select",
            "disagreement_gate_else_original",
            "disagreement_gate_else_base_ensemble",
        )
    ]
    interventions.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            -int(row["net_gain_count"]),
            str(row["condition_id"]),
            str(row["method"]),
        )
    )

    controls = [
        row for row in rows
        if row["condition_id"] != "baseline"
        and bool(row["control"])
        and row["method"] == "amplified_ensemble"
    ]
    controls.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            str(row["condition_id"]),
        )
    )

    header = (
        f"{'Condition':<30}{'Method':<36}{'N':>6}"
        f"{'Acc':>9}{'Repair':>8}{'Damage':>8}{'Net':>7}"
        f"{'Changed':>9}{'GT margin':>11}{'Agree':>9}"
    )

    def f4(value: Any) -> str:
        try:
            if value is None:
                return "-"
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "-"

    lines = [
        "=" * len(header),
        "COCO ABOVE/BELOW COUNTERFACTUAL STATE AMPLIFICATION",
        (
            f"model={model_name} | n={n_samples} | "
            f"baseline branch disagreements={baseline_disagreement_count}"
        ),
        "=" * len(header),
        "BASELINES",
        header,
        "-" * len(header),
    ]

    for row in baseline:
        lines.append(
            f"{str(row['condition_id']):<30}"
            f"{str(row['method']):<36}"
            f"{int(row['n']):>6}"
            f"{f4(row['accuracy']):>9}"
            f"{int(row['repaired']):>8}"
            f"{int(row['damaged']):>8}"
            f"{int(row['net_gain_count']):>7}"
            f"{int(row['changed']):>9}"
            f"{f4(row['mean_gt_margin']):>11}"
            f"{f4(row['branch_agreement_rate']):>9}"
        )

    lines.extend([
        "",
        "TOP INTERNAL INTERVENTIONS",
        header,
        "-" * len(header),
    ])
    for row in interventions[:60]:
        lines.append(
            f"{str(row['condition_id']):<30}"
            f"{str(row['method']):<36}"
            f"{int(row['n']):>6}"
            f"{f4(row['accuracy']):>9}"
            f"{int(row['repaired']):>8}"
            f"{int(row['damaged']):>8}"
            f"{int(row['net_gain_count']):>7}"
            f"{int(row['changed']):>9}"
            f"{f4(row['mean_gt_margin']):>11}"
            f"{f4(row['branch_agreement_rate']):>9}"
        )

    if controls:
        lines.extend([
            "",
            "RANDOM-TEXT CONTROLS — AMPLIFIED ENSEMBLE",
            header,
            "-" * len(header),
        ])
        for row in controls[:30]:
            lines.append(
                f"{str(row['condition_id']):<30}"
                f"{str(row['method']):<36}"
                f"{int(row['n']):>6}"
                f"{f4(row['accuracy']):>9}"
                f"{int(row['repaired']):>8}"
                f"{int(row['damaged']):>8}"
                f"{int(row['net_gain_count']):>7}"
                f"{int(row['changed']):>9}"
                f"{f4(row['mean_gt_margin']):>11}"
                f"{f4(row['branch_agreement_rate']):>9}"
            )

    lines.extend([
        "",
        "METHOD DEFINITIONS",
        "- baseline_original: normal original-image prediction.",
        "- baseline_flip_mapped: horizontal-flip prediction mapped back "
        "(above↔below).",
        "- baseline_flip_ensemble: mean of original and mapped-flip relation logits.",
        "- original_amplified: only the original branch receives hO + γ(hO-hF).",
        "- amplified_ensemble: mean of amplified original and mapped amplified-flip logits.",
        "- disagreement_gate_else_original: use amplified ensemble only when "
        "the two baseline branches disagree; otherwise keep original baseline.",
        "- disagreement_gate_else_base_ensemble: use amplified ensemble only "
        "on disagreement; otherwise use the ordinary flip ensemble.",
        "",
        "A useful internal intervention should exceed baseline_flip_ensemble, "
        "have Repair > Damage, and outperform the matched random-text control.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.condition_batch_size <= 0:
        raise ValueError("--condition-batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = import_file(Path(args.base_script), "_coco_amp_base")
    data_module = base.import_two_object_module()
    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    prompt_rows = base.load_standard_prompts(Path(args.prompt_jsonl))

    specs = base.merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model!r}; available={sorted(specs)}"
        )
    model_spec = specs[args.model]

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    method_path = output_dir / "method_results.jsonl"
    baseline_path = output_dir / "baseline_pairs.jsonl"
    error_path = output_dir / "errors.jsonl"

    model_cls = getattr(transformers, model_spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"does not expose {model_spec.model_class}"
        )
    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(model_spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": model_spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}")
    print(f"Loading {args.model}: {model_spec.repo_id}")
    model = model_cls.from_pretrained(
        model_spec.repo_id,
        **load_kwargs,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_spec.repo_id,
        trust_remote_code=model_spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    base_conditions = parse_condition_specs(
        args.conditions,
        len(decoder_layers),
    )
    capture_layers = sorted({
        condition.layer for condition in base_conditions
    })
    gammas = parse_gammas(args.gammas)
    token_map = base.relation_token_variants(processor.tokenizer)

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": model_spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "conditions": [
            {
                "layer": condition.layer,
                "token_group": condition.token_group,
            }
            for condition in base_conditions
        ],
        "gammas": gammas,
        "include_random_control": args.include_random_control,
        "condition_batch_size": args.condition_batch_size,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "audit": audit,
        "transform": "vertical_flip_for_all_evaluated_samples",
        "evaluated_relations": ["above", "below"],
        "patch_location": "decoder_block_output",
        "patches_visual_tokens": False,
        "uses_ground_truth_to_choose_transform": False,
        "uses_external_model": False,
        "uses_centroid_prediction": False,
        "updates_model_weights": False,
        "evaluation": "first_answer_token_relation_logits",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Decoder={decoder_path}; n_layers={len(decoder_layers)}; "
        f"capture_layers={capture_layers}"
    )
    print(
        "Conditions:",
        ", ".join(
            f"L{condition.layer}:{condition.token_group}"
            for condition in base_conditions
        ),
    )
    print("Gammas:", gammas)
    print("Random control:", args.include_random_control)

    evaluated = 0
    baseline_disagreement_count = 0
    start_time = time.time()
    counts: Counter = Counter()

    try:
        for record in tqdm(
            records,
            desc=f"flip-state-amplification:{args.model}",
        ):
            sid = int(record.sid)
            original_batch: Optional[Dict[str, Any]] = None
            flipped_batch: Optional[Dict[str, Any]] = None
            original_image: Optional[Image.Image] = None
            flipped_image: Optional[Image.Image] = None

            try:
                if sid not in prompt_rows:
                    raise KeyError(f"Missing prompt row for sid={sid}")
                prompt_row = prompt_rows[sid]
                ground_truth = base.normalize_relation(
                    prompt_row["answer_raw"]
                )
                ground_truth = {
                    "on": "above",
                    "under": "below",
                    "over": "above",
                    "beneath": "below",
                }.get(ground_truth, ground_truth)
                if ground_truth not in ("above", "below"):
                    continue

                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])

                original_image = base.record_image(record).convert("RGB")
                flipped_image = original_image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )
                rendered_prompt = base.build_prompt(processor, question)

                original_batch = base.move_batch(
                    processor(
                        text=[rendered_prompt],
                        images=[original_image],
                        return_tensors="pt",
                    ),
                    device,
                )
                flipped_batch = base.move_batch(
                    processor(
                        text=[rendered_prompt],
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
                        "Original/flip input_ids differ"
                    )

                subject_span, reference_span = base.locate_object_spans(
                    processor.tokenizer,
                    original_ids,
                    subject,
                    reference,
                )
                subject_positions = span_positions(subject_span)
                reference_positions = span_positions(reference_span)
                prompt_last = len(original_ids) - 1
                visual_indices = base.resolve_visual_indices(
                    model,
                    processor,
                    original_batch,
                    original_ids,
                )
                visual_set = set(map(int, visual_indices))
                text_positions = [
                    index
                    for index in range(len(original_ids))
                    if index not in visual_set
                ]

                original_results, original_capture = run_forward(
                    model=model,
                    batch=original_batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    capture_layers=capture_layers,
                )
                flipped_results, flipped_capture = run_forward(
                    model=model,
                    batch=flipped_batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    capture_layers=capture_layers,
                )
                original_result = original_results[0]
                flipped_result = flipped_results[0]

                original_scores = dict(original_result["scores"])
                mapped_flip_scores = map_vertical_scores(
                    flipped_result["scores"]
                )
                base_ensemble_scores = mean_scores(
                    original_scores,
                    mapped_flip_scores,
                )
                base_confidence_scores = confidence_select_scores(
                    original_scores,
                    mapped_flip_scores,
                )

                original_prediction = prediction(original_scores)
                mapped_flip_prediction = prediction(mapped_flip_scores)
                baseline_disagreement = (
                    original_prediction != mapped_flip_prediction
                )
                baseline_disagreement_count += int(
                    baseline_disagreement
                )
                evaluated += 1

                flipped_ground_truth = VERTICAL_MAP[ground_truth]
                counts["original_correct"] += int(
                    original_prediction == ground_truth
                )
                counts["flipped_correct_before_mapping"] += int(
                    flipped_result["prediction"] == flipped_ground_truth
                )
                counts["mapped_flip_correct"] += int(
                    mapped_flip_prediction == ground_truth
                )
                counts["baseline_branch_agreement"] += int(
                    not baseline_disagreement
                )

                append_jsonl(baseline_path, {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "question": question,
                    "ground_truth": ground_truth,
                    "flipped_ground_truth": flipped_ground_truth,
                    "subject_span": list(subject_span),
                    "reference_span": list(reference_span),
                    "prompt_last": prompt_last,
                    "original_prediction": original_prediction,
                    "flipped_prediction": flipped_result["prediction"],
                    "mapped_flip_prediction": mapped_flip_prediction,
                    "baseline_disagreement": baseline_disagreement,
                    "original_scores": original_scores,
                    "flipped_scores": flipped_result["scores"],
                    "mapped_flip_scores": mapped_flip_scores,
                })

                add_method_row(
                    path=method_path,
                    sid=sid,
                    ground_truth=ground_truth,
                    baseline_prediction=original_prediction,
                    method="baseline_original",
                    scores=original_scores,
                    baseline_disagreement=baseline_disagreement,
                )
                add_method_row(
                    path=method_path,
                    sid=sid,
                    ground_truth=ground_truth,
                    baseline_prediction=original_prediction,
                    method="baseline_flip_mapped",
                    scores=mapped_flip_scores,
                    baseline_disagreement=baseline_disagreement,
                )
                add_method_row(
                    path=method_path,
                    sid=sid,
                    ground_truth=ground_truth,
                    baseline_prediction=original_prediction,
                    method="baseline_flip_ensemble",
                    scores=base_ensemble_scores,
                    baseline_disagreement=baseline_disagreement,
                    branch_agreement=not baseline_disagreement,
                )
                add_method_row(
                    path=method_path,
                    sid=sid,
                    ground_truth=ground_truth,
                    baseline_prediction=original_prediction,
                    method="baseline_confidence_select",
                    scores=base_confidence_scores,
                    baseline_disagreement=baseline_disagreement,
                    branch_agreement=not baseline_disagreement,
                )

                runtime_conditions = build_runtime_conditions(
                    base_conditions=base_conditions,
                    gammas=gammas,
                    include_random_control=args.include_random_control,
                    subject_positions=subject_positions,
                    reference_positions=reference_positions,
                    prompt_last=prompt_last,
                    text_positions=text_positions,
                    sid=sid,
                    seed=args.seed,
                )

                for chunk_start in range(
                    0,
                    len(runtime_conditions),
                    args.condition_batch_size,
                ):
                    chunk = runtime_conditions[
                        chunk_start:
                        chunk_start + args.condition_batch_size
                    ]
                    repeats = len(chunk)

                    repeated_original = repeated_batch(
                        processor=processor,
                        rendered_prompt=rendered_prompt,
                        image=original_image,
                        repeats=repeats,
                        device=device,
                        base=base,
                    )
                    repeated_flipped = repeated_batch(
                        processor=processor,
                        rendered_prompt=rendered_prompt,
                        image=flipped_image,
                        repeats=repeats,
                        device=device,
                        base=base,
                    )
                    if (
                        int(repeated_original["input_ids"].shape[1])
                        != len(original_ids)
                    ):
                        raise RuntimeError(
                            "Repeated original sequence length differs"
                        )
                    if (
                        int(repeated_flipped["input_ids"].shape[1])
                        != len(original_ids)
                    ):
                        raise RuntimeError(
                            "Repeated flipped sequence length differs"
                        )

                    with SymmetricAmplificationHooks(
                        decoder_layers=decoder_layers,
                        original_capture=original_capture,
                        flipped_capture=flipped_capture,
                        conditions=chunk,
                        branch="original",
                        sequence_length=len(original_ids),
                    ) as original_hooks:
                        amplified_original_results, _ = run_forward(
                            model=model,
                            batch=repeated_original,
                            token_map=token_map,
                            decoder_layers=decoder_layers,
                        )

                    with SymmetricAmplificationHooks(
                        decoder_layers=decoder_layers,
                        original_capture=original_capture,
                        flipped_capture=flipped_capture,
                        conditions=chunk,
                        branch="flipped",
                        sequence_length=len(original_ids),
                    ) as flipped_hooks:
                        amplified_flipped_results, _ = run_forward(
                            model=model,
                            batch=repeated_flipped,
                            token_map=token_map,
                            decoder_layers=decoder_layers,
                        )

                    if any(count != 1 for count in original_hooks.patch_counts):
                        raise RuntimeError(
                            "An original-branch condition was not patched "
                            "exactly once: "
                            f"{original_hooks.patch_counts}"
                        )
                    if any(count != 1 for count in flipped_hooks.patch_counts):
                        raise RuntimeError(
                            "A flipped-branch condition was not patched "
                            "exactly once: "
                            f"{flipped_hooks.patch_counts}"
                        )

                    for condition, amplified_original, amplified_flipped in zip(
                        chunk,
                        amplified_original_results,
                        amplified_flipped_results,
                    ):
                        amplified_original_scores = dict(
                            amplified_original["scores"]
                        )
                        mapped_amplified_flip_scores = map_vertical_scores(
                            amplified_flipped["scores"]
                        )
                        amplified_ensemble_scores = mean_scores(
                            amplified_original_scores,
                            mapped_amplified_flip_scores,
                        )
                        amplified_confidence_scores = (
                            confidence_select_scores(
                                amplified_original_scores,
                                mapped_amplified_flip_scores,
                            )
                        )
                        branch_agreement = (
                            prediction(amplified_original_scores)
                            == prediction(mapped_amplified_flip_scores)
                        )

                        if baseline_disagreement:
                            gated_else_original = amplified_ensemble_scores
                            gated_else_ensemble = amplified_ensemble_scores
                        else:
                            gated_else_original = original_scores
                            gated_else_ensemble = base_ensemble_scores

                        common = {
                            "path": method_path,
                            "sid": sid,
                            "ground_truth": ground_truth,
                            "baseline_prediction": original_prediction,
                            "layer": condition.layer,
                            "token_group": condition.token_group,
                            "gamma": condition.gamma,
                            "control": condition.control,
                            "condition_id": condition.condition_id,
                            "baseline_disagreement": baseline_disagreement,
                            "branch_agreement": branch_agreement,
                        }
                        add_method_row(
                            **common,
                            method="original_amplified",
                            scores=amplified_original_scores,
                        )
                        add_method_row(
                            **common,
                            method="flip_amplified_mapped",
                            scores=mapped_amplified_flip_scores,
                        )
                        add_method_row(
                            **common,
                            method="amplified_ensemble",
                            scores=amplified_ensemble_scores,
                        )
                        add_method_row(
                            **common,
                            method="amplified_confidence_select",
                            scores=amplified_confidence_scores,
                        )
                        add_method_row(
                            **common,
                            method="disagreement_gate_else_original",
                            scores=gated_else_original,
                        )
                        add_method_row(
                            **common,
                            method=(
                                "disagreement_gate_else_base_ensemble"
                            ),
                            scores=gated_else_ensemble,
                        )

                    del repeated_original, repeated_flipped
                    del amplified_original_results, amplified_flipped_results

                if (
                    args.print_every > 0
                    and evaluated % args.print_every == 0
                ):
                    tqdm.write(
                        f"\n[{evaluated}] sid={sid} gt={ground_truth} "
                        f"orig={original_prediction} "
                        f"flip(mapped)={mapped_flip_prediction} "
                        f"disagree={baseline_disagreement}"
                    )

                del original_capture, flipped_capture
                if evaluated % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(error_path, {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": (
                        traceback.format_exc().splitlines()[-30:]
                    ),
                })
                tqdm.write(
                    f"\n[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                if original_batch is not None:
                    del original_batch
                if flipped_batch is not None:
                    del flipped_batch
                if original_image is not None:
                    del original_image
                if flipped_image is not None:
                    del flipped_image

        method_rows = read_jsonl(method_path)
        if not method_rows:
            raise RuntimeError(
                "No method results were produced; inspect errors.jsonl"
            )
        summary_rows = summarize_method_rows(method_rows)
        write_csv(output_dir / "method_summary.csv", summary_rows)

        report = report_text(
            model_name=args.model,
            n_samples=evaluated,
            baseline_disagreement_count=baseline_disagreement_count,
            rows=summary_rows,
        )
        print("\n" + report)
        (output_dir / "report.txt").write_text(
            report,
            encoding="utf-8",
        )

        best_internal = [
            row for row in summary_rows
            if row["condition_id"] != "baseline"
            and not bool(row["control"])
            and row["method"] in (
                "original_amplified",
                "amplified_ensemble",
                "disagreement_gate_else_original",
                "disagreement_gate_else_base_ensemble",
            )
        ]
        best_internal.sort(
            key=lambda row: (
                -float(row["accuracy"]),
                -int(row["net_gain_count"]),
            )
        )
        summary_json = {
            "config": config,
            "evaluated": evaluated,
            "counts": dict(counts),
            "baseline_disagreement_count": baseline_disagreement_count,
            "baseline_disagreement_rate": (
                baseline_disagreement_count / evaluated
                if evaluated else None
            ),
            "elapsed_minutes": (
                time.time() - start_time
            ) / 60.0,
            "best_internal_conditions": best_internal[:50],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(
                summary_json,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("Saved:")
        for name in (
            "report.txt",
            "method_summary.csv",
            "summary.json",
            "baseline_pairs.jsonl",
            "method_results.jsonl",
        ):
            print(" ", output_dir / name)
        if error_path.exists():
            print(" ", error_path)

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
