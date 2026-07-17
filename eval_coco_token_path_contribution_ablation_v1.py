#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Causal token-path contribution ablation for COCO-two spatial reasoning.

This script does NOT use CLIP, object boxes, image coordinates, centroid
predictions, relation directions, or the ground-truth relation during the
intervention.

It tests two internal token-to-token pathways:

1) pair
   Remove the actual attention contribution from the subject object tokens to
   the later reference object-token queries:

       sum_h W_O^h sum_{k in subject} A^h(reference, k) V_k^h

2) route
   Remove the actual attention contribution from both object-token spans to the
   prompt-last decision query:

       sum_h W_O^h sum_{k in subject U reference} A^h(prompt_last, k) V_k^h

The removed quantity is A x V x W_O, not only an attention weight.  By default,
all heads in a layer window are ablated together.  An optional label-free
top-contribution mode keeps only the heads carrying the largest path norm for
that sample and layer.

Matched random-text-token controls can be enabled.  They remove the same number
of source-token contributions from non-object prompt tokens while preserving the
same query positions, layer window, and number of selected heads.

The fast default evaluates the first-token four-relation decision.  A slower
normal-generation mode is also available for validating selected windows.

Ground truth is used only after each forward/generation to measure correctness,
broken/fixed counts, and GT-margin changes.  No model parameter is trained.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-token-path-contribution-ablation-v1"
PATHWAYS = ("pair", "route")
CONTROLS = ("none", "random")
EVALUATIONS = ("first_token", "generation")
HEAD_SELECTIONS = ("all", "top_contribution")
CONDITION_SAMPLE_MODES = ("baseline_correct", "all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
        help="COCO helper script already used by the centroid experiments.",
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager"],
        help="Complete attention probabilities are required.",
    )
    p.add_argument(
        "--pathways",
        default="pair,route",
        help="Comma-separated subset of: pair,route.",
    )
    p.add_argument(
        "--layer-windows",
        default="auto:8",
        help=(
            "Layer windows. Use auto:N for consecutive windows of N layers, "
            "or explicit windows such as 0-3,4-7,8-11. Single layers are allowed."
        ),
    )
    p.add_argument(
        "--head-selection",
        default="all",
        choices=HEAD_SELECTIONS,
        help=(
            "all removes the path through every head in the selected layers; "
            "top_contribution dynamically removes the largest path-norm heads."
        ),
    )
    p.add_argument(
        "--head-fraction",
        type=float,
        default=1.0,
        help="Fraction used only when --head-selection top_contribution.",
    )
    p.add_argument(
        "--control",
        default="random",
        choices=CONTROLS,
        help="Add a matched non-object text-token ablation control.",
    )
    p.add_argument(
        "--evaluation",
        default="first_token",
        choices=EVALUATIONS,
        help=(
            "first_token is the recommended fast causal scan; generation runs "
            "normal greedy generation and is much slower."
        ),
    )
    p.add_argument(
        "--condition-samples",
        default="baseline_correct",
        choices=CONDITION_SAMPLE_MODES,
        help=(
            "baseline_correct runs necessity ablations only on baseline-correct "
            "samples. all also permits wrong-to-correct changes."
        ),
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_python_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing helper script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_names(value: str, allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)
    result: List[str] = []
    for raw in str(value).split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in allowed_set:
            raise ValueError(f"Unsupported value {name!r}; allowed={sorted(allowed_set)}")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("No values selected")
    return result


def parse_layer_windows(value: str, n_layers: int) -> List[Tuple[int, ...]]:
    text = str(value).strip().lower()
    if text.startswith("auto:"):
        size = int(text.split(":", 1)[1])
        if size <= 0:
            raise ValueError("auto window size must be positive")
        return [
            tuple(range(start, min(start + size, n_layers)))
            for start in range(0, n_layers, size)
        ]

    result: List[Tuple[int, ...]] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw:
            left, right = raw.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(raw)
        if start > end:
            raise ValueError(f"Invalid layer window {raw!r}")
        if start < 0 or end >= n_layers:
            raise ValueError(
                f"Layer window {raw!r} outside decoder range 0..{n_layers - 1}"
            )
        window = tuple(range(start, end + 1))
        if window not in result:
            result.append(window)
    if not result:
        raise ValueError("--layer-windows produced no windows")
    return result


def window_name(window: Sequence[int]) -> str:
    if len(window) == 1:
        return f"L{int(window[0]):02d}"
    return f"L{int(window[0]):02d}-{int(window[-1]):02d}"


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_output_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("No language-model logits returned")


def find_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise AttributeError(
        f"Could not find self-attention inside {type(layer).__name__}"
    )


def find_attention_weights(output: Any) -> Optional[torch.Tensor]:
    if not isinstance(output, (tuple, list)):
        return None
    for value in output[1:]:
        if (
            torch.is_tensor(value)
            and value.ndim == 4
            and value.shape[-1] >= value.shape[-2]
        ):
            return value
    return None


def replace_first_output(output: Any, first: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (first,) + output[1:]
    if isinstance(output, list):
        return [first] + list(output[1:])
    if torch.is_tensor(output):
        return first
    raise TypeError(f"Unsupported attention output type: {type(output).__name__}")


def project_without_bias(o_proj: Any, value: torch.Tensor) -> torch.Tensor:
    weight = getattr(o_proj, "weight", None)
    if weight is None:
        raise AttributeError(
            f"Output projection {type(o_proj).__name__} has no weight"
        )
    return F.linear(value.to(dtype=weight.dtype), weight, bias=None)


def span_positions(span: Tuple[int, int]) -> List[int]:
    start, end = [int(x) for x in span]
    return list(range(start, end + 1))


def choose_random_key_positions(
    *,
    input_ids: Sequence[int],
    tokenizer: Any,
    visual_indices: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    query_positions: Sequence[int],
    count: int,
    seed: int,
) -> List[int]:
    excluded = set(int(x) for x in visual_indices)
    excluded.update(int(x) for x in subject_positions)
    excluded.update(int(x) for x in reference_positions)
    excluded.update(int(x) for x in query_positions)

    special = set(int(x) for x in (getattr(tokenizer, "all_special_ids", []) or []))
    max_causal_key = min(int(x) for x in query_positions)
    candidates = [
        index
        for index, token_id in enumerate(input_ids)
        if index < max_causal_key
        and index not in excluded
        and int(token_id) not in special
    ]
    if len(candidates) < count:
        # Permit special prompt tokens only as a last-resort matched control.
        candidates = [
            index
            for index in range(max_causal_key)
            if index not in excluded
        ]
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} matched random keys available; need {count}"
        )
    rng = random.Random(int(seed))
    return sorted(rng.sample(candidates, count))


class TokenPathContributionKnockout:
    """Remove a selected token-to-token A x V x W_O path during prefill."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        layer_indices: Sequence[int],
        query_positions: Sequence[int],
        key_positions: Sequence[int],
        prompt_length: int,
        head_selection: str,
        head_fraction: float,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer_indices = [int(x) for x in layer_indices]
        self.query_positions = sorted(set(int(x) for x in query_positions))
        self.key_positions = sorted(set(int(x) for x in key_positions))
        self.prompt_length = int(prompt_length)
        self.head_selection = str(head_selection)
        self.head_fraction = float(head_fraction)

        self.handles: List[Any] = []
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.patch_events = 0
        self.patched_layers: Counter = Counter()
        self.selected_head_counts: Dict[int, List[int]] = defaultdict(list)
        self.removed_norms: Dict[int, List[float]] = defaultdict(list)

    def __enter__(self) -> "TokenPathContributionKnockout":
        for layer_index in self.layer_indices:
            if not (0 <= layer_index < len(self.decoder_layers)):
                raise ValueError(f"Layer {layer_index} outside decoder range")
            attention = find_self_attention(self.decoder_layers[layer_index])
            v_proj = getattr(attention, "v_proj", None)
            o_proj = getattr(attention, "o_proj", None)
            if v_proj is None or o_proj is None:
                raise AttributeError(
                    f"Layer {layer_index} does not expose v_proj/o_proj"
                )

            def make_v_hook(layer_id: int):
                def v_hook(_module: Any, _inputs: Any, output: Any) -> None:
                    value = output[0] if isinstance(output, tuple) else output
                    if torch.is_tensor(value):
                        self.value_cache[layer_id] = value
                return v_hook

            def make_attention_hook(
                layer_id: int,
                output_projection: Any,
            ):
                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    attention_output = (
                        output[0] if isinstance(output, (tuple, list)) else output
                    )
                    weights = find_attention_weights(output)
                    values_raw = self.value_cache.pop(layer_id, None)

                    if (
                        not torch.is_tensor(attention_output)
                        or weights is None
                        or values_raw is None
                    ):
                        return output
                    if attention_output.ndim != 3:
                        return output

                    batch, query_length, hidden_size = attention_output.shape
                    if query_length != self.prompt_length:
                        # Decode steps use query length 1. Only prefill is patched.
                        return output

                    weights = weights.float()
                    if int(weights.shape[0]) != batch:
                        return output
                    n_heads = int(weights.shape[1])
                    key_length = int(weights.shape[-1])
                    if (
                        int(weights.shape[-2]) != query_length
                        or key_length != self.prompt_length
                    ):
                        return output
                    if hidden_size % n_heads != 0:
                        raise RuntimeError(
                            f"hidden={hidden_size} is not divisible by heads={n_heads}"
                        )

                    head_dim = hidden_size // n_heads
                    if values_raw.ndim != 3:
                        raise RuntimeError(
                            f"Unexpected v_proj shape {tuple(values_raw.shape)}"
                        )
                    if (
                        int(values_raw.shape[0]) != batch
                        or int(values_raw.shape[1]) != key_length
                        or int(values_raw.shape[-1]) % head_dim != 0
                    ):
                        raise RuntimeError(
                            "v_proj output incompatible with returned attention: "
                            f"values={tuple(values_raw.shape)}, "
                            f"weights={tuple(weights.shape)}"
                        )

                    n_kv_heads = int(values_raw.shape[-1] // head_dim)
                    if n_heads % n_kv_heads != 0:
                        raise RuntimeError(
                            f"heads={n_heads} is not divisible by kv_heads={n_kv_heads}"
                        )

                    values = values_raw.float().reshape(
                        batch, key_length, n_kv_heads, head_dim
                    )
                    values = values.permute(0, 2, 1, 3)
                    repeat = n_heads // n_kv_heads
                    if repeat > 1:
                        values = values.repeat_interleave(repeat, dim=1)

                    query_positions = [
                        q for q in self.query_positions if 0 <= q < query_length
                    ]
                    key_positions = [
                        k for k in self.key_positions if 0 <= k < key_length
                    ]
                    if not query_positions or not key_positions:
                        raise RuntimeError("No valid query/key positions for path ablation")

                    # Causal validation: every selected key must be visible to every query.
                    if max(key_positions) > min(query_positions):
                        raise RuntimeError(
                            f"Non-causal path: max key={max(key_positions)} "
                            f"> min query={min(query_positions)}"
                        )

                    query_index = torch.as_tensor(
                        query_positions, device=weights.device, dtype=torch.long
                    )
                    key_index = torch.as_tensor(
                        key_positions, device=weights.device, dtype=torch.long
                    )

                    selected_weights = (
                        weights.index_select(2, query_index)
                        .index_select(3, key_index)
                    )  # [B,H,Q,Ks]
                    selected_values = values.index_select(2, key_index)  # [B,H,Ks,D]
                    path_output = torch.einsum(
                        "bhqk,bhkd->bhqd",
                        selected_weights,
                        selected_values,
                    )  # [B,H,Q,D]

                    if self.head_selection == "all":
                        head_mask = torch.ones(
                            (batch, n_heads),
                            device=path_output.device,
                            dtype=torch.bool,
                        )
                    elif self.head_selection == "top_contribution":
                        if not (0.0 < self.head_fraction <= 1.0):
                            raise ValueError("--head-fraction must be in (0,1]")
                        k = max(1, int(math.ceil(n_heads * self.head_fraction)))
                        norms = path_output.pow(2).sum(dim=(2, 3)).sqrt()
                        top = torch.topk(norms, k=k, dim=1).indices
                        head_mask = torch.zeros(
                            (batch, n_heads),
                            device=path_output.device,
                            dtype=torch.bool,
                        )
                        head_mask.scatter_(1, top, True)
                    else:
                        raise ValueError(
                            f"Unsupported head selection {self.head_selection!r}"
                        )

                    masked_path = path_output * head_mask[:, :, None, None]
                    delta = -masked_path

                    delta_heads = torch.zeros(
                        (batch, n_heads, query_length, head_dim),
                        device=attention_output.device,
                        dtype=torch.float32,
                    )
                    delta_heads.index_copy_(
                        2,
                        query_index.to(delta_heads.device),
                        delta.to(delta_heads.device),
                    )

                    delta_concat = (
                        delta_heads.permute(0, 2, 1, 3)
                        .contiguous()
                        .reshape(batch, query_length, hidden_size)
                    )
                    projected_delta = project_without_bias(
                        output_projection,
                        delta_concat,
                    )
                    new_attention_output = attention_output + projected_delta.to(
                        dtype=attention_output.dtype,
                        device=attention_output.device,
                    )

                    self.patch_events += 1
                    self.patched_layers[layer_id] += 1
                    self.selected_head_counts[layer_id].extend(
                        head_mask.sum(dim=1).detach().cpu().tolist()
                    )
                    self.removed_norms[layer_id].extend(
                        masked_path.float()
                        .pow(2)
                        .sum(dim=(1, 2, 3))
                        .sqrt()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    return replace_first_output(output, new_attention_output)

                return hook

            self.handles.append(
                v_proj.register_forward_hook(make_v_hook(layer_index))
            )
            self.handles.append(
                attention.register_forward_hook(
                    make_attention_hook(layer_index, o_proj)
                )
            )
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback_value: Any,
    ) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.value_cache.clear()


def run_first_token(
    *,
    model: Any,
    batch: Dict[str, Any],
    relation_token_map: Dict[str, List[int]],
    gt: str,
    base: Any,
) -> Dict[str, Any]:
    with torch.inference_mode():
        outputs = model(
            **batch,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    logits = get_output_logits(outputs)
    score = base.relation_scores(
        logits[0, -1, :],
        relation_token_map,
        gt,
    )
    del outputs, logits
    return {
        "prediction": score["prediction"],
        "correct": bool(score["prediction"] == gt),
        "gt_margin": score["gt_margin"],
        "top1_margin": score["top1_margin"],
        "generated_text": None,
    }


def run_generation(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    max_new_tokens: int,
    gt: str,
    base: Any,
) -> Dict[str, Any]:
    input_length = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        sequences = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=False,
        )
    text = base.decode_new_tokens(processor, sequences, input_length)
    prediction = base.normalize_relation(text)
    del sequences
    return {
        "prediction": prediction,
        "correct": bool(prediction == gt),
        "gt_margin": None,
        "top1_margin": None,
        "generated_text": text,
    }


def run_evaluation(
    *,
    evaluation: str,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    relation_token_map: Dict[str, List[int]],
    max_new_tokens: int,
    gt: str,
    base: Any,
) -> Dict[str, Any]:
    if evaluation == "first_token":
        return run_first_token(
            model=model,
            batch=batch,
            relation_token_map=relation_token_map,
            gt=gt,
            base=base,
        )
    if evaluation == "generation":
        return run_generation(
            model=model,
            processor=processor,
            batch=batch,
            max_new_tokens=max_new_tokens,
            gt=gt,
            base=base,
        )
    raise ValueError(f"Unsupported evaluation {evaluation!r}")


def condition_definitions(
    *,
    windows: Sequence[Sequence[int]],
    pathways: Sequence[str],
    include_random: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for window in windows:
        for pathway in pathways:
            rows.append({
                "name": f"{pathway}_{window_name(window)}",
                "pathway": pathway,
                "window": list(window),
                "control": False,
            })
            if include_random:
                rows.append({
                    "name": f"{pathway}_random_{window_name(window)}",
                    "pathway": pathway,
                    "window": list(window),
                    "control": True,
                })
    return rows


def summarize_conditions(
    rows: Sequence[Mapping[str, Any]],
    condition_defs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    by_name = {str(item["name"]): dict(item) for item in condition_defs}

    for name, definition in by_name.items():
        valid: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for row in rows:
            result = dict(row.get("conditions", {}).get(name) or {})
            if result.get("prediction") is None:
                continue
            valid.append((row, result))

        n = len(valid)
        if not n:
            output.append({
                **definition,
                "n": 0,
            })
            continue

        baseline_correct = np.asarray(
            [bool(row.get("baseline_correct")) for row, _ in valid],
            dtype=bool,
        )
        condition_correct = np.asarray(
            [bool(result.get("correct")) for _, result in valid],
            dtype=bool,
        )
        broken = int(np.sum(baseline_correct & ~condition_correct))
        fixed = int(np.sum(~baseline_correct & condition_correct))
        changed = int(np.sum([
            row.get("baseline_prediction") != result.get("prediction")
            for row, result in valid
        ]))

        baseline_acc = float(baseline_correct.mean())
        condition_acc = float(condition_correct.mean())

        margin_deltas = [
            float(result["gt_margin"]) - float(row["baseline_gt_margin"])
            for row, result in valid
            if result.get("gt_margin") is not None
            and row.get("baseline_gt_margin") is not None
        ]
        correct_margin_deltas = [
            float(result["gt_margin"]) - float(row["baseline_gt_margin"])
            for row, result in valid
            if bool(row.get("baseline_correct"))
            and result.get("gt_margin") is not None
            and row.get("baseline_gt_margin") is not None
        ]

        removed_norms = [
            float(result.get("removed_norm", 0.0))
            for _, result in valid
        ]
        selected_heads = [
            float(result.get("selected_heads_mean", 0.0))
            for _, result in valid
        ]

        output.append({
            **definition,
            "n": n,
            "baseline_accuracy": baseline_acc,
            "condition_accuracy": condition_acc,
            "accuracy_change": condition_acc - baseline_acc,
            "broken": broken,
            "fixed": fixed,
            "net_fixed_minus_broken": fixed - broken,
            "changed_predictions": changed,
            "broken_rate_among_baseline_correct": (
                broken / int(baseline_correct.sum())
                if int(baseline_correct.sum()) else None
            ),
            "mean_gt_margin_change": (
                float(np.mean(margin_deltas)) if margin_deltas else None
            ),
            "mean_gt_margin_change_on_baseline_correct": (
                float(np.mean(correct_margin_deltas))
                if correct_margin_deltas else None
            ),
            "mean_removed_path_norm": (
                float(np.mean(removed_norms)) if removed_norms else None
            ),
            "mean_selected_heads_per_layer": (
                float(np.mean(selected_heads)) if selected_heads else None
            ),
        })

    # Matched-control specificity.
    lookup = {str(row["name"]): row for row in output}
    for row in output:
        if bool(row.get("control")):
            continue
        control_name = (
            f"{row['pathway']}_random_"
            f"{window_name(row['window'])}"
        )
        control = lookup.get(control_name)
        if not control or not control.get("n"):
            row["excess_broken_vs_random"] = None
            row["extra_margin_drop_vs_random"] = None
            continue
        row["excess_broken_vs_random"] = (
            int(row.get("broken", 0)) - int(control.get("broken", 0))
        )
        real_delta = row.get("mean_gt_margin_change_on_baseline_correct")
        control_delta = control.get("mean_gt_margin_change_on_baseline_correct")
        row["extra_margin_drop_vs_random"] = (
            float(control_delta) - float(real_delta)
            if real_delta is not None and control_delta is not None
            else None
        )

    return output


def report_text(
    *,
    model: str,
    evaluation: str,
    n_rows: int,
    baseline_accuracy: Optional[float],
    summary_rows: Sequence[Mapping[str, Any]],
) -> str:
    header = (
        f"{'Condition':<30}{'N':>6}{'CondAcc':>10}{'Delta':>10}"
        f"{'Broken':>9}{'Break%':>9}{'MarginΔ':>11}"
        f"{'ExBreak':>9}{'ExtraDrop':>11}"
    )
    lines = [
        "=" * len(header),
        "COCO-TWO TOKEN-PATH CONTRIBUTION ABLATION",
        f"model={model} | evaluation={evaluation} | rows={n_rows}",
        (
            f"baseline_accuracy={baseline_accuracy:.4f}"
            if baseline_accuracy is not None
            else "baseline_accuracy=n/a"
        ),
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    def f4(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "-"

    ordered = sorted(
        summary_rows,
        key=lambda row: (
            bool(row.get("control")),
            -float(row.get("broken_rate_among_baseline_correct") or 0.0),
            float(row.get("mean_gt_margin_change_on_baseline_correct") or 0.0),
        ),
    )
    for row in ordered:
        lines.append(
            f"{str(row.get('name', '-')):<30}"
            f"{str(row.get('n', 0)):>6}"
            f"{f4(row.get('condition_accuracy')):>10}"
            f"{f4(row.get('accuracy_change')):>10}"
            f"{str(row.get('broken', 0)):>9}"
            f"{f4(row.get('broken_rate_among_baseline_correct')):>9}"
            f"{f4(row.get('mean_gt_margin_change_on_baseline_correct')):>11}"
            f"{str(row.get('excess_broken_vs_random', '-')):>9}"
            f"{f4(row.get('extra_margin_drop_vs_random')):>11}"
        )

    lines.extend([
        "",
        "Interpretation:",
        "- A necessary spatial path should break baseline-correct samples and "
        "decrease GT margin.",
        "- The real object-token path should damage more than its matched random "
        "text-token control.",
        "- A large effect only after ablating many layers may indicate a "
        "distributed/redundant circuit rather than one critical head.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0")
    if args.head_selection == "top_contribution":
        if not (0.0 < args.head_fraction <= 1.0):
            raise ValueError("--head-fraction must be in (0,1]")

    pathways = parse_names(args.pathways, PATHWAYS)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = import_python_file(Path(args.base_script), "_coco_ablation_base")
    module = base.import_two_object_module()
    records, audit = module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    if not records:
        raise RuntimeError("No usable COCO-two records")

    prompt_path = Path(args.prompt_jsonl)
    prompt_rows = base.load_standard_prompts(prompt_path)
    missing = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Prompt file is missing {len(missing)} records; first={missing[:10]}"
        )

    specs = base.merged_model_specs(module)
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
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite or another directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"

    model_cls = getattr(transformers, model_spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {model_spec.model_class}"
        )
    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(model_spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": model_spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Loading {args.model}: {model_spec.repo_id}")
    model = model_cls.from_pretrained(model_spec.repo_id, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_spec.repo_id,
        trust_remote_code=model_spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    windows = parse_layer_windows(args.layer_windows, len(decoder_layers))
    condition_defs = condition_definitions(
        windows=windows,
        pathways=pathways,
        include_random=(args.control == "random"),
    )
    relation_token_map = base.relation_token_variants(processor.tokenizer)

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "prompt_jsonl": str(prompt_path),
        "model": args.model,
        "repo_id": model_spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "pathways": pathways,
        "layer_windows": [list(window) for window in windows],
        "head_selection": args.head_selection,
        "head_fraction": args.head_fraction,
        "control": args.control,
        "evaluation": args.evaluation,
        "condition_samples": args.condition_samples,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "n_records": len(records),
        "audit": audit,
        "uses_gt_for_intervention": False,
        "uses_visual_coordinates": False,
        "uses_external_model": False,
        "updates_model_weights": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Decoder: {decoder_path}, n={len(decoder_layers)}")
    print("Windows:", ", ".join(window_name(window) for window in windows))
    print("Pathways:", ", ".join(pathways))
    print(
        f"Head selection: {args.head_selection}"
        + (
            f" fraction={args.head_fraction}"
            if args.head_selection == "top_contribution"
            else ""
        )
    )
    print(f"Evaluation: {args.evaluation}")
    print(f"Condition samples: {args.condition_samples}")

    completed = 0
    started = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"token-path-ablation:{args.model}",
        ):
            sid = int(record.sid)
            image: Optional[Image.Image] = None
            batch: Optional[Dict[str, Any]] = None
            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question_text = str(prompt_row["question_text"])
                gt = base.normalize_relation(prompt_row["answer_raw"])
                if gt not in base.RELATIONS:
                    raise ValueError(f"Unsupported GT {gt!r}")

                image = base.record_image(record)
                rendered = base.build_prompt(processor, question_text)
                batch = processor(
                    text=[rendered],
                    images=[image],
                    return_tensors="pt",
                )
                batch = base.move_batch(batch, device)

                input_ids = (
                    batch["input_ids"][0].detach().cpu().tolist()
                )
                subject_span, reference_span = base.locate_object_spans(
                    processor.tokenizer,
                    input_ids,
                    subject,
                    reference,
                )
                subject_positions = span_positions(subject_span)
                reference_positions = span_positions(reference_span)
                prompt_last = len(input_ids) - 1
                visual_indices = base.resolve_visual_indices(
                    model,
                    processor,
                    batch,
                    input_ids,
                )

                baseline = run_evaluation(
                    evaluation=args.evaluation,
                    model=model,
                    processor=processor,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    max_new_tokens=args.max_new_tokens,
                    gt=gt,
                    base=base,
                )

                row: Dict[str, Any] = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "question": question_text,
                    "gt": gt,
                    "baseline_prediction": baseline["prediction"],
                    "baseline_correct": baseline["correct"],
                    "baseline_gt_margin": baseline["gt_margin"],
                    "baseline_top1_margin": baseline["top1_margin"],
                    "baseline_generated_text": baseline["generated_text"],
                    "subject_span": list(subject_span),
                    "reference_span": list(reference_span),
                    "prompt_last": prompt_last,
                    "conditions": {},
                }

                run_conditions = (
                    args.condition_samples == "all"
                    or bool(baseline["correct"])
                )

                if run_conditions:
                    for condition in condition_defs:
                        pathway = str(condition["pathway"])
                        is_control = bool(condition["control"])
                        window = [int(x) for x in condition["window"]]

                        if pathway == "pair":
                            query_positions = reference_positions
                            object_keys = subject_positions
                        elif pathway == "route":
                            query_positions = [prompt_last]
                            object_keys = subject_positions + reference_positions
                        else:
                            raise ValueError(f"Unsupported pathway {pathway!r}")

                        if is_control:
                            key_positions = choose_random_key_positions(
                                input_ids=input_ids,
                                tokenizer=processor.tokenizer,
                                visual_indices=visual_indices,
                                subject_positions=subject_positions,
                                reference_positions=reference_positions,
                                query_positions=query_positions,
                                count=len(object_keys),
                                seed=args.seed * 1000003 + sid * 101 + sum(window),
                            )
                        else:
                            key_positions = object_keys

                        with TokenPathContributionKnockout(
                            decoder_layers=decoder_layers,
                            layer_indices=window,
                            query_positions=query_positions,
                            key_positions=key_positions,
                            prompt_length=len(input_ids),
                            head_selection=args.head_selection,
                            head_fraction=args.head_fraction,
                        ) as intervention:
                            result = run_evaluation(
                                evaluation=args.evaluation,
                                model=model,
                                processor=processor,
                                batch=batch,
                                relation_token_map=relation_token_map,
                                max_new_tokens=args.max_new_tokens,
                                gt=gt,
                                base=base,
                            )

                        if intervention.patch_events != len(window):
                            raise RuntimeError(
                                f"{condition['name']} expected {len(window)} "
                                f"prefill patch events, got {intervention.patch_events}"
                            )

                        selected_head_values = [
                            value
                            for values in intervention.selected_head_counts.values()
                            for value in values
                        ]
                        removed_norm_values = [
                            value
                            for values in intervention.removed_norms.values()
                            for value in values
                        ]

                        row["conditions"][condition["name"]] = {
                            **result,
                            "pathway": pathway,
                            "control": is_control,
                            "window": window,
                            "query_positions": query_positions,
                            "key_positions": key_positions,
                            "patch_events": intervention.patch_events,
                            "selected_heads_mean": (
                                float(np.mean(selected_head_values))
                                if selected_head_values else 0.0
                            ),
                            "removed_norm": (
                                float(np.mean(removed_norm_values))
                                if removed_norm_values else 0.0
                            ),
                        }

                append_jsonl(rows_path, row)
                completed += 1

                if args.print_every > 0 and completed % args.print_every == 0:
                    condition_results = row["conditions"]
                    broken = sum(
                        int(
                            bool(row["baseline_correct"])
                            and not bool(result.get("correct"))
                        )
                        for result in condition_results.values()
                        if not bool(result.get("control"))
                    )
                    tqdm.write(
                        f"\n[{completed}/{len(records)}] sid={sid} | "
                        f"GT/base={gt}/{baseline['prediction']} | "
                        f"base_correct={int(bool(baseline['correct']))} | "
                        f"conditions={len(condition_results)} | "
                        f"real-path broken events={broken}"
                    )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-20:],
                    },
                )
                tqdm.write(
                    f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if batch is not None:
                    del batch
                if image is not None:
                    del image

        rows = read_jsonl(rows_path)
        if not rows:
            raise RuntimeError("No samples completed; inspect errors.jsonl")

        baseline_valid = [
            row
            for row in rows
            if row.get("baseline_prediction") is not None
        ]
        baseline_accuracy = (
            float(np.mean([
                bool(row.get("baseline_correct"))
                for row in baseline_valid
            ]))
            if baseline_valid else None
        )

        summary_rows = summarize_conditions(rows, condition_defs)
        write_csv(output_dir / "condition_summary.csv", summary_rows)

        report = report_text(
            model=args.model,
            evaluation=args.evaluation,
            n_rows=len(rows),
            baseline_accuracy=baseline_accuracy,
            summary_rows=summary_rows,
        )
        print("\n" + report)
        (output_dir / "report.txt").write_text(report, encoding="utf-8")

        summary_json = {
            "config": config,
            "n_rows": len(rows),
            "n_baseline_valid": len(baseline_valid),
            "baseline_accuracy": baseline_accuracy,
            "condition_summary": summary_rows,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("Saved:")
        print(" ", output_dir / "report.txt")
        print(" ", output_dir / "condition_summary.csv")
        print(" ", output_dir / "summary.json")
        print(" ", rows_path)
        if errors_path.exists():
            print(" ", errors_path)

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
