#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three training-free causal repair experiments for spatial reasoning in VLMs.

Experiments
-----------
1. object_route_amplification
   Amplify the sample's OWN subject/reference-token contribution to the
   last-token attention output. No GT label, centroid prediction, donor sample,
   or externally learned direction is injected.

2. mlp_scaling
   Scale the MLP output written to the last token in selected decoder layers.

3. source_contribution_reweighting
   Reweight the last-token attention-output contributions from:
       object text tokens / visual tokens / other prompt text tokens.
   Generated-token contributions are left unchanged.

Evaluation
----------
The existing A/B/C grouping is read from:
    <input-root>/<model>/pass2_transfer_trace/sample_metadata.jsonl

A: centroid correct + baseline generation correct
B: centroid correct + baseline generation wrong
C: centroid wrong   + baseline generation correct

For every intervention the script reports:
- B repair rate;
- A damage rate;
- C damage rate;
- overall intervention accuracy;
- parse rate;
- net repair = B repair rate - A damage rate;
- conservative net repair =
      B repair rate - 0.5 * (A damage rate + C damage rate).

Implementation notes
--------------------
The attention interventions reconstruct source-specific A*V contributions from
the eager attention probabilities and the attention module's own V projection.
The source contribution is passed through o_proj WITHOUT adding o_proj bias.

The script is designed for the llava16 AdaptVis branch and reuses model/data
utilities from:
    trace_centroid_generation_groups_v2_1.py

Supported model aliases are whatever the repository's two-object backend exposes,
including qwen-3b, qwen-7b, llava-7b, and llava-13b when available.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "run-spatial-repair-three-experiments-v1"

RELATIONS = ("left", "right", "above", "below")

GROUP_A = "A_centroid_correct_generation_correct"
GROUP_B = "B_centroid_correct_generation_wrong"
GROUP_C = "C_centroid_wrong_generation_correct"
GROUP_D = "D_centroid_wrong_generation_wrong"

PRIMARY_GROUPS = (GROUP_A, GROUP_B, GROUP_C)

# Evidence-driven defaults for the two Qwen models already analyzed.
# Other architectures use a decoder-depth fallback and should later be refined.
AUTO_WINDOWS: Dict[str, Dict[str, str]] = {
    "qwen-3b": {
        "object_amp": "23-32",
        "mlp_scale": "28-32",
        "source_reweight": "23-32",
    },
    "qwen-7b": {
        "object_amp": "20-23",
        "mlp_scale": "20-23",
        "source_reweight": "20-23",
    },
}


# ---------------------------------------------------------------------------
# CLI and generic utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--models",
        default="qwen-3b,qwen-7b",
        help="Comma-separated repository model aliases.",
    )
    parser.add_argument("--dataset", default="coco_two")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--input-root",
        default="output/three_group_transfer_fresh/coco",
        help="Root containing each model's pass2_transfer_trace.",
    )
    parser.add_argument(
        "--output-root",
        default="output/spatial_repair_three_experiments/coco",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=["eager"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--groups",
        default="A,B,C",
        help="Any subset of A,B,C,D.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional deterministic sample cap per group.",
    )
    parser.add_argument(
        "--sid",
        type=int,
        default=None,
        help="Run one sample only for debugging.",
    )

    parser.add_argument(
        "--experiments",
        default="object_amp,mlp_scale,source_reweight",
        help="Subset of object_amp,mlp_scale,source_reweight.",
    )
    parser.add_argument(
        "--object-alphas",
        default="0.25,0.5,1.0",
        help="Object contribution is changed by +alpha*object_route.",
    )
    parser.add_argument(
        "--mlp-betas",
        default="0.0,0.25,0.5,0.75",
        help="Selected last-token MLP outputs are multiplied by beta.",
    )
    parser.add_argument(
        "--source-presets",
        default=(
            "obj1.5_vis0.75_other1.0;"
            "obj2.0_vis0.5_other1.0;"
            "obj2.0_vis0.5_other0.5"
        ),
        help=(
            "Semicolon-separated source scales. Format: "
            "objX_visY_otherZ"
        ),
    )

    parser.add_argument(
        "--object-layers",
        default="auto",
        help="Layer spec such as 23-32, 23,25,28, or auto.",
    )
    parser.add_argument(
        "--mlp-layers",
        default="auto",
        help="Layer spec such as 28-32 or auto.",
    )
    parser.add_argument(
        "--source-layers",
        default="auto",
        help="Layer spec such as 23-32 or auto.",
    )
    parser.add_argument(
        "--selected-heads",
        default=None,
        help=(
            "Optional layer:head list for attention interventions, e.g. "
            "'23:4,23:9,28:7'. Omit to use every head in active layers."
        ),
    )

    parser.add_argument(
        "--verify-baseline",
        action="store_true",
        help=(
            "Rerun an unmodified baseline generation for each sample. "
            "Otherwise use the stored baseline grouping/prediction."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous result files for each model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed (sid, config_id) rows.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--core-module",
        default="trace_centroid_generation_groups_v2_1",
    )

    return parser.parse_args()


def import_core(name: str):
    module = importlib.import_module(name)
    required = (
        "import_two_object_module",
        "resolve_prompt_path",
        "load_standard_prompts",
        "record_image",
        "make_question_batch",
        "resolve_dtype",
        "configure_processor",
        "resolve_decoder_layers",
    )
    missing = [key for key in required if not hasattr(module, key)]
    if missing:
        raise RuntimeError(
            f"Core module {name!r} is missing required functions: {missing}"
        )
    return module


def parse_models(text: str) -> List[str]:
    values: List[str] = []
    for item in str(text).split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    if not values:
        raise ValueError("No models selected.")
    return values


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"Empty float list: {text!r}")
    return values


def parse_experiments(text: str) -> List[str]:
    allowed = {"object_amp", "mlp_scale", "source_reweight"}
    values = [
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    ]
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(f"Unknown experiments: {invalid}")
    return values


def parse_groups(text: str) -> List[str]:
    aliases = {
        "A": GROUP_A,
        "B": GROUP_B,
        "C": GROUP_C,
        "D": GROUP_D,
        GROUP_A: GROUP_A,
        GROUP_B: GROUP_B,
        GROUP_C: GROUP_C,
        GROUP_D: GROUP_D,
    }
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item not in aliases:
            raise ValueError(f"Unknown group: {item}")
        group = aliases[item]
        if group not in values:
            values.append(group)
    if not values:
        raise ValueError("No groups selected.")
    return values


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    layers: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            left, right = int(match.group(1)), int(match.group(2))
            if right < left:
                left, right = right, left
            layers.extend(range(left, right + 1))
        else:
            layers.append(int(part))

    layers = sorted(set(layers))
    invalid = [layer for layer in layers if not 0 <= layer < n_layers]
    if invalid:
        raise ValueError(
            f"Layers outside [0,{n_layers - 1}]: {invalid}"
        )
    if not layers:
        raise ValueError(f"Layer spec is empty: {text!r}")
    return layers


def generic_window(
    experiment: str,
    n_layers: int,
) -> List[int]:
    if experiment == "mlp_scale":
        start = int(math.floor(0.70 * n_layers))
        stop = int(math.ceil(0.90 * n_layers))
    else:
        start = int(math.floor(0.60 * n_layers))
        stop = int(math.ceil(0.90 * n_layers))
    start = max(0, min(start, n_layers - 1))
    stop = max(start + 1, min(stop, n_layers))
    return list(range(start, stop))


def resolve_window(
    *,
    model_name: str,
    experiment: str,
    requested: str,
    n_layers: int,
) -> List[int]:
    if str(requested).strip().lower() != "auto":
        return parse_layer_spec(requested, n_layers)

    auto = AUTO_WINDOWS.get(model_name, {}).get(experiment)
    if auto is not None:
        return parse_layer_spec(auto, n_layers)
    return generic_window(experiment, n_layers)


def parse_selected_heads(
    text: Optional[str],
    n_layers: int,
) -> Optional[Dict[int, set[int]]]:
    if text is None:
        return None
    result: Dict[int, set[int]] = defaultdict(set)
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid head spec {item!r}; expected layer:head."
            )
        layer_text, head_text = item.split(":", 1)
        layer, head = int(layer_text), int(head_text)
        if not 0 <= layer < n_layers:
            raise ValueError(f"Head layer outside model: {layer}")
        if head < 0:
            raise ValueError(f"Negative head index: {head}")
        result[layer].add(head)
    if not result:
        raise ValueError("--selected-heads produced no heads.")
    return dict(result)


def parse_source_presets(text: str) -> List[Tuple[float, float, float]]:
    pattern = re.compile(
        r"obj([-+]?\d*\.?\d+)_vis([-+]?\d*\.?\d+)_other([-+]?\d*\.?\d+)"
    )
    presets: List[Tuple[float, float, float]] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        match = pattern.fullmatch(item)
        if not match:
            raise ValueError(
                f"Invalid source preset {item!r}; "
                "expected objX_visY_otherZ."
            )
        presets.append(tuple(float(match.group(i)) for i in range(1, 4)))
    if not presets:
        raise ValueError("No source presets selected.")
    return presets


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


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


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    exact = {
        "left": "left",
        "left of": "left",
        "to the left": "left",
        "to the left of": "left",
        "right": "right",
        "right of": "right",
        "to the right": "right",
        "to the right of": "right",
        "above": "above",
        "over": "above",
        "on": "above",
        "on top of": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
        "bottom": "below",
    }
    if text in exact:
        return exact[text]

    # Match complete words only. Prefer the first relation word generated.
    candidates: List[Tuple[int, str]] = []
    for token, relation in (
        ("left", "left"),
        ("right", "right"),
        ("above", "above"),
        ("below", "below"),
        ("under", "below"),
        ("beneath", "below"),
        ("over", "above"),
        ("on", "above"),
    ):
        match = re.search(rf"\b{re.escape(token)}\b", text)
        if match:
            candidates.append((match.start(), relation))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def group_short(group: str) -> str:
    return {
        GROUP_A: "A",
        GROUP_B: "B",
        GROUP_C: "C",
        GROUP_D: "D",
    }.get(group, group)


def cap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_groups: Sequence[str],
    max_per_group: Optional[int],
    seed: int,
    sid: Optional[int],
) -> List[Dict[str, Any]]:
    filtered = [
        dict(row)
        for row in rows
        if str(row.get("group")) in selected_groups
    ]
    if sid is not None:
        filtered = [
            row for row in filtered
            if int(row.get("sid", -1)) == sid
        ]

    if max_per_group is None:
        return sorted(filtered, key=lambda row: int(row["sid"]))

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        grouped[str(row["group"])].append(row)

    output: List[Dict[str, Any]] = []
    for group in selected_groups:
        candidates = grouped.get(group, [])
        candidates.sort(key=lambda row: int(row["sid"]))
        if len(candidates) > max_per_group:
            candidates = rng.sample(candidates, max_per_group)
            candidates.sort(key=lambda row: int(row["sid"]))
        output.extend(candidates)
    return sorted(output, key=lambda row: (str(row["group"]), int(row["sid"])))


# ---------------------------------------------------------------------------
# Intervention configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterventionConfig:
    config_id: str
    experiment: str
    layers: Tuple[int, ...]
    object_scale: float = 1.0
    visual_scale: float = 1.0
    other_scale: float = 1.0
    mlp_beta: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "experiment": self.experiment,
            "layers": ",".join(str(x) for x in self.layers),
            "object_scale": self.object_scale,
            "visual_scale": self.visual_scale,
            "other_scale": self.other_scale,
            "mlp_beta": self.mlp_beta,
        }


def build_configs(
    *,
    experiments: Sequence[str],
    object_layers: Sequence[int],
    mlp_layers: Sequence[int],
    source_layers: Sequence[int],
    object_alphas: Sequence[float],
    mlp_betas: Sequence[float],
    source_presets: Sequence[Tuple[float, float, float]],
) -> List[InterventionConfig]:
    configs: List[InterventionConfig] = []

    if "object_amp" in experiments:
        for alpha in object_alphas:
            configs.append(
                InterventionConfig(
                    config_id=(
                        f"object_amp__L{object_layers[0]}-{object_layers[-1]}"
                        f"__alpha{alpha:g}"
                    ),
                    experiment="object_amp",
                    layers=tuple(object_layers),
                    object_scale=1.0 + float(alpha),
                )
            )

    if "mlp_scale" in experiments:
        for beta in mlp_betas:
            configs.append(
                InterventionConfig(
                    config_id=(
                        f"mlp_scale__L{mlp_layers[0]}-{mlp_layers[-1]}"
                        f"__beta{beta:g}"
                    ),
                    experiment="mlp_scale",
                    layers=tuple(mlp_layers),
                    mlp_beta=float(beta),
                )
            )

    if "source_reweight" in experiments:
        for object_scale, visual_scale, other_scale in source_presets:
            configs.append(
                InterventionConfig(
                    config_id=(
                        f"source_reweight__L{source_layers[0]}-{source_layers[-1]}"
                        f"__obj{object_scale:g}_vis{visual_scale:g}"
                        f"_other{other_scale:g}"
                    ),
                    experiment="source_reweight",
                    layers=tuple(source_layers),
                    object_scale=float(object_scale),
                    visual_scale=float(visual_scale),
                    other_scale=float(other_scale),
                )
            )

    if not configs:
        raise ValueError("No intervention configurations were generated.")
    return configs


# ---------------------------------------------------------------------------
# Token-position resolution
# ---------------------------------------------------------------------------

def tokenizer_variants(tokenizer: Any, text: str) -> List[List[int]]:
    variants: List[List[int]] = []
    for candidate in (text, " " + text, text.strip(), " " + text.strip()):
        if not candidate:
            continue
        encoded = tokenizer(
            candidate,
            add_special_tokens=False,
        ).input_ids
        if encoded and encoded not in variants:
            variants.append(list(encoded))
    return variants


def find_subsequence_all(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> List[List[int]]:
    if not pattern or len(pattern) > len(sequence):
        return []
    output: List[List[int]] = []
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if list(sequence[start : start + width]) == list(pattern):
            output.append(list(range(start, start + width)))
    return output


def find_text_token_positions(
    tokenizer: Any,
    input_ids: Sequence[int],
    text: str,
) -> List[int]:
    matches: List[List[int]] = []
    for pattern in tokenizer_variants(tokenizer, text):
        matches.extend(find_subsequence_all(input_ids, pattern))
    if not matches:
        raise RuntimeError(
            f"Unable to find token span for object text {text!r}."
        )
    # Object names are expected in the final user question. Prefer the last match.
    matches.sort(key=lambda positions: (positions[-1], len(positions)))
    return matches[-1]


def candidate_image_token_ids(model: Any, tokenizer: Any) -> set[int]:
    ids: set[int] = set()

    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ]
    for config in configs:
        if config is None:
            continue
        for attribute in (
            "image_token_id",
            "image_token_index",
            "vision_token_id",
            "video_token_id",
        ):
            value = getattr(config, attribute, None)
            if isinstance(value, (int, np.integer)) and int(value) >= 0:
                ids.add(int(value))

    for token in (
        "<image>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_pad|>",
    ):
        try:
            value = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if (
            isinstance(value, (int, np.integer))
            and int(value) >= 0
            and int(value) != getattr(tokenizer, "unk_token_id", None)
        ):
            ids.add(int(value))

    return ids


@dataclass
class PromptPositionSpec:
    input_length: int
    subject_input_positions: Tuple[int, ...]
    reference_input_positions: Tuple[int, ...]
    image_input_positions: Tuple[int, ...]
    image_token_ids: Tuple[int, ...]


@dataclass
class ExpandedPositionSpec:
    prompt_length: int
    subject_positions: Tuple[int, ...]
    reference_positions: Tuple[int, ...]
    visual_positions: Tuple[int, ...]
    other_positions: Tuple[int, ...]


def build_prompt_position_spec(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    subject: str,
    reference: str,
) -> PromptPositionSpec:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"Only batch size 1 is supported, got input_ids={tuple(input_ids.shape)}"
        )

    ids = input_ids[0].detach().cpu().tolist()
    subject_positions = find_text_token_positions(
        tokenizer, ids, subject
    )
    reference_positions = find_text_token_positions(
        tokenizer, ids, reference
    )

    image_ids = candidate_image_token_ids(model, tokenizer)
    image_positions = [
        index for index, token_id in enumerate(ids)
        if int(token_id) in image_ids
    ]
    if not image_positions:
        raise RuntimeError(
            "No image token positions found in processor input_ids. "
            f"Candidate image token IDs were: {sorted(image_ids)}"
        )

    return PromptPositionSpec(
        input_length=len(ids),
        subject_input_positions=tuple(subject_positions),
        reference_input_positions=tuple(reference_positions),
        image_input_positions=tuple(image_positions),
        image_token_ids=tuple(sorted(image_ids)),
    )


def expand_positions(
    prompt: PromptPositionSpec,
    hidden_length: int,
) -> ExpandedPositionSpec:
    """
    Map processor input positions to decoder hidden-sequence positions.

    Case 1:
        hidden_length == input_length
        The processor has already expanded visual tokens.

    Case 2:
        hidden_length > input_length and there is one image marker
        The model expanded that marker into a visual-token span.
    """
    if hidden_length == prompt.input_length:
        subject = list(prompt.subject_input_positions)
        reference = list(prompt.reference_input_positions)
        visual = list(prompt.image_input_positions)

    elif (
        hidden_length > prompt.input_length
        and len(prompt.image_input_positions) == 1
    ):
        marker = int(prompt.image_input_positions[0])
        visual_count = hidden_length - prompt.input_length + 1
        shift = visual_count - 1

        def map_text(position: int) -> int:
            return position if position < marker else position + shift

        subject = [map_text(x) for x in prompt.subject_input_positions]
        reference = [map_text(x) for x in prompt.reference_input_positions]
        visual = list(range(marker, marker + visual_count))

    else:
        raise RuntimeError(
            "Unable to map processor tokens to decoder positions: "
            f"input_length={prompt.input_length}, "
            f"hidden_length={hidden_length}, "
            f"image_input_positions={prompt.image_input_positions}. "
            "This architecture needs a model-specific mapping."
        )

    object_set = set(subject) | set(reference)
    visual_set = set(visual)
    if object_set & visual_set:
        raise RuntimeError("Object text positions overlap visual positions.")

    all_prompt = set(range(hidden_length))
    other = sorted(all_prompt - object_set - visual_set)

    return ExpandedPositionSpec(
        prompt_length=hidden_length,
        subject_positions=tuple(sorted(set(subject))),
        reference_positions=tuple(sorted(set(reference))),
        visual_positions=tuple(sorted(visual_set)),
        other_positions=tuple(other),
    )


# ---------------------------------------------------------------------------
# Attention and MLP hooks
# ---------------------------------------------------------------------------

def replace_first_output(
    output: Any,
    new_first: torch.Tensor,
) -> Any:
    if torch.is_tensor(output):
        return new_first
    if isinstance(output, tuple):
        return (new_first, *output[1:])
    if isinstance(output, list):
        return [new_first, *output[1:]]
    raise TypeError(
        f"Unsupported module output type: {type(output).__name__}"
    )


def output_first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    raise TypeError(
        f"Unable to resolve first tensor from {type(output).__name__}"
    )


def find_attention_weights(
    output: Any,
    q_length: int,
) -> Optional[torch.Tensor]:
    if not isinstance(output, (tuple, list)):
        return None
    candidates = []
    for value in output[1:]:
        if (
            torch.is_tensor(value)
            and value.ndim == 4
            and value.shape[-2] == q_length
        ):
            candidates.append(value)
    if not candidates:
        return None
    # Attention probabilities normally have the largest key dimension.
    candidates.sort(key=lambda value: int(value.shape[-1]), reverse=True)
    return candidates[0]


def resolve_attention_projections(module: torch.nn.Module):
    v_proj = getattr(module, "v_proj", None)
    o_proj = getattr(module, "o_proj", None)

    if v_proj is None:
        raise RuntimeError(
            f"{type(module).__name__} has no v_proj. "
            "Fused QKV attention is not yet supported by this script."
        )
    if o_proj is None:
        o_proj = getattr(module, "dense", None)
    if o_proj is None or not hasattr(o_proj, "weight"):
        raise RuntimeError(
            f"{type(module).__name__} has no supported o_proj/dense."
        )
    return v_proj, o_proj


def repeat_kv(
    hidden_states: torch.Tensor,
    number_of_heads: int,
) -> torch.Tensor:
    """
    hidden_states: [B, KV_heads, K, D]
    returns:       [B, attention_heads, K, D]
    """
    kv_heads = int(hidden_states.shape[1])
    if kv_heads == number_of_heads:
        return hidden_states
    if number_of_heads % kv_heads != 0:
        raise RuntimeError(
            f"Cannot repeat {kv_heads} KV heads to {number_of_heads} heads."
        )
    groups = number_of_heads // kv_heads
    return hidden_states.repeat_interleave(groups, dim=1)


@dataclass
class LayerAttentionCache:
    hidden_states: Optional[torch.Tensor] = None
    prompt_values: Optional[torch.Tensor] = None
    positions: Optional[ExpandedPositionSpec] = None


class RepairInterventionManager:
    def __init__(
        self,
        *,
        model: Any,
        layers: Sequence[torch.nn.Module],
        attention_layers: Sequence[int],
        mlp_layers: Sequence[int],
        selected_heads: Optional[Mapping[int, set[int]]],
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.attention_layers = sorted(set(int(x) for x in attention_layers))
        self.mlp_layers = sorted(set(int(x) for x in mlp_layers))
        self.selected_heads = selected_heads

        self.current_config: Optional[InterventionConfig] = None
        self.prompt_spec: Optional[PromptPositionSpec] = None
        self.layer_cache: Dict[int, LayerAttentionCache] = {
            layer: LayerAttentionCache()
            for layer in self.attention_layers
        }
        self.handles: List[Any] = []

        for layer_index in self.attention_layers:
            layer = self.layers[layer_index]
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise RuntimeError(
                    f"Decoder layer {layer_index} has no self_attn."
                )

            self.handles.append(
                attention.register_forward_pre_hook(
                    self._make_attention_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._make_attention_post_hook(layer_index),
                    with_kwargs=True,
                )
            )

        for layer_index in self.mlp_layers:
            layer = self.layers[layer_index]
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                raise RuntimeError(
                    f"Decoder layer {layer_index} has no mlp."
                )
            self.handles.append(
                mlp.register_forward_hook(
                    self._make_mlp_hook(layer_index)
                )
            )

    def close(self) -> None:
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()
        self.reset()

    def reset(self) -> None:
        self.current_config = None
        self.prompt_spec = None
        for layer_cache in self.layer_cache.values():
            layer_cache.hidden_states = None
            layer_cache.prompt_values = None
            layer_cache.positions = None

    def configure(
        self,
        *,
        config: InterventionConfig,
        prompt_spec: PromptPositionSpec,
    ) -> None:
        self.reset()
        self.current_config = config
        self.prompt_spec = prompt_spec

    def _active_attention(self, layer_index: int) -> bool:
        config = self.current_config
        return (
            config is not None
            and config.experiment in {"object_amp", "source_reweight"}
            and layer_index in config.layers
        )

    def _active_mlp(self, layer_index: int) -> bool:
        config = self.current_config
        return (
            config is not None
            and config.experiment == "mlp_scale"
            and layer_index in config.layers
        )

    def _make_attention_pre_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ):
            if not self._active_attention(layer_index):
                return None

            hidden_states = None
            if args and torch.is_tensor(args[0]):
                hidden_states = args[0]
            elif torch.is_tensor(kwargs.get("hidden_states")):
                hidden_states = kwargs["hidden_states"]

            if hidden_states is None:
                raise RuntimeError(
                    f"Unable to capture attention input at L{layer_index}."
                )

            self.layer_cache[layer_index].hidden_states = hidden_states
            return None

        return hook

    def _head_mask(
        self,
        *,
        layer_index: int,
        number_of_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.ones(
            (1, number_of_heads, 1),
            device=device,
            dtype=dtype,
        )
        if self.selected_heads is None:
            return mask

        selected = self.selected_heads.get(layer_index, set())
        mask.zero_()
        for head in selected:
            if not 0 <= int(head) < number_of_heads:
                raise RuntimeError(
                    f"Selected head L{layer_index}H{head} outside "
                    f"number_of_heads={number_of_heads}."
                )
            mask[:, int(head), :] = 1.0
        return mask

    def _source_contribution(
        self,
        *,
        weights: torch.Tensor,
        values: torch.Tensor,
        positions: Sequence[int],
        o_proj: torch.nn.Module,
        layer_index: int,
    ) -> torch.Tensor:
        """
        weights: [B,H,K] for last query
        values:  [B,H,K,D]
        output:  [B,1,hidden]
        """
        if not positions:
            batch = int(weights.shape[0])
            hidden = int(o_proj.weight.shape[0])
            return torch.zeros(
                (batch, 1, hidden),
                device=weights.device,
                dtype=values.dtype,
            )

        key_length = int(weights.shape[-1])
        valid_positions = [
            int(position)
            for position in positions
            if 0 <= int(position) < key_length
        ]
        if not valid_positions:
            batch = int(weights.shape[0])
            hidden = int(o_proj.weight.shape[0])
            return torch.zeros(
                (batch, 1, hidden),
                device=weights.device,
                dtype=values.dtype,
            )

        index = torch.as_tensor(
            valid_positions,
            device=weights.device,
            dtype=torch.long,
        )
        selected_weights = weights.index_select(-1, index)
        selected_values = values.index_select(-2, index)

        contribution_heads = torch.einsum(
            "bhk,bhkd->bhd",
            selected_weights,
            selected_values,
        )

        head_mask = self._head_mask(
            layer_index=layer_index,
            number_of_heads=int(contribution_heads.shape[1]),
            device=contribution_heads.device,
            dtype=contribution_heads.dtype,
        )
        contribution_heads = contribution_heads * head_mask

        pre_projection = contribution_heads.reshape(
            contribution_heads.shape[0],
            1,
            -1,
        )

        # Important: source decomposition must not add o_proj bias.
        return F.linear(
            pre_projection,
            o_proj.weight,
            bias=None,
        )

    def _make_attention_post_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            output: Any,
        ):
            if not self._active_attention(layer_index):
                return output

            config = self.current_config
            prompt_spec = self.prompt_spec
            if config is None or prompt_spec is None:
                return output

            attention_output = output_first_tensor(output)
            q_length = int(attention_output.shape[-2])
            attention_weights = find_attention_weights(output, q_length)
            if attention_weights is None:
                raise RuntimeError(
                    f"L{layer_index} did not return eager attention weights. "
                    "Run with --attn-impl eager."
                )

            cache = self.layer_cache[layer_index]
            hidden_states = cache.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    f"Missing attention input cache at L{layer_index}."
                )

            v_proj, o_proj = resolve_attention_projections(module)

            current_values = v_proj(hidden_states)
            number_of_heads = int(attention_weights.shape[1])
            hidden_per_head = int(o_proj.weight.shape[1]) // number_of_heads
            if hidden_per_head <= 0:
                raise RuntimeError(
                    f"Invalid head dimension at L{layer_index}."
                )
            if current_values.shape[-1] % hidden_per_head != 0:
                raise RuntimeError(
                    f"v_proj width {current_values.shape[-1]} is not divisible "
                    f"by head_dim={hidden_per_head} at L{layer_index}."
                )

            kv_heads = int(current_values.shape[-1]) // hidden_per_head
            current_values = current_values.view(
                current_values.shape[0],
                current_values.shape[1],
                kv_heads,
                hidden_per_head,
            ).transpose(1, 2)
            current_values = repeat_kv(current_values, number_of_heads)

            # Prefill captures the decoder-expanded prompt V states and positions.
            if int(hidden_states.shape[1]) > 1:
                cache.prompt_values = current_values.detach()
                cache.positions = expand_positions(
                    prompt_spec,
                    hidden_length=int(hidden_states.shape[1]),
                )

            if cache.prompt_values is None or cache.positions is None:
                raise RuntimeError(
                    f"L{layer_index} received decode step before prompt cache."
                )

            prompt_values = cache.prompt_values.to(
                device=attention_weights.device,
                dtype=attention_weights.dtype,
            )
            key_length = int(attention_weights.shape[-1])
            prompt_length = min(
                int(cache.positions.prompt_length),
                int(prompt_values.shape[-2]),
                key_length,
            )
            values = prompt_values[:, :, :prompt_length, :]
            last_weights = attention_weights[:, :, -1, :prompt_length]

            object_positions = (
                tuple(cache.positions.subject_positions)
                + tuple(cache.positions.reference_positions)
            )
            object_contribution = self._source_contribution(
                weights=last_weights,
                values=values,
                positions=object_positions,
                o_proj=o_proj,
                layer_index=layer_index,
            )

            delta = (
                float(config.object_scale) - 1.0
            ) * object_contribution

            if config.experiment == "source_reweight":
                visual_contribution = self._source_contribution(
                    weights=last_weights,
                    values=values,
                    positions=cache.positions.visual_positions,
                    o_proj=o_proj,
                    layer_index=layer_index,
                )
                other_contribution = self._source_contribution(
                    weights=last_weights,
                    values=values,
                    positions=cache.positions.other_positions,
                    o_proj=o_proj,
                    layer_index=layer_index,
                )
                delta = (
                    delta
                    + (float(config.visual_scale) - 1.0)
                    * visual_contribution
                    + (float(config.other_scale) - 1.0)
                    * other_contribution
                )

            modified = attention_output.clone()
            modified[:, -1:, :] = modified[:, -1:, :] + delta.to(
                dtype=modified.dtype,
                device=modified.device,
            )
            return replace_first_output(output, modified)

        return hook

    def _make_mlp_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            args: Tuple[Any, ...],
            output: Any,
        ):
            if not self._active_mlp(layer_index):
                return output

            config = self.current_config
            if config is None:
                return output

            mlp_output = output_first_tensor(output)
            modified = mlp_output.clone()
            modified[:, -1:, :] = (
                modified[:, -1:, :]
                * float(config.mlp_beta)
            )
            return replace_first_output(output, modified)

        return hook


# ---------------------------------------------------------------------------
# Generation and aggregation
# ---------------------------------------------------------------------------

def move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


@torch.inference_mode()
def generate_relation(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    max_new_tokens: int,
    need_attentions: bool,
) -> Tuple[str, Optional[str]]:
    input_ids = batch.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise RuntimeError("Generation batch has no input_ids tensor.")

    generated = model.generate(
        **batch,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        output_attentions=need_attentions,
        return_dict_in_generate=False,
    )

    if not torch.is_tensor(generated):
        sequences = getattr(generated, "sequences", None)
        if not torch.is_tensor(sequences):
            raise RuntimeError(
                f"Unsupported generate output: {type(generated).__name__}"
            )
        generated = sequences

    new_token_ids = generated[0, input_ids.shape[1] :]
    text = processor.tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()
    return text, normalize_relation(text)


def completed_keys(path: Path) -> set[Tuple[int, str]]:
    if not path.exists():
        return set()
    keys: set[Tuple[int, str]] = set()
    for row in read_jsonl(path):
        try:
            keys.add((int(row["sid"]), str(row["config_id"])))
        except Exception:
            continue
    return keys


def aggregate_results(
    rows: Sequence[Mapping[str, Any]],
    configs: Sequence[InterventionConfig],
) -> List[Dict[str, Any]]:
    config_lookup = {config.config_id: config for config in configs}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for config_id, candidates in grouped.items():
        config = config_lookup.get(config_id)
        if config is None:
            continue

        summary: Dict[str, Any] = {
            **config.as_dict(),
            "n": len(candidates),
            "parse_rate": float(
                np.mean([bool(row.get("parsed")) for row in candidates])
            ),
            "intervention_accuracy": float(
                np.mean([bool(row.get("intervention_correct")) for row in candidates])
            ),
        }

        group_rates: Dict[str, float] = {}
        for group in PRIMARY_GROUPS:
            values = [
                row for row in candidates
                if str(row.get("group")) == group
            ]
            short = group_short(group)
            summary[f"{short}_n"] = len(values)
            if values:
                accuracy = float(
                    np.mean([
                        bool(row.get("intervention_correct"))
                        for row in values
                    ])
                )
                group_rates[short] = accuracy
                summary[f"{short}_accuracy"] = accuracy
            else:
                group_rates[short] = float("nan")
                summary[f"{short}_accuracy"] = float("nan")

        b_repair = group_rates["B"]
        a_damage = (
            1.0 - group_rates["A"]
            if np.isfinite(group_rates["A"])
            else float("nan")
        )
        c_damage = (
            1.0 - group_rates["C"]
            if np.isfinite(group_rates["C"])
            else float("nan")
        )

        summary["B_repair_rate"] = b_repair
        summary["A_damage_rate"] = a_damage
        summary["C_damage_rate"] = c_damage
        summary["net_repair"] = (
            b_repair - a_damage
            if np.isfinite(b_repair) and np.isfinite(a_damage)
            else float("nan")
        )
        summary["conservative_net_repair"] = (
            b_repair - 0.5 * (a_damage + c_damage)
            if all(np.isfinite(x) for x in (b_repair, a_damage, c_damage))
            else float("nan")
        )
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            -float(row.get("conservative_net_repair", -np.inf))
            if np.isfinite(float(row.get("conservative_net_repair", np.nan)))
            else np.inf,
            -float(row.get("net_repair", -np.inf))
            if np.isfinite(float(row.get("net_repair", np.nan)))
            else np.inf,
        )
    )
    return summaries


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 138)
    print("INTERVENTION SUMMARY")
    print("=" * 138)
    header = (
        f"{'Config':58s} {'N':>5s} {'Parse':>7s} "
        f"{'B repair':>9s} {'A damage':>9s} {'C damage':>9s} "
        f"{'Net':>8s} {'ConsNet':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['config_id'])[:58]:58s} "
            f"{int(row['n']):5d} "
            f"{float(row['parse_rate']):7.3f} "
            f"{float(row.get('B_repair_rate', np.nan)):9.3f} "
            f"{float(row.get('A_damage_rate', np.nan)):9.3f} "
            f"{float(row.get('C_damage_rate', np.nan)):9.3f} "
            f"{float(row.get('net_repair', np.nan)):8.3f} "
            f"{float(row.get('conservative_net_repair', np.nan)):8.3f}"
        )


# ---------------------------------------------------------------------------
# Main model loop
# ---------------------------------------------------------------------------

def run_model(
    *,
    args: argparse.Namespace,
    core: Any,
    backend_module: Any,
    model_name: str,
    selected_groups: Sequence[str],
    experiments: Sequence[str],
    object_alphas: Sequence[float],
    mlp_betas: Sequence[float],
    source_presets: Sequence[Tuple[float, float, float]],
) -> None:
    input_root = Path(args.input_root)
    model_input_root = input_root / model_name
    metadata_path = (
        model_input_root
        / "pass2_transfer_trace"
        / "sample_metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    prior_rows = cap_rows(
        read_jsonl(metadata_path),
        selected_groups=selected_groups,
        max_per_group=args.max_per_group,
        seed=args.seed,
        sid=args.sid,
    )
    if not prior_rows:
        raise RuntimeError(f"No selected samples for {model_name}.")

    record_rows, audit = backend_module.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {
        int(record.sid): record
        for record in record_rows
    }

    prompt_path = core.resolve_prompt_path(args)
    prompt_rows = core.load_standard_prompts(prompt_path)

    selected_sids = [int(row["sid"]) for row in prior_rows]
    missing = [
        sid for sid in selected_sids
        if sid not in record_by_sid or sid not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Missing records/prompts for {len(missing)} SIDs; first={missing[:10]}"
        )

    if model_name not in backend_module.SPECS:
        raise ValueError(
            f"Unknown model alias {model_name!r}; "
            f"available={sorted(backend_module.SPECS)}"
        )
    spec = backend_module.SPECS[model_name]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {spec.model_class}"
        )

    print("\n" + "=" * 138)
    print(f"LOADING MODEL: {model_name} -> {spec.repo_id}")
    print("=" * 138)

    load_kwargs: Dict[str, Any] = {
        "dtype": core.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }
    model = model_cls.from_pretrained(
        spec.repo_id,
        **load_kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.configure_processor(model, processor)

    device = torch.device(args.device)
    layers, layers_path = core.resolve_decoder_layers(model)
    n_layers = len(layers)

    object_layers = resolve_window(
        model_name=model_name,
        experiment="object_amp",
        requested=args.object_layers,
        n_layers=n_layers,
    )
    mlp_layers = resolve_window(
        model_name=model_name,
        experiment="mlp_scale",
        requested=args.mlp_layers,
        n_layers=n_layers,
    )
    source_layers = resolve_window(
        model_name=model_name,
        experiment="source_reweight",
        requested=args.source_layers,
        n_layers=n_layers,
    )
    selected_heads = parse_selected_heads(
        args.selected_heads,
        n_layers,
    )

    configs = build_configs(
        experiments=experiments,
        object_layers=object_layers,
        mlp_layers=mlp_layers,
        source_layers=source_layers,
        object_alphas=object_alphas,
        mlp_betas=mlp_betas,
        source_presets=source_presets,
    )

    print(f"Decoder layers: {layers_path} ({n_layers})")
    print(f"object_amp layers:      {object_layers}")
    print(f"mlp_scale layers:       {mlp_layers}")
    print(f"source_reweight layers: {source_layers}")
    print(f"configs: {len(configs)}")
    print(
        "groups: "
        + ", ".join(
            f"{group_short(group)}={sum(str(row['group']) == group for row in prior_rows)}"
            for group in selected_groups
        )
    )

    attention_union = sorted(
        set(object_layers if "object_amp" in experiments else [])
        | set(source_layers if "source_reweight" in experiments else [])
    )
    mlp_union = sorted(
        set(mlp_layers if "mlp_scale" in experiments else [])
    )

    manager = RepairInterventionManager(
        model=model,
        layers=layers,
        attention_layers=attention_union,
        mlp_layers=mlp_union,
        selected_heads=selected_heads,
    )

    output_dir = Path(args.output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_output = output_dir / "sample_results.jsonl"
    errors_output = output_dir / "errors.jsonl"
    summary_output = output_dir / "summary.csv"
    config_output = output_dir / "run_config.json"

    if args.overwrite:
        for path in (
            sample_output,
            errors_output,
            summary_output,
            config_output,
        ):
            if path.exists():
                path.unlink()

    if (
        (sample_output.exists() or summary_output.exists())
        and not args.resume
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Results already exist in {output_dir}. "
            "Pass --resume or --overwrite."
        )

    completed = completed_keys(sample_output) if args.resume else set()

    config_output.write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "model": model_name,
                "repo_id": spec.repo_id,
                "dataset": args.dataset,
                "prompt_jsonl": args.prompt_jsonl,
                "groups": list(selected_groups),
                "sample_count": len(prior_rows),
                "object_layers": object_layers,
                "mlp_layers": mlp_layers,
                "source_layers": source_layers,
                "selected_heads": (
                    {
                        str(layer): sorted(heads)
                        for layer, heads in selected_heads.items()
                    }
                    if selected_heads is not None
                    else None
                ),
                "configs": [config.as_dict() for config in configs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_pending = sum(
        1
        for row in prior_rows
        for config in configs
        if (int(row["sid"]), config.config_id) not in completed
    )
    progress = tqdm(
        total=total_pending,
        desc=f"repair:{model_name}",
        unit="generation",
        dynamic_ncols=True,
    )

    processed_samples = 0
    started = time.time()

    try:
        for prior in prior_rows:
            sid = int(prior["sid"])
            record = record_by_sid[sid]
            prompt_row = prompt_rows[sid]

            image = None
            batch = None
            try:
                image = core.record_image(record)
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Invalid GT for sid={sid}: {prompt_row['answer_raw']!r}"
                    )

                batch = core.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                batch = move_batch_to_device(batch, device)

                prompt_spec = build_prompt_position_spec(
                    model=model,
                    tokenizer=processor.tokenizer,
                    input_ids=batch["input_ids"],
                    subject=subject,
                    reference=reference,
                )

                if args.verify_baseline:
                    baseline_text, baseline_prediction = generate_relation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=args.max_new_tokens,
                        need_attentions=False,
                    )
                else:
                    baseline_text = str(
                        prior.get("generation_text")
                        or prior.get("baseline_generated_text")
                        or ""
                    )
                    baseline_prediction = normalize_relation(
                        prior.get("baseline_prediction")
                    )

                for config in configs:
                    key = (sid, config.config_id)
                    if key in completed:
                        continue

                    manager.configure(
                        config=config,
                        prompt_spec=prompt_spec,
                    )
                    try:
                        generated_text, prediction = generate_relation(
                            model=model,
                            processor=processor,
                            batch=batch,
                            max_new_tokens=args.max_new_tokens,
                            need_attentions=(
                                config.experiment
                                in {"object_amp", "source_reweight"}
                            ),
                        )

                        row = {
                            "model": model_name,
                            "sid": sid,
                            "group": str(prior["group"]),
                            "group_short": group_short(str(prior["group"])),
                            "gt": gt,
                            "subject": subject,
                            "reference": reference,
                            "baseline_prediction": baseline_prediction,
                            "baseline_text": baseline_text,
                            **config.as_dict(),
                            "intervention_prediction": prediction,
                            "intervention_text": generated_text,
                            "parsed": prediction in RELATIONS,
                            "intervention_correct": prediction == gt,
                            "repaired": (
                                str(prior["group"]) == GROUP_B
                                and prediction == gt
                            ),
                            "damaged": (
                                str(prior["group"]) in {GROUP_A, GROUP_C}
                                and prediction != gt
                            ),
                        }
                        append_jsonl(sample_output, row)
                        completed.add(key)

                    except Exception as exc:
                        append_jsonl(
                            errors_output,
                            {
                                "model": model_name,
                                "sid": sid,
                                "group": prior.get("group"),
                                "config_id": config.config_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback_tail": traceback.format_exc().splitlines()[-20:],
                            },
                        )
                        raise
                    finally:
                        manager.reset()

                    progress.update(1)

                processed_samples += 1
                if (
                    args.print_every > 0
                    and processed_samples % args.print_every == 0
                ):
                    elapsed = time.time() - started
                    tqdm.write(
                        f"[{model_name}] samples={processed_samples}/{len(prior_rows)} "
                        f"elapsed={elapsed / 60:.1f} min"
                    )

                if (
                    args.empty_cache_every > 0
                    and processed_samples % args.empty_cache_every == 0
                ):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            finally:
                manager.reset()
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

    finally:
        progress.close()
        manager.close()

    rows = read_jsonl(sample_output)
    summary_rows = aggregate_results(rows, configs)
    write_csv(summary_output, summary_rows)
    print_summary(summary_rows)

    print(f"\nSample results: {sample_output}")
    print(f"Summary:        {summary_output}")
    if errors_output.exists() and errors_output.stat().st_size:
        print(f"Errors:         {errors_output}")

    del model, processor, layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = parse_models(args.models)
    selected_groups = parse_groups(args.groups)
    experiments = parse_experiments(args.experiments)
    object_alphas = parse_float_list(args.object_alphas)
    mlp_betas = parse_float_list(args.mlp_betas)
    source_presets = parse_source_presets(args.source_presets)

    core = import_core(args.core_module)
    backend_module = core.import_two_object_module()

    print("=" * 138)
    print("THREE TRAINING-FREE SPATIAL REPAIR EXPERIMENTS")
    print("=" * 138)
    print(f"models={models}")
    print(f"experiments={experiments}")
    print(f"groups={[group_short(x) for x in selected_groups]}")
    print(
        "Interventions use only the current sample's internal routes/updates; "
        "GT labels are used only for evaluation."
    )

    completed_models = 0
    failures: List[Tuple[str, str]] = []

    for model_name in models:
        try:
            run_model(
                args=args,
                core=core,
                backend_module=backend_module,
                model_name=model_name,
                selected_groups=selected_groups,
                experiments=experiments,
                object_alphas=object_alphas,
                mlp_betas=mlp_betas,
                source_presets=source_presets,
            )
            completed_models += 1
        except Exception as exc:
            failures.append(
                (model_name, f"{type(exc).__name__}: {exc}")
            )
            print(
                f"\n[ERROR] {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    print("\n" + "=" * 138)
    print(f"COMPLETE: {completed_models}/{len(models)} models")
    for model_name, error in failures:
        print(f"  failed {model_name}: {error}")

    if completed_models == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
