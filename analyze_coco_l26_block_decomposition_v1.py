#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L26 block decomposition for COCO two-object spatial reasoning.

NO intervention is performed.

Purpose
=======
The previous failure-localization run found approximately:

    direction                ~80%
    L26 all-head object->last write ~80%
    L26 prompt-last block state     ~65%

This script isolates L26 and asks:

    If the natural L26 object->prompt-last contribution already carries the
    correct relation, what happens when it is combined with:
        (a) the incoming prompt-last residual,
        (b) the rest of attention,
        (c) the L26 MLP?

Natural Qwen/LLaMA decoder-block trajectory
===========================================
At prompt-last:

    x        = L26 block input residual
    a        = L26 attention output after o_proj
    r_attn   = x + a
    m        = L26 MLP output
    y        = r_attn + m        (L26 block output)

Attention source decomposition
==============================
Using the already-tested eager-attention replay:

    a = c_obj + c_visual + c_other_text

where every c_* is reconstructed as

    c_group = sum_h W_O^h sum_{s in group} A_h[last,s] V_h[s]

Groups:
    object       = subject/reference TEXT-token spans in the question
    visual       = all visual tokens
    other_text   = all remaining source positions, including instruction,
                   options/special/prompt-last tokens

The script also computes:

    c_nonobj       = c_visual + c_other_text
    x_plus_obj     = x + c_obj
    x_plus_obj_vis = x + c_obj + c_visual

IMPORTANT:
    x_plus_obj and x_plus_obj_vis are ALGEBRAIC PARTIAL RECONSTRUCTIONS.
    They are not natural temporal states of the unmodified model.
    The natural states are x, a, r_attn, m, y.

Receiver shortlist
==================
For comparison, the script also reconstructs:

    c_obj_selected

using the default L26 receiver shortlist:
    L26H4,L26H2,L26H6,L26H0

but all mechanistic "receiver-correct" conditional analyses default to
c_obj_all, so an incomplete shortlist does not create a fake bottleneck.

Train/eval
==========
To exactly reuse the same held-out population as the recent direction/failure
experiments, --direction-dir is used only for:
    sample SIDs, labels, and deterministic stratified train/eval split.

No direction vectors are used in the L26 decomposition itself.

For every vector component/state, a TRAIN-only four-class cosine codebook is fit.
Held-out accuracy is reported first. Sample-level transitions should only be trusted
for components whose held-out probe accuracy is sufficiently high.

Primary conditional cohort
==========================
    baseline greedy generation WRONG
    AND
    c_obj_all probe == GT

For this cohort the script reports how often GT remains readable in:

    x
    c_obj_all
    c_visual
    c_other_text
    c_nonobj
    a
    x_plus_obj
    x_plus_obj_vis
    r_attn = x+a
    m
    y

It also reports conditional transitions:

    c_obj_all correct -> x_plus_obj correct
    c_obj_all correct -> a correct
    c_obj_all correct -> r_attn correct
    r_attn correct     -> y correct

Interpretation examples
=======================
1) c_obj high, x_plus_obj sharply lower:
       incoming residual x conflicts with/cancels receiver evidence.

2) x_plus_obj high, r_attn sharply lower:
       non-object attention is the main candidate interference.

3) r_attn high, y sharply lower:
       L26 MLP is the main candidate interference.

4) selected c_obj low but all-head c_obj high:
       receiver shortlist is incomplete.

These are diagnostic representation results, not yet causal proof.  If a candidate
interference source is identified, the NEXT experiment should remove/scale only
that component and measure true generation W->C / C->W.

Required repository modules
===========================
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_receiver_qkv_v1.py
    analyze_spatial_storage_transport_utilization_v3.py
    analyze_coco_flip_attention_spatial_vectors_v1.py
    extract_two_object_relation_states.py

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_coco_l26_block_decomposition_v1.py \
  --model qwen-3b \
  --direction-dir output/qwen3b_coco_head_object_residual_direction_probe \
  --layer 26 \
  --receiver-heads "26:4,26:2,26:6,26:0" \
  --train-ratio 0.15 \
  --seed 17 \
  --device cuda:0 \
  --output-dir output/qwen3b_l26_block_decomposition_v1 \
  --overwrite

Outputs
=======
config.json
component_accuracy.csv
baseline_eval.csv
conditional_receiver_correct.csv
conditional_summary.csv
generation_group_summary.csv
source_strength_summary.csv
decomposition_closure.csv
sample_vectors/<sid>.npz
report.txt
errors.jsonl
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
import shutil
import traceback
from collections import Counter, defaultdict
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

SCRIPT_VERSION = "coco-l26-block-decomposition-v1"
RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-12

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)
DEFAULT_RECEIVERS = "26:4,26:2,26:6,26:0"

# Order used in printed tables.
COMPONENT_ORDER = (
    "x",
    "c_obj_selected",
    "c_obj_all",
    "c_visual",
    "c_other_text",
    "c_nonobj",
    "a",
    "x_plus_obj",
    "x_plus_obj_vis",
    "r_attn",
    "m",
    "y",
)

NATURAL_STATE = {
    "x": True,
    "c_obj_selected": False,
    "c_obj_all": False,
    "c_visual": False,
    "c_other_text": False,
    "c_nonobj": False,
    "a": False,             # natural module output/contribution, not residual state
    "x_plus_obj": False,    # algebraic partial reconstruction
    "x_plus_obj_vis": False,# algebraic partial reconstruction
    "r_attn": True,
    "m": False,             # natural module output/contribution
    "y": True,
}


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))

    p.add_argument(
        "--direction-dir",
        required=True,
        help=(
            "Existing direction probe directory. Used ONLY to recover the same "
            "sample SIDs/labels/split; direction vectors are not analyzed."
        ),
    )
    p.add_argument("--layer", type=int, default=26)
    p.add_argument("--receiver-heads", default=DEFAULT_RECEIVERS)
    p.add_argument("--trace-layer-chunk", type=int, default=1)

    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--replay-warning-threshold",
        type=float,
        default=1e-3,
        help="Warn in tables when attention replay relative error exceeds this.",
    )
    p.add_argument(
        "--closure-warning-threshold",
        type=float,
        default=1e-3,
        help="Warn when y != x+a+m or source writes != a beyond this relative error.",
    )

    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--receiver-module",
        default="analyze_coco_receiver_qkv_v1",
    )
    p.add_argument(
        "--v3-module",
        default="analyze_spatial_storage_transport_utilization_v3",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument("--empty-cache-every", type=int, default=10)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "left": "left", "leftward": "left",
        "right": "right", "rightward": "right",
        "above": "above", "over": "above", "top": "above",
        "on top": "above", "on": "above",
        "below": "below", "under": "below", "underneath": "below",
        "beneath": "below", "bottom": "below",
    }
    if text in aliases:
        return aliases[text]
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(below|under|underneath|beneath|bottom)\b", "below"),
        (r"\b(above|over|on top|top)\b", "above"),
        (r"\bon\b", "above"),
    ]
    hits = []
    for pattern, relation in patterns:
        match = re.search(pattern, text)
        if match:
            hits.append((match.start(), relation))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def parse_head(text: str) -> Tuple[int, int]:
    value = str(text).strip().upper().replace("L", "").replace("H", ":")
    while "::" in value:
        value = value.replace("::", ":")
    if ":" not in value:
        raise ValueError(f"Bad head spec {text!r}")
    a, b = value.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        head = parse_head(item)
        if head not in seen:
            seen.add(head)
            out.append(head)
    return out


def hname(head: Tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]:02d}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def safe_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.mean(xs)) if xs else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.median(xs)) if xs else float("nan")


def relative_error(actual: np.ndarray, reconstructed: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    reconstructed = np.asarray(reconstructed, dtype=np.float64).reshape(-1)
    return float(
        np.linalg.norm(actual - reconstructed)
        / max(float(np.linalg.norm(actual)), EPS)
    )


def clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, name):
            setattr(cfg, name, None)


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for relation in RELATIONS:
        ids = set()
        for text in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))
        if not ids:
            token_ids = tokenizer.encode(
                " " + relation, add_special_tokens=False
            )
            if not token_ids:
                raise RuntimeError(f"No token variant for {relation}")
            ids.add(int(token_ids[-1]))
        out[relation] = sorted(ids)
    return out


def stratified_split(
    labels: np.ndarray,
    ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not 0 < ratio < 1:
        raise ValueError("--train-ratio must be in (0,1)")
    rng = random.Random(seed)
    train: List[int] = []
    eval_: List[int] = []
    for relation in RELATIONS:
        ids = [
            index
            for index, value in enumerate(labels)
            if str(value) == relation
        ]
        rng.shuffle(ids)
        if len(ids) < 2:
            raise RuntimeError(f"Not enough samples for {relation}")
        n_train = max(1, int(round(len(ids) * ratio)))
        n_train = min(n_train, len(ids) - 1)
        train.extend(ids[:n_train])
        eval_.extend(ids[n_train:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def stratified_limit(
    indices: Sequence[int],
    labels: np.ndarray,
    limit: int,
    seed: int,
) -> List[int]:
    indices = list(map(int, indices))
    if limit <= 0 or len(indices) <= limit:
        return indices
    groups: Dict[str, List[int]] = defaultdict(list)
    for index in indices:
        groups[str(labels[index])].append(index)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    cursors = {relation: 0 for relation in RELATIONS}
    out = []
    while len(out) < limit:
        moved = False
        for relation in RELATIONS:
            group = groups.get(relation, [])
            cursor = cursors[relation]
            if cursor < len(group) and len(out) < limit:
                out.append(group[cursor])
                cursors[relation] += 1
                moved = True
        if not moved:
            break
    return out


# =============================================================================
# Codebook
# =============================================================================

class CosineCodebook:
    def __init__(self, center: np.ndarray, directions: np.ndarray):
        self.center = np.asarray(center, dtype=np.float32)
        self.directions = np.asarray(directions, dtype=np.float32)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "CosineCodebook":
        X = np.asarray(X, dtype=np.float32)
        center = X.mean(axis=0)
        centered = X - center
        directions = []
        for relation in RELATIONS:
            mask = y == relation
            if not np.any(mask):
                raise RuntimeError(f"No train samples for {relation}")
            vector = centered[mask].mean(axis=0)
            vector = vector / max(float(np.linalg.norm(vector)), EPS)
            directions.append(vector)
        return cls(center, np.stack(directions, axis=0))

    def scores(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32) - self.center
        vector = vector / max(float(np.linalg.norm(vector)), EPS)
        return (vector @ self.directions.T).astype(np.float32)

    def predict(self, vector: np.ndarray) -> Tuple[str, np.ndarray]:
        scores = self.scores(vector)
        return RELATIONS[int(np.argmax(scores))], scores


def gt_margin(scores: np.ndarray, gt: str) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    return float(scores[RID[gt]] - scores[RID[OPPOSITE[gt]]])


# =============================================================================
# L26 capture
# =============================================================================

def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and torch.is_tensor(output[0])
    ):
        return output[0]
    raise TypeError(f"Could not find tensor in {type(output).__name__}")


def resolve_mlp(layer: Any) -> torch.nn.Module:
    for name in ("mlp", "feed_forward", "ffn"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise RuntimeError(f"{type(layer).__name__} has no MLP module")


class L26NaturalCapture:
    """
    Capture block input x and MLP output m at prompt-last during the REAL model
    forward. Attention output a and block output y are taken from the validated
    attention replay trace.
    """

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
        prompt_last: int,
    ) -> None:
        self.layer = layer
        self.prompt_last = int(prompt_last)
        self.mlp = resolve_mlp(layer)
        self.x: Optional[torch.Tensor] = None
        self.m: Optional[torch.Tensor] = None
        self.events: Counter = Counter()
        self.handles: List[Any] = []

    def __enter__(self) -> "L26NaturalCapture":
        def block_pre(_module: Any, inputs: Tuple[Any, ...]) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("L26 block pre-hook missing hidden_states")
            hidden = inputs[0]
            if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
                raise RuntimeError(
                    f"L26 block input expected [1,S,D], got {tuple(hidden.shape)}"
                )
            if int(hidden.shape[1]) <= self.prompt_last:
                return
            self.x = (
                hidden[0, self.prompt_last]
                .detach()
                .float()
                .cpu()
            )
            self.events["x"] += 1

        def mlp_hook(
            _module: Any,
            _inputs: Tuple[Any, ...],
            output: Any,
        ) -> Any:
            hidden = first_tensor(output)
            if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
                raise RuntimeError(
                    f"L26 MLP output expected [1,S,D], got {tuple(hidden.shape)}"
                )
            if int(hidden.shape[1]) <= self.prompt_last:
                return output
            self.m = (
                hidden[0, self.prompt_last]
                .detach()
                .float()
                .cpu()
            )
            self.events["m"] += 1
            return output

        self.handles.append(
            self.layer.register_forward_pre_hook(block_pre)
        )
        self.handles.append(
            self.mlp.register_forward_hook(mlp_hook)
        )
        return self

    def validate(self) -> None:
        if self.x is None or self.events["x"] < 1:
            raise RuntimeError("Missing L26 block input x")
        if self.m is None or self.events["m"] < 1:
            raise RuntimeError("Missing L26 MLP output m")

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# =============================================================================
# Attention source contributions
# =============================================================================

def trace_target_index(trace: Any, prompt_last: int) -> int:
    lookup = {
        int(global_position): local_index
        for local_index, global_position in enumerate(trace.target_positions)
    }
    if int(prompt_last) not in lookup:
        raise RuntimeError(
            f"prompt_last={prompt_last} absent from trace targets "
            f"{trace.target_positions}"
        )
    return int(lookup[int(prompt_last)])


def group_head_writes(
    *,
    trace: Any,
    prompt_last: int,
    source_positions: Sequence[int],
) -> torch.Tensor:
    """
    Return post-W_O source contribution per query head: [H,D].
    """
    local = trace_target_index(trace, prompt_last)
    source_positions = sorted(set(map(int, source_positions)))
    H = int(trace.attention_weights.shape[0])
    D = int(trace.o_proj_weight.shape[0])

    if not source_positions:
        return torch.zeros((H, D), dtype=torch.float32)

    source = torch.as_tensor(source_positions, dtype=torch.long)
    if int(source.max()) >= int(trace.value_states.shape[1]):
        raise RuntimeError(
            f"Source position {int(source.max())} exceeds trace source length "
            f"{int(trace.value_states.shape[1])}"
        )

    weights = (
        trace.attention_weights[:, local, :]
        .index_select(1, source)
        .float()
    )                                           # [H,Sg]
    values = (
        trace.value_states
        .index_select(1, source)
        .float()
    )                                           # [H,Sg,Dh]
    pre = torch.einsum("hs,hsd->hd", weights, values)
    weight = trace.o_proj_weight.float()        # [D,H,Dh]
    post = torch.einsum("hd,ohd->ho", pre, weight)
    return post


def source_decomposition(
    *,
    trace: Any,
    prompt_last: int,
    object_positions: Sequence[int],
    visual_positions: Sequence[int],
    selected_heads: Sequence[int],
) -> Dict[str, Any]:
    source_length = int(trace.value_states.shape[1])
    all_positions = set(range(source_length))
    object_set = set(map(int, object_positions))
    visual_set = set(map(int, visual_positions))

    overlap = object_set & visual_set
    if overlap:
        raise RuntimeError(
            f"Object text positions unexpectedly overlap visual positions: "
            f"{sorted(overlap)[:20]}"
        )

    other_text = sorted(all_positions - object_set - visual_set)
    object_positions = sorted(object_set)
    visual_positions = sorted(visual_set)

    obj_heads = group_head_writes(
        trace=trace,
        prompt_last=prompt_last,
        source_positions=object_positions,
    )
    visual_heads = group_head_writes(
        trace=trace,
        prompt_last=prompt_last,
        source_positions=visual_positions,
    )
    other_heads = group_head_writes(
        trace=trace,
        prompt_last=prompt_last,
        source_positions=other_text,
    )

    c_obj = obj_heads.sum(dim=0)
    c_visual = visual_heads.sum(dim=0)
    c_other = other_heads.sum(dim=0)
    c_nonobj = c_visual + c_other
    c_sources_total = c_obj + c_nonobj

    selected = sorted(set(map(int, selected_heads)))
    H = int(obj_heads.shape[0])
    bad = [head for head in selected if not 0 <= head < H]
    if bad:
        raise RuntimeError(f"Selected query heads outside 0..{H-1}: {bad}")

    if selected:
        index = torch.as_tensor(selected, dtype=torch.long)
        c_obj_selected = obj_heads.index_select(0, index).sum(dim=0)
    else:
        c_obj_selected = torch.zeros_like(c_obj)

    local = trace_target_index(trace, prompt_last)
    attention_actual = trace.attention_output[local].float()

    return {
        "c_obj_selected": c_obj_selected.numpy().astype(np.float32),
        "c_obj_all": c_obj.numpy().astype(np.float32),
        "c_visual": c_visual.numpy().astype(np.float32),
        "c_other_text": c_other.numpy().astype(np.float32),
        "c_nonobj": c_nonobj.numpy().astype(np.float32),
        "source_sum": c_sources_total.numpy().astype(np.float32),
        "a": attention_actual.numpy().astype(np.float32),
        "object_positions": object_positions,
        "visual_positions": visual_positions,
        "other_text_positions": other_text,
        "object_mass_all": float(
            trace.attention_weights[:, local, :]
            .index_select(
                1,
                torch.as_tensor(object_positions, dtype=torch.long),
            )
            .float()
            .sum()
        ) if object_positions else 0.0,
        "visual_mass_all": float(
            trace.attention_weights[:, local, :]
            .index_select(
                1,
                torch.as_tensor(visual_positions, dtype=torch.long),
            )
            .float()
            .sum()
        ) if visual_positions else 0.0,
    }


# =============================================================================
# Batch / generation
# =============================================================================

def build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Any:
    rendered = probe.build_chat_prompt(
        processor,
        question,
        True,
    )
    return probe.process_inputs(
        processor,
        rendered,
        image,
        device,
    )


@torch.inference_mode()
def greedy_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    input_length = int(batch["input_ids"].shape[1])
    output_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    text = processor.tokenizer.decode(
        output_ids[0, input_length:],
        skip_special_tokens=True,
    ).strip()
    del output_ids
    return normalize_relation(text), text


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.trace_layer_chunk < 1:
        raise ValueError("--trace-layer-chunk must be >=1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = output_dir / "sample_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    # Companion modules.
    probe = importlib.import_module(args.probe_module)
    receiver = importlib.import_module(args.receiver_module)
    v3 = importlib.import_module(args.v3_module)
    attention_helper = importlib.import_module(
        args.attention_helper_module
    )
    base = probe.base

    # Reuse exact SID/label population from direction cache.
    direction_dir = Path(args.direction_dir)
    npz_path = direction_dir / "relation_vectors.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        sids = np.asarray(data["sample_index"], dtype=np.int64)
        labels = np.asarray(
            [normalize_relation(x) for x in data["relation"].tolist()],
            dtype=object,
        )

    valid = np.asarray(
        [label in RELATIONS for label in labels],
        dtype=bool,
    )
    sids = sids[valid]
    labels = labels[valid]

    if args.max_samples > 0 and args.max_samples < len(labels):
        keep = stratified_limit(
            range(len(labels)),
            labels,
            args.max_samples,
            args.seed,
        )
        sids = sids[keep]
        labels = labels[keep]

    train_idx, eval_idx = stratified_split(
        labels,
        args.train_ratio,
        args.seed,
    )
    if args.eval_max_samples > 0:
        eval_idx = stratified_limit(
            eval_idx,
            labels,
            args.eval_max_samples,
            args.seed + 1,
        )

    train_sids = [int(sids[i]) for i in train_idx]
    eval_sids = [int(sids[i]) for i in eval_idx]
    label_by_sid = {
        int(sids[i]): str(labels[i])
        for i in range(len(sids))
    }

    # Dataset.
    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {
        int(record.sid): record
        for record in records
    }
    train_sids = [
        sid for sid in train_sids
        if sid in record_by_sid
    ]
    eval_sids = [
        sid for sid in eval_sids
        if sid in record_by_sid
    ]
    if not train_sids or not eval_sids:
        raise RuntimeError("No train/eval SIDs overlap dataset")

    # Selected L26 receiver query heads.
    receiver_specs = parse_heads(args.receiver_heads)
    selected_query_heads = [
        head
        for layer, head in receiver_specs
        if int(layer) == int(args.layer)
    ]
    if not selected_query_heads:
        raise RuntimeError(
            f"No --receiver-heads belong to L{args.layer}"
        )

    spec = base.SPECS[args.model]
    model_class = getattr(transformers, spec.model_class)
    load_kwargs = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }

    model = None
    processor = None

    # sid -> component vectors/meta
    extracted: Dict[int, Dict[str, Any]] = {}
    basic_rows: List[Dict[str, Any]] = []

    try:
        print(
            f"Loading {args.model}: {spec.repo_id}",
            flush=True,
        )
        model = model_class.from_pretrained(
            spec.repo_id,
            **load_kwargs,
        )
        model.eval()
        clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        decoder_layers, decoder_path = probe.resolve_decoder_layers(model)
        if not 0 <= args.layer < len(decoder_layers):
            raise ValueError(
                f"L{args.layer} outside decoder 0..{len(decoder_layers)-1}"
            )

        layer_module = decoder_layers[args.layer]
        attention = attention_helper.resolve_self_attention(layer_module)
        shape = receiver.resolve_attention_shape(attention)
        for head in selected_query_heads:
            if not 0 <= head < shape.n_query_heads:
                raise ValueError(
                    f"L{args.layer}H{head} outside query heads "
                    f"0..{shape.n_query_heads-1}"
                )

        relation_token_map = relation_token_variants(
            processor.tokenizer
        )

        print("\n" + "=" * 132)
        print(f"L{args.layer} BLOCK DECOMPOSITION")
        print("=" * 132)
        print(
            "selected receiver heads:",
            ", ".join(
                hname((args.layer, head))
                for head in selected_query_heads
            ),
        )
        print("train/eval:", len(train_sids), "/", len(eval_sids))
        print("prompt:", args.prompt_template)
        print(
            "natural trajectory:",
            f"x -> r_attn=x+a -> y=r_attn+m",
        )
        print(
            "attention sources:",
            "a = c_obj + c_visual + c_other_text",
        )
        print("=" * 132, flush=True)

        train_set = set(train_sids)
        ordered_sids = train_sids + eval_sids

        for sample_index, sid in enumerate(
            tqdm(ordered_sids, desc=f"L{args.layer}-decompose"),
            start=1,
        ):
            image = None
            batch = None
            try:
                record = record_by_sid[sid]
                gt = label_by_sid[sid]

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )
                image = Image.open(record.image_path).convert("RGB")
                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(x)
                    for x in batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                prompt_last = len(input_ids) - 1

                a_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.subject),
                )
                b_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.reference),
                )
                object_positions = sorted(
                    set(map(int, a_positions + b_positions))
                )

                visual_positions = base.resolve_visual_indices(
                    model,
                    processor,
                    dict(batch),
                    input_ids,
                )
                visual_positions = sorted(
                    set(map(int, visual_positions))
                )

                # Actual model forward + tested L26 attention replay.
                with L26NaturalCapture(
                    layer=layer_module,
                    prompt_last=prompt_last,
                ) as capture:
                    baseline, traces = v3.trace_prompt_chunks(
                        attention_helper=attention_helper,
                        model=model,
                        batch=batch,
                        relation_token_map=relation_token_map,
                        decoder_layers=decoder_layers,
                        layers=[args.layer],
                        target_positions=[prompt_last],
                        chunk_size=args.trace_layer_chunk,
                    )
                    capture.validate()

                trace = traces[args.layer]
                source = source_decomposition(
                    trace=trace,
                    prompt_last=prompt_last,
                    object_positions=object_positions,
                    visual_positions=visual_positions,
                    selected_heads=selected_query_heads,
                )

                x = capture.x.numpy().astype(np.float32)
                m = capture.m.numpy().astype(np.float32)

                local = trace_target_index(trace, prompt_last)
                y = (
                    trace.block_output[local]
                    .numpy()
                    .astype(np.float32)
                )
                a = np.asarray(
                    source["a"],
                    dtype=np.float32,
                )
                c_obj = np.asarray(
                    source["c_obj_all"],
                    dtype=np.float32,
                )
                c_visual = np.asarray(
                    source["c_visual"],
                    dtype=np.float32,
                )
                c_other = np.asarray(
                    source["c_other_text"],
                    dtype=np.float32,
                )

                r_attn = (x + a).astype(np.float32)
                x_plus_obj = (x + c_obj).astype(np.float32)
                x_plus_obj_vis = (
                    x + c_obj + c_visual
                ).astype(np.float32)

                components: Dict[str, np.ndarray] = {
                    "x": x,
                    "c_obj_selected": np.asarray(
                        source["c_obj_selected"],
                        dtype=np.float32,
                    ),
                    "c_obj_all": c_obj,
                    "c_visual": c_visual,
                    "c_other_text": c_other,
                    "c_nonobj": np.asarray(
                        source["c_nonobj"],
                        dtype=np.float32,
                    ),
                    "a": a,
                    "x_plus_obj": x_plus_obj,
                    "x_plus_obj_vis": x_plus_obj_vis,
                    "r_attn": r_attn,
                    "m": m,
                    "y": y,
                }

                source_closure = relative_error(
                    a,
                    source["source_sum"],
                )
                block_closure = relative_error(
                    y,
                    x + a + m,
                )
                replay_error = float(
                    trace.replay_relative_error
                )

                generation_prediction = None
                generation_text = None
                if sid not in train_set:
                    (
                        generation_prediction,
                        generation_text,
                    ) = greedy_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=args.max_new_tokens,
                    )

                native_prediction = str(baseline["prediction"])
                native_scores = np.asarray(
                    [
                        float(baseline["scores"][relation])
                        for relation in RELATIONS
                    ],
                    dtype=np.float32,
                )

                extracted[sid] = {
                    "gt": gt,
                    "components": components,
                    "native_prediction": native_prediction,
                    "native_scores": native_scores,
                    "generation_prediction": generation_prediction,
                    "generation_text": generation_text,
                    "source_closure": source_closure,
                    "block_closure": block_closure,
                    "replay_error": replay_error,
                    "object_mass_all": source["object_mass_all"],
                    "visual_mass_all": source["visual_mass_all"],
                    "object_positions": source["object_positions"],
                    "visual_positions": source["visual_positions"],
                    "other_text_positions": source["other_text_positions"],
                }

                npz_payload = {
                    "sid": np.asarray(sid, dtype=np.int64),
                    "gt_id": np.asarray(RID[gt], dtype=np.int8),
                    "native_scores": native_scores,
                }
                for name, vector in components.items():
                    npz_payload[name] = vector
                np.savez_compressed(
                    vectors_dir / f"{sid}.npz",
                    **npz_payload,
                )

                basic_rows.append({
                    "sid": sid,
                    "split": "train" if sid in train_set else "eval",
                    "gt": gt,
                    "native_prediction": native_prediction,
                    "native_correct": native_prediction == gt,
                    "generation_prediction": generation_prediction,
                    "generation_correct": (
                        generation_prediction == gt
                        if sid not in train_set
                        else None
                    ),
                    "generation_text": generation_text,
                    "attention_replay_relative_error": replay_error,
                    "source_sum_relative_error": source_closure,
                    "block_closure_relative_error": block_closure,
                    "object_mass_all_heads": source["object_mass_all"],
                    "visual_mass_all_heads": source["visual_mass_all"],
                    "x_norm": float(np.linalg.norm(x)),
                    "c_obj_all_norm": float(np.linalg.norm(c_obj)),
                    "c_visual_norm": float(np.linalg.norm(c_visual)),
                    "c_other_text_norm": float(np.linalg.norm(c_other)),
                    "a_norm": float(np.linalg.norm(a)),
                    "m_norm": float(np.linalg.norm(m)),
                    "y_norm": float(np.linalg.norm(y)),
                })

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "extract",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(
            output_dir / "all_samples_basic.csv",
            basic_rows,
        )

        valid_train = [
            sid for sid in train_sids
            if sid in extracted
        ]
        valid_eval = [
            sid for sid in eval_sids
            if sid in extracted
        ]
        if not valid_train or not valid_eval:
            raise RuntimeError("No valid completed train/eval samples")

        # ------------------------------------------------------------------
        # TRAIN-only component codebooks.
        # ------------------------------------------------------------------
        y_train = np.asarray(
            [extracted[sid]["gt"] for sid in valid_train],
            dtype=object,
        )
        codebooks: Dict[str, CosineCodebook] = {}

        for name in COMPONENT_ORDER:
            X_train = np.stack(
                [
                    extracted[sid]["components"][name]
                    for sid in valid_train
                ],
                axis=0,
            )
            codebooks[name] = CosineCodebook.fit(
                X_train,
                y_train,
            )

        # ------------------------------------------------------------------
        # Held-out evaluation.
        # ------------------------------------------------------------------
        component_correct = Counter()
        component_covered = Counter()
        eval_rows: List[Dict[str, Any]] = []

        for sid in valid_eval:
            item = extracted[sid]
            gt = str(item["gt"])

            row: Dict[str, Any] = {
                "sid": sid,
                "gt": gt,
                "native_prediction": item["native_prediction"],
                "native_correct": item["native_prediction"] == gt,
                "native_gt_margin": gt_margin(
                    item["native_scores"],
                    gt,
                ),
                "generation_prediction": item["generation_prediction"],
                "generation_correct": item["generation_prediction"] == gt,
                "generation_text": item["generation_text"],
                "attention_replay_relative_error": item["replay_error"],
                "source_sum_relative_error": item["source_closure"],
                "block_closure_relative_error": item["block_closure"],
                "object_mass_all_heads": item["object_mass_all"],
                "visual_mass_all_heads": item["visual_mass_all"],
            }

            for name in COMPONENT_ORDER:
                prediction, scores = codebooks[name].predict(
                    item["components"][name]
                )
                correct = prediction == gt
                row[f"{name}_pred"] = prediction
                row[f"{name}_correct"] = correct
                row[f"{name}_gt_margin"] = gt_margin(
                    scores,
                    gt,
                )
                row[f"{name}_norm"] = float(
                    np.linalg.norm(
                        item["components"][name]
                    )
                )
                component_covered[name] += 1
                component_correct[name] += int(correct)

            eval_rows.append(row)

        write_csv(
            output_dir / "baseline_eval.csv",
            eval_rows,
        )

        component_accuracy_rows = []
        for name in COMPONENT_ORDER:
            n = component_covered[name]
            component_accuracy_rows.append({
                "component": name,
                "type": (
                    "natural_state"
                    if NATURAL_STATE[name]
                    else "contribution_or_partial_reconstruction"
                ),
                "N_eval": n,
                "accuracy": component_correct[name] / max(n, 1),
            })

        # Native LM and generation appended for context.
        component_accuracy_rows.extend([
            {
                "component": "native_lm_4way",
                "type": "native_readout",
                "N_eval": len(eval_rows),
                "accuracy": safe_mean(
                    float(bool(row["native_correct"]))
                    for row in eval_rows
                ),
            },
            {
                "component": "generation",
                "type": "native_readout",
                "N_eval": len(eval_rows),
                "accuracy": safe_mean(
                    float(bool(row["generation_correct"]))
                    for row in eval_rows
                ),
            },
        ])
        write_csv(
            output_dir / "component_accuracy.csv",
            component_accuracy_rows,
        )

        # ------------------------------------------------------------------
        # Primary cohort:
        # generation wrong AND all-head L26 object write correct.
        # ------------------------------------------------------------------
        generation_wrong = [
            row
            for row in eval_rows
            if not bool(row["generation_correct"])
        ]
        receiver_correct = [
            row
            for row in generation_wrong
            if bool(row["c_obj_all_correct"])
        ]
        receiver_wrong = [
            row
            for row in generation_wrong
            if not bool(row["c_obj_all_correct"])
        ]

        conditional_rows = []
        for row in receiver_correct:
            outrow = dict(row)
            # Key binary transitions.
            outrow["obj_to_xplusobj_retained"] = bool(
                row["x_plus_obj_correct"]
            )
            outrow["obj_to_attn_retained"] = bool(
                row["a_correct"]
            )
            outrow["obj_to_postattn_retained"] = bool(
                row["r_attn_correct"]
            )
            outrow["postattn_to_block_retained"] = bool(
                row["r_attn_correct"]
                and row["y_correct"]
            )
            outrow["postattn_correct_block_wrong"] = bool(
                row["r_attn_correct"]
                and not row["y_correct"]
            )
            conditional_rows.append(outrow)

        write_csv(
            output_dir / "conditional_receiver_correct.csv",
            conditional_rows,
        )

        # ------------------------------------------------------------------
        # Conditional component retention table.
        # ------------------------------------------------------------------
        conditional_summary: List[Dict[str, Any]] = []

        for name in COMPONENT_ORDER:
            conditional_summary.append({
                "cohort": "generation_wrong_and_c_obj_all_correct",
                "component": name,
                "N": len(receiver_correct),
                "accuracy_within_cohort": safe_mean(
                    float(bool(row[f"{name}_correct"]))
                    for row in receiver_correct
                ),
                "mean_gt_margin": safe_mean(
                    row[f"{name}_gt_margin"]
                    for row in receiver_correct
                ),
                "mean_norm": safe_mean(
                    row[f"{name}_norm"]
                    for row in receiver_correct
                ),
            })

        # Explicit transitions that answer the hypothesis.
        transition_specs = [
            (
                "c_obj_all_correct -> x_plus_obj_correct",
                lambda row: bool(row["x_plus_obj_correct"]),
            ),
            (
                "c_obj_all_correct -> a_correct",
                lambda row: bool(row["a_correct"]),
            ),
            (
                "c_obj_all_correct -> r_attn_correct",
                lambda row: bool(row["r_attn_correct"]),
            ),
            (
                "c_obj_all_correct -> y_correct",
                lambda row: bool(row["y_correct"]),
            ),
        ]
        for label, fn in transition_specs:
            conditional_summary.append({
                "cohort": "transition",
                "component": label,
                "N": len(receiver_correct),
                "accuracy_within_cohort": safe_mean(
                    float(fn(row))
                    for row in receiver_correct
                ),
                "mean_gt_margin": float("nan"),
                "mean_norm": float("nan"),
            })

        postattn_correct = [
            row
            for row in receiver_correct
            if bool(row["r_attn_correct"])
        ]
        conditional_summary.append({
            "cohort": "transition",
            "component": "r_attn_correct -> y_correct",
            "N": len(postattn_correct),
            "accuracy_within_cohort": safe_mean(
                float(bool(row["y_correct"]))
                for row in postattn_correct
            ),
            "mean_gt_margin": float("nan"),
            "mean_norm": float("nan"),
        })

        write_csv(
            output_dir / "conditional_summary.csv",
            conditional_summary,
        )

        # ------------------------------------------------------------------
        # Generation-correct vs wrong comparison.
        # ------------------------------------------------------------------
        generation_correct = [
            row
            for row in eval_rows
            if bool(row["generation_correct"])
        ]
        generation_group_rows: List[Dict[str, Any]] = []
        for group_name, group in (
            ("generation_correct", generation_correct),
            ("generation_wrong", generation_wrong),
            (
                "generation_wrong_c_obj_correct",
                receiver_correct,
            ),
            (
                "generation_wrong_c_obj_wrong",
                receiver_wrong,
            ),
        ):
            for name in COMPONENT_ORDER:
                generation_group_rows.append({
                    "group": group_name,
                    "component": name,
                    "N": len(group),
                    "accuracy": safe_mean(
                        float(bool(row[f"{name}_correct"]))
                        for row in group
                    ),
                    "mean_gt_margin": safe_mean(
                        row[f"{name}_gt_margin"]
                        for row in group
                    ),
                    "mean_norm": safe_mean(
                        row[f"{name}_norm"]
                        for row in group
                    ),
                })

        write_csv(
            output_dir / "generation_group_summary.csv",
            generation_group_rows,
        )

        # ------------------------------------------------------------------
        # Source strength and closure diagnostics.
        # ------------------------------------------------------------------
        source_strength_rows: List[Dict[str, Any]] = []
        for group_name, group in (
            ("all_eval", eval_rows),
            ("generation_correct", generation_correct),
            ("generation_wrong", generation_wrong),
            (
                "generation_wrong_c_obj_correct",
                receiver_correct,
            ),
            (
                "generation_wrong_c_obj_wrong",
                receiver_wrong,
            ),
        ):
            for metric in (
                "object_mass_all_heads",
                "visual_mass_all_heads",
                "c_obj_all_norm",
                "c_visual_norm",
                "c_other_text_norm",
                "a_norm",
                "m_norm",
                "x_norm",
            ):
                source_strength_rows.append({
                    "group": group_name,
                    "metric": metric,
                    "N": len(group),
                    "mean": safe_mean(
                        row.get(metric)
                        for row in group
                    ),
                    "median": safe_median(
                        row.get(metric)
                        for row in group
                    ),
                })

        write_csv(
            output_dir / "source_strength_summary.csv",
            source_strength_rows,
        )

        closure_rows = [
            {
                "metric": "attention_replay_relative_error",
                "N": len(eval_rows),
                "mean": safe_mean(
                    row["attention_replay_relative_error"]
                    for row in eval_rows
                ),
                "max": max(
                    float(row["attention_replay_relative_error"])
                    for row in eval_rows
                ),
                "warning_threshold": args.replay_warning_threshold,
            },
            {
                "metric": "source_sum_relative_error",
                "N": len(eval_rows),
                "mean": safe_mean(
                    row["source_sum_relative_error"]
                    for row in eval_rows
                ),
                "max": max(
                    float(row["source_sum_relative_error"])
                    for row in eval_rows
                ),
                "warning_threshold": args.closure_warning_threshold,
            },
            {
                "metric": "block_closure_relative_error_y_vs_x+a+m",
                "N": len(eval_rows),
                "mean": safe_mean(
                    row["block_closure_relative_error"]
                    for row in eval_rows
                ),
                "max": max(
                    float(row["block_closure_relative_error"])
                    for row in eval_rows
                ),
                "warning_threshold": args.closure_warning_threshold,
            },
        ]
        write_csv(
            output_dir / "decomposition_closure.csv",
            closure_rows,
        )

        # ------------------------------------------------------------------
        # Print compact result.
        # ------------------------------------------------------------------
        accuracy_map = {
            row["component"]: float(row["accuracy"])
            for row in component_accuracy_rows
        }

        print("\n" + "=" * 150)
        print(f"L{args.layer} BLOCK DECOMPOSITION SUMMARY")
        print("=" * 150)
        print(
            f"Eval generation ACC: "
            f"{100*accuracy_map['generation']:.2f}% "
            f"| wrong={len(generation_wrong)}/{len(eval_rows)}"
        )
        print(
            f"Primary cohort: generation wrong AND c_obj_all correct = "
            f"{len(receiver_correct)}/{len(generation_wrong)} wrong samples "
            f"({100*len(receiver_correct)/max(len(generation_wrong),1):.2f}%)"
        )

        print("\nHeld-out component accuracies:")
        for row in component_accuracy_rows:
            print(
                f"  {str(row['component']):<24s} "
                f"{100*float(row['accuracy']):6.2f}% "
                f"N={int(row['N_eval']):3d} "
                f"[{row['type']}]"
            )

        print("\nWithin generation-wrong & c_obj_all-correct cohort:")
        for name in COMPONENT_ORDER:
            rows = [
                row
                for row in conditional_summary
                if row["cohort"] == "generation_wrong_and_c_obj_all_correct"
                and row["component"] == name
            ]
            if not rows:
                continue
            row = rows[0]
            print(
                f"  {name:<24s} "
                f"{100*float(row['accuracy_within_cohort']):6.2f}% "
                f"margin={float(row['mean_gt_margin']):+.4f}"
            )

        print("\nKey transitions:")
        for row in conditional_summary:
            if row["cohort"] != "transition":
                continue
            rate = float(row["accuracy_within_cohort"])
            print(
                f"  {str(row['component']):<46s} "
                f"N={int(row['N']):3d} "
                f"retain={100*rate:6.2f}%"
            )

        print("\nClosure diagnostics:")
        for row in closure_rows:
            print(
                f"  {row['metric']:<44s} "
                f"mean={float(row['mean']):.3e} "
                f"max={float(row['max']):.3e}"
            )
        print("=" * 150)

        # ------------------------------------------------------------------
        # Config/report.
        # ------------------------------------------------------------------
        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "data_root": args.data_root,
            "layer": args.layer,
            "receiver_heads": [
                hname((args.layer, head))
                for head in selected_query_heads
            ],
            "direction_dir_for_split_only": str(direction_dir),
            "prompt_template": args.prompt_template,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "N_train": len(valid_train),
            "N_eval": len(valid_eval),
            "N_generation_wrong": len(generation_wrong),
            "N_generation_wrong_c_obj_all_correct": len(receiver_correct),
            "uses_intervention": False,
            "natural_trajectory": {
                "x": "block input residual at prompt-last",
                "a": "attention output after o_proj at prompt-last",
                "r_attn": "x + a",
                "m": "MLP output at prompt-last",
                "y": "r_attn + m / block output",
            },
            "attention_source_decomposition": {
                "c_obj_all": (
                    "all-head object TEXT-token -> prompt-last post-W_O write"
                ),
                "c_obj_selected": (
                    "selected receiver-head object TEXT-token -> prompt-last write"
                ),
                "c_visual": "all visual-token -> prompt-last post-W_O write",
                "c_other_text": (
                    "all remaining non-object, non-visual positions -> prompt-last write"
                ),
                "c_nonobj": "c_visual + c_other_text",
            },
            "partial_reconstruction_warning": (
                "x_plus_obj and x_plus_obj_vis are algebraic partial sums, "
                "not natural temporal states."
            ),
            "receiver_correct_condition": (
                "generation wrong AND TRAIN-codebook(c_obj_all)==GT"
            ),
            "dataset_audit": audit,
        }
        (
            output_dir / "config.json"
        ).write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        report = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"layer: L{args.layer}",
            (
                f"generation ACC: "
                f"{100*accuracy_map['generation']:.2f}%"
            ),
            (
                "primary cohort: generation wrong & c_obj_all correct = "
                f"{len(receiver_correct)}/{len(generation_wrong)}"
            ),
            "",
            "HELD-OUT COMPONENT ACCURACY",
        ]
        for row in component_accuracy_rows:
            report.append(
                f"{row['component']}: "
                f"{100*float(row['accuracy']):.2f}%"
            )

        report += [
            "",
            "PRIMARY COHORT",
        ]
        for name in COMPONENT_ORDER:
            rows = [
                row
                for row in conditional_summary
                if row["cohort"]
                == "generation_wrong_and_c_obj_all_correct"
                and row["component"] == name
            ]
            if rows:
                report.append(
                    f"{name}: "
                    f"{100*float(rows[0]['accuracy_within_cohort']):.2f}% "
                    f"margin={float(rows[0]['mean_gt_margin']):+.4f}"
                )

        report += [
            "",
            "INTERPRETATION",
            (
                "c_obj_all high -> x_plus_obj low: incoming residual is the "
                "first candidate conflict."
            ),
            (
                "x_plus_obj high -> r_attn low: non-object attention is the "
                "first candidate conflict."
            ),
            (
                "r_attn high -> y low: L26 MLP is the first candidate conflict."
            ),
            (
                "c_obj_selected low but c_obj_all high: receiver shortlist "
                "is incomplete."
            ),
            "",
            "CAUTION",
            (
                "The component probes are descriptive. x_plus_obj is an algebraic "
                "partial reconstruction, not a naturally visited state. Once a "
                "candidate conflict is identified, validate it causally by scaling/"
                "removing only that component and measuring generation W->C/C->W."
            ),
        ]
        (
            output_dir / "report.txt"
        ).write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "config.json",
            "component_accuracy.csv",
            "baseline_eval.csv",
            "conditional_receiver_correct.csv",
            "conditional_summary.csv",
            "generation_group_summary.csv",
            "source_strength_summary.csv",
            "decomposition_closure.csv",
            "report.txt",
        ):
            print(" ", output_dir / name)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
