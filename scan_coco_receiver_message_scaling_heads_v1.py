#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-head natural object->prompt-last receiver scaling scan.

Goal
====
Find OTHER layers/heads whose own natural object->prompt-last message becomes
useful when its gain is increased.

This is a causal FUNCTIONAL scan, not a probe scan.

For each eval sample and each candidate query head (L,h), first obtain the CLEAN
sample-specific natural object->prompt-last post-W_O contribution:

    c_{L,h}
      = W_O^{L,h} sum_{s in subject/reference text tokens}
                    A_{L,h}[last,s] V_{L,h}[s]

Then run full greedy model.generate() while patching ONLY that head's layer at
prompt-last during PREFILL:

    attention_out'_L[last]
      = attention_out_L[last] + scale * c_{L,h}

Because attention_out_L already contains one copy of c_{L,h}, scale=6 means the
effective coefficient of that clean natural message is approximately 7 instead
of 1.

No GT is used in the intervention.

Primary ranking
===============
For each head:
    patched full-generation ACC
    W->C
    C->W
    net = W->C - C->W
    generation changed rate

Rank by net repairs, then patched ACC.

Default scan
============
    layers = L24-L31
    all query heads in each layer
    scale = 6
    gate = all samples

Reference sanity check
======================
By default also patch the known bundle:

    L26H4,L26H2,L26H6,L26H0

with the sum of their CLEAN natural messages.  On the same 375 eval rows this
should approximately reproduce the previous:

    obj_selected + all + scale=6
    66.93% -> 76.53%

If the reference bundle is far from the prior result, do NOT interpret the
single-head ranking until the implementation mismatch is resolved.

Important scope
===============
* This script scans SINGLE heads independently.
* A single-head effect can be weak even if a bundle is strong (redundancy/synergy).
* Cross-layer bundles are NOT tested here.
* Since each intervention changes only one layer, its c_{L,h} is exactly the
  head's CLEAN natural message before any upstream patch.
* This is full autoregressive generation, not a probe and not closed-set ACC.

Efficiency / resume
===================
A full L24-L31 scan on Qwen-3B is 8 layers x 16 heads = 128 patched generations
per sample, plus one reference-bundle generation.  For 375 samples this is
~48k generations.

The script is SAMPLE-MAJOR:
    sample -> clean trace once -> all head patches for that sample

It appends patch_results.jsonl after every patched generation and supports
--resume.  On restart it skips already completed (sid, layer, head, scale)
conditions.

You can first use --max-eval-samples 100 for discovery, then rerun all 375.

Required repository modules
===========================
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py
    extract_two_object_relation_states.py

Required prior output
=====================
    <decomposition-dir>/config.json
    <decomposition-dir>/baseline_eval.csv

The decomposition directory is used ONLY for:
    exact eval SIDs
    exact baseline generation result
    exact prompt template / split compatibility

Its cached L26 vectors are NOT used for the head scan.

Example: full scan
==================
CUDA_VISIBLE_DEVICES=0 python -u \
  scan_coco_receiver_message_scaling_heads_v1.py \
  --decomposition-dir output/qwen3b_l26_block_decomposition_v1_1 \
  --model qwen-3b \
  --layers 24-31 \
  --scales 6 \
  --device cuda:0 \
  --output-dir output/qwen3b_receiver_single_head_scan_L24_L31_s6_v1 \
  --overwrite

Example: quick discovery
========================
CUDA_VISIBLE_DEVICES=0 python -u \
  scan_coco_receiver_message_scaling_heads_v1.py \
  --decomposition-dir output/qwen3b_l26_block_decomposition_v1_1 \
  --model qwen-3b \
  --layers 24-31 \
  --scales 6 \
  --max-eval-samples 100 \
  --device cuda:0 \
  --output-dir output/qwen3b_receiver_single_head_scan_L24_L31_s6_n100_v1 \
  --overwrite

Outputs
=======
config.json
patch_results.jsonl
patch_results.csv
single_head_summary.csv
layer_summary.csv
relation_summary.csv
reference_bundle_summary.csv
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-receiver-message-single-head-scan-v1"
RELATIONS = ("left", "right", "above", "below")
EPS = 1e-12
DEFAULT_REFERENCE_BUNDLE = "26:4,26:2,26:6,26:0"


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--decomposition-dir",
        required=True,
        help="Output from analyze_coco_l26_block_decomposition_v1_1.py.",
    )
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))

    p.add_argument(
        "--layers",
        default="24-31",
        help="Layer range/list, e.g. 24-31 or 24,25,26,27.",
    )
    p.add_argument(
        "--heads",
        default="",
        help=(
            "Optional explicit heads, e.g. 26:4,27:5. Empty = all query heads "
            "in --layers."
        ),
    )
    p.add_argument(
        "--scales",
        default="6",
        help="Comma-separated scaling deltas. Discovery should usually use one.",
    )
    p.add_argument(
        "--reference-bundle",
        default=DEFAULT_REFERENCE_BUNDLE,
        help=(
            "Optional same-layer sanity bundle. Empty disables. "
            "Default should reproduce prior L26 selected-bundle scaling."
        ),
    )
    p.add_argument(
        "--run-reference-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="0 = all eval rows. Positive value takes a deterministic stratified subset.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )
    p.add_argument(
        "--trace-chunk-size",
        type=int,
        default=0,
        help=(
            "0 = trace all requested layers in one clean run_and_trace call. "
            "Positive = split layers into chunks to reduce peak memory."
        ),
    )

    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def parse_int_list_or_ranges(text: str) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a_text, b_text = chunk.split("-", 1)
            a, b = int(a_text), int(b_text)
            step = 1 if b >= a else -1
            values = range(a, b + step, step)
        else:
            values = [int(chunk)]
        for value in values:
            if value not in seen:
                out.append(value)
                seen.add(value)
    return out


def parse_scales(text: str) -> List[float]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError("Scale must be non-negative")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("No scales provided")
    return out


def parse_head(text: str) -> Tuple[int, int]:
    value = str(text).strip().upper()
    value = value.replace("L", "").replace("H", ":")
    while "::" in value:
        value = value.replace("::", ":")
    if ":" not in value:
        raise ValueError(f"Bad head spec: {text!r}")
    a, b = value.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        head = parse_head(item)
        if head not in seen:
            out.append(head)
            seen.add(head)
    return out


def hname(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in RELATIONS:
        return text

    # Prefer exact word matches in generated text.
    hits = []
    for relation in RELATIONS:
        match = re.search(rf"\b{re.escape(relation)}\b", text)
        if match:
            hits.append((match.start(), relation))
    aliases = (
        (r"\bunder(?:neath)?\b|\bbeneath\b", "below"),
        (r"\bover\b|\bon top\b", "above"),
    )
    for pattern, relation in aliases:
        match = re.search(pattern, text)
        if match:
            hits.append((match.start(), relation))

    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
                fields.append(key)
                seen.add(key)

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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
                " " + relation,
                add_special_tokens=False,
            )
            if not token_ids:
                raise RuntimeError(f"No token ID for {relation}")
            ids.add(int(token_ids[-1]))
        out[relation] = sorted(ids)
    return out


def deterministic_stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if limit <= 0 or len(rows) <= limit:
        return rows

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gt = normalize_relation(row.get("gt"))
        if gt in RELATIONS:
            grouped[gt].append(row)

    rng = random.Random(seed)
    for relation in RELATIONS:
        rng.shuffle(grouped[relation])

    cursors = {relation: 0 for relation in RELATIONS}
    selected: List[Dict[str, Any]] = []
    while len(selected) < limit:
        moved = False
        for relation in RELATIONS:
            group = grouped[relation]
            cursor = cursors[relation]
            if cursor < len(group) and len(selected) < limit:
                selected.append(group[cursor])
                cursors[relation] += 1
                moved = True
        if not moved:
            break

    # Stable SID order is convenient for resume.
    selected.sort(key=lambda row: int(row["sid"]))
    return selected


# =============================================================================
# Attention-output patch
# =============================================================================

def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected attention output [B,S,D], got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise RuntimeError("Could not find 3D attention output")


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return tuple(items)

    if isinstance(output, list):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return items

    raise RuntimeError("Could not replace 3D attention output")


class PromptLastLayerDelta:
    """
    Add a fixed clean residual-space vector to ONE attention layer's prompt-last
    module output during PREFILL only.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layer: int,
        prompt_length: int,
        prompt_last: int,
        delta: np.ndarray,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layer = int(layer)
        self.prompt_length = int(prompt_length)
        self.prompt_last = int(prompt_last)
        self.delta = np.asarray(delta, dtype=np.float32)
        self.handle = None
        self.applications = 0

    def __enter__(self) -> "PromptLastLayerDelta":
        attention = self.attention_helper.resolve_self_attention(
            self.decoder_layers[self.layer]
        )

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_3d(output)

            # Only full prompt prefill. Decode calls generally have q_len=1.
            if int(hidden.shape[1]) != self.prompt_length:
                return None

            if not 0 <= self.prompt_last < int(hidden.shape[1]):
                raise RuntimeError(
                    f"prompt_last={self.prompt_last} outside q_len={hidden.shape[1]}"
                )

            if int(hidden.shape[-1]) != int(self.delta.shape[0]):
                raise RuntimeError(
                    f"Delta dim {self.delta.shape[0]} != hidden dim {hidden.shape[-1]}"
                )

            modified = hidden.clone()
            vector = torch.as_tensor(
                self.delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            modified[0, self.prompt_last] += vector
            self.applications += 1
            return replace_first_3d(output, modified)

        self.handle = attention.register_forward_hook(hook)
        return self

    def validate(self) -> None:
        if self.applications != 1:
            raise RuntimeError(
                f"L{self.layer}: expected exactly one prefill patch, "
                f"got {self.applications}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Clean head-message extraction
# =============================================================================

def trace_target_index(trace: Any, prompt_last: int) -> int:
    lookup = {
        int(global_position): local
        for local, global_position in enumerate(trace.target_positions)
    }
    if int(prompt_last) not in lookup:
        raise RuntimeError(
            f"prompt_last={prompt_last} missing from trace targets "
            f"{trace.target_positions}"
        )
    return int(lookup[int(prompt_last)])


def all_head_object_writes(
    *,
    trace: Any,
    prompt_last: int,
    object_positions: Sequence[int],
) -> np.ndarray:
    """
    Return CLEAN post-W_O object->prompt-last message for every QUERY head.

    Shape:
        [n_query_heads, hidden_size]
    """
    object_positions = sorted(set(map(int, object_positions)))
    if not object_positions:
        raise RuntimeError("No object source positions")

    local = trace_target_index(trace, prompt_last)

    source = torch.as_tensor(
        object_positions,
        dtype=torch.long,
    )
    if int(source.max()) >= int(trace.value_states.shape[1]):
        raise RuntimeError(
            f"Object position {int(source.max())} exceeds source length "
            f"{trace.value_states.shape[1]}"
        )

    weights = (
        trace.attention_weights[:, local, :]
        .index_select(1, source)
        .float()
    )                                             # [H,Sobj]

    values = (
        trace.value_states
        .index_select(1, source)
        .float()
    )                                             # [H,Sobj,Dh]

    pre = torch.einsum(
        "hs,hsd->hd",
        weights,
        values,
    )                                             # [H,Dh]

    # trace.o_proj_weight is [D_model,H,Dh].
    post = torch.einsum(
        "hd,ohd->ho",
        pre,
        trace.o_proj_weight.float(),
    )                                             # [H,D_model]

    return post.detach().cpu().numpy().astype(np.float32)


def extract_clean_messages(
    *,
    attention_helper: Any,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    prompt_last: int,
    object_positions: Sequence[int],
    chunk_size: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """
    One clean trace (or chunked traces) -> per-layer [H,D] natural head messages.
    """
    all_messages: Dict[int, np.ndarray] = {}
    replay_errors: Dict[int, float] = {}

    if chunk_size <= 0:
        _, traces = attention_helper.run_and_trace(
            model=model,
            batch=batch,
            token_map=relation_token_map,
            decoder_layers=decoder_layers,
            layer_indices=list(layers),
            target_positions=[prompt_last],
        )
        for layer in layers:
            trace = traces[int(layer)]
            all_messages[int(layer)] = all_head_object_writes(
                trace=trace,
                prompt_last=prompt_last,
                object_positions=object_positions,
            )
            replay_errors[int(layer)] = float(trace.replay_relative_error)
        del traces
        return all_messages, replay_errors

    layers = list(layers)
    for start in range(0, len(layers), int(chunk_size)):
        chunk = layers[start : start + int(chunk_size)]
        _, traces = attention_helper.run_and_trace(
            model=model,
            batch=batch,
            token_map=relation_token_map,
            decoder_layers=decoder_layers,
            layer_indices=chunk,
            target_positions=[prompt_last],
        )
        for layer in chunk:
            trace = traces[int(layer)]
            all_messages[int(layer)] = all_head_object_writes(
                trace=trace,
                prompt_last=prompt_last,
                object_positions=object_positions,
            )
            replay_errors[int(layer)] = float(trace.replay_relative_error)
        del traces

    return all_messages, replay_errors


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
def patched_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    layer: int,
    prompt_last: int,
    delta: np.ndarray,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    prompt_length = int(batch["input_ids"].shape[1])

    with PromptLastLayerDelta(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layer=layer,
        prompt_length=prompt_length,
        prompt_last=prompt_last,
        delta=delta,
    ) as patch:
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        patch.validate()

    text = processor.tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()

    del output_ids
    return normalize_relation(text), text


# =============================================================================
# Resume keys
# =============================================================================

def condition_key(
    *,
    sid: int,
    kind: str,
    layer: int,
    head: int,
    scale: float,
    name: str = "",
) -> Tuple[Any, ...]:
    return (
        int(sid),
        str(kind),
        int(layer),
        int(head),
        round(float(scale), 9),
        str(name),
    )


def row_condition_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return condition_key(
        sid=int(row["sid"]),
        kind=str(row["kind"]),
        layer=int(row.get("layer", -1)),
        head=int(row.get("head", -1)),
        scale=float(row["scale"]),
        name=str(row.get("name", "")),
    )


# =============================================================================
# Summaries
# =============================================================================

def summarize_conditions(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_sid = {
        int(row["sid"]): dict(row)
        for row in baseline_rows
    }
    baseline_acc = safe_mean(
        float(parse_bool(row["generation_correct"]))
        for row in baseline_rows
    )

    grouped: Dict[
        Tuple[str, int, int, float, str],
        List[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in patch_rows:
        grouped[
            (
                str(row["kind"]),
                int(row.get("layer", -1)),
                int(row.get("head", -1)),
                float(row["scale"]),
                str(row.get("name", "")),
            )
        ].append(row)

    summary: List[Dict[str, Any]] = []

    for (kind, layer, head, scale, name), rows in grouped.items():
        # Summary is only valid when every baseline SID has a completed patched row.
        patched_by_sid = {
            int(row["sid"]): normalize_relation(
                row["patched_generation_prediction"]
            )
            for row in rows
        }

        covered_sids = [
            sid for sid in baseline_by_sid if sid in patched_by_sid
        ]

        N_expected = len(baseline_by_sid)
        N_completed = len(covered_sids)

        if not covered_sids:
            continue

        patched_acc_completed = safe_mean(
            float(
                patched_by_sid[sid]
                == normalize_relation(baseline_by_sid[sid]["gt"])
            )
            for sid in covered_sids
        )

        w2c = 0
        c2w = 0
        changed = 0

        for sid in covered_sids:
            base = baseline_by_sid[sid]
            gt = normalize_relation(base["gt"])
            base_pred = normalize_relation(
                base["generation_prediction"]
            )
            new_pred = patched_by_sid[sid]
            base_correct = parse_bool(
                base["generation_correct"]
            )
            new_correct = new_pred == gt

            w2c += int((not base_correct) and new_correct)
            c2w += int(base_correct and (not new_correct))
            changed += int(new_pred != base_pred)

        summary.append({
            "kind": kind,
            "name": name,
            "layer": layer,
            "head": head,
            "head_name": (
                hname(layer, head)
                if kind == "single_head"
                else name
            ),
            "scale": scale,
            "N_expected": N_expected,
            "N_completed": N_completed,
            "complete": N_completed == N_expected,
            "baseline_acc": baseline_acc,
            "patched_acc_completed": patched_acc_completed,
            "delta_acc_completed": patched_acc_completed - safe_mean(
                float(parse_bool(baseline_by_sid[sid]["generation_correct"]))
                for sid in covered_sids
            ),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net_repairs": w2c - c2w,
            "generation_changed": changed,
            "generation_changed_rate": changed / max(N_completed, 1),
            "mean_delta_norm": safe_mean(
                row["delta_norm"] for row in rows
            ),
            "mean_message_norm": safe_mean(
                row["message_norm"] for row in rows
            ),
            "mean_replay_error": safe_mean(
                row["replay_relative_error"] for row in rows
            ),
        })

    summary.sort(
        key=lambda row: (
            0 if row["complete"] else 1,
            -int(row["net_repairs"]),
            -float(row["patched_acc_completed"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    return summary


def make_layer_summary(
    single_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_layer: Dict[Tuple[int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in single_rows:
        if str(row["kind"]) != "single_head":
            continue
        by_layer[
            (int(row["layer"]), float(row["scale"]))
        ].append(row)

    out = []
    for (layer, scale), rows in sorted(by_layer.items()):
        complete = [row for row in rows if bool(row["complete"])]
        use = complete if complete else rows
        if not use:
            continue

        ranked = sorted(
            use,
            key=lambda row: (
                -int(row["net_repairs"]),
                -float(row["patched_acc_completed"]),
            ),
        )
        best = ranked[0]
        out.append({
            "layer": layer,
            "scale": scale,
            "N_heads": len(rows),
            "N_complete_heads": len(complete),
            "best_head": best["head_name"],
            "best_patched_acc": best["patched_acc_completed"],
            "best_delta_acc": best["delta_acc_completed"],
            "best_net_repairs": best["net_repairs"],
            "positive_net_heads": sum(
                int(row["net_repairs"]) > 0
                for row in use
            ),
            "zero_net_heads": sum(
                int(row["net_repairs"]) == 0
                for row in use
            ),
            "negative_net_heads": sum(
                int(row["net_repairs"]) < 0
                for row in use
            ),
            "mean_net_repairs": safe_mean(
                row["net_repairs"] for row in use
            ),
            "median_net_repairs": safe_median(
                row["net_repairs"] for row in use
            ),
            "mean_delta_acc": safe_mean(
                row["delta_acc_completed"] for row in use
            ),
        })
    return out


def make_relation_summary(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    top_conditions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_sid = {
        int(row["sid"]): row for row in baseline_rows
    }

    patch_lookup: Dict[
        Tuple[str, int, int, float, str],
        Dict[int, Mapping[str, Any]]
    ] = defaultdict(dict)

    for row in patch_rows:
        key = (
            str(row["kind"]),
            int(row.get("layer", -1)),
            int(row.get("head", -1)),
            float(row["scale"]),
            str(row.get("name", "")),
        )
        patch_lookup[key][int(row["sid"])] = row

    out = []
    for condition in top_conditions:
        key = (
            str(condition["kind"]),
            int(condition["layer"]),
            int(condition["head"]),
            float(condition["scale"]),
            str(condition["name"]),
        )
        rows_by_sid = patch_lookup.get(key, {})

        for relation in RELATIONS:
            sids = [
                sid for sid, base in baseline_by_sid.items()
                if normalize_relation(base["gt"]) == relation
                and sid in rows_by_sid
            ]
            if not sids:
                continue

            base_acc = safe_mean(
                float(
                    normalize_relation(
                        baseline_by_sid[sid]["generation_prediction"]
                    ) == relation
                )
                for sid in sids
            )
            patch_acc = safe_mean(
                float(
                    normalize_relation(
                        rows_by_sid[sid]["patched_generation_prediction"]
                    ) == relation
                )
                for sid in sids
            )

            out.append({
                "condition": condition["head_name"],
                "kind": condition["kind"],
                "layer": condition["layer"],
                "head": condition["head"],
                "scale": condition["scale"],
                "relation": relation,
                "N": len(sids),
                "baseline_acc": base_acc,
                "patched_acc": patch_acc,
                "delta_acc": patch_acc - base_acc,
            })

    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = parse_int_list_or_ranges(args.layers)
    if not layers:
        raise ValueError("No layers")

    explicit_heads = parse_heads(args.heads)
    scales = parse_scales(args.scales)
    reference_bundle = parse_heads(args.reference_bundle)

    decomposition_dir = Path(args.decomposition_dir)
    config_path = decomposition_dir / "config.json"
    baseline_path = decomposition_dir / "baseline_eval.csv"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    decomposition_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    baseline_rows = read_csv(baseline_path)
    baseline_rows = deterministic_stratified_subset(
        baseline_rows,
        args.max_eval_samples,
        args.seed,
    )

    if not baseline_rows:
        raise RuntimeError("No baseline eval rows")

    # Stable sample order.
    baseline_rows.sort(key=lambda row: int(row["sid"]))

    output_dir = Path(args.output_dir)

    # Resume semantics:
    # --overwrite removes everything. Otherwise existing jsonl may be resumed.
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_jsonl = output_dir / "patch_results.jsonl"
    errors_path = output_dir / "errors.jsonl"

    if (
        not args.overwrite
        and output_dir.exists()
        and not args.resume
        and any(output_dir.iterdir())
    ):
        raise RuntimeError(
            f"{output_dir} is not empty. Use --overwrite or --resume."
        )

    prior_patch_rows = load_jsonl(patch_jsonl) if args.resume else []
    completed = {
        row_condition_key(row)
        for row in prior_patch_rows
    }

    probe = importlib.import_module(args.probe_module)
    attention_helper = importlib.import_module(
        args.attention_helper_module
    )
    base = probe.base

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {
        int(record.sid): record
        for record in records
    }

    prompt_template = str(
        decomposition_config.get("prompt_template")
        or (
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        )
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

    try:
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)

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

        for layer in layers:
            if not 0 <= layer < len(decoder_layers):
                raise ValueError(
                    f"L{layer} outside decoder range 0..{len(decoder_layers)-1}"
                )

        # Resolve number of query heads per layer from trace-compatible attention modules.
        n_heads_by_layer: Dict[int, int] = {}
        hidden_by_layer: Dict[int, int] = {}

        for layer in layers:
            attention = attention_helper.resolve_self_attention(
                decoder_layers[layer]
            )
            # o_proj input width / head_dim is robust for Qwen query-head count.
            o_proj = getattr(attention, "o_proj", None)
            q_proj = getattr(attention, "q_proj", None)
            if o_proj is None or q_proj is None:
                raise RuntimeError(
                    f"L{layer} attention missing q_proj/o_proj"
                )

            num_heads = getattr(attention, "num_heads", None)
            if num_heads is None:
                num_heads = getattr(
                    getattr(attention, "config", None),
                    "num_attention_heads",
                    None,
                )
            if num_heads is None:
                # Q projection output / configured head_dim fallback.
                head_dim = getattr(attention, "head_dim", None)
                if head_dim is None:
                    raise RuntimeError(
                        f"Cannot infer query-head count for L{layer}"
                    )
                num_heads = int(q_proj.out_features) // int(head_dim)

            n_heads_by_layer[layer] = int(num_heads)
            hidden_by_layer[layer] = int(o_proj.out_features)

        if explicit_heads:
            scan_heads = []
            for layer, head in explicit_heads:
                if layer not in layers:
                    raise ValueError(
                        f"Explicit {hname(layer, head)} is outside --layers"
                    )
                if not 0 <= head < n_heads_by_layer[layer]:
                    raise ValueError(
                        f"{hname(layer, head)} outside 0..{n_heads_by_layer[layer]-1}"
                    )
                scan_heads.append((layer, head))
        else:
            scan_heads = [
                (layer, head)
                for layer in layers
                for head in range(n_heads_by_layer[layer])
            ]

        # Reference bundle must be same-layer because patch is one module output.
        reference_layer: Optional[int] = None
        reference_heads: List[int] = []
        if args.run_reference_bundle and reference_bundle:
            ref_layers = sorted({layer for layer, _ in reference_bundle})
            if len(ref_layers) != 1:
                raise ValueError(
                    "--reference-bundle must be same-layer in this script"
                )
            reference_layer = ref_layers[0]
            reference_heads = [
                head for _, head in reference_bundle
            ]
            if reference_layer not in layers:
                # Add it to tracing, but do not add all its heads to scan.
                layers = sorted(set(layers + [reference_layer]))
                attention = attention_helper.resolve_self_attention(
                    decoder_layers[reference_layer]
                )
                num_heads = getattr(attention, "num_heads", None)
                if num_heads is None:
                    num_heads = getattr(
                        getattr(attention, "config", None),
                        "num_attention_heads",
                        None,
                    )
                if num_heads is None:
                    head_dim = getattr(attention, "head_dim", None)
                    num_heads = int(attention.q_proj.out_features) // int(head_dim)
                n_heads_by_layer[reference_layer] = int(num_heads)

            for head in reference_heads:
                if not 0 <= head < n_heads_by_layer[reference_layer]:
                    raise ValueError(
                        f"Reference {hname(reference_layer, head)} invalid"
                    )

        relation_token_map = relation_token_variants(
            processor.tokenizer
        )

        baseline_acc = safe_mean(
            float(parse_bool(row["generation_correct"]))
            for row in baseline_rows
        )

        expected_single_conditions = len(scan_heads) * len(scales)
        expected_ref_conditions = (
            len(scales)
            if args.run_reference_bundle and reference_bundle
            else 0
        )
        total_expected_generations = (
            len(baseline_rows)
            * (expected_single_conditions + expected_ref_conditions)
        )

        print("\n" + "=" * 160)
        print("SINGLE-HEAD NATURAL RECEIVER-MESSAGE SCAN")
        print("=" * 160)
        print("layers            :", layers)
        print("query heads/layer :", n_heads_by_layer)
        print("N scan heads      :", len(scan_heads))
        print("scales            :", scales)
        print("N eval            :", len(baseline_rows))
        print("baseline ACC      :", f"{100*baseline_acc:.2f}%")
        print(
            "reference bundle  :",
            (
                ",".join(
                    hname(reference_layer, head)
                    for head in reference_heads
                )
                if reference_layer is not None
                else "disabled"
            ),
        )
        print("expected patched generations:", total_expected_generations)
        print("already completed :", len(completed))
        print("prompt            :", prompt_template)
        print("=" * 160, flush=True)

        for sample_counter, base_row in enumerate(
            tqdm(baseline_rows, desc="samples"),
            start=1,
        ):
            sid = int(base_row["sid"])
            image = None
            batch = None

            try:
                if sid not in record_by_sid:
                    raise RuntimeError(f"SID {sid} missing from dataset")

                record = record_by_sid[sid]
                gt = normalize_relation(base_row["gt"])
                baseline_prediction = normalize_relation(
                    base_row["generation_prediction"]
                )
                baseline_correct = parse_bool(
                    base_row["generation_correct"]
                )

                question = prompt_template.format(
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

                input_ids = (
                    batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                input_ids = [int(x) for x in input_ids]
                prompt_last = len(input_ids) - 1

                subject_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.subject),
                )
                reference_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.reference),
                )
                object_positions = sorted(
                    set(
                        map(
                            int,
                            subject_positions + reference_positions,
                        )
                    )
                )
                if not object_positions:
                    raise RuntimeError(
                        f"SID {sid}: no object text-token positions"
                    )

                # We only need a clean trace if at least one condition for this SID
                # remains unfinished.
                needed_single = []
                for layer, head in scan_heads:
                    for scale in scales:
                        key = condition_key(
                            sid=sid,
                            kind="single_head",
                            layer=layer,
                            head=head,
                            scale=scale,
                        )
                        if key not in completed:
                            needed_single.append((layer, head, scale))

                needed_ref = []
                if reference_layer is not None:
                    ref_name = "+".join(
                        hname(reference_layer, head)
                        for head in reference_heads
                    )
                    for scale in scales:
                        key = condition_key(
                            sid=sid,
                            kind="reference_bundle",
                            layer=reference_layer,
                            head=-1,
                            scale=scale,
                            name=ref_name,
                        )
                        if key not in completed:
                            needed_ref.append(scale)

                if not needed_single and not needed_ref:
                    continue

                needed_layers = sorted(
                    set(
                        [layer for layer, _, _ in needed_single]
                        + (
                            [reference_layer]
                            if needed_ref and reference_layer is not None
                            else []
                        )
                    )
                )

                clean_messages, replay_errors = extract_clean_messages(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layers=needed_layers,
                    prompt_last=prompt_last,
                    object_positions=object_positions,
                    chunk_size=args.trace_chunk_size,
                )

                # Validate expected head counts from trace.
                for layer in needed_layers:
                    messages = clean_messages[layer]
                    expected_heads = n_heads_by_layer[layer]
                    if int(messages.shape[0]) != expected_heads:
                        raise RuntimeError(
                            f"L{layer}: trace returned {messages.shape[0]} heads, "
                            f"expected {expected_heads}"
                        )

                # Single heads.
                for layer, head, scale in needed_single:
                    message = clean_messages[layer][head]
                    delta = float(scale) * message

                    patched_prediction, patched_text = patched_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        layer=layer,
                        prompt_last=prompt_last,
                        delta=delta,
                        max_new_tokens=args.max_new_tokens,
                    )

                    patched_correct = patched_prediction == gt
                    row = {
                        "sid": sid,
                        "kind": "single_head",
                        "name": "",
                        "layer": layer,
                        "head": head,
                        "head_name": hname(layer, head),
                        "scale": float(scale),
                        "gt": gt,
                        "baseline_generation_prediction": baseline_prediction,
                        "baseline_generation_correct": baseline_correct,
                        "patched_generation_prediction": patched_prediction,
                        "patched_generation_text": patched_text,
                        "patched_generation_correct": patched_correct,
                        "wrong_to_correct": (
                            (not baseline_correct)
                            and patched_correct
                        ),
                        "correct_to_wrong": (
                            baseline_correct
                            and (not patched_correct)
                        ),
                        "generation_changed": (
                            patched_prediction != baseline_prediction
                        ),
                        "message_norm": float(
                            np.linalg.norm(message)
                        ),
                        "delta_norm": float(
                            np.linalg.norm(delta)
                        ),
                        "replay_relative_error": float(
                            replay_errors[layer]
                        ),
                        "N_object_positions": len(object_positions),
                    }
                    append_jsonl(patch_jsonl, row)
                    prior_patch_rows.append(row)
                    completed.add(row_condition_key(row))

                # Known L26 bundle sanity reference.
                if needed_ref and reference_layer is not None:
                    ref_name = "+".join(
                        hname(reference_layer, head)
                        for head in reference_heads
                    )
                    ref_message = clean_messages[
                        reference_layer
                    ][reference_heads].sum(axis=0)

                    for scale in needed_ref:
                        delta = float(scale) * ref_message

                        patched_prediction, patched_text = patched_generation(
                            model=model,
                            processor=processor,
                            batch=batch,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            layer=reference_layer,
                            prompt_last=prompt_last,
                            delta=delta,
                            max_new_tokens=args.max_new_tokens,
                        )

                        patched_correct = patched_prediction == gt
                        row = {
                            "sid": sid,
                            "kind": "reference_bundle",
                            "name": ref_name,
                            "layer": reference_layer,
                            "head": -1,
                            "head_name": ref_name,
                            "scale": float(scale),
                            "gt": gt,
                            "baseline_generation_prediction": baseline_prediction,
                            "baseline_generation_correct": baseline_correct,
                            "patched_generation_prediction": patched_prediction,
                            "patched_generation_text": patched_text,
                            "patched_generation_correct": patched_correct,
                            "wrong_to_correct": (
                                (not baseline_correct)
                                and patched_correct
                            ),
                            "correct_to_wrong": (
                                baseline_correct
                                and (not patched_correct)
                            ),
                            "generation_changed": (
                                patched_prediction != baseline_prediction
                            ),
                            "message_norm": float(
                                np.linalg.norm(ref_message)
                            ),
                            "delta_norm": float(
                                np.linalg.norm(delta)
                            ),
                            "replay_relative_error": float(
                                replay_errors[reference_layer]
                            ),
                            "N_object_positions": len(object_positions),
                        }
                        append_jsonl(patch_jsonl, row)
                        prior_patch_rows.append(row)
                        completed.add(row_condition_key(row))

                del clean_messages

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "sample_scan",
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
                    and sample_counter % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        # Reload from disk as source of truth after possible resume.
        patch_rows = load_jsonl(patch_jsonl)
        write_csv(
            output_dir / "patch_results.csv",
            patch_rows,
        )

        summaries = summarize_conditions(
            baseline_rows=baseline_rows,
            patch_rows=patch_rows,
        )

        single_summary = [
            row for row in summaries
            if row["kind"] == "single_head"
        ]
        reference_summary = [
            row for row in summaries
            if row["kind"] == "reference_bundle"
        ]

        write_csv(
            output_dir / "single_head_summary.csv",
            single_summary,
        )
        write_csv(
            output_dir / "reference_bundle_summary.csv",
            reference_summary,
        )

        layer_summary = make_layer_summary(single_summary)
        write_csv(
            output_dir / "layer_summary.csv",
            layer_summary,
        )

        # Relation breakdown for top 20 complete heads + reference.
        complete_heads = [
            row for row in single_summary
            if bool(row["complete"])
        ]
        top_conditions = complete_heads[:20] + [
            row for row in reference_summary
            if bool(row["complete"])
        ]
        relation_rows = make_relation_summary(
            baseline_rows=baseline_rows,
            patch_rows=patch_rows,
            top_conditions=top_conditions,
        )
        write_csv(
            output_dir / "relation_summary.csv",
            relation_rows,
        )

        # ------------------------------------------------------------------
        # Print
        # ------------------------------------------------------------------
        print("\n" + "=" * 168)
        print("SINGLE-HEAD RECEIVER SCAN SUMMARY")
        print("=" * 168)
        print(
            f"Baseline generation ACC: {100*baseline_acc:.2f}% "
            f"| N={len(baseline_rows)}"
        )

        if reference_summary:
            print("\nReference bundle sanity:")
            for row in reference_summary:
                status = "FULL" if row["complete"] else "PARTIAL"
                print(
                    f"  {row['head_name']:<34s} "
                    f"scale={float(row['scale']):>4.1f} "
                    f"N={int(row['N_completed']):>3d}/{int(row['N_expected']):>3d} "
                    f"ACC={100*float(row['patched_acc_completed']):6.2f}% "
                    f"delta={100*float(row['delta_acc_completed']):+6.2f}pp "
                    f"W->C={int(row['wrong_to_correct']):3d} "
                    f"C->W={int(row['correct_to_wrong']):3d} "
                    f"net={int(row['net_repairs']):+3d} "
                    f"[{status}]"
                )

        print("\nTop single heads:")
        print(
            f"  {'rank':>4s} {'head':<9s} {'scale':>5s} "
            f"{'N':>7s} {'ACC':>8s} {'delta':>9s} "
            f"{'W->C':>5s} {'C->W':>5s} {'net':>5s} {'chg':>7s}"
        )
        for rank, row in enumerate(single_summary[:30], start=1):
            print(
                f"  {rank:>4d} "
                f"{row['head_name']:<9s} "
                f"{float(row['scale']):>5.1f} "
                f"{int(row['N_completed']):>3d}/{int(row['N_expected']):<3d} "
                f"{100*float(row['patched_acc_completed']):>7.2f}% "
                f"{100*float(row['delta_acc_completed']):>+8.2f} "
                f"{int(row['wrong_to_correct']):>5d} "
                f"{int(row['correct_to_wrong']):>5d} "
                f"{int(row['net_repairs']):>+5d} "
                f"{100*float(row['generation_changed_rate']):>6.2f}%"
            )

        print("\nBest head per layer:")
        for row in layer_summary:
            print(
                f"  L{int(row['layer']):02d} "
                f"best={row['best_head']:<9s} "
                f"ACC={100*float(row['best_patched_acc']):6.2f}% "
                f"delta={100*float(row['best_delta_acc']):+6.2f}pp "
                f"net={int(row['best_net_repairs']):+3d} "
                f"positive_heads={int(row['positive_net_heads']):2d}/"
                f"{int(row['N_heads']):2d}"
            )
        print("=" * 168)

        # ------------------------------------------------------------------
        # Config / report
        # ------------------------------------------------------------------
        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "data_root": args.data_root,
            "decomposition_dir": str(decomposition_dir),
            "prompt_template": prompt_template,
            "layers": layers,
            "scan_heads": [
                hname(layer, head)
                for layer, head in scan_heads
            ],
            "scales": scales,
            "reference_bundle": (
                [
                    hname(reference_layer, head)
                    for head in reference_heads
                ]
                if reference_layer is not None
                else []
            ),
            "N_eval": len(baseline_rows),
            "baseline_acc": baseline_acc,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "trace_chunk_size": args.trace_chunk_size,
            "patch_formula": (
                "attention_out_L[last] += scale * clean_c_Lh, "
                "where clean_c_Lh = W_O^h sum_object A[last,s]V[s]"
            ),
            "uses_eval_gt_for_patch": False,
            "generation_metric": "full greedy model.generate()",
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

        lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"baseline ACC: {100*baseline_acc:.2f}%",
            f"N eval: {len(baseline_rows)}",
            "",
            "REFERENCE BUNDLE",
        ]
        for row in reference_summary:
            lines.append(
                f"{row['head_name']} scale={float(row['scale']):.1f}: "
                f"ACC={100*float(row['patched_acc_completed']):.2f}% "
                f"delta={100*float(row['delta_acc_completed']):+.2f}pp "
                f"W->C={int(row['wrong_to_correct'])} "
                f"C->W={int(row['correct_to_wrong'])} "
                f"net={int(row['net_repairs']):+d} "
                f"complete={bool(row['complete'])}"
            )

        lines += [
            "",
            "TOP SINGLE HEADS",
        ]
        for rank, row in enumerate(single_summary[:30], start=1):
            lines.append(
                f"{rank:02d} {row['head_name']} scale={float(row['scale']):.1f}: "
                f"ACC={100*float(row['patched_acc_completed']):.2f}% "
                f"delta={100*float(row['delta_acc_completed']):+.2f}pp "
                f"W->C={int(row['wrong_to_correct'])} "
                f"C->W={int(row['correct_to_wrong'])} "
                f"net={int(row['net_repairs']):+d} "
                f"complete={bool(row['complete'])}"
            )

        lines += [
            "",
            "INTERPRETATION",
            (
                "A positive single-head net effect means amplifying that head's "
                "own CLEAN sample-specific object->last message causally improves "
                "full generation more often than it harms it."
            ),
            (
                "Weak single heads plus a strong reference bundle imply distributed/"
                "synergistic receiver behavior rather than a single dominant head."
            ),
            (
                "Strong heads across multiple late layers motivate a later cross-layer "
                "ONLINE scaling experiment, where downstream messages are recomputed "
                "after upstream modifications."
            ),
            "",
            "CAUTION",
            (
                "The scan uses scale selection fixed before the scan (default 6), "
                "but head selection is performed on these eval samples. Any final ACC "
                "claim for a chosen head/bundle requires a fresh validation split."
            ),
        ]
        (
            output_dir / "report.txt"
        ).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "config.json",
            "patch_results.jsonl",
            "patch_results.csv",
            "single_head_summary.csv",
            "layer_summary.csv",
            "relation_summary.csv",
            "reference_bundle_summary.csv",
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
