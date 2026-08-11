#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO/Qwen-3B single-head swapped-query state injection.

Question pair
-------------
Q_AB: relation of A relative to B?   -> GT r
Q_BA: relation of B relative to A?   -> GT opposite(r)

Intervention
------------
For one attention head h at one layer L, capture the REAL pre-W_O head output
at prompt-last from the other query branch, then inject that slice into the
current branch at the same layer/head/prompt-last:

    target_head_state(prompt_last) <- source_head_state(prompt_last)

No LM-readout direction is constructed, and no A/B object-token swap is done.
All downstream computation is left live and recomputed normally.

The script has two stages:
1) fast first-answer-token scan over selected heads;
2) optional full model.generate() validation for the strongest scan heads.

By default, selected heads are the current Step-1 Top-20 centroid heads from:
    <step1-dir>/aggregate_metrics.npz

Use --head-selection all to scan all decoder attention heads on a small pilot.

Repository dependencies
-----------------------
- trace_centroid_generation_groups_v2_1.py
- extract_two_object_relation_states.py

Recommended pilot
-----------------
CUDA_VISIBLE_DEVICES=0 python -u eval_coco_qwen3b_swapped_head_state_injection_v1.py \
  --step1-dir output/qwen3b_coco_attention_flow_swap_step1 \
  --head-selection centroid-top \
  --top-k 20 \
  --directions swapped_into_original,original_into_swapped \
  --max-samples 20 \
  --generation-top-k 5 \
  --max-new-tokens 8 \
  --device cuda:0 \
  --output-dir output/qwen3b_swapped_head_state_injection_pilot20 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import trace_centroid_generation_groups_v2_1 as base

SCRIPT_VERSION = "coco-qwen3b-swapped-head-state-injection-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
DIRECTIONS = ("swapped_into_original", "original_into_swapped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step1-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--prompt-jsonl", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--head-selection", choices=("centroid-top", "all"), default="centroid-top")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--directions", default="swapped_into_original,original_into_swapped")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--sample-status", choices=("all", "both_correct", "original_correct", "swapped_correct"), default="all")
    p.add_argument("--generation-top-k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--min-new-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def csv_tokens(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Bad JSONL {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.std(vals)) if vals else float("nan")


def normalize_relation(x: Any) -> Optional[str]:
    y = base.normalize_relation(x)
    return y if y in RELATIONS else None


def one_line(x: Any) -> str:
    return " ".join(str(x).split())


def build_swapped_question(subject: str, reference: str) -> str:
    return (
        f"Where is the {reference} in relation to the {subject}? "
        "Answer with left, right, above, or below."
    )


# -----------------------------------------------------------------------------
# Step-1 / centroid metadata
# -----------------------------------------------------------------------------

def load_step1_config(step1_dir: Path) -> Dict[str, Any]:
    p = step1_dir / "config.json"
    if p.exists():
        return read_json(p)
    p = step1_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing config.json/summary.json in {step1_dir}")
    payload = read_json(p)
    cfg = payload.get("config")
    if isinstance(cfg, dict):
        return cfg
    return payload


def load_centroid_accuracy(step1_dir: Path) -> Tuple[np.ndarray, List[int]]:
    p = step1_dir / "aggregate_metrics.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        if "attention_average_accuracy" not in z:
            raise KeyError(f"{p} has no attention_average_accuracy")
        acc = np.asarray(z["attention_average_accuracy"], dtype=np.float64)
        layers = np.asarray(
            z["layer_indices"] if "layer_indices" in z else np.arange(acc.shape[0]),
            dtype=np.int64,
        ).tolist()
    if acc.ndim != 2 or len(layers) != acc.shape[0]:
        raise RuntimeError(f"Bad centroid array shape acc={acc.shape}, layers={len(layers)}")
    return acc, [int(x) for x in layers]


def build_selected_heads(
    *,
    decoder_layers: Sequence[Any],
    centroid_acc: np.ndarray,
    centroid_layers: Sequence[int],
    selection: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    centroid_pos = {int(layer): i for i, layer in enumerate(centroid_layers)}
    heads: List[Dict[str, Any]] = []

    if selection == "centroid-top":
        order = np.argsort(centroid_acc.reshape(-1))[::-1]
        for flat in order[:top_k]:
            lp, h = np.unravel_index(int(flat), centroid_acc.shape)
            heads.append({
                "layer": int(centroid_layers[lp]),
                "head": int(h),
                "centroid_accuracy": float(centroid_acc[lp, h]),
            })
    elif selection == "all":
        for layer_idx, layer in enumerate(decoder_layers):
            attn = find_self_attention(layer)
            n_heads, _head_dim, _hidden = resolve_head_geometry(attn)
            for h in range(n_heads):
                lp = centroid_pos.get(layer_idx)
                cacc = float(centroid_acc[lp, h]) if lp is not None and h < centroid_acc.shape[1] else float("nan")
                heads.append({"layer": layer_idx, "head": h, "centroid_accuracy": cacc})
    else:
        raise ValueError(selection)

    for rank, row in enumerate(sorted(heads, key=lambda x: (-float(x["centroid_accuracy"]) if math.isfinite(float(x["centroid_accuracy"])) else 1e9, x["layer"], x["head"])), 1):
        row["centroid_rank"] = rank
    return heads


# -----------------------------------------------------------------------------
# Attention / pre-WO helpers
# -----------------------------------------------------------------------------

def find_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        obj = getattr(layer, name, None)
        if obj is not None and getattr(obj, "o_proj", None) is not None:
            return obj
    # one shallow nested fallback
    for parent_name in ("decoder_layer", "layer", "block"):
        parent = getattr(layer, parent_name, None)
        if parent is None:
            continue
        for name in ("self_attn", "attention", "attn"):
            obj = getattr(parent, name, None)
            if obj is not None and getattr(obj, "o_proj", None) is not None:
                return obj
    raise AttributeError(f"Could not locate self-attention/o_proj in {type(layer).__name__}")


def resolve_head_geometry(attention: Any) -> Tuple[int, int, int]:
    o_proj = getattr(attention, "o_proj", None)
    if o_proj is None:
        raise AttributeError(f"{type(attention).__name__} has no o_proj")
    in_features = getattr(o_proj, "in_features", None)
    if in_features is None:
        weight = getattr(o_proj, "weight", None)
        if weight is None or weight.ndim != 2:
            raise RuntimeError("Cannot infer o_proj input width")
        in_features = int(weight.shape[1])
    else:
        in_features = int(in_features)

    head_dim = getattr(attention, "head_dim", None)
    if head_dim is not None:
        head_dim = int(head_dim)
        if head_dim > 0 and in_features % head_dim == 0:
            return in_features // head_dim, head_dim, in_features

    for name in ("num_heads", "num_attention_heads"):
        n_heads = getattr(attention, name, None)
        if n_heads is not None:
            n_heads = int(n_heads)
            if n_heads > 0 and in_features % n_heads == 0:
                return n_heads, in_features // n_heads, in_features

    cfg = getattr(attention, "config", None)
    if cfg is not None:
        for name in ("num_attention_heads", "num_heads"):
            n_heads = getattr(cfg, name, None)
            if n_heads is not None:
                n_heads = int(n_heads)
                if n_heads > 0 and in_features % n_heads == 0:
                    return n_heads, in_features // n_heads, in_features
    raise RuntimeError(f"Cannot infer head geometry for {type(attention).__name__}")


class PromptLastPreWOCapture:
    def __init__(self, decoder_layers: Sequence[Any], layer_ids: Sequence[int], prompt_length: int):
        self.decoder_layers = decoder_layers
        self.layer_ids = sorted(set(int(x) for x in layer_ids))
        self.prompt_length = int(prompt_length)
        self.states: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []

    def __enter__(self) -> "PromptLastPreWOCapture":
        for layer_id in self.layer_ids:
            attn = find_self_attention(self.decoder_layers[layer_id])
            o_proj = attn.o_proj

            def make_hook(lid: int):
                def pre_hook(_module: Any, inputs: Tuple[Any, ...]):
                    if not inputs:
                        return None
                    x = inputs[0]
                    if not torch.is_tensor(x) or x.ndim != 3:
                        return None
                    if int(x.shape[1]) != self.prompt_length:
                        return None
                    self.states[lid] = x[0, -1, :].detach().float().cpu().clone()
                    return None
                return pre_hook

            self.handles.append(o_proj.register_forward_pre_hook(make_hook(layer_id)))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class InjectOneHeadAtPromptLast:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        layer: int,
        head: int,
        source_state_cpu: torch.Tensor,
        prompt_length: int,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.layer = int(layer)
        self.head = int(head)
        self.source_state_cpu = source_state_cpu
        self.prompt_length = int(prompt_length)
        self.handle = None
        self.applications = 0

    def __enter__(self) -> "InjectOneHeadAtPromptLast":
        attn = find_self_attention(self.decoder_layers[self.layer])
        n_heads, head_dim, hidden = resolve_head_geometry(attn)
        if not 0 <= self.head < n_heads:
            raise ValueError(f"L{self.layer}H{self.head} outside n_heads={n_heads}")
        if int(self.source_state_cpu.numel()) != hidden:
            raise RuntimeError(f"Source state width={self.source_state_cpu.numel()} expected={hidden}")
        start, stop = self.head * head_dim, (self.head + 1) * head_dim

        def pre_hook(_module: Any, inputs: Tuple[Any, ...]):
            if not inputs:
                return None
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim != 3:
                return None
            # prefill only; decode q_len=1 is untouched
            if int(x.shape[1]) != self.prompt_length:
                return None
            y = x.clone()
            src = self.source_state_cpu[start:stop].to(device=y.device, dtype=y.dtype)
            y[:, -1, start:stop] = src.unsqueeze(0).expand(y.shape[0], -1)
            self.applications += 1
            return (y,) + tuple(inputs[1:])

        self.handle = attn.o_proj.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# -----------------------------------------------------------------------------
# Relation logits / generation
# -----------------------------------------------------------------------------

def relation_scores(logits: torch.Tensor, label_ids: Mapping[str, Sequence[int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in RELATIONS:
        ids = torch.as_tensor(list(label_ids[r]), device=logits.device, dtype=torch.long)
        out[r] = float(logits.index_select(0, ids).float().max().item())
    return out


def score_prediction(scores: Mapping[str, float]) -> str:
    return max(RELATIONS, key=lambda r: float(scores[r]))


def branch_margin(scores: Mapping[str, float], source_gt: str, target_gt: str) -> float:
    return float(scores[source_gt] - scores[target_gt])


def forward_capture(
    *,
    model: Any,
    batch: Dict[str, Any],
    decoder_layers: Sequence[Any],
    layer_ids: Sequence[int],
    label_ids: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, Any], Dict[int, torch.Tensor]]:
    prompt_length = int(batch["input_ids"].shape[1])
    with PromptLastPreWOCapture(decoder_layers, layer_ids, prompt_length) as cap:
        with torch.inference_mode():
            out = model(**batch, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    scores = relation_scores(out.logits[0, -1], label_ids)
    result = {"prediction": score_prediction(scores), "scores": scores, "prompt_length": prompt_length}
    states = dict(cap.states)
    del out
    missing = sorted(set(layer_ids) - set(states))
    if missing:
        raise RuntimeError(f"Did not capture pre-WO prompt-last states for layers {missing[:10]}")
    return result, states


def forward_injected(
    *,
    model: Any,
    batch: Dict[str, Any],
    decoder_layers: Sequence[Any],
    layer: int,
    head: int,
    source_state_cpu: torch.Tensor,
    label_ids: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    with InjectOneHeadAtPromptLast(
        decoder_layers=decoder_layers,
        layer=layer,
        head=head,
        source_state_cpu=source_state_cpu,
        prompt_length=prompt_length,
    ) as patch:
        with torch.inference_mode():
            out = model(**batch, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    if patch.applications != 1:
        raise RuntimeError(f"Expected exactly one patch, got {patch.applications} for L{layer}H{head}")
    scores = relation_scores(out.logits[0, -1], label_ids)
    result = {"prediction": score_prediction(scores), "scores": scores}
    del out
    return result


REL_PATTERNS: Dict[str, Sequence[str]] = {
    "left": (r"\bleft\s+of\b", r"\bto\s+the\s+left\b", r"\bleft\b"),
    "right": (r"\bright\s+of\b", r"\bto\s+the\s+right\b", r"\bright\b"),
    "above": (r"\bon\s+top\s+of\b", r"\batop\b", r"\babove\b", r"\bover\b"),
    "below": (r"\bunderneath\b", r"\bbeneath\b", r"\bbelow\b", r"\bunder\b"),
}


def parse_generation(text: str) -> Optional[str]:
    text = one_line(text).lower()
    hits: List[Tuple[int, int, str]] = []
    for rel, patterns in REL_PATTERNS.items():
        for pri, pat in enumerate(patterns):
            m = re.search(pat, text)
            if m:
                hits.append((m.start(), pri, rel))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def generation_kwargs(processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    tok = processor.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kw = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": pad,
    }
    if tok.eos_token_id is not None:
        kw["eos_token_id"] = tok.eos_token_id
    return kw


def run_generate(model: Any, processor: Any, batch: Dict[str, Any], gen_kw: Mapping[str, Any]) -> Dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(**batch, **dict(gen_kw))
    seq = out.sequences[0]
    new = seq[prompt_length:]
    text = processor.tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    scores = None
    if getattr(out, "scores", None):
        scores = out.scores[0][0]
    result = {
        "text": one_line(text),
        "prediction": parse_generation(text),
        "new_token_ids": [int(x) for x in new.detach().cpu().tolist()],
    }
    if scores is not None:
        result["first_step_scores"] = scores.detach().float().cpu()
    del out
    return result


def run_generate_injected(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    decoder_layers: Sequence[Any],
    layer: int,
    head: int,
    source_state_cpu: torch.Tensor,
    gen_kw: Mapping[str, Any],
) -> Dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    with InjectOneHeadAtPromptLast(
        decoder_layers=decoder_layers,
        layer=layer,
        head=head,
        source_state_cpu=source_state_cpu,
        prompt_length=prompt_length,
    ) as patch:
        result = run_generate(model, processor, batch, gen_kw)
    if patch.applications != 1:
        raise RuntimeError(f"Generation patch count={patch.applications} for L{layer}H{head}")
    return result


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def corr(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": float("nan"), "spearman": float("nan")}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    pear = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")
    rx, ry = rank_average(x), rank_average(y)
    spear = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 1e-12 and np.std(ry) > 1e-12 else float("nan")
    return {"n": len(pairs), "pearson": pear, "spearman": spear}


def summarize_scan(rows: Sequence[Mapping[str, Any]], selected_heads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    info = {(int(h["layer"]), int(h["head"])): h for h in selected_heads}
    groups: Dict[Tuple[int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["layer"]), int(r["head"]), str(r["direction"]))].append(r)

    per_direction: List[Dict[str, Any]] = []
    for (layer, head, direction), vals in groups.items():
        meta = info[(layer, head)]
        per_direction.append({
            "layer": layer,
            "head": head,
            "direction": direction,
            "centroid_accuracy": float(meta["centroid_accuracy"]),
            "N": len(vals),
            "baseline_target_accuracy": safe_mean(int(bool(v["baseline_target_correct"])) for v in vals),
            "patched_target_accuracy": safe_mean(int(bool(v["patched_target_correct"])) for v in vals),
            "prediction_change_rate": safe_mean(int(bool(v["prediction_changed"])) for v in vals),
            "became_source_gt_rate": safe_mean(int(bool(v["became_source_gt"])) for v in vals),
            "exact_branch_reversal_rate": safe_mean(int(bool(v["exact_branch_reversal"])) for v in vals),
            "crossed_to_source_side_rate": safe_mean(int(bool(v["crossed_to_source_side"])) for v in vals),
            "mean_delta_source_margin": safe_mean(float(v["delta_source_margin"]) for v in vals),
            "mean_abs_delta_source_margin": safe_mean(abs(float(v["delta_source_margin"])) for v in vals),
        })

    head_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_direction:
        head_groups[(row["layer"], row["head"])].append(row)

    per_head: List[Dict[str, Any]] = []
    for (layer, head), vals in head_groups.items():
        meta = info[(layer, head)]
        per_head.append({
            "layer": layer,
            "head": head,
            "centroid_accuracy": float(meta["centroid_accuracy"]),
            "centroid_rank": int(meta.get("centroid_rank", -1)),
            "directions_n": len(vals),
            "mean_prediction_change_rate": safe_mean(v["prediction_change_rate"] for v in vals),
            "mean_became_source_gt_rate": safe_mean(v["became_source_gt_rate"] for v in vals),
            "mean_exact_branch_reversal_rate": safe_mean(v["exact_branch_reversal_rate"] for v in vals),
            "mean_crossed_to_source_side_rate": safe_mean(v["crossed_to_source_side_rate"] for v in vals),
            "mean_delta_source_margin": safe_mean(v["mean_delta_source_margin"] for v in vals),
            "mean_abs_delta_source_margin": safe_mean(v["mean_abs_delta_source_margin"] for v in vals),
        })

    per_head.sort(key=lambda r: (-r["mean_prediction_change_rate"], -r["mean_abs_delta_source_margin"], -r["centroid_accuracy"]))
    for rank, row in enumerate(per_head, 1):
        row["causal_rank"] = rank

    cacc = [r["centroid_accuracy"] for r in per_head]
    correlations = {
        "centroid_vs_prediction_change": corr(cacc, [r["mean_prediction_change_rate"] for r in per_head]),
        "centroid_vs_exact_branch_reversal": corr(cacc, [r["mean_exact_branch_reversal_rate"] for r in per_head]),
        "centroid_vs_delta_source_margin": corr(cacc, [r["mean_delta_source_margin"] for r in per_head]),
        "centroid_vs_abs_delta_source_margin": corr(cacc, [r["mean_abs_delta_source_margin"] for r in per_head]),
    }
    return {"per_direction": per_direction, "per_head": per_head, "correlations": correlations}


def summarize_generation(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["layer"]), int(r["head"]), str(r["direction"]))].append(r)
    out: List[Dict[str, Any]] = []
    for (layer, head, direction), vals in groups.items():
        out.append({
            "layer": layer,
            "head": head,
            "direction": direction,
            "N": len(vals),
            "baseline_parse_rate": safe_mean(int(v["baseline_prediction"] is not None) for v in vals),
            "patched_parse_rate": safe_mean(int(v["patched_prediction"] is not None) for v in vals),
            "prediction_change_rate": safe_mean(int(bool(v["prediction_changed"])) for v in vals),
            "became_source_gt_rate": safe_mean(int(bool(v["became_source_gt"])) for v in vals),
            "exact_branch_reversal_rate": safe_mean(int(bool(v["exact_branch_reversal"])) for v in vals),
        })
    out.sort(key=lambda r: (-r["prediction_change_rate"], -r["exact_branch_reversal_rate"], r["layer"], r["head"], r["direction"]))
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def print_scan_summary(summary: Mapping[str, Any], limit: int = 30) -> None:
    rows = summary["per_head"][:limit]
    print("\n" + "=" * 112)
    print("SWAPPED-HEAD STATE INJECTION — TOP CAUSAL HEADS")
    print("=" * 112)
    print("rank head     cent_acc  change   srcGT   exactRev  crossSide  dSrcMargin  |dMargin|")
    for r in rows:
        print(
            f"{int(r['causal_rank']):4d} L{int(r['layer']):02d}H{int(r['head']):02d} "
            f"{float(r['centroid_accuracy']):8.4f} "
            f"{float(r['mean_prediction_change_rate']):7.4f} "
            f"{float(r['mean_became_source_gt_rate']):7.4f} "
            f"{float(r['mean_exact_branch_reversal_rate']):9.4f} "
            f"{float(r['mean_crossed_to_source_side_rate']):9.4f} "
            f"{float(r['mean_delta_source_margin']):10.4f} "
            f"{float(r['mean_abs_delta_source_margin']):9.4f}"
        )
    print("\nCentroid accuracy correlations:")
    for name, c in summary["correlations"].items():
        print(f"  {name:42s} n={c['n']:3d} pearson={c['pearson']:+.4f} spearman={c['spearman']:+.4f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.top_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Invalid top-k/max-new-tokens")

    directions = list(dict.fromkeys(csv_tokens(args.directions)))
    bad = [x for x in directions if x not in DIRECTIONS]
    if bad or not directions:
        raise ValueError(f"Invalid directions {directions}; allowed={DIRECTIONS}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    step1_dir = Path(args.step1_dir)
    cfg1 = load_step1_config(step1_dir)
    model_name = args.model or cfg1.get("model") or "qwen-3b"
    dataset = args.dataset or cfg1.get("dataset") or "coco_two"
    data_root = Path(args.data_root) if args.data_root else Path(cfg1.get("data_root", "data"))
    prompt_jsonl = args.prompt_jsonl if args.prompt_jsonl is not None else cfg1.get("prompt_jsonl")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_path = out_dir / "scan_rows.jsonl"
    scan_errors_path = out_dir / "scan_errors.jsonl"
    gen_path = out_dir / "generation_rows.jsonl"
    gen_errors_path = out_dir / "generation_errors.jsonl"

    support = base.import_two_object_module()
    records, audit = support.load_records(dataset, data_root, None)
    records_by_sid = {int(r.sid): r for r in records}
    prompt_args = argparse.Namespace(dataset=dataset, prompt_jsonl=prompt_jsonl)
    prompt_path = base.resolve_prompt_path(prompt_args)
    prompt_rows = base.load_standard_prompts(prompt_path)

    sids = sorted(set(records_by_sid) & set(prompt_rows))
    if args.max_samples is not None:
        sids = sids[: int(args.max_samples)]
    if not sids:
        raise RuntimeError("No samples selected")

    if model_name not in support.SPECS:
        raise KeyError(f"Unknown model {model_name}; choices={list(support.SPECS)}")
    spec = support.SPECS[model_name]
    model_cls = getattr(base.transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers {base.transformers.__version__} has no {spec.model_class}")

    print(f"Loading {model_name}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()
    processor = base.AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)
    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    label_ids = base.label_token_id_variants(processor.tokenizer)

    centroid_acc, centroid_layers = load_centroid_accuracy(step1_dir)
    selected_heads = build_selected_heads(
        decoder_layers=decoder_layers,
        centroid_acc=centroid_acc,
        centroid_layers=centroid_layers,
        selection=args.head_selection,
        top_k=args.top_k,
    )
    selected_layer_ids = sorted(set(int(h["layer"]) for h in selected_heads))

    # validate geometry and head ids
    layer_geometry: Dict[int, Tuple[int, int, int]] = {}
    for layer_id in selected_layer_ids:
        geom = resolve_head_geometry(find_self_attention(decoder_layers[layer_id]))
        layer_geometry[layer_id] = geom
    for h in selected_heads:
        n_heads = layer_geometry[int(h["layer"])][0]
        if not 0 <= int(h["head"]) < n_heads:
            raise RuntimeError(f"Invalid selected head L{h['layer']}H{h['head']} n_heads={n_heads}")

    print(f"decoder={decoder_path} layers={len(decoder_layers)} selected_heads={len(selected_heads)}")
    print("Selected heads:")
    for i, h in enumerate(selected_heads[:40], 1):
        print(f"  {i:02d}. L{h['layer']:02d}H{h['head']:02d} centroid={h['centroid_accuracy']:.4f}")
    if len(selected_heads) > 40:
        print(f"  ... {len(selected_heads)-40} more")

    config = {
        "script_version": SCRIPT_VERSION,
        "step1_dir": str(step1_dir),
        "model": model_name,
        "repo_id": spec.repo_id,
        "dataset": dataset,
        "data_root": str(data_root),
        "prompt_jsonl": str(prompt_path),
        "decoder_path": decoder_path,
        "head_selection": args.head_selection,
        "top_k": args.top_k,
        "selected_heads": selected_heads,
        "directions": directions,
        "max_samples": args.max_samples,
        "sample_status": args.sample_status,
        "generation_top_k": args.generation_top_k,
        "max_new_tokens": args.max_new_tokens,
        "intervention": {
            "source": "other query branch's real pre-W_O head output at prompt-last",
            "target": "same layer/head prompt-last in current branch",
            "downstream": "all downstream computation live/recomputed",
            "object_token_swap": False,
            "lm_direction_constructed": False,
        },
        "audit_count": len(audit),
        "transformers_version": base.transformers.__version__,
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "selected_heads.json", selected_heads)

    # ------------------------------------------------------------------
    # Stage 1: first-answer-token scan
    # ------------------------------------------------------------------
    previous_scan = read_jsonl(scan_path)
    done_scan = {(int(r["sid"]), int(r["layer"]), int(r["head"]), str(r["direction"])) for r in previous_scan if r.get("status") == "ok"}
    scan_rows = list(previous_scan)
    accepted_sids: List[int] = []
    started = time.time()

    for sample_i, sid in enumerate(tqdm(sids, desc=f"scan:{model_name}"), 1):
        image = original_batch = swapped_batch = None
        try:
            prompt = prompt_rows[sid]
            subject = str(prompt["subject"])
            reference = str(prompt["reference"])
            gt = normalize_relation(prompt["answer_raw"])
            if gt not in RELATIONS:
                raise RuntimeError(f"Invalid GT {gt}")
            swapped_gt = OPPOSITE[gt]
            original_question = str(prompt["question_text"])
            swapped_question = build_swapped_question(subject, reference)
            image = base.record_image(records_by_sid[sid])
            original_batch = base.make_question_batch(processor=processor, image=image, question_text=original_question, device=device)
            swapped_batch = base.make_question_batch(processor=processor, image=image, question_text=swapped_question, device=device)

            original_base, original_states = forward_capture(
                model=model, batch=original_batch, decoder_layers=decoder_layers,
                layer_ids=selected_layer_ids, label_ids=label_ids,
            )
            swapped_base, swapped_states = forward_capture(
                model=model, batch=swapped_batch, decoder_layers=decoder_layers,
                layer_ids=selected_layer_ids, label_ids=label_ids,
            )
            original_correct = original_base["prediction"] == gt
            swapped_correct = swapped_base["prediction"] == swapped_gt
            status_ok = (
                args.sample_status == "all"
                or (args.sample_status == "both_correct" and original_correct and swapped_correct)
                or (args.sample_status == "original_correct" and original_correct)
                or (args.sample_status == "swapped_correct" and swapped_correct)
            )
            if not status_ok:
                continue
            accepted_sids.append(sid)

            for hinfo in selected_heads:
                layer, head = int(hinfo["layer"]), int(hinfo["head"])
                for direction in directions:
                    key = (sid, layer, head, direction)
                    if key in done_scan:
                        continue
                    if direction == "swapped_into_original":
                        target_batch = original_batch
                        target_base = original_base
                        target_gt = gt
                        source_base = swapped_base
                        source_gt = swapped_gt
                        source_state = swapped_states[layer]
                    else:
                        target_batch = swapped_batch
                        target_base = swapped_base
                        target_gt = swapped_gt
                        source_base = original_base
                        source_gt = gt
                        source_state = original_states[layer]

                    patched = forward_injected(
                        model=model, batch=target_batch, decoder_layers=decoder_layers,
                        layer=layer, head=head, source_state_cpu=source_state, label_ids=label_ids,
                    )
                    baseline_margin = branch_margin(target_base["scores"], source_gt, target_gt)
                    patched_margin = branch_margin(patched["scores"], source_gt, target_gt)
                    row = {
                        "status": "ok",
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "gt": gt,
                        "swapped_gt": swapped_gt,
                        "direction": direction,
                        "layer": layer,
                        "head": head,
                        "centroid_accuracy": float(hinfo["centroid_accuracy"]),
                        "target_gt": target_gt,
                        "source_gt": source_gt,
                        "baseline_target_prediction": target_base["prediction"],
                        "source_branch_prediction": source_base["prediction"],
                        "patched_prediction": patched["prediction"],
                        "baseline_target_correct": bool(target_base["prediction"] == target_gt),
                        "patched_target_correct": bool(patched["prediction"] == target_gt),
                        "prediction_changed": bool(patched["prediction"] != target_base["prediction"]),
                        "became_source_branch_prediction": bool(patched["prediction"] == source_base["prediction"]),
                        "became_source_gt": bool(patched["prediction"] == source_gt),
                        "exact_branch_reversal": bool(target_base["prediction"] == target_gt and patched["prediction"] == source_gt),
                        "baseline_source_margin": baseline_margin,
                        "patched_source_margin": patched_margin,
                        "delta_source_margin": patched_margin - baseline_margin,
                        "crossed_to_source_side": bool(baseline_margin < 0 <= patched_margin),
                    }
                    append_jsonl(scan_path, row)
                    scan_rows.append(row)
                    done_scan.add(key)

            if args.print_every > 0 and sample_i % args.print_every == 0:
                tqdm.write(
                    f"sid={sid} GT={gt}/{swapped_gt} base={original_base['prediction']}/{swapped_base['prediction']} "
                    f"accepted={len(accepted_sids)} rows={len(scan_rows)}"
                )
        except Exception as exc:
            append_jsonl(scan_errors_path, {
                "sid": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            raise
        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            del original_batch, swapped_batch
            gc.collect()
            if torch.cuda.is_available() and args.empty_cache_every > 0 and sample_i % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    scan_summary = summarize_scan(scan_rows, selected_heads)
    write_json(out_dir / "scan_summary.json", scan_summary)
    write_csv(out_dir / "scan_per_head.csv", scan_summary["per_head"])
    write_csv(out_dir / "scan_per_direction.csv", scan_summary["per_direction"])
    print_scan_summary(scan_summary, limit=min(30, len(scan_summary["per_head"])))

    # ------------------------------------------------------------------
    # Stage 2: full generation for top scan heads
    # ------------------------------------------------------------------
    generation_rows = read_jsonl(gen_path)
    if args.generation_top_k > 0 and scan_summary["per_head"]:
        generation_heads = scan_summary["per_head"][: int(args.generation_top_k)]
        generation_head_keys = {(int(r["layer"]), int(r["head"])) for r in generation_heads}
        generation_layer_ids = sorted({layer for layer, _ in generation_head_keys})
        gen_kw = generation_kwargs(processor, args)
        done_gen = {(int(r["sid"]), int(r["layer"]), int(r["head"]), str(r["direction"])) for r in generation_rows if r.get("status") == "ok"}

        print("\nFull-generation validation heads:")
        for r in generation_heads:
            print(
                f"  L{int(r['layer']):02d}H{int(r['head']):02d} "
                f"change={float(r['mean_prediction_change_rate']):.4f} "
                f"exactRev={float(r['mean_exact_branch_reversal_rate']):.4f} "
                f"centroid={float(r['centroid_accuracy']):.4f}"
            )

        for sample_i, sid in enumerate(tqdm(sorted(set(accepted_sids)), desc=f"generation:{model_name}"), 1):
            image = original_batch = swapped_batch = None
            try:
                prompt = prompt_rows[sid]
                subject = str(prompt["subject"])
                reference = str(prompt["reference"])
                gt = normalize_relation(prompt["answer_raw"])
                swapped_gt = OPPOSITE[gt]
                original_question = str(prompt["question_text"])
                swapped_question = build_swapped_question(subject, reference)
                image = base.record_image(records_by_sid[sid])
                original_batch = base.make_question_batch(processor=processor, image=image, question_text=original_question, device=device)
                swapped_batch = base.make_question_batch(processor=processor, image=image, question_text=swapped_question, device=device)

                # Capture only needed source layers once, then run baseline generation once per branch.
                original_forward, original_states = forward_capture(
                    model=model, batch=original_batch, decoder_layers=decoder_layers,
                    layer_ids=generation_layer_ids, label_ids=label_ids,
                )
                swapped_forward, swapped_states = forward_capture(
                    model=model, batch=swapped_batch, decoder_layers=decoder_layers,
                    layer_ids=generation_layer_ids, label_ids=label_ids,
                )
                original_gen = run_generate(model, processor, original_batch, gen_kw)
                swapped_gen = run_generate(model, processor, swapped_batch, gen_kw)

                for layer, head in sorted(generation_head_keys):
                    for direction in directions:
                        key = (sid, layer, head, direction)
                        if key in done_gen:
                            continue
                        if direction == "swapped_into_original":
                            target_batch = original_batch
                            baseline_gen = original_gen
                            source_gen = swapped_gen
                            target_gt = gt
                            source_gt = swapped_gt
                            source_state = swapped_states[layer]
                        else:
                            target_batch = swapped_batch
                            baseline_gen = swapped_gen
                            source_gen = original_gen
                            target_gt = swapped_gt
                            source_gt = gt
                            source_state = original_states[layer]

                        patched = run_generate_injected(
                            model=model, processor=processor, batch=target_batch,
                            decoder_layers=decoder_layers, layer=layer, head=head,
                            source_state_cpu=source_state, gen_kw=gen_kw,
                        )
                        row = {
                            "status": "ok",
                            "sid": sid,
                            "gt": gt,
                            "swapped_gt": swapped_gt,
                            "direction": direction,
                            "layer": layer,
                            "head": head,
                            "target_gt": target_gt,
                            "source_gt": source_gt,
                            "baseline_text": baseline_gen["text"],
                            "baseline_prediction": baseline_gen["prediction"],
                            "source_text": source_gen["text"],
                            "source_prediction": source_gen["prediction"],
                            "patched_text": patched["text"],
                            "patched_prediction": patched["prediction"],
                            "prediction_changed": bool(
                                baseline_gen["prediction"] is not None
                                and patched["prediction"] is not None
                                and baseline_gen["prediction"] != patched["prediction"]
                            ),
                            "became_source_gt": bool(patched["prediction"] == source_gt),
                            "became_source_branch_prediction": bool(
                                source_gen["prediction"] is not None and patched["prediction"] == source_gen["prediction"]
                            ),
                            "exact_branch_reversal": bool(
                                baseline_gen["prediction"] == target_gt and patched["prediction"] == source_gt
                            ),
                        }
                        append_jsonl(gen_path, row)
                        generation_rows.append(row)
                        done_gen.add(key)

            except Exception as exc:
                append_jsonl(gen_errors_path, {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del original_batch, swapped_batch
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache_every > 0 and sample_i % args.empty_cache_every == 0:
                    torch.cuda.empty_cache()

    generation_summary = summarize_generation(generation_rows)
    write_json(out_dir / "generation_summary.json", generation_summary)
    write_csv(out_dir / "generation_summary.csv", generation_summary)

    final_summary = {
        "script_version": SCRIPT_VERSION,
        "model": model_name,
        "selected_samples": len(accepted_sids),
        "selected_heads": len(selected_heads),
        "scan": scan_summary,
        "generation": generation_summary,
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "summary.json", final_summary)

    if generation_summary:
        print("\n" + "=" * 104)
        print("FULL GENERATION VALIDATION")
        print("=" * 104)
        print("head     direction               N   changed   srcGT   exactRev")
        for r in generation_summary[:40]:
            print(
                f"L{int(r['layer']):02d}H{int(r['head']):02d} "
                f"{str(r['direction']):24s} {int(r['N']):3d} "
                f"{float(r['prediction_change_rate']):9.4f} "
                f"{float(r['became_source_gt_rate']):7.4f} "
                f"{float(r['exact_branch_reversal_rate']):9.4f}"
            )

    print(f"\nSaved to: {out_dir}")
    print(f"Elapsed: {(time.time()-started)/60:.1f} min")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
