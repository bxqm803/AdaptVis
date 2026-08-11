#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-centroid single-head counterfactual directional generation experiment.

Input
-----
A Step-1 directory produced by:
    analyze_coco_attention_flow_swap_step1_v1.py

No old prior/trace output is required.

Experiment
----------
1. Rank all heads by Step-1 `attention_average_accuracy` and keep Top-K.
2. For each sample, reconstruct each selected head's object-derived contribution
   to the final prompt token using the same A*V -> isolated W_O procedure as
   eval_directional_head_repair_v1.py.
3. By default, use THAT HEAD'S sample-specific Step-1 centroid prediction as
   the source relation. Target is the opposite relation:
       left <-> right, above <-> below.
4. Let d = normalize(W_source - W_target), alpha = <v_h, d>.
   remove: delta = -strength * alpha * d
   flip:   delta = -2 * strength * alpha * d
   At strength=1, flip reflects the signed relation-axis component alpha -> -alpha.
5. Patch only this single head-derived residual contribution at the selected
   layer's attention output on the final PREFILL token, then run model.generate().

Default source: --source head-centroid
Alternatives:   --source baseline | gt

Important metrics
-----------------
- prediction_change_rate
- target_hit_rate
- exact_flip_rate_given_baseline_matches_source
- exact_flip_rate_given_clean
  clean := baseline == source == GT
- exact_flip_rate_given_clean_and_alpha_positive
- mean first-token target-vs-source margin delta
- correlation of centroid accuracy with the above causal effects

Dependencies (repository root)
------------------------------
- trace_centroid_generation_groups_v2_1.py
- eval_directional_head_repair_v1.py
- extract_two_object_relation_states.py

Example
-------
CUDA_VISIBLE_DEVICES=0 python \
  eval_coco_top20_centroid_head_counterfactual_flip_generation_v1.py \
  --step1-dir output/qwen3b_coco_attention_flow_swap_step1 \
  --top-k 20 \
  --source head-centroid \
  --variants flip \
  --strength 1.0 \
  --max-new-tokens 8 \
  --device cuda:0 \
  --output-dir output/qwen3b_top20_centroid_head_counterfactual_flip \
  --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import trace_centroid_generation_groups_v2_1 as base
import eval_directional_head_repair_v1 as repair

SCRIPT_VERSION = "eval-coco-top20-centroid-head-counterfactual-flip-generation-v1"
RELATIONS = ("left", "right", "above", "below")
INDEX_TO_RELATION = {i: r for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
ALLOWED_VARIANTS = ("remove", "flip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step1-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--prompt-jsonl", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--source",
        choices=["head-centroid", "baseline", "gt"],
        default="head-centroid",
    )
    p.add_argument("--variants", default="flip")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--min-new-tokens", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--only-step1-correct", action="store_true")
    p.add_argument("--only-current-baseline-correct", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Could not parse {path}:{line_no}: {exc}") from exc
    return rows


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def normalize_relation(value: Any) -> Optional[str]:
    x = base.normalize_relation(value)
    return x if x in RELATIONS else None


def one_line(text: Any) -> str:
    return " ".join(str(text).split())


# -----------------------------------------------------------------------------
# Free-generation parsing
# -----------------------------------------------------------------------------

RELATION_PATTERNS: Dict[str, Sequence[str]] = {
    "left": (r"\bleft\s+of\b", r"\bto\s+the\s+left\b", r"\bon\s+the\s+left\b", r"\bleft\b"),
    "right": (r"\bright\s+of\b", r"\bto\s+the\s+right\b", r"\bon\s+the\s+right\b", r"\bright\b"),
    "above": (r"\bon\s+top\s+of\b", r"\batop\b", r"\babove\b", r"\bover\b", r"\bon\b"),
    "below": (r"\bunderneath\b", r"\bbeneath\b", r"\bbelow\b", r"\bunder\b"),
}


def parse_generated_relation(text: str) -> Optional[str]:
    text = one_line(text).lower()
    candidates: List[Tuple[int, int, str]] = []
    for relation, patterns in RELATION_PATTERNS.items():
        for priority, pattern in enumerate(patterns):
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m is not None:
                candidates.append((m.start(), priority, relation))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


# -----------------------------------------------------------------------------
# Step-1 loading / Top-K
# -----------------------------------------------------------------------------

def load_step1_config(step1_dir: Path) -> Dict[str, Any]:
    p = step1_dir / "config.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    p = step1_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing config.json/summary.json in {step1_dir}")
    summary = json.loads(p.read_text(encoding="utf-8"))
    config = summary.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"No config object in {p}")
    return config


def load_top_heads(step1_dir: Path, top_k: int) -> List[Dict[str, Any]]:
    p = step1_dir / "aggregate_metrics.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        acc = np.asarray(z["attention_average_accuracy"], dtype=np.float64)
        layers = np.asarray(
            z["layer_indices"] if "layer_indices" in z else np.arange(acc.shape[0]),
            dtype=np.int64,
        )
        map_cos = (
            np.asarray(z["same_object_map_cosine"], dtype=np.float64).mean(-1)
            if "same_object_map_cosine" in z else np.full_like(acc, np.nan)
        )
        sep = (
            0.5 * (
                np.asarray(z["original_object_separation"], dtype=np.float64)
                + np.asarray(z["swapped_object_separation"], dtype=np.float64)
            )
            if "original_object_separation" in z and "swapped_object_separation" in z
            else np.full_like(acc, np.nan)
        )
        if "original_prompt_visual_mass" in z and "swapped_prompt_visual_mass" in z:
            ov = np.asarray(z["original_prompt_visual_mass"], dtype=np.float64)
            sv = np.asarray(z["swapped_prompt_visual_mass"], dtype=np.float64)
            vis = 0.25 * (ov[:, :, 0] + ov[:, :, 1] + sv[:, :, 0] + sv[:, :, 1])
        else:
            vis = np.full_like(acc, np.nan)

    if acc.ndim != 2 or len(layers) != acc.shape[0]:
        raise RuntimeError(f"Invalid Step1 aggregate shapes: acc={acc.shape}, layers={layers.shape}")

    order = np.argsort(acc.reshape(-1))[::-1][:top_k]
    rows = []
    for rank, flat in enumerate(order, 1):
        lp, h = np.unravel_index(int(flat), acc.shape)
        rows.append({
            "rank": rank,
            "layer_position": int(lp),
            "layer": int(layers[lp]),
            "head": int(h),
            "centroid_accuracy": float(acc[lp, h]),
            "map_cosine": float(map_cos[lp, h]),
            "object_separation": float(sep[lp, h]),
            "object_visual_mass": float(vis[lp, h]),
        })
    return rows


def load_step1_samples(step1_dir: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(step1_dir / "samples.jsonl")
    out = {int(r["sid"]): r for r in rows if "sid" in r}
    if not out:
        raise RuntimeError("No Step1 sample rows")
    return out


def sample_head_centroids(
    step1_dir: Path,
    sid: int,
    top_heads: Sequence[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    p = step1_dir / "sample_arrays" / f"{sid}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        pred = np.asarray(z["attention_average_prediction"])
        layers = [
            int(x) for x in np.asarray(
                z["layer_indices"] if "layer_indices" in z else np.arange(pred.shape[0])
            ).tolist()
        ]
    layer_pos = {layer: i for i, layer in enumerate(layers)}
    out: Dict[str, Optional[str]] = {}
    for info in top_heads:
        layer, head = int(info["layer"]), int(info["head"])
        key = f"{layer}:{head}"
        pos = layer_pos.get(layer)
        if pos is None or head >= pred.shape[1]:
            out[key] = None
        else:
            out[key] = INDEX_TO_RELATION.get(int(pred[pos, head]))
    return out


# -----------------------------------------------------------------------------
# Baseline trace: same A*V -> isolated W_O reconstruction as old repair script
# -----------------------------------------------------------------------------

def reconstruct_head_vectors(
    *,
    model: Any,
    processor: Any,
    collector: Any,
    batch: Dict[str, Any],
    subject: str,
    reference: str,
    requested_heads: Sequence[Tuple[int, int]],
    label_ids: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    input_ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    s_span, r_span = base.locate_object_spans(
        processor.tokenizer, input_ids, subject, reference
    )
    s_idx = base.span_indices(s_span)
    r_idx = base.span_indices(r_span)
    v_idx = base.resolve_visual_indices(model, processor, batch, input_ids)
    last_index = len(input_ids) - 1

    collector.set_sample(
        subject_indices=s_idx,
        reference_indices=r_idx,
        visual_indices=v_idx,
        last_index=last_index,
    )
    outputs = None
    try:
        with torch.inference_mode():
            outputs = model(
                **batch,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        collector.active = False

    if outputs is None:
        raise RuntimeError("Trace forward returned no outputs")
    attentions = base.attention_tuple(outputs)
    vectors = repair.reconstruct_heads(
        base,
        attentions,
        collector,
        s_idx,
        r_idx,
        last_index,
        requested_heads,
    )
    scores = repair.relation_scores_from_logits(
        outputs.logits[0, -1], label_ids, RELATIONS
    )
    info = {
        "prompt_length": len(input_ids),
        "last_index": last_index,
        "subject_span": [int(x) for x in s_span],
        "reference_span": [int(x) for x in r_span],
        "n_visual_tokens": len(v_idx),
        "forward_relation_prediction": RELATIONS[int(np.argmax(scores))],
        "forward_relation_scores": {r: float(scores[i]) for i, r in enumerate(RELATIONS)},
    }
    del outputs, attentions
    return vectors, info


# -----------------------------------------------------------------------------
# Directional edit
# -----------------------------------------------------------------------------

def relation_direction(
    relation_vectors: Mapping[str, torch.Tensor],
    source: str,
) -> torch.Tensor:
    target = OPPOSITE[source]
    d = relation_vectors[source].float() - relation_vectors[target].float()
    norm = float(d.norm().item())
    if norm <= 1e-12:
        raise RuntimeError(f"Degenerate direction {source} vs {target}")
    return d / norm


def build_delta(
    vector_cpu: torch.Tensor,
    direction_cpu: torch.Tensor,
    mode: str,
    strength: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    v = vector_cpu.float()
    d = direction_cpu.to(v).float()
    alpha = float(torch.dot(v, d).item())
    if mode == "remove":
        delta = -strength * alpha * d
    elif mode == "flip":
        delta = -2.0 * strength * alpha * d
    else:
        raise ValueError(mode)
    alpha_delta = float(torch.dot(delta, d).item())
    return delta.cpu(), {
        "alpha_source_before": alpha,
        "alpha_delta": alpha_delta,
        "alpha_source_after_estimate": alpha + alpha_delta,
        "alpha_positive_before": bool(alpha > 0.0),
        "vector_norm": float(v.norm().item()),
        "delta_norm": float(delta.norm().item()),
    }


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def make_generation_kwargs(processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    tok = processor.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kw: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
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


def first_step_relation_scores(
    output: Any,
    label_ids: Mapping[str, Sequence[int]],
) -> Optional[Dict[str, float]]:
    scores = getattr(output, "scores", None)
    if not scores:
        return None
    x = scores[0][0].float()
    return {
        r: float(x.index_select(0, torch.as_tensor(label_ids[r], device=x.device)).max().item())
        for r in RELATIONS
    }


def decode_output(
    processor: Any,
    output: Any,
    prompt_length: int,
    label_ids: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    seq = output.sequences[0]
    new = seq[prompt_length:] if len(seq) >= prompt_length else seq
    ids = [int(x) for x in new.detach().cpu().tolist()]
    text = processor.tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text = one_line(text)
    pred = parse_generated_relation(text)
    return {
        "text": text,
        "token_ids": ids,
        "prediction": pred,
        "parsed": pred is not None,
        "relation_scores_first_step": first_step_relation_scores(output, label_ids),
    }


def baseline_generate(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    prompt_length: int,
    generation_kwargs: Dict[str, Any],
    label_ids: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    with torch.inference_mode():
        out = model.generate(**batch, **generation_kwargs)
    result = decode_output(processor, out, prompt_length, label_ids)
    del out
    return result


def patched_generate(
    *,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    attention_module: torch.nn.Module,
    delta_cpu: torch.Tensor,
    prompt_length: int,
    generation_kwargs: Dict[str, Any],
    label_ids: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    state = {"calls": 0, "applications": 0}

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        state["calls"] += 1
        tensor = base.first_tensor(output)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(f"Unexpected attention output {tuple(tensor.shape)}")
        seq_len = int(tensor.shape[1])
        if seq_len != prompt_length or state["applications"] != 0:
            return output
        modified = tensor.clone()
        delta = delta_cpu.to(device=modified.device, dtype=modified.dtype)
        if delta.numel() != modified.shape[-1]:
            raise RuntimeError(
                f"Delta width {delta.numel()} != hidden width {modified.shape[-1]}"
            )
        modified[0, seq_len - 1, :] += delta
        state["applications"] += 1
        return repair.replace_first_tensor(output, modified)

    handle = attention_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model.generate(**batch, **generation_kwargs)
    finally:
        handle.remove()
    if state["applications"] != 1:
        raise RuntimeError(
            f"Expected one prefill patch, got {state['applications']} calls={state['calls']}"
        )
    result = decode_output(processor, out, prompt_length, label_ids)
    del out
    return result, {"module_calls": state["calls"], "applications": state["applications"]}


def target_margin(scores: Optional[Mapping[str, float]], source: str) -> float:
    if scores is None:
        return float("nan")
    target = OPPOSITE[source]
    return float(scores[target] - scores[source])


def choose_source(
    mode: str,
    head_centroid: Optional[str],
    baseline_prediction: Optional[str],
    gt: str,
) -> Optional[str]:
    if mode == "head-centroid":
        return head_centroid
    if mode == "baseline":
        return baseline_prediction
    if mode == "gt":
        return gt
    raise ValueError(mode)


# -----------------------------------------------------------------------------
# Summary
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
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": float("nan"), "spearman": float("nan")}
    x = np.asarray([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")
    rx, ry = rank_average(x), rank_average(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 1e-12 and np.std(ry) > 1e-12 else float("nan")
    return {"n": len(pairs), "pearson": pearson, "spearman": spearman}


def summarize_head(
    rows: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
    variant: str,
) -> Dict[str, Any]:
    key = f"{info['layer']}:{info['head']}"
    vals = []
    for row in rows:
        h = row.get("heads", {}).get(key, {})
        v = h.get("variants", {}).get(variant)
        if isinstance(v, dict):
            vals.append((row, v))
    if not vals:
        return {**info, "variant": variant, "n": 0}

    def mean_bool(name: str) -> float:
        return float(np.mean([bool(v[name]) for _, v in vals]))

    base_source = [(row, v) for row, v in vals if v["baseline_matches_source"]]
    clean = [(row, v) for row, v in vals if v["clean"]]
    clean_alpha = [(row, v) for row, v in clean if v["delta"]["alpha_positive_before"]]

    def subset_rate(subset: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], field: str) -> float:
        return float(np.mean([bool(v[field]) for _, v in subset])) if subset else float("nan")

    base_correct = np.asarray([row["baseline"]["prediction"] == row["gt"] for row, _ in vals], dtype=bool)
    new_correct = np.asarray([v["prediction"] == row["gt"] for row, v in vals], dtype=bool)

    return {
        **info,
        "variant": variant,
        "n": len(vals),
        "baseline_accuracy": float(base_correct.mean()),
        "patched_accuracy": float(new_correct.mean()),
        "accuracy_change": float(new_correct.mean() - base_correct.mean()),
        "alpha_positive_rate": float(np.mean([v["delta"]["alpha_positive_before"] for _, v in vals])),
        "prediction_change_rate": mean_bool("prediction_changed"),
        "target_hit_rate": mean_bool("target_hit"),
        "baseline_matches_source_n": len(base_source),
        "exact_flip_rate_given_baseline_matches_source": subset_rate(base_source, "exact_flip"),
        "clean_n": len(clean),
        "exact_flip_rate_given_clean": subset_rate(clean, "exact_flip"),
        "clean_alpha_positive_n": len(clean_alpha),
        "exact_flip_rate_given_clean_and_alpha_positive": subset_rate(clean_alpha, "exact_flip"),
        "fixed": int(sum(v["fixed"] for _, v in vals)),
        "broken": int(sum(v["broken"] for _, v in vals)),
        "mean_delta_target_minus_source_margin": safe_mean(
            v["delta_target_minus_source_margin"] for _, v in vals
        ),
    }


def build_summary(
    rows: Sequence[Dict[str, Any]],
    top_heads: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    parsed = [r for r in ok if r["baseline"]["prediction"] is not None]
    head_rows = [summarize_head(ok, info, v) for info in top_heads for v in variants]
    correlations = {}
    for variant in variants:
        subset = [r for r in head_rows if r["variant"] == variant and r.get("n", 0) > 0]
        for metric in (
            "prediction_change_rate",
            "target_hit_rate",
            "exact_flip_rate_given_baseline_matches_source",
            "exact_flip_rate_given_clean",
            "exact_flip_rate_given_clean_and_alpha_positive",
            "mean_delta_target_minus_source_margin",
        ):
            correlations[f"{variant}:{metric}"] = corr(
                [r["centroid_accuracy"] for r in subset],
                [r.get(metric, float("nan")) for r in subset],
            )
    return {
        "script_version": SCRIPT_VERSION,
        "config": config,
        "n_ok_samples": len(ok),
        "baseline_parse_rate": len(parsed) / len(ok) if ok else float("nan"),
        "baseline_accuracy_among_parsed": (
            float(np.mean([r["baseline"]["prediction"] == r["gt"] for r in parsed]))
            if parsed else float("nan")
        ),
        "current_vs_step1_generation_agreement": safe_mean(
            float(r["baseline"]["prediction"] == r.get("step1_original_prediction"))
            for r in ok
            if r["baseline"]["prediction"] is not None and r.get("step1_original_prediction") is not None
        ),
        "top_heads": list(top_heads),
        "head_variant_summary": head_rows,
        "centroid_accuracy_correlations": correlations,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 150)
    print("TOP-CENTROID SINGLE-HEAD COUNTERFACTUAL DIRECTIONAL GENERATION")
    print("=" * 150)
    print(
        f"samples={summary['n_ok_samples']} | "
        f"baseline_parse={summary['baseline_parse_rate']:.4f} | "
        f"baseline_acc={summary['baseline_accuracy_among_parsed']:.4f} | "
        f"Step1_agreement={summary['current_vs_step1_generation_agreement']:.4f}"
    )
    print("\nrank head     var     cent_acc alpha+   changed target_hit exact|base=src exact|clean exact|clean,a+ dTargetMargin")
    print("-" * 150)
    for r in summary["head_variant_summary"]:
        if r.get("n", 0) == 0:
            continue
        def f(key: str) -> str:
            x = float(r.get(key, float("nan")))
            return f"{x:.4f}" if math.isfinite(x) else "nan"
        print(
            f"{r['rank']:>4d} L{r['layer']:02d}H{r['head']:02d} "
            f"{r['variant']:<7s} {r['centroid_accuracy']:.4f} "
            f"{f('alpha_positive_rate'):>7s} {f('prediction_change_rate'):>8s} "
            f"{f('target_hit_rate'):>10s} {f('exact_flip_rate_given_baseline_matches_source'):>14s} "
            f"{f('exact_flip_rate_given_clean'):>11s} "
            f"{f('exact_flip_rate_given_clean_and_alpha_positive'):>14s} "
            f"{f('mean_delta_target_minus_source_margin'):>13s}"
        )
    print("\nCentroid accuracy vs causal effect:")
    for name, c in summary["centroid_accuracy_correlations"].items():
        print(f"  {name:60s} n={c['n']:2d} pearson={c['pearson']:+.4f} spearman={c['spearman']:+.4f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.top_k <= 0 or args.max_new_tokens <= 0 or args.strength < 0:
        raise ValueError("Invalid top-k/max-new-tokens/strength")

    variants = list(dict.fromkeys(csv_list(args.variants)))
    unknown = [v for v in variants if v not in ALLOWED_VARIANTS]
    if unknown or not variants:
        raise ValueError(f"Invalid --variants={variants}; allowed={ALLOWED_VARIANTS}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    step1_dir = Path(args.step1_dir)
    cfg1 = load_step1_config(step1_dir)
    model_name = args.model or cfg1.get("model")
    dataset = args.dataset or cfg1.get("dataset")
    data_root = Path(args.data_root) if args.data_root else Path(cfg1.get("data_root", "data"))
    prompt_jsonl = args.prompt_jsonl if args.prompt_jsonl is not None else cfg1.get("prompt_jsonl")
    if not model_name or not dataset:
        raise RuntimeError("Could not infer model/dataset from Step1 config")

    top_heads = load_top_heads(step1_dir, args.top_k)
    requested_heads = [(int(h["layer"]), int(h["head"])) for h in top_heads]
    print("\nTop heads:")
    for h in top_heads:
        print(
            f"  #{h['rank']:02d} L{h['layer']:02d}H{h['head']:02d} "
            f"acc={h['centroid_accuracy']:.4f} map_cos={h['map_cosine']:.4f} "
            f"sep={h['object_separation']:.4f} vis_mass={h['object_visual_mass']:.4f}"
        )

    step1_samples = load_step1_samples(step1_dir)
    sids = sorted(step1_samples)
    if args.only_step1_correct:
        sids = [sid for sid in sids if bool(step1_samples[sid].get("original_correct"))]
    if args.max_samples is not None:
        sids = sids[: args.max_samples]
    if not sids:
        raise RuntimeError("No samples selected")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    errors_path = out_dir / "errors.jsonl"
    top_path = out_dir / "top_heads.json"
    config_path = out_dir / "config.json"
    summary_path = out_dir / "summary.json"
    top_path.write_text(json.dumps(top_heads, ensure_ascii=False, indent=2), encoding="utf-8")
    done = {int(r["sid"]) for r in read_jsonl(samples_path) if r.get("status") == "ok" and "sid" in r}

    support = base.import_two_object_module()
    records, audit = support.load_records(dataset, data_root, None)
    record_by_sid = {int(r.sid): r for r in records}
    prompt_args = argparse.Namespace(dataset=dataset, prompt_jsonl=prompt_jsonl)
    prompt_path = base.resolve_prompt_path(prompt_args)
    prompt_rows = base.load_standard_prompts(prompt_path)
    missing = [sid for sid in sids if sid not in record_by_sid or sid not in prompt_rows]
    if missing:
        raise RuntimeError(f"Missing records/prompts: {missing[:10]}")

    spec = support.SPECS[model_name]
    model_cls = getattr(base.transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers {base.transformers.__version__} has no {spec.model_class}")
    print(f"\nLoading {model_name}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()
    processor = base.AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)
    layers, layers_path = base.resolve_decoder_layers(model)
    for layer, _head in requested_heads:
        if not 0 <= layer < len(layers):
            raise RuntimeError(f"Selected L{layer} outside model layers={len(layers)}")

    collector = base.LayerTraceCollector(layers, [])
    attention_modules = list(collector.attention_modules)
    label_ids = base.label_token_id_variants(processor.tokenizer)
    relation_vectors = repair.readout_vectors(model, label_ids, RELATIONS)
    gen_kwargs = make_generation_kwargs(processor, args)

    config = {
        "script_version": SCRIPT_VERSION,
        "step1_dir": str(step1_dir),
        "dataset": dataset,
        "data_root": str(data_root),
        "prompt_jsonl": str(prompt_path),
        "model": model_name,
        "repo_id": spec.repo_id,
        "transformers_version": base.transformers.__version__,
        "decoder_path": layers_path,
        "top_k": args.top_k,
        "source": args.source,
        "variants": variants,
        "strength": args.strength,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "only_step1_correct": args.only_step1_correct,
        "only_current_baseline_correct": args.only_current_baseline_correct,
        "selected_heads": top_heads,
        "n_candidate_samples": len(sids),
        "intervention": {
            "head_vector": "object-derived A*V projected through isolated W_O",
            "patch": "selected-layer attention output at final prefill token",
            "remove": "delta=-strength*alpha*d",
            "flip": "delta=-2*strength*alpha*d",
            "direction": "d=normalize(W_source-W_opposite(source))",
            "unconditional_signed_reflection": True,
        },
        "audit_count": len(audit),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\nRun: samples={len(sids)} source={args.source} variants={variants} "
        f"strength={args.strength} resume_done={len(done)}"
    )
    started = time.time()
    completed = 0

    try:
        for sid in tqdm(sids, desc=f"counterfactual:{model_name}"):
            if sid in done:
                continue
            image = batch = None
            try:
                prompt = prompt_rows[sid]
                subject = str(prompt["subject"])
                reference = str(prompt["reference"])
                question = str(prompt["question_text"])
                gt = normalize_relation(prompt["answer_raw"])
                if gt not in RELATIONS:
                    raise RuntimeError(f"Invalid GT {gt}")

                step1_row = step1_samples[sid]
                head_centroids = sample_head_centroids(step1_dir, sid, top_heads)
                image = base.record_image(record_by_sid[sid])
                batch = base.make_question_batch(
                    processor=processor, image=image, question_text=question, device=device
                )
                vectors, trace = reconstruct_head_vectors(
                    model=model,
                    processor=processor,
                    collector=collector,
                    batch=batch,
                    subject=subject,
                    reference=reference,
                    requested_heads=requested_heads,
                    label_ids=label_ids,
                )
                prompt_length = int(trace["prompt_length"])
                baseline = baseline_generate(
                    model, processor, batch, prompt_length, gen_kwargs, label_ids
                )
                baseline["correct"] = bool(baseline["prediction"] == gt)
                step1_pred = normalize_relation(step1_row.get("original_prediction"))

                row = {
                    "status": "ok",
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "question": question,
                    "gt": gt,
                    "step1_original_prediction": step1_pred,
                    "step1_original_generated_text": step1_row.get("original_generated_text"),
                    "baseline": baseline,
                    "trace": trace,
                    "heads": {},
                }
                skip_all = args.only_current_baseline_correct and baseline["prediction"] != gt

                for info in top_heads:
                    layer, head = int(info["layer"]), int(info["head"])
                    key = f"{layer}:{head}"
                    centroid_pred = head_centroids.get(key)
                    source = choose_source(args.source, centroid_pred, baseline["prediction"], gt)
                    hrow = {
                        "rank": int(info["rank"]),
                        "layer": layer,
                        "head": head,
                        "centroid_accuracy": float(info["centroid_accuracy"]),
                        "head_centroid_prediction": centroid_pred,
                        "head_centroid_correct": bool(centroid_pred == gt if centroid_pred else False),
                        "source_relation": source,
                        "target_relation": OPPOSITE.get(source),
                        "variants": {},
                    }
                    if skip_all:
                        hrow["skip_reason"] = "current_baseline_not_correct"
                        row["heads"][key] = hrow
                        continue
                    if source not in OPPOSITE:
                        hrow["skip_reason"] = "source_unavailable"
                        row["heads"][key] = hrow
                        continue
                    direction = relation_direction(relation_vectors, source)
                    vector = vectors[key]
                    for variant in variants:
                        delta, dinfo = build_delta(vector, direction, variant, args.strength)
                        patched, hook_info = patched_generate(
                            model=model,
                            processor=processor,
                            batch=batch,
                            attention_module=attention_modules[layer],
                            delta_cpu=delta,
                            prompt_length=prompt_length,
                            generation_kwargs=gen_kwargs,
                            label_ids=label_ids,
                        )
                        target = OPPOSITE[source]
                        base_margin = target_margin(baseline["relation_scores_first_step"], source)
                        new_margin = target_margin(patched["relation_scores_first_step"], source)
                        baseline_matches_source = baseline["prediction"] == source
                        clean = baseline_matches_source and source == gt
                        patched.update({
                            "correct": bool(patched["prediction"] == gt),
                            "source_relation": source,
                            "target_relation": target,
                            "baseline_matches_source": bool(baseline_matches_source),
                            "clean": bool(clean),
                            "prediction_changed": bool(
                                baseline["prediction"] is not None
                                and patched["prediction"] is not None
                                and patched["prediction"] != baseline["prediction"]
                            ),
                            "target_hit": bool(patched["prediction"] == target),
                            "exact_flip": bool(baseline_matches_source and patched["prediction"] == target),
                            "fixed": bool(baseline["prediction"] != gt and patched["prediction"] == gt),
                            "broken": bool(baseline["prediction"] == gt and patched["prediction"] != gt),
                            "baseline_target_minus_source_margin": base_margin,
                            "patched_target_minus_source_margin": new_margin,
                            "delta_target_minus_source_margin": (
                                new_margin - base_margin
                                if math.isfinite(base_margin) and math.isfinite(new_margin)
                                else float("nan")
                            ),
                            "delta": dinfo,
                            "hook": hook_info,
                        })
                        hrow["variants"][variant] = patched
                    row["heads"][key] = hrow

                append_jsonl(samples_path, row)
                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    tested = changed = exact = 0
                    for h in row["heads"].values():
                        for v in h.get("variants", {}).values():
                            tested += 1
                            changed += int(v["prediction_changed"])
                            exact += int(v["exact_flip"])
                    tqdm.write(
                        f"\nsid={sid} GT/base/Step1={gt}/{baseline['prediction']}/{step1_pred} "
                        f"tested={tested} changed={changed} exact_flip={exact}"
                    )
            except Exception as exc:
                collector.active = False
                append_jsonl(errors_path, {
                    "status": "error",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-24:],
                })
                tqdm.write(f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}")
            finally:
                collector.active = False
                if batch is not None:
                    del batch
                if image is not None:
                    del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        collector.close()

    rows = read_jsonl(samples_path)
    summary = build_summary(rows, top_heads, variants, config)
    summary["elapsed_minutes"] = (time.time() - started) / 60.0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary)
    print("\nSaved:")
    print(f"  {config_path}")
    print(f"  {top_path}")
    print(f"  {samples_path}")
    print(f"  {errors_path}")
    print(f"  {summary_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
