#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate whether a "grounded spatial evidence -> deployment gap" hypothesis is
worth pursuing, using outputs already produced by:

    analyze_coco_head_object_residual_direction_probe_v1.py

The existing extractor saves, for every sample / decoder layer / attention head:

    img      = (z_img,subject - z_img,reference)
    no_image = (z_noimg,subject - z_noimg,reference)
    residual = img - no_image

where z is the pre-W_O per-head output captured at the object text tokens.

This script DOES NOT claim that a failed direction readout means "no spatial
information".  It uses direction heads only as positive evidence.  The main
question is:

    Among baseline-generation errors, how often does a high-confidence
    multi-head residual consensus still predict the ground-truth relation?

If that population is large on a held-out test set, then there is empirical
motivation for a second-stage "re-deploy internal spatial evidence" repair.

Protocol
--------
1) Load relation_vectors.npz from the existing residual-direction experiment.
2) Make ONE stratified train/dev/test split (default 20/10/70).
3) Fit four relation prototypes per head on TRAIN residual vectors only.
4) Rank/select heads on DEV only.
5) Tune a conservative selective-consensus operating point on DEV only.
6) Run/restore baseline free generation for all samples.
7) Report on untouched TEST:
      - baseline generation accuracy
      - residual-consensus coverage and selective accuracy
      - deployment-gap count/rate:
            generation wrong + residual consensus covered + consensus == GT
      - harmful disagreement:
            generation correct + consensus covered + consensus != GT
      - img/no_image controls for the same selected heads.

Important interpretation
------------------------
* consensus absent/wrong is NOT evidence that the model contains no spatial info.
* consensus correct is only positive evidence under this readout family.
* this script is a feasibility test, not a causal proof and not a repair method.

Expected repository context
---------------------------
Run from the AdaptVis llava16 branch root so these modules are importable:
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py

Example
-------
First produce/reuse the existing direction vectors:

python analyze_coco_head_object_residual_direction_probe_v1.py \
  --dataset coco_two \
  --data-root data \
  --model qwen-3b \
  --device cuda:0 \
  --train-ratio 0.15 \
  --repeats 5 \
  --output-dir output/qwen3b_coco_head_direction_residual \
  --overwrite

Then run this feasibility test:

python validate_grounded_spatial_consensus_v1.py \
  --direction-dir output/qwen3b_coco_head_direction_residual \
  --dataset coco_two \
  --data-root data \
  --model qwen-3b \
  --device cuda:0 \
  --output-dir output/qwen3b_coco_grounded_consensus_v1 \
  --overwrite

For a cheap smoke test, add --max-generation-samples 40.  Do NOT use that smoke
run for the final feasibility conclusion because the split statistics become noisy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base


SCRIPT_VERSION = "validate-grounded-spatial-consensus-v1"
RELATIONS: Tuple[str, ...] = ("left", "right", "above", "below")
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12

REL_ALIASES = {
    "left": "left",
    "left of": "left",
    "right": "right",
    "right of": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "top": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "bottom": "below",
}

RELATION_PATTERNS: Dict[str, Sequence[str]] = {
    "left": (
        r"\bleft\s+of\b",
        r"\bto\s+the\s+left\b",
        r"\bon\s+the\s+left\b",
        r"\bleft\b",
    ),
    "right": (
        r"\bright\s+of\b",
        r"\bto\s+the\s+right\b",
        r"\bon\s+the\s+right\b",
        r"\bright\b",
    ),
    "above": (
        r"\bon\s+top\s+of\b",
        r"\batop\b",
        r"\babove\b",
        r"\bover\b",
        r"\bon\b",
    ),
    "below": (
        r"\bunderneath\b",
        r"\bbelow\b",
        r"\bunder\b",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--direction-dir",
        required=True,
        help="Directory containing relation_vectors.npz from the existing residual-direction script.",
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b", choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument("--train-frac", type=float, default=0.20)
    p.add_argument("--dev-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--top-k-grid",
        default="5,10,20,30,50",
        help="Numbers of DEV-ranked residual heads considered by the consensus.",
    )
    p.add_argument(
        "--head-margin-quantile-grid",
        default="0.00,0.25,0.50,0.75",
        help=(
            "Per-head confidence gate. For q, a head votes on a sample only when its "
            "top1-top2 margin exceeds that head's q-quantile DEV margin."
        ),
    )
    p.add_argument("--min-active-grid", default="2,3,5")
    p.add_argument(
        "--support-grid",
        default="0.50,0.60,0.70,0.80",
        help="Minimum winning vote-mass fraction for declaring consensus.",
    )
    p.add_argument(
        "--target-dev-precision",
        type=float,
        default=0.75,
        help=(
            "DEV selective accuracy target. Among settings that reach it, choose max coverage. "
            "If none reaches it, use a conservative accuracy/coverage fallback score."
        ),
    )

    p.add_argument(
        "--generation-jsonl",
        default=None,
        help=(
            "Optional baseline generation cache. If omitted, use <output-dir>/generation.jsonl "
            "and generate missing SIDs."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument(
        "--max-generation-samples",
        type=int,
        default=None,
        help="Smoke-test cap only. If set, analysis is restricted to these generated SIDs.",
    )
    p.add_argument("--print-every", type=int, default=20)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return REL_ALIASES.get(text)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_generated_relation(text: str) -> Optional[str]:
    normalized = normalize_space(text).lower()
    if not normalized:
        return None
    candidates: List[Tuple[int, int, str]] = []
    for relation, patterns in RELATION_PATTERNS.items():
        for priority, pattern in enumerate(patterns):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match is not None:
                candidates.append((int(match.start()), int(priority), relation))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def row_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def balanced_accuracy_per_head(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred [N,H], gt [N] -> [H]."""
    if pred.ndim != 2:
        raise ValueError(pred.shape)
    scores = []
    for relation_idx in range(len(RELATIONS)):
        mask = gt == relation_idx
        if not mask.any():
            continue
        scores.append(np.mean(pred[mask] == gt[mask, None], axis=0))
    if not scores:
        return np.zeros(pred.shape[1], dtype=np.float64)
    return np.mean(np.stack(scores, axis=0), axis=0)


def stratified_split(
    labels: np.ndarray,
    train_frac: float,
    dev_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_frac < 1.0 and 0.0 < dev_frac < 1.0):
        raise ValueError("train/dev fractions must lie in (0,1)")
    if train_frac + dev_frac >= 1.0:
        raise ValueError("train_frac + dev_frac must be < 1")
    rng = np.random.default_rng(seed)
    train: List[int] = []
    dev: List[int] = []
    test: List[int] = []
    labels = np.asarray(labels)
    for relation in RELATIONS:
        ids = np.flatnonzero(labels == relation)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * train_frac)))
        n_dev = max(1, int(round(n * dev_frac)))
        if n_train + n_dev >= n:
            raise RuntimeError(f"Too few samples for relation={relation}: n={n}")
        train.extend(ids[:n_train].tolist())
        dev.extend(ids[n_train:n_train + n_dev].tolist())
        test.extend(ids[n_train + n_dev:].tolist())
    for group in (train, dev, test):
        rng.shuffle(group)
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(dev, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def flatten_heads(x: np.ndarray) -> np.ndarray:
    """[N,L,H,D] -> [N,L*H,D]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [N,L,H,D], got {x.shape}")
    n, l, h, d = x.shape
    return np.asarray(x, dtype=np.float32).reshape(n, l * h, d)


def fit_prototypes(
    x: np.ndarray,  # [N,H,D]
    labels: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return center [H,D] and unit prototypes [R,H,D]."""
    train = np.asarray(x[train_idx], dtype=np.float64)
    center = train.mean(axis=0)
    centered = train - center[None, :, :]
    protos = []
    train_labels = labels[train_idx]
    for relation in RELATIONS:
        mask = train_labels == relation
        if not mask.any():
            raise RuntimeError(f"No train examples for relation={relation}")
        proto = centered[mask].mean(axis=0)
        protos.append(row_unit(proto))
    return center.astype(np.float32), np.stack(protos, axis=0).astype(np.float32)


def score_prototypes(
    x: np.ndarray,  # [N,H,D]
    idx: np.ndarray,
    center: np.ndarray,  # [H,D]
    prototypes: np.ndarray,  # [R,H,D]
) -> np.ndarray:
    current = row_unit(np.asarray(x[idx], dtype=np.float64) - center[None, :, :])
    # [N,H,D] x [R,H,D] -> [N,H,R]
    return np.einsum("nhd,rhd->nhr", current, prototypes, optimize=True).astype(np.float32)


def top_prediction_and_margin(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.argmax(scores, axis=-1)
    sorted_scores = np.sort(scores, axis=-1)
    margin = sorted_scores[..., -1] - sorted_scores[..., -2]
    return pred.astype(np.int64), margin.astype(np.float32)


def consensus_from_scores(
    scores: np.ndarray,  # [N,Hsel,R]
    selected_global_heads: np.ndarray,
    reliability_global: np.ndarray,
    per_head_margin_threshold_global: np.ndarray,
    min_active: int,
    min_support: float,
) -> Dict[str, np.ndarray]:
    if scores.ndim != 3:
        raise ValueError(scores.shape)
    pred, margin = top_prediction_and_margin(scores)
    reliability = reliability_global[selected_global_heads][None, :]
    margin_threshold = per_head_margin_threshold_global[selected_global_heads][None, :]
    active = margin >= margin_threshold

    # Positive evidence only.  A head that is uncertain simply abstains.
    vote_strength = active.astype(np.float64) * reliability * np.maximum(margin, 0.0)
    n_samples, n_heads = pred.shape
    relation_vote = np.zeros((n_samples, len(RELATIONS)), dtype=np.float64)
    for relation_idx in range(len(RELATIONS)):
        relation_vote[:, relation_idx] = np.sum(
            vote_strength * (pred == relation_idx), axis=1
        )

    total = relation_vote.sum(axis=1)
    top = np.argmax(relation_vote, axis=1)
    sorted_vote = np.sort(relation_vote, axis=1)
    top_mass = sorted_vote[:, -1]
    second_mass = sorted_vote[:, -2]
    support = np.divide(top_mass, np.maximum(total, EPS))
    normalized_gap = np.divide(top_mass - second_mass, np.maximum(total, EPS))
    active_count = active.sum(axis=1)
    covered = (
        (active_count >= int(min_active))
        & (total > EPS)
        & (support >= float(min_support))
    )
    return {
        "prediction": top.astype(np.int64),
        "covered": covered.astype(bool),
        "support": support.astype(np.float32),
        "gap": normalized_gap.astype(np.float32),
        "active_count": active_count.astype(np.int64),
        "total_vote": total.astype(np.float32),
    }


def summarize_consensus(result: Mapping[str, np.ndarray], gt: np.ndarray) -> Dict[str, float]:
    covered = np.asarray(result["covered"], dtype=bool)
    pred = np.asarray(result["prediction"], dtype=np.int64)
    if len(gt) == 0:
        return {"n": 0, "coverage": float("nan"), "accuracy": float("nan")}
    accuracy = float(np.mean(pred[covered] == gt[covered])) if covered.any() else float("nan")
    return {
        "n": int(len(gt)),
        "covered_n": int(covered.sum()),
        "coverage": float(covered.mean()),
        "accuracy": accuracy,
    }


def choose_operating_point(
    residual_dev_scores_all: np.ndarray,  # [Ndev,Hall,R]
    gt_dev: np.ndarray,
    dev_bal_acc: np.ndarray,
    k_grid: Sequence[int],
    q_grid: Sequence[float],
    min_active_grid: Sequence[int],
    support_grid: Sequence[float],
    target_precision: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    n_heads = residual_dev_scores_all.shape[1]
    order = np.argsort(-dev_bal_acc)
    reliability = np.maximum(dev_bal_acc - 0.25, 1e-4)

    # Per-head DEV margins; q-specific thresholds are built below.
    _, dev_margin_all = top_prediction_and_margin(residual_dev_scores_all)

    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[Dict[str, Any], np.ndarray]] = []
    for k_raw in k_grid:
        k = min(int(k_raw), n_heads)
        if k <= 0:
            continue
        selected = order[:k]
        scores = residual_dev_scores_all[:, selected, :]
        for q in q_grid:
            thresholds = np.zeros(n_heads, dtype=np.float32)
            thresholds[selected] = np.quantile(
                dev_margin_all[:, selected],
                float(q),
                axis=0,
            ).astype(np.float32)
            for min_active in min_active_grid:
                if int(min_active) > k:
                    continue
                for min_support in support_grid:
                    result = consensus_from_scores(
                        scores=scores,
                        selected_global_heads=selected,
                        reliability_global=reliability,
                        per_head_margin_threshold_global=thresholds,
                        min_active=int(min_active),
                        min_support=float(min_support),
                    )
                    summary = summarize_consensus(result, gt_dev)
                    acc = float(summary["accuracy"])
                    cov = float(summary["coverage"])
                    row = {
                        "top_k": int(k),
                        "head_margin_quantile": float(q),
                        "min_active": int(min_active),
                        "min_support": float(min_support),
                        **summary,
                    }
                    rows.append(row)
                    candidates.append((row, thresholds.copy()))

    if not candidates:
        raise RuntimeError("No consensus operating points were evaluated")

    eligible = [
        pair for pair in candidates
        if math.isfinite(float(pair[0]["accuracy"]))
        and float(pair[0]["accuracy"]) >= float(target_precision)
        and int(pair[0]["covered_n"]) > 0
    ]
    if eligible:
        # Conservative positive evidence: reach target precision, then maximize coverage.
        best_row, best_thresholds = max(
            eligible,
            key=lambda pair: (
                float(pair[0]["coverage"]),
                float(pair[0]["accuracy"]),
                -int(pair[0]["top_k"]),
            ),
        )
        selection_reason = "max_coverage_subject_to_target_dev_precision"
    else:
        # Fallback discourages a tiny near-perfect subset but still rewards precision.
        finite = [
            pair for pair in candidates
            if math.isfinite(float(pair[0]["accuracy"])) and int(pair[0]["covered_n"]) > 0
        ]
        if not finite:
            raise RuntimeError("Every DEV operating point had zero coverage")
        best_row, best_thresholds = max(
            finite,
            key=lambda pair: (
                float(pair[0]["accuracy"]) * math.sqrt(max(float(pair[0]["coverage"]), 0.0)),
                float(pair[0]["accuracy"]),
            ),
        )
        selection_reason = "fallback_accuracy_times_sqrt_coverage"

    best = dict(best_row)
    best["selection_reason"] = selection_reason
    best["target_dev_precision"] = float(target_precision)
    return best, rows, best_thresholds


def decode_new_tokens(
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_length: int,
) -> Tuple[str, List[int]]:
    if sequences.ndim != 2 or int(sequences.shape[0]) != 1:
        raise RuntimeError(f"Expected [1,T] generation, got {tuple(sequences.shape)}")
    seq = sequences[0]
    if int(seq.shape[0]) >= int(prompt_length):
        new = seq[int(prompt_length):]
    else:
        new = seq
    ids = [int(x) for x in new.detach().cpu().tolist()]
    text = tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return normalize_space(text), ids


def load_generation_cache(path: Path) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if "sid" not in row:
            continue
        result[int(row["sid"])] = row
    return result


def generate_missing(
    *,
    cache_path: Path,
    target_sids: Sequence[int],
    dataset: str,
    data_root: Path,
    model_alias: str,
    device_name: str,
    attn_impl: str,
    prompt_template: str,
    max_new_tokens: int,
    print_every: int,
) -> Dict[int, Dict[str, Any]]:
    cache = load_generation_cache(cache_path)
    needed = [int(sid) for sid in target_sids if int(sid) not in cache]
    if not needed:
        print(f"[generation] cache complete: {cache_path}")
        return cache

    records, _audit = base.load_records(dataset, data_root, None)
    by_sid = {int(r.sid): r for r in records}
    missing_records = [sid for sid in needed if sid not in by_sid]
    if missing_records:
        raise RuntimeError(f"Generation SIDs absent from dataset loader: {missing_records[:10]}")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    spec = base.SPECS[model_alias]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )
    kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": device_name},
    }
    if attn_impl != "none":
        kwargs["attn_implementation"] = attn_impl

    print(f"[generation] loading {model_alias} from {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(device_name)
    tokenizer = processor.tokenizer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    generation_kwargs: Dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)

    started = time.time()
    errors = 0
    try:
        for pos, sid in enumerate(tqdm(needed, desc="baseline-generation"), 1):
            rec = by_sid[int(sid)]
            image = None
            try:
                q = prompt_template.format(subject=rec.subject, reference=rec.reference)
                image = Image.open(rec.image_path).convert("RGB")
                rendered = direction_base.build_chat_prompt(processor, q, True)
                batch = direction_base.process_inputs(processor, rendered, image, device)
                prompt_length = int(batch["input_ids"].shape[1])
                with torch.inference_mode():
                    sequences = model.generate(**batch, **generation_kwargs)
                text, token_ids = decode_new_tokens(tokenizer, sequences, prompt_length)
                prediction = parse_generated_relation(text)
                gt = normalize_relation(rec.relation)
                row = {
                    "sid": int(sid),
                    "gt": gt,
                    "prediction": prediction,
                    "parsed": bool(prediction is not None),
                    "correct": bool(prediction == gt),
                    "text": text,
                    "token_ids": token_ids,
                    "subject": str(rec.subject),
                    "reference": str(rec.reference),
                }
            except Exception as exc:
                errors += 1
                row = {
                    "sid": int(sid),
                    "gt": normalize_relation(rec.relation),
                    "prediction": None,
                    "parsed": False,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                }
                tqdm.write(f"[generation ERROR] sid={sid}: {row['error']}")
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            append_jsonl(cache_path, row)
            cache[int(sid)] = row
            if print_every > 0 and pos % int(print_every) == 0:
                current = [cache[int(x)] for x in needed[:pos] if int(x) in cache]
                acc = safe_mean(float(bool(x.get("correct", False))) for x in current)
                parse_rate = safe_mean(float(bool(x.get("parsed", False))) for x in current)
                tqdm.write(
                    f"[generation] {pos}/{len(needed)} parse={parse_rate:.4f} acc={acc:.4f} errors={errors}"
                )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[generation] done in {(time.time()-started)/60.0:.1f} min, errors={errors}")
    return cache


def state_control_summary(
    *,
    x_state: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    selected_heads: np.ndarray,
    reliability: np.ndarray,
    chosen: Mapping[str, Any],
) -> Dict[str, Any]:
    center, proto = fit_prototypes(x_state, labels, train_idx)
    dev_scores_all = score_prototypes(x_state, dev_idx, center, proto)
    test_scores_all = score_prototypes(x_state, test_idx, center, proto)

    # Same q, K, min-active, support; state-specific DEV margin thresholds so
    # scale differences do not unfairly punish controls.
    q = float(chosen["head_margin_quantile"])
    _, dev_margins = top_prediction_and_margin(dev_scores_all)
    thresholds = np.zeros(x_state.shape[1], dtype=np.float32)
    thresholds[selected_heads] = np.quantile(
        dev_margins[:, selected_heads], q, axis=0
    ).astype(np.float32)

    test_result = consensus_from_scores(
        scores=test_scores_all[:, selected_heads, :],
        selected_global_heads=selected_heads,
        reliability_global=reliability,
        per_head_margin_threshold_global=thresholds,
        min_active=int(chosen["min_active"]),
        min_support=float(chosen["min_support"]),
    )
    gt_test = np.asarray([REL_TO_IDX[str(x)] for x in labels[test_idx]], dtype=np.int64)
    return {
        **summarize_consensus(test_result, gt_test),
        "prediction": test_result["prediction"],
        "covered": test_result["covered"],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    direction_dir = Path(args.direction_dir)
    npz_path = direction_dir / "relation_vectors.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Run analyze_coco_head_object_residual_direction_probe_v1.py first."
        )

    with np.load(npz_path, allow_pickle=True) as data:
        required = {"sample_index", "relation", "residual", "img", "no_image"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(
                f"{npz_path} is missing {missing}. Re-run the direction extractor WITHOUT --no-controls."
            )
        sids = np.asarray(data["sample_index"], dtype=np.int64)
        labels = np.asarray([normalize_relation(x) for x in data["relation"]], dtype=object)
        residual_4d = np.asarray(data["residual"], dtype=np.float32)
        img_4d = np.asarray(data["img"], dtype=np.float32)
        noimg_4d = np.asarray(data["no_image"], dtype=np.float32)
        layer_ids = np.asarray(
            data["decoder_block_index"] if "decoder_block_index" in data.files else np.arange(residual_4d.shape[1]),
            dtype=np.int64,
        )
        head_ids = np.asarray(
            data["head_index"] if "head_index" in data.files else np.arange(residual_4d.shape[2]),
            dtype=np.int64,
        )

    valid = np.asarray([x in REL_TO_IDX for x in labels], dtype=bool)
    sids = sids[valid]
    labels = labels[valid]
    residual_4d = residual_4d[valid]
    img_4d = img_4d[valid]
    noimg_4d = noimg_4d[valid]

    if len(set(int(x) for x in sids.tolist())) != len(sids):
        raise RuntimeError("Duplicate SIDs in relation_vectors.npz")

    n, n_layers, n_heads, head_dim = residual_4d.shape
    print(
        f"[vectors] N={n} layers={n_layers} heads/layer={n_heads} head_dim={head_dim} "
        f"relations={dict(Counter(str(x) for x in labels))}"
    )

    # If requested, restrict the whole analysis to a deterministic smoke subset.
    if args.max_generation_samples is not None and int(args.max_generation_samples) < n:
        rng = np.random.default_rng(args.seed)
        keep: List[int] = []
        cap = int(args.max_generation_samples)
        # proportional-ish stratified smoke subset
        per_rel = max(1, cap // len(RELATIONS))
        for rel in RELATIONS:
            ids = np.flatnonzero(labels == rel)
            rng.shuffle(ids)
            keep.extend(ids[:per_rel].tolist())
        if len(keep) < cap:
            remaining = [i for i in range(n) if i not in set(keep)]
            rng.shuffle(remaining)
            keep.extend(remaining[: cap - len(keep)])
        keep = sorted(keep[:cap])
        idx_keep = np.asarray(keep, dtype=np.int64)
        sids = sids[idx_keep]
        labels = labels[idx_keep]
        residual_4d = residual_4d[idx_keep]
        img_4d = img_4d[idx_keep]
        noimg_4d = noimg_4d[idx_keep]
        n = len(sids)
        print(f"[SMOKE MODE] restricted to N={n}; do not use this as final evidence")

    residual = flatten_heads(residual_4d)
    img = flatten_heads(img_4d)
    noimg = flatten_heads(noimg_4d)
    total_heads = residual.shape[1]
    head_names = [
        f"L{int(layer_ids[l])}H{int(head_ids[h]):02d}"
        for l in range(n_layers)
        for h in range(n_heads)
    ]

    train_idx, dev_idx, test_idx = stratified_split(
        labels,
        args.train_frac,
        args.dev_frac,
        args.seed,
    )
    print(
        f"[split] train={len(train_idx)} dev={len(dev_idx)} test={len(test_idx)} "
        f"({args.train_frac:.2f}/{args.dev_frac:.2f}/{1-args.train_frac-args.dev_frac:.2f})"
    )

    split_rows = []
    split_name = {}
    for name, ids in (("train", train_idx), ("dev", dev_idx), ("test", test_idx)):
        for i in ids:
            split_name[int(i)] = name
            split_rows.append({
                "row_index": int(i),
                "sid": int(sids[i]),
                "relation": str(labels[i]),
                "split": name,
            })
    write_csv(out / "split.csv", split_rows)

    # ------------------------------------------------------------------
    # Residual prototypes and DEV head ranking
    # ------------------------------------------------------------------
    res_center, res_proto = fit_prototypes(residual, labels, train_idx)
    res_dev_scores = score_prototypes(residual, dev_idx, res_center, res_proto)
    dev_gt = np.asarray([REL_TO_IDX[str(x)] for x in labels[dev_idx]], dtype=np.int64)
    res_dev_pred, _res_dev_margin = top_prediction_and_margin(res_dev_scores)
    dev_bal_acc = balanced_accuracy_per_head(res_dev_pred, dev_gt)
    reliability = np.maximum(dev_bal_acc - 0.25, 1e-4)

    # Controls: same TRAIN split, separate prototype banks.
    img_center, img_proto = fit_prototypes(img, labels, train_idx)
    no_center, no_proto = fit_prototypes(noimg, labels, train_idx)
    img_dev_scores = score_prototypes(img, dev_idx, img_center, img_proto)
    no_dev_scores = score_prototypes(noimg, dev_idx, no_center, no_proto)
    img_dev_pred, _ = top_prediction_and_margin(img_dev_scores)
    no_dev_pred, _ = top_prediction_and_margin(no_dev_scores)
    img_dev_bal = balanced_accuracy_per_head(img_dev_pred, dev_gt)
    no_dev_bal = balanced_accuracy_per_head(no_dev_pred, dev_gt)

    rank = np.argsort(-dev_bal_acc)
    head_rows = []
    for r, hidx in enumerate(rank, 1):
        layer_pos = int(hidx // n_heads)
        head_pos = int(hidx % n_heads)
        head_rows.append({
            "rank": int(r),
            "flat_head_index": int(hidx),
            "head_name": head_names[int(hidx)],
            "layer": int(layer_ids[layer_pos]),
            "head": int(head_ids[head_pos]),
            "residual_dev_bal_acc": float(dev_bal_acc[hidx]),
            "img_dev_bal_acc": float(img_dev_bal[hidx]),
            "noimage_dev_bal_acc": float(no_dev_bal[hidx]),
            "residual_minus_img": float(dev_bal_acc[hidx] - img_dev_bal[hidx]),
            "residual_minus_noimage": float(dev_bal_acc[hidx] - no_dev_bal[hidx]),
            "reliability_weight": float(reliability[hidx]),
        })
    write_csv(out / "head_ranking_dev.csv", head_rows)

    chosen, grid_rows, residual_margin_thresholds = choose_operating_point(
        residual_dev_scores_all=res_dev_scores,
        gt_dev=dev_gt,
        dev_bal_acc=dev_bal_acc,
        k_grid=parse_csv_ints(args.top_k_grid),
        q_grid=parse_csv_floats(args.head_margin_quantile_grid),
        min_active_grid=parse_csv_ints(args.min_active_grid),
        support_grid=parse_csv_floats(args.support_grid),
        target_precision=args.target_dev_precision,
    )
    write_csv(out / "dev_operating_grid.csv", grid_rows)

    selected_heads = rank[: int(chosen["top_k"])]
    print("[DEV chosen operating point]")
    print(json.dumps(chosen, indent=2))
    print("[DEV selected heads]")
    print(", ".join(head_names[int(i)] for i in selected_heads))

    # ------------------------------------------------------------------
    # Baseline free generation.  Cache is append-only and reusable.
    # ------------------------------------------------------------------
    generation_path = (
        Path(args.generation_jsonl)
        if args.generation_jsonl is not None
        else out / "generation.jsonl"
    )
    generation = generate_missing(
        cache_path=generation_path,
        target_sids=sids.tolist(),
        dataset=args.dataset,
        data_root=Path(args.data_root),
        model_alias=args.model,
        device_name=args.device,
        attn_impl=args.attn_impl,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        print_every=args.print_every,
    )

    # Verify that generation GT agrees with vector metadata wherever present.
    for sid, gt in zip(sids.tolist(), labels.tolist()):
        row = generation.get(int(sid))
        if row is None:
            raise RuntimeError(f"Missing generation for sid={sid}")
        cached_gt = normalize_relation(row.get("gt"))
        if cached_gt is not None and cached_gt != str(gt):
            raise RuntimeError(f"GT mismatch sid={sid}: vectors={gt} generation={cached_gt}")

    # ------------------------------------------------------------------
    # Untouched TEST evaluation
    # ------------------------------------------------------------------
    res_test_scores_all = score_prototypes(residual, test_idx, res_center, res_proto)
    test_gt = np.asarray([REL_TO_IDX[str(x)] for x in labels[test_idx]], dtype=np.int64)
    res_test_result = consensus_from_scores(
        scores=res_test_scores_all[:, selected_heads, :],
        selected_global_heads=selected_heads,
        reliability_global=reliability,
        per_head_margin_threshold_global=residual_margin_thresholds,
        min_active=int(chosen["min_active"]),
        min_support=float(chosen["min_support"]),
    )
    res_test_summary = summarize_consensus(res_test_result, test_gt)

    # Controls using identical selected head identities and same operating-point
    # hyperparameters, but state-specific DEV margin scale.
    img_control = state_control_summary(
        x_state=img,
        labels=labels,
        train_idx=train_idx,
        dev_idx=dev_idx,
        test_idx=test_idx,
        selected_heads=selected_heads,
        reliability=reliability,
        chosen=chosen,
    )
    noimg_control = state_control_summary(
        x_state=noimg,
        labels=labels,
        train_idx=train_idx,
        dev_idx=dev_idx,
        test_idx=test_idx,
        selected_heads=selected_heads,
        reliability=reliability,
        chosen=chosen,
    )

    gen_pred_idx = np.full(len(test_idx), -1, dtype=np.int64)
    gen_parsed = np.zeros(len(test_idx), dtype=bool)
    gen_correct = np.zeros(len(test_idx), dtype=bool)
    for local_i, global_i in enumerate(test_idx.tolist()):
        sid = int(sids[global_i])
        row = generation[sid]
        pred_rel = normalize_relation(row.get("prediction"))
        if pred_rel in REL_TO_IDX:
            gen_pred_idx[local_i] = REL_TO_IDX[pred_rel]
            gen_parsed[local_i] = True
        gen_correct[local_i] = bool(pred_rel == str(labels[global_i]))

    covered = np.asarray(res_test_result["covered"], dtype=bool)
    internal_pred = np.asarray(res_test_result["prediction"], dtype=np.int64)
    internal_correct = internal_pred == test_gt
    gen_wrong = ~gen_correct

    deployment_gap = gen_wrong & covered & internal_correct
    harmful_disagreement = gen_correct & covered & (~internal_correct)
    covered_wrong = gen_wrong & covered
    covered_correct = gen_correct & covered

    # A useful stricter mismatch: internal consensus is correct and explicitly
    # disagrees with a parsed generation answer.
    parsed_conflict_repairable = (
        deployment_gap
        & gen_parsed
        & (gen_pred_idx != internal_pred)
    )

    samples_rows: List[Dict[str, Any]] = []
    for local_i, global_i in enumerate(test_idx.tolist()):
        sid = int(sids[global_i])
        grow = generation[sid]
        samples_rows.append({
            "sid": sid,
            "gt": str(labels[global_i]),
            "generation_prediction": normalize_relation(grow.get("prediction")),
            "generation_parsed": int(gen_parsed[local_i]),
            "generation_correct": int(gen_correct[local_i]),
            "residual_consensus_prediction": RELATIONS[int(internal_pred[local_i])],
            "residual_consensus_covered": int(covered[local_i]),
            "residual_consensus_correct": int(internal_correct[local_i]),
            "residual_consensus_support": float(res_test_result["support"][local_i]),
            "residual_consensus_gap": float(res_test_result["gap"][local_i]),
            "residual_active_heads": int(res_test_result["active_count"][local_i]),
            "deployment_gap_candidate": int(deployment_gap[local_i]),
            "parsed_conflict_repairable": int(parsed_conflict_repairable[local_i]),
            "harmful_internal_disagreement": int(harmful_disagreement[local_i]),
            "img_control_prediction": RELATIONS[int(img_control["prediction"][local_i])],
            "img_control_covered": int(img_control["covered"][local_i]),
            "noimage_control_prediction": RELATIONS[int(noimg_control["prediction"][local_i])],
            "noimage_control_covered": int(noimg_control["covered"][local_i]),
            "generation_text": grow.get("text", ""),
        })
    write_csv(out / "test_samples.csv", samples_rows)

    n_test = len(test_idx)
    n_wrong = int(gen_wrong.sum())
    n_correct = int(gen_correct.sum())
    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "model": args.model,
        "direction_dir": str(direction_dir),
        "generation_jsonl": str(generation_path),
        "n_total": int(n),
        "n_train": int(len(train_idx)),
        "n_dev": int(len(dev_idx)),
        "n_test": int(n_test),
        "seed": int(args.seed),
        "selected_operating_point": chosen,
        "selected_heads": [head_names[int(i)] for i in selected_heads],
        "generation": {
            "parse_rate": float(gen_parsed.mean()) if n_test else float("nan"),
            "accuracy": float(gen_correct.mean()) if n_test else float("nan"),
            "correct_n": n_correct,
            "wrong_n": n_wrong,
        },
        "residual_consensus": {
            k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
            for k, v in res_test_summary.items()
        },
        "img_control_consensus": {
            "n": int(img_control["n"]),
            "covered_n": int(img_control["covered_n"]),
            "coverage": float(img_control["coverage"]),
            "accuracy": float(img_control["accuracy"]),
        },
        "noimage_control_consensus": {
            "n": int(noimg_control["n"]),
            "covered_n": int(noimg_control["covered_n"]),
            "coverage": float(noimg_control["coverage"]),
            "accuracy": float(noimg_control["accuracy"]),
        },
        "deployment_gap": {
            "count": int(deployment_gap.sum()),
            "fraction_of_all_test": float(deployment_gap.mean()) if n_test else float("nan"),
            "fraction_of_generation_errors": (
                float(deployment_gap.sum() / n_wrong) if n_wrong else float("nan")
            ),
            "precision_within_covered_generation_errors": (
                float(deployment_gap.sum() / covered_wrong.sum())
                if covered_wrong.any() else float("nan")
            ),
            "covered_generation_errors_n": int(covered_wrong.sum()),
            "parsed_conflict_repairable_n": int(parsed_conflict_repairable.sum()),
        },
        "harmful_disagreement": {
            "count": int(harmful_disagreement.sum()),
            "fraction_of_generation_correct": (
                float(harmful_disagreement.sum() / n_correct) if n_correct else float("nan")
            ),
            "precision_within_covered_generation_correct": (
                float((covered_correct & internal_correct).sum() / covered_correct.sum())
                if covered_correct.any() else float("nan")
            ),
        },
        "interpretation": {
            "positive_result": (
                "A non-trivial deployment_gap.fraction_of_generation_errors together with high "
                "residual_consensus.accuracy supports moving to an oracle/GT-free re-deployment repair test."
            ),
            "negative_result": (
                "Low deployment-gap prevalence means this specific multi-head direction readout does not "
                "explain enough generation errors; it does NOT prove that spatial information is absent."
            ),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("GROUNDED SPATIAL CONSENSUS FEASIBILITY")
    print("=" * 100)
    print(f"TEST N                                      : {n_test}")
    print(f"baseline generation parse rate             : {summary['generation']['parse_rate']:.4f}")
    print(f"baseline generation accuracy               : {summary['generation']['accuracy']:.4f}")
    print(f"residual consensus coverage                : {res_test_summary['coverage']:.4f}")
    print(f"residual consensus acc | covered           : {res_test_summary['accuracy']:.4f}")
    print(f"img consensus acc | covered                : {img_control['accuracy']:.4f}")
    print(f"no-image consensus acc | covered           : {noimg_control['accuracy']:.4f}")
    print("-")
    print(f"generation wrong                           : {n_wrong}")
    print(f"covered generation wrong                   : {int(covered_wrong.sum())}")
    print(f"deployment-gap candidates                  : {int(deployment_gap.sum())}")
    print(
        "deployment-gap / all generation errors        : "
        f"{summary['deployment_gap']['fraction_of_generation_errors']:.4f}"
    )
    print(
        "internal correct | covered generation errors  : "
        f"{summary['deployment_gap']['precision_within_covered_generation_errors']:.4f}"
    )
    print(f"parsed explicit conflict candidates         : {int(parsed_conflict_repairable.sum())}")
    print("-")
    print(f"generation correct                         : {n_correct}")
    print(f"harmful internal disagreements             : {int(harmful_disagreement.sum())}")
    print(
        "harmful disagreement / generation correct     : "
        f"{summary['harmful_disagreement']['fraction_of_generation_correct']:.4f}"
    )
    print("\nSelected heads:")
    print(", ".join(summary["selected_heads"]))
    print("\nSaved:")
    for filename in (
        "summary.json",
        "test_samples.csv",
        "head_ranking_dev.csv",
        "dev_operating_grid.csv",
        "split.csv",
    ):
        print(f"  {out / filename}")
    print(f"  {generation_path}")


if __name__ == "__main__":
    main()
