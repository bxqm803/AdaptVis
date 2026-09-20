#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_qwen_causal_state_graypatch_validation_v1.py

Lightweight causal validation for the current writer-guided causal-state ranking.

Question
========
The current candidate states are ranked by

    M(L,p) = (h_real[L,p] - h_gray[L,p])^T grad_h J_writer.

Because this ranking uses a gradient, do the selected states actually matter for
THE FINAL ANSWER, rather than merely having a large first-order score?

This script performs an activation-removal test that does NOT push along the
ranking gradient.

For selected source states (L,p), during a fresh REAL forward pass we replace

    h_real[L,p]  -->  h_gray[L,p]

using the clean gray-image activation captured for the same prompt.  We then
measure four-way teacher-forced answer sequence scores and the GT margin

    margin = S_GT - max_{r != GT} S_r.

If the selected states carry unusually strong effective decision influence,
removing their image-dependent state should decrease this final answer margin
more than matched controls.

Controls
========
1) top
   Top-K states from the existing ranked_k36_tokens.csv.

2) norm_matched
   For every Top-K state, choose a DIFFERENT prompt token at the SAME layer and
   same broad token category, excluding all K36-ranked token positions, whose
   ||h_real-h_gray|| is closest to the Top-K state's norm.

3) random_matched
   Same layer + same broad token category, excluding all K36 positions, sampled
   deterministically.  Multiple repeats are supported.

4) tail
   The last K states inside the same positive K36 ranking.  This is a weaker
   control than norm_matched/random because layer/category distributions need
   not match, but it tests whether the ranking itself separates strong from
   lower-ranked positive states.

Important
=========
- This validates the EXISTING M-ranked states.  It is not a pure grad-norm test.
- The ranking remains oracle because the original writer objective used the GT
  relation.  GT is also used here only to report the answer margin.
- The primary outcome is FINAL answer sequence-score margin, not the late writer
  projection used to construct the ranking.
- No intervention is taken in the gradient direction.

Expected prior run
==================
By default this reads the L31-35 dense-K run already produced by:

  output/qwen3b_sparse_bottleneck_prefix_L31_35_denseK_all440_v1/
      ranked_k36_tokens.csv
      metadata.json

Recommended smoke test
======================
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_causal_state_graypatch_validation_v1.py \
  --ranked-dir output/qwen3b_sparse_bottleneck_prefix_L31_35_denseK_all440_v1 \
  --top-k 7 \
  --eval-max-samples 80 \
  --random-repeats 3 \
  --output-dir output/qwen3b_causal_state_graypatch_top7_n80_v1 \
  --overwrite

Full 440
========
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_causal_state_graypatch_validation_v1.py \
  --ranked-dir output/qwen3b_sparse_bottleneck_prefix_L31_35_denseK_all440_v1 \
  --top-k 7 \
  --eval-max-samples 0 \
  --random-repeats 3 \
  --output-dir output/qwen3b_causal_state_graypatch_top7_all440_v1 \
  --overwrite

Optional actual generation (more expensive)
===========================================
Add:

  --run-generation

Primary outputs
===============
per_sample_scores.csv
    Clean and patched four-way sequence scores, GT margin, and margin drop.

summary.csv
    Mean/median final-margin drop and sequence-score accuracy for each condition.

paired_top_vs_controls.csv
    Per-sample paired effect: Top-K margin drop vs matched-control margin drop.

selected_states.csv
    Exact Top / norm-matched / random / tail states used.

figure_margin_drop.png / .pdf
    Small diagnostic bar plot.  This is optional evidence, not necessarily a
    paper figure.

metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoProcessor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn
import eval_real_causal_token_update_gating_v1 as gate


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# CLI / basic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--ranked-dir", default=(
        "output/qwen3b_sparse_bottleneck_prefix_L31_35_denseK_all440_v1"
    ))
    p.add_argument("--ranked-csv", default="",
                   help="Optional explicit ranked_k36_tokens.csv; overrides --ranked-dir.")

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-layers", default="",
                   help="Empty = infer from ranked CSV / metadata.")

    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--random-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Optional cap BEFORE selecting eval SIDs; 0=all dataset rows.")
    p.add_argument("--eval-max-samples", type=int, default=80,
                   help="0 = all SIDs present in ranking CSV.")

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    # Same final-answer score used elsewhere in the repo.
    p.add_argument("--answer-surface", default="above_below",
                   choices=["above_below", "on_under"])
    p.add_argument("--answer-prefix", default=" ")
    p.add_argument("--answer-suffix", default="")
    p.add_argument("--sequence-score-reduction", default="mean",
                   choices=["sum", "mean"])

    p.add_argument("--run-generation", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=6)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.median(vals)) if vals else float("nan")


def safe_sem(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if x is not None and math.isfinite(float(x))],
        dtype=np.float64,
    )
    if len(vals) <= 1:
        return float("nan")
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def canonical_display(x: str) -> str:
    x = str(x).strip().lower()
    table = {
        "left": "left", "right": "right",
        "above": "above", "on": "above",
        "below": "below", "under": "below",
    }
    return table.get(x, x)


# =============================================================================
# Data / prior ranking
# =============================================================================

def load_data(a):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, int(a.max_samples), int(a.seed))
    return two, meta, rec_by_sid


def resolve_ranking(a) -> Tuple[Path, pd.DataFrame, Mapping]:
    ranked_path = Path(a.ranked_csv) if a.ranked_csv else Path(a.ranked_dir) / "ranked_k36_tokens.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(ranked_path)

    d = pd.read_csv(ranked_path)
    required = {
        "sid", "rank", "source_layer", "position", "token",
        "category", "broad_category", "mediation",
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"{ranked_path} missing columns: {sorted(missing)}")

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    d["mediation"] = pd.to_numeric(d["mediation"], errors="coerce")

    meta = {}
    meta_path = ranked_path.parent / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return ranked_path, d, meta


# =============================================================================
# Exact activation patching: REAL state -> clean GRAY state
# =============================================================================

class GrayStatePatcher:
    """
    exact_map:
        layer -> {prompt_position -> gray_hidden_vector}

    We patch decoder-block OUTPUT states.  In teacher-forced candidate scoring,
    later answer tokens cannot affect earlier prompt states under a causal mask,
    so gray prompt activations captured without appended answers are valid.
    """
    def __init__(self, decoder_layers, exact_map, full_len):
        self.handles = []
        self.full_len = int(full_len)
        self.applied = defaultdict(int)

        for L, pos_map in sorted(exact_map.items()):
            if not pos_map:
                continue
            self.handles.append(
                decoder_layers[int(L)].register_forward_hook(
                    self._make_hook(int(L), dict(pos_map))
                )
            )

    def _make_hook(self, L, pos_map):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) != self.full_len:
                return out
            y = h.float().clone()
            for p, gray_np in pos_map.items():
                p = int(p)
                if 0 <= p < int(y.shape[1]):
                    gray = torch.as_tensor(gray_np, device=y.device, dtype=torch.float32)
                    y[0, p, :] = gray
                    self.applied[L] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def sequence_score_with_gray_patch(
    *, model, decoder_layers, batch, answer_ids, reduction, exact_map
):
    ext, T = gate.extend_batch_with_candidate(batch, answer_ids)
    full_len = int(ext["input_ids"].shape[1])

    with torch.inference_mode():
        with GrayStatePatcher(decoder_layers, exact_map, full_len):
            kw = dict(ext)
            kw["use_cache"] = False
            kw["return_dict"] = True
            out = model(**kw)
            score = gate.sequence_score_from_logits(
                out.logits, T, answer_ids, reduction
            )
    return float(score.item())


def all_scores_with_gray_patch(
    *, model, decoder_layers, batch, candidate_ids, reduction, exact_map
):
    return {
        r: sequence_score_with_gray_patch(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
            exact_map=exact_map,
        )
        for r in REL
    }


def generation_with_gray_patch(
    *, model, processor, decoder_layers, batch, exact_map, max_new_tokens
):
    full_len = int(batch["input_ids"].shape[1])
    with GrayStatePatcher(decoder_layers, exact_map, full_len):
        text = base.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )
    pred = traj.normalize_relation(base, text)
    return text, pred


# =============================================================================
# State/control selection
# =============================================================================

def rows_to_exact_map(rows, hgray):
    exact = defaultdict(dict)
    for r in rows:
        L = int(r["source_layer"])
        p = int(r["position"])
        if L not in hgray or p < 0 or p >= int(hgray[L].shape[1]):
            continue
        exact[L][p] = np.asarray(hgray[L][0, p], dtype=np.float32)
    return {int(L): dict(v) for L, v in exact.items()}


def state_delta_norm(hreal, hgray, L: int, p: int) -> float:
    if L not in hreal or L not in hgray:
        return float("nan")
    if not (0 <= p < hreal[L].shape[1] and 0 <= p < hgray[L].shape[1]):
        return float("nan")
    d = hreal[L][0, p].astype(np.float32) - hgray[L][0, p].astype(np.float32)
    return float(np.linalg.norm(d))


def build_candidate_positions(ids, cats, source_layers, hreal, hgray, exclude_positions):
    npos = min(
        len(ids), len(cats),
        min(int(hreal[L].shape[1]) for L in source_layers),
        min(int(hgray[L].shape[1]) for L in source_layers),
    )
    out = []
    # Match ranking convention: exclude source last token.
    for p in range(max(0, npos - 1)):
        if int(p) in exclude_positions:
            continue
        cat = str(cats[p])
        broad = dyn.broad_category(cat)
        for L in source_layers:
            dn = state_delta_norm(hreal, hgray, int(L), int(p))
            out.append({
                "source_layer": int(L),
                "position": int(p),
                "token_id": int(ids[p]),
                "category": cat,
                "broad_category": broad,
                "delta_h_norm": dn,
            })
    return out


def select_norm_matched(top_rows, pool_rows):
    """Greedy SAME layer + SAME broad category + closest Real-Gray norm."""
    used_positions = set()
    selected = []

    for ref in top_rows:
        L = int(ref["source_layer"])
        cat = str(ref["broad_category"])
        target_norm = float(ref["actual_delta_h_norm"])

        pools = [
            [x for x in pool_rows
             if int(x["source_layer"]) == L
             and str(x["broad_category"]) == cat
             and int(x["position"]) not in used_positions],
            [x for x in pool_rows
             if int(x["source_layer"]) == L
             and int(x["position"]) not in used_positions],
        ]
        cand = next((z for z in pools if z), [])
        if not cand:
            continue

        # Relative/log distance is more stable across norm scales.
        def dist(x):
            a = max(float(x["delta_h_norm"]), 1e-12)
            b = max(target_norm, 1e-12)
            return abs(math.log(a) - math.log(b))

        x = min(cand, key=dist)
        rr = dict(x)
        rr["match_ref_layer"] = L
        rr["match_ref_position"] = int(ref["position"])
        rr["match_ref_delta_h_norm"] = target_norm
        rr["norm_log_distance"] = dist(x)
        selected.append(rr)
        used_positions.add(int(x["position"]))

    return selected


def select_random_matched(top_rows, pool_rows, seed):
    rng = random.Random(int(seed))
    used_positions = set()
    selected = []

    for ref in top_rows:
        L = int(ref["source_layer"])
        cat = str(ref["broad_category"])
        pools = [
            [x for x in pool_rows
             if int(x["source_layer"]) == L
             and str(x["broad_category"]) == cat
             and int(x["position"]) not in used_positions],
            [x for x in pool_rows
             if int(x["source_layer"]) == L
             and int(x["position"]) not in used_positions],
        ]
        cand = next((z for z in pools if z), [])
        if not cand:
            continue
        x = rng.choice(cand)
        rr = dict(x)
        rr["match_ref_layer"] = L
        rr["match_ref_position"] = int(ref["position"])
        selected.append(rr)
        used_positions.add(int(x["position"]))

    return selected


def add_actual_norm(rows, hreal, hgray):
    out = []
    for r in rows:
        rr = dict(r)
        rr["actual_delta_h_norm"] = state_delta_norm(
            hreal, hgray, int(rr["source_layer"]), int(rr["position"])
        )
        out.append(rr)
    return out


# =============================================================================
# Final-answer metrics
# =============================================================================

def score_metrics(scores: Mapping[str, float], gt: str):
    gt_score = float(scores[gt])
    others = {r: float(scores[r]) for r in REL if r != gt}
    best_other_rel = max(others, key=others.get)
    best_other = float(others[best_other_rel])
    pred = max(REL, key=lambda r: float(scores[r]))
    return {
        "gt_score": gt_score,
        "best_other_relation": best_other_rel,
        "best_other_score": best_other,
        "margin": gt_score - best_other,
        "score_prediction": pred,
        "score_correct": pred == gt,
    }


def make_score_row(sid, gt, condition, repeat, scores, clean_metrics=None):
    met = score_metrics(scores, gt)
    row = {
        "sid": int(sid),
        "gt": DISPLAY[gt],
        "condition": str(condition),
        "repeat": int(repeat),
        **{f"score_{DISPLAY[r]}": float(scores[r]) for r in REL},
        "gt_score": met["gt_score"],
        "best_other_relation": DISPLAY[met["best_other_relation"]],
        "best_other_score": met["best_other_score"],
        "margin": met["margin"],
        "score_prediction": DISPLAY[met["score_prediction"]],
        "score_correct": bool(met["score_correct"]),
    }
    if clean_metrics is not None:
        row["gt_score_drop"] = float(clean_metrics["gt_score"] - met["gt_score"])
        row["margin_drop"] = float(clean_metrics["margin"] - met["margin"])
        row["changed_score_prediction"] = (
            met["score_prediction"] != clean_metrics["score_prediction"]
        )
    else:
        row["gt_score_drop"] = 0.0
        row["margin_drop"] = 0.0
        row["changed_score_prediction"] = False
    return row, met


# =============================================================================
# Summary / plotting
# =============================================================================

def aggregate_random_per_sample(df):
    """Collapse random repeats to one mean effect per sample for fair pairing."""
    r = df[df["condition"] == "random_matched"].copy()
    if r.empty:
        return pd.DataFrame()
    return (
        r.groupby("sid", as_index=False)
        .agg(
            random_margin_drop=("margin_drop", "mean"),
            random_gt_score_drop=("gt_score_drop", "mean"),
            random_margin=("margin", "mean"),
        )
    )


def summarize_scores(df):
    rows = []
    clean = df[df["condition"] == "clean"].drop_duplicates("sid")
    clean_by_sid = clean.set_index("sid") if not clean.empty else None

    # Random repeats should not artificially multiply N. Average per sample first.
    pieces = []
    for condition, g in df.groupby("condition"):
        if condition == "random_matched":
            z = (
                g.groupby("sid", as_index=False)
                .agg(
                    gt_score=("gt_score", "mean"),
                    margin=("margin", "mean"),
                    gt_score_drop=("gt_score_drop", "mean"),
                    margin_drop=("margin_drop", "mean"),
                    score_correct=("score_correct", "mean"),
                    changed_score_prediction=("changed_score_prediction", "mean"),
                )
            )
            z["condition"] = condition
            pieces.append(z)
        else:
            z = g.drop_duplicates(["sid", "condition"]).copy()
            pieces.append(z)

    agg = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[:0].copy()

    for condition, g in agg.groupby("condition"):
        n = len(g)
        acc = safe_mean(g["score_correct"])
        margin_drop = list(map(float, g["margin_drop"]))
        gt_drop = list(map(float, g["gt_score_drop"]))

        c2w = w2c = 0
        if clean_by_sid is not None and condition != "clean":
            for _, row in g.iterrows():
                sid = int(row["sid"])
                if sid not in clean_by_sid.index:
                    continue
                c0 = bool(clean_by_sid.loc[sid, "score_correct"])
                c1 = float(row["score_correct"]) >= 0.5
                c2w += int(c0 and not c1)
                w2c += int((not c0) and c1)

        rows.append({
            "condition": condition,
            "N": int(n),
            "sequence_score_accuracy": acc,
            "mean_gt_score": safe_mean(g["gt_score"]),
            "mean_margin": safe_mean(g["margin"]),
            "mean_gt_score_drop": safe_mean(gt_drop),
            "median_gt_score_drop": safe_median(gt_drop),
            "mean_margin_drop": safe_mean(margin_drop),
            "median_margin_drop": safe_median(margin_drop),
            "sem_margin_drop": safe_sem(margin_drop),
            "fraction_margin_decreased": safe_mean(float(x > 0) for x in margin_drop),
            "C2W_by_sequence_score": int(c2w),
            "W2C_by_sequence_score": int(w2c),
        })
    return pd.DataFrame(rows), agg


def make_paired_table(agg):
    if agg.empty:
        return pd.DataFrame()

    wide = agg.pivot_table(
        index="sid", columns="condition", values="margin_drop", aggfunc="mean"
    )
    if "top" not in wide.columns:
        return pd.DataFrame()

    out = pd.DataFrame({"sid": wide.index, "top_margin_drop": wide["top"].values})
    for c in ("norm_matched", "random_matched", "tail"):
        if c in wide.columns:
            out[f"{c}_margin_drop"] = wide[c].values
            out[f"top_minus_{c}"] = wide["top"].values - wide[c].values
    return out.reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, outdir: Path):
    order = ["top", "norm_matched", "random_matched", "tail"]
    z = summary[summary["condition"].isin(order)].copy()
    if z.empty:
        return
    z["_order"] = z["condition"].map({x: i for i, x in enumerate(order)})
    z = z.sort_values("_order")

    labels = {
        "top": "Top-K",
        "norm_matched": "Norm-matched",
        "random_matched": "Random-matched",
        "tail": "K36 tail",
    }

    x = np.arange(len(z))
    y = z["mean_margin_drop"].to_numpy(dtype=float)
    e = z["sem_margin_drop"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.bar(x, y, yerr=e, capsize=3)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in z["condition"]])
    ax.set_ylabel("Final answer margin drop")
    ax.set_title("Removing image-dependent state at selected positions")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(outdir / "figure_margin_drop.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "figure_margin_drop.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if a.random_repeats < 0:
        raise ValueError("--random-repeats must be >= 0")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    ranked_path, ranked, prior_meta = resolve_ranking(a)
    ranked_sids = set(map(int, ranked["sid"].unique()))

    if a.source_layers.strip():
        source_layers = parse_ints(a.source_layers)
    else:
        source_layers = sorted(set(map(int, ranked["source_layer"].unique())))
        # Prefer explicit metadata if it agrees / exists.
        meta_layers = prior_meta.get("source_layers", None) if isinstance(prior_meta, dict) else None
        if isinstance(meta_layers, list) and meta_layers:
            source_layers = sorted(set(map(int, meta_layers)))

    two, meta, rec_by_sid = load_data(a)
    meta = [m for m in meta if int(m["sid"]) in ranked_sids]
    if int(a.eval_max_samples) > 0:
        meta = traj.stratified_cap(meta, int(a.eval_max_samples), int(a.seed) + 1)

    if not meta:
        raise RuntimeError("No evaluation samples overlap the ranked causal CSV")

    # Model load.
    spec = base.merged_model_specs(two)[a.model]
    cls = getattr(transformers, spec.model_class)
    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None
    score_rows = []
    selected_rows = []
    generation_rows = []

    try:
        print(f"[model] loading {spec.repo_id}", flush=True)
        try:
            model = cls.from_pretrained(spec.repo_id, **kw)
        except TypeError:
            kw["torch_dtype"] = kw.pop("dtype")
            model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)
        for L in source_layers:
            if not 0 <= int(L) < n_layers:
                raise ValueError(f"source L{L} invalid for n_layers={n_layers}")

        # Reuse repo-standard teacher-forced candidate scoring.
        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)
        device = torch.device(a.device)

        print("=" * 132)
        print("CAUSAL-STATE VALIDATION: REAL -> GRAY ACTIVATION PATCH")
        print("=" * 132)
        print(f"ranking={ranked_path}")
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(f"source layers={source_layers}")
        print(f"eval N={len(meta)} | top_k={a.top_k} | random repeats={a.random_repeats}")
        print("Primary metric: clean GT sequence margin - patched GT sequence margin")
        print("Positive margin_drop => removing selected state hurts the GT final decision.")
        print()

        ranked_by_sid = {
            int(sid): g.sort_values("rank").copy()
            for sid, g in ranked.groupby("sid")
        }

        for m in tqdm(meta, desc="GRAY-PATCH causal validation"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            real = gray = rb = gb = None

            try:
                g = ranked_by_sid[sid].copy()
                g = g[g["source_layer"].isin(source_layers)].sort_values("rank")
                if len(g) < a.top_k:
                    raise RuntimeError(f"sid={sid}: only {len(g)} ranked states")

                top_df = g.head(int(a.top_k)).copy()
                tail_df = g.tail(int(a.top_k)).copy()

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = dyn.build_categories(
                    model, processor, rb, ids, m["subject"], m["reference"]
                )

                # Clean source activations only; no gradients are used here.
                hreal = dyn.capture_cpu(model, decoder_layers, rb, source_layers)
                hgray = dyn.capture_cpu(model, decoder_layers, gb, source_layers)

                top_rows = add_actual_norm(top_df.to_dict("records"), hreal, hgray)
                tail_rows = add_actual_norm(tail_df.to_dict("records"), hreal, hgray)

                # Exclude EVERY token position already in the K36 ranking from matched controls.
                exclude_positions = set(map(int, g["position"].tolist()))
                pool = build_candidate_positions(
                    ids, cats, source_layers, hreal, hgray, exclude_positions
                )

                norm_rows = select_norm_matched(top_rows, pool)

                random_sets = []
                for rep in range(int(a.random_repeats)):
                    rseed = int(a.seed) * 1000003 + sid * 1009 + rep * 7919 + int(a.top_k)
                    random_sets.append(select_random_matched(top_rows, pool, rseed))

                # Export selected states.
                conditions = [("top", 0, top_rows), ("norm_matched", 0, norm_rows), ("tail", 0, tail_rows)]
                conditions += [("random_matched", rep, rr) for rep, rr in enumerate(random_sets, 1)]
                for cond, rep, rows in conditions:
                    for j, r in enumerate(rows, 1):
                        selected_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "condition": cond,
                            "repeat": int(rep),
                            "selection_index": j,
                            "source_layer": int(r["source_layer"]),
                            "position": int(r["position"]),
                            "token": str(toks[int(r["position"])]) if 0 <= int(r["position"]) < len(toks) else str(r.get("token", "")),
                            "category": str(r.get("category", "")),
                            "broad_category": str(r.get("broad_category", "")),
                            "mediation": float(r.get("mediation", float("nan"))),
                            "delta_h_norm": float(r.get("actual_delta_h_norm", r.get("delta_h_norm", float("nan")))),
                            "match_ref_layer": r.get("match_ref_layer", ""),
                            "match_ref_position": r.get("match_ref_position", ""),
                            "match_ref_delta_h_norm": r.get("match_ref_delta_h_norm", ""),
                            "norm_log_distance": r.get("norm_log_distance", ""),
                        })

                # CLEAN final answer sequence scores.
                clean_scores = gate.all_sequence_scores(
                    model, rb, candidate_ids, a.sequence_score_reduction
                )
                clean_row, clean_met = make_score_row(
                    sid, gt, "clean", 0, clean_scores, None
                )
                clean_row["n_selected"] = 0
                score_rows.append(clean_row)

                if a.run_generation:
                    clean_text = base.generate_text(
                        model, processor, rb, max_new_tokens=a.max_new_tokens
                    )
                    clean_pred = traj.normalize_relation(base, clean_text)
                    generation_rows.append({
                        "sid": sid, "gt": DISPLAY[gt], "condition": "clean", "repeat": 0,
                        "prediction": DISPLAY.get(clean_pred, clean_pred),
                        "correct": clean_pred == gt, "text": clean_text,
                    })

                # Patch each condition in a fresh forward.
                for cond, rep, rows in conditions:
                    if not rows:
                        continue
                    exact_map = rows_to_exact_map(rows, hgray)
                    patched_scores = all_scores_with_gray_patch(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        candidate_ids=candidate_ids,
                        reduction=a.sequence_score_reduction,
                        exact_map=exact_map,
                    )
                    row, _ = make_score_row(
                        sid, gt, cond, rep, patched_scores, clean_met
                    )
                    row["n_selected"] = int(sum(len(v) for v in exact_map.values()))
                    score_rows.append(row)

                    if a.run_generation:
                        text, pred = generation_with_gray_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            exact_map=exact_map,
                            max_new_tokens=a.max_new_tokens,
                        )
                        generation_rows.append({
                            "sid": sid, "gt": DISPLAY[gt], "condition": cond,
                            "repeat": int(rep),
                            "prediction": DISPLAY.get(pred, pred),
                            "correct": pred == gt, "text": text,
                            "n_selected": int(sum(len(v) for v in exact_map.values())),
                        })

            except Exception as exc:
                append_jsonl(error_path, {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}", flush=True)
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        score_df = pd.DataFrame(score_rows)
        selected_df = pd.DataFrame(selected_rows)
        score_df.to_csv(outdir / "per_sample_scores.csv", index=False)
        selected_df.to_csv(outdir / "selected_states.csv", index=False)

        summary, agg = summarize_scores(score_df)
        summary.to_csv(outdir / "summary.csv", index=False)
        agg.to_csv(outdir / "per_sample_condition_aggregated.csv", index=False)

        paired = make_paired_table(agg)
        paired.to_csv(outdir / "paired_top_vs_controls.csv", index=False)

        if generation_rows:
            gen_df = pd.DataFrame(generation_rows)
            gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)
            gen_sum = []
            # Random repeats average per sample before accuracy summary.
            for cond, gg in gen_df.groupby("condition"):
                if cond == "random_matched":
                    per_sid = gg.groupby("sid")["correct"].mean()
                    acc = float(per_sid.mean())
                    N = int(len(per_sid))
                else:
                    zz = gg.drop_duplicates(["sid", "condition"])
                    acc = safe_mean(float(x) for x in zz["correct"])
                    N = int(len(zz))
                gen_sum.append({"condition": cond, "N": N, "generation_accuracy": acc})
            pd.DataFrame(gen_sum).to_csv(outdir / "generation_summary.csv", index=False)

        plot_summary(summary, outdir)

        # Compact paired statistics.
        paired_stats = []
        for c in ("norm_matched", "random_matched", "tail"):
            col = f"top_minus_{c}"
            if col in paired.columns:
                vals = pd.to_numeric(paired[col], errors="coerce").dropna().to_numpy(dtype=float)
                paired_stats.append({
                    "comparison": f"top_vs_{c}",
                    "N": int(len(vals)),
                    "mean_top_minus_control_margin_drop": safe_mean(vals),
                    "median_top_minus_control_margin_drop": safe_median(vals),
                    "fraction_top_drop_larger": safe_mean(float(v > 0) for v in vals),
                })
        pd.DataFrame(paired_stats).to_csv(outdir / "paired_summary.csv", index=False)

        # Human-readable report.
        report = []
        report.append("=" * 108)
        report.append("GRAY-PATCH VALIDATION OF M-RANKED DECISION STATES")
        report.append("=" * 108)
        report.append(f"ranking: {ranked_path}")
        report.append(f"N requested: {len(meta)} | top_k={a.top_k}")
        report.append("")
        if not summary.empty:
            for _, r in summary.iterrows():
                report.append(
                    f"{str(r['condition']):>15s} | N={int(r['N']):3d} "
                    f"| score_acc={float(r['sequence_score_accuracy']):.4f} "
                    f"| mean margin drop={float(r['mean_margin_drop']):+.5f} "
                    f"| median={float(r['median_margin_drop']):+.5f} "
                    f"| P(drop>0)={float(r['fraction_margin_decreased']):.3f}"
                )
        report.append("")
        report.append("Interpretation:")
        report.append("  Positive margin_drop means REAL->GRAY patching hurt the GT final answer margin.")
        report.append("  Evidence for high-leverage states: Top-K drop >> norm/random/tail controls.")
        report.append("  This validates downstream influence without intervening along the gradient direction.")
        report_text = "\n".join(report) + "\n"
        print("\n" + report_text)
        (outdir / "analysis_summary.txt").write_text(report_text, encoding="utf-8")

        write_json(outdir / "metadata.json", {
            "script": "eval_qwen_causal_state_graypatch_validation_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "ranked_csv": str(ranked_path),
            "source_layers": source_layers,
            "top_k": int(a.top_k),
            "random_repeats": int(a.random_repeats),
            "eval_N_requested": len(meta),
            "gray_value": int(a.gray_value),
            "answer_surface": a.answer_surface,
            "candidate_texts": texts,
            "sequence_score_reduction": a.sequence_score_reduction,
            "patch": "exact decoder-block output replacement h_REAL[L,p] -> h_GRAY[L,p]",
            "primary_metric": "clean GT sequence margin - patched GT sequence margin",
            "ranking_note": (
                "States are selected by the prior oracle writer-guided M=(Real-Gray)^T grad J_writer ranking; "
                "this script uses no gradient to perform the validation patch."
            ),
            "control_note": (
                "norm/random controls exclude every token position in the prior K36 ranking; norm control "
                "matches source layer, broad token category, and Real-Gray norm as closely as possible."
            ),
        })

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
