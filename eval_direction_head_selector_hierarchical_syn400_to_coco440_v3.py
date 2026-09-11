#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_selector_hierarchical_syn400_to_coco440_v3.py

Source-only Direction-Head selector experiments built on top of the cached
residuals produced by eval_direction_head_selector_synthetic400_to_coco440_v2.py.

Goal
----
Improve the frozen Synthetic-400 -> COCO-440 selector WITHOUT using any COCO
label for fitting, head selection, weighting, or hyperparameter choice.

This script tests three families:

1) Original four-way Top-K ensemble (sanity check; should reproduce v2).
2) Hierarchical selector:
       axis: Horizontal (LEFT/RIGHT) vs Vertical (ABOVE/BELOW)
       then LEFT vs RIGHT or ABOVE vs BELOW.
   Axis, LR and AB use independently ranked Direction-Head pools.
3) Relation-specific head pools:
       each relation gets its own source-OOF-ranked head pool.

All task-specific head rankings are computed from held-out Synthetic OOF scores.
For the hierarchical family, K_axis / K_LR / K_AB are chosen ONLY by Synthetic
OOF performance; the chosen configuration is then frozen and transferred to COCO.

No model forward pass is required if the v2 source cache and COCO target cache
already exist.

Typical
-------
python -u eval_direction_head_selector_hierarchical_syn400_to_coco440_v3.py \
  --v2-run-dir output/qwen3b_direction_selector_syn400_to_coco440 \
  --target-direction-dir output/qwen3b_head_object_residual_direction \
  --task-topks 1,3,5,7,10,15,20,30 \
  --output-dir output/qwen3b_direction_selector_hier_v3
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import eval_direction_head_selector_synthetic400_to_coco440_v2 as v2
except Exception as exc:
    raise SystemExit(
        "Could not import eval_direction_head_selector_synthetic400_to_coco440_v2.py.\n"
        "Run this script from the AdaptVis repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
HORIZONTAL = {"left", "right"}
VERTICAL = {"above", "below"}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--v2-run-dir",
        required=True,
        help=(
            "Existing v2 output directory containing "
            "synthetic_direction_head_relation_vectors.npz."
        ),
    )
    p.add_argument(
        "--target-direction-dir",
        default="",
        help=(
            "COCO direction-head relation_vectors.npz or directory. "
            "If omitted, read target_path from v2 metadata.json."
        ),
    )
    p.add_argument("--cv-folds", type=int, default=0,
                   help="0 = reuse v2 metadata value.")
    p.add_argument("--cv-seed", type=int, default=-1,
                   help="-1 = reuse v2 metadata value.")
    p.add_argument(
        "--task-topks",
        default="1,3,5,7,10,15,20,30",
        help="Candidate K values for axis/LR/AB and relation-specific pools.",
    )
    p.add_argument(
        "--candidate-layers",
        default="",
        help="Optional comma-separated layers; empty = all.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used inside each head.",
    )
    p.add_argument(
        "--max-grid-rows",
        type=int,
        default=30,
        help="How many top source-only hierarchical grid rows to print/save preview for.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def normalize_rel(x):
    return v2.canon_rel(x)


def head_name(l, h):
    return f"L{int(l)}H{int(h):02d}"


def softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-8)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


def safe_mean(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def evaluate_pred(pred_rel, y):
    pred_rel = np.asarray(pred_rel, dtype=object)
    y = np.asarray(y, dtype=object)
    correct = pred_rel == y
    row = {
        "N": int(len(y)),
        "acc_all": float(correct.mean()),
    }
    vals = []
    for r in REL:
        m = y == r
        acc = float(correct[m].mean()) if np.any(m) else np.nan
        row[f"N_{r}"] = int(m.sum())
        row[f"acc_{r}"] = acc
        vals.append(acc)
    row["macro_acc"] = safe_mean(vals)
    row["min_relation_acc"] = float(np.nanmin(vals))
    return row, correct


def source_oof_scores(X, y, n_folds, seed):
    """
    Exact source-only OOF relation scores [N,L,H,4].
    Each sample is scored by a codebook fit without its held-out fold.
    """
    N = len(y)
    folds = v2.stratified_folds(y, n_folds, seed)
    all_idx = np.arange(N)
    out = np.full(
        (N, X.shape[1], X.shape[2], len(REL)),
        np.nan,
        dtype=np.float32,
    )
    fold_id = np.full(N, -1, dtype=int)

    for f, te in enumerate(folds):
        keep = np.ones(N, dtype=bool)
        keep[te] = False
        tr = all_idx[keep]
        center, dirs = v2.fit_codebook_all_heads(X[tr], y[tr])
        out[te] = v2.score_all_heads(X[te], center, dirs).astype(np.float32)
        fold_id[te] = f

    if np.isnan(out).any() or np.any(fold_id < 0):
        raise RuntimeError("OOF score construction incomplete.")
    return out, fold_id


def task_margins(scores, temperature):
    """
    Return per-head signed evidence [N,L,H]:
      axis >0 => horizontal; <0 => vertical
      lr   >0 => left;       <0 => right
      ab   >0 => above;      <0 => below

    Use probabilities inside each head so scale stays bounded.
    """
    p4 = softmax(scores, axis=-1, temperature=temperature)
    axis = (p4[..., RID["left"]] + p4[..., RID["right"]]) - (
        p4[..., RID["above"]] + p4[..., RID["below"]]
    )

    plr = softmax(
        scores[..., [RID["left"], RID["right"]]],
        axis=-1,
        temperature=temperature,
    )
    lr = plr[..., 0] - plr[..., 1]

    pab = softmax(
        scores[..., [RID["above"], RID["below"]]],
        axis=-1,
        temperature=temperature,
    )
    ab = pab[..., 0] - pab[..., 1]
    return axis, lr, ab


def task_head_accuracy(oof_scores, y, temperature):
    axis_m, lr_m, ab_m = task_margins(oof_scores, temperature)

    y = np.asarray(y, dtype=object)
    y_axis_h = np.asarray([r in HORIZONTAL for r in y], dtype=bool)

    axis_pred_h = axis_m >= 0
    axis_acc = (axis_pred_h == y_axis_h[:, None, None]).mean(axis=0)

    m_lr = np.isin(y, np.asarray(["left", "right"], dtype=object))
    lr_true_left = (y[m_lr] == "left")
    lr_pred_left = lr_m[m_lr] >= 0
    lr_acc = (
        lr_pred_left == lr_true_left[:, None, None]
    ).mean(axis=0)

    m_ab = np.isin(y, np.asarray(["above", "below"], dtype=object))
    ab_true_above = (y[m_ab] == "above")
    ab_pred_above = ab_m[m_ab] >= 0
    ab_acc = (
        ab_pred_above == ab_true_above[:, None, None]
    ).mean(axis=0)

    return {
        "axis": axis_acc.astype(np.float32),
        "lr": lr_acc.astype(np.float32),
        "ab": ab_acc.astype(np.float32),
    }


def fourway_head_accuracy(oof_scores, y):
    pred = np.argmax(oof_scores, axis=-1)
    yi = np.asarray([RID[r] for r in y], dtype=int)
    return (pred == yi[:, None, None]).mean(axis=0).astype(np.float32)


def relation_specific_balanced_accuracy(oof_scores, y):
    """
    For each relation r and head h:
      positive iff that head's four-way argmax == r.
    Rank by one-vs-rest balanced accuracy = (TPR + TNR)/2.
    This avoids a trivial 75%-negative classifier dominating the ranking.
    Returns [4,L,H].
    """
    y = np.asarray(y, dtype=object)
    pred = np.argmax(oof_scores, axis=-1)  # [N,L,H]
    out = np.zeros(
        (len(REL), oof_scores.shape[1], oof_scores.shape[2]),
        dtype=np.float32,
    )
    for ri, r in enumerate(REL):
        true_pos = y == r
        pred_pos = pred == ri
        tpr = pred_pos[true_pos].mean(axis=0)
        tnr = (~pred_pos[~true_pos]).mean(axis=0)
        out[ri] = 0.5 * (tpr + tnr)
    return out


def rank_acc(acc, candidate_layers=None):
    rows = []
    allowed = set(candidate_layers) if candidate_layers else None
    for l in range(acc.shape[0]):
        if allowed is not None and l not in allowed:
            continue
        for h in range(acc.shape[1]):
            rows.append((float(acc[l, h]), int(l), int(h)))
    rows.sort(key=lambda z: (z[0], -z[1], -z[2]), reverse=True)
    return rows


def selected_heads(ranked, k):
    if k > len(ranked):
        raise ValueError(f"K={k} > available heads={len(ranked)}")
    return [(l, h) for _, l, h in ranked[:k]]


def aggregate_margin(margin, heads, reliability, weighted):
    """
    margin [N,L,H], return [N].
    Binary task weights are reliability - 0.5.
    """
    out = np.zeros(margin.shape[0], dtype=np.float64)
    den = 0.0
    for l, h in heads:
        w = max(float(reliability[l, h]) - 0.5, 0.0) if weighted else 1.0
        if w <= 0:
            continue
        out += w * margin[:, l, h]
        den += w
    if den <= EPS:
        return np.mean(
            np.stack([margin[:, l, h] for l, h in heads], axis=1),
            axis=1,
        )
    return out / den


def hierarchical_predict(
    scores,
    axis_heads,
    lr_heads,
    ab_heads,
    task_acc,
    temperature,
    weighted,
):
    axis_m, lr_m, ab_m = task_margins(scores, temperature)

    axis_score = aggregate_margin(
        axis_m, axis_heads, task_acc["axis"], weighted
    )
    lr_score = aggregate_margin(
        lr_m, lr_heads, task_acc["lr"], weighted
    )
    ab_score = aggregate_margin(
        ab_m, ab_heads, task_acc["ab"], weighted
    )

    pred = np.empty(scores.shape[0], dtype=object)
    horizontal = axis_score >= 0
    pred[horizontal & (lr_score >= 0)] = "left"
    pred[horizontal & (lr_score < 0)] = "right"
    pred[(~horizontal) & (ab_score >= 0)] = "above"
    pred[(~horizontal) & (ab_score < 0)] = "below"

    # A conservative two-stage confidence surrogate: the weakest decision margin.
    local_margin = np.where(horizontal, np.abs(lr_score), np.abs(ab_score))
    margin = np.minimum(np.abs(axis_score), local_margin)

    return pred, {
        "axis_score": axis_score,
        "lr_score": lr_score,
        "ab_score": ab_score,
        "margin": margin,
    }


def fourway_ensemble_predict(
    scores,
    heads,
    reliability,
    temperature,
    weighted=False,
):
    probs = softmax(scores, axis=-1, temperature=temperature)
    out = np.zeros((scores.shape[0], len(REL)), dtype=np.float64)
    den = 0.0
    for l, h in heads:
        w = max(float(reliability[l, h]) - 0.25, 0.0) if weighted else 1.0
        if w <= 0:
            continue
        out += w * probs[:, l, h, :]
        den += w
    if den <= EPS:
        out = np.mean(
            np.stack([probs[:, l, h, :] for l, h in heads], axis=1),
            axis=1,
        )
    else:
        out /= den

    idx = np.argmax(out, axis=1)
    pred = np.asarray([REL[int(i)] for i in idx], dtype=object)
    s = np.sort(out, axis=1)
    return pred, out, s[:, -1] - s[:, -2]


def relation_specific_predict(
    scores,
    pools,
    rel_reliability,
    temperature,
    weighted,
):
    """
    Relation r gets a different head pool. Candidate relation score is the mean
    p_h(r) over its source-OOF-selected pool.

    Weight for relation r/head h: balanced_acc(r,h) - 0.5.
    """
    probs = softmax(scores, axis=-1, temperature=temperature)
    out = np.zeros((scores.shape[0], len(REL)), dtype=np.float64)

    for ri, r in enumerate(REL):
        heads = pools[r]
        den = 0.0
        for l, h in heads:
            w = (
                max(float(rel_reliability[ri, l, h]) - 0.5, 0.0)
                if weighted else 1.0
            )
            if w <= 0:
                continue
            out[:, ri] += w * probs[:, l, h, ri]
            den += w
        if den <= EPS:
            out[:, ri] = np.mean(
                np.stack([probs[:, l, h, ri] for l, h in heads], axis=1),
                axis=1,
            )
        else:
            out[:, ri] /= den

    idx = np.argmax(out, axis=1)
    pred = np.asarray([REL[int(i)] for i in idx], dtype=object)
    s = np.sort(out, axis=1)
    return pred, out, s[:, -1] - s[:, -2]


def head_table(task_name, ranked, n=30):
    rows = []
    for rank, (acc, l, h) in enumerate(ranked[:n], 1):
        rows.append({
            "task": task_name,
            "rank": rank,
            "layer": l,
            "head": h,
            "head_name": head_name(l, h),
            "source_oof_task_accuracy": acc,
            "weight_over_binary_chance": max(acc - 0.5, 0.0),
        })
    return rows


def grid_hierarchical(
    oof_scores,
    y,
    task_ranked,
    task_acc,
    ks,
    temperature,
    weighted,
):
    """
    Hyperparameter selection uses ONLY source OOF predictions.
    """
    # Precompute aggregate scores per K.
    axis_m, lr_m, ab_m = task_margins(oof_scores, temperature)
    cached = {"axis": {}, "lr": {}, "ab": {}}
    for task, margin in [("axis", axis_m), ("lr", lr_m), ("ab", ab_m)]:
        for k in ks:
            heads = selected_heads(task_ranked[task], k)
            cached[task][k] = aggregate_margin(
                margin, heads, task_acc[task], weighted
            )

    rows = []
    for ka in ks:
        a = cached["axis"][ka]
        horizontal = a >= 0
        for klr in ks:
            lr = cached["lr"][klr]
            for kab in ks:
                ab = cached["ab"][kab]
                pred = np.empty(len(y), dtype=object)
                pred[horizontal & (lr >= 0)] = "left"
                pred[horizontal & (lr < 0)] = "right"
                pred[(~horizontal) & (ab >= 0)] = "above"
                pred[(~horizontal) & (ab < 0)] = "below"
                met, _ = evaluate_pred(pred, y)
                rows.append({
                    "weighted": bool(weighted),
                    "K_axis": int(ka),
                    "K_LR": int(klr),
                    "K_AB": int(kab),
                    **met,
                    "total_K_nominal": int(ka + klr + kab),
                })

    df = pd.DataFrame(rows).sort_values(
        ["acc_all", "min_relation_acc", "total_K_nominal"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return df


def add_detail_columns(detail, method, pred, margin=None, extra=None):
    detail[f"pred_{method}"] = pred
    detail[f"correct_{method}"] = pred == detail["gt"].to_numpy(object)
    if margin is not None:
        detail[f"margin_{method}"] = np.asarray(margin, dtype=float)
    if extra:
        for name, val in extra.items():
            detail[f"{name}_{method}"] = np.asarray(val)


def main():
    a = parse_args()
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    run = Path(a.v2_run_dir)
    source_cache = run / "synthetic_direction_head_relation_vectors.npz"
    meta_path = run / "metadata.json"
    if not source_cache.exists():
        raise FileNotFoundError(source_cache)

    meta_v2 = {}
    if meta_path.exists():
        meta_v2 = json.loads(meta_path.read_text(encoding="utf-8"))

    cv_folds = (
        int(a.cv_folds) if int(a.cv_folds) > 0
        else int(meta_v2.get("source_cv_folds", 5))
    )
    cv_seed = (
        int(a.cv_seed) if int(a.cv_seed) >= 0
        else int(meta_v2.get("source_cv_seed", 17))
    )

    target_arg = a.target_direction_dir or meta_v2.get("target_path", "")
    if not target_arg:
        raise RuntimeError(
            "Need --target-direction-dir because v2 metadata lacks target_path."
        )

    ks = parse_ints(a.task_topks)
    if not ks:
        raise ValueError("--task-topks is empty")
    candidate_layers = (
        parse_ints(a.candidate_layers) if a.candidate_layers else None
    )

    source_sid, source_y, source_X = v2.load_source_cache(source_cache)
    source_y = np.asarray([normalize_rel(x) for x in source_y], dtype=object)
    valid = np.isin(source_y, np.asarray(REL, dtype=object))
    source_sid, source_y, source_X = (
        source_sid[valid], source_y[valid], source_X[valid]
    )

    target_path, target_sid, target_y, target_X = v2.load_target_cache(target_arg)
    target_y = np.asarray([normalize_rel(x) for x in target_y], dtype=object)

    if source_X.shape[1:] != target_X.shape[1:]:
        raise RuntimeError(
            f"Geometry mismatch source={source_X.shape}, target={target_X.shape}"
        )

    max_heads = source_X.shape[1] * source_X.shape[2]
    if candidate_layers:
        max_heads = len(candidate_layers) * source_X.shape[2]
    bad_k = [k for k in ks if k <= 0 or k > max_heads]
    if bad_k:
        raise ValueError(f"Invalid K values {bad_k}; available heads={max_heads}")

    print("=" * 170)
    print("SOURCE-ONLY HIERARCHICAL DIRECTION-HEAD SELECTOR")
    print("=" * 170)
    print(
        f"Synthetic source N={len(source_y)} counts={dict(Counter(source_y.tolist()))}"
    )
    print(
        f"COCO target N={len(target_y)} counts={dict(Counter(target_y.tolist()))}"
    )
    print(f"source cache={source_cache}")
    print(f"target cache={target_path}")
    print(f"CV={cv_folds}-fold seed={cv_seed}")
    print(f"candidate layers={candidate_layers if candidate_layers else 'ALL'}")
    print(f"candidate task K={ks}")
    print("Target GT is evaluation-only.")
    print()

    # ------------------------------------------------------------------
    # Source-only OOF scores and task-specific reliability.
    # ------------------------------------------------------------------
    print("[1/4] Building Synthetic source OOF scores...", flush=True)
    oof_scores, fold_id = source_oof_scores(
        source_X, source_y, cv_folds, cv_seed
    )
    four_acc = fourway_head_accuracy(oof_scores, source_y)
    task_acc = task_head_accuracy(oof_scores, source_y, a.temperature)
    rel_balacc = relation_specific_balanced_accuracy(oof_scores, source_y)

    ranked4 = rank_acc(four_acc, candidate_layers)
    task_ranked = {
        task: rank_acc(task_acc[task], candidate_layers)
        for task in ("axis", "lr", "ab")
    }
    rel_ranked = {
        r: rank_acc(rel_balacc[ri], candidate_layers)
        for ri, r in enumerate(REL)
    }

    task_rows = []
    for task in ("axis", "lr", "ab"):
        task_rows.extend(head_table(task, task_ranked[task], n=50))
    for r in REL:
        for rank, (acc, l, h) in enumerate(rel_ranked[r][:50], 1):
            task_rows.append({
                "task": f"ovr_{r}",
                "rank": rank,
                "layer": l,
                "head": h,
                "head_name": head_name(l, h),
                "source_oof_task_accuracy": acc,
                "weight_over_binary_chance": max(acc - 0.5, 0.0),
            })
    pd.DataFrame(task_rows).to_csv(
        outdir / "source_oof_task_head_rankings.csv", index=False
    )

    # ------------------------------------------------------------------
    # Select hierarchical K ONLY from Synthetic OOF.
    # ------------------------------------------------------------------
    print("[2/4] Searching K_axis/K_LR/K_AB on Synthetic OOF only...", flush=True)
    grid_equal = grid_hierarchical(
        oof_scores, source_y, task_ranked, task_acc,
        ks, a.temperature, weighted=False
    )
    grid_weighted = grid_hierarchical(
        oof_scores, source_y, task_ranked, task_acc,
        ks, a.temperature, weighted=True
    )
    grid_all = pd.concat(
        [
            grid_equal.assign(weighting="equal"),
            grid_weighted.assign(weighting="task_reliability"),
        ],
        ignore_index=True,
    ).sort_values(
        ["acc_all", "min_relation_acc", "total_K_nominal"],
        ascending=[False, False, True],
    )
    grid_all.to_csv(outdir / "source_oof_hierarchical_grid.csv", index=False)

    best_equal = grid_equal.iloc[0].to_dict()
    best_weighted = grid_weighted.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Fit source codebook on ALL Synthetic only, freeze, target transfer.
    # ------------------------------------------------------------------
    print("[3/4] Refit source codebook on all Synthetic-400 and freeze...", flush=True)
    center, dirs = v2.fit_codebook_all_heads(source_X, source_y)
    target_scores = v2.score_all_heads(target_X, center, dirs)

    # Also use full-source-fitted scores only for saved codebook / target;
    # all rankings and K choices above came from source OOF.
    np.savez_compressed(
        outdir / "synthetic_fitted_direction_codebook_v3.npz",
        center=center,
        directions=dirs,
        relations=np.asarray(REL, dtype=object),
    )

    result_rows = []
    detail = pd.DataFrame({"sid": target_sid, "gt": target_y})

    # Sanity: original four-way Top10 equal, selected by source OOF 4-way acc.
    k0 = 10 if max_heads >= 10 else min(ks)
    base_heads = selected_heads(ranked4, k0)
    pred, probs, margin = fourway_ensemble_predict(
        target_scores, base_heads, four_acc, a.temperature, weighted=False
    )
    met, _ = evaluate_pred(pred, target_y)
    method = f"fourway_top{k0}_equal_repro"
    result_rows.append({
        "method": method,
        "selection_source": "Synthetic OOF only",
        "K_axis": np.nan, "K_LR": np.nan, "K_AB": np.nan,
        "heads": ",".join(head_name(l,h) for l,h in base_heads),
        **met,
    })
    add_detail_columns(detail, method, pred, margin)

    # Hierarchical fixed same-K curves.
    for k in ks:
        pools = {
            task: selected_heads(task_ranked[task], k)
            for task in ("axis", "lr", "ab")
        }
        for weighted in (False, True):
            pred, info = hierarchical_predict(
                target_scores,
                pools["axis"], pools["lr"], pools["ab"],
                task_acc, a.temperature, weighted,
            )
            met, _ = evaluate_pred(pred, target_y)
            method = (
                f"hier_fixed{k}_"
                + ("weighted" if weighted else "equal")
            )
            result_rows.append({
                "method": method,
                "selection_source": "Synthetic OOF only",
                "K_axis": k, "K_LR": k, "K_AB": k,
                "axis_heads": ",".join(head_name(l,h) for l,h in pools["axis"]),
                "lr_heads": ",".join(head_name(l,h) for l,h in pools["lr"]),
                "ab_heads": ",".join(head_name(l,h) for l,h in pools["ab"]),
                **met,
            })
            add_detail_columns(
                detail, method, pred, info["margin"],
                extra={
                    "axis_score": info["axis_score"],
                    "lr_score": info["lr_score"],
                    "ab_score": info["ab_score"],
                },
            )

    # Hierarchical source-OOF champions (the clean methods to carry forward).
    for label, best, weighted in [
        ("hier_source_best_equal", best_equal, False),
        ("hier_source_best_weighted", best_weighted, True),
    ]:
        ka, klr, kab = (
            int(best["K_axis"]), int(best["K_LR"]), int(best["K_AB"])
        )
        pools = {
            "axis": selected_heads(task_ranked["axis"], ka),
            "lr": selected_heads(task_ranked["lr"], klr),
            "ab": selected_heads(task_ranked["ab"], kab),
        }
        pred, info = hierarchical_predict(
            target_scores,
            pools["axis"], pools["lr"], pools["ab"],
            task_acc, a.temperature, weighted,
        )
        met, _ = evaluate_pred(pred, target_y)
        result_rows.append({
            "method": label,
            "selection_source": "Synthetic OOF champion; COCO unseen for selection",
            "source_oof_acc": float(best["acc_all"]),
            "source_oof_min_relation_acc": float(best["min_relation_acc"]),
            "K_axis": ka, "K_LR": klr, "K_AB": kab,
            "axis_heads": ",".join(head_name(l,h) for l,h in pools["axis"]),
            "lr_heads": ",".join(head_name(l,h) for l,h in pools["lr"]),
            "ab_heads": ",".join(head_name(l,h) for l,h in pools["ab"]),
            **met,
        })
        add_detail_columns(
            detail, label, pred, info["margin"],
            extra={
                "axis_score": info["axis_score"],
                "lr_score": info["lr_score"],
                "ab_score": info["ab_score"],
            },
        )

    # Relation-specific pools. K itself is source-selected from this family:
    # evaluate all K on source OOF first, choose champion, then target.
    for weighted in (False, True):
        source_candidates = []
        for k in ks:
            pools = {
                r: selected_heads(rel_ranked[r], k)
                for r in REL
            }
            spred, _, _ = relation_specific_predict(
                oof_scores, pools, rel_balacc, a.temperature, weighted
            )
            smet, _ = evaluate_pred(spred, source_y)
            source_candidates.append((smet["acc_all"], smet["min_relation_acc"], k))

            # Target fixed-K diagnostics.
            tpred, tscore, tmargin = relation_specific_predict(
                target_scores, pools, rel_balacc, a.temperature, weighted
            )
            tmet, _ = evaluate_pred(tpred, target_y)
            method = (
                f"relspecific_fixed{k}_"
                + ("weighted" if weighted else "equal")
            )
            result_rows.append({
                "method": method,
                "selection_source": "Synthetic OOF only",
                "K_relation_each": k,
                **tmet,
            })
            add_detail_columns(detail, method, tpred, tmargin)

        source_candidates.sort(key=lambda z: (z[0], z[1], -z[2]), reverse=True)
        sacc, smin, bestk = source_candidates[0]
        pools = {r: selected_heads(rel_ranked[r], bestk) for r in REL}
        tpred, tscore, tmargin = relation_specific_predict(
            target_scores, pools, rel_balacc, a.temperature, weighted
        )
        tmet, _ = evaluate_pred(tpred, target_y)
        method = (
            "relspecific_source_best_"
            + ("weighted" if weighted else "equal")
        )
        result_rows.append({
            "method": method,
            "selection_source": "Synthetic OOF champion; COCO unseen for selection",
            "source_oof_acc": sacc,
            "source_oof_min_relation_acc": smin,
            "K_relation_each": bestk,
            "left_heads": ",".join(head_name(l,h) for l,h in pools["left"]),
            "right_heads": ",".join(head_name(l,h) for l,h in pools["right"]),
            "above_heads": ",".join(head_name(l,h) for l,h in pools["above"]),
            "below_heads": ",".join(head_name(l,h) for l,h in pools["below"]),
            **tmet,
        })
        add_detail_columns(detail, method, tpred, tmargin)

    print("[4/4] Saving frozen COCO evaluation...", flush=True)
    results = pd.DataFrame(result_rows).sort_values(
        ["acc_all", "min_relation_acc"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results.to_csv(outdir / "coco440_selector_results_v3.csv", index=False)
    detail.to_csv(outdir / "coco440_predictions_v3.csv", index=False)

    # Compact task-head table for terminal.
    task_top = pd.DataFrame(task_rows)
    terminal_head_rows = []
    for task in ("axis", "lr", "ab"):
        terminal_head_rows.append(
            task_top[task_top["task"] == task].head(10)
        )
    terminal_heads = pd.concat(terminal_head_rows, ignore_index=True)

    metadata = {
        "script": "eval_direction_head_selector_hierarchical_syn400_to_coco440_v3.py",
        "source_cache": str(source_cache),
        "target_cache": str(target_path),
        "source_N": int(len(source_y)),
        "target_N": int(len(target_y)),
        "source_cv_folds": int(cv_folds),
        "source_cv_seed": int(cv_seed),
        "candidate_layers": candidate_layers,
        "task_topks": ks,
        "temperature": float(a.temperature),
        "target_gt_usage": "evaluation only",
        "source_only_selection": True,
        "hierarchical_rule": (
            "axis H/V first; then LR or ABOVE/BELOW; "
            "separate source-OOF-ranked head pools"
        ),
        "hier_equal_source_champion": {
            "K_axis": int(best_equal["K_axis"]),
            "K_LR": int(best_equal["K_LR"]),
            "K_AB": int(best_equal["K_AB"]),
            "source_oof_acc": float(best_equal["acc_all"]),
        },
        "hier_weighted_source_champion": {
            "K_axis": int(best_weighted["K_axis"]),
            "K_LR": int(best_weighted["K_LR"]),
            "K_AB": int(best_weighted["K_AB"]),
            "source_oof_acc": float(best_weighted["acc_all"]),
        },
        "note": (
            "Gray residual ensemble is intentionally not included in this fast v3: "
            "the existing v2/COCO caches store Real-NoImage residuals. "
            "This run isolates whether structured head specialization alone improves transfer."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 170)
    print("SOURCE OOF TASK-SPECIFIC TOP HEADS")
    print("=" * 170)
    print(
        terminal_heads[
            ["task", "rank", "head_name", "source_oof_task_accuracy"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print()
    print("=" * 170)
    print("TOP SOURCE-ONLY HIERARCHICAL K CONFIGURATIONS")
    print("=" * 170)
    print(
        grid_all.head(a.max_grid_rows)[
            [
                "weighting", "K_axis", "K_LR", "K_AB",
                "acc_all", "acc_left", "acc_right", "acc_above", "acc_below",
                "min_relation_acc",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print()
    print("=" * 170)
    print("FROZEN SYNTHETIC-400 -> COCO-440 SELECTOR RESULTS")
    print("=" * 170)
    show = [
        "method", "acc_all", "acc_left", "acc_right",
        "acc_above", "acc_below", "min_relation_acc",
    ]
    print(
        results[show].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Clean source-selected methods to compare against v2 top10_equal=0.8205:")
    keep = results[
        results["method"].isin(
            [
                "fourway_top10_equal_repro",
                "hier_source_best_equal",
                "hier_source_best_weighted",
                "relspecific_source_best_equal",
                "relspecific_source_best_weighted",
            ]
        )
    ]
    print(
        keep[show].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print()
    print("Saved:", outdir)
    print("  source_oof_task_head_rankings.csv")
    print("  source_oof_hierarchical_grid.csv")
    print("  coco440_selector_results_v3.csv")
    print("  coco440_predictions_v3.csv")
    print("  synthetic_fitted_direction_codebook_v3.npz")
    print("  metadata.json")


if __name__ == "__main__":
    main()
