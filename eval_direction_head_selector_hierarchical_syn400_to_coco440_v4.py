#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_selector_hierarchical_syn400_to_coco440_v4.py

Pure selector post-processing.
NO model loading.
NO image forward.
NO causal intervention.
NO dependency on eval_direction_head_selector_synthetic400_to_coco440_v2.py.

Inputs
------
1) Synthetic-400 cached direction-head residuals:
   synthetic_direction_head_relation_vectors.npz

2) COCO-440 direction-head cache:
   relation_vectors.npz

Both caches must contain:
    residual : [N, L, H, D]
    relation : [N]
and preferably:
    sample_index : [N]

Methods
-------
A. fourway_top10_equal_repro
   Reproduce the original source-only Top10 four-way selector.

B. hierarchical selector
   Stage 1: Horizontal vs Vertical
   Stage 2a: Left vs Right
   Stage 2b: Above vs Below

   Axis / LR / AB each rank their OWN heads using Synthetic OOF only.
   K_axis / K_LR / K_AB are selected using Synthetic OOF only.

C. relation-specific pools
   LEFT / RIGHT / ABOVE / BELOW each get their own source-OOF-ranked heads.

COCO labels are used ONLY for final reporting.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--source-cache",
        required=True,
        help="Synthetic-400 synthetic_direction_head_relation_vectors.npz",
    )
    p.add_argument(
        "--target-direction-dir",
        required=True,
        help="COCO relation_vectors.npz OR directory containing it",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=17)
    p.add_argument(
        "--task-topks",
        default="1,3,5,7,10,15,20,30",
    )
    p.add_argument(
        "--candidate-layers",
        default="",
        help="Optional comma-separated layers; empty = all layers",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    mp = {
        "left": "left",
        "left_of": "left",
        "left of": "left",
        "l": "left",
        "right": "right",
        "right_of": "right",
        "right of": "right",
        "r": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "bottom": "below",
    }
    return mp.get(s, s)


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def head_name(l, h):
    return f"L{int(l)}H{int(h):02d}"


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-8)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


def resolve_npz(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    q = p / "relation_vectors.npz"
    if not q.exists():
        raise FileNotFoundError(q)
    return q


def load_cache(path_or_dir):
    p = resolve_npz(path_or_dir)
    z = np.load(p, allow_pickle=True)

    required = {"relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{p}: missing {sorted(missing)}")

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray([canon_rel(v) for v in z["relation"]], dtype=object)

    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    elif "sid" in z.files:
        sid = np.asarray(z["sid"]).astype(int)
    else:
        sid = np.arange(len(y), dtype=int)

    valid = np.isin(y, np.asarray(REL, dtype=object))
    X = X[valid]
    y = y[valid]
    sid = sid[valid]

    if X.ndim != 4:
        raise RuntimeError(f"{p}: expected [N,L,H,D], got {X.shape}")
    if len(y) != len(X):
        raise RuntimeError(f"{p}: label/residual length mismatch")

    return p, sid, y, X


def fit_codebook(X, y):
    """
    center_h = mean(X_h)
    d_h,r = normalize(mean(X_h | r) - center_h)
    """
    center = X.mean(axis=0)
    dirs = np.zeros(
        (X.shape[1], X.shape[2], len(REL), X.shape[3]),
        dtype=np.float32,
    )

    for ri, r in enumerate(REL):
        m = y == r
        if not np.any(m):
            raise RuntimeError(f"Missing source relation={r}")
        d = X[m].mean(axis=0) - center
        dirs[:, :, ri, :] = normalize(d, axis=-1)

    return center.astype(np.float32), dirs


def score_codebook(X, center, dirs):
    """
    cos(X - source_center, d_r)
    -> [N,L,H,4]
    """
    Xc = X - center[None, :, :, :]
    Xn = normalize(Xc, axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", Xn, dirs, optimize=True)


def stratified_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]

    for r in REL:
        idx = np.where(y == r)[0].copy()
        rng.shuffle(idx)
        pieces = np.array_split(idx, n_folds)
        for f, piece in enumerate(pieces):
            folds[f].extend(piece.tolist())

    return [np.asarray(sorted(x), dtype=int) for x in folds]


def source_oof_scores(X, y, n_folds, seed):
    """
    Exact source-only OOF scores for every sample/head/relation.
    """
    N = len(y)
    out = np.full(
        (N, X.shape[1], X.shape[2], len(REL)),
        np.nan,
        dtype=np.float32,
    )

    folds = stratified_folds(y, n_folds, seed)
    all_idx = np.arange(N)

    for f, te in enumerate(folds):
        keep = np.ones(N, dtype=bool)
        keep[te] = False
        tr = all_idx[keep]

        center, dirs = fit_codebook(X[tr], y[tr])
        out[te] = score_codebook(X[te], center, dirs)

    if np.isnan(out).any():
        raise RuntimeError("OOF score matrix incomplete")

    return out


def fourway_head_acc(scores, y):
    pred = np.argmax(scores, axis=-1)
    yi = np.asarray([RID[r] for r in y], dtype=int)
    return (pred == yi[:, None, None]).mean(axis=0).astype(np.float32)


def binary_task_margins(scores, temperature):
    """
    axis > 0 : horizontal
    LR   > 0 : left
    AB   > 0 : above
    """
    p4 = softmax(scores, axis=-1, temperature=temperature)

    axis = (
        p4[..., RID["left"]]
        + p4[..., RID["right"]]
        - p4[..., RID["above"]]
        - p4[..., RID["below"]]
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


def binary_task_head_acc(oof_scores, y, temperature):
    axis, lr, ab = binary_task_margins(oof_scores, temperature)
    y = np.asarray(y, dtype=object)

    true_h = np.isin(y, ["left", "right"])
    axis_acc = ((axis >= 0) == true_h[:, None, None]).mean(axis=0)

    m_lr = np.isin(y, ["left", "right"])
    true_left = y[m_lr] == "left"
    lr_acc = (
        (lr[m_lr] >= 0)
        == true_left[:, None, None]
    ).mean(axis=0)

    m_ab = np.isin(y, ["above", "below"])
    true_above = y[m_ab] == "above"
    ab_acc = (
        (ab[m_ab] >= 0)
        == true_above[:, None, None]
    ).mean(axis=0)

    return {
        "axis": axis_acc.astype(np.float32),
        "lr": lr_acc.astype(np.float32),
        "ab": ab_acc.astype(np.float32),
    }


def relation_specific_balanced_acc(oof_scores, y):
    """
    Rank heads separately for each relation with one-vs-rest balanced accuracy.
    """
    pred = np.argmax(oof_scores, axis=-1)
    y = np.asarray(y, dtype=object)

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


def rank_heads(acc, candidate_layers=None):
    allowed = set(candidate_layers) if candidate_layers else None
    rows = []

    for l in range(acc.shape[0]):
        if allowed is not None and l not in allowed:
            continue
        for h in range(acc.shape[1]):
            rows.append((float(acc[l, h]), int(l), int(h)))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def select_top(ranked, k):
    return [(l, h) for _, l, h in ranked[:k]]


def aggregate_binary_margin(margin, heads, reliability, weighted):
    out = np.zeros(margin.shape[0], dtype=np.float64)
    den = 0.0

    for l, h in heads:
        w = (
            max(float(reliability[l, h]) - 0.5, 0.0)
            if weighted
            else 1.0
        )
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
    weighted=False,
):
    axis_m, lr_m, ab_m = binary_task_margins(scores, temperature)

    axis_s = aggregate_binary_margin(
        axis_m, axis_heads, task_acc["axis"], weighted
    )
    lr_s = aggregate_binary_margin(
        lr_m, lr_heads, task_acc["lr"], weighted
    )
    ab_s = aggregate_binary_margin(
        ab_m, ab_heads, task_acc["ab"], weighted
    )

    horizontal = axis_s >= 0

    pred = np.empty(scores.shape[0], dtype=object)
    pred[horizontal & (lr_s >= 0)] = "left"
    pred[horizontal & (lr_s < 0)] = "right"
    pred[(~horizontal) & (ab_s >= 0)] = "above"
    pred[(~horizontal) & (ab_s < 0)] = "below"

    local = np.where(horizontal, np.abs(lr_s), np.abs(ab_s))
    margin = np.minimum(np.abs(axis_s), local)

    return pred, axis_s, lr_s, ab_s, margin


def fourway_predict(scores, heads, reliability, temperature, weighted=False):
    probs = softmax(scores, axis=-1, temperature=temperature)
    out = np.zeros((scores.shape[0], len(REL)), dtype=np.float64)
    den = 0.0

    for l, h in heads:
        w = (
            max(float(reliability[l, h]) - 0.25, 0.0)
            if weighted
            else 1.0
        )
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
    margin = s[:, -1] - s[:, -2]

    return pred, margin


def relation_specific_predict(
    scores,
    pools,
    reliability,
    temperature,
    weighted=False,
):
    probs = softmax(scores, axis=-1, temperature=temperature)
    out = np.zeros((scores.shape[0], len(REL)), dtype=np.float64)

    for ri, r in enumerate(REL):
        heads = pools[r]
        den = 0.0

        for l, h in heads:
            w = (
                max(float(reliability[ri, l, h]) - 0.5, 0.0)
                if weighted
                else 1.0
            )
            if w <= 0:
                continue

            out[:, ri] += w * probs[:, l, h, ri]
            den += w

        if den <= EPS:
            out[:, ri] = np.mean(
                np.stack(
                    [probs[:, l, h, ri] for l, h in heads],
                    axis=1,
                ),
                axis=1,
            )
        else:
            out[:, ri] /= den

    idx = np.argmax(out, axis=1)
    pred = np.asarray([REL[int(i)] for i in idx], dtype=object)

    s = np.sort(out, axis=1)
    margin = s[:, -1] - s[:, -2]

    return pred, margin


def evaluate(pred, y):
    pred = np.asarray(pred, dtype=object)
    y = np.asarray(y, dtype=object)
    correct = pred == y

    row = {
        "N": len(y),
        "acc_all": float(correct.mean()),
    }

    vals = []
    for r in REL:
        m = y == r
        v = float(correct[m].mean())
        row[f"acc_{r}"] = v
        row[f"N_{r}"] = int(m.sum())
        vals.append(v)

    row["macro_acc"] = float(np.mean(vals))
    row["min_relation_acc"] = float(np.min(vals))

    return row, correct


def choose_hierarchical_k_from_source(
    oof_scores,
    y,
    task_ranked,
    task_acc,
    ks,
    temperature,
    weighted,
):
    rows = []

    for ka in ks:
        ah = select_top(task_ranked["axis"], ka)

        for klr in ks:
            lh = select_top(task_ranked["lr"], klr)

            for kab in ks:
                bh = select_top(task_ranked["ab"], kab)

                pred, *_ = hierarchical_predict(
                    oof_scores,
                    ah,
                    lh,
                    bh,
                    task_acc,
                    temperature,
                    weighted,
                )
                met, _ = evaluate(pred, y)

                rows.append({
                    "weighted": weighted,
                    "K_axis": ka,
                    "K_LR": klr,
                    "K_AB": kab,
                    **met,
                })

    df = pd.DataFrame(rows).sort_values(
        ["acc_all", "min_relation_acc", "K_axis", "K_LR", "K_AB"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    return df


def main():
    a = parse_args()

    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ks = parse_ints(a.task_topks)
    candidate_layers = (
        parse_ints(a.candidate_layers)
        if a.candidate_layers
        else None
    )

    source_path, source_sid, source_y, source_X = load_cache(a.source_cache)
    target_path, target_sid, target_y, target_X = load_cache(
        a.target_direction_dir
    )

    if source_X.shape[1:] != target_X.shape[1:]:
        raise RuntimeError(
            f"Geometry mismatch:\nsource={source_X.shape}\ntarget={target_X.shape}"
        )

    print("=" * 170)
    print("PURE SOURCE-ONLY DIRECTION-HEAD SELECTOR TEST")
    print("=" * 170)
    print(
        f"source: {source_path} | N={len(source_y)} "
        f"| counts={dict(Counter(source_y.tolist()))}"
    )
    print(
        f"target: {target_path} | N={len(target_y)} "
        f"| counts={dict(Counter(target_y.tolist()))}"
    )
    print(f"shape source={source_X.shape} target={target_X.shape}")
    print(f"CV folds={a.cv_folds} seed={a.cv_seed}")
    print(f"K candidates={ks}")
    print(f"candidate layers={candidate_layers if candidate_layers else 'ALL'}")
    print("No model loading. COCO GT is evaluation only.")
    print()

    # ------------------------------------------------------------
    # 1. Synthetic OOF scores.
    # ------------------------------------------------------------
    print("[1/3] Synthetic OOF scoring...", flush=True)

    oof_scores = source_oof_scores(
        source_X,
        source_y,
        a.cv_folds,
        a.cv_seed,
    )

    four_acc = fourway_head_acc(oof_scores, source_y)
    task_acc = binary_task_head_acc(
        oof_scores,
        source_y,
        a.temperature,
    )
    rel_acc = relation_specific_balanced_acc(
        oof_scores,
        source_y,
    )

    ranked4 = rank_heads(four_acc, candidate_layers)
    task_ranked = {
        t: rank_heads(task_acc[t], candidate_layers)
        for t in ("axis", "lr", "ab")
    }
    rel_ranked = {
        r: rank_heads(rel_acc[ri], candidate_layers)
        for ri, r in enumerate(REL)
    }

    # Save task-specific rankings.
    rank_rows = []

    for task in ("axis", "lr", "ab"):
        for rank, (acc, l, h) in enumerate(task_ranked[task], 1):
            rank_rows.append({
                "task": task,
                "rank": rank,
                "layer": l,
                "head": h,
                "head_name": head_name(l, h),
                "source_oof_accuracy": acc,
            })

    for ri, r in enumerate(REL):
        for rank, (acc, l, h) in enumerate(rel_ranked[r], 1):
            rank_rows.append({
                "task": f"ovr_{r}",
                "rank": rank,
                "layer": l,
                "head": h,
                "head_name": head_name(l, h),
                "source_oof_accuracy": acc,
            })

    pd.DataFrame(rank_rows).to_csv(
        outdir / "source_oof_task_head_rankings.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 2. Select hierarchical K only on Synthetic OOF.
    # ------------------------------------------------------------
    print("[2/3] Source-only K search...", flush=True)

    grid_equal = choose_hierarchical_k_from_source(
        oof_scores,
        source_y,
        task_ranked,
        task_acc,
        ks,
        a.temperature,
        weighted=False,
    )

    grid_weighted = choose_hierarchical_k_from_source(
        oof_scores,
        source_y,
        task_ranked,
        task_acc,
        ks,
        a.temperature,
        weighted=True,
    )

    grid_equal.assign(weighting="equal").to_csv(
        outdir / "source_hier_grid_equal.csv",
        index=False,
    )
    grid_weighted.assign(weighting="weighted").to_csv(
        outdir / "source_hier_grid_weighted.csv",
        index=False,
    )

    best_eq = grid_equal.iloc[0]
    best_wt = grid_weighted.iloc[0]

    # Relation-specific K selection on source OOF.
    rel_best = {}

    for weighted in (False, True):
        candidates = []

        for k in ks:
            pools = {
                r: select_top(rel_ranked[r], k)
                for r in REL
            }

            pred, _ = relation_specific_predict(
                oof_scores,
                pools,
                rel_acc,
                a.temperature,
                weighted,
            )

            met, _ = evaluate(pred, source_y)
            candidates.append({
                "K": k,
                "weighted": weighted,
                **met,
            })

        d = pd.DataFrame(candidates).sort_values(
            ["acc_all", "min_relation_acc", "K"],
            ascending=[False, False, True],
        )

        name = "weighted" if weighted else "equal"
        d.to_csv(
            outdir / f"source_relspecific_k_{name}.csv",
            index=False,
        )
        rel_best[name] = d.iloc[0]

    # ------------------------------------------------------------
    # 3. Fit ALL Synthetic, freeze, evaluate COCO once.
    # ------------------------------------------------------------
    print("[3/3] Frozen Synthetic -> COCO evaluation...", flush=True)

    center, dirs = fit_codebook(source_X, source_y)
    target_scores = score_codebook(target_X, center, dirs)

    results = []
    detail = pd.DataFrame({
        "sid": target_sid,
        "gt": target_y,
    })

    def add_method(name, pred, margin, extra=None):
        met, correct = evaluate(pred, target_y)

        row = {
            "method": name,
            **met,
        }

        if extra:
            row.update(extra)

        results.append(row)

        detail[f"pred_{name}"] = pred
        detail[f"correct_{name}"] = correct
        detail[f"margin_{name}"] = margin

    # ---- original v2 Top10 sanity check ----
    top10 = select_top(ranked4, 10)

    pred, margin = fourway_predict(
        target_scores,
        top10,
        four_acc,
        a.temperature,
        weighted=False,
    )

    add_method(
        "fourway_top10_equal_repro",
        pred,
        margin,
        {
            "heads": ",".join(head_name(l, h) for l, h in top10)
        },
    )

    # ---- hierarchical source-selected champions ----
    for label, best, weighted in [
        ("hier_source_best_equal", best_eq, False),
        ("hier_source_best_weighted", best_wt, True),
    ]:
        ka = int(best["K_axis"])
        klr = int(best["K_LR"])
        kab = int(best["K_AB"])

        ah = select_top(task_ranked["axis"], ka)
        lh = select_top(task_ranked["lr"], klr)
        bh = select_top(task_ranked["ab"], kab)

        pred, axis_s, lr_s, ab_s, margin = hierarchical_predict(
            target_scores,
            ah,
            lh,
            bh,
            task_acc,
            a.temperature,
            weighted,
        )

        add_method(
            label,
            pred,
            margin,
            {
                "source_oof_acc": float(best["acc_all"]),
                "K_axis": ka,
                "K_LR": klr,
                "K_AB": kab,
                "axis_heads": ",".join(head_name(l, h) for l, h in ah),
                "lr_heads": ",".join(head_name(l, h) for l, h in lh),
                "ab_heads": ",".join(head_name(l, h) for l, h in bh),
            },
        )

        detail[f"axis_score_{label}"] = axis_s
        detail[f"lr_score_{label}"] = lr_s
        detail[f"ab_score_{label}"] = ab_s

    # ---- hierarchical fixed-K diagnostics ----
    for k in ks:
        ah = select_top(task_ranked["axis"], k)
        lh = select_top(task_ranked["lr"], k)
        bh = select_top(task_ranked["ab"], k)

        pred, axis_s, lr_s, ab_s, margin = hierarchical_predict(
            target_scores,
            ah,
            lh,
            bh,
            task_acc,
            a.temperature,
            weighted=False,
        )

        add_method(
            f"hier_fixed{k}_equal",
            pred,
            margin,
            {
                "K_axis": k,
                "K_LR": k,
                "K_AB": k,
            },
        )

    # ---- relation-specific source-selected champions ----
    for name, weighted in [
        ("equal", False),
        ("weighted", True),
    ]:
        best = rel_best[name]
        k = int(best["K"])

        pools = {
            r: select_top(rel_ranked[r], k)
            for r in REL
        }

        pred, margin = relation_specific_predict(
            target_scores,
            pools,
            rel_acc,
            a.temperature,
            weighted,
        )

        add_method(
            f"relspecific_source_best_{name}",
            pred,
            margin,
            {
                "source_oof_acc": float(best["acc_all"]),
                "K_relation_each": k,
                "left_heads": ",".join(
                    head_name(l, h) for l, h in pools["left"]
                ),
                "right_heads": ",".join(
                    head_name(l, h) for l, h in pools["right"]
                ),
                "above_heads": ",".join(
                    head_name(l, h) for l, h in pools["above"]
                ),
                "below_heads": ",".join(
                    head_name(l, h) for l, h in pools["below"]
                ),
            },
        )

    results = pd.DataFrame(results).sort_values(
        ["acc_all", "min_relation_acc"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results.to_csv(
        outdir / "coco440_selector_results_v4.csv",
        index=False,
    )
    detail.to_csv(
        outdir / "coco440_predictions_v4.csv",
        index=False,
    )

    np.savez_compressed(
        outdir / "synthetic_fitted_codebook_v4.npz",
        center=center,
        directions=dirs,
        relations=np.asarray(REL, dtype=object),
    )

    metadata = {
        "source_cache": str(source_path),
        "target_cache": str(target_path),
        "source_N": int(len(source_y)),
        "target_N": int(len(target_y)),
        "cv_folds": int(a.cv_folds),
        "cv_seed": int(a.cv_seed),
        "task_topks": ks,
        "candidate_layers": candidate_layers,
        "target_gt_usage": "evaluation only",
        "model_forward": False,
        "causal_intervention": False,
    }

    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------
    print()
    print("=" * 150)
    print("SOURCE OOF TOP HEADS BY TASK")
    print("=" * 150)

    for task in ("axis", "lr", "ab"):
        print(f"\n[{task}]")
        for rank, (acc, l, h) in enumerate(task_ranked[task][:10], 1):
            print(
                f"{rank:2d}  {head_name(l,h):8s}  {acc:.4f}"
            )

    print()
    print("=" * 150)
    print("SOURCE-ONLY SELECTED HIERARCHICAL CONFIG")
    print("=" * 150)

    print(
        f"equal    : K_axis={int(best_eq['K_axis'])} "
        f"K_LR={int(best_eq['K_LR'])} "
        f"K_AB={int(best_eq['K_AB'])} "
        f"source_OOF={best_eq['acc_all']:.4f}"
    )

    print(
        f"weighted : K_axis={int(best_wt['K_axis'])} "
        f"K_LR={int(best_wt['K_LR'])} "
        f"K_AB={int(best_wt['K_AB'])} "
        f"source_OOF={best_wt['acc_all']:.4f}"
    )

    print()
    print("=" * 170)
    print("FROZEN SYNTHETIC-400 -> COCO-440 SELECTOR RESULTS")
    print("=" * 170)

    cols = [
        "method",
        "acc_all",
        "acc_left",
        "acc_right",
        "acc_above",
        "acc_below",
        "min_relation_acc",
    ]

    print(
        results[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Sanity check:")
    sanity = results[
        results["method"] == "fourway_top10_equal_repro"
    ]

    if len(sanity):
        val = float(sanity.iloc[0]["acc_all"])
        print(
            f"  fourway_top10_equal_repro = {val:.4f} "
            f"(expected around previous 0.8205)"
        )

    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
