#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test whether the bbox-derived hidden-space spatial axes are genuinely CONTINUOUS,
rather than merely separating categorical left/right or above/below labels.

This script reuses:
  1) raw-question A/B/last states from
       <state-dir>/raw__correct__all_layers.npz
       <state-dir>/raw__no_image__all_layers.npz
  2) GroundingDINO boxes from
       bboxes_by_sid.jsonl
  3) helper functions from
       analyze_coco_raw_token_gdino_linear_encoding_v1.py

Main tests
----------
For each target (A_residual, B_residual, last_residual):

A. Global bbox axis
   Fit on TRAIN only:
       Y ~ pair_center + dx + dy + size_A + size_B
   extracting beta_dx [D] and beta_dy [D].

B. Held-out projection continuity
   Project TEST hidden states onto beta_dx / beta_dy and test:
       corr(proj_x, dx)
       corr(proj_y, dy)
   plus the critical within-category tests:
       corr(proj_x, dx | left)
       corr(proj_x, dx | right)
       corr(proj_y, dy | above)
       corr(proj_y, dy | below)

   Also pool left+right after subtracting each class's TRAIN mean. This removes the
   between-class left/right offset before measuring continuity.

C. Within-category axis fit (stronger test)
   Restrict to left/right, subtract TRAIN class means from BOTH geometry and hidden
   states, then fit a new beta_dx_within using only within-left/right variation.
   Thus the model cannot use the left-vs-right mean difference to learn this axis.
   Analogously for above/below and beta_dy_within.

   Report:
       unique DeltaR2 of dx (or dy) after class-centering + other bbox controls
       cos(beta_dx_within, beta_dx_global)
       cos(beta_dx_within, right-left categorical axis)
       split stability of beta_dx_within

Interpretation
--------------
Strong evidence for a continuous internal spatial axis requires, ideally:
  * positive within-category correlations on held-out samples;
  * positive unique within-category DeltaR2 for dx/dy;
  * stable within-category beta across splits;
  * beta_within aligned with the global bbox axis.

This remains correlational because the images are natural COCO images. A causal
claim still requires controlled image-position interventions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import analyze_coco_raw_token_gdino_linear_encoding_v1 as base
except Exception as exc:
    raise RuntimeError(
        "This script expects analyze_coco_raw_token_gdino_linear_encoding_v1.py "
        "in the same repository/PYTHONPATH. Put both scripts in AdaptVis root."
    ) from exc

EPS = 1e-12
HORIZONTAL = ("left", "right")
VERTICAL = ("above", "below")
TARGETS = ("A_residual", "B_residual", "last_residual")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--state-dir", required=True)
    p.add_argument(
        "--bbox-jsonl",
        default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fixed-layer", type=int, default=25)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument(
        "--require-gt-consistent",
        action="store_true",
        help="Sensitivity analysis only. Primary analysis should leave this OFF.",
    )
    p.add_argument("--bins", type=int, default=4, help="Quantile bins per relation for diagnostic monotonicity tables.")
    p.add_argument(
        "--include-raw",
        action="store_true",
        help="Also analyze A_raw/B_raw/last_raw in addition to Image-NoImage residuals.",
    )
    return p.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean(); y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den < EPS:
        return float("nan")
    return float(np.dot(x, y) / den)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson(rx, ry)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def mean_pairwise_cos(vecs: Sequence[np.ndarray]) -> float:
    vals: List[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            vals.append(cosine(vecs[i], vecs[j]))
    return float(np.mean(vals)) if vals else float("nan")


def global_r2_zero(pred: np.ndarray, target: np.ndarray) -> float:
    """R2 around zero after class-centering; zero is the appropriate baseline."""
    sse = float(np.sum((target - pred) ** 2))
    sst = float(np.sum(target ** 2))
    return 1.0 - sse / max(sst, EPS)


def fit_ridge_centered(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    ridge: float,
    feature_idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs are already class-centered when used for within-category analysis.
    Returns pred_te, W, xmu, xsd.
    """
    idx = np.asarray(list(feature_idx), dtype=np.int64)
    if len(idx) == 0:
        return np.zeros((len(Xte), Ytr.shape[1]), dtype=np.float64), np.zeros((0, Ytr.shape[1])), np.zeros(0), np.ones(0)
    a = Xtr[:, idx]
    b = Xte[:, idx]
    xmu = a.mean(axis=0)
    xsd = a.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    a = (a - xmu) / xsd
    b = (b - xmu) / xsd
    gram = a.T @ a
    lam = float(ridge) * float(np.trace(gram) / max(len(idx), 1))
    W = np.linalg.solve(gram + lam * np.eye(len(idx)), a.T @ Ytr)
    return b @ W, W, xmu, xsd


def class_center_from_train(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    classes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict to classes, then subtract class-specific TRAIN means from X and Y.

    Output shapes:
      Xtr_c [Ntr_axis, P]
      Ytr_c [Ntr_axis, D]
      Xte_c [Nte_axis, P]
      Yte_c [Nte_axis, D]
      tr_axis [Ntr_axis]
      te_axis [Nte_axis]
    """
    classes = tuple(classes)
    tr_axis = np.asarray([i for i in tr if labels[i] in classes], dtype=np.int64)
    te_axis = np.asarray([i for i in te if labels[i] in classes], dtype=np.int64)
    if len(tr_axis) < 10 or len(te_axis) < 5:
        raise ValueError(f"Too few axis samples: train={len(tr_axis)}, test={len(te_axis)}, classes={classes}")

    Xtr_c = np.empty_like(X[tr_axis], dtype=np.float64)
    Ytr_c = np.empty_like(Y[tr_axis], dtype=np.float64)
    Xte_c = np.empty_like(X[te_axis], dtype=np.float64)
    Yte_c = np.empty_like(Y[te_axis], dtype=np.float64)

    for c in classes:
        tr_pos = np.where(labels[tr_axis] == c)[0]
        te_pos = np.where(labels[te_axis] == c)[0]
        if len(tr_pos) < 3 or len(te_pos) < 2:
            raise ValueError(f"Too few samples for class {c}: train={len(tr_pos)}, test={len(te_pos)}")
        xmu = X[tr_axis[tr_pos]].mean(axis=0, keepdims=True)
        ymu = Y[tr_axis[tr_pos]].mean(axis=0, keepdims=True)
        Xtr_c[tr_pos] = X[tr_axis[tr_pos]] - xmu
        Ytr_c[tr_pos] = Y[tr_axis[tr_pos]] - ymu
        Xte_c[te_pos] = X[te_axis[te_pos]] - xmu
        Yte_c[te_pos] = Y[te_axis[te_pos]] - ymu

    return Xtr_c, Ytr_c, Xte_c, Yte_c, tr_axis, te_axis


def train_class_projection_means(
    Y: np.ndarray,
    labels: np.ndarray,
    tr: np.ndarray,
    beta: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, float]:
    b = beta / max(float(np.linalg.norm(beta)), EPS)
    out: Dict[str, float] = {}
    for c in classes:
        idx = tr[labels[tr] == c]
        out[c] = float(np.mean(Y[idx] @ b))
    return out


def train_class_scalar_means(
    v: np.ndarray,
    labels: np.ndarray,
    tr: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, float]:
    return {c: float(np.mean(v[tr[labels[tr] == c]])) for c in classes}


def relation_centered_vectors(
    values: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    train_means: Mapping[str, float],
) -> np.ndarray:
    return np.asarray([float(values[i]) - float(train_means[str(labels[i])]) for i in idx], dtype=np.float64)


def correlation_row(
    target: str,
    split: int,
    axis: str,
    cohort: str,
    x: np.ndarray,
    p: np.ndarray,
) -> Dict[str, Any]:
    return {
        "target": target,
        "split": int(split),
        "axis": axis,
        "cohort": cohort,
        "n": int(len(x)),
        "pearson": pearson(p, x),
        "spearman": spearman(p, x),
    }


def make_bins(
    target: str,
    split: int,
    relation: str,
    axis: str,
    geom: np.ndarray,
    proj: np.ndarray,
    n_bins: int,
) -> List[Dict[str, Any]]:
    if len(geom) < max(8, n_bins * 2):
        return []
    # Rank-based bins are robust to duplicated distances.
    ranks = pd.Series(geom).rank(method="first", pct=True).to_numpy()
    b = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    rows = []
    for bi in range(n_bins):
        m = b == bi
        if not np.any(m):
            continue
        rows.append({
            "target": target,
            "split": int(split),
            "relation": relation,
            "axis": axis,
            "bin": int(bi),
            "n": int(np.sum(m)),
            "mean_geometry": float(np.mean(geom[m])),
            "mean_projection": float(np.mean(proj[m])),
        })
    return rows


def analyze_target(
    target_name: str,
    Y: np.ndarray,
    Xg: np.ndarray,
    labels: np.ndarray,
    splits,
    args: argparse.Namespace,
):
    corr_rows: List[Dict[str, Any]] = []
    within_fit_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    global_dx_vecs: List[np.ndarray] = []
    global_dy_vecs: List[np.ndarray] = []
    within_dx_vecs: List[np.ndarray] = []
    within_dy_vecs: List[np.ndarray] = []

    all_idx = list(range(Xg.shape[1]))
    dx_i = base.GEOM_FEATURES.index("dx")
    dy_i = base.GEOM_FEATURES.index("dy")

    for si, (tr, te) in enumerate(splits):
        # --------------------------------------------------------------
        # 1) Global geometry fit on train only.
        # --------------------------------------------------------------
        gf = base.ridge_fit_predict(Xg, Y, tr, te, args.ridge, all_idx)
        W = np.asarray(gf["W"], dtype=np.float64)
        beta_dx = W[dx_i]
        beta_dy = W[dy_i]
        bx = beta_dx / max(float(np.linalg.norm(beta_dx)), EPS)
        by = beta_dy / max(float(np.linalg.norm(beta_dy)), EPS)
        global_dx_vecs.append(bx)
        global_dy_vecs.append(by)

        # Project raw hidden states. Pearson is translation invariant; for pooled
        # within-class tests we explicitly subtract TRAIN class means below.
        sx_all = Y @ bx
        sy_all = Y @ by
        dx = Xg[:, dx_i]
        dy = Xg[:, dy_i]

        corr_rows.append(correlation_row(target_name, si, "x", "all_test", dx[te], sx_all[te]))
        corr_rows.append(correlation_row(target_name, si, "y", "all_test", dy[te], sy_all[te]))
        # Cross-axis negative controls.
        corr_rows.append(correlation_row(target_name, si, "x_vs_dy_control", "all_test", dy[te], sx_all[te]))
        corr_rows.append(correlation_row(target_name, si, "y_vs_dx_control", "all_test", dx[te], sy_all[te]))

        # Per-relation correlations: label is fixed, only magnitude/within-class
        # geometry varies.
        for rel in HORIZONTAL:
            idx = te[labels[te] == rel]
            corr_rows.append(correlation_row(target_name, si, "x", rel, dx[idx], sx_all[idx]))
            bin_rows.extend(make_bins(target_name, si, rel, "x", dx[idx], sx_all[idx], args.bins))
        for rel in VERTICAL:
            idx = te[labels[te] == rel]
            corr_rows.append(correlation_row(target_name, si, "y", rel, dy[idx], sy_all[idx]))
            bin_rows.extend(make_bins(target_name, si, rel, "y", dy[idx], sy_all[idx], args.bins))

        # Stronger pooled within-class correlation: remove each relation's TRAIN
        # mean from both geometry and hidden projection first.
        hte = te[np.isin(labels[te], HORIZONTAL)]
        vte = te[np.isin(labels[te], VERTICAL)]
        sx_mu = train_class_projection_means(Y, labels, tr, beta_dx, HORIZONTAL)
        dx_mu = train_class_scalar_means(dx, labels, tr, HORIZONTAL)
        sy_mu = train_class_projection_means(Y, labels, tr, beta_dy, VERTICAL)
        dy_mu = train_class_scalar_means(dy, labels, tr, VERTICAL)
        sx_c = relation_centered_vectors(sx_all, labels, hte, sx_mu)
        dx_c = relation_centered_vectors(dx, labels, hte, dx_mu)
        sy_c = relation_centered_vectors(sy_all, labels, vte, sy_mu)
        dy_c = relation_centered_vectors(dy, labels, vte, dy_mu)
        corr_rows.append(correlation_row(target_name, si, "x", "left_right_relation_centered", dx_c, sx_c))
        corr_rows.append(correlation_row(target_name, si, "y", "above_below_relation_centered", dy_c, sy_c))

        # --------------------------------------------------------------
        # 2) Stronger within-category axis fit.
        #    Remove left/right mean BEFORE fitting beta_dx_within.
        # --------------------------------------------------------------
        # Horizontal
        Xtrc, Ytrc, Xtec, Ytec, htr, hte2 = class_center_from_train(
            Xg, Y, labels, tr, te, HORIZONTAL
        )
        pred_full, Wh, _, _ = fit_ridge_centered(Xtrc, Ytrc, Xtec, args.ridge, all_idx)
        pred_nodx, _, _, _ = fit_ridge_centered(
            Xtrc, Ytrc, Xtec, args.ridge, [i for i in all_idx if i != dx_i]
        )
        r2_full = global_r2_zero(pred_full, Ytec)
        r2_nodx = global_r2_zero(pred_nodx, Ytec)
        beta_dx_within = Wh[dx_i]
        bdxw = beta_dx_within / max(float(np.linalg.norm(beta_dx_within)), EPS)
        within_dx_vecs.append(bdxw)
        proj_w = Ytec @ bdxw
        geom_w = Xtec[:, dx_i]  # already class-centered using train class mean

        # Old categorical right-left axis, train only.
        d_rl = Y[tr][labels[tr] == "right"].mean(axis=0) - Y[tr][labels[tr] == "left"].mean(axis=0)
        within_fit_rows.append({
            "target": target_name,
            "split": si,
            "axis": "x",
            "train_n": int(len(htr)),
            "test_n": int(len(hte2)),
            "within_full_r2": r2_full,
            "within_without_axis_r2": r2_nodx,
            "axis_unique_delta_r2": r2_full - r2_nodx,
            "within_projection_pearson": pearson(proj_w, geom_w),
            "within_projection_spearman": spearman(proj_w, geom_w),
            "cos_within_vs_global_beta": cosine(beta_dx_within, beta_dx),
            "cos_within_vs_categorical_axis": cosine(beta_dx_within, d_rl),
        })

        # Vertical
        Xtrc, Ytrc, Xtec, Ytec, vtr, vte2 = class_center_from_train(
            Xg, Y, labels, tr, te, VERTICAL
        )
        pred_full, Wv, _, _ = fit_ridge_centered(Xtrc, Ytrc, Xtec, args.ridge, all_idx)
        pred_nody, _, _, _ = fit_ridge_centered(
            Xtrc, Ytrc, Xtec, args.ridge, [i for i in all_idx if i != dy_i]
        )
        r2_full = global_r2_zero(pred_full, Ytec)
        r2_nody = global_r2_zero(pred_nody, Ytec)
        beta_dy_within = Wv[dy_i]
        bdyw = beta_dy_within / max(float(np.linalg.norm(beta_dy_within)), EPS)
        within_dy_vecs.append(bdyw)
        proj_w = Ytec @ bdyw
        geom_w = Xtec[:, dy_i]
        d_ba = Y[tr][labels[tr] == "below"].mean(axis=0) - Y[tr][labels[tr] == "above"].mean(axis=0)
        within_fit_rows.append({
            "target": target_name,
            "split": si,
            "axis": "y",
            "train_n": int(len(vtr)),
            "test_n": int(len(vte2)),
            "within_full_r2": r2_full,
            "within_without_axis_r2": r2_nody,
            "axis_unique_delta_r2": r2_full - r2_nody,
            "within_projection_pearson": pearson(proj_w, geom_w),
            "within_projection_spearman": spearman(proj_w, geom_w),
            "cos_within_vs_global_beta": cosine(beta_dy_within, beta_dy),
            "cos_within_vs_categorical_axis": cosine(beta_dy_within, d_ba),
        })

    stability = {
        "target": target_name,
        "global_beta_dx_stability": mean_pairwise_cos(global_dx_vecs),
        "global_beta_dy_stability": mean_pairwise_cos(global_dy_vecs),
        "within_beta_dx_stability": mean_pairwise_cos(within_dx_vecs),
        "within_beta_dy_stability": mean_pairwise_cos(within_dy_vecs),
    }
    return corr_rows, within_fit_rows, bin_rows, stability


def mean_std(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    if len(df) == 0:
        return float("nan"), float("nan")
    return float(df[col].mean()), float(df[col].std())


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.repeats < 2:
        raise ValueError("Use --repeats >= 2 so axis stability is meaningful")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_dir = Path(args.state_dir)
    correct = base.load_npz(state_dir / "raw__correct__all_layers.npz")
    noimg = base.load_npz(state_dir / "raw__no_image__all_layers.npz")
    sids, states, relation, subject, reference, image_id, layers, layer_to_idx, meta = base.align_states(correct, noimg)

    valid = np.asarray([r in base.RELATIONS for r in relation], dtype=bool)
    sids = sids[valid]
    relation = relation[valid]
    subject = subject[valid]
    reference = reference[valid]
    image_id = image_id[valid]
    states = {k: v[valid] for k, v in states.items()}

    # Reuse bbox filtering logic from v1.
    bbox_args = argparse.Namespace(
        include_ambiguous=args.include_ambiguous,
        min_score=args.min_score,
        require_gt_consistent=args.require_gt_consistent,
    )
    gdino = base.load_gdino_rows(Path(args.bbox_jsonl))
    geom_df, audit = base.build_geometry_table(
        sids, relation, subject, reference, image_id, gdino, bbox_args
    )
    if len(geom_df) < 40:
        raise RuntimeError(f"Only {len(geom_df)} usable bbox samples")

    sid_to_state = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray([sid_to_state[int(s)] for s in geom_df["sid"].tolist()], dtype=np.int64)
    y = relation[ridx]
    geom_df["relation"] = y
    geom_df["subject"] = subject[ridx]
    geom_df["reference"] = reference[ridx]
    states_keep = {k: v[ridx] for k, v in states.items()}

    if int(args.fixed_layer) not in layer_to_idx:
        raise ValueError(f"L{args.fixed_layer} unavailable; layers={layers.tolist()}")
    li = layer_to_idx[int(args.fixed_layer)]
    Xg = geom_df[list(base.GEOM_FEATURES)].to_numpy(dtype=np.float64)
    splits = base.make_stratified_splits(y, args.train_ratio, args.repeats, args.seed)

    target_names = list(TARGETS)
    if args.include_raw:
        target_names += ["A_raw", "B_raw", "last_raw"]

    all_corr: List[Dict[str, Any]] = []
    all_within: List[Dict[str, Any]] = []
    all_bins: List[Dict[str, Any]] = []
    stabs: List[Dict[str, Any]] = []

    for target_name in target_names:
        Y = np.asarray(states_keep[target_name][:, li], dtype=np.float64)
        c, w, b, s = analyze_target(target_name, Y, Xg, y, splits, args)
        all_corr.extend(c)
        all_within.extend(w)
        all_bins.extend(b)
        stabs.append(s)

    corr_df = pd.DataFrame(all_corr)
    within_df = pd.DataFrame(all_within)
    bins_df = pd.DataFrame(all_bins)
    stab_df = pd.DataFrame(stabs)

    corr_df.to_csv(out / "heldout_projection_continuity.csv", index=False)
    within_df.to_csv(out / "within_category_axis_fit.csv", index=False)
    bins_df.to_csv(out / "within_relation_bins.csv", index=False)
    stab_df.to_csv(out / "axis_stability.csv", index=False)
    geom_df.to_csv(out / "gdino_geometry_used.csv", index=False)

    # Compact aggregate tables.
    corr_summary = (
        corr_df.groupby(["target", "axis", "cohort"], as_index=False)
        .agg(
            n_mean=("n", "mean"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
    )
    within_summary = (
        within_df.groupby(["target", "axis"], as_index=False)
        .agg(
            unique_delta_r2_mean=("axis_unique_delta_r2", "mean"),
            unique_delta_r2_std=("axis_unique_delta_r2", "std"),
            projection_pearson_mean=("within_projection_pearson", "mean"),
            projection_pearson_std=("within_projection_pearson", "std"),
            projection_spearman_mean=("within_projection_spearman", "mean"),
            cos_with_global_mean=("cos_within_vs_global_beta", "mean"),
            cos_with_categorical_mean=("cos_within_vs_categorical_axis", "mean"),
        )
    )
    corr_summary.to_csv(out / "heldout_projection_continuity_summary.csv", index=False)
    within_summary.to_csv(out / "within_category_axis_fit_summary.csv", index=False)

    print("\n" + "=" * 120)
    print("CONTINUOUS SPATIAL AXIS TEST")
    print("critical question: after LEFT/RIGHT (or ABOVE/BELOW) is fixed/removed, does hidden projection still track dx/dy?")
    print("=" * 120)
    print(f"usable bbox samples={len(geom_df)} | layer=L{args.fixed_layer} | repeats={args.repeats} | train_ratio={args.train_ratio:.2f}")
    print(f"bbox relation sign consistency={100*float(geom_df['bbox_relation_consistent'].mean()):.2f}%")

    for target in TARGETS:
        print(f"\n{target}")
        # Per-category held-out correlation using global beta.
        for axis, cohorts in (
            ("x", ["left", "right", "left_right_relation_centered"]),
            ("y", ["above", "below", "above_below_relation_centered"]),
        ):
            print(f"  axis {axis} -- global beta, held-out continuity")
            for cohort in cohorts:
                d = corr_summary[(corr_summary.target == target) & (corr_summary.axis == axis) & (corr_summary.cohort == cohort)]
                if len(d):
                    r = d.iloc[0]
                    print(
                        f"    {cohort:29s} | Pearson={r.pearson_mean:+.4f}±{r.pearson_std:.4f} "
                        f"| Spearman={r.spearman_mean:+.4f}"
                    )

        print("  stronger within-category fit (class means removed BEFORE learning beta)")
        for axis in ("x", "y"):
            d = within_summary[(within_summary.target == target) & (within_summary.axis == axis)]
            if len(d):
                r = d.iloc[0]
                stab_col = "within_beta_dx_stability" if axis == "x" else "within_beta_dy_stability"
                st = stab_df[stab_df.target == target].iloc[0]
                print(
                    f"    {axis}: unique ΔR2={r.unique_delta_r2_mean:+.4f}±{r.unique_delta_r2_std:.4f} "
                    f"| proj↔geometry r={r.projection_pearson_mean:+.4f} "
                    f"| cos(within,global)={r.cos_with_global_mean:+.4f} "
                    f"| cos(within,categorical)={r.cos_with_categorical_mean:+.4f} "
                    f"| stability={float(st[stab_col]):+.4f}"
                )

    summary = {
        "script": "analyze_coco_gdino_continuous_spatial_axis_v1.py",
        "fixed_layer": int(args.fixed_layer),
        "n_usable": int(len(geom_df)),
        "train_ratio": float(args.train_ratio),
        "repeats": int(args.repeats),
        "ridge": float(args.ridge),
        "bbox_sign_consistency": float(geom_df["bbox_relation_consistent"].mean()),
        "corr_summary": corr_summary.to_dict(orient="records"),
        "within_summary": within_summary.to_dict(orient="records"),
        "stability": stab_df.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    for fn in (
        "heldout_projection_continuity.csv",
        "heldout_projection_continuity_summary.csv",
        "within_category_axis_fit.csv",
        "within_category_axis_fit_summary.csv",
        "within_relation_bins.csv",
        "axis_stability.csv",
        "gdino_geometry_used.csv",
        "summary.json",
    ):
        print(f"  {out / fn}")


if __name__ == "__main__":
    main()
