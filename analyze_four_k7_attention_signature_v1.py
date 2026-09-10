#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K7 ATTENTION-SIGNATURE ANALYSIS
===============================

Post-process the four writer-specific K7 candidates for each sample and ask:

1) Correct vs wrong:
   Does the TRUE/GT K7 have a different *attention organization* when the
   baseline answer is correct vs wrong?

2) Wrong samples:
   For a sample whose baseline answer is wrong, is the baseline-WRONG K7
   more strongly/cleanly utilized than the GT K7?

3) Four-way K7 selection:
   Can a relation-free endogenous attention/activity signature choose among
   K7_left / K7_right / K7_on / K7_under without using GT at test time?

Unlike the previous script, this does NOT collapse each K7 to only one mean.
It explicitly analyzes:
  - layer trajectory / layer concentration
  - within-K7 attention concentration (std, entropy, top1/top2 share, Gini)
  - head-level attention pattern (if head-scan columns exist)
  - exact-state vs position-level utilization
  - statistically strict correct-vs-wrong differences:
      relation-stratified permutation p + Benjamini-Hochberg FDR q
  - a cross-fitted "normal-correct K7 attention prototype" diagnostic

No VLM forward is required if you already have:
  FOURWAY_DIR/selected_topk_all_directions.csv
  FEATURE_DIR/all_candidate_self_features.pkl.gz

Recommended:
python analyze_four_k7_attention_signature_v1.py \
  --fourway-dir output/qwen3b_four_writer_k7_N80 \
  --feature-dir output/qwen3b_coco_L20_26_self_predict_core \
  --selection-type raw \
  --perm 10000 \
  --folds 5 \
  --output-dir output/qwen3b_four_k7_attention_signature_N80
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REL_ORDER = ["left", "right", "on", "under"]
EPS = 1e-12


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--selection-type", default="raw",
                   choices=["raw", "contrast"])
    p.add_argument("--perm", type=int, default=10000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower()
    m = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under",
        "bottom": "under",
    }
    return m.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_mean(x):
    a = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x):
    a = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def safe_div(a, b):
    return float(a / b) if abs(float(b)) > EPS else float("nan")


def entropy01(vals):
    """Entropy normalized to [0,1] for nonnegative mass."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)
    if len(x) <= 1 or x.sum() <= EPS:
        return 0.0
    p = x / x.sum()
    h = -float(np.sum(p * np.log(p + EPS)))
    return float(h / np.log(len(p)))


def gini_nonnegative(vals):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)
    if len(x) == 0 or x.sum() <= EPS:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float(
        (2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum()))
        - (n + 1) / n
    )


def concentration_stats(vals, prefix):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    xp = np.clip(x, 0.0, None)
    total = float(xp.sum())
    sx = np.sort(xp)[::-1]
    return {
        f"{prefix}__mean": float(x.mean()),
        f"{prefix}__std": float(x.std(ddof=0)),
        f"{prefix}__max": float(x.max()),
        f"{prefix}__min": float(x.min()),
        f"{prefix}__range": float(x.max() - x.min()),
        f"{prefix}__entropy": entropy01(xp),
        f"{prefix}__gini": gini_nonnegative(xp),
        f"{prefix}__top1_share": safe_div(sx[:1].sum(), total) if total > EPS else 0.0,
        f"{prefix}__top2_share": safe_div(sx[:2].sum(), total) if total > EPS else 0.0,
    }


def percentile_rank_high(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.rank(method="average", pct=True)


def ensure_rank(df, raw_col):
    rc = "RANK_" + raw_col
    if rc not in df.columns and raw_col in df.columns:
        df[rc] = (
            df.groupby(["sid", "source_layer"], group_keys=False)[raw_col]
            .transform(percentile_rank_high)
        )
    return rc if rc in df.columns else None


def mean_existing(df, cols, out):
    good = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not good:
        return None
    df[out] = df[good].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return q
    ps = p[idx]
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ranked)
    qq = ranked * m / np.arange(1, m + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    qq = np.clip(qq, 0.0, 1.0)
    q[idx[order]] = qq
    return q


# -----------------------------------------------------------------------------
# Feature preparation
# -----------------------------------------------------------------------------

def add_self_scores(df):
    out = df.copy()

    raw_candidates = [
        "last_attn_next_max_positive_delta",
        "last_attn_next_delta_mean",
        "last_attn_next_real_mean",
        "last_attn_next_real_std_heads",
        "last_attn_same_real_mean",
        "last_attn_same_real_max",
        "last_attn_same_delta_mean",
        "last_attn_same_abs_delta_mean",
        "last_attn_same_max_positive_delta",
        "hidden_real_norm",
        "hidden_delta_norm_recomputed",
        "hidden_relative_delta",
        "attn_out_real_norm",
        "attn_out_delta_norm",
        "mlp_out_real_norm",
        "mlp_out_delta_norm",
    ]
    for c in raw_candidates:
        ensure_rank(out, c)

    mean_existing(
        out,
        [
            "RANK_last_attn_next_max_positive_delta",
            "RANK_last_attn_next_delta_mean",
            "RANK_last_attn_next_real_mean",
            "RANK_last_attn_next_real_std_heads",
            "RANK_last_attn_same_real_mean",
            "RANK_last_attn_same_real_max",
        ],
        "SCORE_attn_ensemble",
    )

    mean_existing(
        out,
        [
            "RANK_hidden_delta_norm_recomputed",
            "RANK_attn_out_delta_norm",
            "RANK_mlp_out_delta_norm",
        ],
        "SCORE_activity_ensemble",
    )

    mean_existing(
        out,
        [
            "RANK_hidden_real_norm",
            "RANK_attn_out_real_norm",
            "RANK_mlp_out_real_norm",
        ],
        "SCORE_absolute_activation_ensemble",
    )

    return out


def load_and_merge(a):
    selected_path = Path(a.fourway_dir) / "selected_topk_all_directions.csv"
    feat_path = Path(a.feature_dir) / "all_candidate_self_features.pkl.gz"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    sel = pd.read_csv(selected_path)
    feat = pd.read_pickle(feat_path, compression="gzip")

    for d in (sel, feat):
        d["sid"] = num(d["sid"]).astype(int)
        d["source_layer"] = num(d["source_layer"]).astype(int)
        d["position"] = num(d["position"]).astype(int)

    sel = sel[sel["selection_type"].astype(str) == a.selection_type].copy()
    sel["candidate_relation"] = sel["candidate_relation"].map(canon_rel)
    sel["gt"] = sel["gt"].map(canon_rel)
    sel["baseline_prediction"] = sel["baseline_prediction"].map(canon_rel)
    sel["baseline_correct"] = sel["baseline_correct"].map(boolify)
    sel["is_oracle_writer"] = sel["is_oracle_writer"].map(boolify)

    feat = feat[feat["sid"].isin(sel["sid"].unique())].copy()
    feat = add_self_scores(feat)

    # Keep all scalar feature columns plus any per-head columns.
    key = ["sid", "source_layer", "position"]
    head_cols = [
        c for c in feat.columns
        if re.fullmatch(r"last_attn_same_head\d+_(real|delta)", str(c))
    ]

    scalar_candidates = [
        "SCORE_attn_ensemble",
        "RANK_last_attn_next_max_positive_delta",
        "RANK_last_attn_next_delta_mean",
        "RANK_last_attn_next_real_mean",
        "RANK_last_attn_same_real_mean",
        "RANK_last_attn_same_real_max",
        "RANK_last_attn_same_delta_mean",
        "RANK_last_attn_same_abs_delta_mean",
        "RANK_last_attn_same_max_positive_delta",
        "SCORE_activity_ensemble",
        "SCORE_absolute_activation_ensemble",
        "RANK_hidden_real_norm",
        "RANK_hidden_delta_norm_recomputed",
        "RANK_attn_out_real_norm",
        "RANK_attn_out_delta_norm",
        "RANK_mlp_out_real_norm",
        "RANK_mlp_out_delta_norm",
    ]
    scalar_cols = [c for c in scalar_candidates if c in feat.columns]

    exact = feat[key + scalar_cols + head_cols].drop_duplicates(key)
    merged = sel.merge(exact, on=key, how="left", validate="many_to_one")

    # Position-only max across source band. This tests whether K7 POSITION is
    # strongly utilized somewhere, even if writer's exact layer is imperfect.
    pos_cols = [
        c for c in [
            "SCORE_attn_ensemble",
            "RANK_last_attn_next_max_positive_delta",
            "RANK_last_attn_next_delta_mean",
            "RANK_last_attn_next_real_mean",
            "RANK_last_attn_same_real_mean",
            "SCORE_activity_ensemble",
            "RANK_hidden_delta_norm_recomputed",
        ]
        if c in feat.columns
    ]
    if pos_cols:
        posmax = (
            feat[["sid", "position"] + pos_cols]
            .groupby(["sid", "position"], as_index=False)[pos_cols]
            .max()
            .rename(columns={c: c + "__POSMAX" for c in pos_cols})
        )
        merged = merged.merge(
            posmax, on=["sid", "position"], how="left", validate="many_to_one"
        )

    return merged, scalar_cols, head_cols


# -----------------------------------------------------------------------------
# Candidate signature construction
# -----------------------------------------------------------------------------

def layer_distribution(g, layers):
    counts = np.array([(g["source_layer"] == L).sum() for L in layers], float)
    return counts / counts.sum() if counts.sum() else counts


def weighted_layer_stats(g, layers, weight_col, prefix):
    if weight_col not in g.columns:
        return {}
    w = num(g[weight_col]).to_numpy(float)
    L = num(g["source_layer"]).to_numpy(float)
    good = np.isfinite(w) & np.isfinite(L)
    if not good.any():
        return {}
    w, L = w[good], L[good]
    w = np.clip(w, 0.0, None)
    if w.sum() <= EPS:
        return {}
    p = w / w.sum()
    mu = float(np.sum(p * L))
    var = float(np.sum(p * (L - mu) ** 2))
    # Mass by each actual source layer
    out = {
        f"{prefix}__layer_com": mu,
        f"{prefix}__layer_spread": math.sqrt(max(var, 0.0)),
    }
    mass = []
    for layer in layers:
        z = float(w[L == layer].sum())
        mass.append(z)
        out[f"{prefix}__mass_L{int(layer)}"] = z / float(w.sum())
    out[f"{prefix}__layer_entropy"] = entropy01(mass)
    return out


def head_signature(g, head_cols):
    """
    Aggregate same-layer last-query attention across K7 states.
    Returns per-head means + structural summaries.
    """
    out = {}
    if not head_cols:
        return out

    real_cols = sorted([c for c in head_cols if c.endswith("_real")])
    delta_cols = sorted([c for c in head_cols if c.endswith("_delta")])

    for kind, cols in [("head_real", real_cols), ("head_delta", delta_cols)]:
        if not cols:
            continue
        vec = np.array([safe_mean(g[c]) for c in cols], dtype=float)
        good = np.isfinite(vec)
        if not good.any():
            continue
        vv = vec[good]
        names = np.array(cols, dtype=object)[good]

        # Delta may be negative. For "positive recruitment" concentration,
        # use positive part; also retain signed per-head means.
        mass = np.clip(vv, 0.0, None)

        out[f"{kind}__std_heads"] = float(np.std(vv))
        out[f"{kind}__range_heads"] = float(np.max(vv) - np.min(vv))
        out[f"{kind}__entropy_pos"] = entropy01(mass)
        out[f"{kind}__gini_pos"] = gini_nonnegative(mass)

        if mass.sum() > EPS:
            order = np.argsort(mass)[::-1]
            out[f"{kind}__top1_share_pos"] = float(mass[order[:1]].sum()/mass.sum())
            out[f"{kind}__top3_share_pos"] = float(mass[order[:3]].sum()/mass.sum())
            out[f"{kind}__dominant_head"] = str(names[order[0]])
        else:
            out[f"{kind}__top1_share_pos"] = 0.0
            out[f"{kind}__top3_share_pos"] = 0.0
            out[f"{kind}__dominant_head"] = ""

        for c, v in zip(names, vv):
            # Strip prefix to compact HEADxx feature name.
            h = re.search(r"head(\d+)", str(c))
            hname = h.group(1) if h else str(c)
            out[f"{kind}__H{hname}"] = float(v)

    return out


def build_candidate_signatures(merged, scalar_cols, head_cols):
    layers = sorted(merged["source_layer"].dropna().astype(int).unique().tolist())

    # Metrics whose within-K7 structure we want, not only mean.
    structure_cols = [
        c for c in [
            "SCORE_attn_ensemble",
            "RANK_last_attn_next_max_positive_delta",
            "RANK_last_attn_next_delta_mean",
            "RANK_last_attn_next_real_mean",
            "RANK_last_attn_same_real_mean",
            "SCORE_activity_ensemble",
            "SCORE_absolute_activation_ensemble",
            "RANK_hidden_delta_norm_recomputed",
            "RANK_attn_out_delta_norm",
            "RANK_mlp_out_delta_norm",
        ]
        if c in merged.columns
    ]
    structure_cols += [
        c for c in merged.columns
        if c.endswith("__POSMAX")
        and (
            "attn" in c.lower()
            or "activity" in c.lower()
            or "hidden_delta" in c.lower()
        )
    ]

    group_cols = [
        "sid", "candidate_relation", "gt",
        "baseline_prediction", "baseline_correct"
    ]

    rows = []
    for key, g in merged.groupby(group_cols, dropna=False):
        sid, cand, gt, bp, bc = key
        row = {
            "sid": int(sid),
            "candidate_relation": cand,
            "gt": gt,
            "baseline_prediction": bp,
            "baseline_correct": bool(bc),
            "K": int(len(g)),
        }

        # Layer-count trajectory.
        ld = layer_distribution(g, layers)
        for L, v in zip(layers, ld):
            row[f"layer_count_share__L{L}"] = float(v)
        row["layer_count__com"] = float(
            np.sum(ld * np.asarray(layers, float))
        ) if ld.sum() else np.nan
        row["layer_count__entropy"] = entropy01(ld)

        # Within-K7 concentration signatures.
        for c in structure_cols:
            row.update(concentration_stats(num(g[c]).to_numpy(float), c))

        # Attention-weighted layer trajectory.
        for c in [
            "SCORE_attn_ensemble",
            "RANK_last_attn_next_delta_mean",
            "RANK_last_attn_next_real_mean",
            "RANK_last_attn_same_real_mean",
            "RANK_attn_out_delta_norm",
        ]:
            if c in g.columns:
                row.update(weighted_layer_stats(g, layers, c, c))

        # Head-level signature.
        row.update(head_signature(g, head_cols))
        rows.append(row)

    return pd.DataFrame(rows), layers


# -----------------------------------------------------------------------------
# Strict correct-vs-wrong tests
# -----------------------------------------------------------------------------

def stratified_stat(values, correct, relations):
    df = pd.DataFrame({
        "v": values,
        "correct": correct,
        "rel": relations,
    })
    diffs, ws = [], []
    for rel, g in df.groupby("rel"):
        c = num(g.loc[g["correct"], "v"]).dropna()
        w = num(g.loc[~g["correct"], "v"]).dropna()
        if len(c) and len(w):
            diffs.append(float(c.mean() - w.mean()))
            ws.append(len(c) + len(w))
    return float(np.average(diffs, weights=ws)) if diffs else np.nan


def relation_centered_effect(values, correct, relations):
    df = pd.DataFrame({
        "v": values,
        "correct": correct,
        "rel": relations,
    })
    df["v"] = num(df["v"])
    df = df.dropna()
    if not len(df):
        return np.nan
    df["resid"] = df["v"] - df.groupby("rel")["v"].transform("mean")
    c = df.loc[df.correct, "resid"].to_numpy(float)
    w = df.loc[~df.correct, "resid"].to_numpy(float)
    if len(c) < 2 or len(w) < 2:
        return np.nan
    pooled = math.sqrt(
        ((len(c)-1)*np.var(c, ddof=1) + (len(w)-1)*np.var(w, ddof=1))
        / max(len(c)+len(w)-2, 1)
    )
    if pooled <= EPS:
        return 0.0
    return float((np.mean(c)-np.mean(w))/pooled)


def stratified_perm(values, correct, relations, B, rng):
    v = np.asarray(values, float)
    y = np.asarray(correct, bool)
    r = np.asarray(relations, object)
    good = np.isfinite(v)
    v, y, r = v[good], y[good], r[good]
    obs = stratified_stat(v, y, r)
    if not np.isfinite(obs):
        return obs, np.nan

    idx_by_rel = {z: np.where(r == z)[0] for z in np.unique(r)}
    null = np.empty(B, float)
    for b in range(B):
        yp = y.copy()
        for _, idx in idx_by_rel.items():
            yp[idx] = rng.permutation(yp[idx])
        null[b] = stratified_stat(v, yp, r)
    q = np.isfinite(null)
    p = (1 + np.sum(np.abs(null[q]) >= abs(obs))) / (1 + q.sum())
    return obs, float(p)


def strict_gt_correct_wrong_tests(sig, B, seed):
    gt = sig[sig["candidate_relation"] == sig["gt"]].copy()

    meta = {
        "sid", "candidate_relation", "gt", "baseline_prediction",
        "baseline_correct", "K",
    }
    numeric_cols = [
        c for c in gt.columns
        if c not in meta and pd.api.types.is_numeric_dtype(gt[c])
    ]

    rng = np.random.default_rng(seed)
    rows = []
    for i, c in enumerate(numeric_cols):
        vals = num(gt[c]).to_numpy(float)
        corr = gt["baseline_correct"].to_numpy(bool)
        rel = gt["gt"].to_numpy(object)

        obs, p = stratified_perm(
            vals, corr, rel, B, np.random.default_rng(seed + i * 7919)
        )

        gc = gt[gt["baseline_correct"]]
        gw = gt[~gt["baseline_correct"]]

        rows.append({
            "feature": c,
            "N_correct": int(num(gc[c]).notna().sum()),
            "N_wrong": int(num(gw[c]).notna().sum()),
            "correct_mean": safe_mean(gc[c]),
            "wrong_mean": safe_mean(gw[c]),
            "pooled_correct_minus_wrong": safe_mean(gc[c]) - safe_mean(gw[c]),
            "relation_stratified_correct_minus_wrong": obs,
            "relation_centered_effect_d": relation_centered_effect(
                vals, corr, rel
            ),
            "perm_p": p,
        })

    out = pd.DataFrame(rows)
    out["fdr_q"] = bh_fdr(out["perm_p"].to_numpy(float))
    out["abs_effect_d"] = out["relation_centered_effect_d"].abs()
    return out.sort_values(
        ["fdr_q", "abs_effect_d"],
        ascending=[True, False],
        na_position="last",
    )


# -----------------------------------------------------------------------------
# Wrong-sample GT K7 vs baseline-wrong K7
# -----------------------------------------------------------------------------

def paired_significance(a, b, seed=17):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    d = a[good] - b[good]
    if len(d) == 0:
        return np.nan
    try:
        from scipy.stats import wilcoxon
        if np.allclose(d, 0):
            return 1.0
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        rng = np.random.default_rng(seed)
        obs = abs(d.mean())
        B = 10000
        null = np.empty(B)
        for i in range(B):
            null[i] = abs(np.mean(d * rng.choice([-1.0, 1.0], len(d))))
        return float((1 + np.sum(null >= obs))/(B + 1))


def wrong_gt_vs_wrong_candidate(sig):
    meta = {
        "sid", "candidate_relation", "gt", "baseline_prediction",
        "baseline_correct", "K",
    }
    features = [
        c for c in sig.columns
        if c not in meta and pd.api.types.is_numeric_dtype(sig[c])
    ]

    wrong = sig[~sig["baseline_correct"]].copy()
    rows = []
    for feat in features:
        gtv, wv = [], []
        for sid, g in wrong.groupby("sid"):
            gt = str(g["gt"].iloc[0])
            bp = str(g["baseline_prediction"].iloc[0])
            if gt == bp:
                continue
            a = g[g["candidate_relation"] == gt]
            b = g[g["candidate_relation"] == bp]
            if len(a) != 1 or len(b) != 1:
                continue
            av = float(a[feat].iloc[0])
            bv = float(b[feat].iloc[0])
            if np.isfinite(av) and np.isfinite(bv):
                gtv.append(av)
                wv.append(bv)

        if not gtv:
            continue

        gtarr, warr = np.asarray(gtv), np.asarray(wv)
        rows.append({
            "feature": feat,
            "N_wrong": len(gtarr),
            "GT_mean": float(gtarr.mean()),
            "baseline_wrong_mean": float(warr.mean()),
            "GT_minus_wrong_mean": float((gtarr-warr).mean()),
            "P_GT_gt_wrong": float(np.mean(gtarr > warr)),
            "P_wrong_gt_GT": float(np.mean(warr > gtarr)),
            "paired_p": paired_significance(gtarr, warr),
        })

    out = pd.DataFrame(rows)
    out["paired_fdr_q"] = bh_fdr(out["paired_p"].to_numpy(float))
    return out.sort_values(
        ["paired_fdr_q", "P_GT_gt_wrong"],
        ascending=[True, False],
        na_position="last",
    )


# -----------------------------------------------------------------------------
# Four-way selector diagnostics from individual signature metrics
# -----------------------------------------------------------------------------

def candidate_selection_by_scalar(sig):
    meta = {
        "sid", "candidate_relation", "gt", "baseline_prediction",
        "baseline_correct", "K",
    }
    features = [
        c for c in sig.columns
        if c not in meta
        and pd.api.types.is_numeric_dtype(sig[c])
        and sig[c].notna().sum() >= max(8, int(0.5 * len(sig)))
    ]

    rows = []
    details = []
    for feat in features:
        for orientation in ("high", "low"):
            ss = []
            for sid, g in sig.groupby("sid"):
                gg = g[["candidate_relation", "gt", "baseline_prediction",
                        "baseline_correct", feat]].dropna()
                if len(gg) < 4:
                    continue
                idx = gg[feat].idxmax() if orientation == "high" else gg[feat].idxmin()
                z = gg.loc[idx]
                pred = str(z["candidate_relation"])
                gt = str(z["gt"])
                bp = str(z["baseline_prediction"])
                bc = bool(z["baseline_correct"])
                ss.append({
                    "sid": int(sid),
                    "feature": feat,
                    "orientation": orientation,
                    "gt": gt,
                    "baseline_prediction": bp,
                    "baseline_correct": bc,
                    "pred": pred,
                    "pred_is_gt": pred == gt,
                    "pred_matches_baseline": pred == bp,
                })
            if not ss:
                continue
            d = pd.DataFrame(ss)
            corr = d["baseline_correct"]
            wrong = ~corr
            rows.append({
                "feature": feat,
                "orientation": orientation,
                "N": len(d),
                "GTacc_all": float(d["pred_is_gt"].mean()),
                "GTacc_correct": float(d.loc[corr, "pred_is_gt"].mean()) if corr.any() else np.nan,
                "GTacc_wrong": float(d.loc[wrong, "pred_is_gt"].mean()) if wrong.any() else np.nan,
                "match_baseline_wrong": float(
                    d.loc[wrong, "pred_matches_baseline"].mean()
                ) if wrong.any() else np.nan,
            })
            details.extend(ss)

    out = pd.DataFrame(rows)
    return (
        out.sort_values(["GTacc_wrong", "GTacc_all"], ascending=False),
        pd.DataFrame(details),
    )


# -----------------------------------------------------------------------------
# Cross-fitted "normal correct-K7" attention prototype
# -----------------------------------------------------------------------------

def choose_prototype_features(sig, strict_tests):
    """
    Use a compact, non-head-heavy set to avoid dimensional blowup on N=80.
    Features are still ATTENTION/ACTIVITY ONLY.

    We preferentially include:
      - layer trajectory
      - attention concentration
      - attention-weighted layer COM/entropy
      - head concentration summaries
    but exclude individual head identities from the default prototype.
    """
    candidates = []
    for c in sig.columns:
        if c.startswith("layer_count_share__"):
            candidates.append(c)
        elif c in {"layer_count__com", "layer_count__entropy"}:
            candidates.append(c)
        elif (
            ("SCORE_attn_ensemble__" in c
             or "RANK_last_attn_next_delta_mean__" in c
             or "RANK_last_attn_next_real_mean__" in c
             or "RANK_last_attn_same_real_mean__" in c)
            and any(k in c for k in [
                "__mean", "__std", "__entropy", "__gini",
                "__top1_share", "__top2_share",
                "__layer_com", "__layer_spread", "__layer_entropy",
            ])
        ):
            candidates.append(c)
        elif (
            c.startswith("head_real__") or c.startswith("head_delta__")
        ) and not re.search(r"__H\d+$", c):
            candidates.append(c)

    # Keep columns with enough finite values and nonzero variance.
    out = []
    for c in dict.fromkeys(candidates):
        x = num(sig[c])
        if x.notna().mean() >= 0.8 and float(x.std()) > 1e-8:
            out.append(c)
    return out


def crossfit_prototype_selection(sig, feature_cols, folds=5, seed=17,
                                 relation_specific=False):
    """
    Diagnostic only:
      Train fold: use GT K7 from BASELINE-CORRECT samples as "normal K7" prototype.
      Held-out sample: score all four candidate K7s by negative standardized
      squared distance to prototype; choose the most typical.

    At held-out inference this uses only endogenous K7 signatures, not GT.
    But prototype calibration itself uses labeled GT/correct samples.
    """
    sids = np.array(sorted(sig["sid"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    fold_id = {int(s): i % max(2, folds) for i, s in enumerate(sids)}

    detail = []
    for f in range(max(2, folds)):
        train = sig[~sig["sid"].map(fold_id).eq(f)].copy()
        test = sig[sig["sid"].map(fold_id).eq(f)].copy()

        normal = train[
            train["baseline_correct"]
            & (train["candidate_relation"] == train["gt"])
        ].copy()

        if len(normal) < 5:
            continue

        proto = {}
        if relation_specific:
            for rel in REL_ORDER:
                n = normal[normal["gt"] == rel]
                if len(n) < 2:
                    continue
                mu = n[feature_cols].apply(num).mean(axis=0)
                sd = n[feature_cols].apply(num).std(axis=0).replace(0, np.nan)
                proto[rel] = (mu, sd)
        else:
            mu = normal[feature_cols].apply(num).mean(axis=0)
            sd = normal[feature_cols].apply(num).std(axis=0).replace(0, np.nan)
            proto["GLOBAL"] = (mu, sd)

        for sid, g in test.groupby("sid"):
            scored = []
            for _, r in g.iterrows():
                key = r["candidate_relation"] if relation_specific else "GLOBAL"
                if key not in proto:
                    continue
                mu, sd = proto[key]
                x = num(r[feature_cols])
                good = x.notna() & mu.notna() & sd.notna() & (sd > 1e-8)
                if good.sum() < max(3, int(0.3 * len(feature_cols))):
                    continue
                z2 = ((x[good] - mu[good]) / sd[good]) ** 2
                score = -float(z2.mean())
                scored.append((score, str(r["candidate_relation"])))

            if len(scored) < 4:
                continue
            scored.sort(reverse=True)
            pred = scored[0][1]
            gt = str(g["gt"].iloc[0])
            bp = str(g["baseline_prediction"].iloc[0])
            bc = bool(g["baseline_correct"].iloc[0])
            detail.append({
                "sid": int(sid),
                "fold": f,
                "prototype": "relation_specific" if relation_specific else "global_correct",
                "pred": pred,
                "gt": gt,
                "baseline_prediction": bp,
                "baseline_correct": bc,
                "pred_is_gt": pred == gt,
                "pred_matches_baseline": pred == bp,
                "best_score": scored[0][0],
                "second_score": scored[1][0],
                "margin": scored[0][0] - scored[1][0],
            })

    d = pd.DataFrame(detail)
    if not len(d):
        return pd.DataFrame(), d

    corr = d["baseline_correct"]
    wrong = ~corr
    summary = pd.DataFrame([{
        "prototype": d["prototype"].iloc[0],
        "N": len(d),
        "n_features": len(feature_cols),
        "GTacc_all": float(d["pred_is_gt"].mean()),
        "GTacc_correct": float(d.loc[corr, "pred_is_gt"].mean()) if corr.any() else np.nan,
        "GTacc_wrong": float(d.loc[wrong, "pred_is_gt"].mean()) if wrong.any() else np.nan,
        "match_baseline_wrong": float(
            d.loc[wrong, "pred_matches_baseline"].mean()
        ) if wrong.any() else np.nan,
        "mean_margin_correct": safe_mean(d.loc[corr, "margin"]),
        "mean_margin_wrong": safe_mean(d.loc[wrong, "margin"]),
    }])
    return summary, d


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def feature_family(name):
    s = str(name)
    if re.search(r"head_(real|delta)__H\d+$", s):
        return "individual_head"
    if s.startswith("head_real__") or s.startswith("head_delta__"):
        return "head_structure"
    if s.startswith("layer_count"):
        return "layer_pattern"
    if "__layer_" in s or "__mass_L" in s:
        return "layer_trajectory"
    if any(k in s for k in ["entropy", "gini", "top1_share", "top2_share", "std", "range"]):
        return "within_k7_concentration"
    if "attn" in s.lower():
        return "attention_strength"
    if any(k in s.lower() for k in ["activity", "hidden", "mlp", "attn_out"]):
        return "activation"
    return "other"


def render_summary(a, sig, head_cols, strict, wrongpair, scalar_sel,
                   proto_summaries, proto_features):
    lines = []
    lines.append("=" * 170)
    lines.append("K7 ATTENTION SIGNATURE: CORRECT-vs-WRONG + FOUR-WAY SELECTION")
    lines.append("=" * 170)
    lines.append(
        f"N={sig.sid.nunique()} | candidate sets={len(sig)} | "
        f"selection_type={a.selection_type} | "
        f"head-level cache={'YES' if head_cols else 'NO'}"
    )
    lines.append(
        "Strict tests compare the GT K7 between baseline-correct and baseline-wrong "
        "samples using relation-stratified permutation tests and BH-FDR."
    )
    lines.append("")

    lines.append("1) STRICTEST CORRECT-vs-WRONG GT-K7 DIFFERENCES (FDR ranked)")
    lines.append("-" * 170)
    shown = strict.copy()
    shown["family"] = shown["feature"].map(feature_family)
    for r in shown.head(30).itertuples():
        lines.append(
            f"{r.feature:<78s} "
            f"correct={r.correct_mean:+.4f} wrong={r.wrong_mean:+.4f} "
            f"relΔ={r.relation_stratified_correct_minus_wrong:+.4f} "
            f"d={r.relation_centered_effect_d:+.3f} "
            f"p={r.perm_p:.4g} q={r.fdr_q:.4g}"
        )
    lines.append("")

    sig_hits = strict[strict["fdr_q"] < 0.05]
    lines.append(
        f"FDR q<.05 features: {len(sig_hits)} / {len(strict)}"
    )
    if len(sig_hits):
        fam = sig_hits["feature"].map(feature_family).value_counts()
        lines.append(
            "Significant families: "
            + ", ".join(f"{k}={v}" for k, v in fam.items())
        )
    lines.append("")

    lines.append("2) WRONG SAMPLES: GT K7 vs MODEL'S BASELINE-WRONG K7")
    lines.append("-" * 170)
    for r in wrongpair.head(25).itertuples():
        lines.append(
            f"{r.feature:<78s} "
            f"GT={r.GT_mean:+.4f} wrongK7={r.baseline_wrong_mean:+.4f} "
            f"GT-wrong={r.GT_minus_wrong_mean:+.4f} "
            f"P(GT>wrong)={r.P_GT_gt_wrong:.3f} "
            f"p={r.paired_p:.4g} q={r.paired_fdr_q:.4g}"
        )
    lines.append("")

    lines.append("3) BEST SINGLE-SIGNATURE RULES FOR FOUR-WAY K7 SELECTION")
    lines.append("-" * 170)
    for r in scalar_sel.head(25).itertuples():
        lines.append(
            f"{r.feature:<72s} {r.orientation:<4s} | "
            f"GTacc all={r.GTacc_all:.3f} correct={r.GTacc_correct:.3f} "
            f"wrong={r.GTacc_wrong:.3f} | "
            f"matchBaselineWrong={r.match_baseline_wrong:.3f}"
        )
    lines.append("")

    lines.append("4) CROSSFITTED 'NORMAL-CORRECT K7 ATTENTION SIGNATURE' DIAGNOSTIC")
    lines.append("-" * 170)
    lines.append(
        f"Prototype feature count={len(proto_features)}. "
        "Calibration uses GT K7s from baseline-correct TRAIN folds; held-out selection "
        "uses only attention signatures. This is NOT label-free, so treat it as a diagnostic ceiling."
    )
    if proto_summaries:
        for ss in proto_summaries:
            if not len(ss):
                continue
            r = ss.iloc[0]
            lines.append(
                f"{r['prototype']:<20s} N={int(r['N'])} | "
                f"GTacc all={r['GTacc_all']:.3f} correct={r['GTacc_correct']:.3f} "
                f"wrong={r['GTacc_wrong']:.3f} | "
                f"matchBaselineWrong={r['match_baseline_wrong']:.3f} | "
                f"margin correct/wrong={r['mean_margin_correct']:+.3f}/"
                f"{r['mean_margin_wrong']:+.3f}"
            )
    else:
        lines.append("No prototype result produced.")
    lines.append("")

    lines.append("HOW TO READ THIS")
    lines.append("-" * 170)
    lines.append(
        "A useful K7 selector must do more than separate correct vs wrong samples globally. "
        "On each sample it must rank the GT candidate above the other three; therefore "
        "the key column in section 3/4 is GTacc_wrong."
    )
    lines.append(
        "If correct-vs-wrong differences survive FDR but four-way selection remains near chance, "
        "the feature is a failure-state diagnostic, not a candidate-K7 selector."
    )
    lines.append(
        "If individual heads survive FDR or head-pattern prototype beats mean-attention rules, "
        "the missing information is in structured head utilization rather than scalar attention magnitude."
    )
    lines.append(
        "If layer trajectory/concentration features dominate, the practical next selector should "
        "score the temporal/depth organization of each candidate K7, not its average activation."
    )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    a = parse_args()
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged, scalar_cols, head_cols = load_and_merge(a)
    sig, layers = build_candidate_signatures(merged, scalar_cols, head_cols)

    # Save merged exact rows for manual inspection.
    merged.to_csv(outdir / "selected_k7_exact_rows_with_features.csv", index=False)
    sig.to_csv(outdir / "candidate_k7_attention_signatures.csv", index=False)

    strict = strict_gt_correct_wrong_tests(sig, a.perm, a.seed)
    strict.to_csv(outdir / "gt_k7_correct_vs_wrong_strict_fdr.csv", index=False)

    # Separate head-only table for readability.
    head_test = strict[
        strict["feature"].str.startswith("head_real__")
        | strict["feature"].str.startswith("head_delta__")
    ].copy()
    head_test.to_csv(outdir / "head_pattern_correct_vs_wrong_fdr.csv", index=False)

    layer_test = strict[
        strict["feature"].str.startswith("layer_count")
        | strict["feature"].str.contains("__layer_", regex=False)
        | strict["feature"].str.contains("__mass_L", regex=False)
    ].copy()
    layer_test.to_csv(outdir / "layer_pattern_correct_vs_wrong_fdr.csv", index=False)

    concentration_test = strict[
        strict["feature"].str.contains(
            "__entropy|__gini|__top1_share|__top2_share|__std|__range",
            regex=True
        )
    ].copy()
    concentration_test.to_csv(
        outdir / "within_k7_concentration_correct_vs_wrong_fdr.csv",
        index=False
    )

    wrongpair = wrong_gt_vs_wrong_candidate(sig)
    wrongpair.to_csv(outdir / "wrong_samples_gtK7_vs_baselineWrongK7.csv", index=False)

    scalar_sel, scalar_detail = candidate_selection_by_scalar(sig)
    scalar_sel.to_csv(outdir / "fourway_single_signature_selection.csv", index=False)
    scalar_detail.to_csv(outdir / "fourway_single_signature_selection_detail.csv", index=False)

    # Cross-fitted attention-pattern typicality diagnostic.
    proto_features = choose_prototype_features(sig, strict)

    proto_summaries = []
    proto_details = []
    if len(proto_features) >= 3:
        s1, d1 = crossfit_prototype_selection(
            sig, proto_features, folds=a.folds, seed=a.seed,
            relation_specific=False
        )
        if len(s1):
            proto_summaries.append(s1)
            proto_details.append(d1)

        s2, d2 = crossfit_prototype_selection(
            sig, proto_features, folds=a.folds, seed=a.seed,
            relation_specific=True
        )
        if len(s2):
            proto_summaries.append(s2)
            proto_details.append(d2)

    if proto_summaries:
        ps = pd.concat(proto_summaries, ignore_index=True)
    else:
        ps = pd.DataFrame()
    if proto_details:
        pd.concat(proto_details, ignore_index=True).to_csv(
            outdir / "crossfit_attention_prototype_detail.csv", index=False
        )
    ps.to_csv(outdir / "crossfit_attention_prototype_summary.csv", index=False)

    (outdir / "prototype_features.txt").write_text(
        "\n".join(proto_features) + "\n", encoding="utf-8"
    )

    meta = {
        "N_samples": int(sig.sid.nunique()),
        "N_candidate_sets": int(len(sig)),
        "K_mean": float(sig.K.mean()),
        "layers": layers,
        "selection_type": a.selection_type,
        "n_head_columns_in_cache": len(head_cols),
        "head_scan_available": bool(head_cols),
        "n_signature_features": int(
            len([c for c in sig.columns if c not in {
                "sid","candidate_relation","gt","baseline_prediction",
                "baseline_correct","K"
            }])
        ),
        "n_prototype_features": len(proto_features),
        "strict_test": (
            "GT-K7 correct vs wrong; correctness labels permuted within GT relation; "
            "BH-FDR across all tested signature features"
        ),
        "prototype_note": (
            "cross-fitted diagnostic only; calibration uses labeled baseline-correct "
            "GT K7s, held-out scoring uses only endogenous attention signatures"
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    text = render_summary(
        a, sig, head_cols, strict, wrongpair, scalar_sel,
        proto_summaries, proto_features
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    print(text)
    if not head_cols:
        print(
            "\nWARNING: no per-head columns were found. Layer/concentration analyses "
            "still ran, but head-pattern analysis is unavailable. Re-run the previous "
            "self-feature extraction with --attention-head-scan if needed."
        )
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
