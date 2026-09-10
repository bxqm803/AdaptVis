#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze endogenous ATTENTION / ACTIVATION strength on the four writer-specific K7s.

No VLM forward pass is required.

Inputs
------
1) FOUR-WAY writer tracing output:
   <fourway-dir>/selected_topk_all_directions.csv

   For each sample it must contain four candidate K7s:
       K7_left, K7_right, K7_on, K7_under

2) Relation-free self-feature cache:
   <feature-dir>/all_candidate_self_features.pkl.gz

What this script asks
---------------------
A. On BASELINE-CORRECT samples:
   Is the GT writer's K7 more strongly attended/activated than the other
   three candidate K7s?

B. On BASELINE-WRONG samples:
   Is the model's WRONG-ANSWER K7 more strongly attended/activated than
   the GT K7?
   Or does the GT K7 remain strongly attended even when behavior is wrong?

C. Across correct vs wrong samples:
   Does the GT K7 itself have systematically different endogenous
   attention/activity strength?
   A relation-stratified permutation test is included so LEFT/RIGHT/ON/UNDER
   class imbalance does not trivially create the difference.

D. Can any endogenous score choose among the four K7s WITHOUT GT?
   For every metric:
       predicted relation = argmax score(K7_relation)
   and report accuracy overall / baseline-correct / baseline-wrong,
   plus how often it merely follows the baseline wrong answer.

Important
---------
This script is diagnostic. It does NOT use GT to compute candidate scores.
GT is used only after scoring to evaluate which candidate was correct.

Recommended command
-------------------
python analyze_four_k7_endogenous_strength_correct_wrong_v1.py \
  --fourway-dir output/qwen3b_four_writer_k7_N80 \
  --feature-dir output/qwen3b_coco_L20_26_self_predict_core \
  --selection-type raw \
  --output-dir output/qwen3b_four_k7_endogenous_strength_N80
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REL_ORDER = ["left", "right", "on", "under"]


# =============================================================================
# Utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument(
        "--selection-type",
        default="raw",
        choices=["raw", "contrast"],
        help="Which four writer-specific K7 sets to analyze.",
    )
    p.add_argument(
        "--perm",
        type=int,
        default=5000,
        help="Relation-stratified permutations for correct-vs-wrong GT-K7 tests.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower()
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under", "bottom": "under",
    }
    return table.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_mean(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def percentile_rank_high(s):
    """Within-group percentile rank; larger raw value -> larger rank."""
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


def add_scores(features):
    """
    Recreate the same relation-free rank/ensemble features used by the previous
    self-core analysis. All ranks are within SID x layer, which reduces layer
    scale confounding when comparing writer-specific K7s.
    """
    df = features.copy()

    raw_cols = [
        # attention
        "last_attn_next_max_positive_delta",
        "last_attn_next_delta_mean",
        "last_attn_next_real_mean",
        "last_attn_next_real_std_heads",
        "last_attn_same_real_mean",
        "last_attn_same_real_max",
        # hidden / module activity
        "hidden_real_norm",
        "hidden_delta_norm_recomputed",
        "hidden_relative_delta",
        "attn_out_real_norm",
        "attn_out_delta_norm",
        "mlp_out_real_norm",
        "mlp_out_delta_norm",
    ]

    for c in raw_cols:
        ensure_rank(df, c)

    # Rebuild attention ensemble if absent.
    mean_existing(
        df,
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

    # Activity ensemble: image-induced change magnitude, not raw absolute norm.
    mean_existing(
        df,
        [
            "RANK_hidden_delta_norm_recomputed",
            "RANK_attn_out_delta_norm",
            "RANK_mlp_out_delta_norm",
        ],
        "SCORE_activity_ensemble",
    )

    # Absolute activation ensemble: how active the state/modules are in REAL,
    # irrespective of Real-Gray change.
    mean_existing(
        df,
        [
            "RANK_hidden_real_norm",
            "RANK_attn_out_real_norm",
            "RANK_mlp_out_real_norm",
        ],
        "SCORE_absolute_activation_ensemble",
    )

    return df


def stratified_correct_wrong_stat(values, correct, relations):
    """
    Weighted within-relation mean(correct)-mean(wrong).
    Only relations containing both groups contribute.
    """
    tmp = pd.DataFrame(
        {"v": values, "correct": correct, "rel": relations}
    )
    parts = []
    weights = []
    for rel, g in tmp.groupby("rel"):
        c = pd.to_numeric(g.loc[g.correct, "v"], errors="coerce").dropna()
        w = pd.to_numeric(g.loc[~g.correct, "v"], errors="coerce").dropna()
        if len(c) and len(w):
            parts.append(float(c.mean() - w.mean()))
            weights.append(len(c) + len(w))
    if not parts:
        return float("nan")
    return float(np.average(parts, weights=weights))


def stratified_perm_test(values, correct, relations, n_perm=5000, seed=17):
    """
    Shuffle correctness labels independently within each GT relation while
    preserving the correct/wrong count in that relation.
    Two-sided p-value for weighted within-relation mean difference.
    """
    values = np.asarray(values, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    relations = np.asarray(relations, dtype=object)

    finite = np.isfinite(values)
    values = values[finite]
    correct = correct[finite]
    relations = relations[finite]

    obs = stratified_correct_wrong_stat(values, correct, relations)
    if not np.isfinite(obs):
        return obs, float("nan")

    rng = np.random.default_rng(seed)
    rel_indices = {
        r: np.where(relations == r)[0]
        for r in np.unique(relations)
    }

    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        y = correct.copy()
        for r, idx in rel_indices.items():
            if len(idx) <= 1:
                continue
            y[idx] = rng.permutation(y[idx])
        null[b] = stratified_correct_wrong_stat(values, y, relations)

    valid = np.isfinite(null)
    if not valid.any():
        return obs, float("nan")
    p = (1 + np.sum(np.abs(null[valid]) >= abs(obs))) / (1 + valid.sum())
    return obs, float(p)


def paired_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) == 0:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        if np.allclose(a, b):
            return 1.0
        return float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except Exception:
        # fallback random sign-flip p
        d = a - b
        d = d[np.abs(d) > 1e-12]
        if len(d) == 0:
            return 1.0
        rng = np.random.default_rng(17)
        obs = abs(d.mean())
        B = 10000
        null = np.empty(B)
        for i in range(B):
            signs = rng.choice([-1.0, 1.0], size=len(d))
            null[i] = abs(np.mean(d * signs))
        return float((1 + np.sum(null >= obs)) / (B + 1))


# =============================================================================
# Build exact-state and position-level endogenous scores
# =============================================================================

def prepare_data(a):
    fourway_dir = Path(a.fourway_dir)
    feature_dir = Path(a.feature_dir)

    spath = fourway_dir / "selected_topk_all_directions.csv"
    fpath = feature_dir / "all_candidate_self_features.pkl.gz"

    if not spath.exists():
        raise FileNotFoundError(spath)
    if not fpath.exists():
        raise FileNotFoundError(fpath)

    sel = pd.read_csv(spath)
    feat = pd.read_pickle(fpath, compression="gzip")

    for d in [sel, feat]:
        d["sid"] = pd.to_numeric(d["sid"], errors="raise").astype(int)
        d["source_layer"] = pd.to_numeric(
            d["source_layer"], errors="raise"
        ).astype(int)
        d["position"] = pd.to_numeric(
            d["position"], errors="raise"
        ).astype(int)

    sel = sel[sel["selection_type"].astype(str) == a.selection_type].copy()
    if not len(sel):
        raise RuntimeError(
            f"No rows with selection_type={a.selection_type!r} in {spath}"
        )

    sel["candidate_relation"] = sel["candidate_relation"].map(canon_rel)
    sel["gt"] = sel["gt"].map(canon_rel)
    sel["baseline_prediction"] = sel["baseline_prediction"].map(canon_rel)
    sel["baseline_correct"] = sel["baseline_correct"].map(boolify)
    sel["is_oracle_writer"] = sel["is_oracle_writer"].map(boolify)

    feat = add_scores(feat)

    # Candidate sample set only.
    sids = sorted(sel["sid"].unique())
    feat = feat[feat["sid"].isin(sids)].copy()

    # Exact (L,p) merge.
    meta_cols = ["sid", "source_layer", "position"]
    useful = [
        "SCORE_attn_ensemble",
        "RANK_last_attn_next_delta_mean",
        "RANK_last_attn_next_max_positive_delta",
        "RANK_last_attn_next_real_mean",
        "RANK_last_attn_same_real_mean",
        "SCORE_activity_ensemble",
        "RANK_hidden_delta_norm_recomputed",
        "RANK_attn_out_delta_norm",
        "RANK_mlp_out_delta_norm",
        "SCORE_absolute_activation_ensemble",
        "RANK_hidden_real_norm",
        "RANK_attn_out_real_norm",
        "RANK_mlp_out_real_norm",
    ]
    useful = [c for c in useful if c in feat.columns]

    exact_feat = feat[meta_cols + useful].drop_duplicates(meta_cols)
    x = sel.merge(exact_feat, on=meta_cols, how="left", validate="many_to_one")

    # Position-level max across L20-L26 for each endogenous feature.
    # This intentionally ignores writer-selected exact layer and asks only:
    # "Is this token position strongly read/activated anywhere in the causal band?"
    pos_base = feat[["sid", "position"] + useful].copy()
    posmax = (
        pos_base.groupby(["sid", "position"], as_index=False)[useful]
        .max()
        .rename(columns={c: c + "__POSMAX" for c in useful})
    )
    x = x.merge(posmax, on=["sid", "position"], how="left", validate="many_to_one")

    return x, useful


def candidate_score_columns(x, useful):
    """
    Aggregate each candidate K7 into one score per sample x candidate relation.

    We use mean across seven selected states. Since all component scores are
    percentile/rank-like or ensembles in [roughly 0,1], higher = stronger.
    """
    score_sources = []

    preferred_exact = [
        "SCORE_attn_ensemble",
        "RANK_last_attn_next_delta_mean",
        "RANK_last_attn_next_max_positive_delta",
        "RANK_last_attn_next_real_mean",
        "RANK_last_attn_same_real_mean",
        "SCORE_activity_ensemble",
        "RANK_hidden_delta_norm_recomputed",
        "RANK_attn_out_delta_norm",
        "RANK_mlp_out_delta_norm",
        "SCORE_absolute_activation_ensemble",
        "RANK_hidden_real_norm",
    ]
    for c in preferred_exact:
        if c in x.columns:
            score_sources.append((c, c + "__K7MEAN"))

    # Position-only scores are especially important because prior experiments
    # showed position localization was easier than exact layer localization.
    for c in [
        "SCORE_attn_ensemble",
        "RANK_last_attn_next_delta_mean",
        "RANK_last_attn_next_max_positive_delta",
        "SCORE_activity_ensemble",
        "RANK_hidden_delta_norm_recomputed",
        "SCORE_absolute_activation_ensemble",
    ]:
        pc = c + "__POSMAX"
        if pc in x.columns:
            score_sources.append((pc, c + "__POSMAX_K7MEAN"))

    group_cols = [
        "sid", "candidate_relation", "gt",
        "baseline_prediction", "baseline_correct",
    ]

    rows = []
    for key, g in x.groupby(group_cols, dropna=False):
        sid, cand, gt, bp, bc = key
        row = {
            "sid": int(sid),
            "candidate_relation": cand,
            "gt": gt,
            "baseline_prediction": bp,
            "baseline_correct": bool(bc),
            "K": int(len(g)),
        }
        for src, out in score_sources:
            row[out] = safe_mean(g[src])

        # Simple "how many of K7 are in endogenous top-20%" diagnostics.
        if "SCORE_attn_ensemble" in g.columns:
            vals = pd.to_numeric(g["SCORE_attn_ensemble"], errors="coerce")
            row["ATTN_EXACT_HIGH80_COUNT"] = int((vals >= 0.80).sum())
        if "SCORE_attn_ensemble__POSMAX" in g.columns:
            vals = pd.to_numeric(
                g["SCORE_attn_ensemble__POSMAX"], errors="coerce"
            )
            row["ATTN_POSMAX_HIGH80_COUNT"] = int((vals >= 0.80).sum())
        if "SCORE_activity_ensemble" in g.columns:
            vals = pd.to_numeric(g["SCORE_activity_ensemble"], errors="coerce")
            row["ACTIVITY_EXACT_HIGH80_COUNT"] = int((vals >= 0.80).sum())

        rows.append(row)

    cand = pd.DataFrame(rows)
    score_cols = [
        c for c in cand.columns
        if c.endswith("__K7MEAN") or c.endswith("_HIGH80_COUNT")
    ]
    return cand, score_cols


# =============================================================================
# Analyses
# =============================================================================

def build_sample_comparison(cand, metric):
    rows = []
    for sid, g in cand.groupby("sid"):
        if len(g) < 4:
            continue
        gt = str(g["gt"].iloc[0])
        bp = str(g["baseline_prediction"].iloc[0])
        bc = bool(g["baseline_correct"].iloc[0])

        score = {
            str(r.candidate_relation): float(getattr(r, metric))
            for r in g.itertuples()
            if np.isfinite(float(getattr(r, metric)))
        }
        if gt not in score:
            continue

        others = [v for k, v in score.items() if k != gt]
        if not others:
            continue

        pred = max(score, key=score.get)
        gt_score = score[gt]
        best_other = max(others)

        row = {
            "sid": int(sid),
            "metric": metric,
            "gt": gt,
            "baseline_prediction": bp,
            "baseline_correct": bc,
            "pred_candidate": pred,
            "pred_is_gt": pred == gt,
            "pred_matches_baseline": pred == bp,
            "gt_score": gt_score,
            "best_other_score": best_other,
            "gt_minus_best_other": gt_score - best_other,
        }

        if (not bc) and bp in score and bp != gt:
            row["baseline_wrong_score"] = score[bp]
            row["wrong_minus_gt"] = score[bp] - gt_score
            row["wrong_gt_pair_available"] = True
        else:
            row["baseline_wrong_score"] = np.nan
            row["wrong_minus_gt"] = np.nan
            row["wrong_gt_pair_available"] = False

        for r in REL_ORDER:
            row["score_" + r] = score.get(r, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def selection_summary(comparisons):
    out = []
    for metric, g in comparisons.groupby("metric"):
        correct = g["baseline_correct"].astype(bool)
        wrong = ~correct

        out.append({
            "metric": metric,
            "N": len(g),
            "fourway_gt_acc_all": float(g["pred_is_gt"].mean()),
            "fourway_gt_acc_baseline_correct": (
                float(g.loc[correct, "pred_is_gt"].mean())
                if correct.any() else np.nan
            ),
            "fourway_gt_acc_baseline_wrong": (
                float(g.loc[wrong, "pred_is_gt"].mean())
                if wrong.any() else np.nan
            ),
            "match_baseline_all": float(g["pred_matches_baseline"].mean()),
            "match_baseline_when_wrong": (
                float(g.loc[wrong, "pred_matches_baseline"].mean())
                if wrong.any() else np.nan
            ),
            "mean_gt_margin_correct": safe_mean(
                g.loc[correct, "gt_minus_best_other"]
            ),
            "mean_gt_margin_wrong": safe_mean(
                g.loc[wrong, "gt_minus_best_other"]
            ),
        })

    return (
        pd.DataFrame(out)
        .sort_values(
            ["fourway_gt_acc_baseline_wrong", "fourway_gt_acc_all"],
            ascending=False,
        )
    )


def correct_wrong_gt_strength(comparisons, n_perm, seed):
    """
    Does the GT K7 itself have different endogenous strength in correct vs wrong
    samples? Pooled means + relation-stratified permutation test.
    """
    out = []
    for metric, g in comparisons.groupby("metric"):
        c = g[g["baseline_correct"]]
        w = g[~g["baseline_correct"]]

        stat, p = stratified_perm_test(
            g["gt_score"].to_numpy(float),
            g["baseline_correct"].to_numpy(bool),
            g["gt"].to_numpy(object),
            n_perm=n_perm,
            seed=seed,
        )

        out.append({
            "metric": metric,
            "N_correct": len(c),
            "N_wrong": len(w),
            "GT_K7_score_correct_mean": safe_mean(c["gt_score"]),
            "GT_K7_score_correct_std": safe_std(c["gt_score"]),
            "GT_K7_score_wrong_mean": safe_mean(w["gt_score"]),
            "GT_K7_score_wrong_std": safe_std(w["gt_score"]),
            "pooled_correct_minus_wrong": (
                safe_mean(c["gt_score"]) - safe_mean(w["gt_score"])
            ),
            "relation_stratified_correct_minus_wrong": stat,
            "relation_stratified_perm_p": p,
        })

    return pd.DataFrame(out).sort_values(
        "relation_stratified_perm_p", na_position="last"
    )


def wrong_sample_pair_analysis(comparisons):
    """
    On baseline-wrong samples, directly compare the K7 of the model's emitted
    WRONG relation against the GT relation's K7.
    """
    out = []
    wrong = comparisons[
        (~comparisons["baseline_correct"])
        & comparisons["wrong_gt_pair_available"]
    ].copy()

    for metric, g in wrong.groupby("metric"):
        wrong_score = pd.to_numeric(g["baseline_wrong_score"], errors="coerce")
        gt_score = pd.to_numeric(g["gt_score"], errors="coerce")
        good = wrong_score.notna() & gt_score.notna()
        wrong_score = wrong_score[good].to_numpy(float)
        gt_score = gt_score[good].to_numpy(float)

        out.append({
            "metric": metric,
            "N_wrong": len(gt_score),
            "GT_K7_mean": safe_mean(gt_score),
            "baseline_WRONG_K7_mean": safe_mean(wrong_score),
            "wrong_minus_GT_mean": safe_mean(wrong_score - gt_score),
            "fraction_wrong_K7_stronger_than_GT": (
                float(np.mean(wrong_score > gt_score))
                if len(gt_score) else np.nan
            ),
            "fraction_GT_K7_stronger_than_wrong": (
                float(np.mean(gt_score > wrong_score))
                if len(gt_score) else np.nan
            ),
            "paired_wilcoxon_p": paired_wilcoxon(wrong_score, gt_score),
        })

    return pd.DataFrame(out).sort_values(
        "fraction_GT_K7_stronger_than_wrong",
        ascending=False,
    )


def correct_sample_gt_vs_other(comparisons):
    out = []
    corr = comparisons[comparisons["baseline_correct"]].copy()
    for metric, g in corr.groupby("metric"):
        gt = pd.to_numeric(g["gt_score"], errors="coerce").to_numpy(float)
        other = pd.to_numeric(
            g["best_other_score"], errors="coerce"
        ).to_numpy(float)
        good = np.isfinite(gt) & np.isfinite(other)
        gt, other = gt[good], other[good]
        out.append({
            "metric": metric,
            "N_correct": len(gt),
            "GT_K7_mean": safe_mean(gt),
            "best_wrong_candidate_mean": safe_mean(other),
            "GT_minus_best_other_mean": safe_mean(gt - other),
            "fraction_GT_stronger_than_every_other": (
                float(np.mean(gt > other)) if len(gt) else np.nan
            ),
            "paired_wilcoxon_p": paired_wilcoxon(gt, other),
        })
    return pd.DataFrame(out).sort_values(
        "fraction_GT_stronger_than_every_other",
        ascending=False,
    )


def by_relation_strength(comparisons):
    rows = []
    for (metric, rel, bc), g in comparisons.groupby(
        ["metric", "gt", "baseline_correct"]
    ):
        rows.append({
            "metric": metric,
            "gt": rel,
            "baseline_group": "correct" if bc else "wrong",
            "N": len(g),
            "GT_K7_mean": safe_mean(g["gt_score"]),
            "GT_margin_mean": safe_mean(g["gt_minus_best_other"]),
            "fourway_gt_rate": float(g["pred_is_gt"].mean()),
            "match_baseline_rate": float(g["pred_matches_baseline"].mean()),
            "baseline_wrong_K7_mean": safe_mean(g["baseline_wrong_score"]),
            "wrong_minus_GT_mean": safe_mean(g["wrong_minus_gt"]),
        })
    return pd.DataFrame(rows)


def render_summary(
    a,
    cand,
    score_cols,
    selection,
    gt_cw,
    wrong_pair,
    correct_pair,
):
    lines = []
    lines.append("=" * 160)
    lines.append(
        "FOUR K7s: ENDOGENOUS ATTENTION / ACTIVATION STRENGTH — CORRECT vs WRONG"
    )
    lines.append("=" * 160)
    lines.append(
        f"selection_type={a.selection_type} | "
        f"N samples={cand['sid'].nunique()} | "
        f"candidate K mean={cand['K'].mean():.2f}"
    )
    lines.append(
        "All candidate scores are relation-free. GT/baseline labels are used only after scoring for diagnostics."
    )
    lines.append(
        "Rank-based features are normalized within sample x layer; POSMAX variants ignore exact layer and use the strongest score for that position across L20-L26."
    )
    lines.append("")

    lines.append("1) CAN ENDOGENOUS STRENGTH CHOOSE ONE OF THE FOUR K7s?")
    lines.append("-" * 160)
    for r in selection.head(20).itertuples():
        lines.append(
            f"{r.metric:<58s} "
            f"GTacc all={r.fourway_gt_acc_all:.3f} "
            f"correct={r.fourway_gt_acc_baseline_correct:.3f} "
            f"wrong={r.fourway_gt_acc_baseline_wrong:.3f} | "
            f"matchBaselineWrong={r.match_baseline_when_wrong:.3f} | "
            f"GTmargin correct/wrong={r.mean_gt_margin_correct:+.3f}/{r.mean_gt_margin_wrong:+.3f}"
        )
    lines.append("")

    lines.append("2) ON BASELINE-WRONG SAMPLES: IS WRONG-ANSWER K7 STRONGER THAN GT K7?")
    lines.append("-" * 160)
    for r in wrong_pair.head(20).itertuples():
        lines.append(
            f"{r.metric:<58s} "
            f"GT={r.GT_K7_mean:.3f} wrongAns={r.baseline_WRONG_K7_mean:.3f} "
            f"wrong-GT={r.wrong_minus_GT_mean:+.3f} | "
            f"P(wrong>GT)={r.fraction_wrong_K7_stronger_than_GT:.3f} "
            f"P(GT>wrong)={r.fraction_GT_K7_stronger_than_wrong:.3f} "
            f"paired-p={r.paired_wilcoxon_p:.4g}"
        )
    lines.append("")

    lines.append("3) ON BASELINE-CORRECT SAMPLES: IS GT K7 STRONGER THAN ALL OTHER K7s?")
    lines.append("-" * 160)
    for r in correct_pair.head(20).itertuples():
        lines.append(
            f"{r.metric:<58s} "
            f"GT={r.GT_K7_mean:.3f} bestOther={r.best_wrong_candidate_mean:.3f} "
            f"margin={r.GT_minus_best_other_mean:+.3f} | "
            f"P(GT>all others)={r.fraction_GT_stronger_than_every_other:.3f} "
            f"paired-p={r.paired_wilcoxon_p:.4g}"
        )
    lines.append("")

    lines.append("4) DOES THE TRUE GT K7 ITSELF LOOK DIFFERENT IN CORRECT vs WRONG SAMPLES?")
    lines.append("-" * 160)
    for r in gt_cw.head(20).itertuples():
        lines.append(
            f"{r.metric:<58s} "
            f"correct={r.GT_K7_score_correct_mean:.3f} "
            f"wrong={r.GT_K7_score_wrong_mean:.3f} "
            f"pooledΔ={r.pooled_correct_minus_wrong:+.3f} "
            f"relation-controlledΔ={r.relation_stratified_correct_minus_wrong:+.3f} "
            f"perm-p={r.relation_stratified_perm_p:.4g}"
        )
    lines.append("")

    if len(selection):
        best = selection.iloc[0]
        lines.append("QUICK READ")
        lines.append("-" * 160)
        lines.append(
            f"Best current non-oracle K7 judge by baseline-wrong GT accuracy: "
            f"{best['metric']} -> wrong-group GT selection "
            f"{best['fourway_gt_acc_baseline_wrong']:.3f}, overall "
            f"{best['fourway_gt_acc_all']:.3f}."
        )
        lines.append(
            "If a metric has high P(wrong-answer K7 > GT K7) on wrong samples and high "
            "P(GT > all others) on correct samples, it is decision-congruent: it tracks "
            "the model's current chosen decision rather than an independent corrective signal."
        )
        lines.append(
            "If GT K7 remains equally/highly strong on wrong samples but is not ranked first, "
            "the problem is relative competition among candidate K7s rather than disappearance "
            "of the true causal carrier set."
        )
        lines.append(
            "If POSMAX attention beats exact-layer attention, use attention to judge candidate "
            "POSITIONS and treat layer choice as a separate problem."
        )

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    x, useful = prepare_data(a)
    cand, score_cols = candidate_score_columns(x, useful)

    if cand["sid"].nunique() == 0:
        raise RuntimeError("No candidate K7s remained after merging.")

    # Warn about merge coverage.
    missing_counts = {}
    for c in score_cols:
        missing_counts[c] = int(cand[c].isna().sum())

    all_comps = []
    for metric in score_cols:
        comp = build_sample_comparison(cand, metric)
        if len(comp):
            all_comps.append(comp)
    comparisons = (
        pd.concat(all_comps, ignore_index=True)
        if all_comps else pd.DataFrame()
    )
    if not len(comparisons):
        raise RuntimeError(
            "No endogenous score columns could be evaluated. "
            "Inspect all_candidate_self_features.pkl.gz columns."
        )

    selection = selection_summary(comparisons)
    gt_cw = correct_wrong_gt_strength(
        comparisons, n_perm=a.perm, seed=a.seed
    )
    wrong_pair = wrong_sample_pair_analysis(comparisons)
    correct_pair = correct_sample_gt_vs_other(comparisons)
    by_rel = by_relation_strength(comparisons)

    # Save.
    x.to_csv(outdir / "selected_k7_with_endogenous_features.csv", index=False)
    cand.to_csv(outdir / "candidate_k7_scores.csv", index=False)
    comparisons.to_csv(outdir / "per_sample_metric_comparisons.csv", index=False)
    selection.to_csv(outdir / "nonoracle_k7_selection_summary.csv", index=False)
    gt_cw.to_csv(outdir / "gt_k7_correct_vs_wrong.csv", index=False)
    wrong_pair.to_csv(outdir / "wrong_samples_wrongK7_vs_gtK7.csv", index=False)
    correct_pair.to_csv(outdir / "correct_samples_gtK7_vs_others.csv", index=False)
    by_rel.to_csv(outdir / "strength_by_relation_and_correctness.csv", index=False)

    metadata = {
        "selection_type": a.selection_type,
        "N_samples": int(cand["sid"].nunique()),
        "N_candidate_sets": int(len(cand)),
        "score_columns": score_cols,
        "candidate_score_missing_counts": missing_counts,
        "feature_file": str(
            Path(a.feature_dir) / "all_candidate_self_features.pkl.gz"
        ),
        "selected_file": str(
            Path(a.fourway_dir) / "selected_topk_all_directions.csv"
        ),
        "interpretation": {
            "exact": "score at writer-selected exact (layer,position) states",
            "posmax": "for each selected position, maximum endogenous score across source layers before averaging K7",
            "higher": "all reported candidate selection rules use higher score = stronger",
            "GT_usage": "GT only evaluates candidate identity after scores are computed",
        },
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    text = render_summary(
        a, cand, score_cols, selection, gt_cw, wrong_pair, correct_pair
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
