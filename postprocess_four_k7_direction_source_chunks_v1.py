#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process cached chunks from validate_four_k7_vs_direction_head_sources_v2/v4.
No model loading, no GPU forward pass.

Use this after the expensive run already finished but failed during summary.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REL = ("left", "right", "on", "under")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="Existing output dir containing chunks/sid_*.pkl.gz")
    p.add_argument("--ks", default="1,3,5,7,14,24,36")
    p.add_argument("--selector-k", type=int, default=7)
    p.add_argument("--focus-heads", default="L26H03,L23H01,L23H05")
    return p.parse_args()


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def parse_head_name(s):
    m = re.fullmatch(r"\s*L(\d+)H(\d+)\s*", str(s), flags=re.I)
    if not m:
        raise ValueError(f"Bad head name: {s!r}; expected L26H03")
    return int(m.group(1)), int(m.group(2))


def head_name(L, H):
    return f"L{int(L)}H{int(H):02d}"


def safe_mean(xs):
    x = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def safe_div(a, b):
    try:
        a, b = float(a), float(b)
    except Exception:
        return np.nan
    return a / b if abs(b) > EPS else np.nan


def spearman(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce")
    b = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def summarize_oracle_states(state_df, ks):
    require_cols(state_df, [
        "sid", "causal_rank", "head_layer", "head", "baseline_correct",
        "direction_accuracy", "spatial_percentile", "top10_hit", "top05_hit", "top01_exact"
    ], "state_df")
    rows = []
    for K in ks:
        q = state_df[state_df["causal_rank"] <= K]
        for (HL, h, bc), g in q.groupby(["head_layer", "head", "baseline_correct"]):
            rows.append({
                "K": int(K), "head_layer": int(HL), "head": int(h),
                "head_name": head_name(HL, h),
                "baseline_group": "correct" if bool(bc) else "wrong",
                "N_samples": int(g["sid"].nunique()), "N_states": int(len(g)),
                "direction_accuracy": safe_mean(g["direction_accuracy"]),
                "mean_spatial_percentile": safe_mean(g["spatial_percentile"]),
                "top10_rate": float(g["top10_hit"].mean()),
                "top05_rate": float(g["top05_hit"].mean()),
                "top01_rate": float(g["top01_exact"].mean()),
            })
        for (HL, h), g in q.groupby(["head_layer", "head"]):
            rows.append({
                "K": int(K), "head_layer": int(HL), "head": int(h),
                "head_name": head_name(HL, h), "baseline_group": "all",
                "N_samples": int(g["sid"].nunique()), "N_states": int(len(g)),
                "direction_accuracy": safe_mean(g["direction_accuracy"]),
                "mean_spatial_percentile": safe_mean(g["spatial_percentile"]),
                "top10_rate": float(g["top10_hit"].mean()),
                "top05_rate": float(g["top05_hit"].mean()),
                "top01_rate": float(g["top01_exact"].mean()),
            })
    return pd.DataFrame(rows)


def summarize_overlap(overlap_df):
    if not len(overlap_df):
        return pd.DataFrame()
    require_cols(overlap_df, [
        "sid", "K", "head_layer", "head", "baseline_correct", "budget_m",
        "intersection", "random_expected_recall", "direction_accuracy"
    ], "overlap_df")
    rows = []
    tmp = overlap_df.copy()
    tmp["baseline_group"] = np.where(tmp["baseline_correct"], "correct", "wrong")
    for (K, HL, h, group), g in tmp.groupby(["K", "head_layer", "head", "baseline_group"]):
        denom = float(g["budget_m"].sum())
        micro = safe_div(g["intersection"].sum(), denom)
        expected = safe_div(np.sum(g["random_expected_recall"] * g["budget_m"]), denom)
        rows.append({
            "K": int(K), "head_layer": int(HL), "head": int(h),
            "head_name": head_name(HL, h), "baseline_group": group,
            "N_samples": int(g["sid"].nunique()), "total_budget": int(denom),
            "micro_recall": micro, "random_expected_recall": expected,
            "enrichment_ratio": safe_div(micro, expected),
            "direction_accuracy": safe_mean(g["direction_accuracy"]),
        })
    for (K, HL, h), g in overlap_df.groupby(["K", "head_layer", "head"]):
        denom = float(g["budget_m"].sum())
        micro = safe_div(g["intersection"].sum(), denom)
        expected = safe_div(np.sum(g["random_expected_recall"] * g["budget_m"]), denom)
        rows.append({
            "K": int(K), "head_layer": int(HL), "head": int(h),
            "head_name": head_name(HL, h), "baseline_group": "all",
            "N_samples": int(g["sid"].nunique()), "total_budget": int(denom),
            "micro_recall": micro, "random_expected_recall": expected,
            "enrichment_ratio": safe_div(micro, expected),
            "direction_accuracy": safe_mean(g["direction_accuracy"]),
        })
    return pd.DataFrame(rows)


def head_accuracy_alignment_correlations(state_summary):
    if not len(state_summary):
        return pd.DataFrame()
    rows = []
    allg = state_summary[state_summary["baseline_group"] == "all"].copy()
    for (K, HL), g in allg.groupby(["K", "head_layer"]):
        rows.append({
            "K": int(K), "head_layer": int(HL), "n_heads": int(len(g)),
            "rho_directionAcc_vs_percentile": spearman(g["direction_accuracy"], g["mean_spatial_percentile"]),
            "rho_directionAcc_vs_top10": spearman(g["direction_accuracy"], g["top10_rate"]),
            "rho_directionAcc_vs_top01": spearman(g["direction_accuracy"], g["top01_rate"]),
        })
    for K, g in allg.groupby("K"):
        z = g.copy()
        z["acc_rank"] = z.groupby("head_layer")["direction_accuracy"].rank(pct=True)
        z["pct_rank"] = z.groupby("head_layer")["mean_spatial_percentile"].rank(pct=True)
        z["top10_rank"] = z.groupby("head_layer")["top10_rate"].rank(pct=True)
        rows.append({
            "K": int(K), "head_layer": -1, "n_heads": int(len(z)),
            "rho_directionAcc_vs_percentile": spearman(z["acc_rank"], z["pct_rank"]),
            "rho_directionAcc_vs_top10": spearman(z["acc_rank"], z["top10_rank"]),
            "rho_directionAcc_vs_top01": np.nan,
        })
    return pd.DataFrame(rows)


def selector_summary(candidate_df):
    score_cols = [
        "overall_heads_mean_percentile",
        "overall_heads_rankweighted_percentile",
        "relation_heads_mean_percentile",
        "relation_heads_rankweighted_percentile",
        "overall_heads_matched_overlap",
        "relation_heads_matched_overlap",
    ]
    require_cols(candidate_df, [
        "sid", "K", "candidate_relation", "gt", "baseline_prediction", "baseline_correct"
    ] + score_cols, "candidate_df")

    rows, details = [], []
    for (K, sid), g in candidate_df.groupby(["K", "sid"]):
        if set(g["candidate_relation"]) != set(REL):
            continue
        gt = str(g["gt"].iloc[0])
        bp = str(g["baseline_prediction"].iloc[0])
        bc = bool(g["baseline_correct"].iloc[0])

        for score_col in score_cols:
            gg = g[["candidate_relation", score_col]].dropna()
            if len(gg) < 4:
                continue
            winner = gg.loc[gg[score_col].idxmax()]
            pred = str(winner["candidate_relation"])
            vals = gg.sort_values(score_col, ascending=False)[score_col].to_numpy(float)
            d = {
                "sid": int(sid), "K": int(K), "score": score_col,
                "gt": gt, "baseline_prediction": bp, "baseline_correct": bc,
                "pred": pred, "pred_is_gt": pred == gt,
                "pred_matches_baseline": pred == bp,
                "winner_margin": float(vals[0] - vals[1]),
            }
            for r in REL:
                hit = gg[gg["candidate_relation"] == r]
                d[f"score_{r}"] = float(hit[score_col].iloc[0]) if len(hit) else np.nan
            details.append(d)

    detail_df = pd.DataFrame(details)
    if not len(detail_df):
        return pd.DataFrame(), detail_df

    for (K, score), g in detail_df.groupby(["K", "score"]):
        c = g["baseline_correct"]
        w = ~c
        rows.append({
            "K": int(K), "score": score, "N": int(len(g)),
            "GTacc_all": float(g["pred_is_gt"].mean()),
            "GTacc_correct": float(g.loc[c, "pred_is_gt"].mean()) if c.any() else np.nan,
            "GTacc_wrong": float(g.loc[w, "pred_is_gt"].mean()) if w.any() else np.nan,
            "matchBaseline_all": float(g["pred_matches_baseline"].mean()),
            "matchBaseline_wrong": float(g.loc[w, "pred_matches_baseline"].mean()) if w.any() else np.nan,
            "margin_correct": safe_mean(g.loc[c, "winner_margin"]),
            "margin_wrong": safe_mean(g.loc[w, "winner_margin"]),
        })
    return pd.DataFrame(rows).sort_values(["GTacc_wrong", "GTacc_all"], ascending=False), detail_df


def focus_head_report(state_summary, overlap_summary, focus_heads, ks):
    rows = []
    for L, H in focus_heads:
        hn = head_name(L, H)
        for K in ks:
            s = state_summary[
                (state_summary["head_layer"] == L)
                & (state_summary["head"] == H)
                & (state_summary["K"] == K)
                & (state_summary["baseline_group"] == "all")
            ]
            if len(overlap_summary):
                o = overlap_summary[
                    (overlap_summary["head_layer"] == L)
                    & (overlap_summary["head"] == H)
                    & (overlap_summary["K"] == K)
                    & (overlap_summary["baseline_group"] == "all")
                ]
            else:
                o = pd.DataFrame()
            if not len(s) and not len(o):
                continue
            rows.append({
                "head_name": hn, "K": int(K),
                "N_samples": int(s["N_samples"].iloc[0]) if len(s) else int(o["N_samples"].iloc[0]),
                "N_states": int(s["N_states"].iloc[0]) if len(s) else 0,
                "direction_accuracy": float(s["direction_accuracy"].iloc[0]) if len(s) else float(o["direction_accuracy"].iloc[0]),
                "mean_spatial_percentile": float(s["mean_spatial_percentile"].iloc[0]) if len(s) else np.nan,
                "top10_rate": float(s["top10_rate"].iloc[0]) if len(s) else np.nan,
                "top05_rate": float(s["top05_rate"].iloc[0]) if len(s) else np.nan,
                "top01_rate": float(s["top01_rate"].iloc[0]) if len(s) else np.nan,
                "matched_micro_recall": float(o["micro_recall"].iloc[0]) if len(o) else np.nan,
                "random_expected_recall": float(o["random_expected_recall"].iloc[0]) if len(o) else np.nan,
                "overlap_enrichment": float(o["enrichment_ratio"].iloc[0]) if len(o) else np.nan,
            })
    return pd.DataFrame(rows)


def fmt(x, n=3):
    return "nan" if not np.isfinite(float(x)) else f"{float(x):.{n}f}"


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    chunk_dir = outdir / "chunks"
    files = sorted(chunk_dir.glob("sid_*.pkl.gz"))
    if not files:
        raise FileNotFoundError(f"No cached chunks under {chunk_dir}")

    state_rows, overlap_rows, candidate_rows = [], [], []
    bad = []
    for f in files:
        try:
            obj = pd.read_pickle(f, compression="gzip")
            if not isinstance(obj, dict):
                raise TypeError(f"expected dict, got {type(obj).__name__}")
            state_rows.extend(obj.get("state", []))
            overlap_rows.extend(obj.get("overlap", []))
            candidate_rows.extend(obj.get("candidate", []))
        except Exception as e:
            bad.append({"file": str(f), "error": f"{type(e).__name__}: {e}"})

    state_df = pd.DataFrame(state_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    if not len(state_df):
        raise RuntimeError("Cached chunks contained no state rows")

    ks = parse_ints(args.ks)
    if args.selector_k not in ks:
        ks = sorted(set(ks + [args.selector_k]))
    focus_heads = [parse_head_name(x) for x in args.focus_heads.split(",") if x.strip()]

    state_summary = summarize_oracle_states(state_df, ks)
    overlap_summary = summarize_overlap(overlap_df)
    corr_df = head_accuracy_alignment_correlations(state_summary)
    selector_df, selector_detail = selector_summary(candidate_df)
    focus_df = focus_head_report(state_summary, overlap_summary, focus_heads, ks)

    state_df.to_csv(outdir / "oracle_causal_states_vs_head_source_rank.csv", index=False)
    state_summary.to_csv(outdir / "oracle_head_rank_convergence_summary.csv", index=False)
    overlap_df.to_csv(outdir / "oracle_matched_budget_overlap_detail.csv", index=False)
    overlap_summary.to_csv(outdir / "oracle_matched_budget_overlap_summary.csv", index=False)
    corr_df.to_csv(outdir / "direction_accuracy_vs_alignment_correlation.csv", index=False)
    candidate_df.to_csv(outdir / "fourway_candidate_crossmechanism_scores.csv", index=False)
    selector_df.to_csv(outdir / "fourway_selector_summary.csv", index=False)
    selector_detail.to_csv(outdir / "fourway_selector_detail.csv", index=False)
    focus_df.to_csv(outdir / "focus_heads_rankwise_convergence.csv", index=False)
    (outdir / "postprocess_errors.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")

    lines = []
    lines.append("=" * 150)
    lines.append("POSTPROCESS: CAUSAL K vs DIRECTION-HEAD SPATIAL SOURCE RANK")
    lines.append("=" * 150)
    lines.append(f"chunks={len(files)} | successful state SIDs={state_df['sid'].nunique()} | bad chunks={len(bad)}")
    lines.append("")
    lines.append("1) FOCUS HEADS")
    lines.append("-" * 150)
    for hn in [head_name(*h) for h in focus_heads]:
        g = focus_df[focus_df["head_name"] == hn].sort_values("K")
        if not len(g):
            lines.append(f"{hn}: no aligned states")
            continue
        lines.append(f"{hn}:")
        for _, r in g.iterrows():
            lines.append(
                f"  K={int(r['K']):2d} Nsample={int(r['N_samples']):2d} Nstate={int(r['N_states']):3d} | "
                f"pct={fmt(r['mean_spatial_percentile'])} Top10={fmt(r['top10_rate'])} "
                f"Top5={fmt(r['top05_rate'])} Top1={fmt(r['top01_rate'])} | "
                f"matchedRecall={fmt(r['matched_micro_recall'])} rand={fmt(r['random_expected_recall'])} "
                f"enrich={fmt(r['overlap_enrichment'],2)}x"
            )
    lines.append("")
    lines.append("2) WITHIN-LAYER HEAD ACCURACY vs ALIGNMENT")
    lines.append("-" * 150)
    for _, r in corr_df[corr_df["K"].isin([1,3,5,7,36])].iterrows():
        L = "POOLED" if int(r["head_layer"]) == -1 else f"L{int(r['head_layer']):02d}"
        lines.append(
            f"K={int(r['K']):2d} {L:<8s} nHeads={int(r['n_heads']):3d} | "
            f"rho(acc,pct)={fmt(r['rho_directionAcc_vs_percentile'])} "
            f"rho(acc,Top10)={fmt(r['rho_directionAcc_vs_top10'])}"
        )
    lines.append("")
    lines.append("3) FOUR-WAY SELECTOR")
    lines.append("-" * 150)
    if len(selector_df):
        z = selector_df[selector_df["K"].isin(sorted(set([3,5,7,args.selector_k])))]
        for _, r in z.sort_values(["GTacc_wrong", "GTacc_all"], ascending=False).head(30).iterrows():
            lines.append(
                f"K={int(r['K']):2d} {r['score']:<42s} | "
                f"GTacc all={fmt(r['GTacc_all'])} correct={fmt(r['GTacc_correct'])} "
                f"wrong={fmt(r['GTacc_wrong'])} | matchBaselineWrong={fmt(r['matchBaseline_wrong'])}"
            )
    else:
        lines.append("No complete 4-way candidate rows.")

    text = "\n".join(lines) + "\n"
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
