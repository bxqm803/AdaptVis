#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze whether forced K36 contains a smaller sample-specific causal core.

Pure post-processing:
- NO VLM forward
- NO new gradients
- Uses existing mediation_tokens.csv and selected_tokens.csv

It analyzes:
1) exact selected K36 concentration;
2) sample-specific core size by cumulative positive mediation mass;
3) relative-to-max mediation thresholds;
4) automatic elbow/knee;
5) same-position trajectories collapsed across L20..L26 to reduce adjacent-layer redundancy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--bundle", default="L20+L21+L22+L23+L24+L25+L26[global_unique]")
    p.add_argument("--k", type=int, default=36)
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--mass-thresholds", default="0.5,0.7,0.8,0.9,0.95")
    p.add_argument("--relative-thresholds", default="0.5,0.3,0.2,0.1,0.05")
    p.add_argument("--trajectory-score", choices=["peak", "sum_positive"], default="peak")
    p.add_argument("--positive-eps", type=float, default=0.0)
    p.add_argument("--elbow-min-k", type=int, default=2)
    p.add_argument("--elbow-max-k", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_float_list(s):
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("Threshold list is empty")
    return vals


def canon_rel(x):
    x = str(x).strip().lower()
    return {"above": "on", "below": "under"}.get(x, x)


def fmt_thr(x):
    return str(x).replace(".", "p")


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def knee_rank_from_decreasing_scores(scores, min_k=2, max_k=0):
    y = np.asarray(scores, dtype=np.float64)
    y = y[np.isfinite(y) & (y > 0)]
    n = len(y)
    if n == 0:
        return 0, float("nan")
    if n <= max(2, min_k):
        return n, 0.0

    if max_k and max_k > 0:
        y = y[: min(n, int(max_k))]
        n = len(y)

    x = np.linspace(0.0, 1.0, n)
    y0, y1 = float(y[0]), float(y[-1])
    if abs(y0 - y1) < 1e-12:
        return min(max(min_k, 1), n), 0.0

    yn = (y - y1) / (y0 - y1)
    line = 1.0 - x
    distance = yn - line

    start = max(0, int(min_k) - 1)
    start = min(start, n - 1)
    idx = start + int(np.argmax(distance[start:]))
    return idx + 1, float(distance[idx])


def cumulative_mass(scores):
    pos = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    total = float(pos.sum())
    if total <= 0:
        return np.full(len(pos), np.nan), total
    return np.cumsum(pos) / total, total


def first_rank_reaching(cum, threshold):
    c = np.asarray(cum, dtype=np.float64)
    idx = np.where(np.isfinite(c) & (c >= threshold))[0]
    return int(idx[0] + 1) if len(idx) else 0


def load_tables(a):
    run_dir = Path(a.run_dir)
    med_path = run_dir / "mediation_tokens.csv"
    sel_path = run_dir / "selected_tokens.csv"
    if not med_path.exists():
        raise FileNotFoundError(med_path)
    if not sel_path.exists():
        raise FileNotFoundError(sel_path)

    med = pd.read_csv(med_path)
    sel = pd.read_csv(sel_path)

    req_med = {
        "sid","relation","source_layer","position","token",
        "category","broad_category","mediation"
    }
    req_sel = req_med | {"condition","selection_strategy","source_bundle","k"}

    m1 = sorted(req_med - set(med.columns))
    m2 = sorted(req_sel - set(sel.columns))
    if m1:
        raise RuntimeError("mediation_tokens.csv missing: " + ", ".join(m1))
    if m2:
        raise RuntimeError("selected_tokens.csv missing: " + ", ".join(m2))

    for df in [med, sel]:
        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
        df["source_layer"] = pd.to_numeric(df["source_layer"], errors="raise").astype(int)
        df["position"] = pd.to_numeric(df["position"], errors="raise").astype(int)
        df["mediation"] = pd.to_numeric(df["mediation"], errors="coerce")
        df["relation"] = df["relation"].map(canon_rel)
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)

    filt = (
        (sel["condition"].astype(str) == str(a.condition))
        & (sel["selection_strategy"].astype(str) == str(a.selection_strategy))
        & (sel["source_bundle"].astype(str) == str(a.bundle))
        & (sel["k"] == int(a.k))
    )
    sel = sel[filt].copy()
    if not len(sel):
        avail = pd.read_csv(sel_path)[
            ["condition","selection_strategy","source_bundle","k"]
        ].drop_duplicates().head(50)
        raise RuntimeError(
            "No selected rows matched requested configuration.\n"
            "Available examples:\n" + avail.to_string(index=False)
        )

    source_layers = sorted(sel["source_layer"].unique().tolist())
    sids = sorted(sel["sid"].unique().tolist())

    med = med[
        med["sid"].isin(sids) & med["source_layer"].isin(source_layers)
    ].copy()

    med = (
        med.sort_values(
            ["sid","source_layer","position","mediation"],
            ascending=[True,True,True,False]
        )
        .drop_duplicates(["sid","source_layer","position"], keep="first")
    )
    sel = (
        sel.sort_values(
            ["sid","source_layer","position","mediation"],
            ascending=[True,True,True,False]
        )
        .drop_duplicates(["sid","source_layer","position"], keep="first")
    )
    return med, sel, source_layers, sids


def analyze_selected(sel, mass_thresholds, relative_thresholds, a):
    chunks, sample_rows = [], []

    for sid, g0 in sel.groupby("sid", sort=True):
        g = g0.sort_values("mediation", ascending=False).reset_index(drop=True).copy()
        g["analysis_rank_selected"] = np.arange(1, len(g) + 1)

        m = g["mediation"].to_numpy(np.float64)
        pos = np.where(m > a.positive_eps, np.maximum(m, 0.0), 0.0)
        posmax = float(pos.max()) if np.any(pos > 0) else float("nan")

        g["positive_mediation"] = pos
        g["m_over_positive_max"] = (
            pos / posmax if np.isfinite(posmax) and posmax > 0 else np.nan
        )

        cum, total_pos = cumulative_mass(pos)
        g["cum_positive_mass_selectedK"] = cum

        for t in mass_thresholds:
            kk = first_rank_reaching(cum, t)
            g[f"in_selected_mass_core_{fmt_thr(t)}"] = (
                g["analysis_rank_selected"] <= kk
            ) if kk > 0 else False

        for t in relative_thresholds:
            g[f"in_selected_relative_core_{fmt_thr(t)}"] = (
                g["m_over_positive_max"] >= t
            )

        positive_scores = np.sort(m[m > a.positive_eps])[::-1]
        elbow_k, elbow_strength = knee_rank_from_decreasing_scores(
            positive_scores, a.elbow_min_k, a.elbow_max_k
        )
        g["in_selected_elbow_core"] = (
            (g["analysis_rank_selected"] <= elbow_k)
            & (g["mediation"] > a.positive_eps)
        )
        chunks.append(g)

        row = {
            "sid": int(sid),
            "relation": g["relation"].iloc[0],
            "selected_N": int(len(g)),
            "selected_positive_N": int((m > a.positive_eps).sum()),
            "selected_nonpositive_N": int((m <= a.positive_eps).sum()),
            "selected_positive_mass": total_pos,
            "selected_M_max": float(np.nanmax(m)),
            "selected_M_mean": float(np.nanmean(m)),
            "selected_M_median": float(np.nanmedian(m)),
            "selected_elbow_k": int(elbow_k),
            "selected_elbow_strength": elbow_strength,
            "selected_elbow_mass": (
                float(cum[elbow_k-1])
                if elbow_k > 0 and len(cum) >= elbow_k and np.isfinite(cum[elbow_k-1])
                else float("nan")
            ),
        }

        for t in mass_thresholds:
            row[f"selected_k_for_mass_{fmt_thr(t)}"] = first_rank_reaching(cum, t)

        for t in relative_thresholds:
            col = g["m_over_positive_max"].to_numpy(np.float64)
            n = int(np.sum(np.isfinite(col) & (col >= t)))
            row[f"selected_n_relative_ge_{fmt_thr(t)}"] = n
            row[f"selected_frac_relative_ge_{fmt_thr(t)}"] = safe_div(n, len(g))

        sample_rows.append(row)

    return pd.concat(chunks, ignore_index=True), pd.DataFrame(sample_rows)


def build_trajectories(med, sel, source_layers, a):
    selected_map = {}
    for sid, g in sel.groupby("sid"):
        selected_map[int(sid)] = {
            int(r.position): (int(r.source_layer), float(r.mediation))
            for r in g.itertuples()
        }

    rows = []
    for (sid, pos), g0 in med.groupby(["sid","position"], sort=False):
        g = g0.sort_values("source_layer").copy()
        m = g["mediation"].to_numpy(np.float64)
        layers = g["source_layer"].to_numpy(int)
        finite = np.isfinite(m)
        if not finite.any():
            continue

        mf = np.where(finite, m, -np.inf)
        peak_idx = int(np.argmax(mf))
        peak = g.iloc[peak_idx]

        positive = finite & (m > a.positive_eps)
        posm = np.where(positive, np.maximum(m, 0.0), 0.0)
        pos_layers = layers[positive]

        if len(pos_layers):
            span = int(pos_layers.max() - pos_layers.min() + 1)
            count = int(len(pos_layers))
            contig = safe_div(count, span)
        else:
            span, count, contig = 0, 0, float("nan")

        chosen = selected_map.get(int(sid), {}).get(int(pos))

        row = {
            "sid": int(sid),
            "relation": canon_rel(peak["relation"]),
            "position": int(pos),
            "token_at_peak": str(peak["token"]),
            "category_at_peak": str(peak["category"]),
            "broad_category_at_peak": str(peak["broad_category"]),
            "peak_layer": int(peak["source_layer"]),
            "peak_mediation": float(peak["mediation"]),
            "sum_positive_mediation": float(posm.sum()),
            "mean_positive_mediation": float(posm[positive].mean()) if positive.any() else 0.0,
            "positive_layer_count": count,
            "positive_layer_span": span,
            "positive_layer_contiguity": contig,
            "selected_in_K": bool(chosen is not None),
            "selected_layer_in_K": chosen[0] if chosen else np.nan,
            "selected_mediation_in_K": chosen[1] if chosen else np.nan,
        }

        by_layer = {
            int(r.source_layer): float(r.mediation)
            for r in g.itertuples()
        }
        for L in source_layers:
            row[f"M_L{L}"] = by_layer.get(int(L), np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_trajectory(traj, mass_thresholds, relative_thresholds, a):
    chunks, sample_rows = [], []
    score_col = "peak_mediation" if a.trajectory_score == "peak" else "sum_positive_mediation"

    for sid, g0 in traj.groupby("sid", sort=True):
        g = g0.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
        g["trajectory_rank"] = np.arange(1, len(g) + 1)

        s = pd.to_numeric(g[score_col], errors="coerce").to_numpy(np.float64)
        pos = np.where(s > a.positive_eps, np.maximum(s, 0.0), 0.0)
        posmax = float(pos.max()) if np.any(pos > 0) else float("nan")

        g["trajectory_positive_score"] = pos
        g["trajectory_score_over_max"] = (
            pos / posmax if np.isfinite(posmax) and posmax > 0 else np.nan
        )
        cum, total_pos = cumulative_mass(pos)
        g["cum_positive_trajectory_mass"] = cum

        for t in mass_thresholds:
            kk = first_rank_reaching(cum, t)
            g[f"in_trajectory_mass_core_{fmt_thr(t)}"] = (
                g["trajectory_rank"] <= kk
            ) if kk > 0 else False

        for t in relative_thresholds:
            g[f"in_trajectory_relative_core_{fmt_thr(t)}"] = (
                g["trajectory_score_over_max"] >= t
            )

        positive_scores = np.sort(s[s > a.positive_eps])[::-1]
        elbow_k, elbow_strength = knee_rank_from_decreasing_scores(
            positive_scores, a.elbow_min_k, a.elbow_max_k
        )
        g["in_trajectory_elbow_core"] = (
            (g["trajectory_rank"] <= elbow_k)
            & (g[score_col] > a.positive_eps)
        )
        chunks.append(g)

        selected = g["selected_in_K"].astype(bool).to_numpy()

        row = {
            "sid": int(sid),
            "relation": g["relation"].iloc[0],
            "trajectory_N": int(len(g)),
            "trajectory_positive_N": int((s > a.positive_eps).sum()),
            "trajectory_positive_mass": total_pos,
            "trajectory_score_max": posmax,
            "trajectory_elbow_k": int(elbow_k),
            "trajectory_elbow_strength": elbow_strength,
            "trajectory_elbow_mass": (
                float(cum[elbow_k-1])
                if elbow_k > 0 and len(cum) >= elbow_k and np.isfinite(cum[elbow_k-1])
                else float("nan")
            ),
            "selectedK_share_of_all_trajectory_positive_mass": safe_div(
                float(pos[selected].sum()), total_pos
            ),
        }

        for t in mass_thresholds:
            row[f"trajectory_k_for_mass_{fmt_thr(t)}"] = first_rank_reaching(cum, t)
            core = g[f"in_trajectory_mass_core_{fmt_thr(t)}"].astype(bool).to_numpy()
            row[f"selectedK_recall_of_trajectory_mass_{fmt_thr(t)}_core"] = safe_div(
                int(np.sum(selected & core)), int(np.sum(core))
            )

        for t in relative_thresholds:
            col = g["trajectory_score_over_max"].to_numpy(np.float64)
            n = int(np.sum(np.isfinite(col) & (col >= t)))
            row[f"trajectory_n_relative_ge_{fmt_thr(t)}"] = n

        sample_rows.append(row)

    return pd.concat(chunks, ignore_index=True), pd.DataFrame(sample_rows)


def distribution_table(df):
    rows = []
    numeric_cols = [
        c for c in df.columns
        if c not in {"sid","relation"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    groups = [("ALL", df)]
    for rel in ["left","right","on","under"]:
        g = df[df["relation"] == rel]
        if len(g):
            groups.append((rel, g))

    for name, g in groups:
        for c in numeric_cols:
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            if not len(x):
                continue
            rows.append({
                "group": name, "metric": c, "N": len(x),
                "mean": float(x.mean()),
                "std": float(x.std(ddof=0)),
                "p10": float(x.quantile(.10)),
                "p25": float(x.quantile(.25)),
                "median": float(x.median()),
                "p75": float(x.quantile(.75)),
                "p90": float(x.quantile(.90)),
                "min": float(x.min()),
                "max": float(x.max()),
            })
    return pd.DataFrame(rows)


def render_summary(sel_sum, traj_sum, sel_rows, mass_thresholds, relative_thresholds, layers, a):
    lines = []
    lines.append("=" * 112)
    lines.append("DYNAMIC K36 CAUSAL-CORE / TRAJECTORY DIAGNOSTIC")
    lines.append("=" * 112)
    lines.append(f"N samples={sel_sum['sid'].nunique()} | source layers={layers} | forced K={a.k}")
    lines.append("")

    lines.append("A) Exact selected-K concentration")
    lines.append(
        f"  positive selected tokens/sample: mean={sel_sum['selected_positive_N'].mean():.2f}, "
        f"median={sel_sum['selected_positive_N'].median():.1f}"
    )
    for t in mass_thresholds:
        c = f"selected_k_for_mass_{fmt_thr(t)}"
        lines.append(
            f"  K for {int(round(100*t))}% selected positive M: "
            f"mean={sel_sum[c].mean():.2f}, median={sel_sum[c].median():.1f}, "
            f"p25={sel_sum[c].quantile(.25):.1f}, p75={sel_sum[c].quantile(.75):.1f}"
        )
    lines.append(
        f"  elbow K: mean={sel_sum['selected_elbow_k'].mean():.2f}, "
        f"median={sel_sum['selected_elbow_k'].median():.1f}"
    )
    lines.append(
        f"  M mass at elbow: mean={sel_sum['selected_elbow_mass'].mean():.3f}, "
        f"median={sel_sum['selected_elbow_mass'].median():.3f}"
    )

    lines.append("")
    lines.append("  Relative-to-sample-max selected core sizes:")
    for t in relative_thresholds:
        c = f"selected_n_relative_ge_{fmt_thr(t)}"
        lines.append(
            f"    M/Mmax >= {t:.2f}: mean N={sel_sum[c].mean():.2f}, "
            f"median N={sel_sum[c].median():.1f}"
        )

    lines.append("")
    lines.append("B) Same-position trajectories collapsed across layers")
    lines.append(
        f"  positive independent positions/sample: mean={traj_sum['trajectory_positive_N'].mean():.2f}, "
        f"median={traj_sum['trajectory_positive_N'].median():.1f}"
    )
    for t in mass_thresholds:
        c = f"trajectory_k_for_mass_{fmt_thr(t)}"
        lines.append(
            f"  positions for {int(round(100*t))}% positive trajectory-{a.trajectory_score} mass: "
            f"mean={traj_sum[c].mean():.2f}, median={traj_sum[c].median():.1f}, "
            f"p25={traj_sum[c].quantile(.25):.1f}, p75={traj_sum[c].quantile(.75):.1f}"
        )
    lines.append(
        f"  trajectory elbow K: mean={traj_sum['trajectory_elbow_k'].mean():.2f}, "
        f"median={traj_sum['trajectory_elbow_k'].median():.1f}"
    )
    lines.append(
        "  selected K share of ALL positive trajectory mass: "
        f"mean={traj_sum['selectedK_share_of_all_trajectory_positive_mass'].mean():.3f}, "
        f"median={traj_sum['selectedK_share_of_all_trajectory_positive_mass'].median():.3f}"
    )

    lines.append("")
    lines.append("C) Forced-K weak-tail estimate")
    for t in [0.30,0.20,0.10,0.05]:
        c = f"in_selected_relative_core_{fmt_thr(t)}"
        if c in sel_rows.columns:
            per_sid = sel_rows.groupby("sid")[c].mean()
            weak = 1.0 - per_sid
            lines.append(
                f"  fraction of selected K with M/Mmax < {t:.2f}: "
                f"mean={weak.mean():.3f}, median={weak.median():.3f}"
            )

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  If 80-90% of positive mediation mass needs far fewer than K=36 tokens, "
        "then exact forced-K membership is a noisy target for correlation analysis."
    )
    lines.append(
        "  If trajectory-level core is smaller again, adjacent layers contain redundant "
        "versions of the same token-position carrier."
    )
    lines.append(
        "  Then re-test attention/hidden-state predictors against mass80/mass90 or "
        "relative-M core labels, preferably after trajectory collapse."
    )
    return "\n".join(lines) + "\n"


def main():
    a = parse_args()
    mass_thresholds = parse_float_list(a.mass_thresholds)
    relative_thresholds = parse_float_list(a.relative_thresholds)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Output dir is non-empty: {outdir}; use --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)

    med, sel, layers, sids = load_tables(a)

    selected_rows, selected_summary = analyze_selected(
        sel, mass_thresholds, relative_thresholds, a
    )
    traj_raw = build_trajectories(med, sel, layers, a)
    traj_rows, traj_summary = analyze_trajectory(
        traj_raw, mass_thresholds, relative_thresholds, a
    )

    merged = selected_summary.merge(
        traj_summary, on=["sid","relation"], how="outer"
    )

    selected_rows.to_csv(outdir/"selectedK_ranked_core_labels.csv", index=False)
    selected_summary.to_csv(outdir/"selectedK_sample_core_summary.csv", index=False)
    traj_rows.to_csv(outdir/"trajectory_ranked_core_labels.csv", index=False)
    traj_summary.to_csv(outdir/"trajectory_sample_core_summary.csv", index=False)
    merged.to_csv(outdir/"sample_core_summary_merged.csv", index=False)
    distribution_table(selected_summary).to_csv(
        outdir/"selectedK_core_distribution.csv", index=False
    )
    distribution_table(traj_summary).to_csv(
        outdir/"trajectory_core_distribution.csv", index=False
    )

    # Aggregate layer-role composition for selected cores.
    for t in mass_thresholds:
        c = f"in_selected_mass_core_{fmt_thr(t)}"
        q = selected_rows[selected_rows[c]].copy()
        if len(q):
            (
                q.groupby(["source_layer","broad_category"], as_index=False)
                .agg(
                    token_count=("sid","size"),
                    sample_presence_N=("sid","nunique"),
                    mean_mediation=("mediation","mean"),
                    median_mediation=("mediation","median"),
                )
                .sort_values("token_count", ascending=False)
                .to_csv(outdir/f"selected_mass{int(round(100*t))}_layer_role.csv", index=False)
            )

    for t in relative_thresholds:
        c = f"in_selected_relative_core_{fmt_thr(t)}"
        q = selected_rows[selected_rows[c]].copy()
        if len(q):
            (
                q.groupby(["source_layer","broad_category"], as_index=False)
                .agg(
                    token_count=("sid","size"),
                    sample_presence_N=("sid","nunique"),
                    mean_mediation=("mediation","mean"),
                    median_mediation=("mediation","median"),
                )
                .sort_values("token_count", ascending=False)
                .to_csv(outdir/f"selected_relative_{fmt_thr(t)}_layer_role.csv", index=False)
            )

    summary = render_summary(
        selected_summary, traj_summary, selected_rows,
        mass_thresholds, relative_thresholds, layers, a
    )
    (outdir/"analysis_summary.txt").write_text(summary, encoding="utf-8")

    metadata = {
        "run_dir": str(a.run_dir),
        "bundle": a.bundle,
        "k": a.k,
        "condition": a.condition,
        "selection_strategy": a.selection_strategy,
        "N_samples": len(sids),
        "source_layers": [int(x) for x in layers],
        "mass_thresholds": mass_thresholds,
        "relative_thresholds": relative_thresholds,
        "trajectory_score": a.trajectory_score,
        "note": "Exploratory post-processing only; thresholds are not proof of necessity.",
    }
    (outdir/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(summary)
    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    main()
