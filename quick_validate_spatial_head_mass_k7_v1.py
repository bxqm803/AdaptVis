#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST sanity check: do high-direction spatial heads preferentially attend to high-causal-M states?

This intentionally does ONLY the cheapest checks:
  1) oracle/GT writer only
  2) global-unique Top36 causal states per sample
  3) Top-N direction heads per layer (default N=5)
  4) same-layer last-query -> token attention mass
  5) K7 vs ranks 8-36, plus causal-rank Spearman
  6) high-direction heads vs bottom-N direction-head control
  7) baseline-correct / baseline-wrong split

No permutations, no full K sweep, no four-way selector, no expensive per-state Python loops.
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--direction-heads", required=True)
    p.add_argument("--top-head-count", type=int, default=5)
    p.add_argument("--max-rank", type=int, default=36)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def resolve_file(path_or_dir, filename):
    p = Path(path_or_dir)
    return p if p.is_file() else p / filename


def canon_rel(x):
    s = str(x).strip().lower()
    return {
        "left": "left", "right": "right",
        "above": "on", "on": "on",
        "below": "under", "under": "under",
    }.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def spearman(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    xr = x[ok].rank()
    yr = y[ok].rank()
    if xr.nunique() < 2 or yr.nunique() < 2:
        return np.nan
    return float(xr.corr(yr))


def global_unique_top36(g, max_rank):
    q = g.copy()
    q["mediation"] = pd.to_numeric(q["mediation"], errors="coerce")
    q = q[q["mediation"] > 0]
    if not len(q):
        return q
    idx = q.groupby("position")["mediation"].idxmax()
    q = q.loc[idx].sort_values("mediation", ascending=False).head(max_rank).copy()
    q["causal_rank"] = np.arange(1, len(q) + 1)
    return q


def head_col(h, kind):
    return f"last_attn_same_head{int(h):02d}_{kind}"


def summarize_split(df, split_name):
    z = df.copy()
    if split_name == "correct":
        z = z[z["baseline_correct"]]
    elif split_name == "wrong":
        z = z[~z["baseline_correct"]]

    rows = []
    for metric in ["top_real_mass", "top_delta_mass", "bottom_real_mass", "bottom_delta_mass"]:
        rhos = []
        for sid, g in z.groupby("sid"):
            r = spearman(g["causal_rank"], g[metric])
            if np.isfinite(r):
                rhos.append(r)

        top7 = z[z["causal_rank"] <= 7].groupby("sid")[metric].mean()
        tail = z[(z["causal_rank"] >= 8) & (z["causal_rank"] <= 36)].groupby("sid")[metric].mean()
        both = pd.concat([top7.rename("top7"), tail.rename("tail")], axis=1).dropna()
        diff = both["top7"] - both["tail"]

        rows.append({
            "split": split_name,
            "metric": metric,
            "N_samples_rankcorr": len(rhos),
            "mean_spearman_rank_vs_mass": float(np.mean(rhos)) if rhos else np.nan,
            "median_spearman_rank_vs_mass": float(np.median(rhos)) if rhos else np.nan,
            "fraction_negative_rankcorr": float(np.mean(np.array(rhos) < 0)) if rhos else np.nan,
            "N_samples_top7_tail": len(both),
            "top7_mean": float(both["top7"].mean()) if len(both) else np.nan,
            "rank8_36_mean": float(both["tail"].mean()) if len(both) else np.nan,
            "top7_minus_tail": float(diff.mean()) if len(diff) else np.nan,
            "fraction_top7_gt_tail": float(np.mean(diff > 0)) if len(diff) else np.nan,
        })
    return rows


def main():
    a = parse_args()
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    med_path = Path(a.fourway_dir) / "all_four_writer_mediation.pkl.gz"
    feat_path = resolve_file(a.feature_cache, "all_candidate_self_features.pkl.gz")
    head_path = resolve_file(a.direction_heads, "head_results.csv")

    print("Loading...")
    med = pd.read_pickle(med_path, compression="gzip")
    feat = pd.read_pickle(feat_path, compression="gzip")
    heads = pd.read_csv(head_path)

    for d in (med, feat):
        d["sid"] = pd.to_numeric(d["sid"], errors="raise").astype(int)
        d["source_layer"] = pd.to_numeric(d["source_layer"], errors="raise").astype(int)
        d["position"] = pd.to_numeric(d["position"], errors="raise").astype(int)

    med["gt"] = med["gt"].map(canon_rel)
    med["candidate_relation"] = med["candidate_relation"].map(canon_rel)
    med["baseline_correct"] = med["baseline_correct"].map(boolify)

    heads["layer"] = pd.to_numeric(heads["layer"], errors="raise").astype(int)
    heads["head"] = pd.to_numeric(heads["head"], errors="raise").astype(int)

    oracle = med[med["candidate_relation"] == med["gt"]].copy()

    print("Selecting global-unique Top36 per sample...")
    chunks = []
    for sid, g in oracle.groupby("sid"):
        q = global_unique_top36(g, a.max_rank)
        if len(q):
            chunks.append(q)
    ranked = pd.concat(chunks, ignore_index=True)

    top_by_layer = {}
    bottom_by_layer = {}
    for L, g in heads.groupby("layer"):
        gg = g.dropna(subset=["residual_accuracy_mean"]).sort_values(
            "residual_accuracy_mean", ascending=False
        )
        top_by_layer[int(L)] = gg.head(a.top_head_count)["head"].astype(int).tolist()
        bottom_by_layer[int(L)] = gg.tail(a.top_head_count)["head"].astype(int).tolist()

    needed_cols = ["sid", "source_layer", "position"]
    for hs in list(top_by_layer.values()) + list(bottom_by_layer.values()):
        for h in hs:
            for kind in ("real", "delta"):
                c = head_col(h, kind)
                if c in feat.columns and c not in needed_cols:
                    needed_cols.append(c)

    print(f"Merging only {len(needed_cols)-3} head-attention columns...")
    f = feat[needed_cols].drop_duplicates(["sid", "source_layer", "position"])
    x = ranked.merge(
        f,
        on=["sid", "source_layer", "position"],
        how="left",
        validate="many_to_one",
    )

    print("Computing Top-N vs Bottom-N spatial-head mass...")
    out_rows = []
    for r in x.itertuples(index=False):
        d = r._asdict()
        L = int(d["source_layer"])
        row = dict(d)
        for label, hb in [("top", top_by_layer), ("bottom", bottom_by_layer)]:
            hs = hb.get(L, [])
            for kind in ("real", "delta"):
                vals = []
                for h in hs:
                    c = head_col(h, kind)
                    v = d.get(c, np.nan)
                    if pd.notna(v):
                        vals.append(float(v))
                row[f"{label}_{kind}_mass"] = float(np.mean(vals)) if vals else np.nan
        out_rows.append(row)

    z = pd.DataFrame(out_rows)
    summary = []
    for split in ["all", "correct", "wrong"]:
        summary.extend(summarize_split(z, split))
    summary = pd.DataFrame(summary)

    control_rows = []
    for split in ["all", "correct", "wrong"]:
        g = z.copy()
        if split == "correct":
            g = g[g["baseline_correct"]]
        elif split == "wrong":
            g = g[~g["baseline_correct"]]
        g7 = g[g["causal_rank"] <= 7]
        per = g7.groupby("sid").agg(
            top_real=("top_real_mass", "mean"),
            bottom_real=("bottom_real_mass", "mean"),
            top_delta=("top_delta_mass", "mean"),
            bottom_delta=("bottom_delta_mass", "mean"),
        )
        for kind in ("real", "delta"):
            diff = per[f"top_{kind}"] - per[f"bottom_{kind}"]
            control_rows.append({
                "split": split,
                "kind": kind,
                "N": len(per),
                "top_direction_head_mass": float(per[f"top_{kind}"].mean()),
                "bottom_direction_head_mass": float(per[f"bottom_{kind}"].mean()),
                "top_minus_bottom": float(diff.mean()),
                "fraction_top_gt_bottom": float(np.mean(diff > 0)),
            })
    control = pd.DataFrame(control_rows)

    headcorr_rows = []
    top7 = ranked[ranked["causal_rank"] <= 7]
    for L, hg in heads.groupby("layer"):
        tg = top7[top7["source_layer"] == int(L)]
        if not len(tg):
            continue
        fs = feat[
            (feat["source_layer"] == int(L))
            & (feat["sid"].isin(tg["sid"].unique()))
        ]
        tkeys = tg[["sid", "source_layer", "position"]]
        m = tkeys.merge(
            fs,
            on=["sid", "source_layer", "position"],
            how="left",
        )
        for kind in ("real", "delta"):
            accs, masses = [], []
            for hr in hg.itertuples():
                c = head_col(hr.head, kind)
                if c not in m.columns:
                    continue
                mass = pd.to_numeric(m[c], errors="coerce").mean()
                if pd.notna(mass):
                    accs.append(float(hr.residual_accuracy_mean))
                    masses.append(float(mass))
            headcorr_rows.append({
                "layer": int(L),
                "kind": kind,
                "n_heads": len(accs),
                "spearman_directionAcc_vs_oracleTop7Mass": spearman(accs, masses),
            })
    headcorr = pd.DataFrame(headcorr_rows)

    z.to_csv(outdir / "oracle_top36_with_spatial_head_mass.csv", index=False)
    summary.to_csv(outdir / "quick_rank_vs_mass_summary.csv", index=False)
    control.to_csv(outdir / "quick_top_vs_bottom_head_control.csv", index=False)
    headcorr.to_csv(outdir / "quick_head_directionacc_vs_top7mass.csv", index=False)

    print()
    print("=" * 130)
    print("QUICK SPATIAL-HEAD / CAUSAL-K VALIDATION")
    print("=" * 130)
    print(f"N={z.sid.nunique()} | Top direction heads/layer={a.top_head_count}")
    print()

    print("A) CAUSAL RANK vs SPATIAL-HEAD MASS")
    print("-" * 130)
    for r in summary.itertuples():
        print(
            f"{r.split:<7s} {r.metric:<18s} | "
            f"rho(rank,mass)={r.mean_spearman_rank_vs_mass:+.3f} "
            f"P(rho<0)={r.fraction_negative_rankcorr:.3f} | "
            f"Top7={r.top7_mean:.5f} ranks8-36={r.rank8_36_mean:.5f} "
            f"Δ={r.top7_minus_tail:+.5f} "
            f"P(Top7>tail)={r.fraction_top7_gt_tail:.3f}"
        )

    print()
    print("B) TOP DIRECTION HEADS vs BOTTOM DIRECTION HEADS ON EXACT SAME K7")
    print("-" * 130)
    for r in control.itertuples():
        print(
            f"{r.split:<7s} {r.kind:<5s} | "
            f"top={r.top_direction_head_mass:.6f} "
            f"bottom={r.bottom_direction_head_mass:.6f} "
            f"Δ={r.top_minus_bottom:+.6f} "
            f"P(top>bottom)={r.fraction_top_gt_bottom:.3f}"
        )

    print()
    print("C) HEAD-LEVEL: DIRECTION ACCURACY vs ORACLE-TOP7 ATTENTION MASS")
    print("-" * 130)
    for r in headcorr.itertuples():
        print(
            f"L{int(r.layer):02d} {r.kind:<5s} nHeads={int(r.n_heads):2d} | "
            f"rho={r.spearman_directionAcc_vs_oracleTop7Mass:+.3f}"
        )

    print()
    print("Strong evidence = rank-vs-mass rho < 0, Top7 > ranks8-36,")
    print("                  AND top direction heads > bottom heads.")
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
