#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_qwen3b_sparse_bottleneck_v1.py

Plot Experiment 1: sparse causal bottleneck from an existing
eval_qwen_oracle_k36_prefix_curve_v1.py output directory.

Inputs
------
<run-dir>/summary.csv
<run-dir>/rank_mass_summary.csv

Outputs
-------
<output-dir>/qwen3b_sparse_causal_bottleneck_curve.png
<output-dir>/qwen3b_sparse_causal_bottleneck_curve.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        default="output/qwen3b_sparse_bottleneck_prefix_L31_35_denseK_all440_v1",
    )
    p.add_argument(
        "--output-dir",
        default="output/qwen3b_sparse_bottleneck_prefix_L31_35_denseK_all440_v1/figures",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    a = parse_args()

    run_dir = Path(a.run_dir)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.csv"
    mass_path = run_dir / "rank_mass_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not mass_path.exists():
        raise FileNotFoundError(mass_path)

    s = pd.read_csv(summary_path)
    m = pd.read_csv(mass_path)

    required_s = {"k", "baseline_acc", "edited_acc"}
    required_m = {"k", "mean_K36_mediation_mass_fraction"}

    miss_s = required_s - set(s.columns)
    miss_m = required_m - set(m.columns)

    if miss_s:
        raise RuntimeError(f"{summary_path} missing columns: {sorted(miss_s)}")
    if miss_m:
        raise RuntimeError(f"{mass_path} missing columns: {sorted(miss_m)}")

    s["k"] = pd.to_numeric(s["k"], errors="raise").astype(int)
    s["baseline_acc"] = pd.to_numeric(s["baseline_acc"], errors="raise")
    s["edited_acc"] = pd.to_numeric(s["edited_acc"], errors="raise")

    m["k"] = pd.to_numeric(m["k"], errors="raise").astype(int)
    m["mean_K36_mediation_mass_fraction"] = pd.to_numeric(
        m["mean_K36_mediation_mass_fraction"], errors="raise"
    )

    s = s.sort_values("k")
    m = m.sort_values("k")

    baseline = float(s["baseline_acc"].iloc[0])

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8.4, 3.25),
        constrained_layout=True,
    )

    # ---------------------------------------------------------
    # (a) Behavioral sufficiency
    # ---------------------------------------------------------
    ax = axes[0]

    ax.plot(
        s["k"],
        100.0 * s["edited_acc"],
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        label="Top-K causal states",
    )

    ax.axhline(
        100.0 * baseline,
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline ({100.0 * baseline:.1f}%)",
    )

    # Annotate only a few informative K values so the plot stays clean.
    annotate_k = {1, 4, 7, 20, int(s["k"].max())}
    for row in s.itertuples():
        if int(row.k) in annotate_k:
            ax.annotate(
                f"{100.0 * float(row.edited_acc):.1f}",
                (int(row.k), 100.0 * float(row.edited_acc)),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Number of causal states, K")
    ax.set_ylabel("Generation accuracy (%)")
    ax.set_title("(a) Behavioral sufficiency")
    ax.set_xticks(s["k"].tolist())
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ymin = min(100.0 * baseline, 100.0 * s["edited_acc"].min())
    ymax = 100.0 * s["edited_acc"].max()
    pad = max(1.5, 0.12 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + 2.0 * pad)

    # ---------------------------------------------------------
    # (b) Attribution concentration
    # ---------------------------------------------------------
    ax = axes[1]

    ax.plot(
        m["k"],
        100.0 * m["mean_K36_mediation_mass_fraction"],
        marker="o",
        linewidth=2.0,
        markersize=4.8,
    )

    annotate_k_mass = {1, 7, 20, int(m["k"].max())}
    for row in m.itertuples():
        if int(row.k) in annotate_k_mass:
            val = 100.0 * float(row.mean_K36_mediation_mass_fraction)
            ax.annotate(
                f"{val:.1f}",
                (int(row.k), val),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Number of causal states, K")
    ax.set_ylabel("Cumulative positive mediation mass (%)")
    ax.set_title("(b) Attribution concentration")
    ax.set_xticks(m["k"].tolist())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(0, 106)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Decision control is concentrated in a sparse causal core",
        fontsize=11,
    )

    png = out_dir / "qwen3b_sparse_causal_bottleneck_curve.png"
    pdf = out_dir / "qwen3b_sparse_causal_bottleneck_curve.pdf"

    fig.savefig(png, dpi=a.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # Also save a compact merged table for paper bookkeeping.
    merged = s.merge(
        m[["k", "mean_K36_mediation_mass_fraction"]],
        on="k",
        how="left",
    )
    merged["edited_acc_percent"] = 100.0 * merged["edited_acc"]
    merged["baseline_acc_percent"] = 100.0 * merged["baseline_acc"]
    merged["mediation_mass_percent"] = (
        100.0 * merged["mean_K36_mediation_mass_fraction"]
    )
    merged.to_csv(out_dir / "plot_values.csv", index=False)

    print("Saved:")
    print(" ", png)
    print(" ", pdf)
    print(" ", out_dir / "plot_values.csv")


if __name__ == "__main__":
    main()
