#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_reverse_spatial_summary_v1.py

Pure post-processing for eval_reverse_spatial_to_causal_core_all440_v1.py.

Why needed
----------
In v1, oracle_layer_budget has sample-dependent K (= number of selected oracle-
budget text states). The summary grouped by K, fragmenting the 440 samples into
rows such as N=3,29,102,161...

This script correctly aggregates oracle_layer_budget across ALL samples without
grouping by K, and additionally reports performance conditional on whether the
predicted relation was correct.

No model forward. No gradients. No cache regeneration.

Usage
-----
python -u fix_reverse_spatial_summary_v1.py \
  --run-dir output/qwen3b_reverse_spatial_to_causal_all440_v1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(s):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def aggregate(g):
    exact_hits = int(g["exact_hits"].sum())
    target_n = int(g["target_N"].sum())
    pos_hits = int(g["position_hits"].sum())
    target_pos_n = int(g["target_position_N"].sum())
    selected_n = int(g["selected_N"].sum())

    return {
        "N": int(g["sid"].nunique()),
        "mean_selected_N": safe_mean(g["selected_N"]),
        "relation_accuracy": safe_mean(g["relation_correct"].astype(float)),
        "micro_exact_recall": safe_div(exact_hits, target_n),
        "macro_exact_recall": safe_mean(g["exact_recall"]),
        "micro_exact_precision": safe_div(exact_hits, selected_n),
        "micro_position_recall": safe_div(pos_hits, target_pos_n),
        "macro_position_recall": safe_mean(g["position_recall"]),
        "micro_position_precision": safe_div(pos_hits, selected_n),
        "mean_exact_hits": safe_mean(g["exact_hits"]),
        "mean_position_hits": safe_mean(g["position_hits"]),
    }


def group_summary(df, keys):
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update(aggregate(g))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    root = Path(a.run_dir)
    p = root / "reverse_recovery_per_sample.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    if "relation_correct" in df.columns:
        df["relation_correct"] = (
            df["relation_correct"].astype(str).str.lower()
            .isin({"true", "1", "yes", "t"})
        )

    # 1) Correct full-N oracle-budget aggregation: DO NOT group by K.
    q = df[df["selection_mode"] == "oracle_layer_budget"].copy()
    oracle = group_summary(
        q,
        ["bank", "heads_per_layer", "relation_mode", "target"],
    )
    oracle = oracle.sort_values(
        ["target", "relation_mode", "micro_exact_recall"],
        ascending=[True, True, False],
    )
    oracle.to_csv(
        root / "corrected_oracle_layer_budget_summary.csv",
        index=False,
    )

    # 2) Fixed-K practical/global summaries.
    fixed = df[df["selection_mode"].isin(
        ["calib_layer_prior", "global_unique"]
    )].copy()
    fixed_summary = group_summary(
        fixed,
        [
            "bank",
            "heads_per_layer",
            "relation_mode",
            "selection_mode",
            "K",
            "target",
        ],
    )
    fixed_summary = fixed_summary.sort_values(
        ["target", "micro_exact_recall"],
        ascending=[True, False],
    )
    fixed_summary.to_csv(
        root / "corrected_fixedK_summary.csv",
        index=False,
    )

    # 3) Predicted-relation methods conditioned on routing correctness.
    pred = df[df["relation_mode"] == "pred_vote"].copy()
    cond = group_summary(
        pred,
        [
            "bank",
            "heads_per_layer",
            "selection_mode",
            "K",
            "target",
            "relation_correct",
        ],
    )
    cond = cond.sort_values(
        ["target", "selection_mode", "heads_per_layer", "K", "relation_correct"]
    )
    cond.to_csv(
        root / "recovery_conditioned_on_relation_correctness.csv",
        index=False,
    )

    # 4) Compact report.
    lines = []
    lines.append("=" * 165)
    lines.append("CORRECTED FULL-SAMPLE ORACLE-LAYER-BUDGET SUMMARY")
    lines.append("=" * 165)
    show = [
        "bank",
        "heads_per_layer",
        "relation_mode",
        "target",
        "N",
        "mean_selected_N",
        "relation_accuracy",
        "micro_exact_recall",
        "micro_exact_precision",
        "micro_position_recall",
        "micro_position_precision",
    ]
    lines.append(
        oracle[show].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    lines.append("")
    lines.append("=" * 165)
    lines.append("SYN-RG PRACTICAL: ROUTING-CORRECT VS ROUTING-WRONG")
    lines.append("=" * 165)

    c = cond[
        (cond["bank"] == "syn_rg")
        & (cond["selection_mode"] == "calib_layer_prior")
    ].copy()
    show2 = [
        "heads_per_layer",
        "K",
        "target",
        "relation_correct",
        "N",
        "micro_exact_recall",
        "micro_exact_precision",
        "micro_position_recall",
    ]
    lines.append(
        c[show2].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    report = "\n".join(lines) + "\n"
    (root / "corrected_analysis_summary.txt").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
