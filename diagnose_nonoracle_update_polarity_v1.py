#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_nonoracle_update_polarity_v1.py

Goal
====
We already found that a GLOBAL fixed token scaffold can work surprisingly well.
This script asks the remaining question:

    Can the model decide, sample by sample, whether the REAL update written
    into a fixed scaffold position at layer L should be treated as
    POSITIVE or NEGATIVE — WITHOUT using GT to choose the relation?

NO generation is run here.  This is a fast polarity diagnostic.

Setup
=====
Use the completed global-fixed scaffold run:

    <global-run-dir>/resolved_global_slots_per_sample.csv
    <global-run-dir>/per_real_update_decision_score.csv

The fixed WHERE is therefore held constant.  For every evaluation sample and
every resolved global slot p, recapture:

    a_REAL[L,p] = h_REAL[L,p] - h_REAL[L-1,p]

for update layers (default L20..L26).

For each of the four candidate answers r in
    left / right / above / below
compute a sequence score and its gradient:

    S_r
    g_r[L,p] = d S_r / d h_REAL[L,p]

Then define the local response of the ACTUAL update to candidate r:

    q_r[L,p] = < a_REAL[L,p], g_r[L,p] >

Oracle label
============
The oracle behavioral sign is exactly the previous definition:

    competitor_GT = strongest non-GT candidate by clean sequence score

    B_oracle[L,p]
      = q_GT[L,p] - q_competitor_GT[L,p]

    y_oracle = sign(B_oracle)

This is used ONLY as the evaluation label.

Non-oracle polarity rules
=========================

1) sequence_winner
------------------
Use the model's own clean sequence-score winner:

    r_hat = argmax_r S_r
    foil  = argmax_{r != r_hat} S_r

    B_hat = q_r_hat - q_foil

No GT.  This asks whether the model can judge its own update from its own final
answer preference.  It is expected to be weak on baseline-wrong samples, but is
an important control.

2) baseline_generation
----------------------
Use the model's actual baseline generated relation from the prior run:

    r_hat = baseline_generation_prediction
    foil  = strongest clean sequence-score candidate != r_hat

Again no GT.

3) direction_head
-----------------
Read a frozen Synthetic-400 -> COCO Direction-Head selector CSV, normally:

    output/qwen3b_direction_selector_syn400_to_coco440/
        synthetic400_to_coco440_predictions.csv

Default relation:
    pred_top10_equal

with sample-level confidence:
    confidence_top10_equal

and margin:
    margin_top10_equal

Then:

    r_hat = Direction-Head consensus relation
    foil  = strongest clean sequence-score candidate != r_hat

    B_hat = q_r_hat - q_foil

This is the main training-free candidate.

4) direction_head with abstention
---------------------------------
For several selector-confidence / selector-margin thresholds:

    confident sample -> use direction_head polarity
    low-confidence    -> ABSTAIN (do not edit)

We report:
    coverage
    sign accuracy on covered updates
    sign accuracy on baseline-wrong samples
    |B_oracle|-weighted sign accuracy
    covered oracle decision-mass fraction

The last metric matters because making mistakes on a tiny |B| update is less
important than getting a large causal update wrong.

5) leave-one-sample-out fixed cell majority (diagnostic only)
-------------------------------------------------------------
For each (canonical_slot, layer), predict the sign from OTHER samples' oracle
sign majority.  The current sample is excluded.

This is NOT a deployable non-oracle method because it uses oracle labels from
other evaluation samples.  It is only a diagnostic:

    if LOO cell-majority is already very high,
        sign is mostly slot/layer-stable;
    if it is near chance but Direction Heads work,
        sample-specific internal relation information is genuinely useful.

Recommended N=80 run
====================
CUDA_VISIBLE_DEVICES=0 python -u diagnose_nonoracle_update_polarity_v1.py \
  --model qwen-3b \
  --global-run-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --selector-csv output/qwen3b_direction_selector_syn400_to_coco440/synthetic400_to_coco440_predictions.csv \
  --selector-method top10_equal \
  --update-layers 20-26 \
  --margin-thresholds 0,0.02,0.05,0.10,0.15,0.20 \
  --confidence-thresholds 0,0.30,0.35,0.40,0.50,0.60 \
  --output-dir output/qwen3b_global_fixed_polarity_diag_n80_v1 \
  --overwrite

Main outputs
============
per_update_sign_predictions.csv
    One row per sample x fixed slot x layer.
    Contains q_left/q_right/q_above/q_below, oracle B/sign, and every predicted sign.

route_summary.csv
    Relation-routing accuracy of sequence winner / baseline generation /
    Direction Heads, split baseline-correct vs baseline-wrong.

sign_summary.csv
    Sign accuracy for the three non-oracle rules + LOO cell-majority.

sign_by_layer.csv
sign_by_slot_layer.csv
    Where polarity is easy/hard.

direction_abstention_summary.csv
    Direction-head confidence/margin threshold sweep.

oracle_cell_stability.csv
    How intrinsically stable each fixed slot/layer sign is across samples.

analysis_summary.txt
metadata.json
errors.jsonl

Interpretation target
=====================
The most important numbers are:

    direction_head sign_accuracy on baseline-WRONG
    direction_head weighted_sign_accuracy on baseline-WRONG

and the abstention curve.

If Direction Heads beat ~50% clearly on baseline-wrong updates, especially on
|B|-weighted accuracy, then we have evidence that the model's own mid-layer
spatial representation can judge whether a later fixed-scaffold write supports
or conflicts with the spatial decision.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument(
        "--global-run-dir",
        required=True,
        help="Completed eval_global_fixed_l26_top10_trajectory_gating_v1.py output dir.",
    )
    p.add_argument(
        "--prior-real-update-dir",
        required=True,
        help="Completed eval_real_causal_token_update_gating_v1.py output dir.",
    )

    p.add_argument(
        "--selector-csv",
        default=(
            "output/qwen3b_direction_selector_syn400_to_coco440/"
            "synthetic400_to_coco440_predictions.csv"
        ),
    )
    p.add_argument(
        "--selector-method",
        default="top10_equal",
        help=(
            "Method suffix in selector CSV, e.g. top10_equal, "
            "top10_weighted, top10_equal_layerbalanced."
        ),
    )

    p.add_argument("--update-layers", default="20-26")
    p.add_argument(
        "--margin-thresholds",
        default="0,0.02,0.05,0.10,0.15,0.20",
    )
    p.add_argument(
        "--confidence-thresholds",
        default="0,0.30,0.35,0.40,0.50,0.60",
    )
    p.add_argument(
        "--oracle-sign-eps",
        type=float,
        default=1e-10,
        help="Oracle |B| <= eps is treated as neutral and excluded from sign accuracy.",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all SIDs present in resolved_global_slots_per_sample.csv.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    out = set()
    for x in str(text).split(","):
        x = x.strip().upper().replace("L", "")
        if not x:
            continue
        if "-" in x:
            a, b = x.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(x))
    return sorted(out)


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    aliases = {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "bottom": "below",
    }
    return aliases.get(s, s)


def sign3(x, eps=1e-10):
    x = float(x)
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def safe_mean(xs):
    vals = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def weighted_accuracy(correct, weights):
    c = np.asarray(correct, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(c) & np.isfinite(w) & (w >= 0)
    if not np.any(ok):
        return float("nan")
    denom = float(w[ok].sum())
    if denom <= EPS:
        return float("nan")
    return float(np.sum(c[ok] * w[ok]) / denom)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Inputs
# =============================================================================

def load_global_run(global_dir: Path, update_layers):
    resolved_path = global_dir / "resolved_global_slots_per_sample.csv"
    oracle_path = global_dir / "per_real_update_decision_score.csv"
    global_top_path = global_dir / "global_topk_slots.csv"

    for p in (resolved_path, oracle_path):
        if not p.exists():
            raise FileNotFoundError(p)

    resolved = pd.read_csv(resolved_path)
    oracle = pd.read_csv(oracle_path)

    resolved["sid"] = pd.to_numeric(resolved["sid"], errors="raise").astype(int)
    resolved["position"] = pd.to_numeric(
        resolved["position"], errors="coerce"
    ).fillna(-1).astype(int)

    if "resolved" in resolved.columns:
        if resolved["resolved"].dtype != bool:
            resolved["resolved"] = (
                resolved["resolved"].astype(str).str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )
        resolved = resolved[resolved["resolved"]].copy()

    resolved = resolved[resolved["position"] >= 0].copy()

    oracle["sid"] = pd.to_numeric(oracle["sid"], errors="raise").astype(int)
    oracle["update_layer"] = pd.to_numeric(
        oracle["update_layer"], errors="raise"
    ).astype(int)
    oracle["real_position"] = pd.to_numeric(
        oracle["real_position"], errors="raise"
    ).astype(int)
    oracle = oracle[oracle["update_layer"].isin(set(update_layers))].copy()

    if "gt" in oracle.columns:
        oracle["gt"] = oracle["gt"].map(canon_rel)

    global_top = pd.DataFrame()
    if global_top_path.exists():
        global_top = pd.read_csv(global_top_path)

    return resolved, oracle, global_top


def load_selector(path: Path, method: str):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "sid" not in df.columns:
        raise RuntimeError(f"{path} has no sid column")

    pred_col = f"pred_{method}"
    conf_col = f"confidence_{method}"
    margin_col = f"margin_{method}"

    if pred_col not in df.columns:
        raise RuntimeError(
            f"{path} missing {pred_col}; available columns include:\n"
            + ", ".join(df.columns[:80])
        )

    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
    df[pred_col] = df[pred_col].map(canon_rel)

    if conf_col not in df.columns:
        df[conf_col] = np.nan
    if margin_col not in df.columns:
        df[margin_col] = np.nan

    keep = ["sid", pred_col, conf_col, margin_col]
    if "gt" in df.columns:
        df["selector_gt"] = df["gt"].map(canon_rel)
        keep.append("selector_gt")

    return (
        df[keep].drop_duplicates("sid"),
        pred_col,
        conf_col,
        margin_col,
    )


def build_specs_for_sid(resolved_sid: pd.DataFrame, selector_layer: int):
    specs = []
    seen_pos = set()

    for r in resolved_sid.sort_values("global_rank").itertuples():
        p = int(r.position)
        if p in seen_pos:
            continue
        seen_pos.add(p)

        specs.append(
            {
                "real_position": p,
                "max_target_layer": int(selector_layer),
                "min_target_layer": int(selector_layer),
                "best_rank": int(getattr(r, "global_rank", len(specs) + 1)),
                "token": str(getattr(r, "token", "")),
                "category": str(getattr(r, "category", "")),
                "broad_category": str(getattr(r, "broad_category", "")),
                "canonical_slot": str(getattr(r, "canonical_slot", "")),
                "global_rank": int(getattr(r, "global_rank", len(specs) + 1)),
            }
        )
    return specs


# =============================================================================
# Polarity
# =============================================================================

def response_for_relation(entry, grad_by_relation, rel):
    L = int(entry["update_layer"])
    p = int(entry["real_position"])
    g = grad_by_relation[rel].get(L, None)
    if g is None or not (0 <= p < g.shape[1]):
        return float("nan")
    a = np.asarray(entry["_real_update"], np.float32)
    return float(np.dot(a, g[0, p].astype(np.float32)))


def strongest_other(scores: Dict[str, float], relation: str):
    choices = [r for r in REL if r != relation]
    return max(choices, key=lambda r: float(scores[r]))


def predicted_B(q, scores, route):
    route = canon_rel(route)
    if route not in REL:
        return float("nan"), ""
    foil = strongest_other(scores, route)
    return float(q[route] - q[foil]), foil


# =============================================================================
# LOO cell-majority diagnostic
# =============================================================================

def add_loo_cell_majority(df: pd.DataFrame):
    """
    For each row, predict oracle sign using OTHER samples from the same
    (canonical_slot, update_layer) cell.

    Weighted vote uses |B_oracle|; plain vote uses counts.
    """
    out = df.copy()
    out["pred_sign_loo_cell_majority"] = 0
    out["pred_sign_loo_cell_weighted"] = 0

    keys = ["canonical_slot", "update_layer"]

    for _, idx in out.groupby(keys).groups.items():
        idx = list(idx)
        if len(idx) <= 1:
            continue

        signs = out.loc[idx, "oracle_sign"].to_numpy(int)
        mass = out.loc[idx, "oracle_abs_B"].to_numpy(float)
        sids = out.loc[idx, "sid"].to_numpy(int)

        # Aggregate by SID first so multiple duplicate rows from one sample
        # cannot overweight that sample.
        unique_sids = np.unique(sids)

        sid_count_vote = {}
        sid_mass_vote = {}
        for sid in unique_sids:
            m = sids == sid
            sid_count_vote[int(sid)] = float(np.sum(signs[m]))
            sid_mass_vote[int(sid)] = float(np.sum(signs[m] * mass[m]))

        total_count = float(sum(sid_count_vote.values()))
        total_mass = float(sum(sid_mass_vote.values()))

        for j, row_idx in enumerate(idx):
            sid = int(sids[j])

            cv = total_count - sid_count_vote[sid]
            mv = total_mass - sid_mass_vote[sid]

            out.at[row_idx, "pred_sign_loo_cell_majority"] = sign3(cv, 0.0)
            out.at[row_idx, "pred_sign_loo_cell_weighted"] = sign3(mv, 0.0)

    return out


# =============================================================================
# Summaries
# =============================================================================

def summarize_route(sample_df):
    rows = []

    route_cols = [
        ("sequence_winner", "route_sequence_winner"),
        ("baseline_generation", "route_baseline_generation"),
        ("direction_head", "route_direction_head"),
    ]

    for method, col in route_cols:
        if col not in sample_df.columns:
            continue

        for cohort_name, g in [
            ("all", sample_df),
            ("baseline_wrong", sample_df[~sample_df["baseline_correct"]]),
            ("baseline_correct", sample_df[sample_df["baseline_correct"]]),
        ]:
            valid = g[col].isin(REL)
            gg = g[valid]
            rows.append(
                {
                    "method": method,
                    "cohort": cohort_name,
                    "N": int(len(gg)),
                    "relation_accuracy": (
                        float((gg[col] == gg["gt"]).mean())
                        if len(gg) else np.nan
                    ),
                    "match_baseline_prediction": (
                        float((gg[col] == gg["baseline_prediction"]).mean())
                        if len(gg) else np.nan
                    ),
                    "mean_selector_confidence": (
                        safe_mean(gg["selector_confidence"])
                        if "selector_confidence" in gg.columns else np.nan
                    ),
                    "mean_selector_margin": (
                        safe_mean(gg["selector_margin"])
                        if "selector_margin" in gg.columns else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def one_sign_summary(df, pred_col, method, cohort_name):
    x = df.copy()
    valid = (
        (x["oracle_sign"] != 0)
        & x[pred_col].isin([-1, 1])
    )
    x = x[valid]

    if cohort_name == "baseline_wrong":
        x = x[~x["baseline_correct"]]
    elif cohort_name == "baseline_correct":
        x = x[x["baseline_correct"]]

    if not len(x):
        return {
            "method": method,
            "cohort": cohort_name,
            "N_updates": 0,
            "N_samples": 0,
            "sign_accuracy": np.nan,
            "weighted_sign_accuracy": np.nan,
            "oracle_positive_fraction": np.nan,
            "pred_positive_fraction": np.nan,
            "mean_abs_oracle_B": np.nan,
        }

    correct = (
        x[pred_col].to_numpy(int)
        == x["oracle_sign"].to_numpy(int)
    )

    return {
        "method": method,
        "cohort": cohort_name,
        "N_updates": int(len(x)),
        "N_samples": int(x["sid"].nunique()),
        "sign_accuracy": float(np.mean(correct)),
        "weighted_sign_accuracy": weighted_accuracy(
            correct,
            x["oracle_abs_B"].to_numpy(float),
        ),
        "oracle_positive_fraction": float(
            np.mean(x["oracle_sign"].to_numpy(int) > 0)
        ),
        "pred_positive_fraction": float(
            np.mean(x[pred_col].to_numpy(int) > 0)
        ),
        "mean_abs_oracle_B": float(
            x["oracle_abs_B"].mean()
        ),
    }


def summarize_signs(df):
    methods = [
        ("sequence_winner", "pred_sign_sequence_winner"),
        ("baseline_generation", "pred_sign_baseline_generation"),
        ("direction_head", "pred_sign_direction_head"),
        ("loo_cell_majority", "pred_sign_loo_cell_majority"),
        ("loo_cell_weighted", "pred_sign_loo_cell_weighted"),
    ]

    rows = []
    for method, col in methods:
        if col not in df.columns:
            continue
        for cohort in ("all", "baseline_wrong", "baseline_correct"):
            rows.append(one_sign_summary(df, col, method, cohort))
    return pd.DataFrame(rows)


def summarize_by_layer(df):
    rows = []
    methods = [
        ("sequence_winner", "pred_sign_sequence_winner"),
        ("baseline_generation", "pred_sign_baseline_generation"),
        ("direction_head", "pred_sign_direction_head"),
    ]

    for L, g in df.groupby("update_layer"):
        for method, col in methods:
            for cohort_name, h in [
                ("all", g),
                ("baseline_wrong", g[~g["baseline_correct"]]),
                ("baseline_correct", g[g["baseline_correct"]]),
            ]:
                row = one_sign_summary(h, col, method, "all")
                row["cohort"] = cohort_name
                row["update_layer"] = int(L)
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_slot_layer(df):
    rows = []

    for (slot, rank, L), g in df.groupby(
        ["canonical_slot", "global_rank", "update_layer"]
    ):
        valid = g[g["oracle_sign"] != 0]
        if not len(valid):
            continue

        base = {
            "canonical_slot": slot,
            "global_rank": int(rank),
            "update_layer": int(L),
            "N_updates": int(len(valid)),
            "N_samples": int(valid["sid"].nunique()),
            "oracle_positive_fraction": float(
                np.mean(valid["oracle_sign"].to_numpy(int) > 0)
            ),
            "oracle_negative_fraction": float(
                np.mean(valid["oracle_sign"].to_numpy(int) < 0)
            ),
            "mean_oracle_B": float(valid["oracle_B"].mean()),
            "mean_abs_oracle_B": float(valid["oracle_abs_B"].mean()),
        }

        for name, col in [
            ("sequence", "pred_sign_sequence_winner"),
            ("baseline", "pred_sign_baseline_generation"),
            ("direction", "pred_sign_direction_head"),
        ]:
            ok = valid[col].isin([-1, 1])
            v = valid[ok]
            if len(v):
                c = (
                    v[col].to_numpy(int)
                    == v["oracle_sign"].to_numpy(int)
                )
                base[f"{name}_sign_acc"] = float(np.mean(c))
                base[f"{name}_weighted_sign_acc"] = weighted_accuracy(
                    c,
                    v["oracle_abs_B"].to_numpy(float),
                )
            else:
                base[f"{name}_sign_acc"] = np.nan
                base[f"{name}_weighted_sign_acc"] = np.nan

        rows.append(base)

    return pd.DataFrame(rows)


def summarize_abstention(
    df,
    *,
    confidence_thresholds,
    margin_thresholds,
):
    rows = []

    total_mass_all = float(
        df.loc[df["oracle_sign"] != 0, "oracle_abs_B"].sum()
    )
    total_mass_wrong = float(
        df.loc[
            (df["oracle_sign"] != 0) & (~df["baseline_correct"]),
            "oracle_abs_B",
        ].sum()
    )

    def add_sweep(kind, thresholds, value_col):
        for t in thresholds:
            for cohort_name, base_mask, total_mass in [
                (
                    "all",
                    np.ones(len(df), dtype=bool),
                    total_mass_all,
                ),
                (
                    "baseline_wrong",
                    (~df["baseline_correct"]).to_numpy(bool),
                    total_mass_wrong,
                ),
                (
                    "baseline_correct",
                    df["baseline_correct"].to_numpy(bool),
                    float(
                        df.loc[
                            (df["oracle_sign"] != 0) & df["baseline_correct"],
                            "oracle_abs_B",
                        ].sum()
                    ),
                ),
            ]:
                x = df[
                    base_mask
                    & (df["oracle_sign"] != 0)
                    & df["pred_sign_direction_head"].isin([-1, 1])
                ].copy()

                if kind == "margin":
                    covered = x["selector_margin"] >= float(t)
                else:
                    covered = x["selector_confidence"] >= float(t)

                xc = x[covered].copy()

                if len(x):
                    sample_base = x[["sid", value_col]].drop_duplicates("sid")
                    sample_cov = sample_base[value_col] >= float(t)
                    sample_coverage = float(sample_cov.mean())
                else:
                    sample_coverage = np.nan

                if len(xc):
                    correct = (
                        xc["pred_sign_direction_head"].to_numpy(int)
                        == xc["oracle_sign"].to_numpy(int)
                    )
                    acc = float(np.mean(correct))
                    wacc = weighted_accuracy(
                        correct,
                        xc["oracle_abs_B"].to_numpy(float),
                    )
                    covered_mass = float(xc["oracle_abs_B"].sum())
                else:
                    acc = np.nan
                    wacc = np.nan
                    covered_mass = 0.0

                rows.append(
                    {
                        "threshold_type": kind,
                        "threshold": float(t),
                        "cohort": cohort_name,
                        "N_updates_total": int(len(x)),
                        "N_updates_covered": int(len(xc)),
                        "update_coverage": (
                            len(xc) / len(x) if len(x) else np.nan
                        ),
                        "N_samples_total": int(x["sid"].nunique()),
                        "N_samples_covered": int(xc["sid"].nunique()),
                        "sample_coverage": sample_coverage,
                        "covered_sign_accuracy": acc,
                        "covered_weighted_sign_accuracy": wacc,
                        "covered_oracle_abs_mass_fraction": (
                            covered_mass / total_mass
                            if total_mass > EPS else np.nan
                        ),
                    }
                )

    add_sweep("margin", margin_thresholds, "selector_margin")
    add_sweep("confidence", confidence_thresholds, "selector_confidence")

    return pd.DataFrame(rows)


def oracle_cell_stability(df):
    rows = []
    for (slot, rank, L), g in df.groupby(
        ["canonical_slot", "global_rank", "update_layer"]
    ):
        g = g[g["oracle_sign"] != 0]
        if not len(g):
            continue

        pos = float(np.mean(g["oracle_sign"].to_numpy(int) > 0))
        neg = float(np.mean(g["oracle_sign"].to_numpy(int) < 0))
        majority_acc = max(pos, neg)

        rows.append(
            {
                "canonical_slot": slot,
                "global_rank": int(rank),
                "update_layer": int(L),
                "N": int(len(g)),
                "positive_fraction": pos,
                "negative_fraction": neg,
                "majority_sign_accuracy_in_sample": majority_acc,
                "mean_oracle_B": float(g["oracle_B"].mean()),
                "mean_abs_oracle_B": float(g["oracle_abs_B"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["majority_sign_accuracy_in_sample", "mean_abs_oracle_B"],
        ascending=[False, False],
    )


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    update_layers = parse_ints(a.update_layers)
    margin_thresholds = parse_floats(a.margin_thresholds)
    confidence_thresholds = parse_floats(a.confidence_thresholds)

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    global_dir = Path(a.global_run_dir)
    prior_dir = Path(a.prior_real_update_dir)

    resolved, old_oracle, global_top = load_global_run(
        global_dir,
        update_layers,
    )

    selector, pred_col, conf_col, margin_col = load_selector(
        Path(a.selector_csv),
        a.selector_method,
    )

    # Only samples successfully covered by the global fixed run.
    sids = sorted(set(resolved["sid"].astype(int)))
    if int(a.eval_max_samples) > 0 and len(sids) > int(a.eval_max_samples):
        # Use prior cohort metadata for a relation/correctness stratified cap.
        prior_cohort, _, prior_metadata, _ = l26.load_prior_run(prior_dir)
        prior_cohort = prior_cohort[prior_cohort["sid"].isin(sids)].copy()
        prior_cohort = l26.stratified_cap_df(
            prior_cohort,
            int(a.eval_max_samples),
            int(a.seed) + 41,
        )
        sids = sorted(prior_cohort["sid"].astype(int).tolist())

    resolved = resolved[resolved["sid"].isin(sids)].copy()
    old_oracle = old_oracle[old_oracle["sid"].isin(sids)].copy()
    selector = selector[selector["sid"].isin(sids)].copy()

    # Prior baseline/competitor metadata + dataset.
    prior_cohort, _, prior_metadata, _ = l26.load_prior_run(prior_dir)
    prior_cohort = prior_cohort[prior_cohort["sid"].isin(sids)].copy()

    two, eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        prior_cohort,
    )
    eval_meta = [m for m in eval_meta if int(m["sid"]) in set(sids)]

    meta_by_sid = {int(m["sid"]): m for m in eval_meta}
    selector_by_sid = selector.set_index("sid").to_dict("index")

    # Candidate answer surfaces exactly as prior run.
    texts = prior_metadata.get(
        "candidate_texts",
        {
            "left": "left",
            "right": "right",
            "above": "above",
            "below": "below",
        },
    )
    texts = {canon_rel(k): str(v) for k, v in texts.items()}
    for r in REL:
        if r not in texts:
            raise RuntimeError(f"candidate_texts missing {r}: {texts}")

    reduction = str(
        prior_metadata.get("sequence_score_reduction", "mean")
    )
    if reduction not in ("mean", "sum"):
        reduction = "mean"

    model = processor = None
    per_update_rows = []
    sample_route_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = l26.load_model(
            a,
            two,
        )
        device = torch.device(a.device)
        candidate_ids = gate.encode_candidate_ids(processor, texts)

        print("=" * 190)
        print("NON-ORACLE REAL-UPDATE POLARITY DIAGNOSTIC")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"global_run={global_dir}")
        print(f"N={len(eval_meta)}")
        print(f"update_layers={update_layers}")
        print(f"selector_method={a.selector_method}")
        print(f"selector_csv={a.selector_csv}")
        print("NO GENERATION WILL BE RUN.")
        print()

        for m in tqdm(eval_meta, desc="POLARITY diagnostic"):
            sid = int(m["sid"])
            real = None

            try:
                if sid not in selector_by_sid:
                    raise RuntimeError(
                        f"SID {sid} missing from direction selector CSV"
                    )

                rs = resolved[resolved["sid"] == sid].copy()
                specs = build_specs_for_sid(rs, selector_layer=max(update_layers))
                if not specs:
                    raise RuntimeError("No resolved global slots")

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )

                capture_layers = sorted(
                    set(update_layers + [L - 1 for L in update_layers])
                )
                real_states = gate.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                entries = gate.build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=bool(m["baseline_correct"]),
                    specs=specs,
                    r2n={},
                    real_states=real_states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=False,
                )
                if not entries:
                    raise RuntimeError("No REAL updates")

                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                # -------------------------------------------------------------
                # Four candidate sequence scores + gradients.
                # No GT is needed to compute any of these.
                # -------------------------------------------------------------
                scores = {}
                grads = {}
                for r in REL:
                    s, g = gate.sequence_score_and_grads(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        answer_ids=candidate_ids[r],
                        reduction=reduction,
                        grad_layers=grad_layers,
                    )
                    scores[r] = float(s)
                    grads[r] = g

                seq_winner = max(REL, key=lambda r: scores[r])
                gt = canon_rel(m["gt"])
                gt_comp = strongest_other(scores, gt)

                baseline_route = canon_rel(m["baseline_prediction"])
                sel_row = selector_by_sid[sid]
                dir_route = canon_rel(sel_row[pred_col])
                sel_conf = float(sel_row.get(conf_col, np.nan))
                sel_margin = float(sel_row.get(margin_col, np.nan))

                sample_route_rows.append(
                    {
                        "sid": sid,
                        "gt": gt,
                        "baseline_correct": bool(m["baseline_correct"]),
                        "baseline_prediction": baseline_route,
                        "route_sequence_winner": seq_winner,
                        "route_baseline_generation": baseline_route,
                        "route_direction_head": dir_route,
                        "selector_confidence": sel_conf,
                        "selector_margin": sel_margin,
                        **{f"score_{r}": scores[r] for r in REL},
                    }
                )

                # map concrete position -> canonical metadata
                spec_meta = {
                    int(s["real_position"]): s
                    for s in specs
                }

                # Optional replay lookup from previous global run.
                old_sid = old_oracle[old_oracle["sid"] == sid]
                old_lookup = {}
                for rr in old_sid.itertuples():
                    old_lookup[
                        (int(rr.update_layer), int(rr.real_position))
                    ] = float(rr.real_update_decision_score)

                for e in entries:
                    q = {
                        r: response_for_relation(e, grads, r)
                        for r in REL
                    }
                    if not all(np.isfinite(q[r]) for r in REL):
                        continue

                    oracle_B = float(q[gt] - q[gt_comp])
                    oracle_sign = sign3(
                        oracle_B,
                        float(a.oracle_sign_eps),
                    )

                    B_seq, foil_seq = predicted_B(
                        q, scores, seq_winner
                    )
                    B_base, foil_base = predicted_B(
                        q, scores, baseline_route
                    )
                    B_dir, foil_dir = predicted_B(
                        q, scores, dir_route
                    )

                    L = int(e["update_layer"])
                    p = int(e["real_position"])
                    sm = spec_meta[p]

                    prior_B = old_lookup.get((L, p), np.nan)

                    per_update_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "baseline_correct": bool(m["baseline_correct"]),
                            "baseline_prediction": baseline_route,
                            "canonical_slot": str(
                                sm.get("canonical_slot", "")
                            ),
                            "global_rank": int(
                                sm.get("global_rank", sm.get("best_rank", -1))
                            ),
                            "update_layer": L,
                            "position": p,
                            "token": str(e["token"]),
                            "category": str(e["category"]),
                            "broad_category": str(e["broad_category"]),
                            "real_update_norm": float(e["real_update_norm"]),
                            "oracle_competitor": gt_comp,
                            "oracle_B": oracle_B,
                            "oracle_abs_B": abs(oracle_B),
                            "oracle_sign": oracle_sign,
                            "prior_saved_oracle_B": prior_B,
                            "oracle_B_replay_abs_error": (
                                abs(oracle_B - prior_B)
                                if np.isfinite(prior_B) else np.nan
                            ),
                            **{f"q_{r}": float(q[r]) for r in REL},
                            **{f"score_{r}": float(scores[r]) for r in REL},
                            "route_sequence_winner": seq_winner,
                            "foil_sequence_winner": foil_seq,
                            "Bhat_sequence_winner": B_seq,
                            "pred_sign_sequence_winner": sign3(
                                B_seq, float(a.oracle_sign_eps)
                            ),
                            "route_baseline_generation": baseline_route,
                            "foil_baseline_generation": foil_base,
                            "Bhat_baseline_generation": B_base,
                            "pred_sign_baseline_generation": sign3(
                                B_base, float(a.oracle_sign_eps)
                            ),
                            "route_direction_head": dir_route,
                            "foil_direction_head": foil_dir,
                            "Bhat_direction_head": B_dir,
                            "pred_sign_direction_head": sign3(
                                B_dir, float(a.oracle_sign_eps)
                            ),
                            "selector_confidence": sel_conf,
                            "selector_margin": sel_margin,
                            "direction_relation_correct": dir_route == gt,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-80:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        df = pd.DataFrame(per_update_rows)
        sample_df = pd.DataFrame(sample_route_rows)

        if not len(df):
            raise RuntimeError("No polarity rows produced")

        # LOO cell diagnostic.
        df = add_loo_cell_majority(df)

        # Save raw.
        df.to_csv(
            outdir / "per_update_sign_predictions.csv",
            index=False,
        )
        sample_df.to_csv(
            outdir / "sample_relation_routes.csv",
            index=False,
        )

        # Summaries.
        route_summary = summarize_route(sample_df)
        sign_summary = summarize_signs(df)
        layer_summary = summarize_by_layer(df)
        slot_layer_summary = summarize_by_slot_layer(df)
        abstain_summary = summarize_abstention(
            df,
            confidence_thresholds=confidence_thresholds,
            margin_thresholds=margin_thresholds,
        )
        cell_stability = oracle_cell_stability(df)

        route_summary.to_csv(outdir / "route_summary.csv", index=False)
        sign_summary.to_csv(outdir / "sign_summary.csv", index=False)
        layer_summary.to_csv(outdir / "sign_by_layer.csv", index=False)
        slot_layer_summary.to_csv(
            outdir / "sign_by_slot_layer.csv",
            index=False,
        )
        abstain_summary.to_csv(
            outdir / "direction_abstention_summary.csv",
            index=False,
        )
        cell_stability.to_csv(
            outdir / "oracle_cell_stability.csv",
            index=False,
        )

        # Replay validation.
        replay = df[
            np.isfinite(
                pd.to_numeric(
                    df["prior_saved_oracle_B"],
                    errors="coerce",
                )
            )
        ].copy()

        replay_mae = (
            float(replay["oracle_B_replay_abs_error"].mean())
            if len(replay) else np.nan
        )
        replay_max = (
            float(replay["oracle_B_replay_abs_error"].max())
            if len(replay) else np.nan
        )
        replay_sign_agree = (
            float(
                np.mean(
                    np.sign(replay["oracle_B"].to_numpy(float))
                    == np.sign(
                        replay["prior_saved_oracle_B"].to_numpy(float)
                    )
                )
            )
            if len(replay) else np.nan
        )

        # Best abstention rows on baseline-wrong by weighted accuracy,
        # but show coverage too.
        wrong_abs = abstain_summary[
            abstain_summary["cohort"] == "baseline_wrong"
        ].copy()
        wrong_abs = wrong_abs[
            wrong_abs["N_updates_covered"] > 0
        ]
        if len(wrong_abs):
            wrong_abs = wrong_abs.sort_values(
                [
                    "covered_weighted_sign_accuracy",
                    "covered_oracle_abs_mass_fraction",
                ],
                ascending=[False, False],
            )

        print("=" * 190)
        print("NON-ORACLE UPDATE POLARITY DIAGNOSTIC")
        print("=" * 190)
        print(
            f"N samples={df['sid'].nunique()} | "
            f"N updates={len(df)} | "
            f"layers={update_layers}"
        )
        print(
            "Oracle replay: "
            f"MAE={replay_mae:.6g} | "
            f"max={replay_max:.6g} | "
            f"sign_agreement={replay_sign_agree:.4f}"
        )

        print("\nRELATION ROUTING")
        print("-" * 190)
        print(
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nPOLARITY SIGN PREDICTION")
        print("-" * 190)
        print(
            sign_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nPOLARITY BY LAYER")
        print("-" * 190)
        print(
            layer_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nBEST DIRECTION-HEAD ABSTENTION SETTINGS ON BASELINE-WRONG")
        print("-" * 190)
        if len(wrong_abs):
            print(
                wrong_abs.head(12).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        else:
            print("EMPTY")

        # compact text report
        report = [
            "=" * 190,
            "NON-ORACLE UPDATE POLARITY DIAGNOSTIC",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N samples={df['sid'].nunique()}",
            f"N updates={len(df)}",
            f"update layers={update_layers}",
            f"selector method={a.selector_method}",
            f"selector csv={a.selector_csv}",
            "",
            "ORACLE REPLAY",
            f"MAE={replay_mae:.8f}",
            f"MAX={replay_max:.8f}",
            f"sign_agreement={replay_sign_agree:.6f}",
            "",
            "RELATION ROUTING",
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "POLARITY SIGN PREDICTION",
            sign_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "BEST ABSTENTION ON BASELINE-WRONG",
            (
                wrong_abs.head(20).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(wrong_abs)
                else "EMPTY"
            ),
            "",
            "Interpretation guide:",
            "  1) Focus first on baseline_wrong, not overall.",
            "  2) weighted_sign_accuracy is more important than raw sign_accuracy",
            "     because large-|B| mistakes are behaviorally more consequential.",
            "  3) If direction_head > sequence_winner/baseline_generation on wrong",
            "     samples, the independent spatial representation is useful for HOW.",
            "  4) If confidence/margin abstention raises wrong-sample weighted accuracy",
            "     substantially while retaining useful oracle mass coverage, use that",
            "     as the next generation policy.",
            "  5) LOO cell-majority is diagnostic only. If it is already very high,",
            "     much of HOW is fixed by slot/layer rather than sample-specific.",
            "",
            "No generation was run in this script.",
        ]

        report_text = "\n".join(report) + "\n"
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        metadata = {
            "script": "diagnose_nonoracle_update_polarity_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "global_run_dir": str(global_dir),
            "prior_real_update_dir": str(prior_dir),
            "selector_csv": str(a.selector_csv),
            "selector_method": str(a.selector_method),
            "pred_column": pred_col,
            "confidence_column": conf_col,
            "margin_column": margin_col,
            "update_layers": update_layers,
            "sequence_score_reduction": reduction,
            "candidate_texts": texts,
            "oracle_sign_definition": (
                "sign(dot(a_REAL, grad[S_GT-S_best_nonGT]))"
            ),
            "direction_sign_definition": (
                "route by frozen Direction-Head selector; "
                "sign(dot(a_REAL, grad[S_route-S_best_other]))"
            ),
            "generation_run": False,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
