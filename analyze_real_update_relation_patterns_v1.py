#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_real_update_relation_patterns_v1.py

Offline analysis for eval_real_causal_token_update_gating_v1.py outputs.

Main questions:
1) How much does each intervention improve LEFT / RIGHT / ON(ABOVE) / UNDER(BELOW)?
2) For each relation, at which layers are actual REAL causal-token updates
   predominantly decision-positive vs decision-negative?
3) Are layer signs stable enough to replace per-sample oracle signs with a
   fixed global or relation-conditioned template?
4) On a held-out split, how well do:
      - global layer template
      - GT-relation layer template      (diagnostic oracle routing)
      - baseline-predicted-relation template (non-oracle routing diagnostic)
   predict the per-sample oracle update sign?

This script does NOT run the VLM. It only analyzes CSVs already produced by:
    eval_real_causal_token_update_gating_v1.py

Usage
=====
python analyze_real_update_relation_patterns_v1.py \
  --input-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --output-dir output/qwen3b_real_update_relation_patterns_all440_v1 \
  --train-frac 0.7 \
  --seed 17 \
  --overwrite

Important
=========
Using all 440 samples to DISCOVER patterns is fine for mechanism exploration.
Do not report a template learned and evaluated on the same 440 as a non-oracle
held-out result. This script therefore also creates a deterministic stratified
train/test split and reports template sign prediction only on held-out samples.

Relation display:
    above -> on
    below -> under
to match the user's four-way answer surface terminology.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REL_ORDER_INTERNAL = ["left", "right", "above", "below"]
REL_DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--stable-threshold",
        type=float,
        default=0.65,
        help="A layer sign is called stable if max(pos_frac, neg_frac) >= threshold.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_rel(x):
    s = str(x).strip().lower()
    aliases = {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "below": "below",
        "under": "below",
    }
    return aliases.get(s, s)


def display_rel(x):
    return REL_DISPLAY.get(normalize_rel(x), str(x))


def safe_mean(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(a.mean()) if len(a) else float("nan")


def sign_label(v, eps=0.0):
    v = float(v)
    if v > eps:
        return "+"
    if v < -eps:
        return "-"
    return "0"


def ensure_dir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_inputs(inp: Path):
    update_path = inp / "per_real_update_decision_score.csv"
    gen_path = inp / "generation_per_sample.csv"

    if not update_path.exists():
        raise FileNotFoundError(update_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    upd = pd.read_csv(update_path)
    gen = pd.read_csv(gen_path)

    req_upd = {
        "sid",
        "gt",
        "update_layer",
        "real_position",
        "real_update_decision_score",
    }
    req_gen = {
        "sid",
        "gt",
        "condition",
        "scale",
        "prediction",
        "correct",
    }

    mu = req_upd - set(upd.columns)
    mg = req_gen - set(gen.columns)
    if mu:
        raise RuntimeError(f"{update_path} missing columns: {sorted(mu)}")
    if mg:
        raise RuntimeError(f"{gen_path} missing columns: {sorted(mg)}")

    upd["sid"] = pd.to_numeric(upd["sid"], errors="raise").astype(int)
    upd["update_layer"] = pd.to_numeric(
        upd["update_layer"], errors="raise"
    ).astype(int)
    upd["real_update_decision_score"] = pd.to_numeric(
        upd["real_update_decision_score"], errors="coerce"
    )
    upd["gt"] = upd["gt"].map(normalize_rel)

    gen["sid"] = pd.to_numeric(gen["sid"], errors="raise").astype(int)
    gen["scale"] = pd.to_numeric(gen["scale"], errors="coerce")
    gen["gt"] = gen["gt"].map(normalize_rel)
    gen["prediction"] = gen["prediction"].map(normalize_rel)

    # Robust boolean normalization.
    if gen["correct"].dtype != bool:
        gen["correct"] = (
            gen["correct"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    return upd, gen


def baseline_table(gen):
    b = gen[gen["condition"] == "baseline"].copy()
    b = b.sort_values(["sid"]).drop_duplicates("sid")
    if b.empty:
        raise RuntimeError("generation_per_sample.csv contains no baseline rows")

    return b[
        ["sid", "gt", "prediction", "correct"]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "correct": "baseline_correct",
        }
    )


def relation_generation_summary(gen):
    b = baseline_table(gen).set_index("sid")
    rows = []

    # Baseline relation rows.
    for gt in REL_ORDER_INTERNAL:
        bg = b[b["gt"] == gt]
        if len(bg):
            rows.append(
                {
                    "relation": display_rel(gt),
                    "relation_internal": gt,
                    "condition": "baseline",
                    "scale": 0.0,
                    "N": len(bg),
                    "baseline_accuracy": float(bg["baseline_correct"].mean()),
                    "patched_accuracy": float(bg["baseline_correct"].mean()),
                    "gain": 0.0,
                    "wrong_to_correct": 0,
                    "correct_to_wrong": 0,
                    "net": 0,
                    "changed": 0,
                    "repair_rate_on_wrong": 0.0,
                    "preserve_rate_on_correct": 1.0,
                }
            )

    patched = gen[gen["condition"] != "baseline"].copy()

    for (cond, scale, gt), g in patched.groupby(
        ["condition", "scale", "gt"], dropna=False
    ):
        x = g.sort_values("sid").drop_duplicates("sid").set_index("sid")
        bg = b[b["gt"] == gt]
        common = sorted(set(bg.index) & set(x.index))
        if not common:
            continue

        bc = bg.loc[common, "baseline_correct"].astype(bool).to_numpy()
        pc = x.loc[common, "correct"].astype(bool).to_numpy()
        bp = bg.loc[common, "baseline_prediction"].astype(str).to_numpy()
        pp = x.loc[common, "prediction"].astype(str).to_numpy()

        w2c = int(np.sum((~bc) & pc))
        c2w = int(np.sum(bc & (~pc)))

        rows.append(
            {
                "relation": display_rel(gt),
                "relation_internal": gt,
                "condition": str(cond),
                "scale": float(scale),
                "N": len(common),
                "baseline_accuracy": float(np.mean(bc)),
                "patched_accuracy": float(np.mean(pc)),
                "gain": float(np.mean(pc) - np.mean(bc)),
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net": w2c - c2w,
                "changed": int(np.sum(bp != pp)),
                "repair_rate_on_wrong": (
                    w2c / max(int((~bc).sum()), 1)
                ),
                "preserve_rate_on_correct": (
                    1 - c2w / max(int(bc.sum()), 1)
                ),
            }
        )

    out = pd.DataFrame(rows)
    rel_rank = {r: i for i, r in enumerate(["left", "right", "on", "under"])}
    out["_rr"] = out["relation"].map(rel_rank).fillna(99)
    out = out.sort_values(
        ["condition", "scale", "_rr"]
    ).drop(columns="_rr").reset_index(drop=True)
    return out


def layer_summary(df, stable_threshold):
    rows = []

    for (gt, L), g in df.groupby(["gt", "update_layer"]):
        s = g["real_update_decision_score"].dropna().astype(float).to_numpy()
        if len(s) == 0:
            continue

        pos = float(np.mean(s > 0))
        neg = float(np.mean(s < 0))
        majority = "+" if pos >= neg else "-"
        consistency = max(pos, neg)

        rows.append(
            {
                "relation": display_rel(gt),
                "relation_internal": gt,
                "update_layer": int(L),
                "N_samples": int(g["sid"].nunique()),
                "N_updates": int(len(g)),
                "mean_score": float(np.mean(s)),
                "median_score": float(np.median(s)),
                "positive_fraction": pos,
                "negative_fraction": neg,
                "positive_mass_mean": float(np.maximum(s, 0).mean()),
                "negative_mass_mean": float(np.maximum(-s, 0).mean()),
                "majority_sign": majority,
                "sign_consistency": consistency,
                "stable_sign": (
                    majority if consistency >= stable_threshold else "mixed"
                ),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        rel_rank = {
            r: i for i, r in enumerate(["left", "right", "on", "under"])
        }
        out["_rr"] = out["relation"].map(rel_rank).fillna(99)
        out = out.sort_values(
            ["_rr", "update_layer"]
        ).drop(columns="_rr").reset_index(drop=True)
    return out


def make_sign_matrix(summary, value_col="stable_sign"):
    if summary.empty:
        return pd.DataFrame()
    m = summary.pivot(
        index="relation",
        columns="update_layer",
        values=value_col,
    )
    order = [x for x in ["left", "right", "on", "under"] if x in m.index]
    m = m.reindex(order)
    return m.reset_index()


def stratified_sid_split(base_df, train_frac, seed):
    """
    Stratify by GT relation and baseline correctness because both can strongly
    affect trajectory statistics.
    """
    if not (0 < train_frac < 1):
        raise ValueError("--train-frac must be in (0,1)")

    rng = random.Random(seed)
    train = set()
    test = set()

    x = base_df.copy()
    x["baseline_correct"] = x["baseline_correct"].astype(bool)

    for _, g in x.groupby(["gt", "baseline_correct"], dropna=False):
        sids = sorted(g["sid"].astype(int).tolist())
        rng.shuffle(sids)

        if len(sids) <= 1:
            ntr = len(sids)
        else:
            ntr = int(round(len(sids) * train_frac))
            ntr = min(max(ntr, 1), len(sids) - 1)

        train.update(sids[:ntr])
        test.update(sids[ntr:])

    return train, test


def learn_templates(train_updates):
    """
    Template value = sign of mean oracle B on TRAIN only.
    """
    global_rows = []
    for L, g in train_updates.groupby("update_layer"):
        v = float(g["real_update_decision_score"].mean())
        global_rows.append(
            {
                "template": "global_layer",
                "relation_internal": "*",
                "relation": "*",
                "update_layer": int(L),
                "train_mean_score": v,
                "template_sign": sign_label(v),
                "N_updates_train": len(g),
                "N_samples_train": g["sid"].nunique(),
            }
        )

    relation_rows = []
    for (gt, L), g in train_updates.groupby(["gt", "update_layer"]):
        v = float(g["real_update_decision_score"].mean())
        relation_rows.append(
            {
                "template": "relation_layer",
                "relation_internal": gt,
                "relation": display_rel(gt),
                "update_layer": int(L),
                "train_mean_score": v,
                "template_sign": sign_label(v),
                "N_updates_train": len(g),
                "N_samples_train": g["sid"].nunique(),
            }
        )

    return pd.DataFrame(global_rows + relation_rows)


def template_maps(template_df):
    glob = {}
    rel = {}

    for r in template_df.itertuples():
        if r.template == "global_layer":
            glob[int(r.update_layer)] = str(r.template_sign)
        elif r.template == "relation_layer":
            rel[(str(r.relation_internal), int(r.update_layer))] = str(
                r.template_sign
            )

    return glob, rel


def evaluate_template_signs(test_updates, base_df, templates):
    """
    This evaluates whether a fixed sign template can predict the oracle B sign.
    It does NOT run generation.
    """
    glob, relmap = template_maps(templates)

    bmeta = base_df.set_index("sid").to_dict("index")
    rows = []

    for e in test_updates.itertuples():
        sid = int(e.sid)
        gt = normalize_rel(e.gt)
        L = int(e.update_layer)
        score = float(e.real_update_decision_score)

        if not math.isfinite(score) or score == 0:
            continue

        oracle_sign = "+" if score > 0 else "-"
        pred_rel = normalize_rel(
            bmeta.get(sid, {}).get("baseline_prediction", "")
        )

        candidates = {
            "global_layer": glob.get(L, "0"),
            "gt_relation_layer": relmap.get((gt, L), "0"),
            "baseline_pred_relation_layer": relmap.get((pred_rel, L), "0"),
        }

        for method, psign in candidates.items():
            if psign not in ("+", "-"):
                continue

            rows.append(
                {
                    "sid": sid,
                    "gt": gt,
                    "relation": display_rel(gt),
                    "baseline_prediction": pred_rel,
                    "baseline_prediction_display": display_rel(pred_rel),
                    "baseline_correct": bool(
                        bmeta.get(sid, {}).get("baseline_correct", False)
                    ),
                    "update_layer": L,
                    "real_position": int(e.real_position),
                    "oracle_sign": oracle_sign,
                    "template_method": method,
                    "template_sign": psign,
                    "sign_correct": psign == oracle_sign,
                    "oracle_score": score,
                }
            )

    pred = pd.DataFrame(rows)
    if pred.empty:
        return pred, pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    for (method, relation), g in pred.groupby(
        ["template_method", "relation"], dropna=False
    ):
        summary_rows.append(
            {
                "template_method": method,
                "relation": relation,
                "N_updates": len(g),
                "N_samples": g["sid"].nunique(),
                "sign_accuracy": float(g["sign_correct"].mean()),
                "baseline_wrong_sign_accuracy": safe_mean(
                    g.loc[~g["baseline_correct"], "sign_correct"]
                ),
                "baseline_correct_sign_accuracy": safe_mean(
                    g.loc[g["baseline_correct"], "sign_correct"]
                ),
            }
        )

    # Overall rows
    for method, g in pred.groupby("template_method"):
        summary_rows.append(
            {
                "template_method": method,
                "relation": "ALL",
                "N_updates": len(g),
                "N_samples": g["sid"].nunique(),
                "sign_accuracy": float(g["sign_correct"].mean()),
                "baseline_wrong_sign_accuracy": safe_mean(
                    g.loc[~g["baseline_correct"], "sign_correct"]
                ),
                "baseline_correct_sign_accuracy": safe_mean(
                    g.loc[g["baseline_correct"], "sign_correct"]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Per-sample sign agreement, then macro-average.
    sample = (
        pred.groupby(
            ["template_method", "sid", "relation", "baseline_correct"],
            as_index=False,
        )["sign_correct"]
        .mean()
        .rename(columns={"sign_correct": "sample_sign_agreement"})
    )

    return pred, summary, sample


def main():
    a = parse_args()
    inp = Path(a.input_dir)
    out = Path(a.output_dir)
    ensure_dir(out, a.overwrite)

    upd, gen = load_inputs(inp)
    base_df = baseline_table(gen)

    # Join baseline correctness/prediction into updates from the actual run.
    join_meta = base_df[
        ["sid", "baseline_prediction", "baseline_correct"]
    ]
    upd = upd.drop(
        columns=["baseline_correct"],
        errors="ignore",
    ).merge(join_meta, on="sid", how="left")

    # ---------------------------------------------------------------------
    # 1. Relation-wise generation gain
    # ---------------------------------------------------------------------
    gen_rel = relation_generation_summary(gen)
    gen_rel.to_csv(out / "relation_generation_summary.csv", index=False)

    # ---------------------------------------------------------------------
    # 2. Relation x layer oracle sign patterns
    # ---------------------------------------------------------------------
    all_layer = layer_summary(upd, a.stable_threshold)
    all_layer.to_csv(
        out / "relation_layer_sign_summary_all.csv",
        index=False,
    )

    wrong_upd = upd[upd["baseline_correct"] == False].copy()  # noqa: E712
    correct_upd = upd[upd["baseline_correct"] == True].copy()  # noqa: E712

    wrong_layer = layer_summary(wrong_upd, a.stable_threshold)
    correct_layer = layer_summary(correct_upd, a.stable_threshold)

    wrong_layer.to_csv(
        out / "relation_layer_sign_summary_wrong.csv",
        index=False,
    )
    correct_layer.to_csv(
        out / "relation_layer_sign_summary_correct.csv",
        index=False,
    )

    make_sign_matrix(all_layer).to_csv(
        out / "relation_layer_stable_sign_matrix_all.csv",
        index=False,
    )
    make_sign_matrix(wrong_layer).to_csv(
        out / "relation_layer_stable_sign_matrix_wrong.csv",
        index=False,
    )

    # Also save mean-score and positive-fraction matrices for quick inspection.
    if len(all_layer):
        all_layer.pivot(
            index="relation",
            columns="update_layer",
            values="mean_score",
        ).reindex(
            [r for r in ["left", "right", "on", "under"]
             if r in set(all_layer["relation"])]
        ).reset_index().to_csv(
            out / "relation_layer_mean_score_matrix.csv",
            index=False,
        )

        all_layer.pivot(
            index="relation",
            columns="update_layer",
            values="positive_fraction",
        ).reindex(
            [r for r in ["left", "right", "on", "under"]
             if r in set(all_layer["relation"])]
        ).reset_index().to_csv(
            out / "relation_layer_positive_fraction_matrix.csv",
            index=False,
        )

    # ---------------------------------------------------------------------
    # 3. Held-out fixed-template sign predictability
    # ---------------------------------------------------------------------
    train_sids, test_sids = stratified_sid_split(
        base_df,
        a.train_frac,
        a.seed,
    )

    split_rows = []
    for r in base_df.itertuples():
        split_rows.append(
            {
                "sid": int(r.sid),
                "gt": normalize_rel(r.gt),
                "relation": display_rel(r.gt),
                "baseline_prediction": normalize_rel(r.baseline_prediction),
                "baseline_prediction_display": display_rel(
                    r.baseline_prediction
                ),
                "baseline_correct": bool(r.baseline_correct),
                "split": (
                    "train"
                    if int(r.sid) in train_sids
                    else "test"
                ),
            }
        )
    pd.DataFrame(split_rows).to_csv(
        out / "train_test_split.csv",
        index=False,
    )

    train_upd = upd[upd["sid"].isin(train_sids)].copy()
    test_upd = upd[upd["sid"].isin(test_sids)].copy()

    templates = learn_templates(train_upd)
    templates.to_csv(
        out / "fixed_layer_templates_train_only.csv",
        index=False,
    )

    pred, template_summary, sample_agree = evaluate_template_signs(
        test_upd,
        base_df,
        templates,
    )

    pred.to_csv(
        out / "heldout_template_sign_predictions.csv",
        index=False,
    )
    template_summary.to_csv(
        out / "heldout_template_sign_summary.csv",
        index=False,
    )
    sample_agree.to_csv(
        out / "heldout_template_sample_agreement.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Text report
    # ---------------------------------------------------------------------
    best_scale_rows = pd.DataFrame()
    if len(gen_rel):
        tmp = gen_rel[
            (gen_rel["condition"] != "baseline")
        ].copy()
        if len(tmp):
            # one best patched accuracy per condition x relation
            idx = tmp.groupby(
                ["condition", "relation"]
            )["patched_accuracy"].idxmax()
            best_scale_rows = tmp.loc[idx].sort_values(
                ["condition", "relation"]
            )

    stable_rows = all_layer[
        all_layer["stable_sign"].isin(["+", "-"])
    ].copy() if len(all_layer) else pd.DataFrame()

    lines = [
        "=" * 160,
        "RELATION-WISE REAL CAUSAL-TOKEN UPDATE PATTERNS",
        "=" * 160,
        f"input_dir={inp}",
        f"N baseline samples={base_df['sid'].nunique()}",
        f"N update rows={len(upd)}",
        f"train/test={len(train_sids)}/{len(test_sids)}",
        f"stable sign threshold={a.stable_threshold:.2f}",
        "",
        "RELATION-WISE GENERATION RESULTS",
        "-" * 160,
        (
            gen_rel.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(gen_rel)
            else "EMPTY"
        ),
        "",
        "RELATION x LAYER SIGN SUMMARY (ALL)",
        "-" * 160,
        (
            all_layer.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(all_layer)
            else "EMPTY"
        ),
        "",
        "STABLE RELATION x LAYER SIGNS",
        "-" * 160,
        (
            stable_rows[
                [
                    "relation",
                    "update_layer",
                    "mean_score",
                    "positive_fraction",
                    "negative_fraction",
                    "stable_sign",
                    "sign_consistency",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(stable_rows)
            else "NO STABLE SIGNS AT CURRENT THRESHOLD"
        ),
        "",
        "HELD-OUT FIXED-TEMPLATE SIGN PREDICTION",
        "-" * 160,
        (
            template_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(template_summary)
            else "EMPTY"
        ),
        "",
        "How to read template methods:",
        "  global_layer:",
        "      one +/- sign per layer, no relation needed; strongest no-oracle candidate.",
        "  gt_relation_layer:",
        "      one +/- sign per relation x layer, selected with GT relation; diagnostic ceiling.",
        "  baseline_pred_relation_layer:",
        "      same relation-specific templates, routed by the model's own baseline prediction;",
        "      this is a no-GT routing diagnostic (but still does not remove causal-token oracle).",
        "",
        "Decision rule for next experiment:",
        "  If global_layer held-out sign accuracy is strong and stable, test a fixed layer schedule.",
        "  If gt_relation_layer >> global_layer and baseline_pred_relation_layer remains strong,",
        "      use predicted-relation routing.",
        "  If all template accuracies are near chance, signs are sample/token-specific and a fixed",
        "      non-oracle layer schedule is unlikely to work; then learn/search an internal sign cue.",
        "",
        "Important: this analysis only evaluates sign predictability. It does not establish",
        "generation accuracy for fixed templates. That requires a second VLM run using the",
        "TRAIN-derived template on TEST samples.",
    ]

    report = "\n".join(lines) + "\n"
    print(report)
    (out / "analysis_summary.txt").write_text(report, encoding="utf-8")

    metadata = {
        "input_dir": str(inp),
        "N_samples": int(base_df["sid"].nunique()),
        "N_update_rows": int(len(upd)),
        "train_frac": float(a.train_frac),
        "seed": int(a.seed),
        "N_train_sids": len(train_sids),
        "N_test_sids": len(test_sids),
        "stable_threshold": float(a.stable_threshold),
        "relation_display_map": REL_DISPLAY,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
