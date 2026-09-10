#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-way writer competition + K-token comparison.

Purpose
-------
For every evaluation sample, DO NOT use GT to choose a writer during tracing.
Instead, trace all four learned late writers:

    LEFT / RIGHT / ON / UNDER

For each candidate writer r:
    J_r = sum_T <h[T,last], normalize(s_r^T)>
    M_{r,L,p} = (h_real[L,p] - h_gray[L,p]) dot dJ_r/dh_real[L,p]

Then:
  1) select raw Top-K positive global-unique token states for EACH writer;
  2) compare the GT/oracle writer's Top-K against the three other writers;
  3) build a non-oracle 4-way classifier from causal support mass;
  4) compare the GT-writer Top-K for baseline-CORRECT vs baseline-WRONG
     samples within each true relation;
  5) also compute a contrastive mediation score:
       C_{r,L,p} = M_{r,L,p} - mean_{q != r} M_{q,L,p}

No generation/intervention is done here. This is a tracing/selection diagnostic.

This script is self-contained except for two repo-native modules already used by
the AdaptVis experiments:
    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py

Typical quick run (same N=80 style as recent diagnostics):
CUDA_VISIBLE_DEVICES=0 python -u eval_four_writer_k7_competition_v1.py \
  --run-dir output/qwen3b_coco_dynamic_L20_26_K36_all440 \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --target-layers 32,34,35 \
  --k 7 \
  --max-eval-samples 80 \
  --output-dir output/qwen3b_four_writer_k7_N80 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
INV_DISPLAY = {"left": "left", "right": "right", "on": "above", "under": "below"}
EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True,
                   help="Existing oracle dynamic run containing baseline.csv and learned_writers.npz")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=80,
        help="0 = all samples available in baseline.csv",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return sorted({
        int(x.strip().upper().replace("L", ""))
        for x in str(s).split(",") if x.strip()
    })


def canon_rel(x):
    s = str(x).strip().lower()
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def disp_rel(x):
    return DISPLAY.get(canon_rel(x), str(x))


def normalize_np(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v / n).astype(np.float32)


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def clean_token(x):
    s = str(x).replace("\\n", " ").replace("Ġ", " ").replace("▁", " ")
    return re.sub(r"\s+", " ", s).strip()


def write_csv(path, rows):
    path = Path(path)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def span_positions(span):
    return set(range(int(span[0]), int(span[1]) + 1))


def all_subsequence_positions(ids, pat):
    if not pat or len(pat) > len(ids):
        return []
    ans = []
    n = len(pat)
    for i in range(len(ids) - n + 1):
        if ids[i:i+n] == pat:
            ans.extend(range(i, i+n))
    return sorted(set(ans))


def find_text_positions(tokenizer, ids, text):
    ans = set()
    for variant in (text, " " + text):
        pat = tokenizer.encode(variant, add_special_tokens=False)
        ans.update(all_subsequence_positions(ids, pat))
    return sorted(ans)


def build_categories(model, processor, batch, ids, subject, reference):
    tokenizer = processor.tokenizer
    toks = [str(x) for x in tokenizer.convert_ids_to_tokens(ids)]

    try:
        sspan, rspan = base.locate_object_spans(tokenizer, ids, subject, reference)
        sub = span_positions(sspan)
        ref = span_positions(rspan)
    except Exception:
        sub = set(find_text_positions(tokenizer, ids, subject))
        ref = set(find_text_positions(tokenizer, ids, reference))

    try:
        visual = set(map(int, base.resolve_visual_indices(model, processor, batch, ids)))
    except Exception:
        visual = set()
        for p, tok in enumerate(toks):
            if "image_pad" in tok or "video_pad" in tok:
                visual.add(p)

    rel_pos = {}
    for word in ("left", "right", "above", "below", "on", "under"):
        rel_pos[word] = set(find_text_positions(tokenizer, ids, word))

    cats = []
    for p in range(len(ids)):
        if p == len(ids) - 1:
            cat = "last"
        elif p in sub:
            cat = "subject"
        elif p in ref:
            cat = "reference"
        else:
            hit = next((w for w, ps in rel_pos.items() if p in ps), None)
            if hit is not None:
                cat = f"relation_word:{hit}"
            elif p in visual:
                cat = "visual"
            else:
                cat = "other_text"
        cats.append(cat)
    return cats, toks


def broad_category(cat):
    cat = str(cat)
    if cat.startswith("relation_word:"):
        return "relation_words"
    if cat in ("visual", "subject", "reference"):
        return cat
    if cat == "last":
        return "last"
    return "other_text"


# -----------------------------------------------------------------------------
# Activation capture
# -----------------------------------------------------------------------------

class Capture:
    def __init__(self, decoder_layers, layers, cut_layer=None, cpu=False):
        self.states = {}
        self.handles = []
        self.cut_layer = cut_layer
        self.cpu = cpu
        for L in layers:
            if cut_layer is not None and L == cut_layer:
                h = decoder_layers[L].register_forward_hook(self._cut(L))
            else:
                h = decoder_layers[L].register_forward_hook(self._keep(L))
            self.handles.append(h)

    def _cut(self, L):
        def hook(_m, _inp, out):
            x = traj.first_tensor(out)
            y = x.detach().clone().requires_grad_(True)
            self.states[L] = y
            return traj.replace_first_tensor(out, y)
        return hook

    def _keep(self, L):
        def hook(_m, _inp, out):
            x = traj.first_tensor(out)
            self.states[L] = x.detach().float().cpu() if self.cpu else x
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_cpu(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers, cpu=True)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured states: {missing}")
        return {
            L: cap.states[L].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


def forward_graph(model, decoder_layers, batch, layers, cut):
    cap = Capture(decoder_layers, layers, cut_layer=cut, cpu=False)
    kw = dict(batch)
    kw["use_cache"] = False
    _ = model(**kw)
    missing = [L for L in layers if L not in cap.states]
    if missing:
        cap.close()
        raise RuntimeError(f"Missing graph states: {missing}")
    return cap


# -----------------------------------------------------------------------------
# Writers and selection
# -----------------------------------------------------------------------------

def load_writers(npz_path, targets):
    z = np.load(npz_path)
    writers = {T: {} for T in targets}
    missing = []
    for T in targets:
        for r in REL:
            key = f"L{T}_{DISPLAY[r]}"
            if key not in z:
                missing.append(key)
            else:
                writers[T][r] = np.asarray(z[key], dtype=np.float32)
    if missing:
        raise RuntimeError(
            "learned_writers.npz missing keys: " + ", ".join(missing)
        )
    return writers


def global_unique_topk(rows, k, score_key="mediation", positive_only=True):
    """
    K total positions across all source layers.
    Same token position may appear at several layers; retain only the layer with
    maximal requested score for that position.
    """
    best = {}
    for r in rows:
        score = float(r[score_key])
        if not np.isfinite(score):
            continue
        if positive_only and score <= 0:
            continue
        p = int(r["position"])
        if p not in best or score > float(best[p][score_key]):
            best[p] = r

    vals = list(best.values())
    vals.sort(key=lambda x: float(x[score_key]), reverse=True)
    return vals[:int(k)]


def set_overlap(a_rows, b_rows):
    a_exact = {(int(r["source_layer"]), int(r["position"])) for r in a_rows}
    b_exact = {(int(r["source_layer"]), int(r["position"])) for r in b_rows}
    a_pos = {int(r["position"]) for r in a_rows}
    b_pos = {int(r["position"]) for r in b_rows}

    exact_i = len(a_exact & b_exact)
    pos_i = len(a_pos & b_pos)
    return {
        "exact_intersection": exact_i,
        "exact_jaccard": safe_div(exact_i, len(a_exact | b_exact)),
        "position_intersection": pos_i,
        "position_jaccard": safe_div(pos_i, len(a_pos | b_pos)),
        "oracle_position_recall": safe_div(pos_i, len(a_pos)),
    }


def score_selected(rows, key="mediation"):
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    if not vals:
        return {
            "sum": 0.0, "mean": 0.0, "top1": 0.0,
            "top1_share": 0.0, "n": 0,
        }
    s = float(np.sum(vals))
    return {
        "sum": s,
        "mean": float(np.mean(vals)),
        "top1": float(max(vals)),
        "top1_share": safe_div(max(vals), s),
        "n": len(vals),
    }


def role_hist(rows):
    c = Counter(str(r["broad_category"]) for r in rows)
    return c


def hist_l1(a, b):
    cats = set(a) | set(b)
    sa, sb = sum(a.values()), sum(b.values())
    if sa == 0 or sb == 0:
        return float("nan")
    return float(sum(abs(a.get(c,0)/sa - b.get(c,0)/sb) for c in cats))


# -----------------------------------------------------------------------------
# Aggregation / reporting
# -----------------------------------------------------------------------------

def chi2_optional(table):
    try:
        from scipy.stats import chi2_contingency
        arr = np.asarray(table, dtype=np.float64)
        # Drop all-zero rows/columns.
        arr = arr[arr.sum(axis=1) > 0]
        arr = arr[:, arr.sum(axis=0) > 0]
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return float("nan"), float("nan"), 0
        chi2, p, dof, _ = chi2_contingency(arr)
        return float(chi2), float(p), int(dof)
    except Exception:
        return float("nan"), float("nan"), 0


def classification_summary(per_sample):
    df = pd.DataFrame(per_sample)
    rows = []
    for method_col, name in [
        ("pred_raw_sum", "raw_topk_sum"),
        ("pred_raw_mean", "raw_topk_mean"),
        ("pred_contrast_sum", "contrast_topk_sum"),
        ("pred_contrast_mean", "contrast_topk_mean"),
    ]:
        ok = df[method_col].astype(str) == df["gt"].astype(str)
        bmatch = df[method_col].astype(str) == df["baseline_prediction"].astype(str)

        wrong = ~df["baseline_correct"].astype(bool)
        correct = df["baseline_correct"].astype(bool)

        rows.append({
            "method": name,
            "N": len(df),
            "relation_acc": float(ok.mean()),
            "acc_on_baseline_correct": float(ok[correct].mean()) if correct.any() else np.nan,
            "acc_on_baseline_wrong": float(ok[wrong].mean()) if wrong.any() else np.nan,
            "match_baseline_prediction_all": float(bmatch.mean()),
            "match_baseline_prediction_when_baseline_wrong": (
                float(bmatch[wrong].mean()) if wrong.any() else np.nan
            ),
        })
    return pd.DataFrame(rows)


def confusion(df, pred_col):
    tab = pd.crosstab(df["gt"], df[pred_col])
    order = ["left", "right", "on", "under"]
    tab = tab.reindex(index=order, columns=order, fill_value=0)
    tab.insert(0, "gt", tab.index)
    return tab.reset_index(drop=True)


def correct_wrong_oracle_summary(selected_df, per_sample_df, k):
    """
    Compare the GT/oracle writer's raw TopK between baseline-correct/wrong
    samples within each GT relation.
    """
    x = selected_df[
        (selected_df["selection_type"] == "raw")
        & (selected_df["is_oracle_writer"] == True)
    ].copy()

    role_rows = []
    layer_rows = []
    token_rows = []
    stat_rows = []

    cats = ["visual", "subject", "reference", "relation_words", "other_text"]

    for gt in ["left", "right", "on", "under"]:
        g = x[x["gt"] == gt].copy()
        if not len(g):
            continue

        # token-level role distributions
        for bc, n in g.groupby(["baseline_correct", "broad_category"]).size().items():
            corr, cat = bc
            total = int((g["baseline_correct"] == corr).sum())
            role_rows.append({
                "gt": gt,
                "baseline_group": "correct" if bool(corr) else "wrong",
                "broad_category": cat,
                "count": int(n),
                "share": safe_div(int(n), total),
                "N_selected": total,
            })

        for (corr, L), n in g.groupby(["baseline_correct", "source_layer"]).size().items():
            total = int((g["baseline_correct"] == corr).sum())
            layer_rows.append({
                "gt": gt,
                "baseline_group": "correct" if bool(corr) else "wrong",
                "source_layer": int(L),
                "count": int(n),
                "share": safe_div(int(n), total),
                "N_selected": total,
            })

        for corr in [True, False]:
            gg = g[g["baseline_correct"] == corr]
            if not len(gg):
                continue
            vc = gg["token_clean"].value_counts().head(20)
            for tok, n in vc.items():
                token_rows.append({
                    "gt": gt,
                    "baseline_group": "correct" if corr else "wrong",
                    "token": tok,
                    "count": int(n),
                    "share": safe_div(int(n), len(gg)),
                })

        # chi-square correct/wrong x broad role
        ct = pd.crosstab(g["baseline_correct"], g["broad_category"])
        chi2, p, dof = chi2_optional(ct.values)

        ps = per_sample_df[per_sample_df["gt"] == gt]
        for corr in [True, False]:
            s = ps[ps["baseline_correct"] == corr]
            if not len(s):
                continue
            stat_rows.append({
                "gt": gt,
                "baseline_group": "correct" if corr else "wrong",
                "N_samples": len(s),
                "oracle_raw_sum_mean": safe_mean(s["oracle_raw_sum"]),
                "oracle_raw_top1_mean": safe_mean(s["oracle_raw_top1"]),
                "oracle_raw_top1_share_mean": safe_mean(s["oracle_raw_top1_share"]),
                "oracle_contrast_sum_mean": safe_mean(s["oracle_contrast_sum"]),
                "raw_gt_margin_mean": safe_mean(s["raw_gt_margin"]),
                "contrast_gt_margin_mean": safe_mean(s["contrast_gt_margin"]),
                "raw_selector_gt_rate": float((s["pred_raw_sum"] == s["gt"]).mean()),
                "contrast_selector_gt_rate": float((s["pred_contrast_sum"] == s["gt"]).mean()),
                "raw_selector_matches_baseline_rate": float(
                    (s["pred_raw_sum"] == s["baseline_prediction"]).mean()
                ),
                "contrast_selector_matches_baseline_rate": float(
                    (s["pred_contrast_sum"] == s["baseline_prediction"]).mean()
                ),
                "role_correct_vs_wrong_chi2": chi2,
                "role_correct_vs_wrong_p": p,
                "role_correct_vs_wrong_df": dof,
            })

    return (
        pd.DataFrame(role_rows),
        pd.DataFrame(layer_rows),
        pd.DataFrame(token_rows),
        pd.DataFrame(stat_rows),
    )


def render_summary(
    a,
    per_sample_df,
    class_df,
    overlap_df,
    selected_df,
    cw_stats,
):
    lines = []
    lines.append("=" * 150)
    lines.append("FOUR-WAY WRITER COMPETITION + ORACLE-vs-OTHER K COMPARISON")
    lines.append("=" * 150)
    lines.append(
        f"N={len(per_sample_df)} | K={a.k} | source={a.source_layers} | target={a.target_layers}"
    )
    lines.append(
        "All four writers are traced for every sample. GT is NOT used to choose which writer is evaluated."
    )
    lines.append(
        "GT is used only afterward to label which of the four candidate writers is the oracle/true relation."
    )
    lines.append("")

    lines.append("1) NON-ORACLE 4-WAY SELECTION FROM CAUSAL SUPPORT")
    lines.append("-" * 150)
    for r in class_df.itertuples():
        lines.append(
            f"{r.method:<24s} relation_acc={r.relation_acc:.4f} | "
            f"correct-group={r.acc_on_baseline_correct:.4f} wrong-group={r.acc_on_baseline_wrong:.4f} | "
            f"match baseline={r.match_baseline_prediction_all:.4f} "
            f"(when baseline wrong={r.match_baseline_prediction_when_baseline_wrong:.4f})"
        )
    lines.append("")

    lines.append("2) HOW DIFFERENT IS THE TRUE/ORACLE WRITER'S K FROM THE OTHER THREE?")
    lines.append("-" * 150)
    if len(overlap_df):
        oo = (
            overlap_df.groupby(["gt", "candidate_relation"])
            .agg(
                N=("sid", "size"),
                pos_intersection=("position_intersection", "mean"),
                pos_jaccard=("position_jaccard", "mean"),
                exact_intersection=("exact_intersection", "mean"),
                exact_jaccard=("exact_jaccard", "mean"),
                role_l1=("role_l1", "mean"),
            )
            .reset_index()
        )
        for gt in ["left", "right", "on", "under"]:
            g = oo[oo["gt"] == gt]
            if not len(g):
                continue
            lines.append(f"GT={gt.upper()}")
            for r in g.itertuples():
                lines.append(
                    f"  vs {r.candidate_relation:<5s}: "
                    f"same positions={r.pos_intersection:.2f}/{a.k} "
                    f"Jpos={r.pos_jaccard:.3f} | "
                    f"same exact(L,p)={r.exact_intersection:.2f}/{a.k} "
                    f"Jexact={r.exact_jaccard:.3f} | role-L1={r.role_l1:.3f}"
                )
    lines.append("")

    lines.append("3) TRUE-WRITER K: BASELINE-CORRECT vs BASELINE-WRONG WITHIN SAME RELATION")
    lines.append("-" * 150)
    if len(cw_stats):
        for gt in ["left", "right", "on", "under"]:
            g = cw_stats[cw_stats["gt"] == gt]
            if not len(g):
                continue
            lines.append(f"{gt.upper()}:")
            for r in g.itertuples():
                lines.append(
                    f"  {r.baseline_group:<7s} N={int(r.N_samples):2d} | "
                    f"oracle rawSum={r.oracle_raw_sum_mean:+.3f} "
                    f"contrastSum={r.oracle_contrast_sum_mean:+.3f} | "
                    f"GT-vs-bestWrong margin raw={r.raw_gt_margin_mean:+.3f} "
                    f"contrast={r.contrast_gt_margin_mean:+.3f} | "
                    f"4wayGT raw={r.raw_selector_gt_rate:.3f} "
                    f"contrast={r.contrast_selector_gt_rate:.3f} | "
                    f"4way matches baseline raw={r.raw_selector_matches_baseline_rate:.3f}"
                )
            # p is duplicated for group rows, show once.
            pvals = g["role_correct_vs_wrong_p"].dropna()
            if len(pvals):
                lines.append(
                    f"  correct-vs-wrong broad-role chi-square p={float(pvals.iloc[0]):.4g}"
                )
    lines.append("")

    lines.append("4) TOP TOKENS OF THE TRUE/ORACLE K BY RELATION AND CORRECTNESS")
    lines.append("-" * 150)
    ox = selected_df[
        (selected_df["selection_type"] == "raw")
        & (selected_df["is_oracle_writer"] == True)
    ]
    for gt in ["left", "right", "on", "under"]:
        lines.append(f"{gt.upper()}:")
        for corr, label in [(True, "correct"), (False, "wrong")]:
            g = ox[(ox["gt"] == gt) & (ox["baseline_correct"] == corr)]
            if not len(g):
                continue
            roles = g["broad_category"].value_counts(normalize=True)
            toks = g["token_clean"].replace("", np.nan).dropna().value_counts().head(10)
            layers = g["source_layer"].value_counts(normalize=True).sort_index()
            lines.append(
                f"  {label:<7s} roles: "
                + ", ".join(f"{k}={v:.1%}" for k,v in roles.items())
            )
            lines.append(
                f"           top tokens: "
                + ", ".join(f"{repr(k)}={int(v)}" for k,v in toks.items())
            )
            lines.append(
                f"           layers: "
                + ", ".join(f"L{int(k)}={v:.1%}" for k,v in layers.items())
            )
    lines.append("")

    lines.append("Files of interest:")
    lines.append("  per_sample_direction_scores.csv      : four support scores + selected direction for every sample")
    lines.append("  selected_topk_all_directions.csv     : the actual K tokens for all 4 writer hypotheses")
    lines.append("  oracle_vs_other_overlap_per_sample.csv: oracle K vs each wrong-writer K")
    lines.append("  oracle_correct_wrong_roles.csv       : true-writer K role distribution, correct vs wrong")
    lines.append("  oracle_correct_wrong_stats.csv       : support/margin differences, correct vs wrong")
    lines.append("")
    lines.append("Interpretation guardrail:")
    lines.append(
        "  learned writers come from labeled calibration data; therefore 4-way competition is non-oracle at test time, "
        "but not unsupervised."
    )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    a = parse_args()
    if a.k <= 0:
        raise ValueError("--k must be positive")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    targets = parse_ints(a.target_layers)
    graph_layers = sorted(set(source_layers + targets))
    cut = min(source_layers)

    run_dir = Path(a.run_dir)
    baseline_path = run_dir / "baseline.csv"
    writer_path = run_dir / "learned_writers.npz"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not writer_path.exists():
        raise FileNotFoundError(writer_path)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dir = outdir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_path)
    baseline["sid"] = pd.to_numeric(baseline["sid"], errors="raise").astype(int)
    baseline["gt_internal"] = baseline["gt"].map(canon_rel)
    baseline["gt"] = baseline["gt_internal"].map(DISPLAY)
    baseline["baseline_prediction_internal"] = baseline["baseline_prediction"].map(canon_rel)
    baseline["baseline_prediction"] = baseline["baseline_prediction_internal"].map(
        lambda x: DISPLAY.get(x, str(x))
    )
    if "baseline_correct" not in baseline.columns:
        baseline["baseline_correct"] = (
            baseline["gt_internal"] == baseline["baseline_prediction_internal"]
        )
    else:
        # robust bool conversion
        baseline["baseline_correct"] = baseline["baseline_correct"].map(
            lambda x: str(x).strip().lower() in {"true","1","yes"} if not isinstance(x, (bool, np.bool_)) else bool(x)
        )

    valid = baseline[baseline["gt_internal"].isin(REL)].copy()
    valid_sids = sorted(valid["sid"].unique().tolist())

    if a.max_eval_samples and a.max_eval_samples > 0:
        rng = np.random.default_rng(a.seed)
        valid_sids = sorted(
            rng.choice(
                valid_sids,
                size=min(a.max_eval_samples, len(valid_sids)),
                replace=False,
            ).tolist()
        )
        valid = valid[valid["sid"].isin(valid_sids)].copy()

    baseline_by_sid = {
        int(r.sid): {
            "gt_internal": r.gt_internal,
            "gt": r.gt,
            "baseline_prediction_internal": r.baseline_prediction_internal,
            "baseline_prediction": r.baseline_prediction,
            "baseline_correct": bool(r.baseline_correct),
        }
        for r in valid.itertuples()
    }

    writers = load_writers(writer_path, targets)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _ = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    # sample metadata from prompt file
    meta = {}
    for sid in valid_sids:
        if sid not in prompts:
            raise RuntimeError(f"sid={sid} missing from prompt JSONL")
        p = prompts[sid]
        meta[sid] = {
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        }

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    print(f"Loading {spec.repo_id}", flush=True)
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model, processor)
    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    for L in source_layers + targets:
        if not (0 <= L < n_layers):
            raise ValueError(f"L{L} invalid; model has L0..L{n_layers-1}")

    print("=" * 150)
    print("FOUR-WAY WRITER COMPETITION + K7 COMPARISON")
    print("=" * 150)
    print(f"N={len(valid_sids)} | model={a.model} | decoder={decoder_path}")
    print(f"source={source_layers} | targets={targets} | K={a.k}")
    print("Each sample traces ALL FOUR writers: left/right/on/under.")
    print("GT is used only after tracing to identify the oracle writer for analysis.")
    print("No generation; baseline correctness/prediction are read from existing run.")
    print()

    all_med_rows = []
    all_selected_rows = []
    per_sample_rows = []
    overlap_rows = []

    try:
        for sid in tqdm(valid_sids, desc="4-way trace"):
            cache = chunk_dir / f"sid_{sid}.pkl.gz"
            if cache.exists():
                obj = pd.read_pickle(cache, compression="gzip")
                all_med_rows.extend(obj["mediation"])
                all_selected_rows.extend(obj["selected"])
                per_sample_rows.extend(obj["per_sample"])
                overlap_rows.extend(obj["overlap"])
                continue

            b = baseline_by_sid[sid]
            gt_internal = b["gt_internal"]
            gt_disp = b["gt"]
            pmeta = meta[sid]

            real = gray = rb = gb = cap = None
            sid_med_rows = []
            sid_selected_rows = []
            sid_per_sample = []
            sid_overlap = []

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=pmeta["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=pmeta["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = build_categories(
                    model, processor, rb, ids,
                    pmeta["subject"], pmeta["reference"]
                )

                hgray = capture_cpu(model, decoder_layers, gb, source_layers)

                # One clean REAL graph; reuse it for four writer objectives.
                with torch.enable_grad():
                    cap = forward_graph(
                        model, decoder_layers, rb, graph_layers, cut
                    )

                    # Cache REAL states and deltas once.
                    real_states = {}
                    delta_states = {}
                    npos_by_layer = {}
                    for S in source_layers:
                        Hreal = cap.states[S][0].detach().float().cpu().numpy().astype(np.float32)
                        Hgray = hgray[S][0].astype(np.float32)
                        npos = min(
                            len(ids), len(cats), len(toks),
                            Hreal.shape[0], Hgray.shape[0]
                        )
                        # exclude source last token
                        npos = max(0, npos - 1)
                        real_states[S] = Hreal
                        delta_states[S] = (Hreal[:npos] - Hgray[:npos]).astype(np.float32)
                        npos_by_layer[S] = npos

                    med_by_rel = {}

                    for rel_i, r in enumerate(REL):
                        terms = []
                        for T in targets:
                            s_hat = torch.as_tensor(
                                normalize_np(writers[T][r]),
                                device=cap.states[T].device,
                                dtype=torch.float32,
                            )
                            terms.append(
                                torch.dot(cap.states[T][0, -1].float(), s_hat)
                            )
                        objective = torch.stack(terms).sum()

                        grads = torch.autograd.grad(
                            objective,
                            [cap.states[S] for S in source_layers],
                            retain_graph=(rel_i < len(REL) - 1),
                            create_graph=False,
                            allow_unused=False,
                        )

                        rows_r = []
                        for S, g in zip(source_layers, grads):
                            G = g[0].detach().float().cpu().numpy().astype(np.float32)
                            npos = min(npos_by_layer[S], G.shape[0])
                            D = delta_states[S]

                            for pos in range(npos):
                                medv = float(np.dot(D[pos], G[pos]))
                                row = {
                                    "sid": sid,
                                    "gt": gt_disp,
                                    "baseline_prediction": b["baseline_prediction"],
                                    "baseline_correct": b["baseline_correct"],
                                    "candidate_relation": DISPLAY[r],
                                    "is_oracle_writer": bool(r == gt_internal),
                                    "source_layer": int(S),
                                    "position": int(pos),
                                    "token_id": int(ids[pos]),
                                    "token": str(toks[pos]).replace("\n", "\\n"),
                                    "token_clean": clean_token(toks[pos]),
                                    "category": cats[pos],
                                    "broad_category": broad_category(cats[pos]),
                                    "mediation": medv,
                                    "delta_h_norm": float(np.linalg.norm(D[pos])),
                                    "grad_norm": float(np.linalg.norm(G[pos])),
                                }
                                rows_r.append(row)
                        med_by_rel[r] = rows_r

                    # Add contrastive score at every exact (layer, position).
                    lookup = {}
                    for r in REL:
                        lookup[r] = {
                            (int(z["source_layer"]), int(z["position"])): float(z["mediation"])
                            for z in med_by_rel[r]
                        }

                    for r in REL:
                        for z in med_by_rel[r]:
                            key = (int(z["source_layer"]), int(z["position"]))
                            others = [lookup[q][key] for q in REL if q != r]
                            z["contrast_mediation"] = float(
                                z["mediation"] - np.mean(others)
                            )
                            sid_med_rows.append(dict(z))

                cap.close()
                cap = None

                # Top-K sets for all four relations: RAW and CONTRASTIVE.
                raw_sel = {}
                ctr_sel = {}
                raw_scores = {}
                ctr_scores = {}

                for r in REL:
                    raw_sel[r] = global_unique_topk(
                        med_by_rel[r], a.k, "mediation", positive_only=True
                    )

                    # Need contrast values from sid_med_rows.
                    rows_ctr = [
                        z for z in sid_med_rows
                        if z["candidate_relation"] == DISPLAY[r]
                    ]
                    ctr_sel[r] = global_unique_topk(
                        rows_ctr, a.k, "contrast_mediation", positive_only=True
                    )

                    raw_scores[r] = score_selected(raw_sel[r], "mediation")
                    ctr_scores[r] = score_selected(ctr_sel[r], "contrast_mediation")

                    # Export actual selected tokens.
                    for selection_type, chosen, score_key in [
                        ("raw", raw_sel[r], "mediation"),
                        ("contrast", ctr_sel[r], "contrast_mediation"),
                    ]:
                        for rank, z in enumerate(chosen, 1):
                            sid_selected_rows.append({
                                **{k:v for k,v in z.items()
                                   if k not in {"grad_norm", "delta_h_norm"}},
                                "selection_type": selection_type,
                                "rank": rank,
                                "selection_score": float(z[score_key]),
                            })

                # Candidate relation predicted by score, without GT.
                def argmax_rel(score_dict, field):
                    return max(REL, key=lambda r: float(score_dict[r][field]))

                pred_raw_sum_i = argmax_rel(raw_scores, "sum")
                pred_raw_mean_i = argmax_rel(raw_scores, "mean")
                pred_ctr_sum_i = argmax_rel(ctr_scores, "sum")
                pred_ctr_mean_i = argmax_rel(ctr_scores, "mean")

                wrong_raw_sums = [raw_scores[r]["sum"] for r in REL if r != gt_internal]
                wrong_ctr_sums = [ctr_scores[r]["sum"] for r in REL if r != gt_internal]

                ps = {
                    "sid": sid,
                    "gt": gt_disp,
                    "baseline_prediction": b["baseline_prediction"],
                    "baseline_correct": b["baseline_correct"],
                    "pred_raw_sum": DISPLAY[pred_raw_sum_i],
                    "pred_raw_mean": DISPLAY[pred_raw_mean_i],
                    "pred_contrast_sum": DISPLAY[pred_ctr_sum_i],
                    "pred_contrast_mean": DISPLAY[pred_ctr_mean_i],
                    "oracle_raw_sum": raw_scores[gt_internal]["sum"],
                    "oracle_raw_mean": raw_scores[gt_internal]["mean"],
                    "oracle_raw_top1": raw_scores[gt_internal]["top1"],
                    "oracle_raw_top1_share": raw_scores[gt_internal]["top1_share"],
                    "oracle_contrast_sum": ctr_scores[gt_internal]["sum"],
                    "oracle_contrast_mean": ctr_scores[gt_internal]["mean"],
                    "raw_gt_margin": (
                        raw_scores[gt_internal]["sum"] - max(wrong_raw_sums)
                    ),
                    "contrast_gt_margin": (
                        ctr_scores[gt_internal]["sum"] - max(wrong_ctr_sums)
                    ),
                }
                for r in REL:
                    d = DISPLAY[r]
                    ps[f"raw_sum_{d}"] = raw_scores[r]["sum"]
                    ps[f"raw_mean_{d}"] = raw_scores[r]["mean"]
                    ps[f"contrast_sum_{d}"] = ctr_scores[r]["sum"]
                    ps[f"contrast_mean_{d}"] = ctr_scores[r]["mean"]
                sid_per_sample.append(ps)

                # Compare true/oracle raw TopK with each other raw TopK.
                oracle_rows = raw_sel[gt_internal]
                oracle_roles = role_hist(oracle_rows)
                for q in REL:
                    if q == gt_internal:
                        continue
                    other_rows = raw_sel[q]
                    ov = set_overlap(oracle_rows, other_rows)
                    sid_overlap.append({
                        "sid": sid,
                        "gt": gt_disp,
                        "baseline_prediction": b["baseline_prediction"],
                        "baseline_correct": b["baseline_correct"],
                        "candidate_relation": DISPLAY[q],
                        **ov,
                        "role_l1": hist_l1(oracle_roles, role_hist(other_rows)),
                        "oracle_raw_sum": raw_scores[gt_internal]["sum"],
                        "other_raw_sum": raw_scores[q]["sum"],
                        "oracle_minus_other_raw_sum": (
                            raw_scores[gt_internal]["sum"] - raw_scores[q]["sum"]
                        ),
                    })

                pd.to_pickle(
                    {
                        "mediation": sid_med_rows,
                        "selected": sid_selected_rows,
                        "per_sample": sid_per_sample,
                        "overlap": sid_overlap,
                    },
                    cache,
                    compression="gzip",
                )

                all_med_rows.extend(sid_med_rows)
                all_selected_rows.extend(sid_selected_rows)
                per_sample_rows.extend(sid_per_sample)
                overlap_rows.extend(sid_overlap)

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------
    per_sample_df = pd.DataFrame(per_sample_rows)
    selected_df = pd.DataFrame(all_selected_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    class_df = classification_summary(per_sample_rows)

    role_df, layer_df, token_df, cw_stats = correct_wrong_oracle_summary(
        selected_df, per_sample_df, a.k
    )

    # Aggregate oracle-vs-other overlap correct/wrong too.
    overlap_summary = (
        overlap_df.groupby(["gt", "baseline_correct", "candidate_relation"])
        .agg(
            N=("sid", "size"),
            position_intersection=("position_intersection", "mean"),
            position_jaccard=("position_jaccard", "mean"),
            exact_intersection=("exact_intersection", "mean"),
            exact_jaccard=("exact_jaccard", "mean"),
            role_l1=("role_l1", "mean"),
            oracle_minus_other_raw_sum=("oracle_minus_other_raw_sum", "mean"),
        )
        .reset_index()
        if len(overlap_df) else pd.DataFrame()
    )

    # Per relation candidate-score averages, split correct/wrong.
    score_group_rows = []
    for gt in ["left", "right", "on", "under"]:
        for corr in [True, False]:
            g = per_sample_df[
                (per_sample_df["gt"] == gt)
                & (per_sample_df["baseline_correct"] == corr)
            ]
            if not len(g):
                continue
            for cand in ["left", "right", "on", "under"]:
                score_group_rows.append({
                    "gt": gt,
                    "baseline_group": "correct" if corr else "wrong",
                    "candidate_relation": cand,
                    "N_samples": len(g),
                    "raw_sum_mean": safe_mean(g[f"raw_sum_{cand}"]),
                    "raw_sum_std": safe_std(g[f"raw_sum_{cand}"]),
                    "contrast_sum_mean": safe_mean(g[f"contrast_sum_{cand}"]),
                    "contrast_sum_std": safe_std(g[f"contrast_sum_{cand}"]),
                })
    score_group_df = pd.DataFrame(score_group_rows)

    # Confusions.
    conf_raw = confusion(per_sample_df, "pred_raw_sum")
    conf_ctr = confusion(per_sample_df, "pred_contrast_sum")

    # Save compact files. Full all-state mediation can be huge, so make it
    # optional-ish by saving compressed pickle rather than CSV.
    pd.to_pickle(
        pd.DataFrame(all_med_rows),
        outdir / "all_four_writer_mediation.pkl.gz",
        compression="gzip",
    )
    write_csv(outdir / "selected_topk_all_directions.csv", selected_df)
    write_csv(outdir / "per_sample_direction_scores.csv", per_sample_df)
    write_csv(outdir / "fourway_classification_summary.csv", class_df)
    write_csv(outdir / "confusion_raw_sum.csv", conf_raw)
    write_csv(outdir / "confusion_contrast_sum.csv", conf_ctr)
    write_csv(outdir / "oracle_vs_other_overlap_per_sample.csv", overlap_df)
    write_csv(outdir / "oracle_vs_other_overlap_summary.csv", overlap_summary)
    write_csv(outdir / "oracle_correct_wrong_roles.csv", role_df)
    write_csv(outdir / "oracle_correct_wrong_layers.csv", layer_df)
    write_csv(outdir / "oracle_correct_wrong_top_tokens.csv", token_df)
    write_csv(outdir / "oracle_correct_wrong_stats.csv", cw_stats)
    write_csv(outdir / "candidate_scores_correct_wrong.csv", score_group_df)

    text = render_summary(
        a, per_sample_df, class_df, overlap_df, selected_df, cw_stats
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    metadata = {
        "N": len(per_sample_df),
        "k": a.k,
        "source_layers": source_layers,
        "target_layers": targets,
        "writer_file": str(writer_path),
        "baseline_file": str(baseline_path),
        "seed": a.seed,
        "max_eval_samples": a.max_eval_samples,
        "writer_objective": (
            "sum_T dot(h_real[T,last], unit learned writer[T, candidate_relation])"
        ),
        "mediation": "delta_h(real-gray) dot gradient(candidate writer objective)",
        "raw_selection": "positive global_unique TopK per candidate writer",
        "contrastive_selection": (
            "positive global_unique TopK of M_r - mean(M_other_relations)"
        ),
        "important": (
            "All four candidate writers are traced for every test sample. "
            "GT is not used to choose a writer during tracing. "
            "Learned writers themselves were estimated from labeled calibration data."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
