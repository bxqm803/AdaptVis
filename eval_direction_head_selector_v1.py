#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_selector_v1.py

FAST post-processing test of a pure Direction-Head selector.

No VLM forward pass is required if you already have:
  1) output/.../relation_vectors.npz
  2) output/.../head_results.csv
  3) four-way N80 per_sample_direction_scores.csv

Idea
----
For each head h, fit four relation directions/centroids from calibration
samples EXCLUDING the current N80 evaluation SIDs:

    d_{h,r} = mean(z_h | r) - mean(z_h)

where z_h is the per-sample Direction-Head residual saved by
analyze_coco_head_object_residual_direction_probe_v1.py.

For an evaluation sample x:

    e_{h,r}(x) = cosine(z_h(x), d_{h,r})

Each head therefore gives a GT-free 4-way spatial readout.

We then test:
  - single known heads (e.g. L26H03)
  - equal ensemble of known heads
  - top-K heads selected ONLY by calibration accuracy
  - calibration-accuracy-weighted ensembles
  - layer-balanced weighted ensembles

Important:
  * Evaluation GT is used only for final reporting.
  * Head reliability is recomputed on calibration samples after removing N80.
    It does NOT use head_results.csv accuracy to rank heads, avoiding test leakage.
  * No causal-K information is used here. This is the clean "Direction Head only"
    selector baseline.

Typical command
---------------
python -u eval_direction_head_selector_v1.py \
  --fourway-dir output/qwen3b_four_writer_k7_N80 \
  --direction-dir output/qwen3b_head_object_residual_direction \
  --focus-heads L26H03,L23H01,L23H05 \
  --topks 1,3,5,10,20 \
  --output-dir output/qwen3b_direction_head_selector_N80
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REL = ("left", "right", "on", "under")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--focus-heads", default="L26H03,L23H01,L23H05")
    p.add_argument("--topks", default="1,3,5,10,20")
    p.add_argument(
        "--min-calib-per-class",
        type=int,
        default=3,
        help="Require at least this many calibration examples per relation.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for per-head relation probabilities.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    mp = {
        "left": "left", "left_of": "left", "l": "left",
        "right": "right", "right_of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under",
        "bottom": "under",
    }
    return mp.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_head_name(s):
    m = re.fullmatch(r"\s*L(\d+)H(\d+)\s*", str(s), flags=re.I)
    if not m:
        raise ValueError(f"Bad head name: {s!r}; expected L26H03")
    return int(m.group(1)), int(m.group(2))


def head_name(L, H):
    return f"L{int(L)}H{int(H):02d}"


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


def safe_mean(x):
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def load_data(args):
    four = Path(args.fourway_dir)
    ddir = Path(args.direction_dir)

    ps_path = four / "per_sample_direction_scores.csv"
    npz_path = ddir / "relation_vectors.npz"
    head_csv = ddir / "head_results.csv"

    for p in (ps_path, npz_path):
        if not p.exists():
            raise FileNotFoundError(p)

    ps = pd.read_csv(ps_path)
    ps["sid"] = pd.to_numeric(ps["sid"], errors="raise").astype(int)
    ps["gt"] = ps["gt"].map(canon_rel)
    ps["baseline_prediction"] = ps["baseline_prediction"].map(canon_rel)
    ps["baseline_correct"] = ps["baseline_correct"].map(boolify)

    z = np.load(npz_path, allow_pickle=True)
    need = {"sample_index", "relation", "residual"}
    miss = need - set(z.files)
    if miss:
        raise RuntimeError(f"{npz_path} missing arrays: {sorted(miss)}")

    sids = np.asarray(z["sample_index"]).astype(int)
    labels = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)
    residual = np.asarray(z["residual"], dtype=np.float32)

    if residual.ndim != 4:
        raise RuntimeError(
            f"Expected residual [N,layer,head,dim], got {residual.shape}"
        )
    if len(sids) != residual.shape[0] or len(labels) != residual.shape[0]:
        raise RuntimeError("sample_index/relation/residual length mismatch")

    if head_csv.exists():
        heads_saved = pd.read_csv(head_csv)
    else:
        heads_saved = pd.DataFrame()

    return ps, sids, labels, residual, heads_saved, {
        "per_sample_scores": str(ps_path),
        "relation_vectors": str(npz_path),
        "head_results": str(head_csv) if head_csv.exists() else None,
    }


def fit_directions(X, y, min_per_class):
    """
    X [N,L,H,D]
    Return dirs [L,H,4,D], centered class means.
    """
    L, H, D = X.shape[1:]
    dirs = np.zeros((L, H, len(REL), D), dtype=np.float32)

    global_mean = X.mean(axis=0)  # [L,H,D]
    counts = {}
    for ri, r in enumerate(REL):
        m = (y == r)
        counts[r] = int(m.sum())
        if counts[r] < min_per_class:
            raise RuntimeError(
                f"Calibration relation {r} has only {counts[r]} examples"
            )
        d = X[m].mean(axis=0) - global_mean
        dirs[:, :, ri, :] = normalize(d, axis=-1)
    return dirs, counts


def head_scores(X, dirs):
    """
    X [N,L,H,D], dirs [L,H,R,D]
    cosine score [N,L,H,R].
    """
    Xn = normalize(X, axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", Xn, dirs, optimize=True)


def predict_per_head(scores):
    return np.argmax(scores, axis=-1)  # [N,L,H]


def calibration_head_accuracy(calib_scores, calib_y):
    y_idx = np.asarray([REL.index(r) for r in calib_y], dtype=int)
    pred = predict_per_head(calib_scores)
    return (pred == y_idx[:, None, None]).mean(axis=0)  # [L,H]


def evaluate_prediction(name, pred_idx, ps_eval):
    pred_rel = np.asarray([REL[int(i)] for i in pred_idx], dtype=object)
    gt = ps_eval["gt"].to_numpy(object)
    bp = ps_eval["baseline_prediction"].to_numpy(object)
    bc = ps_eval["baseline_correct"].to_numpy(bool)
    wrong = ~bc

    return {
        "selector": name,
        "N": len(ps_eval),
        "acc_all": float(np.mean(pred_rel == gt)),
        "acc_baseline_correct": (
            float(np.mean(pred_rel[bc] == gt[bc])) if bc.any() else np.nan
        ),
        "acc_baseline_wrong": (
            float(np.mean(pred_rel[wrong] == gt[wrong])) if wrong.any() else np.nan
        ),
        "match_baseline_all": float(np.mean(pred_rel == bp)),
        "match_baseline_wrong": (
            float(np.mean(pred_rel[wrong] == bp[wrong])) if wrong.any() else np.nan
        ),
        "disagree_baseline_wrong": (
            float(np.mean(pred_rel[wrong] != bp[wrong])) if wrong.any() else np.nan
        ),
    }, pred_rel


def aggregate_heads(
    eval_scores,
    selected_heads,
    calib_acc,
    temperature,
    weighted=False,
    layer_balanced=False,
):
    """
    eval_scores [N,L,H,R].

    Convert every head's four scores to probabilities first to avoid
    raw-score scale differences, then aggregate.
    """
    probs = softmax(eval_scores, axis=-1, temperature=temperature)
    N = probs.shape[0]

    if not selected_heads:
        return np.full((N, len(REL)), 1.0 / len(REL), dtype=float)

    if not layer_balanced:
        total = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0
        for L, H in selected_heads:
            w = max(float(calib_acc[L, H]) - 0.25, 0.0) if weighted else 1.0
            if w <= 0:
                continue
            total += w * probs[:, L, H, :]
            denom += w
        if denom <= EPS:
            return np.mean(
                np.stack([probs[:, L, H, :] for L, H in selected_heads], axis=0),
                axis=0,
            )
        return total / denom

    # First aggregate heads inside each layer, then average layers equally.
    by_layer = {}
    for L, H in selected_heads:
        by_layer.setdefault(int(L), []).append(int(H))

    layer_outputs = []
    for L, hs in sorted(by_layer.items()):
        total = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0
        for H in hs:
            w = max(float(calib_acc[L, H]) - 0.25, 0.0) if weighted else 1.0
            if w <= 0:
                continue
            total += w * probs[:, L, H, :]
            denom += w
        if denom > EPS:
            layer_outputs.append(total / denom)
        else:
            layer_outputs.append(np.mean(probs[:, L, hs, :], axis=1))

    return np.mean(np.stack(layer_outputs, axis=0), axis=0)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    focus_heads = [
        parse_head_name(x)
        for x in args.focus_heads.split(",")
        if x.strip()
    ]
    topks = parse_ints(args.topks)

    ps, sids, labels, residual, heads_saved, paths = load_data(args)

    eval_sids = set(ps["sid"].astype(int).tolist())
    valid_rel = np.isin(labels, np.asarray(REL, dtype=object))
    calib_mask = valid_rel & (~np.isin(sids, np.asarray(sorted(eval_sids), dtype=int)))
    eval_mask = valid_rel & np.isin(sids, np.asarray(sorted(eval_sids), dtype=int))

    if calib_mask.sum() == 0:
        raise RuntimeError(
            "No calibration examples remain after excluding evaluation SIDs."
        )

    print("=" * 140)
    print("DIRECTION-HEAD ONLY SELECTOR")
    print("=" * 140)
    print(
        f"relation_vectors={residual.shape} | "
        f"calibration N={int(calib_mask.sum())} | "
        f"eval vectors N={int(eval_mask.sum())} | "
        f"requested eval N={len(ps)}"
    )

    X_cal = residual[calib_mask]
    y_cal = labels[calib_mask]

    dirs, counts = fit_directions(
        X_cal, y_cal, args.min_calib_per_class
    )

    cal_scores = head_scores(X_cal, dirs)
    cal_acc = calibration_head_accuracy(cal_scores, y_cal)

    # Build exact SID -> vector row map for evaluation.
    sid_to_rows = {}
    for idx in np.where(eval_mask)[0]:
        sid_to_rows.setdefault(int(sids[idx]), []).append(int(idx))

    # If duplicate SIDs exist, require exactly one matching row or pick the
    # row whose saved relation agrees with PS GT. That uses GT only to resolve
    # duplicated storage identity, NOT to score relations.
    eval_rows = []
    missing = []
    duplicates = []
    for r in ps.itertuples():
        sid = int(r.sid)
        rows = sid_to_rows.get(sid, [])
        if not rows:
            missing.append(sid)
            continue
        if len(rows) == 1:
            eval_rows.append(rows[0])
        else:
            duplicates.append((sid, len(rows)))
            gt = canon_rel(r.gt)
            match = [i for i in rows if labels[i] == gt]
            eval_rows.append(match[0] if match else rows[0])

    if missing:
        raise RuntimeError(
            f"{len(missing)} evaluation SIDs are absent from relation_vectors.npz. "
            f"First missing: {missing[:20]}"
        )

    X_eval = residual[np.asarray(eval_rows)]
    ps_eval = ps.reset_index(drop=True).copy()

    eval_scores = head_scores(X_eval, dirs)
    Lmax, Hmax = cal_acc.shape

    # Head reliability table from calibration only.
    reli_rows = []
    for L in range(Lmax):
        for H in range(Hmax):
            reli_rows.append({
                "layer": L,
                "head": H,
                "head_name": head_name(L, H),
                "calibration_accuracy": float(cal_acc[L, H]),
                "weight_over_chance": max(float(cal_acc[L, H]) - 0.25, 0.0),
            })
    reliability = pd.DataFrame(reli_rows).sort_values(
        "calibration_accuracy", ascending=False
    )
    reliability.to_csv(outdir / "calibration_head_reliability.csv", index=False)

    summary_rows = []
    detail = ps_eval[
        ["sid", "gt", "baseline_prediction", "baseline_correct"]
    ].copy()

    # ----------------------------------------------------------------------
    # 1. Every focus head individually.
    # ----------------------------------------------------------------------
    valid_focus = []
    for L, H in focus_heads:
        if 0 <= L < Lmax and 0 <= H < Hmax:
            valid_focus.append((L, H))
            pred = np.argmax(eval_scores[:, L, H, :], axis=-1)
            s, p = evaluate_prediction(
                f"single_{head_name(L,H)}", pred, ps_eval
            )
            s["n_heads"] = 1
            s["head_set"] = head_name(L, H)
            s["mean_calib_head_acc"] = float(cal_acc[L, H])
            summary_rows.append(s)
            detail[f"pred_single_{head_name(L,H)}"] = p

    # Focus ensemble variants.
    if valid_focus:
        for weighted in (False, True):
            for layer_balanced in (False, True):
                probs = aggregate_heads(
                    eval_scores,
                    valid_focus,
                    cal_acc,
                    args.temperature,
                    weighted=weighted,
                    layer_balanced=layer_balanced,
                )
                pred = np.argmax(probs, axis=-1)
                name = (
                    "focus_"
                    + ("weighted" if weighted else "equal")
                    + ("_layerbalanced" if layer_balanced else "")
                )
                s, p = evaluate_prediction(name, pred, ps_eval)
                s["n_heads"] = len(valid_focus)
                s["head_set"] = ",".join(head_name(*x) for x in valid_focus)
                s["mean_calib_head_acc"] = safe_mean(
                    cal_acc[L, H] for L, H in valid_focus
                )
                summary_rows.append(s)
                detail[f"pred_{name}"] = p

    # ----------------------------------------------------------------------
    # 2. Top-K heads selected only from calibration reliability.
    # ----------------------------------------------------------------------
    ranked_heads = [
        (int(r.layer), int(r.head))
        for r in reliability.itertuples()
    ]

    for K in topks:
        selected = ranked_heads[: min(K, len(ranked_heads))]
        for weighted in (False, True):
            for layer_balanced in (False, True):
                probs = aggregate_heads(
                    eval_scores,
                    selected,
                    cal_acc,
                    args.temperature,
                    weighted=weighted,
                    layer_balanced=layer_balanced,
                )
                pred = np.argmax(probs, axis=-1)
                name = (
                    f"top{K}_"
                    + ("weighted" if weighted else "equal")
                    + ("_layerbalanced" if layer_balanced else "")
                )
                s, p = evaluate_prediction(name, pred, ps_eval)
                s["n_heads"] = len(selected)
                s["head_set"] = ",".join(head_name(*x) for x in selected)
                s["mean_calib_head_acc"] = safe_mean(
                    cal_acc[L, H] for L, H in selected
                )
                summary_rows.append(s)
                detail[f"pred_{name}"] = p

    # ----------------------------------------------------------------------
    # 3. All above-chance heads, weighted by calibration accuracy.
    # ----------------------------------------------------------------------
    above = [
        (int(r.layer), int(r.head))
        for r in reliability.itertuples()
        if float(r.calibration_accuracy) > 0.25
    ]
    for layer_balanced in (False, True):
        probs = aggregate_heads(
            eval_scores,
            above,
            cal_acc,
            args.temperature,
            weighted=True,
            layer_balanced=layer_balanced,
        )
        pred = np.argmax(probs, axis=-1)
        name = "all_abovechance_weighted" + (
            "_layerbalanced" if layer_balanced else ""
        )
        s, p = evaluate_prediction(name, pred, ps_eval)
        s["n_heads"] = len(above)
        s["head_set"] = "all_calibration_acc>0.25"
        s["mean_calib_head_acc"] = safe_mean(
            cal_acc[L, H] for L, H in above
        )
        summary_rows.append(s)
        detail[f"pred_{name}"] = p

    summary = pd.DataFrame(summary_rows).sort_values(
        ["acc_baseline_wrong", "acc_all"],
        ascending=False,
    )

    # Baseline row for reference.
    baseline_ref = pd.DataFrame([{
        "selector": "BASELINE_GENERATION",
        "N": len(ps_eval),
        "acc_all": float(ps_eval["baseline_correct"].mean()),
        "acc_baseline_correct": 1.0,
        "acc_baseline_wrong": 0.0,
        "match_baseline_all": 1.0,
        "match_baseline_wrong": 1.0,
        "disagree_baseline_wrong": 0.0,
        "n_heads": 0,
        "head_set": "",
        "mean_calib_head_acc": np.nan,
    }])

    out_summary = pd.concat([baseline_ref, summary], ignore_index=True)
    out_summary.to_csv(outdir / "direction_head_selector_summary.csv", index=False)
    detail.to_csv(outdir / "direction_head_selector_detail.csv", index=False)

    # Confusion table for best non-baseline selector.
    best = summary.iloc[0]
    best_col = f"pred_{best['selector']}"
    if best_col in detail.columns:
        confusion = pd.crosstab(
            detail["gt"],
            detail[best_col],
            rownames=["GT"],
            colnames=["Prediction"],
            dropna=False,
        )
        confusion.to_csv(outdir / "best_selector_confusion.csv")

    meta = {
        "calibration_n": int(calib_mask.sum()),
        "calibration_counts": counts,
        "evaluation_n": len(ps_eval),
        "focus_heads": [head_name(*x) for x in valid_focus],
        "topks": topks,
        "temperature": args.temperature,
        "head_selection": (
            "recomputed calibration accuracy after excluding all evaluation SIDs"
        ),
        "per_head_score": (
            "cosine(eval head residual, centered calibration relation direction)"
        ),
        "paths": paths,
        "duplicate_eval_sids_in_npz": duplicates,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print()
    print("CALIBRATION TOP HEADS")
    print("-" * 140)
    for r in reliability.head(20).itertuples():
        print(
            f"{r.head_name:<8s} calib_acc={r.calibration_accuracy:.4f} "
            f"weight={r.weight_over_chance:.4f}"
        )

    print()
    print("SELECTOR RESULTS")
    print("-" * 140)
    print(
        f"{'selector':<42s} {'all':>7s} {'baseC':>7s} "
        f"{'baseW':>7s} {'matchW':>7s} {'nH':>4s}"
    )
    for r in out_summary.itertuples():
        print(
            f"{r.selector:<42s} "
            f"{r.acc_all:7.3f} "
            f"{r.acc_baseline_correct:7.3f} "
            f"{r.acc_baseline_wrong:7.3f} "
            f"{r.match_baseline_wrong:7.3f} "
            f"{int(r.n_heads):4d}"
        )

    if len(summary):
        b = summary.iloc[0]
        print()
        print("BEST BY BASELINE-WRONG ACCURACY")
        print("-" * 140)
        print(
            f"{b['selector']} | all={b['acc_all']:.3f} | "
            f"baseline-correct={b['acc_baseline_correct']:.3f} | "
            f"baseline-wrong={b['acc_baseline_wrong']:.3f} | "
            f"matchBaselineWrong={b['match_baseline_wrong']:.3f}"
        )
        print("heads:", b["head_set"])

    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
