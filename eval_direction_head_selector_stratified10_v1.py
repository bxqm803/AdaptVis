#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_selector_stratified10_v1.py

Pure post-processing experiment:
  - use 10% of the 440 samples as a proportional stratified calibration set
  - estimate per-head relation directions only from that 10%
  - select Top-K Direction Heads only from that 10%
  - freeze everything
  - evaluate on the remaining 90%

Default repeats 5 independent stratified 10/90 splits.

Important:
  * No VLM parameter update / no SGD.
  * The 90% test split is never used to estimate relation directions or choose heads.
  * Head reliability is ranked with leave-one-out (LOO) accuracy INSIDE the 10%
    calibration set, reducing resubstitution optimism.
  * After heads are selected, relation directions are refit once on all calibration
    samples and used on the held-out 90%.

Expected input:
  <direction-dir>/relation_vectors.npz
from analyze_coco_head_object_residual_direction_probe_v1.py

The NPZ must contain:
  sample_index : [N]
  relation     : [N]
  residual     : [N, layers, heads, head_dim]

Typical:
python -u eval_direction_head_selector_stratified10_v1.py \
  --direction-dir output/qwen3b_head_object_residual_direction \
  --calib-frac 0.10 \
  --seeds 0,1,2,3,4 \
  --topks 1,3,5,10 \
  --output-dir output/qwen3b_direction_selector_strat10
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REL = ("left", "right", "on", "under")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--calib-frac", type=float, default=0.10)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--topks", default="1,3,5,10")
    p.add_argument(
        "--candidate-layers",
        default="",
        help="Optional comma-separated head layers. Empty = scan all layers.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature before head ensembling.",
    )
    p.add_argument(
        "--use-loo-head-ranking",
        action="store_true",
        default=True,
        help="Rank heads by leave-one-out accuracy on calibration set.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return {
        "left": "left", "left_of": "left", "left of": "left", "l": "left",
        "right": "right", "right_of": "right", "right of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under", "bottom": "under",
    }.get(s, s)


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


def head_name(L, H):
    return f"L{int(L)}H{int(H):02d}"


def proportional_quota(labels, frac):
    """
    Exact total n_cal = round(N*frac).
    Per-class quotas use the largest-remainder method, so the split follows
    the original class proportions as closely as possible.
    """
    labels = np.asarray(labels, dtype=object)
    N = len(labels)
    target = int(round(N * float(frac)))
    counts = {r: int(np.sum(labels == r)) for r in REL}

    raw = {r: counts[r] * float(frac) for r in REL}
    q = {r: int(math.floor(raw[r])) for r in REL}

    # Ensure every present relation has at least one calibration sample.
    for r in REL:
        if counts[r] > 0 and q[r] == 0:
            q[r] = 1

    current = sum(q.values())

    if current < target:
        order = sorted(
            REL,
            key=lambda r: (raw[r] - math.floor(raw[r]), counts[r]),
            reverse=True,
        )
        i = 0
        while current < target:
            r = order[i % len(order)]
            if q[r] < counts[r]:
                q[r] += 1
                current += 1
            i += 1

    elif current > target:
        order = sorted(
            REL,
            key=lambda r: (raw[r] - math.floor(raw[r]), counts[r]),
        )
        i = 0
        while current > target:
            r = order[i % len(order)]
            if q[r] > 1:
                q[r] -= 1
                current -= 1
            i += 1

    return q


def make_stratified_split(labels, frac, seed):
    labels = np.asarray(labels, dtype=object)
    quota = proportional_quota(labels, frac)
    rng = np.random.default_rng(seed)

    cal = []
    for r in REL:
        idx = np.where(labels == r)[0]
        take = int(quota[r])
        chosen = rng.choice(idx, size=take, replace=False)
        cal.extend(chosen.tolist())

    cal = np.asarray(sorted(cal), dtype=int)
    mask = np.ones(len(labels), dtype=bool)
    mask[cal] = False
    test = np.where(mask)[0]
    return cal, test, quota


def fit_directions(X, y):
    """
    X [N,L,H,D].
    d_r = mean(X|r) - mean(X), L2 normalized.
    """
    mean_all = X.mean(axis=0)
    L, H, D = X.shape[1:]
    dirs = np.zeros((L, H, len(REL), D), dtype=np.float32)

    for ri, r in enumerate(REL):
        m = (y == r)
        if not np.any(m):
            raise RuntimeError(f"No calibration sample for relation={r}")
        d = X[m].mean(axis=0) - mean_all
        dirs[:, :, ri, :] = normalize(d, axis=-1)

    return dirs


def score_with_dirs(X, dirs):
    """
    Cosine(X, d_r), returns [N,L,H,R].
    """
    Xn = normalize(X, axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", Xn, dirs, optimize=True)


def loo_head_accuracy(X, y):
    """
    Exact leave-one-out head selection accuracy on the calibration set.

    For each calibration sample i:
      fit d_r from the other N-1 examples,
      classify i,
    then compute 4-way accuracy for every [layer, head].

    Returns [L,H].
    """
    N, L, H, D = X.shape
    y_idx = np.asarray([REL.index(r) for r in y], dtype=int)

    total_sum = X.sum(axis=0)                    # [L,H,D]
    class_sum = {r: X[y == r].sum(axis=0) for r in REL}
    class_n = {r: int(np.sum(y == r)) for r in REL}

    correct = np.zeros((L, H), dtype=np.int32)

    for i in range(N):
        xi = X[i]
        yi = y[i]

        global_mean = (total_sum - xi) / float(N - 1)

        dirs_i = np.zeros((L, H, len(REL), D), dtype=np.float32)
        for ri, r in enumerate(REL):
            nr = class_n[r] - (1 if yi == r else 0)
            if nr <= 0:
                raise RuntimeError(
                    f"LOO impossible: relation={r} has <=1 calibration sample"
                )
            s = class_sum[r] - (xi if yi == r else 0.0)
            mean_r = s / float(nr)
            dirs_i[:, :, ri, :] = normalize(mean_r - global_mean, axis=-1)

        xin = normalize(xi, axis=-1)
        scores = np.einsum("lhd,lhrd->lhr", xin, dirs_i, optimize=True)
        pred = np.argmax(scores, axis=-1)
        correct += (pred == y_idx[i])

    return correct.astype(np.float32) / float(N)


def in_sample_head_accuracy(X, y, dirs):
    scores = score_with_dirs(X, dirs)
    pred = np.argmax(scores, axis=-1)
    y_idx = np.asarray([REL.index(r) for r in y], dtype=int)
    return (pred == y_idx[:, None, None]).mean(axis=0).astype(np.float32)


def candidate_head_list(acc, layers=None):
    L, H = acc.shape
    rows = []
    allowed = set(layers) if layers else None
    for l in range(L):
        if allowed is not None and l not in allowed:
            continue
        for h in range(H):
            rows.append((float(acc[l, h]), l, h))
    rows.sort(reverse=True)
    return rows


def ensemble_probs(scores, selected, acc, temperature, weighted=False):
    """
    scores [N,L,H,R].
    Convert each head's relation scores to a probability distribution first,
    then ensemble. This avoids arbitrary raw-cosine scale differences.
    """
    probs = softmax(scores, axis=-1, temperature=temperature)
    out = np.zeros((scores.shape[0], len(REL)), dtype=np.float64)
    denom = 0.0

    for L, H in selected:
        w = max(float(acc[L, H]) - 0.25, 0.0) if weighted else 1.0
        if w <= 0:
            continue
        out += w * probs[:, L, H, :]
        denom += w

    if denom <= EPS:
        arr = np.stack([probs[:, L, H, :] for L, H in selected], axis=0)
        return arr.mean(axis=0)
    return out / denom


def eval_pred(pred_idx, y):
    gt_idx = np.asarray([REL.index(r) for r in y], dtype=int)
    overall = float(np.mean(pred_idx == gt_idx))
    per_rel = {}
    for ri, r in enumerate(REL):
        m = (gt_idx == ri)
        per_rel[r] = float(np.mean(pred_idx[m] == gt_idx[m])) if m.any() else np.nan
    return overall, per_rel


def main():
    a = parse_args()
    seeds = parse_ints(a.seeds)
    topks = parse_ints(a.topks)
    candidate_layers = parse_ints(a.candidate_layers) if a.candidate_layers else None

    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(a.direction_dir) / "relation_vectors.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    z = np.load(npz_path, allow_pickle=True)
    required = {"sample_index", "relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{npz_path} missing arrays: {sorted(missing)}")

    sids = np.asarray(z["sample_index"]).astype(int)
    y = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)
    X = np.asarray(z["residual"], dtype=np.float32)

    if X.ndim != 4:
        raise RuntimeError(f"Expected residual [N,L,H,D], got {X.shape}")
    if len(sids) != len(y) or len(sids) != X.shape[0]:
        raise RuntimeError("sample_index/relation/residual size mismatch")

    valid = np.isin(y, np.asarray(REL, dtype=object))
    sids = sids[valid]
    y = y[valid]
    X = X[valid]

    N, L, H, D = X.shape
    print("=" * 150)
    print("10% PROPORTIONAL-STRATIFIED DIRECTION-HEAD SELECTOR")
    print("=" * 150)
    print(f"N={N} | residual={X.shape} | calib_frac={a.calib_frac:.3f}")
    print("dataset relation counts:", {r: int(np.sum(y == r)) for r in REL})
    print("candidate layers:", candidate_layers if candidate_layers else "ALL")
    print("seeds:", seeds)
    print()

    all_result_rows = []
    selected_rows = []
    split_rows = []
    pred_rows = []

    for seed in seeds:
        cal_idx, test_idx, quota = make_stratified_split(y, a.calib_frac, seed)
        Xcal, ycal = X[cal_idx], y[cal_idx]
        Xtest, ytest = X[test_idx], y[test_idx]

        print(f"[seed {seed}] calibration N={len(cal_idx)} test N={len(test_idx)}")
        print(
            "  calibration:",
            {r: int(np.sum(ycal == r)) for r in REL},
            "| test:",
            {r: int(np.sum(ytest == r)) for r in REL},
        )

        # Fit full calibration directions for held-out testing.
        dirs = fit_directions(Xcal, ycal)

        # Select heads only from the calibration split.
        if a.use_loo_head_ranking:
            acc_cal = loo_head_accuracy(Xcal, ycal)
            rank_method = "LOO_calibration_accuracy"
        else:
            acc_cal = in_sample_head_accuracy(Xcal, ycal, dirs)
            rank_method = "in_sample_calibration_accuracy"

        ranked = candidate_head_list(acc_cal, candidate_layers)
        test_scores = score_with_dirs(Xtest, dirs)

        # Store split membership.
        for i in cal_idx:
            split_rows.append({
                "seed": seed,
                "sid": int(sids[i]),
                "relation": y[i],
                "split": "calibration",
            })
        for i in test_idx:
            split_rows.append({
                "seed": seed,
                "sid": int(sids[i]),
                "relation": y[i],
                "split": "test",
            })

        # Top head table.
        for rank, (acc, l, h) in enumerate(ranked[:max(max(topks), 20)], start=1):
            selected_rows.append({
                "seed": seed,
                "rank": rank,
                "layer": l,
                "head": h,
                "head_name": head_name(l, h),
                "calibration_accuracy": acc,
                "ranking_method": rank_method,
            })

        for K in topks:
            selected = [(l, h) for _, l, h in ranked[:K]]

            for weighted in (False, True):
                probs = ensemble_probs(
                    test_scores,
                    selected,
                    acc_cal,
                    temperature=a.temperature,
                    weighted=weighted,
                )
                pred = np.argmax(probs, axis=-1)
                overall, per_rel = eval_pred(pred, ytest)

                method = f"top{K}_{'weighted' if weighted else 'equal'}"
                all_result_rows.append({
                    "seed": seed,
                    "method": method,
                    "N_calibration": len(cal_idx),
                    "N_test": len(test_idx),
                    "acc_all": overall,
                    "acc_left": per_rel["left"],
                    "acc_right": per_rel["right"],
                    "acc_on": per_rel["on"],
                    "acc_under": per_rel["under"],
                    "heads": ",".join(head_name(l, h) for l, h in selected),
                    "mean_calibration_head_acc": float(
                        np.mean([acc_cal[l, h] for l, h in selected])
                    ),
                })

                for local_i, global_i in enumerate(test_idx):
                    pred_rows.append({
                        "seed": seed,
                        "method": method,
                        "sid": int(sids[global_i]),
                        "gt": y[global_i],
                        "prediction": REL[int(pred[local_i])],
                        "correct": bool(REL[int(pred[local_i])] == y[global_i]),
                        "confidence": float(np.max(probs[local_i])),
                        "margin": float(
                            np.sort(probs[local_i])[-1] - np.sort(probs[local_i])[-2]
                        ),
                    })

        print(
            "  top heads:",
            ", ".join(
                f"{head_name(l,h)}={acc:.3f}"
                for acc, l, h in ranked[:5]
            ),
        )

    results = pd.DataFrame(all_result_rows)
    selected_df = pd.DataFrame(selected_rows)
    split_df = pd.DataFrame(split_rows)
    pred_df = pd.DataFrame(pred_rows)

    # Aggregate across random splits.
    agg_rows = []
    for method, g in results.groupby("method"):
        agg_rows.append({
            "method": method,
            "n_seeds": g["seed"].nunique(),
            "mean_acc": float(g["acc_all"].mean()),
            "std_acc": float(g["acc_all"].std(ddof=1)) if len(g) > 1 else 0.0,
            "min_acc": float(g["acc_all"].min()),
            "max_acc": float(g["acc_all"].max()),
            "mean_left": float(g["acc_left"].mean()),
            "mean_right": float(g["acc_right"].mean()),
            "mean_on": float(g["acc_on"].mean()),
            "mean_under": float(g["acc_under"].mean()),
        })
    aggregate = pd.DataFrame(agg_rows).sort_values(
        ["mean_acc", "std_acc"], ascending=[False, True]
    )

    # Head stability across seeds.
    stability_rows = []
    for K in sorted(set(topks)):
        topk = selected_df[selected_df["rank"] <= K]
        counts = Counter(topk["head_name"].tolist())
        for hn, n in counts.most_common():
            stability_rows.append({
                "K": K,
                "head_name": hn,
                "selected_in_seeds": n,
                "selection_frequency": n / float(len(seeds)),
            })
    stability = pd.DataFrame(stability_rows)

    # Exact proportional quota shown once (same counts for every seed).
    quota = proportional_quota(y, a.calib_frac)
    quota_df = pd.DataFrame([
        {
            "relation": r,
            "full_N": int(np.sum(y == r)),
            "full_fraction": float(np.mean(y == r)),
            "calibration_N": int(quota[r]),
            "calibration_fraction_of_44": quota[r] / float(sum(quota.values())),
            "test_N": int(np.sum(y == r)) - int(quota[r]),
        }
        for r in REL
    ])

    results.to_csv(outdir / "per_seed_selector_results.csv", index=False)
    aggregate.to_csv(outdir / "aggregate_selector_results.csv", index=False)
    selected_df.to_csv(outdir / "selected_heads_per_seed.csv", index=False)
    stability.to_csv(outdir / "head_selection_stability.csv", index=False)
    split_df.to_csv(outdir / "split_membership.csv", index=False)
    pred_df.to_csv(outdir / "heldout_predictions.csv", index=False)
    quota_df.to_csv(outdir / "stratified_quota.csv", index=False)

    meta = {
        "N": int(N),
        "residual_shape": list(X.shape),
        "calib_frac": float(a.calib_frac),
        "calibration_N": int(sum(quota.values())),
        "test_N": int(N - sum(quota.values())),
        "full_counts": {r: int(np.sum(y == r)) for r in REL},
        "calibration_quota": {r: int(quota[r]) for r in REL},
        "seeds": seeds,
        "topks": topks,
        "candidate_layers": candidate_layers,
        "head_ranking": (
            "leave-one-out calibration accuracy"
            if a.use_loo_head_ranking
            else "in-sample calibration accuracy"
        ),
        "direction_fit": "centered relation mean on 10% calibration only",
        "test_policy": "remaining 90% held out from both direction fit and head selection",
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 150)
    print("STRATIFIED QUOTA")
    print("=" * 150)
    print(quota_df.to_string(index=False))

    print()
    print("=" * 150)
    print("HELD-OUT 90% SELECTOR RESULTS")
    print("=" * 150)
    print(
        aggregate[
            [
                "method", "mean_acc", "std_acc",
                "min_acc", "max_acc",
                "mean_left", "mean_right", "mean_on", "mean_under"
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print()
    print("=" * 150)
    print("TOP-3 HEAD STABILITY")
    print("=" * 150)
    if len(stability):
        s3 = stability[stability["K"] == 3].sort_values(
            ["selected_in_seeds", "head_name"], ascending=[False, True]
        )
        print(s3.head(20).to_string(index=False))

    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
