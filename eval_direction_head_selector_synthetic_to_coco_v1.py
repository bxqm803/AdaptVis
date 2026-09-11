#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_selector_synthetic_to_coco_v1.py

Cross-dataset Direction-Head selector:

    Synthetic spatial dataset (e.g. N=400)
        -> identify reliable Direction Heads
        -> fit each head's 4 relation directions
        -> freeze everything
        -> evaluate on ALL COCO_two samples (e.g. N=440)

No VLM parameter update / no SGD.
No COCO label is used for fitting relation directions or selecting heads.

Expected source/target cache format
-----------------------------------
Both source and target should come from the same Direction-Head extraction
pipeline and contain relation_vectors.npz with:

    sample_index : [N]                    (or sid)
    relation     : [N]
    residual     : [N, n_layers, n_heads, head_dim]

where residual is the Real-Gray subject-reference per-head residual used by the
Direction-Head probe.

Why source CV?
--------------
The source Synthetic-400 is completely disjoint from COCO, so using all 400
source examples would not leak target labels.  Still, to avoid choosing heads
because of source-set resubstitution, this script ranks heads by STRATIFIED
K-FOLD out-of-fold accuracy on Synthetic.  After head selection, relation
directions are refit on ALL source samples and transferred unchanged to COCO.

Selector
--------
For each head h:

    d_{h,r} = mean(z_h | relation=r) - mean(z_h)

For target sample x:

    e_{h,r}(x) = cos(z_h(x) - g_h, d_{h,r})

Each selected head gives a 4-way relation distribution. We test:
    Top1, Top3, Top5, Top10, Top20
    equal ensemble
    source-CV-accuracy weighted ensemble
    layer-balanced variants

Typical command
---------------
python -u eval_direction_head_selector_synthetic_to_coco_v1.py \
  --source-direction-dir output/qwen3b_synthetic_head_direction \
  --target-direction-dir output/qwen3b_head_object_residual_direction \
  --source-name synthetic400 \
  --target-name coco440 \
  --cv-folds 5 \
  --topks 1,3,5,10,20 \
  --target-sample-csv output/qwen3b_four_writer_k7_N80/per_sample_direction_scores.csv \
  --output-dir output/qwen3b_direction_selector_synthetic_to_coco

For full COCO-440 baseline-correct/wrong analysis, pass a sample CSV containing
all 440 SIDs; otherwise omit --target-sample-csv.
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


# =============================================================================
# Args / normalization
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--source-direction-dir", required=True)
    p.add_argument("--target-direction-dir", required=True)
    p.add_argument("--source-name", default="synthetic400")
    p.add_argument("--target-name", default="coco440")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=17)
    p.add_argument("--topks", default="1,3,5,10,20")
    p.add_argument(
        "--candidate-layers",
        default="",
        help="Optional head layers, e.g. 19,20,21,22,23,24,25,26,27. Empty=all.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--target-sample-csv",
        default="",
        help="Optional CSV with sid/gt/baseline_prediction/baseline_correct.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return {
        "left": "left", "left_of": "left", "left of": "left", "l": "left",
        "right": "right", "right_of": "right", "right of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on", "up": "on",
        "below": "under", "under": "under", "beneath": "under", "bottom": "under",
        "down": "under",
    }.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def head_name(L, H):
    return f"L{int(L)}H{int(H):02d}"


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


# =============================================================================
# NPZ loading
# =============================================================================

def resolve_npz(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    q = p / "relation_vectors.npz"
    if not q.exists():
        raise FileNotFoundError(q)
    return q


def load_relation_npz(path_or_dir, name):
    p = resolve_npz(path_or_dir)
    z = np.load(p, allow_pickle=True)

    required = {"relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{name}: {p} missing required arrays {sorted(missing)}; "
            f"found={sorted(z.files)}"
        )

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray([canon_rel(v) for v in z["relation"]], dtype=object)

    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    elif "sid" in z.files:
        # Allow numeric-like strings.
        raw = np.asarray(z["sid"])
        try:
            sid = raw.astype(int)
        except Exception:
            sid = np.arange(len(raw), dtype=int)
    else:
        sid = np.arange(len(y), dtype=int)

    if X.ndim != 4:
        raise RuntimeError(
            f"{name}: expected residual [N,L,H,D], got {X.shape}"
        )
    if len(y) != X.shape[0] or len(sid) != X.shape[0]:
        raise RuntimeError(
            f"{name}: sample_index/relation/residual length mismatch"
        )

    valid = np.isin(y, np.asarray(REL, dtype=object))
    X = X[valid]
    y = y[valid]
    sid = sid[valid]

    if len(X) == 0:
        raise RuntimeError(f"{name}: no usable 4-relation samples.")

    return {
        "path": str(p),
        "X": X,
        "y": y,
        "sid": sid,
        "counts": {r: int(np.sum(y == r)) for r in REL},
    }


# =============================================================================
# Directions / scoring
# =============================================================================

def fit_codebook(X, y):
    """
    Per head:
      center g_h = source mean
      direction d_{h,r} = normalized(mean(z_h|r)-g_h)

    X [N,L,H,D]
    center [L,H,D]
    dirs [L,H,R,D]
    """
    center = X.mean(axis=0)
    L, H, D = X.shape[1:]
    dirs = np.zeros((L, H, len(REL), D), dtype=np.float32)

    for ri, r in enumerate(REL):
        m = (y == r)
        if not np.any(m):
            raise RuntimeError(f"Training source missing relation={r}")
        d = X[m].mean(axis=0) - center
        dirs[:, :, ri, :] = normalize(d, axis=-1)

    return center.astype(np.float32), dirs


def score_codebook(X, center, dirs):
    """
    Test:
      cos(z_h - g_h, d_{h,r})
    returns [N,L,H,R]
    """
    Xc = X - center[None, :, :, :]
    Xn = normalize(Xc, axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", Xn, dirs, optimize=True)


def accuracy_from_scores(scores, y):
    pred = np.argmax(scores, axis=-1)  # [N,L,H]
    yi = np.asarray([REL.index(r) for r in y], dtype=int)
    return (pred == yi[:, None, None]).mean(axis=0).astype(np.float32)


# =============================================================================
# Source stratified CV for unbiased head ranking
# =============================================================================

def stratified_folds(y, n_folds, seed):
    """
    Return list of test-index arrays. Each relation is shuffled separately,
    then split approximately equally across folds.
    """
    y = np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_folds)]

    for r in REL:
        idx = np.where(y == r)[0].copy()
        rng.shuffle(idx)
        pieces = np.array_split(idx, n_folds)
        for f, piece in enumerate(pieces):
            buckets[f].extend(piece.tolist())

    return [np.asarray(sorted(b), dtype=int) for b in buckets]


def source_oof_head_accuracy(X, y, n_folds, seed):
    N, L, H, D = X.shape
    folds = stratified_folds(y, n_folds, seed)
    yi = np.asarray([REL.index(r) for r in y], dtype=int)

    correct = np.zeros((L, H), dtype=np.int64)
    seen = np.zeros((L, H), dtype=np.int64)

    fold_rows = []

    all_idx = np.arange(N)
    for f, te in enumerate(folds):
        mask = np.ones(N, dtype=bool)
        mask[te] = False
        tr = all_idx[mask]

        center, dirs = fit_codebook(X[tr], y[tr])
        scores = score_codebook(X[te], center, dirs)
        pred = np.argmax(scores, axis=-1)

        c = (pred == yi[te, None, None])
        correct += c.sum(axis=0)
        seen += len(te)

        fold_acc = c.mean(axis=0)
        fold_rows.append((f, len(tr), len(te), fold_acc))

    acc = correct.astype(np.float32) / np.maximum(seen, 1)

    # Per-fold head table can be useful for stability diagnostics.
    rows = []
    for f, ntr, nte, fa in fold_rows:
        for l in range(L):
            for h in range(H):
                rows.append({
                    "fold": f,
                    "N_train": ntr,
                    "N_val": nte,
                    "layer": l,
                    "head": h,
                    "head_name": head_name(l, h),
                    "fold_accuracy": float(fa[l, h]),
                })
    return acc, pd.DataFrame(rows)


# =============================================================================
# Head selection / ensemble
# =============================================================================

def ranked_heads(acc, candidate_layers=None):
    L, H = acc.shape
    allowed = set(candidate_layers) if candidate_layers else None
    rows = []
    for l in range(L):
        if allowed is not None and l not in allowed:
            continue
        for h in range(H):
            rows.append((float(acc[l, h]), int(l), int(h)))
    rows.sort(key=lambda t: t[0], reverse=True)
    return rows


def ensemble_probs(scores, heads, reliability, temperature, weighted, layer_balanced):
    """
    Convert each head's 4 relation cosines to softmax first, then aggregate.
    """
    probs = softmax(scores, axis=-1, temperature=temperature)
    N = scores.shape[0]

    if not heads:
        raise RuntimeError("No selected heads.")

    if not layer_balanced:
        out = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0
        for l, h in heads:
            w = max(float(reliability[l, h]) - 0.25, 0.0) if weighted else 1.0
            if w <= 0:
                continue
            out += w * probs[:, l, h, :]
            denom += w
        if denom <= EPS:
            return np.mean(
                np.stack([probs[:, l, h, :] for l, h in heads], axis=0),
                axis=0,
            )
        return out / denom

    by_layer = defaultdict(list)
    for l, h in heads:
        by_layer[l].append(h)

    layer_out = []
    for l, hs in sorted(by_layer.items()):
        out = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0
        for h in hs:
            w = max(float(reliability[l, h]) - 0.25, 0.0) if weighted else 1.0
            if w <= 0:
                continue
            out += w * probs[:, l, h, :]
            denom += w
        if denom <= EPS:
            out = np.mean(probs[:, l, hs, :], axis=1)
        else:
            out /= denom
        layer_out.append(out)

    return np.mean(np.stack(layer_out, axis=0), axis=0)


def evaluate_preds(pred_idx, y):
    yi = np.asarray([REL.index(r) for r in y], dtype=int)
    correct = pred_idx == yi
    row = {
        "acc_all": float(correct.mean()),
        "N": int(len(y)),
    }
    for ri, r in enumerate(REL):
        m = yi == ri
        row[f"N_{r}"] = int(m.sum())
        row[f"acc_{r}"] = float(correct[m].mean()) if m.any() else np.nan
    return row, correct


# =============================================================================
# Optional baseline-correct / baseline-wrong merge
# =============================================================================

def load_target_baseline(csv_path):
    if not csv_path:
        return None

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    if "sid" not in df.columns:
        raise RuntimeError(f"{p}: missing sid column")

    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)

    if "gt" in df.columns:
        df["gt"] = df["gt"].map(canon_rel)
    if "baseline_prediction" in df.columns:
        df["baseline_prediction"] = df["baseline_prediction"].map(canon_rel)

    if "baseline_correct" in df.columns:
        df["baseline_correct"] = df["baseline_correct"].map(boolify)
    elif {"gt", "baseline_prediction"} <= set(df.columns):
        df["baseline_correct"] = df["gt"] == df["baseline_prediction"]
    else:
        raise RuntimeError(
            f"{p}: need baseline_correct or gt+baseline_prediction"
        )

    return df.drop_duplicates("sid").set_index("sid")


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    topks = parse_ints(a.topks)
    candidate_layers = (
        parse_ints(a.candidate_layers) if a.candidate_layers else None
    )

    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    src = load_relation_npz(a.source_direction_dir, a.source_name)
    tgt = load_relation_npz(a.target_direction_dir, a.target_name)

    Xs, ys, sids_s = src["X"], src["y"], src["sid"]
    Xt, yt, sids_t = tgt["X"], tgt["y"], tgt["sid"]

    if Xs.shape[1:] != Xt.shape[1:]:
        raise RuntimeError(
            "Source/target residual geometry mismatch:\n"
            f"  source={Xs.shape}\n"
            f"  target={Xt.shape}\n"
            "They must come from the same model / head extraction definition."
        )

    if a.cv_folds < 2:
        raise ValueError("--cv-folds must be >=2")

    min_class = min(src["counts"].values())
    if min_class < a.cv_folds:
        raise RuntimeError(
            f"Source smallest relation has {min_class} samples, "
            f"cannot do {a.cv_folds}-fold stratified CV."
        )

    print("=" * 160)
    print("SYNTHETIC -> COCO DIRECTION-HEAD SELECTOR")
    print("=" * 160)
    print(f"source: {a.source_name} | {src['path']} | shape={Xs.shape}")
    print("source counts:", src["counts"])
    print(f"target: {a.target_name} | {tgt['path']} | shape={Xt.shape}")
    print("target counts:", tgt["counts"])
    print(f"source head ranking: {a.cv_folds}-fold stratified OOF")
    print("candidate layers:", candidate_layers if candidate_layers else "ALL")
    print()

    # 1) Source-only OOF head reliability.
    cv_acc, fold_head_df = source_oof_head_accuracy(
        Xs, ys, a.cv_folds, a.cv_seed
    )
    ranked = ranked_heads(cv_acc, candidate_layers)

    reliability_rows = []
    for rank, (acc, l, h) in enumerate(ranked, start=1):
        reliability_rows.append({
            "rank": rank,
            "layer": l,
            "head": h,
            "head_name": head_name(l, h),
            "source_oof_accuracy": acc,
            "weight_over_chance": max(acc - 0.25, 0.0),
        })
    reliability_df = pd.DataFrame(reliability_rows)

    # 2) Fit final relation directions on ALL source data.
    center_full, dirs_full = fit_codebook(Xs, ys)

    # 3) Frozen transfer to ALL target samples.
    target_scores = score_codebook(Xt, center_full, dirs_full)

    baseline = load_target_baseline(a.target_sample_csv)

    result_rows = []
    detail = pd.DataFrame({
        "sid": sids_t.astype(int),
        "gt": yt,
    })

    for K in topks:
        selected = [(l, h) for _, l, h in ranked[:K]]
        selected_names = ",".join(head_name(l, h) for l, h in selected)

        for weighted in (False, True):
            for layer_balanced in (False, True):
                probs = ensemble_probs(
                    target_scores,
                    selected,
                    cv_acc,
                    a.temperature,
                    weighted=weighted,
                    layer_balanced=layer_balanced,
                )
                pred_idx = np.argmax(probs, axis=-1)
                pred_rel = np.asarray([REL[i] for i in pred_idx], dtype=object)

                method = (
                    f"top{K}_"
                    + ("weighted" if weighted else "equal")
                    + ("_layerbalanced" if layer_balanced else "")
                )

                metrics, correct = evaluate_preds(pred_idx, yt)
                row = {
                    "method": method,
                    **metrics,
                    "source_cv_mean_head_acc": safe_mean(
                        cv_acc[l, h] for l, h in selected
                    ),
                    "heads": selected_names,
                }

                # Optional split by target model baseline correctness.
                if baseline is not None:
                    bc = []
                    valid_b = []
                    for sid in sids_t:
                        sid = int(sid)
                        if sid in baseline.index:
                            val = baseline.loc[sid, "baseline_correct"]
                            if isinstance(val, pd.Series):
                                val = val.iloc[0]
                            bc.append(bool(val))
                            valid_b.append(True)
                        else:
                            bc.append(False)
                            valid_b.append(False)

                    bc = np.asarray(bc, dtype=bool)
                    valid_b = np.asarray(valid_b, dtype=bool)
                    c_mask = valid_b & bc
                    w_mask = valid_b & (~bc)

                    row["N_with_baseline"] = int(valid_b.sum())
                    row["N_baseline_correct"] = int(c_mask.sum())
                    row["N_baseline_wrong"] = int(w_mask.sum())
                    row["acc_baseline_correct"] = (
                        float(correct[c_mask].mean()) if c_mask.any() else np.nan
                    )
                    row["acc_baseline_wrong"] = (
                        float(correct[w_mask].mean()) if w_mask.any() else np.nan
                    )

                    # Compare selector prediction to baseline relation if available.
                    if "baseline_prediction" in baseline.columns:
                        bp = np.asarray([
                            canon_rel(baseline.loc[int(sid), "baseline_prediction"])
                            if int(sid) in baseline.index else ""
                            for sid in sids_t
                        ], dtype=object)
                        row["match_baseline_all"] = (
                            float(np.mean(pred_rel[valid_b] == bp[valid_b]))
                            if valid_b.any() else np.nan
                        )
                        row["match_baseline_wrong"] = (
                            float(np.mean(pred_rel[w_mask] == bp[w_mask]))
                            if w_mask.any() else np.nan
                        )

                result_rows.append(row)

                detail[f"pred_{method}"] = pred_rel
                detail[f"correct_{method}"] = correct
                detail[f"confidence_{method}"] = probs.max(axis=1)
                sp = np.sort(probs, axis=1)
                detail[f"margin_{method}"] = sp[:, -1] - sp[:, -2]

    results = pd.DataFrame(result_rows).sort_values(
        ["acc_all", "source_cv_mean_head_acc"],
        ascending=False,
    )

    # Head stability across source CV folds:
    # for each fold, rank heads by that fold's held-out accuracy.
    stability_rows = []
    for K in sorted(set(topks + [3])):
        counts = Counter()
        for fold, g in fold_head_df.groupby("fold"):
            gg = g.copy()
            if candidate_layers:
                gg = gg[gg["layer"].isin(candidate_layers)]
            gg = gg.sort_values("fold_accuracy", ascending=False).head(K)
            counts.update(gg["head_name"].tolist())
        for hn, n in counts.most_common():
            stability_rows.append({
                "K": K,
                "head_name": hn,
                "selected_in_folds": int(n),
                "selection_frequency": float(n / a.cv_folds),
            })
    stability_df = pd.DataFrame(stability_rows)

    # Save.
    reliability_df.to_csv(outdir / "source_oof_head_reliability.csv", index=False)
    fold_head_df.to_csv(outdir / "source_fold_head_accuracy.csv", index=False)
    stability_df.to_csv(outdir / "source_head_selection_stability.csv", index=False)
    results.to_csv(outdir / "synthetic_to_coco_selector_summary.csv", index=False)
    detail.to_csv(outdir / "synthetic_to_coco_predictions.csv", index=False)

    np.savez_compressed(
        outdir / "synthetic_fitted_direction_codebook.npz",
        center=center_full,
        directions=dirs_full,
        relation_names=np.asarray(REL, dtype=object),
        source_name=np.asarray([a.source_name], dtype=object),
        source_path=np.asarray([src["path"]], dtype=object),
    )

    metadata = {
        "source_name": a.source_name,
        "target_name": a.target_name,
        "source_path": src["path"],
        "target_path": tgt["path"],
        "source_shape": list(Xs.shape),
        "target_shape": list(Xt.shape),
        "source_counts": src["counts"],
        "target_counts": tgt["counts"],
        "head_ranking": f"{a.cv_folds}-fold stratified source-only OOF accuracy",
        "direction_fit": "all source samples after source-only head selection",
        "target_label_usage": "evaluation only",
        "topks": topks,
        "candidate_layers": candidate_layers,
        "temperature": float(a.temperature),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Terminal report.
    print()
    print("=" * 160)
    print("SOURCE-ONLY TOP HEADS")
    print("=" * 160)
    print(
        reliability_df.head(20)[
            ["rank", "head_name", "source_oof_accuracy", "weight_over_chance"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print()
    print("=" * 160)
    print("FROZEN SYNTHETIC -> COCO RESULTS")
    print("=" * 160)

    show_cols = [
        "method", "acc_all",
        "acc_left", "acc_right", "acc_on", "acc_under",
    ]
    if baseline is not None:
        show_cols += [
            "N_with_baseline",
            "acc_baseline_correct",
            "acc_baseline_wrong",
        ]

    print(
        results[show_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 160)
    print("SOURCE TOP-3 STABILITY ACROSS CV FOLDS")
    print("=" * 160)
    s3 = stability_df[stability_df["K"] == 3].sort_values(
        ["selected_in_folds", "head_name"],
        ascending=[False, True],
    )
    print(s3.head(20).to_string(index=False))

    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
