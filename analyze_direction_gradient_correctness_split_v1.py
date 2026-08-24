#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process Direction-head gradient results into correctness-conditioned groups.

No model/GPU forward is required.

For each selected Direction head, this script:
  1) Fits a four-way residual Direction codebook on the existing TRAIN split.
  2) Predicts each untouched TEST sample for that head.
  3) Joins the head prediction with baseline generation correctness.
  4) Splits samples into four groups:
       Gen correct / Head correct
       Gen correct / Head wrong
       Gen wrong   / Head correct
       Gen wrong   / Head wrong
  5) Summarizes whole-head and object-token gradient/contribution metrics.

The most diagnostic comparison is usually:
    Gen-wrong & Head-correct  vs  Gen-correct & Head-correct
because both groups contain a Direction head that decoded the relation correctly;
the difference is whether the final model decision also succeeded.

Expected inputs are outputs from:
  - analyze_spatial_head_gradient_contribution_v1.py
  - analyze_coco_head_object_residual_direction_probe_v1.py
  - validate_grounded_spatial_consensus_v1.py

Designed for the current Qwen-3B / COCO_two experiment, but the post-processing
is model-agnostic as long as the file formats match.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

RELATIONS = ("left", "right", "above", "below")
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--gradient-dir",
        default="output/qwen3b_spatial_head_gradient_v1",
        help="Directory containing per_sample_head_metrics.csv from the gradient analysis.",
    )
    p.add_argument(
        "--direction-vectors-npz",
        default="output/qwen3b_coco_head_direction_residual/relation_vectors.npz",
        help="Existing residual relation_vectors.npz.",
    )
    p.add_argument(
        "--feasibility-dir",
        default="output/qwen3b_coco_grounded_consensus_v1",
        help="Directory containing split.csv and test_samples.csv.",
    )
    p.add_argument(
        "--heads",
        default="all_direction",
        help="Comma-separated LxHy names, or all_direction to use all Direction rows in gradient CSV.",
    )
    p.add_argument(
        "--output-dir",
        default="output/qwen3b_direction_gradient_correctness_v1",
    )
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap replicates for family-level 95%% CIs; 0 disables.")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_vals(xs: Iterable[float]) -> np.ndarray:
    a = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    return a


def mean(xs: Iterable[float]) -> float:
    a = safe_vals(xs)
    return float(a.mean()) if a.size else float("nan")


def median(xs: Iterable[float]) -> float:
    a = safe_vals(xs)
    return float(np.median(a)) if a.size else float("nan")


def std(xs: Iterable[float]) -> float:
    a = safe_vals(xs)
    return float(a.std()) if a.size else float("nan")


def positive_fraction(xs: Iterable[float]) -> float:
    a = safe_vals(xs)
    return float((a > 0).mean()) if a.size else float("nan")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, EPS)


def fit_codebook(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = X.mean(axis=0)
    Xc = X - center
    dirs = []
    for rel in RELATIONS:
        m = y == rel
        if not np.any(m):
            raise RuntimeError(f"No training samples for relation={rel}")
        d = Xc[m].mean(axis=0)
        d = d / max(float(np.linalg.norm(d)), EPS)
        dirs.append(d)
    return center.astype(np.float32), np.stack(dirs).astype(np.float32)


def predict_head(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center, dirs = fit_codebook(X_train, y_train)
    Xt = normalize_rows(X_test.astype(np.float32) - center[None, :])
    scores = Xt @ dirs.T
    pred = scores.argmax(axis=1)
    order = np.sort(scores, axis=1)
    margin = order[:, -1] - order[:, -2]
    return pred.astype(np.int64), margin.astype(np.float32)


METRICS = [
    # whole-head sensitivity/contribution
    "grad_norm_all",
    "grad_x_act_abs_all",
    "grad_x_act_signed_all",
    # object-token sensitivity/contribution
    "grad_norm_object_pair",
    "grad_x_act_object_pair",
    "relation_grad_norm",
    # spatially specific Direction residual contribution
    "residual_relation_contribution",
    "residual_relation_alignment",
]

GROUP_ORDER = [
    (True, True, "gen_correct__head_correct"),
    (True, False, "gen_correct__head_wrong"),
    (False, True, "gen_wrong__head_correct"),
    (False, False, "gen_wrong__head_wrong"),
]


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    for m in METRICS:
        vals = [as_float(r.get(m)) for r in rows]
        out[f"mean_{m}"] = mean(vals)
        out[f"median_{m}"] = median(vals)
        out[f"std_{m}"] = std(vals)
        out[f"mean_abs_{m}"] = mean(abs(v) for v in vals if math.isfinite(v))
        if m in {"grad_x_act_signed_all", "grad_x_act_object_pair", "residual_relation_contribution", "residual_relation_alignment"}:
            out[f"positive_fraction_{m}"] = positive_fraction(vals)
    return out


def bootstrap_mean_ci(vals: Sequence[float], n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    a = safe_vals(vals)
    if a.size < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, a.size, size=a.size)
        means[b] = a[idx].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    grad_dir = Path(args.gradient_dir)
    feas_dir = Path(args.feasibility_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grad_path = grad_dir / "per_sample_head_metrics.csv"
    split_path = feas_dir / "split.csv"
    test_samples_path = feas_dir / "test_samples.csv"
    vec_path = Path(args.direction_vectors_npz)
    for p in (grad_path, split_path, test_samples_path, vec_path):
        if not p.exists():
            raise FileNotFoundError(p)

    grad_rows_all = read_csv(grad_path)
    direction_rows = [r for r in grad_rows_all if str(r.get("family", "")).lower() == "direction"]
    if not direction_rows:
        raise RuntimeError("No family=direction rows in gradient CSV")

    all_heads = sorted(set(r["head_name"] for r in direction_rows), key=lambda s: (int(s.split("H")[0][1:]), int(s.split("H")[1])))
    if args.heads.strip().lower() == "all_direction":
        heads = all_heads
    else:
        heads = [x.strip() for x in args.heads.split(",") if x.strip()]
        missing = [h for h in heads if h not in all_heads]
        if missing:
            raise ValueError(f"Requested heads not present in gradient CSV: {missing}")

    # Split mapping: use exactly the train/test split from the feasibility experiment.
    split_rows = read_csv(split_path)
    sid_to_split = {int(r["sid"]): r["split"] for r in split_rows}
    train_sids = {sid for sid, sp in sid_to_split.items() if sp == "train"}
    test_sids = {sid for sid, sp in sid_to_split.items() if sp == "test"}

    # Baseline generation correctness on untouched test samples.
    test_rows = read_csv(test_samples_path)
    gen_by_sid: Dict[int, Dict[str, Any]] = {int(r["sid"]): r for r in test_rows}

    # Direction vectors + labels.
    z = np.load(vec_path, allow_pickle=False)
    residual = np.asarray(z["residual"], dtype=np.float32)  # [N,L,H,D]
    sids = np.asarray(z["sample_index"], dtype=np.int64)
    labels = np.asarray(z["relation"]).astype(str)
    sid_to_vec = {int(sid): i for i, sid in enumerate(sids.tolist())}

    train_idx = np.asarray([sid_to_vec[s] for s in sorted(train_sids) if s in sid_to_vec], dtype=np.int64)
    test_sid_order = [int(r["sid"]) for r in test_rows if int(r["sid"]) in test_sids and int(r["sid"]) in sid_to_vec]
    test_idx = np.asarray([sid_to_vec[s] for s in test_sid_order], dtype=np.int64)
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    gt_test_idx = np.asarray([REL_TO_IDX[x] for x in y_test], dtype=np.int64)

    # Gradient rows lookup by (sid, head).
    grad_lookup: Dict[Tuple[int, str], Dict[str, str]] = {}
    for r in direction_rows:
        sid = int(r["sid"])
        if sid in test_sids:
            grad_lookup[(sid, r["head_name"])] = r

    merged_rows: List[Dict[str, Any]] = []
    head_summaries: List[Dict[str, Any]] = []

    for head_name in heads:
        l = int(head_name.split("H")[0][1:])
        h = int(head_name.split("H")[1])
        if l >= residual.shape[1] or h >= residual.shape[2]:
            raise ValueError(f"{head_name} outside residual tensor {residual.shape}")

        X_train = residual[train_idx, l, h, :]
        X_test = residual[test_idx, l, h, :]
        pred_idx, probe_margin = predict_head(X_train, y_train, X_test)
        head_correct_arr = pred_idx == gt_test_idx
        test_acc = float(head_correct_arr.mean())

        probe_acc_existing = float("nan")
        for r in direction_rows:
            if r["head_name"] == head_name:
                probe_acc_existing = as_float(r.get("probe_accuracy"))
                break

        this_head_rows: List[Dict[str, Any]] = []
        for j, sid in enumerate(test_sid_order):
            grow = grad_lookup.get((sid, head_name))
            if grow is None:
                continue
            gen = gen_by_sid[sid]
            gen_correct = as_bool(gen.get("generation_correct"))
            head_correct = bool(head_correct_arr[j])
            row: Dict[str, Any] = dict(grow)
            row.update({
                "split": "test",
                "generation_correct_eval": int(gen_correct),
                "generation_prediction_eval": gen.get("generation_prediction", ""),
                "head_prediction": RELATIONS[int(pred_idx[j])],
                "head_correct": int(head_correct),
                "head_probe_margin": float(probe_margin[j]),
                "head_test_accuracy": test_acc,
                "correctness_group": (
                    "gen_correct__head_correct" if gen_correct and head_correct else
                    "gen_correct__head_wrong" if gen_correct and not head_correct else
                    "gen_wrong__head_correct" if (not gen_correct) and head_correct else
                    "gen_wrong__head_wrong"
                ),
            })
            merged_rows.append(row)
            this_head_rows.append(row)

        summary: Dict[str, Any] = {
            "head_name": head_name,
            "layer": l,
            "head": h,
            "probe_accuracy_repeated_original": probe_acc_existing,
            "probe_accuracy_current_train_test": test_acc,
            "n_test": len(this_head_rows),
        }

        for gen_c, head_c, gname in GROUP_ORDER:
            grp = [
                r for r in this_head_rows
                if bool(int(r["generation_correct_eval"])) == gen_c and bool(int(r["head_correct"])) == head_c
            ]
            gs = summarize_group(grp)
            for k, v in gs.items():
                summary[f"{gname}__{k}"] = v

        # Binary marginal comparisons too.
        for flag_key, true_name, false_name in [
            ("generation_correct_eval", "generation_correct", "generation_wrong"),
            ("head_correct", "head_correct", "head_wrong"),
        ]:
            for target, name in ((1, true_name), (0, false_name)):
                grp = [r for r in this_head_rows if int(r[flag_key]) == target]
                gs = summarize_group(grp)
                for k, v in gs.items():
                    summary[f"{name}__{k}"] = v

        # Direct deltas for the key comparison: both have head correct, final gen differs.
        ghc = [r for r in this_head_rows if int(r["generation_correct_eval"]) == 1 and int(r["head_correct"]) == 1]
        ghw = [r for r in this_head_rows if int(r["generation_correct_eval"]) == 0 and int(r["head_correct"]) == 1]
        for m in ["grad_norm_all", "grad_x_act_abs_all", "grad_x_act_signed_all", "grad_norm_object_pair", "residual_relation_contribution", "residual_relation_alignment"]:
            summary[f"delta_genWrongHeadCorrect_minus_genCorrectHeadCorrect__{m}"] = mean(as_float(r.get(m)) for r in ghw) - mean(as_float(r.get(m)) for r in ghc)

        head_summaries.append(summary)

    write_csv(out_dir / "per_sample_direction_gradient_with_correctness.csv", merged_rows)
    write_csv(out_dir / "head_fourway_summary.csv", head_summaries)

    # Family-level aggregate, treating each sample-head row as an observation for a descriptive view.
    family_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for gen_c, head_c, gname in GROUP_ORDER:
        grp = [
            r for r in merged_rows
            if bool(int(r["generation_correct_eval"])) == gen_c and bool(int(r["head_correct"])) == head_c
        ]
        s: Dict[str, Any] = {"group": gname, "n_sample_head_rows": len(grp)}
        for m in METRICS:
            vals = [as_float(r.get(m)) for r in grp]
            s[f"mean_{m}"] = mean(vals)
            s[f"median_{m}"] = median(vals)
            s[f"std_{m}"] = std(vals)
            s[f"mean_abs_{m}"] = mean(abs(v) for v in vals if math.isfinite(v))
            if m in {"grad_x_act_signed_all", "grad_x_act_object_pair", "residual_relation_contribution", "residual_relation_alignment"}:
                s[f"positive_fraction_{m}"] = positive_fraction(vals)
            if args.bootstrap > 0:
                lo, hi = bootstrap_mean_ci(vals, args.bootstrap, rng)
                s[f"mean_{m}_ci95_lo"] = lo
                s[f"mean_{m}_ci95_hi"] = hi
        family_rows.append(s)
    write_csv(out_dir / "family_fourway_summary.csv", family_rows)

    summary_obj = {
        "heads": heads,
        "n_heads": len(heads),
        "n_test": len(test_sid_order),
        "train_n": int(len(train_idx)),
        "definition": {
            "head_correct": "four-way residual Direction prototype prediction, codebook fit only on feasibility TRAIN split and evaluated on untouched TEST",
            "generation_correct": "baseline generation correctness from feasibility test_samples.csv",
            "key_comparison": "gen_wrong & head_correct versus gen_correct & head_correct",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compact console report.
    print("\n" + "=" * 150)
    print("DIRECTION GRADIENT: GENERATION CORRECTNESS x HEAD CORRECTNESS")
    print("=" * 150)
    print(f"TEST N={len(test_sid_order)}  heads={len(heads)}")
    print("Key metrics: |g*x|all = whole-head absolute first-order contribution; obj|grad| = object-pair sensitivity; |res g.r| / signed res g.r / res cos = spatial residual utilization")
    print("-")
    hdr = (
        "head      testACC   group       n    |g*x|all   obj|grad|   |res g.r|   signed res g.r   res cos"
    )
    print(hdr)
    for hs in head_summaries:
        first = True
        for _gc, _hc, gname in GROUP_ORDER:
            short = {
                "gen_correct__head_correct": "G+ H+",
                "gen_correct__head_wrong": "G+ H-",
                "gen_wrong__head_correct": "G- H+",
                "gen_wrong__head_wrong": "G- H-",
            }[gname]
            n = hs.get(f"{gname}__n", 0)
            gx = hs.get(f"{gname}__mean_grad_x_act_abs_all", float("nan"))
            og = hs.get(f"{gname}__mean_grad_norm_object_pair", float("nan"))
            rg_abs = hs.get(f"{gname}__mean_abs_residual_relation_contribution", float("nan"))
            rg = hs.get(f"{gname}__mean_residual_relation_contribution", float("nan"))
            rc = hs.get(f"{gname}__mean_residual_relation_alignment", float("nan"))
            if first:
                print(f"{hs['head_name']:<8s} {hs['probe_accuracy_current_train_test']:8.4f}  {short:<7s} {n:4d}  {gx:10.5f}  {og:10.5f}  {rg_abs:10.5f}  {rg:14.6f}  {rc:8.5f}")
                first = False
            else:
                print(f"{'':<8s} {'':8s}  {short:<7s} {n:4d}  {gx:10.5f}  {og:10.5f}  {rg_abs:10.5f}  {rg:14.6f}  {rc:8.5f}")
        print("-")

    print("\nFamily-level four-way aggregate:")
    print("group     nrows   |g*x|all   obj|grad|   |res g.r|   signed res g.r   res cos")
    for r in family_rows:
        print(
            f"{r['group']:<22s} {int(r['n_sample_head_rows']):5d} "
            f"{float(r.get('mean_grad_x_act_abs_all', float('nan'))):10.5f} "
            f"{float(r.get('mean_grad_norm_object_pair', float('nan'))):10.5f} "
            f"{float(r.get('mean_abs_residual_relation_contribution', float('nan'))):10.5f} "
            f"{float(r.get('mean_residual_relation_contribution', float('nan'))):14.6f} "
            f"{float(r.get('mean_residual_relation_alignment', float('nan'))):8.5f}"
        )

    print("\nSaved:")
    for name in [
        "per_sample_direction_gradient_with_correctness.csv",
        "head_fourway_summary.csv",
        "family_fourway_summary.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
