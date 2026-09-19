#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find an Introduction-friendly pair of COCO samples with:

  * the same GT spatial relation (default: left),
  * one final-correct sample and one sample that finally answers the opposite
    relation (default: right),
  * closely matched *multi-layer* spatial evidence before the decision forms,
  * optionally closely matched early decision evidence,
  * clearly divergent late decision evidence.

This version is designed for the paper figure:

  Panel A: two curves only
      target-vs-opposite SPATIAL evidence across decoder depth
      (Final correct vs Final wrong -> opposite)

  Panel B: two curves only
      target-vs-opposite DECISION evidence across decoder depth
      (Final correct vs Final wrong -> opposite)

Unlike v1, pair selection is NOT based on a single layer.  It matches the
entire spatial trajectory over --spatial-match-layers, and can also match the
pre-decision trajectory over --decision-match-layers.  Late decision behavior
is only used as a sign constraint / tie-break, so the illustrative pair is
chosen for similar upstream states rather than for the prettiest late split.

Input
-----
Reads the per-sample CSV produced by:
  eval_qwen3b_coco_spatial_margin_matched_v1.py

Expected columns include:
  sid, layer, gt, mapping, final_prediction, final_correct,
  spatial_score_left/right/above/below,
  decision_score_A/B/C/D

Example
-------
python find_intro_spatial_pair_multilayer_v2.py \
  --per-sample-csv output/qwen3b_coco_spatial_margin_matched_v1/per_sample_layer.csv \
  --target-relation left \
  --wrong-relation right \
  --spatial-match-layers 19 20 21 22 23 24 25 \
  --decision-match-layers 22 23 24 25 26 \
  --late-layers 27 28 29 30 31 32 \
  --min-spatial-positive-frac 0.80 \
  --min-spatial-mean-gap 0.05 \
  --late-sign-threshold 0.05 \
  --topk 30 \
  --output-dir output/intro_left_pair_multilayer_v2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REL = ("left", "right", "above", "below")
DISPLAY_REL = {"left": "Left", "right": "Right", "above": "On", "below": "Under"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--per-sample-csv", required=True)
    p.add_argument("--target-relation", default="left", choices=REL)
    p.add_argument("--wrong-relation", default="right", choices=REL)

    # Multi-layer matching windows.
    p.add_argument(
        "--spatial-match-layers", nargs="+", type=int,
        default=[19, 20, 21, 22, 23, 24, 25],
        help="Layers whose target-vs-opposite spatial trajectory is matched between the two samples.",
    )
    p.add_argument(
        "--decision-match-layers", nargs="+", type=int,
        default=[22, 23, 24, 25, 26],
        help="Optional pre-decision layers whose target-vs-opposite decision trajectory should also be similar.",
    )
    p.add_argument(
        "--late-layers", nargs="+", type=int,
        default=[27, 28, 29, 30, 31, 32],
        help="Layers used only to verify/tie-break the final decision split.",
    )

    # Candidate constraints.
    p.add_argument(
        "--min-spatial-positive-frac", type=float, default=0.80,
        help="Fraction of spatial-match layers where target-vs-opposite spatial evidence must be > 0.",
    )
    p.add_argument(
        "--min-spatial-mean-gap", type=float, default=0.05,
        help="Minimum mean target-vs-opposite spatial evidence over the spatial-match window for each sample.",
    )
    p.add_argument(
        "--late-sign-threshold", type=float, default=0.05,
        help=(
            "Require mean late decision gap > +threshold for the correct sample and < -threshold "
            "for the wrong sample. Set <0 to disable this sign constraint."
        ),
    )
    p.add_argument(
        "--max-spatial-rmse", type=float, default=-1.0,
        help="Optional hard caliper on multi-layer spatial trajectory RMSE; <0 disables.",
    )
    p.add_argument(
        "--max-predecision-rmse", type=float, default=-1.0,
        help="Optional hard caliper on pre-decision trajectory RMSE; <0 disables.",
    )

    # Ranking weights: upstream matching dominates; late separation is only a tie-break.
    p.add_argument("--decision-match-weight", type=float, default=0.50)
    p.add_argument("--spatial-mean-weight", type=float, default=0.25)
    p.add_argument("--late-tiebreak-weight", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def canonical_relation(x: str) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "left": "left", "left_of": "left",
        "right": "right", "right_of": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    if s not in table:
        raise ValueError(f"Unknown relation: {x!r}")
    return table[s]


def relation_label(r: str) -> str:
    return DISPLAY_REL.get(r, r.title())


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def as_int(row: Mapping[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def parse_mapping(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(s).split(","):
        if "->" not in item:
            continue
        rel, letter = item.split("->", 1)
        out[canonical_relation(rel)] = letter.strip()
    if set(out) != set(REL):
        raise ValueError(f"Could not parse mapping {s!r}; parsed={out}")
    return out


def prediction_relation(row: Mapping[str, str]) -> str:
    mp = parse_mapping(row["mapping"])
    inv = {letter: rel for rel, letter in mp.items()}
    pred = str(row["final_prediction"]).strip()
    if pred not in inv:
        raise ValueError(f"Prediction {pred!r} not in mapping {mp}")
    return inv[pred]


def group_by_sid(rows: Sequence[Mapping[str, str]]) -> Dict[int, Dict[int, Dict[str, str]]]:
    out: Dict[int, Dict[int, Dict[str, str]]] = {}
    for r0 in rows:
        r = dict(r0)
        sid = as_int(r, "sid")
        layer = as_int(r, "layer")
        out.setdefault(sid, {})[layer] = r
    return out


def ensure_layers(sample: Mapping[int, Mapping[str, str]], layers: Sequence[int]) -> bool:
    return all(int(L) in sample for L in layers)


def spatial_gap(row: Mapping[str, str], target: str, opposite: str) -> float:
    return as_float(row, f"spatial_score_{target}") - as_float(row, f"spatial_score_{opposite}")


def decision_relation_score(row: Mapping[str, str], relation: str) -> float:
    mp = parse_mapping(row["mapping"])
    letter = mp[relation]
    return as_float(row, f"decision_score_{letter}")


def decision_gap(row: Mapping[str, str], target: str, opposite: str) -> float:
    return decision_relation_score(row, target) - decision_relation_score(row, opposite)


def trace(sample: Mapping[int, Mapping[str, str]], layers: Sequence[int], fn) -> np.ndarray:
    return np.asarray([fn(sample[int(L)]) for L in layers], dtype=np.float64)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def candidate_info(
    sid: int,
    sample: Mapping[int, Mapping[str, str]],
    target: str,
    opposite: str,
    spatial_layers: Sequence[int],
    decision_match_layers: Sequence[int],
    late_layers: Sequence[int],
) -> Dict[str, Any]:
    ref = sample[int(spatial_layers[len(spatial_layers) // 2])]
    sp = trace(sample, spatial_layers, lambda r: spatial_gap(r, target, opposite))
    pre_dec = trace(sample, decision_match_layers, lambda r: decision_gap(r, target, opposite))
    late_dec = trace(sample, late_layers, lambda r: decision_gap(r, target, opposite))
    pred_rel = prediction_relation(ref)
    return {
        "sid": sid,
        "gt": canonical_relation(ref["gt"]),
        "gt_option": str(ref.get("gt_option", "")),
        "mapping": str(ref["mapping"]),
        "final_prediction_letter": str(ref["final_prediction"]),
        "final_prediction_relation": pred_rel,
        "final_correct": as_int(ref, "final_correct"),
        "spatial_positive_frac": float(np.mean(sp > 0)),
        "spatial_mean_gap": float(sp.mean()),
        "spatial_min_gap": float(sp.min()),
        "spatial_max_gap": float(sp.max()),
        "predecision_mean_gap": float(pre_dec.mean()),
        "late_decision_mean_gap": float(late_dec.mean()),
        "late_decision_min_gap": float(late_dec.min()),
        "late_decision_max_gap": float(late_dec.max()),
    }


def pair_metrics(
    c_sample: Mapping[int, Mapping[str, str]],
    w_sample: Mapping[int, Mapping[str, str]],
    target: str,
    opposite: str,
    spatial_layers: Sequence[int],
    decision_match_layers: Sequence[int],
    late_layers: Sequence[int],
) -> Dict[str, float]:
    c_sp = trace(c_sample, spatial_layers, lambda r: spatial_gap(r, target, opposite))
    w_sp = trace(w_sample, spatial_layers, lambda r: spatial_gap(r, target, opposite))
    c_pre = trace(c_sample, decision_match_layers, lambda r: decision_gap(r, target, opposite))
    w_pre = trace(w_sample, decision_match_layers, lambda r: decision_gap(r, target, opposite))
    c_late = trace(c_sample, late_layers, lambda r: decision_gap(r, target, opposite))
    w_late = trace(w_sample, late_layers, lambda r: decision_gap(r, target, opposite))

    return {
        "spatial_multilayer_rmse": rmse(c_sp, w_sp),
        "spatial_mean_gap_difference": float(abs(c_sp.mean() - w_sp.mean())),
        "predecision_multilayer_rmse": rmse(c_pre, w_pre),
        "predecision_mean_gap_difference": float(abs(c_pre.mean() - w_pre.mean())),
        "correct_late_mean": float(c_late.mean()),
        "wrong_late_mean": float(w_late.mean()),
        "late_decision_separation": float(c_late.mean() - w_late.mean()),
    }


def rank_score(m: Mapping[str, float], a: argparse.Namespace) -> float:
    # Smaller is better.  Matching the upstream trajectories is primary.
    # Late separation only breaks near-ties among already matched pairs.
    return (
        float(m["spatial_multilayer_rmse"])
        + a.decision_match_weight * float(m["predecision_multilayer_rmse"])
        + a.spatial_mean_weight * float(m["spatial_mean_gap_difference"])
        - a.late_tiebreak_weight * float(m["late_decision_separation"])
    )


def save_two_curve_figure(
    path: Path,
    c_sample: Mapping[int, Mapping[str, str]],
    w_sample: Mapping[int, Mapping[str, str]],
    all_layers: Sequence[int],
    spatial_match_layers: Sequence[int],
    decision_match_layers: Sequence[int],
    late_layers: Sequence[int],
    target: str,
    opposite: str,
    c_sid: int,
    w_sid: int,
    best: Mapping[str, Any],
) -> None:
    x = np.asarray(all_layers, dtype=np.int64)
    c_sp = trace(c_sample, all_layers, lambda r: spatial_gap(r, target, opposite))
    w_sp = trace(w_sample, all_layers, lambda r: spatial_gap(r, target, opposite))
    c_dec = trace(c_sample, all_layers, lambda r: decision_gap(r, target, opposite))
    w_dec = trace(w_sample, all_layers, lambda r: decision_gap(r, target, opposite))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), dpi=280, sharex=True)

    # Spatial panel: exactly two trajectories.
    axes[0].axvspan(
        min(spatial_match_layers) - 0.45,
        max(spatial_match_layers) + 0.45,
        alpha=0.08,
        color="0.45",
        zorder=0,
    )
    axes[0].plot(x, c_sp, marker="o", linewidth=2.3, markersize=4.2, label=f"Final correct (sid={c_sid})")
    axes[0].plot(x, w_sp, marker="o", linewidth=2.3, markersize=4.2, linestyle="--",
                 label=f"Final wrong → {relation_label(opposite).lower()} (sid={w_sid})")
    axes[0].axhline(0.0, linestyle="--", linewidth=1.1, color="0.4")
    axes[0].set_title("Intermediate spatial evidence", fontsize=13)
    axes[0].set_ylabel(f"{relation_label(target)} − {relation_label(opposite)} spatial gap", fontsize=11.5)
    axes[0].set_xlabel("Decoder layer", fontsize=11.5)
    axes[0].grid(True, alpha=0.18)
    axes[0].tick_params(axis="both", labelsize=10)
    axes[0].legend(frameon=False, fontsize=9.2, loc="best")
    axes[0].text(
        0.03, 0.04,
        f"mid-layer RMSE = {float(best['spatial_multilayer_rmse']):.3f}",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.78", alpha=0.9),
    )

    # Decision panel: exactly two trajectories.
    axes[1].axvspan(
        min(decision_match_layers) - 0.45,
        max(decision_match_layers) + 0.45,
        alpha=0.055,
        color="0.45",
        zorder=0,
    )
    axes[1].axvspan(
        min(late_layers) - 0.45,
        max(late_layers) + 0.45,
        alpha=0.10,
        color="0.70",
        zorder=0,
    )
    axes[1].plot(x, c_dec, marker="o", linewidth=2.3, markersize=4.2, label=f"Final correct (sid={c_sid})")
    axes[1].plot(x, w_dec, marker="o", linewidth=2.3, markersize=4.2, linestyle="--",
                 label=f"Final wrong → {relation_label(opposite).lower()} (sid={w_sid})")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.1, color="0.4")
    axes[1].set_title("Downstream decision evidence", fontsize=13)
    axes[1].set_ylabel(f"{relation_label(target)} − {relation_label(opposite)} decision gap", fontsize=11.5)
    axes[1].set_xlabel("Decoder layer", fontsize=11.5)
    axes[1].grid(True, alpha=0.18)
    axes[1].tick_params(axis="both", labelsize=10)
    axes[1].text(
        0.03, 0.04,
        f"pre-decision RMSE = {float(best['predecision_multilayer_rmse']):.3f}\n"
        f"late separation = {float(best['late_decision_separation']):.3f}",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.78", alpha=0.9),
    )

    fig.tight_layout(pad=0.7, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    target = canonical_relation(a.target_relation)
    opposite = canonical_relation(a.wrong_relation)
    if target == opposite:
        raise ValueError("--target-relation and --wrong-relation must differ")

    spatial_layers = sorted(set(a.spatial_match_layers))
    decision_match_layers = sorted(set(a.decision_match_layers))
    late_layers = sorted(set(a.late_layers))
    needed = sorted(set(spatial_layers + decision_match_layers + late_layers))

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(a.per_sample_csv))
    by_sid = group_by_sid(rows)
    all_layers = sorted({as_int(r, "layer") for r in rows})

    correct: List[Tuple[int, Dict[int, Dict[str, str]], Dict[str, Any]]] = []
    wrong: List[Tuple[int, Dict[int, Dict[str, str]], Dict[str, Any]]] = []

    for sid, sample in by_sid.items():
        if not ensure_layers(sample, needed):
            continue
        ref = sample[spatial_layers[len(spatial_layers) // 2]]
        if canonical_relation(ref["gt"]) != target:
            continue

        info = candidate_info(
            sid, sample, target, opposite,
            spatial_layers, decision_match_layers, late_layers,
        )

        if info["spatial_positive_frac"] < a.min_spatial_positive_frac:
            continue
        if info["spatial_mean_gap"] < a.min_spatial_mean_gap:
            continue

        if info["final_correct"] == 1 and info["final_prediction_relation"] == target:
            if a.late_sign_threshold >= 0 and info["late_decision_mean_gap"] <= a.late_sign_threshold:
                continue
            correct.append((sid, sample, info))
        elif info["final_correct"] == 0 and info["final_prediction_relation"] == opposite:
            if a.late_sign_threshold >= 0 and info["late_decision_mean_gap"] >= -a.late_sign_threshold:
                continue
            wrong.append((sid, sample, info))

    print(f"[CANDIDATES] correct {target}: {len(correct)}")
    print(f"[CANDIDATES] wrong {target}->{opposite}: {len(wrong)}")
    if not correct or not wrong:
        raise RuntimeError(
            "No valid candidates. Try lowering --min-spatial-mean-gap, "
            "--min-spatial-positive-frac, or --late-sign-threshold."
        )

    sample_lookup: Dict[int, Dict[int, Dict[str, str]]] = {}
    for sid, sample, _ in correct + wrong:
        sample_lookup[sid] = sample

    pairs: List[Dict[str, Any]] = []
    for c_sid, c_sample, c_info in correct:
        for w_sid, w_sample, w_info in wrong:
            m = pair_metrics(
                c_sample, w_sample, target, opposite,
                spatial_layers, decision_match_layers, late_layers,
            )
            if a.max_spatial_rmse >= 0 and m["spatial_multilayer_rmse"] > a.max_spatial_rmse:
                continue
            if a.max_predecision_rmse >= 0 and m["predecision_multilayer_rmse"] > a.max_predecision_rmse:
                continue

            row: Dict[str, Any] = {
                "correct_sid": c_sid,
                "wrong_sid": w_sid,
                "target_relation": target,
                "wrong_relation": opposite,
                "correct_mapping": c_info["mapping"],
                "wrong_mapping": w_info["mapping"],
                "correct_spatial_mean": c_info["spatial_mean_gap"],
                "wrong_spatial_mean": w_info["spatial_mean_gap"],
                "correct_spatial_positive_frac": c_info["spatial_positive_frac"],
                "wrong_spatial_positive_frac": w_info["spatial_positive_frac"],
                **m,
            }
            row["rank_score"] = rank_score(row, a)
            pairs.append(row)

    if not pairs:
        raise RuntimeError("No pair survived the requested RMSE calipers.")

    pairs.sort(key=lambda r: (float(r["rank_score"]), -float(r["late_decision_separation"])))
    top = pairs[: max(1, a.topk)]
    write_csv(out / "top_pairs.csv", top)
    (out / "top_pairs.json").write_text(json.dumps(top, indent=2, ensure_ascii=False), encoding="utf-8")

    best = top[0]
    c_sid = int(best["correct_sid"])
    w_sid = int(best["wrong_sid"])
    c_sample = sample_lookup[c_sid]
    w_sample = sample_lookup[w_sid]

    # Save per-layer traces for inspection/replotting.
    detail_rows: List[Dict[str, Any]] = []
    for group, sid, sample in [
        ("final_correct", c_sid, c_sample),
        ("final_wrong", w_sid, w_sample),
    ]:
        for L in all_layers:
            if L not in sample:
                continue
            r = sample[L]
            detail_rows.append({
                "group": group,
                "sid": sid,
                "layer": L,
                "gt": r["gt"],
                "mapping": r["mapping"],
                "final_prediction": r["final_prediction"],
                "final_prediction_relation": prediction_relation(r),
                "spatial_target_score": as_float(r, f"spatial_score_{target}"),
                "spatial_opposite_score": as_float(r, f"spatial_score_{opposite}"),
                "spatial_target_minus_opposite": spatial_gap(r, target, opposite),
                "decision_target_score": decision_relation_score(r, target),
                "decision_opposite_score": decision_relation_score(r, opposite),
                "decision_target_minus_opposite": decision_gap(r, target, opposite),
            })
    write_csv(out / "best_pair_layer.csv", detail_rows)

    save_two_curve_figure(
        out / "intro_pair_multilayer_two_curves.png",
        c_sample, w_sample, all_layers,
        spatial_layers, decision_match_layers, late_layers,
        target, opposite, c_sid, w_sid, best,
    )

    metadata = {
        "target_relation": target,
        "wrong_relation": opposite,
        "spatial_match_layers": spatial_layers,
        "decision_match_layers": decision_match_layers,
        "late_layers": late_layers,
        "candidate_constraints": {
            "min_spatial_positive_frac": a.min_spatial_positive_frac,
            "min_spatial_mean_gap": a.min_spatial_mean_gap,
            "late_sign_threshold": a.late_sign_threshold,
        },
        "ranking": (
            "spatial_multilayer_rmse + decision_match_weight * predecision_multilayer_rmse + "
            "spatial_mean_weight * spatial_mean_gap_difference - "
            "late_tiebreak_weight * late_decision_separation"
        ),
        "best_pair": best,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("BEST MULTI-LAYER MATCHED INTRO PAIR")
    print("=" * 100)
    print(f"correct sid={c_sid} | wrong sid={w_sid} | GT={target} | wrong->{opposite}")
    print(f"spatial multi-layer RMSE     = {float(best['spatial_multilayer_rmse']):.6f}")
    print(f"pre-decision multi-layer RMSE= {float(best['predecision_multilayer_rmse']):.6f}")
    print(f"correct late mean gap        = {float(best['correct_late_mean']):+.6f}")
    print(f"wrong late mean gap          = {float(best['wrong_late_mean']):+.6f}")
    print(f"late decision separation     = {float(best['late_decision_separation']):+.6f}")
    print(f"[SAVED] {out / 'intro_pair_multilayer_two_curves.png'}")
    print(f"[SAVED] {out / 'top_pairs.csv'}")
    print(f"[SAVED] {out / 'best_pair_layer.csv'}")


if __name__ == "__main__":
    main()
