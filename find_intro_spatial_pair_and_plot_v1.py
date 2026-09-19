#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find an Introduction-friendly pair of COCO samples from the output of
`eval_qwen3b_coco_spatial_margin_matched_v1.py`.

Goal
====
Find two samples with the SAME ground-truth spatial relation (default: left)
whose intermediate spatial states are as similar as possible, but whose final
answers diverge:

  Sample A: GT=left, final=left  (correct)
  Sample B: GT=left, final=right (wrong -> opposite)

Both samples must already support the target relation across a middle-layer
window.  Pair ranking is driven primarily by the similarity of their spatial
margin trajectories, NOT by late decision behavior.  Late decision separation
is used only as a tie-break after the required final outcomes are enforced.

The script then draws a 2x2 paper-style figure:

  top row    : spatial relation scores (Left vs Right)
  bottom row : downstream decision readout scores for the options currently
               mapped to Left and Right

Thus the desired pattern is visually explicit:

    both samples: middle layers -> Left spatial state
    correct one : late decision -> Left
    wrong one   : late decision -> Right

Input
=====
Use the per-sample CSV already produced by:

  output/.../per_sample_layer.csv

Required columns are the ones written by
`eval_qwen3b_coco_spatial_margin_matched_v1.py`, including:
  sid, layer, gt, gt_option, mapping, final_prediction, final_correct,
  spatial_margin, decision_margin, spatial_pred,
  spatial_score_left/right/above/below,
  decision_score_A/B/C/D

No model forward pass is required.

Example
=======
python find_intro_spatial_pair_and_plot_v1.py \
  --per-sample-csv output/qwen3b_coco_spatial_margin_matched_v1/per_sample_layer.csv \
  --target-relation left \
  --wrong-relation right \
  --mid-layers 21 22 23 24 25 \
  --match-layer 22 \
  --late-layers 27 28 29 30 31 32 \
  --min-mid-correct-layers 4 \
  --topk 30 \
  --output-dir output/intro_left_pair_v1

Optional image version (run from AdaptVis repo root):
  add --include-images --data-root data --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
DISPLAY_REL = {"left": "Left", "right": "Right", "above": "On", "below": "Under"}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--per-sample-csv", required=True)
    p.add_argument("--target-relation", default="left", choices=REL)
    p.add_argument("--wrong-relation", default="right", choices=REL)
    p.add_argument("--mid-layers", nargs="+", type=int, default=[21, 22, 23, 24, 25])
    p.add_argument("--match-layer", type=int, default=22)
    p.add_argument("--late-layers", nargs="+", type=int, default=[27, 28, 29, 30, 31, 32])
    p.add_argument("--min-mid-correct-layers", type=int, default=4,
                   help="Minimum number of mid layers whose spatial_pred equals the GT target relation.")
    p.add_argument("--min-mid-positive-layers", type=int, default=4,
                   help="Minimum number of mid layers with positive GT-vs-best-other spatial margin.")
    p.add_argument("--min-mid-mean-margin", type=float, default=0.0,
                   help="Require mean spatial margin over the middle window to exceed this value.")
    p.add_argument("--max-mid-rmse", type=float, default=-1.0,
                   help="Optional hard caliper on pair mid-window spatial-margin RMSE. <0 disables.")
    p.add_argument("--max-match-gap", type=float, default=-1.0,
                   help="Optional hard caliper on |spatial margin difference| at --match-layer. <0 disables.")
    p.add_argument("--require-late-sign", action="store_true",
                   help="Require mean late Left-vs-Right decision gap >0 for correct and <0 for wrong.")
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--output-dir", required=True)

    # Optional image/question rendering.
    p.add_argument("--include-images", action="store_true")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
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
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_mapping(s: str) -> Dict[str, str]:
    """Parse `left->A,right->B,above->C,below->D`."""
    out: Dict[str, str] = {}
    for item in str(s).split(","):
        item = item.strip()
        if not item or "->" not in item:
            continue
        rel, letter = item.split("->", 1)
        out[canonical_relation(rel)] = letter.strip()
    if set(out) != set(REL):
        raise ValueError(f"Could not parse full relation mapping from {s!r}; parsed={out}")
    return out


def prediction_relation(row: Mapping[str, str]) -> str:
    mp = parse_mapping(row["mapping"])
    inv = {v: k for k, v in mp.items()}
    pred_letter = str(row["final_prediction"]).strip()
    if pred_letter not in inv:
        raise ValueError(f"Prediction {pred_letter!r} missing from mapping {mp}")
    return inv[pred_letter]


def as_float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: Mapping[str, str], key: str) -> int:
    return int(float(row[key]))


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


def spatial_margin_trace(sample: Mapping[int, Mapping[str, str]], layers: Sequence[int]) -> np.ndarray:
    return np.asarray([as_float(sample[int(L)], "spatial_margin") for L in layers], np.float64)


def spatial_target_opposite_gap(
    row: Mapping[str, str], target: str, opposite: str
) -> float:
    return as_float(row, f"spatial_score_{target}") - as_float(row, f"spatial_score_{opposite}")


def decision_relation_score(row: Mapping[str, str], relation: str) -> float:
    mp = parse_mapping(row["mapping"])
    letter = mp[relation]
    return as_float(row, f"decision_score_{letter}")


def decision_target_opposite_gap(row: Mapping[str, str], target: str, opposite: str) -> float:
    return decision_relation_score(row, target) - decision_relation_score(row, opposite)


def sample_descriptor(
    sid: int,
    sample: Mapping[int, Mapping[str, str]],
    mid_layers: Sequence[int],
    late_layers: Sequence[int],
    match_layer: int,
    target: str,
    wrong_relation: str,
) -> Dict[str, Any]:
    ref = sample[int(match_layer)]
    mid_margin = spatial_margin_trace(sample, mid_layers)
    mid_correct = sum(str(sample[int(L)]["spatial_pred"]) == target for L in mid_layers)
    mid_positive = int(np.sum(mid_margin > 0))

    late_dec_gap = np.asarray(
        [decision_target_opposite_gap(sample[int(L)], target, wrong_relation) for L in late_layers],
        np.float64,
    )
    pred_rel = prediction_relation(ref)

    return {
        "sid": sid,
        "gt": str(ref["gt"]),
        "gt_option": str(ref["gt_option"]),
        "mapping": str(ref["mapping"]),
        "final_prediction_letter": str(ref["final_prediction"]),
        "final_prediction_relation": pred_rel,
        "final_correct": as_int(ref, "final_correct"),
        "match_margin": as_float(ref, "spatial_margin"),
        "mid_margin_mean": float(mid_margin.mean()),
        "mid_margin_min": float(mid_margin.min()),
        "mid_margin_max": float(mid_margin.max()),
        "mid_correct_layers": int(mid_correct),
        "mid_positive_layers": int(mid_positive),
        "late_target_vs_wrong_mean": float(late_dec_gap.mean()),
        "late_target_vs_wrong_min": float(late_dec_gap.min()),
        "late_target_vs_wrong_max": float(late_dec_gap.max()),
    }


def pair_metrics(
    corr_sample: Mapping[int, Mapping[str, str]],
    wrong_sample: Mapping[int, Mapping[str, str]],
    mid_layers: Sequence[int],
    late_layers: Sequence[int],
    match_layer: int,
    target: str,
    wrong_relation: str,
) -> Dict[str, float]:
    c_mid = spatial_margin_trace(corr_sample, mid_layers)
    w_mid = spatial_margin_trace(wrong_sample, mid_layers)
    rmse = float(np.sqrt(np.mean((c_mid - w_mid) ** 2)))
    mean_gap = float(abs(c_mid.mean() - w_mid.mean()))
    match_gap = float(abs(
        as_float(corr_sample[int(match_layer)], "spatial_margin")
        - as_float(wrong_sample[int(match_layer)], "spatial_margin")
    ))

    c_late = np.asarray(
        [decision_target_opposite_gap(corr_sample[int(L)], target, wrong_relation) for L in late_layers],
        np.float64,
    )
    w_late = np.asarray(
        [decision_target_opposite_gap(wrong_sample[int(L)], target, wrong_relation) for L in late_layers],
        np.float64,
    )
    late_sep = float(c_late.mean() - w_late.mean())
    return {
        "mid_spatial_rmse": rmse,
        "mid_spatial_mean_gap": mean_gap,
        "match_layer_gap": match_gap,
        "correct_late_target_vs_wrong_mean": float(c_late.mean()),
        "wrong_late_target_vs_wrong_mean": float(w_late.mean()),
        "late_decision_separation": late_sep,
    }


def rank_key(p: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    # Primary goal: spatially matched pair.  Late decision divergence is only a tie-break.
    return (
        float(p["mid_spatial_rmse"]),
        float(p["match_layer_gap"]),
        float(p["mid_spatial_mean_gap"]),
        -float(p["late_decision_separation"]),
    )


def relation_label(r: str) -> str:
    return DISPLAY_REL.get(r, r.title())


def _plot_spatial_panel(
    ax: Any,
    sample: Mapping[int, Mapping[str, str]],
    layers: Sequence[int],
    target: str,
    opposite: str,
    mid_layers: Sequence[int],
    title: str,
) -> None:
    x = np.asarray(layers, np.int64)
    y_t = np.asarray([as_float(sample[L], f"spatial_score_{target}") for L in layers], np.float64)
    y_o = np.asarray([as_float(sample[L], f"spatial_score_{opposite}") for L in layers], np.float64)
    ax.axvspan(min(mid_layers) - 0.45, max(mid_layers) + 0.45, alpha=0.08, color="0.45")
    ax.plot(x, y_t, marker="o", linewidth=2.2, markersize=4.5, label=relation_label(target))
    ax.plot(x, y_o, marker="o", linewidth=2.2, markersize=4.5, label=relation_label(opposite))
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Spatial readout score", fontsize=11.5)
    ax.grid(True, alpha=0.18)
    ax.tick_params(axis="both", labelsize=10)


def _plot_decision_panel(
    ax: Any,
    sample: Mapping[int, Mapping[str, str]],
    layers: Sequence[int],
    target: str,
    opposite: str,
    late_layers: Sequence[int],
) -> None:
    x = np.asarray(layers, np.int64)
    y_t = np.asarray([decision_relation_score(sample[L], target) for L in layers], np.float64)
    y_o = np.asarray([decision_relation_score(sample[L], opposite) for L in layers], np.float64)
    ax.axvspan(min(late_layers) - 0.45, max(late_layers) + 0.45, alpha=0.08, color="0.45")
    ax.plot(x, y_t, marker="o", linewidth=2.2, markersize=4.5, label=relation_label(target))
    ax.plot(x, y_o, marker="o", linewidth=2.2, markersize=4.5, label=relation_label(opposite))
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlabel("Decoder layer", fontsize=11.5)
    ax.set_ylabel("Decision readout score", fontsize=11.5)
    ax.grid(True, alpha=0.18)
    ax.tick_params(axis="both", labelsize=10)


def save_pair_2x2(
    path: Path,
    corr_sample: Mapping[int, Mapping[str, str]],
    wrong_sample: Mapping[int, Mapping[str, str]],
    all_layers: Sequence[int],
    mid_layers: Sequence[int],
    late_layers: Sequence[int],
    target: str,
    wrong_relation: str,
    corr_sid: int,
    wrong_sid: int,
    match_layer: int,
) -> None:
    c_match = as_float(corr_sample[match_layer], "spatial_margin")
    w_match = as_float(wrong_sample[match_layer], "spatial_margin")
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.2), dpi=260, sharex="col")

    _plot_spatial_panel(
        axes[0, 0], corr_sample, all_layers, target, wrong_relation, mid_layers,
        f"Final correct  (sid={corr_sid})",
    )
    _plot_spatial_panel(
        axes[0, 1], wrong_sample, all_layers, target, wrong_relation, mid_layers,
        f"Final wrong → {relation_label(wrong_relation).lower()}  (sid={wrong_sid})",
    )
    _plot_decision_panel(axes[1, 0], corr_sample, all_layers, target, wrong_relation, late_layers)
    _plot_decision_panel(axes[1, 1], wrong_sample, all_layers, target, wrong_relation, late_layers)

    # Compact annotations: show that mid spatial strength is matched.
    axes[0, 0].text(
        0.03, 0.05,
        f"L{match_layer} spatial margin = {c_match:.3f}",
        transform=axes[0, 0].transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.9),
    )
    axes[0, 1].text(
        0.03, 0.05,
        f"L{match_layer} spatial margin = {w_match:.3f}",
        transform=axes[0, 1].transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.9),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96], pad=0.8, w_pad=1.0, h_pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pair_compact(
    path: Path,
    corr_sample: Mapping[int, Mapping[str, str]],
    wrong_sample: Mapping[int, Mapping[str, str]],
    all_layers: Sequence[int],
    mid_layers: Sequence[int],
    late_layers: Sequence[int],
    target: str,
    wrong_relation: str,
    corr_sid: int,
    wrong_sid: int,
) -> None:
    """A smaller Introduction-oriented view using pairwise margins."""
    x = np.asarray(all_layers, np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), dpi=260)

    for sample, label, ls in [
        (corr_sample, "Final correct", "-"),
        (wrong_sample, f"Final wrong → {relation_label(wrong_relation).lower()}", "--"),
    ]:
        y_sp = np.asarray(
            [spatial_target_opposite_gap(sample[L], target, wrong_relation) for L in all_layers],
            np.float64,
        )
        axes[0].plot(x, y_sp, marker="o", linewidth=2.2, markersize=4.3, linestyle=ls, label=label)

        y_dec = np.asarray(
            [decision_target_opposite_gap(sample[L], target, wrong_relation) for L in all_layers],
            np.float64,
        )
        axes[1].plot(x, y_dec, marker="o", linewidth=2.2, markersize=4.3, linestyle=ls, label=label)

    axes[0].axvspan(min(mid_layers) - 0.45, max(mid_layers) + 0.45, alpha=0.08, color="0.45")
    axes[1].axvspan(min(late_layers) - 0.45, max(late_layers) + 0.45, alpha=0.08, color="0.45")
    for ax in axes:
        ax.axhline(0.0, linestyle="--", linewidth=1.1, color="0.4")
        ax.set_xlabel("Decoder layer", fontsize=11.5)
        ax.grid(True, alpha=0.18)
        ax.tick_params(axis="both", labelsize=10)
    axes[0].set_title(f"Spatial evidence: {relation_label(target)} vs {relation_label(wrong_relation)}", fontsize=12.5)
    axes[1].set_title(f"Decision evidence: {relation_label(target)} vs {relation_label(wrong_relation)}", fontsize=12.5)
    axes[0].set_ylabel("Pairwise spatial score gap", fontsize=11.5)
    axes[1].set_ylabel("Pairwise decision score gap", fontsize=11.5)
    axes[0].legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout(pad=0.7, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_optional_images_and_questions(
    sids: Sequence[int], data_root: str, prompt_jsonl: str
) -> Dict[int, Dict[str, Any]]:
    """Best-effort: uses existing AdaptVis utilities; no model is loaded."""
    try:
        import analyze_coco_centroid_generation_step1_v4 as base  # existing repo utility
    except Exception as e:
        raise RuntimeError(
            "--include-images requires running this script from the AdaptVis repo root so "
            "analyze_coco_centroid_generation_step1_v4.py can be imported."
        ) from e

    prompts = base.load_standard_prompts(Path(prompt_jsonl))
    two = base.import_two_object_module()
    records, _audit = two.load_records("coco_two", Path(data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}
    out: Dict[int, Dict[str, Any]] = {}
    for sid in sids:
        if sid not in rec_by_sid:
            continue
        img = base.record_image(rec_by_sid[sid])
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        p = prompts.get(sid, {})
        subject = str(p.get("subject", "subject"))
        reference = str(p.get("reference", "reference"))
        question = f"Where is the {subject} relative to the {reference}?"
        out[sid] = {"image": img.copy(), "question": question}
        try:
            img.close()
        except Exception:
            pass
    return out


def save_pair_with_images(
    path: Path,
    corr_sample: Mapping[int, Mapping[str, str]],
    wrong_sample: Mapping[int, Mapping[str, str]],
    all_layers: Sequence[int],
    mid_layers: Sequence[int],
    late_layers: Sequence[int],
    target: str,
    wrong_relation: str,
    corr_sid: int,
    wrong_sid: int,
    image_info: Mapping[int, Mapping[str, Any]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.1, 8.2), dpi=250, gridspec_kw={"height_ratios": [1.15, 1.0, 1.0]})
    for col, (sid, outcome) in enumerate([
        (corr_sid, "Final correct"),
        (wrong_sid, f"Final wrong → {relation_label(wrong_relation).lower()}"),
    ]):
        ax = axes[0, col]
        info = image_info.get(sid)
        if info is not None:
            ax.imshow(info["image"])
            ax.set_title(f"{outcome}\n{info['question']}", fontsize=11.5)
        else:
            ax.text(0.5, 0.5, f"sid={sid}\n{outcome}", ha="center", va="center", fontsize=12)
        ax.axis("off")

    _plot_spatial_panel(axes[1, 0], corr_sample, all_layers, target, wrong_relation, mid_layers, "Intermediate spatial state")
    _plot_spatial_panel(axes[1, 1], wrong_sample, all_layers, target, wrong_relation, mid_layers, "Intermediate spatial state")
    _plot_decision_panel(axes[2, 0], corr_sample, all_layers, target, wrong_relation, late_layers)
    _plot_decision_panel(axes[2, 1], wrong_sample, all_layers, target, wrong_relation, late_layers)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97], pad=0.7, w_pad=0.9, h_pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    target = canonical_relation(a.target_relation)
    wrong_relation = canonical_relation(a.wrong_relation)
    if target == wrong_relation:
        raise ValueError("--target-relation and --wrong-relation must differ")

    mid_layers = sorted(set(int(x) for x in a.mid_layers))
    late_layers = sorted(set(int(x) for x in a.late_layers))
    if a.match_layer not in mid_layers:
        raise ValueError("--match-layer should be included in --mid-layers")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(a.per_sample_csv))
    by_sid = group_by_sid(rows)
    all_layers = sorted({as_int(r, "layer") for r in rows})
    needed_layers = sorted(set(all_layers) | set(mid_layers) | set(late_layers) | {a.match_layer})

    # Candidate samples.
    correct: List[Tuple[int, Dict[int, Dict[str, str]], Dict[str, Any]]] = []
    wrong: List[Tuple[int, Dict[int, Dict[str, str]], Dict[str, Any]]] = []

    for sid, sample in by_sid.items():
        if not ensure_layers(sample, sorted(set(mid_layers + late_layers + [a.match_layer]))):
            continue
        ref = sample[a.match_layer]
        gt = canonical_relation(ref["gt"])
        if gt != target:
            continue

        desc = sample_descriptor(
            sid, sample, mid_layers, late_layers, a.match_layer, target, wrong_relation
        )
        if desc["mid_correct_layers"] < a.min_mid_correct_layers:
            continue
        if desc["mid_positive_layers"] < a.min_mid_positive_layers:
            continue
        if desc["mid_margin_mean"] < a.min_mid_mean_margin:
            continue

        if desc["final_correct"] == 1 and desc["final_prediction_relation"] == target:
            if (not a.require_late_sign) or desc["late_target_vs_wrong_mean"] > 0:
                correct.append((sid, sample, desc))
        elif desc["final_correct"] == 0 and desc["final_prediction_relation"] == wrong_relation:
            if (not a.require_late_sign) or desc["late_target_vs_wrong_mean"] < 0:
                wrong.append((sid, sample, desc))

    print(f"[CANDIDATES] correct {target}: {len(correct)}")
    print(f"[CANDIDATES] wrong {target}->{wrong_relation}: {len(wrong)}")
    if not correct or not wrong:
        raise RuntimeError(
            "No candidate pair. Try lowering --min-mid-correct-layers / --min-mid-positive-layers, "
            "or remove --require-late-sign."
        )

    pairs: List[Dict[str, Any]] = []
    sample_lookup = {sid: sample for sid, sample, _ in correct + wrong}
    desc_lookup = {sid: desc for sid, _, desc in correct + wrong}

    for c_sid, c_sample, c_desc in correct:
        for w_sid, w_sample, w_desc in wrong:
            met = pair_metrics(
                c_sample, w_sample, mid_layers, late_layers, a.match_layer, target, wrong_relation
            )
            if a.max_mid_rmse >= 0 and met["mid_spatial_rmse"] > a.max_mid_rmse:
                continue
            if a.max_match_gap >= 0 and met["match_layer_gap"] > a.max_match_gap:
                continue
            row: Dict[str, Any] = {
                "correct_sid": c_sid,
                "wrong_sid": w_sid,
                "target_relation": target,
                "wrong_relation": wrong_relation,
                "correct_gt_option": c_desc["gt_option"],
                "wrong_gt_option": w_desc["gt_option"],
                "correct_mapping": c_desc["mapping"],
                "wrong_mapping": w_desc["mapping"],
                "correct_match_margin": c_desc["match_margin"],
                "wrong_match_margin": w_desc["match_margin"],
                "correct_mid_margin_mean": c_desc["mid_margin_mean"],
                "wrong_mid_margin_mean": w_desc["mid_margin_mean"],
                "correct_mid_correct_layers": c_desc["mid_correct_layers"],
                "wrong_mid_correct_layers": w_desc["mid_correct_layers"],
                "correct_mid_positive_layers": c_desc["mid_positive_layers"],
                "wrong_mid_positive_layers": w_desc["mid_positive_layers"],
                **met,
            }
            pairs.append(row)

    if not pairs:
        raise RuntimeError("No pair survived the requested calipers.")
    pairs.sort(key=rank_key)
    top = pairs[: max(1, a.topk)]
    write_csv(out / "top_pairs.csv", top)

    with (out / "top_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)

    best = top[0]
    c_sid = int(best["correct_sid"])
    w_sid = int(best["wrong_sid"])
    c_sample = sample_lookup[c_sid]
    w_sample = sample_lookup[w_sid]

    # Save all per-layer values for the chosen pair so the figure can be reproduced/inspected.
    chosen_rows: List[Dict[str, Any]] = []
    for group, sid, sample in [("final_correct", c_sid, c_sample), ("final_wrong", w_sid, w_sample)]:
        for L in all_layers:
            if L not in sample:
                continue
            r = sample[L]
            chosen_rows.append({
                "group": group,
                "sid": sid,
                "layer": L,
                "gt": r["gt"],
                "mapping": r["mapping"],
                "final_prediction": r["final_prediction"],
                "final_prediction_relation": prediction_relation(r),
                "spatial_margin": float(r["spatial_margin"]),
                "decision_margin": float(r["decision_margin"]),
                "spatial_pred": r["spatial_pred"],
                "spatial_target_score": float(r[f"spatial_score_{target}"]),
                "spatial_wrong_score": float(r[f"spatial_score_{wrong_relation}"]),
                "spatial_target_vs_wrong_gap": spatial_target_opposite_gap(r, target, wrong_relation),
                "decision_target_score": decision_relation_score(r, target),
                "decision_wrong_score": decision_relation_score(r, wrong_relation),
                "decision_target_vs_wrong_gap": decision_target_opposite_gap(r, target, wrong_relation),
            })
    write_csv(out / "best_pair_layer.csv", chosen_rows)

    save_pair_2x2(
        out / "intro_pair_2x2.png",
        c_sample, w_sample, all_layers, mid_layers, late_layers,
        target, wrong_relation, c_sid, w_sid, a.match_layer,
    )
    save_pair_compact(
        out / "intro_pair_compact.png",
        c_sample, w_sample, all_layers, mid_layers, late_layers,
        target, wrong_relation, c_sid, w_sid,
    )

    if a.include_images:
        info = load_optional_images_and_questions(
            [c_sid, w_sid], a.data_root, a.prompt_jsonl
        )
        save_pair_with_images(
            out / "intro_pair_with_images.png",
            c_sample, w_sample, all_layers, mid_layers, late_layers,
            target, wrong_relation, c_sid, w_sid, info,
        )

    print("\n" + "=" * 96)
    print("BEST INTRO PAIR")
    print("=" * 96)
    print(f"correct sid={c_sid} | wrong sid={w_sid} | GT={target} | wrong->{wrong_relation}")
    print(f"L{a.match_layer} spatial margins: correct={best['correct_match_margin']:.6f} "
          f"wrong={best['wrong_match_margin']:.6f} gap={best['match_layer_gap']:.6f}")
    print(f"mid spatial trajectory RMSE ({mid_layers}) = {best['mid_spatial_rmse']:.6f}")
    print(f"mid mean margins: correct={best['correct_mid_margin_mean']:.6f} "
          f"wrong={best['wrong_mid_margin_mean']:.6f}")
    print(f"late {target}-vs-{wrong_relation} decision mean: "
          f"correct={best['correct_late_target_vs_wrong_mean']:.6f} "
          f"wrong={best['wrong_late_target_vs_wrong_mean']:.6f}")
    print(f"late decision separation = {best['late_decision_separation']:.6f}")
    print(f"[SAVED] {out / 'top_pairs.csv'}")
    print(f"[SAVED] {out / 'best_pair_layer.csv'}")
    print(f"[SAVED] {out / 'intro_pair_2x2.png'}")
    print(f"[SAVED] {out / 'intro_pair_compact.png'}")
    if a.include_images:
        print(f"[SAVED] {out / 'intro_pair_with_images.png'}")


if __name__ == "__main__":
    main()
