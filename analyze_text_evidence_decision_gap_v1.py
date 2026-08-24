#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare correct vs wrong samples along the text-evidence -> last-decision pathway.

This script is a POST-PROCESSOR for:
    analyze_text_stream_visual_causal_transfer_v1.py

It does NOT run the VLM and does NOT require a GPU.

Core quantities
---------------
For each sample i and layer l:

E_l = GT margin from the "sufficiency / all_text" branch
      GRAY base + REAL all-text states at layer l.
      Interpretation: how sufficient the image-derived information already
      present in the text stream is for the model's own downstream computation.

D_l = GT margin from the "sufficiency / last" branch
      GRAY base + REAL last-token state at layer l.
      Interpretation: how sufficient the current prediction-token state is
      for the model's own downstream computation.

The script compares these trajectories between:
    - normal REAL generation correct samples
    - normal REAL generation wrong samples

It also derives per-sample stage summaries:
    * stable evidence onset
    * stable decision onset
    * onset gap
    * whether correct evidence ever/stably existed before a final error
    * whether a correct last-token decision appeared and was later lost

Important
---------
This is NOT a spatial probe.  Every E_l / D_l prediction comes from the
original model's remaining layers + original LM head after causal patching.

Recommended workflow
--------------------
1) First run a DENSE causal scan, e.g. L18-L31:

CUDA_VISIBLE_DEVICES=0 python analyze_text_stream_visual_causal_transfer_v1.py \
  --dataset coco_two \
  --data-root data \
  --model qwen-3b \
  --device cuda:0 \
  --layers 18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --roles last,all_text \
  --output-dir output/qwen3b_text_stream_dense_l18_l31 \
  --overwrite

2) Then run this analyzer:

python analyze_text_evidence_decision_gap_v1.py \
  --input-dir output/qwen3b_text_stream_dense_l18_l31 \
  --min-layer 18 \
  --max-layer 31 \
  --stable-k 3 \
  --output-dir output/qwen3b_text_evidence_decision_gap_v1 \
  --overwrite

Outputs
-------
per_sample_stage_metrics.csv
    One row/sample with E/D onset, gaps, late-flip flags and failure category.

per_sample_layer_trajectories.csv
    One row/sample/layer containing E_l, D_l and E_l-D_l.

per_layer_correct_vs_wrong.csv
    Correct-vs-wrong group comparison for every layer.

failure_taxonomy.csv
    Counts/proportions among normal REAL generation errors.

transition_counts.csv
    Per-layer counts of branch correctness transitions.

summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "text-evidence-decision-gap-v1"
EPS = 1e-12


# -----------------------------------------------------------------------------
# IO / parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Output directory produced by "
            "analyze_text_stream_visual_causal_transfer_v1.py"
        ),
    )
    p.add_argument("--min-layer", type=int, default=None)
    p.add_argument("--max-layer", type=int, default=None)
    p.add_argument(
        "--stable-k",
        type=int,
        default=3,
        help=(
            "Number of consecutive OBSERVED layers that must be correct to call "
            "evidence/decision stable. Dense consecutive layers are strongly recommended."
        ),
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for correct-vs-wrong mean-difference 95%% CIs. 0 disables.",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
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
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        raise ValueError(f"Cannot parse boolean value: {x!r}")


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def finite(xs: Iterable[float]) -> np.ndarray:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return np.asarray(vals, dtype=np.float64)


def mean(xs: Iterable[float]) -> float:
    v = finite(xs)
    return float(v.mean()) if len(v) else float("nan")


def median(xs: Iterable[float]) -> float:
    v = finite(xs)
    return float(np.median(v)) if len(v) else float("nan")


def frac(xs: Iterable[bool]) -> float:
    v = list(bool(x) for x in xs)
    return float(np.mean(v)) if v else float("nan")


def std(xs: Iterable[float]) -> float:
    v = finite(xs)
    return float(v.std()) if len(v) else float("nan")


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Return (mean(b)-mean(a), lo, hi), where a=correct group and b=wrong group.
    """
    aa = finite(a)
    bb = finite(b)
    if len(aa) == 0 or len(bb) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(bb.mean() - aa.mean())
    if n_boot <= 0:
        return obs, float("nan"), float("nan")
    sims = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        sa = aa[rng.integers(0, len(aa), len(aa))]
        sb = bb[rng.integers(0, len(bb), len(bb))]
        sims[k] = sb.mean() - sa.mean()
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return obs, float(lo), float(hi)


def stable_onset(
    layers: Sequence[int],
    correct_by_layer: Mapping[int, bool],
    stable_k: int,
) -> Optional[int]:
    """
    Earliest observed layer starting a run of stable_k correct observations.
    This intentionally operates on the observed layer grid.  For mechanistic
    onset interpretation, use a dense consecutive scan.
    """
    if stable_k <= 0:
        raise ValueError("--stable-k must be >= 1")
    ordered = [int(l) for l in layers]
    for i in range(len(ordered)):
        window = ordered[i:i + stable_k]
        if len(window) < stable_k:
            break
        if all(bool(correct_by_layer.get(l, False)) for l in window):
            return int(window[0])
    return None


def first_correct_layer(
    layers: Sequence[int],
    correct_by_layer: Mapping[int, bool],
) -> Optional[int]:
    for l in layers:
        if bool(correct_by_layer.get(int(l), False)):
            return int(l)
    return None


def first_wrong_after_correct(
    layers: Sequence[int],
    correct_by_layer: Mapping[int, bool],
) -> Optional[int]:
    seen_correct = False
    for l in layers:
        c = bool(correct_by_layer.get(int(l), False))
        if c:
            seen_correct = True
        elif seen_correct:
            return int(l)
    return None


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    baseline_path = input_dir / "baseline_samples.csv"
    intervention_path = input_dir / "per_sample_interventions.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not intervention_path.exists():
        raise FileNotFoundError(intervention_path)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out_dir} exists. Use --overwrite or choose another --output-dir."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_raw = read_csv(baseline_path)
    intervention_raw = read_csv(intervention_path)

    baseline: Dict[int, Dict[str, Any]] = {}
    for r in baseline_raw:
        sid = int(r["sid"])
        baseline[sid] = {
            "sid": sid,
            "gt": str(r["gt"]),
            "real_pred": str(r["real_pred"]),
            "real_correct": as_bool(r["real_correct"]),
            "real_margin": as_float(r["real_margin"]),
            "gray_pred": str(r["gray_pred"]),
            "gray_correct": as_bool(r["gray_correct"]),
            "gray_margin": as_float(r["gray_margin"]),
            "subject": r.get("subject", ""),
            "reference": r.get("reference", ""),
        }

    # Keep only the two causal sufficiency branches central to this analysis.
    # E_l: all_text sufficiency
    # D_l: last sufficiency
    by_sid_layer: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(dict)
    observed_layers = set()

    for r in intervention_raw:
        if str(r.get("condition", "")) != "sufficiency":
            continue
        role = str(r.get("role", ""))
        if role not in {"all_text", "last"}:
            continue
        sid = int(r["sid"])
        if sid not in baseline:
            continue
        layer = int(r["layer"])
        if args.min_layer is not None and layer < args.min_layer:
            continue
        if args.max_layer is not None and layer > args.max_layer:
            continue
        observed_layers.add(layer)

        prefix = "E" if role == "all_text" else "D"
        by_sid_layer[(sid, layer)][f"{prefix}_pred"] = str(r["pred"])
        by_sid_layer[(sid, layer)][f"{prefix}_correct"] = as_bool(r["correct"])
        by_sid_layer[(sid, layer)][f"{prefix}_margin"] = as_float(r["margin"])

    layers = sorted(observed_layers)
    if not layers:
        raise RuntimeError("No sufficiency rows for roles all_text/last in the selected layer range.")

    # Diagnostics: dense scans are much more interpretable.
    layer_gaps = [b - a for a, b in zip(layers[:-1], layers[1:])]
    dense = all(g == 1 for g in layer_gaps) if layer_gaps else True
    if not dense:
        print(
            "[WARN] Selected layer grid is not dense/consecutive:",
            layers,
        )
        print(
            "       Stable onset is computed over consecutive OBSERVED checkpoints, "
            "not necessarily consecutive decoder layers."
        )

    # Require matched E and D at each sample/layer for trajectory rows.
    traj_rows: List[Dict[str, Any]] = []
    sample_maps: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    for sid in sorted(baseline):
        b = baseline[sid]
        for layer in layers:
            d = by_sid_layer.get((sid, layer), {})
            if not all(k in d for k in ("E_pred", "E_correct", "E_margin", "D_pred", "D_correct", "D_margin")):
                continue
            row = {
                "sid": sid,
                "gt": b["gt"],
                "subject": b["subject"],
                "reference": b["reference"],
                "real_pred": b["real_pred"],
                "real_correct": b["real_correct"],
                "real_margin": b["real_margin"],
                "gray_pred": b["gray_pred"],
                "gray_correct": b["gray_correct"],
                "gray_margin": b["gray_margin"],
                "layer": layer,
                "E_pred": d["E_pred"],
                "E_correct": d["E_correct"],
                "E_margin": d["E_margin"],
                "D_pred": d["D_pred"],
                "D_correct": d["D_correct"],
                "D_margin": d["D_margin"],
                "E_minus_D_margin": float(d["E_margin"] - d["D_margin"]),
                "E_agrees_with_final": bool(d["E_pred"] == b["real_pred"]),
                "D_agrees_with_final": bool(d["D_pred"] == b["real_pred"]),
            }
            traj_rows.append(row)
            sample_maps[sid][layer] = row

    write_csv(out_dir / "per_sample_layer_trajectories.csv", traj_rows)

    # ------------------------------------------------------------------
    # Per-layer correct vs wrong group comparison.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    per_layer_rows: List[Dict[str, Any]] = []

    for layer in layers:
        rr = [r for r in traj_rows if int(r["layer"]) == layer]
        groups = {
            "correct": [r for r in rr if bool(r["real_correct"])],
            "wrong": [r for r in rr if not bool(r["real_correct"])],
        }
        c = groups["correct"]
        w = groups["wrong"]

        row: Dict[str, Any] = {
            "layer": layer,
            "n_correct": len(c),
            "n_wrong": len(w),
            "correct__E_acc": frac(r["E_correct"] for r in c),
            "wrong__E_acc": frac(r["E_correct"] for r in w),
            "correct__D_acc": frac(r["D_correct"] for r in c),
            "wrong__D_acc": frac(r["D_correct"] for r in w),
            "correct__mean_E_margin": mean(r["E_margin"] for r in c),
            "wrong__mean_E_margin": mean(r["E_margin"] for r in w),
            "correct__median_E_margin": median(r["E_margin"] for r in c),
            "wrong__median_E_margin": median(r["E_margin"] for r in w),
            "correct__mean_D_margin": mean(r["D_margin"] for r in c),
            "wrong__mean_D_margin": mean(r["D_margin"] for r in w),
            "correct__median_D_margin": median(r["D_margin"] for r in c),
            "wrong__median_D_margin": median(r["D_margin"] for r in w),
            "correct__mean_gap": mean(r["E_minus_D_margin"] for r in c),
            "wrong__mean_gap": mean(r["E_minus_D_margin"] for r in w),
            "correct__median_gap": median(r["E_minus_D_margin"] for r in c),
            "wrong__median_gap": median(r["E_minus_D_margin"] for r in w),
            "correct__E_correct_D_wrong": frac(
                bool(r["E_correct"]) and (not bool(r["D_correct"])) for r in c
            ),
            "wrong__E_correct_D_wrong": frac(
                bool(r["E_correct"]) and (not bool(r["D_correct"])) for r in w
            ),
        }

        for metric, field in [
            ("E_margin", "E_margin"),
            ("D_margin", "D_margin"),
            ("gap", "E_minus_D_margin"),
        ]:
            diff, lo, hi = bootstrap_mean_diff(
                [r[field] for r in c],
                [r[field] for r in w],
                n_boot=args.bootstrap,
                rng=rng,
            )
            row[f"wrong_minus_correct__{metric}"] = diff
            row[f"wrong_minus_correct__{metric}_ci95_lo"] = lo
            row[f"wrong_minus_correct__{metric}_ci95_hi"] = hi

        per_layer_rows.append(row)

    write_csv(out_dir / "per_layer_correct_vs_wrong.csv", per_layer_rows)

    # ------------------------------------------------------------------
    # Per-sample stage metrics and failure taxonomy.
    # ------------------------------------------------------------------
    sample_rows: List[Dict[str, Any]] = []

    for sid in sorted(sample_maps):
        m = sample_maps[sid]
        available = [l for l in layers if l in m]
        if not available:
            continue
        b = baseline[sid]

        E_corr = {l: bool(m[l]["E_correct"]) for l in available}
        D_corr = {l: bool(m[l]["D_correct"]) for l in available}

        E_onset = stable_onset(available, E_corr, args.stable_k)
        D_onset = stable_onset(available, D_corr, args.stable_k)

        E_first = first_correct_layer(available, E_corr)
        D_first = first_correct_layer(available, D_corr)

        D_flip_layer = first_wrong_after_correct(available, D_corr)

        last_layer = available[-1]
        E_last_correct = E_corr[last_layer]
        D_last_correct = D_corr[last_layer]

        evidence_ever = any(E_corr.values())
        decision_ever = any(D_corr.values())
        stable_evidence = E_onset is not None
        stable_decision = D_onset is not None

        # Disjoint descriptive taxonomy for normal REAL errors.
        failure_type = "normal_correct"
        if not b["real_correct"]:
            if not stable_evidence:
                failure_type = "no_stable_text_evidence"
            elif not stable_decision:
                failure_type = "stable_text_evidence_no_stable_last_decision"
            elif D_flip_layer is not None and not D_last_correct:
                failure_type = "stable_last_decision_then_branch_flip"
            elif D_last_correct:
                failure_type = "stable_last_decision_but_full_path_wrong"
            else:
                failure_type = "other_late_stage_conflict"

        row = {
            "sid": sid,
            "gt": b["gt"],
            "subject": b["subject"],
            "reference": b["reference"],
            "real_pred": b["real_pred"],
            "real_correct": b["real_correct"],
            "real_margin": b["real_margin"],
            "gray_pred": b["gray_pred"],
            "gray_correct": b["gray_correct"],
            "gray_margin": b["gray_margin"],
            "first_layer": available[0],
            "last_layer": last_layer,
            "n_layers": len(available),
            "stable_k": args.stable_k,
            "E_first_correct_layer": E_first if E_first is not None else "",
            "D_first_correct_layer": D_first if D_first is not None else "",
            "E_stable_onset": E_onset if E_onset is not None else "",
            "D_stable_onset": D_onset if D_onset is not None else "",
            "D_minus_E_onset_gap": (
                int(D_onset - E_onset)
                if E_onset is not None and D_onset is not None
                else ""
            ),
            "E_ever_correct": evidence_ever,
            "D_ever_correct": decision_ever,
            "E_stable_correct": stable_evidence,
            "D_stable_correct": stable_decision,
            "E_last_correct": E_last_correct,
            "D_last_correct": D_last_correct,
            "D_first_wrong_after_any_correct": (
                D_flip_layer if D_flip_layer is not None else ""
            ),
            "D_branch_late_flip": bool(decision_ever and not D_last_correct),
            "mean_E_margin": mean(m[l]["E_margin"] for l in available),
            "mean_D_margin": mean(m[l]["D_margin"] for l in available),
            "mean_E_minus_D_margin": mean(
                m[l]["E_minus_D_margin"] for l in available
            ),
            "max_E_margin": max(float(m[l]["E_margin"]) for l in available),
            "max_D_margin": max(float(m[l]["D_margin"]) for l in available),
            "E_correct_fraction": frac(E_corr.values()),
            "D_correct_fraction": frac(D_corr.values()),
            "failure_type": failure_type,
            # Broad, intentionally cautious flag:
            "post_evidence_failure_candidate": bool(
                (not b["real_correct"]) and stable_evidence
            ),
        }
        sample_rows.append(row)

    write_csv(out_dir / "per_sample_stage_metrics.csv", sample_rows)

    # ------------------------------------------------------------------
    # Failure taxonomy among final errors.
    # ------------------------------------------------------------------
    wrong_rows = [r for r in sample_rows if not bool(r["real_correct"])]
    taxonomy = Counter(str(r["failure_type"]) for r in wrong_rows)
    taxonomy_rows = []
    n_wrong = len(wrong_rows)
    for name, n in sorted(taxonomy.items(), key=lambda kv: (-kv[1], kv[0])):
        taxonomy_rows.append({
            "failure_type": name,
            "n": n,
            "fraction_of_generation_errors": (n / n_wrong) if n_wrong else float("nan"),
        })

    # Add broad aggregates useful for deciding whether a deployment-focused FT is justified.
    broad = [
        (
            "ANY_stable_text_evidence_among_errors",
            sum(bool(r["E_stable_correct"]) for r in wrong_rows),
        ),
        (
            "ANY_stable_last_decision_among_errors",
            sum(bool(r["D_stable_correct"]) for r in wrong_rows),
        ),
        (
            "post_evidence_failure_candidate",
            sum(bool(r["post_evidence_failure_candidate"]) for r in wrong_rows),
        ),
        (
            "last_branch_late_flip_among_errors",
            sum(bool(r["D_branch_late_flip"]) for r in wrong_rows),
        ),
        (
            "stable_text_evidence_no_stable_last_decision",
            sum(
                str(r["failure_type"]) == "stable_text_evidence_no_stable_last_decision"
                for r in wrong_rows
            ),
        ),
    ]
    for name, n in broad:
        taxonomy_rows.append({
            "failure_type": name,
            "n": n,
            "fraction_of_generation_errors": (n / n_wrong) if n_wrong else float("nan"),
        })

    write_csv(out_dir / "failure_taxonomy.csv", taxonomy_rows)

    # ------------------------------------------------------------------
    # Per-layer correctness transition counts.
    # This directly exposes:
    #   E correct / D wrong
    #   E wrong / D wrong
    #   E correct / D correct
    # separately for final-correct and final-wrong samples.
    # ------------------------------------------------------------------
    transition_rows: List[Dict[str, Any]] = []
    for layer in layers:
        rr = [r for r in traj_rows if int(r["layer"]) == layer]
        for final_group, selector in [
            ("generation_correct", True),
            ("generation_wrong", False),
        ]:
            gg = [r for r in rr if bool(r["real_correct"]) == selector]
            counts = Counter()
            for r in gg:
                e = "E+" if bool(r["E_correct"]) else "E-"
                d = "D+" if bool(r["D_correct"]) else "D-"
                counts[f"{e}{d}"] += 1
            n = len(gg)
            transition_rows.append({
                "layer": layer,
                "final_group": final_group,
                "n": n,
                "E+D+": counts["E+D+"],
                "E+D-": counts["E+D-"],
                "E-D+": counts["E-D+"],
                "E-D-": counts["E-D-"],
                "frac_E+D+": counts["E+D+"] / n if n else float("nan"),
                "frac_E+D-": counts["E+D-"] / n if n else float("nan"),
                "frac_E-D+": counts["E-D+"] / n if n else float("nan"),
                "frac_E-D-": counts["E-D-"] / n if n else float("nan"),
            })
    write_csv(out_dir / "transition_counts.csv", transition_rows)

    # ------------------------------------------------------------------
    # Onset summary by final correctness.
    # ------------------------------------------------------------------
    onset_summary = []
    for label, want_correct in [("generation_correct", True), ("generation_wrong", False)]:
        rr = [r for r in sample_rows if bool(r["real_correct"]) == want_correct]

        def numeric(field: str) -> List[float]:
            out = []
            for x in rr:
                v = x.get(field, "")
                if v == "" or v is None:
                    continue
                out.append(float(v))
            return out

        onset_summary.append({
            "group": label,
            "n": len(rr),
            "fraction_with_stable_E": frac(bool(r["E_stable_correct"]) for r in rr),
            "fraction_with_stable_D": frac(bool(r["D_stable_correct"]) for r in rr),
            "median_E_stable_onset": median(numeric("E_stable_onset")),
            "median_D_stable_onset": median(numeric("D_stable_onset")),
            "median_D_minus_E_gap": median(numeric("D_minus_E_onset_gap")),
            "mean_E_correct_fraction": mean(r["E_correct_fraction"] for r in rr),
            "mean_D_correct_fraction": mean(r["D_correct_fraction"] for r in rr),
            "mean_E_margin": mean(r["mean_E_margin"] for r in rr),
            "mean_D_margin": mean(r["mean_D_margin"] for r in rr),
            "mean_E_minus_D_margin": mean(r["mean_E_minus_D_margin"] for r in rr),
        })
    write_csv(out_dir / "onset_summary.csv", onset_summary)

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------
    n_total = len(sample_rows)
    n_correct = sum(bool(r["real_correct"]) for r in sample_rows)
    n_wrong = n_total - n_correct

    print("\n" + "=" * 132)
    print("TEXT EVIDENCE vs LAST-TOKEN DECISION: CORRECT / WRONG COMPARISON")
    print("=" * 132)
    print(f"samples               : {n_total}")
    print(f"generation correct    : {n_correct}")
    print(f"generation wrong      : {n_wrong}")
    print(f"layers                : {layers}")
    print(f"dense layer scan      : {dense}")
    print(f"stable-k              : {args.stable_k}")
    print("-")
    print(
        "E_l = all-text sufficiency GT margin; "
        "D_l = last-token sufficiency GT margin; positive margin => restricted four-way prediction is GT."
    )
    print("-")
    print(
        f"{'layer':<6} {'Eacc C':>8} {'Eacc W':>8} {'Dacc C':>8} {'Dacc W':>8} "
        f"{'Emean C':>10} {'Emean W':>10} {'Dmean C':>10} {'Dmean W':>10} "
        f"{'gap C':>9} {'gap W':>9} {'W:E+D-':>9}"
    )
    for r in per_layer_rows:
        print(
            f"L{int(r['layer']):02d}  "
            f"{r['correct__E_acc']:8.3f} {r['wrong__E_acc']:8.3f} "
            f"{r['correct__D_acc']:8.3f} {r['wrong__D_acc']:8.3f} "
            f"{r['correct__mean_E_margin']:10.4f} {r['wrong__mean_E_margin']:10.4f} "
            f"{r['correct__mean_D_margin']:10.4f} {r['wrong__mean_D_margin']:10.4f} "
            f"{r['correct__mean_gap']:9.4f} {r['wrong__mean_gap']:9.4f} "
            f"{r['wrong__E_correct_D_wrong']:9.3f}"
        )

    print("\nOnset summary:")
    for r in onset_summary:
        print(
            f"{r['group']:<20} n={int(r['n']):3d} "
            f"stableE={r['fraction_with_stable_E']:.3f} "
            f"stableD={r['fraction_with_stable_D']:.3f} "
            f"median LE={r['median_E_stable_onset']:.2f} "
            f"median LD={r['median_D_stable_onset']:.2f} "
            f"median gap={r['median_D_minus_E_gap']:.2f}"
        )

    print("\nGeneration-error taxonomy:")
    for r in taxonomy_rows:
        print(
            f"{r['failure_type']:<48} "
            f"{int(r['n']):4d}/{n_wrong:<4d}  "
            f"{r['fraction_of_generation_errors']:.3f}"
        )

    # A simple go/no-go statement based only on descriptive evidence.
    stable_E_errors = sum(bool(r["E_stable_correct"]) for r in wrong_rows)
    no_stable_D_given_E = sum(
        bool(r["E_stable_correct"]) and (not bool(r["D_stable_correct"]))
        for r in wrong_rows
    )
    stable_D_but_final_wrong = sum(
        bool(r["D_stable_correct"]) for r in wrong_rows
    )

    print("\nKey counts for fine-tuning decision:")
    print(
        f"  generation errors with stable correct text evidence : "
        f"{stable_E_errors}/{n_wrong} "
        f"({stable_E_errors / n_wrong if n_wrong else float('nan'):.3f})"
    )
    print(
        f"  ...but no stable correct last-token decision        : "
        f"{no_stable_D_given_E}/{n_wrong} "
        f"({no_stable_D_given_E / n_wrong if n_wrong else float('nan'):.3f})"
    )
    print(
        f"  generation errors where last branch was stably correct at some point: "
        f"{stable_D_but_final_wrong}/{n_wrong} "
        f"({stable_D_but_final_wrong / n_wrong if n_wrong else float('nan'):.3f})"
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "args": vars(args),
        "input_dir": str(input_dir),
        "layers": layers,
        "dense_layer_scan": dense,
        "n_samples": n_total,
        "n_generation_correct": n_correct,
        "n_generation_wrong": n_wrong,
        "n_errors_with_stable_correct_text_evidence": stable_E_errors,
        "n_errors_with_stable_text_evidence_but_no_stable_last_decision": no_stable_D_given_E,
        "n_errors_with_stable_last_decision": stable_D_but_final_wrong,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    for name in [
        "per_sample_layer_trajectories.csv",
        "per_layer_correct_vs_wrong.csv",
        "per_sample_stage_metrics.csv",
        "failure_taxonomy.csv",
        "transition_counts.csv",
        "onset_summary.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
