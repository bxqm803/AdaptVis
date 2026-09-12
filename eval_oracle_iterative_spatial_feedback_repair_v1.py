#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_oracle_iterative_spatial_feedback_repair_v1.py

Closed-loop oracle spatial feedback repair.

Goal
====
For baseline-WRONG samples only, use the GT relation as the desired target and
repeatedly:

  1) read the CURRENT four-way teacher-forced decision scores;
  2) use a frozen calibration operator A_L to solve the minimum 2-D spatial edit
     predicted to move GT above every competitor by margin gamma;
  3) apply only a SMALL natural-norm step toward that solution;
  4) re-run the model with the cumulative spatial edit;
  5) re-measure both four-way decision scores and actual free generation;
  6) stop when free generation becomes GT, the step budget is exhausted, or the
     cumulative spatial-edit budget is reached.

The model parameters are never changed.  Each feedback step is a fresh forward
from the original prompt with the current cumulative L-layer relation-state
patch.  Therefore there is no hook/state carry-over between steps.

Spatial coordinates
===================
At source layer L, let d_H and d_V be frozen spatial axes fit on the spatial
calibration split.  A requested 2-D coordinate edit dz = [dz_H,dz_V] is realized
as a hidden-state relation edit dx satisfying exactly

    [d_H^T ; d_V^T] dx = dz.

The relation edit is applied symmetrically:

    h_sub <- h_sub + dx/2
    h_ref <- h_ref - dx/2.

Frozen decision transport
=========================
A_L is loaded from a previous spatial->decision controller run:

    delta s_centered ~= A_L delta z,       A_L in R^{4 x 2}.

At every feedback step we use the CURRENT score vector s_k and solve

    min ||x||_2
    s.t. (s_t-s_j) + (A_t-A_j) D x >= gamma  for all j != t,

where x is measured in natural spatial half-gap units and
D=diag(half_gap_H, half_gap_V).

Unlike the previous one-shot repair, the full required x is NOT applied at once.
Only at most --step-natural is applied, then the real model is queried again and
the controller replans from the new decision state.

Primary outputs
===============
baseline.csv
per_step.csv
per_repair.csv
generation_summary.csv
repair_by_relation.csv
step_summary.csv
below_trajectories.csv
analysis_summary.txt
metadata.json
errors.jsonl

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_oracle_iterative_spatial_feedback_repair_v1.py \
  --model qwen-3b \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --transport-run-dir output/qwen3b_spatial_to_decision_controller_L25_n80_v1 \
  --source-layers 25 \
  --spatial-fit-ratio 0.30 \
  --eval-scope heldout \
  --eval-max-samples 80 \
  --required-margins 0.25,0.5 \
  --step-natural 0.75 \
  --max-steps 12 \
  --max-total-natural 8.0 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --max-new-tokens 6 \
  --output-dir output/qwen3b_oracle_iterative_spatial_feedback_L25_n80_v1 \
  --overwrite
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
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import eval_real_causal_token_update_gating_v1 as gate
except Exception as exc:
    raise SystemExit(
        "Could not import eval_real_causal_token_update_gating_v1.py.\n"
        "Run from the AdaptVis repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12


def parse_layers(s: str) -> List[int]:
    out = set()
    for part in str(s).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"No layers parsed from {s!r}")
    return sorted(out)


def parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"No floats parsed from {s!r}")
    if any(v < 0 for v in vals):
        raise ValueError("Required margins must be >=0")
    return sorted(set(vals))


def norm_rel(x) -> str:
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return (v / n).astype(np.float32)


def score_array(scores: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(scores[r]) for r in REL], dtype=np.float64)


def argmax_relation(scores: Dict[str, float]) -> str:
    return max(REL, key=lambda r: float(scores[r]))


def min_gt_margin(scores: Dict[str, float], target: str) -> float:
    return min(float(scores[target]) - float(scores[r]) for r in REL if r != target)


def stratified_cap(rows: Sequence[dict], n: int, seed: int) -> List[dict]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    return list(gate.traj.stratified_cap(rows, int(n), int(seed)))


def load_spatial_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        required = {"relation_vectors", "relation", "decoder_block_index", "sample_index"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
        X = np.asarray(z["relation_vectors"], dtype=np.float32)
        y = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        layers = [int(v) for v in z["decoder_block_index"].tolist()]
        sids = np.asarray(z["sample_index"], dtype=np.int64)
    if X.ndim != 3 or X.shape[0] != len(y) or X.shape[0] != len(sids):
        raise RuntimeError(f"Bad spatial NPZ shapes X={X.shape}, y={y.shape}, sid={sids.shape}")
    return X, y, layers, sids


def make_spatial_split(y, sids, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    fit, held = [], []
    for r in REL:
        idx = np.where(y == r)[0]
        if len(idx) < 2:
            raise RuntimeError(f"Need >=2 samples for {r}, got {len(idx)}")
        idx = idx.copy()
        rng.shuffle(idx)
        nfit = max(1, min(len(idx) - 1, int(round(len(idx) * ratio))))
        fit.extend(idx[:nfit].tolist())
        held.extend(idx[nfit:].tolist())
    fit = np.asarray(sorted(fit), dtype=np.int64)
    held = np.asarray(sorted(held), dtype=np.int64)
    return fit, held, set(map(int, sids[fit])), set(map(int, sids[held]))


def fit_spatial_geometry(X, y, layers, fit_idx, source_layers):
    layer_to_i = {int(L): i for i, L in enumerate(layers)}
    missing = [L for L in source_layers if L not in layer_to_i]
    if missing:
        raise RuntimeError(f"Spatial NPZ missing layers {missing}; has {layers}")

    geom = {}
    axis_rows = []
    for L in source_layers:
        li = layer_to_i[L]
        Xf = X[fit_idx, li].astype(np.float64)
        yf = y[fit_idx]
        center = Xf.mean(axis=0)
        means = {r: Xf[yf == r].mean(axis=0) for r in REL}
        class_dirs = {r: unit(means[r] - center).astype(np.float64) for r in REL}

        dH = unit(class_dirs["right"] - class_dirs["left"]).astype(np.float64)
        dV = unit(class_dirs["above"] - class_dirs["below"]).astype(np.float64)
        gapH = float(np.dot(means["right"] - means["left"], dH))
        gapV = float(np.dot(means["above"] - means["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV

        B = np.stack([dH, dV], axis=1)  # D x 2
        dual = B @ np.linalg.inv(B.T @ B)
        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)
        geom[L] = {
            "layer_index": li,
            "center": center.astype(np.float32),
            "B": B.astype(np.float32),
            "dual": dual.astype(np.float32),
            "natural_half_H": float(halfH),
            "natural_half_V": float(halfV),
        }
        axis_rows.append({
            "source_layer": L,
            "fit_N": len(fit_idx),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "natural_full_gap_H": float(gapH),
            "natural_half_gap_H": float(halfH),
            "natural_full_gap_V": float(gapV),
            "natural_half_gap_V": float(halfV),
            "dual_check_max_abs": float(np.max(np.abs(B.T @ dual - np.eye(2)))),
        })
    return geom, pd.DataFrame(axis_rows)


def realize_delta(geom_L, dz):
    dual = geom_L["dual"].astype(np.float64)
    dz = np.asarray(dz, dtype=np.float64).reshape(2)
    dx = dual @ dz
    realized = geom_L["B"].astype(np.float64).T @ dx
    if np.max(np.abs(realized - dz)) > 1e-4 * max(1.0, float(np.linalg.norm(dz))):
        raise RuntimeError(f"Dual-basis realization mismatch requested={dz}, realized={realized}")
    return dx.astype(np.float32)


def load_all_data(a):
    old_max = getattr(a, "max_samples", None)
    old_eval = a.eval_max_samples
    a.max_samples = 0
    a.eval_max_samples = 0
    two, meta, rec_by_sid = gate.load_data(a)
    a.eval_max_samples = old_eval
    if old_max is None:
        delattr(a, "max_samples")
    else:
        a.max_samples = old_max
    return two, meta, rec_by_sid


def find_object_positions(two, processor, batch, subject: str, reference: str):
    tok = gate.tokenizer_of(processor)
    ids = batch["input_ids"][0].detach().cpu().tolist()
    sub = int(two.find_phrase_last_token(tok, ids, subject))
    ref = int(two.find_phrase_last_token(tok, ids, reference))
    if sub == ref:
        raise RuntimeError(f"subject/reference positions collide at {sub}")
    return sub, ref


def all_scores(model, batch, candidate_ids, reduction, decoder_layers=None, patch_map=None):
    return {
        r: gate.sequence_score(
            model=model,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
            decoder_layers=decoder_layers,
            patch_map=patch_map,
        )
        for r in REL
    }


def make_relation_patch(L, sub_pos, ref_pos, delta_relation):
    d = np.asarray(delta_relation, dtype=np.float32)
    return {
        int(L): {
            int(sub_pos): (+0.5 * d).astype(np.float32),
            int(ref_pos): (-0.5 * d).astype(np.float32),
        }
    }


def load_frozen_transport(run_dir: Path, source_layers: Sequence[int]):
    op_path = run_dir / "transport_operator_4x2.csv"
    if not op_path.exists():
        raise FileNotFoundError(op_path)
    op_df = pd.read_csv(op_path)
    required = {"source_layer", "relation", "dscore_dzH", "dscore_dzV"}
    missing = required - set(op_df.columns)
    if missing:
        raise RuntimeError(f"{op_path} missing columns {sorted(missing)}")

    operators, t_rows = {}, []
    for L in source_layers:
        q = op_df[pd.to_numeric(op_df.source_layer, errors="coerce") == int(L)]
        if len(q) == 0:
            raise RuntimeError(f"{op_path} has no rows for L{L}")
        A = np.zeros((4, 2), dtype=np.float64)
        for ri, r in enumerate(REL):
            qr = q[q.relation.astype(str).map(norm_rel) == r]
            if len(qr) != 1:
                raise RuntimeError(f"Expected one operator row for L{L}/{r}, got {len(qr)}")
            A[ri, 0] = float(qr.iloc[0].dscore_dzH)
            A[ri, 1] = float(qr.iloc[0].dscore_dzV)
        A = A - A.mean(axis=0, keepdims=True)
        T = np.asarray([
            A[RID["right"]] - A[RID["left"]],
            A[RID["above"]] - A[RID["below"]],
        ], dtype=np.float64)
        svals = np.linalg.svd(A, compute_uv=False)
        operators[L] = {"A": A, "T": T}
        diag = 0.5 * (abs(T[0, 0]) + abs(T[1, 1]))
        cross = 0.5 * (abs(T[0, 1]) + abs(T[1, 0]))
        t_rows.append({
            "source_layer": L,
            "T_HH": T[0,0], "T_HV": T[0,1],
            "T_VH": T[1,0], "T_VV": T[1,1],
            "diag_abs_mean": diag,
            "cross_abs_mean": cross,
            "diag_to_cross": diag / max(cross, EPS),
            "A_singular_value_1": float(svals[0]),
            "A_singular_value_2": float(svals[1]),
            "A_condition_number": float(svals[0] / max(svals[-1], EPS)),
        })
    return operators, op_df, pd.DataFrame(t_rows)


def _feasible(C: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float) -> bool:
    return bool(np.all(C @ x >= b - tol))


def solve_min_norm_halfspaces_2d(C: np.ndarray, b: np.ndarray, tol: float = 1e-8):
    """Solve min ||x||_2 subject to Cx>=b in R^2 by active-set enumeration."""
    C = np.asarray(C, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if C.ndim != 2 or C.shape[1] != 2 or C.shape[0] != len(b):
        raise ValueError(f"Bad halfspace shapes C={C.shape}, b={b.shape}")

    candidates = []
    x0 = np.zeros(2, dtype=np.float64)
    if _feasible(C, b, x0, tol):
        candidates.append((0.0, x0, ()))

    for i in range(len(b)):
        c = C[i]
        denom = float(np.dot(c, c))
        if denom < EPS:
            continue
        x = (float(b[i]) / denom) * c
        if _feasible(C, b, x, tol):
            candidates.append((float(np.dot(x, x)), x, (i,)))

    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            M = np.stack([C[i], C[j]], axis=0)
            if abs(float(np.linalg.det(M))) < 1e-10:
                continue
            x = np.linalg.solve(M, np.asarray([b[i], b[j]], dtype=np.float64))
            if _feasible(C, b, x, tol):
                candidates.append((float(np.dot(x, x)), x, (i, j)))

    if not candidates:
        return np.full(2, np.nan, dtype=np.float64), False, ()
    candidates.sort(key=lambda q: q[0])
    return candidates[0][1], True, candidates[0][2]


def solve_remaining_edit_natural(
    *, current_scores: Dict[str, float], target: str, A: np.ndarray,
    geom_L: dict, required_margin: float, tol: float,
):
    """Return minimum *incremental* edit x in natural units from the current state."""
    half = np.asarray([
        float(geom_L["natural_half_H"]),
        float(geom_L["natural_half_V"]),
    ], dtype=np.float64)
    D = np.diag(half)
    s = score_array(current_scores)
    t = RID[target]
    competitors = [r for r in REL if r != target]
    C_rows, b_rows = [], []
    for r in competitors:
        j = RID[r]
        C_rows.append((A[t] - A[j]) @ D)
        b_rows.append(float(required_margin) - float(s[t] - s[j]))
    C = np.asarray(C_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    x, feasible, active = solve_min_norm_halfspaces_2d(C, b, tol=tol)
    return {
        "x_required": x,
        "feasible": bool(feasible),
        "required_natural_norm": float(np.linalg.norm(x)) if feasible else np.nan,
        "active_constraints": active,
        "competitors": competitors,
    }


def natural_to_dz(geom_L: dict, x: np.ndarray) -> np.ndarray:
    half = np.asarray([
        float(geom_L["natural_half_H"]),
        float(geom_L["natural_half_V"]),
    ], dtype=np.float64)
    return half * np.asarray(x, dtype=np.float64).reshape(2)


def clip_step(x_req: np.ndarray, step_natural: float, gain: float) -> np.ndarray:
    x = np.asarray(x_req, dtype=np.float64).reshape(2) * float(gain)
    n = float(np.linalg.norm(x))
    if n < EPS:
        return np.zeros(2, dtype=np.float64)
    if step_natural > 0 and n > step_natural:
        x = x * (float(step_natural) / n)
    return x


def clip_total_step(x_total: np.ndarray, x_step: np.ndarray, max_total: float):
    proposed = np.asarray(x_total, dtype=np.float64) + np.asarray(x_step, dtype=np.float64)
    if max_total <= 0:
        return np.asarray(x_step, dtype=np.float64), False
    n_prop = float(np.linalg.norm(proposed))
    if n_prop <= max_total + 1e-12:
        return np.asarray(x_step, dtype=np.float64), False

    # Intersect the ray x_total + lambda*x_step with ||.|| = max_total, lambda in [0,1].
    a = float(np.dot(x_step, x_step))
    b = 2.0 * float(np.dot(x_total, x_step))
    c = float(np.dot(x_total, x_total) - max_total * max_total)
    if a < EPS:
        return np.zeros(2, dtype=np.float64), True
    disc = max(0.0, b*b - 4*a*c)
    roots = [(-b + math.sqrt(disc)) / (2*a), (-b - math.sqrt(disc)) / (2*a)]
    valid = [lam for lam in roots if -1e-10 <= lam <= 1.0 + 1e-10]
    if not valid:
        # Conservative fallback: no further step.
        return np.zeros(2, dtype=np.float64), True
    lam = max(0.0, min(1.0, max(valid)))
    return np.asarray(x_step, dtype=np.float64) * lam, True


def build_patch_from_total(L, sub_pos, ref_pos, geom_L, x_total):
    dz_total = natural_to_dz(geom_L, x_total)
    dx_total = realize_delta(geom_L, dz_total)
    return make_relation_patch(L, sub_pos, ref_pos, dx_total), dz_total, dx_total


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--spatial-states-npz", required=True)
    p.add_argument("--transport-run-dir", required=True)
    p.add_argument("--source-layers", default="25")
    p.add_argument("--spatial-fit-ratio", type=float, default=0.30)
    p.add_argument("--spatial-fit-seed", type=int, default=1)
    p.add_argument("--eval-scope", default="heldout", choices=["heldout", "all_data"])
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--required-margins", default="0.25,0.5",
        help="Desired GT-vs-every-competitor margin used by each feedback replan.",
    )
    p.add_argument(
        "--step-natural", type=float, default=0.75,
        help="Maximum natural-norm edit added per feedback step.",
    )
    p.add_argument(
        "--step-gain", type=float, default=1.0,
        help="Multiply the locally solved remaining edit before the per-step norm cap.",
    )
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument(
        "--max-total-natural", type=float, default=8.0,
        help="Maximum cumulative natural-norm spatial edit. <=0 disables.",
    )
    p.add_argument("--qp-tol", type=float, default=1e-8)
    p.add_argument(
        "--generation-margin-escalation", type=float, default=0.25,
        help=(
            "If free generation is still wrong but the current teacher-forced GT margin already "
            "meets --required-margins, ask for this much additional GT margin and continue. "
            "This handles the observed mismatch between teacher-forced argmax and free generation."
        ),
    )
    p.add_argument(
        "--stop-on", default="generation", choices=["generation", "teacher_forced", "either"],
        help="Stopping criterion after each feedback step. 'generation' is the main experiment.",
    )
    p.add_argument(
        "--generation-check-every", type=int, default=1,
        help="Run free generation every N feedback steps; always runs on the final step.",
    )

    p.add_argument(
        "--answer-surface", default="above_below", choices=["above_below", "on_under"]
    )
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument("--sequence-score-reduction", default="mean", choices=["mean", "sum"])
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    a = p.parse_args()
    if not (0 < a.spatial_fit_ratio < 1):
        p.error("--spatial-fit-ratio must be in (0,1)")
    if a.step_natural <= 0:
        p.error("--step-natural must be >0")
    if a.step_gain <= 0:
        p.error("--step-gain must be >0")
    if a.max_steps <= 0:
        p.error("--max-steps must be >0")
    if a.generation_margin_escalation < 0:
        p.error("--generation-margin-escalation must be >=0")
    if a.generation_check_every <= 0:
        p.error("--generation-check-every must be >0")
    return a


def should_stop(stop_on: str, gen_correct: bool, tf_correct: bool) -> bool:
    if stop_on == "generation":
        return bool(gen_correct)
    if stop_on == "teacher_forced":
        return bool(tf_correct)
    return bool(gen_correct or tf_correct)


def summarize_generation(repair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (L, margin), g in repair_df.groupby(["source_layer", "required_margin"]):
        base = g.baseline_correct.astype(bool).to_numpy()
        patch = g.patched_correct.astype(bool).to_numpy()
        wrong = ~base
        w2c = int(np.sum(wrong & patch))
        c2w = int(np.sum(base & (~patch)))
        gw = g.loc[wrong]
        rows.append({
            "source_layer": int(L),
            "required_margin": float(margin),
            "N": len(g),
            "baseline_accuracy": float(np.mean(base)),
            "patched_accuracy": float(np.mean(patch)),
            "gain": float(np.mean(patch) - np.mean(base)),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "baseline_wrong_N": int(np.sum(wrong)),
            "repair_rate_on_wrong": float(w2c / max(int(np.sum(wrong)), 1)),
            "preserve_rate_on_correct": float(1.0 - c2w / max(int(np.sum(base)), 1)),
            "mean_steps_wrong": float(gw.steps_taken.mean()) if len(gw) else np.nan,
            "median_steps_wrong": float(gw.steps_taken.median()) if len(gw) else np.nan,
            "mean_final_total_natural_wrong": float(gw.final_total_natural_norm.mean()) if len(gw) else np.nan,
            "fraction_hit_total_cap_wrong": float(gw.hit_total_cap.mean()) if len(gw) else np.nan,
            "fraction_ever_tf_target_wrong": float(gw.ever_tf_target.mean()) if len(gw) else np.nan,
            "final_tf_target_rate_wrong": float((gw["final_tf_prediction"] == gw["gt"]).mean()) if len(gw) else np.nan,
            "mean_final_gt_margin_wrong": float(gw.final_gt_min_margin.mean()) if len(gw) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["source_layer", "required_margin"]).reset_index(drop=True)


def summarize_relation(repair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (L, margin, rel), g in repair_df.groupby(["source_layer", "required_margin", "gt"]):
        base = g.baseline_correct.astype(bool).to_numpy()
        patch = g.patched_correct.astype(bool).to_numpy()
        wrong = ~base
        w2c = int(np.sum(wrong & patch))
        gw = g.loc[wrong]
        rows.append({
            "source_layer": int(L),
            "required_margin": float(margin),
            "relation": rel,
            "N": len(g),
            "baseline_accuracy": float(np.mean(base)),
            "patched_accuracy": float(np.mean(patch)),
            "gain": float(np.mean(patch)-np.mean(base)),
            "wrong_to_correct": w2c,
            "baseline_wrong_N": int(np.sum(wrong)),
            "repair_rate_on_wrong": float(w2c / max(int(np.sum(wrong)),1)),
            "mean_steps_wrong": float(gw.steps_taken.mean()) if len(gw) else np.nan,
            "mean_total_natural_wrong": float(gw.final_total_natural_norm.mean()) if len(gw) else np.nan,
            "mean_final_gt_margin_wrong": float(gw.final_gt_min_margin.mean()) if len(gw) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["source_layer", "required_margin", "relation"])


def summarize_steps(step_df: pd.DataFrame) -> pd.DataFrame:
    if len(step_df) == 0:
        return pd.DataFrame()
    rows = []
    for (L, margin, step), g in step_df.groupby(["source_layer", "required_margin", "step"]):
        rows.append({
            "source_layer": int(L),
            "required_margin": float(margin),
            "step": int(step),
            "N_active": len(g),
            "mean_step_natural": float(g.step_natural_norm.mean()),
            "mean_total_natural": float(g.total_natural_norm.mean()),
            "mean_gt_margin_before": float(g.gt_margin_before.mean()),
            "mean_gt_margin_after": float(g.gt_margin_after.mean()),
            "mean_actual_margin_gain": float(g.actual_gt_margin_gain.mean()),
            "mean_predicted_margin_gain": float(g.predicted_gt_margin_gain.mean()),
            "corr_pred_actual_margin_gain": (
                float(g[["predicted_gt_margin_gain", "actual_gt_margin_gain"]].corr().iloc[0,1])
                if len(g) >= 3 and g.predicted_gt_margin_gain.std() > 1e-12 and g.actual_gt_margin_gain.std() > 1e-12
                else np.nan
            ),
            "tf_target_rate_after": float(g.tf_correct_after.mean()),
            "generation_target_rate_after": float(g.gen_correct_after.dropna().mean()) if g.gen_correct_after.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["source_layer", "required_margin", "step"])


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_layers(a.source_layers)
    margins = parse_float_list(a.required_margins)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

    spatial_path = Path(a.spatial_states_npz)
    X, y, spatial_layers, spatial_sids = load_spatial_npz(spatial_path)
    fit_idx, held_idx, fit_sids, held_sids = make_spatial_split(
        y, spatial_sids, a.spatial_fit_ratio, a.spatial_fit_seed
    )
    geom, axis_df = fit_spatial_geometry(X, y, spatial_layers, fit_idx, source_layers)
    axis_df.to_csv(outdir / "spatial_axis_geometry.csv", index=False)

    spatial_sid_set = set(map(int, spatial_sids.tolist()))
    two, all_meta, rec_by_sid = load_all_data(a)
    all_meta = [m for m in all_meta if int(m["sid"]) in spatial_sid_set]
    eval_meta = all_meta
    if a.eval_scope == "heldout":
        eval_meta = [m for m in eval_meta if int(m["sid"]) in held_sids]
    eval_meta = stratified_cap(eval_meta, a.eval_max_samples, a.seed)
    if not eval_meta:
        raise RuntimeError("No evaluation samples")

    transport_source = Path(a.transport_run_dir)
    operators, op_df, t_df = load_frozen_transport(transport_source, source_layers)
    op_df.to_csv(outdir / "transport_operator_4x2.csv", index=False)
    t_df.to_csv(outdir / "transport_matrix_2x2.csv", index=False)

    model = processor = None
    baseline_rows, repair_rows, step_rows = [], [], []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        if max(source_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(source_layers)} but model has {len(decoder_layers)} layers"
            )
        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)
        device = torch.device(a.device)

        print("\n" + "=" * 190)
        print("ORACLE ITERATIVE SPATIAL FEEDBACK REPAIR")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source layers={source_layers}")
        print(f"transport={transport_source}")
        print(f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}")
        print(f"evaluation N={len(eval_meta)} scope={a.eval_scope}")
        print(f"margins={margins}; step_natural={a.step_natural}; max_steps={a.max_steps}; max_total={a.max_total_natural}")
        print(f"stop_on={a.stop_on}; generation_check_every={a.generation_check_every}")
        print("baseline-correct samples are never edited")
        print("=" * 190, flush=True)

        for m in tqdm(eval_meta, desc="EVAL iterative spatial feedback"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            image = batch = None
            try:
                image = gate.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = gate.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                sub_pos, ref_pos = find_object_positions(
                    two, processor, batch, m["subject"], m["reference"]
                )

                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map={},
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = base_gen_pred == gt
                base_scores = all_scores(
                    model, batch, candidate_ids, a.sequence_score_reduction
                )
                base_tf_pred = argmax_relation(base_scores)
                base_gt_margin = min_gt_margin(base_scores, gt)

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "subject": m["subject"],
                    "reference": m["reference"],
                    "subject_position": sub_pos,
                    "reference_position": ref_pos,
                    "baseline_generation_prediction": base_gen_pred,
                    "baseline_generation_correct": bool(base_correct),
                    "baseline_generation_text": base_gen_text,
                    "baseline_tf_prediction": base_tf_pred,
                    "baseline_tf_correct": bool(base_tf_pred == gt),
                    "baseline_gt_min_margin": float(base_gt_margin),
                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                })

                for L in source_layers:
                    g = geom[L]
                    A = operators[L]["A"]

                    for margin in margins:
                        if base_correct:
                            repair_rows.append({
                                "sid": sid, "gt": gt, "source_layer": L,
                                "required_margin": margin,
                                "baseline_prediction": base_gen_pred,
                                "patched_prediction": base_gen_pred,
                                "baseline_correct": True, "patched_correct": True,
                                "steps_taken": 0,
                                "final_total_natural_norm": 0.0,
                                "final_x_H": 0.0, "final_x_V": 0.0,
                                "final_dz_H": 0.0, "final_dz_V": 0.0,
                                "hit_total_cap": False,
                                "solver_failed": False,
                                "ever_tf_target": bool(base_tf_pred == gt),
                                "final_tf_prediction": base_tf_pred,
                                "final_gt_min_margin": float(base_gt_margin),
                                "baseline_generation_text": base_gen_text,
                                "patched_generation_text": base_gen_text,
                                "stop_reason": "baseline_correct_preserved",
                            })
                            continue

                        x_total = np.zeros(2, dtype=np.float64)
                        current_scores = dict(base_scores)
                        current_tf_pred = base_tf_pred
                        current_gen_pred = base_gen_pred
                        current_gen_text = base_gen_text
                        ever_tf_target = current_tf_pred == gt
                        hit_total_cap = False
                        solver_failed = False
                        stop_reason = "max_steps"
                        steps_taken = 0

                        for step in range(1, a.max_steps + 1):
                            gt_margin_before = min_gt_margin(current_scores, gt)
                            # If teacher-forced scores already satisfy the nominal margin but free
                            # generation is still wrong, keep pushing a little farther instead of
                            # declaring a zero-step solution.
                            effective_margin = float(margin)
                            if (
                                a.stop_on in {"generation", "either"}
                                and current_gen_pred != gt
                                and gt_margin_before >= effective_margin - a.qp_tol
                                and a.generation_margin_escalation > 0
                            ):
                                effective_margin = float(gt_margin_before + a.generation_margin_escalation)

                            sol = solve_remaining_edit_natural(
                                current_scores=current_scores,
                                target=gt,
                                A=A,
                                geom_L=g,
                                required_margin=effective_margin,
                                tol=a.qp_tol,
                            )
                            if not sol["feasible"]:
                                solver_failed = True
                                stop_reason = "solver_infeasible"
                                break

                            x_req = sol["x_required"]
                            req_norm = float(np.linalg.norm(x_req))
                            if req_norm < 1e-10:
                                if current_gen_pred == gt:
                                    stop_reason = "generation_repaired"
                                else:
                                    stop_reason = "zero_remaining_edit"
                                break

                            x_step = clip_step(x_req, a.step_natural, a.step_gain)
                            x_step, clipped_by_total = clip_total_step(
                                x_total, x_step, a.max_total_natural
                            )
                            if clipped_by_total:
                                hit_total_cap = True
                            step_norm = float(np.linalg.norm(x_step))
                            if step_norm < 1e-10:
                                stop_reason = "total_cap_reached"
                                break

                            x_before = x_total.copy()
                            x_total = x_total + x_step
                            patch_map, dz_total, dx_total = build_patch_from_total(
                                L, sub_pos, ref_pos, g, x_total
                            )

                            dz_step = natural_to_dz(g, x_step)
                            predicted_next_arr = score_array(current_scores) + A @ dz_step
                            pred_scores = {r: float(predicted_next_arr[RID[r]]) for r in REL}
                            predicted_gt_margin = min_gt_margin(pred_scores, gt)
                            predicted_gain = float(predicted_gt_margin - gt_margin_before)

                            new_scores = all_scores(
                                model,
                                batch,
                                candidate_ids,
                                a.sequence_score_reduction,
                                decoder_layers=decoder_layers,
                                patch_map=patch_map,
                            )
                            new_tf_pred = argmax_relation(new_scores)
                            new_tf_correct = new_tf_pred == gt
                            ever_tf_target = bool(ever_tf_target or new_tf_correct)
                            new_gt_margin = min_gt_margin(new_scores, gt)
                            actual_gain = float(new_gt_margin - gt_margin_before)

                            run_gen_now = (
                                step % a.generation_check_every == 0
                                or step == a.max_steps
                                or new_tf_correct
                                or (a.max_total_natural > 0 and float(np.linalg.norm(x_total)) >= a.max_total_natural - 1e-8)
                            )
                            if run_gen_now:
                                new_gen_pred, new_gen_text = gate.generate_with_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=batch,
                                    patch_map=patch_map,
                                    max_new_tokens=a.max_new_tokens,
                                )
                                new_gen_correct = new_gen_pred == gt
                                current_gen_pred, current_gen_text = new_gen_pred, new_gen_text
                            else:
                                new_gen_pred, new_gen_text, new_gen_correct = None, None, None

                            steps_taken = step
                            step_rows.append({
                                "sid": sid, "gt": gt, "source_layer": L,
                                "required_margin": margin, "step": step,
                                "baseline_prediction": base_gen_pred,
                                "tf_prediction_before": current_tf_pred,
                                "tf_prediction_after": new_tf_pred,
                                "tf_correct_after": bool(new_tf_correct),
                                "generation_prediction_after": new_gen_pred,
                                "gen_correct_after": (np.nan if new_gen_correct is None else bool(new_gen_correct)),
                                "solver_required_natural_norm": req_norm,
                                "effective_required_margin": float(effective_margin),
                                "step_natural_norm": step_norm,
                                "total_natural_norm": float(np.linalg.norm(x_total)),
                                "step_x_H": float(x_step[0]), "step_x_V": float(x_step[1]),
                                "total_x_H": float(x_total[0]), "total_x_V": float(x_total[1]),
                                "total_dz_H": float(dz_total[0]), "total_dz_V": float(dz_total[1]),
                                "gt_margin_before": float(gt_margin_before),
                                "predicted_gt_margin_after": float(predicted_gt_margin),
                                "gt_margin_after": float(new_gt_margin),
                                "predicted_gt_margin_gain": predicted_gain,
                                "actual_gt_margin_gain": actual_gain,
                                "prediction_error_margin_gain": float(actual_gain - predicted_gain),
                                "hit_total_cap_this_step": bool(clipped_by_total),
                                "active_constraints": ",".join(map(str, sol["active_constraints"])),
                                **{f"score_before_{r}": float(current_scores[r]) for r in REL},
                                **{f"score_after_{r}": float(new_scores[r]) for r in REL},
                            })

                            current_scores = new_scores
                            current_tf_pred = new_tf_pred

                            gen_correct_for_stop = bool(new_gen_correct) if new_gen_correct is not None else False
                            if should_stop(a.stop_on, gen_correct_for_stop, new_tf_correct):
                                stop_reason = (
                                    "generation_repaired" if gen_correct_for_stop
                                    else "teacher_forced_target"
                                )
                                break

                            if a.max_total_natural > 0 and float(np.linalg.norm(x_total)) >= a.max_total_natural - 1e-8:
                                stop_reason = "total_cap_reached"
                                break

                        # Always obtain final free generation for the cumulative edit if the
                        # last loop iteration did not already evaluate it.
                        if steps_taken > 0:
                            patch_map, dz_total, dx_total = build_patch_from_total(
                                L, sub_pos, ref_pos, g, x_total
                            )
                            if current_gen_pred is None or (
                                len(step_rows) > 0
                                and step_rows[-1]["sid"] == sid
                                and step_rows[-1]["source_layer"] == L
                                and float(step_rows[-1]["required_margin"]) == float(margin)
                                and pd.isna(step_rows[-1]["gen_correct_after"])
                            ):
                                current_gen_pred, current_gen_text = gate.generate_with_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=batch,
                                    patch_map=patch_map,
                                    max_new_tokens=a.max_new_tokens,
                                )
                        else:
                            dz_total = natural_to_dz(g, x_total)

                        final_gt_margin = min_gt_margin(current_scores, gt)
                        repair_rows.append({
                            "sid": sid, "gt": gt, "source_layer": L,
                            "required_margin": margin,
                            "baseline_prediction": base_gen_pred,
                            "patched_prediction": current_gen_pred,
                            "baseline_correct": False,
                            "patched_correct": bool(current_gen_pred == gt),
                            "steps_taken": int(steps_taken),
                            "final_total_natural_norm": float(np.linalg.norm(x_total)),
                            "final_x_H": float(x_total[0]), "final_x_V": float(x_total[1]),
                            "final_dz_H": float(dz_total[0]), "final_dz_V": float(dz_total[1]),
                            "hit_total_cap": bool(hit_total_cap),
                            "solver_failed": bool(solver_failed),
                            "ever_tf_target": bool(ever_tf_target),
                            "final_tf_prediction": current_tf_pred,
                            "final_gt_min_margin": float(final_gt_margin),
                            "baseline_generation_text": base_gen_text,
                            "patched_generation_text": current_gen_text,
                            "stop_reason": stop_reason,
                        })

            except Exception as exc:
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "stage": "evaluation", "sid": sid,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-18:],
                    }, ensure_ascii=False) + "\n")
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                batch = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not repair_rows:
            raise RuntimeError(f"No successful evaluation rows. See {err_path}")

        base_df = pd.DataFrame(baseline_rows).drop_duplicates("sid")
        repair_df = pd.DataFrame(repair_rows)
        step_df = pd.DataFrame(step_rows)
        gen_df = summarize_generation(repair_df)
        rel_df = summarize_relation(repair_df)
        step_summary_df = summarize_steps(step_df)

        base_df.to_csv(outdir / "baseline.csv", index=False)
        repair_df.to_csv(outdir / "per_repair.csv", index=False)
        step_df.to_csv(outdir / "per_step.csv", index=False)
        gen_df.to_csv(outdir / "generation_summary.csv", index=False)
        rel_df.to_csv(outdir / "repair_by_relation.csv", index=False)
        step_summary_df.to_csv(outdir / "step_summary.csv", index=False)

        below_df = step_df[step_df["gt"] == "below"].copy() if len(step_df) else pd.DataFrame()
        below_df.to_csv(outdir / "below_trajectories.csv", index=False)

        lines = [
            "=" * 190,
            "ORACLE ITERATIVE SPATIAL FEEDBACK REPAIR",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N successful={base_df.sid.nunique()} / requested={len(eval_meta)}",
            f"source layers={source_layers}",
            f"transport source={transport_source}",
            f"spatial states={spatial_path}",
            f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}; eval_scope={a.eval_scope}",
            f"required margins={margins}",
            f"step_natural={a.step_natural}; step_gain={a.step_gain}; max_steps={a.max_steps}; max_total_natural={a.max_total_natural}",
            f"stop_on={a.stop_on}; generation_check_every={a.generation_check_every}; generation_margin_escalation={a.generation_margin_escalation}",
            f"actual generation baseline accuracy={base_df.baseline_generation_correct.mean():.4f}",
            f"teacher-forced baseline argmax accuracy={base_df.baseline_tf_correct.mean():.4f}",
            "",
            "FROZEN 2x2 AXIS TRANSPORT",
            "-" * 190,
            t_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "",
            "ACTUAL GENERATION REPAIR",
            "-" * 190,
            gen_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "",
            "BY RELATION",
            "-" * 190,
            rel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "",
            "STEPWISE FEEDBACK",
            "-" * 190,
            (step_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(step_summary_df) else "EMPTY"),
            "",
            "Interpretation:",
            "  Each step replans from the model's CURRENT four-way decision state.",
            "  Only a small spatial edit is applied per step; the cumulative edit is re-evaluated from the original prompt.",
            "  Correct baseline generations are never edited, so C->W must remain 0.",
            "  For BELOW, inspect below_trajectories.csv: if actual margin gains stay positive but shrink, the issue is saturation/nonlinearity;",
            "  if predicted gains are positive while actual gains flip sign, the frozen transport A is failing for that region.",
        ]
        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(report, encoding="utf-8")

        metadata = {
            "script": "eval_oracle_iterative_spatial_feedback_repair_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "spatial_states_npz": str(spatial_path),
            "transport_run_dir": str(transport_source),
            "source_layers": source_layers,
            "spatial_fit_ratio": a.spatial_fit_ratio,
            "spatial_fit_seed": a.spatial_fit_seed,
            "eval_scope": a.eval_scope,
            "eval_requested": len(eval_meta),
            "required_margins": margins,
            "step_natural": a.step_natural,
            "step_gain": a.step_gain,
            "max_steps": a.max_steps,
            "max_total_natural": a.max_total_natural,
            "stop_on": a.stop_on,
            "generation_check_every": a.generation_check_every,
            "generation_margin_escalation": a.generation_margin_escalation,
            "answer_surface": a.answer_surface,
            "sequence_score_reduction": a.sequence_score_reduction,
            "oracle_note": "GT chooses the desired relation only for baseline-wrong samples; spatial axes and A are frozen from calibration.",
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
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
