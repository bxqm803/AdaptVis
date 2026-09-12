#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spatial_to_decision_controller_v1.py

Learn a small, explicit spatial -> decision transport operator and use it as a
controller on held-out COCO samples.

Core goal
=========
We want an operational mapping, not merely a correlation:

    delta spatial state  --->  predicted delta decision scores

and, by inversion,

    desired decision direction  --->  required spatial-state edit.

At source layer L, fit two frozen REAL-NoImage object-relation axes on a labeled
calibration split:

    d_H : LEFT <-> RIGHT
    d_V : BELOW <-> ABOVE

Spatial coordinates are the projections of the REAL-NoImage relation state

    r_L = h_L(subject) - h_L(reference)

onto [d_H, d_V].  Because the axes need not be perfectly orthogonal, edits are
implemented through the dual basis so a requested coordinate change (dz_H,dz_V)
produces exactly that projection change.

Four-way decision operator
==========================
On a disjoint controller-calibration subset, use finite central differences to
estimate a 4x2 operator A_L:

                         [delta z_H]
    delta s_centered ~= A [delta z_V]

where s = [S_left,S_right,S_above,S_below] are the actual teacher-forced answer
sequence scores and centered means subtract the common score shift across the
four answers.

The corresponding 2x2 axis transport is also reported:

    M_H = S_right - S_left
    M_V = S_above - S_below

    [delta M_H]       [delta z_H]
    [delta M_V] ~= T  [delta z_V]

Two held-out controllers
========================
1) prototype
   Read the held-out sample's frozen REAL-NoImage spatial coordinate z(x) from
   --spatial-states-npz and migrate it toward the requested relation prototype:

       dz = strength * (mu_target - z(x)).

   Predict decision change with A dz, then perform the actual spatial edit and
   measure the real four-way decision change.

2) inverse
   Define the requested decision direction as a centered four-way target vector

       q_target = onehot(target) - 1/4.

   Solve

       dz = pinv(A) [decision_step * unit(q_target)]

   optionally cap the edit in natural spatial half-gap units, apply it, and
   compare predicted vs actual decision movement.

No GT relation is used to choose the requested target during evaluation: every
held-out sample is independently asked to move toward ALL four targets.  GT is
retained only for diagnostics.

Primary outputs
===============
transport_operator_4x2.csv
transport_matrix_2x2.csv
spatial_prototypes.csv
per_request.csv
controller_summary.csv
prediction_fidelity_summary.csv
analysis_summary.txt
metadata.json
errors.jsonl

Recommended exploratory run
===========================
CUDA_VISIBLE_DEVICES=0 python -u eval_spatial_to_decision_controller_v1.py \
  --model qwen-3b \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --source-layers 24,25 \
  --spatial-fit-ratio 0.30 \
  --transport-calibration-samples 40 \
  --transport-calibration-scale 0.5 \
  --eval-max-samples 80 \
  --controllers prototype,inverse \
  --prototype-strength 1.0 \
  --decision-step 0.5 \
  --inverse-max-natural 2.0 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --output-dir output/qwen3b_spatial_to_decision_controller_n80_v1 \
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import eval_real_causal_token_update_gating_v1 as gate
except Exception as exc:
    raise SystemExit(
        "Could not import eval_real_causal_token_update_gating_v1.py.\n"
        "Run this script from the AdaptVis repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12


# =============================================================================
# CLI / generic helpers
# =============================================================================

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
    p.add_argument("--source-layers", default="24,25")
    p.add_argument("--spatial-fit-ratio", type=float, default=0.30)
    p.add_argument("--spatial-fit-seed", type=int, default=1)
    p.add_argument(
        "--transport-calibration-samples",
        type=int,
        default=40,
        help="0 = all spatial-fit samples. Finite differences are estimated only here.",
    )
    p.add_argument(
        "--transport-calibration-scale",
        type=float,
        default=0.5,
        help="Finite-difference probe amplitude in natural half-gap units.",
    )
    p.add_argument(
        "--eval-scope", default="heldout", choices=["heldout", "all_data"]
    )
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--controllers",
        default="prototype,inverse",
        help="Comma subset of prototype,inverse.",
    )
    p.add_argument("--prototype-strength", type=float, default=1.0)
    p.add_argument(
        "--decision-step",
        type=float,
        default=0.5,
        help="Requested centered four-way decision displacement norm for inverse control.",
    )
    p.add_argument(
        "--inverse-max-natural",
        type=float,
        default=2.0,
        help=(
            "Cap inverse-controller spatial coordinate displacement by this L2 norm "
            "after dividing H/V by their natural half-gaps; <=0 disables cap."
        ),
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Small ridge for inverse decision->spatial solve.",
    )

    p.add_argument(
        "--answer-surface",
        default="above_below",
        choices=["above_below", "on_under"],
    )
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument(
        "--sequence-score-reduction", default="mean", choices=["mean", "sum"]
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    a = p.parse_args()
    if not (0.0 < a.spatial_fit_ratio < 1.0):
        p.error("--spatial-fit-ratio must be in (0,1)")
    if a.transport_calibration_scale <= 0:
        p.error("--transport-calibration-scale must be >0")
    if a.prototype_strength <= 0:
        p.error("--prototype-strength must be >0")
    if a.decision_step <= 0:
        p.error("--decision-step must be >0")
    return a


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


def parse_controllers(s: str) -> List[str]:
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    allowed = {"prototype", "inverse"}
    bad = sorted(set(vals) - allowed)
    if bad or not vals:
        raise ValueError(f"Bad --controllers {bad}; allowed={sorted(allowed)}")
    return list(dict.fromkeys(vals))


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


def center4(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(4)
    return v - float(v.mean())


def score_array(scores: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(scores[r]) for r in REL], dtype=np.float64)


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def stratified_cap(rows: Sequence[dict], n: int, seed: int) -> List[dict]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    return list(gate.traj.stratified_cap(rows, int(n), int(seed)))


# =============================================================================
# Spatial NPZ / split / 2D coordinate system
# =============================================================================

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
    axis_rows, proto_rows = [], []
    for L in source_layers:
        li = layer_to_i[L]
        Xf = X[fit_idx, li].astype(np.float64)
        yf = y[fit_idx]
        center = Xf.mean(axis=0)

        means = {r: Xf[yf == r].mean(axis=0) for r in REL}
        class_dirs = {r: unit(means[r] - center).astype(np.float64) for r in REL}
        dH = unit(class_dirs["right"] - class_dirs["left"]).astype(np.float64)
        dV = unit(class_dirs["above"] - class_dirs["below"]).astype(np.float64)

        # Orient axes semantically.
        gapH = float(np.dot(means["right"] - means["left"], dH))
        gapV = float(np.dot(means["above"] - means["below"], dV))
        if gapH < 0:
            dH = -dH
            gapH = -gapH
        if gapV < 0:
            dV = -dV
            gapV = -gapV

        B = np.stack([dH, dV], axis=1)  # D x 2
        gram = B.T @ B
        gram_inv = np.linalg.inv(gram)
        # Dual realization: delta_x = B (B^T B)^-1 delta_z, so B^T delta_x = delta_z exactly.
        dual = B @ gram_inv

        def coord(v):
            return B.T @ (np.asarray(v, dtype=np.float64) - center)

        prototypes = {r: coord(means[r]) for r in REL}
        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)

        geom[L] = {
            "layer_index": li,
            "center": center.astype(np.float32),
            "B": B.astype(np.float32),
            "dual": dual.astype(np.float32),
            "prototypes": {r: prototypes[r].astype(np.float32) for r in REL},
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
        for r in REL:
            proto_rows.append({
                "source_layer": L,
                "relation": r,
                "prototype_H": float(prototypes[r][0]),
                "prototype_V": float(prototypes[r][1]),
            })

    return geom, pd.DataFrame(axis_rows), pd.DataFrame(proto_rows)


def spatial_coord(geom_L, relation_vector):
    B = geom_L["B"].astype(np.float64)
    center = geom_L["center"].astype(np.float64)
    return (B.T @ (np.asarray(relation_vector, dtype=np.float64) - center)).astype(np.float64)


def realize_delta(geom_L, dz):
    dual = geom_L["dual"].astype(np.float64)
    dz = np.asarray(dz, dtype=np.float64).reshape(2)
    dx = dual @ dz
    realized = geom_L["B"].astype(np.float64).T @ dx
    if np.max(np.abs(realized - dz)) > 1e-4 * max(1.0, float(np.linalg.norm(dz))):
        raise RuntimeError(f"Dual-basis realization mismatch requested={dz}, realized={realized}")
    return dx.astype(np.float32)


# =============================================================================
# Dataset / model / scoring / patching
# =============================================================================

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


def argmax_relation(scores):
    return max(REL, key=lambda r: float(scores[r]))


def make_relation_patch(L, sub_pos, ref_pos, delta_relation):
    d = np.asarray(delta_relation, dtype=np.float32)
    return {
        int(L): {
            int(sub_pos): (+0.5 * d).astype(np.float32),
            int(ref_pos): (-0.5 * d).astype(np.float32),
        }
    }


def target_margin(scores: Dict[str, float], target: str) -> float:
    t = float(scores[target])
    other = max(float(scores[r]) for r in REL if r != target)
    return t - other


def decision_axis_delta(delta_s):
    d = np.asarray(delta_s, dtype=np.float64).reshape(4)
    return np.asarray([
        d[RID["right"]] - d[RID["left"]],
        d[RID["above"]] - d[RID["below"]],
    ], dtype=np.float64)


def target_oriented_axis(delta_m, target: str):
    if target == "right":
        return float(delta_m[0]), float(delta_m[1])
    if target == "left":
        return float(-delta_m[0]), float(delta_m[1])
    if target == "above":
        return float(delta_m[1]), float(delta_m[0])
    if target == "below":
        return float(-delta_m[1]), float(delta_m[0])
    raise KeyError(target)


# =============================================================================
# Calibrate empirical 4x2 operator A on calibration samples
# =============================================================================

def calibrate_transport(
    *, a, meta, rec_by_sid, spatial_by_sid, geom, source_layers,
    model, processor, decoder_layers, two, candidate_ids, device, err_path
):
    rows = []
    scale = float(a.transport_calibration_scale)

    for m in tqdm(meta, desc="CALIBRATE 4x2 transport"):
        sid = int(m["sid"])
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

            for L in source_layers:
                g = geom[L]
                amp = {
                    "H": scale * g["natural_half_H"],
                    "V": scale * g["natural_half_V"],
                }
                score_pm = {}
                for axis_i, axis_name in enumerate(("H", "V")):
                    for sign in (+1, -1):
                        dz = np.zeros(2, dtype=np.float64)
                        dz[axis_i] = sign * amp[axis_name]
                        dx = realize_delta(g, dz)
                        patch = make_relation_patch(L, sub_pos, ref_pos, dx)
                        sc = all_scores(
                            model, batch, candidate_ids, a.sequence_score_reduction,
                            decoder_layers=decoder_layers, patch_map=patch,
                        )
                        score_pm[(axis_name, sign)] = center4(score_array(sc))

                colH = (score_pm[("H", +1)] - score_pm[("H", -1)]) / (2.0 * amp["H"])
                colV = (score_pm[("V", +1)] - score_pm[("V", -1)]) / (2.0 * amp["V"])
                A = np.stack([colH, colV], axis=1)  # 4 x 2
                T = np.asarray([
                    A[RID["right"]] - A[RID["left"]],
                    A[RID["above"]] - A[RID["below"]],
                ])
                row = {
                    "sid": sid,
                    "gt": m["gt"],
                    "source_layer": L,
                    "calibration_scale": scale,
                    "amp_H": amp["H"],
                    "amp_V": amp["V"],
                    "T_HH": T[0, 0], "T_HV": T[0, 1],
                    "T_VH": T[1, 0], "T_VV": T[1, 1],
                }
                for ri, r in enumerate(REL):
                    row[f"A_{r}_H"] = A[ri, 0]
                    row[f"A_{r}_V"] = A[ri, 1]
                rows.append(row)

        except Exception as exc:
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "stage": "transport_calibration",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-14:],
                }, ensure_ascii=False) + "\n")
        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            batch = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No successful transport calibration samples")

    per = pd.DataFrame(rows)
    operators, op_rows, t_rows = {}, [], []
    for L in source_layers:
        q = per[per.source_layer == L]
        if len(q) == 0:
            raise RuntimeError(f"No transport calibration rows for L{L}")
        A = np.zeros((4, 2), dtype=np.float64)
        for ri, r in enumerate(REL):
            A[ri, 0] = float(q[f"A_{r}_H"].mean())
            A[ri, 1] = float(q[f"A_{r}_V"].mean())
        # Numerical centering protects against tiny common-mode drift.
        A = A - A.mean(axis=0, keepdims=True)
        T = np.asarray([
            A[RID["right"]] - A[RID["left"]],
            A[RID["above"]] - A[RID["below"]],
        ], dtype=np.float64)
        svals = np.linalg.svd(A, compute_uv=False)
        operators[L] = {"A": A, "T": T}
        for ri, r in enumerate(REL):
            op_rows.append({
                "source_layer": L,
                "relation": r,
                "dscore_dzH": A[ri, 0],
                "dscore_dzV": A[ri, 1],
            })
        t_rows.append({
            "source_layer": L,
            "N_calibration": len(q),
            "T_HH": T[0, 0], "T_HV": T[0, 1],
            "T_VH": T[1, 0], "T_VV": T[1, 1],
            "diag_abs_mean": 0.5 * (abs(T[0,0]) + abs(T[1,1])),
            "cross_abs_mean": 0.5 * (abs(T[0,1]) + abs(T[1,0])),
            "diag_to_cross": (
                0.5 * (abs(T[0,0]) + abs(T[1,1]))
                / max(0.5 * (abs(T[0,1]) + abs(T[1,0])), EPS)
            ),
            "A_singular_value_1": float(svals[0]) if len(svals) > 0 else np.nan,
            "A_singular_value_2": float(svals[1]) if len(svals) > 1 else np.nan,
            "A_condition_number": float(svals[0] / max(svals[-1], EPS)),
        })
    return operators, per, pd.DataFrame(op_rows), pd.DataFrame(t_rows)


# =============================================================================
# Controllers
# =============================================================================

def inverse_solve(A, target: str, decision_step: float, ridge: float):
    q = np.full(4, -0.25, dtype=np.float64)
    q[RID[target]] = 0.75
    q = q / max(float(np.linalg.norm(q)), EPS)
    desired = float(decision_step) * q
    ATA = A.T @ A + float(ridge) * np.eye(2)
    dz = np.linalg.solve(ATA, A.T @ desired)
    return dz.astype(np.float64), desired.astype(np.float64)


def cap_inverse_dz(dz, g, max_natural: float):
    dz = np.asarray(dz, dtype=np.float64).copy()
    if max_natural <= 0:
        return dz, 1.0
    normed = np.asarray([
        dz[0] / max(float(g["natural_half_H"]), EPS),
        dz[1] / max(float(g["natural_half_V"]), EPS),
    ])
    n = float(np.linalg.norm(normed))
    if n <= max_natural or n < EPS:
        return dz, 1.0
    fac = float(max_natural / n)
    return dz * fac, fac


# =============================================================================
# Summaries
# =============================================================================

def summarize_requests(df: pd.DataFrame):
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    gcols = ["controller", "source_layer", "target_relation"]
    for (controller, L, target), g in df.groupby(gcols, dropna=False):
        base_not_target = g[g.baseline_prediction != target]
        base_target = g[g.baseline_prediction == target]
        rows.append({
            "controller": controller,
            "source_layer": int(L),
            "target_relation": target,
            "N": len(g),
            "mean_spatial_dz_H": float(g.spatial_dz_H.mean()),
            "mean_spatial_dz_V": float(g.spatial_dz_V.mean()),
            "mean_spatial_delta_norm": float(g.spatial_delta_norm.mean()),
            "mean_pred_target_axis_delta": float(g.pred_target_axis_delta.mean()),
            "mean_actual_target_axis_delta": float(g.actual_target_axis_delta.mean()),
            "fraction_actual_moves_target_way": float((g.actual_target_axis_delta > 0).mean()),
            "mean_abs_actual_cross_axis_delta": float(g.actual_cross_axis_delta.abs().mean()),
            "mean_actual_target_minus_abs_cross": float(
                (g.actual_target_axis_delta - g.actual_cross_axis_delta.abs()).mean()
            ),
            "mean_delta_target_vs_bestother_margin": float(g.delta_target_margin.mean()),
            "fraction_target_margin_improves": float((g.delta_target_margin > 0).mean()),
            "patched_target_prediction_rate": float((g.patched_prediction == target).mean()),
            "conversion_to_target_rate_if_not_target": (
                float((base_not_target.patched_prediction == target).mean())
                if len(base_not_target) else np.nan
            ),
            "preserve_target_rate_if_already_target": (
                float((base_target.patched_prediction == target).mean())
                if len(base_target) else np.nan
            ),
            "pred_vs_actual_target_axis_corr": safe_corr(
                g.pred_target_axis_delta, g.actual_target_axis_delta
            ),
            "pred_vs_actual_target_margin_corr": safe_corr(
                g.pred_target_margin_delta, g.delta_target_margin
            ),
            "mean_pred_actual_decision_cos": float(g.pred_actual_decision_cos.mean()),
        })
    return pd.DataFrame(rows).sort_values(gcols).reset_index(drop=True)


def fidelity_summary(df: pd.DataFrame):
    rows = []
    for (controller, L), g in df.groupby(["controller", "source_layer"]):
        pred = np.stack(g.pred_centered_delta.apply(lambda x: np.asarray(json.loads(x), dtype=float)))
        actual = np.stack(g.actual_centered_delta.apply(lambda x: np.asarray(json.loads(x), dtype=float)))
        pred_axis = np.stack(g.pred_axis_delta.apply(lambda x: np.asarray(json.loads(x), dtype=float)))
        actual_axis = np.stack(g.actual_axis_delta.apply(lambda x: np.asarray(json.loads(x), dtype=float)))
        rows.append({
            "controller": controller,
            "source_layer": int(L),
            "N_requests": len(g),
            "corr_centered_scores_flat": safe_corr(pred.reshape(-1), actual.reshape(-1)),
            "corr_axis_H": safe_corr(pred_axis[:,0], actual_axis[:,0]),
            "corr_axis_V": safe_corr(pred_axis[:,1], actual_axis[:,1]),
            "corr_axis_flat": safe_corr(pred_axis.reshape(-1), actual_axis.reshape(-1)),
            "mean_centered_score_mae": float(np.mean(np.abs(pred - actual))),
            "mean_axis_mae": float(np.mean(np.abs(pred_axis - actual_axis))),
            "mean_decision_cos": float(np.nanmean([
                cosine(p, q) for p, q in zip(pred, actual)
            ])),
            "fraction_target_way": float((g.actual_target_axis_delta > 0).mean()),
            "mean_target_axis_delta": float(g.actual_target_axis_delta.mean()),
            "mean_abs_cross_axis_delta": float(g.actual_cross_axis_delta.abs().mean()),
            "target_prediction_rate": float((g.patched_prediction == g.target_relation).mean()),
        })
    return pd.DataFrame(rows).sort_values(["controller", "source_layer"]).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_layers(a.source_layers)
    controllers = parse_controllers(a.controllers)
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
    geom, axis_df, proto_df = fit_spatial_geometry(
        X, y, spatial_layers, fit_idx, source_layers
    )
    axis_df.to_csv(outdir / "spatial_axis_geometry.csv", index=False)
    proto_df.to_csv(outdir / "spatial_prototypes.csv", index=False)

    sid_to_npz_i = {int(s): i for i, s in enumerate(spatial_sids.tolist())}
    spatial_by_sid = {int(s): X[i] for i, s in enumerate(spatial_sids.tolist())}

    two, all_meta, rec_by_sid = load_all_data(a)
    spatial_sid_set = set(sid_to_npz_i)
    all_meta = [m for m in all_meta if int(m["sid"]) in spatial_sid_set]
    calib_meta = [m for m in all_meta if int(m["sid"]) in fit_sids]
    calib_meta = stratified_cap(calib_meta, a.transport_calibration_samples, a.seed + 101)

    eval_meta = all_meta
    if a.eval_scope == "heldout":
        eval_meta = [m for m in eval_meta if int(m["sid"]) in held_sids]
    eval_meta = stratified_cap(eval_meta, a.eval_max_samples, a.seed)
    if not calib_meta:
        raise RuntimeError("No transport calibration samples")
    if not eval_meta:
        raise RuntimeError("No evaluation samples")

    model = processor = None
    request_rows = []
    baseline_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        device = torch.device(a.device)
        if max(source_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(source_layers)} but model has {len(decoder_layers)} layers"
            )
        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)

        print("\n" + "=" * 168)
        print("SPATIAL -> DECISION CONTROLLER")
        print("=" * 168)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source layers={source_layers}")
        print(f"spatial states={spatial_path}")
        print(f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}")
        print(f"transport calibration N={len(calib_meta)} scale={a.transport_calibration_scale}")
        print(f"evaluation N={len(eval_meta)} scope={a.eval_scope}")
        print(f"controllers={controllers}")
        print(f"decoder={decoder_path}")
        print("=" * 168, flush=True)

        operators, transport_per, op_df, t_df = calibrate_transport(
            a=a,
            meta=calib_meta,
            rec_by_sid=rec_by_sid,
            spatial_by_sid=spatial_by_sid,
            geom=geom,
            source_layers=source_layers,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            two=two,
            candidate_ids=candidate_ids,
            device=device,
            err_path=err_path,
        )
        transport_per.to_csv(outdir / "transport_calibration_per_sample.csv", index=False)
        op_df.to_csv(outdir / "transport_operator_4x2.csv", index=False)
        t_df.to_csv(outdir / "transport_matrix_2x2.csv", index=False)

        for m in tqdm(eval_meta, desc="EVAL spatial controller"):
            sid = int(m["sid"])
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
                base_scores = all_scores(
                    model, batch, candidate_ids, a.sequence_score_reduction
                )
                base_arr = score_array(base_scores)
                base_center = center4(base_arr)
                base_pred = argmax_relation(base_scores)
                baseline_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "baseline_prediction": base_pred,
                    "baseline_correct": base_pred == m["gt"],
                    **{f"baseline_score_{r}": base_scores[r] for r in REL},
                })

                npz_i = sid_to_npz_i[sid]
                for L in source_layers:
                    g = geom[L]
                    li = g["layer_index"]
                    current_vec = X[npz_i, li]
                    z = spatial_coord(g, current_vec)
                    A = operators[L]["A"]
                    T = operators[L]["T"]

                    for target in REL:
                        jobs = []
                        if "prototype" in controllers:
                            target_z = g["prototypes"][target].astype(np.float64)
                            dz = float(a.prototype_strength) * (target_z - z)
                            jobs.append(("prototype", dz, 1.0, None))

                        if "inverse" in controllers:
                            dz0, desired = inverse_solve(
                                A, target, a.decision_step, a.ridge
                            )
                            dz, cap_factor = cap_inverse_dz(
                                dz0, g, a.inverse_max_natural
                            )
                            jobs.append(("inverse", dz, cap_factor, desired))

                        for controller, dz, cap_factor, desired in jobs:
                            dx = realize_delta(g, dz)
                            pred_center = A @ dz
                            pred_axis = T @ dz
                            patch = make_relation_patch(L, sub_pos, ref_pos, dx)
                            patched_scores = all_scores(
                                model, batch, candidate_ids, a.sequence_score_reduction,
                                decoder_layers=decoder_layers, patch_map=patch,
                            )
                            patch_arr = score_array(patched_scores)
                            actual_center = center4(patch_arr - base_arr)
                            actual_axis = decision_axis_delta(patch_arr - base_arr)
                            pred_target_axis, pred_cross = target_oriented_axis(pred_axis, target)
                            actual_target_axis, actual_cross = target_oriented_axis(actual_axis, target)
                            base_tmargin = target_margin(base_scores, target)
                            patch_tmargin = target_margin(patched_scores, target)
                            # Predicted target-vs-best-other delta using the baseline strongest competitor.
                            comp = max((r for r in REL if r != target), key=lambda r: base_scores[r])
                            pred_tmargin_delta = float(
                                pred_center[RID[target]] - pred_center[RID[comp]]
                            )

                            request_rows.append({
                                "sid": sid,
                                "gt": m["gt"],
                                "source_layer": L,
                                "controller": controller,
                                "target_relation": target,
                                "subject": m["subject"],
                                "reference": m["reference"],
                                "subject_position": sub_pos,
                                "reference_position": ref_pos,
                                "spatial_z_H": float(z[0]),
                                "spatial_z_V": float(z[1]),
                                "target_proto_H": float(g["prototypes"][target][0]),
                                "target_proto_V": float(g["prototypes"][target][1]),
                                "spatial_dz_H": float(dz[0]),
                                "spatial_dz_V": float(dz[1]),
                                "spatial_delta_norm": float(np.linalg.norm(dx)),
                                "spatial_coordinate_delta_norm": float(np.linalg.norm(dz)),
                                "inverse_cap_factor": float(cap_factor),
                                "baseline_prediction": base_pred,
                                "patched_prediction": argmax_relation(patched_scores),
                                "baseline_correct": base_pred == m["gt"],
                                "patched_correct": argmax_relation(patched_scores) == m["gt"],
                                "pred_centered_delta": json.dumps(pred_center.tolist()),
                                "actual_centered_delta": json.dumps(actual_center.tolist()),
                                "pred_axis_delta": json.dumps(pred_axis.tolist()),
                                "actual_axis_delta": json.dumps(actual_axis.tolist()),
                                "pred_delta_H": float(pred_axis[0]),
                                "pred_delta_V": float(pred_axis[1]),
                                "actual_delta_H": float(actual_axis[0]),
                                "actual_delta_V": float(actual_axis[1]),
                                "pred_target_axis_delta": float(pred_target_axis),
                                "actual_target_axis_delta": float(actual_target_axis),
                                "pred_cross_axis_delta": float(pred_cross),
                                "actual_cross_axis_delta": float(actual_cross),
                                "baseline_target_margin": float(base_tmargin),
                                "patched_target_margin": float(patch_tmargin),
                                "delta_target_margin": float(patch_tmargin - base_tmargin),
                                "pred_target_margin_delta": float(pred_tmargin_delta),
                                "pred_actual_decision_cos": cosine(pred_center, actual_center),
                                "baseline_strongest_nontarget": comp,
                                **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                                **{f"patched_score_{r}": float(patched_scores[r]) for r in REL},
                            })

            except Exception as exc:
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "stage": "evaluation",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-14:],
                    }, ensure_ascii=False) + "\n")
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                batch = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not request_rows:
            raise RuntimeError(f"No successful controller requests. See {err_path}")

        req_df = pd.DataFrame(request_rows)
        base_df = pd.DataFrame(baseline_rows).drop_duplicates("sid")
        req_df.to_csv(outdir / "per_request.csv", index=False)
        base_df.to_csv(outdir / "baseline_scores.csv", index=False)

        summary_df = summarize_requests(req_df)
        summary_df.to_csv(outdir / "controller_summary.csv", index=False)
        fidelity_df = fidelity_summary(req_df)
        fidelity_df.to_csv(outdir / "prediction_fidelity_summary.csv", index=False)

        baseline_acc = float(base_df.baseline_correct.mean()) if len(base_df) else np.nan
        lines = []
        lines.append("=" * 168)
        lines.append("SPATIAL -> DECISION CONTROLLER")
        lines.append("=" * 168)
        lines.append(f"model={a.model} repo={spec.repo_id}")
        lines.append(f"N successful eval={base_df.sid.nunique()} / requested={len(eval_meta)}")
        lines.append(f"source layers={source_layers}")
        lines.append(f"spatial states={spatial_path}")
        lines.append(f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}")
        lines.append(f"transport calibration N requested={len(calib_meta)}")
        lines.append(f"controllers={controllers}")
        lines.append(f"teacher-forced baseline argmax accuracy={baseline_acc:.4f}")
        lines.append("")
        lines.append("CALIBRATED 2x2 AXIS TRANSPORT")
        lines.append("-" * 168)
        lines.append(t_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("CONTROLLER SUMMARY")
        lines.append("-" * 168)
        lines.append(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("PREDICTED -> ACTUAL DECISION FIDELITY")
        lines.append("-" * 168)
        lines.append(fidelity_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  prototype: move each sample's measured RN spatial coordinate toward a requested relation prototype.")
        lines.append("  inverse: invert the frozen 4x2 transport operator to ask for a requested four-way decision direction.")
        lines.append("  A useful controller requires BOTH control and prediction:")
        lines.append("    (1) requested target-axis movement is positive on held-out samples;")
        lines.append("    (2) target score-vs-best-other margin improves;")
        lines.append("    (3) A*dz predicts the actual four-way/axis score displacement;")
        lines.append("    (4) cross-axis movement stays smaller than target-axis movement.")
        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(report, encoding="utf-8")

        metadata = {
            "script": "eval_spatial_to_decision_controller_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "spatial_states_npz": str(spatial_path),
            "source_layers": source_layers,
            "spatial_fit_ratio": a.spatial_fit_ratio,
            "spatial_fit_seed": a.spatial_fit_seed,
            "transport_calibration_samples": len(calib_meta),
            "transport_calibration_scale": a.transport_calibration_scale,
            "eval_scope": a.eval_scope,
            "eval_requested": len(eval_meta),
            "controllers": controllers,
            "prototype_strength": a.prototype_strength,
            "decision_step": a.decision_step,
            "inverse_max_natural": a.inverse_max_natural,
            "ridge": a.ridge,
            "answer_surface": a.answer_surface,
            "sequence_score_reduction": a.sequence_score_reduction,
            "relations": list(REL),
            "note": (
                "All four requested targets are evaluated for every held-out sample. "
                "GT is not used to choose target. The transport operator and spatial "
                "prototypes are frozen from the calibration split."
            ),
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
