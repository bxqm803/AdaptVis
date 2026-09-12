#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_oracle_adaptive_spatial_repair_v2_standalone.py

Oracle adaptive spatial repair using a frozen spatial -> decision transport map.

Question
========
Given:
  (1) the GT spatial relation (oracle target),
  (2) the sample's current four-way decision scores,
  (3) a transport operator A learned only on a calibration split,

can we repair an actually wrong generation by changing ONLY the mid-layer
subject-reference spatial state?

At source layer L, the frozen operator maps a 2-D spatial-coordinate edit to
centered four-way sequence-score change:

    delta s ~= A_L delta z,

where delta z = [delta z_H, delta z_V].

For a baseline-wrong sample with GT target t, choose the minimum-natural-norm
spatial edit such that the linearized decision predicts that GT beats every
competitor j by at least gamma:

    (S_t - S_j) + (A_t - A_j) delta z >= gamma,   for all j != t.

The optimization is only two-dimensional.  We solve it exactly by enumerating
the active constraints of the convex minimum-norm half-space problem.  The norm
is measured in natural spatial half-gap units, so horizontal and vertical edits
are comparably scaled.

Important experimental protocol
================================
* Spatial axes/prototypes: fit on --spatial-fit-ratio calibration split.
* A_L: finite-difference calibrated on a subset of that calibration split.
* Evaluation: held-out by default.
* Oracle GT is used ONLY to specify the desired target relation on baseline-wrong
  evaluation samples.
* Baseline-correct samples are NEVER edited.  Therefore C->W should be exactly 0
  unless there is an implementation/accounting error.
* Primary metric is ACTUAL free-generation accuracy after patching, not merely
  teacher-forced argmax accuracy.

Outputs
=======
transport_operator_4x2.csv
transport_matrix_2x2.csv
spatial_axis_geometry.csv
spatial_prototypes.csv
baseline.csv
per_repair.csv
generation_summary.csv
repair_by_relation.csv
analysis_summary.txt
metadata.json
errors.jsonl

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_oracle_adaptive_spatial_repair_v2_standalone.py \
  --model qwen-3b \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --source-layers 25 \
  --spatial-fit-ratio 0.30 \
  --transport-run-dir output/qwen3b_spatial_to_decision_controller_L25_n80_v1 \
  --eval-scope heldout \
  --eval-max-samples 80 \
  --required-margins 0,0.1,0.25,0.5 \
  --max-natural 2.0 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --max-new-tokens 6 \
  --output-dir output/qwen3b_oracle_adaptive_spatial_repair_L25_n80_v1 \
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
        "Run from the AdaptVis repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12

# Standalone helpers copied from eval_spatial_to_decision_controller_v1.py
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
# CLI
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
    p.add_argument("--source-layers", default="25")
    p.add_argument(
        "--transport-run-dir",
        default="",
        help=(
            "Optional completed eval_spatial_to_decision_controller_v1.py output dir. "
            "If provided, load its frozen transport_operator_4x2.csv instead of recalibrating A."
        ),
    )

    p.add_argument("--spatial-fit-ratio", type=float, default=0.30)
    p.add_argument("--spatial-fit-seed", type=int, default=1)
    p.add_argument(
        "--transport-calibration-samples",
        type=int,
        default=40,
        help="0 = all spatial-fit samples.",
    )
    p.add_argument(
        "--transport-calibration-scale",
        type=float,
        default=0.5,
        help="Finite-difference amplitude in natural half-gap units.",
    )

    p.add_argument("--eval-scope", default="heldout", choices=["heldout", "all_data"])
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--required-margins",
        default="0,0.1,0.25,0.5",
        help="Comma-separated desired GT-vs-every-competitor score margins.",
    )
    p.add_argument(
        "--max-natural",
        type=float,
        default=2.0,
        help=(
            "Maximum edit norm in natural H/V half-gap coordinates. "
            "<=0 disables the cap. If the minimum predicted repair exceeds the cap, "
            "the edit is scaled to the cap and still evaluated."
        ),
    )
    p.add_argument(
        "--qp-tol",
        type=float,
        default=1e-8,
        help="Feasibility tolerance for the 2-D minimum-norm half-space solver.",
    )

    p.add_argument(
        "--answer-surface", default="above_below", choices=["above_below", "on_under"]
    )
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument(
        "--sequence-score-reduction", default="mean", choices=["mean", "sum"]
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
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
    return a


def parse_float_list(s: str) -> List[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError(f"No floats parsed from {s!r}")
    if any(v < 0 for v in vals):
        raise ValueError("Required margins must be >= 0")
    return sorted(set(vals))




def load_frozen_transport(run_dir: Path, source_layers: Sequence[int]):
    op_path = run_dir / "transport_operator_4x2.csv"
    if not op_path.exists():
        raise FileNotFoundError(op_path)
    op_df = pd.read_csv(op_path)
    required = {"source_layer", "relation", "dscore_dzH", "dscore_dzV"}
    missing = required - set(op_df.columns)
    if missing:
        raise RuntimeError(f"{op_path} missing columns {sorted(missing)}")

    operators = {}
    t_rows = []
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
        t_rows.append({
            "source_layer": L,
            "N_calibration": np.nan,
            "T_HH": T[0,0], "T_HV": T[0,1],
            "T_VH": T[1,0], "T_VV": T[1,1],
            "diag_abs_mean": 0.5 * (abs(T[0,0]) + abs(T[1,1])),
            "cross_abs_mean": 0.5 * (abs(T[0,1]) + abs(T[1,0])),
            "diag_to_cross": (
                0.5 * (abs(T[0,0]) + abs(T[1,1]))
                / max(0.5 * (abs(T[0,1]) + abs(T[1,0])), EPS)
            ),
            "A_singular_value_1": float(svals[0]),
            "A_singular_value_2": float(svals[1]),
            "A_condition_number": float(svals[0] / max(svals[-1], EPS)),
        })
    return operators, op_df, pd.DataFrame(t_rows)

# =============================================================================
# Exact 2-D minimum-natural-norm controller
# =============================================================================

def _feasible(C: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float) -> bool:
    return bool(np.all(C @ x >= b - tol))


def solve_min_norm_halfspaces_2d(C: np.ndarray, b: np.ndarray, tol: float = 1e-8):
    """
    Solve
        min ||x||_2
        s.t. C x >= b
    for x in R^2 by enumerating possible active sets (0, 1, or 2 constraints).

    Returns (x, feasible, active_tuple).  If infeasible, x is [nan,nan].
    """
    C = np.asarray(C, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if C.ndim != 2 or C.shape[1] != 2 or C.shape[0] != len(b):
        raise ValueError(f"Bad halfspace shapes C={C.shape}, b={b.shape}")

    candidates = []
    x0 = np.zeros(2, dtype=np.float64)
    if _feasible(C, b, x0, tol):
        candidates.append((float(np.dot(x0, x0)), x0, ()))

    # One active boundary: Euclidean projection of the origin to c_i x = b_i.
    for i in range(len(b)):
        c = C[i]
        denom = float(np.dot(c, c))
        if denom < EPS:
            continue
        x = (float(b[i]) / denom) * c
        if _feasible(C, b, x, tol):
            candidates.append((float(np.dot(x, x)), x, (i,)))

    # Two active boundaries: their intersection.
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


def oracle_minimum_spatial_edit(
    *, base_scores: Dict[str, float], target: str, A: np.ndarray,
    geom_L: dict, required_margin: float, max_natural: float, tol: float
):
    """
    Work in normalized natural coordinates x:
        dz = diag(half_H, half_V) x.
    Minimize ||x|| subject to the predicted GT margin constraints.
    """
    half = np.asarray([
        float(geom_L["natural_half_H"]),
        float(geom_L["natural_half_V"]),
    ], dtype=np.float64)
    D = np.diag(half)
    s = score_array(base_scores)
    t = RID[target]

    competitors = [r for r in REL if r != target]
    C_rows, b_rows = [], []
    for r in competitors:
        j = RID[r]
        # (s_t-s_j) + (A_t-A_j) D x >= gamma
        C_rows.append((A[t] - A[j]) @ D)
        b_rows.append(float(required_margin) - float(s[t] - s[j]))

    C = np.asarray(C_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    x_req, feasible, active = solve_min_norm_halfspaces_2d(C, b, tol=tol)

    if not feasible:
        return {
            "solver_feasible": False,
            "x_required": np.full(2, np.nan),
            "x_applied": np.zeros(2),
            "dz_required": np.full(2, np.nan),
            "dz_applied": np.zeros(2),
            "required_natural_norm": np.nan,
            "applied_natural_norm": 0.0,
            "cap_factor": 0.0,
            "predicted_constraints_satisfied_after_cap": False,
            "active_constraints": (),
            "competitors": competitors,
            "constraint_rhs": b,
        }

    req_norm = float(np.linalg.norm(x_req))
    x_apply = x_req.copy()
    cap_factor = 1.0
    if max_natural > 0 and req_norm > max_natural and req_norm > EPS:
        cap_factor = float(max_natural / req_norm)
        x_apply *= cap_factor

    dz_req = D @ x_req
    dz_apply = D @ x_apply
    pred_ok = _feasible(C, b, x_apply, tol)

    return {
        "solver_feasible": True,
        "x_required": x_req,
        "x_applied": x_apply,
        "dz_required": dz_req,
        "dz_applied": dz_apply,
        "required_natural_norm": req_norm,
        "applied_natural_norm": float(np.linalg.norm(x_apply)),
        "cap_factor": cap_factor,
        "predicted_constraints_satisfied_after_cap": bool(pred_ok),
        "active_constraints": active,
        "competitors": competitors,
        "constraint_rhs": b,
    }


def min_gt_margin(scores: Dict[str, float], target: str) -> float:
    return min(float(scores[target]) - float(scores[r]) for r in REL if r != target)


# =============================================================================
# Summaries
# =============================================================================

def generation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source_layer", "required_margin"]
    for (L, margin), g in df.groupby(keys, dropna=False):
        base_correct = g.baseline_correct.astype(bool).to_numpy()
        patch_correct = g.patched_correct.astype(bool).to_numpy()
        w2c = int(np.sum((~base_correct) & patch_correct))
        c2w = int(np.sum(base_correct & (~patch_correct)))
        wrong_n = int(np.sum(~base_correct))
        correct_n = int(np.sum(base_correct))
        edited = g[g.edit_applied.astype(bool)]
        wrong = g[~g.baseline_correct.astype(bool)]
        rows.append({
            "source_layer": int(L),
            "required_margin": float(margin),
            "N": len(g),
            "baseline_accuracy": float(np.mean(base_correct)),
            "patched_accuracy": float(np.mean(patch_correct)),
            "gain": float(np.mean(patch_correct) - np.mean(base_correct)),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "baseline_wrong_N": wrong_n,
            "repair_rate_on_wrong": float(w2c / wrong_n) if wrong_n else np.nan,
            "preserve_rate_on_correct": float(1.0 - c2w / correct_n) if correct_n else np.nan,
            "edited_N": len(edited),
            "solver_feasible_rate_on_wrong": float(wrong.solver_feasible.mean()) if len(wrong) else np.nan,
            "predicted_feasible_after_cap_rate_on_wrong": float(
                wrong.predicted_constraints_satisfied_after_cap.mean()
            ) if len(wrong) else np.nan,
            "capped_rate_on_wrong": (
                float((wrong.loc[wrong.solver_feasible.astype(bool), "cap_factor"] < 0.999999).mean())
                if bool(wrong.solver_feasible.astype(bool).any()) else np.nan
            ),
            "mean_required_natural_norm_wrong": float(wrong.required_natural_norm.mean()) if len(wrong) else np.nan,
            "mean_applied_natural_norm_wrong": float(wrong.applied_natural_norm.mean()) if len(wrong) else np.nan,
            "tf_target_after_rate_on_wrong": float((wrong.patched_tf_prediction == wrong.gt).mean()) if len(wrong) else np.nan,
            "mean_actual_gt_min_margin_after_wrong": float(wrong.patched_gt_min_margin.mean()) if len(wrong) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def relation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (L, margin, rel), g in df.groupby(["source_layer", "required_margin", "gt"]):
        base = g.baseline_correct.astype(bool).to_numpy()
        patch = g.patched_correct.astype(bool).to_numpy()
        wrong = ~base
        w2c = int(np.sum(wrong & patch))
        rows.append({
            "source_layer": int(L),
            "required_margin": float(margin),
            "relation": rel,
            "N": len(g),
            "baseline_accuracy": float(np.mean(base)),
            "patched_accuracy": float(np.mean(patch)),
            "gain": float(np.mean(patch) - np.mean(base)),
            "wrong_to_correct": w2c,
            "baseline_wrong_N": int(np.sum(wrong)),
            "repair_rate_on_wrong": float(w2c / max(int(np.sum(wrong)), 1)),
        })
    return pd.DataFrame(rows).sort_values(["source_layer", "required_margin", "relation"])


# =============================================================================
# Main
# =============================================================================

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
    calib_meta = stratified_cap(
        calib_meta, a.transport_calibration_samples, a.seed + 101
    )

    eval_meta = all_meta
    if a.eval_scope == "heldout":
        eval_meta = [m for m in eval_meta if int(m["sid"]) in held_sids]
    eval_meta = stratified_cap(eval_meta, a.eval_max_samples, a.seed)
    if not calib_meta:
        raise RuntimeError("No transport calibration samples")
    if not eval_meta:
        raise RuntimeError("No evaluation samples")

    model = processor = None
    baseline_rows = []
    repair_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        device = torch.device(a.device)
        if max(source_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(source_layers)} but model has {len(decoder_layers)} layers"
            )

        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)

        print("\n" + "=" * 176)
        print("ORACLE ADAPTIVE SPATIAL REPAIR")
        print("=" * 176)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source layers={source_layers}")
        print(f"spatial states={spatial_path}")
        print(f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}")
        if a.transport_run_dir:
            print(f"transport operator source={a.transport_run_dir}")
        else:
            print(f"transport calibration N={len(calib_meta)} scale={a.transport_calibration_scale}")
        print(f"evaluation N={len(eval_meta)} scope={a.eval_scope}")
        print(f"required margins={margins}; max natural norm={a.max_natural}")
        print("baseline-correct samples are never edited")
        print(f"decoder={decoder_path}")
        print("=" * 176, flush=True)

        if a.transport_run_dir:
            transport_source = Path(a.transport_run_dir)
            operators, op_df, t_df = load_frozen_transport(transport_source, source_layers)
            op_df.to_csv(outdir / "transport_operator_4x2.csv", index=False)
            t_df.to_csv(outdir / "transport_matrix_2x2.csv", index=False)
        else:
            transport_source = None
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

        for m in tqdm(eval_meta, desc="EVAL oracle spatial repair"):
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

                # Actual baseline generation: this defines correct/wrong and the main accuracy.
                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map={},
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = (base_gen_pred == gt)

                # Current decision state used by the adaptive controller.
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
                    "baseline_generation_correct": base_correct,
                    "baseline_generation_text": base_gen_text,
                    "baseline_tf_prediction": base_tf_pred,
                    "baseline_tf_correct": base_tf_pred == gt,
                    "baseline_gt_min_margin": base_gt_margin,
                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                })

                for L in source_layers:
                    g = geom[L]
                    A = operators[L]["A"]

                    for margin in margins:
                        # Correct generation is explicitly preserved: no edit, no second generate call.
                        if base_correct:
                            repair_rows.append({
                                "sid": sid,
                                "gt": gt,
                                "source_layer": L,
                                "required_margin": margin,
                                "baseline_prediction": base_gen_pred,
                                "patched_prediction": base_gen_pred,
                                "baseline_correct": True,
                                "patched_correct": True,
                                "edit_applied": False,
                                "solver_feasible": True,
                                "predicted_constraints_satisfied_after_cap": True,
                                "required_natural_norm": 0.0,
                                "applied_natural_norm": 0.0,
                                "cap_factor": 1.0,
                                "dz_H": 0.0,
                                "dz_V": 0.0,
                                "spatial_delta_norm": 0.0,
                                "baseline_tf_prediction": base_tf_pred,
                                "patched_tf_prediction": base_tf_pred,
                                "baseline_gt_min_margin": base_gt_margin,
                                "predicted_gt_min_margin": base_gt_margin,
                                "patched_gt_min_margin": base_gt_margin,
                                "baseline_generation_text": base_gen_text,
                                "patched_generation_text": base_gen_text,
                                "active_constraints": "",
                            })
                            continue

                        sol = oracle_minimum_spatial_edit(
                            base_scores=base_scores,
                            target=gt,
                            A=A,
                            geom_L=g,
                            required_margin=margin,
                            max_natural=a.max_natural,
                            tol=a.qp_tol,
                        )

                        dz = sol["dz_applied"]
                        edit_applied = bool(
                            sol["solver_feasible"] and np.linalg.norm(dz) > 1e-10
                        )

                        if edit_applied:
                            dx = realize_delta(g, dz)
                            patch_map = make_relation_patch(L, sub_pos, ref_pos, dx)
                            patched_scores = all_scores(
                                model,
                                batch,
                                candidate_ids,
                                a.sequence_score_reduction,
                                decoder_layers=decoder_layers,
                                patch_map=patch_map,
                            )
                            patch_gen_pred, patch_gen_text = gate.generate_with_patch(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=batch,
                                patch_map=patch_map,
                                max_new_tokens=a.max_new_tokens,
                            )
                            spatial_delta_norm = float(np.linalg.norm(dx))
                        else:
                            patched_scores = dict(base_scores)
                            patch_gen_pred = base_gen_pred
                            patch_gen_text = base_gen_text
                            spatial_delta_norm = 0.0

                        pred_scores_arr = score_array(base_scores) + A @ dz
                        pred_scores = {r: float(pred_scores_arr[RID[r]]) for r in REL}
                        patch_tf_pred = argmax_relation(patched_scores)
                        pred_gt_margin = min_gt_margin(pred_scores, gt)
                        patch_gt_margin = min_gt_margin(patched_scores, gt)

                        repair_rows.append({
                            "sid": sid,
                            "gt": gt,
                            "source_layer": L,
                            "required_margin": margin,
                            "baseline_prediction": base_gen_pred,
                            "patched_prediction": patch_gen_pred,
                            "baseline_correct": False,
                            "patched_correct": patch_gen_pred == gt,
                            "edit_applied": edit_applied,
                            "solver_feasible": bool(sol["solver_feasible"]),
                            "predicted_constraints_satisfied_after_cap": bool(
                                sol["predicted_constraints_satisfied_after_cap"]
                            ),
                            "required_natural_norm": float(sol["required_natural_norm"]),
                            "applied_natural_norm": float(sol["applied_natural_norm"]),
                            "cap_factor": float(sol["cap_factor"]),
                            "dz_H": float(dz[0]),
                            "dz_V": float(dz[1]),
                            "spatial_delta_norm": spatial_delta_norm,
                            "baseline_tf_prediction": base_tf_pred,
                            "patched_tf_prediction": patch_tf_pred,
                            "baseline_gt_min_margin": base_gt_margin,
                            "predicted_gt_min_margin": pred_gt_margin,
                            "patched_gt_min_margin": patch_gt_margin,
                            "baseline_generation_text": base_gen_text,
                            "patched_generation_text": patch_gen_text,
                            "active_constraints": ",".join(map(str, sol["active_constraints"])),
                            **{f"patched_score_{r}": float(patched_scores[r]) for r in REL},
                        })

            except Exception as exc:
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "stage": "evaluation",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-16:],
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
        base_df.to_csv(outdir / "baseline.csv", index=False)
        repair_df.to_csv(outdir / "per_repair.csv", index=False)

        gen_df = generation_summary(repair_df)
        rel_df = relation_summary(repair_df)
        gen_df.to_csv(outdir / "generation_summary.csv", index=False)
        rel_df.to_csv(outdir / "repair_by_relation.csv", index=False)

        lines = []
        lines.append("=" * 176)
        lines.append("ORACLE ADAPTIVE SPATIAL REPAIR")
        lines.append("=" * 176)
        lines.append(f"model={a.model} repo={spec.repo_id}")
        lines.append(f"N successful={base_df.sid.nunique()} / requested={len(eval_meta)}")
        lines.append(f"source layers={source_layers}")
        lines.append(f"spatial states={spatial_path}")
        lines.append(f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}; eval_scope={a.eval_scope}")
        lines.append(
            f"transport source={a.transport_run_dir}" if a.transport_run_dir
            else f"transport calibration N={len(calib_meta)}"
        )
        lines.append(f"required margins={margins}; max_natural={a.max_natural}")
        lines.append(f"actual generation baseline accuracy={base_df.baseline_generation_correct.mean():.4f}")
        lines.append(f"teacher-forced baseline argmax accuracy={base_df.baseline_tf_correct.mean():.4f}")
        lines.append("")
        lines.append("CALIBRATED 2x2 AXIS TRANSPORT")
        lines.append("-" * 176)
        lines.append(t_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("ACTUAL GENERATION REPAIR")
        lines.append("-" * 176)
        lines.append(gen_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("BY RELATION")
        lines.append("-" * 176)
        lines.append(rel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  GT specifies only the desired relation on baseline-wrong samples.")
        lines.append("  The controller uses the CURRENT four-way decision gaps plus frozen A_L to solve")
        lines.append("  the minimum natural-norm spatial edit predicted to put GT above every competitor.")
        lines.append("  Correct baseline generations are not edited, so any C->W != 0 indicates a bug.")
        lines.append("  The primary result is actual free-generation W->C / accuracy gain.")
        lines.append("  If larger required_margin helps until capping dominates, the mapping is useful but")
        lines.append("  the linear model underestimates the displacement needed to cross generation boundaries.")
        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(report, encoding="utf-8")

        metadata = {
            "script": "eval_oracle_adaptive_spatial_repair_v2_standalone.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "spatial_states_npz": str(spatial_path),
            "source_layers": source_layers,
            "spatial_fit_ratio": a.spatial_fit_ratio,
            "spatial_fit_seed": a.spatial_fit_seed,
            "transport_run_dir": a.transport_run_dir or None,
            "transport_calibration_samples": (None if a.transport_run_dir else len(calib_meta)),
            "transport_calibration_scale": a.transport_calibration_scale,
            "eval_scope": a.eval_scope,
            "eval_requested": len(eval_meta),
            "required_margins": margins,
            "max_natural": a.max_natural,
            "answer_surface": a.answer_surface,
            "sequence_score_reduction": a.sequence_score_reduction,
            "max_new_tokens": a.max_new_tokens,
            "oracle_note": (
                "GT chooses the target relation only for baseline-wrong evaluation samples. "
                "Spatial geometry and transport A are frozen from calibration data."
            ),
            "primary_metric": "actual free-generation accuracy after spatial repair",
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
