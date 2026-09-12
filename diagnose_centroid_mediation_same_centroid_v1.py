#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_centroid_mediation_same_centroid_v1.py

Question
========
Does the ATTENTION CENTROID itself mediate the causal effect of a spatial
attention head on the downstream object-pair residual spatial state?

This script is deliberately NOT a correction / generation experiment.
It continues the previous causal diagnostic:

    diagnose_centroid_to_residual_causal_transport_v1_fixed.py

Previous result established:

    do(L24H05 visual-attention redistribution) -> downstream residual change

but a single attention redistribution changes both:
  (1) the centroid, and
  (2) higher-order attention shape / visual-value mixture.

The present experiment separates those two possibilities.

Matched-centroid intervention
=============================
For each sample and each direction H+/H-/V+/V-, choose one target RELATIVE
centroid displacement.  Subject and reference are moved antisymmetrically, and
for each object query we specify the complete 2D target centroid (x,y), keeping
the orthogonal coordinate fixed.

We then construct THREE different visual-attention distributions with the SAME:

    * total visual attention mass,
    * subject 2D centroid,
    * reference 2D centroid,

but different higher-order shapes:

  1) min_kl
       q(j) ∝ p(j) exp(lambda_x x_j + lambda_y y_j)
       lambda is solved so E_q[(x,y)] equals the requested target centroid.
       This is the minimum-KL / exponential-family realization relative to p.

  2) radial_null
       Start from min_kl, then add a large perturbation lying in the exact
       nullspace of [sum(q), E_q[x], E_q[y]].  The perturbation has a radial
       spatial pattern.  Sum and 2D centroid remain unchanged.

  3) checker_null
       Same, but use a checker / high-frequency spatial pattern projected into
       the same moment-nullspace.

For each realization, the exact A'V-AV head-output delta is injected at the
selected object's pre-W_O head slice, exactly as in the previous script.

Shape-only control
==================
For radial_null and checker_null, we also perturb the CLEAN visual-attention
shape while preserving each object's CLEAN 2D centroid exactly.  Therefore:

    Delta centroid == 0

by construction.  If downstream residual state still changes substantially,
centroid is NOT a causally sufficient statistic of this head's computation.

Interpretation
==============
Strong evidence that centroid displacement mediates the effect would look like:

  A. Centroid matching error across min_kl/radial_null/checker_null is tiny.
  B. Their downstream Delta z_H / Delta z_V are very similar for the same target.
  C. Between-realization residual spread is small relative to the directional
     H+ vs H- / V+ vs V- effect.
  D. Shape-only controls (same centroid as clean) have near-zero residual effect.

If A holds but B/C/D fail, then centroid is an informative diagnostic of the
head's spatial computation but is not a causally sufficient statistic; the
higher-order attention/value mixture matters.

Dependency
==========
Place this script in the AdaptVis repo root next to the already-fixed script:

    diagnose_centroid_to_residual_causal_transport_v1_fixed.py

Recommended pilot
=================
CUDA_VISIBLE_DEVICES=0 python -u diagnose_centroid_mediation_same_centroid_v1.py \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --intervention-layer 24 \
  --target-head 5 \
  --readout-layers 23,24,25,26 \
  --target-relative-shift 0.20 \
  --shape-fraction 0.65 \
  --max-samples 80 \
  --device cuda:0 \
  --output-dir output/qwen3b_centroid_mediation_same_centroid_n80_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import diagnose_centroid_to_residual_causal_transport_v1_fixed as base
except Exception as exc:
    raise SystemExit(
        "Could not import diagnose_centroid_to_residual_causal_transport_v1_fixed.py. "
        "Put this script next to that fixed script in the AdaptVis repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )


EPS = 1e-12
METHODS = ("min_kl", "radial_null", "checker_null")
DIRECTIONS = ("H+", "H-", "V+", "V-")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument("--intervention-layer", type=int, default=24)
    p.add_argument("--target-head", type=int, default=5)
    p.add_argument("--readout-layers", default="23,24,25,26")
    p.add_argument(
        "--target-relative-shift",
        type=float,
        default=0.20,
        help=(
            "Requested relative centroid displacement in normalized visual-coordinate "
            "units. Subject/reference each move half this amount in opposite directions. "
            "Per-sample value is clipped only when needed for feasibility."
        ),
    )
    p.add_argument(
        "--feasible-safety",
        type=float,
        default=0.90,
        help="Fraction of available coordinate room allowed when clipping target shifts.",
    )
    p.add_argument(
        "--min-relative-shift",
        type=float,
        default=0.03,
        help="Skip a direction for a sample if the feasible matched shift is smaller.",
    )
    p.add_argument(
        "--shape-fraction",
        type=float,
        default=0.65,
        help=(
            "Relative multiplicative nullspace deformation amplitude used for "
            "radial/checker realizations. Larger means more different attention shapes "
            "while preserving sum and centroid exactly; must stay below 1."
        ),
    )
    p.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of min_kl,radial_null,checker_null.",
    )
    p.add_argument(
        "--shape-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run same-centroid-as-clean radial/checker shape-only controls.",
    )
    p.add_argument(
        "--object-pool",
        default="mean",
        choices=["mean", "last"],
    )
    p.add_argument("--max-samples", type=int, default=80, help="0 = all 440")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.readout_layers_parsed = base.parse_layers(args.readout_layers)
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    bad = [x for x in methods if x not in METHODS]
    if bad:
        p.error(f"Unknown --methods: {bad}; allowed={METHODS}")
    if "min_kl" not in methods:
        p.error("--methods must include min_kl because null realizations are built from it")
    args.methods_parsed = methods

    if args.target_relative_shift <= 0:
        p.error("--target-relative-shift must be > 0")
    if not (0 < args.feasible_safety < 1):
        p.error("--feasible-safety must be in (0,1)")
    if args.min_relative_shift <= 0:
        p.error("--min-relative-shift must be > 0")
    if not (0 < args.shape_fraction < 1):
        p.error("--shape-fraction must be in (0,1)")
    return args


# =============================================================================
# Distribution / moment helpers
# =============================================================================

def normalize_prob(p: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.maximum(p, floor)
    s = float(p.sum())
    if not math.isfinite(s) or s <= 0:
        raise RuntimeError("Probability vector is degenerate")
    return p / s


def centroid(q: np.ndarray, coords: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) @ np.asarray(coords, dtype=np.float64)


def kl_div(q: np.ndarray, p: np.ndarray) -> float:
    q = normalize_prob(q)
    p = normalize_prob(p)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_prob(p)
    q = normalize_prob(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def solve_min_kl_centroid(
    p: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """I-projection of p onto E_q[coords] = target via 2D exponential tilt."""
    p = normalize_prob(p)
    C = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(2)
    lam = np.zeros(2, dtype=np.float64)

    def eval_lam(x: np.ndarray):
        logits = np.log(p) + C @ x
        logits -= float(np.max(logits))
        q = np.exp(logits)
        q /= float(q.sum())
        mu = q @ C
        centered = C - mu[None, :]
        cov = (centered * q[:, None]).T @ centered
        return q, mu, cov

    q, mu, cov = eval_lam(lam)
    initial_err = float(np.linalg.norm(mu - target))

    for _ in range(max_iter):
        err = target - mu
        err_norm = float(np.linalg.norm(err))
        if err_norm <= tol:
            break
        H = cov + np.eye(2, dtype=np.float64) * 1e-10
        try:
            step = np.linalg.solve(H, err)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ err
        step_norm = float(np.linalg.norm(step))
        if step_norm > 25.0:
            step *= 25.0 / step_norm

        old = err_norm
        accepted = False
        scale = 1.0
        for _ls in range(20):
            trial = lam + scale * step
            q_t, mu_t, cov_t = eval_lam(trial)
            new = float(np.linalg.norm(target - mu_t))
            if new < old:
                lam, q, mu, cov = trial, q_t, mu_t, cov_t
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

    final_err = float(np.linalg.norm(mu - target))
    if final_err > 2e-5:
        raise RuntimeError(
            f"min-KL centroid solve did not converge: initial={initial_err:.3e} "
            f"final={final_err:.3e} target={target.tolist()} achieved={mu.tolist()}"
        )
    return q, {
        "solver_centroid_error": final_err,
        "solver_lambda_x": float(lam[0]),
        "solver_lambda_y": float(lam[1]),
        "solver_kl_to_clean": kl_div(q, p),
    }


def weighted_relative_null_pattern(
    q_base: np.ndarray,
    coords: np.ndarray,
    kind: str,
) -> np.ndarray:
    """
    Construct a relative perturbation u such that n = q_base * u satisfies:

        sum n = 0
        sum n*x = 0
        sum n*y = 0

    Hence q' = q_base * (1 + alpha*u) preserves normalization and both
    centroid coordinates exactly.  Weighting the nullspace by q_base avoids a
    tiny-probability token determining the usable perturbation amplitude.
    """
    q0 = normalize_prob(q_base)
    C = np.asarray(coords, dtype=np.float64)
    x, y = C[:, 0], C[:, 1]
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    sx = max(float(np.std(x0)), 1e-6)
    sy = max(float(np.std(y0)), 1e-6)

    if kind == "radial_null":
        raw = (x0 / sx) ** 2 + (y0 / sy) ** 2
    elif kind == "checker_null":
        xn = (x - float(np.min(x))) / max(float(np.max(x) - np.min(x)), 1e-6)
        yn = (y - float(np.min(y))) / max(float(np.max(y) - np.min(y)), 1e-6)
        raw = np.sin(4.0 * np.pi * xn) * np.sin(4.0 * np.pi * yn)
    else:
        raise ValueError(kind)
    raw = raw - float(np.sum(q0 * raw))

    # Constraints on u are B @ u = 0 where B = A diag(q0).
    A = np.vstack([
        np.ones(len(C), dtype=np.float64),
        C[:, 0],
        C[:, 1],
    ])
    B = A * q0[None, :]
    gram = B @ B.T
    u = raw - B.T @ (np.linalg.pinv(gram) @ (B @ raw))
    # Re-project once for numerical cleanliness.
    u = u - B.T @ (np.linalg.pinv(gram) @ (B @ u))
    max_abs = float(np.max(np.abs(u)))
    if max_abs <= 1e-12:
        raise RuntimeError(f"Weighted moment-null pattern {kind} collapsed to zero")
    return u / max_abs


def add_null_shape(
    q_base: np.ndarray,
    coords: np.ndarray,
    kind: str,
    fraction: float,
    floor: float = 1e-15,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Strong multiplicative shape change with exactly unchanged sum and x/y centroid."""
    q0 = normalize_prob(q_base, floor=floor)
    u = weighted_relative_null_pattern(q0, coords, kind)
    alpha = float(fraction)
    if not (0.0 < alpha < 1.0):
        raise ValueError("fraction must be in (0,1)")
    q = q0 * (1.0 + alpha * u)
    if float(np.min(q)) <= 0.0:
        raise RuntimeError(f"{kind} produced a non-positive probability")
    # The constraints make renormalization unnecessary, but normalize once to
    # remove machine-precision drift.
    q /= float(q.sum())
    c_err = float(np.linalg.norm(centroid(q, coords) - centroid(q0, coords)))
    if c_err > 2e-7:
        raise RuntimeError(f"{kind} centroid drift too large: {c_err:.3e}")
    return q, {
        "null_alpha": alpha,
        "null_l1_from_base": l1_dist(q, q0),
        "null_js_from_base": js_div(q, q0),
        "null_centroid_error": c_err,
        "null_max_relative_change": float(np.max(np.abs((q - q0) / q0))),
    }


def extract_query_visual_distribution(
    *,
    prompt_attention: torch.Tensor,
    head: int,
    query_index: int,
    visual_indices: Sequence[int],
    coords: torch.Tensor,
) -> Dict[str, Any]:
    device = prompt_attention.device
    vidx = torch.as_tensor(visual_indices, device=device, dtype=torch.long)
    full_row = prompt_attention[int(head), int(query_index), :]
    a, p_t, mass_t = base.normalized_visual_map(full_row, vidx)
    p = p_t.detach().float().cpu().numpy().astype(np.float64)
    p = normalize_prob(p)
    C = coords.detach().float().cpu().numpy().astype(np.float64)
    return {
        "p": p,
        "mass": float(mass_t.item()),
        "centroid": centroid(p, C),
        "coords": C,
        "visual_idx": vidx,
        "a": a.detach(),
    }


def direction_raw_signs(direction: str) -> Tuple[int, int, int]:
    """Return coordinate axis index, subject raw-coordinate sign, reference sign."""
    if direction == "H+":
        return 0, +1, -1
    if direction == "H-":
        return 0, -1, +1
    # cV = -(y_sub-y_ref); V+ means above => subject y decreases, ref y increases.
    if direction == "V+":
        return 1, -1, +1
    if direction == "V-":
        return 1, +1, -1
    raise ValueError(direction)


def feasible_relative_shift(
    sub_centroid: np.ndarray,
    ref_centroid: np.ndarray,
    coords: np.ndarray,
    direction: str,
    requested: float,
    safety: float,
) -> float:
    axis, s_sign, r_sign = direction_raw_signs(direction)
    lo = float(np.min(coords[:, axis]))
    hi = float(np.max(coords[:, axis]))

    def room(value: float, sign: int) -> float:
        return (hi - value) if sign > 0 else (value - lo)

    room_sub = room(float(sub_centroid[axis]), s_sign)
    room_ref = room(float(ref_centroid[axis]), r_sign)
    # each object moves t = relative_shift / 2
    max_relative = 2.0 * max(0.0, min(room_sub, room_ref)) * float(safety)
    return float(min(float(requested), max_relative))


def target_centroids_for_direction(
    sub_centroid: np.ndarray,
    ref_centroid: np.ndarray,
    direction: str,
    relative_shift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    axis, s_sign, r_sign = direction_raw_signs(direction)
    t = 0.5 * float(relative_shift)
    sub = np.asarray(sub_centroid, dtype=np.float64).copy()
    ref = np.asarray(ref_centroid, dtype=np.float64).copy()
    sub[axis] += float(s_sign) * t
    ref[axis] += float(r_sign) * t
    return sub, ref


def make_realizations(
    p: np.ndarray,
    coords: np.ndarray,
    target_centroid: np.ndarray,
    methods: Sequence[str],
    shape_fraction: float,
) -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
    q_kl, meta_kl = solve_min_kl_centroid(p, coords, target_centroid)
    out: Dict[str, Tuple[np.ndarray, Dict[str, float]]] = {
        "min_kl": (q_kl, dict(meta_kl))
    }
    for method in methods:
        if method == "min_kl":
            continue
        q_alt, meta_alt = add_null_shape(
            q_kl, coords, method, fraction=shape_fraction
        )
        meta = dict(meta_kl)
        meta.update(meta_alt)
        out[method] = (q_alt, meta)
    return {m: out[m] for m in methods}


def distribution_to_head_delta(
    *,
    p_clean: np.ndarray,
    q_new: np.ndarray,
    mass: float,
    values: torch.Tensor,
    head: int,
    visual_indices: Sequence[int],
) -> torch.Tensor:
    device = values.device
    vidx = torch.as_tensor(visual_indices, device=device, dtype=torch.long)
    delta_a = float(mass) * (
        torch.as_tensor(q_new, device=device, dtype=torch.float32)
        - torch.as_tensor(p_clean, device=device, dtype=torch.float32)
    )
    v = values[0, int(head)].index_select(0, vidx).float()
    return torch.matmul(delta_a.unsqueeze(0), v).squeeze(0).detach()


@dataclass(frozen=True)
class BuiltIntervention:
    condition: str
    method: str
    deltas: Dict[int, torch.Tensor]
    metrics: Dict[str, float]


def build_matched_interventions(
    *,
    clean: Mapping[str, Any],
    head: int,
    direction: str,
    args: argparse.Namespace,
) -> List[BuiltIntervention]:
    sub = extract_query_visual_distribution(
        prompt_attention=clean["prompt_attention"],
        head=head,
        query_index=clean["subject_index"],
        visual_indices=clean["visual_indices"],
        coords=clean["coords"],
    )
    ref = extract_query_visual_distribution(
        prompt_attention=clean["prompt_attention"],
        head=head,
        query_index=clean["reference_index"],
        visual_indices=clean["visual_indices"],
        coords=clean["coords"],
    )
    C = sub["coords"]
    if not np.allclose(C, ref["coords"]):
        raise RuntimeError("subject/reference coordinate grids differ")

    rel_shift = feasible_relative_shift(
        sub["centroid"],
        ref["centroid"],
        C,
        direction,
        requested=args.target_relative_shift,
        safety=args.feasible_safety,
    )
    if rel_shift < args.min_relative_shift:
        raise RuntimeError(
            f"feasible relative shift {rel_shift:.4f} < min {args.min_relative_shift:.4f}"
        )

    target_sub, target_ref = target_centroids_for_direction(
        sub["centroid"], ref["centroid"], direction, rel_shift
    )
    sub_real = make_realizations(
        sub["p"], C, target_sub, args.methods_parsed, args.shape_fraction
    )
    ref_real = make_realizations(
        ref["p"], C, target_ref, args.methods_parsed, args.shape_fraction
    )

    clean_dx = float(sub["centroid"][0] - ref["centroid"][0])
    clean_dy = float(sub["centroid"][1] - ref["centroid"][1])
    clean_cH, clean_cV = clean_dx, -clean_dy

    out = []
    for method in args.methods_parsed:
        qs, ms = sub_real[method]
        qr, mr = ref_real[method]
        cs = centroid(qs, C)
        cr = centroid(qr, C)
        cf_dx = float(cs[0] - cr[0])
        cf_dy = float(cs[1] - cr[1])
        cf_cH, cf_cV = cf_dx, -cf_dy

        dsub = distribution_to_head_delta(
            p_clean=sub["p"], q_new=qs, mass=sub["mass"],
            values=clean["values"], head=head,
            visual_indices=clean["visual_indices"],
        )
        dref = distribution_to_head_delta(
            p_clean=ref["p"], q_new=qr, mass=ref["mass"],
            values=clean["values"], head=head,
            visual_indices=clean["visual_indices"],
        )

        metrics = {
            "requested_relative_shift": float(args.target_relative_shift),
            "effective_relative_shift": float(rel_shift),
            "clean_cH": clean_cH,
            "clean_cV": clean_cV,
            "cf_cH": cf_cH,
            "cf_cV": cf_cV,
            "delta_cH": cf_cH - clean_cH,
            "delta_cV": cf_cV - clean_cV,
            "subject_target_x": float(target_sub[0]),
            "subject_target_y": float(target_sub[1]),
            "reference_target_x": float(target_ref[0]),
            "reference_target_y": float(target_ref[1]),
            "subject_centroid_error": float(np.linalg.norm(cs - target_sub)),
            "reference_centroid_error": float(np.linalg.norm(cr - target_ref)),
            "subject_l1_from_clean": l1_dist(qs, sub["p"]),
            "reference_l1_from_clean": l1_dist(qr, ref["p"]),
            "subject_js_from_clean": js_div(qs, sub["p"]),
            "reference_js_from_clean": js_div(qr, ref["p"]),
            "subject_kl_from_clean": kl_div(qs, sub["p"]),
            "reference_kl_from_clean": kl_div(qr, ref["p"]),
            "subject_visual_mass": float(sub["mass"]),
            "reference_visual_mass": float(ref["mass"]),
            "delta_head_norm_subject": float(torch.linalg.vector_norm(dsub.float()).item()),
            "delta_head_norm_reference": float(torch.linalg.vector_norm(dref.float()).item()),
            "subject_null_l1_from_base": float(ms.get("null_l1_from_base", 0.0)),
            "reference_null_l1_from_base": float(mr.get("null_l1_from_base", 0.0)),
            "subject_null_js_from_base": float(ms.get("null_js_from_base", 0.0)),
            "reference_null_js_from_base": float(mr.get("null_js_from_base", 0.0)),
        }
        out.append(
            BuiltIntervention(
                condition=direction,
                method=method,
                deltas={
                    int(clean["subject_index"]): dsub,
                    int(clean["reference_index"]): dref,
                },
                metrics=metrics,
            )
        )
    return out


def build_shape_only_interventions(
    *,
    clean: Mapping[str, Any],
    head: int,
    args: argparse.Namespace,
) -> List[BuiltIntervention]:
    sub = extract_query_visual_distribution(
        prompt_attention=clean["prompt_attention"], head=head,
        query_index=clean["subject_index"], visual_indices=clean["visual_indices"],
        coords=clean["coords"],
    )
    ref = extract_query_visual_distribution(
        prompt_attention=clean["prompt_attention"], head=head,
        query_index=clean["reference_index"], visual_indices=clean["visual_indices"],
        coords=clean["coords"],
    )
    C = sub["coords"]
    clean_cH = float(sub["centroid"][0] - ref["centroid"][0])
    clean_cV = -float(sub["centroid"][1] - ref["centroid"][1])
    out = []
    for method in ("radial_null", "checker_null"):
        if method not in args.methods_parsed:
            continue
        qs, ms = add_null_shape(sub["p"], C, method, args.shape_fraction)
        qr, mr = add_null_shape(ref["p"], C, method, args.shape_fraction)
        cs = centroid(qs, C)
        cr = centroid(qr, C)
        cf_cH = float(cs[0] - cr[0])
        cf_cV = -float(cs[1] - cr[1])
        dsub = distribution_to_head_delta(
            p_clean=sub["p"], q_new=qs, mass=sub["mass"], values=clean["values"],
            head=head, visual_indices=clean["visual_indices"],
        )
        dref = distribution_to_head_delta(
            p_clean=ref["p"], q_new=qr, mass=ref["mass"], values=clean["values"],
            head=head, visual_indices=clean["visual_indices"],
        )
        out.append(
            BuiltIntervention(
                condition="shape_only",
                method=method,
                deltas={int(clean["subject_index"]): dsub,
                        int(clean["reference_index"]): dref},
                metrics={
                    "requested_relative_shift": 0.0,
                    "effective_relative_shift": 0.0,
                    "clean_cH": clean_cH,
                    "clean_cV": clean_cV,
                    "cf_cH": cf_cH,
                    "cf_cV": cf_cV,
                    "delta_cH": cf_cH - clean_cH,
                    "delta_cV": cf_cV - clean_cV,
                    "subject_target_x": float(sub["centroid"][0]),
                    "subject_target_y": float(sub["centroid"][1]),
                    "reference_target_x": float(ref["centroid"][0]),
                    "reference_target_y": float(ref["centroid"][1]),
                    "subject_centroid_error": float(np.linalg.norm(cs - sub["centroid"])),
                    "reference_centroid_error": float(np.linalg.norm(cr - ref["centroid"])),
                    "subject_l1_from_clean": l1_dist(qs, sub["p"]),
                    "reference_l1_from_clean": l1_dist(qr, ref["p"]),
                    "subject_js_from_clean": js_div(qs, sub["p"]),
                    "reference_js_from_clean": js_div(qr, ref["p"]),
                    "subject_kl_from_clean": kl_div(qs, sub["p"]),
                    "reference_kl_from_clean": kl_div(qr, ref["p"]),
                    "subject_visual_mass": float(sub["mass"]),
                    "reference_visual_mass": float(ref["mass"]),
                    "delta_head_norm_subject": float(torch.linalg.vector_norm(dsub.float()).item()),
                    "delta_head_norm_reference": float(torch.linalg.vector_norm(dref.float()).item()),
                    "subject_null_l1_from_base": float(ms.get("null_l1_from_base", 0.0)),
                    "reference_null_l1_from_base": float(mr.get("null_l1_from_base", 0.0)),
                    "subject_null_js_from_base": float(ms.get("null_js_from_base", 0.0)),
                    "reference_null_js_from_base": float(mr.get("null_js_from_base", 0.0)),
                },
            )
        )
    return out


# =============================================================================
# Aggregation
# =============================================================================

def matched_centroid_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = df[df.condition.isin(DIRECTIONS)].copy()
    for (cond, L), g in sub.groupby(["condition", "readout_layer"], sort=False):
        method_stats = {}
        for m in METHODS:
            gm = g[g.method == m]
            if len(gm):
                method_stats[m] = {
                    "N": int(len(gm)),
                    "mean_dzH": float(gm.delta_z_H.mean()),
                    "mean_dzV": float(gm.delta_z_V.mean()),
                    "mean_dz_norm": float(np.sqrt(gm.delta_z_H**2 + gm.delta_z_V**2).mean()),
                }
        if not method_stats:
            continue

        # Per-sample spread across methods at same target centroid.
        spreads = []
        for sid, gs in g.groupby("sid"):
            if gs.method.nunique() < 2:
                continue
            z = gs[["delta_z_H", "delta_z_V"]].to_numpy(dtype=float)
            center_z = z.mean(axis=0, keepdims=True)
            spread = float(np.sqrt(np.sum((z - center_z) ** 2, axis=1)).mean())
            effect = float(np.linalg.norm(center_z[0]))
            spreads.append((spread, effect, spread / max(effect, EPS)))

        row = {
            "condition": cond,
            "readout_layer": int(L),
            "N_samples": int(g.sid.nunique()),
            "max_subject_centroid_error": float(g.subject_centroid_error.max()),
            "max_reference_centroid_error": float(g.reference_centroid_error.max()),
            "max_abs_delta_cH_range_across_methods": float(
                g.groupby("sid").delta_cH.agg(lambda x: float(x.max()-x.min())).max()
            ),
            "max_abs_delta_cV_range_across_methods": float(
                g.groupby("sid").delta_cV.agg(lambda x: float(x.max()-x.min())).max()
            ),
            "mean_method_spread_z": float(np.mean([x[0] for x in spreads])) if spreads else float("nan"),
            "mean_matched_effect_norm": float(np.mean([x[1] for x in spreads])) if spreads else float("nan"),
            "mean_spread_over_effect": float(np.mean([x[2] for x in spreads])) if spreads else float("nan"),
        }
        for m, s in method_stats.items():
            row[f"{m}_mean_dzH"] = s["mean_dzH"]
            row[f"{m}_mean_dzV"] = s["mean_dzV"]
            row[f"{m}_mean_dz_norm"] = s["mean_dz_norm"]
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_method_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = df[df.condition.isin(DIRECTIONS)].copy()
    pairs = [("min_kl", "radial_null"), ("min_kl", "checker_null"),
             ("radial_null", "checker_null")]
    for (cond, L), g in sub.groupby(["condition", "readout_layer"], sort=False):
        for a, b in pairs:
            ga = g[g.method == a].set_index("sid")
            gb = g[g.method == b].set_index("sid")
            ids = ga.index.intersection(gb.index)
            if len(ids) == 0:
                continue
            za = ga.loc[ids, ["delta_z_H", "delta_z_V"]].to_numpy(dtype=float)
            zb = gb.loc[ids, ["delta_z_H", "delta_z_V"]].to_numpy(dtype=float)
            diff = np.linalg.norm(za-zb, axis=1)
            denom = 0.5*(np.linalg.norm(za, axis=1)+np.linalg.norm(zb, axis=1))
            target_axis = 0 if cond.startswith("H") else 1
            sign_agree = np.mean(np.sign(za[:,target_axis]) == np.sign(zb[:,target_axis]))
            rows.append({
                "condition": cond,
                "readout_layer": int(L),
                "method_a": a,
                "method_b": b,
                "N": int(len(ids)),
                "corr_dzH": base.safe_corr(za[:,0], zb[:,0]),
                "corr_dzV": base.safe_corr(za[:,1], zb[:,1]),
                "target_axis_sign_agreement": float(sign_agree),
                "mean_delta_z_distance": float(diff.mean()),
                "mean_relative_delta_z_distance": float(np.mean(diff/np.maximum(denom, EPS))),
            })
    return pd.DataFrame(rows)


def shape_only_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = df[df.condition == "shape_only"]
    for (method, L), g in sub.groupby(["method", "readout_layer"], sort=False):
        rows.append({
            "method": method,
            "readout_layer": int(L),
            "N": int(len(g)),
            "max_abs_delta_cH": float(np.abs(g.delta_cH).max()),
            "max_abs_delta_cV": float(np.abs(g.delta_cV).max()),
            "mean_l1_shape_subject": float(g.subject_l1_from_clean.mean()),
            "mean_l1_shape_reference": float(g.reference_l1_from_clean.mean()),
            "mean_abs_delta_zH": float(np.abs(g.delta_z_H).mean()),
            "mean_abs_delta_zV": float(np.abs(g.delta_z_V).mean()),
            "mean_delta_z_norm": float(np.sqrt(g.delta_z_H**2 + g.delta_z_V**2).mean()),
            "median_delta_z_norm": float(np.median(np.sqrt(g.delta_z_H**2 + g.delta_z_V**2))),
        })
    return pd.DataFrame(rows)


def directional_scale_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Reference size of centroid-driven directional effects for each method/layer."""
    rows = []
    sub = df[df.condition.isin(DIRECTIONS)]
    for (method, L), g in sub.groupby(["method", "readout_layer"], sort=False):
        vals = []
        for sid, gs in g.groupby("sid"):
            by = {r.condition: r for _, r in gs.iterrows()}
            if all(k in by for k in DIRECTIONS):
                h = np.array([
                    float(by["H+"].delta_z_H - by["H-"].delta_z_H),
                    float(by["H+"].delta_z_V - by["H-"].delta_z_V),
                ])
                v = np.array([
                    float(by["V+"].delta_z_H - by["V-"].delta_z_H),
                    float(by["V+"].delta_z_V - by["V-"].delta_z_V),
                ])
                vals.append((float(np.linalg.norm(h)), float(np.linalg.norm(v))))
        if vals:
            rows.append({
                "method": method,
                "readout_layer": int(L),
                "N": int(len(vals)),
                "mean_H_plus_minus_effect_norm": float(np.mean([x[0] for x in vals])),
                "mean_V_plus_minus_effect_norm": float(np.mean([x[1] for x in vals])),
                "mean_directional_effect_norm": float(np.mean([0.5*(x[0]+x[1]) for x in vals])),
            })
    return pd.DataFrame(rows)


def mediation_ratio_summary(shape: pd.DataFrame, directional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(shape) == 0 or len(directional) == 0:
        return pd.DataFrame(rows)
    for _, s in shape.iterrows():
        d = directional[
            (directional.method == s.method) &
            (directional.readout_layer == s.readout_layer)
        ]
        if len(d) == 0:
            continue
        dr = d.iloc[0]
        rows.append({
            "method": s.method,
            "readout_layer": int(s.readout_layer),
            "shape_only_delta_z_norm": float(s.mean_delta_z_norm),
            "directional_plus_minus_effect_norm": float(dr.mean_directional_effect_norm),
            "shape_only_over_directional": float(
                s.mean_delta_z_norm / max(float(dr.mean_directional_effect_norm), EPS)
            ),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Xs, ys, Ls, sids_s, source_def = base.load_state_npz(
        Path(args.source_spatial_npz), require_labels=True
    )
    Xt, yt, Lt, sids_t, target_def = base.load_state_npz(
        Path(args.target_spatial_npz), require_labels=False
    )
    geom, geom_df = base.fit_hv_geometry(Xs, ys, Ls, args.readout_layers_parsed)
    geom_df.to_csv(outdir / "hv_geometry.csv", index=False)
    target_lmap = {int(L): i for i, L in enumerate(Lt)}
    target_sid_to_i = {int(sid): i for i, sid in enumerate(sids_t)}
    for L in args.readout_layers_parsed:
        if L not in target_lmap:
            raise RuntimeError(f"Target cache missing readout L{L}")

    # Compatibility with centexp helper APIs imported inside base.
    args.max_samples = None if int(args.max_samples) == 0 else int(args.max_samples)
    records, audit = base.centexp.load_coco_records(args)
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]

    model, processor, layers, decoder_path, spec = base.centexp.load_model_and_processor(args)
    if not (0 <= args.intervention_layer < len(layers)):
        raise RuntimeError(
            f"intervention L{args.intervention_layer} outside 0..{len(layers)-1}"
        )

    detail_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    skipped_direction_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 184)
    print("CENTROID MEDIATION TEST: SAME 2D CENTROID, DIFFERENT ATTENTION SHAPES")
    print("=" * 184)
    print(
        f"model={args.model} | N={len(records)} | intervention=L{args.intervention_layer}H{args.target_head} "
        f"| target_relative_shift={args.target_relative_shift} | shape_fraction={args.shape_fraction}"
    )
    print(f"methods={args.methods_parsed} | shape_only={args.shape_only}")
    print(f"readout_layers={args.readout_layers_parsed} | pool={args.object_pool}")
    print("No GT routing, no generation, no correction objective.")
    print("=" * 184, flush=True)

    for record in tqdm(records, desc="centroid mediation"):
        sid = int(record["sid"])
        image = None
        clean = None
        try:
            if sid not in target_sid_to_i:
                raise RuntimeError(f"sid={sid} missing from target spatial cache")
            ti = target_sid_to_i[sid]
            image = base.centexp.open_record_image(record)
            clean = base.clean_trace(
                model=model,
                processor=processor,
                layers=layers,
                image=image,
                record=record,
                intervention_layer=args.intervention_layer,
                args=args,
            )
            if args.target_head >= clean["n_heads"]:
                raise RuntimeError(
                    f"target_head={args.target_head} >= n_heads={clean['n_heads']}"
                )

            clean_residual: Dict[int, np.ndarray] = {}
            clean_z: Dict[int, np.ndarray] = {}
            for L in args.readout_layers_parsed:
                r = Xt[ti, target_lmap[L]].astype(np.float64)
                clean_residual[int(L)] = r
                clean_z[int(L)] = base.read_coord(r, geom[int(L)])

            interventions: List[BuiltIntervention] = []
            for direction in DIRECTIONS:
                try:
                    interventions.extend(
                        build_matched_interventions(
                            clean=clean,
                            head=args.target_head,
                            direction=direction,
                            args=args,
                        )
                    )
                except Exception as exc:
                    skipped_direction_rows.append({
                        "sid": sid,
                        "direction": direction,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })

            if args.shape_only:
                interventions.extend(
                    build_shape_only_interventions(
                        clean=clean,
                        head=args.target_head,
                        args=args,
                    )
                )

            if not interventions:
                raise RuntimeError("No feasible interventions for sample")

            for intervention in interventions:
                int_pairs = base.run_intervention(
                    model=model,
                    clean=clean,
                    head=args.target_head,
                    deltas=intervention.deltas,
                    args=args,
                )
                for L in args.readout_layers_parsed:
                    pair_delta = (
                        int_pairs[int(L)] - clean["clean_pairs"][int(L)]
                    ).numpy().astype(np.float64)
                    r_int = clean_residual[int(L)] + pair_delta
                    z_int = base.read_coord(r_int, geom[int(L)])
                    z0 = clean_z[int(L)]
                    detail_rows.append({
                        "sid": sid,
                        "gt": record["relation"],
                        "condition": intervention.condition,
                        "method": intervention.method,
                        "intervention_layer": int(args.intervention_layer),
                        "head": int(args.target_head),
                        "readout_layer": int(L),
                        **intervention.metrics,
                        "clean_z_H": float(z0[0]),
                        "clean_z_V": float(z0[1]),
                        "z_H": float(z_int[0]),
                        "z_V": float(z_int[1]),
                        "delta_z_H": float(z_int[0] - z0[0]),
                        "delta_z_V": float(z_int[1] - z0[1]),
                        "pair_delta_l2": float(np.linalg.norm(pair_delta)),
                    })

            if len(detail_rows) and sid % 10 == 0:
                pd.DataFrame(detail_rows).to_csv(
                    outdir / "sample_method_layer.csv", index=False
                )

        except Exception as exc:
            errors.append({
                "sid": sid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": " | ".join(traceback.format_exc().splitlines()[-12:]),
            })
            tqdm.write(f"[ERROR sid={sid}] {type(exc).__name__}: {exc}")
        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
            if clean is not None:
                try:
                    del clean
                except Exception:
                    pass
            base.cleanup()

    if not detail_rows:
        raise RuntimeError("No successful intervention rows")

    df = pd.DataFrame(detail_rows)
    df.to_csv(outdir / "sample_method_layer.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "errors.csv", index=False)
    if skipped_direction_rows:
        pd.DataFrame(skipped_direction_rows).to_csv(
            outdir / "skipped_directions.csv", index=False
        )

    matched = matched_centroid_summary(df)
    pairwise = pairwise_method_agreement(df)
    shape = shape_only_summary(df)
    directional = directional_scale_summary(df)
    mediation = mediation_ratio_summary(shape, directional)

    matched.to_csv(outdir / "matched_centroid_summary.csv", index=False)
    pairwise.to_csv(outdir / "pairwise_method_agreement.csv", index=False)
    shape.to_csv(outdir / "shape_only_summary.csv", index=False)
    directional.to_csv(outdir / "directional_effect_scale.csv", index=False)
    mediation.to_csv(outdir / "shape_vs_directional_ratio.csv", index=False)

    # One compact audit of how different the three attention realizations actually are.
    shape_diff = (
        df[df.condition.isin(DIRECTIONS)]
        .groupby(["condition", "method"], sort=False)
        .agg(
            N=("sid", "nunique"),
            mean_subject_l1_from_clean=("subject_l1_from_clean", "mean"),
            mean_reference_l1_from_clean=("reference_l1_from_clean", "mean"),
            mean_subject_null_l1_from_base=("subject_null_l1_from_base", "mean"),
            mean_reference_null_l1_from_base=("reference_null_l1_from_base", "mean"),
            max_subject_centroid_error=("subject_centroid_error", "max"),
            max_reference_centroid_error=("reference_centroid_error", "max"),
        )
        .reset_index()
    )
    shape_diff.to_csv(outdir / "attention_realization_audit.csv", index=False)

    print("\n" + "=" * 184)
    print("MATCHED-CENTROID MEDIATION SUMMARY")
    print("=" * 184)
    if len(matched):
        print(matched.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPAIRWISE DOWNSTREAM AGREEMENT AT THE SAME 2D CENTROID")
    print("-" * 184)
    if len(pairwise):
        print(pairwise.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSHAPE-ONLY CONTROL: CLEAN CENTROID HELD FIXED")
    print("-" * 184)
    if len(shape):
        print(shape.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSHAPE-ONLY EFFECT RELATIVE TO DIRECTIONAL CENTROID EFFECT")
    print("-" * 184)
    if len(mediation):
        print(mediation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nATTENTION REALIZATION AUDIT")
    print("-" * 184)
    if len(shape_diff):
        print(shape_diff.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    config = {
        "script": "diagnose_centroid_mediation_same_centroid_v1.py",
        "model": args.model,
        "repo_id": getattr(spec, "repo_id", ""),
        "decoder_path": decoder_path,
        "N_requested": len(records),
        "N_success": int(df.sid.nunique()),
        "N_errors": len(errors),
        "intervention_layer": args.intervention_layer,
        "target_head": args.target_head,
        "readout_layers": args.readout_layers_parsed,
        "target_relative_shift": args.target_relative_shift,
        "feasible_safety": args.feasible_safety,
        "min_relative_shift": args.min_relative_shift,
        "shape_fraction": args.shape_fraction,
        "methods": args.methods_parsed,
        "shape_only": args.shape_only,
        "object_pool": args.object_pool,
        "source_spatial_npz": args.source_spatial_npz,
        "target_spatial_npz": args.target_spatial_npz,
        "source_definition": source_def,
        "target_definition": target_def,
        "test_definition": (
            "same subject/reference 2D attention centroids and same visual mass, "
            "different higher-order visual-attention distributions; exact A'V-AV "
            "counterfactual head-output patch before W_O"
        ),
        "audit": audit,
    }
    (outdir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (outdir / "DONE").write_text("ok\n", encoding="utf-8")

    print(f"\n[saved] {outdir / 'sample_method_layer.csv'}")
    print(f"[saved] {outdir / 'matched_centroid_summary.csv'}")
    print(f"[saved] {outdir / 'pairwise_method_agreement.csv'}")
    print(f"[saved] {outdir / 'shape_only_summary.csv'}")
    print(f"[saved] {outdir / 'shape_vs_directional_ratio.csv'}")
    print(f"[success] N={df.sid.nunique()} errors={len(errors)} skipped_directions={len(skipped_direction_rows)}")


if __name__ == "__main__":
    main()
