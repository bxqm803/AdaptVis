#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nonoracle_synthetic400_to_coco440_spatial_repair_v1.py

Cross-domain, no-test-oracle spatial repair.

Source / target split
=====================
SOURCE: a labeled synthetic-shapes residual-state NPZ (e.g. 400 samples).
TARGET: a COCO_two residual-state NPZ (e.g. all 440 samples).

Only SOURCE labels are used to define the per-layer spatial geometry:

    mu_{L,r}, d_H^L, d_V^L, natural H/V scale.

TARGET labels are never used for routing, gating, target selection, spatial
optimization, centering, scaling, thresholding, or any other inference-time
operation.  They are read only after predictions exist to report accuracy,
W2C/C2W, and per-relation diagnostics.

Spatial selector
================
For each source layer L, fit four class directions from the complete synthetic
source set:

    c_{L,r} = unit(mu^syn_{L,r} - mu^syn_L)

For a target COCO sample x, score its precomputed REAL-NoImage relation vector
using ONLY the frozen synthetic geometry:

    e_{L,r}(x) = cos(x_L - mu^syn_L, c^syn_{L,r})
    E_r(x)     = mean_L e_{L,r}(x)
    r_spatial  = argmax_r E_r(x)

No COCO calibration split is created.  All target samples whose sample IDs are
present in the target NPZ are eligible for evaluation.

Non-oracle repair
=================
If r_spatial conflicts with the model's current generated decision, treat the
synthetic-derived spatial prediction as the target.  The repair optimizer may
edit ONLY the frozen synthetic H/V spatial subspaces at the selected layers.
At each step it maximizes the target-vs-all-competitors smooth decision margin:

    J = S_target - tau * logsumexp({S_j/tau : j != target})

using normalized local gradients and backtracking line search.  Model weights
are never updated.

The script computes candidate repairs for all spatial-decision conflicts once,
then reports a GT-free risk/coverage curve based on spatial confidence.

Important
=========
This is cross-domain transfer, not fully unsupervised learning: synthetic
relation labels define the frozen spatial geometry.  COCO labels are evaluation
only.
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


def norm_rel(x) -> str:
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"No layers parsed from {text!r}")
    return sorted(out)


def parse_float_list(text: str) -> List[float]:
    vals = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if v < 0 or v > 1:
            raise ValueError("coverage values must be in [0,1]")
        vals.append(v)
    if not vals:
        raise ValueError("empty coverage grid")
    return sorted(set(vals))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def argmax_relation(scores: Dict[str, float]) -> str:
    return max(REL, key=lambda r: float(scores[r]))


def top2_margin(scores: Dict[str, float]) -> float:
    vals = sorted([float(scores[r]) for r in REL], reverse=True)
    return float(vals[0] - vals[1])


def min_target_margin(scores: Dict[str, float], target: str) -> float:
    return min(float(scores[target]) - float(scores[r]) for r in REL if r != target)


def strongest_competitor(scores: Dict[str, float], target: str) -> str:
    return max((r for r in REL if r != target), key=lambda r: float(scores[r]))


def smooth_competitor_weights(scores: Dict[str, float], target: str, tau: float):
    comps = [r for r in REL if r != target]
    vals = np.asarray([float(scores[r]) for r in comps], dtype=np.float64) / float(tau)
    vals -= float(np.max(vals))
    w = np.exp(vals)
    w /= max(float(np.sum(w)), EPS)
    return comps, w


def smooth_target_objective(scores: Dict[str, float], target: str, tau: float) -> float:
    comps = [r for r in REL if r != target]
    vals = np.asarray([float(scores[r]) for r in comps], dtype=np.float64) / float(tau)
    vmax = float(np.max(vals))
    lse = vmax + math.log(float(np.sum(np.exp(vals - vmax))))
    return float(scores[target]) - float(tau) * lse


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def fit_spatial_geometry_and_selector(X, y, spatial_layers, fit_idx, source_layers):
    layer_to_i = {int(L): i for i, L in enumerate(spatial_layers)}
    missing = [L for L in source_layers if L not in layer_to_i]
    if missing:
        raise RuntimeError(f"Spatial NPZ missing layers {missing}; has {spatial_layers}")

    geom = {}
    rows = []
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
        B = np.stack([dH, dV], axis=1)
        gram = B.T @ B
        dual = B @ np.linalg.inv(gram)
        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)
        natural_M = dual @ np.diag([halfH, halfV])

        geom[L] = {
            "layer_index": li,
            "center": center.astype(np.float32),
            "class_dirs": {r: class_dirs[r].astype(np.float32) for r in REL},
            "natural_M": natural_M.astype(np.float32),
            "natural_half_H": float(halfH),
            "natural_half_V": float(halfV),
        }
        rows.append({
            "source_layer": L,
            "fit_N": len(fit_idx),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "natural_full_gap_H": float(gapH),
            "natural_full_gap_V": float(gapV),
        })
    return geom, pd.DataFrame(rows), layer_to_i


def spatial_selector_for_sample(Xrow: np.ndarray, geom: dict, source_layers: Sequence[int]):
    layer_scores = {}
    layer_preds = {}
    for L in source_layers:
        li = int(geom[L]["layer_index"])
        v = Xrow[li].astype(np.float64) - geom[L]["center"].astype(np.float64)
        nv = float(np.linalg.norm(v))
        if nv < EPS:
            scores = {r: 0.0 for r in REL}
        else:
            vv = v / nv
            scores = {
                r: float(np.dot(vv, geom[L]["class_dirs"][r].astype(np.float64)))
                for r in REL
            }
        layer_scores[L] = scores
        layer_preds[L] = argmax_relation(scores)

    agg = {r: float(np.mean([layer_scores[L][r] for L in source_layers])) for r in REL}
    ordered = sorted(REL, key=lambda r: agg[r], reverse=True)
    pred = ordered[0]
    margin = float(agg[ordered[0]] - agg[ordered[1]])
    agreement = float(np.mean([layer_preds[L] == pred for L in source_layers]))
    return {
        "prediction": pred,
        "margin": margin,
        "agreement": agreement,
        "aggregate_scores": agg,
        "layer_scores": layer_scores,
        "layer_predictions": layer_preds,
    }


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


def natural_coords_to_patch(layers, geom, sub_pos, ref_pos, coords):
    coords = np.asarray(coords, dtype=np.float64)
    patch = {}
    for i, L in enumerate(layers):
        dx = geom[L]["natural_M"].astype(np.float64) @ coords[i]
        dx = dx.astype(np.float32)
        patch[int(L)] = {
            int(sub_pos): (+0.5 * dx).astype(np.float32),
            int(ref_pos): (-0.5 * dx).astype(np.float32),
        }
    return patch


def all_scores(model, batch, candidate_ids, reduction, decoder_layers, patch_map):
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


class DifferentiableSpatialPatch:
    def __init__(
        self, *, decoder_layers, layers, geom, sub_pos, ref_pos,
        base_coords, delta_ctrl, prompt_len,
    ):
        self.handles = []
        self.prompt_len = int(prompt_len)
        base_coords = np.asarray(base_coords, dtype=np.float32)
        for i, L in enumerate(layers):
            M_np = geom[L]["natural_M"].astype(np.float32)
            base_i = torch.as_tensor(base_coords[i], device=delta_ctrl.device, dtype=torch.float32)

            def make_hook(local_i, local_M_np, local_base):
                def hook(_m, _inp, out):
                    x = gate.first_tensor(out)
                    if int(x.shape[1]) != self.prompt_len:
                        return None
                    M = torch.as_tensor(local_M_np, device=x.device, dtype=x.dtype)
                    b = local_base.to(device=x.device, dtype=x.dtype)
                    c = b + delta_ctrl[local_i].to(device=x.device, dtype=x.dtype)
                    dx = M @ c
                    y = x.clone()
                    if 0 <= sub_pos < int(y.shape[1]):
                        y[0, int(sub_pos)] = y[0, int(sub_pos)] + 0.5 * dx
                    if 0 <= ref_pos < int(y.shape[1]):
                        y[0, int(ref_pos)] = y[0, int(ref_pos)] - 0.5 * dx
                    return gate.replace_first_tensor(out, y)
                return hook

            self.handles.append(
                decoder_layers[L].register_forward_hook(make_hook(i, M_np, base_i))
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def sequence_score_and_spatial_grad(
    *, model, decoder_layers, batch, answer_ids, reduction,
    layers, geom, sub_pos, ref_pos, base_coords,
):
    ext, T = gate.extend_batch_with_candidate(batch, answer_ids)
    full_len = T + len(answer_ids)
    dev = batch["input_ids"].device
    delta = torch.zeros((len(layers), 2), device=dev, dtype=torch.float32, requires_grad=True)
    ctx = DifferentiableSpatialPatch(
        decoder_layers=decoder_layers,
        layers=layers,
        geom=geom,
        sub_pos=sub_pos,
        ref_pos=ref_pos,
        base_coords=base_coords,
        delta_ctrl=delta,
        prompt_len=full_len,
    )
    with ctx, torch.enable_grad():
        kw = dict(ext)
        kw["use_cache"] = False
        kw["return_dict"] = True
        out = model(**kw)
        score = gate.sequence_score_from_logits(out.logits, T, answer_ids, reduction)
        grad = torch.autograd.grad(
            score, delta, retain_graph=False, create_graph=False, allow_unused=False
        )[0]
    return float(score.detach().item()), grad.detach().float().cpu().numpy().astype(np.float64)


def normalize_direction(g):
    g = np.asarray(g, dtype=np.float64)
    n = float(np.linalg.norm(g.reshape(-1)))
    if n < EPS:
        return None, n
    return g / n, n


def apply_control_step(coords, direction, step, max_total, max_layer):
    cand = np.asarray(coords, dtype=np.float64) + float(step) * np.asarray(direction, dtype=np.float64)
    if max_layer > 0:
        for i in range(len(cand)):
            n = float(np.linalg.norm(cand[i]))
            if n > max_layer:
                cand[i] *= max_layer / max(n, EPS)
    nall = float(np.linalg.norm(cand.reshape(-1)))
    hit_total = False
    if max_total > 0 and nall > max_total:
        cand *= max_total / max(nall, EPS)
        hit_total = True
    return cand, hit_total


def compute_gate_score(selector, decision_margin: float, mode: str) -> float:
    m = max(float(selector["margin"]), 0.0)
    a = max(float(selector["agreement"]), 0.0)
    d = max(float(decision_margin), 0.0)
    if mode == "spatial_margin":
        return m
    if mode == "agreement":
        return a
    if mode == "consensus_over_decision":
        return (m * a) / (0.05 + d)
    return m * a  # consensus


def optimize_to_spatial_target(
    *, a, model, processor, decoder_layers, batch, candidate_ids, layers, geom,
    sub_pos, ref_pos, target, base_scores, base_gen_pred, base_gen_text, common,
    step_rows,
):
    coords = np.zeros((len(layers), 2), dtype=np.float64)
    current_scores = dict(base_scores)
    current_tf_pred = argmax_relation(current_scores)
    current_gen_pred = base_gen_pred
    current_gen_text = base_gen_text
    hit_total_cap = False
    line_search_failed = False
    stop_reason = "max_steps"
    steps_taken = 0

    for step_idx in range(1, a.max_steps + 1):
        current_margin = min_target_margin(current_scores, target)
        tf_ready = current_margin >= a.required_margin
        gen_ready = current_gen_pred == target
        if gen_ready:
            stop_reason = "generation_reached_spatial_target"
            break

        current_objective = smooth_target_objective(
            current_scores, target, a.competitor_temperature
        )

        score_grad = {}
        for r in REL:
            sr, gr = sequence_score_and_spatial_grad(
                model=model,
                decoder_layers=decoder_layers,
                batch=batch,
                answer_ids=candidate_ids[r],
                reduction=a.sequence_score_reduction,
                layers=layers,
                geom=geom,
                sub_pos=sub_pos,
                ref_pos=ref_pos,
                base_coords=coords,
            )
            score_grad[r] = (float(sr), gr)

        grad_scores = {r: score_grad[r][0] for r in REL}
        comp_names, comp_weights = smooth_competitor_weights(
            grad_scores, target, a.competitor_temperature
        )
        g_target = score_grad[target][1]
        weighted_comp_grad = np.zeros_like(g_target, dtype=np.float64)
        for r, w in zip(comp_names, comp_weights):
            weighted_comp_grad += float(w) * score_grad[r][1]
        grad2d = g_target - weighted_comp_grad
        direction, raw_grad_norm = normalize_direction(grad2d)
        if direction is None:
            stop_reason = "zero_spatial_target_gradient"
            break

        accepted = False
        trial_logs = []
        accepted_coords = None
        accepted_scores = None
        accepted_step = None
        accepted_cap = False
        for ls in range(a.line_search_tries):
            step_size = float(a.step_natural * (a.line_search_shrink ** ls))
            cand_coords, trial_hit_cap = apply_control_step(
                coords, direction, step_size, a.max_total_natural, a.max_layer_natural
            )
            dc = cand_coords - coords
            step_norm = float(np.linalg.norm(dc.reshape(-1)))
            if step_norm < 1e-10:
                continue
            cand_patch = natural_coords_to_patch(layers, geom, sub_pos, ref_pos, cand_coords)
            cand_scores = all_scores(
                model, batch, candidate_ids, a.sequence_score_reduction,
                decoder_layers, cand_patch,
            )
            cand_obj = smooth_target_objective(cand_scores, target, a.competitor_temperature)
            gain = float(cand_obj - current_objective)
            trial_logs.append({
                "ls": int(ls), "step_size": step_size, "step_norm": step_norm,
                "objective_gain": gain,
            })
            if gain >= a.min_objective_improvement:
                accepted = True
                accepted_coords = cand_coords
                accepted_scores = cand_scores
                accepted_step = (ls, step_size, step_norm)
                accepted_cap = trial_hit_cap
                break

        if not accepted:
            line_search_failed = True
            stop_reason = "line_search_failed"
            step_rows.append({
                **common,
                "step": step_idx,
                "accepted": False,
                "target": target,
                "target_margin_before": float(current_margin),
                "smooth_objective_before": float(current_objective),
                "raw_spatial_target_grad_norm": float(raw_grad_norm),
                "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                "line_search_trials": json.dumps(trial_logs),
                "tf_prediction_after": current_tf_pred,
                "generation_prediction_after": current_gen_pred,
            })
            break

        prev_coords = coords.copy()
        prev_margin = float(current_margin)
        coords = accepted_coords
        current_scores = accepted_scores
        current_tf_pred = argmax_relation(current_scores)
        new_margin = min_target_margin(current_scores, target)
        steps_taken = step_idx
        hit_total_cap = bool(hit_total_cap or accepted_cap)

        patch_now = natural_coords_to_patch(layers, geom, sub_pos, ref_pos, coords)
        current_gen_pred, current_gen_text = gate.generate_with_patch(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            batch=batch,
            patch_map=patch_now,
            max_new_tokens=a.max_new_tokens,
        )

        dc = coords - prev_coords
        row = {
            **common,
            "step": step_idx,
            "accepted": True,
            "target": target,
            "strongest_competitor_before": strongest_competitor(current_scores, target),
            "target_margin_before": prev_margin,
            "target_margin_after": float(new_margin),
            "actual_margin_gain": float(new_margin - prev_margin),
            "smooth_objective_before": float(current_objective),
            "smooth_objective_after": float(
                smooth_target_objective(current_scores, target, a.competitor_temperature)
            ),
            "raw_spatial_target_grad_norm": float(raw_grad_norm),
            "accepted_line_search_index": int(accepted_step[0]),
            "accepted_nominal_step": float(accepted_step[1]),
            "accepted_step_natural": float(accepted_step[2]),
            "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
            "tf_prediction_after": current_tf_pred,
            "generation_prediction_after": current_gen_pred,
            "generation_reached_target": bool(current_gen_pred == target),
            "competitor_weights": json.dumps(
                {r: float(w) for r, w in zip(comp_names, comp_weights)}
            ),
        }
        for i, L in enumerate(layers):
            row[f"step_x_L{L}_H"] = float(dc[i, 0])
            row[f"step_x_L{L}_V"] = float(dc[i, 1])
            row[f"total_x_L{L}_H"] = float(coords[i, 0])
            row[f"total_x_L{L}_V"] = float(coords[i, 1])
        step_rows.append(row)

        if current_gen_pred == target:
            stop_reason = "generation_reached_spatial_target"
            break
        if (
            a.max_total_natural > 0
            and float(np.linalg.norm(coords.reshape(-1))) >= a.max_total_natural - 1e-8
        ):
            hit_total_cap = True
            stop_reason = "total_cap_reached"
            break

    final_patch = natural_coords_to_patch(layers, geom, sub_pos, ref_pos, coords) \
        if float(np.linalg.norm(coords)) > 0 else {}
    final_scores = all_scores(
        model, batch, candidate_ids, a.sequence_score_reduction,
        decoder_layers, final_patch,
    )
    final_tf_pred = argmax_relation(final_scores)
    final_gen_pred, final_gen_text = gate.generate_with_patch(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        batch=batch,
        patch_map=final_patch,
        max_new_tokens=a.max_new_tokens,
    )
    return {
        "patched_prediction": final_gen_pred,
        "patched_generation_text": final_gen_text,
        "final_tf_prediction": final_tf_pred,
        "final_target_margin": float(min_target_margin(final_scores, target)),
        "steps_taken": int(steps_taken),
        "final_total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
        "hit_total_cap": bool(hit_total_cap),
        "line_search_failed": bool(line_search_failed),
        "stop_reason": stop_reason,
        "final_coords": coords,
        "final_scores": final_scores,
    }


def simulate_coverage(result_df: pd.DataFrame, coverages: Sequence[float], eligible_mask):
    df = result_df.copy()
    eligible = df.loc[eligible_mask].sort_values(
        ["gate_score", "spatial_margin", "spatial_agreement", "sid"],
        ascending=[False, False, False, True],
    )
    ranked_sids = eligible["sid"].astype(int).tolist()
    rows = []
    base_correct = df["baseline_correct"].astype(bool).to_numpy()
    gt = df["gt"].astype(str).to_numpy()

    for cov in coverages:
        k = int(math.ceil(float(cov) * len(ranked_sids))) if cov > 0 else 0
        chosen = set(ranked_sids[:k])
        final_pred = np.asarray([
            row.candidate_patched_prediction if int(row.sid) in chosen else row.baseline_prediction
            for row in df.itertuples(index=False)
        ], dtype=object)
        final_correct = final_pred == gt
        w2c = int(np.sum((~base_correct) & final_correct))
        c2w = int(np.sum(base_correct & (~final_correct)))
        chosen_df = df[df["sid"].isin(chosen)] if chosen else df.iloc[0:0]
        rows.append({
            "coverage_of_eligible_conflicts": float(cov),
            "N": len(df),
            "eligible_conflicts_N": len(ranked_sids),
            "edited_N": int(k),
            "edited_fraction_all": float(k / max(len(df), 1)),
            "baseline_accuracy": float(np.mean(base_correct)),
            "final_accuracy": float(np.mean(final_correct)),
            "gain": float(np.mean(final_correct) - np.mean(base_correct)),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "target_correct_rate_edited": float(chosen_df["spatial_target_correct"].mean()) if len(chosen_df) else np.nan,
            "mean_gate_score_edited": float(chosen_df["gate_score"].mean()) if len(chosen_df) else np.nan,
        })
    return pd.DataFrame(rows)


def selected_policy_rows(result_df: pd.DataFrame, eligible_mask, coverage: float):
    eligible = result_df.loc[eligible_mask].sort_values(
        ["gate_score", "spatial_margin", "spatial_agreement", "sid"],
        ascending=[False, False, False, True],
    )
    k = int(math.ceil(float(coverage) * len(eligible))) if coverage > 0 else 0
    chosen = set(eligible.head(k)["sid"].astype(int).tolist())
    out = result_df.copy()
    out["selected_for_repair"] = out["sid"].astype(int).isin(chosen)
    out["final_prediction"] = np.where(
        out["selected_for_repair"], out["candidate_patched_prediction"], out["baseline_prediction"]
    )
    out["final_correct"] = out["final_prediction"].astype(str) == out["gt"].astype(str)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl"
    )
    p.add_argument(
        "--source-spatial-states-npz", required=True,
        help="Synthetic source REAL-NoImage relation-state NPZ. All rows are used to fit frozen spatial geometry.",
    )
    p.add_argument(
        "--target-spatial-states-npz", required=True,
        help="COCO target REAL-NoImage relation-state NPZ. Labels in this NPZ are evaluation-only.",
    )
    p.add_argument("--source-layers", default="20-26")
    p.add_argument("--eval-max-samples", type=int, default=0, help="0 = all target samples")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--gate-score", default="consensus",
        choices=["consensus", "spatial_margin", "agreement", "consensus_over_decision"],
    )
    p.add_argument("--min-spatial-margin", type=float, default=0.0)
    p.add_argument("--min-agreement", type=float, default=0.0)
    p.add_argument(
        "--coverage-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    )
    p.add_argument(
        "--selected-coverage", type=float, default=0.25,
        help="Fixed non-oracle operating point to print by-relation diagnostics.",
    )

    p.add_argument("--required-margin", type=float, default=0.25)
    p.add_argument("--competitor-temperature", type=float, default=0.25)
    p.add_argument("--step-natural", type=float, default=0.75)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--max-total-natural", type=float, default=12.0)
    p.add_argument("--max-layer-natural", type=float, default=0.0)
    p.add_argument("--line-search-shrink", type=float, default=0.5)
    p.add_argument("--line-search-tries", type=int, default=8)
    p.add_argument("--min-objective-improvement", type=float, default=1e-6)

    p.add_argument("--answer-surface", default="above_below", choices=["above_below", "on_under"])
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

    if not (0 <= a.min_agreement <= 1):
        p.error("--min-agreement must be in [0,1]")
    if not (0 <= a.selected_coverage <= 1):
        p.error("--selected-coverage must be in [0,1]")
    if a.step_natural <= 0 or a.max_steps <= 0:
        p.error("step/max-steps must be positive")
    if not (0 < a.line_search_shrink < 1):
        p.error("--line-search-shrink must be in (0,1)")
    if a.competitor_temperature <= 0:
        p.error("--competitor-temperature must be >0")
    return a


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_layers(a.source_layers)
    coverages = parse_float_list(a.coverage_grid)
    if a.selected_coverage not in coverages:
        coverages = sorted(set(coverages + [float(a.selected_coverage)]))

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

    # SOURCE synthetic geometry: every source row is used; there is no target-domain fit.
    X_src, y_src, src_layers, src_sids = load_spatial_npz(Path(a.source_spatial_states_npz))
    src_counts = {r: int(np.sum(y_src == r)) for r in REL}
    missing_rel = [r for r, n in src_counts.items() if n == 0]
    if missing_rel:
        raise RuntimeError(f"Synthetic source has no samples for relations: {missing_rel}; counts={src_counts}")
    src_fit_idx = np.arange(len(y_src), dtype=np.int64)
    geom, axis_df, src_layer_to_i = fit_spatial_geometry_and_selector(
        X_src, y_src, src_layers, src_fit_idx, source_layers
    )
    axis_df.insert(1, "source_domain", "synthetic")
    axis_df.to_csv(outdir / "synthetic_spatial_axis_geometry.csv", index=False)

    # TARGET COCO: relation labels are kept only for post-hoc evaluation.
    X_tgt, y_tgt_eval_only, tgt_layers, tgt_sids = load_spatial_npz(Path(a.target_spatial_states_npz))
    src_dim = int(X_src.shape[-1])
    tgt_dim = int(X_tgt.shape[-1])
    if src_dim != tgt_dim:
        raise RuntimeError(f"Source/target hidden dim mismatch: source={src_dim}, target={tgt_dim}")
    src_layer_set, tgt_layer_set = set(map(int, src_layers)), set(map(int, tgt_layers))
    missing_src = [L for L in source_layers if L not in src_layer_set]
    missing_tgt = [L for L in source_layers if L not in tgt_layer_set]
    if missing_src or missing_tgt:
        raise RuntimeError(
            f"Requested layers missing. source_missing={missing_src}, target_missing={missing_tgt}; "
            f"source_layers={src_layers}; target_layers={tgt_layers}"
        )

    # The geometry object stores source-layer indices.  For target selection we need
    # the target NPZ layer indices while retaining all source-fitted vectors/scales.
    tgt_layer_to_i = {int(L): i for i, L in enumerate(tgt_layers)}
    geom_tgt = {}
    for L in source_layers:
        geom_tgt[L] = dict(geom[L])
        geom_tgt[L]["layer_index"] = int(tgt_layer_to_i[L])

    sid_to_target_spatial_idx = {int(sid): i for i, sid in enumerate(tgt_sids.tolist())}

    two, all_meta, rec_by_sid = load_all_data(a)
    target_sid_set = set(sid_to_target_spatial_idx)
    all_meta = [m for m in all_meta if int(m["sid"]) in target_sid_set]
    eval_meta = stratified_cap(all_meta, a.eval_max_samples, a.seed)
    if not eval_meta:
        raise RuntimeError("No target COCO evaluation samples overlap the target spatial NPZ")

    model = processor = None
    result_rows, step_rows = [], []
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
        print("SYNTHETIC400 -> COCO FULL NON-ORACLE SPATIAL REPAIR")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source layers={source_layers}")
        print(f"synthetic source N={len(X_src)} relation_counts={src_counts}")
        print(f"target COCO N={len(eval_meta)} / target_npz_N={len(X_tgt)}")
        print(f"source spatial NPZ={a.source_spatial_states_npz}")
        print(f"target spatial NPZ={a.target_spatial_states_npz}")
        print(f"gate_score={a.gate_score} min_margin={a.min_spatial_margin} min_agreement={a.min_agreement}")
        print("Synthetic labels define frozen centers/axes/scales. No COCO labels calibrate the method.")
        print("COCO GT is used only after predictions exist, for evaluation metrics.")
        print("=" * 190, flush=True)

        for m in tqdm(eval_meta, desc="EVAL non-oracle spatial repair"):
            sid = int(m["sid"])
            gt = str(m["gt"])  # evaluation only
            image = batch = None
            try:
                sp = spatial_selector_for_sample(X_tgt[sid_to_target_spatial_idx[sid]], geom_tgt, source_layers)
                spatial_pred = sp["prediction"]

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
                    model, batch, candidate_ids, a.sequence_score_reduction,
                    decoder_layers, {},
                )
                base_tf_pred = argmax_relation(base_scores)
                decision_margin = top2_margin(base_scores)
                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map={},
                    max_new_tokens=a.max_new_tokens,
                )

                conflict = spatial_pred != base_gen_pred
                gate_score = compute_gate_score(sp, decision_margin, a.gate_score)
                common = {
                    "sid": sid,
                    "gt": gt,
                    "subject": m["subject"],
                    "reference": m["reference"],
                    "spatial_prediction": spatial_pred,
                    "spatial_target_correct": bool(spatial_pred == gt),
                    "spatial_margin": float(sp["margin"]),
                    "spatial_agreement": float(sp["agreement"]),
                    "decision_margin": float(decision_margin),
                    "gate_score": float(gate_score),
                    "spatial_decision_conflict": bool(conflict),
                    "baseline_prediction": base_gen_pred,
                    "baseline_correct": bool(base_gen_pred == gt),
                    "baseline_tf_prediction": base_tf_pred,
                    "baseline_generation_text": base_gen_text,
                    **{f"spatial_score_{r}": float(sp["aggregate_scores"][r]) for r in REL},
                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                    **{f"layer_pred_L{L}": sp["layer_predictions"][L] for L in source_layers},
                }

                if not conflict:
                    result_rows.append({
                        **common,
                        "candidate_patched_prediction": base_gen_pred,
                        "candidate_patched_correct": bool(base_gen_pred == gt),
                        "candidate_generation_text": base_gen_text,
                        "candidate_steps": 0,
                        "candidate_total_natural_norm": 0.0,
                        "candidate_line_search_failed": False,
                        "candidate_stop_reason": "no_spatial_decision_conflict",
                        "candidate_final_tf_prediction": base_tf_pred,
                        "candidate_final_target_margin": float(min_target_margin(base_scores, spatial_pred)),
                    })
                    continue

                opt = optimize_to_spatial_target(
                    a=a,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    candidate_ids=candidate_ids,
                    layers=source_layers,
                    geom=geom,
                    sub_pos=sub_pos,
                    ref_pos=ref_pos,
                    target=spatial_pred,
                    base_scores=base_scores,
                    base_gen_pred=base_gen_pred,
                    base_gen_text=base_gen_text,
                    common=common,
                    step_rows=step_rows,
                )
                result_rows.append({
                    **common,
                    "candidate_patched_prediction": opt["patched_prediction"],
                    "candidate_patched_correct": bool(opt["patched_prediction"] == gt),
                    "candidate_generation_text": opt["patched_generation_text"],
                    "candidate_steps": int(opt["steps_taken"]),
                    "candidate_total_natural_norm": float(opt["final_total_natural_norm"]),
                    "candidate_line_search_failed": bool(opt["line_search_failed"]),
                    "candidate_stop_reason": opt["stop_reason"],
                    "candidate_final_tf_prediction": opt["final_tf_prediction"],
                    "candidate_final_target_margin": float(opt["final_target_margin"]),
                })

            except Exception as exc:
                append_jsonl(err_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}", flush=True)
            finally:
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        result_df = pd.DataFrame(result_rows)
        step_df = pd.DataFrame(step_rows)
        if len(result_df) == 0:
            raise RuntimeError("No successful samples")
        result_df.to_csv(outdir / "per_sample_candidate_repair.csv", index=False)
        step_df.to_csv(outdir / "per_step.csv", index=False)

        conflict_mask = result_df["spatial_decision_conflict"].astype(bool)
        eligible_mask = (
            conflict_mask
            & (pd.to_numeric(result_df["spatial_margin"], errors="coerce") >= a.min_spatial_margin)
            & (pd.to_numeric(result_df["spatial_agreement"], errors="coerce") >= a.min_agreement)
        )

        coverage_df = simulate_coverage(result_df, coverages, eligible_mask)
        coverage_df.to_csv(outdir / "coverage_curve.csv", index=False)

        selected_df = selected_policy_rows(result_df, eligible_mask, a.selected_coverage)
        selected_df.to_csv(outdir / "selected_policy_per_sample.csv", index=False)

        byrel_rows = []
        for rel, g in selected_df.groupby("gt", sort=False):
            base = g["baseline_correct"].astype(bool).to_numpy()
            fin = g["final_correct"].astype(bool).to_numpy()
            byrel_rows.append({
                "relation": rel,
                "N": len(g),
                "baseline_accuracy": float(np.mean(base)),
                "final_accuracy": float(np.mean(fin)),
                "gain": float(np.mean(fin) - np.mean(base)),
                "wrong_to_correct": int(np.sum((~base) & fin)),
                "correct_to_wrong": int(np.sum(base & (~fin))),
                "selected_N": int(g["selected_for_repair"].sum()),
            })
        byrel_df = pd.DataFrame(byrel_rows)
        byrel_df.to_csv(outdir / "selected_policy_by_relation.csv", index=False)

        # Selector diagnostics use GT only for analysis, never for routing.
        conflicts = result_df.loc[conflict_mask]
        selector_rows = [{
            "N": len(result_df),
            "baseline_accuracy": float(result_df["baseline_correct"].mean()),
            "spatial_selector_accuracy": float(result_df["spatial_target_correct"].mean()),
            "conflict_N": int(conflict_mask.sum()),
            "conflict_fraction": float(conflict_mask.mean()),
            "spatial_target_accuracy_on_conflicts": float(conflicts["spatial_target_correct"].mean()) if len(conflicts) else np.nan,
            "baseline_wrong_N": int((~result_df["baseline_correct"].astype(bool)).sum()),
            "conflicts_on_baseline_wrong_N": int((conflict_mask & (~result_df["baseline_correct"].astype(bool))).sum()),
            "conflicts_on_baseline_correct_N": int((conflict_mask & result_df["baseline_correct"].astype(bool)).sum()),
            "eligible_conflict_N": int(eligible_mask.sum()),
        }]
        selector_df = pd.DataFrame(selector_rows)
        selector_df.to_csv(outdir / "selector_summary.csv", index=False)

        selected_row = coverage_df.iloc[
            int(np.argmin(np.abs(coverage_df["coverage_of_eligible_conflicts"].to_numpy() - a.selected_coverage)))
        ]

        print("\n" + "=" * 190)
        print("SYNTHETIC -> COCO NON-ORACLE SPATIAL SELECTOR")
        print("=" * 190)
        print(selector_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nCOVERAGE -> ACCURACY / W2C / C2W")
        print("-" * 190)
        print(coverage_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nFIXED NON-ORACLE OPERATING POINT: coverage={a.selected_coverage:.2f}")
        print("-" * 190)
        print(pd.DataFrame([selected_row]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nBY RELATION AT FIXED OPERATING POINT")
        print("-" * 190)
        print(byrel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        metadata = {
            "model": a.model,
            "repo": spec.repo_id,
            "decoder_path": decoder_path,
            "source_spatial_states_npz": a.source_spatial_states_npz,
            "target_spatial_states_npz": a.target_spatial_states_npz,
            "source_domain": "synthetic",
            "target_domain": "COCO_two",
            "source_layers": source_layers,
            "synthetic_source_N": int(len(X_src)),
            "synthetic_relation_counts": src_counts,
            "target_npz_N": int(len(X_tgt)),
            "eval_N": int(len(result_df)),
            "gate_score": a.gate_score,
            "min_spatial_margin": a.min_spatial_margin,
            "min_agreement": a.min_agreement,
            "selected_coverage": a.selected_coverage,
            "required_margin": a.required_margin,
            "competitor_temperature": a.competitor_temperature,
            "step_natural": a.step_natural,
            "max_steps": a.max_steps,
            "max_total_natural": a.max_total_natural,
            "note": "Synthetic labels fit frozen spatial geometry. COCO GT is post-hoc evaluation only and is never used for routing/target/optimization.",
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
