#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp.py

Oracle upper bound for spatial -> decision control.

Question
========
If the GT relation is known, and we are allowed to edit ONLY the model's
mid-layer spatial H/V subspace, can we keep pushing the GT decision until it
becomes the highest-scoring answer (and ideally the generated answer)?

Unlike the earlier frozen-A controller, this script does NOT use a learned or
calibrated spatial->decision transport matrix.  At every optimization step it
computes the local gradient of the REAL teacher-forced decision margin with
respect to spatial control coordinates at the CURRENT patched state.

For a layer L, spatial geometry is fit on the calibration split from the
REAL-NoImage relation vectors.  Natural control coordinates x_L=(x_H,x_V)
are realized as a relation-state edit

    dz_L = [half_gap_H * x_H, half_gap_V * x_V]
    delta_r_L = dual_L @ dz_L

and injected symmetrically at object-token positions

    h_sub <- h_sub + delta_r_L/2
    h_ref <- h_ref - delta_r_L/2.

For a layer group G (e.g. 20-26), the optimization variables are ONLY these
spatial coordinates, 2*|G| scalars in free2d mode.  At each step:

  1) measure current four-way sequence scores;
  2) build one smooth multi-competitor objective

         J_tau(x) = S_GT(x) - tau * logsumexp({S_j(x)/tau : j != GT})

     which smoothly approximates S_GT - max_{j!=GT} S_j;
  3) compute its exact local gradient in the spatial control coordinates;
  4) take a normalized step inside the spatial subspace;
  5) backtracking-line-search using THE SAME J_tau objective;
  6) replan from the new state;
  7) stop when GT is highest by the requested margin, generation is repaired,
     or the optimization budget is exhausted.

No model parameter is trained or changed.

The exact stopping criterion is still the true worst-case GT margin

    S_GT - max_{j!=GT} S_j >= required_margin,

so the smooth objective is used only to obtain a stable local ascent direction
across competitor switches.

Control modes
=============
free2d:
    Every controlled layer gets independent H and V coordinates.  This is the
    strongest spatial-subspace oracle upper bound.

gt_axis:
    Every layer gets one non-negative amplitude along the semantic GT axis:
      left=-H, right=+H, above=+V, below=-V.
    This is stricter and answers whether simply moving the spatial state toward
    the GT relation is sufficient.

Primary outputs
===============
baseline.csv
per_step.csv
per_repair.csv
summary.csv
by_relation.csv
layer_usage.csv
spatial_axis_geometry.csv
metadata.json
errors.jsonl

Recommended first upper-bound run
=================================
CUDA_VISIBLE_DEVICES=0 python -u eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp.py \
  --model qwen-3b \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --layer-groups "20-26" \
  --control-modes free2d \
  --spatial-fit-ratio 0.30 \
  --eval-scope heldout \
  --eval-max-samples 80 \
  --required-margin 0.25 \
  --step-natural 0.75 \
  --max-steps 20 \
  --max-total-natural 12.0 \
  --line-search-shrink 0.5 \
  --line-search-tries 5 \
  --stop-on generation \
  --generation-margin-escalation 0.25 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --max-new-tokens 6 \
  --output-dir output/qwen3b_oracle_multilayer_spatial_logit_multicomp_L20_26_n80_v2 \
  --overwrite

Layerwise + window scan example
===============================
  --layer-groups "20;21;22;23;24;25;26;24-25;23-26;20-26"
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


def parse_layer_groups(text: str) -> List[Tuple[str, List[int]]]:
    groups = []
    seen = set()
    for raw in str(text).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        layers = parse_layers(raw)
        key = tuple(layers)
        if key in seen:
            continue
        seen.add(key)
        if len(layers) == 1:
            label = f"L{layers[0]}"
        elif layers == list(range(min(layers), max(layers) + 1)):
            label = f"L{min(layers)}-{max(layers)}"
        else:
            label = "L" + "_".join(map(str, layers))
        groups.append((label, layers))
    if not groups:
        raise ValueError("--layer-groups produced no groups")
    return groups


def parse_modes(text: str) -> List[str]:
    vals = []
    for x in str(text).split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x not in {"free2d", "gt_axis"}:
            raise ValueError(f"Unknown control mode {x!r}; use free2d or gt_axis")
        if x not in vals:
            vals.append(x)
    if not vals:
        raise ValueError("No control modes")
    return vals


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


def argmax_relation(scores: Dict[str, float]) -> str:
    return max(REL, key=lambda r: float(scores[r]))


def strongest_competitor(scores: Dict[str, float], target: str) -> str:
    return max((r for r in REL if r != target), key=lambda r: float(scores[r]))


def min_gt_margin(scores: Dict[str, float], target: str) -> float:
    return min(float(scores[target]) - float(scores[r]) for r in REL if r != target)


def smooth_competitor_weights(scores: Dict[str, float], target: str, tau: float):
    """Softmax weights over the three non-target competitors."""
    comps = [r for r in REL if r != target]
    vals = np.asarray([float(scores[r]) for r in comps], dtype=np.float64) / float(tau)
    vals = vals - float(np.max(vals))
    w = np.exp(vals)
    w = w / max(float(np.sum(w)), EPS)
    return comps, w


def smooth_gt_objective(scores: Dict[str, float], target: str, tau: float) -> float:
    """S_GT - tau*logsumexp(S_comp/tau), a smooth worst-competitor margin."""
    comps = [r for r in REL if r != target]
    vals = np.asarray([float(scores[r]) for r in comps], dtype=np.float64) / float(tau)
    vmax = float(np.max(vals))
    lse = vmax + math.log(float(np.sum(np.exp(vals - vmax))))
    return float(scores[target]) - float(tau) * lse


def stratified_cap(rows: Sequence[dict], n: int, seed: int) -> List[dict]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    return list(gate.traj.stratified_cap(rows, int(n), int(seed)))


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

        B = np.stack([dH, dV], axis=1)  # hidden x 2
        gram = B.T @ B
        dual = B @ np.linalg.inv(gram)
        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)
        natural_M = dual @ np.diag([halfH, halfV])  # hidden x 2; x in natural units

        geom[L] = {
            "layer_index": li,
            "B": B.astype(np.float32),
            "dual": dual.astype(np.float32),
            "natural_M": natural_M.astype(np.float32),
            "natural_half_H": float(halfH),
            "natural_half_V": float(halfV),
        }
        rows.append({
            "source_layer": L,
            "fit_N": len(fit_idx),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "natural_full_gap_H": float(gapH),
            "natural_half_gap_H": float(halfH),
            "natural_full_gap_V": float(gapV),
            "natural_half_gap_V": float(halfV),
            "dual_check_max_abs": float(np.max(np.abs(B.T @ dual - np.eye(2)))),
        })
    return geom, pd.DataFrame(rows)


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


def natural_coords_to_patch(
    layers: Sequence[int],
    geom: dict,
    sub_pos: int,
    ref_pos: int,
    coords: np.ndarray,
):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (len(layers), 2):
        raise ValueError(f"coords shape {coords.shape}, expected {(len(layers), 2)}")
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
    """Patch current natural coordinates plus a differentiable local delta."""

    def __init__(
        self,
        *,
        decoder_layers,
        layers: Sequence[int],
        geom: dict,
        sub_pos: int,
        ref_pos: int,
        base_coords: np.ndarray,
        delta_ctrl: torch.Tensor,
        prompt_len: int,
    ):
        self.handles = []
        self.prompt_len = int(prompt_len)
        self.layers = list(map(int, layers))
        base_coords = np.asarray(base_coords, dtype=np.float32)

        for i, L in enumerate(self.layers):
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
    *,
    model,
    decoder_layers,
    batch,
    answer_ids,
    reduction,
    layers,
    geom,
    sub_pos,
    ref_pos,
    base_coords,
):
    """Return score and gradient wrt a local natural-coordinate delta at base_coords."""
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
            score,
            delta,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

    return float(score.detach().item()), grad.detach().float().cpu().numpy().astype(np.float64)


def gt_axis_direction(gt: str) -> np.ndarray:
    if gt == "left":
        return np.asarray([-1.0, 0.0], dtype=np.float64)
    if gt == "right":
        return np.asarray([+1.0, 0.0], dtype=np.float64)
    if gt == "above":
        return np.asarray([0.0, +1.0], dtype=np.float64)
    if gt == "below":
        return np.asarray([0.0, -1.0], dtype=np.float64)
    raise ValueError(gt)


def projected_gradient_and_coords(
    grad2d: np.ndarray,
    coords: np.ndarray,
    mode: str,
    gt: str,
):
    grad2d = np.asarray(grad2d, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if mode == "free2d":
        return grad2d.copy(), coords.copy(), None

    d = gt_axis_direction(gt)
    amps = coords @ d
    amps = np.maximum(amps, 0.0)
    g_amp = grad2d @ d
    return g_amp, amps, d


def apply_control_step(
    coords: np.ndarray,
    direction,
    step: float,
    mode: str,
    gt: str,
    max_total: float,
    max_layer: float,
):
    coords = np.asarray(coords, dtype=np.float64)
    if mode == "free2d":
        cand = coords + float(step) * np.asarray(direction, dtype=np.float64)
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

    d = gt_axis_direction(gt)
    amps = np.maximum(coords @ d, 0.0)
    amps = amps + float(step) * np.asarray(direction, dtype=np.float64)
    amps = np.maximum(amps, 0.0)
    if max_layer > 0:
        amps = np.minimum(amps, max_layer)
    nall = float(np.linalg.norm(amps))
    hit_total = False
    if max_total > 0 and nall > max_total:
        amps *= max_total / max(nall, EPS)
        hit_total = True
    cand = amps[:, None] * d[None, :]
    return cand, hit_total


def effective_grad_norm(g) -> float:
    return float(np.linalg.norm(np.asarray(g, dtype=np.float64).reshape(-1)))


def normalize_direction(g):
    g = np.asarray(g, dtype=np.float64)
    n = effective_grad_norm(g)
    if n < EPS:
        return None, n
    return g / n, n


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
    p.add_argument(
        "--layer-groups",
        default="20-26",
        help='Semicolon-separated layer groups, e.g. "25;24-25;23-26;20-26".',
    )
    p.add_argument(
        "--control-modes",
        default="free2d",
        help="Comma-separated: free2d,gt_axis",
    )
    p.add_argument("--spatial-fit-ratio", type=float, default=0.30)
    p.add_argument("--spatial-fit-seed", type=int, default=1)
    p.add_argument("--eval-scope", default="heldout", choices=["heldout", "all_data"])
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--required-margin", type=float, default=0.25,
        help="GT must exceed every competitor by this teacher-forced sequence-score margin.",
    )
    p.add_argument(
        "--step-natural", type=float, default=0.75,
        help="Initial global natural-coordinate step length per iteration.",
    )
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument(
        "--max-total-natural", type=float, default=12.0,
        help="Global L2 cap across all controlled natural coordinates; <=0 disables.",
    )
    p.add_argument(
        "--max-layer-natural", type=float, default=0.0,
        help="Optional per-layer natural-coordinate L2 cap; <=0 disables.",
    )
    p.add_argument("--line-search-shrink", type=float, default=0.5)
    p.add_argument("--line-search-tries", type=int, default=8)
    p.add_argument(
        "--min-objective-improvement", type=float, default=1e-6,
        help="Required improvement in the SAME smooth multi-competitor objective for accepting a step.",
    )
    p.add_argument(
        "--competitor-temperature", type=float, default=0.25,
        help="Temperature tau for smooth max over the three non-GT competitors; smaller approaches hard max.",
    )
    p.add_argument(
        "--stop-on", default="generation", choices=["generation", "teacher_forced", "either"],
    )
    p.add_argument(
        "--generation-margin-escalation", type=float, default=0.25,
        help=(
            "When stop-on includes generation and GT is already TF-top but generation is still wrong, "
            "continue widening the GT margin instead of stopping at a zero gradient request."
        ),
    )
    p.add_argument(
        "--generation-check-every", type=int, default=1,
        help="Check free generation every N accepted optimization steps.",
    )

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
    if not (0 < a.spatial_fit_ratio < 1):
        p.error("--spatial-fit-ratio must be in (0,1)")
    if a.required_margin < 0:
        p.error("--required-margin must be >=0")
    if a.step_natural <= 0:
        p.error("--step-natural must be >0")
    if a.max_steps <= 0:
        p.error("--max-steps must be >0")
    if not (0 < a.line_search_shrink < 1):
        p.error("--line-search-shrink must be in (0,1)")
    if a.line_search_tries <= 0:
        p.error("--line-search-tries must be >0")
    if a.generation_margin_escalation < 0:
        p.error("--generation-margin-escalation must be >=0")
    if a.competitor_temperature <= 0:
        p.error("--competitor-temperature must be >0")
    if a.min_objective_improvement < 0:
        p.error("--min-objective-improvement must be >=0")
    if a.generation_check_every <= 0:
        p.error("--generation-check-every must be >0")
    return a


def stop_reached(stop_on: str, gen_correct: bool, tf_correct: bool) -> bool:
    if stop_on == "generation":
        return bool(gen_correct)
    if stop_on == "teacher_forced":
        return bool(tf_correct)
    return bool(gen_correct or tf_correct)


def summarize(repair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, mode), g in repair_df.groupby(["layer_group", "control_mode"], sort=False):
        base = g["baseline_correct"].astype(bool).to_numpy()
        patch = g["patched_correct"].astype(bool).to_numpy()
        wrong = ~base
        gw = g.loc[wrong]
        w2c = int(np.sum(wrong & patch))
        c2w = int(np.sum(base & (~patch)))
        rows.append({
            "layer_group": group,
            "control_mode": mode,
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
            "tf_target_rate_on_wrong": float((gw["final_tf_prediction"] == gw["gt"]).mean()) if len(gw) else np.nan,
            "mean_steps_wrong": float(gw["steps_taken"].mean()) if len(gw) else np.nan,
            "median_steps_wrong": float(gw["steps_taken"].median()) if len(gw) else np.nan,
            "mean_final_total_natural_wrong": float(gw["final_total_natural_norm"].mean()) if len(gw) else np.nan,
            "fraction_hit_total_cap_wrong": float(gw["hit_total_cap"].mean()) if len(gw) else np.nan,
            "fraction_line_search_failed_wrong": float(gw["line_search_failed"].mean()) if len(gw) else np.nan,
            "mean_final_gt_margin_wrong": float(gw["final_gt_min_margin"].mean()) if len(gw) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_relation(repair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, mode, rel), g in repair_df.groupby(["layer_group", "control_mode", "gt"], sort=False):
        base = g["baseline_correct"].astype(bool).to_numpy()
        patch = g["patched_correct"].astype(bool).to_numpy()
        wrong = ~base
        gw = g.loc[wrong]
        w2c = int(np.sum(wrong & patch))
        rows.append({
            "layer_group": group,
            "control_mode": mode,
            "relation": rel,
            "N": len(g),
            "baseline_accuracy": float(np.mean(base)),
            "patched_accuracy": float(np.mean(patch)),
            "gain": float(np.mean(patch) - np.mean(base)),
            "baseline_wrong_N": int(np.sum(wrong)),
            "wrong_to_correct": w2c,
            "repair_rate_on_wrong": float(w2c / max(int(np.sum(wrong)), 1)),
            "tf_target_rate_on_wrong": float((gw["final_tf_prediction"] == gw["gt"]).mean()) if len(gw) else np.nan,
            "mean_steps_wrong": float(gw["steps_taken"].mean()) if len(gw) else np.nan,
            "mean_final_total_natural_wrong": float(gw["final_total_natural_norm"].mean()) if len(gw) else np.nan,
            "mean_final_gt_margin_wrong": float(gw["final_gt_min_margin"].mean()) if len(gw) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_layer_usage(repair_df: pd.DataFrame, all_layers: Sequence[int]) -> pd.DataFrame:
    rows = []
    for (group, mode), g in repair_df.groupby(["layer_group", "control_mode"], sort=False):
        gw = g.loc[~g["baseline_correct"].astype(bool)]
        if len(gw) == 0:
            continue
        for L in all_layers:
            hcol = f"final_x_L{L}_H"
            vcol = f"final_x_L{L}_V"
            if hcol not in gw.columns or vcol not in gw.columns:
                continue
            h = pd.to_numeric(gw[hcol], errors="coerce")
            v = pd.to_numeric(gw[vcol], errors="coerce")
            valid = h.notna() & v.notna()
            if not valid.any():
                continue
            norms = np.sqrt(h[valid].to_numpy() ** 2 + v[valid].to_numpy() ** 2)
            rows.append({
                "layer_group": group,
                "control_mode": mode,
                "source_layer": int(L),
                "N_wrong": int(valid.sum()),
                "mean_abs_x_H": float(np.mean(np.abs(h[valid]))),
                "mean_abs_x_V": float(np.mean(np.abs(v[valid]))),
                "mean_layer_natural_norm": float(np.mean(norms)),
                "median_layer_natural_norm": float(np.median(norms)),
            })
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    groups = parse_layer_groups(a.layer_groups)
    modes = parse_modes(a.control_modes)
    all_source_layers = sorted(set(L for _, ls in groups for L in ls))

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

    X, y, spatial_layers, spatial_sids = load_spatial_npz(Path(a.spatial_states_npz))
    fit_idx, held_idx, fit_sids, held_sids = make_spatial_split(
        y, spatial_sids, a.spatial_fit_ratio, a.spatial_fit_seed
    )
    geom, axis_df = fit_spatial_geometry(
        X, y, spatial_layers, fit_idx, all_source_layers
    )
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

    model = processor = None
    baseline_rows, repair_rows, step_rows = [], [], []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        if max(all_source_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(all_source_layers)} but model has {len(decoder_layers)} layers"
            )
        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)
        device = torch.device(a.device)

        print("\n" + "=" * 190)
        print("ORACLE MULTI-LAYER SPATIAL -> DECISION OPTIMIZATION")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"layer groups={[label for label, _ in groups]}")
        print(f"control modes={modes}")
        print(f"spatial fit/heldout={len(fit_idx)}/{len(held_idx)}")
        print(f"evaluation N={len(eval_meta)} scope={a.eval_scope}")
        print(
            f"required_margin={a.required_margin} step={a.step_natural} "
            f"max_steps={a.max_steps} max_total={a.max_total_natural}"
        )
        print(
            "No frozen A. Each step backpropagates one smooth GT-vs-ALL-competitors objective "
            f"(tau={a.competitor_temperature})."
        )
        print("Baseline-correct generations are preserved and never edited.")
        print("=" * 190, flush=True)

        for m in tqdm(eval_meta, desc="EVAL oracle spatial optimization"):
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

                base_patch = {}
                base_scores = all_scores(
                    model, batch, candidate_ids, a.sequence_score_reduction,
                    decoder_layers, base_patch,
                )
                base_tf_pred = argmax_relation(base_scores)
                base_gt_margin = min_gt_margin(base_scores, gt)
                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map=base_patch,
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = base_gen_pred == gt

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

                for group_label, layers in groups:
                    for mode in modes:
                        common = {
                            "sid": sid,
                            "gt": gt,
                            "layer_group": group_label,
                            "control_mode": mode,
                            "layers": ",".join(map(str, layers)),
                        }

                        if base_correct:
                            row = {
                                **common,
                                "baseline_prediction": base_gen_pred,
                                "patched_prediction": base_gen_pred,
                                "baseline_correct": True,
                                "patched_correct": True,
                                "baseline_tf_prediction": base_tf_pred,
                                "final_tf_prediction": base_tf_pred,
                                "baseline_gt_min_margin": float(base_gt_margin),
                                "final_gt_min_margin": float(base_gt_margin),
                                "steps_taken": 0,
                                "final_total_natural_norm": 0.0,
                                "hit_total_cap": False,
                                "line_search_failed": False,
                                "stop_reason": "baseline_correct_preserved",
                                "baseline_generation_text": base_gen_text,
                                "patched_generation_text": base_gen_text,
                            }
                            for L in all_source_layers:
                                row[f"final_x_L{L}_H"] = 0.0 if L in layers else np.nan
                                row[f"final_x_L{L}_V"] = 0.0 if L in layers else np.nan
                            repair_rows.append(row)
                            continue

                        coords = np.zeros((len(layers), 2), dtype=np.float64)
                        current_scores = dict(base_scores)
                        current_tf_pred = base_tf_pred
                        current_gen_pred = base_gen_pred
                        current_gen_text = base_gen_text
                        hit_total_cap = False
                        line_search_failed = False
                        stop_reason = "max_steps"
                        steps_taken = 0

                        for step_idx in range(1, a.max_steps + 1):
                            current_margin = min_gt_margin(current_scores, gt)
                            tf_ready = current_margin >= a.required_margin
                            gen_ready = current_gen_pred == gt
                            if stop_reached(a.stop_on, gen_ready, tf_ready):
                                stop_reason = (
                                    "generation_repaired" if gen_ready else "teacher_forced_target"
                                )
                                break

                            # If TF is already target but generation is not, keep widening the
                            # margin.  The actual gradient objective remains GT-vs-strongest competitor.
                            effective_required_margin = float(a.required_margin)
                            if (
                                a.stop_on in {"generation", "either"}
                                and not gen_ready
                                and tf_ready
                                and a.generation_margin_escalation > 0
                            ):
                                effective_required_margin = float(
                                    current_margin + a.generation_margin_escalation
                                )

                            competitor = strongest_competitor(current_scores, gt)
                            current_objective = smooth_gt_objective(
                                current_scores, gt, a.competitor_temperature
                            )

                            # Exact local gradient of ONE CONSISTENT multi-competitor objective:
                            #   J = S_GT - tau*logsumexp(S_comp/tau).
                            # This avoids the old top-1-gradient / worst-margin-line-search mismatch.
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
                                grad_scores, gt, a.competitor_temperature
                            )
                            s_gt, g_gt = score_grad[gt]
                            weighted_comp_grad = np.zeros_like(g_gt, dtype=np.float64)
                            weighted_comp_score = 0.0
                            for r, w in zip(comp_names, comp_weights):
                                weighted_comp_grad += float(w) * score_grad[r][1]
                                weighted_comp_score += float(w) * score_grad[r][0]
                            grad2d = g_gt - weighted_comp_grad
                            projected_g, current_param, semantic_dir = projected_gradient_and_coords(
                                grad2d, coords, mode, gt
                            )
                            direction, raw_grad_norm = normalize_direction(projected_g)
                            if direction is None:
                                stop_reason = "zero_spatial_margin_gradient"
                                break

                            accepted = False
                            accepted_scores = None
                            accepted_coords = None
                            accepted_step = None
                            accepted_total_cap = False
                            trial_logs = []

                            for ls in range(a.line_search_tries):
                                step_size = float(a.step_natural * (a.line_search_shrink ** ls))
                                cand_coords, trial_hit_cap = apply_control_step(
                                    coords=coords,
                                    direction=direction,
                                    step=step_size,
                                    mode=mode,
                                    gt=gt,
                                    max_total=a.max_total_natural,
                                    max_layer=a.max_layer_natural,
                                )
                                actual_delta_coords = cand_coords - coords
                                actual_step_norm = float(np.linalg.norm(actual_delta_coords.reshape(-1)))
                                if actual_step_norm < 1e-10:
                                    trial_logs.append({
                                        "ls": int(ls),
                                        "step_size": float(step_size),
                                        "min_gt_margin": None,
                                        "smooth_objective": None,
                                        "objective_gain": None,
                                        "step_norm": float(actual_step_norm),
                                        "accepted": False,
                                    })
                                    continue

                                cand_patch = natural_coords_to_patch(
                                    layers, geom, sub_pos, ref_pos, cand_coords
                                )
                                cand_scores = all_scores(
                                    model, batch, candidate_ids, a.sequence_score_reduction,
                                    decoder_layers, cand_patch,
                                )
                                cand_margin = min_gt_margin(cand_scores, gt)
                                cand_objective = smooth_gt_objective(
                                    cand_scores, gt, a.competitor_temperature
                                )
                                objective_gain = float(cand_objective - current_objective)
                                improved = objective_gain >= a.min_objective_improvement
                                trial_logs.append({
                                    "ls": int(ls),
                                    "step_size": float(step_size),
                                    "min_gt_margin": float(cand_margin),
                                    "smooth_objective": float(cand_objective),
                                    "objective_gain": float(objective_gain),
                                    "step_norm": float(actual_step_norm),
                                    "accepted": bool(improved),
                                })
                                if improved:
                                    accepted = True
                                    accepted_scores = cand_scores
                                    accepted_coords = cand_coords
                                    accepted_step = (ls, step_size, actual_step_norm)
                                    accepted_total_cap = trial_hit_cap
                                    break

                            if not accepted:
                                line_search_failed = True
                                stop_reason = "line_search_failed"
                                step_rows.append({
                                    **common,
                                    "step": step_idx,
                                    "accepted": False,
                                    "competitor": competitor,
                                    "gt_margin_before": float(current_margin),
                                    "gt_margin_after": float(current_margin),
                                    "smooth_objective_before": float(current_objective),
                                    "smooth_objective_after": float(current_objective),
                                    "actual_objective_gain": 0.0,
                                    "actual_margin_gain": 0.0,
                                    "raw_spatial_margin_grad_norm": float(raw_grad_norm),
                                    "competitor_weights": json.dumps(
                                        {r: float(w) for r, w in zip(comp_names, comp_weights)}
                                    ),
                                    "accepted_step_natural": 0.0,
                                    "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                                    "line_search_trials": json.dumps(trial_logs),
                                    "tf_prediction_after": current_tf_pred,
                                    "generation_prediction_after": current_gen_pred,
                                    "effective_required_margin": effective_required_margin,
                                })
                                break

                            prev_coords = coords.copy()
                            prev_margin = float(current_margin)
                            coords = accepted_coords
                            current_scores = accepted_scores
                            current_tf_pred = argmax_relation(current_scores)
                            new_margin = min_gt_margin(current_scores, gt)
                            hit_total_cap = bool(hit_total_cap or accepted_total_cap)
                            steps_taken = step_idx

                            # First-order predicted gain for the smooth multi-competitor objective.
                            dc = coords - prev_coords
                            predicted_gain = float(np.sum(grad2d * dc))
                            actual_gain = float(new_margin - prev_margin)

                            run_generation = (
                                step_idx % a.generation_check_every == 0
                                or current_tf_pred == gt
                                or step_idx == a.max_steps
                                or hit_total_cap
                            )
                            if run_generation:
                                patch_now = natural_coords_to_patch(
                                    layers, geom, sub_pos, ref_pos, coords
                                )
                                current_gen_pred, current_gen_text = gate.generate_with_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=batch,
                                    patch_map=patch_now,
                                    max_new_tokens=a.max_new_tokens,
                                )

                            row = {
                                **common,
                                "step": step_idx,
                                "accepted": True,
                                "competitor": competitor,
                                "gt_score_at_grad": float(s_gt),
                                "strongest_competitor_at_grad": competitor,
                                "weighted_competitor_score_at_grad": float(weighted_comp_score),
                                "competitor_weights": json.dumps(
                                    {r: float(w) for r, w in zip(comp_names, comp_weights)}
                                ),
                                "gt_margin_before": prev_margin,
                                "gt_margin_after": float(new_margin),
                                "smooth_objective_before": float(current_objective),
                                "smooth_objective_after": float(
                                    smooth_gt_objective(current_scores, gt, a.competitor_temperature)
                                ),
                                "predicted_local_objective_gain": predicted_gain,
                                "actual_objective_gain": float(
                                    smooth_gt_objective(current_scores, gt, a.competitor_temperature)
                                    - current_objective
                                ),
                                "actual_margin_gain": actual_gain,
                                "raw_spatial_margin_grad_norm": float(raw_grad_norm),
                                "accepted_line_search_index": int(accepted_step[0]),
                                "accepted_nominal_step": float(accepted_step[1]),
                                "accepted_step_natural": float(accepted_step[2]),
                                "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                                "hit_total_cap": bool(accepted_total_cap),
                                "line_search_trials": json.dumps(trial_logs),
                                "tf_prediction_after": current_tf_pred,
                                "tf_target_after": bool(current_tf_pred == gt),
                                "generation_prediction_after": current_gen_pred if run_generation else None,
                                "generation_target_after": (
                                    bool(current_gen_pred == gt) if run_generation else np.nan
                                ),
                                "effective_required_margin": effective_required_margin,
                                **{f"score_after_{r}": float(current_scores[r]) for r in REL},
                            }
                            for i, L in enumerate(layers):
                                row[f"step_x_L{L}_H"] = float(dc[i, 0])
                                row[f"step_x_L{L}_V"] = float(dc[i, 1])
                                row[f"total_x_L{L}_H"] = float(coords[i, 0])
                                row[f"total_x_L{L}_V"] = float(coords[i, 1])
                                row[f"grad_L{L}_H"] = float(grad2d[i, 0])
                                row[f"grad_L{L}_V"] = float(grad2d[i, 1])
                            step_rows.append(row)

                            tf_ready = new_margin >= a.required_margin
                            gen_ready = current_gen_pred == gt
                            if stop_reached(a.stop_on, gen_ready, tf_ready):
                                stop_reason = (
                                    "generation_repaired" if gen_ready else "teacher_forced_target"
                                )
                                break
                            if (
                                a.max_total_natural > 0
                                and float(np.linalg.norm(coords.reshape(-1))) >= a.max_total_natural - 1e-8
                            ):
                                stop_reason = "total_cap_reached"
                                hit_total_cap = True
                                break

                        # Final exact scores + final free generation for this group/mode.
                        final_patch = natural_coords_to_patch(
                            layers, geom, sub_pos, ref_pos, coords
                        ) if np.linalg.norm(coords) > 0 else {}
                        final_scores = all_scores(
                            model, batch, candidate_ids, a.sequence_score_reduction,
                            decoder_layers, final_patch,
                        )
                        final_tf_pred = argmax_relation(final_scores)
                        final_margin = min_gt_margin(final_scores, gt)
                        final_gen_pred, final_gen_text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=batch,
                            patch_map=final_patch,
                            max_new_tokens=a.max_new_tokens,
                        )

                        row = {
                            **common,
                            "baseline_prediction": base_gen_pred,
                            "patched_prediction": final_gen_pred,
                            "baseline_correct": bool(base_correct),
                            "patched_correct": bool(final_gen_pred == gt),
                            "baseline_tf_prediction": base_tf_pred,
                            "final_tf_prediction": final_tf_pred,
                            "baseline_gt_min_margin": float(base_gt_margin),
                            "final_gt_min_margin": float(final_margin),
                            "steps_taken": int(steps_taken),
                            "final_total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                            "hit_total_cap": bool(hit_total_cap),
                            "line_search_failed": bool(line_search_failed),
                            "stop_reason": stop_reason,
                            "baseline_generation_text": base_gen_text,
                            "patched_generation_text": final_gen_text,
                            **{f"final_score_{r}": float(final_scores[r]) for r in REL},
                        }
                        for L in all_source_layers:
                            if L in layers:
                                i = layers.index(L)
                                row[f"final_x_L{L}_H"] = float(coords[i, 0])
                                row[f"final_x_L{L}_V"] = float(coords[i, 1])
                            else:
                                row[f"final_x_L{L}_H"] = np.nan
                                row[f"final_x_L{L}_V"] = np.nan
                        repair_rows.append(row)

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

        baseline_df = pd.DataFrame(baseline_rows)
        repair_df = pd.DataFrame(repair_rows)
        step_df = pd.DataFrame(step_rows)
        baseline_df.to_csv(outdir / "baseline.csv", index=False)
        repair_df.to_csv(outdir / "per_repair.csv", index=False)
        step_df.to_csv(outdir / "per_step.csv", index=False)

        if len(repair_df) == 0:
            raise RuntimeError("No successful repair rows")

        summary_df = summarize(repair_df)
        relation_df = summarize_relation(repair_df)
        usage_df = summarize_layer_usage(repair_df, all_source_layers)
        summary_df.to_csv(outdir / "summary.csv", index=False)
        relation_df.to_csv(outdir / "by_relation.csv", index=False)
        usage_df.to_csv(outdir / "layer_usage.csv", index=False)

        metadata = {
            "model": a.model,
            "repo": spec.repo_id,
            "decoder_path": decoder_path,
            "spatial_states_npz": str(a.spatial_states_npz),
            "layer_groups": [{"label": x, "layers": ls} for x, ls in groups],
            "control_modes": modes,
            "spatial_fit_ratio": a.spatial_fit_ratio,
            "spatial_fit_seed": a.spatial_fit_seed,
            "spatial_fit_N": int(len(fit_idx)),
            "spatial_heldout_N": int(len(held_idx)),
            "eval_scope": a.eval_scope,
            "eval_N_successful_baseline": int(len(baseline_df)),
            "required_margin": a.required_margin,
            "competitor_temperature": a.competitor_temperature,
            "min_objective_improvement": a.min_objective_improvement,
            "step_natural": a.step_natural,
            "max_steps": a.max_steps,
            "max_total_natural": a.max_total_natural,
            "max_layer_natural": a.max_layer_natural,
            "stop_on": a.stop_on,
            "answer_surface": a.answer_surface,
            "sequence_score_reduction": a.sequence_score_reduction,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\n" + "=" * 190)
        print("ORACLE SPATIAL OPTIMIZATION SUMMARY")
        print("=" * 190)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nBY RELATION")
        print("-" * 190)
        print(relation_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if len(usage_df):
            print("\nLAYER USAGE ON BASELINE-WRONG SAMPLES")
            print("-" * 190)
            print(usage_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        lines = [
            "Oracle multi-layer spatial->decision optimization",
            "",
            "GT is used only as the target relation.",
            "All edits are constrained to fitted per-layer spatial H/V subspaces.",
            "No frozen transport A is used; each step backpropagates a smooth GT-vs-all-competitors objective.",
            "Baseline-correct generations are not edited.",
            "",
            summary_df.to_string(index=False),
        ]
        (outdir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")

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
