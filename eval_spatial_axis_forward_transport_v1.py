#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spatial_axis_forward_transport_v1.py

Direct finite-intervention test of the proposed spatial -> decision transport.

Question
========
The back-projected writer experiment found structured alignment, but cosine around
0.35-0.45 does not by itself prove that the mid-layer object spatial code is
actually transported into the final answer decision.

This script therefore avoids cosine as the primary endpoint.  At a source layer L
(default L24,L25), fit two frozen REAL-NoImage object-relation axes on a calibration
split:

    d_H = unit(d_right - d_left)
    d_V = unit(d_above - d_below)

For an evaluation sample, directly edit the REAL object-token states:

    h_sub <- h_sub + sign * amplitude/2 * d_axis
    h_ref <- h_ref - sign * amplitude/2 * d_axis

so the relation state changes by exactly

    (h_sub - h_ref) <- (h_sub - h_ref) + sign * amplitude * d_axis.

Then continue the ordinary Transformer forward pass and re-score the actual four
answer sequences.  Define decision margins

    M_H = S_right - S_left
    M_V = S_above - S_below

where S_r is the teacher-forced answer sequence log-probability (mean by default).

A selective spatial->decision transport bridge predicts:

    +d_H : Delta M_H > 0,  |Delta M_H| >> |Delta M_V|
    -d_H : Delta M_H < 0,  |Delta M_H| >> |Delta M_V|
    +d_V : Delta M_V > 0,  |Delta M_V| >> |Delta M_H|
    -d_V : Delta M_V < 0,  |Delta M_V| >> |Delta M_H|

The cleanest summary uses central finite differences, per sample:

    T_HH = [M_H(+H)-M_H(-H)] / (2*amplitude_H)
    T_HV = [M_V(+H)-M_V(-H)] / (2*amplitude_H)
    T_VH = [M_H(+V)-M_H(-V)] / (2*amplitude_V)
    T_VV = [M_V(+V)-M_V(-V)] / (2*amplitude_V)

Thus the empirical 2x2 forward transport matrix is

          output H    output V
    in H    T_HH       T_HV
    in V    T_VH       T_VV

Evidence for a spatially selective bridge is diagonal dominance:

    T_HH > 0, T_VV > 0,
    diagonal magnitude >> cross-axis magnitude.

Scale modes
===========
natural (default):
    amplitude_H at each layer is half of the calibration-set separation between
    mean RIGHT and LEFT projections on d_H.  amplitude_V is defined analogously.
    --scales are multipliers of this natural half-gap.

unit:
    amplitude = --scale directly in residual-stream units.

This is a MECHANISM experiment.  Relation labels are used to fit the frozen spatial
axes on the calibration split, but evaluation interventions do not use each sample's
GT relation.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_spatial_axis_forward_transport_v1.py \
  --model qwen-3b \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --source-layers 24,25 \
  --spatial-fit-ratio 0.30 \
  --eval-scope heldout \
  --eval-max-samples 80 \
  --scale-mode natural \
  --scales 1.0 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --output-dir output/qwen3b_spatial_axis_forward_transport_n80_v1 \
  --overwrite

Useful follow-up after the smoke test:
    --scales 0.25,0.5,1.0,2.0

Main outputs
============
spatial_axis_calibration.csv
per_condition_scores.csv
per_sample_transport_matrix.csv
transport_summary.csv
condition_summary.csv
analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
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
EPS = 1e-12


# =============================================================================
# CLI / generic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--eval-scope",
        default="heldout",
        choices=["heldout", "all_data"],
    )
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--scale-mode",
        default="natural",
        choices=["natural", "unit"],
    )
    p.add_argument(
        "--scales",
        default="1.0",
        help="Comma-separated multipliers. In natural mode, 1.0 = one calibration half-gap.",
    )
    p.add_argument(
        "--answer-surface",
        default="above_below",
        choices=["above_below", "on_under"],
    )
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument(
        "--sequence-score-reduction",
        default="mean",
        choices=["mean", "sum"],
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


def parse_floats(s: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals or any(x <= 0 for x in vals):
        raise ValueError("--scales must contain positive numbers")
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
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return (v / n).astype(np.float32)


def safe_mean(x):
    x = np.asarray(list(x), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def safe_median(x):
    x = np.asarray(list(x), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def stratified_cap(rows: Sequence[dict], n: int, seed: int) -> List[dict]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    return list(gate.traj.stratified_cap(rows, int(n), int(seed)))


# =============================================================================
# Spatial codebook / split
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
        raise RuntimeError(f"Bad spatial NPZ shapes: X={X.shape}, y={y.shape}, sid={sids.shape}")
    return X, y, layers, sids


def make_spatial_split(y, sids, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    fit, held = [], []
    for r in REL:
        idx = np.where(y == r)[0]
        if len(idx) < 2:
            raise RuntimeError(f"Need >=2 samples for relation {r}, got {len(idx)}")
        idx = idx.copy()
        rng.shuffle(idx)
        nfit = max(1, min(len(idx) - 1, int(round(len(idx) * ratio))))
        fit.extend(idx[:nfit].tolist())
        held.extend(idx[nfit:].tolist())
    fit = np.asarray(sorted(fit), dtype=np.int64)
    held = np.asarray(sorted(held), dtype=np.int64)
    return fit, held, set(map(int, sids[fit])), set(map(int, sids[held]))


def fit_axes(X, y, all_layers, fit_idx, source_layers):
    layer_to_i = {int(L): i for i, L in enumerate(all_layers)}
    missing = [L for L in source_layers if L not in layer_to_i]
    if missing:
        raise RuntimeError(f"Spatial NPZ does not contain layers {missing}; has {all_layers}")

    axes = {}
    rows = []
    for L in source_layers:
        li = layer_to_i[L]
        Xf = X[fit_idx, li].astype(np.float64)
        yf = y[fit_idx]
        center = Xf.mean(axis=0)
        means = {}
        dirs = {}
        for r in REL:
            mu = Xf[yf == r].mean(axis=0)
            means[r] = mu
            dirs[r] = unit(mu - center)

        dH = unit(dirs["right"] - dirs["left"])
        dV = unit(dirs["above"] - dirs["below"])

        projH = Xf @ dH.astype(np.float64)
        projV = Xf @ dV.astype(np.float64)
        mean_R = float(projH[yf == "right"].mean())
        mean_L = float(projH[yf == "left"].mean())
        mean_A = float(projV[yf == "above"].mean())
        mean_B = float(projV[yf == "below"].mean())
        gapH = mean_R - mean_L
        gapV = mean_A - mean_B

        # Orient axes so +H means more RIGHT and +V means more ABOVE.
        if gapH < 0:
            dH = -dH
            gapH = -gapH
            mean_R, mean_L = -mean_R, -mean_L
        if gapV < 0:
            dV = -dV
            gapV = -gapV
            mean_A, mean_B = -mean_A, -mean_B

        halfH = max(float(gapH) / 2.0, EPS)
        halfV = max(float(gapV) / 2.0, EPS)
        axes[L] = {
            "H": dH.astype(np.float32),
            "V": dV.astype(np.float32),
            "natural_amp_H": halfH,
            "natural_amp_V": halfV,
        }
        rows.append({
            "source_layer": L,
            "fit_N": len(fit_idx),
            "natural_full_gap_H": float(gapH),
            "natural_half_gap_H": halfH,
            "natural_full_gap_V": float(gapV),
            "natural_half_gap_V": halfV,
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "mean_proj_right_H": mean_R,
            "mean_proj_left_H": mean_L,
            "mean_proj_above_V": mean_A,
            "mean_proj_below_V": mean_B,
        })
    return axes, pd.DataFrame(rows)


# =============================================================================
# Dataset / token positions / scoring
# =============================================================================

def load_all_data(a):
    # gate.load_data expects max_samples/eval_max_samples attributes and otherwise
    # gives us the exact repository COCO/question path used by prior experiments.
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
    out = {}
    for r in REL:
        out[r] = gate.sequence_score(
            model=model,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
            decoder_layers=decoder_layers,
            patch_map=patch_map,
        )
    return out


def margins(scores: Dict[str, float]):
    return {
        "H": float(scores["right"] - scores["left"]),
        "V": float(scores["above"] - scores["below"]),
    }


def argmax_relation(scores):
    return max(REL, key=lambda r: float(scores[r]))


def make_object_axis_patch(L, sub_pos, ref_pos, axis_vec, signed_amplitude):
    delta = float(signed_amplitude) * np.asarray(axis_vec, dtype=np.float32)
    return {
        int(L): {
            int(sub_pos): (+0.5 * delta).astype(np.float32),
            int(ref_pos): (-0.5 * delta).astype(np.float32),
        }
    }


# =============================================================================
# Summaries
# =============================================================================

def summarize_conditions(df: pd.DataFrame):
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    gcols = ["source_layer", "scale", "axis", "sign"]
    for key, g in df.groupby(gcols, dropna=False):
        L, scale, axis, sign = key
        target_col = "delta_margin_H" if axis == "H" else "delta_margin_V"
        cross_col = "delta_margin_V" if axis == "H" else "delta_margin_H"
        oriented = float(sign) * g[target_col].astype(float)
        rows.append({
            "source_layer": int(L),
            "scale": float(scale),
            "axis": axis,
            "sign": int(sign),
            "N": len(g),
            "mean_target_delta": float(g[target_col].mean()),
            "mean_oriented_target_delta": float(oriented.mean()),
            "median_oriented_target_delta": float(oriented.median()),
            "fraction_target_moves_expected_way": float((oriented > 0).mean()),
            "mean_abs_cross_delta": float(g[cross_col].abs().mean()),
            "median_abs_cross_delta": float(g[cross_col].abs().median()),
            "mean_abs_target_delta": float(g[target_col].abs().mean()),
            "selectivity_abs_target_minus_cross": float(
                g[target_col].abs().mean() - g[cross_col].abs().mean()
            ),
            "prediction_change_rate": float((g["patched_prediction"] != g["baseline_prediction"]).mean()),
        })
    return pd.DataFrame(rows).sort_values(gcols).reset_index(drop=True)


def build_transport(per_condition: pd.DataFrame):
    rows = []
    keys = ["sid", "source_layer", "scale"]
    for (sid, L, scale), g in per_condition.groupby(keys):
        lookup = {(str(r.axis), int(r.sign)): r for r in g.itertuples()}
        need = [("H", +1), ("H", -1), ("V", +1), ("V", -1)]
        if not all(k in lookup for k in need):
            continue
        hp, hm = lookup[("H", +1)], lookup[("H", -1)]
        vp, vm = lookup[("V", +1)], lookup[("V", -1)]
        ampH = float(hp.amplitude)
        ampV = float(vp.amplitude)
        if ampH <= EPS or ampV <= EPS:
            continue

        THH = (float(hp.patched_margin_H) - float(hm.patched_margin_H)) / (2.0 * ampH)
        THV = (float(hp.patched_margin_V) - float(hm.patched_margin_V)) / (2.0 * ampH)
        TVH = (float(vp.patched_margin_H) - float(vm.patched_margin_H)) / (2.0 * ampV)
        TVV = (float(vp.patched_margin_V) - float(vm.patched_margin_V)) / (2.0 * ampV)

        diag = 0.5 * (THH + TVV)
        cross_abs = 0.5 * (abs(THV) + abs(TVH))
        diag_abs = 0.5 * (abs(THH) + abs(TVV))
        rows.append({
            "sid": int(sid),
            "gt": str(hp.gt),
            "source_layer": int(L),
            "scale": float(scale),
            "amplitude_H": ampH,
            "amplitude_V": ampV,
            "T_HH": THH,
            "T_HV": THV,
            "T_VH": TVH,
            "T_VV": TVV,
            "diag_signed_mean": diag,
            "diag_abs_mean": diag_abs,
            "cross_abs_mean": cross_abs,
            "diag_abs_minus_cross_abs": diag_abs - cross_abs,
            "diag_to_cross_ratio": diag_abs / max(cross_abs, EPS),
            "HH_positive": THH > 0,
            "VV_positive": TVV > 0,
        })
    return pd.DataFrame(rows)


def summarize_transport(df: pd.DataFrame):
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for (L, scale), g in df.groupby(["source_layer", "scale"]):
        diag_abs = 0.5 * (g["T_HH"].abs() + g["T_VV"].abs())
        cross_abs = 0.5 * (g["T_HV"].abs() + g["T_VH"].abs())
        rows.append({
            "source_layer": int(L),
            "scale": float(scale),
            "N": len(g),
            "mean_T_HH": float(g["T_HH"].mean()),
            "mean_T_HV": float(g["T_HV"].mean()),
            "mean_T_VH": float(g["T_VH"].mean()),
            "mean_T_VV": float(g["T_VV"].mean()),
            "median_T_HH": float(g["T_HH"].median()),
            "median_T_VV": float(g["T_VV"].median()),
            "fraction_HH_positive": float((g["T_HH"] > 0).mean()),
            "fraction_VV_positive": float((g["T_VV"] > 0).mean()),
            "fraction_both_diag_positive": float(((g["T_HH"] > 0) & (g["T_VV"] > 0)).mean()),
            "mean_diag_signed": float(0.5 * (g["T_HH"].mean() + g["T_VV"].mean())),
            "mean_diag_abs": float(diag_abs.mean()),
            "mean_cross_abs": float(cross_abs.mean()),
            "mean_diag_abs_minus_cross_abs": float((diag_abs - cross_abs).mean()),
            "median_diag_abs_minus_cross_abs": float((diag_abs - cross_abs).median()),
            "diag_to_cross_ratio_of_means": float(diag_abs.mean() / max(cross_abs.mean(), EPS)),
        })
    return pd.DataFrame(rows).sort_values(["source_layer", "scale"]).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_layers(a.source_layers)
    scales = parse_floats(a.scales)

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
    axes, calibration_df = fit_axes(X, y, spatial_layers, fit_idx, source_layers)
    calibration_df.to_csv(outdir / "spatial_axis_calibration.csv", index=False)

    two, all_meta, rec_by_sid = load_all_data(a)
    spatial_sid_set = set(map(int, spatial_sids.tolist()))
    meta = [m for m in all_meta if int(m["sid"]) in spatial_sid_set]
    if a.eval_scope == "heldout":
        meta = [m for m in meta if int(m["sid"]) in held_sids]
    meta = stratified_cap(meta, a.eval_max_samples, a.seed)
    if not meta:
        raise RuntimeError("No evaluation samples after filtering")

    model = processor = None
    condition_rows = []
    baseline_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        device = torch.device(a.device)
        if max(source_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(source_layers)} but model has {len(decoder_layers)} decoder layers"
            )

        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)

        print("\n" + "=" * 150)
        print("DIRECT SPATIAL-AXIS FORWARD TRANSPORT")
        print("=" * 150)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N requested={len(meta)} eval_scope={a.eval_scope}")
        print(f"source layers={source_layers}")
        print(f"spatial states={spatial_path}")
        print(f"spatial fit/eval={len(fit_idx)}/{len(held_idx)}")
        print(f"scale_mode={a.scale_mode}; scales={scales}")
        print(f"answer surface={a.answer_surface}; score reduction={a.sequence_score_reduction}")
        print(f"decoder={decoder_path}")
        print("Primary endpoint: finite change in actual answer margins, NOT cosine/gradient.")
        print("=" * 150, flush=True)

        for m in tqdm(meta, desc="FORWARD spatial transport"):
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
                base_margin = margins(base_scores)
                base_pred = argmax_relation(base_scores)
                baseline_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "subject": m["subject"],
                    "reference": m["reference"],
                    "subject_position": sub_pos,
                    "reference_position": ref_pos,
                    "baseline_prediction": base_pred,
                    "baseline_correct": base_pred == m["gt"],
                    "baseline_score_left": base_scores["left"],
                    "baseline_score_right": base_scores["right"],
                    "baseline_score_above": base_scores["above"],
                    "baseline_score_below": base_scores["below"],
                    "baseline_margin_H": base_margin["H"],
                    "baseline_margin_V": base_margin["V"],
                })

                for L in source_layers:
                    for scale in scales:
                        for axis_name in ("H", "V"):
                            natural_amp = axes[L][f"natural_amp_{axis_name}"]
                            amp = float(scale) * (
                                natural_amp if a.scale_mode == "natural" else 1.0
                            )
                            vec = axes[L][axis_name]

                            for sign in (+1, -1):
                                patch_map = make_object_axis_patch(
                                    L, sub_pos, ref_pos, vec, sign * amp
                                )
                                patched_scores = all_scores(
                                    model=model,
                                    batch=batch,
                                    candidate_ids=candidate_ids,
                                    reduction=a.sequence_score_reduction,
                                    decoder_layers=decoder_layers,
                                    patch_map=patch_map,
                                )
                                pm = margins(patched_scores)
                                ppred = argmax_relation(patched_scores)
                                condition_rows.append({
                                    "sid": sid,
                                    "gt": m["gt"],
                                    "source_layer": L,
                                    "scale": float(scale),
                                    "scale_mode": a.scale_mode,
                                    "axis": axis_name,
                                    "sign": int(sign),
                                    "amplitude": amp,
                                    "relation_state_delta_norm": amp,
                                    "per_object_patch_norm": amp / 2.0,
                                    "baseline_prediction": base_pred,
                                    "patched_prediction": ppred,
                                    "baseline_correct": base_pred == m["gt"],
                                    "patched_correct": ppred == m["gt"],
                                    "baseline_margin_H": base_margin["H"],
                                    "baseline_margin_V": base_margin["V"],
                                    "patched_margin_H": pm["H"],
                                    "patched_margin_V": pm["V"],
                                    "delta_margin_H": pm["H"] - base_margin["H"],
                                    "delta_margin_V": pm["V"] - base_margin["V"],
                                    "patched_score_left": patched_scores["left"],
                                    "patched_score_right": patched_scores["right"],
                                    "patched_score_above": patched_scores["above"],
                                    "patched_score_below": patched_scores["below"],
                                })

            except Exception as exc:
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
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

        if not condition_rows:
            raise RuntimeError(f"No successful intervention rows. See {err_path}")

        cond_df = pd.DataFrame(condition_rows)
        base_df = pd.DataFrame(baseline_rows).drop_duplicates("sid")
        cond_df.to_csv(outdir / "per_condition_scores.csv", index=False)
        base_df.to_csv(outdir / "baseline_scores.csv", index=False)

        condition_summary = summarize_conditions(cond_df)
        condition_summary.to_csv(outdir / "condition_summary.csv", index=False)

        transport_df = build_transport(cond_df)
        transport_df.to_csv(outdir / "per_sample_transport_matrix.csv", index=False)
        transport_summary = summarize_transport(transport_df)
        transport_summary.to_csv(outdir / "transport_summary.csv", index=False)

        # Baseline score-argmax accuracy is a teacher-forced diagnostic, not free generation.
        baseline_acc = float(base_df["baseline_correct"].mean()) if len(base_df) else float("nan")

        lines = []
        lines.append("=" * 160)
        lines.append("DIRECT SPATIAL-AXIS FORWARD TRANSPORT")
        lines.append("=" * 160)
        lines.append(f"model={a.model} repo={spec.repo_id}")
        lines.append(f"N successful={base_df['sid'].nunique()} / requested={len(meta)}")
        lines.append(f"source layers={source_layers}")
        lines.append(f"spatial states={spatial_path}")
        lines.append(f"spatial fit/eval={len(fit_idx)}/{len(held_idx)}; eval_scope={a.eval_scope}")
        lines.append(f"scale_mode={a.scale_mode}; scales={scales}")
        lines.append(f"teacher-forced 4-way baseline argmax accuracy={baseline_acc:.4f}")
        lines.append("")
        lines.append("SPATIAL AXIS CALIBRATION")
        lines.append("-" * 160)
        lines.append(calibration_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("EMPIRICAL 2x2 FORWARD TRANSPORT MATRIX")
        lines.append("-" * 160)
        lines.append(
            transport_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
            if len(transport_summary) else "EMPTY"
        )
        lines.append("")
        lines.append("FINITE INTERVENTION CONDITIONS")
        lines.append("-" * 160)
        lines.append(
            condition_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
            if len(condition_summary) else "EMPTY"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  T_HH = d(R-L decision margin) / d(horizontal spatial relation state)")
        lines.append("  T_HV = d(A-B decision margin) / d(horizontal spatial relation state)")
        lines.append("  T_VH = d(R-L decision margin) / d(vertical spatial relation state)")
        lines.append("  T_VV = d(A-B decision margin) / d(vertical spatial relation state)")
        lines.append("")
        lines.append("Strong forward-transport evidence requires:")
        lines.append("  1) mean_T_HH > 0 and mean_T_VV > 0;")
        lines.append("  2) diagonal effects dominate cross-axis effects;")
        lines.append("  3) the sign holds for a substantial fraction of individual samples;")
        lines.append("  4) the pattern persists across reasonable finite scales (not only infinitesimal gradients).")
        lines.append("")
        lines.append("This test does not rely on writer cosine or back-projected cosine as its endpoint.")
        report = "\n".join(lines) + "\n"
        print("\n" + report, flush=True)
        (outdir / "analysis_summary.txt").write_text(report, encoding="utf-8")

        metadata = {
            "script": "eval_spatial_axis_forward_transport_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "spatial_states_npz": str(spatial_path),
            "source_layers": source_layers,
            "spatial_fit_ratio": a.spatial_fit_ratio,
            "spatial_fit_seed": a.spatial_fit_seed,
            "spatial_fit_N": len(fit_idx),
            "spatial_heldout_N": len(held_idx),
            "eval_scope": a.eval_scope,
            "eval_requested_N": len(meta),
            "eval_successful_N": int(base_df["sid"].nunique()),
            "scale_mode": a.scale_mode,
            "scales": scales,
            "answer_surface": a.answer_surface,
            "answer_prefix": a.answer_prefix,
            "answer_suffix": a.answer_suffix,
            "sequence_score_reduction": a.sequence_score_reduction,
            "intervention": (
                "h_sub[L]+=sign*amp/2*d_axis; h_ref[L]-=sign*amp/2*d_axis; "
                "therefore relation_state changes by sign*amp*d_axis"
            ),
            "primary_endpoint": "finite teacher-forced answer-margin transport matrix",
            "uses_sample_gt_for_intervention": False,
            "uses_relation_labels_for_spatial_axis_calibration": True,
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
