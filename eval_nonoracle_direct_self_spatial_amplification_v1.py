#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nonoracle_direct_self_spatial_amplification_v1.py

Direct non-oracle amplification of the model's OWN continuous multi-layer
spatial state.

Core question
=============
The oracle multi-layer free2d controller showed that editing only the H/V
spatial relation subspace at L20-26 can strongly control the final decision.
Here we remove the oracle decision objective completely.

For each controlled layer L, Synthetic-400 defines a source-only continuous
H/V coordinate system.  A clean COCO relation state x_L is read as natural
coordinates

    z_L = (z_H,L, z_V,L).

No argmax over left/right/above/below is formed.  No output logit is used to
choose a direction.  We simply amplify the already-existing clean spatial
state by injecting

    delta_x_L = alpha * z_L                         [raw]

or a norm-matched version of the same 14D direction

    delta_x = alpha * z / ||z||                    [global_unit].

The natural spatial actuator is exactly the same family used by the oracle
free2d controller:

    delta_r_L = natural_M_L @ delta_x_L
    h_sub,L <- h_sub,L + delta_r_L / 2
    h_ref,L <- h_ref,L - delta_r_L / 2.

Thus this experiment asks the cleanest possible question:

    If we already know the model's own continuous spatial state, does simply
    strengthening THAT state across multiple middle layers improve generation?

Oracle status
=============
COCO inference/intervention uses:
    * NO COCO GT relation to choose a direction;
    * NO relation argmax / selector;
    * NO decision/logit objective;
    * NO gradient;
    * NO update positive/negative classifier;
    * NO GT-dependent trigger or stopping rule.

Synthetic-400 relation labels are used only to define the source H/V coordinate
system.  COCO GT is loaded only for evaluation statistics (accuracy, W2C/C2W,
and optional direction diagnostics).

IMPORTANT: a sweep over scales/modes is diagnostic.  Picking the best scale
using COCO GT would be target-domain tuning.  For a fully frozen non-oracle
claim, choose the scale/mode before looking at COCO evaluation.

Recommended N=80
================
CUDA_VISIBLE_DEVICES=0 python -u eval_nonoracle_direct_self_spatial_amplification_v1.py \
  --model qwen-3b \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --layer-groups "20-26" \
  --direction-modes raw,global_unit \
  --scales 0.25,0.5,0.75,1.0,1.5,3.0,6.0 \
  --eval-max-samples 80 \
  --seed 17 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --max-new-tokens 6 \
  --device cuda:0 \
  --attn-impl eager \
  --output-dir output/qwen3b_direct_self_spatial_amp_n80_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
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
    import eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp as ora
except Exception as exc:
    raise SystemExit(
        "Could not import AdaptVis dependencies. Run from repository root and keep:\n"
        "  eval_real_causal_token_update_gating_v1.py\n"
        "  eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp.py\n"
        f"available.\n{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
EPS = 1e-12
SCRIPT_VERSION = "v1_direct_clean_multilayer_spatial_amplification"


def norm_rel(x) -> str:
    return ora.norm_rel(x)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return v / n


def parse_float_list(text: str) -> List[float]:
    vals = []
    for p in str(text).split(","):
        p = p.strip()
        if not p:
            continue
        x = float(p)
        if x < 0:
            raise ValueError("scales must be non-negative")
        if x not in vals:
            vals.append(x)
    if not vals:
        raise ValueError("No scales parsed")
    return vals


def parse_modes(text: str) -> List[str]:
    allowed = {"raw", "global_unit", "layer_unit"}
    vals = []
    for p in str(text).split(","):
        p = p.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise ValueError(f"Unknown direction mode {p!r}; choose from {sorted(allowed)}")
        if p not in vals:
            vals.append(p)
    if not vals:
        raise ValueError("No direction modes parsed")
    return vals


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
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument(
        "--layer-groups",
        default="20-26",
        help='Semicolon-separated controlled layer groups, e.g. "25;23-26;20-26".',
    )
    p.add_argument(
        "--direction-modes",
        default="raw,global_unit",
        help=(
            "raw: delta_x_L = alpha*z_L; "
            "global_unit: concatenate all layer z_L and normalize once in 2|G|D; "
            "layer_unit: normalize each layer z_L separately."
        ),
    )
    p.add_argument(
        "--scales",
        default="0.25,0.5,0.75,1.0,1.5,3.0,6.0",
        help="Comma-separated positive natural-coordinate amplification scales.",
    )
    p.add_argument(
        "--include-negative-control",
        action="store_true",
        help="Also inject -alpha times the same self-spatial direction as a causal sign control.",
    )
    p.add_argument(
        "--max-total-natural",
        type=float,
        default=12.0,
        help="Optional total L2 cap on injected natural coordinates; <=0 disables.",
    )
    p.add_argument(
        "--max-layer-natural",
        type=float,
        default=0.0,
        help="Optional per-layer L2 cap; <=0 disables.",
    )
    p.add_argument(
        "--min-layer-z-norm",
        type=float,
        default=0.0,
        help="Set a layer's z_L to zero when ||z_L|| is below this value; 0 disables.",
    )
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all aligned targets")
    p.add_argument("--seed", type=int, default=17)

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
    try:
        a.scales_parsed = parse_float_list(a.scales)
        a.direction_modes_parsed = parse_modes(a.direction_modes)
    except ValueError as exc:
        p.error(str(exc))
    if a.min_layer_z_norm < 0:
        p.error("--min-layer-z-norm must be >=0")
    return a


def load_state_npz(path: Path, *, require_labels: bool):
    """Load relation_vectors or old img/no_image cache as REAL-NoImage relation state."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            vector_definition = (
                str(z["vector_definition"].item())
                if "vector_definition" in keys
                else "relation_vectors"
            )
        elif {"img", "no_image"}.issubset(keys):
            X = np.asarray(z["img"], dtype=np.float32) - np.asarray(
                z["no_image"], dtype=np.float32
            )
            vector_definition = "img_minus_no_image"
        else:
            raise RuntimeError(
                f"{path} must contain relation_vectors OR both img and no_image; "
                f"keys={sorted(keys)}"
            )

        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]
        sids = (
            np.asarray(z["sample_index"], dtype=np.int64)
            if "sample_index" in keys
            else np.arange(X.shape[0], dtype=np.int64)
        )
        labels = None
        if "relation" in keys:
            labels = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels for source geometry")

    if X.ndim != 3:
        raise RuntimeError(f"Bad state shape {X.shape}; expected [N,L,D]")
    if X.shape[0] != len(sids):
        raise RuntimeError("X/sample_index length mismatch")
    if labels is not None and len(labels) != X.shape[0]:
        raise RuntimeError("X/relation length mismatch")
    return X, labels, layers, sids, vector_definition


def fit_source_geometry(X, y, layers, fit_layers: Sequence[int]):
    """Fit Synthetic-only H/V natural-coordinate system using ALL source samples."""
    if y is None:
        raise RuntimeError("Source labels required")
    layer_to_i = {int(L): i for i, L in enumerate(layers)}
    missing = [int(L) for L in fit_layers if int(L) not in layer_to_i]
    if missing:
        raise RuntimeError(f"Source NPZ missing layers {missing}; has {layers}")

    geom = {}
    rows = []
    for L in fit_layers:
        li = layer_to_i[int(L)]
        Xf = X[:, li].astype(np.float64)
        center = Xf.mean(axis=0)
        means = {}
        for r in REL:
            mask = y == r
            if not np.any(mask):
                raise RuntimeError(f"Source has no samples for relation={r}")
            means[r] = Xf[mask].mean(axis=0)

        # Same spatial geometry convention as the oracle free2d controller.
        class_dirs = {r: unit(means[r] - center) for r in REL}
        dH = unit(class_dirs["right"] - class_dirs["left"])
        dV = unit(class_dirs["above"] - class_dirs["below"])
        gapH = float(np.dot(means["right"] - means["left"], dH))
        gapV = float(np.dot(means["above"] - means["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV

        B = np.stack([dH, dV], axis=1)  # hidden x 2
        dual = B @ np.linalg.inv(B.T @ B)
        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)
        natural_M = dual @ np.diag([halfH, halfV])

        geom[int(L)] = {
            "layer_index": int(li),
            "center": center.astype(np.float32),
            "B": B.astype(np.float32),
            "dual": dual.astype(np.float32),
            "natural_M": natural_M.astype(np.float32),
            "natural_half_H": float(halfH),
            "natural_half_V": float(halfV),
        }
        rows.append({
            "source_layer": int(L),
            "fit_N": int(X.shape[0]),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "natural_full_gap_H": float(gapH),
            "natural_half_gap_H": float(halfH),
            "natural_full_gap_V": float(gapV),
            "natural_half_gap_V": float(halfV),
            "dual_check_max_abs": float(np.max(np.abs(B.T @ dual - np.eye(2)))),
        })
    return geom, pd.DataFrame(rows)


def read_layer_natural_coord(x: np.ndarray, g: dict) -> np.ndarray:
    """Read clean state as H/V natural coordinates in the source-defined basis."""
    residual = np.asarray(x, dtype=np.float64) - g["center"].astype(np.float64)
    coeff = g["dual"].astype(np.float64).T @ residual
    return np.asarray([
        coeff[0] / max(float(g["natural_half_H"]), EPS),
        coeff[1] / max(float(g["natural_half_V"]), EPS),
    ], dtype=np.float64)


def build_target_layer_coords(Xt, target_layers, target_sids, geom, needed_layers):
    tmap = {int(L): i for i, L in enumerate(target_layers)}
    missing = [int(L) for L in needed_layers if int(L) not in tmap]
    if missing:
        raise RuntimeError(f"Target NPZ missing layers {missing}; has {target_layers}")

    refs = {}
    rows = []
    for i, sid in enumerate(target_sids.tolist()):
        per = {}
        for L in needed_layers:
            z = read_layer_natural_coord(Xt[i, tmap[int(L)]], geom[int(L)])
            per[int(L)] = z
            rows.append({
                "sid": int(sid),
                "layer": int(L),
                "z_H": float(z[0]),
                "z_V": float(z[1]),
                "z_norm": float(np.linalg.norm(z)),
            })
        refs[int(sid)] = per
    return refs, pd.DataFrame(rows)


def direction_from_layer_coords(
    per_layer: Dict[int, np.ndarray],
    layers: Sequence[int],
    mode: str,
    min_layer_z_norm: float,
) -> np.ndarray:
    Z = np.stack([np.asarray(per_layer[int(L)], dtype=np.float64) for L in layers], axis=0)
    if min_layer_z_norm > 0:
        norms = np.linalg.norm(Z, axis=1)
        Z = Z.copy()
        Z[norms < min_layer_z_norm] = 0.0

    if mode == "raw":
        return Z
    if mode == "global_unit":
        n = float(np.linalg.norm(Z.reshape(-1)))
        return Z / n if n > EPS else np.zeros_like(Z)
    if mode == "layer_unit":
        out = np.zeros_like(Z)
        for i in range(len(Z)):
            n = float(np.linalg.norm(Z[i]))
            if n > EPS:
                out[i] = Z[i] / n
        return out
    raise ValueError(mode)


def scaled_and_capped_coords(direction, scale, max_total, max_layer):
    coords = float(scale) * np.asarray(direction, dtype=np.float64)
    hit_layer = False
    if max_layer > 0:
        coords = coords.copy()
        for i in range(len(coords)):
            n = float(np.linalg.norm(coords[i]))
            if n > max_layer:
                coords[i] *= max_layer / max(n, EPS)
                hit_layer = True
    hit_total = False
    nall = float(np.linalg.norm(coords.reshape(-1)))
    if max_total > 0 and nall > max_total:
        coords *= max_total / max(nall, EPS)
        hit_total = True
    return coords, hit_total, hit_layer


def decision_xy(scores: Dict[str, float]) -> np.ndarray:
    return np.asarray([
        float(scores["right"]) - float(scores["left"]),
        float(scores["above"]) - float(scores["below"]),
    ], dtype=np.float64)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    keys = ["layer_group", "direction_mode", "sign", "scale"]
    for key, g in df.groupby(keys, sort=False):
        b = g["baseline_correct"].astype(bool).to_numpy()
        p = g["patched_correct"].astype(bool).to_numpy()
        w2c = int(np.sum((~b) & p))
        c2w = int(np.sum(b & (~p)))
        wrong_n = int(np.sum(~b))
        correct_n = int(np.sum(b))
        rows.append({
            "layer_group": key[0],
            "direction_mode": key[1],
            "sign": key[2],
            "scale": float(key[3]),
            "N": int(len(g)),
            "baseline_accuracy": float(np.mean(b)),
            "patched_accuracy": float(np.mean(p)),
            "gain": float(np.mean(p) - np.mean(b)),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": int(w2c - c2w),
            "changed": int(np.sum(g["baseline_prediction"] != g["patched_prediction"])),
            "baseline_wrong_N": wrong_n,
            "repair_rate_on_wrong": float(w2c / wrong_n) if wrong_n else np.nan,
            "preserve_rate_on_correct": float((correct_n - c2w) / correct_n) if correct_n else np.nan,
            "mean_injected_total_natural": float(g["injected_total_natural_norm"].mean()),
            "mean_tf_spatial_alignment_shift": float(g["tf_clean_spatial_alignment_shift"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["patched_accuracy", "net", "direction_mode", "scale"],
        ascending=[False, False, True, True],
    )


def summarize_relation(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    keys = ["layer_group", "direction_mode", "sign", "scale", "gt"]
    for key, g in df.groupby(keys, sort=False):
        b = g["baseline_correct"].astype(bool).to_numpy()
        p = g["patched_correct"].astype(bool).to_numpy()
        rows.append({
            "layer_group": key[0],
            "direction_mode": key[1],
            "sign": key[2],
            "scale": float(key[3]),
            "relation": key[4],
            "N": int(len(g)),
            "baseline_accuracy": float(np.mean(b)),
            "patched_accuracy": float(np.mean(p)),
            "gain": float(np.mean(p) - np.mean(b)),
            "wrong_to_correct": int(np.sum((~b) & p)),
            "correct_to_wrong": int(np.sum(b & (~p))),
        })
    return pd.DataFrame(rows)


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gt_axis_diagnostic(gt: str, Z: np.ndarray):
    """Evaluation-only; never used for intervention."""
    zmean = np.mean(np.asarray(Z, dtype=np.float64), axis=0)
    if gt == "left":
        main, orth = -float(zmean[0]), abs(float(zmean[1]))
    elif gt == "right":
        main, orth = float(zmean[0]), abs(float(zmean[1]))
    elif gt == "above":
        main, orth = float(zmean[1]), abs(float(zmean[0]))
    elif gt == "below":
        main, orth = -float(zmean[1]), abs(float(zmean[0]))
    else:
        return np.nan, np.nan, False
    return main, orth, bool(main > 0)


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    groups = ora.parse_layer_groups(a.layer_groups)
    needed_layers = sorted(set(L for _, ls in groups for L in ls))
    modes = list(a.direction_modes_parsed)
    scales = list(a.scales_parsed)
    signs = [("positive", +1.0)]
    if a.include_negative_control:
        signs.append(("negative_control", -1.0))

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

    Xs, ys, source_layers, source_sids, source_def = load_state_npz(
        Path(a.source_spatial_npz), require_labels=True
    )
    Xt, yt_diag, target_layers, target_sids, target_def = load_state_npz(
        Path(a.target_spatial_npz), require_labels=False
    )
    if Xs.shape[-1] != Xt.shape[-1]:
        raise RuntimeError(
            f"Source/target hidden dim mismatch: {Xs.shape[-1]} vs {Xt.shape[-1]}"
        )

    geom, axis_df = fit_source_geometry(Xs, ys, source_layers, needed_layers)
    axis_df.to_csv(outdir / "synthetic_spatial_axis_geometry.csv", index=False)

    refs, ref_layer_df = build_target_layer_coords(
        Xt, target_layers, target_sids, geom, needed_layers
    )
    ref_layer_df.to_csv(outdir / "clean_spatial_coordinates_per_layer.csv", index=False)

    target_sid_set = set(map(int, target_sids.tolist()))
    two, all_meta, rec_by_sid = ora.load_all_data(a)
    eval_meta = [m for m in all_meta if int(m["sid"]) in target_sid_set]
    eval_meta = ora.stratified_cap(eval_meta, a.eval_max_samples, a.seed)
    if not eval_meta:
        raise RuntimeError("No evaluation samples after target-state alignment")

    model = processor = None
    baseline_rows = []
    result_rows = []
    direction_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        if max(needed_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(needed_layers)} but model has {len(decoder_layers)} decoder layers"
            )
        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)
        device = torch.device(a.device)

        print("\n" + "=" * 190)
        print("DIRECT CLEAN MULTI-LAYER SPATIAL AMPLIFICATION — NO SELECTOR / NO LOGIT OBJECTIVE")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source={a.source_spatial_npz} N={len(Xs)} definition={source_def}")
        print(f"target={a.target_spatial_npz} N={len(Xt)} definition={target_def}")
        print(f"layer groups={[g for g, _ in groups]}")
        print(f"direction modes={modes}")
        print(f"scales={scales}")
        print(f"negative control={a.include_negative_control}")
        print("Direction is read independently at EACH controlled layer from the clean spatial state.")
        print("NO COCO GT / NO relation argmax / NO decision objective / NO gradients / NO trigger.")
        print("GT below is evaluation-only.")
        print("=" * 190, flush=True)

        for m in tqdm(eval_meta, desc="EVAL direct self-spatial amp"):
            sid = int(m["sid"])
            gt = str(m["gt"])  # evaluation only
            batch = None
            try:
                if sid not in refs:
                    raise RuntimeError(f"sid={sid} missing clean spatial coordinates")

                image = gate.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = gate.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                sub_pos, ref_pos = ora.find_object_positions(
                    two, processor, batch, m["subject"], m["reference"]
                )

                base_scores = ora.all_scores(
                    model, batch, candidate_ids, a.sequence_score_reduction,
                    decoder_layers, {},
                )
                base_tf_pred = ora.argmax_relation(base_scores)
                base_y = decision_xy(base_scores)
                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map={},
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = bool(base_gen_pred == gt)

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "subject": m["subject"],
                    "reference": m["reference"],
                    "baseline_prediction": base_gen_pred,
                    "baseline_correct": base_correct,
                    "baseline_tf_prediction": base_tf_pred,
                    "baseline_y_H": float(base_y[0]),
                    "baseline_y_V": float(base_y[1]),
                    "baseline_generation_text": base_gen_text,
                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                })

                for group_label, layers in groups:
                    Zraw = np.stack([refs[sid][int(L)] for L in layers], axis=0)
                    mean_z = np.mean(Zraw, axis=0)
                    gt_main, gt_orth, gt_sign_ok = gt_axis_diagnostic(gt, Zraw)

                    direction_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "layer_group": group_label,
                        "raw_14d_norm": float(np.linalg.norm(Zraw.reshape(-1))),
                        "mean_z_H": float(mean_z[0]),
                        "mean_z_V": float(mean_z[1]),
                        "eval_only_gt_axis_evidence": float(gt_main),
                        "eval_only_orthogonal_evidence": float(gt_orth),
                        "eval_only_gt_axis_sign_correct": bool(gt_sign_ok),
                        **{
                            f"z_L{L}_{axis}": float(Zraw[i, j])
                            for i, L in enumerate(layers)
                            for j, axis in enumerate(("H", "V"))
                        },
                    })

                    for mode in modes:
                        direction = direction_from_layer_coords(
                            refs[sid], layers, mode, a.min_layer_z_norm
                        )
                        direction_norm = float(np.linalg.norm(direction.reshape(-1)))

                        for sign_name, sign_value in signs:
                            signed_direction = sign_value * direction
                            for scale in scales:
                                coords, hit_total_cap, hit_layer_cap = scaled_and_capped_coords(
                                    signed_direction,
                                    scale,
                                    a.max_total_natural,
                                    a.max_layer_natural,
                                )
                                inject_norm = float(np.linalg.norm(coords.reshape(-1)))

                                patch_map = ora.natural_coords_to_patch(
                                    layers, geom, sub_pos, ref_pos, coords
                                ) if inject_norm > EPS else {}

                                patched_scores = ora.all_scores(
                                    model, batch, candidate_ids,
                                    a.sequence_score_reduction,
                                    decoder_layers, patch_map,
                                )
                                patched_tf_pred = ora.argmax_relation(patched_scores)
                                patched_y = decision_xy(patched_scores)
                                patched_gen_pred, patched_gen_text = gate.generate_with_patch(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=batch,
                                    patch_map=patch_map,
                                    max_new_tokens=a.max_new_tokens,
                                )
                                patched_correct = bool(patched_gen_pred == gt)

                                # Diagnostic only: did final continuous decision move along
                                # the clean spatial direction? This is NOT used to choose/edit.
                                mean_z_norm = float(np.linalg.norm(mean_z))
                                z_for_diag = mean_z / mean_z_norm if mean_z_norm > EPS else mean_z
                                base_align = float(np.dot(z_for_diag, base_y))
                                patched_align = float(np.dot(z_for_diag, patched_y))

                                result_rows.append({
                                    "sid": sid,
                                    "gt": gt,
                                    "layer_group": group_label,
                                    "direction_mode": mode,
                                    "sign": sign_name,
                                    "scale": float(scale),
                                    "baseline_prediction": base_gen_pred,
                                    "patched_prediction": patched_gen_pred,
                                    "baseline_correct": base_correct,
                                    "patched_correct": patched_correct,
                                    "baseline_tf_prediction": base_tf_pred,
                                    "patched_tf_prediction": patched_tf_pred,
                                    "direction_14d_norm_before_scale": direction_norm,
                                    "injected_total_natural_norm": inject_norm,
                                    "hit_total_cap": bool(hit_total_cap),
                                    "hit_layer_cap": bool(hit_layer_cap),
                                    "mean_clean_z_H": float(mean_z[0]),
                                    "mean_clean_z_V": float(mean_z[1]),
                                    "eval_only_gt_axis_sign_correct": bool(gt_sign_ok),
                                    "baseline_y_H": float(base_y[0]),
                                    "baseline_y_V": float(base_y[1]),
                                    "patched_y_H": float(patched_y[0]),
                                    "patched_y_V": float(patched_y[1]),
                                    "tf_clean_spatial_alignment_before": base_align,
                                    "tf_clean_spatial_alignment_after": patched_align,
                                    "tf_clean_spatial_alignment_shift": float(patched_align - base_align),
                                    "baseline_generation_text": base_gen_text,
                                    "patched_generation_text": patched_gen_text,
                                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                                    **{f"patched_score_{r}": float(patched_scores[r]) for r in REL},
                                })

            except Exception as exc:
                append_jsonl(err_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}", flush=True)
            finally:
                if batch is not None:
                    del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        baseline_df = pd.DataFrame(baseline_rows)
        result_df = pd.DataFrame(result_rows)
        direction_df = pd.DataFrame(direction_rows)
        summary_df = summarize(result_df)
        byrel_df = summarize_relation(result_df)

        baseline_df.to_csv(outdir / "baseline.csv", index=False)
        result_df.to_csv(outdir / "per_condition.csv", index=False)
        direction_df.to_csv(outdir / "clean_multilayer_direction_diagnostic.csv", index=False)
        summary_df.to_csv(outdir / "summary.csv", index=False)
        byrel_df.to_csv(outdir / "by_relation.csv", index=False)

        # Evaluation-only diagnostic: how often the clean continuous state points
        # with the correct sign along the GT axis. No 4-way selector is constructed.
        diag_rows = []
        if len(direction_df):
            for group_label, g in direction_df.groupby("layer_group", sort=False):
                for cohort_name, mask in [
                    ("all", np.ones(len(g), dtype=bool)),
                    ("baseline_wrong", ~g["sid"].map(
                        baseline_df.set_index("sid")["baseline_correct"]
                    ).astype(bool).to_numpy()),
                    ("baseline_correct", g["sid"].map(
                        baseline_df.set_index("sid")["baseline_correct"]
                    ).astype(bool).to_numpy()),
                ]:
                    gg = g.loc[mask]
                    if len(gg) == 0:
                        continue
                    diag_rows.append({
                        "layer_group": group_label,
                        "cohort": cohort_name,
                        "N": int(len(gg)),
                        "gt_axis_sign_correct_rate_eval_only": float(
                            gg["eval_only_gt_axis_sign_correct"].astype(bool).mean()
                        ),
                        "mean_gt_axis_evidence_eval_only": float(
                            gg["eval_only_gt_axis_evidence"].mean()
                        ),
                        "mean_abs_orthogonal_evidence_eval_only": float(
                            gg["eval_only_orthogonal_evidence"].mean()
                        ),
                    })
        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_csv(outdir / "clean_spatial_direction_gt_diagnostic_eval_only.csv", index=False)

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "source_spatial_npz": a.source_spatial_npz,
            "target_spatial_npz": a.target_spatial_npz,
            "source_vector_definition": source_def,
            "target_vector_definition": target_def,
            "source_N": int(len(Xs)),
            "target_cache_N": int(len(Xt)),
            "eval_N": int(len(baseline_df)),
            "layer_groups": [{"label": x, "layers": ls} for x, ls in groups],
            "direction_modes": modes,
            "scales": scales,
            "include_negative_control": bool(a.include_negative_control),
            "max_total_natural": float(a.max_total_natural),
            "max_layer_natural": float(a.max_layer_natural),
            "min_layer_z_norm": float(a.min_layer_z_norm),
            "uses_target_GT_for_intervention": False,
            "uses_target_GT_for_trigger": False,
            "uses_target_GT_for_direction": False,
            "uses_relation_argmax_selector": False,
            "uses_output_logits_for_direction": False,
            "uses_gradient": False,
            "source_labels_define_spatial_axes": True,
            "GT_usage": "evaluation only",
            "canonical_raw_rule": "delta_x_L = alpha * clean_z_L for each controlled layer",
            "global_unit_rule": "delta_x = alpha * clean_z / ||clean_z|| over concatenated 2|G| coordinates",
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        base_acc = float(baseline_df["baseline_correct"].astype(bool).mean()) if len(baseline_df) else float("nan")
        print("\n" + "=" * 190)
        print("DIRECT SELF-SPATIAL AMPLIFICATION: GENERATION")
        print("=" * 190)
        print(f"baseline_accuracy={base_acc:.4f} N={len(baseline_df)}")
        if len(summary_df):
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("EMPTY")

        print("\nBY RELATION (evaluation only)")
        print("-" * 190)
        if len(byrel_df):
            print(byrel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("EMPTY")

        print("\nCLEAN SPATIAL DIRECTION DIAGNOSTIC (GT evaluation only; NOT used by controller)")
        print("-" * 190)
        if len(diag_df):
            print(diag_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("EMPTY")

        print("\nInterpretation:")
        print("  raw is the literal hypothesis: each layer amplifies its own clean H/V coordinates.")
        print("  global_unit is the norm-matched version closest to one normalized oracle free2d step.")
        print("  If + direction helps and - direction hurts at matched norm, directionality is causal evidence.")
        print("  A best scale selected from this COCO sweep is diagnostic, not a frozen zero-shot policy.")

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
