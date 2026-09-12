
# -*- coding: utf-8 -*-
"""
eval_nonoracle_self_spatial_multilayer_controller_v1.py

NON-ORACLE continuous self-spatial -> decision controller.

Core question
=============
Can the model's OWN clean middle-layer spatial state provide the objective for
our already-validated multi-layer spatial actuator, without ever selecting a
relation label (left/right/above/below) and without using target-domain GT?

Source-only spatial geometry
============================
Synthetic-400 defines, for each layer L, a continuous H/V spatial coordinate
system.  No COCO labels are used to fit this geometry.

For source relation state r_L:

    d_H,L  ~ right - left
    d_V,L  ~ above - below

The dual basis and source natural half-gaps define natural coordinates.  For a
clean target state x_L, read

    z_L = diag(1/half_gap_L) @ dual_L^T @ (x_L - source_center_L)

and aggregate across read layers:

    z_ref = mean_L z_L

Then FREEZE / stop-gradient this clean evidence.  No argmax relation is formed.

Continuous decision coordinates
===============================
For four teacher-forced sequence scores:

    y_H(x) = S_right(x) - S_left(x)
    y_V(x) = S_above(x) - S_below(x)

Normalize the clean spatial evidence by default:

    z_hat = z_ref / ||z_ref||

and optimize the continuous self-alignment objective

    J_self(x) = z_hat_H * y_H(x) + z_hat_V * y_V(x)

This does NOT say which discrete answer must win.  It only asks the final
continuous H/V decision to move in the direction already represented by the
model's clean spatial state.

Actuator
========
Exactly as in the oracle multi-layer free2d controller, each controlled layer
gets two independent natural H/V coordinates.  For L20-26 this is 14 scalars.
At every accepted step:

    1) score all four answers at the CURRENT patched state;
    2) compute exact local gradients dS_r / d[all layer H/V coords];
    3) combine them into dJ_self/dx;
    4) take a normalized step in the 14D spatial subspace;
    5) backtracking line-search using THE SAME J_self;
    6) replan from the new state.

Non-oracle stopping
===================
No GT is used to stop.  The default stop condition is simply

    J_self >= required_self_margin

Samples already satisfying this at the clean state are preserved.

Important oracle-status statement
=================================
TARGET/COCO inference uses:
    * NO target GT to choose a relation;
    * NO target GT in the objective;
    * NO target GT in the gradient;
    * NO target GT in the line search;
    * NO target GT in the stopping rule;
    * NO discrete spatial relation selector.

Synthetic source labels ARE used to define the H/V coordinate system.  GT is
read only after the intervention for evaluation statistics.

Recommended N=80 first
======================
CUDA_VISIBLE_DEVICES=0 python -u eval_nonoracle_self_spatial_multilayer_controller_v1.py \
  --model qwen-3b \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --read-layers 20-26 \
  --layer-groups "20-26" \
  --z-mode unit \
  --required-self-margin 0.25 \
  --step-natural 0.75 \
  --max-steps 20 \
  --max-total-natural 12.0 \
  --line-search-shrink 0.5 \
  --line-search-tries 8 \
  --eval-max-samples 80 \
  --seed 17 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --max-new-tokens 6 \
  --device cuda:0 \
  --attn-impl eager \
  --output-dir output/qwen3b_nonoracle_self_spatial_multilayer_n80_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
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
    import eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp as ora
except Exception as exc:
    raise SystemExit(
        "Could not import the existing AdaptVis controller dependencies.\n"
        "Run this script from the AdaptVis repository root and keep:\n"
        "  eval_real_causal_token_update_gating_v1.py\n"
        "  eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp.py\n"
        f"available there.\n{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
EPS = 1e-12
SCRIPT_VERSION = "v1_self_spatial_continuous_no_relation_selector"


def parse_layers(text: str) -> List[int]:
    return ora.parse_layers(text)


def parse_layer_groups(text: str) -> List[Tuple[str, List[int]]]:
    return ora.parse_layer_groups(text)


def norm_rel(x) -> str:
    return ora.norm_rel(x)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return v / n


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
        "--read-layers",
        default="20-26",
        help="Layers whose CLEAN target spatial coordinates are averaged into z_ref.",
    )
    p.add_argument(
        "--layer-groups",
        default="20-26",
        help='Semicolon-separated actuator groups, e.g. "25;23-26;20-26".',
    )
    p.add_argument(
        "--z-mode",
        default="unit",
        choices=["unit", "raw"],
        help=(
            "unit: objective uses z_ref/||z_ref||, so J has sequence-score units; "
            "raw: preserve source-natural evidence magnitude."
        ),
    )
    p.add_argument(
        "--min-z-norm",
        type=float,
        default=0.0,
        help="Non-oracle abstention for weak spatial evidence; 0 disables.",
    )
    p.add_argument(
        "--required-self-margin",
        type=float,
        default=0.25,
        help=(
            "Stop/preserve once J_self = z_used dot [S_R-S_L, S_A-S_B] reaches this value. "
            "No GT is involved."
        ),
    )
    p.add_argument("--step-natural", type=float, default=0.75)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument(
        "--max-total-natural",
        type=float,
        default=12.0,
        help="Global L2 cap across all controlled natural coordinates; <=0 disables.",
    )
    p.add_argument(
        "--max-layer-natural",
        type=float,
        default=0.0,
        help="Optional per-layer coordinate L2 cap; <=0 disables.",
    )
    p.add_argument("--line-search-shrink", type=float, default=0.5)
    p.add_argument("--line-search-tries", type=int, default=8)
    p.add_argument("--min-objective-improvement", type=float, default=1e-6)
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all target samples")
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
    if a.required_self_margin < 0:
        p.error("--required-self-margin must be >=0")
    if a.min_z_norm < 0:
        p.error("--min-z-norm must be >=0")
    if a.step_natural <= 0:
        p.error("--step-natural must be >0")
    if a.max_steps <= 0:
        p.error("--max-steps must be >0")
    if not (0 < a.line_search_shrink < 1):
        p.error("--line-search-shrink must be in (0,1)")
    if a.line_search_tries <= 0:
        p.error("--line-search-tries must be >0")
    if a.min_objective_improvement < 0:
        p.error("--min-objective-improvement must be >=0")
    return a


def load_state_npz(path: Path, *, require_labels: bool):
    """Load either relation_vectors NPZ or old img/no_image cache as REAL-NoImage."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            vector_definition = str(z["vector_definition"].item()) if "vector_definition" in keys else "relation_vectors"
        elif {"img", "no_image"}.issubset(keys):
            X = np.asarray(z["img"], dtype=np.float32) - np.asarray(z["no_image"], dtype=np.float32)
            vector_definition = "img_minus_no_image"
        else:
            raise RuntimeError(
                f"{path} must contain relation_vectors OR both img and no_image; keys={sorted(keys)}"
            )

        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]

        if "sample_index" in keys:
            sids = np.asarray(z["sample_index"], dtype=np.int64)
        else:
            sids = np.arange(X.shape[0], dtype=np.int64)

        labels = None
        if "relation" in keys:
            labels = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels for SOURCE geometry fitting")

    if X.ndim != 3:
        raise RuntimeError(f"Bad state shape {X.shape} in {path}; expected [N,L,D]")
    if X.shape[0] != len(sids):
        raise RuntimeError(f"X/sid mismatch in {path}: {X.shape[0]} vs {len(sids)}")
    if labels is not None and len(labels) != X.shape[0]:
        raise RuntimeError(f"X/relation mismatch in {path}: {X.shape[0]} vs {len(labels)}")
    return X, labels, layers, sids, vector_definition


def fit_source_geometry(X, y, layers, fit_layers: Sequence[int]):
    """Source-only H/V geometry; adds center needed to read continuous target z."""
    if y is None:
        raise RuntimeError("Source labels are required")
    layer_to_i = {int(L): i for i, L in enumerate(layers)}
    missing = [int(L) for L in fit_layers if int(L) not in layer_to_i]
    if missing:
        raise RuntimeError(f"Source NPZ missing layers {missing}; has {layers}")

    geom = {}
    rows = []
    fit_idx = np.arange(X.shape[0], dtype=np.int64)
    for L in fit_layers:
        li = layer_to_i[int(L)]
        Xf = X[fit_idx, li].astype(np.float64)
        yf = y[fit_idx]
        center = Xf.mean(axis=0)
        means = {}
        for r in REL:
            mask = yf == r
            if not np.any(mask):
                raise RuntimeError(f"Source has no samples for relation {r}")
            means[r] = Xf[mask].mean(axis=0)

        # Keep exactly the oracle-controller convention.
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
        gram = B.T @ B
        dual = B @ np.linalg.inv(gram)
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
            "fit_N": int(len(fit_idx)),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "natural_full_gap_H": float(gapH),
            "natural_half_gap_H": float(halfH),
            "natural_full_gap_V": float(gapV),
            "natural_half_gap_V": float(halfV),
            "dual_check_max_abs": float(np.max(np.abs(B.T @ dual - np.eye(2)))),
        })
    return geom, pd.DataFrame(rows)


def read_layer_natural_coord(x: np.ndarray, g: dict) -> np.ndarray:
    """Least-squares coordinate in the same natural basis used by the actuator."""
    residual = np.asarray(x, dtype=np.float64) - g["center"].astype(np.float64)
    coeff = g["dual"].astype(np.float64).T @ residual
    return np.asarray([
        coeff[0] / max(float(g["natural_half_H"]), EPS),
        coeff[1] / max(float(g["natural_half_V"]), EPS),
    ], dtype=np.float64)


def build_target_reference_table(
    Xt: np.ndarray,
    target_layers: Sequence[int],
    target_sids: np.ndarray,
    geom: dict,
    read_layers: Sequence[int],
    z_mode: str,
):
    tmap = {int(L): i for i, L in enumerate(target_layers)}
    missing = [int(L) for L in read_layers if int(L) not in tmap]
    if missing:
        raise RuntimeError(f"Target NPZ missing read layers {missing}; has {target_layers}")

    refs = {}
    rows = []
    for i, sid in enumerate(target_sids.tolist()):
        per_layer = []
        row = {"sid": int(sid)}
        for L in read_layers:
            zL = read_layer_natural_coord(Xt[i, tmap[int(L)]], geom[int(L)])
            per_layer.append(zL)
            row[f"z_L{L}_H"] = float(zL[0])
            row[f"z_L{L}_V"] = float(zL[1])
        z = np.mean(np.stack(per_layer, axis=0), axis=0)
        znorm = float(np.linalg.norm(z))
        if z_mode == "unit" and znorm > EPS:
            z_used = z / znorm
        else:
            z_used = z.copy()
        refs[int(sid)] = {
            "z_raw": z.astype(np.float64),
            "z_used": z_used.astype(np.float64),
            "z_norm": znorm,
        }
        row.update({
            "z_raw_H": float(z[0]),
            "z_raw_V": float(z[1]),
            "z_norm": float(znorm),
            "z_used_H": float(z_used[0]),
            "z_used_V": float(z_used[1]),
            "dominant_axis": "H" if abs(float(z[0])) >= abs(float(z[1])) else "V",
        })
        rows.append(row)
    return refs, pd.DataFrame(rows)


def decision_xy(scores: Dict[str, float]) -> np.ndarray:
    return np.asarray([
        float(scores["right"]) - float(scores["left"]),
        float(scores["above"]) - float(scores["below"]),
    ], dtype=np.float64)


def self_objective(scores: Dict[str, float], z_used: np.ndarray) -> float:
    return float(np.dot(np.asarray(z_used, dtype=np.float64), decision_xy(scores)))


def self_objective_grad(score_grad: Dict[str, Tuple[float, np.ndarray]], z_used: np.ndarray):
    z = np.asarray(z_used, dtype=np.float64)
    gH = score_grad["right"][1] - score_grad["left"][1]
    gV = score_grad["above"][1] - score_grad["below"][1]
    g = float(z[0]) * gH + float(z[1]) * gV
    y = np.asarray([
        score_grad["right"][0] - score_grad["left"][0],
        score_grad["above"][0] - score_grad["below"][0],
    ], dtype=np.float64)
    J = float(np.dot(z, y))
    return J, np.asarray(g, dtype=np.float64), y


def apply_free2d_step(coords, direction, step, max_total, max_layer):
    coords = np.asarray(coords, dtype=np.float64)
    cand = coords + float(step) * np.asarray(direction, dtype=np.float64)
    if max_layer > 0:
        for i in range(len(cand)):
            n = float(np.linalg.norm(cand[i]))
            if n > max_layer:
                cand[i] *= float(max_layer) / max(n, EPS)
    nall = float(np.linalg.norm(cand.reshape(-1)))
    hit_total = False
    if max_total > 0 and nall > max_total:
        cand *= float(max_total) / max(nall, EPS)
        hit_total = True
    return cand, hit_total


def summarize(repair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, g in repair_df.groupby("layer_group", sort=False):
        base = g["baseline_correct"].astype(bool).to_numpy()
        patch = g["patched_correct"].astype(bool).to_numpy()
        wrong = ~base
        w2c = int(np.sum(wrong & patch))
        c2w = int(np.sum(base & (~patch)))
        edited = g["steps_taken"].to_numpy() > 0
        rows.append({
            "layer_group": group,
            "N": int(len(g)),
            "baseline_accuracy": float(np.mean(base)),
            "patched_accuracy": float(np.mean(patch)),
            "gain": float(np.mean(patch) - np.mean(base)),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "edited_N": int(np.sum(edited)),
            "edited_fraction": float(np.mean(edited)),
            "baseline_wrong_N": int(np.sum(wrong)),
            "repair_rate_on_wrong": float(w2c / max(int(np.sum(wrong)), 1)),
            "preserve_rate_on_correct": float(1.0 - c2w / max(int(np.sum(base)), 1)),
            "mean_initial_self_margin": float(g["initial_self_margin"].mean()),
            "mean_final_self_margin": float(g["final_self_margin"].mean()),
            "fraction_initially_self_aligned": float(g["initial_self_aligned"].mean()),
            "fraction_final_self_aligned": float(g["final_self_aligned"].mean()),
            "mean_steps": float(g["steps_taken"].mean()),
            "line_search_fail_fraction": float(g["line_search_failed"].mean()),
        })
    return pd.DataFrame(rows)


def summarize_relation(repair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, rel), g in repair_df.groupby(["layer_group", "gt"], sort=False):
        base = g["baseline_correct"].astype(bool).to_numpy()
        patch = g["patched_correct"].astype(bool).to_numpy()
        rows.append({
            "layer_group": group,
            "relation": rel,
            "N": int(len(g)),
            "baseline_accuracy": float(np.mean(base)),
            "patched_accuracy": float(np.mean(patch)),
            "gain": float(np.mean(patch) - np.mean(base)),
            "wrong_to_correct": int(np.sum((~base) & patch)),
            "correct_to_wrong": int(np.sum(base & (~patch))),
            "edited_N": int(np.sum(g["steps_taken"].to_numpy() > 0)),
        })
    return pd.DataFrame(rows)


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    read_layers = parse_layers(a.read_layers)
    groups = parse_layer_groups(a.layer_groups)
    actuator_layers = sorted(set(L for _, ls in groups for L in ls))
    geom_layers = sorted(set(read_layers) | set(actuator_layers))

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

    geom, axis_df = fit_source_geometry(Xs, ys, source_layers, geom_layers)
    axis_df.to_csv(outdir / "synthetic_spatial_axis_geometry.csv", index=False)

    refs, ref_df = build_target_reference_table(
        Xt=Xt,
        target_layers=target_layers,
        target_sids=target_sids,
        geom=geom,
        read_layers=read_layers,
        z_mode=a.z_mode,
    )
    ref_df.to_csv(outdir / "continuous_spatial_reference.csv", index=False)

    target_sid_set = set(map(int, target_sids.tolist()))
    two, all_meta, rec_by_sid = ora.load_all_data(a)
    eval_meta = [m for m in all_meta if int(m["sid"]) in target_sid_set]
    eval_meta = ora.stratified_cap(eval_meta, a.eval_max_samples, a.seed)
    if not eval_meta:
        raise RuntimeError("No evaluation samples after target-state alignment")

    model = processor = None
    baseline_rows = []
    repair_rows = []
    step_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        if max(geom_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(geom_layers)} but model has {len(decoder_layers)} decoder layers"
            )
        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)
        device = torch.device(a.device)

        print("\n" + "=" * 190)
        print("NON-ORACLE CONTINUOUS SELF-SPATIAL -> DECISION CONTROLLER")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source geometry={a.source_spatial_npz} N={len(Xs)} definition={source_def}")
        print(f"target clean state={a.target_spatial_npz} N={len(Xt)} definition={target_def}")
        print(f"read layers={read_layers} | actuator groups={[g for g, _ in groups]}")
        print(f"z_mode={a.z_mode} min_z_norm={a.min_z_norm}")
        print(
            f"J_self = z_H*(S_R-S_L) + z_V*(S_A-S_B); "
            f"required_self_margin={a.required_self_margin}"
        )
        print(
            f"step={a.step_natural} max_steps={a.max_steps} "
            f"max_total={a.max_total_natural}"
        )
        print("NO target GT / NO relation argmax / NO GT stopping. GT is evaluation-only.")
        print("=" * 190, flush=True)

        for m in tqdm(eval_meta, desc="EVAL self-spatial controller"):
            sid = int(m["sid"])
            gt = str(m["gt"])  # evaluation only
            image = batch = None
            try:
                if sid not in refs:
                    raise RuntimeError(f"sid={sid} missing target spatial reference")
                zraw = refs[sid]["z_raw"]
                zused = refs[sid]["z_used"]
                znorm = float(refs[sid]["z_norm"])

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
                base_self = self_objective(base_scores, zused)
                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map={},
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = base_gen_pred == gt

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "subject": m["subject"],
                    "reference": m["reference"],
                    "z_raw_H": float(zraw[0]),
                    "z_raw_V": float(zraw[1]),
                    "z_norm": float(znorm),
                    "z_used_H": float(zused[0]),
                    "z_used_V": float(zused[1]),
                    "baseline_y_H": float(base_y[0]),
                    "baseline_y_V": float(base_y[1]),
                    "baseline_self_margin": float(base_self),
                    "baseline_tf_prediction": base_tf_pred,
                    "baseline_generation_prediction": base_gen_pred,
                    "baseline_generation_correct": bool(base_correct),
                    "baseline_generation_text": base_gen_text,
                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                })

                for group_label, layers in groups:
                    common = {
                        "sid": sid,
                        "gt": gt,  # evaluation only
                        "layer_group": group_label,
                        "layers": ",".join(map(str, layers)),
                        "z_raw_H": float(zraw[0]),
                        "z_raw_V": float(zraw[1]),
                        "z_norm": float(znorm),
                        "z_used_H": float(zused[0]),
                        "z_used_V": float(zused[1]),
                    }

                    # Non-oracle abstention only from spatial evidence strength.
                    if znorm < a.min_z_norm:
                        repair_rows.append({
                            **common,
                            "baseline_prediction": base_gen_pred,
                            "patched_prediction": base_gen_pred,
                            "baseline_correct": bool(base_correct),
                            "patched_correct": bool(base_correct),
                            "baseline_tf_prediction": base_tf_pred,
                            "final_tf_prediction": base_tf_pred,
                            "initial_self_margin": float(base_self),
                            "final_self_margin": float(base_self),
                            "initial_self_aligned": bool(base_self >= a.required_self_margin),
                            "final_self_aligned": bool(base_self >= a.required_self_margin),
                            "steps_taken": 0,
                            "final_total_natural_norm": 0.0,
                            "hit_total_cap": False,
                            "line_search_failed": False,
                            "stop_reason": "weak_spatial_evidence_abstain",
                            "baseline_generation_text": base_gen_text,
                            "patched_generation_text": base_gen_text,
                        })
                        continue

                    coords = np.zeros((len(layers), 2), dtype=np.float64)
                    current_scores = dict(base_scores)
                    current_obj = float(base_self)
                    hit_total_cap = False
                    line_search_failed = False
                    stop_reason = "max_steps"
                    steps_taken = 0

                    if current_obj >= a.required_self_margin:
                        stop_reason = "already_self_aligned"
                    else:
                        for step_idx in range(1, a.max_steps + 1):
                            # Exact local gradients of all four answer scores with respect
                            # to the CURRENT multi-layer spatial coordinates.
                            score_grad = {}
                            for r in REL:
                                sr, gr = ora.sequence_score_and_spatial_grad(
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

                            current_obj_grad, grad2d, current_y = self_objective_grad(
                                score_grad, zused
                            )
                            # Numerical consistency check against separately evaluated scores.
                            current_obj = float(current_obj_grad)
                            direction, raw_grad_norm = ora.normalize_direction(grad2d)
                            if direction is None:
                                stop_reason = "zero_self_spatial_gradient"
                                break

                            accepted = False
                            accepted_scores = None
                            accepted_coords = None
                            accepted_step = None
                            accepted_hit_cap = False
                            trial_logs = []

                            for ls in range(a.line_search_tries):
                                step_size = float(a.step_natural * (a.line_search_shrink ** ls))
                                cand_coords, trial_hit_cap = apply_free2d_step(
                                    coords=coords,
                                    direction=direction,
                                    step=step_size,
                                    max_total=a.max_total_natural,
                                    max_layer=a.max_layer_natural,
                                )
                                dc = cand_coords - coords
                                step_norm = float(np.linalg.norm(dc.reshape(-1)))
                                if step_norm < 1e-10:
                                    trial_logs.append({
                                        "ls": int(ls),
                                        "step_size": float(step_size),
                                        "self_margin": None,
                                        "objective_gain": None,
                                        "step_norm": float(step_norm),
                                        "accepted": False,
                                    })
                                    continue

                                cand_patch = ora.natural_coords_to_patch(
                                    layers, geom, sub_pos, ref_pos, cand_coords
                                )
                                cand_scores = ora.all_scores(
                                    model, batch, candidate_ids, a.sequence_score_reduction,
                                    decoder_layers, cand_patch,
                                )
                                cand_obj = self_objective(cand_scores, zused)
                                gain = float(cand_obj - current_obj)
                                improved = gain >= a.min_objective_improvement
                                trial_logs.append({
                                    "ls": int(ls),
                                    "step_size": float(step_size),
                                    "self_margin": float(cand_obj),
                                    "objective_gain": float(gain),
                                    "step_norm": float(step_norm),
                                    "accepted": bool(improved),
                                })
                                if improved:
                                    accepted = True
                                    accepted_scores = cand_scores
                                    accepted_coords = cand_coords
                                    accepted_step = (ls, step_size, step_norm)
                                    accepted_hit_cap = bool(trial_hit_cap)
                                    break

                            if not accepted:
                                line_search_failed = True
                                stop_reason = "line_search_failed"
                                step_rows.append({
                                    **common,
                                    "step": int(step_idx),
                                    "accepted": False,
                                    "self_margin_before": float(current_obj),
                                    "self_margin_after": float(current_obj),
                                    "objective_gain": 0.0,
                                    "raw_self_gradient_norm": float(raw_grad_norm),
                                    "decision_y_H_before": float(current_y[0]),
                                    "decision_y_V_before": float(current_y[1]),
                                    "accepted_step_natural": 0.0,
                                    "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                                    "line_search_trials": json.dumps(trial_logs),
                                })
                                break

                            prev_obj = float(current_obj)
                            coords = accepted_coords
                            current_scores = accepted_scores
                            current_obj = self_objective(current_scores, zused)
                            current_y_after = decision_xy(current_scores)
                            hit_total_cap = bool(hit_total_cap or accepted_hit_cap)
                            steps_taken = int(step_idx)
                            dc = coords if step_idx == 1 else None

                            step_rows.append({
                                **common,
                                "step": int(step_idx),
                                "accepted": True,
                                "self_margin_before": float(prev_obj),
                                "self_margin_after": float(current_obj),
                                "objective_gain": float(current_obj - prev_obj),
                                "raw_self_gradient_norm": float(raw_grad_norm),
                                "decision_y_H_before": float(current_y[0]),
                                "decision_y_V_before": float(current_y[1]),
                                "decision_y_H_after": float(current_y_after[0]),
                                "decision_y_V_after": float(current_y_after[1]),
                                "accepted_line_search_index": int(accepted_step[0]),
                                "accepted_step_size": float(accepted_step[1]),
                                "accepted_step_natural": float(accepted_step[2]),
                                "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                                "line_search_trials": json.dumps(trial_logs),
                            })

                            if current_obj >= a.required_self_margin:
                                stop_reason = "self_alignment_margin_reached"
                                break
                            if (
                                a.max_total_natural > 0
                                and np.linalg.norm(coords.reshape(-1)) >= a.max_total_natural - 1e-8
                            ):
                                stop_reason = "total_natural_cap"
                                break

                    final_patch = ora.natural_coords_to_patch(
                        layers, geom, sub_pos, ref_pos, coords
                    ) if np.linalg.norm(coords) > 0 else {}
                    final_scores = current_scores if np.linalg.norm(coords) > 0 else dict(base_scores)
                    final_tf_pred = ora.argmax_relation(final_scores)
                    final_y = decision_xy(final_scores)
                    final_obj = self_objective(final_scores, zused)

                    if np.linalg.norm(coords) > 0:
                        final_gen_pred, final_gen_text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=batch,
                            patch_map=final_patch,
                            max_new_tokens=a.max_new_tokens,
                        )
                    else:
                        final_gen_pred, final_gen_text = base_gen_pred, base_gen_text

                    row = {
                        **common,
                        "baseline_prediction": base_gen_pred,
                        "patched_prediction": final_gen_pred,
                        "baseline_correct": bool(base_correct),
                        "patched_correct": bool(final_gen_pred == gt),
                        "baseline_tf_prediction": base_tf_pred,
                        "final_tf_prediction": final_tf_pred,
                        "initial_self_margin": float(base_self),
                        "final_self_margin": float(final_obj),
                        "initial_self_aligned": bool(base_self >= a.required_self_margin),
                        "final_self_aligned": bool(final_obj >= a.required_self_margin),
                        "baseline_y_H": float(base_y[0]),
                        "baseline_y_V": float(base_y[1]),
                        "final_y_H": float(final_y[0]),
                        "final_y_V": float(final_y[1]),
                        "steps_taken": int(steps_taken),
                        "final_total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                        "hit_total_cap": bool(hit_total_cap),
                        "line_search_failed": bool(line_search_failed),
                        "stop_reason": stop_reason,
                        "baseline_generation_text": base_gen_text,
                        "patched_generation_text": final_gen_text,
                    }
                    for L in actuator_layers:
                        row[f"final_x_L{L}_H"] = np.nan
                        row[f"final_x_L{L}_V"] = np.nan
                    for i, L in enumerate(layers):
                        row[f"final_x_L{L}_H"] = float(coords[i, 0])
                        row[f"final_x_L{L}_V"] = float(coords[i, 1])
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
        summary_df = summarize(repair_df) if len(repair_df) else pd.DataFrame()
        byrel_df = summarize_relation(repair_df) if len(repair_df) else pd.DataFrame()

        baseline_df.to_csv(outdir / "baseline.csv", index=False)
        repair_df.to_csv(outdir / "per_repair.csv", index=False)
        step_df.to_csv(outdir / "per_step.csv", index=False)
        summary_df.to_csv(outdir / "summary.csv", index=False)
        byrel_df.to_csv(outdir / "by_relation.csv", index=False)

        # Evaluation-only diagnostics: relation labels are never used upstream.
        if len(baseline_df):
            baseline_acc = float(baseline_df["baseline_generation_correct"].mean())
        else:
            baseline_acc = float("nan")

        print("\n" + "=" * 190)
        print("NON-ORACLE SELF-SPATIAL CONTROLLER: GENERATION")
        print("=" * 190)
        print(f"baseline_accuracy={baseline_acc:.4f} N={len(baseline_df)}")
        if len(summary_df):
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nBY RELATION (evaluation only)")
        print("-" * 190)
        if len(byrel_df):
            print(byrel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Useful diagnostic: do correct/wrong baselines differ in SELF alignment?
        if len(baseline_df):
            diag = []
            for label, g in [
                ("all", baseline_df),
                ("baseline_wrong", baseline_df.loc[~baseline_df["baseline_generation_correct"].astype(bool)]),
                ("baseline_correct", baseline_df.loc[baseline_df["baseline_generation_correct"].astype(bool)]),
            ]:
                if len(g):
                    diag.append({
                        "cohort": label,
                        "N": int(len(g)),
                        "mean_self_margin": float(g["baseline_self_margin"].mean()),
                        "median_self_margin": float(g["baseline_self_margin"].median()),
                        "fraction_self_margin_ge_threshold": float(
                            (g["baseline_self_margin"] >= a.required_self_margin).mean()
                        ),
                        "mean_z_norm": float(g["z_norm"].mean()),
                    })
            diag_df = pd.DataFrame(diag)
            diag_df.to_csv(outdir / "baseline_self_alignment_diagnostic.csv", index=False)
            print("\nBASELINE SELF-ALIGNMENT DIAGNOSTIC")
            print("-" * 190)
            print(diag_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "source_spatial_npz": str(a.source_spatial_npz),
            "target_spatial_npz": str(a.target_spatial_npz),
            "source_state_definition": source_def,
            "target_state_definition": target_def,
            "source_N": int(len(Xs)),
            "target_state_N": int(len(Xt)),
            "eval_N": int(len(eval_meta)),
            "read_layers": list(map(int, read_layers)),
            "layer_groups": {label: list(map(int, ls)) for label, ls in groups},
            "z_mode": a.z_mode,
            "min_z_norm": float(a.min_z_norm),
            "required_self_margin": float(a.required_self_margin),
            "objective": "zH*(S_right-S_left) + zV*(S_above-S_below)",
            "z_reference": (
                "clean target REAL-NoImage h_sub-h_ref state projected into frozen "
                "Synthetic-400 H/V natural coordinates and averaged across read layers"
            ),
            "z_reference_detached": True,
            "relation_argmax_used_for_routing": False,
            "target_GT_used_for_objective": False,
            "target_GT_used_for_gradient": False,
            "target_GT_used_for_line_search": False,
            "target_GT_used_for_stopping": False,
            "target_GT_used_for_evaluation_only": True,
            "source_relation_labels_used_for_geometry": True,
            "actuator": "multi-layer free2d H/V natural-coordinate object-relation patch",
            "step_natural": float(a.step_natural),
            "max_steps": int(a.max_steps),
            "max_total_natural": float(a.max_total_natural),
            "max_layer_natural": float(a.max_layer_natural),
            "line_search_shrink": float(a.line_search_shrink),
            "line_search_tries": int(a.line_search_tries),
            "min_objective_improvement": float(a.min_objective_improvement),
            "sequence_score_reduction": a.sequence_score_reduction,
            "answer_surface": a.answer_surface,
            "seed": int(a.seed),
            "decoder_path": decoder_path,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
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
