#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_causal_update_spatial_decomposition_v1.py

Question
========
Why does an oracle-signed causal-token REAL block update improve spatial
answers?  Is the useful causal effect carried by the update's component in a
previously identified spatial subspace, or by the orthogonal remainder?

For each selected causal-token trajectory and update layer L:

    u = h_real[L,p] - h_real[L-1,p]

Build a frozen spatial subspace S_L from an EXISTING relation-vector NPZ
(the same format used by spatial_affine_train30_test70.py):

    relation_vectors      [N, n_layers, d]
    relation              [N]
    decoder_block_index   [n_layers]

Default spatial basis (axis2):

    horizontal = mu_right - mu_left
    vertical   = mu_above - mu_below
    Q_L = orthonormal_basis([horizontal, vertical])

Then decompose:

    u_spatial = Q_L Q_L^T u
    u_orth    = u - u_spatial

Oracle sign control
===================
This is a MECHANISM experiment, not a deployment method.

The selected WHERE positions remain the existing oracle-ranked causal tokens.
The sign is computed ONCE from the FULL real update using the same GT-vs-best-
competitor sequence-margin gradient as eval_real_causal_token_update_gating_v1:

    B_full = <u, d(S_GT-S_comp)/dh[L,p]>

The SAME sign is then applied to all decomposition conditions:

    full_signed:       sign(B_full) * u
    spatial_signed:    sign(B_full) * u_spatial
    orthogonal_signed: sign(B_full) * u_orth

Thus spatial vs orthogonal cannot win merely because it received a different
GT gate.  It isolates WHERE the causal efficacy of the already-identified
oracle update lives.

Optional norm-matched controls rescale a component back to ||u||, separating
'direction/subspace' from the trivial effect of smaller projected norm.

Interpretation
==============
If spatial_signed ~= full_signed >> orthogonal_signed at matched alpha, then
spatial components largely carry the causal repair effect.

If orthogonal_signed ~= full_signed >> spatial_signed, then the useful code has
likely been transformed out of the original spatial basis before/at these
updates; the next experiment should estimate the spatial->decision transport.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_causal_update_spatial_decomposition_v1.py \
  --model qwen-3b \
  --ranked-causal output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --spatial-states-npz PATH/TO/YOUR_SPATIAL_RELATION_STATES.npz \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --update-layers 20-26 \
  --subspace-mode axis2 \
  --scales 0.25,0.5,1.0 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_causal_update_spatial_decomp_v1 \
  --overwrite

The spatial NPZ should come from the representation you actually want to test
(e.g. REAL-NoImage relation vectors if that is your claimed spatial code).
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

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import eval_real_causal_token_update_gating_v1 as old


REL = ("left", "right", "above", "below")
EPS = 1e-12


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
    p.add_argument("--ranked-causal", required=True)
    p.add_argument(
        "--spatial-states-npz",
        required=True,
        help=(
            "Existing relation-vector states NPZ with relation_vectors, relation, "
            "decoder_block_index. Use the exact spatial representation you want to test."
        ),
    )
    p.add_argument(
        "--subspace-mode",
        default="axis2",
        choices=["axis2", "codebook"],
        help=(
            "axis2 uses (right-left) and (above-below); codebook uses the four "
            "centered relation centroids and SVD."
        ),
    )
    p.add_argument(
        "--subspace-rank",
        type=int,
        default=0,
        help="0=automatic numerical rank; axis2 is at most 2, codebook at most 3 after centering.",
    )
    p.add_argument("--basis-svd-rtol", type=float, default=1e-6)

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument("--causal-categories", default="")
    p.add_argument("--update-layers", default="20-26")
    p.add_argument("--exclude-target-layer", action="store_true")

    p.add_argument("--scales", default="0.25,0.5,1.0")
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=0.0,
        help="Patch only updates with |B_full| > threshold. Same mask/sign for every condition.",
    )
    p.add_argument(
        "--conditions",
        default="full_signed,spatial_signed,orthogonal_signed",
        help=(
            "Comma separated. Available: full_signed, spatial_signed, orthogonal_signed, "
            "spatial_signed_normmatched, orthogonal_signed_normmatched."
        ),
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

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if a.subspace_rank < 0:
        p.error("--subspace-rank must be >=0")
    if not (0 < a.basis_svd_rtol < 1):
        p.error("--basis-svd-rtol must be in (0,1)")
    if a.causal_top_k < 1:
        p.error("--causal-top-k must be >=1")
    if a.max_new_tokens < 1:
        p.error("--max-new-tokens must be >=1")
    if a.decision_threshold < 0 or not np.isfinite(a.decision_threshold):
        p.error("--decision-threshold must be finite and >=0")
    return a


def normalize_relation(x):
    s = str(x).strip().lower()
    return {"on": "above", "under": "below"}.get(s, s)


def safe_ratio(a, b):
    return float(a) / max(float(b), EPS)


def validate_ranked_tokens(rows, batch, tokenizer):
    ids = batch["input_ids"][0].detach().cpu().tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    seen = set()
    for r in rows.itertuples():
        p = int(r.position)
        if p in seen:
            continue
        seen.add(p)
        if not 0 <= p < len(ids):
            raise RuntimeError(f"Ranked causal position {p} outside prompt length {len(ids)}")
        expected = str(r.token)
        actual = str(toks[p]).replace("\n", "\\n")
        if expected != actual:
            raise RuntimeError(
                f"Ranked causal token mismatch at position {p}: csv={expected!r}, prompt={actual!r}"
            )


def load_spatial_subspaces(path, needed_layers, mode, requested_rank, rtol):
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        required = {"relation_vectors", "relation", "decoder_block_index"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{path} missing NPZ keys: {sorted(missing)}")
        X = np.asarray(z["relation_vectors"], dtype=np.float64)
        y = np.asarray([normalize_relation(v) for v in z["relation"].tolist()], dtype=object)
        layers = [int(v) for v in z["decoder_block_index"].tolist()]

    if X.ndim != 3:
        raise RuntimeError(f"relation_vectors must be [N,n_layers,d], got {X.shape}")
    if X.shape[0] != len(y) or X.shape[1] != len(layers):
        raise RuntimeError(
            f"NPZ shape mismatch: X={X.shape}, len(relation)={len(y)}, len(layers)={len(layers)}"
        )
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values in relation_vectors")

    keep = np.isin(y, REL)
    X = X[keep]
    y = y[keep]
    if len(y) == 0:
        raise RuntimeError("No left/right/above/below samples in spatial NPZ")
    counts = {r: int(np.sum(y == r)) for r in REL}
    missing_rel = [r for r, n in counts.items() if n == 0]
    if missing_rel:
        raise RuntimeError(f"Spatial NPZ missing relations {missing_rel}; counts={counts}")

    layer_to_i = {L: i for i, L in enumerate(layers)}
    missing_layers = [L for L in needed_layers if L not in layer_to_i]
    if missing_layers:
        raise RuntimeError(
            f"Spatial NPZ lacks exact update layers {missing_layers}; available={layers}. "
            "Do not silently use nearest layers for this mechanism test."
        )

    basis = {}
    basis_rows = []
    centroid_rows = []

    for L in needed_layers:
        li = layer_to_i[L]
        XL = X[:, li, :]
        global_center = XL.mean(axis=0)
        Xc = XL - global_center
        mu = {r: Xc[y == r].mean(axis=0) for r in REL}

        if mode == "axis2":
            raw = np.stack(
                [mu["right"] - mu["left"], mu["above"] - mu["below"]],
                axis=1,
            )
            raw_names = ["horizontal_right_minus_left", "vertical_above_minus_below"]
        elif mode == "codebook":
            raw = np.stack([mu[r] for r in REL], axis=1)
            raw_names = list(REL)
        else:
            raise ValueError(mode)

        U, S, _ = np.linalg.svd(raw, full_matrices=False)
        if not len(S) or float(S[0]) <= EPS:
            raise RuntimeError(f"Degenerate spatial basis at layer {L}")
        auto_rank = int(np.sum(S > float(S[0]) * float(rtol)))
        max_rank = raw.shape[1]
        rank = auto_rank if requested_rank == 0 else min(int(requested_rank), max_rank)
        rank = min(rank, auto_rank)
        if rank < 1:
            raise RuntimeError(
                f"Requested/available spatial rank is zero at L{L}; singular values={S.tolist()}"
            )
        Q = np.asarray(U[:, :rank], dtype=np.float32)
        ortho_err = float(np.linalg.norm(Q.T @ Q - np.eye(rank, dtype=np.float32)))
        basis[L] = Q

        basis_rows.append(
            {
                "update_layer": L,
                "mode": mode,
                "hidden_dim": int(Q.shape[0]),
                "rank": int(rank),
                "auto_rank": int(auto_rank),
                "orthonormality_error": ortho_err,
                "singular_values": json.dumps([float(s) for s in S]),
                "raw_basis_names": json.dumps(raw_names),
                **{f"calib_N_{r}": counts[r] for r in REL},
            }
        )
        for r in REL:
            centroid_rows.append(
                {
                    "update_layer": L,
                    "direction": r,
                    "centroid_norm": float(np.linalg.norm(mu[r])),
                    "fraction_in_basis": safe_ratio(
                        np.linalg.norm(Q @ (Q.T @ mu[r].astype(np.float32))),
                        np.linalg.norm(mu[r]),
                    ),
                }
            )

    return basis, pd.DataFrame(basis_rows), pd.DataFrame(centroid_rows)


def decompose_scored_entries(scored, basis_by_layer):
    rows = []
    enriched = []
    for e in scored:
        L = int(e["update_layer"])
        if L not in basis_by_layer:
            raise RuntimeError(f"No frozen spatial basis for update layer L{L}")
        Q = basis_by_layer[L]
        u = np.asarray(e["_real_update"], dtype=np.float32)
        g = np.asarray(e["_decision_grad"], dtype=np.float32)
        if u.ndim != 1 or g.ndim != 1 or Q.shape[0] != u.size or g.size != u.size:
            raise RuntimeError(
                f"Dimension mismatch at L{L}: Q={Q.shape}, u={u.shape}, grad={g.shape}"
            )

        spatial = (Q @ (Q.T @ u)).astype(np.float32)
        orth = (u - spatial).astype(np.float32)
        full_norm = float(np.linalg.norm(u))
        sp_norm = float(np.linalg.norm(spatial))
        orth_norm = float(np.linalg.norm(orth))

        b_full = float(e["real_update_decision_score"])
        b_sp = float(np.dot(spatial, g))
        b_orth = float(np.dot(orth, g))
        err = abs(b_full - (b_sp + b_orth))

        z = dict(e)
        z["_spatial_update"] = spatial
        z["_orthogonal_update"] = orth
        z["spatial_update_norm"] = sp_norm
        z["orthogonal_update_norm"] = orth_norm
        z["spatial_norm_fraction"] = safe_ratio(sp_norm, full_norm)
        z["spatial_energy_fraction"] = safe_ratio(sp_norm * sp_norm, full_norm * full_norm)
        z["spatial_decision_score"] = b_sp
        z["orthogonal_decision_score"] = b_orth
        z["decision_decomposition_error"] = err
        enriched.append(z)

        rows.append(
            {
                "sid": int(e["sid"]),
                "gt": str(e["gt"]),
                "baseline_correct": bool(e["baseline_correct"]),
                "update_layer": L,
                "position": int(e["real_position"]),
                "token": str(e["token"]),
                "category": str(e["category"]),
                "broad_category": str(e["broad_category"]),
                "max_target_layer": int(e["max_target_layer"]),
                "full_update_norm": full_norm,
                "spatial_update_norm": sp_norm,
                "orthogonal_update_norm": orth_norm,
                "spatial_norm_fraction": z["spatial_norm_fraction"],
                "spatial_energy_fraction": z["spatial_energy_fraction"],
                "full_decision_score": b_full,
                "spatial_decision_score": b_sp,
                "orthogonal_decision_score": b_orth,
                "decision_decomposition_error": err,
                "spatial_signed_fraction_of_full": b_sp / b_full if abs(b_full) > EPS else float("nan"),
                "orthogonal_signed_fraction_of_full": b_orth / b_full if abs(b_full) > EPS else float("nan"),
                "spatial_abs_share": abs(b_sp) / max(abs(b_sp) + abs(b_orth), EPS),
            }
        )
    return enriched, rows


def component_for_condition(e, condition):
    if condition == "full_signed":
        return np.asarray(e["_real_update"], np.float32), False
    if condition == "spatial_signed":
        return np.asarray(e["_spatial_update"], np.float32), False
    if condition == "orthogonal_signed":
        return np.asarray(e["_orthogonal_update"], np.float32), False
    if condition == "spatial_signed_normmatched":
        return np.asarray(e["_spatial_update"], np.float32), True
    if condition == "orthogonal_signed_normmatched":
        return np.asarray(e["_orthogonal_update"], np.float32), True
    raise ValueError(condition)


def build_same_sign_patch_map(entries, condition, scale, threshold):
    patch_map = {}
    counts = {"positive": 0, "negative": 0, "neutral": 0, "patched": 0, "zero_component": 0}

    for e in entries:
        b = float(e["real_update_decision_score"])
        if b > threshold:
            sign = +1.0
            counts["positive"] += 1
        elif b < -threshold:
            sign = -1.0
            counts["negative"] += 1
        else:
            counts["neutral"] += 1
            continue

        comp, normmatch = component_for_condition(e, condition)
        comp = comp.astype(np.float32, copy=True)
        comp_norm = float(np.linalg.norm(comp))
        if comp_norm <= EPS:
            counts["zero_component"] += 1
            continue

        if normmatch:
            full_norm = float(np.linalg.norm(np.asarray(e["_real_update"], np.float32)))
            comp *= full_norm / comp_norm

        vec = float(scale) * sign * comp
        old.add_patch(
            patch_map,
            e["update_layer"],
            e["real_position"],
            vec,
        )
        counts["patched"] += 1

    return patch_map, counts


def summarize_decomposition(df):
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for keys, g in df.groupby(["update_layer", "baseline_correct"], dropna=False):
        L, correct = keys
        bf = g["full_decision_score"].astype(float).to_numpy()
        bs = g["spatial_decision_score"].astype(float).to_numpy()
        bo = g["orthogonal_decision_score"].astype(float).to_numpy()
        rows.append(
            {
                "update_layer": int(L),
                "baseline_correct": bool(correct),
                "N_samples": int(g["sid"].nunique()),
                "N_updates": int(len(g)),
                "mean_spatial_norm_fraction": float(g["spatial_norm_fraction"].mean()),
                "mean_spatial_energy_fraction": float(g["spatial_energy_fraction"].mean()),
                "mean_full_decision_score": float(np.mean(bf)),
                "mean_spatial_decision_score": float(np.mean(bs)),
                "mean_orthogonal_decision_score": float(np.mean(bo)),
                "mean_abs_full_decision_score": float(np.mean(np.abs(bf))),
                "mean_abs_spatial_decision_score": float(np.mean(np.abs(bs))),
                "mean_abs_orthogonal_decision_score": float(np.mean(np.abs(bo))),
                "spatial_share_of_abs_decision_mass": float(np.sum(np.abs(bs)) / max(np.sum(np.abs(bf)), EPS)),
                "orthogonal_share_of_abs_decision_mass": float(np.sum(np.abs(bo)) / max(np.sum(np.abs(bf)), EPS)),
                "max_decomposition_error": float(g["decision_decomposition_error"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["update_layer", "baseline_correct"]).reset_index(drop=True)


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers = old.parse_layers(a.causal_layers)
    update_layers = old.parse_layers(a.update_layers)
    scales = old.parse_floats(a.scales)
    categories = old.parse_set(a.causal_categories)
    conditions = old.parse_set(a.conditions)
    valid_conditions = {
        "full_signed",
        "spatial_signed",
        "orthogonal_signed",
        "spatial_signed_normmatched",
        "orthogonal_signed_normmatched",
    }
    unknown = set(conditions) - valid_conditions
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if any(L < 1 for L in update_layers):
        raise ValueError("--update-layers must be >=1 because u_L uses L-1")
    if not scales or any((not np.isfinite(s) or s < 0) for s in scales):
        raise ValueError("--scales must contain finite nonnegative values")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}; use --overwrite or a new directory")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    basis_by_layer, basis_df, centroid_df = load_spatial_subspaces(
        a.spatial_states_npz,
        update_layers,
        a.subspace_mode,
        a.subspace_rank,
        a.basis_svd_rtol,
    )
    basis_df.to_csv(outdir / "spatial_basis_summary.csv", index=False)
    centroid_df.to_csv(outdir / "spatial_centroid_projection_summary.csv", index=False)
    np.savez(
        outdir / "frozen_spatial_basis.npz",
        **{f"L{int(L)}_Q": np.asarray(Q, np.float32) for L, Q in basis_by_layer.items()},
    )

    two, meta, rec_by_sid = old.load_data(a)
    eval_sids = {int(m["sid"]) for m in meta}
    causal_sel = old.load_causal_selection(
        Path(a.ranked_causal),
        eval_sids,
        causal_layers,
        a.causal_top_k,
        categories,
    )
    causal_sel.to_csv(outdir / "selected_causal_states.csv", index=False)
    causal_by_sid = {
        int(sid): g.sort_values("rank").copy()
        for sid, g in causal_sel.groupby("sid")
    }

    generation_rows = []
    decomp_rows = []
    model = processor = decoder_layers = None

    try:
        model, processor, decoder_layers, decoder_path, spec = old.load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)
        capture_layers = sorted(set(update_layers + [L - 1 for L in update_layers]))
        bad = [L for L in capture_layers if not 0 <= L < n_layers]
        if bad:
            raise ValueError(f"Requested update/capture layers outside 0..{n_layers-1}: {bad}")

        hidden_dims = {int(Q.shape[0]) for Q in basis_by_layer.values()}
        if len(hidden_dims) != 1:
            raise RuntimeError(f"Spatial bases have inconsistent hidden dims: {hidden_dims}")

        texts = old.candidate_texts(a)
        candidate_ids = old.encode_candidate_ids(processor, texts)

        print("=" * 150)
        print("CAUSAL UPDATE SPATIAL / ORTHOGONAL DECOMPOSITION")
        print("=" * 150)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(meta)}")
        print(f"causal_layers={causal_layers} topK={a.causal_top_k}")
        print(f"update_layers={update_layers}")
        print(f"spatial_npz={a.spatial_states_npz}")
        print(f"subspace_mode={a.subspace_mode} rank={basis_df[['update_layer','rank']].values.tolist()}")
        print(f"conditions={conditions}")
        print(f"scales={scales}")
        print("IMPORTANT: WHERE and sign are oracle; SAME full-update sign is reused for all components.")
        print()

        for m in tqdm(meta, desc="SPATIAL DECOMP CAUSAL UPDATES"):
            sid = int(m["sid"])
            if sid not in causal_by_sid:
                continue
            image = None
            try:
                image = old.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = old.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                validate_ranked_tokens(causal_by_sid[sid], batch, old.tokenizer_of(processor))

                states = old.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    batch,
                    capture_layers,
                )

                base_text = old.base.generate_text(
                    model,
                    processor,
                    batch,
                    max_new_tokens=a.max_new_tokens,
                )
                base_pred = old.traj.normalize_relation(old.base, base_text) or "invalid"
                base_correct = base_pred == m["gt"]
                generation_rows.append(
                    {
                        "sid": sid,
                        "gt": m["gt"],
                        "condition": "baseline",
                        "scale": 0.0,
                        "prediction": base_pred,
                        "correct": base_correct,
                        "n_patched_updates": 0,
                        "n_positive_updates": 0,
                        "n_negative_updates": 0,
                        "n_zero_component": 0,
                        "text": base_text,
                    }
                )

                clean_scores = old.all_sequence_scores(
                    model,
                    batch,
                    candidate_ids,
                    a.sequence_score_reduction,
                )
                competitor = max(
                    (r for r in REL if r != m["gt"]),
                    key=lambda r: clean_scores[r],
                )

                specs = old.causal_position_specs(causal_by_sid[sid])
                entries = old.build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=base_correct,
                    specs=specs,
                    r2n={},
                    real_states=states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=a.exclude_target_layer,
                )
                del states
                if not entries:
                    raise RuntimeError("No REAL causal-token updates")

                grad_layers = sorted(set(int(e["update_layer"]) for e in entries))
                gt_score_graph, grad_gt = old.sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    answer_ids=candidate_ids[m["gt"]],
                    reduction=a.sequence_score_reduction,
                    grad_layers=grad_layers,
                )
                comp_score_graph, grad_comp = old.sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    answer_ids=candidate_ids[competitor],
                    reduction=a.sequence_score_reduction,
                    grad_layers=grad_layers,
                )
                clean_margin = float(clean_scores[m["gt"]] - clean_scores[competitor])
                graph_margin = float(gt_score_graph - comp_score_graph)
                if abs(clean_margin - graph_margin) > 5e-3:
                    raise RuntimeError(
                        f"Sequence margin replay mismatch: clean={clean_margin:.6f}, graph={graph_margin:.6f}"
                    )

                scored = old.attach_decision_scores(entries, grad_gt, grad_comp)
                if not scored:
                    raise RuntimeError("No REAL update received decision gradient")
                enriched, local_rows = decompose_scored_entries(scored, basis_by_layer)
                for row in local_rows:
                    row["baseline_prediction"] = base_pred
                    row["competitor"] = competitor
                    row["sequence_margin"] = clean_margin
                decomp_rows.extend(local_rows)

                max_err = max(float(e["decision_decomposition_error"]) for e in enriched)
                if max_err > 1e-3:
                    raise RuntimeError(f"u = spatial + orth decision decomposition error too large: {max_err:g}")

                for condition in conditions:
                    for scale in scales:
                        pmap, counts = build_same_sign_patch_map(
                            enriched,
                            condition,
                            scale,
                            a.decision_threshold,
                        )
                        pred, text = old.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=batch,
                            patch_map=pmap,
                            max_new_tokens=a.max_new_tokens,
                        )
                        pred = pred or "invalid"
                        generation_rows.append(
                            {
                                "sid": sid,
                                "gt": m["gt"],
                                "condition": condition,
                                "scale": float(scale),
                                "prediction": pred,
                                "correct": pred == m["gt"],
                                "n_patched_updates": int(counts["patched"]),
                                "n_positive_updates": int(counts["positive"]),
                                "n_negative_updates": int(counts["negative"]),
                                "n_zero_component": int(counts["zero_component"]),
                                "text": text,
                            }
                        )

                # Incremental checkpoint after every sample.
                pd.DataFrame(generation_rows).to_csv(outdir / "generation_per_sample.csv", index=False)
                pd.DataFrame(decomp_rows).to_csv(outdir / "per_update_spatial_decomposition.csv", index=False)

                del batch, entries, scored, enriched, grad_gt, grad_comp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                old.append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            finally:
                if image is not None and hasattr(image, "close"):
                    image.close()

        gen_df = pd.DataFrame(generation_rows)
        dec_df = pd.DataFrame(decomp_rows)
        gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)
        dec_df.to_csv(outdir / "per_update_spatial_decomposition.csv", index=False)

        gen_summary = old.summarize_generation(gen_df)
        gen_summary.to_csv(outdir / "generation_summary.csv", index=False)
        old.summarize_generation_by_relation(gen_df).to_csv(
            outdir / "generation_by_relation.csv", index=False
        )
        dec_summary = summarize_decomposition(dec_df)
        dec_summary.to_csv(outdir / "decomposition_summary_by_layer.csv", index=False)

        overall = {
            "mean_spatial_norm_fraction": float(dec_df["spatial_norm_fraction"].mean()),
            "mean_spatial_energy_fraction": float(dec_df["spatial_energy_fraction"].mean()),
            "spatial_share_of_abs_decision_mass": float(
                dec_df["spatial_decision_score"].abs().sum()
                / max(dec_df["full_decision_score"].abs().sum(), EPS)
            ),
            "orthogonal_share_of_abs_decision_mass": float(
                dec_df["orthogonal_decision_score"].abs().sum()
                / max(dec_df["full_decision_score"].abs().sum(), EPS)
            ),
            "max_decision_decomposition_error": float(dec_df["decision_decomposition_error"].max()),
        }

        report = []
        report.append("=" * 150)
        report.append("CAUSAL UPDATE SPATIAL / ORTHOGONAL DECOMPOSITION")
        report.append("=" * 150)
        report.append(f"model={a.model} repo={spec.repo_id}")
        report.append(f"N baseline={gen_df[gen_df.condition == 'baseline'].sid.nunique()}")
        report.append(f"causal layers={causal_layers} topK={a.causal_top_k}")
        report.append(f"update layers={update_layers}")
        report.append(f"spatial basis={a.subspace_mode}; NPZ={a.spatial_states_npz}")
        report.append("WHERE=oracle-ranked; SIGN=oracle GT-vs-best-competitor; same sign reused across components.")
        report.append("")
        report.append("OVERALL DECOMPOSITION")
        report.append("-" * 150)
        for k, v in overall.items():
            report.append(f"{k:40s}: {v:.6f}")
        report.append("")
        report.append("GENERATION")
        report.append("-" * 150)
        report.append(gen_summary.to_string(index=False) if len(gen_summary) else "<empty>")
        report.append("")
        report.append("BY UPDATE LAYER / BASELINE CORRECTNESS")
        report.append("-" * 150)
        report.append(dec_summary.to_string(index=False) if len(dec_summary) else "<empty>")
        report.append("")
        report.append("READOUT:")
        report.append("  spatial ~= full >> orthogonal  => original spatial subspace carries causal repair.")
        report.append("  orthogonal ~= full >> spatial  => spatial code was transformed before decision use.")
        report.append("  normmatched spatial succeeds only after rescale => right direction, weak natural amplitude.")
        report.append("  both partial, full strongest    => mixed code / nonlinear interaction; next test transport map.")
        report_text = "\n".join(report) + "\n"
        print(report_text)
        (outdir / "analysis_summary.txt").write_text(report_text, encoding="utf-8")

        old.write_json(
            outdir / "metadata.json",
            {
                "script": "eval_causal_update_spatial_decomposition_v1.py",
                "args": vars(a),
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "oracle_status": {
                    "where": "oracle-ranked causal positions",
                    "sign": "GT-vs-best-competitor gradient on FULL update",
                    "component_sign_control": "same full-update sign reused for full/spatial/orthogonal",
                },
                "overall": overall,
            },
        )

    finally:
        del model, processor, decoder_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
