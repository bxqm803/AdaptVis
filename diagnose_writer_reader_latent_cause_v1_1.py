#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_writer_reader_latent_cause_v1.py

Purpose
=======
We know the local behavioral polarity of a realized causal-token block update is

    B_{i,L,p} = a_{i,L,p}^T g_{i,L,p}

where

    a = h_out - h_in                         ("writer" / realized block update)
    g = d(S_GT - S_comp) / d h_out           ("reader" / downstream sensitivity)

This script asks a more fundamental question than "Attention or MLP?":

    WHY is one update positive and another negative?

There are three possibilities:

  1) WRITER-DOMINANT:
       the geometry of a itself carries the sign.
       A negative writer remains negative even under a positive sample's reader,
       and a positive writer remains positive under a negative sample's reader.

  2) READER-DOMINANT:
       the downstream reader g determines the sign.
       Swapping readers flips the polarity of otherwise positive/negative writers.

  3) INTERACTION / COMPATIBILITY:
       neither side has a stable sign by itself; sign emerges from the pairing
       between a and g.

The core diagnostic is a matched cross-swap:

    B++ = a+ · g+   > 0
    B-- = a- · g-   < 0

then compute

    B-+ = a- · g+       negative writer under positive reader
    B+- = a+ · g-       positive writer under negative reader

Interpretation:

    writer-follow:
        B-+ < 0  AND  B+- > 0

    reader-follow:
        B-+ > 0  AND  B+- < 0

    mixed:
        everything else

Pairs are matched within increasingly strict strata so the result cannot be
explained simply by layer or relation:

    relation:
        update_layer + GT + competitor

    role:
        update_layer + GT + competitor + broad_category

    category:
        update_layer + GT + competitor + category

    position:
        update_layer + GT + competitor + real_position

The script ALSO performs a coarse "first sign separability" scan.  It captures,
at the SAME causal token and layer:

    block_in
    post_attn = block_in + attention_residual
    block_out
    attention_residual
    mlp_residual
    real_update = block_out - block_in
    reader_gradient g

It then asks whether a linear probe can predict sign(B) on held-out SIDs.
To suppress trivial shortcuts, positive/negative rows are balanced WITHIN
matched strata before cross-validation.  Features are random-projected to a
small dimension before probing; this is only a localization diagnostic.

IMPORTANT
=========
* The script does NOT claim a latent cause merely because a probe is accurate.
* Cross-swap is still a counterfactual dot-product diagnostic, not a model
  intervention.  It tests writer-vs-reader geometry under the same first-order
  definition used to define B.
* If a stage first becomes strongly sign-decodable, the next experiment should
  patch that stage and measure whether B / generation actually changes.
* GT and the strongest GT competitor are used, exactly as in the prior oracle
  mechanism diagnostic.  This is NOT a deployable selector.

Dependencies
============
Run from the AdaptVis repository root.  This script reuses:

    analyze_real_update_module_head_sources_v1.py

which in turn reuses the exact model/data/prompt/gradient machinery from the
current project.

Recommended quick N=80 run
==========================
CUDA_VISIBLE_DEVICES=0 python -u diagnose_writer_reader_latent_cause_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --trace-layers 8-26 \
  --max-samples 80 \
  --pair-repeats 20 \
  --probe-dim 256 \
  --output-dir output/qwen3b_writer_reader_latent_n80_v1 \
  --overwrite

Full 440
========
CUDA_VISIBLE_DEVICES=0 python -u diagnose_writer_reader_latent_cause_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --trace-layers 8-26 \
  --pair-repeats 20 \
  --probe-dim 256 \
  --output-dir output/qwen3b_writer_reader_latent_all440_v1 \
  --overwrite

Main outputs
============
per_update_metadata.csv
cross_swap_summary.csv
cross_swap_pair_examples.csv
probe_stage_summary.csv
probe_stage_by_layer.csv
analysis_summary.txt
metadata.json
errors.jsonl
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import analyze_real_update_module_head_sources_v1 as src
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_real_update_module_head_sources_v1.py.\n"
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

EPS = 1e-12

STAGES = (
    "block_in",
    "post_attn",
    "block_out",
    "attn_update",
    "mlp_update",
    "real_update",
    "reader_g",
)

MATCH_MODES = {
    "relation": ["update_layer", "gt", "competitor"],
    "role": ["update_layer", "gt", "competitor", "broad_category"],
    "category": ["update_layer", "gt", "competitor", "category"],
    "position": ["update_layer", "gt", "competitor", "real_position"],
}


# =============================================================================
# CLI / utilities
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
    p.add_argument("--real-update-dir", required=True)
    p.add_argument("--trace-layers", default="8-26")
    p.add_argument("--exclude-target-layer", action="store_true")

    p.add_argument(
        "--max-samples",
        type=int,
        default=80,
        help="0 = all samples from the prior successful run.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--b-threshold",
        type=float,
        default=1e-8,
        help="Exclude |B| <= threshold from polarity analysis.",
    )
    p.add_argument(
        "--pair-repeats",
        type=int,
        default=20,
        help="Independent deterministic re-pairings inside matched strata.",
    )
    p.add_argument(
        "--match-modes",
        default="relation,role,category,position",
        help="Subset of: relation,role,category,position.",
    )
    p.add_argument(
        "--min-stratum-each-sign",
        type=int,
        default=1,
        help="Minimum positive AND negative rows in a matching stratum.",
    )

    p.add_argument(
        "--probe-dim",
        type=int,
        default=256,
        help="Gaussian random-projection dimension for sign probes; 0 = disable probes.",
    )
    p.add_argument(
        "--probe-folds",
        type=int,
        default=5,
    )
    p.add_argument(
        "--probe-min-rows",
        type=int,
        default=80,
        help="Minimum balanced rows for a reported probe.",
    )
    p.add_argument(
        "--probe-match-mode",
        default="role",
        choices=list(MATCH_MODES.keys()),
        help="Strata used to balance +/- before linear probing.",
    )
    p.add_argument(
        "--probe-c",
        type=float,
        default=1.0,
        help="LogisticRegression C.",
    )

    p.add_argument(
        "--save-full-vectors",
        action="store_true",
        help=(
            "Save full fp16 a and g matrices. Useful for later experiments but "
            "can be hundreds of MB on all-440."
        ),
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", ""}:
        return False
    return bool(x)


def safe_mean(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def cosine_rows(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > EPS)
    return out


def sign_arr(x, threshold=0.0):
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x > threshold,
        1,
        np.where(x < -threshold, -1, 0),
    ).astype(np.int8)


def selected_match_modes(text):
    modes = []
    for x in str(text).split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x not in MATCH_MODES:
            raise ValueError(
                f"Unknown match mode={x!r}; choices={list(MATCH_MODES)}"
            )
        modes.append(x)
    if not modes:
        raise ValueError("--match-modes is empty")
    return modes


# =============================================================================
# Capture realized writer vectors + reader gradients
# =============================================================================

def valid_sample_layers(specs, trace_layers, exclude_target_layer):
    layers = []
    for L in trace_layers:
        include = False
        for s in specs:
            cmax = int(s["max_target_layer"])
            ok = L < cmax if exclude_target_layer else L <= cmax
            if ok:
                include = True
                break
        if include:
            layers.append(int(L))
    return layers


def collect_sample_rows(
    *,
    m,
    causal_rows,
    model,
    processor,
    decoder_layers,
    rec,
    candidate_ids,
    reduction,
    trace_layers,
    exclude_target_layer,
):
    specs = src.causal_position_specs(causal_rows)
    positions = sorted(set(int(s["real_position"]) for s in specs))
    sample_layers = valid_sample_layers(
        specs,
        trace_layers,
        exclude_target_layer,
    )
    if not sample_layers:
        return [], {}

    image = src.base.record_image(rec)
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    try:
        batch = src.base.make_question_batch(
            processor=processor,
            image=image,
            question_text=m["question_text"],
            device=torch.device(next(model.parameters()).device),
        )

        gt_score, grad_gt, caps = src.run_answer_with_grad_and_optional_capture(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            answer_ids=candidate_ids[m["gt"]],
            reduction=reduction,
            layers=sample_layers,
            positions=positions,
            capture_modules=True,
        )

        comp_score, grad_comp, _ = src.run_answer_with_grad_and_optional_capture(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            answer_ids=candidate_ids[m["competitor"]],
            reduction=reduction,
            layers=sample_layers,
            positions=positions,
            capture_modules=False,
        )

        replay_margin = float(gt_score - comp_score)

        rows = []
        vectors = {
            stage: []
            for stage in STAGES
        }

        for s in specs:
            p = int(s["real_position"])
            cmax = int(s["max_target_layer"])
            pi = src.position_index(caps, p)

            for L in sample_layers:
                L = int(L)
                if exclude_target_layer:
                    if L >= cmax:
                        continue
                else:
                    if L > cmax:
                        continue

                gg = grad_gt.get(L, None)
                gcomp = grad_comp.get(L, None)
                if gg is None or gcomp is None:
                    continue
                if not (
                    0 <= p < gg.shape[1]
                    and 0 <= p < gcomp.shape[1]
                ):
                    continue

                g = (
                    gg[0, p].astype(np.float32)
                    - gcomp[0, p].astype(np.float32)
                )

                x_in = caps["block_in"][L][pi].astype(np.float32)
                x_out = caps["block_out"][L][pi].astype(np.float32)
                a_attn = caps["attn_out"][L][pi].astype(np.float32)
                a_mlp = caps["mlp_out"][L][pi].astype(np.float32)

                post_attn = (x_in + a_attn).astype(np.float32)
                a_real = (x_out - x_in).astype(np.float32)

                B = float(np.dot(a_real, g))
                B_attn = float(np.dot(a_attn, g))
                B_mlp = float(np.dot(a_mlp, g))

                closure = float(
                    np.linalg.norm(a_real - (a_attn + a_mlp))
                    / max(float(np.linalg.norm(a_real)), EPS)
                )

                rows.append(
                    {
                        "sid": int(m["sid"]),
                        "gt": str(m["gt"]),
                        "competitor": str(m["competitor"]),
                        "baseline_prediction": str(m["baseline_prediction"]),
                        "baseline_correct": bool(m["baseline_correct"]),
                        "prior_sequence_margin": float(m["prior_sequence_margin"]),
                        "replayed_sequence_margin": replay_margin,
                        "real_position": p,
                        "token": str(s["token"]),
                        "category": str(s["category"]),
                        "broad_category": str(s["broad_category"]),
                        "max_target_layer": cmax,
                        "update_layer": L,
                        "B_real": B,
                        "B_attn": B_attn,
                        "B_mlp": B_mlp,
                        "real_update_norm": float(np.linalg.norm(a_real)),
                        "reader_grad_norm": float(np.linalg.norm(g)),
                        "block_in_norm": float(np.linalg.norm(x_in)),
                        "post_attn_norm": float(np.linalg.norm(post_attn)),
                        "block_out_norm": float(np.linalg.norm(x_out)),
                        "module_vector_closure_relative_error": closure,
                    }
                )

                vectors["block_in"].append(x_in)
                vectors["post_attn"].append(post_attn)
                vectors["block_out"].append(x_out)
                vectors["attn_update"].append(a_attn)
                vectors["mlp_update"].append(a_mlp)
                vectors["real_update"].append(a_real)
                vectors["reader_g"].append(g)

        return rows, vectors

    finally:
        with contextlib.suppress(Exception):
            image.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Random projection for coarse held-out sign localization
# =============================================================================

def make_projection(hidden_dim, probe_dim, seed):
    if probe_dim <= 0:
        return None
    rng = np.random.default_rng(seed)
    # Gaussian JL projection. Scale does not matter for logistic regression,
    # but 1/sqrt(k) keeps norms stable.
    R = rng.standard_normal(
        (hidden_dim, probe_dim),
        dtype=np.float32,
    )
    R /= math.sqrt(float(probe_dim))
    return R


def project_vector(v, R):
    if R is None:
        return None
    return np.asarray(v, dtype=np.float32) @ R


# =============================================================================
# Matched writer-reader cross swap
# =============================================================================

def build_pairings(
    df,
    *,
    keys,
    repeat,
    seed,
    min_each_sign,
):
    """
    One-to-one random +/- pairing inside each matched stratum.
    Prefer different SIDs; if impossible, permit same-SID pair rather than
    discarding the entire stratum, and record it.
    """
    rng = np.random.default_rng(seed + 1009 * repeat)
    pairs = []

    for _, g in df.groupby(keys, dropna=False, sort=False):
        pos = g[g["sign"] > 0]["row_index"].to_numpy(int)
        neg = g[g["sign"] < 0]["row_index"].to_numpy(int)

        if (
            len(pos) < min_each_sign
            or len(neg) < min_each_sign
        ):
            continue

        pos = pos.copy()
        neg = neg.copy()
        rng.shuffle(pos)
        rng.shuffle(neg)

        n = min(len(pos), len(neg))
        pos = pos[:n]
        neg = neg[:n]

        # Try to reduce same-SID pairings with cyclic shifts.
        if n > 1:
            pos_sids = df.set_index("row_index").loc[pos, "sid"].to_numpy()
            best_neg = neg
            best_same = n + 1
            for shift in range(min(n, 16)):
                cand = np.roll(neg, shift)
                neg_sids = df.set_index("row_index").loc[cand, "sid"].to_numpy()
                same = int(np.sum(pos_sids == neg_sids))
                if same < best_same:
                    best_same = same
                    best_neg = cand
                if same == 0:
                    break
            neg = best_neg

        pairs.extend(zip(pos.tolist(), neg.tolist()))

    return pairs


def cross_swap_for_scope(
    *,
    df,
    A,
    G,
    cohort_name,
    match_mode,
    pair_repeats,
    seed,
    min_each_sign,
    b_threshold,
):
    keys = MATCH_MODES[match_mode]
    summaries = []
    examples = []

    scope = df.copy()
    if len(scope) == 0:
        return summaries, examples

    for rep in range(pair_repeats):
        pairs = build_pairings(
            scope,
            keys=keys,
            repeat=rep,
            seed=seed,
            min_each_sign=min_each_sign,
        )
        if not pairs:
            continue

        pidx = np.asarray([p for p, _ in pairs], dtype=int)
        nidx = np.asarray([n for _, n in pairs], dtype=int)

        ap = np.asarray(A[pidx], dtype=np.float32)
        an = np.asarray(A[nidx], dtype=np.float32)
        gp = np.asarray(G[pidx], dtype=np.float32)
        gn = np.asarray(G[nidx], dtype=np.float32)

        Bpp = np.sum(ap * gp, axis=1)
        Bnn = np.sum(an * gn, axis=1)
        Bnp = np.sum(an * gp, axis=1)  # negative writer + positive reader
        Bpn = np.sum(ap * gn, axis=1)  # positive writer + negative reader

        snp = sign_arr(Bnp, b_threshold)
        spn = sign_arr(Bpn, b_threshold)

        writer_follow = (snp < 0) & (spn > 0)
        reader_follow = (snp > 0) & (spn < 0)
        mixed = ~(writer_follow | reader_follow)

        # Independent one-sided persistence metrics are often more stable than
        # requiring BOTH cross-dots to behave the same way.
        neg_writer_persists = snp < 0
        pos_writer_persists = spn > 0
        positive_reader_persists = snp > 0
        negative_reader_persists = spn < 0

        same_sid = (
            scope.set_index("row_index").loc[pidx, "sid"].to_numpy()
            ==
            scope.set_index("row_index").loc[nidx, "sid"].to_numpy()
        )

        summaries.append(
            {
                "cohort": cohort_name,
                "match_mode": match_mode,
                "repeat": rep,
                "N_pairs": int(len(pairs)),
                "same_sid_pair_fraction": float(np.mean(same_sid)),
                "writer_follow_both_fraction": float(np.mean(writer_follow)),
                "reader_follow_both_fraction": float(np.mean(reader_follow)),
                "mixed_fraction": float(np.mean(mixed)),
                "negative_writer_persists_fraction": float(
                    np.mean(neg_writer_persists)
                ),
                "positive_writer_persists_fraction": float(
                    np.mean(pos_writer_persists)
                ),
                "positive_reader_persists_fraction": float(
                    np.mean(positive_reader_persists)
                ),
                "negative_reader_persists_fraction": float(
                    np.mean(negative_reader_persists)
                ),
                "mean_B_neg_writer_pos_reader": float(np.mean(Bnp)),
                "mean_B_pos_writer_neg_reader": float(np.mean(Bpn)),
                "mean_cos_neg_writer_pos_reader": safe_mean(
                    cosine_rows(an, gp)
                ),
                "mean_cos_pos_writer_neg_reader": safe_mean(
                    cosine_rows(ap, gn)
                ),
                "sanity_original_positive_fraction": float(
                    np.mean(Bpp > b_threshold)
                ),
                "sanity_original_negative_fraction": float(
                    np.mean(Bnn < -b_threshold)
                ),
            }
        )

        if rep == 0:
            # Save enough pair detail to inspect concrete examples without
            # exploding file size over repeated re-pairings.
            idx_to_row = scope.set_index("row_index")
            for j, (pi, ni) in enumerate(pairs):
                pr = idx_to_row.loc[pi]
                nr = idx_to_row.loc[ni]
                examples.append(
                    {
                        "cohort": cohort_name,
                        "match_mode": match_mode,
                        "pair_id": j,
                        "positive_row_index": int(pi),
                        "negative_row_index": int(ni),
                        "positive_sid": int(pr["sid"]),
                        "negative_sid": int(nr["sid"]),
                        "update_layer": int(pr["update_layer"]),
                        "gt": str(pr["gt"]),
                        "competitor": str(pr["competitor"]),
                        "positive_position": int(pr["real_position"]),
                        "negative_position": int(nr["real_position"]),
                        "positive_category": str(pr["category"]),
                        "negative_category": str(nr["category"]),
                        "positive_B_original": float(Bpp[j]),
                        "negative_B_original": float(Bnn[j]),
                        "B_neg_writer_pos_reader": float(Bnp[j]),
                        "B_pos_writer_neg_reader": float(Bpn[j]),
                        "cross_class": (
                            "writer_follow"
                            if writer_follow[j]
                            else "reader_follow"
                            if reader_follow[j]
                            else "mixed"
                        ),
                    }
                )

    return summaries, examples


def aggregate_cross_swap(rep_df):
    if not len(rep_df):
        return pd.DataFrame()

    metric_cols = [
        "N_pairs",
        "same_sid_pair_fraction",
        "writer_follow_both_fraction",
        "reader_follow_both_fraction",
        "mixed_fraction",
        "negative_writer_persists_fraction",
        "positive_writer_persists_fraction",
        "positive_reader_persists_fraction",
        "negative_reader_persists_fraction",
        "mean_B_neg_writer_pos_reader",
        "mean_B_pos_writer_neg_reader",
        "mean_cos_neg_writer_pos_reader",
        "mean_cos_pos_writer_neg_reader",
        "sanity_original_positive_fraction",
        "sanity_original_negative_fraction",
    ]

    rows = []
    for (cohort, mode), g in rep_df.groupby(
        ["cohort", "match_mode"],
        sort=False,
    ):
        row = {
            "cohort": cohort,
            "match_mode": mode,
            "repeats": int(len(g)),
            "mean_N_pairs": safe_mean(g["N_pairs"]),
        }
        for c in metric_cols[1:]:
            row[c] = safe_mean(g[c])
            row[f"{c}_std"] = float(
                np.nanstd(
                    pd.to_numeric(g[c], errors="coerce").to_numpy(float)
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Matched, held-out SID linear probe
# =============================================================================

def balanced_indices_within_strata(
    df,
    keys,
    seed,
):
    rng = np.random.default_rng(seed)
    chosen = []

    for _, g in df.groupby(keys, dropna=False, sort=False):
        pos = g[g["sign"] > 0]["row_index"].to_numpy(int)
        neg = g[g["sign"] < 0]["row_index"].to_numpy(int)
        n = min(len(pos), len(neg))
        if n <= 0:
            continue

        if len(pos) > n:
            pos = rng.choice(pos, n, replace=False)
        if len(neg) > n:
            neg = rng.choice(neg, n, replace=False)

        chosen.extend(pos.tolist())
        chosen.extend(neg.tolist())

    return np.asarray(sorted(set(chosen)), dtype=int)


def run_probe_cv(
    X,
    y,
    groups,
    n_splits,
    C,
    seed,
):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            splits = splitter.split(X, y, groups)
        except Exception:
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups)
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for the optional probe scan. "
            "Install sklearn or run with --probe-dim 0. "
            f"Original error: {type(exc).__name__}: {exc}"
        )

    rows = []
    for fold, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(C),
                max_iter=2000,
                class_weight=None,
                solver="liblinear",
                random_state=seed + fold,
            ),
        )
        clf.fit(X[tr], y[tr])

        pred = clf.predict(X[te])
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X[te])[:, 1]
        else:
            prob = clf.decision_function(X[te])

        rows.append(
            {
                "fold": int(fold),
                "N_train": int(len(tr)),
                "N_test": int(len(te)),
                "accuracy": float(accuracy_score(y[te], pred)),
                "auc": float(roc_auc_score(y[te], prob)),
                "positive_fraction_test": float(np.mean(y[te])),
            }
        )

    return pd.DataFrame(rows)


def probe_stage(
    *,
    df,
    projected,
    cohort,
    stage,
    match_mode,
    folds,
    min_rows,
    C,
    seed,
):
    scope = df.copy()
    if cohort == "baseline_wrong":
        scope = scope[scope["baseline_correct"] == False].copy()  # noqa: E712
    elif cohort == "baseline_correct":
        scope = scope[scope["baseline_correct"] == True].copy()  # noqa: E712

    if len(scope) == 0:
        return None, None

    idx = balanced_indices_within_strata(
        scope,
        MATCH_MODES[match_mode],
        seed,
    )
    if len(idx) < min_rows:
        return None, None

    sel = df.set_index("row_index").loc[idx]
    X = np.asarray(projected[stage][idx], dtype=np.float32)
    y = (sel["sign"].to_numpy(int) > 0).astype(int)
    groups = sel["sid"].to_numpy(int)

    n_groups = len(np.unique(groups))
    n_splits = min(int(folds), int(n_groups))
    if n_splits < 2:
        return None, None

    fold_df = run_probe_cv(
        X,
        y,
        groups,
        n_splits,
        C,
        seed,
    )
    if not len(fold_df):
        return None, None

    summary = {
        "cohort": cohort,
        "stage": stage,
        "match_mode": match_mode,
        "N_balanced_rows": int(len(idx)),
        "N_sids": int(n_groups),
        "folds_completed": int(len(fold_df)),
        "accuracy_mean": safe_mean(fold_df["accuracy"]),
        "accuracy_std": float(np.nanstd(fold_df["accuracy"])),
        "auc_mean": safe_mean(fold_df["auc"]),
        "auc_std": float(np.nanstd(fold_df["auc"])),
    }
    fold_df["cohort"] = cohort
    fold_df["stage"] = stage
    fold_df["match_mode"] = match_mode

    return summary, fold_df


def probe_by_layer(
    *,
    df,
    projected,
    cohort,
    stage,
    match_mode,
    folds,
    min_rows,
    C,
    seed,
):
    rows = []

    for L in sorted(df["update_layer"].unique()):
        scope = df[df["update_layer"] == L].copy()
        if cohort == "baseline_wrong":
            scope = scope[scope["baseline_correct"] == False].copy()  # noqa: E712
        elif cohort == "baseline_correct":
            scope = scope[scope["baseline_correct"] == True].copy()  # noqa: E712

        if len(scope) == 0:
            continue

        idx = balanced_indices_within_strata(
            scope,
            MATCH_MODES[match_mode],
            seed + int(L),
        )
        if len(idx) < min_rows:
            continue

        sel = df.set_index("row_index").loc[idx]
        X = np.asarray(projected[stage][idx], dtype=np.float32)
        y = (sel["sign"].to_numpy(int) > 0).astype(int)
        groups = sel["sid"].to_numpy(int)

        n_groups = len(np.unique(groups))
        n_splits = min(int(folds), int(n_groups))
        if n_splits < 2:
            continue

        try:
            fdf = run_probe_cv(
                X,
                y,
                groups,
                n_splits,
                C,
                seed + int(L),
            )
        except Exception:
            continue

        if not len(fdf):
            continue

        rows.append(
            {
                "cohort": cohort,
                "stage": stage,
                "match_mode": match_mode,
                "update_layer": int(L),
                "N_balanced_rows": int(len(idx)),
                "N_sids": int(n_groups),
                "accuracy_mean": safe_mean(fdf["accuracy"]),
                "auc_mean": safe_mean(fdf["auc"]),
            }
        )

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    trace_layers = src.parse_layers(a.trace_layers)
    match_modes = selected_match_modes(a.match_modes)

    run_dir = Path(a.real_update_dir)

    (
        cohort,
        selected_by_sid,
        prior_update,
        prior_metadata,
    ) = src.load_prior_run(
        run_dir,
        int(a.max_samples),
        int(a.seed),
    )

    two, meta, rec_by_sid = src.load_dataset_for_sids(
        a,
        cohort,
    )

    rows_all = []
    full_vectors = {stage: [] for stage in STAGES}
    projected = {stage: [] for stage in STAGES}

    model = processor = None
    projection = None
    hidden_dim = None

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        ) = src.load_model(a, two)

        n_layers = len(decoder_layers)
        bad = [L for L in trace_layers if not 0 <= L < n_layers]
        if bad:
            raise ValueError(
                f"trace layers outside 0..{n_layers-1}: {bad}"
            )

        candidate_texts = src.get_candidate_texts(prior_metadata)
        candidate_ids = src.encode_candidate_ids(
            processor,
            candidate_texts,
        )
        reduction = str(
            prior_metadata.get(
                "sequence_score_reduction",
                "mean",
            )
        )
        if reduction not in ("mean", "sum"):
            reduction = "mean"

        print("=" * 190)
        print("WRITER vs READER LATENT-CAUSE DIAGNOSTIC")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N samples={len(meta)}")
        print(f"trace_layers={trace_layers}")
        print(f"pair_repeats={a.pair_repeats}")
        print(f"match_modes={match_modes}")
        print(f"probe_dim={a.probe_dim}")
        print()

        for m in tqdm(meta, desc="CAPTURE a / g / intermediate states"):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            try:
                sample_rows, vecs = collect_sample_rows(
                    m=m,
                    causal_rows=selected_by_sid[sid],
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    rec=rec_by_sid[sid],
                    candidate_ids=candidate_ids,
                    reduction=reduction,
                    trace_layers=trace_layers,
                    exclude_target_layer=bool(a.exclude_target_layer),
                )

                if not sample_rows:
                    continue

                if hidden_dim is None:
                    hidden_dim = int(vecs["real_update"][0].shape[0])
                    projection = make_projection(
                        hidden_dim,
                        int(a.probe_dim),
                        int(a.seed) + 99173,
                    )

                for j, row in enumerate(sample_rows):
                    rows_all.append(row)

                    for stage in STAGES:
                        v = np.asarray(vecs[stage][j], dtype=np.float32)
                        # Keep a/g full precision enough for cross dot products.
                        if stage in ("real_update", "reader_g"):
                            full_vectors[stage].append(v.astype(np.float16))
                        elif a.save_full_vectors:
                            full_vectors[stage].append(v.astype(np.float16))

                        if projection is not None:
                            projected[stage].append(
                                project_vector(v, projection).astype(np.float16)
                            )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-60:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows_all:
        raise RuntimeError("No updates were captured.")

    df = pd.DataFrame(rows_all)
    df["row_index"] = np.arange(len(df), dtype=int)
    df["sign"] = sign_arr(
        df["B_real"].to_numpy(float),
        float(a.b_threshold),
    )
    df["polarity"] = np.where(
        df["sign"] > 0,
        "positive",
        np.where(df["sign"] < 0, "negative", "neutral"),
    )

    # Keep only non-neutral rows for cross swap / probes.
    active = df[df["sign"] != 0].copy()

    A = np.asarray(full_vectors["real_update"], dtype=np.float16)
    G = np.asarray(full_vectors["reader_g"], dtype=np.float16)

    if len(A) != len(df) or len(G) != len(df):
        raise RuntimeError(
            f"Vector/row mismatch: A={len(A)} G={len(G)} rows={len(df)}"
        )

    # Convert projected lists to arrays in exact row order.
    proj_arrays = {}
    if projection is not None:
        for stage in STAGES:
            arr = np.asarray(projected[stage], dtype=np.float16)
            if len(arr) != len(df):
                raise RuntimeError(
                    f"Projected stage {stage} rows={len(arr)} != metadata {len(df)}"
                )
            proj_arrays[stage] = arr

    df.to_csv(
        outdir / "per_update_metadata.csv",
        index=False,
    )

    if a.save_full_vectors:
        save_payload = {
            "row_index": df["row_index"].to_numpy(int),
            "real_update": A,
            "reader_g": G,
        }
        for stage in STAGES:
            if stage in ("real_update", "reader_g"):
                continue
            if len(full_vectors[stage]) == len(df):
                save_payload[stage] = np.asarray(
                    full_vectors[stage],
                    dtype=np.float16,
                )
        np.savez_compressed(
            outdir / "full_latent_vectors_fp16.npz",
            **save_payload,
        )

    # -------------------------------------------------------------------------
    # A) Matched a/g cross swap.
    # -------------------------------------------------------------------------
    rep_rows = []
    example_rows = []

    cohort_defs = {
        "all": active,
        "baseline_wrong": active[
            active["baseline_correct"] == False  # noqa: E712
        ],
        "baseline_correct": active[
            active["baseline_correct"] == True  # noqa: E712
        ],
    }

    for cohort_name, scope in cohort_defs.items():
        for mode in match_modes:
            srows, erows = cross_swap_for_scope(
                df=scope,
                A=A,
                G=G,
                cohort_name=cohort_name,
                match_mode=mode,
                pair_repeats=int(a.pair_repeats),
                seed=int(a.seed),
                min_each_sign=int(a.min_stratum_each_sign),
                b_threshold=float(a.b_threshold),
            )
            rep_rows.extend(srows)
            example_rows.extend(erows)

    cross_repeats = pd.DataFrame(rep_rows)
    cross_examples = pd.DataFrame(example_rows)
    cross_summary = aggregate_cross_swap(cross_repeats)

    cross_repeats.to_csv(
        outdir / "cross_swap_repeats.csv",
        index=False,
    )
    cross_summary.to_csv(
        outdir / "cross_swap_summary.csv",
        index=False,
    )
    cross_examples.to_csv(
        outdir / "cross_swap_pair_examples.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # B) Coarse first-sign-separability scan.
    # -------------------------------------------------------------------------
    probe_summary_rows = []
    probe_fold_parts = []
    probe_layer_rows = []

    if projection is not None:
        for cohort_name in (
            "all",
            "baseline_wrong",
            "baseline_correct",
        ):
            for stage in STAGES:
                try:
                    summary, fdf = probe_stage(
                        df=active,
                        projected=proj_arrays,
                        cohort=cohort_name,
                        stage=stage,
                        match_mode=a.probe_match_mode,
                        folds=int(a.probe_folds),
                        min_rows=int(a.probe_min_rows),
                        C=float(a.probe_c),
                        seed=int(a.seed),
                    )
                    if summary is not None:
                        probe_summary_rows.append(summary)
                    if fdf is not None:
                        probe_fold_parts.append(fdf)

                    probe_layer_rows.extend(
                        probe_by_layer(
                            df=active,
                            projected=proj_arrays,
                            cohort=cohort_name,
                            stage=stage,
                            match_mode=a.probe_match_mode,
                            folds=int(a.probe_folds),
                            min_rows=max(40, int(a.probe_min_rows) // 2),
                            C=float(a.probe_c),
                            seed=int(a.seed),
                        )
                    )

                except Exception as exc:
                    append_jsonl(
                        error_path,
                        {
                            "phase": "probe",
                            "cohort": cohort_name,
                            "stage": stage,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback_tail": traceback.format_exc().splitlines()[-40:],
                        },
                    )

    probe_summary = pd.DataFrame(probe_summary_rows)
    probe_folds = (
        pd.concat(probe_fold_parts, ignore_index=True)
        if probe_fold_parts
        else pd.DataFrame()
    )
    probe_layers = pd.DataFrame(probe_layer_rows)

    probe_summary.to_csv(
        outdir / "probe_stage_summary.csv",
        index=False,
    )
    probe_folds.to_csv(
        outdir / "probe_stage_folds.csv",
        index=False,
    )
    probe_layers.to_csv(
        outdir / "probe_stage_by_layer.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # C) Report.
    # -------------------------------------------------------------------------
    report = [
        "=" * 190,
        "WRITER vs READER LATENT-CAUSE DIAGNOSTIC",
        "=" * 190,
        (
            f"N updates={len(df)} | active={len(active)} | "
            f"N samples={df['sid'].nunique()} | hidden_dim={hidden_dim}"
        ),
        (
            f"positive={int((active['sign']>0).sum())} | "
            f"negative={int((active['sign']<0).sum())}"
        ),
        "",
        "A. MATCHED a/g CROSS-SWAP",
        "-" * 190,
        (
            cross_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(cross_summary)
            else "EMPTY"
        ),
        "",
        "How to read cross-swap:",
        "  writer_follow_both:",
        "      a-·g+ stays negative AND a+·g- stays positive.",
        "      High value => polarity is carried mainly by the writer/update geometry.",
        "  reader_follow_both:",
        "      a-·g+ becomes positive AND a+·g- becomes negative.",
        "      High value => polarity follows the downstream reader/gradient.",
        "  mixed:",
        "      neither simple factorization works; sign is pair-specific compatibility.",
        "",
        "Most important cohort is baseline_wrong and the strictest match mode",
        "that still has a reasonable number of pairs.",
        "",
        "B. HELD-OUT SID SIGN SEPARABILITY (MATCH-BALANCED)",
        "-" * 190,
        (
            probe_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(probe_summary)
            else "PROBE DISABLED / EMPTY"
        ),
        "",
        "Probe reading:",
        "  block_in high:",
        "      sign-relevant latent state already exists before this block.",
        "  post_attn jumps above block_in:",
        "      Attention creates/reveals sign-separable structure.",
        "  mlp_update / block_out jump:",
        "      MLP transformation is a likely formation site.",
        "  real_update high but block_in low:",
        "      writer geometry is formed inside the block.",
        "  reader_g high while writer states are weak:",
        "      downstream reader state is a stronger candidate.",
        "",
        "Do NOT interpret a probe jump as causality by itself.",
        "The next causal test is to patch the earliest strongly separable state and",
        "measure whether B and generation actually change.",
    ]

    report_text = "\n".join(report) + "\n"
    print(report_text)
    (outdir / "analysis_summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script": "diagnose_writer_reader_latent_cause_v1.py",
        "model": a.model,
        "real_update_dir": str(run_dir),
        "N_samples": int(df["sid"].nunique()),
        "N_updates": int(len(df)),
        "N_active_updates": int(len(active)),
        "trace_layers": trace_layers,
        "b_threshold": float(a.b_threshold),
        "pair_repeats": int(a.pair_repeats),
        "match_modes": match_modes,
        "probe_dim": int(a.probe_dim),
        "probe_match_mode": str(a.probe_match_mode),
        "probe_folds": int(a.probe_folds),
        "sequence_score_reduction": str(
            prior_metadata.get("sequence_score_reduction", "mean")
        ),
        "writer_definition": "a_real = block_out - block_in",
        "reader_definition": "g = grad(S_GT - S_competitor) wrt block_out",
        "behavior_definition": "B = dot(a_real, g)",
        "cross_swap_definition": {
            "B_neg_writer_pos_reader": "dot(a_negative, g_positive)",
            "B_pos_writer_neg_reader": "dot(a_positive, g_negative)",
        },
        "probe_stages": list(STAGES),
        "probe_note": (
            "Random-projected linear probe, positive/negative balanced within "
            f"{a.probe_match_mode} strata, CV grouped by sid."
        ),
        "uses_GT": True,
        "generation_performed": False,
        "model_intervention_performed": False,
    }

    (outdir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
