#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELF-PREDICT each sample's causal-core tokens WITHOUT LEFT/RIGHT/ON/UNDER routing.

Target:
  Existing oracle dynamic run:
    source band L20..L26, global_unique K36.

Ground-truth core labels are used ONLY FOR EVALUATION:
  Core50 / Core70 / Core80 = the smallest prefix of that sample's selected K36
  covering 50% / 70% / 80% of its positive mediation mass.

What predicts the core:
  ONLY relation-free information from the current sample:
    - next-layer last->token attention change
    - next-layer attention magnitude / head dispersion
    - same-layer attention features
    - hidden Real-Gray magnitude
    - attention-output / MLP-output magnitude
    - similarity to the sample's own source/late Real-Gray last-state delta
    - rank ensembles of the above

It evaluates:
  1) fixed-size self selection:
       use the GLOBAL MEDIAN Core50/70/80 size, not each sample's true size.
  2) oracle-size localization diagnostic:
       give the selector only the number of true core tokens, NOT their identity.
       This separates "where are the tokens?" from "how many should I choose?"
  3) exact (layer, position) recall / precision / Jaccard
  4) position-only recall
  5) +/-1-layer tolerant recall for the same token position
  6) oracle mediation mass captured by the predicted relation-free set

Also includes an OPTIONAL leave-one-sample-out layer-role frequency prior:
  prior(core | layer, broad_role), estimated from the other 439 samples only.
This uses no relation label, but it IS calibration-dependent and is reported
separately from pure self-signal selectors.

Requirements:
  Put one of these earlier scripts in the same repo directory:
    analyze_dynamic_k24_predictive_correlates_v3.py  (preferred)
    analyze_dynamic_k24_predictive_correlates_v2.py
    analyze_dynamic_k24_predictive_correlates_v1.py

The script caches forward-pass features by SID. If interrupted, rerun WITHOUT
--overwrite and it resumes.

Example:
CUDA_VISIBLE_DEVICES=0 python -u eval_self_predict_causal_core_v1.py \
  --run-dir output/qwen3b_coco_dynamic_L20_26_K36_all440 \
  --model qwen-3b \
  --bundle "L20+L21+L22+L23+L24+L25+L26[global_unique]" \
  --k 36 \
  --target-layers 32,34,35 \
  --core-masses 0.5,0.7,0.8 \
  --with-attention-weights \
  --attention-head-scan \
  --output-dir output/qwen3b_coco_L20_26_self_predict_core \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor
from tqdm import tqdm


# =============================================================================
# Import the already-tested feature collector from the previous experiment.
# =============================================================================

CORR = None
CORR_NAME = None
for _name in [
    "analyze_dynamic_k24_predictive_correlates_v3",
    "analyze_dynamic_k24_predictive_correlates_v2",
    "analyze_dynamic_k24_predictive_correlates_v1",
]:
    try:
        CORR = importlib.import_module(_name)
        CORR_NAME = _name
        break
    except ModuleNotFoundError:
        pass

if CORR is None:
    raise RuntimeError(
        "Could not import the earlier feature collector. Put "
        "analyze_dynamic_k24_predictive_correlates_v3.py (preferred) "
        "in the same directory as this script."
    )

base = CORR.base


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument(
        "--bundle",
        default="L20+L21+L22+L23+L24+L25+L26[global_unique]",
    )
    p.add_argument("--k", type=int, default=36)
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--core-masses", default="0.5,0.7,0.8")
    p.add_argument(
        "--with-attention-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--attention-head-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument(
        "--max-pred-k",
        type=int,
        default=36,
        help="Upper bound when a self-gap selector estimates its own K.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def fmt_mass(x):
    return f"core{int(round(100*x))}"


def canon_rel(x):
    x = str(x).strip().lower()
    return {"above": "on", "below": "under"}.get(x, x)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    good = np.isfinite(score)
    y = y[good]
    score = score[good]
    if len(y) < 4 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def percentile_rank_high(s):
    """
    0..1, larger raw value -> larger rank. NaNs stay NaN.
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(method="average", pct=True)


def within_sid_layer_rank(df, feature):
    return (
        df.groupby(["sid", "source_layer"], group_keys=False)[feature]
        .transform(percentile_rank_high)
        .astype(float)
    )


def unique_position_sorted(g, score_col):
    """
    global_unique-compatible ranking:
      sort all (layer,position) candidates by score, then keep only the first
      occurrence of each token position across layers.
    """
    q = g[np.isfinite(pd.to_numeric(g[score_col], errors="coerce"))].copy()
    if not len(q):
        return q
    q = q.sort_values(
        [score_col, "source_layer", "position"],
        ascending=[False, True, True],
    )
    q = q.drop_duplicates("position", keep="first")
    return q.reset_index(drop=True)


def largest_relative_gap_k(q, score_col, max_k=36, min_k=2):
    """
    Fully self-derived token count. Uses the largest relative drop among the
    first max_k unique-position scores. Diagnostic, because score distributions
    can be smooth.
    """
    if not len(q):
        return 0
    vals = pd.to_numeric(q[score_col], errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    n = min(len(vals), int(max_k))
    vals = vals[:n]
    if n <= min_k:
        return n

    # Shift to positive only for stable relative-drop computation.
    shifted = vals - np.min(vals) + 1e-8
    drops = (shifted[:-1] - shifted[1:]) / (np.abs(shifted[:-1]) + 1e-8)

    start = max(1, int(min_k)) - 1
    if start >= len(drops):
        return n
    j = start + int(np.argmax(drops[start:]))
    return int(j + 1)


# =============================================================================
# Existing run + core ground truth
# =============================================================================

def load_run(a):
    source_layers = CORR.infer_source_layers(a.bundle)
    target_layers = parse_ints(a.target_layers)

    ns = SimpleNamespace(
        run_dir=a.run_dir,
        condition=a.condition,
        selection_strategy=a.selection_strategy,
        bundle=a.bundle,
        k=a.k,
        max_eval_samples=a.max_eval_samples,
        seed=a.seed,
    )
    med, sel, baseline, generation, best_alpha, valid_sids = (
        CORR.load_run_tables(ns, source_layers)
    )

    med["relation"] = med["relation"].map(canon_rel)
    sel["relation"] = sel["relation"].map(canon_rel)

    return (
        med, sel, baseline, generation, best_alpha,
        valid_sids, source_layers, target_layers
    )


def build_core_truth(sel, core_masses):
    """
    Returns:
      truth_by_target[target][sid] = set((layer,pos))
      truth_pos_by_target[target][sid] = set(pos)
      size summary
      selected positive-M lookup
    """
    truth = {fmt_mass(t): {} for t in core_masses}
    truth_pos = {fmt_mass(t): {} for t in core_masses}
    size_rows = []
    selected_m_lookup = {}
    selected_total_mass = {}

    for sid, g0 in sel.groupby("sid", sort=True):
        g = g0.sort_values("mediation", ascending=False).copy()
        m = pd.to_numeric(g["mediation"], errors="coerce").to_numpy(np.float64)
        posm = np.maximum(np.where(np.isfinite(m), m, 0.0), 0.0)
        total = float(posm.sum())
        cum = np.cumsum(posm) / total if total > 0 else np.zeros(len(g))

        selected_total_mass[int(sid)] = total

        for r in g.itertuples():
            selected_m_lookup[
                (int(sid), int(r.source_layer), int(r.position))
            ] = max(float(r.mediation), 0.0)

        row = {
            "sid": int(sid),
            "relation": canon_rel(g["relation"].iloc[0]),
            "selected_positive_mass": total,
        }

        for t in core_masses:
            name = fmt_mass(t)
            idx = np.where(cum >= t)[0]
            kk = int(idx[0] + 1) if len(idx) else len(g)
            core = g.iloc[:kk]

            exact = set(
                (int(r.source_layer), int(r.position))
                for r in core.itertuples()
            )
            positions = set(int(r.position) for r in core.itertuples())

            truth[name][int(sid)] = exact
            truth_pos[name][int(sid)] = positions
            row[f"{name}_N"] = len(exact)
            row[f"{name}_actual_mass_fraction"] = (
                float(posm[:kk].sum() / total) if total > 0 else np.nan
            )

        size_rows.append(row)

    return (
        truth,
        truth_pos,
        pd.DataFrame(size_rows),
        selected_m_lookup,
        selected_total_mass,
    )


# =============================================================================
# Relation-free feature collection
# =============================================================================

def collect_or_load_features(
    a,
    outdir,
    med,
    sel,
    baseline,
    valid_sids,
    source_layers,
    target_layers,
):
    cache_dir = outdir / "feature_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If all chunks exist, skip model load entirely.
    all_cached = all(
        (cache_dir / f"sid_{sid}.pkl.gz").exists()
        for sid in valid_sids
    )
    if all_cached:
        print("All per-sample feature chunks already cached; skipping VLM forward.")
        parts = [
            pd.read_pickle(cache_dir / f"sid_{sid}.pkl.gz", compression="gzip")
            for sid in valid_sids
        ]
        return pd.concat(parts, ignore_index=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    specs = base.merged_model_specs(two)
    spec = specs[a.model]

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    model_cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = None
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        base.configure_processor(model, processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        hidden_layers = sorted(set(source_layers + target_layers))
        module_layers = list(source_layers)
        requested_attn_layers = sorted(
            {
                L
                for L in (
                    source_layers
                    + [x + 1 for x in source_layers]
                    + target_layers
                )
                if 0 <= L < n_layers
            }
        )

        print("Feature collector:", CORR_NAME)
        print("decoder:", decoder_path)
        print("source layers:", source_layers)
        print("attention layers requested:", requested_attn_layers)

        for sid in tqdm(valid_sids, desc="Collect self-prediction features"):
            chunk_path = cache_dir / f"sid_{sid}.pkl.gz"
            if chunk_path.exists():
                continue

            if sid not in prompts or sid not in rec_by_sid:
                raise RuntimeError(f"sid={sid} missing prompt or data record")

            pr = prompts[sid]
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=str(pr["question_text"]),
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=str(pr["question_text"]),
                    device=device,
                )

                real_cap = CORR.rich_forward(
                    model,
                    decoder_layers,
                    rb,
                    hidden_layers,
                    module_layers,
                    requested_attn_layers,
                    a.with_attention_weights,
                )
                gray_cap = CORR.rich_forward(
                    model,
                    decoder_layers,
                    gb,
                    hidden_layers,
                    module_layers,
                    requested_attn_layers,
                    a.with_attention_weights,
                )

                sid_med = med[med["sid"] == sid].copy()
                token_chunk, _sample_feat = CORR.build_sample_features(
                    sid_med,
                    source_layers,
                    target_layers,
                    real_cap,
                    gray_cap,
                    a.with_attention_weights,
                    a.attention_head_scan,
                )
                token_chunk.to_pickle(chunk_path, compression="gzip")

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        if model is not None:
            del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parts = [
        pd.read_pickle(cache_dir / f"sid_{sid}.pkl.gz", compression="gzip")
        for sid in valid_sids
    ]
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# Build pure self scores
# =============================================================================

def available(df, name):
    return name in df.columns and df[name].notna().sum() > 0


def mean_existing(df, cols, out_name):
    valid = [c for c in cols if available(df, c)]
    if not valid:
        return None

    X = np.stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in valid],
        axis=1,
    )
    finite = np.isfinite(X)
    n = finite.sum(axis=1)
    val = np.full(len(df), np.nan, dtype=np.float64)
    sums = np.where(finite, X, 0.0).sum(axis=1)
    np.divide(sums, n, out=val, where=n > 0)
    df[out_name] = val
    return out_name


def add_self_scores(token_df):
    """
    Add relation-free scores. Raw features are converted to within-SID-layer
    percentile ranks before ensembles so different layers/scales are comparable.
    """
    df = token_df.copy()

    raw_candidates = [
        "last_attn_next_max_positive_delta",
        "last_attn_next_delta_mean",
        "last_attn_next_real_mean",
        "last_attn_next_real_std_heads",
        "last_attn_same_real_mean",
        "last_attn_same_real_max",
        "hidden_delta_norm_recomputed",
        "attn_out_delta_norm",
        "mlp_out_delta_norm",
        "delta_cos_source_last_delta",
        "delta_cos_late_mean",
        "delta_cos_late_max",
    ]

    rank_cols = []
    for c in raw_candidates:
        if available(df, c):
            rc = "RANK_" + c
            df[rc] = within_sid_layer_rank(df, c)
            rank_cols.append(rc)

    selectors = []

    # Raw attention score (the strongest previous single feature).
    if available(df, "last_attn_next_max_positive_delta"):
        selectors.append(
            ("attn_next_posdelta_raw",
             "last_attn_next_max_positive_delta",
             "pure_self")
        )
        selectors.append(
            ("attn_next_posdelta_rank",
             "RANK_last_attn_next_max_positive_delta",
             "pure_self")
        )

    if available(df, "last_attn_next_delta_mean"):
        selectors.append(
            ("attn_next_delta_rank",
             "RANK_last_attn_next_delta_mean",
             "pure_self")
        )

    if available(df, "last_attn_next_real_mean"):
        selectors.append(
            ("attn_next_real_rank",
             "RANK_last_attn_next_real_mean",
             "pure_self")
        )

    # Attention ensemble.
    attn_rank_features = [
        "RANK_last_attn_next_max_positive_delta",
        "RANK_last_attn_next_delta_mean",
        "RANK_last_attn_next_real_mean",
        "RANK_last_attn_next_real_std_heads",
        "RANK_last_attn_same_real_mean",
        "RANK_last_attn_same_real_max",
    ]
    if mean_existing(df, attn_rank_features, "SCORE_attn_ensemble") is not None:
        selectors.append(
            ("attn_ensemble", "SCORE_attn_ensemble", "pure_self")
        )

    geom_rank_features = [
        "RANK_delta_cos_source_last_delta",
        "RANK_delta_cos_late_mean",
        "RANK_delta_cos_late_max",
    ]
    if mean_existing(df, geom_rank_features, "SCORE_geometry_ensemble") is not None:
        selectors.append(
            ("geometry_ensemble", "SCORE_geometry_ensemble", "pure_self")
        )

    activity_rank_features = [
        "RANK_hidden_delta_norm_recomputed",
        "RANK_attn_out_delta_norm",
        "RANK_mlp_out_delta_norm",
    ]
    if mean_existing(df, activity_rank_features, "SCORE_activity_ensemble") is not None:
        selectors.append(
            ("activity_ensemble", "SCORE_activity_ensemble", "pure_self")
        )

    all_components = [
        c for c in [
            "SCORE_attn_ensemble",
            "SCORE_geometry_ensemble",
            "SCORE_activity_ensemble",
        ]
        if available(df, c)
    ]
    if mean_existing(df, all_components, "SCORE_all_ensemble") is not None:
        selectors.append(
            ("all_ensemble", "SCORE_all_ensemble", "pure_self")
        )

    return df, selectors


# =============================================================================
# Leave-one-sample-out layer-role prior (no relation labels)
# =============================================================================

def add_loo_structural_prior(
    token_df,
    truth_exact_by_sid,
    target_name,
):
    """
    P(core | layer, broad_category) estimated from OTHER samples.
    This is relation-free but calibration-dependent.
    """
    df = token_df.copy()
    is_core = np.array(
        [
            (int(L), int(pos)) in truth_exact_by_sid.get(int(sid), set())
            for sid, L, pos in zip(df["sid"], df["source_layer"], df["position"])
        ],
        dtype=np.int8,
    )
    temp = pd.DataFrame({
        "sid": df["sid"].astype(int).to_numpy(),
        "bucket": (
            "L" + df["source_layer"].astype(int).astype(str)
            + ":" + df["broad_category"].astype(str)
        ).to_numpy(),
        "is_core": is_core,
    })

    global_stats = temp.groupby("bucket")["is_core"].agg(["sum", "count"])
    sid_stats = temp.groupby(["sid", "bucket"])["is_core"].agg(["sum", "count"])

    priors = np.empty(len(temp), dtype=np.float64)

    for i, r in enumerate(temp.itertuples(index=False)):
        gs = global_stats.loc[r.bucket]
        try:
            ss = sid_stats.loc[(r.sid, r.bucket)]
            num = float(gs["sum"] - ss["sum"])
            den = float(gs["count"] - ss["count"])
        except KeyError:
            num = float(gs["sum"])
            den = float(gs["count"])
        priors[i] = num / den if den > 0 else np.nan

    col = f"SCORE_loo_structural_prior_{target_name}"
    df[col] = priors
    return df, col


# =============================================================================
# Core localization metrics
# =============================================================================

def evaluate_one_prediction(
    sid,
    pred_rows,
    truth_exact,
    truth_pos,
    mediation_lookup,
    true_core_mass,
):
    pred_exact = set(
        (int(r.source_layer), int(r.position))
        for r in pred_rows.itertuples()
    )
    pred_pos = set(int(r.position) for r in pred_rows.itertuples())

    inter = pred_exact & truth_exact

    # +/-1-layer tolerance for the SAME position.
    tolerant_hits = 0
    for Lt, pt in truth_exact:
        ok = any(
            (pp == pt and abs(Lp - Lt) <= 1)
            for Lp, pp in pred_exact
        )
        tolerant_hits += int(ok)

    exact_recall = safe_div(len(inter), len(truth_exact))
    exact_precision = safe_div(len(inter), len(pred_exact))
    union = truth_exact | pred_exact
    exact_jaccard = safe_div(len(inter), len(union))

    pos_inter = pred_pos & truth_pos
    pos_recall = safe_div(len(pos_inter), len(truth_pos))
    pos_precision = safe_div(len(pos_inter), len(pred_pos))
    pos_union = pred_pos | truth_pos
    pos_jaccard = safe_div(len(pos_inter), len(pos_union))

    # How much oracle positive M is carried by the actually predicted rows?
    pred_mass = 0.0
    for L, p in pred_exact:
        pred_mass += max(
            float(mediation_lookup.get((int(sid), int(L), int(p)), 0.0)),
            0.0,
        )

    return {
        "pred_N": len(pred_exact),
        "true_N": len(truth_exact),
        "exact_recall": exact_recall,
        "exact_precision": exact_precision,
        "exact_jaccard": exact_jaccard,
        "position_recall": pos_recall,
        "position_precision": pos_precision,
        "position_jaccard": pos_jaccard,
        "layer_pm1_recall": safe_div(tolerant_hits, len(truth_exact)),
        "pred_positive_M": pred_mass,
        "pred_M_over_true_core_M": safe_div(pred_mass, true_core_mass),
    }


def core_mass_for_truth(sid, truth_set, mediation_lookup):
    return float(
        sum(
            max(float(mediation_lookup.get((sid, L, p), 0.0)), 0.0)
            for L, p in truth_set
        )
    )


def evaluate_selector(
    token_df,
    selector_name,
    score_col,
    selector_kind,
    target_name,
    truth,
    truth_pos,
    global_fixed_k,
    mediation_lookup,
    max_pred_k,
):
    rows = []

    for sid, g in token_df.groupby("sid", sort=True):
        sid = int(sid)
        true = truth[sid]
        true_pos = truth_pos[sid]
        true_mass = core_mass_for_truth(
            sid, true, mediation_lookup
        )

        ranked = unique_position_sorted(g, score_col)
        if not len(ranked):
            continue

        # 1. Deployable fixed global core-size prior.
        k_fixed = min(int(global_fixed_k), len(ranked))
        pred = ranked.iloc[:k_fixed]
        m = evaluate_one_prediction(
            sid, pred, true, true_pos, mediation_lookup, true_mass
        )
        m.update({
            "sid": sid,
            "relation": canon_rel(g["relation"].iloc[0]),
            "target_core": target_name,
            "selector": selector_name,
            "selector_kind": selector_kind,
            "size_mode": "fixed_global_median",
            "score_col": score_col,
        })
        rows.append(m)

        # 2. Diagnostic: exact number of core tokens is revealed, identities are not.
        k_true = min(len(true), len(ranked))
        pred = ranked.iloc[:k_true]
        m = evaluate_one_prediction(
            sid, pred, true, true_pos, mediation_lookup, true_mass
        )
        m.update({
            "sid": sid,
            "relation": canon_rel(g["relation"].iloc[0]),
            "target_core": target_name,
            "selector": selector_name,
            "selector_kind": selector_kind,
            "size_mode": "oracle_core_size_DIAGNOSTIC",
            "score_col": score_col,
        })
        rows.append(m)

        # 3. Fully self-derived K from score-gap.
        k_gap = largest_relative_gap_k(
            ranked, score_col, max_k=max_pred_k, min_k=2
        )
        k_gap = max(1, min(k_gap, len(ranked)))
        pred = ranked.iloc[:k_gap]
        m = evaluate_one_prediction(
            sid, pred, true, true_pos, mediation_lookup, true_mass
        )
        m.update({
            "sid": sid,
            "relation": canon_rel(g["relation"].iloc[0]),
            "target_core": target_name,
            "selector": selector_name,
            "selector_kind": selector_kind,
            "size_mode": "self_largest_score_gap",
            "score_col": score_col,
        })
        rows.append(m)

    return rows


def summarize_metrics(detail):
    group_cols = [
        "target_core", "selector", "selector_kind", "size_mode"
    ]
    metrics = [
        "pred_N",
        "true_N",
        "exact_recall",
        "exact_precision",
        "exact_jaccard",
        "position_recall",
        "position_precision",
        "position_jaccard",
        "layer_pm1_recall",
        "pred_M_over_true_core_M",
    ]
    return (
        detail.groupby(group_cols, as_index=False)[metrics]
        .mean()
        .sort_values(
            ["target_core", "size_mode", "position_recall"],
            ascending=[True, True, False],
        )
    )


def summarize_by_relation(detail):
    group_cols = [
        "target_core", "selector", "selector_kind", "size_mode", "relation"
    ]
    metrics = [
        "pred_N",
        "true_N",
        "exact_recall",
        "position_recall",
        "layer_pm1_recall",
        "pred_M_over_true_core_M",
    ]
    return (
        detail.groupby(group_cols, as_index=False)[metrics]
        .mean()
        .sort_values(
            ["target_core", "size_mode", "relation", "position_recall"],
            ascending=[True, True, True, False],
        )
    )


def render_summary(summary, core_sizes, selector_meta):
    lines = []
    lines.append("=" * 126)
    lines.append("RELATION-FREE SELF-PREDICTION OF SAMPLE-SPECIFIC CAUSAL CORE")
    lines.append("=" * 126)

    for c in sorted(
        [x for x in core_sizes.columns if x.endswith("_N")]
    ):
        target = c[:-2]
        x = core_sizes[c]
        lines.append(
            f"{target}: mean true N={x.mean():.2f}, "
            f"median={x.median():.1f}, p25={x.quantile(.25):.1f}, "
            f"p75={x.quantile(.75):.1f}"
        )

    lines.append("")
    lines.append(
        "Primary metric: position_recall = did the selector find the same token "
        "position, even if it chose an adjacent layer?"
    )
    lines.append(
        "Exact_recall requires the exact same (layer,position); layer_pm1_recall "
        "allows +/-1 layer at the same position."
    )

    for target in sorted(summary["target_core"].unique()):
        lines.append("")
        lines.append("-" * 126)
        lines.append(target.upper())
        lines.append("-" * 126)

        for mode in [
            "fixed_global_median",
            "oracle_core_size_DIAGNOSTIC",
            "self_largest_score_gap",
        ]:
            g = summary[
                (summary["target_core"] == target)
                & (summary["size_mode"] == mode)
            ].sort_values(
                ["position_recall", "exact_recall"],
                ascending=False,
            )
            if not len(g):
                continue
            lines.append(f"  [{mode}]")
            for r in g.head(12).itertuples():
                lines.append(
                    f"    {r.selector:<34s} "
                    f"kind={r.selector_kind:<20s} "
                    f"N={r.pred_N:5.2f} | "
                    f"exactR={r.exact_recall:.3f} "
                    f"posR={r.position_recall:.3f} "
                    f"±1R={r.layer_pm1_recall:.3f} "
                    f"J={r.position_jaccard:.3f} "
                    f"M/coreM={r.pred_M_over_true_core_M:.3f}"
                )

    lines.append("")
    lines.append("Interpretation guardrails:")
    lines.append(
        "  pure_self selectors use only the current sample's relation-free forward-pass quantities."
    )
    lines.append(
        "  calibrated_no_relation selectors use a leave-one-sample-out layer-role prior from other samples; "
        "they still do NOT use LEFT/RIGHT/ON/UNDER, but they are calibration-dependent."
    )
    lines.append(
        "  oracle_core_size_DIAGNOSTIC is NOT deployable; it isolates localization quality from core-size prediction."
    )
    lines.append(
        "  A large jump from exactR to posR/±1R means the self signal often finds the right token trajectory "
        "but chooses a neighboring layer."
    )
    lines.append(
        "  M/coreM can exceed exact overlap: that means 'wrong' predicted positions may still carry substantial "
        "positive oracle mediation, consistent with redundant causal carriers."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    core_masses = parse_floats(a.core_masses)
    for t in core_masses:
        if not (0 < t <= 1):
            raise ValueError(f"core mass must be in (0,1], got {t}")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (
        med, sel, baseline, generation, best_alpha,
        valid_sids, source_layers, target_layers
    ) = load_run(a)

    (
        truth,
        truth_pos,
        core_sizes,
        selected_m_lookup,
        selected_total_mass,
    ) = build_core_truth(sel, core_masses)

    core_sizes.to_csv(outdir / "oracle_core_size_by_sample.csv", index=False)

    print("=" * 120)
    print("SELF-PREDICT SAMPLE-SPECIFIC CORE")
    print("=" * 120)
    print("N samples:", len(valid_sids))
    print("source layers:", source_layers)
    print("target layers:", target_layers)
    print("forced oracle K:", a.k)
    print("core targets:", [fmt_mass(x) for x in core_masses])
    for t in core_masses:
        c = fmt_mass(t) + "_N"
        print(
            f"  {fmt_mass(t)}: mean={core_sizes[c].mean():.2f}, "
            f"median={core_sizes[c].median():.1f}"
        )
    print()

    token_df = collect_or_load_features(
        a, outdir, med, sel, baseline, valid_sids,
        source_layers, target_layers
    )

    # Ensure numeric mediation lookup includes ALL candidate rows, not just K36.
    mediation_lookup = {
        (int(r.sid), int(r.source_layer), int(r.position)): max(float(r.mediation), 0.0)
        for r in med.itertuples()
        if np.isfinite(float(r.mediation))
    }

    token_df, base_selectors = add_self_scores(token_df)

    # Save candidate features once.
    token_df.to_pickle(
        outdir / "all_candidate_self_features.pkl.gz",
        compression="gzip",
    )

    all_detail_rows = []
    selector_meta = []

    for t in core_masses:
        target = fmt_mass(t)
        sizes = core_sizes[target + "_N"]
        fixed_k = int(round(float(sizes.median())))

        # Pure self selectors.
        for selector_name, score_col, kind in base_selectors:
            selector_meta.append({
                "target_core": target,
                "selector": selector_name,
                "score_col": score_col,
                "selector_kind": kind,
            })
            all_detail_rows.extend(
                evaluate_selector(
                    token_df,
                    selector_name,
                    score_col,
                    kind,
                    target,
                    truth[target],
                    truth_pos[target],
                    fixed_k,
                    mediation_lookup,
                    a.max_pred_k,
                )
            )

        # LOO structural prior; no relation label.
        prior_df, prior_col = add_loo_structural_prior(
            token_df,
            truth[target],
            target,
        )

        prior_selectors = [
            (f"{target}_loo_role_prior", prior_col, "calibrated_no_relation")
        ]

        # Combine prior with the strongest self attention family.
        attn_col = None
        for candidate in [
            "SCORE_attn_ensemble",
            "RANK_last_attn_next_max_positive_delta",
            "last_attn_next_max_positive_delta",
        ]:
            if available(prior_df, candidate):
                attn_col = candidate
                break

        if attn_col is not None:
            # Convert both to within-sample ranks before mixing.
            prior_rank = f"RANK_{prior_col}"
            prior_df[prior_rank] = (
                prior_df.groupby("sid", group_keys=False)[prior_col]
                .transform(percentile_rank_high)
                .astype(float)
            )

            attn_mix_rank = f"RANK_MIXBASE_{target}"
            prior_df[attn_mix_rank] = (
                prior_df.groupby(["sid","source_layer"], group_keys=False)[attn_col]
                .transform(percentile_rank_high)
                .astype(float)
            )

            for w_attn in [0.50, 0.75]:
                w_prior = 1.0 - w_attn
                col = f"SCORE_{target}_attn{int(w_attn*100)}_prior{int(w_prior*100)}"
                prior_df[col] = (
                    w_attn * prior_df[attn_mix_rank]
                    + w_prior * prior_df[prior_rank]
                )
                prior_selectors.append(
                    (
                        f"{target}_attn{int(w_attn*100)}_roleprior{int(w_prior*100)}",
                        col,
                        "calibrated_no_relation",
                    )
                )

        for selector_name, score_col, kind in prior_selectors:
            selector_meta.append({
                "target_core": target,
                "selector": selector_name,
                "score_col": score_col,
                "selector_kind": kind,
            })
            all_detail_rows.extend(
                evaluate_selector(
                    prior_df,
                    selector_name,
                    score_col,
                    kind,
                    target,
                    truth[target],
                    truth_pos[target],
                    fixed_k,
                    mediation_lookup,
                    a.max_pred_k,
                )
            )

    detail = pd.DataFrame(all_detail_rows)
    summary = summarize_metrics(detail)
    by_relation = summarize_by_relation(detail)

    detail.to_csv(outdir / "self_core_prediction_per_sample.csv", index=False)
    summary.to_csv(outdir / "self_core_prediction_summary.csv", index=False)
    by_relation.to_csv(
        outdir / "self_core_prediction_by_relation.csv",
        index=False,
    )
    pd.DataFrame(selector_meta).drop_duplicates().to_csv(
        outdir / "selector_definitions.csv",
        index=False,
    )

    text = render_summary(
        summary,
        core_sizes,
        pd.DataFrame(selector_meta),
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    metadata = {
        "feature_collector_module": CORR_NAME,
        "run_dir": a.run_dir,
        "bundle": a.bundle,
        "k": a.k,
        "source_layers": source_layers,
        "target_layers": target_layers,
        "core_masses": core_masses,
        "N_samples": len(valid_sids),
        "best_dynamic_alpha_original_run": best_alpha,
        "important": (
            "Core labels and mediation are oracle evaluation targets only. "
            "Selectors tagged pure_self never use GT relation or writer direction. "
            "Selectors tagged calibrated_no_relation use leave-one-sample-out "
            "layer-role core-frequency priors but no relation labels."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print(text)
    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    main()
