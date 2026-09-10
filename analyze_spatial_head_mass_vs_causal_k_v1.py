#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate whether spatial Direction Heads preferentially route/read the causal K states.

This is a POST-PROCESSING script: no VLM forward pass is required.

Required inputs
---------------
A) Four-writer causal tracing output from:
   eval_four_writer_k7_competition_v1.py

   <fourway-dir>/all_four_writer_mediation.pkl.gz
   <fourway-dir>/per_sample_direction_scores.csv

B) Relation-free per-token/per-head attention cache from:
   eval_self_predict_causal_core_v2_selfcontained.py --attention-head-scan

   <feature-cache>/all_candidate_self_features.pkl.gz
   or pass the .pkl.gz file directly.

C) Direction-head probe output from:
   analyze_coco_head_object_residual_direction_probe_v1.py

   <direction-heads>/head_results.csv
   or pass the CSV directly.

Core questions
--------------
1. HEAD-LEVEL:
   Within the same layer, do heads with higher spatial-direction decoding
   accuracy place more attention mass on oracle-causal Top-K states?

2. STATE-LEVEL:
   Within the same sample and same layer, does a token-state with larger
   causal contribution M also receive larger mass from spatial heads?

3. RANK MONOTONICITY:
   Are causal ranks 1-3 read more strongly by spatial heads than ranks
   4-7, 8-14, 15-24, 25-36?

4. ENRICHMENT:
   Does oracle Top-K have more spatial-head attention mass than random
   states matched on sample AND source-layer counts?

5. K7 SELECTION:
   Each sample has K^left, K^right, K^on, K^under.
   If we score each candidate K only by spatial-head attention mass,
   can it choose the GT K without using GT at test-time?

Controls
--------
- Same-layer normalization removes trivial layer-scale confounds.
- Bottom-direction heads are reported as deterministic controls.
- Head-label permutation gives a random-head null for head-level correlation.
- Both overall spatial-head accuracy and candidate-relation-specific
  accuracy weights are tested.
- Baseline-correct and baseline-wrong samples are split.

Attention mass available from the existing cache
-------------------------------------------------
This script uses:
  last_attn_same_headXX_real
  last_attn_same_headXX_delta

Interpretation:
  last query -> candidate token attention in the SAME source layer.
This is a READ/routing-mass diagnostic.

The existing cache does NOT contain per-head pre-W_O output norms at every
candidate token. Therefore this script validates attention/input routing mass,
not per-head output-write norm. If this signal is real, output-write mass can
be extracted in a second experiment.

Typical command
---------------
python analyze_spatial_head_mass_vs_causal_k_v1.py \
  --fourway-dir output/qwen3b_four_writer_k7_N80 \
  --feature-cache output/qwen3b_coco_L20_26_self_predict_core \
  --direction-heads output/qwen3b_head_object_residual_direction \
  --ks 1,3,5,7,14,24,36 \
  --top-head-counts 1,3,5,10 \
  --random-repeats 500 \
  --output-dir output/qwen3b_spatial_head_mass_vs_causal_k_N80
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REL = ("left", "right", "on", "under")
EPS = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--direction-heads", required=True)
    p.add_argument("--ks", default="1,3,5,7,14,24,36")
    p.add_argument("--top-head-counts", default="1,3,5,10")
    p.add_argument(
        "--random-repeats",
        type=int,
        default=500,
        help="Random head-label / layer-matched state null repeats.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def canon_rel(x):
    s = str(x).strip().lower()
    mp = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under",
        "bottom": "under",
    }
    return mp.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def resolve_file(path_or_dir, filename):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    q = p / filename
    if not q.exists():
        raise FileNotFoundError(q)
    return q


def safe_mean(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def safe_std(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else np.nan


def safe_div(a, b):
    return float(a / b) if abs(float(b)) > EPS else np.nan


def spearman(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.rank().corr(y.rank(), method="pearson"))


def wilcoxon_p(x, center=0.0):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(a) == 0:
        return np.nan
    try:
        from scipy.stats import wilcoxon
        d = a - center
        if np.allclose(d, 0):
            return 1.0
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    q = np.full(len(p), np.nan)
    good = np.isfinite(p)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return q
    pv = p[idx]
    order = np.argsort(pv)
    s = pv[order]
    m = len(s)
    z = s * m / np.arange(1, m + 1)
    z = np.minimum.accumulate(z[::-1])[::-1]
    z = np.clip(z, 0, 1)
    q[idx[order]] = z
    return q


def percentile_rank_within_group(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.rank(method="average", pct=True)


# =============================================================================
# Load data
# =============================================================================

def load_inputs(a):
    four = Path(a.fourway_dir)
    med_path = four / "all_four_writer_mediation.pkl.gz"
    score_path = four / "per_sample_direction_scores.csv"
    if not med_path.exists():
        raise FileNotFoundError(med_path)
    if not score_path.exists():
        raise FileNotFoundError(score_path)

    feat_path = resolve_file(
        a.feature_cache, "all_candidate_self_features.pkl.gz"
    )
    head_path = resolve_file(a.direction_heads, "head_results.csv")

    med = pd.read_pickle(med_path, compression="gzip")
    sample = pd.read_csv(score_path)
    feat = pd.read_pickle(feat_path, compression="gzip")
    heads = pd.read_csv(head_path)

    for d in (med, feat):
        d["sid"] = pd.to_numeric(d["sid"], errors="raise").astype(int)
        d["source_layer"] = pd.to_numeric(
            d["source_layer"], errors="raise"
        ).astype(int)
        d["position"] = pd.to_numeric(d["position"], errors="raise").astype(int)

    sample["sid"] = pd.to_numeric(sample["sid"], errors="raise").astype(int)

    med["candidate_relation"] = med["candidate_relation"].map(canon_rel)
    med["gt"] = med["gt"].map(canon_rel)
    med["baseline_prediction"] = med["baseline_prediction"].map(canon_rel)
    med["baseline_correct"] = med["baseline_correct"].map(boolify)
    med["is_oracle_writer"] = med["is_oracle_writer"].map(boolify)

    sample["gt"] = sample["gt"].map(canon_rel)
    sample["baseline_prediction"] = sample["baseline_prediction"].map(canon_rel)
    sample["baseline_correct"] = sample["baseline_correct"].map(boolify)

    heads["layer"] = pd.to_numeric(heads["layer"], errors="raise").astype(int)
    heads["head"] = pd.to_numeric(heads["head"], errors="raise").astype(int)

    needed = ["residual_accuracy_mean"]
    for c in needed:
        if c not in heads.columns:
            raise RuntimeError(
                f"{head_path} missing required column {c!r}. "
                "Expected output from analyze_coco_head_object_residual_direction_probe_v1.py"
            )

    head_cols = sorted(
        [
            c for c in feat.columns
            if re.fullmatch(r"last_attn_same_head\d+_(real|delta)", str(c))
        ]
    )
    if not head_cols:
        raise RuntimeError(
            "No per-head attention columns found in feature cache.\n"
            "Re-run eval_self_predict_causal_core_v2_selfcontained.py with "
            "--attention-head-scan."
        )

    # Infer head ids actually present in cache.
    cache_heads = sorted(
        {
            int(re.search(r"head(\d+)", c).group(1))
            for c in head_cols
        }
    )

    # Restrict to samples and layers shared by all sources.
    common_sids = sorted(
        set(med.sid.unique())
        & set(feat.sid.unique())
        & set(sample.sid.unique())
    )
    med = med[med.sid.isin(common_sids)].copy()
    feat = feat[feat.sid.isin(common_sids)].copy()

    layers = sorted(
        set(med.source_layer.unique())
        & set(feat.source_layer.unique())
        & set(heads.layer.unique())
    )
    med = med[med.source_layer.isin(layers)].copy()
    feat = feat[feat.source_layer.isin(layers)].copy()
    heads = heads[
        heads.layer.isin(layers) & heads["head"].isin(cache_heads)
    ].copy()

    return med, sample, feat, heads, head_cols, layers, cache_heads, {
        "mediation": str(med_path),
        "sample_scores": str(score_path),
        "features": str(feat_path),
        "direction_heads": str(head_path),
    }


# =============================================================================
# Direction-head weights and state-level spatial mass
# =============================================================================

def relation_accuracy_col(rel):
    return {
        "left": "residual_left_accuracy",
        "right": "residual_right_accuracy",
        "on": "residual_on_accuracy",
        "under": "residual_under_accuracy",
    }[rel]


def make_head_weights(heads):
    """
    Return table with:
      overall_w       = positive excess over 4-way chance (0.25)
      rel_<r>_w       = positive per-relation accuracy excess over chance
      layer-relative direction rank
    """
    h = heads.copy()

    h["overall_w"] = (
        pd.to_numeric(h["residual_accuracy_mean"], errors="coerce") - 0.25
    ).clip(lower=0.0)

    h["overall_rank_pct"] = (
        h.groupby("layer")["residual_accuracy_mean"]
        .rank(method="average", pct=True)
    )

    for r in REL:
        c = relation_accuracy_col(r)
        if c in h.columns:
            h[f"{r}_w"] = (
                pd.to_numeric(h[c], errors="coerce") - 0.25
            ).clip(lower=0.0)
            h[f"{r}_rank_pct"] = (
                h.groupby("layer")[c].rank(method="average", pct=True)
            )
        else:
            h[f"{r}_w"] = h["overall_w"]
            h[f"{r}_rank_pct"] = h["overall_rank_pct"]

    return h


def feature_head_col(h, kind):
    return f"last_attn_same_head{int(h):02d}_{kind}"


def weighted_head_mass(row, layer_heads, rel, kind, scheme):
    vals = []
    ws = []
    for hr in layer_heads.itertuples():
        c = feature_head_col(hr.head, kind)
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue

        if scheme == "overall":
            w = float(hr.overall_w)
        elif scheme == "relation_specific":
            w = float(getattr(hr, f"{rel}_w"))
        else:
            raise ValueError(scheme)

        vals.append(float(v))
        ws.append(max(w, 0.0))

    if not vals:
        return np.nan
    vals = np.asarray(vals, float)
    ws = np.asarray(ws, float)
    if ws.sum() <= EPS:
        return float(vals.mean())
    return float(np.average(vals, weights=ws))


def top_head_mass(row, layer_heads, rel, kind, n, scheme, top=True):
    if scheme == "overall":
        score_col = "residual_accuracy_mean"
    else:
        score_col = relation_accuracy_col(rel)
        if score_col not in layer_heads.columns:
            score_col = "residual_accuracy_mean"

    hh = layer_heads.sort_values(score_col, ascending=not top).head(n)
    vals = []
    for hr in hh.itertuples():
        c = feature_head_col(hr.head, kind)
        if c in row.index and pd.notna(row[c]):
            vals.append(float(row[c]))
    return float(np.mean(vals)) if vals else np.nan


def build_state_table(med, feat, heads, top_counts):
    # Feature table is relation-free, so merge once by exact state.
    key = ["sid", "source_layer", "position"]
    head_feature_cols = [
        c for c in feat.columns
        if re.fullmatch(r"last_attn_same_head\d+_(real|delta)", str(c))
    ]
    f = feat[key + head_feature_cols].drop_duplicates(key)
    x = med.merge(f, on=key, how="left", validate="many_to_one")

    by_layer = {
        int(L): g.copy()
        for L, g in heads.groupby("layer")
    }

    records = []
    for r in x.itertuples(index=False):
        d = r._asdict()
        L = int(d["source_layer"])
        rel = str(d["candidate_relation"])
        hs = by_layer.get(L)
        if hs is None or not len(hs):
            continue
        row = pd.Series(d)

        out = {
            k: d[k] for k in [
                "sid", "gt", "baseline_prediction", "baseline_correct",
                "candidate_relation", "is_oracle_writer",
                "source_layer", "position", "token", "token_clean",
                "category", "broad_category", "mediation",
            ] if k in d
        }

        for kind in ("real", "delta"):
            # Weighted mass over all spatial-informative heads.
            for scheme in ("overall", "relation_specific"):
                out[f"spatial_{scheme}_{kind}_weighted"] = weighted_head_mass(
                    row, hs, rel, kind, scheme
                )

            # Top/bottom direction-head controls.
            for n in top_counts:
                for scheme in ("overall", "relation_specific"):
                    out[f"spatial_{scheme}_{kind}_top{n}"] = top_head_mass(
                        row, hs, rel, kind, n, scheme, top=True
                    )
                    out[f"control_{scheme}_{kind}_bottom{n}"] = top_head_mass(
                        row, hs, rel, kind, n, scheme, top=False
                    )

        records.append(out)

    s = pd.DataFrame(records)

    # Positive delta version: only increased attention from Real over Gray.
    for c in list(s.columns):
        if "_delta_" in c and (
            c.startswith("spatial_") or c.startswith("control_")
        ):
            s[c + "_pos"] = pd.to_numeric(s[c], errors="coerce").clip(lower=0.0)

    # Within sample x candidate relation x layer ranks remove layer-scale.
    mass_cols = [
        c for c in s.columns
        if c.startswith("spatial_") or c.startswith("control_")
    ]
    s["M_rank_within_layer"] = (
        s.groupby(["sid", "candidate_relation", "source_layer"])["mediation"]
        .transform(percentile_rank_within_group)
    )
    for c in mass_cols:
        s[c + "__rank_within_layer"] = (
            s.groupby(["sid", "candidate_relation", "source_layer"])[c]
            .transform(percentile_rank_within_group)
        )

    return s, mass_cols


# =============================================================================
# Exact global-unique causal ranking
# =============================================================================

def global_unique_rank(g, max_k=36):
    """
    Keep one layer per token position: the (L,p) with largest positive M.
    Then rank states by M descending.
    """
    q = g.copy()
    q["mediation"] = pd.to_numeric(q["mediation"], errors="coerce")
    q = q[np.isfinite(q["mediation"]) & (q["mediation"] > 0)].copy()
    if not len(q):
        return q
    idx = q.groupby("position")["mediation"].idxmax()
    q = q.loc[idx].sort_values("mediation", ascending=False).head(max_k).copy()
    q["causal_rank"] = np.arange(1, len(q) + 1)
    return q


def build_ranked_states(state, max_k):
    chunks = []
    for (_, _), g in state.groupby(["sid", "candidate_relation"]):
        q = global_unique_rank(g, max_k=max_k)
        if len(q):
            chunks.append(q)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def rank_bin(rank):
    r = int(rank)
    if r <= 3:
        return "01-03"
    if r <= 7:
        return "04-07"
    if r <= 14:
        return "08-14"
    if r <= 24:
        return "15-24"
    return "25-36"


# =============================================================================
# 1) Head-level: direction accuracy vs attention on causal states
# =============================================================================

def head_level_topk_mass_vs_accuracy(
    ranked, feat, heads, ks, random_repeats, seed
):
    """
    For ORACLE relation only.

    Within each sample/layer:
      - take oracle causal TopK states that fall in that layer
      - compute mean attention mass to those states for every head
    Then pool sample-level head masses by layer/head and correlate across heads
    with independent direction decoding accuracy.

    We compare causal TopK mass to matched random positions from the same
    sample/layer.
    """
    oracle = ranked[ranked["candidate_relation"] == ranked["gt"]].copy()
    feat_key = feat.set_index(["sid", "source_layer", "position"])
    rng = np.random.default_rng(seed)
    rows = []

    layers = sorted(set(oracle.source_layer.unique()) & set(heads.layer.unique()))

    for K in ks:
        top = oracle[oracle["causal_rank"] <= K].copy()

        for kind in ("real", "delta"):
            # Per sample-layer/head topK masses.
            obs = defaultdict(list)
            rnd = defaultdict(list)

            for (sid, L), g in top.groupby(["sid", "source_layer"]):
                positions = sorted(set(g.position.astype(int)))
                if not positions:
                    continue

                # Candidate universe in this sample/layer from feature cache.
                try:
                    fsl = feat[
                        (feat.sid == int(sid))
                        & (feat.source_layer == int(L))
                    ]
                except Exception:
                    continue

                universe = sorted(set(fsl.position.astype(int)))
                nonselected = [p for p in universe if p not in set(positions)]
                if not nonselected:
                    continue

                hrows = heads[heads.layer == int(L)]
                for hr in hrows.itertuples():
                    col = feature_head_col(hr.head, kind)
                    if col not in feat.columns:
                        continue

                    vals = []
                    for p in positions:
                        hit = fsl[fsl.position == p]
                        if len(hit) and pd.notna(hit[col].iloc[0]):
                            vals.append(float(hit[col].iloc[0]))
                    if not vals:
                        continue
                    obs[(int(L), int(hr.head))].append(float(np.mean(vals)))

                    # One matched random draw per sample/layer here; repeat
                    # aggregation later over many label/head permutations too.
                    n = min(len(positions), len(nonselected))
                    rp = rng.choice(nonselected, size=n, replace=False)
                    rv = []
                    for p in rp:
                        hit = fsl[fsl.position == int(p)]
                        if len(hit) and pd.notna(hit[col].iloc[0]):
                            rv.append(float(hit[col].iloc[0]))
                    if rv:
                        rnd[(int(L), int(hr.head))].append(float(np.mean(rv)))

            for L in layers:
                hrows = heads[heads.layer == L].copy()
                hdata = []
                for hr in hrows.itertuples():
                    key = (int(L), int(hr.head))
                    if key not in obs or not obs[key]:
                        continue
                    hdata.append({
                        "head": int(hr.head),
                        "direction_acc": float(hr.residual_accuracy_mean),
                        "topk_mass": safe_mean(obs[key]),
                        "random_mass": safe_mean(rnd.get(key, [])),
                    })
                hd = pd.DataFrame(hdata)
                if len(hd) < 4:
                    continue

                for measure in ("topk_mass", "random_mass"):
                    rho = spearman(hd["direction_acc"], hd[measure])

                    # Permute head accuracy labels to obtain null.
                    null = []
                    acc = hd["direction_acc"].to_numpy(float)
                    mass = hd[measure].to_numpy(float)
                    for _ in range(random_repeats):
                        null.append(spearman(rng.permutation(acc), mass))
                    null = np.asarray(null, float)
                    null = null[np.isfinite(null)]
                    p = (
                        (1 + np.sum(np.abs(null) >= abs(rho)))
                        / (1 + len(null))
                        if np.isfinite(rho) and len(null)
                        else np.nan
                    )

                    rows.append({
                        "K": K,
                        "kind": kind,
                        "layer": int(L),
                        "measure": measure,
                        "n_heads": len(hd),
                        "spearman_directionAcc_vs_mass": rho,
                        "head_label_perm_p": p,
                        "mean_mass": safe_mean(hd[measure]),
                    })

    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_q"] = bh_fdr(out["head_label_perm_p"].to_numpy(float))
    return out


# =============================================================================
# 2) State-level within-layer correlation
# =============================================================================

def state_level_correlations(state, mass_cols):
    """
    Per sample x candidate x layer Spearman(M, spatial-head mass), then summarize.

    Most important subsets:
      oracle_all
      oracle_correct
      oracle_wrong
      all_candidates
    """
    rows = []
    subsets = {
        "all_candidates": state,
        "oracle_all": state[state.candidate_relation == state.gt],
        "oracle_correct": state[
            (state.candidate_relation == state.gt) & state.baseline_correct
        ],
        "oracle_wrong": state[
            (state.candidate_relation == state.gt) & (~state.baseline_correct)
        ],
    }

    for subset_name, df in subsets.items():
        for mass in mass_cols:
            corr_vals = []
            group_rows = []
            for (sid, cand, L), g in df.groupby(
                ["sid", "candidate_relation", "source_layer"]
            ):
                # Only positive M for the causal-support ranking analysis.
                q = g[pd.to_numeric(g.mediation, errors="coerce") > 0]
                rho = spearman(q["mediation"], q[mass])
                if np.isfinite(rho):
                    corr_vals.append(rho)
                    group_rows.append({
                        "subset": subset_name,
                        "mass_metric": mass,
                        "sid": int(sid),
                        "candidate_relation": cand,
                        "layer": int(L),
                        "rho": rho,
                        "n_states": len(q),
                    })
            if corr_vals:
                rows.append({
                    "subset": subset_name,
                    "mass_metric": mass,
                    "n_sample_layers": len(corr_vals),
                    "mean_rho": float(np.mean(corr_vals)),
                    "median_rho": float(np.median(corr_vals)),
                    "positive_fraction": float(np.mean(np.asarray(corr_vals) > 0)),
                    "wilcoxon_vs_zero_p": wilcoxon_p(corr_vals, 0.0),
                })

    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_q"] = bh_fdr(out["wilcoxon_vs_zero_p"].to_numpy(float))
    return out


# =============================================================================
# 3) Causal-rank monotonicity
# =============================================================================

def rank_monotonicity(ranked, mass_cols):
    oracle = ranked[ranked.candidate_relation == ranked.gt].copy()
    oracle["rank_bin"] = oracle["causal_rank"].map(rank_bin)

    bin_rows = []
    trend_rows = []

    for split_name, df in [
        ("all", oracle),
        ("baseline_correct", oracle[oracle.baseline_correct]),
        ("baseline_wrong", oracle[~oracle.baseline_correct]),
    ]:
        for mass in mass_cols:
            # Per-sample bin means first, to avoid samples with more states
            # dominating the aggregate.
            per = (
                df.groupby(["sid", "rank_bin"])[mass]
                .mean()
                .reset_index()
            )
            for b in ["01-03", "04-07", "08-14", "15-24", "25-36"]:
                z = per[per.rank_bin == b]
                if len(z):
                    bin_rows.append({
                        "split": split_name,
                        "mass_metric": mass,
                        "rank_bin": b,
                        "N_samples": z.sid.nunique(),
                        "mean_mass": safe_mean(z[mass]),
                        "std_mass": safe_std(z[mass]),
                    })

            # Sample-level Spearman: rank number ↑ should mass ↓, so expected rho < 0.
            rhos = []
            for sid, g in df.groupby("sid"):
                rho = spearman(g["causal_rank"], g[mass])
                if np.isfinite(rho):
                    rhos.append(rho)
            if rhos:
                trend_rows.append({
                    "split": split_name,
                    "mass_metric": mass,
                    "N_samples": len(rhos),
                    "mean_spearman_rank_vs_mass": float(np.mean(rhos)),
                    "median_spearman_rank_vs_mass": float(np.median(rhos)),
                    "fraction_negative": float(np.mean(np.asarray(rhos) < 0)),
                    "wilcoxon_vs_zero_p": wilcoxon_p(rhos),
                })

    trend = pd.DataFrame(trend_rows)
    if len(trend):
        trend["fdr_q"] = bh_fdr(trend["wilcoxon_vs_zero_p"].to_numpy(float))
    return pd.DataFrame(bin_rows), trend


# =============================================================================
# 4) Layer-matched TopK enrichment
# =============================================================================

def layer_matched_enrichment(
    state, ranked, mass_cols, ks, repeats, seed
):
    """
    Oracle candidate only. For each sample and K, take oracle TopK.
    Compare mean mass to random states matched exactly by source-layer counts.
    """
    rng = np.random.default_rng(seed)
    oracle_state = state[state.candidate_relation == state.gt].copy()
    oracle_ranked = ranked[ranked.candidate_relation == ranked.gt].copy()

    details = []

    for K in ks:
        top = oracle_ranked[oracle_ranked.causal_rank <= K]
        for sid, tg in top.groupby("sid"):
            pool = oracle_state[oracle_state.sid == sid]
            if not len(pool):
                continue

            selected_keys = set(
                zip(tg.source_layer.astype(int), tg.position.astype(int))
            )
            layer_counts = tg.source_layer.value_counts().to_dict()

            for mass in mass_cols:
                obs = safe_mean(tg[mass])
                null = []

                for _ in range(repeats):
                    vals = []
                    possible = True
                    for L, n in layer_counts.items():
                        pg = pool[
                            (pool.source_layer == int(L))
                            & (~pool.apply(
                                lambda r: (int(r.source_layer), int(r.position))
                                in selected_keys,
                                axis=1
                            ))
                        ]
                        pg = pg[pd.to_numeric(pg[mass], errors="coerce").notna()]
                        if len(pg) < int(n):
                            # Allow selected states back into universe if the
                            # layer is tiny; matching layer is more important.
                            pg = pool[
                                (pool.source_layer == int(L))
                                & pd.to_numeric(pool[mass], errors="coerce").notna()
                            ]
                        if len(pg) < int(n):
                            possible = False
                            break
                        take = rng.choice(
                            pg.index.to_numpy(), size=int(n), replace=False
                        )
                        vals.extend(pd.to_numeric(pg.loc[take, mass], errors="coerce"))
                    if possible and vals:
                        null.append(float(np.mean(vals)))

                if null:
                    details.append({
                        "sid": int(sid),
                        "gt": tg.gt.iloc[0],
                        "baseline_correct": bool(tg.baseline_correct.iloc[0]),
                        "K": K,
                        "mass_metric": mass,
                        "topk_mass": obs,
                        "matched_random_mean": float(np.mean(null)),
                        "enrichment": obs - float(np.mean(null)),
                        "empirical_p_greater": (
                            1 + np.sum(np.asarray(null) >= obs)
                        ) / (1 + len(null)),
                    })

    det = pd.DataFrame(details)
    if not len(det):
        return det, pd.DataFrame()

    summary = (
        det.groupby(["K", "mass_metric", "baseline_correct"])
        .agg(
            N=("sid", "nunique"),
            mean_topk_mass=("topk_mass", "mean"),
            mean_random_mass=("matched_random_mean", "mean"),
            mean_enrichment=("enrichment", "mean"),
            median_enrichment=("enrichment", "median"),
            positive_enrichment_fraction=("enrichment", lambda x: np.mean(np.asarray(x) > 0)),
        )
        .reset_index()
    )

    ps = []
    for r in summary.itertuples():
        g = det[
            (det.K == r.K)
            & (det.mass_metric == r.mass_metric)
            & (det.baseline_correct == r.baseline_correct)
        ]
        ps.append(wilcoxon_p(g["enrichment"], 0.0))
    summary["wilcoxon_enrichment_p"] = ps
    summary["fdr_q"] = bh_fdr(summary["wilcoxon_enrichment_p"].to_numpy(float))
    return det, summary


# =============================================================================
# 5) Four-way candidate selection by spatial-head mass
# =============================================================================

def fourway_selector(ranked, mass_cols, ks):
    rows = []
    detail = []

    for K in ks:
        q = ranked[ranked.causal_rank <= K]
        for mass in mass_cols:
            for score_type in ("mean", "rank_weighted"):
                sample_rows = []
                for sid, sg in q.groupby("sid"):
                    cand_scores = {}
                    for cand, cg in sg.groupby("candidate_relation"):
                        vals = pd.to_numeric(cg[mass], errors="coerce").to_numpy(float)
                        ranks = pd.to_numeric(
                            cg["causal_rank"], errors="coerce"
                        ).to_numpy(float)
                        good = np.isfinite(vals) & np.isfinite(ranks)
                        vals, ranks = vals[good], ranks[good]
                        if not len(vals):
                            continue

                        if score_type == "mean":
                            sc = float(vals.mean())
                        else:
                            w = 1.0 / ranks
                            sc = float(np.average(vals, weights=w))
                        cand_scores[str(cand)] = sc

                    if len(cand_scores) < 4:
                        continue

                    pred = max(cand_scores, key=cand_scores.get)
                    gt = str(sg.gt.iloc[0])
                    bp = str(sg.baseline_prediction.iloc[0])
                    bc = bool(sg.baseline_correct.iloc[0])
                    ordered = sorted(cand_scores.values(), reverse=True)

                    z = {
                        "sid": int(sid),
                        "K": K,
                        "mass_metric": mass,
                        "score_type": score_type,
                        "gt": gt,
                        "baseline_prediction": bp,
                        "baseline_correct": bc,
                        "pred": pred,
                        "pred_is_gt": pred == gt,
                        "pred_matches_baseline": pred == bp,
                        "winner_margin": ordered[0] - ordered[1],
                    }
                    for r in REL:
                        z[f"score_{r}"] = cand_scores.get(r, np.nan)
                    sample_rows.append(z)
                    detail.append(z)

                if not sample_rows:
                    continue
                d = pd.DataFrame(sample_rows)
                cor = d.baseline_correct
                wr = ~cor
                rows.append({
                    "K": K,
                    "mass_metric": mass,
                    "score_type": score_type,
                    "N": len(d),
                    "GTacc_all": float(d.pred_is_gt.mean()),
                    "GTacc_correct": (
                        float(d.loc[cor, "pred_is_gt"].mean())
                        if cor.any() else np.nan
                    ),
                    "GTacc_wrong": (
                        float(d.loc[wr, "pred_is_gt"].mean())
                        if wr.any() else np.nan
                    ),
                    "matchBaselineWrong": (
                        float(d.loc[wr, "pred_matches_baseline"].mean())
                        if wr.any() else np.nan
                    ),
                    "margin_correct": safe_mean(d.loc[cor, "winner_margin"]),
                    "margin_wrong": safe_mean(d.loc[wr, "winner_margin"]),
                })

    return (
        pd.DataFrame(rows).sort_values(
            ["GTacc_wrong", "GTacc_all"], ascending=False
        ),
        pd.DataFrame(detail),
    )


# =============================================================================
# 6) Spatial heads vs bottom heads on the same causal states
# =============================================================================

def spatial_vs_bottom(state, ranked, top_counts, ks):
    """
    Direct deterministic control:
      mean attention of Top-N direction heads on causal Top-K
      minus Bottom-N direction heads on the same exact states.
    """
    oracle = ranked[ranked.candidate_relation == ranked.gt].copy()
    rows = []

    for K in ks:
        q = oracle[oracle.causal_rank <= K]
        for n in top_counts:
            pairs = [
                (
                    f"spatial_overall_real_top{n}",
                    f"control_overall_real_bottom{n}",
                    "overall", "real"
                ),
                (
                    f"spatial_relation_specific_real_top{n}",
                    f"control_relation_specific_real_bottom{n}",
                    "relation_specific", "real"
                ),
                (
                    f"spatial_overall_delta_top{n}_pos",
                    f"control_overall_delta_bottom{n}_pos",
                    "overall", "delta_pos"
                ),
                (
                    f"spatial_relation_specific_delta_top{n}_pos",
                    f"control_relation_specific_delta_bottom{n}_pos",
                    "relation_specific", "delta_pos"
                ),
            ]

            for spatial_col, ctrl_col, scheme, kind in pairs:
                if spatial_col not in q.columns or ctrl_col not in q.columns:
                    continue

                per = (
                    q.groupby(["sid", "baseline_correct"])[
                        [spatial_col, ctrl_col]
                    ]
                    .mean()
                    .reset_index()
                )
                per["diff"] = per[spatial_col] - per[ctrl_col]

                for bc, g in per.groupby("baseline_correct"):
                    rows.append({
                        "K": K,
                        "top_head_count": n,
                        "weight_scheme": scheme,
                        "kind": kind,
                        "baseline_group": "correct" if bc else "wrong",
                        "N": g.sid.nunique(),
                        "spatial_head_mass": safe_mean(g[spatial_col]),
                        "bottom_head_mass": safe_mean(g[ctrl_col]),
                        "spatial_minus_bottom": safe_mean(g["diff"]),
                        "positive_fraction": float(np.mean(g["diff"] > 0)),
                        "paired_wilcoxon_p": wilcoxon_p(g["diff"], 0.0),
                    })

    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_q"] = bh_fdr(out["paired_wilcoxon_p"].to_numpy(float))
    return out


# =============================================================================
# Reporting
# =============================================================================

def pick_metrics(mass_cols):
    """
    Keep a focused set for expensive analyses.
    """
    wanted = []
    priority = [
        "spatial_overall_real_weighted",
        "spatial_relation_specific_real_weighted",
        "spatial_overall_delta_weighted_pos",
        "spatial_relation_specific_delta_weighted_pos",
        "spatial_overall_real_top5",
        "spatial_relation_specific_real_top5",
        "spatial_overall_delta_top5_pos",
        "spatial_relation_specific_delta_top5_pos",
        "control_overall_real_bottom5",
        "control_overall_delta_bottom5_pos",
    ]
    for c in priority:
        if c in mass_cols:
            wanted.append(c)

    # If Top5 doesn't exist because user chose different N, add first few
    # available spatial metrics.
    for c in mass_cols:
        if c.startswith("spatial_") and c not in wanted:
            wanted.append(c)
        if len(wanted) >= 14:
            break
    return wanted


def render_summary(
    a, state, ranked, focused,
    headcorr, statecorr, ranktrend, rankbins,
    enrich, selector, controls
):
    lines = []
    lines.append("=" * 172)
    lines.append("SPATIAL DIRECTION-HEAD MASS vs CAUSAL K — VALIDATION")
    lines.append("=" * 172)
    lines.append(
        f"N={state.sid.nunique()} | source layers="
        f"{sorted(state.source_layer.unique().tolist())} | "
        f"max causal rank={int(ranked.causal_rank.max()) if len(ranked) else 0}"
    )
    lines.append(
        "Spatial heads are independently defined by ref-sub Real-NoImage direction decoding. "
        "Mass is same-layer last-query -> token attention from the existing head-scan cache."
    )
    lines.append("")

    lines.append("1) HEAD LEVEL: DO HIGH-DIRECTION-ACC HEADS PUT MORE MASS ON CAUSAL Top-K?")
    lines.append("-" * 172)
    if len(headcorr):
        z = headcorr[
            (headcorr.measure == "topk_mass")
            & (headcorr.K.isin([7, 14, 36]))
        ].sort_values(
            ["K", "spearman_directionAcc_vs_mass"],
            ascending=[True, False]
        )
        for r in z.head(30).itertuples():
            lines.append(
                f"K={int(r.K):2d} {r.kind:<5s} L{int(r.layer):02d} | "
                f"rho(head direction acc, causal mass)={r.spearman_directionAcc_vs_mass:+.3f} "
                f"perm-p={r.head_label_perm_p:.4g} q={r.fdr_q:.4g}"
            )
    else:
        lines.append("No head-level results.")
    lines.append("")

    lines.append("2) STATE LEVEL: WITHIN SAME SAMPLE + SAME LAYER, DOES HIGHER M MEAN HIGHER SPATIAL-HEAD MASS?")
    lines.append("-" * 172)
    if len(statecorr):
        z = statecorr[
            (statecorr.subset.isin(["oracle_all", "oracle_correct", "oracle_wrong"]))
            & (statecorr.mass_metric.isin(focused))
        ].sort_values(
            ["subset", "mean_rho"], ascending=[True, False]
        )
        for r in z.itertuples():
            lines.append(
                f"{r.subset:<15s} {r.mass_metric:<55s} | "
                f"mean rho={r.mean_rho:+.3f} median={r.median_rho:+.3f} "
                f"P(rho>0)={r.positive_fraction:.3f} "
                f"p={r.wilcoxon_vs_zero_p:.4g} q={r.fdr_q:.4g}"
            )
    lines.append("")

    lines.append("3) CAUSAL-RANK MONOTONICITY: RANK 1 SHOULD HAVE MORE SPATIAL MASS THAN RANK 36")
    lines.append("-" * 172)
    if len(ranktrend):
        z = ranktrend[
            (ranktrend.mass_metric.isin(focused))
        ].sort_values(
            ["split", "mean_spearman_rank_vs_mass"]
        )
        for r in z.itertuples():
            lines.append(
                f"{r.split:<17s} {r.mass_metric:<55s} | "
                f"rho(rank,mass)={r.mean_spearman_rank_vs_mass:+.3f} "
                f"P(negative)={r.fraction_negative:.3f} "
                f"p={r.wilcoxon_vs_zero_p:.4g} q={r.fdr_q:.4g}"
            )
    lines.append("")
    if len(rankbins):
        for metric in [m for m in focused[:4] if m in set(rankbins.mass_metric)]:
            z = rankbins[
                (rankbins.split == "all") & (rankbins.mass_metric == metric)
            ]
            if len(z):
                bits = [
                    f"{r.rank_bin}:{r.mean_mass:.4f}"
                    for r in z.itertuples()
                ]
                lines.append(f"  {metric}: " + " | ".join(bits))
    lines.append("")

    lines.append("4) ORACLE Top-K ENRICHMENT vs SAMPLE+LAYER-MATCHED RANDOM STATES")
    lines.append("-" * 172)
    if len(enrich):
        z = enrich[
            enrich.mass_metric.isin(focused)
            & enrich.K.isin([7, 14, 36])
        ].sort_values(
            ["K", "baseline_correct", "mean_enrichment"],
            ascending=[True, False, False]
        )
        for r in z.itertuples():
            lines.append(
                f"K={int(r.K):2d} "
                f"{'correct' if r.baseline_correct else 'wrong':<7s} "
                f"{r.mass_metric:<52s} | "
                f"TopK={r.mean_topk_mass:.4f} rnd={r.mean_random_mass:.4f} "
                f"enrich={r.mean_enrichment:+.4f} "
                f"P(>0)={r.positive_enrichment_fraction:.3f} "
                f"p={r.wilcoxon_enrichment_p:.4g} q={r.fdr_q:.4g}"
            )
    lines.append("")

    lines.append("5) CONTROL: TOP DIRECTION HEADS vs BOTTOM DIRECTION HEADS ON SAME CAUSAL STATES")
    lines.append("-" * 172)
    if len(controls):
        z = controls[
            controls.K.isin([7, 14, 36])
        ].sort_values(
            ["K", "spatial_minus_bottom"],
            ascending=[True, False]
        )
        for r in z.head(40).itertuples():
            lines.append(
                f"K={int(r.K):2d} {r.baseline_group:<7s} "
                f"Top{int(r.top_head_count):02d} {r.weight_scheme:<17s} {r.kind:<9s} | "
                f"spatial={r.spatial_head_mass:.5f} bottom={r.bottom_head_mass:.5f} "
                f"Δ={r.spatial_minus_bottom:+.5f} "
                f"P(Δ>0)={r.positive_fraction:.3f} "
                f"p={r.paired_wilcoxon_p:.4g} q={r.fdr_q:.4g}"
            )
    lines.append("")

    lines.append("6) CAN SPATIAL-HEAD MASS CHOOSE ONE OF THE FOUR CANDIDATE K SETS?")
    lines.append("-" * 172)
    if len(selector):
        z = selector[
            selector.mass_metric.isin(focused)
        ].sort_values(
            ["GTacc_wrong", "GTacc_all"], ascending=False
        )
        for r in z.head(35).itertuples():
            lines.append(
                f"K={int(r.K):2d} {r.score_type:<13s} {r.mass_metric:<50s} | "
                f"GTacc all={r.GTacc_all:.3f} "
                f"correct={r.GTacc_correct:.3f} wrong={r.GTacc_wrong:.3f} | "
                f"matchBaselineWrong={r.matchBaselineWrong:.3f}"
            )
    lines.append("")

    lines.append("DECISION RULE")
    lines.append("-" * 172)
    lines.append(
        "The correlation hypothesis is convincing only if it survives SAME-LAYER analysis "
        "and spatial heads outperform bottom/random-head controls. A raw pooled correlation alone is not sufficient."
    )
    lines.append(
        "For K7 selection, the decisive number is GTacc_wrong: on baseline-wrong samples, "
        "does spatial-head mass select the true candidate K more often than generic attention did?"
    )
    lines.append(
        "If correlation/enrichment is strong but four-way selection is weak, spatial-head routing "
        "is genuinely related to causal states but is not relation-specific enough to identify which K7 is correct."
    )
    lines.append(
        "If relation-specific direction-head weights improve GTacc_wrong substantially, then the "
        "same functional spatial heads that encode ref-sub direction provide a plausible endogenous selector."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    ks = ints(a.ks)
    top_counts = ints(a.top_head_counts)
    if not ks or not top_counts:
        raise ValueError("--ks and --top-head-counts must be non-empty")

    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    med, sample, feat, heads0, head_cols, layers, cache_heads, paths = load_inputs(a)
    heads = make_head_weights(heads0)

    print("=" * 160)
    print("SPATIAL DIRECTION-HEAD MASS vs CAUSAL K")
    print("=" * 160)
    print(f"N={med.sid.nunique()} | layers={layers} | heads/cache={len(cache_heads)}")
    print(f"Ks={ks} | top-head controls={top_counts}")
    print("Building state-level spatial-head mass...", flush=True)

    state, mass_cols = build_state_table(med, feat, heads, top_counts)
    focused = pick_metrics(mass_cols)
    max_k = max(ks)
    ranked = build_ranked_states(state, max_k=max_k)

    state.to_pickle(
        outdir / "all_candidate_states_with_spatial_head_mass.pkl.gz",
        compression="gzip",
    )
    ranked.to_csv(outdir / "causal_global_unique_ranked_states.csv", index=False)

    print("1/6 head-level direction accuracy vs causal mass...", flush=True)
    headcorr = head_level_topk_mass_vs_accuracy(
        ranked, feat, heads, ks, a.random_repeats, a.seed
    )
    headcorr.to_csv(
        outdir / "head_direction_accuracy_vs_causal_topk_mass.csv",
        index=False,
    )

    print("2/6 within-layer state-level correlations...", flush=True)
    statecorr = state_level_correlations(state, focused)
    statecorr.to_csv(
        outdir / "within_layer_M_vs_spatial_head_mass.csv",
        index=False,
    )

    print("3/6 causal-rank monotonicity...", flush=True)
    rankbins, ranktrend = rank_monotonicity(ranked, focused)
    rankbins.to_csv(
        outdir / "causal_rank_bins_spatial_head_mass.csv",
        index=False,
    )
    ranktrend.to_csv(
        outdir / "causal_rank_monotonicity.csv",
        index=False,
    )

    print("4/6 layer-matched TopK enrichment...", flush=True)
    # Enrichment is the expensive part. Use focused metrics only.
    enrich_detail, enrich = layer_matched_enrichment(
        state, ranked, focused, ks, a.random_repeats, a.seed + 101
    )
    enrich_detail.to_csv(
        outdir / "topk_vs_layer_matched_random_detail.csv",
        index=False,
    )
    enrich.to_csv(
        outdir / "topk_vs_layer_matched_random_summary.csv",
        index=False,
    )

    print("5/6 spatial-head vs bottom-head controls...", flush=True)
    controls = spatial_vs_bottom(state, ranked, top_counts, ks)
    controls.to_csv(
        outdir / "spatial_vs_bottom_head_control.csv",
        index=False,
    )

    print("6/6 four-way candidate K selection...", flush=True)
    selector, selector_detail = fourway_selector(ranked, focused, ks)
    selector.to_csv(
        outdir / "fourway_K_selection_by_spatial_head_mass.csv",
        index=False,
    )
    selector_detail.to_csv(
        outdir / "fourway_K_selection_detail.csv",
        index=False,
    )

    meta = {
        "N_samples": int(state.sid.nunique()),
        "layers": layers,
        "cache_head_ids": cache_heads,
        "ks": ks,
        "top_head_counts": top_counts,
        "random_repeats": a.random_repeats,
        "paths": paths,
        "mass_interpretation": (
            "same-layer last-query -> candidate-token attention; "
            "READ/routing mass, not per-head output-write norm"
        ),
        "direction_head_definition": (
            "ref-sub Real-NoImage pre-W_O head residual direction decoding accuracy"
        ),
        "overall_head_weight": "max(residual_accuracy_mean - 0.25, 0)",
        "relation_specific_head_weight": (
            "max(residual_<candidate relation>_accuracy - 0.25, 0)"
        ),
        "focused_mass_metrics": focused,
        "controls": [
            "within sample+candidate+layer correlation",
            "bottom direction heads",
            "head-label permutation",
            "sample+layer matched random token states",
            "baseline-correct vs baseline-wrong split",
        ],
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    text = render_summary(
        a, state, ranked, focused,
        headcorr, statecorr, ranktrend, rankbins,
        enrich, selector, controls
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    print()
    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
