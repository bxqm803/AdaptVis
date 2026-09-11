
# -*- coding: utf-8 -*-

"""
analyze_helpful_harmful_updates_vs_direction_heads_v1.py

QUESTION
========
Do behaviorally HELPFUL vs HARMFUL actual causal-token updates have any
relationship to:

  (1) Direction Heads?
  (2) the spatial relation represented by those Direction Heads?
  (3) optionally, direct update-vs-spatial-direction geometry?

NO generation.  NO model forward.  Pure post-processing of existing caches.

Definitions
===========
Each actual block update is already labeled by the oracle diagnostic:

    B_REAL(L,p)
      = < a_REAL(L,p),
          grad [S_GT - S_comp] >

Helpful update:
    B_REAL > +threshold

Harmful update:
    B_REAL < -threshold

This script does NOT use that sign to intervene.  It uses it only as an
analysis label.

A. Direction-Head involvement in each actual update
----------------------------------------------------
Read:
    per_head_decision_contribution.csv

For each concrete update u=(sid,L,p), aggregate the head-level behavioral
contributions:

    B_h = <m_h, grad final GT margin>

For a Direction-Head set D:

    abs_share_D(u)
      = sum_{h in D} |B_h| / sum_all_heads |B_h|

    signed_D(u)
      = sum_{h in D} B_h

    positive_D(u)
      = sum_{h in D} max(B_h,0)

    negative_D(u)
      = sum_{h in D} max(-B_h,0)

Then compare helpful vs harmful updates.

Three Direction-Head definitions are reported:

  known_direction
      `is_direction_head` saved by
      analyze_real_update_module_head_sources_v1.py.

  source_topK
      Global Top-K heads ranked only by Synthetic-400 OOF accuracy.
      Default K=10.

  source_layer_topK
      Top-K Synthetic-OOF heads independently inside each layer.
      Default K=3/layer.

This answers:
    Are harmful updates unusually dominated by Direction Heads?
    Do Direction Heads contribute with the SAME sign as the total update?

B. Spatial semantic sign vs behavioral sign of a Direction Head
---------------------------------------------------------------
Inputs:
    synthetic_fitted_direction_codebook.npz
    source_oof_head_reliability.csv
    COCO relation_vectors.npz

The COCO cache contains the sample-level pre-W_O Direction-head residual:

    r_res(i,L,h)
      = [z_REAL(sub)-z_REAL(ref)]
        - [z_NOIMAGE(sub)-z_NOIMAGE(ref)]

The source-only codebook contains:
    center[L,h]
    d[L,h,left/right/above/below]

For every sample/head:

    score_h(r)
      = cosine(r_res - center, d_r)

    spatial_pred_h
      = argmax_r score_h(r)

    spatial_GT_margin_h
      = score_h(GT) - max_{r!=GT} score_h(r)

Then join this spatial semantic result to EVERY causal-token message emitted by
that same head/layer.

For each Direction-head message we therefore have TWO independent signs:

    spatial semantic:
        spatial_pred == GT ?
        spatial_GT_margin > 0 ?

    behavioral:
        B_h > 0 ? helps final GT decision
        B_h < 0 ? hurts final GT decision

The key 2x2 table is:

    spatial correct   & behavioral positive
    spatial correct   & behavioral negative   <-- representation right, use wrong
    spatial wrong     & behavioral positive
    spatial wrong     & behavioral negative

The especially interesting quantity is:

    P(B_h < 0 | spatial head decodes GT correctly)

and whether it is larger on:
    baseline-wrong samples
or inside:
    harmful B_REAL < 0 updates.

C. Optional direct update-direction geometry
--------------------------------------------
If --spatial-projection-dir is supplied and contains:

    per_update_spatial_projection.csv

the script also summarizes the already-computed direct geometry:

    spatial4_margin
    axis_margin
vs
    oracle_B

This is optional because the earlier experiment already suggested this direct
cosine-style route is near chance.

IMPORTANT INTERPRETATION
========================
This script tests ASSOCIATION / attribution only.

It can support statements such as:
    "Direction Heads are overrepresented in harmful updates"
or
    "spatially correct Direction-head messages are often behaviorally negative."

It cannot by itself prove that a head CAUSES the error.  Head ablation/restore
would be the next experiment only if a relationship is found.

Recommended full cached analysis
================================
python -u analyze_helpful_harmful_updates_vs_direction_heads_v1.py \
  --module-head-dir output/qwen3b_real_update_module_head_sources_all440_v1 \
  --direction-selector-dir output/qwen3b_direction_selector_syn400_to_coco440 \
  --target-direction-dir output/qwen3b_head_object_residual_direction \
  --source-topk 10 \
  --source-layer-topk 3 \
  --output-dir output/qwen3b_helpful_harmful_vs_direction_heads_v1 \
  --overwrite

Optional direct-vector section:
  --spatial-projection-dir output/qwen3b_update_spatial_direction_generation_n80_v1

Outputs
=======
per_update_direction_involvement.csv
    one row per actual update.

per_direction_head_message_semantics.csv
    one row per selected Direction-head message, with spatial + behavioral signs.

update_group_summary.csv
    helpful vs harmful update comparison.

update_layer_summary.csv
    same comparison per layer.

semantic_behavior_quadrants.csv
    core 2x2 spatial-semantic x behavioral-sign table.

semantic_behavior_by_baseline.csv
    especially baseline-wrong vs baseline-correct.

semantic_behavior_by_update_polarity.csv
    helpful vs harmful B_REAL context.

direction_message_correlations.csv

top_head_enrichment_by_update_polarity.csv

direct_projection_summary.csv
    only when optional spatial projection input is provided.

analysis_summary.txt
metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--module-head-dir",
        required=True,
        help=(
            "Output of analyze_real_update_module_head_sources_v1.py; must contain "
            "per_module_decision_contribution.csv and per_head_decision_contribution.csv."
        ),
    )

    p.add_argument(
        "--direction-selector-dir",
        required=True,
        help=(
            "Output of eval_direction_head_selector_synthetic400_to_coco440_v2.py; "
            "must contain synthetic_fitted_direction_codebook.npz and "
            "source_oof_head_reliability.csv."
        ),
    )

    p.add_argument(
        "--target-direction-dir",
        required=True,
        help="COCO Direction-head relation_vectors.npz OR directory containing it.",
    )

    p.add_argument(
        "--spatial-projection-dir",
        default="",
        help=(
            "Optional output directory from eval_update_spatial_direction_polarity_generation_v1.py "
            "or direct path to per_update_spatial_projection.csv."
        ),
    )

    p.add_argument(
        "--source-topk",
        type=int,
        default=10,
        help="Global Synthetic-OOF Direction-head set.",
    )
    p.add_argument(
        "--source-layer-topk",
        type=int,
        default=3,
        help="Synthetic-OOF Top-K selected independently within each layer.",
    )

    p.add_argument(
        "--b-threshold",
        type=float,
        default=1e-8,
        help="|B_REAL| <= threshold is called neutral and excluded from helpful/harmful comparison.",
    )
    p.add_argument(
        "--head-b-threshold",
        type=float,
        default=1e-10,
        help="|B_head| <= threshold is neutral in semantic/behavior quadrant analysis.",
    )

    p.add_argument(
        "--top-head-k",
        default="1,3,5",
        help="Top-|B_head| budgets for Direction-head enrichment.",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return {
        "left": "left",
        "left_of": "left",
        "left of": "left",
        "right": "right",
        "right_of": "right",
        "right of": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "bottom": "below",
    }.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(x)


def parse_ints(s):
    return sorted(
        {
            int(x.strip())
            for x in str(s).split(",")
            if x.strip()
        }
    )


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def safe_mean(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def safe_median(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def safe_corr(x, y, method="pearson"):
    a = pd.Series(np.asarray(list(x), dtype=np.float64))
    b = pd.Series(np.asarray(list(y), dtype=np.float64))
    ok = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    if int(ok.sum()) < 3:
        return float("nan")
    return float(a[ok].corr(b[ok], method=method))


def standardized_mean_diff(a, b):
    """
    Cohen-style standardized difference: mean(a)-mean(b) / pooled SD.
    """
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = math.sqrt(
        max(
            ((len(a) - 1) * va + (len(b) - 1) * vb)
            / max(len(a) + len(b) - 2, 1),
            0.0,
        )
    )
    if pooled <= EPS:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def binary_auc(labels, scores):
    """
    Rank-based AUC. labels: 1 = positive class (here harmful).
    """
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=np.float64)
    ok = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[ok]
    s = s[ok]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    sum_r1 = float(ranks[y == 1].sum())
    u1 = sum_r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, EPS)


# =============================================================================
# Load inputs
# =============================================================================

def load_module_head_tables(run_dir: Path):
    mpath = run_dir / "per_module_decision_contribution.csv"
    hpath = run_dir / "per_head_decision_contribution.csv"

    if not mpath.exists():
        raise FileNotFoundError(mpath)
    if not hpath.exists():
        raise FileNotFoundError(hpath)

    module = pd.read_csv(mpath)
    head = pd.read_csv(hpath)

    required_m = {
        "sid",
        "gt",
        "baseline_correct",
        "real_position",
        "update_layer",
        "B_real",
        "B_attn",
        "B_mlp",
    }
    required_h = {
        "sid",
        "gt",
        "baseline_correct",
        "real_position",
        "update_layer",
        "head",
        "head_name",
        "head_decision_score",
        "layer_B_real",
    }

    miss = required_m - set(module.columns)
    if miss:
        raise RuntimeError(f"{mpath} missing columns: {sorted(miss)}")

    miss = required_h - set(head.columns)
    if miss:
        raise RuntimeError(f"{hpath} missing columns: {sorted(miss)}")

    for df in (module, head):
        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
        df["update_layer"] = pd.to_numeric(
            df["update_layer"], errors="raise"
        ).astype(int)
        df["real_position"] = pd.to_numeric(
            df["real_position"], errors="raise"
        ).astype(int)
        df["gt"] = df["gt"].map(canon_rel)
        df["baseline_correct"] = df["baseline_correct"].map(boolify)

    head["head"] = pd.to_numeric(head["head"], errors="raise").astype(int)
    head["head_decision_score"] = pd.to_numeric(
        head["head_decision_score"], errors="coerce"
    )
    head["layer_B_real"] = pd.to_numeric(
        head["layer_B_real"], errors="coerce"
    )

    if "is_direction_head" in head.columns:
        head["is_direction_head"] = head["is_direction_head"].map(boolify)
    else:
        head["is_direction_head"] = False

    module["B_real"] = pd.to_numeric(module["B_real"], errors="coerce")
    module["B_attn"] = pd.to_numeric(module["B_attn"], errors="coerce")
    module["B_mlp"] = pd.to_numeric(module["B_mlp"], errors="coerce")

    return module, head, mpath, hpath


def load_source_direction(selector_dir: Path):
    cpath = selector_dir / "synthetic_fitted_direction_codebook.npz"
    rpath = selector_dir / "source_oof_head_reliability.csv"

    if not cpath.exists():
        raise FileNotFoundError(cpath)
    if not rpath.exists():
        raise FileNotFoundError(rpath)

    z = np.load(cpath, allow_pickle=True)
    required = {"center", "directions"}
    miss = required - set(z.files)
    if miss:
        raise RuntimeError(f"{cpath} missing arrays {sorted(miss)}")

    center = np.asarray(z["center"], dtype=np.float32)
    dirs = np.asarray(z["directions"], dtype=np.float32)

    if "relations" in z.files:
        names = [canon_rel(x) for x in z["relations"].tolist()]
    else:
        names = list(REL)

    if set(names) != set(REL):
        raise RuntimeError(f"Unexpected codebook relation order: {names}")

    order = [names.index(r) for r in REL]
    dirs = dirs[:, :, order, :]

    rel = pd.read_csv(rpath)
    for c in ("layer", "head", "rank"):
        if c in rel.columns:
            rel[c] = pd.to_numeric(rel[c], errors="raise").astype(int)

    if "source_oof_accuracy" not in rel.columns:
        raise RuntimeError(f"{rpath} missing source_oof_accuracy")

    rel["source_oof_accuracy"] = pd.to_numeric(
        rel["source_oof_accuracy"], errors="raise"
    ).astype(float)

    return center, dirs, rel, cpath, rpath


def resolve_target_npz(path: Path):
    if path.is_dir():
        path = path / "relation_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_target_direction(path: Path):
    p = resolve_target_npz(path)
    z = np.load(p, allow_pickle=True)

    required = {"residual", "relation"}
    miss = required - set(z.files)
    if miss:
        raise RuntimeError(f"{p} missing arrays {sorted(miss)}")

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)

    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    else:
        sid = np.arange(len(y), dtype=int)

    return p, sid, y, X


# =============================================================================
# Head-set annotations
# =============================================================================

def annotate_source_sets(
    head: pd.DataFrame,
    reliability: pd.DataFrame,
    source_topk: int,
    source_layer_topk: int,
):
    rank_map = {
        (int(r.layer), int(r.head)): int(r.rank)
        for r in reliability.itertuples()
        if hasattr(r, "rank")
    }
    acc_map = {
        (int(r.layer), int(r.head)): float(r.source_oof_accuracy)
        for r in reliability.itertuples()
    }

    layer_rank_map = {}
    for L, g in reliability.groupby("layer"):
        z = g.sort_values(
            ["source_oof_accuracy", "head"],
            ascending=[False, True],
        )
        for j, r in enumerate(z.itertuples(), 1):
            layer_rank_map[(int(L), int(r.head))] = int(j)

    keys = list(
        zip(
            head["update_layer"].astype(int),
            head["head"].astype(int),
        )
    )

    head = head.copy()
    head["source_oof_rank"] = [
        rank_map.get(k, -1) for k in keys
    ]
    head["source_oof_accuracy"] = [
        acc_map.get(k, np.nan) for k in keys
    ]
    head["source_layer_rank"] = [
        layer_rank_map.get(k, -1) for k in keys
    ]

    head["is_source_topK"] = (
        (head["source_oof_rank"] > 0)
        & (head["source_oof_rank"] <= int(source_topk))
    )
    head["is_source_layer_topK"] = (
        (head["source_layer_rank"] > 0)
        & (head["source_layer_rank"] <= int(source_layer_topk))
    )

    return head


# =============================================================================
# Spatial semantic score for every sample/head/layer
# =============================================================================

def build_semantic_table(
    *,
    head: pd.DataFrame,
    center: np.ndarray,
    dirs: np.ndarray,
    target_sid: np.ndarray,
    target_y: np.ndarray,
    target_X: np.ndarray,
):
    sid_to_row = {int(s): i for i, s in enumerate(target_sid.tolist())}

    unique = (
        head[
            [
                "sid",
                "gt",
                "baseline_correct",
                "update_layer",
                "head",
                "head_name",
                "is_direction_head",
                "is_source_topK",
                "is_source_layer_topK",
                "source_oof_rank",
                "source_layer_rank",
                "source_oof_accuracy",
            ]
        ]
        .drop_duplicates(["sid", "update_layer", "head"])
        .copy()
    )

    rows = []

    for r in unique.itertuples():
        sid = int(r.sid)
        L = int(r.update_layer)
        h = int(r.head)
        gt = canon_rel(r.gt)

        if sid not in sid_to_row:
            continue
        rr = sid_to_row[sid]

        if not (
            0 <= L < target_X.shape[1]
            and 0 <= h < target_X.shape[2]
            and L < center.shape[0]
            and h < center.shape[1]
        ):
            continue

        x = np.asarray(target_X[rr, L, h], dtype=np.float32)
        c = np.asarray(center[L, h], dtype=np.float32)
        d = np.asarray(dirs[L, h], dtype=np.float32)

        xc = x - c
        xn = xc / max(float(np.linalg.norm(xc)), EPS)
        dn = normalize_rows(d)

        scores = np.einsum("rd,d->r", dn, xn)
        pred_idx = int(np.argmax(scores))
        pred = REL[pred_idx]

        gi = RID[gt]
        strongest_other_idx = max(
            (i for i in range(len(REL)) if i != gi),
            key=lambda i: float(scores[i]),
        )
        strongest_other = REL[strongest_other_idx]
        margin = float(scores[gi] - scores[strongest_other_idx])

        rows.append(
            {
                "sid": sid,
                "gt": gt,
                "baseline_correct": bool(r.baseline_correct),
                "update_layer": L,
                "head": h,
                "head_name": str(r.head_name),
                "is_direction_head": bool(r.is_direction_head),
                "is_source_topK": bool(r.is_source_topK),
                "is_source_layer_topK": bool(r.is_source_layer_topK),
                "source_oof_rank": int(r.source_oof_rank),
                "source_layer_rank": int(r.source_layer_rank),
                "source_oof_accuracy": float(r.source_oof_accuracy),
                "target_cache_gt": canon_rel(target_y[rr]),
                "spatial_pred": pred,
                "spatial_correct": pred == gt,
                "spatial_gt_margin": margin,
                "spatial_strongest_other": strongest_other,
                **{
                    f"spatial_score_{rel}": float(scores[RID[rel]])
                    for rel in REL
                },
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Update-level Direction-head involvement
# =============================================================================

UPDATE_KEYS = ["sid", "update_layer", "real_position"]


def aggregate_update_involvement(
    module: pd.DataFrame,
    head: pd.DataFrame,
    b_threshold: float,
):
    meta_cols = [
        "sid",
        "gt",
        "baseline_correct",
        "real_position",
        "update_layer",
        "B_real",
        "B_attn",
        "B_mlp",
    ]
    for c in (
        "baseline_prediction",
        "competitor",
        "token",
        "category",
        "broad_category",
    ):
        if c in module.columns:
            meta_cols.append(c)

    meta = module[meta_cols].drop_duplicates(UPDATE_KEYS).copy()

    def label_b(v):
        v = float(v)
        if v > b_threshold:
            return "helpful"
        if v < -b_threshold:
            return "harmful"
        return "neutral"

    meta["update_polarity"] = meta["B_real"].map(label_b)

    definitions = {
        "known_direction": "is_direction_head",
        "source_topK": "is_source_topK",
        "source_layer_topK": "is_source_layer_topK",
    }

    rows = []

    for key, g in head.groupby(UPDATE_KEYS, sort=False):
        sid, L, p = map(int, key)
        total_abs = float(g["head_decision_score"].abs().sum())
        total_signed = float(g["head_decision_score"].sum())

        row = {
            "sid": sid,
            "update_layer": L,
            "real_position": p,
            "head_behavior_abs_total": total_abs,
            "head_behavior_signed_total": total_signed,
        }

        for name, col in definitions.items():
            z = g[g[col].astype(bool)]
            signed = float(z["head_decision_score"].sum())
            abs_sum = float(z["head_decision_score"].abs().sum())
            pos = float(z["head_decision_score"].clip(lower=0).sum())
            neg = float((-z["head_decision_score"].clip(upper=0)).sum())

            row[f"{name}_n_heads"] = int(len(z))
            row[f"{name}_signed_B"] = signed
            row[f"{name}_abs_B"] = abs_sum
            row[f"{name}_positive_mass"] = pos
            row[f"{name}_negative_mass"] = neg
            row[f"{name}_abs_share"] = (
                abs_sum / max(total_abs, EPS)
            )
            row[f"{name}_signed_over_abs_total"] = (
                signed / max(total_abs, EPS)
            )

        rows.append(row)

    agg = pd.DataFrame(rows)
    out = meta.merge(agg, on=UPDATE_KEYS, how="left")

    for name in definitions:
        col = f"{name}_signed_B"
        out[f"{name}_same_sign_as_update"] = (
            np.sign(out[col].to_numpy(float))
            == np.sign(out["B_real"].to_numpy(float))
        )

    return out


def update_group_summary(update_df):
    metrics = []
    for definition in (
        "known_direction",
        "source_topK",
        "source_layer_topK",
    ):
        metrics.extend(
            [
                f"{definition}_abs_share",
                f"{definition}_signed_B",
                f"{definition}_signed_over_abs_total",
                f"{definition}_positive_mass",
                f"{definition}_negative_mass",
            ]
        )

    rows = []

    for cohort_name, cohort in [
        ("all", update_df),
        ("baseline_wrong", update_df[~update_df["baseline_correct"]]),
        ("baseline_correct", update_df[update_df["baseline_correct"]]),
    ]:
        h = cohort[cohort["update_polarity"] == "helpful"]
        w = cohort[cohort["update_polarity"] == "harmful"]

        for metric in metrics:
            if metric not in cohort.columns:
                continue

            hv = h[metric].to_numpy(float)
            wv = w[metric].to_numpy(float)

            rows.append(
                {
                    "cohort": cohort_name,
                    "metric": metric,
                    "N_helpful": int(len(h)),
                    "N_harmful": int(len(w)),
                    "mean_helpful": safe_mean(hv),
                    "mean_harmful": safe_mean(wv),
                    "harmful_minus_helpful": (
                        safe_mean(wv) - safe_mean(hv)
                    ),
                    "std_effect_harmful_minus_helpful": standardized_mean_diff(
                        wv, hv
                    ),
                    "AUC_harmful_from_metric": binary_auc(
                        np.concatenate(
                            [
                                np.zeros(len(h), dtype=int),
                                np.ones(len(w), dtype=int),
                            ]
                        ),
                        np.concatenate([hv, wv]),
                    ),
                }
            )

    return pd.DataFrame(rows)


def update_layer_summary(update_df):
    rows = []

    for L, g in update_df.groupby("update_layer"):
        h = g[g["update_polarity"] == "helpful"]
        w = g[g["update_polarity"] == "harmful"]

        for definition in (
            "known_direction",
            "source_topK",
            "source_layer_topK",
        ):
            metric = f"{definition}_abs_share"
            rows.append(
                {
                    "update_layer": int(L),
                    "definition": definition,
                    "N_helpful": int(len(h)),
                    "N_harmful": int(len(w)),
                    "mean_abs_share_helpful": safe_mean(h[metric]),
                    "mean_abs_share_harmful": safe_mean(w[metric]),
                    "harmful_minus_helpful": (
                        safe_mean(w[metric])
                        - safe_mean(h[metric])
                    ),
                    "AUC_harmful_from_abs_share": binary_auc(
                        np.concatenate(
                            [
                                np.zeros(len(h), dtype=int),
                                np.ones(len(w), dtype=int),
                            ]
                        ),
                        np.concatenate(
                            [
                                h[metric].to_numpy(float),
                                w[metric].to_numpy(float),
                            ]
                        ),
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Per-head semantic x behavioral relationship
# =============================================================================

def join_head_semantics(
    head: pd.DataFrame,
    semantic: pd.DataFrame,
    module: pd.DataFrame,
    b_threshold: float,
    head_b_threshold: float,
):
    sem_cols = [
        "sid",
        "update_layer",
        "head",
        "spatial_pred",
        "spatial_correct",
        "spatial_gt_margin",
        "spatial_strongest_other",
        "target_cache_gt",
    ] + [f"spatial_score_{r}" for r in REL]

    z = head.merge(
        semantic[sem_cols],
        on=["sid", "update_layer", "head"],
        how="left",
        validate="many_to_one",
    )

    module_meta = (
        module[
            [
                "sid",
                "update_layer",
                "real_position",
                "B_real",
            ]
        ]
        .drop_duplicates(UPDATE_KEYS)
    )

    z = z.merge(
        module_meta,
        on=UPDATE_KEYS,
        how="left",
        suffixes=("", "_module"),
        validate="many_to_one",
    )

    B_real = z["B_real"].to_numpy(float)
    B_head = z["head_decision_score"].to_numpy(float)

    z["update_polarity"] = np.where(
        B_real > b_threshold,
        "helpful",
        np.where(
            B_real < -b_threshold,
            "harmful",
            "neutral",
        ),
    )

    z["head_behavior_sign"] = np.where(
        B_head > head_b_threshold,
        "positive",
        np.where(
            B_head < -head_b_threshold,
            "negative",
            "neutral",
        ),
    )

    z["semantic_sign"] = np.where(
        z["spatial_gt_margin"].to_numpy(float) > 0,
        "correct",
        np.where(
            z["spatial_gt_margin"].to_numpy(float) < 0,
            "wrong",
            "neutral",
        ),
    )

    z["spatial_correct_behavior_negative"] = (
        (z["semantic_sign"] == "correct")
        & (z["head_behavior_sign"] == "negative")
    )

    z["spatial_correct_behavior_positive"] = (
        (z["semantic_sign"] == "correct")
        & (z["head_behavior_sign"] == "positive")
    )

    return z


def quadrant_summary(head_sem):
    rows = []

    definitions = [
        ("known_direction", "is_direction_head"),
        ("source_topK", "is_source_topK"),
        ("source_layer_topK", "is_source_layer_topK"),
    ]

    for def_name, col in definitions:
        base = head_sem[head_sem[col].astype(bool)].copy()

        for cohort_name, g in [
            ("all", base),
            ("baseline_wrong", base[~base["baseline_correct"]]),
            ("baseline_correct", base[base["baseline_correct"]]),
        ]:
            g = g[
                g["semantic_sign"].isin(["correct", "wrong"])
                & g["head_behavior_sign"].isin(["positive", "negative"])
            ]

            denom = max(len(g), 1)
            for sem in ("correct", "wrong"):
                for beh in ("positive", "negative"):
                    n = int(
                        (
                            (g["semantic_sign"] == sem)
                            & (g["head_behavior_sign"] == beh)
                        ).sum()
                    )
                    rows.append(
                        {
                            "definition": def_name,
                            "cohort": cohort_name,
                            "semantic": sem,
                            "behavioral": beh,
                            "N": n,
                            "fraction_of_all_nonzero_messages": n / denom,
                        }
                    )

            sem_correct = g[g["semantic_sign"] == "correct"]
            rows.append(
                {
                    "definition": def_name,
                    "cohort": cohort_name,
                    "semantic": "correct",
                    "behavioral": "P_negative_given_semantic_correct",
                    "N": int(len(sem_correct)),
                    "fraction_of_all_nonzero_messages": (
                        float(
                            (
                                sem_correct["head_behavior_sign"]
                                == "negative"
                            ).mean()
                        )
                        if len(sem_correct)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def semantic_by_baseline(head_sem):
    rows = []

    definitions = [
        ("known_direction", "is_direction_head"),
        ("source_topK", "is_source_topK"),
        ("source_layer_topK", "is_source_layer_topK"),
    ]

    for def_name, col in definitions:
        z = head_sem[head_sem[col].astype(bool)].copy()

        for baseline_name, g in [
            ("wrong", z[~z["baseline_correct"]]),
            ("correct", z[z["baseline_correct"]]),
        ]:
            q = g[
                (g["semantic_sign"] == "correct")
                & g["head_behavior_sign"].isin(["positive", "negative"])
            ]
            rows.append(
                {
                    "definition": def_name,
                    "baseline": baseline_name,
                    "N_messages": int(len(g)),
                    "N_semantic_correct_nonzero": int(len(q)),
                    "spatial_correct_rate": float(
                        g["spatial_correct"].mean()
                    ) if len(g) else np.nan,
                    "negative_given_spatial_correct": float(
                        (q["head_behavior_sign"] == "negative").mean()
                    ) if len(q) else np.nan,
                    "mean_B_head_when_spatial_correct": safe_mean(
                        q["head_decision_score"]
                    ),
                    "mean_spatial_margin": safe_mean(
                        g["spatial_gt_margin"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def semantic_by_update_polarity(head_sem):
    rows = []

    definitions = [
        ("known_direction", "is_direction_head"),
        ("source_topK", "is_source_topK"),
        ("source_layer_topK", "is_source_layer_topK"),
    ]

    for def_name, col in definitions:
        z = head_sem[head_sem[col].astype(bool)].copy()

        for upol in ("helpful", "harmful"):
            g = z[z["update_polarity"] == upol]
            q = g[
                (g["semantic_sign"] == "correct")
                & g["head_behavior_sign"].isin(["positive", "negative"])
            ]

            rows.append(
                {
                    "definition": def_name,
                    "update_polarity": upol,
                    "N_messages": int(len(g)),
                    "N_semantic_correct_nonzero": int(len(q)),
                    "spatial_correct_rate": (
                        float(g["spatial_correct"].mean())
                        if len(g) else np.nan
                    ),
                    "negative_given_spatial_correct": (
                        float(
                            (q["head_behavior_sign"] == "negative").mean()
                        )
                        if len(q) else np.nan
                    ),
                    "mean_B_head_when_spatial_correct": safe_mean(
                        q["head_decision_score"]
                    ),
                    "mean_abs_B_head_when_spatial_correct": safe_mean(
                        q["head_decision_score"].abs()
                    ),
                    "mean_spatial_margin": safe_mean(
                        g["spatial_gt_margin"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def correlation_summary(head_sem):
    rows = []

    definitions = [
        ("all_heads", None),
        ("known_direction", "is_direction_head"),
        ("source_topK", "is_source_topK"),
        ("source_layer_topK", "is_source_layer_topK"),
    ]

    for def_name, col in definitions:
        if col is None:
            z = head_sem.copy()
        else:
            z = head_sem[head_sem[col].astype(bool)].copy()

        for cohort_name, g in [
            ("all", z),
            ("baseline_wrong", z[~z["baseline_correct"]]),
            ("baseline_correct", z[z["baseline_correct"]]),
            ("helpful_update", z[z["update_polarity"] == "helpful"]),
            ("harmful_update", z[z["update_polarity"] == "harmful"]),
        ]:
            rows.append(
                {
                    "definition": def_name,
                    "cohort": cohort_name,
                    "N_messages": int(len(g)),
                    "pearson_spatial_margin_vs_B_head": safe_corr(
                        g["spatial_gt_margin"],
                        g["head_decision_score"],
                        "pearson",
                    ),
                    "spearman_spatial_margin_vs_B_head": safe_corr(
                        g["spatial_gt_margin"],
                        g["head_decision_score"],
                        "spearman",
                    ),
                    "pearson_spatial_margin_vs_abs_B_head": safe_corr(
                        g["spatial_gt_margin"],
                        g["head_decision_score"].abs(),
                        "pearson",
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Top-|B_head| enrichment within helpful/harmful updates
# =============================================================================

def top_head_enrichment(head_sem, topks):
    rows = []

    definitions = [
        ("known_direction", "is_direction_head"),
        ("source_topK", "is_source_topK"),
        ("source_layer_topK", "is_source_layer_topK"),
    ]

    z = head_sem[
        head_sem["update_polarity"].isin(["helpful", "harmful"])
    ].copy()

    for upol in ("helpful", "harmful"):
        gpol = z[z["update_polarity"] == upol]

        for def_name, col in definitions:
            base_rate = float(gpol[col].astype(bool).mean()) if len(gpol) else np.nan

            for k in topks:
                selected_parts = []
                for _, g in gpol.groupby(UPDATE_KEYS, sort=False):
                    zz = g.assign(
                        _absB=g["head_decision_score"].abs()
                    ).sort_values("_absB", ascending=False).head(int(k))
                    selected_parts.append(zz)

                selected = (
                    pd.concat(selected_parts, ignore_index=True)
                    if selected_parts
                    else gpol.iloc[:0].copy()
                )

                top_rate = (
                    float(selected[col].astype(bool).mean())
                    if len(selected)
                    else np.nan
                )

                rows.append(
                    {
                        "update_polarity": upol,
                        "definition": def_name,
                        "top_k_per_update": int(k),
                        "base_head_rate": base_rate,
                        "top_absB_head_rate": top_rate,
                        "enrichment": (
                            top_rate / base_rate
                            if (
                                np.isfinite(base_rate)
                                and base_rate > EPS
                                and np.isfinite(top_rate)
                            )
                            else np.nan
                        ),
                        "N_top_rows": int(len(selected)),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Optional direct update-vs-direction projection
# =============================================================================

def resolve_projection_csv(path_text: str):
    if not path_text:
        return None

    p = Path(path_text)
    if p.is_dir():
        p = p / "per_update_spatial_projection.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def direct_projection_summary(path: Optional[Path]):
    if path is None:
        return pd.DataFrame(), None

    df = pd.read_csv(path)

    required = {
        "sid",
        "update_layer",
        "real_position",
        "oracle_B",
        "spatial4_margin",
        "axis_margin",
    }
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(
            f"{path} missing direct-projection columns: {sorted(miss)}"
        )

    if "baseline_correct" not in df.columns:
        df["baseline_correct"] = np.nan
    else:
        df["baseline_correct"] = df["baseline_correct"].map(boolify)

    rows = []

    for method in ("spatial4_margin", "axis_margin"):
        for cohort_name, g in [
            ("all", df),
            (
                "baseline_wrong",
                df[df["baseline_correct"] == False],  # noqa: E712
            ),
            (
                "baseline_correct",
                df[df["baseline_correct"] == True],  # noqa: E712
            ),
        ]:
            if cohort_name != "all" and not len(g):
                continue

            q = g[
                np.isfinite(g[method])
                & np.isfinite(g["oracle_B"])
                & (np.sign(g[method]) != 0)
                & (np.sign(g["oracle_B"]) != 0)
            ]

            agreement = (
                np.sign(q[method].to_numpy(float))
                == np.sign(q["oracle_B"].to_numpy(float))
            )

            rows.append(
                {
                    "method": method,
                    "cohort": cohort_name,
                    "N": int(len(q)),
                    "sign_agreement_with_oracle_B": (
                        float(np.mean(agreement))
                        if len(q) else np.nan
                    ),
                    "pearson_with_oracle_B": safe_corr(
                        q[method],
                        q["oracle_B"],
                        "pearson",
                    ),
                    "spearman_with_oracle_B": safe_corr(
                        q[method],
                        q["oracle_B"],
                        "spearman",
                    ),
                }
            )

    return pd.DataFrame(rows), df


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)

    module_dir = Path(a.module_head_dir)
    selector_dir = Path(a.direction_selector_dir)

    module, head, module_path, head_path = load_module_head_tables(
        module_dir
    )
    center, dirs, reliability, code_path, reliability_path = (
        load_source_direction(selector_dir)
    )
    target_path, target_sid, target_y, target_X = load_target_direction(
        Path(a.target_direction_dir)
    )

    # Cohort sanity.
    target_sid_set = set(map(int, target_sid.tolist()))
    module_sid_set = set(map(int, module["sid"].unique().tolist()))
    overlap = module_sid_set & target_sid_set
    if not overlap:
        raise RuntimeError(
            "No sample-ID overlap between module/head run and target Direction cache."
        )

    head = annotate_source_sets(
        head,
        reliability,
        source_topk=int(a.source_topk),
        source_layer_topk=int(a.source_layer_topk),
    )

    print("=" * 200)
    print("HELPFUL / HARMFUL ACTUAL UPDATES vs DIRECTION HEADS")
    print("=" * 200)
    print(f"module/head samples={module['sid'].nunique()}")
    print(f"target Direction samples={len(target_sid)}")
    print(f"overlap samples={len(overlap)}")
    print(f"updates={module[UPDATE_KEYS].drop_duplicates().shape[0]}")
    print(f"head messages={len(head)}")
    print(f"source global TopK={a.source_topk}")
    print(f"source layer TopK={a.source_layer_topk}")
    print()

    # -------------------------------------------------------------------------
    # 1) Update-level head involvement.
    # -------------------------------------------------------------------------
    update_df = aggregate_update_involvement(
        module,
        head,
        float(a.b_threshold),
    )
    update_df.to_csv(
        outdir / "per_update_direction_involvement.csv",
        index=False,
    )

    update_summary = update_group_summary(update_df)
    update_summary.to_csv(
        outdir / "update_group_summary.csv",
        index=False,
    )

    layer_summary = update_layer_summary(update_df)
    layer_summary.to_csv(
        outdir / "update_layer_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 2) Spatial semantic sign of every sample/head.
    # -------------------------------------------------------------------------
    semantic = build_semantic_table(
        head=head,
        center=center,
        dirs=dirs,
        target_sid=target_sid,
        target_y=target_y,
        target_X=target_X,
    )
    semantic.to_csv(
        outdir / "per_sample_head_spatial_semantics.csv",
        index=False,
    )

    head_sem = join_head_semantics(
        head,
        semantic,
        module,
        float(a.b_threshold),
        float(a.head_b_threshold),
    )

    # Keep the most relevant messages: at least one Direction-head definition.
    selected_mask = (
        head_sem["is_direction_head"].astype(bool)
        | head_sem["is_source_topK"].astype(bool)
        | head_sem["is_source_layer_topK"].astype(bool)
    )
    selected_messages = head_sem[selected_mask].copy()

    selected_messages.to_csv(
        outdir / "per_direction_head_message_semantics.csv",
        index=False,
    )

    quadrants = quadrant_summary(head_sem)
    quadrants.to_csv(
        outdir / "semantic_behavior_quadrants.csv",
        index=False,
    )

    by_baseline = semantic_by_baseline(head_sem)
    by_baseline.to_csv(
        outdir / "semantic_behavior_by_baseline.csv",
        index=False,
    )

    by_update = semantic_by_update_polarity(head_sem)
    by_update.to_csv(
        outdir / "semantic_behavior_by_update_polarity.csv",
        index=False,
    )

    corr = correlation_summary(head_sem)
    corr.to_csv(
        outdir / "direction_message_correlations.csv",
        index=False,
    )

    enrich = top_head_enrichment(
        head_sem,
        parse_ints(a.top_head_k),
    )
    enrich.to_csv(
        outdir / "top_head_enrichment_by_update_polarity.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 3) Optional direct update -> spatial direction geometry.
    # -------------------------------------------------------------------------
    projection_path = resolve_projection_csv(
        a.spatial_projection_dir
    )
    direct_summary, direct_df = direct_projection_summary(
        projection_path
    )
    if len(direct_summary):
        direct_summary.to_csv(
            outdir / "direct_projection_summary.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Compact console/report.
    # -------------------------------------------------------------------------
    def get_metric(summary, cohort, metric):
        z = summary[
            (summary["cohort"] == cohort)
            & (summary["metric"] == metric)
        ]
        return z.iloc[0] if len(z) else None

    report_lines = []
    report_lines.append("=" * 200)
    report_lines.append("HELPFUL / HARMFUL ACTUAL UPDATES vs DIRECTION HEADS")
    report_lines.append("=" * 200)
    report_lines.append(
        f"N samples={module['sid'].nunique()} | "
        f"N updates={update_df[UPDATE_KEYS].drop_duplicates().shape[0]} | "
        f"N head messages={len(head)}"
    )
    report_lines.append("")
    report_lines.append(
        "A. DO HARMFUL UPDATES CONTAIN MORE DIRECTION-HEAD BEHAVIORAL MASS?"
    )
    report_lines.append("-" * 200)

    focus_metrics = []
    for definition in (
        "known_direction",
        "source_topK",
        "source_layer_topK",
    ):
        focus_metrics.extend(
            [
                f"{definition}_abs_share",
                f"{definition}_signed_over_abs_total",
            ]
        )

    focus = update_summary[
        update_summary["metric"].isin(focus_metrics)
    ].copy()

    report_lines.append(
        focus.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )
    report_lines.append("")

    report_lines.append(
        "B. SPATIALLY CORRECT DIRECTION HEAD -> BEHAVIORALLY NEGATIVE?"
    )
    report_lines.append("-" * 200)
    report_lines.append(
        by_baseline.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )
    report_lines.append("")

    report_lines.append(
        "C. SAME QUESTION CONDITIONED ON HELPFUL/HARMFUL TOTAL UPDATE"
    )
    report_lines.append("-" * 200)
    report_lines.append(
        by_update.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )
    report_lines.append("")

    report_lines.append(
        "D. SPATIAL MARGIN vs BEHAVIORAL HEAD CONTRIBUTION"
    )
    report_lines.append("-" * 200)
    report_lines.append(
        corr.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )
    report_lines.append("")

    report_lines.append(
        "E. ARE DIRECTION HEADS ENRICHED AMONG STRONGEST |B_h| CONTRIBUTORS?"
    )
    report_lines.append("-" * 200)
    report_lines.append(
        enrich.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )
    report_lines.append("")

    if len(direct_summary):
        report_lines.append(
            "F. OPTIONAL DIRECT UPDATE-vs-DIRECTION GEOMETRY"
        )
        report_lines.append("-" * 200)
        report_lines.append(
            direct_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )
        report_lines.append("")

    report_lines.extend(
        [
            "Reading guide:",
            "  1) abs_share harmful >> helpful:",
            "       harmful updates disproportionately involve Direction Heads.",
            "  2) signed_over_abs_total flips positive->negative:",
            "       the same Direction-head family contributes with behaviorally opposite signs.",
            "  3) negative_given_spatial_correct is high, especially baseline_wrong/harmful:",
            "       the head decodes the spatial relation correctly but its realized write",
            "       is behaviorally misaligned with the final GT decision.",
            "  4) spatial-margin/B_head correlation ~0:",
            "       spatial semantic strength does not determine behavioral polarity.",
            "  5) no group difference + no enrichment + no semantic/behavior relation:",
            "       Direction Heads are probably not the primary explanation for update sign.",
            "",
            "Caveat:",
            "  These are associations / additive attributions.  Do not call a head a causal",
            "  error source until a targeted ablation/restore test confirms it.",
        ]
    )

    report = "\n".join(report_lines) + "\n"
    print(report)

    (outdir / "analysis_summary.txt").write_text(
        report,
        encoding="utf-8",
    )

    metadata = {
        "script": "analyze_helpful_harmful_updates_vs_direction_heads_v1.py",
        "module_head_dir": str(module_dir),
        "per_module_csv": str(module_path),
        "per_head_csv": str(head_path),
        "direction_selector_dir": str(selector_dir),
        "source_codebook": str(code_path),
        "source_reliability": str(reliability_path),
        "target_direction_npz": str(target_path),
        "spatial_projection_csv": (
            str(projection_path) if projection_path is not None else ""
        ),
        "source_topk": int(a.source_topk),
        "source_layer_topk": int(a.source_layer_topk),
        "b_threshold": float(a.b_threshold),
        "head_b_threshold": float(a.head_b_threshold),
        "N_module_samples": int(module["sid"].nunique()),
        "N_direction_cache_samples": int(len(target_sid)),
        "N_overlap_samples": int(len(overlap)),
        "N_updates": int(
            update_df[UPDATE_KEYS].drop_duplicates().shape[0]
        ),
        "N_head_messages": int(len(head)),
        "definitions": {
            "helpful_update": "B_real > +b_threshold",
            "harmful_update": "B_real < -b_threshold",
            "head_behavior": "B_head=head_decision_score",
            "spatial_head_score": (
                "cosine(target residual - synthetic source center, "
                "synthetic source relation direction)"
            ),
            "spatial_correct": "argmax relation score == GT; evaluation only",
            "known_direction": "is_direction_head annotation from module/head run",
            "source_topK": (
                f"global Synthetic-OOF rank <= {int(a.source_topk)}"
            ),
            "source_layer_topK": (
                f"within-layer Synthetic-OOF rank <= {int(a.source_layer_topk)}"
            ),
        },
        "generation_performed": False,
        "model_forward_performed": False,
        "causal_claim": False,
    }

    (outdir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
