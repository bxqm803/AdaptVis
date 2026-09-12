#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_spatial_continuous_all440_v2.py

Threshold-free continuous decomposition of the model's middle-layer spatial
state on ALL target samples.

Motivation
==========
Do not first discretize each sample into weak / mixed / wrong-dominant using an
arbitrary evidence threshold.  Instead, for every target sample and every
requested layer, project the relation vector into the Synthetic-400 H/V spatial
coordinate system:

    z_L = (z_H, z_V)

and define continuous directional evidence

    e_left  = -z_H
    e_right = +z_H
    e_above = +z_V
    e_below = -z_V.

For each target sample, aggregate across layers and directly report:

    G = mean evidence for GT
    W = max_{r != GT} mean evidence for r
    M = G - W

M > 0 means GT is the strongest mean spatial direction.
M < 0 means some wrong direction is stronger than GT.
No COCO-tuned threshold is needed for this distinction.

Absolute strength is reported separately using source-only empirical percentiles:

  * GT evidence percentile: how strong the sample's GT evidence is relative to
    genuine Synthetic examples of that GT relation.
  * wrong evidence percentile: same for each wrong relation.
  * spatial norm percentile: how large ||z_L|| is relative to Synthetic-400.
  * top-evidence percentile: how large max_r e_r is relative to Synthetic-400.

Thus one can distinguish continuously:

  - GT-leading but weak: M > 0, low source-percentile strength.
  - clear GT-leading utilization failure: M >> 0 with high GT percentile.
  - mixed/ambiguous: M ~ 0, often both GT and competitor have high percentiles.
  - wrong-leading representation/competition: M < 0.
  - weak/OOD: low spatial-norm/top-evidence source percentile.

The script also checks whether the model's baseline wrong answer equals the
strongest wrong spatial direction.

No model is loaded. No intervention is performed. Synthetic labels define the
source H/V coordinate system and source reference distributions. Target GT is
used only for post-hoc diagnosis/evaluation.

Recommended ALL-440 run
=======================
python -u diagnose_spatial_continuous_all440_v2.py \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --baseline-csv \
    output/qwen3b_synthetic400_to_coco440_originalprompt_mean_v2/per_sample_candidate_repair.csv \
  --layers 20-26 \
  --expected-n 440 \
  --output-dir output/qwen3b_spatial_continuous_all440_v2 \
  --overwrite

Main outputs
============
per_sample_continuous_spatial.csv
    All 440 samples, one row each. Main table.
per_sample_layer_continuous_spatial.csv
    All sample x layer rows.
cohort_summary.csv
    all / baseline_wrong / baseline_correct continuous summaries.
correctness_x_spatial_leading.csv
    Main 2x3 decomposition: baseline correct/wrong x GT-leading/wrong-leading/tie.
spatial_leading_by_correctness.csv
    Compact counts/rates of GT-leading and wrong-leading inside correct and wrong samples.
by_gt_x_correctness_x_spatial_leading.csv
    Relation-wise decomposition of the same quadrants.
gt_rank_distribution.csv
    GT spatial rank 1/2/3/4 by cohort.
margin_quantiles.csv
    Distribution of M = GT - best-wrong evidence.
by_gt_summary.csv
by_baseline_prediction_summary.csv
baseline_wrong_gt_leading.csv
    Wrong-generation samples with M > 0, sorted from clearest GT-leading cases.
baseline_wrong_wrong_leading.csv
    Wrong-generation samples with M < 0, sorted from strongest wrong-leading cases.
baseline_wrong_most_ambiguous.csv
    Wrong-generation samples sorted by |M| ascending.
source_reference_statistics.csv
analysis_summary.txt
metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REL = ("left", "right", "above", "below")
EPS = 1e-10
SCRIPT_VERSION = "v2_correct_wrong_x_spatial_leading_all_samples"


def norm_rel(x) -> str:
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return v / n


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("No layers parsed")
    return sorted(out)


def axis_evidence(z) -> Dict[str, float]:
    h, v = float(z[0]), float(z[1])
    return {"left": -h, "right": h, "above": v, "below": -v}


def empirical_percentile(sorted_values: np.ndarray, x: float) -> float:
    vals = np.asarray(sorted_values, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        return float("nan")
    # midpoint-style ECDF: avoids exact 0/1 unless clearly outside range.
    left = np.searchsorted(vals, x, side="left")
    right = np.searchsorted(vals, x, side="right")
    rank = 0.5 * (left + right)
    return float((rank + 0.5) / (len(vals) + 1.0))


def safe_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().str.strip().isin(["true", "1", "t", "yes", "y"])


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--layers", default="20-26")
    p.add_argument(
        "--expected-n", type=int, default=440,
        help="Refuse to run if the baseline table does not contain this many unique samples. Use 0 to disable.",
    )
    p.add_argument(
        "--unstable-winner-fraction", type=float, default=0.60,
        help="Auxiliary flag only: modal layerwise winner fraction below this is cross-layer unstable.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    try:
        a.layers_parsed = parse_layers(a.layers)
    except ValueError as exc:
        p.error(str(exc))
    if a.expected_n < 0:
        p.error("--expected-n must be >= 0")
    if not (0 < a.unstable_winner_fraction <= 1):
        p.error("--unstable-winner-fraction must be in (0,1]")
    return a


def load_state_npz(path: Path, require_labels: bool):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            definition = str(z["vector_definition"].item()) if "vector_definition" in keys else "relation_vectors"
        elif {"img", "no_image"}.issubset(keys):
            X = np.asarray(z["img"], dtype=np.float32) - np.asarray(z["no_image"], dtype=np.float32)
            definition = "img_minus_no_image"
        else:
            raise RuntimeError(f"Bad NPZ keys in {path}: {sorted(keys)}")

        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]
        sids = (
            np.asarray(z["sample_index"], dtype=np.int64)
            if "sample_index" in keys else np.arange(len(X), dtype=np.int64)
        )
        labels = None
        if "relation" in keys:
            labels = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels")

    if X.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {X.shape}")
    if len(sids) != len(X):
        raise RuntimeError("sample_index length mismatch")
    return X, labels, layers, sids, definition


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_baseline(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = set(df.columns)

    sid_col = _first_existing(cols, ["sid", "sample_index", "sample_id", "index"])
    gt_col = _first_existing(cols, ["gt", "relation", "ground_truth", "answer"])
    pred_col = _first_existing(
        cols,
        [
            "baseline_prediction",
            "baseline_pred",
            "base_prediction",
            "generation_prediction",
            "generated_relation",
            "prediction",
        ],
    )
    if sid_col is None or gt_col is None or pred_col is None:
        raise RuntimeError(
            "Could not infer baseline columns. Need sample id, GT relation, and baseline prediction.\n"
            f"columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "sid": df[sid_col].astype(int),
        "gt": df[gt_col].map(norm_rel),
        "baseline_prediction": df[pred_col].map(norm_rel),
    })

    bc_col = _first_existing(cols, ["baseline_correct", "generation_correct", "correct"])
    if bc_col is not None:
        out["baseline_correct"] = safe_bool_series(df[bc_col])
    else:
        out["baseline_correct"] = out["gt"] == out["baseline_prediction"]

    if out["sid"].duplicated().any():
        dup = out.loc[out["sid"].duplicated(), "sid"].tolist()[:10]
        raise RuntimeError(f"baseline csv contains duplicate sid, e.g. {dup}")
    if any(r not in REL for r in set(out["gt"])):
        raise RuntimeError(f"Unexpected GT labels: {sorted(set(out['gt']) - set(REL))}")
    if any(r not in REL for r in set(out["baseline_prediction"])):
        raise RuntimeError(
            f"Unexpected baseline predictions: {sorted(set(out['baseline_prediction']) - set(REL))}"
        )
    return out.sort_values("sid").reset_index(drop=True)


def fit_hv_geometry(X, y, source_layers, wanted_layers):
    lmap = {L: i for i, L in enumerate(source_layers)}
    geom = {}
    rows = []
    for L in wanted_layers:
        if L not in lmap:
            raise RuntimeError(f"Source missing L{L}")
        Xf = X[:, lmap[L]].astype(np.float64)
        center = Xf.mean(axis=0)
        mus = {r: Xf[y == r].mean(axis=0) for r in REL}
        dirs = {r: unit(mus[r] - center) for r in REL}

        dH = unit(dirs["right"] - dirs["left"])
        dV = unit(dirs["above"] - dirs["below"])
        gapH = float(np.dot(mus["right"] - mus["left"], dH))
        gapV = float(np.dot(mus["above"] - mus["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV

        B = np.stack([dH, dV], axis=1)  # [D,2]
        dual = B @ np.linalg.inv(B.T @ B)
        geom[L] = {
            "center": center,
            "dual": dual,
            "halfH": max(gapH / 2.0, EPS),
            "halfV": max(gapV / 2.0, EPS),
            "layer_index": lmap[L],
        }
        rows.append({
            "layer": L,
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "full_gap_H": gapH,
            "full_gap_V": gapV,
            "half_gap_H": max(gapH / 2.0, EPS),
            "half_gap_V": max(gapV / 2.0, EPS),
        })
    return geom, pd.DataFrame(rows)


def read_coord(x, g):
    res = np.asarray(x, dtype=np.float64) - g["center"]
    c = g["dual"].T @ res
    return np.asarray([c[0] / g["halfH"], c[1] / g["halfV"]], dtype=np.float64)


def fit_source_reference(X, y, source_layers, geom, layers):
    lmap = {L: i for i, L in enumerate(source_layers)}
    ref = {}
    rows = []
    for L in layers:
        Z = np.stack([read_coord(X[i, lmap[L]], geom[L]) for i in range(len(X))])
        evidence = {r: np.asarray([axis_evidence(z)[r] for z in Z], dtype=np.float64) for r in REL}
        own_sorted = {}
        for r in REL:
            own = np.sort(evidence[r][y == r])
            own_sorted[r] = own
            rows.append({
                "layer": L,
                "statistic": f"own_evidence_{r}",
                "N": len(own),
                "mean": float(np.mean(own)),
                "std": float(np.std(own)),
                "q10": float(np.quantile(own, 0.10)),
                "q25": float(np.quantile(own, 0.25)),
                "q50": float(np.quantile(own, 0.50)),
                "q75": float(np.quantile(own, 0.75)),
                "q90": float(np.quantile(own, 0.90)),
            })

        norms = np.linalg.norm(Z, axis=1)
        top_evidence = np.max(np.stack([evidence[r] for r in REL], axis=1), axis=1)
        sorted_norm = np.sort(norms)
        sorted_top = np.sort(top_evidence)
        for name, vals in [("spatial_norm_all", norms), ("top_direction_evidence_all", top_evidence)]:
            rows.append({
                "layer": L,
                "statistic": name,
                "N": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "q10": float(np.quantile(vals, 0.10)),
                "q25": float(np.quantile(vals, 0.25)),
                "q50": float(np.quantile(vals, 0.50)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
            })

        ref[L] = {
            "own_evidence_sorted": own_sorted,
            "norm_sorted": sorted_norm,
            "top_evidence_sorted": sorted_top,
        }
    return ref, pd.DataFrame(rows)


def rank_desc(scores: Dict[str, float], relation: str) -> int:
    order = sorted(REL, key=lambda r: (-float(scores[r]), REL.index(r)))
    return int(order.index(relation) + 1)


def summarize_cohort(g: pd.DataFrame, name: str) -> dict:
    if len(g) == 0:
        return {"cohort": name, "N": 0}
    row = {
        "cohort": name,
        "N": len(g),
        "baseline_accuracy": float(g["baseline_correct"].mean()),
        "gt_top1_by_mean_evidence_rate": float(g["gt_rank_by_mean_evidence"].eq(1).mean()),
        "gt_top2_by_mean_evidence_rate": float(g["gt_rank_by_mean_evidence"].le(2).mean()),
        "gt_leads_best_wrong_rate": float(g["gt_minus_bestwrong_mean_evidence"].gt(0).mean()),
        "wrong_leads_gt_rate": float(g["gt_minus_bestwrong_mean_evidence"].lt(0).mean()),
        "mean_gt_evidence": float(g["mean_gt_direction_evidence"].mean()),
        "mean_best_wrong_evidence": float(g["mean_best_wrong_direction_evidence"].mean()),
        "mean_gt_minus_bestwrong": float(g["gt_minus_bestwrong_mean_evidence"].mean()),
        "median_gt_minus_bestwrong": float(g["gt_minus_bestwrong_mean_evidence"].median()),
        "mean_normalized_gt_margin": float(g["normalized_gt_vs_bestwrong_margin"].mean()),
        "mean_gt_source_percentile": float(g["mean_gt_evidence_source_percentile"].mean()),
        "median_gt_source_percentile": float(g["mean_gt_evidence_source_percentile"].median()),
        "mean_bestwrong_source_percentile": float(g["mean_bestwrong_evidence_source_percentile"].mean()),
        "mean_spatial_norm": float(g["mean_spatial_norm"].mean()),
        "mean_spatial_norm_source_percentile": float(g["mean_spatial_norm_source_percentile"].mean()),
        "mean_top_evidence_source_percentile": float(g["mean_top_evidence_source_percentile"].mean()),
        "mean_gt_layer_win_fraction": float(g["gt_top1_layer_fraction"].mean()),
        "cross_layer_unstable_rate": float(g["cross_layer_unstable"].mean()),
    }
    if (~g["baseline_correct"]).any():
        e = g.loc[~g["baseline_correct"]]
        row.update({
            "wrong_N": len(e),
            "baseline_equals_bestwrong_mean_evidence_rate_on_wrong": float(
                e["baseline_equals_bestwrong_mean_evidence"].mean()
            ),
            "baseline_equals_bestwrong_percentile_rate_on_wrong": float(
                e["baseline_equals_bestwrong_percentile"].mean()
            ),
        })
    else:
        row.update({
            "wrong_N": 0,
            "baseline_equals_bestwrong_mean_evidence_rate_on_wrong": np.nan,
            "baseline_equals_bestwrong_percentile_rate_on_wrong": np.nan,
        })
    return row


def grouped_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, sort=False):
        r = summarize_cohort(g, str(key))
        r[col] = key
        r.pop("cohort", None)
        rows.append(r)
    return pd.DataFrame(rows)


def correctness_leading_tables(samples: pd.DataFrame):
    """Build explicit correct/wrong x GT-leading/wrong-leading/tie tables."""
    rows = []
    compact = []
    status_order = ("gt_leading", "wrong_leading", "tie")
    correctness_order = ((True, "baseline_correct"), (False, "baseline_wrong"))
    N_all = len(samples)

    for bc, cname in correctness_order:
        cg = samples.loc[samples["baseline_correct"].eq(bc)].copy()
        Nc = len(cg)
        crow = {
            "baseline_cohort": cname,
            "N": Nc,
            "fraction_of_all": float(Nc / N_all) if N_all else np.nan,
        }
        for status in status_order:
            g = cg.loc[cg["spatial_leading_status"].eq(status)].copy()
            n = len(g)
            rows.append({
                "baseline_cohort": cname,
                "baseline_correct": bool(bc),
                "spatial_leading_status": status,
                "N": n,
                "fraction_of_all": float(n / N_all) if N_all else np.nan,
                "fraction_within_baseline_cohort": float(n / Nc) if Nc else np.nan,
                "mean_gt_evidence": float(g["mean_gt_direction_evidence"].mean()) if n else np.nan,
                "mean_best_wrong_evidence": float(g["mean_best_wrong_direction_evidence"].mean()) if n else np.nan,
                "mean_gt_minus_bestwrong": float(g["gt_minus_bestwrong_mean_evidence"].mean()) if n else np.nan,
                "median_gt_minus_bestwrong": float(g["gt_minus_bestwrong_mean_evidence"].median()) if n else np.nan,
                "mean_gt_source_percentile": float(g["mean_gt_evidence_source_percentile"].mean()) if n else np.nan,
                "mean_bestwrong_source_percentile": float(g["mean_bestwrong_evidence_source_percentile"].mean()) if n else np.nan,
                "mean_spatial_norm_source_percentile": float(g["mean_spatial_norm_source_percentile"].mean()) if n else np.nan,
                "gt_top1_rate": float(g["gt_rank_by_mean_evidence"].eq(1).mean()) if n else np.nan,
                "gt_top2_rate": float(g["gt_rank_by_mean_evidence"].le(2).mean()) if n else np.nan,
                "mean_gt_layer_win_fraction": float(g["gt_top1_layer_fraction"].mean()) if n else np.nan,
                "cross_layer_unstable_rate": float(g["cross_layer_unstable"].mean()) if n else np.nan,
                "baseline_equals_top1_direction_rate": float(g["baseline_equals_top1_direction"].mean()) if n else np.nan,
                "baseline_equals_bestwrong_rate": float(g["baseline_equals_bestwrong_mean_evidence"].mean()) if n else np.nan,
            })
            crow[f"N_{status}"] = n
            crow[f"frac_{status}"] = float(n / Nc) if Nc else np.nan
        compact.append(crow)
    return pd.DataFrame(rows), pd.DataFrame(compact)


def by_gt_correctness_leading(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gt in REL:
        gg = samples.loc[samples["gt"].eq(gt)]
        for bc, cname in ((True, "baseline_correct"), (False, "baseline_wrong")):
            cg = gg.loc[gg["baseline_correct"].eq(bc)]
            Nc = len(cg)
            for status in ("gt_leading", "wrong_leading", "tie"):
                g = cg.loc[cg["spatial_leading_status"].eq(status)]
                n = len(g)
                rows.append({
                    "gt": gt,
                    "baseline_cohort": cname,
                    "spatial_leading_status": status,
                    "N": n,
                    "cohort_N": Nc,
                    "fraction_within_gt_and_correctness": float(n / Nc) if Nc else np.nan,
                    "mean_gt_minus_bestwrong": float(g["gt_minus_bestwrong_mean_evidence"].mean()) if n else np.nan,
                    "mean_gt_evidence": float(g["mean_gt_direction_evidence"].mean()) if n else np.nan,
                    "mean_best_wrong_evidence": float(g["mean_best_wrong_direction_evidence"].mean()) if n else np.nan,
                    "mean_gt_source_percentile": float(g["mean_gt_evidence_source_percentile"].mean()) if n else np.nan,
                    "mean_bestwrong_source_percentile": float(g["mean_bestwrong_evidence_source_percentile"].mean()) if n else np.nan,
                    "baseline_equals_bestwrong_rate": float(g["baseline_equals_bestwrong_mean_evidence"].mean()) if n else np.nan,
                })
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    out = Path(a.output_dir)
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"Output exists: {out}; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    Xs, ys, slayers, ssids, sdef = load_state_npz(Path(a.source_spatial_npz), True)
    Xt, yt, tlayers, tsids, tdef = load_state_npz(Path(a.target_spatial_npz), False)
    base = load_baseline(Path(a.baseline_csv))
    layers = a.layers_parsed

    if a.expected_n > 0 and len(base) != a.expected_n:
        raise RuntimeError(
            f"Expected {a.expected_n} unique baseline samples, found {len(base)}. "
            "For the requested all-440 diagnosis, use the full per_sample_candidate_repair.csv "
            "from the Synthetic400->COCO440 run, not the N80 baseline.csv."
        )
    if any(r not in REL for r in set(ys.tolist())):
        raise RuntimeError("Bad source labels")

    t_sid = {int(s): i for i, s in enumerate(tsids.tolist())}
    t_l = {L: i for i, L in enumerate(tlayers)}
    missing = [int(s) for s in base["sid"].tolist() if int(s) not in t_sid]
    if missing:
        raise RuntimeError(f"Target cache missing sids, e.g. {missing[:10]}")
    if any(L not in t_l for L in layers):
        raise RuntimeError(f"Target missing requested layers {layers}")

    if yt is not None:
        bad = []
        for r in base.itertuples(index=False):
            if norm_rel(yt[t_sid[int(r.sid)]]) != norm_rel(r.gt):
                bad.append(int(r.sid))
        if bad:
            raise RuntimeError(f"GT mismatch target NPZ/baseline e.g. {bad[:10]}")

    geom, geom_df = fit_hv_geometry(Xs, ys, slayers, layers)
    source_ref, source_stats = fit_source_reference(Xs, ys, slayers, geom, layers)
    geom_df.to_csv(out / "source_spatial_geometry.csv", index=False)
    source_stats.to_csv(out / "source_reference_statistics.csv", index=False)

    sample_rows = []
    layer_rows = []

    for br in base.itertuples(index=False):
        sid = int(br.sid)
        gt = norm_rel(br.gt)
        bp = norm_rel(br.baseline_prediction)
        bc = bool(br.baseline_correct)
        ti = t_sid[sid]

        ev_by_rel = {r: [] for r in REL}
        pct_by_rel = {r: [] for r in REL}
        layer_winners = []
        spatial_norms = []
        norm_pcts = []
        top_ev_pcts = []
        gt_ranks_layer = []

        for L in layers:
            z = read_coord(Xt[ti, t_l[L]], geom[L])
            ev = axis_evidence(z)
            winner = max(REL, key=lambda r: float(ev[r]))
            top_ev = float(ev[winner])
            znorm = float(np.linalg.norm(z))

            pcts = {
                r: empirical_percentile(source_ref[L]["own_evidence_sorted"][r], ev[r])
                for r in REL
            }
            norm_pct = empirical_percentile(source_ref[L]["norm_sorted"], znorm)
            top_pct = empirical_percentile(source_ref[L]["top_evidence_sorted"], top_ev)

            for r in REL:
                ev_by_rel[r].append(float(ev[r]))
                pct_by_rel[r].append(float(pcts[r]))
            layer_winners.append(winner)
            spatial_norms.append(znorm)
            norm_pcts.append(norm_pct)
            top_ev_pcts.append(top_pct)
            gt_ranks_layer.append(rank_desc(ev, gt))

            wrong_rel = max((r for r in REL if r != gt), key=lambda r: float(ev[r]))
            layer_rows.append({
                "sid": sid,
                "gt": gt,
                "baseline_prediction": bp,
                "baseline_correct": bc,
                "layer": L,
                "z_H": float(z[0]),
                "z_V": float(z[1]),
                "spatial_norm": znorm,
                "spatial_norm_source_percentile": norm_pct,
                "top_direction": winner,
                "top_direction_evidence": top_ev,
                "top_evidence_source_percentile": top_pct,
                "gt_rank": rank_desc(ev, gt),
                "gt_direction_evidence": float(ev[gt]),
                "gt_evidence_source_percentile": float(pcts[gt]),
                "best_wrong_direction": wrong_rel,
                "best_wrong_direction_evidence": float(ev[wrong_rel]),
                "best_wrong_evidence_source_percentile": float(pcts[wrong_rel]),
                "gt_minus_bestwrong_evidence": float(ev[gt] - ev[wrong_rel]),
                **{f"direction_evidence_{r}": float(ev[r]) for r in REL},
                **{f"evidence_source_percentile_{r}": float(pcts[r]) for r in REL},
            })

        mean_ev = {r: float(np.mean(ev_by_rel[r])) for r in REL}
        mean_pct = {r: float(np.mean(pct_by_rel[r])) for r in REL}
        best_wrong = max((r for r in REL if r != gt), key=lambda r: mean_ev[r])
        best_wrong_pct = max((r for r in REL if r != gt), key=lambda r: mean_pct[r])
        gt_ev = mean_ev[gt]
        wrong_ev = mean_ev[best_wrong]
        margin = float(gt_ev - wrong_ev)
        denom = abs(gt_ev) + abs(wrong_ev) + EPS
        normalized_margin = float(margin / denom)
        gt_rank = rank_desc(mean_ev, gt)
        spatial_leading_status = (
            "gt_leading" if margin > 0 else
            "wrong_leading" if margin < 0 else
            "tie"
        )

        cnt = Counter(layer_winners)
        mode, mode_n = cnt.most_common(1)[0]
        mode_frac = float(mode_n / len(layers))
        cross_unstable = bool(mode_frac < a.unstable_winner_fraction)

        top1_rel = max(REL, key=lambda r: mean_ev[r])
        top2_rel = sorted(REL, key=lambda r: mean_ev[r], reverse=True)[1]
        top1_gap = float(mean_ev[top1_rel] - mean_ev[top2_rel])

        sample_rows.append({
            "sid": sid,
            "gt": gt,
            "baseline_prediction": bp,
            "baseline_correct": bc,
            "N_layers": len(layers),
            **{f"mean_direction_evidence_{r}": mean_ev[r] for r in REL},
            **{f"mean_evidence_source_percentile_{r}": mean_pct[r] for r in REL},
            "mean_gt_direction_evidence": gt_ev,
            "mean_gt_evidence_source_percentile": mean_pct[gt],
            "best_wrong_direction_by_mean_evidence": best_wrong,
            "mean_best_wrong_direction_evidence": wrong_ev,
            "mean_bestwrong_evidence_source_percentile": mean_pct[best_wrong],
            "best_wrong_direction_by_source_percentile": best_wrong_pct,
            "mean_bestwrong_relation_percentile": mean_pct[best_wrong_pct],
            "gt_minus_bestwrong_mean_evidence": margin,
            "spatial_leading_status": spatial_leading_status,
            "normalized_gt_vs_bestwrong_margin": normalized_margin,
            "abs_gt_vs_bestwrong_margin": abs(margin),
            "gt_rank_by_mean_evidence": gt_rank,
            "gt_is_top1_by_mean_evidence": bool(gt_rank == 1),
            "gt_is_top2_by_mean_evidence": bool(gt_rank <= 2),
            "top1_direction_by_mean_evidence": top1_rel,
            "top2_direction_by_mean_evidence": top2_rel,
            "top1_minus_top2_mean_evidence": top1_gap,
            "baseline_equals_top1_direction": bool(bp == top1_rel),
            "baseline_equals_bestwrong_mean_evidence": bool((not bc) and bp == best_wrong),
            "baseline_equals_bestwrong_percentile": bool((not bc) and bp == best_wrong_pct),
            "baseline_direction_mean_evidence": mean_ev[bp],
            "baseline_direction_mean_source_percentile": mean_pct[bp],
            "gt_top1_layer_fraction": float(np.mean([r == gt for r in layer_winners])),
            "baseline_top1_layer_fraction": float(np.mean([r == bp for r in layer_winners])),
            "modal_layerwise_top_direction": mode,
            "modal_layerwise_top_direction_fraction": mode_frac,
            "distinct_layerwise_top_directions": len(cnt),
            "layerwise_top_direction_transitions": int(sum(
                layer_winners[i] != layer_winners[i - 1] for i in range(1, len(layer_winners))
            )),
            "cross_layer_unstable": cross_unstable,
            "mean_layerwise_gt_rank": float(np.mean(gt_ranks_layer)),
            "mean_spatial_norm": float(np.mean(spatial_norms)),
            "mean_spatial_norm_source_percentile": float(np.mean(norm_pcts)),
            "mean_top_evidence_source_percentile": float(np.mean(top_ev_pcts)),
        })

    samples = pd.DataFrame(sample_rows).sort_values("sid").reset_index(drop=True)
    layers_df = pd.DataFrame(layer_rows).sort_values(["sid", "layer"]).reset_index(drop=True)
    errors = samples.loc[~samples["baseline_correct"]].copy()
    correct = samples.loc[samples["baseline_correct"]].copy()

    samples.to_csv(out / "per_sample_continuous_spatial.csv", index=False)
    layers_df.to_csv(out / "per_sample_layer_continuous_spatial.csv", index=False)

    # Explicit all-sample correct/wrong x spatial-leading decomposition.
    quad_df, compact_leading_df = correctness_leading_tables(samples)
    quad_df.to_csv(out / "correctness_x_spatial_leading.csv", index=False)
    compact_leading_df.to_csv(out / "spatial_leading_by_correctness.csv", index=False)

    by_gt_quad_df = by_gt_correctness_leading(samples)
    by_gt_quad_df.to_csv(out / "by_gt_x_correctness_x_spatial_leading.csv", index=False)

    # Save each main quadrant as a sample-level CSV for inspection.
    for bc, cname in ((True, "baseline_correct"), (False, "baseline_wrong")):
        for status in ("gt_leading", "wrong_leading", "tie"):
            q = samples.loc[
                samples["baseline_correct"].eq(bc)
                & samples["spatial_leading_status"].eq(status)
            ].copy()
            if status == "gt_leading":
                q = q.sort_values(
                    ["gt_minus_bestwrong_mean_evidence", "mean_gt_evidence_source_percentile"],
                    ascending=[False, False],
                )
            elif status == "wrong_leading":
                q = q.sort_values(
                    ["gt_minus_bestwrong_mean_evidence", "mean_bestwrong_evidence_source_percentile"],
                    ascending=[True, False],
                )
            q.to_csv(out / f"{cname}_{status}.csv", index=False)

    # Threshold-free error views retained for compatibility.
    gt_leading = errors.loc[errors["gt_minus_bestwrong_mean_evidence"] > 0].copy()
    gt_leading = gt_leading.sort_values(
        ["gt_minus_bestwrong_mean_evidence", "mean_gt_evidence_source_percentile"],
        ascending=[False, False],
    )
    wrong_leading = errors.loc[errors["gt_minus_bestwrong_mean_evidence"] < 0].copy()
    wrong_leading = wrong_leading.sort_values(
        ["gt_minus_bestwrong_mean_evidence", "mean_bestwrong_evidence_source_percentile"],
        ascending=[True, False],
    )
    ambiguous = errors.assign(
        abs_margin_sort=errors["gt_minus_bestwrong_mean_evidence"].abs()
    ).sort_values(["abs_margin_sort", "mean_spatial_norm_source_percentile"], ascending=[True, False])
    ambiguous = ambiguous.drop(columns="abs_margin_sort")

    gt_leading.to_csv(out / "baseline_wrong_gt_leading.csv", index=False)
    wrong_leading.to_csv(out / "baseline_wrong_wrong_leading.csv", index=False)
    ambiguous.to_csv(out / "baseline_wrong_most_ambiguous.csv", index=False)

    cohort_rows = [
        summarize_cohort(samples, "all"),
        summarize_cohort(errors, "baseline_wrong"),
        summarize_cohort(correct, "baseline_correct"),
    ]
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(out / "cohort_summary.csv", index=False)

    # GT rank distribution.
    rank_rows = []
    for cname, g in [("all", samples), ("baseline_wrong", errors), ("baseline_correct", correct)]:
        N = len(g)
        for rank in (1, 2, 3, 4):
            n = int((g["gt_rank_by_mean_evidence"] == rank).sum())
            rank_rows.append({
                "cohort": cname,
                "gt_rank": rank,
                "N": n,
                "fraction": float(n / N) if N else np.nan,
            })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out / "gt_rank_distribution.csv", index=False)

    # Margin quantiles: no thresholding, only distribution description.
    margin_rows = []
    qs = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
    for cname, g in [("all", samples), ("baseline_wrong", errors), ("baseline_correct", correct)]:
        vals = g["gt_minus_bestwrong_mean_evidence"].to_numpy(dtype=float)
        for q in qs:
            margin_rows.append({
                "cohort": cname,
                "quantile": q,
                "gt_minus_bestwrong_mean_evidence": float(np.quantile(vals, q)) if len(vals) else np.nan,
            })
    margin_df = pd.DataFrame(margin_rows)
    margin_df.to_csv(out / "margin_quantiles.csv", index=False)

    by_gt = grouped_summary(samples, "gt")
    by_gt.to_csv(out / "by_gt_summary.csv", index=False)
    by_bp = grouped_summary(samples, "baseline_prediction")
    by_bp.to_csv(out / "by_baseline_prediction_summary.csv", index=False)

    N = len(samples)
    Nw = len(errors)
    acc = float(samples["baseline_correct"].mean())
    wrong_gt_lead_n = int((errors["gt_minus_bestwrong_mean_evidence"] > 0).sum())
    wrong_wrong_lead_n = int((errors["gt_minus_bestwrong_mean_evidence"] < 0).sum())
    wrong_tie_n = int((errors["gt_minus_bestwrong_mean_evidence"] == 0).sum())
    wrong_bp_best_n = int(errors["baseline_equals_bestwrong_mean_evidence"].sum())
    Nc = len(correct)
    correct_gt_lead_n = int(correct["spatial_leading_status"].eq("gt_leading").sum())
    correct_wrong_lead_n = int(correct["spatial_leading_status"].eq("wrong_leading").sum())
    correct_tie_n = int(correct["spatial_leading_status"].eq("tie").sum())

    report = [
        "=" * 200,
        "ALL-SAMPLE CONTINUOUS SPATIAL GEOMETRY — THRESHOLD-FREE PRIMARY DIAGNOSTIC",
        "=" * 200,
        f"source={a.source_spatial_npz} N={len(Xs)} definition={sdef}",
        f"target={a.target_spatial_npz} cacheN={len(Xt)} definition={tdef}",
        f"baseline={a.baseline_csv}",
        f"eval N={N} baseline_accuracy={acc:.4f} wrong={Nw}",
        f"layers={layers}",
        "",
        "PRIMARY DEFINITIONS",
        "-" * 200,
        "G = mean directional evidence for GT across layers",
        "W = strongest mean directional evidence among the three wrong relations",
        "M = G - W",
        "M > 0: GT spatial direction is strongest; M < 0: a wrong spatial direction is stronger.",
        "Absolute strength is reported separately by Synthetic-source empirical percentiles.",
        "",
        "COHORT SUMMARY",
        "-" * 200,
        cohort_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "SPATIAL LEADING STATUS BY BASELINE CORRECTNESS",
        "-" * 200,
        f"baseline_correct: GT-leading={correct_gt_lead_n}/{Nc}={correct_gt_lead_n/max(Nc,1):.4f} | wrong-leading={correct_wrong_lead_n}/{Nc}={correct_wrong_lead_n/max(Nc,1):.4f} | tie={correct_tie_n}/{Nc}={correct_tie_n/max(Nc,1):.4f}",
        f"baseline_wrong  : GT-leading={wrong_gt_lead_n}/{Nw}={wrong_gt_lead_n/max(Nw,1):.4f} | wrong-leading={wrong_wrong_lead_n}/{Nw}={wrong_wrong_lead_n/max(Nw,1):.4f} | tie={wrong_tie_n}/{Nw}={wrong_tie_n/max(Nw,1):.4f}",
        f"wrong samples: baseline wrong answer == strongest wrong dir: {wrong_bp_best_n}/{Nw} = {wrong_bp_best_n/max(Nw,1):.4f}",
        "",
        "CORRECT/WRONG x GT-LEADING/WRONG-LEADING QUADRANTS",
        "-" * 200,
        quad_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "BY GT x BASELINE CORRECTNESS x SPATIAL LEADING STATUS",
        "-" * 200,
        by_gt_quad_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "GT RANK DISTRIBUTION",
        "-" * 200,
        rank_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "MARGIN QUANTILES: M = GT - BEST WRONG",
        "-" * 200,
        margin_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "BY GT",
        "-" * 200,
        by_gt.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "BY BASELINE PREDICTION",
        "-" * 200,
        by_bp.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
    ]
    txt = "\n".join(report) + "\n"
    print(txt)
    (out / "analysis_summary.txt").write_text(txt, encoding="utf-8")

    metadata = {
        "script_version": SCRIPT_VERSION,
        "source_spatial_npz": a.source_spatial_npz,
        "target_spatial_npz": a.target_spatial_npz,
        "baseline_csv": a.baseline_csv,
        "layers": layers,
        "source_N": int(len(Xs)),
        "eval_N": int(N),
        "baseline_wrong_N": int(Nw),
        "baseline_correct_N": int(Nc),
        "baseline_correct_gt_leading_N": int(correct_gt_lead_n),
        "baseline_correct_wrong_leading_N": int(correct_wrong_lead_n),
        "baseline_wrong_gt_leading_N": int(wrong_gt_lead_n),
        "baseline_wrong_wrong_leading_N": int(wrong_wrong_lead_n),
        "baseline_accuracy": acc,
        "expected_N": int(a.expected_n),
        "source_labels_used_for_HV_geometry_and_reference_percentiles": True,
        "target_GT_used_only_for_posthoc_diagnosis": True,
        "target_labels_used_to_fit_geometry": False,
        "primary_diagnostic_uses_COCO_tuned_threshold": False,
        "model_loaded": False,
        "intervention_performed": False,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
