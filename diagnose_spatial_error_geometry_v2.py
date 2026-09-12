#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_spatial_error_geometry_v2.py

Pure-NPZ decomposition of baseline spatial errors into:
  1) GT-dominant      : GT directional evidence is strong; wrong directions are not.
  2) GT+wrong mixed   : GT and >=1 wrong direction are both strong.
  3) wrong-dominant   : GT is weak; >=1 wrong direction is strong.
  4) weak             : all four directional evidences are weak.

Cross-layer instability and distance-to-source-class-center are reported as
independent diagnostics.

Why directional evidence is primary
===================================
The controlled spatial subspace is 2-D.  For z_L=(z_H,z_V):

    e_left  = -z_H
    e_right = +z_H
    e_above = +z_V
    e_below = -z_V

A diagonal state such as z=(+1,-1) should be diagnosed as simultaneous RIGHT
and BELOW evidence, not as "weak" merely because it lies between two class
centers.  Therefore the four-way error category is based on source-calibrated
DIRECTIONAL evidence.  Mahalanobis distance to the four Synthetic class centers
is retained as a secondary manifold / ambiguity diagnostic.

Strong-direction calibration
============================
For every layer and relation r, compute the source own-class evidence

    e_r(z_i),  i: y_i=r.

A target direction is "strong" at that layer if its evidence exceeds a lower
quantile of the Synthetic own-class distribution (default q=0.25).  Thus the
strength threshold is source-only rather than hand-tuned on COCO.

Multi-layer support is the fraction of L20-26 for which that direction is
strong.  By default a relation is called supported if >=50% of layers support it.

Center-distance diagnostic
==========================
Also fit four source class centers in the same 2-D H/V coordinate system and a
pooled within-class covariance.  Report Mahalanobis distance and whether the
sample falls inside the source 95% own-class envelope.  This separates:

  * directional mixed state: e.g. both GT and orthogonal wrong are strong;
  * off-manifold state: far from all four source class clouds.

No model is loaded and no intervention is performed.  COCO GT is used only to
label post-hoc diagnostic categories.

Recommended current N80 run
===========================
python -u diagnose_spatial_error_geometry_v2.py \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --baseline-csv \
    output/qwen3b_direct_self_spatial_amp_n80_v1/baseline.csv \
  --layers 20-26 \
  --evidence-lower-quantiles 0.10,0.25,0.50 \
  --primary-evidence-lower-quantile 0.25 \
  --support-layer-fraction 0.50 \
  --center-distance-quantile 0.95 \
  --output-dir output/qwen3b_spatial_error_geometry_n80_v2 \
  --overwrite

Main outputs
============
per_error_spatial_geometry.csv
    Main per-error table.  Inspect this first.
per_error_layer_geometry.csv
    Error sample x layer details.
error_category_summary.csv
    Four-way decomposition.
error_category_by_gt.csv
error_category_by_baseline_prediction.csv
support_sensitivity.csv
    Sensitivity to source-only evidence threshold.
gt_dominant_utilization_failures.csv
mixed_spatial_errors.csv
wrong_dominant_representation_errors.csv
weak_spatial_errors.csv
source_direction_evidence_thresholds.csv
source_center_distance_thresholds.csv
analysis_summary.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REL = ("left", "right", "above", "below")
EPS = 1e-10
SCRIPT_VERSION = "v2_direction_evidence_primary_plus_center_distance"


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


def parse_float_list(text: str) -> List[float]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        q = float(x)
        if not (0 < q < 1):
            raise ValueError("quantiles must lie in (0,1)")
        if q not in out:
            out.append(q)
    if not out:
        raise ValueError("No quantiles parsed")
    return sorted(out)


def qtag(q: float) -> str:
    return f"q{int(round(100*q)):02d}"


def axis_evidence(z) -> Dict[str, float]:
    h, v = float(z[0]), float(z[1])
    return {"left": -h, "right": h, "above": v, "below": -v}


def gt_axis_components(gt: str, z: np.ndarray):
    h, v = float(z[0]), float(z[1])
    if gt == "left":
        return -h, abs(v)
    if gt == "right":
        return h, abs(v)
    if gt == "above":
        return v, abs(h)
    if gt == "below":
        return -v, abs(h)
    raise ValueError(gt)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--layers", default="20-26")
    p.add_argument("--evidence-lower-quantiles", default="0.10,0.25,0.50")
    p.add_argument("--primary-evidence-lower-quantile", type=float, default=0.25)
    p.add_argument("--support-layer-fraction", type=float, default=0.50)
    p.add_argument("--center-distance-quantile", type=float, default=0.95)
    p.add_argument("--unstable-winner-fraction", type=float, default=0.60)
    p.add_argument("--cov-shrinkage", type=float, default=0.10)
    p.add_argument("--cov-ridge", type=float, default=1e-4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    try:
        a.layers_parsed = parse_layers(a.layers)
        a.evidence_quantiles = parse_float_list(a.evidence_lower_quantiles)
    except ValueError as exc:
        p.error(str(exc))
    if all(abs(a.primary_evidence_lower_quantile - q) > 1e-9 for q in a.evidence_quantiles):
        a.evidence_quantiles.append(float(a.primary_evidence_lower_quantile))
        a.evidence_quantiles = sorted(set(a.evidence_quantiles))
    if not (0 < a.support_layer_fraction <= 1):
        p.error("--support-layer-fraction must be in (0,1]")
    if not (0 < a.center_distance_quantile < 1):
        p.error("--center-distance-quantile must be in (0,1)")
    if not (0 < a.unstable_winner_fraction <= 1):
        p.error("--unstable-winner-fraction must be in (0,1]")
    if not (0 <= a.cov_shrinkage <= 1):
        p.error("--cov-shrinkage must be in [0,1]")
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
        sids = np.asarray(z["sample_index"], dtype=np.int64) if "sample_index" in keys else np.arange(len(X))
        labels = None
        if "relation" in keys:
            labels = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels")
    if X.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {X.shape}")
    return X, labels, layers, sids, definition


def safe_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().str.strip().isin(["true", "1", "t", "yes", "y"])


def load_baseline(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    need = {"sid", "gt", "baseline_prediction"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"baseline csv missing {sorted(miss)}")
    df = df.copy()
    df["sid"] = df["sid"].astype(int)
    df["gt"] = df["gt"].map(norm_rel)
    df["baseline_prediction"] = df["baseline_prediction"].map(norm_rel)
    if "baseline_correct" in df:
        df["baseline_correct"] = safe_bool_series(df["baseline_correct"])
    else:
        df["baseline_correct"] = df["gt"] == df["baseline_prediction"]
    if df["sid"].duplicated().any():
        raise RuntimeError("baseline csv contains duplicate sid")
    return df


def fit_hv_geometry(X, y, layers, wanted_layers):
    lmap = {L: i for i, L in enumerate(layers)}
    geom, rows = {}, []
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
        B = np.stack([dH, dV], axis=1)
        dual = B @ np.linalg.inv(B.T @ B)
        geom[L] = {
            "center": center,
            "dual": dual,
            "halfH": max(gapH / 2, EPS),
            "halfV": max(gapV / 2, EPS),
            "layer_index": lmap[L],
        }
        rows.append({
            "layer": L,
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "full_gap_H": gapH,
            "full_gap_V": gapV,
            "half_gap_H": max(gapH/2, EPS),
            "half_gap_V": max(gapV/2, EPS),
        })
    return geom, pd.DataFrame(rows)


def read_coord(x, g):
    res = np.asarray(x, dtype=np.float64) - g["center"]
    c = g["dual"].T @ res
    return np.asarray([c[0] / g["halfH"], c[1] / g["halfV"]], dtype=np.float64)


def fit_source_calibration(X, y, source_layers, geom, layers, evidence_qs, center_q, shrink, ridge):
    lmap = {L: i for i, L in enumerate(source_layers)}
    model = {}
    ev_rows, dist_rows = [], []
    for L in layers:
        Z = np.stack([read_coord(X[i, lmap[L]], geom[L]) for i in range(len(X))])
        mu = {r: Z[y == r].mean(axis=0) for r in REL}

        # pooled within-class covariance in 2D
        residuals = np.stack([Z[i] - mu[str(y[i])] for i in range(len(Z))])
        cov = np.cov(residuals.T, bias=False)
        diag = np.diag(np.diag(cov))
        cov = (1-shrink)*cov + shrink*diag
        scale = max(float(np.trace(cov)/2), EPS)
        cov = cov + np.eye(2)*(ridge*scale + EPS)
        inv_cov = np.linalg.inv(cov)

        evidence_thresholds = {}
        center_thresholds = {}
        for r in REL:
            own_idx = np.where(y == r)[0]
            own_ev = np.asarray([axis_evidence(Z[i])[r] for i in own_idx], dtype=np.float64)
            evidence_thresholds[r] = {q: float(np.quantile(own_ev, q)) for q in evidence_qs}
            ev_row = {
                "layer": L, "relation": r, "N": len(own_ev),
                "own_evidence_mean": float(own_ev.mean()),
                "own_evidence_median": float(np.median(own_ev)),
            }
            for q in evidence_qs:
                ev_row[f"lower_threshold_{qtag(q)}"] = evidence_thresholds[r][q]
            ev_rows.append(ev_row)

            own_d2 = []
            for i in own_idx:
                d = Z[i] - mu[r]
                own_d2.append(float(d @ inv_cov @ d))
            own_d2 = np.asarray(own_d2)
            center_thresholds[r] = float(np.quantile(own_d2, center_q))
            dist_rows.append({
                "layer": L, "relation": r, "N": len(own_d2),
                "own_d2_mean": float(own_d2.mean()),
                "own_d2_median": float(np.median(own_d2)),
                "center_distance_quantile": center_q,
                "center_d2_threshold": center_thresholds[r],
            })

        model[L] = {
            "mu": mu,
            "inv_cov": inv_cov,
            "evidence_thresholds": evidence_thresholds,
            "center_thresholds": center_thresholds,
        }
    return model, pd.DataFrame(ev_rows), pd.DataFrame(dist_rows)


def classify(gt, supported):
    gt_ok = supported[gt]
    wrong = [r for r in REL if r != gt and supported[r]]
    if gt_ok and not wrong:
        return "gt_dominant"
    if gt_ok and wrong:
        return "mixed_gt_wrong"
    if (not gt_ok) and wrong:
        return "wrong_dominant"
    return "weak"


def summarize_errors(df):
    if len(df) == 0:
        return pd.DataFrame()
    rows=[]; N=len(df)
    for cat,g in df.groupby("primary_category", sort=False):
        rows.append({
            "primary_category": cat,
            "N": len(g),
            "fraction_of_baseline_wrong": len(g)/N,
            "baseline_answer_direction_supported_rate": float(g["baseline_answer_direction_supported"].mean()),
            "baseline_equals_best_wrong_direction_rate": float(g["baseline_equals_best_wrong_direction"].mean()),
            "center_off_manifold_rate": float(g["center_off_manifold"].mean()),
            "cross_layer_unstable_rate": float(g["cross_layer_unstable"].mean()),
            "mean_gt_direction_support_fraction": float(g["gt_direction_support_layer_fraction"].mean()),
            "mean_best_wrong_direction_support_fraction": float(g["best_wrong_direction_support_layer_fraction"].mean()),
            "mean_gt_evidence": float(g["mean_gt_direction_evidence"].mean()),
            "mean_best_wrong_evidence": float(g["mean_best_wrong_direction_evidence"].mean()),
            "mean_abs_orthogonal_over_gt_ratio": float(g["mean_abs_orthogonal_over_gt_ratio"].replace([np.inf], np.nan).mean()),
        })
    return pd.DataFrame(rows).sort_values("N", ascending=False)


def grouped(df, col):
    if len(df)==0: return pd.DataFrame()
    rows=[]
    for key,g in df.groupby(col, sort=False):
        c=g["primary_category"].value_counts(); N=len(g)
        row={col:key,"N_wrong":N}
        for cat in ["gt_dominant","mixed_gt_wrong","wrong_dominant","weak"]:
            row[f"N_{cat}"]=int(c.get(cat,0)); row[f"frac_{cat}"]=float(c.get(cat,0)/N)
        row["baseline_answer_direction_supported_rate"]=float(g["baseline_answer_direction_supported"].mean())
        row["center_off_manifold_rate"]=float(g["center_off_manifold"].mean())
        row["cross_layer_unstable_rate"]=float(g["cross_layer_unstable"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a=parse_args(); out=Path(a.output_dir)
    if out.exists():
        if not a.overwrite: raise SystemExit(f"Output exists: {out}; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    Xs,ys,slayers,ssids,sdef=load_state_npz(Path(a.source_spatial_npz),True)
    Xt,yt,tlayers,tsids,tdef=load_state_npz(Path(a.target_spatial_npz),False)
    base=load_baseline(Path(a.baseline_csv))
    layers=a.layers_parsed; qs=a.evidence_quantiles; q0=float(a.primary_evidence_lower_quantile); q0t=qtag(q0)
    if any(r not in REL for r in set(ys.tolist())): raise RuntimeError("Bad source labels")
    t_sid={int(s):i for i,s in enumerate(tsids.tolist())}; t_l={L:i for i,L in enumerate(tlayers)}
    miss=[s for s in base.sid.tolist() if int(s) not in t_sid]
    if miss: raise RuntimeError(f"Target cache missing sids e.g. {miss[:10]}")
    if any(L not in t_l for L in layers): raise RuntimeError("Target missing requested layers")
    if yt is not None:
        bad=[]
        for r in base.itertuples(index=False):
            if norm_rel(yt[t_sid[int(r.sid)]]) != norm_rel(r.gt): bad.append(int(r.sid))
        if bad: raise RuntimeError(f"GT mismatch target NPZ/baseline e.g. {bad[:10]}")

    geom,geom_df=fit_hv_geometry(Xs,ys,slayers,layers)
    cal,ev_thr_df,dist_thr_df=fit_source_calibration(
        Xs,ys,slayers,geom,layers,qs,a.center_distance_quantile,a.cov_shrinkage,a.cov_ridge
    )
    geom_df.to_csv(out/"source_spatial_geometry.csv",index=False)
    ev_thr_df.to_csv(out/"source_direction_evidence_thresholds.csv",index=False)
    dist_thr_df.to_csv(out/"source_center_distance_thresholds.csv",index=False)

    sample_rows=[]; error_layer_rows=[]; sens_long=[]
    for br in base.itertuples(index=False):
        sid=int(br.sid); gt=norm_rel(br.gt); bp=norm_rel(br.baseline_prediction); bc=bool(br.baseline_correct); ti=t_sid[sid]
        hit={q:{r:0 for r in REL} for q in qs}; evs={r:[] for r in REL}; center_hit={r:0 for r in REL}; d2s={r:[] for r in REL}
        nearest=[]; maxev=[]; gt_main=[]; gt_orth=[]; layer_local=[]
        for L in layers:
            z=read_coord(Xt[ti,t_l[L]],geom[L]); evidence=axis_evidence(z); dm=cal[L]
            d2={}
            for r in REL:
                d=z-dm["mu"][r]; d2[r]=float(d@dm["inv_cov"]@d); d2s[r].append(d2[r]); evs[r].append(evidence[r])
                center_hit[r]+=int(d2[r] <= dm["center_thresholds"][r])
                for q in qs: hit[q][r]+=int(evidence[r] >= dm["evidence_thresholds"][r][q])
            nr=min(REL,key=lambda r:d2[r]); er=max(REL,key=lambda r:evidence[r]); nearest.append(nr); maxev.append(er)
            gm,go=gt_axis_components(gt,z); gt_main.append(gm); gt_orth.append(go)
            row={
                "sid":sid,"gt":gt,"baseline_prediction":bp,"baseline_correct":bc,"layer":L,"z_H":z[0],"z_V":z[1],
                "max_direction_evidence_relation":er,"nearest_center_relation":nr,"gt_axis_evidence":gm,"abs_orthogonal_evidence":go,
                **{f"direction_evidence_{r}":evidence[r] for r in REL},
                **{f"mahalanobis_d2_{r}":d2[r] for r in REL},
                **{f"center_supported_{r}":bool(d2[r] <= dm["center_thresholds"][r]) for r in REL},
            }
            for q in qs:
                qt=qtag(q)
                for r in REL:
                    thr=dm["evidence_thresholds"][r][q]
                    row[f"direction_strong_{qt}_{r}"]=bool(evidence[r]>=thr)
                    row[f"direction_evidence_over_threshold_{qt}_{r}"]=float(evidence[r]/max(abs(thr),EPS)) if thr>0 else np.nan
            layer_local.append(row)

        # Cross-layer evidence winner stability.
        cnt=Counter(maxev); mode,modeN=cnt.most_common(1)[0]; modefrac=modeN/len(layers)
        unstable=bool(modefrac < a.unstable_winner_fraction)
        center_fr={r:center_hit[r]/len(layers) for r in REL}
        center_any=max(center_fr.values()) >= a.support_layer_fraction

        common={
            "sid":sid,"gt":gt,"baseline_prediction":bp,"baseline_correct":bc,"N_layers":len(layers),
            "max_evidence_relation_mode":mode,"max_evidence_relation_mode_fraction":modefrac,
            "max_evidence_relation_distinct_count":len(cnt),
            "max_evidence_relation_transition_count":sum(maxev[i]!=maxev[i-1] for i in range(1,len(maxev))),
            "cross_layer_unstable":unstable,
            "gt_max_evidence_layer_fraction":float(np.mean([r==gt for r in maxev])),
            "baseline_max_evidence_layer_fraction":float(np.mean([r==bp for r in maxev])),
            "center_off_manifold":bool(not center_any),
            "center_gt_support_layer_fraction":center_fr[gt],
            "center_best_relation":max(REL,key=lambda r:center_fr[r]),
            **{f"center_support_layer_fraction_{r}":center_fr[r] for r in REL},
            **{f"mean_direction_evidence_{r}":float(np.mean(evs[r])) for r in REL},
            **{f"median_direction_evidence_{r}":float(np.median(evs[r])) for r in REL},
            **{f"mean_mahalanobis_d2_{r}":float(np.mean(d2s[r])) for r in REL},
            "mean_gt_direction_evidence":float(np.mean(gt_main)),
            "mean_abs_orthogonal_evidence":float(np.mean(gt_orth)),
            "mean_abs_orthogonal_over_gt_ratio":float(np.mean(gt_orth)/(max(abs(float(np.mean(gt_main))),EPS))),
        }

        cats={}
        for q in qs:
            qt=qtag(q); fr={r:hit[q][r]/len(layers) for r in REL}; sup={r:fr[r]>=a.support_layer_fraction for r in REL}
            cat=classify(gt,sup); cats[q]=cat
            # strongest wrong by mean normalized directional evidence against source threshold
            # Use per-layer evidence/threshold ratios, not discrete class argmax.
            ratios={}
            for r in REL:
                vals=[]
                for L,row in zip(layers,layer_local):
                    thr=cal[L]["evidence_thresholds"][r][q]
                    vals.append(row[f"direction_evidence_{r}"]/max(abs(thr),EPS) if thr>0 else -np.inf)
                ratios[r]=float(np.mean(vals))
            bw=max((r for r in REL if r!=gt), key=lambda r:ratios[r])
            common.update({
                f"category_{qt}":cat,
                f"supported_relations_{qt}":"|".join([r for r in REL if sup[r]]),
                f"n_supported_relations_{qt}":int(sum(sup.values())),
                f"gt_direction_supported_{qt}":bool(sup[gt]),
                f"baseline_direction_supported_{qt}":bool(sup.get(bp,False)),
                f"best_wrong_direction_{qt}":bw,
                f"gt_direction_support_fraction_{qt}":fr[gt],
                f"best_wrong_direction_support_fraction_{qt}":fr[bw],
                **{f"direction_support_fraction_{qt}_{r}":fr[r] for r in REL},
                **{f"mean_evidence_over_threshold_{qt}_{r}":ratios[r] for r in REL},
            })
            if not bc:
                sens_long.append({"sid":sid,"support_quantile":q,"category":cat,"baseline_answer_supported":bool(sup.get(bp,False)),"cross_layer_unstable":unstable})

        cat=cats[q0]; bw=common[f"best_wrong_direction_{q0t}"]
        common.update({
            "primary_evidence_lower_quantile":q0,
            "primary_category":cat,
            "supported_relations":common[f"supported_relations_{q0t}"],
            "gt_direction_supported":bool(common[f"gt_direction_supported_{q0t}"]),
            "baseline_answer_direction_supported":bool(common[f"baseline_direction_supported_{q0t}"]),
            "best_wrong_direction":bw,
            "baseline_equals_best_wrong_direction":bool(bp==bw),
            "gt_direction_support_layer_fraction":float(common[f"gt_direction_support_fraction_{q0t}"]),
            "best_wrong_direction_support_layer_fraction":float(common[f"best_wrong_direction_support_fraction_{q0t}"]),
            "mean_best_wrong_direction_evidence":float(np.mean(evs[bw])),
            "gt_minus_bestwrong_mean_evidence":float(np.mean(evs[gt])-np.mean(evs[bw])),
            "baseline_mean_direction_evidence":float(np.mean(evs[bp])) if bp in REL else np.nan,
        })
        if cat=="mixed_gt_wrong":
            common["mixed_subtype"]="mixed_gt_stronger" if common["gt_minus_bestwrong_mean_evidence"]>=0 else "mixed_wrong_stronger"
        else: common["mixed_subtype"]=""
        sample_rows.append(common)
        if not bc: error_layer_rows.extend(layer_local)

    samples=pd.DataFrame(sample_rows); errors=samples.loc[~samples.baseline_correct.astype(bool)].copy(); error_layers=pd.DataFrame(error_layer_rows)
    order={"gt_dominant":0,"mixed_gt_wrong":1,"wrong_dominant":2,"weak":3}
    if len(errors):
        errors["_o"]=errors.primary_category.map(order).fillna(9)
        errors=errors.sort_values(["_o","gt_minus_bestwrong_mean_evidence","sid"],ascending=[True,False,True]).drop(columns="_o")
    summary=summarize_errors(errors); bygt=grouped(errors,"gt"); bybp=grouped(errors,"baseline_prediction")

    sens=pd.DataFrame(sens_long); srows=[]
    if len(sens):
        for q,g in sens.groupby("support_quantile"):
            c=g.category.value_counts(); N=len(g); row={"evidence_lower_quantile":q,"N_baseline_wrong":N,"baseline_answer_direction_supported_rate":float(g.baseline_answer_supported.mean()),"cross_layer_unstable_rate":float(g.cross_layer_unstable.mean())}
            for cat in ["gt_dominant","mixed_gt_wrong","wrong_dominant","weak"]:
                row[f"N_{cat}"]=int(c.get(cat,0)); row[f"frac_{cat}"]=float(c.get(cat,0)/N)
            srows.append(row)
    senssum=pd.DataFrame(srows)

    samples.to_csv(out/"per_sample_spatial_geometry.csv",index=False)
    errors.to_csv(out/"per_error_spatial_geometry.csv",index=False)
    error_layers.to_csv(out/"per_error_layer_geometry.csv",index=False)
    summary.to_csv(out/"error_category_summary.csv",index=False)
    bygt.to_csv(out/"error_category_by_gt.csv",index=False)
    bybp.to_csv(out/"error_category_by_baseline_prediction.csv",index=False)
    senssum.to_csv(out/"support_sensitivity.csv",index=False)
    if len(errors):
        errors[errors.primary_category=="gt_dominant"].to_csv(out/"gt_dominant_utilization_failures.csv",index=False)
        errors[errors.primary_category=="mixed_gt_wrong"].to_csv(out/"mixed_spatial_errors.csv",index=False)
        errors[errors.primary_category=="wrong_dominant"].to_csv(out/"wrong_dominant_representation_errors.csv",index=False)
        errors[errors.primary_category=="weak"].to_csv(out/"weak_spatial_errors.csv",index=False)

    acc=float(samples.baseline_correct.mean()); Nw=len(errors)
    report=["="*190,"SPATIAL ERROR GEOMETRY DECOMPOSITION — DIRECTIONAL EVIDENCE PRIMARY","="*190,
            f"source={a.source_spatial_npz} N={len(Xs)} definition={sdef}",f"target={a.target_spatial_npz} cacheN={len(Xt)} definition={tdef}",
            f"baseline={a.baseline_csv} N={len(samples)} accuracy={acc:.4f} wrong={Nw}",f"layers={layers}",
            f"primary source own-evidence lower quantile={q0:.2f}",f"multi-layer support fraction threshold={a.support_layer_fraction:.2f}",
            f"center-distance source quantile={a.center_distance_quantile:.2f}","",
            "BASELINE-WRONG PRIMARY DECOMPOSITION","-"*190,summary.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(summary) else "EMPTY","",
            "EVIDENCE-THRESHOLD SENSITIVITY","-"*190,senssum.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(senssum) else "EMPTY","",
            "BY GT","-"*190,bygt.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(bygt) else "EMPTY","",
            "BY BASELINE WRONG PREDICTION","-"*190,bybp.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(bybp) else "EMPTY"]
    if Nw:
        report += ["","HIGH-VALUE COUNTS","-"*190,
                   f"GT-dominant utilization failures        : {int((errors.primary_category=='gt_dominant').sum())}",
                   f"GT+wrong mixed spatial errors            : {int((errors.primary_category=='mixed_gt_wrong').sum())}",
                   f"wrong-dominant representation errors     : {int((errors.primary_category=='wrong_dominant').sum())}",
                   f"all-directions-weak errors               : {int((errors.primary_category=='weak').sum())}",
                   f"center off-manifold errors               : {int(errors.center_off_manifold.sum())}",
                   f"cross-layer unstable                     : {int(errors.cross_layer_unstable.sum())}",
                   f"baseline wrong answer direction supported: {int(errors.baseline_answer_direction_supported.sum())}",
                   f"baseline == strongest wrong direction    : {int(errors.baseline_equals_best_wrong_direction.sum())}"]
    txt="\n".join(report)+"\n"; print(txt); (out/"analysis_summary.txt").write_text(txt,encoding="utf-8")
    meta={"script_version":SCRIPT_VERSION,"source_spatial_npz":a.source_spatial_npz,"target_spatial_npz":a.target_spatial_npz,"baseline_csv":a.baseline_csv,
          "layers":layers,"source_N":len(Xs),"eval_N":len(samples),"baseline_wrong_N":Nw,"evidence_lower_quantiles":qs,"primary_evidence_lower_quantile":q0,
          "support_layer_fraction":a.support_layer_fraction,"center_distance_quantile":a.center_distance_quantile,"source_labels_used_for_calibration":True,
          "target_GT_used_only_for_posthoc_diagnosis":True,"model_loaded":False,"intervention_performed":False}
    (out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
