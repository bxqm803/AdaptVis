#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_layerwise_spatial_evolution_vote_v1.py

Layer-by-layer evolution of a layer-specific spatial subspace.

Goal
====
For every requested decoder layer (default L16-L26), independently fit a
2-D H/V spatial coordinate system from the labeled synthetic source cache,
then read the target COCO sample in that layer's own spatial coordinate system.

This script is diagnostic only: no steering, no generation, no learned target
probe, and no target-GT information is used to construct the spatial subspace.
Target GT is used only for evaluation / diagnostics.

Per-layer construction
======================
For source layer L, let x_i^L be the cached relation vector.  Fit that layer's
own axes from source class means:

    dH_L ~ right - left
    dV_L ~ above - below

and read target x^L as normalized coordinates z=(z_H,z_V).  Direction evidence:

    e_left  = -z_H
    e_right = +z_H
    e_above = +z_V
    e_below = -z_V

Thus L16, L17, ..., L26 each have a SEPARATE source-defined spatial subspace.
There is no shared projection across layers.

Main requested analyses
=======================
1) Per-layer spatial accuracy and four quadrants
   For every layer report:

      baseline correct + spatial GT-lead
      baseline correct + spatial non-GT-lead
      baseline wrong   + spatial GT-lead
      baseline wrong   + spatial non-GT-lead

   Also report GT rank 1/2/3/4, GT-in-top2, margins, etc.

2) Layer-to-layer state transitions
   For each adjacent layer pair, count transitions among:

      C_GT   = baseline correct, spatial top1 == GT
      C_NGT  = baseline correct, spatial top1 != GT
      W_GT   = baseline wrong,   spatial top1 == GT
      W_NGT  = baseline wrong,   spatial top1 != GT

   Also save binary GT-lead transitions, GT-rank transitions, and the actual
   top-relation transitions (left/right/above/below).

3) Cross-layer voting
   Each layer casts one NON-ORACLE vote for its top spatial relation.
   Report:

      * majority-vote relation prediction and accuracy
      * deterministic tie-break using summed layer evidence among tied relations
      * evidence-sum / evidence-mean prediction for comparison
      * per-prefix vote accuracy (L16 only, L16-17, ..., L16-26)

   Separately, for diagnosis only, save how many layers have GT as top1:

      gt_lead_count, non_gt_lead_count, gt_lead_fraction

   These GT-count columns use target GT and are NOT a deployable selector.

Recommended run
===============
python -u diagnose_layerwise_spatial_evolution_vote_v1.py \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --baseline-csv \
    output/qwen3b_synthetic400_to_coco440_originalprompt_mean_v2/per_sample_candidate_repair.csv \
  --layers 16-26 \
  --expected-n 440 \
  --output-dir output/qwen3b_layerwise_spatial_evolution_L16_26_v1 \
  --overwrite

Outputs
=======
source_spatial_geometry.csv
per_sample_layer_spatial.csv
per_layer_summary.csv
per_layer_four_quadrants.csv
per_layer_gt_rank_distribution.csv
per_sample_trajectory.csv
adjacent_four_state_transitions.csv
adjacent_gtlead_transitions.csv
adjacent_gt_rank_transitions.csv
adjacent_top_relation_transitions.csv
transition_events.csv
vote_summary.csv
per_sample_vote.csv
vote_accuracy_by_prefix.csv
gt_lead_count_distribution.csv
gt_lead_count_by_baseline_correctness.csv
analysis_summary.txt
metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REL: Tuple[str, ...] = ("left", "right", "above", "below")
STATE_ORDER: Tuple[str, ...] = ("C_GT", "C_NGT", "W_GT", "W_NGT")
EPS = 1e-10
SCRIPT_VERSION = "v1_layer_specific_L16_26_evolution_vote"


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


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes", "y"])


def _first_existing(columns: Iterable[str], candidates: Sequence[str]):
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--layers", default="16-26")
    p.add_argument(
        "--expected-n", type=int, default=440,
        help="Require this many unique baseline samples; 0 disables the check.",
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
    return a


def load_state_npz(path: Path, require_labels: bool):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            definition = (
                str(z["vector_definition"].item())
                if "vector_definition" in keys else "relation_vectors"
            )
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
        raise RuntimeError(f"Expected [N,L,D], got {X.shape} in {path}")
    if len(sids) != len(X):
        raise RuntimeError(f"sample_index length mismatch in {path}")
    if labels is not None and len(labels) != len(X):
        raise RuntimeError(f"relation length mismatch in {path}")
    return X, labels, layers, sids, definition


def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    sid_col = _first_existing(df.columns, ["sid", "sample_index", "sample_id", "index"])
    gt_col = _first_existing(df.columns, ["gt", "relation", "ground_truth", "answer"])
    pred_col = _first_existing(
        df.columns,
        [
            "baseline_prediction", "baseline_pred", "base_prediction",
            "generation_prediction", "generated_relation", "prediction",
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
    bc_col = _first_existing(df.columns, ["baseline_correct", "generation_correct", "correct"])
    if bc_col is not None:
        out["baseline_correct"] = safe_bool_series(df[bc_col])
    else:
        out["baseline_correct"] = out["gt"].eq(out["baseline_prediction"])

    if out["sid"].duplicated().any():
        dup = out.loc[out["sid"].duplicated(), "sid"].tolist()[:10]
        raise RuntimeError(f"baseline csv contains duplicate sid, e.g. {dup}")
    bad_gt = sorted(set(out["gt"]) - set(REL))
    bad_pred = sorted(set(out["baseline_prediction"]) - set(REL))
    if bad_gt:
        raise RuntimeError(f"Unexpected GT labels: {bad_gt}")
    if bad_pred:
        raise RuntimeError(f"Unexpected baseline predictions: {bad_pred}")
    return out.sort_values("sid").reset_index(drop=True)


def fit_hv_geometry(
    X: np.ndarray,
    y: np.ndarray,
    source_layers: Sequence[int],
    wanted_layers: Sequence[int],
):
    """Fit a SEPARATE 2-D H/V coordinate system for every layer."""
    lmap = {L: i for i, L in enumerate(source_layers)}
    geom = {}
    rows = []
    for L in wanted_layers:
        if L not in lmap:
            raise RuntimeError(f"Source cache missing requested L{L}")
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

        B = np.stack([dH, dV], axis=1)  # [D, 2]
        gram = B.T @ B
        if np.linalg.cond(gram) > 1e8:
            raise RuntimeError(f"Ill-conditioned H/V basis at L{L}: cond={np.linalg.cond(gram):.3g}")
        dual = B @ np.linalg.inv(gram)

        geom[L] = {
            "center": center,
            "dual": dual,
            "halfH": max(gapH / 2.0, EPS),
            "halfV": max(gapV / 2.0, EPS),
            "layer_index": lmap[L],
        }
        rows.append({
            "layer": L,
            "source_N": len(Xf),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "full_gap_H": gapH,
            "full_gap_V": gapV,
            "half_gap_H": max(gapH / 2.0, EPS),
            "half_gap_V": max(gapV / 2.0, EPS),
            "basis_gram_condition_number": float(np.linalg.cond(gram)),
        })
    return geom, pd.DataFrame(rows)


def read_coord(x: np.ndarray, g: Mapping[str, object]) -> np.ndarray:
    res = np.asarray(x, dtype=np.float64) - np.asarray(g["center"], dtype=np.float64)
    c = np.asarray(g["dual"], dtype=np.float64).T @ res
    return np.asarray(
        [c[0] / float(g["halfH"]), c[1] / float(g["halfV"])], dtype=np.float64
    )


def axis_evidence(z: Sequence[float]) -> Dict[str, float]:
    h, v = float(z[0]), float(z[1])
    return {"left": -h, "right": h, "above": v, "below": -v}


def ordered_relations(scores: Mapping[str, float]) -> List[str]:
    # deterministic REL-order tie-break only for exact floating ties
    return sorted(REL, key=lambda r: (-float(scores[r]), REL.index(r)))


def rank_desc(scores: Mapping[str, float], relation: str) -> int:
    return int(ordered_relations(scores).index(relation) + 1)


def top1(scores: Mapping[str, float]) -> str:
    return ordered_relations(scores)[0]


def four_state(baseline_correct: bool, gt_lead: bool) -> str:
    if baseline_correct:
        return "C_GT" if gt_lead else "C_NGT"
    return "W_GT" if gt_lead else "W_NGT"


def majority_vote_prediction(
    layer_winners: Sequence[str],
    evidence_sum: Mapping[str, float],
) -> Tuple[str, int, bool, str]:
    """Non-oracle relation majority vote; tie-break by summed evidence."""
    cnt = Counter(layer_winners)
    max_votes = max(cnt.get(r, 0) for r in REL)
    tied = [r for r in REL if cnt.get(r, 0) == max_votes]
    if len(tied) == 1:
        pred = tied[0]
        tie = False
    else:
        pred = sorted(tied, key=lambda r: (-float(evidence_sum[r]), REL.index(r)))[0]
        tie = True
    vote_signature = ";".join(f"{r}:{cnt.get(r,0)}" for r in REL)
    return pred, int(max_votes), bool(tie), vote_signature


def accuracy_ci_wald(correct: int, n: int) -> Tuple[float, float, float]:
    # Descriptive only; Wilson would be nicer, but this table mainly reports counts.
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = correct / n
    se = np.sqrt(max(p * (1 - p), 0.0) / n)
    return float(p), float(max(0.0, p - 1.96 * se)), float(min(1.0, p + 1.96 * se))


def build_per_layer_summary(layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L, g in layer_df.groupby("layer", sort=True):
        n = len(g)
        gt_top1_n = int(g["gt_lead"].sum())
        gt_top2_n = int(g["gt_rank"].le(2).sum())
        rows.append({
            "layer": int(L),
            "N": n,
            "baseline_accuracy": float(g["baseline_correct"].mean()),
            "spatial_top1_accuracy": float(g["gt_lead"].mean()),
            "spatial_gt_top1_N": gt_top1_n,
            "spatial_gt_top2_accuracy": float(g["gt_rank"].le(2).mean()),
            "spatial_gt_top2_N": gt_top2_n,
            "mean_gt_rank": float(g["gt_rank"].mean()),
            "median_gt_rank": float(g["gt_rank"].median()),
            "mean_gt_minus_bestwrong": float(g["gt_minus_bestwrong_evidence"].mean()),
            "median_gt_minus_bestwrong": float(g["gt_minus_bestwrong_evidence"].median()),
            "mean_top1_minus_top2": float(g["top1_minus_top2_evidence"].mean()),
            "C_GT": int(g["four_state"].eq("C_GT").sum()),
            "C_NGT": int(g["four_state"].eq("C_NGT").sum()),
            "W_GT": int(g["four_state"].eq("W_GT").sum()),
            "W_NGT": int(g["four_state"].eq("W_NGT").sum()),
            # If one were to replace baseline by this single layer's spatial top1:
            "single_layer_W2C": int((~g["baseline_correct"] & g["gt_lead"]).sum()),
            "single_layer_C2W": int((g["baseline_correct"] & ~g["gt_lead"]).sum()),
            "single_layer_net_vs_baseline": int(
                (~g["baseline_correct"] & g["gt_lead"]).sum()
                - (g["baseline_correct"] & ~g["gt_lead"]).sum()
            ),
        })
    return pd.DataFrame(rows)


def build_quadrant_table(layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L, g in layer_df.groupby("layer", sort=True):
        n = len(g)
        for state in STATE_ORDER:
            q = g[g["four_state"].eq(state)]
            rows.append({
                "layer": int(L),
                "state": state,
                "N": len(q),
                "fraction_of_layer": float(len(q) / n) if n else np.nan,
                "mean_gt_rank": float(q["gt_rank"].mean()) if len(q) else np.nan,
                "mean_gt_minus_bestwrong": (
                    float(q["gt_minus_bestwrong_evidence"].mean()) if len(q) else np.nan
                ),
                "mean_top1_minus_top2": (
                    float(q["top1_minus_top2_evidence"].mean()) if len(q) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def build_gt_rank_distribution(layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L, g in layer_df.groupby("layer", sort=True):
        for cohort_name, c in [
            ("all", g),
            ("baseline_correct", g[g["baseline_correct"]]),
            ("baseline_wrong", g[~g["baseline_correct"]]),
        ]:
            N = len(c)
            for rank in (1, 2, 3, 4):
                n = int(c["gt_rank"].eq(rank).sum())
                rows.append({
                    "layer": int(L),
                    "cohort": cohort_name,
                    "gt_rank": rank,
                    "N": n,
                    "cohort_N": N,
                    "fraction": float(n / N) if N else np.nan,
                })
    return pd.DataFrame(rows)


def adjacent_transition_tables(layer_df: pd.DataFrame, layers: Sequence[int]):
    idx = layer_df.set_index(["sid", "layer"]).sort_index()
    four_rows = []
    lead_rows = []
    rank_rows = []
    rel_rows = []
    event_rows = []

    for La, Lb in zip(layers[:-1], layers[1:]):
        sa = idx.xs(La, level="layer")
        sb = idx.xs(Lb, level="layer")
        common = sa.index.intersection(sb.index)
        if len(common) == 0:
            continue
        a = sa.loc[common]
        b = sb.loc[common]

        for s0 in STATE_ORDER:
            for s1 in STATE_ORDER:
                n = int((a["four_state"].eq(s0) & b["four_state"].eq(s1)).sum())
                four_rows.append({
                    "from_layer": La, "to_layer": Lb,
                    "from_state": s0, "to_state": s1, "N": n,
                    "pair_N": len(common), "fraction_of_pair": float(n / len(common)),
                })

        for x0 in (True, False):
            for x1 in (True, False):
                mask = a["gt_lead"].eq(x0) & b["gt_lead"].eq(x1)
                n = int(mask.sum())
                lead_rows.append({
                    "from_layer": La, "to_layer": Lb,
                    "from_gt_lead": bool(x0), "to_gt_lead": bool(x1),
                    "N": n, "pair_N": len(common),
                    "fraction_of_pair": float(n / len(common)),
                })

        for r0 in (1, 2, 3, 4):
            for r1 in (1, 2, 3, 4):
                n = int((a["gt_rank"].eq(r0) & b["gt_rank"].eq(r1)).sum())
                rank_rows.append({
                    "from_layer": La, "to_layer": Lb,
                    "from_gt_rank": r0, "to_gt_rank": r1,
                    "N": n, "pair_N": len(common),
                    "fraction_of_pair": float(n / len(common)),
                })

        for r0 in REL:
            for r1 in REL:
                n = int((a["top_direction"].eq(r0) & b["top_direction"].eq(r1)).sum())
                rel_rows.append({
                    "from_layer": La, "to_layer": Lb,
                    "from_top_relation": r0, "to_top_relation": r1,
                    "N": n, "pair_N": len(common),
                    "fraction_of_pair": float(n / len(common)),
                })

        for sid in common:
            ra = a.loc[sid]
            rb = b.loc[sid]
            event_rows.append({
                "sid": int(sid),
                "gt": ra["gt"],
                "baseline_prediction": ra["baseline_prediction"],
                "baseline_correct": bool(ra["baseline_correct"]),
                "from_layer": La,
                "to_layer": Lb,
                "from_state": ra["four_state"],
                "to_state": rb["four_state"],
                "state_changed": bool(ra["four_state"] != rb["four_state"]),
                "from_gt_lead": bool(ra["gt_lead"]),
                "to_gt_lead": bool(rb["gt_lead"]),
                "gt_lead_changed": bool(ra["gt_lead"] != rb["gt_lead"]),
                "from_gt_rank": int(ra["gt_rank"]),
                "to_gt_rank": int(rb["gt_rank"]),
                "from_top_relation": ra["top_direction"],
                "to_top_relation": rb["top_direction"],
                "top_relation_changed": bool(ra["top_direction"] != rb["top_direction"]),
                "from_gt_margin": float(ra["gt_minus_bestwrong_evidence"]),
                "to_gt_margin": float(rb["gt_minus_bestwrong_evidence"]),
                "delta_gt_margin": float(
                    rb["gt_minus_bestwrong_evidence"] - ra["gt_minus_bestwrong_evidence"]
                ),
            })

    return (
        pd.DataFrame(four_rows),
        pd.DataFrame(lead_rows),
        pd.DataFrame(rank_rows),
        pd.DataFrame(rel_rows),
        pd.DataFrame(event_rows),
    )


def main() -> None:
    a = parse_args()
    layers = a.layers_parsed
    out = Path(a.output_dir)
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"Output directory exists: {out} (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    src_path = Path(a.source_spatial_npz)
    tgt_path = Path(a.target_spatial_npz)
    base_path = Path(a.baseline_csv)

    Xs, ys, slayers, ssids, sdef = load_state_npz(src_path, require_labels=True)
    Xt, yt, tlayers, tsids, tdef = load_state_npz(tgt_path, require_labels=False)
    base = load_baseline(base_path)

    if a.expected_n and len(base) != a.expected_n:
        raise RuntimeError(f"Expected {a.expected_n} baseline samples, found {len(base)}")

    missing_source = [L for L in layers if L not in slayers]
    missing_target = [L for L in layers if L not in tlayers]
    if missing_source:
        raise RuntimeError(f"Source cache missing requested layers: {missing_source}")
    if missing_target:
        raise RuntimeError(f"Target cache missing requested layers: {missing_target}")

    t_sid_to_idx = {int(sid): i for i, sid in enumerate(tsids.tolist())}
    missing_sid = [int(sid) for sid in base["sid"] if int(sid) not in t_sid_to_idx]
    if missing_sid:
        raise RuntimeError(f"Target cache missing baseline sids, e.g. {missing_sid[:10]}")

    if yt is not None:
        mismatch = []
        for row in base.itertuples(index=False):
            ti = t_sid_to_idx[int(row.sid)]
            if norm_rel(yt[ti]) != norm_rel(row.gt):
                mismatch.append(int(row.sid))
        if mismatch:
            raise RuntimeError(f"GT mismatch between target NPZ and baseline CSV, e.g. {mismatch[:10]}")

    geom, geom_df = fit_hv_geometry(Xs, ys, slayers, layers)
    geom_df.to_csv(out / "source_spatial_geometry.csv", index=False)

    t_layer_index = {L: i for i, L in enumerate(tlayers)}
    layer_rows = []

    for br in base.itertuples(index=False):
        sid = int(br.sid)
        gt = norm_rel(br.gt)
        bp = norm_rel(br.baseline_prediction)
        bc = bool(br.baseline_correct)
        ti = t_sid_to_idx[sid]

        for L in layers:
            z = read_coord(Xt[ti, t_layer_index[L]], geom[L])
            ev = axis_evidence(z)
            order = ordered_relations(ev)
            winner, runner = order[0], order[1]
            gt_rank = int(order.index(gt) + 1)
            best_wrong = max((r for r in REL if r != gt), key=lambda r: float(ev[r]))
            gt_lead = bool(winner == gt)

            layer_rows.append({
                "sid": sid,
                "gt": gt,
                "baseline_prediction": bp,
                "baseline_correct": bc,
                "layer": L,
                "z_H": float(z[0]),
                "z_V": float(z[1]),
                "spatial_norm": float(np.linalg.norm(z)),
                **{f"evidence_{r}": float(ev[r]) for r in REL},
                "top_direction": winner,
                "second_direction": runner,
                "top_direction_evidence": float(ev[winner]),
                "second_direction_evidence": float(ev[runner]),
                "top1_minus_top2_evidence": float(ev[winner] - ev[runner]),
                "gt_rank": gt_rank,
                "gt_lead": gt_lead,
                "gt_in_top2": bool(gt_rank <= 2),
                "gt_direction_evidence": float(ev[gt]),
                "best_wrong_direction": best_wrong,
                "best_wrong_direction_evidence": float(ev[best_wrong]),
                "gt_minus_bestwrong_evidence": float(ev[gt] - ev[best_wrong]),
                "four_state": four_state(bc, gt_lead),
                "baseline_equals_spatial_top1": bool(bp == winner),
            })

    layer_df = pd.DataFrame(layer_rows).sort_values(["sid", "layer"]).reset_index(drop=True)
    layer_df.to_csv(out / "per_sample_layer_spatial.csv", index=False)

    per_layer = build_per_layer_summary(layer_df)
    quadrants = build_quadrant_table(layer_df)
    rank_dist = build_gt_rank_distribution(layer_df)
    per_layer.to_csv(out / "per_layer_summary.csv", index=False)
    quadrants.to_csv(out / "per_layer_four_quadrants.csv", index=False)
    rank_dist.to_csv(out / "per_layer_gt_rank_distribution.csv", index=False)

    # ------------------------------------------------------------------
    # Sample trajectories + full-window vote
    # ------------------------------------------------------------------
    traj_rows = []
    vote_rows = []
    for sid, g in layer_df.groupby("sid", sort=True):
        g = g.sort_values("layer")
        gt = str(g["gt"].iloc[0])
        bp = str(g["baseline_prediction"].iloc[0])
        bc = bool(g["baseline_correct"].iloc[0])

        winners = g["top_direction"].tolist()
        states = g["four_state"].tolist()
        ranks = [int(v) for v in g["gt_rank"].tolist()]
        margins = [float(v) for v in g["gt_minus_bestwrong_evidence"].tolist()]
        gt_count = int(sum(w == gt for w in winners))
        non_gt_count = int(len(winners) - gt_count)
        evidence_sum = {r: float(g[f"evidence_{r}"].sum()) for r in REL}
        evidence_mean = {r: float(g[f"evidence_{r}"].mean()) for r in REL}
        vote_pred, max_votes, had_tie, vote_signature = majority_vote_prediction(
            winners, evidence_sum
        )
        sum_pred = top1(evidence_sum)
        mean_pred = top1(evidence_mean)
        cnt = Counter(winners)

        traj_rows.append({
            "sid": int(sid),
            "gt": gt,
            "baseline_prediction": bp,
            "baseline_correct": bc,
            "layers": ",".join(str(v) for v in g["layer"].tolist()),
            "top_direction_trajectory": ">".join(winners),
            "four_state_trajectory": ">".join(states),
            "gt_rank_trajectory": ">".join(str(v) for v in ranks),
            "gt_margin_trajectory": ">".join(f"{v:.5f}" for v in margins),
            "top_direction_transition_count": int(sum(
                winners[i] != winners[i - 1] for i in range(1, len(winners))
            )),
            "gt_lead_transition_count": int(sum(
                (winners[i] == gt) != (winners[i - 1] == gt)
                for i in range(1, len(winners))
            )),
        })

        vote_rows.append({
            "sid": int(sid),
            "gt": gt,
            "baseline_prediction": bp,
            "baseline_correct": bc,
            "N_layers": len(winners),
            "gt_lead_count": gt_count,  # diagnostic, uses GT
            "non_gt_lead_count": non_gt_count,
            "gt_lead_fraction": float(gt_count / len(winners)),
            "gt_has_strict_layer_majority": bool(gt_count > non_gt_count),
            "gt_has_half_or_more_layers": bool(gt_count * 2 >= len(winners)),
            **{f"vote_count_{r}": int(cnt.get(r, 0)) for r in REL},
            "majority_vote_prediction": vote_pred,
            "majority_vote_correct": bool(vote_pred == gt),
            "majority_vote_max_votes": max_votes,
            "majority_vote_had_tie": had_tie,
            "vote_signature": vote_signature,
            "evidence_sum_prediction": sum_pred,
            "evidence_sum_correct": bool(sum_pred == gt),
            "evidence_mean_prediction": mean_pred,
            "evidence_mean_correct": bool(mean_pred == gt),
            **{f"evidence_sum_{r}": evidence_sum[r] for r in REL},
            **{f"evidence_mean_{r}": evidence_mean[r] for r in REL},
        })

    traj_df = pd.DataFrame(traj_rows).sort_values("sid").reset_index(drop=True)
    vote_df = pd.DataFrame(vote_rows).sort_values("sid").reset_index(drop=True)
    traj_df.to_csv(out / "per_sample_trajectory.csv", index=False)
    vote_df.to_csv(out / "per_sample_vote.csv", index=False)

    # ------------------------------------------------------------------
    # Adjacent-layer transitions
    # ------------------------------------------------------------------
    four_trans, lead_trans, rank_trans, rel_trans, events = adjacent_transition_tables(
        layer_df, layers
    )
    four_trans.to_csv(out / "adjacent_four_state_transitions.csv", index=False)
    lead_trans.to_csv(out / "adjacent_gtlead_transitions.csv", index=False)
    rank_trans.to_csv(out / "adjacent_gt_rank_transitions.csv", index=False)
    rel_trans.to_csv(out / "adjacent_top_relation_transitions.csv", index=False)
    events.to_csv(out / "transition_events.csv", index=False)

    # ------------------------------------------------------------------
    # Vote summary and vote-by-prefix
    # ------------------------------------------------------------------
    N = len(vote_df)
    baseline_acc = float(vote_df["baseline_correct"].mean())
    majority_acc = float(vote_df["majority_vote_correct"].mean())
    sum_acc = float(vote_df["evidence_sum_correct"].mean())
    mean_acc = float(vote_df["evidence_mean_correct"].mean())

    vote_summary = pd.DataFrame([
        {
            "method": "baseline_generation",
            "N": N,
            "correct_N": int(vote_df["baseline_correct"].sum()),
            "accuracy": baseline_acc,
            "uses_target_GT_for_prediction": False,
        },
        {
            "method": f"relation_majority_vote_L{layers[0]}_L{layers[-1]}",
            "N": N,
            "correct_N": int(vote_df["majority_vote_correct"].sum()),
            "accuracy": majority_acc,
            "uses_target_GT_for_prediction": False,
        },
        {
            "method": f"evidence_sum_L{layers[0]}_L{layers[-1]}",
            "N": N,
            "correct_N": int(vote_df["evidence_sum_correct"].sum()),
            "accuracy": sum_acc,
            "uses_target_GT_for_prediction": False,
        },
        {
            "method": f"evidence_mean_L{layers[0]}_L{layers[-1]}",
            "N": N,
            "correct_N": int(vote_df["evidence_mean_correct"].sum()),
            "accuracy": mean_acc,
            "uses_target_GT_for_prediction": False,
        },
        {
            "method": "diagnostic_GT_has_strict_majority_of_layers",
            "N": N,
            "correct_N": int(vote_df["gt_has_strict_layer_majority"].sum()),
            "accuracy": float(vote_df["gt_has_strict_layer_majority"].mean()),
            "uses_target_GT_for_prediction": True,
        },
    ])
    vote_summary.to_csv(out / "vote_summary.csv", index=False)

    prefix_rows = []
    idx = layer_df.set_index(["sid", "layer"]).sort_index()
    for k in range(1, len(layers) + 1):
        use_layers = layers[:k]
        corr_vote = 0
        corr_sum = 0
        gt_strict_majority = 0
        gt_top2_by_sum = 0
        for br in base.itertuples(index=False):
            sid = int(br.sid)
            gt = norm_rel(br.gt)
            gg = idx.loc[sid].loc[use_layers]
            winners = gg["top_direction"].tolist()
            es = {r: float(gg[f"evidence_{r}"].sum()) for r in REL}
            pred, _, _, _ = majority_vote_prediction(winners, es)
            sum_order = ordered_relations(es)
            corr_vote += int(pred == gt)
            corr_sum += int(sum_order[0] == gt)
            gt_strict_majority += int(sum(w == gt for w in winners) > len(winners) / 2)
            gt_top2_by_sum += int(gt in sum_order[:2])
        prefix_rows.append({
            "start_layer": layers[0],
            "end_layer": use_layers[-1],
            "N_layers": len(use_layers),
            "N": len(base),
            "majority_vote_correct_N": corr_vote,
            "majority_vote_accuracy": float(corr_vote / len(base)),
            "evidence_sum_correct_N": corr_sum,
            "evidence_sum_accuracy": float(corr_sum / len(base)),
            "evidence_sum_gt_top2_N": gt_top2_by_sum,
            "evidence_sum_gt_top2_rate": float(gt_top2_by_sum / len(base)),
            # diagnostic only; uses GT to count how many layer winners equal GT
            "GT_strict_layer_majority_N": gt_strict_majority,
            "GT_strict_layer_majority_rate_diagnostic": float(gt_strict_majority / len(base)),
        })
    prefix_df = pd.DataFrame(prefix_rows)
    prefix_df.to_csv(out / "vote_accuracy_by_prefix.csv", index=False)

    # ------------------------------------------------------------------
    # GT-lead count distribution (diagnostic; uses GT)
    # ------------------------------------------------------------------
    count_rows = []
    for k in range(len(layers) + 1):
        g = vote_df[vote_df["gt_lead_count"].eq(k)]
        count_rows.append({
            "gt_lead_count": k,
            "non_gt_lead_count": len(layers) - k,
            "N": len(g),
            "fraction": float(len(g) / N) if N else np.nan,
            "baseline_accuracy": float(g["baseline_correct"].mean()) if len(g) else np.nan,
            "majority_vote_accuracy": float(g["majority_vote_correct"].mean()) if len(g) else np.nan,
            "evidence_sum_accuracy": float(g["evidence_sum_correct"].mean()) if len(g) else np.nan,
        })
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(out / "gt_lead_count_distribution.csv", index=False)

    by_bc_rows = []
    for bc, cname in [(True, "baseline_correct"), (False, "baseline_wrong")]:
        sub = vote_df[vote_df["baseline_correct"].eq(bc)]
        for k in range(len(layers) + 1):
            g = sub[sub["gt_lead_count"].eq(k)]
            by_bc_rows.append({
                "baseline_cohort": cname,
                "gt_lead_count": k,
                "non_gt_lead_count": len(layers) - k,
                "N": len(g),
                "cohort_N": len(sub),
                "fraction_within_cohort": float(len(g) / len(sub)) if len(sub) else np.nan,
            })
    by_bc_df = pd.DataFrame(by_bc_rows)
    by_bc_df.to_csv(out / "gt_lead_count_by_baseline_correctness.csv", index=False)

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 160)
    print("PER-LAYER SPATIAL EVOLUTION (EACH LAYER HAS ITS OWN SOURCE-FITTED H/V SUBSPACE)")
    print("=" * 160)
    print(per_layer.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 160)
    print("ADJACENT-LAYER GT-LEAD TRANSITIONS")
    print("=" * 160)
    compact_trans = lead_trans.pivot_table(
        index=["from_layer", "to_layer"],
        columns=["from_gt_lead", "to_gt_lead"],
        values="N",
        aggfunc="sum",
        fill_value=0,
    )
    print(compact_trans.to_string())

    print("\n" + "=" * 160)
    print("VOTE SUMMARY")
    print("=" * 160)
    print(vote_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 160)
    print("VOTE / EVIDENCE ACCURACY BY PREFIX")
    print("=" * 160)
    print(prefix_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 160)
    print("GT-LEAD LAYER COUNT DISTRIBUTION (DIAGNOSTIC ONLY; THIS TABLE USES TARGET GT)")
    print("=" * 160)
    print(count_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Text summary
    best_layer_row = per_layer.loc[per_layer["spatial_top1_accuracy"].idxmax()]
    best_prefix_vote = prefix_df.loc[prefix_df["majority_vote_accuracy"].idxmax()]
    best_prefix_sum = prefix_df.loc[prefix_df["evidence_sum_accuracy"].idxmax()]

    summary_lines = [
        f"script_version={SCRIPT_VERSION}",
        f"layers={layers}",
        f"N={N}",
        f"baseline_accuracy={baseline_acc:.6f}",
        f"best_single_layer=L{int(best_layer_row['layer'])} acc={float(best_layer_row['spatial_top1_accuracy']):.6f}",
        f"full_window_majority_vote_accuracy={majority_acc:.6f}",
        f"full_window_evidence_sum_accuracy={sum_acc:.6f}",
        f"full_window_evidence_mean_accuracy={mean_acc:.6f}",
        (
            "best_prefix_majority_vote="
            f"L{int(best_prefix_vote['start_layer'])}-L{int(best_prefix_vote['end_layer'])} "
            f"acc={float(best_prefix_vote['majority_vote_accuracy']):.6f}"
        ),
        (
            "best_prefix_evidence_sum="
            f"L{int(best_prefix_sum['start_layer'])}-L{int(best_prefix_sum['end_layer'])} "
            f"acc={float(best_prefix_sum['evidence_sum_accuracy']):.6f}"
        ),
        "NOTE: gt_lead_count / GT_strict_layer_majority are post-hoc diagnostics using target GT;",
        "      relation_majority_vote and evidence_sum/mean predictions are non-oracle with respect to target GT.",
    ]
    (out / "analysis_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    metadata = {
        "script_version": SCRIPT_VERSION,
        "source_spatial_npz": str(src_path),
        "target_spatial_npz": str(tgt_path),
        "baseline_csv": str(base_path),
        "source_vector_definition": sdef,
        "target_vector_definition": tdef,
        "layers": layers,
        "N_source": int(len(Xs)),
        "N_target": int(len(base)),
        "relations": list(REL),
        "baseline_accuracy": baseline_acc,
        "full_window_majority_vote_accuracy": majority_acc,
        "full_window_evidence_sum_accuracy": sum_acc,
        "full_window_evidence_mean_accuracy": mean_acc,
        "target_GT_used_to_fit_subspace": False,
        "target_GT_used_for_gt_lead_diagnostics": True,
        "vote_prediction_uses_target_GT": False,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n[saved] {out}")
    print("[success]")


if __name__ == "__main__":
    main()
