#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer-update Direction harm scan (CPU-only).

Goal
----
Use the cached layer-wise Direction representation to identify WHICH LAYER
UPDATE makes spatial evidence worse, rather than only asking which layer state
is already wrong.

This script does NOT load the VLM and does NOT run generation. It reuses:
    <direction-dir>/vectors.npz
    <direction-dir>/sample_split_and_generation.csv

Correct / wrong grouping comes from the cached ACTUAL model.generate() result.

Core quantities
---------------
Cached residual relation representation:
    r_l = (h_sub^img - h_ref^img)
          - (h_sub^noimg - h_ref^noimg)

Train-set codebook at each layer:
    q_l = r_l - center_l
    mu_left,l, mu_right,l, mu_above,l, mu_below,l

For transition L(l-1) -> L(l):
    delta_r_l = r_l - r_{l-1}

For GT g and competitor c, define the relation axis:
    d_l(g,c) = mu_g,l - mu_c,l

Then the layer-update effect is:
    U_l^current = delta_r_l dot d_l(g,c)

Interpretation:
    U > 0 : this layer update pushes toward GT relative to competitor.
    U < 0 : this layer update pushes toward competitor relative to GT.
    harm = -U, so larger positive harm = more negative update.

Because the Direction prototype basis itself changes across layers, the script
also evaluates the SAME update on the previous-layer axis:
    U_l^previous = delta_r_l dot d_{l-1}(g,c)

A "robust harmful update" has:
    U_l^current < 0 AND U_l^previous < 0

Two competitor analyses
------------------------
1) final_generated_wrong
   For generation-wrong samples only, c is the actual wrong relation produced
   by model.generate(). This is the main error-localization metric.

2) best_competitor
   For every sample, c is the strongest non-GT Direction competitor at the
   CURRENT layer. This allows generation-correct vs generation-wrong group
   comparison.

State-margin change
-------------------
For reference only, the script also computes:
    M_l = q_l dot d_l(g,c)
    DeltaM_l = M_l - M_{l-1}

DeltaM uses different layer-specific codebooks at the two endpoints, so it is
a state-trajectory diagnostic, NOT a pure layer-update attribution. For actual
update attribution, prefer U_l^current / U_l^previous.

Main outputs
------------
per_sample_layer_update.csv
    Every sample x layer transition.

layer_update_summary.csv
    Correct-vs-wrong comparison using strongest non-GT competitor.

wrong_final_relation_summary.csv
    Generation-wrong samples only, using the ACTUAL final generated wrong
    relation.

worst_layer_frequency.csv
    How often each layer is the single most harmful update for a wrong sample.

first_robust_harmful_layer.csv
    How often each layer is the first robust harmful update for a wrong sample.

top_harmful_layers.csv
    Ranked candidate layers using actual generated-wrong relation.

Recommended run
---------------
python analyze_layer_update_direction_harm_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --split test \
  --output-dir output/qwen7b_layer_update_direction_harm_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


# =============================================================================
# CLI / I/O
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--generation-groups",
        default="correct,wrong",
        help="Cached actual-generation groups to include.",
    )
    p.add_argument(
        "--layers",
        default="all",
        help=(
            "Target/output layers whose update r_l-r_(l-1) is analyzed. "
            "Example: 12,13,...,20. L0 has no previous layer and is ignored."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--robust-threshold",
        type=float,
        default=0.0,
        help=(
            "Require both current-axis and previous-axis UNIT effects to be "
            "below -threshold to call an update robust harmful."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_relation(x: Any) -> str:
    t = str(x).strip().lower()
    aliases = {
        "left of": "left",
        "right of": "right",
        "on": "above",
        "over": "above",
        "up": "above",
        "under": "below",
        "beneath": "below",
        "down": "below",
    }
    if t in REL2ID:
        return t
    if t in aliases:
        return aliases[t]

    # Conservative substring fallback for cached outputs.
    for rel in RELATIONS:
        if rel in t.split():
            return rel
    return ""


def safe_mean(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(a.mean()) if len(a) else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(np.median(a)) if len(a) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_layers(text: str, n_layers: int) -> List[int]:
    if str(text).strip().lower() == "all":
        return list(range(1, n_layers))

    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(chunk))

    out = sorted(set(out))
    if 0 in out:
        print("[warn] L0 has no previous layer; ignoring L0.")
        out = [x for x in out if x != 0]

    bad = [x for x in out if not 1 <= x < n_layers]
    if bad:
        raise ValueError(
            f"Invalid target layers {bad}; valid update targets are "
            f"L1..L{n_layers-1}"
        )
    if not out:
        raise ValueError("No valid target layers.")
    return out


# =============================================================================
# Cached assets / codebooks
# =============================================================================

def fit_codebook(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    center = X_train.mean(axis=0)
    Xc = X_train - center

    protos = []
    for rel in RELATIONS:
        mask = y_train == rel
        if not np.any(mask):
            raise RuntimeError(f"No train examples for {rel}")
        p = Xc[mask].mean(axis=0)
        n = float(np.linalg.norm(p))
        if n <= EPS:
            raise RuntimeError(f"Degenerate prototype for {rel}")
        protos.append(p / n)

    return (
        center.astype(np.float32),
        np.stack(protos).astype(np.float32),
    )


def load_assets(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    gen_path = direction_dir / "sample_split_and_generation.csv"

    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    with np.load(vec_path, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}

    required = {"sample_index", "relation", "residual"}
    missing = required - set(arrays)
    if missing:
        raise KeyError(
            f"{vec_path} missing required arrays: {sorted(missing)}"
        )

    generation_rows = read_csv(gen_path)

    sids = arrays["sample_index"].astype(np.int64)
    labels = np.asarray(
        [norm_relation(x) for x in arrays["relation"]],
        dtype=object,
    )
    residual = np.asarray(arrays["residual"], dtype=np.float32)

    if residual.ndim != 3:
        raise ValueError(
            f"Expected residual [N,L,D], got {residual.shape}"
        )

    idx_by_sid = {int(s): i for i, s in enumerate(sids.tolist())}

    gen_by_sid = {}
    split_by_sid = {}
    for r in generation_rows:
        sid = int(r["sample_index"])
        split_by_sid[sid] = str(r.get("split", "")).strip()

        group = str(r.get("generation_group", "")).strip().lower()
        pred = norm_relation(r.get("generation_pred", ""))
        text = str(r.get("generation_text", ""))

        # If generation_group is missing but pred is parseable, recover it.
        idx = idx_by_sid.get(sid)
        gt = labels[idx] if idx is not None else ""
        if group not in ("correct", "wrong"):
            if pred in REL2ID and gt in REL2ID:
                group = "correct" if pred == gt else "wrong"

        gen_by_sid[sid] = {
            "generation_group": group,
            "generation_pred": pred,
            "generation_text": text,
        }

    train_indices = np.asarray(
        [
            idx_by_sid[sid]
            for sid, split in split_by_sid.items()
            if split == "train" and sid in idx_by_sid
        ],
        dtype=np.int64,
    )
    if len(train_indices) == 0:
        raise RuntimeError(
            "No train rows found in sample_split_and_generation.csv"
        )

    n_layers = residual.shape[1]
    codebooks = {}
    for li in range(n_layers):
        center, protos = fit_codebook(
            residual[train_indices, li, :],
            labels[train_indices],
        )
        codebooks[li] = {
            "center": center,
            "protos": protos,
        }

    return {
        "sids": sids,
        "labels": labels,
        "residual": residual,
        "idx_by_sid": idx_by_sid,
        "generation": gen_by_sid,
        "split": split_by_sid,
        "codebooks": codebooks,
        "n_layers": n_layers,
    }


# =============================================================================
# Geometry
# =============================================================================

def relation_axis(
    protos: np.ndarray,
    gt: str,
    competitor: str,
) -> np.ndarray:
    return (
        protos[REL2ID[gt]]
        - protos[REL2ID[competitor]]
    ).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v)
    return v / n


def state_scores(
    r: np.ndarray,
    center: np.ndarray,
    protos: np.ndarray,
) -> np.ndarray:
    q = r - center
    return q @ protos.T


def competitor_from_state(
    r: np.ndarray,
    center: np.ndarray,
    protos: np.ndarray,
    gt: str,
) -> str:
    scores = state_scores(r, center, protos)
    tmp = scores.copy()
    tmp[REL2ID[gt]] = -np.inf
    return RELATIONS[int(np.argmax(tmp))]


def compute_pair_metrics(
    *,
    delta_r: np.ndarray,
    r_prev: np.ndarray,
    r_cur: np.ndarray,
    prev_cb: Mapping[str, np.ndarray],
    cur_cb: Mapping[str, np.ndarray],
    gt: str,
    competitor: str,
) -> Dict[str, float]:
    d_prev = relation_axis(
        prev_cb["protos"], gt, competitor
    )
    d_cur = relation_axis(
        cur_cb["protos"], gt, competitor
    )

    d_prev_u = unit(d_prev)
    d_cur_u = unit(d_cur)

    update_raw_previous = float(delta_r @ d_prev)
    update_raw_current = float(delta_r @ d_cur)

    update_unit_previous = float(delta_r @ d_prev_u)
    update_unit_current = float(delta_r @ d_cur_u)

    q_prev = r_prev - prev_cb["center"]
    q_cur = r_cur - cur_cb["center"]

    margin_prev = float(q_prev @ d_prev)
    margin_cur = float(q_cur @ d_cur)
    state_margin_change = margin_cur - margin_prev

    update_norm = float(np.linalg.norm(delta_r))
    if update_norm <= EPS:
        cos_previous = float("nan")
        cos_current = float("nan")
    else:
        cos_previous = float(
            delta_r @ d_prev_u / update_norm
        )
        cos_current = float(
            delta_r @ d_cur_u / update_norm
        )

    return {
        "update_raw_previous_axis": update_raw_previous,
        "update_raw_current_axis": update_raw_current,
        "update_unit_previous_axis": update_unit_previous,
        "update_unit_current_axis": update_unit_current,
        "harm_unit_previous_axis": -update_unit_previous,
        "harm_unit_current_axis": -update_unit_current,
        "update_cos_previous_axis": cos_previous,
        "update_cos_current_axis": cos_current,
        "state_margin_previous": margin_prev,
        "state_margin_current": margin_cur,
        "state_margin_change": state_margin_change,
        "update_norm": update_norm,
    }


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_two_group_gap(
    correct_vals: Sequence[float],
    wrong_vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    """
    gap = mean(wrong) - mean(correct).
    For harm metrics, positive gap = wrong group receives more harmful updates.
    """
    a = np.asarray(correct_vals, dtype=np.float64)
    b = np.asarray(wrong_vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")

    obs = float(b.mean() - a.mean())
    boots = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = bb.mean() - aa.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_mean_ci(
    vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")

    obs = float(a.mean())
    boots = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        boots[i] = aa.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


# =============================================================================
# Main per-sample scan
# =============================================================================

def build_per_sample_rows(
    assets,
    selected_layers,
    split,
    wanted_groups,
    robust_threshold,
):
    rows = []

    for sid, idx in assets["idx_by_sid"].items():
        sample_split = assets["split"].get(sid, "")
        if split != "all" and sample_split != split:
            continue

        gen = assets["generation"].get(sid, {})
        group = gen.get("generation_group", "")
        if group not in wanted_groups:
            continue

        gt = assets["labels"][idx]
        if gt not in REL2ID:
            continue

        gen_pred = gen.get("generation_pred", "")
        residual = assets["residual"][idx]

        for li in selected_layers:
            prev = li - 1
            r_prev = residual[prev]
            r_cur = residual[li]
            delta_r = r_cur - r_prev

            prev_cb = assets["codebooks"][prev]
            cur_cb = assets["codebooks"][li]

            best_comp = competitor_from_state(
                r_cur,
                cur_cb["center"],
                cur_cb["protos"],
                gt,
            )

            best = compute_pair_metrics(
                delta_r=delta_r,
                r_prev=r_prev,
                r_cur=r_cur,
                prev_cb=prev_cb,
                cur_cb=cur_cb,
                gt=gt,
                competitor=best_comp,
            )

            row = {
                "sid": sid,
                "layer": li,
                "previous_layer": prev,
                "gt": gt,
                "generation_group": group,
                "generation_pred": gen_pred,
                "best_competitor_current_layer": best_comp,

                "best_update_raw_previous_axis":
                    best["update_raw_previous_axis"],
                "best_update_raw_current_axis":
                    best["update_raw_current_axis"],
                "best_update_unit_previous_axis":
                    best["update_unit_previous_axis"],
                "best_update_unit_current_axis":
                    best["update_unit_current_axis"],
                "best_harm_unit_previous_axis":
                    best["harm_unit_previous_axis"],
                "best_harm_unit_current_axis":
                    best["harm_unit_current_axis"],
                "best_update_cos_previous_axis":
                    best["update_cos_previous_axis"],
                "best_update_cos_current_axis":
                    best["update_cos_current_axis"],
                "best_state_margin_previous":
                    best["state_margin_previous"],
                "best_state_margin_current":
                    best["state_margin_current"],
                "best_state_margin_change":
                    best["state_margin_change"],
                "update_norm": best["update_norm"],

                "best_robust_harmful": int(
                    best["update_unit_previous_axis"]
                    < -robust_threshold
                    and best["update_unit_current_axis"]
                    < -robust_threshold
                ),
            }

            # Main error-localization metric:
            # compare update against ACTUAL final generated wrong relation.
            if (
                group == "wrong"
                and gen_pred in REL2ID
                and gen_pred != gt
            ):
                final = compute_pair_metrics(
                    delta_r=delta_r,
                    r_prev=r_prev,
                    r_cur=r_cur,
                    prev_cb=prev_cb,
                    cur_cb=cur_cb,
                    gt=gt,
                    competitor=gen_pred,
                )

                row.update({
                    "final_wrong_relation": gen_pred,
                    "final_update_raw_previous_axis":
                        final["update_raw_previous_axis"],
                    "final_update_raw_current_axis":
                        final["update_raw_current_axis"],
                    "final_update_unit_previous_axis":
                        final["update_unit_previous_axis"],
                    "final_update_unit_current_axis":
                        final["update_unit_current_axis"],
                    "final_harm_unit_previous_axis":
                        final["harm_unit_previous_axis"],
                    "final_harm_unit_current_axis":
                        final["harm_unit_current_axis"],
                    "final_update_cos_previous_axis":
                        final["update_cos_previous_axis"],
                    "final_update_cos_current_axis":
                        final["update_cos_current_axis"],
                    "final_state_margin_previous":
                        final["state_margin_previous"],
                    "final_state_margin_current":
                        final["state_margin_current"],
                    "final_state_margin_change":
                        final["state_margin_change"],
                    "final_robust_harmful": int(
                        final["update_unit_previous_axis"]
                        < -robust_threshold
                        and final["update_unit_current_axis"]
                        < -robust_threshold
                    ),
                })
            else:
                row.update({
                    "final_wrong_relation": "",
                    "final_update_raw_previous_axis": float("nan"),
                    "final_update_raw_current_axis": float("nan"),
                    "final_update_unit_previous_axis": float("nan"),
                    "final_update_unit_current_axis": float("nan"),
                    "final_harm_unit_previous_axis": float("nan"),
                    "final_harm_unit_current_axis": float("nan"),
                    "final_update_cos_previous_axis": float("nan"),
                    "final_update_cos_current_axis": float("nan"),
                    "final_state_margin_previous": float("nan"),
                    "final_state_margin_current": float("nan"),
                    "final_state_margin_change": float("nan"),
                    "final_robust_harmful": 0,
                })

            rows.append(row)

    return rows


# =============================================================================
# Summaries
# =============================================================================

def summarize_all_group(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)
    out = []

    for li in selected_layers:
        rr = [r for r in rows if int(r["layer"]) == li]
        correct = [
            r for r in rr if r["generation_group"] == "correct"
        ]
        wrong = [
            r for r in rr if r["generation_group"] == "wrong"
        ]

        c_harm = [
            float(r["best_harm_unit_current_axis"])
            for r in correct
        ]
        w_harm = [
            float(r["best_harm_unit_current_axis"])
            for r in wrong
        ]

        gap, lo, hi = bootstrap_two_group_gap(
            c_harm, w_harm, bootstrap, rng
        )

        c_prev_harm = [
            float(r["best_harm_unit_previous_axis"])
            for r in correct
        ]
        w_prev_harm = [
            float(r["best_harm_unit_previous_axis"])
            for r in wrong
        ]
        gap_prev, lo_prev, hi_prev = bootstrap_two_group_gap(
            c_prev_harm, w_prev_harm, bootstrap, rng
        )

        out.append({
            "layer": li,
            "n_correct": len(correct),
            "n_wrong": len(wrong),

            "mean_best_harm_current_correct":
                safe_mean(c_harm),
            "mean_best_harm_current_wrong":
                safe_mean(w_harm),
            "wrong_minus_correct_harm_current_gap": gap,
            "harm_current_gap_ci95_lo": lo,
            "harm_current_gap_ci95_hi": hi,

            "mean_best_harm_previous_correct":
                safe_mean(c_prev_harm),
            "mean_best_harm_previous_wrong":
                safe_mean(w_prev_harm),
            "wrong_minus_correct_harm_previous_gap": gap_prev,
            "harm_previous_gap_ci95_lo": lo_prev,
            "harm_previous_gap_ci95_hi": hi_prev,

            "robust_harmful_rate_correct": safe_frac(
                bool(r["best_robust_harmful"])
                for r in correct
            ),
            "robust_harmful_rate_wrong": safe_frac(
                bool(r["best_robust_harmful"])
                for r in wrong
            ),

            "mean_state_margin_change_correct": safe_mean(
                r["best_state_margin_change"] for r in correct
            ),
            "mean_state_margin_change_wrong": safe_mean(
                r["best_state_margin_change"] for r in wrong
            ),

            "mean_update_norm_correct": safe_mean(
                r["update_norm"] for r in correct
            ),
            "mean_update_norm_wrong": safe_mean(
                r["update_norm"] for r in wrong
            ),
        })

    return out


def summarize_wrong_final_relation(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed + 1000)
    out = []

    wrong_rows = [
        r for r in rows
        if r["generation_group"] == "wrong"
        and r["final_wrong_relation"] in REL2ID
    ]

    for li in selected_layers:
        rr = [
            r for r in wrong_rows
            if int(r["layer"]) == li
        ]

        harm_cur = [
            float(r["final_harm_unit_current_axis"])
            for r in rr
        ]
        harm_prev = [
            float(r["final_harm_unit_previous_axis"])
            for r in rr
        ]

        mean_cur, lo_cur, hi_cur = bootstrap_mean_ci(
            harm_cur, bootstrap, rng
        )
        mean_prev, lo_prev, hi_prev = bootstrap_mean_ci(
            harm_prev, bootstrap, rng
        )

        out.append({
            "layer": li,
            "n_wrong": len(rr),

            "mean_harm_current_axis": mean_cur,
            "harm_current_ci95_lo": lo_cur,
            "harm_current_ci95_hi": hi_cur,

            "mean_harm_previous_axis": mean_prev,
            "harm_previous_ci95_lo": lo_prev,
            "harm_previous_ci95_hi": hi_prev,

            "mean_update_effect_current_axis": safe_mean(
                r["final_update_unit_current_axis"] for r in rr
            ),
            "mean_update_effect_previous_axis": safe_mean(
                r["final_update_unit_previous_axis"] for r in rr
            ),

            "harmful_rate_current_axis": safe_frac(
                float(r["final_update_unit_current_axis"]) < 0
                for r in rr
            ),
            "harmful_rate_previous_axis": safe_frac(
                float(r["final_update_unit_previous_axis"]) < 0
                for r in rr
            ),
            "robust_harmful_rate": safe_frac(
                bool(r["final_robust_harmful"])
                for r in rr
            ),

            "mean_state_margin_change": safe_mean(
                r["final_state_margin_change"] for r in rr
            ),
            "state_margin_decrease_rate": safe_frac(
                float(r["final_state_margin_change"]) < 0
                for r in rr
            ),

            "mean_update_norm": safe_mean(
                r["update_norm"] for r in rr
            ),
        })

    return out


def worst_layer_stats(rows, selected_layers):
    wrong_by_sid = defaultdict(list)
    for r in rows:
        if (
            r["generation_group"] == "wrong"
            and r["final_wrong_relation"] in REL2ID
        ):
            wrong_by_sid[int(r["sid"])].append(r)

    worst_counter = Counter()
    first_robust_counter = Counter()
    per_sample = []

    for sid, rr in wrong_by_sid.items():
        rr = sorted(rr, key=lambda x: int(x["layer"]))

        # Highest positive harm = most negative GT-vs-final-wrong update.
        worst = max(
            rr,
            key=lambda x: float(
                x["final_harm_unit_current_axis"]
            ),
        )
        worst_layer = int(worst["layer"])
        worst_counter[worst_layer] += 1

        robust = [
            r for r in rr if int(r["final_robust_harmful"]) == 1
        ]
        first_layer = int(robust[0]["layer"]) if robust else -1
        if first_layer >= 0:
            first_robust_counter[first_layer] += 1

        per_sample.append({
            "sid": sid,
            "gt": rr[0]["gt"],
            "generation_pred": rr[0]["generation_pred"],
            "worst_harm_layer": worst_layer,
            "worst_harm_current_axis":
                worst["final_harm_unit_current_axis"],
            "worst_update_effect_current_axis":
                worst["final_update_unit_current_axis"],
            "first_robust_harmful_layer": (
                first_layer if first_layer >= 0 else ""
            ),
        })

    n = len(wrong_by_sid)

    worst_rows = []
    for li in selected_layers:
        worst_rows.append({
            "layer": li,
            "count_as_most_harmful": worst_counter[li],
            "fraction_of_wrong_samples":
                worst_counter[li] / n if n else float("nan"),
        })

    first_rows = []
    for li in selected_layers:
        first_rows.append({
            "layer": li,
            "count_as_first_robust_harmful":
                first_robust_counter[li],
            "fraction_of_wrong_samples":
                first_robust_counter[li] / n if n else float("nan"),
        })

    return per_sample, worst_rows, first_rows


def rank_harmful_layers(
    wrong_summary,
    group_summary,
    worst_rows,
):
    group_by_layer = {
        int(r["layer"]): r for r in group_summary
    }
    worst_by_layer = {
        int(r["layer"]): r for r in worst_rows
    }

    ranked = []
    for r in wrong_summary:
        li = int(r["layer"])
        g = group_by_layer[li]
        w = worst_by_layer[li]

        # No opaque composite score: preserve interpretable columns and rank
        # primarily by mean harm toward ACTUAL generated wrong relation.
        ranked.append({
            "layer": li,
            "mean_harm_to_final_generated_wrong":
                r["mean_harm_current_axis"],
            "final_wrong_harm_ci95_lo":
                r["harm_current_ci95_lo"],
            "final_wrong_harm_ci95_hi":
                r["harm_current_ci95_hi"],
            "robust_harmful_rate_on_wrong":
                r["robust_harmful_rate"],
            "harmful_rate_current_on_wrong":
                r["harmful_rate_current_axis"],
            "state_margin_decrease_rate_on_wrong":
                r["state_margin_decrease_rate"],
            "wrong_minus_correct_best_comp_harm_gap":
                g["wrong_minus_correct_harm_current_gap"],
            "best_comp_harm_gap_ci95_lo":
                g["harm_current_gap_ci95_lo"],
            "best_comp_harm_gap_ci95_hi":
                g["harm_current_gap_ci95_hi"],
            "fraction_as_single_worst_layer":
                w["fraction_of_wrong_samples"],
        })

    ranked.sort(
        key=lambda x: float(
            x["mean_harm_to_final_generated_wrong"]
        ),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["mean_harm_ci_positive"] = int(
            math.isfinite(
                float(r["final_wrong_harm_ci95_lo"])
            )
            and float(r["final_wrong_harm_ci95_lo"]) > 0
        )
        r["correct_wrong_gap_ci_positive"] = int(
            math.isfinite(
                float(r["best_comp_harm_gap_ci95_lo"])
            )
            and float(r["best_comp_harm_gap_ci95_lo"]) > 0
        )

    return ranked


# =============================================================================
# Console output
# =============================================================================

def print_wrong_summary(rows):
    print("\n" + "=" * 154)
    print("LAYER UPDATE -> ACTUAL FINAL GENERATED WRONG RELATION")
    print("=" * 154)
    print(
        "layer Nwrong  harm(cur)       95%CI              harm(prev)  "
        "harmRate(cur/prev) robustRate  dStateMargin  stateDown%  updateNorm"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d}   "
            f"{int(r['n_wrong']):3d}   "
            f"{float(r['mean_harm_current_axis']):+8.4f}   "
            f"[{float(r['harm_current_ci95_lo']):+7.4f},"
            f"{float(r['harm_current_ci95_hi']):+7.4f}]   "
            f"{float(r['mean_harm_previous_axis']):+8.4f}   "
            f"{float(r['harmful_rate_current_axis']):.3f}/"
            f"{float(r['harmful_rate_previous_axis']):.3f}       "
            f"{float(r['robust_harmful_rate']):.3f}      "
            f"{float(r['mean_state_margin_change']):+8.3f}      "
            f"{float(r['state_margin_decrease_rate']):.3f}      "
            f"{float(r['mean_update_norm']):.3f}"
        )


def print_group_summary(rows):
    print("\n" + "=" * 150)
    print("CORRECT vs WRONG — UPDATE HARM TO CURRENT STRONGEST NON-GT COMPETITOR")
    print("=" * 150)
    print(
        "layer  harmCorrect  harmWrong  W-C gap       95%CI             "
        "robust(correct/wrong)  dStateMargin(correct/wrong)"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d}    "
            f"{float(r['mean_best_harm_current_correct']):+8.4f}   "
            f"{float(r['mean_best_harm_current_wrong']):+8.4f}   "
            f"{float(r['wrong_minus_correct_harm_current_gap']):+8.4f}   "
            f"[{float(r['harm_current_gap_ci95_lo']):+7.4f},"
            f"{float(r['harm_current_gap_ci95_hi']):+7.4f}]      "
            f"{float(r['robust_harmful_rate_correct']):.3f}/"
            f"{float(r['robust_harmful_rate_wrong']):.3f}              "
            f"{float(r['mean_state_margin_change_correct']):+7.3f}/"
            f"{float(r['mean_state_margin_change_wrong']):+7.3f}"
        )


def print_top(ranked, n=15):
    print("\n" + "=" * 150)
    print("TOP HARMFUL LAYER-UPDATES")
    print("=" * 150)
    print(
        "rank layer finalWrongHarm      95%CI            robust% "
        "singleWorst%  W-C bestCompGap    gap95%CI"
    )

    for r in ranked[:n]:
        print(
            f"{int(r['rank']):>3d}  "
            f"L{int(r['layer']):02d}   "
            f"{float(r['mean_harm_to_final_generated_wrong']):+8.4f}   "
            f"[{float(r['final_wrong_harm_ci95_lo']):+7.4f},"
            f"{float(r['final_wrong_harm_ci95_hi']):+7.4f}]   "
            f"{float(r['robust_harmful_rate_on_wrong']):.3f}     "
            f"{float(r['fraction_as_single_worst_layer']):.3f}       "
            f"{float(r['wrong_minus_correct_best_comp_harm_gap']):+8.4f}   "
            f"[{float(r['best_comp_harm_gap_ci95_lo']):+7.4f},"
            f"{float(r['best_comp_harm_gap_ci95_hi']):+7.4f}]"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    direction_dir = Path(args.direction_dir)
    out_dir = Path(args.output_dir)

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_assets(direction_dir)
    selected_layers = parse_layers(
        args.layers,
        assets["n_layers"],
    )

    wanted_groups = {
        x.strip().lower()
        for x in args.generation_groups.split(",")
        if x.strip()
    }

    rows = build_per_sample_rows(
        assets=assets,
        selected_layers=selected_layers,
        split=args.split,
        wanted_groups=wanted_groups,
        robust_threshold=args.robust_threshold,
    )

    if not rows:
        raise RuntimeError("No samples matched requested split/groups.")

    sample_ids = sorted({int(r["sid"]) for r in rows})
    group_counts = Counter(
        next(
            r["generation_group"]
            for r in rows
            if int(r["sid"]) == sid
        )
        for sid in sample_ids
    )

    print(
        f"[data] samples={len(sample_ids)}, split={args.split}, "
        f"groups={dict(group_counts)}, layers={selected_layers}"
    )

    group_summary = summarize_all_group(
        rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    wrong_summary = summarize_wrong_final_relation(
        rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )

    (
        per_wrong_sample,
        worst_rows,
        first_rows,
    ) = worst_layer_stats(
        rows,
        selected_layers,
    )

    ranked = rank_harmful_layers(
        wrong_summary,
        group_summary,
        worst_rows,
    )

    write_csv(
        out_dir / "per_sample_layer_update.csv",
        rows,
    )
    write_csv(
        out_dir / "layer_update_summary.csv",
        group_summary,
    )
    write_csv(
        out_dir / "wrong_final_relation_summary.csv",
        wrong_summary,
    )
    write_csv(
        out_dir / "per_wrong_sample_worst_layer.csv",
        per_wrong_sample,
    )
    write_csv(
        out_dir / "worst_layer_frequency.csv",
        worst_rows,
    )
    write_csv(
        out_dir / "first_robust_harmful_layer.csv",
        first_rows,
    )
    write_csv(
        out_dir / "top_harmful_layers.csv",
        ranked,
    )

    print_wrong_summary(wrong_summary)
    print_group_summary(group_summary)
    print_top(ranked)

    meta = {
        "experiment": "layer-update Direction harm scan",
        "direction_dir": str(direction_dir),
        "split": args.split,
        "generation_groups": sorted(wanted_groups),
        "n_samples": len(sample_ids),
        "group_counts": dict(group_counts),
        "selected_target_layers": selected_layers,
        "correctness_definition":
            "cached actual model.generate() grouping",
        "primary_error_metric":
            "delta_r_l dot unit(mu_GT,l - mu_finalGeneratedWrong,l)",
        "primary_harm_metric":
            "negative of primary error metric",
        "robust_harm_definition":
            "current-layer-axis update effect < -threshold AND "
            "previous-layer-axis update effect < -threshold",
        "important_note":
            "State-margin change uses different codebooks at adjacent layers "
            "and is diagnostic only. Pure update attribution should rely on "
            "the update projections, especially agreement between current and "
            "previous axes.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "per_sample_layer_update.csv",
        "layer_update_summary.csv",
        "wrong_final_relation_summary.csv",
        "per_wrong_sample_worst_layer.csv",
        "worst_layer_frequency.csv",
        "first_robust_harmful_layer.csv",
        "top_harmful_layers.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
