#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_update_spatial_direction_polarity_generation_v1.py

Question
========
For each actual REAL block update on the fixed global token scaffold,

    a_{i,L,p} = h^REAL_{i,L,p} - h^REAL_{i,L-1,p},

can we decide whether it should be amplified or cancelled by asking:

    "What spatial direction does this update itself carry?"

instead of using an answer-gradient?

This implements the mechanism proposed in the current experiment:

    actual update a_{L,p}
        -> four spatial projections
           s_left, s_right, s_above, s_below

    independent Direction-Head sample belief
        -> r_hat_sample

    compare update direction with r_hat_sample
        -> predicted POSITIVE / NEGATIVE sign
        -> actual generation intervention

Crucially, the NON-ORACLE conditions below do NOT use:
    * GT relation to choose a sign;
    * GT answer gradients;
    * candidate-answer gradients at all.

GT/oracle sign is loaded only for evaluation and the optional oracle ceiling.

----------------------------------------------------------------------
1. Building residual-space spatial directions
----------------------------------------------------------------------

The existing Synthetic-400 Direction-Head selector fits, for every attention
head, four source-only directions in PRE-W_O head space:

    d_{L,h,r} in R^{head_dim}

from:
    <direction-selector-dir>/synthetic_fitted_direction_codebook.npz

For an attention head h at decoder block L, map the head direction through its
actual output projection slice:

    d^res_{L,h,r}
        = W_O[L][:, head_slice(h)] d_{L,h,r}

because PyTorch Linear applies y = x W^T.

Normalize each projected head direction, select the strongest source-OOF
Direction Heads WITHIN EACH LAYER, then aggregate them into one residual-space
direction per layer/relation:

    d^res_{L,r}
        = normalize( sum_h w_h normalize(d^res_{L,h,r}) )

Default:
    top 3 heads per layer by Synthetic source OOF accuracy
    w_h = max(source_oof_accuracy - 0.25, 0)

Thus the direction bank is source-only; COCO GT is not used to build it.

----------------------------------------------------------------------
2. Four projections of every actual update
----------------------------------------------------------------------

Normalize the actual block update:

    a_hat = a_REAL / ||a_REAL||

Then:

    s_r = <a_hat, d^res_{L,r}>

for r in {left,right,above,below}.

We test TWO direct polarity rules.

A) FOUR-WAY consistency

    B_spatial4
        = s_{r_hat} - max_{r != r_hat} s_r

    B_spatial4 > 0  -> update itself is most compatible with sample belief
                        => amplify +alpha*a_REAL

    B_spatial4 < 0  -> some other relation is more compatible
                        => cancel -alpha*a_REAL

B) AXIS consistency

Use only the opposite relation on the same spatial axis:

    opposite(left)=right
    opposite(right)=left
    opposite(above)=below
    opposite(below)=above

    B_axis
        = s_{r_hat} - s_{opposite(r_hat)}

This asks a cleaner LR / UD question and does not penalize a horizontal update
for also having a large vertical projection.

----------------------------------------------------------------------
3. Sample-level spatial belief
----------------------------------------------------------------------

r_hat_sample comes from the already-computed frozen Synthetic-400 Direction-Head
selector, via:

    <polarity-dir>/sample_relation_routes.csv
        route_direction_head

This is the same independent sample-level spatial belief that previously gave
strong non-oracle polarity prediction.

----------------------------------------------------------------------
4. Generation conditions
----------------------------------------------------------------------

spatial4_signed
axis_signed
    Apply the direct spatial sign to every fixed-scaffold update.

spatial4_thr_<t>
axis_thr_<t>
    Per-UPDATE abstention:
      if |B_spatial| <= t, leave this update untouched.
    This is different from the previous sample-level Direction-Head confidence
    threshold and directly measures confidence in WHAT THIS UPDATE CARRIES.

spatial4_disagree
axis_disagree
    Apply only to samples where:
        Direction-Head relation != baseline generated relation.
    This is a fully non-oracle repair trigger.

spatial4_disagree_thr_<t>
axis_disagree_thr_<t>
    Combine sample disagreement trigger + update-level spatial abstention.

oracle_signed_reference
    Optional GT-gradient sign ceiling, loaded from the previous polarity
    diagnostic. Explicitly ORACLE.

----------------------------------------------------------------------
Important alignment caveat
----------------------------------------------------------------------

a_REAL[L] is the net residual-stream change across decoder block L, while the
spatial bank is built from the pre-W_O attention heads INSIDE decoder block L
and mapped through W_O.  Therefore the comparison is:

    net block update at L
        versus
    attention-derived spatial direction basis at L.

This is an interpretable geometry test, not a claim that the entire block update
was produced only by those heads.  MLP can transform/reverse the attention
contribution inside the same block.

----------------------------------------------------------------------
Recommended N80 run
----------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python -u \
  eval_update_spatial_direction_polarity_generation_v1.py \
  --model qwen-3b \
  --global-run-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --polarity-dir output/qwen3b_global_fixed_polarity_diag_n80_v1 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --direction-selector-dir output/qwen3b_direction_selector_syn400_to_coco440 \
  --update-layers 20-26 \
  --heads-per-layer 3 \
  --head-weighting reliability \
  --scale 0.5 \
  --projection-thresholds 0.02,0.05 \
  --include-oracle-reference \
  --output-dir output/qwen3b_update_spatial_direction_generation_n80_v1 \
  --overwrite

Main outputs
============
residual_spatial_direction_bank.npz
residual_spatial_direction_bank_heads.csv

per_update_spatial_projection.csv
    s_left/s_right/s_above/s_below
    B_spatial4 / B_axis
    predicted signs
    oracle sign only for evaluation

sign_summary.csv
sign_by_layer.csv
sign_by_slot_layer.csv
projection_threshold_summary.csv

generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
condition_trigger_summary.csv

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
import re
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import analyze_coco_head_object_residual_direction_probe_v1 as dh
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
EPS = 1e-12


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

    p.add_argument("--global-run-dir", required=True)
    p.add_argument("--polarity-dir", required=True)
    p.add_argument("--prior-real-update-dir", required=True)

    p.add_argument(
        "--direction-selector-dir",
        required=True,
        help=(
            "Output directory from "
            "eval_direction_head_selector_synthetic400_to_coco440_v2.py; "
            "must contain synthetic_fitted_direction_codebook.npz and "
            "source_oof_head_reliability.csv."
        ),
    )

    p.add_argument("--update-layers", default="20-26")
    p.add_argument(
        "--heads-per-layer",
        type=int,
        default=3,
        help="Top source-OOF Direction Heads used to build each layer direction bank.",
    )
    p.add_argument(
        "--head-weighting",
        default="reliability",
        choices=["equal", "reliability"],
        help=(
            "equal: equal average after W_O projection. "
            "reliability: weight by max(source_oof_accuracy-0.25,0)."
        ),
    )

    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument(
        "--projection-thresholds",
        default="0.02,0.05",
        help="Per-update |spatial margin| abstention thresholds used in generation.",
    )
    p.add_argument("--include-oracle-reference", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=6)

    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all samples present in polarity diagnostic.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    out = set()
    for x in str(text).split(","):
        x = x.strip().upper().replace("L", "")
        if not x:
            continue
        if "-" in x:
            a, b = x.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(x))
    return sorted(out)


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "bottom": "below",
    }.get(s, s)


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def sign3(x, threshold=0.0):
    x = float(x)
    t = float(threshold)
    if x > t:
        return 1
    if x < -t:
        return -1
    return 0


def threshold_tag(x: float):
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def weighted_accuracy(correct, weights):
    c = np.asarray(correct, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(c) & np.isfinite(w) & (w >= 0)
    if not np.any(ok):
        return float("nan")
    den = float(w[ok].sum())
    if den <= EPS:
        return float("nan")
    return float(np.sum(c[ok] * w[ok]) / den)


# =============================================================================
# Existing run inputs
# =============================================================================

def load_run_inputs(
    global_dir: Path,
    polarity_dir: Path,
    update_layers: List[int],
):
    resolved_path = global_dir / "resolved_global_slots_per_sample.csv"
    signs_path = polarity_dir / "per_update_sign_predictions.csv"
    routes_path = polarity_dir / "sample_relation_routes.csv"

    for p in (resolved_path, signs_path, routes_path):
        if not p.exists():
            raise FileNotFoundError(p)

    resolved = pd.read_csv(resolved_path)
    signs = pd.read_csv(signs_path)
    routes = pd.read_csv(routes_path)

    resolved["sid"] = pd.to_numeric(resolved["sid"], errors="raise").astype(int)
    resolved["position"] = pd.to_numeric(
        resolved["position"], errors="coerce"
    ).fillna(-1).astype(int)

    if "resolved" in resolved.columns:
        if resolved["resolved"].dtype != bool:
            resolved["resolved"] = (
                resolved["resolved"].astype(str).str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )
        resolved = resolved[resolved["resolved"]].copy()

    resolved = resolved[resolved["position"] >= 0].copy()

    signs["sid"] = pd.to_numeric(signs["sid"], errors="raise").astype(int)
    signs["update_layer"] = pd.to_numeric(
        signs["update_layer"], errors="raise"
    ).astype(int)
    signs["position"] = pd.to_numeric(
        signs["position"], errors="raise"
    ).astype(int)
    signs = signs[signs["update_layer"].isin(set(update_layers))].copy()

    for c in ("oracle_sign",):
        if c in signs.columns:
            signs[c] = pd.to_numeric(
                signs[c], errors="coerce"
            ).fillna(0).astype(int)

    signs["gt"] = signs["gt"].map(canon_rel)

    routes["sid"] = pd.to_numeric(routes["sid"], errors="raise").astype(int)
    routes["gt"] = routes["gt"].map(canon_rel)
    routes["baseline_prediction"] = routes["baseline_prediction"].map(canon_rel)
    routes["route_direction_head"] = routes["route_direction_head"].map(canon_rel)

    for c in ("selector_margin", "selector_confidence"):
        routes[c] = pd.to_numeric(routes[c], errors="coerce")

    return resolved, signs, routes


def build_specs(resolved_sid: pd.DataFrame, max_layer: int):
    specs = []
    seen = set()

    sort_cols = [c for c in ("global_rank", "position") if c in resolved_sid.columns]
    x = resolved_sid.sort_values(sort_cols) if sort_cols else resolved_sid

    for r in x.itertuples():
        p = int(r.position)
        if p in seen:
            continue
        seen.add(p)

        specs.append(
            {
                "real_position": p,
                "max_target_layer": int(max_layer),
                "min_target_layer": int(max_layer),
                "best_rank": int(getattr(r, "global_rank", len(specs) + 1)),
                "token": str(getattr(r, "token", "")),
                "category": str(getattr(r, "category", "")),
                "broad_category": str(getattr(r, "broad_category", "")),
                "canonical_slot": str(getattr(r, "canonical_slot", "")),
                "global_rank": int(getattr(r, "global_rank", len(specs) + 1)),
            }
        )

    return specs


# =============================================================================
# Synthetic Direction-Head codebook -> residual-space direction bank
# =============================================================================

def load_direction_source(selector_dir: Path):
    codebook_path = selector_dir / "synthetic_fitted_direction_codebook.npz"
    reliability_path = selector_dir / "source_oof_head_reliability.csv"

    if not codebook_path.exists():
        raise FileNotFoundError(codebook_path)
    if not reliability_path.exists():
        raise FileNotFoundError(reliability_path)

    z = np.load(codebook_path, allow_pickle=True)
    if "directions" not in z.files:
        raise RuntimeError(
            f"{codebook_path} missing directions; keys={z.files}"
        )

    directions = np.asarray(z["directions"], dtype=np.float32)

    if "relations" in z.files:
        relations = [canon_rel(x) for x in z["relations"].tolist()]
    else:
        relations = list(REL)

    if set(relations) != set(REL):
        raise RuntimeError(f"Unexpected codebook relations: {relations}")

    reliability = pd.read_csv(reliability_path)
    for c in ("layer", "head"):
        reliability[c] = pd.to_numeric(
            reliability[c], errors="raise"
        ).astype(int)

    if "source_oof_accuracy" not in reliability.columns:
        raise RuntimeError(
            f"{reliability_path} missing source_oof_accuracy"
        )

    reliability["source_oof_accuracy"] = pd.to_numeric(
        reliability["source_oof_accuracy"], errors="raise"
    ).astype(float)

    if "weight_over_chance" not in reliability.columns:
        reliability["weight_over_chance"] = np.maximum(
            reliability["source_oof_accuracy"] - 0.25,
            0.0,
        )

    return directions, relations, reliability, codebook_path, reliability_path


def resolve_o_proj_for_layer(decoder_layers, L):
    attn = dh.resolve_self_attention(decoder_layers[int(L)])
    return dh.resolve_o_proj(attn)


def build_residual_direction_bank(
    *,
    model,
    decoder_layers,
    directions,
    codebook_relations,
    reliability,
    update_layers,
    heads_per_layer,
    weighting,
):
    """
    Returns:
      bank[L][relation] -> unit residual vector [hidden]
      head_rows          -> metadata for selected heads
    """
    rel_to_src_idx = {
        canon_rel(r): i
        for i, r in enumerate(codebook_relations)
    }

    # infer H,D from source codebook
    if directions.ndim != 4:
        raise RuntimeError(
            f"Expected directions [L,H,R,D], got {directions.shape}"
        )

    n_code_layers, n_heads, n_rel, head_dim = directions.shape
    bank = {}
    head_rows = []

    for L in update_layers:
        if not (0 <= int(L) < n_code_layers):
            raise RuntimeError(
                f"L{L} outside source codebook layer range 0..{n_code_layers-1}"
            )

        relL = reliability[reliability["layer"] == int(L)].copy()
        relL = relL.sort_values(
            ["source_oof_accuracy", "head"],
            ascending=[False, True],
        )

        if int(heads_per_layer) > 0:
            relL = relL.head(int(heads_per_layer))

        if len(relL) == 0:
            raise RuntimeError(
                f"No source reliability heads available at L{L}"
            )

        o_proj = resolve_o_proj_for_layer(decoder_layers, L)
        W = o_proj.weight.detach().float().cpu().numpy().astype(np.float32)

        if W.ndim != 2:
            raise RuntimeError(
                f"L{L} o_proj weight expected 2D, got {W.shape}"
            )

        # PyTorch Linear: y = x @ W.T, W:[out,in]
        expected_in = int(n_heads * head_dim)
        if int(W.shape[1]) != expected_in:
            raise RuntimeError(
                f"L{L}: o_proj input dim {W.shape[1]} != "
                f"codebook H*D={n_heads}*{head_dim}={expected_in}"
            )

        selected = []
        for rr in relL.itertuples():
            h = int(rr.head)
            if not (0 <= h < n_heads):
                continue

            acc = float(rr.source_oof_accuracy)
            if weighting == "equal":
                w = 1.0
            else:
                w = max(acc - 0.25, 0.0)
                if w <= 0:
                    # keep a tiny nonzero fallback rather than silently deleting
                    w = 1e-6

            selected.append((h, acc, w))

        if not selected:
            raise RuntimeError(f"L{L}: no valid selected heads")

        bank[L] = {}

        for r in REL:
            ri = rel_to_src_idx[r]
            vecs = []
            ws = []

            for h, acc, w in selected:
                d_head = directions[int(L), h, ri].astype(np.float32)
                d_head = normalize_np(d_head)

                start = h * head_dim
                stop = start + head_dim
                W_h = W[:, start:stop]  # [hidden, head_dim]

                d_res = (W_h @ d_head).astype(np.float32)
                d_res = normalize_np(d_res)

                if float(np.linalg.norm(d_res)) <= EPS:
                    continue

                vecs.append(d_res)
                ws.append(float(w))

                head_rows.append(
                    {
                        "layer": int(L),
                        "head": int(h),
                        "head_name": f"L{int(L)}H{int(h):02d}",
                        "relation": r,
                        "source_oof_accuracy": float(acc),
                        "aggregation_weight": float(w),
                        "prewo_direction_norm": float(
                            np.linalg.norm(d_head)
                        ),
                        "postwo_direction_norm_before_normalize": float(
                            np.linalg.norm(W_h @ d_head)
                        ),
                    }
                )

            if not vecs:
                raise RuntimeError(
                    f"L{L}/{r}: no nonzero projected head directions"
                )

            V = np.stack(vecs, axis=0)
            wv = np.asarray(ws, dtype=np.float32)
            agg = np.sum(V * wv[:, None], axis=0)
            agg = normalize_np(agg)

            if float(np.linalg.norm(agg)) <= EPS:
                raise RuntimeError(
                    f"L{L}/{r}: aggregate residual direction collapsed to zero"
                )

            bank[L][r] = agg

    return bank, pd.DataFrame(head_rows)


def save_direction_bank(path: Path, bank):
    payload = {}
    for L, rr in bank.items():
        for r, v in rr.items():
            payload[f"L{int(L)}_{r}"] = np.asarray(v, np.float32)
    np.savez_compressed(path, **payload)


# =============================================================================
# Update spatial projection / sign
# =============================================================================

def project_update(a_real, layer_bank):
    a_hat = normalize_np(a_real)
    return {
        r: float(np.dot(a_hat, layer_bank[r]))
        for r in REL
    }


def spatial_margins(scores, route):
    route = canon_rel(route)
    if route not in REL:
        return float("nan"), float("nan"), ""

    strongest_other = max(
        (r for r in REL if r != route),
        key=lambda r: float(scores[r]),
    )

    fourway = float(
        scores[route] - scores[strongest_other]
    )

    opp = OPPOSITE[route]
    axis = float(
        scores[route] - scores[opp]
    )

    return fourway, axis, strongest_other


def attach_spatial_scores(
    *,
    entries,
    layer_bank,
    route,
    oracle_lookup,
    spec_meta,
):
    scored = []

    for e in entries:
        L = int(e["update_layer"])
        p = int(e["real_position"])
        a = np.asarray(e["_real_update"], np.float32)

        if L not in layer_bank:
            continue

        s = project_update(a, layer_bank[L])
        b4, bax, strongest_other = spatial_margins(s, route)

        key = (L, p)
        oo = oracle_lookup.get(key, {})

        sm = spec_meta.get(p, {})

        row = dict(e)
        row["_spatial4_score"] = float(b4)
        row["_axis_score"] = float(bax)

        row["canonical_slot"] = str(
            sm.get("canonical_slot", "")
        )
        row["global_rank"] = int(
            sm.get("global_rank", sm.get("best_rank", -1))
        )

        row["route_direction_head"] = route
        row["route_opposite"] = (
            OPPOSITE[route] if route in REL else ""
        )
        row["strongest_spatial_other"] = strongest_other

        for r in REL:
            row[f"spatial_projection_{r}"] = float(s[r])

        row["spatial4_margin"] = float(b4)
        row["axis_margin"] = float(bax)
        row["pred_sign_spatial4"] = sign3(b4, 0.0)
        row["pred_sign_axis"] = sign3(bax, 0.0)

        row["oracle_sign"] = int(oo.get("oracle_sign", 0))
        row["oracle_B"] = float(oo.get("oracle_B", np.nan))
        row["oracle_abs_B"] = abs(row["oracle_B"]) if np.isfinite(row["oracle_B"]) else np.nan
        row["gradient_direction_sign"] = int(
            oo.get("pred_sign_direction_head", 0)
        )
        row["gradient_direction_Bhat"] = float(
            oo.get("Bhat_direction_head", np.nan)
        )

        scored.append(row)

    return scored


# =============================================================================
# Patch / generation
# =============================================================================

def entries_for_spatial_patch(
    scored,
    score_key,
):
    out = []
    for e in scored:
        ee = dict(e)
        ee["real_update_decision_score"] = float(e[score_key])
        out.append(ee)
    return out


def condition_specs(projection_thresholds, include_oracle):
    conds = [
        ("spatial4_signed", "spatial4", 0.0, False),
        ("axis_signed", "axis", 0.0, False),
        ("spatial4_disagree", "spatial4", 0.0, True),
        ("axis_disagree", "axis", 0.0, True),
    ]

    for t in projection_thresholds:
        tag = threshold_tag(t)
        conds.extend(
            [
                (f"spatial4_thr_{tag}", "spatial4", float(t), False),
                (f"axis_thr_{tag}", "axis", float(t), False),
                (
                    f"spatial4_disagree_thr_{tag}",
                    "spatial4",
                    float(t),
                    True,
                ),
                (
                    f"axis_disagree_thr_{tag}",
                    "axis",
                    float(t),
                    True,
                ),
            ]
        )

    if include_oracle:
        conds.append(
            ("oracle_signed_reference", "oracle", 0.0, False)
        )

    return conds


def build_oracle_entries(entries, oracle_lookup):
    out = []
    missing = []

    for e in entries:
        key = (int(e["update_layer"]), int(e["real_position"]))
        if key not in oracle_lookup:
            missing.append(key)
            continue

        s = int(oracle_lookup[key].get("oracle_sign", 0))
        ee = dict(e)
        ee["real_update_decision_score"] = float(s)
        out.append(ee)

    return out, missing


def summarize_trigger(gen_df):
    rows = []
    x = gen_df[gen_df["condition"] != "baseline"].copy()

    for cond, g in x.groupby("condition"):
        rows.append(
            {
                "condition": cond,
                "N": int(g["sid"].nunique()),
                "triggered_samples": int(
                    g.loc[g["triggered"], "sid"].nunique()
                ),
                "trigger_rate": float(
                    g.groupby("sid")["triggered"].first().mean()
                ),
                "mean_patched_updates": float(
                    g["n_patched_updates"].mean()
                ),
                "mean_positive_updates": float(
                    g["n_positive_updates"].mean()
                ),
                "mean_negative_updates": float(
                    g["n_negative_updates"].mean()
                ),
                "mean_neutral_updates": float(
                    g["n_neutral_updates"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Sign diagnostics
# =============================================================================

def one_sign_summary(df, pred_col, method, cohort):
    x = df[
        (df["oracle_sign"].isin([-1, 1]))
        & (df[pred_col].isin([-1, 1]))
    ].copy()

    if cohort == "baseline_wrong":
        x = x[~x["baseline_correct"]]
    elif cohort == "baseline_correct":
        x = x[x["baseline_correct"]]

    if not len(x):
        return {
            "method": method,
            "cohort": cohort,
            "N_updates": 0,
            "N_samples": 0,
            "sign_accuracy": np.nan,
            "weighted_sign_accuracy": np.nan,
            "oracle_positive_fraction": np.nan,
            "pred_positive_fraction": np.nan,
        }

    correct = (
        x[pred_col].to_numpy(int)
        == x["oracle_sign"].to_numpy(int)
    )

    return {
        "method": method,
        "cohort": cohort,
        "N_updates": int(len(x)),
        "N_samples": int(x["sid"].nunique()),
        "sign_accuracy": float(np.mean(correct)),
        "weighted_sign_accuracy": weighted_accuracy(
            correct,
            x["oracle_abs_B"].to_numpy(float),
        ),
        "oracle_positive_fraction": float(
            np.mean(x["oracle_sign"].to_numpy(int) > 0)
        ),
        "pred_positive_fraction": float(
            np.mean(x[pred_col].to_numpy(int) > 0)
        ),
    }


def summarize_signs(df):
    rows = []

    methods = [
        ("spatial4", "pred_sign_spatial4"),
        ("axis", "pred_sign_axis"),
        ("previous_direction_gradient", "gradient_direction_sign"),
    ]

    for method, col in methods:
        for cohort in ("all", "baseline_wrong", "baseline_correct"):
            rows.append(
                one_sign_summary(df, col, method, cohort)
            )

    return pd.DataFrame(rows)


def summarize_sign_by_layer(df):
    rows = []
    for L, g in df.groupby("update_layer"):
        for method, col in [
            ("spatial4", "pred_sign_spatial4"),
            ("axis", "pred_sign_axis"),
            ("previous_direction_gradient", "gradient_direction_sign"),
        ]:
            for cohort_name, h in [
                ("all", g),
                ("baseline_wrong", g[~g["baseline_correct"]]),
                ("baseline_correct", g[g["baseline_correct"]]),
            ]:
                row = one_sign_summary(
                    h,
                    col,
                    method,
                    "all",
                )
                row["cohort"] = cohort_name
                row["update_layer"] = int(L)
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_sign_by_slot_layer(df):
    rows = []

    for (slot, rank, L), g in df.groupby(
        ["canonical_slot", "global_rank", "update_layer"]
    ):
        g = g[g["oracle_sign"].isin([-1, 1])]
        if not len(g):
            continue

        row = {
            "canonical_slot": slot,
            "global_rank": int(rank),
            "update_layer": int(L),
            "N": int(len(g)),
            "mean_abs_oracle_B": float(
                g["oracle_abs_B"].mean()
            ),
        }

        for name, col in [
            ("spatial4", "pred_sign_spatial4"),
            ("axis", "pred_sign_axis"),
            ("prev_grad", "gradient_direction_sign"),
        ]:
            v = g[g[col].isin([-1, 1])]
            if len(v):
                correct = (
                    v[col].to_numpy(int)
                    == v["oracle_sign"].to_numpy(int)
                )
                row[f"{name}_acc"] = float(np.mean(correct))
                row[f"{name}_weighted_acc"] = weighted_accuracy(
                    correct,
                    v["oracle_abs_B"].to_numpy(float),
                )
            else:
                row[f"{name}_acc"] = np.nan
                row[f"{name}_weighted_acc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_projection_thresholds(df, thresholds):
    rows = []

    for method, score_col in [
        ("spatial4", "spatial4_margin"),
        ("axis", "axis_margin"),
    ]:
        pred_col = (
            "pred_sign_spatial4"
            if method == "spatial4"
            else "pred_sign_axis"
        )

        for t in [0.0] + list(thresholds):
            for cohort_name, g in [
                ("all", df),
                ("baseline_wrong", df[~df["baseline_correct"]]),
                ("baseline_correct", df[df["baseline_correct"]]),
            ]:
                valid = g[g["oracle_sign"].isin([-1, 1])].copy()
                covered = valid[np.abs(valid[score_col]) > float(t)].copy()

                if len(covered):
                    pred = np.sign(
                        covered[score_col].to_numpy(float)
                    ).astype(int)
                    oracle = covered["oracle_sign"].to_numpy(int)
                    correct = pred == oracle
                    acc = float(np.mean(correct))
                    wacc = weighted_accuracy(
                        correct,
                        covered["oracle_abs_B"].to_numpy(float),
                    )
                    mass = float(covered["oracle_abs_B"].sum())
                else:
                    acc = np.nan
                    wacc = np.nan
                    mass = 0.0

                total_mass = float(valid["oracle_abs_B"].sum())

                rows.append(
                    {
                        "method": method,
                        "threshold": float(t),
                        "cohort": cohort_name,
                        "N_updates_total": int(len(valid)),
                        "N_updates_covered": int(len(covered)),
                        "update_coverage": (
                            len(covered) / len(valid)
                            if len(valid) else np.nan
                        ),
                        "sign_accuracy_covered": acc,
                        "weighted_sign_accuracy_covered": wacc,
                        "oracle_abs_mass_coverage": (
                            mass / total_mass
                            if total_mass > EPS else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    update_layers = parse_ints(a.update_layers)
    projection_thresholds = parse_floats(
        a.projection_thresholds
    )

    if not update_layers:
        raise ValueError("No update layers")

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    global_dir = Path(a.global_run_dir)
    polarity_dir = Path(a.polarity_dir)
    prior_dir = Path(a.prior_real_update_dir)
    selector_dir = Path(a.direction_selector_dir)

    resolved, old_signs, routes = load_run_inputs(
        global_dir,
        polarity_dir,
        update_layers,
    )

    directions, codebook_relations, reliability, codebook_path, reliability_path = (
        load_direction_source(selector_dir)
    )

    # Common cohort.
    sids = sorted(
        set(resolved["sid"])
        & set(old_signs["sid"])
        & set(routes["sid"])
    )

    prior_cohort, _, prior_metadata, _ = l26.load_prior_run(
        prior_dir
    )
    prior_cohort = prior_cohort[
        prior_cohort["sid"].isin(sids)
    ].copy()

    if int(a.eval_max_samples) > 0:
        prior_cohort = l26.stratified_cap_df(
            prior_cohort,
            int(a.eval_max_samples),
            int(a.seed) + 311,
        )

    sids = sorted(
        prior_cohort["sid"].astype(int).tolist()
    )

    resolved = resolved[resolved["sid"].isin(sids)].copy()
    old_signs = old_signs[old_signs["sid"].isin(sids)].copy()
    routes = routes[routes["sid"].isin(sids)].copy()

    two, eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        prior_cohort,
    )
    eval_meta = [
        m for m in eval_meta
        if int(m["sid"]) in set(sids)
    ]

    route_by_sid = routes.set_index("sid").to_dict("index")

    old_lookup_by_sid = {}
    for sid, g in old_signs.groupby("sid"):
        dd = {}
        for r in g.itertuples():
            dd[
                (int(r.update_layer), int(r.position))
            ] = {
                "oracle_sign": int(r.oracle_sign),
                "oracle_B": float(r.oracle_B),
                "pred_sign_direction_head": int(
                    getattr(r, "pred_sign_direction_head", 0)
                ),
                "Bhat_direction_head": float(
                    getattr(r, "Bhat_direction_head", np.nan)
                ),
            }
        old_lookup_by_sid[int(sid)] = dd

    model = processor = None
    projection_rows = []
    generation_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = (
            l26.load_model(a, two)
        )
        device = torch.device(a.device)

        # -------------------------------------------------------------
        # Build one source-only residual spatial direction bank.
        # -------------------------------------------------------------
        layer_bank, bank_head_rows = build_residual_direction_bank(
            model=model,
            decoder_layers=decoder_layers,
            directions=directions,
            codebook_relations=codebook_relations,
            reliability=reliability,
            update_layers=update_layers,
            heads_per_layer=int(a.heads_per_layer),
            weighting=a.head_weighting,
        )

        save_direction_bank(
            outdir / "residual_spatial_direction_bank.npz",
            layer_bank,
        )
        bank_head_rows.to_csv(
            outdir / "residual_spatial_direction_bank_heads.csv",
            index=False,
        )

        conditions = condition_specs(
            projection_thresholds,
            a.include_oracle_reference,
        )

        print("=" * 190)
        print("ACTUAL UPDATE -> SPATIAL DIRECTION CONSISTENCY -> GENERATION")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(eval_meta)}")
        print(f"update_layers={update_layers}")
        print(
            f"direction bank: top{a.heads_per_layer}/layer, "
            f"weighting={a.head_weighting}"
        )
        print(f"scale={a.scale}")
        print(f"projection_thresholds={projection_thresholds}")
        print()
        print("Selected source-only heads per layer:")
        show = (
            bank_head_rows[
                ["layer", "head_name", "source_oof_accuracy"]
            ]
            .drop_duplicates(["layer", "head_name"])
            .sort_values(["layer", "source_oof_accuracy"], ascending=[True, False])
        )
        print(show.to_string(index=False))
        print()

        for m in tqdm(
            eval_meta,
            desc="SPATIAL-DIRECTION generation",
        ):
            sid = int(m["sid"])
            real = None

            route_row = route_by_sid[sid]
            route = canon_rel(
                route_row["route_direction_head"]
            )
            baseline_pred = canon_rel(
                route_row["baseline_prediction"]
            )
            disagree = (
                route in REL
                and baseline_pred in REL
                and route != baseline_pred
            )

            generation_rows.append(
                {
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "scale": 0.0,
                    "prediction": m["baseline_prediction"],
                    "correct": bool(m["baseline_correct"]),
                    "triggered": False,
                    "route_direction_head": route,
                    "baseline_prediction": baseline_pred,
                    "route_disagrees_baseline": bool(disagree),
                    "n_patched_updates": 0,
                    "n_positive_updates": 0,
                    "n_negative_updates": 0,
                    "n_neutral_updates": 0,
                    "text": "",
                }
            )

            try:
                if route not in REL:
                    raise RuntimeError(
                        f"Invalid Direction-Head route: {route}"
                    )

                rs = resolved[
                    resolved["sid"] == sid
                ].copy()

                specs = build_specs(
                    rs,
                    max(update_layers),
                )
                if not specs:
                    raise RuntimeError(
                        "No resolved global scaffold positions"
                    )

                spec_meta = {
                    int(s["real_position"]): s
                    for s in specs
                }

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )

                capture_layers = sorted(
                    set(
                        update_layers
                        + [L - 1 for L in update_layers]
                    )
                )

                real_states = gate.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                entries = gate.build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=bool(m["baseline_correct"]),
                    specs=specs,
                    r2n={},
                    real_states=real_states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=False,
                )

                if not entries:
                    raise RuntimeError(
                        "No REAL updates on global scaffold"
                    )

                oracle_lookup = old_lookup_by_sid.get(
                    sid,
                    {},
                )

                scored = attach_spatial_scores(
                    entries=entries,
                    layer_bank=layer_bank,
                    route=route,
                    oracle_lookup=oracle_lookup,
                    spec_meta=spec_meta,
                )

                if not scored:
                    raise RuntimeError(
                        "No update received spatial projections"
                    )

                # Export projections, excluding private vectors.
                for e in scored:
                    export = {
                        k: v
                        for k, v in e.items()
                        if not k.startswith("_")
                    }
                    export["baseline_prediction"] = baseline_pred
                    export["route_disagrees_baseline"] = bool(disagree)
                    projection_rows.append(export)

                spatial4_entries = entries_for_spatial_patch(
                    scored,
                    "_spatial4_score",
                )
                axis_entries = entries_for_spatial_patch(
                    scored,
                    "_axis_score",
                )

                oracle_entries = []
                missing_oracle = []
                if a.include_oracle_reference:
                    oracle_entries, missing_oracle = build_oracle_entries(
                        entries,
                        oracle_lookup,
                    )
                    if missing_oracle:
                        raise RuntimeError(
                            f"Missing oracle signs for "
                            f"{len(missing_oracle)} updates, "
                            f"e.g. {missing_oracle[:5]}"
                        )

                # -----------------------------------------------------
                # Actual generation.
                # -----------------------------------------------------
                for cond_name, kind, threshold, disagree_only in conditions:
                    if kind == "oracle":
                        active = True
                        patch_entries = oracle_entries
                        patch_threshold = 0.0
                    else:
                        active = (
                            bool(disagree)
                            if disagree_only
                            else True
                        )
                        patch_entries = (
                            spatial4_entries
                            if kind == "spatial4"
                            else axis_entries
                        )
                        patch_threshold = float(threshold)

                    if active:
                        patch_map, counts = gate.build_real_update_patch_map(
                            patch_entries,
                            "real_signed",
                            float(a.scale),
                            float(patch_threshold),
                        )
                    else:
                        patch_map = {}
                        counts = {
                            "patched": 0,
                            "positive": 0,
                            "negative": 0,
                            "neutral": len(patch_entries),
                        }

                    # If sample-level disagreement trigger abstains, behavior
                    # is exactly baseline; reuse it to save GPU time.
                    if not active:
                        pred = m["baseline_prediction"]
                        text = ""
                    else:
                        pred, text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            max_new_tokens=a.max_new_tokens,
                        )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": m["gt"],
                            "condition": cond_name,
                            "scale": float(a.scale),
                            "prediction": pred,
                            "correct": pred == m["gt"],
                            "triggered": bool(active),
                            "route_direction_head": route,
                            "baseline_prediction": baseline_pred,
                            "route_disagrees_baseline": bool(disagree),
                            "n_patched_updates": int(counts["patched"]),
                            "n_positive_updates": int(counts["positive"]),
                            "n_negative_updates": int(counts["negative"]),
                            "n_neutral_updates": int(counts["neutral"]),
                            "text": text,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-80:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =====================================================================
        # Save + diagnostics
        # =====================================================================
        proj_df = pd.DataFrame(projection_rows)
        gen_df = pd.DataFrame(generation_rows)

        if not len(proj_df):
            raise RuntimeError("No projection rows produced")

        proj_df.to_csv(
            outdir / "per_update_spatial_projection.csv",
            index=False,
        )
        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )

        sign_summary = summarize_signs(proj_df)
        sign_layer = summarize_sign_by_layer(proj_df)
        sign_slot = summarize_sign_by_slot_layer(proj_df)
        threshold_summary = summarize_projection_thresholds(
            proj_df,
            projection_thresholds,
        )

        sign_summary.to_csv(
            outdir / "sign_summary.csv",
            index=False,
        )
        sign_layer.to_csv(
            outdir / "sign_by_layer.csv",
            index=False,
        )
        sign_slot.to_csv(
            outdir / "sign_by_slot_layer.csv",
            index=False,
        )
        threshold_summary.to_csv(
            outdir / "projection_threshold_summary.csv",
            index=False,
        )

        gen_summary = gate.summarize_generation(gen_df)
        gen_rel = gate.summarize_generation_by_relation(gen_df)
        trigger_summary = summarize_trigger(gen_df)

        gen_summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )
        gen_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )
        trigger_summary.to_csv(
            outdir / "condition_trigger_summary.csv",
            index=False,
        )

        # Relation belief quality for context.
        route_sample = routes[
            routes["sid"].isin(proj_df["sid"].unique())
        ].copy()

        route_acc_all = float(
            np.mean(
                route_sample["route_direction_head"]
                == route_sample["gt"]
            )
        )

        wrong = route_sample[
            ~route_sample["sid"].map(
                prior_cohort.set_index("sid")["baseline_correct"]
            ).astype(bool)
        ] if len(route_sample) else pd.DataFrame()

        route_acc_wrong = (
            float(
                np.mean(
                    wrong["route_direction_head"]
                    == wrong["gt"]
                )
            )
            if len(wrong) else np.nan
        )

        print("=" * 190)
        print("UPDATE SPATIAL DIRECTION -> POLARITY")
        print("=" * 190)
        print(
            f"N samples={proj_df['sid'].nunique()} | "
            f"N updates={len(proj_df)}"
        )
        print(
            f"Direction sample belief relation acc: "
            f"all={route_acc_all:.4f} | "
            f"baseline_wrong={route_acc_wrong:.4f}"
        )

        print("\nSIGN PREDICTION")
        print("-" * 190)
        print(
            sign_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nPROJECTION THRESHOLD SWEEP")
        print("-" * 190)
        print(
            threshold_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nACTUAL GENERATION")
        print("-" * 190)
        print(
            gen_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nTRIGGER / PATCH COVERAGE")
        print("-" * 190)
        print(
            trigger_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        report = [
            "=" * 190,
            "ACTUAL UPDATE -> SPATIAL DIRECTION CONSISTENCY -> GENERATION",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N samples={proj_df['sid'].nunique()}",
            f"N updates={len(proj_df)}",
            f"update layers={update_layers}",
            f"heads per layer={a.heads_per_layer}",
            f"head weighting={a.head_weighting}",
            f"scale={a.scale}",
            f"projection thresholds={projection_thresholds}",
            f"sample Direction-Head relation acc all={route_acc_all:.4f}",
            f"sample Direction-Head relation acc baseline_wrong={route_acc_wrong:.4f}",
            "",
            "SIGN PREDICTION",
            sign_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "PROJECTION THRESHOLD SWEEP",
            threshold_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "GENERATION",
            gen_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "Interpretation:",
            "  spatial4 asks whether the actual update is most aligned with the",
            "  sample's independently predicted relation among all four directions.",
            "",
            "  axis asks only whether the update favors the predicted relation over",
            "  its opposite on the same LR/UD axis.",
            "",
            "  If either direct spatial rule predicts oracle sign and improves",
            "  generation, then HOW can be interpreted as whether each block writes",
            "  spatial information consistent with the sample-level spatial belief,",
            "  without using answer gradients.",
            "",
            "  Compare against previous_direction_gradient in sign_summary. That",
            "  previous rule still used candidate-answer gradients after routing;",
            "  spatial4/axis do not.",
            "",
            "Caveats:",
            "  * Current WHERE may still be GT-writer-discovered if the global run",
            "    used split_mode=all_data.",
            "  * The residual direction basis comes from attention heads mapped through",
            "    W_O; a_REAL is the full block update including MLP.",
        ]

        report_text = "\n".join(report) + "\n"
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        metadata = {
            "script": "eval_update_spatial_direction_polarity_generation_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "global_run_dir": str(global_dir),
            "polarity_dir": str(polarity_dir),
            "prior_real_update_dir": str(prior_dir),
            "direction_selector_dir": str(selector_dir),
            "synthetic_codebook": str(codebook_path),
            "source_oof_reliability": str(reliability_path),
            "update_layers": update_layers,
            "heads_per_layer": int(a.heads_per_layer),
            "head_weighting": a.head_weighting,
            "scale": float(a.scale),
            "projection_thresholds": projection_thresholds,
            "spatial_projection_definition": (
                "cosine(actual REAL block update, source-only residual-space "
                "relation direction obtained by pre-W_O Synthetic Direction "
                "Head direction -> same head W_O slice -> per-layer aggregation)"
            ),
            "spatial4_sign_definition": (
                "sign(s_route - max_{r!=route} s_r)"
            ),
            "axis_sign_definition": (
                "sign(s_route - s_opposite(route))"
            ),
            "sample_relation_route": (
                "route_direction_head from prior non-oracle polarity diagnostic"
            ),
            "gt_usage_nonoracle_conditions": (
                "GT/oracle sign used only for diagnostics and final accuracy; "
                "not used to choose non-oracle sign or trigger"
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
