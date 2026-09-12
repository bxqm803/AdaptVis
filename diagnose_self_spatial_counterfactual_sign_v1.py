#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_self_spatial_counterfactual_sign_v1.py

Goal
====
Find a NO-RELATION-SELECTOR signal for whether a realized causal-token update
is spatially "reinforcing" or "opposing", using only the model's own
subject/reference spatial state.

NO left/right/above/below label is used to DEFINE the proposed sign.
NO Synthetic relation codebook is used.
NO final-answer gradient is used to DEFINE the proposed sign.

GT/oracle B_REAL is loaded ONLY AFTERWARD for evaluation.

Core construction
=================
At an anchor layer K, define the model's image-conditioned object-relation
state

    r_real = h_sub^REAL - h_ref^REAL
    r_no   = h_sub^NOIMAGE - h_ref^NOIMAGE
    s_RN   = r_real - r_no

We keep the REAL pair midpoint fixed and construct three trajectories by
editing ONLY the subject/reference difference at block K:

    (+) clean REAL:
        relation difference = r_real = r_no + s_RN

    (0) spatial-null:
        relation difference = r_no

    (-) spatial-reversed:
        relation difference = r_no - s_RN

Equivalently, with beta=1:

    zero:
        h_sub <- h_sub - 0.5*s_RN
        h_ref <- h_ref + 0.5*s_RN

    minus:
        h_sub <- h_sub - 1.0*s_RN
        h_ref <- h_ref + 1.0*s_RN

For beta != 1 the same perturbation is scaled by beta.

The intervention does NOT need to know whether the sample is left/right/etc.
It only uses the sample's own REAL-vs-NoImage object-relation residual.

Self-spatial effect at a downstream causal token
================================================
For a downstream layer L > K and causal-token position p:

    h+_in  = h+_{L-1,p}
    h0_in  = h0_{L-1,p}
    h-_in  = h-_{L-1,p}

    h+_out = h+_{L,p}
    h0_out = h0_{L,p}
    h-_out = h-_{L,p}

Define the symmetric model-internal spatial effects

    d_in  = 0.5 * (h+_in  - h-_in)
    d_out = 0.5 * (h+_out - h-_out)

and the clean realized update

    a_clean = h+_out - h+_in.

Primary NO-ORACLE sign candidate:

    C_receiver_out = cosine(a_clean, d_out)

Interpretation:
    C > 0 : the realized update points along the direction that this sample's
            own spatial state makes this causal token move.
    C < 0 : the realized update opposes that self-spatial effect.

Secondary rules:
    C_receiver_in  = cosine(a_clean, d_in)
    C_receiver_mid = cosine(a_clean, normalize(d_in + d_out))
    C_spatial_gain = dot(d_out - d_in, d_in) / ||d_in||^2

The first rule works already at L=K+1.  receiver_in/gain can be undefined or
tiny there because the object-state perturbation has not yet reached a later
causal token at the INPUT of the first downstream block.

Why this is not the earlier tautology
=====================================
Do NOT use

    (a+ - a0) dot (a+ - a-)

as the sign.  Under a locally linear response it is nearly positive by
construction.  This script intentionally avoids that quantity.

The proposed sign compares the NATURAL clean update a_clean with a
COUNTERFACTUALLY transported self-spatial direction d_out/d_in.  Its sign is
not algebraically forced.

Counterfactual quality checks
=============================
For each token/layer:

    plus0_out  = h+_out - h0_out
    zeroMinus_out = h0_out - h-_out

We report:
    cf_linearity_cos_out = cosine(plus0_out, zeroMinus_out)

and
    cf_asymmetry_ratio_out =
        || 0.5*(h+_out+h-_out) - h0_out || / ||d_out||

A clean odd/symmetric spatial response should have high linearity cosine and
low asymmetry.

Evaluation against prior oracle B
=================================
The existing prior run provides

    B_REAL = a_clean dot grad(S_GT - S_comp)

ONLY as an evaluation label.

We report sign agreement for:
    all
    baseline_wrong
    baseline_correct
    high-linearity subsets

This tests whether the model's own spatial state can recover HOW polarity
without first selecting LEFT/RIGHT/ABOVE/BELOW.

Optional generation
===================
With --run-generation, use one chosen no-oracle rule (default receiver_out):

    predicted positive -> add +alpha*a_clean
    predicted negative -> add -alpha*a_clean

at downstream causal-token updates.  This still uses the existing oracle
causal-token WHERE scaffold, but the HOW sign itself uses no relation label,
no selector, and no answer gradient.

Recommended first run
=====================
Anchor L19 gives coverage over updates L20-26.  Anchor L23 tests the stronger
mid-layer spatial state but only downstream updates L24-26.

CUDA_VISIBLE_DEVICES=0 python -u diagnose_self_spatial_counterfactual_sign_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --anchor-layers 19,23 \
  --update-layers 20-26 \
  --max-samples 80 \
  --spatial-scale 1.0 \
  --output-dir output/qwen3b_self_spatial_cf_sign_n80_v1 \
  --overwrite

If the diagnostic is promising, add:

  --run-generation --gating-rule receiver_out --gating-scale 0.5

Main outputs
============
per_update_self_spatial.csv
sign_agreement_summary.csv
sign_agreement_by_layer.csv
counterfactual_quality_summary.csv
anchor_state_summary.csv
generation_per_sample.csv              (if --run-generation)
generation_summary.csv                 (if --run-generation)
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
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import analyze_real_update_module_head_sources_v1 as src
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_real_update_module_head_sources_v1.py.\n"
        "Run from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import eval_real_causal_token_update_gating_v1 as upd
except Exception as exc:
    raise SystemExit(
        "Could not import eval_real_causal_token_update_gating_v1.py.\n"
        "Run from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import analyze_coco_head_object_residual_direction_probe_v1 as dh
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_head_object_residual_direction_probe_v1.py.\n"
        "It is used only for the existing phrase-position locator.\n"
        f"{type(exc).__name__}: {exc}"
    )

EPS = 1e-12

RULES = (
    "receiver_out",
    "receiver_in",
    "receiver_mid",
    "spatial_gain",
)


# =============================================================================
# CLI
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

    p.add_argument(
        "--anchor-layers",
        default="19,23",
        help="Comma/range syntax, e.g. 19,23 or 19-23.",
    )
    p.add_argument(
        "--update-layers",
        default="20-26",
        help="Only downstream L > anchor are scored.",
    )
    p.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "last"],
        help="Pooling for multi-token subject/reference phrase states.",
    )
    p.add_argument(
        "--spatial-scale",
        type=float,
        default=1.0,
        help=(
            "beta for RN relation-state edit. beta=1 makes the zero branch "
            "use the NoImage relation difference and the minus branch reverse "
            "the RN component around NoImage."
        ),
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=80,
        help="0 = all samples from prior successful run.",
    )
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--behavior-threshold",
        type=float,
        default=1e-8,
        help="Threshold for prior oracle B_REAL evaluation sign.",
    )
    p.add_argument(
        "--self-score-threshold",
        type=float,
        default=1e-8,
        help="Threshold for proposed no-oracle score sign.",
    )
    p.add_argument(
        "--linearity-threshold",
        type=float,
        default=0.5,
        help="High-quality CF subset requires cf_linearity_cos_out >= this.",
    )
    p.add_argument(
        "--max-asymmetry-ratio",
        type=float,
        default=1.0,
        help="High-quality CF subset also requires asymmetry <= this.",
    )
    p.add_argument(
        "--min-spatial-effect-norm",
        type=float,
        default=1e-6,
        help="Ignore sign when d_out/d_in is effectively zero.",
    )

    p.add_argument(
        "--run-generation",
        action="store_true",
        help="Run no-oracle HOW signed gating using the chosen rule.",
    )
    p.add_argument(
        "--gating-rule",
        default="receiver_out",
        choices=RULES,
    )
    p.add_argument(
        "--gating-scale",
        type=float,
        default=0.5,
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sign_threshold(v, threshold):
    v = float(v)
    if not np.isfinite(v):
        return 0
    if v > threshold:
        return 1
    if v < -threshold:
        return -1
    return 0


def norm_np(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float32)))


def cosine_np(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = norm_np(a) * norm_np(b)
    if den <= EPS:
        return float("nan")
    return float(np.dot(a, b) / den)


def safe_mean(xs):
    x = np.asarray(list(xs), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def safe_weighted_accuracy(correct, weights):
    c = np.asarray(correct, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(c) & np.isfinite(w) & (w >= 0)
    if not np.any(ok):
        return float("nan")
    den = float(w[ok].sum())
    if den <= EPS:
        return float("nan")
    return float(np.sum(c[ok] * w[ok]) / den)


def pool_np(state, positions, pool):
    """
    state: [1,T,D] numpy array.
    Every phrase token receives the same perturbation later, so mean/last
    pooling both have an exactly controlled shift.
    """
    pos = list(map(int, positions))
    if not pos:
        raise RuntimeError("Empty phrase position list")
    x = np.asarray(state[0, pos, :], dtype=np.float32)
    if pool == "mean":
        return x.mean(axis=0)
    if pool == "last":
        return x[-1]
    raise ValueError(pool)


def tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def locate_positions(processor, batch, phrase):
    ids = [
        int(x)
        for x in batch["input_ids"][0].detach().cpu().tolist()
    ]
    return dh.locate_phrase_positions(
        tokenizer_of(processor),
        ids,
        str(phrase),
    )


# =============================================================================
# Patched forward with block-output capture
# =============================================================================

class PatchAndBlockCapture:
    """
    One prefill forward:
      * at patch_layer, edit selected block-output token states;
      * capture the MODIFIED output at patch_layer;
      * capture downstream block outputs.

    Decode steps are irrelevant because this function is only used for prompt
    state capture.
    """

    def __init__(
        self,
        decoder_layers,
        capture_layers,
        patch_layer,
        pos_map,
        prompt_len,
    ):
        self.decoder_layers = decoder_layers
        self.capture_layers = sorted(set(map(int, capture_layers)))
        self.patch_layer = int(patch_layer)
        self.pos_map = {
            int(p): np.asarray(v, dtype=np.float32)
            for p, v in pos_map.items()
        }
        self.prompt_len = int(prompt_len)
        self.states = {}
        self.handles = []

    def __enter__(self):
        for L in self.capture_layers:
            if L == self.patch_layer:
                self.handles.append(
                    self.decoder_layers[L].register_forward_hook(
                        self._make_patch_capture_hook(L)
                    )
                )
            else:
                self.handles.append(
                    self.decoder_layers[L].register_forward_hook(
                        self._make_capture_hook(L)
                    )
                )
        return self

    def _make_capture_hook(self, L):
        def hook(_module, _inputs, output):
            x = upd.first_tensor(output)
            if int(x.shape[1]) == self.prompt_len:
                self.states[int(L)] = (
                    x.detach().float().cpu().numpy()
                )
            return None
        return hook

    def _make_patch_capture_hook(self, L):
        def hook(_module, _inputs, output):
            x = upd.first_tensor(output)
            if int(x.shape[1]) != self.prompt_len:
                return None

            y = x.clone()
            for p, vec in self.pos_map.items():
                if 0 <= int(p) < int(y.shape[1]):
                    y[0, int(p)] += torch.as_tensor(
                        vec,
                        device=y.device,
                        dtype=y.dtype,
                    )

            self.states[int(L)] = (
                y.detach().float().cpu().numpy()
            )
            return upd.replace_first_tensor(output, y)
        return hook

    def __exit__(self, *args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def capture_with_anchor_patch(
    *,
    model,
    decoder_layers,
    batch,
    capture_layers,
    anchor_layer,
    pos_map,
):
    prompt_len = int(batch["input_ids"].shape[1])
    cap = PatchAndBlockCapture(
        decoder_layers=decoder_layers,
        capture_layers=capture_layers,
        patch_layer=anchor_layer,
        pos_map=pos_map,
        prompt_len=prompt_len,
    )
    with cap:
        model(
            **batch,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    missing = [
        L for L in capture_layers
        if L not in cap.states
    ]
    if missing:
        raise RuntimeError(
            f"Patched capture missing layers: {missing}"
        )
    return dict(cap.states)


# =============================================================================
# Counterfactual construction
# =============================================================================

def relation_states_at_anchor(
    *,
    real_states,
    no_states,
    anchor,
    real_sub_pos,
    real_ref_pos,
    no_sub_pos,
    no_ref_pos,
    pool,
):
    hs_r = pool_np(
        real_states[anchor],
        real_sub_pos,
        pool,
    )
    hr_r = pool_np(
        real_states[anchor],
        real_ref_pos,
        pool,
    )
    hs_n = pool_np(
        no_states[anchor],
        no_sub_pos,
        pool,
    )
    hr_n = pool_np(
        no_states[anchor],
        no_ref_pos,
        pool,
    )

    r_real = hs_r - hr_r
    r_no = hs_n - hr_n
    s_rn = r_real - r_no
    midpoint_real = 0.5 * (hs_r + hr_r)

    return {
        "hsub_real": hs_r,
        "href_real": hr_r,
        "hsub_no": hs_n,
        "href_no": hr_n,
        "r_real": r_real,
        "r_no": r_no,
        "s_rn": s_rn,
        "midpoint_real": midpoint_real,
    }


def phrase_shift_map(
    *,
    sub_pos,
    ref_pos,
    s_rn,
    multiplier,
):
    """
    If every subject token receives delta and every reference token receives
    -delta, the pooled subject-reference difference changes by 2*delta.

    multiplier=0.5:
        delta_sub = -0.5*s_RN
        delta_ref = +0.5*s_RN
        -> relation difference r_real - s_RN = r_no

    multiplier=1.0:
        -> relation difference r_real - 2*s_RN = r_no - s_RN
    """
    out = {}
    ds = -float(multiplier) * np.asarray(s_rn, dtype=np.float32)
    dr = +float(multiplier) * np.asarray(s_rn, dtype=np.float32)

    for p in sub_pos:
        out[int(p)] = out.get(
            int(p),
            np.zeros_like(ds),
        ) + ds

    for p in ref_pos:
        out[int(p)] = out.get(
            int(p),
            np.zeros_like(dr),
        ) + dr

    return out


# =============================================================================
# Per-update self-spatial scores
# =============================================================================

def causal_specs_for_sid(selected_by_sid, sid):
    rows = selected_by_sid.get(int(sid), None)
    if rows is None or len(rows) == 0:
        return []
    return src.causal_position_specs(rows)


def oracle_lookup_for_sid(prior_update, sid):
    if prior_update is None or len(prior_update) == 0:
        return {}

    g = prior_update[
        prior_update["sid"].astype(int) == int(sid)
    ].copy()
    if not len(g):
        return {}

    out = {}
    for (L, p), z in g.groupby(
        ["update_layer", "real_position"],
        sort=False,
    ):
        vals = pd.to_numeric(
            z["real_update_decision_score"],
            errors="coerce",
        ).to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[(int(L), int(p))] = float(vals.mean())
    return out


def build_update_rows(
    *,
    sid,
    meta,
    anchor,
    specs,
    update_layers,
    plus_states,
    zero_states,
    minus_states,
    oracle_map,
    self_score_threshold,
    behavior_threshold,
    min_effect_norm,
):
    rows = []
    allowed = set(map(int, update_layers))

    for spec in specs:
        p = int(spec["real_position"])
        cmax = int(spec["max_target_layer"])

        for L in sorted(allowed):
            # Anchor edit is applied after block K, so only downstream updates
            # are meaningful.  L=K+1 is allowed for receiver_out.
            if L <= int(anchor):
                continue
            if L > cmax:
                continue
            if (L - 1) not in plus_states or L not in plus_states:
                continue
            if (L - 1) not in zero_states or L not in zero_states:
                continue
            if (L - 1) not in minus_states or L not in minus_states:
                continue

            shapes = [
                plus_states[L - 1].shape[1],
                plus_states[L].shape[1],
                zero_states[L - 1].shape[1],
                zero_states[L].shape[1],
                minus_states[L - 1].shape[1],
                minus_states[L].shape[1],
            ]
            if not all(0 <= p < T for T in shapes):
                continue

            hp_in = plus_states[L - 1][0, p].astype(np.float32)
            h0_in = zero_states[L - 1][0, p].astype(np.float32)
            hm_in = minus_states[L - 1][0, p].astype(np.float32)

            hp_out = plus_states[L][0, p].astype(np.float32)
            h0_out = zero_states[L][0, p].astype(np.float32)
            hm_out = minus_states[L][0, p].astype(np.float32)

            a_clean = hp_out - hp_in
            a_zero = h0_out - h0_in
            a_minus = hm_out - hm_in

            d_in = 0.5 * (hp_in - hm_in)
            d_out = 0.5 * (hp_out - hm_out)

            plus0_in = hp_in - h0_in
            zero_minus_in = h0_in - hm_in
            plus0_out = hp_out - h0_out
            zero_minus_out = h0_out - hm_out

            dmid = d_in + d_out

            score_receiver_out = cosine_np(a_clean, d_out)
            score_receiver_in = cosine_np(a_clean, d_in)
            score_receiver_mid = cosine_np(a_clean, dmid)

            din2 = float(np.dot(d_in, d_in))
            score_spatial_gain = (
                float(np.dot(d_out - d_in, d_in) / din2)
                if din2 > EPS
                else float("nan")
            )

            lin_in = cosine_np(
                plus0_in,
                zero_minus_in,
            )
            lin_out = cosine_np(
                plus0_out,
                zero_minus_out,
            )

            asym_in_num = norm_np(
                0.5 * (hp_in + hm_in) - h0_in
            )
            asym_out_num = norm_np(
                0.5 * (hp_out + hm_out) - h0_out
            )
            din_norm = norm_np(d_in)
            dout_norm = norm_np(d_out)

            asym_in = (
                asym_in_num / max(din_norm, EPS)
                if din_norm > EPS
                else float("nan")
            )
            asym_out = (
                asym_out_num / max(dout_norm, EPS)
                if dout_norm > EPS
                else float("nan")
            )

            B = oracle_map.get((int(L), int(p)), float("nan"))
            oracle_sign = sign_threshold(
                B,
                behavior_threshold,
            )

            row = {
                "sid": int(sid),
                "gt": str(meta["gt"]),
                "baseline_prediction": str(
                    meta["baseline_prediction"]
                ),
                "baseline_correct": bool(
                    meta["baseline_correct"]
                ),
                "anchor_layer": int(anchor),
                "update_layer": int(L),
                "real_position": int(p),
                "token": str(spec["token"]),
                "category": str(spec["category"]),
                "broad_category": str(
                    spec["broad_category"]
                ),
                "max_target_layer": int(cmax),

                "clean_update_norm": norm_np(a_clean),
                "zero_update_norm": norm_np(a_zero),
                "minus_update_norm": norm_np(a_minus),

                "self_spatial_in_norm": din_norm,
                "self_spatial_out_norm": dout_norm,

                "score_receiver_out": score_receiver_out,
                "score_receiver_in": score_receiver_in,
                "score_receiver_mid": score_receiver_mid,
                "score_spatial_gain": score_spatial_gain,

                "pred_sign_receiver_out": (
                    sign_threshold(
                        score_receiver_out,
                        self_score_threshold,
                    )
                    if dout_norm >= min_effect_norm
                    else 0
                ),
                "pred_sign_receiver_in": (
                    sign_threshold(
                        score_receiver_in,
                        self_score_threshold,
                    )
                    if din_norm >= min_effect_norm
                    else 0
                ),
                "pred_sign_receiver_mid": sign_threshold(
                    score_receiver_mid,
                    self_score_threshold,
                ),
                "pred_sign_spatial_gain": (
                    sign_threshold(
                        score_spatial_gain,
                        self_score_threshold,
                    )
                    if din_norm >= min_effect_norm
                    else 0
                ),

                "cf_linearity_cos_in": lin_in,
                "cf_linearity_cos_out": lin_out,
                "cf_asymmetry_ratio_in": asym_in,
                "cf_asymmetry_ratio_out": asym_out,

                "oracle_B": B,
                "oracle_abs_B": (
                    abs(float(B))
                    if np.isfinite(B)
                    else float("nan")
                ),
                "oracle_sign": int(oracle_sign),
            }

            rows.append((row, a_clean))

    return rows


# =============================================================================
# Summaries
# =============================================================================

def sign_agreement_summary(
    df,
    *,
    linearity_threshold,
    max_asymmetry_ratio,
):
    rows = []

    cohort_defs = [
        ("all", df),
        (
            "baseline_wrong",
            df[df["baseline_correct"] == False],  # noqa: E712
        ),
        (
            "baseline_correct",
            df[df["baseline_correct"] == True],  # noqa: E712
        ),
    ]

    for anchor in sorted(df["anchor_layer"].unique()):
        adf = df[df["anchor_layer"] == anchor].copy()

        for cohort_name, _ in cohort_defs:
            if cohort_name == "all":
                base = adf
            elif cohort_name == "baseline_wrong":
                base = adf[
                    adf["baseline_correct"] == False  # noqa: E712
                ]
            else:
                base = adf[
                    adf["baseline_correct"] == True  # noqa: E712
                ]

            subset_defs = [
                ("all_cf", base),
                (
                    "high_cf_quality",
                    base[
                        (
                            pd.to_numeric(
                                base["cf_linearity_cos_out"],
                                errors="coerce",
                            )
                            >= float(linearity_threshold)
                        )
                        & (
                            pd.to_numeric(
                                base["cf_asymmetry_ratio_out"],
                                errors="coerce",
                            )
                            <= float(max_asymmetry_ratio)
                        )
                    ],
                ),
            ]

            for subset_name, g0 in subset_defs:
                if not len(g0):
                    continue

                for rule in RULES:
                    pred = g0[
                        f"pred_sign_{rule}"
                    ].to_numpy(int)
                    oracle = g0[
                        "oracle_sign"
                    ].to_numpy(int)
                    w = g0[
                        "oracle_abs_B"
                    ].to_numpy(float)

                    valid = (
                        (pred != 0)
                        & (oracle != 0)
                        & np.isfinite(w)
                    )

                    if not np.any(valid):
                        continue

                    corr = (
                        pred[valid] == oracle[valid]
                    ).astype(float)

                    rows.append(
                        {
                            "anchor_layer": int(anchor),
                            "cohort": cohort_name,
                            "cf_subset": subset_name,
                            "rule": rule,
                            "N_rows_total": int(len(g0)),
                            "N_valid_sign": int(
                                np.sum(valid)
                            ),
                            "coverage": float(
                                np.mean(pred != 0)
                            ),
                            "accuracy": float(
                                corr.mean()
                            ),
                            "weighted_accuracy_absB": (
                                safe_weighted_accuracy(
                                    corr,
                                    w[valid],
                                )
                            ),
                            "mean_oracle_absB": float(
                                np.mean(w[valid])
                            ),
                            "pred_positive_fraction": float(
                                np.mean(
                                    pred[valid] > 0
                                )
                            ),
                            "oracle_positive_fraction": float(
                                np.mean(
                                    oracle[valid] > 0
                                )
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def sign_agreement_by_layer(df):
    rows = []

    for (
        anchor,
        L,
        baseline_correct,
    ), g0 in df.groupby(
        [
            "anchor_layer",
            "update_layer",
            "baseline_correct",
        ],
        dropna=False,
        sort=True,
    ):
        cohort = (
            "baseline_correct"
            if bool(baseline_correct)
            else "baseline_wrong"
        )

        for rule in RULES:
            pred = g0[
                f"pred_sign_{rule}"
            ].to_numpy(int)
            oracle = g0[
                "oracle_sign"
            ].to_numpy(int)
            w = g0[
                "oracle_abs_B"
            ].to_numpy(float)

            valid = (
                (pred != 0)
                & (oracle != 0)
                & np.isfinite(w)
            )
            if not np.any(valid):
                continue

            corr = (
                pred[valid] == oracle[valid]
            ).astype(float)

            rows.append(
                {
                    "anchor_layer": int(anchor),
                    "update_layer": int(L),
                    "cohort": cohort,
                    "rule": rule,
                    "N_valid": int(np.sum(valid)),
                    "accuracy": float(corr.mean()),
                    "weighted_accuracy_absB": (
                        safe_weighted_accuracy(
                            corr,
                            w[valid],
                        )
                    ),
                    "mean_score": safe_mean(
                        g0.loc[
                            valid,
                            f"score_{rule}",
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


def cf_quality_summary(df):
    rows = []

    for (anchor, L), g in df.groupby(
        ["anchor_layer", "update_layer"],
        sort=True,
    ):
        rows.append(
            {
                "anchor_layer": int(anchor),
                "update_layer": int(L),
                "N": int(len(g)),
                "mean_self_spatial_in_norm": safe_mean(
                    g["self_spatial_in_norm"]
                ),
                "mean_self_spatial_out_norm": safe_mean(
                    g["self_spatial_out_norm"]
                ),
                "mean_cf_linearity_cos_in": safe_mean(
                    g["cf_linearity_cos_in"]
                ),
                "mean_cf_linearity_cos_out": safe_mean(
                    g["cf_linearity_cos_out"]
                ),
                "mean_cf_asymmetry_ratio_in": safe_mean(
                    g["cf_asymmetry_ratio_in"]
                ),
                "mean_cf_asymmetry_ratio_out": safe_mean(
                    g["cf_asymmetry_ratio_out"]
                ),
            }
        )

    return pd.DataFrame(rows)


def generation_summary(gdf):
    if gdf is None or not len(gdf):
        return pd.DataFrame()

    rows = []

    for anchor, g in gdf.groupby(
        "anchor_layer",
        sort=True,
    ):
        base_correct = g[
            "baseline_correct"
        ].to_numpy(bool)
        edit_correct = g[
            "edited_correct"
        ].to_numpy(bool)

        rows.append(
            {
                "anchor_layer": int(anchor),
                "N": int(len(g)),
                "baseline_acc": float(
                    base_correct.mean()
                ),
                "edited_acc": float(
                    edit_correct.mean()
                ),
                "gain": float(
                    edit_correct.mean()
                    - base_correct.mean()
                ),
                "W2C": int(
                    np.sum(
                        (~base_correct)
                        & edit_correct
                    )
                ),
                "C2W": int(
                    np.sum(
                        base_correct
                        & (~edit_correct)
                    )
                ),
                "changed_prediction": int(
                    np.sum(
                        g["baseline_prediction"]
                        != g["edited_prediction"]
                    )
                ),
                "mean_n_positive": safe_mean(
                    g["n_positive"]
                ),
                "mean_n_negative": safe_mean(
                    g["n_negative"]
                ),
                "mean_n_neutral": safe_mean(
                    g["n_neutral"]
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Generation patch from no-oracle sign
# =============================================================================

def build_generation_patch(
    scored_rows_with_vec,
    *,
    rule,
    scale,
):
    pmap = {}
    counts = {
        "n_positive": 0,
        "n_negative": 0,
        "n_neutral": 0,
    }

    for row, a_clean in scored_rows_with_vec:
        s = int(row[f"pred_sign_{rule}"])

        if s > 0:
            vec = float(scale) * a_clean
            counts["n_positive"] += 1
        elif s < 0:
            vec = -float(scale) * a_clean
            counts["n_negative"] += 1
        else:
            vec = None
            counts["n_neutral"] += 1

        if vec is not None:
            upd.add_patch(
                pmap,
                int(row["update_layer"]),
                int(row["real_position"]),
                np.asarray(vec, dtype=np.float32),
            )

    return pmap, counts


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    anchor_layers = parse_layers(a.anchor_layers)
    update_layers = parse_layers(a.update_layers)

    if not anchor_layers:
        raise ValueError("No anchor layers")
    if not update_layers:
        raise ValueError("No update layers")
    if a.spatial_scale <= 0:
        raise ValueError("--spatial-scale must be > 0")

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

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

    if prior_update is None:
        raise RuntimeError(
            f"{run_dir}/per_real_update_decision_score.csv is required "
            "for evaluation."
        )

    two, meta, rec_by_sid = src.load_dataset_for_sids(
        a,
        cohort,
    )

    model = processor = None
    per_update_rows = []
    anchor_rows = []
    generation_rows = []

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        ) = src.load_model(a, two)

        n_layers = len(decoder_layers)
        requested = sorted(
            set(
                anchor_layers
                + update_layers
                + [L - 1 for L in update_layers]
            )
        )
        bad = [
            L for L in requested
            if not (0 <= L < n_layers)
        ]
        if bad:
            raise ValueError(
                f"Requested layers outside 0..{n_layers-1}: {bad}"
            )

        device = torch.device(a.device)

        print("=" * 190)
        print("SELF-SPATIAL COUNTERFACTUAL SIGN DIAGNOSTIC")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N samples={len(meta)}")
        print(f"anchor_layers={anchor_layers}")
        print(f"update_layers={update_layers}")
        print(f"spatial_scale={a.spatial_scale}")
        print(
            "sign uses NO relation label / NO relation selector / "
            "NO final-answer gradient"
        )
        print(
            "oracle B_REAL is used only to evaluate sign agreement"
        )
        print()

        for m in tqdm(meta, desc="SELF-SPATIAL CF"):
            sid = int(m["sid"])
            image = None

            try:
                if sid not in selected_by_sid:
                    continue

                image = src.base.record_image(
                    rec_by_sid[sid]
                )
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                rb = src.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                nb = upd.build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

                # Phrase positions are located independently in the REAL and
                # NoImage prompts.  No token-index alignment is assumed.
                real_sub_pos = locate_positions(
                    processor,
                    rb,
                    m["subject"],
                )
                real_ref_pos = locate_positions(
                    processor,
                    rb,
                    m["reference"],
                )
                no_sub_pos = locate_positions(
                    processor,
                    nb,
                    m["subject"],
                )
                no_ref_pos = locate_positions(
                    processor,
                    nb,
                    m["reference"],
                )

                specs = causal_specs_for_sid(
                    selected_by_sid,
                    sid,
                )
                if not specs:
                    raise RuntimeError(
                        "No causal position specs"
                    )

                # Capture clean REAL states needed for every anchor/update.
                clean_capture_layers = sorted(
                    set(
                        anchor_layers
                        + [
                            L
                            for L in update_layers
                        ]
                        + [
                            L - 1
                            for L in update_layers
                        ]
                    )
                )

                plus_states = upd.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    clean_capture_layers,
                )

                # NoImage is required ONLY at anchors to define s_RN.
                no_states = upd.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    nb,
                    anchor_layers,
                )

                oracle_map = oracle_lookup_for_sid(
                    prior_update,
                    sid,
                )

                for anchor in anchor_layers:
                    # Only need K and downstream requested update layers.
                    downstream_updates = [
                        L for L in update_layers
                        if L > anchor
                    ]
                    if not downstream_updates:
                        continue

                    capture_layers = sorted(
                        set(
                            [anchor]
                            + downstream_updates
                            + [
                                L - 1
                                for L in downstream_updates
                            ]
                        )
                    )

                    rel = relation_states_at_anchor(
                        real_states=plus_states,
                        no_states=no_states,
                        anchor=anchor,
                        real_sub_pos=real_sub_pos,
                        real_ref_pos=real_ref_pos,
                        no_sub_pos=no_sub_pos,
                        no_ref_pos=no_ref_pos,
                        pool=a.pool,
                    )

                    s_rn = np.asarray(
                        rel["s_rn"],
                        dtype=np.float32,
                    )

                    # beta=1:
                    # zero branch removes the RN relation-difference component.
                    zero_map = phrase_shift_map(
                        sub_pos=real_sub_pos,
                        ref_pos=real_ref_pos,
                        s_rn=s_rn,
                        multiplier=0.5
                        * float(a.spatial_scale),
                    )

                    # beta=1:
                    # minus branch reverses the RN component around NoImage.
                    minus_map = phrase_shift_map(
                        sub_pos=real_sub_pos,
                        ref_pos=real_ref_pos,
                        s_rn=s_rn,
                        multiplier=1.0
                        * float(a.spatial_scale),
                    )

                    zero_states = capture_with_anchor_patch(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        capture_layers=capture_layers,
                        anchor_layer=anchor,
                        pos_map=zero_map,
                    )
                    minus_states = capture_with_anchor_patch(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        capture_layers=capture_layers,
                        anchor_layer=anchor,
                        pos_map=minus_map,
                    )

                    # Clean plus_states already exists; restrict logically via
                    # capture_layers but do not copy large arrays.
                    anchor_rows.append(
                        {
                            "sid": sid,
                            "gt": m["gt"],
                            "baseline_prediction": m[
                                "baseline_prediction"
                            ],
                            "baseline_correct": bool(
                                m["baseline_correct"]
                            ),
                            "anchor_layer": int(anchor),
                            "real_relation_norm": norm_np(
                                rel["r_real"]
                            ),
                            "noimage_relation_norm": norm_np(
                                rel["r_no"]
                            ),
                            "rn_spatial_norm": norm_np(
                                s_rn
                            ),
                            "real_noimage_relation_cosine": (
                                cosine_np(
                                    rel["r_real"],
                                    rel["r_no"],
                                )
                            ),
                            "n_real_sub_tokens": len(
                                real_sub_pos
                            ),
                            "n_real_ref_tokens": len(
                                real_ref_pos
                            ),
                        }
                    )

                    scored = build_update_rows(
                        sid=sid,
                        meta=m,
                        anchor=anchor,
                        specs=specs,
                        update_layers=downstream_updates,
                        plus_states=plus_states,
                        zero_states=zero_states,
                        minus_states=minus_states,
                        oracle_map=oracle_map,
                        self_score_threshold=float(
                            a.self_score_threshold
                        ),
                        behavior_threshold=float(
                            a.behavior_threshold
                        ),
                        min_effect_norm=float(
                            a.min_spatial_effect_norm
                        ),
                    )

                    for row, _vec in scored:
                        per_update_rows.append(row)

                    if a.run_generation and scored:
                        pmap, counts = build_generation_patch(
                            scored,
                            rule=a.gating_rule,
                            scale=float(
                                a.gating_scale
                            ),
                        )

                        pred, text = upd.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=pmap,
                            max_new_tokens=int(
                                a.max_new_tokens
                            ),
                        )
                        pred = src.normalize_rel(pred)

                        generation_rows.append(
                            {
                                "sid": sid,
                                "gt": m["gt"],
                                "anchor_layer": int(
                                    anchor
                                ),
                                "rule": a.gating_rule,
                                "scale": float(
                                    a.gating_scale
                                ),
                                "baseline_prediction": m[
                                    "baseline_prediction"
                                ],
                                "baseline_correct": bool(
                                    m["baseline_correct"]
                                ),
                                "edited_prediction": pred,
                                "edited_correct": (
                                    pred == m["gt"]
                                ),
                                **counts,
                                "text": text,
                            }
                        )

                    del zero_states, minus_states

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "traceback_tail": (
                            traceback.format_exc()
                            .splitlines()[-80:]
                        ),
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                with contextlib.suppress(Exception):
                    if image is not None:
                        image.close()
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

    if not per_update_rows:
        raise RuntimeError(
            "No self-spatial per-update rows produced."
        )

    udf = pd.DataFrame(per_update_rows)
    adf = pd.DataFrame(anchor_rows)

    udf.to_csv(
        outdir / "per_update_self_spatial.csv",
        index=False,
    )
    adf.to_csv(
        outdir / "anchor_state_summary.csv",
        index=False,
    )

    sign_sum = sign_agreement_summary(
        udf,
        linearity_threshold=float(
            a.linearity_threshold
        ),
        max_asymmetry_ratio=float(
            a.max_asymmetry_ratio
        ),
    )
    sign_sum.to_csv(
        outdir / "sign_agreement_summary.csv",
        index=False,
    )

    sign_layer = sign_agreement_by_layer(udf)
    sign_layer.to_csv(
        outdir / "sign_agreement_by_layer.csv",
        index=False,
    )

    cf_sum = cf_quality_summary(udf)
    cf_sum.to_csv(
        outdir / "counterfactual_quality_summary.csv",
        index=False,
    )

    gsum = pd.DataFrame()
    if generation_rows:
        gdf = pd.DataFrame(generation_rows)
        gdf.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )
        gsum = generation_summary(gdf)
        gsum.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )

    # Compact console report.
    report = [
        "=" * 190,
        "SELF-SPATIAL COUNTERFACTUAL SIGN DIAGNOSTIC",
        "=" * 190,
        (
            f"N samples={udf['sid'].nunique()} | "
            f"N update rows={len(udf)}"
        ),
        f"anchor_layers={anchor_layers}",
        f"update_layers={update_layers}",
        f"spatial_scale={a.spatial_scale}",
        "",
        "Definition:",
        "  s_RN = (h_sub^REAL-h_ref^REAL) - (h_sub^NoImage-h_ref^NoImage)",
        "  + trajectory = clean REAL",
        "  0 trajectory = remove RN relation-difference component at anchor",
        "  - trajectory = reverse RN component around NoImage relation state",
        "  d_out = 0.5*(h+_out-h-_out) at each downstream causal token",
        "  receiver_out = cosine(clean actual update, d_out)",
        "",
        "IMPORTANT:",
        "  No relation label, relation selector, Synthetic codebook, or final-answer",
        "  gradient enters receiver_out/in/mid/spatial_gain. Oracle B_REAL is only",
        "  an evaluation label. Prior causal-token WHERE positions remain oracle-derived.",
        "",
        "A. COUNTERFACTUAL QUALITY",
        "-" * 190,
        cf_sum.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "B. SIGN AGREEMENT WITH ORACLE B_REAL",
        "-" * 190,
        sign_sum.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "C. SIGN AGREEMENT BY DOWNSTREAM LAYER",
        "-" * 190,
        sign_layer.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
    ]

    if len(gsum):
        report += [
            "",
            "D. OPTIONAL GENERATION UNDER NO-ORACLE HOW SIGN",
            "-" * 190,
            gsum.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
        ]

    report += [
        "",
        "How to judge the experiment:",
        "  * First inspect CF quality. A spatial effect with near-zero norm or poor",
        "    +/- symmetry is not a trustworthy self-spatial axis.",
        "  * The key row is baseline_wrong / receiver_out. Chance-like ~0.50 means",
        "    the model's transported self-spatial state does NOT recover HOW sign.",
        "  * A substantial >0.5 agreement, especially |B|-weighted and on high-CF",
        "    rows, is evidence that harmful/helpful updates relate to whether the",
        "    natural update opposes/reinforces the model's own spatial influence.",
        "  * Even a strong sign agreement is diagnostic. Generation improvement is",
        "    the causal behavioral validation.",
        "",
        "Methodological caveat:",
        "  This removes the external RELATION selector, but not oracle WHERE:",
        "  causal-token positions are inherited from the prior oracle ranking.",
    ]

    report_text = "\n".join(report) + "\n"
    print(report_text)

    (outdir / "analysis_summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script": "diagnose_self_spatial_counterfactual_sign_v1.py",
        "model": a.model,
        "repo_id": getattr(spec, "repo_id", ""),
        "decoder_path": decoder_path,
        "real_update_dir": str(run_dir),
        "N_samples": int(udf["sid"].nunique()),
        "anchor_layers": anchor_layers,
        "update_layers": update_layers,
        "pool": a.pool,
        "spatial_scale": float(a.spatial_scale),
        "primary_self_sign": (
            "receiver_out = cosine(a_clean, "
            "0.5*(h_plus_out-h_minus_out))"
        ),
        "secondary_rules": {
            "receiver_in": (
                "cosine(a_clean, 0.5*(h_plus_in-h_minus_in))"
            ),
            "receiver_mid": (
                "cosine(a_clean, d_in+d_out)"
            ),
            "spatial_gain": (
                "dot(d_out-d_in,d_in)/||d_in||^2"
            ),
        },
        "spatial_component": (
            "s_RN=(hsub_real-href_real)-"
            "(hsub_noimage-href_noimage)"
        ),
        "uses_relation_label_for_sign": False,
        "uses_relation_selector_for_sign": False,
        "uses_synthetic_codebook_for_sign": False,
        "uses_final_answer_gradient_for_sign": False,
        "oracle_B_used_only_for_evaluation": True,
        "prior_where_is_oracle": True,
        "run_generation": bool(a.run_generation),
        "gating_rule": a.gating_rule,
        "gating_scale": float(a.gating_scale),
        "important_non_tautology_note": (
            "The script intentionally does not use "
            "(a_plus-a_zero) dot (a_plus-a_minus) as a sign, "
            "because local linearity can make that positive by construction."
        ),
    }

    (outdir / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
