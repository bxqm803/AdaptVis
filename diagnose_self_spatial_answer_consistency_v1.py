#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_self_spatial_answer_consistency_v1.py

Question
========
Can the model's OWN continuous spatial information determine whether a
causal-token block update is positive or negative, WITHOUT first predicting
LEFT/RIGHT/ABOVE/BELOW?

This script tests exactly that idea in the model's own 4-way answer space.

No relation selector is used to define the sign.
No GT relation is used to define the sign.
No Synthetic relation classifier/codebook is used to define the sign.
No GT-vs-competitor gradient is used to define the sign.

The existing oracle B_REAL is loaded only AFTERWARD as an evaluation label.

---------------------------------------------------------------------------
1. MODEL'S OWN SPATIAL STATE
---------------------------------------------------------------------------

At anchor layer K:

    r_REAL = h_sub^REAL - h_ref^REAL
    r_NO   = h_sub^NOIMAGE - h_ref^NOIMAGE

    s_RN = r_REAL - r_NO

Construct three trajectories by editing only the subject/reference difference
at block K while keeping the REAL pair midpoint fixed:

    (+) clean REAL
    (0) remove the RN object-relation component
    (-) reverse the RN component around the NoImage relation state

For spatial_scale beta=1:

    zero:
        h_sub <- h_sub - 0.5*s_RN
        h_ref <- h_ref + 0.5*s_RN

    minus:
        h_sub <- h_sub - 1.0*s_RN
        h_ref <- h_ref + 1.0*s_RN

Thus the sign definition never needs to know whether s_RN "means" left,
right, above, or below.

---------------------------------------------------------------------------
2. MODEL-NATIVE 4-WAY READOUT
---------------------------------------------------------------------------

For any mid-layer hidden vector h at a causal token, use the model's OWN
final normalization and LM-head rows for the four answer tokens:

    z(h) = [logit(left), logit(right), logit(above), logit(below)]

Then remove the 4-way common mode:

    zc(h) = z(h) - mean(z(h))

This is a standard zero-training logit-lens style readout.  It is NOT assumed
to be a perfect "true" internal decision variable; that is what the experiment
is testing.

Qwen-3B in the current project has single-token candidates:
    left, right, above, below.
The script checks this and stops if a candidate is multi-token.

---------------------------------------------------------------------------
3. SELF-SPATIAL ANSWER DIRECTION
---------------------------------------------------------------------------

At downstream causal token p, before block L:

    v_spatial_in(L,p)
        = 0.5 * [ zc(h+_{L-1,p}) - zc(h-_{L-1,p}) ]

This is the direction in the MODEL'S OWN ANSWER SPACE that the sample's own
spatial state has produced at this token BEFORE the current block update.

After block L:

    v_spatial_out(L,p)
        = 0.5 * [ zc(h+_{L,p}) - zc(h-_{L,p}) ]

The clean block's own answer-space movement is

    delta_z_update(L,p)
        = zc(h+_{L,p}) - zc(h+_{L-1,p})

PRIMARY no-oracle HOW score:

    C_answer_in
        = cosine(delta_z_update, v_spatial_in)

Equivalent signed projection is also saved:

    P_answer_in
        = dot(delta_z_update, v_spatial_in) / ||v_spatial_in||

Interpretation:
    C/P > 0:
        this block moves the causal token's answer readout in the SAME
        direction as the model's already-present self-spatial evidence.

    C/P < 0:
        this block moves AGAINST the model's already-present self-spatial
        evidence.

Why use the INPUT spatial direction as primary?
-----------------------------------------------
It is temporally prior to the current block update, so the sign is not defined
using the very output that the update itself created.

At L=K+1, a different causal token often has almost zero v_spatial_in because
the anchor perturbation first reaches that token INSIDE block K+1.  Those rows
are therefore left neutral by the primary rule.

Secondary diagnostics:
    answer_out:
        compare update with v_spatial_out
    answer_mid:
        compare update with v_spatial_in + v_spatial_out

---------------------------------------------------------------------------
4. ORACLE EVALUATION ONLY
---------------------------------------------------------------------------

The old run supplies:

    B_REAL = a_REAL dot grad(S_GT - S_comp)

We compare sign(C/P) against sign(B_REAL), but B_REAL NEVER enters the
self-spatial sign construction.

Report:
    accuracy
    balanced_accuracy
    positive_recall
    negative_recall
    |B|-weighted accuracy
    prediction positive fraction
    coverage

Balanced accuracy is included so a trivial all-positive/all-negative rule
cannot look good merely because of class imbalance.

---------------------------------------------------------------------------
5. OPTIONAL LAYER-NET TEST
---------------------------------------------------------------------------

For each sample x layer, sum the continuous no-oracle projection:

    P_layer = sum_p P_answer_in(L,p)

and compare sign(P_layer) with

    sign(sum_p B_REAL)

This is only a layer-net diagnostic because prior work already showed that
different causal positions in one layer can carry opposite oracle signs.

---------------------------------------------------------------------------
6. OPTIONAL GENERATION
---------------------------------------------------------------------------

If --run-generation is enabled:

    P_answer_in > 0  -> add +alpha * actual clean update
    P_answer_in < 0  -> add -alpha * actual clean update
    neutral          -> no edit

This uses:
    NO GT relation
    NO relation selector
    NO Synthetic spatial classifier
    NO GT answer gradient for HOW

BUT it still uses the prior oracle-selected causal-token WHERE scaffold.
So a successful result would establish non-oracle HOW first, not a fully
non-oracle system.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u diagnose_self_spatial_answer_consistency_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --anchor-layers 19,23 \
  --update-layers 20-26 \
  --max-samples 80 \
  --spatial-scale 1.0 \
  --output-dir output/qwen3b_self_spatial_answer_consistency_n80_v1 \
  --overwrite

If the diagnostic is promising:

  --run-generation --gating-scale 0.5

Main outputs
============
per_update_answer_consistency.csv
sign_agreement_summary.csv
sign_agreement_by_layer.csv
layer_net_consistency.csv
layer_net_summary.csv
counterfactual_answer_quality.csv
anchor_state_summary.csv
generation_per_sample.csv        (optional)
generation_summary.csv           (optional)
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
from typing import Dict, List

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
        "It is used only for the project's existing phrase-position locator.\n"
        f"{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12

RULES = (
    "answer_in",
    "answer_out",
    "answer_mid",
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
    p.add_argument("--update-layers", default="20-26")
    p.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "last"],
    )
    p.add_argument(
        "--spatial-scale",
        type=float,
        default=1.0,
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
    )
    p.add_argument(
        "--self-score-threshold",
        type=float,
        default=1e-8,
        help="Threshold on signed projection in 4-way centered-logit space.",
    )
    p.add_argument(
        "--min-spatial-answer-norm",
        type=float,
        default=1e-5,
        help="Rows below this ||v_spatial|| are neutral.",
    )

    p.add_argument(
        "--high-spatial-answer-norm-quantile",
        type=float,
        default=0.5,
        help=(
            "Also summarize rows above this within-anchor quantile of "
            "||v_spatial_in||. Set <0 to disable."
        ),
    )

    p.add_argument(
        "--run-generation",
        action="store_true",
    )
    p.add_argument(
        "--gating-rule",
        default="answer_in",
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
# Generic utilities
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


def safe_mean(xs):
    x = np.asarray(list(xs), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def norm_np(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float32)))


def cosine_np(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = norm_np(a) * norm_np(b)
    if den <= EPS:
        return float("nan")
    return float(np.dot(a, b) / den)


def sign_threshold(v, threshold):
    v = float(v)
    if not np.isfinite(v):
        return 0
    if v > threshold:
        return 1
    if v < -threshold:
        return -1
    return 0


def weighted_accuracy(correct, weights):
    c = np.asarray(correct, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(c) & np.isfinite(w) & (w >= 0)
    if not np.any(ok):
        return float("nan")
    den = float(w[ok].sum())
    if den <= EPS:
        return float("nan")
    return float(np.sum(c[ok] * w[ok]) / den)


def binary_metrics(pred, true, weights=None):
    pred = np.asarray(pred, dtype=int)
    true = np.asarray(true, dtype=int)

    valid = (pred != 0) & (true != 0)
    if not np.any(valid):
        return None

    p = pred[valid]
    y = true[valid]
    corr = (p == y).astype(float)

    pos = y > 0
    neg = y < 0

    pos_recall = (
        float(np.mean(p[pos] > 0))
        if np.any(pos)
        else float("nan")
    )
    neg_recall = (
        float(np.mean(p[neg] < 0))
        if np.any(neg)
        else float("nan")
    )

    recalls = [
        x for x in (pos_recall, neg_recall)
        if np.isfinite(x)
    ]
    bal = (
        float(np.mean(recalls))
        if recalls
        else float("nan")
    )

    wacc = float("nan")
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[valid]
        wacc = weighted_accuracy(corr, w)

    return {
        "N_valid": int(np.sum(valid)),
        "accuracy": float(np.mean(corr)),
        "balanced_accuracy": bal,
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "weighted_accuracy_absB": wacc,
        "pred_positive_fraction": float(np.mean(p > 0)),
        "oracle_positive_fraction": float(np.mean(y > 0)),
    }


def pool_np(state, positions, pool):
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


def get_attr_path(obj, path):
    cur = obj
    for name in str(path).split("."):
        if not name:
            continue
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


# =============================================================================
# Model-native 4-way logit-lens readout
# =============================================================================

class FourWayReadout:
    """
    Applies the model's own final norm and only the four LM-head rows needed
    for LEFT/RIGHT/ABOVE/BELOW.

    This avoids computing the full vocabulary logits for every stored hidden.
    """

    def __init__(
        self,
        *,
        model,
        decoder_path,
        candidate_ids,
        device,
    ):
        self.device = torch.device(device)

        # Current Qwen setup uses single-token answers.  Refuse to silently
        # reduce a multi-token answer to its first token.
        bad = {
            r: ids
            for r, ids in candidate_ids.items()
            if len(ids) != 1
        }
        if bad:
            raise RuntimeError(
                "Model-native mid-layer 4-way readout requires single-token "
                f"candidate answers. Multi-token candidates: {bad}"
            )

        self.token_ids = [
            int(candidate_ids[r][0])
            for r in REL
        ]

        # Derive final norm from decoder parent first.
        norm = None
        norm_path = None

        dp = str(decoder_path)
        parent_candidates = []
        if dp.endswith(".layers"):
            parent_candidates.append(
                dp[: -len(".layers")]
            )
        if ".layers" in dp:
            parent_candidates.append(
                dp.rsplit(".layers", 1)[0]
            )

        for pp in parent_candidates:
            parent = get_attr_path(model, pp)
            if parent is not None:
                for n in (
                    "norm",
                    "final_layernorm",
                    "final_layer_norm",
                    "ln_f",
                ):
                    x = getattr(parent, n, None)
                    if isinstance(x, torch.nn.Module):
                        norm = x
                        norm_path = f"{pp}.{n}"
                        break
            if norm is not None:
                break

        # Architecture fallbacks.
        if norm is None:
            for path in (
                "model.language_model.norm",
                "model.norm",
                "language_model.model.norm",
                "language_model.norm",
                "transformer.ln_f",
                "model.decoder.final_layer_norm",
            ):
                x = get_attr_path(model, path)
                if isinstance(x, torch.nn.Module):
                    norm = x
                    norm_path = path
                    break

        if norm is None:
            raise RuntimeError(
                "Could not resolve the language model's final normalization "
                f"from decoder_path={decoder_path!r}"
            )

        head = model.get_output_embeddings()
        if head is None:
            head = getattr(model, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            raise RuntimeError(
                "Could not resolve output embedding / lm_head weight"
            )

        self.norm = norm
        self.norm_path = str(norm_path)
        self.head = head

        weight = head.weight
        idx = torch.as_tensor(
            self.token_ids,
            dtype=torch.long,
            device=weight.device,
        )
        self.W4 = weight.index_select(
            0,
            idx,
        ).detach()

        bias = getattr(head, "bias", None)
        self.b4 = (
            bias.index_select(0, idx).detach()
            if bias is not None
            else None
        )

        self.weight_device = self.W4.device
        self.weight_dtype = self.W4.dtype

    @torch.inference_mode()
    def logits4(self, h):
        """
        h: numpy [D] or torch [D].
        returns np.float32 [4], in REL order.
        """
        if torch.is_tensor(h):
            x = h.detach().to(
                device=self.weight_device,
                dtype=self.weight_dtype,
            )
        else:
            x = torch.as_tensor(
                np.asarray(h),
                device=self.weight_device,
                dtype=self.weight_dtype,
            )

        x = x.reshape(1, 1, -1)
        y = self.norm(x)
        y = y.reshape(1, -1)

        z = y @ self.W4.T
        if self.b4 is not None:
            z = z + self.b4.reshape(1, -1)

        return (
            z[0]
            .float()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def centered4(self, h):
        z = self.logits4(h)
        return (z - float(np.mean(z))).astype(
            np.float32
        )


# =============================================================================
# Patched forward capture
# =============================================================================

class PatchAndBlockCapture:
    def __init__(
        self,
        decoder_layers,
        capture_layers,
        patch_layer,
        pos_map,
        prompt_len,
    ):
        self.decoder_layers = decoder_layers
        self.capture_layers = sorted(
            set(map(int, capture_layers))
        )
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
                    self.decoder_layers[
                        L
                    ].register_forward_hook(
                        self._make_patch_hook(L)
                    )
                )
            else:
                self.handles.append(
                    self.decoder_layers[
                        L
                    ].register_forward_hook(
                        self._make_capture_hook(L)
                    )
                )
        return self

    def _make_capture_hook(self, L):
        def hook(_module, _inputs, output):
            x = upd.first_tensor(output)
            if int(x.shape[1]) == self.prompt_len:
                self.states[int(L)] = (
                    x.detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            return None
        return hook

    def _make_patch_hook(self, L):
        def hook(_module, _inputs, output):
            x = upd.first_tensor(output)
            if int(x.shape[1]) != self.prompt_len:
                return None

            y = x.clone()
            for p, vec in self.pos_map.items():
                if 0 <= p < int(y.shape[1]):
                    y[0, p] += torch.as_tensor(
                        vec,
                        device=y.device,
                        dtype=y.dtype,
                    )

            self.states[int(L)] = (
                y.detach()
                .float()
                .cpu()
                .numpy()
            )
            return upd.replace_first_tensor(
                output,
                y,
            )
        return hook

    def __exit__(self, *args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def capture_with_patch(
    *,
    model,
    decoder_layers,
    batch,
    capture_layers,
    anchor_layer,
    pos_map,
):
    cap = PatchAndBlockCapture(
        decoder_layers=decoder_layers,
        capture_layers=capture_layers,
        patch_layer=anchor_layer,
        pos_map=pos_map,
        prompt_len=int(
            batch["input_ids"].shape[1]
        ),
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
# Self-spatial anchor construction
# =============================================================================

def anchor_relation_states(
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

    return {
        "hsub_real": hs_r,
        "href_real": hr_r,
        "hsub_no": hs_n,
        "href_no": hr_n,
        "r_real": r_real,
        "r_no": r_no,
        "s_rn": s_rn,
    }


def phrase_shift_map(
    *,
    sub_pos,
    ref_pos,
    s_rn,
    multiplier,
):
    out = {}
    s = np.asarray(s_rn, dtype=np.float32)

    ds = -float(multiplier) * s
    dr = +float(multiplier) * s

    for p in sub_pos:
        p = int(p)
        out[p] = out.get(
            p,
            np.zeros_like(ds),
        ) + ds

    for p in ref_pos:
        p = int(p)
        out[p] = out.get(
            p,
            np.zeros_like(dr),
        ) + dr

    return out


# =============================================================================
# Prior causal-token / oracle evaluation lookup
# =============================================================================

def oracle_lookup_for_sid(
    prior_update,
    sid,
):
    if prior_update is None or not len(prior_update):
        return {}

    g = prior_update[
        prior_update["sid"].astype(int)
        == int(sid)
    ]

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
            out[(int(L), int(p))] = float(
                vals.mean()
            )

    return out


# =============================================================================
# Per-update answer-space consistency
# =============================================================================

def centered_linearity(
    z_plus,
    z_zero,
    z_minus,
):
    p0 = np.asarray(
        z_plus - z_zero,
        dtype=np.float32,
    )
    zm = np.asarray(
        z_zero - z_minus,
        dtype=np.float32,
    )
    return cosine_np(p0, zm)


def centered_asymmetry(
    z_plus,
    z_zero,
    z_minus,
):
    d = 0.5 * (
        np.asarray(z_plus)
        - np.asarray(z_minus)
    )
    num = norm_np(
        0.5 * (
            np.asarray(z_plus)
            + np.asarray(z_minus)
        )
        - np.asarray(z_zero)
    )
    den = norm_np(d)
    return (
        num / max(den, EPS)
        if den > EPS
        else float("nan")
    )


def make_rule_values(
    *,
    delta_z_update,
    v_in,
    v_out,
    min_spatial_norm,
):
    dzu = np.asarray(
        delta_z_update,
        dtype=np.float32,
    )
    vin = np.asarray(v_in, dtype=np.float32)
    vout = np.asarray(v_out, dtype=np.float32)
    vmid = vin + vout

    result = {}

    for name, v in (
        ("answer_in", vin),
        ("answer_out", vout),
        ("answer_mid", vmid),
    ):
        vn = norm_np(v)
        if vn < float(min_spatial_norm):
            result[
                f"score_{name}"
            ] = float("nan")
            result[
                f"projection_{name}"
            ] = float("nan")
            result[
                f"spatial_norm_{name}"
            ] = vn
            continue

        result[f"score_{name}"] = cosine_np(
            dzu,
            v,
        )
        result[
            f"projection_{name}"
        ] = float(
            np.dot(dzu, v)
            / max(vn, EPS)
        )
        result[
            f"spatial_norm_{name}"
        ] = vn

    return result


def build_rows_for_anchor(
    *,
    sid,
    meta,
    anchor,
    specs,
    update_layers,
    plus_states,
    zero_states,
    minus_states,
    readout,
    oracle_map,
    self_score_threshold,
    behavior_threshold,
    min_spatial_norm,
):
    rows_with_vec = []

    for spec in specs:
        p = int(spec["real_position"])
        cmax = int(spec["max_target_layer"])

        for L in update_layers:
            L = int(L)
            if L <= anchor:
                continue
            if L > cmax:
                continue

            needed = (L - 1, L)
            if not all(
                x in plus_states
                and x in zero_states
                and x in minus_states
                for x in needed
            ):
                continue

            seq_lens = [
                plus_states[L - 1].shape[1],
                plus_states[L].shape[1],
                zero_states[L - 1].shape[1],
                zero_states[L].shape[1],
                minus_states[L - 1].shape[1],
                minus_states[L].shape[1],
            ]
            if not all(
                0 <= p < T
                for T in seq_lens
            ):
                continue

            hp_in = plus_states[
                L - 1
            ][0, p].astype(np.float32)
            hp_out = plus_states[
                L
            ][0, p].astype(np.float32)

            h0_in = zero_states[
                L - 1
            ][0, p].astype(np.float32)
            h0_out = zero_states[
                L
            ][0, p].astype(np.float32)

            hm_in = minus_states[
                L - 1
            ][0, p].astype(np.float32)
            hm_out = minus_states[
                L
            ][0, p].astype(np.float32)

            # Model-native centered 4-way readouts.
            zp_in = readout.centered4(hp_in)
            zp_out = readout.centered4(hp_out)

            z0_in = readout.centered4(h0_in)
            z0_out = readout.centered4(h0_out)

            zm_in = readout.centered4(hm_in)
            zm_out = readout.centered4(hm_out)

            delta_z_update = (
                zp_out - zp_in
            ).astype(np.float32)

            v_in = (
                0.5 * (zp_in - zm_in)
            ).astype(np.float32)
            v_out = (
                0.5 * (zp_out - zm_out)
            ).astype(np.float32)

            rule = make_rule_values(
                delta_z_update=delta_z_update,
                v_in=v_in,
                v_out=v_out,
                min_spatial_norm=float(
                    min_spatial_norm
                ),
            )

            # Natural residual update vector, used only if optional generation
            # gating is later requested.
            a_clean = (
                hp_out - hp_in
            ).astype(np.float32)

            B = oracle_map.get(
                (L, p),
                float("nan"),
            )
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

                "clean_update_hidden_norm": norm_np(
                    a_clean
                ),
                "delta_z_update_norm": norm_np(
                    delta_z_update
                ),
                "v_spatial_in_norm": norm_np(v_in),
                "v_spatial_out_norm": norm_np(v_out),

                "cf_answer_linearity_in": (
                    centered_linearity(
                        zp_in,
                        z0_in,
                        zm_in,
                    )
                ),
                "cf_answer_linearity_out": (
                    centered_linearity(
                        zp_out,
                        z0_out,
                        zm_out,
                    )
                ),
                "cf_answer_asymmetry_in": (
                    centered_asymmetry(
                        zp_in,
                        z0_in,
                        zm_in,
                    )
                ),
                "cf_answer_asymmetry_out": (
                    centered_asymmetry(
                        zp_out,
                        z0_out,
                        zm_out,
                    )
                ),

                "oracle_B": B,
                "oracle_abs_B": (
                    abs(float(B))
                    if np.isfinite(B)
                    else float("nan")
                ),
                "oracle_sign": int(
                    oracle_sign
                ),
            }

            # Save continuous four-dimensional directions.  These labels are
            # diagnostic only; no argmax relation is used to define sign.
            for r in REL:
                j = RID[r]
                row[
                    f"delta_z_update_{r}"
                ] = float(
                    delta_z_update[j]
                )
                row[
                    f"v_spatial_in_{r}"
                ] = float(v_in[j])
                row[
                    f"v_spatial_out_{r}"
                ] = float(v_out[j])

            row.update(rule)

            for name in RULES:
                proj = row[
                    f"projection_{name}"
                ]
                row[
                    f"pred_sign_{name}"
                ] = sign_threshold(
                    proj,
                    self_score_threshold,
                )

            # Purely diagnostic: what relation would the spatial effect favor?
            # NOT used by the sign rule or generation gating.
            if norm_np(v_in) >= min_spatial_norm:
                row[
                    "diag_spatial_in_argmax"
                ] = REL[
                    int(np.argmax(v_in))
                ]
                row[
                    "diag_spatial_in_gt_match"
                ] = (
                    row[
                        "diag_spatial_in_argmax"
                    ]
                    == meta["gt"]
                )
            else:
                row[
                    "diag_spatial_in_argmax"
                ] = ""
                row[
                    "diag_spatial_in_gt_match"
                ] = np.nan

            if norm_np(v_out) >= min_spatial_norm:
                row[
                    "diag_spatial_out_argmax"
                ] = REL[
                    int(np.argmax(v_out))
                ]
                row[
                    "diag_spatial_out_gt_match"
                ] = (
                    row[
                        "diag_spatial_out_argmax"
                    ]
                    == meta["gt"]
                )
            else:
                row[
                    "diag_spatial_out_argmax"
                ] = ""
                row[
                    "diag_spatial_out_gt_match"
                ] = np.nan

            rows_with_vec.append(
                (row, a_clean)
            )

    return rows_with_vec


# =============================================================================
# Summaries
# =============================================================================

def make_strength_thresholds(
    df,
    quantile,
):
    out = {}
    if quantile < 0:
        return out

    q = min(max(float(quantile), 0.0), 1.0)

    for anchor, g in df.groupby(
        "anchor_layer",
        sort=True,
    ):
        x = pd.to_numeric(
            g["v_spatial_in_norm"],
            errors="coerce",
        ).to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x):
            out[int(anchor)] = float(
                np.quantile(x, q)
            )

    return out


def sign_summary(
    df,
    *,
    strength_thresholds,
):
    rows = []

    for anchor in sorted(
        df["anchor_layer"].unique()
    ):
        adf = df[
            df["anchor_layer"] == anchor
        ].copy()

        cohort_defs = [
            ("all", adf),
            (
                "baseline_wrong",
                adf[
                    adf["baseline_correct"]
                    == False  # noqa: E712
                ],
            ),
            (
                "baseline_correct",
                adf[
                    adf["baseline_correct"]
                    == True  # noqa: E712
                ],
            ),
        ]

        for cohort, base in cohort_defs:
            subset_defs = [
                ("all_rows", base)
            ]

            if int(anchor) in strength_thresholds:
                th = strength_thresholds[
                    int(anchor)
                ]
                subset_defs.append(
                    (
                        "strong_spatial_in",
                        base[
                            pd.to_numeric(
                                base[
                                    "v_spatial_in_norm"
                                ],
                                errors="coerce",
                            )
                            >= th
                        ],
                    )
                )

            for subset, g in subset_defs:
                if not len(g):
                    continue

                for rule in RULES:
                    pred = g[
                        f"pred_sign_{rule}"
                    ].to_numpy(int)
                    true = g[
                        "oracle_sign"
                    ].to_numpy(int)
                    w = g[
                        "oracle_abs_B"
                    ].to_numpy(float)

                    met = binary_metrics(
                        pred,
                        true,
                        weights=w,
                    )
                    if met is None:
                        continue

                    rows.append(
                        {
                            "anchor_layer": int(
                                anchor
                            ),
                            "cohort": cohort,
                            "subset": subset,
                            "rule": rule,
                            "N_rows": int(len(g)),
                            "coverage": float(
                                np.mean(pred != 0)
                            ),
                            "mean_spatial_norm": safe_mean(
                                g[
                                    f"spatial_norm_{rule}"
                                ]
                            ),
                            "mean_projection": safe_mean(
                                g[
                                    f"projection_{rule}"
                                ]
                            ),
                            **met,
                        }
                    )

    return pd.DataFrame(rows)


def sign_by_layer(df):
    rows = []

    for (
        anchor,
        L,
        baseline_correct,
    ), g in df.groupby(
        [
            "anchor_layer",
            "update_layer",
            "baseline_correct",
        ],
        sort=True,
    ):
        cohort = (
            "baseline_correct"
            if bool(baseline_correct)
            else "baseline_wrong"
        )

        for rule in RULES:
            pred = g[
                f"pred_sign_{rule}"
            ].to_numpy(int)
            true = g[
                "oracle_sign"
            ].to_numpy(int)
            w = g[
                "oracle_abs_B"
            ].to_numpy(float)

            met = binary_metrics(
                pred,
                true,
                weights=w,
            )
            if met is None:
                continue

            rows.append(
                {
                    "anchor_layer": int(anchor),
                    "update_layer": int(L),
                    "cohort": cohort,
                    "rule": rule,
                    "N_rows": int(len(g)),
                    "coverage": float(
                        np.mean(pred != 0)
                    ),
                    "mean_projection": safe_mean(
                        g[
                            f"projection_{rule}"
                        ]
                    ),
                    "mean_delta_z_norm": safe_mean(
                        g[
                            "delta_z_update_norm"
                        ]
                    ),
                    **met,
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
                "mean_v_spatial_in_norm": safe_mean(
                    g["v_spatial_in_norm"]
                ),
                "mean_v_spatial_out_norm": safe_mean(
                    g["v_spatial_out_norm"]
                ),
                "mean_delta_z_update_norm": safe_mean(
                    g["delta_z_update_norm"]
                ),
                "mean_cf_answer_linearity_in": safe_mean(
                    g[
                        "cf_answer_linearity_in"
                    ]
                ),
                "mean_cf_answer_linearity_out": safe_mean(
                    g[
                        "cf_answer_linearity_out"
                    ]
                ),
                "mean_cf_answer_asymmetry_in": safe_mean(
                    g[
                        "cf_answer_asymmetry_in"
                    ]
                ),
                "mean_cf_answer_asymmetry_out": safe_mean(
                    g[
                        "cf_answer_asymmetry_out"
                    ]
                ),
                "diag_spatial_in_gt_match": safe_mean(
                    pd.to_numeric(
                        g[
                            "diag_spatial_in_gt_match"
                        ],
                        errors="coerce",
                    )
                ),
                "diag_spatial_out_gt_match": safe_mean(
                    pd.to_numeric(
                        g[
                            "diag_spatial_out_gt_match"
                        ],
                        errors="coerce",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def build_layer_net(df):
    """
    Sum continuous answer_in projections across causal positions at one layer.
    Compare to sum of old oracle B only for evaluation.
    """
    rows = []

    for (
        sid,
        anchor,
        L,
    ), g in df.groupby(
        [
            "sid",
            "anchor_layer",
            "update_layer",
        ],
        sort=False,
    ):
        proj = pd.to_numeric(
            g["projection_answer_in"],
            errors="coerce",
        ).to_numpy(float)

        B = pd.to_numeric(
            g["oracle_B"],
            errors="coerce",
        ).to_numpy(float)

        p_ok = np.isfinite(proj)
        b_ok = np.isfinite(B)

        pred_sum = (
            float(np.sum(proj[p_ok]))
            if np.any(p_ok)
            else float("nan")
        )
        oracle_sum = (
            float(np.sum(B[b_ok]))
            if np.any(b_ok)
            else float("nan")
        )

        first = g.iloc[0]

        rows.append(
            {
                "sid": int(sid),
                "anchor_layer": int(anchor),
                "update_layer": int(L),
                "gt": first["gt"],
                "baseline_prediction": first[
                    "baseline_prediction"
                ],
                "baseline_correct": bool(
                    first["baseline_correct"]
                ),
                "N_updates": int(len(g)),
                "N_nonzero_self": int(
                    np.sum(p_ok)
                ),
                "sum_self_projection": pred_sum,
                "pred_layer_sign": (
                    sign_threshold(
                        pred_sum,
                        1e-12,
                    )
                ),
                "sum_oracle_B": oracle_sum,
                "oracle_layer_sign": (
                    sign_threshold(
                        oracle_sum,
                        1e-12,
                    )
                ),
                "oracle_abs_layer_B": (
                    abs(oracle_sum)
                    if np.isfinite(oracle_sum)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_layer_net(layer_df):
    rows = []

    for (
        anchor,
        cohort,
    ) in [
        (a, c)
        for a in sorted(
            layer_df[
                "anchor_layer"
            ].unique()
        )
        for c in (
            "all",
            "baseline_wrong",
            "baseline_correct",
        )
    ]:
        g = layer_df[
            layer_df["anchor_layer"] == anchor
        ].copy()

        if cohort == "baseline_wrong":
            g = g[
                g["baseline_correct"]
                == False  # noqa: E712
            ]
        elif cohort == "baseline_correct":
            g = g[
                g["baseline_correct"]
                == True  # noqa: E712
            ]

        if not len(g):
            continue

        met = binary_metrics(
            g["pred_layer_sign"].to_numpy(int),
            g["oracle_layer_sign"].to_numpy(int),
            weights=g[
                "oracle_abs_layer_B"
            ].to_numpy(float),
        )
        if met is None:
            continue

        rows.append(
            {
                "anchor_layer": int(anchor),
                "cohort": cohort,
                "N_sample_layers": int(len(g)),
                "coverage": float(
                    np.mean(
                        g["pred_layer_sign"]
                        .to_numpy(int)
                        != 0
                    )
                ),
                **met,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Optional generation
# =============================================================================

def build_generation_patch(
    rows_with_vec,
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

    for row, a_clean in rows_with_vec:
        s = int(
            row[f"pred_sign_{rule}"]
        )

        if s > 0:
            vec = (
                float(scale)
                * np.asarray(
                    a_clean,
                    dtype=np.float32,
                )
            )
            counts["n_positive"] += 1
        elif s < 0:
            vec = (
                -float(scale)
                * np.asarray(
                    a_clean,
                    dtype=np.float32,
                )
            )
            counts["n_negative"] += 1
        else:
            vec = None
            counts["n_neutral"] += 1

        if vec is not None:
            upd.add_patch(
                pmap,
                int(row["update_layer"]),
                int(row["real_position"]),
                vec,
            )

    return pmap, counts


def summarize_generation(gdf):
    rows = []

    for anchor, g in gdf.groupby(
        "anchor_layer",
        sort=True,
    ):
        b = g[
            "baseline_correct"
        ].to_numpy(bool)
        e = g[
            "edited_correct"
        ].to_numpy(bool)

        rows.append(
            {
                "anchor_layer": int(anchor),
                "N": int(len(g)),
                "baseline_acc": float(
                    np.mean(b)
                ),
                "edited_acc": float(
                    np.mean(e)
                ),
                "gain": float(
                    np.mean(e) - np.mean(b)
                ),
                "W2C": int(
                    np.sum((~b) & e)
                ),
                "C2W": int(
                    np.sum(b & (~e))
                ),
                "changed_prediction": int(
                    np.sum(
                        g[
                            "baseline_prediction"
                        ]
                        != g[
                            "edited_prediction"
                        ]
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
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    anchors = parse_layers(a.anchor_layers)
    update_layers = parse_layers(a.update_layers)

    if not anchors:
        raise ValueError("No anchor layers")
    if not update_layers:
        raise ValueError("No update layers")
    if float(a.spatial_scale) <= 0:
        raise ValueError(
            "--spatial-scale must be >0"
        )

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
            "Prior per_real_update_decision_score.csv "
            "is required only for evaluation."
        )

    two, meta, rec_by_sid = (
        src.load_dataset_for_sids(
            a,
            cohort,
        )
    )

    model = processor = None
    update_rows = []
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
                anchors
                + update_layers
                + [
                    L - 1
                    for L in update_layers
                ]
            )
        )
        bad = [
            L for L in requested
            if not 0 <= L < n_layers
        ]
        if bad:
            raise ValueError(
                f"Requested layers outside 0..{n_layers-1}: {bad}"
            )

        candidate_texts = src.get_candidate_texts(
            prior_metadata
        )
        candidate_ids = src.encode_candidate_ids(
            processor,
            candidate_texts,
        )

        readout = FourWayReadout(
            model=model,
            decoder_path=decoder_path,
            candidate_ids=candidate_ids,
            device=a.device,
        )

        device = torch.device(a.device)

        print("=" * 190)
        print("SELF-SPATIAL ANSWER-SPACE CONSISTENCY")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N samples={len(meta)}")
        print(f"anchors={anchors}")
        print(f"update_layers={update_layers}")
        print(f"spatial_scale={a.spatial_scale}")
        print(f"final_norm={readout.norm_path}")
        print(
            "candidate token ids="
            + str(
                {
                    r: candidate_ids[r]
                    for r in REL
                }
            )
        )
        print(
            "PRIMARY: projection of clean block-induced 4-way logit-lens "
            "change onto pre-block self-spatial 4-way direction"
        )
        print(
            "No GT / relation selector / Synthetic classifier / "
            "GT answer gradient enters the sign."
        )
        print()

        for m in tqdm(
            meta,
            desc="SELF-SPATIAL ANSWER",
        ):
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
                    question_text=m[
                        "question_text"
                    ],
                    device=device,
                )

                nb = upd.build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

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

                specs = src.causal_position_specs(
                    selected_by_sid[sid]
                )
                if not specs:
                    raise RuntimeError(
                        "No causal position specs"
                    )

                clean_capture_layers = sorted(
                    set(
                        anchors
                        + update_layers
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

                no_states = upd.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    nb,
                    anchors,
                )

                oracle_map = oracle_lookup_for_sid(
                    prior_update,
                    sid,
                )

                for anchor in anchors:
                    downstream = [
                        L for L in update_layers
                        if L > anchor
                    ]
                    if not downstream:
                        continue

                    capture_layers = sorted(
                        set(
                            [anchor]
                            + downstream
                            + [
                                L - 1
                                for L in downstream
                            ]
                        )
                    )

                    ar = anchor_relation_states(
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
                        ar["s_rn"],
                        dtype=np.float32,
                    )

                    zero_map = phrase_shift_map(
                        sub_pos=real_sub_pos,
                        ref_pos=real_ref_pos,
                        s_rn=s_rn,
                        multiplier=(
                            0.5
                            * float(
                                a.spatial_scale
                            )
                        ),
                    )

                    minus_map = phrase_shift_map(
                        sub_pos=real_sub_pos,
                        ref_pos=real_ref_pos,
                        s_rn=s_rn,
                        multiplier=(
                            1.0
                            * float(
                                a.spatial_scale
                            )
                        ),
                    )

                    zero_states = capture_with_patch(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        capture_layers=capture_layers,
                        anchor_layer=anchor,
                        pos_map=zero_map,
                    )

                    minus_states = capture_with_patch(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        capture_layers=capture_layers,
                        anchor_layer=anchor,
                        pos_map=minus_map,
                    )

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
                            "anchor_layer": int(
                                anchor
                            ),
                            "real_relation_norm": norm_np(
                                ar["r_real"]
                            ),
                            "noimage_relation_norm": norm_np(
                                ar["r_no"]
                            ),
                            "rn_spatial_norm": norm_np(
                                s_rn
                            ),
                            "real_noimage_relation_cosine": (
                                cosine_np(
                                    ar["r_real"],
                                    ar["r_no"],
                                )
                            ),
                        }
                    )

                    scored = build_rows_for_anchor(
                        sid=sid,
                        meta=m,
                        anchor=anchor,
                        specs=specs,
                        update_layers=downstream,
                        plus_states=plus_states,
                        zero_states=zero_states,
                        minus_states=minus_states,
                        readout=readout,
                        oracle_map=oracle_map,
                        self_score_threshold=float(
                            a.self_score_threshold
                        ),
                        behavior_threshold=float(
                            a.behavior_threshold
                        ),
                        min_spatial_norm=float(
                            a.min_spatial_answer_norm
                        ),
                    )

                    for row, _vec in scored:
                        update_rows.append(row)

                    if (
                        a.run_generation
                        and scored
                    ):
                        pmap, counts = (
                            build_generation_patch(
                                scored,
                                rule=a.gating_rule,
                                scale=float(
                                    a.gating_scale
                                ),
                            )
                        )

                        pred, text = (
                            upd.generate_with_patch(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                patch_map=pmap,
                                max_new_tokens=int(
                                    a.max_new_tokens
                                ),
                            )
                        )
                        pred = src.normalize_rel(
                            pred
                        )

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
                                    m[
                                        "baseline_correct"
                                    ]
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
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                        "traceback_tail": (
                            traceback.format_exc()
                            .splitlines()[-100:]
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

    if not update_rows:
        raise RuntimeError(
            "No answer-consistency rows produced."
        )

    udf = pd.DataFrame(update_rows)
    adf = pd.DataFrame(anchor_rows)

    udf.to_csv(
        outdir
        / "per_update_answer_consistency.csv",
        index=False,
    )
    adf.to_csv(
        outdir / "anchor_state_summary.csv",
        index=False,
    )

    strength_thresholds = (
        make_strength_thresholds(
            udf,
            float(
                a.high_spatial_answer_norm_quantile
            ),
        )
    )

    ssum = sign_summary(
        udf,
        strength_thresholds=strength_thresholds,
    )
    ssum.to_csv(
        outdir / "sign_agreement_summary.csv",
        index=False,
    )

    slayer = sign_by_layer(udf)
    slayer.to_csv(
        outdir
        / "sign_agreement_by_layer.csv",
        index=False,
    )

    cf = cf_quality_summary(udf)
    cf.to_csv(
        outdir
        / "counterfactual_answer_quality.csv",
        index=False,
    )

    lnet = build_layer_net(udf)
    lnet.to_csv(
        outdir
        / "layer_net_consistency.csv",
        index=False,
    )

    lnet_sum = summarize_layer_net(lnet)
    lnet_sum.to_csv(
        outdir
        / "layer_net_summary.csv",
        index=False,
    )

    gsum = pd.DataFrame()
    if generation_rows:
        gdf = pd.DataFrame(
            generation_rows
        )
        gdf.to_csv(
            outdir
            / "generation_per_sample.csv",
            index=False,
        )
        gsum = summarize_generation(
            gdf
        )
        gsum.to_csv(
            outdir
            / "generation_summary.csv",
            index=False,
        )

    report = [
        "=" * 190,
        "SELF-SPATIAL ANSWER-SPACE CONSISTENCY",
        "=" * 190,
        (
            f"N samples={udf['sid'].nunique()} | "
            f"N update rows={len(udf)}"
        ),
        f"anchors={anchors}",
        f"update_layers={update_layers}",
        f"final_norm={readout.norm_path}",
        "",
        "PRIMARY NO-ORACLE HOW:",
        "  v_spatial_in = 0.5 * [zc(h+_in)-zc(h-_in)]",
        "  delta_z_update = zc(h+_out)-zc(h+_in)",
        "  projection_answer_in = dot(delta_z_update,v_spatial_in)/||v_spatial_in||",
        "  sign(projection_answer_in) is the proposed HOW sign.",
        "",
        "No GT relation / no relation selector / no Synthetic classifier /",
        "no GT-vs-competitor gradient enters this sign.",
        "",
        "A. COUNTERFACTUAL SPATIAL EFFECT IN MODEL-NATIVE 4-WAY ANSWER SPACE",
        "-" * 190,
        cf.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "Diagnostic only: diag_spatial_*_gt_match asks whether the model's",
        "self-spatial perturbation happens to favor the GT candidate; it is",
        "NEVER used to choose the sign.",
        "",
        "B. PER-UPDATE SIGN AGREEMENT WITH ORACLE B_REAL",
        "-" * 190,
        ssum.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "C. PER-UPDATE SIGN AGREEMENT BY LAYER",
        "-" * 190,
        slayer.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        "D. SAMPLE x LAYER NET SIGN",
        "-" * 190,
        lnet_sum.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
    ]

    if len(gsum):
        report += [
            "",
            "E. OPTIONAL GENERATION",
            "-" * 190,
            gsum.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
        ]

    report += [
        "",
        "How to judge:",
        "  * For the PRIMARY answer_in rule, balanced_accuracy matters more than",
        "    raw accuracy. An all-positive/all-negative bias cannot fake it.",
        "  * L=K+1 may have low answer_in coverage because the spatial perturbation",
        "    reaches another causal token only inside that first downstream block.",
        "  * Strong result: baseline_wrong answer_in balanced accuracy clearly >0.5,",
        "    stable across layers/anchors, preferably with stronger |B|-weighted",
        "    accuracy, and without collapsing one oracle class recall.",
        "  * If answer_out works but answer_in does not, be cautious: answer_out uses",
        "    a spatial direction measured after the update itself.",
        "  * Final validation is generation under answer_in gating.",
        "",
        "Remaining oracle dependency:",
        "  WHERE (causal-token positions) still comes from the old oracle ranking.",
        "  This experiment tests whether HOW can become model-internal and selector-free.",
    ]

    report_text = "\n".join(report) + "\n"
    print(report_text)

    (
        outdir
        / "analysis_summary.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script": (
            "diagnose_self_spatial_answer_consistency_v1.py"
        ),
        "model": a.model,
        "repo_id": getattr(
            spec,
            "repo_id",
            "",
        ),
        "decoder_path": decoder_path,
        "final_norm_path": readout.norm_path,
        "candidate_texts": candidate_texts,
        "candidate_ids": candidate_ids,
        "real_update_dir": str(run_dir),
        "N_samples": int(
            udf["sid"].nunique()
        ),
        "anchor_layers": anchors,
        "update_layers": update_layers,
        "pool": a.pool,
        "spatial_scale": float(
            a.spatial_scale
        ),
        "answer_readout": (
            "zc(h)=center4(lm_head(final_norm(h))) "
            "restricted to the four single-token relation candidates"
        ),
        "primary_spatial_direction": (
            "v_spatial_in=0.5*(zc(h_plus_in)-zc(h_minus_in))"
        ),
        "primary_update_direction": (
            "delta_z_update=zc(h_plus_out)-zc(h_plus_in)"
        ),
        "primary_score": (
            "projection_answer_in="
            "dot(delta_z_update,v_spatial_in)/||v_spatial_in||"
        ),
        "uses_relation_label_for_sign": False,
        "uses_relation_selector_for_sign": False,
        "uses_synthetic_codebook_for_sign": False,
        "uses_gt_answer_gradient_for_sign": False,
        "oracle_B_used_only_for_evaluation": True,
        "prior_where_is_oracle": True,
        "run_generation": bool(
            a.run_generation
        ),
        "gating_rule": a.gating_rule,
        "gating_scale": float(
            a.gating_scale
        ),
        "important_caveat": (
            "Mid-layer final-norm+LM-head readout is a zero-training logit lens. "
            "It is model-native but not assumed to be a perfectly calibrated "
            "internal decision variable; causal generation is the final validation."
        ),
    }

    (
        outdir / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
