#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_layerwise_direction_local_transition_gating_v1_1.py

Purpose
=======
Test the objection that the current best non-oracle repair is still:

    ONE sample-level Direction selector
        ->
    final-answer / sequence-score gradient
        ->
    +/- steering

This script replaces the single sample-level Direction route with a genuinely
LAYER-SPECIFIC spatial belief, and then tests two different HOW definitions.

For every update layer L (default 20..26), use Direction Heads at layer L to
decode a local spatial belief:

    r_hat_L in {left, right, above, below}

The Direction heads / codebook are selected ONLY from Synthetic-400 source OOF
results.  No COCO GT is used to select heads or define r_hat_L.

We then test:

B) LAYER-SPECIFIC ROUTE -> FINAL DECISION GRADIENT
-------------------------------------------------
For each actual block update:

    a_L,p = h_L,p - h_{L-1,p}

use the relation predicted by Direction Heads AT THAT LAYER:

    B_final(L,p)
      = < a_L,p,
          grad_h [ S_{r_hat_L} - S_{foil(r_hat_L)} ] >

This still uses final sequence scores, but no longer uses one global relation
for every layer.  It tests whether spatial belief evolves with depth.

C) LOCAL SPATIAL TRANSITION JACOBIAN -- NO FINAL LOGIT IN HOW
--------------------------------------------------------------
Use the Direction belief at layer L as the *incoming* spatial belief r_hat_L.

Then define the Direction-Head readout at the NEXT layer L+1:

    D_{L+1}(r)

where D is the exact source-only Direction selector probability built from
pre-W_O subject-reference Image-NoImage residuals.

For the current relation r_hat_L and the strongest local competitor at L+1:

    M_local(L)
       = D_{L+1}(r_hat_L)
         - D_{L+1}(foil_local)

Then:

    B_local(L,p)
       = < a_L,p,
           grad_{h_L,p} M_local(L) >

Interpretation:

    B_local > 0:
        the actual block-L update helps the NEXT layer preserve/read the
        spatial belief that layer L currently holds.

    B_local < 0:
        the actual block-L update pushes the next Direction representation
        away from layer L's spatial belief.

Crucially, B_local NEVER uses answer logits or answer sequence scores.

The actual generation intervention is still performed at the fixed global
canonical causal positions:

    B > 0  -> h_L,p += alpha * a_L,p
    B < 0  -> h_L,p -= alpha * a_L,p

So the experiment separates:

    WHERE : fixed global causal scaffold
    WHAT  : layer-local spatial belief
    HOW-B : final decision Jacobian
    HOW-C : local spatial-transition Jacobian

Why L -> L+1?
=============
The actual update a_L is the block OUTPUT of L.  It cannot causally affect an
attention head that already ran inside the same block L.  Therefore the clean
local causal question is:

    Direction belief at L
        -> actual block-L update
        -> Direction readout at L+1

This also matches the project's earlier source/receiver alignment:

    causal block-output L  ---> Direction Head at L+1.

Direction selector definition
=============================
For a Direction head h at layer H:

    r_real_h = mean(z_REAL(subject)) - mean(z_REAL(reference))
    r_no_h   = mean(z_NOIMAGE(subject)) - mean(z_NOIMAGE(reference))

    x_h = (r_real_h - r_no_h) - source_center[H,h]

    c_h(r) = cosine(x_h, source_direction[H,h,r])
    p_h(r) = softmax(c_h(r) / T)

Take the strongest source-OOF heads WITHIN that layer and ensemble their p_h.
Default is Top3 heads/layer with equal weights.

Thus each layer independently outputs:

    p_L(left/right/above/below), r_hat_L

without COCO labels.

Conditions
==========
baseline

global_final_reference
    Optional old method, if --selector-csv is supplied:
    one global top10_equal route for every L + final sequence gradient.

layer_final_all
    r_hat_L separately for every L + final sequence gradient.

layer_final_layer_disagree
    Same, but edit only updates from layers where r_hat_L != baseline answer.

layer_final_consensus_disagree
    Same layer-specific signs, but edit the sample only when the mean layer
    posterior's argmax != baseline answer.

local_transition_all
    local L -> L+1 spatial Jacobian only; NO answer logits in HOW.

local_transition_layer_disagree
    Local Jacobian, but only layers whose r_hat_L != baseline answer.

local_transition_consensus_disagree
    Local Jacobian only when the mean layer posterior disagrees with baseline.

oracle_signed_reference
    Uses GT final sequence gradient only as a diagnostic ceiling.

Sign diagnostics
================
Because all four candidate sequence gradients are already computed for method B,
we also obtain the oracle sign:

    B_oracle = <a, grad(S_GT - S_best_nonGT)>

and compare:
    sign(B_layer_final)
    sign(B_local_transition)
    sign(B_global_reference)
against oracle sign.

Recommended first run: N=80
===========================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_layerwise_direction_local_transition_gating_v1_1.py \
  --model qwen-3b \
  --global-template-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --direction-selector-dir output/qwen3b_direction_selector_syn400_to_coco440 \
  --selector-csv output/qwen3b_direction_selector_syn400_to_coco440/synthetic400_to_coco440_predictions.csv \
  --selector-method top10_equal \
  --global-k 10 \
  --update-layers 20-26 \
  --heads-per-layer 3 \
  --head-weighting equal \
  --scale 0.5 \
  --eval-max-samples 80 \
  --include-oracle-reference \
  --output-dir output/qwen3b_layerwise_direction_local_transition_n80_v1 \
  --overwrite

If local_transition sign accuracy / generation is promising, then run 440 by
setting:
    --eval-max-samples 0

Main outputs
============
layer_route_summary.csv
per_sample_layer_routes.csv
layer_transition_summary.csv

per_update_scores.csv
sign_summary.csv
sign_by_layer.csv

generation_summary.csv
generation_by_relation.csv
generation_per_sample.csv

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import analyze_coco_head_object_residual_direction_probe_v1 as dh
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26
import eval_global_fixed_l26_top10_trajectory_gating_v1 as gfix


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# CLI / basic
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

    p.add_argument("--global-template-dir", required=True)
    p.add_argument("--prior-real-update-dir", required=True)

    p.add_argument(
        "--direction-selector-dir",
        required=True,
        help=(
            "Existing Synthetic-400 selector output containing "
            "synthetic_fitted_direction_codebook.npz and "
            "source_oof_head_reliability.csv."
        ),
    )

    p.add_argument(
        "--selector-csv",
        default="",
        help=(
            "Optional old global-selector predictions CSV. If supplied, "
            "global_final_reference is generated."
        ),
    )
    p.add_argument("--selector-method", default="top10_equal")

    p.add_argument("--global-k", type=int, default=10)
    p.add_argument("--update-layers", default="20-26")

    p.add_argument(
        "--heads-per-layer",
        type=int,
        default=3,
        help="Top Synthetic source-OOF Direction heads selected independently in each layer.",
    )
    p.add_argument(
        "--head-weighting",
        default="equal",
        choices=["equal", "reliability"],
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "last"],
        help="Must normally match Direction selector extraction.",
    )

    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--include-oracle-reference", action="store_true")

    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all prior-run samples.",
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


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


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


# =============================================================================
# Global fixed WHERE
# =============================================================================

def load_global_top(template_dir: Path, K: int):
    ranking_path = template_dir / "global_slot_ranking.csv"
    top_path = template_dir / "global_topk_slots.csv"

    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
        if "global_rank" not in df.columns:
            raise RuntimeError(f"{ranking_path} missing global_rank")
        df["global_rank"] = pd.to_numeric(df["global_rank"], errors="raise").astype(int)
        df = df.sort_values("global_rank")
        if len(df) < int(K):
            raise RuntimeError(f"Only {len(df)} slots available; requested K={K}")
        top = df.head(int(K)).copy()

        # Sanity: for K=10, ensure the prefix is the old exact Top10 if available.
        if int(K) == 10 and top_path.exists():
            old = pd.read_csv(top_path).sort_values("global_rank")
            if len(old) >= 10:
                a = top["canonical_slot"].astype(str).tolist()
                b = old.head(10)["canonical_slot"].astype(str).tolist()
                if a != b:
                    raise RuntimeError(
                        "global_slot_ranking[:10] != old global_topk_slots.csv"
                    )
        return top

    if not top_path.exists():
        raise FileNotFoundError(
            f"Need {ranking_path} or {top_path}"
        )

    top = pd.read_csv(top_path).sort_values("global_rank")
    if len(top) < int(K):
        raise RuntimeError(
            f"{top_path} has only {len(top)} rows; requested K={K}"
        )
    return top.head(int(K)).copy()


# =============================================================================
# Source-only Direction codebook / per-layer head selection
# =============================================================================

def load_direction_source(selector_dir: Path):
    code_path = selector_dir / "synthetic_fitted_direction_codebook.npz"
    rel_path = selector_dir / "source_oof_head_reliability.csv"

    if not code_path.exists():
        raise FileNotFoundError(code_path)
    if not rel_path.exists():
        raise FileNotFoundError(rel_path)

    z = np.load(code_path, allow_pickle=True)
    center = np.asarray(z["center"], dtype=np.float32)
    dirs = np.asarray(z["directions"], dtype=np.float32)

    if "relations" in z.files:
        names = [canon_rel(x) for x in z["relations"].tolist()]
    else:
        names = list(REL)

    if set(names) != set(REL):
        raise RuntimeError(f"Unexpected codebook relations: {names}")

    # Reorder direction axis to REL.
    idx = [names.index(r) for r in REL]
    dirs = dirs[:, :, idx, :]

    reliability = pd.read_csv(rel_path)
    for c in ("layer", "head"):
        reliability[c] = pd.to_numeric(reliability[c], errors="raise").astype(int)
    reliability["source_oof_accuracy"] = pd.to_numeric(
        reliability["source_oof_accuracy"], errors="raise"
    ).astype(float)

    return center, dirs, reliability, code_path, rel_path


def choose_heads_by_layer(
    reliability: pd.DataFrame,
    layers: Sequence[int],
    K: int,
):
    out = {}
    rows = []

    for L in sorted(set(map(int, layers))):
        g = reliability[reliability["layer"] == L].copy()
        g = g.sort_values(
            ["source_oof_accuracy", "head"],
            ascending=[False, True],
        )
        g = g.head(int(K))

        if len(g) == 0:
            raise RuntimeError(f"No Direction heads available at L{L}")

        selected = []
        for r in g.itertuples():
            h = int(r.head)
            acc = float(r.source_oof_accuracy)
            selected.append((h, acc))
            rows.append(
                {
                    "layer": L,
                    "head": h,
                    "head_name": f"L{L}H{h:02d}",
                    "source_oof_accuracy": acc,
                    "rank_within_layer": len(selected),
                }
            )
        out[L] = selected

    return out, pd.DataFrame(rows)


# =============================================================================
# Optional old global selector route
# =============================================================================

def load_global_routes(path: str, method: str):
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    pred_col = f"pred_{method}"

    if "sid" not in df.columns or pred_col not in df.columns:
        raise RuntimeError(
            f"{p} needs sid and {pred_col}; prediction columns="
            f"{[c for c in df.columns if c.startswith('pred_')]}"
        )

    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
    df["route"] = df[pred_col].map(canon_rel)

    return dict(zip(df["sid"], df["route"]))


# =============================================================================
# Graph capture
# =============================================================================

class JointGraphCapture:
    """
    Capture:
      * block OUTPUT tensors for requested block layers;
      * pre-W_O attention outputs for requested Direction-head layers.

    A grad-enabled cut is inserted at the earliest update-layer block output,
    exactly as in gate.GraphBlockCapture.  This is necessary because the loaded
    model can be fully frozen (parameters require_grad=False).  Numerically the
    trajectory is unchanged; downstream activations become differentiable with
    respect to that cut state.
    """

    def __init__(
        self,
        decoder_layers,
        block_layers: Sequence[int],
        prewo_layers: Sequence[int],
        cut_layer: int,
    ):
        self.decoder_layers = decoder_layers
        self.block_layers = sorted(set(map(int, block_layers)))
        self.prewo_layers = sorted(set(map(int, prewo_layers)))
        self.cut_layer = int(cut_layer)
        self.block: Dict[int, torch.Tensor] = {}
        self.prewo: Dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        # IMPORTANT:
        # The loaded VLM is used as a frozen model in this project, so merely
        # capturing intermediate activations does NOT guarantee that they carry
        # an autograd graph.  Mirror gate.GraphBlockCapture: cut the graph at
        # the earliest update layer and re-introduce a leaf that requires grad.
        #
        # Numerically y == x, so the REAL trajectory is unchanged.  The only
        # purpose is to make all downstream L->L+1 Direction readouts
        # differentiable w.r.t. the causal block outputs.
        if self.cut_layer not in self.block_layers:
            raise ValueError(
                f"cut_layer={self.cut_layer} must be included in block_layers"
            )

        def cut_hook(_module, _inp, out):
            x = gate.first_tensor(out)
            y = x.detach().clone().requires_grad_(True)
            return gate.replace_first_tensor(out, y)

        # Register this FIRST.  The later capture hook at the same layer will
        # therefore see the replaced grad-enabled output.
        self.handles.append(
            self.decoder_layers[self.cut_layer].register_forward_hook(cut_hook)
        )

        for L in self.block_layers:
            def make_block(li):
                def hook(_module, _inp, out):
                    self.block[li] = gate.first_tensor(out)
                    return None
                return hook
            self.handles.append(
                self.decoder_layers[L].register_forward_hook(make_block(L))
            )

        for H in self.prewo_layers:
            op = dh.resolve_o_proj(
                dh.resolve_self_attention(self.decoder_layers[H])
            )

            def make_prewo(li):
                def hook(_module, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{li} o_proj input unavailable")
                    self.prewo[li] = inputs[0]
                    return None
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_prewo(H))
            )

        return self

    def validate(self):
        mb = [L for L in self.block_layers if L not in self.block]
        mp = [L for L in self.prewo_layers if L not in self.prewo]
        if mb or mp:
            raise RuntimeError(
                f"Capture missing block={mb}, prewo={mp}"
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __exit__(self, *args):
        self.close()


class PreWOCapture:
    def __init__(self, decoder_layers, layers):
        self.decoder_layers = decoder_layers
        self.layers = sorted(set(map(int, layers)))
        self.prewo = {}
        self.handles = []

    def __enter__(self):
        for L in self.layers:
            op = dh.resolve_o_proj(
                dh.resolve_self_attention(self.decoder_layers[L])
            )

            def make_hook(li):
                def hook(_module, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{li} o_proj input unavailable")
                    self.prewo[li] = inputs[0]
                    return None
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_hook(L))
            )
        return self

    def validate(self):
        miss = [L for L in self.layers if L not in self.prewo]
        if miss:
            raise RuntimeError(f"Missing pre-WO captures: {miss}")

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Phrase relation vectors / layer Direction readout
# =============================================================================

def phrase_relation_from_prewo(
    x: torch.Tensor,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    n_heads: int,
    head_dim: int,
    pool: str,
):
    """
    x: [1,S,H*Dh]
    return [H,Dh] subject-reference relation.
    """
    zs = dh.pool_positions(x, subject_positions, pool)
    zr = dh.pool_positions(x, reference_positions, pool)

    if int(zs.numel()) != int(n_heads * head_dim):
        raise RuntimeError(
            f"pre-WO pooled width {zs.numel()} != H*Dh={n_heads*head_dim}"
        )

    return (
        zs.view(n_heads, head_dim).float()
        - zr.view(n_heads, head_dim).float()
    )


def layer_direction_probs_graph(
    *,
    real_relation: torch.Tensor,   # [H,Dh], graph
    no_relation: torch.Tensor,     # [H,Dh], detached
    layer: int,
    selected_heads,
    center_np,
    dirs_np,
    temperature: float,
    weighting: str,
):
    """
    Exact differentiable analogue of the source-only selector for ONE layer.
    returns [4] probability tensor.
    """
    device = real_relation.device
    probs = []
    weights = []

    for h, acc in selected_heads:
        h = int(h)

        x = real_relation[h] - no_relation[h]
        center = torch.as_tensor(
            center_np[layer, h],
            device=device,
            dtype=torch.float32,
        )
        dirs = torch.as_tensor(
            dirs_np[layer, h],
            device=device,
            dtype=torch.float32,
        )  # [4,Dh]

        xc = x.float() - center
        xn = xc / xc.norm().clamp_min(EPS)

        scores = torch.einsum("rd,d->r", dirs, xn)
        p = torch.softmax(
            scores / max(float(temperature), 1e-8),
            dim=-1,
        )

        if weighting == "reliability":
            w = max(float(acc) - 0.25, 0.0)
            if w <= 0:
                continue
        else:
            w = 1.0

        probs.append(p)
        weights.append(float(w))

    if not probs:
        raise RuntimeError(
            f"No usable selected heads at Direction layer {layer}"
        )

    stack = torch.stack(probs, dim=0)
    w = torch.as_tensor(
        weights,
        device=stack.device,
        dtype=stack.dtype,
    )
    return (stack * w[:, None]).sum(dim=0) / w.sum().clamp_min(EPS)


def strongest_other_dict(scores: Mapping[str, float], route: str):
    return max(
        (r for r in REL if r != route),
        key=lambda r: float(scores[r]),
    )


# =============================================================================
# Patch-scored entries
# =============================================================================

def copies_with_score(entries, score_by_key, allowed_layers=None):
    out = []

    for e in entries:
        L = int(e["update_layer"])
        p = int(e["real_position"])

        if allowed_layers is not None and L not in allowed_layers:
            continue

        key = (L, p)
        if key not in score_by_key:
            continue

        ee = dict(e)
        ee["real_update_decision_score"] = float(score_by_key[key])
        out.append(ee)

    return out


def generate_condition(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    baseline,
    gt,
    entries,
    scale,
    max_new_tokens,
):
    if not entries:
        return baseline, "", {
            "patched": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
        }

    patch_map, counts = gate.build_real_update_patch_map(
        entries,
        "real_signed",
        float(scale),
        0.0,
    )

    if not patch_map:
        return baseline, "", counts

    pred, text = gate.generate_with_patch(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        batch=batch,
        patch_map=patch_map,
        max_new_tokens=max_new_tokens,
    )
    return pred, text, counts


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(df):
    if df is None or df.empty or "condition" not in df.columns:
        return pd.DataFrame()

    base_df = (
        df[df["condition"] == "baseline"]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows = []

    for cond, g in df[
        df["condition"] != "baseline"
    ].groupby("condition"):
        gg = g.drop_duplicates("sid").set_index("sid")
        sids = sorted(set(base_df.index) & set(gg.index))
        if not sids:
            continue

        b = base_df.loc[sids]
        p = gg.loc[sids]

        bc = b["correct"].astype(bool).to_numpy()
        pc = p["correct"].astype(bool).to_numpy()

        w2c = int(np.sum((~bc) & pc))
        c2w = int(np.sum(bc & (~pc)))

        rows.append(
            {
                "condition": cond,
                "N": len(sids),
                "baseline_accuracy": float(np.mean(bc)),
                "patched_accuracy": float(np.mean(pc)),
                "gain": float(np.mean(pc) - np.mean(bc)),
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net": w2c - c2w,
                "changed": int(
                    np.sum(
                        b["prediction"].astype(str).to_numpy()
                        != p["prediction"].astype(str).to_numpy()
                    )
                ),
                "repair_rate_on_wrong": (
                    w2c / max(int(np.sum(~bc)), 1)
                ),
                "preserve_rate_on_correct": (
                    1.0 - c2w / max(int(np.sum(bc)), 1)
                ),
                "trigger_rate": float(
                    p["triggered"].astype(bool).mean()
                ),
                "mean_patched_updates": float(
                    p["n_patched_updates"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["patched_accuracy", "correct_to_wrong"],
        ascending=[False, True],
    )


def summarize_by_relation(df):
    rows = []
    if df is None or df.empty or "gt" not in df.columns:
        return pd.DataFrame()
    for gt, g in df.groupby("gt"):
        s = summarize_generation(g)
        if len(s):
            s.insert(1, "relation", DISPLAY.get(gt, gt))
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_signs(df):
    rows = []

    required = {
        "baseline_correct",
        "B_layer_final",
        "B_local_transition",
        "B_oracle",
        "sid",
    }
    if df is None or df.empty or not required.issubset(set(df.columns)):
        return pd.DataFrame(
            columns=[
                "method",
                "cohort",
                "N_updates",
                "N_samples",
                "sign_accuracy",
                "weighted_sign_accuracy",
                "coverage",
            ]
        )

    methods = [
        ("layer_final", "B_layer_final"),
        ("local_transition", "B_local_transition"),
    ]

    if "B_global_reference" in df.columns:
        methods.append(("global_final_reference", "B_global_reference"))

    for method, col in methods:
        for cohort, g in [
            ("all", df),
            ("baseline_wrong", df[~df["baseline_correct"]]),
            ("baseline_correct", df[df["baseline_correct"]]),
        ]:
            v = g[
                np.isfinite(g[col])
                & np.isfinite(g["B_oracle"])
                & (np.sign(g[col]) != 0)
                & (np.sign(g["B_oracle"]) != 0)
            ].copy()

            if not len(v):
                rows.append(
                    {
                        "method": method,
                        "cohort": cohort,
                        "N_updates": 0,
                        "N_samples": 0,
                        "sign_accuracy": np.nan,
                        "weighted_sign_accuracy": np.nan,
                        "coverage": 0.0,
                    }
                )
                continue

            pred = np.sign(v[col].to_numpy(float)).astype(int)
            oracle = np.sign(v["B_oracle"].to_numpy(float)).astype(int)
            correct = pred == oracle

            denom = max(
                int(
                    np.sum(
                        np.isfinite(g["B_oracle"])
                        & (np.sign(g["B_oracle"]) != 0)
                    )
                ),
                1,
            )

            rows.append(
                {
                    "method": method,
                    "cohort": cohort,
                    "N_updates": int(len(v)),
                    "N_samples": int(v["sid"].nunique()),
                    "sign_accuracy": float(np.mean(correct)),
                    "weighted_sign_accuracy": weighted_accuracy(
                        correct,
                        np.abs(v["B_oracle"].to_numpy(float)),
                    ),
                    "coverage": len(v) / denom,
                }
            )

    return pd.DataFrame(rows)


def summarize_signs_by_layer(df):
    rows = []

    if df is None or df.empty or "update_layer" not in df.columns:
        return pd.DataFrame()

    for L, g in df.groupby("update_layer"):
        s = summarize_signs(g)
        if len(s):
            s.insert(0, "update_layer", int(L))
            rows.append(s)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    update_layers = parse_ints(a.update_layers)
    if not update_layers:
        raise ValueError("--update-layers is empty")

    route_layers = sorted(set(update_layers))
    next_layers = sorted(set(L + 1 for L in update_layers))
    all_direction_layers = sorted(set(route_layers + next_layers))
    block_capture_layers = sorted(
        set(update_layers + [L - 1 for L in update_layers])
    )

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    global_top = load_global_top(
        Path(a.global_template_dir),
        int(a.global_k),
    )
    global_top.to_csv(
        outdir / "global_slots_used.csv",
        index=False,
    )

    center_np, dirs_np, reliability, code_path, rel_path = (
        load_direction_source(Path(a.direction_selector_dir))
    )

    selected_by_layer, selected_head_df = choose_heads_by_layer(
        reliability,
        all_direction_layers,
        int(a.heads_per_layer),
    )
    selected_head_df.to_csv(
        outdir / "selected_direction_heads_by_layer.csv",
        index=False,
    )

    global_routes = load_global_routes(
        a.selector_csv,
        a.selector_method,
    )

    prior_cohort, _, prior_metadata, _ = l26.load_prior_run(
        Path(a.prior_real_update_dir)
    )

    if int(a.eval_max_samples) > 0:
        prior_cohort = l26.stratified_cap_df(
            prior_cohort,
            int(a.eval_max_samples),
            int(a.seed) + 733,
        )

    two, eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        prior_cohort,
    )

    texts = prior_metadata.get(
        "candidate_texts",
        {
            "left": "left",
            "right": "right",
            "above": "above",
            "below": "below",
        },
    )
    texts = {canon_rel(k): str(v) for k, v in texts.items()}

    reduction = str(
        prior_metadata.get("sequence_score_reduction", "mean")
    )
    if reduction not in ("mean", "sum"):
        reduction = "mean"

    model = processor = None
    layer_route_rows = []
    update_rows = []
    generation_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = (
            l26.load_model(a, two)
        )
        device = torch.device(a.device)

        n_heads, head_dim = dh.scan_shape(
            model,
            decoder_layers,
        )

        if center_np.shape[:2] != (len(decoder_layers), n_heads):
            raise RuntimeError(
                f"Codebook shape {center_np.shape} incompatible with "
                f"model layers={len(decoder_layers)}, heads={n_heads}"
            )
        if center_np.shape[-1] != head_dim:
            raise RuntimeError(
                f"Codebook head_dim={center_np.shape[-1]} != model head_dim={head_dim}"
            )

        bad_layers = [
            L for L in (
                all_direction_layers
                + block_capture_layers
            )
            if not 0 <= L < len(decoder_layers)
        ]
        if bad_layers:
            raise RuntimeError(
                f"Requested layers out of range: {sorted(set(bad_layers))}"
            )

        candidate_ids = gate.encode_candidate_ids(
            processor,
            texts,
        )

        print("=" * 210)
        print("LAYER-SPECIFIC DIRECTION -> FINAL vs LOCAL SPATIAL JACOBIAN")
        print("=" * 210)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(eval_meta)}")
        print(f"global WHERE K={a.global_k}")
        print(f"update layers={update_layers}")
        print(f"route Direction layers={route_layers}")
        print(f"next Direction layers={next_layers}")
        print(
            f"heads/layer={a.heads_per_layer} "
            f"weighting={a.head_weighting} "
            f"temperature={a.temperature}"
        )
        print(f"scale={a.scale}")
        print(f"old global selector reference={'yes' if global_routes else 'no'}")
        print()
        print(
            selected_head_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print()

        for m in tqdm(
            eval_meta,
            desc="layer-specific spatial HOW",
        ):
            sid = int(m["sid"])
            gt = canon_rel(m["gt"])
            baseline = canon_rel(m["baseline_prediction"])
            image = None

            generation_rows.append(
                {
                    "sid": sid,
                    "gt": gt,
                    "condition": "baseline",
                    "prediction": baseline,
                    "correct": bool(m["baseline_correct"]),
                    "triggered": False,
                    "consensus_route": "",
                    "n_patched_updates": 0,
                    "text": "",
                }
            )

            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                nb = gate.build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

                real_ids = [
                    int(x)
                    for x in rb["input_ids"][0].detach().cpu().tolist()
                ]
                no_ids = [
                    int(x)
                    for x in nb["input_ids"][0].detach().cpu().tolist()
                ]

                real_sub_pos = dh.locate_phrase_positions(
                    processor.tokenizer,
                    real_ids,
                    m["subject"],
                )
                real_ref_pos = dh.locate_phrase_positions(
                    processor.tokenizer,
                    real_ids,
                    m["reference"],
                )
                no_sub_pos = dh.locate_phrase_positions(
                    processor.tokenizer,
                    no_ids,
                    m["subject"],
                )
                no_ref_pos = dh.locate_phrase_positions(
                    processor.tokenizer,
                    no_ids,
                    m["reference"],
                )

                # ---------------------------------------------------------
                # NoImage Direction relation vectors (constant control).
                # ---------------------------------------------------------
                with PreWOCapture(
                    decoder_layers,
                    all_direction_layers,
                ) as ncap:
                    with torch.inference_mode():
                        kw = dict(nb)
                        kw["use_cache"] = False
                        kw["return_dict"] = True
                        _ = model(**kw)
                    ncap.validate()

                no_relation_by_layer = {}
                for H in all_direction_layers:
                    no_relation_by_layer[H] = (
                        phrase_relation_from_prewo(
                            ncap.prewo[H],
                            no_sub_pos,
                            no_ref_pos,
                            n_heads,
                            head_dim,
                            a.pool,
                        )
                        .detach()
                        .float()
                    )

                # ---------------------------------------------------------
                # REAL graph:
                #   block outputs + Direction pre-WO activations.
                # ---------------------------------------------------------
                with JointGraphCapture(
                    decoder_layers,
                    block_capture_layers,
                    all_direction_layers,
                    cut_layer=min(update_layers),
                ) as cap:
                    kw = dict(rb)
                    kw["use_cache"] = False
                    kw["return_dict"] = True
                    _ = model(**kw)
                    cap.validate()

                    # Live layer-specific Direction probabilities.
                    probs_by_layer = {}
                    real_relation_by_layer = {}

                    for H in all_direction_layers:
                        rr = phrase_relation_from_prewo(
                            cap.prewo[H],
                            real_sub_pos,
                            real_ref_pos,
                            n_heads,
                            head_dim,
                            a.pool,
                        )
                        real_relation_by_layer[H] = rr

                        probs_by_layer[H] = layer_direction_probs_graph(
                            real_relation=rr,
                            no_relation=no_relation_by_layer[H],
                            layer=H,
                            selected_heads=selected_by_layer[H],
                            center_np=center_np,
                            dirs_np=dirs_np,
                            temperature=float(a.temperature),
                            weighting=a.head_weighting,
                        )

                    # Route summaries.
                    route_by_layer = {}
                    for H in all_direction_layers:
                        p = probs_by_layer[H]
                        ri = int(torch.argmax(p.detach()).item())
                        route = REL[ri]
                        route_by_layer[H] = route

                        order = torch.argsort(
                            p.detach(),
                            descending=True,
                        ).tolist()
                        margin = float(
                            (
                                p.detach()[order[0]]
                                - p.detach()[order[1]]
                            ).item()
                        )

                        layer_route_rows.append(
                            {
                                "sid": sid,
                                "gt": gt,
                                "baseline_prediction": baseline,
                                "baseline_correct": bool(m["baseline_correct"]),
                                "direction_layer": H,
                                "route": route,
                                "route_correct": route == gt,
                                "margin": margin,
                                **{
                                    f"prob_{r}": float(
                                        p.detach()[RID[r]].item()
                                    )
                                    for r in REL
                                },
                            }
                        )

                    # Mean layer posterior only for sample-level trigger summary.
                    route_stack = torch.stack(
                        [probs_by_layer[L].detach() for L in route_layers],
                        dim=0,
                    )
                    mean_route_prob = route_stack.mean(dim=0)
                    consensus_route = REL[
                        int(torch.argmax(mean_route_prob).item())
                    ]
                    consensus_disagree = (
                        consensus_route != baseline
                    )

                    # -----------------------------------------------------
                    # Actual REAL block updates a_L = h_L - h_{L-1}.
                    # -----------------------------------------------------
                    real_states = {
                        L: cap.block[L]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                        for L in block_capture_layers
                    }

                    # Resolve the SAME global WHERE slots for this sample.
                    ids = real_ids
                    cats, toks = gfix.dyn.build_categories(
                        model,
                        processor,
                        rb,
                        ids,
                        m["subject"],
                        m["reference"],
                    )
                    mapping, canon_seq, sig = gfix.canonical_mapping(
                        model=model,
                        processor=processor,
                        batch=rb,
                        ids=ids,
                        subject=m["subject"],
                        reference=m["reference"],
                    )
                    specs, resolved = gfix.make_specs_from_global_slots(
                        global_top=global_top,
                        mapping=mapping,
                        ids=ids,
                        cats=cats,
                        toks=toks,
                        selector_layer=max(update_layers),
                    )

                    if not specs:
                        raise RuntimeError(
                            "No fixed global causal positions resolved"
                        )

                    entries = gate.build_real_updates(
                        sid=sid,
                        gt=gt,
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
                            "No actual REAL updates on fixed WHERE"
                        )

                    # -----------------------------------------------------
                    # C) LOCAL spatial-transition Jacobian.
                    #
                    # target relation = Direction belief at L
                    # readout          = Direction probability at L+1
                    # -----------------------------------------------------
                    local_grad_by_layer = {}
                    local_foil_by_layer = {}
                    local_margin_value = {}

                    for i, L in enumerate(update_layers):
                        incoming_route = route_by_layer[L]
                        pnext = probs_by_layer[L + 1]

                        foil = max(
                            (r for r in REL if r != incoming_route),
                            key=lambda r: float(
                                pnext.detach()[RID[r]].item()
                            ),
                        )
                        local_foil_by_layer[L] = foil

                        margin = (
                            pnext[RID[incoming_route]]
                            - pnext[RID[foil]]
                        )
                        local_margin_value[L] = float(
                            margin.detach().item()
                        )

                        if not margin.requires_grad:
                            raise RuntimeError(
                                f"Local margin L{L}->L{L+1} does not require grad. "
                                f"pnext.requires_grad={pnext.requires_grad}; "
                                f"block_L.requires_grad={cap.block[L].requires_grad}. "
                                "This means the local Direction readout was detached "
                                "from the REAL causal trajectory."
                            )

                        if not cap.block[L].requires_grad:
                            raise RuntimeError(
                                f"Captured block output L{L} does not require grad. "
                                f"cut_layer={min(update_layers)}."
                            )

                        grad = torch.autograd.grad(
                            margin,
                            cap.block[L],
                            retain_graph=(
                                i < len(update_layers) - 1
                            ),
                            create_graph=False,
                            allow_unused=True,
                        )[0]

                        if grad is None:
                            raise RuntimeError(
                                f"Local margin L{L}->L{L+1} has no gradient "
                                f"to block output L{L}"
                            )

                        local_grad_by_layer[L] = (
                            grad.detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                # End REAL local graph.  We now hold detached local gradients.

                # ---------------------------------------------------------
                # B) FINAL sequence gradients for all four relation candidates.
                #    Needed for layer-specific final method + oracle diagnostic.
                # ---------------------------------------------------------
                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                final_scores = {}
                final_grads = {}

                for r in REL:
                    sc, gr = gate.sequence_score_and_grads(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        answer_ids=candidate_ids[r],
                        reduction=reduction,
                        grad_layers=grad_layers,
                    )
                    final_scores[r] = float(sc)
                    final_grads[r] = gr

                # Layer-specific final competitor.
                final_foil_by_layer = {}
                for L in update_layers:
                    rr = route_by_layer[L]
                    final_foil_by_layer[L] = strongest_other_dict(
                        final_scores,
                        rr,
                    )

                oracle_foil = strongest_other_dict(
                    final_scores,
                    gt,
                )

                global_route = (
                    canon_rel(global_routes[sid])
                    if sid in global_routes
                    else None
                )
                global_foil = (
                    strongest_other_dict(final_scores, global_route)
                    if global_route in REL
                    else None
                )

                # ---------------------------------------------------------
                # Per-update B values.
                # ---------------------------------------------------------
                B_layer_final = {}
                B_local = {}
                B_oracle = {}
                B_global = {}

                for e in entries:
                    L = int(e["update_layer"])
                    p = int(e["real_position"])
                    a_vec = np.asarray(
                        e["_real_update"],
                        dtype=np.float32,
                    )

                    rr = route_by_layer[L]
                    rf = final_foil_by_layer[L]

                    g_layer = (
                        final_grads[rr][L][0, p]
                        - final_grads[rf][L][0, p]
                    ).astype(np.float32)
                    B_layer_final[(L, p)] = float(
                        np.dot(a_vec, g_layer)
                    )

                    g_local = local_grad_by_layer[L][0, p]
                    B_local[(L, p)] = float(
                        np.dot(a_vec, g_local)
                    )

                    g_oracle = (
                        final_grads[gt][L][0, p]
                        - final_grads[oracle_foil][L][0, p]
                    ).astype(np.float32)
                    B_oracle[(L, p)] = float(
                        np.dot(a_vec, g_oracle)
                    )

                    if global_route in REL:
                        g_global = (
                            final_grads[global_route][L][0, p]
                            - final_grads[global_foil][L][0, p]
                        ).astype(np.float32)
                        B_global[(L, p)] = float(
                            np.dot(a_vec, g_global)
                        )

                    update_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "baseline_prediction": baseline,
                            "baseline_correct": bool(m["baseline_correct"]),
                            "update_layer": L,
                            "next_direction_layer": L + 1,
                            "position": p,
                            "token": str(e.get("token", "")),
                            "category": str(e.get("category", "")),
                            "canonical_slot": str(e.get("canonical_slot", "")),
                            "global_rank": int(e.get("global_rank", -1)),
                            "route_L": route_by_layer[L],
                            "route_L_correct": route_by_layer[L] == gt,
                            "route_next": route_by_layer[L + 1],
                            "route_next_correct": route_by_layer[L + 1] == gt,
                            "route_changed_L_to_next": (
                                route_by_layer[L]
                                != route_by_layer[L + 1]
                            ),
                            "final_foil_for_route_L": final_foil_by_layer[L],
                            "local_foil_at_next": local_foil_by_layer[L],
                            "local_margin_value": local_margin_value[L],
                            "consensus_route": consensus_route,
                            "global_route": (
                                global_route if global_route in REL else ""
                            ),
                            "B_layer_final": B_layer_final[(L, p)],
                            "B_local_transition": B_local[(L, p)],
                            "B_oracle": B_oracle[(L, p)],
                            "B_global_reference": (
                                B_global.get((L, p), np.nan)
                            ),
                            "real_update_norm": float(
                                e["real_update_norm"]
                            ),
                        }
                    )

                # ---------------------------------------------------------
                # Actual generation conditions.
                # ---------------------------------------------------------
                generation_specs = []

                # Layer-specific route -> final answer gradient.
                generation_specs.append(
                    (
                        "layer_final_all",
                        copies_with_score(
                            entries,
                            B_layer_final,
                        ),
                        True,
                    )
                )

                allowed_disagree_layers = {
                    L for L in update_layers
                    if route_by_layer[L] != baseline
                }
                generation_specs.append(
                    (
                        "layer_final_layer_disagree",
                        copies_with_score(
                            entries,
                            B_layer_final,
                            allowed_layers=allowed_disagree_layers,
                        ),
                        bool(allowed_disagree_layers),
                    )
                )

                generation_specs.append(
                    (
                        "layer_final_consensus_disagree",
                        (
                            copies_with_score(
                                entries,
                                B_layer_final,
                            )
                            if consensus_disagree
                            else []
                        ),
                        bool(consensus_disagree),
                    )
                )

                # Pure local spatial transition Jacobian.
                generation_specs.append(
                    (
                        "local_transition_all",
                        copies_with_score(
                            entries,
                            B_local,
                        ),
                        True,
                    )
                )

                generation_specs.append(
                    (
                        "local_transition_layer_disagree",
                        copies_with_score(
                            entries,
                            B_local,
                            allowed_layers=allowed_disagree_layers,
                        ),
                        bool(allowed_disagree_layers),
                    )
                )

                generation_specs.append(
                    (
                        "local_transition_consensus_disagree",
                        (
                            copies_with_score(
                                entries,
                                B_local,
                            )
                            if consensus_disagree
                            else []
                        ),
                        bool(consensus_disagree),
                    )
                )

                if global_route in REL:
                    generation_specs.append(
                        (
                            "global_final_reference",
                            copies_with_score(
                                entries,
                                B_global,
                            ),
                            True,
                        )
                    )

                if a.include_oracle_reference:
                    generation_specs.append(
                        (
                            "oracle_signed_reference",
                            copies_with_score(
                                entries,
                                B_oracle,
                            ),
                            True,
                        )
                    )

                for cond, patch_entries, triggered in generation_specs:
                    if not triggered:
                        pred = baseline
                        text = ""
                        counts = {
                            "patched": 0,
                            "positive": 0,
                            "negative": 0,
                            "neutral": len(entries),
                        }
                    else:
                        pred, text, counts = generate_condition(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            baseline=baseline,
                            gt=gt,
                            entries=patch_entries,
                            scale=float(a.scale),
                            max_new_tokens=a.max_new_tokens,
                        )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "condition": cond,
                            "prediction": pred,
                            "correct": pred == gt,
                            "triggered": bool(triggered),
                            "consensus_route": consensus_route,
                            "consensus_route_correct": (
                                consensus_route == gt
                            ),
                            "n_patched_updates": int(
                                counts["patched"]
                            ),
                            "n_positive_updates": int(
                                counts["positive"]
                            ),
                            "n_negative_updates": int(
                                counts["negative"]
                            ),
                            "text": text,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-100:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =====================================================================
        # Save / summaries
        # =====================================================================
        route_df = pd.DataFrame(layer_route_rows)
        upd_df = pd.DataFrame(update_rows)
        gen_df = pd.DataFrame(generation_rows)

        route_df.to_csv(
            outdir / "per_sample_layer_routes.csv",
            index=False,
        )
        upd_df.to_csv(
            outdir / "per_update_scores.csv",
            index=False,
        )
        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )

        # Layer route accuracy.
        route_summary_rows = []
        for H, g in route_df.groupby("direction_layer"):
            for cohort, x in [
                ("all", g),
                ("baseline_wrong", g[~g["baseline_correct"]]),
                ("baseline_correct", g[g["baseline_correct"]]),
            ]:
                route_summary_rows.append(
                    {
                        "direction_layer": int(H),
                        "cohort": cohort,
                        "N": int(x["sid"].nunique()),
                        "route_accuracy": (
                            float(x["route_correct"].mean())
                            if len(x) else np.nan
                        ),
                        "mean_margin": safe_mean(
                            x["margin"].to_numpy(float)
                        ),
                    }
                )
        route_summary = pd.DataFrame(route_summary_rows)
        route_summary.to_csv(
            outdir / "layer_route_summary.csv",
            index=False,
        )

        # Transition stability.
        transition_rows = []
        if len(upd_df):
            sample_trans = (
                upd_df[
                    [
                        "sid",
                        "gt",
                        "baseline_correct",
                        "update_layer",
                        "route_L",
                        "route_next",
                        "route_changed_L_to_next",
                    ]
                ]
                .drop_duplicates(["sid", "update_layer"])
            )

            for L, g in sample_trans.groupby("update_layer"):
                transition_rows.append(
                    {
                        "update_layer": int(L),
                        "N": int(g["sid"].nunique()),
                        "route_change_rate": float(
                            g["route_changed_L_to_next"].mean()
                        ),
                        "incoming_route_accuracy": float(
                            (g["route_L"] == g["gt"]).mean()
                        ),
                        "next_route_accuracy": float(
                            (g["route_next"] == g["gt"]).mean()
                        ),
                        "wrong_to_correct_route_transition": float(
                            (
                                (g["route_L"] != g["gt"])
                                & (g["route_next"] == g["gt"])
                            ).mean()
                        ),
                        "correct_to_wrong_route_transition": float(
                            (
                                (g["route_L"] == g["gt"])
                                & (g["route_next"] != g["gt"])
                            ).mean()
                        ),
                    }
                )

        transition_summary = pd.DataFrame(transition_rows)
        transition_summary.to_csv(
            outdir / "layer_transition_summary.csv",
            index=False,
        )

        sign_summary = summarize_signs(upd_df)
        sign_by_layer = summarize_signs_by_layer(upd_df)
        sign_summary.to_csv(
            outdir / "sign_summary.csv",
            index=False,
        )
        sign_by_layer.to_csv(
            outdir / "sign_by_layer.csv",
            index=False,
        )

        gen_summary = summarize_generation(gen_df)
        gen_by_rel = summarize_by_relation(gen_df)
        gen_summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )
        gen_by_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )

        print("=" * 210)
        print("LAYER ROUTE ACCURACY")
        print("=" * 210)
        print(
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nLAYER -> NEXT-LAYER ROUTE TRANSITIONS")
        print("-" * 210)
        print(
            transition_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nSIGN ACCURACY VS ORACLE FINAL-GRAD SIGN")
        print("-" * 210)
        print(
            sign_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nACTUAL GENERATION")
        print("-" * 210)
        print(
            gen_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        report = [
            "=" * 210,
            "LAYER-SPECIFIC DIRECTION -> FINAL vs LOCAL SPATIAL JACOBIAN",
            "=" * 210,
            f"model={a.model} repo={spec.repo_id}",
            f"N successful updates={upd_df['sid'].nunique() if len(upd_df) else 0}",
            f"global WHERE K={a.global_k}",
            f"update layers={update_layers}",
            f"heads/layer={a.heads_per_layer}",
            f"head weighting={a.head_weighting}",
            f"scale={a.scale}",
            "",
            "LAYER ROUTE ACCURACY",
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "LAYER TRANSITIONS",
            transition_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "SIGN ACCURACY",
            sign_summary.to_string(
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
            "  layer_final tests whether one sample-global relation selector was",
            "  hiding important depth-wise changes in spatial belief.",
            "",
            "  local_transition is the stronger mechanistic test:",
            "      Direction belief at L",
            "        -> actual block-L update",
            "        -> Direction readout at L+1.",
            "  Its HOW sign never reads final answer logits / sequence scores.",
            "",
            "  If local_transition predicts oracle sign above chance and improves",
            "  actual generation, that is evidence for a local spatial-to-spatial",
            "  transformation that causally constrains the later decision.",
            "",
            "  If layer_final works but local_transition fails, the useful HOW signal",
            "  is still primarily defined by downstream decision geometry rather than",
            "  by preservation of the next-layer Direction representation.",
        ]

        (outdir / "analysis_summary.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "script": "eval_layerwise_direction_local_transition_gating_v1_1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "global_template_dir": a.global_template_dir,
            "global_k": int(a.global_k),
            "prior_real_update_dir": a.prior_real_update_dir,
            "direction_selector_dir": a.direction_selector_dir,
            "source_codebook": str(code_path),
            "source_reliability": str(rel_path),
            "selector_csv": a.selector_csv,
            "selector_method": a.selector_method,
            "update_layers": update_layers,
            "route_layers": route_layers,
            "next_direction_layers": next_layers,
            "heads_per_layer": int(a.heads_per_layer),
            "head_weighting": a.head_weighting,
            "temperature": float(a.temperature),
            "pool": a.pool,
            "scale": float(a.scale),
            "layer_final_definition": (
                "actual update dot grad(final sequence score of layer-L "
                "Direction route minus strongest final competitor)"
            ),
            "local_transition_definition": (
                "actual block-L update dot gradient w.r.t. block-L output of "
                "next-layer Direction probability margin for relation predicted "
                "by Direction Heads at layer L"
            ),
            "local_transition_uses_final_answer_logits": False,
            "gt_usage": (
                "GT used only for evaluation and oracle sign/reference. "
                "Layer routes and local-transition HOW are GT-free."
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
