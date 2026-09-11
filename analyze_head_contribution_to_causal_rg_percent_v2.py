#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_head_contribution_to_causal_rg_percent_v1.py

Goal
====
For each already-selected causal TEXT state (C,p), quantify how much each
upstream attention head's natural Real-Gray change contributes to that causal
state's OWN Real-Gray direction.

For causal state (C,p):

    delta_h_RG = h_real[C,p] - h_gray[C,p]
    u_RG       = delta_h_RG / ||delta_h_RG||

Define a scalar objective:

    J_RG(C,p) = < h_real[C,p], u_RG >

For every upstream head (L,h), L <= C, using PRE-W_O head output z:

    delta_z[L,h,q] = z_real[L,h,q] - z_gray[L,h,q]

    M_RG[h -> (C,p)]
      = sum_q delta_z[L,h,q]^T
              d J_RG(C,p) / d z_real[L,h,q]

This is the same Real-Gray x gradient first-order mediation used in the supplier
scan, but now it is decomposed PER CAUSAL TOKEN instead of summing Top-K causal
states together.

Percentages
===========
1) rg_explained_pct

    100 * M_RG / ||delta_h_RG||

Why this denominator?
Because the causal state's total displacement along its own normalized RG
direction is exactly:

    <delta_h_RG, u_RG> = ||delta_h_RG||

So rg_explained_pct is the first-order fraction of the causal token's own
Real-Gray directional displacement attributable to this head.

IMPORTANT: these percentages do NOT have to sum to 100%.
Transformer computation is nonlinear and heads interact.

2) positive_share_pct

    100 * max(M_RG,0) / sum_j max(M_RG_j,0)

This DOES sum to 100% across positive head contributors for a given causal
state.  It answers:

    "Among all positive head suppliers of this causal token's RG direction,
     what share belongs to this head?"

3) abs_share_pct

    100 * |M_RG| / sum_j |M_RG_j|

This sums to 100% across absolute first-order head contributions and is useful
when positive and negative paths cancel.

Outputs
=======
head_to_causal_rg_per_state.csv
    one row per sample x causal state x upstream head

causal_rg_state_totals.csv
    decomposition totals for each causal state

head_rg_contribution_summary.csv
    head-level aggregate across causal states

direction_head_rg_contribution_summary.csv
    only known Direction heads

top_heads_per_causal_state.csv
    top positive contributors for convenient inspection

analysis_summary.txt
metadata.json
errors.jsonl

Recommended quick N=80
======================
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_head_contribution_to_causal_rg_percent_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --supplier-layers 1-26 \
  --eval-max-samples 80 \
  --top-per-state 20 \
  --output-dir output/qwen3b_head_to_causal_rg_percent_n80_v1 \
  --overwrite

Full 440
========
Same command with:

    --eval-max-samples 0
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn
import scan_qwen_spatial_heads_vs_causal_core_v1 as spatialscan


EPS = 1e-12


DEFAULT_DIRECTION_HEADS = ",".join([
    "26:3", "23:1", "23:5", "26:2", "22:9",
    "23:0", "22:13", "22:2", "21:14", "23:10",
    "21:5", "22:12", "21:1", "26:1", "27:2",
    "27:1", "21:11", "22:14", "21:3", "22:10",
])

DEFAULT_DIRECTION_ACCURACY = {
    (26,3): 0.815909, (23,1): 0.793182, (23,5): 0.786364,
    (26,2): 0.779545, (22,9): 0.775000, (23,0): 0.770455,
    (22,13): 0.770455, (22,2): 0.759091, (21,14): 0.752273,
    (23,10): 0.752273, (21,5): 0.747727, (22,12): 0.747727,
    (21,1): 0.747727, (26,1): 0.743182, (27,2): 0.734091,
    (27,1): 0.729545, (21,11): 0.725000, (22,14): 0.725000,
    (21,3): 0.722727, (22,10): 0.713636,
}


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "").replace("H", ":")
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad head specification {part!r}")
        a, b = part.split(":", 1)
        pair = (int(a), int(b))
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


def parse_categories(text: str) -> Optional[set]:
    xs = {x.strip() for x in str(text).split(",") if x.strip()}
    return xs if xs else None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_coco_meta(args):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(args.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in ("left", "right", "above", "below"):
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, args.max_samples, args.seed)
    train, heldout = traj.stratified_split(meta, 0.30, args.seed)
    return two, meta, train, heldout, rec_by_sid


def load_causal_selection(
    path: Path,
    causal_layers: Sequence[int],
    top_k: int,
    categories: Optional[set],
    allowed_sids: set,
) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {
        "sid", "rank", "source_layer", "position",
        "token", "category", "broad_category", "mediation",
    }
    missing = need - set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    d["mediation"] = pd.to_numeric(d["mediation"], errors="coerce")

    d = d[d["sid"].isin(allowed_sids)].copy()
    d = d[d["source_layer"].isin(set(map(int, causal_layers)))].copy()
    d = d[d["broad_category"].astype(str) != "visual"].copy()
    d = d[d["broad_category"].astype(str) != "last"].copy()

    if categories is not None:
        d = d[d["broad_category"].astype(str).isin(categories)].copy()

    rows = []
    for sid, g in d.groupby("sid"):
        g = g.sort_values("rank").head(int(top_k)).copy()
        g["causal_text_rank"] = np.arange(1, len(g) + 1)
        rows.append(g)

    if not rows:
        return d.iloc[:0].copy()
    return pd.concat(rows, ignore_index=True)


def load_model(args, two):
    specs = base.merged_model_specs(two)
    spec = specs[args.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"Loading {spec.repo_id}", flush=True)
    try:
        model = cls.from_pretrained(spec.repo_id, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        model = cls.from_pretrained(spec.repo_id, **kw)

    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    return model, processor, decoder_layers, decoder_path, spec


def infer_head_geometry(model, decoder_layers, head_layers):
    cfg = spatialscan.get_text_config(model)
    fallback_nh = int(cfg.num_attention_heads)
    out = {}

    for L in head_layers:
        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)

        nh = getattr(attn, "num_heads", None)
        if nh is None:
            nh = getattr(
                getattr(attn, "config", None),
                "num_attention_heads",
                None,
            )
        if nh is None:
            nh = fallback_nh
        nh = int(nh)

        hidden = int(op.in_features)
        if hidden % nh != 0:
            raise RuntimeError(
                f"L{L}: o_proj input={hidden} not divisible by heads={nh}"
            )

        out[L] = {
            "n_heads": nh,
            "head_dim": hidden // nh,
            "hidden": hidden,
        }
    return out


class GrayStateHeadCapture:
    def __init__(self, decoder_layers, state_layers, head_layers):
        self.states = {}
        self.pre_o = {}
        self.handles = []

        for L in sorted(set(map(int, state_layers))):
            def make_state_hook(layer):
                def hook(_m, _inp, out):
                    x = traj.first_tensor(out)
                    self.states[layer] = (
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook
            self.handles.append(
                decoder_layers[L].register_forward_hook(make_state_hook(L))
            )

        for L in sorted(set(map(int, head_layers))):
            attn = spatialscan.resolve_attn(decoder_layers[L])
            op = spatialscan.resolve_o_proj(attn)

            def make_pre_hook(layer):
                def hook(_m, inputs):
                    self.pre_o[layer] = (
                        inputs[0].detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook
            self.handles.append(
                op.register_forward_pre_hook(make_pre_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_gray_capture(
    model,
    decoder_layers,
    batch,
    state_layers,
    head_layers,
):
    cap = GrayStateHeadCapture(decoder_layers, state_layers, head_layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)

        ms = [L for L in state_layers if L not in cap.states]
        mh = [L for L in head_layers if L not in cap.pre_o]
        if ms or mh:
            raise RuntimeError(
                f"Gray capture missing states={ms}, head_layers={mh}"
            )
        return cap.states, cap.pre_o
    finally:
        cap.close()


class RealGraphHeadCapture:
    def __init__(
        self,
        decoder_layers,
        state_layers,
        head_layers,
        cut_layer=0,
    ):
        self.states = {}
        self.pre_o = {}
        self.handles = []
        self.cut_layer = int(cut_layer)

        def cut_hook(_m, _inp, out):
            x = traj.first_tensor(out)
            y = x.detach().clone().requires_grad_(True)
            self.states[self.cut_layer] = y
            return traj.replace_first_tensor(out, y)

        self.handles.append(
            decoder_layers[self.cut_layer].register_forward_hook(cut_hook)
        )

        for L in sorted(set(map(int, state_layers))):
            if L == self.cut_layer:
                continue

            def make_state_hook(layer):
                def hook(_m, _inp, out):
                    x = traj.first_tensor(out)
                    self.states[layer] = x
                    return None
                return hook
            self.handles.append(
                decoder_layers[L].register_forward_hook(make_state_hook(L))
            )

        for L in sorted(set(map(int, head_layers))):
            if L <= self.cut_layer:
                raise ValueError(
                    f"Supplier head L{L} must be after graph cut L{self.cut_layer}"
                )
            attn = spatialscan.resolve_attn(decoder_layers[L])
            op = spatialscan.resolve_o_proj(attn)

            def make_pre_hook(layer):
                def hook(_m, inputs):
                    self.pre_o[layer] = inputs[0]
                return hook
            self.handles.append(
                op.register_forward_pre_hook(make_pre_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def run_real_graph(
    model,
    decoder_layers,
    batch,
    state_layers,
    head_layers,
):
    cap = RealGraphHeadCapture(
        decoder_layers,
        state_layers=state_layers,
        head_layers=head_layers,
        cut_layer=0,
    )
    kw = dict(batch)
    kw["use_cache"] = False
    _ = model(**kw)

    ms = [L for L in state_layers if L not in cap.states]
    mh = [L for L in head_layers if L not in cap.pre_o]
    if ms or mh:
        cap.close()
        raise RuntimeError(
            f"Real graph missing states={ms}, head_layers={mh}"
        )
    return cap



# =============================================================================
# CLI / utils
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
    p.add_argument("--ranked-causal", required=True)

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument(
        "--causal-categories",
        default="",
        help="Optional broad categories. Empty = all non-visual text states.",
    )

    p.add_argument(
        "--supplier-layers",
        default="1-26",
        help="Upstream attention layers to decompose.",
    )
    p.add_argument(
        "--direction-heads",
        default=DEFAULT_DIRECTION_HEADS,
        help="Known Direction heads, used only as post-hoc labels.",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all available samples.",
    )
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--top-per-state",
        type=int,
        default=20,
        help="How many positive heads per causal state to save in convenience CSV.",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_mean(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def hname(L: int, h: int) -> str:
    return f"L{int(L):02d}H{int(h):02d}"


# =============================================================================
# Core per-state decomposition
# =============================================================================

def decompose_one_causal_state(
    *,
    sid: int,
    gt: str,
    causal_row,
    cap,
    gray_states: Mapping[int, np.ndarray],
    gray_pre_o: Mapping[int, np.ndarray],
    supplier_layers: Sequence[int],
    head_geometry: Mapping[int, Mapping[str, int]],
    direction_set: set,
    cats: Sequence[str],
):
    """
    Returns:
        head_rows : one row per eligible head
        state_row : totals for this causal state
    """
    C = int(causal_row.source_layer)
    p = int(causal_row.position)

    hr = cap.states[C]
    hg_np = gray_states[C]

    n_state = min(int(hr.shape[1]), int(hg_np.shape[1]))
    if not (0 <= p < n_state):
        raise RuntimeError(
            f"sid={sid} causal L{C} p={p} outside aligned length={n_state}"
        )

    hvec = hr[0, p].float()
    gray_vec = torch.as_tensor(
        hg_np[0, p],
        device=hvec.device,
        dtype=torch.float32,
    )

    delta_h = (hvec.detach() - gray_vec).float()
    rg_norm = float(delta_h.norm().item())

    if rg_norm <= EPS:
        raise RuntimeError(
            f"sid={sid} causal L{C} p={p} has zero Real-Gray norm"
        )

    u_rg = (delta_h / rg_norm).detach()

    # Scalar amount of the REAL state along the causal token's own RG direction.
    J_rg = torch.dot(hvec, u_rg)

    eligible_layers = [L for L in supplier_layers if int(L) <= C]
    preo_tensors = [cap.pre_o[L] for L in eligible_layers]

    grads = torch.autograd.grad(
        J_rg,
        preo_tensors,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    rows = []

    for L, grad in zip(eligible_layers, grads):
        zr_t = cap.pre_o[L]
        zr = zr_t.detach().float().cpu().numpy().astype(np.float32)
        zg = np.asarray(gray_pre_o[L], dtype=np.float32)

        n = min(
            int(zr.shape[1]),
            int(zg.shape[1]),
            len(cats),
        )

        H = int(head_geometry[L]["n_heads"])
        D = int(head_geometry[L]["head_dim"])

        dz = (
            zr[0, :n] - zg[0, :n]
        ).reshape(n, H, D)

        if grad is None:
            G = np.zeros_like(dz, dtype=np.float32)
        else:
            G = (
                grad[0, :n]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(n, H, D)
            )

        # Per-position mediation, then sum query positions to get head contribution.
        M_pos = np.einsum("phd,phd->ph", dz, G).astype(np.float32)

        broad = [dyn.broad_category(str(cats[q])) for q in range(n)]

        for h in range(H):
            mh = M_pos[:, h]
            M = float(mh.sum())

            # Optional source-query breakdown, useful later.
            def cat_sum(name: str) -> float:
                idx = [q for q, bc in enumerate(broad) if bc == name]
                if not idx:
                    return 0.0
                return float(mh[np.asarray(idx, dtype=np.int64)].sum())

            rows.append({
                "sid": sid,
                "gt": gt,

                "causal_text_rank": int(causal_row.causal_text_rank),
                "global_rank": int(causal_row.rank),
                "causal_layer": C,
                "causal_position": p,
                "causal_token": str(causal_row.token),
                "causal_category": str(causal_row.category),
                "causal_broad_category": str(causal_row.broad_category),
                "causal_ranking_mediation": float(causal_row.mediation),
                "causal_RG_norm": rg_norm,

                "supplier_layer": int(L),
                "head": int(h),
                "head_name": hname(L, h),
                "is_direction_head": (int(L), int(h)) in direction_set,
                "direction_coco_accuracy": DEFAULT_DIRECTION_ACCURACY.get(
                    (int(L), int(h)), np.nan
                ),

                "rg_mediation": M,
                "rg_explained_pct": 100.0 * M / rg_norm,

                "delta_z_norm": float(np.linalg.norm(dz[:, h, :])),
                "grad_norm": float(np.linalg.norm(G[:, h, :])),

                "visual_rg_mediation": cat_sum("visual"),
                "subject_rg_mediation": cat_sum("subject"),
                "reference_rg_mediation": cat_sum("reference"),
                "relation_words_rg_mediation": cat_sum("relation_words"),
                "other_text_rg_mediation": cat_sum("other_text"),
                "last_rg_mediation": cat_sum("last"),
            })

    if not rows:
        raise RuntimeError(
            f"sid={sid} causal L{C} p={p}: no eligible head rows"
        )

    # Normalize shares WITHIN THIS causal state.
    ms = np.asarray([r["rg_mediation"] for r in rows], dtype=np.float64)
    pos = np.maximum(ms, 0.0)
    abs_m = np.abs(ms)

    pos_total = float(pos.sum())
    abs_total = float(abs_m.sum())
    signed_total = float(ms.sum())

    # Rank all heads by signed positive contribution.
    order = np.argsort(-ms)
    signed_rank = np.empty(len(rows), dtype=np.int64)
    signed_rank[order] = np.arange(1, len(rows) + 1)

    order_abs = np.argsort(-abs_m)
    abs_rank = np.empty(len(rows), dtype=np.int64)
    abs_rank[order_abs] = np.arange(1, len(rows) + 1)

    for i, r in enumerate(rows):
        r["positive_share_pct"] = (
            100.0 * pos[i] / pos_total
            if pos_total > EPS else 0.0
        )
        r["abs_share_pct"] = (
            100.0 * abs_m[i] / abs_total
            if abs_total > EPS else 0.0
        )
        r["signed_supplier_rank_for_state"] = int(signed_rank[i])
        r["abs_supplier_rank_for_state"] = int(abs_rank[i])

    state_row = {
        "sid": sid,
        "gt": gt,
        "causal_text_rank": int(causal_row.causal_text_rank),
        "global_rank": int(causal_row.rank),
        "causal_layer": C,
        "causal_position": p,
        "causal_token": str(causal_row.token),
        "causal_category": str(causal_row.category),
        "causal_broad_category": str(causal_row.broad_category),

        "causal_RG_norm": rg_norm,

        "N_eligible_heads": len(rows),

        "sum_signed_head_mediation": signed_total,
        "sum_positive_head_mediation": pos_total,
        "sum_abs_head_mediation": abs_total,

        # This is the percentage obtained if ALL signed first-order head effects
        # are summed.  It need not equal 100%.
        "sum_signed_explained_pct": 100.0 * signed_total / rg_norm,
        "sum_positive_over_RG_pct": 100.0 * pos_total / rg_norm,
        "sum_abs_over_RG_pct": 100.0 * abs_total / rg_norm,

        "positive_head_fraction": float(np.mean(ms > 0)),
        "negative_head_fraction": float(np.mean(ms < 0)),
    }

    return rows, state_row


# =============================================================================
# Aggregation
# =============================================================================

def build_head_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    rows = []
    for (L, h, hn), g in df.groupby(
        ["supplier_layer", "head", "head_name"]
    ):
        row = {
            "supplier_layer": int(L),
            "head": int(h),
            "head_name": hn,
            "N_states": len(g),
            "N_samples": int(g["sid"].nunique()),
            "is_direction_head": bool(g["is_direction_head"].iloc[0]),
            "direction_coco_accuracy": safe_mean(g["direction_coco_accuracy"]),

            "mean_rg_mediation": safe_mean(g["rg_mediation"]),
            "median_rg_mediation": safe_median(g["rg_mediation"]),

            "mean_rg_explained_pct": safe_mean(g["rg_explained_pct"]),
            "median_rg_explained_pct": safe_median(g["rg_explained_pct"]),

            "mean_positive_share_pct": safe_mean(g["positive_share_pct"]),
            "median_positive_share_pct": safe_median(g["positive_share_pct"]),

            "mean_abs_share_pct": safe_mean(g["abs_share_pct"]),
            "median_abs_share_pct": safe_median(g["abs_share_pct"]),

            "positive_state_fraction": float(
                (g["rg_mediation"] > 0).mean()
            ),

            "mean_signed_state_rank": safe_mean(
                g["signed_supplier_rank_for_state"]
            ),
            "median_signed_state_rank": safe_median(
                g["signed_supplier_rank_for_state"]
            ),

            "mean_visual_rg_mediation": safe_mean(g["visual_rg_mediation"]),
            "mean_subject_rg_mediation": safe_mean(g["subject_rg_mediation"]),
            "mean_reference_rg_mediation": safe_mean(g["reference_rg_mediation"]),
            "mean_relation_words_rg_mediation": safe_mean(
                g["relation_words_rg_mediation"]
            ),
            "mean_other_text_rg_mediation": safe_mean(
                g["other_text_rg_mediation"]
            ),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    # Prefer positive directional contribution for the main aggregate rank.
    out["mean_rg_explained_rank"] = (
        out["mean_rg_explained_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    out["mean_positive_share_rank"] = (
        out["mean_positive_share_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return out.sort_values(
        ["mean_rg_explained_rank", "mean_positive_share_rank"]
    ).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers = parse_layers(a.causal_layers)
    supplier_layers = parse_layers(a.supplier_layers)
    direction_heads = parse_heads(a.direction_heads)
    direction_set = set(direction_heads)
    categories = parse_categories(a.causal_categories)

    if any(L <= 0 for L in supplier_layers):
        raise ValueError(
            "Current differentiable graph cuts after block 0; "
            "--supplier-layers must be >= 1."
        )

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    two, meta, _train, _test_unused, rec_by_sid = load_coco_meta(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    ranking_sids = set(
        pd.to_numeric(
            pd.read_csv(a.ranked_causal, usecols=["sid"])["sid"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )

    eval_meta = [
        x for x in meta
        if int(x["sid"]) in ranking_sids
    ]

    if a.eval_max_samples > 0:
        eval_meta = traj.stratified_cap(
            eval_meta,
            a.eval_max_samples,
            a.seed + 1,
        )

    eval_sids = {int(x["sid"]) for x in eval_meta}

    selected = load_causal_selection(
        Path(a.ranked_causal),
        causal_layers=causal_layers,
        top_k=a.causal_top_k,
        categories=categories,
        allowed_sids=eval_sids,
    )
    selected["gt"] = selected["sid"].map(
        lambda sid: meta_by_sid[int(sid)]["gt"]
    )
    selected.to_csv(
        outdir / "selected_causal_text_states.csv",
        index=False,
    )
    selected_by_sid = {
        int(sid): g.copy()
        for sid, g in selected.groupby("sid")
    }

    model = processor = None
    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(
            a, two
        )
        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        supplier_layers = [
            L for L in supplier_layers
            if 0 < L < n_layers
        ]

        geom = infer_head_geometry(
            model,
            decoder_layers,
            supplier_layers,
        )

        print("=" * 180)
        print("PER-HEAD CONTRIBUTION TO EACH CAUSAL TOKEN'S OWN REAL-GRAY DIRECTION")
        print("=" * 180)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"eval N={len(eval_meta)}")
        print(
            f"causal layers={causal_layers} | "
            f"causal topK={a.causal_top_k}"
        )
        print(f"supplier layers={supplier_layers}")
        print(
            "rg_explained_pct = 100 * "
            "[(delta_z_head)^T grad J_RG] / ||delta_h_causal_RG||"
        )
        print()

        all_head_rows = []
        all_state_rows = []

        for m in tqdm(eval_meta, desc="HEAD -> individual causal RG"):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            real = gray = rb = gb = cap = None
            try:
                crows = selected_by_sid[sid].copy()
                gt = m["gt"]

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                ids_r = rb["input_ids"][0].detach().cpu().tolist()
                ids_g = gb["input_ids"][0].detach().cpu().tolist()
                if ids_r != ids_g:
                    raise RuntimeError(
                        "Real/Gray token IDs differ; token alignment invalid."
                    )

                cats, toks = dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids_r,
                    m["subject"],
                    m["reference"],
                )

                state_layers_sid = sorted(
                    set(crows["source_layer"].astype(int).tolist())
                )
                max_C = max(state_layers_sid)

                supplier_sid = [
                    L for L in supplier_layers
                    if L <= max_C
                ]

                # Gray values, no graph needed.
                gray_states, gray_pre_o = run_gray_capture(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=gb,
                    state_layers=state_layers_sid,
                    head_layers=supplier_sid,
                )

                # One real graph; reuse it for each causal state.
                with torch.enable_grad():
                    cap = run_real_graph(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        state_layers=state_layers_sid,
                        head_layers=supplier_sid,
                    )

                    crows_sorted = crows.sort_values(
                        ["causal_text_rank", "rank"]
                    )

                    for crow in crows_sorted.itertuples():
                        head_rows, state_row = decompose_one_causal_state(
                            sid=sid,
                            gt=gt,
                            causal_row=crow,
                            cap=cap,
                            gray_states=gray_states,
                            gray_pre_o=gray_pre_o,
                            supplier_layers=supplier_sid,
                            head_geometry=geom,
                            direction_set=direction_set,
                            cats=cats,
                        )
                        all_head_rows.extend(head_rows)
                        all_state_rows.append(state_row)

                cap.close()
                cap = None

            except Exception as exc:
                append_jsonl(error_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-25:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid} {type(exc).__name__}: {exc}"
                )
            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        df = pd.DataFrame(all_head_rows)
        states = pd.DataFrame(all_state_rows)

        if len(df) == 0:
            raise RuntimeError("No head contribution rows produced.")

        df.to_csv(
            outdir / "head_to_causal_rg_per_state.csv",
            index=False,
        )
        states.to_csv(
            outdir / "causal_rg_state_totals.csv",
            index=False,
        )

        summary = build_head_summary(df)
        summary.to_csv(
            outdir / "head_rg_contribution_summary.csv",
            index=False,
        )

        dsummary = summary[
            summary["is_direction_head"].astype(bool)
        ].copy()
        dsummary.to_csv(
            outdir / "direction_head_rg_contribution_summary.csv",
            index=False,
        )

        # Convenience: top-N positive heads for each individual causal state.
        top_rows = []
        keys = [
            "sid",
            "causal_text_rank",
            "causal_layer",
            "causal_position",
        ]

        for _, g in df.groupby(keys):
            z = g.sort_values(
                "rg_mediation",
                ascending=False,
            ).head(max(1, int(a.top_per_state))).copy()
            z["top_positive_rank"] = np.arange(1, len(z) + 1)
            top_rows.append(z)

        top_df = (
            pd.concat(top_rows, ignore_index=True)
            if top_rows else pd.DataFrame()
        )
        top_df.to_csv(
            outdir / "top_heads_per_causal_state.csv",
            index=False,
        )

        # Summary by causal token category and Direction/non-Direction.
        cat_rows = []
        for (bc, flag), g in df.groupby(
            ["causal_broad_category", "is_direction_head"]
        ):
            cat_rows.append({
                "causal_broad_category": bc,
                "head_group": "direction" if flag else "nondirection",
                "N_rows": len(g),
                "N_states": int(
                    g[
                        ["sid", "causal_text_rank",
                         "causal_layer", "causal_position"]
                    ].drop_duplicates().shape[0]
                ),
                "mean_rg_explained_pct": safe_mean(g["rg_explained_pct"]),
                "mean_positive_share_pct": safe_mean(g["positive_share_pct"]),
                "positive_fraction": float((g["rg_mediation"] > 0).mean()),
            })

        pd.DataFrame(cat_rows).to_csv(
            outdir / "contribution_by_causal_category.csv",
            index=False,
        )

        # Console report.
        report = []
        report.append("=" * 180)
        report.append("HEAD CONTRIBUTION TO CAUSAL TOKEN REAL-GRAY DIRECTION")
        report.append("=" * 180)
        report.append(
            f"completed samples={df['sid'].nunique()} | "
            f"causal states={states.shape[0]} | "
            f"head-state rows={df.shape[0]}"
        )
        report.append("")

        report.append("TOP 30 HEADS BY MEAN RG EXPLAINED %")
        report.append("-" * 180)
        cols = [
            "mean_rg_explained_rank",
            "head_name",
            "is_direction_head",
            "direction_coco_accuracy",
            "N_states",
            "mean_rg_mediation",
            "mean_rg_explained_pct",
            "median_rg_explained_pct",
            "mean_positive_share_pct",
            "positive_state_fraction",
            "median_signed_state_rank",
        ]
        report.append(
            summary[cols].head(30).to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )
        report.append("")

        report.append("KNOWN DIRECTION HEADS")
        report.append("-" * 180)
        if len(dsummary):
            dcols = [
                "head_name",
                "direction_coco_accuracy",
                "mean_rg_explained_rank",
                "mean_rg_explained_pct",
                "median_rg_explained_pct",
                "mean_positive_share_pct",
                "positive_state_fraction",
            ]
            report.append(
                dsummary[dcols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("CAUSAL-STATE DECOMPOSITION TOTALS")
        report.append("-" * 180)
        report.append(
            "Mean signed-head total / own RG norm = "
            f"{safe_mean(states['sum_signed_explained_pct']):.3f}%"
        )
        report.append(
            "Median signed-head total / own RG norm = "
            f"{safe_median(states['sum_signed_explained_pct']):.3f}%"
        )
        report.append(
            "Mean positive-head total / own RG norm = "
            f"{safe_mean(states['sum_positive_over_RG_pct']):.3f}%"
        )
        report.append("")
        report.append(
            "NOTE: rg_explained_pct is a first-order mediation percentage "
            "and does not need to sum to 100%. positive_share_pct does sum "
            "to 100% among positive heads for each causal state."
        )

        report_text = "\n".join(report) + "\n"
        print(report_text)

        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "analyze_head_contribution_to_causal_rg_percent_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "ranked_causal": str(a.ranked_causal),
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "supplier_layers": supplier_layers,
                "eval_N_requested": a.eval_max_samples,
                "rg_explained_pct_definition": (
                    "100 * [(z_real-z_gray)^T grad_z "
                    "<h_causal_real, normalize(h_real-h_gray)>] "
                    "/ ||h_causal_real-h_causal_gray||"
                ),
                "positive_share_pct_definition": (
                    "100 * max(M_head,0) / sum_heads max(M_head,0), "
                    "computed separately for each causal state"
                ),
                "abs_share_pct_definition": (
                    "100 * abs(M_head) / sum_heads abs(M_head), "
                    "computed separately for each causal state"
                ),
                "direction_heads_are_labels_only": [
                    hname(L,h) for L,h in direction_heads
                ],
                "oracle_warning": (
                    "The selected causal tokens originate from the existing "
                    "GT/writer-guided causal ranking."
                ),
            },
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
