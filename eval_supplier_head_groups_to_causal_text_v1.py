#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_supplier_head_groups_to_causal_text_v1.py

Question
========
We already ranked attention heads by how strongly their natural Real-Gray change
supplies behaviorally causal TEXT states.  Now causally test the TOP supplier
heads by amplifying their own Real-Gray head output at the selected causal-token
positions.

For supplier Top-K, split into three groups:

    1) all_topK
    2) direction_topK
       = Direction heads that occur INSIDE supplier Top-K
    3) nondirection_topK
       = the remaining heads inside supplier Top-K

Run this for K=10 and K=20.

Intervention
============
For selected head (L,h) and selected causal state (C,p), only if L <= C:

    z_RG[L,h,p]
      = z_real[L,h,p] - z_gray[L,h,p]          # PRE-W_O head output

Project that head slice through its own O-projection block:

    m_RG[L,h,p]
      = W_O^{L,h} z_RG[L,h,p]                  # residual-space message

During a fresh REAL-image PREFILL:

    attention_out'_L[p]
      = attention_out_L[p]
        + alpha * sum_{h in selected group} m_RG[L,h,p]

Therefore:
    alpha=1 adds one extra copy of the selected heads' own sample-specific
    Real-Gray head message at the causal-token trajectory.

This is NOT:
    * a LEFT/RIGHT prototype injection;
    * a spatial probe direction injection;
    * direct editing of h_causal;
    * attention-softmax editing.

It tests whether amplifying the natural information carried by the strongest
causal-state supplier heads improves generation.

Why full head Real-Gray output?
==============================
The supplier ranking was computed from PRE-W_O head Real-Gray deltas:

    M_head = (z_real - z_gray)^T grad_z J_causal

so the intervention should amplify the SAME object that was ranked.  We do not
restrict the message to visual-source A·V here, because non-Direction supplier
heads may carry binding/routing/decision information rather than a purely visual
message.

Group construction
==================
Read head_supplier_summary.csv from:

    scan_head_suppliers_to_causal_text_states_v1.py

Sort by useful_supplier_rank and form:

    Top10 all
    Top10 Direction subset
    Top10 non-Direction subset
    Top20 all
    Top20 Direction subset
    Top20 non-Direction subset

Direction membership comes from the summary's is_direction_head column by
default, so it exactly matches the earlier supplier analysis.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_supplier_head_groups_to_causal_text_v1.py \
  --model qwen-3b \
  --supplier-summary \
    output/qwen3b_head_suppliers_to_causal_text_n80_v1/head_supplier_summary.csv \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --ks 10,20 \
  --scales 0.25,0.5,1,2,4 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_supplier_groups_causal_injection_n80_v1 \
  --overwrite

Primary outputs
===============
group_heads.csv
generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
message_stats.csv
analysis_summary.txt
metadata.json
errors.jsonl

Interpretation
==============
The most informative comparison is within the SAME K and SAME alpha:

    all_topK
    vs direction_topK
    vs nondirection_topK

Possible outcomes:

A) direction ~= all >> nondirection
   The useful supplier effect is dominated by Direction heads.

B) nondirection ~= all >> direction
   Spatially decodable heads are not the main utilization pathway; other
   supplier heads carry the decision-relevant information.

C) all > both subsets
   Direction and non-Direction suppliers are complementary / synergistic.

D) all, direction, nondirection all weak
   First-order supplier attribution does not translate into sufficient
   intervention by simple message amplification.

Notes
=====
* The causal token ranking is still oracle/writer-guided.
* This script selects heads from a supplier scan that may have been run on the
  same N=80.  That is fine for mechanism discovery, but a paper-level effect
  should be re-tested on a fresh held-out sample set / full 440 after fixing
  the head groups.
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


REL = ("left", "right", "above", "below")
EPS = 1e-12


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
    p.add_argument(
        "--supplier-summary",
        required=True,
        help="head_supplier_summary.csv from the supplier-gradient scan.",
    )
    p.add_argument("--ranked-causal", required=True)

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument(
        "--causal-categories",
        default="",
        help="Optional broad categories. Empty = all non-visual text causal states.",
    )

    p.add_argument("--ks", default="10,20")
    p.add_argument("--scales", default="0.25,0.5,1,2,4")

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all available samples.",
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


# =============================================================================
# Generic parsing
# =============================================================================

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


def parse_ints(text: str) -> List[int]:
    xs = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            xs.append(int(part))
    return xs


def parse_floats(text: str) -> List[float]:
    xs = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            xs.append(float(part))
    return xs


def parse_categories(text: str) -> Optional[set]:
    xs = {x.strip() for x in str(text).split(",") if x.strip()}
    return xs if xs else None


def hname(L: int, h: int) -> str:
    return f"L{int(L):02d}H{int(h):02d}"


def safe_mean(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            z = float(x)
        except Exception:
            continue
        if math.isfinite(z):
            vals.append(z)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            z = float(x)
        except Exception:
            continue
        if math.isfinite(z):
            vals.append(z)
    return float(np.median(vals)) if vals else float("nan")


def parse_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# =============================================================================
# Group selection
# =============================================================================

def load_supplier_groups(
    summary_path: Path,
    ks: Sequence[int],
) -> Tuple[Dict[str, List[Tuple[int, int]]], pd.DataFrame]:
    d = pd.read_csv(summary_path)

    required = {
        "layer",
        "head",
        "head_name",
        "useful_supplier_rank",
        "is_direction_head",
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(
            f"{summary_path} missing columns: {sorted(missing)}"
        )

    for c in ("layer", "head", "useful_supplier_rank"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    d["is_direction_head"] = d["is_direction_head"].map(parse_bool)

    d = d.sort_values(
        ["useful_supplier_rank", "layer", "head"]
    ).reset_index(drop=True)

    groups: Dict[str, List[Tuple[int, int]]] = {}
    rows = []

    for K in sorted(set(map(int, ks))):
        top = d.nsmallest(K, "useful_supplier_rank").copy()

        specs = [
            (f"top{K}_all", top),
            (
                f"top{K}_direction",
                top[top["is_direction_head"].astype(bool)],
            ),
            (
                f"top{K}_nondirection",
                top[~top["is_direction_head"].astype(bool)],
            ),
        ]

        for gname, g in specs:
            heads = [
                (int(r.layer), int(r.head))
                for r in g.itertuples()
            ]
            groups[gname] = heads

            for r in g.itertuples():
                rows.append({
                    "group": gname,
                    "K_parent": K,
                    "group_head_N": len(heads),
                    "layer": int(r.layer),
                    "head": int(r.head),
                    "head_name": str(r.head_name),
                    "useful_supplier_rank": int(r.useful_supplier_rank),
                    "is_direction_head": bool(r.is_direction_head),
                    "mean_useful_signed_sum": getattr(
                        r, "mean_useful_signed_sum", np.nan
                    ),
                    "mean_useful_positive_sum": getattr(
                        r, "mean_useful_positive_sum", np.nan
                    ),
                    "direction_coco_accuracy": getattr(
                        r, "direction_coco_accuracy", np.nan
                    ),
                })

    return groups, pd.DataFrame(rows)


# =============================================================================
# Causal targets
# =============================================================================

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


# =============================================================================
# Dataset/model
# =============================================================================

def load_coco_meta(args):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(args.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, args.max_samples, args.seed)
    return two, meta, rec_by_sid


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


# =============================================================================
# PRE-W_O capture
# =============================================================================

class PreOCapture:
    def __init__(
        self,
        decoder_layers,
        layers: Sequence[int],
    ):
        self.pre_o = {}
        self.handles = []

        for L in sorted(set(map(int, layers))):
            attn = spatialscan.resolve_attn(decoder_layers[L])
            op = spatialscan.resolve_o_proj(attn)

            def make_hook(layer):
                def hook(_m, inputs):
                    self.pre_o[layer] = (
                        inputs[0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def capture_pre_o(
    *,
    model,
    decoder_layers,
    batch,
    layers,
):
    cap = PreOCapture(decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.pre_o]
        if missing:
            raise RuntimeError(f"Missing PRE-W_O layers: {missing}")
        return cap.pre_o
    finally:
        cap.close()


def infer_head_geometry(
    model,
    decoder_layers,
    layers: Sequence[int],
):
    cfg = spatialscan.get_text_config(model)
    fallback_nh = int(cfg.num_attention_heads)

    geom = {}
    for L in layers:
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

        width = int(op.in_features)
        if width % nh != 0:
            raise RuntimeError(
                f"L{L}: o_proj input width={width} not divisible by heads={nh}"
            )

        geom[L] = {
            "n_heads": nh,
            "head_dim": width // nh,
        }
    return geom


# =============================================================================
# Build post-W_O Real-Gray messages for one head group
# =============================================================================

def causal_position_layer_map(causal_rows: pd.DataFrame) -> Dict[int, int]:
    """
    ranked global_unique should already have one selected state per token position.
    If duplicates occur after external changes, use the LARGEST causal layer, so an
    upstream head is not incorrectly discarded when it can reach a later selected
    state at the same token position.
    """
    out = {}
    for r in causal_rows.itertuples():
        p = int(r.position)
        C = int(r.source_layer)
        out[p] = max(out.get(p, -1), C)
    return out


def build_group_patch_map(
    *,
    decoder_layers,
    real_pre_o: Mapping[int, np.ndarray],
    gray_pre_o: Mapping[int, np.ndarray],
    head_geometry: Mapping[int, Mapping[str, int]],
    heads: Sequence[Tuple[int, int]],
    causal_rows: pd.DataFrame,
):
    """
    patch_map[L][p] = sum_h W_O^{L,h}(zR-zG)[L,h,p]
    for selected group heads with L <= causal layer at p.
    """
    pos_to_C = causal_position_layer_map(causal_rows)
    patch_map: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    edge_rows = []

    for L, h in heads:
        L, h = int(L), int(h)
        if L not in real_pre_o or L not in gray_pre_o:
            continue

        zr = np.asarray(real_pre_o[L], dtype=np.float32)
        zg = np.asarray(gray_pre_o[L], dtype=np.float32)

        n = min(zr.shape[1], zg.shape[1])
        H = int(head_geometry[L]["n_heads"])
        D = int(head_geometry[L]["head_dim"])

        if zr.shape[-1] != H * D or zg.shape[-1] != H * D:
            raise RuntimeError(
                f"L{L}: PRE-W_O widths inconsistent with H={H}, D={D}"
            )
        if not (0 <= h < H):
            raise RuntimeError(f"{hname(L,h)} invalid for H={H}")

        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)
        W = (
            op.weight.detach().float().cpu().numpy().astype(np.float32)
        )  # [Dmodel, H*D]
        Wh = W[:, h * D : (h + 1) * D]                # [Dmodel,D]

        for p, C in pos_to_C.items():
            if L > C:
                continue
            if not (0 <= p < n):
                continue

            dr = (
                zr[0, p, h * D : (h + 1) * D]
                - zg[0, p, h * D : (h + 1) * D]
            ).astype(np.float32)

            msg = (Wh @ dr).astype(np.float32)

            if p not in patch_map[L]:
                patch_map[L][p] = np.zeros_like(msg, dtype=np.float32)
            patch_map[L][p] += msg

            edge_rows.append({
                "layer": L,
                "head": h,
                "head_name": hname(L,h),
                "causal_layer": C,
                "causal_position": p,
                "pre_o_rg_norm": float(np.linalg.norm(dr)),
                "post_o_rg_message_norm": float(np.linalg.norm(msg)),
            })

    return {L: dict(v) for L, v in patch_map.items()}, edge_rows


# =============================================================================
# Attention-output patch
# =============================================================================

def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected [B,S,D], got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise RuntimeError("Could not locate 3D attention output")


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        xs = list(output)
        for i, item in enumerate(xs):
            if torch.is_tensor(item) and item.ndim == 3:
                xs[i] = replacement
                return tuple(xs)

    if isinstance(output, list):
        xs = list(output)
        for i, item in enumerate(xs):
            if torch.is_tensor(item) and item.ndim == 3:
                xs[i] = replacement
                return xs

    raise RuntimeError("Could not replace 3D attention output")


class MultiLayerPositionDelta:
    def __init__(
        self,
        *,
        decoder_layers,
        patch_map: Mapping[int, Mapping[int, np.ndarray]],
        prompt_len: int,
        scale: float,
    ):
        self.decoder_layers = decoder_layers
        self.patch_map = patch_map
        self.prompt_len = int(prompt_len)
        self.scale = float(scale)
        self.handles = []
        self.applications = defaultdict(int)

    def __enter__(self):
        for L, by_pos in self.patch_map.items():
            attn = spatialscan.resolve_attn(self.decoder_layers[int(L)])

            def make_hook(layer, pos_map):
                def hook(_module, _inputs, output):
                    hidden = first_3d(output)

                    # PREFILL only; decode usually q_len=1.
                    if int(hidden.shape[1]) != self.prompt_len:
                        return None

                    modified = hidden.clone()
                    for p, vec in pos_map.items():
                        p = int(p)
                        if not (0 <= p < int(hidden.shape[1])):
                            raise RuntimeError(
                                f"L{layer} p={p} outside q_len={hidden.shape[1]}"
                            )

                        v = torch.as_tensor(
                            vec,
                            device=hidden.device,
                            dtype=hidden.dtype,
                        )
                        if int(v.numel()) != int(hidden.shape[-1]):
                            raise RuntimeError(
                                f"L{layer}: delta dim={v.numel()} "
                                f"hidden={hidden.shape[-1]}"
                            )

                        modified[0, p] += self.scale * v
                        self.applications[(int(layer), p)] += 1

                    return replace_first_3d(output, modified)
                return hook

            self.handles.append(
                attn.register_forward_hook(
                    make_hook(int(L), dict(by_pos))
                )
            )
        return self

    def validate(self):
        expected = {
            (int(L), int(p))
            for L, by_pos in self.patch_map.items()
            for p in by_pos
        }
        bad = [
            (key, int(self.applications.get(key, 0)))
            for key in expected
            if int(self.applications.get(key, 0)) != 1
        ]
        if bad:
            raise RuntimeError(
                f"Expected each prefill patch once; mismatches={bad[:10]}"
            )

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def patched_generation(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    patch_map,
    scale,
    max_new_tokens,
):
    prompt_len = int(batch["input_ids"].shape[1])

    with MultiLayerPositionDelta(
        decoder_layers=decoder_layers,
        patch_map=patch_map,
        prompt_len=prompt_len,
        scale=scale,
    ) as patcher:
        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )

    patcher.validate()
    pred = traj.normalize_relation(base, text)
    return pred, text, dict(patcher.applications)


@torch.inference_mode()
def baseline_generation(
    *,
    model,
    processor,
    batch,
    max_new_tokens,
):
    text = base.generate_text(
        model,
        processor,
        batch,
        max_new_tokens=max_new_tokens,
    )
    pred = traj.normalize_relation(base, text)
    return pred, text


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    base_df = (
        df[df["condition"] == "baseline"]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows = []
    patched = df[df["condition"] != "baseline"].copy()

    for (K, group, scale), g in patched.groupby(
        ["K_parent", "group", "scale"]
    ):
        x = g.set_index("sid")
        common = sorted(set(base_df.index) & set(x.index))
        if not common:
            continue

        bcor = (
            base_df.loc[common]["correct"]
            .astype(bool)
            .to_numpy()
        )
        pcor = x.loc[common]["correct"].astype(bool).to_numpy()

        bpred = (
            base_df.loc[common]["prediction"]
            .astype(str)
            .to_numpy()
        )
        ppred = x.loc[common]["prediction"].astype(str).to_numpy()

        w2c = int(np.sum((~bcor) & pcor))
        c2w = int(np.sum(bcor & (~pcor)))

        rows.append({
            "K_parent": int(K),
            "group": str(group),
            "group_head_N": int(g["group_head_N"].iloc[0]),
            "scale": float(scale),
            "N": len(common),
            "baseline_accuracy": float(bcor.mean()),
            "patched_accuracy": float(pcor.mean()),
            "gain": float(pcor.mean() - bcor.mean()),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "changed": int(np.sum(bpred != ppred)),
            "wrong_N": int((~bcor).sum()),
            "repair_rate_on_wrong": (
                float(w2c / max(int((~bcor).sum()), 1))
            ),
            "preserve_rate_on_correct": (
                float(1.0 - c2w / max(int(bcor.sum()), 1))
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["K_parent", "scale", "group"])
        .reset_index(drop=True)
    )


def summarize_by_relation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gt, g in df.groupby("gt"):
        z = summarize_generation(g)
        if len(z):
            z.insert(0, "gt", gt)
            rows.append(z)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_messages(edge_df: pd.DataFrame) -> pd.DataFrame:
    if len(edge_df) == 0:
        return pd.DataFrame()

    rows = []
    for (K, group), g in edge_df.groupby(["K_parent", "group"]):
        rows.append({
            "K_parent": int(K),
            "group": group,
            "group_head_N": int(g["group_head_N"].iloc[0]),
            "N_edges": len(g),
            "N_samples": g["sid"].nunique(),
            "mean_pre_o_rg_norm": safe_mean(g["pre_o_rg_norm"]),
            "median_pre_o_rg_norm": safe_median(g["pre_o_rg_norm"]),
            "mean_post_o_rg_message_norm": safe_mean(
                g["post_o_rg_message_norm"]
            ),
            "median_post_o_rg_message_norm": safe_median(
                g["post_o_rg_message_norm"]
            ),
        })
    return pd.DataFrame(rows).sort_values(["K_parent", "group"])


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    ks = parse_ints(a.ks)
    scales = parse_floats(a.scales)
    causal_layers = parse_layers(a.causal_layers)
    categories = parse_categories(a.causal_categories)

    if not ks:
        raise ValueError("No --ks")
    if not scales:
        raise ValueError("No --scales")

    groups, group_df = load_supplier_groups(
        Path(a.supplier_summary),
        ks=ks,
    )

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    errors_path = outdir / "errors.jsonl"

    group_df.to_csv(outdir / "group_heads.csv", index=False)

    print("=" * 180)
    print("SUPPLIER TOP-K GROUPS")
    print("=" * 180)
    for gname, heads in groups.items():
        print(
            f"{gname:24s} N={len(heads):2d}  "
            + ",".join(hname(L,h) for L,h in heads)
        )
    print()

    two, meta, rec_by_sid = load_coco_meta(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    ranking_sids = set(
        pd.to_numeric(
            pd.read_csv(a.ranked_causal, usecols=["sid"])["sid"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )
    eval_meta = [x for x in meta if int(x["sid"]) in ranking_sids]
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

        all_heads = sorted(
            set(
                head
                for heads in groups.values()
                for head in heads
            )
        )
        needed_layers = sorted(set(L for L, _ in all_heads))

        for L in sorted(set(needed_layers + causal_layers)):
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"L{L} invalid; model has layers 0..{n_layers-1}"
                )

        geom = infer_head_geometry(
            model,
            decoder_layers,
            needed_layers,
        )

        # Validate head IDs.
        for L, h in all_heads:
            H = int(geom[L]["n_heads"])
            if not (0 <= h < H):
                raise ValueError(
                    f"{hname(L,h)} invalid; L{L} has {H} heads"
                )

        print("=" * 180)
        print("SUPPLIER-HEAD GROUP -> CAUSAL TEXT MESSAGE AMPLIFICATION")
        print("=" * 180)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"eval N={len(eval_meta)}")
        print(
            f"causal layers={causal_layers} TopK states={a.causal_top_k}"
        )
        print(f"scales={scales}")
        print(
            "intervention = extra full PRE-W_O head Real-Gray output "
            "projected through that head's W_O at selected causal-token positions"
        )
        print()

        gen_rows = []
        edge_rows_all = []

        for m in tqdm(eval_meta, desc="SUPPLIER GROUP INJECTION"):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            real = gray = rb = gb = None
            try:
                causal_rows = selected_by_sid[sid]

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
                        "Real/Gray token IDs differ; cannot align PRE-W_O states."
                    )

                # Only need supplier layers that can precede at least one selected
                # causal state in this sample.
                max_C = int(causal_rows["source_layer"].max())
                layers_sid = [L for L in needed_layers if L <= max_C]
                if not layers_sid:
                    raise RuntimeError(
                        f"No supplier layer <= max causal layer {max_C}"
                    )

                pre_r = capture_pre_o(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    layers=layers_sid,
                )
                pre_g = capture_pre_o(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=gb,
                    layers=layers_sid,
                )

                base_pred, base_text = baseline_generation(
                    model=model,
                    processor=processor,
                    batch=rb,
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = base_pred == m["gt"]

                gen_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "K_parent": 0,
                    "group": "baseline",
                    "group_head_N": 0,
                    "scale": 0.0,
                    "prediction": base_pred,
                    "correct": base_correct,
                    "text": base_text,
                })

                # Build each group's message map once, then sweep alpha.
                for gname, heads in groups.items():
                    K_parent = int(
                        gname.split("_", 1)[0].replace("top", "")
                    )
                    heads_sid = [
                        (L,h)
                        for L,h in heads
                        if L in layers_sid
                    ]

                    # Empty direction/non-direction subset is legal in principle,
                    # but should be explicit rather than silently patching nothing.
                    if not heads_sid:
                        continue

                    patch_map, edge_rows = build_group_patch_map(
                        decoder_layers=decoder_layers,
                        real_pre_o=pre_r,
                        gray_pre_o=pre_g,
                        head_geometry=geom,
                        heads=heads_sid,
                        causal_rows=causal_rows,
                    )

                    if not patch_map:
                        continue

                    for e in edge_rows:
                        e.update({
                            "sid": sid,
                            "gt": m["gt"],
                            "K_parent": K_parent,
                            "group": gname,
                            "group_head_N": len(heads),
                        })
                        edge_rows_all.append(e)

                    for alpha in scales:
                        pred, text, applications = patched_generation(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            scale=alpha,
                            max_new_tokens=a.max_new_tokens,
                        )
                        correct = pred == m["gt"]

                        gen_rows.append({
                            "sid": sid,
                            "gt": m["gt"],
                            "condition": gname,
                            "K_parent": K_parent,
                            "group": gname,
                            "group_head_N": len(heads),
                            "scale": float(alpha),
                            "prediction": pred,
                            "correct": correct,
                            "text": text,
                            "baseline_prediction": base_pred,
                            "baseline_correct": base_correct,
                            "wrong_to_correct": (
                                (not base_correct) and correct
                            ),
                            "correct_to_wrong": (
                                base_correct and (not correct)
                            ),
                            "patch_location_N": len(applications),
                        })

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid} {type(exc).__name__}: {exc}"
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gen_df = pd.DataFrame(gen_rows)
        edge_df = pd.DataFrame(edge_rows_all)

        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )
        edge_df.to_csv(
            outdir / "message_edges.csv",
            index=False,
        )

        summary = summarize_generation(gen_df)
        summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )

        by_rel = summarize_by_relation(gen_df)
        by_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )

        msg_summary = summarize_messages(edge_df)
        msg_summary.to_csv(
            outdir / "message_stats.csv",
            index=False,
        )

        # Add simple within-K winners by scale.
        winners = []
        if len(summary):
            for (K, alpha), g in summary.groupby(
                ["K_parent", "scale"]
            ):
                best = g.sort_values(
                    ["patched_accuracy", "net"],
                    ascending=[False, False],
                ).iloc[0]
                winners.append({
                    "K_parent": int(K),
                    "scale": float(alpha),
                    "best_group": best["group"],
                    "best_accuracy": float(best["patched_accuracy"]),
                    "best_gain": float(best["gain"]),
                    "best_net": int(best["net"]),
                })
        pd.DataFrame(winners).to_csv(
            outdir / "best_group_by_k_scale.csv",
            index=False,
        )

        report = []
        report.append("=" * 180)
        report.append("SUPPLIER TOP-K GROUP MESSAGE AMPLIFICATION")
        report.append("=" * 180)
        completed = (
            gen_df[gen_df["condition"] == "baseline"]["sid"].nunique()
            if len(gen_df)
            else 0
        )
        baseline_acc = safe_mean(
            gen_df[
                gen_df["condition"] == "baseline"
            ]["correct"].astype(float)
        )
        report.append(
            f"completed N={completed} | baseline={baseline_acc:.4f}"
        )
        report.append("")

        report.append("GROUPS")
        report.append("-" * 180)
        for gname, heads in groups.items():
            report.append(
                f"{gname:24s} N={len(heads):2d}  "
                + ",".join(hname(L,h) for L,h in heads)
            )
        report.append("")

        report.append("GENERATION")
        report.append("-" * 180)
        if len(summary):
            report.append(
                summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        report.append("")

        report.append("MESSAGE MAGNITUDE")
        report.append("-" * 180)
        if len(msg_summary):
            report.append(
                msg_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("READING GUIDE")
        report.append("-" * 180)
        report.append(
            "Compare all/direction/nondirection at the same K and alpha."
        )
        report.append(
            "direction ~= all >> nondirection: Direction suppliers dominate."
        )
        report.append(
            "nondirection ~= all >> direction: non-Direction utilization pathway dominates."
        )
        report.append(
            "all > both subsets: complementary/synergistic supplier families."
        )
        report.append(
            "all weak: supplier-gradient attribution does not become a strong repair "
            "under simple message amplification."
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
                "script": "eval_supplier_head_groups_to_causal_text_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "supplier_summary": str(a.supplier_summary),
                "ranked_causal": str(a.ranked_causal),
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "ks": ks,
                "scales": scales,
                "groups": {
                    name: [hname(L,h) for L,h in heads]
                    for name, heads in groups.items()
                },
                "intervention": (
                    "At each selected causal-token position p and supplier head "
                    "(L,h) with L<=causal layer, compute PRE-W_O z_real-z_gray, "
                    "project through that head's W_O slice, and add alpha times "
                    "the resulting residual-space message to attention_out[L,p]."
                ),
                "selection_note": (
                    "TopK is selected by useful_supplier_rank first; direction and "
                    "non-direction groups are subsets of the same TopK."
                ),
                "oracle_warning": (
                    "Existing causal-token ranking is GT/writer-guided."
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
