#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_directJ_heads_and_reproduce_causalRG_v1.py

Goal
====
1) Rank attention heads DIRECTLY by their natural Real-Gray contribution to the
   original late writer objective J (no causal-token mediation in the ranking).

       J = sum_T < h_real[T,last], normalize(s_GT^T) >

       delta_z[L,h,q] = z_real[L,h,q] - z_gray[L,h,q]   # PRE-W_O

       M_direct[L,h]
         = sum_{q in query_scope}
             delta_z[L,h,q]^T dJ/dz_real[L,h,q]

2) Compare the resulting head ranking with the previously identified spatial /
   Direction heads.

3) Select global Top-K direct-J heads and causally amplify them with ONE SHARED
   scale alpha:

       z'[L,h,q] = z_real[L,h,q] + alpha * delta_z[L,h,q]

   implemented equivalently after W_O as:

       attn_out'[L,q]
         = attn_out[L,q]
           + alpha * sum_{h selected at L} W_O^{L,h} delta_z[L,h,q]

   No head-specific coefficient is learned.

4) Compare this head intervention against the existing direct causal-state
   reference:

       h'[C,p] = h_real[C,p] + alpha_causal *
                 (h_real[C,p] - h_gray[C,p])

   for the selected Top-K causal text states (default K=7).

Primary question
================
Can a fixed subset of heads, selected by direct gradient to J, reproduce the
late-last J increase and generation improvement caused by directly amplifying
the causal tokens' Real-Gray states?

Head selection is GLOBAL, not sample-specific:
- writers are calibrated on the usual 30% TRAIN split;
- direct-J head ranking is also aggregated on TRAIN by default;
- evaluation can use all_data (for exact comparison with exploratory N=80
  causal-RG runs) or heldout (cleaner paper-style validation).

Spatial overlap
===============
If --spatial-head-csv is omitted, the script uses the fixed Qwen3B Direction
Top20 list from the prior synthetic->COCO direction-head analysis.

If supplied, --spatial-head-csv should preferably contain:
    head_name
and one of:
    direction_coco_accuracy / spatial_accuracy / coco_accuracy / accuracy

Examples
========

Exploratory run matching the previous all-data N=80 style:

CUDA_VISIBLE_DEVICES=0 python -u \
  scan_directJ_heads_and_reproduce_causalRG_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --head-layers 18-27 \
  --target-layers 32,34,35 \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --head-ks 5,10,20 \
  --head-scales 0.5,1,2 \
  --causal-scale 1 \
  --eval-scope all_data \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_directJ_heads_reproduce_causalRG_n80_v1 \
  --overwrite

Paper-cleaner heldout evaluation:

CUDA_VISIBLE_DEVICES=0 python -u \
  scan_directJ_heads_and_reproduce_causalRG_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --head-layers 18-27 \
  --target-layers 32,34,35 \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --head-ks 5,10,20 \
  --head-scales 1,2 \
  --causal-scale 1 \
  --eval-scope heldout \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_directJ_heads_reproduce_causalRG_heldout80_v1 \
  --overwrite

Outputs
=======
directJ_head_per_sample.csv
directJ_head_summary.csv
directJ_head_top.csv
spatial_overlap.csv
overlap_summary.csv
selected_head_groups.csv
eval_per_sample.csv
eval_summary.csv
eval_J_summary.csv
eval_by_relation.csv
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

# Fixed prior Direction Top20, with COCO held-out accuracies where available.
# This is used only if --spatial-head-csv is not provided.
FIXED_DIRECTION = [
    ("L26H03", 0.815909),
    ("L23H01", 0.793182),
    ("L23H05", 0.786364),
    ("L26H02", 0.779545),
    ("L22H09", 0.775000),
    ("L23H00", 0.770455),
    ("L22H13", 0.770455),
    ("L22H02", 0.759091),
    ("L21H14", 0.752273),
    ("L23H10", 0.752273),
    ("L21H05", 0.747727),
    ("L22H12", 0.747727),
    ("L21H01", 0.747727),
    ("L26H01", 0.743182),
    ("L27H02", 0.734091),
    ("L27H01", 0.729545),
    ("L21H11", 0.725000),
    ("L22H14", 0.725000),
    ("L21H03", 0.722727),
    ("L22H10", 0.713636),
]


# =============================================================================
# CLI / basic helpers
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

    p.add_argument("--head-layers", default="18-27")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)

    p.add_argument(
        "--query-scope",
        choices=["text", "all"],
        default="text",
        help="Positions summed in direct head->J ranking and amplified at eval.",
    )
    p.add_argument(
        "--exclude-last-query",
        action="store_true",
        help="Exclude the prompt-last query from head ranking/intervention.",
    )

    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)

    p.add_argument(
        "--spatial-head-csv",
        default="",
        help="Optional prior spatial-head CSV. Empty uses embedded Direction Top20.",
    )
    p.add_argument(
        "--spatial-top-n",
        type=int,
        default=20,
        help="Number of spatial heads used for overlap.",
    )

    p.add_argument("--head-ks", default="5,10,20")
    p.add_argument("--head-scales", default="0.5,1,2")
    p.add_argument("--causal-scale", type=float, default=1.0)

    p.add_argument(
        "--scan-max-samples",
        type=int,
        default=0,
        help="0 = use all TRAIN samples for global direct-J head ranking.",
    )
    p.add_argument(
        "--eval-scope",
        default="all_data",
        choices=["all_data", "heldout"],
    )
    p.add_argument("--eval-max-samples", type=int, default=80)

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def hname(L: int, h: int) -> str:
    return f"L{int(L):02d}H{int(h):02d}"


def parse_hname(name: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L?(\d+)H(\d+)", str(name).strip().upper())
    if not m:
        raise ValueError(f"Bad head name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


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


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Data / model
# =============================================================================

def load_data(a):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
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

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, heldout = traj.stratified_split(meta, a.train_ratio, a.seed)
    return two, meta, train, heldout, rec_by_sid


def load_model(a, two):
    spec = base.merged_model_specs(two)[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

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
# Writer calibration
# =============================================================================

def learn_writers(
    *,
    model,
    processor,
    decoder_layers,
    train,
    rec_by_sid,
    target_layers,
    gray_value,
    device,
    writer_mode,
    error_path,
):
    q_by_sid = {}

    for m in tqdm(train, desc="CALIBRATE late writers"):
        sid = int(m["sid"])
        real = gray = None
        try:
            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")
            gray = dyn.make_gray_image(real, gray_value)

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

            hr = dyn.capture_cpu(model, decoder_layers, rb, target_layers)
            hg = dyn.capture_cpu(model, decoder_layers, gb, target_layers)

            q_by_sid[sid] = {
                T: (hr[T][0, -1] - hg[T][0, -1]).astype(np.float32)
                for T in target_layers
            }

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "writer_calibration",
                "sid": sid,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            if real is not None:
                with contextlib.suppress(Exception):
                    real.close()
            if gray is not None:
                with contextlib.suppress(Exception):
                    gray.close()

    writers, geometry = dyn.learn_writers(
        train, q_by_sid, target_layers, writer_mode
    )
    return writers, geometry


def writer_value_np(
    states: Mapping[int, np.ndarray],
    writers_for_relation: Mapping[int, np.ndarray],
    target_layers: Sequence[int],
) -> float:
    val = 0.0
    for T in target_layers:
        shat = normalize_np(writers_for_relation[T])
        val += float(np.dot(states[T][0, -1].astype(np.float32), shat))
    return val


def build_writer_objective(
    states: Mapping[int, torch.Tensor],
    writers_for_relation: Mapping[int, np.ndarray],
    target_layers: Sequence[int],
):
    terms = []
    for T in target_layers:
        shat = torch.as_tensor(
            normalize_np(writers_for_relation[T]),
            device=states[T].device,
            dtype=torch.float32,
        )
        terms.append(torch.dot(states[T][0, -1].float(), shat))
    return torch.stack(terms).sum()


# =============================================================================
# Head geometry / captures
# =============================================================================

def infer_head_geometry(model, decoder_layers, layers):
    cfg = spatialscan.get_text_config(model)
    fallback_heads = int(cfg.num_attention_heads)
    out = {}

    for L in layers:
        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)

        H = getattr(attn, "num_heads", None)
        if H is None:
            H = getattr(getattr(attn, "config", None), "num_attention_heads", None)
        if H is None:
            H = fallback_heads

        H = int(H)
        width = int(op.in_features)
        if width % H != 0:
            raise RuntimeError(f"L{L}: o_proj.in_features={width}, heads={H}")

        out[L] = {
            "n_heads": H,
            "head_dim": width // H,
        }

    return out


class CpuCapture:
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

            self.handles.append(op.register_forward_pre_hook(make_pre_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_cpu_capture(
    *,
    model,
    decoder_layers,
    batch,
    state_layers,
    head_layers,
):
    cap = CpuCapture(decoder_layers, state_layers, head_layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)

        ms = [L for L in state_layers if L not in cap.states]
        mh = [L for L in head_layers if L not in cap.pre_o]
        if ms or mh:
            raise RuntimeError(f"Missing states={ms}, pre_o={mh}")

        return cap.states, cap.pre_o
    finally:
        cap.close()


class GraphCapture:
    """
    Cut after decoder block 0 so downstream activations form a differentiable
    graph even though model parameters are frozen. Capture target states and
    PRE-W_O head outputs.
    """
    def __init__(
        self,
        decoder_layers,
        target_layers,
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
            return traj.replace_first_tensor(out, y)

        self.handles.append(
            decoder_layers[self.cut_layer].register_forward_hook(cut_hook)
        )

        for L in sorted(set(map(int, target_layers))):
            def make_state_hook(layer):
                def hook(_m, _inp, out):
                    self.states[layer] = traj.first_tensor(out)
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
                    self.pre_o[layer] = inputs[0]
                return hook

            self.handles.append(op.register_forward_pre_hook(make_pre_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def query_positions(
    *,
    model,
    processor,
    batch,
    ids,
    scope,
    exclude_last,
):
    S = len(ids)

    if scope == "all":
        qs = list(range(S))
    else:
        vis = set(
            map(
                int,
                base.resolve_visual_indices(
                    model, processor, batch, ids
                ),
            )
        )
        qs = [q for q in range(S) if q not in vis]

    if exclude_last and qs:
        last = S - 1
        qs = [q for q in qs if q != last]

    return qs


# =============================================================================
# Direct J head scan
# =============================================================================

def scan_one_sample(
    *,
    model,
    processor,
    decoder_layers,
    rb,
    gb,
    gt,
    writers,
    target_layers,
    head_layers,
    geom,
    query_scope,
    exclude_last,
):
    ids = rb["input_ids"][0].detach().cpu().tolist()
    ids_g = gb["input_ids"][0].detach().cpu().tolist()
    if ids != ids_g:
        raise RuntimeError("Real/Gray prompt tokenization differs.")

    qpos = query_positions(
        model=model,
        processor=processor,
        batch=rb,
        ids=ids,
        scope=query_scope,
        exclude_last=exclude_last,
    )

    # Gray PRE-W_O reference.
    _, gray_pre = run_cpu_capture(
        model=model,
        decoder_layers=decoder_layers,
        batch=gb,
        state_layers=[],
        head_layers=head_layers,
    )

    cap = GraphCapture(
        decoder_layers,
        target_layers=target_layers,
        head_layers=head_layers,
        cut_layer=0,
    )

    try:
        kw = dict(rb)
        kw["use_cache"] = False
        _ = model(**kw)

        writers_for_relation = {
            T: writers[T][gt] for T in target_layers
        }

        J = build_writer_objective(
            cap.states,
            writers_for_relation,
            target_layers,
        )

        tensors = [cap.pre_o[L] for L in head_layers]
        grads = torch.autograd.grad(
            J,
            tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        rows = []

        for L, g in zip(head_layers, grads):
            if g is None:
                continue

            zr = cap.pre_o[L].detach().float().cpu().numpy().astype(np.float32)
            zg = gray_pre[L]
            gg = g.detach().float().cpu().numpy().astype(np.float32)

            n = min(zr.shape[1], zg.shape[1], gg.shape[1])
            q = [x for x in qpos if x < n]
            if not q:
                continue

            H = int(geom[L]["n_heads"])
            D = int(geom[L]["head_dim"])

            for h in range(H):
                sl = slice(h * D, (h + 1) * D)

                dz = zr[0, q, sl] - zg[0, q, sl]
                gh = gg[0, q, sl]

                edge = np.sum(dz * gh, axis=-1)
                signed = float(edge.sum())
                pos = float(np.maximum(edge, 0).sum())
                neg = float(np.minimum(edge, 0).sum())
                ab = float(np.abs(edge).sum())

                rows.append({
                    "layer": L,
                    "head": h,
                    "head_name": hname(L, h),
                    "signed_sum": signed,
                    "positive_sum": pos,
                    "negative_sum": neg,
                    "absolute_sum": ab,
                    "max_query_score": float(edge.max()),
                    "min_query_score": float(edge.min()),
                    "positive_query_fraction": float(np.mean(edge > 0)),
                    "delta_z_norm": float(np.linalg.norm(dz)),
                    "grad_norm": float(np.linalg.norm(gh)),
                    "N_query": len(q),
                    "J_real": float(J.detach().cpu()),
                })

        return rows

    finally:
        cap.close()


def aggregate_head_scan(per_sample: pd.DataFrame) -> pd.DataFrame:
    if len(per_sample) == 0:
        return pd.DataFrame()

    rows = []
    for (L, h, name), g in per_sample.groupby(["layer", "head", "head_name"]):
        rows.append({
            "layer": int(L),
            "head": int(h),
            "head_name": str(name),
            "N_samples": int(g["sid"].nunique()),
            "mean_signed_sum": safe_mean(g["signed_sum"]),
            "median_signed_sum": safe_median(g["signed_sum"]),
            "mean_positive_sum": safe_mean(g["positive_sum"]),
            "mean_negative_sum": safe_mean(g["negative_sum"]),
            "mean_absolute_sum": safe_mean(g["absolute_sum"]),
            "positive_sample_fraction": float(np.mean(g["signed_sum"] > 0)),
            "mean_positive_query_fraction": safe_mean(g["positive_query_fraction"]),
            "mean_delta_z_norm": safe_mean(g["delta_z_norm"]),
            "mean_grad_norm": safe_mean(g["grad_norm"]),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["mean_signed_sum", "mean_positive_sum"],
        ascending=[False, False],
    ).reset_index(drop=True)
    out["directJ_rank"] = np.arange(1, len(out) + 1)
    return out


# =============================================================================
# Spatial-head overlap
# =============================================================================

def load_spatial_heads(path: str, top_n: int) -> pd.DataFrame:
    if not path:
        d = pd.DataFrame(
            FIXED_DIRECTION,
            columns=["head_name", "spatial_accuracy"],
        )
        d["spatial_rank"] = np.arange(1, len(d) + 1)
        return d.head(top_n).copy()

    d = pd.read_csv(path)

    if "head_name" not in d.columns:
        layer_col = next(
            (c for c in ["layer", "head_layer", "L"] if c in d.columns),
            None,
        )
        head_col = next(
            (c for c in ["head", "head_idx", "H"] if c in d.columns),
            None,
        )
        if layer_col is None or head_col is None:
            raise RuntimeError(
                f"{path} needs head_name or layer/head columns. "
                f"Columns={list(d.columns)}"
            )
        d["head_name"] = [
            hname(int(L), int(h))
            for L, h in zip(d[layer_col], d[head_col])
        ]

    acc_col = next(
        (
            c for c in [
                "direction_coco_accuracy",
                "spatial_accuracy",
                "coco_accuracy",
                "accuracy",
                "img_accuracy_mean",
            ]
            if c in d.columns
        ),
        None,
    )

    if acc_col is not None:
        d["spatial_accuracy"] = pd.to_numeric(d[acc_col], errors="coerce")
        d = d.sort_values("spatial_accuracy", ascending=False)
    else:
        d["spatial_accuracy"] = np.nan

    d = d.drop_duplicates("head_name").head(top_n).copy()
    d["spatial_rank"] = np.arange(1, len(d) + 1)
    return d[["head_name", "spatial_accuracy", "spatial_rank"]]


def overlap_tables(
    head_summary: pd.DataFrame,
    spatial: pd.DataFrame,
    ks: Sequence[int],
):
    sset = set(spatial["head_name"].astype(str))
    merged = head_summary.merge(
        spatial,
        on="head_name",
        how="left",
    )
    merged["is_spatial_head"] = merged["head_name"].isin(sset)

    rows = []
    for K in sorted(set(list(ks) + [5, 10, 20, 30, 50])):
        K = min(int(K), len(head_summary))
        if K <= 0:
            continue
        top = head_summary.head(K)
        ov = [x for x in top["head_name"] if x in sset]
        rows.append({
            "directJ_topK": K,
            "spatial_N": len(sset),
            "overlap_N": len(ov),
            "overlap_fraction_of_directJ_topK": len(ov) / K,
            "overlap_fraction_of_spatial": len(ov) / max(len(sset), 1),
            "overlap_heads": ",".join(ov),
        })

    return merged, pd.DataFrame(rows)


# =============================================================================
# Causal Top-K load
# =============================================================================

def load_causal_selection(
    path: Path,
    allowed_sids: set,
    causal_layers: Sequence[int],
    top_k: int,
):
    d = pd.read_csv(path)

    required = {
        "sid", "rank", "source_layer", "position",
        "token", "category", "broad_category",
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)

    d = d[d["sid"].isin(allowed_sids)].copy()
    d = d[d["source_layer"].isin(set(map(int, causal_layers)))].copy()
    d = d[d["broad_category"].astype(str) != "visual"].copy()
    d = d[d["broad_category"].astype(str) != "last"].copy()

    rows = []
    for sid, g in d.groupby("sid"):
        z = g.sort_values("rank").head(int(top_k)).copy()
        z["causal_text_rank"] = np.arange(1, len(z) + 1)
        rows.append(z)

    if not rows:
        return d.iloc[:0].copy()

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Build head and causal patch maps
# =============================================================================

def build_head_patch_map(
    *,
    decoder_layers,
    real_pre,
    gray_pre,
    geom,
    selected_heads,
    query_positions_list,
):
    patch_map = defaultdict(dict)
    stat_rows = []

    for L, h in selected_heads:
        L, h = int(L), int(h)

        if L not in real_pre or L not in gray_pre:
            continue

        zr = real_pre[L]
        zg = gray_pre[L]
        n = min(zr.shape[1], zg.shape[1])
        qs = [q for q in query_positions_list if q < n]
        if not qs:
            continue

        H = int(geom[L]["n_heads"])
        D = int(geom[L]["head_dim"])
        if not (0 <= h < H):
            raise RuntimeError(f"{hname(L,h)} invalid; H={H}")

        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)
        W = op.weight.detach().float().cpu().numpy().astype(np.float32)
        Wh = W[:, h * D:(h + 1) * D]

        norms = []
        for q in qs:
            dz = (
                zr[0, q, h * D:(h + 1) * D]
                - zg[0, q, h * D:(h + 1) * D]
            ).astype(np.float32)

            msg = (Wh @ dz).astype(np.float32)

            if q not in patch_map[L]:
                patch_map[L][q] = np.zeros_like(msg)
            patch_map[L][q] += msg
            norms.append(float(np.linalg.norm(msg)))

        stat_rows.append({
            "layer": L,
            "head": h,
            "head_name": hname(L, h),
            "N_query": len(qs),
            "mean_postO_RG_message_norm": safe_mean(norms),
            "sum_postO_RG_message_norm": float(np.sum(norms)),
        })

    return {L: dict(mp) for L, mp in patch_map.items()}, stat_rows


def build_causal_patch_map(
    *,
    real_states,
    gray_states,
    causal_rows,
):
    patch_map = defaultdict(dict)

    for r in causal_rows.itertuples():
        C = int(r.source_layer)
        p = int(r.position)

        if C not in real_states or C not in gray_states:
            continue

        n = min(real_states[C].shape[1], gray_states[C].shape[1])
        if not (0 <= p < n):
            continue

        delta = (
            real_states[C][0, p]
            - gray_states[C][0, p]
        ).astype(np.float32)

        # In case same (C,p) appears twice, only apply once.
        patch_map[C][p] = delta

    return {L: dict(mp) for L, mp in patch_map.items()}


# =============================================================================
# Patch machinery
# =============================================================================

def first_3d(output):
    if torch.is_tensor(output):
        if output.ndim == 3:
            return output
    elif isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x) and x.ndim == 3:
                return x
    raise RuntimeError("Could not locate 3D tensor in module output.")


def replace_first_3d(output, replacement):
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        xs = list(output)
        for i, x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim == 3:
                xs[i] = replacement
                return tuple(xs)

    if isinstance(output, list):
        xs = list(output)
        for i, x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim == 3:
                xs[i] = replacement
                return xs

    raise RuntimeError("Could not replace 3D tensor in module output.")


class AttentionOutputPatch:
    def __init__(self, decoder_layers, patch_map, prompt_len, scale):
        self.decoder_layers = decoder_layers
        self.patch_map = patch_map
        self.prompt_len = int(prompt_len)
        self.scale = float(scale)
        self.handles = []
        self.counts = defaultdict(int)

    def __enter__(self):
        for L, by_pos in self.patch_map.items():
            attn = spatialscan.resolve_attn(self.decoder_layers[int(L)])

            def make_hook(layer, pos_map):
                def hook(_m, _inp, out):
                    x = first_3d(out)

                    # Prefill only.
                    if int(x.shape[1]) != self.prompt_len:
                        return None

                    y = x.clone()

                    for q, vec in pos_map.items():
                        q = int(q)
                        if q >= int(y.shape[1]):
                            continue
                        v = torch.as_tensor(
                            vec,
                            device=y.device,
                            dtype=y.dtype,
                        )
                        y[0, q] += self.scale * v
                        self.counts[(int(layer), q)] += 1

                    return replace_first_3d(out, y)

                return hook

            self.handles.append(
                attn.register_forward_hook(make_hook(int(L), dict(by_pos)))
            )

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class BlockOutputPatch:
    def __init__(self, decoder_layers, patch_map, prompt_len, scale):
        self.decoder_layers = decoder_layers
        self.patch_map = patch_map
        self.prompt_len = int(prompt_len)
        self.scale = float(scale)
        self.handles = []

    def __enter__(self):
        for L, by_pos in self.patch_map.items():

            def make_hook(layer, pos_map):
                def hook(_m, _inp, out):
                    x = traj.first_tensor(out)

                    if int(x.shape[1]) != self.prompt_len:
                        return None

                    y = x.clone()
                    for p, vec in pos_map.items():
                        p = int(p)
                        if p >= int(y.shape[1]):
                            continue
                        y[0, p] += self.scale * torch.as_tensor(
                            vec,
                            device=y.device,
                            dtype=y.dtype,
                        )

                    return traj.replace_first_tensor(out, y)

                return hook

            self.handles.append(
                self.decoder_layers[int(L)].register_forward_hook(
                    make_hook(int(L), dict(by_pos))
                )
            )

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class PrefillTargetCapture:
    def __init__(self, decoder_layers, target_layers, prompt_len):
        self.states = {}
        self.handles = []
        self.prompt_len = int(prompt_len)

        for T in target_layers:
            def make_hook(layer):
                def hook(_m, _inp, out):
                    x = traj.first_tensor(out)
                    if int(x.shape[1]) == self.prompt_len:
                        self.states[layer] = (
                            x.detach().float().cpu().numpy().astype(np.float32)
                        )
                    return None
                return hook

            self.handles.append(
                decoder_layers[int(T)].register_forward_hook(make_hook(int(T)))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def generation_with_patch(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    target_layers,
    writers_for_relation,
    max_new_tokens,
    attention_patch_map=None,
    block_patch_map=None,
    scale=1.0,
):
    prompt_len = int(batch["input_ids"].shape[1])
    cap = PrefillTargetCapture(
        decoder_layers,
        target_layers,
        prompt_len,
    )

    try:
        attn_ctx = (
            AttentionOutputPatch(
                decoder_layers,
                attention_patch_map,
                prompt_len,
                scale,
            )
            if attention_patch_map
            else contextlib.nullcontext()
        )
        block_ctx = (
            BlockOutputPatch(
                decoder_layers,
                block_patch_map,
                prompt_len,
                scale,
            )
            if block_patch_map
            else contextlib.nullcontext()
        )

        with attn_ctx:
            with block_ctx:
                text = base.generate_text(
                    model,
                    processor,
                    batch,
                    max_new_tokens=max_new_tokens,
                )

        missing = [T for T in target_layers if T not in cap.states]
        if missing:
            raise RuntimeError(f"Generation did not capture targets {missing}")

        pred = traj.normalize_relation(base, text)
        J = writer_value_np(
            cap.states,
            writers_for_relation,
            target_layers,
        )

        return pred, text, J

    finally:
        cap.close()


# =============================================================================
# Eval summaries
# =============================================================================

def summarize_eval(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    b = (
        df[df["condition"] == "baseline"]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows = []

    for (condition, K, alpha), g in df[
        df["condition"] != "baseline"
    ].groupby(["condition", "head_K", "alpha"], dropna=False):

        x = g.set_index("sid")
        common = sorted(set(b.index) & set(x.index))
        if not common:
            continue

        bc = b.loc[common, "correct"].astype(bool).to_numpy()
        pc = x.loc[common, "correct"].astype(bool).to_numpy()
        bp = b.loc[common, "prediction"].astype(str).to_numpy()
        pp = x.loc[common, "prediction"].astype(str).to_numpy()

        w2c = int(np.sum((~bc) & pc))
        c2w = int(np.sum(bc & (~pc)))

        rows.append({
            "condition": condition,
            "head_K": K,
            "alpha": float(alpha),
            "N": len(common),
            "baseline_accuracy": float(bc.mean()),
            "patched_accuracy": float(pc.mean()),
            "gain": float(pc.mean() - bc.mean()),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "changed": int(np.sum(bp != pp)),
            "repair_rate_on_wrong": float(
                w2c / max(int((~bc).sum()), 1)
            ),
            "preserve_rate_on_correct": float(
                1.0 - c2w / max(int(bc.sum()), 1)
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "head_K", "alpha"])
        .reset_index(drop=True)
    )


def summarize_J(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    baseJ = (
        df[df["condition"] == "baseline"]
        .drop_duplicates("sid")
        .set_index("sid")["J"]
    )

    # causal reference per sample
    causal = (
        df[df["condition"] == "causal_rg"]
        .drop_duplicates("sid")
        .set_index("sid")["J"]
    )

    rows = []

    for (condition, K, alpha), g in df[
        df["condition"] != "baseline"
    ].groupby(["condition", "head_K", "alpha"], dropna=False):

        x = g.set_index("sid")["J"]
        common = sorted(set(baseJ.index) & set(x.index))
        if not common:
            continue

        d = x.loc[common].to_numpy(float) - baseJ.loc[common].to_numpy(float)

        row = {
            "condition": condition,
            "head_K": K,
            "alpha": float(alpha),
            "N": len(common),
            "mean_J": float(x.loc[common].mean()),
            "mean_delta_J": float(d.mean()),
            "median_delta_J": float(np.median(d)),
            "positive_delta_J_fraction": float(np.mean(d > 0)),
        }

        common_c = sorted(set(common) & set(causal.index))
        if common_c:
            d_head = (
                x.loc[common_c].to_numpy(float)
                - baseJ.loc[common_c].to_numpy(float)
            )
            d_causal = (
                causal.loc[common_c].to_numpy(float)
                - baseJ.loc[common_c].to_numpy(float)
            )

            denom = float(d_causal.mean())
            row["causal_reference_mean_delta_J"] = float(d_causal.mean())
            row["mean_deltaJ_reproduction_ratio"] = (
                float(d_head.mean() / denom)
                if abs(denom) > EPS else np.nan
            )

            good = np.abs(d_causal) > 1e-8
            row["median_samplewise_reproduction_ratio"] = (
                float(np.median(d_head[good] / d_causal[good]))
                if np.any(good) else np.nan
            )
        else:
            row["causal_reference_mean_delta_J"] = np.nan
            row["mean_deltaJ_reproduction_ratio"] = np.nan
            row["median_samplewise_reproduction_ratio"] = np.nan

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "head_K", "alpha"])
        .reset_index(drop=True)
    )


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    head_layers = parse_layers(a.head_layers)
    target_layers = parse_layers(a.target_layers)
    causal_layers = parse_layers(a.causal_layers)
    head_ks = parse_ints(a.head_ks)
    head_scales = parse_floats(a.head_scales)

    if min(head_layers) <= 0:
        raise ValueError(
            "Direct-gradient capture currently requires head layers >=1 "
            "because block 0 is used as the differentiable graph cut."
        )

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    error_path = outdir / "errors.jsonl"

    two, meta, train, heldout, rec_by_sid = load_data(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    model = processor = None

    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        device = torch.device(a.device)

        if max(target_layers) >= len(decoder_layers):
            raise ValueError(
                f"Target layer {max(target_layers)} >= n_layers={len(decoder_layers)}"
            )

        geom = infer_head_geometry(model, decoder_layers, head_layers)

        # ---------------------------------------------------------------------
        # Phase 1: calibrate the SAME old late writer family.
        # ---------------------------------------------------------------------
        writers, writer_geometry = learn_writers(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            train=train,
            rec_by_sid=rec_by_sid,
            target_layers=target_layers,
            gray_value=a.gray_value,
            device=device,
            writer_mode=a.writer_mode,
            error_path=error_path,
        )

        pd.DataFrame(writer_geometry).to_csv(
            outdir / "writer_geometry.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Phase 2: global direct J -> head scan on TRAIN.
        # ---------------------------------------------------------------------
        scan_meta = list(train)
        if a.scan_max_samples > 0:
            scan_meta = traj.stratified_cap(
                scan_meta,
                a.scan_max_samples,
                a.seed + 13,
            )

        scan_rows = []

        for m in tqdm(scan_meta, desc="SCAN direct J -> heads"):
            sid = int(m["sid"])
            real = gray = None

            try:
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

                with torch.enable_grad():
                    rows = scan_one_sample(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        rb=rb,
                        gb=gb,
                        gt=m["gt"],
                        writers=writers,
                        target_layers=target_layers,
                        head_layers=head_layers,
                        geom=geom,
                        query_scope=a.query_scope,
                        exclude_last=a.exclude_last_query,
                    )

                for r in rows:
                    r.update({
                        "sid": sid,
                        "gt": m["gt"],
                    })
                    scan_rows.append(r)

            except Exception as exc:
                append_jsonl(error_path, {
                    "phase": "directJ_head_scan",
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
                tqdm.write(
                    f"[SCAN ERROR] sid={sid}: {type(exc).__name__}: {exc}"
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

        per_head_sample = pd.DataFrame(scan_rows)
        per_head_sample.to_csv(
            outdir / "directJ_head_per_sample.csv",
            index=False,
        )

        head_summary = aggregate_head_scan(per_head_sample)
        if len(head_summary) == 0:
            raise RuntimeError("Direct-J head scan produced no valid heads.")

        head_summary.to_csv(
            outdir / "directJ_head_summary.csv",
            index=False,
        )
        head_summary.head(max(max(head_ks), 50)).to_csv(
            outdir / "directJ_head_top.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Phase 3: overlap with prior spatial heads.
        # ---------------------------------------------------------------------
        spatial = load_spatial_heads(
            a.spatial_head_csv,
            a.spatial_top_n,
        )
        spatial.to_csv(outdir / "spatial_heads_used.csv", index=False)

        overlap_detail, overlap_summary = overlap_tables(
            head_summary,
            spatial,
            head_ks,
        )
        overlap_detail.to_csv(
            outdir / "spatial_overlap.csv",
            index=False,
        )
        overlap_summary.to_csv(
            outdir / "overlap_summary.csv",
            index=False,
        )

        # Optional Spearman on heads having spatial accuracies.
        matching = overlap_detail[
            overlap_detail["spatial_accuracy"].notna()
        ].copy()

        spatial_spearman = np.nan
        if len(matching) >= 3:
            try:
                from scipy.stats import spearmanr
                spatial_spearman = float(
                    spearmanr(
                        matching["mean_signed_sum"].to_numpy(float),
                        matching["spatial_accuracy"].to_numpy(float),
                    ).statistic
                )
            except Exception:
                spatial_spearman = float(
                    matching["mean_signed_sum"].rank().corr(
                        matching["spatial_accuracy"].rank()
                    )
                )

        # ---------------------------------------------------------------------
        # Build fixed global head groups.
        # ---------------------------------------------------------------------
        selected_groups = {}
        group_rows = []

        positive_global = head_summary[
            head_summary["mean_signed_sum"] > 0
        ].copy()

        for K in head_ks:
            z = positive_global.head(int(K))
            heads = [
                (int(r.layer), int(r.head))
                for r in z.itertuples()
            ]
            selected_groups[int(K)] = heads

            for rank_in_group, r in enumerate(z.itertuples(), 1):
                group_rows.append({
                    "head_K": int(K),
                    "rank_in_group": rank_in_group,
                    "layer": int(r.layer),
                    "head": int(r.head),
                    "head_name": str(r.head_name),
                    "directJ_rank": int(r.directJ_rank),
                    "mean_signed_sum": float(r.mean_signed_sum),
                    "mean_positive_sum": float(r.mean_positive_sum),
                    "positive_sample_fraction": float(r.positive_sample_fraction),
                    "is_spatial_head": str(r.head_name) in set(spatial.head_name),
                })

        pd.DataFrame(group_rows).to_csv(
            outdir / "selected_head_groups.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Phase 4: evaluation.
        # ---------------------------------------------------------------------
        rank_sids = set(
            pd.to_numeric(
                pd.read_csv(
                    a.ranked_causal,
                    usecols=["sid"],
                )["sid"],
                errors="coerce",
            ).dropna().astype(int).tolist()
        )

        eval_pool = meta if a.eval_scope == "all_data" else heldout
        eval_meta = [
            x for x in eval_pool
            if int(x["sid"]) in rank_sids
        ]

        if a.eval_max_samples > 0:
            eval_meta = traj.stratified_cap(
                eval_meta,
                a.eval_max_samples,
                a.seed + 1,
            )

        eval_sids = {int(x["sid"]) for x in eval_meta}

        causal_sel = load_causal_selection(
            Path(a.ranked_causal),
            eval_sids,
            causal_layers,
            a.causal_top_k,
        )
        causal_by_sid = {
            int(sid): g.copy()
            for sid, g in causal_sel.groupby("sid")
        }

        eval_rows = []
        message_rows = []

        eval_state_layers = sorted(
            set(causal_layers + target_layers)
        )

        for m in tqdm(eval_meta, desc="EVAL head amplification vs causal RG"):
            sid = int(m["sid"])
            if sid not in causal_by_sid:
                continue

            real = gray = None

            try:
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

                ids = rb["input_ids"][0].detach().cpu().tolist()
                ids_g = gb["input_ids"][0].detach().cpu().tolist()
                if ids != ids_g:
                    raise RuntimeError("Real/Gray tokenization mismatch.")

                qpos = query_positions(
                    model=model,
                    processor=processor,
                    batch=rb,
                    ids=ids,
                    scope=a.query_scope,
                    exclude_last=a.exclude_last_query,
                )

                real_states, real_pre = run_cpu_capture(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    state_layers=eval_state_layers,
                    head_layers=head_layers,
                )
                gray_states, gray_pre = run_cpu_capture(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=gb,
                    state_layers=eval_state_layers,
                    head_layers=head_layers,
                )

                writers_for_relation = {
                    T: writers[T][m["gt"]] for T in target_layers
                }

                baseline_J = writer_value_np(
                    real_states,
                    writers_for_relation,
                    target_layers,
                )

                # Baseline actual generation.
                base_pred, base_text = base.generate_text(
                    model,
                    processor,
                    rb,
                    max_new_tokens=a.max_new_tokens,
                ), None

                # base.generate_text returns text, not pred.
                if isinstance(base_pred, str):
                    base_text = base_pred
                    base_pred = traj.normalize_relation(base, base_text)
                else:
                    base_text = str(base_pred)
                    base_pred = traj.normalize_relation(base, base_text)

                base_ok = base_pred == m["gt"]

                eval_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "head_K": 0,
                    "alpha": 0.0,
                    "prediction": base_pred,
                    "correct": base_ok,
                    "text": base_text,
                    "J": baseline_J,
                    "delta_J": 0.0,
                })

                # -------------------------------------------------------------
                # Direct causal-token RG reference.
                # -------------------------------------------------------------
                causal_rows = causal_by_sid[sid].sort_values("rank")
                causal_patch = build_causal_patch_map(
                    real_states=real_states,
                    gray_states=gray_states,
                    causal_rows=causal_rows,
                )

                cpred, ctext, cJ = generation_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    target_layers=target_layers,
                    writers_for_relation=writers_for_relation,
                    max_new_tokens=a.max_new_tokens,
                    block_patch_map=causal_patch,
                    scale=a.causal_scale,
                )
                cok = cpred == m["gt"]

                eval_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "causal_rg",
                    "head_K": 0,
                    "alpha": float(a.causal_scale),
                    "prediction": cpred,
                    "correct": cok,
                    "text": ctext,
                    "J": cJ,
                    "delta_J": cJ - baseline_J,
                    "wrong_to_correct": (not base_ok) and cok,
                    "correct_to_wrong": base_ok and (not cok),
                })

                # -------------------------------------------------------------
                # Fixed global direct-J Top-K head groups.
                # -------------------------------------------------------------
                for K, heads in selected_groups.items():
                    patch_map, stats = build_head_patch_map(
                        decoder_layers=decoder_layers,
                        real_pre=real_pre,
                        gray_pre=gray_pre,
                        geom=geom,
                        selected_heads=heads,
                        query_positions_list=qpos,
                    )

                    for sr in stats:
                        sr.update({
                            "sid": sid,
                            "head_K": K,
                        })
                        message_rows.append(sr)

                    for alpha in head_scales:
                        pred, text, Jp = generation_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            target_layers=target_layers,
                            writers_for_relation=writers_for_relation,
                            max_new_tokens=a.max_new_tokens,
                            attention_patch_map=patch_map,
                            scale=alpha,
                        )
                        ok = pred == m["gt"]

                        eval_rows.append({
                            "sid": sid,
                            "gt": m["gt"],
                            "condition": "directJ_heads",
                            "head_K": K,
                            "alpha": float(alpha),
                            "prediction": pred,
                            "correct": ok,
                            "text": text,
                            "J": Jp,
                            "delta_J": Jp - baseline_J,
                            "wrong_to_correct": (not base_ok) and ok,
                            "correct_to_wrong": base_ok and (not ok),
                        })

            except Exception as exc:
                append_jsonl(error_path, {
                    "phase": "evaluation",
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-25:],
                })
                tqdm.write(
                    f"[EVAL ERROR] sid={sid}: {type(exc).__name__}: {exc}"
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

        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(
            outdir / "eval_per_sample.csv",
            index=False,
        )

        pd.DataFrame(message_rows).to_csv(
            outdir / "head_message_stats.csv",
            index=False,
        )

        eval_summary = summarize_eval(eval_df)
        eval_summary.to_csv(
            outdir / "eval_summary.csv",
            index=False,
        )

        J_summary = summarize_J(eval_df)
        J_summary.to_csv(
            outdir / "eval_J_summary.csv",
            index=False,
        )

        # Relation summary
        rels = []
        for rel, g in eval_df.groupby("gt"):
            z = summarize_eval(g)
            if len(z):
                z.insert(0, "gt", rel)
                rels.append(z)
        (
            pd.concat(rels, ignore_index=True)
            if rels else pd.DataFrame()
        ).to_csv(
            outdir / "eval_by_relation.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------------
        report = []
        report += [
            "=" * 190,
            "DIRECT J -> HEADS -> LATE J / GENERATION",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"head layers={head_layers}",
            f"target writer layers={target_layers}",
            f"query scope={a.query_scope} exclude_last={a.exclude_last_query}",
            f"writer calibration train N={len(train)}",
            f"head scan train N={len(scan_meta)}",
            f"eval scope={a.eval_scope} N={len(eval_meta)}",
            "",
            "DIRECT-J TOP HEADS",
            "-" * 190,
            head_summary.head(30).to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            ),
            "",
            "SPATIAL OVERLAP",
            "-" * 190,
            overlap_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            ),
            f"Spearman(spatial accuracy, direct-J mean signed score) "
            f"on matched heads = {spatial_spearman:.5f}",
            "",
            "GENERATION",
            "-" * 190,
            eval_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ) if len(eval_summary) else "EMPTY",
            "",
            "LATE-J REPRODUCTION",
            "-" * 190,
            J_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            ) if len(J_summary) else "EMPTY",
            "",
            "Interpretation:",
            "  directJ head score = sum_q (z_real-z_gray)^T grad_z J_writer.",
            "  The head intervention amplifies the SAME Real-Gray object with one shared alpha.",
            "  causal_rg is the direct Top-K causal-token Real-Gray reference.",
            "  mean_deltaJ_reproduction_ratio ~1 means the head group reproduces the average late-J increase of causal_rg.",
            "  Generation accuracy / W2C / C2W decide whether matching J also reproduces behavior.",
        ]

        report_text = "\n".join(report) + "\n"
        print(report_text)

        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "scan_directJ_heads_and_reproduce_causalRG_v2.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "head_layers": head_layers,
                "target_layers": target_layers,
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "query_scope": a.query_scope,
                "exclude_last_query": a.exclude_last_query,
                "writer_mode": a.writer_mode,
                "train_ratio": a.train_ratio,
                "scan_N": len(scan_meta),
                "eval_scope": a.eval_scope,
                "eval_N": len(eval_meta),
                "head_ks": head_ks,
                "head_scales": head_scales,
                "causal_scale": a.causal_scale,
                "spatial_head_csv": a.spatial_head_csv or None,
                "spatial_top_n": a.spatial_top_n,
                "spatial_spearman": spatial_spearman,
                "directJ_definition": (
                    "sum over selected query positions of "
                    "(z_real-z_gray)^T dJ_writer/dz_real, PRE-W_O"
                ),
                "head_intervention": (
                    "all selected heads use same alpha; add W_O^h(z_real-z_gray) "
                    "at every selected query position during real-image prefill"
                ),
                "causal_reference": (
                    "directly add causal token h_real-h_gray at top ranked causal states"
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
