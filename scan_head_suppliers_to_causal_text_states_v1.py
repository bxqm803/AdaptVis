#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_head_suppliers_to_causal_text_states_v1.py

Purpose
=======
Start from already identified behaviorally causal TEXT states and trace backward
to discover which upstream attention heads actually supply them.

This intentionally does NOT assume that Direction Heads are the causal pathway.

For each sample, selected causal state (C,p) comes from the existing writer-guided
ranking:

    M_causal(C,p)
      = (h_real[C,p] - h_gray[C,p])^T
        d J_writer / d h_real[C,p]

We then define TWO causal-state objectives.

(1) Decision-useful causal-state component
------------------------------------------
First pull the late writer gradient back to each selected causal state:

    g_C,p = d J_writer / d h[C,p]

Normalize it:

    u_useful(C,p) = g_C,p / ||g_C,p||

Then:

    J_useful
      = sum_(C,p selected) < h_real[C,p], u_useful(C,p) >

Question:
    Which upstream heads naturally contribute to the part of each causal state
    that the late decision writer actually cares about?

(2) Full Real-Gray causal-state component
-----------------------------------------
For each selected causal state:

    delta_C,p = h_real[C,p] - h_gray[C,p]
    u_RG(C,p) = delta_C,p / ||delta_C,p||

Then:

    J_RG
      = sum_(C,p selected) < h_real[C,p], u_RG(C,p) >

Question:
    Which upstream heads naturally contribute to the causal state's complete
    image-induced Real-Gray displacement?

Head supplier score
===================
For every upstream attention head (L,h), capture its PRE-W_O output z[L,h,q]
at every query token q.  Using the same Real-Gray mediation logic:

    delta_z[L,h,q]
      = z_real[L,h,q] - z_gray[L,h,q]

    G_useful[L,h,q]
      = d J_useful / d z_real[L,h,q]

    G_RG[L,h,q]
      = d J_RG / d z_real[L,h,q]

Then:

    M_useful[L,h,q] = delta_z^T G_useful
    M_RG[L,h,q]     = delta_z^T G_RG

Aggregate over query positions q for each head:
    signed_sum
    positive_sum
    negative_sum
    absolute_sum
    max_position_score

We also aggregate the mediation by QUERY token class:
    visual
    subject
    reference
    relation_words
    other_text
    last

Interpretation
==============
A head can be highly spatially decodable yet have low M_useful and low M_RG.
That would support:

    decodable spatial representation != naturally used supplier pathway

Conversely, if Direction Heads rank highly in M_useful, they are not merely
readable: their natural Real-Gray change contributes to constructing the
decision-relevant causal text state.

Important
=========
* PRE-W_O head output is used so each query head is separable.
* The score is sample-specific Real-Gray x gradient mediation.
* This is a first-order attribution / mediation diagnostic, not by itself a
  causal ablation.  Top supplier heads should subsequently be ablated/patched.
* The causal-state ranking and late writer use GT relation, so this remains an
  ORACLE MECHANISM diagnostic.
* Head layer L can only supply a causal block-output state at C if L <= C.
  The autograd graph enforces this automatically.

Default Direction-head markers
==============================
These are ONLY marked after the supplier scan.  They are the current Top-20
Synthetic400 Real-Gray directions transferred to COCO, ordered by COCO accuracy:

    L26H03 L23H01 L23H05 L26H02 L22H09
    L23H00 L22H13 L22H02 L21H14 L23H10
    L21H05 L22H12 L21H01 L26H01 L27H02
    L27H01 L21H11 L22H14 L21H03 L22H10

The scan itself uses ALL heads in --supplier-layers.

Recommended N=80 run
====================
CUDA_VISIBLE_DEVICES=0 python -u scan_head_suppliers_to_causal_text_states_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --supplier-layers 1-26 \
  --target-layers 32,34,35 \
  --eval-scope all_data \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_head_suppliers_to_causal_text_n80_v1 \
  --overwrite

Then full 440:
    --eval-max-samples 0

Outputs
=======
selected_causal_text_states.csv
writer_geometry.csv
per_sample_head_supplier_scores.csv
head_supplier_summary.csv
head_supplier_correct_vs_wrong.csv
direction_head_supplier_ranks.csv
direction_vs_other_summary.csv
causal_state_geometry.csv
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

DEFAULT_DIRECTION_HEADS = ",".join([
    "26:3", "23:1", "23:5", "26:2", "22:9",
    "23:0", "22:13", "22:2", "21:14", "23:10",
    "21:5", "22:12", "21:1", "26:1", "27:2",
    "27:1", "21:11", "22:14", "21:3", "22:10",
])

# Current Syn400 -> COCO accuracy, used only as optional display metadata.
DEFAULT_DIRECTION_ACCURACY = {
    (26,3): 0.815909,
    (23,1): 0.793182,
    (23,5): 0.786364,
    (26,2): 0.779545,
    (22,9): 0.775000,
    (23,0): 0.770455,
    (22,13): 0.770455,
    (22,2): 0.759091,
    (21,14): 0.752273,
    (23,10): 0.752273,
    (21,5): 0.747727,
    (22,12): 0.747727,
    (21,1): 0.747727,
    (26,1): 0.743182,
    (27,2): 0.734091,
    (27,1): 0.729545,
    (21,11): 0.725000,
    (22,14): 0.725000,
    (21,3): 0.722727,
    (22,10): 0.713636,
}


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
    p.add_argument("--ranked-causal", required=True)

    p.add_argument(
        "--causal-layers",
        default="20-26",
        help="Eligible causal block-output layers.",
    )
    p.add_argument(
        "--causal-top-k",
        type=int,
        default=7,
        help="Top-K TEXT causal states per sample after filtering.",
    )
    p.add_argument(
        "--causal-categories",
        default="",
        help=(
            "Optional broad categories, e.g. reference,subject,relation_words. "
            "Empty = all non-visual text causal states."
        ),
    )

    p.add_argument(
        "--supplier-layers",
        default="1-26",
        help="Attention head layers to scan.",
    )
    p.add_argument(
        "--target-layers",
        default="32,34,35",
        help="Late writer layers, matching the original Qwen3B causal ranking.",
    )
    p.add_argument(
        "--direction-heads",
        default=DEFAULT_DIRECTION_HEADS,
        help="Heads marked post-hoc as known Direction heads.",
    )

    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-scope",
        default="all_data",
        choices=["test", "all_data"],
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all samples in eval scope.",
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)

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
# Generic
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


def hname(L: int, h: int) -> str:
    return f"L{int(L):02d}H{int(h):02d}"


def normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# =============================================================================
# Data
# =============================================================================

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
    train, heldout = traj.stratified_split(meta, args.train_ratio, args.seed)

    if args.eval_scope == "all_data":
        test = list(meta)
    else:
        test = list(heldout)

    if args.eval_max_samples > 0:
        test = traj.stratified_cap(test, args.eval_max_samples, args.seed + 1)

    return two, meta, train, test, rec_by_sid


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
# Model
# =============================================================================

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
# Gray capture: block outputs + PRE-W_O head outputs
# =============================================================================

class GrayStateHeadCapture:
    def __init__(
        self,
        decoder_layers,
        state_layers: Sequence[int],
        head_layers: Sequence[int],
    ):
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
    cap = GrayStateHeadCapture(
        decoder_layers,
        state_layers,
        head_layers,
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)

        ms = [L for L in state_layers if int(L) not in cap.states]
        mh = [L for L in head_layers if int(L) not in cap.pre_o]
        if ms or mh:
            raise RuntimeError(
                f"Gray capture missing states={ms}, head_layers={mh}"
            )
        return cap.states, cap.pre_o
    finally:
        cap.close()


# =============================================================================
# Real differentiable graph capture
# =============================================================================

class RealGraphHeadCapture:
    """
    Cut at decoder block 0 OUTPUT to create a grad-enabled leaf, preserving all
    supplier-head graphs from layer 1 onward.

    Stores:
      states[L] : real block output tensor
      pre_o[L]  : real PRE-W_O tensor [B,S,H*Dh]
    """

    def __init__(
        self,
        decoder_layers,
        state_layers: Sequence[int],
        head_layers: Sequence[int],
        cut_layer: int = 0,
    ):
        self.states = {}
        self.pre_o = {}
        self.handles = []
        self.cut_layer = int(cut_layer)

        # Cut first, so all L>=1 computations require grad even though model
        # parameters are frozen.
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
                    f"Supplier head L{L} is not after graph cut L{self.cut_layer}. "
                    "Use supplier layers >= 1."
                )

            attn = spatialscan.resolve_attn(decoder_layers[L])
            op = spatialscan.resolve_o_proj(attn)

            def make_pre_hook(layer):
                def hook(_m, inputs):
                    # Do NOT detach: this is the differentiable head output.
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

    ms = [L for L in state_layers if int(L) not in cap.states]
    mh = [L for L in head_layers if int(L) not in cap.pre_o]
    if ms or mh:
        cap.close()
        raise RuntimeError(
            f"Real graph missing states={ms}, head_layers={mh}"
        )
    return cap


# =============================================================================
# Late writer calibration
# =============================================================================

def calibrate_writers(
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
        real = gray = rb = gb = None
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

            hr = dyn.capture_cpu(
                model, decoder_layers, rb, target_layers
            )
            hg = dyn.capture_cpu(
                model, decoder_layers, gb, target_layers
            )

            q_by_sid[sid] = {
                T: (hr[T][0, -1] - hg[T][0, -1]).astype(np.float32)
                for T in target_layers
            }

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "writer_calibration",
                "sid": sid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-15:],
            })
            raise
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

    writers, geometry = dyn.learn_writers(
        train, q_by_sid, target_layers, writer_mode
    )
    return writers, geometry


# =============================================================================
# Causal objective construction
# =============================================================================

def build_writer_objective(
    cap,
    target_layers: Sequence[int],
    writers_for_relation: Mapping[int, np.ndarray],
):
    terms = []
    for T in target_layers:
        shat = torch.as_tensor(
            normalize_np(writers_for_relation[T]),
            device=cap.states[T].device,
            dtype=torch.float32,
        )
        terms.append(
            torch.dot(cap.states[T][0, -1].float(), shat)
        )
    return torch.stack(terms).sum()


def state_weight(row) -> float:
    # Equal state weight after direction normalization.  This prevents a single
    # huge-norm state from dominating the supplier scan.
    return 1.0


def causal_objectives(
    *,
    cap,
    gray_states: Mapping[int, np.ndarray],
    causal_rows: pd.DataFrame,
    target_layers: Sequence[int],
    writers_for_relation: Mapping[int, np.ndarray],
):
    """
    Returns:
      J_writer
      J_useful
      J_RG
      state diagnostics

    We first backprop J_writer to entire causal layer outputs, then use each
    selected (C,p) writer-gradient direction as a detached direction for J_useful.
    """
    J_writer = build_writer_objective(
        cap,
        target_layers,
        writers_for_relation,
    )

    causal_layers = sorted(
        set(map(int, causal_rows["source_layer"].tolist()))
    )
    causal_tensors = [cap.states[C] for C in causal_layers]

    writer_grads = torch.autograd.grad(
        J_writer,
        causal_tensors,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    grad_by_layer = {
        C: g for C, g in zip(causal_layers, writer_grads)
    }

    useful_terms = []
    rg_terms = []
    diag = []

    for r in causal_rows.itertuples():
        C = int(r.source_layer)
        p = int(r.position)
        hr = cap.states[C]
        hg_np = gray_states[C]

        npos = min(int(hr.shape[1]), int(hg_np.shape[1]))
        if not (0 <= p < npos):
            raise RuntimeError(
                f"Selected causal L{C} p{p} outside aligned length {npos}"
            )

        hvec = hr[0, p].float()
        gray_vec = torch.as_tensor(
            hg_np[0, p],
            device=hvec.device,
            dtype=torch.float32,
        )

        delta = (hvec.detach() - gray_vec).float()
        gvec = grad_by_layer[C][0, p].detach().float()

        dn = float(delta.norm().item())
        gn = float(gvec.norm().item())

        u_rg = delta / max(dn, EPS)
        u_use = gvec / max(gn, EPS)

        w = float(state_weight(r))
        rg_terms.append(w * torch.dot(hvec, u_rg))
        useful_terms.append(w * torch.dot(hvec, u_use))

        med_recomputed = float(torch.dot(delta, gvec).item())

        diag.append({
            "sid": int(r.sid),
            "gt": str(r.gt) if hasattr(r, "gt") else "",
            "causal_text_rank": int(r.causal_text_rank),
            "global_rank": int(r.rank),
            "causal_layer": C,
            "position": p,
            "token": str(r.token),
            "category": str(r.category),
            "broad_category": str(r.broad_category),
            "ranking_mediation": float(r.mediation),
            "recomputed_mediation": med_recomputed,
            "delta_h_norm": dn,
            "writer_grad_norm": gn,
            "delta_vs_writer_grad_cos": (
                med_recomputed / (dn * gn)
                if dn > EPS and gn > EPS
                else np.nan
            ),
        })

    if not useful_terms or not rg_terms:
        raise RuntimeError("No valid selected causal states.")

    J_useful = torch.stack(useful_terms).sum()
    J_rg = torch.stack(rg_terms).sum()

    return J_writer, J_useful, J_rg, diag


# =============================================================================
# Head supplier scoring
# =============================================================================

def infer_head_geometry(model, decoder_layers, head_layers):
    cfg = spatialscan.get_text_config(model)
    n_heads_default = int(cfg.num_attention_heads)

    out = {}
    for L in head_layers:
        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)

        n_heads = getattr(attn, "num_heads", None)
        if n_heads is None:
            n_heads = getattr(
                getattr(attn, "config", None),
                "num_attention_heads",
                None,
            )
        if n_heads is None:
            n_heads = n_heads_default
        n_heads = int(n_heads)

        hidden = int(op.in_features)
        if hidden % n_heads != 0:
            raise RuntimeError(
                f"L{L}: o_proj input={hidden} not divisible by heads={n_heads}"
            )
        out[L] = {
            "n_heads": n_heads,
            "head_dim": hidden // n_heads,
            "hidden": hidden,
        }
    return out


def score_heads_for_sample(
    *,
    sid: int,
    gt: str,
    baseline_correct: bool,
    cap,
    gray_pre_o: Mapping[int, np.ndarray],
    supplier_layers: Sequence[int],
    head_geometry: Mapping[int, Mapping[str, int]],
    cats: Sequence[str],
    toks: Sequence[str],
    J_useful,
    J_rg,
    direction_set: set,
):
    preo_tensors = [cap.pre_o[L] for L in supplier_layers]

    grads_use = torch.autograd.grad(
        J_useful,
        preo_tensors,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grads_rg = torch.autograd.grad(
        J_rg,
        preo_tensors,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    rows = []

    for L, gu, gr in zip(supplier_layers, grads_use, grads_rg):
        if gu is None and gr is None:
            continue

        zr_t = cap.pre_o[L]
        zr = zr_t.detach().float().cpu().numpy().astype(np.float32)
        zg = np.asarray(gray_pre_o[L], dtype=np.float32)

        n = min(
            zr.shape[1],
            zg.shape[1],
            len(cats),
            len(toks),
        )

        H = int(head_geometry[L]["n_heads"])
        D = int(head_geometry[L]["head_dim"])

        dz = (zr[0, :n] - zg[0, :n]).reshape(n, H, D)

        if gu is None:
            Gu = np.zeros_like(dz)
        else:
            Gu = (
                gu[0, :n].detach().float().cpu().numpy()
                .astype(np.float32).reshape(n, H, D)
            )

        if gr is None:
            Gr = np.zeros_like(dz)
        else:
            Gr = (
                gr[0, :n].detach().float().cpu().numpy()
                .astype(np.float32).reshape(n, H, D)
            )

        Mu = np.einsum("phd,phd->ph", dz, Gu).astype(np.float32)
        Mr = np.einsum("phd,phd->ph", dz, Gr).astype(np.float32)

        broad = [
            dyn.broad_category(str(cats[p]))
            for p in range(n)
        ]

        broad_names = (
            "visual",
            "subject",
            "reference",
            "relation_words",
            "other_text",
            "last",
        )

        for h in range(H):
            mu = Mu[:, h]
            mr = Mr[:, h]

            row = {
                "sid": sid,
                "gt": gt,
                "baseline_correct": bool(baseline_correct),
                "layer": L,
                "head": h,
                "head_name": hname(L, h),
                "is_direction_head": (L, h) in direction_set,
                "direction_coco_accuracy": DEFAULT_DIRECTION_ACCURACY.get(
                    (L, h), np.nan
                ),

                "useful_signed_sum": float(mu.sum()),
                "useful_positive_sum": float(np.maximum(mu, 0).sum()),
                "useful_negative_sum": float(np.minimum(mu, 0).sum()),
                "useful_abs_sum": float(np.abs(mu).sum()),
                "useful_max_position": float(mu.max()) if len(mu) else np.nan,
                "useful_min_position": float(mu.min()) if len(mu) else np.nan,

                "rg_signed_sum": float(mr.sum()),
                "rg_positive_sum": float(np.maximum(mr, 0).sum()),
                "rg_negative_sum": float(np.minimum(mr, 0).sum()),
                "rg_abs_sum": float(np.abs(mr).sum()),
                "rg_max_position": float(mr.max()) if len(mr) else np.nan,
                "rg_min_position": float(mr.min()) if len(mr) else np.nan,

                "delta_z_norm": float(np.linalg.norm(dz[:, h, :])),
                "useful_grad_norm": float(np.linalg.norm(Gu[:, h, :])),
                "rg_grad_norm": float(np.linalg.norm(Gr[:, h, :])),
            }

            for bc in broad_names:
                idx = np.asarray(
                    [i for i, x in enumerate(broad) if x == bc],
                    dtype=np.int64,
                )
                if len(idx):
                    row[f"useful_{bc}_signed"] = float(mu[idx].sum())
                    row[f"useful_{bc}_positive"] = float(
                        np.maximum(mu[idx], 0).sum()
                    )
                    row[f"rg_{bc}_signed"] = float(mr[idx].sum())
                    row[f"rg_{bc}_positive"] = float(
                        np.maximum(mr[idx], 0).sum()
                    )
                else:
                    row[f"useful_{bc}_signed"] = 0.0
                    row[f"useful_{bc}_positive"] = 0.0
                    row[f"rg_{bc}_signed"] = 0.0
                    row[f"rg_{bc}_positive"] = 0.0

            # Which query position is strongest for useful mediation?
            if len(mu):
                j = int(np.argmax(mu))
                row["useful_best_query_position"] = j
                row["useful_best_query_token"] = str(toks[j]).replace(
                    "\n", "\\n"
                )
                row["useful_best_query_category"] = str(cats[j])
                row["useful_best_query_broad_category"] = broad[j]
            else:
                row["useful_best_query_position"] = -1
                row["useful_best_query_token"] = ""
                row["useful_best_query_category"] = ""
                row["useful_best_query_broad_category"] = ""

            rows.append(row)

    return rows


# =============================================================================
# Summaries
# =============================================================================

def build_head_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    numeric_means = [
        "useful_signed_sum",
        "useful_positive_sum",
        "useful_negative_sum",
        "useful_abs_sum",
        "useful_max_position",
        "rg_signed_sum",
        "rg_positive_sum",
        "rg_negative_sum",
        "rg_abs_sum",
        "rg_max_position",
        "delta_z_norm",
        "useful_grad_norm",
        "rg_grad_norm",
        "useful_visual_signed",
        "useful_subject_signed",
        "useful_reference_signed",
        "useful_relation_words_signed",
        "useful_other_text_signed",
        "rg_visual_signed",
        "rg_subject_signed",
        "rg_reference_signed",
        "rg_relation_words_signed",
        "rg_other_text_signed",
    ]
    numeric_means = [c for c in numeric_means if c in df.columns]

    rows = []
    for (L, h, hn), g in df.groupby(["layer", "head", "head_name"]):
        row = {
            "layer": int(L),
            "head": int(h),
            "head_name": hn,
            "N_samples": int(g["sid"].nunique()),
            "is_direction_head": bool(g["is_direction_head"].iloc[0]),
            "direction_coco_accuracy": safe_mean(
                g["direction_coco_accuracy"]
            ),
            "useful_positive_sample_fraction": float(
                (g["useful_signed_sum"] > 0).mean()
            ),
            "rg_positive_sample_fraction": float(
                (g["rg_signed_sum"] > 0).mean()
            ),
        }

        for c in numeric_means:
            row[f"mean_{c}"] = safe_mean(g[c])
            row[f"median_{c}"] = safe_median(g[c])

        rows.append(row)

    out = pd.DataFrame(rows)

    out["useful_supplier_rank"] = (
        out["mean_useful_signed_sum"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["useful_positive_rank"] = (
        out["mean_useful_positive_sum"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["rg_supplier_rank"] = (
        out["mean_rg_signed_sum"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["rg_positive_rank"] = (
        out["mean_rg_positive_sum"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return out.sort_values(
        ["useful_supplier_rank", "rg_supplier_rank"]
    ).reset_index(drop=True)


def correct_wrong_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    rows = []
    for (hn, ok), g in df.groupby(["head_name", "baseline_correct"]):
        rows.append({
            "head_name": hn,
            "baseline_correct": bool(ok),
            "N": len(g),
            "mean_useful_signed_sum": safe_mean(g["useful_signed_sum"]),
            "mean_useful_positive_sum": safe_mean(g["useful_positive_sum"]),
            "mean_rg_signed_sum": safe_mean(g["rg_signed_sum"]),
            "mean_rg_positive_sum": safe_mean(g["rg_positive_sum"]),
        })

    out = pd.DataFrame(rows)

    piv = []
    for hn, g in out.groupby("head_name"):
        rr = {"head_name": hn}
        for r in g.itertuples():
            suffix = "correct" if bool(r.baseline_correct) else "wrong"
            rr[f"N_{suffix}"] = int(r.N)
            rr[f"useful_signed_{suffix}"] = float(
                r.mean_useful_signed_sum
            )
            rr[f"useful_positive_{suffix}"] = float(
                r.mean_useful_positive_sum
            )
            rr[f"rg_signed_{suffix}"] = float(r.mean_rg_signed_sum)
            rr[f"rg_positive_{suffix}"] = float(r.mean_rg_positive_sum)

        if (
            "useful_signed_correct" in rr
            and "useful_signed_wrong" in rr
        ):
            rr["useful_signed_correct_minus_wrong"] = (
                rr["useful_signed_correct"]
                - rr["useful_signed_wrong"]
            )
        if (
            "rg_signed_correct" in rr
            and "rg_signed_wrong" in rr
        ):
            rr["rg_signed_correct_minus_wrong"] = (
                rr["rg_signed_correct"]
                - rr["rg_signed_wrong"]
            )
        piv.append(rr)

    return pd.DataFrame(piv)


def direction_vs_other_summary(head_summary: pd.DataFrame) -> pd.DataFrame:
    if len(head_summary) == 0:
        return pd.DataFrame()

    rows = []
    for flag, g in head_summary.groupby("is_direction_head"):
        rows.append({
            "group": "direction_heads" if flag else "other_heads",
            "N_heads": len(g),
            "mean_useful_supplier_rank": safe_mean(
                g["useful_supplier_rank"]
            ),
            "median_useful_supplier_rank": safe_median(
                g["useful_supplier_rank"]
            ),
            "mean_rg_supplier_rank": safe_mean(
                g["rg_supplier_rank"]
            ),
            "median_rg_supplier_rank": safe_median(
                g["rg_supplier_rank"]
            ),
            "mean_useful_signed": safe_mean(
                g["mean_useful_signed_sum"]
            ),
            "mean_useful_positive": safe_mean(
                g["mean_useful_positive_sum"]
            ),
            "mean_rg_signed": safe_mean(
                g["mean_rg_signed_sum"]
            ),
            "mean_rg_positive": safe_mean(
                g["mean_rg_positive_sum"]
            ),
        })
    return pd.DataFrame(rows)


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
    target_layers = parse_layers(a.target_layers)
    direction_heads = parse_heads(a.direction_heads)
    direction_set = set(direction_heads)
    categories = parse_categories(a.causal_categories)

    if any(L <= 0 for L in supplier_layers):
        raise ValueError(
            "This implementation creates the differentiable graph at block-0 "
            "output, so --supplier-layers must start at L1 or later."
        )

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    two, meta, train, test, rec_by_sid = load_coco_meta(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    ranking_sids = set(
        pd.to_numeric(
            pd.read_csv(a.ranked_causal, usecols=["sid"])["sid"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )
    test = [m for m in test if int(m["sid"]) in ranking_sids]
    eval_sids = {int(m["sid"]) for m in test}

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

        for L in sorted(
            set(causal_layers + supplier_layers + target_layers + [0])
        ):
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"L{L} invalid; model has 0..{n_layers-1}"
                )

        max_causal = max(causal_layers)
        supplier_layers = [
            L for L in supplier_layers
            if L <= max_causal
        ]

        head_geometry = infer_head_geometry(
            model, decoder_layers, supplier_layers
        )

        print("=" * 180)
        print("UPSTREAM HEAD SUPPLIERS -> CAUSAL TEXT STATES")
        print("=" * 180)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(
            f"writer calibration N={len(train)} | "
            f"eval scope={a.eval_scope} N={len(test)}"
        )
        print(
            f"causal layers={causal_layers} | topK text={a.causal_top_k} | "
            f"selected states={len(selected)}"
        )
        print(f"supplier layers={supplier_layers}")
        print(f"late writer targets={target_layers}")
        print(
            "Direction heads are MARKERS ONLY:",
            ",".join(hname(L,h) for L,h in direction_heads),
        )
        print()

        writers, writer_geometry = calibrate_writers(
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

        all_head_rows = []
        all_state_rows = []

        for m in tqdm(test, desc="TRACE head suppliers"):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            real = gray = rb = gb = cap = None
            try:
                causal_rows = selected_by_sid[sid]
                gt = m["gt"]

                # Per-sample only scan layers that can precede at least one selected
                # causal state.  The objective itself removes impossible later paths.
                max_C = int(causal_rows["source_layer"].max())
                supplier_sid = [L for L in supplier_layers if L <= max_C]
                if not supplier_sid:
                    continue

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
                        "Real and Gray input_ids differ; token alignment invalid."
                    )

                cats, toks = dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids_r,
                    m["subject"],
                    m["reference"],
                )

                # Actual generation is used ONLY to split supplier coupling by
                # baseline-correct vs baseline-wrong.
                clean = dyn.run_generation_with_hooks(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    target_layers=target_layers,
                    writers_for_relation={
                        T: writers[T][gt] for T in target_layers
                    },
                    max_new_tokens=a.max_new_tokens,
                )
                baseline_correct = clean["prediction"] == gt

                state_layers_sid = sorted(
                    set(
                        causal_rows["source_layer"].astype(int).tolist()
                        + target_layers
                    )
                )

                gray_states, gray_pre_o = run_gray_capture(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=gb,
                    state_layers=state_layers_sid,
                    head_layers=supplier_sid,
                )

                with torch.enable_grad():
                    cap = run_real_graph(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        state_layers=state_layers_sid,
                        head_layers=supplier_sid,
                    )

                    J_writer, J_useful, J_rg, state_diag = causal_objectives(
                        cap=cap,
                        gray_states=gray_states,
                        causal_rows=causal_rows,
                        target_layers=target_layers,
                        writers_for_relation={
                            T: writers[T][gt] for T in target_layers
                        },
                    )

                    for row in state_diag:
                        row["baseline_prediction"] = clean["prediction"]
                        row["baseline_correct"] = baseline_correct
                        all_state_rows.append(row)

                    head_rows = score_heads_for_sample(
                        sid=sid,
                        gt=gt,
                        baseline_correct=baseline_correct,
                        cap=cap,
                        gray_pre_o=gray_pre_o,
                        supplier_layers=supplier_sid,
                        head_geometry=head_geometry,
                        cats=cats,
                        toks=toks,
                        J_useful=J_useful,
                        J_rg=J_rg,
                        direction_set=direction_set,
                    )
                    all_head_rows.extend(head_rows)

                cap.close()
                cap = None

            except Exception as exc:
                append_jsonl(error_path, {
                    "phase": "supplier_scan",
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

        head_df = pd.DataFrame(all_head_rows)
        state_df = pd.DataFrame(all_state_rows)

        head_df.to_csv(
            outdir / "per_sample_head_supplier_scores.csv",
            index=False,
        )
        state_df.to_csv(
            outdir / "causal_state_geometry.csv",
            index=False,
        )

        summary = build_head_summary(head_df)
        summary.to_csv(
            outdir / "head_supplier_summary.csv",
            index=False,
        )

        cw = correct_wrong_summary(head_df)
        cw.to_csv(
            outdir / "head_supplier_correct_vs_wrong.csv",
            index=False,
        )

        dheads = summary[
            summary["is_direction_head"].astype(bool)
        ].copy()
        dheads = dheads.sort_values("useful_supplier_rank")
        dheads.to_csv(
            outdir / "direction_head_supplier_ranks.csv",
            index=False,
        )

        group_summary = direction_vs_other_summary(summary)
        group_summary.to_csv(
            outdir / "direction_vs_other_summary.csv",
            index=False,
        )

        # Rank overlap diagnostics.
        topks = [5, 10, 20, 30, 50]
        overlap_rows = []
        for K in topks:
            top = summary.nsmallest(K, "useful_supplier_rank")
            n_dir = int(top["is_direction_head"].astype(bool).sum())
            overlap_rows.append({
                "supplier_top_k": K,
                "direction_heads_in_topk": n_dir,
                "direction_fraction": n_dir / K,
            })
        overlap_df = pd.DataFrame(overlap_rows)
        overlap_df.to_csv(
            outdir / "direction_supplier_topk_overlap.csv",
            index=False,
        )

        # Optional Spearman between known direction accuracy and supplier score.
        dir_with_acc = dheads[
            np.isfinite(dheads["direction_coco_accuracy"])
        ].copy()
        spearman = np.nan
        if len(dir_with_acc) >= 3:
            spearman = float(
                dir_with_acc[
                    ["direction_coco_accuracy", "mean_useful_signed_sum"]
                ].corr(method="spearman").iloc[0, 1]
            )

        report = []
        report.append("=" * 180)
        report.append("HEAD SUPPLIERS -> CAUSAL TEXT STATES")
        report.append("=" * 180)
        report.append(
            f"completed samples={head_df['sid'].nunique() if len(head_df) else 0} "
            f"| scored head-sample rows={len(head_df)}"
        )
        report.append("")

        report.append("TOP 30: DECISION-USEFUL CAUSAL-STATE SUPPLIERS")
        report.append("-" * 180)
        if len(summary):
            cols = [
                "useful_supplier_rank",
                "head_name",
                "is_direction_head",
                "direction_coco_accuracy",
                "mean_useful_signed_sum",
                "mean_useful_positive_sum",
                "useful_positive_sample_fraction",
                "rg_supplier_rank",
                "mean_rg_signed_sum",
                "mean_delta_z_norm",
                "mean_useful_grad_norm",
            ]
            report.append(
                summary[cols].head(30).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("KNOWN DIRECTION HEADS: WHERE DO THEY RANK AS SUPPLIERS?")
        report.append("-" * 180)
        if len(dheads):
            cols = [
                "head_name",
                "direction_coco_accuracy",
                "useful_supplier_rank",
                "mean_useful_signed_sum",
                "useful_positive_sample_fraction",
                "rg_supplier_rank",
                "mean_rg_signed_sum",
            ]
            report.append(
                dheads[cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        else:
            report.append("No marked Direction head occurred in scanned layers.")
        report.append("")

        report.append("DIRECTION-HEAD OVERLAP WITH TOP SUPPLIERS")
        report.append("-" * 180)
        report.append(overlap_df.to_string(index=False))
        report.append("")

        report.append("DIRECTION vs OTHER HEADS")
        report.append("-" * 180)
        if len(group_summary):
            report.append(
                group_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append(
            "Spearman(direction COCO decoding accuracy, "
            f"decision-useful supplier score) = {spearman:.5f}"
        )
        report.append("")

        report.append("HOW TO READ THIS")
        report.append("-" * 180)
        report.append(
            "If Direction heads decode well but have poor useful_supplier_rank, "
            "their spatial information is readable but is not a major natural "
            "supplier of the behaviorally causal text states."
        )
        report.append(
            "If different heads dominate useful_supplier_rank, those are the "
            "first candidates for causal ablation/path-patching."
        )
        report.append(
            "If a head ranks high for RG but low for useful, it helps construct "
            "the causal token's image-induced state, but not the component most "
            "aligned with the late decision writer."
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
                "script": "scan_head_suppliers_to_causal_text_states_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "ranked_causal": str(a.ranked_causal),
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "supplier_layers": supplier_layers,
                "target_layers": target_layers,
                "writer_mode": a.writer_mode,
                "direction_heads_are_posthoc_markers_only": [
                    hname(L,h) for L,h in direction_heads
                ],
                "useful_objective": (
                    "sum selected causal states dot normalized "
                    "late-writer gradient at that state"
                ),
                "rg_objective": (
                    "sum selected causal states dot normalized own Real-Gray delta"
                ),
                "supplier_score": (
                    "PRE-W_O head Real-Gray delta dot gradient of causal-state objective"
                ),
                "oracle_warning": (
                    "GT relation selects late writer and existing causal-state "
                    "ranking was writer-guided."
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
