#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_causal_rg_formation_v1.py

Purpose
=======
We already know that selected causal text states are behaviorally useful and that
amplifying their own sample-specific Image-Gray displacement can repair answers:

    delta_target(C,p) = h_real[C,p] - h_gray[C,p]

    h'[C,p] = h_real[C,p] + alpha * delta_target(C,p)

This script asks the upstream formation question directly:

    WHERE DOES delta_target(C,p) COME FROM?

For every selected causal state c=(C,p), and for every decoder block L<=C, capture
REAL and GRAY at the SAME token position p:

    x_L       : block input residual
    a_L       : self-attention output after W_O, before residual add
    m_L       : MLP output, before residual add
    h_L       : block output residual

For Qwen decoder blocks:

    h_L = x_L + a_L + m_L

so the Real-Gray displacement satisfies the exact within-block identity:

    Delta h_L = Delta x_L + Delta a_L + Delta m_L

(up to numerical precision).

We then use the FINAL selected causal-state Image-Gray vector as the target:

    t = Delta h_C[p]

and project every earlier component onto t:

    progress(v -> t) = <v,t> / ||t||^2

Thus for each L<=C:

    progress(Delta h_L)
      = progress(Delta x_L)
      + progress(Delta a_L)
      + progress(Delta m_L)

and at L=C:

    progress(Delta h_C) = 1

This tells us where the FINAL useful Image-Gray direction is already present,
where attention adds to it, where MLP adds to it, and where later blocks cancel
or rotate earlier contributions.

Exact head decomposition of attention RG
========================================
At token p, attention output after W_O decomposes exactly over query heads:

    Delta a_L[p]
      = sum_h W_O^{L,h} (
            z_real[L,h,p] - z_gray[L,h,p]
        )

where z is PRE-W_O head output.

For every head we therefore compute:

    msg_RG(L,h,p) = W_O^{L,h} Delta z(L,h,p)

    head_target_progress
      = <msg_RG, t> / ||t||^2

and validate:

    sum_h msg_RG ~= Delta a_L

This is NOT a gradient attribution. It is the actual observed Real-Gray
attention-message decomposition at that block/token.

Important interpretation
========================
- Within one block, Delta input + Delta attention + Delta MLP = Delta output is
  an exact algebraic decomposition.
- The head sum exactly decomposes that block's Delta attention.
- Comparing an earlier-layer component with the final target t describes the
  observed residual trajectory. It is not by itself a causal statement about
  what would survive all later nonlinear blocks.
- A later intervention/patching experiment can validate the strongest formation
  layers/heads causally.

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u analyze_causal_rg_formation_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --trace-layers 0-26 \
  --eval-max-samples 20 \
  --output-dir output/qwen3b_causal_rg_formation_n20_v1 \
  --overwrite

Then N=80:
    --eval-max-samples 80

Primary outputs
===============
formation_per_state_layer.csv
    Exact residual / attention / MLP Real-Gray decomposition for every
    causal-state trajectory.

formation_layer_summary.csv
    Aggregate trajectory statistics by target causal layer C and source block L.

formation_distance_summary.csv
    Aggregate by distance C-L.

head_rg_per_state_layer.csv
    Exact per-head decomposition of Delta attention at causal token p.

head_rg_summary.csv
    Global head aggregation.

top_heads_per_causal_state.csv
    Strongest positive heads for each concrete causal state.

sample_summary.csv
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

DIRECTION_TOP20 = {
    "L26H03","L23H01","L23H05","L26H02","L22H09",
    "L23H00","L22H13","L22H02","L21H14","L23H10",
    "L21H05","L22H12","L21H01","L26H01","L27H02",
    "L27H01","L21H11","L22H14","L21H03","L22H10",
}


# =============================================================================
# CLI / helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b","qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--ranked-causal", required=True)

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument(
        "--trace-layers",
        default="0-26",
        help=(
            "Decoder blocks whose input/attention/MLP/output RG is captured. "
            "Only trace L<=target causal layer C is analyzed."
        ),
    )

    p.add_argument(
        "--top-heads-per-state",
        type=int,
        default=10,
        help="Save this many positive head contributors per concrete causal state/layer.",
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=20)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager","sdpa","flash_attention_2","none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L","")
        if not part:
            continue
        if "-" in part:
            a,b = part.split("-",1)
            a,b = int(a),int(b)
            out.update(range(min(a,b),max(a,b)+1))
        else:
            out.add(int(part))
    return sorted(out)


def hname(L,h):
    return f"L{int(L):02d}H{int(h):02d}"


def safe_mean(xs):
    vals=[]
    for x in xs:
        try:
            v=float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs):
    vals=[]
    for x in xs:
        try:
            v=float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def append_jsonl(path,row):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(dict(row),ensure_ascii=False)+"\n")


def write_json(path,obj):
    Path(path).write_text(
        json.dumps(obj,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )


def cosine(a,b):
    a=np.asarray(a,dtype=np.float32)
    b=np.asarray(b,dtype=np.float32)
    an=float(np.linalg.norm(a))
    bn=float(np.linalg.norm(b))
    if an<=EPS or bn<=EPS:
        return float("nan")
    return float(np.dot(a,b)/(an*bn))


def target_progress(v,t):
    v=np.asarray(v,dtype=np.float32)
    t=np.asarray(t,dtype=np.float32)
    den=float(np.dot(t,t))
    if den<=EPS:
        return float("nan")
    return float(np.dot(v,t)/den)


def norm_ratio(v,t):
    vn=float(np.linalg.norm(v))
    tn=float(np.linalg.norm(t))
    if tn<=EPS:
        return float("nan")
    return vn/tn


# =============================================================================
# Data / model
# =============================================================================

def load_data(a):
    two=base.import_two_object_module()
    prompts=base.load_standard_prompts(Path(a.prompt_jsonl))
    records,_audit=two.load_records("coco_two",Path(a.data_root),None)
    rec_by_sid={int(r.sid):r for r in records}

    meta=[]
    for rec in records:
        sid=int(rec.sid)
        if sid not in prompts:
            continue
        p=prompts[sid]
        gt=traj.normalize_relation(base,p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid":sid,
            "gt":gt,
            "subject":str(p["subject"]),
            "reference":str(p["reference"]),
            "question_text":str(p["question_text"]),
        })

    meta=traj.stratified_cap(meta,a.max_samples,a.seed)
    return two,meta,rec_by_sid


def load_model(a,two):
    spec=base.merged_model_specs(two)[a.model]
    cls=getattr(transformers,spec.model_class)

    kw=dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"":a.device},
    )
    if a.attn_impl!="none":
        kw["attn_implementation"]=a.attn_impl

    print(f"[model] loading {spec.repo_id}",flush=True)
    try:
        model=cls.from_pretrained(spec.repo_id,**kw)
    except TypeError:
        kw["torch_dtype"]=kw.pop("dtype")
        model=cls.from_pretrained(spec.repo_id,**kw)

    model.eval()
    processor=AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model,processor)

    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers,decoder_path=base.resolve_decoder_layers(model)
    return model,processor,decoder_layers,decoder_path,spec


def load_causal_selection(path,allowed_sids,causal_layers,top_k):
    d=pd.read_csv(path)

    required={
        "sid","rank","source_layer","position",
        "token","category","broad_category",
    }
    miss=required-set(d.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns {sorted(miss)}")

    for c in ["sid","rank","source_layer","position"]:
        d[c]=pd.to_numeric(d[c],errors="raise").astype(int)

    d=d[d.sid.isin(allowed_sids)].copy()
    d=d[d.source_layer.isin(set(map(int,causal_layers)))].copy()
    d=d[d.broad_category.astype(str)!="visual"].copy()
    d=d[d.broad_category.astype(str)!="last"].copy()

    rows=[]
    for sid,g in d.groupby("sid"):
        z=g.sort_values("rank").head(int(top_k)).copy()
        z["causal_text_rank"]=np.arange(1,len(z)+1)
        rows.append(z)

    return pd.concat(rows,ignore_index=True) if rows else d.iloc[:0].copy()


# =============================================================================
# Module resolution
# =============================================================================

def resolve_mlp(layer):
    for name in ("mlp","feed_forward","ffn"):
        x=getattr(layer,name,None)
        if isinstance(x,torch.nn.Module):
            return x
    raise RuntimeError(
        f"Could not resolve MLP module on {type(layer).__name__}"
    )


def get_text_config(model):
    return spatialscan.get_text_config(model)


def infer_head_geometry(model,decoder_layers,layers):
    cfg=get_text_config(model)
    fallback_heads=int(cfg.num_attention_heads)
    out={}

    for L in layers:
        attn=spatialscan.resolve_attn(decoder_layers[L])
        op=spatialscan.resolve_o_proj(attn)

        H=getattr(attn,"num_heads",None)
        if H is None:
            H=getattr(getattr(attn,"config",None),"num_attention_heads",None)
        if H is None:
            H=fallback_heads
        H=int(H)

        width=int(op.in_features)
        if width%H!=0:
            raise RuntimeError(
                f"L{L}: o_proj.in_features={width} incompatible with heads={H}"
            )

        out[L]={
            "n_heads":H,
            "head_dim":width//H,
            "pre_o_width":width,
        }

    return out


def first_tensor_any(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x,(tuple,list)):
        for y in x:
            if torch.is_tensor(y):
                return y
    if isinstance(x,dict):
        for y in x.values():
            if torch.is_tensor(y):
                return y
    raise RuntimeError(f"No tensor found in {type(x).__name__}")


# =============================================================================
# Position-only capture
# =============================================================================

class FormationCapture:
    """
    Capture only requested token positions, keeping CPU memory small.

    For each traced block L:
      block_in[L]   [P,D]
      attn_out[L]   [P,D]   after W_O, before residual add
      mlp_out[L]    [P,D]   before residual add
      block_out[L]  [P,D]
      pre_o[L]      [P,H*Dh]
    """

    def __init__(self,decoder_layers,layers,positions):
        self.layers=list(map(int,layers))
        self.positions=sorted(set(map(int,positions)))
        self.block_in={}
        self.attn_out={}
        self.mlp_out={}
        self.block_out={}
        self.pre_o={}
        self.handles=[]

        def take_positions(x):
            if x.ndim!=3 or x.shape[0]!=1:
                raise RuntimeError(
                    f"Expected [1,S,D], got {tuple(x.shape)}"
                )
            good=[p for p in self.positions if 0<=p<int(x.shape[1])]
            if len(good)!=len(self.positions):
                bad=sorted(set(self.positions)-set(good))
                raise RuntimeError(
                    f"Requested token positions outside sequence: {bad}, S={x.shape[1]}"
                )
            idx=torch.as_tensor(good,device=x.device,dtype=torch.long)
            return (
                x[0].index_select(0,idx)
                .detach().float().cpu().numpy().astype(np.float32)
            )

        for L in self.layers:
            block=decoder_layers[L]
            attn=spatialscan.resolve_attn(block)
            op=spatialscan.resolve_o_proj(attn)
            mlp=resolve_mlp(block)

            def make_block_pre(layer):
                def hook(_m,inputs):
                    x=first_tensor_any(inputs)
                    self.block_in[layer]=take_positions(x)
                return hook

            def make_attn_hook(layer):
                def hook(_m,_inp,out):
                    x=first_tensor_any(out)
                    self.attn_out[layer]=take_positions(x)
                    return None
                return hook

            def make_pre_o_hook(layer):
                def hook(_m,inputs):
                    x=first_tensor_any(inputs)
                    self.pre_o[layer]=take_positions(x)
                return hook

            def make_mlp_hook(layer):
                def hook(_m,_inp,out):
                    x=first_tensor_any(out)
                    self.mlp_out[layer]=take_positions(x)
                    return None
                return hook

            def make_block_out(layer):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    self.block_out[layer]=take_positions(x)
                    return None
                return hook

            self.handles.append(
                block.register_forward_pre_hook(make_block_pre(L))
            )
            self.handles.append(
                attn.register_forward_hook(make_attn_hook(L))
            )
            self.handles.append(
                op.register_forward_pre_hook(make_pre_o_hook(L))
            )
            self.handles.append(
                mlp.register_forward_hook(make_mlp_hook(L))
            )
            self.handles.append(
                block.register_forward_hook(make_block_out(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def run_formation_capture(model,decoder_layers,batch,layers,positions):
    cap=FormationCapture(
        decoder_layers=decoder_layers,
        layers=layers,
        positions=positions,
    )
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        _=model(**kw)

        missing=[]
        for L in layers:
            for name,d in [
                ("block_in",cap.block_in),
                ("attn_out",cap.attn_out),
                ("mlp_out",cap.mlp_out),
                ("block_out",cap.block_out),
                ("pre_o",cap.pre_o),
            ]:
                if L not in d:
                    missing.append((L,name))
        if missing:
            raise RuntimeError(f"Capture missing: {missing[:20]}")

        return {
            "block_in":cap.block_in,
            "attn_out":cap.attn_out,
            "mlp_out":cap.mlp_out,
            "block_out":cap.block_out,
            "pre_o":cap.pre_o,
            "positions":list(cap.positions),
        }
    finally:
        cap.close()


# =============================================================================
# Analysis
# =============================================================================

def pos_index(cap,position):
    return cap["positions"].index(int(position))


def analyze_one_state(
    *,
    sid,
    gt,
    causal_row,
    real_cap,
    gray_cap,
    trace_layers,
    decoder_layers,
    geom,
):
    C=int(causal_row.source_layer)
    p=int(causal_row.position)
    pi=pos_index(real_cap,p)

    target=(
        real_cap["block_out"][C][pi]
        - gray_cap["block_out"][C][pi]
    ).astype(np.float32)

    tn=float(np.linalg.norm(target))
    if tn<=EPS:
        return [],[]

    formation_rows=[]
    head_rows=[]

    for L in trace_layers:
        if L>C:
            continue

        din=(
            real_cap["block_in"][L][pi]
            - gray_cap["block_in"][L][pi]
        ).astype(np.float32)
        datt=(
            real_cap["attn_out"][L][pi]
            - gray_cap["attn_out"][L][pi]
        ).astype(np.float32)
        dmlp=(
            real_cap["mlp_out"][L][pi]
            - gray_cap["mlp_out"][L][pi]
        ).astype(np.float32)
        dout=(
            real_cap["block_out"][L][pi]
            - gray_cap["block_out"][L][pi]
        ).astype(np.float32)

        closure=(din+datt+dmlp-dout).astype(np.float32)
        closure_rel=float(
            np.linalg.norm(closure)/max(float(np.linalg.norm(dout)),EPS)
        )

        pin=target_progress(din,target)
        patt=target_progress(datt,target)
        pmlp=target_progress(dmlp,target)
        pout=target_progress(dout,target)

        formation_rows.append({
            "sid":sid,
            "gt":gt,
            "causal_rank":int(causal_row.rank),
            "causal_text_rank":int(causal_row.causal_text_rank),
            "causal_layer":C,
            "causal_position":p,
            "causal_token":str(causal_row.token),
            "causal_category":str(causal_row.broad_category),
            "trace_layer":L,
            "distance_to_causal":C-L,
            "target_rg_norm":tn,

            "input_rg_norm":float(np.linalg.norm(din)),
            "attn_rg_norm":float(np.linalg.norm(datt)),
            "mlp_rg_norm":float(np.linalg.norm(dmlp)),
            "output_rg_norm":float(np.linalg.norm(dout)),

            "input_cos_target":cosine(din,target),
            "attn_cos_target":cosine(datt,target),
            "mlp_cos_target":cosine(dmlp,target),
            "output_cos_target":cosine(dout,target),

            "input_target_progress":pin,
            "attn_target_progress":patt,
            "mlp_target_progress":pmlp,
            "update_target_progress":patt+pmlp,
            "output_target_progress":pout,
            "progress_closure_error":pout-(pin+patt+pmlp),

            "input_norm_ratio":norm_ratio(din,target),
            "attn_norm_ratio":norm_ratio(datt,target),
            "mlp_norm_ratio":norm_ratio(dmlp,target),
            "output_norm_ratio":norm_ratio(dout,target),

            "block_vector_closure_relative_error":closure_rel,
        })

        # Exact head decomposition of Delta attention.
        zr=real_cap["pre_o"][L][pi]
        zg=gray_cap["pre_o"][L][pi]
        dz=(zr-zg).astype(np.float32)

        H=int(geom[L]["n_heads"])
        D=int(geom[L]["head_dim"])

        attn=spatialscan.resolve_attn(decoder_layers[L])
        op=spatialscan.resolve_o_proj(attn)
        W=op.weight.detach().float().cpu().numpy().astype(np.float32)

        head_sum=np.zeros_like(datt,dtype=np.float32)

        for h in range(H):
            dh=dz[h*D:(h+1)*D]
            msg=(W[:,h*D:(h+1)*D]@dh).astype(np.float32)
            head_sum+=msg

            head_rows.append({
                "sid":sid,
                "gt":gt,
                "causal_rank":int(causal_row.rank),
                "causal_text_rank":int(causal_row.causal_text_rank),
                "causal_layer":C,
                "causal_position":p,
                "causal_token":str(causal_row.token),
                "causal_category":str(causal_row.broad_category),
                "trace_layer":L,
                "distance_to_causal":C-L,
                "head":h,
                "head_name":hname(L,h),
                "is_direction_head":hname(L,h) in DIRECTION_TOP20,

                "head_preO_rg_norm":float(np.linalg.norm(dh)),
                "head_postO_rg_norm":float(np.linalg.norm(msg)),
                "head_cos_final_target":cosine(msg,target),
                "head_final_target_progress":target_progress(msg,target),

                "head_cos_layer_attn_rg":cosine(msg,datt),
                "head_layer_attn_projection_fraction":(
                    float(np.dot(msg,datt)/max(float(np.dot(datt,datt)),EPS))
                    if float(np.linalg.norm(datt))>EPS
                    else float("nan")
                ),
            })

        head_closure=float(
            np.linalg.norm(head_sum-datt)/
            max(float(np.linalg.norm(datt)),EPS)
        )
        formation_rows[-1]["head_sum_vs_attn_relative_error"]=head_closure
        formation_rows[-1]["head_progress_sum"]=target_progress(head_sum,target)
        formation_rows[-1]["head_vs_attn_progress_error"]=(
            formation_rows[-1]["head_progress_sum"]-patt
        )

    return formation_rows,head_rows


# =============================================================================
# Summaries
# =============================================================================

def formation_summary(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    group_cols=["causal_layer","trace_layer","distance_to_causal"]

    for key,g in df.groupby(group_cols):
        C,L,d=key
        rows.append({
            "causal_layer":int(C),
            "trace_layer":int(L),
            "distance_to_causal":int(d),
            "N_states":len(g),
            "N_samples":g.sid.nunique(),

            "mean_input_target_progress":safe_mean(g.input_target_progress),
            "mean_attn_target_progress":safe_mean(g.attn_target_progress),
            "mean_mlp_target_progress":safe_mean(g.mlp_target_progress),
            "mean_update_target_progress":safe_mean(g.update_target_progress),
            "mean_output_target_progress":safe_mean(g.output_target_progress),

            "median_output_target_progress":safe_median(g.output_target_progress),
            "mean_output_cos_target":safe_mean(g.output_cos_target),
            "median_output_cos_target":safe_median(g.output_cos_target),

            "mean_input_norm_ratio":safe_mean(g.input_norm_ratio),
            "mean_attn_norm_ratio":safe_mean(g.attn_norm_ratio),
            "mean_mlp_norm_ratio":safe_mean(g.mlp_norm_ratio),
            "mean_output_norm_ratio":safe_mean(g.output_norm_ratio),

            "mean_block_closure_error":safe_mean(
                g.block_vector_closure_relative_error
            ),
            "mean_head_sum_closure_error":safe_mean(
                g.head_sum_vs_attn_relative_error
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["causal_layer","trace_layer"]
    ).reset_index(drop=True)


def distance_summary(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for d,g in df.groupby("distance_to_causal"):
        rows.append({
            "distance_to_causal":int(d),
            "N_states":len(g),
            "N_samples":g.sid.nunique(),
            "mean_input_target_progress":safe_mean(g.input_target_progress),
            "mean_attn_target_progress":safe_mean(g.attn_target_progress),
            "mean_mlp_target_progress":safe_mean(g.mlp_target_progress),
            "mean_update_target_progress":safe_mean(g.update_target_progress),
            "mean_output_target_progress":safe_mean(g.output_target_progress),
            "mean_output_cos_target":safe_mean(g.output_cos_target),
            "mean_output_norm_ratio":safe_mean(g.output_norm_ratio),
        })
    return pd.DataFrame(rows).sort_values(
        "distance_to_causal",ascending=False
    ).reset_index(drop=True)


def head_summary(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (L,h,name,isdir),g in df.groupby(
        ["trace_layer","head","head_name","is_direction_head"]
    ):
        v=g.head_final_target_progress.to_numpy(float)
        rows.append({
            "trace_layer":int(L),
            "head":int(h),
            "head_name":str(name),
            "is_direction_head":bool(isdir),
            "N_edges":len(g),
            "N_samples":g.sid.nunique(),
            "mean_final_target_progress":safe_mean(v),
            "median_final_target_progress":safe_median(v),
            "mean_positive_progress":safe_mean(np.maximum(v,0)),
            "mean_negative_progress":safe_mean(np.minimum(v,0)),
            "positive_fraction":float(np.mean(v>0)),
            "mean_abs_progress":safe_mean(np.abs(v)),
            "mean_head_cos_final_target":safe_mean(g.head_cos_final_target),
            "mean_head_postO_rg_norm":safe_mean(g.head_postO_rg_norm),
            "mean_layer_attn_projection_fraction":safe_mean(
                g.head_layer_attn_projection_fraction
            ),
        })

    out=pd.DataFrame(rows)
    return out.sort_values(
        ["mean_final_target_progress","positive_fraction"],
        ascending=[False,False],
    ).reset_index(drop=True)


def top_heads_per_state(head_df,top_n):
    if len(head_df)==0:
        return pd.DataFrame()

    rows=[]
    group_cols=[
        "sid","causal_layer","causal_position","causal_text_rank","trace_layer"
    ]
    for _,g in head_df.groupby(group_cols,sort=False):
        z=g[g.head_final_target_progress>0].sort_values(
            "head_final_target_progress",ascending=False
        ).head(int(top_n)).copy()
        if len(z):
            z["positive_head_rank"]=np.arange(1,len(z)+1)
            rows.append(z)
    return pd.concat(rows,ignore_index=True) if rows else head_df.iloc[:0].copy()


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers=parse_layers(a.causal_layers)
    trace_layers=parse_layers(a.trace_layers)

    outdir=Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)
    error_path=outdir/"errors.jsonl"

    two,meta,rec_by_sid=load_data(a)

    ranked_sids=set(
        pd.to_numeric(
            pd.read_csv(a.ranked_causal,usecols=["sid"])["sid"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )
    eval_meta=[m for m in meta if int(m["sid"]) in ranked_sids]
    if a.eval_max_samples>0:
        eval_meta=traj.stratified_cap(
            eval_meta,a.eval_max_samples,a.seed+1
        )

    eval_sids={int(m["sid"]) for m in eval_meta}
    causal_sel=load_causal_selection(
        Path(a.ranked_causal),
        eval_sids,
        causal_layers,
        a.causal_top_k,
    )
    causal_sel.to_csv(outdir/"selected_causal_states.csv",index=False)

    causal_by_sid={
        int(sid):g.sort_values("rank").copy()
        for sid,g in causal_sel.groupby("sid")
    }

    model=processor=None
    all_form=[]
    all_head=[]
    sample_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        n_layers=len(decoder_layers)
        bad=[L for L in trace_layers+causal_layers if not 0<=L<n_layers]
        if bad:
            raise ValueError(
                f"Layer(s) outside decoder range 0..{n_layers-1}: {sorted(set(bad))}"
            )

        # Need every traced layer that can be upstream of a selected causal state.
        max_causal=max(causal_layers)
        trace_layers=[L for L in trace_layers if L<=max_causal]
        geom=infer_head_geometry(model,decoder_layers,trace_layers)

        for m in tqdm(eval_meta,desc="CAUSAL RG FORMATION"):
            sid=int(m["sid"])
            if sid not in causal_by_sid:
                continue

            real=gray=None
            try:
                rows=causal_by_sid[sid]
                positions=sorted(set(map(int,rows.position.tolist())))
                if not positions:
                    continue

                # Per sample we only need trace blocks <= max actual target C.
                max_C=int(rows.source_layer.max())
                layers_sid=[L for L in trace_layers if L<=max_C]
                if not layers_sid:
                    continue

                real=base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real=real.convert("RGB")
                gray=dyn.make_gray_image(real,a.gray_value)

                rb=base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb=base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                ids=rb["input_ids"][0].detach().cpu().tolist()
                ids_g=gb["input_ids"][0].detach().cpu().tolist()
                if ids!=ids_g:
                    raise RuntimeError("REAL/GRAY tokenization differs")

                if max(positions)>=len(ids):
                    raise RuntimeError(
                        f"causal position max={max(positions)} >= seq={len(ids)}"
                    )

                rc=run_formation_capture(
                    model,decoder_layers,rb,layers_sid,positions
                )
                gc_=run_formation_capture(
                    model,decoder_layers,gb,layers_sid,positions
                )

                n_state_before=len(all_form)
                n_head_before=len(all_head)

                for r in rows.itertuples():
                    C=int(r.source_layer)
                    if C not in layers_sid:
                        continue

                    fr,hr=analyze_one_state(
                        sid=sid,
                        gt=m["gt"],
                        causal_row=r,
                        real_cap=rc,
                        gray_cap=gc_,
                        trace_layers=layers_sid,
                        decoder_layers=decoder_layers,
                        geom=geom,
                    )
                    all_form.extend(fr)
                    all_head.extend(hr)

                fnew=all_form[n_state_before:]
                hnew=all_head[n_head_before:]

                sample_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "N_causal_states":len(rows),
                    "N_formation_rows":len(fnew),
                    "N_head_rows":len(hnew),
                    "max_block_closure_error":(
                        max(
                            [
                                float(x["block_vector_closure_relative_error"])
                                for x in fnew
                            ],
                            default=float("nan"),
                        )
                    ),
                    "max_head_sum_closure_error":(
                        max(
                            [
                                float(x["head_sum_vs_attn_relative_error"])
                                for x in fnew
                            ],
                            default=float("nan"),
                        )
                    ),
                })

            except Exception as exc:
                append_jsonl(error_path,{
                    "phase":"sample",
                    "sid":sid,
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc().splitlines()[-30:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
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

        form_df=pd.DataFrame(all_form)
        head_df=pd.DataFrame(all_head)
        sample_df=pd.DataFrame(sample_rows)

        form_df.to_csv(
            outdir/"formation_per_state_layer.csv",index=False
        )
        head_df.to_csv(
            outdir/"head_rg_per_state_layer.csv",index=False
        )
        sample_df.to_csv(
            outdir/"sample_summary.csv",index=False
        )

        fs=formation_summary(form_df)
        fs.to_csv(
            outdir/"formation_layer_summary.csv",index=False
        )

        ds=distance_summary(form_df)
        ds.to_csv(
            outdir/"formation_distance_summary.csv",index=False
        )

        hs=head_summary(head_df)
        if len(hs):
            hs["global_head_rank"]=np.arange(1,len(hs)+1)
        hs.to_csv(
            outdir/"head_rg_summary.csv",index=False
        )

        top=top_heads_per_state(
            head_df,a.top_heads_per_state
        )
        top.to_csv(
            outdir/"top_heads_per_causal_state.csv",index=False
        )

        # Aggregate which layers CREATE positive target-axis RG.
        if len(form_df):
            layer_create=(
                form_df.groupby("trace_layer")
                .agg(
                    N_states=("sid","size"),
                    N_samples=("sid","nunique"),
                    mean_attn_progress=("attn_target_progress","mean"),
                    mean_mlp_progress=("mlp_target_progress","mean"),
                    mean_update_progress=("update_target_progress","mean"),
                    mean_output_progress=("output_target_progress","mean"),
                    positive_update_fraction=(
                        "update_target_progress",
                        lambda x: float(np.mean(np.asarray(x,dtype=float)>0))
                    ),
                )
                .reset_index()
                .sort_values("mean_update_progress",ascending=False)
            )
        else:
            layer_create=pd.DataFrame()
        layer_create.to_csv(
            outdir/"formation_layer_creation_ranking.csv",index=False
        )

        # Top heads restricted to known Direction set.
        if len(hs):
            dir_hs=hs[hs.is_direction_head].copy()
        else:
            dir_hs=pd.DataFrame()
        dir_hs.to_csv(
            outdir/"direction_head_rg_summary.csv",index=False
        )

        report=[
            "="*190,
            "CAUSAL TOKEN IMAGE-GRAY FORMATION DECOMPOSITION",
            "="*190,
            f"model={a.model} repo={spec.repo_id}",
            f"decoder={decoder_path} n_layers={n_layers}",
            f"N eval requested/completed={len(eval_meta)}/{sample_df.sid.nunique() if len(sample_df) else 0}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"trace layers={trace_layers}",
            "",
            "SANITY / EXACT DECOMPOSITION",
            "-"*190,
        ]

        if len(sample_df):
            report += [
                f"mean max block closure error per sample = "
                f"{safe_mean(sample_df.max_block_closure_error):.6e}",
                f"mean max head-sum closure error per sample = "
                f"{safe_mean(sample_df.max_head_sum_closure_error):.6e}",
            ]
        else:
            report += ["EMPTY"]

        report += [
            "",
            "LAYERS THAT ADD THE MOST FINAL-TARGET RG PROGRESS",
            "-"*190,
            layer_create.head(20).to_string(
                index=False,float_format=lambda x:f"{x:.5f}"
            ) if len(layer_create) else "EMPTY",
            "",
            "GLOBAL HEADS BY SIGNED CONTRIBUTION TO FINAL CAUSAL-RG DIRECTION",
            "-"*190,
            hs.head(30).to_string(
                index=False,float_format=lambda x:f"{x:.6f}"
            ) if len(hs) else "EMPTY",
            "",
            "DIRECTION HEADS",
            "-"*190,
            dir_hs.head(30).to_string(
                index=False,float_format=lambda x:f"{x:.6f}"
            ) if len(dir_hs) else "EMPTY",
            "",
            "How to read:",
            "  output_target_progress at target layer C should be ~1 by definition.",
            "  input_target_progress is RG already carried into a block.",
            "  attn_target_progress is new signed target-axis RG added by attention at that block.",
            "  mlp_target_progress is new signed target-axis RG added by MLP at that block.",
            "  update_target_progress = attention + MLP.",
            "  Positive means adding along the final causal-RG direction; negative means cancelling/rotating away on that axis.",
            "  Per-head progress exactly sums to the attention progress of that same block/token (up to numerical precision).",
            "  This is formation/trajectory decomposition, not yet a cross-layer causal mediation proof.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,encoding="utf-8"
        )

        write_json(outdir/"metadata.json",{
            "script":"analyze_causal_rg_formation_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "n_layers":n_layers,
            "eval_requested_N":len(eval_meta),
            "eval_completed_N":(
                int(sample_df.sid.nunique()) if len(sample_df) else 0
            ),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "trace_layers":trace_layers,
            "definition":{
                "target":"Delta h_C,p = h_real[C,p] - h_gray[C,p]",
                "block_identity":"Delta h_L = Delta input_L + Delta attention_L + Delta MLP_L",
                "target_progress":"dot(component,target) / ||target||^2",
                "head_message":"W_O^{L,h} (z_real[L,h,p]-z_gray[L,h,p])",
            },
        })

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__=="__main__":
    main()
