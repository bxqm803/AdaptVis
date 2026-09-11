#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_head_token_to_causal_rg_reconstruction_v1.py

Question
========
For a behaviorally causal text state c=(C,p), can we turn ON a small number of
specific HEAD->TOKEN edges so that the downstream causal state moves toward the
same target as direct Real-Gray causal-state amplification?

Target
======
For each selected causal state:

    delta_c = h_real[C,p] - h_gray[C,p]

Direct causal-state reference:

    h'[C,p] = h_real[C,p] + alpha_causal * delta_c

Head-token intervention
=======================
For an upstream attention head a=(L,h), L<=C, at the SAME query/token position p:

    delta_z[L,h,p] = z_real[L,h,p] - z_gray[L,h,p]   # PRE-W_O

Enhance ONLY that head at ONLY that token:

    z'[L,h,p] = z_real[L,h,p] + alpha * delta_z[L,h,p]

Implemented equivalently after W_O:

    attn_out'[L,p]
      = attn_out[L,p] + alpha * W_O^{L,h} delta_z[L,h,p]

All other heads and all other token positions are unchanged.

This is an exact "one head -> one query token" intervention. It does NOT edit
attention probabilities and does NOT modify model weights.

Binary selection
================
No head-specific coefficient is learned. Every selected edge is either:

    OFF: m=0
    ON : m=1

and all selected edges share the same alpha.

We rank candidate head->causal-token edges with a first-order reconstruction score.
For causal state c=(C,p):

    u_c = normalize(delta_c)
    J_c = <h_real[C,p], u_c>

For PRE-W_O head output z:

    score_same(L,h -> C,p)
      = delta_z[L,h,p]^T dJ_c/dz[L,h,p]

Positive score means that increasing this head's current sample-specific
Real-Gray output at the causal-token position locally pushes the downstream causal
state toward its own Real-Gray target.

For comparison, also compute:

    score_all_text
      = sum_{q in text} delta_z[L,h,q]^T dJ_c/dz[L,h,q]

The intervention selection defaults to score_same because the main experiment
is specifically HEAD h -> CAUSAL TOKEN p.

Two intervention modes
======================
same_token:
    Use the selected (L,h,p) edges exactly. This tests whether a particular head
    can be strengthened specifically at the causal token position.

all_query:
    Take the union of heads selected by same_token ranking and amplify each selected
    head at ALL text query positions. This is a control for indirect routing:
    another query token q may later influence causal token p.

Causal reconstruction metrics
=============================
After intervention:

    move_c = h_patched[C,p] - h_real[C,p]

Report:

    cosine       = cos(move_c, delta_c)
    progress     = <move_c, delta_c> / ||delta_c||^2
    norm_ratio   = ||move_c|| / ||delta_c||
    relative_err = ||move_c - delta_c|| / ||delta_c||

Ideal reconstruction:
    cosine -> 1
    progress -> 1
    norm_ratio -> 1
    relative_err -> 0

The script also runs actual greedy generation and compares against direct
causal-state Real-Gray amplification.

Important
=========
The causal states are read from the existing oracle writer-guided ranked CSV,
so the selection of WHICH causal states remains oracle. However, the target
delta_c = Real-Gray for the chosen state and the head-edge ranking itself do not
use the GT relation.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_head_token_to_causal_rg_reconstruction_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --head-layers 18-26 \
  --edge-ks 1,2,4,8 \
  --scales 0.5,1,2 \
  --modes same_token,all_query \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_head_token_to_causal_rg_n80_v1 \
  --overwrite

Useful first quick run:
    --eval-max-samples 20 --edge-ks 1,2,4 --scales 1 --modes same_token

Outputs
=======
selected_causal_states.csv
edge_scores.csv
selected_edges.csv
generation_per_sample.csv
generation_summary.csv
reconstruction_per_state.csv
reconstruction_summary.csv
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

DIRECTION_TOP20 = {
    "L26H03","L23H01","L23H05","L26H02","L22H09",
    "L23H00","L22H13","L22H02","L21H14","L23H10",
    "L21H05","L22H12","L21H01","L26H01","L27H02",
    "L27H01","L21H11","L22H14","L21H03","L22H10",
}


# =============================================================================
# CLI / utilities
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
        "--head-layers",
        default="18-26",
        help="Candidate attention layers. Only L<=causal layer C is eligible.",
    )
    p.add_argument(
        "--candidate-heads",
        default="",
        help=(
            'Optional explicit subset "23:1,23:5,26:2". '
            "Empty scans every head in --head-layers."
        ),
    )

    p.add_argument(
        "--edge-ks",
        default="1,2,4,8",
        help="Top-K positive head->causal-token edges PER causal state.",
    )
    p.add_argument("--scales", default="0.5,1,2")
    p.add_argument(
        "--modes",
        default="same_token,all_query",
        help="same_token and/or all_query.",
    )
    p.add_argument("--causal-scale", type=float, default=1.0)

    p.add_argument(
        "--all-query-scope",
        choices=["text","all"],
        default="text",
        help="Positions amplified in all_query control.",
    )
    p.add_argument(
        "--exclude-last-query",
        action="store_true",
        help="Exclude prompt last query from all_query mode.",
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)

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


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_modes(text: str) -> List[str]:
    modes = [x.strip() for x in str(text).split(",") if x.strip()]
    valid = {"same_token","all_query"}
    bad = set(modes)-valid
    if bad:
        raise ValueError(f"Unknown modes: {sorted(bad)}")
    return modes


def parse_candidate_heads(text: str) -> Optional[set]:
    text = str(text).strip()
    if not text:
        return None
    out = set()
    for item in text.split(","):
        item = item.strip().upper().replace("L","")
        if not item:
            continue
        if ":" in item:
            L,h = item.split(":",1)
        else:
            m = re.fullmatch(r"(\d+)H(\d+)",item)
            if not m:
                raise ValueError(f"Bad head spec: {item}")
            L,h = m.group(1),m.group(2)
        out.add((int(L),int(h)))
    return out


def hname(L,h):
    return f"L{int(L):02d}H{int(h):02d}"


def normalize_np(v):
    v = np.asarray(v,dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v)
    return (v/n).astype(np.float32)


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


# =============================================================================
# Data / model
# =============================================================================

def load_data(a):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records,_audit = two.load_records("coco_two",Path(a.data_root),None)
    rec_by_sid = {int(r.sid):r for r in records}

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

    try:
        model=cls.from_pretrained(spec.repo_id,**kw)
    except TypeError:
        kw["torch_dtype"]=kw.pop("dtype")
        model=cls.from_pretrained(spec.repo_id,**kw)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    processor=AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model,processor)
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
        raise RuntimeError(f"{path} missing {sorted(miss)}")

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
# Capture
# =============================================================================

def infer_head_geometry(model,decoder_layers,layers):
    cfg=spatialscan.get_text_config(model)
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
            raise RuntimeError(f"L{L}: width={width}, H={H}")
        out[L]={"n_heads":H,"head_dim":width//H}
    return out


class CpuCapture:
    def __init__(self,decoder_layers,state_layers,head_layers):
        self.states={}
        self.pre_o={}
        self.handles=[]

        for L in sorted(set(map(int,state_layers))):
            def make_state_hook(layer):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    self.states[layer]=(
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook
            self.handles.append(
                decoder_layers[L].register_forward_hook(make_state_hook(L))
            )

        for L in sorted(set(map(int,head_layers))):
            attn=spatialscan.resolve_attn(decoder_layers[L])
            op=spatialscan.resolve_o_proj(attn)

            def make_pre_hook(layer):
                def hook(_m,inputs):
                    self.pre_o[layer]=(
                        inputs[0].detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            self.handles.append(op.register_forward_pre_hook(make_pre_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def run_cpu_capture(model,decoder_layers,batch,state_layers,head_layers):
    cap=CpuCapture(decoder_layers,state_layers,head_layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        _=model(**kw)
        return cap.states,cap.pre_o
    finally:
        cap.close()


class GraphCapture:
    def __init__(self,decoder_layers,state_layers,head_layers,cut_layer=0):
        self.states={}
        self.pre_o={}
        self.handles=[]

        def cut_hook(_m,_inp,out):
            x=traj.first_tensor(out)
            y=x.detach().clone().requires_grad_(True)
            return traj.replace_first_tensor(out,y)

        self.handles.append(
            decoder_layers[int(cut_layer)].register_forward_hook(cut_hook)
        )

        for L in sorted(set(map(int,state_layers))):
            def make_state_hook(layer):
                def hook(_m,_inp,out):
                    self.states[layer]=traj.first_tensor(out)
                    return None
                return hook
            self.handles.append(
                decoder_layers[L].register_forward_hook(make_state_hook(L))
            )

        for L in sorted(set(map(int,head_layers))):
            attn=spatialscan.resolve_attn(decoder_layers[L])
            op=spatialscan.resolve_o_proj(attn)

            def make_pre_hook(layer):
                def hook(_m,inputs):
                    self.pre_o[layer]=inputs[0]
                return hook

            self.handles.append(op.register_forward_pre_hook(make_pre_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


def text_query_positions(model,processor,batch,ids,scope="text",exclude_last=False):
    S=len(ids)
    if scope=="all":
        qs=list(range(S))
    else:
        vis=set(map(int,base.resolve_visual_indices(model,processor,batch,ids)))
        qs=[q for q in range(S) if q not in vis]
    if exclude_last:
        qs=[q for q in qs if q != S-1]
    return qs


# =============================================================================
# Edge scoring
# =============================================================================

def score_edges_for_sample(
    *,
    model,
    processor,
    decoder_layers,
    rb,
    gb,
    causal_rows,
    head_layers,
    geom,
    candidate_heads,
    all_query_scope,
    exclude_last,
):
    ids=rb["input_ids"][0].detach().cpu().tolist()
    ids_g=gb["input_ids"][0].detach().cpu().tolist()
    if ids!=ids_g:
        raise RuntimeError("Real/Gray tokenization mismatch")

    q_all=text_query_positions(
        model,processor,rb,ids,
        scope=all_query_scope,
        exclude_last=exclude_last,
    )

    state_layers=sorted(set(map(int,causal_rows.source_layer.tolist())))

    gray_states,gray_pre=run_cpu_capture(
        model,decoder_layers,gb,state_layers,head_layers
    )

    cap=GraphCapture(
        decoder_layers,
        state_layers=state_layers,
        head_layers=head_layers,
        cut_layer=0,
    )

    try:
        kw=dict(rb)
        kw["use_cache"]=False
        _=model(**kw)

        real_states={
            C:cap.states[C].detach().float().cpu().numpy().astype(np.float32)
            for C in state_layers
        }
        real_pre={
            L:cap.pre_o[L].detach().float().cpu().numpy().astype(np.float32)
            for L in head_layers
        }

        score_rows=[]

        causal_list=list(causal_rows.sort_values("rank").itertuples())
        for ci,r in enumerate(causal_list):
            C=int(r.source_layer)
            p=int(r.position)

            if p>=real_states[C].shape[1] or p>=gray_states[C].shape[1]:
                continue

            delta_c=(
                real_states[C][0,p]-gray_states[C][0,p]
            ).astype(np.float32)
            dn=float(np.linalg.norm(delta_c))
            if dn<=EPS:
                continue

            u=torch.as_tensor(
                delta_c/dn,
                device=cap.states[C].device,
                dtype=torch.float32,
            )
            Jc=torch.dot(cap.states[C][0,p].float(),u)

            eligible_layers=[L for L in head_layers if L<=C]
            tensors=[cap.pre_o[L] for L in eligible_layers]

            grads=torch.autograd.grad(
                Jc,
                tensors,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            for L,g in zip(eligible_layers,grads):
                if g is None:
                    continue

                zr=real_pre[L]
                zg=gray_pre[L]
                gg=g.detach().float().cpu().numpy().astype(np.float32)

                n=min(zr.shape[1],zg.shape[1],gg.shape[1])
                if not (0<=p<n):
                    continue

                H=int(geom[L]["n_heads"])
                D=int(geom[L]["head_dim"])

                qs=[q for q in q_all if q<n]

                for h in range(H):
                    if candidate_heads is not None and (L,h) not in candidate_heads:
                        continue

                    sl=slice(h*D,(h+1)*D)

                    dzp=zr[0,p,sl]-zg[0,p,sl]
                    gp=gg[0,p,sl]
                    score_same=float(np.dot(dzp,gp))

                    if qs:
                        dz=zr[0,qs,sl]-zg[0,qs,sl]
                        ga=gg[0,qs,sl]
                        score_all=float(np.sum(dz*ga))
                    else:
                        score_all=float("nan")

                    score_rows.append({
                        "causal_rank":int(r.rank),
                        "causal_text_rank":int(r.causal_text_rank),
                        "causal_layer":C,
                        "causal_position":p,
                        "causal_token":str(r.token),
                        "causal_category":str(r.broad_category),
                        "causal_rg_norm":dn,
                        "head_layer":L,
                        "head":h,
                        "head_name":hname(L,h),
                        "is_direction_head":hname(L,h) in DIRECTION_TOP20,
                        "score_same_token":score_same,
                        "score_all_text":score_all,
                        "delta_z_same_norm":float(np.linalg.norm(dzp)),
                        "grad_same_norm":float(np.linalg.norm(gp)),
                    })

        return score_rows,real_states,gray_states,real_pre,gray_pre,q_all

    finally:
        cap.close()


# =============================================================================
# Selection and patch maps
# =============================================================================

def select_edges(score_df,edge_k):
    rows=[]
    for key,g in score_df.groupby(
        ["sid","causal_layer","causal_position","causal_text_rank"],
        sort=False,
    ):
        z=g[g.score_same_token>0].sort_values(
            "score_same_token",ascending=False
        ).head(int(edge_k)).copy()
        if len(z)==0:
            continue
        z["edge_rank_for_state"]=np.arange(1,len(z)+1)
        rows.append(z)
    return pd.concat(rows,ignore_index=True) if rows else score_df.iloc[:0].copy()


def build_same_token_patch(
    *,
    decoder_layers,
    selected_edges,
    real_pre,
    gray_pre,
    geom,
):
    # Deduplicate actual (L,h,p) interventions. The same upstream edge may have
    # been selected for multiple downstream causal layers at the same position.
    uniq=selected_edges[
        ["head_layer","head","causal_position"]
    ].drop_duplicates()

    patch_map=defaultdict(dict)
    rows=[]

    for r in uniq.itertuples():
        L=int(r.head_layer)
        h=int(r.head)
        p=int(r.causal_position)

        if L not in real_pre or L not in gray_pre:
            continue

        zr=real_pre[L]
        zg=gray_pre[L]
        n=min(zr.shape[1],zg.shape[1])
        if not (0<=p<n):
            continue

        H=int(geom[L]["n_heads"])
        D=int(geom[L]["head_dim"])
        if not (0<=h<H):
            continue

        dz=(zr[0,p,h*D:(h+1)*D]-zg[0,p,h*D:(h+1)*D]).astype(np.float32)

        attn=spatialscan.resolve_attn(decoder_layers[L])
        op=spatialscan.resolve_o_proj(attn)
        W=op.weight.detach().float().cpu().numpy().astype(np.float32)
        msg=(W[:,h*D:(h+1)*D]@dz).astype(np.float32)

        if p not in patch_map[L]:
            patch_map[L][p]=np.zeros_like(msg)
        patch_map[L][p]+=msg

        rows.append({
            "head_layer":L,
            "head":h,
            "head_name":hname(L,h),
            "query_position":p,
            "preO_delta_norm":float(np.linalg.norm(dz)),
            "postO_message_norm":float(np.linalg.norm(msg)),
        })

    return {L:dict(v) for L,v in patch_map.items()},rows


def build_all_query_patch(
    *,
    decoder_layers,
    selected_edges,
    real_pre,
    gray_pre,
    geom,
    q_all,
):
    # In this control, if a head was selected for any causal state, turn on the
    # whole head at all chosen query positions.
    heads=selected_edges[
        ["head_layer","head"]
    ].drop_duplicates()

    patch_map=defaultdict(dict)
    rows=[]

    for r in heads.itertuples():
        L=int(r.head_layer)
        h=int(r.head)

        if L not in real_pre or L not in gray_pre:
            continue

        zr=real_pre[L]
        zg=gray_pre[L]
        n=min(zr.shape[1],zg.shape[1])
        qs=[q for q in q_all if q<n]
        if not qs:
            continue

        H=int(geom[L]["n_heads"])
        D=int(geom[L]["head_dim"])

        attn=spatialscan.resolve_attn(decoder_layers[L])
        op=spatialscan.resolve_o_proj(attn)
        W=op.weight.detach().float().cpu().numpy().astype(np.float32)
        Wh=W[:,h*D:(h+1)*D]

        norms=[]
        for q in qs:
            dz=(zr[0,q,h*D:(h+1)*D]-zg[0,q,h*D:(h+1)*D]).astype(np.float32)
            msg=(Wh@dz).astype(np.float32)

            if q not in patch_map[L]:
                patch_map[L][q]=np.zeros_like(msg)
            patch_map[L][q]+=msg
            norms.append(float(np.linalg.norm(msg)))

        rows.append({
            "head_layer":L,
            "head":h,
            "head_name":hname(L,h),
            "N_query":len(qs),
            "mean_postO_message_norm":safe_mean(norms),
        })

    return {L:dict(v) for L,v in patch_map.items()},rows


def build_causal_rg_patch(real_states,gray_states,causal_rows):
    patch_map=defaultdict(dict)
    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)
        if C not in real_states or C not in gray_states:
            continue
        n=min(real_states[C].shape[1],gray_states[C].shape[1])
        if not (0<=p<n):
            continue
        patch_map[C][p]=(
            real_states[C][0,p]-gray_states[C][0,p]
        ).astype(np.float32)
    return {L:dict(v) for L,v in patch_map.items()}


# =============================================================================
# Patch hooks / generation
# =============================================================================

def first_3d(output):
    if torch.is_tensor(output) and output.ndim==3:
        return output
    if isinstance(output,(tuple,list)):
        for x in output:
            if torch.is_tensor(x) and x.ndim==3:
                return x
    raise RuntimeError("No 3D tensor found")


def replace_first_3d(output,replacement):
    if torch.is_tensor(output):
        return replacement
    if isinstance(output,tuple):
        xs=list(output)
        for i,x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim==3:
                xs[i]=replacement
                return tuple(xs)
    if isinstance(output,list):
        xs=list(output)
        for i,x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim==3:
                xs[i]=replacement
                return xs
    raise RuntimeError("Could not replace 3D tensor")


class AttentionOutputPatch:
    def __init__(self,decoder_layers,patch_map,prompt_len,scale):
        self.decoder_layers=decoder_layers
        self.patch_map=patch_map
        self.prompt_len=int(prompt_len)
        self.scale=float(scale)
        self.handles=[]

    def __enter__(self):
        for L,by_pos in self.patch_map.items():
            attn=spatialscan.resolve_attn(self.decoder_layers[int(L)])

            def make_hook(pos_map):
                def hook(_m,_inp,out):
                    x=first_3d(out)
                    if int(x.shape[1])!=self.prompt_len:
                        return None
                    y=x.clone()
                    for q,vec in pos_map.items():
                        q=int(q)
                        if 0<=q<int(y.shape[1]):
                            y[0,q]+=self.scale*torch.as_tensor(
                                vec,device=y.device,dtype=y.dtype
                            )
                    return replace_first_3d(out,y)
                return hook

            self.handles.append(
                attn.register_forward_hook(make_hook(dict(by_pos)))
            )
        return self

    def __exit__(self,*args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


class BlockOutputPatch:
    def __init__(self,decoder_layers,patch_map,prompt_len,scale):
        self.decoder_layers=decoder_layers
        self.patch_map=patch_map
        self.prompt_len=int(prompt_len)
        self.scale=float(scale)
        self.handles=[]

    def __enter__(self):
        for L,by_pos in self.patch_map.items():
            def make_hook(pos_map):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    if int(x.shape[1])!=self.prompt_len:
                        return None
                    y=x.clone()
                    for p,vec in pos_map.items():
                        p=int(p)
                        if 0<=p<int(y.shape[1]):
                            y[0,p]+=self.scale*torch.as_tensor(
                                vec,device=y.device,dtype=y.dtype
                            )
                    return traj.replace_first_tensor(out,y)
                return hook

            self.handles.append(
                self.decoder_layers[int(L)].register_forward_hook(
                    make_hook(dict(by_pos))
                )
            )
        return self

    def __exit__(self,*args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


class StateCapture:
    def __init__(self,decoder_layers,state_layers,prompt_len):
        self.states={}
        self.handles=[]
        self.prompt_len=int(prompt_len)

        for L in sorted(set(map(int,state_layers))):
            def make_hook(layer):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    if int(x.shape[1])==self.prompt_len:
                        self.states[layer]=(
                            x.detach().float().cpu().numpy().astype(np.float32)
                        )
                    return None
                return hook
            self.handles.append(
                decoder_layers[L].register_forward_hook(make_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def generate_and_capture(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    state_layers,
    max_new_tokens,
    attention_patch=None,
    block_patch=None,
    scale=1.0,
):
    prompt_len=int(batch["input_ids"].shape[1])

    attn_ctx=(
        AttentionOutputPatch(
            decoder_layers,attention_patch,prompt_len,scale
        ) if attention_patch else contextlib.nullcontext()
    )
    block_ctx=(
        BlockOutputPatch(
            decoder_layers,block_patch,prompt_len,scale
        ) if block_patch else contextlib.nullcontext()
    )

    # Register patch hooks first; state capture afterwards so it observes
    # post-patch block outputs.
    with attn_ctx:
        with block_ctx:
            cap=StateCapture(decoder_layers,state_layers,prompt_len)
            try:
                text=base.generate_text(
                    model,processor,batch,max_new_tokens=max_new_tokens
                )
                states=dict(cap.states)
            finally:
                cap.close()

    pred=traj.normalize_relation(base,text)
    return pred,text,states


# =============================================================================
# Reconstruction metrics
# =============================================================================

def reconstruction_rows(
    *,
    sid,
    gt,
    condition,
    edge_k,
    alpha,
    causal_rows,
    real_states,
    gray_states,
    patched_states,
):
    rows=[]
    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)

        if C not in patched_states:
            continue
        n=min(
            real_states[C].shape[1],
            gray_states[C].shape[1],
            patched_states[C].shape[1],
        )
        if not (0<=p<n):
            continue

        target=(
            real_states[C][0,p]-gray_states[C][0,p]
        ).astype(np.float32)
        move=(
            patched_states[C][0,p]-real_states[C][0,p]
        ).astype(np.float32)

        tn=float(np.linalg.norm(target))
        mn=float(np.linalg.norm(move))
        if tn<=EPS:
            continue

        dot=float(np.dot(move,target))
        cos=dot/(max(mn,EPS)*tn)
        progress=dot/(tn*tn)
        norm_ratio=mn/tn
        rel_err=float(np.linalg.norm(move-target)/tn)

        rows.append({
            "sid":sid,
            "gt":gt,
            "condition":condition,
            "edge_K":edge_k,
            "alpha":alpha,
            "causal_rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "causal_layer":C,
            "causal_position":p,
            "causal_token":str(r.token),
            "causal_category":str(r.broad_category),
            "target_rg_norm":tn,
            "move_norm":mn,
            "cos_move_vs_rg":cos,
            "rg_progress":progress,
            "move_over_rg_norm":norm_ratio,
            "relative_reconstruction_error":rel_err,
        })
    return rows


def summarize_generation(df):
    if len(df)==0:
        return pd.DataFrame()

    b=df[df.condition=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]

    for (cond,K,a),g in df[df.condition!="baseline"].groupby(
        ["condition","edge_K","alpha"],dropna=False
    ):
        x=g.set_index("sid")
        common=sorted(set(b.index)&set(x.index))
        if not common:
            continue

        bc=b.loc[common,"correct"].astype(bool).to_numpy()
        pc=x.loc[common,"correct"].astype(bool).to_numpy()
        bp=b.loc[common,"prediction"].astype(str).to_numpy()
        pp=x.loc[common,"prediction"].astype(str).to_numpy()

        w2c=int(np.sum((~bc)&pc))
        c2w=int(np.sum(bc&(~pc)))

        rows.append({
            "condition":cond,
            "edge_K":K,
            "alpha":float(a),
            "N":len(common),
            "baseline_accuracy":float(bc.mean()),
            "patched_accuracy":float(pc.mean()),
            "gain":float(pc.mean()-bc.mean()),
            "wrong_to_correct":w2c,
            "correct_to_wrong":c2w,
            "net":w2c-c2w,
            "changed":int(np.sum(bp!=pp)),
            "repair_rate_on_wrong":w2c/max(int((~bc).sum()),1),
            "preserve_rate_on_correct":1-c2w/max(int(bc.sum()),1),
        })

    return pd.DataFrame(rows).sort_values(
        ["condition","edge_K","alpha"]
    ).reset_index(drop=True)


def summarize_reconstruction(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (cond,K,a),g in df.groupby(
        ["condition","edge_K","alpha"],dropna=False
    ):
        rows.append({
            "condition":cond,
            "edge_K":K,
            "alpha":float(a),
            "N_states":len(g),
            "N_samples":g.sid.nunique(),
            "mean_cos_move_vs_rg":safe_mean(g.cos_move_vs_rg),
            "median_cos_move_vs_rg":safe_median(g.cos_move_vs_rg),
            "mean_rg_progress":safe_mean(g.rg_progress),
            "median_rg_progress":safe_median(g.rg_progress),
            "mean_move_over_rg_norm":safe_mean(g.move_over_rg_norm),
            "median_move_over_rg_norm":safe_median(g.move_over_rg_norm),
            "mean_relative_reconstruction_error":safe_mean(
                g.relative_reconstruction_error
            ),
            "median_relative_reconstruction_error":safe_median(
                g.relative_reconstruction_error
            ),
            "cos_gt_0p5_fraction":float(
                np.mean(g.cos_move_vs_rg.to_numpy(float)>0.5)
            ),
            "progress_0p5_to_1p5_fraction":float(
                np.mean(
                    (g.rg_progress.to_numpy(float)>0.5)&
                    (g.rg_progress.to_numpy(float)<1.5)
                )
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["condition","edge_K","alpha"]
    ).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers=parse_layers(a.causal_layers)
    head_layers=parse_layers(a.head_layers)
    edge_ks=parse_ints(a.edge_ks)
    scales=parse_floats(a.scales)
    modes=parse_modes(a.modes)
    candidate_heads=parse_candidate_heads(a.candidate_heads)

    if min(head_layers)<=0:
        raise ValueError("head-layers must be >=1 because block0 is graph cut")

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
    causal_by_sid={
        int(sid):g.sort_values("rank").copy()
        for sid,g in causal_sel.groupby("sid")
    }
    causal_sel.to_csv(outdir/"selected_causal_states.csv",index=False)

    model=processor=None

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        if max(causal_layers)>=len(decoder_layers):
            raise ValueError("causal layer out of range")
        if max(head_layers)>=len(decoder_layers):
            raise ValueError("head layer out of range")

        geom=infer_head_geometry(model,decoder_layers,head_layers)

        edge_score_rows=[]
        selected_edge_rows=[]
        gen_rows=[]
        recon_rows=[]

        for m in tqdm(eval_meta,desc="HEAD->CAUSAL RG"):
            sid=int(m["sid"])
            if sid not in causal_by_sid:
                continue

            real=gray=None

            try:
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

                causal_rows=causal_by_sid[sid]

                with torch.enable_grad():
                    scores,real_states,gray_states,real_pre,gray_pre,q_all=(
                        score_edges_for_sample(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            rb=rb,
                            gb=gb,
                            causal_rows=causal_rows,
                            head_layers=head_layers,
                            geom=geom,
                            candidate_heads=candidate_heads,
                            all_query_scope=a.all_query_scope,
                            exclude_last=a.exclude_last_query,
                        )
                    )

                for r in scores:
                    r.update({"sid":sid,"gt":m["gt"]})
                    edge_score_rows.append(r)

                score_df=pd.DataFrame(scores)
                if len(score_df)==0:
                    continue
                score_df.insert(0,"sid",sid)

                state_layers=sorted(set(map(int,causal_rows.source_layer)))

                # Baseline generation.
                bpred,btext,bstates=generate_and_capture(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    state_layers=state_layers,
                    max_new_tokens=a.max_new_tokens,
                )
                bok=bpred==m["gt"]
                gen_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"baseline",
                    "edge_K":0,
                    "alpha":0.0,
                    "prediction":bpred,
                    "correct":bok,
                    "text":btext,
                })

                # Direct causal RG reference.
                causal_patch=build_causal_rg_patch(
                    real_states,gray_states,causal_rows
                )
                cpred,ctext,cstates=generate_and_capture(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    state_layers=state_layers,
                    max_new_tokens=a.max_new_tokens,
                    block_patch=causal_patch,
                    scale=a.causal_scale,
                )
                cok=cpred==m["gt"]
                gen_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"causal_rg",
                    "edge_K":0,
                    "alpha":float(a.causal_scale),
                    "prediction":cpred,
                    "correct":cok,
                    "text":ctext,
                })
                recon_rows.extend(
                    reconstruction_rows(
                        sid=sid,
                        gt=m["gt"],
                        condition="causal_rg",
                        edge_k=0,
                        alpha=float(a.causal_scale),
                        causal_rows=causal_rows,
                        real_states=real_states,
                        gray_states=gray_states,
                        patched_states=cstates,
                    )
                )

                for K in edge_ks:
                    sel=select_edges(score_df,K)
                    if len(sel)==0:
                        continue

                    sel=sel.copy()
                    sel["edge_K"]=K
                    selected_edge_rows.extend(sel.to_dict("records"))

                    for mode in modes:
                        if mode=="same_token":
                            patch_map,_stats=build_same_token_patch(
                                decoder_layers=decoder_layers,
                                selected_edges=sel,
                                real_pre=real_pre,
                                gray_pre=gray_pre,
                                geom=geom,
                            )
                        elif mode=="all_query":
                            patch_map,_stats=build_all_query_patch(
                                decoder_layers=decoder_layers,
                                selected_edges=sel,
                                real_pre=real_pre,
                                gray_pre=gray_pre,
                                geom=geom,
                                q_all=q_all,
                            )
                        else:
                            raise AssertionError(mode)

                        if not patch_map:
                            continue

                        for alpha in scales:
                            pred,text,pstates=generate_and_capture(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                state_layers=state_layers,
                                max_new_tokens=a.max_new_tokens,
                                attention_patch=patch_map,
                                scale=alpha,
                            )
                            ok=pred==m["gt"]

                            gen_rows.append({
                                "sid":sid,
                                "gt":m["gt"],
                                "condition":mode,
                                "edge_K":K,
                                "alpha":float(alpha),
                                "prediction":pred,
                                "correct":ok,
                                "text":text,
                                "wrong_to_correct":(not bok) and ok,
                                "correct_to_wrong":bok and (not ok),
                            })

                            recon_rows.extend(
                                reconstruction_rows(
                                    sid=sid,
                                    gt=m["gt"],
                                    condition=mode,
                                    edge_k=K,
                                    alpha=float(alpha),
                                    causal_rows=causal_rows,
                                    real_states=real_states,
                                    gray_states=gray_states,
                                    patched_states=pstates,
                                )
                            )

            except Exception as exc:
                append_jsonl(error_path,{
                    "phase":"sample",
                    "sid":sid,
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc().splitlines()[-25:],
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

        edge_df=pd.DataFrame(edge_score_rows)
        edge_df.to_csv(outdir/"edge_scores.csv",index=False)

        sel_df=pd.DataFrame(selected_edge_rows)
        sel_df.to_csv(outdir/"selected_edges.csv",index=False)

        gen_df=pd.DataFrame(gen_rows)
        gen_df.to_csv(outdir/"generation_per_sample.csv",index=False)
        gen_summary=summarize_generation(gen_df)
        gen_summary.to_csv(outdir/"generation_summary.csv",index=False)

        recon_df=pd.DataFrame(recon_rows)
        recon_df.to_csv(outdir/"reconstruction_per_state.csv",index=False)
        recon_summary=summarize_reconstruction(recon_df)
        recon_summary.to_csv(
            outdir/"reconstruction_summary.csv",index=False
        )

        # Head frequency among selected same-token edges.
        if len(sel_df):
            hf=(
                sel_df.groupby(
                    ["edge_K","head_layer","head","head_name","is_direction_head"]
                )
                .size()
                .reset_index(name="selected_count")
                .sort_values(
                    ["edge_K","selected_count"],
                    ascending=[True,False],
                )
            )
        else:
            hf=pd.DataFrame()
        hf.to_csv(outdir/"selected_head_frequency.csv",index=False)

        report=[
            "="*180,
            "HEAD->CAUSAL-TOKEN BINARY RG RECONSTRUCTION",
            "="*180,
            f"model={a.model} repo={spec.repo_id}",
            f"N eval={len(eval_meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"candidate head layers={head_layers}",
            f"candidate heads={'ALL' if candidate_heads is None else len(candidate_heads)}",
            f"edge Ks={edge_ks} scales={scales} modes={modes}",
            "",
            "GENERATION",
            "-"*180,
            gen_summary.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(gen_summary) else "EMPTY",
            "",
            "CAUSAL-STATE RECONSTRUCTION",
            "-"*180,
            recon_summary.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(recon_summary) else "EMPTY",
            "",
            "MOST FREQUENTLY SELECTED HEADS",
            "-"*180,
            hf.groupby("edge_K").head(20).to_string(
                index=False
            ) if len(hf) else "EMPTY",
            "",
            "How to read:",
            "  same_token = only head h at the causal token's own query position p is amplified.",
            "  all_query  = union of selected heads is amplified at every text query position.",
            "  If same_token reconstructs RG well, the head can directly write toward that causal token.",
            "  If all_query >> same_token, indirect routing through other query positions matters.",
            "  Ideal reconstruction: cosine~1, progress~1, norm_ratio~1, relative_error~0.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,encoding="utf-8"
        )

        write_json(outdir/"metadata.json",{
            "script":"eval_head_token_to_causal_rg_reconstruction_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_N":len(eval_meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "head_layers":head_layers,
            "candidate_heads":(
                None if candidate_heads is None
                else sorted([hname(L,h) for L,h in candidate_heads])
            ),
            "edge_ks":edge_ks,
            "scales":scales,
            "modes":modes,
            "causal_scale":a.causal_scale,
            "edge_score_same_token":(
                "delta_z[L,h,p]^T grad_z <h[C,p], normalize(h_real-h_gray)>"
            ),
            "same_token_intervention":(
                "add shared alpha * W_O^h(z_real-z_gray)[L,h,p] "
                "only at causal query position p"
            ),
            "all_query_control":(
                "selected head union amplified at all selected query positions"
            ),
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
