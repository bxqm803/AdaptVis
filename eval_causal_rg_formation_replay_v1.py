#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_causal_rg_formation_replay_v1.py

Question
========
We already know that directly amplifying selected causal text states by their
sample-specific Image-Gray displacement is behaviorally useful:

    target(C,p) = h_real[C,p] - h_gray[C,p]

    direct causal-RG:
        h'[C,p] = h_real[C,p] + alpha * target(C,p)

The formation analysis showed that this target is accumulated through residual
blocks from actual Real-Gray changes in attention and MLP outputs.

This script asks the causal replay question:

    Can replaying those actual formation components reproduce direct causal-RG?

For every selected causal state c=(C,p) and every upstream block L<=C:

    dA[L,p] = AttnOut_real[L,p] - AttnOut_gray[L,p]
    dM[L,p] = MLPOut_real[L,p]  - MLPOut_gray[L,p]

During a fresh REAL-image generation, replay one extra copy at the SAME token:

    attention replay:
        AttnOut'[L,p] = AttnOut[L,p] + alpha * dA[L,p]

    MLP replay:
        MLPOut'[L,p]  = MLPOut[L,p]  + alpha * dM[L,p]

    both replay:
        apply both edits at that block/token.

This is not a gradient approximation. dA and dM are the actual observed
sample-specific Real-Gray formation components.

Layer selection
===============
For each concrete causal state (C,p), use its full causal-RG target t:

    t = h_real[C,p] - h_gray[C,p]

and score every L<=C:

    attention_progress(L) = <dA[L,p], t> / ||t||^2
    mlp_progress(L)       = <dM[L,p], t> / ||t||^2
    update_progress(L)    = <dA[L,p]+dM[L,p], t> / ||t||^2

For numeric --layer-ks, select Top-K POSITIVE layers independently for each
causal state:
    attn condition -> rank by attention_progress
    mlp condition  -> rank by mlp_progress
    both condition -> rank by update_progress

For K=all:
    replay EVERY eligible L<=C, including negative/cancelling components.
This is the strongest "replay the observed formation trajectory" condition.

If multiple selected causal states request the same module/layer/token patch,
the physical patch is applied ONCE, not double-counted.

References
==========
baseline:
    untouched REAL generation

causal_rg:
    direct block-output causal-state amplification
        h[C,p] += causal_scale * (h_real-h_gray)[C,p]

Replay conditions:
    attn
    mlp
    both

Reconstruction metrics
======================
After each intervention, for every selected causal state:

    move = h_patched[C,p] - h_real[C,p]
    t    = h_real[C,p] - h_gray[C,p]

Report:

    cosine     = cos(move,t)
    progress   = <move,t>/||t||^2
    norm_ratio = ||move||/||t||
    rel_error  = ||move-t||/||t||

Ideal reproduction of direct causal-RG:
    cosine -> 1
    progress -> 1
    norm_ratio -> 1
    rel_error -> 0

Also compare actual greedy generation:
    baseline vs causal_rg vs attn/mlp/both replay.

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_causal_rg_formation_replay_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --trace-layers 0-26 \
  --conditions attn,mlp,both \
  --layer-ks 1,2,4,all \
  --scales 1 \
  --eval-max-samples 20 \
  --output-dir output/qwen3b_causal_rg_formation_replay_n20_v1 \
  --overwrite

Then N=80:
    --eval-max-samples 80

Primary outputs
===============
component_scores.csv
selected_components.csv
generation_per_sample.csv
generation_summary.csv
reconstruction_per_state.csv
reconstruction_summary.csv
condition_component_stats.csv
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
    p.add_argument("--trace-layers", default="0-26")

    p.add_argument(
        "--conditions",
        default="attn,mlp,both",
        help="Comma-separated subset of attn,mlp,both.",
    )
    p.add_argument(
        "--layer-ks",
        default="1,2,4,all",
        help="Per-causal-state Top-K positive formation layers; 'all' replays all eligible layers.",
    )
    p.add_argument(
        "--scales",
        default="1",
        help="Shared replay alpha(s). alpha=1 is one extra copy of actual Real-Gray component.",
    )
    p.add_argument("--causal-scale", type=float, default=1.0)

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
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
    out=set()
    for part in str(text).split(","):
        part=part.strip().upper().replace("L","")
        if not part:
            continue
        if "-" in part:
            a,b=part.split("-",1)
            a,b=int(a),int(b)
            out.update(range(min(a,b),max(a,b)+1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_conditions(text):
    out=[x.strip() for x in str(text).split(",") if x.strip()]
    valid={"attn","mlp","both"}
    bad=set(out)-valid
    if bad:
        raise ValueError(f"Unknown conditions {sorted(bad)}")
    return out


def parse_layer_ks(text):
    out=[]
    for x in str(text).split(","):
        x=x.strip().lower()
        if not x:
            continue
        if x=="all":
            out.append("all")
        else:
            k=int(x)
            if k<=0:
                raise ValueError("numeric layer K must be >0")
            out.append(k)
    return out


def parse_floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def klabel(k):
    return "all" if k=="all" else str(int(k))


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


def progress(v,t):
    v=np.asarray(v,dtype=np.float32)
    t=np.asarray(t,dtype=np.float32)
    den=float(np.dot(t,t))
    if den<=EPS:
        return float("nan")
    return float(np.dot(v,t)/den)


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

def resolve_mlp(layer):
    for name in ("mlp","feed_forward","ffn"):
        x=getattr(layer,name,None)
        if isinstance(x,torch.nn.Module):
            return x
    raise RuntimeError(f"Could not resolve MLP on {type(layer).__name__}")


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
    raise RuntimeError(f"No tensor in {type(x).__name__}")


class ComponentCapture:
    """
    Capture only required positions:
      attn[L]  : self-attention output after W_O, before residual add
      mlp[L]   : MLP output before residual add
      block[L] : decoder block output
    """
    def __init__(self,decoder_layers,layers,positions):
        self.layers=list(map(int,layers))
        self.positions=sorted(set(map(int,positions)))
        self.attn={}
        self.mlp={}
        self.block={}
        self.handles=[]

        def take(x):
            if x.ndim!=3 or x.shape[0]!=1:
                raise RuntimeError(f"Expected [1,S,D], got {tuple(x.shape)}")
            good=[p for p in self.positions if 0<=p<int(x.shape[1])]
            if len(good)!=len(self.positions):
                bad=sorted(set(self.positions)-set(good))
                raise RuntimeError(f"Bad positions {bad}; S={x.shape[1]}")
            idx=torch.as_tensor(good,device=x.device,dtype=torch.long)
            return (
                x[0].index_select(0,idx)
                .detach().float().cpu().numpy().astype(np.float32)
            )

        for L in self.layers:
            block=decoder_layers[L]
            attn=spatialscan.resolve_attn(block)
            mlp=resolve_mlp(block)

            def make_attn(layer):
                def hook(_m,_inp,out):
                    self.attn[layer]=take(first_tensor_any(out))
                    return None
                return hook

            def make_mlp(layer):
                def hook(_m,_inp,out):
                    self.mlp[layer]=take(first_tensor_any(out))
                    return None
                return hook

            def make_block(layer):
                def hook(_m,_inp,out):
                    self.block[layer]=take(traj.first_tensor(out))
                    return None
                return hook

            self.handles.append(attn.register_forward_hook(make_attn(L)))
            self.handles.append(mlp.register_forward_hook(make_mlp(L)))
            self.handles.append(block.register_forward_hook(make_block(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def run_component_capture(model,decoder_layers,batch,layers,positions):
    cap=ComponentCapture(decoder_layers,layers,positions)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        _=model(**kw)

        missing=[]
        for L in layers:
            for name,d in [("attn",cap.attn),("mlp",cap.mlp),("block",cap.block)]:
                if L not in d:
                    missing.append((L,name))
        if missing:
            raise RuntimeError(f"Missing capture {missing[:20]}")

        return {
            "attn":cap.attn,
            "mlp":cap.mlp,
            "block":cap.block,
            "positions":list(cap.positions),
        }
    finally:
        cap.close()


def pos_index(cap,p):
    return cap["positions"].index(int(p))


# =============================================================================
# Component score / selection
# =============================================================================

def build_component_scores(sid,gt,causal_rows,real_cap,gray_cap,trace_layers):
    rows=[]

    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)
        pi=pos_index(real_cap,p)

        t=(
            real_cap["block"][C][pi]
            - gray_cap["block"][C][pi]
        ).astype(np.float32)
        tn=float(np.linalg.norm(t))
        if tn<=EPS:
            continue

        for L in trace_layers:
            if L>C:
                continue

            da=(
                real_cap["attn"][L][pi]
                - gray_cap["attn"][L][pi]
            ).astype(np.float32)
            dm=(
                real_cap["mlp"][L][pi]
                - gray_cap["mlp"][L][pi]
            ).astype(np.float32)
            du=(da+dm).astype(np.float32)

            rows.append({
                "sid":sid,
                "gt":gt,
                "causal_rank":int(r.rank),
                "causal_text_rank":int(r.causal_text_rank),
                "causal_layer":C,
                "causal_position":p,
                "causal_token":str(r.token),
                "causal_category":str(r.broad_category),
                "trace_layer":L,
                "distance_to_causal":C-L,
                "target_rg_norm":tn,

                "attn_rg_norm":float(np.linalg.norm(da)),
                "mlp_rg_norm":float(np.linalg.norm(dm)),
                "update_rg_norm":float(np.linalg.norm(du)),

                "attn_cos_target":cosine(da,t),
                "mlp_cos_target":cosine(dm,t),
                "update_cos_target":cosine(du,t),

                "attn_progress":progress(da,t),
                "mlp_progress":progress(dm,t),
                "update_progress":progress(du,t),
            })

    return pd.DataFrame(rows)


def select_component_rows(score_df,condition,K):
    if condition=="attn":
        col="attn_progress"
    elif condition=="mlp":
        col="mlp_progress"
    elif condition=="both":
        col="update_progress"
    else:
        raise ValueError(condition)

    selected=[]

    groups=[
        "sid","causal_layer","causal_position","causal_text_rank"
    ]
    for _,g in score_df.groupby(groups,sort=False):
        if K=="all":
            z=g.sort_values("trace_layer").copy()
        else:
            z=g[g[col]>0].sort_values(
                col,ascending=False
            ).head(int(K)).copy()

        if len(z):
            z["condition"]=condition
            z["layer_K"]=klabel(K)
            z["selection_score"]=z[col]
            z["selection_rank"]=np.arange(1,len(z)+1)
            selected.append(z)

    if not selected:
        return score_df.iloc[:0].copy()

    return pd.concat(selected,ignore_index=True)


def build_patch_maps(selected,condition,real_cap,gray_cap):
    """
    Returns physical module patches, deduplicated by (L,p).
    If multiple downstream target causal states selected the same physical edge,
    it is applied once.
    """
    attn_map=defaultdict(dict)
    mlp_map=defaultdict(dict)

    physical=selected[
        ["trace_layer","causal_position"]
    ].drop_duplicates()

    stats=[]

    for r in physical.itertuples():
        L=int(r.trace_layer)
        p=int(r.causal_position)
        pi=pos_index(real_cap,p)

        da=(
            real_cap["attn"][L][pi]
            - gray_cap["attn"][L][pi]
        ).astype(np.float32)
        dm=(
            real_cap["mlp"][L][pi]
            - gray_cap["mlp"][L][pi]
        ).astype(np.float32)

        if condition in ("attn","both"):
            attn_map[L][p]=da
        if condition in ("mlp","both"):
            mlp_map[L][p]=dm

        stats.append({
            "condition":condition,
            "trace_layer":L,
            "position":p,
            "attn_rg_norm":float(np.linalg.norm(da)),
            "mlp_rg_norm":float(np.linalg.norm(dm)),
        })

    return (
        {L:dict(v) for L,v in attn_map.items()},
        {L:dict(v) for L,v in mlp_map.items()},
        stats,
    )


def build_causal_patch(real_cap,gray_cap,causal_rows):
    out=defaultdict(dict)
    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)
        pi=pos_index(real_cap,p)
        out[C][p]=(
            real_cap["block"][C][pi]
            - gray_cap["block"][C][pi]
        ).astype(np.float32)
    return {L:dict(v) for L,v in out.items()}


# =============================================================================
# Intervention hooks
# =============================================================================

def first_3d(out):
    if torch.is_tensor(out) and out.ndim==3:
        return out
    if isinstance(out,(tuple,list)):
        for x in out:
            if torch.is_tensor(x) and x.ndim==3:
                return x
    raise RuntimeError("No 3D tensor found")


def replace_first_3d(out,y):
    if torch.is_tensor(out):
        return y
    if isinstance(out,tuple):
        xs=list(out)
        for i,x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim==3:
                xs[i]=y
                return tuple(xs)
    if isinstance(out,list):
        xs=list(out)
        for i,x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim==3:
                xs[i]=y
                return xs
    raise RuntimeError("Could not replace 3D tensor")


class ModuleOutputPatch:
    def __init__(self,modules_by_layer,patch_map,prompt_len,scale):
        self.modules_by_layer=modules_by_layer
        self.patch_map=patch_map
        self.prompt_len=int(prompt_len)
        self.scale=float(scale)
        self.handles=[]

    def __enter__(self):
        for L,by_pos in self.patch_map.items():
            mod=self.modules_by_layer[int(L)]

            def make_hook(pos_map):
                def hook(_m,_inp,out):
                    x=first_3d(out)
                    if int(x.shape[1])!=self.prompt_len:
                        return None

                    y=x.clone()
                    for p,vec in pos_map.items():
                        p=int(p)
                        if 0<=p<int(y.shape[1]):
                            y[0,p]+=self.scale*torch.as_tensor(
                                vec,device=y.device,dtype=y.dtype
                            )
                    return replace_first_3d(out,y)
                return hook

            self.handles.append(
                mod.register_forward_hook(make_hook(dict(by_pos)))
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
        self.prompt_len=int(prompt_len)
        self.handles=[]

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
    attn_patch=None,
    mlp_patch=None,
    block_patch=None,
    scale=1.0,
):
    prompt_len=int(batch["input_ids"].shape[1])

    attn_modules={
        L:spatialscan.resolve_attn(decoder_layers[L])
        for L in (attn_patch or {})
    }
    mlp_modules={
        L:resolve_mlp(decoder_layers[L])
        for L in (mlp_patch or {})
    }

    attn_ctx=(
        ModuleOutputPatch(attn_modules,attn_patch,prompt_len,scale)
        if attn_patch else contextlib.nullcontext()
    )
    mlp_ctx=(
        ModuleOutputPatch(mlp_modules,mlp_patch,prompt_len,scale)
        if mlp_patch else contextlib.nullcontext()
    )
    block_ctx=(
        BlockOutputPatch(decoder_layers,block_patch,prompt_len,scale)
        if block_patch else contextlib.nullcontext()
    )

    # Patches first, then state capture, so captured block outputs reflect edits.
    with attn_ctx:
        with mlp_ctx:
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
# Metrics
# =============================================================================

def reconstruction_rows(
    *,
    sid,
    gt,
    condition,
    layer_K,
    alpha,
    causal_rows,
    real_cap,
    gray_cap,
    patched_states,
):
    rows=[]

    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)

        if C not in patched_states:
            continue

        pi=pos_index(real_cap,p)
        if not (0<=p<patched_states[C].shape[1]):
            continue

        target=(
            real_cap["block"][C][pi]
            - gray_cap["block"][C][pi]
        ).astype(np.float32)
        move=(
            patched_states[C][0,p]
            - real_cap["block"][C][pi]
        ).astype(np.float32)

        tn=float(np.linalg.norm(target))
        mn=float(np.linalg.norm(move))
        if tn<=EPS:
            continue

        dot=float(np.dot(move,target))

        rows.append({
            "sid":sid,
            "gt":gt,
            "condition":condition,
            "layer_K":str(layer_K),
            "alpha":float(alpha),
            "causal_rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "causal_layer":C,
            "causal_position":p,
            "causal_token":str(r.token),
            "causal_category":str(r.broad_category),
            "target_rg_norm":tn,
            "move_norm":mn,
            "cos_move_vs_rg":dot/(max(mn,EPS)*tn),
            "rg_progress":dot/(tn*tn),
            "move_over_rg_norm":mn/tn,
            "relative_reconstruction_error":float(
                np.linalg.norm(move-target)/tn
            ),
        })

    return rows


def summarize_generation(df):
    if len(df)==0:
        return pd.DataFrame()

    b=df[df.condition=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]

    for (cond,K,a),g in df[df.condition!="baseline"].groupby(
        ["condition","layer_K","alpha"],dropna=False
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
            "layer_K":K,
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

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["condition","layer_K","alpha"]
    ).reset_index(drop=True)


def summarize_reconstruction(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (cond,K,a),g in df.groupby(
        ["condition","layer_K","alpha"],dropna=False
    ):
        rows.append({
            "condition":cond,
            "layer_K":K,
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
        ["condition","layer_K","alpha"]
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
    trace_layers=parse_layers(a.trace_layers)
    conditions=parse_conditions(a.conditions)
    layer_ks=parse_layer_ks(a.layer_ks)
    scales=parse_floats(a.scales)

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

    all_scores=[]
    all_selected=[]
    gen_rows=[]
    recon_rows=[]
    component_stat_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        n_layers=len(decoder_layers)
        trace_layers=[
            L for L in trace_layers
            if 0<=L<n_layers and L<=max(causal_layers)
        ]

        for m in tqdm(eval_meta,desc="FORMATION REPLAY"):
            sid=int(m["sid"])
            if sid not in causal_by_sid:
                continue

            real=gray=None

            try:
                causal_rows=causal_by_sid[sid]
                positions=sorted(set(map(int,causal_rows.position.tolist())))
                max_C=int(causal_rows.source_layer.max())
                layers_sid=[L for L in trace_layers if L<=max_C]

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
                    raise RuntimeError("REAL/GRAY tokenization mismatch")

                rc=run_component_capture(
                    model,decoder_layers,rb,layers_sid,positions
                )
                gc_=run_component_capture(
                    model,decoder_layers,gb,layers_sid,positions
                )

                scores=build_component_scores(
                    sid,m["gt"],causal_rows,rc,gc_,layers_sid
                )
                if len(scores)==0:
                    continue
                all_scores.extend(scores.to_dict("records"))

                state_layers=sorted(set(map(int,causal_rows.source_layer)))

                # Baseline.
                bpred,btext,_bstates=generate_and_capture(
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
                    "layer_K":"0",
                    "alpha":0.0,
                    "prediction":bpred,
                    "correct":bok,
                    "text":btext,
                })

                # Direct causal-RG reference.
                causal_patch=build_causal_patch(rc,gc_,causal_rows)
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
                    "layer_K":"0",
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
                        layer_K="0",
                        alpha=float(a.causal_scale),
                        causal_rows=causal_rows,
                        real_cap=rc,
                        gray_cap=gc_,
                        patched_states=cstates,
                    )
                )

                # Replay formation components.
                for condition in conditions:
                    for K in layer_ks:
                        sel=select_component_rows(scores,condition,K)
                        if len(sel)==0:
                            continue

                        all_selected.extend(sel.to_dict("records"))

                        attn_map,mlp_map,stats=build_patch_maps(
                            sel,condition,rc,gc_
                        )
                        for s in stats:
                            s.update({
                                "sid":sid,
                                "gt":m["gt"],
                                "layer_K":klabel(K),
                            })
                            component_stat_rows.append(s)

                        for alpha in scales:
                            pred,text,pstates=generate_and_capture(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                state_layers=state_layers,
                                max_new_tokens=a.max_new_tokens,
                                attn_patch=attn_map,
                                mlp_patch=mlp_map,
                                scale=alpha,
                            )
                            ok=pred==m["gt"]

                            gen_rows.append({
                                "sid":sid,
                                "gt":m["gt"],
                                "condition":condition,
                                "layer_K":klabel(K),
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
                                    condition=condition,
                                    layer_K=klabel(K),
                                    alpha=float(alpha),
                                    causal_rows=causal_rows,
                                    real_cap=rc,
                                    gray_cap=gc_,
                                    patched_states=pstates,
                                )
                            )

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

        score_df=pd.DataFrame(all_scores)
        sel_df=pd.DataFrame(all_selected)
        gen_df=pd.DataFrame(gen_rows)
        recon_df=pd.DataFrame(recon_rows)
        stat_df=pd.DataFrame(component_stat_rows)

        score_df.to_csv(outdir/"component_scores.csv",index=False)
        sel_df.to_csv(outdir/"selected_components.csv",index=False)
        gen_df.to_csv(outdir/"generation_per_sample.csv",index=False)
        recon_df.to_csv(outdir/"reconstruction_per_state.csv",index=False)
        stat_df.to_csv(outdir/"condition_component_stats.csv",index=False)

        gs=summarize_generation(gen_df)
        rs=summarize_reconstruction(recon_df)

        gs.to_csv(outdir/"generation_summary.csv",index=False)
        rs.to_csv(outdir/"reconstruction_summary.csv",index=False)

        # Aggregate which layers get selected most.
        if len(sel_df):
            sel_layer=(
                sel_df.groupby(
                    ["condition","layer_K","trace_layer"]
                )
                .agg(
                    selected_count=("sid","size"),
                    N_samples=("sid","nunique"),
                    mean_selection_score=("selection_score","mean"),
                )
                .reset_index()
                .sort_values(
                    ["condition","layer_K","selected_count"],
                    ascending=[True,True,False],
                )
            )
        else:
            sel_layer=pd.DataFrame()
        sel_layer.to_csv(
            outdir/"selected_layer_frequency.csv",index=False
        )

        report=[
            "="*190,
            "CAUSAL-RG FORMATION COMPONENT REPLAY",
            "="*190,
            f"model={a.model} repo={spec.repo_id}",
            f"N eval={len(eval_meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"trace layers={trace_layers}",
            f"conditions={conditions} layer_ks={[klabel(k) for k in layer_ks]} scales={scales}",
            "",
            "GENERATION",
            "-"*190,
            gs.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(gs) else "EMPTY",
            "",
            "CAUSAL-STATE RECONSTRUCTION",
            "-"*190,
            rs.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(rs) else "EMPTY",
            "",
            "MOST FREQUENTLY SELECTED FORMATION LAYERS",
            "-"*190,
            sel_layer.groupby(
                ["condition","layer_K"],group_keys=False
            ).head(10).to_string(
                index=False,float_format=lambda x:f"{x:.5f}"
            ) if len(sel_layer) else "EMPTY",
            "",
            "How to read:",
            "  causal_rg is the direct h[C,p] += (h_real-h_gray)[C,p] reference.",
            "  attn replays actual dAttention at selected same-token layers.",
            "  mlp replays actual dMLP at selected same-token layers.",
            "  both replays both actual components.",
            "  K=all includes every eligible formation layer, including cancelling components.",
            "  If both/all alpha=1 approaches causal_rg in cosine/progress/error and generation,",
            "  the useful causal-RG can be reproduced by replaying its observed block formation updates.",
            "  If MLP >> attention, the previous head-only failure is explained by substantial MLP formation.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,encoding="utf-8"
        )

        write_json(outdir/"metadata.json",{
            "script":"eval_causal_rg_formation_replay_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_N":len(eval_meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "trace_layers":trace_layers,
            "conditions":conditions,
            "layer_ks":[klabel(k) for k in layer_ks],
            "scales":scales,
            "causal_scale":a.causal_scale,
            "definitions":{
                "target":"h_real[C,p]-h_gray[C,p]",
                "attn_component":"AttnOut_real[L,p]-AttnOut_gray[L,p]",
                "mlp_component":"MLPOut_real[L,p]-MLPOut_gray[L,p]",
                "selection_progress":"dot(component,target)/||target||^2",
                "all":"all L<=C, including negative/cancelling components",
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
