#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_layerwise_rn_transport_to_causal_tokens_v1.py

Question
========
For a behaviorally causal text state c=(C,p), we already know that

    t_C = h_real[C,p] - h_noimage[C,q]

is a useful causal-state Real-NoImage (RN) direction.

This script uses that final causal-token RN direction as a PROBE and asks:

    At every earlier layer L, the SAME text token p has its own RN displacement

        d_L = h_real[L,p] - h_noimage[L,q].

    If we amplify that token at layer L by its own current RN displacement,

        h'[L,p] = h_real[L,p] + alpha * d_L,

    how much does the downstream causal state at layer C move along its final
    RN direction t_C?

This directly measures where RN evidence becomes "transportable" into the
behaviorally causal state.

Two complementary measurements
===============================

(1) Natural RN formation trajectory
-----------------------------------
Without intervention, for every L <= C:

    natural_projection(L->C)
      = <d_L, t_C> / ||t_C||^2

    natural_cosine(L->C)
      = cos(d_L, t_C)

    natural_norm_ratio(L->C)
      = ||d_L|| / ||t_C||

This shows when the SAME token's RN state naturally becomes aligned with the
eventual causal-token RN direction.

(2) Causal transport by layerwise RN amplification
---------------------------------------------------
Patch ONLY the same token p at layer L:

    h'[L,p] = h[L,p] + injection

Natural-scale mode:
    injection = alpha * d_L

Optional norm-matched mode:
    injection = alpha * ||t_C||/||d_L|| * d_L

Then let all later layers recompute naturally and measure:

    move_C = h_patched[C,p] - h_real[C,p]

    final_progress(L->C)
      = <move_C, t_C> / ||t_C||^2

    final_move_cosine
      = cos(move_C, t_C)

    final_move_norm_ratio
      = ||move_C|| / ||t_C||

Interpretation
==============
If d_L is already present but final_progress is tiny:
    RN information is present at L but not yet in a form downstream computation
    can efficiently turn into the final causal state.

If final_progress suddenly becomes large around some layer:
    that layer is a candidate "transport / utilization transition" and a
    promising intervention stage.

Example:
    natural_projection:
        L18 .30, L20 .45, L23 .62
    final_progress after +d_L:
        L18 .03, L20 .07, L23 .38

Then RN information exists before L23, but becomes much more behaviorally
transportable around L23.

Target selection
================
Causal target states come from the existing oracle causal-ranking CSV.
Therefore this script is a MECHANISM diagnostic, not a non-oracle method.

The intervention itself never uses relation-specific vectors. It uses each
target token's own sample-specific Real-NoImage difference.

Patch granularity
=================
Default: isolated
    Each selected causal target is patched separately. This is the cleanest
    answer to "for THIS causal token, which layer transports its RN evidence?"

Optional: joint
    At a given probe layer, patch all selected causal-token positions together.
    Faster and more method-like, but effects can interact.

Recommended quick run
=====================
Start small because isolated mode needs one forward per target x probe layer:

CUDA_VISIBLE_DEVICES=0 python -u probe_layerwise_rn_transport_to_causal_tokens_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 3 \
  --probe-layers 8-26 \
  --modes natural \
  --alphas 1 \
  --patch-granularity isolated \
  --eval-max-samples 10 \
  --output-dir output/qwen3b_layerwise_rn_transport_n10_v1 \
  --overwrite

Optional mechanism control with equalized injection norm:
    --modes natural,normmatched

For the strongest layer(s), then test actual generation separately.

Outputs
=======
selected_causal_states.csv
alignment_summary.csv
baseline_generation.csv
natural_trajectory_per_target.csv
natural_trajectory_summary.csv
intervention_per_target.csv
layer_transport_summary.csv
layer_transport_by_baseline.csv
layer_transport_by_relation.csv
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
EPS = 1e-12


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
    p.add_argument("--causal-top-k", type=int, default=3)
    p.add_argument(
        "--causal-categories",
        default="",
        help="Optional broad_category filter. Empty = all nonvisual/nonlast causal text states.",
    )

    p.add_argument(
        "--probe-layers",
        default="8-26",
        help="Layer outputs whose same-token RN displacement will be tested.",
    )
    p.add_argument(
        "--modes",
        default="natural",
        help="Comma-separated intervention modes: natural,normmatched.",
    )
    p.add_argument(
        "--alphas",
        default="1",
        help="Comma-separated amplification coefficients.",
    )
    p.add_argument(
        "--patch-granularity",
        default="isolated",
        choices=["isolated","joint"],
        help=(
            "isolated = patch each causal target separately; "
            "joint = patch all selected causal token positions at a probe layer together."
        ),
    )
    p.add_argument(
        "--normmatch-cap",
        type=float,
        default=10.0,
        help="Maximum scale ||t_C||/||d_L|| in normmatched mode.",
    )

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--skip-baseline-generation",
        action="store_true",
        help="Skip one greedy baseline generation per sample.",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=10)

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


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_set(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


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


def cosine_np(a,b):
    a=np.asarray(a,np.float32)
    b=np.asarray(b,np.float32)
    na=float(np.linalg.norm(a))
    nb=float(np.linalg.norm(b))
    if na<=EPS or nb<=EPS:
        return float("nan")
    return float(np.dot(a,b)/(na*nb))


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
    if a.eval_max_samples>0:
        meta=traj.stratified_cap(
            meta,a.eval_max_samples,a.seed+1
        )
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


def load_causal_selection(
    path: Path,
    allowed_sids: set,
    causal_layers: Sequence[int],
    top_k: int,
    categories: Sequence[str],
):
    d=pd.read_csv(path)
    required={
        "sid","rank","source_layer","position",
        "token","category","broad_category",
    }
    missing=required-set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")

    for c in ("sid","rank","source_layer","position"):
        d[c]=pd.to_numeric(d[c],errors="raise").astype(int)

    d=d[d["sid"].isin(allowed_sids)].copy()
    d=d[d["source_layer"].isin(set(map(int,causal_layers)))].copy()
    d=d[d["broad_category"].astype(str)!="visual"].copy()
    d=d[d["broad_category"].astype(str)!="last"].copy()

    if categories:
        wanted=set(map(str,categories))
        d=d[d["broad_category"].astype(str).isin(wanted)].copy()

    rows=[]
    for sid,g in d.groupby("sid"):
        z=g.sort_values("rank").head(int(top_k)).copy()
        z["causal_text_rank"]=np.arange(1,len(z)+1)
        rows.append(z)

    return pd.concat(rows,ignore_index=True) if rows else d.iloc[:0].copy()


# =============================================================================
# NoImage / alignment
# =============================================================================

def move_batch(batch,device):
    return {
        k:(v.to(device) if torch.is_tensor(v) else v)
        for k,v in batch.items()
    }


def build_noimage_batch(processor,question_text,device):
    messages=[
        {
            "role":"user",
            "content":[{"type":"text","text":str(question_text)}],
        }
    ]
    try:
        prompt=processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt=str(question_text)

    last_error=None
    for fn in [
        lambda: processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            return_tensors="pt",
        ),
    ]:
        try:
            return move_batch(fn(),device)
        except Exception as exc:
            last_error=exc

    raise RuntimeError(f"NoImage processor failed: {last_error}")


def lcs_token_map(real_ids: List[int],no_ids: List[int]) -> Dict[int,int]:
    a=list(map(int,real_ids))
    b=list(map(int,no_ids))
    n,m=len(a),len(b)

    dp=np.zeros((n+1,m+1),dtype=np.uint16)
    for i in range(n-1,-1,-1):
        ai=a[i]
        row=dp[i]
        below=dp[i+1]
        for j in range(m-1,-1,-1):
            if ai==b[j]:
                row[j]=1+below[j+1]
            else:
                x=below[j]
                y=row[j+1]
                row[j]=x if x>=y else y

    out={}
    i=j=0
    while i<n and j<m:
        if a[i]==b[j] and dp[i,j]==1+dp[i+1,j+1]:
            out[i]=j
            i+=1
            j+=1
        elif dp[i+1,j]>=dp[i,j+1]:
            i+=1
        else:
            j+=1
    return out


# =============================================================================
# Block-state capture and patch
# =============================================================================

def tensor_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output,(tuple,list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported layer output type: {type(output).__name__}")


def replace_tensor_output(original_output,tensor):
    if torch.is_tensor(original_output):
        return tensor
    if isinstance(original_output,tuple):
        return (tensor,*original_output[1:])
    if isinstance(original_output,list):
        return [tensor,*original_output[1:]]
    raise RuntimeError(f"Unsupported layer output type: {type(original_output).__name__}")


class BlockCapture:
    def __init__(self,decoder_layers,layers):
        self.states={}
        self.handles=[]

        for L in sorted(set(map(int,layers))):
            def make_hook(layer):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)
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
def capture_blocks(model,decoder_layers,batch,layers):
    cap=BlockCapture(decoder_layers,layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        _=model(**kw)

        missing=[L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing block captures: {missing}")

        return dict(cap.states)
    finally:
        cap.close()


class ResidualStatePatch:
    """
    Patch decoder block OUTPUT at one probe layer.
    patch_map: real_position -> numpy vector to ADD.
    """
    def __init__(self,decoder_layers,layer,patch_map):
        self.handle=None
        self.layer=int(layer)
        self.patch_map=dict(patch_map)

        def hook(_m,_inp,out):
            x=tensor_from_output(out)
            y=x.clone()
            for p,vec in self.patch_map.items():
                p=int(p)
                if 0<=p<int(y.shape[1]):
                    y[0,p]+=torch.as_tensor(
                        vec,
                        device=y.device,
                        dtype=y.dtype,
                    )
            return replace_tensor_output(out,y)

        self.handle=decoder_layers[self.layer].register_forward_hook(hook)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
            self.handle=None


@torch.inference_mode()
def run_patched_to_targets(
    *,
    model,
    decoder_layers,
    real_batch,
    patch_layer,
    patch_map,
    target_layers,
):
    """
    Patch block-output at patch_layer and capture downstream target block outputs.
    Requires target layers > patch_layer. For L==C the caller computes analytically.
    """
    with ResidualStatePatch(
        decoder_layers,
        patch_layer,
        patch_map,
    ):
        cap=BlockCapture(decoder_layers,target_layers)
        try:
            kw=dict(real_batch)
            kw["use_cache"]=False
            kw["return_dict"]=True
            _=model(**kw)

            missing=[L for L in target_layers if L not in cap.states]
            if missing:
                raise RuntimeError(f"Missing patched target states: {missing}")
            return dict(cap.states)
        finally:
            cap.close()


# =============================================================================
# Target preparation
# =============================================================================

def prepare_targets(
    *,
    causal_rows,
    r2n,
    real_states,
    no_states,
    probe_layers,
):
    targets=[]

    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)
        q=r2n.get(p,None)
        if q is None:
            continue
        if C not in real_states or C not in no_states:
            continue

        R=real_states[C]
        N=no_states[C]
        if not (0<=p<R.shape[1] and 0<=q<N.shape[1]):
            continue

        hR_C=R[0,p].astype(np.float32)
        hN_C=N[0,q].astype(np.float32)
        target=(hR_C-hN_C).astype(np.float32)
        target_norm=float(np.linalg.norm(target))
        if target_norm<=EPS:
            continue

        valid_probe=[
            int(L)
            for L in probe_layers
            if int(L)<=C
            and int(L) in real_states
            and int(L) in no_states
            and 0<=p<real_states[int(L)].shape[1]
            and 0<=q<no_states[int(L)].shape[1]
        ]
        if not valid_probe:
            continue

        targets.append({
            "sid":int(r.sid),
            "rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "target_layer":C,
            "real_position":p,
            "noimage_position":int(q),
            "token":str(r.token),
            "category":str(r.category),
            "broad_category":str(r.broad_category),
            "target":target,
            "target_norm":target_norm,
            "hR_C":hR_C,
            "hN_C":hN_C,
            "probe_layers":valid_probe,
        })

    return targets


def source_delta_for_target(t,layer,real_states,no_states):
    L=int(layer)
    p=int(t["real_position"])
    q=int(t["noimage_position"])
    return (
        real_states[L][0,p].astype(np.float32)
        -
        no_states[L][0,q].astype(np.float32)
    )


# =============================================================================
# Natural trajectory
# =============================================================================

def natural_rows_for_target(
    *,
    sid,
    gt,
    baseline_correct,
    target,
    real_states,
    no_states,
):
    rows=[]
    t=target["target"]
    tn=float(target["target_norm"])

    previous_projection=None

    for L in sorted(target["probe_layers"]):
        d=source_delta_for_target(
            target,L,real_states,no_states
        )
        dn=float(np.linalg.norm(d))

        projection=float(np.dot(d,t)/(tn*tn))
        cosine=cosine_np(d,t)
        norm_ratio=dn/tn

        increment=(
            float("nan")
            if previous_projection is None
            else projection-previous_projection
        )
        previous_projection=projection

        rows.append({
            "sid":int(sid),
            "gt":str(gt),
            "baseline_correct":baseline_correct,
            "target_rank":int(target["rank"]),
            "causal_text_rank":int(target["causal_text_rank"]),
            "target_layer":int(target["target_layer"]),
            "probe_layer":int(L),
            "distance_to_target":int(target["target_layer"]-L),
            "real_position":int(target["real_position"]),
            "noimage_position":int(target["noimage_position"]),
            "token":str(target["token"]),
            "category":str(target["category"]),
            "broad_category":str(target["broad_category"]),
            "target_rn_norm":tn,
            "probe_rn_norm":dn,
            "natural_projection":projection,
            "natural_cosine":cosine,
            "natural_norm_ratio":norm_ratio,
            "incremental_projection_gain":increment,
        })

    return rows


# =============================================================================
# Intervention metrics
# =============================================================================

def build_injection(delta,target_norm,mode,alpha,normmatch_cap):
    d=np.asarray(delta,np.float32)
    dn=float(np.linalg.norm(d))
    if dn<=EPS:
        return None,float("nan"),float("nan")

    if mode=="natural":
        scale=float(alpha)
    elif mode=="normmatched":
        scale=float(alpha)*(float(target_norm)/dn)
        if normmatch_cap>0:
            scale=min(scale,float(normmatch_cap))
    else:
        raise ValueError(mode)

    inj=(scale*d).astype(np.float32)
    input_norm_ratio=float(np.linalg.norm(inj))/max(float(target_norm),EPS)
    return inj,scale,input_norm_ratio


def intervention_metric_row(
    *,
    sid,
    gt,
    baseline_correct,
    target,
    probe_layer,
    mode,
    alpha,
    injection_scale,
    input_norm_ratio,
    patched_target_state,
):
    t=target["target"]
    tn=float(target["target_norm"])
    hR=target["hR_C"]
    hN=target["hN_C"]
    hp=np.asarray(patched_target_state,np.float32)

    move=(hp-hR).astype(np.float32)
    final_rn=(hp-hN).astype(np.float32)

    progress=float(np.dot(move,t)/(tn*tn))
    move_norm=float(np.linalg.norm(move))
    retention=float(np.dot(final_rn,t)/(tn*tn))

    efficiency=(
        progress/input_norm_ratio
        if input_norm_ratio>EPS
        else float("nan")
    )

    return {
        "sid":int(sid),
        "gt":str(gt),
        "baseline_correct":baseline_correct,
        "target_rank":int(target["rank"]),
        "causal_text_rank":int(target["causal_text_rank"]),
        "target_layer":int(target["target_layer"]),
        "probe_layer":int(probe_layer),
        "distance_to_target":int(target["target_layer"]-probe_layer),
        "real_position":int(target["real_position"]),
        "noimage_position":int(target["noimage_position"]),
        "token":str(target["token"]),
        "category":str(target["category"]),
        "broad_category":str(target["broad_category"]),
        "mode":str(mode),
        "alpha":float(alpha),
        "injection_scale":float(injection_scale),
        "input_norm_ratio":float(input_norm_ratio),
        "final_progress":progress,
        "final_move_cosine":cosine_np(move,t),
        "final_move_norm_ratio":move_norm/tn,
        "final_rn_retention":retention,
        "transport_efficiency":efficiency,
    }


# =============================================================================
# Summary
# =============================================================================

def summarize_natural(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for L,g in df.groupby("probe_layer"):
        rows.append({
            "probe_layer":int(L),
            "N_samples":g["sid"].nunique(),
            "N_target_states":len(g),
            "mean_natural_projection":safe_mean(g["natural_projection"]),
            "median_natural_projection":safe_median(g["natural_projection"]),
            "mean_natural_cosine":safe_mean(g["natural_cosine"]),
            "mean_natural_norm_ratio":safe_mean(g["natural_norm_ratio"]),
            "mean_incremental_projection_gain":safe_mean(
                g["incremental_projection_gain"]
            ),
        })

    return pd.DataFrame(rows).sort_values("probe_layer").reset_index(drop=True)


def summarize_transport(df,extra_keys=None):
    if len(df)==0:
        return pd.DataFrame()

    extra_keys=list(extra_keys or [])
    keys=extra_keys+["mode","alpha","probe_layer"]

    rows=[]
    for key,g in df.groupby(keys,dropna=False):
        if not isinstance(key,tuple):
            key=(key,)

        d={k:v for k,v in zip(keys,key)}
        progress=g["final_progress"].astype(float).to_numpy()

        d.update({
            "N_samples":g["sid"].nunique(),
            "N_target_states":len(g),
            "mean_final_progress":float(np.mean(progress)),
            "median_final_progress":float(np.median(progress)),
            "std_final_progress":float(np.std(progress)),
            "positive_progress_fraction":float(np.mean(progress>0)),
            "progress_gt_0p1_fraction":float(np.mean(progress>0.1)),
            "mean_final_move_cosine":safe_mean(g["final_move_cosine"]),
            "mean_final_move_norm_ratio":safe_mean(g["final_move_norm_ratio"]),
            "mean_final_rn_retention":safe_mean(g["final_rn_retention"]),
            "mean_input_norm_ratio":safe_mean(g["input_norm_ratio"]),
            "mean_transport_efficiency":safe_mean(g["transport_efficiency"]),
        })
        rows.append(d)

    out=pd.DataFrame(rows)
    sort_cols=extra_keys+["mode","alpha","probe_layer"]
    return out.sort_values(sort_cols).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers=parse_layers(a.causal_layers)
    probe_layers=parse_layers(a.probe_layers)
    modes=parse_set(a.modes)
    alphas=parse_floats(a.alphas)
    categories=parse_set(a.causal_categories)

    bad_modes=set(modes)-{"natural","normmatched"}
    if bad_modes:
        raise ValueError(f"Unknown modes: {sorted(bad_modes)}")

    outdir=Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)
    error_path=outdir/"errors.jsonl"

    two,meta,rec_by_sid=load_data(a)
    eval_sids={int(m["sid"]) for m in meta}

    causal_sel=load_causal_selection(
        Path(a.ranked_causal),
        eval_sids,
        causal_layers,
        a.causal_top_k,
        categories,
    )
    causal_sel.to_csv(
        outdir/"selected_causal_states.csv",
        index=False,
    )

    causal_by_sid={
        int(sid):g.sort_values("rank").copy()
        for sid,g in causal_sel.groupby("sid")
    }

    model=processor=None

    alignment_rows=[]
    baseline_rows=[]
    natural_rows=[]
    intervention_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        n_layers=len(decoder_layers)
        bad=[
            L for L in sorted(set(causal_layers+probe_layers))
            if not 0<=L<n_layers
        ]
        if bad:
            raise ValueError(
                f"Requested layers outside 0..{n_layers-1}: {bad}"
            )

        capture_layers=sorted(set(causal_layers+probe_layers))

        print("="*184)
        print("LAYERWISE SAME-TOKEN RN TRANSPORT -> CAUSAL STATE")
        print("="*184)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(meta)}")
        print(f"causal_layers={causal_layers} topK={a.causal_top_k}")
        print(f"probe_layers={probe_layers}")
        print(f"modes={modes} alphas={alphas}")
        print(f"patch_granularity={a.patch_granularity}")
        print()

        for m in tqdm(meta,desc="LAYERWISE RN TRANSPORT"):
            sid=int(m["sid"])
            if sid not in causal_by_sid:
                continue

            image=None

            try:
                image=base.record_image(rec_by_sid[sid])
                if hasattr(image,"convert"):
                    image=image.convert("RGB")

                rb=base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                nb=build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

                rid=rb["input_ids"][0].detach().cpu().tolist()
                nid=nb["input_ids"][0].detach().cpu().tolist()
                r2n=lcs_token_map(rid,nid)

                real_states=capture_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )
                no_states=capture_blocks(
                    model,
                    decoder_layers,
                    nb,
                    capture_layers,
                )

                targets=prepare_targets(
                    causal_rows=causal_by_sid[sid],
                    r2n=r2n,
                    real_states=real_states,
                    no_states=no_states,
                    probe_layers=probe_layers,
                )

                if not targets:
                    raise RuntimeError(
                        "No selected causal target could be aligned REAL->NoImage"
                    )

                # One normal generation, only for wrong/correct stratification.
                if a.skip_baseline_generation:
                    bpred=""
                    btext=""
                    bcorrect=None
                else:
                    btext=base.generate_text(
                        model,
                        processor,
                        rb,
                        max_new_tokens=a.max_new_tokens,
                    )
                    bpred=traj.normalize_relation(base,btext)
                    bcorrect=(bpred==m["gt"])

                baseline_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "prediction":bpred,
                    "correct":bcorrect,
                    "text":btext,
                })

                alignment_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "real_seq_len":len(rid),
                    "noimage_seq_len":len(nid),
                    "lcs_matches":len(r2n),
                    "selected_targets_requested":len(causal_by_sid[sid]),
                    "selected_targets_aligned":len(targets),
                })

                # ---------------------------------------------------------
                # Natural formation trajectory
                # ---------------------------------------------------------
                for t in targets:
                    natural_rows.extend(
                        natural_rows_for_target(
                            sid=sid,
                            gt=m["gt"],
                            baseline_correct=bcorrect,
                            target=t,
                            real_states=real_states,
                            no_states=no_states,
                        )
                    )

                # ---------------------------------------------------------
                # Isolated target patch
                # ---------------------------------------------------------
                if a.patch_granularity=="isolated":
                    for t in targets:
                        C=int(t["target_layer"])
                        p=int(t["real_position"])
                        tn=float(t["target_norm"])

                        for L in t["probe_layers"]:
                            d=source_delta_for_target(
                                t,L,real_states,no_states
                            )

                            for mode in modes:
                                for alpha in alphas:
                                    inj,scale,input_norm_ratio=build_injection(
                                        d,
                                        tn,
                                        mode,
                                        alpha,
                                        a.normmatch_cap,
                                    )
                                    if inj is None:
                                        continue

                                    # L==C is analytic: patching the target block
                                    # output itself produces exactly hR_C + injection.
                                    if int(L)==C:
                                        hp=(t["hR_C"]+inj).astype(np.float32)
                                    else:
                                        patched=run_patched_to_targets(
                                            model=model,
                                            decoder_layers=decoder_layers,
                                            real_batch=rb,
                                            patch_layer=L,
                                            patch_map={p:inj},
                                            target_layers=[C],
                                        )
                                        hp=patched[C][0,p].astype(np.float32)

                                    intervention_rows.append(
                                        intervention_metric_row(
                                            sid=sid,
                                            gt=m["gt"],
                                            baseline_correct=bcorrect,
                                            target=t,
                                            probe_layer=L,
                                            mode=mode,
                                            alpha=alpha,
                                            injection_scale=scale,
                                            input_norm_ratio=input_norm_ratio,
                                            patched_target_state=hp,
                                        )
                                    )

                # ---------------------------------------------------------
                # Joint same-token positions at each probe layer
                # ---------------------------------------------------------
                else:
                    for L in probe_layers:
                        active=[
                            t for t in targets
                            if int(L) in set(t["probe_layers"])
                        ]
                        if not active:
                            continue

                        for mode in modes:
                            for alpha in alphas:
                                patch_map={}
                                per_target_injection={}

                                for t in active:
                                    p=int(t["real_position"])
                                    d=source_delta_for_target(
                                        t,L,real_states,no_states
                                    )
                                    inj,scale,input_norm_ratio=build_injection(
                                        d,
                                        t["target_norm"],
                                        mode,
                                        alpha,
                                        a.normmatch_cap,
                                    )
                                    if inj is None:
                                        continue

                                    # Same token position may correspond to multiple
                                    # causal states at different target layers.
                                    # Add only one injection per position. If repeated,
                                    # keep the first; its d_L is identical for same p,L.
                                    if p not in patch_map:
                                        patch_map[p]=inj

                                    per_target_injection[
                                        (int(t["target_layer"]),p,int(t["rank"]))
                                    ]=(scale,input_norm_ratio,inj)

                                if not patch_map:
                                    continue

                                downstream_layers=sorted({
                                    int(t["target_layer"])
                                    for t in active
                                    if int(t["target_layer"])>int(L)
                                })

                                patched={}
                                if downstream_layers:
                                    patched=run_patched_to_targets(
                                        model=model,
                                        decoder_layers=decoder_layers,
                                        real_batch=rb,
                                        patch_layer=L,
                                        patch_map=patch_map,
                                        target_layers=downstream_layers,
                                    )

                                for t in active:
                                    C=int(t["target_layer"])
                                    p=int(t["real_position"])
                                    key=(C,p,int(t["rank"]))
                                    if key not in per_target_injection:
                                        continue
                                    scale,input_norm_ratio,inj=per_target_injection[key]

                                    if int(L)==C:
                                        hp=(t["hR_C"]+patch_map[p]).astype(np.float32)
                                    else:
                                        hp=patched[C][0,p].astype(np.float32)

                                    intervention_rows.append(
                                        intervention_metric_row(
                                            sid=sid,
                                            gt=m["gt"],
                                            baseline_correct=bcorrect,
                                            target=t,
                                            probe_layer=L,
                                            mode=mode,
                                            alpha=alpha,
                                            injection_scale=scale,
                                            input_norm_ratio=input_norm_ratio,
                                            patched_target_state=hp,
                                        )
                                    )

            except Exception as exc:
                append_jsonl(error_path,{
                    "phase":"sample",
                    "sid":sid,
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc().splitlines()[-50:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -----------------------------------------------------------------
        # Save raw outputs
        # -----------------------------------------------------------------
        align_df=pd.DataFrame(alignment_rows)
        base_df=pd.DataFrame(baseline_rows)
        natural_df=pd.DataFrame(natural_rows)
        int_df=pd.DataFrame(intervention_rows)

        align_df.to_csv(
            outdir/"alignment_summary.csv",
            index=False,
        )
        base_df.to_csv(
            outdir/"baseline_generation.csv",
            index=False,
        )
        natural_df.to_csv(
            outdir/"natural_trajectory_per_target.csv",
            index=False,
        )
        int_df.to_csv(
            outdir/"intervention_per_target.csv",
            index=False,
        )

        # -----------------------------------------------------------------
        # Summaries
        # -----------------------------------------------------------------
        natural_summary=summarize_natural(natural_df)
        transport_summary=summarize_transport(int_df)

        natural_summary.to_csv(
            outdir/"natural_trajectory_summary.csv",
            index=False,
        )
        transport_summary.to_csv(
            outdir/"layer_transport_summary.csv",
            index=False,
        )

        if len(int_df) and "baseline_correct" in int_df.columns:
            valid=int_df[int_df["baseline_correct"].notna()].copy()
            by_base=summarize_transport(
                valid,
                extra_keys=["baseline_correct"],
            )
        else:
            by_base=pd.DataFrame()

        by_base.to_csv(
            outdir/"layer_transport_by_baseline.csv",
            index=False,
        )

        by_rel=summarize_transport(
            int_df,
            extra_keys=["gt"],
        ) if len(int_df) else pd.DataFrame()

        by_rel.to_csv(
            outdir/"layer_transport_by_relation.csv",
            index=False,
        )

        # Rank layers by actual downstream progress, excluding trivial L==C
        # when possible.
        ranked=transport_summary.copy()
        if len(ranked):
            # This aggregate can mix target layers; L close to target is expected
            # to be stronger. Keep it descriptive, not a causal ranking across
            # unequal target cohorts.
            ranked=ranked.sort_values(
                ["mean_final_progress","positive_progress_fraction"],
                ascending=[False,False],
            )

        report=[
            "="*194,
            "LAYERWISE SAME-TOKEN RN TRANSPORT -> CAUSAL STATE",
            "="*194,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested={len(meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"probe layers={probe_layers}",
            f"modes={modes} alphas={alphas}",
            f"patch granularity={a.patch_granularity}",
            "",
            "NATURAL RN FORMATION TRAJECTORY",
            "-"*194,
            natural_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(natural_summary) else "EMPTY",
            "",
            "LAYERWISE RN AMPLIFICATION -> FINAL CAUSAL RN PROGRESS",
            "-"*194,
            transport_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(transport_summary) else "EMPTY",
            "",
            "STRATIFIED BY BASELINE CORRECTNESS",
            "-"*194,
            by_base.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(by_base) else "EMPTY / baseline generation skipped",
            "",
            "How to read:",
            "  natural_projection = how much the current layer's RN difference already lies",
            "                       on the final causal-token RN target axis.",
            "  final_progress      = after adding this layer's RN difference, how much the",
            "                       downstream causal state moves along its final RN target.",
            "  transport_efficiency= final_progress / injected_norm_ratio.",
            "",
            "Important control:",
            "  probe_layer == target_layer is a trivial sanity point: natural alpha=1 should",
            "  give final_progress ~= 1. Do NOT interpret that point as evidence of transport.",
            "",
            "What would be interesting:",
            "  A sharp rise in final_progress at L << C indicates a transition where existing",
            "  RN information becomes downstream-usable/transportable.",
            "  If wrong samples have normal natural_projection but weak final_progress, that",
            "  supports a utilization/transport bottleneck rather than missing evidence.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(outdir/"metadata.json",{
            "script":"probe_layerwise_rn_transport_to_causal_tokens_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_requested_N":len(meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "probe_layers":probe_layers,
            "modes":modes,
            "alphas":alphas,
            "patch_granularity":a.patch_granularity,
            "natural_target":"t_C = h_real[C,p]-h_noimage[C,q]",
            "patch":"h[L,p] <- h[L,p] + alpha*(h_real[L,p]-h_noimage[L,q])",
            "final_progress":"dot(h_patched[C,p]-h_real[C,p], t_C)/||t_C||^2",
            "alignment":"exact token-ID LCS",
            "oracle_note":"Causal target states originate from prior oracle writer-guided ranking; this is a mechanism diagnostic.",
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
