#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_rn_sources_to_causal_tokens_v1.py

Question
========
We already know that, at selected behaviorally causal text states c=(C,p),

    t_c = h_real[C,p] - h_noimage[C,q(p)]

is a strong additive control direction.

This script asks a different question:

    WHERE DOES THIS REAL-NoImage CAUSAL-STATE SIGNAL COME FROM?

Instead of asking whether an upstream head "looks like" t_c, we causally REMOVE
one upstream module's REAL-specific contribution and measure how much of t_c
disappears downstream.

Core mediation test
===================
For a selected causal target c=(C,p), define:

    t_c = hR[C,p] - hN[C,q]

where q is the NoImage token aligned to REAL token p by exact token-ID LCS.

Now run a fresh REAL forward, but at an upstream layer L replace ONE source
module's output by its NoImage counterpart at aligned query positions.

Examples:

Whole attention:
    AttnOut_R[L,p] <- AttnOut_N[L,q]

MLP:
    MLPOut_R[L,p] <- MLPOut_N[L,q]

Individual head (PRE-W_O):
    z_R[L,h,p] <- z_N[L,h,q]

Everything else stays on the REAL trajectory and is recomputed naturally.

Let the patched downstream target state be hP[C,p]. Define remaining RN signal:

    u_c = hP[C,p] - hN[C,q]

Target-axis retention:
    retention = <u_c, t_c> / ||t_c||^2

Mediation/removal loss:
    loss = 1 - retention
         = <hR[C,p] - hP[C,p], t_c> / ||t_c||^2

Interpretation:
    loss = 0.00 : removing this source did not remove target-axis RN signal
    loss = 0.30 : about 30% of target-axis RN signal was removed
    loss = 1.00 : the target-axis RN signal was completely removed
    loss < 0    : replacement paradoxically strengthened target-axis RN signal
    loss > 1    : replacement overshot past the NoImage-side target

This is a finite-step causal mediation / necessity diagnostic, not a cosine
correlation and not a linear decomposition.

Patch scopes
============
causal_positions (default):
    At source layer L, replace the module only at the UNION of aligned query
    positions belonging to selected causal targets with target layer C >= L.
    This asks whether the module's Real-specific outputs at causal-token
    positions help form later causal states.

all_text:
    Replace the module at every REAL/NoImage-aligned text/structural position
    (excluding prompt-last). This captures indirect routes through other text
    positions, but is a broader intervention.

The default is causal_positions because scanning every head is already expensive.

Head scan
=========
All attention heads in --head-layers are scanned by default. The known Qwen3B
Direction Top20 are only ANNOTATED; they are not privileged in selection.

This lets us test:
    spatially decodable head != necessarily causal RN supplier.

Selected causal targets
=======================
Targets are read from the existing ranked causal CSV. This remains an ORACLE
MECHANISM experiment: GT/writer information was used upstream to discover those
causal states. The purpose here is circuit tracing, not non-oracle inference.

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u scan_rn_sources_to_causal_tokens_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --source-layers 18-26 \
  --scopes causal_positions \
  --eval-max-samples 20 \
  --output-dir output/qwen3b_rn_source_mediation_n20_v1 \
  --overwrite

If the N20 result is informative, run N80.

Broader indirect-routing control for selected heads:
    use --scopes causal_positions,all_text
and optionally:
    --heads "23:1,23:5,26:2,21:11,26:6,23:12"

Outputs
=======
selected_causal_states.csv
alignment_summary.csv
per_target_mediation.csv
module_summary.csv
head_summary.csv
spatial_vs_nonspatial_heads.csv
top_modules.txt
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
import scan_qwen_spatial_heads_vs_causal_core_v1 as spatialscan


REL = ("left", "right", "above", "below")
EPS = 1e-12

# Known Direction Top20 from the established Qwen3B synthetic->COCO direction scan.
DEFAULT_DIRECTION_TOP20 = (
    "26:3,23:1,23:5,26:2,22:9,23:0,22:13,22:2,21:14,23:10,"
    "21:5,22:12,21:1,26:1,27:2,27:1,21:11,22:14,21:3,22:10"
)


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
        "--causal-categories",
        default="",
        help="Optional broad_category filter. Empty = all nonvisual/nonlast causal text states.",
    )

    p.add_argument("--source-layers", default="18-26")
    p.add_argument(
        "--module-types",
        default="attention,mlp,head",
        help="Subset of attention,mlp,head.",
    )
    p.add_argument(
        "--scopes",
        default="causal_positions",
        help="Subset of causal_positions,all_text.",
    )
    p.add_argument(
        "--heads",
        default="",
        help=(
            'Optional explicit head subset "23:1,23:5,26:2". '
            "Empty = every attention head in --source-layers."
        ),
    )
    p.add_argument(
        "--spatial-heads",
        default=DEFAULT_DIRECTION_TOP20,
        help="Heads to annotate as known spatial Direction heads.",
    )

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


def parse_set(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_heads(text: str) -> List[Tuple[int,int]]:
    out=[]
    if not str(text).strip():
        return out
    for part in str(text).split(","):
        part=part.strip().upper().replace("L","").replace("H",":")
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad head spec {part!r}; expected L:H")
        a,b=part.split(":",1)
        out.append((int(a),int(b)))
    return sorted(set(out))


def head_name(L,h):
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


def cosine_np(a,b):
    a=np.asarray(a,np.float32)
    b=np.asarray(b,np.float32)
    na=float(np.linalg.norm(a))
    nb=float(np.linalg.norm(b))
    if na<=EPS or nb<=EPS:
        return float("nan")
    return float(np.dot(a,b)/(na*nb))


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
# NoImage batch / alignment
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
    """
    Exact token-ID LCS map:
        REAL multimodal position -> NoImage text-only position.
    """
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
# Model module helpers
# =============================================================================

def resolve_mlp(layer):
    for name in ("mlp","feed_forward","ffn","ff"):
        module=getattr(layer,name,None)
        if isinstance(module,torch.nn.Module):
            return module
    raise AttributeError(f"Unable to locate MLP on {type(layer).__name__}")


def tensor_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output,(tuple,list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported output type {type(output).__name__}")


def replace_tensor_output(original_output,tensor):
    if torch.is_tensor(original_output):
        return tensor
    if isinstance(original_output,tuple):
        return (tensor,*original_output[1:])
    if isinstance(original_output,list):
        return [tensor,*original_output[1:]]
    raise RuntimeError(f"Unsupported output type {type(original_output).__name__}")


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
            raise RuntimeError(f"L{L}: o_proj input width={width} not divisible by H={H}")
        out[L]={"n_heads":H,"head_dim":width//H,"width":width}
    return out


# =============================================================================
# Baseline activation capture
# =============================================================================

class FullCapture:
    """
    Capture:
      - block outputs at target layers
      - attention module post-WO outputs at source layers
      - attention PRE-WO vectors at source layers
      - MLP outputs at source layers

    Full sequence is stored on CPU for baseline REAL/NoImage only.
    """
    def __init__(
        self,
        decoder_layers,
        target_layers,
        source_layers,
    ):
        self.block={}
        self.attn={}
        self.pre_o={}
        self.mlp={}
        self.handles=[]

        for L in sorted(set(map(int,target_layers))):
            def make_block_hook(layer):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)
                    self.block[layer]=(
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook
            self.handles.append(
                decoder_layers[L].register_forward_hook(make_block_hook(L))
            )

        for L in sorted(set(map(int,source_layers))):
            layer=decoder_layers[L]
            attn=spatialscan.resolve_attn(layer)
            op=spatialscan.resolve_o_proj(attn)
            mlp=resolve_mlp(layer)

            def make_pre_hook(layer_index):
                def hook(_m,inputs):
                    if not inputs:
                        raise RuntimeError("o_proj prehook received no input")
                    x=inputs[0]
                    self.pre_o[layer_index]=(
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook

            def make_attn_hook(layer_index):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)
                    self.attn[layer_index]=(
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook

            def make_mlp_hook(layer_index):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)
                    self.mlp[layer_index]=(
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook

            self.handles.append(op.register_forward_pre_hook(make_pre_hook(L)))
            self.handles.append(attn.register_forward_hook(make_attn_hook(L)))
            self.handles.append(mlp.register_forward_hook(make_mlp_hook(L)))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def run_full_capture(
    model,
    decoder_layers,
    batch,
    target_layers,
    source_layers,
):
    cap=FullCapture(
        decoder_layers,
        target_layers,
        source_layers,
    )
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        _=model(**kw)

        miss_b=[L for L in target_layers if L not in cap.block]
        miss_a=[L for L in source_layers if L not in cap.attn]
        miss_o=[L for L in source_layers if L not in cap.pre_o]
        miss_m=[L for L in source_layers if L not in cap.mlp]
        if miss_b or miss_a or miss_o or miss_m:
            raise RuntimeError(
                f"Capture missing block={miss_b} attn={miss_a} pre_o={miss_o} mlp={miss_m}"
            )

        return {
            "block":dict(cap.block),
            "attn":dict(cap.attn),
            "pre_o":dict(cap.pre_o),
            "mlp":dict(cap.mlp),
        }
    finally:
        cap.close()


# =============================================================================
# Patch intervention
# =============================================================================

class SourceReplacement:
    """
    Replace one source module's REAL activation with aligned NoImage activation.

    kind:
      attention : replace self-attention module post-WO output
      mlp       : replace MLP output
      head      : replace one PRE-WO head slice at o_proj input
    """
    def __init__(
        self,
        *,
        decoder_layers,
        kind,
        layer,
        positions_map,   # real_position -> noimage_position
        no_cache,
        geom,
        head=None,
    ):
        self.handles=[]
        self.kind=str(kind)
        self.layer=int(layer)
        self.head=None if head is None else int(head)
        self.positions_map=dict(positions_map)

        layer_mod=decoder_layers[self.layer]

        if self.kind=="attention":
            module=spatialscan.resolve_attn(layer_mod)
            source=no_cache["attn"][self.layer]

            def hook(_m,_inp,out):
                x=tensor_from_output(out)
                y=x.clone()
                for p,q in self.positions_map.items():
                    if 0<=p<int(y.shape[1]) and 0<=q<int(source.shape[1]):
                        y[0,p]=torch.as_tensor(
                            source[0,q],
                            device=y.device,
                            dtype=y.dtype,
                        )
                return replace_tensor_output(out,y)

            self.handles.append(module.register_forward_hook(hook))

        elif self.kind=="mlp":
            module=resolve_mlp(layer_mod)
            source=no_cache["mlp"][self.layer]

            def hook(_m,_inp,out):
                x=tensor_from_output(out)
                y=x.clone()
                for p,q in self.positions_map.items():
                    if 0<=p<int(y.shape[1]) and 0<=q<int(source.shape[1]):
                        y[0,p]=torch.as_tensor(
                            source[0,q],
                            device=y.device,
                            dtype=y.dtype,
                        )
                return replace_tensor_output(out,y)

            self.handles.append(module.register_forward_hook(hook))

        elif self.kind=="head":
            if self.head is None:
                raise ValueError("head intervention requires head index")

            attn=spatialscan.resolve_attn(layer_mod)
            op=spatialscan.resolve_o_proj(attn)
            source=no_cache["pre_o"][self.layer]
            H=int(geom[self.layer]["n_heads"])
            D=int(geom[self.layer]["head_dim"])
            h=int(self.head)
            if not 0<=h<H:
                raise ValueError(f"L{self.layer} head {h} outside 0..{H-1}")
            a=h*D
            b=(h+1)*D

            def prehook(_m,inputs):
                if not inputs:
                    raise RuntimeError("o_proj prehook received no input")
                x=inputs[0]
                if not torch.is_tensor(x) or x.ndim!=3:
                    raise RuntimeError("o_proj input must be [B,S,D]")
                y=x.clone()
                for p,q in self.positions_map.items():
                    if 0<=p<int(y.shape[1]) and 0<=q<int(source.shape[1]):
                        y[0,p,a:b]=torch.as_tensor(
                            source[0,q,a:b],
                            device=y.device,
                            dtype=y.dtype,
                        )
                return (y,*inputs[1:])

            self.handles.append(op.register_forward_pre_hook(prehook))

        else:
            raise ValueError(f"Unknown kind={self.kind}")

    def __enter__(self):
        return self

    def __exit__(self,*args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


class TargetCapture:
    def __init__(self,decoder_layers,target_layers):
        self.block={}
        self.handles=[]
        for L in sorted(set(map(int,target_layers))):
            def make_hook(layer):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)
                    self.block[layer]=(
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
def run_patched_capture(
    *,
    model,
    decoder_layers,
    real_batch,
    target_layers,
    patch_kind,
    patch_layer,
    positions_map,
    no_cache,
    geom,
    head=None,
):
    with SourceReplacement(
        decoder_layers=decoder_layers,
        kind=patch_kind,
        layer=patch_layer,
        positions_map=positions_map,
        no_cache=no_cache,
        geom=geom,
        head=head,
    ):
        cap=TargetCapture(decoder_layers,target_layers)
        try:
            kw=dict(real_batch)
            kw["use_cache"]=False
            kw["return_dict"]=True
            _=model(**kw)
            missing=[L for L in target_layers if L not in cap.block]
            if missing:
                raise RuntimeError(f"Missing patched target block states {missing}")
            return dict(cap.block)
        finally:
            cap.close()


# =============================================================================
# Target geometry / mediation
# =============================================================================

def target_rows_for_sample(
    causal_rows,
    r2n,
    real_cache,
    no_cache,
):
    """
    Precompute valid target RN vectors.
    """
    targets=[]
    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)
        q=r2n.get(p,None)
        if q is None:
            continue
        if C not in real_cache["block"] or C not in no_cache["block"]:
            continue
        R=real_cache["block"][C]
        N=no_cache["block"][C]
        if not (0<=p<R.shape[1] and 0<=q<N.shape[1]):
            continue

        hR=R[0,p].astype(np.float32)
        hN=N[0,q].astype(np.float32)
        t=(hR-hN).astype(np.float32)
        tn=float(np.linalg.norm(t))
        if tn<=EPS:
            continue

        targets.append({
            "sid":int(r.sid),
            "gt":str(getattr(r,"gt","")),
            "rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "target_layer":C,
            "real_position":p,
            "noimage_position":int(q),
            "token":str(r.token),
            "category":str(r.category),
            "broad_category":str(r.broad_category),
            "target":t,
            "target_norm":tn,
            "hR":hR,
            "hN":hN,
        })
    return targets


def causal_position_map_for_source_layer(targets,L,r2n):
    """
    Union of selected causal target query positions whose target layer C >= L.
    """
    out={}
    for t in targets:
        if int(t["target_layer"]) < int(L):
            continue
        p=int(t["real_position"])
        q=r2n.get(p,None)
        if q is not None:
            out[p]=int(q)
    return out


def all_text_position_map(r2n,real_len,no_len):
    """
    All exact-LCS aligned positions except prompt-last on either branch.
    """
    return {
        int(p):int(q)
        for p,q in r2n.items()
        if 0<=p<real_len-1 and 0<=q<no_len-1
    }


def mediation_rows(
    *,
    sid,
    gt,
    scope,
    kind,
    source_layer,
    head,
    targets,
    patched_blocks,
    spatial_heads,
):
    rows=[]

    for t in targets:
        C=int(t["target_layer"])
        if int(source_layer)>C:
            continue
        p=int(t["real_position"])
        if C not in patched_blocks or not 0<=p<patched_blocks[C].shape[1]:
            continue

        hp=patched_blocks[C][0,p].astype(np.float32)
        hR=t["hR"]
        hN=t["hN"]
        target=t["target"]
        tn=float(t["target_norm"])

        remaining=(hp-hN).astype(np.float32)
        removed=(hR-hp).astype(np.float32)

        retention=float(np.dot(remaining,target)/(tn*tn))
        loss=float(np.dot(removed,target)/(tn*tn))

        rn_norm=float(np.linalg.norm(remaining))
        removed_norm=float(np.linalg.norm(removed))

        hn=head_name(source_layer,head) if head is not None else ""

        rows.append({
            "sid":int(sid),
            "gt":str(gt),
            "scope":str(scope),
            "module_type":str(kind),
            "source_layer":int(source_layer),
            "head":(-1 if head is None else int(head)),
            "head_name":hn,
            "is_spatial_head":bool(
                head is not None and (int(source_layer),int(head)) in spatial_heads
            ),
            "target_rank":int(t["rank"]),
            "causal_text_rank":int(t["causal_text_rank"]),
            "target_layer":C,
            "layer_distance":int(C-int(source_layer)),
            "real_position":p,
            "noimage_position":int(t["noimage_position"]),
            "token":str(t["token"]),
            "category":str(t["category"]),
            "broad_category":str(t["broad_category"]),
            "target_rn_norm":tn,
            "retention":retention,
            "mediation_loss":loss,
            "remaining_rn_norm_ratio":rn_norm/tn,
            "remaining_cos_to_target":cosine_np(remaining,target),
            "removed_norm_over_target":removed_norm/tn,
            "removed_cos_to_target":cosine_np(removed,target),
        })

    return rows


# =============================================================================
# Summaries
# =============================================================================

def summarize_modules(df):
    if len(df)==0:
        return pd.DataFrame()

    keys=["scope","module_type","source_layer","head","head_name","is_spatial_head"]
    rows=[]

    for key,g in df.groupby(keys,dropna=False):
        scope,kind,L,h,hname,is_spatial=key
        vals=g["mediation_loss"].astype(float).to_numpy()
        rows.append({
            "scope":scope,
            "module_type":kind,
            "source_layer":int(L),
            "head":int(h),
            "head_name":str(hname),
            "is_spatial_head":bool(is_spatial),
            "N_samples":int(g["sid"].nunique()),
            "N_target_states":int(len(g)),
            "mean_mediation_loss":float(np.mean(vals)),
            "median_mediation_loss":float(np.median(vals)),
            "std_mediation_loss":float(np.std(vals)),
            "positive_loss_fraction":float(np.mean(vals>0)),
            "loss_gt_0p1_fraction":float(np.mean(vals>0.1)),
            "loss_gt_0p25_fraction":float(np.mean(vals>0.25)),
            "mean_retention":safe_mean(g["retention"]),
            "mean_remaining_cos":safe_mean(g["remaining_cos_to_target"]),
            "mean_remaining_norm_ratio":safe_mean(g["remaining_rn_norm_ratio"]),
            "mean_removed_norm_over_target":safe_mean(g["removed_norm_over_target"]),
            "mean_removed_cos_to_target":safe_mean(g["removed_cos_to_target"]),
        })

    d=pd.DataFrame(rows)
    return d.sort_values(
        ["scope","module_type","mean_mediation_loss"],
        ascending=[True,True,False],
    ).reset_index(drop=True)


def summarize_spatial_groups(head_summary):
    if len(head_summary)==0:
        return pd.DataFrame()

    rows=[]
    for (scope,is_spatial),g in head_summary.groupby(
        ["scope","is_spatial_head"],
        dropna=False,
    ):
        rows.append({
            "scope":scope,
            "group":"known_direction_head" if bool(is_spatial) else "other_head",
            "N_heads":len(g),
            "mean_of_head_mean_mediation":safe_mean(g["mean_mediation_loss"]),
            "median_of_head_mean_mediation":safe_median(g["mean_mediation_loss"]),
            "mean_positive_loss_fraction":safe_mean(g["positive_loss_fraction"]),
            "mean_loss_gt_0p1_fraction":safe_mean(g["loss_gt_0p1_fraction"]),
        })
    return pd.DataFrame(rows)


def top_text(module_summary,topn=30):
    if len(module_summary)==0:
        return "EMPTY"

    cols=[
        "scope","module_type","source_layer","head_name","is_spatial_head",
        "N_samples","N_target_states","mean_mediation_loss",
        "median_mediation_loss","positive_loss_fraction",
        "mean_remaining_cos",
    ]
    z=module_summary.sort_values(
        "mean_mediation_loss",ascending=False
    ).head(topn)
    return z[cols].to_string(
        index=False,
        float_format=lambda x:f"{x:.4f}",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers=parse_layers(a.causal_layers)
    source_layers=parse_layers(a.source_layers)
    module_types=parse_set(a.module_types)
    scopes=parse_set(a.scopes)
    causal_categories=parse_set(a.causal_categories)

    valid_types={"attention","mlp","head"}
    bad_types=set(module_types)-valid_types
    if bad_types:
        raise ValueError(f"Unknown module types {sorted(bad_types)}")

    valid_scopes={"causal_positions","all_text"}
    bad_scopes=set(scopes)-valid_scopes
    if bad_scopes:
        raise ValueError(f"Unknown scopes {sorted(bad_scopes)}")

    explicit_heads=parse_heads(a.heads)
    spatial_heads=set(parse_heads(a.spatial_heads))

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
        causal_categories,
    )
    causal_sel.to_csv(outdir/"selected_causal_states.csv",index=False)

    causal_by_sid={
        int(sid):g.sort_values("rank").copy()
        for sid,g in causal_sel.groupby("sid")
    }

    model=processor=None
    alignment_rows=[]
    target_baseline_rows=[]
    mediation_all=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)
        n_layers=len(decoder_layers)

        bad=[L for L in sorted(set(source_layers+causal_layers)) if not 0<=L<n_layers]
        if bad:
            raise ValueError(
                f"Requested layers outside 0..{n_layers-1}: {bad}"
            )

        geom=infer_head_geometry(model,decoder_layers,source_layers)

        # Resolve actual head scan list.
        if explicit_heads:
            head_candidates=[
                (L,h) for L,h in explicit_heads
                if L in set(source_layers)
            ]
        else:
            head_candidates=[]
            for L in source_layers:
                H=int(geom[L]["n_heads"])
                head_candidates.extend((L,h) for h in range(H))

        # Validate explicit heads.
        for L,h in head_candidates:
            if not 0<=h<int(geom[L]["n_heads"]):
                raise ValueError(
                    f"Requested {head_name(L,h)} but L{L} has "
                    f"{geom[L]['n_heads']} heads"
                )

        print("="*170)
        print("REAL-NoImage SOURCE -> CAUSAL TOKEN MEDIATION SCAN")
        print("="*170)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(eval_meta)}")
        print(f"causal_layers={causal_layers} topK={a.causal_top_k}")
        print(f"source_layers={source_layers}")
        print(f"module_types={module_types}")
        print(f"scopes={scopes}")
        print(f"head candidates={len(head_candidates)}")
        print()

        for m in tqdm(eval_meta,desc="RN SOURCE MEDIATION"):
            sid=int(m["sid"])
            if sid not in causal_by_sid:
                continue

            image=None
            try:
                causal_rows=causal_by_sid[sid]
                target_layers=sorted(
                    set(map(int,causal_rows["source_layer"].tolist()))
                )

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

                real_cache=run_full_capture(
                    model,
                    decoder_layers,
                    rb,
                    target_layers,
                    source_layers,
                )
                no_cache=run_full_capture(
                    model,
                    decoder_layers,
                    nb,
                    target_layers,
                    source_layers,
                )

                # Add GT into causal dataframe view for downstream rows.
                causal_rows=causal_rows.copy()
                causal_rows["gt"]=m["gt"]

                targets=target_rows_for_sample(
                    causal_rows,
                    r2n,
                    real_cache,
                    no_cache,
                )
                if not targets:
                    raise RuntimeError("No causal target could be aligned REAL->NoImage")

                target_positions=set(
                    int(x["real_position"]) for x in targets
                )

                alignment_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "real_seq_len":len(rid),
                    "noimage_seq_len":len(nid),
                    "lcs_matches":len(r2n),
                    "selected_targets_requested":len(causal_rows),
                    "selected_targets_aligned":len(targets),
                    "unique_target_positions":len(target_positions),
                    "all_text_patch_positions":len(
                        all_text_position_map(r2n,len(rid),len(nid))
                    ),
                })

                for t in targets:
                    target_baseline_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "target_rank":int(t["rank"]),
                        "causal_text_rank":int(t["causal_text_rank"]),
                        "target_layer":int(t["target_layer"]),
                        "real_position":int(t["real_position"]),
                        "noimage_position":int(t["noimage_position"]),
                        "token":str(t["token"]),
                        "category":str(t["category"]),
                        "broad_category":str(t["broad_category"]),
                        "target_rn_norm":float(t["target_norm"]),
                    })

                scope_maps={}
                if "all_text" in scopes:
                    scope_maps["all_text"]=all_text_position_map(
                        r2n,len(rid),len(nid)
                    )

                # ---------------------------------------------------------
                # Whole attention / MLP scans
                # ---------------------------------------------------------
                for scope in scopes:
                    for L in source_layers:
                        if scope=="causal_positions":
                            pmap=causal_position_map_for_source_layer(
                                targets,L,r2n
                            )
                        else:
                            pmap=scope_maps["all_text"]

                        if not pmap:
                            continue

                        if "attention" in module_types:
                            patched=run_patched_capture(
                                model=model,
                                decoder_layers=decoder_layers,
                                real_batch=rb,
                                target_layers=target_layers,
                                patch_kind="attention",
                                patch_layer=L,
                                positions_map=pmap,
                                no_cache=no_cache,
                                geom=geom,
                                head=None,
                            )
                            mediation_all.extend(
                                mediation_rows(
                                    sid=sid,
                                    gt=m["gt"],
                                    scope=scope,
                                    kind="attention",
                                    source_layer=L,
                                    head=None,
                                    targets=targets,
                                    patched_blocks=patched,
                                    spatial_heads=spatial_heads,
                                )
                            )

                        if "mlp" in module_types:
                            patched=run_patched_capture(
                                model=model,
                                decoder_layers=decoder_layers,
                                real_batch=rb,
                                target_layers=target_layers,
                                patch_kind="mlp",
                                patch_layer=L,
                                positions_map=pmap,
                                no_cache=no_cache,
                                geom=geom,
                                head=None,
                            )
                            mediation_all.extend(
                                mediation_rows(
                                    sid=sid,
                                    gt=m["gt"],
                                    scope=scope,
                                    kind="mlp",
                                    source_layer=L,
                                    head=None,
                                    targets=targets,
                                    patched_blocks=patched,
                                    spatial_heads=spatial_heads,
                                )
                            )

                # ---------------------------------------------------------
                # Individual head scan
                # ---------------------------------------------------------
                if "head" in module_types:
                    for scope in scopes:
                        for L,h in head_candidates:
                            if scope=="causal_positions":
                                pmap=causal_position_map_for_source_layer(
                                    targets,L,r2n
                                )
                            else:
                                pmap=scope_maps["all_text"]

                            if not pmap:
                                continue

                            patched=run_patched_capture(
                                model=model,
                                decoder_layers=decoder_layers,
                                real_batch=rb,
                                target_layers=target_layers,
                                patch_kind="head",
                                patch_layer=L,
                                positions_map=pmap,
                                no_cache=no_cache,
                                geom=geom,
                                head=h,
                            )
                            mediation_all.extend(
                                mediation_rows(
                                    sid=sid,
                                    gt=m["gt"],
                                    scope=scope,
                                    kind="head",
                                    source_layer=L,
                                    head=h,
                                    targets=targets,
                                    patched_blocks=patched,
                                    spatial_heads=spatial_heads,
                                )
                            )

            except Exception as exc:
                append_jsonl(error_path,{
                    "phase":"sample",
                    "sid":sid,
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc().splitlines()[-40:],
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

        align_df=pd.DataFrame(alignment_rows)
        target_df=pd.DataFrame(target_baseline_rows)
        med_df=pd.DataFrame(mediation_all)

        align_df.to_csv(outdir/"alignment_summary.csv",index=False)
        target_df.to_csv(outdir/"baseline_target_geometry.csv",index=False)
        med_df.to_csv(outdir/"per_target_mediation.csv",index=False)

        module_summary=summarize_modules(med_df)
        module_summary.to_csv(outdir/"module_summary.csv",index=False)

        head_summary=module_summary[
            module_summary["module_type"]=="head"
        ].copy() if len(module_summary) else pd.DataFrame()
        head_summary.to_csv(outdir/"head_summary.csv",index=False)

        spatial_group=summarize_spatial_groups(head_summary)
        spatial_group.to_csv(
            outdir/"spatial_vs_nonspatial_heads.csv",
            index=False,
        )

        # Separate layer-level whole-attention / MLP table for easy reading.
        if len(module_summary):
            layer_summary=module_summary[
                module_summary["module_type"].isin(["attention","mlp"])
            ].copy()
        else:
            layer_summary=pd.DataFrame()
        layer_summary.to_csv(
            outdir/"attention_mlp_summary.csv",
            index=False,
        )

        top_txt=top_text(module_summary,topn=40)
        (outdir/"top_modules.txt").write_text(
            top_txt+"\n",
            encoding="utf-8",
        )

        report=[
            "="*190,
            "REAL-NoImage SOURCE -> CAUSAL TOKEN MEDIATION",
            "="*190,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested={len(eval_meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"source layers={source_layers}",
            f"module types={module_types}",
            f"scopes={scopes}",
            f"head candidates={len(head_candidates)}",
            "",
            "TOP MODULES BY MEAN TARGET-AXIS RN SIGNAL REMOVED",
            "-"*190,
            top_txt,
            "",
            "KNOWN DIRECTION HEADS vs OTHER HEADS",
            "-"*190,
            spatial_group.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(spatial_group) else "EMPTY",
            "",
            "WHOLE ATTENTION / MLP",
            "-"*190,
            layer_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(layer_summary) else "EMPTY",
            "",
            "How to read mediation_loss:",
            "  0.00 = replacing this source with NoImage removed none of the downstream target-axis RN signal.",
            "  0.30 = about 30% of that target-axis RN signal was removed.",
            "  1.00 = target-axis RN signal was completely removed.",
            "  negative values mean the replacement strengthened the target-axis RN signal.",
            "",
            "Paper-level diagnostic:",
            "  If only a subset of known spatial heads has large positive mediation_loss,",
            "  spatial decoding and causal supply are functionally dissociated.",
            "  If whole MLP removal is consistently larger than individual spatial-head removal,",
            "  that supports selective spatial supply followed by nonlinear integration/transformation.",
        ]
        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(outdir/"metadata.json",{
            "script":"scan_rn_sources_to_causal_tokens_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_requested_N":len(eval_meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "source_layers":source_layers,
            "module_types":module_types,
            "scopes":scopes,
            "explicit_heads":explicit_heads,
            "head_candidate_count":len(head_candidates),
            "known_spatial_heads":[
                head_name(L,h) for L,h in sorted(spatial_heads)
            ],
            "target":"h_real[C,p]-h_noimage[C,q]",
            "mediation_loss":"dot(h_real-h_patched, target) / ||target||^2",
            "retention":"dot(h_patched-h_noimage, target) / ||target||^2",
            "noimage_alignment":"exact token-ID LCS",
            "note":"Mechanism/oracle diagnostic: causal target states were discovered by prior oracle writer-guided ranking.",
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
