#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_causal_token_gray_interference_v1.py

Goal
====
Disentangle three different explanations of the strong causal-token Real-Gray
intervention.

For every selected causal text state c=(L,p), cache:

    hR = h_real[L,p]
    hG = h_gray[L,p]
    dRG = hR - hG

The established reference is:

    rg_add:
        h <- h + dRG

At baseline REAL trajectory this is approximately:
        hR + (hR-hG) = 2hR-hG

This script adds the diagnostics that were missing:

1) delta_replace
       h <- dRG
   Literal test of:
       "replace the causal-token vector by the Image-Gray vector itself"

2) gray_replace
       h <- hG
   Test whether moving the causal state back to the Gray state hurts behavior.

3) RG-axis sweep
       h <- h + alpha*dRG
   alpha=-1, -0.5, +0.5, +1 by default.
   Negative alpha moves toward Gray; positive alpha moves away from Gray.

Optional NoImage decomposition
==============================
A text-only NoImage prompt has different sequence length because it has no visual
placeholder/tokens. We therefore align REAL and NoImage token IDs using an LCS
sequence alignment and map every selected REAL causal text position p to its
corresponding NoImage text position q.

Cache:
    hN = h_noimage[L,q]

Then test:

4) noimage_replace
       h <- hN

5) anti_gray
       h <- h - beta*(hG-hN)

   This asks whether subtracting the Gray-specific displacement helps without
   copying an additional REAL state.

6) real_evidence
       h <- h + beta*(hR-hN)

   This asks whether amplifying REAL-vs-NoImage evidence helps.

Interpretation
==============
A) If gray_replace strongly hurts and RG-axis accuracy increases monotonically
   from alpha<0 to alpha>0, Real-Gray is a meaningful behavioral axis.

B) If anti_gray ~= rg_add while real_evidence is weak:
       Gray-specific interference is a plausible major source of the repair.

C) If real_evidence ~= rg_add while anti_gray is weak:
       amplification of REAL-specific evidence is the cleaner explanation.

D) If delta_replace itself works well:
       the useful causal state may be largely carried by dRG itself rather than
       requiring the full baseline hR.

Important
=========
- Selected causal states still come from the existing oracle causal-ranking CSV.
- REAL and GRAY have identical multimodal tokenization, so hR/hG use the exact
  same position p.
- NoImage states use LCS token-ID alignment. Mapping diagnostics are saved and
  NoImage conditions run only when ALL selected causal positions for a sample map.
- Multi-layer replacement means every selected state is edited at its own layer
  during one fresh REAL generation, as in the previous causal-RG experiments.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u eval_causal_token_gray_interference_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --rg-alphas=-1,-0.5,0.5,1 \
  --noimage-betas 1 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_causal_gray_interference_n80_v1 \
  --overwrite

If you only want the Real/Gray tests first:
    add --skip-noimage

Outputs
=======
selected_causal_states.csv
noimage_mapping.csv
generation_per_sample.csv
generation_summary.csv
movement_per_state.csv
movement_summary.csv
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


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

    p.add_argument(
        "--rg-alphas",
        default="-1,-0.5,0.5,1",
        help="REAL-Gray axis add coefficients. Baseline alpha=0 is reported separately.",
    )
    p.add_argument(
        "--noimage-betas",
        default="1",
        help="Shared beta(s) for anti_gray and real_evidence NoImage diagnostics.",
    )
    p.add_argument("--skip-noimage", action="store_true")

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


def decode_one(processor,tid):
    tok=getattr(processor,"tokenizer",processor)
    try:
        return tok.decode([int(tid)],skip_special_tokens=False)
    except Exception:
        return str(int(tid))


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
# NoImage batch and token alignment
# =============================================================================

def move_batch(batch,device):
    return {
        k:(v.to(device) if torch.is_tensor(v) else v)
        for k,v in batch.items()
    }


def build_noimage_batch(processor,question_text,device):
    # Matches the repository's established NoImage construction:
    # a text-only user message, no image content item.
    messages=[
        {
            "role":"user",
            "content":[
                {"type":"text","text":str(question_text)}
            ],
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
            batch=fn()
            return move_batch(batch,device)
        except Exception as exc:
            last_error=exc

    raise RuntimeError(f"NoImage processor failed: {last_error}")


def lcs_token_map(real_ids: List[int], no_ids: List[int]) -> Dict[int,int]:
    """
    Exact token-ID LCS mapping REAL multimodal sequence -> NoImage text-only sequence.

    The image placeholder/pad block present only in REAL is naturally skipped.
    For matched text tokens, returns real_position -> noimage_position.
    """
    a=list(map(int,real_ids))
    b=list(map(int,no_ids))
    n,m=len(a),len(b)

    # NoImage sequence is normally short; uint16 is enough for sequence lengths.
    dp=np.zeros((n+1,m+1),dtype=np.uint16)

    for i in range(n-1,-1,-1):
        ai=a[i]
        row=dp[i]
        row1=dp[i+1]
        for j in range(m-1,-1,-1):
            if ai==b[j]:
                row[j]=1+row1[j+1]
            else:
                x=row1[j]
                y=row[j+1]
                row[j]=x if x>=y else y

    mapping={}
    i=j=0
    while i<n and j<m:
        if a[i]==b[j] and dp[i,j]==1+dp[i+1,j+1]:
            mapping[i]=j
            i+=1
            j+=1
        elif dp[i+1,j]>=dp[i,j+1]:
            i+=1
        else:
            j+=1

    return mapping


# =============================================================================
# State capture
# =============================================================================

class StateCapture:
    def __init__(self,decoder_layers,layers,prompt_len=None):
        self.states={}
        self.prompt_len=prompt_len
        self.handles=[]

        for L in sorted(set(map(int,layers))):
            def make_hook(layer):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    if self.prompt_len is None or int(x.shape[1])==int(self.prompt_len):
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
def run_state_capture(model,decoder_layers,batch,layers):
    cap=StateCapture(decoder_layers,layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        _=model(**kw)
        return dict(cap.states)
    finally:
        cap.close()


# =============================================================================
# Intervention definitions
# =============================================================================

def positions_by_layer(causal_rows):
    out={}
    for r in causal_rows.itertuples():
        out.setdefault(int(r.source_layer),set()).add(int(r.position))
    return {L:sorted(v) for L,v in out.items()}


def rg_delta_map(real_states,gray_states,causal_rows):
    out={}
    for r in causal_rows.itertuples():
        L,p=int(r.source_layer),int(r.position)
        out.setdefault(L,{})[p]=(
            real_states[L][0,p]-gray_states[L][0,p]
        ).astype(np.float32)
    return out


def absolute_map_from_states(states,causal_rows,position_map=None):
    out={}
    for r in causal_rows.itertuples():
        L,p=int(r.source_layer),int(r.position)
        q=p if position_map is None else position_map.get(p,None)
        if q is None:
            continue
        if L not in states or not (0<=q<states[L].shape[1]):
            continue
        out.setdefault(L,{})[p]=states[L][0,q].astype(np.float32)
    return out


def delta_replace_map(real_states,gray_states,causal_rows):
    # Absolute replacement value = hR-hG.
    return rg_delta_map(real_states,gray_states,causal_rows)


def noimage_maps(real_states,gray_states,no_states,causal_rows,r2n):
    anti_gray={}
    real_evidence={}
    no_abs={}

    for r in causal_rows.itertuples():
        L,p=int(r.source_layer),int(r.position)
        q=r2n.get(p,None)
        if q is None:
            continue
        if not (
            L in real_states and L in gray_states and L in no_states
            and 0<=p<real_states[L].shape[1]
            and 0<=p<gray_states[L].shape[1]
            and 0<=q<no_states[L].shape[1]
        ):
            continue

        hR=real_states[L][0,p].astype(np.float32)
        hG=gray_states[L][0,p].astype(np.float32)
        hN=no_states[L][0,q].astype(np.float32)

        no_abs.setdefault(L,{})[p]=hN
        anti_gray.setdefault(L,{})[p]=(hN-hG).astype(np.float32)
        real_evidence.setdefault(L,{})[p]=(hR-hN).astype(np.float32)

    return no_abs,anti_gray,real_evidence


# =============================================================================
# Generic patch hooks
# =============================================================================

class AddPatch:
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


class ReplacePatch:
    def __init__(self,decoder_layers,absolute_map,prompt_len):
        self.decoder_layers=decoder_layers
        self.absolute_map=absolute_map
        self.prompt_len=int(prompt_len)
        self.handles=[]

    def __enter__(self):
        for L,by_pos in self.absolute_map.items():
            def make_hook(pos_map):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    if int(x.shape[1])!=self.prompt_len:
                        return None
                    y=x.clone()
                    for p,vec in pos_map.items():
                        p=int(p)
                        if 0<=p<int(y.shape[1]):
                            y[0,p]=torch.as_tensor(
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


@torch.inference_mode()
def generate_and_capture(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    state_layers,
    max_new_tokens,
    add_map=None,
    add_scale=1.0,
    replace_map=None,
):
    prompt_len=int(batch["input_ids"].shape[1])

    add_ctx=(
        AddPatch(decoder_layers,add_map,prompt_len,add_scale)
        if add_map else contextlib.nullcontext()
    )
    replace_ctx=(
        ReplacePatch(decoder_layers,replace_map,prompt_len)
        if replace_map else contextlib.nullcontext()
    )

    # Never use add+replace in one condition in this script.
    with add_ctx:
        with replace_ctx:
            cap=StateCapture(
                decoder_layers,state_layers,prompt_len=prompt_len
            )
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
# Per-state geometry relative to dRG
# =============================================================================

def movement_rows(
    *,
    sid,
    gt,
    condition,
    parameter,
    causal_rows,
    real_states,
    gray_states,
    patched_states,
):
    rows=[]

    for r in causal_rows.itertuples():
        L,p=int(r.source_layer),int(r.position)
        if L not in patched_states:
            continue
        if not (
            0<=p<real_states[L].shape[1]
            and 0<=p<gray_states[L].shape[1]
            and 0<=p<patched_states[L].shape[1]
        ):
            continue

        hR=real_states[L][0,p].astype(np.float32)
        hG=gray_states[L][0,p].astype(np.float32)
        d=(hR-hG).astype(np.float32)
        move=(patched_states[L][0,p]-hR).astype(np.float32)

        dn=float(np.linalg.norm(d))
        mn=float(np.linalg.norm(move))
        if dn<=EPS:
            continue

        dot=float(np.dot(move,d))

        rows.append({
            "sid":sid,
            "gt":gt,
            "condition":condition,
            "parameter":float(parameter),
            "causal_rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "causal_layer":L,
            "causal_position":p,
            "causal_token":str(r.token),
            "causal_category":str(r.broad_category),
            "rg_norm":dn,
            "move_norm":mn,
            "cos_move_vs_rg":dot/(max(mn,EPS)*dn),
            "rg_progress":dot/(dn*dn),
            "move_over_rg_norm":mn/dn,
            "relative_error_to_plus1_rg":float(
                np.linalg.norm(move-d)/dn
            ),
            "patched_state_norm":float(
                np.linalg.norm(patched_states[L][0,p])
            ),
            "real_state_norm":float(np.linalg.norm(hR)),
            "gray_state_norm":float(np.linalg.norm(hG)),
        })

    return rows


def summarize_generation(df):
    if len(df)==0:
        return pd.DataFrame()

    b=df[df.condition=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]

    for (cond,param),g in df[df.condition!="baseline"].groupby(
        ["condition","parameter"],dropna=False
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
            "parameter":float(param),
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
        ["condition","parameter"]
    ).reset_index(drop=True)


def summarize_movement(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (cond,param),g in df.groupby(
        ["condition","parameter"],dropna=False
    ):
        rows.append({
            "condition":cond,
            "parameter":float(param),
            "N_states":len(g),
            "N_samples":g.sid.nunique(),
            "mean_cos_move_vs_rg":safe_mean(g.cos_move_vs_rg),
            "median_cos_move_vs_rg":safe_median(g.cos_move_vs_rg),
            "mean_rg_progress":safe_mean(g.rg_progress),
            "median_rg_progress":safe_median(g.rg_progress),
            "mean_move_over_rg_norm":safe_mean(g.move_over_rg_norm),
            "median_move_over_rg_norm":safe_median(g.move_over_rg_norm),
            "mean_relative_error_to_plus1_rg":safe_mean(
                g.relative_error_to_plus1_rg
            ),
            "median_relative_error_to_plus1_rg":safe_median(
                g.relative_error_to_plus1_rg
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["condition","parameter"]
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
    rg_alphas=parse_floats(a.rg_alphas)
    no_betas=parse_floats(a.noimage_betas)

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
    gen_rows=[]
    move_rows=[]
    mapping_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        for m in tqdm(eval_meta,desc="GRAY INTERFERENCE"):
            sid=int(m["sid"])
            if sid not in causal_by_sid:
                continue

            real=gray=None

            try:
                causal_rows=causal_by_sid[sid]
                state_layers=sorted(set(map(int,causal_rows.source_layer.tolist())))

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

                rid=rb["input_ids"][0].detach().cpu().tolist()
                gid=gb["input_ids"][0].detach().cpu().tolist()
                if rid!=gid:
                    raise RuntimeError("REAL/GRAY tokenization mismatch")

                rstates=run_state_capture(
                    model,decoder_layers,rb,state_layers
                )
                gstates=run_state_capture(
                    model,decoder_layers,gb,state_layers
                )

                dmap=rg_delta_map(rstates,gstates,causal_rows)
                gray_abs=absolute_map_from_states(
                    gstates,causal_rows,position_map=None
                )
                delta_abs=delta_replace_map(
                    rstates,gstates,causal_rows
                )

                # ---------------------------------------------------------
                # Baseline
                # ---------------------------------------------------------
                bpred,btext,_=generate_and_capture(
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
                    "parameter":0.0,
                    "prediction":bpred,
                    "correct":bok,
                    "text":btext,
                })

                # ---------------------------------------------------------
                # Literal delta replacement: h <- hR-hG
                # ---------------------------------------------------------
                pred,text,pstates=generate_and_capture(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    state_layers=state_layers,
                    max_new_tokens=a.max_new_tokens,
                    replace_map=delta_abs,
                )
                ok=pred==m["gt"]
                gen_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"delta_replace",
                    "parameter":1.0,
                    "prediction":pred,
                    "correct":ok,
                    "text":text,
                })
                move_rows.extend(
                    movement_rows(
                        sid=sid,gt=m["gt"],
                        condition="delta_replace",parameter=1.0,
                        causal_rows=causal_rows,
                        real_states=rstates,
                        gray_states=gstates,
                        patched_states=pstates,
                    )
                )

                # ---------------------------------------------------------
                # Gray state replacement: h <- hG
                # ---------------------------------------------------------
                pred,text,pstates=generate_and_capture(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    state_layers=state_layers,
                    max_new_tokens=a.max_new_tokens,
                    replace_map=gray_abs,
                )
                ok=pred==m["gt"]
                gen_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"gray_replace",
                    "parameter":1.0,
                    "prediction":pred,
                    "correct":ok,
                    "text":text,
                })
                move_rows.extend(
                    movement_rows(
                        sid=sid,gt=m["gt"],
                        condition="gray_replace",parameter=1.0,
                        causal_rows=causal_rows,
                        real_states=rstates,
                        gray_states=gstates,
                        patched_states=pstates,
                    )
                )

                # ---------------------------------------------------------
                # Real-Gray axis sweep
                # ---------------------------------------------------------
                for alpha in rg_alphas:
                    pred,text,pstates=generate_and_capture(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        state_layers=state_layers,
                        max_new_tokens=a.max_new_tokens,
                        add_map=dmap,
                        add_scale=float(alpha),
                    )
                    ok=pred==m["gt"]

                    cond="rg_add"
                    gen_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "condition":cond,
                        "parameter":float(alpha),
                        "prediction":pred,
                        "correct":ok,
                        "text":text,
                    })
                    move_rows.extend(
                        movement_rows(
                            sid=sid,gt=m["gt"],
                            condition=cond,parameter=float(alpha),
                            causal_rows=causal_rows,
                            real_states=rstates,
                            gray_states=gstates,
                            patched_states=pstates,
                        )
                    )

                # ---------------------------------------------------------
                # Optional NoImage decomposition
                # ---------------------------------------------------------
                if not a.skip_noimage:
                    nb=build_noimage_batch(
                        processor,m["question_text"],device
                    )
                    nid=nb["input_ids"][0].detach().cpu().tolist()
                    r2n=lcs_token_map(rid,nid)

                    selected_positions=sorted(set(
                        map(int,causal_rows.position.tolist())
                    ))
                    all_mapped=all(p in r2n for p in selected_positions)

                    for r in causal_rows.itertuples():
                        p=int(r.position)
                        q=r2n.get(p,None)
                        mapping_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "causal_rank":int(r.rank),
                            "causal_layer":int(r.source_layer),
                            "real_position":p,
                            "noimage_position":q,
                            "mapped":q is not None,
                            "real_token_id":(
                                int(rid[p]) if 0<=p<len(rid) else None
                            ),
                            "noimage_token_id":(
                                int(nid[q]) if q is not None and 0<=q<len(nid)
                                else None
                            ),
                            "real_token_decoded":(
                                decode_one(processor,rid[p])
                                if 0<=p<len(rid) else ""
                            ),
                            "noimage_token_decoded":(
                                decode_one(processor,nid[q])
                                if q is not None and 0<=q<len(nid) else ""
                            ),
                            "sample_all_selected_mapped":all_mapped,
                            "real_seq_len":len(rid),
                            "noimage_seq_len":len(nid),
                            "lcs_matches":len(r2n),
                        })

                    if all_mapped:
                        nstates=run_state_capture(
                            model,decoder_layers,nb,state_layers
                        )

                        no_abs,anti_gray_map,real_ev_map=noimage_maps(
                            rstates,gstates,nstates,
                            causal_rows,r2n
                        )

                        # NoImage replacement
                        pred,text,pstates=generate_and_capture(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            state_layers=state_layers,
                            max_new_tokens=a.max_new_tokens,
                            replace_map=no_abs,
                        )
                        ok=pred==m["gt"]
                        gen_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":"noimage_replace",
                            "parameter":1.0,
                            "prediction":pred,
                            "correct":ok,
                            "text":text,
                        })
                        move_rows.extend(
                            movement_rows(
                                sid=sid,gt=m["gt"],
                                condition="noimage_replace",parameter=1.0,
                                causal_rows=causal_rows,
                                real_states=rstates,
                                gray_states=gstates,
                                patched_states=pstates,
                            )
                        )

                        for beta in no_betas:
                            # h - beta*(hG-hN) = h + beta*(hN-hG)
                            pred,text,pstates=generate_and_capture(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                state_layers=state_layers,
                                max_new_tokens=a.max_new_tokens,
                                add_map=anti_gray_map,
                                add_scale=float(beta),
                            )
                            ok=pred==m["gt"]
                            gen_rows.append({
                                "sid":sid,
                                "gt":m["gt"],
                                "condition":"anti_gray",
                                "parameter":float(beta),
                                "prediction":pred,
                                "correct":ok,
                                "text":text,
                            })
                            move_rows.extend(
                                movement_rows(
                                    sid=sid,gt=m["gt"],
                                    condition="anti_gray",parameter=float(beta),
                                    causal_rows=causal_rows,
                                    real_states=rstates,
                                    gray_states=gstates,
                                    patched_states=pstates,
                                )
                            )

                            # h + beta*(hR-hN)
                            pred,text,pstates=generate_and_capture(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                state_layers=state_layers,
                                max_new_tokens=a.max_new_tokens,
                                add_map=real_ev_map,
                                add_scale=float(beta),
                            )
                            ok=pred==m["gt"]
                            gen_rows.append({
                                "sid":sid,
                                "gt":m["gt"],
                                "condition":"real_evidence",
                                "parameter":float(beta),
                                "prediction":pred,
                                "correct":ok,
                                "text":text,
                            })
                            move_rows.extend(
                                movement_rows(
                                    sid=sid,gt=m["gt"],
                                    condition="real_evidence",parameter=float(beta),
                                    causal_rows=causal_rows,
                                    real_states=rstates,
                                    gray_states=gstates,
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

        gen_df=pd.DataFrame(gen_rows)
        move_df=pd.DataFrame(move_rows)
        map_df=pd.DataFrame(mapping_rows)

        gen_df.to_csv(outdir/"generation_per_sample.csv",index=False)
        move_df.to_csv(outdir/"movement_per_state.csv",index=False)
        map_df.to_csv(outdir/"noimage_mapping.csv",index=False)

        gs=summarize_generation(gen_df)
        ms=summarize_movement(move_df)

        gs.to_csv(outdir/"generation_summary.csv",index=False)
        ms.to_csv(outdir/"movement_summary.csv",index=False)

        if len(map_df):
            mapped_state_rate=float(map_df.mapped.mean())
            sample_map_rate=float(
                map_df.groupby("sid").sample_all_selected_mapped.first().mean()
            )
        else:
            mapped_state_rate=float("nan")
            sample_map_rate=float("nan")

        report=[
            "="*190,
            "CAUSAL TOKEN GRAY-INTERFERENCE / DIFFERENCE-VECTOR DIAGNOSTIC",
            "="*190,
            f"model={a.model} repo={spec.repo_id}",
            f"N eval={len(eval_meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"RG alphas={rg_alphas}",
            f"NoImage={'OFF' if a.skip_noimage else 'ON'} betas={no_betas}",
            f"NoImage mapped causal-state fraction={mapped_state_rate:.4f}",
            f"NoImage fully-mapped sample fraction={sample_map_rate:.4f}",
            "",
            "GENERATION",
            "-"*190,
            gs.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(gs) else "EMPTY",
            "",
            "MOVEMENT RELATIVE TO +1 REAL-GRAY DIRECTION",
            "-"*190,
            ms.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(ms) else "EMPTY",
            "",
            "Key condition definitions:",
            "  delta_replace : h <- h_real - h_gray",
            "  gray_replace  : h <- h_gray",
            "  rg_add(a)     : h <- h + a*(h_real-h_gray)",
            "  noimage_replace: h <- h_noimage (semantic token aligned by token-ID LCS)",
            "  anti_gray(b)  : h <- h - b*(h_gray-h_noimage)",
            "  real_evidence(b): h <- h + b*(h_real-h_noimage)",
            "",
            "Read the generation table first.",
            "  If delta_replace is strong, dRG itself may be close to a sufficient state.",
            "  If gray_replace hurts and positive rg_add helps monotonically, RG is a behavioral axis.",
            "  If anti_gray approaches rg_add, Gray-specific interference is a plausible contributor.",
            "  If real_evidence approaches rg_add instead, REAL-specific evidence amplification is cleaner.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,encoding="utf-8"
        )

        write_json(outdir/"metadata.json",{
            "script":"eval_causal_token_gray_interference_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_N":len(eval_meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "rg_alphas":rg_alphas,
            "noimage_betas":no_betas,
            "skip_noimage":a.skip_noimage,
            "conditions":{
                "delta_replace":"h <- hR-hG",
                "gray_replace":"h <- hG",
                "rg_add":"h <- h + alpha*(hR-hG)",
                "noimage_replace":"h <- hN",
                "anti_gray":"h <- h + beta*(hN-hG)",
                "real_evidence":"h <- h + beta*(hR-hN)",
            },
            "noimage_alignment":"exact token-ID LCS between REAL multimodal and NoImage text-only input_ids",
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
