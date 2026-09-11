#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_rn_directional_causal_router_v1.py

Purpose
=======
Test a fully test-time / non-oracle router for causal-token Real-NoImage steering.

For each TEST sample we can compute, without GT:

    dRN[L,p] = h_real[L,p] - h_noimage[L,q(p)]

where q(p) is the text-token position aligned to REAL position p by exact
token-ID LCS.

The problem is that different spatial relations can have different causal
tokens. Therefore we do NOT rank tokens with one relation-agnostic sensitivity.

Instead, for every candidate state c=(L,p) and every relation
r in {left,right,above,below}, define the next-token relation margin

    m_r = logit_r - mean_{j != r} logit_j

and calculate

    S[c,r] = dRN[c]^T (d m_r / d h[c]).

Interpretation:
    S[c,r] > 0 means the sample-specific Real-NoImage displacement at this state,
    if locally amplified, pushes the model toward relation r.

Thus each relation gets its OWN causal-token ranking:

    C_r = TopK_c S[c,r].

Non-oracle routing
==================
For every relation:

    E_r = sum_{c in C_r, S[c,r] > 0} S[c,r]

and choose

    r_hat = argmax_r E_r.

No GT is used for r_hat or token selection.

Then steer only the selected states for r_hat:

    h[c] <- h[c] + alpha * dRN[c],   c in C_{r_hat}.

The script reports:
1) baseline greedy generation accuracy;
2) non-oracle router relation accuracy;
3) non-oracle RN steering accuracy / W2C / C2W;
4) conflict-only non-oracle steering:
       steer only when r_hat != baseline generated relation;
5) oracle-direction diagnostic:
       use GT relation ONLY to choose which relation-specific TopK map to steer.
   This is NOT a method; it diagnoses whether failure is routing or token scoring.

Candidate states
================
Default layers: L20-L26.
Candidate token positions are positions shared between REAL multimodal and
NoImage text-only token sequences under exact token-ID LCS, excluding prompt
last. This naturally excludes visual-only tokens while retaining structural
text/special tokens that exist in both branches.

Relation logits
===============
Default surface words match the existing COCO setup:
    left / right / on / under
with normalization:
    on -> above, under -> below.

Use --answer-surface above_below if the current prompt/model instead expects
left/right/above/below.

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_rn_directional_causal_router_v1.py \
  --model qwen-3b \
  --source-layers 20-26 \
  --top-ks 3,5,7 \
  --alphas 1 \
  --eval-max-samples 20 \
  --output-dir output/qwen3b_rn_directional_router_n20_v1 \
  --overwrite

Then N=80:
CUDA_VISIBLE_DEVICES=0 python -u eval_rn_directional_causal_router_v1.py \
  --model qwen-3b \
  --source-layers 20-26 \
  --top-ks 3,5,7 \
  --alphas 1 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_rn_directional_router_n80_v1 \
  --overwrite

Primary outputs
===============
per_state_relation_scores.csv
    S[c,r], dRN norm, gradient norm, cosine, token metadata.

router_per_sample.csv
    E_left/right/above/below, r_hat, route accuracy, baseline relation logits.

generation_per_sample.csv
generation_summary.csv
    baseline / nonoracle_all / nonoracle_conflict / oracle_direction.

selected_states.csv
    Exact states selected for each sample/relation/K.

analysis_summary.txt
metadata.json
errors.jsonl

Important interpretation
========================
- This is NOT "pick tokens that are sensitive to logits".
- It explicitly models four different relation-conditioned causal maps.
- The non-oracle question is whether the model's own Real-NoImage causal evidence
  E_r identifies the correct relation well enough to choose the correct map.
- GT is used only for evaluation and the oracle_direction diagnostic.
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

SURFACE_MAPS = {
    "on_under": {
        "left": "left",
        "right": "right",
        "above": "on",
        "below": "under",
    },
    "above_below": {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
    },
}


# =============================================================================
# CLI / generic
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

    p.add_argument("--source-layers", default="20-26")
    p.add_argument("--top-ks", default="3,5,7")
    p.add_argument("--alphas", default="1")

    p.add_argument(
        "--answer-surface",
        default="on_under",
        choices=sorted(SURFACE_MAPS),
    )
    p.add_argument(
        "--positive-only",
        action="store_true",
        default=True,
        help="Select only states with positive S[c,r].",
    )
    p.add_argument(
        "--include-negative",
        action="store_true",
        help="Override --positive-only and allow negative states in TopK.",
    )

    p.add_argument(
        "--exclude-relation-words",
        action="store_true",
        help=(
            "Optional stricter control: exclude prompt occurrences of "
            "left/right/above/below/on/under from candidate positions."
        ),
    )
    p.add_argument(
        "--exclude-object-tokens",
        action="store_true",
        help="Optional control: exclude subject/reference text-token positions.",
    )

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


def parse_ints(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text):
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
    if a.eval_max_samples>0:
        meta=traj.stratified_cap(meta,a.eval_max_samples,a.seed+1)
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


# =============================================================================
# Token utilities
# =============================================================================

def tokenizer_of(processor):
    return getattr(processor,"tokenizer",processor)


def resolve_single_token(tokenizer,word):
    # Follow the project's existing convention: bare lowercase first, then space.
    candidates=[
        word,
        " "+word,
        word.capitalize(),
        " "+word.capitalize(),
    ]
    tried=[]
    for text in candidates:
        ids=tokenizer.encode(text,add_special_tokens=False)
        tried.append((text,list(map(int,ids))))
        if len(ids)==1:
            return int(ids[0]),text,tried
    raise RuntimeError(
        f"Could not resolve one-token output for {word!r}; tried={tried}"
    )


def relation_token_setup(processor,surface_name):
    tok=tokenizer_of(processor)
    words=SURFACE_MAPS[surface_name]
    out={}
    for r in REL:
        tid,variant,tried=resolve_single_token(tok,words[r])
        out[r]={
            "word":words[r],
            "token_id":tid,
            "variant":variant,
            "tried":tried,
        }

    ids=[out[r]["token_id"] for r in REL]
    if len(set(ids))!=4:
        raise RuntimeError(f"Relation token IDs are not unique: {out}")

    print("\nRelation answer tokens:")
    for r in REL:
        info=out[r]
        dec=tok.decode([info["token_id"]])
        print(
            f"  {r:>5s}: word={info['word']!r} "
            f"id={info['token_id']} decoded={dec!r}"
        )
    return out


def all_subseq_positions(ids: List[int],pat: List[int]) -> List[int]:
    if not pat or len(pat)>len(ids):
        return []
    out=[]
    n=len(pat)
    for i in range(len(ids)-n+1):
        if ids[i:i+n]==pat:
            out.extend(range(i,i+n))
    return sorted(set(out))


def text_positions_for_string(tokenizer,ids,text):
    out=set()
    for s in (str(text)," "+str(text)):
        pat=list(map(int,tokenizer.encode(s,add_special_tokens=False)))
        out.update(all_subseq_positions(ids,pat))
    return sorted(out)


def excluded_semantic_positions(
    processor,
    ids,
    subject,
    reference,
    exclude_relation_words,
    exclude_object_tokens,
):
    tok=tokenizer_of(processor)
    bad=set()

    if exclude_object_tokens:
        bad.update(text_positions_for_string(tok,ids,subject))
        bad.update(text_positions_for_string(tok,ids,reference))

    if exclude_relation_words:
        for w in ("left","right","above","below","on","under"):
            bad.update(text_positions_for_string(tok,ids,w))

    return bad


def token_string(processor,tid):
    tok=tokenizer_of(processor)
    try:
        return str(tok.convert_ids_to_tokens(int(tid)))
    except Exception:
        try:
            return tok.decode([int(tid)],skip_special_tokens=False)
        except Exception:
            return str(int(tid))


# =============================================================================
# NoImage batch / sequence alignment
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
# State capture
# =============================================================================

class InferenceStateCapture:
    def __init__(self,decoder_layers,layers):
        self.states={}
        self.handles=[]
        for L in sorted(set(map(int,layers))):
            def make_hook(layer):
                def hook(_m,_inp,out):
                    h=traj.first_tensor(out)
                    self.states[layer]=(
                        h.detach().float().cpu().numpy().astype(np.float32)
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
def capture_inference_states(model,decoder_layers,batch,layers):
    cap=InferenceStateCapture(decoder_layers,layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        _=model(**kw)
        missing=[L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing states {missing}")
        return dict(cap.states)
    finally:
        cap.close()


class MultiLayerGradCapture:
    """
    Cut graph at earliest requested layer, keep later requested states in graph.
    Mirrors the project's established multi-source gradient capture.
    """
    def __init__(self,decoder_layers,layers):
        self.layers=sorted(set(map(int,layers)))
        self.cut=min(self.layers)
        self.states={}
        self.handles=[]

        for L in self.layers:
            if L==self.cut:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._cut_hook(L)
                    )
                )
            else:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._capture_hook(L)
                    )
                )

    def _cut_hook(self,L):
        def hook(_m,_inp,out):
            h=traj.first_tensor(out)
            y=h.detach().clone().requires_grad_(True)
            self.states[L]=y
            return traj.replace_first_tensor(out,y)
        return hook

    def _capture_hook(self,L):
        def hook(_m,_inp,out):
            h=traj.first_tensor(out)
            self.states[L]=h
            return None
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


def forward_four_relation_grads(
    model,
    decoder_layers,
    batch,
    source_layers,
    relation_info,
):
    """
    One forward, four autograd calls.

    margin_r = logit_r - mean(other three).

    Returns:
      real_states[L] : numpy [S,D]
      grads[r][L]    : numpy [S,D]
      relation_logits[r]
    """
    cap=MultiLayerGradCapture(decoder_layers,source_layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        out=model(**kw)

        logits=out.logits[0,-1,:].float()
        rel_logits={
            r:logits[int(relation_info[r]["token_id"])]
            for r in REL
        }

        missing=[L for L in source_layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing gradient states {missing}")

        tensors=[cap.states[L] for L in source_layers]
        grads_by_rel={}

        for ri,r in enumerate(REL):
            others=[rel_logits[o] for o in REL if o!=r]
            margin=rel_logits[r]-torch.stack(others).mean()

            gs=torch.autograd.grad(
                margin,
                tensors,
                retain_graph=(ri<len(REL)-1),
                create_graph=False,
                allow_unused=False,
            )
            grads_by_rel[r]={
                L:g[0].detach().float().cpu().numpy().astype(np.float32)
                for L,g in zip(source_layers,gs)
            }

        real_states={
            L:cap.states[L][0].detach().float().cpu().numpy().astype(np.float32)
            for L in source_layers
        }
        logits_float={r:float(rel_logits[r].detach().item()) for r in REL}

        return real_states,grads_by_rel,logits_float

    finally:
        cap.close()


# =============================================================================
# Direction-conditioned state scoring / routing
# =============================================================================

def build_scores(
    *,
    sid,
    gt,
    processor,
    real_ids,
    no_ids,
    r2n,
    candidate_positions,
    source_layers,
    real_states,
    no_states,
    grads_by_rel,
):
    rows=[]

    for L in source_layers:
        R=real_states[L]
        N=no_states[L][0]

        for p in candidate_positions:
            q=r2n.get(int(p),None)
            if q is None:
                continue
            if not (0<=p<R.shape[0] and 0<=q<N.shape[0]):
                continue

            d=(R[p]-N[q]).astype(np.float32)
            dn=float(np.linalg.norm(d))
            if dn<=EPS:
                continue

            for r in REL:
                g=grads_by_rel[r][L][p].astype(np.float32)
                gn=float(np.linalg.norm(g))
                s=float(np.dot(d,g))
                rows.append({
                    "sid":sid,
                    "gt":gt,
                    "source_layer":int(L),
                    "real_position":int(p),
                    "noimage_position":int(q),
                    "token_id":int(real_ids[p]),
                    "token":token_string(processor,real_ids[p]),
                    "relation":r,
                    "score":s,
                    "delta_rn_norm":dn,
                    "grad_norm":gn,
                    "delta_grad_cos":(
                        s/(dn*gn) if gn>EPS else float("nan")
                    ),
                })

    return pd.DataFrame(rows)


def topk_for_relation(score_df,relation,K,positive_only=True):
    z=score_df[score_df.relation==relation].copy()
    if positive_only:
        z=z[z.score>0].copy()
    z=z.sort_values("score",ascending=False).head(int(K)).copy()
    if len(z):
        z["selected_rank"]=np.arange(1,len(z)+1)
    return z


def route_one(score_df,K,positive_only=True):
    evidence={}
    selected={}

    for r in REL:
        z=topk_for_relation(score_df,r,K,positive_only)
        selected[r]=z
        evidence[r]=float(z.score.sum()) if len(z) else 0.0

    ordered=sorted(evidence.items(),key=lambda kv:kv[1],reverse=True)
    rhat=ordered[0][0]
    gap=float(ordered[0][1]-ordered[1][1])
    return rhat,evidence,gap,selected


def build_patch_map(selected_df,real_states,no_states,r2n):
    patch=defaultdict(dict)

    for r in selected_df.itertuples():
        L=int(r.source_layer)
        p=int(r.real_position)
        q=r2n.get(p,None)
        if q is None:
            continue
        d=(
            real_states[L][p]
            - no_states[L][0,q]
        ).astype(np.float32)
        patch[L][p]=d

    return {L:dict(v) for L,v in patch.items()}


# =============================================================================
# Generation patch
# =============================================================================

class AddStatePatch:
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
                    h=traj.first_tensor(out)
                    # Edit prefill only, never one-token decode steps.
                    if int(h.shape[1])!=self.prompt_len:
                        return None
                    y=h.clone()
                    for p,vec in pos_map.items():
                        p=int(p)
                        if 0<=p<int(y.shape[1]):
                            y[0,p]+=self.scale*torch.as_tensor(
                                vec,
                                device=y.device,
                                dtype=y.dtype,
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
def generate_relation(
    model,
    processor,
    decoder_layers,
    batch,
    max_new_tokens,
    patch_map=None,
    scale=1.0,
):
    prompt_len=int(batch["input_ids"].shape[1])

    ctx=(
        AddStatePatch(
            decoder_layers,patch_map,prompt_len,scale
        )
        if patch_map else contextlib.nullcontext()
    )

    with ctx:
        text=base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )

    pred=traj.normalize_relation(base,text)
    return pred,text


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(df):
    if len(df)==0:
        return pd.DataFrame()

    b=df[df.condition=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]

    for (cond,K,alpha),g in df[df.condition!="baseline"].groupby(
        ["condition","top_k","alpha"],
        dropna=False,
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
            "top_k":int(K),
            "alpha":float(alpha),
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
        ["condition","top_k","alpha"]
    ).reset_index(drop=True)


def summarize_router(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for K,g in df.groupby("top_k"):
        rows.append({
            "top_k":int(K),
            "N":len(g),
            "router_accuracy":float(
                np.mean(g["route_prediction"].astype(str)==g["gt"].astype(str))
            ),
            "prefill_logit_accuracy":float(
                np.mean(g["prefill_prediction"].astype(str)==g["gt"].astype(str))
            ),
            "mean_route_gap":safe_mean(g.route_gap),
            "median_route_gap":float(np.median(g.route_gap.astype(float))),
            "route_vs_prefill_agreement":float(
                np.mean(
                    g["route_prediction"].astype(str)
                    ==g["prefill_prediction"].astype(str)
                )
            ),
        })
    return pd.DataFrame(rows).sort_values("top_k").reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers=parse_layers(a.source_layers)
    top_ks=parse_ints(a.top_ks)
    alphas=parse_floats(a.alphas)
    positive_only=not bool(a.include_negative)

    outdir=Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)
    error_path=outdir/"errors.jsonl"

    two,meta,rec_by_sid=load_data(a)

    model=processor=None
    score_rows=[]
    router_rows=[]
    selected_rows=[]
    generation_rows=[]
    mapping_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        n_layers=len(decoder_layers)
        bad=[L for L in source_layers if not 0<=L<n_layers]
        if bad:
            raise ValueError(
                f"source layers outside 0..{n_layers-1}: {bad}"
            )

        relation_info=relation_token_setup(
            processor,a.answer_surface
        )

        for m in tqdm(meta,desc="RN DIRECTIONAL ROUTER"):
            sid=int(m["sid"])
            real_image=None

            try:
                real_image=base.record_image(rec_by_sid[sid])
                if hasattr(real_image,"convert"):
                    real_image=real_image.convert("RGB")

                rb=base.make_question_batch(
                    processor=processor,
                    image=real_image,
                    question_text=m["question_text"],
                    device=device,
                )
                nb=build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

                real_ids=rb["input_ids"][0].detach().cpu().tolist()
                no_ids=nb["input_ids"][0].detach().cpu().tolist()
                r2n=lcs_token_map(real_ids,no_ids)

                # Shared text/structural positions only. Prompt-last is excluded.
                candidate_positions=[
                    int(p) for p,q in sorted(r2n.items())
                    if p < len(real_ids)-1
                    and q < len(no_ids)-1
                    and int(real_ids[p])==int(no_ids[q])
                ]

                bad_semantic=excluded_semantic_positions(
                    processor,
                    real_ids,
                    m["subject"],
                    m["reference"],
                    a.exclude_relation_words,
                    a.exclude_object_tokens,
                )
                candidate_positions=[
                    p for p in candidate_positions if p not in bad_semantic
                ]

                if not candidate_positions:
                    raise RuntimeError("No candidate shared text positions")

                mapping_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "real_seq_len":len(real_ids),
                    "noimage_seq_len":len(no_ids),
                    "lcs_matches":len(r2n),
                    "candidate_positions":len(candidate_positions),
                    "candidate_fraction_real":(
                        len(candidate_positions)/max(len(real_ids),1)
                    ),
                })

                # NoImage hidden states.
                no_states=capture_inference_states(
                    model,
                    decoder_layers,
                    nb,
                    source_layers,
                )

                # REAL state + four relation-conditioned gradients.
                with torch.enable_grad():
                    real_states,grads_by_rel,rel_logits=(
                        forward_four_relation_grads(
                            model,
                            decoder_layers,
                            rb,
                            source_layers,
                            relation_info,
                        )
                    )

                prefill_pred=max(
                    REL,
                    key=lambda r:rel_logits[r],
                )

                score_df=build_scores(
                    sid=sid,
                    gt=m["gt"],
                    processor=processor,
                    real_ids=real_ids,
                    no_ids=no_ids,
                    r2n=r2n,
                    candidate_positions=candidate_positions,
                    source_layers=source_layers,
                    real_states=real_states,
                    no_states=no_states,
                    grads_by_rel=grads_by_rel,
                )
                if len(score_df)==0:
                    raise RuntimeError("Empty state-relation score table")

                score_rows.extend(score_df.to_dict("records"))

                # Baseline generation once.
                base_pred,base_text=generate_relation(
                    model,
                    processor,
                    decoder_layers,
                    rb,
                    a.max_new_tokens,
                )
                base_ok=base_pred==m["gt"]

                generation_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"baseline",
                    "top_k":0,
                    "alpha":0.0,
                    "route_prediction":"",
                    "prediction":base_pred,
                    "correct":base_ok,
                    "text":base_text,
                })

                for K in top_ks:
                    rhat,evidence,gap,selected=route_one(
                        score_df,
                        K,
                        positive_only=positive_only,
                    )

                    router_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "top_k":int(K),
                        "route_prediction":rhat,
                        "route_correct":rhat==m["gt"],
                        "route_gap":gap,
                        "prefill_prediction":prefill_pred,
                        "prefill_correct":prefill_pred==m["gt"],
                        "baseline_generation_prediction":base_pred,
                        "baseline_generation_correct":base_ok,
                        **{
                            f"evidence_{r}":float(evidence[r])
                            for r in REL
                        },
                        **{
                            f"logit_{r}":float(rel_logits[r])
                            for r in REL
                        },
                    })

                    # Save all four relation-specific TopK maps.
                    for r in REL:
                        z=selected[r]
                        for row in z.to_dict("records"):
                            row.update({
                                "top_k":int(K),
                                "map_relation":r,
                                "route_prediction":rhat,
                                "route_is_winner":r==rhat,
                                "map_is_gt":r==m["gt"],
                            })
                            selected_rows.append(row)

                    nonoracle_sel=selected[rhat]
                    oracle_sel=selected[m["gt"]]

                    nonoracle_patch=build_patch_map(
                        nonoracle_sel,
                        real_states,
                        no_states,
                        r2n,
                    )
                    oracle_patch=build_patch_map(
                        oracle_sel,
                        real_states,
                        no_states,
                        r2n,
                    )

                    for alpha in alphas:
                        # -------------------------------------------------
                        # Non-oracle: always steer selected route.
                        # -------------------------------------------------
                        pred,text=generate_relation(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            a.max_new_tokens,
                            patch_map=nonoracle_patch,
                            scale=alpha,
                        )
                        generation_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":"nonoracle_all",
                            "top_k":int(K),
                            "alpha":float(alpha),
                            "route_prediction":rhat,
                            "prediction":pred,
                            "correct":pred==m["gt"],
                            "text":text,
                        })

                        # -------------------------------------------------
                        # Non-oracle conflict-only:
                        # if route agrees with baseline, do nothing.
                        # -------------------------------------------------
                        if rhat==base_pred:
                            cpred,ctext=base_pred,base_text
                        else:
                            cpred,ctext=generate_relation(
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                a.max_new_tokens,
                                patch_map=nonoracle_patch,
                                scale=alpha,
                            )

                        generation_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":"nonoracle_conflict",
                            "top_k":int(K),
                            "alpha":float(alpha),
                            "route_prediction":rhat,
                            "prediction":cpred,
                            "correct":cpred==m["gt"],
                            "text":ctext,
                        })

                        # -------------------------------------------------
                        # Oracle-direction diagnostic only.
                        # Uses the same score definition and same TopK budget,
                        # but GT chooses which of the four maps is applied.
                        # -------------------------------------------------
                        opred,otext=generate_relation(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            a.max_new_tokens,
                            patch_map=oracle_patch,
                            scale=alpha,
                        )
                        generation_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":"oracle_direction",
                            "top_k":int(K),
                            "alpha":float(alpha),
                            "route_prediction":m["gt"],
                            "prediction":opred,
                            "correct":opred==m["gt"],
                            "text":otext,
                        })

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
                if real_image is not None:
                    with contextlib.suppress(Exception):
                        real_image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        score_df=pd.DataFrame(score_rows)
        router_df=pd.DataFrame(router_rows)
        selected_df=pd.DataFrame(selected_rows)
        gen_df=pd.DataFrame(generation_rows)
        map_df=pd.DataFrame(mapping_rows)

        score_df.to_csv(
            outdir/"per_state_relation_scores.csv",
            index=False,
        )
        router_df.to_csv(
            outdir/"router_per_sample.csv",
            index=False,
        )
        selected_df.to_csv(
            outdir/"selected_states.csv",
            index=False,
        )
        gen_df.to_csv(
            outdir/"generation_per_sample.csv",
            index=False,
        )
        map_df.to_csv(
            outdir/"noimage_mapping_summary.csv",
            index=False,
        )

        router_summary=summarize_router(router_df)
        generation_summary=summarize_generation(gen_df)

        router_summary.to_csv(
            outdir/"router_summary.csv",
            index=False,
        )
        generation_summary.to_csv(
            outdir/"generation_summary.csv",
            index=False,
        )

        # Relation confusion table for each K.
        confusion_rows=[]
        if len(router_df):
            for K,g in router_df.groupby("top_k"):
                for gt in REL:
                    q=g[g["gt"]==gt]
                    for pred in REL:
                        confusion_rows.append({
                            "top_k":int(K),
                            "gt":gt,
                            "prediction":pred,
                            "count":int(np.sum(q["route_prediction"]==pred)),
                            "N_gt":len(q),
                            "fraction":(
                                float(np.mean(q["route_prediction"]==pred))
                                if len(q) else float("nan")
                            ),
                        })
        pd.DataFrame(confusion_rows).to_csv(
            outdir/"router_confusion.csv",
            index=False,
        )

        report=[
            "="*190,
            "REAL-NOIMAGE DIRECTION-CONDITIONED CAUSAL ROUTER",
            "="*190,
            f"model={a.model} repo={spec.repo_id}",
            f"decoder={decoder_path}",
            f"N requested={len(meta)}",
            f"source_layers={source_layers}",
            f"top_ks={top_ks} alphas={alphas}",
            f"answer_surface={a.answer_surface}",
            f"positive_only={positive_only}",
            f"exclude_relation_words={a.exclude_relation_words}",
            f"exclude_object_tokens={a.exclude_object_tokens}",
            "",
            "ROUTER",
            "-"*190,
            router_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(router_summary) else "EMPTY",
            "",
            "GENERATION",
            "-"*190,
            generation_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(generation_summary) else "EMPTY",
            "",
            "Interpretation:",
            "  S[c,r] = (h_real-h_noimage)[c] dot grad_h margin_r.",
            "  Every relation gets a different TopK causal-state map.",
            "  route_prediction chooses the relation whose TopK positive RN causal evidence has the largest signed sum.",
            "  nonoracle_all uses no GT.",
            "  nonoracle_conflict uses no GT and edits only when the route disagrees with baseline generation.",
            "  oracle_direction uses GT only as a diagnostic ceiling for choosing among the four maps.",
            "",
            "Critical diagnosis:",
            "  - If oracle_direction improves strongly but nonoracle does not: routing is the bottleneck.",
            "  - If even oracle_direction is weak: S[c,r] is not selecting useful causal states.",
            "  - If router accuracy > baseline but steering still fails: route is informative, but alpha/TopK/edit geometry needs work.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(outdir/"metadata.json",{
            "script":"eval_rn_directional_causal_router_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_requested_N":len(meta),
            "source_layers":source_layers,
            "top_ks":top_ks,
            "alphas":alphas,
            "answer_surface":a.answer_surface,
            "relation_tokens":relation_info,
            "positive_only":positive_only,
            "score_definition":"S[c,r] = (h_real[c]-h_noimage[c]) dot grad_h[c] (logit_r - mean_other_relation_logits)",
            "router_definition":"argmax_r sum TopK positive S[c,r]",
            "noimage_alignment":"exact token-ID LCS",
            "conditions":{
                "baseline":"no edit",
                "nonoracle_all":"route by E_r; amplify RN at TopK states of routed relation",
                "nonoracle_conflict":"same, but edit only if routed relation differs from baseline generation",
                "oracle_direction":"GT selects relation-specific map; diagnostic only",
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
