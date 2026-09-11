#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_rn_causal_evidence_vs_decision_margin_v1.py

Goal
====
We already know that adding the selected causal-token Real-NoImage direction

    t_c = h_real[C,p] - h_noimage[C,q]

can strongly improve generation.

This script asks WHY.

Main hypotheses
===============

H1. Missing / weak causal evidence
    Wrong samples may have weak RN displacement at the causal states.

H2. Evidence exists and is decision-useful, but is underweighted
    Wrong samples may already contain RN displacement that points toward the
    correct answer, yet the current GT margin remains negative because competing
    decision components are stronger.

For each selected causal state c=(C,p), measure:

    t_c = hR[C,p] - hN[C,q]

and the local GT-vs-current-competitor margin gradient:

    m_GT = logit_GT - logit_competitor

    g_c = d m_GT / d h[C,p]

Decision leverage of the naturally present RN displacement:

    E_c = <t_c, g_c>

E_c > 0 means that, locally, adding more of this sample's own RN causal-state
difference pushes the decision toward the GT answer.

Then causally test the first-order prediction by an isolated patch:

    h[C,p] <- h[C,p] + beta * t_c

and measure the ACTUAL margin gain.

We also run the JOINT top-K causal-state patch and compare:

    predicted_joint_gain = beta * sum_c E_c

against:

    actual_joint_gain

and (optionally) actual greedy generation.

A second, simpler diagnostic compares REAL and NoImage output margins:

    output_RN_gain = margin_GT(REAL) - margin_GT(NoImage)

If a wrong REAL sample has:
    output_RN_gain > 0
but:
    margin_GT(REAL) < 0

then the image moved the model toward the correct answer, but not far enough to
win the final decision competition. This is direct evidence for
"present but insufficiently weighted" visual evidence.

Answer surface
==============
Default is the repository's common four-way surface:
    left  -> "left"
    right -> "right"
    above -> "on"
    below -> "under"

Use:
    --answer-surface above_below
if the prompt/model actually predicts literal "above"/"below".

Important
=========
The selected causal states come from the existing oracle causal-ranking CSV.
Therefore this is a MECHANISM diagnostic, not a non-oracle method.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u analyze_rn_causal_evidence_vs_decision_margin_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --patch-beta 1 \
  --answer-surface on_under \
  --eval-max-samples 40 \
  --output-dir output/qwen3b_rn_evidence_vs_decision_n40_v1 \
  --overwrite

Outputs
=======
per_target_leverage.csv
sample_decision_summary.csv
leverage_by_generation_correctness.csv
leverage_by_prefill_correctness.csv
wrong_case_diagnosis.csv
generation_per_sample.csv
generation_summary.csv
alignment_summary.csv
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
# CLI / basic utilities
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

    p.add_argument("--patch-beta", type=float, default=1.0)
    p.add_argument(
        "--answer-surface",
        default="on_under",
        choices=["on_under","above_below"],
    )

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip baseline/joint-patch greedy generation; margin diagnostics still run.",
    )
    p.add_argument(
        "--skip-isolated-patches",
        action="store_true",
        help="Skip one-at-a-time causal-state patch forwards; useful for a fast gradient-only scan.",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=40)

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
# Relation answer tokens / margins
# =============================================================================

def tokenizer_of(processor):
    return getattr(processor,"tokenizer",processor)


def answer_words(surface):
    if surface=="on_under":
        return {
            "left":"left",
            "right":"right",
            "above":"on",
            "below":"under",
        }
    return {
        "left":"left",
        "right":"right",
        "above":"above",
        "below":"below",
    }


def relation_token_ids(processor,surface):
    tok=tokenizer_of(processor)
    words=answer_words(surface)
    result={}

    print("\nRelation answer tokens:")

    for relation in REL:
        word=words[relation]
        candidates=[
            word,
            " "+word,
            word.capitalize(),
            " "+word.capitalize(),
        ]

        chosen=None
        tried=[]

        for text in candidates:
            ids=tok.encode(text,add_special_tokens=False)
            tried.append((text,list(map(int,ids))))
            if len(ids)==1:
                chosen=int(ids[0])
                break

        if chosen is None:
            raise RuntimeError(
                f"Could not resolve a single answer token for {relation} "
                f"(word={word!r}); tried={tried}"
            )

        result[relation]=chosen
        print(
            f"  {relation:>5s}: word={word!r} id={chosen} "
            f"decoded={tok.decode([chosen])!r}"
        )

    if len(set(result.values()))!=4:
        raise RuntimeError(f"Relation token IDs not unique: {result}")

    return result


def scores_from_logits(logits_last,token_ids):
    return {
        r:float(logits_last[token_ids[r]].detach().float().item())
        for r in REL
    }


def pred_from_scores(scores):
    return max(REL,key=lambda r:scores[r])


def dynamic_gt_margin(scores,gt):
    return float(
        scores[gt]
        -
        max(scores[r] for r in REL if r!=gt)
    )


def best_other(scores,gt):
    return max((r for r in REL if r!=gt),key=lambda r:scores[r])


def fixed_margin(scores,gt,competitor):
    return float(scores[gt]-scores[competitor])


# =============================================================================
# Hidden-state capture
# =============================================================================

def first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output,(tuple,list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported layer output type: {type(output).__name__}")


def replace_first_tensor(output,tensor):
    if torch.is_tensor(output):
        return tensor
    if isinstance(output,tuple):
        return (tensor,*output[1:])
    if isinstance(output,list):
        return [tensor,*output[1:]]
    raise RuntimeError(f"Unsupported layer output type: {type(output).__name__}")


class CpuBlockCapture:
    def __init__(self,decoder_layers,layers):
        self.states={}
        self.handles=[]

        for L in sorted(set(map(int,layers))):
            def make_hook(layer):
                def hook(_m,_inp,out):
                    x=first_tensor(out)
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
def run_cpu_capture(model,decoder_layers,batch,layers):
    cap=CpuBlockCapture(decoder_layers,layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        out=model(**kw)

        missing=[L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"CPU capture missing layers {missing}")

        logits=out.logits[0,-1].detach().float().cpu()
        return dict(cap.states),logits
    finally:
        cap.close()


class GraphBlockCapture:
    """
    Cut autograd at earliest requested causal layer and retain all requested
    block outputs downstream of that cut.
    """
    def __init__(self,decoder_layers,layers):
        self.states={}
        self.handles=[]

        layers=sorted(set(map(int,layers)))
        if not layers:
            raise ValueError("GraphBlockCapture needs at least one layer")

        cut=min(layers)

        def cut_hook(_m,_inp,out):
            x=first_tensor(out)
            y=x.detach().clone().requires_grad_(True)
            return replace_first_tensor(out,y)

        # Register cut first so subsequent state hook sees the cut tensor.
        self.handles.append(
            decoder_layers[cut].register_forward_hook(cut_hook)
        )

        for L in layers:
            def make_state_hook(layer):
                def hook(_m,_inp,out):
                    self.states[layer]=first_tensor(out)
                    return None
                return hook

            self.handles.append(
                decoder_layers[L].register_forward_hook(make_state_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


def run_real_graph(
    *,
    model,
    decoder_layers,
    batch,
    target_layers,
    token_ids,
    gt,
):
    cap=GraphBlockCapture(decoder_layers,target_layers)

    try:
        with torch.enable_grad():
            kw=dict(batch)
            kw["use_cache"]=False
            kw["return_dict"]=True
            out=model(**kw)

            missing=[L for L in target_layers if L not in cap.states]
            if missing:
                raise RuntimeError(f"Graph capture missing layers {missing}")

            logits_last=out.logits[0,-1]
            score_tensors={
                r:logits_last[token_ids[r]]
                for r in REL
            }
            scores={
                r:float(score_tensors[r].detach().float().item())
                for r in REL
            }

            competitor=best_other(scores,gt)
            margin_tensor=score_tensors[gt]-score_tensors[competitor]

            grad_layers=sorted(set(map(int,target_layers)))
            state_tensors=[cap.states[L] for L in grad_layers]

            grads=torch.autograd.grad(
                margin_tensor,
                state_tensors,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            grad_by_layer={}
            for L,g in zip(grad_layers,grads):
                if g is None:
                    grad_by_layer[L]=None
                else:
                    grad_by_layer[L]=(
                        g.detach().float().cpu().numpy().astype(np.float32)
                    )

            state_cpu={
                L:cap.states[L].detach().float().cpu().numpy().astype(np.float32)
                for L in grad_layers
            }

            return {
                "states":state_cpu,
                "grads":grad_by_layer,
                "scores":scores,
                "competitor":competitor,
                "dynamic_margin":dynamic_gt_margin(scores,gt),
                "fixed_margin":fixed_margin(scores,gt,competitor),
            }

    finally:
        cap.close()


# =============================================================================
# Causal-state RN target preparation
# =============================================================================

def prepare_targets(
    *,
    causal_rows,
    r2n,
    real_states,
    no_states,
    grad_by_layer,
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
        G=grad_by_layer.get(C,None)

        if G is None:
            continue

        if not (
            0<=p<R.shape[1]
            and 0<=q<N.shape[1]
            and 0<=p<G.shape[1]
        ):
            continue

        hR=R[0,p].astype(np.float32)
        hN=N[0,q].astype(np.float32)
        t=(hR-hN).astype(np.float32)
        g=G[0,p].astype(np.float32)

        tn=float(np.linalg.norm(t))
        gn=float(np.linalg.norm(g))

        if tn<=EPS:
            continue

        dot=float(np.dot(t,g))

        targets.append({
            "sid":int(r.sid),
            "rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "layer":C,
            "real_position":p,
            "noimage_position":int(q),
            "token":str(r.token),
            "category":str(r.category),
            "broad_category":str(r.broad_category),
            "hR":hR,
            "hN":hN,
            "rn":t,
            "grad":g,
            "rn_norm":tn,
            "grad_norm":gn,
            "leverage_dot":dot,
            "leverage_cosine":cosine_np(t,g),
            "leverage_per_rn_norm":dot/max(tn,EPS),
            "leverage_per_grad_norm":dot/max(gn,EPS),
        })

    return targets


# =============================================================================
# State patching
# =============================================================================

class ResidualAddPatch:
    """
    Add vectors to decoder block OUTPUTS.

    patch_map:
        layer -> {real_position -> vector_to_add}
    """
    def __init__(self,decoder_layers,patch_map):
        self.handles=[]

        for L,pos_map in sorted(patch_map.items()):
            if not pos_map:
                continue

            def make_hook(local_pos_map):
                def hook(_m,_inp,out):
                    x=first_tensor(out)
                    y=x.clone()

                    for p,vec in local_pos_map.items():
                        p=int(p)
                        if 0<=p<int(y.shape[1]):
                            y[0,p]+=torch.as_tensor(
                                vec,
                                device=y.device,
                                dtype=y.dtype,
                            )

                    return replace_first_tensor(out,y)
                return hook

            self.handles.append(
                decoder_layers[int(L)].register_forward_hook(
                    make_hook(dict(pos_map))
                )
            )

    def __enter__(self):
        return self

    def __exit__(self,*args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


def one_target_patch_map(target,beta):
    return {
        int(target["layer"]):{
            int(target["real_position"]):
                (float(beta)*target["rn"]).astype(np.float32)
        }
    }


def joint_patch_map(targets,beta):
    out={}

    for t in targets:
        L=int(t["layer"])
        p=int(t["real_position"])
        v=(float(beta)*t["rn"]).astype(np.float32)

        out.setdefault(L,{})
        if p in out[L]:
            # Duplicate state should not normally occur. Do not double-add.
            continue
        out[L][p]=v

    return out


@torch.inference_mode()
def forward_scores_with_patch(
    *,
    model,
    decoder_layers,
    batch,
    token_ids,
    patch_map,
):
    with ResidualAddPatch(decoder_layers,patch_map):
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        out=model(**kw)
        logits=out.logits[0,-1].float()
        scores=scores_from_logits(logits,token_ids)
        return scores


@torch.inference_mode()
def generate_with_patch(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    patch_map,
    max_new_tokens,
):
    with ResidualAddPatch(decoder_layers,patch_map):
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

def summarize_per_target(df,key):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]

    for value,g in df.groupby(key,dropna=False):
        rows.append({
            key:value,
            "N_samples":g["sid"].nunique(),
            "N_target_states":len(g),
            "mean_rn_norm":safe_mean(g["rn_norm"]),
            "median_rn_norm":safe_median(g["rn_norm"]),
            "mean_grad_norm":safe_mean(g["grad_norm"]),
            "mean_leverage_dot":safe_mean(g["leverage_dot"]),
            "median_leverage_dot":safe_median(g["leverage_dot"]),
            "positive_leverage_fraction":float(
                np.mean(g["leverage_dot"].astype(float).to_numpy()>0)
            ),
            "mean_leverage_cosine":safe_mean(g["leverage_cosine"]),
            "mean_predicted_gain":safe_mean(g["predicted_fixed_gain"]),
            "mean_actual_fixed_gain":safe_mean(g["actual_fixed_gain"]),
            "mean_actual_dynamic_gain":safe_mean(g["actual_dynamic_gain"]),
            "mean_linearity_ratio":safe_mean(g["linearity_ratio"]),
            "isolated_prefill_W2C":int(
                np.sum(
                    (~g["prefill_correct"].astype(bool))
                    &
                    g["patched_prefill_correct"].astype(bool)
                )
            ),
            "isolated_prefill_C2W":int(
                np.sum(
                    g["prefill_correct"].astype(bool)
                    &
                    (~g["patched_prefill_correct"].astype(bool))
                )
            ),
        })

    return pd.DataFrame(rows)


def summarize_generation(df):
    if len(df)==0:
        return pd.DataFrame()

    b=df[df["condition"]=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]

    for cond,g in df[df["condition"]!="baseline"].groupby("condition"):
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
            "condition":str(cond),
            "N":len(common),
            "baseline_accuracy":float(np.mean(bc)),
            "patched_accuracy":float(np.mean(pc)),
            "gain":float(np.mean(pc)-np.mean(bc)),
            "wrong_to_correct":w2c,
            "correct_to_wrong":c2w,
            "net":w2c-c2w,
            "changed":int(np.sum(bp!=pp)),
        })

    return pd.DataFrame(rows)


def summarize_sample_groups(df,key):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]

    for value,g in df.groupby(key,dropna=False):
        rows.append({
            key:value,
            "N":len(g),
            "mean_real_gt_margin":safe_mean(g["real_gt_margin"]),
            "mean_noimage_gt_margin":safe_mean(g["noimage_gt_margin"]),
            "mean_output_rn_margin_gain":safe_mean(g["output_rn_margin_gain"]),
            "output_rn_gain_positive_fraction":float(
                np.mean(g["output_rn_margin_gain"].astype(float).to_numpy()>0)
            ),
            "mean_sum_leverage":safe_mean(g["sum_leverage_dot"]),
            "sum_leverage_positive_fraction":float(
                np.mean(g["sum_leverage_dot"].astype(float).to_numpy()>0)
            ),
            "mean_predicted_joint_gain":safe_mean(g["predicted_joint_fixed_gain"]),
            "mean_actual_joint_fixed_gain":safe_mean(g["actual_joint_fixed_gain"]),
            "mean_actual_joint_dynamic_gain":safe_mean(g["actual_joint_dynamic_gain"]),
            "joint_prefill_correct_fraction":float(
                np.mean(g["joint_prefill_correct"].astype(bool))
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers=parse_layers(a.causal_layers)
    categories=parse_set(a.causal_categories)

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
    target_rows=[]
    sample_rows=[]
    generation_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        token_ids=relation_token_ids(
            processor,
            a.answer_surface,
        )

        n_layers=len(decoder_layers)
        bad=[L for L in causal_layers if not 0<=L<n_layers]
        if bad:
            raise ValueError(
                f"Requested causal layers outside 0..{n_layers-1}: {bad}"
            )

        print("="*190)
        print("RN CAUSAL EVIDENCE vs FINAL DECISION MARGIN")
        print("="*190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(meta)}")
        print(f"causal_layers={causal_layers} topK={a.causal_top_k}")
        print(f"patch_beta={a.patch_beta}")
        print(f"answer_surface={a.answer_surface}")
        print(f"skip_isolated_patches={a.skip_isolated_patches}")
        print(f"skip_generation={a.skip_generation}")
        print()

        for m in tqdm(meta,desc="RN EVIDENCE vs DECISION"):
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

                # NoImage target states and NoImage output logits.
                no_states,no_logits=run_cpu_capture(
                    model,
                    decoder_layers,
                    nb,
                    target_layers,
                )
                no_scores=scores_from_logits(no_logits,token_ids)

                # REAL states + GT-margin gradients in one graph forward.
                real_graph=run_real_graph(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    target_layers=target_layers,
                    token_ids=token_ids,
                    gt=m["gt"],
                )

                real_scores=real_graph["scores"]
                competitor=real_graph["competitor"]

                targets=prepare_targets(
                    causal_rows=causal_rows,
                    r2n=r2n,
                    real_states=real_graph["states"],
                    no_states=no_states,
                    grad_by_layer=real_graph["grads"],
                )

                if not targets:
                    raise RuntimeError(
                        "No causal target survived REAL-NoImage alignment/gradient capture"
                    )

                prefill_pred=pred_from_scores(real_scores)
                prefill_correct=(prefill_pred==m["gt"])

                noimage_pred=pred_from_scores(no_scores)

                real_dynamic_margin=dynamic_gt_margin(
                    real_scores,m["gt"]
                )
                no_dynamic_margin=dynamic_gt_margin(
                    no_scores,m["gt"]
                )

                real_fixed=real_graph["fixed_margin"]
                no_fixed=fixed_margin(
                    no_scores,m["gt"],competitor
                )

                output_rn_dynamic_gain=(
                    real_dynamic_margin-no_dynamic_margin
                )
                output_rn_fixed_gain=real_fixed-no_fixed

                # ---------------------------------------------------------
                # Baseline generation (optional)
                # ---------------------------------------------------------
                if a.skip_generation:
                    base_pred=""
                    base_text=""
                    base_correct=None
                else:
                    base_text=base.generate_text(
                        model,
                        processor,
                        rb,
                        max_new_tokens=a.max_new_tokens,
                    )
                    base_pred=traj.normalize_relation(base,base_text)
                    base_correct=(base_pred==m["gt"])

                    generation_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "condition":"baseline",
                        "prediction":base_pred,
                        "correct":base_correct,
                        "text":base_text,
                    })

                alignment_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "real_seq_len":len(rid),
                    "noimage_seq_len":len(nid),
                    "lcs_matches":len(r2n),
                    "selected_targets_requested":len(causal_rows),
                    "selected_targets_aligned":len(targets),
                })

                # ---------------------------------------------------------
                # Per-target leverage + isolated finite patch
                # ---------------------------------------------------------
                for t in targets:
                    predicted=float(a.patch_beta)*float(t["leverage_dot"])

                    if a.skip_isolated_patches:
                        patched_scores=None
                        actual_fixed=float("nan")
                        actual_dynamic=float("nan")
                        patched_pred=""
                        patched_correct=False
                        ratio=float("nan")
                    else:
                        patched_scores=forward_scores_with_patch(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            token_ids=token_ids,
                            patch_map=one_target_patch_map(
                                t,a.patch_beta
                            ),
                        )

                        patched_fixed=fixed_margin(
                            patched_scores,m["gt"],competitor
                        )
                        patched_dynamic=dynamic_gt_margin(
                            patched_scores,m["gt"]
                        )

                        actual_fixed=patched_fixed-real_fixed
                        actual_dynamic=(
                            patched_dynamic-real_dynamic_margin
                        )

                        patched_pred=pred_from_scores(patched_scores)
                        patched_correct=(patched_pred==m["gt"])

                        ratio=(
                            actual_fixed/predicted
                            if abs(predicted)>1e-8
                            else float("nan")
                        )

                    target_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "generation_correct":base_correct,
                        "prefill_prediction":prefill_pred,
                        "prefill_correct":prefill_correct,
                        "noimage_prefill_prediction":noimage_pred,
                        "competitor":competitor,
                        "real_gt_margin":real_dynamic_margin,
                        "real_fixed_comp_margin":real_fixed,
                        "noimage_gt_margin":no_dynamic_margin,
                        "noimage_fixed_comp_margin":no_fixed,
                        "output_rn_margin_gain":output_rn_dynamic_gain,
                        "output_rn_fixed_comp_gain":output_rn_fixed_gain,
                        "target_rank":int(t["rank"]),
                        "causal_text_rank":int(t["causal_text_rank"]),
                        "source_layer":int(t["layer"]),
                        "real_position":int(t["real_position"]),
                        "noimage_position":int(t["noimage_position"]),
                        "token":str(t["token"]),
                        "category":str(t["category"]),
                        "broad_category":str(t["broad_category"]),
                        "rn_norm":float(t["rn_norm"]),
                        "grad_norm":float(t["grad_norm"]),
                        "leverage_dot":float(t["leverage_dot"]),
                        "leverage_cosine":float(t["leverage_cosine"]),
                        "leverage_per_rn_norm":float(t["leverage_per_rn_norm"]),
                        "leverage_per_grad_norm":float(t["leverage_per_grad_norm"]),
                        "patch_beta":float(a.patch_beta),
                        "predicted_fixed_gain":predicted,
                        "actual_fixed_gain":actual_fixed,
                        "actual_dynamic_gain":actual_dynamic,
                        "linearity_ratio":ratio,
                        "patched_prefill_prediction":patched_pred,
                        "patched_prefill_correct":patched_correct,
                    })

                # ---------------------------------------------------------
                # Joint top-K patch
                # ---------------------------------------------------------
                sum_leverage=float(
                    sum(float(t["leverage_dot"]) for t in targets)
                )
                predicted_joint=float(a.patch_beta)*sum_leverage

                jmap=joint_patch_map(targets,a.patch_beta)
                joint_scores=forward_scores_with_patch(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    token_ids=token_ids,
                    patch_map=jmap,
                )

                joint_pred=pred_from_scores(joint_scores)
                joint_prefill_correct=(joint_pred==m["gt"])

                joint_fixed=fixed_margin(
                    joint_scores,m["gt"],competitor
                )
                joint_dynamic=dynamic_gt_margin(
                    joint_scores,m["gt"]
                )

                actual_joint_fixed=joint_fixed-real_fixed
                actual_joint_dynamic=(
                    joint_dynamic-real_dynamic_margin
                )

                needed_to_cross=max(0.0,-real_dynamic_margin)

                if needed_to_cross>EPS:
                    actual_gain_over_deficit=(
                        actual_joint_dynamic/needed_to_cross
                    )
                    predicted_gain_over_deficit=(
                        predicted_joint/needed_to_cross
                    )
                else:
                    actual_gain_over_deficit=float("nan")
                    predicted_gain_over_deficit=float("nan")

                # Optional joint greedy generation.
                if a.skip_generation:
                    joint_gen_pred=""
                    joint_gen_text=""
                    joint_gen_correct=None
                else:
                    joint_gen_pred,joint_gen_text=generate_with_patch(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        patch_map=jmap,
                        max_new_tokens=a.max_new_tokens,
                    )
                    joint_gen_correct=(joint_gen_pred==m["gt"])

                    generation_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "condition":"joint_rn",
                        "prediction":joint_gen_pred,
                        "correct":joint_gen_correct,
                        "text":joint_gen_text,
                    })

                wrong_visual_helped=bool(
                    real_dynamic_margin<0
                    and output_rn_dynamic_gain>0
                )
                wrong_positive_causal_leverage=bool(
                    real_dynamic_margin<0
                    and sum_leverage>0
                )
                wrong_joint_crossed=bool(
                    real_dynamic_margin<0
                    and joint_dynamic>0
                )

                sample_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "generation_prediction":base_pred,
                    "generation_correct":base_correct,
                    "prefill_prediction":prefill_pred,
                    "prefill_correct":prefill_correct,
                    "noimage_prefill_prediction":noimage_pred,
                    "competitor":competitor,
                    "real_gt_margin":real_dynamic_margin,
                    "real_fixed_comp_margin":real_fixed,
                    "noimage_gt_margin":no_dynamic_margin,
                    "noimage_fixed_comp_margin":no_fixed,
                    "output_rn_margin_gain":output_rn_dynamic_gain,
                    "output_rn_fixed_comp_gain":output_rn_fixed_gain,
                    "N_causal_targets":len(targets),
                    "mean_rn_norm":safe_mean(
                        t["rn_norm"] for t in targets
                    ),
                    "mean_leverage_dot":safe_mean(
                        t["leverage_dot"] for t in targets
                    ),
                    "positive_leverage_fraction":float(
                        np.mean([
                            float(t["leverage_dot"])>0
                            for t in targets
                        ])
                    ),
                    "sum_leverage_dot":sum_leverage,
                    "predicted_joint_fixed_gain":predicted_joint,
                    "joint_prefill_prediction":joint_pred,
                    "joint_prefill_correct":joint_prefill_correct,
                    "joint_gt_margin":joint_dynamic,
                    "joint_fixed_comp_margin":joint_fixed,
                    "actual_joint_fixed_gain":actual_joint_fixed,
                    "actual_joint_dynamic_gain":actual_joint_dynamic,
                    "needed_margin_to_cross_zero":needed_to_cross,
                    "predicted_gain_over_deficit":predicted_gain_over_deficit,
                    "actual_gain_over_deficit":actual_gain_over_deficit,
                    "wrong_visual_helped_but_insufficient":wrong_visual_helped,
                    "wrong_positive_causal_leverage":wrong_positive_causal_leverage,
                    "wrong_joint_patch_crossed_zero":wrong_joint_crossed,
                    "joint_generation_prediction":joint_gen_pred,
                    "joint_generation_correct":joint_gen_correct,
                })

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
        # Save raw tables
        # -----------------------------------------------------------------
        target_df=pd.DataFrame(target_rows)
        sample_df=pd.DataFrame(sample_rows)
        align_df=pd.DataFrame(alignment_rows)
        gen_df=pd.DataFrame(generation_rows)

        target_df.to_csv(
            outdir/"per_target_leverage.csv",
            index=False,
        )
        sample_df.to_csv(
            outdir/"sample_decision_summary.csv",
            index=False,
        )
        align_df.to_csv(
            outdir/"alignment_summary.csv",
            index=False,
        )
        gen_df.to_csv(
            outdir/"generation_per_sample.csv",
            index=False,
        )

        # -----------------------------------------------------------------
        # Summaries
        # -----------------------------------------------------------------
        if len(target_df):
            by_prefill=summarize_per_target(
                target_df,"prefill_correct"
            )
            by_prefill.to_csv(
                outdir/"leverage_by_prefill_correctness.csv",
                index=False,
            )

            valid_gen=target_df[
                target_df["generation_correct"].notna()
            ].copy()

            if len(valid_gen):
                by_gen=summarize_per_target(
                    valid_gen,"generation_correct"
                )
            else:
                by_gen=pd.DataFrame()
        else:
            by_prefill=pd.DataFrame()
            by_gen=pd.DataFrame()

        by_gen.to_csv(
            outdir/"leverage_by_generation_correctness.csv",
            index=False,
        )

        sample_by_prefill=summarize_sample_groups(
            sample_df,"prefill_correct"
        ) if len(sample_df) else pd.DataFrame()

        sample_by_prefill.to_csv(
            outdir/"sample_summary_by_prefill_correctness.csv",
            index=False,
        )

        if len(sample_df):
            valid_gen_samples=sample_df[
                sample_df["generation_correct"].notna()
            ].copy()
            sample_by_gen=summarize_sample_groups(
                valid_gen_samples,
                "generation_correct",
            ) if len(valid_gen_samples) else pd.DataFrame()
        else:
            sample_by_gen=pd.DataFrame()

        sample_by_gen.to_csv(
            outdir/"sample_summary_by_generation_correctness.csv",
            index=False,
        )

        wrong_df=sample_df[
            sample_df["real_gt_margin"]<0
        ].copy() if len(sample_df) else pd.DataFrame()

        wrong_df.to_csv(
            outdir/"wrong_case_diagnosis.csv",
            index=False,
        )

        gen_summary=summarize_generation(gen_df)
        gen_summary.to_csv(
            outdir/"generation_summary.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Compact wrong-case diagnostics
        # -------------------------------------------------------------
        if len(wrong_df):
            wrong_n=len(wrong_df)
            visual_help_frac=float(
                np.mean(
                    wrong_df[
                        "wrong_visual_helped_but_insufficient"
                    ].astype(bool)
                )
            )
            pos_lev_frac=float(
                np.mean(
                    wrong_df[
                        "wrong_positive_causal_leverage"
                    ].astype(bool)
                )
            )
            cross_frac=float(
                np.mean(
                    wrong_df[
                        "wrong_joint_patch_crossed_zero"
                    ].astype(bool)
                )
            )
        else:
            wrong_n=0
            visual_help_frac=float("nan")
            pos_lev_frac=float("nan")
            cross_frac=float("nan")

        report=[
            "="*196,
            "RN CAUSAL EVIDENCE vs FINAL DECISION MARGIN",
            "="*196,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested={len(meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"patch_beta={a.patch_beta}",
            f"answer_surface={a.answer_surface}",
            "",
            "PER-TARGET LEVERAGE BY PREFILL CORRECTNESS",
            "-"*196,
            by_prefill.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(by_prefill) else "EMPTY",
            "",
            "SAMPLE-LEVEL DECISION DIAGNOSTIC BY PREFILL CORRECTNESS",
            "-"*196,
            sample_by_prefill.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(sample_by_prefill) else "EMPTY",
            "",
            "WRONG-PREFILL CASE DIAGNOSIS",
            "-"*196,
            f"N wrong by four-way prefill margin       : {wrong_n}",
            f"fraction image improves GT margin > 0   : {visual_help_frac:.4f}",
            f"fraction sum causal RN leverage > 0     : {pos_lev_frac:.4f}",
            f"fraction joint +RN crosses margin zero  : {cross_frac:.4f}",
            "",
            "GENERATION",
            "-"*196,
            gen_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(gen_summary) else "EMPTY / generation skipped",
            "",
            "How to interpret:",
            "  leverage_dot = <hR-hN, d(GT-vs-current-competitor margin)/dh>.",
            "  Positive leverage means more of the already-present RN displacement locally",
            "  pushes the current decision toward GT.",
            "",
            "Strong support for 'present but insufficiently weighted' would look like:",
            "  (1) many wrong samples have output_rn_margin_gain > 0,",
            "  (2) their sum_leverage_dot is also > 0,",
            "  (3) joint +RN produces positive actual margin gain and often crosses zero,",
            "  while RN norms are not dramatically absent relative to correct samples.",
            "",
            "If wrong samples instead have much smaller RN norms and weak/negative leverage,",
            "  the problem is more consistent with evidence formation/quality failure.",
            "",
            "Caution:",
            "  E_c is a local first-order attribution. actual_fixed_gain is the finite",
            "  intervention test and should be preferred when the two disagree.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)

        (outdir/"analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(outdir/"metadata.json",{
            "script":"analyze_rn_causal_evidence_vs_decision_margin_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_requested_N":len(meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "patch_beta":a.patch_beta,
            "answer_surface":a.answer_surface,
            "relation_token_ids":token_ids,
            "target":"t_c = h_real[C,p]-h_noimage[C,q]",
            "gradient_objective":"GT answer logit - highest non-GT answer logit on baseline REAL",
            "leverage":"dot(t_c, gradient_objective_gradient_at_c)",
            "joint_patch":"h[C,p] <- h[C,p] + beta*t_c for selected oracle causal states",
            "alignment":"exact token-ID LCS",
            "oracle_note":"Causal target states originate from prior oracle writer-guided ranking; GT is used in this script to define the diagnostic answer margin.",
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
