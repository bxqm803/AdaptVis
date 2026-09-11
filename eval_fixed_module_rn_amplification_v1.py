#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_fixed_module_rn_amplification_v1.py

Goal
====
Test whether the causal-token formation analysis can be turned into a NON-ORACLE
accuracy improvement.

The intervention never uses:
  - GT relation
  - oracle causal-token locations
  - relation routing
  - trained selector
  - spatial-head labels

For each test sample, run:
    REAL image
    NoImage text-only control

At a FIXED upstream module/layer L, cache its outputs:

    X_R[L,p]     = module output on REAL at aligned text position p
    X_N[L,q(p)]  = module output on NoImage at the matched token q(p)

Define the sample-specific image-induced module displacement:

    dX[L,p] = X_R[L,p] - X_N[L,q(p)]

Then, during a fresh REAL generation, amplify it:

    X'[L,p] = X[L,p] + alpha * dX[L,p]

for all REAL/NoImage-aligned text/structural positions, excluding prompt-last
by default.

Supported module types
======================
A:L   = whole self-attention module output at decoder layer L
M:L   = whole MLP output at decoder layer L

Examples:
    A:23
    M:20
    M:23
    M:25

These are fixed globally for every sample. The direction dX is sample-specific,
so no LEFT/RIGHT/ABOVE/BELOW routing is required.

Why this test
=============
If causal-token RN signal is naturally formed through a fixed transformation
stage, then strengthening the REAL-NoImage displacement at that stage may let
the downstream network naturally reform stronger causal states.

This is different from:
  - directly steering oracle causal tokens;
  - enhancing fixed spatial heads;
  - copying a fixed relation direction.

Default quick grid
==================
The defaults are motivated by the preliminary N=10 mediation scan:
    A23
    M20
    M23
    M25

with:
    alpha = 0.25, 0.5, 1.0

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_fixed_module_rn_amplification_v1.py \
  --model qwen-3b \
  --attention-layers 23 \
  --mlp-layers 20,23,25 \
  --alphas 0.25,0.5,1 \
  --eval-max-samples 20 \
  --output-dir output/qwen3b_fixed_module_rn_n20_v1 \
  --overwrite

Then N=80:
CUDA_VISIBLE_DEVICES=0 python -u eval_fixed_module_rn_amplification_v1.py \
  --model qwen-3b \
  --attention-layers 23 \
  --mlp-layers 20,23,25 \
  --alphas 0.25,0.5,1 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_fixed_module_rn_n80_v1 \
  --overwrite

Optional broader grid:
    --attention-layers 18,19,21,23,24,26
    --mlp-layers 18,19,20,21,22,23,24,25,26

Optional bundle syntax:
    --bundles "A23+M25;A23+M23;M20+M23"

Bundle note
===========
For bundles, every dX is cached from the unmodified REAL and NoImage baseline
trajectories, then added during one fresh REAL generation. Therefore bundles
can show nonlinear interaction/overshoot. Single-module conditions are the
primary experiment.

Outputs
=======
generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
module_delta_per_sample.csv
module_delta_summary.csv
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
import re
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

    p.add_argument(
        "--attention-layers",
        default="23",
        help="Comma/range list of whole-attention layers tested individually.",
    )
    p.add_argument(
        "--mlp-layers",
        default="20,23,25",
        help="Comma/range list of MLP layers tested individually.",
    )
    p.add_argument(
        "--alphas",
        default="0.25,0.5,1.0",
        help="Extra Real-NoImage module-displacement coefficients.",
    )
    p.add_argument(
        "--bundles",
        default="",
        help=(
            'Optional semicolon-separated bundles, e.g. '
            '"A23+M25;A23+M23;M20+M23".'
        ),
    )

    p.add_argument(
        "--include-last",
        action="store_true",
        help="Also amplify prompt-last. Default excludes it to avoid direct answer-stage shortcut.",
    )
    p.add_argument(
        "--exclude-relation-words",
        action="store_true",
        help="Exclude prompt occurrences of left/right/above/below/on/under from patch positions.",
    )
    p.add_argument(
        "--exclude-object-tokens",
        action="store_true",
        help="Exclude subject/reference text-token positions from patch positions.",
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


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_bundles(text: str) -> List[Tuple[str, Tuple[Tuple[str,int],...]]]:
    """
    "A23+M25;A23+M23" ->
      [("A23+M25", (("attention",23),("mlp",25))), ...]
    """
    out=[]
    if not str(text).strip():
        return out

    for raw in str(text).split(";"):
        raw=raw.strip()
        if not raw:
            continue

        nodes=[]
        canonical=[]
        for part in raw.split("+"):
            part=part.strip().upper().replace(":","")
            m=re.fullmatch(r"([AM])L?(\d+)",part)
            if not m:
                raise ValueError(
                    f"Bad bundle node {part!r}; use A23 or M25"
                )
            kind="attention" if m.group(1)=="A" else "mlp"
            L=int(m.group(2))
            nodes.append((kind,L))
            canonical.append(("A" if kind=="attention" else "M")+str(L))

        nodes=tuple(nodes)
        label="+".join(canonical)
        out.append((label,nodes))
    return out


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


# =============================================================================
# NoImage / token alignment
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


def tokenizer_of(processor):
    return getattr(processor,"tokenizer",processor)


def all_subseq_positions(ids: List[int],pat: List[int]) -> List[int]:
    if not pat or len(pat)>len(ids):
        return []
    out=[]
    n=len(pat)
    for i in range(len(ids)-n+1):
        if ids[i:i+n]==pat:
            out.extend(range(i,i+n))
    return sorted(set(out))


def positions_for_text(processor,ids,text):
    tok=tokenizer_of(processor)
    out=set()
    for s in (str(text)," "+str(text)):
        pat=list(map(int,tok.encode(s,add_special_tokens=False)))
        out.update(all_subseq_positions(ids,pat))
    return sorted(out)


def build_patch_position_map(
    *,
    processor,
    real_ids,
    no_ids,
    r2n,
    subject,
    reference,
    include_last,
    exclude_relation_words,
    exclude_object_tokens,
):
    bad=set()

    if exclude_object_tokens:
        bad.update(positions_for_text(processor,real_ids,subject))
        bad.update(positions_for_text(processor,real_ids,reference))

    if exclude_relation_words:
        for w in ("left","right","above","below","on","under"):
            bad.update(positions_for_text(processor,real_ids,w))

    out={}
    for p,q in sorted(r2n.items()):
        p,q=int(p),int(q)

        if not include_last:
            if p==len(real_ids)-1 or q==len(no_ids)-1:
                continue

        if p in bad:
            continue

        if not (
            0<=p<len(real_ids)
            and 0<=q<len(no_ids)
            and int(real_ids[p])==int(no_ids[q])
        ):
            continue

        out[p]=q

    return out


# =============================================================================
# Module helpers / baseline capture
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
    raise RuntimeError(f"Unsupported output type: {type(output).__name__}")


def replace_tensor_output(original_output,tensor):
    if torch.is_tensor(original_output):
        return tensor
    if isinstance(original_output,tuple):
        return (tensor,*original_output[1:])
    if isinstance(original_output,list):
        return [tensor,*original_output[1:]]
    raise RuntimeError(f"Unsupported output type: {type(original_output).__name__}")


def resolve_module(decoder_layers,kind,L):
    layer=decoder_layers[int(L)]
    if kind=="attention":
        return spatialscan.resolve_attn(layer)
    if kind=="mlp":
        return resolve_mlp(layer)
    raise ValueError(kind)


class ModuleCapture:
    def __init__(self,decoder_layers,nodes):
        self.outputs={}
        self.handles=[]

        for kind,L in nodes:
            module=resolve_module(decoder_layers,kind,L)
            key=(str(kind),int(L))

            def make_hook(k):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)
                    self.outputs[k]=(
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook

            self.handles.append(
                module.register_forward_hook(make_hook(key))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def capture_module_outputs(model,decoder_layers,batch,nodes):
    cap=ModuleCapture(decoder_layers,nodes)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True
        _=model(**kw)

        missing=[node for node in nodes if node not in cap.outputs]
        if missing:
            raise RuntimeError(f"Missing module captures: {missing}")
        return dict(cap.outputs)
    finally:
        cap.close()


# =============================================================================
# RN patch map / generation intervention
# =============================================================================

def build_delta_for_node(
    *,
    node,
    real_cache,
    no_cache,
    position_map,
):
    kind,L=node
    R=real_cache[(kind,L)]
    N=no_cache[(kind,L)]

    by_pos={}
    norms=[]

    for p,q in position_map.items():
        if not (
            0<=p<R.shape[1]
            and 0<=q<N.shape[1]
        ):
            continue

        d=(R[0,p]-N[0,q]).astype(np.float32)
        by_pos[int(p)]=d
        norms.append(float(np.linalg.norm(d)))

    return by_pos,norms


class MultiModuleRNAmplifier:
    """
    Add cached baseline dX = X_R - X_N to selected module outputs during REAL
    prompt prefill. Decode-token steps are untouched.
    """
    def __init__(
        self,
        *,
        decoder_layers,
        node_delta_maps,
        prompt_len,
        alpha,
    ):
        self.handles=[]
        self.prompt_len=int(prompt_len)
        self.alpha=float(alpha)

        for (kind,L),by_pos in node_delta_maps.items():
            module=resolve_module(decoder_layers,kind,L)

            def make_hook(pos_map):
                def hook(_m,_inp,out):
                    x=tensor_from_output(out)

                    # Patch prompt prefill only.
                    if int(x.shape[1])!=self.prompt_len:
                        return None

                    y=x.clone()
                    for p,vec in pos_map.items():
                        p=int(p)
                        if 0<=p<int(y.shape[1]):
                            y[0,p]+=self.alpha*torch.as_tensor(
                                vec,
                                device=y.device,
                                dtype=y.dtype,
                            )
                    return replace_tensor_output(out,y)
                return hook

            self.handles.append(
                module.register_forward_hook(
                    make_hook(dict(by_pos))
                )
            )

    def __enter__(self):
        return self

    def __exit__(self,*args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


@torch.inference_mode()
def generate_condition(
    *,
    model,
    processor,
    decoder_layers,
    real_batch,
    max_new_tokens,
    node_delta_maps=None,
    alpha=0.0,
):
    prompt_len=int(real_batch["input_ids"].shape[1])

    ctx=(
        MultiModuleRNAmplifier(
            decoder_layers=decoder_layers,
            node_delta_maps=node_delta_maps,
            prompt_len=prompt_len,
            alpha=alpha,
        )
        if node_delta_maps else contextlib.nullcontext()
    )

    with ctx:
        text=base.generate_text(
            model,
            processor,
            real_batch,
            max_new_tokens=max_new_tokens,
        )

    pred=traj.normalize_relation(base,text)
    return pred,text


# =============================================================================
# Condition construction
# =============================================================================

def node_label(node):
    kind,L=node
    return ("A" if kind=="attention" else "M")+str(int(L))


def build_conditions(attention_layers,mlp_layers,bundles):
    """
    Returns:
      [(label, tuple(nodes)), ...]
    """
    out=[]

    for L in attention_layers:
        out.append((f"A{L}",(("attention",int(L)),)))

    for L in mlp_layers:
        out.append((f"M{L}",(("mlp",int(L)),)))

    out.extend(bundles)

    # Dedupe by label while preserving order.
    seen=set()
    clean=[]
    for label,nodes in out:
        if label in seen:
            continue
        seen.add(label)
        clean.append((label,tuple(nodes)))
    return clean


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(df):
    if len(df)==0:
        return pd.DataFrame()

    b=df[df["condition"]=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]

    for (cond,alpha),g in df[df["condition"]!="baseline"].groupby(
        ["condition","alpha"],
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
            "condition":str(cond),
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
        ["patched_accuracy","condition","alpha"],
        ascending=[False,True,True],
    ).reset_index(drop=True)


def summarize_by_relation(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (cond,alpha,gt),g in df.groupby(
        ["condition","alpha","gt"],
        dropna=False,
    ):
        rows.append({
            "condition":str(cond),
            "alpha":float(alpha),
            "relation":str(gt),
            "N":len(g),
            "accuracy":float(g["correct"].astype(bool).mean()),
        })

    return pd.DataFrame(rows).sort_values(
        ["condition","alpha","relation"]
    ).reset_index(drop=True)


def summarize_delta_stats(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (node,base_correct),g in df.groupby(
        ["node","baseline_correct"],
        dropna=False,
    ):
        rows.append({
            "node":str(node),
            "baseline_correct":bool(base_correct),
            "N_samples":g["sid"].nunique(),
            "mean_aligned_positions":safe_mean(g["aligned_positions"]),
            "mean_delta_norm":safe_mean(g["mean_delta_norm"]),
            "median_delta_norm":safe_median(g["mean_delta_norm"]),
            "mean_max_delta_norm":safe_mean(g["max_delta_norm"]),
        })

    return pd.DataFrame(rows).sort_values(
        ["node","baseline_correct"]
    ).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    attention_layers=parse_layers(a.attention_layers)
    mlp_layers=parse_layers(a.mlp_layers)
    alphas=parse_floats(a.alphas)
    bundles=parse_bundles(a.bundles)

    conditions=build_conditions(
        attention_layers,
        mlp_layers,
        bundles,
    )

    all_nodes=sorted(
        {
            node
            for _label,nodes in conditions
            for node in nodes
        },
        key=lambda x:(x[1],x[0]),
    )

    outdir=Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)
    error_path=outdir/"errors.jsonl"

    two,meta,rec_by_sid=load_data(a)

    model=processor=None
    generation_rows=[]
    alignment_rows=[]
    delta_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        n_layers=len(decoder_layers)
        bad=[L for _kind,L in all_nodes if not 0<=L<n_layers]
        if bad:
            raise ValueError(
                f"Requested module layers outside 0..{n_layers-1}: {sorted(set(bad))}"
            )

        print("="*180)
        print("FIXED-MODULE REAL-NoImage AMPLIFICATION")
        print("="*180)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(meta)}")
        print(f"conditions={[label for label,_ in conditions]}")
        print(f"alphas={alphas}")
        print(f"include_last={a.include_last}")
        print(f"exclude_relation_words={a.exclude_relation_words}")
        print(f"exclude_object_tokens={a.exclude_object_tokens}")
        print()

        for m in tqdm(meta,desc="FIXED MODULE RN"):
            sid=int(m["sid"])
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

                position_map=build_patch_position_map(
                    processor=processor,
                    real_ids=rid,
                    no_ids=nid,
                    r2n=r2n,
                    subject=m["subject"],
                    reference=m["reference"],
                    include_last=a.include_last,
                    exclude_relation_words=a.exclude_relation_words,
                    exclude_object_tokens=a.exclude_object_tokens,
                )
                if not position_map:
                    raise RuntimeError("No aligned patch positions")

                alignment_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "real_seq_len":len(rid),
                    "noimage_seq_len":len(nid),
                    "lcs_matches":len(r2n),
                    "patch_positions":len(position_map),
                    "patch_fraction_real":len(position_map)/max(len(rid),1),
                })

                # Baseline module outputs on both branches.
                real_cache=capture_module_outputs(
                    model,
                    decoder_layers,
                    rb,
                    all_nodes,
                )
                no_cache=capture_module_outputs(
                    model,
                    decoder_layers,
                    nb,
                    all_nodes,
                )

                node_delta_maps={}
                node_norms={}

                for node in all_nodes:
                    by_pos,norms=build_delta_for_node(
                        node=node,
                        real_cache=real_cache,
                        no_cache=no_cache,
                        position_map=position_map,
                    )
                    if not by_pos:
                        raise RuntimeError(
                            f"No aligned deltas for node {node_label(node)}"
                        )
                    node_delta_maps[node]=by_pos
                    node_norms[node]=norms

                # Baseline generation.
                bpred,btext=generate_condition(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    real_batch=rb,
                    max_new_tokens=a.max_new_tokens,
                )
                bok=bpred==m["gt"]

                generation_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"baseline",
                    "alpha":0.0,
                    "prediction":bpred,
                    "correct":bok,
                    "text":btext,
                })

                # Delta geometry stratified later by baseline correctness.
                for node,norms in node_norms.items():
                    delta_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "baseline_prediction":bpred,
                        "baseline_correct":bok,
                        "node":node_label(node),
                        "module_type":node[0],
                        "layer":int(node[1]),
                        "aligned_positions":len(norms),
                        "mean_delta_norm":float(np.mean(norms)),
                        "median_delta_norm":float(np.median(norms)),
                        "max_delta_norm":float(np.max(norms)),
                    })

                # Fixed module / bundle conditions.
                for label,nodes in conditions:
                    maps={
                        node:node_delta_maps[node]
                        for node in nodes
                    }

                    for alpha in alphas:
                        pred,text=generate_condition(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            real_batch=rb,
                            max_new_tokens=a.max_new_tokens,
                            node_delta_maps=maps,
                            alpha=alpha,
                        )

                        generation_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":label,
                            "alpha":float(alpha),
                            "prediction":pred,
                            "correct":pred==m["gt"],
                            "text":text,
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
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gen_df=pd.DataFrame(generation_rows)
        align_df=pd.DataFrame(alignment_rows)
        delta_df=pd.DataFrame(delta_rows)

        gen_df.to_csv(
            outdir/"generation_per_sample.csv",
            index=False,
        )
        align_df.to_csv(
            outdir/"alignment_summary.csv",
            index=False,
        )
        delta_df.to_csv(
            outdir/"module_delta_per_sample.csv",
            index=False,
        )

        gen_summary=summarize_generation(gen_df)
        rel_summary=summarize_by_relation(gen_df)
        delta_summary=summarize_delta_stats(delta_df)

        gen_summary.to_csv(
            outdir/"generation_summary.csv",
            index=False,
        )
        rel_summary.to_csv(
            outdir/"generation_by_relation.csv",
            index=False,
        )
        delta_summary.to_csv(
            outdir/"module_delta_summary.csv",
            index=False,
        )

        report=[
            "="*190,
            "FIXED-MODULE REAL-NoImage AMPLIFICATION",
            "="*190,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested={len(meta)}",
            f"conditions={[label for label,_ in conditions]}",
            f"alphas={alphas}",
            f"include_last={a.include_last}",
            "",
            "GENERATION",
            "-"*190,
            gen_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(gen_summary) else "EMPTY",
            "",
            "MODULE RN NORM BY BASELINE CORRECTNESS",
            "-"*190,
            delta_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(delta_summary) else "EMPTY",
            "",
            "Interpretation:",
            "  A23 means: at every aligned non-last text/structural position,",
            "      AttnOut23 <- AttnOut23 + alpha*(AttnOut23_REAL-AttnOut23_NoImage).",
            "  M25 is the analogous intervention on the MLP output.",
            "  No GT, causal-token location, relation router, or trained selector is used.",
            "",
            "What counts as success:",
            "  gain > 0 with W2C > C2W on a fixed module/layer.",
            "  Prefer a moderate alpha whose gain survives N80/all440.",
            "",
            "If all single modules are <= baseline:",
            "  causal formation may be mechanistically distributed but not safely amplifiable",
            "  at a whole-module granularity; do not force a positive method claim.",
        ]
        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(outdir/"metadata.json",{
            "script":"eval_fixed_module_rn_amplification_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_requested_N":len(meta),
            "attention_layers":attention_layers,
            "mlp_layers":mlp_layers,
            "alphas":alphas,
            "conditions":{
                label:[
                    {"module_type":kind,"layer":L}
                    for kind,L in nodes
                ]
                for label,nodes in conditions
            },
            "include_last":a.include_last,
            "exclude_relation_words":a.exclude_relation_words,
            "exclude_object_tokens":a.exclude_object_tokens,
            "patch_rule":"X <- X + alpha*(X_REAL - X_NoImage) at exact-LCS aligned positions",
            "oracle_usage":"none at test/intervention time; GT used only to compute reported accuracy",
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
