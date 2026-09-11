#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_causal_token_self_scale_v1.py

Question
========
We already know that amplifying selected causal text states along their own
sample-specific Image-Gray direction is useful:

    delta_rg(C,p) = h_real[C,p] - h_gray[C,p]

    causal-RG:
        h'[C,p] = h[C,p] + alpha * delta_rg(C,p)

This script tests a simpler alternative:

    SELF SCALE:
        h'[C,p] = factor * h[C,p]

For factor=2:

    h'[C,p] = 2 h[C,p] = h[C,p] + h[C,p]

So this directly tests whether the previous repair effect is merely due to
"making the causal token stronger / larger", rather than specifically adding
the Image-Gray direction.

Conditions
==========
baseline
    untouched REAL generation

self_x{factor}
    multiply every selected causal token state by factor at its selected layer

causal_rg
    reference:
        h'[C,p] = h[C,p] + causal_scale * (h_real[C,p]-h_gray[C,p])

Important
=========
Selected causal states come from the existing ranked causal CSV, so causal-state
selection remains oracle in the same sense as previous experiments.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u eval_causal_token_self_scale_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --self-factors 2 \
  --causal-scale 1 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_causal_token_self_x2_n80_v1 \
  --overwrite

Outputs
=======
generation_per_sample.csv
generation_summary.csv
reconstruction_per_state.csv
reconstruction_summary.csv
selected_causal_states.csv
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
from typing import List

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
        "--self-factors",
        default="2",
        help="Comma-separated multiplicative factors, e.g. 1.25,1.5,2",
    )
    p.add_argument("--causal-scale", type=float, default=1.0)

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
# Baseline REAL/GRAY state capture
# =============================================================================

class StateCapture:
    def __init__(self,decoder_layers,state_layers,prompt_len=None):
        self.states={}
        self.prompt_len=prompt_len
        self.handles=[]

        for L in sorted(set(map(int,state_layers))):
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
def run_state_capture(model,decoder_layers,batch,state_layers):
    cap=StateCapture(decoder_layers,state_layers)
    try:
        kw=dict(batch)
        kw["use_cache"]=False
        _=model(**kw)
        return dict(cap.states)
    finally:
        cap.close()


# =============================================================================
# Intervention hooks
# =============================================================================

class SelfScalePatch:
    """
    Multiply selected block-output token states in-place by factor.

        y[L,p] <- factor * y[L,p]

    Applied only during prompt prefill (sequence length == prompt_len).
    """
    def __init__(self,decoder_layers,positions_by_layer,prompt_len,factor):
        self.decoder_layers=decoder_layers
        self.positions_by_layer=positions_by_layer
        self.prompt_len=int(prompt_len)
        self.factor=float(factor)
        self.handles=[]

    def __enter__(self):
        for L,positions in self.positions_by_layer.items():
            pos=sorted(set(map(int,positions)))

            def make_hook(ps):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    if int(x.shape[1])!=self.prompt_len:
                        return None

                    y=x.clone()
                    for p in ps:
                        if 0<=p<int(y.shape[1]):
                            y[0,p] *= self.factor
                    return traj.replace_first_tensor(out,y)
                return hook

            self.handles.append(
                self.decoder_layers[int(L)].register_forward_hook(
                    make_hook(pos)
                )
            )
        return self

    def __exit__(self,*args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles=[]


class AddVectorPatch:
    """
    Add cached vectors to selected block-output states:

        y[L,p] <- y[L,p] + scale * vec[L,p]
    """
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
                            y[0,p] += self.scale*torch.as_tensor(
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
    self_positions=None,
    self_factor=None,
    vector_patch=None,
    vector_scale=1.0,
):
    prompt_len=int(batch["input_ids"].shape[1])

    self_ctx=(
        SelfScalePatch(
            decoder_layers,self_positions,prompt_len,self_factor
        )
        if self_positions is not None else contextlib.nullcontext()
    )
    vec_ctx=(
        AddVectorPatch(
            decoder_layers,vector_patch,prompt_len,vector_scale
        )
        if vector_patch is not None else contextlib.nullcontext()
    )

    with self_ctx:
        with vec_ctx:
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
# Metrics
# =============================================================================

def build_positions_by_layer(causal_rows):
    out={}
    for r in causal_rows.itertuples():
        L=int(r.source_layer)
        p=int(r.position)
        out.setdefault(L,set()).add(p)
    return {L:sorted(v) for L,v in out.items()}


def build_causal_rg_patch(real_states,gray_states,causal_rows):
    out={}
    for r in causal_rows.itertuples():
        L=int(r.source_layer)
        p=int(r.position)

        if L not in real_states or L not in gray_states:
            continue
        n=min(real_states[L].shape[1],gray_states[L].shape[1])
        if not (0<=p<n):
            continue

        out.setdefault(L,{})[p]=(
            real_states[L][0,p]-gray_states[L][0,p]
        ).astype(np.float32)

    return out


def reconstruction_rows(
    *,
    sid,
    gt,
    condition,
    factor,
    causal_rows,
    real_states,
    gray_states,
    patched_states,
):
    rows=[]

    for r in causal_rows.itertuples():
        L=int(r.source_layer)
        p=int(r.position)

        if L not in patched_states:
            continue

        n=min(
            real_states[L].shape[1],
            gray_states[L].shape[1],
            patched_states[L].shape[1],
        )
        if not (0<=p<n):
            continue

        target=(
            real_states[L][0,p]-gray_states[L][0,p]
        ).astype(np.float32)
        move=(
            patched_states[L][0,p]-real_states[L][0,p]
        ).astype(np.float32)

        tn=float(np.linalg.norm(target))
        mn=float(np.linalg.norm(move))
        if tn<=EPS:
            continue

        dot=float(np.dot(move,target))
        base_h=real_states[L][0,p].astype(np.float32)
        hn=float(np.linalg.norm(base_h))

        rows.append({
            "sid":sid,
            "gt":gt,
            "condition":condition,
            "factor_or_scale":float(factor),
            "causal_rank":int(r.rank),
            "causal_text_rank":int(r.causal_text_rank),
            "causal_layer":L,
            "causal_position":p,
            "causal_token":str(r.token),
            "causal_category":str(r.broad_category),

            "real_state_norm":hn,
            "target_rg_norm":tn,
            "real_over_rg_norm":hn/max(tn,EPS),
            "real_cos_rg":(
                float(np.dot(base_h,target)/(max(hn,EPS)*tn))
            ),

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

    for (cond,val),g in df[df.condition!="baseline"].groupby(
        ["condition","factor_or_scale"],dropna=False
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
            "factor_or_scale":float(val),
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
        ["condition","factor_or_scale"]
    ).reset_index(drop=True)


def summarize_reconstruction(df):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]
    for (cond,val),g in df.groupby(
        ["condition","factor_or_scale"],dropna=False
    ):
        rows.append({
            "condition":cond,
            "factor_or_scale":float(val),
            "N_states":len(g),
            "N_samples":g.sid.nunique(),

            "mean_real_over_rg_norm":safe_mean(g.real_over_rg_norm),
            "median_real_over_rg_norm":safe_median(g.real_over_rg_norm),
            "mean_real_cos_rg":safe_mean(g.real_cos_rg),
            "median_real_cos_rg":safe_median(g.real_cos_rg),

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
        })

    return pd.DataFrame(rows).sort_values(
        ["condition","factor_or_scale"]
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
    factors=parse_floats(a.self_factors)

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
    recon_rows=[]

    try:
        model,processor,decoder_layers,decoder_path,spec=load_model(a,two)
        device=torch.device(a.device)

        for m in tqdm(eval_meta,desc="CAUSAL TOKEN SELF SCALE"):
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

                ids=rb["input_ids"][0].detach().cpu().tolist()
                ids_g=gb["input_ids"][0].detach().cpu().tolist()
                if ids!=ids_g:
                    raise RuntimeError("REAL/GRAY tokenization mismatch")

                real_states=run_state_capture(
                    model,decoder_layers,rb,state_layers
                )
                gray_states=run_state_capture(
                    model,decoder_layers,gb,state_layers
                )

                positions_by_layer=build_positions_by_layer(causal_rows)
                causal_patch=build_causal_rg_patch(
                    real_states,gray_states,causal_rows
                )

                # Baseline
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
                    "factor_or_scale":0.0,
                    "prediction":bpred,
                    "correct":bok,
                    "text":btext,
                })

                # Direct causal-RG reference
                cpred,ctext,cstates=generate_and_capture(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    state_layers=state_layers,
                    max_new_tokens=a.max_new_tokens,
                    vector_patch=causal_patch,
                    vector_scale=a.causal_scale,
                )
                cok=cpred==m["gt"]

                gen_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"causal_rg",
                    "factor_or_scale":float(a.causal_scale),
                    "prediction":cpred,
                    "correct":cok,
                    "text":ctext,
                })

                recon_rows.extend(
                    reconstruction_rows(
                        sid=sid,
                        gt=m["gt"],
                        condition="causal_rg",
                        factor=float(a.causal_scale),
                        causal_rows=causal_rows,
                        real_states=real_states,
                        gray_states=gray_states,
                        patched_states=cstates,
                    )
                )

                # Self scaling
                for factor in factors:
                    pred,text,pstates=generate_and_capture(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        state_layers=state_layers,
                        max_new_tokens=a.max_new_tokens,
                        self_positions=positions_by_layer,
                        self_factor=factor,
                    )
                    ok=pred==m["gt"]

                    gen_rows.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "condition":"self_scale",
                        "factor_or_scale":float(factor),
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
                            condition="self_scale",
                            factor=float(factor),
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
        recon_df=pd.DataFrame(recon_rows)

        gen_df.to_csv(outdir/"generation_per_sample.csv",index=False)
        recon_df.to_csv(outdir/"reconstruction_per_state.csv",index=False)

        gs=summarize_generation(gen_df)
        rs=summarize_reconstruction(recon_df)

        gs.to_csv(outdir/"generation_summary.csv",index=False)
        rs.to_csv(outdir/"reconstruction_summary.csv",index=False)

        report=[
            "="*180,
            "CAUSAL TOKEN SELF-SCALE vs IMAGE-GRAY",
            "="*180,
            f"model={a.model} repo={spec.repo_id}",
            f"N eval={len(eval_meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"self factors={factors}",
            f"causal RG scale={a.causal_scale}",
            "",
            "GENERATION",
            "-"*180,
            gs.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(gs) else "EMPTY",
            "",
            "MOVEMENT RELATIVE TO CAUSAL IMAGE-GRAY",
            "-"*180,
            rs.to_string(
                index=False,float_format=lambda x:f"{x:.4f}"
            ) if len(rs) else "EMPTY",
            "",
            "Interpretation:",
            "  factor=2 means h <- 2h, i.e. add one copy of the current causal-token state itself.",
            "  If self_scale x2 ~= causal_rg behaviorally, the effect may largely be generic state amplification.",
            "  If self_scale x2 << causal_rg or causes C2W damage, the useful intervention is direction-specific rather than simple norm amplification.",
            "  real_cos_rg reports how aligned the original causal state h_real already is with its own Image-Gray vector.",
            "  real_over_rg_norm reports how much larger the whole hidden state is than the Image-Gray displacement.",
        ]

        report_text="\n".join(report)+"\n"
        print(report_text)
        (outdir/"analysis_summary.txt").write_text(
            report_text,encoding="utf-8"
        )

        write_json(outdir/"metadata.json",{
            "script":"eval_causal_token_self_scale_v1.py",
            "model":a.model,
            "repo_id":spec.repo_id,
            "decoder_path":decoder_path,
            "eval_N":len(eval_meta),
            "causal_layers":causal_layers,
            "causal_top_k":a.causal_top_k,
            "self_factors":factors,
            "causal_scale":a.causal_scale,
            "self_intervention":"h[L,p] <- factor * h[L,p]",
            "causal_rg_intervention":"h[L,p] <- h[L,p] + scale*(h_real[L,p]-h_gray[L,p])",
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
