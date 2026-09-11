#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decompose each selected causal token's Real-Gray displacement into:
  full_rg
  decision1  = projection onto grad[GT score - mean wrong score]
  decision3  = projection onto span of three grad[GT score - wrong score]
  orthogonal3 = full_rg - decision3

Decision scores are actual teacher-forced answer-sequence log-probs for:
  left -> " left", right -> " right", above -> " on", below -> " under"

Then directly patch the selected causal states and run model.generate().
"""

from __future__ import annotations
import argparse, contextlib, gc, json, math, random, shutil, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
WORD = {"left":"left", "right":"right", "above":"on", "below":"under"}
EPS = 1e-12


def args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b","qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--ranked-causal", required=True)
    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument("--causal-categories", default="")
    p.add_argument("--scales", default="1")
    p.add_argument("--conditions", default="full_rg,decision1,decision3,orthogonal3")
    p.add_argument("--candidate-prefix", default=" ")
    p.add_argument("--score-mode", default="mean", choices=["mean","sum"])
    p.add_argument("--svd-rcond", type=float, default=1e-6)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager","sdpa","flash_attention_2","none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_layers(s):
    out=set()
    for x in str(s).split(","):
        x=x.strip().upper().replace("L","")
        if not x: continue
        if "-" in x:
            a,b=map(int,x.split("-",1))
            out.update(range(min(a,b),max(a,b)+1))
        else:
            out.add(int(x))
    return sorted(out)


def parse_floats(s): return [float(x) for x in str(s).split(",") if x.strip()]
def parse_strings(s): return [x.strip() for x in str(s).split(",") if x.strip()]
def parse_categories(s):
    z={x.strip() for x in str(s).split(",") if x.strip()}
    return z or None


def safe_mean(xs):
    z=[]
    for x in xs:
        try: v=float(x)
        except: continue
        if math.isfinite(v): z.append(v)
    return float(np.mean(z)) if z else float("nan")


def safe_median(xs):
    z=[]
    for x in xs:
        try: v=float(x)
        except: continue
        if math.isfinite(v): z.append(v)
    return float(np.median(z)) if z else float("nan")


def append_jsonl(path,row):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(dict(row),ensure_ascii=False)+"\n")


def load_meta(a):
    two=base.import_two_object_module()
    prompts=base.load_standard_prompts(Path(a.prompt_jsonl))
    records,_=two.load_records("coco_two",Path(a.data_root),None)
    rec={int(r.sid):r for r in records}
    meta=[]
    for r in records:
        sid=int(r.sid)
        if sid not in prompts: continue
        p=prompts[sid]
        gt=traj.normalize_relation(base,p["answer_raw"])
        if gt not in REL: continue
        meta.append(dict(
            sid=sid,gt=gt,subject=str(p["subject"]),
            reference=str(p["reference"]),question_text=str(p["question_text"])
        ))
    meta=traj.stratified_cap(meta,a.max_samples,a.seed)
    return two,meta,rec


def load_selected(path, allowed, layers, topk, cats):
    d=pd.read_csv(path)
    for c in ["sid","rank","source_layer","position"]:
        d[c]=pd.to_numeric(d[c],errors="raise").astype(int)
    d=d[d.sid.isin(allowed)]
    d=d[d.source_layer.isin(set(layers))]
    d=d[(d.broad_category.astype(str)!="visual") & (d.broad_category.astype(str)!="last")]
    if cats is not None:
        d=d[d.broad_category.astype(str).isin(cats)]
    out=[]
    for sid,g in d.groupby("sid"):
        z=g.sort_values("rank").head(topk).copy()
        z["causal_text_rank"]=np.arange(1,len(z)+1)
        out.append(z)
    return pd.concat(out,ignore_index=True) if out else d.iloc[:0].copy()


def load_model(a,two):
    spec=base.merged_model_specs(two)[a.model]
    cls=getattr(transformers,spec.model_class)
    kw=dict(dtype=base.resolve_dtype(spec.dtype_name),low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,device_map={"":a.device})
    if a.attn_impl!="none": kw["attn_implementation"]=a.attn_impl
    try:
        model=cls.from_pretrained(spec.repo_id,**kw)
    except TypeError:
        kw["torch_dtype"]=kw.pop("dtype")
        model=cls.from_pretrained(spec.repo_id,**kw)
    model.eval()
    proc=AutoProcessor.from_pretrained(spec.repo_id,trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model,proc)
    for p in model.parameters(): p.requires_grad_(False)
    layers,path=base.resolve_decoder_layers(model)
    return model,proc,layers,path,spec


class Capture:
    def __init__(self,layers,which,cpu=True,cut=False):
        self.states={}; self.handles=[]; self.cpu=cpu
        if cut:
            def ch(_m,_i,out):
                x=traj.first_tensor(out)
                y=x.detach().clone().requires_grad_(True)
                self.states[0]=y
                return traj.replace_first_tensor(out,y)
            self.handles.append(layers[0].register_forward_hook(ch))
        for L in sorted(set(which)):
            if cut and L==0: continue
            def mk(layer):
                def h(_m,_i,out):
                    x=traj.first_tensor(out)
                    self.states[layer]=(x.detach().float().cpu().numpy().astype(np.float32)
                                        if self.cpu else x)
                    return None
                return h
            self.handles.append(layers[L].register_forward_hook(mk(L)))
    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception): h.remove()


@torch.inference_mode()
def capture_cpu(model,layers,batch,which):
    c=Capture(layers,which,cpu=True,cut=False)
    try:
        kw=dict(batch); kw["use_cache"]=False
        _=model(**kw)
        return c.states
    finally:
        c.close()


def append_candidate(batch,cids):
    out={}
    plen=int(batch["input_ids"].shape[1])
    cand=torch.tensor(cids,dtype=batch["input_ids"].dtype,
                      device=batch["input_ids"].device).view(1,-1)
    out["input_ids"]=torch.cat([batch["input_ids"],cand],1)
    if "attention_mask" in batch:
        m=batch["attention_mask"]
        out["attention_mask"]=torch.cat(
            [m,torch.ones((m.shape[0],cand.shape[1]),dtype=m.dtype,device=m.device)],1)
    for k,v in batch.items():
        if k in {"input_ids","attention_mask","position_ids","cache_position","labels"}: continue
        out[k]=v
    return out,plen


def score_candidate(model,layers,rb,state_layers,cids,mode):
    batch,plen=append_candidate(rb,cids)
    cap=Capture(layers,state_layers,cpu=False,cut=True)
    try:
        kw=dict(batch); kw["use_cache"]=False; kw["return_dict"]=True
        out=model(**kw)
        vals=[]
        for i,tok in enumerate(cids):
            pos=plen-1+i
            vals.append(torch.log_softmax(out.logits[0,pos].float(),-1)[int(tok)])
        vals=torch.stack(vals)
        score=vals.mean() if mode=="mean" else vals.sum()
        grads=torch.autograd.grad(score,[cap.states[L] for L in state_layers],
                                  retain_graph=False,create_graph=False)
        return dict(
            score=float(score.detach().cpu()),
            token_lps=[float(x) for x in vals.detach().cpu().tolist()],
            grads={L:g.detach().float().cpu().numpy().astype(np.float32)
                   for L,g in zip(state_layers,grads)}
        )
    finally:
        cap.close()


def row_basis(vecs,rcond):
    G=np.stack(vecs,0).astype(np.float64)
    _,S,Vt=np.linalg.svd(G,full_matrices=False)
    if len(S)==0 or S[0]<=EPS:
        return np.zeros((0,G.shape[1]),np.float32),S
    rank=int(np.sum(S>S[0]*rcond))
    return Vt[:rank].astype(np.float32),S.astype(np.float32)


def proj_basis(v,B):
    if B.shape[0]==0: return np.zeros_like(v)
    return (B.T@(B@v)).astype(np.float32)


def proj_vec(v,g):
    den=float(np.dot(g,g))
    return np.zeros_like(v) if den<=EPS else (np.dot(v,g)/den*g).astype(np.float32)


def components(delta,gt,gradmap,rcond):
    wrong=[r for r in REL if r!=gt]
    ggt=gradmap[gt]
    contrasts=[ggt-gradmap[r] for r in wrong]
    gmean=(ggt-np.mean(np.stack([gradmap[r] for r in wrong]),0)).astype(np.float32)
    B,S=row_basis(contrasts,rcond)
    d1=proj_vec(delta,gmean)
    d3=proj_basis(delta,B)
    orth=(delta-d3).astype(np.float32)
    return dict(full_rg=delta,decision1=d1,decision3=d3,orthogonal3=orth,
                gmean=gmean,B=B,S=S,contrasts=contrasts,wrong=wrong)


class Patch:
    def __init__(self,layers,pmap,plen,scale):
        self.layers=layers; self.pmap=pmap; self.plen=plen; self.scale=scale
        self.hs=[]; self.count=defaultdict(int)
    def __enter__(self):
        for L,mp in self.pmap.items():
            def mk(layer,posmap):
                def h(_m,_i,out):
                    x=traj.first_tensor(out)
                    if int(x.shape[1])!=self.plen: return None
                    y=x.clone()
                    for p,v in posmap.items():
                        y[0,int(p)] += self.scale*torch.as_tensor(v,device=y.device,dtype=y.dtype)
                        self.count[(layer,int(p))]+=1
                    return traj.replace_first_tensor(out,y)
                return h
            self.hs.append(self.layers[L].register_forward_hook(mk(L,dict(mp))))
        return self
    def validate(self):
        exp={(int(L),int(p)) for L,mp in self.pmap.items() for p in mp}
        bad=[(k,self.count.get(k,0)) for k in exp if self.count.get(k,0)!=1]
        if bad: raise RuntimeError(f"patch count mismatch: {bad[:10]}")
    def __exit__(self,*_):
        for h in reversed(self.hs):
            with contextlib.suppress(Exception): h.remove()


@torch.inference_mode()
def gen_clean(model,proc,batch,n):
    text=base.generate_text(model,proc,batch,max_new_tokens=n)
    return traj.normalize_relation(base,text),text


@torch.inference_mode()
def gen_patch(model,proc,layers,batch,pmap,scale,n):
    with Patch(layers,pmap,int(batch["input_ids"].shape[1]),scale) as p:
        text=base.generate_text(model,proc,batch,max_new_tokens=n)
    p.validate()
    return traj.normalize_relation(base,text),text


def gen_summary(df):
    if len(df)==0: return pd.DataFrame()
    b=df[df.condition=="baseline"].drop_duplicates("sid").set_index("sid")
    rows=[]
    for (cond,scale),g in df[df.condition!="baseline"].groupby(["condition","scale"]):
        x=g.set_index("sid"); common=sorted(set(b.index)&set(x.index))
        bc=b.loc[common,"correct"].astype(bool).to_numpy()
        pc=x.loc[common,"correct"].astype(bool).to_numpy()
        bp=b.loc[common,"prediction"].astype(str).to_numpy()
        pp=x.loc[common,"prediction"].astype(str).to_numpy()
        w2c=int(np.sum((~bc)&pc)); c2w=int(np.sum(bc&(~pc)))
        rows.append(dict(
            condition=cond,scale=float(scale),N=len(common),
            baseline_accuracy=float(bc.mean()),patched_accuracy=float(pc.mean()),
            gain=float(pc.mean()-bc.mean()),wrong_to_correct=w2c,
            correct_to_wrong=c2w,net=w2c-c2w,changed=int(np.sum(bp!=pp)),
            wrong_N=int((~bc).sum()),
            repair_rate_on_wrong=float(w2c/max(1,int((~bc).sum()))),
            preserve_rate_on_correct=float(1-c2w/max(1,int(bc.sum())))
        ))
    return pd.DataFrame(rows).sort_values(["scale","condition"])


def geom_summary(df):
    rows=[]
    for name,g in [("all",df),("baseline_correct",df[df.baseline_correct]),
                   ("baseline_wrong",df[~df.baseline_correct])]:
        if len(g)==0: continue
        row=dict(scope=name,N_states=len(g),N_samples=g.sid.nunique())
        for c in [
            "decision1_norm_fraction","decision1_energy_fraction",
            "decision3_norm_fraction","decision3_energy_fraction",
            "orthogonal3_norm_fraction","decision_subspace_rank",
            "cos_RG_vs_gmean","linear_margin_full",
            "linear_margin_decision1","linear_margin_decision3",
            "linear_margin_orthogonal3"
        ]:
            row["mean_"+c]=safe_mean(g[c]); row["median_"+c]=safe_median(g[c])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a=args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    clayers=parse_layers(a.causal_layers)
    scales=parse_floats(a.scales)
    conds=parse_strings(a.conditions)
    valid={"full_rg","decision1","decision3","orthogonal3"}
    if set(conds)-valid: raise ValueError(f"bad conditions {set(conds)-valid}")
    cats=parse_categories(a.causal_categories)

    outdir=Path(a.output_dir)
    if a.overwrite and outdir.exists(): shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()): raise RuntimeError(f"nonempty {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)
    err=outdir/"errors.jsonl"

    two,meta,rec=load_meta(a)
    rank_sids=set(pd.read_csv(a.ranked_causal,usecols=["sid"]).sid.astype(int))
    ev=[m for m in meta if int(m["sid"]) in rank_sids]
    if a.eval_max_samples>0:
        ev=traj.stratified_cap(ev,a.eval_max_samples,a.seed+1)
    selected=load_selected(Path(a.ranked_causal),{int(m["sid"]) for m in ev},
                           clayers,a.causal_top_k,cats)
    bysid={int(s):g.copy() for s,g in selected.groupby("sid")}
    selected.to_csv(outdir/"selected_causal_text_states.csv",index=False)

    model=proc=None
    try:
        model,proc,layers,dpath,spec=load_model(a,two)
        tok=getattr(proc,"tokenizer",proc)
        ctext={r:a.candidate_prefix+WORD[r] for r in REL}
        cids={r:[int(x) for x in tok.encode(ctext[r],add_special_tokens=False)] for r in REL}
        print("Candidate tokenization:")
        for r in REL:
            print(r,repr(ctext[r]),cids[r],
                  repr(tok.decode(cids[r],skip_special_tokens=False)))

        geom_rows=[]; cand_rows=[]; gen_rows=[]
        device=torch.device(a.device)

        for m in tqdm(ev,desc="LOGIT-SENSITIVE RG"):
            sid=int(m["sid"])
            if sid not in bysid: continue
            real=gray=rb=gb=None
            try:
                cr=bysid[sid].sort_values(["causal_text_rank","rank"])
                gt=m["gt"]
                state_layers=sorted(set(cr.source_layer.astype(int)))

                real=base.record_image(rec[sid])
                if hasattr(real,"convert"): real=real.convert("RGB")
                gray=dyn.make_gray_image(real,a.gray_value)
                rb=base.make_question_batch(processor=proc,image=real,
                    question_text=m["question_text"],device=device)
                gb=base.make_question_batch(processor=proc,image=gray,
                    question_text=m["question_text"],device=device)

                hr=capture_cpu(model,layers,rb,state_layers)
                hg=capture_cpu(model,layers,gb,state_layers)

                bp,bt=gen_clean(model,proc,rb,a.max_new_tokens)
                bok=(bp==gt)
                gen_rows.append(dict(sid=sid,gt=gt,condition="baseline",scale=0.,
                                     prediction=bp,correct=bok,text=bt))

                rr={}
                with torch.enable_grad():
                    for r in REL:
                        rr[r]=score_candidate(model,layers,rb,state_layers,cids[r],a.score_mode)

                scores={r:rr[r]["score"] for r in REL}
                wrong=[scores[r] for r in REL if r!=gt]
                crow=dict(sid=sid,gt=gt,baseline_prediction=bp,baseline_correct=bok,
                           teacher_forced_prediction=max(REL,key=lambda r:scores[r]),
                           gt_score=scores[gt],
                           gt_minus_mean_wrong=scores[gt]-float(np.mean(wrong)),
                           gt_minus_max_wrong=scores[gt]-float(np.max(wrong)))
                for r in REL:
                    crow["score_"+r]=scores[r]
                    crow["candidate_ids_"+r]=",".join(map(str,cids[r]))
                cand_rows.append(crow)

                pmaps={c:defaultdict(dict) for c in conds}

                for row in cr.itertuples():
                    C=int(row.source_layer); p=int(row.position)
                    delta=(hr[C][0,p]-hg[C][0,p]).astype(np.float32)
                    gradmap={r:rr[r]["grads"][C][0,p].astype(np.float32) for r in REL}
                    cp=components(delta,gt,gradmap,a.svd_rcond)

                    rn=float(np.linalg.norm(delta))
                    n1=float(np.linalg.norm(cp["decision1"]))
                    n3=float(np.linalg.norm(cp["decision3"]))
                    no=float(np.linalg.norm(cp["orthogonal3"]))
                    gm=cp["gmean"]
                    gr=dict(
                        sid=sid,gt=gt,baseline_prediction=bp,baseline_correct=bok,
                        causal_text_rank=int(row.causal_text_rank),global_rank=int(row.rank),
                        causal_layer=C,position=p,token=str(row.token),
                        category=str(row.category),broad_category=str(row.broad_category),
                        RG_norm=rn,
                        decision1_norm=n1,
                        decision1_norm_fraction=n1/rn if rn>EPS else np.nan,
                        decision1_energy_fraction=(n1/rn)**2 if rn>EPS else np.nan,
                        decision3_norm=n3,
                        decision3_norm_fraction=n3/rn if rn>EPS else np.nan,
                        decision3_energy_fraction=(n3/rn)**2 if rn>EPS else np.nan,
                        orthogonal3_norm=no,
                        orthogonal3_norm_fraction=no/rn if rn>EPS else np.nan,
                        decision_subspace_rank=int(cp["B"].shape[0]),
                        cos_RG_vs_gmean=(
                            float(np.dot(delta,gm)/(np.linalg.norm(delta)*np.linalg.norm(gm)))
                            if np.linalg.norm(delta)>EPS and np.linalg.norm(gm)>EPS else np.nan
                        ),
                        linear_margin_full=float(np.dot(delta,gm)),
                        linear_margin_decision1=float(np.dot(cp["decision1"],gm)),
                        linear_margin_decision3=float(np.dot(cp["decision3"],gm)),
                        linear_margin_orthogonal3=float(np.dot(cp["orthogonal3"],gm)),
                    )
                    for i,s in enumerate(cp["S"][:3],1): gr[f"sv{i}"]=float(s)
                    for wr,gcontrast in zip(cp["wrong"],cp["contrasts"]):
                        gr[f"full_margin_vs_{wr}"]=float(np.dot(delta,gcontrast))
                        gr[f"decision3_margin_vs_{wr}"]=float(np.dot(cp["decision3"],gcontrast))
                        gr[f"orthogonal3_margin_vs_{wr}"]=float(np.dot(cp["orthogonal3"],gcontrast))
                    geom_rows.append(gr)

                    for cond in conds:
                        pmaps[cond][C][p]=cp[cond]

                for cond in conds:
                    pmap={L:dict(x) for L,x in pmaps[cond].items()}
                    for alpha in scales:
                        pred,text=gen_patch(model,proc,layers,rb,pmap,alpha,a.max_new_tokens)
                        ok=(pred==gt)
                        gen_rows.append(dict(
                            sid=sid,gt=gt,condition=cond,scale=alpha,
                            prediction=pred,correct=ok,text=text,
                            baseline_prediction=bp,baseline_correct=bok,
                            wrong_to_correct=(not bok and ok),
                            correct_to_wrong=(bok and not ok)
                        ))

            except Exception as e:
                append_jsonl(err,dict(
                    sid=sid,error=f"{type(e).__name__}: {e}",
                    traceback_tail=traceback.format_exc().splitlines()[-25:]
                ))
                tqdm.write(f"[ERROR] sid={sid}: {type(e).__name__}: {e}")
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        gd=pd.DataFrame(geom_rows); cd=pd.DataFrame(cand_rows); ed=pd.DataFrame(gen_rows)
        gs=geom_summary(gd); es=gen_summary(ed)

        gd.to_csv(outdir/"component_geometry_per_state.csv",index=False)
        gs.to_csv(outdir/"component_geometry_summary.csv",index=False)
        cd.to_csv(outdir/"candidate_scores_per_sample.csv",index=False)
        ed.to_csv(outdir/"generation_per_sample.csv",index=False)
        es.to_csv(outdir/"generation_summary.csv",index=False)

        relout=[]
        for r,g in ed.groupby("gt"):
            z=gen_summary(g)
            if len(z):
                z.insert(0,"gt",r); relout.append(z)
        (pd.concat(relout,ignore_index=True) if relout else pd.DataFrame()).to_csv(
            outdir/"generation_by_relation.csv",index=False)

        report=[]
        report += ["="*170,"IMAGE-GRAY -> FINAL-ANSWER-SENSITIVE COMPONENT","="*170]
        report += [f"completed N={ed[ed.condition=='baseline'].sid.nunique() if len(ed) else 0}",
                   "", "GEOMETRY","-"*170,
                   gs.to_string(index=False,float_format=lambda x:f"{x:.5f}") if len(gs) else "EMPTY",
                   "", "GENERATION","-"*170,
                   es.to_string(index=False,float_format=lambda x:f"{x:.4f}") if len(es) else "EMPTY",
                   "",
                   "Main test: decision3 ~= full_rg AND orthogonal3 ~= baseline, "
                   "especially when decision3_norm_fraction is small."]
        txt="\n".join(report)+"\n"
        print(txt)
        (outdir/"analysis_summary.txt").write_text(txt,encoding="utf-8")

        write_json(outdir/"metadata.json",dict(
            script="eval_causal_rg_logit_sensitive_decomposition_v1.py",
            model=a.model,repo_id=spec.repo_id,decoder_path=dpath,
            ranked_causal=str(a.ranked_causal),causal_layers=clayers,
            causal_top_k=a.causal_top_k,conditions=conds,scales=scales,
            candidate_text=ctext,candidate_ids=cids,score_mode=a.score_mode,
            decision1="projection onto grad(GT score - mean wrong score)",
            decision3="projection onto span of three grad(GT score - wrong score)",
            orthogonal3="full_rg - decision3",
            oracle=True
        ))
    finally:
        if model is not None: del model
        if proc is not None: del proc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=="__main__":
    main()
