#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline-only failure localization for COCO two-object spatial reasoning.

No intervention.  On held-out samples, trace:
  early direction-head ensemble -> prompt-last block states ->
  L26/L27 object->last writes -> final block -> final norm -> LM 4-way -> generation.

Receiver write is the actual post-W_O contribution:
  c_h = W_O^h sum_{s in object} A_h[last,s] V_h[s]

Uses TRAIN-only cosine codebooks.  For L26/L27, reports both selected receiver
bundle write and ALL-head object->last write so an incomplete shortlist is not
mistaken for information loss.
"""
from __future__ import annotations
import argparse, contextlib, csv, gc, importlib, json, random, re, shutil, traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

RELATIONS=("left","right","above","below")
RID={r:i for i,r in enumerate(RELATIONS)}
OPPOSITE={"left":"right","right":"left","above":"below","below":"above"}
EPS=1e-12
DEFAULT_PROMPT=("Determine the spatial relation of the {subject} to the {reference} "
                "in the image. Answer with left, right, above, or below.")
DEFAULT_RECEIVERS="26:4,26:2,26:6,26:0,27:5"
SCRIPT_VERSION="coco-wrong-failure-localization-v1"

def args_parse():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",default="coco_two",choices=("coco_two",))
    p.add_argument("--data-root",default="data")
    p.add_argument("--model",default="qwen-3b")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--attn-impl",default="eager",choices=("eager",))
    p.add_argument("--direction-dir",required=True)
    p.add_argument("--direction-vector-mode",default="residual",choices=("residual","img"))
    p.add_argument("--direction-rank-metric",default="auto")
    p.add_argument("--direction-top-k",type=int,default=5)
    p.add_argument("--direction-max-layer",type=int,default=23)
    p.add_argument("--direction-heads",default="")
    p.add_argument("--receiver-heads",default=DEFAULT_RECEIVERS)
    p.add_argument("--prompt-layers",default="23,24,25,26,27,28")
    p.add_argument("--trace-layer-chunk",type=int,default=8)
    p.add_argument("--prompt-template",default=DEFAULT_PROMPT)
    p.add_argument("--pool",default="mean",choices=("mean","last"))
    p.add_argument("--train-ratio",type=float,default=.15)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--eval-max-samples",type=int,default=0)
    p.add_argument("--max-new-tokens",type=int,default=6)
    p.add_argument("--probe-module",default="analyze_coco_head_object_residual_direction_probe_v1")
    p.add_argument("--receiver-module",default="analyze_coco_receiver_qkv_v1")
    p.add_argument("--v3-module",default="analyze_spatial_storage_transport_utilization_v3")
    p.add_argument("--attention-helper-module",default="analyze_coco_flip_attention_spatial_vectors_v1")
    p.add_argument("--empty-cache-every",type=int,default=10)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--fail-fast",action=argparse.BooleanOptionalAction,default=True)
    return p.parse_args()

def hname(h): return f"L{h[0]}H{h[1]:02d}"
def parse_head(x):
    s=str(x).strip().upper().replace("L","").replace("H",":")
    while "::" in s:s=s.replace("::",":")
    a,b=s.split(":",1); return int(a),int(b)
def parse_heads(x):
    out=[]
    for s in str(x).split(","):
        if s.strip():
            h=parse_head(s)
            if h not in out: out.append(h)
    return out
def parse_layers(x):
    out=[]
    for s in str(x).split(","):
        s=s.strip().upper().replace("L","")
        if not s: continue
        vals=range(*((lambda a,b:(min(a,b),max(a,b)+1))(*map(int,s.split("-",1))))) if "-" in s else [int(s)]
        for v in vals:
            if v not in out: out.append(v)
    return sorted(out)
def normalize_relation(v):
    if v is None:return None
    t=str(v).strip().lower()
    alias={"left":"left","leftward":"left","right":"right","rightward":"right",
           "above":"above","over":"above","top":"above","on top":"above","on":"above",
           "below":"below","under":"below","underneath":"below","beneath":"below","bottom":"below"}
    if t in alias:return alias[t]
    pats=[(r"\b(left|leftward)\b","left"),(r"\b(right|rightward)\b","right"),
          (r"\b(below|under|underneath|beneath|bottom)\b","below"),
          (r"\b(above|over|on top|top)\b","above"),(r"\bon\b","above")]
    hits=[]
    for p,r in pats:
        m=re.search(p,t)
        if m:hits.append((m.start(),r))
    return sorted(hits)[0][1] if hits else None
def read_csv(p):
    with open(p,encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:p.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with open(p,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def append_jsonl(p,row):
    with open(p,"a",encoding="utf-8") as f:f.write(json.dumps(dict(row),ensure_ascii=False)+"\n")
def safe_mean(xs):
    vals=[]
    for x in xs:
        try:
            v=float(x)
            if np.isfinite(v):vals.append(v)
        except:pass
    return float(np.mean(vals)) if vals else float("nan")
def stratified_split(y,ratio,seed):
    rng=random.Random(seed);tr=[];te=[]
    for rel in RELATIONS:
        ids=[i for i,v in enumerate(y) if v==rel];rng.shuffle(ids)
        n=max(1,min(len(ids)-1,int(round(len(ids)*ratio))))
        tr+=ids[:n];te+=ids[n:]
    rng.shuffle(tr);rng.shuffle(te);return tr,te
def stratified_limit(ids,y,n,seed):
    ids=list(ids)
    if n<=0 or len(ids)<=n:return ids
    rng=random.Random(seed);g=defaultdict(list)
    for i in ids:g[y[i]].append(i)
    for v in g.values():rng.shuffle(v)
    ptr=Counter();out=[]
    while len(out)<n:
        moved=False
        for r in RELATIONS:
            if ptr[r]<len(g[r]) and len(out)<n:
                out.append(g[r][ptr[r]]);ptr[r]+=1;moved=True
        if not moved:break
    return out

def relation_token_variants(tok):
    out={}
    for r in RELATIONS:
        ids=set()
        for t in (r," "+r,r.capitalize()," "+r.capitalize()):
            z=tok.encode(t,add_special_tokens=False)
            if len(z)==1:ids.add(int(z[0]))
        if not ids:ids.add(int(tok.encode(" "+r,add_special_tokens=False)[-1]))
        out[r]=sorted(ids)
    return out

def get_attr_path(root,path):
    o=root
    for p in path.split("."):
        o=getattr(o,p,None)
        if o is None:return None
    return o
def resolve_final_norm(model,decoder_path):
    parent=decoder_path.rsplit(".",1)[0] if "." in decoder_path else ""
    c=[]
    if parent:c += [f"{parent}.norm",f"{parent}.final_layernorm",f"{parent}.ln_f"]
    c += ["model.language_model.norm","model.model.language_model.norm","language_model.model.norm",
          "language_model.norm","model.model.norm","model.norm","model.text_model.norm","text_model.norm"]
    for p in dict.fromkeys(c):
        m=get_attr_path(model,p)
        if isinstance(m,torch.nn.Module):return m,p
    return None,"unresolved"
def first_tensor(o):
    if torch.is_tensor(o):return o
    if isinstance(o,(tuple,list)) and o and torch.is_tensor(o[0]):return o[0]
    raise TypeError(type(o).__name__)
class CaptureToken:
    def __init__(self,module,pos):
        self.pos=int(pos);self.value=None;self.h=module.register_forward_hook(self.hook)
    def hook(self,_m,_i,o):
        x=first_tensor(o)
        if x.ndim==3 and x.shape[0]==1 and x.shape[1]>self.pos:self.value=x[0,self.pos].detach().float().cpu()
        return o
    def close(self):
        with contextlib.suppress(Exception):self.h.remove()
    def __enter__(self):return self
    def __exit__(self,*a):self.close()

class Codebook:
    def __init__(self,c,d):self.c=np.asarray(c,np.float32);self.d=np.asarray(d,np.float32)
    @classmethod
    def fit(cls,X,y):
        X=np.asarray(X,np.float32);c=X.mean(0);xc=X-c;ds=[]
        for r in RELATIONS:
            d=xc[y==r].mean(0);d/=max(float(np.linalg.norm(d)),EPS);ds.append(d)
        return cls(c,np.stack(ds))
    def scores(self,x):
        x=np.asarray(x,np.float32)-self.c;x/=max(float(np.linalg.norm(x)),EPS);return x@self.d.T
    def pred(self,x):
        s=self.scores(x);return RELATIONS[int(np.argmax(s))],s

def gt_margin(s,gt):return float(s[RID[gt]]-s[RID[OPPOSITE[gt]]])

def select_direction(rows,explicit,metric,k,max_layer):
    if explicit:
        z=[h for h in explicit if h[0]<=max_layer]
        if len(z)<k:raise ValueError("Not enough explicit early direction heads")
        return z[:k]
    vals=[]
    for r in rows:
        try:
            l,h,s=int(r["layer"]),int(r["head"]),float(r[metric])
            if l<=max_layer and np.isfinite(s):vals.append((s,l,h))
        except:pass
    vals.sort(reverse=True)
    return [(l,h) for _,l,h in vals[:k]]

def trace_target_state(trace,pos):
    lookup={int(p):i for i,p in enumerate(trace.target_positions)}
    return trace.block_output[lookup[int(pos)]].detach().float().cpu().numpy().astype(np.float32)

def receiver_vectors(receiver,trace,prompt_last,objpos,selected):
    sel=set(selected);ss=None;aa=None;sm=am=0.;per={}
    for h in range(int(trace.attention_weights.shape[0])):
        w,m=receiver.object_head_write(trace=trace,head=h,target_position=prompt_last,source_positions=objpos)
        w=w.detach().float().cpu().numpy().astype(np.float32)
        aa=w.copy() if aa is None else aa+w;am+=float(m)
        if h in sel:
            ss=w.copy() if ss is None else ss+w;sm+=float(m);per[h]=(w,float(m))
    if ss is None:ss=np.zeros_like(aa)
    return ss,aa,sm,am,per

def build_batch(probe,processor,q,image,device):
    return probe.process_inputs(processor,probe.build_chat_prompt(processor,q,True),image,device)
def generate(model,processor,batch,n):
    L=int(batch["input_ids"].shape[1])
    ids=model.generate(**batch,max_new_tokens=n,do_sample=False,use_cache=True)
    text=processor.tokenizer.decode(ids[0,L:],skip_special_tokens=True).strip();del ids
    return normalize_relation(text),text

def main():
    a=args_parse()
    if a.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.output_dir)
    if a.overwrite and out.exists():shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):raise RuntimeError(f"non-empty output: {out}")
    out.mkdir(parents=True,exist_ok=True);(out/"sample_vectors").mkdir(exist_ok=True)
    errors=out/"errors.jsonl"

    probe=importlib.import_module(a.probe_module);receiver=importlib.import_module(a.receiver_module)
    v3=importlib.import_module(a.v3_module);attnhelp=importlib.import_module(a.attention_helper_module);base=probe.base

    ddir=Path(a.direction_dir)
    with np.load(ddir/"relation_vectors.npz",allow_pickle=True) as z:d={k:np.asarray(z[k]) for k in z.files}
    if a.direction_vector_mode not in d:raise RuntimeError(f"missing {a.direction_vector_mode} in direction cache")
    Xdir=np.asarray(d[a.direction_vector_mode],np.float32);sids=np.asarray(d["sample_index"],np.int64)
    y=np.asarray([normalize_relation(x) for x in d["relation"].tolist()],object)
    good=np.asarray([x in RELATIONS for x in y]);Xdir=Xdir[good];sids=sids[good];y=y[good]
    if a.max_samples>0 and a.max_samples<len(y):
        keep=stratified_limit(range(len(y)),y,a.max_samples,a.seed);Xdir=Xdir[keep];sids=sids[keep];y=y[keep]
    sid2i={int(s):i for i,s in enumerate(sids)}
    tr,te=stratified_split(y,a.train_ratio,a.seed)
    if a.eval_max_samples>0:te=stratified_limit(te,y,a.eval_max_samples,a.seed+1)
    tr_sids=[int(sids[i]) for i in tr];te_sids=[int(sids[i]) for i in te]

    metric=a.direction_rank_metric
    if metric=="auto":metric="residual_accuracy_mean" if a.direction_vector_mode=="residual" else "img_accuracy_mean"
    dir_heads=select_direction(read_csv(ddir/"head_results.csv"),parse_heads(a.direction_heads),metric,a.direction_top_k,a.direction_max_layer)
    dir_books={h:Codebook.fit(Xdir[np.asarray(tr),h[0],h[1],:],y[np.asarray(tr)]) for h in dir_heads}
    dhrows=[]
    for h in dir_heads:
        ok=sum(dir_books[h].pred(Xdir[i,h[0],h[1],:])[0]==y[i] for i in te)
        dhrows.append({"head":hname(h),"accuracy":ok/max(len(te),1),"N_eval":len(te)})
    write_csv(out/"direction_head_accuracy.csv",dhrows)

    records,audit=base.load_records(a.dataset,Path(a.data_root),None);recmap={int(r.sid):r for r in records}
    tr_sids=[s for s in tr_sids if s in recmap];te_sids=[s for s in te_sids if s in recmap]
    spec=base.SPECS[a.model];cls=getattr(transformers,spec.model_class)
    kw=dict(dtype=base.resolve_dtype(spec.dtype_name),low_cpu_mem_usage=True,trust_remote_code=spec.trust_remote_code,
            device_map={"":a.device},attn_implementation=a.attn_impl)
    model=processor=None
    try:
        print(f"Loading {a.model}: {spec.repo_id}",flush=True)
        model=cls.from_pretrained(spec.repo_id,**kw);model.eval()
        if getattr(model,"generation_config",None):
            for x in ("temperature","top_p","top_k"):
                if hasattr(model.generation_config,x):setattr(model.generation_config,x,None)
        processor=AutoProcessor.from_pretrained(spec.repo_id,trust_remote_code=spec.trust_remote_code);base.configure_processor(model,processor)
        device=torch.device(a.device);layers,decoder_path=probe.resolve_decoder_layers(model);final_layer=len(layers)-1
        fnorm,fnorm_path=resolve_final_norm(model,decoder_path)
        if fnorm is None:raise RuntimeError("final norm unresolved")
        recv_heads=parse_heads(a.receiver_heads);recv_by_layer=defaultdict(list)
        for l,h in recv_heads:recv_by_layer[l].append(h)
        prompt_layers=parse_layers(a.prompt_layers)
        if final_layer not in prompt_layers:prompt_layers.append(final_layer);prompt_layers.sort()
        trace_layers=sorted(set(prompt_layers)|set(recv_by_layer));tokmap=relation_token_variants(processor.tokenizer)
        print("\nDirection:",", ".join(hname(h) for h in dir_heads))
        print("Receivers:",", ".join(hname(h) for h in recv_heads))
        print("Prompt layers:",prompt_layers,"final_norm=",fnorm_path,flush=True)

        internal={};basic=[];trainset=set(tr_sids)
        for n,sid in enumerate(tqdm(tr_sids+te_sids,desc="natural-trace"),1):
            image=batch=None
            try:
                rec=recmap[sid];gt=normalize_relation(rec.relation);q=a.prompt_template.format(subject=rec.subject,reference=rec.reference)
                image=Image.open(rec.image_path).convert("RGB");batch=build_batch(probe,processor,q,image,device)
                ids=[int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
                ap=probe.locate_phrase_positions(processor.tokenizer,ids,str(rec.subject));bp=probe.locate_phrase_positions(processor.tokenizer,ids,str(rec.reference))
                objpos=sorted(set(ap+bp));last=len(ids)-1
                with CaptureToken(fnorm,last) as cap:
                    baseline,traces=v3.trace_prompt_chunks(attention_helper=attnhelp,model=model,batch=batch,relation_token_map=tokmap,
                        decoder_layers=layers,layers=trace_layers,target_positions=[last],chunk_size=a.trace_layer_chunk)
                if cap.value is None:raise RuntimeError("final norm capture missing")
                ci=sid2i[sid];scores=[];per={}
                for h in dir_heads:
                    p,s=dir_books[h].pred(Xdir[ci,h[0],h[1],:]);scores.append(s);per[hname(h)]=p
                ds=np.mean(np.stack(scores),0);dp=RELATIONS[int(np.argmax(ds))]
                ps={l:trace_target_state(traces[l],last) for l in prompt_layers}
                rv={};rm={}
                for l,hs in recv_by_layer.items():
                    ss,aa,sm,am,perh=receiver_vectors(receiver,traces[l],last,objpos,hs)
                    rv[f"recv_L{l}_selected"]=ss;rv[f"recv_L{l}_all"]=aa
                    rm[f"recv_L{l}_selected_mass"]=sm;rm[f"recv_L{l}_all_mass"]=am
                    rm[f"recv_L{l}_selected_norm"]=float(np.linalg.norm(ss));rm[f"recv_L{l}_all_norm"]=float(np.linalg.norm(aa))
                    for h,(w,m) in perh.items():rv[f"recv_L{l}_H{h:02d}"]=w;rm[f"recv_L{l}_H{h:02d}_mass"]=m;rm[f"recv_L{l}_H{h:02d}_norm"]=float(np.linalg.norm(w))
                lm_s=np.asarray([float(baseline["scores"][r]) for r in RELATIONS],np.float32);lm_p=str(baseline["prediction"])
                gp=gtxt=None
                if sid not in trainset:gp,gtxt=generate(model,processor,batch,a.max_new_tokens)
                internal[sid]={"gt":gt,"dir_pred":dp,"dir_scores":ds,"prompt":ps,"recv":rv,"recv_meta":rm,"final_norm":cap.value.numpy().astype(np.float32),
                               "lm_pred":lm_p,"lm_scores":lm_s,"gen_pred":gp,"gen_text":gtxt}
                payload={"sid":np.asarray(sid),"prompt_layers":np.asarray(prompt_layers),"prompt_last_states":np.stack([ps[l] for l in prompt_layers]),
                         "final_norm":internal[sid]["final_norm"],"direction_scores":ds,"lm_scores":lm_s}
                payload.update(rv);np.savez_compressed(out/"sample_vectors"/f"{sid}.npz",**payload)
                basic.append({"sid":sid,"split":"train" if sid in trainset else "eval","gt":gt,"direction_pred":dp,"lm_pred":lm_p,"generation_pred":gp,"generation_text":gtxt,**rm})
            except Exception as e:
                append_jsonl(errors,{"phase":"trace","sid":sid,"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
                if a.fail_fast:raise
            finally:
                if image is not None:image.close()
                del batch;gc.collect()
                if torch.cuda.is_available() and a.empty_cache_every>0 and n%a.empty_cache_every==0:torch.cuda.empty_cache()
        write_csv(out/"all_samples_basic.csv",basic)
        vtr=[s for s in tr_sids if s in internal];vte=[s for s in te_sids if s in internal];ytr=np.asarray([internal[s]["gt"] for s in vtr],object)
        books={}
        for l in prompt_layers:books[f"prompt_L{l}"]=Codebook.fit(np.stack([internal[s]["prompt"][l] for s in vtr]),ytr)
        books["final_norm"]=Codebook.fit(np.stack([internal[s]["final_norm"] for s in vtr]),ytr)
        recv_keys=sorted(set.intersection(*[set(internal[s]["recv"]) for s in vtr]))
        for k in recv_keys:books[k]=Codebook.fit(np.stack([internal[s]["recv"][k] for s in vtr]),ytr)

        evalrows=[];counts=Counter();cover=Counter()
        for sid in vte:
            z=internal[sid];gt=z["gt"];r={"sid":sid,"gt":gt,"direction_pred":z["dir_pred"],"direction_margin":gt_margin(z["dir_scores"],gt)}
            cover["direction"]+=1;counts["direction"]+=z["dir_pred"]==gt
            for k,b in books.items():
                vec=z["final_norm"] if k=="final_norm" else z["prompt"][int(k.split("L",1)[1])] if k.startswith("prompt_L") else z["recv"][k]
                p,s=b.pred(vec);r[k+"_pred"]=p;r[k+"_margin"]=gt_margin(s,gt);cover[k]+=1;counts[k]+=p==gt
            r["lm_4way_pred"]=z["lm_pred"];r["lm_4way_margin"]=gt_margin(z["lm_scores"],gt);cover["lm_4way"]+=1;counts["lm_4way"]+=z["lm_pred"]==gt
            r["generation_pred"]=z["gen_pred"];r["generation_correct"]=z["gen_pred"]==gt;cover["generation"]+=1;counts["generation"]+=z["gen_pred"]==gt
            r.update(z["recv_meta"]);evalrows.append(r)
        write_csv(out/"baseline_eval.csv",evalrows)

        reported=["direction"]
        for l in prompt_layers:
            if l in recv_by_layer:
                reported += [f"recv_L{l}_selected",f"recv_L{l}_all"]+[f"recv_L{l}_H{h:02d}" for h in recv_by_layer[l]]
            reported += [f"prompt_L{l}"]
        reported += ["final_norm","lm_4way","generation"];reported=list(dict.fromkeys(reported))
        stagerows=[{"stage":k,"N_eval":cover[k],"accuracy":counts[k]/cover[k]} for k in reported if cover[k]]
        write_csv(out/"stage_accuracy.csv",stagerows)

        mainseq=["direction"]
        for l in prompt_layers:
            if l in recv_by_layer:mainseq.append(f"recv_L{l}_all")
            mainseq.append(f"prompt_L{l}")
        mainseq += ["final_norm","lm_4way","generation"];mainseq=list(dict.fromkeys(mainseq))
        def pred(r,s):
            if s=="direction":return r.get("direction_pred")
            if s=="lm_4way":return r.get("lm_4way_pred")
            if s=="generation":return r.get("generation_pred")
            return r.get(s+"_pred")
        wrong=[r for r in evalrows if not r["generation_correct"]];correct=[r for r in evalrows if r["generation_correct"]]
        wtable=[];fl=Counter()
        for r in wrong:
            x=dict(r);first=None
            for s in mainseq:
                p=pred(r,s)
                if p is None:continue
                c=p==r["gt"];x[s+"_correct"]=c
                if first is None and not c:first=s
            first=first or "none_all_internal_correct";x["first_loss_stage"]=first;fl[first]+=1;wtable.append(x)
        write_csv(out/"wrong_sample_stage_table.csv",wtable)

        trans=[]
        pairs=list(zip(mainseq[:-1],mainseq[1:]))
        extra=[("direction","recv_L26_all"),("direction","prompt_L26"),("direction","recv_L27_all"),("direction","prompt_L27"),("direction","final_norm"),("direction","lm_4way")]
        for pr,cu in list(dict.fromkeys(pairs+extra)):
            if not any(pred(r,pr) is not None and pred(r,cu) is not None for r in wrong):continue
            elig=[r for r in wrong if pred(r,pr)==r["gt"] and pred(r,cu) is not None];keep=[r for r in elig if pred(r,cu)==r["gt"]]
            trans.append({"previous_stage":pr,"current_stage":cu,"N_wrong_prev_correct":len(elig),"N_current_correct":len(keep),
                          "retention_rate":len(keep)/len(elig) if elig else float("nan"),"correct_to_wrong_count":len(elig)-len(keep)})
        write_csv(out/"wrong_transition_summary.csv",trans)

        groups=[]
        for gn,g in (("generation_correct",correct),("generation_wrong",wrong)):
            for s in reported:
                cov=[r for r in g if pred(r,s) is not None]
                if not cov:continue
                mk="direction_margin" if s=="direction" else "lm_4way_margin" if s=="lm_4way" else s+"_margin"
                groups.append({"group":gn,"stage":s,"N":len(cov),"accuracy":safe_mean(pred(r,s)==r["gt"] for r in cov),
                               "mean_gt_margin":safe_mean(r.get(mk) for r in cov) if s!="generation" else float("nan")})
        write_csv(out/"generation_group_stage_summary.csv",groups)

        strength=[];keys=sorted({k for r in evalrows for k in r if k.startswith("recv_") and (k.endswith("_mass") or k.endswith("_norm"))})
        for gn,g in (("generation_correct",correct),("generation_wrong",wrong),
                     ("generation_wrong_direction_correct",[r for r in wrong if r["direction_pred"]==r["gt"]]),
                     ("generation_wrong_direction_wrong",[r for r in wrong if r["direction_pred"]!=r["gt"]])):
            for k in keys:strength.append({"group":gn,"metric":k,"N":len(g),"mean":safe_mean(r.get(k) for r in g)})
        write_csv(out/"receiver_strength_summary.csv",strength)

        print("\n"+"="*132);print("FAILURE LOCALIZATION SUMMARY");print("="*132)
        print(f"Eval generation ACC: {100*counts['generation']/cover['generation']:.2f}% | wrong={len(wrong)}/{len(evalrows)}")
        print("\nStage held-out accuracies:")
        for r in stagerows:print(f"  {r['stage']:<28s} {100*r['accuracy']:6.2f}%")
        print("\nFirst-loss diagnostic on baseline-WRONG:")
        for s,n in fl.most_common():print(f"  {s:<28s} {n:4d}/{len(wrong):4d} ({100*n/max(len(wrong),1):6.2f}%)")
        print("\nKey direction retention:")
        for r in trans:
            if r["previous_stage"]=="direction" and r["current_stage"] in {"recv_L26_all","prompt_L26","recv_L27_all","prompt_L27","final_norm","lm_4way"}:
                print(f"  direction -> {r['current_stage']:<18s} N={r['N_wrong_prev_correct']:3d} retain={100*r['retention_rate']:6.2f}% C->W={r['correct_to_wrong_count']:3d}")
        print("="*132)

        cfg={"script_version":SCRIPT_VERSION,"model":a.model,"direction_vector_mode":a.direction_vector_mode,"direction_rank_metric":metric,
             "direction_heads":[hname(h) for h in dir_heads],"receiver_heads":[hname(h) for h in recv_heads],"prompt_layers":prompt_layers,
             "main_stage_sequence":mainseq,"final_norm_path":fnorm_path,"train_ratio":a.train_ratio,"N_train":len(vtr),"N_eval":len(vte),
             "N_wrong":len(wrong),"uses_intervention":False,"receiver_write":"W_O^h sum_object A[last,s]V[s]",
             "warning":"first-loss is only meaningful where stage held-out probe accuracy is sufficiently high","audit":audit}
        (out/"config.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
        report=[f"generation_acc={100*counts['generation']/cover['generation']:.2f}%",f"wrong={len(wrong)}/{len(evalrows)}","","STAGE ACCURACY"]
        report += [f"{r['stage']}: {100*r['accuracy']:.2f}%" for r in stagerows]
        report += ["","FIRST LOSS"]+[f"{s}: {n}/{len(wrong)}" for s,n in fl.most_common()]
        report += ["","Interpretation: selected receiver wrong but all-head receiver correct => shortlist incomplete;",
                   "direction correct but L26 all-head write wrong => transport/utilization gap;",
                   "L26 write correct but prompt-L26 wrong => within-block integration gap;",
                   "final norm correct but LM wrong => LM readout mismatch; LM correct but generation wrong => decoding/format mismatch."]
        (out/"report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
        print("\nSaved to",out)
    finally:
        if model is not None:del model
        if processor is not None:del processor
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
if __name__=="__main__":main()
