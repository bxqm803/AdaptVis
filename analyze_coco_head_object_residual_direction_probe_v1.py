#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor
import extract_two_object_relation_states as base

SCRIPT_VERSION = "coco-head-object-residual-direction-probe-v1"
EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_ALIASES = {
    "left": "left", "right": "right",
    "above": "above", "on": "above", "top": "above", "over": "above",
    "below": "below", "under": "below", "underneath": "below", "bottom": "below",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager","sdpa","flash_attention_2","none"])
    p.add_argument("--prompt-template", default=(
        "Determine the spatial relation of the {subject} to the {reference} "
        "in the image. Answer with left, right, above, or below."
    ))
    p.add_argument("--pool", default="mean", choices=["mean","last"])
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def norm_relation(x):
    return REL_ALIASES.get(str(x).strip().lower().replace("-","_"),
                           str(x).strip().lower().replace("-","_"))

def head_name(l,h): return f"L{l}H{h:02d}"

def resolve_decoder_layers(model):
    candidates = [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("model.model.model.layers", lambda m: m.model.model.model.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers):
                return layers, name
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers")

def resolve_self_attention(layer):
    for name in ("self_attn","attention","attn"):
        m = getattr(layer,name,None)
        if m is not None: return m
    raise RuntimeError("Could not resolve self-attention")

def resolve_o_proj(attn):
    for name in ("o_proj","out_proj","proj"):
        m = getattr(attn,name,None)
        if isinstance(m, torch.nn.Module): return m
    raise RuntimeError("Could not resolve o_proj")

def config_num_heads(model):
    cfgs = [
        getattr(model,"config",None),
        getattr(getattr(model,"config",None),"text_config",None),
        getattr(getattr(model,"config",None),"language_config",None),
    ]
    for cfg in cfgs:
        if cfg is None: continue
        for attr in ("num_attention_heads","n_head","num_heads"):
            v = getattr(cfg,attr,None)
            if v is not None:
                try: return int(v)
                except Exception: pass
    return None

def attention_num_heads(attn, model):
    for attr in ("num_heads","num_attention_heads","n_heads"):
        v = getattr(attn,attr,None)
        if v is not None:
            try: return int(v)
            except Exception: pass
    v = config_num_heads(model)
    if v is None: raise RuntimeError("Could not determine num_heads")
    return v

def scan_shape(model,layers):
    nh_ref = hd_ref = None
    for li, layer in enumerate(layers):
        attn = resolve_self_attention(layer)
        op = resolve_o_proj(attn)
        nh = attention_num_heads(attn,model)
        inf = getattr(op,"in_features",None)
        if inf is None and hasattr(op,"weight"): inf = int(op.weight.shape[1])
        if inf is None: raise RuntimeError(f"L{li}: cannot infer o_proj width")
        inf = int(inf)
        if inf % nh: raise RuntimeError(f"L{li}: width={inf}, heads={nh}")
        hd = inf // nh
        if nh_ref is None: nh_ref, hd_ref = nh, hd
        elif (nh,hd)!=(nh_ref,hd_ref):
            raise RuntimeError("Non-uniform head shape")
    return nh_ref, hd_ref

def build_chat_prompt(processor, question, with_image):
    content = ([{"type":"image"}] if with_image else []) + [{"type":"text","text":question}]
    return processor.apply_chat_template(
        [{"role":"user","content":content}],
        tokenize=False,
        add_generation_prompt=True,
    )

def process_inputs(processor, rendered, image, device):
    if image is None:
        batch = processor(text=[rendered], padding=True, return_tensors="pt")
    else:
        batch = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return batch.to(device)

def find_subsequence_last(haystack, needle):
    best = None
    n = len(needle)
    for i in range(len(haystack)-n+1):
        if list(haystack[i:i+n]) == list(needle): best = (i,i+n)
    return best

def locate_phrase_positions(tokenizer,input_ids,phrase):
    hits=[]
    for text in (str(phrase)," "+str(phrase)):
        try: ids = tokenizer.encode(text,add_special_tokens=False)
        except Exception: ids=[]
        if ids:
            hit = find_subsequence_last(input_ids,ids)
            if hit is not None: hits.append(hit)
    if hits:
        s,e=max(hits,key=lambda x:x[0])
        return list(range(s,e))
    idx = int(base.find_phrase_last_token(tokenizer,list(input_ids),str(phrase)))
    return [idx]

def pool_positions(tensor,positions,mode):
    valid=[int(p) for p in positions if 0 <= int(p) < int(tensor.shape[1])]
    if not valid: raise RuntimeError("No valid object positions")
    if mode=="last": return tensor[0,valid[-1]]
    idx=torch.as_tensor(valid,device=tensor.device,dtype=torch.long)
    return tensor[0].index_select(0,idx).mean(dim=0)

class Capture:
    def __init__(self,layers,n_heads,head_dim,a_pos,b_pos,pool):
        self.layers=layers; self.n_heads=n_heads; self.head_dim=head_dim
        self.a_pos=a_pos; self.b_pos=b_pos; self.pool=pool
        self.out=torch.empty((len(layers),n_heads,2,head_dim),dtype=torch.float32)
        self.seen=set(); self.handles=[]
    def __enter__(self):
        for li,layer in enumerate(self.layers):
            op=resolve_o_proj(resolve_self_attention(layer))
            def make_hook(li):
                def hook(_m,inputs):
                    x=inputs[0]
                    a=pool_positions(x,self.a_pos,self.pool).view(self.n_heads,self.head_dim)
                    b=pool_positions(x,self.b_pos,self.pool).view(self.n_heads,self.head_dim)
                    self.out[li,:,0]=a.detach().float().cpu()
                    self.out[li,:,1]=b.detach().float().cpu()
                    self.seen.add(li)
                return hook
            self.handles.append(op.register_forward_pre_hook(make_hook(li)))
        return self
    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception): h.remove()
        self.handles=[]
    def __exit__(self,*args): self.close()
    def finalize(self):
        missing=[i for i in range(len(self.layers)) if i not in self.seen]
        if missing: raise RuntimeError(f"Missing captures: {missing[:10]}")
        return self.out.numpy()

def capture_condition(model,processor,device,layers,n_heads,head_dim,
                      question,subject,reference,image,pool):
    rendered=build_chat_prompt(processor,question,image is not None)
    batch=process_inputs(processor,rendered,image,device)
    ids=[int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    apos=locate_phrase_positions(processor.tokenizer,ids,subject)
    bpos=locate_phrase_positions(processor.tokenizer,ids,reference)
    cap=Capture(layers,n_heads,head_dim,apos,bpos,pool)
    try:
        with cap:
            with torch.inference_mode():
                model(**batch,output_attentions=False,output_hidden_states=False,
                      use_cache=False,return_dict=True)
        out=cap.finalize()
    finally:
        cap.close()
        del batch
    return out

def normalize_rows(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)

def fit_codebook(X,y):
    center=X.mean(axis=0)
    Xc=X-center
    dirs=[]
    for rel in RELATIONS:
        m=(y==rel)
        if int(m.sum())==0: raise RuntimeError(f"no train samples for {rel}")
        d=Xc[m].mean(axis=0)
        d=d/max(float(np.linalg.norm(d)),EPS)
        dirs.append(d)
    return center,np.stack(dirs)

def make_splits(n,ratio,repeats,seed):
    out=[]
    for rep in range(repeats):
        ids=list(range(n))
        random.Random(seed+rep).shuffle(ids)
        nt=int(n*ratio)
        out.append((np.asarray(ids[:nt]),np.asarray(ids[nt:])))
    return out

def eval_tensor(X,y,splits,metric):
    N,L,H,D=X.shape
    gt=np.asarray([RELATIONS.index(v) for v in y])
    rows=[]
    for l in tqdm(range(L),desc=f"probe:{metric}",leave=False):
        for h in range(H):
            Xh=np.asarray(X[:,l,h,:],dtype=np.float32)
            accs=[]; cls={r:[] for r in RELATIONS}; lr=[]; ud=[]
            for tr,te in splits:
                c,dirs=fit_codebook(Xh[tr],y[tr])
                Xt=normalize_rows(Xh[te]-c)
                pred=np.argmax(Xt@dirs.T,axis=1)
                g=gt[te]
                accs.append(float(np.mean(pred==g)))
                for ri,r in enumerate(RELATIONS):
                    m=(g==ri)
                    cls[r].append(float(np.mean(pred[m]==g[m])) if m.any() else np.nan)
                lr.append(float(np.dot(dirs[0],dirs[1])))
                ud.append(float(np.dot(dirs[2],dirs[3])))
            rows.append({
                "metric":metric,"layer":l,"head":h,"head_name":head_name(l,h),
                "accuracy_mean":float(np.mean(accs)),
                "accuracy_std":float(np.std(accs)),
                "left_accuracy":float(np.nanmean(cls["left"])),
                "right_accuracy":float(np.nanmean(cls["right"])),
                "on_accuracy":float(np.nanmean(cls["above"])),
                "under_accuracy":float(np.nanmean(cls["below"])),
                "lr_direction_cosine":float(np.mean(lr)),
                "on_under_direction_cosine":float(np.mean(ud)),
            })
    return rows

def save_csv(path,rows):
    if not rows: return
    fields=list(rows[0].keys())
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    args=parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not (0<args.train_ratio<1): raise ValueError("bad train ratio")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.output_dir)
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    records,audit=base.load_records(args.dataset,Path(args.data_root),args.max_samples)
    records=[r for r in records if norm_relation(r.relation) in RELATIONS]
    print(f"[{args.dataset}] n={len(records)} counts={dict(Counter(norm_relation(r.relation) for r in records))}")

    spec=base.SPECS[args.model]
    cls=getattr(transformers,spec.model_class)
    kw=dict(dtype=base.resolve_dtype(spec.dtype_name),low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,device_map={"":args.device})
    if args.attn_impl!="none": kw["attn_implementation"]=args.attn_impl

    model=cls.from_pretrained(spec.repo_id,**kw); model.eval()
    processor=AutoProcessor.from_pretrained(spec.repo_id,trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model,processor)
    device=torch.device(args.device)

    layers,path=resolve_decoder_layers(model)
    nh,hd=scan_shape(model,layers)
    print(f"decoder={path} layers={len(layers)} heads/layer={nh} head_dim={hd} total={len(layers)*nh}")

    dtype_np=np.float32 if args.keep_fp32 else np.float16
    sids=[]; labels=[]; res_list=[]; img_list=[]; noimg_list=[]; errors=[]

    for rec in tqdm(records,desc="extract"):
        image=None
        try:
            q=args.prompt_template.format(subject=rec.subject,reference=rec.reference)
            image=Image.open(rec.image_path).convert("RGB")
            zi=capture_condition(model,processor,device,layers,nh,hd,q,
                                 str(rec.subject),str(rec.reference),image,args.pool)
            zn=capture_condition(model,processor,device,layers,nh,hd,q,
                                 str(rec.subject),str(rec.reference),None,args.pool)
            ri=zi[:,:,0]-zi[:,:,1]
            rn=zn[:,:,0]-zn[:,:,1]
            rr=ri-rn
            sids.append(int(rec.sid)); labels.append(norm_relation(rec.relation))
            res_list.append(rr.astype(dtype_np))
            if not args.no_controls:
                img_list.append(ri.astype(dtype_np)); noimg_list.append(rn.astype(dtype_np))
            del zi,zn,ri,rn,rr
        except Exception as e:
            errors.append({"sid":int(rec.sid),"error":str(e),
                           "traceback_tail":traceback.format_exc().splitlines()[-12:]})
            tqdm.write(f"[ERROR] sid={rec.sid}: {type(e).__name__}: {e}")
        finally:
            if image is not None: image.close()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    Xr=np.stack(res_list)
    y=np.asarray(labels)
    arrays={"sample_index":np.asarray(sids),"relation":y,
            "residual":Xr,"decoder_block_index":np.arange(len(layers)),
            "head_index":np.arange(nh)}
    if not args.no_controls:
        Xi=np.stack(img_list); Xn=np.stack(noimg_list)
        arrays["img"]=Xi; arrays["no_image"]=Xn
    np.savez_compressed(out/"relation_vectors.npz",**arrays)

    splits=make_splits(len(y),args.train_ratio,args.repeats,args.seed)
    print(f"train={len(splits[0][0])} test={len(splits[0][1])} repeats={args.repeats}")

    res=eval_tensor(Xr,y,splits,"residual")
    controls={}
    if not args.no_controls:
        controls["img"]=eval_tensor(Xi,y,splits,"img")
        controls["no_image"]=eval_tensor(Xn,y,splits,"no_image")

    by={}
    for row in res:
        by[(row["layer"],row["head"])]={
            "layer":row["layer"],"head":row["head"],"head_name":row["head_name"],
            **{f"residual_{k}":v for k,v in row.items() if k not in ("metric","layer","head","head_name")}
        }
    for metric,rows in controls.items():
        for row in rows:
            item=by[(row["layer"],row["head"])]
            for k,v in row.items():
                if k not in ("metric","layer","head","head_name"):
                    item[f"{metric}_{k}"]=v
    for item in by.values():
        if "img_accuracy_mean" in item:
            item["residual_minus_img"]=item["residual_accuracy_mean"]-item["img_accuracy_mean"]
            item["residual_minus_no_image"]=item["residual_accuracy_mean"]-item["no_image_accuracy_mean"]

    merged=sorted(by.values(),key=lambda r:r["residual_accuracy_mean"],reverse=True)
    for i,r in enumerate(merged,1): r["rank"]=i
    save_csv(out/"head_results.csv",merged)
    (out/"errors.json").write_text(json.dumps(errors,indent=2),encoding="utf-8")
    (out/"summary.json").write_text(json.dumps({
        "script_version":SCRIPT_VERSION,
        "model":args.model,"dataset":args.dataset,
        "n":len(y),"train_ratio":args.train_ratio,"repeats":args.repeats,
        "seed":args.seed,"layers":len(layers),"heads_per_layer":nh,"head_dim":hd,
        "vector_definition":"[(z_img_A-z_noimg_A)-(z_img_B-z_noimg_B)] pre-W_O per head",
        "top_heads":merged[:args.top_k],
    },indent=2),encoding="utf-8")

    print("\nrank head     residual_acc      img_acc  noimg_acc  res-img   left  right   on  under")
    for r in merged[:args.top_k]:
        print(
            f"{r['rank']:02d}. {r['head_name']:<8s} "
            f"{r['residual_accuracy_mean']:.4f}±{r['residual_accuracy_std']:.4f}  "
            f"{r.get('img_accuracy_mean',float('nan')):.4f}   "
            f"{r.get('no_image_accuracy_mean',float('nan')):.4f}   "
            f"{r.get('residual_minus_img',float('nan')):+.4f}  "
            f"{r['residual_left_accuracy']:.4f} "
            f"{r['residual_right_accuracy']:.4f} "
            f"{r['residual_on_accuracy']:.4f} "
            f"{r['residual_under_accuracy']:.4f}"
        )

if __name__=="__main__":
    main()
