#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training-free 2x2 ROLE x VISUAL head-function discovery.

Default layers:
    L1-L3, L18-L26, L31-L32

Four natural conditions:
    C00 = original query + original image        -> GT
    C10 = swapped query  + original image        -> opposite(GT)
    C01 = original query + horizontal-flip image -> opposite(GT)
    C11 = swapped query  + horizontal-flip image -> GT

No probe, no DAS, no ablation, no learned intervention.

For every selected layer, a pre-hook on attention.o_proj captures its input,
i.e. the concatenated per-head attention outputs BEFORE W_O.  We split it into
[num_heads, head_dim] and record identity-A, identity-B, and prompt-last.

Identity alignment:
    A is compared with A across query swap;
    B is compared with B across query swap.

Factorial signatures:
    ROLE        R = (C10 + C11 - C00 - C01) / 4
    VISUAL      V = (C01 + C11 - C00 - C10) / 4
    INTERACTION I = (C00 + C11 - C10 - C01) / 4

A high ROLE score means natural head output follows query-role manipulation.
A high VISUAL score means it follows horizontal geometry with role fixed.
A high INTERACTION score means single flips change it but double flip restores
it -- the XOR signature expected from role x geometry relation integration.

Discovery metrics:
    effect_strength = RMS(effect) / RMS(natural head activation)
    effect_share    = effect_energy / (R_energy + V_energy + I_energy)
    consistency     = cross-sample directional consistency after LEFT/RIGHT
                      semantic sign alignment
    functional_score = effect_strength * effect_share

Default ranking uses only QUAD-CORRECT examples, i.e. the model naturally gets
C00/C10/C01/C11 all correct.  This is still discovery, not causal proof.

Recommended:
CUDA_VISIBLE_DEVICES=0 python -u scan_coco_head_function_2x2_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --flip-pairs-jsonl output/qwen3b_coco_horizontal_flip_generation_v1/flip_generation_pairs.jsonl \
  --layers 1-3,18-26,31-32 \
  --flip-status both_correct \
  --metric-subset quad_correct \
  --device cuda:0 \
  --output-dir output/qwen3b_head_function_2x2_selected \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

VERSION = "coco-head-function-2x2-v1"
RELATIONS = ("left", "right", "above", "below")
LR = ("left", "right")
OPPOSITE = {"left":"right", "right":"left", "above":"below", "below":"above"}
EFFECTS = ("role", "visual", "interaction")
SITES = ("A", "B", "last")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--object-state", default="mean")
    p.add_argument("--source-output-dir", default="output/spatial_storage_transport_utilization/coco/qwen-3b")
    p.add_argument("--flip-pairs-jsonl", required=True)
    p.add_argument("--layers", default="1-3,18-26,31-32")
    p.add_argument("--base-pair-status", default="all", choices=("all","both_correct","original_only","swapped_only","both_wrong"))
    p.add_argument("--flip-status", default="both_correct", choices=("all","both_correct","clean_correct","flip_correct"))
    p.add_argument("--metric-subset", default="quad_correct", choices=("quad_correct","all_successful"))
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--empty-cache-every", type=int, default=8)
    p.add_argument("--flip-helper", default="eval_coco_horizontal_flip_generation_v3.py")
    p.add_argument("--ioi-script", default="analyze_coco_ioi_backward_circuit_v1.py")
    p.add_argument("--producer-script", default="analyze_coco_producer_qk_ov_v1.py")
    p.add_argument("--receiver-script", default="analyze_coco_receiver_qkv_v1.py")
    p.add_argument("--v3-script", default="analyze_spatial_storage_transport_utilization_v3.py")
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def import_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_layers(text: str):
    out, seen = [], set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            vals = range(min(a,b), max(a,b)+1)
        else:
            vals = [int(part)]
        for v in vals:
            if v not in seen:
                seen.add(v); out.append(v)
    if not out:
        raise ValueError("empty layer specification")
    return sorted(out)


def normalize_positions(x):
    if x is None: return []
    if isinstance(x, (int, np.integer)): return [int(x)]
    if torch.is_tensor(x): return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, np.ndarray): return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple, set)): return [int(v) for v in x]
    return [int(x)]


def stratified_subset(rows, limit, seed):
    rows = [dict(r) for r in rows]
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows: groups[str(r["gt"])].append(r)
    for g in groups.values(): rng.shuffle(g)
    keys = sorted(groups); idx = {k:0 for k in keys}; out=[]
    while len(out) < limit:
        moved=False
        for k in keys:
            if len(out) >= limit: break
            if idx[k] < len(groups[k]):
                out.append(groups[k][idx[k]]); idx[k]+=1; moved=True
        if not moved: break
    rng.shuffle(out)
    return out


def load_rows(extraction_path, flip_path, base_status, flip_status):
    ext = {int(r["sid"]):dict(r) for r in read_jsonl(extraction_path) if str(r.get("gt")) in LR}
    flips = {int(r["sid"]):dict(r) for r in read_jsonl(flip_path)}
    out=[]
    for sid in sorted(set(ext) & set(flips)):
        r=dict(ext[sid]); f=flips[sid]
        if str(r.get("gt")) != str(f.get("gt")): continue
        if base_status != "all" and str(r.get("generation_pair_status","")) != base_status: continue
        clean_ok=bool(f.get("clean_correct",False)); flip_ok=bool(f.get("flip_correct_aligned",False))
        if flip_status=="both_correct" and not(clean_ok and flip_ok): continue
        if flip_status=="clean_correct" and not clean_ok: continue
        if flip_status=="flip_correct" and not flip_ok: continue
        out.append(r)
    return out


def resolve_attention(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name): return getattr(layer, name)
    raise AttributeError(f"no attention module in {type(layer).__name__}")


def resolve_o_proj(attn):
    for name in ("o_proj", "out_proj", "dense"):
        if hasattr(attn, name) and hasattr(getattr(attn,name), "weight"):
            return getattr(attn,name)
    raise AttributeError(f"no o_proj in {type(attn).__name__}")


def infer_num_heads(attn, o_proj, model):
    n=None
    for name in ("num_heads", "num_attention_heads", "n_heads"):
        v=getattr(attn,name,None)
        if v is not None: n=int(v); break
    if n is None:
        for cfg in (getattr(attn,"config",None), getattr(getattr(model,"config",None),"text_config",None), getattr(model,"config",None)):
            if cfg is not None and getattr(cfg,"num_attention_heads",None) is not None:
                n=int(cfg.num_attention_heads); break
    if n is None: raise RuntimeError("cannot infer num_attention_heads")
    width=int(o_proj.weight.shape[1])
    if width % n: raise RuntimeError(f"o_proj input {width} not divisible by {n}")
    return n, width//n


class Capture:
    """Capture per-head attention output immediately before W_O."""
    def __init__(self, model, decoder_layers, layers):
        self.layers=list(layers); self.positions={}; self.current={}; self.handles=[]; self.shapes={}
        for L in self.layers:
            attn=resolve_attention(decoder_layers[L]); proj=resolve_o_proj(attn)
            H,D=infer_num_heads(attn,proj,model); self.shapes[L]=(H,D)
            def make_hook(layer_idx, nh, hd):
                def hook(module, inputs):
                    x=inputs[0]
                    if not torch.is_tensor(x) or x.ndim != 3:
                        raise RuntimeError(f"L{layer_idx} o_proj input shape invalid: {getattr(x,'shape',None)}")
                    if x.shape[-1] != nh*hd:
                        raise RuntimeError(f"L{layer_idx}: {x.shape[-1]} != {nh}*{hd}")
                    d={}
                    for site,pos in self.positions.items():
                        pp=sorted(set(normalize_positions(pos)))
                        if not pp: continue
                        idx=torch.as_tensor(pp,device=x.device,dtype=torch.long)
                        d[site]=x[0].index_select(0,idx).mean(0).reshape(nh,hd).detach().float().cpu()
                    self.current[layer_idx]=d
                return hook
            self.handles.append(proj.register_forward_pre_hook(make_hook(L,H,D)))
    def start(self, positions):
        self.positions={k:normalize_positions(v) for k,v in positions.items()}; self.current={}
    def snapshot(self):
        return {L:{s:t.clone() for s,t in d.items()} for L,d in self.current.items()}
    def close(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass


def relation_prediction(logits_last, relation_token_map):
    vals=[]
    for r in RELATIONS:
        ids=[int(v) for v in relation_token_map[r]]
        vals.append(float(logits_last[ids].detach().float().max().item()))
    a=np.asarray(vals,dtype=np.float32)
    return RELATIONS[int(a.argmax())], a


@torch.inference_mode()
def run_condition(model,batch,capture,positions,relation_token_map):
    capture.start(positions)
    out=model(**batch,use_cache=False,output_attentions=False,output_hidden_states=False,return_dict=True)
    pred,scores=relation_prediction(out.logits[0,-1],relation_token_map)
    z=capture.snapshot(); del out
    return z,pred,scores


class Accumulator:
    def __init__(self):
        self.e_sum={}; self.base_sum={}; self.dir_sum={}; self.dir_n={}; self.n=defaultdict(int)
    def _ensure(self,L,site,effect,H,D):
        ek=(L,site,effect); bk=(L,site)
        if ek not in self.e_sum:
            self.e_sum[ek]=np.zeros(H,np.float64); self.dir_sum[ek]=np.zeros((H,D),np.float64); self.dir_n[ek]=np.zeros(H,np.float64)
        if bk not in self.base_sum: self.base_sum[bk]=np.zeros(H,np.float64)
    def update(self,L,site,z00,z10,z01,z11,sign):
        a00=z00.numpy(); a10=z10.numpy(); a01=z01.numpy(); a11=z11.numpy(); H,D=a00.shape
        R=(a10+a11-a00-a01)/4.0; V=(a01+a11-a00-a10)/4.0; I=(a00+a11-a10-a01)/4.0
        eff={"role":R,"visual":V,"interaction":I}
        base=(np.sum(a00*a00,1)+np.sum(a10*a10,1)+np.sum(a01*a01,1)+np.sum(a11*a11,1))/(4.0*D)
        self.base_sum.setdefault((L,site),np.zeros(H,np.float64)); self.base_sum[(L,site)] += base; self.n[(L,site)] += 1
        for name,e in eff.items():
            self._ensure(L,site,name,H,D); key=(L,site,name)
            self.e_sum[key] += np.sum(e*e,1)/D
            aligned=e*float(sign); norms=np.linalg.norm(aligned,axis=1); good=norms>1e-12
            unit=np.zeros_like(aligned,dtype=np.float64); unit[good]=aligned[good]/norms[good,None]
            self.dir_sum[key] += unit; self.dir_n[key] += good.astype(np.float64)
    def finalize(self):
        rows=[]
        for (L,site),N in sorted(self.n.items()):
            base=self.base_sum[(L,site)]/N
            energies={e:self.e_sum[(L,site,e)]/N for e in EFFECTS}
            total=energies["role"]+energies["visual"]+energies["interaction"]
            H=len(base)
            for h in range(H):
                for e in EFFECTS:
                    en=float(energies[e][h]); b=float(base[h]); strength=math.sqrt(max(en,0.0)/max(b,1e-12)); share=en/max(float(total[h]),1e-12)
                    dn=float(self.dir_n[(L,site,e)][h]); consistency=float(np.linalg.norm(self.dir_sum[(L,site,e)][h])/dn) if dn>0 else float("nan")
                    rows.append(dict(version=VERSION,layer=L,head=h,site=site,effect=e,N=N,effect_energy=en,baseline_energy=b,effect_strength=strength,effect_share=share,direction_consistency=consistency,functional_score=strength*share))
        # objects_mean from A/B at energy level
        lookup={(int(r["layer"]),int(r["head"]),str(r["site"]),str(r["effect"])):r for r in rows}
        extra=[]
        for L in sorted(set(int(r["layer"]) for r in rows)):
            heads=sorted(set(int(r["head"]) for r in rows if int(r["layer"])==L))
            for h in heads:
                if any((L,h,s,e) not in lookup for s in ("A","B") for e in EFFECTS): continue
                vals={}
                for e in EFFECTS:
                    ra=lookup[(L,h,"A",e)]; rb=lookup[(L,h,"B",e)]
                    vals[e]=dict(en=(float(ra["effect_energy"])+float(rb["effect_energy"]))/2,b=(float(ra["baseline_energy"])+float(rb["baseline_energy"]))/2,c=(float(ra["direction_consistency"])+float(rb["direction_consistency"]))/2,N=min(int(ra["N"]),int(rb["N"])))
                total=sum(vals[e]["en"] for e in EFFECTS)
                for e in EFFECTS:
                    en=vals[e]["en"]; b=vals[e]["b"]; strength=math.sqrt(max(en,0.0)/max(b,1e-12)); share=en/max(total,1e-12)
                    extra.append(dict(version=VERSION,layer=L,head=h,site="objects_mean",effect=e,N=vals[e]["N"],effect_energy=en,baseline_energy=b,effect_strength=strength,effect_share=share,direction_consistency=vals[e]["c"],functional_score=strength*share))
        return rows+extra


def print_top(rows,effect,site,k):
    vals=[dict(r) for r in rows if r["effect"]==effect and r["site"]==site]
    vals.sort(key=lambda r:float(r["functional_score"]),reverse=True)
    print("\n"+"="*112); print(f"TOP {effect.upper()} HEADS @ {site}"); print("="*112)
    print(f"{'#':>3} {'head':>9} {'score':>9} {'strength':>10} {'share':>8} {'consist':>9} {'N':>5}")
    for i,r in enumerate(vals[:k],1):
        print(f"{i:>3d} L{int(r['layer']):02d}H{int(r['head']):02d} {float(r['functional_score']):>9.4f} {float(r['effect_strength']):>10.4f} {float(r['effect_share']):>8.3f} {float(r['direction_consistency']):>9.3f} {int(r['N']):>5d}")


def main():
    args=parse_args(); layers=parse_layers(args.layers)
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    random.seed(args.sample_seed); np.random.seed(args.sample_seed); torch.manual_seed(args.sample_seed)
    out=Path(args.output_dir)
    if args.overwrite and out.exists(): shutil.rmtree(out)
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"output dir not empty: {out}")
    out.mkdir(parents=True,exist_ok=True); errors=out/"errors.jsonl"

    fliphelper=import_file(Path(args.flip_helper),"_headfun_fliphelper")
    ioi=import_file(Path(args.ioi_script),"_headfun_ioi")
    producer=import_file(Path(args.producer_script),"_headfun_prod")
    receiver=import_file(Path(args.receiver_script),"_headfun_recv")
    v3=import_file(Path(args.v3_script),"_headfun_v3")
    base=import_file(Path(args.base_script),"_headfun_base")

    rows=load_rows(Path(args.source_output_dir)/"extraction.jsonl",Path(args.flip_pairs_jsonl),args.base_pair_status,args.flip_status)
    rows=stratified_subset(rows,args.max_samples,args.sample_seed)
    if not rows: raise RuntimeError("no samples after filtering")

    model=processor=capture=None; acc=Accumulator(); sample_rows=[]; successful=quad=0; device=torch.device(args.device)
    try:
        model,processor,spec,decoder_layers,decoder_path,relation_token_map=producer.load_model_bundle(args=args,base=base); model.eval()
        for L in layers:
            if not 0<=L<len(decoder_layers): raise ValueError(f"L{L} outside 0..{len(decoder_layers)-1}")
        saved=getattr(args,"max_samples",None); args.max_samples=None
        try: records_by_sid,prompt_rows,audit=ioi.prepare_data_helpers(args,base)
        finally: args.max_samples=saved
        capture=Capture(model,decoder_layers,layers)
        shapes={str(L):dict(num_heads=capture.shapes[L][0],head_dim=capture.shapes[L][1]) for L in layers}
        print("\n"+"="*126); print("TRAINING-FREE 2x2 HEAD FUNCTION DISCOVERY"); print("="*126)
        print("model          :",args.model); print("layers         :",layers); print("candidate N    :",len(rows)); print("metric subset  :",args.metric_subset); print("head state     : pre-WO attention head output"); print("head shapes    :",shapes); print("="*126,flush=True)

        for idx,row in enumerate(tqdm(rows,desc="2x2-head-scan"),1):
            pair=None
            try:
                pair=receiver.prepare_pair(args=args,row=row,records_by_sid=records_by_sid,prompt_rows=prompt_rows,base=base,v3=v3,processor=processor,device=device)
                image,_=fliphelper.infer_image(row,records_by_sid,pair); flipped=fliphelper.horizontal_flip_pil(image)
                b00=pair.original_batch; b10=pair.swapped_batch
                b01=fliphelper.build_flip_batch_from_original(processor=processor,original_batch=b00,flipped_image=flipped,device=device)
                b11=fliphelper.build_flip_batch_from_original(processor=processor,original_batch=b10,flipped_image=flipped,device=device)
                pos0=dict(A=pair.original_a_positions,B=pair.original_b_positions,last=[pair.original_prompt_last])
                pos1=dict(A=pair.swapped_a_positions,B=pair.swapped_b_positions,last=[pair.swapped_prompt_last])
                z00,p00,_=run_condition(model,b00,capture,pos0,relation_token_map)
                z10,p10,_=run_condition(model,b10,capture,pos1,relation_token_map)
                z01,p01,_=run_condition(model,b01,capture,pos0,relation_token_map)
                z11,p11,_=run_condition(model,b11,capture,pos1,relation_token_map)
                gt=str(row["gt"]); opp=OPPOSITE[gt]
                c00=p00==gt; c10=p10==opp; c01=p01==opp; c11=p11==gt; qc=c00 and c10 and c01 and c11
                successful+=1; quad+=int(qc); use=(args.metric_subset=="all_successful" or qc)
                if use:
                    sign=1.0 if gt=="left" else -1.0
                    for L in layers:
                        for site in SITES:
                            acc.update(L,site,z00[L][site],z10[L][site],z01[L][site],z11[L][site],sign)
                sr=dict(sid=int(row["sid"]),gt=gt,pred_C00=p00,pred_C10=p10,pred_C01=p01,pred_C11=p11,correct_C00=c00,correct_C10=c10,correct_C01=c01,correct_C11=c11,quad_correct=qc,used_for_metrics=use)
                sample_rows.append(sr); append_jsonl(out/"sample_quad_status.jsonl",sr)
            except Exception as exc:
                append_jsonl(errors,dict(sid=int(row["sid"]),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
                if args.fail_fast: raise
            finally:
                if pair is not None: receiver.release_pair(pair)
                if torch.cuda.is_available() and args.empty_cache_every>0 and idx%args.empty_cache_every==0: torch.cuda.empty_cache()

        scores=acc.finalize()
        if not scores: raise RuntimeError("no scores; quad_correct subset may be empty")
        write_csv(out/"head_function_scores_all.csv",scores); write_csv(out/"sample_quad_status.csv",sample_rows)
        tops=[]
        for site in ("objects_mean","last"):
            for effect in EFFECTS:
                vals=[dict(r) for r in scores if r["site"]==site and r["effect"]==effect]; vals.sort(key=lambda r:float(r["functional_score"]),reverse=True)
                for rank,r in enumerate(vals[:args.top_k],1): rr=dict(r); rr["rank"]=rank; tops.append(rr)
        write_csv(out/"top_heads_by_function.csv",tops)
        summary=dict(version=VERSION,model=args.model,layers=layers,candidate_rows=len(rows),successful_rows=successful,quad_correct_rows=quad,quad_correct_rate=quad/successful if successful else None,metric_subset=args.metric_subset,metric_N=quad if args.metric_subset=="quad_correct" else successful,head_shapes=shapes,definitions=dict(role="(C10+C11-C00-C01)/4",visual="(C01+C11-C00-C10)/4",interaction="(C00+C11-C10-C01)/4",functional_score="effect_strength * effect_share"),scope="natural functional discovery, not causal proof")
        write_json(out/"summary.json",summary)
        print("\nQUAD AUDIT: successful=%d quad_correct=%d (%.2f%%) ranking_N=%s"%(successful,quad,100*quad/successful if successful else float('nan'),summary["metric_N"]))
        for e in EFFECTS: print_top(scores,e,"objects_mean",args.top_k)
        for e in EFFECTS: print_top(scores,e,"last",min(args.top_k,10))
        (out/"report.txt").write_text("\n".join([f"version: {VERSION}",f"layers: {layers}",f"successful: {successful}",f"quad_correct: {quad}",f"metric_subset: {args.metric_subset}","","ROLE        = (C10+C11-C00-C01)/4","VISUAL      = (C01+C11-C00-C10)/4","INTERACTION = (C00+C11-C10-C01)/4","","High scores are natural signatures only. Next: natural activation patching on sparse candidates + matched controls."])+"\n",encoding="utf-8")
        print("\nSaved:"); [print(" ",out/n) for n in ("summary.json","head_function_scores_all.csv","top_heads_by_function.csv","sample_quad_status.csv","report.txt")]
    finally:
        if capture is not None: capture.close()
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
