#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-contained Stage-III QUERY-SWAP receiver discovery + TopK bundle patch.

No producer_top_heads.json required.
No pre-existing receiver_head_scan.csv required.

Workflow
========
1) Build same-image counterfactual pair:
       target : A relative to B = r
       source : B relative to A = opposite(r)

2) Receiver discovery scan over --scan-layers (default 24-35):
   For every query head h, compute natural object->prompt-last post-W_O write

       c_h = W_O^h sum_{s in object tokens} A_h[last,s] V_h[s]

   in original and swapped branches, then rank by how much

       delta_h = c_h(original) - c_h(swapped)

   projects onto the full prompt-last block difference

       Delta_block = block_last(original) - block_last(swapped)

   score = <delta_h, Delta_block> / ||Delta_block||^2

   This is the same receiver-discovery metric family as
   analyze_coco_receiver_qkv_v1.py's mean_projection_fraction_block.

3) Take cumulative Top5/Top10/Top15/Top20 query-head bundles.

4) On ORIGINAL run, simultaneously transplant the SWAPPED branch's receiver
   projection channels:

   Q: swapped prompt-last q_proj -> original prompt-last
   K: swapped identity-aligned object k_proj -> original A/B
   V: swapped identity-aligned object v_proj -> original A/B

   Modes: q, k, v, qkv.  K/V are GQA-deduplicated.

5) Compare selected TopK to same-layer matched-random TopK.

Primary metric
==============
source_follow_rate = P(patched prediction == swapped-query answer)

source_progress =
    (M_original - M_patch) / (M_original - M_swapped)
where M = logit(GT) - logit(opposite).

Recommended command
===================
CUDA_VISIBLE_DEVICES=0 python -u \
validate_stage3_query_swap_receiver_bundle_autoscan_v2.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --scan-layers 24-35 \
  --scan-max-samples 100 \
  --bundle-sizes 5,10,15,20 \
  --patch-modes q,k,v,qkv \
  --baseline-filter both_correct \
  --max-samples 0 \
  --device cuda:0 \
  --output-dir output/qwen3b_stage3_queryswap_receiver_autoscan_v2 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

VERSION = "stage3-query-swap-receiver-bundle-autoscan-v2"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {x: i for i, x in enumerate(RELATIONS)}
OPPOSITE = {"left":"right", "right":"left", "above":"below", "below":"above"}
PATCH_MODES = ("q", "k", "v", "qkv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--source-output-dir", default="output/spatial_storage_transport_utilization/coco/qwen-3b")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", choices=("last","mean"), default="mean")

    p.add_argument("--scan-layers", default="24-35", help="e.g. 24-35 or 24,25,26,27")
    p.add_argument("--scan-max-samples", type=int, default=100, help="0 = all eligible pairs")
    p.add_argument("--scan-status", choices=("both_correct","original_correct","all"), default="both_correct")
    p.add_argument("--rank-metric", choices=("mean_projection_fraction_block","mean_abs_projection_fraction_block","mean_delta_norm","mean_object_attention_mass","positive_projection_rate"), default="mean_projection_fraction_block")

    p.add_argument("--bundle-sizes", default="5,10,15,20")
    p.add_argument("--patch-modes", default="q,k,v,qkv")
    p.add_argument("--baseline-filter", choices=("both_correct","original_correct","all"), default="both_correct")
    p.add_argument("--relation-scope", choices=("lr","all4"), default="all4")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument("--random-seed", type=int, default=303)
    p.add_argument("--min-margin-denominator", type=float, default=1e-6)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--ioi-script", default="analyze_coco_ioi_backward_circuit_v1.py")
    p.add_argument("--producer-script", default="analyze_coco_producer_qk_ov_v1.py")
    p.add_argument("--receiver-script", default="analyze_coco_receiver_qkv_v1.py")
    p.add_argument("--v3-script", default="analyze_spatial_storage_transport_utilization_v3.py")
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--attention-helper", default="analyze_coco_flip_attention_spatial_vectors_v1.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_file(path: Path, name: str):
    if not path.exists(): raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def read_jsonl(path):
    out=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out


def append_jsonl(path,row):
    with Path(path).open("a",encoding="utf-8") as f: f.write(json.dumps(dict(row),ensure_ascii=False)+"\n")


def write_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def write_csv(path,rows):
    path=Path(path)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def safe_mean(xs):
    a=np.asarray(list(xs),dtype=np.float64); a=a[np.isfinite(a)]; return float(a.mean()) if a.size else float("nan")

def safe_std(xs):
    a=np.asarray(list(xs),dtype=np.float64); a=a[np.isfinite(a)]; return float(a.std()) if a.size else float("nan")

def hname(h): return f"L{int(h[0])}H{int(h[1])}"


def parse_layers(text,n_layers):
    vals=[]
    for raw in str(text).split(","):
        raw=raw.strip()
        if not raw: continue
        if "-" in raw:
            a,b=raw.split("-",1); vals.extend(range(int(a),int(b)+1))
        else: vals.append(int(raw))
    vals=sorted(set(vals))
    bad=[x for x in vals if x<0 or x>=n_layers]
    if bad: raise ValueError(f"scan layers outside 0..{n_layers-1}: {bad}")
    if not vals: raise ValueError("No scan layers")
    return vals


def parse_sizes(text,total):
    vals=sorted(set(int(x.strip()) for x in str(text).split(",") if x.strip()))
    if not vals or min(vals)<1: raise ValueError("bad bundle sizes")
    if max(vals)>total: raise ValueError(f"Top{max(vals)} requested but only {total} heads ranked")
    return vals


def parse_modes(text):
    vals=[]
    for x in str(text).split(","):
        x=x.strip().lower()
        if not x: continue
        if x not in PATCH_MODES: raise ValueError(x)
        if x not in vals: vals.append(x)
    return vals


def stratified(rows,limit,seed):
    rows=[dict(r) for r in rows]
    if limit<=0 or limit>=len(rows): return rows
    rng=random.Random(seed); g=defaultdict(list)
    for r in rows: g[str(r["gt"])].append(r)
    for v in g.values(): rng.shuffle(v)
    keys=sorted(g); idx={k:0 for k in keys}; out=[]
    while len(out)<limit:
        moved=False
        for k in keys:
            if len(out)>=limit: break
            i=idx[k]
            if i<len(g[k]): out.append(g[k][i]); idx[k]+=1; moved=True
        if not moved: break
    rng.shuffle(out); return out


def margin(logits,gt): return float(logits[REL_TO_ID[gt]]-logits[REL_TO_ID[OPPOSITE[gt]]])


def status_ok(mode,orig_ok,swap_ok):
    if mode=="both_correct": return orig_ok and swap_ok
    if mode=="original_correct": return orig_ok
    return True


# ---------- receiver discovery ----------
@torch.inference_mode()
def scan_one_pair(*, pair, scan_layers, model, base, relation_token_map, decoder_layers, v3, attention_helper, receiver):
    targets_orig=sorted(set(pair.original_object_positions+[pair.original_prompt_last]))
    targets_swap=sorted(set(pair.swapped_object_positions+[pair.swapped_prompt_last]))

    original_result, original_traces = v3.trace_prompt_chunks(
        attention_helper=attention_helper, model=model, batch=pair.original_batch,
        relation_token_map=relation_token_map, decoder_layers=decoder_layers,
        layers=scan_layers, target_positions=targets_orig, chunk_size=max(1,min(4,len(scan_layers))))
    swapped_result, swapped_traces = v3.trace_prompt_chunks(
        attention_helper=attention_helper, model=model, batch=pair.swapped_batch,
        relation_token_map=relation_token_map, decoder_layers=decoder_layers,
        layers=scan_layers, target_positions=targets_swap, chunk_size=max(1,min(4,len(scan_layers))))

    rows=[]
    for layer in scan_layers:
        ot=original_traces[layer]; st=swapped_traces[layer]
        # trace block_output is captured at target positions; lookup prompt-last local index.
        o_lookup={int(p):i for i,p in enumerate(ot.target_positions)}
        s_lookup={int(p):i for i,p in enumerate(st.target_positions)}
        oi=o_lookup[int(pair.original_prompt_last)]; si=s_lookup[int(pair.swapped_prompt_last)]
        block_delta=ot.block_output[oi].float()-st.block_output[si].float()
        denom=float(block_delta.pow(2).sum())

        attn=attention_helper.resolve_self_attention(decoder_layers[layer])
        shape=receiver.resolve_attention_shape(attn)
        for h in range(int(shape.n_query_heads)):
            ow,om=receiver.object_head_write(trace=ot,head=h,target_position=pair.original_prompt_last,source_positions=pair.original_object_positions)
            sw,sm=receiver.object_head_write(trace=st,head=h,target_position=pair.swapped_prompt_last,source_positions=pair.swapped_object_positions)
            d=ow.float()-sw.float()
            proj=float(torch.dot(d,block_delta)/denom) if denom>1e-12 else float("nan")
            rows.append({
                "sid":int(pair.sid),"layer":int(layer),"head":int(h),
                "kv_head":int(shape.kv_head_for_query(h)),
                "projection_fraction_block":proj,
                "abs_projection_fraction_block":abs(proj) if math.isfinite(proj) else float("nan"),
                "object_write_delta_norm":float(d.norm()),
                "object_attention_mass_mean":0.5*(float(om)+float(sm)),
                "positive_projection":bool(math.isfinite(proj) and proj>0),
            })
    return original_result,swapped_result,rows


def aggregate_scan(rows,metric):
    g=defaultdict(list)
    for r in rows: g[(int(r["layer"]),int(r["head"]))].append(r)
    out=[]
    for (l,h),v in g.items():
        item={
            "layer":l,"head":h,"kv_head":int(v[0]["kv_head"]),"N":len(v),
            "mean_projection_fraction_block":safe_mean(x["projection_fraction_block"] for x in v),
            "mean_abs_projection_fraction_block":safe_mean(x["abs_projection_fraction_block"] for x in v),
            "mean_delta_norm":safe_mean(x["object_write_delta_norm"] for x in v),
            "mean_object_attention_mass":safe_mean(x["object_attention_mass_mean"] for x in v),
            "positive_projection_rate":safe_mean(float(x["positive_projection"]) for x in v),
        }
        out.append(item)
    out.sort(key=lambda r:(-float(r[metric]) if math.isfinite(float(r[metric])) else float("inf"),int(r["layer"]),int(r["head"])))
    for i,r in enumerate(out,1): r["rank"]=i
    return out


# ---------- patching ----------
def validate_heads(heads,decoder_layers,attention_helper,receiver):
    for l,h in heads:
        attn=attention_helper.resolve_self_attention(decoder_layers[l]); shape=receiver.resolve_attention_shape(attn)
        if h<0 or h>=int(shape.n_query_heads): raise ValueError(hname((l,h)))


def random_order(selected,decoder_layers,attention_helper,receiver,seed):
    rng=random.Random(seed); excluded=set(selected); used=set(); out=[]
    for l,_ in selected:
        attn=attention_helper.resolve_self_attention(decoder_layers[l]); shape=receiver.resolve_attention_shape(attn)
        cand=[(l,h) for h in range(int(shape.n_query_heads)) if (l,h) not in excluded and (l,h) not in used]
        if not cand: raise RuntimeError(f"No matched random at L{l}")
        c=rng.choice(cand); used.add(c); out.append(c)
    return out


def channels(mode): return ["q","k","v"] if mode=="qkv" else [mode]

def units_for(heads,mode,decoder_layers,attention_helper,receiver):
    return receiver.causal_units_from_query_heads(query_heads=heads,decoder_layers=decoder_layers,attention_helper=attention_helper,channels=channels(mode))


@torch.inference_mode()
def multi_patch(*,units,pair,original_states,swapped_states,model,base,relation_token_map,decoder_layers,attention_helper,receiver):
    patches=[]
    try:
        for u in units:
            l=int(u["layer"]); ch=str(u["channel"])
            attn=attention_helper.resolve_self_attention(decoder_layers[l]); module=receiver.projection_module(attn,ch)
            side,mapping=receiver.patch_position_map(pair=pair,channel=ch,condition="corrupt_on_original",original_states=original_states,swapped_states=swapped_states,layer=l)
            if side!="original": raise RuntimeError(side)
            p=receiver.ProjectionHeadPatch(module=module,head=int(u["unit_head"]),head_dim=int(u["head_dim"]),target_to_source=mapping)
            patches.append(p)
        result=receiver.run_scores(model=model,batch=pair.original_batch,base=base,relation_token_map=relation_token_map)
        if any(not p.applied for p in patches): raise RuntimeError("Some bundle patches did not fire")
        return result
    finally:
        for p in reversed(patches):
            with contextlib.suppress(Exception): p.close()


def result_row(*,sid,gt,family,mode,k,heads,units,clean,source,patched,min_den):
    sg=OPPOSITE[gt]; cm=margin(clean["logits"],gt); sm=margin(source["logits"],gt); pm=margin(patched["logits"],gt); den=cm-sm
    prog=(cm-pm)/den if abs(den)>=min_den else float("nan")
    pp=str(patched["prediction"]); cp=str(clean["prediction"])
    return {
        "sid":sid,"gt_target":gt,"gt_source":sg,"family":family,"patch_mode":mode,"K_query_heads":k,
        "query_heads":",".join(hname(x) for x in heads),"n_unique_units":len(units),
        "n_q":sum(str(u["channel"])=="q" for u in units),"n_k":sum(str(u["channel"])=="k" for u in units),"n_v":sum(str(u["channel"])=="v" for u in units),
        "clean_prediction":cp,"source_prediction":str(source["prediction"]),"patched_prediction":pp,
        "clean_margin":cm,"source_margin":sm,"patched_margin":pm,"source_progress":prog,
        "source_follow":pp==sg,"crossed":cm>0 and pm<=0,"changed":pp!=cp,
        "source_side":abs(pm-sm)<abs(pm-cm),"other":pp not in (gt,sg),
    }


def summarize(rows):
    g=defaultdict(list)
    for r in rows:g[(r["family"],r["patch_mode"],int(r["K_query_heads"]))].append(r)
    out=[]
    for (fam,mode,k),v in g.items():
        out.append({"family":fam,"patch_mode":mode,"K_query_heads":k,"N":len(v),"query_heads":v[0]["query_heads"],"n_unique_units":v[0]["n_unique_units"],"n_q":v[0]["n_q"],"n_k":v[0]["n_k"],"n_v":v[0]["n_v"],
                    "source_follow_rate":safe_mean(float(x["source_follow"]) for x in v),"crossed_rate":safe_mean(float(x["crossed"]) for x in v),"changed_rate":safe_mean(float(x["changed"]) for x in v),"source_side_rate":safe_mean(float(x["source_side"]) for x in v),"other_rate":safe_mean(float(x["other"]) for x in v),"mean_source_progress":safe_mean(float(x["source_progress"]) for x in v),"std_source_progress":safe_std(float(x["source_progress"]) for x in v)})
    mo={x:i for i,x in enumerate(PATCH_MODES)}; fo={"selected":0,"random":1}
    out.sort(key=lambda r:(r["K_query_heads"],mo[r["patch_mode"]],fo[r["family"]])); return out


def print_summary(rows):
    print("\n"+"="*145); print("STAGE-III QUERY-SWAP AUTO-SCAN + TOPK BUNDLE PATCH"); print("="*145)
    print(f"{'family':<9} {'mode':<4} {'K':>3} {'units':>5} {'Q/K/V':>10} {'N':>5} {'srcFollow':>10} {'crossed':>9} {'changed':>9} {'srcSide':>9} {'progress':>10}")
    print("-"*145)
    for r in rows:
        qkv=f"{r['n_q']}/{r['n_k']}/{r['n_v']}"
        print(f"{r['family']:<9} {r['patch_mode']:<4} {r['K_query_heads']:>3} {r['n_unique_units']:>5} {qkv:>10} {r['N']:>5} {100*r['source_follow_rate']:>9.2f}% {100*r['crossed_rate']:>8.2f}% {100*r['changed_rate']:>8.2f}% {100*r['source_side_rate']:>8.2f}% {r['mean_source_progress']:>+10.3f}")
    print("="*145)


def main():
    args=parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    random.seed(args.sample_seed); np.random.seed(args.sample_seed); torch.manual_seed(args.sample_seed)
    outdir=Path(args.output_dir)
    if args.overwrite and outdir.exists(): shutil.rmtree(outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    ioi=import_file(Path(args.ioi_script),"_s3auto_ioi")
    producer=import_file(Path(args.producer_script),"_s3auto_prod")
    receiver=import_file(Path(args.receiver_script),"_s3auto_recv")
    v3=import_file(Path(args.v3_script),"_s3auto_v3")
    base=import_file(Path(args.base_script),"_s3auto_base")
    attention_helper=import_file(Path(args.attention_helper),"_s3auto_attn")

    extraction=Path(args.source_output_dir)/"extraction.jsonl"
    if not extraction.exists(): raise FileNotFoundError(extraction)
    allowed={"left","right"} if args.relation_scope=="lr" else set(RELATIONS)
    allrows=[r for r in read_jsonl(extraction) if str(r.get("gt")) in allowed]

    model=processor=None
    try:
        model,processor,spec,decoder_layers,decoder_path,relation_token_map=producer.load_model_bundle(args=args,base=base); model.eval()
        scan_layers=parse_layers(args.scan_layers,len(decoder_layers))

        analysis_max=args.max_samples; args.max_samples=None
        try: records_by_sid,prompt_rows,audit=ioi.prepare_data_helpers(args,base)
        finally: args.max_samples=analysis_max

        # ---------------- discovery scan ----------------
        scan_candidates=stratified(allrows,args.scan_max_samples,args.sample_seed+101)
        scan_rows=[]; scan_pair_audit=[]
        print(f"Receiver auto-scan: layers={scan_layers}, requested N={len(scan_candidates)}",flush=True)
        for row in tqdm(scan_candidates,desc="receiver-autoscan"):
            pair=None
            try:
                pair=receiver.prepare_pair(args=args,row=row,records_by_sid=records_by_sid,prompt_rows=prompt_rows,base=base,v3=v3,processor=processor,device=torch.device(args.device))
                clean,src,rows=scan_one_pair(pair=pair,scan_layers=scan_layers,model=model,base=base,relation_token_map=relation_token_map,decoder_layers=decoder_layers,v3=v3,attention_helper=attention_helper,receiver=receiver)
                gt=str(pair.gt); sg=OPPOSITE[gt]; ook=str(clean["prediction"])==gt; sok=str(src["prediction"])==sg
                keep=status_ok(args.scan_status,ook,sok)
                scan_pair_audit.append({"sid":int(pair.sid),"gt":gt,"original_prediction":str(clean["prediction"]),"swapped_prediction":str(src["prediction"]),"original_correct":ook,"swapped_correct":sok,"kept":keep})
                if keep: scan_rows.extend(rows)
            finally:
                if pair is not None: receiver.release_pair(pair)
                gc.collect()

        if not scan_rows: raise RuntimeError("No receiver-scan rows after filter")
        ranking=aggregate_scan(scan_rows,args.rank_metric)
        sizes=parse_sizes(args.bundle_sizes,len(ranking)); maxk=max(sizes)
        selected=[(int(r["layer"]),int(r["head"])) for r in ranking[:maxk]]
        validate_heads(selected,decoder_layers,attention_helper,receiver)
        random_heads=random_order(selected,decoder_layers,attention_helper,receiver,args.random_seed)
        modes=parse_modes(args.patch_modes)

        write_csv(outdir/"receiver_head_scan.csv",ranking)
        write_csv(outdir/"receiver_scan_pair_audit.csv",scan_pair_audit)
        write_csv(outdir/"top_receiver_heads_used.csv",ranking[:maxk])
        write_csv(outdir/"matched_random_heads.csv",[{"rank":i+1,"selected":hname(selected[i]),"random":hname(random_heads[i])} for i in range(maxk)])

        print("Top receiver heads:")
        for r in ranking[:maxk]: print(f"  {r['rank']:02d} {hname((r['layer'],r['head']))} {args.rank_metric}={float(r[args.rank_metric]):+.6f}")

        # ---------------- bundle causal ----------------
        eval_rows=stratified(allrows,args.max_samples,args.sample_seed)
        capture_layers=sorted(set(l for l,_ in selected+random_heads))
        sample_rows=[]; base_audit=[]; eligible_n=0
        for idx,row in enumerate(tqdm(eval_rows,desc="stage3-queryswap-topk"),1):
            pair=None
            try:
                pair=receiver.prepare_pair(args=args,row=row,records_by_sid=records_by_sid,prompt_rows=prompt_rows,base=base,v3=v3,processor=processor,device=torch.device(args.device))
                clean,source,ostates,sstates=receiver.capture_pair_projections(pair=pair,layers=capture_layers,model=model,base=base,relation_token_map=relation_token_map,decoder_layers=decoder_layers,attention_helper=attention_helper)
                gt=str(pair.gt); sg=OPPOSITE[gt]; ook=str(clean["prediction"])==gt; sok=str(source["prediction"])==sg
                cm=margin(clean["logits"],gt); sm=margin(source["logits"],gt); keep=status_ok(args.baseline_filter,ook,sok)
                base_audit.append({"sid":int(pair.sid),"gt_target":gt,"gt_source":sg,"original_prediction":str(clean["prediction"]),"swapped_prediction":str(source["prediction"]),"original_correct":ook,"swapped_correct":sok,"clean_margin":cm,"source_margin":sm,"eligible":keep})
                if not keep or abs(cm-sm)<args.min_margin_denominator: continue
                eligible_n+=1
                for k in sizes:
                    for fam,heads in (("selected",selected[:k]),("random",random_heads[:k])):
                        for mode in modes:
                            units=units_for(heads,mode,decoder_layers,attention_helper,receiver)
                            patched=multi_patch(units=units,pair=pair,original_states=ostates,swapped_states=sstates,model=model,base=base,relation_token_map=relation_token_map,decoder_layers=decoder_layers,attention_helper=attention_helper,receiver=receiver)
                            rr=result_row(sid=int(pair.sid),gt=gt,family=fam,mode=mode,k=k,heads=heads,units=units,clean=clean,source=source,patched=patched,min_den=args.min_margin_denominator)
                            sample_rows.append(rr); append_jsonl(outdir/"sample_patch_results.jsonl",rr)
                if args.print_every>0 and eligible_n%args.print_every==0:
                    r=next(x for x in reversed(sample_rows) if x["family"]=="selected" and x["patch_mode"]==modes[-1] and x["K_query_heads"]==sizes[-1])
                    tqdm.write(f"N={eligible_n} sid={pair.sid} {gt}->{sg} Top{sizes[-1]} {modes[-1]} pred={r['patched_prediction']} progress={r['source_progress']:+.3f}")
            except Exception as e:
                append_jsonl(outdir/"errors.jsonl",{"sid":int(row.get("sid",-1)),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
                if args.fail_fast: raise
            finally:
                if pair is not None: receiver.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache_every>0 and idx%args.empty_cache_every==0: torch.cuda.empty_cache()

        if not sample_rows: raise RuntimeError("No causal results")
        summ=summarize(sample_rows)
        write_csv(outdir/"baseline_audit.csv",base_audit); write_csv(outdir/"sample_patch_results.csv",sample_rows); write_csv(outdir/"summary.csv",summ)
        print_summary(summ)
        write_json(outdir/"config.json",{"version":VERSION,"model":args.model,"decoder_path":decoder_path,"scan_layers":scan_layers,"scan_status":args.scan_status,"scan_max_samples":args.scan_max_samples,"rank_metric":args.rank_metric,"selected_heads":[hname(h) for h in selected],"random_heads":[hname(h) for h in random_heads],"bundle_sizes":sizes,"patch_modes":modes,"baseline_filter":args.baseline_filter,"N_intervention_eligible":eligible_n,"counterfactual":"same image; A relative B -> B relative A","audit":audit})
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=="__main__": main()
