#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-III QUERY-SWAP multi-head receiver causal patch.

TARGET: A relative to B, same image.
SOURCE: B relative to A, same image.

This reproduces the old receiver Q/K/V intervention style, but applies it to
cumulative TopK receiver QUERY-head bundles (default Top5/10/15/20).

Q: swapped prompt-last q_proj -> original prompt-last.
K: swapped identity-aligned object k_proj -> original object tokens.
V: swapped identity-aligned object v_proj -> original object tokens.
QKV: all three simultaneously.

K/V are deduplicated automatically under GQA by reusing
analyze_coco_receiver_qkv_v1.py::causal_units_from_query_heads().

By default, receiver heads are ranked from:
  output/coco_receiver_qkv/qwen-3b/receiver_head_scan.csv
using descending mean_projection_fraction_block.

Example:
CUDA_VISIBLE_DEVICES=0 python -u validate_stage3_query_swap_receiver_bundle_topk_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --receiver-scan-file output/coco_receiver_qkv/qwen-3b/receiver_head_scan.csv \
  --bundle-sizes 5,10,15,20 \
  --patch-modes q,k,v,qkv \
  --baseline-filter both_correct \
  --max-samples 0 \
  --device cuda:0 \
  --output-dir output/qwen3b_stage3_queryswap_receiver_topk_v1 \
  --overwrite
"""
from __future__ import annotations

import argparse, contextlib, csv, gc, importlib.util, json, math, random, shutil, sys, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

VERSION = "stage3-query-swap-receiver-bundle-topk-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {x: i for i, x in enumerate(RELATIONS)}
OPPOSITE = {"left":"right", "right":"left", "above":"below", "below":"above"}
PATCH_MODES = ("q", "k", "v", "qkv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--source-output-dir", default="output/spatial_storage_transport_utilization/coco/qwen-3b")
    p.add_argument("--receiver-scan-file", default="output/coco_receiver_qkv/qwen-3b/receiver_head_scan.csv")
    p.add_argument("--rank-metric", default="mean_projection_fraction_block")
    p.add_argument("--heads", default="", help="Optional explicit ordered query heads, e.g. 26:4,26:2,26:6")
    p.add_argument("--bundle-sizes", default="5,10,15,20")
    p.add_argument("--patch-modes", default="q,k,v,qkv")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", default="mean", choices=("last", "mean"))
    p.add_argument("--baseline-filter", default="both_correct", choices=("both_correct","original_correct","all"))
    p.add_argument("--relation-scope", default="all4", choices=("lr","all4"))
    p.add_argument("--max-samples", type=int, default=0, help="0 = all selected source-cache rows")
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument("--random-seed", type=int, default=303)
    p.add_argument("--min-margin-denominator", type=float, default=1e-6)
    p.add_argument("--print-every", type=int, default=1)
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


def read_jsonl(path: Path):
    out=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: Mapping[str, Any]):
    with path.open("a", encoding="utf-8") as f: f.write(json.dumps(dict(row), ensure_ascii=False)+"\n")


def write_json(path: Path, obj: Any): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    if not rows: path.write_text("", encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def safe_mean(xs: Iterable[float]):
    a=np.asarray(list(xs), dtype=np.float64); a=a[np.isfinite(a)]; return float(a.mean()) if a.size else float("nan")

def safe_std(xs: Iterable[float]):
    a=np.asarray(list(xs), dtype=np.float64); a=a[np.isfinite(a)]; return float(a.std(ddof=0)) if a.size else float("nan")

def hname(h): return f"L{int(h[0])}H{int(h[1])}"


def parse_heads(text: str):
    out=[]; seen=set()
    for p in text.split(","):
        p=p.strip().upper().replace("L","").replace("H",":")
        if not p: continue
        a,b=p.split(":",1); h=(int(a),int(b))
        if h not in seen: seen.add(h); out.append(h)
    return out


def parse_sizes(text: str, total: int):
    vals=[]
    for p in text.split(","):
        p=p.strip().lower()
        if not p: continue
        k=total if p=="all" else int(p)
        if k<1 or k>total: raise ValueError(f"Invalid Top{k}; available={total}")
        vals.append(k)
    return sorted(set(vals))


def parse_modes(text: str):
    out=[]
    for p in text.split(","):
        p=p.strip().lower()
        if not p: continue
        if p not in PATCH_MODES: raise ValueError(f"Bad patch mode {p}; choose {PATCH_MODES}")
        if p not in out: out.append(p)
    return out


def stratified_subset(rows, limit, seed):
    rows=[dict(r) for r in rows]
    if limit<=0 or limit>=len(rows): return rows
    rng=random.Random(seed); groups=defaultdict(list)
    for r in rows: groups[str(r["gt"])].append(r)
    for g in groups.values(): rng.shuffle(g)
    keys=sorted(groups); idx={k:0 for k in keys}; out=[]
    while len(out)<limit:
        moved=False
        for k in keys:
            if len(out)>=limit: break
            i=idx[k]
            if i<len(groups[k]): out.append(groups[k][i]); idx[k]+=1; moved=True
        if not moved: break
    rng.shuffle(out); return out


def load_ranked_heads(scan_file: Path, metric: str, n: int):
    if not scan_file.exists(): raise FileNotFoundError(scan_file)
    with scan_file.open("r", encoding="utf-8") as f: rows=list(csv.DictReader(f))
    usable=[]
    for r in rows:
        try: score=float(r[metric]); layer=int(r["layer"]); head=int(r["head"])
        except Exception: continue
        if math.isfinite(score): usable.append((score,layer,head,r))
    usable.sort(key=lambda x:(-x[0],x[1],x[2]))
    if len(usable)<n: raise RuntimeError(f"Need Top{n}; only {len(usable)} usable receiver rows")
    heads=[]; report=[]
    for rank,(score,l,h,r) in enumerate(usable[:n],1):
        heads.append((l,h)); report.append({
            "rank":rank,"layer":l,"head":h,"unit":hname((l,h)),"score":score,"rank_metric":metric,
            "kv_head":r.get("kv_head",""),
            "mean_projection_fraction_block":r.get("mean_projection_fraction_block",""),
            "mean_abs_projection_fraction_block":r.get("mean_abs_projection_fraction_block",""),
            "mean_delta_norm":r.get("mean_delta_norm",""),
            "mean_object_attention_mass":r.get("mean_object_attention_mass",""),
            "positive_projection_rate":r.get("positive_projection_rate","")})
    return heads, report


def validate_heads(heads, decoder_layers, attention_helper, receiver):
    for l,h in heads:
        attn=attention_helper.resolve_self_attention(decoder_layers[l]); shape=receiver.resolve_attention_shape(attn)
        if not 0<=h<int(shape.n_query_heads): raise ValueError(f"{hname((l,h))} invalid; n_query_heads={shape.n_query_heads}")


def matched_random(selected, decoder_layers, attention_helper, receiver, seed):
    rng=random.Random(seed); excluded=set(selected); used=set(); out=[]
    for l,_ in selected:
        attn=attention_helper.resolve_self_attention(decoder_layers[l]); shape=receiver.resolve_attention_shape(attn)
        cand=[(l,h) for h in range(int(shape.n_query_heads)) if (l,h) not in excluded and (l,h) not in used]
        if not cand: raise RuntimeError(f"No distinct same-layer random head available at L{l}")
        x=rng.choice(cand); used.add(x); out.append(x)
    return out


def channels_for(mode): return ["q","k","v"] if mode=="qkv" else [mode]


def build_units(query_heads, mode, decoder_layers, attention_helper, receiver):
    return receiver.causal_units_from_query_heads(query_heads=query_heads, decoder_layers=decoder_layers,
                                                   attention_helper=attention_helper, channels=channels_for(mode))


@torch.inference_mode()
def run_bundle_patch(units, pair, original_states, swapped_states, model, base, relation_token_map,
                     decoder_layers, attention_helper, receiver):
    patches=[]; names=[]
    try:
        for u in units:
            l=int(u["layer"]); ch=str(u["channel"])
            attn=attention_helper.resolve_self_attention(decoder_layers[l]); module=receiver.projection_module(attn,ch)
            target_side,mapping=receiver.patch_position_map(pair=pair, channel=ch, condition="corrupt_on_original",
                                                             original_states=original_states, swapped_states=swapped_states, layer=l)
            if target_side!="original": raise RuntimeError(f"Unexpected target side {target_side}")
            p=receiver.ProjectionHeadPatch(module=module, head=int(u["unit_head"]), head_dim=int(u["head_dim"]), target_to_source=mapping)
            patches.append(p); names.append(str(u["unit"]))
        result=receiver.run_scores(model=model, batch=pair.original_batch, base=base, relation_token_map=relation_token_map)
        missing=[names[i] for i,p in enumerate(patches) if not p.applied]
        if missing: raise RuntimeError(f"Projection patches did not fire: {missing}")
        result["patched_units"]=names; return result
    finally:
        for p in reversed(patches):
            with contextlib.suppress(Exception): p.close()


def margin(logits, gt): return float(logits[REL_TO_ID[gt]]-logits[REL_TO_ID[OPPOSITE[gt]]])


def eligible(mode, orig_ok, swap_ok):
    return (orig_ok and swap_ok) if mode=="both_correct" else (orig_ok if mode=="original_correct" else True)


def result_row(sid, gt, family, mode, k, qheads, units, orig, swap, patched, eps):
    src=OPPOSITE[gt]; mc=margin(orig["logits"],gt); ms=margin(swap["logits"],gt); mp=margin(patched["logits"],gt)
    den=mc-ms; prog=(mc-mp)/den if abs(den)>=eps else float("nan")
    po,ps,pp=str(orig["prediction"]),str(swap["prediction"]),str(patched["prediction"])
    nq=sum(str(u["channel"])=="q" for u in units); nk=sum(str(u["channel"])=="k" for u in units); nv=sum(str(u["channel"])=="v" for u in units)
    return {"sid":sid,"gt_target":gt,"gt_source":src,"family":family,"patch_mode":mode,"K_query_heads":k,
            "query_heads":",".join(hname(h) for h in qheads),"n_unique_units":len(units),"n_q_units":nq,"n_k_units":nk,"n_v_units":nv,
            "patched_units":",".join(str(u["unit"]) for u in units),"clean_prediction":po,"swapped_prediction":ps,"patched_prediction":pp,
            "clean_margin":mc,"source_margin":ms,"patched_margin":mp,"source_progress":prog,
            "source_follow":pp==src,"target_keep":pp==gt,"changed_from_clean":pp!=po,
            "crossed_decision_boundary":mc>0 and mp<=0,"source_side":abs(mp-ms)<abs(mp-mc),"other_prediction":pp not in (gt,src)}


def summarize(rows):
    g=defaultdict(list)
    for r in rows: g[(r["family"],r["patch_mode"],int(r["K_query_heads"]))].append(r)
    out=[]
    for (fam,mode,k),v in g.items():
        out.append({"family":fam,"patch_mode":mode,"K_query_heads":k,"N":len(v),"query_heads":v[0]["query_heads"],
                    "n_unique_units":int(v[0]["n_unique_units"]),"n_q_units":int(v[0]["n_q_units"]),"n_k_units":int(v[0]["n_k_units"]),"n_v_units":int(v[0]["n_v_units"]),
                    "source_follow_rate":safe_mean(float(x["source_follow"]) for x in v),"crossed_rate":safe_mean(float(x["crossed_decision_boundary"]) for x in v),
                    "changed_rate":safe_mean(float(x["changed_from_clean"]) for x in v),"source_side_rate":safe_mean(float(x["source_side"]) for x in v),
                    "target_keep_rate":safe_mean(float(x["target_keep"]) for x in v),"other_rate":safe_mean(float(x["other_prediction"]) for x in v),
                    "mean_source_progress":safe_mean(float(x["source_progress"]) for x in v),"std_source_progress":safe_std(float(x["source_progress"]) for x in v),
                    "mean_clean_margin":safe_mean(float(x["clean_margin"]) for x in v),"mean_source_margin":safe_mean(float(x["source_margin"]) for x in v),
                    "mean_patched_margin":safe_mean(float(x["patched_margin"]) for x in v)})
    fam_order={"selected":0,"random":1}; mode_order={m:i for i,m in enumerate(PATCH_MODES)}
    out.sort(key=lambda r:(int(r["K_query_heads"]),mode_order[r["patch_mode"]],fam_order[r["family"]])); return out


def print_summary(rows):
    print("\n"+"="*148); print("STAGE-III QUERY-SWAP MULTI-HEAD RECEIVER PATCH"); print("="*148)
    print(f"{'family':<9} {'mode':<4} {'K':>3} {'units':>5} {'Q/K/V':>11} {'N':>5} {'srcFollow':>10} {'crossed':>9} {'changed':>9} {'srcSide':>9} {'progress':>10}")
    print("-"*148)
    for r in rows:
        qkv=f"{r['n_q_units']}/{r['n_k_units']}/{r['n_v_units']}"
        print(f"{r['family']:<9} {r['patch_mode']:<4} {r['K_query_heads']:>3} {r['n_unique_units']:>5} {qkv:>11} {r['N']:>5} "
              f"{100*r['source_follow_rate']:>9.2f}% {100*r['crossed_rate']:>8.2f}% {100*r['changed_rate']:>8.2f}% "
              f"{100*r['source_side_rate']:>8.2f}% {r['mean_source_progress']:>+10.3f}")
    print("="*148)


def main():
    args=parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.sample_seed); np.random.seed(args.sample_seed); torch.manual_seed(args.sample_seed)
    outdir=Path(args.output_dir)
    if args.overwrite and outdir.exists(): shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()): raise RuntimeError(f"Output directory not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    raw_sizes=[int(x.strip()) for x in args.bundle_sizes.split(",") if x.strip() and x.strip().lower()!="all"]
    need=max(raw_sizes) if raw_sizes else 20
    if args.heads.strip():
        selected=parse_heads(args.heads); ranking=[{"rank":i+1,"layer":h[0],"head":h[1],"unit":hname(h),"score":"","rank_metric":"explicit"} for i,h in enumerate(selected)]
    else:
        selected,ranking=load_ranked_heads(Path(args.receiver_scan_file),args.rank_metric,need)
    sizes=parse_sizes(args.bundle_sizes,len(selected)); selected=selected[:max(sizes)]; ranking=ranking[:max(sizes)]; modes=parse_modes(args.patch_modes)

    ioi=import_file(Path(args.ioi_script),"_s3qs_ioi"); producer=import_file(Path(args.producer_script),"_s3qs_prod")
    receiver=import_file(Path(args.receiver_script),"_s3qs_recv"); v3=import_file(Path(args.v3_script),"_s3qs_v3")
    base=import_file(Path(args.base_script),"_s3qs_base"); attn_helper=import_file(Path(args.attention_helper),"_s3qs_attn")

    src_path=Path(args.source_output_dir)/"extraction.jsonl"
    allowed={"left","right"} if args.relation_scope=="lr" else set(RELATIONS)
    source_rows=[r for r in read_jsonl(src_path) if str(r.get("gt")) in allowed]
    source_rows=stratified_subset(source_rows,args.max_samples,args.sample_seed)
    if not source_rows: raise RuntimeError("No source rows")

    model=processor=None; baseline=[]; results=[]
    try:
        model,processor,spec,decoder_layers,decoder_path,relation_token_map=producer.load_model_bundle(args=args,base=base); model.eval()
        validate_heads(selected,decoder_layers,attn_helper,receiver)
        random_heads=matched_random(selected,decoder_layers,attn_helper,receiver,args.random_seed)
        validate_heads(random_heads,decoder_layers,attn_helper,receiver)

        keep=args.max_samples; args.max_samples=None
        try: records_by_sid,prompt_rows,audit=ioi.prepare_data_helpers(args,base)
        finally: args.max_samples=keep
        layers=sorted(set(l for l,_ in selected+random_heads))

        print("\n"+"="*148); print("STAGE-III QUERY-SWAP RECEIVER BUNDLE"); print("="*148)
        print("model           :",args.model); print("rows            :",len(source_rows)); print("filter          :",args.baseline_filter)
        print("bundle sizes    :",sizes); print("patch modes     :",modes); print("selected Top    :",", ".join(hname(h) for h in selected))
        print("matched random  :",", ".join(hname(h) for h in random_heads)); print("source          : same image + B relative to A")
        print("target          : same image + A relative to B"); print("KV alignment    : identity A->A, B->B"); print("="*148,flush=True)

        write_csv(outdir/"top_receiver_heads_used.csv",ranking)
        write_csv(outdir/"matched_random_heads.csv",[{"rank":i+1,"selected":hname(selected[i]),"random":hname(random_heads[i]),"layer":selected[i][0]} for i in range(len(selected))])
        bpath=outdir/"baseline_audit.jsonl"; spath=outdir/"sample_patch_results.jsonl"; epath=outdir/"errors.jsonl"; neligible=0

        for idx,row in enumerate(tqdm(source_rows,desc="stage3-queryswap-bundle"),1):
            pair=None
            try:
                pair=receiver.prepare_pair(args=args,row=row,records_by_sid=records_by_sid,prompt_rows=prompt_rows,base=base,v3=v3,processor=processor,device=torch.device(args.device))
                gt=str(pair.gt); src=OPPOSITE[gt]
                orig,swap,orig_states,swap_states=receiver.capture_pair_projections(pair=pair,layers=layers,model=model,base=base,relation_token_map=relation_token_map,
                                                                                     decoder_layers=decoder_layers,attention_helper=attn_helper)
                po,ps=str(orig["prediction"]),str(swap["prediction"]); ok_o=po==gt; ok_s=ps==src
                mc,ms=margin(orig["logits"],gt),margin(swap["logits"],gt); use=eligible(args.baseline_filter,ok_o,ok_s)
                br={"sid":int(pair.sid),"gt_target":gt,"gt_source":src,"original_prediction":po,"swapped_prediction":ps,"original_correct":ok_o,
                    "swapped_correct":ok_s,"both_correct":ok_o and ok_s,"clean_margin":mc,"source_margin":ms,"eligible":use}
                baseline.append(br); append_jsonl(bpath,br)
                if not use or abs(mc-ms)<args.min_margin_denominator: continue
                neligible+=1
                for k in sizes:
                    for fam,qheads in (("selected",selected[:k]),("random",random_heads[:k])):
                        for mode in modes:
                            units=build_units(qheads,mode,decoder_layers,attn_helper,receiver)
                            patched=run_bundle_patch(units,pair,orig_states,swap_states,model,base,relation_token_map,decoder_layers,attn_helper,receiver)
                            rr=result_row(int(pair.sid),gt,fam,mode,k,qheads,units,orig,swap,patched,args.min_margin_denominator)
                            results.append(rr); append_jsonl(spath,rr)
                if args.print_every>0 and neligible%args.print_every==0:
                    k=sizes[-1]
                    parts=[]
                    for mode in modes:
                        x=next(r for r in reversed(results) if r["sid"]==int(pair.sid) and r["family"]=="selected" and r["patch_mode"]==mode and r["K_query_heads"]==k)
                        parts.append(f"{mode}={x['patched_prediction']}({x['source_progress']:+.2f})")
                    tqdm.write(f"sid={pair.sid} {gt}->{src} Top{k} | "+" ".join(parts))
            except Exception as exc:
                append_jsonl(epath,{"sid":int(row.get("sid",-1)),"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
                if args.fail_fast: raise
            finally:
                if pair is not None: receiver.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache_every>0 and idx%args.empty_cache_every==0: torch.cuda.empty_cache()

        if not results: raise RuntimeError("No patch results generated")
        summary=summarize(results); write_csv(outdir/"baseline_audit.csv",baseline); write_csv(outdir/"sample_patch_results.csv",results); write_csv(outdir/"summary.csv",summary); print_summary(summary)
        write_json(outdir/"config.json",{"version":VERSION,"model":args.model,"repo_id":getattr(spec,"repo_id",""),"decoder_path":decoder_path,
                   "receiver_scan_file":args.receiver_scan_file,"rank_metric":args.rank_metric if not args.heads.strip() else "explicit",
                   "selected_query_heads":[hname(h) for h in selected],"matched_random_query_heads":[hname(h) for h in random_heads],"bundle_sizes":sizes,"patch_modes":modes,
                   "counterfactual":"same image; target A relative B; source B relative A","direction":"corrupt_on_original",
                   "q_patch":"swapped prompt-last Q -> original prompt-last","kv_patch":"swapped identity-aligned object K/V -> original; GQA deduplicated",
                   "baseline_filter":args.baseline_filter,"N_source_rows":len(source_rows),"N_intervention_eligible":neligible,"audit":audit})
        report=[f"version: {VERSION}",f"model: {args.model}","","COUNTERFACTUAL","target: A relative to B, same image","source: B relative to A, same image",
                "patch: old receiver projection-channel corrupt_on_original, now cumulative multi-head","","TOP RECEIVER QUERY HEADS"]
        for r in ranking: report.append(f"rank={int(r['rank']):02d} {r['unit']} score={r['score']}")
        report += ["","SUMMARY"]
        for r in summary:
            report.append(f"{r['family']:<8} {r['patch_mode']:<4} Top{r['K_query_heads']:02d} units={r['n_unique_units']:02d} Q/K/V={r['n_q_units']}/{r['n_k_units']}/{r['n_v_units']} "
                          f"N={r['N']:03d} source_follow={100*r['source_follow_rate']:6.2f}% crossed={100*r['crossed_rate']:6.2f}% progress={r['mean_source_progress']:+.4f}")
        report += ["","Interpretation:","q/k/v reproduce the old single-channel receiver intervention; qkv patches all selected receiver channels together.",
                   "Look for Top5 -> Top10 -> Top15 -> Top20 growth in source-follow/progress versus matched random.",
                   "K/V unit counts can be smaller than K because Qwen GQA shares KV heads."]
        (outdir/"report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
        print("\nSaved:")
        for n in ("config.json","top_receiver_heads_used.csv","matched_random_heads.csv","baseline_audit.csv","sample_patch_results.csv","summary.csv","report.txt"): print(" ",outdir/n)
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__": main()
