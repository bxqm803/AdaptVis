#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-visual Stage-III receiver-bundle replacement.

Target: same prompt + original image.
Source: same prompt + horizontally flipped image.

For each selected receiver head h:
    c_h = W_O^h sum_{s in object tokens} A_h[last,s] V_h[s]

On a fresh original-image run, at prompt-last:
    attn_out' = attn_out + sum_h (c_h^flip - c_h^orig)

Thus only the selected bundle's natural object->prompt-last post-W_O write is
changed. No DAS, no probe direction, no query swap, no whole-head replacement.

Default old Stage-III order:
    L26H4, L26H2, L26H6, L26H0, L24H4

Default cumulative bundles:
    K=1,2,4,5

A distinct same-layer random control head is paired with each selected head, so
all cumulative prefixes are exactly layer-matched.

Primary metrics on LEFT/RIGHT examples:
    source_follow_rate : patched prediction == horizontal-flip/opposite answer
    crossed_rate       : GT-vs-opposite margin crosses <= 0
    source_progress    : (Mclean-Mpatch)/(Mclean-Mflip), 0=clean, 1=flip source

Smoke:
CUDA_VISIBLE_DEVICES=0 python -u validate_stage3_horizontal_flip_receiver_bundle_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --heads 26:4,26:2,26:6,26:0,24:4 \
  --bundle-sizes 1,2,4,5 \
  --baseline-filter both_correct \
  --max-samples 32 \
  --device cuda:0 \
  --output-dir output/qwen3b_stage3_flip_receiver_bundle_smoke \
  --overwrite

Full:
CUDA_VISIBLE_DEVICES=0 python -u validate_stage3_horizontal_flip_receiver_bundle_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --heads 26:4,26:2,26:6,26:0,24:4 \
  --bundle-sizes 1,2,4,5 \
  --baseline-filter both_correct \
  --max-samples 0 \
  --device cuda:0 \
  --output-dir output/qwen3b_stage3_flip_receiver_bundle_v1 \
  --overwrite
"""
from __future__ import annotations

import argparse, contextlib, csv, gc, importlib.util, json, math, random, shutil, sys, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

VERSION = "stage3-horizontal-flip-receiver-bundle-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r:i for i,r in enumerate(RELATIONS)}
OPPOSITE = {"left":"right", "right":"left", "above":"below", "below":"above"}
DEFAULT_HEADS = "26:4,26:2,26:6,26:0,24:4"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--source-output-dir", default="output/spatial_storage_transport_utilization/coco/qwen-3b")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", default="mean", choices=("last","mean"))
    p.add_argument("--heads", default=DEFAULT_HEADS)
    p.add_argument("--bundle-sizes", default="1,2,4,5")
    p.add_argument("--baseline-filter", default="both_correct", choices=("both_correct","clean_correct","all"))
    p.add_argument("--max-samples", type=int, default=0, help="0 = all LEFT/RIGHT rows")
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


def read_jsonl(path):
    out=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(dict(row), ensure_ascii=False)+"\n")


def write_csv(path, rows):
    if not rows: Path(path).write_text("", encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def safe_mean(xs):
    a=np.asarray(list(xs), dtype=np.float64); a=a[np.isfinite(a)]; return float(a.mean()) if a.size else float("nan")


def safe_std(xs):
    a=np.asarray(list(xs), dtype=np.float64); a=a[np.isfinite(a)]; return float(a.std()) if a.size else float("nan")


def parse_heads(text):
    out=[]
    for piece in text.split(","):
        piece=piece.strip().upper().replace("L","").replace("H",":")
        if not piece: continue
        a,b=piece.split(":",1); h=(int(a),int(b))
        if h not in out: out.append(h)
    if not out: raise ValueError("No heads")
    return out


def hname(h): return f"L{h[0]}H{h[1]}"


def parse_ks(text, n):
    ks=set()
    for x in text.split(","):
        x=x.strip().lower()
        if not x: continue
        ks.add(n if x=="all" else min(int(x),n))
    ks.add(n)
    if min(ks)<1: raise ValueError("bundle size must be >=1")
    return sorted(ks)


def stratified(rows, limit, seed):
    rows=[dict(r) for r in rows]
    if limit<=0 or limit>=len(rows): return rows
    rng=random.Random(seed); groups=defaultdict(list)
    for r in rows: groups[str(r["gt"])].append(r)
    for g in groups.values(): rng.shuffle(g)
    out=[]; idx={k:0 for k in groups}
    while len(out)<limit:
        moved=False
        for k in sorted(groups):
            if len(out)>=limit: break
            if idx[k]<len(groups[k]): out.append(groups[k][idx[k]]); idx[k]+=1; moved=True
        if not moved: break
    rng.shuffle(out); return out


def horizontal_flip(image):
    try: return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except AttributeError: return image.transpose(Image.FLIP_LEFT_RIGHT)


def make_flip_batch(base, processor, original_batch, flipped_image, question_text, device):
    b=base.make_question_batch(processor=processor, image=flipped_image, question_text=question_text, device=device)
    for key in ("input_ids","attention_mask"):
        if key in original_batch and key in b and torch.is_tensor(original_batch[key]) and not torch.equal(original_batch[key], b[key]):
            raise RuntimeError(f"{key} changed under same-prompt horizontal flip")
    if "image_grid_thw" in original_batch and "image_grid_thw" in b:
        if not torch.equal(original_batch["image_grid_thw"].to(b["image_grid_thw"].device), b["image_grid_thw"]):
            raise RuntimeError("image_grid_thw changed under horizontal flip")
    return b


def first_3d(output):
    if torch.is_tensor(output):
        if output.ndim!=3: raise RuntimeError(f"Expected 3D, got {tuple(output.shape)}")
        return output
    if isinstance(output,(tuple,list)):
        for x in output:
            if torch.is_tensor(x) and x.ndim==3: return x
    raise RuntimeError("No 3D attention tensor")


def replace_first_3d(output, x):
    if torch.is_tensor(output): return x
    vals=list(output)
    for i,v in enumerate(vals):
        if torch.is_tensor(v) and v.ndim==3:
            vals[i]=x; return tuple(vals) if isinstance(output,tuple) else vals
    raise RuntimeError("No 3D tensor to replace")


class AddPromptLastDelta:
    def __init__(self, decoder_layers, attention_helper, prompt_last, deltas):
        self.layers=decoder_layers; self.helper=attention_helper; self.pos=int(prompt_last)
        self.deltas={int(k):np.asarray(v,dtype=np.float32) for k,v in deltas.items()}; self.handles=[]; self.events=defaultdict(int)
    def __enter__(self):
        for L,vec in self.deltas.items():
            attn=self.helper.resolve_self_attention(self.layers[L])
            def mk(layer, arr):
                def hook(_m,_i,out):
                    h=first_3d(out)
                    if not 0<=self.pos<h.shape[1]: raise RuntimeError(f"L{layer} prompt-last outside sequence")
                    if arr.shape[0]!=h.shape[-1]: raise RuntimeError(f"L{layer} delta dim mismatch")
                    z=h.clone(); z[0,self.pos]+=torch.as_tensor(arr,device=h.device,dtype=h.dtype); self.events[layer]+=1
                    return replace_first_3d(out,z)
                return hook
            self.handles.append(attn.register_forward_hook(mk(L,vec)))
        return self
    def validate(self):
        miss=[L for L in self.deltas if self.events[L]<1]
        if miss: raise RuntimeError(f"Patch did not fire at {miss}")
    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception): h.remove()
        self.handles=[]
    def __exit__(self,*_): self.close()


@torch.inference_mode()
def capture_writes(model,batch,decoder_layers,attention_helper,receiver,v3,relation_token_map,heads,object_positions,prompt_last):
    layers=sorted(set(L for L,_ in heads))
    _,traces=v3.trace_prompt_chunks(attention_helper=attention_helper,model=model,batch=batch,relation_token_map=relation_token_map,
        decoder_layers=decoder_layers,layers=layers,target_positions=sorted(set(map(int,object_positions))|{int(prompt_last)}),chunk_size=max(1,len(layers)))
    out={}
    for L,H in heads:
        w,_=receiver.object_head_write(trace=traces[L],head=H,target_position=prompt_last,source_positions=object_positions)
        out[(L,H)]=w.detach().float().cpu().numpy().astype(np.float32)
    return out


def layer_deltas(target_w, source_w, heads):
    out={}
    for h in heads:
        L=h[0]; d=np.asarray(source_w[h],np.float32)-np.asarray(target_w[h],np.float32)
        if L not in out: out[L]=np.zeros_like(d)
        out[L]+=d
    return out


def validate_heads(heads, layers, helper, receiver):
    for L,H in heads:
        if not 0<=L<len(layers): raise ValueError(f"{hname((L,H))}: invalid layer")
        shape=receiver.resolve_attention_shape(helper.resolve_self_attention(layers[L]))
        if not 0<=H<int(shape.n_query_heads): raise ValueError(f"{hname((L,H))}: invalid query head")


def random_order(target, layers, helper, receiver, seed):
    rng=random.Random(seed); excluded=set(target); used=set(); out=[]
    for L,_ in target:
        n=int(receiver.resolve_attention_shape(helper.resolve_self_attention(layers[L])).n_query_heads)
        cand=[(L,h) for h in range(n) if (L,h) not in excluded and (L,h) not in used]
        if not cand: raise RuntimeError(f"No random control available at L{L}")
        x=rng.choice(cand); out.append(x); used.add(x)
    return out


def margin(logits,gt): return float(logits[REL_TO_ID[gt]]-logits[REL_TO_ID[OPPOSITE[gt]]])


def eligible(mode,clean_ok,flip_ok):
    return (clean_ok and flip_ok) if mode=="both_correct" else clean_ok if mode=="clean_correct" else True


@torch.inference_mode()
def patched_scores(model,batch,layers,helper,receiver,base,relation_token_map,prompt_last,deltas):
    p=AddPromptLastDelta(layers,helper,prompt_last,deltas)
    try:
        with p:
            r=receiver.run_scores(model=model,batch=batch,base=base,relation_token_map=relation_token_map)
        p.validate(); return r
    finally: p.close()


def metric_row(sid,gt,family,k,heads,clean,flip,patched,min_den):
    src=OPPOSITE[gt]; mc=margin(clean["logits"],gt); mf=margin(flip["logits"],gt); mp=margin(patched["logits"],gt); den=mc-mf
    prog=(mc-mp)/den if abs(den)>=min_den else float("nan"); pp=str(patched["prediction"]); cp=str(clean["prediction"])
    return {"sid":sid,"gt_target":gt,"gt_source":src,"family":family,"bundle_size":k,"heads":",".join(hname(h) for h in heads),
        "clean_prediction":cp,"natural_flip_prediction":str(flip["prediction"]),"patched_prediction":pp,
        "clean_margin":mc,"natural_flip_margin":mf,"patched_margin":mp,"source_progress":prog,
        "source_follow":pp==src,"target_keep":pp==gt,"changed_from_clean":pp!=cp,
        "crossed_decision_boundary":mc>0 and mp<=0,"source_side":abs(mp-mf)<abs(mp-mc),"other_prediction":pp not in (gt,src)}


def summarize(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r["family"],r["bundle_size"])].append(r)
    out=[]
    for (fam,k),vals in sorted(groups.items(), key=lambda x:(x[0][1],x[0][0])):
        out.append({"family":fam,"bundle_size":k,"heads":vals[0]["heads"],"N":len(vals),
            "source_follow_rate":safe_mean(float(v["source_follow"]) for v in vals),
            "crossed_rate":safe_mean(float(v["crossed_decision_boundary"]) for v in vals),
            "changed_rate":safe_mean(float(v["changed_from_clean"]) for v in vals),
            "target_keep_rate":safe_mean(float(v["target_keep"]) for v in vals),
            "source_side_rate":safe_mean(float(v["source_side"]) for v in vals),
            "other_rate":safe_mean(float(v["other_prediction"]) for v in vals),
            "mean_source_progress":safe_mean(v["source_progress"] for v in vals),"std_source_progress":safe_std(v["source_progress"] for v in vals),
            "mean_clean_margin":safe_mean(v["clean_margin"] for v in vals),"mean_natural_flip_margin":safe_mean(v["natural_flip_margin"] for v in vals),
            "mean_patched_margin":safe_mean(v["patched_margin"] for v in vals)})
    return out


def print_summary(rows):
    print("\n"+"="*118); print("STAGE-III HORIZONTAL-FLIP RECEIVER-BUNDLE REPLACEMENT"); print("="*118)
    print(f"{'family':<10} {'K':>3} {'N':>5} {'srcFollow':>10} {'crossed':>9} {'changed':>9} {'srcSide':>9} {'progress':>10} {'other':>8}")
    print("-"*118)
    for r in rows:
        print(f"{r['family']:<10} {r['bundle_size']:>3d} {r['N']:>5d} {100*r['source_follow_rate']:>9.2f}% {100*r['crossed_rate']:>8.2f}% "
              f"{100*r['changed_rate']:>8.2f}% {100*r['source_side_rate']:>8.2f}% {r['mean_source_progress']:>+10.3f} {100*r['other_rate']:>7.2f}%")
    print("="*118)


def main():
    args=parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.sample_seed); np.random.seed(args.sample_seed); torch.manual_seed(args.sample_seed)
    out=Path(args.output_dir)
    if args.overwrite and out.exists(): shutil.rmtree(out)
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"Output directory not empty: {out}")
    out.mkdir(parents=True,exist_ok=True)

    ioi=import_file(Path(args.ioi_script),"_s3f_ioi"); producer=import_file(Path(args.producer_script),"_s3f_prod")
    receiver=import_file(Path(args.receiver_script),"_s3f_recv"); v3=import_file(Path(args.v3_script),"_s3f_v3")
    base=import_file(Path(args.base_script),"_s3f_base"); helper=import_file(Path(args.attention_helper),"_s3f_helper")

    src=Path(args.source_output_dir); extraction=src/"extraction.jsonl"
    if not extraction.exists(): raise FileNotFoundError(extraction)
    rows=[r for r in read_jsonl(extraction) if str(r.get("gt")) in ("left","right")]
    rows=stratified(rows,args.max_samples,args.sample_seed)
    named=parse_heads(args.heads); ks=parse_ks(args.bundle_sizes,len(named)); device=torch.device(args.device)
    model=processor=None; baseline_rows=[]; result_rows=[]
    try:
        model,processor,spec,layers,decoder_path,relation_token_map=producer.load_model_bundle(args=args,base=base); model.eval()
        validate_heads(named,layers,helper,receiver); rnd=random_order(named,layers,helper,receiver,args.random_seed); validate_heads(rnd,layers,helper,receiver)
        capture_heads=list(dict.fromkeys(named+rnd))
        analysis_max=args.max_samples; args.max_samples=None
        try: records_by_sid,prompt_rows,audit=ioi.prepare_data_helpers(args,base)
        finally: args.max_samples=analysis_max

        print("\n"+"="*118); print("STAGE-III PURE-VISUAL RECEIVER-BUNDLE CAUSAL REPLACEMENT"); print("="*118)
        print("model           :",args.model); print("rows            :",len(rows)); print("filter          :",args.baseline_filter)
        print("selected order  :",", ".join(hname(h) for h in named)); print("matched random  :",", ".join(hname(h) for h in rnd)); print("bundle sizes    :",ks)
        print("source          : same prompt + horizontal-flipped image"); print("intervention    : object->prompt-last post-WO write replacement"); print("="*118,flush=True)

        eligible_n=0
        for i,row in enumerate(tqdm(rows,desc="stage3-flip-bundle"),1):
            pair=flip_img=None
            try:
                sid=int(row["sid"]); pair=receiver.prepare_pair(args=args,row=row,records_by_sid=records_by_sid,prompt_rows=prompt_rows,base=base,v3=v3,processor=processor,device=device)
                gt=str(pair.gt); src_gt=OPPOSITE[gt]; flip_img=horizontal_flip(pair.image)
                flip_batch=make_flip_batch(base,processor,pair.original_batch,flip_img,str(prompt_rows[sid]["question_text"]),device)
                clean=receiver.run_scores(model=model,batch=pair.original_batch,base=base,relation_token_map=relation_token_map)
                flip=receiver.run_scores(model=model,batch=flip_batch,base=base,relation_token_map=relation_token_map)
                cc=str(clean["prediction"])==gt; fc=str(flip["prediction"])==src_gt; mc=margin(clean["logits"],gt); mf=margin(flip["logits"],gt)
                ok=eligible(args.baseline_filter,cc,fc)
                br={"sid":sid,"gt_target":gt,"gt_source":src_gt,"clean_prediction":clean["prediction"],"natural_flip_prediction":flip["prediction"],
                    "clean_correct":cc,"natural_flip_correct":fc,"both_correct":cc and fc,"clean_margin":mc,"natural_flip_margin":mf,"eligible":ok}
                baseline_rows.append(br); append_jsonl(out/"baseline_audit.jsonl",br)
                if not ok or abs(mc-mf)<args.min_margin_denominator: continue
                eligible_n+=1
                tw=capture_writes(model,pair.original_batch,layers,helper,receiver,v3,relation_token_map,capture_heads,pair.original_object_positions,pair.original_prompt_last)
                sw=capture_writes(model,flip_batch,layers,helper,receiver,v3,relation_token_map,capture_heads,pair.original_object_positions,pair.original_prompt_last)
                for k in ks:
                    for fam,bundle in (("selected",named[:k]),("random",rnd[:k])):
                        patched=patched_scores(model,pair.original_batch,layers,helper,receiver,base,relation_token_map,pair.original_prompt_last,layer_deltas(tw,sw,bundle))
                        mr=metric_row(sid,gt,fam,k,bundle,clean,flip,patched,args.min_margin_denominator); result_rows.append(mr); append_jsonl(out/"sample_bundle_results.jsonl",mr)
                if args.print_every>0 and eligible_n%args.print_every==0:
                    k=ks[-1]; a=next(r for r in reversed(result_rows) if r["sid"]==sid and r["family"]=="selected" and r["bundle_size"]==k)
                    b=next(r for r in reversed(result_rows) if r["sid"]==sid and r["family"]=="random" and r["bundle_size"]==k)
                    tqdm.write(f"sid={sid} {gt}->{src_gt} | selectedK{k}={a['patched_prediction']} prog={a['source_progress']:+.3f} | randomK{k}={b['patched_prediction']} prog={b['source_progress']:+.3f}")
            except Exception as e:
                append_jsonl(out/"errors.jsonl",{"sid":int(row.get("sid",-1)),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
                if args.fail_fast: raise
            finally:
                if flip_img is not None:
                    with contextlib.suppress(Exception): flip_img.close()
                if pair is not None: receiver.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache_every>0 and i%args.empty_cache_every==0: torch.cuda.empty_cache()

        if not result_rows: raise RuntimeError("No intervention results; inspect filter/errors")
        summary=summarize(result_rows); write_csv(out/"baseline_audit.csv",baseline_rows); write_csv(out/"sample_bundle_results.csv",result_rows); write_csv(out/"bundle_summary.csv",summary); print_summary(summary)
        clean_acc=safe_mean(float(r["clean_correct"]) for r in baseline_rows); flip_acc=safe_mean(float(r["natural_flip_correct"]) for r in baseline_rows); both=sum(int(r["both_correct"]) for r in baseline_rows)
        config={"version":VERSION,"model":args.model,"repo_id":getattr(spec,"repo_id",""),"decoder_path":decoder_path,"source_output_dir":str(src),
            "selected_heads_order":[hname(h) for h in named],"matched_random_order":[hname(h) for h in rnd],"bundle_sizes":ks,"baseline_filter":args.baseline_filter,
            "source":"same prompt + horizontal-flipped image; LEFT/RIGHT only","intervention":"add sum_h(flip object->last post-WO write - original object->last post-WO write) at prompt-last",
            "N_baseline":len(baseline_rows),"N_both_correct":both,"N_intervention_eligible":eligible_n,"clean_next_token_acc":clean_acc,"natural_flip_aligned_next_token_acc":flip_acc,"audit":audit}
        (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
        lines=[f"version: {VERSION}",f"model: {args.model}",f"N baseline: {len(baseline_rows)}",f"N both-correct: {both}",f"N eligible: {eligible_n}",
            f"clean ACC: {100*clean_acc:.2f}%",f"flip aligned ACC: {100*flip_acc:.2f}%","","selected: "+", ".join(hname(h) for h in named),"random: "+", ".join(hname(h) for h in rnd),"","SUMMARY"]
        for r in summary: lines.append(f"{r['family']:<9} K={r['bundle_size']:>2} N={r['N']:>4} source_follow={100*r['source_follow_rate']:6.2f}% crossed={100*r['crossed_rate']:6.2f}% changed={100*r['changed_rate']:6.2f}% progress={r['mean_source_progress']:+.4f}")
        lines += ["","LIMITATION","This transfers the complete natural A·V·W_O object->last write. It does not yet isolate V-only content from routing changes."]
        (out/"report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
        print("\nSaved:"); [print(" ",out/n) for n in ("config.json","baseline_audit.csv","bundle_summary.csv","sample_bundle_results.csv","report.txt")]
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=="__main__": main()
