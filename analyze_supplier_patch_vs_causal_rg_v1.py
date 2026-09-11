#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prefill-only diagnostic: compare the causal-state movement induced by supplier-head
message amplification with the causal state's own Real-Gray direction.

MOVE = h_patched[C,p] - h_real[C,p]
RG   = h_real[C,p]    - h_gray[C,p]

Reports cosine(MOVE,RG), projection on normalized RG, and norm ratio.
No generation is performed.

Requires eval_supplier_head_groups_to_causal_text_v1.py in the repo/workdir.
"""
from __future__ import annotations

import argparse, contextlib, gc, json, math, random, shutil, traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn
import eval_supplier_head_groups_to_causal_text_v1 as exp

EPS = 1e-12


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen-3b', choices=['qwen-3b','qwen-7b'])
    p.add_argument('--data-root', default='data')
    p.add_argument('--prompt-jsonl', default='prompts/COCO_QA_two_obj_with_answer_four_options.jsonl')
    p.add_argument('--supplier-summary', required=True)
    p.add_argument('--ranked-causal', required=True)
    p.add_argument('--causal-layers', default='20-26')
    p.add_argument('--causal-top-k', type=int, default=7)
    p.add_argument('--causal-categories', default='')
    p.add_argument('--ks', default='10,20')
    p.add_argument('--groups', default='top10_all,top10_direction,top10_nondirection,top20_all,top20_direction,top20_nondirection')
    p.add_argument('--scales', default='2')
    p.add_argument('--gray-value', type=int, default=128)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--eval-max-samples', type=int, default=80, help='0 = full set')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--attn-impl', default='eager', choices=['eager','sdpa','flash_attention_2','none'])
    p.add_argument('--output-dir', required=True)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def strings(s): return [x.strip() for x in str(s).split(',') if x.strip()]

def safe_mean(xs: Iterable[Any]):
    v=[]
    for x in xs:
        try: z=float(x)
        except Exception: continue
        if math.isfinite(z): v.append(z)
    return float(np.mean(v)) if v else float('nan')

def safe_median(xs: Iterable[Any]):
    v=[]
    for x in xs:
        try: z=float(x)
        except Exception: continue
        if math.isfinite(z): v.append(z)
    return float(np.median(v)) if v else float('nan')

def cos(a,b):
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32)
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    return float(np.dot(a,b)/(na*nb)) if na>EPS and nb>EPS else float('nan')

def append_jsonl(path,row:Mapping[str,Any]):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(row),ensure_ascii=False)+'\n')


class Capture:
    def __init__(self, decoder_layers, state_layers, head_layers=()):
        self.states={}; self.pre_o={}; self.handles=[]
        for L in sorted(set(map(int,state_layers))):
            def make_state(L):
                def hook(_m,_inp,out):
                    x=traj.first_tensor(out)
                    self.states[L]=x.detach().float().cpu().numpy().astype(np.float32)
                return hook
            self.handles.append(decoder_layers[L].register_forward_hook(make_state(L)))
        for L in sorted(set(map(int,head_layers))):
            attn=exp.spatialscan.resolve_attn(decoder_layers[L])
            op=exp.spatialscan.resolve_o_proj(attn)
            def make_pre(L):
                def hook(_m,inp):
                    self.pre_o[L]=inp[0].detach().float().cpu().numpy().astype(np.float32)
                return hook
            self.handles.append(op.register_forward_pre_hook(make_pre(L)))
    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception): h.remove()
        self.handles=[]


@torch.inference_mode()
def capture(model,decoder_layers,batch,state_layers,head_layers=()):
    c=Capture(decoder_layers,state_layers,head_layers)
    try:
        kw=dict(batch); kw['use_cache']=False
        _=model(**kw)
        ms=[L for L in state_layers if L not in c.states]
        mh=[L for L in head_layers if L not in c.pre_o]
        if ms or mh: raise RuntimeError(f'missing states={ms} pre_o={mh}')
        return c.states,c.pre_o
    finally: c.close()


@torch.inference_mode()
def capture_patched(model,decoder_layers,batch,state_layers,patch_map,scale):
    c=Capture(decoder_layers,state_layers)
    prompt_len=int(batch['input_ids'].shape[1])
    try:
        with exp.MultiLayerPositionDelta(
            decoder_layers=decoder_layers, patch_map=patch_map,
            prompt_len=prompt_len, scale=float(scale)
        ) as patcher:
            kw=dict(batch); kw['use_cache']=False
            _=model(**kw)
        patcher.validate()
        ms=[L for L in state_layers if L not in c.states]
        if ms: raise RuntimeError(f'missing patched states={ms}')
        return c.states
    finally: c.close()


def summarize(df):
    rows=[]
    for (group,scale),g in df.groupby(['group','scale']):
        rows.append({
            'group':group, 'scale':float(scale), 'N_states':len(g), 'N_samples':g.sid.nunique(),
            'mean_cos_move_vs_RG':safe_mean(g.cos_move_vs_RG),
            'median_cos_move_vs_RG':safe_median(g.cos_move_vs_RG),
            'mean_projection_on_RG':safe_mean(g.projection_on_RG),
            'median_projection_on_RG':safe_median(g.projection_on_RG),
            'mean_move_norm':safe_mean(g.move_norm),
            'median_move_norm':safe_median(g.move_norm),
            'mean_RG_norm':safe_mean(g.RG_norm),
            'mean_move_over_RG_norm':safe_mean(g.move_over_RG_norm),
            'median_move_over_RG_norm':safe_median(g.move_over_RG_norm),
            'positive_cos_fraction':float((g.cos_move_vs_RG>0).mean()),
            'cos_gt_0p25_fraction':float((g.cos_move_vs_RG>0.25).mean()),
            'cos_gt_0p5_fraction':float((g.cos_move_vs_RG>0.5).mean()),
        })
    return pd.DataFrame(rows).sort_values(['scale','group']).reset_index(drop=True)


def main():
    a=args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    causal_layers=exp.parse_layers(a.causal_layers)
    categories=exp.parse_categories(a.causal_categories)
    ks=exp.parse_ints(a.ks); scales=exp.parse_floats(a.scales)
    wanted=set(strings(a.groups))
    groups_all,group_df=exp.load_supplier_groups(Path(a.supplier_summary),ks)
    unknown=wanted-set(groups_all)
    if unknown: raise ValueError(f'unknown groups: {sorted(unknown)}')
    groups={k:v for k,v in groups_all.items() if k in wanted}

    out=Path(a.output_dir)
    if a.overwrite and out.exists(): shutil.rmtree(out)
    if out.exists() and any(out.iterdir()): raise RuntimeError(f'non-empty {out}')
    out.mkdir(parents=True,exist_ok=True)
    err=out/'errors.jsonl'
    group_df[group_df.group.isin(groups)].to_csv(out/'group_heads.csv',index=False)

    two,meta,rec_by_sid=exp.load_coco_meta(a)
    meta_by_sid={int(x['sid']):x for x in meta}
    ranking_sids=set(pd.to_numeric(pd.read_csv(a.ranked_causal,usecols=['sid']).sid,errors='coerce').dropna().astype(int))
    ev=[x for x in meta if int(x['sid']) in ranking_sids]
    if a.eval_max_samples>0: ev=traj.stratified_cap(ev,a.eval_max_samples,a.seed+1)
    evs={int(x['sid']) for x in ev}
    sel=exp.load_causal_selection(Path(a.ranked_causal),causal_layers,a.causal_top_k,categories,evs)
    sel['gt']=sel.sid.map(lambda s:meta_by_sid[int(s)]['gt'])
    sel.to_csv(out/'selected_causal_text_states.csv',index=False)
    by_sid={int(s):g.copy() for s,g in sel.groupby('sid')}

    model=processor=None
    try:
        model,processor,decoder_layers,decoder_path,spec=exp.load_model(a,two)
        device=torch.device(a.device)
        all_heads=sorted(set(x for hs in groups.values() for x in hs))
        head_layers=sorted(set(L for L,_ in all_heads))
        geom=exp.infer_head_geometry(model,decoder_layers,head_layers)
        print('='*150)
        print('PATCH MOVE vs CAUSAL OWN REAL-GRAY (PREFILL ONLY)')
        print(f'N={len(ev)} groups={list(groups)} scales={scales}')
        print('='*150)
        rows=[]
        for m in tqdm(ev,desc='similarity'):
            sid=int(m['sid'])
            if sid not in by_sid: continue
            real=gray=rb=gb=None
            try:
                cr=by_sid[sid]
                state_layers=sorted(set(cr.source_layer.astype(int)))
                maxC=max(state_layers)
                hl=[L for L in head_layers if L<=maxC]
                real=base.record_image(rec_by_sid[sid])
                if hasattr(real,'convert'): real=real.convert('RGB')
                gray=dyn.make_gray_image(real,a.gray_value)
                rb=base.make_question_batch(processor=processor,image=real,question_text=m['question_text'],device=device)
                gb=base.make_question_batch(processor=processor,image=gray,question_text=m['question_text'],device=device)
                if rb['input_ids'][0].detach().cpu().tolist()!=gb['input_ids'][0].detach().cpu().tolist():
                    raise RuntimeError('Real/Gray tokenization mismatch')
                hr,pre_r=capture(model,decoder_layers,rb,state_layers,hl)
                hg,pre_g=capture(model,decoder_layers,gb,state_layers,hl)
                for gname,heads in groups.items():
                    hs=[(L,h) for L,h in heads if L in hl]
                    if not hs: continue
                    patch_map,_=exp.build_group_patch_map(
                        decoder_layers=decoder_layers,real_pre_o=pre_r,gray_pre_o=pre_g,
                        head_geometry=geom,heads=hs,causal_rows=cr)
                    if not patch_map: continue
                    for alpha in scales:
                        hp=capture_patched(model,decoder_layers,rb,state_layers,patch_map,alpha)
                        for r in cr.itertuples():
                            C=int(r.source_layer); p=int(r.position)
                            n=min(hr[C].shape[1],hg[C].shape[1],hp[C].shape[1])
                            if not (0<=p<n): continue
                            R=hr[C][0,p].astype(np.float32); G=hg[C][0,p].astype(np.float32); P=hp[C][0,p].astype(np.float32)
                            rg=R-G; mv=P-R
                            rn=float(np.linalg.norm(rg)); mn=float(np.linalg.norm(mv))
                            rows.append({
                                'sid':sid,'gt':m['gt'],'group':gname,'scale':float(alpha),
                                'causal_text_rank':int(r.causal_text_rank),'global_rank':int(r.rank),
                                'causal_layer':C,'position':p,'token':str(r.token),
                                'broad_category':str(r.broad_category),'ranking_mediation':float(r.mediation),
                                'RG_norm':rn,'move_norm':mn,'move_over_RG_norm':mn/rn if rn>EPS else np.nan,
                                'cos_move_vs_RG':cos(mv,rg),
                                'projection_on_RG':float(np.dot(mv,rg/rn)) if rn>EPS else np.nan,
                            })
            except Exception as e:
                append_jsonl(err,{'sid':sid,'error':f'{type(e).__name__}: {e}','traceback_tail':traceback.format_exc().splitlines()[-15:]})
                tqdm.write(f'[ERROR] sid={sid}: {type(e).__name__}: {e}')
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        df=pd.DataFrame(rows); df.to_csv(out/'causal_state_similarity_per_state.csv',index=False)
        sm=summarize(df); sm.to_csv(out/'causal_state_similarity_summary.csv',index=False)
        print('\n'+sm.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
        (out/'analysis_summary.txt').write_text(sm.to_string(index=False,float_format=lambda x:f'{x:.6f}')+'\n',encoding='utf-8')
        exp.write_json(out/'metadata.json',{
            'script':'analyze_supplier_patch_vs_causal_rg_v1.py','model':a.model,'repo_id':spec.repo_id,
            'groups':list(groups),'scales':scales,'eval_N_requested':a.eval_max_samples,'no_generation':True,
            'comparison':'MOVE=h_patch-h_real vs RG=h_real-h_gray at selected causal states'})
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=='__main__': main()
