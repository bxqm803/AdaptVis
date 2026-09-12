#!/usr/bin/env python3
"""Actual block update -> four frozen last-token writer directions.

C[L,p,T,r] = a_clean[L,p] dot grad_h[L,p] (unit(v[T,r]) dot h[T,last]).
No answer-logit gradient and no GT-selected direction. All four directions
are evaluated independently, at each requested writer layer T. L must be < T.
Writers are loaded from existing learned_writers.npz (L32_left/right/on/under,
or above/below). They are NOT reconstructed as LM-head rows or refitted.

Uses the legacy oracle-ranked token trajectories, identical to the actual
update experiment: WHERE remains oracle-conditioned, not a deployment claim.
Original standard COCO questions preserve ranked token positions. Baseline
correctness is freshly evaluated by free generation. GT only groups results.
Positive/negative means promote/oppose a writer projection, NOT correct/wrong.
No update is removed or amplified in this statistical diagnostic.

Reports micro update statistics AND equal-weight sample macro statistics.
Near-zero epsilon is explicit, after unit-normalizing writers only. No per-
update normalization hides the magnitude of an actual update. Writer geometry
is exported because the four directions need not be orthogonal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

REL=('left','right','above','below')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',default='qwen-3b',choices=['qwen-3b','qwen-7b'])
    p.add_argument('--data-root',default='data')
    p.add_argument('--prompt-jsonl',default='prompts/COCO_QA_two_obj_with_answer_four_options.jsonl')
    p.add_argument('--writers-npz',required=True,help='Existing learned_writers.npz from the writer calibration run')
    p.add_argument('--writer-layers',default='auto',help='auto=all complete four-direction layers in NPZ; or 32,34,35')
    p.add_argument('--ranked-causal',required=True)
    p.add_argument('--causal-layers',default='20-26')
    p.add_argument('--causal-top-k',type=int,default=7)
    p.add_argument('--causal-categories',default='')
    p.add_argument('--update-layers',default='8-26')
    p.add_argument('--exclude-target-layer',action='store_true')
    p.add_argument('--epsilon',type=float,default=1e-8)
    p.add_argument('--max-new-tokens',type=int,default=6)
    p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--eval-max-samples',type=int,default=80,help='Legacy stratified cap; 0=all')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--attn-impl',default='eager',choices=['eager','sdpa','flash_attention_2','none'])
    p.add_argument('--output-dir',required=True)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if not np.isfinite(a.epsilon) or a.epsilon<0 or a.causal_top_k<1 or min(a.max_samples,a.eval_max_samples)<0:
        p.error('Invalid epsilon, K or sample count')
    if a.max_new_tokens<1:p.error('max-new-tokens must be positive')
    return a


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_writers(path,requested):
    found={}
    with np.load(path,allow_pickle=False) as z:
        for key in z.files:
            m=re.fullmatch(r'L(\d+)_(left|right|above|below|on|under)',key)
            if not m:continue
            T=int(m[1]);r={'on':'above','under':'below'}.get(m[2],m[2])
            if (T,r) in found:raise ValueError(f'Duplicate writer aliases for {(T,r)}')
            v=np.asarray(z[key],dtype=np.float32)
            if v.ndim!=1 or not np.isfinite(v).all() or np.linalg.norm(v)<1e-12:
                raise ValueError(f'Invalid writer {key}: must be a finite, nonzero 1D vector')
            found[T,r]=v
    complete=sorted(T for T in {t for t,r in found} if all((T,r) in found for r in REL))
    wanted=complete if requested=='auto' else sorted(set(int(x) for x in requested.split(',')))
    if not wanted or not set(wanted)<=set(complete):raise ValueError(f'Need full four-direction layers; available={complete}')
    norms={k:float(np.linalg.norm(v)) for k,v in found.items() if k[0] in wanted}
    vecs={k:v/norms[k] for k,v in found.items() if k in norms}
    if len({v.shape for v in vecs.values()})!=1:raise ValueError('Writer hidden dimensions differ')
    return vecs,norms,wanted


def stats(values,eps):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Statistics need nonempty finite values')
    pos=x>eps;neg=x < -eps;zero=~(pos|neg)
    return dict(N_updates=len(x),mean_C=float(x.mean()),mean_abs_C=float(np.abs(x).mean()),
        positive_N=int(pos.sum()),negative_N=int(neg.sum()),neutral_N=int(zero.sum()),
        positive_fraction=float(pos.mean()),negative_fraction=float(neg.mean()),neutral_fraction=float(zero.mean()),
        positive_mean_C=float(x[pos].mean()) if pos.any() else float('nan'),
        negative_mean_C=float(x[neg].mean()) if neg.any() else float('nan'))


def summarize(df,keys,eps):
    rows=[]
    for vals,g in df.groupby(keys,sort=True,dropna=False):
        if not isinstance(vals,tuple):vals=(vals,)
        per=[stats(z.C,eps) for _,z in g.groupby('sid')]
        row=dict(zip(keys,vals));row.update(stats(g.C,eps));row['N_samples']=len(per)
        for col in ['mean_C','mean_abs_C','positive_fraction','negative_fraction','neutral_fraction']:
            row['sample_macro_'+col]=float(np.mean([v[col] for v in per]))
        for name in ['positive_mean_C','negative_mean_C']:
            vv=[v[name] for v in per if np.isfinite(v[name])]
            row['sample_macro_'+name]=float(np.mean(vv)) if vv else float('nan')
            row['samples_with_'+name.split('_')[0]]=len(vv)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_tokens(rows,batch,tok):
    ids=batch['input_ids'][0].detach().cpu().tolist();tokens=tok.convert_ids_to_tokens(ids)
    for r in rows.itertuples():
        p=int(r.position)
        if not 0<=p<len(ids)-1:raise ValueError(f'Invalid WHERE position {p}')
        if str(r.token)!=str(tokens[p]).replace('\n','\\n'):
            raise ValueError(f'WHERE token mismatch at {p}; check model and original prompt')
        if hasattr(r,'token_id') and int(r.token_id)!=ids[p]:raise ValueError('WHERE token ID mismatch')


def direction_contributions(old,torch,model,layers,batch,entries,T,v):
    gl=sorted({e['update_layer'] for e in entries})
    cap=old.GraphBlockCapture(layers,gl+[T])
    try:
        with torch.enable_grad():
            out=model(**batch,use_cache=False,return_dict=True)
            h=cap.states[T][0,-1]
            if h.numel()!=v.size:raise ValueError('Writer/model dimension mismatch')
            objective=torch.dot(h.float(),torch.as_tensor(v,device=h.device,dtype=torch.float32))
            grads=torch.autograd.grad(objective,[cap.states[L] for L in gl],allow_unused=False)
            gmap=dict(zip(gl,grads));values=[]
            for e in entries:
                g=gmap[e['update_layer']][0,e['real_position']].float()
                u=torch.as_tensor(e['_real_update'],device=g.device,dtype=torch.float32)
                values.append(float(torch.dot(u,g).detach().cpu()))
            projection=float(objective.detach().cpu())
            del out,grads,gmap
        return np.asarray(values),projection
    finally:cap.close()


def save_reports(out,rows,base,eps):
    df=pd.DataFrame(rows)
    df.to_csv(out/'per_update_four_direction.csv',index=False)
    pd.DataFrame(base).to_csv(out/'baseline_per_sample.csv',index=False)
    keys=['writer_layer','update_layer','direction']
    tables={
        'summary_all.csv':summarize(df,keys,eps),
        'summary_by_correctness.csv':summarize(df,keys+['baseline_correct'],eps),
        'summary_by_gt_correctness.csv':summarize(df,keys+['gt','baseline_correct'],eps),
        'per_sample_layer_direction.csv':summarize(df,['sid']+keys+['gt','baseline_correct'],eps),
    }
    for name,t in tables.items():t.to_csv(out/name,index=False)
    text='ACTUAL UPDATE -> UNIT LAST-WRITER PROJECTION\n'
    text+=f'Baseline N={len(base)} accuracy={np.mean([r["correct"] for r in base]):.4f}; epsilon={eps:g}\n'
    text+='WHERE uses the existing oracle-ranked trajectories; all four directions evaluated independently.\n'
    text+='Signs mean promote/oppose this projection, not correct/incorrect. Layers are zero-based block outputs.\n'
    text+='Micro means weight updates equally; sample_macro means weight samples equally.\n\n'
    cols=keys+['baseline_correct','N_samples','N_updates','mean_C','mean_abs_C',
        'positive_N','negative_N','neutral_N','positive_fraction','negative_fraction',
        'positive_mean_C','negative_mean_C','sample_macro_mean_C']
    text+=tables['summary_by_correctness.csv'][cols].to_string(index=False)
    text+='\n\nGT-stratified results: summary_by_gt_correctness.csv\n'
    (out/'analysis_summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)


def main():
    a=parse_args()
    import torch
    import eval_real_causal_token_update_gating_v1 as old
    from tqdm import tqdm
    writers,norms,targets=load_writers(a.writers_npz,a.writer_layers)
    ul=old.parse_layers(a.update_layers);cl=old.parse_layers(a.causal_layers)
    if not ul or min(ul)<1 or max(ul)>=min(targets):raise ValueError('Require 1 <= every update layer < every writer layer')
    out=Path(a.output_dir)
    protected=[Path(a.writers_npz),Path(a.ranked_causal),Path(a.prompt_jsonl)]
    if any(out.resolve()==p.resolve() or out.resolve() in p.resolve().parents for p in protected):
        raise ValueError('Output directory must not contain input files')
    files=['metadata.json','updates.jsonl','samples.jsonl','errors.jsonl','writer_geometry.csv',
           'selected_causal_states.csv','per_update_four_direction.csv','baseline_per_sample.csv',
           'summary_all.csv','summary_by_correctness.csv','summary_by_gt_correctness.csv',
           'per_sample_layer_direction.csv','analysis_summary.txt']
    if not a.overwrite and any((out/f).exists() for f in files):raise FileExistsError('Use new output-dir or --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    if a.overwrite:
        for f in files:(out/f).unlink(missing_ok=True)
    geometry=[]
    for T in targets:
        for r in REL:
            row=dict(writer_layer=T,direction=r,original_norm=norms[T,r])
            row.update({f'cos_{s}':float(writers[T,r]@writers[T,s]) for s in REL});geometry.append(row)
    pd.DataFrame(geometry).to_csv(out/'writer_geometry.csv',index=False)
    two,meta,recs=old.load_data(a)
    if not meta:raise ValueError('Empty target set')
    sel=old.load_causal_selection(Path(a.ranked_causal),{m['sid'] for m in meta},cl,a.causal_top_k,old.parse_set(a.causal_categories))
    bysid={int(s):g for s,g in sel.groupby('sid')}
    if any(m['sid'] not in bysid for m in meta):raise ValueError('Missing WHERE samples; no silent dropping')
    sel.to_csv(out/'selected_causal_states.csv',index=False)
    metadata=dict(args=vars(a),requested_N=len(meta),complete_N=0,status='running',writer_layers=targets,
        formula='C = actual_update dot grad(unit_writer dot last_block_output)',
        where='oracle-ranked',writer_provenance='Loaded unchanged from supplied NPZ; calibration overlap not inferred',
        inputs={str(p.resolve()):sha(p) for p in protected})
    old.write_json(out/'metadata.json',metadata)
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    model,processor,layers,_,_=old.load_model(a,two)
    rows=[];baselines=[]
    try:
        if max(targets+cl)>=len(layers):raise ValueError('Layer index outside model')
        for m in tqdm(meta,desc='ACTUAL UPDATE -> FOUR LAST WRITERS'):
            image=None
            try:
                image=old.base.record_image(recs[m['sid']]).convert('RGB')
                batch=old.base.make_question_batch(processor=processor,image=image,question_text=m['question_text'],device=torch.device(a.device))
                validate_tokens(bysid[m['sid']],batch,old.tokenizer_of(processor))
                text=old.base.generate_text(model,processor,batch,max_new_tokens=a.max_new_tokens)
                pred=old.traj.normalize_relation(old.base,text) or 'invalid'
                ok=pred==m['gt']
                states=old.capture_prompt_blocks(model,layers,batch,sorted(set(ul+[L-1 for L in ul])))
                entries=old.build_real_updates(sid=m['sid'],gt=m['gt'],baseline_correct=ok,
                    specs=old.causal_position_specs(bysid[m['sid']]),r2n={},real_states=states,no_states={},
                    update_layers=ul,exclude_target_layer=a.exclude_target_layer)
                del states
                if not entries:raise ValueError('No actual updates')
                current=[]
                for T in targets:
                    for r in REL:
                        C,projection=direction_contributions(old,torch,model,layers,batch,entries,T,writers[T,r])
                        if not np.isfinite(C).all():raise ValueError('Nonfinite contribution')
                        for e,c in zip(entries,C):
                            current.append(dict(sid=m['sid'],gt=m['gt'],baseline_prediction=pred,baseline_correct=ok,
                                writer_layer=T,update_layer=e['update_layer'],direction=r,position=e['real_position'],
                                token=e['token'],actual_update_norm=e['real_update_norm'],C=float(c),
                                polarity='positive' if c>a.epsilon else 'negative' if c < -a.epsilon else 'neutral',
                                clean_last_projection=projection))
                b=dict(sid=m['sid'],gt=m['gt'],prediction=pred,correct=ok,text=text,N_updates=len(entries))
                for row in current:old.append_jsonl(out/'updates.jsonl',row)
                old.append_jsonl(out/'samples.jsonl',b)
                rows.extend(current);baselines.append(b);metadata['complete_N']+=1
                old.write_json(out/'metadata.json',metadata)
                del entries,batch
            except Exception as exc:
                old.append_jsonl(out/'errors.jsonl',dict(sid=m['sid'],error=f'{type(exc).__name__}: {exc}'));raise
            finally:
                if image is not None:image.close()
        metadata['status']='complete'
    except Exception:
        metadata['status']='failed_partial';raise
    finally:
        old.write_json(out/'metadata.json',metadata)
        if rows:save_reports(out,rows,baselines,a.epsilon)
        del model,processor,layers
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':main()
