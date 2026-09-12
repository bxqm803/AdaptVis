#!/usr/bin/env python3
"""Uniform subject/reference swap: spatial readout, generation and update effects.

Independent entry point in AdaptVis/llava16, using existing repository helpers.
Every image is forwarded twice with A->B and B->A questions, plus NoImage
controls. The expected label permutation is L<->R, above<->below. Swapped
hidden states are freshly computed, NEVER obtained by negating the original.

Fixed positions: all tokens of the subject phrase, reference phrase, last token.
No GT-dependent WHERE or per-example policy search. Identity pairing maps the
original subject to swapped reference; role pairing maps subject to subject.
These phrase bundles differ from the old oracle causal core. last is an
output-proximal control, not evidence of spatial computation by itself.

Readout: frozen Synthetic-only centered cosine codebook at L25 (img/residual/
no_image). Update: a_L = block_out[L]-block_out[L-1]. Attribution sums a dot
candidate-score-gradient across each phrase's tokens. Centered 4-way effect
vectors are compared before/after inverse-relation permutation. Fixed-layer
cancellation h_out -= a_clean separately measures actual final score effects.
The effect is clean-minus-cancel. No joint token search or sign-based edit.

GT is used ONLY in correctness and diagnostic GT-margin summaries. Equivariance
alone is not correctness. Differences under swap also include wording/order
and contextual effects; this experiment does not by itself isolate a unique
role-binding mechanism or establish behavior of the old causal core.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

REL=('left','right','above','below')
PERM=np.array([1,0,3,2])
INV=dict(zip(REL,np.asarray(REL)[PERM]))
PROMPT='Determine the spatial relation of the {subject} to the {reference} in the image. Answer with left, right, above, or below.'
ROLES=('subject','reference','last')


def canon(x):
    s=str(x).lower().strip()
    return {'on':'above','under':'below','over':'above'}.get(s,s)


def cosine(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/n) if n>1e-12 else float('nan')


def centered(x):return np.asarray(x,float)-np.mean(x)


def softmax(x):
    z=np.exp(np.asarray(x)-np.max(x));return z/z.sum()


def ratio(x,n):return float(x/n) if n else float('nan')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fit_codebook(X,y):
    center=X.mean(axis=0)
    directions=np.stack([X[y==r].mean(axis=0)-center for r in REL])
    norms=np.linalg.norm(directions,axis=-1,keepdims=True)
    return center,directions/np.maximum(norms,1e-12)


def readout(x,codebook):
    c,d=codebook;v=x-c
    return d@(v/max(np.linalg.norm(v),1e-12))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',default='qwen-3b',choices=['qwen-3b','qwen-7b'])
    p.add_argument('--dataset',default='coco_two',choices=['coco_two','vg_two'])
    p.add_argument('--data-root',default='data')
    p.add_argument('--target-max-samples',type=int,default=80,help='First N valid records; 0=all. Same uniform protocol on VG.')
    p.add_argument('--synthetic-dir',default='synthetic_shapes_4dir_400')
    p.add_argument('--synthetic-labels',default='')
    p.add_argument('--source-cache',default='',help='Existing trusted Synthetic hsub/href NPZ; default based on model')
    p.add_argument('--cache-dir',default='output/qwen3b_hsub_href_spatial_cache')
    p.add_argument('--readout-layer',type=int,default=25)
    p.add_argument('--update-layers',default='20-26')
    p.add_argument('--cancel-layers',default='25',help='Predeclared layers for single-bundle cancellation; empty disables')
    p.add_argument('--pool',default='mean',choices=['mean','last'])
    p.add_argument('--prompt-template',default=PROMPT)
    p.add_argument('--max-new-tokens',type=int,default=6)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--attn-impl',default='eager',choices=['eager','sdpa','flash_attention_2','none'])
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.target_max_samples<0 or a.readout_layer<0 or a.max_new_tokens<1:p.error('invalid sample/layer/token count')
    a.source_max_samples=0;a.keep_fp32=False
    a.answer_surface='above_below';a.answer_prefix='';a.answer_suffix='';a.sequence_score_reduction='mean'
    return a


def source_books(path,a):
    with np.load(path,allow_pickle=True) as z:
        for key,val in [('model',a.model),('pool',a.pool),('prompt_template',a.prompt_template)]:
            if key not in z or str(z[key].item())!=val:raise ValueError(f'Source cache {key} mismatch')
        if not np.array_equal(z['decoder_block_index'],np.arange(z['img'].shape[1])):raise ValueError('Layer indexing mismatch')
        si=z['img'][:,a.readout_layer].astype(np.float32)
        sn=z['no_image'][:,a.readout_layer].astype(np.float32)
        y=np.asarray([canon(r) for r in z['relation']])
        sid=np.asarray(z['sample_index'])
    if len(np.unique(sid))!=len(sid) or len(y)!=len(si) or len(sid)!=len(y):raise ValueError('Invalid source IDs/shapes')
    if not set(y)<=set(REL) or any(np.sum(y==r)<2 for r in REL):raise ValueError('Missing source classes')
    if not np.isfinite(si).all() or not np.isfinite(sn).all():raise ValueError('Nonfinite source')
    ep=path.with_suffix(path.suffix+'.errors.json')
    if ep.exists() and json.loads(ep.read_text()):raise ValueError('Source extraction errors; repair cache')
    return {name:fit_codebook(x,y) for name,x in [('img',si),('residual',si-sn),('no_image',sn)]},len(y)


def object_positions(sp,processor,batch,sub,ref,pool):
    ids=batch['input_ids'][0].detach().cpu().tolist()
    a=sp.dh.locate_phrase_positions(processor.tokenizer,ids,sub)
    b=sp.dh.locate_phrase_positions(processor.tokenizer,ids,ref)
    if not a or not b or set(a)&set(b):raise ValueError('Ambiguous/overlapping object spans')
    # Readout pooling follows source cache; update interventions use the full phrase.
    return dict(subject=a,reference=b,last=[len(ids)-1])


def relation_state(states,L,pos,pool):
    def get(role):
        x=states[L][0,pos[role]]
        return x.mean(axis=0) if pool=='mean' else x[-1]
    return get('subject')-get('reference')


def margin_effect(v,scores,gt):
    i=REL.index(gt);foil=max([j for j in range(4) if j!=i],key=lambda j:scores[j])
    return float(v[i]-v[foil])


def run_view(old,sp,torch,a,model,processor,layers,image,sub,ref,gt,books,update_layers,cancel_layers):
    question=a.prompt_template.format(subject=sub,reference=ref)
    def batch_for(img):
        rendered=sp.dh.build_chat_prompt(processor,question,img is not None)
        return sp.dh.process_inputs(processor,rendered,img,torch.device(a.device))
    rb=batch_for(image);nb=batch_for(None)
    pos=object_positions(sp,processor,rb,sub,ref,a.pool)
    npos=object_positions(sp,processor,nb,sub,ref,a.pool)
    needed=sorted(set([a.readout_layer]+update_layers+[L-1 for L in update_layers]))
    rs=old.capture_prompt_blocks(model,layers,rb,needed)
    ns=old.capture_prompt_blocks(model,layers,nb,[a.readout_layer])
    ri=relation_state(rs,a.readout_layer,pos,a.pool)
    rn=relation_state(ns,a.readout_layer,npos,a.pool)
    spatial={name:readout(x,books[name]) for name,x in [('img',ri),('residual',ri-rn),('no_image',rn)]}
    entries={(L,role):{p:rs[L][0,p].astype(np.float32)-rs[L-1][0,p].astype(np.float32)
                         for p in pos[role]} for L in update_layers for role in ROLES}
    del rs,ns,nb
    text=old.base.generate_text(model,processor,rb,max_new_tokens=a.max_new_tokens)
    pred=old.traj.normalize_relation(old.base,text) or 'invalid'
    ids=old.encode_candidate_ids(processor,old.candidate_texts(a)) if not hasattr(a,'candidate_ids') else a.candidate_ids
    a.candidate_ids=ids
    scores=np.zeros(4);effects={key:np.zeros(4) for key in entries}
    for j,r in enumerate(REL):
        s,grads=old.sequence_score_and_grads(model=model,decoder_layers=layers,batch=rb,
            answer_ids=ids[r],reduction='mean',grad_layers=update_layers)
        scores[j]=s
        for (L,role),vecs in entries.items():
            g=grads[L]
            if g is None:raise ValueError('Missing gradient')
            effects[(L,role)][j]=sum(float(v@g[0,p]) for p,v in vecs.items())
        del grads
    finite={}
    for L in cancel_layers:
        for role in ROLES:
            patch={L:{p:-v for p,v in entries[(L,role)].items()}}
            edited=np.array([old.sequence_score(model=model,batch=rb,answer_ids=ids[r],reduction='mean',
                decoder_layers=layers,patch_map=patch) for r in REL])
            finite[(L,role)]=scores-edited
    if not np.isfinite(scores).all() or any(not np.isfinite(v).all() for v in list(effects.values())+list(finite.values())):
        raise ValueError('Nonfinite candidate score/effect')
    return dict(question=question,gt=gt,pred=pred,text=text,scores=scores,spatial=spatial,
        positions=pos,effects=effects,finite=finite,
        norms={key:float(np.sqrt(sum(float(v@v) for v in vec.values()))) for key,vec in entries.items()})


def compare_pair(sid,original,swapped,update_layers,cancel_layers):
    o,w=original,swapped
    sample=[];pairs=[];effects=[]
    for repr_name in o['spatial']:
        so,sw=o['spatial'][repr_name],w['spatial'][repr_name]
        po,pw=REL[int(so.argmax())],REL[int(sw.argmax())]
        sample.append(dict(sid=sid,representation=repr_name,gt=o['gt'],swapped_gt=w['gt'],
            original_mid=po,swapped_mid=pw,mid_original_correct=po==o['gt'],mid_swapped_correct=pw==w['gt'],
            mid_both_correct=po==o['gt'] and pw==w['gt'],mid_equivariant=pw==INV[po],
            original_generation=o['pred'],swapped_generation=w['pred'],
            gen_original_correct=o['pred']==o['gt'],gen_swapped_correct=w['pred']==w['gt'],
            gen_both_correct=o['pred']==o['gt'] and w['pred']==w['gt'],
            gen_equivariant=o['pred'] in INV and w['pred']==INV.get(o['pred']),
            invalid_generation=o['pred'] not in REL or w['pred'] not in REL,
            spatial_aligned_cosine=cosine(centered(so),centered(sw)[PERM]),
            final_aligned_probability_L1=float(np.abs(softmax(o['scores'])-softmax(w['scores'])[PERM]).sum())))
    for kind,key,ls in [('gradient','effects',update_layers),('cancellation','finite',cancel_layers)]:
        for L in ls:
            for role in ROLES:
                for view,v in [('original',o),('swapped',w)]:
                    e=v[key][(L,role)]
                    row=dict(sid=sid,view=view,kind=kind,layer=L,role=role,n_tokens=len(v['positions'][role]),
                        update_norm=v['norms'][(L,role)],gt_margin_effect=margin_effect(e,v['scores'],v['gt']))
                    row.update({f'effect_{r}':float(e[j]) for j,r in enumerate(REL)})
                    row['clean_candidate_prediction']=REL[int(np.argmax(v['scores']))]
                    if kind=='cancellation':
                        edited=v['scores']-e
                        row['cancel_candidate_prediction']=REL[int(np.argmax(edited))]
                        row['cancel_candidate_correct']=row['cancel_candidate_prediction']==v['gt']
                    effects.append(row)
                for alignment in ('identity','role'):
                    other=({'subject':'reference','reference':'subject','last':'last'}[role]
                           if alignment=='identity' else role)
                    x,y=o[key][(L,role)],w[key][(L,other)]
                    xo,ys=centered(x),centered(y)
                    bo,bw=margin_effect(x,o['scores'],o['gt']),margin_effect(y,w['scores'],w['gt'])
                    active=abs(bo)>1e-8 and abs(bw)>1e-8
                    pairs.append(dict(sid=sid,kind=kind,layer=L,original_role=role,swapped_role=other,
                        alignment=alignment,aligned_cosine=cosine(xo,ys[PERM]),
                        unaligned_cosine=cosine(xo,ys),original_effect_norm=float(np.linalg.norm(xo)),
                        swapped_effect_norm=float(np.linalg.norm(ys)),original_gt_effect=bo,swapped_gt_effect=bw,
                        gt_sign_active=bool(active),gt_sign_agrees=bool(np.sign(bo)==np.sign(bw)) if active else np.nan))
    return sample,pairs,effects


def save_reports(out,samples,pairs,effects):
    sd,pd_,ed=pd.DataFrame(samples),pd.DataFrame(pairs),pd.DataFrame(effects)
    sd.to_csv(out/'per_sample_swap.csv',index=False)
    pd_.to_csv(out/'paired_update_effects.csv',index=False)
    ed.to_csv(out/'per_view_update_effects.csv',index=False)
    summary=[]
    for name,g in sd.groupby('representation'):
        both=g.mid_both_correct.to_numpy(bool);genboth=g.gen_both_correct.to_numpy(bool)
        summary.append(dict(representation=name,N=len(g),original_mid_acc=g.mid_original_correct.mean(),
            swapped_mid_acc=g.mid_swapped_correct.mean(),mid_equivariance=g.mid_equivariant.mean(),
            original_gen_acc=g.gen_original_correct.mean(),swapped_gen_acc=g.gen_swapped_correct.mean(),
            generation_equivariance=g.gen_equivariant.mean(),mid_both_correct_N=int(both.sum()),
            mid_both_correct_but_generation_not_both_N=int(np.sum(both&~genboth)),
            generation_pair_failure_given_mid_both_correct=ratio(np.sum(both&~genboth),both.sum()),
            invalid_generation_pairs=int(g.invalid_generation.sum())))
    su=pd.DataFrame(summary);su.to_csv(out/'swap_summary.csv',index=False)
    merged=pd_.merge(sd[sd.representation=='residual'][['sid','mid_both_correct','gen_both_correct']],on='sid',validate='many_to_one')
    rows=[]
    for keys,g in merged.groupby(['kind','layer','alignment','original_role']):
        for cohort,mask in [('all',np.ones(len(g),bool)),('mid_both_correct',g.mid_both_correct),
                           ('mid_both_correct_gen_failure',g.mid_both_correct&~g.gen_both_correct)]:
            z=g.loc[mask];active=z.gt_sign_active.astype(bool)
            rows.append(dict(kind=keys[0],layer=keys[1],alignment=keys[2],original_role=keys[3],cohort=cohort,
                N=len(z),aligned_valid_N=int(z.aligned_cosine.notna().sum()),
                aligned_cosine_mean=z.aligned_cosine.mean(),unaligned_cosine_mean=z.unaligned_cosine.mean(),
                gt_sign_active_N=int(active.sum()),gt_sign_agreement=z.loc[active,'gt_sign_agrees'].mean()))
    agg=pd.DataFrame(rows);agg.to_csv(out/'update_swap_summary.csv',index=False)
    report='SUBJECT / REFERENCE SWAP — FIXED POSITIONS, NO ORACLE WHERE\n'
    report+='Label inversion is expected; fresh forwards are used in both directions.\nEquivariance alone is not correctness; object-bundle effects do not establish old causal-core behavior.\n\n'
    report+=su.to_string(index=False)
    report+='\n\nACTUAL CANCELLATION EFFECTS (all pairs)\n'
    report+=agg[(agg.kind=='cancellation')&(agg.cohort=='all')].to_string(index=False)
    report+='\n\nFull gradient/cancellation, layer, alignment and cohort results: update_swap_summary.csv\n'
    (out/'analysis_summary.txt').write_text(report,encoding='utf-8');print(report,flush=True)


def main():
    a=parse_args()
    import torch
    import analyze_hsub_href_spatial_update_sign_v1 as sp
    import eval_real_causal_token_update_gating_v1 as old
    from PIL import Image
    from tqdm import tqdm
    update_layers=old.parse_layers(a.update_layers)
    cancel_layers=old.parse_layers(a.cancel_layers) if a.cancel_layers.strip() else []
    if not update_layers or min(update_layers)<1 or not set(cancel_layers)<=set(update_layers):raise ValueError('Invalid update/cancel layers')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()) and not a.overwrite:raise FileExistsError('Use new output-dir or --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    source=Path(a.source_cache) if a.source_cache else Path(a.cache_dir)/f'{a.model}_synthetic_hsub_href_all.npz'
    if out.resolve()==source.parent.resolve() or out.resolve() in source.resolve().parents:raise ValueError('Output must not contain source cache')
    files=['pairs.jsonl','errors.jsonl','metadata.json','per_sample_swap.csv','paired_update_effects.csv',
           'per_view_update_effects.csv','swap_summary.csv','update_swap_summary.csv','analysis_summary.txt']
    if a.overwrite:
        for f in files:(out/f).unlink(missing_ok=True)
    targets,audit=sp.load_target_rows(a)
    if not targets:raise ValueError('No target records')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    model,processor,layers,_,_=old.load_model(a,sp.base)
    metadata=dict(args=vars(a).copy(),requested_N=len(targets),complete_N=0,status='running',
        target_loader_audit=audit,positions='all subject phrase tokens, all reference phrase tokens, last',
        attribution='sum of actual clean updates dot candidate gradients; centered for vector comparison',
        selection='fixed layers and roles; no target-dependent selection',
        caveat='Role swap changes wording/order/context; consistency does not prove correctness.')
    samples=[];pairs=[];effects=[]
    try:
        if max(update_layers+[a.readout_layer])>=len(layers):raise ValueError('Layer out of bounds')
        if not source.exists():
            if a.source_cache:raise FileNotFoundError(source)
            sr,_=sp.load_synthetic_rows(a)
            got=sp.extract_hidden_cache(args=a,rows=sr,model=model,processor=processor,layers=layers,
                cache_path=source,desc='Synthetic spatial codebook')
            if len(got[0])!=len(sr):raise ValueError('Source extraction incomplete')
        books,n=source_books(source,a);metadata['source_N']=n;metadata['source_sha256']=sha(source)
        old.write_json(out/'metadata.json',metadata)
        a.candidate_ids=old.encode_candidate_ids(processor,old.candidate_texts(a))
        for rec in tqdm(targets,desc='ROLE SWAP / FIXED UPDATE EFFECTS'):
            try:
                if rec['subject'].strip().lower()==rec['reference'].strip().lower():raise ValueError('Identical object names')
                with Image.open(rec['image_path']) as raw:
                    image=raw.convert('RGB')
                    try:
                        o=run_view(old,sp,torch,a,model,processor,layers,image,rec['subject'],rec['reference'],
                            rec['relation'],books,update_layers,cancel_layers)
                        w=run_view(old,sp,torch,a,model,processor,layers,image,rec['reference'],rec['subject'],
                            INV[rec['relation']],books,update_layers,cancel_layers)
                    finally:image.close()
                sr,pr,er=compare_pair(rec['sid'],o,w,update_layers,cancel_layers)
                old.append_jsonl(out/'pairs.jsonl',dict(sid=rec['sid'],subject=rec['subject'],reference=rec['reference'],
                    original_question=o['question'],swapped_question=w['question'],
                    original_text=o['text'],swapped_text=w['text'],original_positions=o['positions'],swapped_positions=w['positions'],
                    original_candidate_scores=o['scores'].tolist(),swapped_candidate_scores=w['scores'].tolist(),
                    original_spatial_scores={k:v.tolist() for k,v in o['spatial'].items()},
                    swapped_spatial_scores={k:v.tolist() for k,v in w['spatial'].items()},
                    sample_rows=sr,paired_effect_rows=pr,view_effect_rows=er))
                samples.extend(sr);pairs.extend(pr);effects.extend(er)
                metadata['complete_N']+=1;old.write_json(out/'metadata.json',metadata)
            except Exception as exc:
                old.append_jsonl(out/'errors.jsonl',dict(sid=rec['sid'],error=f'{type(exc).__name__}: {exc}'))
                raise
        metadata['status']='complete'
    except Exception:
        metadata['status']='failed_partial';raise
    finally:
        old.write_json(out/'metadata.json',metadata)
        if samples:save_reports(out,samples,pairs,effects)
        del model,processor,layers
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':main()
