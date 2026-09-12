#!/usr/bin/env python3
"""Frozen oracle-WHERE: compare oracle, generation, hard-middle and soft-middle HOW.

Run in AdaptVis/llava16. Reuses eval_real_causal_token_update_gating_v1.py.
Reads per_sample_layer.csv from the SAME-question confidence diagnostic, but
intervenes on ORIGINAL standard COCO prompts to preserve old WHERE positions.
This is a two-prompt evidence-routing experiment, NOT same-forward steering.
WHERE is still oracle. Mid evidence uses a Synthetic-labeled codebook.

s_r = mean/sum teacher-forced answer-token log probability; p=softmax(s).
q = softmax(frozen middle cosine scores / temperature).
J(q) = sum_r q_r log p_r; dJ/dh = sum_r (q_r-p_r) ds_r/dh.
Contract four candidate gradients with each actual clean update once. Both q
and the gradient coefficients are evaluated at the clean trajectory; no
backpropagation through the middle readout. Frozen clean updates are added
at prefill only: h_out += alpha * sign(B) * a_clean (legacy intervention).

Policies: oracle_margin (legacy GT-best competitor), oracle_ce (one-hot GT),
baseline_ce (one-hot clean generated answer), mid_hard_ce (one-hot mid argmax),
mid_hard_margin (legacy-style hard selector), mid_soft_T*, blind_amplify.
Hard CE and soft CE have identical objective families. Invalid baseline
answers cause baseline_ce to abstain, never use GT as a fallback.

Default N=80 stratified using legacy loader, alpha=.5, temperature=.1, K=7.
Use --eval-max-samples 0 for all 440. Explicit --ranked-causal is required.
No layer/temperature/scale is chosen on target results. Reports use only
complete samples; default fail-fast with incremental JSONL checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

REL = ("left", "right", "above", "below")


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(x - np.max(x))
    return z / z.sum()


def canon(x):
    s = str(x).lower().strip()
    return {"on":"above", "over":"above", "under":"below"}.get(s, s)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', default='qwen-3b', choices=['qwen-3b','qwen-7b'])
    p.add_argument('--data-root', default='data')
    p.add_argument('--prompt-jsonl', default='prompts/COCO_QA_two_obj_with_answer_four_options.jsonl')
    p.add_argument('--ranked-causal', required=True, help='Original ranked/selected causal CSV, same as legacy actual-update run')
    p.add_argument('--evidence-dir', default='output/qwen3b_sameprompt_midlayer_confidence_all440_v1')
    p.add_argument('--representation', default='residual', choices=['residual','img','no_image'])
    p.add_argument('--readout-layer', type=int, default=25)
    p.add_argument('--temperatures', default='0.1', help='Predeclared positive values; no target tuning')
    p.add_argument('--scales', default='0.5')
    p.add_argument('--causal-layers', default='20-26')
    p.add_argument('--causal-top-k', type=int, default=7)
    p.add_argument('--causal-categories', default='')
    p.add_argument('--update-layers', default='8-26')
    p.add_argument('--exclude-target-layer', action='store_true')
    p.add_argument('--decision-threshold', type=float, default=0.0)
    p.add_argument('--answer-surface', default='above_below', choices=['above_below','on_under'])
    p.add_argument('--answer-prefix', default='')
    p.add_argument('--answer-suffix', default='')
    p.add_argument('--sequence-score-reduction', default='mean', choices=['mean','sum'])
    p.add_argument('--max-new-tokens', type=int, default=6)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--eval-max-samples', type=int, default=80)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--attn-impl', default='eager', choices=['eager','sdpa','flash_attention_2','none'])
    p.add_argument('--finite-probe-samples', type=int, default=3,
                   help='Validate one update with central finite differences for the first N samples')
    p.add_argument('--finite-probe-scale', type=float, default=0.05)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    a.temps = sorted(set(float(x) for x in a.temperatures.split(',')))
    a.alphas = sorted(set(float(x) for x in a.scales.split(',')))
    if any(not np.isfinite(v) or v <= 0 for v in a.temps + a.alphas):
        p.error('temperatures and scales must be finite and positive')
    if a.decision_threshold < 0 or not np.isfinite(a.decision_threshold):
        p.error('decision-threshold must be finite and nonnegative')
    if min(a.max_samples,a.eval_max_samples,a.finite_probe_samples) < 0 or a.causal_top_k < 1:
        p.error('invalid sample count or K')
    if a.finite_probe_scale <= 0 or not np.isfinite(a.finite_probe_scale) or a.max_new_tokens < 1:
        p.error('invalid finite probe scale or generation length')
    return a


def evidence(a):
    path = Path(a.evidence_dir)/'per_sample_layer.csv'
    meta = json.loads((path.parent/'metadata.json').read_text())
    if meta.get('args',{}).get('model') != a.model:
        raise ValueError('Evidence model mismatch')
    if meta.get('baseline_protocol') != 'same_readout_question_free_generation':
        raise ValueError('Use the same-question confidence diagnostic evidence directory')
    d = pd.read_csv(path, keep_default_na=False)
    req = {'sid','gt','representation','layer','mid_prediction'} | {f'mid_score_{r}' for r in REL}
    if not req <= set(d):
        raise ValueError(f'Evidence missing: {req-set(d)}')
    d = d[(d.representation==a.representation)&(d.layer==a.readout_layer)].copy()
    if d.empty or d.sid.duplicated().any():
        raise ValueError('Empty or duplicate selected evidence SIDs')
    if not np.isfinite(d[[f'mid_score_{r}' for r in REL]].to_numpy(float)).all():
        raise ValueError('Nonfinite evidence scores')
    d['gt'] = d['gt'].map(canon)
    return d.set_index('sid'), path, meta


def policies(scores, contractions, gt_index, baseline_index, middle_scores, temps):
    """Pure numpy: exact local objective contractions; no hidden target fallback."""
    scores = np.asarray(scores, dtype=float)
    C = np.asarray(contractions, dtype=float)  # updates x 4
    p = softmax(scores)
    eye = np.eye(4)
    mid = int(np.argmax(middle_scores))
    def margin_coeff(k):
        others = [j for j in range(4) if j != k]
        foil = max(others, key=lambda j: scores[j])
        return eye[k]-eye[foil]
    coeffs = {
        'oracle_margin': margin_coeff(gt_index),
        'oracle_ce': eye[gt_index]-p,
        'baseline_ce': eye[baseline_index]-p if baseline_index is not None else np.zeros(4),
        'mid_hard_ce': eye[mid]-p,
        'mid_hard_margin': margin_coeff(mid),
    }
    qs = {}
    for t in temps:
        name = f'mid_soft_T{t:g}'
        qs[name] = softmax(np.asarray(middle_scores)/t)
        coeffs[name] = qs[name]-p
    B = {name:C@c for name,c in coeffs.items()}
    return B, qs, p


def validate_positions(g, batch, tokenizer):
    ids = batch['input_ids'][0].detach().cpu().tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    for row in g.itertuples():
        p = int(row.position)
        if not 0 <= p < len(ids)-1:
            raise ValueError(f'WHERE position {p} is invalid/last for prompt length {len(ids)}')
        token = str(toks[p]).replace('\n','\\n')
        if str(row.token) != token:
            raise ValueError(f'WHERE token mismatch at {p}: CSV={row.token!r}, current={token!r}; check original prompt/model')
        if hasattr(row,'token_id') and int(row.token_id) != int(ids[p]):
            raise ValueError(f'WHERE token ID mismatch at {p}')


def cohort_masks(d):
    bc = d.baseline_correct.to_numpy(bool)
    mc = d.mid_correct.to_numpy(bool)
    return {'all':np.ones(len(d),bool), 'baseline_wrong':~bc, 'baseline_correct':bc,
            'mid_wrong':~mc, 'mid_correct':mc,
            'baseline_correct_mid_wrong':bc & ~mc,
            'baseline_wrong_mid_correct':~bc & mc,
            'both_wrong':~bc & ~mc, 'conflict':d.conflict.to_numpy(bool)}


def ratio(k,n):
    return float(k/n) if n else float('nan')


def reports(out, generation, update_rows):
    gen = pd.DataFrame(generation)
    upd = pd.DataFrame(update_rows)
    gen.to_csv(out/'generation_per_sample.csv',index=False)
    upd.to_csv(out/'per_update_scores.csv',index=False)
    summaries = []
    for (policy,alpha),d in gen.groupby(['condition','scale'],sort=False):
        for cohort,mask in cohort_masks(d).items():
            z = d.loc[mask]
            bc,ec = z.baseline_correct.to_numpy(bool),z.correct.to_numpy(bool)
            w,c = int(np.sum(~bc & ec)),int(np.sum(bc & ~ec))
            summaries.append(dict(condition=policy,scale=alpha,cohort=cohort,N=len(z),
                baseline_accuracy=ratio(bc.sum(),len(z)), edited_accuracy=ratio(ec.sum(),len(z)),
                W2C=w,C2W=c,net=w-c,gain=ratio(w-c,len(z))))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'generation_summary.csv',index=False)
    agreement = []
    for reference in ('oracle_margin','oracle_ce'):
        for policy in [c[2:] for c in upd if c.startswith('B_')]:
            for cohort,mask in cohort_masks(upd).items():
                z = upd.loc[mask]
                ref = z['B_'+reference].to_numpy(float)
                pred = z['B_'+policy].to_numpy(float)
                eligible = np.abs(ref)>1e-8
                active = eligible & (np.abs(pred)>1e-8)
                ok = (np.sign(ref)==np.sign(pred)) & eligible
                weights = np.abs(ref)
                agreement.append(dict(reference=reference,policy=policy,cohort=cohort,N_updates=len(z),
                    eligible_N=int(eligible.sum()),active_N=int(active.sum()),
                    agreement_including_abstention=ratio(ok.sum(),eligible.sum()),
                    agreement_active=ratio((ok & active).sum(),active.sum()),
                    weighted_agreement=ratio(weights[ok].sum(),weights[eligible].sum())))
    pd.DataFrame(agreement).to_csv(out/'sign_agreement.csv',index=False)
    paired = []
    for (policy,alpha),d in gen[gen.condition.str.startswith('mid_soft_T')].groupby(['condition','scale']):
        hard = gen[(gen.condition=='mid_hard_ce') & (gen.scale==alpha)][['sid','correct']]
        z = d.merge(hard,on='sid',suffixes=('','_hard'),validate='one_to_one')
        for cohort,mask in cohort_masks(z).items():
            v=z.loc[mask];h=v.correct_hard.to_numpy(bool);s=v.correct.to_numpy(bool)
            paired.append(dict(condition=policy,scale=alpha,cohort=cohort,N=len(v),
                hard_accuracy=ratio(h.sum(),len(v)),soft_accuracy=ratio(s.sum(),len(v)),
                hard_wrong_soft_correct=int(np.sum(~h&s)),hard_correct_soft_wrong=int(np.sum(h&~s)),
                net=int(s.sum()-h.sum())))
    pd.DataFrame(paired).to_csv(out/'soft_vs_hard.csv',index=False)
    text = 'ORACLE-WHERE / HARD vs SOFT HOW\nOriginal COCO generation prompt; frozen separately-read middle evidence.\n'
    text += 'All policies share positions and frozen clean updates. No target policy selection.\n\n'
    text += summary[summary.cohort=='all'].to_string(index=False)
    text += '\n\nSOFT vs HARD CE (same objective family)\n'
    if paired:
        pair=pd.DataFrame(paired)
        text += pair[pair.cohort.isin(['all','mid_wrong','baseline_correct_mid_wrong'])].to_string(index=False)
    text += '\n\nSign agreement is a local diagnostic; generation determines actual intervention benefit.\n'
    (out/'analysis_summary.txt').write_text(text)
    print(text,flush=True)


def main():
    a=parse_args()
    # Delay GPU/repository imports so --help and numerical checks work on CPU.
    import torch
    import eval_real_causal_token_update_gating_v1 as old
    from tqdm import tqdm
    out=Path(a.output_dir)
    names=['samples.jsonl','updates.jsonl','finite_probes.jsonl','errors.jsonl','metadata.json',
           'selected_causal_states.csv','generation_per_sample.csv','per_update_scores.csv',
           'generation_summary.csv','sign_agreement.csv','soft_vs_hard.csv','analysis_summary.txt']
    if any((out/n).exists() for n in names) and not a.overwrite:
        raise FileExistsError(f'{out}: use a new output-dir or --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    # Only remove this script's files, never the directory or input artifacts.
    protected=[Path(a.ranked_causal).resolve(),Path(a.evidence_dir).resolve()]
    if any(out.resolve()==p or out.resolve() in p.parents for p in protected):
        raise ValueError('output-dir must not contain WHERE/evidence inputs')
    if a.overwrite:
        for n in names:
            (out/n).unlink(missing_ok=True)
    ev,evpath,evmeta=evidence(a)
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    two,meta,recs=old.load_data(a)
    if not meta:raise ValueError('No target samples')
    causal_layers=old.parse_layers(a.causal_layers)
    update_layers=old.parse_layers(a.update_layers)
    if not update_layers or min(update_layers)<1:raise ValueError('update-layers must be >=1')
    selected=old.load_causal_selection(Path(a.ranked_causal),{m['sid'] for m in meta},
        causal_layers,a.causal_top_k,old.parse_set(a.causal_categories))
    bysid={int(sid):g.copy() for sid,g in selected.groupby('sid')}
    for m in meta:
        sid=m['sid']
        if sid not in ev.index or sid not in bysid:raise ValueError(f'Missing evidence or WHERE for sid={sid}; no silent sample dropping')
        if ev.loc[sid,'gt']!=m['gt']:raise ValueError(f'GT mismatch sid={sid}')
        if 'gt' in bysid[sid] and any(bysid[sid]['gt'].map(canon)!=m['gt']):
            raise ValueError(f'WHERE GT mismatch sid={sid}')
    selected.to_csv(out/'selected_causal_states.csv',index=False)
    metadata=dict(args=vars(a),where='oracle fixed ranked positions',
        protocol='original_standard_COCO_prompt_with_frozen_separate_prompt_middle_evidence',
        evidence_prompt=evmeta.get('args',{}).get('prompt_template'),
        objective='J=sum q log softmax(sequence_scores); coefficients q-p frozen at clean state',
        requested_N=len(meta),complete_N=0,status='running',
        inputs={str(p.resolve()):sha(p) for p in [Path(a.ranked_causal),evpath,Path(a.prompt_jsonl)]})
    old.write_json(out/'metadata.json',metadata)
    model,processor,layers,_,_=old.load_model(a,two)
    generation=[];updates=[]
    try:
        capture=sorted(set(update_layers+[L-1 for L in update_layers]))
        if min(capture)<0 or max(capture)>=len(layers) or max(causal_layers)>=len(layers):
            raise ValueError('Layer out of range')
        candidate_ids=old.encode_candidate_ids(processor,old.candidate_texts(a))
        device=torch.device(a.device)
        for index,m in enumerate(tqdm(meta,desc='FROZEN WHERE / SOFT HOW')):
            sid=m['sid'];image=None
            try:
                image=old.base.record_image(recs[sid]).convert('RGB')
                batch=old.base.make_question_batch(processor=processor,image=image,
                    question_text=m['question_text'],device=device)
                validate_positions(bysid[sid],batch,old.tokenizer_of(processor))
                base_text=old.base.generate_text(model,processor,batch,max_new_tokens=a.max_new_tokens)
                base_pred=old.traj.normalize_relation(old.base,base_text)
                base_pred=base_pred if base_pred in REL else 'invalid'
                mid_scores=ev.loc[sid,[f'mid_score_{r}' for r in REL]].to_numpy(float)
                mid_pred=REL[int(mid_scores.argmax())]
                if canon(ev.loc[sid,'mid_prediction'])!=mid_pred:raise ValueError('Evidence argmax mismatch')
                common=dict(sid=sid,gt=m['gt'],baseline_prediction=base_pred,
                    baseline_correct=base_pred==m['gt'],mid_prediction=mid_pred,
                    mid_correct=mid_pred==m['gt'],conflict=mid_pred!=base_pred)
                states=old.capture_prompt_blocks(model,layers,batch,capture)
                entries=old.build_real_updates(sid=sid,gt=m['gt'],baseline_correct=common['baseline_correct'],
                    specs=old.causal_position_specs(bysid[sid]),r2n={},real_states=states,no_states={},
                    update_layers=update_layers,exclude_target_layer=a.exclude_target_layer)
                del states
                if not entries:raise ValueError(f'No updates sid={sid}')
                gl=sorted({e['update_layer'] for e in entries})
                scores=[];C=np.empty((len(entries),4),dtype=np.float64)
                for j,r in enumerate(REL):
                    val,grads=old.sequence_score_and_grads(model=model,decoder_layers=layers,batch=batch,
                        answer_ids=candidate_ids[r],reduction=a.sequence_score_reduction,grad_layers=gl)
                    scores.append(val)
                    for k,e in enumerate(entries):
                        g=grads[e['update_layer']]
                        if g is None:raise RuntimeError('Missing candidate gradient')
                        C[k,j]=np.dot(e['_real_update'],g[0,e['real_position']])
                    del grads
                if not np.isfinite(C).all() or not np.isfinite(scores).all():raise ValueError('Nonfinite scores/gradients')
                B,qs,p=policies(scores,C,REL.index(m['gt']),REL.index(base_pred) if base_pred in REL else None,mid_scores,a.temps)
                sample_updates=[]
                for k,e in enumerate(entries):
                    row=dict(**common,update_layer=e['update_layer'],real_position=e['real_position'],
                        token=e['token'],real_update_norm=e['real_update_norm'])
                    row.update({f'candidate_contribution_{r}':float(C[k,j]) for j,r in enumerate(REL)})
                    row.update({f'B_{name}':float(v[k]) for name,v in B.items()})
                    sample_updates.append(row)
                # Finite-difference diagnostic: one largest-|soft B| update, no policy selection.
                if index<a.finite_probe_samples:
                    soft_name=next(iter(qs));k=int(np.argmax(np.abs(B[soft_name])));e=entries[k]
                    js=[]
                    for direction in [-1,1]:
                        patch={e['update_layer']:{e['real_position']:direction*a.finite_probe_scale*e['_real_update']}}
                        vals=np.array([old.sequence_score(model=model,batch=batch,answer_ids=candidate_ids[r],
                            reduction=a.sequence_score_reduction,decoder_layers=layers,patch_map=patch) for r in REL])
                        logp=vals-np.logaddexp.reduce(vals)
                        js.append(float(qs[soft_name]@logp))
                    fd=(js[1]-js[0])/(2*a.finite_probe_scale)
                    old.append_jsonl(out/'finite_probes.jsonl',dict(sid=sid,policy=soft_name,
                        layer=e['update_layer'],position=e['real_position'],analytic=float(B[soft_name][k]),
                        central_difference=fd,abs_error=abs(fd-B[soft_name][k])))
                sample_gen=[dict(**common,condition='baseline',scale=0.0,prediction=base_pred,
                    correct=common['baseline_correct'],text=base_text,n_patched=0)]
                for alpha in a.alphas:
                    for name,values in list(B.items())+[('blind_amplify',np.ones(len(entries)))]:
                        scored=[dict(e,real_update_decision_score=float(values[k])) for k,e in enumerate(entries)]
                        patch,counts=old.build_real_update_patch_map(scored,
                            'real_all_amplify' if name=='blind_amplify' else 'real_signed',alpha,a.decision_threshold)
                        pred,text=old.generate_with_patch(model=model,processor=processor,decoder_layers=layers,
                            batch=batch,patch_map=patch,max_new_tokens=a.max_new_tokens)
                        sample_gen.append(dict(**common,condition=name,scale=alpha,prediction=pred or 'invalid',
                            correct=pred==m['gt'],text=text,n_patched=counts['patched']))
                # Commit only complete samples to paired results.
                old.append_jsonl(out/'samples.jsonl',dict(**common,generation=sample_gen,
                    sequence_scores=dict(zip(REL,map(float,scores))),candidate_p=p.tolist(),
                    frozen_middle_scores=mid_scores.tolist(),frozen_q={k:v.tolist() for k,v in qs.items()}))
                for row in sample_updates:old.append_jsonl(out/'updates.jsonl',row)
                generation.extend(sample_gen);updates.extend(sample_updates)
                metadata['complete_N']+=1
                old.write_json(out/'metadata.json',metadata)
                del batch,entries,scored,patch,C
            except Exception as exc:
                old.append_jsonl(out/'errors.jsonl',dict(sid=sid,error=f'{type(exc).__name__}: {exc}'))
                raise
            finally:
                if image is not None:image.close()
        metadata['status']='complete'
    except Exception:
        metadata['status']='failed_partial'
        raise
    finally:
        old.write_json(out/'metadata.json',metadata)
        if generation:reports(out,generation,updates)
        del model,processor,layers
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':main()
