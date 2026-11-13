#!/usr/bin/env python3
"""Paired random-direction control for Synthetic400 relation-wise COCO recovery.

Baseline (unchanged): eval_synthetic3d_relationwise_recovery_7models_3datasets_v1.py
Source: synthetic_shapes_4dir_400 (seed 7), with four independent source-fitted
        relation vectors for left/right/above/below at every layer; no H/V pairing.
Target: COCO-two, four relations left/right/above/below (= on/under semantics).

Reader: ALWAYS uses original Synthetic400 source-fitted four relation directions.
Actuator arms:
  - spatial: original unit direction d[target, layer] (one scalar per layer,
             SAME scalar for all 7 layers), original fixed scale search.
  - random:  a Haar-random subspace of the SAME RANK as the FOUR source
             relation-direction span at each layer (rank <= 3 after centering). All relation vectors are
             mapped into this subspace with their pairwise Gram matrix and
             unit lengths preserved. Only actuator geometry changes.
  - random_same_norm: random actuator at spatial arm's final accepted scale;
             provides an exactly equal final edit L2 norm, but not equal compute.

This script intentionally does NOT use the Synthetic400 H/V 2D controller,
and does NOT claim to run the old V6 minimum-norm QP optimization. It reuses
all generation, reader, steering, and search primitives from the genuine
Synthetic-3D relation-wise evaluator.

Saved per-sample JSONL permits resuming with the identical protocol and output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_NAME = 'eval_synthetic3d_relationwise_recovery_7models_3datasets_v1'
RELATIONS = ('left', 'right', 'above', 'below')


def load_source400(args):
    """Load the original generator's 400 records, mapping on/under -> above/below.

    Uses Synthetic400 images/labels only. Its Qwen source prompt has the same
    answer vocabulary as COCO: left / right / above / below.
    """
    from collections import Counter
    root = Path(args.synthetic_dir)
    labels = Path(args.synthetic_labels) if args.synthetic_labels else root / 'labels.jsonl'
    if not labels.is_file():
        raise FileNotFoundError(f'Synthetic400 annotations missing: {labels}')
    alias = {'left':'left', 'right':'right', 'on':'above', 'under':'below',
             'above':'above', 'below':'below'}
    rows=[]
    for ln, text in enumerate(labels.read_text(encoding='utf-8').splitlines(), 1):
        if not text.strip():
            continue
        item = json.loads(text)
        rel_raw = str(item['relation']).lower().strip()
        if rel_raw not in alias:
            raise ValueError(f'{labels}:{ln}: unexpected relation={rel_raw!r}')
        rel = alias[rel_raw]
        image = Path(str(item['image']))
        path = image if image.is_absolute() else root/image
        if not path.is_file():
            raise FileNotFoundError(f'Synthetic400 image missing: {path}')
        subject, reference = str(item['subject']).strip(), str(item['reference']).strip()
        rows.append({
            'sid': int(item.get('id', len(rows))), 'dataset': 'synthetic400',
            'image_path':str(path), 'subject':subject, 'reference':reference,
            'relation':rel,
            'question_text':(
                f'Where is the {subject} in relation to the {reference}? '
                'Answer with left, right, above, or below.'),
        })
    if args.source_max_samples is not None:
        rows=rows[:int(args.source_max_samples)]
    counts=Counter(row['relation'] for row in rows)
    if any(counts[r] < 1 for r in RELATIONS):
        raise ValueError(f'Missing source classes: {counts}')
    if args.source_max_samples is None and (len(rows)!=400 or any(counts[r]!=100 for r in RELATIONS)):
        raise ValueError(f'Expected original 400 rows, 100 per class; got {len(rows)} / {counts}')
    print(f'[Synthetic400] N={len(rows)} counts={dict(counts)}', flush=True)
    return rows

def verify_source_cache(cache_dir, source_records, layers, opt):
    """Prevent silently reusing a cache from Synthetic3D or different settings."""
    npz = cache_dir / 'synthetic3d_relation_directions.npz'  # upstream cache filename
    if not npz.is_file():
        return
    with np.load(npz, allow_pickle=True) as z:
        meta=json.loads(str(z['metadata_json'].item()))
    expected={
        'layers':list(map(int,layers)), 'relations':list(RELATIONS),
        'source_n':len(source_records),
        'synthetic_dir':str(Path(opt.synthetic_dir)),
        'gray_value':int(opt.gray_value),
    }
    inconsistent={key:(meta.get(key), val) for key,val in expected.items() if meta.get(key)!=val}
    if inconsistent:
        raise RuntimeError(
            f'Incompatible source cache {npz}: {inconsistent}. '
            'Use a fresh --source-cache-dir; never mix Synthetic3D and Synthetic400 caches.')


def make_random_directions(spatial: dict, seed: int):
    """Haar-randomize the entire relation span while preserving its Gram matrix.

    This randomizes spatial *orientation* without changing rank, direction norms,
    or pairwise relation-vector angles. It is more tightly controlled than
    independently sampling four vectors. No target GT or evaluation is used.

    For source directions M=D x 4, take thin SVD M=U diag(s) V^T, draw a
    random Haar Q in R^(D x rank), and return M_random=Q diag(s) V^T.
    """
    rng = np.random.default_rng(seed)
    random = {}
    arrays = {}
    checks = []
    for layer in sorted(spatial):
        old = np.column_stack([
            np.asarray(spatial[layer][rel], dtype=np.float64)
            for rel in RELATIONS
        ])
        assert old.ndim == 2 and old.shape[1] == len(RELATIONS)
        dim = old.shape[0]
        U, singular, Vt = np.linalg.svd(old, full_matrices=False)
        rank = int(np.sum(singular > max(1e-10, singular[0] * 1e-6)))
        assert rank > 0 and dim > rank, (layer, dim, rank)
        # Coordinates of all four original relation vectors in their source span.
        coords = singular[:rank, None] * Vt[:rank]
        gaussian = rng.standard_normal((dim, rank))
        haar_Q, _ = np.linalg.qr(gaussian, mode='reduced')
        moved = haar_Q @ coords
        gram_error = float(np.max(np.abs(old.T @ old - moved.T @ moved)))
        if gram_error > 5e-5:
            raise AssertionError(f'L{layer}: Gram mismatch {gram_error:.3g}')
        random[layer] = {}
        arrays[f'L{layer}_random_span_basis'] = haar_Q.astype(np.float32)
        for i, rel in enumerate(RELATIONS):
            original = old[:, i]
            q = moved[:, i].astype(np.float32)
            assert abs(np.linalg.norm(q)-np.linalg.norm(original)) < 1e-4
            random[layer][rel] = q
            arrays[f'L{layer}_{rel}'] = q
            checks.append({
                'layer': int(layer), 'relation': rel, 'hidden_size': dim,
                'relation_span_rank': rank,
                'spatial_norm': float(np.linalg.norm(original)),
                'random_norm': float(np.linalg.norm(q)),
                'spatial_random_cos': float(np.dot(original, q)),
                'max_gram_error_in_layer': gram_error,
            })
    return random, arrays, pd.DataFrame(checks)


def selftest():
    rng = np.random.default_rng(12)
    s = {
        l: {r: (lambda a: (a/np.linalg.norm(a)).astype(np.float32))(
            rng.normal(size=128)) for r in RELATIONS}
        for l in range(20, 27)
    }
    rd, arrays, checks = make_random_directions(s, 17)
    assert len(arrays) == 35  # 4 directions + one random span basis per layer
    assert len(checks) == 28
    assert all(abs(np.linalg.norm(rd[l][r])-1)<1e-5 for l in rd for r in RELATIONS)
    # At same alpha, per-layer relation difference norm and each token edit norm
    # are equal across arms, despite dissimilar directions.
    a = 3.0
    ns = math.sqrt(sum(2*(a*0.5*np.linalg.norm(s[l]['left']))**2 for l in s))
    nr = math.sqrt(sum(2*(a*0.5*np.linalg.norm(rd[l]['left']))**2 for l in rd))
    assert abs(ns-nr) < 1e-5
    print('[SELFTEST PASS] Haar random relation subspaces, Gram structure preserved, same-alpha norm match')


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--synthetic-dir', default='synthetic_shapes_4dir_400')
    ap.add_argument('--synthetic-labels', default=None)
    ap.add_argument('--source-max-samples', type=int, default=None)
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--coco-prompt-jsonl', default='prompts/COCO_QA_two_obj_with_answer_four_options.jsonl')
    ap.add_argument('--target-max-samples', type=int, default=None)
    ap.add_argument('--source-cache-dir', default='output/synthetic400_relationwise_source_cache')
    ap.add_argument('--output-dir', default='output/qwen3b_syn400_relationwise_random_v1')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--random-seed', type=int, default=20261113)
    ap.add_argument('--scales', default='0.25,0.5,1,2,4,8,16')
    ap.add_argument('--binary-refine', type=int, default=3)
    ap.add_argument('--max-new-tokens', type=int, default=8)
    ap.add_argument('--reference-layer-start', type=int, default=20)
    ap.add_argument('--reference-layer-end', type=int, default=26)
    ap.add_argument('--reference-n-layers', type=int, default=36)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--revision', default='main')
    ap.add_argument('--gray-value', type=int, default=128)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--fail-fast', action='store_true')
    return ap.parse_args()


def base_args(wrapper_args, base):
    """Use the original evaluator's argparse defaults and argument semantics."""
    argv = [BASE_NAME+'.py', '--models', 'qwen-3b', '--datasets', 'coco',
            '--synthetic-dir', wrapper_args.synthetic_dir,
            '--data-root', wrapper_args.data_root,
            '--coco-prompt-jsonl', wrapper_args.coco_prompt_jsonl,
            '--output-dir', wrapper_args.output_dir,
            '--seed', str(wrapper_args.seed),
            '--scales', wrapper_args.scales,
            '--binary-refine', str(wrapper_args.binary_refine),
            '--max-new-tokens', str(wrapper_args.max_new_tokens),
            '--reference-layer-start', str(wrapper_args.reference_layer_start),
            '--reference-layer-end', str(wrapper_args.reference_layer_end),
            '--reference-n-layers', str(wrapper_args.reference_n_layers),
            '--dtype', wrapper_args.dtype,
            '--device', wrapper_args.device,
            '--revision', wrapper_args.revision,
            '--gray-value', str(wrapper_args.gray_value)]
    for attr, flag in [('synthetic_labels','--synthetic-labels'),
                       ('source_max_samples','--source-max-samples'),
                       ('target_max_samples','--target-max-samples')]:
        val = getattr(wrapper_args, attr)
        if val is not None:
            argv.extend([flag, str(val)])
    if wrapper_args.fail_fast:
        argv.append('--fail-fast')
    old = sys.argv
    try:
        sys.argv = argv
        return base.parse_args()
    finally:
        sys.argv = old


def optimize_scalar(base, backend, image, record, readout, dataset, directions,
                    layers, scales, args):
    """Unmodified sequence of operations in original evaluate_dataset()."""
    target_rel = readout['source_pred']
    target_pred = readout['target_pred']
    attempts = 0
    last_fail = 0.0
    first_success = None
    best_text = None
    best_pred = None
    for scale in scales:
        text, pred = base.generate_with_relation_steer(
            backend, image, record, dataset, target_rel, float(scale),
            directions, layers, args)
        attempts += 1
        if pred == target_pred:
            first_success = float(scale)
            best_text, best_pred = text, pred
            break
        last_fail = float(scale)
    accepted = 0.0
    if first_success is not None:
        lo, hi = last_fail, first_success
        for _ in range(args.binary_refine):
            mid = 0.5*(lo+hi)
            text, pred = base.generate_with_relation_steer(
                backend, image, record, dataset, target_rel, mid,
                directions, layers, args)
            attempts += 1
            if pred == target_pred:
                hi = mid
                best_text, best_pred = text, pred
            else:
                lo = mid
        accepted = hi
    return {
        'pred': best_pred if first_success is not None else None,
        'text': best_text if first_success is not None else None,
        'alpha': accepted, 'success': first_success is not None,
        'generation_checks': attempts,
    }


def summarise(df):
    from collections import Counter
    base_correct = df['baseline_correct'].astype(bool).to_numpy()
    result = {
        'N': len(df),
        'baseline_accuracy': float(np.mean(base_correct)),
        'reader_accuracy': float(df['reader_correct'].astype(bool).mean()),
    }
    for arm in ('spatial','random','random_same_norm'):
        correct = df[f'{arm}_correct'].astype(bool).to_numpy()
        comply = df[f'{arm}_matches_reader'].astype(bool).to_numpy()
        result.update({
            f'{arm}_accuracy': float(np.mean(correct)),
            f'{arm}_W2C': int(np.sum(~base_correct & correct)),
            f'{arm}_C2W': int(np.sum(base_correct & ~correct)),
            f'{arm}_target_compliance': float(np.mean(comply)),
            f'{arm}_mean_alpha': float(df[f'{arm}_alpha'].mean()),
            f'{arm}_mean_edit_norm': float(df[f'{arm}_edit_norm'].mean()),
            f'{arm}_mean_gen_checks': float(df[f'{arm}_generation_checks'].mean()),
        })
    result['spatial_minus_random_accuracy'] = result['spatial_accuracy']-result['random_accuracy']
    result['random_same_norm_max_abs_norm_diff'] = float(
        np.max(np.abs(df['spatial_edit_norm']-df['random_same_norm_edit_norm'])))
    return pd.DataFrame([result])


def main():
    opt = parser()
    if opt.selftest:
        selftest()
        return
    # Delay loading the GPU stack until real execution (selftest needs only numpy/pandas).
    import importlib
    base = importlib.import_module(BASE_NAME)
    args = base_args(opt, base)
    # The upstream evaluator expects six Blender relations. Override its class
    # set with the four Synthetic400 relations before fitting the very same
    # relation-wise class means. This does NOT change the readout/steering code.
    base.SOURCE_RELATIONS = RELATIONS
    source_records = load_source400(args)
    target_records = base.load_target('coco', args)
    if not source_records or not target_records:
        raise RuntimeError('Source or target dataset is empty')
    print(f'[PREFLIGHT] Synthetic400={len(source_records)}, COCO-two={len(target_records)}', flush=True)
    if len(target_records) != 440 and opt.target_max_samples is None:
        raise RuntimeError(f'Expected all 440 COCO-two rows; found {len(target_records)}')
    if opt.preflight:
        print('[PREFLIGHT PASS] Data files and parsing are ready; no model loaded')
        return

    out = Path(opt.output_dir)
    cache_root = Path(opt.source_cache_dir)
    # Require a fresh source cache when source/data/hyperparameters change.
    protocol = {
        'script': Path(__file__).name,
        'baseline': BASE_NAME+'.py',
        'model': 'qwen-3b', 'dataset': 'coco',
        'synthetic_dir': str(Path(opt.synthetic_dir).resolve()),
        'synthetic_labels': str(Path(opt.synthetic_labels).resolve()) if opt.synthetic_labels else None,
        'source_max_samples': opt.source_max_samples,
        'target_max_samples': opt.target_max_samples,
        'seed': opt.seed, 'random_seed': opt.random_seed,
        'scales': base.parse_scales(opt.scales), 'binary_refine': opt.binary_refine,
        'max_new_tokens': opt.max_new_tokens,
        'reference_layers': [opt.reference_layer_start, opt.reference_layer_end, opt.reference_n_layers],
        'source_cache_dir': str(cache_root.resolve()),
        'source_relation_labels': list(RELATIONS),
        'source_extraction_prompt': 'Where is the <subject> in relation to the <reference>? Answer with left, right, above, or below.',
        'reader': 'ALWAYS original frozen Synthetic400 relation readout',
        'random_actuator': 'fixed Haar-random same-rank relation span preserving source direction Gram matrix',
        'same_norm_arm': 'random steering at spatial arm final accepted alpha',
        'optimizer': 'original relation-wise positive scalar-grid/binary refinement (NOT V6 QP)',
    }
    mf = out/'protocol.json'
    if mf.exists():
        existing = json.loads(mf.read_text())
        if existing != protocol:
            raise RuntimeError('Settings changed. Use a NEW --output-dir, do not mix runs.')
    out.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

    backend = base.load_backend('qwen-3b', args)
    try:
        layers = base.intervention_layers(len(backend.layers), args)
        if len(layers) != 7:
            raise RuntimeError(f'Expected 7 layers, got {layers}')
        print('[LAYERS]', layers, flush=True)
        cache_dir = cache_root/'qwen-3b'
        verify_source_cache(cache_dir, source_records, layers, opt)
        spatial = base.fit_relation_directions(backend, source_records, layers, args, cache_dir)
        random, arrays, checks = make_random_directions(spatial, opt.random_seed)
        np.savez_compressed(out/'random_relation_unit_vectors.npz', **arrays)
        checks.to_csv(out/'random_direction_checks.csv', index=False)
        print('[RANDOM] Generated frozen, Gram-matched Haar random relation subspaces', flush=True)

        per_sample = out/'paired_per_sample.jsonl'
        existing = []
        if per_sample.exists():
            with per_sample.open() as f:
                existing = [json.loads(line) for line in f if line.strip()]
        done = {int(row['sid']) for row in existing}
        errors = out/'errors.jsonl'
        scales = base.parse_scales(args.scales)
        from tqdm import tqdm
        for row in tqdm(target_records, desc='Synthetic400 spatial/random paired'):
            sid = int(row['sid'])
            if sid in done:
                continue
            img = None
            try:
                img = base.open_image(row)
                gt = base.canonical_gt('coco', str(row['relation']))
                question = base.target_question(row, 'coco')
                baseline_text = base.generate_text(backend, img, question, args)
                baseline_pred = base.parse_prediction(baseline_text, 'coco')
                reader = base.relation_readout(backend, img, row, 'coco', spatial, layers, args)
                target_pred = str(reader['target_pred'])
                src_target = str(reader['source_pred'])
                should_search = not (args.skip_if_baseline_agrees_selector and baseline_pred == target_pred)
                if should_search:
                    sp = optimize_scalar(base, backend, img, row, reader, 'coco', spatial, layers, scales, args)
                    rd = optimize_scalar(base, backend, img, row, reader, 'coco', random, layers, scales, args)
                else:
                    sp = {'pred': baseline_pred, 'text': baseline_text, 'alpha':0.0, 'success':True, 'generation_checks':0}
                    rd = dict(sp)
                # Original baseline semantics: if no acceptable scale found, retain baseline.
                for v in (sp, rd):
                    if v['pred'] is None:
                        v['pred'] = baseline_pred
                        v['text'] = baseline_text
                if sp['alpha'] > 0.0:
                    matched_text, matched_pred = base.generate_with_relation_steer(
                        backend, img, row, 'coco', src_target, float(sp['alpha']), random, layers, args)
                    fixed = {'pred': matched_pred, 'text': matched_text, 'alpha':float(sp['alpha']),
                             'success': matched_pred == target_pred, 'generation_checks':1}
                else:
                    fixed = {'pred': baseline_pred, 'text': baseline_text, 'alpha':0.0,
                             'success': baseline_pred==target_pred, 'generation_checks':0}

                result = {
                    'sid': sid, 'gt': gt, 'baseline_pred':baseline_pred,
                    'baseline_correct': int(baseline_pred==gt),
                    'reader_pred':target_pred, 'reader_source_relation':src_target,
                    'reader_correct':int(target_pred==gt),
                    'readout_scores': reader['scores'],
                }
                for name, v in [('spatial',sp), ('random',rd), ('random_same_norm',fixed)]:
                    alpha = float(v['alpha'])
                    # 7 layers, 1 subject and 1 reference token: each gets +/-alpha*d/2.
                    # All vectors unit norm, hence edit norm=sqrt(L/2)*abs(alpha).
                    result.update({
                        f'{name}_pred':v['pred'],
                        f'{name}_correct':int(v['pred']==gt),
                        f'{name}_matches_reader':int(v['pred']==target_pred),
                        f'{name}_alpha':alpha,
                        f'{name}_edit_norm':float(abs(alpha)*math.sqrt(len(layers)/2)),
                        f'{name}_generation_checks':int(v['generation_checks']),
                        f'{name}_search_succeeded':int(v['success']),
                        f'{name}_text':v['text'],
                    })
                with per_sample.open('a', encoding='utf8') as f:
                    f.write(json.dumps(result, ensure_ascii=False)+'\n')
                done.add(sid)
                tqdm.write(f'sid={sid} baseline={baseline_pred} reader={target_pred} spatial={sp["pred"]} random={rd["pred"]} alpha={sp["alpha"]:.4g}/{rd["alpha"]:.4g}')
            except Exception as exc:
                with errors.open('a', encoding='utf8') as f:
                    f.write(json.dumps({'sid':sid,'error':repr(exc),'traceback':traceback.format_exc()}, ensure_ascii=False)+'\n')
                if opt.fail_fast:
                    raise
                print('[ERROR]',sid, repr(exc),flush=True)
            finally:
                if img is not None:
                    img.close()
                base.cleanup()
        all_rows = [json.loads(line) for line in per_sample.read_text().splitlines() if line.strip()] if per_sample.exists() else []
        if not all_rows:
            raise RuntimeError('No successful paired samples')
        df = pd.DataFrame(all_rows).sort_values('sid')
        df.to_csv(out/'paired_per_sample.csv', index=False)
        sm = summarise(df)
        sm.to_csv(out/'paired_summary.csv', index=False)
        print('\n'+sm.T.to_string(header=False),flush=True)
        if len(df) != len(target_records):
            print(f'[CAUTION] Only {len(df)}/{len(target_records)} rows complete; inspect errors.jsonl',flush=True)
        print('[FILES]',out/'paired_summary.csv',out/'paired_per_sample.csv',flush=True)
    finally:
        backend.close()
        base.cleanup()


if __name__ == '__main__':
    main()
