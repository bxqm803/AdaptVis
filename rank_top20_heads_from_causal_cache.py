#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-free provisional head ranking from v3 pass2_causal checkpoints.

Example:
 python rank_top20_heads_from_causal_cache.py \
   --run-dir output/head_qwen3b_all176_all440 \
   --calib-n 80 --top-k 20 --seed 20260912

This only reads stored head interventions; it never uses relations or accuracy.
The rank is provisional until tested on independent samples.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--calib-n', type=int, default=80, help='Number of completed samples to use (0=all).')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--out-dir', type=Path, default=None)
    return p.parse_args()


def rank(paths, names):
    sum_sq = np.zeros((len(names), 2), dtype=np.float64)
    n = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            scores = np.asarray(data['scores'], dtype=np.float64)
        if scores.shape != (len(names), 2) or not np.isfinite(scores).all():
            raise ValueError(f'Bad cached data: {path}; shape={scores.shape}')
        sum_sq += np.square(scores)
        n += 1
    if n == 0:
        raise ValueError('No completed causal samples')
    vals = np.sqrt(sum_sq.sum(axis=1) / n)
    hvals = np.sqrt(sum_sq[:, 0] / n)
    vvals = np.sqrt(sum_sq[:, 1] / n)
    order = sorted(range(len(names)), key=lambda i: (-float(vals[i]), names[i]))
    rows = [dict(rank=j+1, head=names[i], causal_rms=float(vals[i]),
                 H_rms=float(hvals[i]), V_rms=float(vvals[i]), N=n)
            for j, i in enumerate(order)]
    return rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args = get_args()
    manifest_path = args.run_dir / 'causal_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing {manifest_path}; use v3 all176 run dir')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = list(manifest['head_order'])
    if not names or len(names) != len(set(names)):
        raise RuntimeError('Empty/duplicate head_order in manifest')
    completed = sorted((args.run_dir / 'pass2_causal').glob('*.npz'))
    # Only include expected sample IDs, as listed in the original run manifest.
    eligible = {int(sid) for sid in manifest['sids']}
    completed = [p for p in completed if int(p.stem) in eligible]
    if not completed:
        raise RuntimeError('No completed pass2_causal/*.npz files')
    if args.calib_n < 0 or args.top_k < 1 or args.top_k > len(names):
        raise ValueError('calib-n must be >=0 and top-k must be in [1, num_heads]')
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(completed, min(args.calib_n, len(completed)))) if args.calib_n else completed
    calib_rows = rank(chosen, names)
    all_rows = rank(completed, names)
    selected_paths = set(chosen)
    holdout_paths = [p for p in completed if p not in selected_paths]
    holdout_rows = rank(holdout_paths, names) if holdout_paths else None
    out = args.out_dir or args.run_dir / f'gtfree_shortlist_N{len(chosen)}'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'rank_calibration_all_heads.csv', calib_rows)
    write_csv(out / 'top20_calibration.csv', calib_rows[:args.top_k])
    write_csv(out / 'rank_all_completed_for_stability.csv', all_rows)
    if holdout_rows is not None:
        write_csv(out / 'rank_completed_holdout_heads.csv', holdout_rows)
    top_calib = {r['head'] for r in calib_rows[:args.top_k]}
    top_all = {r['head'] for r in all_rows[:args.top_k]}
    top_holdout = {r['head'] for r in holdout_rows[:args.top_k]} if holdout_rows else set()
    details = dict(run_dir=str(args.run_dir), readout=manifest.get('readout_block_input'),
                   num_heads=len(names), all_completed_N=len(completed),
                   calibration_N=len(chosen), completed_holdout_N=len(holdout_paths), seed=args.seed, top_k=args.top_k,
                   selected_heads=[r['head'] for r in calib_rows[:args.top_k]],
                   selected_sample_sids=[int(p.stem) for p in chosen],
                   top_k_overlap_with_all_completed=len(top_calib & top_all),
                   top_k_overlap_with_disjoint_completed_holdout=len(top_calib & top_holdout),
                   selection_uses_coco_GT=False,
                   score_formula='sqrt(mean_i(H_effect_i^2 + V_effect_i^2))',
                   caveat='All-completed stability is not an independent holdout if it includes calibration samples.')
    (out / 'selection_protocol.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'Existing checkpoints: {len(completed)} | calibration: {len(chosen)} | candidate heads: {len(names)}')
    print(' RANK  HEAD     RMS       H-RMS     V-RMS')
    for r in calib_rows[:args.top_k]:
        print(f" {r['rank']:>4}  {r['head']:<7} {r['causal_rms']:.5f}  {r['H_rms']:.5f}   {r['V_rms']:.5f}")
    print(f'Top-{args.top_k} overlap with all {len(completed)} completed samples: {len(top_calib & top_all)}/{args.top_k}')
    if holdout_rows is not None:
        print(f'Top-{args.top_k} overlap with {len(holdout_paths)} disjoint completed holdout: {len(top_calib & top_holdout)}/{args.top_k}')
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
