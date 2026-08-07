#!/usr/bin/env python3
from pathlib import Path
import csv
import argparse

p = argparse.ArgumentParser()
p.add_argument('--root', default='output/coco_correct_noimage_v2')
args = p.parse_args()
root = Path(args.root)

models = [
    'qwen2-2b', 'qwen2-vl-7b', 'qwen-3b', 'qwen-7b',
    'llava-7b', 'llava-13b',
]

print(f"{'Model':16s} {'Correct':>11s} {'NoImage':>11s} {'Residual':>11s} {'ResLayer':>9s} {'N':>6s}")
print('-' * 67)
for model in models:
    f = root / model / 'best_results.tsv'
    if not f.exists():
        print(f'{model:16s} MISSING')
        continue
    with f.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    by = {r['condition']: r for r in rows}
    c = by.get('raw__correct')
    n = by.get('raw__no_image')
    r = by.get('raw__correct_minus_noimage')
    if not (c and n and r):
        print(f'{model:16s} INCOMPLETE')
        continue
    print(
        f"{model:16s} "
        f"{100*float(c['accuracy_mean']):10.2f}% "
        f"{100*float(n['accuracy_mean']):10.2f}% "
        f"{100*float(r['accuracy_mean']):10.2f}% "
        f"L{r['best_layer']:>7s} "
        f"{r['n']:>6s}"
    )
