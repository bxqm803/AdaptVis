#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot only cross-validated 4D relation-score t-SNE.

This matches plot_four_direction_separability_v2.py / analyze_controlledA_relation_direction_probe.py:
  r = layer_L_subject - layer_L_reference when available
  train-fold global centering
  one cosine direction per relation
  test-fold scores = cosine(r - train_center, d_relation)
  visualization input = [score_relation_1, ..., score_relation_K]

Only saves score4d t-SNE PNG/CSV and a small summary JSON.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from plot_four_direction_separability_v2 import (
    available_layers,
    parse_layers,
    load_layer,
    downsample,
    cv_scores,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="controlled_A")
    p.add_argument("--model-name", default="llava-1.5-7b")
    p.add_argument("--layers", default="13,16,31")
    p.add_argument("--relations", default="left,right,on,under")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--split-unit", choices=("pair", "sample", "id"), default="pair")
    p.add_argument("--feature-centering", choices=("global", "none"), default="global")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max-per-label", type=int, default=0)
    return p.parse_args()


def run_tsne(X, perplexity, seed):
    perp = min(float(perplexity), max(5.0, (len(X) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X)


def scatter(Y, labels, relation_order, title, out_png, correct=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    for lab in relation_order:
        m = labels == lab
        ax.scatter(Y[m, 0], Y[m, 1], s=20, alpha=0.76, label=f"{lab} (n={int(m.sum())})")
    if correct is not None and np.any(~correct):
        bad = ~correct
        ax.scatter(
            Y[bad, 0], Y[bad, 1],
            s=48, facecolors="none", edgecolors="black", linewidths=0.7, label="wrong"
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE dim-1")
    ax.set_ylabel("t-SNE dim-2")
    ax.legend(fontsize=8, loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_csv(path, Y, rows, pred, relation_order, correct, signed_margin, top_margin, scores):
    score_cols = [f"score_{lab}" for lab in relation_order]
    with open(path, "w", encoding="utf-8") as f:
        f.write("sid,subject,reference,relation,pred,correct,signed_margin,top1_margin,x,y," + ",".join(score_cols) + "\n")
        for k, (row, pi, ok, sm, tm, xy) in enumerate(zip(rows, pred, correct, signed_margin, top_margin, Y)):
            score_str = ",".join(f"{float(v):.8g}" for v in scores[k])
            f.write(
                f"{row.sid},{row.subject},{row.reference},{row.relation},"
                f"{relation_order[int(pi)]},{int(bool(ok))},{float(sm):.8g},{float(tm):.8g},"
                f"{float(xy[0]):.8g},{float(xy[1]):.8g},{score_str}\n"
            )


def main():
    args = parse_args()
    inp = Path(args.input_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relation_order = [x.strip().lower() for x in args.relations.split(",") if x.strip()]
    with np.load(inp, allow_pickle=True) as z:
        avail = available_layers(set(z.files))
        if not avail and "relation_vectors" in z.files:
            avail = [int(v) for v in z["decoder_block_index"].tolist()]
    layers = parse_layers(args.layers) or avail

    print(f"Input: {inp}")
    print(f"Dataset/model: {args.dataset_name} / {args.model_name}")
    print(f"Available layers: {avail}")
    print(f"Selected layers: {layers}")
    print(f"Relations: {relation_order}")
    print(f"CV: {args.cv_folds}-fold split_unit={args.split_unit} centering={args.feature_centering}")
    print("Output: score4d t-SNE only")

    summary = {
        "input_npz": str(inp),
        "dataset": args.dataset_name,
        "model": args.model_name,
        "relations": relation_order,
        "cv_folds": args.cv_folds,
        "split_unit": args.split_unit,
        "feature_centering": args.feature_centering,
        "layers": {},
    }

    for L in layers:
        rows, X = load_layer(inp, L, relation_order)
        rows, X = downsample(rows, X, relation_order, args.max_per_label, args.seed)
        labels = np.asarray([r.relation for r in rows], dtype=object)
        counts = Counter(labels.tolist())
        pairs = len({r.pair_group for r in rows})
        objects = len({r.subject for r in rows} | {r.reference for r in rows})

        scores, pred, yi, correct, signed_margin, top_margin, Xc, fold_id = cv_scores(
            rows, X, relation_order, args.cv_folds, args.split_unit, args.seed, args.feature_centering
        )
        acc = float(correct.mean())
        mean_margin = float(top_margin.mean())
        print(
            f"L{L}: n={len(rows)}, pairs={pairs}, objects={objects}, "
            f"counts={dict(counts)} acc={acc:.3f} mean_top1_margin={mean_margin:.3f}"
        )

        Y = run_tsne(scores, args.perplexity, args.seed)
        prefix = f"score4d_tsne_L{L}_{'_'.join(relation_order)}_cv{args.cv_folds}_{args.split_unit}_perp{int(args.perplexity)}"
        out_png = out_dir / f"{prefix}.png"
        scatter(
            Y, labels, relation_order,
            f"{args.dataset_name}/{args.model_name}/L{L} 4D score t-SNE acc={acc:.3f}",
            out_png, correct,
        )
        out_csv = out_dir / f"{prefix}.csv"
        save_csv(out_csv, Y, rows, pred, relation_order, correct, signed_margin, top_margin, scores)

        summary["layers"][str(L)] = {
            "n": len(rows),
            "pairs": pairs,
            "objects": objects,
            "counts": dict(counts),
            "acc": acc,
            "mean_top1_margin": mean_margin,
            "score4d_tsne_png": str(out_png),
            "score4d_tsne_csv": str(out_csv),
        }
        print(f"  saved {out_png}")

    out_json = out_dir / "score4d_tsne_only_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
