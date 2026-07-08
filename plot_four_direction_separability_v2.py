#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot four independent relation-direction separability.

V2 is designed to match analyze_controlledA_relation_direction_probe.py for Controlled A:
  - prefer layer_<L>_subject/reference over relation_vectors when both exist
  - use subject_name/reference_name to form unordered pair groups
  - use the same greedy balanced pair-group CV splitter
  - classify by cosine to four train-fold relation directions

It also saves 4D score PCA/t-SNE, mean score heatmap, confusion matrix, and optional centered-raw plots.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

EPS = 1e-12


@dataclass(frozen=True)
class Row:
    index: int
    sid: str
    relation: str
    subject: str
    reference: str

    @property
    def pair_group(self) -> str:
        return " || ".join(sorted((self.subject, self.reference)))


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
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--also-pca2", action="store_true")
    p.add_argument("--no-raw-tsne", action="store_true")
    p.add_argument("--no-score-tsne", action="store_true")
    p.add_argument("--max-per-label", type=int, default=0)
    return p.parse_args()


def str_arr(x):
    return np.asarray([str(v) for v in np.asarray(x, dtype=object).tolist()], dtype=object)


def parse_layers(text: str) -> List[int]:
    if text.lower() in {"auto", "all"}:
        return []
    return sorted({int(t.strip().lstrip("L")) for t in text.split(",") if t.strip()})


def available_layers(keys) -> List[int]:
    out = []
    pat = re.compile(r"^layer_(\d+)_subject$")
    for k in keys:
        m = pat.match(k)
        if m and f"layer_{m.group(1)}_reference" in keys:
            out.append(int(m.group(1)))
    return sorted(out)


def load_layer(npz_path: Path, layer: int, relation_order: List[str]):
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        if f"layer_{layer}_subject" in keys and f"layer_{layer}_reference" in keys:
            subj_state = np.asarray(z[f"layer_{layer}_subject"], dtype=np.float64)
            ref_state = np.asarray(z[f"layer_{layer}_reference"], dtype=np.float64)
            X_all = subj_state - ref_state
        elif "relation_vectors" in keys:
            blocks = [int(v) for v in z["decoder_block_index"].tolist()]
            if layer not in blocks:
                raise RuntimeError(f"Layer {layer} not in relation_vectors blocks={blocks}")
            li = blocks.index(layer)
            X_all = np.asarray(z["relation_vectors"][:, li, :], dtype=np.float64)
        else:
            raise RuntimeError(f"No layer subject/reference or relation_vectors found in {npz_path}")

        labels_all = str_arr(z["relation"])
        n = len(labels_all)
        sid = str_arr(z["sid"]) if "sid" in keys else np.asarray([str(i) for i in range(n)], dtype=object)
        subject = str_arr(z["subject_name"]) if "subject_name" in keys else (str_arr(z["subject"]) if "subject" in keys else np.asarray(["NA"] * n, dtype=object))
        reference = str_arr(z["reference_name"]) if "reference_name" in keys else (str_arr(z["reference"]) if "reference" in keys else np.asarray(["NA"] * n, dtype=object))

    rows = []
    keep_idx = []
    allowed = set(relation_order)
    for i, lab in enumerate(labels_all.tolist()):
        lab = str(lab).strip().lower()
        if lab not in allowed:
            continue
        rows.append(Row(i, str(sid[i]), lab, str(subject[i]), str(reference[i])))
        keep_idx.append(i)
    X = X_all[np.asarray(keep_idx, dtype=np.int64)]
    return rows, X


def downsample(rows, X, relation_order, max_per_label, seed):
    if not max_per_label or max_per_label <= 0:
        return rows, X
    rng = np.random.default_rng(seed)
    chosen = []
    labels = np.asarray([r.relation for r in rows], dtype=object)
    for lab in relation_order:
        idx = np.where(labels == lab)[0]
        if len(idx) > max_per_label:
            idx = rng.choice(idx, size=max_per_label, replace=False)
        chosen.extend(idx.tolist())
    chosen = sorted(chosen)
    return [rows[i] for i in chosen], X[np.asarray(chosen, dtype=np.int64)]


def grouped_folds_controlled(rows: Sequence[Row], relation_order: List[str], n_folds: int, split_unit: str, seed: int):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if split_unit == "pair":
            key = row.pair_group
        elif split_unit == "sample":
            key = row.sid
        else:
            key = row.sid
        groups[key].append(i)
    if len(groups) < n_folds:
        raise RuntimeError(f"Need >= {n_folds} groups; got {len(groups)}")

    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    totals = Counter(r.relation for r in rows)
    wanted = {lab: totals[lab] / n_folds for lab in relation_order}
    wanted_size = len(rows) / n_folds
    folds = [[] for _ in range(n_folds)]
    fold_counts = [Counter() for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]

    for _, idxs in items:
        gc = Counter(rows[i].relation for i in idxs)
        def score(f):
            rel_err = 0.0
            size_err = 0.0
            for cand in range(n_folds):
                sz = fold_sizes[cand] + (len(idxs) if cand == f else 0)
                size_err += abs(sz - wanted_size)
                for lab in relation_order:
                    cnt = fold_counts[cand][lab] + (gc[lab] if cand == f else 0)
                    rel_err += abs(cnt - wanted[lab])
            return rel_err, size_err, fold_sizes[f]
        target = min(range(n_folds), key=score)
        folds[target].extend(idxs)
        fold_counts[target].update(gc)
        fold_sizes[target] += len(idxs)

    out = []
    all_idx = set(range(len(rows)))
    for f, te in enumerate(folds):
        counts = Counter(rows[i].relation for i in te)
        miss = [lab for lab in relation_order if counts[lab] == 0]
        if miss:
            raise RuntimeError(f"Fold {f} missing labels {miss}; reduce folds")
        te = sorted(te)
        tr = sorted(all_idx - set(te))
        out.append((np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)))
    return out


def grouped_folds_sklearn(X, y, groups, cv_folds, seed):
    if StratifiedGroupKFold is not None:
        return list(StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(X, y, groups))
    return list(GroupKFold(n_splits=cv_folds).split(X, y, groups))


def norm_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def fit_codebook(Xtr, ytr, relation_order, feature_centering):
    if feature_centering == "global":
        center = Xtr.mean(axis=0, keepdims=True)
    else:
        center = np.zeros((1, Xtr.shape[1]), dtype=np.float64)
    Xc = Xtr - center
    dirs = []
    means = []
    for lab in relation_order:
        m = ytr == lab
        if not np.any(m):
            raise RuntimeError(f"Train split missing {lab}")
        v = Xc[m].mean(axis=0)
        means.append(v)
        dirs.append(v / max(float(np.linalg.norm(v)), EPS))
    return center.reshape(-1), np.stack(dirs), np.stack(means)


def cv_scores(rows, X, relation_order, cv_folds, split_unit, seed, feature_centering):
    y = np.asarray([r.relation for r in rows], dtype=object)
    label_to_idx = {lab: i for i, lab in enumerate(relation_order)}
    yi = np.asarray([label_to_idx[v] for v in y], dtype=np.int64)
    if split_unit == "pair" and all(r.subject != "NA" and r.reference != "NA" for r in rows):
        splits = grouped_folds_controlled(rows, relation_order, cv_folds, split_unit, seed)
    else:
        groups = np.asarray([r.sid if split_unit != "pair" else r.pair_group for r in rows], dtype=object)
        splits = grouped_folds_sklearn(X, y, groups, cv_folds, seed)

    C = len(relation_order)
    scores = np.zeros((len(rows), C), dtype=np.float64)
    pred = np.zeros(len(rows), dtype=np.int64)
    Xc_plot = np.zeros_like(X, dtype=np.float64)
    fold_id = np.zeros(len(rows), dtype=np.int64)
    dirs_by_fold = []

    for fi, (tr, te) in enumerate(splits):
        center, D, means = fit_codebook(X[tr], y[tr], relation_order, feature_centering)
        Xt = X[te] - center[None, :]
        sc = norm_rows(Xt) @ D.T
        scores[te] = sc
        pred[te] = np.argmax(sc, axis=1)
        Xc_plot[te] = Xt
        fold_id[te] = fi
        dirs_by_fold.append(D)

    correct = pred == yi
    tmp = scores.copy()
    true_score = scores[np.arange(len(rows)), yi]
    tmp[np.arange(len(rows)), yi] = -np.inf
    signed_margin = true_score - tmp.max(axis=1)
    top_margin = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    return scores, pred, yi, correct, signed_margin, top_margin, Xc_plot, fold_id


def run_tsne(X, perplexity, seed):
    perp = min(float(perplexity), max(5.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(X)


def scatter(Y, labels, relation_order, title, out_png, correct=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    for lab in relation_order:
        m = labels == lab
        ax.scatter(Y[m, 0], Y[m, 1], s=20, alpha=0.76, label=f"{lab} (n={int(m.sum())})")
    if correct is not None and np.any(~correct):
        bad = ~correct
        ax.scatter(Y[bad, 0], Y[bad, 1], s=48, facecolors="none", edgecolors="black", linewidths=0.7, label="wrong")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(fontsize=8, loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def heatmap(M, xlabels, ylabels, title, out_png, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=180)
    im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_csv(path, Y, rows, pred, relation_order, correct, signed_margin, top_margin):
    with open(path, "w", encoding="utf-8") as f:
        f.write("sid,subject,reference,relation,pred,correct,signed_margin,top1_margin,x,y\n")
        for row, pi, ok, sm, tm, xy in zip(rows, pred, correct, signed_margin, top_margin, Y):
            f.write(f"{row.sid},{row.subject},{row.reference},{row.relation},{relation_order[int(pi)]},{int(bool(ok))},{float(sm):.8g},{float(tm):.8g},{float(xy[0]):.8g},{float(xy[1]):.8g}\n")


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

    summary = {"input_npz": str(inp), "dataset": args.dataset_name, "model": args.model_name, "relations": relation_order, "cv_folds": args.cv_folds, "split_unit": args.split_unit, "layers": {}}

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
        print(f"L{L}: n={len(rows)}, pairs={pairs}, objects={objects}, counts={dict(counts)} acc={acc:.3f} mean_top1_margin={mean_margin:.3f}")

        prefix = f"four_dir_L{L}_{'_'.join(relation_order)}_cv{args.cv_folds}_{args.split_unit}"
        files = {}

        M = np.zeros((len(relation_order), len(relation_order)), dtype=np.float64)
        for i, lab in enumerate(relation_order):
            M[i] = scores[labels == lab].mean(axis=0)
        p = out_dir / f"{prefix}_mean_score_heatmap.png"
        heatmap(M, [f"dir:{x}" for x in relation_order], [f"true:{x}" for x in relation_order], f"{args.dataset_name}/{args.model_name}/L{L} mean cosine scores", p, vmin=-1, vmax=1)
        files["mean_score_heatmap"] = str(p)

        cm = confusion_matrix(yi, pred, labels=np.arange(len(relation_order))).astype(np.float64)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        p = out_dir / f"{prefix}_confusion_norm.png"
        heatmap(cmn, relation_order, relation_order, f"{args.dataset_name}/{args.model_name}/L{L} confusion acc={acc:.3f}", p, vmin=0, vmax=1)
        files["confusion_norm"] = str(p)

        if args.also_pca2:
            Y = PCA(n_components=2, random_state=args.seed).fit_transform(scores)
            p = out_dir / f"{prefix}_score4d_pca2.png"
            scatter(Y, labels, relation_order, f"{args.dataset_name}/{args.model_name}/L{L} 4D score PCA2 acc={acc:.3f}", p, correct)
            save_csv(out_dir / f"{prefix}_score4d_pca2.csv", Y, rows, pred, relation_order, correct, signed_margin, top_margin)
            files["score4d_pca2"] = str(p)

            Y = PCA(n_components=2, random_state=args.seed).fit_transform(Xc)
            p = out_dir / f"{prefix}_centered_raw_pca2.png"
            scatter(Y, labels, relation_order, f"{args.dataset_name}/{args.model_name}/L{L} centered raw PCA2 acc={acc:.3f}", p, correct)
            save_csv(out_dir / f"{prefix}_centered_raw_pca2.csv", Y, rows, pred, relation_order, correct, signed_margin, top_margin)
            files["centered_raw_pca2"] = str(p)

        if not args.no_score_tsne:
            Y = run_tsne(scores, args.perplexity, args.seed)
            p = out_dir / f"{prefix}_score4d_tsne_perp{int(args.perplexity)}.png"
            scatter(Y, labels, relation_order, f"{args.dataset_name}/{args.model_name}/L{L} 4D score t-SNE acc={acc:.3f}", p, correct)
            save_csv(out_dir / f"{prefix}_score4d_tsne.csv", Y, rows, pred, relation_order, correct, signed_margin, top_margin)
            files["score4d_tsne"] = str(p)

        if not args.no_raw_tsne:
            Xvis = Xc
            prep = "raw"
            if args.pca_dim and args.pca_dim > 0 and Xvis.shape[1] > args.pca_dim and len(Xvis) > args.pca_dim + 2:
                Xvis = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(Xvis)
                prep = f"pca{args.pca_dim}"
            Y = run_tsne(Xvis, args.perplexity, args.seed)
            p = out_dir / f"{prefix}_centered_raw_{prep}_tsne_perp{int(args.perplexity)}.png"
            scatter(Y, labels, relation_order, f"{args.dataset_name}/{args.model_name}/L{L} centered raw {prep} t-SNE acc={acc:.3f}", p, correct)
            save_csv(out_dir / f"{prefix}_centered_raw_tsne.csv", Y, rows, pred, relation_order, correct, signed_margin, top_margin)
            files["centered_raw_tsne"] = str(p)

        np.savez_compressed(
            out_dir / f"{prefix}_cv_scores.npz",
            scores=scores.astype(np.float32),
            labels=labels.astype(str),
            sid=np.asarray([r.sid for r in rows], dtype=object),
            subject=np.asarray([r.subject for r in rows], dtype=object),
            reference=np.asarray([r.reference for r in rows], dtype=object),
            pred=np.asarray([relation_order[i] for i in pred], dtype=object),
            correct=correct,
            signed_margin=signed_margin.astype(np.float32),
            top1_margin=top_margin.astype(np.float32),
            fold_id=fold_id,
            relation_order=np.asarray(relation_order, dtype=object),
        )
        files["cv_scores_npz"] = str(out_dir / f"{prefix}_cv_scores.npz")
        summary["layers"][str(L)] = {"n": len(rows), "pairs": pairs, "objects": objects, "counts": dict(counts), "acc": acc, "mean_top1_margin": mean_margin, "files": files}

    out_json = out_dir / "four_direction_separability_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
