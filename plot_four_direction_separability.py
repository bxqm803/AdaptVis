#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize four independent relation directions.

This script reproduces the "four independent relation directions" probe:
  r_i = h(subject_i) - h(reference_i)
  g = mean_train(r)
  d_c = normalize(mean_train[y=c](r - g))
  score_c(i) = cosine(r_i - g, d_c)
  pred = argmax_c score_c(i)

It then plots:
  1) t-SNE/PCA of the high-dimensional centered relation features used by the probe.
  2) t-SNE/PCA of the 4D cosine-score vectors [score_left, score_right, score_on, score_under].
  3) mean score heatmap: true relation x direction score.
  4) confusion matrix.

Supported NPZ formats:
  - relation_vectors [N,L,H], decoder_block_index, relation, image_id/sid
  - layer_<L>_subject [N,H], layer_<L>_reference [N,H], relation, sid/image_id
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("matplotlib is required: pip install -U matplotlib") from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception:
        StratifiedGroupKFold = None
except Exception as exc:
    raise SystemExit("scikit-learn is required: pip install -U scikit-learn") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--layers", default="auto", help="Comma-separated layers, or auto/all")
    p.add_argument("--relations", default="left,right,on,under", help="Comma-separated label order")
    p.add_argument("--cv-folds", type=int, default=5, help="0/1 = fit on all; >1 = group CV")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pca-dim", type=int, default=50, help="PCA dim before raw-feature t-SNE; 0 disables")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max-per-label", type=int, default=0, help="0 means no downsampling")
    p.add_argument("--no-raw-tsne", action="store_true")
    p.add_argument("--no-score-tsne", action="store_true")
    p.add_argument("--also-pca2", action="store_true", help="Also save PCA2 scatters for raw features and 4D scores")
    return p.parse_args()


def as_str_array(x: np.ndarray) -> np.ndarray:
    return np.asarray([str(v) for v in x.tolist()], dtype=object)


def load_npz(path: Path) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        metadata = {}
        if "metadata_json" in keys:
            try:
                metadata = json.loads(str(z["metadata_json"].item()))
            except Exception:
                metadata = {}

        if "relation_vectors" in keys:
            labels = as_str_array(z["relation"])
            if "image_id" in keys:
                ids = as_str_array(z["image_id"])
            elif "sid" in keys:
                ids = as_str_array(z["sid"])
            else:
                ids = np.asarray([str(i) for i in range(len(labels))], dtype=object)
            vecs = z["relation_vectors"].astype(np.float32)  # [N,L,H]
            blocks = [int(v) for v in z["decoder_block_index"].tolist()]
            layer_map = {block: vecs[:, li, :] for li, block in enumerate(blocks)}
            return layer_map, labels, ids, metadata

        layer_pat = re.compile(r"^layer_(\d+)_subject$")
        layer_ids = []
        for k in keys:
            m = layer_pat.match(k)
            if m and f"layer_{m.group(1)}_reference" in keys:
                layer_ids.append(int(m.group(1)))
        if layer_ids:
            labels = as_str_array(z["relation"])
            if "sid" in keys:
                ids = as_str_array(z["sid"])
            elif "image_id" in keys:
                ids = as_str_array(z["image_id"])
            else:
                ids = np.asarray([str(i) for i in range(len(labels))], dtype=object)
            layer_map = {}
            for L in sorted(layer_ids):
                subj = z[f"layer_{L}_subject"].astype(np.float32)
                ref = z[f"layer_{L}_reference"].astype(np.float32)
                layer_map[L] = subj - ref
            return layer_map, labels, ids, metadata

        raise RuntimeError(
            "Unrecognized NPZ format. Expected relation_vectors or layer_<L>_subject/reference.\n"
            f"Available keys: {sorted(z.files)[:120]}"
        )


def choose_layers(layer_map: Dict[int, np.ndarray], spec: str) -> List[int]:
    available = sorted(layer_map.keys())
    if spec.lower() in {"auto", "all"}:
        return available
    wanted = [int(x.strip().lstrip("L")) for x in spec.split(",") if x.strip()]
    missing = [x for x in wanted if x not in layer_map]
    if missing:
        raise RuntimeError(f"Requested layers {missing} not found. Available: {available}")
    return wanted


def filter_labels(X, labels, ids, relation_order, max_per_label, seed):
    keep = np.isin(labels, relation_order)
    X, labels, ids = X[keep], labels[keep], ids[keep]
    if max_per_label and max_per_label > 0:
        rng = np.random.default_rng(seed)
        chosen = []
        for lab in relation_order:
            idx = np.where(labels == lab)[0]
            if len(idx) > max_per_label:
                idx = rng.choice(idx, size=max_per_label, replace=False)
            chosen.extend(idx.tolist())
        chosen = np.asarray(sorted(chosen), dtype=np.int64)
        X, labels, ids = X[chosen], labels[chosen], ids[chosen]
    return X.astype(np.float32), labels, ids


def get_splits(X, labels, groups, cv_folds, seed):
    if cv_folds <= 1:
        idx = np.arange(len(labels))
        return [(idx, idx)]
    n_groups = len(set(groups.tolist()))
    if n_groups < cv_folds:
        raise RuntimeError(f"Need at least cv_folds unique groups. groups={n_groups}, cv_folds={cv_folds}")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        return list(splitter.split(X, labels, groups))
    splitter = GroupKFold(n_splits=cv_folds)
    return list(splitter.split(X, labels, groups))


def l2_normalize(X: np.ndarray, axis: int = 1) -> np.ndarray:
    denom = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(denom, 1e-8)


def fit_four_directions(X_train: np.ndarray, y_train: np.ndarray, relation_order: List[str]):
    g = X_train.mean(axis=0).astype(np.float32)
    Xc = X_train - g[None, :]
    dirs = []
    for lab in relation_order:
        m = y_train == lab
        if not np.any(m):
            raise RuntimeError(f"Train split missing label {lab}")
        v = Xc[m].mean(axis=0)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            raise RuntimeError(f"Zero direction for label {lab}")
        dirs.append((v / n).astype(np.float32))
    D = np.stack(dirs, axis=0)  # [C,H]
    return g, D


def cv_scores_and_features(X: np.ndarray, labels: np.ndarray, groups: np.ndarray, relation_order: List[str], cv_folds: int, seed: int):
    C = len(relation_order)
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(relation_order)}
    y_idx = np.asarray([label_to_idx[str(l)] for l in labels], dtype=np.int64)

    scores = np.full((n, C), np.nan, dtype=np.float32)
    X_centered_for_plot = np.full_like(X, np.nan, dtype=np.float32)
    pred_idx = np.full(n, -1, dtype=np.int64)
    fold_id = np.full(n, -1, dtype=np.int64)

    splits = get_splits(X, labels, groups, cv_folds, seed)
    for fi, (tr, te) in enumerate(splits):
        g, D = fit_four_directions(X[tr], labels[tr], relation_order)
        Xt = X[te] - g[None, :]
        Xtn = l2_normalize(Xt, axis=1)
        sc = Xtn @ D.T
        scores[te] = sc.astype(np.float32)
        X_centered_for_plot[te] = Xt.astype(np.float32)
        pred_idx[te] = np.argmax(sc, axis=1)
        fold_id[te] = fi

    correct = pred_idx == y_idx
    # Margin = true score - best other score.
    true_score = scores[np.arange(n), y_idx]
    tmp = scores.copy()
    tmp[np.arange(n), y_idx] = -np.inf
    margin = true_score - np.max(tmp, axis=1)
    acc = float(correct.mean())
    return scores, X_centered_for_plot, pred_idx, y_idx, correct, margin, fold_id


def maybe_pca_for_tsne(X: np.ndarray, pca_dim: int, seed: int):
    if pca_dim and pca_dim > 0 and X.shape[1] > pca_dim and X.shape[0] > pca_dim + 2:
        X2 = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
        return X2, f"pca{pca_dim}"
    return X, "raw"


def run_tsne(X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    # t-SNE requires perplexity < n_samples.
    perp = min(float(perplexity), max(5.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(X)


def scatter_labels(Y: np.ndarray, labels: np.ndarray, relation_order: List[str], title: str, out_png: Path, correct: np.ndarray | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    for lab in relation_order:
        m = labels == lab
        if not np.any(m):
            continue
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


def plot_heatmap(M: np.ndarray, relation_order: List[str], title: str, out_png: Path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=180)
    im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(relation_order)))
    ax.set_xticklabels([f"dir:{x}" for x in relation_order], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(relation_order)))
    ax.set_yticklabels([f"true:{x}" for x in relation_order])
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_points_csv(path: Path, Y: np.ndarray, labels: np.ndarray, ids: np.ndarray, pred_idx: np.ndarray, relation_order: List[str], correct: np.ndarray, margin: np.ndarray):
    with path.open("w", encoding="utf-8") as f:
        f.write("id,relation,pred,correct,margin,x,y\n")
        for sample_id, lab, pi, ok, ma, (x, y) in zip(ids.tolist(), labels.tolist(), pred_idx.tolist(), correct.tolist(), margin.tolist(), Y.tolist()):
            sid = str(sample_id).replace(",", "_")
            pred = relation_order[int(pi)] if pi >= 0 else "NA"
            f.write(f"{sid},{lab},{pred},{int(bool(ok))},{float(ma):.8g},{float(x):.8g},{float(y):.8g}\n")


def main():
    args = parse_args()
    inp = Path(args.input_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relation_order = [x.strip() for x in args.relations.split(",") if x.strip()]
    if len(relation_order) < 2:
        raise RuntimeError("Need at least two relations")

    layer_map, labels_all, ids_all, metadata = load_npz(inp)
    layers = choose_layers(layer_map, args.layers)
    dataset_name = args.dataset_name or str(metadata.get("dataset", inp.parent.parent.name))
    model_name = args.model_name or str(metadata.get("model_alias", inp.parent.name))

    print(f"Input: {inp}")
    print(f"Dataset/model: {dataset_name} / {model_name}")
    print(f"Available layers: {sorted(layer_map.keys())}")
    print(f"Selected layers: {layers}")
    print(f"Raw label counts: {dict(Counter(labels_all.tolist()))}")
    print(f"Relation order: {relation_order}")

    summary = {
        "input_npz": str(inp),
        "dataset": dataset_name,
        "model": model_name,
        "relations": relation_order,
        "cv_folds": args.cv_folds,
        "layers": {},
    }

    for L in layers:
        X0 = layer_map[L]
        X, labels, ids = filter_labels(X0, labels_all, ids_all, relation_order, args.max_per_label, args.seed)
        counts = dict(Counter(labels.tolist()))
        if any(counts.get(l, 0) == 0 for l in relation_order):
            raise RuntimeError(f"Layer L{L}: missing relation labels after filtering: counts={counts}")
        scores, Xc, pred_idx, y_idx, correct, margin, fold_id = cv_scores_and_features(
            X, labels, ids, relation_order, args.cv_folds, args.seed
        )
        acc = float(correct.mean())
        mean_margin = float(margin.mean())
        print(f"L{L}: n={len(labels)} counts={counts} acc={acc:.3f} mean_margin={mean_margin:.3f}")

        prefix = f"four_dir_L{L}_{'_'.join(relation_order)}_cv{args.cv_folds}"
        layer_sum = {
            "n": int(len(labels)),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "acc": acc,
            "mean_margin": mean_margin,
            "files": {},
        }

        # Score heatmap: mean cosine score by true label and direction.
        M = np.zeros((len(relation_order), len(relation_order)), dtype=np.float32)
        for i, lab in enumerate(relation_order):
            M[i] = scores[labels == lab].mean(axis=0)
        heat_png = out_dir / f"{prefix}_mean_score_heatmap.png"
        plot_heatmap(M, relation_order, f"{dataset_name} / {model_name} / L{L} / mean cosine scores", heat_png, vmin=-1, vmax=1)
        layer_sum["files"]["mean_score_heatmap"] = str(heat_png)

        # Confusion matrix.
        cm = confusion_matrix(y_idx, pred_idx, labels=np.arange(len(relation_order))).astype(np.float32)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_png = out_dir / f"{prefix}_confusion_norm.png"
        plot_heatmap(cmn, relation_order, f"{dataset_name} / {model_name} / L{L} / confusion acc={acc:.3f}", cm_png, vmin=0, vmax=1)
        layer_sum["files"]["confusion_norm"] = str(cm_png)

        # PCA2 of 4D score space.
        if args.also_pca2:
            if scores.shape[1] >= 2:
                Y = PCA(n_components=2, random_state=args.seed).fit_transform(scores)
            else:
                Y = np.column_stack([scores[:, 0], np.zeros(len(scores), dtype=np.float32)])
            pca_score_png = out_dir / f"{prefix}_score4d_pca2.png"
            scatter_labels(Y, labels, relation_order, f"{dataset_name} / {model_name} / L{L} / 4D score PCA2 acc={acc:.3f}", pca_score_png, correct)
            save_points_csv(out_dir / f"{prefix}_score4d_pca2.csv", Y, labels, ids, pred_idx, relation_order, correct, margin)
            layer_sum["files"]["score4d_pca2"] = str(pca_score_png)

            Xraw_pca = PCA(n_components=2, random_state=args.seed).fit_transform(Xc)
            pca_raw_png = out_dir / f"{prefix}_centered_raw_pca2.png"
            scatter_labels(Xraw_pca, labels, relation_order, f"{dataset_name} / {model_name} / L{L} / centered raw PCA2 acc={acc:.3f}", pca_raw_png, correct)
            save_points_csv(out_dir / f"{prefix}_centered_raw_pca2.csv", Xraw_pca, labels, ids, pred_idx, relation_order, correct, margin)
            layer_sum["files"]["centered_raw_pca2"] = str(pca_raw_png)

        # t-SNE on 4D score space: this visualizes classifier separability most directly.
        if not args.no_score_tsne:
            Y = run_tsne(scores, args.perplexity, args.seed)
            score_png = out_dir / f"{prefix}_score4d_tsne_perp{int(args.perplexity)}.png"
            scatter_labels(Y, labels, relation_order, f"{dataset_name} / {model_name} / L{L} / 4D score t-SNE acc={acc:.3f}", score_png, correct)
            save_points_csv(out_dir / f"{prefix}_score4d_tsne.csv", Y, labels, ids, pred_idx, relation_order, correct, margin)
            layer_sum["files"]["score4d_tsne"] = str(score_png)

        # t-SNE on centered raw features: closer to representation, but may still be dominated by non-relation variance.
        if not args.no_raw_tsne:
            Xvis, prep = maybe_pca_for_tsne(Xc, args.pca_dim, args.seed)
            Y = run_tsne(Xvis, args.perplexity, args.seed)
            raw_png = out_dir / f"{prefix}_centered_raw_{prep}_tsne_perp{int(args.perplexity)}.png"
            scatter_labels(Y, labels, relation_order, f"{dataset_name} / {model_name} / L{L} / centered raw {prep} t-SNE acc={acc:.3f}", raw_png, correct)
            save_points_csv(out_dir / f"{prefix}_centered_raw_tsne.csv", Y, labels, ids, pred_idx, relation_order, correct, margin)
            layer_sum["files"]["centered_raw_tsne"] = str(raw_png)

        np.savez_compressed(
            out_dir / f"{prefix}_cv_scores.npz",
            scores=scores,
            labels=labels.astype(str),
            ids=ids.astype(str),
            pred=np.asarray([relation_order[i] for i in pred_idx], dtype=object),
            correct=correct,
            margin=margin,
            fold_id=fold_id,
            relation_order=np.asarray(relation_order, dtype=object),
        )
        layer_sum["files"]["cv_scores_npz"] = str(out_dir / f"{prefix}_cv_scores.npz")
        summary["layers"][str(L)] = layer_sum

    summary_path = out_dir / f"four_direction_separability_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
