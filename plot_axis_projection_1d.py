#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit a centroid-difference relation axis and plot 1D projections.

Supports the NPZ formats used in current experiments:
1) COCO/VG two-object extractor:
   relation_vectors [N,L,H], decoder_block_index, relation, image_id
2) Controlled A saved states:
   layer_<L>_subject [N,H], layer_<L>_reference [N,H], relation, sid

For a binary relation pair, e.g. negative=left, positive=right:
  r_i = h(subject_i) - h(reference_i)
  d = normalize(mu_positive - mu_negative)
  c = (mu_positive + mu_negative) / 2
  score_i = (r_i - c)^T d

If --cv-folds > 1, the axis is fitted on train folds and scores are reported on
held-out groups only. If --cv-folds 0/1, the axis is fitted on all samples.
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
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception:  # older sklearn fallback
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
    p.add_argument("--negative-label", default="left")
    p.add_argument("--positive-label", default="right")
    p.add_argument("--cv-folds", type=int, default=0, help="0/1 = fit on all; >1 = group CV")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-per-label", type=int, default=0, help="0 means no downsampling")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--jitter", type=float, default=0.08)
    p.add_argument("--save-projected-vectors", action="store_true", help="Save r_parallel = c + score*d for each layer as npz")
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


def filter_binary(X, labels, ids, neg_label, pos_label, max_per_label, seed):
    keep = np.isin(labels, [neg_label, pos_label])
    X, labels, ids = X[keep], labels[keep], ids[keep]
    y = np.where(labels == pos_label, 1, -1).astype(np.int64)

    if max_per_label and max_per_label > 0:
        rng = np.random.default_rng(seed)
        chosen = []
        for val in [-1, 1]:
            idx = np.where(y == val)[0]
            if len(idx) > max_per_label:
                idx = rng.choice(idx, size=max_per_label, replace=False)
            chosen.extend(idx.tolist())
        chosen = np.asarray(sorted(chosen), dtype=np.int64)
        X, labels, ids, y = X[chosen], labels[chosen], ids[chosen], y[chosen]
    return X.astype(np.float32), labels, ids, y


def fit_axis(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    neg = X_train[y_train == -1]
    pos = X_train[y_train == 1]
    if len(neg) == 0 or len(pos) == 0:
        raise RuntimeError("Train split must contain both classes.")
    mu_neg = neg.mean(axis=0)
    mu_pos = pos.mean(axis=0)
    v = mu_pos - mu_neg
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RuntimeError("Positive and negative centroids are numerically identical; cannot define axis.")
    d = v / n
    c = (mu_pos + mu_neg) / 2.0
    return d.astype(np.float32), c.astype(np.float32)


def project_scores(X: np.ndarray, d: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((X - c[None, :]) @ d).astype(np.float32)


def compute_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    pred = np.where(scores > 0, 1, -1)
    acc = float((pred == y).mean())
    out = {"accuracy_at_zero": acc}
    if len(set(y.tolist())) == 2:
        y01 = (y == 1).astype(np.int64)
        try:
            out["roc_auc"] = float(roc_auc_score(y01, scores))
        except Exception:
            out["roc_auc"] = None
        try:
            out["average_precision"] = float(average_precision_score(y01, scores))
        except Exception:
            out["average_precision"] = None
    neg_scores = scores[y == -1]
    pos_scores = scores[y == 1]
    out.update({
        "n_negative": int(len(neg_scores)),
        "n_positive": int(len(pos_scores)),
        "mean_negative": float(np.mean(neg_scores)) if len(neg_scores) else None,
        "mean_positive": float(np.mean(pos_scores)) if len(pos_scores) else None,
        "std_negative": float(np.std(neg_scores)) if len(neg_scores) else None,
        "std_positive": float(np.std(pos_scores)) if len(pos_scores) else None,
        "median_negative": float(np.median(neg_scores)) if len(neg_scores) else None,
        "median_positive": float(np.median(pos_scores)) if len(pos_scores) else None,
        "frac_negative_below_zero": float((neg_scores < 0).mean()) if len(neg_scores) else None,
        "frac_positive_above_zero": float((pos_scores > 0).mean()) if len(pos_scores) else None,
    })
    if len(neg_scores) and len(pos_scores):
        pooled = np.sqrt((np.var(neg_scores) + np.var(pos_scores)) / 2.0)
        out["d_prime_like"] = float((np.mean(pos_scores) - np.mean(neg_scores)) / max(pooled, 1e-12))
    return out


def get_splits(X, y, groups, cv_folds, seed):
    if cv_folds <= 1:
        idx = np.arange(len(y))
        return [(idx, idx)]
    n_groups = len(set(groups.tolist()))
    if n_groups < cv_folds:
        raise RuntimeError(f"Need at least cv_folds unique groups. groups={n_groups}, cv_folds={cv_folds}")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        return list(splitter.split(X, y, groups))
    splitter = GroupKFold(n_splits=cv_folds)
    return list(splitter.split(X, y, groups))


def plot_1d(scores: np.ndarray, labels: np.ndarray, y: np.ndarray, neg_label: str, pos_label: str,
            title: str, out_png: Path, bins: int, jitter: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), dpi=180, gridspec_kw={"height_ratios": [2.0, 1.15]})

    ax = axes[0]
    neg_scores = scores[y == -1]
    pos_scores = scores[y == 1]
    all_min = float(np.min(scores))
    all_max = float(np.max(scores))
    pad = 0.05 * max(all_max - all_min, 1e-6)
    hist_range = (all_min - pad, all_max + pad)
    ax.hist(neg_scores, bins=bins, range=hist_range, alpha=0.55, density=True, label=f"{neg_label} (n={len(neg_scores)})")
    ax.hist(pos_scores, bins=bins, range=hist_range, alpha=0.55, density=True, label=f"{pos_label} (n={len(pos_scores)})")
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.35)

    ax = axes[1]
    y_base = np.where(y == -1, 0.0, 1.0)
    y_jit = y_base + rng.normal(0.0, jitter, size=len(y))
    for val, lab, yy in [(-1, neg_label, 0), (1, pos_label, 1)]:
        mask = y == val
        ax.scatter(scores[mask], y_jit[mask], s=16, alpha=0.72, label=lab)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([neg_label, pos_label])
    ax.set_xlabel("axis projection score: (r - c)^T d")
    ax.set_ylim(-0.35, 1.35)
    ax.grid(True, linewidth=0.3, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_scores_csv(path: Path, scores: np.ndarray, labels: np.ndarray, ids: np.ndarray, y: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("id,relation,y,score,pred,correct\n")
        pred = np.where(scores > 0, 1, -1)
        for sample_id, lab, yi, s, pi in zip(ids.tolist(), labels.tolist(), y.tolist(), scores.tolist(), pred.tolist()):
            safe_id = str(sample_id).replace(",", "_")
            f.write(f"{safe_id},{lab},{int(yi)},{float(s):.8g},{int(pi)},{int(pi == yi)}\n")


def main() -> None:
    args = parse_args()
    inp = Path(args.input_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_map, labels_all, ids_all, metadata = load_npz(inp)
    layers = choose_layers(layer_map, args.layers)
    dataset_name = args.dataset_name or str(metadata.get("dataset", inp.parent.parent.name))
    model_name = args.model_name or str(metadata.get("model_alias", inp.parent.name))

    print(f"Input: {inp}")
    print(f"Dataset/model: {dataset_name} / {model_name}")
    print(f"Available layers: {sorted(layer_map.keys())}")
    print(f"Selected layers: {layers}")
    print(f"Raw label counts: {dict(Counter(labels_all.tolist()))}")
    print(f"Binary pair: negative={args.negative_label}, positive={args.positive_label}; cv_folds={args.cv_folds}")

    summary = {
        "input_npz": str(inp),
        "dataset": dataset_name,
        "model": model_name,
        "negative_label": args.negative_label,
        "positive_label": args.positive_label,
        "cv_folds": int(args.cv_folds),
        "selected_layers": layers,
        "raw_counts": dict(Counter(labels_all.tolist())),
        "layers": {},
    }

    projected_store = {}
    for L in layers:
        X, labels, ids, y = filter_binary(
            layer_map[L], labels_all, ids_all,
            args.negative_label, args.positive_label,
            args.max_per_label, args.seed,
        )
        if len(X) < 4 or len(set(y.tolist())) < 2:
            print(f"Skip L{L}: too few binary samples. n={len(X)}, counts={dict(Counter(labels.tolist()))}")
            continue
        print(f"L{L}: n={len(X)} counts={dict(Counter(labels.tolist()))} hidden_dim={X.shape[1]}")

        if args.cv_folds and args.cv_folds > 1:
            scores = np.full(len(y), np.nan, dtype=np.float32)
            fold_info = []
            for fold_id, (tr, te) in enumerate(get_splits(X, y, ids, args.cv_folds, args.seed), start=1):
                # Some group folds can be class-degenerate on tiny data; skip if needed.
                if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                    print(f"  skip fold {fold_id}: train/test class-degenerate")
                    continue
                d, c = fit_axis(X[tr], y[tr])
                scores[te] = project_scores(X[te], d, c)
                fold_metrics = compute_metrics(scores[te], y[te])
                fold_info.append({"fold": fold_id, "n_train": int(len(tr)), "n_test": int(len(te)), **fold_metrics})
            valid = ~np.isnan(scores)
            scores_eval = scores[valid]
            labels_eval = labels[valid]
            ids_eval = ids[valid]
            y_eval = y[valid]
            fit_name = f"cv{args.cv_folds}"
            metrics = compute_metrics(scores_eval, y_eval)
            metrics["folds"] = fold_info
        else:
            d, c = fit_axis(X, y)
            scores_eval = project_scores(X, d, c)
            labels_eval, ids_eval, y_eval = labels, ids, y
            fit_name = "fitall"
            metrics = compute_metrics(scores_eval, y_eval)
            if args.save_projected_vectors:
                # r_parallel = c + score*d, same dimensionality as original r.
                projected_store[f"layer_{L}_projected_vectors"] = (c[None, :] + scores_eval[:, None] * d[None, :]).astype(np.float32)
                projected_store[f"layer_{L}_scores"] = scores_eval.astype(np.float32)

        png = out_dir / f"axis_projection_1d_L{L}_{args.negative_label}_vs_{args.positive_label}_{fit_name}.png"
        csv = out_dir / f"axis_projection_1d_L{L}_{args.negative_label}_vs_{args.positive_label}_{fit_name}.csv"
        title = (
            f"{dataset_name} | {model_name} | L{L} | "
            f"{args.negative_label} vs {args.positive_label} | {fit_name} | "
            f"acc={metrics['accuracy_at_zero']:.3f}"
        )
        plot_1d(scores_eval, labels_eval, y_eval, args.negative_label, args.positive_label,
                title, png, args.bins, args.jitter, args.seed)
        save_scores_csv(csv, scores_eval, labels_eval, ids_eval, y_eval)
        print(f"  acc@0={metrics['accuracy_at_zero']:.4f} auc={metrics.get('roc_auc')} d'={metrics.get('d_prime_like')}")
        print(f"  saved {png}")

        summary["layers"][str(L)] = {
            "n": int(len(scores_eval)),
            "counts": dict(Counter(labels_eval.tolist())),
            "fit_name": fit_name,
            "metrics": metrics,
            "plot_png": str(png),
            "scores_csv": str(csv),
        }

    if args.save_projected_vectors and projected_store:
        projected_store["relation"] = labels_eval
        projected_store["id"] = ids_eval
        projected_store["negative_label"] = np.asarray(args.negative_label)
        projected_store["positive_label"] = np.asarray(args.positive_label)
        npz_path = out_dir / f"projected_vectors_{args.negative_label}_vs_{args.positive_label}.npz"
        np.savez_compressed(npz_path, **projected_store)
        summary["projected_vectors_npz"] = str(npz_path)

    summary_path = out_dir / f"axis_projection_summary_{args.negative_label}_vs_{args.positive_label}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
