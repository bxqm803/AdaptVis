#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot four spatial relations using two learned centroid-difference axes.

For each layer, construct raw relation vectors:
  r_i = h(subject_i) - h(reference_i)

Fit horizontal and vertical axes from labeled pairs:
  horizontal: negative_h -> positive_h, e.g. left -> right
  vertical:   negative_v -> positive_v, e.g. below -> above / under -> on

For held-out samples, compute:
  x_i = (r_i - c_h)^T d_h
  y_i = (r_i - c_v)^T d_v

Then plot all four labels in the 2D axis-coordinate plane.

Supports NPZ formats used in current experiments:
1) COCO/VG two-object extractor:
   relation_vectors [N,L,H], decoder_block_index, relation, image_id
2) Controlled A saved states:
   layer_<L>_subject [N,H], layer_<L>_reference [N,H], relation, sid
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
    p.add_argument("--negative-h", default="left")
    p.add_argument("--positive-h", default="right")
    p.add_argument("--negative-v", default="below")
    p.add_argument("--positive-v", default="above")
    p.add_argument("--cv-folds", type=int, default=5, help="0/1 = fit on all; >1 = group CV")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-per-label", type=int, default=0, help="0 means no downsampling")
    p.add_argument("--equal-aspect", action="store_true")
    p.add_argument("--annotate-accuracy", action="store_true")
    p.add_argument("--save-axis-vectors", action="store_true")
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


def filter_four(X, labels, ids, labs, max_per_label, seed):
    allowed = set(labs)
    keep = np.isin(labels, sorted(allowed))
    X, labels, ids = X[keep], labels[keep], ids[keep]

    if max_per_label and max_per_label > 0:
        rng = np.random.default_rng(seed)
        chosen = []
        for lab in sorted(set(labels.tolist())):
            idx = np.where(labels == lab)[0]
            if len(idx) > max_per_label:
                idx = rng.choice(idx, size=max_per_label, replace=False)
            chosen.extend(idx.tolist())
        chosen = np.asarray(sorted(chosen), dtype=np.int64)
        X, labels, ids = X[chosen], labels[chosen], ids[chosen]
    return X.astype(np.float32), labels, ids


def fit_axis(X_train: np.ndarray, labels_train: np.ndarray, neg_label: str, pos_label: str) -> Tuple[np.ndarray, np.ndarray]:
    neg = X_train[labels_train == neg_label]
    pos = X_train[labels_train == pos_label]
    if len(neg) == 0 or len(pos) == 0:
        raise RuntimeError(f"Train split lacks both classes for axis {neg_label}->{pos_label}: n_neg={len(neg)}, n_pos={len(pos)}")
    mu_neg = neg.mean(axis=0)
    mu_pos = pos.mean(axis=0)
    v = mu_pos - mu_neg
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RuntimeError(f"Centroids are identical for axis {neg_label}->{pos_label}")
    d = v / n
    c = (mu_pos + mu_neg) / 2.0
    return d.astype(np.float32), c.astype(np.float32)


def score_axis(X: np.ndarray, d: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((X - c[None, :]) @ d).astype(np.float32)


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


def projection_metrics(x, y, labels, neg_h, pos_h, neg_v, pos_v):
    out = {}
    h_mask = np.isin(labels, [neg_h, pos_h])
    if h_mask.any():
        h_true = np.where(labels[h_mask] == pos_h, 1, -1)
        h_pred = np.where(x[h_mask] > 0, 1, -1)
        out["horizontal_acc_at_x0"] = float((h_pred == h_true).mean())
        out["horizontal_n"] = int(h_mask.sum())
        out["horizontal_frac_neg_left"] = float((x[labels == neg_h] < 0).mean()) if np.any(labels == neg_h) else None
        out["horizontal_frac_pos_right"] = float((x[labels == pos_h] > 0).mean()) if np.any(labels == pos_h) else None
    v_mask = np.isin(labels, [neg_v, pos_v])
    if v_mask.any():
        v_true = np.where(labels[v_mask] == pos_v, 1, -1)
        v_pred = np.where(y[v_mask] > 0, 1, -1)
        out["vertical_acc_at_y0"] = float((v_pred == v_true).mean())
        out["vertical_n"] = int(v_mask.sum())
        out["vertical_frac_neg_below"] = float((y[labels == neg_v] < 0).mean()) if np.any(labels == neg_v) else None
        out["vertical_frac_pos_above"] = float((y[labels == pos_v] > 0).mean()) if np.any(labels == pos_v) else None
    # Useful leakage/noise diagnostics: irrelevant-axis means.
    for lab in [neg_h, pos_h, neg_v, pos_v]:
        m = labels == lab
        if m.any():
            out[f"mean_x_{lab}"] = float(np.mean(x[m]))
            out[f"mean_y_{lab}"] = float(np.mean(y[m]))
    return out


def plot_2d(x, y, labels, title, out_png, neg_h, pos_h, neg_v, pos_v, metrics, equal_aspect, annotate):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    order = [neg_h, pos_h, neg_v, pos_v]
    for lab in order:
        m = labels == lab
        if m.any():
            ax.scatter(x[m], y[m], s=20, alpha=0.75, label=f"{lab} (n={int(m.sum())})")
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"horizontal score: {pos_h} minus {neg_h}")
    ax.set_ylabel(f"vertical score: {pos_v} minus {neg_v}")
    ax.set_title(title, fontsize=10)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(fontsize=8, loc="best", frameon=True)
    if annotate:
        lines = []
        if "horizontal_acc_at_x0" in metrics:
            lines.append(f"H acc@x=0: {metrics['horizontal_acc_at_x0']:.3f}")
        if "vertical_acc_at_y0" in metrics:
            lines.append(f"V acc@y=0: {metrics['vertical_acc_at_y0']:.3f}")
        if lines:
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, linewidth=0.5))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_csv(path, x, y, labels, ids):
    with path.open("w", encoding="utf-8") as f:
        f.write("id,relation,horizontal_score,vertical_score\n")
        for sid, lab, xi, yi in zip(ids.tolist(), labels.tolist(), x.tolist(), y.tolist()):
            safe_id = str(sid).replace(",", "_")
            f.write(f"{safe_id},{lab},{float(xi):.8g},{float(yi):.8g}\n")


def main():
    args = parse_args()
    inp = Path(args.input_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_map, labels_all, ids_all, metadata = load_npz(inp)
    layers = choose_layers(layer_map, args.layers)
    dataset_name = args.dataset_name or str(metadata.get("dataset", inp.parent.parent.name))
    model_name = args.model_name or str(metadata.get("model_alias", inp.parent.name))
    labs = [args.negative_h, args.positive_h, args.negative_v, args.positive_v]

    print(f"Input: {inp}")
    print(f"Dataset/model: {dataset_name} / {model_name}")
    print(f"Available layers: {sorted(layer_map.keys())}")
    print(f"Selected layers: {layers}")
    print(f"Raw label counts: {dict(Counter(labels_all.tolist()))}")
    print(f"Axes: H {args.negative_h}->{args.positive_h}; V {args.negative_v}->{args.positive_v}; cv_folds={args.cv_folds}")

    summary = {
        "input_npz": str(inp),
        "dataset": dataset_name,
        "model": model_name,
        "layers": layers,
        "negative_h": args.negative_h,
        "positive_h": args.positive_h,
        "negative_v": args.negative_v,
        "positive_v": args.positive_v,
        "cv_folds": args.cv_folds,
        "results": {},
    }

    for L in layers:
        X0 = layer_map[L]
        X, labels, ids = filter_four(X0, labels_all, ids_all, labs, args.max_per_label, args.seed)
        counts = dict(Counter(labels.tolist()))
        missing = [lab for lab in labs if counts.get(lab, 0) == 0]
        if missing:
            print(f"L{L}: SKIP missing labels: {missing}; counts={counts}")
            continue
        print(f"L{L}: n={len(labels)} counts={counts} hidden_dim={X.shape[1]}")

        scores_h = np.full(len(labels), np.nan, dtype=np.float32)
        scores_v = np.full(len(labels), np.nan, dtype=np.float32)
        fold_ids = np.full(len(labels), -1, dtype=np.int64)
        saved_axes = []
        splits = get_splits(X, labels, ids, args.cv_folds, args.seed)
        for fold, (train_idx, test_idx) in enumerate(splits):
            d_h, c_h = fit_axis(X[train_idx], labels[train_idx], args.negative_h, args.positive_h)
            d_v, c_v = fit_axis(X[train_idx], labels[train_idx], args.negative_v, args.positive_v)
            scores_h[test_idx] = score_axis(X[test_idx], d_h, c_h)
            scores_v[test_idx] = score_axis(X[test_idx], d_v, c_v)
            fold_ids[test_idx] = fold
            if args.save_axis_vectors:
                saved_axes.append({
                    "fold": fold,
                    "d_h": d_h,
                    "c_h": c_h,
                    "d_v": d_v,
                    "c_v": c_v,
                })

        valid = np.isfinite(scores_h) & np.isfinite(scores_v)
        if not valid.all():
            X = X[valid]
            labels = labels[valid]
            ids = ids[valid]
            scores_h = scores_h[valid]
            scores_v = scores_v[valid]
            fold_ids = fold_ids[valid]

        metrics = projection_metrics(scores_h, scores_v, labels, args.negative_h, args.positive_h, args.negative_v, args.positive_v)
        summary["results"][str(L)] = {"counts": counts, **metrics}
        hacc = metrics.get("horizontal_acc_at_x0", None)
        vacc = metrics.get("vertical_acc_at_y0", None)
        print(f"  H-acc={hacc:.3f} | V-acc={vacc:.3f}" if hacc is not None and vacc is not None else f"  metrics={metrics}")

        tag = f"{args.negative_h}_vs_{args.positive_h}__{args.negative_v}_vs_{args.positive_v}"
        cvtag = f"cv{args.cv_folds}" if args.cv_folds and args.cv_folds > 1 else "fitall"
        out_png = out_dir / f"axis_projection_2d_L{L}_{tag}_{cvtag}.png"
        title = f"{dataset_name} / {model_name} / L{L} / 2D axis projection ({cvtag})"
        plot_2d(scores_h, scores_v, labels, title, out_png, args.negative_h, args.positive_h, args.negative_v, args.positive_v, metrics, args.equal_aspect, args.annotate_accuracy)
        print(f"  saved {out_png}")

        out_csv = out_dir / f"axis_projection_2d_L{L}_{tag}_{cvtag}.csv"
        save_csv(out_csv, scores_h, scores_v, labels, ids)

        if args.save_axis_vectors and saved_axes:
            npz_path = out_dir / f"axis_vectors_L{L}_{tag}_{cvtag}.npz"
            np.savez_compressed(
                npz_path,
                d_h=np.stack([a["d_h"] for a in saved_axes], axis=0),
                c_h=np.stack([a["c_h"] for a in saved_axes], axis=0),
                d_v=np.stack([a["d_v"] for a in saved_axes], axis=0),
                c_v=np.stack([a["c_v"] for a in saved_axes], axis=0),
                fold=np.asarray([a["fold"] for a in saved_axes], dtype=np.int64),
            )

    summary_path = out_dir / f"axis_projection_2d_summary_{args.negative_h}_vs_{args.positive_h}__{args.negative_v}_vs_{args.positive_v}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
