#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot t-SNE/PCA of raw relation vectors r = h(subject) - h(reference).

Supports two NPZ formats used in the current experiments:
1) COCO/VG two-object extractor:
   relation_vectors [N, L, H], decoder_block_index, relation, image_id
2) Controlled A saved states:
   layer_<L>_subject [N,H], layer_<L>_reference [N,H], relation, sid

This script does not fit relation axes, does not center by object, and does not
residualize semantics. It visualizes the raw r vectors.
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
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required: pip install -U matplotlib") from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install -U scikit-learn") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", required=True, help="Path to states .npz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument(
        "--layers",
        default="auto",
        help=(
            "Comma-separated decoder block indices, e.g. 12,16. "
            "Use 'auto' to choose all stored layers, or 'all' for all stored layers."
        ),
    )
    p.add_argument(
        "--relations",
        default="left,right,above,below,on,under,front,behind",
        help="Comma-separated relation labels to keep. Use 'all' to keep all labels.",
    )
    p.add_argument("--max-per-label", type=int, default=0, help="0 means no downsampling")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pca-dim", type=int, default=50, help="PCA dim before t-SNE; 0 disables PCA pre-step")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--standardize", action="store_true", help="Standardize feature dimensions before PCA/t-SNE")
    p.add_argument("--l2-normalize", action="store_true", help="L2-normalize each raw r before visualization")
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--also-pca2", action="store_true", help="Also save a direct 2D PCA scatter")
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
            if "image_id" in keys:
                ids = as_str_array(z["image_id"])
            elif "sid" in keys:
                ids = as_str_array(z["sid"])
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
            f"Available keys: {sorted(z.files)[:80]}"
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


def filter_and_sample(
    X: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    relation_spec: str,
    max_per_label: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones(len(labels), dtype=bool)
    if relation_spec.lower() != "all":
        allowed = {x.strip() for x in relation_spec.split(",") if x.strip()}
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
    return X, labels, ids


def preprocess(X: np.ndarray, pca_dim: int, standardize: bool, l2_normalize: bool, seed: int) -> Tuple[np.ndarray, str]:
    X = X.astype(np.float32)
    steps = []
    if l2_normalize:
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(denom, 1e-8)
        steps.append("l2")
    if standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        steps.append("std")
    if pca_dim and pca_dim > 0 and X.shape[1] > pca_dim and X.shape[0] > pca_dim + 2:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
        steps.append(f"pca{pca_dim}")
    return X, "+".join(steps) if steps else "raw"


def scatter_plot(Y: np.ndarray, labels: np.ndarray, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=180)
    labs = sorted(set(labels.tolist()))
    for lab in labs:
        mask = labels == lab
        ax.scatter(Y[mask, 0], Y[mask, 1], s=18, alpha=0.78, label=f"{lab} (n={int(mask.sum())})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(fontsize=8, loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_coords(out_csv: Path, Y: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> None:
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("id,relation,x,y\n")
        for sample_id, lab, (x, y) in zip(ids.tolist(), labels.tolist(), Y.tolist()):
            safe_id = str(sample_id).replace(",", "_")
            f.write(f"{safe_id},{lab},{float(x):.8g},{float(y):.8g}\n")


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

    summary = {
        "input_npz": str(inp),
        "dataset": dataset_name,
        "model": model_name,
        "selected_layers": layers,
        "raw_counts": dict(Counter(labels_all.tolist())),
        "plots": [],
        "note": "raw r = h(subject)-h(reference); no axis fitting, no object-centering unless flags say otherwise",
    }

    for L in layers:
        X, labels, ids = filter_and_sample(
            layer_map[L], labels_all, ids_all, args.relations, args.max_per_label, args.seed
        )
        if len(X) < 5:
            print(f"Skip L{L}: too few samples after filtering: {len(X)}")
            continue
        print(f"L{L}: n={len(X)} counts={dict(Counter(labels.tolist()))} hidden_dim={X.shape[1]}")
        Xp, prep_name = preprocess(X, args.pca_dim, args.standardize, args.l2_normalize, args.seed)

        # Direct PCA-2 visualization from raw/preprocessed features.
        if args.also_pca2:
            if Xp.shape[1] >= 2:
                Yp = PCA(n_components=2, random_state=args.seed).fit_transform(Xp)
            else:
                Yp = np.c_[Xp[:, 0], np.zeros(len(Xp))]
            png = out_dir / f"raw_r_pca2_L{L}_{prep_name}.png"
            csv = out_dir / f"raw_r_pca2_L{L}_{prep_name}.csv"
            scatter_plot(Yp, labels, f"raw r PCA2 | {dataset_name} | {model_name} | L{L}", png)
            save_coords(csv, Yp, labels, ids)
            summary["plots"].append(str(png))

        if not args.no_tsne:
            # t-SNE perplexity must be < n_samples.
            perplexity = min(float(args.perplexity), max(2.0, (len(Xp) - 1) / 3.0))
            Y = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=args.seed,
                metric="euclidean",
            ).fit_transform(Xp)
            png = out_dir / f"raw_r_tsne_L{L}_{prep_name}_perp{perplexity:g}.png"
            csv = out_dir / f"raw_r_tsne_L{L}_{prep_name}_perp{perplexity:g}.csv"
            scatter_plot(Y, labels, f"raw r t-SNE | {dataset_name} | {model_name} | L{L}", png)
            save_coords(csv, Y, labels, ids)
            summary["plots"].append(str(png))
            print(f"  saved {png}")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

