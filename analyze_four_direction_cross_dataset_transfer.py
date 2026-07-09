#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-dataset transfer for four-relation direction codebooks.

Train relation directions on all samples from one dataset, then test by cosine
argmax on all samples from another dataset.

Example label alignment:
  Controlled A: left,right,on,under
  COCO_two:     left,right,above,below
  canonical:    left,right,up,down

For each layer:
  r_i = h(subject_i) - h(reference_i)
  train: compute center g and class directions d_c from train dataset only
  test:  predict argmax_c cos(r_i - g, d_c) on test dataset only
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Row:
    index: int
    sid: str
    relation_raw: str
    label: str
    subject: str
    reference: str

    @property
    def pair_group(self) -> str:
        return " || ".join(sorted((self.subject, self.reference)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-npz", required=True)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-dataset-name", default="controlled_A")
    p.add_argument("--test-dataset-name", default="coco_two")
    p.add_argument("--model-name", default="model")
    p.add_argument("--layers", default="auto", help="auto/common or comma-separated layer ids")
    p.add_argument("--train-relations", default="left,right,on,under")
    p.add_argument("--test-relations", default="left,right,above,below")
    p.add_argument(
        "--class-names",
        default="left,right,up,down",
        help="Canonical class names aligned by position with train/test relations.",
    )
    p.add_argument("--feature-centering", choices=("global", "none"), default="global")
    p.add_argument("--max-train-per-label", type=int, default=0)
    p.add_argument("--max-test-per-label", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def split_csv(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def str_arr(x):
    return np.asarray([str(v) for v in np.asarray(x, dtype=object).tolist()], dtype=object)


def parse_layers(text: str) -> List[int]:
    if text.lower() in {"auto", "all", "common"}:
        return []
    return sorted({int(t.strip().lstrip("L")) for t in text.split(",") if t.strip()})


def available_layers_from_keys(keys) -> List[int]:
    out = []
    pat = re.compile(r"^layer_(\d+)_subject$")
    for k in keys:
        m = pat.match(k)
        if m and f"layer_{m.group(1)}_reference" in keys:
            out.append(int(m.group(1)))
    return sorted(out)


def available_layers(npz_path: Path) -> List[int]:
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        out = available_layers_from_keys(keys)
        if out:
            return out
        if "relation_vectors" in keys and "decoder_block_index" in keys:
            return [int(v) for v in z["decoder_block_index"].tolist()]
    return []


def load_layer(npz_path: Path, layer: int, rel_to_class: Dict[str, str]) -> Tuple[List[Row], np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        if f"layer_{layer}_subject" in keys and f"layer_{layer}_reference" in keys:
            subj_state = np.asarray(z[f"layer_{layer}_subject"], dtype=np.float64)
            ref_state = np.asarray(z[f"layer_{layer}_reference"], dtype=np.float64)
            X_all = subj_state - ref_state
        elif "relation_vectors" in keys and "decoder_block_index" in keys:
            blocks = [int(v) for v in z["decoder_block_index"].tolist()]
            if layer not in blocks:
                raise RuntimeError(f"Layer {layer} not in relation_vectors blocks={blocks} for {npz_path}")
            li = blocks.index(layer)
            X_all = np.asarray(z["relation_vectors"][:, li, :], dtype=np.float64)
        else:
            raise RuntimeError(f"No layer subject/reference or relation_vectors found in {npz_path}")

        labels_all = str_arr(z["relation"])
        n = len(labels_all)
        sid = str_arr(z["sid"]) if "sid" in keys else np.asarray([str(i) for i in range(n)], dtype=object)
        subject = (
            str_arr(z["subject_name"]) if "subject_name" in keys
            else (str_arr(z["subject"]) if "subject" in keys else np.asarray(["NA"] * n, dtype=object))
        )
        reference = (
            str_arr(z["reference_name"]) if "reference_name" in keys
            else (str_arr(z["reference"]) if "reference" in keys else np.asarray(["NA"] * n, dtype=object))
        )

    rows = []
    keep_idx = []
    for i, lab0 in enumerate(labels_all.tolist()):
        lab = str(lab0).strip().lower()
        if lab not in rel_to_class:
            continue
        rows.append(Row(i, str(sid[i]), lab, rel_to_class[lab], str(subject[i]), str(reference[i])))
        keep_idx.append(i)
    if not keep_idx:
        raise RuntimeError(f"No usable rows after relation filtering for {npz_path}. Allowed={sorted(rel_to_class)}")
    X = X_all[np.asarray(keep_idx, dtype=np.int64)]
    return rows, X


def downsample(rows: List[Row], X: np.ndarray, class_names: List[str], max_per_label: int, seed: int):
    if not max_per_label or max_per_label <= 0:
        return rows, X
    rng = np.random.default_rng(seed)
    labels = np.asarray([r.label for r in rows], dtype=object)
    chosen = []
    for lab in class_names:
        idx = np.where(labels == lab)[0]
        if len(idx) > max_per_label:
            idx = rng.choice(idx, size=max_per_label, replace=False)
        chosen.extend(idx.tolist())
    chosen = sorted(chosen)
    return [rows[i] for i in chosen], X[np.asarray(chosen, dtype=np.int64)]


def norm_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def fit_codebook(Xtr: np.ndarray, ytr: np.ndarray, class_names: List[str], feature_centering: str):
    if feature_centering == "global":
        center = Xtr.mean(axis=0, keepdims=True)
    else:
        center = np.zeros((1, Xtr.shape[1]), dtype=np.float64)
    Xc = Xtr - center
    dirs = []
    means = []
    train_counts = {}
    for lab in class_names:
        m = ytr == lab
        train_counts[lab] = int(m.sum())
        if not np.any(m):
            raise RuntimeError(f"Train set missing class {lab}")
        v = Xc[m].mean(axis=0)
        means.append(v)
        dirs.append(v / max(float(np.linalg.norm(v)), EPS))
    return center.reshape(-1), np.stack(dirs), np.stack(means), train_counts


def evaluate(Xtr, ytr, Xte, yte, class_names, feature_centering):
    label_to_idx = {lab: i for i, lab in enumerate(class_names)}
    yte_i = np.asarray([label_to_idx[v] for v in yte], dtype=np.int64)
    center, D, means, train_counts = fit_codebook(Xtr, ytr, class_names, feature_centering)
    scores = norm_rows(Xte - center[None, :]) @ D.T
    pred = np.argmax(scores, axis=1)
    correct = pred == yte_i

    tmp = scores.copy()
    true_score = scores[np.arange(len(yte_i)), yte_i]
    tmp[np.arange(len(yte_i)), yte_i] = -np.inf
    signed_margin = true_score - tmp.max(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    top1_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(yte_i.tolist(), pred.tolist()):
        confusion[t, p] += 1

    per_class = {}
    for lab, ci in label_to_idx.items():
        m = yte_i == ci
        per_class[lab] = {
            "n": int(m.sum()),
            "acc": float((pred[m] == ci).mean()) if np.any(m) else None,
            "mean_signed_margin": float(signed_margin[m].mean()) if np.any(m) else None,
        }

    return {
        "acc": float(correct.mean()),
        "mean_signed_margin": float(signed_margin.mean()),
        "mean_top1_margin": float(top1_margin.mean()),
        "train_counts": train_counts,
        "test_counts": dict(Counter(yte.tolist())),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "dirs_gram": (D @ D.T).tolist(),
    }


def main():
    args = parse_args()
    train_npz = Path(args.train_npz)
    test_npz = Path(args.test_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rels = split_csv(args.train_relations)
    test_rels = split_csv(args.test_relations)
    class_names = split_csv(args.class_names)
    if not (len(train_rels) == len(test_rels) == len(class_names)):
        raise ValueError(
            f"Lengths must match: train_rels={train_rels}, test_rels={test_rels}, class_names={class_names}"
        )
    train_map = {r: c for r, c in zip(train_rels, class_names)}
    test_map = {r: c for r, c in zip(test_rels, class_names)}

    train_layers = available_layers(train_npz)
    test_layers = available_layers(test_npz)
    common_layers = sorted(set(train_layers) & set(test_layers))
    layers = parse_layers(args.layers) or common_layers
    if not layers:
        raise RuntimeError(f"No common layers. train={train_layers}, test={test_layers}")
    missing = [L for L in layers if L not in common_layers]
    if missing:
        raise RuntimeError(f"Requested layers not common: {missing}. common={common_layers}")

    print(f"Model: {args.model_name}")
    print(f"Train: {args.train_dataset_name} <- {train_npz}")
    print(f"Test : {args.test_dataset_name} <- {test_npz}")
    print(f"Train layers: {train_layers}")
    print(f"Test layers : {test_layers}")
    print(f"Selected common layers: {layers}")
    print(f"Train relation map: {train_map}")
    print(f"Test relation map : {test_map}")
    print(f"Class order: {class_names}")
    print(f"Readout: train-dataset-only class mean directions; test-dataset-only cosine argmax; centering={args.feature_centering}")

    summary = {
        "model": args.model_name,
        "train_dataset": args.train_dataset_name,
        "test_dataset": args.test_dataset_name,
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "train_relations": train_rels,
        "test_relations": test_rels,
        "class_names": class_names,
        "train_relation_map": train_map,
        "test_relation_map": test_map,
        "feature_centering": args.feature_centering,
        "layers": {},
    }

    for L in layers:
        train_rows, Xtr = load_layer(train_npz, L, train_map)
        test_rows, Xte = load_layer(test_npz, L, test_map)
        train_rows, Xtr = downsample(train_rows, Xtr, class_names, args.max_train_per_label, args.seed)
        test_rows, Xte = downsample(test_rows, Xte, class_names, args.max_test_per_label, args.seed + 17)

        ytr = np.asarray([r.label for r in train_rows], dtype=object)
        yte = np.asarray([r.label for r in test_rows], dtype=object)
        if Xtr.shape[1] != Xte.shape[1]:
            raise RuntimeError(f"Hidden dims differ at L{L}: train={Xtr.shape}, test={Xte.shape}")

        res = evaluate(Xtr, ytr, Xte, yte, class_names, args.feature_centering)
        train_pairs = len({r.pair_group for r in train_rows})
        test_pairs = len({r.pair_group for r in test_rows})
        train_objects = len({r.subject for r in train_rows} | {r.reference for r in train_rows})
        test_objects = len({r.subject for r in test_rows} | {r.reference for r in test_rows})

        print(
            f"L{L}: train_n={len(train_rows)} test_n={len(test_rows)} "
            f"train_counts={dict(Counter(ytr.tolist()))} test_counts={dict(Counter(yte.tolist()))} "
            f"acc={res['acc']:.3f} signed_margin={res['mean_signed_margin']:.3f} "
            f"top1_margin={res['mean_top1_margin']:.3f}"
        )
        print(f"  per_class={res['per_class']}")

        summary["layers"][str(L)] = {
            "train_n": int(len(train_rows)),
            "test_n": int(len(test_rows)),
            "train_pairs": int(train_pairs),
            "test_pairs": int(test_pairs),
            "train_objects": int(train_objects),
            "test_objects": int(test_objects),
            **res,
        }

    tag = f"{args.train_dataset_name}_to_{args.test_dataset_name}"
    out_json = out_dir / "four_direction_cross_dataset_transfer_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_json}")

    out_tsv = out_dir / "four_direction_cross_dataset_transfer_summary.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("model\ttrain_dataset\ttest_dataset\tlayer\ttrain_n\ttest_n\tacc\tsigned_margin\ttop1_margin\n")
        for L, info in summary["layers"].items():
            f.write(
                f"{args.model_name}\t{args.train_dataset_name}\t{args.test_dataset_name}\tL{L}\t"
                f"{info['train_n']}\t{info['test_n']}\t{info['acc']:.6f}\t"
                f"{info['mean_signed_margin']:.6f}\t{info['mean_top1_margin']:.6f}\n"
            )
    print(f"Saved TSV: {out_tsv}")


if __name__ == "__main__":
    main()
