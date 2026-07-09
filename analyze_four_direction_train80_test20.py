#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Four-relation direction codebook with explicit 80/20 train/test split.

For each layer:
  r_i = h(subject_i) - h(reference_i)
  train set: compute global center g and one normalized mean direction per relation
  test set: predict by argmax_c cos(r_i - g, d_c)

Works with the npz files produced by the Controlled A / COCO/VG relation-state extractors:
  - layer_<L>_subject and layer_<L>_reference, or
  - relation_vectors with decoder_block_index

Default uses grouped split by unordered object pair, so the same object-pair will not appear
in both train and test when subject/reference metadata is available.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

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
    p.add_argument("--model-name", default="model")
    p.add_argument("--layers", default="auto", help="auto/all or comma-separated layer ids")
    p.add_argument("--relations", default="left,right,on,under")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--repeats", type=int, default=1, help="number of random 80/20 splits")
    p.add_argument("--split-unit", choices=("pair", "sample", "id"), default="pair")
    p.add_argument("--feature-centering", choices=("global", "none"), default="global")
    p.add_argument("--seed", type=int, default=1)
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


def available_layers_relation_vectors(z) -> List[int]:
    if "relation_vectors" in z.files and "decoder_block_index" in z.files:
        return [int(v) for v in z["decoder_block_index"].tolist()]
    return []


def load_layer(npz_path: Path, layer: int, relation_order: List[str]):
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        if f"layer_{layer}_subject" in keys and f"layer_{layer}_reference" in keys:
            subj_state = np.asarray(z[f"layer_{layer}_subject"], dtype=np.float64)
            ref_state = np.asarray(z[f"layer_{layer}_reference"], dtype=np.float64)
            X_all = subj_state - ref_state
        elif "relation_vectors" in keys and "decoder_block_index" in keys:
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


def make_groups(rows: Sequence[Row], split_unit: str) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if split_unit == "pair" and row.subject != "NA" and row.reference != "NA":
            key = row.pair_group
        elif split_unit == "id":
            key = row.sid
        else:
            key = f"sample::{i}"
        groups[key].append(i)
    return dict(groups)


def split_train_test_greedy(
    rows: Sequence[Row],
    relation_order: List[str],
    train_ratio: float,
    split_unit: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced grouped 80/20 split.

    Greedily selects whole groups for test so that test label counts and size are close
    to the requested target. Retries with different shuffles if train/test miss any label.
    """
    assert 0.0 < train_ratio < 1.0
    test_ratio = 1.0 - train_ratio
    n = len(rows)
    groups = make_groups(rows, split_unit)
    if len(groups) < 2:
        raise RuntimeError(f"Need at least 2 groups, got {len(groups)}")

    y_all = [r.relation for r in rows]
    total = Counter(y_all)
    target_test_counts = {lab: total[lab] * test_ratio for lab in relation_order}
    target_test_size = n * test_ratio

    for attempt in range(100):
        rng = random.Random(seed + attempt * 9973)
        items = list(groups.items())
        rng.shuffle(items)
        # place larger/more mixed groups earlier, but keep deterministic random tie-breaking
        items.sort(key=lambda kv: (-len(kv[1]), rng.random()))

        test_idxs: List[int] = []
        test_counts = Counter()

        def objective_after_add(idxs: List[int]) -> Tuple[float, float, float]:
            cand_counts = test_counts.copy()
            cand_counts.update(rows[i].relation for i in idxs)
            cand_size = len(test_idxs) + len(idxs)
            rel_err = sum((cand_counts[lab] - target_test_counts[lab]) ** 2 for lab in relation_order)
            size_err = (cand_size - target_test_size) ** 2
            over_penalty = 0.0
            for lab in relation_order:
                if cand_counts[lab] > math.ceil(target_test_counts[lab]) + 2:
                    over_penalty += 1000.0 * (cand_counts[lab] - target_test_counts[lab]) ** 2
            if cand_size > math.ceil(target_test_size) + max(1, int(0.02 * n)):
                over_penalty += 1000.0 * (cand_size - target_test_size) ** 2
            return rel_err + size_err + over_penalty, rel_err, size_err

        # Iteratively add the group that most improves the target, until target size reached.
        remaining = items[:]
        while remaining and len(test_idxs) < int(round(target_test_size)):
            best_j = None
            best_score = None
            for j, (_, idxs) in enumerate(remaining):
                score = objective_after_add(idxs)
                if best_score is None or score < best_score:
                    best_score = score
                    best_j = j
            _, idxs = remaining.pop(best_j)
            test_idxs.extend(idxs)
            test_counts.update(rows[i].relation for i in idxs)

        te = sorted(set(test_idxs))
        tr = sorted(set(range(n)) - set(te))
        train_counts = Counter(rows[i].relation for i in tr)
        test_counts = Counter(rows[i].relation for i in te)
        if all(train_counts[lab] > 0 for lab in relation_order) and all(test_counts[lab] > 0 for lab in relation_order):
            return np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)

    raise RuntimeError(
        f"Could not make valid {train_ratio:.2f}/{1-train_ratio:.2f} split with all labels. "
        f"Try --split-unit sample or reduce relation set."
    )


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


def evaluate_split(rows, X, relation_order, tr, te, feature_centering):
    y = np.asarray([r.relation for r in rows], dtype=object)
    label_to_idx = {lab: i for i, lab in enumerate(relation_order)}
    yi = np.asarray([label_to_idx[v] for v in y], dtype=np.int64)

    center, D, means = fit_codebook(X[tr], y[tr], relation_order, feature_centering)
    Xte = X[te] - center[None, :]
    scores = norm_rows(Xte) @ D.T
    pred = np.argmax(scores, axis=1)
    true = yi[te]
    correct = pred == true

    tmp = scores.copy()
    true_score = scores[np.arange(len(te)), true]
    tmp[np.arange(len(te)), true] = -np.inf
    signed_margin = true_score - tmp.max(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    top_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    per_class = {}
    confusion = np.zeros((len(relation_order), len(relation_order)), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        confusion[t, p] += 1
    for lab, ci in label_to_idx.items():
        m = true == ci
        per_class[lab] = {
            "n": int(m.sum()),
            "acc": float((pred[m] == ci).mean()) if np.any(m) else None,
        }

    return {
        "acc": float(correct.mean()),
        "mean_signed_margin": float(signed_margin.mean()),
        "mean_top1_margin": float(top_margin.mean()),
        "train_counts": dict(Counter(y[tr].tolist())),
        "test_counts": dict(Counter(y[te].tolist())),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "dirs_gram": (D @ D.T).tolist(),
    }


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def main():
    args = parse_args()
    inp = Path(args.input_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relation_order = [x.strip().lower() for x in args.relations.split(",") if x.strip()]
    with np.load(inp, allow_pickle=True) as z:
        avail = available_layers(set(z.files))
        if not avail:
            avail = available_layers_relation_vectors(z)
    layers = parse_layers(args.layers) or avail

    print(f"Input: {inp}")
    print(f"Dataset/model: {args.dataset_name} / {args.model_name}")
    print(f"Available layers: {avail}")
    print(f"Selected layers: {layers}")
    print(f"Relations: {relation_order}")
    print(f"Split: train_ratio={args.train_ratio:.2f}, test_ratio={1-args.train_ratio:.2f}, repeats={args.repeats}, split_unit={args.split_unit}")
    print(f"Readout: train-only class mean directions; test-only cosine argmax; centering={args.feature_centering}")

    summary = {
        "input_npz": str(inp),
        "dataset": args.dataset_name,
        "model": args.model_name,
        "relations": relation_order,
        "train_ratio": args.train_ratio,
        "test_ratio": 1 - args.train_ratio,
        "repeats": args.repeats,
        "split_unit": args.split_unit,
        "feature_centering": args.feature_centering,
        "seed": args.seed,
        "layers": {},
    }

    for L in layers:
        rows, X = load_layer(inp, L, relation_order)
        rows, X = downsample(rows, X, relation_order, args.max_per_label, args.seed)
        labels = np.asarray([r.relation for r in rows], dtype=object)
        groups = make_groups(rows, args.split_unit)
        pairs = len({r.pair_group for r in rows})
        objects = len({r.subject for r in rows} | {r.reference for r in rows})
        counts = dict(Counter(labels.tolist()))

        repeat_results = []
        for rep in range(args.repeats):
            tr, te = split_train_test_greedy(
                rows=rows,
                relation_order=relation_order,
                train_ratio=args.train_ratio,
                split_unit=args.split_unit,
                seed=args.seed + rep * 1009,
            )
            res = evaluate_split(rows, X, relation_order, tr, te, args.feature_centering)
            res["repeat"] = rep
            res["train_n"] = int(len(tr))
            res["test_n"] = int(len(te))
            repeat_results.append(res)
            print(
                f"L{L} rep{rep}: n={len(rows)} train={len(tr)} test={len(te)} groups={len(groups)} "
                f"counts={counts} acc={res['acc']:.3f} signed_margin={res['mean_signed_margin']:.3f} "
                f"top1_margin={res['mean_top1_margin']:.3f} "
                f"train_counts={res['train_counts']} test_counts={res['test_counts']}"
            )

        acc_mean, acc_std = mean_std([r["acc"] for r in repeat_results])
        sm_mean, sm_std = mean_std([r["mean_signed_margin"] for r in repeat_results])
        tm_mean, tm_std = mean_std([r["mean_top1_margin"] for r in repeat_results])
        print(
            f"L{L} summary: acc={acc_mean:.3f}±{acc_std:.3f}, "
            f"signed_margin={sm_mean:.3f}±{sm_std:.3f}, top1_margin={tm_mean:.3f}±{tm_std:.3f}"
        )

        summary["layers"][str(L)] = {
            "n": int(len(rows)),
            "groups": int(len(groups)),
            "pairs": int(pairs),
            "objects": int(objects),
            "counts": counts,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "mean_signed_margin_mean": sm_mean,
            "mean_signed_margin_std": sm_std,
            "mean_top1_margin_mean": tm_mean,
            "mean_top1_margin_std": tm_std,
            "repeats": repeat_results,
        }

    out_json = out_dir / "four_direction_train80_test20_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_json}")

    # flat TSV for quick grep/cat
    out_tsv = out_dir / "four_direction_train80_test20_summary.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("dataset\tmodel\tlayer\tn\tgroups\tacc_mean\tacc_std\tsigned_margin_mean\ttop1_margin_mean\n")
        for L, info in summary["layers"].items():
            f.write(
                f"{args.dataset_name}\t{args.model_name}\tL{L}\t{info['n']}\t{info['groups']}\t"
                f"{info['acc_mean']:.6f}\t{info['acc_std']:.6f}\t"
                f"{info['mean_signed_margin_mean']:.6f}\t{info['mean_top1_margin_mean']:.6f}\n"
            )
    print(f"Saved TSV: {out_tsv}")


if __name__ == "__main__":
    main()
