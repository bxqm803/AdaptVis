#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-NPZ relation-direction codebook with explicit train/test split.

Use case: combine Controlled A + Controlled B and test whether six relation
classes (left/right/on/under/in_front/behind) are separable.

For each layer:
  r_i = h(subject_i) - h(reference_i)
  train set: compute global center g and one normalized mean direction per class
  test set: predict by argmax_c cos(r_i - g, d_c)

Supports npz files with either:
  - layer_<L>_subject and layer_<L>_reference, or
  - relation_vectors with decoder_block_index.
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
    source: str
    sid: str
    relation: str
    subject: str
    reference: str

    @property
    def pair_group(self) -> str:
        if self.subject != "NA" and self.reference != "NA":
            return " || ".join(sorted((self.subject, self.reference)))
        return f"{self.source}::sid::{self.sid}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npzs", required=True, help="comma-separated npz paths")
    p.add_argument("--dataset-names", default="", help="comma-separated names; defaults to file parent names")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="model")
    p.add_argument("--layers", default="auto", help="auto/all or comma-separated layer ids")
    p.add_argument("--relations", default="left,right,on,under,in_front,behind")
    p.add_argument("--train-ratio", type=float, default=0.3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--split-unit", choices=("pair", "sample", "id", "source_id"), default="pair")
    p.add_argument("--feature-centering", choices=("global", "none"), default="global")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-per-label", type=int, default=0)
    p.add_argument("--strict-relations", action="store_true", help="fail if any requested relation is missing")
    return p.parse_args()


def normalize_relation(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._,")

    aliases = {
        "infront": "in_front",
        "in_front": "in_front",
        "in_front_of": "in_front",
        "front": "in_front",
        "front_of": "in_front",
        "before": "in_front",
        "behind": "behind",
        "back": "behind",
        "back_of": "behind",
        "left_of": "left",
        "right_of": "right",
        "above": "on",
        "above_of": "on",
        "top": "on",
        "on_top": "on",
        "on_top_of": "on",
        "below": "under",
        "below_of": "under",
        "bottom": "under",
        "underneath": "under",
    }
    return aliases.get(s, s)


def str_arr(x):
    return np.asarray([str(v) for v in np.asarray(x, dtype=object).tolist()], dtype=object)


def parse_layers(text: str) -> List[int]:
    if text.lower() in {"auto", "all"}:
        return []
    return sorted({int(t.strip().lstrip("L")) for t in text.split(",") if t.strip()})


def available_layers_keys(keys) -> List[int]:
    out = []
    pat = re.compile(r"^layer_(\d+)_subject$")
    for k in keys:
        m = pat.match(k)
        if m and f"layer_{m.group(1)}_reference" in keys:
            out.append(int(m.group(1)))
    return sorted(out)


def available_layers_npz(path: Path) -> List[int]:
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        avail = available_layers_keys(keys)
        if avail:
            return avail
        if "relation_vectors" in keys and "decoder_block_index" in keys:
            return [int(v) for v in z["decoder_block_index"].tolist()]
    return []


def load_one_layer(npz_path: Path, source: str, layer: int, relation_order: List[str]):
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
            raise RuntimeError(f"No usable layer states found in {npz_path}")

        if "relation" not in keys:
            raise RuntimeError(f"No relation key in {npz_path}")
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

    allowed = set(relation_order)
    rows = []
    keep_idx = []
    for i, lab0 in enumerate(labels_all.tolist()):
        lab = normalize_relation(lab0)
        if lab not in allowed:
            continue
        rows.append(Row(i, source, str(sid[i]), lab, str(subject[i]), str(reference[i])))
        keep_idx.append(i)
    X = X_all[np.asarray(keep_idx, dtype=np.int64)] if keep_idx else np.zeros((0, X_all.shape[-1]), dtype=np.float64)
    return rows, X


def load_layer(paths: List[Path], sources: List[str], layer: int, relation_order: List[str]):
    all_rows = []
    all_X = []
    for p, src in zip(paths, sources):
        rows, X = load_one_layer(p, src, layer, relation_order)
        all_rows.extend(rows)
        if len(rows):
            all_X.append(X)
    if not all_X:
        raise RuntimeError(f"No rows found for layer {layer}")
    Xcat = np.concatenate(all_X, axis=0)
    return all_rows, Xcat


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
        if split_unit == "pair":
            key = row.pair_group
        elif split_unit == "id":
            key = row.sid
        elif split_unit == "source_id":
            key = f"{row.source}::{row.sid}"
        else:
            key = f"sample::{i}"
        groups[key].append(i)
    return dict(groups)


def split_train_test_greedy(rows, relation_order, train_ratio, split_unit, seed):
    assert 0.0 < train_ratio < 1.0
    test_ratio = 1.0 - train_ratio
    n = len(rows)
    groups = make_groups(rows, split_unit)
    if len(groups) < 2:
        raise RuntimeError(f"Need at least 2 groups, got {len(groups)}")

    total = Counter(r.relation for r in rows)
    target_test_counts = {lab: total[lab] * test_ratio for lab in relation_order}
    target_test_size = n * test_ratio

    for attempt in range(200):
        rng = random.Random(seed + attempt * 9973)
        items = list(groups.items())
        rng.shuffle(items)
        items.sort(key=lambda kv: (-len(kv[1]), rng.random()))

        test_idxs = []
        test_counts = Counter()
        remaining = items[:]

        def obj_add(idxs):
            cand = test_counts.copy()
            cand.update(rows[i].relation for i in idxs)
            cand_size = len(test_idxs) + len(idxs)
            rel_err = sum((cand[lab] - target_test_counts[lab]) ** 2 for lab in relation_order)
            size_err = (cand_size - target_test_size) ** 2
            over = 0.0
            for lab in relation_order:
                if cand[lab] > math.ceil(target_test_counts[lab]) + 2:
                    over += 1000.0 * (cand[lab] - target_test_counts[lab]) ** 2
            if cand_size > math.ceil(target_test_size) + max(1, int(0.02 * n)):
                over += 1000.0 * (cand_size - target_test_size) ** 2
            return rel_err + size_err + over

        while remaining and len(test_idxs) < int(round(target_test_size)):
            best_j = min(range(len(remaining)), key=lambda j: obj_add(remaining[j][1]))
            _, idxs = remaining.pop(best_j)
            test_idxs.extend(idxs)
            test_counts.update(rows[i].relation for i in idxs)

        te = sorted(set(test_idxs))
        tr = sorted(set(range(n)) - set(te))
        train_counts = Counter(rows[i].relation for i in tr)
        test_counts = Counter(rows[i].relation for i in te)
        if all(train_counts[lab] > 0 for lab in relation_order) and all(test_counts[lab] > 0 for lab in relation_order):
            return np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)

    raise RuntimeError("Could not make valid split with all labels. Try --split-unit sample.")


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
    scores = norm_rows(X[te] - center[None, :]) @ D.T
    pred = np.argmax(scores, axis=1)
    true = yi[te]
    correct = pred == true

    tmp = scores.copy()
    true_score = scores[np.arange(len(te)), true]
    tmp[np.arange(len(te)), true] = -np.inf
    signed_margin = true_score - tmp.max(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    top_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    confusion = np.zeros((len(relation_order), len(relation_order)), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        confusion[t, p] += 1

    per_class = {}
    for lab, ci in label_to_idx.items():
        m = true == ci
        per_class[lab] = {
            "n": int(m.sum()),
            "acc": float((pred[m] == ci).mean()) if np.any(m) else None,
        }

    # per-source test accuracy
    sources = np.asarray([r.source for r in rows], dtype=object)
    per_source = {}
    for src in sorted(set(sources[te].tolist())):
        m = sources[te] == src
        per_source[src] = {"n": int(m.sum()), "acc": float(correct[m].mean())}

    return {
        "acc": float(correct.mean()),
        "mean_signed_margin": float(signed_margin.mean()),
        "mean_top1_margin": float(top_margin.mean()),
        "train_counts": dict(Counter(y[tr].tolist())),
        "test_counts": dict(Counter(y[te].tolist())),
        "per_class": per_class,
        "per_source": per_source,
        "confusion": confusion.tolist(),
        "dirs_gram": (D @ D.T).tolist(),
    }


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def main():
    args = parse_args()
    paths = [Path(x.strip()) for x in args.input_npzs.split(",") if x.strip()]
    if not paths:
        raise SystemExit("No --input-npzs")
    if args.dataset_names.strip():
        sources = [x.strip() for x in args.dataset_names.split(",") if x.strip()]
        if len(sources) != len(paths):
            raise SystemExit("--dataset-names count must match --input-npzs count")
    else:
        sources = [p.parent.parent.name if p.name == "states.npz" else p.stem for p in paths]

    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing npz: {p}")

    relation_order = [normalize_relation(x) for x in args.relations.split(",") if x.strip()]
    if len(set(relation_order)) != len(relation_order):
        raise SystemExit(f"Duplicate normalized relations: {relation_order}")

    avails = [available_layers_npz(p) for p in paths]
    common = sorted(set(avails[0]).intersection(*[set(a) for a in avails[1:]]))
    layers = parse_layers(args.layers) or common
    missing_layers = [L for L in layers if L not in common]
    if missing_layers:
        raise SystemExit(f"Requested layers not common to all npzs: {missing_layers}; common={common}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Inputs:")
    for src, p, av in zip(sources, paths, avails):
        print(f"  {src}: {p} | layers={av}")
    print(f"Model: {args.model_name}")
    print(f"Common layers: {common}")
    print(f"Selected layers: {layers}")
    print(f"Relations: {relation_order}")
    print(f"Split: train_ratio={args.train_ratio:.2f}, test_ratio={1-args.train_ratio:.2f}, repeats={args.repeats}, split_unit={args.split_unit}")
    print(f"Centering: {args.feature_centering}")

    summary = {
        "input_npzs": [str(p) for p in paths],
        "sources": sources,
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
        rows, X = load_layer(paths, sources, L, relation_order)
        rows, X = downsample(rows, X, relation_order, args.max_per_label, args.seed)
        labels = np.asarray([r.relation for r in rows], dtype=object)
        counts = dict(Counter(labels.tolist()))
        missing = [lab for lab in relation_order if counts.get(lab, 0) == 0]
        if missing:
            msg = f"L{L}: missing requested relations {missing}; counts={counts}"
            if args.strict_relations:
                raise RuntimeError(msg)
            print("[WARN] " + msg)
            continue
        groups = make_groups(rows, args.split_unit)
        pairs = len({r.pair_group for r in rows})
        objects = len({r.subject for r in rows} | {r.reference for r in rows})
        src_counts = dict(Counter(r.source for r in rows))

        repeat_results = []
        for rep in range(args.repeats):
            tr, te = split_train_test_greedy(rows, relation_order, args.train_ratio, args.split_unit, args.seed + rep * 1009)
            res = evaluate_split(rows, X, relation_order, tr, te, args.feature_centering)
            res["repeat"] = rep
            res["train_n"] = int(len(tr))
            res["test_n"] = int(len(te))
            repeat_results.append(res)
            print(
                f"L{L} rep{rep}: n={len(rows)} train={len(tr)} test={len(te)} groups={len(groups)} "
                f"counts={counts} acc={res['acc']:.3f} signed_margin={res['mean_signed_margin']:.3f} "
                f"top1_margin={res['mean_top1_margin']:.3f} train_counts={res['train_counts']} test_counts={res['test_counts']} "
                f"per_source={res['per_source']}"
            )

        acc_mean, acc_std = mean_std([r["acc"] for r in repeat_results])
        sm_mean, sm_std = mean_std([r["mean_signed_margin"] for r in repeat_results])
        tm_mean, tm_std = mean_std([r["mean_top1_margin"] for r in repeat_results])
        print(f"L{L} summary: acc={acc_mean:.3f}±{acc_std:.3f}, signed_margin={sm_mean:.3f}±{sm_std:.3f}, top1_margin={tm_mean:.3f}±{tm_std:.3f}")

        summary["layers"][str(L)] = {
            "n": int(len(rows)),
            "groups": int(len(groups)),
            "pairs": int(pairs),
            "objects": int(objects),
            "counts": counts,
            "source_counts": src_counts,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "mean_signed_margin_mean": sm_mean,
            "mean_signed_margin_std": sm_std,
            "mean_top1_margin_mean": tm_mean,
            "mean_top1_margin_std": tm_std,
            "repeats": repeat_results,
        }

    out_json = out_dir / "multi_direction_train_test_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_json}")

    out_tsv = out_dir / "multi_direction_train_test_summary.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("model\tlayer\tn\tgroups\tacc_mean\tacc_std\tsigned_margin_mean\ttop1_margin_mean\tcounts\tsource_counts\n")
        for L, info in summary["layers"].items():
            f.write(
                f"{args.model_name}\tL{L}\t{info['n']}\t{info['groups']}\t"
                f"{info['acc_mean']:.6f}\t{info['acc_std']:.6f}\t"
                f"{info['mean_signed_margin_mean']:.6f}\t{info['mean_top1_margin_mean']:.6f}\t"
                f"{json.dumps(info['counts'], ensure_ascii=False)}\t{json.dumps(info['source_counts'], ensure_ascii=False)}\n"
            )
    print(f"Saved TSV: {out_tsv}")


if __name__ == "__main__":
    main()
