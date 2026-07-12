#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare spatial information carried by different text-token representations.

All representations use exactly the same pair-aware 30/70 splits.

Evaluated representations:
  subject_last
  reference_last
  object_diff
  object_concat
  object_concat_diff
  object_mean_diff
  object_mean_concat
  question_last
  relation_anchor
  question_mean
  answer_readout

Two frozen-state classifiers are reported:
  1) cosine codebook: same class-mean direction logic as the existing pipeline
  2) linear ridge probe: dual-form regularized linear classifier

No VLM parameters are updated.
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

EPS = 1e-8

REPRESENTATIONS = [
    "subject_last",
    "reference_last",
    "object_diff",
    "object_concat",
    "object_concat_diff",
    "object_mean_diff",
    "object_mean_concat",
    "question_last",
    "relation_anchor",
    "question_mean",
    "answer_readout",
]


@dataclass(frozen=True)
class Row:
    index: int
    sid: str
    relation: str
    subject: str
    reference: str
    group: str


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="model")
    p.add_argument("--dataset-name", default="")
    p.add_argument("--relations", default="left,right,on,under")
    p.add_argument("--representations", default=",".join(REPRESENTATIONS))
    p.add_argument("--layers", default="auto")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--split-unit", choices=["pair", "sample", "id"], default="pair")
    p.add_argument("--feature-centering", choices=["global", "none"], default="global")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def normalize_relation(x: str) -> str:
    s = str(x).strip().lower().replace("-", "_")
    s = re.sub(r"\s+", "_", s).strip("._,")
    aliases = {
        "left_of": "left",
        "right_of": "right",
        "above": "on",
        "top": "on",
        "on_top": "on",
        "on_top_of": "on",
        "below": "under",
        "bottom": "under",
        "underneath": "under",
    }
    return aliases.get(s, s)


def parse_layers(raw: str, available: Sequence[int]) -> List[int]:
    if raw.lower() in {"auto", "all"}:
        return list(available)
    layers = sorted({
        int(x.strip().lstrip("L"))
        for x in raw.split(",")
        if x.strip()
    })
    missing = [x for x in layers if x not in available]
    if missing:
        raise ValueError(f"Requested layers {missing} not in available {list(available)}")
    return layers


def make_groups(rows: Sequence[Row], split_unit: str) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if split_unit == "pair":
            key = row.group
        elif split_unit == "id":
            key = row.sid
        else:
            key = f"sample::{i}"
        groups[key].append(i)
    return dict(groups)


def split_train_test_greedy(
    rows: Sequence[Row],
    relation_order: Sequence[str],
    train_ratio: float,
    split_unit: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    test_ratio = 1.0 - train_ratio
    n = len(rows)
    groups = make_groups(rows, split_unit)
    total = Counter(row.relation for row in rows)
    target_test_counts = {
        label: total[label] * test_ratio
        for label in relation_order
    }
    target_test_size = n * test_ratio

    for attempt in range(200):
        rng = random.Random(seed + attempt * 9973)
        items = list(groups.items())
        rng.shuffle(items)
        items.sort(key=lambda kv: (-len(kv[1]), rng.random()))

        test_indices: List[int] = []
        test_counts = Counter()
        remaining = items[:]

        def objective(idxs: Sequence[int]) -> float:
            candidate = test_counts.copy()
            candidate.update(rows[i].relation for i in idxs)
            candidate_size = len(test_indices) + len(idxs)
            relation_error = sum(
                (candidate[label] - target_test_counts[label]) ** 2
                for label in relation_order
            )
            size_error = (candidate_size - target_test_size) ** 2
            overflow = 0.0
            for label in relation_order:
                if candidate[label] > math.ceil(target_test_counts[label]) + 2:
                    overflow += 1000.0 * (
                        candidate[label] - target_test_counts[label]
                    ) ** 2
            return relation_error + size_error + overflow

        while remaining and len(test_indices) < int(round(target_test_size)):
            best = min(range(len(remaining)), key=lambda j: objective(remaining[j][1]))
            _, idxs = remaining.pop(best)
            test_indices.extend(idxs)
            test_counts.update(rows[i].relation for i in idxs)

        te = sorted(set(test_indices))
        tr = sorted(set(range(n)) - set(te))
        train_counts = Counter(rows[i].relation for i in tr)
        final_test_counts = Counter(rows[i].relation for i in te)

        if all(train_counts[x] > 0 for x in relation_order) and all(
            final_test_counts[x] > 0 for x in relation_order
        ):
            return np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)

    raise RuntimeError("Could not build a valid split with all requested labels.")


def row_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def codebook_predict(
    X: np.ndarray,
    y_idx: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    n_classes: int,
    centering: str,
) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = X[tr].astype(np.float32, copy=False)
    Xte = X[te].astype(np.float32, copy=False)
    center = (
        Xtr.mean(axis=0, keepdims=True)
        if centering == "global"
        else np.zeros((1, X.shape[1]), dtype=np.float32)
    )
    Xtrc = Xtr - center
    directions = []
    for class_id in range(n_classes):
        vector = Xtrc[y_idx[tr] == class_id].mean(axis=0)
        vector = vector / max(float(np.linalg.norm(vector)), EPS)
        directions.append(vector)
    D = np.stack(directions, axis=0)
    scores = row_normalize(Xte - center) @ D.T
    return np.argmax(scores, axis=1), scores


def ridge_predict(
    X: np.ndarray,
    y_idx: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    n_classes: int,
    centering: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = X[tr].astype(np.float32, copy=False)
    Xte = X[te].astype(np.float32, copy=False)
    center = (
        Xtr.mean(axis=0, keepdims=True)
        if centering == "global"
        else np.zeros((1, X.shape[1]), dtype=np.float32)
    )

    Xtr = row_normalize(Xtr - center)
    Xte = row_normalize(Xte - center)

    Y = np.zeros((len(tr), n_classes), dtype=np.float32)
    Y[np.arange(len(tr)), y_idx[tr]] = 1.0

    kernel = Xtr @ Xtr.T
    system = kernel + float(alpha) * np.eye(len(tr), dtype=np.float32)
    dual = np.linalg.solve(system, Y)
    scores = (Xte @ Xtr.T) @ dual
    return np.argmax(scores, axis=1), scores


def metrics(
    pred: np.ndarray,
    scores: np.ndarray,
    true: np.ndarray,
    relation_order: Sequence[str],
) -> Dict:
    correct = pred == true
    confusion = np.zeros(
        (len(relation_order), len(relation_order)),
        dtype=np.int64,
    )
    for t, p in zip(true.tolist(), pred.tolist()):
        confusion[t, p] += 1

    per_class = {}
    for class_id, label in enumerate(relation_order):
        mask = true == class_id
        per_class[label] = {
            "n": int(mask.sum()),
            "acc": float((pred[mask] == class_id).mean()) if np.any(mask) else None,
        }

    sorted_scores = np.sort(scores, axis=1)
    top_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    return {
        "acc": float(correct.mean()),
        "top1_margin": float(top_margin.mean()),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def get_representation(z, layer_index: int, name: str) -> np.ndarray:
    sub_last = np.asarray(z["subject_last_states"][:, layer_index, :], dtype=np.float32)
    ref_last = np.asarray(z["reference_last_states"][:, layer_index, :], dtype=np.float32)
    sub_mean = np.asarray(z["subject_mean_states"][:, layer_index, :], dtype=np.float32)
    ref_mean = np.asarray(z["reference_mean_states"][:, layer_index, :], dtype=np.float32)

    if name == "subject_last":
        return sub_last
    if name == "reference_last":
        return ref_last
    if name == "object_diff":
        return sub_last - ref_last
    if name == "object_concat":
        return np.concatenate([sub_last, ref_last], axis=1)
    if name == "object_concat_diff":
        return np.concatenate([sub_last, ref_last, sub_last - ref_last], axis=1)
    if name == "object_mean_diff":
        return sub_mean - ref_mean
    if name == "object_mean_concat":
        return np.concatenate([sub_mean, ref_mean], axis=1)
    if name == "question_last":
        return np.asarray(z["question_last_states"][:, layer_index, :], dtype=np.float32)
    if name == "relation_anchor":
        return np.asarray(z["relation_anchor_states"][:, layer_index, :], dtype=np.float32)
    if name == "question_mean":
        return np.asarray(z["question_mean_states"][:, layer_index, :], dtype=np.float32)
    if name == "answer_readout":
        return np.asarray(z["answer_readout_states"][:, layer_index, :], dtype=np.float32)
    raise ValueError(f"Unknown representation: {name}")


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=0))


def main():
    args = parse_args()
    input_path = Path(args.input_npz)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    relation_order = [
        normalize_relation(x)
        for x in args.relations.split(",")
        if x.strip()
    ]
    representations = [
        x.strip()
        for x in args.representations.split(",")
        if x.strip()
    ]
    unknown = [x for x in representations if x not in REPRESENTATIONS]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=True) as z:
        required = {
            "sid",
            "subject",
            "reference",
            "relation",
            "decoder_block_index",
            "subject_last_states",
            "reference_last_states",
            "subject_mean_states",
            "reference_mean_states",
            "question_last_states",
            "relation_anchor_states",
            "question_mean_states",
            "answer_readout_states",
        }
        missing = sorted(required - set(z.files))
        if missing:
            raise RuntimeError(f"Input NPZ is missing keys: {missing}")

        sids = [str(x) for x in z["sid"].tolist()]
        subjects = [str(x) for x in z["subject"].tolist()]
        references = [str(x) for x in z["reference"].tolist()]
        labels_raw = [normalize_relation(x) for x in z["relation"].tolist()]
        if "group" in z.files:
            groups = [str(x) for x in z["group"].tolist()]
        else:
            groups = [
                " || ".join(sorted((s, r)))
                for s, r in zip(subjects, references)
            ]
        available_layers = [int(x) for x in z["decoder_block_index"].tolist()]
        layers = parse_layers(args.layers, available_layers)

        keep = [i for i, label in enumerate(labels_raw) if label in relation_order]
        if len(keep) != len(labels_raw):
            print(f"[INFO] keeping {len(keep)}/{len(labels_raw)} requested-relation samples")

        rows = [
            Row(
                index=original_i,
                sid=sids[original_i],
                relation=labels_raw[original_i],
                subject=subjects[original_i],
                reference=references[original_i],
                group=groups[original_i],
            )
            for original_i in keep
        ]
        keep_idx = np.asarray(keep, dtype=np.int64)

        counts = Counter(row.relation for row in rows)
        missing_labels = [label for label in relation_order if counts[label] == 0]
        if missing_labels:
            raise RuntimeError(f"Missing labels {missing_labels}; counts={dict(counts)}")

        label_to_idx = {label: i for i, label in enumerate(relation_order)}
        y_idx = np.asarray([label_to_idx[row.relation] for row in rows], dtype=np.int64)

        splits = [
            split_train_test_greedy(
                rows,
                relation_order,
                args.train_ratio,
                args.split_unit,
                args.seed + repeat * 1009,
            )
            for repeat in range(args.repeats)
        ]

        summary = {
            "input_npz": str(input_path),
            "dataset": args.dataset_name or input_path.parent.name,
            "model": args.model_name,
            "relations": relation_order,
            "counts": dict(counts),
            "train_ratio": args.train_ratio,
            "test_ratio": 1.0 - args.train_ratio,
            "repeats": args.repeats,
            "split_unit": args.split_unit,
            "feature_centering": args.feature_centering,
            "ridge_alpha": args.ridge_alpha,
            "layers": layers,
            "representations": {},
        }

        for rep_name in representations:
            print(f"\n===== {rep_name} =====")
            rep_result = {"layers": {}}

            for layer in layers:
                li = available_layers.index(layer)
                X_all = get_representation(z, li, rep_name)
                X = X_all[keep_idx]

                codebook_runs = []
                ridge_runs = []

                for repeat, (tr, te) in enumerate(splits):
                    true = y_idx[te]

                    code_pred, code_scores = codebook_predict(
                        X,
                        y_idx,
                        tr,
                        te,
                        len(relation_order),
                        args.feature_centering,
                    )
                    ridge_pred, ridge_scores = ridge_predict(
                        X,
                        y_idx,
                        tr,
                        te,
                        len(relation_order),
                        args.feature_centering,
                        args.ridge_alpha,
                    )

                    code_metric = metrics(code_pred, code_scores, true, relation_order)
                    ridge_metric = metrics(ridge_pred, ridge_scores, true, relation_order)
                    code_metric["repeat"] = repeat
                    ridge_metric["repeat"] = repeat
                    codebook_runs.append(code_metric)
                    ridge_runs.append(ridge_metric)

                code_mean, code_std = mean_std([x["acc"] for x in codebook_runs])
                ridge_mean, ridge_std = mean_std([x["acc"] for x in ridge_runs])

                rep_result["layers"][str(layer)] = {
                    "feature_dim": int(X.shape[1]),
                    "codebook": {
                        "acc_mean": code_mean,
                        "acc_std": code_std,
                        "repeats": codebook_runs,
                    },
                    "ridge": {
                        "acc_mean": ridge_mean,
                        "acc_std": ridge_std,
                        "repeats": ridge_runs,
                    },
                }

                print(
                    f"L{layer:>3} dim={X.shape[1]:>6} | "
                    f"codebook={code_mean:.3f}±{code_std:.3f} | "
                    f"ridge={ridge_mean:.3f}±{ridge_std:.3f}"
                )

            summary["representations"][rep_name] = rep_result

        best_rows = []
        for rep_name, rep_info in summary["representations"].items():
            for classifier in ["codebook", "ridge"]:
                best_layer, best_info = max(
                    rep_info["layers"].items(),
                    key=lambda item: item[1][classifier]["acc_mean"],
                )
                best_rows.append({
                    "representation": rep_name,
                    "classifier": classifier,
                    "best_layer": int(best_layer),
                    "acc_mean": best_info[classifier]["acc_mean"],
                    "acc_std": best_info[classifier]["acc_std"],
                    "feature_dim": best_info["feature_dim"],
                })
        summary["best"] = best_rows

    json_path = out_dir / "carrier_ablation_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tsv_path = out_dir / "carrier_ablation_best.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("representation\tclassifier\tbest_layer\tacc_mean\tacc_std\tfeature_dim\n")
        for row in best_rows:
            f.write(
                f"{row['representation']}\t{row['classifier']}\tL{row['best_layer']}\t"
                f"{row['acc_mean']:.6f}\t{row['acc_std']:.6f}\t{row['feature_dim']}\n"
            )

    print("\n===== BEST RESULTS =====")
    for row in best_rows:
        print(
            f"{row['representation']:22s} {row['classifier']:8s} "
            f"L{row['best_layer']:<3d} "
            f"{row['acc_mean']:.3f}±{row['acc_std']:.3f}"
        )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()
