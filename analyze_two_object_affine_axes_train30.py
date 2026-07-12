#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


EPS = 1e-12

ALIASES = {
    "above": "above",
    "on": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "left_of": "left",
    "right_of": "right",
}


def norm_rel(x):
    x = str(x).lower().strip().replace("-", "_")
    return ALIASES.get(x, x)


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        required = {"relation", "relation_vectors", "decoder_block_index"}
        missing = required.difference(z.files)
        if missing:
            raise KeyError(
                f"NPZ is missing required keys: {sorted(missing)}. "
                f"Available keys: {sorted(z.files)}"
            )

        y = np.array(
            [norm_rel(x) for x in z["relation"].tolist()],
            dtype=object,
        )
        X = z["relation_vectors"].astype(np.float64)
        layers = [int(x) for x in z["decoder_block_index"].tolist()]

    if X.ndim != 3:
        raise ValueError(
            f"relation_vectors must have shape [N, L, H], got {X.shape}"
        )
    if X.shape[0] != len(y):
        raise ValueError(
            f"Sample mismatch: relation has {len(y)}, "
            f"relation_vectors has {X.shape[0]}"
        )
    if X.shape[1] != len(layers):
        raise ValueError(
            f"Layer mismatch: decoder_block_index has {len(layers)}, "
            f"relation_vectors has {X.shape[1]}"
        )

    return y, X, layers


def norm_rows(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        EPS,
    )


def fit_codebook(X, y, relations):
    center = X.mean(axis=0)
    dirs = []
    Xc = X - center

    for r in relations:
        m = y == r
        if m.sum() == 0:
            raise RuntimeError(f"Missing relation {r} in training set")

        d = Xc[m].mean(axis=0)
        d /= max(np.linalg.norm(d), EPS)
        dirs.append(d)

    return center, np.stack(dirs)


def evaluate(X, y, center, dirs, relations):
    score = norm_rows(X - center) @ dirs.T
    pred = np.argmax(score, axis=1)
    gt = np.array([relations.index(x) for x in y], dtype=np.int64)

    acc = float(np.mean(pred == gt))

    true_score = score[np.arange(len(gt)), gt]
    other_score = score.copy()
    other_score[np.arange(len(gt)), gt] = -np.inf
    margin = float(
        (true_score - other_score.max(axis=1)).mean()
    )

    return {
        "accuracy": acc,
        "margin": margin,
        "n": int(len(gt)),
        "counts": dict(Counter(y.tolist())),
    }


def stratified_train_test_split(
    y,
    relations,
    train_ratio,
    seed,
):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"--train-ratio must be between 0 and 1, got {train_ratio}"
        )

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for relation in relations:
        indices = np.flatnonzero(y == relation)

        if len(indices) < 2:
            raise RuntimeError(
                f"Relation {relation} has only {len(indices)} sample(s); "
                "at least 2 are required for train/test splitting"
            )

        indices = indices.copy()
        rng.shuffle(indices)

        n_train = int(round(len(indices) * train_ratio))
        n_train = max(1, min(n_train, len(indices) - 1))

        train_indices.extend(indices[:n_train].tolist())
        test_indices.extend(indices[n_train:].tolist())

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument(
        "--relations",
        default="left,right,above,below",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    relations = [
        norm_rel(x)
        for x in args.relations.split(",")
        if x.strip()
    ]

    if len(relations) != len(set(relations)):
        raise ValueError(
            f"Relations become duplicated after normalization: {relations}"
        )

    y, X, layers = load_npz(args.input_npz)

    mask = np.isin(y, relations)
    y = y[mask]
    X = X[mask]

    if len(y) == 0:
        raise RuntimeError(
            f"No samples match requested relations: {relations}"
        )

    full_counts = Counter(y.tolist())
    missing_relations = [
        relation
        for relation in relations
        if full_counts[relation] == 0
    ]
    if missing_relations:
        raise RuntimeError(
            f"Missing requested relations in input: {missing_relations}"
        )

    train_indices, test_indices = stratified_train_test_split(
        y=y,
        relations=relations,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    y_train = y[train_indices]
    y_test = y[test_indices]
    X_train = X[train_indices]
    X_test = X[test_indices]

    train_counts = dict(Counter(y_train.tolist()))
    test_counts = dict(Counter(y_test.tolist()))

    print("Full counts:", dict(full_counts))
    print(
        f"Train: {len(train_indices)} "
        f"({args.train_ratio:.0%}) {train_counts}"
    )
    print(
        f"Test:  {len(test_indices)} "
        f"({1.0 - args.train_ratio:.0%}) {test_counts}"
    )

    output = {
        "input": args.input_npz,
        "relations": relations,
        "split": {
            "train_ratio": args.train_ratio,
            "test_ratio": 1.0 - args.train_ratio,
            "seed": args.seed,
            "n_total": int(len(y)),
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "full_counts": dict(full_counts),
            "train_counts": train_counts,
            "test_counts": test_counts,
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
        },
        "layers": {},
    }

    for layer_position, layer in enumerate(layers):
        center, dirs = fit_codebook(
            X_train[:, layer_position, :],
            y_train,
            relations,
        )

        result = evaluate(
            X_test[:, layer_position, :],
            y_test,
            center,
            dirs,
            relations,
        )

        output["layers"][str(layer)] = result

        print(
            f"L{layer}: "
            f"acc={result['accuracy']:.3f} "
            f"margin={result['margin']:.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
