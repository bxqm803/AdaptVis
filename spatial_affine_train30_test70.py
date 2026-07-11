#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatial direction train/test probe.

Based on affine direction codebook:
    r = h(subject) - h(reference)

Train:
    30% samples
    compute global center
    compute one direction vector per relation

Test:
    70% samples
    cosine similarity with learned direction vectors
    argmax prediction

Supports:
    controlled_A
    controlled_B
    coco_two
    vg_two

Any model:
    llava
    qwen
    internvl
    etc.

Input format:
states.npz containing:
    relation_vectors
    relation
    decoder_block_index
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np


EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-npz",
        required=True
    )

    p.add_argument(
        "--relations",
        default="left,right,on,under"
    )

    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.3
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=5
    )

    p.add_argument(
        "--split-unit",
        choices=["sample"],
        default="sample"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1
    )

    p.add_argument(
        "--output",
        required=True
    )

    return p.parse_args()


def normalize(x):
    return x / max(np.linalg.norm(x), EPS)


def load_npz(path):

    with np.load(path, allow_pickle=True) as z:

        relation = np.asarray(
            [str(x) for x in z["relation"].tolist()],
            dtype=object
        )

        vectors = z["relation_vectors"].astype(
            np.float64
        )

        layers = [
            int(x)
            for x in z["decoder_block_index"].tolist()
        ]

    return relation, vectors, layers


def split_indices(n, ratio, seed):

    rng = random.Random(seed)

    ids = list(range(n))
    rng.shuffle(ids)

    train_n = int(n * ratio)

    return (
        np.asarray(ids[:train_n]),
        np.asarray(ids[train_n:])
    )


def fit_codebook(X, y, relations):

    center = X.mean(axis=0)

    Xc = X - center

    dirs = []

    for r in relations:

        mask = y == r

        if mask.sum() == 0:
            raise RuntimeError(
                f"Missing relation {r}"
            )

        v = Xc[mask].mean(axis=0)

        dirs.append(
            normalize(v)
        )

    return center, np.stack(dirs)


def predict(X, center, dirs):

    Xc = X - center

    Xc = Xc / np.maximum(
        np.linalg.norm(Xc, axis=1, keepdims=True),
        EPS
    )

    return np.argmax(
        Xc @ dirs.T,
        axis=1
    )


def evaluate(
    X,
    y,
    layers,
    train_idx,
    test_idx,
    relations
):

    result = {}

    y_train = y[train_idx]
    y_test = y[test_idx]

    for li, layer in enumerate(layers):

        center, dirs = fit_codebook(
            X[train_idx, li, :],
            y_train,
            relations
        )

        pred = predict(
            X[test_idx, li, :],
            center,
            dirs
        )

        gt = np.asarray(
            [
                relations.index(v)
                for v in y_test
            ]
        )

        acc = float(
            np.mean(pred == gt)
        )

        result[str(layer)] = {
            "accuracy": acc,
            "train_count": dict(
                Counter(y_train.tolist())
            ),
            "test_count": dict(
                Counter(y_test.tolist())
            )
        }

    return result


def main():

    args = parse_args()

    relations = [
        x.strip()
        for x in args.relations.split(",")
        if x.strip()
    ]

    labels, vectors, layers = load_npz(
        args.input_npz
    )

    mask = np.isin(
        labels,
        relations
    )

    labels = labels[mask]
    vectors = vectors[mask]

    summary = {
        "input": args.input_npz,
        "relations": relations,
        "train_ratio": args.train_ratio,
        "layers": {}
    }

    all_results = []

    for rep in range(args.repeats):

        train_idx, test_idx = split_indices(
            len(labels),
            args.train_ratio,
            args.seed + rep
        )

        result = evaluate(
            vectors,
            labels,
            layers,
            train_idx,
            test_idx,
            relations
        )

        all_results.append(result)

    for layer in layers:

        vals = [
            x[str(layer)]["accuracy"]
            for x in all_results
        ]

        summary["layers"][str(layer)] = {
            "accuracy_mean": float(np.mean(vals)),
            "accuracy_std": float(np.std(vals)),
            "repeat_accuracy": vals
        }

        print(
            f"L{layer}: "
            f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        )


    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "Saved:",
        out
    )


if __name__ == "__main__":
    main()
