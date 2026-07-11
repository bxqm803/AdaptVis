#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30/70 split spatial direction probe.

Logic:
1. Randomly split one dataset into 30% train and 70% test.
2. Train split:
   - compute four relation prototypes:
       left, right, above, below
   - compute relation direction vectors:
       right-left
       above-below
3. Test split:
   - classify by cosine similarity to four learned direction prototypes.
   - report:
       * four-way accuracy
       * left/right accuracy
       * above/below accuracy

Works for:
    controlled_A
    controlled_B
    coco_two
    vg_two

and all extracted models.

Input:
states.npz from extract_two_object_relation_states.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np


RELATIONS = [
    "left",
    "right",
    "above",
    "below",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-npz",
        required=True
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )

    parser.add_argument(
        "--output",
        required=True
    )

    return parser.parse_args()


def normalize(x):
    n = np.linalg.norm(x)
    return x / max(n, 1e-8)


def load_npz(path):

    with np.load(path, allow_pickle=True) as f:

        metadata = json.loads(
            str(f["metadata_json"].item())
        )

        labels = np.asarray(
            [str(x) for x in f["relation"].tolist()],
            dtype=object
        )

        vectors = f["relation_vectors"].astype(
            np.float32
        )

        blocks = [
            int(x)
            for x in f["decoder_block_index"].tolist()
        ]

    return metadata, labels, vectors, blocks


def split_data(labels, ratio, seed):

    rng = np.random.default_rng(seed)

    idx = np.arange(len(labels))
    rng.shuffle(idx)

    n_train = int(len(idx) * ratio)

    return idx[:n_train], idx[n_train:]


def build_relation_prototypes(x, y):

    proto = {}

    for r in RELATIONS:
        if np.sum(y == r) == 0:
            continue

        proto[r] = np.mean(
            x[y == r],
            axis=0
        )

        proto[r] = normalize(proto[r])

    return proto


def cosine_predict(x, proto):

    names = list(proto.keys())

    P = np.stack(
        [proto[n] for n in names],
        axis=0
    )

    x = x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-8
    )

    sim = x @ P.T

    pred = np.asarray(
        [names[i] for i in np.argmax(sim, axis=1)],
        dtype=object
    )

    return pred


def accuracy(a, b):

    return float(
        np.mean(a == b)
    )


def evaluate_layer(x_train, y_train, x_test, y_test):

    proto = build_relation_prototypes(
        x_train,
        y_train
    )

    pred = cosine_predict(
        x_test,
        proto
    )

    result = {}

    result["four_way_accuracy"] = accuracy(
        y_test,
        pred
    )

    mask_lr = np.isin(
        y_test,
        ["left", "right"]
    )

    if np.any(mask_lr):
        result["left_right_accuracy"] = accuracy(
            y_test[mask_lr],
            pred[mask_lr]
        )

    mask_ud = np.isin(
        y_test,
        ["above", "below"]
    )

    if np.any(mask_ud):
        result["above_below_accuracy"] = accuracy(
            y_test[mask_ud],
            pred[mask_ud]
        )

    result["test_count"] = int(len(y_test))

    return result


def main():

    args = parse_args()

    meta, labels, vectors, blocks = load_npz(
        args.input_npz
    )

    train_idx, test_idx = split_data(
        labels,
        args.train_ratio,
        args.seed
    )

    summary = {
        "metadata": meta,
        "train_ratio": args.train_ratio,
        "label_count": dict(Counter(labels.tolist())),
        "layers": {}
    }


    for i, block in enumerate(blocks):

        train_x = vectors[
            train_idx,
            i,
            :
        ]

        test_x = vectors[
            test_idx,
            i,
            :
        ]

        result = evaluate_layer(
            train_x,
            labels[train_idx],
            test_x,
            labels[test_idx]
        )

        summary["layers"][str(block)] = result

        print(
            f"L{block}: "
            f"four={result.get('four_way_accuracy',0):.3f} "
            f"LR={result.get('left_right_accuracy',0):.3f} "
            f"UD={result.get('above_below_accuracy',0):.3f}"
        )


    out = Path(args.output)

    if out.suffix != ".json":
        out = out.with_suffix(".json")

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

    print("Saved:", out)


if __name__ == "__main__":
    main()
