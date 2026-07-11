#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-dataset spatial direction transfer.

Same logic as analyze_spatial_direction_train_test_pair.py:

relation vector:
    r = h_subject - h_reference

Train dataset:
    fit global center
    fit one normalized direction vector per relation

Test dataset:
    evaluate by cosine similarity + argmax

Example:
VG_two -> Controlled_A
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np

EPS = 1e-12


def args():
    p = argparse.ArgumentParser()

    p.add_argument("--train-npz", required=True)
    p.add_argument("--test-npz", required=True)

    p.add_argument(
        "--relations",
        default="left,right,on,under"
    )

    p.add_argument(
        "--feature-centering",
        choices=["global", "none"],
        default="global"
    )

    p.add_argument(
        "--layers",
        default="auto"
    )

    p.add_argument(
        "--output",
        required=True
    )

    return p.parse_args()


def normalize_relation(x):
    s = str(x).strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)

    aliases = {
        "above": "on",
        "below": "under",
        "left_of": "left",
        "right_of": "right",
        "on_top": "on",
        "on_top_of": "on",
        "underneath": "under",
    }

    return aliases.get(s, s)


def load_npz(path):

    with np.load(path, allow_pickle=True) as z:

        labels = np.asarray(
            [normalize_relation(x) for x in z["relation"].tolist()],
            dtype=object
        )

        X = z["relation_vectors"].astype(np.float64)

        layers = [
            int(x)
            for x in z["decoder_block_index"].tolist()
        ]

    return labels, X, layers


def normalize_rows(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        EPS
    )


def fit_codebook(X, y, relations, centering):

    if centering == "global":
        center = X.mean(axis=0)
    else:
        center = np.zeros(X.shape[1])

    Xc = X - center

    dirs = []

    for r in relations:

        mask = y == r

        if mask.sum() == 0:
            raise RuntimeError(
                f"Missing relation {r}"
            )

        d = Xc[mask].mean(axis=0)

        d = d / max(np.linalg.norm(d), EPS)

        dirs.append(d)

    return center, np.stack(dirs)


def evaluate(X_test, y_test, center, dirs, relations):

    scores = normalize_rows(
        X_test - center
    ) @ dirs.T

    pred = np.argmax(scores, axis=1)

    gt = np.asarray(
        [relations.index(x) for x in y_test]
    )

    acc = float(
        np.mean(pred == gt)
    )

    tmp = scores.copy()

    true_score = scores[
        np.arange(len(gt)),
        gt
    ]

    tmp[
        np.arange(len(gt)),
        gt
    ] = -np.inf

    margin = true_score - tmp.max(axis=1)

    return {
        "accuracy": acc,
        "signed_margin": float(margin.mean()),
        "n": int(len(gt)),
        "counts": dict(Counter(y_test.tolist()))
    }


def main():

    a = args()

    relations = [
        normalize_relation(x)
        for x in a.relations.split(",")
    ]

    y_train, X_train, layers_train = load_npz(
        a.train_npz
    )

    y_test, X_test, layers_test = load_npz(
        a.test_npz
    )

    layers = sorted(
        set(layers_train).intersection(
            set(layers_test)
        )
    )

    result = {
        "train": a.train_npz,
        "test": a.test_npz,
        "relations": relations,
        "layers": {}
    }

    for i, layer in enumerate(layers):

        ti = layers_train.index(layer)
        vi = layers_test.index(layer)

        center, dirs = fit_codebook(
            X_train[:, ti, :],
            y_train,
            relations,
            a.feature_centering
        )

        r = evaluate(
            X_test[:, vi, :],
            y_test,
            center,
            dirs,
            relations
        )

        result["layers"][str(layer)] = r

        print(
            f"L{layer}: "
            f"acc={r['accuracy']:.3f}, "
            f"margin={r['signed_margin']:.3f}"
        )


    out = Path(a.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("Saved:", out)


if __name__ == "__main__":
    main()
