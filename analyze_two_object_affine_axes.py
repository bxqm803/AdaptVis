#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
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
        y = np.array([norm_rel(x) for x in z["relation"].tolist()], dtype=object)
        X = z["relation_vectors"].astype(np.float64)
        layers = [int(x) for x in z["decoder_block_index"].tolist()]
    return y, X, layers

def norm_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)

def fit_codebook(X, y, relations):
    center = X.mean(axis=0)
    dirs = []
    Xc = X - center
    for r in relations:
        m = y == r
        if m.sum() == 0:
            raise RuntimeError(f"Missing relation {r}")
        d = Xc[m].mean(axis=0)
        d /= max(np.linalg.norm(d), EPS)
        dirs.append(d)
    return center, np.stack(dirs)

def evaluate(X, y, center, dirs, relations):
    score = norm_rows(X-center) @ dirs.T
    pred = np.argmax(score, axis=1)
    gt = np.array([relations.index(x) for x in y])
    acc = float(np.mean(pred == gt))
    ts = score[np.arange(len(gt)), gt]
    tmp = score.copy()
    tmp[np.arange(len(gt)), gt] = -np.inf
    return {
        "accuracy": acc,
        "margin": float((ts - tmp.max(axis=1)).mean()),
        "n": int(len(gt)),
        "counts": dict(Counter(y.tolist()))
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-npz", required=True)
    p.add_argument("--relations", default="left,right,above,below")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    relations = [norm_rel(x) for x in a.relations.split(",")]

    y, X, layers = load_npz(a.input_npz)

    mask = np.isin(y, relations)
    y, X = y[mask], X[mask]

    print("Counts:", Counter(y.tolist()))

    out = {"input": a.input_npz, "relations": relations, "layers": {}}

    for i, layer in enumerate(layers):
        center, dirs = fit_codebook(X[:, i, :], y, relations)
        r = evaluate(X[:, i, :], y, center, dirs, relations)
        out["layers"][str(layer)] = r
        print(f"L{layer}: acc={r['accuracy']:.3f} margin={r['margin']:.3f}")

    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", a.output)

if __name__ == "__main__":
    main()
