#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, defaultdict
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


def load_npz(path, split_unit):
    with np.load(path, allow_pickle=True) as z:
        required = {"relation", "relation_vectors", "decoder_block_index"}
        missing = required.difference(z.files)
        if missing:
            raise KeyError(
                f"Missing required NPZ keys: {sorted(missing)}; "
                f"available={sorted(z.files)}"
            )

        y = np.array(
            [norm_rel(x) for x in z["relation"].tolist()],
            dtype=object,
        )
        X = np.asarray(z["relation_vectors"], dtype=np.float64)
        layers = [int(x) for x in z["decoder_block_index"].tolist()]

        if split_unit == "group":
            if "group" not in z.files:
                raise KeyError(
                    "--split-unit group requires the standard NPZ key "
                    "'group'. Controlled-A states already contain it. "
                    "For the 800-image VG set, use --split-unit sample."
                )
            groups = np.asarray(
                [str(x) for x in z["group"].tolist()],
                dtype=object,
            )
        else:
            groups = np.asarray(
                [f"sample::{i}" for i in range(len(y))],
                dtype=object,
            )

    if X.ndim != 3:
        raise ValueError(
            f"relation_vectors must be [N,L,H], got {X.shape}"
        )
    if X.shape[0] != len(y):
        raise ValueError(
            f"N mismatch: relation={len(y)}, vectors={X.shape[0]}"
        )
    if X.shape[1] != len(layers):
        raise ValueError(
            f"L mismatch: decoder_block_index={len(layers)}, "
            f"vectors={X.shape[1]}"
        )
    if len(groups) != len(y):
        raise ValueError(
            f"group mismatch: group={len(groups)}, relation={len(y)}"
        )

    return y, X, layers, groups


def norm_rows(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        EPS,
    )


def fit_codebook(X, y, relations):
    # 与用户给出的原版完全一致。
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
    # 与用户给出的原版完全一致。
    score = norm_rows(X - center) @ dirs.T
    pred = np.argmax(score, axis=1)
    gt = np.array(
        [relations.index(x) for x in y],
        dtype=np.int64,
    )

    acc = float(np.mean(pred == gt))
    true_score = score[np.arange(len(gt)), gt]
    tmp = score.copy()
    tmp[np.arange(len(gt)), gt] = -np.inf

    return {
        "accuracy": acc,
        "margin": float(
            (true_score - tmp.max(axis=1)).mean()
        ),
        "n": int(len(gt)),
        "counts": dict(Counter(y.tolist())),
    }


def group_signature(indices, y):
    """
    用一个 group 内的关系组成作为分层签名。

    Controlled-A:
      horizontal pair -> (left, right)
      vertical pair   -> (above, below)

    VG 采用 sample split 时，每个 group 只有一条：
      (left,) / (right,) / (above,) / (below,)
    """
    counts = Counter(y[indices].tolist())
    return tuple(sorted(counts.items()))


def stratified_group_split(
    y,
    groups,
    train_ratio,
    seed,
):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"train_ratio must be in (0,1), got {train_ratio}"
        )

    group_to_indices = defaultdict(list)
    for index, group in enumerate(groups.tolist()):
        group_to_indices[str(group)].append(index)

    if len(group_to_indices) < 2:
        raise RuntimeError(
            f"Need at least 2 groups, got {len(group_to_indices)}"
        )

    # 按 group 的标签组成分层，避免将 Controlled-A 的
    # horizontal/vertical group 比例切坏。
    strata = defaultdict(list)
    for group, indices in group_to_indices.items():
        signature = group_signature(indices, y)
        strata[signature].append(group)

    rng = np.random.default_rng(seed)
    train_groups = set()
    test_groups = set()

    for signature, stratum_groups in sorted(
        strata.items(),
        key=lambda item: str(item[0]),
    ):
        stratum_groups = list(stratum_groups)
        rng.shuffle(stratum_groups)

        n_groups = len(stratum_groups)
        if n_groups == 1:
            raise RuntimeError(
                "A group-label stratum has only one group and cannot "
                f"be split: signature={signature}"
            )

        n_train = int(round(n_groups * train_ratio))
        n_train = max(1, min(n_train, n_groups - 1))

        train_groups.update(stratum_groups[:n_train])
        test_groups.update(stratum_groups[n_train:])

    train_idx = np.asarray(
        [
            i
            for i, group in enumerate(groups.tolist())
            if str(group) in train_groups
        ],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [
            i
            for i, group in enumerate(groups.tolist())
            if str(group) in test_groups
        ],
        dtype=np.int64,
    )

    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(
            f"Internal split error: {len(overlap)} groups overlap"
        )

    if len(train_idx) + len(test_idx) != len(y):
        raise RuntimeError(
            "Internal split error: samples were lost or duplicated"
        )

    return train_idx, test_idx, train_groups, test_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-npz", required=True)
    p.add_argument(
        "--relations",
        default="left,right,above,below",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--split-unit",
        choices=("group", "sample"),
        default="group",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    relations = [
        norm_rel(x)
        for x in a.relations.split(",")
        if x.strip()
    ]
    if len(relations) != len(set(relations)):
        raise ValueError(
            f"Duplicate normalized relations: {relations}"
        )

    y, X, layers, groups = load_npz(
        a.input_npz,
        a.split_unit,
    )

    mask = np.isin(y, relations)
    y = y[mask]
    X = X[mask]
    groups = groups[mask]

    counts = Counter(y.tolist())
    missing = [
        relation
        for relation in relations
        if counts[relation] == 0
    ]
    if missing:
        raise RuntimeError(
            f"Missing requested relations: {missing}; counts={counts}"
        )

    (
        train_idx,
        test_idx,
        train_groups,
        test_groups,
    ) = stratified_group_split(
        y=y,
        groups=groups,
        train_ratio=a.train_ratio,
        seed=a.seed,
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Full counts:", dict(counts))
    print(
        f"Split unit: {a.split_unit} | "
        f"groups={len(set(groups.tolist()))} | "
        f"train groups={len(train_groups)} | "
        f"test groups={len(test_groups)} | overlap=0"
    )
    print(
        f"Train: {len(train_idx)} "
        f"{dict(Counter(y_train.tolist()))}"
    )
    print(
        f"Test:  {len(test_idx)} "
        f"{dict(Counter(y_test.tolist()))}"
    )

    out = {
        "input": a.input_npz,
        "relations": relations,
        "split": {
            "split_unit": a.split_unit,
            "train_ratio": a.train_ratio,
            "test_ratio": 1.0 - a.train_ratio,
            "seed": a.seed,
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_groups": int(len(set(groups.tolist()))),
            "n_train_groups": int(len(train_groups)),
            "n_test_groups": int(len(test_groups)),
            "group_overlap": 0,
            "train_counts": dict(
                Counter(y_train.tolist())
            ),
            "test_counts": dict(
                Counter(y_test.tolist())
            ),
        },
        "layers": {},
    }

    for i, layer in enumerate(layers):
        layer_X = X[:, i, :]

        # 原版 full-fit，必须复现旧脚本的结果。
        full_center, full_dirs = fit_codebook(
            layer_X,
            y,
            relations,
        )
        full_result = evaluate(
            layer_X,
            y,
            full_center,
            full_dirs,
            relations,
        )

        # 真正的 30% train -> 70% held-out test。
        train_center, train_dirs = fit_codebook(
            layer_X[train_idx],
            y_train,
            relations,
        )
        train_result = evaluate(
            layer_X[train_idx],
            y_train,
            train_center,
            train_dirs,
            relations,
        )
        test_result = evaluate(
            layer_X[test_idx],
            y_test,
            train_center,
            train_dirs,
            relations,
        )

        out["layers"][str(layer)] = {
            "full_data_reference": full_result,
            "train_fit_train_eval": train_result,
            "train_fit_test_eval": test_result,
        }

        print(
            f"L{layer}: "
            f"full={full_result['accuracy']:.3f} "
            f"train={train_result['accuracy']:.3f} "
            f"test={test_result['accuracy']:.3f} | "
            f"test_margin={test_result['margin']:.3f}"
        )

    output_path = Path(a.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
