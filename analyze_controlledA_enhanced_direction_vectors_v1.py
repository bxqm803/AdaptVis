#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_controlledA_enhanced_direction_vectors_v1.py

Use the SUCCESSFUL Controlled_Images_A AdaptVis run to learn better
last-token relation directions.

Input:
  NPZ produced by analyze_llava_controlledA_lasttoken_reproduce_v5.py --save-vectors

Expected keys:
  relation, layers, B_last, C_last, transition_group
Optional:
  delta_adapt, selected_weight, sample_index

Controlled-A labels:
  left / right / on / under

Direction definition:
  center_l = mean(x_train)
  d_{r,l} = normalize(mean(x_train[y=r] - center_l))

Comparisons:
  1) native_correct_on_native
     fit B-correct TRAIN states, test B states
  2) enhanced_correct_on_enhanced
     fit C-correct TRAIN states, test C states   <-- primary
  3) enhanced_correct_on_native
     fit C-correct TRAIN states, test B states
  4) enhanced_all_on_enhanced
     fit all C TRAIN states, test C states
  5) adapt_delta_correct_on_delta
     fit C-B on C-correct TRAIN samples, test C-B
  6) adapt_delta_W2C_on_delta
     fit C-B only on W2C TRAIN samples, test C-B

Also reports:
  - per-relation train counts
  - per-relation test accuracy, especially "on"
  - direction stability across repeated TRAIN/TEST splits
  - alignment between native B-correct and enhanced C-correct directions

Example:
python analyze_controlledA_enhanced_direction_vectors_v1.py \
  --vectors output/llava_controlledA_lasttoken_reproduce_v5/lasttoken_vectors_ABC.npz \
  --train-ratio 0.30 \
  --repeats 20 \
  --min-train-per-class 2 \
  --output-dir output/controlledA_enhanced_direction_vectors_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

RELATIONS = ("left", "right", "on", "under")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vectors",
        default=(
            "output/llava_controlledA_lasttoken_reproduce_v5/"
            "lasttoken_vectors_ABC.npz"
        ),
    )
    p.add_argument("--layers", default="all")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-train-per-class", type=int, default=2)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--save-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def safe_mean(values: Iterable[Any]) -> float:
    vals = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: Iterable[Any]) -> float:
    vals = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.std(vals)) if vals else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def parse_layers(spec: str, available_layers: np.ndarray):
    available_layers = np.asarray(available_layers, dtype=np.int64)
    layer_to_pos = {
        int(layer): pos
        for pos, layer in enumerate(available_layers.tolist())
    }

    raw = str(spec).strip().lower()
    if raw == "all":
        selected = available_layers.copy()
    else:
        vals = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                step = 1 if b >= a else -1
                vals.extend(range(a, b + step, step))
            else:
                vals.append(int(part))
        selected = np.asarray(list(dict.fromkeys(vals)), dtype=np.int64)

    missing = [
        int(layer)
        for layer in selected.tolist()
        if int(layer) not in layer_to_pos
    ]
    if missing:
        raise ValueError(
            f"Requested layers missing from cache: {missing}; "
            f"available={available_layers.tolist()}"
        )

    positions = np.asarray(
        [layer_to_pos[int(layer)] for layer in selected.tolist()],
        dtype=np.int64,
    )
    return selected, positions


def stratified_split(labels: np.ndarray, train_ratio: float, seed: int):
    rng = random.Random(int(seed))
    train, test = [], []

    for relation in RELATIONS:
        ids = np.flatnonzero(labels == relation).tolist()
        if len(ids) < 2:
            raise RuntimeError(f"Need >=2 samples for {relation}, got {len(ids)}")

        rng.shuffle(ids)
        n_train = int(round(len(ids) * float(train_ratio)))
        n_train = max(1, min(n_train, len(ids) - 1))
        train.extend(ids[:n_train])
        test.extend(ids[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def fit_directions(
    X: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    *,
    min_train_per_class: int,
):
    fit_idx = np.asarray(fit_idx, dtype=np.int64)

    counts = {
        r: int(np.sum(labels[fit_idx] == r))
        for r in RELATIONS
    }

    missing = [
        r
        for r in RELATIONS
        if counts[r] < int(min_train_per_class)
    ]

    if missing:
        return {
            "valid": False,
            "counts": counts,
            "missing": missing,
            "center": None,
            "directions": None,
        }

    Xfit = np.asarray(X[fit_idx], dtype=np.float64)
    center = Xfit.mean(axis=0)
    centered = Xfit - center
    yfit = labels[fit_idx]

    directions = []
    for relation in RELATIONS:
        d = centered[yfit == relation].mean(axis=0)
        norm = float(np.linalg.norm(d))
        if norm <= EPS:
            return {
                "valid": False,
                "counts": counts,
                "missing": [f"{relation}:near_zero"],
                "center": None,
                "directions": None,
            }
        directions.append(d / norm)

    return {
        "valid": True,
        "counts": counts,
        "missing": [],
        "center": center.astype(np.float32),
        "directions": np.stack(directions, axis=0).astype(np.float32),
    }


def score_directions(
    X: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    centered = (
        np.asarray(X, dtype=np.float64)
        - np.asarray(center, dtype=np.float64)
    )
    return normalize_rows(centered) @ normalize_rows(directions).T


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    *,
    eligible_test_mask: Optional[np.ndarray] = None,
):
    test_idx = np.asarray(test_idx, dtype=np.int64)

    if eligible_test_mask is not None:
        keep = np.asarray(eligible_test_mask[test_idx], dtype=bool)
        test_idx = test_idx[keep]
        scores = scores[keep]

    if len(test_idx) == 0:
        out = {
            "N_test": 0,
            "accuracy": float("nan"),
            "gt_cosine": float("nan"),
            "gt_margin": float("nan"),
        }
        for r in RELATIONS:
            out[f"{r}_N"] = 0
            out[f"{r}_accuracy"] = float("nan")
        return out

    gt_ids = np.asarray(
        [REL_TO_ID[str(x)] for x in labels[test_idx]],
        dtype=np.int64,
    )
    pred_ids = np.argmax(scores, axis=1)
    row_ids = np.arange(len(test_idx), dtype=np.int64)

    gt_score = scores[row_ids, gt_ids]
    masked = scores.copy()
    masked[row_ids, gt_ids] = -np.inf
    best_other = np.max(masked, axis=1)
    margin = gt_score - best_other

    out = {
        "N_test": int(len(test_idx)),
        "accuracy": float(np.mean(pred_ids == gt_ids)),
        "gt_cosine": float(np.mean(gt_score)),
        "gt_margin": float(np.mean(margin)),
    }

    for r in RELATIONS:
        rid = REL_TO_ID[r]
        mask = gt_ids == rid
        n = int(mask.sum())
        out[f"{r}_N"] = n
        out[f"{r}_accuracy"] = (
            float(np.mean(pred_ids[mask] == gt_ids[mask]))
            if n else float("nan")
        )

    return out


def mean_pairwise_cosine(vectors):
    vectors = [np.asarray(v, dtype=np.float64) for v in vectors]
    if len(vectors) < 2:
        return float("nan"), float("nan"), len(vectors)

    vals = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vals.append(cosine(vectors[i], vectors[j]))

    return safe_mean(vals), safe_std(vals), len(vectors)


def main():
    args = parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.repeats < 1:
        raise ValueError("--repeats must be >=1")
    if args.min_train_per_class < 1:
        raise ValueError("--min-train-per-class must be >=1")

    src = Path(args.vectors)
    if not src.exists():
        raise FileNotFoundError(f"Missing vectors file: {src}")

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(src, allow_pickle=True) as z:
        required = {
            "relation",
            "layers",
            "B_last",
            "C_last",
            "transition_group",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(
                f"NPZ missing required keys={missing}; available={z.files}"
            )

        labels = np.asarray(z["relation"], dtype=object)
        all_layers = np.asarray(z["layers"], dtype=np.int64)
        B = np.asarray(z["B_last"], dtype=np.float32)
        C = np.asarray(z["C_last"], dtype=np.float32)
        delta = (
            np.asarray(z["delta_adapt"], dtype=np.float32)
            if "delta_adapt" in z.files
            else (C - B).astype(np.float32)
        )
        transition = np.asarray(z["transition_group"], dtype=object)
        selected_weight = (
            np.asarray(z["selected_weight"], dtype=np.float32)
            if "selected_weight" in z.files
            else np.full(len(labels), np.nan, dtype=np.float32)
        )
        sample_index = (
            np.asarray(z["sample_index"], dtype=np.int64)
            if "sample_index" in z.files
            else np.arange(len(labels), dtype=np.int64)
        )

    labels = np.asarray(
        [str(x).strip().lower() for x in labels.tolist()],
        dtype=object,
    )

    unexpected = sorted(set(labels.tolist()) - set(RELATIONS))
    if unexpected:
        raise RuntimeError(
            f"Expected Controlled-A labels {RELATIONS}, got {unexpected}"
        )

    if B.shape != C.shape or B.shape != delta.shape:
        raise RuntimeError(
            f"State shape mismatch B={B.shape}, C={C.shape}, delta={delta.shape}"
        )
    if B.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {B.shape}")

    selected_layers, layer_pos = parse_layers(args.layers, all_layers)
    B = B[:, layer_pos, :]
    C = C[:, layer_pos, :]
    delta = delta[:, layer_pos, :]

    N = len(labels)

    # From v5 transition labels.
    B_correct = np.isin(transition, ["C2W", "C2C"])
    C_correct = np.isin(transition, ["W2C", "C2C"])
    W2C = transition == "W2C"

    print("\n" + "=" * 150)
    print("CONTROLLED-A: NATIVE vs ENHANCED DIRECTION EXTRACTION")
    print("=" * 150)
    print(f"vectors={src}")
    print(f"N={N} | layers={selected_layers.tolist()}")
    print(
        f"B correct={int(B_correct.sum())}/{N} ({B_correct.mean():.4f}) | "
        f"C correct={int(C_correct.sum())}/{N} ({C_correct.mean():.4f})"
    )
    print("Per-relation correct counts:")

    relation_rows = []
    for relation in RELATIONS:
        mask = labels == relation
        row = {
            "relation": relation,
            "N": int(mask.sum()),
            "B_correct": int(np.sum(mask & B_correct)),
            "C_correct": int(np.sum(mask & C_correct)),
            "B_correct_rate": float(np.mean(B_correct[mask])),
            "C_correct_rate": float(np.mean(C_correct[mask])),
        }
        relation_rows.append(row)
        print(
            f"  {relation:>5s}: N={row['N']:3d} | "
            f"B_correct={row['B_correct']:3d} ({row['B_correct_rate']:.3f}) | "
            f"C_correct={row['C_correct']:3d} ({row['C_correct_rate']:.3f})"
        )
    print("=" * 150)

    write_csv(outdir / "relation_correct_counts.csv", relation_rows)

    methods = {
        "native_correct_on_native": {
            "fit_X": B,
            "fit_eligible": B_correct,
            "test_X": B,
            "test_eligible": None,
            "source_name": "B_correct",
        },
        "native_correct_on_native_correct_test": {
            "fit_X": B,
            "fit_eligible": B_correct,
            "test_X": B,
            "test_eligible": B_correct,
            "source_name": "B_correct",
        },
        "enhanced_correct_on_enhanced": {
            "fit_X": C,
            "fit_eligible": C_correct,
            "test_X": C,
            "test_eligible": None,
            "source_name": "C_correct",
        },
        "enhanced_correct_on_enhanced_correct_test": {
            "fit_X": C,
            "fit_eligible": C_correct,
            "test_X": C,
            "test_eligible": C_correct,
            "source_name": "C_correct",
        },
        "enhanced_correct_on_native": {
            "fit_X": C,
            "fit_eligible": C_correct,
            "test_X": B,
            "test_eligible": None,
            "source_name": "C_correct",
        },
        "enhanced_all_on_enhanced": {
            "fit_X": C,
            "fit_eligible": np.ones(N, dtype=bool),
            "test_X": C,
            "test_eligible": None,
            "source_name": "C_all",
        },
        "adapt_delta_correct_on_delta": {
            "fit_X": delta,
            "fit_eligible": C_correct,
            "test_X": delta,
            "test_eligible": None,
            "source_name": "delta_C_correct",
        },
        "adapt_delta_W2C_on_delta": {
            "fit_X": delta,
            "fit_eligible": W2C,
            "test_X": delta,
            "test_eligible": None,
            "source_name": "delta_W2C",
        },
    }

    raw_rows = []
    learned_direction_bank = {}
    learned_count_bank = {}

    for repeat in range(args.repeats):
        train_idx, test_idx = stratified_split(
            labels,
            args.train_ratio,
            args.seed + repeat,
        )

        for li, layer in enumerate(selected_layers.tolist()):
            source_fits = {}
            unique_sources = {}

            for spec in methods.values():
                unique_sources.setdefault(spec["source_name"], spec)

            for source_name, spec in unique_sources.items():
                eligible = np.asarray(spec["fit_eligible"], dtype=bool)
                fit_idx = train_idx[eligible[train_idx]]

                fit_result = fit_directions(
                    spec["fit_X"][:, li, :],
                    labels,
                    fit_idx,
                    min_train_per_class=args.min_train_per_class,
                )

                source_fits[source_name] = fit_result

                for relation in RELATIONS:
                    learned_count_bank[
                        (source_name, repeat, relation)
                    ] = fit_result["counts"][relation]

                if fit_result["valid"]:
                    for relation in RELATIONS:
                        rid = REL_TO_ID[relation]
                        learned_direction_bank[
                            (source_name, repeat, int(layer), relation)
                        ] = fit_result["directions"][rid].copy()

            for method_name, spec in methods.items():
                source_name = spec["source_name"]
                fit_result = source_fits[source_name]

                base_row = {
                    "repeat": repeat,
                    "layer": int(layer),
                    "method": method_name,
                    "source": source_name,
                    "fit_valid": int(bool(fit_result["valid"])),
                    "fit_missing": ";".join(str(x) for x in fit_result["missing"]),
                    "train_left": fit_result["counts"]["left"],
                    "train_right": fit_result["counts"]["right"],
                    "train_on": fit_result["counts"]["on"],
                    "train_under": fit_result["counts"]["under"],
                }

                if not fit_result["valid"]:
                    raw_rows.append({
                        **base_row,
                        "N_test": 0,
                        "accuracy": float("nan"),
                        "gt_cosine": float("nan"),
                        "gt_margin": float("nan"),
                    })
                    continue

                scores = score_directions(
                    spec["test_X"][test_idx, li, :],
                    fit_result["center"],
                    fit_result["directions"],
                )

                metrics = evaluate_scores(
                    scores,
                    labels,
                    test_idx,
                    eligible_test_mask=spec["test_eligible"],
                )
                raw_rows.append({**base_row, **metrics})

    write_csv(outdir / "repeat_layer_metrics.csv", raw_rows)

    summary_rows = []
    for method_name in methods:
        for layer in selected_layers.tolist():
            rows = [
                r for r in raw_rows
                if r["method"] == method_name
                and int(r["layer"]) == int(layer)
            ]
            valid_rows = [r for r in rows if int(r["fit_valid"]) == 1]

            item = {
                "method": method_name,
                "layer": int(layer),
                "valid_repeats": len(valid_rows),
                "valid_fraction": len(valid_rows) / max(len(rows), 1),
                "accuracy_mean": safe_mean(
                    r.get("accuracy", np.nan) for r in valid_rows
                ),
                "accuracy_std": safe_std(
                    r.get("accuracy", np.nan) for r in valid_rows
                ),
                "gt_cosine_mean": safe_mean(
                    r.get("gt_cosine", np.nan) for r in valid_rows
                ),
                "gt_margin_mean": safe_mean(
                    r.get("gt_margin", np.nan) for r in valid_rows
                ),
                "train_left_mean": safe_mean(r["train_left"] for r in rows),
                "train_right_mean": safe_mean(r["train_right"] for r in rows),
                "train_on_mean": safe_mean(r["train_on"] for r in rows),
                "train_under_mean": safe_mean(r["train_under"] for r in rows),
            }

            for relation in RELATIONS:
                item[f"{relation}_accuracy_mean"] = safe_mean(
                    r.get(f"{relation}_accuracy", np.nan)
                    for r in valid_rows
                )

            summary_rows.append(item)

    write_csv(outdir / "layer_summary.csv", summary_rows)

    stability_rows = []
    source_names = sorted({
        spec["source_name"]
        for spec in methods.values()
    })

    for source_name in source_names:
        for layer in selected_layers.tolist():
            for relation in RELATIONS:
                vecs = []
                counts = []

                for repeat in range(args.repeats):
                    key = (source_name, repeat, int(layer), relation)
                    if key in learned_direction_bank:
                        vecs.append(learned_direction_bank[key])

                    ck = (source_name, repeat, relation)
                    if ck in learned_count_bank:
                        counts.append(learned_count_bank[ck])

                pair_cos, pair_std, n_valid = mean_pairwise_cosine(vecs)

                stability_rows.append({
                    "source": source_name,
                    "layer": int(layer),
                    "relation": relation,
                    "valid_direction_repeats": n_valid,
                    "mean_train_count": safe_mean(counts),
                    "pairwise_direction_cosine_mean": pair_cos,
                    "pairwise_direction_cosine_std": pair_std,
                })

    write_csv(outdir / "direction_stability.csv", stability_rows)

    alignment_rows = []
    for repeat in range(args.repeats):
        for layer in selected_layers.tolist():
            for relation in RELATIONS:
                kb = ("B_correct", repeat, int(layer), relation)
                kc = ("C_correct", repeat, int(layer), relation)
                if kb in learned_direction_bank and kc in learned_direction_bank:
                    alignment_rows.append({
                        "repeat": repeat,
                        "layer": int(layer),
                        "relation": relation,
                        "cos_native_enhanced": cosine(
                            learned_direction_bank[kb],
                            learned_direction_bank[kc],
                        ),
                    })

    write_csv(
        outdir / "native_vs_enhanced_alignment.csv",
        alignment_rows,
    )

    if args.save_directions:
        direction_dir = outdir / "learned_directions"
        direction_dir.mkdir(parents=True, exist_ok=True)

        for source_name in source_names:
            for repeat in range(args.repeats):
                layer_ids = []
                direction_layers = []

                for layer in selected_layers.tolist():
                    dirs = []
                    valid = True

                    for relation in RELATIONS:
                        key = (
                            source_name,
                            repeat,
                            int(layer),
                            relation,
                        )
                        if key not in learned_direction_bank:
                            valid = False
                            break
                        dirs.append(learned_direction_bank[key])

                    if valid:
                        layer_ids.append(int(layer))
                        direction_layers.append(
                            np.stack(dirs, axis=0).astype(np.float32)
                        )

                if layer_ids:
                    np.savez_compressed(
                        direction_dir / f"{source_name}_repeat{repeat}.npz",
                        relation_order=np.asarray(RELATIONS, dtype=object),
                        decoder_block_index=np.asarray(layer_ids, dtype=np.int64),
                        directions=np.stack(direction_layers, axis=0),
                        source=np.asarray(source_name, dtype=object),
                        repeat=np.asarray(repeat, dtype=np.int64),
                    )

    lookup = {
        (r["method"], int(r["layer"])): r
        for r in summary_rows
    }

    def get(method, layer, key):
        row = lookup.get((method, int(layer)))
        return float(row.get(key, np.nan)) if row else float("nan")

    print("\n" + "=" * 190)
    print("DIRECTION QUALITY: NATIVE vs SUCCESSFULLY ENHANCED")
    print("=" * 190)
    print(
        f"{'layer':>5s} | "
        f"{'Bcorr->B':>9s} | "
        f"{'Ccorr->C':>9s} | "
        f"{'Ccorr->B':>9s} | "
        f"{'Call->C':>9s} | "
        f"{'dCcorr->d':>10s} | "
        f"{'B_On':>7s} | "
        f"{'C_On':>7s} | "
        f"{'Bfit%':>7s} | "
        f"{'Cfit%':>7s}"
    )
    print("-" * 190)

    for layer in selected_layers.tolist():
        print(
            f"L{int(layer):02d}   | "
            f"{get('native_correct_on_native', layer, 'accuracy_mean'):9.4f} | "
            f"{get('enhanced_correct_on_enhanced', layer, 'accuracy_mean'):9.4f} | "
            f"{get('enhanced_correct_on_native', layer, 'accuracy_mean'):9.4f} | "
            f"{get('enhanced_all_on_enhanced', layer, 'accuracy_mean'):9.4f} | "
            f"{get('adapt_delta_correct_on_delta', layer, 'accuracy_mean'):10.4f} | "
            f"{get('native_correct_on_native', layer, 'on_accuracy_mean'):7.4f} | "
            f"{get('enhanced_correct_on_enhanced', layer, 'on_accuracy_mean'):7.4f} | "
            f"{get('native_correct_on_native', layer, 'valid_fraction'):7.3f} | "
            f"{get('enhanced_correct_on_enhanced', layer, 'valid_fraction'):7.3f}"
        )

    print("=" * 190)

    enhanced_rows = [
        r for r in summary_rows
        if r["method"] == "enhanced_correct_on_enhanced"
        and math.isfinite(float(r["accuracy_mean"]))
    ]

    if enhanced_rows:
        best = max(enhanced_rows, key=lambda r: float(r["accuracy_mean"]))
        best_layer = int(best["layer"])

        print(
            "\nBEST enhanced_correct_on_enhanced: "
            f"L{best_layer:02d} "
            f"acc={float(best['accuracy_mean']):.4f} "
            f"on_acc={float(best['on_accuracy_mean']):.4f} "
            f"GT_margin={float(best['gt_margin_mean']):+.4f} "
            f"valid={int(best['valid_repeats'])}/{args.repeats}"
        )

        for source in [
            "B_correct",
            "C_correct",
            "C_all",
            "delta_C_correct",
        ]:
            rows = [
                r for r in stability_rows
                if r["source"] == source
                and int(r["layer"]) == best_layer
                and r["relation"] == "on"
            ]
            if rows:
                r = rows[0]
                print(
                    f"On stability @L{best_layer:02d} {source:>15s}: "
                    f"train_count={float(r['mean_train_count']):.1f}, "
                    f"repeat_cos={float(r['pairwise_direction_cosine_mean']):.4f}, "
                    f"valid={int(r['valid_direction_repeats'])}/{args.repeats}"
                )

    metadata = {
        "source_vectors": str(src),
        "N": N,
        "relations": list(RELATIONS),
        "selected_layers": selected_layers.tolist(),
        "train_ratio": args.train_ratio,
        "repeats": args.repeats,
        "seed": args.seed,
        "min_train_per_class": args.min_train_per_class,
        "B_correct_N": int(B_correct.sum()),
        "C_correct_N": int(C_correct.sum()),
        "primary_method": "enhanced_correct_on_enhanced",
        "direction_definition": (
            "centered mean directions fitted on TRAIN only; "
            "four independent Controlled-A directions"
        ),
    }

    (outdir / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[saved] {outdir / 'layer_summary.csv'}")
    print(f"[saved] {outdir / 'direction_stability.csv'}")
    print(f"[saved] {outdir / 'native_vs_enhanced_alignment.csv'}")
    if args.save_directions:
        print(f"[saved] {outdir / 'learned_directions'}")


if __name__ == "__main__":
    main()
