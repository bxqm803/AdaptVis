#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run full all-layer/all-head attention-centroid accuracy on COCO_two (or VG_two)
by reusing the repository's existing:
    analyze_coco_centroid_generation_step1_v4.py

Why this wrapper:
- keeps the exact centroid definition already used in the repo;
- forces --layers all;
- exports per-(layer, head) centroid accuracy to a simple CSV;
- can either run Step-1 from scratch or only export an existing aggregate_metrics.npz.

Main outputs:
    <output-dir>/all_head_centroid_acc.csv
        CentroidACC on ALL samples processed by Step-1.

    <output-dir>/all_head_centroid_acc_passport_subset.csv
        If --passport-sample is supplied, recompute CentroidACC on exactly the
        unique SIDs contained in the head-passport sample CSV. This is the file
        to use when comparing against head_passport_summary.csv.

The all-sample target is:
    attention_original_accuracy[layer, head]

The matched-subset target is recomputed from each sample NPZ:
    original_object_prediction[layer, head] == gold relation

Both use the ORIGINAL prompt, not the original/swap average.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STEP1_SCRIPT = Path("analyze_coco_centroid_generation_step1_v4.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--model",
        required=True,
        choices=["qwen-3b", "qwen-7b"],
        help=(
            "Repo model alias. qwen-3b = Qwen2.5-VL-3B-Instruct; "
            "qwen-7b = Qwen2.5-VL-7B-Instruct."
        ),
    )
    p.add_argument(
        "--dataset",
        default="coco_two",
        choices=["coco_two", "vg_two"],
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default=None,
        help=(
            "Optional explicit prompt JSONL. If omitted, the underlying Step-1 "
            "script uses its dataset default."
        ),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Step-1 outputs and the exported all-head CSV.",
    )
    p.add_argument(
        "--passport-sample",
        default=None,
        help=(
            "Optional head_passport_sample.csv(.gz). If supplied, the wrapper "
            "recomputes all-head CentroidACC on exactly its unique sample SIDs. "
            "Use this for apples-to-apples comparison with DirectionACC."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full dataset.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing Step-1 output directory.",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help=(
            "Do not launch Step-1; only export CSV from an already existing "
            "<output-dir>/aggregate_metrics.npz."
        ),
    )
    p.add_argument(
        "--step1-script",
        default=str(STEP1_SCRIPT),
        help="Path to analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the Step-1 script.",
    )
    return p.parse_args()


def run_step1(args: argparse.Namespace) -> None:
    step1 = Path(args.step1_script)
    if not step1.exists():
        raise FileNotFoundError(
            f"Missing Step-1 script: {step1}\n"
            "Run this wrapper from the AdaptVis repo root, or pass --step1-script."
        )

    out_dir = Path(args.output_dir)

    cmd = [
        args.python,
        str(step1),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--model",
        args.model,
        "--device",
        args.device,
        "--attn-impl",
        "eager",
        "--layers",
        "all",
        # 2 is the practical minimum for the existing script's trace path.
        "--max-new-tokens",
        "2",
        "--print-every",
        "0",
        "--output-dir",
        str(out_dir),
    ]

    if args.prompt_jsonl is not None:
        cmd += ["--prompt-jsonl", args.prompt_jsonl]

    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]

    if args.overwrite:
        cmd += ["--overwrite"]

    print("=" * 120)
    print("RUNNING FULL ALL-HEAD CENTROID EXTRACTION")
    print("=" * 120)
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


def validate_shapes(
    layer_indices: np.ndarray,
    original_acc: np.ndarray,
    swapped_acc: np.ndarray | None,
    average_acc: np.ndarray | None,
    consistency: np.ndarray | None,
) -> None:
    if original_acc.ndim != 2:
        raise RuntimeError(
            "Expected attention_original_accuracy to be [n_layers, n_heads], "
            f"got shape={original_acc.shape}"
        )

    n_layers, n_heads = original_acc.shape

    if len(layer_indices) != n_layers:
        raise RuntimeError(
            f"layer_indices has {len(layer_indices)} entries but accuracy has "
            f"{n_layers} layers"
        )

    for name, x in [
        ("attention_swapped_accuracy", swapped_acc),
        ("attention_average_accuracy", average_acc),
        ("attention_relation_consistency", consistency),
    ]:
        if x is not None and x.shape != original_acc.shape:
            raise RuntimeError(
                f"{name} shape={x.shape} != original shape={original_acc.shape}"
            )

    print(
        f"Detected matrix: n_layers={n_layers}, n_heads={n_heads}, "
        f"total_heads={n_layers * n_heads}"
    )



RELATION_TO_INDEX = {
    "left": 0,
    "right": 1,
    "above": 2,
    "below": 3,
    "under": 3,
    "underneath": 3,
}


def export_passport_subset(
    args: argparse.Namespace,
    layer_indices: np.ndarray,
    n_heads: int,
) -> Path | None:
    """
    Recompute original-prompt centroid accuracy on exactly the SIDs present
    in a head-passport sample file.

    This avoids comparing:
        CentroidACC(all 440)
    against
        DirectionACC(passport test subset, e.g. 310).
    """
    if args.passport_sample is None:
        return None

    passport_path = Path(args.passport_sample)
    if not passport_path.exists():
        raise FileNotFoundError(f"Missing passport sample file: {passport_path}")

    hp = pd.read_csv(passport_path)

    if "sid" not in hp.columns or "relation" not in hp.columns:
        raise RuntimeError(
            f"{passport_path} must contain sid and relation columns. "
            f"Columns={list(hp.columns)}"
        )

    sid_rel = hp[["sid", "relation"]].drop_duplicates().copy()
    rel_counts = sid_rel.groupby("sid")["relation"].nunique()
    bad = rel_counts[rel_counts != 1]
    if len(bad):
        raise RuntimeError(
            "Some SIDs map to multiple gold relations in passport sample: "
            f"{bad.head(10).to_dict()}"
        )

    sid_to_relation = (
        sid_rel.drop_duplicates("sid")
        .set_index("sid")["relation"]
        .astype(str)
        .str.lower()
        .to_dict()
    )

    arrays_dir = Path(args.output_dir) / "sample_arrays"

    correct = np.zeros((len(layer_indices), n_heads), dtype=np.float64)
    count = 0
    missing = []

    for sid, relation in sorted(sid_to_relation.items()):
        relation = relation.strip().lower()

        if relation not in RELATION_TO_INDEX:
            raise RuntimeError(
                f"Unsupported relation {relation!r} for sid={sid}"
            )

        npz_path = arrays_dir / f"{int(sid)}.npz"

        if not npz_path.exists():
            missing.append(int(sid))
            continue

        z = np.load(npz_path, allow_pickle=False)

        if "original_object_prediction" not in z.files:
            raise RuntimeError(
                f"{npz_path} lacks original_object_prediction. "
                f"Available keys={sorted(z.files)}"
            )

        pred = np.asarray(z["original_object_prediction"])
        sample_layers = np.asarray(z["layer_indices"]).astype(int)

        expected_shape = (len(layer_indices), n_heads)
        if pred.shape != expected_shape:
            raise RuntimeError(
                f"{npz_path}: prediction shape={pred.shape}, "
                f"expected={expected_shape}"
            )

        if not np.array_equal(sample_layers, layer_indices):
            raise RuntimeError(
                f"{npz_path}: layer indices do not match aggregate."
            )

        gt = RELATION_TO_INDEX[relation]
        correct += (pred == gt)
        count += 1

    if count == 0:
        raise RuntimeError(
            "No passport SIDs matched Step-1 sample_arrays."
        )

    acc = correct / float(count)

    rows = []
    for layer_pos, layer in enumerate(layer_indices):
        for head in range(n_heads):
            rows.append({
                "model": args.model,
                "dataset": args.dataset,
                "layer": int(layer),
                "head": int(head),
                "head_name": f"L{int(layer)}H{head:02d}",
                "centroid_acc": float(acc[layer_pos, head]),
                "n_eval": int(count),
            })

    df = pd.DataFrame(rows).sort_values(["layer", "head"]).reset_index(drop=True)

    out = Path(args.output_dir) / "all_head_centroid_acc_passport_subset.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 120)
    print("PASSPORT-MATCHED CENTROID ACC")
    print("=" * 120)
    print(f"Passport file:   {passport_path}")
    print(f"Unique SIDs:     {len(sid_to_relation)}")
    print(f"Matched SIDs:    {count}")
    print(f"Missing SIDs:    {len(missing)}")
    if missing:
        print(f"First missing:   {missing[:20]}")
    print(f"Output:          {out}")
    print(f"Mean head ACC:   {df['centroid_acc'].mean():.4f}")
    print(f"Median head ACC: {df['centroid_acc'].median():.4f}")
    print(f"Best head ACC:   {df['centroid_acc'].max():.4f}")

    print("\nTop 20 heads on matched subset:")
    print(
        df.sort_values("centroid_acc", ascending=False)
        .head(20)[["head_name", "centroid_acc", "n_eval"]]
        .to_string(index=False)
    )

    return out


def export_csv(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    aggregate_path = out_dir / "aggregate_metrics.npz"

    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Missing {aggregate_path}\n"
            "Run without --skip-run first, or point --output-dir to an existing run."
        )

    z = np.load(aggregate_path, allow_pickle=False)

    required = {
        "layer_indices",
        "attention_original_accuracy",
    }
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{aggregate_path} is missing required arrays: {sorted(missing)}\n"
            f"Available keys: {sorted(z.files)}"
        )

    layer_indices = np.asarray(z["layer_indices"]).astype(int)
    original_acc = np.asarray(z["attention_original_accuracy"], dtype=np.float64)

    swapped_acc = (
        np.asarray(z["attention_swapped_accuracy"], dtype=np.float64)
        if "attention_swapped_accuracy" in z.files
        else None
    )
    average_acc = (
        np.asarray(z["attention_average_accuracy"], dtype=np.float64)
        if "attention_average_accuracy" in z.files
        else None
    )
    consistency = (
        np.asarray(z["attention_relation_consistency"], dtype=np.float64)
        if "attention_relation_consistency" in z.files
        else None
    )

    validate_shapes(
        layer_indices,
        original_acc,
        swapped_acc,
        average_acc,
        consistency,
    )

    rows = []
    n_layers, n_heads = original_acc.shape

    for layer_pos in range(n_layers):
        layer = int(layer_indices[layer_pos])

        for head in range(n_heads):
            row = {
                "model": args.model,
                "dataset": args.dataset,
                "layer": layer,
                "head": head,
                "head_name": f"L{layer}H{head:02d}",
                # THIS is the target to merge with head_passport_summary.csv.
                "centroid_acc": float(original_acc[layer_pos, head]),
            }

            if swapped_acc is not None:
                row["centroid_swapped_acc"] = float(
                    swapped_acc[layer_pos, head]
                )

            if average_acc is not None:
                row["centroid_original_swap_avg_acc"] = float(
                    average_acc[layer_pos, head]
                )

            if consistency is not None:
                row["centroid_relation_consistency"] = float(
                    consistency[layer_pos, head]
                )

            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["layer", "head"]).reset_index(drop=True)

    csv_path = out_dir / "all_head_centroid_acc.csv"
    df.to_csv(csv_path, index=False)

    layer_summary = (
        df.groupby("layer")["centroid_acc"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    layer_summary.to_csv(
        out_dir / "all_head_centroid_acc_layer_summary.csv",
        index=False,
    )

    top = df.sort_values("centroid_acc", ascending=False).head(30)
    top.to_csv(
        out_dir / "all_head_centroid_acc_top30.csv",
        index=False,
    )

    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "aggregate_metrics": str(aggregate_path),
        "centroid_target": "attention_original_accuracy",
        "n_layers": int(n_layers),
        "n_heads_per_layer": int(n_heads),
        "n_total_heads": int(len(df)),
        "mean_centroid_acc": float(df["centroid_acc"].mean()),
        "median_centroid_acc": float(df["centroid_acc"].median()),
        "max_centroid_acc": float(df["centroid_acc"].max()),
        "best_head": str(top.iloc[0]["head_name"]) if len(top) else None,
    }
    (out_dir / "all_head_centroid_acc_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print("ALL-HEAD CENTROID ACC EXPORTED")
    print("=" * 120)
    print(f"CSV:      {csv_path}")
    print(f"N heads:  {len(df)}")
    print(f"Mean ACC: {df['centroid_acc'].mean():.4f}")
    print(f"Median:   {df['centroid_acc'].median():.4f}")
    print(f"Max ACC:  {df['centroid_acc'].max():.4f}")
    print("\nTop 20 heads:")
    print(
        top[
            [
                "head_name",
                "centroid_acc",
                "centroid_swapped_acc",
                "centroid_original_swap_avg_acc",
                "centroid_relation_consistency",
            ]
        ].head(20).to_string(index=False)
        if all(
            c in top.columns
            for c in [
                "centroid_swapped_acc",
                "centroid_original_swap_avg_acc",
                "centroid_relation_consistency",
            ]
        )
        else top[["head_name", "centroid_acc"]].head(20).to_string(index=False)
    )

    export_passport_subset(
        args,
        layer_indices=layer_indices,
        n_heads=n_heads,
    )

    return csv_path


def main() -> None:
    args = parse_args()

    if not args.skip_run:
        run_step1(args)

    export_csv(args)


if __name__ == "__main__":
    main()
