#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize fixed-threshold prototype_head error detection across VLMs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        value = float(fn(*args, **kwargs))
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def top_heads(path: Path, limit: int = 5) -> str:
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str) == "prototype_head"]
    if frame.empty or "head" not in frame.columns:
        return ""
    sort_cols = [
        column
        for column in ("max_nonzero_fold_fraction", "max_mean_abs_coefficient")
        if column in frame.columns
    ]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return ",".join(frame["head"].astype(str).head(limit).tolist())


def summarize_model(model: str, detector_dir: Path, threshold: float) -> Dict[str, Any]:
    prediction_path = detector_dir / "oof_predictions.csv"
    if not prediction_path.exists():
        return {
            "model": model,
            "status": "missing",
            "detector_dir": str(detector_dir),
        }

    frame = pd.read_csv(prediction_path)
    probability_column = "error_probability__prototype_head"
    required = {"error_label", probability_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {
            "model": model,
            "status": f"missing_columns:{','.join(missing)}",
            "detector_dir": str(detector_dir),
        }

    y = frame["error_label"].astype(int).to_numpy()
    p = frame[probability_column].astype(float).to_numpy()
    pred = (p >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    error_count = int(y.sum())
    correct_count = int(n - error_count)

    row: Dict[str, Any] = {
        "model": model,
        "status": "ok",
        "N": n,
        "generation_correct": correct_count,
        "generation_wrong": error_count,
        "generation_accuracy": correct_count / max(n, 1),
        "error_prevalence": error_count / max(n, 1),
        "majority_detector_accuracy": max(correct_count, error_count) / max(n, 1),
        "threshold": float(threshold),
        "detector_accuracy": safe_metric(accuracy_score, y, pred),
        "detector_balanced_accuracy": safe_metric(balanced_accuracy_score, y, pred),
        "error_precision": safe_metric(precision_score, y, pred, zero_division=0),
        "error_recall": safe_metric(recall_score, y, pred, zero_division=0),
        "error_f1": safe_metric(f1_score, y, pred, zero_division=0),
        "AUROC": safe_metric(roc_auc_score, y, p) if len(np.unique(y)) == 2 else float("nan"),
        "AUPRC": safe_metric(average_precision_score, y, p),
        "Brier": safe_metric(brier_score_loss, y, p),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "predicted_error_N": int(pred.sum()),
        "top_heads": top_heads(detector_dir / "head_importance_summary.csv"),
        "detector_dir": str(detector_dir),
    }

    config_path = detector_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            row["outer_folds"] = config.get("outer_folds")
            row["outer_repeats"] = config.get("outer_repeats")
            row["stratify"] = config.get("stratify")
        except Exception:
            pass

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="qwen2-2b,qwen-7b,llava-7b,llava-13b,internvl-1b,internvl-2b,internvl-8b",
    )
    parser.add_argument(
        "--detector-root",
        default="output/coco_all_relation_head_prototype_detector",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default="output/coco_all_relation_head_prototype_detector/multimodel_summary",
    )
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    root = Path(args.detector_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        summarize_model(model, root / model, float(args.threshold))
        for model in models
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "prototype_head_multimodel_summary.csv", index=False)

    ok = frame[frame["status"] == "ok"].copy() if "status" in frame.columns else pd.DataFrame()
    if not ok.empty:
        ok = ok.sort_values(
            ["detector_balanced_accuracy", "detector_accuracy"],
            ascending=False,
        )

    lines = [
        "=" * 170,
        "MULTI-MODEL PROTOTYPE_HEAD ERROR DETECTOR",
        f"fixed threshold={args.threshold:.3f}",
        "=" * 170,
        (
            f"{'Model':<16}{'N':>6}{'GenAcc':>10}{'ErrPrev':>10}"
            f"{'DetAcc':>10}{'DetBAcc':>10}{'AUROC':>9}{'AUPRC':>9}"
            f"{'Prec':>9}{'Recall':>9}{'F1':>9}{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}"
        ),
        "-" * 170,
    ]

    for _, row in ok.iterrows():
        lines.append(
            f"{str(row['model']):<16}{int(row['N']):>6}"
            f"{float(row['generation_accuracy']):>10.4f}"
            f"{float(row['error_prevalence']):>10.4f}"
            f"{float(row['detector_accuracy']):>10.4f}"
            f"{float(row['detector_balanced_accuracy']):>10.4f}"
            f"{float(row['AUROC']):>9.4f}"
            f"{float(row['AUPRC']):>9.4f}"
            f"{float(row['error_precision']):>9.4f}"
            f"{float(row['error_recall']):>9.4f}"
            f"{float(row['error_f1']):>9.4f}"
            f"{int(row['TN']):>6}{int(row['FP']):>6}{int(row['FN']):>6}{int(row['TP']):>6}"
        )

    missing_rows = frame[frame["status"] != "ok"] if "status" in frame.columns else pd.DataFrame()
    if not missing_rows.empty:
        lines += ["", "MISSING / FAILED"]
        for _, row in missing_rows.iterrows():
            lines.append(f"{row['model']}: {row['status']}")

    lines += [
        "",
        "Notes:",
        "- detector_accuracy uses the same fixed threshold 0.5 for every model.",
        "- AUROC/AUPRC are threshold-free; accuracy is affected by each model's error prevalence.",
        "- Compare detector_balanced_accuracy and AUROC across models, not ordinary accuracy alone.",
        "- top_heads are predictive features, not causal heads.",
        "",
        f"CSV: {output_dir / 'prototype_head_multimodel_summary.csv'}",
    ]

    report = "\n".join(lines) + "\n"
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
