#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the COCO-two object-pair spatial-failure diagnostic across VLMs.

Default models:
- qwen-3b
- qwen-7b
- llava-7b (LLaVA-1.5-7B)

Any model alias exposed by analyze_coco_centroid_generation_step1_v4.py and
extract_two_object_relation_states.py may be passed through --models.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


SCRIPT_VERSION = "coco-object-pair-spatial-failure-multimodel-v1"
DEFAULT_MODELS = ["qwen-3b", "qwen-7b", "llava-7b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--step1-script",
        default="analyze_coco_object_pair_spatial_failure_v1.py",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--layers", default="all")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--top-head-fraction", type=float, default=0.25)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument(
        "--output-root",
        default="output/coco_object_pair_spatial_failure_multimodel_v1",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def parse_models(value: str) -> List[str]:
    result: List[str] = []
    for raw in str(value).split(","):
        name = raw.strip()
        if name and name not in result:
            result.append(name)
    if not result:
        raise ValueError("--models resolved to an empty list")
    return result


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def family_fields(summary: Mapping[str, Any], family: str) -> Dict[str, Any]:
    row = dict((summary.get("family_best") or {}).get(family) or {})
    return {
        f"{family}_metric": row.get("metric"),
        f"{family}_layer": row.get("layer"),
        f"{family}_correct_mean": row.get("correct_mean"),
        f"{family}_wrong_mean": row.get("wrong_mean"),
        f"{family}_cohen_d": row.get("cohen_d"),
        f"{family}_auc": row.get("auc_correct"),
        f"{family}_cv_ba": row.get("cv_balanced_accuracy"),
    }


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    root = Path(args.output_root)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    status_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for model in models:
        model_dir = root / model
        cmd = [
            args.python,
            args.step1_script,
            "--base-script", args.base_script,
            "--dataset", args.dataset,
            "--data-root", args.data_root,
            "--prompt-jsonl", args.prompt_jsonl,
            "--model", model,
            "--device", args.device,
            "--attn-impl", args.attn_impl,
            "--layers", args.layers,
            "--max-new-tokens", str(args.max_new_tokens),
            "--top-head-fraction", str(args.top_head_fraction),
            "--cv-folds", str(args.cv_folds),
            "--seed", str(args.seed),
            "--print-every", str(args.print_every),
            "--output-dir", str(model_dir),
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.overwrite:
            cmd.append("--overwrite")

        print("\n" + "=" * 120)
        print("RUN:", " ".join(cmd))
        print("=" * 120)

        try:
            completed = subprocess.run(cmd, check=False)
            ok = completed.returncode == 0
            error = None if ok else f"returncode={completed.returncode}"
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        status_rows.append({
            "model": model,
            "ok": ok,
            "error": error,
            "output_dir": str(model_dir),
        })

        if not ok:
            if args.stop_on_error:
                break
            continue

        summary = read_json(model_dir / "summary.json")
        verdict = dict(summary.get("verdict") or {})
        groups = dict(summary.get("failure_group_counts") or {})
        row: Dict[str, Any] = {
            "model": model,
            "n_samples": summary.get("n_samples"),
            "generation_accuracy": summary.get("generation_accuracy"),
            "n_correct": summary.get("n_correct"),
            "n_wrong": summary.get("n_wrong"),
            "verdict": verdict.get("label"),
            "best_cv_balanced_accuracy": verdict.get(
                "best_cv_balanced_accuracy"
            ),
            "pair_formation_missing": groups.get(
                "pair_formation_missing", 0
            ),
            "decision_routing_missing": groups.get(
                "decision_routing_missing", 0
            ),
            "late_relation_overwrite": groups.get(
                "late_relation_overwrite", 0
            ),
            "unclassified_wrong": groups.get(
                "unclassified_wrong", 0
            ),
        }
        for family in ("pair", "routing", "representation"):
            row.update(family_fields(summary, family))
        comparison_rows.append(row)

    write_csv(reports / "run_status.csv", status_rows)
    write_csv(reports / "model_comparison.csv", comparison_rows)

    header = (
        f"{'Model':<14}{'N':>6}{'GenAcc':>10}{'Verdict':>16}"
        f"{'Pair CV':>10}{'Pair L':>8}"
        f"{'Route CV':>11}{'Route L':>9}"
        f"{'Repr CV':>10}{'Repr L':>8}"
        f"{'PairMiss':>10}{'RouteMiss':>11}{'Overwrite':>11}"
    )
    lines = [
        "=" * len(header),
        "COCO-TWO OBJECT-PAIR SPATIAL-FAILURE DIAGNOSTIC",
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    def f4(value: Optional[Any]) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "-"

    for row in comparison_rows:
        lines.append(
            f"{str(row.get('model', '-')):<14}"
            f"{str(row.get('n_samples', '-')):>6}"
            f"{f4(row.get('generation_accuracy')):>10}"
            f"{str(row.get('verdict', '-')):>16}"
            f"{f4(row.get('pair_cv_ba')):>10}"
            f"{str(row.get('pair_layer', '-')):>8}"
            f"{f4(row.get('routing_cv_ba')):>11}"
            f"{str(row.get('routing_layer', '-')):>9}"
            f"{f4(row.get('representation_cv_ba')):>10}"
            f"{str(row.get('representation_layer', '-')):>8}"
            f"{str(row.get('pair_formation_missing', 0)):>10}"
            f"{str(row.get('decision_routing_missing', 0)):>11}"
            f"{str(row.get('late_relation_overwrite', 0)):>11}"
        )

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    (reports / "report.txt").write_text(report, encoding="utf-8")

    print("Saved:")
    print(" ", reports / "model_comparison.csv")
    print(" ", reports / "run_status.csv")
    print(" ", reports / "report.txt")


if __name__ == "__main__":
    main()
