#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COCO token-path contribution ablation across multiple VLMs."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_MODELS = ["qwen-3b", "qwen-7b", "llava-7b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument(
        "--script",
        default="eval_coco_token_path_contribution_ablation_v1.py",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pathways", default="pair,route")
    p.add_argument("--layer-windows", default="auto:8")
    p.add_argument("--head-selection", default="all")
    p.add_argument("--head-fraction", type=float, default=1.0)
    p.add_argument("--control", default="random")
    p.add_argument("--evaluation", default="first_token")
    p.add_argument("--condition-samples", default="baseline_correct")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument(
        "--output-root",
        default="output/coco_token_path_contribution_ablation_multimodel_v1",
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
        raise ValueError("--models produced no models")
    return result


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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def best_real_condition(
    summary: Mapping[str, Any],
    pathway: str,
) -> Dict[str, Any]:
    candidates = [
        dict(row)
        for row in summary.get("condition_summary", [])
        if row.get("pathway") == pathway
        and not bool(row.get("control"))
        and int(row.get("n") or 0) > 0
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            -float(row.get("broken_rate_among_baseline_correct") or 0.0),
            float(
                row.get("mean_gt_margin_change_on_baseline_correct")
                if row.get("mean_gt_margin_change_on_baseline_correct") is not None
                else 0.0
            ),
        )
    )
    return candidates[0]


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
            args.script,
            "--base-script", args.base_script,
            "--dataset", args.dataset,
            "--data-root", args.data_root,
            "--prompt-jsonl", args.prompt_jsonl,
            "--model", model,
            "--device", args.device,
            "--pathways", args.pathways,
            "--layer-windows", args.layer_windows,
            "--head-selection", args.head_selection,
            "--head-fraction", str(args.head_fraction),
            "--control", args.control,
            "--evaluation", args.evaluation,
            "--condition-samples", args.condition_samples,
            "--max-new-tokens", str(args.max_new_tokens),
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
        completed = subprocess.run(cmd, check=False)
        ok = completed.returncode == 0
        status_rows.append({
            "model": model,
            "ok": ok,
            "returncode": completed.returncode,
            "output_dir": str(model_dir),
        })
        if not ok:
            if args.stop_on_error:
                break
            continue

        summary = read_json(model_dir / "summary.json")
        pair = best_real_condition(summary, "pair")
        route = best_real_condition(summary, "route")
        comparison_rows.append({
            "model": model,
            "n_rows": summary.get("n_rows"),
            "baseline_accuracy": summary.get("baseline_accuracy"),
            "pair_best_condition": pair.get("name"),
            "pair_broken_rate": pair.get("broken_rate_among_baseline_correct"),
            "pair_margin_change": pair.get(
                "mean_gt_margin_change_on_baseline_correct"
            ),
            "pair_excess_broken_vs_random": pair.get(
                "excess_broken_vs_random"
            ),
            "pair_extra_margin_drop_vs_random": pair.get(
                "extra_margin_drop_vs_random"
            ),
            "route_best_condition": route.get("name"),
            "route_broken_rate": route.get("broken_rate_among_baseline_correct"),
            "route_margin_change": route.get(
                "mean_gt_margin_change_on_baseline_correct"
            ),
            "route_excess_broken_vs_random": route.get(
                "excess_broken_vs_random"
            ),
            "route_extra_margin_drop_vs_random": route.get(
                "extra_margin_drop_vs_random"
            ),
        })

    write_csv(reports / "run_status.csv", status_rows)
    write_csv(reports / "model_comparison.csv", comparison_rows)

    header = (
        f"{'Model':<14}{'N':>6}{'Base':>9}"
        f"{'Best Pair':>22}{'Break%':>9}{'MarginΔ':>11}{'ExBreak':>9}"
        f"{'Best Route':>22}{'Break%':>9}{'MarginΔ':>11}{'ExBreak':>9}"
    )
    lines = [
        "=" * len(header),
        "COCO TOKEN-PATH CONTRIBUTION ABLATION — MULTIMODEL",
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
            f"{str(row.get('n_rows', '-')):>6}"
            f"{f4(row.get('baseline_accuracy')):>9}"
            f"{str(row.get('pair_best_condition', '-')):>22}"
            f"{f4(row.get('pair_broken_rate')):>9}"
            f"{f4(row.get('pair_margin_change')):>11}"
            f"{str(row.get('pair_excess_broken_vs_random', '-')):>9}"
            f"{str(row.get('route_best_condition', '-')):>22}"
            f"{f4(row.get('route_broken_rate')):>9}"
            f"{f4(row.get('route_margin_change')):>11}"
            f"{str(row.get('route_excess_broken_vs_random', '-')):>9}"
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
