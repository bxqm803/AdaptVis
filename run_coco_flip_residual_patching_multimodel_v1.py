#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COCO image-flip residual patching for Qwen-3B, Qwen-7B and LLaVA-1.5-7B."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_MODELS = ["qwen-3b", "qwen-7b", "llava-7b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--script", default="eval_coco_flip_residual_patching_v1.py")
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--relations", default="left,right,above,below")
    p.add_argument("--layers", default="auto:4")
    p.add_argument("--token-groups", default="subject,reference,both,prompt_last,all_text")
    p.add_argument("--directions", default="orig_to_flip")
    p.add_argument("--pair-mode", default="both_correct")
    p.add_argument("--control", default="random_text")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-clean-pairs", type=int, default=None)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--output-root", default="output/coco_flip_residual_patching_multimodel_v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def parse_models(value: str) -> List[str]:
    out: List[str] = []
    for raw in value.split(","):
        model = raw.strip()
        if model and model not in out:
            out.append(model)
    if not out:
        raise ValueError("No models selected")
    return out


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
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


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    root = Path(args.output_root)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    status_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for model in models:
        model_dir = root / model
        cmd = [
            args.python, args.script,
            "--base-script", args.base_script,
            "--dataset", args.dataset,
            "--data-root", args.data_root,
            "--prompt-jsonl", args.prompt_jsonl,
            "--model", model,
            "--device", args.device,
            "--attn-impl", args.attn_impl,
            "--relations", args.relations,
            "--layers", args.layers,
            "--token-groups", args.token_groups,
            "--directions", args.directions,
            "--pair-mode", args.pair_mode,
            "--control", args.control,
            "--seed", str(args.seed),
            "--print-every", str(args.print_every),
            "--output-dir", str(model_dir),
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        if args.max_clean_pairs is not None:
            cmd += ["--max-clean-pairs", str(args.max_clean_pairs)]
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

        summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        top = summary.get("top_aggregate_conditions", [])
        best = top[0] if top else {}
        comparison_rows.append({
            "model": model,
            "seen": summary.get("seen"),
            "clean_pairs": summary.get("clean_pairs"),
            "best_direction": best.get("direction"),
            "best_layer": best.get("layer"),
            "best_token_group": best.get("token_group"),
            "mean_recovery": best.get("mean_recovery"),
            "median_recovery": best.get("median_recovery"),
            "fraction_recovery_gt_0_50": best.get("fraction_recovery_gt_0_50"),
            "donor_relation_rate": best.get("donor_relation_rate"),
            "excess_recovery_vs_random": best.get("excess_recovery_vs_random"),
        })

    write_csv(reports / "run_status.csv", status_rows)
    write_csv(reports / "model_comparison.csv", comparison_rows)

    header = (
        f"{'Model':<14}{'Seen':>7}{'Clean':>7}{'Direction':>15}{'Layer':>8}"
        f"{'Token group':>16}{'Mean R':>10}{'Median R':>11}"
        f"{'R>0.5':>9}{'Donor%':>9}{'ExR':>9}"
    )
    lines = [
        "=" * len(header),
        "COCO IMAGE-FLIP RESIDUAL PATCHING — MULTIMODEL",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    def f4(v: Any) -> str:
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "-"
    for row in comparison_rows:
        lines.append(
            f"{row.get('model','-'):<14}{str(row.get('seen','-')):>7}{str(row.get('clean_pairs','-')):>7}"
            f"{str(row.get('best_direction','-')):>15}{str(row.get('best_layer','-')):>8}"
            f"{str(row.get('best_token_group','-')):>16}{f4(row.get('mean_recovery')):>10}"
            f"{f4(row.get('median_recovery')):>11}{f4(row.get('fraction_recovery_gt_0_50')):>9}"
            f"{f4(row.get('donor_relation_rate')):>9}{f4(row.get('excess_recovery_vs_random')):>9}"
        )
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    (reports / "report.txt").write_text(report, encoding="utf-8")

    print("Saved:")
    print(" ", reports / "report.txt")
    print(" ", reports / "model_comparison.csv")
    print(" ", reports / "run_status.csv")


if __name__ == "__main__":
    main()
