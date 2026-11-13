#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired spatial-vs-random recovery control for Qwen2.5-VL-3B / COCO-two.

Place this file in the AdaptVis `llava16` repo root. It uses the ORIGINAL
`eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py`
optimizer and its existing data/head-scan machinery.

Spatial arm: source-fitted orthonormal H/V rank-2 basis at each of L20..L26.
Random arm: independent frozen Haar-random rank-2 basis per layer, obtained via
Gaussian QR. Same model, examples, head-selected targets, intervention sites,
QP optimization, trust cap, max steps and generation-based stopping.

This matches available optimization BUDGETS, not necessarily actual achieved
norms or steps. Do not call it an exact norm-/step-matched experiment.

The V6 head ID was chosen using target labels, although per-example head
predictions do not use target labels. This experiment does not change that.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

# Existing and independently inspectable controls from the same llava16 branch.
import eval_qwen3b_recovery_random_subspace_control_v1 as control

BASE_MODULE = "eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6"


def arguments():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--headscan-dir", default="output/syn400_direction_head_7models_targetfull_v2")
    ap.add_argument("--output-dir", default="output/qwen3b_spatial_vs_random_v2")
    ap.add_argument("--geometry-cache-dir", default=None)
    ap.add_argument("--eval-max-samples", type=int, default=80, help="0=all 440 COCO-two samples")
    ap.add_argument("--eval-seed", type=int, default=17)
    ap.add_argument("--random-seed", type=int, default=20261113)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--generation-margin-increment", type=float, default=4.0)
    ap.add_argument("--max-residual-step", type=float, default=5000.0)
    ap.add_argument("--binary-search-steps", type=int, default=4)
    ap.add_argument("--modes", choices=("head", "oracle", "head,oracle"), default="head")
    ap.add_argument("--preflight", action="store_true", help="Check files before GPU use")
    ap.add_argument("--selftest", action="store_true", help="Test random basis; no GPU")
    return ap.parse_args()


def main():
    a = arguments()
    if a.selftest:
        control.selftest()
        return
    if a.eval_max_samples < 0 or a.max_steps <= 0:
        raise ValueError("eval-max-samples >= 0 and max-steps > 0 are required")
    if a.generation_margin_increment <= 0:
        raise ValueError("generation-margin-increment must be > 0")

    headscan = Path(a.headscan_dir)
    headsum = headscan / "qwen-3b" / "coco" / "summary.json"
    headsamples = headscan / "qwen-3b" / "coco" / "samples_best_head.csv"
    required = [
        Path(BASE_MODULE + ".py"),
        Path("eval_qwen3b_recovery_random_subspace_control_v1.py"),
        Path("synthetic_shapes_4dir_400/labels.jsonl"),
        Path("data/coco_qa_two_obj.json"),
        Path("data/val2017"),
        Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl"),
        headsum, headsamples,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    base = importlib.import_module(BASE_MODULE)
    summary = base.validate_headscan_summary(headsum, "qwen-3b", "coco")
    print("[HEAD]", summary.get("best_head"), "[HEADSCAN]", headscan, flush=True)
    print("[DESIGN] same original V6 procedure; only per-layer 2D basis changes", flush=True)
    if a.preflight:
        print("[PREFLIGHT PASS] Required files and head-scan protocol validated", flush=True)
        return

    root = Path(a.output_dir)
    # Keep the original V6 experiment's results safe: never overwrite it.
    protocol = {
        "experiment": "paired_qwen3b_coco_spatial_vs_random_subspace_v2",
        "base_module": BASE_MODULE,
        "headscan_dir": str(headscan.resolve()),
        "head_id": summary.get("best_head"),
        "modes": a.modes,
        "eval_max_samples": a.eval_max_samples,
        "eval_seed": a.eval_seed,
        "random_seed": a.random_seed,
        "max_steps": a.max_steps,
        "generation_margin_increment": a.generation_margin_increment,
        "max_residual_step": a.max_residual_step,
        "binary_search_steps": a.binary_search_steps,
        "geometry_cache_dir": str(Path(a.geometry_cache_dir).resolve()) if a.geometry_cache_dir else None,
        "control_layers": [20, 21, 22, 23, 24, 25, 26],
        "basis_rank": 2,
        "random_basis": "one independent frozen Gaussian-QR rank2 subspace per layer",
        "match": "reader targets, samples, token sites, optimizer, hyperparameters, max budgets",
        "limitation": "actual perturbation norms and completed step counts may differ",
    }
    manifest = root / "protocol.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != protocol:
            raise RuntimeError(
                f"Protocol changed for {root}. Use another --output-dir.\n" +
                json.dumps({"old": old, "new": protocol}, indent=2)
            )
    root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    original_run_pair = base.run_pair

    def run_pair_both(*, model_alias, dataset, model, processor_or_backend,
                      decoder_layers, layers, geom, args, root_out):
        if (model_alias, dataset) != ("qwen-3b", "coco"):
            raise RuntimeError("Only qwen-3b:coco supported")
        if list(map(int, layers)) != list(range(20, 27)):
            raise RuntimeError(f"Expected L20..L26; found {layers}")

        random_geom, random_bases, checks = control.randomize_geometry(
            geom, seed=a.random_seed
        )
        np.savez_compressed(root_out / "random_bases_rank2.npz", **random_bases)
        checks.to_csv(root_out / "random_basis_checks.csv", index=False)
        print("[RANDOM BASIS]", checks.to_string(index=False), flush=True)

        print("\n[ARM 1] Original spatial recovery (V6)\n", flush=True)
        spatial = original_run_pair(
            model_alias=model_alias, dataset=dataset, model=model,
            processor_or_backend=processor_or_backend, decoder_layers=decoder_layers,
            layers=layers, geom=geom, args=args, root_out=root_out / "spatial")

        print("\n[ARM 2] Random 2D subspace recovery (same V6 optimizer)\n", flush=True)
        original_run_pair(
            model_alias=model_alias, dataset=dataset, model=model,
            processor_or_backend=processor_or_backend, decoder_layers=decoder_layers,
            layers=layers, geom=random_geom, args=args, root_out=root_out / "random")

        # The underlying V6 metadata assumes spatial geometry. Override this
        # descriptive label only, without modifying the optimization itself.
        metadata = root_out / "random" / "qwen-3b" / "coco" / "metadata.json"
        if metadata.exists():
            info = json.loads(metadata.read_text(encoding="utf-8"))
            info["spatial_basis"] = "Haar random orthonormal rank-2 basis per layer"
            info["random_basis_seed"] = a.random_seed
            metadata.write_text(json.dumps(info, indent=2), encoding="utf-8")

        paired = control.compare_results(
            root_out, expected_n=a.eval_max_samples if a.eval_max_samples else 440
        )
        if not paired.empty and a.modes in ("head", "head,oracle"):
            sel = paired[(paired["mode"] == "head") & (paired["subset"] == "all")]
            if len(sel):
                record = sel.iloc[0]
                spatial["spatial_accuracy"] = float(record["spatial_accuracy"])
                spatial["random_accuracy"] = float(record["random_accuracy"])
                spatial["spatial_minus_random_acc"] = float(record["spatial_minus_random_acc"])
        return spatial

    forwarded = [
        BASE_MODULE + ".py",
        "--models", "qwen-3b",
        "--datasets", "coco",
        "--pairs", "qwen-3b:coco",
        "--headscan-dir", str(headscan),
        "--output-dir", str(root),
        "--controller-layers", "qwen-3b=20-26",
        "--modes", a.modes,
        "--eval-max-samples", str(a.eval_max_samples),
        "--eval-seed", str(a.eval_seed),
        "--max-steps", str(a.max_steps),
        "--generation-margin-increment", str(a.generation_margin_increment),
        "--max-residual-step", str(a.max_residual_step),
        "--binary-search-steps", str(a.binary_search_steps),
        "--fail-fast",
    ]
    if a.geometry_cache_dir:
        forwarded.extend(["--geometry-cache-dir", a.geometry_cache_dir])

    orig_argv = sys.argv
    base.run_pair = run_pair_both
    try:
        sys.argv = forwarded
        base.main()
    finally:
        base.run_pair = original_run_pair
        sys.argv = orig_argv


if __name__ == "__main__":
    main()
