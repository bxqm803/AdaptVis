#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired Qwen2.5-VL-3B COCO-two recovery ablation: spatial vs random 2D subspaces.

Run from the AdaptVis repository root, next to
  eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py

The ORIGINAL v6 recovery routine is used in both arms. Its head reader,
object-token sites, seven-layer window, minimum-norm optimization, relinearization,
generation stopping and binary search are unchanged. The only difference is
control_basis[L]: a frozen source-fitted H/V basis vs an independent Haar-random
orthonormal 2D basis at each layer, all in the same 2048-dimensional hidden space.

Both arms run in the SAME process/model load and use the SAME sample selection.
The random basis is drawn once (no seed selection) and reused for every sample.

NOTE: V6's selected reader HEAD ID was picked using target labels, while its
per-sample head prediction is GT-free. This comparison preserves that protocol;
it does not turn head-selection into a fully held-out evaluation.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

BASE_MODULE = "eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6"
DEFAULT_OUT = "output/qwen3b_recovery_spatial_vs_random_subspace_v1"


def randomize_geometry(geom: Mapping[int, Mapping[str, Any]], seed: int):
    """Build one full-dimensional Haar-random rank-2 subspace per decoder layer.

    Using QR of an iid-Gaussian D x 2 matrix is uniform over rank-2 subspaces.
    QR gives ||Q a||_2 == ||a||_2, matching the v6 norm convention.
    """
    rng = np.random.default_rng(int(seed))
    output, basis_to_save, report = {}, {}, []
    for layer in sorted(geom):
        orig = np.asarray(geom[layer]["control_basis"], dtype=np.float32)
        hidden, rank = orig.shape
        if rank != 2:
            raise ValueError(f"L{layer}: expected rank-2 H/V basis, got {orig.shape}")
        if hidden <= rank:
            raise ValueError(f"L{layer}: hidden size {hidden} <= rank {rank}")
        matrix = rng.standard_normal((hidden, rank))
        rand_q, _ = np.linalg.qr(matrix, mode="reduced")
        rand_q = np.asarray(rand_q, dtype=np.float32)
        error = float(np.max(np.abs(rand_q.T @ rand_q - np.eye(rank))))
        if error > 1e-5:
            raise AssertionError(f"L{layer} random basis is not orthonormal: {error}")
        new = dict(geom[layer])
        new["control_basis"] = rand_q
        # v6 SpatialPatch reads ONLY control_basis; B is kept consistent for inspection.
        new["B"] = rand_q.copy()
        output[int(layer)] = new
        basis_to_save[f"L{layer}"] = rand_q
        report.append({
            "layer": int(layer), "hidden_size": hidden, "subspace_rank": rank,
            "orthonormal_max_error": error,
            "spatial_random_overlap_fro": float(np.linalg.norm(orig.T @ rand_q)),
        })
    return output, basis_to_save, pd.DataFrame(report)


def choose_headscan(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not (candidate / "qwen-3b" / "coco" / "summary.json").is_file():
            raise FileNotFoundError(
                f"Could not find {candidate}/qwen-3b/coco/summary.json"
            )
        return candidate
    usual = Path("output/syn400_direction_head_7models_targetfull_v2")
    candidates = [usual] if (usual / "qwen-3b/coco/summary.json").is_file() else []
    if not candidates:
        candidates = [
            p.parent.parent.parent
            for p in Path("output").glob("**/qwen-3b/coco/summary.json")
        ]
    valid = []
    for root in candidates:
        try:
            meta = json.loads((root / "qwen-3b/coco/summary.json").read_text())
            if (meta.get("model") == "qwen-3b" and meta.get("dataset") == "coco"
                    and not bool(meta.get("direction_fit_uses_target_gt", True))
                    and bool(meta.get("head_selection_uses_target_gt", False))
                    and float(meta.get("head_selection_frac", 0)) >= 1.0 - 1e-9
                    and (root / "qwen-3b/coco/samples_best_head.csv").is_file()):
                valid.append(root)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    valid = sorted(set(valid))
    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise FileNotFoundError(
            "No V6-compatible head-scan outputs found. Pass --headscan-dir "
            "PATH to an existing full-target head-scan folder containing "
            "qwen-3b/coco/summary.json and samples_best_head.csv. "
            "Note: V6 rejects two-stage/held-out head-scan metadata."
        )
    raise RuntimeError(
        "Multiple valid head-scans found; choose explicitly with --headscan-dir:\n" +
        "\n".join(str(p) for p in valid)
    )


def boolcol(df: pd.DataFrame, col: str) -> pd.Series:
    x = df[col]
    if pd.api.types.is_bool_dtype(x):
        return x
    return x.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def compare_results(root: Path, expected_n: int):
    paths = {
        kind: root / kind / "qwen-3b" / "coco" / "per_sample.csv"
        for kind in ("spatial", "random")
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} run output: {path}")
    sp = pd.read_csv(paths["spatial"])
    rd = pd.read_csv(paths["random"])
    for frame, kind in ((sp, "spatial"), (rd, "random")):
        if frame.duplicated(["sid", "mode"]).any():
            raise RuntimeError(f"Duplicate sid/mode in {kind} per_sample.csv")
    sp = sp.add_prefix("spatial_").rename(columns={"spatial_sid": "sid", "spatial_mode": "mode"})
    rd = rd.add_prefix("random_").rename(columns={"random_sid": "sid", "random_mode": "mode"})
    merged = sp.merge(rd, on=["sid", "mode"], how="outer", indicator=True, validate="one_to_one")
    complete = merged[merged["_merge"] == "both"].copy()
    n_spatial, n_random, n_both = len(sp), len(rd), len(complete)
    if n_both == 0:
        raise RuntimeError("No paired sample IDs between spatial and random outputs")
    unmatched = merged[merged["_merge"] != "both"][["sid", "mode", "_merge"]]
    unmatched.to_csv(root / "unmatched_samples.csv", index=False)
    complete = complete.drop(columns=["_merge"])

    # Consistency: identical reader routing and identical unpatched generation.
    for col in ("gt", "control_target", "head_prediction", "baseline_prediction"):
        l, r = f"spatial_{col}", f"random_{col}"
        mismatches = (complete[l].fillna("<null>") != complete[r].fillna("<null>"))
        if mismatches.any():
            raise RuntimeError(f"{col} differs across paired conditions for {int(mismatches.sum())} samples")

    complete.to_csv(root / "paired_per_sample.csv", index=False)
    print(f"\n[PAIRED COVERAGE] spatial_rows={n_spatial}, random_rows={n_random}, "
          f"paired_rows={n_both}, expected_sample_count={expected_n}", flush=True)
    if n_spatial != n_random or n_both != n_spatial or n_both < expected_n:
        print("[CAUTION] Some requested samples missing/failed; paired summaries use the "
              "intersection only. Inspect both errors.jsonl and unmatched_samples.csv.", flush=True)

    summary = []
    for mode, g in complete.groupby("mode", sort=True):
        for subset_name, mask in (
            ("all", np.ones(len(g), dtype=bool)),
            ("baseline_wrong", ~boolcol(g, "spatial_baseline_correct").to_numpy()),
            ("needs_target_flip", (g["spatial_baseline_prediction"].fillna("")
                                   != g["spatial_control_target"].fillna("")).to_numpy()),
        ):
            z = g.loc[mask]
            if z.empty:
                continue
            base = boolcol(z, "spatial_baseline_correct").to_numpy()
            row = {"mode": mode, "subset": subset_name, "N": len(z),
                   "baseline_accuracy": float(np.mean(base)),
                   "direct_readout_accuracy": float(np.mean(boolcol(z, "spatial_head_correct"))),
                   "reader_target_correct_fraction": float(
                       np.mean(z["spatial_control_target"] == z["spatial_gt"])),
                  }
            for kind in ("spatial", "random"):
                y = boolcol(z, f"{kind}_patched_correct").to_numpy()
                compliance = boolcol(z, f"{kind}_patched_matches_control_target").to_numpy()
                row[f"{kind}_accuracy"] = float(np.mean(y))
                row[f"{kind}_acc_gain"] = float(np.mean(y) - np.mean(base))
                row[f"{kind}_W2C"] = int(np.sum((~base) & y))
                row[f"{kind}_C2W"] = int(np.sum(base & (~y)))
                row[f"{kind}_target_compliance"] = float(np.mean(compliance))
                row[f"{kind}_mean_steps"] = float(z[f"{kind}_steps_taken"].mean())
                row[f"{kind}_mean_norm"] = float(z[f"{kind}_final_total_residual_norm"].mean())
                row[f"{kind}_median_norm"] = float(z[f"{kind}_final_total_residual_norm"].median())
                row[f"{kind}_mean_grad_evals"] = float(z[f"{kind}_gradient_evaluations"].mean())
                row[f"{kind}_mean_gen_checks"] = float(z[f"{kind}_generation_checks"].mean())
            row["spatial_minus_random_acc"] = row["spatial_accuracy"] - row["random_accuracy"]
            row["spatial_only_correct"] = int(np.sum(
                boolcol(z, "spatial_patched_correct").to_numpy() &
                ~boolcol(z, "random_patched_correct").to_numpy()))
            row["random_only_correct"] = int(np.sum(
                ~boolcol(z, "spatial_patched_correct").to_numpy() &
                boolcol(z, "random_patched_correct").to_numpy()))
            summary.append(row)
    result = pd.DataFrame(summary)
    result.to_csv(root / "paired_summary.csv", index=False)
    display_cols = [
        "mode", "subset", "N", "baseline_accuracy", "direct_readout_accuracy",
        "spatial_accuracy", "random_accuracy", "spatial_minus_random_acc",
        "spatial_W2C", "random_W2C", "spatial_C2W", "random_C2W",
        "spatial_mean_norm", "random_mean_norm", "spatial_mean_steps", "random_mean_steps",
        "spatial_mean_gen_checks", "random_mean_gen_checks",
    ]
    print("\n" + "=" * 112)
    print("QWEN-3B / COCO-TWO: SPATIAL vs RANDOM-2D SUBSPACE (PAIRED)")
    print("=" * 112)
    print(result[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[FILES] {root / 'paired_summary.csv'}")
    print(f"[FILES] {root / 'paired_per_sample.csv'}")
    return result


def selftest():
    rng = np.random.default_rng(123)
    geom = {L: {"control_basis": np.linalg.qr(rng.normal(size=(128, 2)))[0].astype(np.float32),
                 "B": np.zeros((128, 2), dtype=np.float32)} for L in (20, 21, 22)}
    random_geom, arrays, info = randomize_geometry(geom, seed=17)
    assert len(random_geom) == 3 and len(arrays) == 3
    for L in geom:
        q = random_geom[L]["control_basis"]
        assert q.shape == (128, 2)
        assert np.allclose(q.T @ q, np.eye(2), atol=1e-5)
        assert np.linalg.norm(geom[L]["control_basis"] - q) > 0.1
    print("[SELFTEST PASS] random rank-2 subspaces, shapes and orthonormality")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--headscan-dir", default=None,
                   help="V6-compatible head scan root; autodetect if omitted")
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--eval-max-samples", type=int, default=80, help="0=all COCO-two")
    p.add_argument("--eval-seed", type=int, default=17)
    p.add_argument("--random-seed", type=int, default=20261113)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--binary-search-steps", type=int, default=4)
    p.add_argument("--max-residual-step", type=float, default=5000.0)
    p.add_argument("--modes", default="head", choices=["head", "oracle", "head,oracle"])
    p.add_argument("--geometry-cache-dir", default=None)
    p.add_argument("--preflight", action="store_true", help="Validate paths; no model load")
    p.add_argument("--selftest", action="store_true", help="Test random bases; no model load")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if args.eval_max_samples < 0:
        raise ValueError("--eval-max-samples must be >= 0")
    root = Path(args.output_dir)
    if args.preflight and not Path(BASE_MODULE + ".py").is_file():
        raise FileNotFoundError(f"Run from AdaptVis repo root: missing {BASE_MODULE}.py")
    headscan = choose_headscan(args.headscan_dir)
    print("[HEADSCAN]", headscan.resolve(), flush=True)
    if args.preflight:
        if not Path("synthetic_shapes_4dir_400/labels.jsonl").is_file():
            print("[CAUTION] Synthetic source labels missing at default path")
        print("[PREFLIGHT PASS] Qwen-3B + COCO; head scan present; no GPU required")
        return

    orig_module_path = Path(BASE_MODULE + ".py")
    if not orig_module_path.is_file():
        raise FileNotFoundError(f"Missing original V6 file in current working directory: {orig_module_path}")
    base = importlib.import_module(BASE_MODULE)
    # V6 itself checks the selected-head protocol and all dataset/model files.
    base.validate_headscan_summary(headscan / "qwen-3b" / "coco" / "summary.json", "qwen-3b", "coco")

    protocol = {
        "script": Path(__file__).name, "base_script": BASE_MODULE + ".py",
        "model": "qwen-3b", "dataset": "coco", "modes": args.modes,
        "eval_max_samples": args.eval_max_samples, "eval_seed": args.eval_seed,
        "random_seed": args.random_seed, "max_steps": args.max_steps,
        "binary_search_steps": args.binary_search_steps,
        "max_residual_step": args.max_residual_step,
        "headscan_dir": str(headscan.resolve()),
        "geometry_cache_dir": str(Path(args.geometry_cache_dir).resolve()) if args.geometry_cache_dir else None,
        "random_basis": "one fixed iid-Gaussian QR rank-2 basis per layer; not selected by target performance",
        "matched_controls": "same target reader, source geometry fit, 7 layers L20-26, object-pair token sites, minimum-norm optimizer, generation-based stopping",
        "headscan_selection_note": "V6 head ID selection used full target labels; same selected head reused across both arms",
    }
    manifest = root / "random_subspace_protocol.json"
    if manifest.exists():
        past = json.loads(manifest.read_text())
        if past != protocol:
            raise RuntimeError("Protocol mismatch with existing run. Use a NEW --output-dir, not --overwrite, to change experimental settings.\n" +
                               json.dumps({"old": past, "new": protocol}, indent=2))
    root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    original_run_pair = base.run_pair

    # Bind a closure with fixed random seed, avoiding any change to V6 itself.
    def run_both(*, model_alias, dataset, model, processor_or_backend, decoder_layers,
                 layers, geom, args: Any, root_out: Path):
        if model_alias != "qwen-3b" or dataset != "coco":
            raise RuntimeError(f"Only qwen-3b:coco is supported; got {model_alias}:{dataset}")
        if list(map(int, layers)) != list(range(20, 27)):
            raise RuntimeError(f"Expected L20-26, got {layers}")
        g_rand, arrays, basis_info = randomize_geometry(geom, seed=int(protocol["random_seed"]))
        np.savez_compressed(root_out / "random_rank2_bases.npz", **arrays)
        basis_info.to_csv(root_out / "random_basis_checks.csv", index=False)
        print("\n[CONTROL] 7 layers (L20-L26), 2 random orthonormal dims per layer; frozen random seed:",
              protocol["random_seed"], flush=True)
        print(basis_info.to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
        print("\n[ARM 1/2] ORIGINAL SPATIAL RECOVERY", flush=True)
        spatial_sum = original_run_pair(
            model_alias=model_alias, dataset=dataset, model=model,
            processor_or_backend=processor_or_backend, decoder_layers=decoder_layers,
            layers=layers, geom=geom, args=args, root_out=root_out / "spatial")
        print("\n[ARM 2/2] RANDOM SUBSPACE RECOVERY (SAME OPTIMIZER)", flush=True)
        random_sum = original_run_pair(
            model_alias=model_alias, dataset=dataset, model=model,
            processor_or_backend=processor_or_backend, decoder_layers=decoder_layers,
            layers=layers, geom=g_rand, args=args, root_out=root_out / "random")
        comp = compare_results(root_out, expected_n=args.eval_max_samples if args.eval_max_samples > 0 else 440)
        summary = dict(spatial_sum)
        if not comp.empty:
            r = comp[(comp["subset"] == "all") & (comp["mode"] == "head")]
            if len(r):
                summary["spatial_accuracy"] = float(r.iloc[0]["spatial_accuracy"])
                summary["random_accuracy"] = float(r.iloc[0]["random_accuracy"])
                summary["spatial_minus_random_acc"] = float(r.iloc[0]["spatial_minus_random_acc"])
        return summary

    base.run_pair = run_both
    forwarded = [
        str(orig_module_path),
        "--models", "qwen-3b",
        "--datasets", "coco",
        "--pairs", "qwen-3b:coco",
        "--headscan-dir", str(headscan),
        "--output-dir", str(root),
        "--modes", args.modes,
        "--controller-layers", "qwen-3b=20-26",
        "--eval-max-samples", str(args.eval_max_samples),
        "--eval-seed", str(args.eval_seed),
        "--max-steps", str(args.max_steps),
        "--binary-search-steps", str(args.binary_search_steps),
        "--max-residual-step", str(args.max_residual_step),
    ]
    if args.geometry_cache_dir:
        forwarded.extend(["--geometry-cache-dir", str(args.geometry_cache_dir)])
    forwarded.append("--fail-fast")
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        base.main()
    finally:
        sys.argv = original_argv
        base.run_pair = original_run_pair


if __name__ == "__main__":
    main()
