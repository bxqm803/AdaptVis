#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_synthetic3d_lr_ab_multimodel_recovery_v1.py

Integrated Synthetic-3D -> COCO-two / Controlled-A recovery experiment.

This keeps the original v6 protocol and changes only the spatial source:

  old source: Synthetic-400 (2D, left/right/above/below)
  new source: Blender Synthetic-3D-600, restricted to
              left/right/above/below (100 examples each; N=400)

For each model, the script:
  1) extracts source pre-W_O Real-Gray head vectors on Synthetic-3D LRAB;
  2) fits frozen four-way relation directions from the source;
  3) selects one head ID using target GT, exactly as in the original v6 protocol;
  4) fits the residual H/V controller geometry from the same Synthetic-3D LRAB source;
  5) runs the original v6 generation-flip controller on COCO-two / Controlled-A.

The actual head-vector extraction, target loaders, model loaders, InternVL support,
and generation controller are reused from the existing AdaptVis scripts:
  scan_synthetic_frozen_direction_heads_7models_v2.py
  eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py

No target hidden-space direction is used to fit the source spatial basis.
The target GT is still used to choose the best head ID, matching the original v6
reported protocol. Per-sample head routing uses only the selected head prediction.

Default Blender source path intentionally uses the existing dataset location:
  /ddnB/work/mwang32/AdaptVis/synthetic_shapes_6dir_600_3d

Outputs:
  <output-dir>/headscan/<model>/<dataset>/...
  <output-dir>/geometry_cache/<model>/...
  <output-dir>/control/<model>/<dataset>/...
  <output-dir>/cross_model_control_summary.csv
  <output-dir>/cross_model_control_summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import scan_synthetic_frozen_direction_heads_7models_v2 as hs
except Exception as exc:
    raise SystemExit(
        "Could not import scan_synthetic_frozen_direction_heads_7models_v2.py. "
        "Run this script from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as v6
except Exception as exc:
    raise SystemExit(
        "Could not import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py. "
        "Run this script from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "synthetic3d-lrab-multimodel-v6-recovery-v1"
REL = ("left", "right", "above", "below")
MODEL_ORDER = (
    "qwen2-2b",
    "qwen-3b",
    "qwen-7b",
    "llava-7b",
    "llava-13b",
    "internvl-1b",
)
DATASET_ORDER = ("coco", "controlled_a")
DEFAULT_SYNTHETIC_3D_DIR = "/ddnB/work/mwang32/AdaptVis/synthetic_shapes_6dir_600_3d"


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def canonical_model_alias(name: str) -> str:
    x = str(name).strip().lower()
    return {
        "qwen22b": "qwen2-2b",
        "qwen2vl-2b": "qwen2-2b",
        "qwen2-vl-2b": "qwen2-2b",
        "qwen25-3b": "qwen-3b",
        "qwen2.5-3b": "qwen-3b",
        "qwen25-7b": "qwen-7b",
        "qwen2.5-7b": "qwen-7b",
        "llava15-7b": "llava-7b",
        "llava1.5-7b": "llava-7b",
        "llava15-13b": "llava-13b",
        "llava1.5-13b": "llava-13b",
        "internvl2.5-1b": "internvl-1b",
        "internvl25-1b": "internvl-1b",
    }.get(x, x)


def parse_name_list(value: str, allowed: Sequence[str], *, canonicalize_models: bool = False) -> List[str]:
    if str(value).strip().lower() == "all":
        return list(allowed)
    vals = [x.strip() for x in str(value).split(",") if x.strip()]
    if canonicalize_models:
        vals = [canonical_model_alias(x) for x in vals]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"Unknown values {bad}; allowed={list(allowed)}")
    out: List[str] = []
    for x in vals:
        if x not in out:
            out.append(x)
    return out


def resolve_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.pairs:
        out: List[Tuple[str, str]] = []
        for item in str(args.pairs).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Bad --pairs item {item!r}; expected model:dataset")
            model, dataset = [x.strip() for x in item.split(":", 1)]
            model = canonical_model_alias(model)
            if model not in MODEL_ORDER:
                raise ValueError(f"Unknown model {model!r}; allowed={MODEL_ORDER}")
            if dataset not in DATASET_ORDER:
                raise ValueError(f"Unknown dataset {dataset!r}; allowed={DATASET_ORDER}")
            if (model, dataset) not in out:
                out.append((model, dataset))
        if not out:
            raise ValueError("--pairs produced no valid pairs")
        return out

    models = parse_name_list(args.models, MODEL_ORDER, canonicalize_models=True)
    datasets = parse_name_list(args.datasets, DATASET_ORDER)
    return [(m, d) for m in models for d in datasets]


def _relation(x: Any) -> Optional[str]:
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left",
        "left_of": "left",
        "left of": "left",
        "right": "right",
        "right_of": "right",
        "right of": "right",
        "above": "above",
        "on": "above",
        "on_top_of": "above",
        "on top of": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "front": "front",
        "in_front": "front",
        "in front": "front",
        "behind": "behind",
    }
    return table.get(s, s)


def load_synthetic3d_lrab(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_3d_dir)
    labels_path = Path(args.synthetic_3d_labels) if args.synthetic_3d_labels else root / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows: List[Dict[str, Any]] = []
    raw_counts: Counter = Counter()
    kept_counts: Counter = Counter()

    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rel = _relation(item.get("relation"))
            raw_counts[str(rel)] += 1
            if rel not in REL:
                continue

            image_value = item.get("image", item.get("image_path"))
            if image_value is None:
                raise RuntimeError(f"{labels_path}:{line_no}: missing image/image_path")
            p = Path(str(image_value))
            image_path = p if p.is_absolute() else root / p
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            subject = str(item.get("subject", "")).strip()
            reference = str(item.get("reference", "")).strip()
            if not subject or not reference:
                raise RuntimeError(f"{labels_path}:{line_no}: missing subject/reference")

            sid_raw = item.get("id", item.get("image_id", len(rows)))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = len(rows)

            rows.append({
                "sid": sid,
                "dataset": "synthetic3d_lrab",
                "image_path": str(image_path),
                "subject": subject,
                "reference": reference,
                "relation": rel,
                "question_text": hs.SYN_PROMPT.format(subject=subject, reference=reference),
            })
            kept_counts[rel] += 1

    rows.sort(key=lambda r: int(r["sid"]))

    # Full protocol uses all 400 LRAB rows. If a smaller source cap is requested,
    # keep it balanced rather than taking the first N rows from labels.jsonl.
    if args.source_max_samples is not None and int(args.source_max_samples) > 0 and int(args.source_max_samples) < len(rows):
        n = int(args.source_max_samples)
        per = n // len(REL)
        rem = n % len(REL)
        capped: List[Dict[str, Any]] = []
        by_rel = {r: [x for x in rows if x["relation"] == r] for r in REL}
        for j, rel in enumerate(REL):
            take = per + (1 if j < rem else 0)
            capped.extend(by_rel[rel][:take])
        rows = sorted(capped, key=lambda r: int(r["sid"]))

    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in REL if counts[r] == 0]
    if missing:
        raise RuntimeError(f"Synthetic-3D LRAB source missing classes {missing}; counts={dict(counts)}")

    print(
        f"[Synthetic-3D LRAB source] n={len(rows)} counts={dict(counts)} "
        f"raw_counts={dict(raw_counts)}"
    )
    return rows


def patch_headscan_summary(path: Path) -> None:
    if not path.exists():
        return
    s = json.loads(path.read_text(encoding="utf-8"))
    s["script_version"] = SCRIPT_VERSION + "/headscan"
    s["source"] = "Blender Synthetic-3D LRAB subset"
    s["source_dataset"] = str(DEFAULT_SYNTHETIC_3D_DIR)
    s["source_relations"] = list(REL)
    s["source_direction_fit_uses_target_gt"] = False
    s["protocol_note"] = (
        "Relation directions are fit only on Synthetic-3D LRAB; target GT is used only "
        "to select the head ID, matching the original v6 protocol."
    )
    write_json(path, s)


def patch_control_metadata(path: Path, args: argparse.Namespace) -> None:
    if not path.exists():
        return
    s = json.loads(path.read_text(encoding="utf-8"))
    s["script_version"] = SCRIPT_VERSION + "/control"
    s["direction_source"] = "Blender Synthetic-3D LRAB subset only"
    s["residual_geometry_source"] = "Blender Synthetic-3D LRAB subset only"
    s["spatial_basis"] = "orthonormal basis spanning Synthetic-3D horizontal/vertical directions"
    s["source_relations"] = list(REL)
    s["synthetic_3d_dir"] = str(args.synthetic_3d_dir)
    write_json(path, s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Pair selection.
    p.add_argument("--models", default="all", help="all or comma-separated model aliases")
    p.add_argument("--datasets", default="all", help="all or comma-separated: coco,controlled_a")
    p.add_argument("--pairs", default=None, help="Optional exact pairs, e.g. qwen-3b:coco,llava-7b:controlled_a")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    # Synthetic-3D source. This is the known dataset location from the current experiments.
    p.add_argument("--synthetic-3d-dir", default=DEFAULT_SYNTHETIC_3D_DIR)
    p.add_argument("--synthetic-3d-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None, help="Default uses all LRAB source rows (400).")

    # Target datasets.
    p.add_argument("--data-root", default="data")
    p.add_argument("--coco-prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    p.add_argument("--target-max-samples", type=int, default=None)

    # Model/runtime args shared with the existing scanner/controller.
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--revision", default="main")
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--control", default="gray", choices=["gray"])
    p.add_argument("--geometry-control", default="gray", choices=["gray"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument("--internvl-use-thumbnail", action=argparse.BooleanOptionalAction, default=True)

    # Head scan. Full target GT selection is kept intentionally for v6 comparability.
    p.add_argument("--head-selection-frac", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-target-vector-success-rate", type=float, default=0.95)
    p.add_argument("--cache-vectors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--with-generation", action=argparse.BooleanOptionalAction, default=False)

    # Controller geometry.
    p.add_argument(
        "--controller-layers",
        default="",
        help="Optional per-model overrides, e.g. qwen-3b=20-26,llava-7b=15-21",
    )
    p.add_argument("--geometry-cache-dir", default=None)

    # Recovery evaluation. Defaults match the full v6 runs used in the current paper.
    p.add_argument("--modes", default="head,oracle")
    p.add_argument("--eval-max-samples", type=int, default=0, help="0 = all target samples")
    p.add_argument("--eval-seed", type=int, default=17)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--decision-margin-eps", type=float, default=1e-4)
    p.add_argument("--generation-margin-increment", type=float, default=4.0)
    p.add_argument("--binary-search-steps", type=int, default=4)
    p.add_argument("--qp-feas-tol", type=float, default=1e-7)
    p.add_argument("--qp-lambda-tol", type=float, default=1e-9)
    p.add_argument("--min-residual-step", type=float, default=1e-8)
    p.add_argument("--max-residual-step", type=float, default=5000.0)
    p.add_argument("--generation-check-every", type=int, default=1)
    p.add_argument("--cuda-cleanup-every", type=int, default=25)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--sequence-score-reduction", default="mean", choices=["mean", "sum"])

    args = p.parse_args()

    # Attributes expected by the original scripts. They are set here so the imported
    # functions can be reused without modifying the repo files.
    args.synthetic_dir = args.synthetic_3d_dir
    args.synthetic_labels = args.synthetic_3d_labels
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if abs(float(args.head_selection_frac) - 1.0) > 1e-12:
        raise ValueError(
            "This script reproduces the original v6 protocol, which uses the full target set "
            "to choose the head ID. Use --head-selection-frac 1.0."
        )

    hs.seed_all(int(args.seed))
    pairs = resolve_pairs(args)
    models_in_order = [m for m in MODEL_ORDER if any(pm == m for pm, _ in pairs)]

    root_out = Path(args.output_dir)
    if args.overwrite and root_out.exists():
        shutil.rmtree(root_out)
    root_out.mkdir(parents=True, exist_ok=True)

    headscan_root = root_out / "headscan"
    control_root = root_out / "control"
    geom_root = Path(args.geometry_cache_dir) if args.geometry_cache_dir else root_out / "geometry_cache"
    headscan_root.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)
    geom_root.mkdir(parents=True, exist_ok=True)
    args.headscan_dir = str(headscan_root)

    source_rows = load_synthetic3d_lrab(args)
    overrides = v6.parse_layer_overrides(args.controller_layers)

    protocol = {
        "script_version": SCRIPT_VERSION,
        "source": "Blender Synthetic-3D-600",
        "source_path": str(args.synthetic_3d_dir),
        "source_relations_used": list(REL),
        "source_n": len(source_rows),
        "targets": sorted(set(d for _, d in pairs)),
        "pairs": [f"{m}:{d}" for m, d in pairs],
        "head_direction_fit": "source only",
        "head_id_selection": "full target GT (same as original v6)",
        "per_sample_routing_uses_target_gt": False,
        "residual_geometry": "Synthetic-3D LRAB Real-Gray",
        "generation_success": "actual model.generate()",
    }
    write_json(root_out / "protocol.json", protocol)

    print(f"[PLAN] pairs={pairs}")
    print(
        "[PROTOCOL] Synthetic-3D LRAB source only | "
        f"source N={len(source_rows)} | target GT selects head ID | "
        "actual generation is the recovery success criterion"
    )

    head_summaries: List[Dict[str, Any]] = []
    control_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # Allow append/resume when --overwrite is omitted.
    old_control = root_out / "cross_model_control_summary.json"
    if old_control.exists() and not args.overwrite:
        try:
            obj = json.loads(old_control.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                control_summaries = [dict(x) for x in obj if isinstance(x, dict)]
        except Exception:
            pass

    for model_alias in models_in_order:
        model_pairs = [(m, d) for m, d in pairs if m == model_alias]
        model = processor = None
        try:
            model, processor, decoder_layers, n_heads, head_dim, spec = hs.load_model_bundle(model_alias, args)

            # -----------------------------------------------------------------
            # Phase 1: Synthetic-3D frozen head directions + target head ID.
            # -----------------------------------------------------------------
            model_head_dir = headscan_root / model_alias
            model_head_dir.mkdir(parents=True, exist_ok=True)
            source_cache = model_head_dir / "synthetic3d_lrab_head_vectors.npz"
            source_pack = hs.extract_vectors(
                source_rows,
                model,
                processor,
                decoder_layers,
                n_heads,
                head_dim,
                model_alias,
                "synthetic3d_lrab",
                args,
                source_cache,
                with_generation=False,
            )
            Xs = np.asarray(source_pack["vectors"])
            ys = np.asarray(source_pack["relation"], dtype=object)
            center, dirs = hs.fit_source_codebooks(Xs, ys)
            np.savez_compressed(
                model_head_dir / "synthetic3d_lrab_frozen_codebooks.npz",
                center=center.astype(np.float32),
                directions=dirs.astype(np.float32),
                relations=np.asarray(REL, dtype=object),
                vector_definition=np.asarray(
                    "pre-W_O per-head [(subject-reference)_real - (subject-reference)_gray]",
                    dtype=object,
                ),
                source=np.asarray("Blender Synthetic-3D LRAB subset", dtype=object),
            )

            for _m, dataset in model_pairs:
                try:
                    hs_summary = hs.run_pair(
                        model_alias,
                        dataset,
                        model,
                        processor,
                        decoder_layers,
                        n_heads,
                        head_dim,
                        source_pack,
                        center,
                        dirs,
                        args,
                        headscan_root,
                    )
                    hs_summary["script_version"] = SCRIPT_VERSION + "/headscan"
                    hs_summary["source"] = "Blender Synthetic-3D LRAB subset"
                    hs_summary["source_path"] = str(args.synthetic_3d_dir)
                    hs_summary["source_relations"] = list(REL)
                    write_json(headscan_root / model_alias / dataset / "summary.json", hs_summary)
                    head_summaries = [
                        x for x in head_summaries
                        if not (x.get("model") == model_alias and x.get("dataset") == dataset)
                    ]
                    head_summaries.append(hs_summary)
                except Exception as exc:
                    failure = {
                        "stage": "headscan",
                        "model": model_alias,
                        "dataset": dataset,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-60:],
                    }
                    failures.append(failure)
                    write_json(root_out / "failures.json", failures)
                    print(f"[HEADSCAN FAILED] {model_alias}/{dataset}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise

            # Only keep pairs that produced usable routing artifacts.
            valid_model_pairs = []
            for _m, dataset in model_pairs:
                try:
                    v6.validate_headscan_summary(headscan_root / model_alias / dataset / "summary.json", model_alias, dataset)
                    valid_model_pairs.append((_m, dataset))
                except Exception as exc:
                    print(f"[CONTROL SKIP] {model_alias}/{dataset}: no valid head routing: {exc}")
            if not valid_model_pairs:
                continue

            # -----------------------------------------------------------------
            # Phase 2: Synthetic-3D residual H/V geometry + v6 controller.
            # -----------------------------------------------------------------
            for p in model.parameters():
                p.requires_grad_(False)

            n_layers = len(decoder_layers)
            control_layers = overrides.get(model_alias, v6.relative7_layers(n_layers))
            if len(control_layers) != 7:
                raise RuntimeError(
                    f"{model_alias}: controller must use exactly 7 layers; got {control_layers}"
                )
            if min(control_layers) < 0 or max(control_layers) >= n_layers:
                raise RuntimeError(
                    f"{model_alias}: invalid controller layers {control_layers} for n_layers={n_layers}"
                )
            print(f"\n[MODEL CONTROLLER] {model_alias}: n_layers={n_layers}, layers={control_layers}")

            geom_model_dir = geom_root / model_alias
            geom_model_dir.mkdir(parents=True, exist_ok=True)
            geom_cache = geom_model_dir / "synthetic3d_lrab_residual_real_minus_gray.npz"
            Xg, yg = v6.build_synthetic_geometry_cache(
                model_alias=model_alias,
                model=model,
                processor_or_backend=processor,
                decoder_layers=decoder_layers,
                layers=control_layers,
                source_rows=source_rows,
                cache_path=geom_cache,
                args=args,
            )
            geom, axis_df = v6.fit_geometry(Xg, yg, control_layers)
            axis_df.to_csv(geom_model_dir / "synthetic3d_lrab_HV_geometry.csv", index=False)

            for _m, dataset in valid_model_pairs:
                try:
                    result = v6.run_pair(
                        model_alias=model_alias,
                        dataset=dataset,
                        model=model,
                        processor_or_backend=processor,
                        decoder_layers=decoder_layers,
                        layers=control_layers,
                        geom=geom,
                        args=args,
                        root_out=control_root,
                    )
                    patch_control_metadata(control_root / model_alias / dataset / "metadata.json", args)

                    control_summaries = [
                        x for x in control_summaries
                        if not (x.get("model") == model_alias and x.get("dataset") == dataset)
                    ]
                    control_summaries.append(result)
                    write_csv(root_out / "cross_model_control_summary.csv", control_summaries)
                    write_json(root_out / "cross_model_control_summary.json", control_summaries)
                    write_csv(control_root / "cross_model_control_summary.csv", control_summaries)
                    write_json(control_root / "cross_model_control_summary.json", control_summaries)
                except Exception as exc:
                    failure = {
                        "stage": "control",
                        "model": model_alias,
                        "dataset": dataset,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-60:],
                    }
                    failures.append(failure)
                    write_json(root_out / "failures.json", failures)
                    print(f"[CONTROL FAILED] {model_alias}/{dataset}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise

        except Exception as exc:
            failure = {
                "stage": "model_or_source",
                "model": model_alias,
                "dataset": "__model__",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-60:],
            }
            failures.append(failure)
            write_json(root_out / "failures.json", failures)
            print(f"[MODEL FAILED] {model_alias}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            cleanup_cuda()

    write_json(root_out / "headscan_summaries.json", head_summaries)
    write_csv(root_out / "cross_model_control_summary.csv", control_summaries)
    write_json(root_out / "cross_model_control_summary.json", control_summaries)
    write_json(root_out / "failures.json", failures)

    print("\n" + "=" * 180)
    print("FINAL SYNTHETIC-3D LRAB -> COCO / CONTROLLED-A RECOVERY SUMMARY")
    print("=" * 180)
    if control_summaries:
        print(pd.DataFrame(control_summaries).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("NO SUCCESSFUL PAIRS")
    print(f"\nSaved to: {root_out}")


if __name__ == "__main__":
    main()
