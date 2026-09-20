#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_crossdataset_source_direction_control_v1.py

Cross-real-dataset direction-source transfer for the current v6 spatial-control
pipeline in AdaptVis/llava16.

Goal
====
Test whether spatial directions transfer across real datasets while keeping the
current v6 recovery logic unchanged. The clean default changes only the actuator
H/V direction source; an optional mode changes both reader and actuator sources.

Default transfers:
    COCO         -> Controlled-A
    Controlled-A -> COCO

Two experiment scopes are supported:
    geometry_only (default): replace only the residual-stream H/V geometry
                             source; keep the existing Synthetic-400 reader.
    both:                    replace both reader directions and H/V geometry
                             with the real source dataset.

Default models (first requested sweep):
    qwen-3b, qwen-7b, llava-7b, llava-13b

For each model and source->target transfer, `--transfer-scope both` uses:

A) Reader directions (pre-W_O attention-head space)
   1. Extract Real-Gray subject-reference residual vectors on the SOURCE dataset.
   2. Fit frozen L/R/A/B directions on SOURCE only:
          center_{L,H} = mean(v)
          d_{L,H,r} = unit(mean(v | r) - center_{L,H})
   3. Extract the same vectors on TARGET.
   4. Apply the frozen SOURCE directions to every TARGET head.
   5. As in the current v6 protocol, use full TARGET GT only to choose the best
      head ID. After head selection, each sample is routed by that head's own
      prediction (no per-sample GT in head mode).

B) Spatial actuator geometry (decoder residual stream)
   1. On SOURCE only, at the seven controller layers, compute
          q_l = (h_sub^real - h_ref^real) - (h_sub^gray - h_ref^gray)
   2. Fit four SOURCE class directions and construct H/V axes.
   3. Reuse v6 fit_geometry(), which orthonormalizes the H/V span into Q_l.

C) Target recovery
   Reuse eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py
   without changing its optimization:
      selected-head prediction -> target relation
      SOURCE-derived H/V subspace -> allowed edit space
      target sample gradients -> layer/axis allocation
      local minimum-residual step + trust cap
      actual model.generate() -> only success criterion
      binary refinement after the first generation flip

Important interpretation
========================
`geometry_only` is the clean direction-transfer test: the existing Synthetic-400
reader/routing is held fixed and only the actuator H/V source changes from
Synthetic-400 to COCO or Controlled-A.

`both` removes Synthetic-400 from both reader directions and actuator geometry.
In this mode the default head-ID selection remains TARGET-supervised to match the
current v6 protocol exactly, so the `head` condition is not fully target-label-free.

Run from the AdaptVis llava16 repository root, next to:
    scan_synthetic_frozen_direction_heads_7models_v2.py
    eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py

Recommended smoke test
======================
CUDA_VISIBLE_DEVICES=0 python -u eval_crossdataset_source_direction_control_v1.py \
  --models qwen-3b,qwen-7b,llava-7b,llava-13b \
  --transfers coco:controlled_a,controlled_a:coco \
  --transfer-scope geometry_only \
  --synthetic-headscan-dir output/syn400_direction_head_7models_targetfull_v2 \
  --eval-max-samples 80 \
  --output-dir output/crossdataset_source_direction_control_v1

Full run (same four models)
===========================
CUDA_VISIBLE_DEVICES=0 python -u eval_crossdataset_source_direction_control_v1.py \
  --models qwen-3b,qwen-7b,llava-7b,llava-13b \
  --transfers coco:controlled_a,controlled_a:coco \
  --transfer-scope geometry_only \
  --synthetic-headscan-dir output/syn400_direction_head_7models_targetfull_v2 \
  --eval-max-samples 0 \
  --generation-margin-increment 4.0 \
  --max-residual-step 5000 \
  --max-steps 64 \
  --output-dir output/crossdataset_source_direction_control_v1

Outputs
=======
<output-dir>/head_vector_cache/<model>/<dataset>.npz
<output-dir>/reader/<source>_to_<target>/<model>/<target>/
    head_ranking.csv
    samples_best_head.csv
    summary.json
<output-dir>/geometry_cache/<model>/<source>/
    source_residual_real_minus_gray.npz
    source_HV_geometry.csv
<output-dir>/control/<source>_to_<target>/<model>/<target>/
    per_sample.jsonl
    per_sample.csv
    summary.csv
    metadata.json
<output-dir>/reader_transfer_summary.csv
<output-dir>/cross_source_control_summary.csv
<output-dir>/failures.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import shutil
import traceback
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
        "Run from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as ctrl
except Exception as exc:
    raise SystemExit(
        "Could not import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py. "
        "Run from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "cross-real-dataset-source-direction-control-v1"
REL = ("left", "right", "above", "below")
DEFAULT_MODELS = ("qwen-3b", "qwen-7b", "llava-7b", "llava-13b")
VALID_DATASETS = ("coco", "controlled_a")
DEFAULT_TRANSFERS = (("coco", "controlled_a"), ("controlled_a", "coco"))


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


def parse_csv_names(text: str, allowed: Sequence[str]) -> List[str]:
    raw = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in raw if x not in allowed]
    if bad:
        raise ValueError(f"Unknown values {bad}; allowed={list(allowed)}")
    if not raw:
        raise ValueError("Empty list")
    return raw


def parse_transfers(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad transfer {item!r}; expected source:target")
        src, tgt = [x.strip() for x in item.split(":", 1)]
        if src not in VALID_DATASETS or tgt not in VALID_DATASETS:
            raise ValueError(f"Bad transfer {item!r}; datasets={VALID_DATASETS}")
        if src == tgt:
            raise ValueError(f"Source and target must differ: {item!r}")
        pair = (src, tgt)
        if pair not in out:
            out.append(pair)
    if not out:
        raise ValueError("No transfers selected")
    return out


def transfer_name(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment matrix.
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument(
        "--transfers",
        default="coco:controlled_a,controlled_a:coco",
        help="Comma-separated source:target dataset pairs.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--transfer-scope",
        default="geometry_only",
        choices=["geometry_only", "both"],
        help=("geometry_only: keep current Synthetic-400 reader and replace only H/V geometry; "
              "both: source real dataset provides both reader directions and H/V geometry."),
    )
    p.add_argument(
        "--synthetic-headscan-dir",
        default="output/syn400_direction_head_7models_targetfull_v2",
        help="Existing current-v6 Synthetic-400 reader outputs, used by geometry_only.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    # Shared model/data args expected by hs and ctrl.
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    # Kept because imported repo helpers expect these attributes, but this script
    # deliberately never calls hs.load_synthetic().
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)
    p.add_argument("--target-max-samples", type=int, default=None)

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

    # Reader protocol. 1.0 intentionally matches current v6.
    p.add_argument("--head-selection-frac", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-target-vector-success-rate", type=float, default=0.95)
    p.add_argument("--cache-vectors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--with-generation", action=argparse.BooleanOptionalAction, default=False)

    # Controller layers.
    p.add_argument(
        "--controller-layers",
        default="",
        help="Optional per-model overrides, e.g. qwen-3b=20-26,llava-7b=17-23.",
    )
    p.add_argument("--geometry-cache-dir", default=None)

    # Current v6 optimization args.
    p.add_argument("--modes", default="head,oracle", help="Comma-separated: head,oracle")
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
    if abs(float(args.head_selection_frac) - 1.0) > 1e-12:
        raise ValueError(
            "This v1 intentionally matches current v6 head selection. "
            "Use --head-selection-frac 1.0. A held-out/source-only head-selection "
            "variant should be treated as a separate experiment."
        )
    return args


def load_rows(dataset: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = hs.load_target(dataset, args)
    if not rows:
        raise RuntimeError(f"No rows loaded for {dataset}")
    return rows


def extract_dataset_head_vectors(
    *,
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    model_alias: str,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    n_heads: int,
    head_dim: int,
    args: argparse.Namespace,
    cache_root: Path,
) -> Dict[str, Any]:
    cache_path = cache_root / model_alias / f"{dataset}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return hs.extract_vectors(
        rows,
        model,
        processor,
        decoder_layers,
        n_heads,
        head_dim,
        model_alias,
        dataset,
        args,
        cache_path,
        with_generation=False,
    )


def build_reader_transfer(
    *,
    model_alias: str,
    source_dataset: str,
    target_dataset: str,
    source_pack: Mapping[str, Any],
    target_pack: Mapping[str, Any],
    requested_target_n: int,
    args: argparse.Namespace,
    reader_root: Path,
) -> Dict[str, Any]:
    """Fit SOURCE frozen directions, select one head on TARGET, save v6 contract."""
    Xs = np.asarray(source_pack["vectors"])
    ys = np.asarray(source_pack["relation"], dtype=object)
    Xt = np.asarray(target_pack["vectors"])
    yt = np.asarray(target_pack["relation"], dtype=object)

    if len(ys) == 0 or len(yt) == 0:
        raise RuntimeError(f"Empty source/target vectors for {source_dataset}->{target_dataset}")

    coverage = float(len(yt) / max(1, int(requested_target_n)))
    if coverage < float(args.min_target_vector_success_rate):
        raise RuntimeError(
            f"{model_alias}/{source_dataset}->{target_dataset}: target vector coverage "
            f"{coverage:.4f} < {float(args.min_target_vector_success_rate):.4f}"
        )

    # Frozen relation codebook comes ONLY from the chosen source dataset.
    center, dirs = hs.fit_source_codebooks(Xs, ys)

    selection_idx = np.arange(len(yt), dtype=np.int64)
    test_idx = selection_idx.copy()
    ranking, predictions = hs.rank_heads(
        Xs, ys, Xt, yt, center, dirs, selection_idx, test_idx
    )
    if not ranking:
        raise RuntimeError("Head ranking is empty")

    best = ranking[0]
    key = (int(best["layer"]), int(best["head"]))
    best_pred, best_margin = predictions[key]

    pair_dir = reader_root / transfer_name(source_dataset, target_dataset) / model_alias / target_dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    hs.write_csv(pair_dir / "head_ranking.csv", ranking)

    sids = np.asarray(target_pack["sid"], dtype=np.int64)
    sample_rows: List[Dict[str, Any]] = []
    for i in range(len(yt)):
        sample_rows.append({
            "row_index": int(i),
            "sid": int(sids[i]),
            "split": "eval",
            "gt": str(yt[i]),
            "best_head": str(best["head_name"]),
            "head_pred": str(best_pred[i]),
            "head_correct": int(best_pred[i] == yt[i]),
            "head_margin": float(best_margin[i]),
            "direction_source_dataset": source_dataset,
            "target_dataset": target_dataset,
        })
    hs.write_csv(pair_dir / "samples_best_head.csv", sample_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model_alias,
        "dataset": target_dataset,  # required by current v6 validator
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "control": args.control,
        "source": source_dataset,
        "direction_source": f"{source_dataset} only",
        "direction_fit_uses_target_gt": False,
        "head_selection_uses_target_gt": True,
        "head_selection_frac": 1.0,
        "controller_compatible": True,
        "target_requested_n": int(requested_target_n),
        "target_vector_n": int(len(yt)),
        "target_vector_success_rate": float(coverage),
        "source_vector_n": int(len(ys)),
        "selection_n": int(len(yt)),
        "eval_n": int(len(yt)),
        "target_n": int(len(yt)),
        "best_head": str(best["head_name"]),
        "best_layer": int(best["layer"]),
        "best_head_index": int(best["head"]),
        "best_source_self_acc": float(best["syn_self_acc"]),
        "best_selection_acc": float(best["selection_acc"]),
        "best_test_acc": float(best["test_acc"]),
        "best_all_target_acc": float(best["all_target_acc"]),
        "top_heads": ranking[: int(args.top_k)],
    }
    hs.write_json(pair_dir / "summary.json", summary)
    return summary


def geometry_for_source(
    *,
    model_alias: str,
    source_dataset: str,
    source_rows: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    controller_layers: Sequence[int],
    args: argparse.Namespace,
    geometry_root: Path,
) -> Tuple[dict, pd.DataFrame, Path]:
    """Use current v6 residual H/V construction, but on a real source dataset."""
    source_dir = geometry_root / model_alias / source_dataset
    source_dir.mkdir(parents=True, exist_ok=True)
    cache_path = source_dir / "source_residual_real_minus_gray.npz"

    # Despite the historical function name, this helper only assumes rows contain
    # image_path/question_text/subject/reference/relation. It is source-generic.
    Xg, yg = ctrl.build_synthetic_geometry_cache(
        model_alias=model_alias,
        model=model,
        processor_or_backend=processor,
        decoder_layers=decoder_layers,
        layers=controller_layers,
        source_rows=source_rows,
        cache_path=cache_path,
        args=args,
    )
    geom, axis_df = ctrl.fit_geometry(Xg, yg, controller_layers)
    axis_path = source_dir / "source_HV_geometry.csv"
    axis_df = axis_df.copy()
    axis_df.insert(0, "source_dataset", source_dataset)
    axis_df.to_csv(axis_path, index=False)
    return geom, axis_df, cache_path


def reader_contract_root(
    reader_root: Path, source_dataset: str, target_dataset: str
) -> Path:
    """Root whose child layout is <model>/<target>/summary.json for ctrl.run_pair."""
    return reader_root / transfer_name(source_dataset, target_dataset)


def main() -> None:
    args = parse_args()
    hs.seed_all(int(args.seed))

    models = parse_csv_names(args.models, hs.MODEL_ORDER)
    transfers = parse_transfers(args.transfers)
    needed_datasets = sorted({d for pair in transfers for d in pair})

    root = Path(args.output_dir)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    reader_root = root / "reader"
    vector_cache_root = root / "head_vector_cache"
    geometry_root = (
        Path(args.geometry_cache_dir)
        if args.geometry_cache_dir
        else root / "geometry_cache"
    )
    control_root = root / "control"

    # v6's run_pair reads this mutable arg. We set it transfer-by-transfer below.
    args.headscan_dir = ""

    layer_overrides = ctrl.parse_layer_overrides(args.controller_layers)
    reader_summary_rows: List[dict] = []
    control_summary_rows: List[dict] = []
    failures: List[dict] = []

    protocol = {
        "script_version": SCRIPT_VERSION,
        "models": models,
        "transfers": [f"{s}->{t}" for s, t in transfers],
        "transfer_scope": args.transfer_scope,
        "direction_source": (
            "real source dataset H/V geometry only; Synthetic-400 reader fixed"
            if args.transfer_scope == "geometry_only"
            else "real source dataset for both reader and H/V geometry"
        ),
        "reader_space": "pre-W_O attention-head Real-Gray subject-reference residual",
        "actuator_space": "decoder residual-stream Real-Gray object-pair H/V subspace",
        "head_id_selection": "full target GT (matches current v6)",
        "per_sample_head_routing_uses_target_gt": False,
        "controller": "reuse eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py",
        "controller_modes": args.modes,
    }
    write_json(root / "protocol.json", protocol)

    for model_alias in models:
        model = processor = None
        try:
            model, processor, decoder_layers, n_heads, head_dim, _spec = hs.load_model_bundle(
                model_alias, args
            )
            for p in model.parameters():
                p.requires_grad_(False)

            n_layers = len(decoder_layers)
            controller_layers = layer_overrides.get(
                model_alias, ctrl.relative7_layers(n_layers)
            )
            if len(controller_layers) != 7:
                raise RuntimeError(
                    f"{model_alias}: expected exactly 7 controller layers, got {controller_layers}"
                )
            print(
                f"\n[MODEL] {model_alias} | decoder_layers={n_layers} "
                f"| controller_layers={controller_layers}",
                flush=True,
            )

            # Load each real dataset once for this model.
            rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {
                d: load_rows(d, args) for d in needed_datasets
            }

            # In `both`, extract pre-W_O head vectors once per real dataset.
            # In `geometry_only`, keep the existing Synthetic-400 reader fixed,
            # so no real-dataset head-vector scan is needed here.
            head_pack_by_dataset: Dict[str, Dict[str, Any]] = {}
            if args.transfer_scope == "both":
                for dataset in needed_datasets:
                    print(f"\n[HEAD VECTORS] {model_alias} / {dataset}", flush=True)
                    head_pack_by_dataset[dataset] = extract_dataset_head_vectors(
                        dataset=dataset,
                        rows=rows_by_dataset[dataset],
                        model_alias=model_alias,
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        args=args,
                        cache_root=vector_cache_root,
                    )

            # Residual H/V geometry is source-specific; cache once per source.
            geom_by_source: Dict[str, dict] = {}
            for source_dataset in sorted({s for s, _ in transfers}):
                print(
                    f"\n[SOURCE GEOMETRY] {model_alias} / source={source_dataset}",
                    flush=True,
                )
                geom, _axis_df, _cache_path = geometry_for_source(
                    model_alias=model_alias,
                    source_dataset=source_dataset,
                    source_rows=rows_by_dataset[source_dataset],
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    controller_layers=controller_layers,
                    args=args,
                    geometry_root=geometry_root,
                )
                geom_by_source[source_dataset] = geom

            for source_dataset, target_dataset in transfers:
                tag = transfer_name(source_dataset, target_dataset)
                try:
                    print("\n" + "=" * 160)
                    print(
                        f"TRANSFER {model_alias}: {source_dataset} -> {target_dataset}",
                        flush=True,
                    )
                    print("=" * 160)

                    if args.transfer_scope == "both":
                        reader_summary = build_reader_transfer(
                            model_alias=model_alias,
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            source_pack=head_pack_by_dataset[source_dataset],
                            target_pack=head_pack_by_dataset[target_dataset],
                            requested_target_n=len(rows_by_dataset[target_dataset]),
                            args=args,
                            reader_root=reader_root,
                        )
                        reader_summary_rows.append(reader_summary)
                        write_csv(root / "reader_transfer_summary.csv", reader_summary_rows)
                        # Cross-source reader contract written above.
                        args.headscan_dir = str(
                            reader_contract_root(reader_root, source_dataset, target_dataset)
                        )
                    else:
                        # Clean geometry-source transfer: reuse exactly the reader
                        # already used by current v6, changing only H/V geometry.
                        args.headscan_dir = str(Path(args.synthetic_headscan_dir))
                    pair_control_root = control_root / tag
                    result = ctrl.run_pair(
                        model_alias=model_alias,
                        dataset=target_dataset,
                        model=model,
                        processor_or_backend=processor,
                        decoder_layers=decoder_layers,
                        layers=controller_layers,
                        geom=geom_by_source[source_dataset],
                        args=args,
                        root_out=pair_control_root,
                    )
                    result = dict(result)
                    result["source_dataset"] = source_dataset
                    result["target_dataset"] = target_dataset
                    result["transfer"] = tag
                    result["transfer_scope"] = args.transfer_scope
                    result["reader_direction_source"] = (
                        source_dataset if args.transfer_scope == "both" else "synthetic400"
                    )
                    result["actuator_geometry_source"] = source_dataset
                    control_summary_rows.append(result)
                    write_csv(root / "cross_source_control_summary.csv", control_summary_rows)
                    write_json(root / "cross_source_control_summary.json", control_summary_rows)

                    # Add source provenance to v6 metadata without changing v6.
                    meta_path = (
                        pair_control_root / model_alias / target_dataset / "metadata.json"
                    )
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta.update({
                            "outer_script_version": SCRIPT_VERSION,
                            "transfer_scope": args.transfer_scope,
                            "reader_direction_source_dataset": (
                                source_dataset if args.transfer_scope == "both" else "synthetic400"
                            ),
                            "residual_geometry_source_dataset": source_dataset,
                            "synthetic400_used_for_reader": bool(args.transfer_scope == "geometry_only"),
                            "synthetic400_used_for_residual_geometry": False,
                            "target_dataset": target_dataset,
                        })
                        write_json(meta_path, meta)

                except Exception as exc:
                    failure = {
                        "model": model_alias,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-80:],
                    }
                    failures.append(failure)
                    write_json(root / "failures.json", failures)
                    print(
                        f"[TRANSFER FAILED] {model_alias} {source_dataset}->{target_dataset}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if args.fail_fast:
                        raise

        except Exception as exc:
            failure = {
                "model": model_alias,
                "source_dataset": "__model__",
                "target_dataset": "__model__",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-80:],
            }
            failures.append(failure)
            write_json(root / "failures.json", failures)
            print(f"[MODEL FAILED] {model_alias}: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            cleanup_cuda()

    write_csv(root / "reader_transfer_summary.csv", reader_summary_rows)
    write_csv(root / "cross_source_control_summary.csv", control_summary_rows)
    write_json(root / "cross_source_control_summary.json", control_summary_rows)
    write_json(root / "failures.json", failures)

    print("\n" + "=" * 180)
    print("FINAL CROSS-SOURCE CONTROL SUMMARY")
    print("=" * 180)
    if control_summary_rows:
        cols = [
            "model", "source_dataset", "target_dataset", "N",
            "baseline_accuracy", "head_readout_accuracy",
            "head_generation_accuracy", "head_gain",
            "oracle_generation_accuracy", "oracle_gain",
            "head_W2C", "head_C2W", "oracle_W2C", "oracle_C2W",
        ]
        df = pd.DataFrame(control_summary_rows)
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("NO SUCCESSFUL TRANSFERS")
    print(f"\nSaved to: {root}")


if __name__ == "__main__":
    main()
