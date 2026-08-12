#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan where the causal spatial-relation state first emerges before L18.

Two stages
==========
Phase A scans L0-L18 with only:
  residual_carry: y' = y + (x_source - x_target)
  block_output  : y' = y_source
for ROLE and IDENTITY alignments.

The primary curve is evaluated against a fixed downstream DAS relation-U
(default U@L22 D16).  residual_carry@L asks how much source/opposite relation
state is already present when ENTERING block L.  block_output@L asks how much
is present after the whole block.

Phase B automatically selects the strongest transition layers and tests:
  attention_out
  mlp_out
for ROLE and IDENTITY, so we can determine which local component constructs
or refines the relation state near its emergence point.

Recommended:
CUDA_VISIBLE_DEVICES=0 python -u scan_relation_u_emergence_v1.py \\
  --model qwen-3b \\
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \\
  --das-dir output/qwen3b_relation_das_full \\
  --component-helper validate_attention_mlp_residual_to_das_u_v1.py \\
  --writer-helper validate_writer_to_das_u_v1.py \\
  --scan-layers 0-18 \\
  --u-layers 22,23,24,25 \\
  --primary-u-layer 22 \\
  --u-dim 16 \\
  --refine-top-n 5 \\
  --refine-neighborhood 1 \\
  --sample-scope das_eval \\
  --pair-status both_correct \\
  --max-samples 0 \\
  --device cuda:0 \\
  --output-dir output/qwen3b_relation_u_emergence_l0_l18 \\
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

VERSION = "relation-u-emergence-scan-v1"
RELATIONS = ("left", "right", "above", "below")


def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_layer_spec(text: str) -> List[int]:
    out, seen = [], set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            aa, bb = int(a), int(b)
            vals = range(min(aa, bb), max(aa, bb) + 1)
        else:
            vals = [int(part)]
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
    if not out:
        raise ValueError(f"empty layer spec: {text!r}")
    return sorted(out)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--object-state", default="mean")

    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
    )
    p.add_argument("--das-dir", default="output/qwen3b_relation_das_full")
    p.add_argument(
        "--component-helper",
        default="validate_attention_mlp_residual_to_das_u_v1.py",
    )
    p.add_argument(
        "--writer-helper",
        default="validate_writer_to_das_u_v1.py",
    )

    p.add_argument("--scan-layers", default="0-18")
    p.add_argument("--u-layers", default="22,23,24,25")
    p.add_argument("--primary-u-layer", type=int, default=22)
    p.add_argument("--u-dim", type=int, default=16)

    p.add_argument("--refine-top-n", type=int, default=5)
    p.add_argument("--refine-neighborhood", type=int, default=1)
    p.add_argument(
        "--refine-layers",
        default="",
        help="Explicit layers for attention/MLP refinement; bypass auto selection.",
    )

    p.add_argument(
        "--sample-scope",
        default="das_eval",
        choices=("das_eval", "all"),
    )
    p.add_argument(
        "--pair-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument(
        "--replacement-mode",
        default="tokenwise_resample",
        choices=("tokenwise_resample", "pooled_broadcast", "mean_shift"),
    )
    p.add_argument("--cache-dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def make_lookup(rows: Sequence[Mapping[str, Any]]):
    return {
        (int(r["intervention_layer"]), str(r["component"]), int(r["u_layer"])): r
        for r in rows
    }


def make_final_lookup(rows: Sequence[Mapping[str, Any]]):
    return {
        (int(r["intervention_layer"]), str(r["component"])): r
        for r in rows
    }


def build_curve(scan_layers, primary_u, ucmp, fcmp):
    u = make_lookup(ucmp)
    f = make_final_lookup(fcmp)
    rows = []

    for L in scan_layers:
        carry = u.get((L, "residual_carry", primary_u))
        block = u.get((L, "block_output", primary_u))
        if carry is None or block is None:
            continue

        cr = safe_float(carry["role_progress"])
        ci = safe_float(carry["identity_progress"])
        cs = safe_float(carry["role_minus_identity"])
        br = safe_float(block["role_progress"])
        bi = safe_float(block["identity_progress"])
        bs = safe_float(block["role_minus_identity"])
        cf = f.get((L, "residual_carry"))
        bf = f.get((L, "block_output"))

        rows.append({
            "layer": L,
            "primary_u_layer": primary_u,
            "carry_role_progress": cr,
            "carry_identity_progress": ci,
            "carry_role_minus_identity": cs,
            "block_role_progress": br,
            "block_identity_progress": bi,
            "block_role_minus_identity": bs,
            "block_minus_carry_role": br - cr,
            "block_minus_carry_specific": bs - cs,
            "carry_final_source_follow": safe_float(cf["role_source_follow"]) if cf else float("nan"),
            "block_final_source_follow": safe_float(bf["role_source_follow"]) if bf else float("nan"),
        })

    rows.sort(key=lambda r: int(r["layer"]))
    by_layer = {int(r["layer"]): r for r in rows}

    for r in rows:
        L = int(r["layer"])
        prev = by_layer.get(L - 1)
        nxt = by_layer.get(L + 1)
        r["carry_growth_from_prev"] = (
            r["carry_role_minus_identity"] - prev["carry_role_minus_identity"]
            if prev is not None else float("nan")
        )
        r["carry_growth_to_next"] = (
            nxt["carry_role_minus_identity"] - r["carry_role_minus_identity"]
            if nxt is not None else float("nan")
        )

        # Ranking heuristic only. These quantities are not assumed additive.
        local_gain = max(0.0, safe_float(r["block_minus_carry_specific"]))
        next_gain = max(0.0, safe_float(r["carry_growth_to_next"]))
        r["formation_score"] = local_gain + next_gain

    return rows


def select_refine_layers(curve, top_n, neighborhood, allowed_layers):
    ranked = sorted(
        [dict(r) for r in curve],
        key=lambda r: safe_float(r["formation_score"]),
        reverse=True,
    )
    allowed = set(map(int, allowed_layers))
    seeds = [int(r["layer"]) for r in ranked[:max(0, top_n)]]
    selected = set()
    for seed in seeds:
        for d in range(-neighborhood, neighborhood + 1):
            if seed + d in allowed:
                selected.add(seed + d)
    return sorted(selected), ranked


def print_curve(rows):
    print("\n" + "=" * 158)
    print("EARLY RELATION-U EMERGENCE CURVE")
    print("=" * 158)
    print(
        f"{'L':>4} {'carryR':>9} {'carryID':>9} {'carrySpec':>10} "
        f"{'blockR':>9} {'blockSpec':>10} {'block-carry':>12} "
        f"{'dCarryNext':>11} {'formation':>10} {'carrySrc':>10}"
    )
    print("-" * 158)
    for r in rows:
        print(
            f"L{int(r['layer']):<3d} "
            f"{safe_float(r['carry_role_progress']):>+8.3f} "
            f"{safe_float(r['carry_identity_progress']):>+8.3f} "
            f"{safe_float(r['carry_role_minus_identity']):>+9.3f} "
            f"{safe_float(r['block_role_progress']):>+8.3f} "
            f"{safe_float(r['block_role_minus_identity']):>+9.3f} "
            f"{safe_float(r['block_minus_carry_specific']):>+11.3f} "
            f"{safe_float(r['carry_growth_to_next']):>+10.3f} "
            f"{safe_float(r['formation_score']):>+9.3f} "
            f"{100*safe_float(r['carry_final_source_follow']):>9.2f}%"
        )


def print_refine_table(ucmp, primary_u, refine_layers):
    lookup = make_lookup(ucmp)
    print("\n" + "=" * 126)
    print("LOCAL COMPONENT REFINEMENT")
    print("=" * 126)
    print(f"{'L':>4} {'attnR':>9} {'attnSpec':>10} {'mlpR':>9} {'mlpSpec':>10} {'carryR':>9} {'blockR':>9}")
    print("-" * 126)
    for L in refine_layers:
        a = lookup.get((L, "attention_out", primary_u))
        m = lookup.get((L, "mlp_out", primary_u))
        c = lookup.get((L, "residual_carry", primary_u))
        b = lookup.get((L, "block_output", primary_u))

        def v(row, key):
            return safe_float(row[key]) if row is not None else float("nan")

        print(
            f"L{L:<3d} {v(a,'role_progress'):>+8.3f} {v(a,'role_minus_identity'):>+9.3f} "
            f"{v(m,'role_progress'):>+8.3f} {v(m,'role_minus_identity'):>+9.3f} "
            f"{v(c,'role_progress'):>+8.3f} {v(b,'role_progress'):>+8.3f}"
        )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scan_layers = parse_layer_spec(args.scan_layers)
    u_layers = parse_layer_spec(args.u_layers)
    if args.primary_u_layer not in u_layers:
        raise ValueError("primary U layer must be included in --u-layers")
    if max(scan_layers) > args.primary_u_layer:
        raise ValueError("scan layers must be <= primary U layer")

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"output directory not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    comp = import_file(Path(args.component_helper), "_em_comp")
    writer = import_file(Path(args.writer_helper), "_em_writer")
    ioi = import_file(Path("analyze_coco_ioi_backward_circuit_v1.py"), "_em_ioi")
    producer = import_file(Path("analyze_coco_producer_qk_ov_v1.py"), "_em_prod")
    receiver = import_file(Path("analyze_coco_receiver_qkv_v1.py"), "_em_recv")
    v3 = import_file(Path("analyze_spatial_storage_transport_utilization_v3.py"), "_em_v3")
    base = import_file(Path("analyze_coco_centroid_generation_step1_v4.py"), "_em_base")
    attn_helper = import_file(Path("analyze_coco_flip_attention_spatial_vectors_v1.py"), "_em_attn")

    rows = [
        r for r in writer.read_jsonl(Path(args.source_output_dir) / "extraction.jsonl")
        if str(r.get("gt")) in RELATIONS
    ]

    dasdir = Path(args.das_dir)
    if args.sample_scope == "das_eval":
        cfg = json.loads((dasdir / "config.json").read_text(encoding="utf-8"))
        eval_sids = set(map(int, cfg.get("eval_sids", [])))
        if not eval_sids:
            raise RuntimeError("DAS config contains no eval_sids")
        rows = [r for r in rows if int(r["sid"]) in eval_sids]

    if args.pair_status != "all":
        rows = [
            r for r in rows
            if str(r.get("generation_pair_status", "")) == args.pair_status
        ]

    rows = writer.stratified_subset(
        rows=rows,
        limit=args.max_samples,
        seed=args.sample_seed,
    )
    if not rows:
        raise RuntimeError("no samples after filtering")

    bases = writer.load_das_bases(
        das_dir=dasdir,
        u_layers=u_layers,
        u_dim=args.u_dim,
    )

    model = processor = None
    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )
        model.eval()

        saved_max = args.max_samples
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)
        finally:
            args.max_samples = saved_max

        print("\n" + "=" * 128)
        print("EARLY RELATION-U EMERGENCE SCAN")
        print("=" * 128)
        print("N samples          :", len(rows))
        print("scan layers        :", scan_layers)
        print("U layers / dim     :", u_layers, "/", args.u_dim)
        print("primary U          :", args.primary_u_layer)
        print("Phase A            : residual_carry + block_output")
        print("Phase B            : attention_out + mlp_out near transitions")
        print("=" * 128, flush=True)

        # Cache x/a/m/y for all early layers once.
        cache, rows, additivity = comp.build_cache(
            args=args,
            rows=rows,
            layers=scan_layers,
            u_layers=u_layers,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            relation_token_map=relation_token_map,
            records_by_sid=records_by_sid,
            prompt_rows=prompt_rows,
            base=base,
            v3=v3,
            receiver=receiver,
            attention_helper=attn_helper,
            helper=writer,
            error_path=error_path,
        )
        write_csv(outdir / "component_additivity_audit.csv", additivity)

        all_u, all_f = [], []

        print("\nPHASE A: residual-carry / full-block scan", flush=True)
        for L in scan_layers:
            for component in ("residual_carry", "block_output"):
                for alignment in ("role", "identity"):
                    print(f">>> L{L} {component} {alignment.upper()}", flush=True)
                    ur, fr = comp.run_condition(
                        args=args,
                        layer=L,
                        component=component,
                        alignment=alignment,
                        rows=rows,
                        cache=cache,
                        u_layers=u_layers,
                        bases=bases,
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        records_by_sid=records_by_sid,
                        prompt_rows=prompt_rows,
                        base=base,
                        v3=v3,
                        receiver=receiver,
                        helper=writer,
                        error_path=error_path,
                        sample_path=outdir / f"samples_phaseA_L{L}_{component}_{alignment}.jsonl",
                    )
                    all_u.extend(ur)
                    all_f.append(fr)

            write_csv(outdir / "component_u_effects_all.csv", all_u)
            write_csv(outdir / "component_final_effects_all.csv", all_f)

        ucmp, fcmp = comp.comparisons(all_u, all_f)
        curve = build_curve(scan_layers, args.primary_u_layer, ucmp, fcmp)
        write_csv(outdir / "emergence_curve_phaseA.csv", curve)
        print_curve(curve)

        if args.refine_layers.strip():
            refine_layers = [
                L for L in parse_layer_spec(args.refine_layers)
                if L in set(scan_layers)
            ]
            ranked = sorted(
                [dict(r) for r in curve],
                key=lambda r: safe_float(r["formation_score"]),
                reverse=True,
            )
        else:
            refine_layers, ranked = select_refine_layers(
                curve,
                args.refine_top_n,
                args.refine_neighborhood,
                scan_layers,
            )

        print("\nSelected refinement layers:", refine_layers)
        print("Top transition seeds:")
        for i, r in enumerate(ranked[:args.refine_top_n], 1):
            print(
                f"  {i:02d}. L{int(r['layer'])} "
                f"formation={safe_float(r['formation_score']):+.3f} "
                f"carrySpec={safe_float(r['carry_role_minus_identity']):+.3f} "
                f"blockSpec={safe_float(r['block_role_minus_identity']):+.3f}"
            )

        print("\nPHASE B: Attention / MLP refinement", flush=True)
        for L in refine_layers:
            for component in ("attention_out", "mlp_out"):
                for alignment in ("role", "identity"):
                    print(f">>> L{L} {component} {alignment.upper()}", flush=True)
                    ur, fr = comp.run_condition(
                        args=args,
                        layer=L,
                        component=component,
                        alignment=alignment,
                        rows=rows,
                        cache=cache,
                        u_layers=u_layers,
                        bases=bases,
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        records_by_sid=records_by_sid,
                        prompt_rows=prompt_rows,
                        base=base,
                        v3=v3,
                        receiver=receiver,
                        helper=writer,
                        error_path=error_path,
                        sample_path=outdir / f"samples_phaseB_L{L}_{component}_{alignment}.jsonl",
                    )
                    all_u.extend(ur)
                    all_f.append(fr)

            write_csv(outdir / "component_u_effects_all.csv", all_u)
            write_csv(outdir / "component_final_effects_all.csv", all_f)

        ucmp, fcmp = comp.comparisons(all_u, all_f)
        write_csv(outdir / "emergence_all_u.csv", ucmp)
        write_csv(outdir / "final_comparison.csv", fcmp)

        refine_rows = [
            r for r in ucmp
            if int(r["intervention_layer"]) in set(refine_layers)
            and str(r["component"]) in ("attention_out", "mlp_out")
        ]
        write_csv(outdir / "refine_component_comparison.csv", refine_rows)

        curve = build_curve(scan_layers, args.primary_u_layer, ucmp, fcmp)
        write_csv(outdir / "emergence_curve.csv", curve)
        print_curve(curve)
        print_refine_table(ucmp, args.primary_u_layer, refine_layers)

        summary = {
            "version": VERSION,
            "model": args.model,
            "N": len(rows),
            "scan_layers": scan_layers,
            "u_layers": u_layers,
            "primary_u_layer": args.primary_u_layer,
            "u_dim": args.u_dim,
            "refine_layers": refine_layers,
            "top_transitions": [
                {
                    "rank": i + 1,
                    "layer": int(r["layer"]),
                    "formation_score": safe_float(r["formation_score"]),
                    "carry_role_minus_identity": safe_float(r["carry_role_minus_identity"]),
                    "block_role_minus_identity": safe_float(r["block_role_minus_identity"]),
                    "block_minus_carry_specific": safe_float(r["block_minus_carry_specific"]),
                    "carry_growth_to_next": safe_float(r["carry_growth_to_next"]),
                }
                for i, r in enumerate(ranked[:10])
            ],
            "caution": (
                "The scan locates a causal precursor/state aligned with later DAS-U. "
                "Do not automatically call an early state an abstract visual relation."
            ),
            "audit": audit,
        }
        write_json(outdir / "emergence_summary.json", summary)

        (outdir / "report.txt").write_text(
            "\n".join([
                f"version: {VERSION}",
                f"N: {len(rows)}",
                f"scan layers: {scan_layers}",
                f"primary U: L{args.primary_u_layer} D{args.u_dim}",
                f"refine layers: {refine_layers}",
                "",
                "carry_role_minus_identity = relation-specific state already present on block entry",
                "block_minus_carry_specific = local full-block gain heuristic",
                "carry_growth_to_next = growth of carried state into next layer",
                "formation_score = positive local gain + positive next-layer carry growth",
                "",
                "formation_score is a ranking heuristic, not an additive causal decomposition.",
            ]) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "emergence_curve.csv",
            "emergence_curve_phaseA.csv",
            "emergence_all_u.csv",
            "refine_component_comparison.csv",
            "final_comparison.csv",
            "emergence_summary.json",
            "component_additivity_audit.csv",
            "report.txt",
        ):
            print(" ", outdir / name)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
