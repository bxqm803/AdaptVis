#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test whether repeated directional correction after L26 can repair spatial errors.

This script reuses eval_directional_head_repair_v1.py for model/data loading and
relation scoring. It intervenes on either:

  attention update: attention_output[last_token]
  full block update: block_output[last_token] - block_input[last_token]

For guide direction d = normalize(W_guide - W_opposite), if update·d < 0:

  remove: delete the negative component
  flip:   reflect the negative component to the positive side

Default experiments test L26-L35, L26-L31, L28-L31, L30-L31, and L26+L30.
Use --scan-single-layers block to additionally test every L26...L35 block alone.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import eval_directional_head_repair_v1 as core


SCRIPT_VERSION = "eval-multilayer-directional-repair-v2"
DEFAULT_VARIANTS = ",".join([
    "attn_remove_all",
    "attn_flip_all",
    "block_remove_all",
    "block_flip_all",
    "block_remove_26_31",
    "block_flip_26_31",
    "block_remove_28_31",
    "block_remove_30_31",
    "block_remove_26_30",
    "block_flip_26_30",
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-module", default=None)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--guide", choices=["centroid", "oracle"], default="centroid")
    p.add_argument(
        "--trigger",
        choices=["conflict", "all", "wrong-only"],
        default="conflict",
    )
    p.add_argument("--centroid-confidence-threshold", type=float, default=0.0)
    p.add_argument("--start-layer", type=int, default=26)
    p.add_argument("--end-layer", type=int, default=35)
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument(
        "--scan-single-layers",
        choices=["none", "block", "attention", "both"],
        default="none",
    )
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--relations", default="left,right,above,below")
    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def safe_mean(values: Iterable[float]) -> float:
    values = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(values)) if values else float("nan")


def make_variant(name: str, start: int, end: int) -> Dict[str, Any]:
    all_layers = list(range(start, end + 1))
    fixed = {
        "attn_remove_all": ("attention", "remove", all_layers),
        "attn_flip_all": ("attention", "flip", all_layers),
        "block_remove_all": ("block", "remove", all_layers),
        "block_flip_all": ("block", "flip", all_layers),
        "block_remove_26_31": (
            "block", "remove", [x for x in range(26, 32) if start <= x <= end]
        ),
        "block_flip_26_31": (
            "block", "flip", [x for x in range(26, 32) if start <= x <= end]
        ),
        "block_remove_28_31": (
            "block", "remove", [x for x in range(28, 32) if start <= x <= end]
        ),
        "block_remove_30_31": (
            "block", "remove", [x for x in range(30, 32) if start <= x <= end]
        ),
        "block_remove_26_30": (
            "block", "remove", [x for x in (26, 30) if start <= x <= end]
        ),
        "block_flip_26_30": (
            "block", "flip", [x for x in (26, 30) if start <= x <= end]
        ),
    }
    if name in fixed:
        target, mode, layers = fixed[name]
        if not layers:
            raise ValueError(f"{name} has no layers inside [{start},{end}]")
        return {"name": name, "target": target, "mode": mode, "layers": layers}

    prefixes = {
        "block_remove_L": ("block", "remove"),
        "block_flip_L": ("block", "flip"),
        "attn_remove_L": ("attention", "remove"),
        "attn_flip_L": ("attention", "flip"),
    }
    for prefix, (target, mode) in prefixes.items():
        if name.startswith(prefix):
            layer = int(name[len(prefix):])
            if not start <= layer <= end:
                raise ValueError(f"{name}: layer outside [{start},{end}]")
            return {
                "name": name,
                "target": target,
                "mode": mode,
                "layers": [layer],
            }
    raise ValueError(f"Unknown variant: {name}")


def build_variants(args: argparse.Namespace) -> List[Dict[str, Any]]:
    names = core.csv_list(args.variants)
    if args.scan_single_layers in ("block", "both"):
        names += [
            f"block_remove_L{layer}"
            for layer in range(args.start_layer, args.end_layer + 1)
        ]
    if args.scan_single_layers in ("attention", "both"):
        names += [
            f"attn_remove_L{layer}"
            for layer in range(args.start_layer, args.end_layer + 1)
        ]
    names = list(dict.fromkeys(names))
    return [make_variant(name, args.start_layer, args.end_layer) for name in names]


def correction_delta(
    update: torch.Tensor,
    direction_cpu: torch.Tensor,
    mode: str,
    strength: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    update_f = update.float()
    direction = direction_cpu.to(device=update.device, dtype=torch.float32)
    alpha = float(torch.dot(update_f, direction).item())

    if alpha >= 0.0:
        delta_f = torch.zeros_like(update_f)
    elif mode == "remove":
        delta_f = -strength * alpha * direction
    elif mode == "flip":
        delta_f = -2.0 * strength * alpha * direction
    else:
        raise ValueError(mode)

    alpha_delta = float(torch.dot(delta_f, direction).item())
    return delta_f.to(dtype=update.dtype), {
        "alpha_before": alpha,
        "alpha_delta": alpha_delta,
        "alpha_after_estimate": alpha + alpha_delta,
        "was_negative": float(alpha < 0.0),
        "update_norm": float(update_f.norm().item()),
        "delta_norm": float(delta_f.norm().item()),
    }


def run_multilayer_intervention(
    base: Any,
    model: Any,
    batch: Dict[str, Any],
    layers: Sequence[torch.nn.Module],
    attention_modules: Sequence[torch.nn.Module],
    variant: Dict[str, Any],
    direction: torch.Tensor,
    strength: float,
    last_index: int,
) -> Tuple[Any, List[Dict[str, Any]]]:
    diagnostics: List[Dict[str, Any]] = []
    handles = []
    modules = attention_modules if variant["target"] == "attention" else layers

    for layer_index in variant["layers"]:
        module = modules[layer_index]

        def make_hook(index: int):
            def hook(_module: Any, inputs: Any, output: Any) -> Any:
                out_tensor = base.first_tensor(output)
                if out_tensor.ndim != 3 or int(out_tensor.shape[0]) != 1:
                    raise RuntimeError(
                        f"Unexpected {variant['target']} output at L{index}: "
                        f"{tuple(out_tensor.shape)}"
                    )

                if variant["target"] == "attention":
                    update = out_tensor[0, last_index, :]
                else:
                    in_tensor = base.first_tensor(inputs)
                    if in_tensor.ndim != 3:
                        raise RuntimeError(
                            f"Unexpected block input at L{index}: "
                            f"{tuple(in_tensor.shape)}"
                        )
                    update = (
                        out_tensor[0, last_index, :]
                        - in_tensor[0, last_index, :]
                    )

                delta, info = correction_delta(
                    update,
                    direction,
                    variant["mode"],
                    strength,
                )
                modified = out_tensor.clone()
                modified[0, last_index, :] += delta.to(
                    device=modified.device,
                    dtype=modified.dtype,
                )
                diagnostics.append({
                    "layer": index,
                    "target": variant["target"],
                    "mode": variant["mode"],
                    **info,
                })
                return core.replace_first_tensor(output, modified)
            return hook

        handles.append(module.register_forward_hook(make_hook(layer_index)))

    try:
        with torch.inference_mode():
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    diagnostics.sort(key=lambda row: int(row["layer"]))
    return outputs, diagnostics


def variant_summary(rows: Sequence[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    usable = [
        row for row in rows
        if row.get("status") == "ok" and variant in row.get("variants", {})
    ]
    if not usable:
        return {"variant": variant, "n": 0}

    base_correct = np.asarray([
        bool(row["baseline"]["correct"]) for row in usable
    ], dtype=bool)
    new_correct = np.asarray([
        bool(row["variants"][variant]["correct"]) for row in usable
    ], dtype=bool)
    triggered = np.asarray([
        bool(row["variants"][variant]["triggered"]) for row in usable
    ], dtype=bool)
    repaired = (~base_correct) & new_correct
    damaged = base_correct & (~new_correct)
    wrong = ~base_correct
    correct = base_correct
    margin_delta = np.asarray([
        float(row["variants"][variant]["gt_margin"])
        - float(row["baseline"]["gt_margin"])
        for row in usable
    ], dtype=np.float64)

    return {
        "variant": variant,
        "n": len(usable),
        "n_triggered": int(triggered.sum()),
        "baseline_accuracy": float(base_correct.mean()),
        "intervention_accuracy": float(new_correct.mean()),
        "accuracy_change": float(new_correct.mean() - base_correct.mean()),
        "repaired_wrong": int(repaired.sum()),
        "damaged_correct": int(damaged.sum()),
        "net_repair": int(repaired.sum() - damaged.sum()),
        "wrong_n": int(wrong.sum()),
        "wrong_repair_rate": (
            float(repaired[wrong].mean()) if wrong.any() else float("nan")
        ),
        "correct_n": int(correct.sum()),
        "correct_damage_rate": (
            float(damaged[correct].mean()) if correct.any() else float("nan")
        ),
        "prediction_changed": int(sum(
            row["baseline"]["prediction"]
            != row["variants"][variant]["prediction"]
            for row in usable
        )),
        "mean_gt_margin_delta": float(margin_delta.mean()),
        "mean_gt_margin_delta_wrong": (
            float(margin_delta[wrong].mean()) if wrong.any() else float("nan")
        ),
        "mean_gt_margin_delta_correct": (
            float(margin_delta[correct].mean()) if correct.any() else float("nan")
        ),
    }


def summarize_layers(
    rows: Sequence[Dict[str, Any]],
    variant_names: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        for variant in variant_names:
            result = row.get("variants", {}).get(variant)
            if not result or not result.get("triggered"):
                continue
            for diag in result.get("diagnostics", []):
                grouped[(variant, int(diag["layer"]))].append(diag)

    output = []
    for (variant, layer), values in sorted(grouped.items()):
        output.append({
            "variant": variant,
            "layer": layer,
            "target": values[0]["target"],
            "mode": values[0]["mode"],
            "n": len(values),
            "negative_update_fraction": safe_mean(
                value["was_negative"] for value in values
            ),
            "mean_alpha_before": safe_mean(
                value["alpha_before"] for value in values
            ),
            "mean_alpha_delta": safe_mean(
                value["alpha_delta"] for value in values
            ),
            "mean_delta_norm": safe_mean(
                value["delta_norm"] for value in values
            ),
        })
    return output


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.start_layer > args.end_layer:
        raise ValueError("--start-layer must be <= --end-layer")
    if args.strength < 0.0:
        raise ValueError("--strength must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = core.load_base(args.base_module)
    relations = [base.normalize_relation(x) for x in core.csv_list(args.relations)]
    if set(relations) != set(core.OPPOSITE):
        raise ValueError("Relations must be left,right,above,below")
    variants = build_variants(args)
    variant_names = [variant["name"] for variant in variants]

    trace_dir = Path(args.trace_dir)
    trace_summary = json.loads(
        (trace_dir / "summary.json").read_text(encoding="utf-8")
    )
    dataset = args.dataset or trace_summary["dataset"]
    model_name = args.model or trace_summary["model"]

    metadata = [
        row for row in base.read_jsonl(trace_dir / "sample_metadata.jsonl")
        if row.get("group") in (base.GROUP_CORRECT, base.GROUP_WRONG)
        and base.normalize_relation(row.get("gt")) in relations
    ]
    for row in metadata:
        row["gt"] = base.normalize_relation(row.get("gt"))
        row["centroid_prediction"] = base.normalize_relation(
            row.get("centroid_prediction")
        )
        row["baseline_prediction"] = base.normalize_relation(
            row.get("baseline_prediction")
        )

    if args.max_per_group is not None:
        rng = random.Random(args.seed)
        selected = []
        for group in (base.GROUP_CORRECT, base.GROUP_WRONG):
            rows = [row for row in metadata if row["group"] == group]
            rng.shuffle(rows)
            selected.extend(rows[:args.max_per_group])
        metadata = sorted(selected, key=lambda row: int(row["sid"]))

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    errors_path = output_dir / "errors.jsonl"
    for path in (samples_path, errors_path):
        if path.exists():
            path.unlink()

    support = base.import_two_object_module()
    records, audit = support.load_records(dataset, Path(args.data_root), None)
    record_by_sid = {int(record.sid): record for record in records}
    prompt_args = argparse.Namespace(
        dataset=dataset,
        prompt_jsonl=args.prompt_jsonl,
    )
    prompt_rows = base.load_standard_prompts(
        base.resolve_prompt_path(prompt_args)
    )

    spec = support.SPECS[model_name]
    model_cls = getattr(base.transformers, spec.model_class)
    print(f"Loading {model_name}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()
    processor = base.AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = base.resolve_decoder_layers(model)
    if args.end_layer >= len(layers):
        raise RuntimeError(
            f"--end-layer={args.end_layer}, model has {len(layers)} layers"
        )

    collector = base.LayerTraceCollector(layers, [])
    attention_modules = list(collector.attention_modules)
    collector.close()

    label_ids = base.label_token_id_variants(processor.tokenizer)
    relation_vectors = core.readout_vectors(model, label_ids, relations)

    print("\nVariants:")
    for variant in variants:
        print(
            f"  {variant['name']:24s} target={variant['target']:9s} "
            f"mode={variant['mode']:6s} layers={variant['layers']}"
        )

    results = []
    running_base = 0
    running = {name: 0 for name in variant_names}
    started = time.time()

    for index, metadata_row in enumerate(
        tqdm(metadata, desc=f"multilayer-repair:{model_name}"),
        1,
    ):
        sid = int(metadata_row["sid"])
        batch = image = None
        try:
            prompt = prompt_rows[sid]
            gt = base.normalize_relation(prompt["answer_raw"])
            guide = (
                metadata_row["centroid_prediction"]
                if args.guide == "centroid"
                else gt
            )
            question = str(prompt["question_text"])
            subject = str(prompt["subject"])
            reference = str(prompt["reference"])

            image = base.record_image(record_by_sid[sid])
            batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=question,
                device=device,
            )
            last_index = int(batch["input_ids"].shape[1]) - 1

            with torch.inference_mode():
                baseline_outputs = model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            baseline_scores = core.relation_scores_from_logits(
                baseline_outputs.logits[0, -1],
                label_ids,
                relations,
            )
            baseline = core.diagnostics(
                baseline_scores,
                gt,
                guide,
                relations,
            )
            del baseline_outputs

            confidence = safe_float(
                metadata_row.get("centroid_confidence")
            )
            trigger = core.trigger_intervention(
                args.trigger,
                metadata_row["group"],
                baseline["prediction"],
                guide,
                confidence,
                args.centroid_confidence_threshold,
                base.GROUP_WRONG,
            )
            direction = core.guide_direction(relation_vectors, guide)

            row = {
                "status": "ok",
                "sid": sid,
                "group": metadata_row["group"],
                "subject": subject,
                "reference": reference,
                "question": question,
                "gt": gt,
                "guide": guide,
                "centroid_prediction": metadata_row["centroid_prediction"],
                "centroid_confidence": confidence,
                "baseline": baseline,
                "variants": {},
            }
            running_base += int(baseline["correct"])

            for variant in variants:
                name = variant["name"]
                if not trigger:
                    result = {
                        **baseline,
                        "triggered": False,
                        "diagnostics": [],
                    }
                else:
                    outputs, diagnostics = run_multilayer_intervention(
                        base,
                        model,
                        batch,
                        layers,
                        attention_modules,
                        variant,
                        direction,
                        args.strength,
                        last_index,
                    )
                    scores = core.relation_scores_from_logits(
                        outputs.logits[0, -1],
                        label_ids,
                        relations,
                    )
                    result = {
                        **core.diagnostics(scores, gt, guide, relations),
                        "triggered": True,
                        "diagnostics": diagnostics,
                    }
                    del outputs

                row["variants"][name] = result
                running[name] += int(result["correct"])

            results.append(row)
            core.append_jsonl(samples_path, row)

            if args.print_every > 0 and (
                index == 1
                or index % args.print_every == 0
                or index == len(metadata)
            ):
                tqdm.write(
                    f"\n[{index}/{len(metadata)}] sid={sid}\n"
                    f"Q: {question}\n"
                    f"GT={gt} guide={guide} baseline={baseline['prediction']} "
                    f"base_acc={running_base/index:.4f}"
                )
                for variant in variants:
                    name = variant["name"]
                    result = row["variants"][name]
                    mark = (
                        "REPAIRED"
                        if not baseline["correct"] and result["correct"]
                        else "DAMAGED"
                        if baseline["correct"] and not result["correct"]
                        else "-"
                    )
                    tqdm.write(
                        f"  {name:24s} pred={result['prediction']:5s} "
                        f"margin={result['gt_margin']:+.4f} "
                        f"acc={running[name]/index:.4f} {mark}"
                    )

        except Exception as exc:
            error = {
                "status": "error",
                "sid": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
            }
            results.append(error)
            core.append_jsonl(errors_path, error)
            tqdm.write(
                f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
            )
        finally:
            if batch is not None:
                del batch
            if image is not None:
                del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summaries = [variant_summary(results, name) for name in variant_names]
    summaries.sort(
        key=lambda row: (
            int(row.get("net_repair", -10**9)),
            float(row.get("intervention_accuracy", -math.inf)),
            float(row.get("mean_gt_margin_delta_wrong", -math.inf)),
        ),
        reverse=True,
    )
    base.write_csv(output_dir / "summary.csv", summaries)

    relation_rows = []
    for relation in relations:
        subset = [
            row for row in results
            if row.get("status") == "ok" and row.get("gt") == relation
        ]
        for name in variant_names:
            item = variant_summary(subset, name)
            item["relation"] = relation
            relation_rows.append(item)
    base.write_csv(output_dir / "summary_by_relation.csv", relation_rows)
    base.write_csv(
        output_dir / "layer_diagnostics.csv",
        summarize_layers(results, variant_names),
    )

    usable = [row for row in results if row.get("status") == "ok"]
    baseline_accuracy = safe_mean(
        float(row["baseline"]["correct"]) for row in usable
    )
    centroid_gt = safe_mean(
        float(row["centroid_prediction"] == row["gt"])
        for row in usable
    )

    final_summary = {
        "script_version": SCRIPT_VERSION,
        "trace_dir": str(trace_dir),
        "dataset": dataset,
        "model": model_name,
        "guide": args.guide,
        "trigger": args.trigger,
        "strength": args.strength,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "variants": variants,
        "n_requested": len(metadata),
        "n_successful": len(usable),
        "baseline_accuracy": baseline_accuracy,
        "centroid_gt_agreement": centroid_gt,
        "best_variant": summaries[0] if summaries else None,
        "variant_summaries": summaries,
        "decoder_layers_path": layers_path,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "audit": audit,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 128)
    print("MULTI-LAYER DIRECTIONAL REPAIR VALIDATION")
    print("=" * 128)
    print(f"baseline accuracy: {baseline_accuracy:.4f}")
    print(f"centroid/GT agreement: {centroid_gt:.4f}")
    print("")
    print(
        "variant                  | base_acc | new_acc | delta    | "
        "repaired | damaged | net | wrong_margin_delta"
    )
    print("-" * 128)
    for item in summaries:
        print(
            f"{item['variant']:24s} | "
            f"{item['baseline_accuracy']:.4f}   | "
            f"{item['intervention_accuracy']:.4f}  | "
            f"{item['accuracy_change']:+.4f} | "
            f"{item['repaired_wrong']:8d} | "
            f"{item['damaged_correct']:7d} | "
            f"{item['net_repair']:3d} | "
            f"{item['mean_gt_margin_delta_wrong']:+.5f}"
        )

    print("\nBest variant:")
    print(json.dumps(
        summaries[0] if summaries else {},
        ensure_ascii=False,
        indent=2,
    ))
    print("\nSaved:")
    for name in (
        "samples.jsonl",
        "summary.csv",
        "summary_by_relation.csv",
        "layer_diagnostics.csv",
        "summary.json",
    ):
        print(f"  {output_dir / name}")
    if errors_path.exists():
        print(f"  {errors_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
