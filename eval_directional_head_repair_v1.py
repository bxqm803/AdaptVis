#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate whether L26 answer-routing interventions can repair centroid-correct
spatial relation errors.

The script imports trace_centroid_generation_groups_v2_2.py (or v2_1),
reconstructs object-derived contributions for L26H0/H2/H5, then performs a
second forward pass with controlled residual edits at the final prompt token.

Default variants:
  h5_knockout : remove all object-derived L26H5 contribution
  h5_remove   : remove only L26H5's component opposing the centroid direction
  h5_flip     : reflect L26H5's opposing component toward the centroid direction
  h025_remove : remove opposing components from L26H0/H2/H5
  h025_flip   : reflect opposing components from L26H0/H2/H5

By default the guide is the centroid prediction and intervention is triggered
only when the current baseline relation prediction conflicts with the centroid.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_VERSION = "eval-directional-head-repair-v1"
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
ALLOWED_VARIANTS = {
    "h5_knockout",
    "h5_remove",
    "h5_flip",
    "h025_remove",
    "h025_flip",
}


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
    p.add_argument(
        "--variants",
        default="h5_knockout,h5_remove,h5_flip,h025_remove,h025_flip",
    )
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--h5-head", default="26:5")
    p.add_argument("--h025-heads", default="26:0,26:2,26:5")
    p.add_argument("--relations", default="left,right,above,below")
    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reuse-baseline-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_base(name: Optional[str]) -> Any:
    candidates = []
    if name:
        candidates.append(name)
    candidates += [
        "trace_centroid_generation_groups_v2_2",
        "trace_centroid_generation_groups_v2_1",
    ]
    required = [
        "RELATIONS", "GROUP_CORRECT", "GROUP_WRONG", "read_jsonl",
        "normalize_relation", "resolve_decoder_layers",
        "label_token_id_variants", "resolve_visual_indices",
        "locate_object_spans", "span_indices", "attention_tuple",
        "reshape_projected_values", "project_one_head_isolated",
        "LayerTraceCollector", "load_standard_prompts",
        "resolve_prompt_path", "make_question_batch", "record_image",
        "configure_processor", "resolve_dtype", "import_two_object_module",
        "write_csv", "first_tensor",
    ]
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            module = importlib.import_module(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        missing = [x for x in required if not hasattr(module, x)]
        if not missing:
            return module
        errors.append(f"{candidate}: missing {missing}")
    raise RuntimeError("Could not import base module:\n  " + "\n  ".join(errors))


def csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_head(value: str) -> Tuple[int, int]:
    pieces = value.strip().split(":")
    if len(pieces) != 2:
        raise ValueError(f"Expected layer:head, got {value!r}")
    return int(pieces[0]), int(pieces[1])


def parse_heads(value: str) -> List[Tuple[int, int]]:
    return list(dict.fromkeys(parse_head(x) for x in csv_list(value)))


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def safe_mean(values: Iterable[float]) -> float:
    values = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(values)) if values else float("nan")


def relation_scores_from_logits(
    logits: torch.Tensor,
    label_ids: Dict[str, List[int]],
    relations: Sequence[str],
) -> np.ndarray:
    if logits.ndim == 2:
        logits = logits[-1]
    scores = []
    for relation in relations:
        ids = torch.as_tensor(label_ids[relation], device=logits.device)
        scores.append(logits.index_select(0, ids).max())
    return torch.stack(scores).detach().float().cpu().numpy().astype(np.float32)


def diagnostics(
    scores: np.ndarray,
    gt: str,
    guide: str,
    relations: Sequence[str],
) -> Dict[str, Any]:
    prediction = relations[int(np.argmax(scores))]
    gt_i = relations.index(gt)
    other = scores.copy()
    other[gt_i] = -np.inf
    gt_margin = float(scores[gt_i] - np.max(other))
    guide_i = relations.index(guide)
    opp_i = relations.index(OPPOSITE[guide])
    return {
        "scores": {r: float(scores[i]) for i, r in enumerate(relations)},
        "prediction": prediction,
        "correct": prediction == gt,
        "gt_margin": gt_margin,
        "guide_opposite_margin": float(scores[guide_i] - scores[opp_i]),
    }


def readout_vectors(
    model: Any,
    label_ids: Dict[str, List[int]],
    relations: Sequence[str],
) -> Dict[str, torch.Tensor]:
    weight = model.get_output_embeddings().weight.detach()
    result = {}
    for relation in relations:
        ids = torch.as_tensor(label_ids[relation], device=weight.device)
        result[relation] = weight.index_select(0, ids).float().mean(0).cpu()
    return result


def guide_direction(vectors: Dict[str, torch.Tensor], guide: str) -> torch.Tensor:
    d = vectors[guide] - vectors[OPPOSITE[guide]]
    norm = float(d.norm().item())
    if norm <= 1e-12:
        raise RuntimeError(f"Degenerate readout direction for {guide}")
    return d / norm


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item):
                items[index] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item):
                items[index] = replacement
                return items
    raise TypeError(f"Unsupported attention output type: {type(output)}")


def reconstruct_heads(
    base: Any,
    attentions: Sequence[torch.Tensor],
    collector: Any,
    subject_indices: Sequence[int],
    reference_indices: Sequence[int],
    last_index: int,
    requested: Sequence[Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in requested:
        grouped[layer].append(head)

    subject = torch.as_tensor(subject_indices, dtype=torch.long)
    reference = torch.as_tensor(reference_indices, dtype=torch.long)
    output: Dict[str, torch.Tensor] = {}

    for layer, heads in grouped.items():
        attention = attentions[layer][0].detach().float().cpu()
        n_heads = int(attention.shape[0])
        flat = collector.object_values[layer]
        if flat is None:
            raise RuntimeError(f"Missing object values at layer {layer}")
        values = base.reshape_projected_values(
            flat,
            n_attention_heads=n_heads,
            attention_module=collector.attention_modules[layer],
        ).float()
        n_subject = len(subject_indices)
        subject_values = values[:n_subject].permute(1, 0, 2)
        reference_values = values[n_subject:].permute(1, 0, 2)
        last_attention = attention[:, last_index, :]
        subject_weights = last_attention.index_select(-1, subject)
        reference_weights = last_attention.index_select(-1, reference)
        combined = (
            torch.einsum("ht,htd->hd", subject_weights, subject_values)
            + torch.einsum("ht,htd->hd", reference_weights, reference_values)
        )
        for head in heads:
            projected = base.project_one_head_isolated(
                combined[head].unsqueeze(0),
                o_proj=collector.o_projections[layer],
                head_index=head,
                total_heads=n_heads,
            )
            if projected is None:
                raise RuntimeError(f"Could not project L{layer}H{head}")
            output[f"{layer}:{head}"] = projected[0].detach().float().cpu()
    return output


def heads_for_variant(
    variant: str,
    h5: Tuple[int, int],
    h025: Sequence[Tuple[int, int]],
) -> Sequence[Tuple[int, int]]:
    if variant.startswith("h5_"):
        return [h5]
    if variant.startswith("h025_"):
        return h025
    raise ValueError(variant)


def build_deltas(
    variant: str,
    cached_vectors: Dict[str, torch.Tensor],
    direction: torch.Tensor,
    h5: Tuple[int, int],
    h025: Sequence[Tuple[int, int]],
    strength: float,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    by_layer: Dict[int, torch.Tensor] = {}
    details = []
    for layer, head in heads_for_variant(variant, h5, h025):
        vector = cached_vectors[f"{layer}:{head}"].float()
        d = direction.to(vector)
        alpha = float(torch.dot(vector, d).item())
        if variant.endswith("_knockout"):
            delta = -strength * vector
        elif variant.endswith("_remove"):
            delta = -strength * alpha * d if alpha < 0 else torch.zeros_like(vector)
        elif variant.endswith("_flip"):
            delta = -2.0 * strength * alpha * d if alpha < 0 else torch.zeros_like(vector)
        else:
            raise ValueError(variant)
        if layer not in by_layer:
            by_layer[layer] = torch.zeros_like(delta)
        by_layer[layer] += delta
        details.append({
            "layer": layer,
            "head": head,
            "alpha_before": alpha,
            "delta_projection": float(torch.dot(delta, d).item()),
            "vector_norm": float(vector.norm().item()),
            "delta_norm": float(delta.norm().item()),
        })
    return by_layer, {
        "heads": details,
        "total_delta_norm": float(math.sqrt(sum(float(x.norm().item()) ** 2 for x in by_layer.values()))),
        "nonzero": any(float(x.norm().item()) > 0 for x in by_layer.values()),
    }


def run_intervention(
    base: Any,
    model: Any,
    batch: Dict[str, Any],
    attention_modules: Sequence[torch.nn.Module],
    deltas: Dict[int, torch.Tensor],
    last_index: int,
) -> Any:
    handles = []
    for layer, delta_cpu in deltas.items():
        def make_hook(delta_value: torch.Tensor, layer_index: int):
            def hook(_module: Any, _args: Any, output: Any) -> Any:
                tensor = base.first_tensor(output)
                if tensor.ndim != 3 or tensor.shape[0] != 1:
                    raise RuntimeError(f"Unexpected L{layer_index} attention output {tuple(tensor.shape)}")
                modified = tensor.clone()
                delta = delta_value.to(device=modified.device, dtype=modified.dtype)
                modified[0, last_index] += delta
                return replace_first_tensor(output, modified)
            return hook
        handles.append(attention_modules[layer].register_forward_hook(make_hook(delta_cpu, layer)))
    try:
        with torch.inference_mode():
            return model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()


def trigger_intervention(
    trigger: str,
    group: str,
    baseline_prediction: str,
    guide: str,
    confidence: float,
    threshold: float,
    wrong_group: str,
) -> bool:
    if not math.isfinite(confidence) or confidence < threshold:
        return False
    if trigger == "all":
        return True
    if trigger == "conflict":
        return baseline_prediction != guide
    if trigger == "wrong-only":
        return group == wrong_group
    raise ValueError(trigger)


def variant_summary(rows: Sequence[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    rows = [r for r in rows if r.get("status") == "ok"]
    base_correct = np.asarray([r["baseline"]["correct"] for r in rows], dtype=bool)
    new_correct = np.asarray([r["variants"][variant]["correct"] for r in rows], dtype=bool)
    triggered = np.asarray([r["variants"][variant]["triggered"] for r in rows], dtype=bool)
    repaired = (~base_correct) & new_correct
    damaged = base_correct & (~new_correct)
    changed = np.asarray([
        r["baseline"]["prediction"] != r["variants"][variant]["prediction"]
        for r in rows
    ], dtype=bool)
    margin_delta = np.asarray([
        r["variants"][variant]["gt_margin"] - r["baseline"]["gt_margin"]
        for r in rows
    ], dtype=np.float64)
    wrong = ~base_correct
    correct = base_correct
    return {
        "variant": variant,
        "n": len(rows),
        "n_triggered": int(triggered.sum()),
        "baseline_accuracy": float(base_correct.mean()),
        "intervention_accuracy": float(new_correct.mean()),
        "accuracy_change": float(new_correct.mean() - base_correct.mean()),
        "repaired_wrong": int(repaired.sum()),
        "damaged_correct": int(damaged.sum()),
        "net_repair": int(repaired.sum() - damaged.sum()),
        "prediction_changed": int(changed.sum()),
        "wrong_n": int(wrong.sum()),
        "wrong_repair_rate": float(repaired[wrong].mean()) if wrong.any() else float("nan"),
        "correct_n": int(correct.sum()),
        "correct_damage_rate": float(damaged[correct].mean()) if correct.any() else float("nan"),
        "mean_gt_margin_delta": float(margin_delta.mean()),
        "mean_gt_margin_delta_wrong": float(margin_delta[wrong].mean()) if wrong.any() else float("nan"),
        "mean_gt_margin_delta_correct": float(margin_delta[correct].mean()) if correct.any() else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.strength < 0:
        raise ValueError("--strength must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = load_base(args.base_module)
    relations = [base.normalize_relation(x) for x in csv_list(args.relations)]
    if set(relations) != set(OPPOSITE):
        raise ValueError("Relations must be left,right,above,below")
    variants = csv_list(args.variants)
    unknown = [x for x in variants if x not in ALLOWED_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    h5 = parse_head(args.h5_head)
    h025 = parse_heads(args.h025_heads)
    requested_heads = list(dict.fromkeys([h5, *h025]))

    trace_dir = Path(args.trace_dir)
    trace_summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    dataset = args.dataset or trace_summary["dataset"]
    model_name = args.model or trace_summary["model"]
    metadata = [
        row for row in base.read_jsonl(trace_dir / "sample_metadata.jsonl")
        if row.get("group") in (base.GROUP_CORRECT, base.GROUP_WRONG)
        and base.normalize_relation(row.get("gt")) in relations
    ]
    for row in metadata:
        row["gt"] = base.normalize_relation(row.get("gt"))
        row["centroid_prediction"] = base.normalize_relation(row.get("centroid_prediction"))
        row["baseline_prediction"] = base.normalize_relation(row.get("baseline_prediction"))

    if args.max_per_group is not None:
        rng = random.Random(args.seed)
        selected = []
        for group in (base.GROUP_CORRECT, base.GROUP_WRONG):
            rows = [r for r in metadata if r["group"] == group]
            rng.shuffle(rows)
            selected.extend(rows[:args.max_per_group])
        metadata = sorted(selected, key=lambda x: int(x["sid"]))

    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / "samples.jsonl"
    errors_path = out / "errors.jsonl"
    for path in (samples_path, errors_path):
        if path.exists():
            path.unlink()

    support = base.import_two_object_module()
    records, audit = support.load_records(dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    prompt_args = argparse.Namespace(dataset=dataset, prompt_jsonl=args.prompt_jsonl)
    prompt_rows = base.load_standard_prompts(base.resolve_prompt_path(prompt_args))

    spec = support.SPECS[model_name]
    model_cls = getattr(base.transformers, spec.model_class)
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()
    processor = base.AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = base.resolve_decoder_layers(model)
    label_ids = base.label_token_id_variants(processor.tokenizer)
    relation_vectors = readout_vectors(model, label_ids, relations)
    collector = base.LayerTraceCollector(layers, [])
    attention_modules = list(collector.attention_modules)

    cache_path = out / "baseline_cache.pt"
    cache_signature = {
        "trace_dir": str(trace_dir.resolve()),
        "sids": [int(r["sid"]) for r in metadata],
        "heads": requested_heads,
        "model": model_name,
    }
    baseline_cache = []
    if args.reuse_baseline_cache and cache_path.exists():
        loaded = torch.load(cache_path, map_location="cpu")
        if loaded.get("signature") != cache_signature:
            raise RuntimeError("Baseline cache does not match this run")
        baseline_cache = loaded["rows"]
        print(f"Reused baseline cache: {cache_path}")
    else:
        print("\nStage 1/2: baseline trace")
        for row in tqdm(metadata, desc="baseline-trace"):
            sid = int(row["sid"])
            batch = image = None
            try:
                prompt = prompt_rows[sid]
                subject = str(prompt["subject"])
                reference = str(prompt["reference"])
                question = str(prompt["question_text"])
                gt = base.normalize_relation(prompt["answer_raw"])
                guide = row["centroid_prediction"] if args.guide == "centroid" else gt
                image = base.record_image(record_by_sid[sid])
                batch = base.make_question_batch(processor=processor, image=image, question_text=question, device=device)
                input_ids = batch["input_ids"][0].detach().cpu().tolist()
                subject_span, reference_span = base.locate_object_spans(processor.tokenizer, input_ids, subject, reference)
                subject_indices = base.span_indices(subject_span)
                reference_indices = base.span_indices(reference_span)
                visual_indices = base.resolve_visual_indices(model, processor, batch, input_ids)
                last_index = len(input_ids) - 1
                collector.set_sample(
                    subject_indices=subject_indices,
                    reference_indices=reference_indices,
                    visual_indices=visual_indices,
                    last_index=last_index,
                )
                try:
                    with torch.inference_mode():
                        outputs = model(**batch, output_attentions=True, output_hidden_states=False, use_cache=False, return_dict=True)
                finally:
                    collector.active = False
                attentions = base.attention_tuple(outputs)
                scores = relation_scores_from_logits(outputs.logits[0, -1], label_ids, relations)
                baseline = diagnostics(scores, gt, guide, relations)
                head_vectors = reconstruct_heads(
                    base, attentions, collector, subject_indices,
                    reference_indices, last_index, requested_heads,
                )
                direction = guide_direction(relation_vectors, guide)
                baseline_cache.append({
                    "sid": sid,
                    "group": row["group"],
                    "gt": gt,
                    "guide": guide,
                    "centroid_prediction": row["centroid_prediction"],
                    "centroid_confidence": float(row.get("centroid_confidence", float("nan"))),
                    "prior_baseline_prediction": row.get("baseline_prediction"),
                    "prior_baseline_correct": bool(row.get("baseline_correct")),
                    "subject": subject,
                    "reference": reference,
                    "question": question,
                    "last_index": last_index,
                    "baseline": baseline,
                    "head_vectors": head_vectors,
                    "head_alpha": {
                        key: float(torch.dot(vector, direction).item())
                        for key, vector in head_vectors.items()
                    },
                })
                del outputs, attentions
            except Exception as exc:
                append_jsonl(errors_path, {
                    "stage": "baseline", "sid": sid,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
            finally:
                collector.active = False
                if batch is not None: del batch
                if image is not None: del image
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        torch.save({"signature": cache_signature, "rows": baseline_cache}, cache_path)

    if not baseline_cache:
        raise RuntimeError("No baseline samples succeeded")
    collector.close()

    prior_match = safe_mean(
        float(r["baseline"]["prediction"] == r["prior_baseline_prediction"])
        for r in baseline_cache
    )
    centroid_gt = safe_mean(
        float(r["centroid_prediction"] == r["gt"])
        for r in baseline_cache
    )
    print(f"Baseline/prior prediction agreement: {prior_match:.4f}")
    print(f"Centroid/GT agreement:              {centroid_gt:.4f}")

    print("\nStage 2/2: intervention")
    results = []
    running = {v: 0 for v in variants}
    running_base = 0

    for index, cached in enumerate(tqdm(baseline_cache, desc="repair"), 1):
        sid = int(cached["sid"])
        batch = image = None
        try:
            image = base.record_image(record_by_sid[sid])
            batch = base.make_question_batch(processor=processor, image=image, question_text=cached["question"], device=device)
            last_index = int(batch["input_ids"].shape[1]) - 1
            if last_index != cached["last_index"]:
                raise RuntimeError("Prompt token length changed")
            direction = guide_direction(relation_vectors, cached["guide"])
            trigger = trigger_intervention(
                args.trigger, cached["group"], cached["baseline"]["prediction"],
                cached["guide"], cached["centroid_confidence"],
                args.centroid_confidence_threshold, base.GROUP_WRONG,
            )
            row = {
                "status": "ok",
                "sid": sid,
                "group": cached["group"],
                "question": cached["question"],
                "subject": cached["subject"],
                "reference": cached["reference"],
                "gt": cached["gt"],
                "guide": cached["guide"],
                "centroid_confidence": cached["centroid_confidence"],
                "prior_baseline_prediction": cached["prior_baseline_prediction"],
                "prior_baseline_correct": cached["prior_baseline_correct"],
                "baseline": cached["baseline"],
                "head_alpha": cached["head_alpha"],
                "variants": {},
            }
            running_base += int(cached["baseline"]["correct"])
            for variant in variants:
                if not trigger:
                    result = {**cached["baseline"], "triggered": False, "delta": {"nonzero": False}}
                else:
                    deltas, delta_info = build_deltas(
                        variant, cached["head_vectors"], direction,
                        h5, h025, args.strength,
                    )
                    if not delta_info["nonzero"]:
                        result = {**cached["baseline"], "triggered": True, "delta": delta_info}
                    else:
                        outputs = run_intervention(base, model, batch, attention_modules, deltas, last_index)
                        scores = relation_scores_from_logits(outputs.logits[0, -1], label_ids, relations)
                        result = {**diagnostics(scores, cached["gt"], cached["guide"], relations), "triggered": True, "delta": delta_info}
                        del outputs
                row["variants"][variant] = result
                running[variant] += int(result["correct"])
            results.append(row)
            append_jsonl(samples_path, row)

            if args.print_every > 0 and (index % args.print_every == 0 or index == 1):
                tqdm.write(
                    f"\n[{index}/{len(baseline_cache)}] sid={sid}\n"
                    f"Q: {cached['question']}\n"
                    f"GT={cached['gt']} centroid={cached['guide']} "
                    f"baseline={cached['baseline']['prediction']} "
                    f"base_acc={running_base/index:.4f}"
                )
                for variant in variants:
                    result = row["variants"][variant]
                    mark = "REPAIRED" if (not cached["baseline"]["correct"] and result["correct"]) else "DAMAGED" if (cached["baseline"]["correct"] and not result["correct"]) else "-"
                    tqdm.write(
                        f"  {variant:14s} pred={result['prediction']:5s} "
                        f"margin={result['gt_margin']:+.4f} "
                        f"acc={running[variant]/index:.4f} {mark}"
                    )
        except Exception as exc:
            error = {
                "status": "error", "sid": sid, "stage": "intervention",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
            }
            results.append(error)
            append_jsonl(errors_path, error)
        finally:
            if batch is not None: del batch
            if image is not None: del image
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    summaries = [variant_summary(results, v) for v in variants]
    base.write_csv(out / "summary.csv", summaries)

    relation_rows = []
    for relation in relations:
        subset = [r for r in results if r.get("status") == "ok" and r["gt"] == relation]
        for variant in variants:
            item = variant_summary(subset, variant)
            item["relation"] = relation
            relation_rows.append(item)
    base.write_csv(out / "summary_by_relation.csv", relation_rows)

    group_rows = []
    for group in (base.GROUP_CORRECT, base.GROUP_WRONG):
        subset = [r for r in results if r.get("status") == "ok" and r["group"] == group]
        for variant in variants:
            item = variant_summary(subset, variant)
            item["prior_group"] = group
            group_rows.append(item)
    base.write_csv(out / "summary_by_prior_group.csv", group_rows)

    best = max(summaries, key=lambda x: (x["net_repair"], x["intervention_accuracy"], x["mean_gt_margin_delta_wrong"]))
    final_summary = {
        "script_version": SCRIPT_VERSION,
        "base_module": base.__name__,
        "trace_dir": str(trace_dir),
        "dataset": dataset,
        "model": model_name,
        "guide": args.guide,
        "trigger": args.trigger,
        "strength": args.strength,
        "variants": variants,
        "n_successful": sum(r.get("status") == "ok" for r in results),
        "prior_prediction_agreement": prior_match,
        "centroid_gt_agreement": centroid_gt,
        "best_variant": best,
        "variant_summaries": summaries,
        "decoder_layers_path": layers_path,
        "audit": audit,
    }
    (out / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("DIRECTIONAL HEAD REPAIR VALIDATION")
    print("=" * 118)
    print("variant         | base_acc | new_acc | delta    | repaired | damaged | net | wrong_margin_delta")
    print("-" * 118)
    for item in summaries:
        print(
            f"{item['variant']:15s} | {item['baseline_accuracy']:.4f}   | "
            f"{item['intervention_accuracy']:.4f}  | {item['accuracy_change']:+.4f} | "
            f"{item['repaired_wrong']:8d} | {item['damaged_correct']:7d} | "
            f"{item['net_repair']:3d} | {item['mean_gt_margin_delta_wrong']:+.5f}"
        )
    print("\nBest variant:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\nSaved:")
    for name in ("baseline_cache.pt", "samples.jsonl", "summary.csv", "summary_by_relation.csv", "summary_by_prior_group.csv", "summary.json"):
        print(f"  {out / name}")
    if errors_path.exists():
        print(f"  {errors_path}")


if __name__ == "__main__":
    main()
