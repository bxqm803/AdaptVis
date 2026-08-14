#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correct-vs-wrong TRUE logit-lens trajectory for Qwen2.5-VL-7B.

This script reuses the already-tested helpers in:
    analyze_qwen7b_l26_l27_attention_overwrite_v1.py

Main grouping:
    native first-step correct vs wrong

For every scanned layer L:
    x      = L block input at prompt-last
    r_attn = x + attention_output
    y      = L block output

All three are read with:
    model final norm -> LM head -> left/right/above/below

Primary margin:
    decision_margin = logit(GT) - max(logit(other 3))

Secondary:
    opposite_margin = logit(GT) - logit(opposite(GT))

Wrong samples are classified as:
    never_formed
        y_L is never correct in the scan.

    lost_at_final_layer
        y_{final-1} is correct but final y is wrong.

    lost_before_final
        y_L was correct at least once, but is already wrong before final layer.

The script also reports x->r_attn and r_attn->y C->W / W->C transitions.

Example:
CUDA_VISIBLE_DEVICES=0 python -u \
analyze_qwen7b_correct_wrong_logitlens_trajectory_v1.py \
  --model qwen-7b \
  --num-samples 0 \
  --layers 14-27 \
  --device cuda:0 \
  --output-dir output/qwen7b_correct_wrong_logitlens_trajectory_v1 \
  --overwrite

Required alongside / importable:
    analyze_qwen7b_l26_l27_attention_overwrite_v1.py
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
SCRIPT_VERSION = "qwen7b-correct-wrong-logitlens-trajectory-v1"

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)


def args_parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--layers", default="14-27")
    p.add_argument("--trace-chunk-size", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--run-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    p.add_argument(
        "--helper-module",
        default="analyze_qwen7b_l26_l27_attention_overwrite_v1",
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out, seen = [], set()
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = map(int, chunk.split("-", 1))
            step = 1 if b >= a else -1
            vals = range(a, b + step, step)
        else:
            vals = [int(chunk)]
        for v in vals:
            if v not in seen:
                out.append(v)
                seen.add(v)
    return out


def safe_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.mean(xs)) if xs else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.median(xs)) if xs else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def relation_metrics(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    gt_id = RID[gt]
    wrong_ids = [i for i in range(4) if i != gt_id]
    competitor = max(wrong_ids, key=lambda i: float(s[i]))
    pred_id = int(np.argmax(s))

    centered = s - np.max(s)
    p = np.exp(centered)
    p /= p.sum()

    return {
        "pred": RELATIONS[pred_id],
        "correct": pred_id == gt_id,
        "decision_margin": float(s[gt_id] - s[competitor]),
        "opposite_margin": float(s[gt_id] - s[RID[OPPOSITE[gt]]]),
        "p_gt_4way": float(p[gt_id]),
        "top_competitor": RELATIONS[competitor],
    }


def longest_true_run(values: Sequence[bool]) -> int:
    best = cur = 0
    for value in values:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def wrong_taxonomy(layers: Sequence[int], y_correct: Sequence[bool]):
    correct_layers = [l for l, c in zip(layers, y_correct) if c]
    first = min(correct_layers) if correct_layers else None
    last = max(correct_layers) if correct_layers else None
    n_correct = len(correct_layers)
    run = longest_true_run(y_correct)

    if not correct_layers:
        category = "never_formed"
    elif len(y_correct) >= 2 and y_correct[-2] and not y_correct[-1]:
        category = "lost_at_final_layer"
    else:
        category = "lost_before_final"

    return category, first, last, n_correct, run


def trace_chunks(
    *,
    ah,
    model,
    batch,
    token_map,
    decoder_layers,
    layers,
    prompt_last,
    chunk_size,
):
    layers = sorted(set(map(int, layers)))
    if chunk_size <= 0:
        chunks = [layers]
    else:
        chunks = [
            layers[i : i + chunk_size]
            for i in range(0, len(layers), chunk_size)
        ]

    baseline = None
    traces_all = {}

    for chunk in chunks:
        current_baseline, traces = ah.run_and_trace(
            model=model,
            batch=batch,
            token_map=token_map,
            decoder_layers=decoder_layers,
            layer_indices=chunk,
            target_positions=[prompt_last],
        )
        if baseline is None:
            baseline = current_baseline
        traces_all.update({int(k): v for k, v in traces.items()})

    missing = [l for l in layers if l not in traces_all]
    if missing:
        raise RuntimeError(f"Missing traces: {missing}")

    return baseline, traces_all


def group_names(sample: Mapping[str, Any]) -> List[str]:
    if sample["firststep_correct"]:
        return ["firststep_correct"]

    out = ["firststep_wrong"]
    tax = sample["wrong_taxonomy"]

    if tax == "never_formed":
        out.append("wrong_never_formed")
    else:
        out.append("wrong_formed_then_lost")
        if tax == "lost_at_final_layer":
            out.append("wrong_lost_at_final")
        else:
            out.append("wrong_lost_before_final")
    return out


def summarize_groups(sample_rows, layer_rows):
    sample_map = {int(r["sid"]): r for r in sample_rows}
    grouped = defaultdict(list)

    for row in layer_rows:
        sample = sample_map[int(row["sid"])]
        for group in group_names(sample):
            grouped[(group, int(row["layer"]))].append(row)

    rows_out = []
    for (group, layer), rows in grouped.items():
        rows_out.append({
            "group": group,
            "layer": layer,
            "N": len(rows),

            "x_acc": safe_mean(float(r["x_correct"]) for r in rows),
            "rattn_acc": safe_mean(float(r["rattn_correct"]) for r in rows),
            "y_acc": safe_mean(float(r["y_correct"]) for r in rows),

            "x_decision_margin_mean": safe_mean(r["x_decision_margin"] for r in rows),
            "rattn_decision_margin_mean": safe_mean(r["rattn_decision_margin"] for r in rows),
            "y_decision_margin_mean": safe_mean(r["y_decision_margin"] for r in rows),

            "x_decision_margin_median": safe_median(r["x_decision_margin"] for r in rows),
            "rattn_decision_margin_median": safe_median(r["rattn_decision_margin"] for r in rows),
            "y_decision_margin_median": safe_median(r["y_decision_margin"] for r in rows),

            "x_opposite_margin_mean": safe_mean(r["x_opposite_margin"] for r in rows),
            "rattn_opposite_margin_mean": safe_mean(r["rattn_opposite_margin"] for r in rows),
            "y_opposite_margin_mean": safe_mean(r["y_opposite_margin"] for r in rows),

            "x_p_gt_4way_mean": safe_mean(r["x_p_gt_4way"] for r in rows),
            "rattn_p_gt_4way_mean": safe_mean(r["rattn_p_gt_4way"] for r in rows),
            "y_p_gt_4way_mean": safe_mean(r["y_p_gt_4way"] for r in rows),

            "attention_decision_gain_mean": safe_mean(r["attention_decision_gain"] for r in rows),
            "mlp_decision_gain_mean": safe_mean(r["mlp_decision_gain"] for r in rows),

            "attention_C_to_W": sum(bool(r["attention_C_to_W"]) for r in rows),
            "attention_W_to_C": sum(bool(r["attention_W_to_C"]) for r in rows),
            "mlp_C_to_W": sum(bool(r["mlp_C_to_W"]) for r in rows),
            "mlp_W_to_C": sum(bool(r["mlp_W_to_C"]) for r in rows),
        })

    order = {
        "firststep_correct": 0,
        "firststep_wrong": 1,
        "wrong_never_formed": 2,
        "wrong_formed_then_lost": 3,
        "wrong_lost_at_final": 4,
        "wrong_lost_before_final": 5,
    }
    rows_out.sort(key=lambda r: (order.get(r["group"], 99), int(r["layer"])))
    return rows_out


def summarize_generation_groups(sample_rows, layer_rows):
    sample_map = {int(r["sid"]): r for r in sample_rows}
    grouped = defaultdict(list)

    for row in layer_rows:
        sample = sample_map[int(row["sid"])]
        if sample["generation_pred"] is None:
            continue
        group = "generation_correct" if sample["generation_correct"] else "generation_wrong"
        grouped[(group, int(row["layer"]))].append(row)

    out = []
    for (group, layer), rows in grouped.items():
        out.append({
            "group": group,
            "layer": layer,
            "N": len(rows),
            "y_acc": safe_mean(float(r["y_correct"]) for r in rows),
            "y_decision_margin_mean": safe_mean(r["y_decision_margin"] for r in rows),
            "attention_decision_gain_mean": safe_mean(r["attention_decision_gain"] for r in rows),
            "mlp_decision_gain_mean": safe_mean(r["mlp_decision_gain"] for r in rows),
        })
    out.sort(key=lambda r: (r["group"], int(r["layer"])))
    return out


def main():
    a = args_parse()

    if a.model != "qwen-7b":
        raise ValueError("v1 supports qwen-7b only.")

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    layers = sorted(set(parse_layers(a.layers)))
    if not layers or min(layers) <= 0:
        raise ValueError("Need scan layers >= 1.")

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    errors = out / "errors.jsonl"

    helper = importlib.import_module(a.helper_module)
    probe = importlib.import_module(a.probe_module)
    ah = importlib.import_module(a.attention_helper_module)
    base = probe.base

    records, audit = base.load_records(a.dataset, Path(a.data_root), None)
    selected = helper.select_records(records, n=a.num_samples, seed=a.seed)

    (out / "selected_sids.json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "N_selected": len(selected),
                "sids": [int(r.sid) for r in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    spec = base.SPECS[a.model]
    cls = getattr(transformers, spec.model_class)

    model = processor = None

    try:
        print(f"Loading {a.model}: {spec.repo_id}", flush=True)

        model = cls.from_pretrained(
            spec.repo_id,
            dtype=base.resolve_dtype(spec.dtype_name),
            low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,
            device_map={"": a.device},
            attn_implementation=a.attn_impl,
        )
        model.eval()
        helper.clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        device = torch.device(a.device)
        decoder_layers, decoder_path = probe.resolve_decoder_layers(model)

        if len(decoder_layers) != 28:
            raise RuntimeError(
                f"Expected 28 Qwen-7B decoder layers, got {len(decoder_layers)}."
            )

        for layer in layers:
            if not 1 <= layer < len(decoder_layers):
                raise ValueError(f"Bad layer L{layer}")

        trace_layers = sorted(set([l - 1 for l in layers] + layers))

        final_norm, final_norm_path = helper.resolve_final_norm(
            model,
            decoder_path,
        )
        if final_norm is None:
            raise RuntimeError("Could not resolve final norm.")

        token_map = helper.relation_token_variants(processor.tokenizer)

        lens = helper.RelationLogitLens(
            model=model,
            final_norm=final_norm,
            token_map=token_map,
        )

        print("\n" + "=" * 190)
        print("QWEN-7B CORRECT-vs-WRONG TRUE LOGIT-LENS TRAJECTORY")
        print("=" * 190)
        print("N:", len(selected))
        print("layers:", layers)
        print("trace layers:", trace_layers)
        print("final norm:", final_norm_path)
        print("run generation:", a.run_generation)
        print("=" * 190, flush=True)

        sample_rows = []
        layer_rows = []
        final_match = 0
        final_match_n = 0

        for index, record in enumerate(
            tqdm(selected, desc="correct-vs-wrong-logitlens"),
            start=1,
        ):
            image = batch = None

            try:
                sid = int(record.sid)
                gt = helper.normalize_relation(record.relation)
                if gt not in RELATIONS:
                    raise RuntimeError(f"Bad GT: {record.relation!r}")

                question = a.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(record.image_path).convert("RGB")

                batch = helper.build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                prompt_last = int(batch["input_ids"].shape[1]) - 1

                baseline, traces = trace_chunks(
                    ah=ah,
                    model=model,
                    batch=batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    layers=trace_layers,
                    prompt_last=prompt_last,
                    chunk_size=a.trace_chunk_size,
                )

                first_pred = helper.normalize_relation(baseline["prediction"])
                if first_pred not in RELATIONS:
                    raise RuntimeError(f"Bad first-step prediction: {baseline['prediction']!r}")

                first_correct = first_pred == gt

                generation_pred = generation_text = None
                generation_correct = None

                if a.run_generation:
                    generation_pred, generation_text = helper.greedy_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=a.max_new_tokens,
                    )
                    generation_correct = generation_pred == gt

                current_rows = []
                y_correct_sequence = []

                for layer in layers:
                    x = helper.trace_block_state(
                        traces[layer - 1],
                        prompt_last,
                    )
                    attn = helper.trace_attention_state(
                        traces[layer],
                        prompt_last,
                    )
                    rattn = (x + attn).astype(np.float32)
                    y = helper.trace_block_state(
                        traces[layer],
                        prompt_last,
                    )
                    mlp = (y - rattn).astype(np.float32)

                    scores = lens.scores(
                        np.stack([x, rattn, y], axis=0)
                    )

                    xm = relation_metrics(scores[0], gt)
                    am = relation_metrics(scores[1], gt)
                    ym = relation_metrics(scores[2], gt)

                    y_correct_sequence.append(bool(ym["correct"]))

                    current_rows.append({
                        "sid": sid,
                        "layer": int(layer),
                        "gt": gt,
                        "native_firststep_pred": first_pred,
                        "firststep_correct": first_correct,
                        "generation_pred": generation_pred,
                        "generation_correct": generation_correct,

                        "x_pred": xm["pred"],
                        "x_correct": xm["correct"],
                        "x_decision_margin": xm["decision_margin"],
                        "x_opposite_margin": xm["opposite_margin"],
                        "x_p_gt_4way": xm["p_gt_4way"],
                        "x_top_competitor": xm["top_competitor"],

                        "rattn_pred": am["pred"],
                        "rattn_correct": am["correct"],
                        "rattn_decision_margin": am["decision_margin"],
                        "rattn_opposite_margin": am["opposite_margin"],
                        "rattn_p_gt_4way": am["p_gt_4way"],
                        "rattn_top_competitor": am["top_competitor"],

                        "y_pred": ym["pred"],
                        "y_correct": ym["correct"],
                        "y_decision_margin": ym["decision_margin"],
                        "y_opposite_margin": ym["opposite_margin"],
                        "y_p_gt_4way": ym["p_gt_4way"],
                        "y_top_competitor": ym["top_competitor"],

                        "attention_decision_gain": (
                            am["decision_margin"] - xm["decision_margin"]
                        ),
                        "mlp_decision_gain": (
                            ym["decision_margin"] - am["decision_margin"]
                        ),

                        "attention_C_to_W": bool(xm["correct"] and not am["correct"]),
                        "attention_W_to_C": bool((not xm["correct"]) and am["correct"]),
                        "mlp_C_to_W": bool(am["correct"] and not ym["correct"]),
                        "mlp_W_to_C": bool((not am["correct"]) and ym["correct"]),

                        "x_norm": float(np.linalg.norm(x)),
                        "attention_output_norm": float(np.linalg.norm(attn)),
                        "mlp_norm": float(np.linalg.norm(mlp)),
                        "y_norm": float(np.linalg.norm(y)),
                    })

                if first_correct:
                    correct_layers = [
                        l for l, c in zip(layers, y_correct_sequence) if c
                    ]
                    tax = "not_applicable"
                    first_correct_layer = min(correct_layers) if correct_layers else None
                    last_correct_layer = max(correct_layers) if correct_layers else None
                    n_correct_layers = len(correct_layers)
                    longest_run = longest_true_run(y_correct_sequence)
                else:
                    (
                        tax,
                        first_correct_layer,
                        last_correct_layer,
                        n_correct_layers,
                        longest_run,
                    ) = wrong_taxonomy(layers, y_correct_sequence)

                final_pred = current_rows[-1]["y_pred"]
                final_match_n += 1
                final_match += int(final_pred == first_pred)

                sample_row = {
                    "sid": sid,
                    "gt": gt,
                    "native_firststep_pred": first_pred,
                    "firststep_correct": first_correct,
                    "generation_pred": generation_pred,
                    "generation_text": generation_text,
                    "generation_correct": generation_correct,

                    "wrong_taxonomy": tax,
                    "ever_correct_in_scan": n_correct_layers > 0,
                    "first_correct_layer": first_correct_layer,
                    "last_correct_layer": last_correct_layer,
                    "number_correct_layers": n_correct_layers,
                    "longest_correct_run": longest_run,
                    "final_scanned_y_pred": final_pred,
                    "final_lens_matches_native": final_pred == first_pred,
                }
                sample_rows.append(sample_row)

                for row in current_rows:
                    row["wrong_taxonomy"] = tax
                    row["first_correct_layer"] = first_correct_layer
                    row["last_correct_layer"] = last_correct_layer
                    layer_rows.append(row)

                del traces

            except Exception as exc:
                append_jsonl(
                    errors,
                    {
                        "phase": "trace",
                        "sid": int(getattr(record, "sid", -1)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if a.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and a.empty_cache_every > 0
                    and index % a.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(out / "sample_summary.csv", sample_rows)
        write_csv(out / "sample_layer_trajectory.csv", layer_rows)

        group_summary = summarize_groups(sample_rows, layer_rows)
        write_csv(out / "group_layer_summary.csv", group_summary)

        generation_summary = summarize_generation_groups(sample_rows, layer_rows)
        write_csv(out / "generation_group_layer_summary.csv", generation_summary)

        # Correct-vs-wrong gap.
        lookup = {
            (row["group"], int(row["layer"])): row
            for row in group_summary
        }

        gap_rows = []
        prev_gap = None

        for layer in layers:
            c = lookup.get(("firststep_correct", layer))
            w = lookup.get(("firststep_wrong", layer))
            if c is None or w is None:
                continue

            y_gap = (
                float(c["y_decision_margin_mean"])
                - float(w["y_decision_margin_mean"])
            )

            gap_rows.append({
                "layer": layer,
                "correct_N": c["N"],
                "wrong_N": w["N"],

                "correct_y_acc": c["y_acc"],
                "wrong_y_acc": w["y_acc"],

                "correct_y_decision_margin": c["y_decision_margin_mean"],
                "wrong_y_decision_margin": w["y_decision_margin_mean"],
                "y_decision_margin_gap": y_gap,
                "gap_increase_from_previous_y": (
                    float("nan")
                    if prev_gap is None
                    else y_gap - prev_gap
                ),

                "correct_attention_gain": c["attention_decision_gain_mean"],
                "wrong_attention_gain": w["attention_decision_gain_mean"],
                "attention_gain_gap": (
                    float(c["attention_decision_gain_mean"])
                    - float(w["attention_decision_gain_mean"])
                ),

                "correct_mlp_gain": c["mlp_decision_gain_mean"],
                "wrong_mlp_gain": w["mlp_decision_gain_mean"],
                "mlp_gain_gap": (
                    float(c["mlp_decision_gain_mean"])
                    - float(w["mlp_decision_gain_mean"])
                ),
            })

            prev_gap = y_gap

        write_csv(out / "correct_vs_wrong_gap.csv", gap_rows)

        # Wrong taxonomy.
        wrong_samples = [
            row for row in sample_rows
            if not bool(row["firststep_correct"])
        ]

        counts = Counter(row["wrong_taxonomy"] for row in wrong_samples)
        taxonomy_rows = []

        for category, count in counts.most_common():
            subset = [
                row for row in wrong_samples
                if row["wrong_taxonomy"] == category
            ]

            taxonomy_rows.append({
                "wrong_taxonomy": category,
                "N": count,
                "fraction_of_firststep_wrong": count / max(len(wrong_samples), 1),
                "mean_number_correct_layers": safe_mean(
                    row["number_correct_layers"] for row in subset
                ),
                "mean_longest_correct_run": safe_mean(
                    row["longest_correct_run"] for row in subset
                ),
            })

        write_csv(out / "wrong_taxonomy_summary.csv", taxonomy_rows)

        # First/last correct-layer histograms for wrong samples.
        hist_rows = []

        for field in ("first_correct_layer", "last_correct_layer"):
            counter = Counter(
                "none" if row[field] is None else f"L{int(row[field])}"
                for row in wrong_samples
            )
            for value, count in counter.items():
                hist_rows.append({
                    "group": "firststep_wrong",
                    "metric": field,
                    "value": value,
                    "N": count,
                    "fraction": count / max(len(wrong_samples), 1),
                })

        write_csv(
            out / "first_last_correct_layer_histogram.csv",
            hist_rows,
        )

        # Detailed module transition counts correct vs wrong.
        transition_rows = []

        for group in ("firststep_correct", "firststep_wrong"):
            sids = {
                int(row["sid"])
                for row in sample_rows
                if (
                    bool(row["firststep_correct"])
                    == (group == "firststep_correct")
                )
            }

            for layer in layers:
                rows = [
                    row for row in layer_rows
                    if int(row["sid"]) in sids and int(row["layer"]) == layer
                ]

                x_correct_n = sum(bool(r["x_correct"]) for r in rows)
                x_wrong_n = len(rows) - x_correct_n
                rattn_correct_n = sum(bool(r["rattn_correct"]) for r in rows)
                rattn_wrong_n = len(rows) - rattn_correct_n

                attn_cw = sum(bool(r["attention_C_to_W"]) for r in rows)
                attn_wc = sum(bool(r["attention_W_to_C"]) for r in rows)
                mlp_cw = sum(bool(r["mlp_C_to_W"]) for r in rows)
                mlp_wc = sum(bool(r["mlp_W_to_C"]) for r in rows)

                transition_rows.append({
                    "group": group,
                    "layer": layer,
                    "N": len(rows),

                    "attention_C_to_W": attn_cw,
                    "attention_W_to_C": attn_wc,
                    "attention_net_repairs": attn_wc - attn_cw,
                    "attention_C_to_W_given_x_correct": (
                        attn_cw / max(x_correct_n, 1)
                    ),
                    "attention_W_to_C_given_x_wrong": (
                        attn_wc / max(x_wrong_n, 1)
                    ),

                    "mlp_C_to_W": mlp_cw,
                    "mlp_W_to_C": mlp_wc,
                    "mlp_net_repairs": mlp_wc - mlp_cw,
                    "mlp_C_to_W_given_rattn_correct": (
                        mlp_cw / max(rattn_correct_n, 1)
                    ),
                    "mlp_W_to_C_given_rattn_wrong": (
                        mlp_wc / max(rattn_wrong_n, 1)
                    ),
                })

        write_csv(out / "transition_summary.csv", transition_rows)

        # Console.
        N = len(sample_rows)
        first_acc = safe_mean(
            float(row["firststep_correct"]) for row in sample_rows
        )
        gen_acc = safe_mean(
            float(row["generation_correct"])
            for row in sample_rows
            if row["generation_correct"] is not None
        )
        lens_match = final_match / max(final_match_n, 1)

        print("\n" + "=" * 200)
        print("CORRECT-vs-WRONG TRUE LOGIT-LENS SUMMARY")
        print("=" * 200)
        print(
            f"N={N} | first-step ACC={100*first_acc:.2f}% "
            + (
                f"| generation ACC={100*gen_acc:.2f}%"
                if math.isfinite(gen_acc)
                else ""
            )
        )
        print(
            f"Final scanned y vs native first-step match: {100*lens_match:.2f}%"
        )

        print("\nWRONG TAXONOMY")
        for row in taxonomy_rows:
            print(
                f"  {row['wrong_taxonomy']:<24s} "
                f"N={int(row['N']):3d}/{len(wrong_samples):3d} "
                f"({100*float(row['fraction_of_firststep_wrong']):6.2f}%)"
            )

        print("\nCORRECT vs WRONG y_L")
        print(
            f"  {'L':>3s} {'Cacc':>7s} {'Wacc':>7s} "
            f"{'Cmargin':>10s} {'Wmargin':>10s} {'gap':>10s} {'gapΔ':>10s} "
            f"{'CattnΔ':>10s} {'WattnΔ':>10s} {'CmlpΔ':>10s} {'WmlpΔ':>10s}"
        )

        for row in gap_rows:
            layer = int(row["layer"])
            gap_delta = float(row["gap_increase_from_previous_y"])
            gap_text = (
                f"{gap_delta:+10.3f}"
                if math.isfinite(gap_delta)
                else f"{'nan':>10s}"
            )

            print(
                f"  {layer:3d} "
                f"{100*float(row['correct_y_acc']):6.2f}% "
                f"{100*float(row['wrong_y_acc']):6.2f}% "
                f"{float(row['correct_y_decision_margin']):+10.3f} "
                f"{float(row['wrong_y_decision_margin']):+10.3f} "
                f"{float(row['y_decision_margin_gap']):+10.3f} "
                f"{gap_text} "
                f"{float(row['correct_attention_gain']):+10.3f} "
                f"{float(row['wrong_attention_gain']):+10.3f} "
                f"{float(row['correct_mlp_gain']):+10.3f} "
                f"{float(row['wrong_mlp_gain']):+10.3f}"
            )

        finite_gap = [
            row for row in gap_rows
            if math.isfinite(float(row["gap_increase_from_previous_y"]))
        ]

        if finite_gap:
            div = max(
                finite_gap,
                key=lambda row: float(row["gap_increase_from_previous_y"]),
            )
            print(
                "\nLargest descriptive gap increase:",
                f"L{int(div['layer'])}",
                f"gapΔ={float(div['gap_increase_from_previous_y']):+.4f}",
            )

        print("\nWRONG SUBTYPE y_L decision margin")
        subtype_names = (
            "wrong_never_formed",
            "wrong_formed_then_lost",
            "wrong_lost_at_final",
            "wrong_lost_before_final",
        )

        for layer in layers:
            parts = [f"L{layer:02d}"]
            for name in subtype_names:
                row = lookup.get((name, layer))
                if row is None:
                    parts.append(f"{name}=N/A")
                else:
                    parts.append(
                        f"{name}={float(row['y_decision_margin_mean']):+.3f}"
                    )
            print("  " + " | ".join(parts))

        print("\nMODULE TRANSITIONS ON FIRSTSTEP-WRONG")
        wrong_transition = {
            int(row["layer"]): row
            for row in transition_rows
            if row["group"] == "firststep_wrong"
        }

        print(
            f"  {'L':>3s} {'A C->W':>7s} {'A W->C':>7s} {'A net':>7s} "
            f"{'M C->W':>7s} {'M W->C':>7s} {'M net':>7s}"
        )

        for layer in layers:
            row = wrong_transition[layer]
            print(
                f"  {layer:3d} "
                f"{int(row['attention_C_to_W']):7d} "
                f"{int(row['attention_W_to_C']):7d} "
                f"{int(row['attention_net_repairs']):+7d} "
                f"{int(row['mlp_C_to_W']):7d} "
                f"{int(row['mlp_W_to_C']):7d} "
                f"{int(row['mlp_net_repairs']):+7d}"
            )

        print("=" * 200)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "dataset": a.dataset,
            "N": N,
            "seed": a.seed,
            "scan_layers": layers,
            "trace_layers": trace_layers,
            "trace_chunk_size": a.trace_chunk_size,
            "final_norm_path": final_norm_path,
            "firststep_acc": first_acc,
            "generation_acc": gen_acc,
            "final_lens_native_match": lens_match,
            "primary_grouping": "native first-step correct vs wrong",
            "primary_margin": "GT logit - max(other 3 relation logits)",
            "secondary_margin": "GT logit - opposite relation logit",
            "wrong_taxonomy": {
                "never_formed": "No scanned y_L predicts GT.",
                "lost_at_final_layer": "y_{final-1}=GT but final y wrong.",
                "lost_before_final": "At least one earlier y_L=GT, but y_{final-1} already wrong.",
            },
            "uses_intervention": False,
            "audit": audit,
        }

        (out / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = [
            f"script_version: {SCRIPT_VERSION}",
            f"N={N}",
            f"firststep_acc={100*first_acc:.2f}%",
            (
                f"generation_acc={100*gen_acc:.2f}%"
                if math.isfinite(gen_acc)
                else "generation_acc=N/A"
            ),
            f"final_lens_native_match={100*lens_match:.2f}%",
            "",
            "WRONG TAXONOMY",
        ]

        for row in taxonomy_rows:
            report.append(
                f"{row['wrong_taxonomy']}: "
                f"{int(row['N'])}/{len(wrong_samples)} "
                f"({100*float(row['fraction_of_firststep_wrong']):.2f}%)"
            )

        report += ["", "CORRECT-vs-WRONG GAP"]
        for row in gap_rows:
            report.append(
                f"L{int(row['layer'])}: "
                f"Cmargin={float(row['correct_y_decision_margin']):+.4f} "
                f"Wmargin={float(row['wrong_y_decision_margin']):+.4f} "
                f"gap={float(row['y_decision_margin_gap']):+.4f} "
                f"Cattn={float(row['correct_attention_gain']):+.4f} "
                f"Wattn={float(row['wrong_attention_gain']):+.4f} "
                f"Cmlp={float(row['correct_mlp_gain']):+.4f} "
                f"Wmlp={float(row['wrong_mlp_gain']):+.4f}"
            )

        report += [
            "",
            "INTERPRETATION",
            (
                "If never_formed dominates, errors mainly reflect failure to form/read out "
                "the correct relation rather than a late overwrite."
            ),
            (
                "If formed_then_lost is substantial, inspect the layer after each sample's "
                "last_correct_layer and the C->W module transitions."
            ),
            (
                "If correct-vs-wrong margins separate around L19-L21 and remain separated, "
                "the main candidate locus is relation integration/object-to-last transfer."
            ),
            (
                "If the trajectories stay similar through L26 and wrong samples selectively "
                "collapse at L27, L27 sample-dependent interference becomes plausible."
            ),
            "This analysis is descriptive; divergence is not itself causal proof.",
        ]

        (out / "report.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for filename in (
            "selected_sids.json",
            "sample_summary.csv",
            "sample_layer_trajectory.csv",
            "group_layer_summary.csv",
            "correct_vs_wrong_gap.csv",
            "transition_summary.csv",
            "wrong_taxonomy_summary.csv",
            "first_last_correct_layer_histogram.csv",
            "generation_group_layer_summary.csv",
            "config.json",
            "report.txt",
        ):
            print(" ", out / filename)

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
