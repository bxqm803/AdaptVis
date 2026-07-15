#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen2-VL-7B Controlled-A centroid evaluation.

Model:
    Qwen/Qwen2-VL-7B-Instruct

This script performs prompt-only forward passes. It does not call generate(),
does not train or fine-tune the model, and does not compute candidate-answer
likelihoods.

For every Controlled-A sample it evaluates the original question

    Where is object A in relation to object B?

and the role-swapped question

    Where is object B in relation to object A?

The swapped result is aligned back to A relative to B. The reported headline
metrics follow the convention used in the previous COCO/Controlled-A runs:

1. Similarity centroid:
       average(original centroid, aligned swapped centroid)
   Accuracy is computed at every decoder layer, and the best layer is reported.

2. Single-head attention centroid:
       average(original centroid, aligned swapped centroid)
   Accuracy is computed for every decoder layer and every attention head, and
   the best single layer/head is reported.

The relation classifier uses the dominant centroid-displacement axis:
    dx < 0 -> left
    dx > 0 -> right
    dy < 0 -> on
    dy > 0 -> under
and chooses the axis with larger absolute displacement.

Important:
The best layer and best head are selected using GT on the full evaluation set.
They are diagnostic oracle statistics, not label-free deployment results.

This script reuses dataset, prompt, image, token-span, and visual-grid helper
functions from:
    analyze_controlledA_similarity_head_generation_step1_v1.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except Exception as exc:
    raise SystemExit(
        "Could not import Qwen2VLForConditionalGeneration. "
        f"transformers error: {type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "qwen2vl7b-controlledA-centroid-v1"
MODEL_KEY = "qwen2-vl-7b"
MODEL_REPO = "Qwen/Qwen2-VL-7B-Instruct"

RELATIONS = ("left", "right", "on", "under")
RELATION_TO_CODE = {
    relation: index for index, relation in enumerate(RELATIONS)
}
CODE_TO_RELATION = {
    index: relation for index, relation in enumerate(RELATIONS)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--helper-module",
        default="analyze_controlledA_similarity_head_generation_step1_v1",
        help="Existing Controlled-A helper module, without .py.",
    )
    parser.add_argument(
        "--controlled-module",
        default="",
        help=(
            "Optional explicit Controlled-A extractor module. Otherwise the "
            "helper module tries its known Controlled-A extractor names."
        ),
    )
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument("--dataset-key", default="Controlled_Images_A")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--layers",
        default="all",
        help=(
            "Comma-separated zero-based language decoder layers or 'all'. "
            "Use all to find the true best layer/head."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for hidden-state similarity maps.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="output/qwen2vl7b_controlledA_centroid",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed sample.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    aliases = {
        "left": "left",
        "left of": "left",
        "right": "right",
        "right of": "right",
        "on": "on",
        "above": "on",
        "over": "on",
        "on top of": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
    }
    return aliases.get(text)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_helper(name: str) -> Any:
    helper = importlib.import_module(name)
    required = [
        "import_controlled_module",
        "load_standard_prompts",
        "record_image",
        "make_question_batch",
        "build_swapped_question",
        "configure_processor",
        "resolve_decoder_layers",
        "parse_layers",
        "locate_object_spans",
        "resolve_visual_indices",
        "visual_coordinates",
        "normalize_attention_tensor",
        "query_attention_metrics",
        "similarity_layer_metrics",
        "relation_codes_from_centroids",
    ]
    missing = [key for key in required if not hasattr(helper, key)]
    if missing:
        raise RuntimeError(
            f"Helper module {name!r} is missing functions: {missing}"
        )
    return helper


def model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not resolve the model input device")


def load_model(
    args: argparse.Namespace,
    helper: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    dtype = resolve_dtype(args.dtype)
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": {"": args.device},
        "attn_implementation": "eager",
    }

    print(f"Loading {MODEL_REPO}")
    print(
        f"dtype={args.dtype} | device={args.device} | "
        "attn_implementation=eager"
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_REPO,
        **load_kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_REPO)
    helper.configure_processor(model, processor)

    decoder_layers, decoder_path = helper.resolve_decoder_layers(model)
    selected_layers = helper.parse_layers(
        args.layers,
        len(decoder_layers),
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "model": MODEL_KEY,
        "repo_id": MODEL_REPO,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": "eager",
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "selected_layers": selected_layers,
        "temperature": args.temperature,
        "uses_generation": False,
        "uses_gt_for_best_layer_head_selection": True,
        "modifies_model": False,
    }
    return model, processor, metadata


def analyze_prompt_forward(
    *,
    helper: Any,
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    temperature: float,
) -> Dict[str, Any]:
    """Extract similarity and object-token attention centroids from one prompt."""
    input_ids_tensor = batch["input_ids"]
    if int(input_ids_tensor.shape[0]) != 1:
        raise ValueError(
            f"Expected batch size 1, got {tuple(input_ids_tensor.shape)}"
        )

    input_ids = input_ids_tensor[0].detach().cpu().tolist()
    input_length = len(input_ids)

    subject_span, reference_span = helper.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    # Keep the same convention as the previous scripts: last token in each
    # object-word span.
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])

    visual_indices = helper.resolve_visual_indices(
        model,
        processor,
        batch,
        input_ids,
    )
    coords = helper.visual_coordinates(
        model,
        batch,
        len(visual_indices),
        input_ids_tensor.device,
    )
    if coords is None:
        raise RuntimeError(
            f"Could not construct coordinates for "
            f"{len(visual_indices)} visual tokens"
        )

    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    attentions = getattr(outputs, "attentions", None)
    hidden_states = getattr(outputs, "hidden_states", None)
    if attentions is None:
        raise RuntimeError(
            "Model forward did not return attentions. "
            "Qwen2-VL must be loaded with attn_implementation='eager'."
        )
    if hidden_states is None:
        raise RuntimeError("Model forward did not return hidden states")

    first_attention = helper.normalize_attention_tensor(
        attentions[selected_layers[0]],
        expected_query_length=input_length,
    )
    n_heads = int(first_attention.shape[0])
    n_layers = len(selected_layers)
    n_visual = len(visual_indices)

    object_maps = np.zeros(
        (n_layers, n_heads, 2, n_visual),
        dtype=np.float32,
    )
    object_centroids = np.zeros(
        (n_layers, n_heads, 2, 2),
        dtype=np.float32,
    )
    object_separation = np.zeros(
        (n_layers, n_heads),
        dtype=np.float32,
    )
    object_visual_mass = np.zeros(
        (n_layers, n_heads, 2),
        dtype=np.float32,
    )

    query_indices = [subject_index, reference_index]

    for output_position, layer in enumerate(selected_layers):
        attention = helper.normalize_attention_tensor(
            attentions[layer],
            expected_query_length=input_length,
        )
        if int(attention.shape[0]) != n_heads:
            raise RuntimeError(
                f"Head count changed at L{layer}: "
                f"{attention.shape[0]} vs {n_heads}"
            )

        rows = attention[:, query_indices, :]
        metrics = helper.query_attention_metrics(
            rows,
            visual_indices,
            coords,
            subject_index,
            reference_index,
        )

        maps = metrics["visual_maps"]
        centers = metrics["centroids"]

        object_maps[output_position] = (
            maps.detach().float().cpu().numpy()
        )
        object_centroids[output_position] = (
            centers.detach().float().cpu().numpy()
        )
        object_visual_mass[output_position] = (
            metrics["visual_mass"].detach().float().cpu().numpy()
        )
        object_separation[output_position] = (
            0.5
            * torch.sum(
                torch.abs(maps[:, 0, :] - maps[:, 1, :]),
                dim=-1,
            )
        ).detach().float().cpu().numpy()

    object_prediction, object_axis_confidence = (
        helper.relation_codes_from_centroids(object_centroids)
    )

    similarity = helper.similarity_layer_metrics(
        hidden_layers=hidden_states,
        selected_layers=selected_layers,
        subject_index=subject_index,
        reference_index=reference_index,
        visual_indices=visual_indices,
        coords=coords,
        temperature=temperature,
    )

    result = {
        "n_heads": n_heads,
        "visual_coordinates": (
            coords.detach().float().cpu().numpy().astype(np.float32)
        ),
        "object_maps": object_maps,
        "object_centroids": object_centroids,
        "object_prediction": object_prediction,
        "object_axis_confidence": object_axis_confidence,
        "object_separation": object_separation,
        "object_visual_mass": object_visual_mass,
        "similarity_centroids": similarity["centroids"],
        "similarity_prediction": similarity["prediction"],
        "similarity_axis_confidence": similarity["axis_confidence"],
        "similarity_separation": similarity["separation"],
    }

    del outputs
    del attentions
    del hidden_states
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def combine_original_swap(
    *,
    helper: Any,
    original: Dict[str, Any],
    swapped: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Align swapped [B,A] back to [A,B], then average centroids."""
    swapped_attention_centroids_aligned = swapped[
        "object_centroids"
    ][:, :, [1, 0], :]
    attention_average_centroids = 0.5 * (
        original["object_centroids"]
        + swapped_attention_centroids_aligned
    )
    attention_average_prediction, attention_axis_confidence = (
        helper.relation_codes_from_centroids(
            attention_average_centroids
        )
    )

    swapped_similarity_centroids_aligned = swapped[
        "similarity_centroids"
    ][:, [1, 0], :]
    similarity_average_centroids = 0.5 * (
        original["similarity_centroids"]
        + swapped_similarity_centroids_aligned
    )
    similarity_average_prediction, similarity_axis_confidence = (
        helper.relation_codes_from_centroids(
            similarity_average_centroids
        )
    )

    return {
        "attention_average_centroids": (
            attention_average_centroids.astype(np.float32)
        ),
        "attention_average_prediction": (
            attention_average_prediction.astype(np.int8)
        ),
        "attention_axis_confidence": (
            attention_axis_confidence.astype(np.float32)
        ),
        "similarity_average_centroids": (
            similarity_average_centroids.astype(np.float32)
        ),
        "similarity_average_prediction": (
            similarity_average_prediction.astype(np.int8)
        ),
        "similarity_axis_confidence": (
            similarity_axis_confidence.astype(np.float32)
        ),
    }


def current_best(
    similarity_correct: np.ndarray,
    attention_correct: np.ndarray,
    count: int,
    selected_layers: Sequence[int],
) -> Tuple[str, str]:
    if count <= 0:
        return "-", "-"

    similarity_accuracy = similarity_correct / count
    similarity_position = int(np.argmax(similarity_accuracy))
    similarity_text = (
        f"{similarity_accuracy[similarity_position]:.4f}"
        f"@L{selected_layers[similarity_position]}"
    )

    attention_accuracy = attention_correct / count
    flat_position = int(np.argmax(attention_accuracy))
    layer_position, head = np.unravel_index(
        flat_position,
        attention_accuracy.shape,
    )
    attention_text = (
        f"{attention_accuracy[layer_position, head]:.4f}"
        f"@L{selected_layers[layer_position]}H{head}"
    )
    return similarity_text, attention_text


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors_path = output_dir / "errors.jsonl"
    progress_path = output_dir / "progress_samples.jsonl"
    selected_samples_path = output_dir / "selected_predictions.csv"
    per_relation_path = output_dir / "per_relation.csv"
    summary_path = output_dir / "summary.json"
    aggregate_path = output_dir / "aggregate_metrics.npz"
    config_path = output_dir / "config.json"

    if progress_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"{progress_path} already exists. Use --overwrite for a fresh run."
        )

    helper = load_helper(args.helper_module)
    controlled_module = helper.import_controlled_module(
        args.controlled_module
    )

    prompt_path = Path(args.prompt_jsonl)
    records, audit = controlled_module.load_records(
        prompt_path,
        dataset_key=args.dataset_key,
        keep_relations=list(RELATIONS),
        download=args.download,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    if not records:
        raise RuntimeError("No usable Controlled-A records")

    prompt_rows = helper.load_standard_prompts(prompt_path)
    missing_sids = [
        int(record.sid)
        for record in records
        if int(record.sid) not in prompt_rows
    ]
    if missing_sids:
        raise RuntimeError(
            f"Prompt file lacks {len(missing_sids)} record IDs; "
            f"first={missing_sids[:10]}"
        )

    model = None
    processor = None
    model, processor, metadata = load_model(args, helper)
    selected_layers = list(metadata["selected_layers"])
    input_device = model_input_device(model)

    metadata.update({
        "prompt_jsonl": str(prompt_path),
        "dataset_key": args.dataset_key,
        "controlled_module": controlled_module.__name__,
        "helper_module": args.helper_module,
        "n_records": len(records),
        "audit": audit,
    })
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    similarity_correct: Optional[np.ndarray] = None
    attention_correct: Optional[np.ndarray] = None
    similarity_count_by_relation: Dict[str, np.ndarray] = {}
    attention_count_by_relation: Dict[str, np.ndarray] = {}
    relation_n: Dict[str, int] = defaultdict(int)

    sample_cache: List[Dict[str, Any]] = []
    completed = 0
    started = time.time()

    print(f"Script: {SCRIPT_VERSION}")
    print(f"Model: {MODEL_REPO}")
    print(f"Samples: {len(records)}")
    print(f"Layers: {selected_layers}")
    print("Protocol: prompt-only forward; original/swap-aligned centroid average")

    try:
        for record in tqdm(
            records,
            desc="qwen2-vl-7b:controlledA:centroid",
        ):
            sid = int(record.sid)
            image = None
            original_batch = None
            swapped_batch = None

            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                gt = normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATION_TO_CODE:
                    raise ValueError(
                        f"Unsupported GT for sid={sid}: "
                        f"{prompt_row['answer_raw']!r}"
                    )
                gt_code = RELATION_TO_CODE[gt]

                image = helper.record_image(record)
                original_batch = helper.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=input_device,
                )

                swapped_question = helper.build_swapped_question(
                    subject,
                    reference,
                )
                swapped_batch = helper.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=swapped_question,
                    device=input_device,
                )

                original = analyze_prompt_forward(
                    helper=helper,
                    model=model,
                    processor=processor,
                    batch=original_batch,
                    subject=subject,
                    reference=reference,
                    selected_layers=selected_layers,
                    temperature=args.temperature,
                )
                swapped = analyze_prompt_forward(
                    helper=helper,
                    model=model,
                    processor=processor,
                    batch=swapped_batch,
                    subject=reference,
                    reference=subject,
                    selected_layers=selected_layers,
                    temperature=args.temperature,
                )

                if original["n_heads"] != swapped["n_heads"]:
                    raise RuntimeError(
                        "Original/swap head count mismatch: "
                        f"{original['n_heads']} vs {swapped['n_heads']}"
                    )

                combined = combine_original_swap(
                    helper=helper,
                    original=original,
                    swapped=swapped,
                )

                similarity_prediction = combined[
                    "similarity_average_prediction"
                ]
                attention_prediction = combined[
                    "attention_average_prediction"
                ]

                if similarity_correct is None:
                    n_layers = len(selected_layers)
                    n_heads = int(original["n_heads"])
                    similarity_correct = np.zeros(
                        n_layers,
                        dtype=np.int64,
                    )
                    attention_correct = np.zeros(
                        (n_layers, n_heads),
                        dtype=np.int64,
                    )
                    for relation in RELATIONS:
                        similarity_count_by_relation[relation] = np.zeros(
                            n_layers,
                            dtype=np.int64,
                        )
                        attention_count_by_relation[relation] = np.zeros(
                            (n_layers, n_heads),
                            dtype=np.int64,
                        )

                assert attention_correct is not None
                similarity_correct += (
                    similarity_prediction == gt_code
                )
                attention_correct += (
                    attention_prediction == gt_code
                )
                similarity_count_by_relation[gt] += (
                    similarity_prediction == gt_code
                )
                attention_count_by_relation[gt] += (
                    attention_prediction == gt_code
                )
                relation_n[gt] += 1

                sample_cache.append({
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "question": question,
                    "gt": gt,
                    "similarity_prediction": similarity_prediction.copy(),
                    "similarity_axis_confidence": combined[
                        "similarity_axis_confidence"
                    ].copy(),
                    "attention_prediction": attention_prediction.copy(),
                    "attention_axis_confidence": combined[
                        "attention_axis_confidence"
                    ].copy(),
                })

                completed += 1
                append_jsonl(progress_path, {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "gt": gt,
                    "completed": completed,
                })

                if (
                    completed == 1
                    or completed % max(1, args.print_every) == 0
                    or completed == len(records)
                ):
                    similarity_text, attention_text = current_best(
                        similarity_correct,
                        attention_correct,
                        completed,
                        selected_layers,
                    )
                    tqdm.write(
                        f"\n[{completed}/{len(records)}] sid={sid} | "
                        f"{subject} -> {reference} | GT={gt}\n"
                        f"  current best similarity: {similarity_text}\n"
                        f"  current best head:       {attention_text}"
                    )

                del original
                del swapped
                del combined

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-24:],
                })
                tqdm.write(
                    f"\n[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if args.stop_on_error:
                    raise
            finally:
                if original_batch is not None:
                    original_batch.clear()
                if swapped_batch is not None:
                    swapped_batch.clear()
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if completed == 0:
            raise RuntimeError(
                f"No sample completed. Inspect {errors_path}"
            )
        assert similarity_correct is not None
        assert attention_correct is not None

        similarity_accuracy = similarity_correct / completed
        attention_accuracy = attention_correct / completed

        best_similarity_position = int(
            np.argmax(similarity_accuracy)
        )
        best_similarity_layer = int(
            selected_layers[best_similarity_position]
        )
        best_similarity_accuracy = float(
            similarity_accuracy[best_similarity_position]
        )

        best_attention_flat = int(np.argmax(attention_accuracy))
        best_attention_layer_position, best_attention_head = (
            np.unravel_index(
                best_attention_flat,
                attention_accuracy.shape,
            )
        )
        best_attention_layer = int(
            selected_layers[best_attention_layer_position]
        )
        best_attention_accuracy = float(
            attention_accuracy[
                best_attention_layer_position,
                best_attention_head,
            ]
        )

        selected_rows: List[Dict[str, Any]] = []
        for sample in sample_cache:
            sim_code = int(
                sample["similarity_prediction"][
                    best_similarity_position
                ]
            )
            head_code = int(
                sample["attention_prediction"][
                    best_attention_layer_position,
                    best_attention_head,
                ]
            )
            gt = sample["gt"]
            selected_rows.append({
                "sid": sample["sid"],
                "subject": sample["subject"],
                "reference": sample["reference"],
                "question": sample["question"],
                "gt": gt,
                "similarity_best_layer": best_similarity_layer,
                "similarity_prediction": CODE_TO_RELATION.get(
                    sim_code,
                    "unknown",
                ),
                "similarity_correct": (
                    sim_code == RELATION_TO_CODE[gt]
                ),
                "similarity_axis_confidence": float(
                    sample["similarity_axis_confidence"][
                        best_similarity_position
                    ]
                ),
                "best_attention_layer": best_attention_layer,
                "best_attention_head": int(best_attention_head),
                "best_attention_prediction": CODE_TO_RELATION.get(
                    head_code,
                    "unknown",
                ),
                "best_attention_correct": (
                    head_code == RELATION_TO_CODE[gt]
                ),
                "best_attention_axis_confidence": float(
                    sample["attention_axis_confidence"][
                        best_attention_layer_position,
                        best_attention_head,
                    ]
                ),
            })
        write_csv(selected_samples_path, selected_rows)

        per_relation_rows: List[Dict[str, Any]] = []
        for relation in RELATIONS:
            n_relation = int(relation_n[relation])
            if n_relation == 0:
                continue
            per_relation_rows.append({
                "relation": relation,
                "n": n_relation,
                "similarity_accuracy": float(
                    similarity_count_by_relation[relation][
                        best_similarity_position
                    ] / n_relation
                ),
                "best_attention_head_accuracy": float(
                    attention_count_by_relation[relation][
                        best_attention_layer_position,
                        best_attention_head,
                    ] / n_relation
                ),
                "similarity_layer": best_similarity_layer,
                "attention_layer": best_attention_layer,
                "attention_head": int(best_attention_head),
            })
        write_csv(per_relation_path, per_relation_rows)

        np.savez_compressed(
            aggregate_path,
            layer_indices=np.asarray(
                selected_layers,
                dtype=np.int32,
            ),
            similarity_correct=similarity_correct,
            similarity_accuracy=similarity_accuracy.astype(np.float32),
            attention_correct=attention_correct,
            attention_accuracy=attention_accuracy.astype(np.float32),
            relation_names=np.asarray(RELATIONS, dtype="<U8"),
        )

        summary = {
            "script_version": SCRIPT_VERSION,
            "model": MODEL_KEY,
            "repo_id": MODEL_REPO,
            "n": completed,
            "n_requested": len(records),
            "similarity_centroid_accuracy": (
                best_similarity_accuracy
            ),
            "similarity_centroid_best_layer": (
                best_similarity_layer
            ),
            "best_attention_head_centroid_accuracy": (
                best_attention_accuracy
            ),
            "best_attention_head_layer": (
                best_attention_layer
            ),
            "best_attention_head": int(best_attention_head),
            "centroid_protocol": (
                "average of original and role-aligned swapped centroids"
            ),
            "relation_rule": (
                "dominant axis of subject-reference centroid displacement"
            ),
            "selection_note": (
                "best layer/head selected using full-set GT; "
                "diagnostic oracle statistic"
            ),
            "temperature": args.temperature,
            "elapsed_minutes": (
                time.time() - started
            ) / 60.0,
            "per_relation": per_relation_rows,
            "config": metadata,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 108)
        print("QWEN2-VL-7B CONTROLLED-A CENTROID RESULT")
        print("=" * 108)
        print(f"Samples: {completed}/{len(records)}")
        print(
            "1. Similarity centroid:          "
            f"{best_similarity_accuracy:.4f} "
            f"at L{best_similarity_layer}"
        )
        print(
            "2. Best attention-head centroid: "
            f"{best_attention_accuracy:.4f} "
            f"at L{best_attention_layer}H{best_attention_head}"
        )
        print(
            "\nSelection caveat: the best layer/head uses full-set GT."
        )
        print(f"\nSummary:       {summary_path}")
        print(f"Aggregate:     {aggregate_path}")
        print(f"Per relation:  {per_relation_path}")
        print(f"Per sample:    {selected_samples_path}")
        if errors_path.exists() and errors_path.stat().st_size:
            print(f"Errors:        {errors_path}")

    finally:
        if processor is not None:
            del processor
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
