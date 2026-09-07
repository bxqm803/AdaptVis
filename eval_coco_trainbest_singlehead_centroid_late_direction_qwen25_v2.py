#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_coco_trainbest_singlehead_centroid_late_direction_qwen25_v2.py

Goal
====
Use the repo's ATTENTION-CENTROID spatial reader to select which already-known
late causal relation direction to inject into the LAST TOKEN.

Models
======
    qwen-3b = Qwen/Qwen2.5-VL-3B-Instruct
    qwen-7b = Qwen/Qwen2.5-VL-7B-Instruct

Dataset
=======
    COCO_QA_two_obj, four relations:
        left / right / above / below

Read -> Select -> Write
======================

READ / SELECT:
    TRAIN 30% only:
      1. For every decoder layer and attention head, extract the subject and
         reference TEXT-token attention centroids over VISUAL tokens.
      2. Run both:
             Q_AB: A relative to B
             Q_BA: B relative to A
      3. Align Q_BA role order [B,A] back to [A,B].
      4. Average centroids:
             c_avg = 0.5 * (c_AB + c_BA_aligned)
      5. Convert each (layer, head) centroid pair to
         left/right/above/below using |dx| vs |dy|.
      6. Select the single (layer, head) with highest TRAIN accuracy.

    TEST:
      The chosen (layer, head) is frozen.
      No TEST GT is used by the selector.
      For each TEST sample, original+swap aligned centroids are averaged and
      the same |dx| vs |dy| rule predicts one relation.

    --centroid-layer auto searches all decoder layers on TRAIN.
    Passing an integer restricts TRAIN head selection to that layer.

WRITE:
    TRAIN only, at known late actuator layers:

        delta_i,l = h_last(real)_i,l - h_last(gray)_i,l
        mu_r,l    = mean(delta_i,l | relation=r)
        mu_global = balanced mean over the four relation means
        s_r,l     = mu_r,l - mu_global,l

    Known actuator windows reused from the repo:
        qwen-3b: L32,L33,L34,L35
        qwen-7b: L25,L26,L27

    TEST:
        target relation = centroid prediction
        h_last,l <- h_last,l + scale * s_target,l

    Exactly ONE relation direction is chosen for every TEST sample.
    There is no GT routing, confidence gate, abstention, probe, or classifier.

Evaluation
==========
Actual greedy model.generate():

    baseline_acc
    centroid_selector_acc
    centroid_guided_steer_acc
    oracle_steer_acc                (default sanity check)

and:
    baseline -> centroid-steer W2C / C2W / net
    selector confusion matrix
    selector-correct vs selector-wrong steering behavior
    per-relation metrics
    dx / dy / axis confidence

TRAIN / TEST separation
=======================
Default: relation-stratified 30/70 split.

TRAIN is used ONLY to fit the late Real-Gray causal directions.
TEST centroid selection is label-free.

The fixed centroid layers L24 / L19 are existing repo presets rather than
layers selected on this TEST split.

Dependencies
============
Run this script from the bxqm803/AdaptVis llava16 repository root. It imports:

    analyze_coco_centroid_generation_step1_v4.py
    eval_crossdataset_late_causal_qwen25_v4_auto_vgroot.py
    extract_two_object_relation_states.py

Examples
========

Qwen2.5-VL-3B:
CUDA_VISIBLE_DEVICES=0 python eval_coco_trainbest_singlehead_centroid_late_direction_qwen25_v2.py \
  --model qwen-3b \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --train-frac 0.30 \
  --template-filter real_correct_gray_wrong \
  --scale 1.0 \
  --output-dir output/coco_centroid_direction_qwen3b_v1 \
  --overwrite

Qwen2.5-VL-7B:
CUDA_VISIBLE_DEVICES=1 python eval_coco_trainbest_singlehead_centroid_late_direction_qwen25_v2.py \
  --model qwen-7b \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --train-frac 0.30 \
  --template-filter real_correct_gray_wrong \
  --scale 1.0 \
  --output-dir output/coco_centroid_direction_qwen7b_v1 \
  --overwrite

Smoke test:
CUDA_VISIBLE_DEVICES=0 python eval_coco_trainbest_singlehead_centroid_late_direction_qwen25_v2.py \
  --model qwen-3b \
  --max-samples 80 \
  --no-run-oracle \
  --output-dir output/coco_centroid_direction_smoke \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

# ---------------------------------------------------------------------------
# Reuse the user's repository implementations.
# ---------------------------------------------------------------------------
try:
    import analyze_coco_centroid_generation_step1_v4 as cent
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_centroid_generation_step1_v4.py. "
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )

try:
    import eval_crossdataset_late_causal_qwen25_v4_auto_vgroot as causal
except Exception as exc:
    raise SystemExit(
        "Could not import eval_crossdataset_late_causal_qwen25_v4_auto_vgroot.py. "
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )


RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12

MODEL_CONFIG = {
    "qwen-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "centroid_layer": 24,
        "actuator_layers": [32, 33, 34, 35],
    },
    "qwen-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "centroid_layer": 19,
        "actuator_layers": [25, 26, 27],
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIG),
    )

    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )

    p.add_argument(
        "--centroid-layer",
        default="auto",
        help=(
            "'auto' searches every decoder layer on TRAIN and selects the "
            "best SINGLE attention head by original+swap averaged-centroid "
            "accuracy. An integer restricts head selection to that layer."
        ),
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=[
            "real_correct_gray_wrong",
            "real_correct",
            "all",
        ],
        help=(
            "Requested TRAIN filter for late Real-Gray actuator templates. "
            "If one relation is missing, relaxes in the same order as the "
            "existing causal script."
        ),
    )

    p.add_argument(
        "--gray-value",
        type=int,
        default=128,
    )

    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )

    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--run-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also run GT-selected steering as an actuator sanity check. "
            "Oracle results are never used by the centroid selector."
        ),
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def cleanup() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(values: Iterable[Any]) -> float:
    vals = []

    for value in values:
        try:
            x = float(value)
        except Exception:
            continue

        if math.isfinite(x):
            vals.append(x)

    return float(np.mean(vals)) if vals else float("nan")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = list(rows)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def normalize_gt(value: Any) -> Optional[str]:
    return cent.normalize_relation(value)


def parse_generation(text: str) -> Optional[str]:
    return cent.normalize_relation(text)


def relation_counts(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    return {
        r: int(
            sum(
                item["relation"] == r
                for item in records
            )
        )
        for r in RELATIONS
    }


# =============================================================================
# Data: exact repo COCO records + authoritative standard prompts
# =============================================================================

def load_coco_records(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    two_obj = cent.import_two_object_module()

    raw_records, audit = two_obj.load_records(
        "coco_two",
        Path(args.data_root),
        args.max_samples,
    )

    prompt_path = Path(
        args.prompt_jsonl
    )

    prompt_rows = cent.load_standard_prompts(
        prompt_path
    )

    records: List[
        Dict[str, Any]
    ] = []

    for raw in raw_records:
        sid = int(raw.sid)

        if sid not in prompt_rows:
            raise RuntimeError(
                f"sid={sid} missing from {prompt_path}"
            )

        prompt = prompt_rows[sid]

        relation = normalize_gt(
            prompt["answer_raw"]
        )

        if relation not in REL_TO_ID:
            raise RuntimeError(
                f"sid={sid}: unsupported relation={relation!r}"
            )

        records.append({
            "sid": sid,
            "relation": relation,
            "subject": str(
                prompt["subject"]
            ),
            "reference": str(
                prompt["reference"]
            ),
            "question_text": str(
                prompt["question_text"]
            ),
            "raw_record": raw,
        })

    records.sort(
        key=lambda x: int(
            x["sid"]
        )
    )

    return records, audit


def stratified_split(
    records: Sequence[Mapping[str, Any]],
    train_frac: float,
    seed: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    rng = random.Random(
        int(seed)
    )

    train: List[
        Dict[str, Any]
    ] = []

    test: List[
        Dict[str, Any]
    ] = []

    for relation in RELATIONS:
        rows = [
            dict(r)
            for r in records
            if r["relation"] == relation
        ]

        rng.shuffle(rows)

        if len(rows) < 2:
            raise RuntimeError(
                f"Need >=2 samples for relation={relation}; "
                f"got {len(rows)}"
            )

        n_train = int(
            round(
                len(rows)
                * float(
                    train_frac
                )
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(rows) - 1,
            ),
        )

        train.extend(
            rows[:n_train]
        )

        test.extend(
            rows[n_train:]
        )

    train.sort(
        key=lambda x: int(
            x["sid"]
        )
    )

    test.sort(
        key=lambda x: int(
            x["sid"]
        )
    )

    return train, test


def open_record_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return cent.record_image(
        record["raw_record"]
    )


# =============================================================================
# Model loading
# =============================================================================

def resolve_dtype(
    name: str,
    spec: Any,
) -> torch.dtype:
    if name == "auto":
        return cent.resolve_dtype(
            spec.dtype_name
        )

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(
    args: argparse.Namespace,
):
    two_obj = cent.import_two_object_module()

    specs = cent.merged_model_specs(
        two_obj
    )

    if args.model not in specs:
        raise ValueError(
            f"{args.model!r} not available; "
            f"repo specs={sorted(specs)}"
        )

    spec = specs[
        args.model
    ]

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )

    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no "
            f"{spec.model_class}"
        )

    dtype = resolve_dtype(
        args.dtype,
        spec,
    )

    kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": (
            spec.trust_remote_code
        ),
        "device_map": {
            "": args.device
        },
        # Needed for full attention probabilities.
        "attn_implementation": "eager",
    }

    try:
        model = model_cls.from_pretrained(
            spec.repo_id,
            dtype=dtype,
            **kwargs,
        )

    except TypeError:
        model = model_cls.from_pretrained(
            spec.repo_id,
            torch_dtype=dtype,
            **kwargs,
        )

    model.eval()

    generation_config = getattr(
        model,
        "generation_config",
        None,
    )

    if generation_config is not None:
        for field in (
            "temperature",
            "top_p",
            "top_k",
        ):
            if hasattr(
                generation_config,
                field,
            ):
                setattr(
                    generation_config,
                    field,
                    None,
                )

    try:
        processor = (
            AutoProcessor
            .from_pretrained(
                spec.repo_id,
                trust_remote_code=(
                    spec.trust_remote_code
                ),
                use_fast=False,
            )
        )
    except TypeError:
        processor = (
            AutoProcessor
            .from_pretrained(
                spec.repo_id,
                trust_remote_code=(
                    spec.trust_remote_code
                ),
            )
        )

    cent.configure_processor(
        model,
        processor,
    )

    layers, decoder_path = (
        cent.resolve_decoder_layers(
            model
        )
    )

    return (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    )


# =============================================================================
# Exact standard COCO prompt batch
# =============================================================================

def make_batch(
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=record[
            "question_text"
        ],
        device=device,
    )


def generate_text(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    return cent.generate_text(
        model,
        processor,
        batch,
        args.max_new_tokens,
    )


# =============================================================================
# TRAIN: late Real-Gray causal directions
# =============================================================================

def capture_last_states(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    record: Mapping[str, Any],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    str,
    Optional[str],
    Dict[int, np.ndarray],
]:
    """
    Actual greedy generation while capturing the full-prompt last token at the
    known actuator layers.

    Uses causal.CaptureStates, but no middle Direction guide is involved.
    """
    device = torch.device(
        args.device
    )

    batch = make_batch(
        processor,
        image,
        record,
        device,
    )

    # Guide layers are empty, therefore object spans are not used by the hook.
    with causal.CaptureStates(
        layers,
        list(
            actuator_layers
        ),
        [],
        [],
        [],
    ) as capture:
        text = generate_text(
            model,
            processor,
            batch,
            args,
        )

        last_states = {
            int(layer): np.asarray(
                value,
                dtype=np.float32,
            )
            for layer, value
            in capture.last_states.items()
        }

    del batch

    missing = [
        int(layer)
        for layer in actuator_layers
        if int(layer)
        not in last_states
    ]

    if missing:
        raise RuntimeError(
            f"Missing captured actuator layers={missing}"
        )

    return (
        text,
        parse_generation(text),
        last_states,
    )


def make_gray(
    image: Image.Image,
    value: int,
) -> Image.Image:
    return causal.make_gray(
        image,
        value,
    )


def template_filter_sequence(
    requested: str,
) -> List[str]:
    # Same relaxation order as existing causal script.
    if requested == "real_correct_gray_wrong":
        return [
            "real_correct_gray_wrong",
            "real_correct",
            "all",
        ]

    if requested == "real_correct":
        return [
            "real_correct",
            "all",
        ]

    return ["all"]


def filter_allowed(
    row: Mapping[str, Any],
    mode: str,
) -> bool:
    if mode == "all":
        return True

    if mode == "real_correct":
        return bool(
            row["real_correct"]
        )

    if mode == "real_correct_gray_wrong":
        return (
            bool(
                row["real_correct"]
            )
            and not bool(
                row["gray_correct"]
            )
        )

    raise ValueError(
        mode
    )


def collect_train(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    train_records: Sequence[
        Mapping[str, Any]
    ],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, Dict[int, np.ndarray]],
]:
    rows: List[
        Dict[str, Any]
    ] = []

    cache: Dict[
        int,
        Dict[int, np.ndarray]
    ] = {}

    for record in tqdm(
        train_records,
        desc=f"TRAIN Real/Gray:{args.model}",
    ):
        real = None
        gray = None

        try:
            real = open_record_image(
                record
            )

            gray = make_gray(
                real,
                args.gray_value,
            )

            (
                real_text,
                real_pred,
                real_last,
            ) = capture_last_states(
                model,
                processor,
                layers,
                real,
                record,
                actuator_layers,
                args,
            )

            (
                gray_text,
                gray_pred,
                gray_last,
            ) = capture_last_states(
                model,
                processor,
                layers,
                gray,
                record,
                actuator_layers,
                args,
            )

            gt = record[
                "relation"
            ]

            rows.append({
                "sid": int(
                    record["sid"]
                ),
                "relation": gt,
                "real_pred": (
                    real_pred or ""
                ),
                "gray_pred": (
                    gray_pred or ""
                ),
                "real_correct": int(
                    real_pred == gt
                ),
                "gray_correct": int(
                    gray_pred == gt
                ),
                "real_text": real_text,
                "gray_text": gray_text,
            })

            cache[
                int(
                    record["sid"]
                )
            ] = {
                int(layer): (
                    real_last[
                        int(layer)
                    ]
                    - gray_last[
                        int(layer)
                    ]
                ).astype(
                    np.float32
                )
                for layer
                in actuator_layers
            }

        except Exception as exc:
            tqdm.write(
                f"[TRAIN ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    return rows, cache


def fit_templates(
    train_records: Sequence[
        Mapping[str, Any]
    ],
    train_rows: Sequence[
        Mapping[str, Any]
    ],
    cache: Mapping[
        int,
        Mapping[
            int,
            np.ndarray,
        ],
    ],
    actuator_layers: Sequence[int],
    requested_filter: str,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    str,
    Dict[str, int],
]:
    row_map = {
        int(row["sid"]): row
        for row in train_rows
    }

    rec_map = {
        int(row["sid"]): row
        for row in train_records
    }

    for mode in template_filter_sequence(
        requested_filter
    ):
        bags = {
            int(layer): {
                relation: []
                for relation
                in RELATIONS
            }
            for layer
            in actuator_layers
        }

        used_relation_count = Counter()

        for sid, layer_values in (
            cache.items()
        ):
            sid = int(sid)

            if (
                sid not in row_map
                or sid not in rec_map
            ):
                continue

            row = row_map[
                sid
            ]

            if not filter_allowed(
                row,
                mode,
            ):
                continue

            relation = rec_map[
                sid
            ]["relation"]

            used_relation_count[
                relation
            ] += 1

            for layer in (
                actuator_layers
            ):
                bags[
                    int(layer)
                ][
                    relation
                ].append(
                    np.asarray(
                        layer_values[
                            int(layer)
                        ],
                        dtype=np.float32,
                    )
                )

        missing = [
            (
                int(layer),
                relation,
            )
            for layer
            in actuator_layers
            for relation
            in RELATIONS
            if not bags[
                int(layer)
            ][relation]
        ]

        if missing:
            print(
                f"[template] filter={mode} "
                f"missing={missing[:8]} -> relax"
            )
            continue

        templates: Dict[
            int,
            Dict[str, Any]
        ] = {}

        for layer in (
            actuator_layers
        ):
            layer = int(
                layer
            )

            relation_mean = {
                relation: (
                    np.stack(
                        bags[
                            layer
                        ][
                            relation
                        ],
                        axis=0,
                    )
                    .mean(
                        axis=0
                    )
                    .astype(
                        np.float32
                    )
                )
                for relation
                in RELATIONS
            }

            # Balanced common Real-Gray component.
            global_mean = (
                np.stack(
                    [
                        relation_mean[
                            relation
                        ]
                        for relation
                        in RELATIONS
                    ],
                    axis=0,
                )
                .mean(
                    axis=0
                )
                .astype(
                    np.float32
                )
            )

            shared = {
                relation: (
                    relation_mean[
                        relation
                    ]
                    - global_mean
                ).astype(
                    np.float32
                )
                for relation
                in RELATIONS
            }

            templates[
                layer
            ] = {
                "global": (
                    global_mean
                ),
                "relation_mean": (
                    relation_mean
                ),
                "shared": shared,
            }

        counts = {
            relation: int(
                used_relation_count[
                    relation
                ]
            )
            for relation
            in RELATIONS
        }

        print(
            f"[template] using filter={mode} "
            f"counts={counts}"
        )

        return (
            templates,
            mode,
            counts,
        )

    raise RuntimeError(
        "Could not fit all four late actuator directions."
    )


# =============================================================================
# TRAIN-SELECTED SINGLE-HEAD ATTENTION CENTROID READER
# =============================================================================

def resolve_attention_tuple(
    outputs: Any,
) -> Sequence[torch.Tensor]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(
            getattr(outputs, "language_model_output", None),
            "attentions",
            None,
        ),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "attentions",
            None,
        ),
    ]

    for value in candidates:
        if (
            isinstance(value, (tuple, list))
            and len(value) > 0
        ):
            return value

    raise RuntimeError(
        "Forward did not return decoder attentions. "
        "Load with attn_implementation='eager'."
    )


def extract_object_attention_centroids(
    model: Any,
    processor: Any,
    image: Image.Image,
    *,
    question_text: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Extract the repo's per-head subject/reference attention centroids.

    Returns
    -------
    centroids:
        [selected_layer, head, 2 objects, 2 xy]

    visual_mass:
        [selected_layer, head, 2 objects]

    No generation and no GT are involved.
    """
    device = torch.device(args.device)

    batch = cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
        device=device,
    )

    input_ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )
    input_length = len(input_ids)

    subject_span, reference_span = cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )

    # Match repository object-query convention:
    # use the final token in each object phrase.
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])

    visual_indices = cent.resolve_visual_indices(
        model,
        processor,
        batch,
        input_ids,
    )

    coords = cent.visual_coordinates(
        model,
        batch,
        len(visual_indices),
        batch["input_ids"].device,
    )

    if coords is None:
        raise RuntimeError(
            f"Could not construct coordinates for "
            f"{len(visual_indices)} visual tokens."
        )

    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

    attentions = resolve_attention_tuple(outputs)

    centers_by_layer: List[np.ndarray] = []
    mass_by_layer: List[np.ndarray] = []
    n_heads: Optional[int] = None

    for layer in selected_layers:
        layer = int(layer)

        if not (0 <= layer < len(attentions)):
            raise RuntimeError(
                f"L{layer} unavailable; returned attentions={len(attentions)}"
            )

        prompt_tensor = cent.normalize_attention_tensor(
            attentions[layer],
            expected_query_length=input_length,
        )

        if n_heads is None:
            n_heads = int(prompt_tensor.shape[0])
        elif int(prompt_tensor.shape[0]) != n_heads:
            raise RuntimeError(
                f"Head count changed at L{layer}: "
                f"{prompt_tensor.shape[0]} vs {n_heads}"
            )

        rows = prompt_tensor[
            :,
            [subject_index, reference_index],
            :,
        ]

        metrics = cent.query_attention_metrics(
            rows,
            visual_indices,
            coords,
            subject_index,
            reference_index,
        )

        centers_by_layer.append(
            metrics["centroids"][:, :2, :]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        mass_by_layer.append(
            metrics["visual_mass"][:, :2]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    result = {
        "centroids": np.stack(
            centers_by_layer,
            axis=0,
        ),
        "visual_mass": np.stack(
            mass_by_layer,
            axis=0,
        ),
        "n_heads": int(n_heads or 0),
        "n_visual_tokens": int(len(visual_indices)),
    }

    del outputs
    del attentions
    del batch

    return result


def extract_original_swap_centroids(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Exact repository original/swap semantic alignment:

      original role order = [A, B]
      swapped role order  = [B, A]
      swapped aligned     = swapped[:, :, [1,0], :]

      average_centroids =
          0.5 * (original + swapped_aligned)
    """
    original = extract_object_attention_centroids(
        model,
        processor,
        image,
        question_text=record["question_text"],
        subject=record["subject"],
        reference=record["reference"],
        selected_layers=selected_layers,
        args=args,
    )

    swapped_question = cent.build_swapped_question(
        record["subject"],
        record["reference"],
    )

    swapped = extract_object_attention_centroids(
        model,
        processor,
        image,
        question_text=swapped_question,
        subject=record["reference"],
        reference=record["subject"],
        selected_layers=selected_layers,
        args=args,
    )

    if (
        original["centroids"].shape
        != swapped["centroids"].shape
    ):
        raise RuntimeError(
            "Original/swap centroid shape mismatch: "
            f"{original['centroids'].shape} vs "
            f"{swapped['centroids'].shape}"
        )

    swapped_centroids_aligned = (
        swapped["centroids"][
            :,
            :,
            [1, 0],
            :,
        ]
    )

    swapped_mass_aligned = (
        swapped["visual_mass"][
            :,
            :,
            [1, 0],
        ]
    )

    average_centroids = (
        0.5
        * (
            original["centroids"]
            + swapped_centroids_aligned
        )
    ).astype(np.float32)

    original_pred, original_conf = (
        cent.relation_codes_from_centroids(
            original["centroids"]
        )
    )

    swapped_pred, swapped_conf = (
        cent.relation_codes_from_centroids(
            swapped_centroids_aligned
        )
    )

    average_pred, average_conf = (
        cent.relation_codes_from_centroids(
            average_centroids
        )
    )

    return {
        "original_centroids": original["centroids"],
        "swapped_centroids_aligned": swapped_centroids_aligned,
        "average_centroids": average_centroids,

        "original_prediction": original_pred,
        "swapped_prediction": swapped_pred,
        "average_prediction": average_pred,

        "original_axis_confidence": original_conf,
        "swapped_axis_confidence": swapped_conf,
        "average_axis_confidence": average_conf,

        "original_visual_mass": original["visual_mass"],
        "swapped_visual_mass_aligned": swapped_mass_aligned,

        "n_heads": original["n_heads"],
        "n_visual_tokens": original["n_visual_tokens"],
    }


def fit_centroid_head_on_train(
    model: Any,
    processor: Any,
    train_records: Sequence[Mapping[str, Any]],
    candidate_layers: Sequence[int],
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[int, int, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Select one (layer, head) using TRAIN GT only.

    Primary selection score:
        original+swap ALIGNED AVERAGE centroid accuracy.

    TEST is never seen here.
    """
    candidate_layers = [
        int(x)
        for x in candidate_layers
    ]

    correct_avg = None
    correct_orig = None
    correct_swap = None
    consistency_sum = None
    axis_conf_sum = None
    relation_correct = None
    relation_count = Counter()
    n_success = 0
    errors: List[Dict[str, Any]] = []

    for record in tqdm(
        train_records,
        desc=f"TRAIN centroid head search:{args.model}",
    ):
        image = None

        try:
            image = open_record_image(record)

            result = extract_original_swap_centroids(
                model,
                processor,
                image,
                record,
                candidate_layers,
                args,
            )

            avg_pred = np.asarray(
                result["average_prediction"],
                dtype=np.int64,
            )
            orig_pred = np.asarray(
                result["original_prediction"],
                dtype=np.int64,
            )
            swap_pred = np.asarray(
                result["swapped_prediction"],
                dtype=np.int64,
            )

            if correct_avg is None:
                shape = avg_pred.shape

                if len(shape) != 2:
                    raise RuntimeError(
                        f"Expected [layer,head] centroid predictions; "
                        f"got {shape}"
                    )

                correct_avg = np.zeros(
                    shape,
                    dtype=np.int64,
                )
                correct_orig = np.zeros(
                    shape,
                    dtype=np.int64,
                )
                correct_swap = np.zeros(
                    shape,
                    dtype=np.int64,
                )
                consistency_sum = np.zeros(
                    shape,
                    dtype=np.int64,
                )
                axis_conf_sum = np.zeros(
                    shape,
                    dtype=np.float64,
                )
                relation_correct = {
                    relation: np.zeros(
                        shape,
                        dtype=np.int64,
                    )
                    for relation in RELATIONS
                }

            assert correct_orig is not None
            assert correct_swap is not None
            assert consistency_sum is not None
            assert axis_conf_sum is not None
            assert relation_correct is not None

            gt = record["relation"]
            gt_code = REL_TO_ID[gt]

            correct_avg += (
                avg_pred == gt_code
            ).astype(np.int64)

            correct_orig += (
                orig_pred == gt_code
            ).astype(np.int64)

            correct_swap += (
                swap_pred == gt_code
            ).astype(np.int64)

            consistency_sum += (
                orig_pred == swap_pred
            ).astype(np.int64)

            axis_conf_sum += np.asarray(
                result["average_axis_confidence"],
                dtype=np.float64,
            )

            relation_correct[gt] += (
                avg_pred == gt_code
            ).astype(np.int64)

            relation_count[gt] += 1
            n_success += 1

        except Exception as exc:
            tqdm.write(
                f"[CENTROID TRAIN ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append({
                "sid": int(record["sid"]),
                "relation": record["relation"],
                "error": f"{type(exc).__name__}: {exc}",
            })

        finally:
            if image is not None:
                image.close()

            cleanup()

    if (
        n_success == 0
        or correct_avg is None
        or correct_orig is None
        or correct_swap is None
        or consistency_sum is None
        or axis_conf_sum is None
        or relation_correct is None
    ):
        raise RuntimeError(
            "No successful TRAIN centroid samples."
        )

    avg_acc = (
        correct_avg.astype(np.float64)
        / float(n_success)
    )

    orig_acc = (
        correct_orig.astype(np.float64)
        / float(n_success)
    )

    swap_acc = (
        correct_swap.astype(np.float64)
        / float(n_success)
    )

    consistency = (
        consistency_sum.astype(np.float64)
        / float(n_success)
    )

    axis_conf = (
        axis_conf_sum
        / float(n_success)
    )

    ranking: List[Dict[str, Any]] = []

    for layer_pos, layer in enumerate(candidate_layers):
        for head in range(avg_acc.shape[1]):
            row = {
                "layer": int(layer),
                "head": int(head),

                "train_average_accuracy": float(
                    avg_acc[layer_pos, head]
                ),

                "train_original_accuracy": float(
                    orig_acc[layer_pos, head]
                ),

                "train_swapped_aligned_accuracy": float(
                    swap_acc[layer_pos, head]
                ),

                "train_original_swap_consistency": float(
                    consistency[layer_pos, head]
                ),

                "train_average_axis_confidence": float(
                    axis_conf[layer_pos, head]
                ),
            }

            for relation in RELATIONS:
                denom = max(
                    1,
                    int(relation_count[relation]),
                )

                row[
                    f"train_{relation}_accuracy"
                ] = float(
                    relation_correct[
                        relation
                    ][
                        layer_pos,
                        head,
                    ]
                    / denom
                )

            ranking.append(row)

    # Primary key exactly matches the desired repository-style
    # original+swap averaged-centroid accuracy.
    # Deterministic tie-breaks prefer greater original/swap consistency,
    # then greater axis confidence, then earlier layer/head.
    ranking.sort(
        key=lambda row: (
            -float(
                row[
                    "train_average_accuracy"
                ]
            ),
            -float(
                row[
                    "train_original_swap_consistency"
                ]
            ),
            -float(
                row[
                    "train_average_axis_confidence"
                ]
            ),
            int(row["layer"]),
            int(row["head"]),
        )
    )

    best = dict(ranking[0])

    best["N_train_centroid"] = int(n_success)
    best["N_train_centroid_errors"] = int(len(errors))
    best["selection_uses_test_gt"] = False
    best["selection_metric"] = (
        "TRAIN original+swap aligned average-centroid accuracy"
    )

    write_csv(
        outdir
        / "train_centroid_head_ranking.csv",
        ranking,
    )

    if errors:
        write_csv(
            outdir
            / "train_centroid_errors.csv",
            errors,
        )

    print("\n" + "=" * 150)
    print("TRAIN-ONLY SINGLE-HEAD CENTROID SELECTION")
    print("=" * 150)

    print(
        f"candidate layers={candidate_layers}"
    )

    print(
        f"successful TRAIN centroid samples="
        f"{n_success}/{len(train_records)}"
    )

    print(
        f"SELECTED: L{best['layer']:02d} H{best['head']:02d} | "
        f"avg={best['train_average_accuracy']:.4f} | "
        f"orig={best['train_original_accuracy']:.4f} | "
        f"swap={best['train_swapped_aligned_accuracy']:.4f} | "
        f"consistency={best['train_original_swap_consistency']:.4f}"
    )

    print(
        "Per relation TRAIN avg-centroid: "
        + " | ".join(
            f"{relation}="
            f"{best[f'train_{relation}_accuracy']:.4f}"
            for relation in RELATIONS
        )
    )

    print("\nTop 10 TRAIN heads:")

    for row in ranking[:10]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} | "
            f"avg={row['train_average_accuracy']:.4f} | "
            f"orig={row['train_original_accuracy']:.4f} | "
            f"swap={row['train_swapped_aligned_accuracy']:.4f} | "
            f"cons={row['train_original_swap_consistency']:.4f}"
        )

    print("=" * 150)

    return (
        int(best["layer"]),
        int(best["head"]),
        best,
        ranking,
    )


def centroid_select(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    selector_layer: int,
    selector_head: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    TEST reader.

    Uses the SINGLE (layer, head) selected on TRAIN.
    Runs original and swapped prompts, aligns the swapped object roles,
    averages the two centroids, and converts the averaged displacement to one
    of left/right/above/below.

    No TEST GT is accessed in this function.
    """
    result = extract_original_swap_centroids(
        model,
        processor,
        image,
        record,
        [int(selector_layer)],
        args,
    )

    head = int(selector_head)

    if not (
        0 <= head
        < int(result["n_heads"])
    ):
        raise RuntimeError(
            f"Selected H{head} outside available "
            f"0..{int(result['n_heads'])-1}"
        )

    layer_pos = 0

    avg_centroid = np.asarray(
        result["average_centroids"][
            layer_pos,
            head,
        ],
        dtype=np.float32,
    )

    original_centroid = np.asarray(
        result["original_centroids"][
            layer_pos,
            head,
        ],
        dtype=np.float32,
    )

    swapped_centroid = np.asarray(
        result["swapped_centroids_aligned"][
            layer_pos,
            head,
        ],
        dtype=np.float32,
    )

    avg_code = int(
        result["average_prediction"][
            layer_pos,
            head,
        ]
    )

    orig_code = int(
        result["original_prediction"][
            layer_pos,
            head,
        ]
    )

    swap_code = int(
        result["swapped_prediction"][
            layer_pos,
            head,
        ]
    )

    prediction = cent.to_relation_name(
        avg_code
    )

    original_prediction = (
        cent.to_relation_name(
            orig_code
        )
    )

    swapped_prediction = (
        cent.to_relation_name(
            swap_code
        )
    )

    dx = float(
        avg_centroid[0, 0]
        - avg_centroid[1, 0]
    )

    dy = float(
        avg_centroid[0, 1]
        - avg_centroid[1, 1]
    )

    original_mass = np.asarray(
        result["original_visual_mass"][
            layer_pos,
            head,
        ],
        dtype=np.float32,
    )

    swapped_mass = np.asarray(
        result["swapped_visual_mass_aligned"][
            layer_pos,
            head,
        ],
        dtype=np.float32,
    )

    avg_mass = (
        0.5
        * (
            original_mass
            + swapped_mass
        )
    )

    return {
        "prediction": prediction,
        "original_prediction": original_prediction,
        "swapped_prediction": swapped_prediction,

        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),

        "axis_confidence": float(
            result["average_axis_confidence"][
                layer_pos,
                head,
            ]
        ),

        "original_axis_confidence": float(
            result["original_axis_confidence"][
                layer_pos,
                head,
            ]
        ),

        "swapped_axis_confidence": float(
            result["swapped_axis_confidence"][
                layer_pos,
                head,
            ]
        ),

        "subject_x": float(
            avg_centroid[0, 0]
        ),
        "subject_y": float(
            avg_centroid[0, 1]
        ),
        "reference_x": float(
            avg_centroid[1, 0]
        ),
        "reference_y": float(
            avg_centroid[1, 1]
        ),

        "original_subject_x": float(
            original_centroid[0, 0]
        ),
        "original_subject_y": float(
            original_centroid[0, 1]
        ),
        "original_reference_x": float(
            original_centroid[1, 0]
        ),
        "original_reference_y": float(
            original_centroid[1, 1]
        ),

        "swapped_aligned_subject_x": float(
            swapped_centroid[0, 0]
        ),
        "swapped_aligned_subject_y": float(
            swapped_centroid[0, 1]
        ),
        "swapped_aligned_reference_x": float(
            swapped_centroid[1, 0]
        ),
        "swapped_aligned_reference_y": float(
            swapped_centroid[1, 1]
        ),

        "subject_visual_mass": float(
            avg_mass[0]
        ),
        "reference_visual_mass": float(
            avg_mass[1]
        ),

        "original_swap_relation_consistent": int(
            orig_code == swap_code
        ),

        "n_visual_tokens": int(
            result["n_visual_tokens"]
        ),
        "n_heads": int(
            result["n_heads"]
        ),
    }


# =============================================================================
# TEST WRITE: centroid-selected / oracle late steering
# =============================================================================

def steer_generate(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    record: Mapping[str, Any],
    templates: Mapping[int, Any],
    actuator_layers: Sequence[int],
    target_relation: str,
    args: argparse.Namespace,
) -> Tuple[
    str,
    Optional[str],
]:
    batch = make_batch(
        processor,
        image,
        record,
        torch.device(
            args.device
        ),
    )

    with causal.SteerLast(
        layers,
        templates,
        list(
            actuator_layers
        ),
        target_relation,
        args.scale,
        "add",
        None,
    ):
        text = generate_text(
            model,
            processor,
            batch,
            args,
        )

    del batch

    return (
        text,
        parse_generation(
            text
        ),
    )


# =============================================================================
# TEST loop
# =============================================================================

def evaluate_test(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    test_records: Sequence[
        Mapping[str, Any]
    ],
    templates: Mapping[int, Any],
    actuator_layers: Sequence[int],
    selector_layer: int,
    selector_head: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    for record in tqdm(
        test_records,
        desc=(
            f"TEST centroid-select+steer:"
            f"{args.model}"
        ),
    ):
        image = None

        try:
            image = open_record_image(
                record
            )

            gt = record[
                "relation"
            ]

            # -------------------------------------------------------------
            # Baseline actual generation.
            # -------------------------------------------------------------
            base_batch = make_batch(
                processor,
                image,
                record,
                torch.device(
                    args.device
                ),
            )

            baseline_text = (
                generate_text(
                    model,
                    processor,
                    base_batch,
                    args,
                )
            )

            del base_batch

            baseline_pred = (
                parse_generation(
                    baseline_text
                )
            )

            # -------------------------------------------------------------
            # Label-free centroid selector.
            # -------------------------------------------------------------
            centroid = centroid_select(
                model,
                processor,
                image,
                record,
                selector_layer,
                selector_head,
                args,
            )

            target = centroid[
                "prediction"
            ]

            if target not in REL_TO_ID:
                raise RuntimeError(
                    f"Invalid centroid target={target!r}"
                )

            # -------------------------------------------------------------
            # Use exactly one selected late relation vector.
            # -------------------------------------------------------------
            edit_text, edit_pred = (
                steer_generate(
                    model,
                    processor,
                    layers,
                    image,
                    record,
                    templates,
                    actuator_layers,
                    target,
                    args,
                )
            )

            # -------------------------------------------------------------
            # Optional oracle control: same actuator, only routing differs.
            # -------------------------------------------------------------
            oracle_text = ""
            oracle_pred = None

            if args.run_oracle:
                (
                    oracle_text,
                    oracle_pred,
                ) = steer_generate(
                    model,
                    processor,
                    layers,
                    image,
                    record,
                    templates,
                    actuator_layers,
                    gt,
                    args,
                )

            base_ok = (
                baseline_pred
                == gt
            )

            selector_ok = (
                target
                == gt
            )

            edit_ok = (
                edit_pred
                == gt
            )

            oracle_ok = (
                oracle_pred
                == gt
                if args.run_oracle
                else False
            )

            rows.append({
                "sid": int(
                    record["sid"]
                ),
                "relation": gt,
                "subject": record[
                    "subject"
                ],
                "reference": record[
                    "reference"
                ],

                "centroid_layer": int(
                    selector_layer
                ),
                "centroid_head": int(
                    selector_head
                ),

                "centroid_pred": target,
                "centroid_original_pred": centroid[
                    "original_prediction"
                ],
                "centroid_swapped_pred": centroid[
                    "swapped_prediction"
                ],
                "centroid_original_correct": int(
                    centroid["original_prediction"] == gt
                ),
                "centroid_swapped_correct": int(
                    centroid["swapped_prediction"] == gt
                ),
                "centroid_original_swap_consistent": int(
                    centroid["original_swap_relation_consistent"]
                ),
                "selector_correct": int(
                    selector_ok
                ),

                "dx": centroid[
                    "dx"
                ],
                "dy": centroid[
                    "dy"
                ],
                "abs_dx": centroid[
                    "abs_dx"
                ],
                "abs_dy": centroid[
                    "abs_dy"
                ],
                "axis_confidence": (
                    centroid[
                        "axis_confidence"
                    ]
                ),

                "subject_x": centroid[
                    "subject_x"
                ],
                "subject_y": centroid[
                    "subject_y"
                ],
                "reference_x": centroid[
                    "reference_x"
                ],
                "reference_y": centroid[
                    "reference_y"
                ],

                "subject_visual_mass": (
                    centroid[
                        "subject_visual_mass"
                    ]
                ),

                "reference_visual_mass": (
                    centroid[
                        "reference_visual_mass"
                    ]
                ),

                "baseline_pred": (
                    baseline_pred
                    or ""
                ),
                "baseline_correct": int(
                    base_ok
                ),
                "baseline_text": (
                    baseline_text
                ),

                "steer_pred": (
                    edit_pred or ""
                ),
                "steer_correct": int(
                    edit_ok
                ),
                "steer_text": edit_text,

                "W2C": int(
                    (not base_ok)
                    and edit_ok
                ),
                "C2W": int(
                    base_ok
                    and (not edit_ok)
                ),
                "changed": int(
                    baseline_pred
                    != edit_pred
                ),

                "oracle_pred": (
                    oracle_pred or ""
                ),
                "oracle_correct": (
                    int(
                        oracle_ok
                    )
                    if args.run_oracle
                    else ""
                ),
                "oracle_text": (
                    oracle_text
                ),
            })

        except Exception as exc:
            tqdm.write(
                f"[TEST ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

            rows.append({
                "sid": int(
                    record["sid"]
                ),
                "relation": (
                    record[
                        "relation"
                    ]
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "traceback_tail": " | ".join(
                    traceback.format_exc()
                    .splitlines()[
                        -8:
                    ]
                ),
            })

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


# =============================================================================
# Summaries
# =============================================================================

def confusion_counts(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> List[Dict[str, Any]]:
    out = []

    for gt in RELATIONS:
        for pred in RELATIONS:
            out.append({
                "gt": gt,
                "pred": pred,
                "count": int(
                    sum(
                        row.get(
                            "relation"
                        )
                        == gt
                        and row.get(
                            "centroid_pred"
                        )
                        == pred
                        for row in rows
                    )
                ),
            })

    return out


def summarize(
    rows: Sequence[
        Mapping[str, Any]
    ],
    args: argparse.Namespace,
    selector_layer: int,
    selector_head: int,
    actuator_layers: Sequence[int],
    chosen_filter: str,
    template_counts: Mapping[str, int],
) -> Dict[str, Any]:
    good = [
        row
        for row in rows
        if "error" not in row
    ]

    N = len(good)

    if N == 0:
        raise RuntimeError(
            "No successful TEST samples."
        )

    base_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in good
    )

    selector_acc = safe_mean(
        row[
            "selector_correct"
        ]
        for row in good
    )

    steer_acc = safe_mean(
        row[
            "steer_correct"
        ]
        for row in good
    )

    oracle_acc = (
        safe_mean(
            row[
                "oracle_correct"
            ]
            for row in good
        )
        if args.run_oracle
        else float("nan")
    )

    W2C = int(
        sum(
            int(
                row["W2C"]
            )
            for row in good
        )
    )

    C2W = int(
        sum(
            int(
                row["C2W"]
            )
            for row in good
        )
    )

    selector_correct_rows = [
        row
        for row in good
        if int(
            row[
                "selector_correct"
            ]
        )
        == 1
    ]

    selector_wrong_rows = [
        row
        for row in good
        if int(
            row[
                "selector_correct"
            ]
        )
        == 0
    ]

    summary: Dict[str, Any] = {
        "model": args.model,
        "model_id": (
            MODEL_CONFIG[
                args.model
            ][
                "model_id"
            ]
        ),
        "N_test_success": N,
        "N_test_errors": (
            len(rows)
            - N
        ),

        "train_frac": (
            args.train_frac
        ),

        "centroid_layer": int(
            selector_layer
        ),
        "centroid_head": int(
            selector_head
        ),

        "centroid_original_same_head_acc": safe_mean(
            row["centroid_original_correct"]
            for row in good
        ),
        "centroid_swapped_same_head_acc": safe_mean(
            row["centroid_swapped_correct"]
            for row in good
        ),
        "centroid_original_swap_consistency": safe_mean(
            row["centroid_original_swap_consistent"]
            for row in good
        ),

        "actuator_layers": (
            ",".join(
                str(int(x))
                for x in actuator_layers
            )
        ),

        "template_filter_requested": (
            args.template_filter
        ),
        "template_filter_used": (
            chosen_filter
        ),

        "template_left_N": int(
            template_counts[
                "left"
            ]
        ),
        "template_right_N": int(
            template_counts[
                "right"
            ]
        ),
        "template_above_N": int(
            template_counts[
                "above"
            ]
        ),
        "template_below_N": int(
            template_counts[
                "below"
            ]
        ),

        "baseline_acc": base_acc,
        "centroid_selector_acc": (
            selector_acc
        ),
        "centroid_steer_acc": (
            steer_acc
        ),
        "gain_vs_baseline": (
            steer_acc
            - base_acc
        ),

        "oracle_steer_acc": (
            oracle_acc
        ),

        "W2C": W2C,
        "C2W": C2W,
        "net": W2C - C2W,

        "changed": int(
            sum(
                int(
                    row[
                        "changed"
                    ]
                )
                for row in good
            )
        ),

        "selector_correct_N": (
            len(
                selector_correct_rows
            )
        ),

        "selector_wrong_N": (
            len(
                selector_wrong_rows
            )
        ),

        "baseline_acc_when_selector_correct": (
            safe_mean(
                row[
                    "baseline_correct"
                ]
                for row
                in selector_correct_rows
            )
        ),

        "steer_acc_when_selector_correct": (
            safe_mean(
                row[
                    "steer_correct"
                ]
                for row
                in selector_correct_rows
            )
        ),

        "baseline_acc_when_selector_wrong": (
            safe_mean(
                row[
                    "baseline_correct"
                ]
                for row
                in selector_wrong_rows
            )
        ),

        "steer_acc_when_selector_wrong": (
            safe_mean(
                row[
                    "steer_correct"
                ]
                for row
                in selector_wrong_rows
            )
        ),

        "axis_confidence_mean": (
            safe_mean(
                row[
                    "axis_confidence"
                ]
                for row in good
            )
        ),

        "scale": args.scale,
        "gray_value": (
            args.gray_value
        ),
    }

    return summary


def per_relation_summary(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> List[Dict[str, Any]]:
    good = [
        row
        for row in rows
        if "error" not in row
    ]

    result = []

    for relation in RELATIONS:
        subset = [
            row
            for row in good
            if row[
                "relation"
            ]
            == relation
        ]

        result.append({
            "relation": relation,
            "N": len(subset),

            "baseline_acc": (
                safe_mean(
                    row[
                        "baseline_correct"
                    ]
                    for row in subset
                )
            ),

            "selector_acc": (
                safe_mean(
                    row[
                        "selector_correct"
                    ]
                    for row in subset
                )
            ),

            "steer_acc": (
                safe_mean(
                    row[
                        "steer_correct"
                    ]
                    for row in subset
                )
            ),

            "W2C": int(
                sum(
                    int(
                        row[
                            "W2C"
                        ]
                    )
                    for row in subset
                )
            ),

            "C2W": int(
                sum(
                    int(
                        row[
                            "C2W"
                        ]
                    )
                    for row in subset
                )
            ),

            "axis_confidence": (
                safe_mean(
                    row[
                        "axis_confidence"
                    ]
                    for row in subset
                )
            ),
        })

    return result


def print_summary(
    summary: Mapping[str, Any],
    relation_rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    print(
        "\n"
        + "=" * 150
    )

    print(
        "COCO TRAIN-BEST SINGLE-HEAD CENTROID -> "
        "LATE DIRECTION GENERATION"
    )

    print(
        "=" * 150
    )

    print(
        f"model={summary['model']} | "
        f"N={summary['N_test_success']} | "
        f"centroid=L{summary['centroid_layer']}"
        f"H{summary['centroid_head']} | "
        f"actuator={summary['actuator_layers']}"
    )

    print(
        f"template filter="
        f"{summary['template_filter_used']} | "
        f"L/R/A/B="
        f"{summary['template_left_N']}/"
        f"{summary['template_right_N']}/"
        f"{summary['template_above_N']}/"
        f"{summary['template_below_N']}"
    )

    print(
        f"baseline generation       : "
        f"{summary['baseline_acc']:.4f}"
    )

    print(
        f"centroid selector avg     : "
        f"{summary['centroid_selector_acc']:.4f}"
    )

    print(
        f"  same head original      : "
        f"{summary['centroid_original_same_head_acc']:.4f}"
    )

    print(
        f"  same head swapped       : "
        f"{summary['centroid_swapped_same_head_acc']:.4f}"
    )

    print(
        f"  orig/swap consistency   : "
        f"{summary['centroid_original_swap_consistency']:.4f}"
    )

    print(
        f"centroid-selected steering: "
        f"{summary['centroid_steer_acc']:.4f} "
        f"({summary['gain_vs_baseline']:+.4f})"
    )

    if math.isfinite(
        float(
            summary[
                "oracle_steer_acc"
            ]
        )
    ):
        print(
            f"oracle steering control   : "
            f"{summary['oracle_steer_acc']:.4f}"
        )

    print(
        f"baseline -> centroid steer: "
        f"W2C={summary['W2C']} "
        f"C2W={summary['C2W']} "
        f"net={summary['net']:+d} "
        f"changed={summary['changed']}"
    )

    print(
        f"selector CORRECT n="
        f"{summary['selector_correct_N']}: "
        f"generation "
        f"{summary['baseline_acc_when_selector_correct']:.4f}"
        f" -> "
        f"{summary['steer_acc_when_selector_correct']:.4f}"
    )

    print(
        f"selector WRONG   n="
        f"{summary['selector_wrong_N']}: "
        f"generation "
        f"{summary['baseline_acc_when_selector_wrong']:.4f}"
        f" -> "
        f"{summary['steer_acc_when_selector_wrong']:.4f}"
    )

    print(
        "\nPer relation:"
    )

    for row in relation_rows:
        print(
            f"  {row['relation']:>6s} "
            f"N={row['N']:3d} | "
            f"base={row['baseline_acc']:.4f} | "
            f"selector={row['selector_acc']:.4f} | "
            f"steer={row['steer_acc']:.4f} | "
            f"W2C/C2W={row['W2C']}/{row['C2W']}"
        )

    print(
        "=" * 150
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if (
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if not (
        0.0
        < args.train_frac
        < 1.0
    ):
        raise ValueError(
            "--train-frac must be in (0,1)."
        )

    if (
        args.max_new_tokens
        < 1
    ):
        raise ValueError(
            "--max-new-tokens must be >=1."
        )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    outdir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and outdir.exists()
    ):
        shutil.rmtree(
            outdir
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, audit = (
        load_coco_records(
            args
        )
    )

    train_records, test_records = (
        stratified_split(
            records,
            args.train_frac,
            args.seed,
        )
    )

    config = MODEL_CONFIG[
        args.model
    ]

    actuator_layers = list(
        config[
            "actuator_layers"
        ]
    )

    print(
        "\n"
        + "=" * 150
    )

    print(
        "COCO TRAIN-BEST SINGLE-HEAD CENTROID -> LATE DIRECTION"
    )

    print(
        "=" * 150
    )

    print(
        f"model={args.model} "
        f"({config['model_id']})"
    )

    print(
        f"ALL={len(records)} "
        f"{relation_counts(records)}"
    )

    print(
        f"TRAIN={len(train_records)} "
        f"{relation_counts(train_records)}"
    )

    print(
        f"TEST ={len(test_records)} "
        f"{relation_counts(test_records)}"
    )

    print(
        "READ: TRAIN-best SINGLE attention head using "
        "original+swap aligned average centroids"
    )

    print(
        f"WRITE: last-token Real-Gray shared directions "
        f"at {actuator_layers}"
    )

    print(
        "TEST target = centroid prediction only; "
        "GT is not used for routing."
    )

    print(
        "=" * 150
    )

    (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    ) = load_model_and_processor(
        args
    )

    n_layers = len(
        layers
    )

    bad_actuator_layers = [
        layer
        for layer in actuator_layers
        if not (
            0 <= int(layer)
            < n_layers
        )
    ]

    if bad_actuator_layers:
        raise RuntimeError(
            f"Actuator layers outside decoder "
            f"0..{n_layers-1}: {bad_actuator_layers}"
        )

    if (
        str(args.centroid_layer)
        .strip()
        .lower()
        == "auto"
    ):
        candidate_centroid_layers = list(
            range(n_layers)
        )
    else:
        requested_centroid_layer = int(
            args.centroid_layer
        )

        if not (
            0 <= requested_centroid_layer
            < n_layers
        ):
            raise RuntimeError(
                f"Requested centroid layer "
                f"L{requested_centroid_layer} outside "
                f"0..{n_layers-1}"
            )

        candidate_centroid_layers = [
            requested_centroid_layer
        ]

    print(
        f"decoder={decoder_path}, "
        f"n_layers={n_layers}, "
        f"transformers={transformers.__version__}"
    )

    # ---------------------------------------------------------------------
    # TRAIN ONLY: select the single best centroid head.
    # Primary score is original+swap aligned AVERAGE centroid accuracy.
    # ---------------------------------------------------------------------
    (
        selector_layer,
        selector_head,
        selector_train_info,
        selector_ranking,
    ) = fit_centroid_head_on_train(
        model,
        processor,
        train_records,
        candidate_centroid_layers,
        args,
        outdir,
    )

    # ---------------------------------------------------------------------
    # TRAIN late causal directions.
    # ---------------------------------------------------------------------
    train_rows, train_cache = (
        collect_train(
            model,
            processor,
            layers,
            train_records,
            actuator_layers,
            args,
        )
    )

    write_csv(
        outdir
        / "train_real_gray.csv",
        train_rows,
    )

    (
        templates,
        chosen_filter,
        template_counts,
    ) = fit_templates(
        train_records,
        train_rows,
        train_cache,
        actuator_layers,
        args.template_filter,
    )

    # Save learned directions for inspection/reuse.
    template_arrays = {}

    for layer in (
        actuator_layers
    ):
        for relation in RELATIONS:
            template_arrays[
                f"L{layer}_{relation}"
            ] = templates[
                int(layer)
            ][
                "shared"
            ][
                relation
            ]

        template_arrays[
            f"L{layer}_global"
        ] = templates[
            int(layer)
        ][
            "global"
        ]

    np.savez_compressed(
        outdir
        / "late_direction_templates.npz",
        relation_order=np.asarray(
            RELATIONS,
            dtype=object,
        ),
        actuator_layers=np.asarray(
            actuator_layers,
            dtype=np.int32,
        ),
        **template_arrays,
    )

    # ---------------------------------------------------------------------
    # TEST read->select->write.
    # ---------------------------------------------------------------------
    test_rows = evaluate_test(
        model,
        processor,
        layers,
        test_records,
        templates,
        actuator_layers,
        selector_layer,
        selector_head,
        args,
    )

    write_csv(
        outdir
        / "test_details.csv",
        test_rows,
    )

    errors = [
        row
        for row in test_rows
        if "error" in row
    ]

    if errors:
        write_csv(
            outdir
            / "test_errors.csv",
            errors,
        )

    summary = summarize(
        test_rows,
        args,
        selector_layer,
        selector_head,
        actuator_layers,
        chosen_filter,
        template_counts,
    )

    summary[
        "centroid_train_selected_accuracy"
    ] = float(
        selector_train_info[
            "train_average_accuracy"
        ]
    )

    summary[
        "centroid_train_selected_original_accuracy"
    ] = float(
        selector_train_info[
            "train_original_accuracy"
        ]
    )

    summary[
        "centroid_train_selected_swapped_accuracy"
    ] = float(
        selector_train_info[
            "train_swapped_aligned_accuracy"
        ]
    )

    relation_rows = (
        per_relation_summary(
            test_rows
        )
    )

    confusion_rows = (
        confusion_counts(
            [
                row
                for row in test_rows
                if "error"
                not in row
            ]
        )
    )

    write_csv(
        outdir
        / "summary.csv",
        [summary],
    )

    write_csv(
        outdir
        / "per_relation.csv",
        relation_rows,
    )

    write_csv(
        outdir
        / "selector_confusion.csv",
        confusion_rows,
    )

    print_summary(
        summary,
        relation_rows,
    )

    metadata = {
        "script": (
            "eval_coco_trainbest_singlehead_centroid_late_direction_qwen25_v2.py"
        ),
        "model": args.model,
        "model_id": (
            config[
                "model_id"
            ]
        ),
        "repo_model_id": (
            spec.repo_id
        ),
        "dataset": "coco_two",
        "prompt_jsonl": (
            args.prompt_jsonl
        ),
        "N_all": len(
            records
        ),
        "N_train": len(
            train_records
        ),
        "N_test": len(
            test_records
        ),
        "train_frac": (
            args.train_frac
        ),
        "seed": args.seed,
        "centroid_layer": (
            selector_layer
        ),
        "centroid_head": (
            selector_head
        ),
        "centroid_train_selection": (
            selector_train_info
        ),
        "centroid_candidate_layers": (
            candidate_centroid_layers
        ),
        "centroid_method": (
            "TRAIN-selected single attention head; "
            "subject/reference text-token -> visual-token attention centroids; "
            "swapped roles aligned back to original semantic order; "
            "original and swapped centroids averaged before |dx| vs |dy| rule"
        ),
        "centroid_axis_rule": (
            "|dx|>=|dy| -> horizontal, otherwise vertical"
        ),
        "actuator_layers": (
            actuator_layers
        ),
        "actuator_definition": (
            "s_r = mean(Real-Gray last-token delta | r) "
            "- balanced four-relation global mean"
        ),
        "template_filter_requested": (
            args.template_filter
        ),
        "template_filter_used": (
            chosen_filter
        ),
        "template_counts": (
            template_counts
        ),
        "scale": args.scale,
        "gray_value": (
            args.gray_value
        ),
        "run_oracle": (
            args.run_oracle
        ),
        "transformers": (
            transformers.__version__
        ),
        "decoder_path": (
            decoder_path
        ),
        "n_decoder_layers": (
            n_layers
        ),
        "audit": audit,
        "test_routing_uses_gt": False,
        "test_head_selection_uses_gt": False,
        "train_head_selection_uses_gt": True,
    }

    (
        outdir
        / "config.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[saved] "
        f"{outdir / 'summary.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'test_details.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'selector_confusion.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'late_direction_templates.npz'}"
    )


if __name__ == "__main__":
    main()
