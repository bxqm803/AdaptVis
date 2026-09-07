#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_coco_centroid_select_late_direction_qwen25_v1.py

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

READ (no GT at TEST):
    At one middle decoder layer, take the subject and reference TEXT tokens as
    queries and their attention to VISUAL tokens.

    For each attention head:
        normalize attention inside visual tokens

    Then average the normalized maps over heads, exactly matching the
    "head-mean" centroid construction in
        analyze_coco_centroid_generation_step1_v4.py

    Compute:
        c_sub = (x_sub, y_sub)
        c_ref = (x_ref, y_ref)

        dx = x_sub - x_ref
        dy = y_sub - y_ref

    Select:
        if |dx| >= |dy|:
            left  if dx < 0 else right
        else:
            above if dy < 0 else below

    Default centroid layers reuse the repo presets:
        qwen-3b: L24
        qwen-7b: L19

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
CUDA_VISIBLE_DEVICES=0 python eval_coco_centroid_select_late_direction_qwen25_v1.py \
  --model qwen-3b \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --train-frac 0.30 \
  --template-filter real_correct_gray_wrong \
  --scale 1.0 \
  --output-dir output/coco_centroid_direction_qwen3b_v1 \
  --overwrite

Qwen2.5-VL-7B:
CUDA_VISIBLE_DEVICES=1 python eval_coco_centroid_select_late_direction_qwen25_v1.py \
  --model qwen-7b \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --train-frac 0.30 \
  --template-filter real_correct_gray_wrong \
  --scale 1.0 \
  --output-dir output/coco_centroid_direction_qwen7b_v1 \
  --overwrite

Smoke test:
CUDA_VISIBLE_DEVICES=0 python eval_coco_centroid_select_late_direction_qwen25_v1.py \
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
            "'auto' uses repo preset (3B=L24, 7B=L19), "
            "or provide a zero-based decoder layer."
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
# TEST READ: attention centroid selector
# =============================================================================

def resolve_attention_tuple(
    outputs: Any,
) -> Sequence[torch.Tensor]:
    candidates = [
        getattr(
            outputs,
            "attentions",
            None,
        ),
        getattr(
            getattr(
                outputs,
                "language_model_output",
                None,
            ),
            "attentions",
            None,
        ),
        getattr(
            getattr(
                outputs,
                "language_model_outputs",
                None,
            ),
            "attentions",
            None,
        ),
    ]

    for value in candidates:
        if (
            isinstance(
                value,
                (tuple, list),
            )
            and len(value) > 0
        ):
            return value

    raise RuntimeError(
        "Forward did not return decoder attentions. "
        "The model must be loaded with attn_implementation='eager'."
    )


def centroid_select(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    selector_layer: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Exact head-mean attention centroid rule used by the repo:

      subject/ref object TEXT tokens -> visual-token attention
      normalize each head inside visual tokens
      average normalized maps across heads
      centroid
      compare |dx| vs |dy|

    No GT is read here.
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

    input_ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )

    input_length = len(
        input_ids
    )

    (
        subject_span,
        reference_span,
    ) = cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        record[
            "subject"
        ],
        record[
            "reference"
        ],
    )

    # Match repo's object-token query choice: final token of each object span.
    subject_index = int(
        subject_span[1]
    )

    reference_index = int(
        reference_span[1]
    )

    visual_indices = (
        cent.resolve_visual_indices(
            model,
            processor,
            batch,
            input_ids,
        )
    )

    coords = cent.visual_coordinates(
        model,
        batch,
        len(
            visual_indices
        ),
        batch[
            "input_ids"
        ].device,
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

    attentions = (
        resolve_attention_tuple(
            outputs
        )
    )

    if not (
        0 <= int(selector_layer)
        < len(attentions)
    ):
        raise RuntimeError(
            f"selector L{selector_layer} unavailable; "
            f"attention layers={len(attentions)}"
        )

    prompt_tensor = (
        cent.normalize_attention_tensor(
            attentions[
                int(
                    selector_layer
                )
            ],
            expected_query_length=(
                input_length
            ),
        )
    )

    # [heads, 2 object queries, key]
    rows = prompt_tensor[
        :,
        [
            subject_index,
            reference_index,
        ],
        :,
    ]

    metrics = (
        cent.query_attention_metrics(
            rows,
            visual_indices,
            coords,
            subject_index,
            reference_index,
        )
    )

    # [heads, 2, visual], each head already normalized over visual tokens.
    maps = (
        metrics[
            "visual_maps"
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # Exact head-mean construction from repo.
    mean_maps = maps.mean(
        axis=0
    )

    coords_np = (
        coords.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # [2 objects, 2 xy]
    centroids = np.einsum(
        "ov,vd->od",
        mean_maps,
        coords_np,
    ).astype(
        np.float32
    )

    dx = float(
        centroids[
            0,
            0,
        ]
        - centroids[
            1,
            0,
        ]
    )

    dy = float(
        centroids[
            0,
            1,
        ]
        - centroids[
            1,
            1,
        ]
    )

    prediction, axis_confidence = (
        cent.relation_from_centroids(
            dx,
            dy,
        )
    )

    # Extra diagnostics only.
    object_visual_mass = (
        metrics[
            "visual_mass"
        ]
        .detach()
        .float()
        .mean(
            dim=0
        )
        .cpu()
        .numpy()
    )

    del outputs
    del attentions
    del batch

    return {
        "prediction": prediction,
        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),
        "axis_confidence": float(
            axis_confidence
        ),
        "subject_x": float(
            centroids[0, 0]
        ),
        "subject_y": float(
            centroids[0, 1]
        ),
        "reference_x": float(
            centroids[1, 0]
        ),
        "reference_y": float(
            centroids[1, 1]
        ),
        "subject_visual_mass": float(
            object_visual_mass[0]
        ),
        "reference_visual_mass": float(
            object_visual_mass[1]
        ),
        "n_visual_tokens": int(
            len(
                visual_indices
            )
        ),
        "n_heads": int(
            maps.shape[0]
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

                "centroid_pred": target,
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
        "COCO ATTENTION-CENTROID SELECTOR -> "
        "LATE DIRECTION GENERATION"
    )

    print(
        "=" * 150
    )

    print(
        f"model={summary['model']} | "
        f"N={summary['N_test_success']} | "
        f"centroid=L{summary['centroid_layer']} | "
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
        f"centroid selector         : "
        f"{summary['centroid_selector_acc']:.4f}"
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

    if (
        str(
            args.centroid_layer
        )
        .strip()
        .lower()
        == "auto"
    ):
        selector_layer = int(
            config[
                "centroid_layer"
            ]
        )
    else:
        selector_layer = int(
            args.centroid_layer
        )

    print(
        "\n"
        + "=" * 150
    )

    print(
        "COCO CENTROID-SELECTED LATE DIRECTION"
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
        f"READ: head-mean object-attention centroid "
        f"at L{selector_layer}"
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

    bad_layers = [
        layer
        for layer in (
            [selector_layer]
            + actuator_layers
        )
        if not (
            0 <= int(layer)
            < n_layers
        )
    ]

    if bad_layers:
        raise RuntimeError(
            f"Configured layers outside decoder "
            f"0..{n_layers-1}: {bad_layers}"
        )

    print(
        f"decoder={decoder_path}, "
        f"n_layers={n_layers}, "
        f"transformers={transformers.__version__}"
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
        actuator_layers,
        chosen_filter,
        template_counts,
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
            "eval_coco_centroid_select_late_direction_qwen25_v1.py"
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
        "centroid_method": (
            "head-mean normalized subject/reference "
            "object-token -> visual-token attention centroid"
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
