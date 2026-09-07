#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_synthetic_shapes_to_coco_late_direction_qwen3b_v1.py

Goal
====
Extract late spatial causal directions from the user's completely synthetic
two-shape dataset, then freeze those directions and test them on COCO_two with
Qwen2.5-VL-3B-Instruct.

SOURCE
======
SyntheticShapes4Dir:
    left / right / on / under
    on    -> above
    under -> below

The source prompt is standardized to the TARGET vocabulary:

    Where is the {subject} relative to the {reference}?
    Answer with left, right, above, or below.

For each synthetic source sample and actuator layer l:

    delta_i,l
        = h_last(real synthetic image)_i,l
        - h_last(gray blank image)_i,l

For each relation r:

    mu_r,l = mean(delta_i,l | relation=r)

Balanced common component:

    mu_global,l
        = 1/4 * sum_r mu_r,l

Synthetic spatial direction:

    s^syn_r,l
        = mu_r,l - mu_global,l

Default source filter:
    all

Optional:
    --template-filter real_correct
    --template-filter real_correct_gray_wrong

TARGET
======
COCO_two, using the repository's standard COCO prompts.

No COCO sample is used to fit the direction vectors.

TEST
====
Actual greedy generation on ALL available COCO target samples:

    baseline:
        normal Qwen3B generation

    synthetic oracle steering:
        GT relation chooses ONLY which frozen synthetic direction to inject

        h_last,l <- h_last,l + scale * s^syn_GT,l

This is intentionally an ORACLE routing experiment.  It isolates the
cross-dataset transferability of the actuator itself.  It is not yet a
deployable non-oracle pipeline.

Default Qwen3B actuator window:
    L32,L33,L34,L35

Outputs
=======
    source_real_gray.csv
    synthetic_direction_templates.npz
    target_details.csv
    summary.csv
    per_relation.csv
    config.json

Example
=======
CUDA_VISIBLE_DEVICES=0 python eval_synthetic_shapes_to_coco_late_direction_qwen3b_v1.py \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --actuator-layers 32-35 \
  --template-filter all \
  --scale 1.0 \
  --output-dir output/synthetic_shapes_to_coco_qwen3b_v1 \
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


# =============================================================================
# Repository dependencies
# =============================================================================

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
SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "above",
    "above": "above",
    "under": "below",
    "below": "below",
}

MODEL_KEY = "qwen-3b"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ACTUATOR_LAYERS = [32, 33, 34, 35]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--synthetic-dir",
        default="synthetic_shapes_4dir_400",
        help="Directory produced by generate_synthetic_shapes_4dir_v1.py",
    )

    p.add_argument(
        "--synthetic-labels",
        default=None,
        help="Optional explicit labels.jsonl path.",
    )

    p.add_argument(
        "--data-root",
        default="data",
    )

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    p.add_argument(
        "--dtype",
        default="auto",
        choices=[
            "auto",
            "bfloat16",
            "float16",
            "float32",
        ],
    )

    p.add_argument(
        "--actuator-layers",
        default="32-35",
        help="Late decoder layer(s), e.g. 32-35 or 32,33,34,35.",
    )

    p.add_argument(
        "--template-filter",
        default="all",
        choices=[
            "all",
            "real_correct",
            "real_correct_gray_wrong",
        ],
        help=(
            "Which synthetic source samples are used to fit directions. "
            "'all' is the clean default for this independent synthetic source."
        ),
    )

    p.add_argument(
        "--gray-value",
        type=int,
        default=128,
        help="Constant value for the blank gray counterfactual image.",
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

    p.add_argument(
        "--source-max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--target-max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1,
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


def parse_layer_spec(
    text: str,
) -> List[int]:
    values: List[int] = []

    for part in str(text).split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a)
            b = int(b)
            step = 1 if b >= a else -1

            values.extend(
                range(
                    a,
                    b + step,
                    step,
                )
            )
        else:
            values.append(
                int(part)
            )

    values = list(
        dict.fromkeys(
            values
        )
    )

    if not values:
        raise ValueError(
            "No actuator layers selected."
        )

    return values


def normalize_generation(
    text: str,
) -> Optional[str]:
    return cent.normalize_relation(
        text
    )


def classify_transition(
    before_ok: bool,
    after_ok: bool,
) -> str:
    if (
        not before_ok
        and after_ok
    ):
        return "W2C"

    if (
        before_ok
        and not after_ok
    ):
        return "C2W"

    if (
        before_ok
        and after_ok
    ):
        return "C2C"

    return "W2W"


# =============================================================================
# Synthetic source dataset
# =============================================================================

def load_synthetic_records(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    root = Path(
        args.synthetic_dir
    )

    labels_path = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Synthetic labels not found: {labels_path}"
        )

    rows: List[Dict[str, Any]] = []

    with labels_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_no, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            item = json.loads(
                line
            )

            raw_relation = (
                str(
                    item["relation"]
                )
                .strip()
                .lower()
            )

            if raw_relation not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: "
                    f"unsupported relation={raw_relation!r}"
                )

            relation = SYN_REL_MAP[
                raw_relation
            ]

            subject = str(
                item["subject"]
            ).strip()

            reference = str(
                item["reference"]
            ).strip()

            image_value = Path(
                str(
                    item["image"]
                )
            )

            image_path = (
                image_value
                if image_value.is_absolute()
                else root / image_value
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"{labels_path}:{line_no}: "
                    f"missing image {image_path}"
                )

            # IMPORTANT:
            # Use the same output vocabulary as the COCO target.
            question_text = (
                f"Where is the {subject} relative to the {reference}? "
                f"Answer with left, right, above, or below."
            )

            rows.append({
                "sid": int(
                    item.get(
                        "id",
                        len(rows),
                    )
                ),
                "relation": relation,
                "source_relation_raw": raw_relation,
                "subject": subject,
                "reference": reference,
                "question_text": question_text,
                "image_path": str(
                    image_path
                ),
            })

    rows.sort(
        key=lambda row: int(
            row["sid"]
        )
    )

    if args.source_max_samples is not None:
        rows = rows[
            : int(
                args.source_max_samples
            )
        ]

    if not rows:
        raise RuntimeError(
            "Synthetic source dataset is empty."
        )

    return rows


def synthetic_relation_counts(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[str, int]:
    return {
        relation: int(
            sum(
                row["relation"]
                == relation
                for row in records
            )
        )
        for relation in RELATIONS
    }


def open_synthetic_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return (
        Image.open(
            record["image_path"]
        )
        .convert("RGB")
    )


# =============================================================================
# COCO target
# =============================================================================

def load_coco_records(
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    two_obj = (
        cent.import_two_object_module()
    )

    raw_records, audit = (
        two_obj.load_records(
            "coco_two",
            Path(
                args.data_root
            ),
            args.target_max_samples,
        )
    )

    prompt_path = Path(
        args.prompt_jsonl
    )

    prompt_rows = (
        cent.load_standard_prompts(
            prompt_path
        )
    )

    records: List[
        Dict[str, Any]
    ] = []

    for raw in raw_records:
        sid = int(
            raw.sid
        )

        if sid not in prompt_rows:
            raise RuntimeError(
                f"sid={sid} missing from {prompt_path}"
            )

        prompt = prompt_rows[
            sid
        ]

        relation = (
            cent.normalize_relation(
                prompt[
                    "answer_raw"
                ]
            )
        )

        if relation not in REL_TO_ID:
            raise RuntimeError(
                f"sid={sid}: unsupported "
                f"COCO relation={relation!r}"
            )

        records.append({
            "sid": sid,
            "relation": relation,
            "subject": str(
                prompt[
                    "subject"
                ]
            ),
            "reference": str(
                prompt[
                    "reference"
                ]
            ),
            "question_text": str(
                prompt[
                    "question_text"
                ]
            ),
            "raw_record": raw,
        })

    records.sort(
        key=lambda row: int(
            row["sid"]
        )
    )

    return records, audit


def open_coco_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return cent.record_image(
        record[
            "raw_record"
        ]
    )


# =============================================================================
# Qwen3B
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
    }[
        name
    ]


def load_model_and_processor(
    args: argparse.Namespace,
):
    two_obj = (
        cent.import_two_object_module()
    )

    specs = (
        cent.merged_model_specs(
            two_obj
        )
    )

    if MODEL_KEY not in specs:
        raise ValueError(
            f"{MODEL_KEY!r} not found in repo model specs. "
            f"Available={sorted(specs)}"
        )

    spec = specs[
        MODEL_KEY
    ]

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )

    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {spec.model_class}"
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
        # This is compatible with the existing repo scripts.
        "attn_implementation": "eager",
    }

    try:
        model = (
            model_cls
            .from_pretrained(
                spec.repo_id,
                dtype=dtype,
                **kwargs,
            )
        )

    except TypeError:
        model = (
            model_cls
            .from_pretrained(
                spec.repo_id,
                torch_dtype=dtype,
                **kwargs,
            )
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
# Prompt / generation / hidden capture
# =============================================================================

def make_batch(
    processor: Any,
    image: Image.Image,
    question_text: str,
    device: torch.device,
) -> Dict[str, Any]:
    return cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
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


def capture_last_states(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    question_text: str,
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    str,
    Optional[str],
    Dict[int, np.ndarray],
]:
    """
    Same CaptureStates mechanism as the user's existing late-causal script.
    No middle guide is involved.
    """
    batch = make_batch(
        processor,
        image,
        question_text,
        torch.device(
            args.device
        ),
    )

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
        normalize_generation(
            text
        ),
        last_states,
    )


# =============================================================================
# SOURCE: Real - Gray synthetic direction extraction
# =============================================================================

def source_filter_allowed(
    row: Mapping[str, Any],
    mode: str,
) -> bool:
    if mode == "all":
        return True

    if mode == "real_correct":
        return bool(
            row[
                "real_correct"
            ]
        )

    if mode == "real_correct_gray_wrong":
        return (
            bool(
                row[
                    "real_correct"
                ]
            )
            and not bool(
                row[
                    "gray_correct"
                ]
            )
        )

    raise ValueError(
        mode
    )


def collect_synthetic_source(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    records: Sequence[
        Mapping[str, Any]
    ],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[
        int,
        Dict[
            int,
            np.ndarray,
        ],
    ],
]:
    rows: List[
        Dict[str, Any]
    ] = []

    deltas: Dict[
        int,
        Dict[
            int,
            np.ndarray,
        ],
    ] = {}

    for record in tqdm(
        records,
        desc="SOURCE synthetic Real/Gray Qwen3B",
    ):
        real = None
        gray = None

        try:
            real = open_synthetic_image(
                record
            )

            gray = causal.make_gray(
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
                record[
                    "question_text"
                ],
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
                record[
                    "question_text"
                ],
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
                "source_relation_raw": (
                    record[
                        "source_relation_raw"
                    ]
                ),
                "subject": (
                    record[
                        "subject"
                    ]
                ),
                "reference": (
                    record[
                        "reference"
                    ]
                ),
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
                "real_text": (
                    real_text
                ),
                "gray_text": (
                    gray_text
                ),
            })

            deltas[
                int(
                    record[
                        "sid"
                    ]
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
                f"[SOURCE ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    return (
        rows,
        deltas,
    )


def fit_synthetic_templates(
    source_records: Sequence[
        Mapping[str, Any]
    ],
    source_rows: Sequence[
        Mapping[str, Any]
    ],
    delta_cache: Mapping[
        int,
        Mapping[
            int,
            np.ndarray,
        ],
    ],
    actuator_layers: Sequence[int],
    mode: str,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[str, int],
]:
    row_map = {
        int(
            row["sid"]
        ): row
        for row in source_rows
    }

    record_map = {
        int(
            row["sid"]
        ): row
        for row in source_records
    }

    bags = {
        int(layer): {
            relation: []
            for relation
            in RELATIONS
        }
        for layer
        in actuator_layers
    }

    counts = Counter()

    for sid, layer_values in (
        delta_cache.items()
    ):
        sid = int(
            sid
        )

        if (
            sid not in row_map
            or sid not in record_map
        ):
            continue

        source_row = row_map[
            sid
        ]

        if not source_filter_allowed(
            source_row,
            mode,
        ):
            continue

        relation = record_map[
            sid
        ][
            "relation"
        ]

        counts[
            relation
        ] += 1

        for layer in actuator_layers:
            layer = int(
                layer
            )

            bags[
                layer
            ][
                relation
            ].append(
                np.asarray(
                    layer_values[
                        layer
                    ],
                    dtype=np.float32,
                )
            )

    missing = [
        (
            int(layer),
            relation,
        )
        for layer in actuator_layers
        for relation in RELATIONS
        if not bags[
            int(layer)
        ][
            relation
        ]
    ]

    if missing:
        raise RuntimeError(
            f"Source filter={mode!r} leaves missing "
            f"layer/relation buckets={missing}. "
            f"Use --template-filter all or a less strict filter."
        )

    templates: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for layer in actuator_layers:
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
            for relation in RELATIONS
        }

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
            "shared": (
                shared
            ),
        }

    relation_counts = {
        relation: int(
            counts[
                relation
            ]
        )
        for relation
        in RELATIONS
    }

    return (
        templates,
        relation_counts,
    )


# =============================================================================
# TARGET COCO: baseline / synthetic-direction oracle
# =============================================================================

def steer_generate(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    record: Mapping[str, Any],
    templates: Mapping[
        int,
        Any,
    ],
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
        record[
            "question_text"
        ],
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
        normalize_generation(
            text
        ),
    )


def evaluate_coco_target(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    records: Sequence[
        Mapping[str, Any]
    ],
    templates: Mapping[
        int,
        Any,
    ],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> List[
    Dict[str, Any]
]:
    rows: List[
        Dict[str, Any]
    ] = []

    for record in tqdm(
        records,
        desc="TARGET COCO baseline + synthetic oracle",
    ):
        image = None

        try:
            image = open_coco_image(
                record
            )

            gt = record[
                "relation"
            ]

            # -------------------------------------------------------------
            # Baseline actual greedy generation.
            # -------------------------------------------------------------
            batch = make_batch(
                processor,
                image,
                record[
                    "question_text"
                ],
                torch.device(
                    args.device
                ),
            )

            baseline_text = generate_text(
                model,
                processor,
                batch,
                args,
            )

            del batch

            baseline_pred = (
                normalize_generation(
                    baseline_text
                )
            )

            baseline_ok = (
                baseline_pred
                == gt
            )

            # -------------------------------------------------------------
            # Frozen synthetic direction.
            # GT only selects relation direction (oracle transfer test).
            # -------------------------------------------------------------
            (
                steered_text,
                steered_pred,
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

            steered_ok = (
                steered_pred
                == gt
            )

            transition = (
                classify_transition(
                    baseline_ok,
                    steered_ok,
                )
            )

            rows.append({
                "sid": int(
                    record[
                        "sid"
                    ]
                ),
                "relation": gt,

                "baseline_text": (
                    baseline_text
                ),
                "baseline_pred": (
                    baseline_pred
                    or ""
                ),
                "baseline_correct": int(
                    baseline_ok
                ),

                "synthetic_oracle_text": (
                    steered_text
                ),
                "synthetic_oracle_pred": (
                    steered_pred
                    or ""
                ),
                "synthetic_oracle_correct": int(
                    steered_ok
                ),

                "transition": (
                    transition
                ),
                "W2C": int(
                    transition
                    == "W2C"
                ),
                "C2W": int(
                    transition
                    == "C2W"
                ),
            })

        except Exception as exc:
            tqdm.write(
                f"[TARGET ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

            rows.append({
                "sid": int(
                    record[
                        "sid"
                    ]
                ),
                "relation": (
                    record[
                        "relation"
                    ]
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


# =============================================================================
# Reporting
# =============================================================================

def build_source_summary(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> List[Dict[str, Any]]:
    good = [
        row
        for row in rows
        if "error" not in row
    ]

    result: List[
        Dict[str, Any]
    ] = []

    for relation in (
        ["ALL"]
        + list(
            RELATIONS
        )
    ):
        subset = (
            good
            if relation == "ALL"
            else [
                row
                for row in good
                if row[
                    "relation"
                ]
                == relation
            ]
        )

        result.append({
            "relation": relation,
            "N": len(
                subset
            ),
            "real_acc": safe_mean(
                row[
                    "real_correct"
                ]
                for row
                in subset
            ),
            "gray_acc": safe_mean(
                row[
                    "gray_correct"
                ]
                for row
                in subset
            ),
            "real_correct_gray_wrong_rate": safe_mean(
                (
                    int(
                        row[
                            "real_correct"
                        ]
                    )
                    == 1
                    and int(
                        row[
                            "gray_correct"
                        ]
                    )
                    == 0
                )
                for row
                in subset
            ),
        })

    return result


def build_target_summary(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    good = [
        row
        for row in rows
        if "error" not in row
    ]

    if not good:
        raise RuntimeError(
            "No successful COCO target samples."
        )

    baseline_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in good
    )

    steered_acc = safe_mean(
        row[
            "synthetic_oracle_correct"
        ]
        for row in good
    )

    W2C = int(
        sum(
            row[
                "W2C"
            ]
            for row in good
        )
    )

    C2W = int(
        sum(
            row[
                "C2W"
            ]
            for row in good
        )
    )

    wrong = [
        row
        for row in good
        if int(
            row[
                "baseline_correct"
            ]
        )
        == 0
    ]

    correct = [
        row
        for row in good
        if int(
            row[
                "baseline_correct"
            ]
        )
        == 1
    ]

    summary = {
        "N": len(
            good
        ),
        "baseline_acc": (
            baseline_acc
        ),
        "synthetic_oracle_acc": (
            steered_acc
        ),
        "gain": (
            steered_acc
            - baseline_acc
        ),
        "W2C": (
            W2C
        ),
        "C2W": (
            C2W
        ),
        "net": (
            W2C
            - C2W
        ),
        "baseline_wrong_N": len(
            wrong
        ),
        "repair_rate_on_baseline_wrong": safe_mean(
            row[
                "synthetic_oracle_correct"
            ]
            for row
            in wrong
        ),
        "baseline_correct_N": len(
            correct
        ),
        "preserve_rate_on_baseline_correct": safe_mean(
            row[
                "synthetic_oracle_correct"
            ]
            for row
            in correct
        ),
    }

    per_relation: List[
        Dict[str, Any]
    ] = []

    for relation in RELATIONS:
        subset = [
            row
            for row in good
            if row[
                "relation"
            ]
            == relation
        ]

        b = safe_mean(
            row[
                "baseline_correct"
            ]
            for row in subset
        )

        s = safe_mean(
            row[
                "synthetic_oracle_correct"
            ]
            for row in subset
        )

        per_relation.append({
            "relation": (
                relation
            ),
            "N": len(
                subset
            ),
            "baseline_acc": (
                b
            ),
            "synthetic_oracle_acc": (
                s
            ),
            "gain": (
                s - b
            ),
            "W2C": int(
                sum(
                    row[
                        "W2C"
                    ]
                    for row
                    in subset
                )
            ),
            "C2W": int(
                sum(
                    row[
                        "C2W"
                    ]
                    for row
                    in subset
                )
            ),
        })

    return (
        summary,
        per_relation,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

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

    actuator_layers = (
        parse_layer_spec(
            args.actuator_layers
        )
    )

    source_records = (
        load_synthetic_records(
            args
        )
    )

    (
        target_records,
        target_audit,
    ) = load_coco_records(
        args
    )

    print("\n" + "=" * 150)
    print("SYNTHETIC SHAPES -> COCO LATE DIRECTION TRANSFER | QWEN2.5-VL-3B")
    print("=" * 150)

    print(
        f"SOURCE synthetic N={len(source_records)} | "
        f"{synthetic_relation_counts(source_records)}"
    )

    print(
        f"TARGET COCO N={len(target_records)} | "
        f"{synthetic_relation_counts(target_records)}"
    )

    print(
        f"actuator layers={actuator_layers} | "
        f"source filter={args.template_filter} | "
        f"scale={args.scale}"
    )

    print(
        "Synthetic label mapping: "
        "on->above, under->below"
    )

    print(
        "SOURCE prompt vocabulary is forced to "
        "left/right/above/below to match COCO."
    )

    print(
        "COCO is NEVER used to fit the directions. "
        "GT is used only to select the frozen synthetic direction at TEST."
    )

    print("=" * 150)

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
        for layer
        in actuator_layers
        if not (
            0
            <= int(layer)
            < n_layers
        )
    ]

    if bad_layers:
        raise RuntimeError(
            f"Actuator layers outside decoder "
            f"0..{n_layers - 1}: {bad_layers}"
        )

    print(
        f"model={spec.repo_id} | "
        f"decoder={decoder_path} | "
        f"n_layers={n_layers} | "
        f"transformers={transformers.__version__}"
    )

    # ---------------------------------------------------------------------
    # 1. Extract Real-Gray deltas from synthetic source.
    # ---------------------------------------------------------------------
    (
        source_rows,
        delta_cache,
    ) = collect_synthetic_source(
        model,
        processor,
        layers,
        source_records,
        actuator_layers,
        args,
    )

    write_csv(
        outdir
        / "source_real_gray.csv",
        source_rows,
    )

    source_summary = (
        build_source_summary(
            source_rows
        )
    )

    write_csv(
        outdir
        / "source_summary.csv",
        source_summary,
    )

    print("\n" + "=" * 150)
    print("SYNTHETIC SOURCE GENERATION")
    print("=" * 150)

    for row in source_summary:
        print(
            f"{row['relation']:>5s} | "
            f"N={row['N']:3d} | "
            f"real={row['real_acc']:.4f} | "
            f"gray={row['gray_acc']:.4f} | "
            f"real-C / gray-W="
            f"{row['real_correct_gray_wrong_rate']:.4f}"
        )

    # ---------------------------------------------------------------------
    # 2. Fit FROZEN synthetic relation directions.
    # ---------------------------------------------------------------------
    (
        templates,
        template_counts,
    ) = fit_synthetic_templates(
        source_records,
        source_rows,
        delta_cache,
        actuator_layers,
        args.template_filter,
    )

    print("\n" + "=" * 150)
    print("FROZEN SYNTHETIC DIRECTION BANK")
    print("=" * 150)

    print(
        f"fit counts={template_counts}"
    )

    for layer in actuator_layers:
        print(
            f"L{layer:02d} | "
            + " | ".join(
                f"{relation}="
                f"{np.linalg.norm(templates[layer]['shared'][relation]):.4f}"
                for relation in RELATIONS
            )
        )

    print("=" * 150)

    arrays: Dict[
        str,
        Any,
    ] = {
        "relation_order": np.asarray(
            RELATIONS,
            dtype=object,
        ),
        "actuator_layers": np.asarray(
            actuator_layers,
            dtype=np.int32,
        ),
    }

    for layer in actuator_layers:
        arrays[
            f"L{layer}_global"
        ] = templates[
            layer
        ][
            "global"
        ]

        for relation in RELATIONS:
            arrays[
                f"L{layer}_{relation}"
            ] = templates[
                layer
            ][
                "shared"
            ][
                relation
            ]

            arrays[
                f"L{layer}_{relation}_raw_mean"
            ] = templates[
                layer
            ][
                "relation_mean"
            ][
                relation
            ]

    np.savez_compressed(
        outdir
        / "synthetic_direction_templates.npz",
        **arrays,
    )

    # ---------------------------------------------------------------------
    # 3. Frozen source directions -> ALL COCO target samples.
    # ---------------------------------------------------------------------
    target_rows = (
        evaluate_coco_target(
            model,
            processor,
            layers,
            target_records,
            templates,
            actuator_layers,
            args,
        )
    )

    write_csv(
        outdir
        / "target_details.csv",
        target_rows,
    )

    errors = [
        row
        for row in target_rows
        if "error" in row
    ]

    if errors:
        write_csv(
            outdir
            / "target_errors.csv",
            errors,
        )

    (
        summary,
        per_relation,
    ) = build_target_summary(
        target_rows
    )

    summary.update({
        "source_N": len(
            source_records
        ),
        "source_filter": (
            args.template_filter
        ),
        "source_left_N": (
            template_counts[
                "left"
            ]
        ),
        "source_right_N": (
            template_counts[
                "right"
            ]
        ),
        "source_above_N": (
            template_counts[
                "above"
            ]
        ),
        "source_below_N": (
            template_counts[
                "below"
            ]
        ),
        "actuator_layers": (
            ",".join(
                str(x)
                for x
                in actuator_layers
            )
        ),
        "scale": (
            args.scale
        ),
    })

    write_csv(
        outdir
        / "summary.csv",
        [summary],
    )

    write_csv(
        outdir
        / "per_relation.csv",
        per_relation,
    )

    print("\n" + "=" * 150)
    print("ACTUAL GREEDY GENERATION: SYNTHETIC DIRECTIONS -> COCO")
    print("=" * 150)

    print(
        f"N_TARGET={summary['N']} | "
        f"source={len(source_records)} synthetic images | "
        f"layers={actuator_layers}"
    )

    print(
        f"COCO baseline                 : "
        f"{summary['baseline_acc']:.4f}"
    )

    print(
        f"synthetic-direction oracle    : "
        f"{summary['synthetic_oracle_acc']:.4f} "
        f"({summary['gain']:+.4f}) | "
        f"W2C={summary['W2C']} "
        f"C2W={summary['C2W']} "
        f"net={summary['net']:+d} | "
        f"repair(base-wrong)="
        f"{summary['repair_rate_on_baseline_wrong']:.4f} | "
        f"preserve(base-correct)="
        f"{summary['preserve_rate_on_baseline_correct']:.4f}"
    )

    print("\nPer relation:")

    for row in per_relation:
        print(
            f"{row['relation']:>5s} | "
            f"N={row['N']:3d} | "
            f"base={row['baseline_acc']:.4f} | "
            f"syn_oracle={row['synthetic_oracle_acc']:.4f} "
            f"({row['gain']:+.4f}) | "
            f"W2C/C2W={row['W2C']}/{row['C2W']}"
        )

    print("=" * 150)

    metadata = {
        "script": (
            "eval_synthetic_shapes_to_coco_late_direction_qwen3b_v1.py"
        ),
        "model_key": (
            MODEL_KEY
        ),
        "model_id": (
            MODEL_ID
        ),
        "repo_model_id": (
            spec.repo_id
        ),
        "source_dataset": (
            str(
                args.synthetic_dir
            )
        ),
        "source_N": len(
            source_records
        ),
        "source_relation_mapping": (
            SYN_REL_MAP
        ),
        "source_prompt_template": (
            "Where is the {subject} relative to the {reference}? "
            "Answer with left, right, above, or below."
        ),
        "source_gray_value": (
            args.gray_value
        ),
        "source_filter": (
            args.template_filter
        ),
        "source_template_counts": (
            template_counts
        ),
        "direction_definition": (
            "s_r,l = mean(h_real_last - h_gray_last | relation=r) "
            "- balanced mean of the four relation means"
        ),
        "target_dataset": (
            "coco_two"
        ),
        "target_N": len(
            target_records
        ),
        "target_prompt_jsonl": (
            args.prompt_jsonl
        ),
        "target_used_for_direction_fitting": (
            False
        ),
        "target_gt_used_for_routing": (
            True
        ),
        "target_gt_routing_purpose": (
            "oracle actuator transfer test only"
        ),
        "actuator_layers": (
            actuator_layers
        ),
        "scale": (
            args.scale
        ),
        "decoder_path": (
            decoder_path
        ),
        "n_decoder_layers": (
            n_layers
        ),
        "transformers": (
            transformers.__version__
        ),
        "target_audit": (
            target_audit
        ),
        "seed": (
            args.seed
        ),
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
        f"{outdir / 'source_real_gray.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'synthetic_direction_templates.npz'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'target_details.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'summary.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'per_relation.csv'}"
    )


if __name__ == "__main__":
    main()
