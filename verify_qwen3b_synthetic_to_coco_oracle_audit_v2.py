#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_qwen3b_synthetic_to_coco_oracle_audit_v2.py

Fresh audit of:

    synthetic_shapes_4dir_400
        -> freshly fit Qwen3B late Real-Gray spatial directions
        -> freeze them
        -> COCO_two baseline + GT-oracle steering

This script is self-contained with respect to previously generated helper files.
It imports ONLY repository-native scripts:

    analyze_coco_centroid_generation_step1_v4.py
    extract_two_object_relation_states.py
    eval_crossdataset_late_causal_qwen25_v4_auto_vgroot.py

It does NOT import:
    eval_synthetic_shapes_to_coco_late_direction_qwen3b_v1.py
    eval_synthetic_shapes_to_coco_multiselector_multimodel_v2.py
    or any previously generated experiment script.

Audit guarantees
================
1. No previously saved direction .npz is loaded.
2. Directions are freshly computed from synthetic shape images only.
3. COCO hidden states are never used to fit any direction.
4. Before fitting, the script reads COCO question text and recovers the
   dominant question template.
5. Synthetic source questions are rewritten to use that exact dominant COCO
   question wording.
6. COCO GT is used only:
       - for metric reporting
       - for the explicitly labeled oracle relation selection

Default Qwen3B writer layers:
    L32-L35
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
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
# ONLY repository-native imports
# =============================================================================

try:
    import analyze_coco_centroid_generation_step1_v4 as cent
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_centroid_generation_step1_v4.py. "
        "Run this script from the AdaptVis repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import extract_two_object_relation_states as twoobj
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import eval_crossdataset_late_causal_qwen25_v4_auto_vgroot as causal
except Exception as exc:
    raise SystemExit(
        "Could not import eval_crossdataset_late_causal_qwen25_v4_auto_vgroot.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


MODEL_KEY = "qwen-3b"

RELATIONS = (
    "left",
    "right",
    "above",
    "below",
)

SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "above",
    "above": "above",
    "under": "below",
    "below": "below",
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
        "--synthetic-dir",
        default="synthetic_shapes_4dir_400",
    )

    p.add_argument(
        "--synthetic-labels",
        default=None,
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
        "--audit-examples",
        type=int,
        default=8,
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

    return float(
        np.mean(vals)
    ) if vals else float("nan")


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


def parse_layers(
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


def sha256_vector(
    value: np.ndarray,
) -> str:
    value = np.ascontiguousarray(
        np.asarray(
            value,
            dtype=np.float32,
        )
    )

    return hashlib.sha256(
        value.tobytes()
    ).hexdigest()


def classify_transition(
    before_ok: bool,
    after_ok: bool,
) -> str:
    if not before_ok and after_ok:
        return "W2C"

    if before_ok and not after_ok:
        return "C2W"

    if before_ok and after_ok:
        return "C2C"

    return "W2W"


# =============================================================================
# COCO records and prompt audit
# =============================================================================

def load_coco_records(
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, Any]],
    Any,
]:
    raw_records, audit = twoobj.load_records(
        "coco_two",
        Path(
            args.data_root
        ),
        args.target_max_samples,
    )

    prompts = cent.load_standard_prompts(
        Path(
            args.prompt_jsonl
        )
    )

    records: List[
        Dict[str, Any]
    ] = []

    for raw in raw_records:
        sid = int(
            raw.sid
        )

        if sid not in prompts:
            raise RuntimeError(
                f"COCO sid={sid} missing from "
                f"{args.prompt_jsonl}"
            )

        prompt = prompts[
            sid
        ]

        relation = cent.normalize_relation(
            prompt[
                "answer_raw"
            ]
        )

        if relation not in RELATIONS:
            raise RuntimeError(
                f"sid={sid}: unexpected relation="
                f"{relation!r}"
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
            "answer_raw": str(
                prompt[
                    "answer_raw"
                ]
            ),
            "raw_record": raw,
        })

    records.sort(
        key=lambda row: int(
            row[
                "sid"
            ]
        )
    )

    return (
        records,
        audit,
    )


def replace_once_case_insensitive(
    text: str,
    phrase: str,
    replacement: str,
) -> Tuple[
    str,
    bool,
]:
    if not phrase:
        return (
            text,
            False,
        )

    pattern = re.compile(
        re.escape(
            phrase
        ),
        flags=re.IGNORECASE,
    )

    new_text, count = pattern.subn(
        replacement,
        text,
        count=1,
    )

    return (
        new_text,
        count == 1,
    )


def normalize_question_template(
    question: str,
    subject: str,
    reference: str,
) -> Optional[str]:
    pairs = [
        (
            str(
                subject
            ),
            "__SUBJECT__",
        ),
        (
            str(
                reference
            ),
            "__REFERENCE__",
        ),
    ]

    # Replace longer phrase first in case one noun phrase contains the other.
    pairs.sort(
        key=lambda item: len(
            item[
                0
            ]
        ),
        reverse=True,
    )

    text = str(
        question
    )

    success = {}

    for phrase, token in pairs:
        text, ok = replace_once_case_insensitive(
            text,
            phrase,
            token,
        )

        success[
            token
        ] = ok

    if not (
        success.get(
            "__SUBJECT__",
            False,
        )
        and success.get(
            "__REFERENCE__",
            False,
        )
    ):
        return None

    return (
        text
        .replace(
            "__SUBJECT__",
            "{subject}",
        )
        .replace(
            "__REFERENCE__",
            "{reference}",
        )
    )


def infer_dominant_coco_template(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[str, Any]:
    counts = Counter()
    failed = []

    for record in records:
        template = normalize_question_template(
            record[
                "question_text"
            ],
            record[
                "subject"
            ],
            record[
                "reference"
            ],
        )

        if template is None:
            failed.append(
                int(
                    record[
                        "sid"
                    ]
                )
            )
            continue

        counts[
            template
        ] += 1

    if not counts:
        raise RuntimeError(
            "Could not infer any reusable COCO question template."
        )

    template, count = counts.most_common(
        1
    )[
        0
    ]

    return {
        "template": template,
        "count": int(
            count
        ),
        "unique_templates": int(
            len(
                counts
            )
        ),
        "parsed_total": int(
            sum(
                counts.values()
            )
        ),
        "failed_sids": failed,
        "all_counts": counts,
    }


# =============================================================================
# Synthetic source
# =============================================================================

def load_synthetic_records(
    args: argparse.Namespace,
    coco_template: str,
) -> List[
    Dict[str, Any]
]:
    root = Path(
        args.synthetic_dir
    )

    labels_path = (
        Path(
            args.synthetic_labels
        )
        if args.synthetic_labels
        else root
        / "labels.jsonl"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            labels_path
        )

    records = []

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

            raw_relation = str(
                item[
                    "relation"
                ]
            ).strip().lower()

            if raw_relation not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: "
                    f"bad relation={raw_relation!r}"
                )

            relation = SYN_REL_MAP[
                raw_relation
            ]

            subject = str(
                item[
                    "subject"
                ]
            ).strip()

            reference = str(
                item[
                    "reference"
                ]
            ).strip()

            image_value = Path(
                str(
                    item[
                        "image"
                    ]
                )
            )

            image_path = (
                image_value
                if image_value.is_absolute()
                else root
                / image_value
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    image_path
                )

            # EXACT dominant COCO wording.
            question_text = coco_template.format(
                subject=subject,
                reference=reference,
            )

            records.append({
                "sid": int(
                    item.get(
                        "id",
                        len(
                            records
                        ),
                    )
                ),
                "relation": relation,
                "raw_relation": raw_relation,
                "subject": subject,
                "reference": reference,
                "question_text": question_text,
                "image_path": str(
                    image_path
                ),
            })

    records.sort(
        key=lambda row: int(
            row[
                "sid"
            ]
        )
    )

    if args.source_max_samples is not None:
        records = records[
            : int(
                args.source_max_samples
            )
        ]

    if not records:
        raise RuntimeError(
            "Synthetic source is empty."
        )

    return records


def open_synthetic_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return Image.open(
        record[
            "image_path"
        ]
    ).convert(
        "RGB"
    )


def open_coco_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return cent.record_image(
        record[
            "raw_record"
        ]
    )


# =============================================================================
# Model loading
# =============================================================================

def resolve_dtype(
    requested: str,
    spec: Any,
) -> torch.dtype:
    name = (
        str(
            spec.dtype_name
        )
        if requested
        == "auto"
        else requested
    )

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype={name!r}"
        )

    return mapping[
        name
    ]


def load_model_and_processor(
    args: argparse.Namespace,
):
    device = torch.device(
        args.device
    )

    print(
        "\n"
        + "="
        * 130
    )
    print(
        "CUDA / MODEL AUDIT"
    )
    print(
        "="
        * 130
    )

    print(
        f"torch={torch.__version__} | "
        f"torch_cuda={torch.version.cuda} | "
        f"CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')} | "
        f"cuda_available="
        f"{torch.cuda.is_available()} | "
        f"cuda_count="
        f"{torch.cuda.device_count()}"
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        index = (
            0
            if device.index is None
            else int(
                device.index
            )
        )

        if not (
            0
            <= index
            < torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"Requested logical CUDA device {index}; "
                f"torch sees {torch.cuda.device_count()} GPU(s)."
            )

        torch.cuda.set_device(
            index
        )

        # Real allocation before loading.
        probe = torch.empty(
            1,
            device=device,
        )
        del probe

        free, total = torch.cuda.mem_get_info(
            index
        )

        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)} | "
            f"free={free/1024**3:.2f} GiB / "
            f"total={total/1024**3:.2f} GiB"
        )

    specs = cent.merged_model_specs(
        twoobj
    )

    if MODEL_KEY not in specs:
        raise RuntimeError(
            f"{MODEL_KEY!r} missing from repo model specs; "
            f"available={sorted(specs)}"
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
            f"does not provide {spec.model_class}"
        )

    dtype = resolve_dtype(
        args.dtype,
        spec,
    )

    kwargs: Dict[
        str,
        Any,
    ] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(
            spec.trust_remote_code
        ),
    }

    print(
        f"repo_id={spec.repo_id} | "
        f"model_class={spec.model_class} | "
        f"dtype={dtype}"
    )

    # NO device_map. Explicit single-GPU placement.
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

    model = model.to(
        device
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
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=bool(
                spec.trust_remote_code
            ),
            use_fast=False,
        )

    except TypeError:
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=bool(
                spec.trust_remote_code
            ),
        )

    cent.configure_processor(
        model,
        processor,
    )

    layers, decoder_path = cent.resolve_decoder_layers(
        model
    )

    print(
        f"decoder={decoder_path} | "
        f"n_layers={len(layers)} | "
        f"first_param_device="
        f"{next(model.parameters()).device}"
    )

    print(
        "="
        * 130
    )

    return (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    )


# =============================================================================
# Batch / generation
# =============================================================================

def make_batch(
    processor: Any,
    image: Image.Image,
    question_text: str,
    args: argparse.Namespace,
) -> Dict[
    str,
    Any,
]:
    return cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
        device=torch.device(
            args.device
        ),
    )


def generate_text(
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
) -> str:
    return cent.generate_text(
        model,
        processor,
        dict(
            batch
        ),
        args.max_new_tokens,
    )


def normalize_prediction(
    text: str,
) -> Optional[str]:
    return cent.normalize_relation(
        text
    )


# =============================================================================
# Fresh synthetic Real-Gray direction extraction
# =============================================================================

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
    batch = make_batch(
        processor,
        image,
        question_text,
        args,
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
            int(
                layer
            ): np.asarray(
                value,
                dtype=np.float32,
            )
            for layer, value
            in capture.last_states.items()
        }

    del batch

    missing = [
        int(
            layer
        )
        for layer
        in actuator_layers
        if int(
            layer
        )
        not in last_states
    ]

    if missing:
        raise RuntimeError(
            f"Missing captured late states={missing}"
        )

    return (
        text,
        normalize_prediction(
            text
        ),
        last_states,
    )


def fit_fresh_synthetic_directions(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    source_records: Sequence[
        Mapping[str, Any]
    ],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    bags = {
        int(
            layer
        ): {
            relation: []
            for relation in RELATIONS
        }
        for layer in actuator_layers
    }

    source_rows = []

    for record in tqdm(
        source_records,
        desc="FRESH synthetic Real-Gray extraction",
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

            relation = record[
                "relation"
            ]

            for layer in actuator_layers:
                layer = int(
                    layer
                )

                delta = (
                    real_last[
                        layer
                    ]
                    - gray_last[
                        layer
                    ]
                ).astype(
                    np.float32
                )

                bags[
                    layer
                ][
                    relation
                ].append(
                    delta
                )

            source_rows.append({
                "sid": int(
                    record[
                        "sid"
                    ]
                ),
                "relation": relation,
                "raw_relation": record[
                    "raw_relation"
                ],
                "subject": record[
                    "subject"
                ],
                "reference": record[
                    "reference"
                ],
                "question_text": record[
                    "question_text"
                ],
                "real_pred": (
                    real_pred
                    or ""
                ),
                "real_correct": int(
                    real_pred
                    == relation
                ),
                "gray_pred": (
                    gray_pred
                    or ""
                ),
                "gray_correct": int(
                    gray_pred
                    == relation
                ),
                "real_text": real_text,
                "gray_text": gray_text,
            })

        except Exception as exc:
            tqdm.write(
                f"[SOURCE ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

            source_rows.append({
                "sid": int(
                    record[
                        "sid"
                    ]
                ),
                "relation": record[
                    "relation"
                ],
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    templates: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for layer in actuator_layers:
        layer = int(
            layer
        )

        relation_mean = {}

        for relation in RELATIONS:
            values = bags[
                layer
            ][
                relation
            ]

            if not values:
                raise RuntimeError(
                    f"No valid source deltas for "
                    f"L{layer} relation={relation}"
                )

            relation_mean[
                relation
            ] = (
                np.stack(
                    values,
                    axis=0,
                )
                .mean(
                    axis=0
                )
                .astype(
                    np.float32
                )
            )

        # Balanced class-common Real-Gray component.
        global_mean = (
            np.stack(
                [
                    relation_mean[
                        relation
                    ]
                    for relation in RELATIONS
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
            for relation in RELATIONS
        }

        templates[
            layer
        ] = {
            "global": global_mean,
            "relation_mean": relation_mean,
            "shared": shared,
        }

    return (
        templates,
        source_rows,
    )


# =============================================================================
# Target baseline / oracle generation
# =============================================================================

def baseline_generate(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
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
        args,
    )

    text = generate_text(
        model,
        processor,
        batch,
        args,
    )

    del batch

    return (
        text,
        normalize_prediction(
            text
        ),
    )


def synthetic_oracle_generate(
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
        args,
    )

    # Explicit oracle:
    # target GT chooses ONLY which already-frozen synthetic vector to add.
    with causal.SteerLast(
        layers,
        templates,
        list(
            actuator_layers
        ),
        record[
            "relation"
        ],
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
        normalize_prediction(
            text
        ),
    )


def evaluate_target(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    target_records: Sequence[
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
    rows = []

    for record in tqdm(
        target_records,
        desc="COCO baseline + fresh synthetic oracle",
    ):
        image = None

        try:
            image = open_coco_image(
                record
            )

            (
                baseline_text,
                baseline_pred,
            ) = baseline_generate(
                model,
                processor,
                image,
                record,
                args,
            )

            baseline_ok = (
                baseline_pred
                == record[
                    "relation"
                ]
            )

            (
                oracle_text,
                oracle_pred,
            ) = synthetic_oracle_generate(
                model,
                processor,
                layers,
                image,
                record,
                templates,
                actuator_layers,
                args,
            )

            oracle_ok = (
                oracle_pred
                == record[
                    "relation"
                ]
            )

            transition = classify_transition(
                baseline_ok,
                oracle_ok,
            )

            rows.append({
                "sid": int(
                    record[
                        "sid"
                    ]
                ),
                "relation": record[
                    "relation"
                ],
                "question_text": record[
                    "question_text"
                ],
                "answer_raw": record[
                    "answer_raw"
                ],

                "baseline_text": baseline_text,
                "baseline_pred": (
                    baseline_pred
                    or ""
                ),
                "baseline_correct": int(
                    baseline_ok
                ),

                "synthetic_oracle_text": oracle_text,
                "synthetic_oracle_pred": (
                    oracle_pred
                    or ""
                ),
                "synthetic_oracle_correct": int(
                    oracle_ok
                ),

                "transition": transition,
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
                "relation": record[
                    "relation"
                ],
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
# Summary
# =============================================================================

def summarize_target(
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
            "No successful COCO target rows."
        )

    baseline_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in good
    )

    oracle_acc = safe_mean(
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
        "baseline_acc": baseline_acc,
        "synthetic_oracle_acc": oracle_acc,
        "gain": (
            oracle_acc
            - baseline_acc
        ),
        "W2C": W2C,
        "C2W": C2W,
        "net": (
            W2C
            - C2W
        ),
        "repair_rate_on_baseline_wrong": safe_mean(
            row[
                "synthetic_oracle_correct"
            ]
            for row in wrong
        ),
        "preserve_rate_on_baseline_correct": safe_mean(
            row[
                "synthetic_oracle_correct"
            ]
            for row in correct
        ),
    }

    per_relation = []

    for relation in RELATIONS:
        subset = [
            row
            for row in good
            if row[
                "relation"
            ]
            == relation
        ]

        per_relation.append({
            "relation": relation,
            "N": len(
                subset
            ),
            "baseline_acc": safe_mean(
                row[
                    "baseline_correct"
                ]
                for row in subset
            ),
            "synthetic_oracle_acc": safe_mean(
                row[
                    "synthetic_oracle_correct"
                ]
                for row in subset
            ),
            "W2C": int(
                sum(
                    row[
                        "W2C"
                    ]
                    for row in subset
                )
            ),
            "C2W": int(
                sum(
                    row[
                        "C2W"
                    ]
                    for row in subset
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

    actuator_layers = parse_layers(
        args.actuator_layers
    )

    # ---------------------------------------------------------------------
    # A. Load COCO metadata / text only.
    # No model forward here.
    # ---------------------------------------------------------------------
    (
        target_records,
        target_audit,
    ) = load_coco_records(
        args
    )

    template_info = infer_dominant_coco_template(
        target_records
    )

    coco_template = template_info[
        "template"
    ]

    # ---------------------------------------------------------------------
    # B. Build synthetic source prompts using exact dominant COCO wording.
    # ---------------------------------------------------------------------
    source_records = load_synthetic_records(
        args,
        coco_template,
    )

    print(
        "\n"
        + "="
        * 150
    )
    print(
        "PROMPT / PROVENANCE AUDIT"
    )
    print(
        "="
        * 150
    )

    print(
        f"Dominant COCO template coverage: "
        f"{template_info['count']}/"
        f"{len(target_records)}"
    )

    print(
        f"Template:\n  {coco_template}"
    )

    print(
        f"Unique normalized COCO templates: "
        f"{template_info['unique_templates']}"
    )

    if template_info[
        "failed_sids"
    ]:
        print(
            f"Template parse failures: "
            f"{template_info['failed_sids'][:20]}"
        )

    print(
        "\nSynthetic relation counts:"
    )

    print(
        dict(
            Counter(
                record[
                    "relation"
                ]
                for record in source_records
            )
        )
    )

    print(
        "\nCOCO relation counts:"
    )

    print(
        dict(
            Counter(
                record[
                    "relation"
                ]
                for record in target_records
            )
        )
    )

    n_examples = min(
        int(
            args.audit_examples
        ),
        len(
            source_records
        ),
        len(
            target_records
        ),
    )

    print(
        "\nSynthetic source questions ACTUALLY used:"
    )

    for record in source_records[
        :n_examples
    ]:
        print(
            f"  SYN sid={record['sid']:3d} | "
            f"GT={record['relation']:>5s} | "
            f"{record['question_text']}"
        )

    print(
        "\nRaw COCO questions ACTUALLY used:"
    )

    for record in target_records[
        :n_examples
    ]:
        print(
            f"  COCO sid={record['sid']:3d} | "
            f"GT={record['relation']:>5s} | "
            f"{record['question_text']}"
        )

    print(
        "\nSynthetic label mapping:"
    )
    print(
        "  left  -> left"
    )
    print(
        "  right -> right"
    )
    print(
        "  on    -> above"
    )
    print(
        "  under -> below"
    )

    print(
        "="
        * 150
    )

    # ---------------------------------------------------------------------
    # C. Fresh Qwen3B.
    # ---------------------------------------------------------------------
    (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    ) = load_model_and_processor(
        args
    )

    for layer in actuator_layers:
        if not (
            0
            <= int(
                layer
            )
            < len(
                layers
            )
        ):
            raise RuntimeError(
                f"Invalid actuator L{layer}; "
                f"model has {len(layers)} decoder blocks."
            )

    # ---------------------------------------------------------------------
    # D. FRESH synthetic direction fit.
    #
    # This occurs BEFORE any COCO model forward.
    # ---------------------------------------------------------------------
    (
        templates,
        source_rows,
    ) = fit_fresh_synthetic_directions(
        model,
        processor,
        layers,
        source_records,
        actuator_layers,
        args,
    )

    write_csv(
        outdir
        / "synthetic_source_generation.csv",
        source_rows,
    )

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

    manifest = []

    print(
        "\n"
        + "="
        * 150
    )
    print(
        "FRESH SYNTHETIC DIRECTION BANK"
    )
    print(
        "="
        * 150
    )

    valid_source_rows = [
        row
        for row in source_rows
        if "error" not in row
    ]

    print(
        f"valid synthetic source rows="
        f"{len(valid_source_rows)}/"
        f"{len(source_records)}"
    )

    for layer in actuator_layers:
        for relation in RELATIONS:
            vector = templates[
                int(
                    layer
                )
            ][
                "shared"
            ][
                relation
            ]

            norm = float(
                np.linalg.norm(
                    vector
                )
            )

            fingerprint = sha256_vector(
                vector
            )

            arrays[
                f"L{layer}_{relation}"
            ] = vector

            manifest.append({
                "source_dataset": str(
                    args.synthetic_dir
                ),
                "layer": int(
                    layer
                ),
                "relation": relation,
                "norm": norm,
                "sha256": fingerprint,
                "fit_uses_coco_hidden_states": False,
            })

            print(
                f"L{layer:02d} {relation:>5s} | "
                f"norm={norm:.6f} | "
                f"sha256={fingerprint[:16]}..."
            )

    np.savez_compressed(
        outdir
        / "fresh_synthetic_direction_bank.npz",
        **arrays,
    )

    write_csv(
        outdir
        / "direction_manifest.csv",
        manifest,
    )

    print(
        "\nDirection source = SYNTHETIC SHAPES ONLY"
    )
    print(
        "COCO hidden states used for direction fit = FALSE"
    )

    print(
        "="
        * 150
    )

    # ---------------------------------------------------------------------
    # E. Only NOW run COCO model forwards.
    # ---------------------------------------------------------------------
    target_rows = evaluate_target(
        model,
        processor,
        layers,
        target_records,
        templates,
        actuator_layers,
        args,
    )

    write_csv(
        outdir
        / "target_details.csv",
        target_rows,
    )

    (
        summary,
        per_relation,
    ) = summarize_target(
        target_rows
    )

    summary.update({
        "model": MODEL_KEY,
        "synthetic_source_N": len(
            source_records
        ),
        "direction_source": str(
            args.synthetic_dir
        ),
        "direction_fit_uses_coco_hidden_states": False,
        "source_prompt_template": coco_template,
        "source_prompt_template_coco_coverage": (
            f"{template_info['count']}/"
            f"{len(target_records)}"
        ),
        "actuator_layers": ",".join(
            str(
                layer
            )
            for layer
            in actuator_layers
        ),
        "scale": args.scale,
    })

    write_csv(
        outdir
        / "summary.csv",
        [
            summary
        ],
    )

    write_csv(
        outdir
        / "per_relation.csv",
        per_relation,
    )

    config = {
        "script": (
            "verify_qwen3b_synthetic_to_coco_oracle_audit_v2.py"
        ),
        "model_alias": MODEL_KEY,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(
            layers
        ),

        "synthetic_source": str(
            args.synthetic_dir
        ),
        "synthetic_source_N": len(
            source_records
        ),

        "loads_previous_direction_cache": False,
        "direction_fit_uses_coco_hidden_states": False,
        "coco_gt_used_for_direction_fit": False,

        "coco_gt_used_for_oracle_selection": True,
        "coco_gt_used_for_metrics": True,

        "dominant_coco_template": coco_template,
        "dominant_coco_template_count": (
            template_info[
                "count"
            ]
        ),
        "coco_N": len(
            target_records
        ),

        "synthetic_uses_dominant_coco_template": True,

        "synthetic_label_mapping": {
            "left": "left",
            "right": "right",
            "on": "above",
            "under": "below",
        },

        "direction_definition": (
            "s_syn(r,l) = "
            "E[h_last(real)-h_last(gray) | relation=r] "
            "- balanced mean of the four relation means"
        ),

        "actuator_layers": actuator_layers,
        "scale": args.scale,
        "gray_value": args.gray_value,
        "seed": args.seed,

        "target_audit": target_audit,
    }

    (
        outdir
        / "audit_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "="
        * 150
    )
    print(
        "AUDITED ACTUAL GREEDY GENERATION: "
        "FRESH SYNTHETIC DIRECTIONS -> COCO"
    )
    print(
        "="
        * 150
    )

    print(
        f"N_TARGET={summary['N']} | "
        f"N_SYNTHETIC={len(source_records)} | "
        f"layers={actuator_layers}"
    )

    print(
        f"COCO baseline              : "
        f"{summary['baseline_acc']:.4f}"
    )

    print(
        f"fresh synthetic oracle     : "
        f"{summary['synthetic_oracle_acc']:.4f} "
        f"({summary['gain']:+.4f}) | "
        f"W2C={summary['W2C']} "
        f"C2W={summary['C2W']} "
        f"net={summary['net']:+d}"
    )

    print(
        f"repair(base-wrong)         : "
        f"{summary['repair_rate_on_baseline_wrong']:.4f}"
    )

    print(
        f"preserve(base-correct)     : "
        f"{summary['preserve_rate_on_baseline_correct']:.4f}"
    )

    print(
        "\nPer relation:"
    )

    for row in per_relation:
        print(
            f"{row['relation']:>5s} | "
            f"N={row['N']:3d} | "
            f"base={row['baseline_acc']:.4f} | "
            f"syn_oracle="
            f"{row['synthetic_oracle_acc']:.4f} | "
            f"W2C/C2W="
            f"{row['W2C']}/{row['C2W']}"
        )

    print(
        "="
        * 150
    )

    print(
        "\nPROVENANCE:"
    )

    print(
        f"  direction source              = "
        f"{args.synthetic_dir}"
    )

    print(
        "  previous direction cache      = FALSE"
    )

    print(
        "  COCO hidden states in fitting = FALSE"
    )

    print(
        f"  synthetic source prompt       = "
        f"{coco_template}"
    )

    print(
        f"  template coverage             = "
        f"{template_info['count']}/"
        f"{len(target_records)} COCO prompts"
    )

    print(
        f"\n[saved] "
        f"{outdir / 'fresh_synthetic_direction_bank.npz'}"
    )
    print(
        f"[saved] "
        f"{outdir / 'direction_manifest.csv'}"
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
        f"{outdir / 'audit_config.json'}"
    )


if __name__ == "__main__":
    main()
