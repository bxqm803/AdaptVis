#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_synthetic_shapes_to_coco_multiselector_multimodel_v1.py

Train EVERYTHING only on the synthetic shapes dataset, then evaluate on the
completely held-out COCO_two target.

Supported model aliases:
    qwen-3b
    qwen-7b
    internvl-2b

SOURCE: synthetic_shapes_4dir_400
================================
Canonical relations:
    left, right, above, below
with:
    on    -> above
    under -> below

The source prompt is forced to the same wording/vocabulary as COCO:
    Where is the {subject} in relation to the {reference}?
    Answer with left, right, above, or below.

Three source-derived components are built.

(1) LATE WRITER
---------------
At model-specific late layers:

    Delta_last(i,l)
      = h_last(real image, i, l)
      - h_last(gray image, i, l)

    mu_r,l = E[Delta_last | relation=r]

    s_r,l = mu_r,l - 1/4 sum_r' mu_r',l

These synthetic directions are the ONLY vectors used for target steering.

Default actuator windows:
    qwen-3b   : L32-L35
    qwen-7b   : L25-L27
    internvl-2b: L21-L23

(2) MIDDLE DIRECTION-SIMILARITY SELECTOR
----------------------------------------
At candidate middle layers:

    p_real(i,l) = h_subject(real) - h_reference(real)
    p_gray(i,l) = h_subject(gray) - h_reference(gray)

    q(i,l) = p_real(i,l) - p_gray(i,l)

For each relation, fit a synthetic relation prototype. We use centered
directions:

    center_l = E[q(i,l)]
    d_r,l = normalize(E[q(i,l)|r] - center_l)

At inference:

    q_test_centered = q_test - center_l
    score(r) = cosine(q_test_centered, d_r,l)
    r_hat = argmax score(r)

The best middle layer is selected ONLY on synthetic data using repeated
stratified fit/validation splits, then prototypes are refit on all synthetic
samples. COCO GT is never used for layer selection or prototype fitting.

Default candidate guide layers:
    automatic middle range = 20%-75% of decoder blocks.

(3) ATTENTION-CENTROID SELECTOR
-------------------------------
On synthetic source only, scan candidate decoder layers and attention heads.

For each sample:
    Q_AB: A relative to B
    Q_BA: B relative to A

Extract subject/reference TEXT-token attention centroids over visual tokens,
align swapped [B,A] back to [A,B], then average:

    c_avg = 0.5 * (c_AB + c_BA_aligned)

Relation rule:
    dx = x_subject - x_reference
    dy = y_subject - y_reference

    if |dx| >= |dy|:
        left if dx < 0 else right
    else:
        above if dy < 0 else below

Choose the best single (layer, head) ONLY by synthetic-source accuracy.
No COCO label is used to choose the head/layer.

Default candidate centroid layers:
    automatic range = 20%-85% of decoder blocks.

TARGET: COCO_two
================
For every target sample, report actual greedy generation for:

    baseline

    oracle:
        COCO GT selects which frozen synthetic late vector to write.
        This is ONLY an upper-bound/control.

    centroid:
        frozen synthetic-selected centroid (layer,head)
        -> predicted relation
        -> frozen synthetic late vector
        -> generation

    middle:
        COCO real-gray middle pair state
        -> cosine against frozen synthetic middle directions
        -> predicted relation
        -> frozen synthetic late vector
        -> generation

Important:
    COCO GT is NOT used by centroid routing.
    COCO GT is NOT used by middle-direction routing.
    COCO GT is used only:
        (a) to score final accuracy/selector accuracy
        (b) for the explicitly labeled oracle control.

Example:
CUDA_VISIBLE_DEVICES=0 python eval_synthetic_shapes_to_coco_multiselector_multimodel_v1.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --output-dir output/syn_to_coco_qwen3b_multiselector_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


# =============================================================================
# Repo imports
# =============================================================================

try:
    import analyze_coco_centroid_generation_step1_v4 as cent
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_centroid_generation_step1_v4.py. "
        "Run this script from the AdaptVis llava16 repository root.\n"
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


RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for r, i in REL_TO_ID.items()}
EPS = 1e-12

SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "above",
    "above": "above",
    "under": "below",
    "below": "below",
}

MODEL_PRESETS = {
    "qwen-3b": {
        "actuator_layers": [32, 33, 34, 35],
    },
    "qwen-7b": {
        "actuator_layers": [25, 26, 27],
    },
    "internvl-2b": {
        "actuator_layers": [21, 22, 23],
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
        choices=sorted(MODEL_PRESETS),
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
        default="auto",
        help="auto uses model preset; otherwise e.g. 32-35 or 21,22,23",
    )

    p.add_argument(
        "--guide-layers",
        default="auto",
        help=(
            "Synthetic middle-selector candidate layers. "
            "auto = all layers from 20%% through 75%% depth."
        ),
    )

    p.add_argument(
        "--centroid-layers",
        default="auto",
        help=(
            "Synthetic centroid head-search candidate layers. "
            "auto = all layers from 20%% through 85%% depth."
        ),
    )

    p.add_argument(
        "--guide-cv-repeats",
        type=int,
        default=20,
    )

    p.add_argument(
        "--guide-cv-fit-frac",
        type=float,
        default=0.70,
    )

    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--gray-value",
        type=int,
        default=128,
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
        "--skip-oracle",
        action="store_true",
    )

    p.add_argument(
        "--skip-centroid",
        action="store_true",
    )

    p.add_argument(
        "--skip-middle",
        action="store_true",
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


def safe_std(values: Iterable[Any]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.std(vals)) if vals else float("nan")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, EPS)


def parse_explicit_layers(
    text: str,
    n_layers: int,
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
            values.extend(range(a, b + step, step))
        else:
            values.append(int(part))

    result = []
    for layer in values:
        if layer < 0:
            layer += n_layers

        if not (0 <= layer < n_layers):
            raise ValueError(
                f"Layer L{layer} outside 0..{n_layers - 1}"
            )

        if layer not in result:
            result.append(layer)

    if not result:
        raise ValueError("Empty layer selection.")

    return result


def auto_fraction_layers(
    n_layers: int,
    low_frac: float,
    high_frac: float,
) -> List[int]:
    low = int(round(low_frac * (n_layers - 1)))
    high = int(round(high_frac * (n_layers - 1)))

    low = max(0, min(low, n_layers - 1))
    high = max(low, min(high, n_layers - 1))

    return list(range(low, high + 1))


def resolve_dtype(
    dtype_name: str,
    spec: Any,
) -> torch.dtype:
    if dtype_name == "auto":
        dtype_name = str(spec.dtype_name)

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype={dtype_name!r}"
        )

    return mapping[dtype_name]


# =============================================================================
# Data
# =============================================================================

def load_synthetic_records(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_dir)

    labels_path = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows = []

    with labels_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            raw_relation = str(
                item["relation"]
            ).strip().lower()

            if raw_relation not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: bad relation={raw_relation!r}"
                )

            relation = SYN_REL_MAP[raw_relation]

            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()

            image_value = Path(str(item["image"]))
            image_path = (
                image_value
                if image_value.is_absolute()
                else root / image_value
            )

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            # Same wording and answer vocabulary as target COCO.
            question_text = (
                f"Where is the {subject} in relation to the {reference}? "
                "Answer with left, right, above, or below."
            )

            rows.append({
                "sid": int(item.get("id", len(rows))),
                "relation": relation,
                "raw_relation": raw_relation,
                "subject": subject,
                "reference": reference,
                "question_text": question_text,
                "image_path": str(image_path),
            })

    rows.sort(key=lambda x: int(x["sid"]))

    if args.source_max_samples is not None:
        rows = rows[: int(args.source_max_samples)]

    if not rows:
        raise RuntimeError("No synthetic records.")

    return rows


def load_coco_records(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Any]:
    raw_records, audit = twoobj.load_records(
        "coco_two",
        Path(args.data_root),
        args.target_max_samples,
    )

    prompt_rows = cent.load_standard_prompts(
        Path(args.prompt_jsonl)
    )

    rows = []

    for raw in raw_records:
        sid = int(raw.sid)

        if sid not in prompt_rows:
            raise RuntimeError(
                f"COCO sid={sid} missing prompt."
            )

        prompt = prompt_rows[sid]

        relation = cent.normalize_relation(
            prompt["answer_raw"]
        )

        if relation not in REL_TO_ID:
            continue

        rows.append({
            "sid": sid,
            "relation": relation,
            "subject": str(prompt["subject"]),
            "reference": str(prompt["reference"]),
            "question_text": str(prompt["question_text"]),
            "raw_record": raw,
        })

    rows.sort(key=lambda x: int(x["sid"]))

    if not rows:
        raise RuntimeError("No COCO target records.")

    return rows, audit


def open_synthetic_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return Image.open(
        record["image_path"]
    ).convert("RGB")


def open_coco_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return cent.record_image(
        record["raw_record"]
    )


# =============================================================================
# Model
# =============================================================================

def load_model(
    args: argparse.Namespace,
):
    specs = cent.merged_model_specs(
        twoobj
    )

    if args.model not in specs:
        raise ValueError(
            f"Model alias {args.model!r} unavailable. "
            f"Repo aliases={sorted(specs)}"
        )

    spec = specs[args.model]

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )

    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no class {spec.model_class}"
        )

    dtype = resolve_dtype(
        args.dtype,
        spec,
    )

    kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(
            spec.trust_remote_code
        ),
        "device_map": {
            "": args.device
        },
        # Required for attention-centroid extraction.
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

    return (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    )


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
    batch: Mapping[str, Any],
    args: argparse.Namespace,
) -> str:
    return cent.generate_text(
        model,
        processor,
        dict(batch),
        args.max_new_tokens,
    )


# =============================================================================
# Hidden states: late writer + middle pair delta
# =============================================================================

def forward_hidden_features(
    model: Any,
    processor: Any,
    image: Image.Image,
    question_text: str,
    subject: str,
    reference: str,
    requested_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    batch = make_batch(
        processor,
        image,
        question_text,
        torch.device(args.device),
    )

    input_ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )

    subject_span, reference_span = cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )

    subject_indices = list(
        range(
            int(subject_span[0]),
            int(subject_span[1]) + 1,
        )
    )

    reference_indices = list(
        range(
            int(reference_span[0]),
            int(reference_span[1]) + 1,
        )
    )

    with torch.inference_mode():
        outputs = model(
            **batch,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    states = cent.hidden_tuple(
        outputs
    )

    if len(states) < 2:
        raise RuntimeError(
            "No decoder hidden states."
        )

    n_blocks = len(states) - 1

    last_states: Dict[int, np.ndarray] = {}
    pair_states: Dict[int, np.ndarray] = {}

    for layer in requested_layers:
        layer = int(layer)

        if not (0 <= layer < n_blocks):
            raise RuntimeError(
                f"L{layer} unavailable; hidden blocks={n_blocks}"
            )

        hidden = states[
            layer + 1
        ][
            0
        ].float()

        if int(hidden.shape[0]) != len(input_ids):
            raise RuntimeError(
                f"L{layer}: hidden seq_len={hidden.shape[0]} "
                f"!= input_ids={len(input_ids)}. "
                "This backend needs merged-token mapping."
            )

        hs = hidden[
            subject_indices
        ].mean(
            dim=0
        )

        hr = hidden[
            reference_indices
        ].mean(
            dim=0
        )

        pair_states[
            layer
        ] = (
            (hs - hr)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        last_states[
            layer
        ] = (
            hidden[
                -1
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    del outputs
    del states
    del batch

    return {
        "pair_states": pair_states,
        "last_states": last_states,
    }


# =============================================================================
# Attention centroid
# =============================================================================

def resolve_attentions(
    outputs: Any,
) -> Tuple[torch.Tensor, ...]:
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
        getattr(
            getattr(
                outputs,
                "text_model_output",
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
            return tuple(
                value
            )

    raise RuntimeError(
        "Forward did not expose decoder attentions."
    )


def attention_centroids(
    model: Any,
    processor: Any,
    image: Image.Image,
    question_text: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    batch = make_batch(
        processor,
        image,
        question_text,
        torch.device(args.device),
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

    subject_span, reference_span = cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )

    subject_index = int(
        subject_span[1]
    )

    reference_index = int(
        reference_span[1]
    )

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
            f"Could not construct visual coordinates "
            f"for {len(visual_indices)} tokens."
        )

    with torch.inference_mode():
        outputs = model(
            **batch,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    attentions = resolve_attentions(
        outputs
    )

    centers = []
    visual_mass = []

    for layer in selected_layers:
        layer = int(layer)

        if not (
            0 <= layer < len(attentions)
        ):
            raise RuntimeError(
                f"Attention L{layer} unavailable; "
                f"returned={len(attentions)}"
            )

        tensor = cent.normalize_attention_tensor(
            attentions[layer],
            expected_query_length=input_length,
        )

        rows = tensor[
            :,
            [
                subject_index,
                reference_index,
            ],
            :,
        ]

        metrics = cent.query_attention_metrics(
            rows,
            visual_indices,
            coords,
            subject_index,
            reference_index,
        )

        centers.append(
            metrics[
                "centroids"
            ][
                :,
                :2,
                :,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        visual_mass.append(
            metrics[
                "visual_mass"
            ][
                :,
                :2,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    del outputs
    del attentions
    del batch

    return {
        "centroids": np.stack(
            centers,
            axis=0,
        ),
        "visual_mass": np.stack(
            visual_mass,
            axis=0,
        ),
    }


def original_swap_centroid_predictions(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    original = attention_centroids(
        model,
        processor,
        image,
        record["question_text"],
        record["subject"],
        record["reference"],
        selected_layers,
        args,
    )

    swapped_question = cent.build_swapped_question(
        record["subject"],
        record["reference"],
    )

    swapped = attention_centroids(
        model,
        processor,
        image,
        swapped_question,
        record["reference"],
        record["subject"],
        selected_layers,
        args,
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

    # Original semantic order: [A,B].
    # Swapped prompt role order: [B,A].
    # Align back to [A,B].
    swapped_aligned = (
        swapped[
            "centroids"
        ][
            :,
            :,
            [1, 0],
            :,
        ]
    )

    average = (
        0.5
        * (
            original[
                "centroids"
            ]
            + swapped_aligned
        )
    ).astype(
        np.float32
    )

    pred, axis_conf = cent.relation_codes_from_centroids(
        average
    )

    orig_pred, _ = cent.relation_codes_from_centroids(
        original[
            "centroids"
        ]
    )

    swap_pred, _ = cent.relation_codes_from_centroids(
        swapped_aligned
    )

    return {
        "prediction": np.asarray(
            pred,
            dtype=np.int8,
        ),
        "axis_confidence": np.asarray(
            axis_conf,
            dtype=np.float32,
        ),
        "original_prediction": np.asarray(
            orig_pred,
            dtype=np.int8,
        ),
        "swapped_prediction": np.asarray(
            swap_pred,
            dtype=np.int8,
        ),
        "visual_mass": (
            0.5
            * (
                original[
                    "visual_mass"
                ]
                + swapped[
                    "visual_mass"
                ][
                    :,
                    :,
                    [1, 0],
                ]
            )
        ).astype(
            np.float32
        ),
    }


# =============================================================================
# Synthetic SOURCE extraction
# =============================================================================

def collect_synthetic_source(
    model: Any,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    actuator_layers: Sequence[int],
    guide_layers: Sequence[int],
    centroid_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    requested_hidden_layers = sorted(
        set(
            list(
                actuator_layers
            )
            + list(
                guide_layers
            )
        )
    )

    late_delta_rows = []
    guide_delta_rows = []
    labels = []
    sids = []

    centroid_predictions = []
    centroid_axis_conf = []
    centroid_consistency = []

    errors: List[Dict[str, Any]] = []

    for record in tqdm(
        records,
        desc=f"SOURCE synthetic:{args.model}",
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

            real_hidden = forward_hidden_features(
                model,
                processor,
                real,
                record["question_text"],
                record["subject"],
                record["reference"],
                requested_hidden_layers,
                args,
            )

            gray_hidden = forward_hidden_features(
                model,
                processor,
                gray,
                record["question_text"],
                record["subject"],
                record["reference"],
                requested_hidden_layers,
                args,
            )

            late_delta_rows.append(
                np.stack(
                    [
                        (
                            real_hidden[
                                "last_states"
                            ][
                                int(layer)
                            ]
                            - gray_hidden[
                                "last_states"
                            ][
                                int(layer)
                            ]
                        )
                        for layer
                        in actuator_layers
                    ],
                    axis=0,
                ).astype(
                    np.float32
                )
            )

            guide_delta_rows.append(
                np.stack(
                    [
                        (
                            real_hidden[
                                "pair_states"
                            ][
                                int(layer)
                            ]
                            - gray_hidden[
                                "pair_states"
                            ][
                                int(layer)
                            ]
                        )
                        for layer
                        in guide_layers
                    ],
                    axis=0,
                ).astype(
                    np.float32
                )
            )

            if not args.skip_centroid:
                centroid_result = (
                    original_swap_centroid_predictions(
                        model,
                        processor,
                        real,
                        record,
                        centroid_layers,
                        args,
                    )
                )

                centroid_predictions.append(
                    centroid_result[
                        "prediction"
                    ]
                )

                centroid_axis_conf.append(
                    centroid_result[
                        "axis_confidence"
                    ]
                )

                centroid_consistency.append(
                    (
                        centroid_result[
                            "original_prediction"
                        ]
                        == centroid_result[
                            "swapped_prediction"
                        ]
                    ).astype(
                        np.int8
                    )
                )

            labels.append(
                record[
                    "relation"
                ]
            )

            sids.append(
                int(
                    record[
                        "sid"
                    ]
                )
            )

        except Exception as exc:
            tqdm.write(
                f"[SOURCE ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append({
                "sid": int(record["sid"]),
                "relation": record["relation"],
                "error": f"{type(exc).__name__}: {exc}",
            })

        finally:
            if real is not None:
                real.close()
            if gray is not None:
                gray.close()
            cleanup()

    if not late_delta_rows:
        raise RuntimeError(
            "No successful synthetic source samples."
        )

    result: Dict[str, Any] = {
        "sid": np.asarray(
            sids,
            dtype=np.int64,
        ),
        "labels": np.asarray(
            labels,
            dtype=object,
        ),
        "late_delta": np.stack(
            late_delta_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "guide_delta": np.stack(
            guide_delta_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "errors": errors,
    }

    if not args.skip_centroid:
        result[
            "centroid_prediction"
        ] = np.stack(
            centroid_predictions,
            axis=0,
        ).astype(
            np.int8
        )

        result[
            "centroid_axis_confidence"
        ] = np.stack(
            centroid_axis_conf,
            axis=0,
        ).astype(
            np.float32
        )

        result[
            "centroid_consistency"
        ] = np.stack(
            centroid_consistency,
            axis=0,
        ).astype(
            np.int8
        )

    return result


# =============================================================================
# Fit synthetic late writer
# =============================================================================

def fit_late_writer(
    late_delta: np.ndarray,
    labels: np.ndarray,
    actuator_layers: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    templates: Dict[int, Dict[str, Any]] = {}

    for lp, layer in enumerate(
        actuator_layers
    ):
        relation_means = {}

        for relation in RELATIONS:
            mask = (
                labels
                == relation
            )

            if int(mask.sum()) == 0:
                raise RuntimeError(
                    f"No source samples for {relation}"
                )

            relation_means[
                relation
            ] = (
                late_delta[
                    mask,
                    lp,
                    :,
                ]
                .mean(
                    axis=0
                )
                .astype(
                    np.float32
                )
            )

        global_mean = (
            np.stack(
                [
                    relation_means[
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
                relation_means[
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
            int(layer)
        ] = {
            "global": global_mean,
            "relation_mean": relation_means,
            "shared": shared,
        }

    return templates


# =============================================================================
# Synthetic middle selector
# =============================================================================

def stratified_fit_val_indices(
    labels: np.ndarray,
    fit_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(
        int(seed)
    )

    fit = []
    val = []

    for relation in RELATIONS:
        ids = np.flatnonzero(
            labels
            == relation
        ).tolist()

        if len(ids) < 2:
            raise RuntimeError(
                f"Need >=2 source samples for {relation}"
            )

        rng.shuffle(
            ids
        )

        n_fit = int(
            round(
                len(ids)
                * float(
                    fit_frac
                )
            )
        )

        n_fit = max(
            1,
            min(
                n_fit,
                len(ids) - 1,
            ),
        )

        fit.extend(
            ids[:n_fit]
        )

        val.extend(
            ids[n_fit:]
        )

    rng.shuffle(fit)
    rng.shuffle(val)

    return (
        np.asarray(
            fit,
            dtype=np.int64,
        ),
        np.asarray(
            val,
            dtype=np.int64,
        ),
    )


def fit_middle_prototypes(
    X: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    X_fit = np.asarray(
        X[
            indices
        ],
        dtype=np.float64,
    )

    center = X_fit.mean(
        axis=0
    )

    directions = []

    for relation in RELATIONS:
        rel_indices = indices[
            labels[
                indices
            ]
            == relation
        ]

        mu = np.asarray(
            X[
                rel_indices
            ],
            dtype=np.float64,
        ).mean(
            axis=0
        )

        direction = (
            mu
            - center
        )

        norm = float(
            np.linalg.norm(
                direction
            )
        )

        if norm <= EPS:
            raise RuntimeError(
                f"Near-zero middle direction for {relation}"
            )

        directions.append(
            direction
            / norm
        )

    return (
        center.astype(
            np.float32
        ),
        np.stack(
            directions,
            axis=0,
        ).astype(
            np.float32
        ),
    )


def middle_scores(
    X: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    Q = np.asarray(
        X,
        dtype=np.float64,
    ) - np.asarray(
        center,
        dtype=np.float64,
    )

    Q = normalize_rows(
        Q
    )

    D = normalize_rows(
        directions
    )

    return (
        Q
        @ D.T
    )


def evaluate_middle(
    X: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> Dict[str, float]:
    scores = middle_scores(
        X[
            indices
        ],
        center,
        directions,
    )

    gt = np.asarray(
        [
            REL_TO_ID[
                str(
                    labels[i]
                )
            ]
            for i
            in indices
        ],
        dtype=np.int64,
    )

    pred = np.argmax(
        scores,
        axis=1,
    )

    rows = np.arange(
        len(gt)
    )

    gt_score = scores[
        rows,
        gt,
    ]

    masked = scores.copy()

    masked[
        rows,
        gt,
    ] = -np.inf

    margin = (
        gt_score
        - np.max(
            masked,
            axis=1,
        )
    )

    result = {
        "accuracy": float(
            np.mean(
                pred
                == gt
            )
        ),
        "gt_margin": float(
            np.mean(
                margin
            )
        ),
    }

    for relation in RELATIONS:
        rid = REL_TO_ID[
            relation
        ]

        mask = (
            gt
            == rid
        )

        result[
            f"{relation}_accuracy"
        ] = (
            float(
                np.mean(
                    pred[
                        mask
                    ]
                    == gt[
                        mask
                    ]
                )
            )
            if int(
                mask.sum()
            )
            > 0
            else float("nan")
        )

    return result


def select_middle_layer_on_synthetic(
    guide_delta: np.ndarray,
    labels: np.ndarray,
    guide_layers: Sequence[int],
    repeats: int,
    fit_frac: float,
    seed: int,
) -> Tuple[
    int,
    np.ndarray,
    np.ndarray,
    List[Dict[str, Any]],
]:
    raw_rows = []

    for repeat in range(
        repeats
    ):
        fit_idx, val_idx = stratified_fit_val_indices(
            labels,
            fit_frac,
            seed + 1000 + repeat,
        )

        for lp, layer in enumerate(
            guide_layers
        ):
            center, directions = (
                fit_middle_prototypes(
                    guide_delta[
                        :,
                        lp,
                        :,
                    ],
                    labels,
                    fit_idx,
                )
            )

            metrics = evaluate_middle(
                guide_delta[
                    :,
                    lp,
                    :,
                ],
                labels,
                val_idx,
                center,
                directions,
            )

            raw_rows.append({
                "repeat": repeat,
                "layer": int(layer),
                "accuracy": metrics["accuracy"],
                "gt_margin": metrics["gt_margin"],
                "left_accuracy": metrics["left_accuracy"],
                "right_accuracy": metrics["right_accuracy"],
                "above_accuracy": metrics["above_accuracy"],
                "below_accuracy": metrics["below_accuracy"],
            })

    summary = []

    for layer in guide_layers:
        rows = [
            row
            for row in raw_rows
            if int(
                row["layer"]
            )
            == int(
                layer
            )
        ]

        summary.append({
            "layer": int(layer),
            "accuracy_mean": safe_mean(
                row["accuracy"]
                for row in rows
            ),
            "accuracy_std": safe_std(
                row["accuracy"]
                for row in rows
            ),
            "gt_margin_mean": safe_mean(
                row["gt_margin"]
                for row in rows
            ),
            "left_accuracy_mean": safe_mean(
                row["left_accuracy"]
                for row in rows
            ),
            "right_accuracy_mean": safe_mean(
                row["right_accuracy"]
                for row in rows
            ),
            "above_accuracy_mean": safe_mean(
                row["above_accuracy"]
                for row in rows
            ),
            "below_accuracy_mean": safe_mean(
                row["below_accuracy"]
                for row in rows
            ),
        })

    summary.sort(
        key=lambda row: (
            -float(
                row[
                    "accuracy_mean"
                ]
            ),
            -float(
                row[
                    "gt_margin_mean"
                ]
            ),
            int(
                row[
                    "layer"
                ]
            ),
        )
    )

    best_layer = int(
        summary[
            0
        ][
            "layer"
        ]
    )

    best_pos = list(
        guide_layers
    ).index(
        best_layer
    )

    all_idx = np.arange(
        len(
            labels
        ),
        dtype=np.int64,
    )

    center, directions = (
        fit_middle_prototypes(
            guide_delta[
                :,
                best_pos,
                :,
            ],
            labels,
            all_idx,
        )
    )

    return (
        best_layer,
        center,
        directions,
        summary,
    )


# =============================================================================
# Synthetic centroid head selection
# =============================================================================

def select_centroid_head_on_synthetic(
    prediction: np.ndarray,
    axis_confidence: np.ndarray,
    consistency: np.ndarray,
    labels: np.ndarray,
    centroid_layers: Sequence[int],
) -> Tuple[
    int,
    int,
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    if prediction.ndim != 3:
        raise RuntimeError(
            f"Expected centroid prediction [N,L,H], got {prediction.shape}"
        )

    gt = np.asarray(
        [
            REL_TO_ID[
                str(
                    value
                )
            ]
            for value
            in labels
        ],
        dtype=np.int64,
    )

    correct = (
        prediction
        == gt[
            :,
            None,
            None,
        ]
    )

    acc = correct.mean(
        axis=0
    )

    conf = axis_confidence.mean(
        axis=0
    )

    cons = consistency.mean(
        axis=0
    )

    ranking = []

    for lp, layer in enumerate(
        centroid_layers
    ):
        for head in range(
            prediction.shape[
                2
            ]
        ):
            row = {
                "layer": int(layer),
                "head": int(head),
                "source_accuracy": float(
                    acc[
                        lp,
                        head,
                    ]
                ),
                "source_axis_confidence": float(
                    conf[
                        lp,
                        head,
                    ]
                ),
                "source_orig_swap_consistency": float(
                    cons[
                        lp,
                        head,
                    ]
                ),
            }

            for relation in RELATIONS:
                rid = REL_TO_ID[
                    relation
                ]

                mask = (
                    gt
                    == rid
                )

                row[
                    f"source_{relation}_accuracy"
                ] = float(
                    correct[
                        mask,
                        lp,
                        head,
                    ].mean()
                )

            ranking.append(
                row
            )

    ranking.sort(
        key=lambda row: (
            -float(
                row[
                    "source_accuracy"
                ]
            ),
            -float(
                row[
                    "source_orig_swap_consistency"
                ]
            ),
            -float(
                row[
                    "source_axis_confidence"
                ]
            ),
            int(
                row[
                    "layer"
                ]
            ),
            int(
                row[
                    "head"
                ]
            ),
        )
    )

    best = dict(
        ranking[
            0
        ]
    )

    return (
        int(
            best[
                "layer"
            ]
        ),
        int(
            best[
                "head"
            ]
        ),
        best,
        ranking,
    )


# =============================================================================
# Target selectors
# =============================================================================

def centroid_predict_target(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    layer: int,
    head: int,
    args: argparse.Namespace,
) -> Tuple[
    str,
    float,
]:
    result = original_swap_centroid_predictions(
        model,
        processor,
        image,
        record,
        [int(layer)],
        args,
    )

    if not (
        0
        <= int(head)
        < result[
            "prediction"
        ].shape[
            1
        ]
    ):
        raise RuntimeError(
            f"Selected centroid H{head} unavailable; "
            f"n_heads={result['prediction'].shape[1]}"
        )

    code = int(
        result[
            "prediction"
        ][
            0,
            int(head),
        ]
    )

    conf = float(
        result[
            "axis_confidence"
        ][
            0,
            int(head),
        ]
    )

    return (
        ID_TO_REL[
            code
        ],
        conf,
    )


def middle_predict_target(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    guide_layer: int,
    center: np.ndarray,
    directions: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[
    str,
    np.ndarray,
    float,
]:
    gray = causal.make_gray(
        image,
        args.gray_value,
    )

    try:
        real_features = forward_hidden_features(
            model,
            processor,
            image,
            record["question_text"],
            record["subject"],
            record["reference"],
            [int(guide_layer)],
            args,
        )

        gray_features = forward_hidden_features(
            model,
            processor,
            gray,
            record["question_text"],
            record["subject"],
            record["reference"],
            [int(guide_layer)],
            args,
        )

        q = (
            real_features[
                "pair_states"
            ][
                int(
                    guide_layer
                )
            ]
            - gray_features[
                "pair_states"
            ][
                int(
                    guide_layer
                )
            ]
        ).astype(
            np.float32
        )

        scores = middle_scores(
            q[
                None,
                :,
            ],
            center,
            directions,
        )[
            0
        ]

        order = np.argsort(
            scores
        )[
            ::-1
        ]

        pred_id = int(
            order[
                0
            ]
        )

        margin = float(
            scores[
                order[
                    0
                ]
            ]
            - scores[
                order[
                    1
                ]
            ]
        )

        return (
            ID_TO_REL[
                pred_id
            ],
            scores.astype(
                np.float32
            ),
            margin,
        )

    finally:
        gray.close()


# =============================================================================
# Steering
# =============================================================================

def steer_generate(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    record: Mapping[str, Any],
    writer_templates: Mapping[
        int,
        Any,
    ],
    actuator_layers: Sequence[int],
    relation: str,
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
        writer_templates,
        list(
            actuator_layers
        ),
        relation,
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
        cent.normalize_relation(
            text
        ),
    )


# =============================================================================
# Target evaluation
# =============================================================================

def evaluate_target(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    records: Sequence[
        Mapping[str, Any]
    ],
    writer_templates: Mapping[
        int,
        Any,
    ],
    actuator_layers: Sequence[int],
    guide_layer: Optional[int],
    guide_center: Optional[np.ndarray],
    guide_directions: Optional[np.ndarray],
    centroid_layer: Optional[int],
    centroid_head: Optional[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows = []

    for record in tqdm(
        records,
        desc=f"TARGET COCO:{args.model}",
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
            # Baseline.
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

            baseline_pred = cent.normalize_relation(
                baseline_text
            )

            baseline_ok = (
                baseline_pred
                == gt
            )

            row: Dict[str, Any] = {
                "sid": int(
                    record[
                        "sid"
                    ]
                ),
                "relation": gt,
                "baseline_text": baseline_text,
                "baseline_pred": baseline_pred or "",
                "baseline_correct": int(
                    baseline_ok
                ),
            }

            # -------------------------------------------------------------
            # Oracle upper-bound.
            # -------------------------------------------------------------
            if not args.skip_oracle:
                oracle_text, oracle_pred = steer_generate(
                    model,
                    processor,
                    layers,
                    image,
                    record,
                    writer_templates,
                    actuator_layers,
                    gt,
                    args,
                )

                oracle_ok = (
                    oracle_pred
                    == gt
                )

                oracle_transition = classify_transition(
                    baseline_ok,
                    oracle_ok,
                )

                row.update({
                    "oracle_text": oracle_text,
                    "oracle_pred": oracle_pred or "",
                    "oracle_correct": int(
                        oracle_ok
                    ),
                    "oracle_transition": oracle_transition,
                })

            # -------------------------------------------------------------
            # Synthetic-selected centroid -> writer.
            # No COCO GT enters the selector.
            # -------------------------------------------------------------
            if not args.skip_centroid:
                assert centroid_layer is not None
                assert centroid_head is not None

                (
                    centroid_pred,
                    centroid_conf,
                ) = centroid_predict_target(
                    model,
                    processor,
                    image,
                    record,
                    centroid_layer,
                    centroid_head,
                    args,
                )

                (
                    centroid_text,
                    centroid_output_pred,
                ) = steer_generate(
                    model,
                    processor,
                    layers,
                    image,
                    record,
                    writer_templates,
                    actuator_layers,
                    centroid_pred,
                    args,
                )

                centroid_selector_ok = (
                    centroid_pred
                    == gt
                )

                centroid_generation_ok = (
                    centroid_output_pred
                    == gt
                )

                centroid_transition = (
                    classify_transition(
                        baseline_ok,
                        centroid_generation_ok,
                    )
                )

                row.update({
                    "centroid_selector_pred": centroid_pred,
                    "centroid_selector_correct": int(
                        centroid_selector_ok
                    ),
                    "centroid_selector_confidence": centroid_conf,
                    "centroid_text": centroid_text,
                    "centroid_output_pred": centroid_output_pred or "",
                    "centroid_correct": int(
                        centroid_generation_ok
                    ),
                    "centroid_transition": centroid_transition,
                })

            # -------------------------------------------------------------
            # Synthetic middle pair-delta cosine selector -> writer.
            # No COCO GT enters the selector.
            # -------------------------------------------------------------
            if not args.skip_middle:
                assert guide_layer is not None
                assert guide_center is not None
                assert guide_directions is not None

                (
                    middle_pred,
                    middle_score_values,
                    middle_margin,
                ) = middle_predict_target(
                    model,
                    processor,
                    image,
                    record,
                    guide_layer,
                    guide_center,
                    guide_directions,
                    args,
                )

                (
                    middle_text,
                    middle_output_pred,
                ) = steer_generate(
                    model,
                    processor,
                    layers,
                    image,
                    record,
                    writer_templates,
                    actuator_layers,
                    middle_pred,
                    args,
                )

                middle_selector_ok = (
                    middle_pred
                    == gt
                )

                middle_generation_ok = (
                    middle_output_pred
                    == gt
                )

                middle_transition = (
                    classify_transition(
                        baseline_ok,
                        middle_generation_ok,
                    )
                )

                row.update({
                    "middle_selector_pred": middle_pred,
                    "middle_selector_correct": int(
                        middle_selector_ok
                    ),
                    "middle_selector_margin": middle_margin,
                    "middle_score_left": float(
                        middle_score_values[
                            REL_TO_ID[
                                "left"
                            ]
                        ]
                    ),
                    "middle_score_right": float(
                        middle_score_values[
                            REL_TO_ID[
                                "right"
                            ]
                        ]
                    ),
                    "middle_score_above": float(
                        middle_score_values[
                            REL_TO_ID[
                                "above"
                            ]
                        ]
                    ),
                    "middle_score_below": float(
                        middle_score_values[
                            REL_TO_ID[
                                "below"
                            ]
                        ]
                    ),
                    "middle_text": middle_text,
                    "middle_output_pred": middle_output_pred or "",
                    "middle_correct": int(
                        middle_generation_ok
                    ),
                    "middle_transition": middle_transition,
                })

            rows.append(
                row
            )

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
# Summaries
# =============================================================================

def method_summary(
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> Dict[str, Any]:
    good = [
        row
        for row in rows
        if (
            "error" not in row
            and f"{method}_correct"
            in row
        )
    ]

    baseline_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in good
    )

    edited_acc = safe_mean(
        row[
            f"{method}_correct"
        ]
        for row in good
    )

    transitions = [
        row[
            f"{method}_transition"
        ]
        for row in good
    ]

    W2C = int(
        sum(
            value
            == "W2C"
            for value
            in transitions
        )
    )

    C2W = int(
        sum(
            value
            == "C2W"
            for value
            in transitions
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

    result = {
        "method": method,
        "N": len(
            good
        ),
        "baseline_acc": baseline_acc,
        "edited_acc": edited_acc,
        "gain": edited_acc - baseline_acc,
        "W2C": W2C,
        "C2W": C2W,
        "net": W2C - C2W,
        "repair_baseline_wrong": safe_mean(
            row[
                f"{method}_correct"
            ]
            for row
            in wrong
        ),
        "preserve_baseline_correct": safe_mean(
            row[
                f"{method}_correct"
            ]
            for row
            in correct
        ),
    }

    if method in (
        "centroid",
        "middle",
    ):
        result[
            "selector_acc"
        ] = safe_mean(
            row[
                f"{method}_selector_correct"
            ]
            for row
            in good
        )

        selector_correct = [
            row
            for row in good
            if int(
                row[
                    f"{method}_selector_correct"
                ]
            )
            == 1
        ]

        selector_wrong = [
            row
            for row in good
            if int(
                row[
                    f"{method}_selector_correct"
                ]
            )
            == 0
        ]

        result[
            "selector_correct_N"
        ] = len(
            selector_correct
        )

        result[
            "selector_wrong_N"
        ] = len(
            selector_wrong
        )

        result[
            "generation_after_when_selector_correct"
        ] = safe_mean(
            row[
                f"{method}_correct"
            ]
            for row
            in selector_correct
        )

        result[
            "generation_after_when_selector_wrong"
        ] = safe_mean(
            row[
                f"{method}_correct"
            ]
            for row
            in selector_wrong
        )

    return result


def per_relation_summary(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    result = []

    good = [
        row
        for row in rows
        if "error" not in row
    ]

    for relation in RELATIONS:
        subset = [
            row
            for row in good
            if row[
                "relation"
            ]
            == relation
        ]

        row_out: Dict[
            str,
            Any,
        ] = {
            "relation": relation,
            "N": len(
                subset
            ),
            "baseline_acc": safe_mean(
                row[
                    "baseline_correct"
                ]
                for row
                in subset
            ),
        }

        for method in methods:
            if not subset:
                continue

            if (
                f"{method}_correct"
                in subset[
                    0
                ]
            ):
                row_out[
                    f"{method}_acc"
                ] = safe_mean(
                    row[
                        f"{method}_correct"
                    ]
                    for row
                    in subset
                )

                row_out[
                    f"{method}_W2C"
                ] = int(
                    sum(
                        row[
                            f"{method}_transition"
                        ]
                        == "W2C"
                        for row
                        in subset
                    )
                )

                row_out[
                    f"{method}_C2W"
                ] = int(
                    sum(
                        row[
                            f"{method}_transition"
                        ]
                        == "C2W"
                        for row
                        in subset
                    )
                )

            if method in (
                "centroid",
                "middle",
            ):
                row_out[
                    f"{method}_selector_acc"
                ] = safe_mean(
                    row[
                        f"{method}_selector_correct"
                    ]
                    for row
                    in subset
                )

        result.append(
            row_out
        )

    return result


def selector_confusion(
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> List[Dict[str, Any]]:
    good = [
        row
        for row in rows
        if (
            "error" not in row
            and f"{method}_selector_pred"
            in row
        )
    ]

    result = []

    for gt in RELATIONS:
        for pred in RELATIONS:
            result.append({
                "method": method,
                "gt": gt,
                "pred": pred,
                "count": int(
                    sum(
                        (
                            row[
                                "relation"
                            ]
                            == gt
                        )
                        and (
                            row[
                                f"{method}_selector_pred"
                            ]
                            == pred
                        )
                        for row
                        in good
                    )
                ),
            })

    return result


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    source_records = load_synthetic_records(
        args
    )

    target_records, target_audit = load_coco_records(
        args
    )

    (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    ) = load_model(
        args
    )

    n_layers = len(
        layers
    )

    # ---------------------------------------------------------------------
    # Model-specific late writer layers.
    # ---------------------------------------------------------------------
    if str(
        args.actuator_layers
    ).strip().lower() == "auto":
        actuator_layers = list(
            MODEL_PRESETS[
                args.model
            ][
                "actuator_layers"
            ]
        )
    else:
        actuator_layers = parse_explicit_layers(
            args.actuator_layers,
            n_layers,
        )

    bad = [
        layer
        for layer in actuator_layers
        if not (
            0 <= int(layer) < n_layers
        )
    ]

    if bad:
        raise RuntimeError(
            f"{args.model}: actuator layers {bad} invalid "
            f"for n_layers={n_layers}. Override --actuator-layers."
        )

    # ---------------------------------------------------------------------
    # Middle candidate layers: source-only selection.
    # ---------------------------------------------------------------------
    if str(
        args.guide_layers
    ).strip().lower() == "auto":
        guide_layers = auto_fraction_layers(
            n_layers,
            0.20,
            0.75,
        )
    else:
        guide_layers = parse_explicit_layers(
            args.guide_layers,
            n_layers,
        )

    # ---------------------------------------------------------------------
    # Centroid candidate layers: source-only selection.
    # ---------------------------------------------------------------------
    if str(
        args.centroid_layers
    ).strip().lower() == "auto":
        centroid_layers = auto_fraction_layers(
            n_layers,
            0.20,
            0.85,
        )
    else:
        centroid_layers = parse_explicit_layers(
            args.centroid_layers,
            n_layers,
        )

    print("\n" + "=" * 160)
    print("SYNTHETIC SHAPES -> COCO | MULTI-SELECTOR / MULTI-MODEL")
    print("=" * 160)

    print(
        f"model={args.model} | repo_id={spec.repo_id}"
    )

    print(
        f"decoder={decoder_path} | n_layers={n_layers} | "
        f"transformers={transformers.__version__}"
    )

    print(
        f"SOURCE synthetic N={len(source_records)} | "
        f"counts={dict(Counter(r['relation'] for r in source_records))}"
    )

    print(
        f"TARGET COCO N={len(target_records)} | "
        f"counts={dict(Counter(r['relation'] for r in target_records))}"
    )

    print(
        f"late writer layers={actuator_layers}"
    )

    print(
        f"middle source-search layers={guide_layers}"
    )

    print(
        f"centroid source-search layers={centroid_layers}"
    )

    print(
        "NO COCO GT is used to fit directions, choose the middle layer, "
        "choose the centroid layer/head, or route the non-oracle methods."
    )

    print("=" * 160)

    # ---------------------------------------------------------------------
    # SOURCE extraction.
    # ---------------------------------------------------------------------
    source = collect_synthetic_source(
        model,
        processor,
        source_records,
        actuator_layers,
        guide_layers,
        centroid_layers,
        args,
    )

    labels = source[
        "labels"
    ]

    if source[
        "errors"
    ]:
        write_csv(
            outdir
            / "source_errors.csv",
            source[
                "errors"
            ],
        )

    # ---------------------------------------------------------------------
    # Fit synthetic late writer.
    # ---------------------------------------------------------------------
    writer_templates = fit_late_writer(
        source[
            "late_delta"
        ],
        labels,
        actuator_layers,
    )

    # ---------------------------------------------------------------------
    # Fit synthetic middle selector.
    # ---------------------------------------------------------------------
    guide_layer = None
    guide_center = None
    guide_directions = None
    guide_cv_summary: List[
        Dict[str, Any]
    ] = []

    if not args.skip_middle:
        (
            guide_layer,
            guide_center,
            guide_directions,
            guide_cv_summary,
        ) = select_middle_layer_on_synthetic(
            source[
                "guide_delta"
            ],
            labels,
            guide_layers,
            args.guide_cv_repeats,
            args.guide_cv_fit_frac,
            args.seed,
        )

        write_csv(
            outdir
            / "synthetic_middle_layer_cv.csv",
            guide_cv_summary,
        )

    # ---------------------------------------------------------------------
    # Select synthetic centroid head.
    # ---------------------------------------------------------------------
    centroid_layer = None
    centroid_head = None
    centroid_best = None
    centroid_ranking: List[
        Dict[str, Any]
    ] = []

    if not args.skip_centroid:
        (
            centroid_layer,
            centroid_head,
            centroid_best,
            centroid_ranking,
        ) = select_centroid_head_on_synthetic(
            source[
                "centroid_prediction"
            ],
            source[
                "centroid_axis_confidence"
            ],
            source[
                "centroid_consistency"
            ],
            labels,
            centroid_layers,
        )

        write_csv(
            outdir
            / "synthetic_centroid_head_ranking.csv",
            centroid_ranking,
        )

    # ---------------------------------------------------------------------
    # Print source-built components before touching target labels.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 160)
    print("SOURCE-ONLY COMPONENTS")
    print("=" * 160)

    for layer in actuator_layers:
        print(
            f"late L{layer:02d} | "
            + " | ".join(
                f"{relation}="
                f"{np.linalg.norm(writer_templates[layer]['shared'][relation]):.4f}"
                for relation in RELATIONS
            )
        )

    if not args.skip_middle:
        assert guide_layer is not None

        best_guide_row = next(
            row
            for row in guide_cv_summary
            if int(row["layer"]) == int(guide_layer)
        )

        print(
            f"\nMIDDLE selector selected on synthetic: "
            f"L{guide_layer:02d} | "
            f"CV acc={best_guide_row['accuracy_mean']:.4f}"
            f"±{best_guide_row['accuracy_std']:.4f} | "
            f"margin={best_guide_row['gt_margin_mean']:+.4f}"
        )

        print(
            "  source per relation="
            f"{best_guide_row['left_accuracy_mean']:.3f}/"
            f"{best_guide_row['right_accuracy_mean']:.3f}/"
            f"{best_guide_row['above_accuracy_mean']:.3f}/"
            f"{best_guide_row['below_accuracy_mean']:.3f}"
        )

    if not args.skip_centroid:
        assert centroid_best is not None

        print(
            f"\nCENTROID selected on synthetic: "
            f"L{centroid_layer:02d}H{centroid_head:02d} | "
            f"source acc={centroid_best['source_accuracy']:.4f} | "
            f"orig/swap consistency="
            f"{centroid_best['source_orig_swap_consistency']:.4f} | "
            f"axis_conf={centroid_best['source_axis_confidence']:.4f}"
        )

        print(
            "  source per relation="
            f"{centroid_best['source_left_accuracy']:.3f}/"
            f"{centroid_best['source_right_accuracy']:.3f}/"
            f"{centroid_best['source_above_accuracy']:.3f}/"
            f"{centroid_best['source_below_accuracy']:.3f}"
        )

    print("=" * 160)

    # ---------------------------------------------------------------------
    # Save source vectors / selectors.
    # ---------------------------------------------------------------------
    arrays: Dict[str, Any] = {
        "relation_order": np.asarray(
            RELATIONS,
            dtype=object,
        ),
        "source_sid": source[
            "sid"
        ],
        "source_relation": labels,
        "actuator_layers": np.asarray(
            actuator_layers,
            dtype=np.int32,
        ),
    }

    for layer in actuator_layers:
        arrays[
            f"writer_L{layer}_global"
        ] = writer_templates[
            layer
        ][
            "global"
        ]

        for relation in RELATIONS:
            arrays[
                f"writer_L{layer}_{relation}"
            ] = writer_templates[
                layer
            ][
                "shared"
            ][
                relation
            ]

    if not args.skip_middle:
        arrays[
            "guide_layer"
        ] = np.asarray(
            int(guide_layer),
            dtype=np.int32,
        )

        arrays[
            "guide_center"
        ] = guide_center

        arrays[
            "guide_directions"
        ] = guide_directions

    if not args.skip_centroid:
        arrays[
            "centroid_layer"
        ] = np.asarray(
            int(centroid_layer),
            dtype=np.int32,
        )

        arrays[
            "centroid_head"
        ] = np.asarray(
            int(centroid_head),
            dtype=np.int32,
        )

    np.savez_compressed(
        outdir
        / "synthetic_source_components.npz",
        **arrays,
    )

    # Release large source arrays before target generation.
    source.pop(
        "late_delta",
        None,
    )

    source.pop(
        "guide_delta",
        None,
    )

    source.pop(
        "centroid_prediction",
        None,
    )

    source.pop(
        "centroid_axis_confidence",
        None,
    )

    source.pop(
        "centroid_consistency",
        None,
    )

    cleanup()

    # ---------------------------------------------------------------------
    # TARGET COCO.
    # ---------------------------------------------------------------------
    target_rows = evaluate_target(
        model,
        processor,
        layers,
        target_records,
        writer_templates,
        actuator_layers,
        guide_layer,
        guide_center,
        guide_directions,
        centroid_layer,
        centroid_head,
        args,
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

    good = [
        row
        for row in target_rows
        if "error" not in row
    ]

    if not good:
        raise RuntimeError(
            "No successful target rows."
        )

    baseline_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in good
    )

    methods = []

    if not args.skip_oracle:
        methods.append(
            "oracle"
        )

    if not args.skip_centroid:
        methods.append(
            "centroid"
        )

    if not args.skip_middle:
        methods.append(
            "middle"
        )

    summaries = [
        method_summary(
            good,
            method,
        )
        for method in methods
    ]

    per_relation = per_relation_summary(
        good,
        methods,
    )

    write_csv(
        outdir
        / "summary.csv",
        summaries,
    )

    write_csv(
        outdir
        / "per_relation.csv",
        per_relation,
    )

    confusion_rows = []

    for method in (
        "centroid",
        "middle",
    ):
        if method in methods:
            confusion_rows.extend(
                selector_confusion(
                    good,
                    method,
                )
            )

    if confusion_rows:
        write_csv(
            outdir
            / "selector_confusion.csv",
            confusion_rows,
        )

    # ---------------------------------------------------------------------
    # Final print.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 160)
    print("ACTUAL GREEDY GENERATION: SYNTHETIC-ONLY COMPONENTS -> COCO")
    print("=" * 160)

    print(
        f"model={args.model} | "
        f"N_TARGET={len(good)} | "
        f"N_SOURCE={len(labels)}"
    )

    print(
        f"COCO baseline                  : "
        f"{baseline_acc:.4f}"
    )

    for summary in summaries:
        method = summary[
            "method"
        ]

        if method == "oracle":
            label = (
                "synthetic oracle"
            )
        elif method == "centroid":
            label = (
                "synthetic centroid selector"
            )
        else:
            label = (
                "synthetic middle selector"
            )

        selector_text = ""

        if "selector_acc" in summary:
            selector_text = (
                f" | selector="
                f"{summary['selector_acc']:.4f}"
            )

        print(
            f"{label:30s}: "
            f"{summary['edited_acc']:.4f} "
            f"({summary['gain']:+.4f})"
            f"{selector_text} | "
            f"W2C={summary['W2C']} "
            f"C2W={summary['C2W']} "
            f"net={summary['net']:+d} | "
            f"repair={summary['repair_baseline_wrong']:.4f} | "
            f"preserve={summary['preserve_baseline_correct']:.4f}"
        )

        if method in (
            "centroid",
            "middle",
        ):
            print(
                f"  selector CORRECT n="
                f"{summary['selector_correct_N']}: "
                f"post-steer generation="
                f"{summary['generation_after_when_selector_correct']:.4f}"
            )

            print(
                f"  selector WRONG   n="
                f"{summary['selector_wrong_N']}: "
                f"post-steer generation="
                f"{summary['generation_after_when_selector_wrong']:.4f}"
            )

    print("\nPer relation:")

    for row in per_relation:
        parts = [
            f"{row['relation']:>5s}",
            f"N={row['N']:3d}",
            f"base={row['baseline_acc']:.4f}",
        ]

        for method in methods:
            if f"{method}_acc" not in row:
                continue

            if method in (
                "centroid",
                "middle",
            ):
                parts.append(
                    f"{method}="
                    f"{row[f'{method}_acc']:.4f}"
                    f"(sel={row[f'{method}_selector_acc']:.4f})"
                )
            else:
                parts.append(
                    f"{method}="
                    f"{row[f'{method}_acc']:.4f}"
                )

        print(
            " | ".join(
                parts
            )
        )

    print("=" * 160)

    # ---------------------------------------------------------------------
    # Metadata.
    # ---------------------------------------------------------------------
    metadata = {
        "script": (
            "eval_synthetic_shapes_to_coco_multiselector_multimodel_v1.py"
        ),
        "model_alias": args.model,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": n_layers,
        "transformers_version": transformers.__version__,

        "source_dataset": str(
            args.synthetic_dir
        ),
        "source_N": int(
            len(
                labels
            )
        ),
        "source_relation_counts": {
            relation: int(
                np.sum(
                    labels
                    == relation
                )
            )
            for relation in RELATIONS
        },

        "target_dataset": (
            "coco_two"
        ),
        "target_N": int(
            len(
                good
            )
        ),

        "target_used_for_fitting": False,

        "target_gt_usage": {
            "oracle_control": (
                not args.skip_oracle
            ),
            "selector_layer_selection": False,
            "centroid_head_selection": False,
            "middle_prototype_fitting": False,
            "centroid_routing": False,
            "middle_routing": False,
            "metric_reporting": True,
        },

        "late_writer": {
            "layers": actuator_layers,
            "scale": args.scale,
            "definition": (
                "Real-Gray last-token relation mean minus "
                "balanced four-relation common mean"
            ),
        },

        "middle_selector": (
            None
            if args.skip_middle
            else {
                "candidate_layers": guide_layers,
                "selected_layer": int(
                    guide_layer
                ),
                "counterfactual": (
                    "Real-Gray object-pair state"
                ),
                "prototype_definition": (
                    "centered synthetic relation mean, unit normalized"
                ),
                "selection": (
                    "synthetic-only repeated stratified CV"
                ),
            }
        ),

        "centroid_selector": (
            None
            if args.skip_centroid
            else {
                "candidate_layers": centroid_layers,
                "selected_layer": int(
                    centroid_layer
                ),
                "selected_head": int(
                    centroid_head
                ),
                "selection": (
                    "synthetic-source original+swap averaged-centroid accuracy"
                ),
            }
        ),

        "gray_value": args.gray_value,
        "seed": args.seed,
        "target_audit": target_audit,
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
        f"{outdir / 'synthetic_source_components.npz'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'synthetic_middle_layer_cv.csv'}"
        if not args.skip_middle
        else "[middle skipped]"
    )

    print(
        f"[saved] "
        f"{outdir / 'synthetic_centroid_head_ranking.csv'}"
        if not args.skip_centroid
        else "[centroid skipped]"
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
