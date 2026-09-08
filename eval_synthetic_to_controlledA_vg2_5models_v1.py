#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_synthetic_to_controlledA_vg2_5models_v1.py

One model per run. Extract all SOURCE components from the same
synthetic_shapes_4dir_400 dataset, then evaluate on:

  1) ControlledA: 4-way left/right/on/under
  2) VG2-LR:      left/right only

Five supported aliases:
  qwen2-2b
  qwen-3b
  qwen-7b
  internvl-1b
  internvl-2b

Read/write definition
=====================

LATE WRITER -- SOURCE ONLY
--------------------------
Synthetic shapes, Real vs constant Gray:

    Delta_i,l = h_last(real)_i,l - h_last(gray)_i,l

    mu_r,l = mean(Delta_i,l | r)

    mu_global,l = 1/4 sum_r mu_r,l

    s_syn(r,l) = mu_r,l - mu_global,l

No target hidden states are used for fitting the late writer.


ORACLE
------
Target TEST GT chooses which frozen synthetic late direction to add.

ControlledA:
    left  -> left
    right -> right
    on    -> above
    under -> below

VG2-LR:
    left/right direct.


MIDDLE -- CORRECTED ORIGINAL LOGIC
---------------------------------
Middle selector is TARGET-DOMAIN TRAIN, NOT synthetic middle prototypes.

For target TRAIN:

    q_i,l =
        [(h_sub - h_ref)_REAL]
        -
        [(h_sub - h_ref)_NOIMAGE]

Fit target relation prototypes.

For target TEST:
    cosine classification -> r_hat
    -> choose frozen synthetic late writer
    -> actual greedy generation

Known Qwen presets:
    qwen-3b: L19
    qwen-7b: L14-L20, mean cosine score

For qwen2-2b / internvl-1b / internvl-2b:
    select top-3 middle layers by inner CV using TARGET TRAIN only.
    TEST labels are never used for layer selection.


CENTROID
--------
Centroid head/layer is selected using SYNTHETIC SOURCE only.

For every synthetic sample:
    original question
    swapped question

Per layer/head:
    average the original and role-aligned swapped object centroids

Pick best (layer, head) by synthetic GT accuracy, freeze it.

On target TEST:
    same frozen (layer, head)
    -> centroid relation prediction
    -> choose frozen synthetic late writer
    -> actual greedy generation

Thus centroid is a full source->target reader+writer transfer, while middle
isolates writer transfer using a target-domain TRAIN reader.


TARGET SPLITS
=============
ControlledA:
    30/70 split by unordered object pair, so layouts of the same pair do not
    cross TRAIN/TEST.

VG2-LR:
    load repo-native four-option VG2
    filter to left/right ONLY
    relation-stratified 30/70 split.


ATTENTION
=========
All models use:
    attn_implementation="eager"

This is required for stable output_attentions used by centroid and keeps the
backend fixed across baseline/oracle/middle/centroid.


COVERAGE
========
By default the script raises after saving errors if fewer than 90% of the
expected TEST samples succeed for a method. This prevents a run such as
N_TARGET=13 from being mistaken for a full-dataset result.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
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
# Repo-native imports only
# =============================================================================

try:
    import analyze_coco_centroid_generation_step1_v4 as cent
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_centroid_generation_step1_v4.py. "
        "Run from the AdaptVis llava16 repository root.\n"
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


SOURCE_RELATIONS = ("left", "right", "above", "below")
SOURCE_REL_TO_ID = {r: i for i, r in enumerate(SOURCE_RELATIONS)}
SOURCE_ID_TO_REL = {i: r for r, i in SOURCE_REL_TO_ID.items()}

SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "above",
    "above": "above",
    "under": "below",
    "below": "below",
}

SYNTHETIC_PROMPT = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with left, right, above, or below."
)

MODEL_PRESETS = {
    # Qwen2-VL-2B has 28 language blocks. No previously established causal
    # window in this project, so start from its final four blocks.
    "qwen2-2b": {
        "actuator_layers": [24, 25, 26, 27],
        "middle_mode": "auto",
        "middle_layers": None,
        "middle_top_k": 3,
    },
    "qwen-3b": {
        "actuator_layers": [32, 33, 34, 35],
        "middle_mode": "fixed",
        "middle_layers": [19],
        "middle_top_k": 1,
    },
    "qwen-7b": {
        "actuator_layers": [25, 26, 27],
        "middle_mode": "fixed",
        "middle_layers": [14, 15, 16, 17, 18, 19, 20],
        "middle_top_k": 7,
    },
    "internvl-1b": {
        "actuator_layers": [20, 21, 22, 23],
        "middle_mode": "auto",
        "middle_layers": None,
        "middle_top_k": 3,
    },
    "internvl-2b": {
        "actuator_layers": [21, 22, 23],
        "middle_mode": "auto",
        "middle_layers": None,
        "middle_top_k": 3,
    },
}


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
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
    p.add_argument("--synthetic-labels", default=None)

    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--controlled-json",
        default="data/controlled_images_dataset.json",
    )

    p.add_argument(
        "--vg-json",
        default="data/vg_qa_two_obj_four_options.json",
    )
    p.add_argument(
        "--vg-prompt-jsonl",
        default="prompts/VG_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--vg-image-root",
        default="auto",
    )

    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )

    p.add_argument(
        "--actuator-layers",
        default="preset",
    )

    p.add_argument(
        "--middle-layers",
        default="preset",
        help="preset, auto, or e.g. 19 / 14-20",
    )

    p.add_argument(
        "--middle-top-k",
        default="preset",
    )

    p.add_argument(
        "--middle-search-layers",
        default="auto",
        help="auto = 20%-75% decoder depth",
    )

    p.add_argument(
        "--middle-cv-repeats",
        type=int,
        default=20,
    )
    p.add_argument(
        "--middle-cv-fit-frac",
        type=float,
        default=0.70,
    )

    p.add_argument(
        "--centroid-layers",
        default="auto",
        help="auto = 20%-85% decoder depth; selected using synthetic source only",
    )

    p.add_argument(
        "--centroid-temperature",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.30,
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)

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
        "--min-success-rate",
        type=float,
        default=0.90,
    )

    p.add_argument(
        "--datasets",
        default="controlled_a,vg2_lr",
        help="Comma-separated subset of controlled_a,vg2_lr",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_layer_range(text: str, n_layers: int):
    values = []

    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            values.extend(range(a, b + step, step))
        else:
            values.append(int(part))

    out = []
    for layer in values:
        if layer < 0:
            layer += n_layers

        if not (0 <= layer < n_layers):
            raise ValueError(
                f"L{layer} outside decoder range 0..{n_layers-1}"
            )

        if layer not in out:
            out.append(layer)

    if not out:
        raise ValueError("Empty layer set.")

    return out


def auto_layer_band(n_layers, lo_frac, hi_frac):
    lo = int(round(lo_frac * (n_layers - 1)))
    hi = int(round(hi_frac * (n_layers - 1)))

    lo = max(0, lo)
    hi = min(n_layers - 1, hi)

    return list(range(lo, hi + 1))


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def relation_counts(records):
    return dict(Counter(r["relation"] for r in records))


def target_labels(dataset):
    if dataset == "controlled_a":
        return ("left", "right", "on", "under")

    if dataset == "vg2_lr":
        return ("left", "right")

    raise ValueError(dataset)


def target_answer_words(dataset):
    return target_labels(dataset)


def target_to_source_relation(dataset, relation):
    if dataset == "controlled_a":
        return {
            "left": "left",
            "right": "right",
            "on": "above",
            "under": "below",
        }[relation]

    if dataset == "vg2_lr":
        return relation

    raise ValueError(dataset)


def source_to_target_relation(dataset, relation):
    if dataset == "controlled_a":
        return {
            "left": "left",
            "right": "right",
            "above": "on",
            "below": "under",
        }.get(relation)

    if dataset == "vg2_lr":
        return relation if relation in ("left", "right") else None

    raise ValueError(dataset)


def classify_transition(before_ok, after_ok):
    if not before_ok and after_ok:
        return "W2C"
    if before_ok and not after_ok:
        return "C2W"
    if before_ok and after_ok:
        return "C2C"
    return "W2W"


# =============================================================================
# Synthetic source
# =============================================================================

def load_synthetic(args):
    root = Path(args.synthetic_dir)

    labels_path = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows = []

    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            raw_rel = str(item["relation"]).strip().lower()

            if raw_rel not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: bad relation={raw_rel!r}"
                )

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

            rows.append({
                "sid": int(item.get("id", len(rows))),
                "relation": SYN_REL_MAP[raw_rel],
                "raw_relation": raw_rel,
                "subject": subject,
                "reference": reference,
                "question_text": SYNTHETIC_PROMPT.format(
                    subject=subject,
                    reference=reference,
                ),
                "image_path": str(image_path),
            })

    rows.sort(key=lambda x: int(x["sid"]))

    if args.source_max_samples is not None:
        rows = rows[: int(args.source_max_samples)]

    return rows


def open_image_path(record):
    return Image.open(record["image_path"]).convert("RGB")


# =============================================================================
# Target loaders / splits
# =============================================================================

def load_target_dataset(dataset, args):
    # The repo-native helper expects args.dataset for relation parsing.
    if dataset == "controlled_a":
        args.dataset = "controlled_a"
        records = causal.load_controlled_a(args)

        if args.target_max_samples is not None:
            records = records[: int(args.target_max_samples)]

        train, test = causal.split_controlled_by_pair(
            records,
            args.train_frac,
            args.seed,
        )

        return records, train, test

    if dataset == "vg2_lr":
        args.dataset = "vg2"
        records = causal.load_vg2(args)

        # User-requested VG2 setting: LEFT / RIGHT ONLY.
        records = [
            r for r in records
            if r["relation"] in ("left", "right")
        ]

        if args.target_max_samples is not None:
            records = records[: int(args.target_max_samples)]

        train, test = causal.split_stratified(
            records,
            ("left", "right"),
            args.train_frac,
            args.seed,
        )

        return records, train, test

    raise ValueError(dataset)


# =============================================================================
# Model loading
# =============================================================================

def resolve_dtype(name, spec):
    if name == "auto":
        name = str(spec.dtype_name)

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(args):
    device = torch.device(args.device)

    print("\n" + "=" * 120)
    print("MODEL / CUDA")
    print("=" * 120)

    print(
        f"torch={torch.__version__} | "
        f"cuda={torch.version.cuda} | "
        f"available={torch.cuda.is_available()} | "
        f"count={torch.cuda.device_count()} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")

        idx = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(idx)

        probe = torch.empty(1, device=device)
        del probe

        print(f"GPU={torch.cuda.get_device_name(idx)}")

    specs = cent.merged_model_specs(twoobj)

    if args.model not in specs:
        raise RuntimeError(
            f"Model alias={args.model!r} missing. Available={sorted(specs)}"
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
            f"does not expose {spec.model_class}"
        )

    dtype = resolve_dtype(args.dtype, spec)

    kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(spec.trust_remote_code),

        # FIXED across all methods/models because centroid requires
        # output_attentions and baseline comparability requires one backend.
        "attn_implementation": "eager",
    }

    print(
        f"alias={args.model} | repo={spec.repo_id} | "
        f"class={spec.model_class} | dtype={dtype} | attn=eager"
    )

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

    model = model.to(device)
    model.eval()

    gc_cfg = getattr(model, "generation_config", None)

    if gc_cfg is not None:
        if hasattr(gc_cfg, "do_sample"):
            gc_cfg.do_sample = False

        if hasattr(gc_cfg, "num_beams"):
            gc_cfg.num_beams = 1

        for field in ("temperature", "top_p", "top_k"):
            if hasattr(gc_cfg, field):
                setattr(gc_cfg, field, None)

    try:
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=bool(spec.trust_remote_code),
            use_fast=False,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=bool(spec.trust_remote_code),
        )

    cent.configure_processor(model, processor)

    layers, decoder_path = cent.resolve_decoder_layers(model)

    print(
        f"decoder={decoder_path} | n_layers={len(layers)} | "
        f"device={next(model.parameters()).device}"
    )

    print("=" * 120)

    return model, processor, layers, spec, decoder_path


# =============================================================================
# Layer presets
# =============================================================================

def resolve_actuator_layers(args, n_layers):
    if args.actuator_layers == "preset":
        requested = MODEL_PRESETS[args.model]["actuator_layers"]
    else:
        return parse_layer_range(args.actuator_layers, n_layers)

    for layer in requested:
        if not (0 <= layer < n_layers):
            raise RuntimeError(
                f"Preset actuator L{layer} invalid for {args.model} "
                f"with n_layers={n_layers}"
            )

    return list(requested)


def resolve_middle_feature_layers(args, n_layers):
    preset = MODEL_PRESETS[args.model]

    if args.middle_layers == "preset":
        if preset["middle_mode"] == "fixed":
            return (
                "fixed",
                list(preset["middle_layers"]),
                int(preset["middle_top_k"]),
            )

        return (
            "auto",
            auto_layer_band(n_layers, 0.20, 0.75),
            int(preset["middle_top_k"]),
        )

    if args.middle_layers == "auto":
        top_k = (
            int(preset["middle_top_k"])
            if args.middle_top_k == "preset"
            else int(args.middle_top_k)
        )

        if args.middle_search_layers == "auto":
            layers = auto_layer_band(n_layers, 0.20, 0.75)
        else:
            layers = parse_layer_range(
                args.middle_search_layers,
                n_layers,
            )

        return "auto", layers, top_k

    layers = parse_layer_range(
        args.middle_layers,
        n_layers,
    )

    return "fixed", layers, len(layers)


def resolve_centroid_layers(args, n_layers):
    if args.centroid_layers == "auto":
        return auto_layer_band(
            n_layers,
            0.20,
            0.85,
        )

    return parse_layer_range(
        args.centroid_layers,
        n_layers,
    )


# =============================================================================
# Synthetic late writer
# =============================================================================

def source_batch(processor, image, question_text, args):
    return cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
        device=torch.device(args.device),
    )


def source_capture_last(
    model,
    processor,
    layers,
    image,
    record,
    actuator_layers,
    args,
):
    batch = source_batch(
        processor,
        image,
        record["question_text"],
        args,
    )

    with causal.CaptureStates(
        layers,
        list(actuator_layers),
        [],
        [],
        [],
    ) as capture:
        cent.generate_text(
            model,
            processor,
            batch,
            args.max_new_tokens,
        )

        states = {
            int(layer): np.asarray(value, dtype=np.float32)
            for layer, value in capture.last_states.items()
        }

    del batch

    missing = [
        layer for layer in actuator_layers
        if layer not in states
    ]

    if missing:
        raise RuntimeError(
            f"Missing source late captures={missing}"
        )

    return states


def fit_synthetic_writer(
    model,
    processor,
    layers,
    source_records,
    actuator_layers,
    args,
):
    bags = {
        layer: {
            rel: []
            for rel in SOURCE_RELATIONS
        }
        for layer in actuator_layers
    }

    errors = []

    for record in tqdm(
        source_records,
        desc=f"source-writer:{args.model}",
    ):
        real = None
        gray = None

        try:
            real = open_image_path(record)
            gray = causal.make_gray(
                real,
                args.gray_value,
            )

            real_last = source_capture_last(
                model,
                processor,
                layers,
                real,
                record,
                actuator_layers,
                args,
            )

            gray_last = source_capture_last(
                model,
                processor,
                layers,
                gray,
                record,
                actuator_layers,
                args,
            )

            rel = record["relation"]

            for layer in actuator_layers:
                bags[layer][rel].append(
                    (
                        real_last[layer]
                        - gray_last[layer]
                    ).astype(np.float32)
                )

        except Exception as exc:
            errors.append({
                "sid": record["sid"],
                "relation": record["relation"],
                "stage": "source_writer",
                "error": f"{type(exc).__name__}: {exc}",
            })

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    templates = {}

    for layer in actuator_layers:
        relation_mean = {}

        for rel in SOURCE_RELATIONS:
            values = bags[layer][rel]

            if not values:
                raise RuntimeError(
                    f"No synthetic writer vectors L{layer}/{rel}"
                )

            relation_mean[rel] = (
                np.stack(values, axis=0)
                .mean(axis=0)
                .astype(np.float32)
            )

        global_mean = (
            np.stack(
                [relation_mean[r] for r in SOURCE_RELATIONS],
                axis=0,
            )
            .mean(axis=0)
            .astype(np.float32)
        )

        templates[layer] = {
            "global": global_mean,
            "relation_mean": relation_mean,
            "shared": {
                rel: (
                    relation_mean[rel]
                    - global_mean
                ).astype(np.float32)
                for rel in SOURCE_RELATIONS
            },
        }

    return templates, errors


# =============================================================================
# Synthetic centroid head selection
# =============================================================================

def source_centroid_prediction(
    model,
    processor,
    image,
    record,
    selected_layers,
    relation_token_map,
    args,
):
    original_batch = cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=record["question_text"],
        device=torch.device(args.device),
    )

    swapped_question = cent.build_swapped_question(
        record["subject"],
        record["reference"],
    )

    swapped_batch = cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=swapped_question,
        device=torch.device(args.device),
    )

    original = cent.analyze_prompt(
        model=model,
        processor=processor,
        batch=original_batch,
        question_text=record["question_text"],
        subject=record["subject"],
        reference=record["reference"],
        selected_layers=selected_layers,
        temperature=args.centroid_temperature,
        max_new_tokens=max(2, args.max_new_tokens),
        relation_token_map=relation_token_map,
        gt=record["relation"],
    )

    swapped = cent.analyze_prompt(
        model=model,
        processor=processor,
        batch=swapped_batch,
        question_text=swapped_question,
        subject=record["reference"],
        reference=record["subject"],
        selected_layers=selected_layers,
        temperature=args.centroid_temperature,
        max_new_tokens=max(2, args.max_new_tokens),
        relation_token_map=relation_token_map,
        gt=cent.invert_relation(record["relation"]),
    )

    swapped_aligned = (
        swapped["object_centroids"][
            :,
            :,
            [1, 0],
            :
        ]
    )

    avg_centroids = 0.5 * (
        original["object_centroids"]
        + swapped_aligned
    )

    codes, _ = cent.relation_codes_from_centroids(
        avg_centroids
    )

    del original_batch
    del swapped_batch
    del original
    del swapped

    return codes


def fit_synthetic_centroid_head(
    model,
    processor,
    source_records,
    centroid_layers,
    args,
):
    relation_token_map = cent.relation_token_variants(
        processor.tokenizer
    )

    correct = None
    count = None
    n_heads = None

    errors = []

    for record in tqdm(
        source_records,
        desc=f"source-centroid:{args.model}",
    ):
        image = None

        try:
            image = open_image_path(record)

            codes = source_centroid_prediction(
                model,
                processor,
                image,
                record,
                centroid_layers,
                relation_token_map,
                args,
            )

            # [n_layers_selected, n_heads]
            if codes.ndim != 2:
                raise RuntimeError(
                    f"Expected centroid codes [L,H], got {codes.shape}"
                )

            if correct is None:
                n_heads = int(codes.shape[1])

                correct = np.zeros(
                    (len(centroid_layers), n_heads),
                    dtype=np.int64,
                )

                count = np.zeros_like(correct)

            if int(codes.shape[1]) != n_heads:
                raise RuntimeError(
                    f"Head count changed {codes.shape[1]} vs {n_heads}"
                )

            gt_code = SOURCE_REL_TO_ID[
                record["relation"]
            ]

            correct += (
                codes
                == gt_code
            ).astype(np.int64)

            count += 1

        except Exception as exc:
            errors.append({
                "sid": record["sid"],
                "relation": record["relation"],
                "stage": "source_centroid",
                "error": f"{type(exc).__name__}: {exc}",
            })

        finally:
            if image is not None:
                image.close()

            cleanup()

    if correct is None:
        raise RuntimeError(
            "No valid synthetic centroid samples."
        )

    acc = correct / np.maximum(count, 1)

    ranking = []

    for lp, layer in enumerate(centroid_layers):
        for head in range(n_heads):
            ranking.append({
                "layer": int(layer),
                "head": int(head),
                "accuracy": float(acc[lp, head]),
                "correct": int(correct[lp, head]),
                "count": int(count[lp, head]),
            })

    ranking.sort(
        key=lambda r: (
            -r["accuracy"],
            r["layer"],
            r["head"],
        )
    )

    best = ranking[0]

    return (
        int(best["layer"]),
        int(best["head"]),
        ranking,
        errors,
    )


# =============================================================================
# Target prompt / batch
# =============================================================================

def set_causal_dataset_arg(args, dataset):
    args.dataset = (
        "controlled_a"
        if dataset == "controlled_a"
        else "vg2"
    )


def target_question(record, dataset):
    if dataset == "vg2_lr":
        # Preserve exact repo-native VG2 prompt.
        return record["question_text"]

    words = target_answer_words(dataset)

    answer_text = (
        ", ".join(words[:-1])
        + f", or {words[-1]}"
    )

    return (
        f"Determine the spatial relation of the {record['subject']} "
        f"to the {record['reference']} in the image. "
        f"Answer with {answer_text}."
    )


def target_real_batch(
    processor,
    image,
    record,
    dataset,
    args,
):
    set_causal_dataset_arg(
        args,
        dataset,
    )

    # causal.build_batch preserves exact VG2 prompt and uses the same
    # ControlledA prompt format as prior cross-dataset experiments.
    return causal.build_batch(
        processor,
        image,
        record,
        args,
        target_answer_words(dataset),
    )


def move_batch(batch, device):
    return {
        k: (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in batch.items()
    }


def target_noimage_batch(
    processor,
    record,
    dataset,
    args,
):
    question = target_question(
        record,
        dataset,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                }
            ],
        }
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = question

    errors = []

    for fn in (
        lambda: processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            return_tensors="pt",
        ),
    ):
        try:
            batch = fn()

            return move_batch(
                batch,
                torch.device(args.device),
            )

        except Exception as exc:
            errors.append(exc)

    raise RuntimeError(
        f"NoImage processor failed: "
        f"{type(errors[-1]).__name__}: {errors[-1]}"
    )


# =============================================================================
# Middle feature: Real - NoImage object-pair state
# =============================================================================

def pair_forward_from_batch(
    model,
    processor,
    layers,
    batch,
    record,
    selected_layers,
):
    ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )

    subject_span, reference_span = causal.locate_pair_spans(
        processor.tokenizer,
        ids,
        record["subject"],
        record["reference"],
    )

    if (
        subject_span is None
        or reference_span is None
    ):
        raise RuntimeError(
            "Could not locate object tokens: "
            f"{record['subject']!r} / {record['reference']!r}"
        )

    with causal.CaptureStates(
        layers,
        [],
        list(selected_layers),
        subject_span,
        reference_span,
    ) as capture:
        with torch.inference_mode():
            model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

        states = {
            int(layer): np.asarray(value, dtype=np.float32)
            for layer, value in capture.pair_states.items()
        }

    missing = [
        layer for layer in selected_layers
        if int(layer) not in states
    ]

    if missing:
        raise RuntimeError(
            f"Missing pair states={missing}"
        )

    return states


def target_middle_q(
    model,
    processor,
    layers,
    image,
    record,
    dataset,
    selected_layers,
    args,
):
    real_batch = target_real_batch(
        processor,
        image,
        record,
        dataset,
        args,
    )

    noimg_batch = target_noimage_batch(
        processor,
        record,
        dataset,
        args,
    )

    try:
        real_pair = pair_forward_from_batch(
            model,
            processor,
            layers,
            real_batch,
            record,
            selected_layers,
        )

        noimg_pair = pair_forward_from_batch(
            model,
            processor,
            layers,
            noimg_batch,
            record,
            selected_layers,
        )

        return {
            layer: (
                real_pair[layer]
                - noimg_pair[layer]
            ).astype(np.float32)
            for layer in selected_layers
        }

    finally:
        del real_batch
        del noimg_batch


def collect_middle_train_features(
    model,
    processor,
    layers,
    records,
    dataset,
    feature_layers,
    args,
):
    features = []
    labels = []
    sids = []
    errors = []

    for record in tqdm(
        records,
        desc=f"target-middle-train:{dataset}:{args.model}",
    ):
        image = None

        try:
            image = open_image_path(record)

            q = target_middle_q(
                model,
                processor,
                layers,
                image,
                record,
                dataset,
                feature_layers,
                args,
            )

            features.append(
                np.stack(
                    [q[layer] for layer in feature_layers],
                    axis=0,
                ).astype(np.float32)
            )

            labels.append(record["relation"])
            sids.append(int(record["sid"]))

        except Exception as exc:
            errors.append({
                "sid": record["sid"],
                "relation": record["relation"],
                "stage": "middle_train",
                "error": f"{type(exc).__name__}: {exc}",
            })

        finally:
            if image is not None:
                image.close()

            cleanup()

    if not features:
        raise RuntimeError(
            f"No middle TRAIN features for {dataset}"
        )

    return (
        np.stack(features, axis=0),
        np.asarray(labels, dtype=object),
        np.asarray(sids, dtype=np.int64),
        errors,
    )


def fit_one_middle_layer(
    X,
    labels,
    indices,
    label_order,
):
    X_fit = np.asarray(
        X[indices],
        dtype=np.float64,
    )

    center = X_fit.mean(axis=0)

    dirs = []

    for rel in label_order:
        rel_idx = indices[
            labels[indices] == rel
        ]

        if len(rel_idx) == 0:
            raise RuntimeError(
                f"No middle fit samples for {rel}"
            )

        mu = (
            np.asarray(
                X[rel_idx],
                dtype=np.float64,
            )
            .mean(axis=0)
        )

        d = mu - center
        norm = np.linalg.norm(d)

        if norm <= 1e-12:
            raise RuntimeError(
                f"Near-zero middle direction for {rel}"
            )

        dirs.append(d / norm)

    return (
        center.astype(np.float32),
        np.stack(dirs, axis=0).astype(np.float32),
    )


def middle_layer_scores(X, center, directions):
    q = normalize_rows(
        np.asarray(X, dtype=np.float64)
        - np.asarray(center, dtype=np.float64)
    )

    dirs = normalize_rows(
        np.asarray(directions, dtype=np.float64)
    )

    return q @ dirs.T


def stratified_inner_split(
    labels,
    label_order,
    fit_frac,
    seed,
):
    rng = random.Random(seed)

    fit = []
    val = []

    for rel in label_order:
        ids = np.flatnonzero(
            labels == rel
        ).tolist()

        if len(ids) < 2:
            raise RuntimeError(
                f"Need >=2 middle TRAIN samples for {rel}"
            )

        rng.shuffle(ids)

        n_fit = int(round(len(ids) * fit_frac))
        n_fit = max(1, min(n_fit, len(ids) - 1))

        fit.extend(ids[:n_fit])
        val.extend(ids[n_fit:])

    rng.shuffle(fit)
    rng.shuffle(val)

    return (
        np.asarray(fit, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
    )


def select_middle_layers_train_only(
    features,
    labels,
    feature_layers,
    label_order,
    top_k,
    repeats,
    fit_frac,
    seed,
):
    rows = []

    rel_to_id = {
        rel: i
        for i, rel in enumerate(label_order)
    }

    for repeat in range(repeats):
        fit_idx, val_idx = stratified_inner_split(
            labels,
            label_order,
            fit_frac,
            seed + 1000 + repeat,
        )

        gt = np.asarray(
            [
                rel_to_id[str(labels[i])]
                for i in val_idx
            ],
            dtype=np.int64,
        )

        for lp, layer in enumerate(feature_layers):
            center, directions = fit_one_middle_layer(
                features[:, lp, :],
                labels,
                fit_idx,
                label_order,
            )

            scores = middle_layer_scores(
                features[val_idx, lp, :],
                center,
                directions,
            )

            pred = np.argmax(scores, axis=1)

            rows.append({
                "repeat": repeat,
                "layer": int(layer),
                "accuracy": float(
                    np.mean(pred == gt)
                ),
            })

    summary = []

    for layer in feature_layers:
        vals = [
            r["accuracy"]
            for r in rows
            if r["layer"] == int(layer)
        ]

        summary.append({
            "layer": int(layer),
            "accuracy_mean": safe_mean(vals),
            "accuracy_std": (
                float(np.std(vals))
                if vals
                else float("nan")
            ),
        })

    summary.sort(
        key=lambda r: (
            -r["accuracy_mean"],
            r["layer"],
        )
    )

    selected = [
        int(r["layer"])
        for r in summary[:top_k]
    ]

    return selected, summary


def fit_middle_selector(
    features,
    labels,
    feature_layers,
    selected_layers,
    label_order,
):
    layer_pos = {
        layer: i
        for i, layer in enumerate(feature_layers)
    }

    all_idx = np.arange(
        len(labels),
        dtype=np.int64,
    )

    selector = {}

    for layer in selected_layers:
        lp = layer_pos[layer]

        center, directions = fit_one_middle_layer(
            features[:, lp, :],
            labels,
            all_idx,
            label_order,
        )

        selector[layer] = {
            "center": center,
            "directions": directions,
        }

    return selector


def predict_middle(
    q_by_layer,
    selector,
    label_order,
):
    scores_all = []

    for layer, params in selector.items():
        scores = middle_layer_scores(
            np.asarray(q_by_layer[layer])[None, :],
            params["center"],
            params["directions"],
        )[0]

        scores_all.append(scores)

    mean_scores = np.stack(
        scores_all,
        axis=0,
    ).mean(axis=0)

    order = np.argsort(mean_scores)[::-1]

    pred = label_order[int(order[0])]

    margin = float(
        mean_scores[order[0]]
        - mean_scores[order[1]]
    ) if len(order) >= 2 else float("nan")

    return pred, mean_scores, margin


# =============================================================================
# Target centroid at frozen source-selected head
# =============================================================================

def target_centroid_predict(
    model,
    processor,
    image,
    record,
    dataset,
    centroid_layer,
    centroid_head,
    relation_token_map,
    args,
):
    question = target_question(
        record,
        dataset,
    )

    original_batch = cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=question,
        device=torch.device(args.device),
    )

    swapped_question = cent.build_swapped_question(
        record["subject"],
        record["reference"],
    )

    swapped_batch = cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=swapped_question,
        device=torch.device(args.device),
    )

    source_gt = target_to_source_relation(
        dataset,
        record["relation"],
    )

    original = cent.analyze_prompt(
        model=model,
        processor=processor,
        batch=original_batch,
        question_text=question,
        subject=record["subject"],
        reference=record["reference"],
        selected_layers=[centroid_layer],
        temperature=args.centroid_temperature,
        max_new_tokens=max(2, args.max_new_tokens),
        relation_token_map=relation_token_map,
        gt=source_gt,
    )

    swapped = cent.analyze_prompt(
        model=model,
        processor=processor,
        batch=swapped_batch,
        question_text=swapped_question,
        subject=record["reference"],
        reference=record["subject"],
        selected_layers=[centroid_layer],
        temperature=args.centroid_temperature,
        max_new_tokens=max(2, args.max_new_tokens),
        relation_token_map=relation_token_map,
        gt=cent.invert_relation(source_gt),
    )

    swapped_aligned = (
        swapped["object_centroids"][
            0,
            centroid_head,
            [1, 0],
            :
        ]
    )

    original_centroids = (
        original["object_centroids"][
            0,
            centroid_head,
            :,
            :
        ]
    )

    avg = 0.5 * (
        original_centroids
        + swapped_aligned
    )

    code, axis_conf = cent.relation_codes_from_centroids(
        avg
    )

    source_pred = SOURCE_ID_TO_REL[
        int(code)
    ]

    target_pred = source_to_target_relation(
        dataset,
        source_pred,
    )

    del original_batch
    del swapped_batch
    del original
    del swapped

    return (
        target_pred,
        source_pred,
        float(axis_conf),
    )


# =============================================================================
# Target generation
# =============================================================================

def target_generate_baseline(
    model,
    processor,
    image,
    record,
    dataset,
    args,
):
    set_causal_dataset_arg(
        args,
        dataset,
    )

    batch = target_real_batch(
        processor,
        image,
        record,
        dataset,
        args,
    )

    text = causal.generate(
        model,
        processor,
        batch,
        args,
    )

    pred = causal.parse_pred(
        text,
        args.dataset,
    )

    del batch

    # VG2-LR should reject vertical outputs rather than relabel them.
    if dataset == "vg2_lr" and pred not in ("left", "right"):
        pred = None

    return text, pred


def target_generate_steered(
    model,
    processor,
    layers,
    image,
    record,
    dataset,
    writer_templates,
    actuator_layers,
    predicted_target_relation,
    args,
):
    if predicted_target_relation is None:
        return "", None

    writer_relation = target_to_source_relation(
        dataset,
        predicted_target_relation,
    )

    set_causal_dataset_arg(
        args,
        dataset,
    )

    batch = target_real_batch(
        processor,
        image,
        record,
        dataset,
        args,
    )

    with causal.SteerLast(
        layers,
        writer_templates,
        list(actuator_layers),
        writer_relation,
        args.scale,
        "add",
        None,
    ):
        text = causal.generate(
            model,
            processor,
            batch,
            args,
        )

    pred = causal.parse_pred(
        text,
        args.dataset,
    )

    del batch

    if dataset == "vg2_lr" and pred not in ("left", "right"):
        pred = None

    return text, pred


# =============================================================================
# Dataset evaluation
# =============================================================================

def evaluate_dataset(
    model,
    processor,
    layers,
    dataset,
    train_records,
    test_records,
    writer_templates,
    actuator_layers,
    centroid_layer,
    centroid_head,
    middle_mode,
    middle_feature_layers,
    middle_top_k,
    args,
    dataset_outdir,
):
    dataset_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_order = target_labels(dataset)

    # ------------------------------------------------------------------
    # Fit target-domain middle reader.
    # ------------------------------------------------------------------
    (
        train_features,
        train_labels,
        train_sids,
        middle_train_errors,
    ) = collect_middle_train_features(
        model,
        processor,
        layers,
        train_records,
        dataset,
        middle_feature_layers,
        args,
    )

    write_csv(
        dataset_outdir / "middle_train_errors.csv",
        middle_train_errors,
    )

    middle_search_rows = []

    if middle_mode == "auto":
        selected_middle_layers, middle_search_rows = (
            select_middle_layers_train_only(
                train_features,
                train_labels,
                middle_feature_layers,
                label_order,
                middle_top_k,
                args.middle_cv_repeats,
                args.middle_cv_fit_frac,
                args.seed,
            )
        )

        write_csv(
            dataset_outdir / "middle_layer_search.csv",
            middle_search_rows,
        )

    else:
        selected_middle_layers = list(
            middle_feature_layers
        )

    middle_selector = fit_middle_selector(
        train_features,
        train_labels,
        middle_feature_layers,
        selected_middle_layers,
        label_order,
    )

    del train_features
    cleanup()

    relation_token_map = cent.relation_token_variants(
        processor.tokenizer
    )

    details = []
    errors = []

    for record in tqdm(
        test_records,
        desc=f"TEST:{dataset}:{args.model}",
    ):
        image = None

        try:
            image = open_image_path(record)
            gt = record["relation"]

            # ----------------------------------------------
            # Baseline
            # ----------------------------------------------
            baseline_text, baseline_pred = target_generate_baseline(
                model,
                processor,
                image,
                record,
                dataset,
                args,
            )

            baseline_ok = (
                baseline_pred == gt
            )

            # ----------------------------------------------
            # Oracle: GT only chooses source writer.
            # ----------------------------------------------
            oracle_text, oracle_pred = target_generate_steered(
                model,
                processor,
                layers,
                image,
                record,
                dataset,
                writer_templates,
                actuator_layers,
                gt,
                args,
            )

            oracle_ok = (
                oracle_pred == gt
            )

            # ----------------------------------------------
            # Middle: target TRAIN reader.
            # ----------------------------------------------
            q = target_middle_q(
                model,
                processor,
                layers,
                image,
                record,
                dataset,
                selected_middle_layers,
                args,
            )

            (
                middle_pred,
                middle_scores,
                middle_margin,
            ) = predict_middle(
                q,
                middle_selector,
                label_order,
            )

            middle_selector_ok = (
                middle_pred == gt
            )

            (
                middle_text,
                middle_final_pred,
            ) = target_generate_steered(
                model,
                processor,
                layers,
                image,
                record,
                dataset,
                writer_templates,
                actuator_layers,
                middle_pred,
                args,
            )

            middle_final_ok = (
                middle_final_pred == gt
            )

            # ----------------------------------------------
            # Centroid: source-selected frozen head.
            # ----------------------------------------------
            (
                centroid_pred,
                centroid_source_pred,
                centroid_axis_conf,
            ) = target_centroid_predict(
                model,
                processor,
                image,
                record,
                dataset,
                centroid_layer,
                centroid_head,
                relation_token_map,
                args,
            )

            centroid_selector_ok = (
                centroid_pred == gt
            )

            (
                centroid_text,
                centroid_final_pred,
            ) = target_generate_steered(
                model,
                processor,
                layers,
                image,
                record,
                dataset,
                writer_templates,
                actuator_layers,
                centroid_pred,
                args,
            )

            centroid_final_ok = (
                centroid_final_pred == gt
            )

            details.append({
                "sid": int(record["sid"]),
                "relation": gt,

                "baseline_pred": baseline_pred or "",
                "baseline_correct": int(baseline_ok),
                "baseline_text": baseline_text,

                "oracle_pred": oracle_pred or "",
                "oracle_correct": int(oracle_ok),
                "oracle_text": oracle_text,

                "middle_selector_pred": middle_pred,
                "middle_selector_correct": int(
                    middle_selector_ok
                ),
                "middle_margin": middle_margin,
                "middle_final_pred": (
                    middle_final_pred or ""
                ),
                "middle_final_correct": int(
                    middle_final_ok
                ),
                "middle_text": middle_text,

                "centroid_selector_pred": (
                    centroid_pred or ""
                ),
                "centroid_source_pred": (
                    centroid_source_pred or ""
                ),
                "centroid_axis_confidence": (
                    centroid_axis_conf
                ),
                "centroid_selector_correct": int(
                    centroid_selector_ok
                ),
                "centroid_final_pred": (
                    centroid_final_pred or ""
                ),
                "centroid_final_correct": int(
                    centroid_final_ok
                ),
                "centroid_text": centroid_text,

                "oracle_transition": classify_transition(
                    baseline_ok,
                    oracle_ok,
                ),
                "middle_transition": classify_transition(
                    baseline_ok,
                    middle_final_ok,
                ),
                "centroid_transition": classify_transition(
                    baseline_ok,
                    centroid_final_ok,
                ),
            })

        except Exception as exc:
            errors.append({
                "sid": int(record["sid"]),
                "relation": record["relation"],
                "stage": "test",
                "error": f"{type(exc).__name__}: {exc}",
            })

            tqdm.write(
                f"[TEST ERROR {dataset} sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                image.close()

            cleanup()

    write_csv(
        dataset_outdir / "test_details.csv",
        details,
    )

    write_csv(
        dataset_outdir / "test_errors.csv",
        errors,
    )

    expected = len(test_records)
    valid = len(details)
    coverage = valid / max(1, expected)

    def acc(field):
        return safe_mean(
            row[field]
            for row in details
        )

    baseline_acc = acc(
        "baseline_correct"
    )
    oracle_acc = acc(
        "oracle_correct"
    )
    middle_selector_acc = acc(
        "middle_selector_correct"
    )
    middle_final_acc = acc(
        "middle_final_correct"
    )
    centroid_selector_acc = acc(
        "centroid_selector_correct"
    )
    centroid_final_acc = acc(
        "centroid_final_correct"
    )

    def transition_counts(field):
        return (
            sum(
                row[field] == "W2C"
                for row in details
            ),
            sum(
                row[field] == "C2W"
                for row in details
            ),
        )

    oracle_w2c, oracle_c2w = transition_counts(
        "oracle_transition"
    )
    middle_w2c, middle_c2w = transition_counts(
        "middle_transition"
    )
    centroid_w2c, centroid_c2w = transition_counts(
        "centroid_transition"
    )

    summary = {
        "model": args.model,
        "dataset": dataset,
        "N_train": len(train_records),
        "N_test_expected": expected,
        "N_test_valid": valid,
        "coverage": coverage,

        "actuator_layers": ",".join(
            map(str, actuator_layers)
        ),

        "middle_layers": ",".join(
            map(str, selected_middle_layers)
        ),

        "centroid_layer": centroid_layer,
        "centroid_head": centroid_head,

        "baseline_acc": baseline_acc,

        "oracle_acc": oracle_acc,
        "oracle_gain": (
            oracle_acc - baseline_acc
        ),
        "oracle_W2C": oracle_w2c,
        "oracle_C2W": oracle_c2w,

        "middle_selector_acc": (
            middle_selector_acc
        ),
        "middle_final_acc": (
            middle_final_acc
        ),
        "middle_gain": (
            middle_final_acc
            - baseline_acc
        ),
        "middle_W2C": middle_w2c,
        "middle_C2W": middle_c2w,

        "centroid_selector_acc": (
            centroid_selector_acc
        ),
        "centroid_final_acc": (
            centroid_final_acc
        ),
        "centroid_gain": (
            centroid_final_acc
            - baseline_acc
        ),
        "centroid_W2C": centroid_w2c,
        "centroid_C2W": centroid_c2w,
    }

    per_relation = []

    for rel in label_order:
        subset = [
            row for row in details
            if row["relation"] == rel
        ]

        per_relation.append({
            "relation": rel,
            "N": len(subset),
            "baseline_acc": safe_mean(
                r["baseline_correct"]
                for r in subset
            ),
            "oracle_acc": safe_mean(
                r["oracle_correct"]
                for r in subset
            ),
            "middle_selector_acc": safe_mean(
                r["middle_selector_correct"]
                for r in subset
            ),
            "middle_final_acc": safe_mean(
                r["middle_final_correct"]
                for r in subset
            ),
            "centroid_selector_acc": safe_mean(
                r["centroid_selector_correct"]
                for r in subset
            ),
            "centroid_final_acc": safe_mean(
                r["centroid_final_correct"]
                for r in subset
            ),
        })

    write_csv(
        dataset_outdir / "summary.csv",
        [summary],
    )

    write_csv(
        dataset_outdir / "per_relation.csv",
        per_relation,
    )

    print("\n" + "=" * 150)
    print(
        f"{args.model} | {dataset} | "
        f"TRAIN={len(train_records)} | "
        f"TEST={valid}/{expected} "
        f"(coverage={coverage:.3f})"
    )
    print("=" * 150)

    print(
        f"baseline                  : "
        f"{baseline_acc:.4f}"
    )

    print(
        f"synthetic oracle          : "
        f"{oracle_acc:.4f} "
        f"({oracle_acc-baseline_acc:+.4f}) | "
        f"W2C={oracle_w2c} C2W={oracle_c2w}"
    )

    print(
        f"target middle selector    : "
        f"{middle_selector_acc:.4f}"
    )

    print(
        f"middle -> syn writer      : "
        f"{middle_final_acc:.4f} "
        f"({middle_final_acc-baseline_acc:+.4f}) | "
        f"W2C={middle_w2c} C2W={middle_c2w}"
    )

    print(
        f"synthetic centroid selector: "
        f"{centroid_selector_acc:.4f}"
    )

    print(
        f"centroid -> syn writer    : "
        f"{centroid_final_acc:.4f} "
        f"({centroid_final_acc-baseline_acc:+.4f}) | "
        f"W2C={centroid_w2c} C2W={centroid_c2w}"
    )

    print(
        f"middle layers             : "
        f"{selected_middle_layers}"
    )

    print(
        f"centroid source head      : "
        f"L{centroid_layer}H{centroid_head}"
    )

    print("\nPer relation:")

    for row in per_relation:
        print(
            f"{row['relation']:>5s} | "
            f"N={row['N']:3d} | "
            f"base={row['baseline_acc']:.4f} | "
            f"oracle={row['oracle_acc']:.4f} | "
            f"middle={row['middle_final_acc']:.4f}"
            f"(sel={row['middle_selector_acc']:.4f}) | "
            f"centroid={row['centroid_final_acc']:.4f}"
            f"(sel={row['centroid_selector_acc']:.4f})"
        )

    print("=" * 150)

    if coverage < args.min_success_rate:
        raise RuntimeError(
            f"{dataset}: TEST success coverage only "
            f"{valid}/{expected}={coverage:.3f}, below "
            f"--min-success-rate={args.min_success_rate}. "
            "Results were saved, but do NOT treat them as a valid "
            "full-dataset comparison."
        )

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not (
        0.0 < args.train_frac < 1.0
    ):
        raise ValueError(
            "--train-frac must lie in (0,1)"
        )

    datasets = [
        item.strip()
        for item in args.datasets.split(",")
        if item.strip()
    ]

    valid_datasets = {
        "controlled_a",
        "vg2_lr",
    }

    unknown = [
        d for d in datasets
        if d not in valid_datasets
    ]

    if unknown:
        raise ValueError(
            f"Unknown datasets={unknown}"
        )

    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_records = load_synthetic(
        args
    )

    print("\n" + "=" * 150)
    print("SOURCE")
    print("=" * 150)

    print(
        f"synthetic={args.synthetic_dir} | "
        f"N={len(source_records)} | "
        f"relations={relation_counts(source_records)}"
    )

    print(
        f"prompt={SYNTHETIC_PROMPT}"
    )

    print("=" * 150)

    (
        model,
        processor,
        layers,
        spec,
        decoder_path,
    ) = load_model(args)

    n_layers = len(layers)

    actuator_layers = resolve_actuator_layers(
        args,
        n_layers,
    )

    (
        middle_mode,
        middle_feature_layers,
        middle_top_k,
    ) = resolve_middle_feature_layers(
        args,
        n_layers,
    )

    centroid_layers = resolve_centroid_layers(
        args,
        n_layers,
    )

    # ------------------------------------------------------------------
    # SOURCE WRITER ONCE PER MODEL
    # ------------------------------------------------------------------
    writer_templates, source_writer_errors = (
        fit_synthetic_writer(
            model,
            processor,
            layers,
            source_records,
            actuator_layers,
            args,
        )
    )

    write_csv(
        outdir / "source_writer_errors.csv",
        source_writer_errors,
    )

    # ------------------------------------------------------------------
    # SOURCE CENTROID HEAD ONCE PER MODEL
    # ------------------------------------------------------------------
    (
        centroid_layer,
        centroid_head,
        centroid_ranking,
        source_centroid_errors,
    ) = fit_synthetic_centroid_head(
        model,
        processor,
        source_records,
        centroid_layers,
        args,
    )

    write_csv(
        outdir / "source_centroid_head_ranking.csv",
        centroid_ranking,
    )

    write_csv(
        outdir / "source_centroid_errors.csv",
        source_centroid_errors,
    )

    print("\n" + "=" * 150)
    print("SOURCE COMPONENTS")
    print("=" * 150)

    print(
        f"late writer layers={actuator_layers}"
    )

    print(
        f"synthetic centroid best="
        f"L{centroid_layer}H{centroid_head} | "
        f"source_acc={centroid_ranking[0]['accuracy']:.4f}"
    )

    print(
        f"middle preset={middle_mode} | "
        f"feature_layers={middle_feature_layers} | "
        f"top_k={middle_top_k}"
    )

    print("=" * 150)

    summary_rows = []

    # ------------------------------------------------------------------
    # TARGET DATASETS
    # ------------------------------------------------------------------
    for dataset in datasets:
        (
            records,
            train_records,
            test_records,
        ) = load_target_dataset(
            dataset,
            args,
        )

        print("\n" + "=" * 150)
        print(f"TARGET LOAD: {dataset}")
        print("=" * 150)

        print(
            f"ALL={len(records)} {relation_counts(records)}"
        )

        print(
            f"TRAIN={len(train_records)} "
            f"{relation_counts(train_records)}"
        )

        print(
            f"TEST={len(test_records)} "
            f"{relation_counts(test_records)}"
        )

        print("=" * 150)

        summary = evaluate_dataset(
            model=model,
            processor=processor,
            layers=layers,
            dataset=dataset,
            train_records=train_records,
            test_records=test_records,
            writer_templates=writer_templates,
            actuator_layers=actuator_layers,
            centroid_layer=centroid_layer,
            centroid_head=centroid_head,
            middle_mode=middle_mode,
            middle_feature_layers=middle_feature_layers,
            middle_top_k=middle_top_k,
            args=args,
            dataset_outdir=outdir / dataset,
        )

        summary_rows.append(summary)

    write_csv(
        outdir / "summary_all_datasets.csv",
        summary_rows,
    )

    config = {
        "script": (
            "eval_synthetic_to_controlledA_vg2_5models_v1.py"
        ),
        "model": args.model,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": n_layers,
        "attn_implementation": "eager",

        "synthetic_source": str(
            args.synthetic_dir
        ),
        "synthetic_N": len(
            source_records
        ),
        "synthetic_prompt": (
            SYNTHETIC_PROMPT
        ),

        "writer_definition": (
            "Real-Gray last-token relation mean "
            "minus balanced four-class common mean"
        ),
        "writer_layers": actuator_layers,

        "middle_definition": (
            "TARGET TRAIN: "
            "(h_sub-h_ref)_Real - "
            "(h_sub-h_ref)_NoImage"
        ),
        "middle_mode": middle_mode,
        "middle_feature_layers": (
            middle_feature_layers
        ),
        "middle_top_k": middle_top_k,

        "centroid_definition": (
            "Synthetic source only; per-head "
            "original+swap aligned centroid; "
            "best (layer,head) frozen to targets"
        ),
        "centroid_layer": centroid_layer,
        "centroid_head": centroid_head,

        "datasets": datasets,
        "vg2_filter": "left/right only",
        "train_frac": args.train_frac,
        "seed": args.seed,
        "scale": args.scale,
        "gray_value": args.gray_value,
        "min_success_rate": args.min_success_rate,

        "target_gt_used_for_oracle_routing": True,
        "target_test_gt_used_for_middle_fit": False,
        "target_test_gt_used_for_centroid_selection": False,
        "target_test_gt_used_for_metrics": True,
    }

    (
        outdir / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 170)
    print("FINAL SUMMARY")
    print("=" * 170)

    for row in summary_rows:
        print(
            f"{row['model']:>12s} | "
            f"{row['dataset']:>12s} | "
            f"N={row['N_test_valid']:3d}/"
            f"{row['N_test_expected']:3d} | "
            f"base={row['baseline_acc']:.4f} | "
            f"oracle={row['oracle_acc']:.4f} | "
            f"middle={row['middle_final_acc']:.4f}"
            f"(sel={row['middle_selector_acc']:.4f}) | "
            f"centroid={row['centroid_final_acc']:.4f}"
            f"(sel={row['centroid_selector_acc']:.4f})"
        )

    print("=" * 170)


if __name__ == "__main__":
    main()
