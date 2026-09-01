#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_crossdataset_late_causal_qwen25_v1.py

Replicate the already-discovered Qwen2.5-VL late last-token causal steering
mechanism on two new spatial datasets without repeating the COCO all-layer search.

Reused model-specific sites
---------------------------
Qwen2.5-VL-3B:
    actuator layers = L32,L33,L34,L35
    middle residual-Direction guide = L19

Qwen2.5-VL-7B:
    actuator layers = L25,L26,L27
    middle residual-Direction guide = L16

Datasets
--------
controlled_a:
    data/controlled_images_dataset.json
    labels: left, right, on, under
    images are read from each record's image_path.
    Split is by unordered object pair, so the four layouts of one pair do not
    cross TRAIN/TEST.

vg2:
    Auto-detects one of:
      data/vg_spatial_4dir.jsonl
      data/vg_spatial_4dir.json
      data/vg_qa_two_obj_filtered.json
      data/vg_qa_two_obj.json
    or pass --vg-json explicitly.

    The loader accepts:
      - dict records with image/image_path, subject, reference, relation
      - standard AdaptVis tuples [image_id, positive_caption, negative_caption]

    It keeps only four canonical relations:
      left, right, above, below
    with aliases top/on-top -> above and under/bottom/beneath -> below.
    front/behind are excluded.

Core method
-----------
TRAIN, at each known actuator layer l:

    delta_i,l = h_last(real)_i,l - h_last(gray)_i,l

    mu_r,l = mean(delta_i,l | relation=r)
    mu_global,l = balanced mean_r(mu_r,l)
    s_r,l = mu_r,l - mu_global,l

TEST oracle:
    h_last,l <- h_last,l + s_GT,l

TEST non-oracle guide:
    at a known middle layer, compute the object-pair residual

      q_i = [(h_sub-h_ref)_real] - [(h_sub-h_ref)_gray]

    classify q_i by TRAIN relation prototypes, then use that prediction to
    choose which late causal direction to add.

This script reports actual model.generate() results:
    baseline
    oracle_all_add
    guide_all_add
    guide_conflict_add
    guide_all_contrast
    guide_conflict_contrast

The goal here is replication on new datasets, not another layer search.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


EPS = 1e-12


# =============================================================================
# CLI / model presets
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--model-id",
        required=True,
        choices=[
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
    )

    p.add_argument(
        "--dataset",
        required=True,
        choices=["controlled_a", "vg2"],
    )

    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--controlled-json",
        default="data/controlled_images_dataset.json",
    )

    p.add_argument(
        "--vg-json",
        default="",
        help="Optional explicit VG2 JSON/JSONL path.",
    )

    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )

    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def model_preset(model_id):
    if "3B" in model_id:
        return {
            "actuator_layers": [32, 33, 34, 35],
            "guide_layers": [19],
        }

    if "7B" in model_id:
        return {
            "actuator_layers": [25, 26, 27],
            "guide_layers": [16],
        }

    raise ValueError(model_id)


def dataset_spec(dataset):
    if dataset == "controlled_a":
        return {
            "labels": ("left", "right", "on", "under"),
            "answer_words": ("left", "right", "on", "under"),
        }

    if dataset == "vg2":
        return {
            "labels": ("left", "right", "above", "below"),
            "answer_words": ("left", "right", "above", "below"),
        }

    raise ValueError(dataset)


# =============================================================================
# generic utilities
# =============================================================================

def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields, seen = [], set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def normalize_spaces(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def strip_terminal(s):
    s = normalize_spaces(s)
    s = re.sub(r"[\s\.,;:!?]+$", "", s)
    return s.strip()


def remove_article(s):
    return re.sub(
        r"^(?:the|a|an)\s+",
        "",
        strip_terminal(s),
        flags=re.I,
    ).strip()


# =============================================================================
# relation normalization
# =============================================================================

CONTROLLED_REL_ALIASES = {
    "left": "left",
    "left_of": "left",
    "left of": "left",
    "right": "right",
    "right_of": "right",
    "right of": "right",
    "on": "on",
    "on top": "on",
    "on top of": "on",
    "above": "on",
    "under": "under",
    "below": "under",
    "beneath": "under",
}

VG_REL_ALIASES = {
    "left": "left",
    "left_of": "left",
    "left of": "left",
    "right": "right",
    "right_of": "right",
    "right of": "right",
    "above": "above",
    "on": "above",
    "on top": "above",
    "on top of": "above",
    "top": "above",
    "top of": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "bottom": "below",
    "bottom of": "below",
}


def normalize_relation(value, dataset):
    x = str(value).strip().lower()
    x = x.replace("-", " ")
    x = normalize_spaces(x)

    aliases = (
        CONTROLLED_REL_ALIASES
        if dataset == "controlled_a"
        else VG_REL_ALIASES
    )

    if x in aliases:
        return aliases[x]

    # phrase fallback
    if re.search(r"\bleft(?:\s+of)?\b", x):
        return "left"

    if re.search(r"\bright(?:\s+of)?\b", x):
        return "right"

    if dataset == "controlled_a":
        if re.search(r"\b(?:on|above|top|over)\b", x):
            return "on"

        if re.search(r"\b(?:under|below|beneath|bottom)\b", x):
            return "under"

    else:
        if re.search(r"\b(?:above|on top of|top of|over)\b", x):
            return "above"

        if re.search(r"\b(?:below|under|beneath|bottom of)\b", x):
            return "below"

    return None


def parse_pred(text, dataset):
    s = str(text).lower()

    hits = []

    candidates = (
        [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("on", r"\bon\b"),
            ("on", r"\babove\b"),
            ("under", r"\bunder(?:neath)?\b"),
            ("under", r"\bbelow\b"),
        ]
        if dataset == "controlled_a"
        else [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("above", r"\babove\b"),
            ("above", r"\bon top of\b"),
            ("below", r"\bbelow\b"),
            ("below", r"\bunder(?:neath)?\b"),
            ("below", r"\bbeneath\b"),
        ]
    )

    for relation, pattern in candidates:
        m = re.search(pattern, s)

        if m:
            hits.append((m.start(), relation))

    if not hits:
        return None

    return sorted(hits)[0][1]


# =============================================================================
# Controlled A loader
# =============================================================================

CONTROLLED_FILENAME_PATTERNS = [
    ("left", "_left_of_"),
    ("right", "_right_of_"),
    ("on", "_on_"),
    ("under", "_under_"),
]


def parse_controlled_filename(image_path):
    name = Path(image_path).stem

    for relation, marker in CONTROLLED_FILENAME_PATTERNS:
        if marker in name:
            subject, reference = name.split(marker, 1)

            return (
                remove_article(subject.replace("_", " ")),
                remove_article(reference.replace("_", " ")),
                relation,
            )

    return None, None, None


def parse_caption_relation(caption, dataset):
    text = strip_terminal(caption)

    patterns = (
        [
            ("left", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(.+)$"),
            ("right", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(.+)$"),
            ("on", r"^(.+?)\s+(?:is\s+)?on(?:\s+top\s+of)?\s+(.+)$"),
            ("under", r"^(.+?)\s+(?:is\s+)?(?:under|below|beneath)\s+(.+)$"),
        ]
        if dataset == "controlled_a"
        else [
            ("left", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(.+)$"),
            ("right", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(.+)$"),
            ("above", r"^(.+?)\s+(?:is\s+)?(?:above|over|on\s+top\s+of|at\s+the\s+top\s+of)\s+(.+)$"),
            ("below", r"^(.+?)\s+(?:is\s+)?(?:below|under|beneath|at\s+the\s+bottom\s+of)\s+(.+)$"),
        ]
    )

    for relation, pattern in patterns:
        m = re.match(pattern, text, flags=re.I)

        if m:
            return (
                remove_article(m.group(1)),
                remove_article(m.group(2)),
                relation,
            )

    return None, None, None


def load_controlled_a(args):
    path = Path(args.controlled_json)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for i, row in enumerate(data):
        image_path = str(row.get("image_path", ""))

        if not image_path:
            continue

        subject, reference, relation = parse_controlled_filename(
            image_path
        )

        if subject is None:
            captions = row.get("caption_options", [])

            if captions:
                subject, reference, relation = parse_caption_relation(
                    captions[0],
                    "controlled_a",
                )

        if (
            subject is None
            or reference is None
            or relation is None
        ):
            continue

        candidate = Path(image_path)

        if not candidate.exists():
            candidate2 = Path(args.data_root) / image_path

            if candidate2.exists():
                candidate = candidate2

        records.append({
            "sid": i,
            "dataset": "controlled_a",
            "image_path": str(candidate),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "pair_group": " || ".join(
                sorted(
                    [
                        subject.lower(),
                        reference.lower(),
                    ]
                )
            ),
        })

    print(
        f"[Controlled_A] loaded={len(records)} | "
        f"relations={dict(Counter(r['relation'] for r in records))}"
    )

    return records


# =============================================================================
# VG2 loader
# =============================================================================

def auto_vg_json(args):
    if args.vg_json:
        path = Path(args.vg_json)

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    candidates = [
        Path(args.data_root) / "vg_spatial_4dir.jsonl",
        Path(args.data_root) / "vg_spatial_4dir.json",
        Path(args.data_root) / "vg_qa_two_obj_filtered.json",
        Path(args.data_root) / "vg_qa_two_obj.json",
    ]

    for path in candidates:
        if path.exists():
            print(f"[VG2] auto-selected annotation: {path}")
            return path

    raise FileNotFoundError(
        "Could not find VG2 annotation. Tried:\n"
        + "\n".join(str(x) for x in candidates)
    )


def load_json_or_jsonl(path):
    if path.suffix.lower() == ".jsonl":
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        return rows

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_vg_image_path(row, image_id, args):
    if isinstance(row, dict):
        for key in [
            "image_path",
            "image",
            "path",
            "file_name",
        ]:
            value = row.get(key)

            if isinstance(value, str) and value:
                candidate = Path(value)

                if candidate.exists():
                    return str(candidate)

                candidate2 = Path(args.data_root) / value

                if candidate2.exists():
                    return str(candidate2)

    if image_id is not None:
        for suffix in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            candidate = (
                Path(args.data_root)
                / "vg_images"
                / f"{image_id}{suffix}"
            )

            if candidate.exists():
                return str(candidate)

        # use the canonical path even if existence check fails so the later
        # error message is informative
        return str(
            Path(args.data_root)
            / "vg_images"
            / f"{image_id}.jpg"
        )

    return ""


def dict_first(row, keys, default=None):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]

    return default


def parse_vg_dict(row, args):
    image_id = dict_first(
        row,
        [
            "image_id",
            "imageId",
            "img_id",
            "id",
        ],
    )

    relation_raw = dict_first(
        row,
        [
            "relation",
            "target_relation",
            "answer",
            "label",
            "spatial_relation",
        ],
    )

    subject = dict_first(
        row,
        [
            "subject",
            "subject_name",
            "object1",
            "obj1",
            "source",
        ],
    )

    reference = dict_first(
        row,
        [
            "reference",
            "reference_name",
            "object2",
            "obj2",
            "target",
        ],
    )

    relation = (
        normalize_relation(
            relation_raw,
            "vg2",
        )
        if relation_raw is not None
        else None
    )

    if (
        relation is not None
        and subject is not None
        and reference is not None
    ):
        return {
            "image_id": image_id,
            "image_path": resolve_vg_image_path(
                row,
                image_id,
                args,
            ),
            "subject": remove_article(subject),
            "reference": remove_article(reference),
            "relation": relation,
        }

    caption = dict_first(
        row,
        [
            "caption",
            "positive_caption",
            "question",
            "statement",
        ],
    )

    if caption is None:
        options = row.get(
            "caption_options",
            row.get(
                "captions",
                None,
            ),
        )

        if (
            isinstance(options, list)
            and options
        ):
            caption = options[0]

    if caption:
        subject2, reference2, relation2 = parse_caption_relation(
            caption,
            "vg2",
        )

        if relation2:
            return {
                "image_id": image_id,
                "image_path": resolve_vg_image_path(
                    row,
                    image_id,
                    args,
                ),
                "subject": subject2,
                "reference": reference2,
                "relation": relation2,
            }

    return None


def parse_vg_sequence(row, args):
    if len(row) < 2:
        return None

    image_id = row[0]
    positive = row[1]

    subject, reference, relation = parse_caption_relation(
        positive,
        "vg2",
    )

    if relation is None:
        return None

    return {
        "image_id": image_id,
        "image_path": resolve_vg_image_path(
            {},
            image_id,
            args,
        ),
        "subject": subject,
        "reference": reference,
        "relation": relation,
    }


def load_vg2(args):
    path = auto_vg_json(args)
    data = load_json_or_jsonl(path)

    records = []
    skipped = Counter()

    for i, row in enumerate(data):
        if isinstance(row, dict):
            parsed = parse_vg_dict(
                row,
                args,
            )

        elif isinstance(row, (list, tuple)):
            parsed = parse_vg_sequence(
                row,
                args,
            )

        else:
            parsed = None

        if parsed is None:
            skipped["unparsed_or_non4dir"] += 1
            continue

        relation = parsed["relation"]

        if relation not in {
            "left",
            "right",
            "above",
            "below",
        }:
            skipped["non4dir"] += 1
            continue

        records.append({
            "sid": i,
            "dataset": "vg2",
            "image_path": parsed["image_path"],
            "subject": parsed["subject"],
            "reference": parsed["reference"],
            "relation": relation,
            "pair_group": " || ".join(
                sorted(
                    [
                        parsed["subject"].lower(),
                        parsed["reference"].lower(),
                    ]
                )
            ),
        })

    counts = Counter(
        r["relation"]
        for r in records
    )

    print(
        f"[VG2] annotation={path} | loaded={len(records)} | "
        f"relations={dict(counts)} | skipped={dict(skipped)}"
    )

    missing = [
        relation
        for relation in (
            "left",
            "right",
            "above",
            "below",
        )
        if counts[relation] == 0
    ]

    if missing:
        raise RuntimeError(
            f"VG2 loader is missing canonical relations {missing}. "
            "If you have the 4-direction VG file, pass it explicitly with --vg-json."
        )

    return records


# =============================================================================
# split
# =============================================================================

def split_controlled_by_pair(records, train_frac, seed):
    groups = defaultdict(list)

    for record in records:
        groups[record["pair_group"]].append(
            record
        )

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_train = int(
        round(
            len(keys)
            * train_frac
        )
    )

    n_train = max(
        1,
        min(
            n_train,
            len(keys) - 1,
        ),
    )

    train_keys = set(
        keys[:n_train]
    )

    train = [
        record
        for record in records
        if record["pair_group"] in train_keys
    ]

    test = [
        record
        for record in records
        if record["pair_group"] not in train_keys
    ]

    return train, test


def split_stratified(records, labels, train_frac, seed):
    rng = random.Random(seed)

    train = []
    test = []

    for relation in labels:
        rows = [
            record
            for record in records
            if record["relation"] == relation
        ]

        rows = sorted(
            rows,
            key=lambda r: r["sid"],
        )

        rng.shuffle(rows)

        n_train = int(
            round(
                len(rows)
                * train_frac
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

    return (
        sorted(
            train,
            key=lambda r: r["sid"],
        ),
        sorted(
            test,
            key=lambda r: r["sid"],
        ),
    )


def make_split(records, args, labels):
    if args.dataset == "controlled_a":
        train, test = split_controlled_by_pair(
            records,
            args.train_frac,
            args.seed,
        )

    else:
        train, test = split_stratified(
            records,
            labels,
            args.train_frac,
            args.seed,
        )

    print(
        f"[split] TRAIN={len(train)} "
        f"{dict(Counter(r['relation'] for r in train))}"
    )

    print(
        f"[split] TEST ={len(test)} "
        f"{dict(Counter(r['relation'] for r in test))}"
    )

    return train, test


# =============================================================================
# Qwen loading / prompt
# =============================================================================

def load_model(args):
    if not hasattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    ):
        raise RuntimeError(
            "Transformers has no Qwen2_5_VLForConditionalGeneration."
        )

    dtype = (
        torch.bfloat16
        if args.dtype == "auto"
        else auto_dtype(args.dtype)
    )

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {
            "": args.device
        },
    }

    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl

    cls = (
        transformers
        .Qwen2_5_VLForConditionalGeneration
    )

    try:
        model = cls.from_pretrained(
            args.model_id,
            dtype=dtype,
            **kwargs,
        )

    except TypeError:
        model = cls.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            **kwargs,
        )

    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            use_fast=False,
        )

    except TypeError:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )

    return model, processor


def resolve_layers(model):
    for path in [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ]:
        current = model

        try:
            for part in path.split("."):
                current = getattr(
                    current,
                    part,
                )

            if len(current):
                return current, path

        except Exception:
            pass

    raise RuntimeError(
        "Could not locate decoder blocks."
    )


def prompt_text(record, answer_words):
    answer_text = ", ".join(
        answer_words[:-1]
    ) + f", or {answer_words[-1]}"

    return (
        f"Determine the spatial relation of the {record['subject']} "
        f"to the {record['reference']} in the image. "
        f"Answer with {answer_text}."
    )


def build_prompt(processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:
        return question


def build_batch(processor, image, record, args, answer_words):
    question = prompt_text(
        record,
        answer_words,
    )

    prompt = build_prompt(
        processor,
        image,
        question,
    )

    errors = []

    for fn in [
        lambda: processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ),
    ]:
        try:
            batch = fn()
            break

        except Exception as exc:
            errors.append(exc)

    else:
        raise RuntimeError(
            f"Processor failed: {errors[-1]}"
        )

    return {
        key: (
            value.to(args.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def generate(model, processor, batch, args):
    input_len = int(
        batch["input_ids"].shape[1]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    if output_ids.shape[1] > input_len:
        generated = output_ids[
            0,
            input_len:
        ]

    else:
        generated = output_ids[0]

    text = processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    del output_ids

    return text


def make_gray(real, value):
    v = int(
        max(
            0,
            min(
                255,
                value,
            ),
        )
    )

    return Image.new(
        "RGB",
        real.size,
        (
            v,
            v,
            v,
        ),
    )


# =============================================================================
# token span finding
# =============================================================================

def find_all_subsequences(sequence, pattern):
    if not pattern:
        return []

    starts = []
    n = len(pattern)

    for i in range(
        len(sequence) - n + 1
    ):
        if (
            list(
                sequence[
                    i:
                    i + n
                ]
            )
            == list(pattern)
        ):
            starts.append(i)

    return starts


def phrase_spans(tokenizer, full_ids, phrase):
    variants = [
        phrase,
        " " + phrase,
        phrase.strip(),
        " " + phrase.strip(),
    ]

    spans = []
    seen = set()

    for variant in variants:
        ids = tokenizer.encode(
            variant,
            add_special_tokens=False,
        )

        for start in find_all_subsequences(
            full_ids,
            ids,
        ):
            span = tuple(
                range(
                    start,
                    start + len(ids),
                )
            )

            if span not in seen:
                seen.add(span)
                spans.append(
                    list(span)
                )

    return spans


def locate_pair_spans(
    tokenizer,
    input_ids,
    subject,
    reference,
):
    subject_spans = phrase_spans(
        tokenizer,
        input_ids,
        subject,
    )

    reference_spans = phrase_spans(
        tokenizer,
        input_ids,
        reference,
    )

    if not subject_spans or not reference_spans:
        return None, None

    best = None

    for s_span in subject_spans:
        for r_span in reference_spans:
            if set(s_span) & set(r_span):
                continue

            # Prompt has subject before reference. Prefer that ordering, then
            # minimal distance, then latest occurrence.
            order_penalty = int(
                s_span[0] >= r_span[0]
            )

            distance = abs(
                float(np.mean(s_span))
                - float(np.mean(r_span))
            )

            score = (
                order_penalty,
                distance,
                -s_span[0],
            )

            if best is None or score < best[0]:
                best = (
                    score,
                    s_span,
                    r_span,
                )

    if best is None:
        return None, None

    return best[1], best[2]


# =============================================================================
# hooks: last state + object pair state
# =============================================================================

def extract_hidden(output):
    if torch.is_tensor(output):
        return output, ("tensor", 0)

    if isinstance(output, tuple):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                return item, ("tuple", i)

    if isinstance(output, list):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                return item, ("list", i)

    raise RuntimeError(
        f"Could not extract hidden from {type(output)}"
    )


def replace_hidden(output, descriptor, hidden):
    kind, index = descriptor

    if kind == "tensor":
        return hidden

    values = list(output)
    values[index] = hidden

    return (
        tuple(values)
        if kind == "tuple"
        else values
    )


class CaptureStates:
    def __init__(
        self,
        layers,
        actuator_layers,
        guide_layers,
        subject_span,
        reference_span,
    ):
        self.handles = []
        self.last_states = {}
        self.pair_states = {}

        selected = sorted(
            set(
                actuator_layers
                + guide_layers
            )
        )

        self.done = {
            layer: False
            for layer in selected
        }

        self.actuator_set = set(
            actuator_layers
        )

        self.guide_set = set(
            guide_layers
        )

        self.subject_span = subject_span
        self.reference_span = reference_span

        for layer in selected:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._hook(layer)
                )
            )

    def _hook(self, layer):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[layer]:
                return output

            hidden, _ = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            if layer in self.actuator_set:
                self.last_states[layer] = (
                    hidden[
                        0,
                        -1,
                        :
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            if layer in self.guide_set:
                max_index = max(
                    self.subject_span
                    + self.reference_span
                )

                if max_index >= hidden.shape[1]:
                    raise RuntimeError(
                        f"L{layer}: object token span exceeds seq_len={hidden.shape[1]}"
                    )

                hs = hidden[
                    0,
                    self.subject_span,
                    :
                ].mean(dim=0)

                hr = hidden[
                    0,
                    self.reference_span,
                    :
                ].mean(dim=0)

                self.pair_states[layer] = (
                    (hs - hr)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            self.done[layer] = True

            return output

        return hook

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SteerLast:
    def __init__(
        self,
        layers,
        templates,
        selected_layers,
        target_relation,
        scale,
        edit_mode,
        source_relation,
    ):
        self.handles = []

        self.done = {
            layer: False
            for layer in selected_layers
        }

        self.templates = templates
        self.target = target_relation
        self.source = source_relation
        self.scale = float(scale)
        self.edit_mode = edit_mode

        for layer in selected_layers:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._hook(layer)
                )
            )

    def vector(self, layer):
        target = np.asarray(
            self.templates[layer]["shared"][
                self.target
            ],
            dtype=np.float32,
        )

        if (
            self.edit_mode == "add"
            or self.source not in self.templates[layer]["shared"]
        ):
            return self.scale * target

        source = np.asarray(
            self.templates[layer]["shared"][
                self.source
            ],
            dtype=np.float32,
        )

        return (
            self.scale
            * (
                target
                - source
            )
        )

    def _hook(self, layer):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[layer]:
                return output

            hidden, descriptor = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            vec = torch.as_tensor(
                self.vector(layer),
                device=hidden.device,
                dtype=hidden.dtype,
            )

            edited = hidden.clone()

            edited[
                :,
                -1,
                :
            ] += vec[
                None,
                :
            ]

            self.done[layer] = True

            return replace_hidden(
                output,
                descriptor,
                edited,
            )

        return hook

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# run real/gray with state capture
# =============================================================================

def prepare_batch_and_spans(
    processor,
    image,
    record,
    args,
    answer_words,
):
    batch = build_batch(
        processor,
        image,
        record,
        args,
        answer_words,
    )

    ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )

    subject_span, reference_span = locate_pair_spans(
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
            f"Could not locate text tokens: "
            f"subject={record['subject']!r}, "
            f"reference={record['reference']!r}"
        )

    return (
        batch,
        subject_span,
        reference_span,
    )


def generate_capture(
    model,
    processor,
    layers,
    image,
    record,
    args,
    answer_words,
    actuator_layers,
    guide_layers,
):
    (
        batch,
        subject_span,
        reference_span,
    ) = prepare_batch_and_spans(
        processor,
        image,
        record,
        args,
        answer_words,
    )

    with CaptureStates(
        layers,
        actuator_layers,
        guide_layers,
        subject_span,
        reference_span,
    ) as capture:

        text = generate(
            model,
            processor,
            batch,
            args,
        )

        last_states = dict(
            capture.last_states
        )

        pair_states = dict(
            capture.pair_states
        )

    del batch

    pred = parse_pred(
        text,
        args.dataset,
    )

    return (
        text,
        pred,
        last_states,
        pair_states,
    )


def forward_capture_pair_only(
    model,
    processor,
    layers,
    image,
    record,
    args,
    answer_words,
    guide_layers,
):
    (
        batch,
        subject_span,
        reference_span,
    ) = prepare_batch_and_spans(
        processor,
        image,
        record,
        args,
        answer_words,
    )

    with CaptureStates(
        layers,
        [],
        guide_layers,
        subject_span,
        reference_span,
    ) as capture:

        with torch.inference_mode():
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

        pair_states = dict(
            capture.pair_states
        )

    del batch
    del outputs

    return pair_states


# =============================================================================
# training collection
# =============================================================================

def filter_sequence(requested):
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


def train_sample_allowed(row, mode):
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

    raise ValueError(mode)


def collect_train(
    model,
    processor,
    layers,
    train_records,
    args,
    answer_words,
    actuator_layers,
    guide_layers,
):
    rows = []
    cache = {}

    for record in tqdm(
        train_records,
        desc="TRAIN Real/Gray states",
    ):
        real = None
        gray = None

        try:
            real = Image.open(
                record["image_path"]
            ).convert("RGB")

            gray = make_gray(
                real,
                args.gray_value,
            )

            (
                real_text,
                real_pred,
                real_last,
                real_pair,
            ) = generate_capture(
                model,
                processor,
                layers,
                real,
                record,
                args,
                answer_words,
                actuator_layers,
                guide_layers,
            )

            (
                gray_text,
                gray_pred,
                gray_last,
                gray_pair,
            ) = generate_capture(
                model,
                processor,
                layers,
                gray,
                record,
                args,
                answer_words,
                actuator_layers,
                guide_layers,
            )

            gt = record["relation"]

            row = {
                "sid": record["sid"],
                "relation": gt,
                "subject": record["subject"],
                "reference": record["reference"],
                "real_pred": real_pred or "",
                "gray_pred": gray_pred or "",
                "real_correct": int(
                    real_pred == gt
                ),
                "gray_correct": int(
                    gray_pred == gt
                ),
                "real_text": real_text,
                "gray_text": gray_text,
            }

            rows.append(row)

            cache[
                record["sid"]
            ] = {
                "last_delta": {
                    layer: (
                        real_last[layer]
                        - gray_last[layer]
                    ).astype(np.float32)
                    for layer in actuator_layers
                },
                "pair_delta": {
                    layer: (
                        real_pair[layer]
                        - gray_pair[layer]
                    ).astype(np.float32)
                    for layer in guide_layers
                },
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


def fit_actuator_templates(
    train_records,
    train_rows,
    cache,
    actuator_layers,
    labels,
    requested_filter,
):
    row_map = {
        row["sid"]: row
        for row in train_rows
    }

    record_map = {
        record["sid"]: record
        for record in train_records
    }

    for mode in filter_sequence(
        requested_filter
    ):
        bags = {
            layer: {
                relation: []
                for relation in labels
            }
            for layer in actuator_layers
        }

        used = []

        for sid, item in cache.items():
            if sid not in row_map:
                continue

            if not train_sample_allowed(
                row_map[sid],
                mode,
            ):
                continue

            relation = record_map[sid][
                "relation"
            ]

            used.append(sid)

            for layer in actuator_layers:
                bags[layer][relation].append(
                    item["last_delta"][
                        layer
                    ]
                )

        missing = [
            (layer, relation)
            for layer in actuator_layers
            for relation in labels
            if not bags[layer][relation]
        ]

        if missing:
            print(
                f"[actuator template] filter={mode} "
                f"missing={missing[:8]} -> relax"
            )

            continue

        templates = {}

        for layer in actuator_layers:
            relation_mean = {
                relation: np.stack(
                    bags[layer][relation],
                    axis=0,
                ).mean(axis=0).astype(
                    np.float32
                )
                for relation in labels
            }

            global_mean = np.stack(
                [
                    relation_mean[relation]
                    for relation in labels
                ],
                axis=0,
            ).mean(axis=0).astype(
                np.float32
            )

            shared = {
                relation: (
                    relation_mean[relation]
                    - global_mean
                ).astype(
                    np.float32
                )
                for relation in labels
            }

            templates[layer] = {
                "global": global_mean,
                "shared": shared,
            }

        print(
            f"[actuator template] requested={requested_filter} "
            f"used={mode} N={len(used)} "
            f"counts={dict(Counter(record_map[sid]['relation'] for sid in used))}"
        )

        return templates, mode

    raise RuntimeError(
        "Could not fit four actuator relation templates."
    )


def fit_guide_codebook(
    train_records,
    train_rows,
    cache,
    guide_layers,
    labels,
):
    record_map = {
        record["sid"]: record
        for record in train_records
    }

    # Guide uses all TRAIN samples; no need for generation correctness.
    codebook = {}

    for layer in guide_layers:
        vectors = []
        y = []

        for sid, item in cache.items():
            vectors.append(
                item["pair_delta"][
                    layer
                ]
            )

            y.append(
                record_map[sid][
                    "relation"
                ]
            )

        X = np.stack(
            vectors,
            axis=0,
        ).astype(np.float64)

        y = np.asarray(
            y,
            dtype=object,
        )

        center = X.mean(
            axis=0
        )

        prototypes = {}

        for relation in labels:
            mask = (
                y == relation
            )

            if not np.any(mask):
                raise RuntimeError(
                    f"Guide L{layer} missing relation={relation}"
                )

            mu = (
                X[mask]
                - center[
                    None,
                    :
                ]
            ).mean(
                axis=0
            )

            norm = np.linalg.norm(
                mu
            )

            if norm <= EPS:
                raise RuntimeError(
                    f"Guide L{layer} degenerate relation={relation}"
                )

            prototypes[
                relation
            ] = (
                mu
                / norm
            ).astype(
                np.float32
            )

        codebook[layer] = {
            "center": center.astype(
                np.float32
            ),
            "prototypes": prototypes,
        }

    return codebook


def guide_predict(
    pair_delta,
    guide_codebook,
    guide_layers,
    labels,
):
    score_sum = {
        relation: 0.0
        for relation in labels
    }

    for layer in guide_layers:
        q = pair_delta[
            layer
        ].astype(
            np.float64
        )

        q = (
            q
            - guide_codebook[
                layer
            ][
                "center"
            ].astype(
                np.float64
            )
        )

        qnorm = np.linalg.norm(
            q
        )

        if qnorm <= EPS:
            continue

        q = q / qnorm

        for relation in labels:
            score_sum[
                relation
            ] += float(
                np.dot(
                    q,
                    guide_codebook[
                        layer
                    ][
                        "prototypes"
                    ][
                        relation
                    ].astype(
                        np.float64
                    ),
                )
            )

    return max(
        labels,
        key=lambda relation:
            score_sum[
                relation
            ],
    )


# =============================================================================
# TEST baseline + guide states
# =============================================================================

def collect_test_baseline_and_guide(
    model,
    processor,
    layers,
    test_records,
    args,
    answer_words,
    guide_layers,
):
    rows = []
    guide_delta_cache = {}

    for record in tqdm(
        test_records,
        desc="TEST baseline + guide",
    ):
        real = None
        gray = None

        try:
            real = Image.open(
                record["image_path"]
            ).convert("RGB")

            gray = make_gray(
                real,
                args.gray_value,
            )

            (
                real_text,
                real_pred,
                _real_last,
                real_pair,
            ) = generate_capture(
                model,
                processor,
                layers,
                real,
                record,
                args,
                answer_words,
                [],
                guide_layers,
            )

            gray_pair = forward_capture_pair_only(
                model,
                processor,
                layers,
                gray,
                record,
                args,
                answer_words,
                guide_layers,
            )

            gt = record["relation"]

            rows.append({
                "sid": record["sid"],
                "relation": gt,
                "baseline_pred": real_pred or "",
                "baseline_correct": int(
                    real_pred == gt
                ),
                "baseline_text": real_text,
            })

            guide_delta_cache[
                record["sid"]
            ] = {
                layer: (
                    real_pair[layer]
                    - gray_pair[layer]
                ).astype(
                    np.float32
                )
                for layer in guide_layers
            }

        except Exception as exc:
            tqdm.write(
                f"[TEST BASE ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    return rows, guide_delta_cache


# =============================================================================
# steering eval
# =============================================================================

def run_steering(
    model,
    processor,
    layers,
    templates,
    actuator_layers,
    test_record_map,
    baseline_rows,
    targets,
    args,
    answer_words,
    edit_mode,
    apply_mode,
    condition,
):
    rows = []

    for base in tqdm(
        baseline_rows,
        desc=condition,
    ):
        sid = int(
            base["sid"]
        )

        record = test_record_map[
            sid
        ]

        baseline_pred = (
            base["baseline_pred"]
            if base["baseline_pred"]
            else None
        )

        target = targets[
            sid
        ]

        if apply_mode == "all":
            apply_edit = True

        elif apply_mode == "conflict_only":
            apply_edit = (
                baseline_pred is not None
                and target
                != baseline_pred
            )

        else:
            raise ValueError(
                apply_mode
            )

        if not apply_edit:
            rows.append({
                "condition": condition,
                "sid": sid,
                "relation": record["relation"],
                "target": target,
                "baseline_pred": baseline_pred or "",
                "edit_pred": baseline_pred or "",
                "baseline_correct": int(
                    base["baseline_correct"]
                ),
                "edit_correct": int(
                    base["baseline_correct"]
                ),
                "applied": 0,
                "W2C": 0,
                "C2W": 0,
                "changed": 0,
            })

            continue

        image = None

        try:
            image = Image.open(
                record["image_path"]
            ).convert("RGB")

            batch = build_batch(
                processor,
                image,
                record,
                args,
                answer_words,
            )

            with SteerLast(
                layers,
                templates,
                actuator_layers,
                target,
                args.scale,
                edit_mode,
                baseline_pred,
            ):
                text = generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            pred = parse_pred(
                text,
                args.dataset,
            )

            base_correct = int(
                base["baseline_correct"]
            )

            edit_correct = int(
                pred
                == record["relation"]
            )

            rows.append({
                "condition": condition,
                "sid": sid,
                "relation": record["relation"],
                "target": target,
                "baseline_pred": baseline_pred or "",
                "edit_pred": pred or "",
                "baseline_correct": base_correct,
                "edit_correct": edit_correct,
                "applied": 1,
                "W2C": int(
                    base_correct == 0
                    and edit_correct == 1
                ),
                "C2W": int(
                    base_correct == 1
                    and edit_correct == 0
                ),
                "changed": int(
                    (pred or "")
                    != (baseline_pred or "")
                ),
                "text": text,
            })

            del batch

        except Exception as exc:
            tqdm.write(
                f"[STEER ERROR sid={sid}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


def summarize(rows, condition):
    n = len(rows)

    base_acc = safe_mean(
        row["baseline_correct"]
        for row in rows
    )

    edit_acc = safe_mean(
        row["edit_correct"]
        for row in rows
    )

    wrong_n = sum(
        1 - int(
            row["baseline_correct"]
        )
        for row in rows
    )

    correct_n = n - wrong_n

    w2c = sum(
        int(row["W2C"])
        for row in rows
    )

    c2w = sum(
        int(row["C2W"])
        for row in rows
    )

    applied = sum(
        int(row["applied"])
        for row in rows
    )

    return {
        "condition": condition,
        "N": n,
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "applied": applied,
        "applied_rate": (
            applied / n
            if n
            else float("nan")
        ),
        "W2C": w2c,
        "W2C_rate_wrong": (
            w2c / wrong_n
            if wrong_n
            else float("nan")
        ),
        "C2W": c2w,
        "C2W_rate_correct": (
            c2w / correct_n
            if correct_n
            else float("nan")
        ),
        "net": w2c - c2w,
        "changed_rate": safe_mean(
            row["changed"]
            for row in rows
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
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

    preset = model_preset(
        args.model_id
    )

    spec = dataset_spec(
        args.dataset
    )

    labels = spec[
        "labels"
    ]

    answer_words = spec[
        "answer_words"
    ]

    if args.dataset == "controlled_a":
        records = load_controlled_a(
            args
        )

    else:
        records = load_vg2(
            args
        )

    if args.max_samples is not None:
        records = records[
            :args.max_samples
        ]

    train_records, test_records = make_split(
        records,
        args,
        labels,
    )

    model, processor = load_model(
        args
    )

    layers, layer_path = resolve_layers(
        model
    )

    actuator_layers = preset[
        "actuator_layers"
    ]

    guide_layers = preset[
        "guide_layers"
    ]

    if max(
        actuator_layers
        + guide_layers
    ) >= len(layers):
        raise RuntimeError(
            f"Model has {len(layers)} blocks, "
            f"but requested layers "
            f"{actuator_layers + guide_layers}"
        )

    print(
        "\n"
        + "=" * 155
    )

    print(
        "CROSS-DATASET LATE CAUSAL REPLICATION"
    )

    print(
        "=" * 155
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"dataset={args.dataset}"
    )

    print(
        f"decoder={layer_path}"
    )

    print(
        f"decoder_blocks={len(layers)}"
    )

    print(
        f"actuator_layers={actuator_layers} [REUSED FROM COCO]"
    )

    print(
        f"guide_layers={guide_layers} [REUSED FROM COCO]"
    )

    print(
        f"labels={labels}"
    )

    print(
        f"TRAIN={len(train_records)} TEST={len(test_records)}"
    )

    print(
        "=" * 155
    )

    # TRAIN
    train_rows, train_cache = collect_train(
        model,
        processor,
        layers,
        train_records,
        args,
        answer_words,
        actuator_layers,
        guide_layers,
    )

    write_csv(
        outdir
        / "train_real_gray_generation.csv",
        train_rows,
    )

    actuator_templates, filter_used = fit_actuator_templates(
        train_records,
        train_rows,
        train_cache,
        actuator_layers,
        labels,
        args.template_filter,
    )

    guide_codebook = fit_guide_codebook(
        train_records,
        train_rows,
        train_cache,
        guide_layers,
        labels,
    )

    # TEST baseline + guide
    baseline_rows, guide_delta_cache = (
        collect_test_baseline_and_guide(
            model,
            processor,
            layers,
            test_records,
            args,
            answer_words,
            guide_layers,
        )
    )

    write_csv(
        outdir
        / "test_baseline.csv",
        baseline_rows,
    )

    test_record_map = {
        record["sid"]: record
        for record in test_records
    }

    oracle_targets = {
        row["sid"]:
            test_record_map[
                row["sid"]
            ][
                "relation"
            ]
        for row in baseline_rows
    }

    guide_targets = {
        row["sid"]:
            guide_predict(
                guide_delta_cache[
                    row["sid"]
                ],
                guide_codebook,
                guide_layers,
                labels,
            )
        for row in baseline_rows
    }

    guide_rows = []

    for row in baseline_rows:
        sid = row["sid"]

        guide_rows.append({
            "sid": sid,
            "relation": test_record_map[sid]["relation"],
            "guide_pred": guide_targets[sid],
            "guide_correct": int(
                guide_targets[sid]
                == test_record_map[sid]["relation"]
            ),
            "baseline_pred": row["baseline_pred"],
            "baseline_correct": row["baseline_correct"],
        })

    write_csv(
        outdir
        / "test_guide_predictions.csv",
        guide_rows,
    )

    guide_acc = safe_mean(
        row["guide_correct"]
        for row in guide_rows
    )

    baseline_acc = safe_mean(
        row["baseline_correct"]
        for row in baseline_rows
    )

    conditions = [
        (
            "oracle_all_add",
            oracle_targets,
            "add",
            "all",
        ),
        (
            "guide_all_add",
            guide_targets,
            "add",
            "all",
        ),
        (
            "guide_conflict_add",
            guide_targets,
            "add",
            "conflict_only",
        ),
        (
            "guide_all_contrast",
            guide_targets,
            "contrast",
            "all",
        ),
        (
            "guide_conflict_contrast",
            guide_targets,
            "contrast",
            "conflict_only",
        ),
    ]

    summaries = []
    details = []

    for (
        condition,
        targets,
        edit_mode,
        apply_mode,
    ) in conditions:

        rows = run_steering(
            model,
            processor,
            layers,
            actuator_templates,
            actuator_layers,
            test_record_map,
            baseline_rows,
            targets,
            args,
            answer_words,
            edit_mode,
            apply_mode,
            condition,
        )

        details.extend(
            rows
        )

        summaries.append(
            summarize(
                rows,
                condition,
            )
        )

        write_csv(
            outdir
            / "test_steering_details.csv",
            details,
        )

        write_csv(
            outdir
            / "test_summary.csv",
            summaries,
        )

    print(
        "\n"
        + "=" * 170
    )

    print(
        "FINAL TEST — CROSS-DATASET LATE CAUSAL REPLICATION"
    )

    print(
        "=" * 170
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"dataset={args.dataset}"
    )

    print(
        f"decoder_blocks={len(layers)}"
    )

    print(
        f"causal_layers={actuator_layers}"
    )

    print(
        f"guide_layers={guide_layers}"
    )

    print(
        f"template_filter={filter_used}"
    )

    print(
        f"baseline={baseline_acc:.4f}"
    )

    print(
        f"guide_acc={guide_acc:.4f}"
    )

    print()

    print(
        "condition                 | "
        "acc base->edit gain | applied | "
        "W2C/wrong | C2W/correct | net | changed"
    )

    for row in summaries:
        print(
            f"{row['condition']:25s} | "
            f"{row['base_acc']:.4f}->"
            f"{row['edit_acc']:.4f} "
            f"{row['gain']:+.4f} | "
            f"{row['applied']}/{row['applied_rate']:.3f} | "
            f"{row['W2C']}/{row['W2C_rate_wrong']:.3f} | "
            f"{row['C2W']}/{row['C2W_rate_correct']:.3f} | "
            f"{row['net']:+d} | "
            f"{row['changed_rate']:.3f}"
        )

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "dataset": args.dataset,
                "decoder_blocks": len(layers),
                "actuator_layers": actuator_layers,
                "guide_layers": guide_layers,
                "labels": list(labels),
                "train_n": len(train_records),
                "test_n": len(test_records),
                "template_filter": filter_used,
                "baseline_acc": baseline_acc,
                "guide_acc": guide_acc,
                "train_counts": dict(
                    Counter(
                        r["relation"]
                        for r in train_records
                    )
                ),
                "test_counts": dict(
                    Counter(
                        r["relation"]
                        for r in test_records
                    )
                ),
                "important_note": (
                    "Actuator and guide layer locations are reused from the "
                    "previous COCO experiments. Dataset-specific Real-Gray "
                    "relation templates are learned only on this dataset's TRAIN split."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
