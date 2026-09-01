#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_cross_dataset_direction_transfer_qwen25_v1.py

Question
--------
Do relation-specific late causal direction vectors learned on one dataset
transfer to another dataset WITHOUT refitting on the target dataset?

Supported transfer pairs
------------------------
1) COCO-two      -> Controlled A
2) COCO-two      -> VG2 left/right
3) Controlled A  -> COCO-two

Models
------
Qwen/Qwen2.5-VL-3B-Instruct
    reuse known late actuator window L32-L35

Qwen/Qwen2.5-VL-7B-Instruct
    reuse known late actuator window L25-L27

Important design
----------------
This experiment isolates vector transfer, so TEST uses GT only to SELECT the
source-trained vector. That is an oracle actuator-transfer test, not a
deployable selector test.

Source TRAIN:
    delta_i,l = h_last(real)_i,l - h_last(gray)_i,l

    mu_r,l = mean(delta_i,l | canonical relation r)

    mu_global,l = balanced mean_r(mu_r,l)

    s_r,l = mu_r,l - mu_global,l

Target TEST:
    h_last,l <- h_last,l + s_GT,l

NO target samples are used to fit/modify s_r.

Canonical relation mapping
--------------------------
COCO:
    left, right, above, below

Controlled A:
    left  -> left
    right -> right
    on    -> above
    under -> below

VG2-LR:
    left, right only

Thus COCO->Controlled A directly tests whether an "above" vector learned from
COCO can drive an "on" answer under the Controlled-A prompt, and similarly
below<->under.

Conditions
----------
baseline
transfer_oracle_add
transfer_oracle_contrast
transfer_wrong_add

The wrong-vector condition uses the opposite relation:
    left<->right
    above<->below

Source templates are cached. Therefore COCO->ControlledA and COCO->VG2-LR for
the same model only build the COCO source vectors once.
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


CANONICAL = ("left", "right", "above", "below")
CANONICAL_SET = set(CANONICAL)
EPS = 1e-12


# =============================================================================
# CLI
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
        "--source",
        required=True,
        choices=["coco", "controlled_a"],
    )

    p.add_argument(
        "--target",
        required=True,
        choices=["coco", "controlled_a", "vg2_lr"],
    )

    p.add_argument("--data-root", default="data")

    # COCO
    p.add_argument(
        "--coco-json",
        default="data/coco_qa_two_obj.json",
    )

    p.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    # Controlled A
    p.add_argument(
        "--controlled-json",
        default="data/controlled_images_dataset.json",
    )

    p.add_argument(
        "--controlled-prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )

    # VG2-LR
    p.add_argument(
        "--vg-filtered-json",
        default="data/vg_qa_two_obj_four_options.json",
    )

    p.add_argument(
        "--vg-original-json",
        default="data/vg_qa_two_obj.json",
    )

    p.add_argument(
        "--vg-six-prompt-jsonl",
        default="prompts/VG_QA_two_obj_with_answer_six_options.jsonl",
    )

    p.add_argument(
        "--vg-image-root",
        default="data/vg_images",
    )

    p.add_argument("--train-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=[
            "real_correct_gray_wrong",
            "real_correct",
            "all",
        ],
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--device", default="cuda:0")

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
        "--attn-impl",
        default="eager",
        choices=[
            "eager",
            "sdpa",
            "flash_attention_2",
            "none",
        ],
    )

    p.add_argument(
        "--source-cache-dir",
        default="output/cross_dataset_direction_cache",
        help=(
            "Source-trained vectors are cached here and reused across targets."
        ),
    )

    p.add_argument(
        "--rebuild-source",
        action="store_true",
        help="Ignore an existing source-template cache and rebuild it.",
    )

    p.add_argument(
        "--target-baseline-csv",
        default="",
        help=(
            "Optional existing target TEST baseline CSV. If compatible, "
            "baseline generation is skipped."
        ),
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def model_preset(model_id):
    if "3B" in model_id:
        return {
            "tag": "qwen25_3b",
            "actuator_layers": [32, 33, 34, 35],
        }

    if "7B" in model_id:
        return {
            "tag": "qwen25_7b",
            "actuator_layers": [25, 26, 27],
        }

    raise ValueError(model_id)


# =============================================================================
# Utilities
# =============================================================================

def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def safe_mean(values):
    out = []

    for value in values:
        try:
            number = float(value)
        except Exception:
            continue

        if math.isfinite(number):
            out.append(number)

    return float(np.mean(out)) if out else float("nan")


def normalize_spaces(text):
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def standard_answer_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""

    return value


def canonical_relation(value):
    """
    Map all datasets/output synonyms into one shared semantic coordinate system.
    """
    value = standard_answer_value(value)
    x = normalize_spaces(value).lower()
    x = x.strip(" .,:;!?")

    exact = {
        "left": "left",
        "right": "right",
        "above": "above",
        "top": "above",
        "on": "above",
        "on top": "above",
        "on top of": "above",
        "over": "above",
        "below": "below",
        "bottom": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
    }

    if x in exact:
        return exact[x]

    # Output-text parser. First occurrence wins.
    hits = []

    patterns = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("above", r"\bon\s+top\s+of\b"),
        ("above", r"\bon\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]

    for relation, pattern in patterns:
        match = re.search(
            pattern,
            x,
            flags=re.I,
        )

        if match:
            hits.append(
                (
                    match.start(),
                    relation,
                )
            )

    if not hits:
        return None

    return sorted(hits)[0][1]


def opposite_relation(relation):
    return {
        "left": "right",
        "right": "left",
        "above": "below",
        "below": "above",
    }[
        relation
    ]


def extract_user_text(raw_question):
    text = str(raw_question).strip()

    text = re.sub(
        r"^\s*<image>\s*",
        "",
        text,
        flags=re.I,
    )

    match = re.search(
        r"\bUSER\s*:\s*(.*?)(?:\s*\bASSISTANT\s*:|\Z)",
        text,
        flags=re.I | re.S,
    )

    if match:
        text = match.group(1)

    return text.strip()


OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+"
    r"in\s+relation\s+to\s+(?:the\s+)?(.+?)\?"
    r"\s*Answer\s+with",
    flags=re.I | re.S,
)


def parse_objects(question_text):
    compact = normalize_spaces(
        question_text
    )

    match = OBJECT_RE.search(
        compact
    )

    if not match:
        raise ValueError(
            f"Could not parse subject/reference: {compact!r}"
        )

    subject = match.group(1).strip()
    reference = match.group(2).strip()

    return subject, reference


def load_jsonl_by_id(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    rows = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_no, line in enumerate(
            handle,
            1,
        ):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)
            sid = int(row["id"])

            if sid in rows:
                raise ValueError(
                    f"Duplicate id={sid} in {path}"
                )

            rows[sid] = row

    return rows


# =============================================================================
# Dataset loaders
# =============================================================================

def load_coco(args):
    prompt_rows = load_jsonl_by_id(
        args.coco_prompt_jsonl
    )

    with Path(args.coco_json).open(
        "r",
        encoding="utf-8",
    ) as handle:
        annotations = json.load(handle)

    records = []

    for sid in sorted(prompt_rows):
        if sid < 0 or sid >= len(annotations):
            continue

        prompt = prompt_rows[sid]

        relation = canonical_relation(
            prompt.get(
                "answer"
            )
        )

        if relation not in CANONICAL_SET:
            continue

        question_text = extract_user_text(
            prompt["question"]
        )

        subject, reference = parse_objects(
            question_text
        )

        image_id = int(
            annotations[sid][0]
        )

        image_path = (
            Path(args.data_root)
            / "val2017"
            / f"{image_id:012d}.jpg"
        )

        records.append({
            "sid": sid,
            "dataset": "coco",
            "image_path": str(image_path),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "question_text": question_text,
            "pair_group": (
                " || ".join(
                    sorted(
                        [
                            subject.lower(),
                            reference.lower(),
                        ]
                    )
                )
            ),
        })

    print(
        f"[COCO] N={len(records)} | "
        f"{dict(Counter(r['relation'] for r in records))}"
    )

    return records


def resolve_controlled_image_path(raw_path, args):
    candidate = Path(
        str(raw_path)
    )

    if candidate.exists():
        return candidate

    candidate2 = (
        Path(args.data_root)
        / candidate
    )

    if candidate2.exists():
        return candidate2

    # Common case: data JSON stores path relative to repo root.
    candidate3 = Path(".") / candidate

    if candidate3.exists():
        return candidate3

    return candidate


def load_controlled_a(args):
    prompt_rows = load_jsonl_by_id(
        args.controlled_prompt_jsonl
    )

    with Path(args.controlled_json).open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    records = []

    for sid in sorted(prompt_rows):
        if sid < 0 or sid >= len(data):
            continue

        prompt = prompt_rows[sid]

        relation = canonical_relation(
            prompt.get(
                "answer"
            )
        )

        if relation not in CANONICAL_SET:
            continue

        question_text = extract_user_text(
            prompt["question"]
        )

        subject, reference = parse_objects(
            question_text
        )

        row = data[sid]

        if not isinstance(row, dict):
            raise TypeError(
                "controlled_images_dataset.json is expected to contain dict rows"
            )

        raw_image_path = row.get(
            "image_path"
        )

        if not raw_image_path:
            raise KeyError(
                f"Controlled A sid={sid} has no image_path"
            )

        image_path = resolve_controlled_image_path(
            raw_image_path,
            args,
        )

        records.append({
            "sid": sid,
            "dataset": "controlled_a",
            "image_path": str(image_path),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "question_text": question_text,
            "pair_group": (
                " || ".join(
                    sorted(
                        [
                            subject.lower(),
                            reference.lower(),
                        ]
                    )
                )
            ),
        })

    print(
        f"[Controlled A] N={len(records)} | "
        f"{dict(Counter(r['relation'] for r in records))}"
    )

    return records


def canonical_json_row(row):
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def find_vg_image(image_root, image_id):
    image_root = Path(
        image_root
    )

    stem = str(
        image_id
    ).strip()

    for suffix in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ):
        candidate = (
            image_root
            / f"{stem}{suffix}"
        )

        if candidate.exists():
            return candidate

    return None


def load_vg2_lr(args):
    filtered_path = Path(
        args.vg_filtered_json
    )

    original_path = Path(
        args.vg_original_json
    )

    prompt_path = Path(
        args.vg_six_prompt_jsonl
    )

    image_root = Path(
        args.vg_image_root
    )

    for path in (
        filtered_path,
        original_path,
        prompt_path,
        image_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    with filtered_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        filtered = json.load(handle)

    with original_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        original = json.load(handle)

    prompts = load_jsonl_by_id(
        prompt_path
    )

    positions = defaultdict(
        list
    )

    for old_id, row in enumerate(
        original
    ):
        positions[
            canonical_json_row(
                row
            )
        ].append(
            old_id
        )

    cursors = defaultdict(
        int
    )

    records = []
    dropped = Counter()

    for new_id, row in enumerate(
        filtered
    ):
        key = canonical_json_row(
            row
        )

        matches = positions.get(
            key,
            [],
        )

        cursor = cursors[
            key
        ]

        if cursor >= len(matches):
            raise RuntimeError(
                f"Could not map VG filtered row {new_id} to original dataset"
            )

        old_id = matches[
            cursor
        ]

        cursors[
            key
        ] += 1

        prompt = prompts[
            old_id
        ]

        relation = canonical_relation(
            prompt.get(
                "answer"
            )
        )

        if relation not in {
            "left",
            "right",
        }:
            dropped[
                str(
                    relation
                )
            ] += 1
            continue

        raw_question = str(
            prompt[
                "question"
            ]
        )

        # Binary target prompt while preserving object wording.
        binary_question = re.sub(
            (
                r"Answer\s+with\s+left\s*,\s*right\s*,\s*front\s*,\s*"
                r"behind\s*,\s*above\s*(?:,?\s*or\s*)below\s*\.?"
            ),
            "Answer with left or right.",
            raw_question,
            flags=re.I,
        )

        question_text = extract_user_text(
            binary_question
        )

        subject, reference = parse_objects(
            question_text
        )

        image_id = row[
            0
        ]

        image_path = find_vg_image(
            image_root,
            image_id,
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Missing VG image id={image_id} under {image_root}"
            )

        records.append({
            "sid": new_id,
            "dataset": "vg2_lr",
            "image_path": str(image_path),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "question_text": question_text,
            "pair_group": (
                " || ".join(
                    sorted(
                        [
                            subject.lower(),
                            reference.lower(),
                        ]
                    )
                )
            ),
        })

    print(
        f"[VG2-LR] N={len(records)} | "
        f"{dict(Counter(r['relation'] for r in records))} | "
        f"dropped={dict(dropped)}"
    )

    return records


def load_dataset(name, args):
    if name == "coco":
        return load_coco(
            args
        )

    if name == "controlled_a":
        return load_controlled_a(
            args
        )

    if name == "vg2_lr":
        return load_vg2_lr(
            args
        )

    raise ValueError(
        name
    )


# =============================================================================
# Splits
# =============================================================================

def split_stratified(
    records,
    train_frac,
    seed,
):
    rng = random.Random(
        seed
    )

    by_relation = defaultdict(
        list
    )

    for record in records:
        by_relation[
            record[
                "relation"
            ]
        ].append(
            record
        )

    train = []
    test = []

    for relation in sorted(
        by_relation
    ):
        rows = sorted(
            by_relation[
                relation
            ],
            key=lambda item:
                item["sid"],
        )

        rng.shuffle(
            rows
        )

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
            rows[
                :n_train
            ]
        )

        test.extend(
            rows[
                n_train:
            ]
        )

    return (
        sorted(
            train,
            key=lambda item:
                item["sid"],
        ),
        sorted(
            test,
            key=lambda item:
                item["sid"],
        ),
    )


def split_controlled_by_pair(
    records,
    train_frac,
    seed,
):
    groups = defaultdict(
        list
    )

    for record in records:
        groups[
            record[
                "pair_group"
            ]
        ].append(
            record
        )

    keys = sorted(
        groups
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        keys
    )

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
        keys[
            :n_train
        ]
    )

    train = []
    test = []

    for record in records:
        if (
            record[
                "pair_group"
            ]
            in train_keys
        ):
            train.append(
                record
            )
        else:
            test.append(
                record
            )

    return (
        sorted(
            train,
            key=lambda item:
                item["sid"],
        ),
        sorted(
            test,
            key=lambda item:
                item["sid"],
        ),
    )


def make_split(
    dataset_name,
    records,
    args,
):
    if dataset_name == "controlled_a":
        train, test = split_controlled_by_pair(
            records,
            args.train_frac,
            args.seed,
        )

    else:
        train, test = split_stratified(
            records,
            args.train_frac,
            args.seed,
        )

    print(
        f"[{dataset_name} split] TRAIN={len(train)} "
        f"{dict(Counter(r['relation'] for r in train))}"
    )

    print(
        f"[{dataset_name} split] TEST={len(test)} "
        f"{dict(Counter(r['relation'] for r in test))}"
    )

    return train, test


# =============================================================================
# Model / generation
# =============================================================================

def load_model(args):
    if not hasattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    ):
        raise RuntimeError(
            "This Transformers build has no "
            "Qwen2_5_VLForConditionalGeneration."
        )

    dtype = {
        "auto": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[
        args.dtype
    ]

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {
            "": args.device
        },
    }

    if args.attn_impl != "none":
        kwargs[
            "attn_implementation"
        ] = args.attn_impl

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
    for path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ):
        current = model

        try:
            for part in path.split(
                "."
            ):
                current = getattr(
                    current,
                    part,
                )

            if len(
                current
            ):
                return current, path

        except Exception:
            pass

    raise RuntimeError(
        "Could not locate decoder blocks."
    )


def build_prompt(
    processor,
    image,
    question_text,
):
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
                    "text": question_text,
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
        return question_text


def build_batch(
    processor,
    image,
    record,
    args,
):
    prompt = build_prompt(
        processor,
        image,
        record[
            "question_text"
        ],
    )

    errors = []

    for fn in (
        lambda: processor(
            text=[
                prompt
            ],
            images=[
                image
            ],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ),
    ):
        try:
            batch = fn()
            break

        except Exception as exc:
            errors.append(
                exc
            )

    else:
        raise RuntimeError(
            f"Processor failed: {errors[-1]}"
        )

    return {
        key: (
            value.to(
                args.device
            )
            if torch.is_tensor(
                value
            )
            else value
        )
        for key, value in batch.items()
    }


def generate(
    model,
    processor,
    batch,
    args,
):
    input_len = int(
        batch[
            "input_ids"
        ].shape[
            1
        ]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    if (
        output_ids.shape[
            1
        ]
        > input_len
    ):
        generated = output_ids[
            0,
            input_len:
        ]

    else:
        generated = output_ids[
            0
        ]

    text = processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    del output_ids

    return (
        text,
        canonical_relation(
            text
        ),
    )


def make_gray(
    image,
    value,
):
    value = int(
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
        image.size,
        (
            value,
            value,
            value,
        ),
    )


# =============================================================================
# Last-token hooks
# =============================================================================

def extract_hidden(
    output
):
    if torch.is_tensor(
        output
    ):
        return output, (
            "tensor",
            0,
        )

    if isinstance(
        output,
        tuple,
    ):
        for index, item in enumerate(
            output
        ):
            if torch.is_tensor(
                item
            ):
                return item, (
                    "tuple",
                    index,
                )

    if isinstance(
        output,
        list,
    ):
        for index, item in enumerate(
            output
        ):
            if torch.is_tensor(
                item
            ):
                return item, (
                    "list",
                    index,
                )

    raise RuntimeError(
        f"Could not extract hidden from {type(output)}"
    )


def replace_hidden(
    output,
    descriptor,
    hidden,
):
    kind, index = descriptor

    if kind == "tensor":
        return hidden

    items = list(
        output
    )

    items[
        index
    ] = hidden

    if kind == "tuple":
        return tuple(
            items
        )

    return items


class CaptureLast:
    def __init__(
        self,
        layers,
        selected,
    ):
        self.handles = []
        self.states = {}

        self.done = {
            layer: False
            for layer in selected
        }

        for layer in selected:
            self.handles.append(
                layers[
                    layer
                ].register_forward_hook(
                    self._hook(
                        layer
                    )
                )
            )

    def _hook(
        self,
        layer,
    ):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[
                layer
            ]:
                return output

            hidden, _ = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            self.states[
                layer
            ] = (
                hidden[
                    0,
                    -1,
                    :
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            self.done[
                layer
            ] = True

            return output

        return hook

    def close(
        self
    ):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

    def __enter__(
        self
    ):
        return self

    def __exit__(
        self,
        *_,
    ):
        self.close()


class SteerLast:
    def __init__(
        self,
        layers,
        templates,
        selected,
        target,
        scale,
        mode,
        source,
    ):
        self.handles = []

        self.done = {
            layer: False
            for layer in selected
        }

        self.templates = templates
        self.target = target
        self.scale = float(
            scale
        )
        self.mode = mode
        self.source = source

        for layer in selected:
            self.handles.append(
                layers[
                    layer
                ].register_forward_hook(
                    self._hook(
                        layer
                    )
                )
            )

    def vector(
        self,
        layer,
    ):
        target = self.templates[
            layer
        ][
            self.target
        ]

        if (
            self.mode == "add"
            or self.source
            not in CANONICAL_SET
        ):
            return (
                self.scale
                * target
            )

        source = self.templates[
            layer
        ][
            self.source
        ]

        return (
            self.scale
            * (
                target
                - source
            )
        )

    def _hook(
        self,
        layer,
    ):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[
                layer
            ]:
                return output

            hidden, descriptor = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            vector = torch.as_tensor(
                self.vector(
                    layer
                ),
                device=hidden.device,
                dtype=hidden.dtype,
            )

            edited = hidden.clone()

            edited[
                :,
                -1,
                :
            ] += vector[
                None,
                :
            ]

            self.done[
                layer
            ] = True

            return replace_hidden(
                output,
                descriptor,
                edited,
            )

        return hook

    def close(
        self
    ):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

    def __enter__(
        self
    ):
        return self

    def __exit__(
        self,
        *_,
    ):
        self.close()


# =============================================================================
# Source-template learning / cache
# =============================================================================

def filter_sequence(
    requested,
):
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

    return [
        "all"
    ]


def allowed(
    row,
    mode,
):
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


def source_cache_paths(
    args,
    preset,
    actuator_layers,
):
    root = Path(
        args.source_cache_dir
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    layer_tag = "-".join(
        map(
            str,
            actuator_layers,
        )
    )

    stem = (
        f"{preset['tag']}"
        f"__source-{args.source}"
        f"__L{layer_tag}"
        f"__seed{args.seed}"
        f"__train{args.train_frac:.2f}"
    )

    return (
        root
        / f"{stem}.npz",
        root
        / f"{stem}.json",
    )


def load_source_template_cache(
    npz_path,
    json_path,
    actuator_layers,
):
    if (
        not npz_path.exists()
        or not json_path.exists()
    ):
        return None

    metadata = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as npz:
        templates = {}

        for layer in actuator_layers:
            templates[
                layer
            ] = {}

            for relation in CANONICAL:
                key = (
                    f"L{layer}"
                    f"__{relation}"
                )

                if key not in npz:
                    raise RuntimeError(
                        f"Missing cached vector {key}"
                    )

                templates[
                    layer
                ][
                    relation
                ] = np.asarray(
                    npz[
                        key
                    ],
                    dtype=np.float32,
                )

    return (
        templates,
        metadata,
    )


def save_source_template_cache(
    npz_path,
    json_path,
    templates,
    metadata,
):
    arrays = {}

    for layer, relation_vectors in templates.items():
        for relation, vector in relation_vectors.items():
            arrays[
                f"L{layer}__{relation}"
            ] = np.asarray(
                vector,
                dtype=np.float32,
            )

    np.savez_compressed(
        npz_path,
        **arrays,
    )

    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[source cache] saved {npz_path}"
    )


def collect_source_train(
    model,
    processor,
    layers,
    source_train,
    actuator_layers,
    args,
):
    rows = []
    deltas = {}

    for record in tqdm(
        source_train,
        desc=f"SOURCE {args.source} TRAIN Real/Gray",
    ):
        real = None
        gray = None

        try:
            real = Image.open(
                record[
                    "image_path"
                ]
            ).convert(
                "RGB"
            )

            gray = make_gray(
                real,
                args.gray_value,
            )

            real_batch = build_batch(
                processor,
                real,
                record,
                args,
            )

            with CaptureLast(
                layers,
                actuator_layers,
            ) as capture:
                real_text, real_pred = generate(
                    model,
                    processor,
                    real_batch,
                    args,
                )

                real_states = dict(
                    capture.states
                )

            del real_batch

            gray_batch = build_batch(
                processor,
                gray,
                record,
                args,
            )

            with CaptureLast(
                layers,
                actuator_layers,
            ) as capture:
                gray_text, gray_pred = generate(
                    model,
                    processor,
                    gray_batch,
                    args,
                )

                gray_states = dict(
                    capture.states
                )

            del gray_batch

            gt = record[
                "relation"
            ]

            rows.append({
                "sid": record["sid"],
                "relation": gt,
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
            })

            deltas[
                record[
                    "sid"
                ]
            ] = {
                layer: (
                    real_states[
                        layer
                    ]
                    - gray_states[
                        layer
                    ]
                ).astype(
                    np.float32
                )
                for layer in actuator_layers
            }

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    return rows, deltas


def fit_source_templates(
    source_train,
    rows,
    deltas,
    actuator_layers,
    requested_filter,
):
    record_map = {
        record[
            "sid"
        ]:
            record
        for record in source_train
    }

    row_map = {
        row[
            "sid"
        ]:
            row
        for row in rows
    }

    for mode in filter_sequence(
        requested_filter
    ):
        bags = {
            layer: {
                relation: []
                for relation in CANONICAL
            }
            for layer in actuator_layers
        }

        used = []

        for sid, layer_values in deltas.items():
            if (
                sid not in row_map
                or not allowed(
                    row_map[
                        sid
                    ],
                    mode,
                )
            ):
                continue

            relation = record_map[
                sid
            ][
                "relation"
            ]

            for layer in actuator_layers:
                bags[
                    layer
                ][
                    relation
                ].append(
                    layer_values[
                        layer
                    ]
                )

            used.append(
                sid
            )

        missing = [
            (
                layer,
                relation,
            )
            for layer in actuator_layers
            for relation in CANONICAL
            if not bags[
                layer
            ][
                relation
            ]
        ]

        if missing:
            print(
                f"[source template] filter={mode} "
                f"missing cells={missing[:8]} -> relax"
            )
            continue

        templates = {}

        for layer in actuator_layers:
            means = {
                relation: np.stack(
                    bags[
                        layer
                    ][
                        relation
                    ],
                    axis=0,
                ).mean(
                    axis=0
                ).astype(
                    np.float32
                )
                for relation in CANONICAL
            }

            balanced_global = np.stack(
                [
                    means[
                        relation
                    ]
                    for relation in CANONICAL
                ],
                axis=0,
            ).mean(
                axis=0
            ).astype(
                np.float32
            )

            templates[
                layer
            ] = {
                relation: (
                    means[
                        relation
                    ]
                    - balanced_global
                ).astype(
                    np.float32
                )
                for relation in CANONICAL
            }

        counts = Counter(
            record_map[
                sid
            ][
                "relation"
            ]
            for sid in used
        )

        print(
            f"[source template] requested={requested_filter} "
            f"used={mode} N={len(used)} counts={dict(counts)}"
        )

        return (
            templates,
            mode,
            used,
        )

    raise RuntimeError(
        "Could not fit all four source relation vectors."
    )


def get_or_build_source_templates(
    model,
    processor,
    layers,
    source_train,
    actuator_layers,
    args,
    preset,
    outdir,
):
    npz_path, json_path = source_cache_paths(
        args,
        preset,
        actuator_layers,
    )

    if not args.rebuild_source:
        cached = load_source_template_cache(
            npz_path,
            json_path,
            actuator_layers,
        )

        if cached is not None:
            templates, metadata = cached

            if (
                metadata.get(
                    "model_id"
                )
                == args.model_id
                and metadata.get(
                    "source"
                )
                == args.source
                and metadata.get(
                    "actuator_layers"
                )
                == actuator_layers
            ):
                print(
                    f"[source cache] REUSED {npz_path}"
                )

                return (
                    templates,
                    metadata,
                )

    rows, deltas = collect_source_train(
        model,
        processor,
        layers,
        source_train,
        actuator_layers,
        args,
    )

    write_csv(
        outdir
        / "source_train_generation.csv",
        rows,
    )

    (
        templates,
        filter_used,
        used_sids,
    ) = fit_source_templates(
        source_train,
        rows,
        deltas,
        actuator_layers,
        args.template_filter,
    )

    metadata = {
        "model_id": args.model_id,
        "source": args.source,
        "actuator_layers": actuator_layers,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "requested_filter": args.template_filter,
        "filter_used": filter_used,
        "source_train_n": len(source_train),
        "template_train_n": len(used_sids),
        "template_relation_counts": dict(
            Counter(
                record[
                    "relation"
                ]
                for record in source_train
                if record[
                    "sid"
                ]
                in set(
                    used_sids
                )
            )
        ),
    }

    save_source_template_cache(
        npz_path,
        json_path,
        templates,
        metadata,
    )

    return (
        templates,
        metadata,
    )


# =============================================================================
# Target baseline
# =============================================================================

def load_existing_baseline(
    path,
    target_test,
):
    if not path:
        return None

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    record_map = {
        record[
            "sid"
        ]:
            record
        for record in target_test
    }

    result = {}

    for row in read_csv(
        path
    ):
        sid_value = None

        for key in (
            "sid",
            "sample_index",
            "id",
        ):
            if (
                key in row
                and str(
                    row[
                        key
                    ]
                ).strip()
            ):
                sid_value = int(
                    row[
                        key
                    ]
                )
                break

        if (
            sid_value is None
            or sid_value not in record_map
        ):
            continue

        pred_value = ""

        for key in (
            "baseline_pred",
            "pred",
            "real_pred",
        ):
            if key in row:
                pred_value = row[
                    key
                ]
                break

        pred = canonical_relation(
            pred_value
        )

        result[
            sid_value
        ] = {
            "sid": sid_value,
            "baseline_pred": pred,
            "baseline_correct": int(
                pred
                == record_map[
                    sid_value
                ][
                    "relation"
                ]
            ),
            "baseline_text": row.get(
                "baseline_text",
                row.get(
                    "text",
                    "",
                ),
            ),
        }

    expected = {
        record[
            "sid"
        ]
        for record in target_test
    }

    if set(
        result
    ) != expected:
        missing = sorted(
            expected
            - set(
                result
            )
        )

        print(
            f"[baseline reuse] incompatible CSV; "
            f"missing {len(missing)} target TEST ids -> regenerate"
        )

        return None

    print(
        f"[baseline reuse] {path} N={len(result)}"
    )

    return [
        result[
            record[
                "sid"
            ]
        ]
        for record in target_test
    ]


def generate_target_baseline(
    model,
    processor,
    target_test,
    args,
):
    rows = []

    for record in tqdm(
        target_test,
        desc=f"TARGET {args.target} baseline",
    ):
        image = None

        try:
            image = Image.open(
                record[
                    "image_path"
                ]
            ).convert(
                "RGB"
            )

            batch = build_batch(
                processor,
                image,
                record,
                args,
            )

            text, pred = generate(
                model,
                processor,
                batch,
                args,
            )

            rows.append({
                "sid": record["sid"],
                "relation": record["relation"],
                "baseline_pred": pred or "",
                "baseline_correct": int(
                    pred
                    == record[
                        "relation"
                    ]
                ),
                "baseline_text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


# =============================================================================
# Target steering
# =============================================================================

def run_target_condition(
    model,
    processor,
    layers,
    templates,
    actuator_layers,
    target_test,
    baseline_rows,
    args,
    condition,
    mode,
    target_policy,
):
    baseline_map = {
        int(
            row[
                "sid"
            ]
        ):
            row
        for row in baseline_rows
    }

    rows = []

    for record in tqdm(
        target_test,
        desc=condition,
    ):
        sid = record[
            "sid"
        ]

        base = baseline_map[
            sid
        ]

        base_pred = canonical_relation(
            base.get(
                "baseline_pred",
                ""
            )
        )

        gt = record[
            "relation"
        ]

        if target_policy == "gt":
            target_relation = gt

        elif target_policy == "wrong":
            target_relation = opposite_relation(
                gt
            )

        else:
            raise ValueError(
                target_policy
            )

        image = None

        try:
            image = Image.open(
                record[
                    "image_path"
                ]
            ).convert(
                "RGB"
            )

            batch = build_batch(
                processor,
                image,
                record,
                args,
            )

            with SteerLast(
                layers=layers,
                templates=templates,
                selected=actuator_layers,
                target=target_relation,
                scale=args.scale,
                mode=mode,
                source=base_pred,
            ):
                text, pred = generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            baseline_correct = int(
                base[
                    "baseline_correct"
                ]
            )

            edit_correct = int(
                pred
                == gt
            )

            rows.append({
                "condition": condition,
                "sid": sid,
                "relation": gt,
                "vector_relation": target_relation,
                "baseline_pred": base_pred or "",
                "edit_pred": pred or "",
                "baseline_correct": baseline_correct,
                "edit_correct": edit_correct,
                "W2C": int(
                    baseline_correct == 0
                    and edit_correct == 1
                ),
                "C2W": int(
                    baseline_correct == 1
                    and edit_correct == 0
                ),
                "changed": int(
                    (pred or "")
                    != (base_pred or "")
                ),
                "edit_text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


def summarize(
    rows,
    condition,
):
    n = len(
        rows
    )

    base_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in rows
    )

    edit_acc = safe_mean(
        row[
            "edit_correct"
        ]
        for row in rows
    )

    wrong_n = sum(
        1
        - int(
            row[
                "baseline_correct"
            ]
        )
        for row in rows
    )

    correct_n = (
        n
        - wrong_n
    )

    w2c = sum(
        int(
            row[
                "W2C"
            ]
        )
        for row in rows
    )

    c2w = sum(
        int(
            row[
                "C2W"
            ]
        )
        for row in rows
    )

    return {
        "condition": condition,
        "N": n,
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
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
            row[
                "changed"
            ]
            for row in rows
        ),
    }


def relation_summary(
    rows,
):
    output = []

    for condition in sorted(
        set(
            row[
                "condition"
            ]
            for row in rows
        )
    ):
        condition_rows = [
            row
            for row in rows
            if row[
                "condition"
            ]
            == condition
        ]

        for relation in CANONICAL:
            subset = [
                row
                for row in condition_rows
                if row[
                    "relation"
                ]
                == relation
            ]

            if not subset:
                continue

            output.append({
                "condition": condition,
                "relation": relation,
                "N": len(subset),
                "base_acc": safe_mean(
                    row[
                        "baseline_correct"
                    ]
                    for row in subset
                ),
                "edit_acc": safe_mean(
                    row[
                        "edit_correct"
                    ]
                    for row in subset
                ),
                "gain": (
                    safe_mean(
                        row[
                            "edit_correct"
                        ]
                        for row in subset
                    )
                    - safe_mean(
                        row[
                            "baseline_correct"
                        ]
                        for row in subset
                    )
                ),
                "W2C": sum(
                    int(
                        row[
                            "W2C"
                        ]
                    )
                    for row in subset
                ),
                "C2W": sum(
                    int(
                        row[
                            "C2W"
                        ]
                    )
                    for row in subset
                ),
            })

    return output


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.source == args.target:
        raise ValueError(
            "--source and --target should be different for a transfer experiment."
        )

    allowed_pairs = {
        (
            "coco",
            "controlled_a",
        ),
        (
            "coco",
            "vg2_lr",
        ),
        (
            "controlled_a",
            "coco",
        ),
    }

    if (
        args.source,
        args.target,
    ) not in allowed_pairs:
        raise ValueError(
            "Supported pairs are: coco->controlled_a, "
            "coco->vg2_lr, controlled_a->coco."
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

    preset = model_preset(
        args.model_id
    )

    actuator_layers = preset[
        "actuator_layers"
    ]

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

    source_records = load_dataset(
        args.source,
        args,
    )

    target_records = load_dataset(
        args.target,
        args,
    )

    source_train, source_unused_test = make_split(
        args.source,
        source_records,
        args,
    )

    target_unused_train, target_test = make_split(
        args.target,
        target_records,
        args,
    )

    del source_unused_test
    del target_unused_train

    print(
        "\n"
        + "=" * 165
    )

    print(
        "CROSS-DATASET LATE DIRECTION TRANSFER"
    )

    print(
        "=" * 165
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"source={args.source}"
    )

    print(
        f"target={args.target}"
    )

    print(
        f"actuator_layers={actuator_layers} [REUSED FROM COCO SEARCH]"
    )

    print(
        f"source_TRAIN={len(source_train)}"
    )

    print(
        f"target_TEST={len(target_test)}"
    )

    print(
        "NO TARGET SAMPLE IS USED TO FIT THE TRANSFER VECTORS"
    )

    print(
        "=" * 165
    )

    model, processor = load_model(
        args
    )

    layers, layer_path = resolve_layers(
        model
    )

    if max(
        actuator_layers
    ) >= len(
        layers
    ):
        raise RuntimeError(
            f"Model has {len(layers)} blocks, "
            f"cannot use actuator layers {actuator_layers}"
        )

    print(
        f"decoder={layer_path} | blocks={len(layers)}"
    )

    source_templates, source_meta = get_or_build_source_templates(
        model,
        processor,
        layers,
        source_train,
        actuator_layers,
        args,
        preset,
        outdir,
    )

    baseline_rows = load_existing_baseline(
        args.target_baseline_csv,
        target_test,
    )

    if baseline_rows is None:
        baseline_rows = generate_target_baseline(
            model,
            processor,
            target_test,
            args,
        )

    write_csv(
        outdir
        / "target_test_baseline.csv",
        baseline_rows,
    )

    baseline_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in baseline_rows
    )

    conditions = [
        (
            "transfer_oracle_add",
            "add",
            "gt",
        ),
        (
            "transfer_oracle_contrast",
            "contrast",
            "gt",
        ),
        (
            "transfer_wrong_add",
            "add",
            "wrong",
        ),
    ]

    all_details = []
    summaries = []

    for (
        condition,
        mode,
        target_policy,
    ) in conditions:
        rows = run_target_condition(
            model,
            processor,
            layers,
            source_templates,
            actuator_layers,
            target_test,
            baseline_rows,
            args,
            condition,
            mode,
            target_policy,
        )

        all_details.extend(
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
            / "transfer_details.csv",
            all_details,
        )

        write_csv(
            outdir
            / "transfer_summary.csv",
            summaries,
        )

    write_csv(
        outdir
        / "transfer_by_relation.csv",
        relation_summary(
            all_details
        ),
    )

    print(
        "\n"
        + "=" * 175
    )

    print(
        "FINAL TEST — CROSS-DATASET DIRECTION TRANSFER"
    )

    print(
        "=" * 175
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"source={args.source}"
    )

    print(
        f"target={args.target}"
    )

    print(
        f"decoder_blocks={len(layers)}"
    )

    print(
        f"causal_layers={actuator_layers}"
    )

    print(
        f"source_template_filter={source_meta.get('filter_used')}"
    )

    print(
        f"source_template_N={source_meta.get('template_train_n')}"
    )

    print(
        f"target_baseline={baseline_acc:.4f}"
    )

    print()

    print(
        "condition                    | "
        "acc base->edit gain | W2C/wrong | C2W/correct | net | changed"
    )

    for row in summaries:
        print(
            f"{row['condition']:28s} | "
            f"{row['base_acc']:.4f}->"
            f"{row['edit_acc']:.4f} "
            f"{row['gain']:+.4f} | "
            f"{row['W2C']}/{row['W2C_rate_wrong']:.3f} | "
            f"{row['C2W']}/{row['C2W_rate_correct']:.3f} | "
            f"{row['net']:+d} | "
            f"{row['changed_rate']:.3f}"
        )

    print()
    print(
        "Interpret transfer_oracle_add first. "
        "transfer_wrong_add is the relation-specificity control."
    )

    summary_json = {
        "model_id": args.model_id,
        "source": args.source,
        "target": args.target,
        "decoder_blocks": len(layers),
        "actuator_layers": actuator_layers,
        "source_train_n": len(source_train),
        "target_test_n": len(target_test),
        "source_template_metadata": source_meta,
        "target_baseline_acc": baseline_acc,
        "important_note": (
            "Source vectors are learned only from SOURCE TRAIN. "
            "Target GT is used only to select which already-learned source "
            "vector to inject. No target fitting/refitting is performed."
        ),
    }

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary_json,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
