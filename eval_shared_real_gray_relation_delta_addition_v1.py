
# -*- coding: utf-8 -*-

"""
eval_shared_real_gray_relation_delta_addition_v1.py

Purpose
-------
Find a NEW spatial direction directly from multiple Real-vs-Gray samples,
instead of using the old h_sub - h_ref Direction codebook.

Core idea
---------
For each TRAIN sample i and layer l:

PAIR site:
    r_real = mean(h_sub_real) - mean(h_ref_real)
    r_gray = mean(h_sub_gray) - mean(h_ref_gray)

    delta_r_i,l = r_real - r_gray

LAST site:
    delta_last_i,l = h_last_real - h_last_gray

For samples with the same spatial relation g, average the Real-Gray delta:

    mu_g,l = E[delta_i,l | relation = g]

Then remove the relation-independent Real-vs-Gray component using a BALANCED
global mean across the four relation means:

    mu_all,l = (mu_left + mu_right + mu_above + mu_below) / 4

    s_g,l = mu_g,l - mu_all,l

This s_g,l is the candidate "shared relation-specific Real-Gray direction".

Why this is different from the old Direction
---------------------------------------------
Old:
    Direction was extracted from h_sub - h_ref itself.

New:
    Direction is extracted from the information that REAL adds relative to GRAY:

        (h_sub - h_ref)_real - (h_sub - h_ref)_gray

    and then averaged across many samples with the SAME relation.

Sample-specific semantics / appearance should vary across samples and tend to
cancel, while a relation-specific component should remain if it is shared.

Causal sufficiency test
-----------------------
TEST cohort defaults to:

    Real generation = correct
    Gray generation = wrong

For GT relation g, add the TRAIN-derived s_g,l to the Gray run.

PAIR:
    h_sub_gray += 0.5 * scale * s_g,l
    h_ref_gray -= 0.5 * scale * s_g,l

so:
    (h_sub - h_ref)_gray += scale * s_g,l

LAST:
    h_last_gray += scale * s_g,l

All metrics use actual model.generate().

Default scan
------------
    pair/text-object site: L10-L24
    last-token site:       L25-L27

Runs:
    * each pair layer individually
    * pair multi-layer
    * each last layer individually
    * last multi-layer
    * joint pair + last multi-layer

Optional controls
-----------------
--controls global
    Add the relation-independent global Real-Gray mean instead of s_GT.

--controls wrong
    Add a deterministic WRONG relation template instead of s_GT.

--controls global,wrong
    Run both controls too.

The default is --controls none for a quick first run.

Train filtering
---------------
Default:
    --train-filter real_correct_gray_wrong

This fits templates only from TRAIN samples where visual evidence actually
matters behaviorally: Real is correct and Gray is wrong.

Other options:
    real_correct
    all

Data / split
------------
No records CSV is needed.

Uses:
    prompts/COCO_QA_two_obj_with_answer_four_options.jsonl
    data/coco_qa_two_obj.json
    data/val2017/{image_id:012d}.jpg

Uses only the TRAIN/TEST assignment from:
    <split-dir>/sample_split_and_generation.csv

It does NOT use old Direction vectors.npz.

Recommended quick run
---------------------
CUDA_VISIBLE_DEVICES=0 python eval_shared_real_gray_relation_delta_addition_v1.py \
  --split-dir output/qwen7b_layer_direction_scan_v1 \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --pair-layers 10-24 \
  --last-layers 25-27 \
  --train-filter real_correct_gray_wrong \
  --controls none \
  --scale 1.0 \
  --output-dir output/qwen7b_shared_real_gray_relation_delta_v1 \
  --overwrite

For a faster first sanity check:
    --pair-layers 14-18 --last-layers "" --no-joint

For controls after seeing a promising result:
    --controls global,wrong
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
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


RELS = ("left", "right", "above", "below")
RELSET = set(RELS)

WRONG_REL = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing sample_split_and_generation.csv.",
    )

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--annotation-json",
        default="data/coco_qa_two_obj.json",
    )
    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument("--pair-layers", default="10-24")
    p.add_argument("--last-layers", default="25-27")

    p.add_argument(
        "--train-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )

    p.add_argument(
        "--cohort",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "all_test"],
    )

    p.add_argument(
        "--controls",
        default="none",
        help="none | global | wrong | global,wrong",
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument(
        "--no-single",
        action="store_true",
        help="Skip individual-layer scans.",
    )
    p.add_argument(
        "--no-multi",
        action="store_true",
        help="Skip pair-multi and last-multi.",
    )
    p.add_argument(
        "--no-joint",
        action="store_true",
        help="Skip joint pair-window + last-window multi-layer intervention.",
    )

    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--max-cohort-samples", type=int, default=None)

    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def normalize_relation(x):
    s = str(x).strip().lower()

    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    if re.search(r"\babove\b", s) or re.search(r"\bover\b", s):
        return "above"
    if (
        re.search(r"\bbelow\b", s)
        or re.search(r"\bunder\b", s)
        or re.search(r"\bbeneath\b", s)
    ):
        return "below"

    return s


def parse_layers(spec, n_layers):
    spec = str(spec).strip()

    if not spec:
        return []

    values = []

    for piece in spec.split(","):
        piece = piece.strip()

        if not piece:
            continue

        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            values.extend(range(a, b + step, step))
        else:
            values.append(int(piece))

    values = sorted(set(values))

    bad = [
        l
        for l in values
        if l < 0 or l >= n_layers
    ]

    if bad:
        raise ValueError(
            f"Invalid layers={bad}; valid model range is 0..{n_layers - 1}"
        )

    return values


def parse_controls(spec):
    spec = str(spec).strip().lower()

    if not spec or spec == "none":
        return []

    controls = [
        x.strip()
        for x in spec.split(",")
        if x.strip()
    ]

    allowed = {
        "global",
        "wrong",
    }

    bad = [
        x
        for x in controls
        if x not in allowed
    ]

    if bad:
        raise ValueError(
            f"Invalid controls={bad}; allowed={sorted(allowed)}"
        )

    return controls


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


# =============================================================================
# Split metadata + COCO-two records
# =============================================================================

def load_split(split_dir):
    path = (
        Path(split_dir)
        / "sample_split_and_generation.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    split = {}

    for row in read_csv(path):
        sid = int(
            row["sample_index"]
        )

        split[
            sid
        ] = str(
            row.get(
                "split",
                "",
            )
        ).strip().lower()

    return split


def parse_subject_reference(question):
    q = str(question)

    patterns = [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for pat in patterns:
        match = re.search(
            pat,
            q,
            flags=re.I | re.S,
        )

        if match:
            subject = re.sub(
                r"\s+",
                " ",
                match.group(1),
            ).strip()

            reference = re.sub(
                r"\s+",
                " ",
                match.group(2),
            ).strip()

            return subject, reference

    return None, None


def load_records(
    prompt_jsonl,
    annotation_json,
    data_root,
    split_map,
):
    with open(
        annotation_json,
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    prompts = []

    with open(
        prompt_jsonl,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                prompts.append(
                    json.loads(line)
                )

    records = {}

    for row_no, row in enumerate(prompts):
        sid = int(
            row.get(
                "id",
                row_no,
            )
        )

        if sid not in split_map:
            continue

        if sid < 0 or sid >= len(
            annotations
        ):
            continue

        question = str(
            row.get(
                "question",
                "",
            )
        )

        subject, reference = parse_subject_reference(
            question
        )

        if subject is None or reference is None:
            continue

        gt = normalize_relation(
            row.get(
                "answer",
                "",
            )
        )

        if gt not in RELSET:
            continue

        ann = annotations[
            sid
        ]

        image_id = int(
            ann[0]
        )

        image_path = (
            Path(data_root)
            / "val2017"
            / f"{image_id:012d}.jpg"
        )

        records[
            sid
        ] = {
            "sid": sid,
            "gt": gt,
            "subject": subject,
            "reference": reference,
            "image_path": str(
                image_path
            ),
            "split": split_map[
                sid
            ],
        }

    existing = sum(
        Path(
            rec["image_path"]
        ).exists()
        for rec in records.values()
    )

    print(
        f"[records] N={len(records)} | "
        f"images={existing}/{len(records)}"
    )

    if not records:
        raise RuntimeError(
            "No records loaded."
        )

    return records


# =============================================================================
# Model / processor
# =============================================================================

def get_attr_path(
    obj,
    path,
):
    current = obj

    for part in path.split("."):
        current = getattr(
            current,
            part,
        )

    return current


def resolve_decoder_layers(model):
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
    ]

    for path in candidates:
        try:
            layers = get_attr_path(
                model,
                path,
            )

            if len(
                layers
            ) > 0:
                return layers, path

        except Exception:
            pass

    raise RuntimeError(
        "Could not resolve language decoder layers."
    )


def load_model_and_processor(
    model_id,
    dtype,
    device,
    attn_impl,
):
    class_names = [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]

    model_cls = next(
        (
            getattr(
                transformers,
                name,
            )
            for name in class_names
            if hasattr(
                transformers,
                name,
            )
        ),
        None,
    )

    if model_cls is None:
        raise RuntimeError(
            "No supported multimodal generation model class found."
        )

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {
            "": device
        },
    }

    if attn_impl != "none":
        kwargs[
            "attn_implementation"
        ] = attn_impl

    print(
        f"[model] {model_cls.__name__} | "
        f"{model_id} | {device}"
    )

    try:
        model = model_cls.from_pretrained(
            model_id,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        model = model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **kwargs,
        )

    model.eval()

    # Match the previous slow-processor protocol as closely as possible.
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    return model, processor


def make_gray_image(
    real_image,
    gray_value,
):
    value = int(
        max(
            0,
            min(
                255,
                gray_value,
            ),
        )
    )

    return Image.new(
        "RGB",
        real_image.size,
        (
            value,
            value,
            value,
        ),
    )


def build_batch(
    processor,
    image,
    question,
    device,
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
                    "text": question,
                },
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
        prompt = processor.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    last_error = None

    attempts = [
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
    ]

    for fn in attempts:
        try:
            batch = fn()
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"Processor failed: {last_error}"
        )

    batch = {
        key: (
            value.to(
                device
            )
            if torch.is_tensor(
                value
            )
            else value
        )
        for key, value in batch.items()
    }

    return batch


def parse_pred(text):
    text = str(
        text
    ).lower()

    hits = []

    for relation, pattern in [
        (
            "left",
            r"\bleft\b",
        ),
        (
            "right",
            r"\bright\b",
        ),
        (
            "above",
            r"\babove\b",
        ),
        (
            "below",
            r"\bbelow\b",
        ),
        (
            "below",
            r"\bunder(?:neath)?\b",
        ),
        (
            "below",
            r"\bbeneath\b",
        ),
    ]:
        match = re.search(
            pattern,
            text,
        )

        if match:
            hits.append(
                (
                    match.start(),
                    relation,
                )
            )

    return (
        sorted(
            hits
        )[0][1]
        if hits
        else None
    )


def generate(
    model,
    processor,
    batch,
    max_new_tokens,
):
    input_len = int(
        batch[
            "input_ids"
        ].shape[1]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    generation_text = processor.tokenizer.decode(
        output_ids[
            0,
            input_len:,
        ],
        skip_special_tokens=True,
    ).strip()

    pred = parse_pred(
        generation_text
    )

    del output_ids

    return generation_text, pred


# =============================================================================
# Token spans
# =============================================================================

def find_subsequence(
    sequence: Sequence[int],
    pattern: Sequence[int],
):
    if not pattern:
        return []

    hits = []
    m = len(
        pattern
    )

    for i in range(
        len(sequence)
        - m
        + 1
    ):
        if list(
            sequence[
                i:i + m
            ]
        ) == list(
            pattern
        ):
            hits.append(i)

    return hits


def phrase_spans(
    tokenizer,
    full_ids,
    phrase,
):
    variants = [
        phrase,
        " " + phrase,
        phrase.strip(),
        " " + phrase.strip(),
    ]

    spans = []
    seen = set()

    for text in variants:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if not ids:
            continue

        for start in find_subsequence(
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
                seen.add(
                    span
                )

                spans.append(
                    list(
                        span
                    )
                )

    return spans


def locate_object_spans(
    tokenizer,
    full_ids,
    subject,
    reference,
):
    subject_spans = phrase_spans(
        tokenizer,
        full_ids,
        subject,
    )

    reference_spans = phrase_spans(
        tokenizer,
        full_ids,
        reference,
    )

    if not subject_spans or not reference_spans:
        return None, None

    best = None

    for subject_span in subject_spans:
        for reference_span in reference_spans:
            if set(
                subject_span
            ) & set(
                reference_span
            ):
                continue

            distance = abs(
                float(
                    np.mean(
                        subject_span
                    )
                )
                - float(
                    np.mean(
                        reference_span
                    )
                )
            )

            score = (
                distance,
                -min(
                    subject_span[
                        0
                    ],
                    reference_span[
                        0
                    ],
                ),
            )

            if (
                best is None
                or score < best[
                    0
                ]
            ):
                best = (
                    score,
                    subject_span,
                    reference_span,
                )

    if best is None:
        return None, None

    return best[
        1
    ], best[
        2
    ]


def infer_last_prompt_position(
    batch,
):
    if "attention_mask" in batch:
        mask = batch[
            "attention_mask"
        ][0]

        nonzero = torch.nonzero(
            mask,
            as_tuple=False,
        ).flatten()

        if len(
            nonzero
        ):
            return int(
                nonzero[
                    -1
                ].item()
            )

    return int(
        batch[
            "input_ids"
        ].shape[1]
        - 1
    )


# =============================================================================
# Hidden capture during actual generate()
# =============================================================================

def extract_hidden(
    output,
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
        f"Could not extract hidden tensor from output type={type(output)}"
    )


def replace_hidden(
    output,
    descriptor,
    hidden,
):
    kind, index = descriptor

    if kind == "tensor":
        return hidden

    values = list(
        output
    )

    values[
        index
    ] = hidden

    return (
        tuple(
            values
        )
        if kind == "tuple"
        else values
    )


class CaptureHooks:
    """
    Capture compact vectors at decoder BLOCK OUTPUT during prompt prefill.

    pair:
        mean(subject tokens) - mean(reference tokens)

    last:
        last prompt token hidden state
    """

    def __init__(
        self,
        decoder_layers,
        pair_layers,
        last_layers,
        subject_span,
        reference_span,
        last_position,
    ):
        self.handles = []
        self.seen = {}

        self.pair_layers = set(
            pair_layers
        )

        self.last_layers = set(
            last_layers
        )

        self.subject_span = list(
            subject_span
        )

        self.reference_span = list(
            reference_span
        )

        self.last_position = int(
            last_position
        )

        self.pair = {}
        self.last = {}

        selected = sorted(
            self.pair_layers
            | self.last_layers
        )

        for layer in selected:
            self.seen[
                layer
            ] = False

            self.handles.append(
                decoder_layers[
                    layer
                ].register_forward_hook(
                    self._make_hook(
                        layer
                    )
                )
            )

    def _make_hook(
        self,
        layer,
    ):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.seen[
                layer
            ]:
                return output

            hidden, _descriptor = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            required_positions = [
                self.last_position
            ]

            if layer in self.pair_layers:
                required_positions += (
                    self.subject_span
                    + self.reference_span
                )

            if (
                required_positions
                and max(
                    required_positions
                ) >= hidden.shape[
                    1
                ]
            ):
                return output

            if layer in self.pair_layers:
                subject_state = hidden[
                    :,
                    self.subject_span,
                    :,
                ].mean(
                    dim=1
                )

                reference_state = hidden[
                    :,
                    self.reference_span,
                    :,
                ].mean(
                    dim=1
                )

                pair = (
                    subject_state
                    - reference_state
                )[
                    0
                ].detach().float().cpu().numpy()

                self.pair[
                    layer
                ] = pair.astype(
                    np.float32
                )

            if layer in self.last_layers:
                last = hidden[
                    0,
                    self.last_position,
                    :,
                ].detach().float().cpu().numpy()

                self.last[
                    layer
                ] = last.astype(
                    np.float32
                )

            self.seen[
                layer
            ] = True

            return output

        return hook

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

        self.handles = []

    def __enter__(self):
        return self

    def __exit__(
        self,
        *_args,
    ):
        self.close()


class AdditionHooks:
    """
    Add one template per selected layer to a GRAY run.

    pair:
        h_sub += +template/2
        h_ref += -template/2

    last:
        h_last += template
    """

    def __init__(
        self,
        decoder_layers,
        templates,
        gt_relation,
        pair_layers,
        last_layers,
        subject_span,
        reference_span,
        last_position,
        scale,
        template_mode,
    ):
        self.handles = []
        self.applied = {}

        self.templates = templates
        self.gt = gt_relation

        self.pair_layers = set(
            pair_layers
        )

        self.last_layers = set(
            last_layers
        )

        self.subject_span = list(
            subject_span
        )

        self.reference_span = list(
            reference_span
        )

        self.last_position = int(
            last_position
        )

        self.scale = float(
            scale
        )

        self.template_mode = template_mode

        selected = sorted(
            self.pair_layers
            | self.last_layers
        )

        for layer in selected:
            self.applied[
                layer
            ] = False

            self.handles.append(
                decoder_layers[
                    layer
                ].register_forward_hook(
                    self._make_hook(
                        layer
                    )
                )
            )

    def _template_for(
        self,
        site,
        layer,
    ):
        entry = self.templates[
            site
        ][
            layer
        ]

        if self.template_mode == "shared":
            vec = entry[
                "shared"
            ][
                self.gt
            ]

        elif self.template_mode == "global":
            vec = entry[
                "global"
            ]

        elif self.template_mode == "wrong":
            vec = entry[
                "shared"
            ][
                WRONG_REL[
                    self.gt
                ]
            ]

        else:
            raise ValueError(
                self.template_mode
            )

        return (
            self.scale
            * np.asarray(
                vec,
                dtype=np.float32,
            )
        )

    def _make_hook(
        self,
        layer,
    ):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.applied[
                layer
            ]:
                return output

            hidden, descriptor = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            required_positions = [
                self.last_position
            ]

            if layer in self.pair_layers:
                required_positions += (
                    self.subject_span
                    + self.reference_span
                )

            if (
                required_positions
                and max(
                    required_positions
                ) >= hidden.shape[
                    1
                ]
            ):
                return output

            edited = hidden.clone()

            if layer in self.pair_layers:
                pair_vec_np = self._template_for(
                    "pair",
                    layer,
                )

                pair_vec = torch.as_tensor(
                    pair_vec_np,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )

                if pair_vec.shape[
                    -1
                ] != hidden.shape[
                    -1
                ]:
                    raise RuntimeError(
                        f"L{layer} pair template dim={pair_vec.shape[-1]} "
                        f"!= hidden dim={hidden.shape[-1]}"
                    )

                half = (
                    0.5
                    * pair_vec
                )

                edited[
                    :,
                    self.subject_span,
                    :,
                ] = (
                    edited[
                        :,
                        self.subject_span,
                        :,
                    ]
                    + half[
                        None,
                        None,
                        :,
                    ]
                )

                edited[
                    :,
                    self.reference_span,
                    :,
                ] = (
                    edited[
                        :,
                        self.reference_span,
                        :,
                    ]
                    - half[
                        None,
                        None,
                        :,
                    ]
                )

            if layer in self.last_layers:
                last_vec_np = self._template_for(
                    "last",
                    layer,
                )

                last_vec = torch.as_tensor(
                    last_vec_np,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )

                if last_vec.shape[
                    -1
                ] != hidden.shape[
                    -1
                ]:
                    raise RuntimeError(
                        f"L{layer} last template dim={last_vec.shape[-1]} "
                        f"!= hidden dim={hidden.shape[-1]}"
                    )

                edited[
                    :,
                    self.last_position,
                    :,
                ] = (
                    edited[
                        :,
                        self.last_position,
                        :,
                    ]
                    + last_vec[
                        None,
                        :,
                    ]
                )

            self.applied[
                layer
            ] = True

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
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

        self.handles = []

    def __enter__(self):
        return self

    def __exit__(
        self,
        *_args,
    ):
        self.close()


# =============================================================================
# One sample: Real/Gray generate + capture
# =============================================================================

def prepare_batch_for_image(
    processor,
    image,
    rec,
    prompt_template,
    device,
):
    question = prompt_template.format(
        subject=rec[
            "subject"
        ],
        reference=rec[
            "reference"
        ],
    )

    batch = build_batch(
        processor,
        image,
        question,
        device,
    )

    ids = batch[
        "input_ids"
    ][0].detach().cpu().tolist()

    subject_span, reference_span = locate_object_spans(
        processor.tokenizer,
        ids,
        rec[
            "subject"
        ],
        rec[
            "reference"
        ],
    )

    last_position = infer_last_prompt_position(
        batch
    )

    return (
        batch,
        subject_span,
        reference_span,
        last_position,
    )


def generate_and_capture(
    model,
    processor,
    decoder_layers,
    image,
    rec,
    pair_layers,
    last_layers,
    args,
    device,
):
    (
        batch,
        subject_span,
        reference_span,
        last_position,
    ) = prepare_batch_for_image(
        processor,
        image,
        rec,
        args.prompt_template,
        device,
    )

    if (
        subject_span is None
        or reference_span is None
    ):
        return {
            "ok": False,
            "error":
                "Could not locate subject/reference token spans.",
        }

    with CaptureHooks(
        decoder_layers=decoder_layers,
        pair_layers=pair_layers,
        last_layers=last_layers,
        subject_span=subject_span,
        reference_span=reference_span,
        last_position=last_position,
    ) as capture:

        text, pred = generate(
            model,
            processor,
            batch,
            args.max_new_tokens,
        )

        pair = dict(
            capture.pair
        )

        last = dict(
            capture.last
        )

    del batch

    return {
        "ok": True,
        "pred": pred,
        "text": text,
        "pair": pair,
        "last": last,
    }


# =============================================================================
# TRAIN: derive shared Real-Gray relation templates
# =============================================================================

def train_sample_allowed(
    real_correct,
    gray_correct,
    train_filter,
):
    if train_filter == "all":
        return True

    if train_filter == "real_correct":
        return bool(
            real_correct
        )

    if train_filter == "real_correct_gray_wrong":
        return bool(
            real_correct
        ) and not bool(
            gray_correct
        )

    raise ValueError(
        train_filter
    )


def collect_train_deltas(
    model,
    processor,
    decoder_layers,
    records,
    train_sids,
    pair_layers,
    last_layers,
    args,
    device,
    outdir,
):
    pair_deltas = {
        layer: {
            rel: []
            for rel in RELS
        }
        for layer in pair_layers
    }

    last_deltas = {
        layer: {
            rel: []
            for rel in RELS
        }
        for layer in last_layers
    }

    rows = []

    for sid in tqdm(
        train_sids,
        desc="TRAIN Real/Gray delta",
    ):
        rec = records[
            sid
        ]

        real_image = None
        gray_image = None

        try:
            real_image = Image.open(
                rec[
                    "image_path"
                ]
            ).convert(
                "RGB"
            )

            gray_image = make_gray_image(
                real_image,
                args.gray_value,
            )

            real = generate_and_capture(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                image=real_image,
                rec=rec,
                pair_layers=pair_layers,
                last_layers=last_layers,
                args=args,
                device=device,
            )

            gray = generate_and_capture(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                image=gray_image,
                rec=rec,
                pair_layers=pair_layers,
                last_layers=last_layers,
                args=args,
                device=device,
            )

            if (
                not real[
                    "ok"
                ]
                or not gray[
                    "ok"
                ]
            ):
                rows.append({
                    "sid": sid,
                    "gt": rec["gt"],
                    "used": 0,
                    "error":
                        real.get("error", "")
                        or gray.get("error", ""),
                })
                continue

            real_correct = int(
                real[
                    "pred"
                ] == rec[
                    "gt"
                ]
            )

            gray_correct = int(
                gray[
                    "pred"
                ] == rec[
                    "gt"
                ]
            )

            used = int(
                train_sample_allowed(
                    real_correct,
                    gray_correct,
                    args.train_filter,
                )
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "real_pred": real["pred"] or "",
                "gray_pred": gray["pred"] or "",
                "real_correct": real_correct,
                "gray_correct": gray_correct,
                "used": used,
                "error": "",
            })

            if not used:
                continue

            gt = rec[
                "gt"
            ]

            for layer in pair_layers:
                if (
                    layer not in real[
                        "pair"
                    ]
                    or layer not in gray[
                        "pair"
                    ]
                ):
                    continue

                delta = (
                    real[
                        "pair"
                    ][
                        layer
                    ]
                    - gray[
                        "pair"
                    ][
                        layer
                    ]
                )

                pair_deltas[
                    layer
                ][
                    gt
                ].append(
                    delta.astype(
                        np.float32
                    )
                )

            for layer in last_layers:
                if (
                    layer not in real[
                        "last"
                    ]
                    or layer not in gray[
                        "last"
                    ]
                ):
                    continue

                delta = (
                    real[
                        "last"
                    ][
                        layer
                    ]
                    - gray[
                        "last"
                    ][
                        layer
                    ]
                )

                last_deltas[
                    layer
                ][
                    gt
                ].append(
                    delta.astype(
                        np.float32
                    )
                )

        except Exception as exc:
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "used": 0,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })

            tqdm.write(
                f"[TRAIN ERROR sid={sid}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real_image is not None:
                real_image.close()

            if gray_image is not None:
                gray_image.close()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(
        outdir
        / "train_real_gray_generation.csv",
        rows,
    )

    return {
        "pair": pair_deltas,
        "last": last_deltas,
    }, rows


def fit_shared_templates(
    collected,
):
    templates = {
        "pair": {},
        "last": {},
    }

    summary = []

    for site in [
        "pair",
        "last",
    ]:
        for layer, relation_lists in collected[
            site
        ].items():

            relation_means = {}

            counts = {}

            for relation in RELS:
                vectors = relation_lists[
                    relation
                ]

                counts[
                    relation
                ] = len(
                    vectors
                )

                if not vectors:
                    raise RuntimeError(
                        f"{site} L{layer}: "
                        f"no TRAIN vectors for relation={relation}. "
                        "Use a less restrictive --train-filter if necessary."
                    )

                relation_means[
                    relation
                ] = np.stack(
                    vectors,
                    axis=0,
                ).mean(
                    axis=0
                ).astype(
                    np.float32
                )

            # Balanced global mean: each relation contributes equally.
            global_mean = np.stack(
                [
                    relation_means[
                        relation
                    ]
                    for relation in RELS
                ],
                axis=0,
            ).mean(
                axis=0
            ).astype(
                np.float32
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
                for relation in RELS
            }

            templates[
                site
            ][
                layer
            ] = {
                "relation_mean":
                    relation_means,
                "global":
                    global_mean,
                "shared":
                    shared,
            }

            summary.append({
                "site": site,
                "layer": layer,
                "n_left":
                    counts["left"],
                "n_right":
                    counts["right"],
                "n_above":
                    counts["above"],
                "n_below":
                    counts["below"],
                "global_norm":
                    float(
                        np.linalg.norm(
                            global_mean
                        )
                    ),
                "shared_left_norm":
                    float(
                        np.linalg.norm(
                            shared[
                                "left"
                            ]
                        )
                    ),
                "shared_right_norm":
                    float(
                        np.linalg.norm(
                            shared[
                                "right"
                            ]
                        )
                    ),
                "shared_above_norm":
                    float(
                        np.linalg.norm(
                            shared[
                                "above"
                            ]
                        )
                    ),
                "shared_below_norm":
                    float(
                        np.linalg.norm(
                            shared[
                                "below"
                            ]
                        )
                    ),
            })

    return templates, summary


# =============================================================================
# TEST Real/Gray baselines
# =============================================================================

def run_test_baselines(
    model,
    processor,
    decoder_layers,
    records,
    test_sids,
    pair_layers,
    last_layers,
    args,
    device,
    outdir,
):
    rows = []

    for sid in tqdm(
        test_sids,
        desc="TEST Real/Gray baseline",
    ):
        rec = records[
            sid
        ]

        real_image = None
        gray_image = None

        try:
            real_image = Image.open(
                rec[
                    "image_path"
                ]
            ).convert(
                "RGB"
            )

            gray_image = make_gray_image(
                real_image,
                args.gray_value,
            )

            real = generate_and_capture(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                image=real_image,
                rec=rec,
                pair_layers=pair_layers,
                last_layers=last_layers,
                args=args,
                device=device,
            )

            gray = generate_and_capture(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                image=gray_image,
                rec=rec,
                pair_layers=pair_layers,
                last_layers=last_layers,
                args=args,
                device=device,
            )

            if (
                not real[
                    "ok"
                ]
                or not gray[
                    "ok"
                ]
            ):
                rows.append({
                    "sid": sid,
                    "gt": rec["gt"],
                    "ok": 0,
                    "error":
                        real.get("error", "")
                        or gray.get("error", ""),
                })
                continue

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "ok": 1,
                "real_pred":
                    real["pred"] or "",
                "gray_pred":
                    gray["pred"] or "",
                "real_correct":
                    int(
                        real[
                            "pred"
                        ] == rec[
                            "gt"
                        ]
                    ),
                "gray_correct":
                    int(
                        gray[
                            "pred"
                        ] == rec[
                            "gt"
                        ]
                    ),
                "real_text":
                    real["text"],
                "gray_text":
                    gray["text"],
                "error": "",
            })

        except Exception as exc:
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "ok": 0,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })

            tqdm.write(
                f"[TEST BASELINE ERROR sid={sid}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real_image is not None:
                real_image.close()

            if gray_image is not None:
                gray_image.close()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(
        outdir
        / "test_real_gray_baseline.csv",
        rows,
    )

    return rows


def select_cohort(
    baseline_rows,
    cohort_name,
    max_samples,
    seed,
):
    if cohort_name == "real_correct_gray_wrong":
        rows = [
            row
            for row in baseline_rows
            if int(
                row.get(
                    "ok",
                    0,
                )
            ) == 1
            and int(
                row.get(
                    "real_correct",
                    0,
                )
            ) == 1
            and int(
                row.get(
                    "gray_correct",
                    0,
                )
            ) == 0
        ]

    elif cohort_name == "all_test":
        rows = [
            row
            for row in baseline_rows
            if int(
                row.get(
                    "ok",
                    0,
                )
            ) == 1
        ]

    else:
        raise ValueError(
            cohort_name
        )

    if (
        max_samples is not None
        and len(
            rows
        ) > max_samples
    ):
        rng = random.Random(
            seed
        )

        rng.shuffle(
            rows
        )

        rows = rows[
            :max_samples
        ]

    return sorted(
        rows,
        key=lambda row: int(
            row["sid"]
        ),
    )


# =============================================================================
# TEST intervention
# =============================================================================

def run_intervention(
    model,
    processor,
    decoder_layers,
    templates,
    records,
    cohort_rows,
    pair_layers,
    last_layers,
    template_mode,
    condition_name,
    args,
    device,
):
    rows = []

    baseline_by_sid = {
        int(
            row["sid"]
        ): row
        for row in cohort_rows
    }

    for sid in tqdm(
        sorted(
            baseline_by_sid
        ),
        desc=condition_name,
    ):
        rec = records[
            sid
        ]

        base = baseline_by_sid[
            sid
        ]

        real_image = None
        gray_image = None

        try:
            real_image = Image.open(
                rec[
                    "image_path"
                ]
            ).convert(
                "RGB"
            )

            gray_image = make_gray_image(
                real_image,
                args.gray_value,
            )

            (
                gray_batch,
                subject_span,
                reference_span,
                last_position,
            ) = prepare_batch_for_image(
                processor,
                gray_image,
                rec,
                args.prompt_template,
                device,
            )

            if (
                subject_span is None
                or reference_span is None
            ):
                continue

            with AdditionHooks(
                decoder_layers=decoder_layers,
                templates=templates,
                gt_relation=rec[
                    "gt"
                ],
                pair_layers=pair_layers,
                last_layers=last_layers,
                subject_span=subject_span,
                reference_span=reference_span,
                last_position=last_position,
                scale=args.scale,
                template_mode=template_mode,
            ):

                edit_text, edit_pred = generate(
                    model,
                    processor,
                    gray_batch,
                    args.max_new_tokens,
                )

            gray_correct = int(
                base[
                    "gray_correct"
                ]
            )

            edit_correct = int(
                edit_pred == rec[
                    "gt"
                ]
            )

            rows.append({
                "condition":
                    condition_name,
                "template_mode":
                    template_mode,
                "sid":
                    sid,
                "gt":
                    rec["gt"],
                "real_pred":
                    base[
                        "real_pred"
                    ],
                "gray_pred":
                    base[
                        "gray_pred"
                    ],
                "edit_pred":
                    edit_pred or "",
                "real_correct":
                    int(
                        base[
                            "real_correct"
                        ]
                    ),
                "gray_correct":
                    gray_correct,
                "edit_correct":
                    edit_correct,
                "rescue":
                    int(
                        gray_correct == 0
                        and edit_correct == 1
                    ),
                "damage":
                    int(
                        gray_correct == 1
                        and edit_correct == 0
                    ),
                "changed":
                    int(
                        (
                            edit_pred or ""
                        )
                        != str(
                            base[
                                "gray_pred"
                            ]
                        )
                    ),
                "edit_text":
                    edit_text,
            })

            del gray_batch

        except Exception as exc:
            tqdm.write(
                f"[INTERVENTION ERROR sid={sid} {condition_name}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real_image is not None:
                real_image.close()

            if gray_image is not None:
                gray_image.close()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def summarize(
    rows,
    condition,
):
    if not rows:
        return {
            "condition":
                condition,
            "N":
                0,
            "gray_acc":
                float("nan"),
            "edit_acc":
                float("nan"),
            "gain":
                float("nan"),
            "rescue":
                0,
            "rescue_rate":
                float("nan"),
            "damage":
                0,
            "damage_rate":
                float("nan"),
            "changed_rate":
                float("nan"),
        }

    gray_wrong = sum(
        1
        - int(
            row[
                "gray_correct"
            ]
        )
        for row in rows
    )

    gray_correct = (
        len(
            rows
        )
        - gray_wrong
    )

    rescue = sum(
        int(
            row[
                "rescue"
            ]
        )
        for row in rows
    )

    damage = sum(
        int(
            row[
                "damage"
            ]
        )
        for row in rows
    )

    gray_acc = safe_mean(
        row[
            "gray_correct"
        ]
        for row in rows
    )

    edit_acc = safe_mean(
        row[
            "edit_correct"
        ]
        for row in rows
    )

    return {
        "condition":
            condition,
        "N":
            len(
                rows
            ),
        "gray_acc":
            gray_acc,
        "edit_acc":
            edit_acc,
        "gain":
            edit_acc
            - gray_acc,
        "rescue":
            rescue,
        "rescue_rate":
            (
                rescue
                / gray_wrong
                if gray_wrong
                else float(
                    "nan"
                )
            ),
        "damage":
            damage,
        "damage_rate":
            (
                damage
                / gray_correct
                if gray_correct
                else float(
                    "nan"
                )
            ),
        "changed_rate":
            safe_mean(
                row[
                    "changed"
                ]
                for row in rows
            ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
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

    controls = parse_controls(
        args.controls
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

    split_map = load_split(
        args.split_dir
    )

    records = load_records(
        args.prompt_jsonl,
        args.annotation_json,
        args.data_root,
        split_map,
    )

    model, processor = load_model_and_processor(
        args.model_id,
        dtype_from_name(
            args.dtype
        ),
        args.device,
        args.attn_impl,
    )

    decoder_layers, decoder_path = resolve_decoder_layers(
        model
    )

    pair_layers = parse_layers(
        args.pair_layers,
        len(
            decoder_layers
        ),
    )

    last_layers = parse_layers(
        args.last_layers,
        len(
            decoder_layers
        ),
    )

    if (
        not pair_layers
        and not last_layers
    ):
        raise RuntimeError(
            "No pair or last layers selected."
        )

    print(
        f"[decoder] {decoder_path}"
    )

    print(
        f"[pair layers] {pair_layers}"
    )

    print(
        f"[last layers] {last_layers}"
    )

    train_sids = sorted(
        [
            sid
            for sid, rec in records.items()
            if rec[
                "split"
            ] == "train"
        ]
    )

    test_sids = sorted(
        [
            sid
            for sid, rec in records.items()
            if rec[
                "split"
            ] == "test"
        ]
    )

    if (
        args.max_train_samples is not None
        and len(
            train_sids
        ) > args.max_train_samples
    ):
        rng = random.Random(
            args.seed
        )

        rng.shuffle(
            train_sids
        )

        train_sids = sorted(
            train_sids[
                :args.max_train_samples
            ]
        )

    if (
        args.max_test_samples is not None
        and len(
            test_sids
        ) > args.max_test_samples
    ):
        rng = random.Random(
            args.seed
            + 17
        )

        rng.shuffle(
            test_sids
        )

        test_sids = sorted(
            test_sids[
                :args.max_test_samples
            ]
        )

    print(
        f"[data] TRAIN={len(train_sids)} TEST={len(test_sids)}"
    )

    device = torch.device(
        args.device
    )

    # -------------------------------------------------------------------------
    # 1. TRAIN: Real-Gray deltas
    # -------------------------------------------------------------------------

    collected, train_rows = collect_train_deltas(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        train_sids=train_sids,
        pair_layers=pair_layers,
        last_layers=last_layers,
        args=args,
        device=device,
        outdir=outdir,
    )

    used_train = [
        row
        for row in train_rows
        if int(
            row.get(
                "used",
                0,
            )
        ) == 1
    ]

    print(
        f"[TRAIN filter={args.train_filter}] "
        f"used={len(used_train)}/{len(train_rows)}"
    )

    relation_counts = {
        relation: sum(
            1
            for row in used_train
            if row.get(
                "gt"
            ) == relation
        )
        for relation in RELS
    }

    print(
        "[TRAIN relation counts] "
        + " ".join(
            f"{relation}={relation_counts[relation]}"
            for relation in RELS
        )
    )

    templates, template_summary = fit_shared_templates(
        collected
    )

    write_csv(
        outdir
        / "shared_template_summary.csv",
        template_summary,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "SHARED REAL-GRAY RELATION TEMPLATES"
    )

    print(
        "=" * 120
    )

    for row in template_summary:
        print(
            f"{row['site']:4s} L{int(row['layer']):02d} | "
            f"N L/R/A/B="
            f"{row['n_left']}/{row['n_right']}/"
            f"{row['n_above']}/{row['n_below']} | "
            f"global={float(row['global_norm']):.3f} | "
            f"shared norms "
            f"L={float(row['shared_left_norm']):.3f} "
            f"R={float(row['shared_right_norm']):.3f} "
            f"A={float(row['shared_above_norm']):.3f} "
            f"B={float(row['shared_below_norm']):.3f}"
        )

    # -------------------------------------------------------------------------
    # 2. TEST Real/Gray baseline
    # -------------------------------------------------------------------------

    baseline_rows = run_test_baselines(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        test_sids=test_sids,
        pair_layers=pair_layers,
        last_layers=last_layers,
        args=args,
        device=device,
        outdir=outdir,
    )

    valid_baseline = [
        row
        for row in baseline_rows
        if int(
            row.get(
                "ok",
                0,
            )
        ) == 1
    ]

    real_acc = safe_mean(
        row[
            "real_correct"
        ]
        for row in valid_baseline
    )

    gray_acc = safe_mean(
        row[
            "gray_correct"
        ]
        for row in valid_baseline
    )

    cohort_rows = select_cohort(
        baseline_rows,
        args.cohort,
        args.max_cohort_samples,
        args.seed,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"TEST baseline | valid N={len(valid_baseline)} | "
        f"Real acc={real_acc:.4f} | Gray acc={gray_acc:.4f}"
    )

    print(
        f"Cohort={args.cohort} | N={len(cohort_rows)}"
    )

    print(
        "=" * 120
    )

    if not cohort_rows:
        raise RuntimeError(
            "Selected TEST cohort is empty."
        )

    # -------------------------------------------------------------------------
    # 3. Intervention conditions
    # -------------------------------------------------------------------------

    conditions = []

    if not args.no_single:
        for layer in pair_layers:
            conditions.append(
                (
                    f"pair_single_L{layer:02d}",
                    [
                        layer
                    ],
                    [],
                )
            )

        for layer in last_layers:
            conditions.append(
                (
                    f"last_single_L{layer:02d}",
                    [],
                    [
                        layer
                    ],
                )
            )

    if not args.no_multi:
        if pair_layers:
            conditions.append(
                (
                    "pair_multi",
                    pair_layers,
                    [],
                )
            )

        if last_layers:
            conditions.append(
                (
                    "last_multi",
                    [],
                    last_layers,
                )
            )

    if (
        not args.no_joint
        and pair_layers
        and last_layers
    ):
        conditions.append(
            (
                "joint_pair_plus_last_multi",
                pair_layers,
                last_layers,
            )
        )

    template_modes = [
        (
            "shared",
            "",
        )
    ]

    for control in controls:
        template_modes.append(
            (
                control,
                f"_{control}",
            )
        )

    all_detail_rows = []
    summary_rows = []

    for (
        base_condition,
        condition_pair_layers,
        condition_last_layers,
    ) in conditions:

        for (
            template_mode,
            suffix,
        ) in template_modes:

            condition_name = (
                base_condition
                + suffix
            )

            rows = run_intervention(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                templates=templates,
                records=records,
                cohort_rows=cohort_rows,
                pair_layers=condition_pair_layers,
                last_layers=condition_last_layers,
                template_mode=template_mode,
                condition_name=condition_name,
                args=args,
                device=device,
            )

            all_detail_rows.extend(
                rows
            )

            summary_rows.append(
                summarize(
                    rows,
                    condition_name,
                )
            )

            write_csv(
                outdir
                / "intervention_details.csv",
                all_detail_rows,
            )

            write_csv(
                outdir
                / "summary.csv",
                summary_rows,
            )

    # -------------------------------------------------------------------------
    # 4. Print
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 155
    )

    print(
        "GRAY + SHARED REAL-GRAY RELATION DELTA — ACTUAL model.generate()"
    )

    print(
        "=" * 155
    )

    print(
        "condition                              | N | "
        "GrayAcc -> EditAcc gain | rescue/rate | damage/rate | changed"
    )

    for row in summary_rows:
        print(
            f"{str(row['condition']):38s} | "
            f"{int(row['N']):3d} | "
            f"{float(row['gray_acc']):.4f}->"
            f"{float(row['edit_acc']):.4f} "
            f"{float(row['gain']):+.4f} | "
            f"{int(row['rescue'])}/"
            f"{float(row['rescue_rate']):.3f} | "
            f"{int(row['damage'])}/"
            f"{float(row['damage_rate']):.3f} | "
            f"{float(row['changed_rate']):.3f}"
        )

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "experiment":
                    "shared relation-specific Real-Gray delta addition",
                "old_direction_vectors_used":
                    False,
                "train_filter":
                    args.train_filter,
                "train_used":
                    len(
                        used_train
                    ),
                "train_relation_counts":
                    relation_counts,
                "template_definition": {
                    "pair_delta":
                        (
                            "(h_sub-h_ref)_real - "
                            "(h_sub-h_ref)_gray"
                        ),
                    "last_delta":
                        (
                            "h_last_real - h_last_gray"
                        ),
                    "relation_mean":
                        "mean TRAIN delta within each relation",
                    "global_mean":
                        (
                            "balanced mean of the four relation means"
                        ),
                    "shared_relation":
                        "relation_mean - global_mean",
                },
                "pair_layers":
                    pair_layers,
                "last_layers":
                    last_layers,
                "scale":
                    args.scale,
                "gray_value":
                    args.gray_value,
                "cohort":
                    args.cohort,
                "test_baseline_real_acc":
                    real_acc,
                "test_baseline_gray_acc":
                    gray_acc,
                "test_cohort_n":
                    len(
                        cohort_rows
                    ),
                "controls":
                    controls,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved:"
    )

    for path in [
        outdir
        / "train_real_gray_generation.csv",
        outdir
        / "shared_template_summary.csv",
        outdir
        / "test_real_gray_baseline.csv",
        outdir
        / "summary.csv",
        outdir
        / "intervention_details.csv",
        outdir
        / "summary.json",
    ]:
        if path.exists():
            print(
                " ",
                path,
            )


if __name__ == "__main__":
    main()
