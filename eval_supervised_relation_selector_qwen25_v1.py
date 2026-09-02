#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_supervised_relation_selector_qwen25_v1.py

Purpose
-------
Test three non-oracle selectors for the already-discovered late causal actuator
on COCO-two:

1) logits-only
2) middle-Direction-only
3) Direction + logits fusion

The selector answers:
    which late relation-specific causal vector should be injected?

Known actuator windows are reused:
    Qwen2.5-VL-3B: L32-L35
    Qwen2.5-VL-7B: L25-L27

Middle Direction features
-------------------------
For layer l:

    r_real,l  = h_real,l(subject) - h_real,l(reference)
    r_noimg,l = h_noimg,l(subject) - h_noimg,l(reference)

    q_l = r_real,l - r_noimg,l

FIT learns four relation prototypes after subtracting the FIT center:

    c_l = mean_i q_i,l
    mu_r,l = mean(q_i,l - c_l | relation=r)

For each sample, the Direction feature is the cosine score to every relation
prototype at every selected middle layer.

Default middle layers:
    3B: L18-L20
    7B: L14-L20

Logit features
--------------
From the REAL-image first-step logits at the final prompt token, extract scores
for left/right/above/below.

Classifier
----------
A small L2-regularized multinomial logistic regression is implemented in pure
PyTorch (no sklearn dependency). C is selected on CAL.

Data discipline
---------------
The previous experiment's TEST ids are reused via --prior-output-dir.

Original TRAIN:
    FIT: train selector / Direction prototypes / CAL actuator
    CAL: choose C and steering policy

TEST:
    untouched until the final evaluation.

Actuator
--------
Late causal templates use TRAIN Real-Gray last-token deltas:

    delta_i,l = h_real,last,l - h_gray,last,l
    s_r,l = mu_r,l - balanced_global_l

For CAL, actuator templates are fit on FIT.
For final TEST, actuator templates are refit on the full original TRAIN.

Policy calibration
------------------
For each selector, CAL searches:
    edit mode: add / contrast
    apply mode: all / conflict_only
    confidence coverage: 1.00 / 0.75 / 0.50 / 0.25

We generate each selector/edit-mode CAL intervention only once, then evaluate
all confidence/apply policies offline.

Outputs
-------
selector_calibration.csv
cal_selector_predictions.csv
cal_policy_search.csv
test_selector_predictions.csv
test_summary.csv
test_details.csv
feature_cache.npz
actuator_train_delta_cache.npz
summary.json
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


RELS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELS)}
RELSET = set(RELS)
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
        "--prior-output-dir",
        required=True,
        help=(
            "Earlier COCO causal-steering output. TEST ids/baseline and, when "
            "available, TRAIN Real/Gray correctness are reused from here."
        ),
    )

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument(
        "--annotation-json",
        default="data/coco_qa_two_obj.json",
    )

    p.add_argument(
        "--data-root",
        default="data",
    )

    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument(
        "--direction-layers",
        default="auto",
        help="auto: 3B=L18-L20, 7B=L14-L20",
    )

    p.add_argument(
        "--actuator-layers",
        default="auto",
        help="auto: 3B=L32-L35, 7B=L25-L27",
    )

    p.add_argument(
        "--cal-frac",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--c-grid",
        default="0.03,0.1,0.3,1,3,10",
        help="L2 softmax-regression C values; larger means weaker regularization.",
    )

    p.add_argument(
        "--coverages",
        default="1.0,0.75,0.5,0.25",
    )

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
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=8)

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
        "--feature-cache",
        default="",
    )

    p.add_argument(
        "--actuator-cache",
        default="",
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


def model_preset(model_id):
    low = model_id.lower()

    if "3b" in low:
        return {
            "tag": "qwen3b",
            "direction_layers": [18, 19, 20],
            "actuator_layers": [32, 33, 34, 35],
        }

    if "7b" in low:
        return {
            "tag": "qwen7b",
            "direction_layers": list(range(14, 21)),
            "actuator_layers": [25, 26, 27],
        }

    raise ValueError(model_id)


# =============================================================================
# Basic utilities
# =============================================================================

def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(values):
    xs = []

    for value in values:
        try:
            value = float(value)
        except Exception:
            continue

        if math.isfinite(value):
            xs.append(value)

    return float(np.mean(xs)) if xs else float("nan")


def parse_int_list(spec):
    return [
        int(x.strip())
        for x in str(spec).split(",")
        if x.strip()
    ]


def parse_float_list(spec):
    return [
        float(x.strip())
        for x in str(spec).split(",")
        if x.strip()
    ]


def parse_layer_spec(spec):
    out = []

    for part in str(spec).split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            a, b = map(
                int,
                part.split("-", 1),
            )

            step = 1 if b >= a else -1

            out.extend(
                range(
                    a,
                    b + step,
                    step,
                )
            )

        else:
            out.append(
                int(part)
            )

    return sorted(
        set(out)
    )


def normalize_relation(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    text = str(value).strip().lower()

    hits = []

    patterns = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]

    for relation, pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.I,
        )

        if match:
            hits.append(
                (
                    match.start(),
                    relation,
                )
            )

    if hits:
        return sorted(
            hits
        )[0][1]

    return text


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
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

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

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


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


# =============================================================================
# Reuse previous TEST split / generation labels
# =============================================================================

def find_existing_file(root, names):
    root = Path(root)

    for name in names:
        candidate = root / name

        if candidate.exists():
            return candidate

    return None


def row_sid(row):
    for key in (
        "sid",
        "sample_index",
        "id",
    ):
        if (
            key in row
            and str(row[key]).strip()
        ):
            return int(
                row[key]
            )

    raise KeyError(
        f"Cannot infer sid from columns={list(row)}"
    )


def row_pred(row):
    for key in (
        "pred",
        "baseline_pred",
        "generation_pred",
        "real_pred",
    ):
        if key in row:
            relation = normalize_relation(
                row[key]
            )

            if relation in RELSET:
                return relation

    return None


def load_existing_test_baseline(prior_dir):
    path = find_existing_file(
        prior_dir,
        [
            "test_baseline.csv",
            "baseline.csv",
        ],
    )

    if path is None:
        return None, None

    result = {}

    for row in read_csv(path):
        sid = row_sid(
            row
        )

        pred = row_pred(
            row
        )

        correct = None

        for key in (
            "correct",
            "baseline_correct",
        ):
            if (
                key in row
                and str(row[key]).strip()
            ):
                try:
                    correct = int(
                        float(
                            row[key]
                        )
                    )
                except Exception:
                    pass

        result[sid] = {
            "sid": sid,
            "pred": pred,
            "correct": correct,
        }

    print(
        f"[reuse] TEST baseline={path} N={len(result)}"
    )

    return result, path


def load_existing_train_generation(prior_dir):
    path = find_existing_file(
        prior_dir,
        [
            "full_train_real_gray_generation.csv",
            "train_real_gray_generation.csv",
            "train_real_gray_generation_for_refit.csv",
            "fit_real_gray_generation.csv",
        ],
    )

    if path is None:
        print(
            "[reuse] no TRAIN Real/Gray correctness CSV; "
            "actuator template filter will effectively use all available samples."
        )
        return None, None

    result = {}

    for row in read_csv(path):
        try:
            sid = row_sid(
                row
            )
        except Exception:
            continue

        real_correct = None
        gray_correct = None

        for key in (
            "real_correct",
            "real_is_correct",
        ):
            if (
                key in row
                and str(row[key]).strip()
            ):
                try:
                    real_correct = int(
                        float(
                            row[key]
                        )
                    )
                except Exception:
                    pass

        for key in (
            "gray_correct",
            "gray_is_correct",
        ):
            if (
                key in row
                and str(row[key]).strip()
            ):
                try:
                    gray_correct = int(
                        float(
                            row[key]
                        )
                    )
                except Exception:
                    pass

        result[sid] = {
            "real_correct": real_correct,
            "gray_correct": gray_correct,
        }

    print(
        f"[reuse] TRAIN Real/Gray labels={path} N={len(result)}"
    )

    return result, path


# =============================================================================
# COCO-two
# =============================================================================

OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+"
    r"in\s+relation\s+to\s+(?:the\s+)?(.+?)\?"
    r"\s*Answer",
    flags=re.I | re.S,
)


def parse_subject_reference(question):
    compact = re.sub(
        r"\s+",
        " ",
        str(question),
    ).strip()

    match = OBJECT_RE.search(
        compact
    )

    if match:
        return (
            match.group(1).strip(),
            match.group(2).strip(),
        )

    raise ValueError(
        f"Could not parse object pair from: {compact!r}"
    )


def load_records(args):
    with Path(args.annotation_json).open(
        "r",
        encoding="utf-8",
    ) as handle:
        annotations = json.load(handle)

    prompts = []

    with Path(args.prompt_jsonl).open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
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

        if (
            sid < 0
            or sid >= len(annotations)
        ):
            continue

        relation = normalize_relation(
            row.get(
                "answer",
                "",
            )
        )

        if relation not in RELSET:
            continue

        subject, reference = parse_subject_reference(
            row.get(
                "question",
                "",
            )
        )

        image_id = int(
            annotations[sid][0]
        )

        image_path = (
            Path(args.data_root)
            / "val2017"
            / f"{image_id:012d}.jpg"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        records[sid] = {
            "sid": sid,
            "gt": relation,
            "subject": subject,
            "reference": reference,
            "image_path": str(image_path),
        }

    if not records:
        raise RuntimeError(
            "No COCO-two samples loaded."
        )

    print(
        f"[COCO-two] N={len(records)} "
        f"{dict(Counter(r['gt'] for r in records.values()))}"
    )

    return records


def derive_train_test_ids(
    records,
    existing_test,
):
    if existing_test is None:
        raise RuntimeError(
            "No prior TEST baseline was found. This experiment intentionally "
            "reuses the previous TEST split; point --prior-output-dir to the "
            "earlier Qwen causal run."
        )

    test_sids = sorted(
        sid
        for sid in existing_test
        if sid in records
    )

    test_set = set(
        test_sids
    )

    train_sids = sorted(
        sid
        for sid in records
        if sid not in test_set
    )

    print(
        f"[split reuse] TRAIN={len(train_sids)} TEST={len(test_sids)}"
    )

    return (
        train_sids,
        test_sids,
    )


def stratified_fit_cal_split(
    records,
    train_sids,
    cal_frac,
    seed,
):
    rng = random.Random(
        seed
    )

    fit = []
    cal = []

    for relation in RELS:
        ids = [
            sid
            for sid in train_sids
            if records[sid]["gt"] == relation
        ]

        ids = sorted(
            ids
        )

        rng.shuffle(
            ids
        )

        if len(ids) < 2:
            raise RuntimeError(
                f"Too few TRAIN {relation} samples: {len(ids)}"
            )

        n_cal = int(
            round(
                len(ids)
                * cal_frac
            )
        )

        n_cal = max(
            1,
            min(
                n_cal,
                len(ids) - 1,
            ),
        )

        cal.extend(
            ids[:n_cal]
        )

        fit.extend(
            ids[n_cal:]
        )

    fit = sorted(
        fit
    )

    cal = sorted(
        cal
    )

    print(
        f"[FIT/CAL] FIT={len(fit)} "
        f"{dict(Counter(records[s]['gt'] for s in fit))}"
    )

    print(
        f"[FIT/CAL] CAL={len(cal)} "
        f"{dict(Counter(records[s]['gt'] for s in cal))}"
    )

    return (
        fit,
        cal,
    )


# =============================================================================
# Model / prompt / hidden-state extraction
# =============================================================================

def auto_dtype(args):
    if args.dtype == "auto":
        return torch.bfloat16

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[
        args.dtype
    ]


def load_model(args):
    if not hasattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    ):
        raise RuntimeError(
            "Transformers has no Qwen2_5_VLForConditionalGeneration."
        )

    cls = (
        transformers
        .Qwen2_5_VLForConditionalGeneration
    )

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

    dtype = auto_dtype(
        args
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

    return (
        model,
        processor,
    )


def resolve_layers(model):
    for path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ):
        current = model

        try:
            for part in path.split("."):
                current = getattr(
                    current,
                    part,
                )

            if len(current):
                return (
                    current,
                    path,
                )

        except Exception:
            pass

    raise RuntimeError(
        "Could not locate Qwen decoder blocks."
    )


def rendered_prompt(
    processor,
    question,
    with_image,
):
    content = []

    if with_image:
        content.append({
            "type": "image",
        })

    content.append({
        "type": "text",
        "text": question,
    })

    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": content,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_question(
    record,
    args,
):
    return args.prompt_template.format(
        subject=record["subject"],
        reference=record["reference"],
    )


def process_inputs(
    processor,
    rendered,
    image,
    args,
):
    if image is None:
        batch = processor(
            text=[rendered],
            padding=True,
            return_tensors="pt",
        )

    else:
        batch = processor(
            text=[rendered],
            images=[image],
            padding=True,
            return_tensors="pt",
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


def hidden_tuple(outputs):
    candidates = [
        getattr(
            outputs,
            "hidden_states",
            None,
        ),
        getattr(
            getattr(
                outputs,
                "language_model_output",
                None,
            ),
            "hidden_states",
            None,
        ),
        getattr(
            getattr(
                outputs,
                "language_model_outputs",
                None,
            ),
            "hidden_states",
            None,
        ),
        getattr(
            getattr(
                outputs,
                "text_model_output",
                None,
            ),
            "hidden_states",
            None,
        ),
    ]

    for states in candidates:
        if (
            isinstance(
                states,
                (tuple, list),
            )
            and states
            and torch.is_tensor(
                states[-1]
            )
        ):
            return tuple(
                states
            )

    raise RuntimeError(
        "No decoder hidden_states returned."
    )


def find_subsequence_last(
    haystack,
    needle,
):
    best = None
    n = len(
        needle
    )

    for i in range(
        len(haystack)
        - n
        + 1
    ):
        if (
            list(
                haystack[
                    i:i+n
                ]
            )
            == list(
                needle
            )
        ):
            best = (
                i,
                i + n,
            )

    return best


def locate_phrase_positions(
    tokenizer,
    input_ids,
    phrase,
):
    hits = []

    for text in (
        str(phrase),
        " " + str(phrase),
    ):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if ids:
            hit = find_subsequence_last(
                input_ids,
                ids,
            )

            if hit is not None:
                hits.append(
                    hit
                )

    if not hits:
        raise RuntimeError(
            f"Could not locate phrase tokens for {phrase!r}"
        )

    start, end = max(
        hits,
        key=lambda x:
            x[0],
    )

    return list(
        range(
            start,
            end,
        )
    )


def pool_positions(
    hidden,
    positions,
):
    valid = [
        int(pos)
        for pos in positions
        if (
            0
            <= int(pos)
            < int(
                hidden.shape[
                    1
                ]
            )
        )
    ]

    if not valid:
        raise RuntimeError(
            "No valid object-token positions."
        )

    indices = torch.as_tensor(
        valid,
        device=hidden.device,
        dtype=torch.long,
    )

    return (
        hidden[
            0
        ]
        .index_select(
            0,
            indices,
        )
        .mean(
            dim=0
        )
    )


def answer_token_candidates(
    tokenizer,
):
    result = {}

    for relation in RELS:
        ids = []

        for text in (
            relation,
            " " + relation,
            "\n" + relation,
        ):
            encoded = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            if encoded:
                ids.append(
                    int(
                        encoded[0]
                    )
                )

        ids = sorted(
            set(ids)
        )

        if not ids:
            raise RuntimeError(
                f"No answer token ids for {relation}"
            )

        result[
            relation
        ] = ids

    print(
        "[answer token candidates]"
    )

    for relation in RELS:
        decoded = [
            tokenizer.decode(
                [token_id]
            )
            for token_id in result[
                relation
            ]
        ]

        print(
            f"  {relation}: ids={result[relation]} tokens={decoded}"
        )

    return result


def answer_logit_features(
    logits_last,
    token_candidates,
):
    scores = []

    for relation in RELS:
        ids = torch.as_tensor(
            token_candidates[
                relation
            ],
            device=logits_last.device,
            dtype=torch.long,
        )

        values = logits_last.index_select(
            0,
            ids,
        )

        scores.append(
            float(
                values.max().item()
            )
        )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    # Remove generic next-token bias; only relative relation evidence matters.
    scores = (
        scores
        - scores.mean()
    ).astype(
        np.float32
    )

    return scores


def capture_real(
    model,
    processor,
    record,
    direction_layers,
    actuator_layers,
    token_candidates,
    args,
):
    image = Image.open(
        record[
            "image_path"
        ]
    ).convert(
        "RGB"
    )

    try:
        question = build_question(
            record,
            args,
        )

        rendered = rendered_prompt(
            processor,
            question,
            with_image=True,
        )

        batch = process_inputs(
            processor,
            rendered,
            image,
            args,
        )

        input_ids = [
            int(x)
            for x in (
                batch[
                    "input_ids"
                ][0]
                .detach()
                .cpu()
                .tolist()
            )
        ]

        subject_pos = locate_phrase_positions(
            processor.tokenizer,
            input_ids,
            record[
                "subject"
            ],
        )

        reference_pos = locate_phrase_positions(
            processor.tokenizer,
            input_ids,
            record[
                "reference"
            ],
        )

        with torch.inference_mode():
            outputs = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        states = hidden_tuple(
            outputs
        )

        n_blocks = (
            len(
                states
            )
            - 1
        )

        relation_vectors = {}

        for layer in direction_layers:
            hidden = states[
                layer
                + 1
            ]

            subject = pool_positions(
                hidden,
                subject_pos,
            )

            reference = pool_positions(
                hidden,
                reference_pos,
            )

            relation_vectors[
                layer
            ] = (
                subject
                - reference
            ).detach().float().cpu().numpy().astype(
                np.float32
            )

        last_states = {
            layer: (
                states[
                    layer
                    + 1
                ][
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
            for layer in actuator_layers
        }

        if not hasattr(
            outputs,
            "logits",
        ):
            raise RuntimeError(
                "Model output has no logits."
            )

        logits_last = outputs.logits[
            0,
            -1,
            :
        ]

        answer_logits = answer_logit_features(
            logits_last,
            token_candidates,
        )

        del outputs
        del states
        del batch

        return (
            relation_vectors,
            last_states,
            answer_logits,
            n_blocks,
        )

    finally:
        image.close()


def capture_noimage_direction(
    model,
    processor,
    record,
    direction_layers,
    args,
):
    question = build_question(
        record,
        args,
    )

    rendered = rendered_prompt(
        processor,
        question,
        with_image=False,
    )

    batch = process_inputs(
        processor,
        rendered,
        None,
        args,
    )

    input_ids = [
        int(x)
        for x in (
            batch[
                "input_ids"
            ][0]
            .detach()
            .cpu()
            .tolist()
        )
    ]

    subject_pos = locate_phrase_positions(
        processor.tokenizer,
        input_ids,
        record[
            "subject"
        ],
    )

    reference_pos = locate_phrase_positions(
        processor.tokenizer,
        input_ids,
        record[
            "reference"
        ],
    )

    with torch.inference_mode():
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    states = hidden_tuple(
        outputs
    )

    relation_vectors = {}

    for layer in direction_layers:
        hidden = states[
            layer
            + 1
        ]

        subject = pool_positions(
            hidden,
            subject_pos,
        )

        reference = pool_positions(
            hidden,
            reference_pos,
        )

        relation_vectors[
            layer
        ] = (
            subject
            - reference
        ).detach().float().cpu().numpy().astype(
            np.float32
        )

    del outputs
    del states
    del batch

    return relation_vectors


def capture_gray_last(
    model,
    processor,
    record,
    actuator_layers,
    args,
):
    real = Image.open(
        record[
            "image_path"
        ]
    ).convert(
        "RGB"
    )

    gray = Image.new(
        "RGB",
        real.size,
        (
            args.gray_value,
            args.gray_value,
            args.gray_value,
        ),
    )

    real.close()

    try:
        question = build_question(
            record,
            args,
        )

        rendered = rendered_prompt(
            processor,
            question,
            with_image=True,
        )

        batch = process_inputs(
            processor,
            rendered,
            gray,
            args,
        )

        with torch.inference_mode():
            outputs = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        states = hidden_tuple(
            outputs
        )

        last_states = {
            layer: (
                states[
                    layer
                    + 1
                ][
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
            for layer in actuator_layers
        }

        del outputs
        del states
        del batch

        return last_states

    finally:
        gray.close()


# =============================================================================
# Feature / actuator caches
# =============================================================================

def build_or_load_caches(
    model,
    processor,
    records,
    train_sids,
    direction_layers,
    actuator_layers,
    token_candidates,
    feature_cache_path,
    actuator_cache_path,
    args,
):
    feature_cache_path = Path(
        feature_cache_path
    )

    actuator_cache_path = Path(
        actuator_cache_path
    )

    feature_ready = (
        feature_cache_path.exists()
        and not args.overwrite
    )

    actuator_ready = (
        actuator_cache_path.exists()
        and not args.overwrite
    )

    if feature_ready:
        with np.load(
            feature_cache_path,
            allow_pickle=False,
        ) as data:
            cached_direction_layers = (
                np.asarray(
                    data[
                        "direction_layers"
                    ],
                    dtype=np.int64,
                )
                .tolist()
            )

            if (
                cached_direction_layers
                != direction_layers
            ):
                raise RuntimeError(
                    f"Feature cache direction layers={cached_direction_layers} "
                    f"!= requested={direction_layers}"
                )

            feature_sids = np.asarray(
                data[
                    "sample_index"
                ],
                dtype=np.int64,
            )

            direction_residual = np.asarray(
                data[
                    "direction_residual"
                ],
                dtype=np.float32,
            )

            answer_logits = np.asarray(
                data[
                    "answer_logits"
                ],
                dtype=np.float32,
            )

        feature_map = {
            int(sid): {
                "direction_residual": direction_residual[
                    i
                ],
                "answer_logits": answer_logits[
                    i
                ],
            }
            for i, sid in enumerate(
                feature_sids.tolist()
            )
        }

        print(
            f"[reuse] selector feature cache={feature_cache_path} "
            f"direction_shape={direction_residual.shape}"
        )

    else:
        feature_map = {}

    if actuator_ready:
        with np.load(
            actuator_cache_path,
            allow_pickle=False,
        ) as data:
            cached_actuator_layers = (
                np.asarray(
                    data[
                        "actuator_layers"
                    ],
                    dtype=np.int64,
                )
                .tolist()
            )

            if (
                cached_actuator_layers
                != actuator_layers
            ):
                raise RuntimeError(
                    f"Actuator cache layers={cached_actuator_layers} "
                    f"!= requested={actuator_layers}"
                )

            actuator_sids = np.asarray(
                data[
                    "sample_index"
                ],
                dtype=np.int64,
            )

            actuator_delta = np.asarray(
                data[
                    "delta"
                ],
                dtype=np.float32,
            )

        actuator_map = {
            int(sid): actuator_delta[
                i
            ]
            for i, sid in enumerate(
                actuator_sids.tolist()
            )
        }

        print(
            f"[reuse] actuator cache={actuator_cache_path} "
            f"shape={actuator_delta.shape}"
        )

    else:
        actuator_map = {}

    needed_feature_sids = sorted(
        set(records)
        - set(feature_map)
    )

    needed_actuator_sids = sorted(
        set(train_sids)
        - set(actuator_map)
    )

    if (
        not needed_feature_sids
        and not needed_actuator_sids
    ):
        return (
            feature_map,
            actuator_map,
        )

    real_last_temp = {}

    # One REAL forward provides:
    # - middle object relation states
    # - answer logits
    # - late last-token states for TRAIN actuator deltas
    process_sids = sorted(
        set(
            needed_feature_sids
        )
        | set(
            needed_actuator_sids
        )
    )

    for sid in tqdm(
        process_sids,
        desc="REAL selector/actuator features",
    ):
        record = records[
            sid
        ]

        (
            real_direction,
            real_last,
            logits4,
            n_blocks,
        ) = capture_real(
            model,
            processor,
            record,
            direction_layers,
            actuator_layers,
            token_candidates,
            args,
        )

        if max(
            direction_layers
            + actuator_layers
        ) >= n_blocks:
            raise RuntimeError(
                f"Model has {n_blocks} blocks; requested "
                f"direction={direction_layers}, actuator={actuator_layers}"
            )

        if sid in needed_feature_sids:
            noimg_direction = capture_noimage_direction(
                model,
                processor,
                record,
                direction_layers,
                args,
            )

            residual = np.stack(
                [
                    (
                        real_direction[
                            layer
                        ]
                        - noimg_direction[
                            layer
                        ]
                    ).astype(
                        np.float32
                    )
                    for layer in direction_layers
                ],
                axis=0,
            )

            feature_map[
                sid
            ] = {
                "direction_residual": residual,
                "answer_logits": logits4,
            }

        if sid in needed_actuator_sids:
            real_last_temp[
                sid
            ] = real_last

        cleanup()

    for sid in tqdm(
        needed_actuator_sids,
        desc="GRAY actuator features",
    ):
        gray_last = capture_gray_last(
            model,
            processor,
            records[
                sid
            ],
            actuator_layers,
            args,
        )

        actuator_map[
            sid
        ] = np.stack(
            [
                (
                    real_last_temp[
                        sid
                    ][
                        layer
                    ]
                    - gray_last[
                        layer
                    ]
                ).astype(
                    np.float32
                )
                for layer in actuator_layers
            ],
            axis=0,
        )

        cleanup()

    # Save full selector feature cache.
    all_feature_sids = np.asarray(
        sorted(
            feature_map
        ),
        dtype=np.int64,
    )

    direction_array = np.stack(
        [
            feature_map[
                int(sid)
            ][
                "direction_residual"
            ]
            for sid in all_feature_sids
        ],
        axis=0,
    ).astype(
        np.float32
    )

    logits_array = np.stack(
        [
            feature_map[
                int(sid)
            ][
                "answer_logits"
            ]
            for sid in all_feature_sids
        ],
        axis=0,
    ).astype(
        np.float32
    )

    feature_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        feature_cache_path,
        sample_index=all_feature_sids,
        direction_layers=np.asarray(
            direction_layers,
            dtype=np.int64,
        ),
        direction_residual=direction_array,
        answer_logits=logits_array,
    )

    print(
        f"[saved] feature cache={feature_cache_path} "
        f"direction={direction_array.shape} logits={logits_array.shape}"
    )

    # Save TRAIN-only actuator cache.
    all_actuator_sids = np.asarray(
        sorted(
            actuator_map
        ),
        dtype=np.int64,
    )

    actuator_array = np.stack(
        [
            actuator_map[
                int(sid)
            ]
            for sid in all_actuator_sids
        ],
        axis=0,
    ).astype(
        np.float32
    )

    actuator_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        actuator_cache_path,
        sample_index=all_actuator_sids,
        actuator_layers=np.asarray(
            actuator_layers,
            dtype=np.int64,
        ),
        delta=actuator_array,
    )

    print(
        f"[saved] actuator cache={actuator_cache_path} "
        f"shape={actuator_array.shape}"
    )

    return (
        feature_map,
        actuator_map,
    )


# =============================================================================
# Direction prototypes / selector matrices
# =============================================================================

def fit_direction_prototypes(
    feature_map,
    records,
    fit_sids,
    direction_layers,
):
    arrays = np.stack(
        [
            feature_map[
                sid
            ][
                "direction_residual"
            ]
            for sid in fit_sids
        ],
        axis=0,
    ).astype(
        np.float64
    )

    centers = arrays.mean(
        axis=0
    )

    prototypes = np.zeros(
        (
            len(
                direction_layers
            ),
            len(
                RELS
            ),
            arrays.shape[
                -1
            ],
        ),
        dtype=np.float64,
    )

    for relation_index, relation in enumerate(
        RELS
    ):
        indices = [
            i
            for i, sid in enumerate(
                fit_sids
            )
            if records[
                sid
            ][
                "gt"
            ]
            == relation
        ]

        if not indices:
            raise RuntimeError(
                f"FIT has no {relation} samples."
            )

        class_centered = (
            arrays[
                indices
            ]
            - centers[
                None,
                :,
                :
            ]
        )

        mean = class_centered.mean(
            axis=0
        )

        norm = np.linalg.norm(
            mean,
            axis=-1,
            keepdims=True,
        )

        prototypes[
            :,
            relation_index,
            :,
        ] = (
            mean
            / np.maximum(
                norm,
                EPS,
            )
        )

    return {
        "centers": centers.astype(
            np.float32
        ),
        "prototypes": prototypes.astype(
            np.float32
        ),
    }


def direction_score_vector(
    residual,
    direction_model,
):
    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    centers = np.asarray(
        direction_model[
            "centers"
        ],
        dtype=np.float64,
    )

    prototypes = np.asarray(
        direction_model[
            "prototypes"
        ],
        dtype=np.float64,
    )

    q = (
        residual
        - centers
    )

    qnorm = np.linalg.norm(
        q,
        axis=-1,
        keepdims=True,
    )

    q = (
        q
        / np.maximum(
            qnorm,
            EPS,
        )
    )

    # [layers, relation]
    scores = np.einsum(
        "ld,lrd->lr",
        q,
        prototypes,
    )

    return scores.astype(
        np.float32
    ).reshape(
        -1
    )


def selector_feature(
    selector_name,
    sid,
    feature_map,
    direction_model,
):
    logits = feature_map[
        sid
    ][
        "answer_logits"
    ].astype(
        np.float32
    )

    direction = direction_score_vector(
        feature_map[
            sid
        ][
            "direction_residual"
        ],
        direction_model,
    )

    if selector_name == "logits":
        return logits

    if selector_name == "direction":
        return direction

    if selector_name == "fusion":
        return np.concatenate(
            [
                direction,
                logits,
            ],
            axis=0,
        ).astype(
            np.float32
        )

    raise ValueError(
        selector_name
    )


def build_xy(
    selector_name,
    sids,
    feature_map,
    direction_model,
    records,
):
    x = np.stack(
        [
            selector_feature(
                selector_name,
                sid,
                feature_map,
                direction_model,
            )
            for sid in sids
        ],
        axis=0,
    ).astype(
        np.float64
    )

    y = np.asarray(
        [
            REL2ID[
                records[
                    sid
                ][
                    "gt"
                ]
            ]
            for sid in sids
        ],
        dtype=np.int64,
    )

    return (
        x,
        y,
    )


# =============================================================================
# Pure-PyTorch standardized multinomial logistic regression
# =============================================================================

class SoftmaxRegression:
    def __init__(
        self,
        c_value,
        max_iter=300,
    ):
        self.c_value = float(
            c_value
        )

        self.max_iter = int(
            max_iter
        )

        self.mean = None
        self.std = None
        self.weight = None
        self.bias = None

    def fit(
        self,
        x,
        y,
    ):
        x = np.asarray(
            x,
            dtype=np.float64,
        )

        y = np.asarray(
            y,
            dtype=np.int64,
        )

        self.mean = x.mean(
            axis=0
        )

        self.std = x.std(
            axis=0
        )

        self.std[
            self.std
            < 1e-8
        ] = 1.0

        z = (
            x
            - self.mean[
                None,
                :
            ]
        ) / self.std[
            None,
            :
        ]

        xt = torch.as_tensor(
            z,
            dtype=torch.float64,
            device="cpu",
        )

        yt = torch.as_tensor(
            y,
            dtype=torch.long,
            device="cpu",
        )

        weight = torch.zeros(
            (
                len(
                    RELS
                ),
                z.shape[
                    1
                ],
            ),
            dtype=torch.float64,
            requires_grad=True,
        )

        bias = torch.zeros(
            (
                len(
                    RELS
                ),
            ),
            dtype=torch.float64,
            requires_grad=True,
        )

        optimizer = torch.optim.LBFGS(
            [
                weight,
                bias,
            ],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        # Rough sklearn-like interpretation: larger C -> weaker L2.
        reg = (
            1.0
            / max(
                self.c_value,
                1e-8,
            )
        )

        def closure():
            optimizer.zero_grad()

            logits = (
                xt
                @ weight.t()
                + bias[
                    None,
                    :
                ]
            )

            loss = F.cross_entropy(
                logits,
                yt,
            )

            loss = (
                loss
                + 0.5
                * reg
                * weight.pow(
                    2
                ).mean()
            )

            loss.backward()

            return loss

        optimizer.step(
            closure
        )

        self.weight = (
            weight.detach().cpu().numpy().astype(
                np.float64
            )
        )

        self.bias = (
            bias.detach().cpu().numpy().astype(
                np.float64
            )
        )

        return self

    def predict_proba(
        self,
        x,
    ):
        x = np.asarray(
            x,
            dtype=np.float64,
        )

        z = (
            x
            - self.mean[
                None,
                :
            ]
        ) / self.std[
            None,
            :
        ]

        logits = (
            z
            @ self.weight.T
            + self.bias[
                None,
                :
            ]
        )

        logits = (
            logits
            - logits.max(
                axis=1,
                keepdims=True,
            )
        )

        exp = np.exp(
            logits
        )

        return (
            exp
            / np.maximum(
                exp.sum(
                    axis=1,
                    keepdims=True,
                ),
                EPS,
            )
        )

    def predict(
        self,
        x,
    ):
        return self.predict_proba(
            x
        ).argmax(
            axis=1
        )


def nll_score(
    probabilities,
    y,
):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    selected = probabilities[
        np.arange(
            len(y)
        ),
        y,
    ]

    return float(
        -np.log(
            np.maximum(
                selected,
                1e-12,
            )
        ).mean()
    )


def prediction_rows(
    selector_name,
    sids,
    x,
    model,
    records,
):
    probabilities = model.predict_proba(
        x
    )

    rows = []

    for index, sid in enumerate(
        sids
    ):
        probs = probabilities[
            index
        ]

        order = np.argsort(
            -probs
        )

        top1 = int(
            order[0]
        )

        top2 = int(
            order[1]
        )

        pred = RELS[
            top1
        ]

        confidence = float(
            probs[
                top1
            ]
            - probs[
                top2
            ]
        )

        row = {
            "selector": selector_name,
            "sid": sid,
            "gt": records[sid]["gt"],
            "pred": pred,
            "correct": int(
                pred
                == records[
                    sid
                ][
                    "gt"
                ]
            ),
            "confidence": confidence,
            "p_left": float(
                probs[
                    REL2ID[
                        "left"
                    ]
                ]
            ),
            "p_right": float(
                probs[
                    REL2ID[
                        "right"
                    ]
                ]
            ),
            "p_above": float(
                probs[
                    REL2ID[
                        "above"
                    ]
                ]
            ),
            "p_below": float(
                probs[
                    REL2ID[
                        "below"
                    ]
                ]
            ),
        }

        rows.append(
            row
        )

    return rows


def train_selectors(
    fit_sids,
    cal_sids,
    feature_map,
    direction_model,
    records,
    c_grid,
):
    selector_models = {}
    calibration_rows = []
    cal_prediction_map = {}

    for selector_name in (
        "logits",
        "direction",
        "fusion",
    ):
        x_fit, y_fit = build_xy(
            selector_name,
            fit_sids,
            feature_map,
            direction_model,
            records,
        )

        x_cal, y_cal = build_xy(
            selector_name,
            cal_sids,
            feature_map,
            direction_model,
            records,
        )

        candidates = []

        for c_value in c_grid:
            model = SoftmaxRegression(
                c_value
            ).fit(
                x_fit,
                y_fit,
            )

            probabilities = model.predict_proba(
                x_cal
            )

            pred = probabilities.argmax(
                axis=1
            )

            accuracy = float(
                np.mean(
                    pred
                    == y_cal
                )
            )

            nll = nll_score(
                probabilities,
                y_cal,
            )

            row = {
                "selector": selector_name,
                "C": c_value,
                "feature_dim": int(
                    x_fit.shape[
                        1
                    ]
                ),
                "FIT_N": len(
                    fit_sids
                ),
                "CAL_N": len(
                    cal_sids
                ),
                "CAL_acc": accuracy,
                "CAL_nll": nll,
            }

            calibration_rows.append(
                row
            )

            candidates.append(
                (
                    accuracy,
                    -nll,
                    -abs(
                        math.log10(
                            c_value
                        )
                    ),
                    c_value,
                    model,
                )
            )

        best = max(
            candidates,
            key=lambda item:
                item[:3],
        )

        best_c = best[
            3
        ]

        best_model = best[
            4
        ]

        selector_models[
            selector_name
        ] = {
            "C": best_c,
            "model": best_model,
        }

        cal_rows = prediction_rows(
            selector_name,
            cal_sids,
            x_cal,
            best_model,
            records,
        )

        cal_prediction_map[
            selector_name
        ] = {
            int(row["sid"]): row
            for row in cal_rows
        }

        print(
            f"[selector CAL] {selector_name:9s} "
            f"C={best_c:g} "
            f"acc={safe_mean(r['correct'] for r in cal_rows):.4f} "
            f"mean_conf={safe_mean(r['confidence'] for r in cal_rows):.4f}"
        )

    return (
        selector_models,
        calibration_rows,
        cal_prediction_map,
    )


# =============================================================================
# Actuator templates
# =============================================================================

def sample_allowed(
    sid,
    train_generation,
    mode,
):
    if mode == "all":
        return True

    if (
        train_generation is None
        or sid not in train_generation
    ):
        return False

    real_correct = train_generation[
        sid
    ].get(
        "real_correct"
    )

    gray_correct = train_generation[
        sid
    ].get(
        "gray_correct"
    )

    if mode == "real_correct":
        return (
            real_correct is not None
            and bool(
                real_correct
            )
        )

    if mode == "real_correct_gray_wrong":
        return (
            real_correct is not None
            and gray_correct is not None
            and bool(
                real_correct
            )
            and not bool(
                gray_correct
            )
        )

    return True


def fit_actuator_templates(
    actuator_map,
    records,
    sids,
    actuator_layers,
    train_generation,
    requested_filter,
):
    filter_sequence = {
        "real_correct_gray_wrong": [
            "real_correct_gray_wrong",
            "real_correct",
            "all",
        ],
        "real_correct": [
            "real_correct",
            "all",
        ],
        "all": [
            "all",
        ],
    }[
        requested_filter
    ]

    for mode in filter_sequence:
        bags = {
            relation: []
            for relation in RELS
        }

        used = []

        for sid in sids:
            if sid not in actuator_map:
                continue

            if not sample_allowed(
                sid,
                train_generation,
                mode,
            ):
                continue

            relation = records[
                sid
            ][
                "gt"
            ]

            bags[
                relation
            ].append(
                actuator_map[
                    sid
                ]
            )

            used.append(
                sid
            )

        missing = [
            relation
            for relation in RELS
            if not bags[
                relation
            ]
        ]

        if missing:
            print(
                f"[actuator] filter={mode} missing={missing} -> relax"
            )
            continue

        means = {
            relation: np.stack(
                bags[
                    relation
                ],
                axis=0,
            ).mean(
                axis=0
            ).astype(
                np.float32
            )
            for relation in RELS
        }

        global_mean = np.stack(
            [
                means[
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

        templates = {}

        for layer_index, layer in enumerate(
            actuator_layers
        ):
            templates[
                layer
            ] = {
                relation: (
                    means[
                        relation
                    ][
                        layer_index
                    ]
                    - global_mean[
                        layer_index
                    ]
                ).astype(
                    np.float32
                )
                for relation in RELS
            }

        print(
            f"[actuator] requested={requested_filter} used={mode} "
            f"N={len(used)} counts="
            f"{dict(Counter(records[s]['gt'] for s in used))}"
        )

        return (
            templates,
            mode,
        )

    raise RuntimeError(
        "Could not fit four-class actuator templates."
    )


# =============================================================================
# Actual generation / steering
# =============================================================================

def build_generation_batch(
    processor,
    record,
    image,
    args,
):
    question = build_question(
        record,
        args,
    )

    rendered = rendered_prompt(
        processor,
        question,
        with_image=True,
    )

    return process_inputs(
        processor,
        rendered,
        image,
        args,
    )


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

    pred = normalize_relation(
        text
    )

    if pred not in RELSET:
        pred = None

    del output_ids

    return (
        text,
        pred,
    )


def extract_hidden_from_hook(output):
    if torch.is_tensor(
        output
    ):
        return (
            output,
            (
                "tensor",
                0,
            ),
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
                return (
                    item,
                    (
                        "tuple",
                        index,
                    ),
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
                return (
                    item,
                    (
                        "list",
                        index,
                    ),
                )

    raise RuntimeError(
        f"Cannot extract hidden from hook output type={type(output)}"
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

    if kind == "tuple":
        return tuple(
            values
        )

    return values


class SteerLast:
    def __init__(
        self,
        layers,
        templates,
        actuator_layers,
        target_relation,
        baseline_relation,
        mode,
        scale,
    ):
        self.handles = []
        self.done = {
            layer: False
            for layer in actuator_layers
        }

        self.templates = templates
        self.target_relation = target_relation
        self.baseline_relation = baseline_relation
        self.mode = mode
        self.scale = float(
            scale
        )

        for layer in actuator_layers:
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
            self.target_relation
        ]

        if (
            self.mode == "add"
            or self.baseline_relation not in RELSET
        ):
            return (
                self.scale
                * target
            )

        source = self.templates[
            layer
        ][
            self.baseline_relation
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

            hidden, descriptor = extract_hidden_from_hook(
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


def generate_baseline_for_sids(
    model,
    processor,
    records,
    sids,
    args,
    desc,
):
    result = {}

    for sid in tqdm(
        sids,
        desc=desc,
    ):
        image = Image.open(
            records[
                sid
            ][
                "image_path"
            ]
        ).convert(
            "RGB"
        )

        try:
            batch = build_generation_batch(
                processor,
                records[
                    sid
                ],
                image,
                args,
            )

            text, pred = generate(
                model,
                processor,
                batch,
                args,
            )

            result[
                sid
            ] = {
                "sid": sid,
                "pred": pred,
                "correct": int(
                    pred
                    == records[
                        sid
                    ][
                        "gt"
                    ]
                ),
                "text": text,
            }

            del batch

        finally:
            image.close()

        cleanup()

    return result


def prepare_test_baseline(
    existing,
    records,
    test_sids,
):
    result = {}

    for sid in test_sids:
        if sid not in existing:
            return None

        pred = existing[
            sid
        ][
            "pred"
        ]

        result[
            sid
        ] = {
            "sid": sid,
            "pred": pred,
            "correct": int(
                pred
                == records[
                    sid
                ][
                    "gt"
                ]
            ),
        }

    return result


def generate_all_selector_edits(
    model,
    processor,
    layers,
    actuator_templates,
    actuator_layers,
    records,
    sids,
    baseline_map,
    selector_map,
    edit_mode,
    args,
    desc,
):
    result = {}

    for sid in tqdm(
        sids,
        desc=desc,
    ):
        record = records[
            sid
        ]

        target_relation = selector_map[
            sid
        ][
            "pred"
        ]

        baseline_pred = baseline_map[
            sid
        ][
            "pred"
        ]

        image = Image.open(
            record[
                "image_path"
            ]
        ).convert(
            "RGB"
        )

        try:
            batch = build_generation_batch(
                processor,
                record,
                image,
                args,
            )

            with SteerLast(
                layers=layers,
                templates=actuator_templates,
                actuator_layers=actuator_layers,
                target_relation=target_relation,
                baseline_relation=baseline_pred,
                mode=edit_mode,
                scale=args.scale,
            ):
                text, pred = generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            result[
                sid
            ] = {
                "sid": sid,
                "pred": pred,
                "correct": int(
                    pred
                    == record[
                        "gt"
                    ]
                ),
                "text": text,
            }

            del batch

        finally:
            image.close()

        cleanup()

    return result


# =============================================================================
# Confidence-policy calibration
# =============================================================================

def selected_by_policy(
    sids,
    selector_map,
    baseline_map,
    apply_mode,
    coverage,
):
    eligible = []

    for sid in sids:
        selector_pred = selector_map[
            sid
        ][
            "pred"
        ]

        baseline_pred = baseline_map[
            sid
        ][
            "pred"
        ]

        confidence = float(
            selector_map[
                sid
            ][
                "confidence"
            ]
        )

        if apply_mode == "all":
            is_eligible = True

        elif apply_mode == "conflict_only":
            is_eligible = (
                baseline_pred in RELSET
                and selector_pred in RELSET
                and selector_pred
                != baseline_pred
            )

        else:
            raise ValueError(
                apply_mode
            )

        if is_eligible:
            eligible.append(
                (
                    confidence,
                    sid,
                )
            )

    eligible.sort(
        reverse=True
    )

    if not eligible:
        return (
            set(),
            float("inf"),
            0,
        )

    if coverage >= 1.0:
        selected = {
            sid
            for _, sid in eligible
        }

        threshold = min(
            confidence
            for confidence, _ in eligible
        )

        return (
            selected,
            threshold,
            len(
                eligible
            ),
        )

    k = int(
        math.ceil(
            coverage
            * len(
                eligible
            )
        )
    )

    k = max(
        1,
        min(
            k,
            len(
                eligible
            ),
        ),
    )

    chosen = eligible[
        :k
    ]

    selected = {
        sid
        for _, sid in chosen
    }

    threshold = chosen[
        -1
    ][
        0
    ]

    return (
        selected,
        threshold,
        len(
            eligible
        ),
    )


def evaluate_offline_policy(
    condition,
    selector_name,
    edit_mode,
    apply_mode,
    coverage,
    sids,
    records,
    baseline_map,
    selector_map,
    steered_map,
):
    (
        selected,
        threshold,
        eligible_n,
    ) = selected_by_policy(
        sids,
        selector_map,
        baseline_map,
        apply_mode,
        coverage,
    )

    rows = []

    for sid in sids:
        baseline_pred = baseline_map[
            sid
        ][
            "pred"
        ]

        baseline_correct = int(
            baseline_map[
                sid
            ][
                "correct"
            ]
        )

        if sid in selected:
            edit_pred = steered_map[
                sid
            ][
                "pred"
            ]

            edit_correct = int(
                steered_map[
                    sid
                ][
                    "correct"
                ]
            )

            applied = 1

        else:
            edit_pred = baseline_pred
            edit_correct = baseline_correct
            applied = 0

        rows.append({
            "condition": condition,
            "selector": selector_name,
            "sid": sid,
            "gt": records[sid]["gt"],
            "selector_pred": selector_map[sid]["pred"],
            "confidence": selector_map[sid]["confidence"],
            "baseline_pred": baseline_pred or "",
            "edit_pred": edit_pred or "",
            "baseline_correct": baseline_correct,
            "edit_correct": edit_correct,
            "applied": applied,
            "W2C": int(
                baseline_correct == 0
                and edit_correct == 1
            ),
            "C2W": int(
                baseline_correct == 1
                and edit_correct == 0
            ),
            "changed": int(
                (edit_pred or "")
                != (baseline_pred or "")
            ),
        })

    n = len(
        rows
    )

    wrong_n = sum(
        1
        - row[
            "baseline_correct"
        ]
        for row in rows
    )

    correct_n = (
        n
        - wrong_n
    )

    w2c = sum(
        row[
            "W2C"
        ]
        for row in rows
    )

    c2w = sum(
        row[
            "C2W"
        ]
        for row in rows
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

    summary = {
        "condition": condition,
        "selector": selector_name,
        "edit_mode": edit_mode,
        "apply_mode": apply_mode,
        "coverage": coverage,
        "confidence_threshold": threshold,
        "eligible_n": eligible_n,
        "applied": len(
            selected
        ),
        "applied_rate": (
            len(
                selected
            )
            / n
            if n
            else float("nan")
        ),
        "selector_acc": safe_mean(
            selector_map[
                sid
            ][
                "correct"
            ]
            for sid in sids
        ),
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": (
            edit_acc
            - base_acc
        ),
        "W2C": w2c,
        "W2C_rate_wrong": (
            w2c
            / wrong_n
            if wrong_n
            else float("nan")
        ),
        "C2W": c2w,
        "C2W_rate_correct": (
            c2w
            / correct_n
            if correct_n
            else float("nan")
        ),
        "net": (
            w2c
            - c2w
        ),
        "changed_rate": safe_mean(
            row[
                "changed"
            ]
            for row in rows
        ),
    }

    return (
        summary,
        rows,
    )


def choose_best_policy(
    policy_rows,
):
    # Primary objective: CAL edited accuracy.
    # Ties: fewer C2W, larger net, fewer interventions.
    return max(
        policy_rows,
        key=lambda row: (
            float(
                row[
                    "edit_acc"
                ]
            ),
            -int(
                row[
                    "C2W"
                ]
            ),
            int(
                row[
                    "net"
                ]
            ),
            -int(
                row[
                    "applied"
                ]
            ),
        ),
    )


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

    preset = model_preset(
        args.model_id
    )

    direction_layers = (
        preset[
            "direction_layers"
        ]
        if args.direction_layers == "auto"
        else parse_layer_spec(
            args.direction_layers
        )
    )

    actuator_layers = (
        preset[
            "actuator_layers"
        ]
        if args.actuator_layers == "auto"
        else parse_layer_spec(
            args.actuator_layers
        )
    )

    c_grid = parse_float_list(
        args.c_grid
    )

    coverages = parse_float_list(
        args.coverages
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

    feature_cache_path = (
        Path(
            args.feature_cache
        )
        if args.feature_cache
        else outdir
        / "feature_cache.npz"
    )

    actuator_cache_path = (
        Path(
            args.actuator_cache
        )
        if args.actuator_cache
        else outdir
        / "actuator_train_delta_cache.npz"
    )

    existing_test, baseline_path = load_existing_test_baseline(
        args.prior_output_dir
    )

    train_generation, train_generation_path = load_existing_train_generation(
        args.prior_output_dir
    )

    records = load_records(
        args
    )

    (
        train_sids,
        test_sids,
    ) = derive_train_test_ids(
        records,
        existing_test,
    )

    (
        fit_sids,
        cal_sids,
    ) = stratified_fit_cal_split(
        records,
        train_sids,
        args.cal_frac,
        args.seed,
    )

    print(
        "\n"
        + "=" * 170
    )

    print(
        "SUPERVISED RELATION SELECTOR — MIDDLE DIRECTION + BASELINE LOGITS"
    )

    print(
        "=" * 170
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"direction_layers={direction_layers}"
    )

    print(
        f"actuator_layers={actuator_layers}"
    )

    print(
        f"FIT={len(fit_sids)} CAL={len(cal_sids)} TEST={len(test_sids)}"
    )

    print(
        "TEST is untouched during selector / policy calibration."
    )

    print(
        "=" * 170
    )

    model, processor = load_model(
        args
    )

    layers, layer_path = resolve_layers(
        model
    )

    print(
        f"decoder={layer_path} blocks={len(layers)}"
    )

    for layer in (
        direction_layers
        + actuator_layers
    ):
        if (
            layer < 0
            or layer >= len(
                layers
            )
        ):
            raise RuntimeError(
                f"Invalid layer {layer} for {len(layers)}-block model."
            )

    token_candidates = answer_token_candidates(
        processor.tokenizer
    )

    (
        feature_map,
        actuator_map,
    ) = build_or_load_caches(
        model,
        processor,
        records,
        train_sids,
        direction_layers,
        actuator_layers,
        token_candidates,
        feature_cache_path,
        actuator_cache_path,
        args,
    )

    # -----------------------------------------------------------------
    # 1. FIT Direction prototypes.
    # -----------------------------------------------------------------
    direction_model = fit_direction_prototypes(
        feature_map,
        records,
        fit_sids,
        direction_layers,
    )

    # -----------------------------------------------------------------
    # 2. Train logits / Direction / fusion selectors on FIT.
    #    Select C only on CAL.
    # -----------------------------------------------------------------
    (
        selector_models,
        selector_calibration_rows,
        cal_selector_maps,
    ) = train_selectors(
        fit_sids,
        cal_sids,
        feature_map,
        direction_model,
        records,
        c_grid,
    )

    write_csv(
        outdir
        / "selector_calibration.csv",
        selector_calibration_rows,
    )

    cal_selector_rows = []

    for selector_name in (
        "logits",
        "direction",
        "fusion",
    ):
        cal_selector_rows.extend(
            cal_selector_maps[
                selector_name
            ].values()
        )

    write_csv(
        outdir
        / "cal_selector_predictions.csv",
        cal_selector_rows,
    )

    # -----------------------------------------------------------------
    # 3. CAL baseline + FIT actuator.
    # -----------------------------------------------------------------
    cal_baseline = generate_baseline_for_sids(
        model,
        processor,
        records,
        cal_sids,
        args,
        "CAL baseline",
    )

    (
        cal_actuator_templates,
        fit_actuator_filter,
    ) = fit_actuator_templates(
        actuator_map,
        records,
        fit_sids,
        actuator_layers,
        train_generation,
        args.template_filter,
    )

    # -----------------------------------------------------------------
    # 4. Each selector/edit-mode is generated once on CAL.
    #    Confidence/apply policies are evaluated offline.
    # -----------------------------------------------------------------
    cal_policy_rows = []
    best_policy_by_selector = {}

    for selector_name in (
        "logits",
        "direction",
        "fusion",
    ):
        selector_map = cal_selector_maps[
            selector_name
        ]

        steered_by_mode = {}

        for edit_mode in (
            "add",
            "contrast",
        ):
            steered_by_mode[
                edit_mode
            ] = generate_all_selector_edits(
                model,
                processor,
                layers,
                cal_actuator_templates,
                actuator_layers,
                records,
                cal_sids,
                cal_baseline,
                selector_map,
                edit_mode,
                args,
                (
                    f"CAL {selector_name} "
                    f"{edit_mode}"
                ),
            )

        selector_policy_candidates = []

        for edit_mode in (
            "add",
            "contrast",
        ):
            for apply_mode in (
                "all",
                "conflict_only",
            ):
                for coverage in coverages:
                    condition = (
                        f"{selector_name}"
                        f"__{edit_mode}"
                        f"__{apply_mode}"
                        f"__cov{coverage:g}"
                    )

                    summary, _ = evaluate_offline_policy(
                        condition=condition,
                        selector_name=selector_name,
                        edit_mode=edit_mode,
                        apply_mode=apply_mode,
                        coverage=coverage,
                        sids=cal_sids,
                        records=records,
                        baseline_map=cal_baseline,
                        selector_map=selector_map,
                        steered_map=steered_by_mode[
                            edit_mode
                        ],
                    )

                    cal_policy_rows.append(
                        summary
                    )

                    selector_policy_candidates.append(
                        summary
                    )

        best = choose_best_policy(
            selector_policy_candidates
        )

        best_policy_by_selector[
            selector_name
        ] = best

        print(
            f"[CAL policy] {selector_name:9s} "
            f"selector_acc={best['selector_acc']:.4f} | "
            f"{best['edit_mode']}/{best['apply_mode']}/cov={best['coverage']:.2f} | "
            f"{best['base_acc']:.4f}->{best['edit_acc']:.4f} "
            f"{best['gain']:+.4f} | W2C={best['W2C']} C2W={best['C2W']}"
        )

    write_csv(
        outdir
        / "cal_policy_search.csv",
        cal_policy_rows,
    )

    # -----------------------------------------------------------------
    # 5. Build TEST selector predictions using the FIT-trained selectors.
    # -----------------------------------------------------------------
    test_selector_maps = {}
    test_selector_rows = []

    for selector_name in (
        "logits",
        "direction",
        "fusion",
    ):
        x_test, _ = build_xy(
            selector_name,
            test_sids,
            feature_map,
            direction_model,
            records,
        )

        model_info = selector_models[
            selector_name
        ]

        rows = prediction_rows(
            selector_name,
            test_sids,
            x_test,
            model_info[
                "model"
            ],
            records,
        )

        test_selector_maps[
            selector_name
        ] = {
            int(row["sid"]): row
            for row in rows
        }

        test_selector_rows.extend(
            rows
        )

        print(
            f"[TEST selector] {selector_name:9s} "
            f"acc={safe_mean(row['correct'] for row in rows):.4f} "
            f"mean_conf={safe_mean(row['confidence'] for row in rows):.4f}"
        )

    write_csv(
        outdir
        / "test_selector_predictions.csv",
        test_selector_rows,
    )

    # -----------------------------------------------------------------
    # 6. Reuse exact prior TEST baseline.
    # -----------------------------------------------------------------
    test_baseline = prepare_test_baseline(
        existing_test,
        records,
        test_sids,
    )

    if test_baseline is None:
        print(
            "[warning] prior TEST baseline could not be aligned; regenerating."
        )

        test_baseline = generate_baseline_for_sids(
            model,
            processor,
            records,
            test_sids,
            args,
            "TEST baseline fallback",
        )

    test_base_acc = safe_mean(
        test_baseline[
            sid
        ][
            "correct"
        ]
        for sid in test_sids
    )

    # -----------------------------------------------------------------
    # 7. Refit actuator only on full TRAIN. Selector remains FIT-trained:
    #    CAL information is not leaked into selector training.
    # -----------------------------------------------------------------
    (
        full_actuator_templates,
        full_actuator_filter,
    ) = fit_actuator_templates(
        actuator_map,
        records,
        train_sids,
        actuator_layers,
        train_generation,
        args.template_filter,
    )

    # -----------------------------------------------------------------
    # 8. Final actual-generation TEST.
    # -----------------------------------------------------------------
    test_summary_rows = []
    test_detail_rows = []

    for selector_name in (
        "logits",
        "direction",
        "fusion",
    ):
        selector_map = test_selector_maps[
            selector_name
        ]

        policy = best_policy_by_selector[
            selector_name
        ]

        (
            selected_sids,
            test_conf_threshold,
            eligible_n,
        ) = selected_by_policy(
            test_sids,
            selector_map,
            test_baseline,
            policy[
                "apply_mode"
            ],
            float(
                policy[
                    "coverage"
                ]
            ),
        )

        # Generate only selected TEST samples.
        selected_steered = generate_all_selector_edits(
            model,
            processor,
            layers,
            full_actuator_templates,
            actuator_layers,
            records,
            sorted(
                selected_sids
            ),
            test_baseline,
            selector_map,
            policy[
                "edit_mode"
            ],
            args,
            (
                f"TEST {selector_name} "
                f"{policy['edit_mode']} "
                f"{policy['apply_mode']} "
                f"cov={policy['coverage']}"
            ),
        )

        # Offline combine selected edits with untouched baseline.
        all_steered = {}

        for sid in test_sids:
            if sid in selected_steered:
                all_steered[
                    sid
                ] = selected_steered[
                    sid
                ]

            else:
                all_steered[
                    sid
                ] = {
                    "sid": sid,
                    "pred": test_baseline[
                        sid
                    ][
                        "pred"
                    ],
                    "correct": test_baseline[
                        sid
                    ][
                        "correct"
                    ],
                    "text": "",
                }

        condition = (
            f"{selector_name}"
            f"__{policy['edit_mode']}"
            f"__{policy['apply_mode']}"
            f"__cov{float(policy['coverage']):g}"
        )

        # Force exact TEST-selected set rather than recomputing after we already
        # computed it above. Coverage selection itself uses no TEST labels.
        rows = []

        for sid in test_sids:
            base = test_baseline[
                sid
            ]

            baseline_correct = int(
                base[
                    "correct"
                ]
            )

            applied = int(
                sid in selected_sids
            )

            if applied:
                edit = selected_steered[
                    sid
                ]

                edit_pred = edit[
                    "pred"
                ]

                edit_correct = int(
                    edit[
                        "correct"
                    ]
                )

                edit_text = edit.get(
                    "text",
                    "",
                )

            else:
                edit_pred = base[
                    "pred"
                ]

                edit_correct = baseline_correct
                edit_text = ""

            rows.append({
                "condition": condition,
                "selector": selector_name,
                "sid": sid,
                "gt": records[sid]["gt"],
                "selector_pred": selector_map[sid]["pred"],
                "confidence": selector_map[sid]["confidence"],
                "baseline_pred": base["pred"] or "",
                "edit_pred": edit_pred or "",
                "baseline_correct": baseline_correct,
                "edit_correct": edit_correct,
                "applied": applied,
                "W2C": int(
                    baseline_correct == 0
                    and edit_correct == 1
                ),
                "C2W": int(
                    baseline_correct == 1
                    and edit_correct == 0
                ),
                "changed": int(
                    (edit_pred or "")
                    != (base["pred"] or "")
                ),
                "edit_text": edit_text,
            })

        n = len(
            rows
        )

        wrong_n = sum(
            1
            - row[
                "baseline_correct"
            ]
            for row in rows
        )

        correct_n = (
            n
            - wrong_n
        )

        w2c = sum(
            row[
                "W2C"
            ]
            for row in rows
        )

        c2w = sum(
            row[
                "C2W"
            ]
            for row in rows
        )

        edit_acc = safe_mean(
            row[
                "edit_correct"
            ]
            for row in rows
        )

        summary = {
            "selector": selector_name,
            "selected_C": selector_models[selector_name]["C"],
            "direction_layers": ",".join(
                map(
                    str,
                    direction_layers,
                )
            ),
            "actuator_layers": ",".join(
                map(
                    str,
                    actuator_layers,
                )
            ),
            "CAL_edit_mode": policy["edit_mode"],
            "CAL_apply_mode": policy["apply_mode"],
            "CAL_coverage": policy["coverage"],
            "TEST_confidence_threshold": test_conf_threshold,
            "TEST_eligible_n": eligible_n,
            "TEST_applied": len(
                selected_sids
            ),
            "TEST_applied_rate": (
                len(
                    selected_sids
                )
                / len(
                    test_sids
                )
            ),
            "selector_acc": safe_mean(
                selector_map[
                    sid
                ][
                    "correct"
                ]
                for sid in test_sids
            ),
            "base_acc": test_base_acc,
            "edit_acc": edit_acc,
            "gain": (
                edit_acc
                - test_base_acc
            ),
            "W2C": w2c,
            "W2C_rate_wrong": (
                w2c
                / wrong_n
                if wrong_n
                else float("nan")
            ),
            "C2W": c2w,
            "C2W_rate_correct": (
                c2w
                / correct_n
                if correct_n
                else float("nan")
            ),
            "net": (
                w2c
                - c2w
            ),
            "changed_rate": safe_mean(
                row[
                    "changed"
                ]
                for row in rows
            ),
        }

        test_summary_rows.append(
            summary
        )

        test_detail_rows.extend(
            rows
        )

    write_csv(
        outdir
        / "test_summary.csv",
        test_summary_rows,
    )

    write_csv(
        outdir
        / "test_details.csv",
        test_detail_rows,
    )

    # -----------------------------------------------------------------
    # Final console summary.
    # -----------------------------------------------------------------
    print(
        "\n"
        + "=" * 180
    )

    print(
        "FINAL TEST — SUPERVISED RELATION SELECTOR"
    )

    print(
        "=" * 180
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"direction_layers={direction_layers}"
    )

    print(
        f"actuator_layers={actuator_layers}"
    )

    print(
        f"FIT actuator filter={fit_actuator_filter}"
    )

    print(
        f"TEST actuator filter={full_actuator_filter}"
    )

    print(
        f"baseline={test_base_acc:.4f}"
    )

    print()

    print(
        "selector   | sel_acc | policy                         | "
        "acc base->edit gain | applied | W2C/wrong | C2W/correct | net"
    )

    for row in test_summary_rows:
        policy_text = (
            f"{row['CAL_edit_mode']}/"
            f"{row['CAL_apply_mode']}/"
            f"cov={float(row['CAL_coverage']):.2f}"
        )

        print(
            f"{row['selector']:10s} | "
            f"{row['selector_acc']:.4f} | "
            f"{policy_text:30s} | "
            f"{row['base_acc']:.4f}->"
            f"{row['edit_acc']:.4f} "
            f"{row['gain']:+.4f} | "
            f"{row['TEST_applied']:3d} | "
            f"{row['W2C']}/{row['W2C_rate_wrong']:.3f} | "
            f"{row['C2W']}/{row['C2W_rate_correct']:.3f} | "
            f"{row['net']:+d}"
        )

    print()
    print(
        "Interpretation order:"
    )

    print(
        "  1) Compare selector_acc: logits vs Direction vs fusion."
    )

    print(
        "  2) Compare final edit_acc / W2C / C2W."
    )

    print(
        "  3) If fusion > both components, middle spatial evidence and output "
        "logits provide complementary information."
    )

    metadata = {
        "model_id": args.model_id,
        "prior_output_dir": args.prior_output_dir,
        "reused_test_baseline": (
            str(
                baseline_path
            )
            if baseline_path is not None
            else None
        ),
        "reused_train_generation": (
            str(
                train_generation_path
            )
            if train_generation_path is not None
            else None
        ),
        "direction_layers": direction_layers,
        "actuator_layers": actuator_layers,
        "FIT_N": len(
            fit_sids
        ),
        "CAL_N": len(
            cal_sids
        ),
        "TEST_N": len(
            test_sids
        ),
        "template_filter_requested": args.template_filter,
        "FIT_actuator_filter_used": fit_actuator_filter,
        "TEST_actuator_filter_used": full_actuator_filter,
        "best_policy_by_selector": best_policy_by_selector,
        "test_summary": test_summary_rows,
    }

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
