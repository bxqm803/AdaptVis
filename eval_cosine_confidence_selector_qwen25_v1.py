#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_cosine_confidence_selector_qwen25_v1.py

Goal
----
Test a NEW non-oracle selector for the already-discovered late causal actuator
in Qwen2.5-VL-3B and Qwen2.5-VL-7B.

We DO NOT repeat:
- full all-layer causal search
- old residual Direction head/layer search
- direction-bundle construction
- TEST baseline generation when an existing baseline CSV is available
- TRAIN Real/Gray correctness generation when an existing CSV is available

Known actuator windows from previous runs:
- Qwen2.5-VL-3B: L32-L35
- Qwen2.5-VL-7B: L25-L27

New selector
------------
At late last-token layer l:

    Delta_i,l = h_last(real)_i,l - h_last(gray)_i,l

TRAIN/FIT relation templates:

    mu_r,l = mean(Delta_i,l | relation=r)
    mu_global,l = balanced mean_r(mu_r,l)
    s_r,l = mu_r,l - mu_global,l

For a sample:

    q_i,l = Delta_i,l - mu_global,l

Relation score over selector window W:

    score(r) = mean_{l in W} cos(q_i,l, s_r,l)

Prediction:
    r_hat = argmax_r score(r)

Confidence:
    margin = top1_score - top2_score

We use the original TRAIN split only:
- FIT learns templates
- CAL selects selector window and confidence coverage
- TEST remains untouched

Confidence is calibrated as COVERAGE, not a raw cosine threshold:
- 1.00 = intervene on all eligible samples
- 0.75 = top 75% most confident eligible samples
- 0.50 = top 50%
- 0.25 = top 25%

CAL uses actual model.generate() to choose:
- edit mode: add / contrast
- apply mode: all / conflict_only
- confidence coverage

After selection, templates are refit on full TRAIN from cached activations.
On TEST we compare:
1) same policy with no confidence filtering (coverage=1.0)
2) selected confidence coverage

This directly measures whether cosine + confidence reduces C2W while retaining W2C.

Examples
--------
Qwen3B:
CUDA_VISIBLE_DEVICES=0 python eval_cosine_confidence_selector_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --prior-output-dir output/qwen3b_last_causal_auto_v1 \
  --output-dir output/qwen3b_cosine_confidence_selector_v1 \
  --overwrite

Qwen7B:
CUDA_VISIBLE_DEVICES=0 python eval_cosine_confidence_selector_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --prior-output-dir output/qwen7b_real_last_causal_steering_v2 \
  --output-dir output/qwen7b_cosine_confidence_selector_v1 \
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
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
EPS = 1e-12


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--model-id", required=True)

    p.add_argument(
        "--prior-output-dir",
        required=True,
        help=(
            "Existing output directory from the earlier causal steering run. "
            "Used to reuse TEST baseline and TRAIN Real/Gray correctness."
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
        "--device",
        default="cuda:0",
    )

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
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )

    p.add_argument(
        "--cal-frac",
        type=float,
        default=0.25,
        help="Fraction of original TRAIN used for selector/policy calibration.",
    )

    p.add_argument(
        "--selector-layers",
        default="auto",
        help=(
            "Late layers to scan for cosine selector. "
            "auto: 3B=L28-35, 7B=L20-27."
        ),
    )

    p.add_argument(
        "--selector-window-lengths",
        default="1,2,3,4",
    )

    p.add_argument(
        "--coverages",
        default="1.0,0.75,0.5,0.25",
        help="Confidence coverages evaluated on CAL.",
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--cache-deltas",
        default="",
        help=(
            "Optional activation cache path. "
            "Default: <output-dir>/late_real_gray_deltas.npz"
        ),
    )

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
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
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


def normalize_relation(x):
    s = str(x).strip().lower()

    if re.search(r"\bleft\b", s):
        return "left"

    if re.search(r"\bright\b", s):
        return "right"

    if (
        re.search(r"\babove\b", s)
        or re.search(r"\bover\b", s)
        or re.search(r"\bon top of\b", s)
    ):
        return "above"

    if (
        re.search(r"\bbelow\b", s)
        or re.search(r"\bunder(?:neath)?\b", s)
        or re.search(r"\bbeneath\b", s)
    ):
        return "below"

    return s


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

    for piece in str(spec).split(","):
        piece = piece.strip()

        if not piece:
            continue

        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1

            out.extend(
                range(
                    a,
                    b + step,
                    step,
                )
            )

        else:
            out.append(int(piece))

    return sorted(set(out))


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_dtype(model_id, dtype):
    if dtype != "auto":
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

    return torch.bfloat16


def model_preset(model_id):
    low = model_id.lower()

    if "3b" in low:
        return {
            "name": "qwen3b",
            "actuator_layers": [32, 33, 34, 35],
            "selector_layers": list(range(28, 36)),
        }

    if "7b" in low:
        return {
            "name": "qwen7b",
            "actuator_layers": [25, 26, 27],
            "selector_layers": list(range(20, 28)),
        }

    raise ValueError(
        "This script is intentionally restricted to the already-analyzed "
        "Qwen2.5-VL-3B and Qwen2.5-VL-7B models."
    )


# =============================================================================
# Existing prior results
# =============================================================================

def find_existing_file(root, names):
    root = Path(root)

    for name in names:
        path = root / name

        if path.exists():
            return path

    return None


def row_sid(row):
    for key in [
        "sid",
        "sample_index",
        "id",
    ]:
        if key in row and str(row[key]).strip():
            return int(row[key])

    raise KeyError(
        f"Could not identify sample id in columns={list(row)}"
    )


def row_pred(row):
    for key in [
        "pred",
        "baseline_pred",
        "generation_pred",
        "real_pred",
    ]:
        if key in row:
            value = normalize_relation(
                row[key]
            )

            if value in RELSET:
                return value

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

    rows = read_csv(path)
    result = {}

    for row in rows:
        sid = row_sid(row)
        pred = row_pred(row)

        correct = None

        for key in [
            "correct",
            "baseline_correct",
        ]:
            if key in row and str(row[key]).strip():
                try:
                    correct = int(float(row[key]))
                except Exception:
                    pass

        result[sid] = {
            "sid": sid,
            "pred": pred,
            "correct": correct,
            "source": str(path),
        }

    print(
        f"[reuse] TEST baseline: {path} | N={len(result)}"
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
        return None, None

    rows = read_csv(path)
    result = {}

    for row in rows:
        try:
            sid = row_sid(row)
        except Exception:
            continue

        real_correct = None
        gray_correct = None

        for key in [
            "real_correct",
            "real_is_correct",
        ]:
            if key in row and str(row[key]).strip():
                try:
                    real_correct = int(float(row[key]))
                except Exception:
                    pass

        for key in [
            "gray_correct",
            "gray_is_correct",
        ]:
            if key in row and str(row[key]).strip():
                try:
                    gray_correct = int(float(row[key]))
                except Exception:
                    pass

        if real_correct is None:
            real_pred = normalize_relation(
                row.get(
                    "real_pred",
                    "",
                )
            )

            gt = normalize_relation(
                row.get(
                    "gt",
                    row.get(
                        "relation",
                        "",
                    ),
                )
            )

            if (
                real_pred in RELSET
                and gt in RELSET
            ):
                real_correct = int(
                    real_pred == gt
                )

        if gray_correct is None:
            gray_pred = normalize_relation(
                row.get(
                    "gray_pred",
                    "",
                )
            )

            gt = normalize_relation(
                row.get(
                    "gt",
                    row.get(
                        "relation",
                        "",
                    ),
                )
            )

            if (
                gray_pred in RELSET
                and gt in RELSET
            ):
                gray_correct = int(
                    gray_pred == gt
                )

        result[sid] = {
            "real_correct": real_correct,
            "gray_correct": gray_correct,
        }

    print(
        f"[reuse] TRAIN Real/Gray generation labels: {path} | N={len(result)}"
    )

    return result, path


# =============================================================================
# Dataset
# =============================================================================

def parse_subject_reference(question):
    q = str(question)

    for pattern in [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]:
        m = re.search(
            pattern,
            q,
            flags=re.I | re.S,
        )

        if m:
            return (
                re.sub(
                    r"\s+",
                    " ",
                    m.group(1),
                ).strip(),
                re.sub(
                    r"\s+",
                    " ",
                    m.group(2),
                ).strip(),
            )

    return None, None


def load_all_records(args):
    with open(
        args.annotation_json,
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    prompts = []

    with open(
        args.prompt_jsonl,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
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

        subject, reference = parse_subject_reference(
            row.get(
                "question",
                "",
            )
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

        image_id = int(
            annotations[sid][0]
        )

        image_path = (
            Path(args.data_root)
            / "val2017"
            / f"{image_id:012d}.jpg"
        )

        records[sid] = {
            "sid": sid,
            "gt": gt,
            "subject": subject,
            "reference": reference,
            "image_path": str(image_path),
        }

    if not records:
        raise RuntimeError(
            "No COCO-two records loaded."
        )

    return records


def derive_train_test_ids(records, existing_test_baseline):
    if existing_test_baseline is None:
        raise RuntimeError(
            "Could not find existing TEST baseline. "
            "This script intentionally reuses the previous split rather than "
            "creating a new one. Point --prior-output-dir to the earlier run."
        )

    test_sids = sorted(
        sid
        for sid in existing_test_baseline
        if sid in records
    )

    test_set = set(test_sids)

    train_sids = sorted(
        sid
        for sid in records
        if sid not in test_set
    )

    print(
        f"[split reuse] TRAIN={len(train_sids)} TEST={len(test_sids)}"
    )

    return train_sids, test_sids


def stratified_fit_cal_split(records, train_sids, cal_frac, seed):
    rng = random.Random(seed)

    fit = []
    cal = []

    train_set = set(train_sids)

    for relation in RELS:
        xs = sorted(
            sid
            for sid in train_set
            if records[sid]["gt"] == relation
        )

        rng.shuffle(xs)

        if len(xs) < 2:
            raise RuntimeError(
                f"Too few TRAIN samples for {relation}: {len(xs)}"
            )

        n_cal = int(
            round(
                len(xs)
                * cal_frac
            )
        )

        n_cal = max(
            1,
            min(
                n_cal,
                len(xs) - 1,
            ),
        )

        cal.extend(
            xs[:n_cal]
        )

        fit.extend(
            xs[n_cal:]
        )

    return sorted(fit), sorted(cal)


# =============================================================================
# Qwen model / prompt / generation
# =============================================================================

def model_class():
    if not hasattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    ):
        raise RuntimeError(
            "Transformers has no Qwen2_5_VLForConditionalGeneration."
        )

    return transformers.Qwen2_5_VLForConditionalGeneration


def load_model(args):
    dtype = auto_dtype(
        args.model_id,
        args.dtype,
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

    cls = model_class()

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
        "Could not locate Qwen decoder layers."
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


def build_batch(processor, image, rec, args):
    question = args.prompt_template.format(
        subject=rec["subject"],
        reference=rec["reference"],
    )

    prompt = build_prompt(
        processor,
        image,
        question,
    )

    last_error = None

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
            last_error = exc

    else:
        raise RuntimeError(
            f"Processor failed: {last_error}"
        )

    return {
        key: (
            value.to(args.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def parse_pred(text):
    return normalize_relation(
        text
    ) if normalize_relation(text) in RELSET else None


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

    pred = parse_pred(text)

    del output_ids

    return text, pred


def make_gray_image(real, value):
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
# Hooks
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
        f"Cannot extract hidden from {type(output)}"
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


class CaptureLast:
    def __init__(self, layers, selected):
        self.handles = []
        self.states = {}
        self.done = {
            layer: False
            for layer in selected
        }

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

            self.states[layer] = (
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
        self.scale = float(scale)
        self.mode = mode
        self.source = source

        for layer in selected:
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
            self.mode == "add"
            or self.source not in RELSET
        ):
            return (
                self.scale
                * target
            )

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
# Efficient Real-Gray delta cache: forward only, no generate()
# =============================================================================

def capture_forward_last(
    model,
    processor,
    layers,
    image,
    rec,
    selected,
    args,
):
    batch = build_batch(
        processor,
        image,
        rec,
        args,
    )

    with CaptureLast(
        layers,
        selected,
    ) as capture:

        with torch.inference_mode():
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

        states = dict(
            capture.states
        )

    del batch
    del outputs

    return states


def build_or_load_delta_cache(
    model,
    processor,
    layers,
    records,
    selected_layers,
    args,
    cache_path,
):
    cache_path = Path(cache_path)

    if (
        cache_path.exists()
        and not args.overwrite
    ):
        with np.load(
            cache_path,
            allow_pickle=True,
        ) as z:
            sids = np.asarray(
                z["sample_index"],
                dtype=np.int64,
            )

            cached_layers = np.asarray(
                z["layers"],
                dtype=np.int64,
            ).tolist()

            deltas = np.asarray(
                z["deltas"],
                dtype=np.float32,
            )

        if cached_layers != selected_layers:
            raise RuntimeError(
                f"Cache layers {cached_layers} != requested {selected_layers}"
            )

        print(
            f"[reuse] activation cache: {cache_path} | "
            f"shape={deltas.shape}"
        )

        return {
            int(sid): deltas[i]
            for i, sid in enumerate(
                sids.tolist()
            )
        }

    result = {}

    for sid in tqdm(
        sorted(records),
        desc="Real-Gray late last-token forward cache",
    ):
        rec = records[sid]
        real = None
        gray = None

        try:
            real = Image.open(
                rec["image_path"]
            ).convert("RGB")

            gray = make_gray_image(
                real,
                args.gray_value,
            )

            real_states = capture_forward_last(
                model,
                processor,
                layers,
                real,
                rec,
                selected_layers,
                args,
            )

            gray_states = capture_forward_last(
                model,
                processor,
                layers,
                gray,
                rec,
                selected_layers,
                args,
            )

            if any(
                layer not in real_states
                or layer not in gray_states
                for layer in selected_layers
            ):
                raise RuntimeError(
                    "Incomplete layer capture."
                )

            result[sid] = np.stack(
                [
                    (
                        real_states[layer]
                        - gray_states[layer]
                    ).astype(np.float32)
                    for layer in selected_layers
                ],
                axis=0,
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    sids = np.asarray(
        sorted(result),
        dtype=np.int64,
    )

    arr = np.stack(
        [
            result[int(sid)]
            for sid in sids
        ],
        axis=0,
    ).astype(np.float32)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        cache_path,
        sample_index=sids,
        layers=np.asarray(
            selected_layers,
            dtype=np.int64,
        ),
        deltas=arr,
    )

    print(
        f"[saved] activation cache: {cache_path} | shape={arr.shape}"
    )

    return result


# =============================================================================
# Template fit / cosine selector
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
        return True

    rc = train_generation[sid].get(
        "real_correct"
    )

    gc = train_generation[sid].get(
        "gray_correct"
    )

    if mode == "real_correct":
        return (
            rc is not None
            and bool(rc)
        )

    if mode == "real_correct_gray_wrong":
        return (
            rc is not None
            and gc is not None
            and bool(rc)
            and not bool(gc)
        )

    return True


def fit_templates(
    delta_cache,
    layer_index,
    records,
    sids,
    selected_layers,
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
    }[requested_filter]

    for mode in filter_sequence:
        bags = {
            layer: {
                relation: []
                for relation in RELS
            }
            for layer in selected_layers
        }

        used = []

        for sid in sids:
            if sid not in delta_cache:
                continue

            if not sample_allowed(
                sid,
                train_generation,
                mode,
            ):
                continue

            relation = records[sid]["gt"]

            for layer in selected_layers:
                bags[layer][relation].append(
                    delta_cache[sid][
                        layer_index[layer]
                    ]
                )

            used.append(sid)

        missing = [
            (layer, relation)
            for layer in selected_layers
            for relation in RELS
            if not bags[layer][relation]
        ]

        if missing:
            print(
                f"[template] filter={mode} missing {len(missing)} cells -> relax"
            )
            continue

        templates = {}

        for layer in selected_layers:
            mu = {
                relation: np.stack(
                    bags[layer][relation],
                    axis=0,
                ).mean(axis=0).astype(
                    np.float32
                )
                for relation in RELS
            }

            global_mu = np.stack(
                [
                    mu[relation]
                    for relation in RELS
                ],
                axis=0,
            ).mean(axis=0).astype(
                np.float32
            )

            shared = {
                relation: (
                    mu[relation]
                    - global_mu
                ).astype(np.float32)
                for relation in RELS
            }

            templates[layer] = {
                "global": global_mu,
                "shared": shared,
            }

        print(
            f"[template] requested={requested_filter} used={mode} N={len(used)}"
        )

        return templates, mode

    raise RuntimeError(
        "Could not fit four relation templates."
    )


def cosine_scores(
    delta_cache,
    layer_index,
    templates,
    sid,
    selector_window,
):
    scores = {
        relation: []
        for relation in RELS
    }

    for layer in selector_window:
        delta = delta_cache[sid][
            layer_index[layer]
        ].astype(np.float64)

        q = (
            delta
            - templates[layer]["global"].astype(
                np.float64
            )
        )

        qnorm = np.linalg.norm(q)

        if qnorm <= EPS:
            for relation in RELS:
                scores[relation].append(
                    0.0
                )

            continue

        q = q / qnorm

        for relation in RELS:
            s = templates[layer]["shared"][
                relation
            ].astype(np.float64)

            snorm = np.linalg.norm(s)

            if snorm <= EPS:
                score = 0.0
            else:
                score = float(
                    np.dot(
                        q,
                        s / snorm,
                    )
                )

            scores[relation].append(
                score
            )

    aggregate = {
        relation: float(
            np.mean(
                scores[relation]
            )
        )
        for relation in RELS
    }

    ordered = sorted(
        aggregate.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    pred = ordered[0][0]
    top1 = ordered[0][1]
    top2 = ordered[1][1]
    margin = top1 - top2

    return pred, margin, aggregate


def candidate_selector_windows(
    selector_layers,
    lengths,
):
    allowed = set(
        selector_layers
    )

    windows = []

    for length in lengths:
        for start in selector_layers:
            window = tuple(
                range(
                    start,
                    start + length,
                )
            )

            if all(
                layer in allowed
                for layer in window
            ):
                windows.append(
                    window
                )

    return sorted(
        set(windows),
        key=lambda window: (
            len(window),
            window[0],
        ),
    )


def evaluate_selector_windows(
    delta_cache,
    layer_index,
    templates,
    records,
    cal_sids,
    selector_layers,
    lengths,
):
    rows = []

    predictions_by_window = {}

    for window in candidate_selector_windows(
        selector_layers,
        lengths,
    ):
        correct = 0
        margins = []
        sample_predictions = {}

        for sid in cal_sids:
            pred, margin, scores = cosine_scores(
                delta_cache,
                layer_index,
                templates,
                sid,
                window,
            )

            correct += int(
                pred == records[sid]["gt"]
            )

            margins.append(
                margin
            )

            sample_predictions[sid] = {
                "pred": pred,
                "margin": margin,
                "scores": scores,
            }

        acc = (
            correct / len(cal_sids)
            if cal_sids
            else float("nan")
        )

        rows.append({
            "layers": ",".join(
                map(str, window)
            ),
            "length": len(window),
            "N": len(cal_sids),
            "selector_acc": acc,
            "mean_margin": safe_mean(margins),
        })

        predictions_by_window[
            window
        ] = sample_predictions

    best = max(
        rows,
        key=lambda row: (
            float(
                row["selector_acc"]
            ),
            float(
                row["mean_margin"]
            ),
            -int(
                row["length"]
            ),
        ),
    )

    best_window = tuple(
        int(x)
        for x in best["layers"].split(",")
    )

    return (
        best_window,
        rows,
        predictions_by_window[
            best_window
        ],
    )


# =============================================================================
# Baseline generation / reuse
# =============================================================================

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
        rec = records[sid]
        image = None

        try:
            image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            batch = build_batch(
                processor,
                image,
                rec,
                args,
            )

            text, pred = generate(
                model,
                processor,
                batch,
                args,
            )

            result[sid] = {
                "sid": sid,
                "pred": pred,
                "correct": int(
                    pred == rec["gt"]
                ),
                "text": text,
            }

            del batch

        finally:
            if image is not None:
                image.close()

            cleanup()

    return result


def prepare_test_baseline(
    existing,
    records,
    test_sids,
):
    if existing is None:
        return None

    result = {}

    for sid in test_sids:
        if sid not in existing:
            return None

        pred = existing[sid]["pred"]

        if pred not in RELSET:
            # Keep unparsable baseline as wrong rather than forcing a label.
            correct = 0
        else:
            correct = int(
                pred == records[sid]["gt"]
            )

        result[sid] = {
            "sid": sid,
            "pred": pred,
            "correct": correct,
        }

    return result


# =============================================================================
# Confidence / actual steering
# =============================================================================

def select_by_coverage(
    sids,
    selector_predictions,
    baseline_map,
    coverage,
    apply_mode,
):
    eligible = []

    for sid in sids:
        pred = selector_predictions[sid]["pred"]
        margin = selector_predictions[sid]["margin"]

        if apply_mode == "all":
            eligible.append(
                (
                    margin,
                    sid,
                )
            )

        elif apply_mode == "conflict_only":
            baseline_pred = baseline_map[sid]["pred"]

            if (
                pred in RELSET
                and baseline_pred in RELSET
                and pred != baseline_pred
            ):
                eligible.append(
                    (
                        margin,
                        sid,
                    )
                )

        else:
            raise ValueError(
                apply_mode
            )

    eligible.sort(
        reverse=True
    )

    if not eligible:
        return set(), float("inf"), 0

    if coverage >= 1.0:
        selected = {
            sid
            for _, sid in eligible
        }

        threshold = min(
            margin
            for margin, _ in eligible
        )

        return (
            selected,
            threshold,
            len(eligible),
        )

    k = int(
        math.ceil(
            len(eligible)
            * coverage
        )
    )

    k = max(
        1,
        min(
            k,
            len(eligible),
        ),
    )

    chosen = eligible[:k]

    selected = {
        sid
        for _, sid in chosen
    }

    threshold = chosen[
        -1
    ][0]

    return (
        selected,
        threshold,
        len(eligible),
    )


def run_steering_policy(
    model,
    processor,
    layers,
    actuator_templates,
    actuator_layers,
    records,
    sids,
    baseline_map,
    selector_predictions,
    selected_sids,
    args,
    edit_mode,
    condition,
):
    rows = []

    for sid in tqdm(
        sids,
        desc=condition,
    ):
        rec = records[sid]
        base = baseline_map[sid]
        baseline_pred = base["pred"]
        baseline_correct = int(
            base["correct"]
        )

        selector_pred = selector_predictions[
            sid
        ]["pred"]

        margin = selector_predictions[
            sid
        ]["margin"]

        if sid not in selected_sids:
            rows.append({
                "condition": condition,
                "sid": sid,
                "gt": rec["gt"],
                "selector_pred": selector_pred,
                "confidence": margin,
                "baseline_pred": baseline_pred or "",
                "edit_pred": baseline_pred or "",
                "baseline_correct": baseline_correct,
                "edit_correct": baseline_correct,
                "applied": 0,
                "W2C": 0,
                "C2W": 0,
                "changed": 0,
            })

            continue

        image = None

        try:
            image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            batch = build_batch(
                processor,
                image,
                rec,
                args,
            )

            with SteerLast(
                layers=layers,
                templates=actuator_templates,
                selected=actuator_layers,
                target=selector_pred,
                scale=args.scale,
                mode=edit_mode,
                source=baseline_pred,
            ):
                text, pred = generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            edit_correct = int(
                pred == rec["gt"]
            )

            rows.append({
                "condition": condition,
                "sid": sid,
                "gt": rec["gt"],
                "selector_pred": selector_pred,
                "confidence": margin,
                "baseline_pred": baseline_pred or "",
                "edit_pred": pred or "",
                "baseline_correct": baseline_correct,
                "edit_correct": edit_correct,
                "applied": 1,
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
                    != (baseline_pred or "")
                ),
                "text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


def summarize_policy(
    rows,
    condition,
    coverage,
    threshold,
    eligible_n,
):
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
        "coverage_target": coverage,
        "confidence_threshold": threshold,
        "eligible_n": eligible_n,
        "applied": applied,
        "applied_rate": (
            applied / n
            if n
            else float("nan")
        ),
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

    preset = model_preset(
        args.model_id
    )

    actuator_layers = preset[
        "actuator_layers"
    ]

    selector_layers = (
        preset["selector_layers"]
        if args.selector_layers == "auto"
        else parse_layer_spec(
            args.selector_layers
        )
    )

    capture_layers = sorted(
        set(
            selector_layers
            + actuator_layers
        )
    )

    layer_index = {
        layer: i
        for i, layer in enumerate(
            capture_layers
        )
    }

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

    existing_test, baseline_path = load_existing_test_baseline(
        args.prior_output_dir
    )

    train_generation, train_generation_path = load_existing_train_generation(
        args.prior_output_dir
    )

    records = load_all_records(
        args
    )

    train_sids, test_sids = derive_train_test_ids(
        records,
        existing_test,
    )

    fit_sids, cal_sids = stratified_fit_cal_split(
        records,
        train_sids,
        args.cal_frac,
        args.seed,
    )

    print(
        "\n"
        + "=" * 145
    )

    print(
        "COSINE + CONFIDENCE SELECTOR"
    )

    print(
        "=" * 145
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"actuator_layers={actuator_layers}  [REUSED; no actuator layer search]"
    )

    print(
        f"selector_scan_layers={selector_layers}"
    )

    print(
        f"capture_layers={capture_layers}"
    )

    print(
        f"FIT={len(fit_sids)} CAL={len(cal_sids)} TEST={len(test_sids)}"
    )

    print(
        f"reused_test_baseline={baseline_path}"
    )

    print(
        f"reused_train_generation={train_generation_path}"
    )

    print(
        "=" * 145
    )

    model, processor = load_model(
        args
    )

    layers, layer_path = resolve_layers(
        model
    )

    if max(capture_layers) >= len(layers):
        raise RuntimeError(
            f"Requested layer {max(capture_layers)} "
            f"but model has only {len(layers)} blocks."
        )

    print(
        f"decoder={layer_path} | blocks={len(layers)}"
    )

    cache_path = (
        Path(args.cache_deltas)
        if args.cache_deltas
        else outdir
        / "late_real_gray_deltas.npz"
    )

    delta_cache = build_or_load_delta_cache(
        model,
        processor,
        layers,
        records,
        capture_layers,
        args,
        cache_path,
    )

    # -----------------------------------------------------------------
    # FIT templates for selector/policy calibration.
    # -----------------------------------------------------------------
    fit_templates_bank, fit_filter_used = fit_templates(
        delta_cache,
        layer_index,
        records,
        fit_sids,
        capture_layers,
        train_generation,
        args.template_filter,
    )

    # -----------------------------------------------------------------
    # Select cosine selector window on CAL, no generation needed.
    # -----------------------------------------------------------------
    (
        selector_window,
        selector_scan_rows,
        cal_selector_predictions,
    ) = evaluate_selector_windows(
        delta_cache,
        layer_index,
        fit_templates_bank,
        records,
        cal_sids,
        selector_layers,
        parse_int_list(
            args.selector_window_lengths
        ),
    )

    write_csv(
        outdir
        / "cal_selector_window_scan.csv",
        selector_scan_rows,
    )

    cal_selector_acc = safe_mean(
        int(
            cal_selector_predictions[sid]["pred"]
            == records[sid]["gt"]
        )
        for sid in cal_sids
    )

    print(
        f"\nSELECTED COSINE SELECTOR WINDOW={list(selector_window)} "
        f"| CAL selector_acc={cal_selector_acc:.4f}"
    )

    # -----------------------------------------------------------------
    # CAL baseline: small (about 33), generate once.
    # -----------------------------------------------------------------
    cal_baseline = generate_baseline_for_sids(
        model,
        processor,
        records,
        cal_sids,
        args,
        "CAL baseline",
    )

    # -----------------------------------------------------------------
    # Actual-generation confidence calibration.
    # -----------------------------------------------------------------
    coverages = parse_float_list(
        args.coverages
    )

    cal_policy_summaries = []
    cal_policy_details = []

    for edit_mode in [
        "add",
        "contrast",
    ]:
        for apply_mode in [
            "all",
            "conflict_only",
        ]:
            for coverage in coverages:
                (
                    selected_sids,
                    threshold,
                    eligible_n,
                ) = select_by_coverage(
                    cal_sids,
                    cal_selector_predictions,
                    cal_baseline,
                    coverage,
                    apply_mode,
                )

                condition = (
                    f"cal_{edit_mode}_{apply_mode}_cov{coverage:.2f}"
                )

                rows = run_steering_policy(
                    model,
                    processor,
                    layers,
                    fit_templates_bank,
                    actuator_layers,
                    records,
                    cal_sids,
                    cal_baseline,
                    cal_selector_predictions,
                    selected_sids,
                    args,
                    edit_mode,
                    condition,
                )

                cal_policy_details.extend(
                    rows
                )

                summary = summarize_policy(
                    rows,
                    condition,
                    coverage,
                    threshold,
                    eligible_n,
                )

                summary[
                    "edit_mode"
                ] = edit_mode

                summary[
                    "apply_mode"
                ] = apply_mode

                cal_policy_summaries.append(
                    summary
                )

                write_csv(
                    outdir
                    / "cal_policy_summary.csv",
                    cal_policy_summaries,
                )

                write_csv(
                    outdir
                    / "cal_policy_details.csv",
                    cal_policy_details,
                )

    best_policy = max(
        cal_policy_summaries,
        key=lambda row: (
            float(row["edit_acc"]),
            int(row["net"]),
            -int(row["C2W"]),
            -float(row["coverage_target"]),
        ),
    )

    selected_coverage = float(
        best_policy[
            "coverage_target"
        ]
    )

    selected_edit_mode = str(
        best_policy[
            "edit_mode"
        ]
    )

    selected_apply_mode = str(
        best_policy[
            "apply_mode"
        ]
    )

    print(
        "\nSELECTED CONFIDENCE POLICY ON CAL"
    )

    print(
        f"edit_mode={selected_edit_mode} | "
        f"apply_mode={selected_apply_mode} | "
        f"coverage={selected_coverage:.2f} | "
        f"CAL {float(best_policy['base_acc']):.4f}->"
        f"{float(best_policy['edit_acc']):.4f} "
        f"{float(best_policy['gain']):+.4f} | "
        f"W2C={int(best_policy['W2C'])} "
        f"C2W={int(best_policy['C2W'])}"
    )

    # -----------------------------------------------------------------
    # Refit templates on FULL original TRAIN from cached activations.
    # No model forward repeated.
    # -----------------------------------------------------------------
    final_templates_bank, final_filter_used = fit_templates(
        delta_cache,
        layer_index,
        records,
        train_sids,
        capture_layers,
        train_generation,
        args.template_filter,
    )

    # TEST cosine selector.
    test_selector_predictions = {}

    for sid in test_sids:
        pred, margin, scores = cosine_scores(
            delta_cache,
            layer_index,
            final_templates_bank,
            sid,
            selector_window,
        )

        test_selector_predictions[sid] = {
            "pred": pred,
            "margin": margin,
            "scores": scores,
        }

    test_selector_acc = safe_mean(
        int(
            test_selector_predictions[sid]["pred"]
            == records[sid]["gt"]
        )
        for sid in test_sids
    )

    selector_detail_rows = []

    for sid in test_sids:
        item = test_selector_predictions[
            sid
        ]

        selector_detail_rows.append({
            "sid": sid,
            "gt": records[sid]["gt"],
            "pred": item["pred"],
            "correct": int(
                item["pred"]
                == records[sid]["gt"]
            ),
            "confidence": item["margin"],
            **{
                f"score_{relation}":
                    item["scores"][relation]
                for relation in RELS
            },
        })

    write_csv(
        outdir
        / "test_cosine_selector.csv",
        selector_detail_rows,
    )

    # Reuse exact prior TEST baseline.
    test_baseline = prepare_test_baseline(
        existing_test,
        records,
        test_sids,
    )

    if test_baseline is None:
        test_baseline = generate_baseline_for_sids(
            model,
            processor,
            records,
            test_sids,
            args,
            "TEST baseline fallback",
        )

    test_base_acc = safe_mean(
        item["correct"]
        for item in test_baseline.values()
    )

    # -----------------------------------------------------------------
    # TEST: same selected policy, compare no confidence vs confidence.
    # Coverage, not absolute threshold, is transferred.
    # -----------------------------------------------------------------
    test_conditions = []

    for label, coverage in [
        (
            "no_conf",
            1.0,
        ),
        (
            "with_conf",
            selected_coverage,
        ),
    ]:
        (
            selected_sids,
            threshold,
            eligible_n,
        ) = select_by_coverage(
            test_sids,
            test_selector_predictions,
            test_baseline,
            coverage,
            selected_apply_mode,
        )

        condition = (
            f"test_cosine_{label}_"
            f"{selected_edit_mode}_{selected_apply_mode}"
        )

        rows = run_steering_policy(
            model,
            processor,
            layers,
            final_templates_bank,
            actuator_layers,
            records,
            test_sids,
            test_baseline,
            test_selector_predictions,
            selected_sids,
            args,
            selected_edit_mode,
            condition,
        )

        summary = summarize_policy(
            rows,
            condition,
            coverage,
            threshold,
            eligible_n,
        )

        test_conditions.append(
            summary
        )

        write_csv(
            outdir
            / f"{condition}_details.csv",
            rows,
        )

    write_csv(
        outdir
        / "test_summary.csv",
        test_conditions,
    )

    print(
        "\n"
        + "=" * 165
    )

    print(
        "FINAL TEST — COSINE SELECTOR + CONFIDENCE"
    )

    print(
        "=" * 165
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"decoder_blocks={len(layers)}"
    )

    print(
        f"actuator_layers={actuator_layers}"
    )

    print(
        f"selector_layers={list(selector_window)}"
    )

    print(
        f"template_filter_fit={fit_filter_used}"
    )

    print(
        f"template_filter_final={final_filter_used}"
    )

    print(
        f"baseline={test_base_acc:.4f}"
    )

    print(
        f"cosine_selector_acc={test_selector_acc:.4f}"
    )

    print(
        f"selected_policy="
        f"{selected_edit_mode}/{selected_apply_mode}/"
        f"coverage={selected_coverage:.2f}"
    )

    print()

    print(
        "condition                                      | "
        "acc base->edit gain | applied | W2C/wrong | C2W/correct | net"
    )

    for row in test_conditions:
        print(
            f"{str(row['condition']):46s} | "
            f"{float(row['base_acc']):.4f}->"
            f"{float(row['edit_acc']):.4f} "
            f"{float(row['gain']):+.4f} | "
            f"{int(row['applied'])}/{float(row['applied_rate']):.3f} | "
            f"{int(row['W2C'])}/{float(row['W2C_rate_wrong']):.3f} | "
            f"{int(row['C2W'])}/{float(row['C2W_rate_correct']):.3f} | "
            f"{int(row['net']):+d}"
        )

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "model_id":
                    args.model_id,
                "actuator_layers":
                    actuator_layers,
                "selector_scan_layers":
                    selector_layers,
                "selected_selector_layers":
                    list(
                        selector_window
                    ),
                "fit_n":
                    len(
                        fit_sids
                    ),
                "cal_n":
                    len(
                        cal_sids
                    ),
                "test_n":
                    len(
                        test_sids
                    ),
                "template_filter_fit":
                    fit_filter_used,
                "template_filter_final":
                    final_filter_used,
                "cal_selector_acc":
                    cal_selector_acc,
                "test_selector_acc":
                    test_selector_acc,
                "selected_edit_mode":
                    selected_edit_mode,
                "selected_apply_mode":
                    selected_apply_mode,
                "selected_confidence_coverage":
                    selected_coverage,
                "test_baseline_acc":
                    test_base_acc,
                "reused_test_baseline":
                    str(
                        baseline_path
                    ),
                "reused_train_generation":
                    str(
                        train_generation_path
                    ),
                "note":
                    (
                        "Known actuator window is reused. Only late Real-Gray "
                        "last-token deltas needed for the new cosine selector "
                        "are recomputed; all-layer causal search is not repeated."
                    ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
