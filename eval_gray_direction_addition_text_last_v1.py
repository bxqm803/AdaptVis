#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_gray_direction_addition_text_last_v1.py

Question
--------
Can the previously learned spatial Direction vector provide enough information
for a GRAY-image run to recover the correct spatial answer?

This is a SUFFICIENCY-style experiment rather than a removal/necessity test.

For each layer l, TRAIN residual Direction vectors are centered and averaged
by relation to obtain four raw relation prototypes:

    mu_left, mu_right, mu_above, mu_below

For a TEST sample with ground-truth relation g, we use:

    d_l = scale * mu_g,l

and add d_l to a GRAY-image forward pass.

Two sites
---------
1) text site (default = subject/reference object text-token pair)

    h_sub += 0.5 * d_l
    h_ref -= 0.5 * d_l

so the object-pair difference changes by +d_l while its pair mean is fixed.

Optional --text-target all_text instead adds +d_l to every non-visual prompt
text token. The pair target is the default because Direction was originally
defined as subject-minus-reference.

2) last-token site

    h_last += d_l

The edit is applied at decoder BLOCK OUTPUT and only on the prompt-prefill
forward pass. Autoregressive decode steps are not repeatedly edited.

Default layer windows
---------------------
    text: 10-24
    last: 25-27

The model's actual number of layers is checked. You can choose arbitrary layers:
    --text-layers 10-24
    --last-layers 25-27
    --text-layers 14,16,19
    --last-layers 26-27

Runs
----
By default the script runs:
    * every text layer individually
    * all text layers simultaneously
    * every last-token layer individually
    * all last-token layers simultaneously
    * joint multi-layer: text-window + last-window simultaneously

Cohort
------
Primary cohort is:
    Real generation = correct
    Gray generation = wrong

This directly asks whether adding Direction to Gray rescues behavior.

Primary metric:
    rescue = Gray wrong -> edited Gray correct

Data
----
No custom CSV is needed. Directly uses:
    prompts/COCO_QA_two_obj_with_answer_four_options.jsonl
    data/coco_qa_two_obj.json
    data/val2017/{image_id:012d}.jpg

Direction bundle:
    <direction-dir>/vectors.npz
    <direction-dir>/sample_split_and_generation.csv

Recommended quick run
---------------------
CUDA_VISIBLE_DEVICES=0 python eval_gray_direction_addition_text_last_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --direction-key residual \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --text-layers 10-24 \
  --last-layers 25-27 \
  --text-target pair \
  --scale 1.0 \
  --output-dir output/qwen7b_gray_direction_add_text10_24_last25_27_v1 \
  --overwrite

To test addition to ALL text tokens instead:
    --text-target all_text
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


# =============================================================================
# CLI / basic utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--direction-dir", required=True)
    p.add_argument("--direction-key", default="residual")

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

    p.add_argument("--text-layers", default="10-24")
    p.add_argument("--last-layers", default="25-27")

    p.add_argument(
        "--text-target",
        default="pair",
        choices=["pair", "all_text"],
        help=(
            "pair: +d/2 to subject and -d/2 to reference; "
            "all_text: +d to every non-visual prompt token."
        ),
    )

    p.add_argument(
        "--train-controls",
        default="correct",
        choices=["correct", "all"],
        help="Which TRAIN samples define the four relation prototypes.",
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument(
        "--cohort",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "all_test"],
    )

    p.add_argument(
        "--no-single",
        action="store_true",
        help="Skip single-layer scans.",
    )
    p.add_argument(
        "--no-multi",
        action="store_true",
        help="Skip text-multi and last-multi runs.",
    )
    p.add_argument(
        "--no-joint",
        action="store_true",
        help="Skip joint text-window + last-window multi-layer run.",
    )

    p.add_argument("--max-new-tokens", type=int, default=8)
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

    fields, seen = [], set()
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
    values = []
    for piece in str(spec).split(","):
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
    bad = [x for x in values if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Invalid layers={bad}; model valid range is 0..{n_layers-1}")
    return values


def torch_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


# =============================================================================
# Direction bundle and relation prototypes
# =============================================================================

def load_direction_bundle(direction_dir, key):
    root = Path(direction_dir)
    vectors_path = root / "vectors.npz"
    meta_path = root / "sample_split_and_generation.csv"

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with np.load(vectors_path, allow_pickle=True) as z:
        if key not in z.files:
            raise KeyError(f"{key!r} not found in vectors.npz; keys={z.files}")
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray(
            [normalize_relation(x) for x in z["relation"]],
            dtype=object,
        )
        arr = np.asarray(z[key], dtype=np.float32)

    n = len(sids)
    if arr.ndim != 3:
        raise ValueError(f"Direction array must be [N,L,D] or [L,N,D], got {arr.shape}")

    if arr.shape[0] == n:
        vectors = arr
    elif arr.shape[1] == n:
        vectors = np.transpose(arr, (1, 0, 2))
    else:
        raise ValueError(f"Could not align N={n} with Direction shape={arr.shape}")

    split, group = {}, {}
    for row in read_csv(meta_path):
        sid = int(row["sample_index"])
        split[sid] = str(row.get("split", "")).strip().lower()
        group[sid] = str(row.get("generation_group", "")).strip().lower()

    return {
        "sids": sids,
        "labels": labels,
        "vectors": vectors,
        "gt": {int(sid): str(labels[i]) for i, sid in enumerate(sids.tolist())},
        "split": split,
        "group": group,
    }


def fit_relation_prototypes(bundle, needed_layers, train_controls):
    train_idx = []

    for i, sid in enumerate(bundle["sids"].tolist()):
        sid = int(sid)
        if bundle["split"].get(sid) != "train":
            continue
        if bundle["labels"][i] not in RELSET:
            continue
        if train_controls == "correct" and bundle["group"].get(sid) != "correct":
            continue
        train_idx.append(i)

    if not train_idx:
        raise RuntimeError("No TRAIN samples available for Direction prototypes.")

    prototypes = {}
    summary = []

    for layer in needed_layers:
        X = bundle["vectors"][train_idx, layer, :].astype(np.float64)
        Y = bundle["labels"][train_idx]

        # Match the existing Direction codebook convention:
        # global TRAIN center first, then mean centered vector per relation.
        center = X.mean(axis=0)
        Xc = X - center[None, :]

        mus = {}
        for rel in RELS:
            mask = Y == rel
            if not np.any(mask):
                raise RuntimeError(f"L{layer}: no TRAIN samples for relation={rel}")
            mus[rel] = Xc[mask].mean(axis=0).astype(np.float32)

        prototypes[layer] = {
            "center": center.astype(np.float32),
            "prototype": mus,
        }

        summary.append({
            "layer": layer,
            "n_train": len(train_idx),
            "norm_left": float(np.linalg.norm(mus["left"])),
            "norm_right": float(np.linalg.norm(mus["right"])),
            "norm_above": float(np.linalg.norm(mus["above"])),
            "norm_below": float(np.linalg.norm(mus["below"])),
        })

    return prototypes, summary


# =============================================================================
# COCO-two data
# =============================================================================

def parse_subject_reference(question):
    q = str(question)
    patterns = [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for pat in patterns:
        m = re.search(pat, q, flags=re.I | re.S)
        if m:
            subject = re.sub(r"\s+", " ", m.group(1)).strip()
            reference = re.sub(r"\s+", " ", m.group(2)).strip()
            return subject, reference

    return None, None


def load_records(prompt_jsonl, annotation_json, data_root, bundle):
    with open(annotation_json, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    prompts = []
    with open(prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    records = {}

    for row_no, row in enumerate(prompts):
        sid = int(row.get("id", row_no))
        if sid < 0 or sid >= len(annotations):
            continue

        subject, reference = parse_subject_reference(row.get("question", ""))
        if subject is None:
            continue

        image_id = int(annotations[sid][0])
        image_path = Path(data_root) / "val2017" / f"{image_id:012d}.jpg"

        records[sid] = {
            "sid": sid,
            "subject": subject,
            "reference": reference,
            "image_path": str(image_path),
            "gt": bundle["gt"].get(sid, ""),
            "split": bundle["split"].get(sid, ""),
        }

    overlap = len(set(records) & set(map(int, bundle["sids"].tolist())))
    existing = sum(Path(r["image_path"]).exists() for r in records.values())

    print(
        f"[records] prompts={len(prompts)} records={len(records)} "
        f"overlap={overlap} existing_images={existing}/{len(records)}"
    )

    if overlap == 0:
        raise RuntimeError("No overlap between COCO-two records and Direction sample ids.")

    return records


# =============================================================================
# Model / processor / generation
# =============================================================================

def get_attr_path(obj, path):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


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
            layers = get_attr_path(model, path)
            if len(layers) > 0:
                return layers, path
        except Exception:
            pass

    raise RuntimeError("Could not resolve language decoder layers.")


def load_model(model_id, dtype, device, attn_impl):
    class_names = [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]

    model_cls = next(
        (getattr(transformers, name) for name in class_names if hasattr(transformers, name)),
        None,
    )

    if model_cls is None:
        raise RuntimeError("No supported multimodal generation model class found.")

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }

    if attn_impl != "none":
        kwargs["attn_implementation"] = attn_impl

    print(f"[model] class={model_cls.__name__} id={model_id} device={device}")

    try:
        model = model_cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        model = model_cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)

    model.eval()

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


def make_gray_image(real_image, gray_value):
    value = int(max(0, min(255, gray_value)))
    return Image.new("RGB", real_image.size, (value, value, value))


def build_batch(processor, image, question, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
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
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(f"Processor failed: {last_error}")

    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    return batch


def parse_pred(text):
    s = str(text).lower()
    hits = []

    for rel, pat in [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))

    return sorted(hits)[0][1] if hits else None


def generate(model, processor, batch, max_new_tokens):
    input_len = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    text = processor.tokenizer.decode(
        output_ids[0, input_len:],
        skip_special_tokens=True,
    ).strip()

    pred = parse_pred(text)
    del output_ids
    return text, pred


# =============================================================================
# Token positions
# =============================================================================

def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]):
    if not pattern:
        return []

    hits = []
    m = len(pattern)

    for i in range(len(sequence) - m + 1):
        if list(sequence[i:i + m]) == list(pattern):
            hits.append(i)

    return hits


def phrase_spans(tokenizer, full_ids, phrase):
    variants = [
        phrase,
        " " + phrase,
        phrase.strip(),
        " " + phrase.strip(),
    ]

    spans, seen = [], set()

    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        for start in find_subsequence(full_ids, ids):
            span = tuple(range(start, start + len(ids)))
            if span not in seen:
                seen.add(span)
                spans.append(list(span))

    return spans


def locate_object_spans(tokenizer, full_ids, subject, reference):
    subject_spans = phrase_spans(tokenizer, full_ids, subject)
    reference_spans = phrase_spans(tokenizer, full_ids, reference)

    if not subject_spans or not reference_spans:
        return None, None

    best = None

    for ss in subject_spans:
        for rr in reference_spans:
            if set(ss) & set(rr):
                continue

            distance = abs(float(np.mean(ss)) - float(np.mean(rr)))
            score = (distance, -min(ss[0], rr[0]))

            if best is None or score < best[0]:
                best = (score, ss, rr)

    if best is None:
        return None, None

    return best[1], best[2]


def infer_last_prompt_position(batch):
    if "attention_mask" in batch:
        mask = batch["attention_mask"][0]
        nz = torch.nonzero(mask, as_tuple=False).flatten()
        if len(nz):
            return int(nz[-1].item())

    return int(batch["input_ids"].shape[1] - 1)


def get_visual_special_ids(model, processor):
    ids = set()

    # Qwen-VL config commonly exposes these ids.
    for obj in [getattr(model, "config", None), getattr(processor, "tokenizer", None)]:
        if obj is None:
            continue
        for name in [
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]:
            value = getattr(obj, name, None)
            if isinstance(value, int):
                ids.add(int(value))

    # Tokenizer string lookup fallback.
    tok = processor.tokenizer
    for token in ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]:
        try:
            tid = tok.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                ids.add(int(tid))
        except Exception:
            pass

    return ids


def infer_all_text_positions(batch, visual_special_ids):
    ids = batch["input_ids"][0].detach().cpu().tolist()

    if "attention_mask" in batch:
        attn = batch["attention_mask"][0].detach().cpu().tolist()
    else:
        attn = [1] * len(ids)

    positions = []

    for pos, (tid, keep) in enumerate(zip(ids, attn)):
        if not keep:
            continue
        if int(tid) in visual_special_ids:
            continue
        positions.append(pos)

    return positions


# =============================================================================
# Block-output hooks
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

    raise RuntimeError(f"Could not extract hidden tensor from output type={type(output)}")


def replace_hidden(output, descriptor, hidden):
    kind, index = descriptor

    if kind == "tensor":
        return hidden

    values = list(output)
    values[index] = hidden

    return tuple(values) if kind == "tuple" else values


class DirectionAdditionHooks:
    """Apply one or more GT Direction additions on prompt prefill only."""

    def __init__(
        self,
        decoder_layers,
        prototypes,
        gt_relation,
        text_layers,
        last_layers,
        text_target,
        subject_span,
        reference_span,
        text_positions,
        last_position,
        scale,
    ):
        self.handles = []
        self.applied = {}
        self.prototypes = prototypes
        self.gt = gt_relation
        self.text_layers = set(text_layers)
        self.last_layers = set(last_layers)
        self.text_target = text_target
        self.subject_span = list(subject_span)
        self.reference_span = list(reference_span)
        self.text_positions = list(text_positions)
        self.last_position = int(last_position)
        self.scale = float(scale)
        self.stats = {}

        selected = sorted(self.text_layers | self.last_layers)

        for layer in selected:
            self.applied[layer] = False
            self.handles.append(
                decoder_layers[layer].register_forward_hook(
                    self._make_hook(layer)
                )
            )

    def _make_hook(self, layer):
        direction_np = (
            self.scale
            * self.prototypes[layer]["prototype"][self.gt]
        ).astype(np.float32)

        def hook(_module, _inputs, output):
            if self.applied[layer]:
                return output

            hidden, descriptor = extract_hidden(output)

            if hidden.ndim != 3:
                return output

            seq_len = int(hidden.shape[1])

            # Only prompt prefill has the original prompt positions available.
            required = [self.last_position]
            if layer in self.text_layers:
                if self.text_target == "pair":
                    required += self.subject_span + self.reference_span
                else:
                    required += self.text_positions

            if required and max(required) >= seq_len:
                return output

            d = torch.as_tensor(
                direction_np,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            if d.shape[-1] != hidden.shape[-1]:
                raise RuntimeError(
                    f"L{layer}: Direction dim={d.shape[-1]} != hidden dim={hidden.shape[-1]}"
                )

            edited = hidden.clone()
            edited_sites = []

            if layer in self.text_layers:
                if self.text_target == "pair":
                    # Add the relation difference while preserving pair mean.
                    half = 0.5 * d
                    edited[:, self.subject_span, :] = (
                        edited[:, self.subject_span, :] + half[None, None, :]
                    )
                    edited[:, self.reference_span, :] = (
                        edited[:, self.reference_span, :] - half[None, None, :]
                    )
                    edited_sites.append("pair_text")

                elif self.text_target == "all_text":
                    edited[:, self.text_positions, :] = (
                        edited[:, self.text_positions, :] + d[None, None, :]
                    )
                    edited_sites.append("all_text")

                else:
                    raise ValueError(self.text_target)

            if layer in self.last_layers:
                edited[:, self.last_position, :] = (
                    edited[:, self.last_position, :] + d[None, :]
                )
                edited_sites.append("last")

            self.applied[layer] = True
            self.stats[layer] = {
                "direction_norm": float(torch.linalg.vector_norm(d.float()).detach().cpu()),
                "sites": "+".join(edited_sites),
            }

            return replace_hidden(output, descriptor, edited)

        return hook

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


# =============================================================================
# Prepare one real/gray sample
# =============================================================================

def prepare_batch_for_image(
    processor,
    image,
    rec,
    prompt_template,
    device,
    visual_special_ids,
):
    question = prompt_template.format(
        subject=rec["subject"],
        reference=rec["reference"],
    )

    batch = build_batch(
        processor,
        image,
        question,
        device,
    )

    ids = batch["input_ids"][0].detach().cpu().tolist()

    subject_span, reference_span = locate_object_spans(
        processor.tokenizer,
        ids,
        rec["subject"],
        rec["reference"],
    )

    text_positions = infer_all_text_positions(
        batch,
        visual_special_ids,
    )

    last_position = infer_last_prompt_position(batch)

    return batch, subject_span, reference_span, text_positions, last_position


# =============================================================================
# Baselines
# =============================================================================

def run_real_gray_baselines(
    model,
    processor,
    records,
    test_sids,
    args,
    device,
    visual_special_ids,
    outdir,
):
    rows = []

    for sid in tqdm(test_sids, desc="Real/Gray baseline"):
        rec = records[sid]
        real_image = None
        gray_image = None

        try:
            real_image = Image.open(rec["image_path"]).convert("RGB")
            gray_image = make_gray_image(real_image, args.gray_value)

            real_batch, rss, rrs, _, _ = prepare_batch_for_image(
                processor,
                real_image,
                rec,
                args.prompt_template,
                device,
                visual_special_ids,
            )

            gray_batch, gss, grs, _, _ = prepare_batch_for_image(
                processor,
                gray_image,
                rec,
                args.prompt_template,
                device,
                visual_special_ids,
            )

            span_ok = int(
                rss is not None
                and rrs is not None
                and gss is not None
                and grs is not None
            )

            if not span_ok:
                rows.append({
                    "sid": sid,
                    "gt": rec["gt"],
                    "span_ok": 0,
                    "real_pred": "",
                    "real_correct": 0,
                    "gray_pred": "",
                    "gray_correct": 0,
                    "error": "object token span not found",
                })
                continue

            real_text, real_pred = generate(
                model,
                processor,
                real_batch,
                args.max_new_tokens,
            )

            gray_text, gray_pred = generate(
                model,
                processor,
                gray_batch,
                args.max_new_tokens,
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "span_ok": 1,
                "real_pred": real_pred or "",
                "real_correct": int(real_pred == rec["gt"]),
                "gray_pred": gray_pred or "",
                "gray_correct": int(gray_pred == rec["gt"]),
                "real_text": real_text,
                "gray_text": gray_text,
                "error": "",
            })

            del real_batch, gray_batch

        except Exception as e:
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "span_ok": 0,
                "real_pred": "",
                "real_correct": 0,
                "gray_pred": "",
                "gray_correct": 0,
                "error": f"{type(e).__name__}: {e}",
            })
            tqdm.write(f"[baseline ERROR sid={sid}] {type(e).__name__}: {e}")

        finally:
            if real_image is not None:
                real_image.close()
            if gray_image is not None:
                gray_image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(outdir / "baseline_real_gray.csv", rows)
    return rows


# =============================================================================
# Direction addition evaluation
# =============================================================================

def select_cohort(baseline_rows, cohort_name, max_cohort_samples, seed):
    if cohort_name == "real_correct_gray_wrong":
        rows = [
            r for r in baseline_rows
            if int(r.get("span_ok", 0)) == 1
            and int(r.get("real_correct", 0)) == 1
            and int(r.get("gray_correct", 0)) == 0
        ]
    elif cohort_name == "all_test":
        rows = [
            r for r in baseline_rows
            if int(r.get("span_ok", 0)) == 1
        ]
    else:
        raise ValueError(cohort_name)

    if max_cohort_samples is not None and len(rows) > max_cohort_samples:
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[:max_cohort_samples]

    return sorted(rows, key=lambda x: int(x["sid"]))


def run_addition_condition(
    *,
    model,
    processor,
    decoder_layers,
    prototypes,
    records,
    cohort_rows,
    text_layers,
    last_layers,
    label,
    args,
    device,
    visual_special_ids,
):
    results = []

    for base in tqdm(cohort_rows, desc=label):
        sid = int(base["sid"])
        rec = records[sid]
        real_image = None
        gray_image = None

        try:
            real_image = Image.open(rec["image_path"]).convert("RGB")
            gray_image = make_gray_image(real_image, args.gray_value)

            gray_batch, subject_span, reference_span, text_positions, last_position = (
                prepare_batch_for_image(
                    processor,
                    gray_image,
                    rec,
                    args.prompt_template,
                    device,
                    visual_special_ids,
                )
            )

            if subject_span is None or reference_span is None:
                continue

            with DirectionAdditionHooks(
                decoder_layers=decoder_layers,
                prototypes=prototypes,
                gt_relation=rec["gt"],
                text_layers=text_layers,
                last_layers=last_layers,
                text_target=args.text_target,
                subject_span=subject_span,
                reference_span=reference_span,
                text_positions=text_positions,
                last_position=last_position,
                scale=args.scale,
            ) as hooks:

                edit_text, edit_pred = generate(
                    model,
                    processor,
                    gray_batch,
                    args.max_new_tokens,
                )

                stats = dict(hooks.stats)

            direction_norms = [
                stats[layer]["direction_norm"]
                for layer in sorted(set(text_layers) | set(last_layers))
                if layer in stats
            ]

            gray_correct = int(base["gray_correct"])
            edit_correct = int(edit_pred == rec["gt"])

            results.append({
                "sid": sid,
                "condition": label,
                "text_layers": ",".join(map(str, text_layers)),
                "last_layers": ",".join(map(str, last_layers)),
                "text_target": args.text_target,
                "scale": args.scale,
                "gt": rec["gt"],
                "real_pred": base["real_pred"],
                "gray_pred": base["gray_pred"],
                "edit_pred": edit_pred or "",
                "gray_correct": gray_correct,
                "edit_correct": edit_correct,
                "rescue": int(gray_correct == 0 and edit_correct == 1),
                "damage": int(gray_correct == 1 and edit_correct == 0),
                "changed": int((edit_pred or "") != str(base["gray_pred"])),
                "mean_direction_norm": safe_mean(direction_norms),
                "generation_text": edit_text,
            })

            del gray_batch

        except Exception as e:
            tqdm.write(
                f"[addition ERROR sid={sid} condition={label}] "
                f"{type(e).__name__}: {e}"
            )

        finally:
            if real_image is not None:
                real_image.close()
            if gray_image is not None:
                gray_image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def summarize_condition(rows):
    if not rows:
        return {
            "N": 0,
            "gray_acc": float("nan"),
            "edit_acc": float("nan"),
            "gain": float("nan"),
            "rescue": 0,
            "rescue_rate_gray_wrong": float("nan"),
            "damage": 0,
            "damage_rate_gray_correct": float("nan"),
            "changed_rate": float("nan"),
        }

    gray_wrong = sum(1 - int(r["gray_correct"]) for r in rows)
    gray_correct = len(rows) - gray_wrong
    rescue = sum(int(r["rescue"]) for r in rows)
    damage = sum(int(r["damage"]) for r in rows)

    gray_acc = safe_mean(r["gray_correct"] for r in rows)
    edit_acc = safe_mean(r["edit_correct"] for r in rows)

    return {
        "N": len(rows),
        "gray_acc": gray_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - gray_acc,
        "rescue": rescue,
        "rescue_rate_gray_wrong": rescue / gray_wrong if gray_wrong else float("nan"),
        "damage": damage,
        "damage_rate_gray_correct": damage / gray_correct if gray_correct else float("nan"),
        "changed_rate": safe_mean(r["changed"] for r in rows),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_direction_bundle(
        args.direction_dir,
        args.direction_key,
    )

    print(
        f"[direction] key={args.direction_key!r}, "
        f"shape={bundle['vectors'].shape}"
    )

    records = load_records(
        args.prompt_jsonl,
        args.annotation_json,
        args.data_root,
        bundle,
    )

    model, processor = load_model(
        args.model_id,
        torch_dtype(args.dtype),
        args.device,
        args.attn_impl,
    )

    decoder, decoder_path = resolve_decoder_layers(model)

    text_layers = parse_layers(args.text_layers, len(decoder))
    last_layers = parse_layers(args.last_layers, len(decoder))
    needed_layers = sorted(set(text_layers) | set(last_layers))

    print(f"[decoder] {decoder_path}")
    print(f"[text layers] {text_layers} | target={args.text_target}")
    print(f"[last layers] {last_layers}")
    print(f"[scale] {args.scale}")

    if max(needed_layers) >= bundle["vectors"].shape[1]:
        raise RuntimeError(
            f"Direction file only has {bundle['vectors'].shape[1]} layers, "
            f"but requested layer {max(needed_layers)}"
        )

    prototypes, prototype_summary = fit_relation_prototypes(
        bundle,
        needed_layers,
        args.train_controls,
    )

    write_csv(
        outdir / "direction_prototype_summary.csv",
        prototype_summary,
    )

    visual_special_ids = get_visual_special_ids(model, processor)
    print(f"[visual special token ids] {sorted(visual_special_ids)}")

    test_sids = sorted([
        int(sid)
        for sid in bundle["sids"].tolist()
        if bundle["split"].get(int(sid)) == "test"
        and int(sid) in records
    ])

    if args.max_test_samples is not None and len(test_sids) > args.max_test_samples:
        rng = random.Random(args.seed)
        rng.shuffle(test_sids)
        test_sids = sorted(test_sids[:args.max_test_samples])

    device = torch.device(args.device)

    baseline_rows = run_real_gray_baselines(
        model=model,
        processor=processor,
        records=records,
        test_sids=test_sids,
        args=args,
        device=device,
        visual_special_ids=visual_special_ids,
        outdir=outdir,
    )

    valid = [r for r in baseline_rows if int(r.get("span_ok", 0)) == 1]
    real_acc = safe_mean(r["real_correct"] for r in valid)
    gray_acc = safe_mean(r["gray_correct"] for r in valid)

    cohort_rows = select_cohort(
        baseline_rows,
        args.cohort,
        args.max_cohort_samples,
        args.seed,
    )

    print("\n" + "=" * 140)
    print(
        f"BASELINES | valid N={len(valid)} | Real acc={real_acc:.4f} | "
        f"Gray acc={gray_acc:.4f}"
    )
    print(
        f"COHORT={args.cohort} | N={len(cohort_rows)} | "
        f"primary metric = Gray->Correct rescue"
    )
    print("=" * 140)

    all_details = []
    summary_rows = []

    def run_and_store(label, tlayers, llayers, family, layer_value=""):
        rows = run_addition_condition(
            model=model,
            processor=processor,
            decoder_layers=decoder,
            prototypes=prototypes,
            records=records,
            cohort_rows=cohort_rows,
            text_layers=tlayers,
            last_layers=llayers,
            label=label,
            args=args,
            device=device,
            visual_special_ids=visual_special_ids,
        )

        for row in rows:
            row["family"] = family
            row["layer"] = layer_value

        all_details.extend(rows)

        s = summarize_condition(rows)
        summary_rows.append({
            "condition": label,
            "family": family,
            "layer": layer_value,
            "text_layers": ",".join(map(str, tlayers)),
            "last_layers": ",".join(map(str, llayers)),
            "text_target": args.text_target,
            "scale": args.scale,
            **s,
        })

        write_csv(outdir / "intervention_details.csv", all_details)
        write_csv(outdir / "summary.csv", summary_rows)

    # ------------------------------------------------------------------
    # Single-layer scans
    # ------------------------------------------------------------------
    if not args.no_single:
        for layer in text_layers:
            run_and_store(
                label=f"text_single_L{layer}",
                tlayers=[layer],
                llayers=[],
                family="text_single",
                layer_value=layer,
            )

        for layer in last_layers:
            run_and_store(
                label=f"last_single_L{layer}",
                tlayers=[],
                llayers=[layer],
                family="last_single",
                layer_value=layer,
            )

    # ------------------------------------------------------------------
    # Multi-layer windows
    # ------------------------------------------------------------------
    if not args.no_multi:
        if text_layers:
            run_and_store(
                label="text_multi",
                tlayers=text_layers,
                llayers=[],
                family="text_multi",
            )

        if last_layers:
            run_and_store(
                label="last_multi",
                tlayers=[],
                llayers=last_layers,
                family="last_multi",
            )

    # ------------------------------------------------------------------
    # Joint multi: text early/mid + last late
    # ------------------------------------------------------------------
    if not args.no_joint and text_layers and last_layers:
        run_and_store(
            label="joint_text_plus_last_multi",
            tlayers=text_layers,
            llayers=last_layers,
            family="joint_multi",
        )

    print("\n" + "=" * 165)
    print("GRAY + GT DIRECTION ADDITION — ACTUAL model.generate()")
    print("=" * 165)
    print(
        "condition                     | N | GrayAcc -> EditAcc gain | "
        "rescue/rate | damage/rate | changed"
    )

    for row in summary_rows:
        print(
            f"{str(row['condition']):29s} | "
            f"{int(row['N']):3d} | "
            f"{float(row['gray_acc']):.4f}->{float(row['edit_acc']):.4f} "
            f"{float(row['gain']):+.4f} | "
            f"{int(row['rescue'])}/{float(row['rescue_rate_gray_wrong']):.3f} | "
            f"{int(row['damage'])}/{float(row['damage_rate_gray_correct']):.3f} | "
            f"{float(row['changed_rate']):.3f}"
        )

    summary_json = {
        "experiment": "Gray + GT Direction prototype sufficiency",
        "direction_key": args.direction_key,
        "prototype": "mean centered TRAIN residual Direction vector per relation",
        "scale": args.scale,
        "gray_value": args.gray_value,
        "text_target": args.text_target,
        "text_layers": text_layers,
        "last_layers": last_layers,
        "text_pair_edit": (
            "subject += 0.5*d, reference -= 0.5*d"
            if args.text_target == "pair"
            else "all non-visual prompt tokens += d"
        ),
        "last_edit": "last prompt token += d",
        "site": "decoder block OUTPUT residual stream",
        "actual_generate": True,
        "real_baseline_acc": real_acc,
        "gray_baseline_acc": gray_acc,
        "cohort": args.cohort,
        "cohort_n": len(cohort_rows),
    }

    (outdir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    for path in [
        outdir / "direction_prototype_summary.csv",
        outdir / "baseline_real_gray.csv",
        outdir / "intervention_details.csv",
        outdir / "summary.csv",
        outdir / "summary.json",
    ]:
        if path.exists():
            print(" ", path)


if __name__ == "__main__":
    main()
