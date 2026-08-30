
# -*- coding: utf-8 -*-

"""
Standalone version: no import from previous experiment scripts.

Learn relation-specific late last-token causal directions from TRAIN Real-vs-Gray
differences, then steer REAL-image TEST generation.

Non-oracle selector:
    middle-layer residual Direction consensus (default L14-20)

Oracle selector:
    GT relation, diagnostic upper bound only

Recommended:
CUDA_VISIBLE_DEVICES=0 python eval_real_last_causal_direction_steering_v2_standalone.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --direction-key residual \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --guide-layers 14-20 \
  --last-layers 25-27 \
  --selectors guide,oracle \
  --apply-modes conflict_only,all \
  --edit-modes add,contrast \
  --windows single,multi \
  --scale 1.0 \
  --output-dir output/qwen7b_real_last_causal_steering_v2 \
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
from typing import Sequence

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
# CLI / utils
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    p.add_argument("--guide-layers", default="14-20")
    p.add_argument("--last-layers", default="25-27")

    p.add_argument(
        "--guide-train-controls",
        default="correct",
        choices=["correct", "all"],
    )
    p.add_argument(
        "--template-train-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )

    p.add_argument(
        "--selectors",
        default="guide,oracle",
        help="guide, oracle, or guide,oracle",
    )
    p.add_argument(
        "--apply-modes",
        default="conflict_only,all",
        help="conflict_only, all, or both",
    )
    p.add_argument(
        "--edit-modes",
        default="add,contrast",
        help="add, contrast, or both",
    )
    p.add_argument(
        "--windows",
        default="single,multi",
        help="single, multi, or both",
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)

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
    out = []

    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue

        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))

    out = sorted(set(out))

    bad = [l for l in out if l < 0 or l >= n_layers]
    if bad:
        raise ValueError(
            f"Invalid layers={bad}; valid range=0..{n_layers-1}"
        )

    return out


def parse_choices(spec, allowed):
    out = [
        x.strip()
        for x in str(spec).split(",")
        if x.strip()
    ]

    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(
            f"Invalid values={bad}; allowed={sorted(allowed)}"
        )

    return out


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


# =============================================================================
# Split / data
# =============================================================================

def load_direction_bundle(direction_dir, direction_key):
    root = Path(direction_dir)

    vectors_path = root / "vectors.npz"
    split_path = root / "sample_split_and_generation.csv"

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with np.load(vectors_path, allow_pickle=True) as z:
        if direction_key not in z.files:
            raise KeyError(
                f"{direction_key!r} not found in {vectors_path}; keys={z.files}"
            )

        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray(
            [normalize_relation(x) for x in z["relation"]],
            dtype=object,
        )
        arr = np.asarray(z[direction_key], dtype=np.float32)

    n = len(sids)

    if arr.ndim != 3:
        raise RuntimeError(
            f"Direction array must be 3D, got {arr.shape}"
        )

    if arr.shape[0] == n:
        vectors = arr
    elif arr.shape[1] == n:
        vectors = np.transpose(arr, (1, 0, 2))
    else:
        raise RuntimeError(
            f"Cannot align N={n} with Direction shape={arr.shape}"
        )

    split = {}
    cached_group = {}

    for row in read_csv(split_path):
        sid = int(row["sample_index"])
        split[sid] = str(
            row.get("split", "")
        ).strip().lower()
        cached_group[sid] = str(
            row.get("generation_group", "")
        ).strip().lower()

    return {
        "sids": sids,
        "labels": labels,
        "vectors": vectors,
        "split": split,
        "cached_group": cached_group,
        "sid_to_index": {
            int(sid): i
            for i, sid in enumerate(sids.tolist())
        },
    }


def parse_subject_reference(question):
    q = str(question)

    patterns = [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for pat in patterns:
        m = re.search(
            pat,
            q,
            flags=re.I | re.S,
        )
        if m:
            subject = re.sub(
                r"\s+",
                " ",
                m.group(1),
            ).strip()
            reference = re.sub(
                r"\s+",
                " ",
                m.group(2),
            ).strip()
            return subject, reference

    return None, None


def load_records(
    prompt_jsonl,
    annotation_json,
    data_root,
    direction_bundle,
):
    with open(
        annotation_json,
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    prompt_rows = []

    with open(
        prompt_jsonl,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if line:
                prompt_rows.append(json.loads(line))

    records = {}

    for row_no, row in enumerate(prompt_rows):
        sid = int(row.get("id", row_no))

        if sid not in direction_bundle["sid_to_index"]:
            continue

        if sid < 0 or sid >= len(annotations):
            continue

        subject, reference = parse_subject_reference(
            row.get("question", "")
        )

        if subject is None or reference is None:
            continue

        gt = normalize_relation(
            row.get("answer", "")
        )

        if gt not in RELSET:
            # fallback to labels stored with Direction data
            idx = direction_bundle["sid_to_index"][sid]
            gt = str(direction_bundle["labels"][idx])

        if gt not in RELSET:
            continue

        ann = annotations[sid]
        image_id = int(ann[0])

        image_path = (
            Path(data_root)
            / "val2017"
            / f"{image_id:012d}.jpg"
        )

        records[sid] = {
            "sid": sid,
            "subject": subject,
            "reference": reference,
            "gt": gt,
            "image_path": str(image_path),
            "split": direction_bundle["split"].get(sid, ""),
        }

    existing = sum(
        Path(r["image_path"]).exists()
        for r in records.values()
    )

    print(
        f"[records] N={len(records)} | "
        f"existing_images={existing}/{len(records)}"
    )

    if not records:
        raise RuntimeError("No records loaded.")

    return records


# =============================================================================
# Model / generation
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

    raise RuntimeError("Could not resolve decoder layers.")


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
            getattr(transformers, name)
            for name in class_names
            if hasattr(transformers, name)
        ),
        None,
    )

    if model_cls is None:
        raise RuntimeError(
            "No supported multimodal generation model class."
        )

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }

    if attn_impl != "none":
        kwargs["attn_implementation"] = attn_impl

    print(
        f"[model] {model_cls.__name__} | {model_id} | {device}"
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


def make_gray_image(real_image, value):
    value = int(max(0, min(255, value)))

    return Image.new(
        "RGB",
        real_image.size,
        (value, value, value),
    )


def build_batch(
    processor,
    image,
    rec,
    prompt_template,
    device,
):
    question = prompt_template.format(
        subject=rec["subject"],
        reference=rec["reference"],
    )

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

    batch = {
        k: (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in batch.items()
    }

    return batch


def infer_last_prompt_position(batch):
    if "attention_mask" in batch:
        mask = batch["attention_mask"][0]
        nz = torch.nonzero(
            mask,
            as_tuple=False,
        ).flatten()

        if len(nz):
            return int(nz[-1].item())

    return int(
        batch["input_ids"].shape[1] - 1
    )


def parse_pred(text):
    s = str(text).lower()

    hits = []

    for rel, pattern in [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]:
        m = re.search(pattern, s)
        if m:
            hits.append((m.start(), rel))

    if not hits:
        return None

    return sorted(hits)[0][1]


def generate(
    model,
    processor,
    batch,
    max_new_tokens,
):
    input_len = int(
        batch["input_ids"].shape[1]
    )

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
        f"Cannot extract hidden tensor from type={type(output)}"
    )


def replace_hidden(output, descriptor, hidden):
    kind, idx = descriptor

    if kind == "tensor":
        return hidden

    vals = list(output)
    vals[idx] = hidden

    if kind == "tuple":
        return tuple(vals)

    return vals


# =============================================================================
# Learn shared late relation-specific causal directions from TRAIN
# =============================================================================

class LastCapture:
    def __init__(
        self,
        decoder_layers,
        selected_layers,
        last_position,
    ):
        self.handles = []
        self.selected_layers = selected_layers
        self.last_position = last_position
        self.done = {
            l: False
            for l in selected_layers
        }
        self.states = {}

        for l in selected_layers:
            self.handles.append(
                decoder_layers[l].register_forward_hook(
                    self._make_hook(l)
                )
            )

    def _make_hook(self, l):
        def hook(_module, _inputs, output):
            if self.done[l]:
                return output

            hidden, _ = extract_hidden(output)

            if (
                hidden.ndim != 3
                or self.last_position >= hidden.shape[1]
            ):
                return output

            self.states[l] = (
                hidden[
                    0,
                    self.last_position,
                    :
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            self.done[l] = True

            return output

        return hook

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def generate_and_capture_last(
    model,
    processor,
    decoder_layers,
    image,
    rec,
    last_layers,
    args,
    device,
):
    batch = build_batch(
        processor,
        image,
        rec,
        args.prompt_template,
        device,
    )

    last_position = infer_last_prompt_position(
        batch
    )

    with LastCapture(
        decoder_layers,
        last_layers,
        last_position,
    ) as capture:

        text, pred = generate(
            model,
            processor,
            batch,
            args.max_new_tokens,
        )

        states = dict(capture.states)

    del batch

    return pred, text, states


def train_filter_ok(
    real_correct,
    gray_correct,
    mode,
):
    if mode == "all":
        return True

    if mode == "real_correct":
        return bool(real_correct)

    if mode == "real_correct_gray_wrong":
        return (
            bool(real_correct)
            and not bool(gray_correct)
        )

    raise ValueError(mode)


def learn_causal_templates(
    model,
    processor,
    decoder_layers,
    records,
    train_sids,
    last_layers,
    args,
    device,
    outdir,
):
    collected = {
        l: {
            rel: []
            for rel in RELS
        }
        for l in last_layers
    }

    rows = []

    for sid in tqdm(
        train_sids,
        desc="TRAIN Real-Gray last delta",
    ):
        rec = records[sid]
        real_image = None
        gray_image = None

        try:
            real_image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            gray_image = make_gray_image(
                real_image,
                args.gray_value,
            )

            real_pred, real_text, real_states = generate_and_capture_last(
                model,
                processor,
                decoder_layers,
                real_image,
                rec,
                last_layers,
                args,
                device,
            )

            gray_pred, gray_text, gray_states = generate_and_capture_last(
                model,
                processor,
                decoder_layers,
                gray_image,
                rec,
                last_layers,
                args,
                device,
            )

            real_correct = int(
                real_pred == rec["gt"]
            )
            gray_correct = int(
                gray_pred == rec["gt"]
            )

            used = int(
                train_filter_ok(
                    real_correct,
                    gray_correct,
                    args.template_train_filter,
                )
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "real_pred": real_pred or "",
                "gray_pred": gray_pred or "",
                "real_correct": real_correct,
                "gray_correct": gray_correct,
                "used": used,
                "real_text": real_text,
                "gray_text": gray_text,
            })

            if not used:
                continue

            for l in last_layers:
                if l not in real_states or l not in gray_states:
                    continue

                delta = (
                    real_states[l]
                    - gray_states[l]
                ).astype(np.float32)

                collected[l][rec["gt"]].append(delta)

        except Exception as exc:
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "used": 0,
                "error": f"{type(exc).__name__}: {exc}",
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
        outdir / "train_real_gray_generation.csv",
        rows,
    )

    templates = {
        "last": {}
    }

    summary = []

    for l in last_layers:
        relation_mean = {}

        for rel in RELS:
            vectors = collected[l][rel]

            if not vectors:
                raise RuntimeError(
                    f"L{l}: no TRAIN delta vectors for relation={rel}. "
                    f"Try --template-train-filter real_correct"
                )

            relation_mean[rel] = (
                np.stack(vectors, axis=0)
                .mean(axis=0)
                .astype(np.float32)
            )

        global_mean = (
            np.stack(
                [
                    relation_mean[rel]
                    for rel in RELS
                ],
                axis=0,
            )
            .mean(axis=0)
            .astype(np.float32)
        )

        shared = {
            rel: (
                relation_mean[rel]
                - global_mean
            ).astype(np.float32)
            for rel in RELS
        }

        templates["last"][l] = {
            "relation_mean": relation_mean,
            "global": global_mean,
            "shared": shared,
        }

        summary.append({
            "layer": l,
            "n_left": len(collected[l]["left"]),
            "n_right": len(collected[l]["right"]),
            "n_above": len(collected[l]["above"]),
            "n_below": len(collected[l]["below"]),
            "global_norm": float(np.linalg.norm(global_mean)),
            "shared_left_norm": float(np.linalg.norm(shared["left"])),
            "shared_right_norm": float(np.linalg.norm(shared["right"])),
            "shared_above_norm": float(np.linalg.norm(shared["above"])),
            "shared_below_norm": float(np.linalg.norm(shared["below"])),
        })

    write_csv(
        outdir / "train_template_summary.csv",
        summary,
    )

    return templates


# =============================================================================
# Non-oracle guide from old middle-layer Direction
# =============================================================================

def fit_guide_codebook(
    direction_bundle,
    guide_layers,
    train_controls,
):
    idx = []

    for i, sid in enumerate(
        direction_bundle["sids"].tolist()
    ):
        sid = int(sid)

        if (
            direction_bundle["split"].get(sid)
            != "train"
        ):
            continue

        if (
            train_controls == "correct"
            and direction_bundle["cached_group"].get(sid)
            != "correct"
        ):
            continue

        idx.append(i)

    if not idx:
        raise RuntimeError(
            "No TRAIN samples for guide."
        )

    codebook = {}

    for l in guide_layers:
        X = direction_bundle["vectors"][
            idx,
            l,
            :
        ].astype(np.float64)

        Y = direction_bundle["labels"][
            idx
        ]

        center = X.mean(axis=0)

        prototypes = {}

        for rel in RELS:
            mask = Y == rel

            if not np.any(mask):
                raise RuntimeError(
                    f"L{l}: no TRAIN guide samples for {rel}"
                )

            mu = (
                X[mask]
                - center[None, :]
            ).mean(axis=0)

            norm = np.linalg.norm(mu)

            if norm <= EPS:
                raise RuntimeError(
                    f"L{l}: degenerate guide prototype {rel}"
                )

            prototypes[rel] = (
                mu / norm
            ).astype(np.float32)

        codebook[l] = {
            "center": center.astype(np.float32),
            "prototypes": prototypes,
        }

    return codebook


def guide_predict(
    direction_bundle,
    codebook,
    sid,
    guide_layers,
):
    i = direction_bundle[
        "sid_to_index"
    ][sid]

    votes = {
        rel: 0
        for rel in RELS
    }

    score_sum = {
        rel: 0.0
        for rel in RELS
    }

    for l in guide_layers:
        q = direction_bundle["vectors"][
            i,
            l,
            :
        ].astype(np.float64)

        q = (
            q
            - codebook[l]["center"].astype(np.float64)
        )

        qnorm = np.linalg.norm(q)

        if qnorm <= EPS:
            continue

        q = q / qnorm

        scores = {
            rel: float(
                np.dot(
                    q,
                    codebook[l]["prototypes"][rel].astype(np.float64),
                )
            )
            for rel in RELS
        }

        pred = max(
            scores,
            key=scores.get,
        )

        votes[pred] += 1

        for rel in RELS:
            score_sum[rel] += scores[rel]

    max_vote = max(
        votes.values()
    )

    tied = [
        rel
        for rel in RELS
        if votes[rel] == max_vote
    ]

    if len(tied) == 1:
        final = tied[0]
    else:
        final = max(
            tied,
            key=lambda rel: score_sum[rel],
        )

    return final, ";".join(
        f"{rel}:{votes[rel]}"
        for rel in RELS
    )


# =============================================================================
# REAL-image baseline + steering
# =============================================================================

def run_real_baseline(
    model,
    processor,
    records,
    test_sids,
    args,
    device,
    outdir,
):
    rows = []

    for sid in tqdm(
        test_sids,
        desc="REAL baseline",
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
                args.prompt_template,
                device,
            )

            text, pred = generate(
                model,
                processor,
                batch,
                args.max_new_tokens,
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "pred": pred or "",
                "correct": int(pred == rec["gt"]),
                "generation_text": text,
            })

            del batch

        except Exception as exc:
            tqdm.write(
                f"[BASELINE ERROR sid={sid}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                image.close()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(
        outdir / "baseline.csv",
        rows,
    )

    return rows


class LastSteeringHook:
    def __init__(
        self,
        decoder_layers,
        templates,
        selected_layers,
        target_relation,
        baseline_relation,
        edit_mode,
        scale,
        last_position,
    ):
        self.handles = []
        self.done = {}
        self.templates = templates
        self.selected_layers = selected_layers
        self.target_relation = target_relation
        self.baseline_relation = baseline_relation
        self.edit_mode = edit_mode
        self.scale = float(scale)
        self.last_position = int(last_position)

        for l in selected_layers:
            self.done[l] = False

            self.handles.append(
                decoder_layers[l].register_forward_hook(
                    self._make_hook(l)
                )
            )

    def vector_for(self, l):
        target = np.asarray(
            self.templates["last"][l]["shared"][
                self.target_relation
            ],
            dtype=np.float32,
        )

        if self.edit_mode == "add":
            return self.scale * target

        if self.edit_mode == "contrast":
            if self.baseline_relation not in RELSET:
                return self.scale * target

            source = np.asarray(
                self.templates["last"][l]["shared"][
                    self.baseline_relation
                ],
                dtype=np.float32,
            )

            return self.scale * (
                target
                - source
            )

        raise ValueError(
            self.edit_mode
        )

    def _make_hook(self, l):
        def hook(_module, _inputs, output):
            if self.done[l]:
                return output

            hidden, descriptor = extract_hidden(
                output
            )

            if (
                hidden.ndim != 3
                or self.last_position >= hidden.shape[1]
            ):
                return output

            vec = torch.as_tensor(
                self.vector_for(l),
                device=hidden.device,
                dtype=hidden.dtype,
            )

            if vec.shape[-1] != hidden.shape[-1]:
                raise RuntimeError(
                    f"L{l}: steering dim={vec.shape[-1]} "
                    f"!= hidden dim={hidden.shape[-1]}"
                )

            edited = hidden.clone()

            edited[
                :,
                self.last_position,
                :
            ] = (
                edited[
                    :,
                    self.last_position,
                    :
                ]
                + vec[None, :]
            )

            self.done[l] = True

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

    def __exit__(self, *_args):
        self.close()


def run_condition(
    model,
    processor,
    decoder_layers,
    templates,
    records,
    baseline_rows,
    guide_predictions,
    selected_layers,
    selector,
    apply_mode,
    edit_mode,
    args,
    device,
    condition_name,
):
    rows = []

    for base in tqdm(
        baseline_rows,
        desc=condition_name,
    ):
        sid = int(base["sid"])
        rec = records[sid]

        baseline_pred = normalize_relation(
            base["pred"]
        )

        if selector == "guide":
            target = guide_predictions[sid]
        elif selector == "oracle":
            target = rec["gt"]
        else:
            raise ValueError(selector)

        if target not in RELSET:
            continue

        if apply_mode == "conflict_only":
            apply_edit = (
                baseline_pred in RELSET
                and target != baseline_pred
            )
        elif apply_mode == "all":
            apply_edit = True
        else:
            raise ValueError(apply_mode)

        if not apply_edit:
            rows.append({
                "condition": condition_name,
                "sid": sid,
                "gt": rec["gt"],
                "target": target,
                "baseline_pred": baseline_pred or "",
                "edit_pred": baseline_pred or "",
                "baseline_correct": int(base["correct"]),
                "edit_correct": int(base["correct"]),
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
                args.prompt_template,
                device,
            )

            last_position = infer_last_prompt_position(
                batch
            )

            with LastSteeringHook(
                decoder_layers=decoder_layers,
                templates=templates,
                selected_layers=selected_layers,
                target_relation=target,
                baseline_relation=baseline_pred,
                edit_mode=edit_mode,
                scale=args.scale,
                last_position=last_position,
            ):

                text, pred = generate(
                    model,
                    processor,
                    batch,
                    args.max_new_tokens,
                )

            base_correct = int(
                base["correct"]
            )

            edit_correct = int(
                pred == rec["gt"]
            )

            rows.append({
                "condition": condition_name,
                "sid": sid,
                "gt": rec["gt"],
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
                "generation_text": text,
            })

            del batch

        except Exception as exc:
            tqdm.write(
                f"[STEER ERROR sid={sid} {condition_name}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                image.close()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def summarize(rows, condition_name):
    if not rows:
        return {
            "condition": condition_name,
            "N": 0,
        }

    n = len(rows)

    base_acc = safe_mean(
        r["baseline_correct"]
        for r in rows
    )

    edit_acc = safe_mean(
        r["edit_correct"]
        for r in rows
    )

    n_wrong = sum(
        1 - int(r["baseline_correct"])
        for r in rows
    )

    n_correct = n - n_wrong

    w2c = sum(
        int(r["W2C"])
        for r in rows
    )

    c2w = sum(
        int(r["C2W"])
        for r in rows
    )

    applied = sum(
        int(r["applied"])
        for r in rows
    )

    return {
        "condition": condition_name,
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
            w2c / n_wrong
            if n_wrong
            else float("nan")
        ),
        "C2W": c2w,
        "C2W_rate_correct": (
            c2w / n_correct
            if n_correct
            else float("nan")
        ),
        "net": w2c - c2w,
        "changed_rate": safe_mean(
            r["changed"]
            for r in rows
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

    selectors = parse_choices(
        args.selectors,
        {"guide", "oracle"},
    )

    apply_modes = parse_choices(
        args.apply_modes,
        {"conflict_only", "all"},
    )

    edit_modes = parse_choices(
        args.edit_modes,
        {"add", "contrast"},
    )

    windows = parse_choices(
        args.windows,
        {"single", "multi"},
    )

    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    direction_bundle = load_direction_bundle(
        args.direction_dir,
        args.direction_key,
    )

    records = load_records(
        args.prompt_jsonl,
        args.annotation_json,
        args.data_root,
        direction_bundle,
    )

    model, processor = load_model_and_processor(
        args.model_id,
        dtype_from_name(args.dtype),
        args.device,
        args.attn_impl,
    )

    decoder_layers, decoder_path = resolve_decoder_layers(
        model
    )

    guide_layers = parse_layers(
        args.guide_layers,
        direction_bundle["vectors"].shape[1],
    )

    last_layers = parse_layers(
        args.last_layers,
        len(decoder_layers),
    )

    print(
        f"[decoder] {decoder_path}"
    )
    print(
        f"[guide layers] {guide_layers}"
    )
    print(
        f"[last causal layers] {last_layers}"
    )

    # -------------------------------------------------------------------------
    # Fit old middle-layer Direction guide
    # -------------------------------------------------------------------------

    guide_codebook = fit_guide_codebook(
        direction_bundle,
        guide_layers,
        args.guide_train_controls,
    )

    # -------------------------------------------------------------------------
    # Learn NEW late causal directions from TRAIN Real-Gray deltas
    # -------------------------------------------------------------------------

    train_sids = sorted(
        sid
        for sid, rec in records.items()
        if rec["split"] == "train"
    )

    if (
        args.max_train_samples is not None
        and len(train_sids) > args.max_train_samples
    ):
        rng = random.Random(args.seed)
        rng.shuffle(train_sids)
        train_sids = sorted(
            train_sids[
                :args.max_train_samples
            ]
        )

    templates = learn_causal_templates(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        train_sids=train_sids,
        last_layers=last_layers,
        args=args,
        device=torch.device(args.device),
        outdir=outdir,
    )

    # -------------------------------------------------------------------------
    # Fresh REAL baseline
    # -------------------------------------------------------------------------

    test_sids = sorted(
        sid
        for sid, rec in records.items()
        if rec["split"] == "test"
    )

    if (
        args.max_test_samples is not None
        and len(test_sids) > args.max_test_samples
    ):
        rng = random.Random(args.seed + 17)
        rng.shuffle(test_sids)
        test_sids = sorted(
            test_sids[
                :args.max_test_samples
            ]
        )

    baseline_rows = run_real_baseline(
        model=model,
        processor=processor,
        records=records,
        test_sids=test_sids,
        args=args,
        device=torch.device(args.device),
        outdir=outdir,
    )

    baseline_acc = safe_mean(
        row["correct"]
        for row in baseline_rows
    )

    # -------------------------------------------------------------------------
    # Non-oracle guide predictions
    # -------------------------------------------------------------------------

    guide_predictions = {}
    guide_rows = []

    for row in baseline_rows:
        sid = int(row["sid"])

        pred, votes = guide_predict(
            direction_bundle,
            guide_codebook,
            sid,
            guide_layers,
        )

        guide_predictions[sid] = pred

        baseline_pred = normalize_relation(
            row["pred"]
        )

        guide_rows.append({
            "sid": sid,
            "gt": records[sid]["gt"],
            "baseline_pred": baseline_pred or "",
            "baseline_correct": int(row["correct"]),
            "guide_pred": pred,
            "guide_correct": int(
                pred == records[sid]["gt"]
            ),
            "conflict": int(
                pred in RELSET
                and baseline_pred in RELSET
                and pred != baseline_pred
            ),
            "votes": votes,
        })

    write_csv(
        outdir / "guide_summary.csv",
        guide_rows,
    )

    guide_acc = safe_mean(
        row["guide_correct"]
        for row in guide_rows
    )

    conflicts = [
        row
        for row in guide_rows
        if int(row["conflict"]) == 1
    ]

    print(
        "\n"
        + "=" * 125
    )

    print(
        f"REAL baseline | N={len(baseline_rows)} | acc={baseline_acc:.4f}"
    )

    print(
        f"guide acc={guide_acc:.4f} | "
        f"conflicts={len(conflicts)}/{len(guide_rows)}"
    )

    if conflicts:
        print(
            "on conflicts | "
            f"baseline_correct="
            f"{safe_mean(r['baseline_correct'] for r in conflicts):.4f} | "
            f"guide_correct="
            f"{safe_mean(r['guide_correct'] for r in conflicts):.4f}"
        )

    print(
        "=" * 125
    )

    # -------------------------------------------------------------------------
    # Steering conditions
    # -------------------------------------------------------------------------

    layer_windows = []

    if "single" in windows:
        for l in last_layers:
            layer_windows.append(
                (
                    f"L{l:02d}",
                    [l],
                )
            )

    if "multi" in windows:
        layer_windows.append(
            (
                "multi",
                last_layers,
            )
        )

    detail_rows = []
    summary_rows = []

    for selector in selectors:
        for apply_mode in apply_modes:
            for edit_mode in edit_modes:
                for window_name, selected_layers in layer_windows:

                    condition_name = (
                        f"{selector}_"
                        f"{apply_mode}_"
                        f"{edit_mode}_"
                        f"{window_name}"
                    )

                    rows = run_condition(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        templates=templates,
                        records=records,
                        baseline_rows=baseline_rows,
                        guide_predictions=guide_predictions,
                        selected_layers=selected_layers,
                        selector=selector,
                        apply_mode=apply_mode,
                        edit_mode=edit_mode,
                        args=args,
                        device=torch.device(args.device),
                        condition_name=condition_name,
                    )

                    detail_rows.extend(
                        rows
                    )

                    summary_rows.append(
                        summarize(
                            rows,
                            condition_name,
                        )
                    )

                    write_csv(
                        outdir / "steering_details.csv",
                        detail_rows,
                    )

                    write_csv(
                        outdir / "summary.csv",
                        summary_rows,
                    )

    print(
        "\n"
        + "=" * 170
    )

    print(
        "REAL-IMAGE LAST-TOKEN CAUSAL DIRECTION STEERING — ACTUAL model.generate()"
    )

    print(
        "=" * 170
    )

    print(
        "condition                                      | "
        "acc base->edit gain | applied | W2C/wrong | C2W/correct | net | changed"
    )

    for row in summary_rows:
        print(
            f"{str(row['condition']):46s} | "
            f"{float(row['base_acc']):.4f}->"
            f"{float(row['edit_acc']):.4f} "
            f"{float(row['gain']):+.4f} | "
            f"{int(row['applied'])}/"
            f"{float(row['applied_rate']):.3f} | "
            f"{int(row['W2C'])}/"
            f"{float(row['W2C_rate_wrong']):.3f} | "
            f"{int(row['C2W'])}/"
            f"{float(row['C2W_rate_correct']):.3f} | "
            f"{int(row['net']):+d} | "
            f"{float(row['changed_rate']):.3f}"
        )

    (
        outdir / "summary.json"
    ).write_text(
        json.dumps(
            {
                "experiment":
                    "real-image last-token relation-specific causal steering",
                "standalone":
                    True,
                "baseline_acc":
                    baseline_acc,
                "guide_acc":
                    guide_acc,
                "guide_layers":
                    guide_layers,
                "last_layers":
                    last_layers,
                "scale":
                    args.scale,
                "template_train_filter":
                    args.template_train_filter,
                "note":
                    (
                        "guide selector is non-oracle; "
                        "oracle selector is only an upper-bound diagnostic."
                    ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved:"
    )

    for path in [
        outdir / "train_real_gray_generation.csv",
        outdir / "train_template_summary.csv",
        outdir / "baseline.csv",
        outdir / "guide_summary.csv",
        outdir / "steering_details.csv",
        outdir / "summary.csv",
        outdir / "summary.json",
    ]:
        if path.exists():
            print(
                " ",
                path,
            )


if __name__ == "__main__":
    main()
