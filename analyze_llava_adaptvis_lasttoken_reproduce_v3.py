#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_llava_adaptvis_lasttoken_reproduce_v3.py

Version-agnostic mechanistic reproduction for bxqm803/AdaptVis llava16.

Designed for the user's current environment (e.g. transformers 4.57.6) where
the custom LlavaForConditionalGenerationScal may not expose `.generate()`.

The script therefore uses the custom model's own:
    forward() + past_key_values
for deterministic greedy generation.

Crucially, it reproduces the ORIGINAL AdaptVis dynamic routing policy:

    B neutral (epsilon=1e-6, weight=1.0)
        -> first-step confidence c

    if round(c, 2) < threshold:
        selected weight = weight1 (COCO-two default 0.5)
    else:
        selected weight = weight2 (COCO-two default 1.2)

Then C uses that sample-specific selected weight.

Three conditions
================

A ORIGINAL:
    RMSNorm epsilon = 1e-5
    no AdaptVis (weight=1.0)

B EPS-ONLY:
    RMSNorm epsilon = 1e-6
    no AdaptVis (weight=1.0)

C FULL:
    RMSNorm epsilon = 1e-6
    exact dynamic AdaptVis routing
    default scope L0-L31

This allows:

    delta_eps   = h_B - h_A
    delta_adapt = h_C - h_B
    delta_total = h_C - h_A

The script FIRST prints actual greedy generation accuracy for A/B/C:
    A acc
    B acc
    C acc
    W2C/C2W

Only then should the hidden-state results be interpreted.

Last-token measurements
=======================

At every decoder layer:

    ||delta_eps||
    ||delta_adapt||
    ||delta_total||
    cosine(A,C)

and actual post-softmax last-query -> visual-token attention mass:

    M_A[l]
    M_B[l]
    M_C[l]

using the exact merged visual-token mask produced by the custom LLaVA model.

Direction readout
=================

Repeated stratified TRAIN/TEST spatial readout for:

    A_last
    B_last
    C_last
    delta_eps
    delta_adapt
    delta_total

Generation transition groups:
    W2C, C2W, C2C, W2W

Dynamic routing groups:
    weight=0.5
    weight=1.2

Example
=======

CUDA_VISIBLE_DEVICES=0 python analyze_llava_adaptvis_lasttoken_reproduce_v3.py \
  --dataset COCO_QA_two_obj \
  --option four \
  --base-rms-eps 1e-5 \
  --enhanced-rms-eps 1e-6 \
  --weight1 0.5 \
  --weight2 1.2 \
  --threshold 0.3 \
  --adaptvis-max-layers 32 \
  --repeats 5 \
  --save-vectors \
  --require-improvement \
  --output-dir output/llava_adaptvis_lasttoken_reproduce_v2 \
  --overwrite

Only first four layers:
    --adaptvis-max-layers 4
"""

from __future__ import annotations

import os
os.environ["ADAPTVIS_ATTENTION_VARIANT"] = "mul_img"
os.environ["SAVE_ATTN"] = "False"

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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset_zoo import get_dataset
from misc import seed_all, _default_collate
from model_zoo import get_model
from model_zoo.llava15 import _is_correct, _norm_gold


RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--dataset",
        default="COCO_QA_two_obj",
        choices=[
            "Controlled_Images_A",
            "Controlled_Images_B",
            "COCO_QA_one_obj",
            "COCO_QA_two_obj",
            "VG_QA_one_obj",
            "VG_QA_two_obj",
        ],
    )

    p.add_argument(
        "--option",
        default="four",
        choices=["two", "four", "six"],
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--base-rms-eps",
        type=float,
        default=1e-5,
    )
    p.add_argument(
        "--enhanced-rms-eps",
        type=float,
        default=1e-6,
    )

    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.2)
    p.add_argument("--threshold", type=float, default=0.3)

    p.add_argument(
        "--adaptvis-max-layers",
        type=int,
        default=32,
        help="32=original full scope; 4=L0-L3 only.",
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--probe-layers",
        default="all",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    p.add_argument(
        "--require-improvement",
        action="store_true",
    )

    p.add_argument(
        "--save-vectors",
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

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(values):
    vals = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)

    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values):
    vals = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)

    return float(np.std(vals)) if vals else float("nan")


def write_csv(path: Path, rows):
    rows = list(rows)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def cosine_np(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= EPS or nb <= EPS:
        return float("nan")

    return float(
        np.dot(a, b)
        / (na * nb)
    )


def normalize_rows(x):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    n = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return x / np.maximum(
        n,
        EPS,
    )


def normalize_relation(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    text = str(value).strip().lower()
    text = re.sub(
        r"[^a-z]+",
        " ",
        text,
    )

    toks = text.split()

    for tok in toks[:12]:
        if tok in REL_TO_ID:
            return tok

    if " left " in f" {text} ":
        return "left"
    if " right " in f" {text} ":
        return "right"
    if "above" in toks or "top" in toks:
        return "above"
    if (
        "below" in toks
        or "under" in toks
        or "underneath" in toks
        or "bottom" in toks
    ):
        return "below"

    return None


def parse_layer_spec(text, n_layers):
    raw = str(text).strip().lower()

    if raw == "all":
        return list(
            range(n_layers)
        )

    if raw == "auto":
        candidates = [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 10, 12,
            14, 16, 18,
            20, 22, 24,
            26, 28, 30,
            n_layers - 1,
        ]

        return sorted({
            x
            for x in candidates
            if 0 <= x < n_layers
        })

    result = []

    for part in raw.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)

            step = (
                1
                if b >= a
                else -1
            )

            result.extend(
                range(
                    a,
                    b + step,
                    step,
                )
            )
        else:
            result.append(
                int(part)
            )

    result = list(
        dict.fromkeys(result)
    )

    bad = [
        x
        for x in result
        if not (0 <= x < n_layers)
    ]

    if bad:
        raise ValueError(
            f"Bad layers={bad}; "
            f"valid 0..{n_layers-1}"
        )

    return result


# =============================================================================
# Prompt/data
# =============================================================================

def load_prompts(
    dataset_name,
    option,
):
    path = Path(
        f"prompts/{dataset_name}_with_answer_{option}_options.jsonl"
    )

    prompts = []
    answers = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)

            prompts.append(
                str(row["question"])
            )

            answers.append(
                row["answer"]
            )

    return prompts, answers


def iter_samples(
    dataset,
    prompts,
    answers,
    *,
    num_workers,
    max_samples,
):
    # Match repository main_aro.py exactly for LLaVA:
    # image_preprocess=None means dataset items contain raw PIL images, so the
    # repository's custom _default_collate must be used. PyTorch's default
    # collate cannot batch PIL.Image.Image objects.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_default_collate,
    )

    sid = 0

    for batch in loader:
        for i_option in batch[
            "image_options"
        ]:
            for image in i_option:
                if sid >= len(prompts):
                    return

                if (
                    max_samples is not None
                    and sid >= int(max_samples)
                ):
                    return

                yield (
                    sid,
                    image,
                    prompts[sid],
                    _norm_gold(
                        answers[sid]
                    ),
                )

                sid += 1


# =============================================================================
# RMSNorm
# =============================================================================

def set_rms_eps(model, eps):
    eps = float(eps)

    changed = 0

    for _, module in model.named_modules():
        cls = module.__class__.__name__.lower()

        if "rmsnorm" not in cls:
            continue

        touched = False

        if hasattr(
            module,
            "variance_epsilon",
        ):
            module.variance_epsilon = eps
            touched = True

        if hasattr(
            module,
            "eps",
        ):
            module.eps = eps
            touched = True

        changed += int(touched)

    configs = [
        getattr(
            model,
            "config",
            None,
        ),
        getattr(
            getattr(
                model,
                "config",
                None,
            ),
            "text_config",
            None,
        ),
        getattr(
            getattr(
                model,
                "language_model",
                None,
            ),
            "config",
            None,
        ),
    ]

    for cfg in configs:
        if (
            cfg is not None
            and hasattr(
                cfg,
                "rms_norm_eps",
            )
        ):
            cfg.rms_norm_eps = eps

    if changed == 0:
        raise RuntimeError(
            "No RMSNorm modules found."
        )

    return changed


def unique_rms_eps(model):
    values = []

    for _, module in model.named_modules():
        cls = module.__class__.__name__.lower()

        if "rmsnorm" not in cls:
            continue

        if hasattr(
            module,
            "variance_epsilon",
        ):
            values.append(
                float(
                    module.variance_epsilon
                )
            )
        elif hasattr(
            module,
            "eps",
        ):
            values.append(
                float(module.eps)
            )

    return sorted(
        set(values)
    )


# =============================================================================
# Restrict AdaptVis to early layers if requested
# =============================================================================

class RestrictAdaptVisLayers:

    def __init__(
        self,
        model,
        max_layers,
    ):
        self.max_layers = int(
            max_layers
        )

        self.handles = []

        if self.max_layers >= 32:
            return

        layers = (
            model
            .language_model
            .model
            .layers
        )

        for layer in layers:
            handle = (
                layer
                .self_attn
                .register_forward_pre_hook(
                    self._hook,
                    with_kwargs=True,
                )
            )

            self.handles.append(
                handle
            )

    def _hook(
        self,
        module,
        args,
        kwargs,
    ):
        idx = kwargs.get(
            "idx",
            None,
        )

        if (
            idx is not None
            and int(idx)
            >= self.max_layers
        ):
            kwargs = dict(kwargs)
            kwargs["weight"] = None

        return args, kwargs

    def close(self):
        for h in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Legacy-safe processor path
# =============================================================================

def build_input(
    wrapper,
    prompt,
    image,
):
    """
    Avoid new LlavaProcessor placeholder expansion.

    The custom Scal model expects ONE <image> token and expands it internally.
    """

    tokenizer = wrapper.processor.tokenizer
    image_processor = (
        wrapper.processor
        .image_processor
    )

    tok = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )

    img = image_processor(
        images=image,
        return_tensors="pt",
    )

    batch = {
        "input_ids": (
            tok["input_ids"]
            .to(wrapper.device)
        ),

        "attention_mask": (
            tok.get(
                "attention_mask",
                torch.ones_like(
                    tok["input_ids"]
                ),
            )
            .to(wrapper.device)
        ),

        "pixel_values": (
            img["pixel_values"]
            .to(wrapper.device)
        ),
    }

    image_token_index = int(
        wrapper.model
        .config
        .image_token_index
    )

    count = int(
        (
            batch["input_ids"]
            == image_token_index
        )
        .sum()
        .item()
    )

    if count != 1:
        raise RuntimeError(
            "Custom legacy Scal model expects exactly "
            f"one <image> token, got {count}."
        )

    return batch


# =============================================================================
# Full prompt forward + state/attention capture
# =============================================================================

def full_prompt_forward(
    model,
    batch,
    *,
    weight,
    output_attentions=True,
    output_hidden_states=True,
    use_cache=True,
):
    with torch.inference_mode():
        out = model(
            **batch,
            weight=float(weight),
            adjust_method="mul_img",
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
        )

    image_mask = getattr(
        model,
        "_adaptvis_last_image_id",
        None,
    )

    merged_mask = getattr(
        model,
        "_adaptvis_last_attention_mask",
        None,
    )

    if image_mask is None:
        raise RuntimeError(
            "Model did not expose merged image mask."
        )

    if merged_mask is None:
        raise RuntimeError(
            "Model did not expose merged attention mask."
        )

    image_mask = (
        image_mask[0]
        .bool()
        .to(batch["input_ids"].device)
    )

    merged_mask = (
        merged_mask[0]
        .bool()
    )

    valid = torch.where(
        merged_mask
    )[0]

    last_idx = int(
        valid[-1].item()
    )

    return (
        out,
        image_mask,
        last_idx,
    )


def capture_trajectory(
    outputs,
    image_mask,
    last_idx,
    probe_layers,
):
    hs = outputs.hidden_states
    atts = outputs.attentions

    vectors = []
    masses = []

    for layer in probe_layers:
        state = hs[
            int(layer) + 1
        ]

        vectors.append(
            state[
                0,
                last_idx,
                :
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        att = atts[
            int(layer)
        ]

        last_att = att[
            0,
            :,
            last_idx,
            :,
        ]

        if (
            int(last_att.shape[-1])
            != int(
                image_mask.numel()
            )
        ):
            raise RuntimeError(
                f"L{layer}: attention/image-mask mismatch "
                f"{last_att.shape[-1]} vs "
                f"{image_mask.numel()}"
            )

        mass = (
            last_att[
                :,
                image_mask,
            ]
            .sum(dim=-1)
            .float()
            .mean()
            .item()
        )

        masses.append(
            float(mass)
        )

    return (
        np.stack(
            vectors,
            axis=0,
        ).astype(np.float32),
        np.asarray(
            masses,
            dtype=np.float32,
        ),
    )


# =============================================================================
# Version-agnostic greedy continuation
# =============================================================================

def eos_ids(
    model,
    tokenizer,
):
    vals = []

    for value in [
        getattr(
            model.config,
            "eos_token_id",
            None,
        ),
        getattr(
            tokenizer,
            "eos_token_id",
            None,
        ),
    ]:
        if value is None:
            continue

        if isinstance(
            value,
            (list, tuple),
        ):
            vals.extend(
                int(x)
                for x in value
            )
        else:
            vals.append(
                int(value)
            )

    return set(
        dict.fromkeys(vals)
    )


def greedy_continue_from_prompt(
    model,
    tokenizer,
    batch,
    prompt_outputs,
    *,
    max_new_tokens,
):
    """
    Continue autoregressive decoding from an already-computed full prompt.

    This avoids GenerationMixin entirely and preserves the exact first-step
    state used for confidence / hidden-state analysis.
    """

    if int(max_new_tokens) < 1:
        return ""

    logits = (
        prompt_outputs
        .logits[
            :,
            -1,
            :
        ]
    )

    next_token = torch.argmax(
        logits,
        dim=-1,
    )

    generated = [
        next_token.detach()
    ]

    past = (
        prompt_outputs
        .past_key_values
    )

    if past is None:
        raise RuntimeError(
            "use_cache=True returned no past_key_values."
        )

    eos = eos_ids(
        model,
        tokenizer,
    )

    # Pre-merge attention mask; custom LLaVA cache branch expands it
    # against the actual cached merged sequence.
    running_mask = (
        batch["attention_mask"]
        .clone()
    )

    running_mask = torch.cat(
        [
            running_mask,
            torch.ones(
                (
                    running_mask.shape[0],
                    1,
                ),
                dtype=running_mask.dtype,
                device=running_mask.device,
            ),
        ],
        dim=1,
    )

    if (
        eos
        and int(
            next_token.item()
        ) in eos
    ):
        ids = torch.stack(
            generated,
            dim=1,
        )

        return tokenizer.decode(
            ids[0],
            skip_special_tokens=True,
        ).strip()

    current = (
        next_token[
            :,
            None,
        ]
    )

    for _ in range(
        1,
        int(max_new_tokens),
    ):
        with torch.inference_mode():
            out = model(
                input_ids=current,
                pixel_values=batch[
                    "pixel_values"
                ],
                attention_mask=running_mask,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                weight=None,
                adjust_method=None,
            )

        next_token = torch.argmax(
            out.logits[
                :,
                -1,
                :
            ],
            dim=-1,
        )

        generated.append(
            next_token.detach()
        )

        past = out.past_key_values

        current = (
            next_token[
                :,
                None,
            ]
        )

        running_mask = torch.cat(
            [
                running_mask,
                torch.ones(
                    (
                        running_mask.shape[0],
                        1,
                    ),
                    dtype=running_mask.dtype,
                    device=running_mask.device,
                ),
            ],
            dim=1,
        )

        if (
            eos
            and int(
                next_token.item()
            ) in eos
        ):
            break

    ids = torch.stack(
        generated,
        dim=1,
    )

    return tokenizer.decode(
        ids[0],
        skip_special_tokens=True,
    ).strip()


# =============================================================================
# Direction probe
# =============================================================================

def stratified_split(
    labels,
    train_ratio,
    seed,
):
    rng = random.Random(seed)

    train = []
    test = []

    for relation in RELATIONS:
        ids = np.flatnonzero(
            labels == relation
        ).tolist()

        rng.shuffle(ids)

        if len(ids) < 2:
            raise RuntimeError(
                f"Too few samples for {relation}."
            )

        n_train = int(
            round(
                len(ids)
                * train_ratio
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(ids) - 1,
            ),
        )

        train.extend(
            ids[:n_train]
        )

        test.extend(
            ids[n_train:]
        )

    rng.shuffle(train)
    rng.shuffle(test)

    return (
        np.asarray(
            train,
            dtype=np.int64,
        ),
        np.asarray(
            test,
            dtype=np.int64,
        ),
    )


def fit_codebook(
    X,
    labels,
):
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    center = X.mean(
        axis=0
    )

    Xc = X - center

    dirs = []

    for relation in RELATIONS:
        mask = (
            labels == relation
        )

        d = Xc[
            mask
        ].mean(
            axis=0
        )

        d = (
            d
            / max(
                float(
                    np.linalg.norm(d)
                ),
                EPS,
            )
        )

        dirs.append(d)

    return (
        center,
        np.stack(
            dirs,
            axis=0,
        ),
    )


def probe_acc(
    X,
    labels,
    train_idx,
    test_idx,
):
    center, dirs = fit_codebook(
        X[train_idx],
        labels[train_idx],
    )

    Xt = normalize_rows(
        X[test_idx]
        - center
    )

    scores = (
        Xt
        @ dirs.T
    )

    pred = np.argmax(
        scores,
        axis=1,
    )

    gt = np.asarray(
        [
            REL_TO_ID[
                str(x)
            ]
            for x
            in labels[
                test_idx
            ]
        ],
        dtype=np.int64,
    )

    return float(
        np.mean(
            pred == gt
        )
    )


def run_probes(
    reps,
    labels,
    probe_layers,
    *,
    train_ratio,
    repeats,
    seed,
):
    raw = []

    for rep in range(
        repeats
    ):
        tr, te = (
            stratified_split(
                labels,
                train_ratio,
                seed + rep,
            )
        )

        for name, tensor in reps.items():
            for li, layer in enumerate(
                probe_layers
            ):
                acc = probe_acc(
                    tensor[
                        :,
                        li,
                        :,
                    ].astype(
                        np.float32
                    ),
                    labels,
                    tr,
                    te,
                )

                raw.append({
                    "repeat": rep,
                    "representation": name,
                    "layer": layer,
                    "accuracy": acc,
                })

    summary = []

    for name in reps:
        for layer in probe_layers:
            rows = [
                r
                for r in raw
                if (
                    r["representation"]
                    == name
                    and int(
                        r["layer"]
                    )
                    == int(layer)
                )
            ]

            summary.append({
                "representation": name,
                "layer": layer,
                "accuracy_mean": safe_mean(
                    r["accuracy"]
                    for r in rows
                ),
                "accuracy_std": safe_std(
                    r["accuracy"]
                    for r in rows
                ),
            })

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("\n" + "=" * 130)
    print("ENVIRONMENT (informational only)")
    print("=" * 130)
    print(
        f"torch={torch.__version__}"
    )
    print(
        f"transformers={transformers.__version__}"
    )
    print(
        "GenerationMixin is NOT required by this script."
    )
    print("=" * 130)

    if not (
        0.0
        < args.train_ratio
        < 1.0
    ):
        raise ValueError(
            "--train-ratio must be in (0,1)"
        )

    if not (
        1
        <= args.adaptvis_max_layers
        <= 32
    ):
        raise ValueError(
            "--adaptvis-max-layers must be 1..32"
        )

    seed_all(args.seed)
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

    prompts, answers = (
        load_prompts(
            args.dataset,
            args.option,
        )
    )

    dataset = get_dataset(
        args.dataset,
        image_preprocess=None,
        download=False,
    )

    wrapper, _ = get_model(
        "llava1.5",
        args.device,
        method="adapt_vis",
        root_dir=args.root_dir,
    )

    model = wrapper.model
    tokenizer = (
        wrapper.processor
        .tokenizer
    )

    n_layers = len(
        model
        .language_model
        .model
        .layers
    )

    probe_layers = (
        parse_layer_spec(
            args.probe_layers,
            n_layers,
        )
    )

    print("\n" + "=" * 150)
    print("EXPERIMENT")
    print("=" * 150)
    print(
        f"A: eps={args.base_rms_eps:g}, weight=1.0"
    )
    print(
        f"B: eps={args.enhanced_rms_eps:g}, weight=1.0"
    )
    print(
        f"C: eps={args.enhanced_rms_eps:g}, dynamic AdaptVis "
        f"({args.weight1}/{args.weight2}, threshold={args.threshold})"
    )
    print(
        f"AdaptVis scope=L0-L{args.adaptvis_max_layers-1}"
    )
    print(
        "dataloader_collate=misc._default_collate "
        "(repository PIL-safe path)"
    )
    print("=" * 150)

    A_vectors = []
    B_vectors = []
    C_vectors = []

    A_mass = []
    B_mass = []
    C_mass = []

    sids = []
    labels = []
    selected_weights = []
    transition_groups = []

    generation_rows = []
    geometry_rows = []

    A_correct = 0
    B_correct = 0
    C_correct = 0

    w2c = c2w = c2c = w2w = 0

    weight1_count = 0
    weight2_count = 0

    with RestrictAdaptVisLayers(
        model,
        args.adaptvis_max_layers,
    ):
        iterator = iter_samples(
            dataset,
            prompts,
            answers,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        for (
            sid,
            image,
            prompt,
            gold,
        ) in tqdm(
            iterator,
            desc="A/B/C generation + last-token trajectory",
        ):
            relation = normalize_relation(
                gold
            )

            if relation not in REL_TO_ID:
                raise RuntimeError(
                    f"sid={sid}: unsupported gold={gold!r}"
                )

            batch = build_input(
                wrapper,
                prompt,
                image,
            )

            # -------------------------------------------------------------
            # A: epsilon 1e-5 neutral
            # -------------------------------------------------------------
            set_rms_eps(
                model,
                args.base_rms_eps,
            )

            (
                A_out,
                A_imgmask,
                A_lastidx,
            ) = full_prompt_forward(
                model,
                batch,
                weight=1.0,
            )

            (
                A_h,
                A_m,
            ) = capture_trajectory(
                A_out,
                A_imgmask,
                A_lastidx,
                probe_layers,
            )

            A_text = (
                greedy_continue_from_prompt(
                    model,
                    tokenizer,
                    batch,
                    A_out,
                    max_new_tokens=args.max_new_tokens,
                )
            )

            # -------------------------------------------------------------
            # B: epsilon 1e-6 neutral.
            # This first-step logits determine original AdaptVis routing.
            # -------------------------------------------------------------
            set_rms_eps(
                model,
                args.enhanced_rms_eps,
            )

            (
                B_out,
                B_imgmask,
                B_lastidx,
            ) = full_prompt_forward(
                model,
                batch,
                weight=1.0,
            )

            (
                B_h,
                B_m,
            ) = capture_trajectory(
                B_out,
                B_imgmask,
                B_lastidx,
                probe_layers,
            )

            first_logits = (
                B_out
                .logits[
                    :,
                    -1,
                    :
                ]
                .detach()
                .float()
            )

            confidence = float(
                torch.softmax(
                    first_logits,
                    dim=-1,
                )[0].max().item()
            )

            confidence_rounded = float(
                np.round(
                    confidence,
                    2,
                )
            )

            selected_weight = (
                float(
                    args.weight1
                )
                if confidence_rounded
                < float(
                    args.threshold
                )
                else float(
                    args.weight2
                )
            )

            B_text = (
                greedy_continue_from_prompt(
                    model,
                    tokenizer,
                    batch,
                    B_out,
                    max_new_tokens=args.max_new_tokens,
                )
            )

            # -------------------------------------------------------------
            # C: exact dynamic AdaptVis at epsilon 1e-6.
            # -------------------------------------------------------------
            (
                C_out,
                C_imgmask,
                C_lastidx,
            ) = full_prompt_forward(
                model,
                batch,
                weight=selected_weight,
            )

            (
                C_h,
                C_m,
            ) = capture_trajectory(
                C_out,
                C_imgmask,
                C_lastidx,
                probe_layers,
            )

            C_text = (
                greedy_continue_from_prompt(
                    model,
                    tokenizer,
                    batch,
                    C_out,
                    max_new_tokens=args.max_new_tokens,
                )
            )

            # -------------------------------------------------------------
            # Accuracy.
            # -------------------------------------------------------------
            a_ok = bool(
                _is_correct(
                    gold,
                    A_text,
                )
            )

            b_ok = bool(
                _is_correct(
                    gold,
                    B_text,
                )
            )

            c_ok = bool(
                _is_correct(
                    gold,
                    C_text,
                )
            )

            A_correct += int(a_ok)
            B_correct += int(b_ok)
            C_correct += int(c_ok)

            if (
                not b_ok
                and c_ok
            ):
                group = "W2C"
                w2c += 1
            elif (
                b_ok
                and not c_ok
            ):
                group = "C2W"
                c2w += 1
            elif (
                b_ok
                and c_ok
            ):
                group = "C2C"
                c2c += 1
            else:
                group = "W2W"
                w2w += 1

            if abs(
                selected_weight
                - float(
                    args.weight1
                )
            ) < 1e-8:
                weight1_count += 1
            else:
                weight2_count += 1

            generation_rows.append({
                "sid": sid,
                "relation": relation,
                "gold": gold,

                "confidence_raw": confidence,
                "confidence_rounded": (
                    confidence_rounded
                ),
                "selected_weight": (
                    selected_weight
                ),

                "A_text": A_text,
                "B_text": B_text,
                "C_text": C_text,

                "A_correct": int(a_ok),
                "B_correct": int(b_ok),
                "C_correct": int(c_ok),

                "B_to_C_group": group,
            })

            # -------------------------------------------------------------
            # Geometry.
            # -------------------------------------------------------------
            d_eps = (
                B_h - A_h
            ).astype(
                np.float32
            )

            d_adapt = (
                C_h - B_h
            ).astype(
                np.float32
            )

            d_total = (
                C_h - A_h
            ).astype(
                np.float32
            )

            for li, layer in enumerate(
                probe_layers
            ):
                geometry_rows.append({
                    "sid": sid,
                    "relation": relation,
                    "layer": layer,
                    "selected_weight": (
                        selected_weight
                    ),
                    "B_to_C_group": group,

                    "A_visual_mass": float(
                        A_m[li]
                    ),
                    "B_visual_mass": float(
                        B_m[li]
                    ),
                    "C_visual_mass": float(
                        C_m[li]
                    ),

                    "delta_eps_visual_mass": float(
                        B_m[li]
                        - A_m[li]
                    ),
                    "delta_adapt_visual_mass": float(
                        C_m[li]
                        - B_m[li]
                    ),
                    "delta_total_visual_mass": float(
                        C_m[li]
                        - A_m[li]
                    ),

                    "delta_eps_norm": float(
                        np.linalg.norm(
                            d_eps[li]
                        )
                    ),
                    "delta_adapt_norm": float(
                        np.linalg.norm(
                            d_adapt[li]
                        )
                    ),
                    "delta_total_norm": float(
                        np.linalg.norm(
                            d_total[li]
                        )
                    ),

                    "cos_A_C": cosine_np(
                        A_h[li],
                        C_h[li],
                    ),
                    "cos_B_C": cosine_np(
                        B_h[li],
                        C_h[li],
                    ),
                })

            A_vectors.append(
                A_h.astype(
                    np.float16
                )
            )
            B_vectors.append(
                B_h.astype(
                    np.float16
                )
            )
            C_vectors.append(
                C_h.astype(
                    np.float16
                )
            )

            A_mass.append(A_m)
            B_mass.append(B_m)
            C_mass.append(C_m)

            sids.append(int(sid))
            labels.append(relation)
            selected_weights.append(
                selected_weight
            )
            transition_groups.append(
                group
            )

            del A_out
            del B_out
            del C_out
            del batch

            cleanup()

    N = len(sids)

    if N == 0:
        raise RuntimeError(
            "No samples processed."
        )

    A_acc = A_correct / N
    B_acc = B_correct / N
    C_acc = C_correct / N

    print("\n" + "=" * 150)
    print("ACTUAL GREEDY GENERATION REPRODUCTION CHECK")
    print("=" * 150)

    print(
        f"N={N}"
    )

    print(
        f"A original eps={args.base_rms_eps:g}, no AdaptVis : "
        f"{A_acc:.4f}"
    )

    print(
        f"B eps-only eps={args.enhanced_rms_eps:g}, no AdaptVis : "
        f"{B_acc:.4f} "
        f"({B_acc-A_acc:+.4f} vs A)"
    )

    print(
        f"C dynamic AdaptVis : "
        f"{C_acc:.4f} "
        f"({C_acc-B_acc:+.4f} vs B; "
        f"{C_acc-A_acc:+.4f} vs A)"
    )

    print(
        f"B -> C: W2C={w2c} "
        f"C2W={c2w} "
        f"C2C={c2c} "
        f"W2W={w2w} "
        f"net={w2c-c2w:+d}"
    )

    print(
        f"selected {args.weight1}: "
        f"{weight1_count} | "
        f"selected {args.weight2}: "
        f"{weight2_count}"
    )

    print("=" * 150)

    improved = (
        C_acc > B_acc
        or C_acc > A_acc
    )

    if improved:
        print(
            "[SUCCESS] This script reproduced an accuracy improvement."
        )
    else:
        print(
            "[WARNING] This script did not reproduce an improvement. "
            "Do not use its hidden-state deltas as the mechanism of the "
            "successful AdaptVis result."
        )

        if args.require_improvement:
            raise RuntimeError(
                "No generation improvement reproduced."
            )

    # ---------------------------------------------------------------------
    # Stack + probes.
    # ---------------------------------------------------------------------
    A_arr = np.stack(
        A_vectors,
        axis=0,
    )

    B_arr = np.stack(
        B_vectors,
        axis=0,
    )

    C_arr = np.stack(
        C_vectors,
        axis=0,
    )

    d_eps = (
        B_arr.astype(np.float32)
        - A_arr.astype(np.float32)
    )

    d_adapt = (
        C_arr.astype(np.float32)
        - B_arr.astype(np.float32)
    )

    d_total = (
        C_arr.astype(np.float32)
        - A_arr.astype(np.float32)
    )

    labels_arr = np.asarray(
        labels,
        dtype=object,
    )

    reps = {
        "A_last": A_arr,
        "B_last": B_arr,
        "C_last": C_arr,
        "delta_eps": d_eps,
        "delta_adapt": d_adapt,
        "delta_total": d_total,
    }

    probe_summary = run_probes(
        reps,
        labels_arr,
        probe_layers,
        train_ratio=args.train_ratio,
        repeats=args.repeats,
        seed=args.seed,
    )

    probe_lookup = {
        (
            r["representation"],
            int(r["layer"]),
        ): r
        for r in probe_summary
    }

    # ---------------------------------------------------------------------
    # Global layer summary.
    # ---------------------------------------------------------------------
    layer_summary = []

    for layer in probe_layers:
        rows = [
            r
            for r in geometry_rows
            if int(
                r["layer"]
            ) == int(layer)
        ]

        layer_summary.append({
            "layer": layer,
            "N": len(rows),

            "A_visual_mass": safe_mean(
                r["A_visual_mass"]
                for r in rows
            ),
            "B_visual_mass": safe_mean(
                r["B_visual_mass"]
                for r in rows
            ),
            "C_visual_mass": safe_mean(
                r["C_visual_mass"]
                for r in rows
            ),

            "delta_eps_visual_mass": safe_mean(
                r[
                    "delta_eps_visual_mass"
                ]
                for r in rows
            ),
            "delta_adapt_visual_mass": safe_mean(
                r[
                    "delta_adapt_visual_mass"
                ]
                for r in rows
            ),
            "delta_total_visual_mass": safe_mean(
                r[
                    "delta_total_visual_mass"
                ]
                for r in rows
            ),

            "delta_eps_norm": safe_mean(
                r["delta_eps_norm"]
                for r in rows
            ),
            "delta_adapt_norm": safe_mean(
                r["delta_adapt_norm"]
                for r in rows
            ),
            "delta_total_norm": safe_mean(
                r["delta_total_norm"]
                for r in rows
            ),

            "cos_A_C": safe_mean(
                r["cos_A_C"]
                for r in rows
            ),
            "cos_B_C": safe_mean(
                r["cos_B_C"]
                for r in rows
            ),

            "A_spatial_acc": probe_lookup[
                ("A_last", layer)
            ]["accuracy_mean"],
            "B_spatial_acc": probe_lookup[
                ("B_last", layer)
            ]["accuracy_mean"],
            "C_spatial_acc": probe_lookup[
                ("C_last", layer)
            ]["accuracy_mean"],

            "delta_eps_spatial_acc": (
                probe_lookup[
                    ("delta_eps", layer)
                ]["accuracy_mean"]
            ),

            "delta_adapt_spatial_acc": (
                probe_lookup[
                    ("delta_adapt", layer)
                ]["accuracy_mean"]
            ),

            "delta_total_spatial_acc": (
                probe_lookup[
                    ("delta_total", layer)
                ]["accuracy_mean"]
            ),
        })

    print("\n" + "=" * 190)
    print("HOW THE SUCCESSFUL ENHANCEMENT ACTS ON THE LAST TOKEN")
    print("=" * 190)

    print(
        f"{'layer':>5s} | "
        f"{'M_A':>7s} | "
        f"{'M_B':>7s} | "
        f"{'M_C':>7s} | "
        f"{'dM_adapt':>9s} | "
        f"{'||d_eps||':>9s} | "
        f"{'||d_adapt||':>11s} | "
        f"{'||d_total||':>11s} | "
        f"{'A_sp':>7s} | "
        f"{'B_sp':>7s} | "
        f"{'C_sp':>7s} | "
        f"{'dAdapt_sp':>9s} | "
        f"{'dTotal_sp':>9s}"
    )

    print("-" * 190)

    for row in layer_summary:
        print(
            f"L{int(row['layer']):02d}   | "
            f"{row['A_visual_mass']:7.4f} | "
            f"{row['B_visual_mass']:7.4f} | "
            f"{row['C_visual_mass']:7.4f} | "
            f"{row['delta_adapt_visual_mass']:+9.4f} | "
            f"{row['delta_eps_norm']:9.4f} | "
            f"{row['delta_adapt_norm']:11.4f} | "
            f"{row['delta_total_norm']:11.4f} | "
            f"{row['A_spatial_acc']:7.4f} | "
            f"{row['B_spatial_acc']:7.4f} | "
            f"{row['C_spatial_acc']:7.4f} | "
            f"{row['delta_adapt_spatial_acc']:9.4f} | "
            f"{row['delta_total_spatial_acc']:9.4f}"
        )

    print("=" * 190)

    # ---------------------------------------------------------------------
    # W2C / C2W grouping.
    # ---------------------------------------------------------------------
    grouped = []

    for group in [
        "W2C",
        "C2W",
        "C2C",
        "W2W",
    ]:
        for layer in probe_layers:
            rows = [
                r
                for r in geometry_rows
                if (
                    r["B_to_C_group"]
                    == group
                    and int(
                        r["layer"]
                    )
                    == int(layer)
                )
            ]

            if not rows:
                continue

            grouped.append({
                "group": group,
                "layer": layer,
                "N": len(rows),

                "B_visual_mass": safe_mean(
                    r["B_visual_mass"]
                    for r in rows
                ),

                "C_visual_mass": safe_mean(
                    r["C_visual_mass"]
                    for r in rows
                ),

                "delta_adapt_visual_mass": safe_mean(
                    r[
                        "delta_adapt_visual_mass"
                    ]
                    for r in rows
                ),

                "delta_adapt_norm": safe_mean(
                    r["delta_adapt_norm"]
                    for r in rows
                ),

                "cos_B_C": safe_mean(
                    r["cos_B_C"]
                    for r in rows
                ),
            })

    # Weight groups.
    weight_grouped = []

    for weight in [
        float(args.weight1),
        float(args.weight2),
    ]:
        for layer in probe_layers:
            rows = [
                r
                for r in geometry_rows
                if (
                    abs(
                        float(
                            r[
                                "selected_weight"
                            ]
                        )
                        - weight
                    )
                    < 1e-8
                    and int(
                        r["layer"]
                    )
                    == int(layer)
                )
            ]

            if not rows:
                continue

            weight_grouped.append({
                "selected_weight": weight,
                "layer": layer,
                "N": len(rows),

                "B_visual_mass": safe_mean(
                    r["B_visual_mass"]
                    for r in rows
                ),

                "C_visual_mass": safe_mean(
                    r["C_visual_mass"]
                    for r in rows
                ),

                "delta_adapt_visual_mass": safe_mean(
                    r[
                        "delta_adapt_visual_mass"
                    ]
                    for r in rows
                ),

                "delta_adapt_norm": safe_mean(
                    r["delta_adapt_norm"]
                    for r in rows
                ),
            })

    # ---------------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------------
    generation_summary = [{
        "N": N,

        "A_acc": A_acc,
        "B_acc": B_acc,
        "C_acc": C_acc,

        "eps_gain": (
            B_acc - A_acc
        ),
        "adaptvis_gain_over_B": (
            C_acc - B_acc
        ),
        "total_gain": (
            C_acc - A_acc
        ),

        "W2C": w2c,
        "C2W": c2w,
        "C2C": c2c,
        "W2W": w2w,
        "net": w2c - c2w,

        "weight1": args.weight1,
        "weight1_count": weight1_count,
        "weight2": args.weight2,
        "weight2_count": weight2_count,
        "threshold": args.threshold,

        "base_rms_eps": (
            args.base_rms_eps
        ),
        "enhanced_rms_eps": (
            args.enhanced_rms_eps
        ),

        "adaptvis_max_layers": (
            args.adaptvis_max_layers
        ),

        "transformers": (
            transformers.__version__
        ),

        "generation_backend": (
            "custom forward + past_key_values"
        ),
    }]

    write_csv(
        outdir
        / "generation_summary.csv",
        generation_summary,
    )

    write_csv(
        outdir
        / "generation_details.csv",
        generation_rows,
    )

    write_csv(
        outdir
        / "lasttoken_layer_summary.csv",
        layer_summary,
    )

    write_csv(
        outdir
        / "lasttoken_per_sample_layer.csv",
        geometry_rows,
    )

    write_csv(
        outdir
        / "lasttoken_transition_groups.csv",
        grouped,
    )

    write_csv(
        outdir
        / "lasttoken_weight_groups.csv",
        weight_grouped,
    )

    write_csv(
        outdir
        / "direction_probe_summary.csv",
        probe_summary,
    )

    if args.save_vectors:
        np.savez_compressed(
            outdir
            / "lasttoken_vectors_ABC.npz",

            sample_index=np.asarray(
                sids,
                dtype=np.int64,
            ),

            relation=labels_arr,

            layers=np.asarray(
                probe_layers,
                dtype=np.int32,
            ),

            selected_weight=np.asarray(
                selected_weights,
                dtype=np.float32,
            ),

            transition_group=np.asarray(
                transition_groups,
                dtype=object,
            ),

            A_last=A_arr,
            B_last=B_arr,
            C_last=C_arr,

            delta_eps=d_eps,
            delta_adapt=d_adapt,
            delta_total=d_total,

            A_visual_mass=np.stack(
                A_mass,
                axis=0,
            ),
            B_visual_mass=np.stack(
                B_mass,
                axis=0,
            ),
            C_visual_mass=np.stack(
                C_mass,
                axis=0,
            ),
        )

    metadata = {
        "dataset": args.dataset,
        "option": args.option,
        "transformers": (
            transformers.__version__
        ),
        "torch": torch.__version__,

        "generation_backend": (
            "custom LlavaForConditionalGenerationScal.forward "
            "+ past_key_values"
        ),

        "processor_path": (
            "tokenizer + image_processor separately; "
            "one legacy <image> placeholder"
        ),

        "dynamic_policy": {
            "weight1": args.weight1,
            "weight2": args.weight2,
            "threshold": args.threshold,
        },

        "base_rms_eps": (
            args.base_rms_eps
        ),
        "enhanced_rms_eps": (
            args.enhanced_rms_eps
        ),

        "adaptvis_max_layers": (
            args.adaptvis_max_layers
        ),

        "probe_layers": probe_layers,
    }

    (
        outdir
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[saved] "
        f"{outdir / 'generation_summary.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'lasttoken_layer_summary.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'lasttoken_transition_groups.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'direction_probe_summary.csv'}"
    )


if __name__ == "__main__":
    main()
