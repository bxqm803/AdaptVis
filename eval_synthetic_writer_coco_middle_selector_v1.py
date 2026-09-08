#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_synthetic_writer_coco_middle_selector_v1.py

Correct read-write transfer experiment.

SOURCE WRITER:
    Synthetic Shapes only.

TARGET READER / SELECTOR:
    COCO TRAIN only.

TARGET TEST:
    COCO held-out TEST only.

The experiment isolates whether the LATE causal direction transfers from the
synthetic shapes domain to COCO, while keeping the original target-domain
middle-layer Direction selector.

Pipeline
========

1. Synthetic source -> late writer directions

For every synthetic sample and late layer l:

    Delta_last(i,l)
        = h_last(real synthetic image)
        - h_last(gray synthetic image)

For relation r:

    mu_r,l = E[Delta_last | r]

    global_l = 1/4 sum_r mu_r,l

    s_syn(r,l) = mu_r,l - global_l


2. COCO TRAIN -> middle Direction selector

IMPORTANT: this uses the ORIGINAL middle guide construction:

    q(i,l)
      = [(h_sub - h_ref)_REAL]
        - [(h_sub - h_ref)_NOIMAGE]

The relation prototypes are fit from COCO TRAIN:

    center_l = E[q_train,l]

    d_target(r,l)
      = normalize(
          E[q_train,l | r]
          - center_l
        )


3. COCO TEST -> non-oracle routing

For each held-out COCO TEST sample:

    q_test,l
      = pair_REAL
        - pair_NOIMAGE

For every selected guide layer:

    score_l(r)
      = cosine(
          q_test,l - center_l,
          d_target(r,l)
        )

If multiple guide layers are used, average the cosine scores:

    score(r) = mean_l score_l(r)

Then:

    r_hat = argmax_r score(r)

NO TEST GT is used to choose r_hat.


4. Write ONLY the synthetic late direction

    h_last,l
      <- h_last,l
         + scale * s_syn(r_hat,l)

Then run actual greedy generation.


Outer split
===========
COCO is relation-stratified 30/70 by default.

COCO TRAIN GT:
    used to fit middle relation prototypes.

COCO TEST GT:
    used only to report selector / generation accuracy.
    It is NOT used for routing.

Synthetic source:
    used only to build late writer directions.
    It does NOT build the middle selector in this script.


Default guide layers
====================
qwen-3b:
    L19

qwen-7b:
    L14-L20, average cosine scores across layers

internvl-2b:
    TRAIN-only automatic layer search over middle 20%-75% depth,
    top 3 layers, then refit on all COCO TRAIN.
    (There is no previously established original InternVL middle-layer preset.)

Default late writer layers
==========================
qwen-3b:
    L32-L35

qwen-7b:
    L25-L27

internvl-2b:
    L21-L23


Example: Qwen3B
===============
CUDA_VISIBLE_DEVICES=0 python eval_synthetic_writer_coco_middle_selector_v1.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --train-frac 0.30 \
  --scale 1.0 \
  --output-dir output/syn_writer_coco_middle_qwen3b_v1 \
  --overwrite
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
# Repository imports
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
        "guide_mode": "fixed",
        "guide_layers": [19],
        "guide_top_k": 1,
    },
    "qwen-7b": {
        "actuator_layers": [25, 26, 27],
        "guide_mode": "fixed",
        "guide_layers": [14, 15, 16, 17, 18, 19, 20],
        "guide_top_k": 7,
    },
    "internvl-2b": {
        "actuator_layers": [21, 22, 23],
        "guide_mode": "search",
        "guide_layers": None,
        "guide_top_k": 3,
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
        default="preset",
        help="preset or e.g. 32-35",
    )

    p.add_argument(
        "--guide-layers",
        default="preset",
        help=(
            "preset uses original Qwen middle layers. "
            "For manual selection use e.g. 19 or 14-20. "
            "Use 'search' for TRAIN-only layer search."
        ),
    )

    p.add_argument(
        "--guide-top-k",
        default="preset",
        help=(
            "Number of selected guide layers whose cosine scores are averaged. "
            "preset = model-specific default."
        ),
    )

    p.add_argument(
        "--guide-search-layers",
        default="auto",
        help=(
            "Candidate layers when --guide-layers search. "
            "auto = 20%-75% of decoder depth."
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
        "--train-frac",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "last"],
        help="Object phrase token pooling, matching original middle guide code.",
    )

    p.add_argument(
        "--gray-value",
        type=int,
        default=128,
    )

    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
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
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


# =============================================================================
# Generic utilities
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
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
    norm = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(norm, EPS)


def parse_layers(
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
                f"L{layer} outside 0..{n_layers - 1}"
            )

        if layer not in result:
            result.append(layer)

    if not result:
        raise ValueError("Empty layer selection.")

    return result


def auto_middle_layers(
    n_layers: int,
) -> List[int]:
    lo = int(round(
        0.20
        * (n_layers - 1)
    ))

    hi = int(round(
        0.75
        * (n_layers - 1)
    ))

    return list(
        range(
            max(0, lo),
            min(n_layers - 1, hi) + 1,
        )
    )


# =============================================================================
# Data loading
# =============================================================================

def load_synthetic_records(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    root = Path(
        args.synthetic_dir
    )

    labels_path = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            labels_path
        )

    rows = []

    with labels_path.open(
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

            item = json.loads(
                line
            )

            raw_relation = str(
                item["relation"]
            ).strip().lower()

            if raw_relation not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: "
                    f"bad relation={raw_relation!r}"
                )

            relation = SYN_REL_MAP[
                raw_relation
            ]

            subject = str(
                item["subject"]
            ).strip()

            reference = str(
                item["reference"]
            ).strip()

            image_value = Path(
                str(
                    item["image"]
                )
            )

            image_path = (
                image_value
                if image_value.is_absolute()
                else root / image_value
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    image_path
                )

            # Keep the same prompt used by the successful synthetic->COCO
            # oracle experiment.
            question_text = (
                f"Where is the {subject} relative to the {reference}? "
                "Answer with left, right, above, or below."
            )

            rows.append({
                "sid": int(
                    item.get(
                        "id",
                        len(rows),
                    )
                ),
                "relation": relation,
                "subject": subject,
                "reference": reference,
                "question_text": question_text,
                "image_path": str(
                    image_path
                ),
            })

    rows.sort(
        key=lambda row: int(
            row["sid"]
        )
    )

    if args.source_max_samples is not None:
        rows = rows[
            : int(
                args.source_max_samples
            )
        ]

    if not rows:
        raise RuntimeError(
            "Synthetic source is empty."
        )

    return rows


def load_coco_records(
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, Any]],
    Any,
]:
    raw_records, audit = twoobj.load_records(
        "coco_two",
        Path(
            args.data_root
        ),
        args.target_max_samples,
    )

    prompts = cent.load_standard_prompts(
        Path(
            args.prompt_jsonl
        )
    )

    rows = []

    for raw in raw_records:
        sid = int(
            raw.sid
        )

        if sid not in prompts:
            raise RuntimeError(
                f"sid={sid} missing COCO prompt."
            )

        prompt = prompts[
            sid
        ]

        relation = cent.normalize_relation(
            prompt[
                "answer_raw"
            ]
        )

        if relation not in REL_TO_ID:
            continue

        rows.append({
            "sid": sid,
            "relation": relation,
            "subject": str(
                prompt[
                    "subject"
                ]
            ),
            "reference": str(
                prompt[
                    "reference"
                ]
            ),
            "question_text": str(
                prompt[
                    "question_text"
                ]
            ),
            "raw_record": raw,
        })

    rows.sort(
        key=lambda row: int(
            row["sid"]
        )
    )

    if not rows:
        raise RuntimeError(
            "COCO target is empty."
        )

    return (
        rows,
        audit,
    )


def open_synthetic_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return Image.open(
        record[
            "image_path"
        ]
    ).convert(
        "RGB"
    )


def open_coco_image(
    record: Mapping[str, Any],
) -> Image.Image:
    return cent.record_image(
        record[
            "raw_record"
        ]
    )


# =============================================================================
# Model loading -- explicit single GPU, no Accelerate device_map
# =============================================================================

def resolve_dtype(
    requested: str,
    spec: Any,
) -> torch.dtype:
    name = (
        str(
            spec.dtype_name
        )
        if requested
        == "auto"
        else requested
    )

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype={name!r}"
        )

    return mapping[
        name
    ]


def load_model(
    args: argparse.Namespace,
):
    device = torch.device(
        args.device
    )

    print("\n" + "=" * 120)
    print("RUNTIME DEVICE CHECK")
    print("=" * 120)
    print(
        f"torch={torch.__version__} | "
        f"torch_cuda={torch.version.cuda} | "
        f"cuda_available={torch.cuda.is_available()} | "
        f"cuda_count={torch.cuda.device_count()} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        index = (
            0
            if device.index is None
            else int(
                device.index
            )
        )

        torch.cuda.set_device(
            index
        )

        probe = torch.empty(
            1,
            device=device,
        )

        del probe

        print(
            f"GPU={torch.cuda.get_device_name(index)}"
        )

    specs = cent.merged_model_specs(
        twoobj
    )

    if args.model not in specs:
        raise ValueError(
            f"Unavailable model={args.model}; "
            f"aliases={sorted(specs)}"
        )

    spec = specs[
        args.model
    ]

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )

    if model_cls is None:
        raise RuntimeError(
            f"No transformers class "
            f"{spec.model_class}"
        )

    dtype = resolve_dtype(
        args.dtype,
        spec,
    )

    kwargs: Dict[
        str,
        Any,
    ] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(
            spec.trust_remote_code
        ),
    }

    print(
        f"[load] {spec.repo_id} | "
        f"dtype={dtype} | "
        f"explicit -> {device}"
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

    model = model.to(
        device
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

    print(
        f"[load] decoder={decoder_path} | "
        f"n_layers={len(layers)} | "
        f"first_param={next(model.parameters()).device}"
    )

    print("=" * 120)

    return (
        model,
        processor,
        layers,
        decoder_path,
        spec,
    )


# =============================================================================
# Prompt batches / object token spans
# =============================================================================

def move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return {
        key: (
            value.to(
                device
            )
            if torch.is_tensor(
                value
            )
            else value
        )
        for key, value
        in batch.items()
    }


def build_real_batch(
    processor: Any,
    image: Image.Image,
    question_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return cent.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
        device=torch.device(
            args.device
        ),
    )


def build_noimage_batch(
    processor: Any,
    question_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    ORIGINAL middle control: text-only / NoImage.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question_text,
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
        prompt = question_text

    last_error = None

    attempts = [
        lambda: processor(
            text=[
                prompt
            ],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            return_tensors="pt",
        ),
    ]

    for fn in attempts:
        try:
            batch = fn()
            return move_batch(
                batch,
                torch.device(
                    args.device
                ),
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"NoImage processor failed: "
        f"{type(last_error).__name__}: "
        f"{last_error}"
    )


def find_all_subsequences(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> List[int]:
    if not pattern:
        return []

    n = len(
        pattern
    )

    return [
        start
        for start
        in range(
            0,
            len(
                sequence
            )
            - n
            + 1,
        )
        if list(
            sequence[
                start:
                start + n
            ]
        )
        == list(
            pattern
        )
    ]


def phrase_spans(
    tokenizer: Any,
    full_ids: Sequence[int],
    phrase: str,
) -> List[
    List[int]
]:
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
                    start
                    + len(
                        ids
                    ),
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


def locate_pair_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    subject: str,
    reference: str,
) -> Tuple[
    List[int],
    List[int],
]:
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

    if (
        not subject_spans
        or not reference_spans
    ):
        raise RuntimeError(
            f"Could not locate subject/reference tokens: "
            f"{subject!r} / {reference!r}"
        )

    best = None

    for s_span in subject_spans:
        for r_span in reference_spans:
            if (
                set(
                    s_span
                )
                & set(
                    r_span
                )
            ):
                continue

            # Prefer subject occurrence before reference, then nearest pair.
            order_penalty = int(
                s_span[
                    0
                ]
                >= r_span[
                    0
                ]
            )

            distance = abs(
                float(
                    np.mean(
                        s_span
                    )
                )
                - float(
                    np.mean(
                        r_span
                    )
                )
            )

            score = (
                order_penalty,
                distance,
                -s_span[
                    0
                ],
            )

            if (
                best is None
                or score
                < best[
                    0
                ]
            ):
                best = (
                    score,
                    s_span,
                    r_span,
                )

    if best is None:
        raise RuntimeError(
            "No non-overlapping object-token spans."
        )

    return (
        best[
            1
        ],
        best[
            2
        ],
    )


def batch_spans(
    processor: Any,
    batch: Mapping[str, Any],
    record: Mapping[str, Any],
) -> Tuple[
    List[int],
    List[int],
]:
    ids = (
        batch[
            "input_ids"
        ][
            0
        ]
        .detach()
        .cpu()
        .tolist()
    )

    return locate_pair_spans(
        processor.tokenizer,
        ids,
        record[
            "subject"
        ],
        record[
            "reference"
        ],
    )


# =============================================================================
# Middle capture: copied conceptually from original Direction implementation
# =============================================================================

class CaptureMiddle:
    def __init__(
        self,
        layers: Sequence[Any],
        selected_layers: Sequence[int],
    ):
        self.handles = []
        self.states: Dict[
            int,
            torch.Tensor,
        ] = {}

        for layer in selected_layers:
            layer = int(
                layer
            )

            self.handles.append(
                layers[
                    layer
                ].register_forward_hook(
                    self._make_hook(
                        layer
                    )
                )
            )

    def _make_hook(
        self,
        layer: int,
    ):
        def hook(
            _module,
            _inputs,
            output,
        ):
            hidden, _ = causal.extract_hidden(
                output
            )

            if (
                torch.is_tensor(
                    hidden
                )
                and hidden.ndim
                == 3
            ):
                self.states[
                    layer
                ] = hidden.detach()

            return output

        return hook

    def validate(
        self,
        selected_layers: Sequence[int],
    ) -> None:
        missing = [
            int(
                layer
            )
            for layer
            in selected_layers
            if int(
                layer
            )
            not in self.states
        ]

        if missing:
            raise RuntimeError(
                f"Missing middle captures: {missing}"
            )

    def close(
        self,
    ) -> None:
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

        self.handles = []

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        *_,
    ):
        self.close()


def pool_hidden(
    hidden_2d: torch.Tensor,
    positions: Sequence[int],
    mode: str,
) -> torch.Tensor:
    pos = [
        int(
            value
        )
        for value
        in positions
        if (
            0
            <= int(
                value
            )
            < int(
                hidden_2d.shape[
                    0
                ]
            )
        )
    ]

    if not pos:
        raise RuntimeError(
            "No valid object-token positions."
        )

    if mode == "last":
        return hidden_2d[
            pos[
                -1
            ]
        ]

    idx = torch.as_tensor(
        pos,
        device=hidden_2d.device,
        dtype=torch.long,
    )

    return hidden_2d.index_select(
        0,
        idx,
    ).mean(
        dim=0
    )


def pair_state(
    hidden: torch.Tensor,
    subject_span: Sequence[int],
    reference_span: Sequence[int],
    pool: str,
) -> np.ndarray:
    h = hidden[
        0
    ]

    subject = pool_hidden(
        h,
        subject_span,
        pool,
    )

    reference = pool_hidden(
        h,
        reference_span,
        pool,
    )

    return (
        subject
        - reference
    ).detach().float().cpu().numpy().astype(
        np.float32
    )


def capture_pair_states_from_batch(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    batch: Mapping[str, Any],
    record: Mapping[str, Any],
    guide_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[
    int,
    np.ndarray,
]:
    subject_span, reference_span = batch_spans(
        processor,
        batch,
        record,
    )

    with CaptureMiddle(
        layers,
        guide_layers,
    ) as capture:
        with torch.inference_mode():
            model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

        capture.validate(
            guide_layers
        )

        result = {
            int(
                layer
            ): pair_state(
                capture.states[
                    int(
                        layer
                    )
                ],
                subject_span,
                reference_span,
                args.pool,
            )
            for layer
            in guide_layers
        }

    return result


def real_noimage_middle_q(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    record: Mapping[str, Any],
    guide_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[
    int,
    np.ndarray,
]:
    """
    ORIGINAL middle Direction feature:

        q_l
          = (h_sub - h_ref)_REAL
            - (h_sub - h_ref)_NOIMAGE
    """
    real_batch = build_real_batch(
        processor,
        image,
        record[
            "question_text"
        ],
        args,
    )

    noimage_batch = build_noimage_batch(
        processor,
        record[
            "question_text"
        ],
        args,
    )

    try:
        real_pair = capture_pair_states_from_batch(
            model,
            processor,
            layers,
            real_batch,
            record,
            guide_layers,
            args,
        )

        noimage_pair = capture_pair_states_from_batch(
            model,
            processor,
            layers,
            noimage_batch,
            record,
            guide_layers,
            args,
        )

        return {
            int(
                layer
            ): (
                real_pair[
                    int(
                        layer
                    )
                ]
                - noimage_pair[
                    int(
                        layer
                    )
                ]
            ).astype(
                np.float32
            )
            for layer
            in guide_layers
        }

    finally:
        del real_batch
        del noimage_batch


# =============================================================================
# Synthetic late writer extraction
# =============================================================================

def capture_synthetic_last_states(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Image.Image,
    record: Mapping[str, Any],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[
    int,
    np.ndarray,
]:
    batch = build_real_batch(
        processor,
        image,
        record[
            "question_text"
        ],
        args,
    )

    # Keep the exact successful late-causal capture path from the previous
    # synthetic->COCO oracle experiment.
    with causal.CaptureStates(
        layers,
        list(
            actuator_layers
        ),
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

        last_states = {
            int(
                layer
            ): np.asarray(
                value,
                dtype=np.float32,
            )
            for layer, value
            in capture.last_states.items()
        }

    del batch

    missing = [
        int(
            layer
        )
        for layer
        in actuator_layers
        if int(
            layer
        )
        not in last_states
    ]

    if missing:
        raise RuntimeError(
            f"Missing late captures={missing}"
        )

    return last_states


def fit_synthetic_writer(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    source_records: Sequence[
        Mapping[str, Any]
    ],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[str, int],
]:
    bags = {
        int(
            layer
        ): {
            relation: []
            for relation
            in RELATIONS
        }
        for layer
        in actuator_layers
    }

    counts = Counter()

    for record in tqdm(
        source_records,
        desc=f"SOURCE synthetic late writer:{args.model}",
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

            real_last = capture_synthetic_last_states(
                model,
                processor,
                layers,
                real,
                record,
                actuator_layers,
                args,
            )

            gray_last = capture_synthetic_last_states(
                model,
                processor,
                layers,
                gray,
                record,
                actuator_layers,
                args,
            )

            relation = record[
                "relation"
            ]

            counts[
                relation
            ] += 1

            for layer in actuator_layers:
                bags[
                    int(
                        layer
                    )
                ][
                    relation
                ].append(
                    (
                        real_last[
                            int(
                                layer
                            )
                        ]
                        - gray_last[
                            int(
                                layer
                            )
                        ]
                    ).astype(
                        np.float32
                    )
                )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    templates: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for layer in actuator_layers:
        layer = int(
            layer
        )

        relation_mean = {}

        for relation in RELATIONS:
            if not bags[
                layer
            ][
                relation
            ]:
                raise RuntimeError(
                    f"No synthetic writer samples "
                    f"L{layer} relation={relation}"
                )

            relation_mean[
                relation
            ] = (
                np.stack(
                    bags[
                        layer
                    ][
                        relation
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

        global_mean = (
            np.stack(
                [
                    relation_mean[
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

        templates[
            layer
        ] = {
            "global": global_mean,
            "relation_mean": relation_mean,
            "shared": {
                relation: (
                    relation_mean[
                        relation
                    ]
                    - global_mean
                ).astype(
                    np.float32
                )
                for relation
                in RELATIONS
            },
        }

    return (
        templates,
        {
            relation: int(
                counts[
                    relation
                ]
            )
            for relation
            in RELATIONS
        },
    )


# =============================================================================
# COCO outer split
# =============================================================================

def stratified_split_records(
    records: Sequence[
        Mapping[str, Any]
    ],
    train_frac: float,
    seed: int,
) -> Tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    rng = random.Random(
        int(
            seed
        )
    )

    train = []
    test = []

    for relation in RELATIONS:
        subset = [
            record
            for record
            in records
            if record[
                "relation"
            ]
            == relation
        ]

        rng.shuffle(
            subset
        )

        if len(
            subset
        ) < 2:
            raise RuntimeError(
                f"Too few COCO samples for {relation}"
            )

        n_train = int(
            round(
                len(
                    subset
                )
                * float(
                    train_frac
                )
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(
                    subset
                )
                - 1,
            ),
        )

        train.extend(
            subset[
                :n_train
            ]
        )

        test.extend(
            subset[
                n_train:
            ]
        )

    rng.shuffle(
        train
    )

    rng.shuffle(
        test
    )

    return (
        train,
        test,
    )


# =============================================================================
# COCO TRAIN middle features / prototypes
# =============================================================================

def collect_coco_middle_features(
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    records: Sequence[
        Mapping[str, Any]
    ],
    guide_layers: Sequence[int],
    args: argparse.Namespace,
    desc: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, Any]],
]:
    features = []
    labels = []
    sids = []
    errors = []

    for record in tqdm(
        records,
        desc=desc,
    ):
        image = None

        try:
            image = open_coco_image(
                record
            )

            q = real_noimage_middle_q(
                model,
                processor,
                layers,
                image,
                record,
                guide_layers,
                args,
            )

            features.append(
                np.stack(
                    [
                        q[
                            int(
                                layer
                            )
                        ]
                        for layer
                        in guide_layers
                    ],
                    axis=0,
                ).astype(
                    np.float32
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
                f"[MIDDLE ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append({
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

    if not features:
        raise RuntimeError(
            "No middle features extracted."
        )

    return (
        np.stack(
            features,
            axis=0,
        ),
        np.asarray(
            labels,
            dtype=object,
        ),
        np.asarray(
            sids,
            dtype=np.int64,
        ),
        errors,
    )


def fit_layer_prototype(
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
        rel_idx = indices[
            labels[
                indices
            ]
            == relation
        ]

        if len(
            rel_idx
        ) == 0:
            raise RuntimeError(
                f"No TRAIN examples for {relation}"
            )

        mu = np.asarray(
            X[
                rel_idx
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
                f"Near-zero middle direction "
                f"for {relation}"
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


def layer_scores(
    X: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    q = (
        np.asarray(
            X,
            dtype=np.float64,
        )
        - np.asarray(
            center,
            dtype=np.float64,
        )
    )

    q = normalize_rows(
        q
    )

    d = normalize_rows(
        directions
    )

    return (
        q
        @ d.T
    )


def stratified_indices(
    labels: np.ndarray,
    fit_frac: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    rng = random.Random(
        int(
            seed
        )
    )

    fit = []
    val = []

    for relation in RELATIONS:
        ids = np.flatnonzero(
            labels
            == relation
        ).tolist()

        if len(
            ids
        ) < 2:
            raise RuntimeError(
                f"Need >=2 TRAIN samples "
                f"for CV relation={relation}"
            )

        rng.shuffle(
            ids
        )

        n_fit = int(
            round(
                len(
                    ids
                )
                * float(
                    fit_frac
                )
            )
        )

        n_fit = max(
            1,
            min(
                n_fit,
                len(
                    ids
                )
                - 1,
            ),
        )

        fit.extend(
            ids[
                :n_fit
            ]
        )

        val.extend(
            ids[
                n_fit:
            ]
        )

    rng.shuffle(
        fit
    )

    rng.shuffle(
        val
    )

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


def choose_top_layers_train_only(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    candidate_layers: Sequence[int],
    top_k: int,
    repeats: int,
    fit_frac: float,
    seed: int,
) -> Tuple[
    List[int],
    List[Dict[str, Any]],
]:
    """
    Only used for --guide-layers search.

    candidate_layers corresponds to train_features axis=1.
    """
    rows = []

    for repeat in range(
        repeats
    ):
        fit_idx, val_idx = stratified_indices(
            train_labels,
            fit_frac,
            seed + 1000 + repeat,
        )

        gt = np.asarray(
            [
                REL_TO_ID[
                    str(
                        train_labels[
                            i
                        ]
                    )
                ]
                for i
                in val_idx
            ],
            dtype=np.int64,
        )

        for lp, layer in enumerate(
            candidate_layers
        ):
            center, directions = fit_layer_prototype(
                train_features[
                    :,
                    lp,
                    :,
                ],
                train_labels,
                fit_idx,
            )

            scores = layer_scores(
                train_features[
                    val_idx,
                    lp,
                    :,
                ],
                center,
                directions,
            )

            pred = np.argmax(
                scores,
                axis=1,
            )

            rows.append({
                "repeat": repeat,
                "layer": int(
                    layer
                ),
                "accuracy": float(
                    np.mean(
                        pred
                        == gt
                    )
                ),
            })

    summary = []

    for layer in candidate_layers:
        subset = [
            row
            for row
            in rows
            if int(
                row[
                    "layer"
                ]
            )
            == int(
                layer
            )
        ]

        summary.append({
            "layer": int(
                layer
            ),
            "accuracy_mean": safe_mean(
                row[
                    "accuracy"
                ]
                for row
                in subset
            ),
            "accuracy_std": safe_std(
                row[
                    "accuracy"
                ]
                for row
                in subset
            ),
        })

    summary.sort(
        key=lambda row: (
            -float(
                row[
                    "accuracy_mean"
                ]
            ),
            int(
                row[
                    "layer"
                ]
            ),
        )
    )

    selected = [
        int(
            row[
                "layer"
            ]
        )
        for row
        in summary[
            : int(
                top_k
            )
        ]
    ]

    return (
        selected,
        summary,
    )


def fit_multilayer_selector(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    feature_layers: Sequence[int],
    selected_layers: Sequence[int],
) -> Dict[
    int,
    Dict[str, np.ndarray],
]:
    layer_to_pos = {
        int(
            layer
        ): pos
        for pos, layer
        in enumerate(
            feature_layers
        )
    }

    all_idx = np.arange(
        len(
            train_labels
        ),
        dtype=np.int64,
    )

    selector = {}

    for layer in selected_layers:
        layer = int(
            layer
        )

        pos = layer_to_pos[
            layer
        ]

        center, directions = fit_layer_prototype(
            train_features[
                :,
                pos,
                :,
            ],
            train_labels,
            all_idx,
        )

        selector[
            layer
        ] = {
            "center": center,
            "directions": directions,
        }

    return selector


def predict_multilayer_selector(
    q_by_layer: Mapping[
        int,
        np.ndarray,
    ],
    selector: Mapping[
        int,
        Mapping[
            str,
            np.ndarray,
        ],
    ],
) -> Tuple[
    str,
    np.ndarray,
    float,
]:
    all_scores = []

    for layer, params in selector.items():
        q = np.asarray(
            q_by_layer[
                int(
                    layer
                )
            ],
            dtype=np.float32,
        )

        scores = layer_scores(
            q[
                None,
                :,
            ],
            params[
                "center"
            ],
            params[
                "directions"
            ],
        )[
            0
        ]

        all_scores.append(
            scores
        )

    mean_scores = np.stack(
        all_scores,
        axis=0,
    ).mean(
        axis=0
    )

    order = np.argsort(
        mean_scores
    )[
        ::-1
    ]

    pred_id = int(
        order[
            0
        ]
    )

    margin = float(
        mean_scores[
            order[
                0
            ]
        ]
        - mean_scores[
            order[
                1
            ]
        ]
    )

    return (
        ID_TO_REL[
            pred_id
        ],
        mean_scores.astype(
            np.float32
        ),
        margin,
    )


# =============================================================================
# Generation
# =============================================================================

def baseline_generate(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[
    str,
    Optional[str],
]:
    batch = build_real_batch(
        processor,
        image,
        record[
            "question_text"
        ],
        args,
    )

    text = cent.generate_text(
        model,
        processor,
        batch,
        args.max_new_tokens,
    )

    del batch

    return (
        text,
        cent.normalize_relation(
            text
        ),
    )


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
    predicted_relation: str,
    args: argparse.Namespace,
) -> Tuple[
    str,
    Optional[str],
]:
    batch = build_real_batch(
        processor,
        image,
        record[
            "question_text"
        ],
        args,
    )

    with causal.SteerLast(
        layers,
        writer_templates,
        list(
            actuator_layers
        ),
        predicted_relation,
        args.scale,
        "add",
        None,
    ):
        text = cent.generate_text(
            model,
            processor,
            batch,
            args.max_new_tokens,
        )

    del batch

    return (
        text,
        cent.normalize_relation(
            text
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if not (
        0.0
        < args.train_frac
        < 1.0
    ):
        raise ValueError(
            "--train-frac must be in (0,1)"
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

    coco_records, coco_audit = load_coco_records(
        args
    )

    train_records, test_records = stratified_split_records(
        coco_records,
        args.train_frac,
        args.seed,
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

    preset = MODEL_PRESETS[
        args.model
    ]

    # ---------------------------------------------------------------------
    # Actuator layers.
    # ---------------------------------------------------------------------
    if (
        str(
            args.actuator_layers
        )
        .strip()
        .lower()
        == "preset"
    ):
        actuator_layers = list(
            preset[
                "actuator_layers"
            ]
        )
    else:
        actuator_layers = parse_layers(
            args.actuator_layers,
            n_layers,
        )

    for layer in actuator_layers:
        if not (
            0
            <= int(
                layer
            )
            < n_layers
        ):
            raise RuntimeError(
                f"Invalid actuator L{layer} "
                f"for n_layers={n_layers}"
            )

    # ---------------------------------------------------------------------
    # Guide mode / layers.
    # ---------------------------------------------------------------------
    guide_arg = str(
        args.guide_layers
    ).strip().lower()

    if guide_arg == "preset":
        guide_mode = str(
            preset[
                "guide_mode"
            ]
        )

        if guide_mode == "fixed":
            feature_guide_layers = list(
                preset[
                    "guide_layers"
                ]
            )
        else:
            feature_guide_layers = auto_middle_layers(
                n_layers
            )

    elif guide_arg == "search":
        guide_mode = "search"

        if (
            str(
                args.guide_search_layers
            )
            .strip()
            .lower()
            == "auto"
        ):
            feature_guide_layers = auto_middle_layers(
                n_layers
            )
        else:
            feature_guide_layers = parse_layers(
                args.guide_search_layers,
                n_layers,
            )

    else:
        guide_mode = "fixed"

        feature_guide_layers = parse_layers(
            args.guide_layers,
            n_layers,
        )

    if (
        str(
            args.guide_top_k
        )
        .strip()
        .lower()
        == "preset"
    ):
        guide_top_k = int(
            preset[
                "guide_top_k"
            ]
        )
    else:
        guide_top_k = int(
            args.guide_top_k
        )

    guide_top_k = max(
        1,
        min(
            guide_top_k,
            len(
                feature_guide_layers
            ),
        ),
    )

    print("\n" + "=" * 160)
    print("SYNTHETIC LATE WRITER + ORIGINAL COCO MIDDLE DIRECTION SELECTOR")
    print("=" * 160)

    print(
        f"model={args.model} | "
        f"repo_id={spec.repo_id}"
    )

    print(
        f"COCO total={len(coco_records)} | "
        f"TRAIN={len(train_records)} | "
        f"TEST={len(test_records)}"
    )

    print(
        f"Synthetic source={len(source_records)} | "
        f"late writer layers={actuator_layers}"
    )

    print(
        f"middle control=REAL-NOIMAGE | "
        f"guide_mode={guide_mode} | "
        f"feature layers={feature_guide_layers}"
    )

    print(
        "COCO TRAIN GT fits the middle selector; "
        "synthetic data fits the writer; "
        "COCO TEST GT is NOT used for routing."
    )

    print("=" * 160)

    # ---------------------------------------------------------------------
    # 1. Synthetic late writer ONLY.
    # ---------------------------------------------------------------------
    (
        writer_templates,
        writer_counts,
    ) = fit_synthetic_writer(
        model,
        processor,
        layers,
        source_records,
        actuator_layers,
        args,
    )

    print("\n" + "=" * 160)
    print("SYNTHETIC LATE WRITER")
    print("=" * 160)

    print(
        f"source counts={writer_counts}"
    )

    for layer in actuator_layers:
        print(
            f"L{layer:02d} | "
            + " | ".join(
                f"{relation}="
                f"{np.linalg.norm(writer_templates[layer]['shared'][relation]):.4f}"
                for relation
                in RELATIONS
            )
        )

    print("=" * 160)

    # ---------------------------------------------------------------------
    # 2. COCO TRAIN middle Real-NoImage features.
    # ---------------------------------------------------------------------
    (
        train_features,
        train_labels,
        train_sids,
        train_errors,
    ) = collect_coco_middle_features(
        model,
        processor,
        layers,
        train_records,
        feature_guide_layers,
        args,
        desc=f"COCO TRAIN middle Real-NoImage:{args.model}",
    )

    if train_errors:
        write_csv(
            outdir
            / "train_middle_errors.csv",
            train_errors,
        )

    # ---------------------------------------------------------------------
    # 3. Choose / freeze guide layer(s) using COCO TRAIN only.
    # ---------------------------------------------------------------------
    layer_search_summary: List[
        Dict[str, Any]
    ] = []

    if guide_mode == "search":
        (
            selected_guide_layers,
            layer_search_summary,
        ) = choose_top_layers_train_only(
            train_features,
            train_labels,
            feature_guide_layers,
            guide_top_k,
            args.guide_cv_repeats,
            args.guide_cv_fit_frac,
            args.seed,
        )

        write_csv(
            outdir
            / "coco_train_middle_layer_search.csv",
            layer_search_summary,
        )

    else:
        selected_guide_layers = list(
            feature_guide_layers
        )

    selector = fit_multilayer_selector(
        train_features,
        train_labels,
        feature_guide_layers,
        selected_guide_layers,
    )

    print("\n" + "=" * 160)
    print("COCO TRAIN MIDDLE SELECTOR")
    print("=" * 160)

    print(
        f"selected guide layers="
        f"{selected_guide_layers}"
    )

    print(
        f"aggregation="
        f"{'single cosine' if len(selected_guide_layers)==1 else 'mean cosine scores across layers'}"
    )

    print(
        "Feature:"
    )

    print(
        "  q_l = (h_sub-h_ref)_REAL "
        "- (h_sub-h_ref)_NOIMAGE"
    )

    print(
        "Prototype source: COCO TRAIN only."
    )

    print("=" * 160)

    # Save vectors before TEST.
    arrays: Dict[
        str,
        Any,
    ] = {
        "relation_order": np.asarray(
            RELATIONS,
            dtype=object,
        ),
        "actuator_layers": np.asarray(
            actuator_layers,
            dtype=np.int32,
        ),
        "guide_layers": np.asarray(
            selected_guide_layers,
            dtype=np.int32,
        ),
    }

    for layer in actuator_layers:
        for relation in RELATIONS:
            arrays[
                f"synthetic_writer_L{layer}_{relation}"
            ] = writer_templates[
                layer
            ][
                "shared"
            ][
                relation
            ]

    for layer in selected_guide_layers:
        arrays[
            f"coco_middle_center_L{layer}"
        ] = selector[
            layer
        ][
            "center"
        ]

        arrays[
            f"coco_middle_directions_L{layer}"
        ] = selector[
            layer
        ][
            "directions"
        ]

    np.savez_compressed(
        outdir
        / "synthetic_writer_coco_middle_selector.npz",
        **arrays,
    )

    # TRAIN features no longer needed.
    del train_features
    cleanup()

    # ---------------------------------------------------------------------
    # 4. Held-out COCO TEST actual generation.
    # ---------------------------------------------------------------------
    rows = []

    for record in tqdm(
        test_records,
        desc=f"COCO TEST middle->synthetic writer:{args.model}",
    ):
        image = None

        try:
            image = open_coco_image(
                record
            )

            gt = record[
                "relation"
            ]

            # Baseline actual greedy generation.
            (
                baseline_text,
                baseline_pred,
            ) = baseline_generate(
                model,
                processor,
                image,
                record,
                args,
            )

            baseline_ok = (
                baseline_pred
                == gt
            )

            # ORIGINAL target-domain middle reader.
            q_by_layer = real_noimage_middle_q(
                model,
                processor,
                layers,
                image,
                record,
                selected_guide_layers,
                args,
            )

            (
                selector_pred,
                selector_scores,
                selector_margin,
            ) = predict_multilayer_selector(
                q_by_layer,
                selector,
            )

            selector_ok = (
                selector_pred
                == gt
            )

            # ONLY the writer is transferred from synthetic.
            (
                steered_text,
                steered_pred,
            ) = steer_generate(
                model,
                processor,
                layers,
                image,
                record,
                writer_templates,
                actuator_layers,
                selector_pred,
                args,
            )

            steered_ok = (
                steered_pred
                == gt
            )

            transition = classify_transition(
                baseline_ok,
                steered_ok,
            )

            rows.append({
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

                "selector_pred": selector_pred,
                "selector_correct": int(
                    selector_ok
                ),
                "selector_margin": selector_margin,
                "selector_score_left": float(
                    selector_scores[
                        REL_TO_ID[
                            "left"
                        ]
                    ]
                ),
                "selector_score_right": float(
                    selector_scores[
                        REL_TO_ID[
                            "right"
                        ]
                    ]
                ),
                "selector_score_above": float(
                    selector_scores[
                        REL_TO_ID[
                            "above"
                        ]
                    ]
                ),
                "selector_score_below": float(
                    selector_scores[
                        REL_TO_ID[
                            "below"
                        ]
                    ]
                ),

                "steered_text": steered_text,
                "steered_pred": steered_pred or "",
                "steered_correct": int(
                    steered_ok
                ),

                "transition": transition,
                "W2C": int(
                    transition
                    == "W2C"
                ),
                "C2W": int(
                    transition
                    == "C2W"
                ),
            })

        except Exception as exc:
            tqdm.write(
                f"[TEST ERROR sid={record['sid']}] "
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

    good = [
        row
        for row in rows
        if "error" not in row
    ]

    if not good:
        raise RuntimeError(
            "No successful TEST samples."
        )

    baseline_acc = safe_mean(
        row[
            "baseline_correct"
        ]
        for row in good
    )

    selector_acc = safe_mean(
        row[
            "selector_correct"
        ]
        for row in good
    )

    steered_acc = safe_mean(
        row[
            "steered_correct"
        ]
        for row in good
    )

    W2C = int(
        sum(
            row[
                "W2C"
            ]
            for row
            in good
        )
    )

    C2W = int(
        sum(
            row[
                "C2W"
            ]
            for row
            in good
        )
    )

    selector_correct_rows = [
        row
        for row in good
        if int(
            row[
                "selector_correct"
            ]
        )
        == 1
    ]

    selector_wrong_rows = [
        row
        for row in good
        if int(
            row[
                "selector_correct"
            ]
        )
        == 0
    ]

    baseline_wrong_rows = [
        row
        for row in good
        if int(
            row[
                "baseline_correct"
            ]
        )
        == 0
    ]

    baseline_correct_rows = [
        row
        for row in good
        if int(
            row[
                "baseline_correct"
            ]
        )
        == 1
    ]

    summary = {
        "model": args.model,
        "N_train": len(
            train_records
        ),
        "N_test": len(
            good
        ),
        "synthetic_source_N": len(
            source_records
        ),
        "guide_layers": ",".join(
            str(
                layer
            )
            for layer
            in selected_guide_layers
        ),
        "actuator_layers": ",".join(
            str(
                layer
            )
            for layer
            in actuator_layers
        ),
        "baseline_acc": baseline_acc,
        "middle_selector_acc": selector_acc,
        "steered_acc": steered_acc,
        "gain": (
            steered_acc
            - baseline_acc
        ),
        "W2C": W2C,
        "C2W": C2W,
        "net": W2C - C2W,
        "repair_baseline_wrong": safe_mean(
            row[
                "steered_correct"
            ]
            for row
            in baseline_wrong_rows
        ),
        "preserve_baseline_correct": safe_mean(
            row[
                "steered_correct"
            ]
            for row
            in baseline_correct_rows
        ),
        "selector_correct_N": len(
            selector_correct_rows
        ),
        "poststeer_when_selector_correct": safe_mean(
            row[
                "steered_correct"
            ]
            for row
            in selector_correct_rows
        ),
        "selector_wrong_N": len(
            selector_wrong_rows
        ),
        "poststeer_when_selector_wrong": safe_mean(
            row[
                "steered_correct"
            ]
            for row
            in selector_wrong_rows
        ),
    }

    per_relation = []

    for relation in RELATIONS:
        subset = [
            row
            for row in good
            if row[
                "relation"
            ]
            == relation
        ]

        per_relation.append({
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
            "selector_acc": safe_mean(
                row[
                    "selector_correct"
                ]
                for row
                in subset
            ),
            "steered_acc": safe_mean(
                row[
                    "steered_correct"
                ]
                for row
                in subset
            ),
            "W2C": int(
                sum(
                    row[
                        "W2C"
                    ]
                    for row
                    in subset
                )
            ),
            "C2W": int(
                sum(
                    row[
                        "C2W"
                    ]
                    for row
                    in subset
                )
            ),
        })

    confusion = []

    for gt in RELATIONS:
        for pred in RELATIONS:
            confusion.append({
                "gt": gt,
                "pred": pred,
                "count": int(
                    sum(
                        row[
                            "relation"
                        ]
                        == gt
                        and row[
                            "selector_pred"
                        ]
                        == pred
                        for row
                        in good
                    )
                ),
            })

    write_csv(
        outdir
        / "test_details.csv",
        rows,
    )

    write_csv(
        outdir
        / "summary.csv",
        [
            summary
        ],
    )

    write_csv(
        outdir
        / "per_relation.csv",
        per_relation,
    )

    write_csv(
        outdir
        / "selector_confusion.csv",
        confusion,
    )

    print("\n" + "=" * 160)
    print("ACTUAL GREEDY GENERATION: COCO MIDDLE SELECTOR -> SYNTHETIC LATE WRITER")
    print("=" * 160)

    print(
        f"model={args.model} | "
        f"TRAIN={len(train_records)} | "
        f"TEST={len(good)}"
    )

    print(
        f"reader guide layers           : "
        f"{selected_guide_layers}"
    )

    print(
        f"synthetic writer layers       : "
        f"{actuator_layers}"
    )

    print(
        f"COCO baseline                 : "
        f"{baseline_acc:.4f}"
    )

    print(
        f"COCO middle selector          : "
        f"{selector_acc:.4f}"
    )

    print(
        f"middle -> synthetic steering  : "
        f"{steered_acc:.4f} "
        f"({steered_acc - baseline_acc:+.4f}) | "
        f"W2C={W2C} "
        f"C2W={C2W} "
        f"net={W2C-C2W:+d}"
    )

    print(
        f"repair(base-wrong)            : "
        f"{summary['repair_baseline_wrong']:.4f}"
    )

    print(
        f"preserve(base-correct)        : "
        f"{summary['preserve_baseline_correct']:.4f}"
    )

    print(
        f"selector CORRECT n={len(selector_correct_rows)}: "
        f"post-steer="
        f"{summary['poststeer_when_selector_correct']:.4f}"
    )

    print(
        f"selector WRONG   n={len(selector_wrong_rows)}: "
        f"post-steer="
        f"{summary['poststeer_when_selector_wrong']:.4f}"
    )

    print("\nPer relation:")

    for row in per_relation:
        print(
            f"{row['relation']:>5s} | "
            f"N={row['N']:3d} | "
            f"base={row['baseline_acc']:.4f} | "
            f"selector={row['selector_acc']:.4f} | "
            f"steer={row['steered_acc']:.4f} | "
            f"W2C/C2W={row['W2C']}/{row['C2W']}"
        )

    print("=" * 160)

    metadata = {
        "script": (
            "eval_synthetic_writer_coco_middle_selector_v1.py"
        ),
        "model": args.model,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": n_layers,

        "synthetic_source": str(
            args.synthetic_dir
        ),
        "synthetic_source_N": len(
            source_records
        ),
        "synthetic_writer_definition": (
            "Real-Gray last-token relation means, "
            "balanced common component removed"
        ),
        "synthetic_writer_layers": (
            actuator_layers
        ),

        "coco_N_total": len(
            coco_records
        ),
        "coco_train_N": len(
            train_records
        ),
        "coco_test_N": len(
            good
        ),
        "coco_train_frac": args.train_frac,

        "middle_feature": (
            "(h_sub-h_ref)_REAL - "
            "(h_sub-h_ref)_NOIMAGE"
        ),
        "middle_prototype_source": (
            "COCO TRAIN only"
        ),
        "middle_guide_mode": (
            guide_mode
        ),
        "middle_feature_layers": (
            feature_guide_layers
        ),
        "middle_selected_layers": (
            selected_guide_layers
        ),
        "middle_layer_score_aggregation": (
            "mean cosine score across selected layers"
        ),

        "coco_test_gt_used_for_routing": False,
        "coco_test_gt_used_for_metrics": True,

        "scale": args.scale,
        "gray_value": args.gray_value,
        "seed": args.seed,
        "transformers": (
            transformers.__version__
        ),
        "coco_audit": (
            coco_audit
        ),
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
        f"{outdir / 'synthetic_writer_coco_middle_selector.npz'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'test_details.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'summary.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'selector_confusion.csv'}"
    )


if __name__ == "__main__":
    main()
