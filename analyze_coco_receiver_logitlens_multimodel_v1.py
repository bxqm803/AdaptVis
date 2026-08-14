#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Natural receiver discovery + true logit-lens path localization for COCO two-object
spatial reasoning, supporting:
  * Qwen2.5-VL-7B  (--model qwen-7b)
  * LLaVA-1.5-7B  (--model llava-7b)

Main questions
==============
1) Which later heads are natural object-text -> prompt-last RECEIVER candidates?

For each layer L and query head h:

    c_obj[L,h]
      = W_O^{L,h} sum_{s in subject/reference TEXT tokens}
                    A_{L,h}[last,s] V_{L,h}[s]

We evaluate receiver candidates with TWO independent diagnostics:

A. HELD-OUT relation decodability of c_obj[L,h]
   - train only a 4-class cosine centroid/codebook on the TRAIN split;
   - report held-out message relation ACC on EVAL.

B. Natural local logit-lens support
   - let r_attn[L] = x[L] + a[L], the natural prompt-last state immediately
     after self-attention residual addition;
   - algebraically remove ONLY this head's object-source write:

         r_minus_h = r_attn[L] - c_obj[L,h]

   - apply TRUE logit lens to both states:
         final_norm -> LM head -> left/right/above/below logits
   - report:
         margin_gain_h = margin(r_attn) - margin(r_minus_h)

   Positive mean margin_gain means this natural object-source message locally
   supports the GT answer under the model's own final readout.

This is candidate discovery / local necessity at the immediate readout.
It is NOT yet full downstream causal proof. Top heads should later be tested
with real generation ablation / QKV / path patching.

2) Is there an object-token -> last-token path problem like Qwen-3B?

At every scanned layer:

    object_relation[L]
        = mean(subject block output) - mean(reference block output)
          -> TRAIN-only held-out cosine codebook

    recv_all[L]
        = sum_h c_obj[L,h]
          -> TRAIN-only held-out cosine codebook

    x[L]
        = prompt-last block INPUT residual
          -> TRUE logit lens

    r_attn[L]
        = x[L] + attention_output[L]
          -> TRUE logit lens

    y[L]
        = block output after MLP
          -> TRUE logit lens

    native_first_step
        = actual final model logits at prompt-last

    generation
        = actual greedy model.generate()

Interpretation pattern
======================
Qwen-3B-like receiver/integration failure would look like:

    object_relation ACC high
        ->
    recv_all ACC high
        ->
    in generation-wrong & recv_all-correct samples:
        x logit-lens often wrong
        r_attn logit-lens remains much worse than recv_all
        y mostly retains r_attn if r_attn is correct

That would support:
    correct object relation is available / transported,
    but prompt-last residual integration suppresses or conflicts with it.

If instead:
    object_relation high, recv_all low
then the main loss is object -> receiver transport.

If:
    object_relation and recv_all both low
then this model may not use the same object-text -> prompt-last representation.

TRUE logit lens
===============
For an intermediate prompt-last residual z:
    scores(z) = LM_head(final_norm(z))
restricted to token variants of:
    left, right, above, below

No trained probe is used for x / r_attn / y.

Sampling
========
Default:
    randomly select 200 COCO-two samples using seed=17
The same seed/sample routine gives the same selected SIDs for both models.

Inside those 200:
    stratified TRAIN/EVAL split (default train_ratio=0.20)
The TRAIN split is used ONLY for the object / receiver cosine codebooks.
All logit-lens metrics require no training.

Default scan windows
====================
qwen-7b:
    receiver/path layers L14-L27
    expected language decoder: 28 layers, 28 query heads

llava-7b:
    receiver/path layers L16-L31
    expected language decoder: 32 layers, 32 query heads

The scan is broader than the previous scaling scan because this script is
clean-forward diagnostic, not hundreds of full-generation interventions.

LLaVA merged-token safety
=========================
LLaVA expands the image placeholder before the language decoder.
This script first captures the ACTUAL decoder prefill length, then maps
subject/reference raw text-token positions into merged decoder coordinates.
It refuses impossible mappings rather than silently tracing wrong positions.

Dependencies in AdaptVis root / PYTHONPATH
==========================================
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py

Example
=======

CUDA_VISIBLE_DEVICES=0 python -u \
analyze_coco_receiver_logitlens_multimodel_v1.py \
  --model llava-7b \
  --num-samples 200 \
  --seed 17 \
  --train-ratio 0.20 \
  --device cuda:0 \
  --output-dir output/llava7b_receiver_logitlens_n200_v1 \
  --overwrite &

CUDA_VISIBLE_DEVICES=1 python -u \
analyze_coco_receiver_logitlens_multimodel_v1.py \
  --model qwen-7b \
  --num-samples 200 \
  --seed 17 \
  --train-ratio 0.20 \
  --device cuda:0 \
  --output-dir output/qwen7b_receiver_logitlens_n200_v1 \
  --overwrite &

wait

Outputs
=======
selected_sids.json
split.json
train_basic.csv
eval_basic.csv
sample_path.csv
receiver_head_summary.csv
receiver_head_top_message.csv
receiver_head_top_local_gain.csv
layer_path_summary.csv
config.json
report.txt
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import re
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-receiver-logitlens-multimodel-v1"

RELATIONS = ("left", "right", "above", "below")
RID = {relation: index for index, relation in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "qwen-7b": {
        "layers": list(range(14, 28)),
        "expected_decoder_layers": 28,
        "expected_query_heads": 28,
        "description": "Qwen2.5-VL-7B",
    },
    "llava-7b": {
        "layers": list(range(16, 32)),
        "expected_decoder_layers": 32,
        "expected_query_heads": 32,
        "description": "LLaVA-1.5-7B",
    },
}

EPS = 1e-12


# =============================================================================
# CLI / generic IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_DEFAULTS),
    )
    parser.add_argument(
        "--dataset",
        default="coco_two",
        choices=("coco_two",),
    )
    parser.add_argument("--data-root", default="data")

    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Randomly selected COCO records; 0 = all.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.20,
        help="TRAIN fraction inside selected samples for receiver/object codebooks.",
    )

    parser.add_argument(
        "--layers",
        default="",
        help=(
            "Optional override. Empty defaults to qwen-7b:L14-L27, "
            "llava-7b:L16-L31."
        ),
    )
    parser.add_argument(
        "--trace-chunk-size",
        type=int,
        default=0,
        help=(
            "0 traces all needed layers in one helper call. "
            "If OOM, try 4 or 2."
        ),
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager",),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
    )

    parser.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    parser.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    parser.add_argument(
        "--top-k-print",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=5,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def parse_layers(text: str) -> List[int]:
    values: List[int] = []
    seen: Set[int] = set()

    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            a_text, b_text = chunk.split("-", 1)
            a = int(a_text)
            b = int(b_text)
            step = 1 if b >= a else -1
            part = range(a, b + step, step)
        else:
            part = [int(chunk)]

        for value in part:
            if value not in seen:
                values.append(value)
                seen.add(value)

    return values


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip().lower()

    aliases = {
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "over": "above",
        "on": "above",
    }

    if text in RELATIONS:
        return text
    if text in aliases:
        return aliases[text]

    hits: List[Tuple[int, str]] = []

    for relation in RELATIONS:
        match = re.search(
            rf"\b{re.escape(relation)}\b",
            text,
        )
        if match:
            hits.append((match.start(), relation))

    for pattern, relation in (
        (r"\bunder(?:neath)?\b|\bbeneath\b", "below"),
        (r"\bover\b|\bon top\b", "above"),
    ):
        match = re.search(pattern, text)
        if match:
            hits.append((match.start(), relation))

    if not hits:
        return None

    hits.sort()
    return hits[0][1]


def safe_mean(values: Iterable[Any]) -> float:
    xs: List[float] = []

    for value in values:
        try:
            x = float(value)
        except Exception:
            continue

        if math.isfinite(x):
            xs.append(x)

    return float(np.mean(xs)) if xs else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    xs: List[float] = []

    for value in values:
        try:
            x = float(value)
        except Exception:
            continue

        if math.isfinite(x):
            xs.append(x)

    return float(np.median(xs)) if xs else float("nan")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

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

        for row in rows:
            writer.writerow(dict(row))


def append_jsonl(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def clear_sampling_defaults(model: Any) -> None:
    config = getattr(
        model,
        "generation_config",
        None,
    )

    if config is None:
        return

    for name in (
        "temperature",
        "top_p",
        "top_k",
    ):
        if hasattr(config, name):
            setattr(config, name, None)


# =============================================================================
# Sampling / split
# =============================================================================

def random_sample_records(
    records: Sequence[Any],
    *,
    n: int,
    seed: int,
) -> List[Any]:
    ordered = sorted(
        records,
        key=lambda record: int(record.sid),
    )

    if n <= 0 or n >= len(ordered):
        return ordered

    selected = random.Random(seed).sample(
        ordered,
        int(n),
    )
    selected.sort(
        key=lambda record: int(record.sid)
    )
    return selected


def stratified_split_records(
    records: Sequence[Any],
    *,
    train_ratio: float,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            "--train-ratio must be strictly between 0 and 1."
        )

    grouped: Dict[str, List[Any]] = defaultdict(list)

    for record in records:
        relation = normalize_relation(
            record.relation
        )
        if relation not in RELATIONS:
            continue
        grouped[relation].append(record)

    train: List[Any] = []
    eval_rows: List[Any] = []

    rng = random.Random(seed)

    for relation in RELATIONS:
        group = list(grouped[relation])
        rng.shuffle(group)

        if len(group) < 2:
            raise RuntimeError(
                f"Need at least 2 selected samples for {relation}; got {len(group)}."
            )

        n_train = int(
            round(
                len(group)
                * float(train_ratio)
            )
        )
        n_train = max(
            1,
            min(
                len(group) - 1,
                n_train,
            ),
        )

        train.extend(
            group[:n_train]
        )
        eval_rows.extend(
            group[n_train:]
        )

    train.sort(
        key=lambda record: int(record.sid)
    )
    eval_rows.sort(
        key=lambda record: int(record.sid)
    )

    return train, eval_rows


# =============================================================================
# Prompt / relation token variants
# =============================================================================

def relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    output: Dict[str, List[int]] = {}

    for relation in RELATIONS:
        ids = set()

        for candidate in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            token_ids = tokenizer.encode(
                candidate,
                add_special_tokens=False,
            )

            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))

        if not ids:
            token_ids = tokenizer.encode(
                " " + relation,
                add_special_tokens=False,
            )

            if not token_ids:
                raise RuntimeError(
                    f"No token ID for relation {relation!r}."
                )

            ids.add(int(token_ids[-1]))

        output[relation] = sorted(ids)

    return output


def build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Any:
    rendered = probe.build_chat_prompt(
        processor,
        question,
        True,
    )

    return probe.process_inputs(
        processor,
        rendered,
        image,
        device,
    )


@torch.inference_mode()
def generate_relation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    raw_prompt_length = int(
        batch["input_ids"].shape[1]
    )

    generated = model.generate(
        **batch,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
    )

    text = processor.tokenizer.decode(
        generated[
            0,
            raw_prompt_length:,
        ],
        skip_special_tokens=True,
    ).strip()

    del generated

    return normalize_relation(text), text


# =============================================================================
# Model final norm + true logit lens
# =============================================================================

def get_attr_path(
    root: Any,
    path: str,
) -> Any:
    current = root

    for part in path.split("."):
        current = getattr(
            current,
            part,
            None,
        )

        if current is None:
            return None

    return current


def resolve_final_norm(
    model: Any,
    decoder_path: str,
) -> Tuple[Optional[torch.nn.Module], str]:
    parent = (
        decoder_path.rsplit(".", 1)[0]
        if "." in decoder_path
        else ""
    )

    candidates: List[str] = []

    if parent:
        candidates += [
            f"{parent}.norm",
            f"{parent}.final_layernorm",
            f"{parent}.ln_f",
        ]

    candidates += [
        "model.language_model.norm",
        "model.model.language_model.norm",
        "language_model.model.norm",
        "language_model.norm",
        "model.model.norm",
        "model.norm",
        "model.text_model.norm",
        "text_model.norm",
    ]

    for path in dict.fromkeys(
        candidates
    ):
        module = get_attr_path(
            model,
            path,
        )

        if isinstance(
            module,
            torch.nn.Module,
        ):
            return module, path

    return None, "unresolved"


class RelationLogitLens:
    """
    TRUE logit lens:
        state -> model final norm -> model output embedding / LM head
    but computes only relation-token rows instead of the full vocabulary.
    """

    def __init__(
        self,
        *,
        model: Any,
        final_norm: torch.nn.Module,
        token_map: Mapping[str, Sequence[int]],
    ) -> None:
        self.model = model
        self.final_norm = final_norm
        self.token_map = {
            relation: [
                int(token_id)
                for token_id in token_map[
                    relation
                ]
            ]
            for relation in RELATIONS
        }

        output_embeddings = (
            model.get_output_embeddings()
        )

        if (
            output_embeddings is None
            or not hasattr(
                output_embeddings,
                "weight",
            )
        ):
            raise RuntimeError(
                "Model has no usable output embedding / LM head weight."
            )

        weight = output_embeddings.weight

        union = sorted(
            {
                token_id
                for ids in self.token_map.values()
                for token_id in ids
            }
        )

        self.union_ids = union
        self.union_lookup = {
            token_id: index
            for index, token_id in enumerate(
                union
            )
        }

        index = torch.as_tensor(
            union,
            device=weight.device,
            dtype=torch.long,
        )

        self.weight_rows = (
            weight.index_select(
                0,
                index,
            )
            .detach()
        )

        bias = getattr(
            output_embeddings,
            "bias",
            None,
        )

        if torch.is_tensor(bias):
            self.bias_rows = (
                bias.index_select(
                    0,
                    index,
                )
                .detach()
            )
        else:
            self.bias_rows = None

        self.device = weight.device
        self.dtype = weight.dtype

    @torch.inference_mode()
    def scores(
        self,
        states: np.ndarray,
    ) -> np.ndarray:
        """
        states: [..., D]
        returns: [..., 4]
        """
        array = np.asarray(
            states,
            dtype=np.float32,
        )

        leading = array.shape[:-1]
        hidden = int(
            array.shape[-1]
        )

        flat = array.reshape(
            -1,
            hidden,
        )

        tensor = torch.as_tensor(
            flat,
            device=self.device,
            dtype=self.dtype,
        )

        normalized = self.final_norm(
            tensor
        )

        if isinstance(
            normalized,
            (tuple, list),
        ):
            normalized = normalized[0]

        token_logits = (
            normalized
            @ self.weight_rows.T
        )

        if self.bias_rows is not None:
            token_logits = (
                token_logits
                + self.bias_rows
            )

        relation_scores: List[torch.Tensor] = []

        for relation in RELATIONS:
            columns = [
                self.union_lookup[
                    token_id
                ]
                for token_id in self.token_map[
                    relation
                ]
            ]

            values = token_logits[
                :,
                columns,
            ]

            relation_scores.append(
                values.max(
                    dim=-1
                ).values
            )

        output = torch.stack(
            relation_scores,
            dim=-1,
        )

        result = (
            output.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return result.reshape(
            *leading,
            len(RELATIONS),
        )


def score_pred(
    scores: np.ndarray,
) -> str:
    return RELATIONS[
        int(
            np.argmax(
                np.asarray(scores)
            )
        )
    ]


def score_margin(
    scores: np.ndarray,
    gt: str,
) -> float:
    values = np.asarray(
        scores,
        dtype=np.float32,
    )

    return float(
        values[RID[gt]]
        - values[
            RID[
                OPPOSITE[
                    gt
                ]
            ]
        ]
    )


# =============================================================================
# Decoder merged-position mapping
# =============================================================================

def locate_hidden_3d(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    candidate = kwargs.get(
        "hidden_states"
    )

    if (
        torch.is_tensor(candidate)
        and candidate.ndim == 3
    ):
        return candidate

    for value in args:
        if (
            torch.is_tensor(value)
            and value.ndim == 3
        ):
            return value

    raise RuntimeError(
        "Cannot locate attention hidden_states."
    )


class CaptureDecoderLength:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layer: int,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layer = int(layer)
        self.lengths: List[int] = []
        self.handle = None

    def __enter__(
        self,
    ) -> "CaptureDecoderLength":
        attention = (
            self.attention_helper.resolve_self_attention(
                self.decoder_layers[
                    self.layer
                ]
            )
        )

        def pre_hook(
            _module: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> None:
            hidden = locate_hidden_3d(
                args,
                kwargs,
            )
            self.lengths.append(
                int(
                    hidden.shape[1]
                )
            )

        self.handle = (
            attention.register_forward_pre_hook(
                pre_hook,
                with_kwargs=True,
            )
        )

        return self

    def merged_length(self) -> int:
        if not self.lengths:
            raise RuntimeError(
                "Decoder-length hook did not fire."
            )

        return max(
            self.lengths
        )

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        if self.handle is not None:
            with contextlib.suppress(
                Exception
            ):
                self.handle.remove()

        self.handle = None


@torch.inference_mode()
def infer_merged_prompt_length(
    *,
    model: Any,
    batch: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    layer: int,
) -> int:
    with CaptureDecoderLength(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layer=layer,
    ) as capture:
        outputs = model(
            **batch,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        length = capture.merged_length()

    del outputs

    return int(length)


def candidate_token_id(
    tokenizer: Any,
    token: str,
) -> Optional[int]:
    try:
        value = tokenizer.convert_tokens_to_ids(
            token
        )
    except Exception:
        return None

    if value is None:
        return None

    try:
        value = int(value)
    except Exception:
        return None

    unk = getattr(
        tokenizer,
        "unk_token_id",
        None,
    )

    if (
        unk is not None
        and value == int(unk)
    ):
        return None

    return value


def image_placeholder_positions(
    *,
    model: Any,
    processor: Any,
    input_ids: Sequence[int],
) -> List[int]:
    token_ids: Set[int] = set()

    objects = [
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
        processor,
        getattr(
            processor,
            "tokenizer",
            None,
        ),
    ]

    for obj in objects:
        if obj is None:
            continue

        for name in (
            "image_token_id",
            "image_token_index",
        ):
            value = getattr(
                obj,
                name,
                None,
            )

            if isinstance(
                value,
                (int, np.integer),
            ):
                token_ids.add(
                    int(value)
                )

    for token in (
        "<image>",
        "<|image_pad|>",
        "<image_token>",
        "<IMG_CONTEXT>",
    ):
        value = candidate_token_id(
            processor.tokenizer,
            token,
        )

        if value is not None:
            token_ids.add(value)

    return [
        index
        for index, token_id in enumerate(
            input_ids
        )
        if int(token_id) in token_ids
    ]


def map_text_positions_to_decoder(
    *,
    model: Any,
    processor: Any,
    input_ids: Sequence[int],
    raw_positions: Sequence[int],
    merged_length: int,
) -> Tuple[List[int], Dict[str, Any]]:
    raw_length = len(
        input_ids
    )

    positions = sorted(
        set(
            map(
                int,
                raw_positions,
            )
        )
    )

    if not positions:
        raise RuntimeError(
            "No raw text positions to map."
        )

    placeholders = image_placeholder_positions(
        model=model,
        processor=processor,
        input_ids=input_ids,
    )

    if merged_length == raw_length:
        return positions, {
            "mode": "identity",
            "shift": 0,
            "raw_length": raw_length,
            "merged_length": merged_length,
            "image_placeholders": placeholders,
        }

    if merged_length < raw_length:
        raise RuntimeError(
            f"merged_length={merged_length} < raw_length={raw_length}; "
            "no safe mapper."
        )

    shift = int(
        merged_length
        - raw_length
    )

    if placeholders:
        last_image = max(
            placeholders
        )

        if any(
            position <= last_image
            for position in positions
        ):
            raise RuntimeError(
                "Target text position does not lie after image placeholder."
            )

        mode = (
            "image_prefix_shift"
        )
    else:
        mode = (
            "image_prefix_shift_no_marker"
        )

    mapped = [
        int(
            position
            + shift
        )
        for position in positions
    ]

    if any(
        position < 0
        or position >= merged_length
        for position in mapped
    ):
        raise RuntimeError(
            f"Mapped positions {mapped} outside merged length {merged_length}."
        )

    return mapped, {
        "mode": mode,
        "shift": shift,
        "raw_length": raw_length,
        "merged_length": merged_length,
        "image_placeholders": placeholders,
    }


# =============================================================================
# Trace extraction
# =============================================================================

def trace_local_index(
    trace: Any,
    global_position: int,
) -> int:
    lookup = {
        int(position): index
        for index, position in enumerate(
            trace.target_positions
        )
    }

    if int(global_position) not in lookup:
        raise RuntimeError(
            f"Position {global_position} missing from trace target positions "
            f"{trace.target_positions}."
        )

    return int(
        lookup[
            int(global_position)
        ]
    )


def block_state_at(
    trace: Any,
    position: int,
) -> np.ndarray:
    index = trace_local_index(
        trace,
        position,
    )

    return (
        trace.block_output[
            index
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def block_state_mean(
    trace: Any,
    positions: Sequence[int],
) -> np.ndarray:
    states = [
        block_state_at(
            trace,
            int(position),
        )
        for position in positions
    ]

    return np.mean(
        np.stack(
            states,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )


def attention_state_at(
    trace: Any,
    position: int,
) -> np.ndarray:
    index = trace_local_index(
        trace,
        position,
    )

    return (
        trace.attention_output[
            index
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def per_head_object_writes(
    *,
    trace: Any,
    prompt_last: int,
    object_positions: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        writes [Hq, Dmodel]
        object_attention_mass [Hq]
    """
    object_positions = sorted(
        set(
            map(
                int,
                object_positions,
            )
        )
    )

    if not object_positions:
        raise RuntimeError(
            "No object source positions."
        )

    local_target = trace_local_index(
        trace,
        prompt_last,
    )

    source = torch.as_tensor(
        object_positions,
        dtype=torch.long,
    )

    if int(
        source.max()
    ) >= int(
        trace.value_states.shape[1]
    ):
        raise RuntimeError(
            "Object position exceeds value sequence length."
        )

    weights = (
        trace.attention_weights[
            :,
            local_target,
            :,
        ]
        .index_select(
            1,
            source,
        )
        .float()
    )  # [Hq,Sobj]

    values = (
        trace.value_states
        .index_select(
            1,
            source,
        )
        .float()
    )  # [Hq,Sobj,Dh]

    pre = torch.einsum(
        "hs,hsd->hd",
        weights,
        values,
    )

    post = torch.einsum(
        "hd,ohd->ho",
        pre,
        trace.o_proj_weight.float(),
    )

    mass = weights.sum(
        dim=-1
    )

    return (
        post.detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32),
        mass.detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32),
    )


def trace_sample(
    *,
    attention_helper: Any,
    model: Any,
    batch: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    trace_layers: Sequence[int],
    target_positions: Sequence[int],
    chunk_size: int,
) -> Tuple[Dict[str, Any], Dict[int, Any]]:
    layers = list(
        map(
            int,
            trace_layers,
        )
    )

    if chunk_size <= 0:
        chunks = [layers]
    else:
        chunks = [
            layers[
                start:
                start
                + int(chunk_size)
            ]
            for start in range(
                0,
                len(layers),
                int(chunk_size),
            )
        ]

    all_traces: Dict[int, Any] = {}
    baseline: Optional[Dict[str, Any]] = None

    for chunk in chunks:
        current_baseline, traces = (
            attention_helper.run_and_trace(
                model=model,
                batch=batch,
                token_map=relation_token_map,
                decoder_layers=decoder_layers,
                layer_indices=chunk,
                target_positions=target_positions,
            )
        )

        if baseline is None:
            baseline = current_baseline

        for layer, trace in traces.items():
            all_traces[
                int(layer)
            ] = trace

    if baseline is None:
        raise RuntimeError(
            "No baseline returned from trace."
        )

    return baseline, all_traces


# =============================================================================
# Attention shape
# =============================================================================

def infer_query_heads(
    *,
    attention: Any,
    model: Any,
) -> int:
    for obj in (
        attention,
        getattr(
            attention,
            "config",
            None,
        ),
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
    ):
        if obj is None:
            continue

        for name in (
            "num_heads",
            "num_attention_heads",
            "n_heads",
        ):
            value = getattr(
                obj,
                name,
                None,
            )

            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass

    raise RuntimeError(
        "Cannot infer query-head count."
    )


def infer_attention_shape(
    *,
    attention: Any,
    model: Any,
) -> Tuple[int, int, int]:
    n_heads = infer_query_heads(
        attention=attention,
        model=model,
    )

    o_proj = getattr(
        attention,
        "o_proj",
        None,
    )

    if (
        o_proj is None
        or not hasattr(
            o_proj,
            "weight",
        )
    ):
        raise RuntimeError(
            "Attention module has no o_proj.weight."
        )

    weight = o_proj.weight

    hidden_size = int(
        weight.shape[0]
    )
    input_size = int(
        weight.shape[1]
    )

    if input_size % n_heads != 0:
        raise RuntimeError(
            f"o_proj input={input_size} not divisible by H={n_heads}."
        )

    head_dim = int(
        input_size
        // n_heads
    )

    return (
        int(n_heads),
        int(head_dim),
        int(hidden_size),
    )


# =============================================================================
# Streaming TRAIN codebook accumulator
# =============================================================================

class ReceiverCodebookAccumulator:
    def __init__(
        self,
        *,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
    ) -> None:
        self.n_layers = int(
            n_layers
        )
        self.n_heads = int(
            n_heads
        )
        self.hidden_size = int(
            hidden_size
        )

        self.counts = np.zeros(
            len(RELATIONS),
            dtype=np.int64,
        )

        self.object_class_sum = np.zeros(
            (
                len(RELATIONS),
                n_layers,
                hidden_size,
            ),
            dtype=np.float32,
        )

        self.recv_all_class_sum = np.zeros(
            (
                len(RELATIONS),
                n_layers,
                hidden_size,
            ),
            dtype=np.float32,
        )

        self.head_class_sum = np.zeros(
            (
                len(RELATIONS),
                n_layers,
                n_heads,
                hidden_size,
            ),
            dtype=np.float32,
        )

    def add(
        self,
        *,
        gt: str,
        object_diff: np.ndarray,
        recv_all: np.ndarray,
        head_writes: np.ndarray,
    ) -> None:
        class_id = RID[
            gt
        ]

        self.counts[
            class_id
        ] += 1

        self.object_class_sum[
            class_id
        ] += np.asarray(
            object_diff,
            dtype=np.float32,
        )

        self.recv_all_class_sum[
            class_id
        ] += np.asarray(
            recv_all,
            dtype=np.float32,
        )

        self.head_class_sum[
            class_id
        ] += np.asarray(
            head_writes,
            dtype=np.float32,
        )

    @staticmethod
    def _finish(
        class_sum: np.ndarray,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        reshape = (
            len(RELATIONS),
        ) + (
            1,
        ) * (
            class_sum.ndim
            - 1
        )

        denom = counts.reshape(
            reshape
        ).astype(
            np.float32
        )

        class_mean = (
            class_sum
            / np.maximum(
                denom,
                1.0,
            )
        )

        total_count = max(
            int(
                counts.sum()
            ),
            1,
        )

        center = (
            class_sum.sum(
                axis=0
            )
            / float(
                total_count
            )
        ).astype(
            np.float32
        )

        directions = (
            class_mean
            - center[
                None,
                ...
            ]
        )

        norm = np.linalg.norm(
            directions,
            axis=-1,
            keepdims=True,
        )

        directions = (
            directions
            / np.maximum(
                norm,
                EPS,
            )
        ).astype(
            np.float32
        )

        # class first -> move class axis immediately before hidden dim
        # object: [C,L,D]      -> [L,C,D]
        # head:   [C,L,H,D]    -> [L,H,C,D]
        if directions.ndim == 3:
            directions = np.transpose(
                directions,
                (
                    1,
                    0,
                    2,
                ),
            )
        elif directions.ndim == 4:
            directions = np.transpose(
                directions,
                (
                    1,
                    2,
                    0,
                    3,
                ),
            )
        else:
            raise RuntimeError(
                f"Unexpected direction ndim={directions.ndim}."
            )

        return center, directions

    def finish(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if np.any(
            self.counts <= 0
        ):
            raise RuntimeError(
                f"Missing TRAIN class in codebook counts: {self.counts.tolist()}"
            )

        return {
            "object": self._finish(
                self.object_class_sum,
                self.counts,
            ),
            "recv_all": self._finish(
                self.recv_all_class_sum,
                self.counts,
            ),
            "head": self._finish(
                self.head_class_sum,
                self.counts,
            ),
        }


def codebook_scores_object(
    vector: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    centered = (
        np.asarray(
            vector,
            dtype=np.float32,
        )
        - center
    )

    centered = (
        centered
        / np.maximum(
            np.linalg.norm(
                centered,
                axis=-1,
                keepdims=True,
            ),
            EPS,
        )
    )

    return np.einsum(
        "ld,lcd->lc",
        centered,
        directions,
    ).astype(
        np.float32
    )


def codebook_scores_heads(
    vectors: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    centered = (
        np.asarray(
            vectors,
            dtype=np.float32,
        )
        - center
    )

    centered = (
        centered
        / np.maximum(
            np.linalg.norm(
                centered,
                axis=-1,
                keepdims=True,
            ),
            EPS,
        )
    )

    return np.einsum(
        "lhd,lhcd->lhc",
        centered,
        directions,
    ).astype(
        np.float32
    )


# =============================================================================
# Per-sample natural state extraction
# =============================================================================

def extract_natural_states(
    *,
    traces: Mapping[int, Any],
    scan_layers: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
) -> Dict[str, np.ndarray]:
    object_diff_rows: List[np.ndarray] = []
    head_write_rows: List[np.ndarray] = []
    mass_rows: List[np.ndarray] = []
    x_rows: List[np.ndarray] = []
    attn_rows: List[np.ndarray] = []
    rattn_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    mlp_rows: List[np.ndarray] = []

    object_positions = sorted(
        set(
            map(
                int,
                list(
                    subject_positions
                )
                + list(
                    reference_positions
                ),
            )
        )
    )

    for layer in scan_layers:
        layer = int(
            layer
        )

        if (
            layer not in traces
            or layer - 1 not in traces
        ):
            raise RuntimeError(
                f"Need traces for L{layer-1} and L{layer}."
            )

        current = traces[
            layer
        ]
        previous = traces[
            layer - 1
        ]

        subject = block_state_mean(
            current,
            subject_positions,
        )
        reference = block_state_mean(
            current,
            reference_positions,
        )

        object_diff_rows.append(
            (
                subject
                - reference
            ).astype(
                np.float32
            )
        )

        head_writes, mass = (
            per_head_object_writes(
                trace=current,
                prompt_last=prompt_last,
                object_positions=object_positions,
            )
        )

        head_write_rows.append(
            head_writes
        )
        mass_rows.append(
            mass
        )

        x = block_state_at(
            previous,
            prompt_last,
        )

        attention_output = attention_state_at(
            current,
            prompt_last,
        )

        r_attn = (
            x
            + attention_output
        ).astype(
            np.float32
        )

        y = block_state_at(
            current,
            prompt_last,
        )

        mlp = (
            y
            - r_attn
        ).astype(
            np.float32
        )

        x_rows.append(x)
        attn_rows.append(
            attention_output
        )
        rattn_rows.append(
            r_attn
        )
        y_rows.append(y)
        mlp_rows.append(mlp)

    head_writes = np.stack(
        head_write_rows,
        axis=0,
    ).astype(
        np.float32
    )

    return {
        "object_diff": np.stack(
            object_diff_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "head_writes": head_writes,
        "recv_all": head_writes.sum(
            axis=1,
        ).astype(
            np.float32
        ),
        "attention_mass": np.stack(
            mass_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "x": np.stack(
            x_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "attention_output": np.stack(
            attn_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "r_attn": np.stack(
            rattn_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "y": np.stack(
            y_rows,
            axis=0,
        ).astype(
            np.float32
        ),
        "mlp": np.stack(
            mlp_rows,
            axis=0,
        ).astype(
            np.float32
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if (
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
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

    defaults = MODEL_DEFAULTS[
        args.model
    ]

    scan_layers = (
        parse_layers(
            args.layers
        )
        if args.layers.strip()
        else list(
            defaults[
                "layers"
            ]
        )
    )

    if not scan_layers:
        raise ValueError(
            "No scan layers."
        )

    scan_layers = sorted(
        set(
            map(
                int,
                scan_layers,
            )
        )
    )

    if min(
        scan_layers
    ) <= 0:
        raise ValueError(
            "This script derives x[L] from block output L-1; scan layers must start >=1."
        )

    output_dir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and output_dir.exists()
    ):
        shutil.rmtree(
            output_dir
        )

    if (
        output_dir.exists()
        and any(
            output_dir.iterdir()
        )
    ):
        raise RuntimeError(
            f"Output directory is non-empty: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors_path = (
        output_dir
        / "errors.jsonl"
    )

    probe = importlib.import_module(
        args.probe_module
    )
    attention_helper = importlib.import_module(
        args.attention_helper_module
    )
    base = probe.base

    records, audit = base.load_records(
        args.dataset,
        Path(
            args.data_root
        ),
        None,
    )

    selected_records = random_sample_records(
        records,
        n=args.num_samples,
        seed=args.seed,
    )

    train_records, eval_records = (
        stratified_split_records(
            selected_records,
            train_ratio=args.train_ratio,
            seed=args.seed + 1009,
        )
    )

    selected_sids = [
        int(
            record.sid
        )
        for record in selected_records
    ]
    train_sids = [
        int(
            record.sid
        )
        for record in train_records
    ]
    eval_sids = [
        int(
            record.sid
        )
        for record in eval_records
    ]

    (
        output_dir
        / "selected_sids.json"
    ).write_text(
        json.dumps(
            {
                "seed": args.seed,
                "N_selected": len(
                    selected_sids
                ),
                "sids": selected_sids,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "split.json"
    ).write_text(
        json.dumps(
            {
                "train_ratio": (
                    args.train_ratio
                ),
                "split_seed": (
                    args.seed
                    + 1009
                ),
                "N_train": len(
                    train_sids
                ),
                "N_eval": len(
                    eval_sids
                ),
                "train_sids": (
                    train_sids
                ),
                "eval_sids": (
                    eval_sids
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    spec = base.SPECS[
        args.model
    ]
    model_class = getattr(
        transformers,
        spec.model_class,
    )

    load_kwargs = {
        "dtype": base.resolve_dtype(
            spec.dtype_name
        ),
        "low_cpu_mem_usage": True,
        "trust_remote_code": (
            spec.trust_remote_code
        ),
        "device_map": {
            "": args.device
        },
        "attn_implementation": (
            args.attn_impl
        ),
    }

    model = None
    processor = None

    try:
        print(
            f"Loading {args.model}: {spec.repo_id}",
            flush=True,
        )

        model = model_class.from_pretrained(
            spec.repo_id,
            **load_kwargs,
        )
        model.eval()

        clear_sampling_defaults(
            model
        )

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )

        base.configure_processor(
            model,
            processor,
        )

        device = torch.device(
            args.device
        )

        decoder_layers, decoder_path = (
            probe.resolve_decoder_layers(
                model
            )
        )

        actual_decoder_layers = len(
            decoder_layers
        )

        expected_decoder_layers = int(
            defaults[
                "expected_decoder_layers"
            ]
        )

        if (
            actual_decoder_layers
            != expected_decoder_layers
        ):
            raise RuntimeError(
                f"{args.model}: expected {expected_decoder_layers} decoder layers, "
                f"got {actual_decoder_layers} at {decoder_path}."
            )

        for layer in scan_layers:
            if not (
                1
                <= layer
                < actual_decoder_layers
            ):
                raise ValueError(
                    f"L{layer} outside valid range L1-L{actual_decoder_layers - 1}."
                )

        first_attention = (
            attention_helper.resolve_self_attention(
                decoder_layers[
                    scan_layers[0]
                ]
            )
        )

        (
            n_heads,
            head_dim,
            hidden_size,
        ) = infer_attention_shape(
            attention=first_attention,
            model=model,
        )

        expected_heads = int(
            defaults[
                "expected_query_heads"
            ]
        )

        if n_heads != expected_heads:
            raise RuntimeError(
                f"{args.model}: expected {expected_heads} query heads, got {n_heads}."
            )

        for layer in scan_layers:
            attention = (
                attention_helper.resolve_self_attention(
                    decoder_layers[
                        layer
                    ]
                )
            )

            shape = infer_attention_shape(
                attention=attention,
                model=model,
            )

            if shape != (
                n_heads,
                head_dim,
                hidden_size,
            ):
                raise RuntimeError(
                    f"Non-uniform attention shape at L{layer}: {shape}."
                )

        final_norm, final_norm_path = (
            resolve_final_norm(
                model,
                decoder_path,
            )
        )

        if final_norm is None:
            raise RuntimeError(
                "Could not resolve language final norm."
            )

        relation_token_map = (
            relation_token_variants(
                processor.tokenizer
            )
        )

        lens = RelationLogitLens(
            model=model,
            final_norm=final_norm,
            token_map=relation_token_map,
        )

        trace_layers = sorted(
            set(
                [layer - 1 for layer in scan_layers]
                + scan_layers
            )
        )

        print(
            "\n"
            + "=" * 170
        )
        print(
            "RECEIVER DISCOVERY + TRUE LOGIT-LENS PATH LOCALIZATION"
        )
        print(
            "=" * 170
        )
        print(
            "model             :",
            args.model,
            f"({defaults['description']})",
        )
        print(
            "repo              :",
            spec.repo_id,
        )
        print(
            "decoder path      :",
            decoder_path,
        )
        print(
            "decoder layers    :",
            actual_decoder_layers,
        )
        print(
            "query heads       :",
            n_heads,
        )
        print(
            "head dim          :",
            head_dim,
        )
        print(
            "hidden size       :",
            hidden_size,
        )
        print(
            "scan layers       :",
            scan_layers,
        )
        print(
            "trace layers      :",
            trace_layers,
        )
        print(
            "final norm        :",
            final_norm_path,
        )
        print(
            "N selected/train/eval:",
            len(selected_records),
            len(train_records),
            len(eval_records),
        )
        print(
            "trace chunk       :",
            args.trace_chunk_size,
        )
        print(
            "=" * 170,
            flush=True,
        )

        accumulator = (
            ReceiverCodebookAccumulator(
                n_layers=len(
                    scan_layers
                ),
                n_heads=n_heads,
                hidden_size=hidden_size,
            )
        )

        train_basic: List[
            Dict[str, Any]
        ] = []

        # ---------------------------------------------------------------------
        # TRAIN: only build relation codebooks from natural clean states.
        # ---------------------------------------------------------------------

        for sample_index, record in enumerate(
            tqdm(
                train_records,
                desc=f"train-trace:{args.model}",
            ),
            start=1,
        ):
            image = None
            batch = None

            try:
                sid = int(
                    record.sid
                )
                gt = normalize_relation(
                    record.relation
                )

                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Unsupported GT: {record.relation!r}"
                    )

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(
                    record.image_path
                ).convert(
                    "RGB"
                )

                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(value)
                    for value in (
                        batch[
                            "input_ids"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ]

                raw_subject = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(
                            record.subject
                        ),
                    )
                )
                raw_reference = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(
                            record.reference
                        ),
                    )
                )

                merged_length = (
                    infer_merged_prompt_length(
                        model=model,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        layer=scan_layers[0],
                    )
                )

                subject_positions, map_meta = (
                    map_text_positions_to_decoder(
                        model=model,
                        processor=processor,
                        input_ids=input_ids,
                        raw_positions=raw_subject,
                        merged_length=merged_length,
                    )
                )

                reference_positions, _ = (
                    map_text_positions_to_decoder(
                        model=model,
                        processor=processor,
                        input_ids=input_ids,
                        raw_positions=raw_reference,
                        merged_length=merged_length,
                    )
                )

                prompt_last = int(
                    merged_length - 1
                )

                target_positions = sorted(
                    set(
                        subject_positions
                        + reference_positions
                        + [
                            prompt_last
                        ]
                    )
                )

                baseline, traces = trace_sample(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    trace_layers=trace_layers,
                    target_positions=target_positions,
                    chunk_size=args.trace_chunk_size,
                )

                natural = extract_natural_states(
                    traces=traces,
                    scan_layers=scan_layers,
                    subject_positions=subject_positions,
                    reference_positions=reference_positions,
                    prompt_last=prompt_last,
                )

                accumulator.add(
                    gt=gt,
                    object_diff=natural[
                        "object_diff"
                    ],
                    recv_all=natural[
                        "recv_all"
                    ],
                    head_writes=natural[
                        "head_writes"
                    ],
                )

                train_basic.append({
                    "sid": sid,
                    "gt": gt,
                    "mapping_mode": (
                        map_meta[
                            "mode"
                        ]
                    ),
                    "position_shift": (
                        map_meta[
                            "shift"
                        ]
                    ),
                    "raw_prompt_length": (
                        len(
                            input_ids
                        )
                    ),
                    "merged_prompt_length": (
                        merged_length
                    ),
                    "native_first_step_pred": (
                        baseline[
                            "prediction"
                        ]
                    ),
                })

                del natural
                del traces

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "train",
                        "sid": int(
                            getattr(
                                record,
                                "sid",
                                -1,
                            )
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(
                        Exception
                    ):
                        image.close()

                del batch

                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index
                    % args.empty_cache_every
                    == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(
            output_dir
            / "train_basic.csv",
            train_basic,
        )

        codebooks = accumulator.finish()

        object_center, object_dirs = (
            codebooks[
                "object"
            ]
        )
        recv_center, recv_dirs = (
            codebooks[
                "recv_all"
            ]
        )
        head_center, head_dirs = (
            codebooks[
                "head"
            ]
        )

        # ---------------------------------------------------------------------
        # EVAL aggregation arrays.
        # ---------------------------------------------------------------------

        n_layers = len(
            scan_layers
        )

        head_correct = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.int64,
        )
        head_margin_sum = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.float64,
        )
        head_mass_sum = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.float64,
        )
        head_norm_sum = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.float64,
        )
        head_local_gain_sum = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.float64,
        )
        head_local_gain_positive = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.int64,
        )
        head_message_correct_and_gain_positive = np.zeros(
            (
                n_layers,
                n_heads,
            ),
            dtype=np.int64,
        )

        eval_basic: List[
            Dict[str, Any]
        ] = []
        sample_path_rows: List[
            Dict[str, Any]
        ] = []

        native_firststep_correct = 0
        generation_correct = 0
        final_lens_native_match = 0
        final_lens_native_covered = 0

        # ---------------------------------------------------------------------
        # EVAL.
        # ---------------------------------------------------------------------

        for sample_index, record in enumerate(
            tqdm(
                eval_records,
                desc=f"eval-trace:{args.model}",
            ),
            start=1,
        ):
            image = None
            batch = None

            try:
                sid = int(
                    record.sid
                )
                gt = normalize_relation(
                    record.relation
                )

                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Unsupported GT: {record.relation!r}"
                    )

                gt_id = RID[
                    gt
                ]
                opposite_id = RID[
                    OPPOSITE[
                        gt
                    ]
                ]

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(
                    record.image_path
                ).convert(
                    "RGB"
                )

                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(value)
                    for value in (
                        batch[
                            "input_ids"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ]

                raw_subject = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(
                            record.subject
                        ),
                    )
                )
                raw_reference = (
                    probe.locate_phrase_positions(
                        processor.tokenizer,
                        input_ids,
                        str(
                            record.reference
                        ),
                    )
                )

                merged_length = (
                    infer_merged_prompt_length(
                        model=model,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        layer=scan_layers[0],
                    )
                )

                subject_positions, map_meta = (
                    map_text_positions_to_decoder(
                        model=model,
                        processor=processor,
                        input_ids=input_ids,
                        raw_positions=raw_subject,
                        merged_length=merged_length,
                    )
                )

                reference_positions, _ = (
                    map_text_positions_to_decoder(
                        model=model,
                        processor=processor,
                        input_ids=input_ids,
                        raw_positions=raw_reference,
                        merged_length=merged_length,
                    )
                )

                prompt_last = int(
                    merged_length - 1
                )

                target_positions = sorted(
                    set(
                        subject_positions
                        + reference_positions
                        + [
                            prompt_last
                        ]
                    )
                )

                baseline, traces = trace_sample(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    trace_layers=trace_layers,
                    target_positions=target_positions,
                    chunk_size=args.trace_chunk_size,
                )

                natural = extract_natural_states(
                    traces=traces,
                    scan_layers=scan_layers,
                    subject_positions=subject_positions,
                    reference_positions=reference_positions,
                    prompt_last=prompt_last,
                )

                # -------------------------------------------------------------
                # TRAIN-only codebook readouts: object, receiver-all, per-head.
                # -------------------------------------------------------------

                object_scores = (
                    codebook_scores_object(
                        natural[
                            "object_diff"
                        ],
                        object_center,
                        object_dirs,
                    )
                )

                recv_scores = (
                    codebook_scores_object(
                        natural[
                            "recv_all"
                        ],
                        recv_center,
                        recv_dirs,
                    )
                )

                head_scores = (
                    codebook_scores_heads(
                        natural[
                            "head_writes"
                        ],
                        head_center,
                        head_dirs,
                    )
                )

                object_pred_id = np.argmax(
                    object_scores,
                    axis=-1,
                )
                recv_pred_id = np.argmax(
                    recv_scores,
                    axis=-1,
                )
                head_pred_id = np.argmax(
                    head_scores,
                    axis=-1,
                )

                # -------------------------------------------------------------
                # TRUE logit lens on natural prompt-last states.
                # -------------------------------------------------------------

                stacked_states = np.concatenate(
                    [
                        natural["x"],
                        natural["r_attn"],
                        natural["y"],
                    ],
                    axis=0,
                )

                stacked_lens = lens.scores(
                    stacked_states
                )

                x_lens = stacked_lens[
                    0:n_layers
                ]
                rattn_lens = stacked_lens[
                    n_layers:
                    2 * n_layers
                ]
                y_lens = stacked_lens[
                    2 * n_layers:
                    3 * n_layers
                ]

                x_pred_id = np.argmax(
                    x_lens,
                    axis=-1,
                )
                rattn_pred_id = np.argmax(
                    rattn_lens,
                    axis=-1,
                )
                y_pred_id = np.argmax(
                    y_lens,
                    axis=-1,
                )

                x_margin = (
                    x_lens[
                        :,
                        gt_id,
                    ]
                    - x_lens[
                        :,
                        opposite_id,
                    ]
                )
                rattn_margin = (
                    rattn_lens[
                        :,
                        gt_id,
                    ]
                    - rattn_lens[
                        :,
                        opposite_id,
                    ]
                )
                y_margin = (
                    y_lens[
                        :,
                        gt_id,
                    ]
                    - y_lens[
                        :,
                        opposite_id,
                    ]
                )

                # -------------------------------------------------------------
                # Per-head local natural message removal under TRUE logit lens.
                # r_minus_h = r_attn - c_obj_h
                # -------------------------------------------------------------

                r_minus_h = (
                    natural[
                        "r_attn"
                    ][
                        :,
                        None,
                        :,
                    ]
                    - natural[
                        "head_writes"
                    ]
                )

                minus_scores = lens.scores(
                    r_minus_h
                )

                minus_margin = (
                    minus_scores[
                        :,
                        :,
                        gt_id,
                    ]
                    - minus_scores[
                        :,
                        :,
                        opposite_id,
                    ]
                )

                local_gain = (
                    rattn_margin[
                        :,
                        None,
                    ]
                    - minus_margin
                )

                message_correct = (
                    head_pred_id
                    == gt_id
                )

                head_correct += (
                    message_correct
                    .astype(
                        np.int64
                    )
                )

                head_margin_sum += (
                    head_scores[
                        :,
                        :,
                        gt_id,
                    ]
                    - head_scores[
                        :,
                        :,
                        opposite_id,
                    ]
                )

                head_mass_sum += (
                    natural[
                        "attention_mass"
                    ]
                )

                head_norm_sum += (
                    np.linalg.norm(
                        natural[
                            "head_writes"
                        ],
                        axis=-1,
                    )
                )

                head_local_gain_sum += (
                    local_gain
                )

                head_local_gain_positive += (
                    local_gain > 0
                ).astype(
                    np.int64
                )

                head_message_correct_and_gain_positive += (
                    message_correct
                    & (
                        local_gain
                        > 0
                    )
                ).astype(
                    np.int64
                )

                # -------------------------------------------------------------
                # Actual final first-step + generation.
                # -------------------------------------------------------------

                native_first_pred = str(
                    baseline[
                        "prediction"
                    ]
                )

                native_firststep_correct += int(
                    native_first_pred
                    == gt
                )

                generation_pred, generation_text = (
                    generate_relation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                )

                gen_correct = (
                    generation_pred
                    == gt
                )

                generation_correct += int(
                    gen_correct
                )

                if (
                    scan_layers[-1]
                    == actual_decoder_layers - 1
                ):
                    final_lens_pred = RELATIONS[
                        int(
                            y_pred_id[
                                -1
                            ]
                        )
                    ]

                    final_lens_native_covered += 1
                    final_lens_native_match += int(
                        final_lens_pred
                        == native_first_pred
                    )
                else:
                    final_lens_pred = None

                eval_basic.append({
                    "sid": sid,
                    "gt": gt,
                    "native_first_step_pred": (
                        native_first_pred
                    ),
                    "native_first_step_correct": (
                        native_first_pred
                        == gt
                    ),
                    "generation_pred": (
                        generation_pred
                    ),
                    "generation_text": (
                        generation_text
                    ),
                    "generation_correct": (
                        gen_correct
                    ),
                    "final_scanned_layer_lens_pred": (
                        final_lens_pred
                    ),
                    "mapping_mode": (
                        map_meta[
                            "mode"
                        ]
                    ),
                    "position_shift": (
                        map_meta[
                            "shift"
                        ]
                    ),
                    "raw_prompt_length": (
                        len(
                            input_ids
                        )
                    ),
                    "merged_prompt_length": (
                        merged_length
                    ),
                })

                for local_layer, layer in enumerate(
                    scan_layers
                ):
                    object_pred = RELATIONS[
                        int(
                            object_pred_id[
                                local_layer
                            ]
                        )
                    ]
                    recv_pred = RELATIONS[
                        int(
                            recv_pred_id[
                                local_layer
                            ]
                        )
                    ]
                    x_pred = RELATIONS[
                        int(
                            x_pred_id[
                                local_layer
                            ]
                        )
                    ]
                    rattn_pred = RELATIONS[
                        int(
                            rattn_pred_id[
                                local_layer
                            ]
                        )
                    ]
                    y_pred = RELATIONS[
                        int(
                            y_pred_id[
                                local_layer
                            ]
                        )
                    ]

                    sample_path_rows.append({
                        "sid": sid,
                        "layer": int(
                            layer
                        ),
                        "gt": gt,
                        "generation_correct": (
                            gen_correct
                        ),
                        "object_pred": (
                            object_pred
                        ),
                        "object_correct": (
                            object_pred
                            == gt
                        ),
                        "object_margin": float(
                            object_scores[
                                local_layer,
                                gt_id,
                            ]
                            - object_scores[
                                local_layer,
                                opposite_id,
                            ]
                        ),
                        "recv_all_pred": (
                            recv_pred
                        ),
                        "recv_all_correct": (
                            recv_pred
                            == gt
                        ),
                        "recv_all_margin": float(
                            recv_scores[
                                local_layer,
                                gt_id,
                            ]
                            - recv_scores[
                                local_layer,
                                opposite_id,
                            ]
                        ),
                        "lens_x_pred": (
                            x_pred
                        ),
                        "lens_x_correct": (
                            x_pred
                            == gt
                        ),
                        "lens_x_margin": float(
                            x_margin[
                                local_layer
                            ]
                        ),
                        "lens_rattn_pred": (
                            rattn_pred
                        ),
                        "lens_rattn_correct": (
                            rattn_pred
                            == gt
                        ),
                        "lens_rattn_margin": float(
                            rattn_margin[
                                local_layer
                            ]
                        ),
                        "attention_margin_gain": float(
                            rattn_margin[
                                local_layer
                            ]
                            - x_margin[
                                local_layer
                            ]
                        ),
                        "lens_y_pred": (
                            y_pred
                        ),
                        "lens_y_correct": (
                            y_pred
                            == gt
                        ),
                        "lens_y_margin": float(
                            y_margin[
                                local_layer
                            ]
                        ),
                        "mlp_margin_gain": float(
                            y_margin[
                                local_layer
                            ]
                            - rattn_margin[
                                local_layer
                            ]
                        ),
                        "recv_all_norm": float(
                            np.linalg.norm(
                                natural[
                                    "recv_all"
                                ][
                                    local_layer
                                ]
                            )
                        ),
                        "attention_output_norm": float(
                            np.linalg.norm(
                                natural[
                                    "attention_output"
                                ][
                                    local_layer
                                ]
                            )
                        ),
                        "x_norm": float(
                            np.linalg.norm(
                                natural[
                                    "x"
                                ][
                                    local_layer
                                ]
                            )
                        ),
                    })

                del natural
                del traces

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "eval",
                        "sid": int(
                            getattr(
                                record,
                                "sid",
                                -1,
                            )
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(
                        Exception
                    ):
                        image.close()

                del batch

                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index
                    % args.empty_cache_every
                    == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(
            output_dir
            / "eval_basic.csv",
            eval_basic,
        )

        write_csv(
            output_dir
            / "sample_path.csv",
            sample_path_rows,
        )

        n_eval = len(
            eval_basic
        )

        if n_eval <= 0:
            raise RuntimeError(
                "No successful EVAL samples."
            )

        # ---------------------------------------------------------------------
        # Receiver head summary.
        # ---------------------------------------------------------------------

        receiver_rows: List[
            Dict[str, Any]
        ] = []

        for local_layer, layer in enumerate(
            scan_layers
        ):
            for head in range(
                n_heads
            ):
                receiver_rows.append({
                    "layer": int(
                        layer
                    ),
                    "head": int(
                        head
                    ),
                    "head_name": (
                        f"L{int(layer)}H{int(head):02d}"
                    ),
                    "N_eval": n_eval,
                    "message_relation_acc": float(
                        head_correct[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                    "mean_message_gt_opp_margin": float(
                        head_margin_sum[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                    "mean_object_attention_mass": float(
                        head_mass_sum[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                    "mean_message_norm": float(
                        head_norm_sum[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                    "mean_local_lens_margin_gain": float(
                        head_local_gain_sum[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                    "positive_local_lens_gain_rate": float(
                        head_local_gain_positive[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                    "message_correct_and_gain_positive_rate": float(
                        head_message_correct_and_gain_positive[
                            local_layer,
                            head,
                        ]
                        / n_eval
                    ),
                })

        by_message = sorted(
            receiver_rows,
            key=lambda row: (
                -float(
                    row[
                        "message_relation_acc"
                    ]
                ),
                -float(
                    row[
                        "mean_local_lens_margin_gain"
                    ]
                ),
                -float(
                    row[
                        "positive_local_lens_gain_rate"
                    ]
                ),
                int(
                    row[
                        "layer"
                    ]
                ),
                int(
                    row[
                        "head"
                    ]
                ),
            ),
        )

        message_rank = {
            (
                int(
                    row[
                        "layer"
                    ]
                ),
                int(
                    row[
                        "head"
                    ]
                ),
            ): rank
            for rank, row in enumerate(
                by_message,
                start=1,
            )
        }

        by_gain = sorted(
            receiver_rows,
            key=lambda row: (
                -float(
                    row[
                        "mean_local_lens_margin_gain"
                    ]
                ),
                -float(
                    row[
                        "positive_local_lens_gain_rate"
                    ]
                ),
                -float(
                    row[
                        "message_relation_acc"
                    ]
                ),
                int(
                    row[
                        "layer"
                    ]
                ),
                int(
                    row[
                        "head"
                    ]
                ),
            ),
        )

        gain_rank = {
            (
                int(
                    row[
                        "layer"
                    ]
                ),
                int(
                    row[
                        "head"
                    ]
                ),
            ): rank
            for rank, row in enumerate(
                by_gain,
                start=1,
            )
        }

        for row in receiver_rows:
            key = (
                int(
                    row[
                        "layer"
                    ]
                ),
                int(
                    row[
                        "head"
                    ]
                ),
            )

            row[
                "rank_message_relation_acc"
            ] = message_rank[
                key
            ]
            row[
                "rank_local_lens_gain"
            ] = gain_rank[
                key
            ]
            row[
                "rank_sum"
            ] = (
                message_rank[
                    key
                ]
                + gain_rank[
                    key
                ]
            )

        receiver_rows.sort(
            key=lambda row: (
                int(
                    row[
                        "rank_sum"
                    ]
                ),
                -float(
                    row[
                        "message_relation_acc"
                    ]
                ),
            )
        )

        write_csv(
            output_dir
            / "receiver_head_summary.csv",
            receiver_rows,
        )
        write_csv(
            output_dir
            / "receiver_head_top_message.csv",
            by_message[
                :
                max(
                    args.top_k_print,
                    50,
                )
            ],
        )
        write_csv(
            output_dir
            / "receiver_head_top_local_gain.csv",
            by_gain[
                :
                max(
                    args.top_k_print,
                    50,
                )
            ],
        )

        # ---------------------------------------------------------------------
        # Layer path summary.
        # ---------------------------------------------------------------------

        grouped_by_layer: Dict[
            int,
            List[Mapping[str, Any]],
        ] = defaultdict(list)

        for row in sample_path_rows:
            grouped_by_layer[
                int(
                    row[
                        "layer"
                    ]
                )
            ].append(row)

        layer_summary: List[
            Dict[str, Any]
        ] = []

        for layer in scan_layers:
            rows = grouped_by_layer[
                int(layer)
            ]

            wrong = [
                row
                for row in rows
                if not bool(
                    row[
                        "generation_correct"
                    ]
                )
            ]

            wrong_obj_correct = [
                row
                for row in wrong
                if bool(
                    row[
                        "object_correct"
                    ]
                )
            ]

            wrong_recv_correct = [
                row
                for row in wrong
                if bool(
                    row[
                        "recv_all_correct"
                    ]
                )
            ]

            wrong_rattn_correct = [
                row
                for row in wrong
                if bool(
                    row[
                        "lens_rattn_correct"
                    ]
                )
            ]

            def rate(
                subset: Sequence[
                    Mapping[str, Any]
                ],
                key: str,
            ) -> float:
                if not subset:
                    return float(
                        "nan"
                    )

                return safe_mean(
                    float(
                        bool(
                            row[
                                key
                            ]
                        )
                    )
                    for row in subset
                )

            layer_summary.append({
                "layer": int(
                    layer
                ),
                "N_eval": len(
                    rows
                ),
                "object_relation_acc": rate(
                    rows,
                    "object_correct",
                ),
                "recv_all_relation_acc": rate(
                    rows,
                    "recv_all_correct",
                ),
                "lens_x_acc": rate(
                    rows,
                    "lens_x_correct",
                ),
                "lens_rattn_acc": rate(
                    rows,
                    "lens_rattn_correct",
                ),
                "lens_y_acc": rate(
                    rows,
                    "lens_y_correct",
                ),
                "mean_object_margin": safe_mean(
                    row[
                        "object_margin"
                    ]
                    for row in rows
                ),
                "mean_recv_all_margin": safe_mean(
                    row[
                        "recv_all_margin"
                    ]
                    for row in rows
                ),
                "mean_lens_x_margin": safe_mean(
                    row[
                        "lens_x_margin"
                    ]
                    for row in rows
                ),
                "mean_lens_rattn_margin": safe_mean(
                    row[
                        "lens_rattn_margin"
                    ]
                    for row in rows
                ),
                "mean_lens_y_margin": safe_mean(
                    row[
                        "lens_y_margin"
                    ]
                    for row in rows
                ),
                "mean_attention_margin_gain": safe_mean(
                    row[
                        "attention_margin_gain"
                    ]
                    for row in rows
                ),
                "mean_mlp_margin_gain": safe_mean(
                    row[
                        "mlp_margin_gain"
                    ]
                    for row in rows
                ),
                "N_generation_wrong": len(
                    wrong
                ),
                "N_wrong_object_correct": len(
                    wrong_obj_correct
                ),
                "wrong_object_correct_to_recv_retain": rate(
                    wrong_obj_correct,
                    "recv_all_correct",
                ),
                "N_wrong_recv_correct": len(
                    wrong_recv_correct
                ),
                "wrong_recv_correct_x_acc": rate(
                    wrong_recv_correct,
                    "lens_x_correct",
                ),
                "wrong_recv_correct_rattn_acc": rate(
                    wrong_recv_correct,
                    "lens_rattn_correct",
                ),
                "wrong_recv_correct_y_acc": rate(
                    wrong_recv_correct,
                    "lens_y_correct",
                ),
                "wrong_recv_correct_mean_x_margin": safe_mean(
                    row[
                        "lens_x_margin"
                    ]
                    for row in wrong_recv_correct
                ),
                "wrong_recv_correct_mean_rattn_margin": safe_mean(
                    row[
                        "lens_rattn_margin"
                    ]
                    for row in wrong_recv_correct
                ),
                "wrong_recv_correct_mean_y_margin": safe_mean(
                    row[
                        "lens_y_margin"
                    ]
                    for row in wrong_recv_correct
                ),
                "N_wrong_rattn_correct": len(
                    wrong_rattn_correct
                ),
                "wrong_rattn_correct_to_y_retain": rate(
                    wrong_rattn_correct,
                    "lens_y_correct",
                ),
                "mean_recv_all_norm": safe_mean(
                    row[
                        "recv_all_norm"
                    ]
                    for row in rows
                ),
                "mean_attention_output_norm": safe_mean(
                    row[
                        "attention_output_norm"
                    ]
                    for row in rows
                ),
                "mean_x_norm": safe_mean(
                    row[
                        "x_norm"
                    ]
                    for row in rows
                ),
                "mean_recv_to_attention_norm_ratio": safe_mean(
                    float(
                        row[
                            "recv_all_norm"
                        ]
                    )
                    / max(
                        float(
                            row[
                                "attention_output_norm"
                            ]
                        ),
                        EPS,
                    )
                    for row in rows
                ),
                "mean_recv_to_x_norm_ratio": safe_mean(
                    float(
                        row[
                            "recv_all_norm"
                        ]
                    )
                    / max(
                        float(
                            row[
                                "x_norm"
                            ]
                        ),
                        EPS,
                    )
                    for row in rows
                ),
            })

        write_csv(
            output_dir
            / "layer_path_summary.csv",
            layer_summary,
        )

        native_acc = (
            native_firststep_correct
            / n_eval
        )
        generation_acc = (
            generation_correct
            / n_eval
        )

        final_match_rate = (
            final_lens_native_match
            / final_lens_native_covered
            if final_lens_native_covered
            else float("nan")
        )

        # ---------------------------------------------------------------------
        # Console report.
        # ---------------------------------------------------------------------

        print(
            "\n"
            + "=" * 180
        )
        print(
            f"{args.model.upper()} RECEIVER + LOGIT-LENS SUMMARY"
        )
        print(
            "=" * 180
        )
        print(
            f"EVAL N={n_eval} | native first-step ACC={100*native_acc:.2f}% "
            f"| generation ACC={100*generation_acc:.2f}%"
        )
        print(
            f"Final scanned-layer true-lens vs native first-step prediction match: "
            f"{100*final_match_rate:.2f}% "
            f"(N={final_lens_native_covered})"
        )

        print(
            "\nTOP RECEIVER CANDIDATES BY HELD-OUT MESSAGE RELATION ACC"
        )
        print(
            f"  {'rank':>4s} {'head':<9s} {'msgACC':>8s} "
            f"{'msgMargin':>10s} {'mass':>9s} {'norm':>9s} "
            f"{'localGain':>10s} {'gain+':>8s} {'both':>8s}"
        )

        for rank, row in enumerate(
            by_message[
                :
                args.top_k_print
            ],
            start=1,
        ):
            print(
                f"  {rank:4d} "
                f"{str(row['head_name']):<9s} "
                f"{100*float(row['message_relation_acc']):7.2f}% "
                f"{float(row['mean_message_gt_opp_margin']):+10.4f} "
                f"{float(row['mean_object_attention_mass']):9.4f} "
                f"{float(row['mean_message_norm']):9.4f} "
                f"{float(row['mean_local_lens_margin_gain']):+10.4f} "
                f"{100*float(row['positive_local_lens_gain_rate']):7.2f}% "
                f"{100*float(row['message_correct_and_gain_positive_rate']):7.2f}%"
            )

        print(
            "\nTOP RECEIVER CANDIDATES BY NATURAL LOCAL LOGIT-LENS MARGIN GAIN"
        )
        print(
            f"  {'rank':>4s} {'head':<9s} {'localGain':>10s} "
            f"{'gain+':>8s} {'msgACC':>8s} {'msgMargin':>10s} "
            f"{'mass':>9s} {'norm':>9s}"
        )

        for rank, row in enumerate(
            by_gain[
                :
                args.top_k_print
            ],
            start=1,
        ):
            print(
                f"  {rank:4d} "
                f"{str(row['head_name']):<9s} "
                f"{float(row['mean_local_lens_margin_gain']):+10.4f} "
                f"{100*float(row['positive_local_lens_gain_rate']):7.2f}% "
                f"{100*float(row['message_relation_acc']):7.2f}% "
                f"{float(row['mean_message_gt_opp_margin']):+10.4f} "
                f"{float(row['mean_object_attention_mass']):9.4f} "
                f"{float(row['mean_message_norm']):9.4f}"
            )

        print(
            "\nLAYER PATH SUMMARY"
        )
        print(
            f"  {'L':>3s} {'obj':>7s} {'recv':>7s} {'xLens':>7s} "
            f"{'aLens':>7s} {'yLens':>7s} "
            f"{'attnΔ':>9s} {'mlpΔ':>9s} "
            f"{'wrongRecv':>9s} {'x|R':>7s} {'a|R':>7s} {'y|R':>7s} "
            f"{'a->yRet':>9s}"
        )

        for row in layer_summary:
            def pct(
                value: Any,
            ) -> str:
                try:
                    x = float(value)
                except Exception:
                    return "  nan "
                if not math.isfinite(x):
                    return "  nan "
                return f"{100*x:6.2f}%"

            print(
                f"  {int(row['layer']):3d} "
                f"{pct(row['object_relation_acc']):>7s} "
                f"{pct(row['recv_all_relation_acc']):>7s} "
                f"{pct(row['lens_x_acc']):>7s} "
                f"{pct(row['lens_rattn_acc']):>7s} "
                f"{pct(row['lens_y_acc']):>7s} "
                f"{float(row['mean_attention_margin_gain']):+9.3f} "
                f"{float(row['mean_mlp_margin_gain']):+9.3f} "
                f"{int(row['N_wrong_recv_correct']):9d} "
                f"{pct(row['wrong_recv_correct_x_acc']):>7s} "
                f"{pct(row['wrong_recv_correct_rattn_acc']):>7s} "
                f"{pct(row['wrong_recv_correct_y_acc']):>7s} "
                f"{pct(row['wrong_rattn_correct_to_y_retain']):>9s}"
            )

        print(
            "=" * 180
        )

        # ---------------------------------------------------------------------
        # Config + report.
        # ---------------------------------------------------------------------

        config = {
            "script_version": (
                SCRIPT_VERSION
            ),
            "model": args.model,
            "model_description": (
                defaults[
                    "description"
                ]
            ),
            "repo_id": (
                spec.repo_id
            ),
            "dataset": (
                args.dataset
            ),
            "data_root": (
                args.data_root
            ),
            "sample_seed": (
                args.seed
            ),
            "N_selected": (
                len(
                    selected_records
                )
            ),
            "train_ratio": (
                args.train_ratio
            ),
            "N_train": (
                len(
                    train_records
                )
            ),
            "N_eval": (
                n_eval
            ),
            "decoder_path": (
                decoder_path
            ),
            "decoder_layers": (
                actual_decoder_layers
            ),
            "query_heads": (
                n_heads
            ),
            "head_dim": (
                head_dim
            ),
            "hidden_size": (
                hidden_size
            ),
            "scan_layers": (
                scan_layers
            ),
            "trace_layers": (
                trace_layers
            ),
            "trace_chunk_size": (
                args.trace_chunk_size
            ),
            "final_norm_path": (
                final_norm_path
            ),
            "native_first_step_acc": (
                native_acc
            ),
            "generation_acc": (
                generation_acc
            ),
            "final_lens_native_match_rate": (
                final_match_rate
            ),
            "uses_intervention_for_receiver_scan": False,
            "receiver_message": (
                "W_O^h sum_object_text A[last,s] V[s]"
            ),
            "true_logit_lens": (
                "LM_head(final_norm(intermediate_prompt_last_residual))"
            ),
            "local_receiver_removal": (
                "r_attn_minus_h = r_attn - clean natural object-source post-WO head write"
            ),
            "audit": audit,
        }

        (
            output_dir
            / "config.json"
        ).write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        report_lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"repo: {spec.repo_id}",
            f"N selected/train/eval: {len(selected_records)}/{len(train_records)}/{n_eval}",
            f"scan layers: {scan_layers}",
            f"query heads: {n_heads}",
            f"final norm: {final_norm_path}",
            f"native first-step ACC: {100*native_acc:.2f}%",
            f"generation ACC: {100*generation_acc:.2f}%",
            f"final lens vs native prediction match: {100*final_match_rate:.2f}%",
            "",
            "TOP BY MESSAGE RELATION ACC",
        ]

        for rank, row in enumerate(
            by_message[
                :
                args.top_k_print
            ],
            start=1,
        ):
            report_lines.append(
                f"{rank:02d} {row['head_name']} "
                f"msgACC={100*float(row['message_relation_acc']):.2f}% "
                f"msgMargin={float(row['mean_message_gt_opp_margin']):+.4f} "
                f"localGain={float(row['mean_local_lens_margin_gain']):+.4f} "
                f"gainPos={100*float(row['positive_local_lens_gain_rate']):.2f}% "
                f"mass={float(row['mean_object_attention_mass']):.4f}"
            )

        report_lines += [
            "",
            "TOP BY LOCAL LOGIT-LENS MARGIN GAIN",
        ]

        for rank, row in enumerate(
            by_gain[
                :
                args.top_k_print
            ],
            start=1,
        ):
            report_lines.append(
                f"{rank:02d} {row['head_name']} "
                f"localGain={float(row['mean_local_lens_margin_gain']):+.4f} "
                f"gainPos={100*float(row['positive_local_lens_gain_rate']):.2f}% "
                f"msgACC={100*float(row['message_relation_acc']):.2f}% "
                f"mass={float(row['mean_object_attention_mass']):.4f}"
            )

        report_lines += [
            "",
            "INTERPRETATION GUIDE",
            (
                "object high -> recv_all low: object-to-receiver transport loss."
            ),
            (
                "recv_all high, but generation-wrong & recv-correct cohort has low "
                "x/r_attn/y lens: receiver message exists but prompt-last integration "
                "is a candidate failure."
            ),
            (
                "r_attn lens correct -> y lens mostly retained: MLP is not the main overwrite."
            ),
            (
                "message ACC high + positive local lens gain identifies stronger receiver "
                "candidates than either metric alone, but full downstream causal tests are "
                "still required."
            ),
            (
                "Absolute intermediate logit-lens ACC can be under-calibrated; prioritize "
                "late-layer trajectories, GT-vs-opposite margins, conditional retention, "
                "and the final-layer lens/native sanity check."
            ),
        ]

        (
            output_dir
            / "report.txt"
        ).write_text(
            "\n".join(
                report_lines
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "\nSaved:"
        )

        for filename in (
            "selected_sids.json",
            "split.json",
            "train_basic.csv",
            "eval_basic.csv",
            "sample_path.csv",
            "receiver_head_summary.csv",
            "receiver_head_top_message.csv",
            "receiver_head_top_local_gain.csv",
            "layer_path_summary.csv",
            "config.json",
            "report.txt",
        ):
            print(
                " ",
                output_dir
                / filename,
            )

    finally:
        if model is not None:
            del model

        if processor is not None:
            del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
