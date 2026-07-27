#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled-A query-swap causal tracing for VLM spatial-relation decisions.

Research question
-----------------
Given the same image, how does the LLM-side computation use the order of the
subject/reference object mentions to form an opposite spatial relation and
transmit it to the final decision?

For every Controlled-A sample, this version rebuilds the question with one
fixed template:

    What is the spatial relation of A relative to B?
    Answer with exactly one word: left, right, on, or under. Do not explain.

The swapped question exchanges A and B while keeping the image and all other
words fixed.  Therefore the correct answer must change to the opposite
relation.  Because the requested answer is the first generated relation label,
the prompt-last position is now the actual relation decision position.

This first-stage script performs:

1. Query-swap baseline
   * unrestricted autoregressive generation for clean and swapped questions;
   * teacher-forced one-label scores for the clean relation r and its opposite;
   * pair status: both_correct / original_only / swapped_only / both_wrong.

2. Clean-state extraction
   * cache decoder-block outputs from the clean question for:
       object A identity,
       object B identity,
       non-object query context,
       answer options,
       prompt-last.

3. Layer x token-group causal tracing
   * run the swapped question;
   * at one decoder layer, replace one selected swapped block-output state with
     the corresponding clean state;
   * recompute the first-answer-label GT-vs-opposite margin.

For the clean relation r, define the fixed-axis margin:

    M(x) = score(r | x) - score(opposite(r) | x)

Ideally:

    M(clean) > 0
    M(swapped) < 0

For a patched swapped run:

    recovery =
        (M(patched) - M(swapped))
        / (M(clean) - M(swapped))

The script does not clip recovery to [0, 1].

What this script can establish
------------------------------
It locates layer/token-group states whose clean activation causally changes the
swapped relation decision back toward the original relation.  It is a node-level
causal-tracing stage.  It does not yet identify exact head-to-head Q/K/V edges;
that should be done after high-recovery layers and token groups are found.

Original repository dependencies
--------------------------------
Run from the AdaptVis repository root.  The script imports:

    analyze_coco_centroid_generation_step1_v4.py
    analyze_coco_flip_same_token_similarity_v1.py
    extract_two_object_relation_states.py
    extract_controlledA_relation_states_standalone.py

Controlled-A is loaded through the original repository loader:

    extract_controlledA_relation_states_standalone.py::load_records
        -> dataset_zoo.get_dataset("Controlled_Images_A")

Main outputs
------------
<output-dir>/
    config.json
    baseline_pairs.jsonl
    baseline_pairs.csv
    baseline_report.txt
    errors.jsonl
    cache/<sid>.npz
    causal_patch.jsonl
    causal_patch_summary.csv
    causal_report.txt
    recovery_heatmap.png

Recommended pilot
-----------------
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_controlledA_query_swap_causal_tracing_v2.py \
  --model qwen-3b \
  --phase all \
  --cache-layers all \
  --patch-layers stride:4 \
  --patch-conditions object_a,object_b,object_pair,query_context,options,last \
  --patch-max-samples 80 \
  --generation-max-new-tokens 16 \
  --candidate-style label \
  --device cuda:0 \
  --output-dir output/controlledA_query_swap_causal/qwen-3b

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_controlledA_query_swap_causal_tracing_v2.py \
  --model qwen-3b \
  --phase all \
  --cache-layers all \
  --patch-layers stride:8 \
  --max-samples 10 \
  --patch-max-samples 4 \
  --generation-max-new-tokens 16 \
  --device cuda:0 \
  --output-dir output/controlledA_query_swap_causal_smoke/qwen-3b \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "controlledA-query-swap-causal-tracing-v2"

RELATIONS = ("left", "right", "on", "under")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "on": "under",
    "under": "on",
}
PAIR_STATUSES = ("both_correct", "original_only", "swapped_only", "both_wrong")
PATCH_CONDITIONS = (
    "object_a",
    "object_b",
    "object_pair",
    "query_context",
    "options",
    "last",
)


@dataclass
class ControlledSample:
    uid: str
    sid: int
    subject_a: str
    reference_b: str
    clean_question: str
    swapped_question: str
    clean_gt: str
    swapped_gt: str
    record: Any


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="qwen-3b")
    parser.add_argument(
        "--phase",
        choices=("all", "baseline", "patch"),
        default="all",
        help=(
            "baseline: generation, margins and clean-state cache; "
            "patch: use existing baseline/cache; all: both."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument(
        "--controlled-script",
        default="extract_controlledA_relation_states_standalone.py",
    )
    parser.add_argument(
        "--controlled-prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
        help=(
            "Used for object names, relation labels, and image alignment. "
            "The actual model question is rebuilt from --question-template."
        ),
    )
    parser.add_argument(
        "--question-template",
        default=(
            "What is the spatial relation of the {subject} relative to the "
            "{reference}? Answer with exactly one word: left, right, on, or "
            "under. Do not explain."
        ),
        help=(
            "Fixed clean/swap question template. It must contain both "
            "{subject} and {reference}."
        ),
    )
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    parser.add_argument(
        "--semantic-helper",
        default="analyze_coco_flip_same_token_similarity_v1.py",
    )
    parser.add_argument(
        "--two-object-script",
        default="extract_two_object_relation_states.py",
        help="Used to construct the merged repository model registry.",
    )

    parser.add_argument(
        "--cache-layers",
        default="all",
        help="'all', 'stride:N', or comma-separated decoder-layer indices.",
    )
    parser.add_argument(
        "--patch-layers",
        default="stride:4",
        help="'all', 'stride:N', or comma-separated decoder-layer indices.",
    )
    parser.add_argument(
        "--patch-conditions",
        default="object_a,object_b,object_pair,query_context,options,last",
    )
    parser.add_argument(
        "--patch-status",
        choices=(
            "both_correct_margin",
            "both_correct",
            "all_margin",
            "all",
        ),
        default="both_correct_margin",
        help="Which baseline pairs are eligible for causal patching.",
    )
    parser.add_argument(
        "--patch-max-samples",
        type=int,
        default=80,
        help=(
            "Maximum eligible pairs patched in this run. Use 0 or a negative "
            "value for all eligible pairs."
        ),
    )
    parser.add_argument(
        "--min-denominator",
        type=float,
        default=1e-4,
        help="Minimum abs(clean_margin - swapped_margin) for recovery.",
    )

    parser.add_argument(
        "--candidate-style",
        choices=("sentence", "label"),
        default="label",
        help=(
            "Use label for the main experiment. With the fixed one-word "
            "instruction this scores the relation at the prompt-last decision "
            "position. sentence is retained only as a diagnostic fallback."
        ),
    )
    parser.add_argument(
        "--candidate-reduction",
        choices=("mean", "sum"),
        default="mean",
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--array-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Files / imports / numerics
# -----------------------------------------------------------------------------


def import_file(path: Path, module_name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def sanitize_uid(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if text:
        return text[:160]
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:20]


def safe_mean(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text.startswith("stride:"):
        stride = int(text.split(":", 1)[1])
        if stride < 1:
            raise ValueError("Layer stride must be >= 1")
        layers = list(range(0, n_layers, stride))
        if n_layers - 1 not in layers:
            layers.append(n_layers - 1)
        return sorted(set(layers))

    layers: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside [0, {n_layers - 1}]")
        layers.append(layer)
    if not layers:
        raise ValueError("Layer selection is empty")
    return sorted(set(layers))


def parse_conditions(value: str) -> List[str]:
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    invalid = sorted(set(result) - set(PATCH_CONDITIONS))
    if invalid:
        raise ValueError(
            f"Unsupported patch conditions {invalid}; allowed={PATCH_CONDITIONS}"
        )
    if not result:
        raise ValueError("No patch conditions selected")
    return list(dict.fromkeys(result))


def pair_status(clean_correct: bool, swapped_correct: bool) -> str:
    if clean_correct and swapped_correct:
        return "both_correct"
    if clean_correct:
        return "original_only"
    if swapped_correct:
        return "swapped_only"
    return "both_wrong"


def normalize_native_relation(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z]+", " ", text).strip()
    tokens = text.split()

    # Prefer explicit horizontal/under/above words before standalone "on".
    for token in tokens:
        if token == "left":
            return "left"
        if token == "right":
            return "right"
        if token in {"under", "below", "beneath"}:
            return "under"
        if token in {"above", "over"}:
            return "on"
    if re.search(r"\bon\s+top\s+of\b", text):
        return "on"
    if "on" in tokens:
        return "on"
    return ""


def parse_generation_relation(text: Any) -> str:
    value = str(text).strip().lower()
    candidates: List[Tuple[int, str]] = []
    patterns = (
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bbelow\b", "under"),
        (r"\bbeneath\b", "under"),
        (r"\babove\b", "on"),
        (r"\bover\b", "on"),
        (r"\bon\s+top\s+of\b", "on"),
    )
    for pattern, relation in patterns:
        match = re.search(pattern, value)
        if match is not None:
            candidates.append((match.start(), relation))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    match = re.search(r"\bon\b", value)
    if match is not None:
        return "on"
    return ""


# -----------------------------------------------------------------------------
# Question swapping / dataset
# -----------------------------------------------------------------------------


def swap_object_mentions(question: str, subject: str, reference: str) -> str:
    """Swap object mentions while preserving the original prompt/options."""
    if subject == reference:
        raise ValueError("Subject and reference are identical")

    sentinel_a = "__QUERY_SWAP_OBJECT_A_7F3D__"
    sentinel_b = "__QUERY_SWAP_OBJECT_B_9C2E__"

    # Exact surface replacement first. The prompt file and object strings are
    # expected to use the same surface forms.
    if subject in question and reference in question:
        swapped = question.replace(subject, sentinel_a)
        swapped = swapped.replace(reference, sentinel_b)
        swapped = swapped.replace(sentinel_a, reference)
        swapped = swapped.replace(sentinel_b, subject)
    else:
        pattern_a = re.compile(re.escape(subject), flags=re.IGNORECASE)
        pattern_b = re.compile(re.escape(reference), flags=re.IGNORECASE)
        if pattern_a.search(question) is None or pattern_b.search(question) is None:
            raise ValueError(
                f"Could not find both object mentions in question: "
                f"subject={subject!r}, reference={reference!r}, question={question!r}"
            )
        swapped = pattern_a.sub(sentinel_a, question)
        swapped = pattern_b.sub(sentinel_b, swapped)
        swapped = swapped.replace(sentinel_a, reference)
        swapped = swapped.replace(sentinel_b, subject)

    if swapped == question:
        raise RuntimeError("Question swap produced no change")
    return swapped


def load_samples(
    *,
    controlled: Any,
    base: Any,
    prompt_path: Path,
    question_template: str,
    max_samples: Optional[int],
    num_workers: int,
    download: bool,
) -> Tuple[List[ControlledSample], Any]:
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    records, audit = controlled.load_records(
        prompt_path,
        download=bool(download),
        max_samples=max_samples,
        num_workers=int(num_workers),
    )
    prompts = base.load_standard_prompts(prompt_path)

    samples: List[ControlledSample] = []
    for record in records:
        sid = int(record.sid)
        prompt = prompts.get(sid)
        if prompt is None:
            continue

        clean_gt = normalize_native_relation(getattr(record, "relation", ""))
        if clean_gt not in OPPOSITE:
            continue

        subject = str(prompt["subject"])
        reference = str(prompt["reference"])

        try:
            clean_question = str(question_template).format(
                subject=subject,
                reference=reference,
            )
            swapped_question = str(question_template).format(
                subject=reference,
                reference=subject,
            )
        except KeyError as exc:
            raise ValueError(
                "--question-template must contain only the named placeholders "
                "{subject} and {reference}"
            ) from exc

        if clean_question == swapped_question:
            raise RuntimeError(
                f"Query swap produced no change for sid={sid}: "
                f"subject={subject!r}, reference={reference!r}"
            )

        samples.append(
            ControlledSample(
                uid=sanitize_uid(sid),
                sid=sid,
                subject_a=subject,
                reference_b=reference,
                clean_question=clean_question,
                swapped_question=swapped_question,
                clean_gt=clean_gt,
                swapped_gt=OPPOSITE[clean_gt],
                record=record,
            )
        )

    if not samples:
        raise RuntimeError("No usable Controlled-A query-swap samples")
    return samples, audit


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model_and_processor(
    *,
    model_key: str,
    base: Any,
    two_object: Any,
    args: argparse.Namespace,
) -> Tuple[Any, Any, Any]:
    specs = base.merged_model_specs(two_object)
    if model_key not in specs:
        raise ValueError(
            f"Unknown model key {model_key!r}; available={sorted(specs)}"
        )
    spec = specs[model_key]
    model_class = getattr(transformers, spec.model_class, None)
    if model_class is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} lacks {spec.model_class}"
        )

    dtype = (
        base.resolve_dtype(args.dtype)
        if args.dtype
        else base.resolve_dtype(spec.dtype_name)
    )
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl

    print(
        f"Loading model={model_key} repo={spec.repo_id} dtype={dtype}",
        flush=True,
    )
    try:
        model = model_class.from_pretrained(spec.repo_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = model_class.from_pretrained(spec.repo_id, **kwargs)

    model.eval()
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if hasattr(generation_config, "do_sample"):
            generation_config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, name):
                setattr(generation_config, name, None)

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    return model, processor, spec


# -----------------------------------------------------------------------------
# Token groups
# -----------------------------------------------------------------------------


def span_positions(span: Sequence[int]) -> List[int]:
    start, end = int(span[0]), int(span[1])
    return list(range(start, end + 1))


def locate_option_synonyms(
    *,
    semantic_helper: Any,
    tokenizer: Any,
    input_ids: Sequence[int],
    reference_span: Sequence[int],
    question_end: int,
) -> List[int]:
    positions: List[int] = []
    for word in ("left", "right", "on", "under", "above", "below", "beneath"):
        spans = semantic_helper.locate_phrase_spans(tokenizer, input_ids, word)
        span = semantic_helper.choose_span(
            spans,
            min_start=int(reference_span[1]) + 1,
            max_end=int(question_end),
            prefer="first",
        )
        if span is not None:
            positions.extend(range(int(span[0]), int(span[1]) + 1))
    return sorted(set(positions))


def build_question_groups(
    *,
    base: Any,
    semantic_helper: Any,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    question_text: str,
    subject_span: Sequence[int],
    reference_span: Sequence[int],
) -> Dict[str, List[int]]:
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    visual_positions = base.resolve_visual_indices(
        model,
        processor,
        dict(batch),
        input_ids,
    )
    visual_set = set(map(int, visual_positions))
    text_positions = [
        index for index in range(len(input_ids))
        if index not in visual_set
    ]

    excluded = (
        set(span_positions(subject_span))
        | set(span_positions(reference_span))
        | {len(input_ids) - 1}
    )

    try:
        semantic = semantic_helper.locate_semantic_spans(
            processor.tokenizer,
            input_ids,
            question_text,
            (int(subject_span[0]), int(subject_span[1])),
            (int(reference_span[0]), int(reference_span[1])),
            text_positions,
        )
        question_span = list(semantic.get("_question_span", []))
        question_end = (
            int(question_span[1])
            if len(question_span) == 2
            else max(text_positions)
        )
        options = sorted(set(map(int, semantic.get("option_all", []))))
        options = sorted(set(options + locate_option_synonyms(
            semantic_helper=semantic_helper,
            tokenizer=processor.tokenizer,
            input_ids=input_ids,
            reference_span=reference_span,
            question_end=question_end,
        )))
        query_context = sorted(set(
            list(map(int, semantic.get("where", [])))
            + list(map(int, semantic.get("copula", [])))
            + list(map(int, semantic.get("relation_connector", [])))
            + list(map(int, semantic.get("relation_keyword", [])))
            + list(map(int, semantic.get("connector_to", [])))
            + list(map(int, semantic.get("answer_instruction", [])))
            + list(map(int, semantic.get("question_other", [])))
        ))
        query_context = [
            position for position in query_context
            if position not in excluded and position not in set(options)
        ]
        options = [
            position for position in options
            if position not in excluded
        ]
    except Exception:
        # Conservative fallback: all text positions except object spans, prompt
        # last and explicit relation-option tokens.
        options = []
        for word in ("left", "right", "on", "under", "above", "below", "beneath"):
            for span in semantic_helper.locate_phrase_spans(
                processor.tokenizer,
                input_ids,
                word,
            ):
                options.extend(range(int(span[0]), int(span[1]) + 1))
        options = sorted(set(options) - excluded)
        query_context = sorted(
            set(text_positions) - excluded - set(options)
        )

    return {
        "query_context": list(map(int, query_context)),
        "options": list(map(int, options)),
        "last": [len(input_ids) - 1],
        "visual_positions": list(map(int, visual_positions)),
    }


# -----------------------------------------------------------------------------
# Candidate continuation scoring
# -----------------------------------------------------------------------------


def relation_sentence(
    relation: str,
    subject: str,
    reference: str,
) -> str:
    if relation == "left":
        return f"The {subject} is to the left of the {reference}."
    if relation == "right":
        return f"The {subject} is to the right of the {reference}."
    if relation == "on":
        return f"The {subject} is on the {reference}."
    if relation == "under":
        return f"The {subject} is under the {reference}."
    raise ValueError(relation)


def candidate_text(
    *,
    style: str,
    relation: str,
    subject: str,
    reference: str,
) -> str:
    if style == "label":
        return relation
    return relation_sentence(relation, subject, reference)


def encode_continuation(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
    )
    ids = encoded.input_ids
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = [int(value) for value in ids]
    if not ids:
        raise RuntimeError(f"Continuation tokenized to zero tokens: {text!r}")
    return ids


def extend_batch_with_continuation(
    batch: Mapping[str, Any],
    continuation_ids: Sequence[int],
) -> Tuple[Dict[str, Any], int]:
    output: Dict[str, Any] = {}
    prompt_ids = batch["input_ids"]
    if int(prompt_ids.shape[0]) != 1:
        raise RuntimeError("Candidate scoring expects batch size 1")
    prompt_length = int(prompt_ids.shape[1])

    suffix = torch.as_tensor(
        list(map(int, continuation_ids)),
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    ).unsqueeze(0)
    output["input_ids"] = torch.cat([prompt_ids, suffix], dim=1)

    for key, value in batch.items():
        if key == "input_ids":
            continue
        if key in {"position_ids", "cache_position", "labels"}:
            # Let the model recompute positions for the longer sequence.
            continue
        if key == "attention_mask" and torch.is_tensor(value):
            extension = torch.ones(
                (int(value.shape[0]), len(continuation_ids)),
                device=value.device,
                dtype=value.dtype,
            )
            output[key] = torch.cat([value, extension], dim=-1)
            continue
        if key == "token_type_ids" and torch.is_tensor(value):
            extension = torch.zeros(
                (int(value.shape[0]), len(continuation_ids)),
                device=value.device,
                dtype=value.dtype,
            )
            output[key] = torch.cat([value, extension], dim=-1)
            continue
        output[key] = value

    return output, prompt_length


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim >= 3:
                return item
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if torch.is_tensor(value) and value.ndim >= 3:
            return value
    raise TypeError(
        f"Could not locate hidden tensor in output type {type(output).__name__}"
    )


def replace_first_tensor(output: Any, modified: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return modified
    if isinstance(output, tuple):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim >= 3:
                items[index] = modified
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim >= 3:
                items[index] = modified
                return items
    raise TypeError(
        f"Could not replace hidden tensor in output type {type(output).__name__}"
    )


class BlockOutputPatch:
    """Replace selected target positions in one decoder block output."""

    def __init__(
        self,
        module: torch.nn.Module,
        mappings: Sequence[Tuple[Sequence[int], np.ndarray]],
    ):
        self.mappings = [
            (
                list(map(int, target_positions)),
                np.asarray(source_states, dtype=np.float32),
            )
            for target_positions, source_states in mappings
        ]
        self.applied = False
        self.alignment_modes: List[str] = []
        self.handle = module.register_forward_hook(self._hook)

    @staticmethod
    def adapt_source(
        source: torch.Tensor,
        target_count: int,
    ) -> Tuple[torch.Tensor, str]:
        if target_count < 1:
            raise ValueError("Target count must be positive")
        if int(source.shape[0]) == target_count:
            return source, "positionwise"
        if int(source.shape[0]) == 1:
            return source.expand(target_count, -1), "broadcast_single"
        mean = source.mean(dim=0, keepdim=True)
        return mean.expand(target_count, -1), "broadcast_mean"

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden = first_tensor(output)
        if hidden.ndim != 3:
            raise RuntimeError(
                f"Expected decoder output [B,T,D], got {tuple(hidden.shape)}"
            )
        modified = hidden.clone()
        modes: List[str] = []

        for target_positions, source_np in self.mappings:
            if not target_positions:
                continue
            if max(target_positions) >= int(hidden.shape[1]):
                raise RuntimeError(
                    f"Patch target {max(target_positions)} outside sequence "
                    f"length {int(hidden.shape[1])}"
                )
            source = torch.as_tensor(
                source_np,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if source.ndim == 1:
                source = source.unsqueeze(0)
            adapted, mode = self.adapt_source(
                source,
                len(target_positions),
            )
            index = torch.as_tensor(
                target_positions,
                device=hidden.device,
                dtype=torch.long,
            )
            modified[0].index_copy_(0, index, adapted)
            modes.append(mode)

        if not modes:
            raise RuntimeError("BlockOutputPatch received no usable mappings")
        self.applied = True
        self.alignment_modes = modes
        return replace_first_tensor(output, modified)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self) -> "BlockOutputPatch":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@torch.inference_mode()
def score_continuation(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    continuation: str,
    reduction: str,
    patch_module: Optional[torch.nn.Module] = None,
    patch_mappings: Optional[
        Sequence[Tuple[Sequence[int], np.ndarray]]
    ] = None,
) -> Dict[str, Any]:
    ids = encode_continuation(processor.tokenizer, continuation)
    extended, prompt_length = extend_batch_with_continuation(batch, ids)

    patch: Optional[BlockOutputPatch] = None
    try:
        if patch_module is not None:
            if not patch_mappings:
                raise ValueError("patch_module supplied without patch mappings")
            patch = BlockOutputPatch(patch_module, patch_mappings)

        outputs = model(
            **extended,
            use_cache=False,
            return_dict=True,
        )
        if patch is not None and not patch.applied:
            raise RuntimeError("Block-output patch hook did not fire")

        logits = outputs.logits[0]
        token_logits = logits[
            prompt_length - 1 : prompt_length + len(ids) - 1
        ].float()
        log_probs = torch.log_softmax(token_logits, dim=-1)
        target = torch.as_tensor(
            ids,
            device=log_probs.device,
            dtype=torch.long,
        )
        selected = log_probs.gather(1, target[:, None]).squeeze(1)

        if reduction == "sum":
            score = float(selected.sum().item())
        else:
            score = float(selected.mean().item())

        return {
            "score": score,
            "token_logprobs": [
                float(value) for value in selected.detach().cpu().tolist()
            ],
            "token_ids": ids,
            "token_count": len(ids),
            "alignment_modes": (
                list(patch.alignment_modes) if patch is not None else []
            ),
        }
    finally:
        if patch is not None:
            patch.close()
        with contextlib.suppress(Exception):
            del outputs
        del extended


def score_relation_axis(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    clean_relation: str,
    subject: str,
    reference: str,
    style: str,
    reduction: str,
    patch_module: Optional[torch.nn.Module] = None,
    patch_mappings: Optional[
        Sequence[Tuple[Sequence[int], np.ndarray]]
    ] = None,
) -> Dict[str, Any]:
    opposite = OPPOSITE[clean_relation]
    clean_text = candidate_text(
        style=style,
        relation=clean_relation,
        subject=subject,
        reference=reference,
    )
    opposite_text = candidate_text(
        style=style,
        relation=opposite,
        subject=subject,
        reference=reference,
    )

    clean_result = score_continuation(
        model=model,
        processor=processor,
        batch=batch,
        continuation=clean_text,
        reduction=reduction,
        patch_module=patch_module,
        patch_mappings=patch_mappings,
    )
    opposite_result = score_continuation(
        model=model,
        processor=processor,
        batch=batch,
        continuation=opposite_text,
        reduction=reduction,
        patch_module=patch_module,
        patch_mappings=patch_mappings,
    )

    margin = float(clean_result["score"] - opposite_result["score"])
    return {
        "clean_relation": clean_relation,
        "opposite_relation": opposite,
        "clean_candidate": clean_text,
        "opposite_candidate": opposite_text,
        "clean_score": clean_result["score"],
        "opposite_score": opposite_result["score"],
        "margin": margin,
        "prediction_on_axis": clean_relation if margin >= 0 else opposite,
        "clean_candidate_token_count": clean_result["token_count"],
        "opposite_candidate_token_count": opposite_result["token_count"],
        "clean_candidate_tokenlogprobs": clean_result["token_logprobs"],
        "opposite_candidate_tokenlogprobs": opposite_result["token_logprobs"],
        "alignment_modes": sorted(set(
            clean_result["alignment_modes"]
            + opposite_result["alignment_modes"]
        )),
    }


# -----------------------------------------------------------------------------
# Clean-state capture / cache
# -----------------------------------------------------------------------------


class CleanGroupCapture:
    """Capture selected clean prompt positions at selected decoder blocks."""

    def __init__(
        self,
        decoder_layers: Sequence[torch.nn.Module],
        selected_layers: Sequence[int],
        groups: Mapping[str, Sequence[int]],
    ):
        self.selected_layers = list(map(int, selected_layers))
        self.groups = {
            str(name): list(map(int, positions))
            for name, positions in groups.items()
        }
        self.values: Dict[str, Dict[int, torch.Tensor]] = {
            name: {} for name in self.groups
        }
        self.handles = [
            decoder_layers[layer].register_forward_hook(
                self._make_hook(layer)
            )
            for layer in self.selected_layers
        ]

    def _make_hook(self, layer: int):
        def hook(
            module: torch.nn.Module,
            inputs: Tuple[Any, ...],
            output: Any,
        ) -> Any:
            hidden = first_tensor(output)
            for name, positions in self.groups.items():
                if positions:
                    index = torch.as_tensor(
                        positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    value = hidden[0].index_select(0, index)
                else:
                    value = hidden.new_zeros((0, int(hidden.shape[-1])))
                self.values[name][layer] = (
                    value.detach().float().cpu()
                )
            return output
        return hook

    def close(self) -> None:
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles = []

    def arrays(self) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {
            "layers": np.asarray(self.selected_layers, dtype=np.int16)
        }
        for name in self.groups:
            missing = [
                layer for layer in self.selected_layers
                if layer not in self.values[name]
            ]
            if missing:
                raise RuntimeError(
                    f"Missing clean captures for group={name}: {missing}"
                )
            arrays[name] = np.stack(
                [
                    self.values[name][layer].numpy()
                    for layer in self.selected_layers
                ],
                axis=0,
            ).astype(np.float32)
        return arrays

    def __enter__(self) -> "CleanGroupCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def cache_path(cache_dir: Path, sid: int) -> Path:
    return cache_dir / f"{sanitize_uid(sid)}.npz"


def save_cache(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    dtype: str,
) -> None:
    payload: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(
                np.float16 if dtype == "float16" else np.float32
            )
        payload[key] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
        }


def cache_layer_index(
    cache: Mapping[str, np.ndarray],
    layer: int,
) -> int:
    layers = np.asarray(cache["layers"], dtype=np.int64)
    matches = np.where(layers == int(layer))[0]
    if len(matches) != 1:
        raise KeyError(
            f"Layer {layer} absent or duplicated in cache layers "
            f"{layers.tolist()}"
        )
    return int(matches[0])


# -----------------------------------------------------------------------------
# Baseline phase
# -----------------------------------------------------------------------------


def run_baseline_phase(
    *,
    args: argparse.Namespace,
    samples: Sequence[ControlledSample],
    base: Any,
    semantic_helper: Any,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[torch.nn.Module],
    cache_layers: Sequence[int],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    baseline_path = output_dir / "baseline_pairs.jsonl"
    errors_path = output_dir / "errors.jsonl"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    completed = {
        int(row["sid"]): row
        for row in read_jsonl(baseline_path)
    } if args.resume else {}

    successful = len(completed)
    device = torch.device(args.device)

    for sample in tqdm(
        samples,
        desc=f"{args.model}:controlledA-query-swap-baseline",
        total=len(samples),
    ):
        cpath = cache_path(cache_dir, sample.sid)
        if sample.sid in completed and cpath.exists():
            continue

        image: Optional[Image.Image] = None
        clean_batch: Optional[Dict[str, Any]] = None
        swap_batch: Optional[Dict[str, Any]] = None
        try:
            image = base.record_image(sample.record)

            clean_batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=sample.clean_question,
                device=device,
            )
            clean_ids = (
                clean_batch["input_ids"][0].detach().cpu().tolist()
            )
            clean_a_span, clean_b_span = base.locate_object_spans(
                processor.tokenizer,
                clean_ids,
                sample.subject_a,
                sample.reference_b,
            )
            clean_groups_extra = build_question_groups(
                base=base,
                semantic_helper=semantic_helper,
                model=model,
                processor=processor,
                batch=clean_batch,
                question_text=sample.clean_question,
                subject_span=clean_a_span,
                reference_span=clean_b_span,
            )

            clean_capture_groups = {
                "object_a": span_positions(clean_a_span),
                "object_b": span_positions(clean_b_span),
                "query_context": clean_groups_extra["query_context"],
                "options": clean_groups_extra["options"],
                "last": clean_groups_extra["last"],
            }

            with CleanGroupCapture(
                decoder_layers,
                cache_layers,
                clean_capture_groups,
            ) as capture:
                with torch.inference_mode():
                    clean_prompt_outputs = model(
                        **clean_batch,
                        use_cache=False,
                        return_dict=True,
                    )
            arrays = capture.arrays()
            save_cache(cpath, arrays, args.array_dtype)

            clean_axis = score_relation_axis(
                model=model,
                processor=processor,
                batch=clean_batch,
                clean_relation=sample.clean_gt,
                subject=sample.subject_a,
                reference=sample.reference_b,
                style=args.candidate_style,
                reduction=args.candidate_reduction,
            )
            clean_generation = base.generate_text(
                model,
                processor,
                dict(clean_batch),
                max_new_tokens=args.generation_max_new_tokens,
            )
            clean_pred = parse_generation_relation(clean_generation)

            swap_batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=sample.swapped_question,
                device=device,
            )
            swap_ids = (
                swap_batch["input_ids"][0].detach().cpu().tolist()
            )

            # In the swapped question B is the grammatical subject and A is the
            # reference. locate_object_spans therefore receives B, A.
            swap_b_subject_span, swap_a_reference_span = (
                base.locate_object_spans(
                    processor.tokenizer,
                    swap_ids,
                    sample.reference_b,
                    sample.subject_a,
                )
            )
            swap_groups_extra = build_question_groups(
                base=base,
                semantic_helper=semantic_helper,
                model=model,
                processor=processor,
                batch=swap_batch,
                question_text=sample.swapped_question,
                subject_span=swap_b_subject_span,
                reference_span=swap_a_reference_span,
            )

            # The fixed axis is always the original relation r versus opposite(r).
            # Candidate sentences use the current swapped query roles B relative
            # to A.
            swap_axis = score_relation_axis(
                model=model,
                processor=processor,
                batch=swap_batch,
                clean_relation=sample.clean_gt,
                subject=sample.reference_b,
                reference=sample.subject_a,
                style=args.candidate_style,
                reduction=args.candidate_reduction,
            )
            swap_generation = base.generate_text(
                model,
                processor,
                dict(swap_batch),
                max_new_tokens=args.generation_max_new_tokens,
            )
            swap_pred = parse_generation_relation(swap_generation)

            clean_correct = clean_pred == sample.clean_gt
            swap_correct = swap_pred == sample.swapped_gt
            status = pair_status(clean_correct, swap_correct)
            margin_sign_valid = (
                float(clean_axis["margin"]) > 0
                and float(swap_axis["margin"]) < 0
            )
            denominator = float(
                clean_axis["margin"] - swap_axis["margin"]
            )

            row = {
                "script_version": SCRIPT_VERSION,
                "model": args.model,
                "dataset": "controlled_a",
                "uid": sample.uid,
                "sid": sample.sid,
                "subject_a": sample.subject_a,
                "reference_b": sample.reference_b,
                "clean_question": sample.clean_question,
                "swapped_question": sample.swapped_question,
                "clean_gt": sample.clean_gt,
                "swapped_gt": sample.swapped_gt,
                "clean_generation": clean_generation,
                "clean_generation_pred": clean_pred,
                "clean_generation_correct": clean_correct,
                "swapped_generation": swap_generation,
                "swapped_generation_pred": swap_pred,
                "swapped_generation_correct": swap_correct,
                "pair_status": status,
                "candidate_style": args.candidate_style,
                "candidate_reduction": args.candidate_reduction,
                "clean_margin": float(clean_axis["margin"]),
                "swapped_margin_fixed_axis": float(swap_axis["margin"]),
                "margin_denominator": denominator,
                "margin_sign_valid": bool(margin_sign_valid),
                "clean_axis_prediction": clean_axis["prediction_on_axis"],
                "swapped_axis_prediction": swap_axis["prediction_on_axis"],
                "clean_candidate": clean_axis["clean_candidate"],
                "clean_opposite_candidate": clean_axis["opposite_candidate"],
                "swapped_clean_candidate": swap_axis["clean_candidate"],
                "swapped_opposite_candidate": swap_axis["opposite_candidate"],
                "clean_score_gt": float(clean_axis["clean_score"]),
                "clean_score_opposite": float(clean_axis["opposite_score"]),
                "swapped_score_clean_relation": float(
                    swap_axis["clean_score"]
                ),
                "swapped_score_opposite_relation": float(
                    swap_axis["opposite_score"]
                ),
                "clean_a_span": list(map(int, clean_a_span)),
                "clean_b_span": list(map(int, clean_b_span)),
                "swap_a_span": list(map(int, swap_a_reference_span)),
                "swap_b_span": list(map(int, swap_b_subject_span)),
                "clean_query_context": clean_groups_extra["query_context"],
                "clean_options": clean_groups_extra["options"],
                "clean_last": clean_groups_extra["last"],
                "swap_query_context": swap_groups_extra["query_context"],
                "swap_options": swap_groups_extra["options"],
                "swap_last": swap_groups_extra["last"],
                "clean_visual_positions": clean_groups_extra["visual_positions"],
                "swap_visual_positions": swap_groups_extra["visual_positions"],
                "cache_file": str(cpath),
            }
            append_jsonl(baseline_path, row)
            completed[sample.sid] = row
            successful += 1

            if args.print_every > 0 and successful % args.print_every == 0:
                print("\n" + "=" * 100, flush=True)
                print(
                    f"sid={sample.sid} status={status} "
                    f"clean_gt={sample.clean_gt} clean_pred={clean_pred or '<unparsed>'} "
                    f"swap_gt={sample.swapped_gt} swap_pred={swap_pred or '<unparsed>'}",
                    flush=True,
                )
                print(
                    f"clean_margin={clean_axis['margin']:+.6f} "
                    f"swap_fixed_margin={swap_axis['margin']:+.6f} "
                    f"sign_valid={int(margin_sign_valid)}",
                    flush=True,
                )
                print(f"clean question:\n{sample.clean_question}", flush=True)
                print(f"clean generation:\n{clean_generation}", flush=True)
                print(f"swapped question:\n{sample.swapped_question}", flush=True)
                print(f"swapped generation:\n{swap_generation}", flush=True)
                print("=" * 100, flush=True)

            del clean_prompt_outputs, arrays
        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "phase": "baseline",
                    "model": args.model,
                    "sid": sample.sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"\n[ERROR baseline sid={sample.sid}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            if clean_batch is not None:
                del clean_batch
            if swap_batch is not None:
                del swap_batch
            gc.collect()
            if torch.cuda.is_available() and (
                args.empty_cache_every <= 1
                or max(successful, 1) % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    rows = list(completed.values())
    rows.sort(key=lambda row: int(row["sid"]))

    # Rewrite deduplicated baseline output after resume.
    baseline_path.unlink(missing_ok=True)
    for row in rows:
        append_jsonl(baseline_path, row)
    write_csv(output_dir / "baseline_pairs.csv", rows)
    write_baseline_report(output_dir, rows)
    return rows


def write_baseline_report(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    clean_acc = safe_mean(
        int(bool(row["clean_generation_correct"])) for row in rows
    )
    swap_acc = safe_mean(
        int(bool(row["swapped_generation_correct"])) for row in rows
    )
    both_acc = safe_mean(
        int(row["pair_status"] == "both_correct") for row in rows
    )
    sign_rate = safe_mean(
        int(bool(row["margin_sign_valid"])) for row in rows
    )
    clean_axis_acc = safe_mean(
        int(float(row["clean_margin"]) > 0) for row in rows
    )
    swap_axis_acc = safe_mean(
        int(float(row["swapped_margin_fixed_axis"]) < 0) for row in rows
    )
    clean_axis_generation_agreement = safe_mean(
        int(
            str(row["clean_axis_prediction"])
            == str(row["clean_generation_pred"])
        )
        for row in rows
        if str(row["clean_generation_pred"])
    )
    swap_axis_generation_agreement = safe_mean(
        int(
            str(row["swapped_axis_prediction"])
            == str(row["swapped_generation_pred"])
        )
        for row in rows
        if str(row["swapped_generation_pred"])
    )

    status_counts: Dict[str, int] = {
        status: 0 for status in PAIR_STATUSES
    }
    for row in rows:
        status_counts[str(row["pair_status"])] += 1

    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"N: {len(rows)}",
        f"clean generation accuracy: {clean_acc:.4f}",
        f"swapped generation accuracy: {swap_acc:.4f}",
        f"both-correct pair rate: {both_acc:.4f}",
        f"clean candidate-axis accuracy: {clean_axis_acc:.4f}",
        f"swapped candidate-axis accuracy: {swap_axis_acc:.4f}",
        (
            "clean candidate/generation agreement: "
            f"{clean_axis_generation_agreement:.4f}"
        ),
        (
            "swapped candidate/generation agreement: "
            f"{swap_axis_generation_agreement:.4f}"
        ),
        f"clean/swap margin sign-valid rate: {sign_rate:.4f}",
        f"pair status counts: {status_counts}",
        "",
        "The causal patch metric is reliable only when its candidate-axis",
        "predictions agree sufficiently with unrestricted generation and the",
        "clean/swapped margins usually have opposite signs.",
    ]
    (output_dir / "baseline_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\nBASELINE SUMMARY")
    for line in lines[1:8]:
        print(line)


# -----------------------------------------------------------------------------
# Patch phase
# -----------------------------------------------------------------------------


def baseline_row_eligible(
    row: Mapping[str, Any],
    mode: str,
    min_denominator: float,
) -> bool:
    status_ok = (
        row["pair_status"] == "both_correct"
        if mode in {"both_correct_margin", "both_correct"}
        else True
    )
    margin_ok = (
        bool(row["margin_sign_valid"])
        if mode in {"both_correct_margin", "all_margin"}
        else True
    )
    denominator_ok = (
        abs(float(row["margin_denominator"])) >= min_denominator
    )
    return bool(status_ok and margin_ok and denominator_ok)


def condition_mappings(
    *,
    condition: str,
    cache: Mapping[str, np.ndarray],
    cache_layer_index_value: int,
    row: Mapping[str, Any],
) -> List[Tuple[Sequence[int], np.ndarray]]:
    def source(name: str) -> np.ndarray:
        return np.asarray(
            cache[name][cache_layer_index_value],
            dtype=np.float32,
        )

    if condition == "object_a":
        return [(row["swap_a_span"], source("object_a"))]
    if condition == "object_b":
        return [(row["swap_b_span"], source("object_b"))]
    if condition == "object_pair":
        return [
            (row["swap_a_span"], source("object_a")),
            (row["swap_b_span"], source("object_b")),
        ]
    if condition == "query_context":
        return [
            (row["swap_query_context"], source("query_context"))
        ]
    if condition == "options":
        return [(row["swap_options"], source("options"))]
    if condition == "last":
        return [(row["swap_last"], source("last"))]
    raise ValueError(condition)


def run_patch_phase(
    *,
    args: argparse.Namespace,
    samples_by_sid: Mapping[int, ControlledSample],
    baseline_rows: Sequence[Mapping[str, Any]],
    base: Any,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[torch.nn.Module],
    patch_layers: Sequence[int],
    patch_conditions: Sequence[str],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    patch_path = output_dir / "causal_patch.jsonl"
    errors_path = output_dir / "errors.jsonl"
    cache_dir = output_dir / "cache"

    completed_rows = read_jsonl(patch_path) if args.resume else []
    completed_keys = {
        (
            int(row["sid"]),
            int(row["layer"]),
            str(row["condition"]),
        )
        for row in completed_rows
    }

    eligible = [
        row for row in baseline_rows
        if baseline_row_eligible(
            row,
            args.patch_status,
            args.min_denominator,
        )
    ]
    eligible.sort(key=lambda row: int(row["sid"]))
    if args.patch_max_samples is not None and args.patch_max_samples > 0:
        eligible = eligible[: int(args.patch_max_samples)]

    print(
        f"Patch eligibility mode={args.patch_status}: "
        f"{len(eligible)} samples, layers={list(patch_layers)}, "
        f"conditions={list(patch_conditions)}",
        flush=True,
    )
    if not eligible:
        raise RuntimeError("No baseline pairs eligible for causal patching")

    device = torch.device(args.device)
    total = len(eligible) * len(patch_layers) * len(patch_conditions)
    progress = tqdm(total=total, desc="query-swap-block-causal-patch")
    new_rows = 0

    for baseline in eligible:
        sid = int(baseline["sid"])
        sample = samples_by_sid.get(sid)
        if sample is None:
            raise KeyError(f"Sample missing sid={sid}")

        image: Optional[Image.Image] = None
        swap_batch: Optional[Dict[str, Any]] = None
        cache: Optional[Dict[str, np.ndarray]] = None

        try:
            cpath = cache_path(cache_dir, sid)
            if not cpath.exists():
                raise FileNotFoundError(cpath)
            cache = load_cache(cpath)

            image = base.record_image(sample.record)
            swap_batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=sample.swapped_question,
                device=device,
            )

            clean_margin = float(baseline["clean_margin"])
            swap_margin = float(baseline["swapped_margin_fixed_axis"])
            denominator = float(clean_margin - swap_margin)

            for layer in patch_layers:
                cache_index = cache_layer_index(cache, layer)
                module = decoder_layers[int(layer)]

                for condition in patch_conditions:
                    key = (sid, int(layer), str(condition))
                    if key in completed_keys:
                        progress.update(1)
                        continue

                    try:
                        mappings = condition_mappings(
                            condition=condition,
                            cache=cache,
                            cache_layer_index_value=cache_index,
                            row=baseline,
                        )
                        if any(len(target) == 0 for target, _ in mappings):
                            raise RuntimeError(
                                f"Empty target token group for condition={condition}"
                            )
                        if any(int(source.shape[0]) == 0 for _, source in mappings):
                            raise RuntimeError(
                                f"Empty clean source group for condition={condition}"
                            )

                        patched = score_relation_axis(
                            model=model,
                            processor=processor,
                            batch=swap_batch,
                            clean_relation=sample.clean_gt,
                            subject=sample.reference_b,
                            reference=sample.subject_a,
                            style=args.candidate_style,
                            reduction=args.candidate_reduction,
                            patch_module=module,
                            patch_mappings=mappings,
                        )
                        patched_margin = float(patched["margin"])
                        margin_shift = float(patched_margin - swap_margin)
                        recovery = float(margin_shift / denominator)

                        row = {
                            "script_version": SCRIPT_VERSION,
                            "model": args.model,
                            "dataset": "controlled_a",
                            "sid": sid,
                            "uid": baseline["uid"],
                            "clean_gt": sample.clean_gt,
                            "swapped_gt": sample.swapped_gt,
                            "pair_status": baseline["pair_status"],
                            "layer": int(layer),
                            "component": "block_output",
                            "condition": condition,
                            "clean_margin": clean_margin,
                            "swapped_margin_fixed_axis": swap_margin,
                            "patched_margin_fixed_axis": patched_margin,
                            "margin_denominator": denominator,
                            "margin_shift_from_swapped": margin_shift,
                            "recovery": recovery,
                            "patched_axis_prediction": (
                                sample.clean_gt
                                if patched_margin >= 0
                                else sample.swapped_gt
                            ),
                            "crossed_to_clean_side": bool(patched_margin > 0),
                            "positive_recovery": bool(recovery > 0),
                            "alignment_modes": patched["alignment_modes"],
                            "clean_score_gt": patched["clean_score"],
                            "opposite_score": patched["opposite_score"],
                        }
                        append_jsonl(patch_path, row)
                        completed_rows.append(row)
                        completed_keys.add(key)
                        new_rows += 1
                    except Exception as exc:
                        append_jsonl(
                            errors_path,
                            {
                                "phase": "patch",
                                "model": args.model,
                                "sid": sid,
                                "layer": int(layer),
                                "condition": condition,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        print(
                            f"\n[ERROR patch sid={sid} L{layer} "
                            f"{condition}] {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if args.fail_fast:
                            raise
                    finally:
                        progress.update(1)

            if args.print_every > 0 and new_rows > 0:
                print(
                    f"\n[patched sid={sid}] clean={clean_margin:+.5f} "
                    f"swap={swap_margin:+.5f}",
                    flush=True,
                )
        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            if swap_batch is not None:
                del swap_batch
            if cache is not None:
                del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress.close()

    completed_rows.sort(
        key=lambda row: (
            int(row["layer"]),
            str(row["condition"]),
            int(row["sid"]),
        )
    )
    patch_path.unlink(missing_ok=True)
    for row in completed_rows:
        append_jsonl(patch_path, row)

    summary = summarize_patch_rows(completed_rows)
    write_csv(output_dir / "causal_patch_summary.csv", summary)
    write_causal_report(output_dir, summary, len(eligible))
    make_heatmap(output_dir, summary)
    return completed_rows


def summarize_patch_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["layer"]), str(row["condition"]))].append(row)

    output: List[Dict[str, Any]] = []
    for (layer, condition), group in sorted(groups.items()):
        output.append(
            {
                "layer": layer,
                "condition": condition,
                "N": len(group),
                "mean_recovery": safe_mean(
                    row["recovery"] for row in group
                ),
                "median_recovery": safe_median(
                    row["recovery"] for row in group
                ),
                "mean_margin_shift": safe_mean(
                    row["margin_shift_from_swapped"] for row in group
                ),
                "positive_recovery_rate": safe_mean(
                    int(bool(row["positive_recovery"])) for row in group
                ),
                "crossed_to_clean_side_rate": safe_mean(
                    int(bool(row["crossed_to_clean_side"])) for row in group
                ),
            }
        )
    return output


def write_causal_report(
    output_dir: Path,
    summary: Sequence[Mapping[str, Any]],
    n_eligible: int,
) -> None:
    ranked = sorted(
        summary,
        key=lambda row: float(row["mean_recovery"]),
        reverse=True,
    )
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"eligible samples: {n_eligible}",
        "",
        "Top layer x token-group block-output patches:",
    ]
    for row in ranked[:20]:
        lines.append(
            f"L{int(row['layer'])} {row['condition']:14s} "
            f"N={int(row['N']):4d} "
            f"meanR={float(row['mean_recovery']):+.6f} "
            f"medianR={float(row['median_recovery']):+.6f} "
            f"dM={float(row['mean_margin_shift']):+.6f} "
            f"positive={float(row['positive_recovery_rate']):.4f} "
            f"crossed={float(row['crossed_to_clean_side_rate']):.4f}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  object_a/object_b/object_pair: clean state is aligned by object identity,",
            "  not by grammatical slot.  For example, clean A-subject is patched into",
            "  swapped A-reference.",
            "  query_context/options/last are patched by token-group order; unequal",
            "  lengths use mean broadcasting and are marked in alignment_modes.",
            "  The fixed one-word instruction makes prompt-last the relation-label",
            "  decision position, so last-state patching is now directly interpretable.",
            "  This remains node-level causal tracing, not an exact sender->receiver path.",
        ]
    )
    (output_dir / "causal_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\nCAUSAL TRACE SUMMARY")
    for line in lines[3:13]:
        print(line)


def make_heatmap(
    output_dir: Path,
    summary: Sequence[Mapping[str, Any]],
) -> None:
    if plt is None or not summary:
        return

    layers = sorted({int(row["layer"]) for row in summary})
    conditions = [
        condition for condition in PATCH_CONDITIONS
        if any(row["condition"] == condition for row in summary)
    ]
    matrix = np.full(
        (len(conditions), len(layers)),
        np.nan,
        dtype=np.float64,
    )
    layer_lookup = {layer: index for index, layer in enumerate(layers)}
    condition_lookup = {
        condition: index for index, condition in enumerate(conditions)
    }
    for row in summary:
        matrix[
            condition_lookup[str(row["condition"])],
            layer_lookup[int(row["layer"])],
        ] = float(row["mean_recovery"])

    width = max(8.0, 0.55 * len(layers))
    height = max(4.0, 0.7 * len(conditions))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Patched clean token group")
    ax.set_title("Controlled-A query-swap block-output recovery")
    fig.colorbar(image, ax=ax, label="Mean normalized recovery")
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_heatmap.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.generation_max_new_tokens < 1:
        raise ValueError("--generation-max-new-tokens must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = import_file(
        Path(args.base_script),
        "query_swap_causal_base",
    )
    semantic_helper = import_file(
        Path(args.semantic_helper),
        "query_swap_causal_semantic",
    )
    two_object = import_file(
        Path(args.two_object_script),
        "query_swap_causal_two_object",
    )
    controlled = import_file(
        Path(args.controlled_script),
        "query_swap_causal_controlled",
    )

    samples, audit = load_samples(
        controlled=controlled,
        base=base,
        prompt_path=Path(args.controlled_prompt_jsonl),
        question_template=args.question_template,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        download=args.download,
    )
    samples_by_sid = {sample.sid: sample for sample in samples}

    model = None
    processor = None
    try:
        model, processor, spec = load_model_and_processor(
            model_key=args.model,
            base=base,
            two_object=two_object,
            args=args,
        )
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        cache_layers = parse_layers(args.cache_layers, len(decoder_layers))
        patch_layers = parse_layers(args.patch_layers, len(decoder_layers))
        missing_patch_layers = sorted(set(patch_layers) - set(cache_layers))
        if missing_patch_layers:
            raise ValueError(
                f"Patch layers absent from cache layers: {missing_patch_layers}"
            )
        patch_conditions = parse_conditions(args.patch_conditions)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": "controlled_a",
            "counterfactual": (
                "same image; fixed one-word question; swap subject/reference "
                "object roles"
            ),
            "question_template": args.question_template,
            "controlled_prompt_jsonl": args.controlled_prompt_jsonl,
            "controlled_script": args.controlled_script,
            "n_samples": len(samples),
            "n_decoder_layers": len(decoder_layers),
            "decoder_path": decoder_path,
            "cache_layers": cache_layers,
            "patch_layers": patch_layers,
            "patch_conditions": patch_conditions,
            "patch_status": args.patch_status,
            "patch_max_samples": args.patch_max_samples,
            "candidate_style": args.candidate_style,
            "candidate_reduction": args.candidate_reduction,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "array_dtype": args.array_dtype,
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        baseline_rows: List[Dict[str, Any]]
        if args.phase in {"all", "baseline"}:
            baseline_rows = run_baseline_phase(
                args=args,
                samples=samples,
                base=base,
                semantic_helper=semantic_helper,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                cache_layers=cache_layers,
                output_dir=output_dir,
            )
        else:
            baseline_rows = read_jsonl(
                output_dir / "baseline_pairs.jsonl"
            )
            if not baseline_rows:
                raise RuntimeError(
                    "--phase patch requires existing baseline_pairs.jsonl"
                )

        if args.phase in {"all", "patch"}:
            run_patch_phase(
                args=args,
                samples_by_sid=samples_by_sid,
                baseline_rows=baseline_rows,
                base=base,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                patch_layers=patch_layers,
                patch_conditions=patch_conditions,
                output_dir=output_dir,
            )

        print(f"\nSaved outputs to {output_dir}", flush=True)
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
