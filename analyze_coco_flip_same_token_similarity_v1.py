#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COCO left/right original-vs-horizontal-flip same-token similarity.

The prompt is held fixed. For every left/right sample, the image is horizontally
flipped and the decoder-block output is captured at every selected layer.

For each text-token position i and layer l, compare the original and flipped
hidden states at the same position:

    cosine(h_orig[l,i], h_flip[l,i])
    relative_l2 = ||h_orig-h_flip|| / mean(||h_orig||, ||h_flip||)

The script does not filter on generation correctness. It records all left/right
samples, their baseline predictions, per-token similarities, and pooled
subject/reference/semantic-group similarities.

This is a descriptive representation analysis, not a causal intervention.
No detector, centroid, trained probe, or model-weight update is used.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import shutil
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

VERSION = "coco-flip-same-token-similarity-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
TOKEN_GROUPS = (
    "subject",
    "subject_first",
    "subject_last",
    "reference",
    "reference_first",
    "reference_last",
    "both",
    "where",
    "copula",
    "relation_connector",
    "relation_keyword",
    "connector_to",
    "answer_instruction",
    "answer",
    "with",
    "one",
    "spatial",
    "answer_relation",
    "option_left",
    "option_right",
    "option_above",
    "option_below",
    "option_all",
    "question_last",
    "chat_prefix",
    "chat_suffix",
    "question_other",
    "prompt_last",
    "all_text",
)
DIRECTIONS = ("orig_to_flip", "flip_to_orig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument(
        "--dataset",
        default="coco_two",
        choices=["coco_two"],
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--relations",
        default="left,right",
        help="This analysis should normally remain left,right.",
    )
    p.add_argument(
        "--layers",
        default="all",
        help="'all', 'auto:N', or explicit zero-based block indices.",
    )
    p.add_argument(
        "--groups",
        default=(
            "subject,reference,both,where,copula,relation_connector,"
            "relation_keyword,connector_to,answer_instruction,option_left,"
            "option_right,option_above,option_below,option_all,question_last,"
            "prompt_last,chat_prefix,chat_suffix,question_other,all_text"
        ),
        help="Comma-separated semantic groups for pooled similarity.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_subset(value: str, allowed: Sequence[str], label: str) -> List[str]:
    allowed = set(allowed)
    out: List[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported {label}: {item}; allowed={sorted(allowed)}")
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = value.strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text.startswith("auto:"):
        stride = int(text.split(":", 1)[1])
        if stride <= 0:
            raise ValueError("auto stride must be positive")
        layers = list(range(stride - 1, n_layers, stride))
        if not layers or layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)
        return sorted(set(layers))
    out: List[int] = []
    for raw in text.split(","):
        if not raw.strip():
            continue
        layer = int(raw)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} outside 0..{n_layers - 1}")
        if layer not in out:
            out.append(layer)
    if not out:
        raise ValueError("No layers selected")
    return out


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported layer output type: {type(output).__name__}")


def replace_first(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    raise TypeError(type(output).__name__)


class CaptureBlockOutputs:
    def __init__(self, layers: Sequence[Any], indices: Sequence[int]) -> None:
        self.layers = layers
        self.indices = list(indices)
        self.handles: List[Any] = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "CaptureBlockOutputs":
        for index in self.indices:
            def make_hook(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    self.outputs[layer_index] = first_tensor(output).detach()
                return hook
            self.handles.append(self.layers[index].register_forward_hook(make_hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


class PatchBlockOutput:
    """Patch one block output for multiple token groups in a single batched pass."""

    def __init__(
        self,
        layer: Any,
        donor_hidden: torch.Tensor,
        position_sets: Sequence[Sequence[int]],
        sequence_length: int,
    ) -> None:
        self.layer = layer
        self.donor_hidden = donor_hidden
        self.position_sets = [sorted(set(map(int, positions))) for positions in position_sets]
        self.sequence_length = int(sequence_length)
        self.handle: Optional[Any] = None
        self.events = 0

    def __enter__(self) -> "PatchBlockOutput":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_tensor(output)
            if hidden.ndim != 3:
                raise RuntimeError(f"Expected [B,S,H], got {tuple(hidden.shape)}")
            batch, seq_len, hidden_size = hidden.shape
            if seq_len != self.sequence_length:
                raise RuntimeError(f"Sequence length changed: {seq_len} != {self.sequence_length}")
            if batch != len(self.position_sets):
                raise RuntimeError(f"Batch/condition mismatch: {batch} != {len(self.position_sets)}")
            donor = self.donor_hidden
            if tuple(donor.shape[1:]) != (seq_len, hidden_size):
                raise RuntimeError(f"Donor shape {tuple(donor.shape)} incompatible with {tuple(hidden.shape)}")
            donor = donor.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden.clone()
            for b, positions in enumerate(self.position_sets):
                valid = [p for p in positions if 0 <= p < seq_len]
                if not valid:
                    raise RuntimeError(f"Empty patch position set for row {b}")
                idx = torch.tensor(valid, device=hidden.device, dtype=torch.long)
                patched[b].index_copy_(0, idx, donor[0].index_select(0, idx))
            self.events += 1
            return replace_first(output, patched)

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("No language-model logits found")


def score_relations(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for relation in RELATIONS:
        ids = [int(x) for x in token_map[relation] if 0 <= int(x) < logits.numel()]
        if not ids:
            raise RuntimeError(f"No token variants for {relation}")
        idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
        scores[relation] = float(logits.index_select(0, idx).max().detach().cpu())
    return scores


def run_forward(
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    capture_layers: Sequence[int] = (),
) -> Tuple[List[Dict[str, Any]], Dict[int, torch.Tensor]]:
    with torch.inference_mode():
        if capture_layers:
            with CaptureBlockOutputs(decoder_layers, capture_layers) as capture:
                outputs = model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            captured = dict(capture.outputs)
        else:
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            captured = {}
    logits = extract_logits(outputs)[:, -1, :]
    results: List[Dict[str, Any]] = []
    for i in range(logits.shape[0]):
        scores = score_relations(logits[i], token_map)
        prediction = max(RELATIONS, key=lambda r: scores[r])
        results.append({"scores": scores, "prediction": prediction})
    del outputs, logits
    return results, captured


def flip_image(image: Image.Image, relation: str) -> Image.Image:
    if relation in ("left", "right"):
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if relation in ("above", "below"):
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    raise ValueError(relation)



def find_subsequence_starts(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> List[int]:
    sequence = list(map(int, sequence))
    pattern = list(map(int, pattern))
    if not pattern or len(pattern) > len(sequence):
        return []
    width = len(pattern)
    return [
        start
        for start in range(len(sequence) - width + 1)
        if sequence[start : start + width] == pattern
    ]


def tokenize_without_specials(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def locate_phrase_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    phrase: str,
) -> List[Tuple[int, int]]:
    phrase = str(phrase)
    variants = []
    for candidate in (
        phrase,
        " " + phrase,
        phrase.strip(),
        " " + phrase.strip(),
    ):
        if candidate and candidate not in variants:
            variants.append(candidate)

    spans = set()
    for candidate in variants:
        token_ids = tokenize_without_specials(tokenizer, candidate)
        for start in find_subsequence_starts(input_ids, token_ids):
            spans.add((start, start + len(token_ids) - 1))
    return sorted(spans)


def choose_span(
    spans: Sequence[Tuple[int, int]],
    *,
    min_start: Optional[int] = None,
    max_end: Optional[int] = None,
    prefer: str = "last",
) -> Optional[Tuple[int, int]]:
    valid = []
    for start, end in spans:
        if min_start is not None and start < min_start:
            continue
        if max_end is not None and end > max_end:
            continue
        valid.append((int(start), int(end)))
    if not valid:
        return None
    if prefer == "first":
        return min(valid, key=lambda item: (item[0], item[1]))
    return max(valid, key=lambda item: (item[0], item[1]))


def span_to_positions(
    span: Optional[Tuple[int, int]],
) -> List[int]:
    if span is None:
        return []
    return list(range(int(span[0]), int(span[1]) + 1))


def decode_single_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return str(text).replace("\n", "\\n")


def normalize_token_text(text: str) -> str:
    value = str(text).replace("\n", "\\n")
    value = value.replace("\t", "\\t")
    value = value.strip()
    return value if value else "<blank>"


def locate_semantic_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    question: str,
    subject_span: Tuple[int, int],
    reference_span: Tuple[int, int],
    text_positions: Sequence[int],
) -> Dict[str, List[int]]:
    """
    Locate stable semantic roles in the rendered chat prompt.

    The COCO prompt may use either "in relation to" or "relative to".
    Missing phrases simply produce an empty group and are skipped later.
    """
    subject = span_positions(subject_span)
    reference = span_positions(reference_span)
    subject_start, subject_end = map(int, subject_span)
    reference_start, reference_end = map(int, reference_span)
    text = sorted(set(map(int, text_positions)))

    question_span = choose_span(
        locate_phrase_spans(tokenizer, input_ids, question),
        prefer="last",
    )
    question_start = question_span[0] if question_span else min(text)
    question_end = question_span[1] if question_span else max(text)

    where_span = choose_span(
        locate_phrase_spans(tokenizer, input_ids, "Where"),
        max_end=subject_start - 1,
        prefer="last",
    )

    copula_span = None
    for phrase in ("is", "are"):
        candidate = choose_span(
            locate_phrase_spans(tokenizer, input_ids, phrase),
            max_end=subject_start - 1,
            prefer="last",
        )
        if candidate is not None:
            if copula_span is None or candidate[0] > copula_span[0]:
                copula_span = candidate

    connector_span = None
    connector_phrase = None
    for phrase in (
        "in relation to",
        "relative to",
        "with respect to",
    ):
        candidate = choose_span(
            locate_phrase_spans(tokenizer, input_ids, phrase),
            min_start=subject_end + 1,
            max_end=reference_start - 1,
            prefer="last",
        )
        if candidate is not None:
            connector_span = candidate
            connector_phrase = phrase
            break

    relation_keyword_span = None
    if connector_phrase is not None:
        keyword = "relative" if "relative" in connector_phrase else "relation"
        relation_keyword_span = choose_span(
            locate_phrase_spans(tokenizer, input_ids, keyword),
            min_start=subject_end + 1,
            max_end=reference_start - 1,
            prefer="last",
        )

    connector_to_span = choose_span(
        locate_phrase_spans(tokenizer, input_ids, "to"),
        min_start=(relation_keyword_span[1] + 1 if relation_keyword_span else subject_end + 1),
        max_end=reference_start - 1,
        prefer="first",
    )

    question_text = str(question)
    question_mark_index = question_text.find("?")
    instruction_text = (
        question_text[question_mark_index + 1 :].strip()
        if question_mark_index >= 0
        else ""
    )
    instruction_span = (
        choose_span(
            locate_phrase_spans(tokenizer, input_ids, instruction_text),
            min_start=reference_end + 1,
            max_end=question_end,
            prefer="last",
        )
        if instruction_text
        else None
    )

    def after_reference_word(word: str, prefer: str = "first") -> Optional[Tuple[int, int]]:
        return choose_span(
            locate_phrase_spans(tokenizer, input_ids, word),
            min_start=reference_end + 1,
            max_end=question_end,
            prefer=prefer,
        )

    answer_span = after_reference_word("Answer")
    with_span = after_reference_word("with")
    one_span = after_reference_word("one")
    spatial_span = after_reference_word("spatial")
    answer_relation_span = after_reference_word("relation", prefer="last")

    option_left_span = after_reference_word("left")
    option_right_span = after_reference_word("right")
    option_above_span = after_reference_word("above")
    option_below_span = after_reference_word("below")
    option_all = sorted(set(
        span_to_positions(option_left_span)
        + span_to_positions(option_right_span)
        + span_to_positions(option_above_span)
        + span_to_positions(option_below_span)
    ))

    semantic = {
        "subject": subject,
        "subject_first": subject[:1],
        "subject_last": subject[-1:],
        "reference": reference,
        "reference_first": reference[:1],
        "reference_last": reference[-1:],
        "both": sorted(set(subject + reference)),
        "where": span_to_positions(where_span),
        "copula": span_to_positions(copula_span),
        "relation_connector": span_to_positions(connector_span),
        "relation_keyword": span_to_positions(relation_keyword_span),
        "connector_to": span_to_positions(connector_to_span),
        "answer_instruction": span_to_positions(instruction_span),
        "answer": span_to_positions(answer_span),
        "with": span_to_positions(with_span),
        "one": span_to_positions(one_span),
        "spatial": span_to_positions(spatial_span),
        "answer_relation": span_to_positions(answer_relation_span),
        "option_left": span_to_positions(option_left_span),
        "option_right": span_to_positions(option_right_span),
        "option_above": span_to_positions(option_above_span),
        "option_below": span_to_positions(option_below_span),
        "option_all": option_all,
        "question_last": [int(question_end)],
        "chat_prefix": [p for p in text if p < question_start],
        "chat_suffix": [p for p in text if p > question_end],
        "prompt_last": [max(text)],
        "all_text": text,
    }

    known_question = set()
    for key in (
        "subject",
        "reference",
        "where",
        "copula",
        "relation_connector",
        "answer_instruction",
    ):
        known_question.update(semantic.get(key, []))
    semantic["question_other"] = [
        p
        for p in text
        if question_start <= p <= question_end and p not in known_question
    ]
    semantic["_question_span"] = [int(question_start), int(question_end)]
    return semantic


def token_role(
    position: int,
    semantic: Mapping[str, Sequence[int]],
) -> str:
    priority = (
        "subject",
        "reference",
        "relation_keyword",
        "connector_to",
        "relation_connector",
        "where",
        "copula",
        "answer",
        "with",
        "one",
        "spatial",
        "option_left",
        "option_right",
        "option_above",
        "option_below",
        "answer_relation",
        "answer_instruction",
        "chat_prefix",
        "chat_suffix",
        "question_other",
    )
    for role in priority:
        if int(position) in set(map(int, semantic.get(role, []))):
            return role
    return "other_text"


def build_token_manifest(
    tokenizer: Any,
    input_ids: Sequence[int],
    text_positions: Sequence[int],
    semantic: Mapping[str, Sequence[int]],
) -> List[Dict[str, Any]]:
    question_bounds = list(semantic.get("_question_span", []))
    q_start = question_bounds[0] if len(question_bounds) == 2 else None
    q_end = question_bounds[1] if len(question_bounds) == 2 else None

    manifest = []
    for text_rank, position in enumerate(sorted(set(map(int, text_positions)))):
        token_id = int(input_ids[position])
        decoded = decode_single_token(tokenizer, token_id)
        role = token_role(position, semantic)
        role_positions = sorted(set(map(int, semantic.get(role, []))))
        role_rank = role_positions.index(position) if position in role_positions else -1
        manifest.append({
            "position": position,
            "text_rank": text_rank,
            "token_id": token_id,
            "token_text": decoded,
            "token_text_norm": normalize_token_text(decoded),
            "token_role": role,
            "role_rank": role_rank,
            "inside_question": bool(
                q_start is not None
                and q_end is not None
                and q_start <= position <= q_end
            ),
        })
    return manifest

def span_positions(span: Tuple[int, int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def build_conditions(
    token_groups: Sequence[str],
    control: str,
    semantic: Mapping[str, Sequence[int]],
    token_manifest: Sequence[Mapping[str, Any]],
    condition_mode: str,
    token_sweep_scope: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    text = sorted(set(map(int, semantic["all_text"])))
    excluded = set(map(int, semantic.get("both", [])))
    excluded.update(map(int, semantic.get("prompt_last", [])))
    random_candidates = [p for p in text if p not in excluded]

    conditions: List[Dict[str, Any]] = []
    positions: List[List[int]] = []

    if condition_mode in ("semantic", "both"):
        for group_index, group in enumerate(token_groups):
            pos = sorted(set(map(int, semantic.get(group, []))))
            if not pos:
                continue

            conditions.append({
                "condition": group,
                "aggregate_key": group,
                "condition_family": "semantic",
                "token_group": group,
                "token_role": group,
                "token_text": None,
                "token_text_norm": None,
                "token_position": None,
                "text_rank": None,
                "role_rank": None,
                "control": False,
                "control_for": group,
                "n_positions": len(pos),
            })
            positions.append(pos)

            if control == "random_text" and group != "all_text":
                candidates = [p for p in random_candidates if p not in pos]
                if len(candidates) < len(pos):
                    continue
                rng = random.Random(seed + 1009 * (group_index + 1))
                rand_pos = sorted(rng.sample(candidates, len(pos)))
                conditions.append({
                    "condition": f"{group}_random_text",
                    "aggregate_key": f"{group}_random_text",
                    "condition_family": "semantic_control",
                    "token_group": group,
                    "token_role": "random_text",
                    "token_text": None,
                    "token_text_norm": None,
                    "token_position": None,
                    "text_rank": None,
                    "role_rank": None,
                    "control": True,
                    "control_for": group,
                    "n_positions": len(rand_pos),
                })
                positions.append(rand_pos)

    if condition_mode in ("token_sweep", "both"):
        for item in token_manifest:
            if (
                token_sweep_scope == "question_only"
                and not bool(item["inside_question"])
            ):
                continue

            position = int(item["position"])
            role = str(item["token_role"])
            token_norm = str(item["token_text_norm"])
            aggregate_key = f"token::{role}::{token_norm}"

            conditions.append({
                "condition": f"token_pos_{position}",
                "aggregate_key": aggregate_key,
                "condition_family": "token_sweep",
                "token_group": "single_text_token",
                "token_role": role,
                "token_text": item["token_text"],
                "token_text_norm": token_norm,
                "token_position": position,
                "text_rank": int(item["text_rank"]),
                "role_rank": int(item["role_rank"]),
                "inside_question": bool(item["inside_question"]),
                "control": False,
                "control_for": None,
                "n_positions": 1,
            })
            positions.append([position])

    if not conditions:
        raise RuntimeError("No patch conditions were constructed.")
    return conditions, positions


def repeated_batch(
    processor: Any,
    rendered: str,
    image: Image.Image,
    repeats: int,
    device: torch.device,
    base: Any,
) -> Dict[str, Any]:
    batch = processor(
        text=[rendered] * repeats,
        images=[image] * repeats,
        return_tensors="pt",
        padding=True,
    )
    return base.move_batch(batch, device)


def eligible_pair(
    mode: str,
    original_prediction: str,
    flipped_prediction: str,
    original_relation: str,
    flipped_relation: str,
) -> bool:
    if mode == "both_correct":
        return original_prediction == original_relation and flipped_prediction == flipped_relation
    if mode == "both_opposite":
        return OPPOSITE.get(original_prediction) == flipped_prediction
    return True


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    level: str = "condition",
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        if level == "role":
            aggregate_key = (
                f"role::{row.get('condition_family')}::"
                f"{row.get('token_role')}"
            )
        else:
            aggregate_key = row.get(
                "aggregate_key",
                row["condition"],
            )

        key = (
            row["axis"],
            row["direction"],
            int(row["layer"]),
            aggregate_key,
            row.get("condition_family", "semantic"),
            row.get("token_group"),
            row.get("token_role"),
            bool(row["control"]),
            row.get("control_for"),
        )
        grouped[key].append(row)

    result: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        (
            axis,
            direction,
            layer,
            aggregate_key,
            condition_family,
            token_group,
            role,
            control,
            control_for,
        ) = key

        recovery = finite_values(
            item.get("recovery")
            for item in items
        )
        shifts = finite_values(
            item.get("margin_shift")
            for item in items
        )
        unique_sids = len({
            int(item["sid"])
            for item in items
        })

        result.append({
            "axis": axis,
            "direction": direction,
            "layer": layer,
            "aggregate_key": aggregate_key,
            "condition_family": condition_family,
            "token_group": token_group,
            "token_role": role,
            "control": control,
            "control_for": control_for,
            "n_rows": len(items),
            "n_sids": unique_sids,
            "mean_recovery": (
                float(np.mean(recovery))
                if recovery else None
            ),
            "median_recovery": (
                float(np.median(recovery))
                if recovery else None
            ),
            "fraction_recovery_gt_0_25": (
                float(np.mean([x > 0.25 for x in recovery]))
                if recovery else None
            ),
            "fraction_recovery_gt_0_50": (
                float(np.mean([x > 0.50 for x in recovery]))
                if recovery else None
            ),
            "mean_margin_shift": (
                float(np.mean(shifts))
                if shifts else None
            ),
            "donor_relation_rate": float(np.mean([
                item["patched_prediction"]
                == item["donor_relation"]
                for item in items
            ])),
            "recipient_relation_rate": float(np.mean([
                item["patched_prediction"]
                == item["recipient_relation"]
                for item in items
            ])),
        })

    controls = {
        (
            row["axis"],
            row["direction"],
            row["layer"],
            row["control_for"],
        ): row
        for row in result
        if row["control"]
        and row["condition_family"] == "semantic_control"
    }

    for row in result:
        if row["control"]:
            continue
        control = controls.get((
            row["axis"],
            row["direction"],
            row["layer"],
            row["aggregate_key"],
        ))
        if (
            control
            and row["mean_recovery"] is not None
            and control["mean_recovery"] is not None
        ):
            row["excess_recovery_vs_random"] = (
                row["mean_recovery"]
                - control["mean_recovery"]
            )
            row["excess_donor_rate_vs_random"] = (
                row["donor_relation_rate"]
                - control["donor_relation_rate"]
            )
        else:
            row["excess_recovery_vs_random"] = None
            row["excess_donor_rate_vs_random"] = None

    return sorted(
        result,
        key=lambda x: (
            x["axis"],
            x["direction"],
            x["layer"],
            x["control"],
            str(x["aggregate_key"]),
        ),
    )



def safe_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    left = left.float()
    right = right.float()
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    denom = (left_norm * right_norm).clamp_min(eps)
    return (left * right).sum(dim=-1) / denom


def vector_metrics(
    original: torch.Tensor,
    flipped: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    original = original.float()
    flipped = flipped.float()
    original_norm = float(original.norm().detach().cpu())
    flipped_norm = float(flipped.norm().detach().cpu())
    difference = original - flipped
    l2_distance = float(difference.norm().detach().cpu())
    average_norm = max(eps, 0.5 * (original_norm + flipped_norm))
    cosine = float(
        safe_cosine(
            original.unsqueeze(0),
            flipped.unsqueeze(0),
            eps=eps,
        )[0].detach().cpu()
    )
    return {
        "cosine_similarity": cosine,
        "cosine_distance": 1.0 - cosine,
        "l2_distance": l2_distance,
        "relative_l2": l2_distance / average_norm,
        "original_norm": original_norm,
        "flipped_norm": flipped_norm,
        "norm_ratio_flip_over_original": (
            flipped_norm / max(eps, original_norm)
        ),
    }


def token_metrics_matrix(
    original: torch.Tensor,
    flipped: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    original = original.float()
    flipped = flipped.float()
    original_norm = original.norm(dim=-1)
    flipped_norm = flipped.norm(dim=-1)
    difference = original - flipped
    l2_distance = difference.norm(dim=-1)
    average_norm = (
        0.5 * (original_norm + flipped_norm)
    ).clamp_min(eps)
    cosine = safe_cosine(original, flipped, eps=eps)
    return {
        "cosine_similarity": cosine,
        "cosine_distance": 1.0 - cosine,
        "l2_distance": l2_distance,
        "relative_l2": l2_distance / average_norm,
        "original_norm": original_norm,
        "flipped_norm": flipped_norm,
        "norm_ratio_flip_over_original": (
            flipped_norm / original_norm.clamp_min(eps)
        ),
    }


def pair_status(
    original_correct: bool,
    flipped_correct: bool,
) -> str:
    if original_correct and flipped_correct:
        return "both_correct"
    if original_correct:
        return "original_only_correct"
    if flipped_correct:
        return "flipped_only_correct"
    return "both_wrong"


def summarize_numeric(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)

    result: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        out = {
            field: value
            for field, value in zip(group_fields, key)
        }
        out["n_rows"] = len(items)
        out["n_sids"] = len({
            int(item["sid"])
            for item in items
        })
        for metric in metric_fields:
            values = finite_values(
                item.get(metric)
                for item in items
            )
            if values:
                array = np.asarray(values, dtype=np.float64)
                out[f"mean_{metric}"] = float(array.mean())
                out[f"median_{metric}"] = float(np.median(array))
                out[f"q10_{metric}"] = float(np.quantile(array, 0.10))
                out[f"q90_{metric}"] = float(np.quantile(array, 0.90))
            else:
                out[f"mean_{metric}"] = None
                out[f"median_{metric}"] = None
                out[f"q10_{metric}"] = None
                out[f"q90_{metric}"] = None
        result.append(out)

    return sorted(
        result,
        key=lambda row: tuple(
            str(row.get(field))
            for field in group_fields
        ),
    )


def build_sample_summary(
    group_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    baseline_by_sid: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups_of_interest = (
        "subject",
        "reference",
        "both",
        "relation_keyword",
        "option_left",
        "option_right",
        "prompt_last",
        "chat_suffix",
    )
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows:
        group = str(row["group"])
        if group in groups_of_interest:
            grouped[(int(row["sid"]), group)].append(row)

    token_by_sid: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in token_rows:
        token_by_sid[int(row["sid"])].append(row)

    result: List[Dict[str, Any]] = []
    for sid in sorted(baseline_by_sid):
        base_row = dict(baseline_by_sid[sid])
        out: Dict[str, Any] = {
            "sid": sid,
            "subject": base_row.get("subject"),
            "reference": base_row.get("reference"),
            "original_relation": base_row.get("original_relation"),
            "original_prediction": base_row.get("original_prediction"),
            "flipped_prediction": base_row.get("flipped_prediction"),
            "original_correct": base_row.get("original_correct"),
            "flipped_correct": base_row.get("flipped_correct"),
            "pair_status": base_row.get("pair_status"),
        }

        for group in groups_of_interest:
            items = grouped.get((sid, group), [])
            if not items:
                continue
            lowest_cos = min(
                items,
                key=lambda row: float(row["pooled_cosine_similarity"]),
            )
            highest_rel = max(
                items,
                key=lambda row: float(row["pooled_relative_l2"]),
            )
            out[f"{group}_min_pooled_cosine"] = float(
                lowest_cos["pooled_cosine_similarity"]
            )
            out[f"{group}_min_cosine_layer"] = int(
                lowest_cos["layer"]
            )
            out[f"{group}_max_pooled_relative_l2"] = float(
                highest_rel["pooled_relative_l2"]
            )
            out[f"{group}_max_relative_l2_layer"] = int(
                highest_rel["layer"]
            )

        token_items = token_by_sid.get(sid, [])
        if token_items:
            most_changed = min(
                token_items,
                key=lambda row: float(row["cosine_similarity"]),
            )
            out["most_changed_token_layer"] = int(
                most_changed["layer"]
            )
            out["most_changed_token_position"] = int(
                most_changed["position"]
            )
            out["most_changed_token_text"] = most_changed["token_text"]
            out["most_changed_token_role"] = most_changed["token_role"]
            out["most_changed_token_cosine"] = float(
                most_changed["cosine_similarity"]
            )
            out["most_changed_token_relative_l2"] = float(
                most_changed["relative_l2"]
            )

        result.append(out)
    return result


def similarity_report(
    model: str,
    seen: int,
    analyzed: int,
    counts: Mapping[str, int],
    group_summary: Sequence[Mapping[str, Any]],
    role_summary: Sequence[Mapping[str, Any]],
) -> str:
    group_rows = sorted(
        group_summary,
        key=lambda row: (
            float(row["mean_pooled_cosine_similarity"])
            if row.get("mean_pooled_cosine_similarity") is not None
            else 999.0
        ),
    )
    role_rows = sorted(
        role_summary,
        key=lambda row: (
            float(row["mean_cosine_similarity"])
            if row.get("mean_cosine_similarity") is not None
            else 999.0
        ),
    )

    lines = [
        "=" * 126,
        "COCO LEFT/RIGHT ORIGINAL-vs-FLIP SAME-TOKEN SIMILARITY",
        f"model={model} | seen={seen} | analyzed_left_right={analyzed}",
        "baseline: " + ", ".join(
            f"{key}={value}"
            for key, value in sorted(counts.items())
        ),
        "=" * 126,
        "",
        "Lowest pooled semantic-group similarities:",
        f"{'Layer':>7}{'Group':>24}{'Status':>24}{'Nsid':>7}"
        f"{'Mean cosine':>14}{'Mean relL2':>14}{'Min-token cos':>15}",
        "-" * 105,
    ]
    for row in group_rows[:40]:
        lines.append(
            f"{int(row['layer']):>7}"
            f"{str(row['group']):>24}"
            f"{str(row['pair_status']):>24}"
            f"{int(row['n_sids']):>7}"
            f"{float(row['mean_pooled_cosine_similarity']):>14.6f}"
            f"{float(row['mean_pooled_relative_l2']):>14.6f}"
            f"{float(row['mean_min_token_cosine']):>15.6f}"
        )

    lines += [
        "",
        "Lowest same-position token-role similarities:",
        f"{'Layer':>7}{'Role':>24}{'Status':>24}{'Nsid':>7}"
        f"{'Mean cosine':>14}{'Mean relL2':>14}",
        "-" * 90,
    ]
    for row in role_rows[:40]:
        lines.append(
            f"{int(row['layer']):>7}"
            f"{str(row['token_role']):>24}"
            f"{str(row['pair_status']):>24}"
            f"{int(row['n_sids']):>7}"
            f"{float(row['mean_cosine_similarity']):>14.6f}"
            f"{float(row['mean_relative_l2']):>14.6f}"
        )

    lines += [
        "",
        "Notes:",
        "- Lower cosine / higher relative-L2 means the same text-token position is more image-flip-sensitive.",
        "- All left/right samples are included; generation correctness is recorded but not used as a filter.",
        "- Raw cosine can remain close to 1 because lexical and prompt identity dominate the residual stream.",
        "- Pooled subject/reference similarity is the main object-state measurement.",
        "- This comparison is descriptive. It identifies where states differ, not whether those differences causally drive the answer.",
    ]
    return "\n".join(lines) + "\n"


def report_text(
    model: str,
    seen: int,
    clean: int,
    counts: Mapping[str, int],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    actual = [
        row
        for row in rows
        if row["axis"] == "all"
        and not row["control"]
    ]

    semantic = [
        row
        for row in actual
        if row["condition_family"] == "semantic"
    ]
    token_rows = [
        row
        for row in actual
        if row["condition_family"] == "token_sweep"
    ]

    def rank_key(row: Mapping[str, Any]) -> Tuple[float, float]:
        excess = row.get("excess_recovery_vs_random")
        recovery = row.get("mean_recovery")
        return (
            -(
                float(excess)
                if excess is not None
                else -999.0
            ),
            -(
                float(recovery)
                if recovery is not None
                else -999.0
            ),
        )

    semantic.sort(key=rank_key)
    token_rows.sort(
        key=lambda row: -(
            float(row["mean_recovery"])
            if row["mean_recovery"] is not None
            else -999.0
        )
    )

    header = (
        f"{'Family':<13}{'Layer':>7}{'Condition':>34}"
        f"{'Nsid':>7}{'Mean R':>10}{'Median R':>11}"
        f"{'R>0.5':>9}{'Donor%':>9}{'ExR':>9}"
    )
    lines = [
        "=" * len(header),
        "COCO LEFT/RIGHT IMAGE-FLIP TOKEN-SWEEP RESIDUAL PATCHING",
        f"model={model} | seen={seen} | clean_pairs={clean}",
        "baseline: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(counts.items())
        ),
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    def f4(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "-"

    for row in semantic[:30]:
        lines.append(
            f"{'semantic':<13}{row['layer']:>7}"
            f"{str(row['aggregate_key'])[-34:]:>34}"
            f"{row['n_sids']:>7}"
            f"{f4(row['mean_recovery']):>10}"
            f"{f4(row['median_recovery']):>11}"
            f"{f4(row['fraction_recovery_gt_0_50']):>9}"
            f"{f4(row['donor_relation_rate']):>9}"
            f"{f4(row['excess_recovery_vs_random']):>9}"
        )

    lines += [
        "",
        "Top individually patched token identities:",
        header,
        "-" * len(header),
    ]
    for row in token_rows[:40]:
        lines.append(
            f"{'token':<13}{row['layer']:>7}"
            f"{str(row['aggregate_key'])[-34:]:>34}"
            f"{row['n_sids']:>7}"
            f"{f4(row['mean_recovery']):>10}"
            f"{f4(row['median_recovery']):>11}"
            f"{f4(row['fraction_recovery_gt_0_50']):>9}"
            f"{f4(row['donor_relation_rate']):>9}"
            f"{'-':>9}"
        )

    lines += [
        "",
        "Interpretation:",
        "- R≈0: patching that token does not transfer the donor left/right state.",
        "- R≈1: patching that token nearly transfers the donor left/right state.",
        "- Semantic ExR compares the named group with a size-matched random-text control.",
        "- Token-sweep rows patch one text token at a time, including chat-template tokens when requested.",
        "- High recovery on an apparently irrelevant token indicates that the token position acts as an information carrier; it does not imply the token's lexical meaning encodes space.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    relations = parse_subset(
        args.relations,
        RELATIONS,
        "relation",
    )
    if set(relations) - {"left", "right"}:
        print(
            "[WARN] This script is designed for horizontal left/right flips.",
            flush=True,
        )
    groups = parse_subset(
        args.groups,
        TOKEN_GROUPS,
        "group",
    )

    base = import_file(
        Path(args.base_script),
        "_coco_flip_similarity_base",
    )
    data_module = base.import_two_object_module()
    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    prompt_rows = base.load_standard_prompts(
        Path(args.prompt_jsonl)
    )

    specs = base.merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model}; "
            f"available={sorted(specs)}"
        )
    spec = specs[args.model]

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"Output directory not empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline_pairs.jsonl"
    token_manifest_path = out_dir / "token_manifest.jsonl"
    token_path = out_dir / "token_similarity.jsonl"
    group_path = out_dir / "group_similarity.jsonl"
    error_path = out_dir / "errors.jsonl"

    model_cls = getattr(
        transformers,
        spec.model_class,
        None,
    )
    if model_cls is None:
        raise RuntimeError(
            f"transformers lacks {spec.model_class}"
        )

    kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        **kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = (
        base.resolve_decoder_layers(model)
    )
    layers = parse_layers(
        args.layers,
        len(decoder_layers),
    )
    token_map = base.relation_token_variants(
        processor.tokenizer
    )

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "relations": relations,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "layers": layers,
        "groups": groups,
        "max_samples": args.max_samples,
        "audit": audit,
        "comparison": (
            "same text-token position, original image "
            "vs horizontal flip"
        ),
        "capture_location": "decoder_block_output",
        "filters_on_generation_correctness": False,
        "uses_external_model": False,
        "uses_centroid": False,
        "updates_model_weights": False,
    }
    (out_dir / "config.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Decoder={decoder_path}, total_layers={len(decoder_layers)}, "
        f"scan={layers}"
    )
    print(f"Groups={groups}")
    print(
        "All left/right records are analyzed; no both-correct filter."
    )

    seen = 0
    analyzed = 0
    counts: Counter = Counter()
    token_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    baseline_by_sid: Dict[int, Dict[str, Any]] = {}
    start_time = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"flip-similarity:{args.model}",
        ):
            sid = int(record.sid)
            seen += 1
            original_image = None
            flipped_image = None
            original_batch = None
            flipped_batch = None
            original_capture = None
            flipped_capture = None

            try:
                prompt_row = prompt_rows[sid]
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question = str(prompt_row["question_text"])
                original_relation = base.normalize_relation(
                    prompt_row["answer_raw"]
                )
                if original_relation not in relations:
                    continue
                if original_relation not in ("left", "right"):
                    continue

                flipped_relation = OPPOSITE[
                    original_relation
                ]
                original_image = (
                    base.record_image(record)
                    .convert("RGB")
                )
                flipped_image = original_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
                rendered = base.build_prompt(
                    processor,
                    question,
                )

                original_batch = base.move_batch(
                    processor(
                        text=[rendered],
                        images=[original_image],
                        return_tensors="pt",
                    ),
                    device,
                )
                flipped_batch = base.move_batch(
                    processor(
                        text=[rendered],
                        images=[flipped_image],
                        return_tensors="pt",
                    ),
                    device,
                )

                original_ids = (
                    original_batch["input_ids"][0]
                    .detach().cpu().tolist()
                )
                flipped_ids = (
                    flipped_batch["input_ids"][0]
                    .detach().cpu().tolist()
                )
                if original_ids != flipped_ids:
                    raise RuntimeError(
                        "Original/flip tokenization differs."
                    )

                subject_span, reference_span = (
                    base.locate_object_spans(
                        processor.tokenizer,
                        original_ids,
                        subject,
                        reference,
                    )
                )
                visual_indices = base.resolve_visual_indices(
                    model,
                    processor,
                    original_batch,
                    original_ids,
                )
                visual_set = set(
                    map(int, visual_indices)
                )
                text_positions = [
                    position
                    for position in range(len(original_ids))
                    if position not in visual_set
                ]
                if not text_positions:
                    raise RuntimeError(
                        "No text positions resolved."
                    )

                semantic = locate_semantic_spans(
                    processor.tokenizer,
                    original_ids,
                    question,
                    subject_span,
                    reference_span,
                    text_positions,
                )
                token_manifest = build_token_manifest(
                    processor.tokenizer,
                    original_ids,
                    text_positions,
                    semantic,
                )
                manifest_by_position = {
                    int(item["position"]): item
                    for item in token_manifest
                }

                original_result, original_capture = (
                    run_forward(
                        model,
                        original_batch,
                        token_map,
                        decoder_layers,
                        layers,
                    )
                )
                flipped_result, flipped_capture = (
                    run_forward(
                        model,
                        flipped_batch,
                        token_map,
                        decoder_layers,
                        layers,
                    )
                )
                original_result = original_result[0]
                flipped_result = flipped_result[0]

                original_prediction = original_result[
                    "prediction"
                ]
                flipped_prediction = flipped_result[
                    "prediction"
                ]
                original_correct = (
                    original_prediction
                    == original_relation
                )
                flipped_correct = (
                    flipped_prediction
                    == flipped_relation
                )
                status = pair_status(
                    original_correct,
                    flipped_correct,
                )

                counts["eligible_relation_seen"] += 1
                counts["original_correct"] += int(
                    original_correct
                )
                counts["flip_correct"] += int(
                    flipped_correct
                )
                counts["both_correct"] += int(
                    original_correct
                    and flipped_correct
                )
                counts["predictions_opposite"] += int(
                    OPPOSITE.get(original_prediction)
                    == flipped_prediction
                )
                counts[status] += 1

                baseline_row = {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "original_relation": original_relation,
                    "flipped_relation": flipped_relation,
                    "original_prediction": original_prediction,
                    "flipped_prediction": flipped_prediction,
                    "original_correct": bool(
                        original_correct
                    ),
                    "flipped_correct": bool(
                        flipped_correct
                    ),
                    "pair_status": status,
                    "original_scores": original_result[
                        "scores"
                    ],
                    "flipped_scores": flipped_result[
                        "scores"
                    ],
                    "subject_span": list(subject_span),
                    "reference_span": list(reference_span),
                    "question_span": semantic.get(
                        "_question_span"
                    ),
                    "n_text_tokens": len(
                        token_manifest
                    ),
                }
                append_jsonl(
                    baseline_path,
                    baseline_row,
                )
                baseline_by_sid[sid] = baseline_row

                for item in token_manifest:
                    append_jsonl(
                        token_manifest_path,
                        {
                            "sid": sid,
                            "subject": subject,
                            "reference": reference,
                            **item,
                        },
                    )

                text_index_tensor = torch.tensor(
                    text_positions,
                    device=original_capture[
                        layers[0]
                    ].device,
                    dtype=torch.long,
                )

                for layer in layers:
                    original_hidden = original_capture[
                        layer
                    ]
                    flipped_hidden = flipped_capture[
                        layer
                    ]
                    if original_hidden.ndim != 3:
                        raise RuntimeError(
                            f"Layer {layer}: expected [B,S,H], "
                            f"got {tuple(original_hidden.shape)}"
                        )
                    if (
                        original_hidden.shape
                        != flipped_hidden.shape
                    ):
                        raise RuntimeError(
                            f"Layer {layer}: capture shapes differ."
                        )

                    original_text = (
                        original_hidden[0]
                        .index_select(
                            0,
                            text_index_tensor,
                        )
                        .float()
                    )
                    flipped_text = (
                        flipped_hidden[0]
                        .index_select(
                            0,
                            text_index_tensor,
                        )
                        .float()
                    )
                    metrics = token_metrics_matrix(
                        original_text,
                        flipped_text,
                    )

                    position_to_local = {
                        int(position): local
                        for local, position in enumerate(
                            text_positions
                        )
                    }

                    for local, position in enumerate(
                        text_positions
                    ):
                        item = manifest_by_position[
                            int(position)
                        ]
                        token_row = {
                            "sid": sid,
                            "layer": int(layer),
                            "subject": subject,
                            "reference": reference,
                            "original_relation": original_relation,
                            "original_prediction": original_prediction,
                            "flipped_prediction": flipped_prediction,
                            "original_correct": bool(
                                original_correct
                            ),
                            "flipped_correct": bool(
                                flipped_correct
                            ),
                            "pair_status": status,
                            **item,
                            "cosine_similarity": float(
                                metrics[
                                    "cosine_similarity"
                                ][local].detach().cpu()
                            ),
                            "cosine_distance": float(
                                metrics[
                                    "cosine_distance"
                                ][local].detach().cpu()
                            ),
                            "l2_distance": float(
                                metrics[
                                    "l2_distance"
                                ][local].detach().cpu()
                            ),
                            "relative_l2": float(
                                metrics[
                                    "relative_l2"
                                ][local].detach().cpu()
                            ),
                            "original_norm": float(
                                metrics[
                                    "original_norm"
                                ][local].detach().cpu()
                            ),
                            "flipped_norm": float(
                                metrics[
                                    "flipped_norm"
                                ][local].detach().cpu()
                            ),
                            "norm_ratio_flip_over_original": float(
                                metrics[
                                    "norm_ratio_flip_over_original"
                                ][local].detach().cpu()
                            ),
                        }
                        append_jsonl(
                            token_path,
                            token_row,
                        )
                        token_rows.append(
                            token_row
                        )

                    for group in groups:
                        positions = sorted(set(
                            int(position)
                            for position in semantic.get(
                                group,
                                [],
                            )
                            if int(position)
                            in position_to_local
                        ))
                        if not positions:
                            continue

                        local_indices = torch.tensor(
                            [
                                position_to_local[
                                    position
                                ]
                                for position in positions
                            ],
                            device=original_text.device,
                            dtype=torch.long,
                        )
                        original_group = (
                            original_text.index_select(
                                0,
                                local_indices,
                            )
                        )
                        flipped_group = (
                            flipped_text.index_select(
                                0,
                                local_indices,
                            )
                        )
                        pooled_original = (
                            original_group.mean(dim=0)
                        )
                        pooled_flipped = (
                            flipped_group.mean(dim=0)
                        )
                        pooled = vector_metrics(
                            pooled_original,
                            pooled_flipped,
                        )
                        group_token_cos = (
                            safe_cosine(
                                original_group,
                                flipped_group,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        group_token_rel = (
                            (
                                (original_group - flipped_group)
                                .norm(dim=-1)
                                / (
                                    0.5
                                    * (
                                        original_group.norm(dim=-1)
                                        + flipped_group.norm(dim=-1)
                                    )
                                ).clamp_min(1e-12)
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )

                        group_row = {
                            "sid": sid,
                            "layer": int(layer),
                            "group": group,
                            "n_positions": len(
                                positions
                            ),
                            "positions": positions,
                            "subject": subject,
                            "reference": reference,
                            "original_relation": original_relation,
                            "original_prediction": original_prediction,
                            "flipped_prediction": flipped_prediction,
                            "original_correct": bool(
                                original_correct
                            ),
                            "flipped_correct": bool(
                                flipped_correct
                            ),
                            "pair_status": status,
                            "pooled_cosine_similarity": pooled[
                                "cosine_similarity"
                            ],
                            "pooled_cosine_distance": pooled[
                                "cosine_distance"
                            ],
                            "pooled_l2_distance": pooled[
                                "l2_distance"
                            ],
                            "pooled_relative_l2": pooled[
                                "relative_l2"
                            ],
                            "pooled_original_norm": pooled[
                                "original_norm"
                            ],
                            "pooled_flipped_norm": pooled[
                                "flipped_norm"
                            ],
                            "mean_token_cosine": float(
                                group_token_cos.mean()
                            ),
                            "min_token_cosine": float(
                                group_token_cos.min()
                            ),
                            "max_token_cosine": float(
                                group_token_cos.max()
                            ),
                            "mean_token_relative_l2": float(
                                group_token_rel.mean()
                            ),
                            "max_token_relative_l2": float(
                                group_token_rel.max()
                            ),
                        }
                        append_jsonl(
                            group_path,
                            group_row,
                        )
                        group_rows.append(
                            group_row
                        )

                analyzed += 1
                if (
                    args.print_every > 0
                    and analyzed % args.print_every == 0
                ):
                    print(
                        f"[{analyzed}] sid={sid} "
                        f"gt={original_relation} "
                        f"orig={original_prediction} "
                        f"flip={flipped_prediction} "
                        f"status={status}",
                        flush=True,
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            finally:
                for image in (
                    original_image,
                    flipped_image,
                ):
                    if image is not None:
                        try:
                            image.close()
                        except Exception:
                            pass
                del (
                    original_batch,
                    flipped_batch,
                    original_capture,
                    flipped_capture,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        pass

    if not token_rows or not group_rows:
        raise RuntimeError(
            "No similarity rows were generated."
        )

    token_df = None
    group_df = None
    try:
        import pandas as pd
        token_df = pd.DataFrame(token_rows)
        group_df = pd.DataFrame(group_rows)
        token_df.to_csv(
            out_dir / "token_similarity.csv",
            index=False,
        )
        group_df.to_csv(
            out_dir / "group_similarity.csv",
            index=False,
        )
    except Exception as exc:
        print(
            f"[WARN] CSV conversion failed: {exc}",
            flush=True,
        )

    token_role_summary = summarize_numeric(
        token_rows,
        group_fields=(
            "layer",
            "token_role",
            "pair_status",
        ),
        metric_fields=(
            "cosine_similarity",
            "relative_l2",
        ),
    )
    token_identity_summary = summarize_numeric(
        token_rows,
        group_fields=(
            "layer",
            "token_role",
            "role_rank",
            "token_text_norm",
            "pair_status",
        ),
        metric_fields=(
            "cosine_similarity",
            "relative_l2",
        ),
    )
    group_summary = summarize_numeric(
        group_rows,
        group_fields=(
            "layer",
            "group",
            "pair_status",
        ),
        metric_fields=(
            "pooled_cosine_similarity",
            "pooled_relative_l2",
            "mean_token_cosine",
            "min_token_cosine",
        ),
    )
    group_summary_all = summarize_numeric(
        group_rows,
        group_fields=(
            "layer",
            "group",
        ),
        metric_fields=(
            "pooled_cosine_similarity",
            "pooled_relative_l2",
            "mean_token_cosine",
            "min_token_cosine",
        ),
    )
    sample_summary = build_sample_summary(
        group_rows,
        token_rows,
        baseline_by_sid,
    )

    write_csv(
        out_dir / "token_role_summary.csv",
        token_role_summary,
    )
    write_csv(
        out_dir / "token_identity_summary.csv",
        token_identity_summary,
    )
    write_csv(
        out_dir / "group_summary_by_pair_status.csv",
        group_summary,
    )
    write_csv(
        out_dir / "group_summary_all.csv",
        group_summary_all,
    )
    write_csv(
        out_dir / "sample_summary.csv",
        sample_summary,
    )

    report = similarity_report(
        args.model,
        seen,
        analyzed,
        counts,
        group_summary,
        token_role_summary,
    )
    (out_dir / "report.txt").write_text(
        report,
        encoding="utf-8",
    )
    print("\n" + report)

    summary = {
        "version": VERSION,
        "model": args.model,
        "seen": seen,
        "analyzed_left_right": analyzed,
        "counts": dict(counts),
        "elapsed_minutes": (
            time.time() - start_time
        ) / 60.0,
        "output_files": [
            "config.json",
            "baseline_pairs.jsonl",
            "token_manifest.jsonl",
            "token_similarity.jsonl",
            "token_similarity.csv",
            "group_similarity.jsonl",
            "group_similarity.csv",
            "token_role_summary.csv",
            "token_identity_summary.csv",
            "group_summary_by_pair_status.csv",
            "group_summary_all.csv",
            "sample_summary.csv",
            "report.txt",
            "errors.jsonl",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
