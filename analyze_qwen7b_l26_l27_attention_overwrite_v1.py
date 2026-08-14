#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-7B L26-vs-L27 attention overwrite decomposition.

Motivation
==========
Previous true-logit-lens result on Qwen-7B:

    L26:
        x Lens   ~65.6%
        x+a Lens ~71.6%
        y Lens   ~77.0%

    L27:
        x Lens   ~77.0%
        x+a Lens ~69.3%
        y Lens   ~68.8%

So L26 attention is net helpful while L27 attention appears to overwrite an
already-good prompt-last state.

This script asks:

    WHAT inside L27 attention causes the negative change?

It compares L26 (positive-attention control) and L27 (negative-attention target)
at two resolutions:

1) SOURCE GROUP
---------------
For prompt-last attention, every source position is assigned to exactly one of:

    object_text
        subject + reference text tokens

    visual
        image/visual tokens

    relation_words
        the option words left/right/above/below in the prompt, excluding object
        tokens and visual tokens

    prompt_last_self
        the prompt-last token attending to itself

    other_text
        every remaining source position

For each group g:

    c_g = sum_h W_O^h sum_{s in g} A_h[last,s] V_h[s]

The groups reconstruct the complete attention output:

    a ~= c_object + c_visual + c_relation_words
         + c_self + c_other_text

2) QUERY HEAD x SOURCE GROUP
----------------------------
For every query head h and group g:

    c_{h,g} = W_O^h sum_{s in g} A_h[last,s] V_h[s]

TRUE LOGIT-LENS DIAGNOSTICS
===========================
For each layer L:

    x       = block input residual at prompt-last
    a       = attention output after o_proj
    r_attn  = x + a
    y       = block output after MLP

True logit lens:
    state -> model final norm -> LM head
restricted to:
    left / right / above / below

For each source/head message c:

A) isolated add gain:
       margin(x + c) - margin(x)

B) natural necessity gain:
       margin(r_attn) - margin(r_attn - c)

C) removal improvement:
       margin(r_attn - c) - margin(r_attn)

Interpretation:
    removal_improvement > 0  => removing c improves GT-vs-opposite margin,
                                so c is locally damaging in the natural mix.

Special overwrite cohort:
    lens(x) == GT
    AND
    lens(r_attn) != GT

For L27 this directly identifies samples where attention turns a correct
prompt-last readout into an incorrect one.

FULL-GENERATION CAUSAL VALIDATION
=================================
After the natural scan, the script performs real model.generate() interventions:

SOURCE removal:
    attention_out_L[last] -= c_g(clean sample)

HEAD removal:
    attention_out_L[last] -= c_h,total(clean sample)

HEAD x SOURCE removal:
    attention_out_L[last] -= c_{h,g}(clean sample)

The intervention is applied at attention output after o_proj, only at prompt-last
during the full multimodal prefill. Downstream layers are recomputed online.

By default causal validation tests:
    * all 5 source groups at L26 and L27;
    * top-K damaging L27 total heads;
    * top-K damaging L27 head-source pairs.

Candidate ranking is exploratory because GT is used to rank natural margin
damage. The causal ACC should later be confirmed on a fresh held-out set if it
is to be reported as an unbiased improvement.

Dataset / sampling
==================
Default --num-samples 0 = all COCO-two records (typically 440).
This matches the prior run where train/eval 20/80 produced EVAL N=352.

For a quick run:
    --num-samples 200

Default causal subset:
    --causal-max-samples 200
deterministically stratified by relation.

Dependencies in AdaptVis root / PYTHONPATH
==========================================
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py

Example
=======
CUDA_VISIBLE_DEVICES=0 python -u \
analyze_qwen7b_l26_l27_attention_overwrite_v1.py \
  --model qwen-7b \
  --num-samples 0 \
  --causal-max-samples 200 \
  --top-damaging-heads 8 \
  --top-damaging-head-sources 8 \
  --device cuda:0 \
  --output-dir output/qwen7b_l26_l27_attention_overwrite_v1 \
  --overwrite

Quick 200-sample diagnostic:
CUDA_VISIBLE_DEVICES=0 python -u \
analyze_qwen7b_l26_l27_attention_overwrite_v1.py \
  --model qwen-7b \
  --num-samples 200 \
  --causal-max-samples 200 \
  --top-damaging-heads 8 \
  --top-damaging-head-sources 8 \
  --device cuda:0 \
  --output-dir output/qwen7b_l26_l27_attention_overwrite_n200_v1 \
  --overwrite

Outputs
=======
natural_sample_layer.csv
source_group_summary.csv
head_source_summary.csv
top_damaging_l27_heads.csv
top_damaging_l27_head_sources.csv
causal_generation_results.jsonl
causal_generation_summary.csv
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


SCRIPT_VERSION = "qwen7b-l26-l27-attention-overwrite-v1"

RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

SOURCE_GROUPS = (
    "object_text",
    "visual",
    "relation_words",
    "prompt_last_self",
    "other_text",
)

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

EPS = 1e-12


# =============================================================================
# CLI / basic utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-7b")
    p.add_argument(
        "--dataset",
        default="coco_two",
        choices=("coco_two",),
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--layers",
        default="26,27",
        help="Comparison layers; default L26 positive control vs L27 target.",
    )

    p.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="0 = all records; positive = deterministic random subset.",
    )
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--causal-max-samples",
        type=int,
        default=200,
        help="0 = all natural-scan samples; otherwise stratified causal subset.",
    )
    p.add_argument(
        "--top-damaging-heads",
        type=int,
        default=8,
    )
    p.add_argument(
        "--top-damaging-head-sources",
        type=int,
        default=8,
    )
    p.add_argument(
        "--rank-cohort",
        default="overwrite",
        choices=("overwrite", "all", "generation_wrong"),
        help="Cohort used to rank damaging L27 heads/head-source pairs.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager",),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )
    p.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
    )

    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument(
        "--empty-cache-every",
        type=int,
        default=5,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            vals = range(a, b + step, step)
        else:
            vals = [int(item)]
        for v in vals:
            if v not in seen:
                out.append(v)
                seen.add(v)
    return out


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
        m = re.search(rf"\b{re.escape(relation)}\b", text)
        if m:
            hits.append((m.start(), relation))

    for pattern, relation in (
        (r"\bunder(?:neath)?\b|\bbeneath\b", "below"),
        (r"\bover\b|\bon top\b", "above"),
    ):
        m = re.search(pattern, text)
        if m:
            hits.append((m.start(), relation))

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
    path.parent.mkdir(parents=True, exist_ok=True)

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

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def append_jsonl(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, name):
            setattr(cfg, name, None)


# =============================================================================
# Dataset sampling
# =============================================================================

def select_records(
    records: Sequence[Any],
    *,
    n: int,
    seed: int,
) -> List[Any]:
    ordered = sorted(records, key=lambda r: int(r.sid))

    if n <= 0 or n >= len(ordered):
        return ordered

    selected = random.Random(seed).sample(ordered, int(n))
    selected.sort(key=lambda r: int(r.sid))
    return selected


def stratified_limit_records(
    records: Sequence[Any],
    *,
    n: int,
    seed: int,
) -> List[Any]:
    records = list(records)

    if n <= 0 or n >= len(records):
        return sorted(records, key=lambda r: int(r.sid))

    groups: Dict[str, List[Any]] = defaultdict(list)

    for record in records:
        gt = normalize_relation(record.relation)
        if gt in RELATIONS:
            groups[gt].append(record)

    rng = random.Random(seed)

    for group in groups.values():
        rng.shuffle(group)

    out: List[Any] = []
    cursors = Counter()

    while len(out) < n:
        moved = False

        for relation in RELATIONS:
            group = groups.get(relation, [])
            cursor = int(cursors[relation])

            if cursor < len(group) and len(out) < n:
                out.append(group[cursor])
                cursors[relation] += 1
                moved = True

        if not moved:
            break

    out.sort(key=lambda r: int(r.sid))
    return out


# =============================================================================
# Prompt / token helpers
# =============================================================================

def relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}

    for relation in RELATIONS:
        ids = set()

        for text in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))

        if not ids:
            token_ids = tokenizer.encode(
                " " + relation,
                add_special_tokens=False,
            )
            if not token_ids:
                raise RuntimeError(f"No token variant for {relation}.")
            ids.add(int(token_ids[-1]))

        out[relation] = sorted(ids)

    return out


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


def find_all_subsequence_positions(
    haystack: Sequence[int],
    needle: Sequence[int],
) -> List[int]:
    haystack = list(map(int, haystack))
    needle = list(map(int, needle))

    if not needle:
        return []

    n = len(needle)
    positions: List[int] = []

    for start in range(len(haystack) - n + 1):
        if haystack[start:start + n] == needle:
            positions.extend(range(start, start + n))

    return positions


def locate_relation_word_positions(
    *,
    tokenizer: Any,
    input_ids: Sequence[int],
    excluded: Set[int],
) -> List[int]:
    hits: Set[int] = set()

    for relation in RELATIONS:
        for text in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            try:
                needle = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            except Exception:
                needle = []

            if not needle:
                continue

            for position in find_all_subsequence_positions(input_ids, needle):
                if int(position) not in excluded:
                    hits.add(int(position))

    return sorted(hits)


def candidate_token_id(
    tokenizer: Any,
    token: str,
) -> Optional[int]:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None

    if token_id is None:
        return None

    try:
        token_id = int(token_id)
    except Exception:
        return None

    unk = getattr(tokenizer, "unk_token_id", None)

    if unk is not None and token_id == int(unk):
        return None

    return token_id


def resolve_visual_indices(
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    input_ids: Sequence[int],
) -> List[int]:
    """
    Qwen2.5-VL language-sequence visual token positions.
    """

    mm_type_ids = batch.get("mm_token_type_ids")

    if torch.is_tensor(mm_type_ids) and mm_type_ids.ndim == 2:
        direct = (
            torch.nonzero(mm_type_ids[0] == 1, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        if direct:
            return [int(x) for x in direct]

    token_type_ids = batch.get("token_type_ids")

    if torch.is_tensor(token_type_ids) and token_type_ids.ndim == 2:
        values = token_type_ids[0].detach().cpu()
        unique = set(int(x) for x in values.tolist())

        if 1 in unique:
            direct = (
                torch.nonzero(values == 1, as_tuple=False)
                .flatten()
                .tolist()
            )
            if direct:
                return [int(x) for x in direct]

    token_ids = set()

    objects = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        processor,
        getattr(processor, "tokenizer", None),
    ]

    for obj in objects:
        if obj is None:
            continue

        for name in ("image_token_id", "image_token_index"):
            value = getattr(obj, name, None)
            if isinstance(value, (int, np.integer)) and int(value) >= 0:
                token_ids.add(int(value))

    tokenizer = processor.tokenizer

    for token in (
        "<|image_pad|>",
        "<image>",
        "<image_token>",
        "<IMG_CONTEXT>",
    ):
        token_id = candidate_token_id(tokenizer, token)
        if token_id is not None:
            token_ids.add(token_id)

    indices = [
        index
        for index, token_id in enumerate(input_ids)
        if int(token_id) in token_ids
    ]

    if indices:
        return indices

    start_ids = {
        token_id
        for token in ("<|vision_start|>", "<image_start>", "<img>")
        if (token_id := candidate_token_id(tokenizer, token)) is not None
    }
    end_ids = {
        token_id
        for token in ("<|vision_end|>", "<image_end>", "</img>")
        if (token_id := candidate_token_id(tokenizer, token)) is not None
    }

    starts = [
        index
        for index, value in enumerate(input_ids)
        if int(value) in start_ids
    ]
    ends = [
        index
        for index, value in enumerate(input_ids)
        if int(value) in end_ids
    ]

    spans = [
        (start, end)
        for start in starts
        for end in ends
        if start < end
    ]

    if spans:
        start, end = min(spans, key=lambda pair: pair[1] - pair[0])
        fallback = list(range(start + 1, end))
        if fallback:
            return fallback

    raise RuntimeError(
        "Could not identify Qwen visual-token positions."
    )


def build_source_groups(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    input_ids: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
    source_length: int,
) -> Dict[str, List[int]]:
    if source_length != len(input_ids):
        raise RuntimeError(
            "This script is Qwen-7B-specific and expects raw input_ids length "
            f"to equal decoder attention source length. raw={len(input_ids)} "
            f"source={source_length}."
        )

    all_positions = set(range(source_length))

    object_set = set(map(int, subject_positions)) | set(
        map(int, reference_positions)
    )

    visual_set = set(
        map(
            int,
            resolve_visual_indices(
                model,
                processor,
                batch,
                input_ids,
            ),
        )
    )

    if object_set & visual_set:
        raise RuntimeError(
            f"Object/visual overlap: {sorted(object_set & visual_set)}"
        )

    relation_set = set(
        locate_relation_word_positions(
            tokenizer=processor.tokenizer,
            input_ids=input_ids,
            excluded=object_set | visual_set,
        )
    )

    self_set = {int(prompt_last)}

    # Priority: object > visual > relation > self > other.
    relation_set -= object_set
    relation_set -= visual_set

    self_set -= object_set
    self_set -= visual_set
    self_set -= relation_set

    used = object_set | visual_set | relation_set | self_set
    other_set = all_positions - used

    groups = {
        "object_text": sorted(object_set),
        "visual": sorted(visual_set),
        "relation_words": sorted(relation_set),
        "prompt_last_self": sorted(self_set),
        "other_text": sorted(other_set),
    }

    union = set()
    for name in SOURCE_GROUPS:
        current = set(groups[name])
        overlap = union & current
        if overlap:
            raise RuntimeError(
                f"Source groups overlap for {name}: {sorted(overlap)[:20]}"
            )
        union |= current

    if union != all_positions:
        missing = sorted(all_positions - union)
        extra = sorted(union - all_positions)
        raise RuntimeError(
            f"Source partition does not close. missing={missing[:20]} "
            f"extra={extra[:20]}"
        )

    return groups


# =============================================================================
# Model final norm + true relation logit lens
# =============================================================================

def get_attr_path(root: Any, path: str) -> Any:
    current = root

    for part in path.split("."):
        current = getattr(current, part, None)
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

    for path in dict.fromkeys(candidates):
        module = get_attr_path(model, path)
        if isinstance(module, torch.nn.Module):
            return module, path

    return None, "unresolved"


class RelationLogitLens:
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
            relation: [int(x) for x in token_map[relation]]
            for relation in RELATIONS
        }

        output_embeddings = model.get_output_embeddings()

        if (
            output_embeddings is None
            or not hasattr(output_embeddings, "weight")
        ):
            raise RuntimeError("No model output embedding / LM head weight.")

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
            for index, token_id in enumerate(union)
        }

        index = torch.as_tensor(
            union,
            device=weight.device,
            dtype=torch.long,
        )

        self.weight_rows = weight.index_select(0, index).detach()

        bias = getattr(output_embeddings, "bias", None)

        if torch.is_tensor(bias):
            self.bias_rows = bias.index_select(0, index).detach()
        else:
            self.bias_rows = None

        self.device = weight.device
        self.dtype = weight.dtype

    @torch.inference_mode()
    def scores(self, states: np.ndarray) -> np.ndarray:
        array = np.asarray(states, dtype=np.float32)

        leading = array.shape[:-1]
        hidden = int(array.shape[-1])

        flat = array.reshape(-1, hidden)

        tensor = torch.as_tensor(
            flat,
            device=self.device,
            dtype=self.dtype,
        )

        normalized = self.final_norm(tensor)

        if isinstance(normalized, (tuple, list)):
            normalized = normalized[0]

        token_logits = normalized @ self.weight_rows.T

        if self.bias_rows is not None:
            token_logits = token_logits + self.bias_rows

        relation_scores: List[torch.Tensor] = []

        for relation in RELATIONS:
            columns = [
                self.union_lookup[token_id]
                for token_id in self.token_map[relation]
            ]

            relation_scores.append(
                token_logits[:, columns].max(dim=-1).values
            )

        output = torch.stack(relation_scores, dim=-1)

        result = (
            output.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return result.reshape(*leading, len(RELATIONS))


def lens_pred(scores: np.ndarray) -> str:
    return RELATIONS[int(np.argmax(np.asarray(scores)))]


def lens_margin(
    scores: np.ndarray,
    gt: str,
) -> float:
    values = np.asarray(scores, dtype=np.float32)
    return float(
        values[RID[gt]]
        - values[RID[OPPOSITE[gt]]]
    )


# =============================================================================
# Attention trace math
# =============================================================================

def trace_target_index(
    trace: Any,
    target_position: int,
) -> int:
    lookup = {
        int(position): index
        for index, position in enumerate(trace.target_positions)
    }

    if int(target_position) not in lookup:
        raise RuntimeError(
            f"Target {target_position} missing from trace positions "
            f"{trace.target_positions}."
        )

    return int(lookup[int(target_position)])


def trace_block_state(
    trace: Any,
    target_position: int,
) -> np.ndarray:
    local = trace_target_index(trace, target_position)

    return (
        trace.block_output[local]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def trace_attention_state(
    trace: Any,
    target_position: int,
) -> np.ndarray:
    local = trace_target_index(trace, target_position)

    return (
        trace.attention_output[local]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def head_group_writes(
    *,
    trace: Any,
    target_position: int,
    source_positions: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        post-WO writes [Hq,Dmodel]
        attention mass [Hq]
    """
    H = int(trace.attention_weights.shape[0])
    D = int(trace.o_proj_weight.shape[0])

    positions = sorted(set(map(int, source_positions)))

    if not positions:
        return (
            np.zeros((H, D), dtype=np.float32),
            np.zeros(H, dtype=np.float32),
        )

    local = trace_target_index(trace, target_position)

    source = torch.as_tensor(
        positions,
        dtype=torch.long,
    )

    if int(source.max()) >= int(trace.value_states.shape[1]):
        raise RuntimeError(
            f"Source position {int(source.max())} >= "
            f"value length {trace.value_states.shape[1]}"
        )

    weights = (
        trace.attention_weights[:, local, :]
        .index_select(1, source)
        .float()
    )  # [H,S]

    values = (
        trace.value_states
        .index_select(1, source)
        .float()
    )  # [H,S,Dh]

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

    mass = weights.sum(dim=-1)

    return (
        post.detach().cpu().numpy().astype(np.float32),
        mass.detach().cpu().numpy().astype(np.float32),
    )


def decompose_layer(
    *,
    trace: Any,
    x: np.ndarray,
    prompt_last: int,
    groups: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    a = trace_attention_state(trace, prompt_last)
    y = trace_block_state(trace, prompt_last)
    r_attn = (np.asarray(x, np.float32) + a).astype(np.float32)
    mlp = (y - r_attn).astype(np.float32)

    group_heads: Dict[str, np.ndarray] = {}
    group_mass: Dict[str, np.ndarray] = {}

    for group in SOURCE_GROUPS:
        writes, mass = head_group_writes(
            trace=trace,
            target_position=prompt_last,
            source_positions=groups[group],
        )
        group_heads[group] = writes
        group_mass[group] = mass

    source_sum = np.zeros_like(a)

    for group in SOURCE_GROUPS:
        source_sum += group_heads[group].sum(axis=0)

    source_closure = float(
        np.linalg.norm(source_sum - a)
        / max(float(np.linalg.norm(a)), EPS)
    )

    H = next(iter(group_heads.values())).shape[0]
    head_total = np.zeros((H, a.shape[0]), dtype=np.float32)

    for group in SOURCE_GROUPS:
        head_total += group_heads[group]

    return {
        "x": np.asarray(x, dtype=np.float32),
        "a": a,
        "r_attn": r_attn,
        "y": y,
        "mlp": mlp,
        "group_heads": group_heads,
        "group_mass": group_mass,
        "head_total": head_total,
        "source_sum": source_sum,
        "source_closure": source_closure,
    }


# =============================================================================
# Full-generation hook
# =============================================================================

def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected attention output [B,S,D], got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise RuntimeError("Could not locate 3D attention output tensor.")


def replace_first_3d(
    output: Any,
    replacement: torch.Tensor,
) -> Any:
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return tuple(items)

    if isinstance(output, list):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return items

    raise RuntimeError("Could not replace 3D attention output tensor.")


class PromptLastDelta:
    """
    Add fixed clean residual-space delta after attention o_proj at prompt-last
    during prefill only.
    """

    def __init__(
        self,
        *,
        attention_module: torch.nn.Module,
        prompt_length: int,
        prompt_last: int,
        delta: np.ndarray,
    ) -> None:
        self.attention_module = attention_module
        self.prompt_length = int(prompt_length)
        self.prompt_last = int(prompt_last)
        self.delta = np.asarray(delta, dtype=np.float32)

        self.handle = None
        self.applications = 0

    def __enter__(self) -> "PromptLastDelta":
        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> Any:
            hidden = first_3d(output)

            # Full prefill only. Cached autoregressive decode q_len is normally 1.
            if int(hidden.shape[1]) != self.prompt_length:
                return None

            if not 0 <= self.prompt_last < int(hidden.shape[1]):
                raise RuntimeError(
                    f"prompt_last={self.prompt_last}, q_len={hidden.shape[1]}"
                )

            if int(hidden.shape[-1]) != int(self.delta.shape[0]):
                raise RuntimeError(
                    f"delta dim {self.delta.shape[0]} != hidden {hidden.shape[-1]}"
                )

            modified = hidden.clone()

            modified[0, self.prompt_last] += torch.as_tensor(
                self.delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            self.applications += 1

            return replace_first_3d(output, modified)

        self.handle = self.attention_module.register_forward_hook(hook)
        return self

    def validate(self) -> None:
        if self.applications != 1:
            raise RuntimeError(
                f"Expected exactly one prefill patch, got {self.applications}."
            )

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


@torch.inference_mode()
def greedy_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    input_length = int(batch["input_ids"].shape[1])

    output_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    text = processor.tokenizer.decode(
        output_ids[0, input_length:],
        skip_special_tokens=True,
    ).strip()

    del output_ids

    return normalize_relation(text), text


@torch.inference_mode()
def patched_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    attention_module: torch.nn.Module,
    prompt_length: int,
    prompt_last: int,
    delta: np.ndarray,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    with PromptLastDelta(
        attention_module=attention_module,
        prompt_length=prompt_length,
        prompt_last=prompt_last,
        delta=delta,
    ) as patch:
        pred, text = greedy_generation(
            model=model,
            processor=processor,
            batch=batch,
            max_new_tokens=max_new_tokens,
        )
        patch.validate()

    return pred, text


# =============================================================================
# Natural aggregation
# =============================================================================

class MetricAccumulator:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))

    def summarize(
        self,
        *,
        group_fields: Sequence[str],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

        for row in self.rows:
            key = tuple(row[field] for field in group_fields)
            grouped[key].append(row)

        output: List[Dict[str, Any]] = []

        for key, rows in grouped.items():
            base = {
                field: value
                for field, value in zip(group_fields, key)
            }

            for cohort in ("all", "overwrite", "generation_wrong"):
                if cohort == "all":
                    subset = rows
                elif cohort == "overwrite":
                    subset = [r for r in rows if bool(r["overwrite_sample"])]
                else:
                    subset = [r for r in rows if not bool(r["generation_correct"])]

                base[f"N_{cohort}"] = len(subset)
                base[f"mean_norm_{cohort}"] = safe_mean(
                    r["message_norm"] for r in subset
                )
                base[f"mean_mass_{cohort}"] = safe_mean(
                    r["attention_mass"] for r in subset
                )
                base[f"mean_isolated_add_gain_{cohort}"] = safe_mean(
                    r["isolated_add_gain"] for r in subset
                )
                base[f"mean_natural_necessity_gain_{cohort}"] = safe_mean(
                    r["natural_necessity_gain"] for r in subset
                )
                base[f"mean_removal_improvement_{cohort}"] = safe_mean(
                    r["removal_improvement"] for r in subset
                )
                base[f"positive_removal_rate_{cohort}"] = safe_mean(
                    float(r["removal_improvement"] > 0)
                    for r in subset
                )

            output.append(base)

        return output


# =============================================================================
# Candidate labels
# =============================================================================

def head_label(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def condition_key(
    *,
    kind: str,
    layer: int,
    group: Optional[str] = None,
    head: Optional[int] = None,
) -> str:
    pieces = [kind, f"L{int(layer)}"]

    if head is not None:
        pieces.append(f"H{int(head):02d}")

    if group is not None:
        pieces.append(str(group))

    return "__".join(pieces)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.model != "qwen-7b":
        raise ValueError(
            "v1 is intentionally restricted to --model qwen-7b."
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = sorted(set(parse_int_list(args.layers)))

    if layers != [26, 27]:
        print(
            f"[warning] default interpretation assumes L26/L27; got {layers}.",
            flush=True,
        )

    if min(layers) <= 0:
        raise ValueError("Need layer >= 1 to recover block input from L-1.")

    output_dir = Path(args.output_dir)

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is non-empty: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    probe = importlib.import_module(args.probe_module)
    attention_helper = importlib.import_module(args.attention_helper_module)
    base = probe.base

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )

    selected_records = select_records(
        records,
        n=args.num_samples,
        seed=args.seed,
    )

    causal_records = stratified_limit_records(
        selected_records,
        n=args.causal_max_samples,
        seed=args.seed + 777,
    )

    (
        output_dir / "selected_sids.json"
    ).write_text(
        json.dumps(
            {
                "seed": args.seed,
                "N_natural": len(selected_records),
                "natural_sids": [int(r.sid) for r in selected_records],
                "N_causal": len(causal_records),
                "causal_sids": [int(r.sid) for r in causal_records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    spec = base.SPECS[args.model]
    model_class = getattr(transformers, spec.model_class)

    load_kwargs = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }

    model = None
    processor = None

    try:
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)

        model = model_class.from_pretrained(
            spec.repo_id,
            **load_kwargs,
        )
        model.eval()
        clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        device = torch.device(args.device)

        decoder_layers, decoder_path = probe.resolve_decoder_layers(model)

        if len(decoder_layers) != 28:
            raise RuntimeError(
                f"Expected Qwen-7B 28 decoder layers; found {len(decoder_layers)} "
                f"at {decoder_path}."
            )

        for layer in layers:
            if not 1 <= layer < len(decoder_layers):
                raise ValueError(
                    f"L{layer} outside L1-L{len(decoder_layers)-1}."
                )

        final_norm, final_norm_path = resolve_final_norm(
            model,
            decoder_path,
        )

        if final_norm is None:
            raise RuntimeError("Could not resolve final language norm.")

        relation_token_map = relation_token_variants(processor.tokenizer)

        lens = RelationLogitLens(
            model=model,
            final_norm=final_norm,
            token_map=relation_token_map,
        )

        attention_modules = {
            layer: attention_helper.resolve_self_attention(
                decoder_layers[layer]
            )
            for layer in layers
        }

        trace_layers = sorted(
            set([layer - 1 for layer in layers] + layers)
        )

        print("\n" + "=" * 170)
        print("QWEN-7B L26-vs-L27 ATTENTION OVERWRITE DECOMPOSITION")
        print("=" * 170)
        print("layers             :", layers)
        print("trace layers       :", trace_layers)
        print("N natural          :", len(selected_records))
        print("N causal           :", len(causal_records))
        print("source groups      :", ", ".join(SOURCE_GROUPS))
        print("rank cohort        :", args.rank_cohort)
        print("top damaging heads :", args.top_damaging_heads)
        print("top head-sources   :", args.top_damaging_head_sources)
        print("final norm         :", final_norm_path)
        print("=" * 170, flush=True)

        source_metrics = MetricAccumulator()
        head_source_metrics = MetricAccumulator()

        natural_layer_rows: List[Dict[str, Any]] = []

        # Save baseline generation per SID so causal W->C/C->W uses same baseline.
        baseline_by_sid: Dict[int, Dict[str, Any]] = {}

        # ---------------------------------------------------------------------
        # Phase A: natural decomposition
        # ---------------------------------------------------------------------

        for sample_index, record in enumerate(
            tqdm(
                selected_records,
                desc="natural-L26-L27",
            ),
            start=1,
        ):
            image = None
            batch = None

            try:
                sid = int(record.sid)
                gt = normalize_relation(record.relation)

                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Unsupported relation {record.relation!r}"
                    )

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(record.image_path).convert("RGB")

                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(x)
                    for x in batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                ]

                prompt_last = len(input_ids) - 1

                subject_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.subject),
                )
                reference_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.reference),
                )

                baseline, traces = attention_helper.run_and_trace(
                    model=model,
                    batch=batch,
                    token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layer_indices=trace_layers,
                    target_positions=[prompt_last],
                )

                baseline_first_pred = str(baseline["prediction"])

                generation_pred, generation_text = greedy_generation(
                    model=model,
                    processor=processor,
                    batch=batch,
                    max_new_tokens=args.max_new_tokens,
                )

                generation_correct = generation_pred == gt

                baseline_by_sid[sid] = {
                    "sid": sid,
                    "gt": gt,
                    "first_step_pred": baseline_first_pred,
                    "generation_pred": generation_pred,
                    "generation_text": generation_text,
                    "generation_correct": generation_correct,
                }

                layer_decomp: Dict[int, Dict[str, Any]] = {}

                for layer in layers:
                    trace = traces[layer]

                    source_length = int(trace.value_states.shape[1])

                    groups = build_source_groups(
                        model=model,
                        processor=processor,
                        batch=dict(batch),
                        input_ids=input_ids,
                        subject_positions=subject_positions,
                        reference_positions=reference_positions,
                        prompt_last=prompt_last,
                        source_length=source_length,
                    )

                    x = trace_block_state(
                        traces[layer - 1],
                        prompt_last,
                    )

                    decomp = decompose_layer(
                        trace=trace,
                        x=x,
                        prompt_last=prompt_last,
                        groups=groups,
                    )

                    layer_decomp[layer] = {
                        "decomp": decomp,
                        "groups": groups,
                    }

                # True lens per layer.
                for layer in layers:
                    decomp = layer_decomp[layer]["decomp"]

                    x_scores = lens.scores(decomp["x"])
                    rattn_scores = lens.scores(decomp["r_attn"])
                    y_scores = lens.scores(decomp["y"])

                    x_pred = lens_pred(x_scores)
                    rattn_pred = lens_pred(rattn_scores)
                    y_pred = lens_pred(y_scores)

                    x_margin = lens_margin(x_scores, gt)
                    rattn_margin = lens_margin(rattn_scores, gt)
                    y_margin = lens_margin(y_scores, gt)

                    overwrite_sample = (
                        x_pred == gt
                        and rattn_pred != gt
                    )

                    natural_layer_rows.append({
                        "sid": sid,
                        "layer": layer,
                        "gt": gt,
                        "generation_pred": generation_pred,
                        "generation_correct": generation_correct,
                        "x_pred": x_pred,
                        "x_correct": x_pred == gt,
                        "x_margin": x_margin,
                        "rattn_pred": rattn_pred,
                        "rattn_correct": rattn_pred == gt,
                        "rattn_margin": rattn_margin,
                        "attention_margin_gain": (
                            rattn_margin - x_margin
                        ),
                        "y_pred": y_pred,
                        "y_correct": y_pred == gt,
                        "y_margin": y_margin,
                        "mlp_margin_gain": (
                            y_margin - rattn_margin
                        ),
                        "overwrite_sample": overwrite_sample,
                        "source_closure_relative_error": decomp[
                            "source_closure"
                        ],
                        "x_norm": float(
                            np.linalg.norm(decomp["x"])
                        ),
                        "attention_norm": float(
                            np.linalg.norm(decomp["a"])
                        ),
                        "mlp_norm": float(
                            np.linalg.norm(decomp["mlp"])
                        ),
                    })

                    # Batch source-group lens evaluations.
                    group_messages = np.stack(
                        [
                            decomp["group_heads"][group].sum(axis=0)
                            for group in SOURCE_GROUPS
                        ],
                        axis=0,
                    ).astype(np.float32)

                    x_plus_group = (
                        decomp["x"][None, :]
                        + group_messages
                    )

                    rattn_minus_group = (
                        decomp["r_attn"][None, :]
                        - group_messages
                    )

                    add_scores = lens.scores(x_plus_group)
                    minus_scores = lens.scores(rattn_minus_group)

                    # Batch all head-total removals.
                    head_total = decomp["head_total"]

                    x_plus_head = (
                        decomp["x"][None, :]
                        + head_total
                    )

                    rattn_minus_head = (
                        decomp["r_attn"][None, :]
                        - head_total
                    )

                    head_add_scores = lens.scores(x_plus_head)
                    head_minus_scores = lens.scores(rattn_minus_head)

                    # Batch head x source evaluations one source at a time.
                    for group_index, group in enumerate(SOURCE_GROUPS):
                        message = group_messages[group_index]

                        add_margin = lens_margin(
                            add_scores[group_index],
                            gt,
                        )
                        minus_margin = lens_margin(
                            minus_scores[group_index],
                            gt,
                        )

                        natural_necessity = (
                            rattn_margin - minus_margin
                        )
                        removal_improvement = (
                            minus_margin - rattn_margin
                        )

                        source_metrics.add({
                            "sid": sid,
                            "layer": layer,
                            "source_group": group,
                            "gt": gt,
                            "generation_correct": generation_correct,
                            "overwrite_sample": overwrite_sample,
                            "message_norm": float(
                                np.linalg.norm(message)
                            ),
                            "attention_mass": float(
                                decomp["group_mass"][group].sum()
                            ),
                            "isolated_add_gain": (
                                add_margin - x_margin
                            ),
                            "natural_necessity_gain": (
                                natural_necessity
                            ),
                            "removal_improvement": (
                                removal_improvement
                            ),
                        })

                    H = int(head_total.shape[0])

                    for head in range(H):
                        message = head_total[head]

                        add_margin = lens_margin(
                            head_add_scores[head],
                            gt,
                        )
                        minus_margin = lens_margin(
                            head_minus_scores[head],
                            gt,
                        )

                        head_source_metrics.add({
                            "sid": sid,
                            "layer": layer,
                            "head": head,
                            "head_name": head_label(layer, head),
                            "source_group": "TOTAL",
                            "gt": gt,
                            "generation_correct": generation_correct,
                            "overwrite_sample": overwrite_sample,
                            "message_norm": float(
                                np.linalg.norm(message)
                            ),
                            "attention_mass": float(
                                sum(
                                    decomp["group_mass"][g][head]
                                    for g in SOURCE_GROUPS
                                )
                            ),
                            "isolated_add_gain": (
                                add_margin - x_margin
                            ),
                            "natural_necessity_gain": (
                                rattn_margin - minus_margin
                            ),
                            "removal_improvement": (
                                minus_margin - rattn_margin
                            ),
                        })

                    for group in SOURCE_GROUPS:
                        messages = decomp["group_heads"][group]

                        add_scores_hg = lens.scores(
                            decomp["x"][None, :]
                            + messages
                        )
                        minus_scores_hg = lens.scores(
                            decomp["r_attn"][None, :]
                            - messages
                        )

                        for head in range(H):
                            add_margin = lens_margin(
                                add_scores_hg[head],
                                gt,
                            )
                            minus_margin = lens_margin(
                                minus_scores_hg[head],
                                gt,
                            )

                            head_source_metrics.add({
                                "sid": sid,
                                "layer": layer,
                                "head": head,
                                "head_name": head_label(layer, head),
                                "source_group": group,
                                "gt": gt,
                                "generation_correct": generation_correct,
                                "overwrite_sample": overwrite_sample,
                                "message_norm": float(
                                    np.linalg.norm(messages[head])
                                ),
                                "attention_mass": float(
                                    decomp["group_mass"][group][head]
                                ),
                                "isolated_add_gain": (
                                    add_margin - x_margin
                                ),
                                "natural_necessity_gain": (
                                    rattn_margin - minus_margin
                                ),
                                "removal_improvement": (
                                    minus_margin - rattn_margin
                                ),
                            })

                del traces

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "natural",
                        "sid": int(getattr(record, "sid", -1)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                del batch
                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(
            output_dir / "natural_sample_layer.csv",
            natural_layer_rows,
        )

        source_summary = source_metrics.summarize(
            group_fields=("layer", "source_group"),
        )

        write_csv(
            output_dir / "source_group_summary.csv",
            source_summary,
        )

        head_source_summary = head_source_metrics.summarize(
            group_fields=(
                "layer",
                "head",
                "head_name",
                "source_group",
            ),
        )

        write_csv(
            output_dir / "head_source_summary.csv",
            head_source_summary,
        )

        # ---------------------------------------------------------------------
        # Rank L27 damaging total heads + head-source pairs.
        # ---------------------------------------------------------------------

        target_layer = max(layers)
        cohort_suffix = args.rank_cohort

        rank_metric = f"mean_removal_improvement_{cohort_suffix}"
        positive_metric = f"positive_removal_rate_{cohort_suffix}"
        n_metric = f"N_{cohort_suffix}"

        l27_total = [
            row
            for row in head_source_summary
            if int(row["layer"]) == target_layer
            and str(row["source_group"]) == "TOTAL"
            and int(row[n_metric]) > 0
        ]

        l27_total.sort(
            key=lambda row: (
                -float(row[rank_metric])
                if math.isfinite(float(row[rank_metric]))
                else float("inf"),
                -float(row[positive_metric])
                if math.isfinite(float(row[positive_metric]))
                else float("inf"),
                int(row["head"]),
            )
        )

        top_heads = l27_total[: max(args.top_damaging_heads, 0)]

        write_csv(
            output_dir / "top_damaging_l27_heads.csv",
            top_heads,
        )

        l27_head_sources = [
            row
            for row in head_source_summary
            if int(row["layer"]) == target_layer
            and str(row["source_group"]) != "TOTAL"
            and int(row[n_metric]) > 0
        ]

        l27_head_sources.sort(
            key=lambda row: (
                -float(row[rank_metric])
                if math.isfinite(float(row[rank_metric]))
                else float("inf"),
                -float(row[positive_metric])
                if math.isfinite(float(row[positive_metric]))
                else float("inf"),
                int(row["head"]),
                str(row["source_group"]),
            )
        )

        top_head_sources = l27_head_sources[
            : max(args.top_damaging_head_sources, 0)
        ]

        write_csv(
            output_dir / "top_damaging_l27_head_sources.csv",
            top_head_sources,
        )

        # ---------------------------------------------------------------------
        # Console natural summary.
        # ---------------------------------------------------------------------

        print("\n" + "=" * 190)
        print("NATURAL SOURCE-GROUP DECOMPOSITION")
        print("=" * 190)

        source_by_layer = defaultdict(list)
        for row in source_summary:
            source_by_layer[int(row["layer"])].append(row)

        layer_rows_map = defaultdict(list)
        for row in natural_layer_rows:
            layer_rows_map[int(row["layer"])].append(row)

        for layer in layers:
            rows = layer_rows_map[layer]

            print(
                f"\nL{layer}: "
                f"xACC={100*safe_mean(float(r['x_correct']) for r in rows):.2f}% "
                f"-> rAttnACC={100*safe_mean(float(r['rattn_correct']) for r in rows):.2f}% "
                f"-> yACC={100*safe_mean(float(r['y_correct']) for r in rows):.2f}% "
                f"| attn margin gain={safe_mean(r['attention_margin_gain'] for r in rows):+.4f} "
                f"| overwrite N={sum(bool(r['overwrite_sample']) for r in rows)}"
            )

            print(
                f"  {'source':<18s} {'norm':>9s} {'mass':>9s} "
                f"{'addGain':>10s} {'removeImp':>11s} {'remove+':>8s} "
                f"{'ovwImp':>10s} {'ovw+':>8s}"
            )

            ordered_source = sorted(
                source_by_layer[layer],
                key=lambda row: -float(
                    row["mean_removal_improvement_overwrite"]
                )
                if math.isfinite(
                    float(row["mean_removal_improvement_overwrite"])
                )
                else float("inf"),
            )

            for row in ordered_source:
                print(
                    f"  {str(row['source_group']):<18s} "
                    f"{float(row['mean_norm_all']):9.4f} "
                    f"{float(row['mean_mass_all']):9.4f} "
                    f"{float(row['mean_isolated_add_gain_all']):+10.4f} "
                    f"{float(row['mean_removal_improvement_all']):+11.4f} "
                    f"{100*float(row['positive_removal_rate_all']):7.2f}% "
                    f"{float(row['mean_removal_improvement_overwrite']):+10.4f} "
                    f"{100*float(row['positive_removal_rate_overwrite']):7.2f}%"
                )

        print("\nTOP DAMAGING L27 TOTAL HEADS")
        print(
            f"  {'rank':>4s} {'head':<9s} "
            f"{rank_metric:>26s} {'remove+':>9s} "
            f"{'allImp':>10s} {'norm':>9s} {'mass':>9s}"
        )

        for rank, row in enumerate(top_heads, start=1):
            print(
                f"  {rank:4d} "
                f"{str(row['head_name']):<9s} "
                f"{float(row[rank_metric]):+26.5f} "
                f"{100*float(row[positive_metric]):8.2f}% "
                f"{float(row['mean_removal_improvement_all']):+10.5f} "
                f"{float(row['mean_norm_all']):9.4f} "
                f"{float(row['mean_mass_all']):9.4f}"
            )

        print("\nTOP DAMAGING L27 HEAD x SOURCE")
        print(
            f"  {'rank':>4s} {'head':<9s} {'source':<18s} "
            f"{rank_metric:>26s} {'remove+':>9s} {'allImp':>10s}"
        )

        for rank, row in enumerate(top_head_sources, start=1):
            print(
                f"  {rank:4d} "
                f"{str(row['head_name']):<9s} "
                f"{str(row['source_group']):<18s} "
                f"{float(row[rank_metric]):+26.5f} "
                f"{100*float(row[positive_metric]):8.2f}% "
                f"{float(row['mean_removal_improvement_all']):+10.5f}"
            )

        print("=" * 190, flush=True)

        # ---------------------------------------------------------------------
        # Build causal condition set.
        # ---------------------------------------------------------------------

        conditions: List[Dict[str, Any]] = []

        # All source groups for both L26 and L27.
        for layer in layers:
            for group in SOURCE_GROUPS:
                conditions.append({
                    "condition": condition_key(
                        kind="remove_source",
                        layer=layer,
                        group=group,
                    ),
                    "kind": "remove_source",
                    "layer": int(layer),
                    "source_group": group,
                    "head": None,
                })

        # L27 all-attention removal as a strong sanity condition.
        conditions.append({
            "condition": condition_key(
                kind="remove_all_attention",
                layer=target_layer,
            ),
            "kind": "remove_all_attention",
            "layer": target_layer,
            "source_group": None,
            "head": None,
        })

        for row in top_heads:
            head = int(row["head"])
            conditions.append({
                "condition": condition_key(
                    kind="remove_head_total",
                    layer=target_layer,
                    head=head,
                ),
                "kind": "remove_head_total",
                "layer": target_layer,
                "source_group": "TOTAL",
                "head": head,
            })

        for row in top_head_sources:
            head = int(row["head"])
            group = str(row["source_group"])

            conditions.append({
                "condition": condition_key(
                    kind="remove_head_source",
                    layer=target_layer,
                    head=head,
                    group=group,
                ),
                "kind": "remove_head_source",
                "layer": target_layer,
                "source_group": group,
                "head": head,
            })

        # Deduplicate.
        unique_conditions: List[Dict[str, Any]] = []
        seen_conditions = set()

        for condition in conditions:
            key = condition["condition"]
            if key not in seen_conditions:
                unique_conditions.append(condition)
                seen_conditions.add(key)

        conditions = unique_conditions

        (
            output_dir / "causal_conditions.json"
        ).write_text(
            json.dumps(
                conditions,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        causal_sids = {int(r.sid) for r in causal_records}

        baseline_causal = {
            sid: baseline_by_sid[sid]
            for sid in causal_sids
            if sid in baseline_by_sid
        }

        if len(baseline_causal) != len(causal_records):
            raise RuntimeError(
                f"Causal baseline coverage {len(baseline_causal)}/"
                f"{len(causal_records)}."
            )

        # ---------------------------------------------------------------------
        # Phase B: actual generation causal removal
        # ---------------------------------------------------------------------

        causal_jsonl = output_dir / "causal_generation_results.jsonl"
        causal_rows: List[Dict[str, Any]] = []

        print(
            f"\nCausal generation: N={len(causal_records)} "
            f"x conditions={len(conditions)} "
            f"= {len(causal_records) * len(conditions)} patched generations",
            flush=True,
        )

        for sample_index, record in enumerate(
            tqdm(
                causal_records,
                desc="causal-generation",
            ),
            start=1,
        ):
            image = None
            batch = None

            try:
                sid = int(record.sid)
                gt = normalize_relation(record.relation)
                baseline_info = baseline_causal[sid]

                question = args.prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )

                image = Image.open(record.image_path).convert("RGB")

                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(x)
                    for x in batch["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                ]

                prompt_length = len(input_ids)
                prompt_last = prompt_length - 1

                subject_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.subject),
                )
                reference_positions = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.reference),
                )

                _, traces = attention_helper.run_and_trace(
                    model=model,
                    batch=batch,
                    token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layer_indices=trace_layers,
                    target_positions=[prompt_last],
                )

                layer_decomp: Dict[int, Dict[str, Any]] = {}

                for layer in layers:
                    trace = traces[layer]

                    groups = build_source_groups(
                        model=model,
                        processor=processor,
                        batch=dict(batch),
                        input_ids=input_ids,
                        subject_positions=subject_positions,
                        reference_positions=reference_positions,
                        prompt_last=prompt_last,
                        source_length=int(trace.value_states.shape[1]),
                    )

                    x = trace_block_state(
                        traces[layer - 1],
                        prompt_last,
                    )

                    layer_decomp[layer] = decompose_layer(
                        trace=trace,
                        x=x,
                        prompt_last=prompt_last,
                        groups=groups,
                    )

                for condition in conditions:
                    layer = int(condition["layer"])
                    decomp = layer_decomp[layer]
                    kind = str(condition["kind"])

                    if kind == "remove_source":
                        group = str(condition["source_group"])
                        contribution = decomp["group_heads"][group].sum(axis=0)

                    elif kind == "remove_all_attention":
                        contribution = np.asarray(
                            decomp["a"],
                            dtype=np.float32,
                        )

                    elif kind == "remove_head_total":
                        head = int(condition["head"])
                        contribution = decomp["head_total"][head]

                    elif kind == "remove_head_source":
                        head = int(condition["head"])
                        group = str(condition["source_group"])
                        contribution = decomp["group_heads"][group][head]

                    else:
                        raise RuntimeError(f"Unknown condition kind={kind}")

                    delta = -np.asarray(contribution, dtype=np.float32)

                    patched_pred, patched_text = patched_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        attention_module=attention_modules[layer],
                        prompt_length=prompt_length,
                        prompt_last=prompt_last,
                        delta=delta,
                        max_new_tokens=args.max_new_tokens,
                    )

                    baseline_pred = normalize_relation(
                        baseline_info["generation_pred"]
                    )
                    baseline_correct = bool(
                        baseline_info["generation_correct"]
                    )
                    patched_correct = patched_pred == gt

                    row = {
                        "sid": sid,
                        "gt": gt,
                        "condition": condition["condition"],
                        "kind": kind,
                        "layer": layer,
                        "head": condition.get("head"),
                        "source_group": condition.get("source_group"),
                        "baseline_pred": baseline_pred,
                        "baseline_correct": baseline_correct,
                        "patched_pred": patched_pred,
                        "patched_text": patched_text,
                        "patched_correct": patched_correct,
                        "wrong_to_correct": (
                            (not baseline_correct)
                            and patched_correct
                        ),
                        "correct_to_wrong": (
                            baseline_correct
                            and (not patched_correct)
                        ),
                        "generation_changed": (
                            patched_pred != baseline_pred
                        ),
                        "contribution_norm": float(
                            np.linalg.norm(contribution)
                        ),
                        "delta_norm": float(
                            np.linalg.norm(delta)
                        ),
                    }

                    causal_rows.append(row)
                    append_jsonl(causal_jsonl, row)

                del traces

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "causal",
                        "sid": int(getattr(record, "sid", -1)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                del batch
                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Causal summary
        # ---------------------------------------------------------------------

        grouped_causal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in causal_rows:
            grouped_causal[str(row["condition"])].append(row)

        baseline_acc_causal = safe_mean(
            float(info["generation_correct"])
            for info in baseline_causal.values()
        )

        causal_summary: List[Dict[str, Any]] = []

        for condition in conditions:
            key = str(condition["condition"])
            rows = grouped_causal.get(key, [])

            if not rows:
                continue

            patched_acc = safe_mean(
                float(row["patched_correct"])
                for row in rows
            )

            w2c = sum(bool(row["wrong_to_correct"]) for row in rows)
            c2w = sum(bool(row["correct_to_wrong"]) for row in rows)
            changed = sum(bool(row["generation_changed"]) for row in rows)

            causal_summary.append({
                "condition": key,
                "kind": condition["kind"],
                "layer": condition["layer"],
                "head": condition.get("head"),
                "source_group": condition.get("source_group"),
                "N": len(rows),
                "baseline_acc": baseline_acc_causal,
                "patched_acc": patched_acc,
                "delta_acc": (
                    patched_acc - baseline_acc_causal
                ),
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net_repairs": w2c - c2w,
                "generation_changed": changed,
                "generation_changed_rate": changed / max(len(rows), 1),
                "mean_contribution_norm": safe_mean(
                    row["contribution_norm"]
                    for row in rows
                ),
            })

        causal_summary.sort(
            key=lambda row: (
                -float(row["delta_acc"]),
                -int(row["net_repairs"]),
                str(row["condition"]),
            )
        )

        write_csv(
            output_dir / "causal_generation_summary.csv",
            causal_summary,
        )

        print("\n" + "=" * 190)
        print("FULL-GENERATION CAUSAL REMOVAL SUMMARY")
        print("=" * 190)
        print(
            f"Baseline generation ACC on causal subset: "
            f"{100*baseline_acc_causal:.2f}% | N={len(causal_records)}"
        )
        print(
            f"  {'rank':>4s} {'condition':<52s} {'ACC':>8s} {'delta':>9s} "
            f"{'W->C':>5s} {'C->W':>5s} {'net':>5s} {'chg':>8s}"
        )

        for rank, row in enumerate(causal_summary, start=1):
            print(
                f"  {rank:4d} "
                f"{str(row['condition']):<52s} "
                f"{100*float(row['patched_acc']):7.2f}% "
                f"{100*float(row['delta_acc']):+8.2f} "
                f"{int(row['wrong_to_correct']):5d} "
                f"{int(row['correct_to_wrong']):5d} "
                f"{int(row['net_repairs']):+5d} "
                f"{100*float(row['generation_changed_rate']):7.2f}%"
            )

        print("=" * 190)

        # ---------------------------------------------------------------------
        # Config/report
        # ---------------------------------------------------------------------

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "data_root": args.data_root,
            "layers": layers,
            "trace_layers": trace_layers,
            "N_natural": len(selected_records),
            "N_causal": len(causal_records),
            "seed": args.seed,
            "rank_cohort": args.rank_cohort,
            "top_damaging_heads": args.top_damaging_heads,
            "top_damaging_head_sources": args.top_damaging_head_sources,
            "source_groups": list(SOURCE_GROUPS),
            "final_norm_path": final_norm_path,
            "true_logit_lens": (
                "LM_head(final_norm(prompt_last_state)) over relation token variants"
            ),
            "source_message": (
                "W_O^h sum_{s in source_group} A_h[last,s] V_h[s]"
            ),
            "natural_removal_improvement": (
                "margin(r_attn - c) - margin(r_attn); positive means c is damaging"
            ),
            "causal_intervention": (
                "attention_out_L[last] -= clean sample-specific source/head contribution "
                "during prefill; all downstream layers recompute online"
            ),
            "selection_bias_warning": (
                "Top damaging heads/head-source pairs are selected using GT margin on the "
                "same natural sample set; treat causal gain as exploratory until frozen "
                "and evaluated on fresh held-out samples."
            ),
            "audit": audit,
        }

        (
            output_dir / "config.json"
        ).write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report_lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"N natural: {len(selected_records)}",
            f"N causal: {len(causal_records)}",
            f"layers: {layers}",
            f"rank cohort: {args.rank_cohort}",
            f"final norm: {final_norm_path}",
            "",
            "NATURAL LAYER TRAJECTORY",
        ]

        for layer in layers:
            rows = layer_rows_map[layer]
            report_lines.append(
                f"L{layer}: "
                f"xACC={100*safe_mean(float(r['x_correct']) for r in rows):.2f}% "
                f"rAttnACC={100*safe_mean(float(r['rattn_correct']) for r in rows):.2f}% "
                f"yACC={100*safe_mean(float(r['y_correct']) for r in rows):.2f}% "
                f"attnMarginGain={safe_mean(r['attention_margin_gain'] for r in rows):+.4f} "
                f"overwriteN={sum(bool(r['overwrite_sample']) for r in rows)}"
            )

        report_lines += [
            "",
            "SOURCE GROUPS",
        ]

        for layer in layers:
            report_lines.append(f"L{layer}")
            ordered_source = sorted(
                source_by_layer[layer],
                key=lambda row: -float(
                    row["mean_removal_improvement_overwrite"]
                )
                if math.isfinite(
                    float(row["mean_removal_improvement_overwrite"])
                )
                else float("inf"),
            )

            for row in ordered_source:
                report_lines.append(
                    f"  {row['source_group']}: "
                    f"removeImpAll={float(row['mean_removal_improvement_all']):+.4f} "
                    f"removeImpOverwrite={float(row['mean_removal_improvement_overwrite']):+.4f} "
                    f"removePosOverwrite={100*float(row['positive_removal_rate_overwrite']):.2f}% "
                    f"norm={float(row['mean_norm_all']):.4f} "
                    f"mass={float(row['mean_mass_all']):.4f}"
                )

        report_lines += [
            "",
            "TOP L27 DAMAGING TOTAL HEADS",
        ]

        for rank, row in enumerate(top_heads, start=1):
            report_lines.append(
                f"{rank:02d} {row['head_name']} "
                f"{rank_metric}={float(row[rank_metric]):+.5f} "
                f"removePos={100*float(row[positive_metric]):.2f}%"
            )

        report_lines += [
            "",
            "TOP L27 DAMAGING HEAD x SOURCE",
        ]

        for rank, row in enumerate(top_head_sources, start=1):
            report_lines.append(
                f"{rank:02d} {row['head_name']} {row['source_group']} "
                f"{rank_metric}={float(row[rank_metric]):+.5f} "
                f"removePos={100*float(row[positive_metric]):.2f}%"
            )

        report_lines += [
            "",
            "CAUSAL GENERATION REMOVAL",
            f"baseline causal ACC={100*baseline_acc_causal:.2f}%",
        ]

        for rank, row in enumerate(causal_summary, start=1):
            report_lines.append(
                f"{rank:02d} {row['condition']} "
                f"ACC={100*float(row['patched_acc']):.2f}% "
                f"delta={100*float(row['delta_acc']):+.2f}pp "
                f"W->C={int(row['wrong_to_correct'])} "
                f"C->W={int(row['correct_to_wrong'])} "
                f"net={int(row['net_repairs']):+d}"
            )

        report_lines += [
            "",
            "INTERPRETATION",
            (
                "If one L27 source group has positive natural removal improvement and "
                "its full-generation removal also raises ACC, that source class is a "
                "causal contributor to late overwrite."
            ),
            (
                "If the group-level effect is broad but a small number of L27 heads "
                "account for it, the overwrite is head-localized."
            ),
            (
                "If L26 shows the opposite sign for the same source/head family, compare "
                "their V content / W_O directions next rather than attention mass alone."
            ),
            (
                "If no source/head removal improves generation despite the logit-lens "
                "signal, the L27 negative lens movement may be compensated downstream or "
                "may reflect a readout artifact rather than a generation-causal failure."
            ),
        ]

        (
            output_dir / "report.txt"
        ).write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for filename in (
            "selected_sids.json",
            "natural_sample_layer.csv",
            "source_group_summary.csv",
            "head_source_summary.csv",
            "top_damaging_l27_heads.csv",
            "top_damaging_l27_head_sources.csv",
            "causal_conditions.json",
            "causal_generation_results.jsonl",
            "causal_generation_summary.csv",
            "config.json",
            "report.txt",
        ):
            print(" ", output_dir / filename)

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
