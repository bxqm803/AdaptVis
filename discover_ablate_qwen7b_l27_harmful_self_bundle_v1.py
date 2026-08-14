#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Discover harmful L27 prompt-last SELF messages on one split, then perform
held-out bundle ablation with full generation on Qwen2.5-VL-7B.

Core message
============
For query head h at L27, prompt-last self contribution is

    c_h^self =
        W_O^h A_h[last,last] V_h[last]

This is ONLY:
    prompt-last source -> prompt-last destination
for one query head, after the head-specific W_O block.

Discovery phase (no intervention)
=================================
Use a stratified DISCOVERY split only.

Natural L27 states:
    x       = L27 block input = L26 block output at prompt-last
    a       = L27 attention output
    r_attn  = x + a
    y       = L27 block output after MLP

True logit lens:
    z -> final_norm(z) -> LM head
restricted to left/right/above/below.

For every head h:

    removal_improvement_h
      = margin(r_attn - c_h^self) - margin(r_attn)

Positive means removing that self-message would locally improve
GT-vs-opposite relation margin.

Main discovery cohort:
    overwrite:
        lens(x) == GT
        and
        lens(r_attn) != GT

Heads are ranked ONLY on the discovery split.

Held-out causal phase
=====================
Freeze the discovery ranking. On a separate EVAL split, run full greedy
model.generate() while ablating CLEAN sample-specific self messages:

    attention_out_L27[last]
      <- attention_out_L27[last]
         - scale * sum_{h in bundle} c_h^self

Default scale=1.0 is exact removal of those natural self contributions.

Bundles tested
==============
1) Prefix bundles from discovery ranking:
       Top1
       Top2
       ...
       TopK

2) The complete shared-KV query-head group containing discovery Top1.
   Qwen2.5-VL-7B normally has 28 query heads and 4 KV heads, so each
   KV head serves 7 query heads.

3) All other complete KV groups as matched structural controls.

4) One or more random same-size bundles for each prefix K, sampled only
   after discovery ranking is frozen.

This directly tests whether:
    weak single-head effects accumulate into a causal harmful self-message
    family.

Important interpretation
========================
Strong evidence would look like:

    Top1          small/no gain
    Top2          larger
    Top3/Top4/... monotonic or saturating gain
    random-K      ~0
    winning KV group > other KV groups

Then it is reasonable to say a small L27 self-message family causally
contributes to late overwrite.

If bundle ablation hurts or does not outperform matched random controls,
the natural logit-lens harmfulness is not a robust generation-causal circuit.

Selection discipline
====================
Discovery and causal evaluation are disjoint by default.
GT is used to rank heads only on DISCOVERY.
No EVAL result is used to select heads.

Dataset
=======
Default --num-samples 0 uses all COCO-two records (typically 440),
then stratified split:
    50% discovery
    50% eval
with --discovery-ratio 0.5.

You can cap held-out generation:
    --eval-max-samples 200

Example
=======
CUDA_VISIBLE_DEVICES=0 python -u \
discover_ablate_qwen7b_l27_harmful_self_bundle_v1.py \
  --model qwen-7b \
  --num-samples 0 \
  --discovery-ratio 0.50 \
  --eval-max-samples 200 \
  --top-k 7 \
  --random-bundles-per-k 1 \
  --ablation-scale 1.0 \
  --device cuda:0 \
  --output-dir output/qwen7b_l27_harmful_self_bundle_v1 \
  --overwrite

Outputs
=======
selected_sids.json
discovery_sample.csv
discovery_head_summary.csv
discovery_ranked_harmful_self_heads.csv
causal_conditions.json
eval_baseline.csv
causal_results.jsonl
causal_summary.csv
causal_summary_by_cohort.csv
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


SCRIPT_VERSION = "qwen7b-l27-harmful-self-bundle-v1"

RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
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

EPS = 1e-12


# =============================================================================
# CLI / IO
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
        "--num-samples",
        type=int,
        default=0,
        help="0 = all dataset records; positive = deterministic random subset.",
    )
    p.add_argument("--seed", type=int, default=17)

    p.add_argument(
        "--discovery-ratio",
        type=float,
        default=0.50,
        help="Fraction used ONLY to discover/rank harmful heads.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=200,
        help="0 = all held-out eval samples; otherwise stratified cap.",
    )

    p.add_argument(
        "--layer",
        type=int,
        default=27,
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="Number of ranked harmful self heads used for prefix bundles.",
    )
    p.add_argument(
        "--rank-cohort",
        default="overwrite",
        choices=("overwrite", "all"),
        help="Discovery cohort used to rank harmful self heads.",
    )
    p.add_argument(
        "--min-overwrite-n",
        type=int,
        default=10,
        help=(
            "If overwrite cohort has fewer samples than this, fall back to "
            "all-discovery ranking."
        ),
    )

    p.add_argument(
        "--ablation-scale",
        type=float,
        default=1.0,
        help="1.0 = exact natural self-message removal.",
    )
    p.add_argument(
        "--random-bundles-per-k",
        type=int,
        default=1,
        help="Matched random bundle controls for each prefix size K.",
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
            hits.append(
                (
                    match.start(),
                    relation,
                )
            )

    for pattern, relation in (
        (r"\bunder(?:neath)?\b|\bbeneath\b", "below"),
        (r"\bover\b|\bon top\b", "above"),
    ):
        match = re.search(
            pattern,
            text,
        )

        if match:
            hits.append(
                (
                    match.start(),
                    relation,
                )
            )

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


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
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
            writer.writerow(
                dict(row)
            )


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
        if hasattr(
            config,
            name,
        ):
            setattr(
                config,
                name,
                None,
            )


# =============================================================================
# Dataset split
# =============================================================================

def sample_records(
    records: Sequence[Any],
    *,
    n: int,
    seed: int,
) -> List[Any]:
    ordered = sorted(
        records,
        key=lambda record: int(
            record.sid
        ),
    )

    if n <= 0 or n >= len(ordered):
        return ordered

    selected = random.Random(
        seed
    ).sample(
        ordered,
        int(n),
    )

    selected.sort(
        key=lambda record: int(
            record.sid
        )
    )

    return selected


def stratified_split(
    records: Sequence[Any],
    *,
    ratio: float,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    if not (
        0.0 < ratio < 1.0
    ):
        raise ValueError(
            "--discovery-ratio must be strictly between 0 and 1."
        )

    grouped: Dict[
        str,
        List[Any],
    ] = defaultdict(list)

    for record in records:
        gt = normalize_relation(
            record.relation
        )

        if gt in RELATIONS:
            grouped[
                gt
            ].append(
                record
            )

    rng = random.Random(
        seed
    )

    discovery: List[Any] = []
    evaluation: List[Any] = []

    for relation in RELATIONS:
        group = list(
            grouped[
                relation
            ]
        )

        rng.shuffle(
            group
        )

        if len(group) < 2:
            raise RuntimeError(
                f"Need >=2 samples for relation {relation}; got {len(group)}."
            )

        n_discovery = int(
            round(
                len(group)
                * ratio
            )
        )

        n_discovery = max(
            1,
            min(
                len(group) - 1,
                n_discovery,
            ),
        )

        discovery.extend(
            group[
                :
                n_discovery
            ]
        )
        evaluation.extend(
            group[
                n_discovery:
            ]
        )

    discovery.sort(
        key=lambda record: int(
            record.sid
        )
    )
    evaluation.sort(
        key=lambda record: int(
            record.sid
        )
    )

    return (
        discovery,
        evaluation,
    )


def stratified_limit(
    records: Sequence[Any],
    *,
    n: int,
    seed: int,
) -> List[Any]:
    records = list(
        records
    )

    if n <= 0 or n >= len(records):
        return sorted(
            records,
            key=lambda record: int(
                record.sid
            ),
        )

    grouped: Dict[
        str,
        List[Any],
    ] = defaultdict(list)

    for record in records:
        gt = normalize_relation(
            record.relation
        )

        if gt in RELATIONS:
            grouped[
                gt
            ].append(
                record
            )

    rng = random.Random(
        seed
    )

    for group in grouped.values():
        rng.shuffle(
            group
        )

    cursors = Counter()
    output: List[Any] = []

    while len(output) < n:
        moved = False

        for relation in RELATIONS:
            group = grouped.get(
                relation,
                [],
            )
            cursor = int(
                cursors[
                    relation
                ]
            )

            if (
                cursor < len(group)
                and len(output) < n
            ):
                output.append(
                    group[
                        cursor
                    ]
                )
                cursors[
                    relation
                ] += 1
                moved = True

        if not moved:
            break

    output.sort(
        key=lambda record: int(
            record.sid
        )
    )

    return output


# =============================================================================
# Prompt / logit lens
# =============================================================================

def relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    output: Dict[
        str,
        List[int],
    ] = {}

    for relation in RELATIONS:
        ids = set()

        for text in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            if len(token_ids) == 1:
                ids.add(
                    int(
                        token_ids[
                            0
                        ]
                    )
                )

        if not ids:
            token_ids = tokenizer.encode(
                " " + relation,
                add_special_tokens=False,
            )

            if not token_ids:
                raise RuntimeError(
                    f"No token id for relation {relation}."
                )

            ids.add(
                int(
                    token_ids[
                        -1
                    ]
                )
            )

        output[
            relation
        ] = sorted(
            ids
        )

    return output


def build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Any:
    rendered = (
        probe.build_chat_prompt(
            processor,
            question,
            True,
        )
    )

    return probe.process_inputs(
        processor,
        rendered,
        image,
        device,
    )


def get_attr_path(
    root: Any,
    path: str,
) -> Any:
    current = root

    for part in path.split(
        "."
    ):
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
) -> Tuple[
    Optional[
        torch.nn.Module
    ],
    str,
]:
    parent = (
        decoder_path.rsplit(
            ".",
            1,
        )[0]
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
            return (
                module,
                path,
            )

    return (
        None,
        "unresolved",
    )


class RelationLogitLens:
    def __init__(
        self,
        *,
        model: Any,
        final_norm: torch.nn.Module,
        token_map: Mapping[
            str,
            Sequence[int],
        ],
    ) -> None:
        self.final_norm = (
            final_norm
        )
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
                "No LM head/output embedding weight."
            )

        weight = (
            output_embeddings.weight
        )

        union = sorted(
            {
                token_id
                for ids in self.token_map.values()
                for token_id in ids
            }
        )

        self.lookup = {
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

        if torch.is_tensor(
            bias
        ):
            self.bias_rows = (
                bias.index_select(
                    0,
                    index,
                )
                .detach()
            )
        else:
            self.bias_rows = None

        self.device = (
            weight.device
        )
        self.dtype = (
            weight.dtype
        )

    @torch.inference_mode()
    def scores(
        self,
        states: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(
            states,
            dtype=np.float32,
        )

        leading = (
            array.shape[
                :-1
            ]
        )
        hidden = int(
            array.shape[
                -1
            ]
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

        normalized = (
            self.final_norm(
                tensor
            )
        )

        if isinstance(
            normalized,
            (
                tuple,
                list,
            ),
        ):
            normalized = (
                normalized[
                    0
                ]
            )

        token_logits = (
            normalized
            @ self.weight_rows.T
        )

        if (
            self.bias_rows
            is not None
        ):
            token_logits = (
                token_logits
                + self.bias_rows
            )

        relation_scores: List[
            torch.Tensor
        ] = []

        for relation in RELATIONS:
            columns = [
                self.lookup[
                    token_id
                ]
                for token_id in self.token_map[
                    relation
                ]
            ]

            relation_scores.append(
                token_logits[
                    :,
                    columns,
                ]
                .max(
                    dim=-1
                )
                .values
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
            .astype(
                np.float32
            )
        )

        return result.reshape(
            *leading,
            len(
                RELATIONS
            ),
        )


def lens_pred(
    scores: np.ndarray,
) -> str:
    return RELATIONS[
        int(
            np.argmax(
                np.asarray(
                    scores
                )
            )
        )
    ]


def lens_margin(
    scores: np.ndarray,
    gt: str,
) -> float:
    values = np.asarray(
        scores,
        dtype=np.float32,
    )

    return float(
        values[
            RID[
                gt
            ]
        ]
        - values[
            RID[
                OPPOSITE[
                    gt
                ]
            ]
        ]
    )


# =============================================================================
# Trace math: x / r_attn / y / per-head self messages
# =============================================================================

def trace_target_index(
    trace: Any,
    position: int,
) -> int:
    lookup = {
        int(value): index
        for index, value in enumerate(
            trace.target_positions
        )
    }

    if int(position) not in lookup:
        raise RuntimeError(
            f"Position {position} missing from trace targets "
            f"{trace.target_positions}."
        )

    return int(
        lookup[
            int(position)
        ]
    )


def trace_block_state(
    trace: Any,
    position: int,
) -> np.ndarray:
    local = trace_target_index(
        trace,
        position,
    )

    return (
        trace.block_output[
            local
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


def trace_attention_state(
    trace: Any,
    position: int,
) -> np.ndarray:
    local = trace_target_index(
        trace,
        position,
    )

    return (
        trace.attention_output[
            local
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


def per_head_self_messages(
    *,
    trace: Any,
    prompt_last: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    c_h^self = W_O^h A_h[last,last] V_h[last]

    Returns:
        messages [Hq,Dmodel]
        self_attention_weight [Hq]
    """
    local_target = (
        trace_target_index(
            trace,
            prompt_last,
        )
    )

    source_position = int(
        prompt_last
    )

    if (
        source_position
        >= int(
            trace.value_states.shape[
                1
            ]
        )
    ):
        raise RuntimeError(
            f"prompt_last={prompt_last} outside value seq "
            f"{trace.value_states.shape[1]}."
        )

    weights = (
        trace.attention_weights[
            :,
            local_target,
            source_position,
        ]
        .float()
    )  # [H]

    values = (
        trace.value_states[
            :,
            source_position,
            :,
        ]
        .float()
    )  # [H,Dh]

    pre = (
        weights[
            :,
            None,
        ]
        * values
    )

    post = torch.einsum(
        "hd,ohd->ho",
        pre,
        trace.o_proj_weight.float(),
    )

    return (
        post.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),
        weights.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),
    )


def infer_attention_counts(
    *,
    attention: Any,
    model: Any,
) -> Tuple[
    int,
    int,
]:
    n_query = None
    n_kv = None

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

        if n_query is None:
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
                        n_query = int(
                            value
                        )
                        break
                    except Exception:
                        pass

        if n_kv is None:
            for name in (
                "num_key_value_heads",
                "num_kv_heads",
            ):
                value = getattr(
                    obj,
                    name,
                    None,
                )

                if value is not None:
                    try:
                        n_kv = int(
                            value
                        )
                        break
                    except Exception:
                        pass

    if n_query is None:
        raise RuntimeError(
            "Cannot infer query-head count."
        )

    if n_kv is None:
        n_kv = n_query

    return (
        int(
            n_query
        ),
        int(
            n_kv
        ),
    )


def kv_head_for_query(
    query_head: int,
    n_query: int,
    n_kv: int,
) -> int:
    if n_query % n_kv != 0:
        raise RuntimeError(
            f"Hq={n_query} not divisible by Hkv={n_kv}."
        )

    group_size = int(
        n_query
        // n_kv
    )

    return int(
        query_head
        // group_size
    )


def query_heads_for_kv(
    kv_head: int,
    n_query: int,
    n_kv: int,
) -> List[int]:
    if n_query % n_kv != 0:
        raise RuntimeError(
            f"Hq={n_query} not divisible by Hkv={n_kv}."
        )

    group_size = int(
        n_query
        // n_kv
    )

    start = int(
        kv_head
        * group_size
    )

    return list(
        range(
            start,
            start
            + group_size,
        )
    )


# =============================================================================
# Generation patch
# =============================================================================

def first_3d(
    output: Any,
) -> torch.Tensor:
    if torch.is_tensor(
        output
    ):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected [B,S,D], got {tuple(output.shape)}."
            )

        return output

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):
        for item in output:
            if (
                torch.is_tensor(
                    item
                )
                and item.ndim == 3
            ):
                return item

    raise RuntimeError(
        "Could not find 3D attention output."
    )


def replace_first_3d(
    output: Any,
    replacement: torch.Tensor,
) -> Any:
    if torch.is_tensor(
        output
    ):
        return replacement

    if isinstance(
        output,
        tuple,
    ):
        items = list(
            output
        )

        for index, item in enumerate(
            items
        ):
            if (
                torch.is_tensor(
                    item
                )
                and item.ndim == 3
            ):
                items[
                    index
                ] = replacement
                return tuple(
                    items
                )

    if isinstance(
        output,
        list,
    ):
        items = list(
            output
        )

        for index, item in enumerate(
            items
        ):
            if (
                torch.is_tensor(
                    item
                )
                and item.ndim == 3
            ):
                items[
                    index
                ] = replacement
                return items

    raise RuntimeError(
        "Could not replace attention output tensor."
    )


class PromptLastDelta:
    def __init__(
        self,
        *,
        attention_module: torch.nn.Module,
        prompt_length: int,
        prompt_last: int,
        delta: np.ndarray,
    ) -> None:
        self.attention_module = (
            attention_module
        )
        self.prompt_length = int(
            prompt_length
        )
        self.prompt_last = int(
            prompt_last
        )
        self.delta = np.asarray(
            delta,
            dtype=np.float32,
        )

        self.handle = None
        self.applications = 0

    def __enter__(
        self,
    ) -> "PromptLastDelta":
        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> Any:
            hidden = first_3d(
                output
            )

            # Apply only to full multimodal prefill.
            if (
                int(
                    hidden.shape[
                        1
                    ]
                )
                != self.prompt_length
            ):
                return None

            if not (
                0
                <= self.prompt_last
                < int(
                    hidden.shape[
                        1
                    ]
                )
            ):
                raise RuntimeError(
                    f"prompt_last={self.prompt_last}, "
                    f"q_len={hidden.shape[1]}."
                )

            modified = hidden.clone()

            modified[
                0,
                self.prompt_last,
            ] += torch.as_tensor(
                self.delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            self.applications += 1

            return replace_first_3d(
                output,
                modified,
            )

        self.handle = (
            self.attention_module.register_forward_hook(
                hook
            )
        )

        return self

    def validate(
        self,
    ) -> None:
        if self.applications != 1:
            raise RuntimeError(
                f"Expected exactly one prefill patch; got "
                f"{self.applications}."
            )

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        if (
            self.handle
            is not None
        ):
            with contextlib.suppress(
                Exception
            ):
                self.handle.remove()

        self.handle = None


@torch.inference_mode()
def greedy_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    max_new_tokens: int,
) -> Tuple[
    Optional[str],
    str,
]:
    raw_prompt_length = int(
        batch[
            "input_ids"
        ].shape[
            1
        ]
    )

    generated = model.generate(
        **batch,
        max_new_tokens=int(
            max_new_tokens
        ),
        do_sample=False,
        use_cache=True,
    )

    text = (
        processor.tokenizer.decode(
            generated[
                0,
                raw_prompt_length:,
            ],
            skip_special_tokens=True,
        )
        .strip()
    )

    del generated

    return (
        normalize_relation(
            text
        ),
        text,
    )


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
) -> Tuple[
    Optional[str],
    str,
]:
    with PromptLastDelta(
        attention_module=attention_module,
        prompt_length=prompt_length,
        prompt_last=prompt_last,
        delta=delta,
    ) as patch:
        pred, text = (
            greedy_generation(
                model=model,
                processor=processor,
                batch=batch,
                max_new_tokens=max_new_tokens,
            )
        )

        patch.validate()

    return (
        pred,
        text,
    )


# =============================================================================
# Discovery ranking
# =============================================================================

def summarize_discovery_heads(
    *,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    n_heads: int,
    n_query: int,
    n_kv: int,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    output: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for head in range(
        n_heads
    ):
        current = [
            row
            for row in rows
            if int(
                row[
                    "head"
                ]
            )
            == head
        ]

        overwrite = [
            row
            for row in current
            if bool(
                row[
                    "overwrite_sample"
                ]
            )
        ]

        output.append({
            "head": head,
            "head_name": (
                f"L27H{head:02d}"
            ),
            "kv_head": (
                kv_head_for_query(
                    head,
                    n_query,
                    n_kv,
                )
            ),
            "N_all": len(
                current
            ),
            "mean_self_weight_all": safe_mean(
                row[
                    "self_attention_weight"
                ]
                for row in current
            ),
            "mean_self_message_norm_all": safe_mean(
                row[
                    "self_message_norm"
                ]
                for row in current
            ),
            "mean_removal_improvement_all": safe_mean(
                row[
                    "removal_improvement"
                ]
                for row in current
            ),
            "positive_removal_rate_all": safe_mean(
                float(
                    row[
                        "removal_improvement"
                    ]
                    > 0
                )
                for row in current
            ),
            "N_overwrite": len(
                overwrite
            ),
            "mean_removal_improvement_overwrite": safe_mean(
                row[
                    "removal_improvement"
                ]
                for row in overwrite
            ),
            "positive_removal_rate_overwrite": safe_mean(
                float(
                    row[
                        "removal_improvement"
                    ]
                    > 0
                )
                for row in overwrite
            ),
        })

    return output


# =============================================================================
# Bundle construction
# =============================================================================

def canonical_bundle(
    heads: Sequence[int],
) -> Tuple[int, ...]:
    return tuple(
        sorted(
            set(
                map(
                    int,
                    heads,
                )
            )
        )
    )


def bundle_name(
    kind: str,
    heads: Sequence[int],
    extra: str = "",
) -> str:
    head_text = "-".join(
        f"H{int(head):02d}"
        for head in canonical_bundle(
            heads
        )
    )

    name = (
        f"{kind}__{head_text}"
    )

    if extra:
        name += (
            "__"
            + str(
                extra
            )
        )

    return name


def build_conditions(
    *,
    ranked_heads: Sequence[int],
    top_k: int,
    n_query: int,
    n_kv: int,
    random_bundles_per_k: int,
    seed: int,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    ranked_heads = list(
        map(
            int,
            ranked_heads,
        )
    )

    top_k = max(
        1,
        min(
            int(
                top_k
            ),
            len(
                ranked_heads
            ),
        ),
    )

    output: List[
        Dict[
            str,
            Any,
        ]
    ] = []
    seen = set()

    def add(
        *,
        kind: str,
        heads: Sequence[int],
        label: str,
        k: int,
        kv_head: Optional[int] = None,
        random_index: Optional[int] = None,
    ) -> None:
        bundle = canonical_bundle(
            heads
        )

        if not bundle:
            return

        key = (
            bundle,
            kind,
            label,
        )

        if key in seen:
            return

        seen.add(
            key
        )

        output.append({
            "condition": (
                bundle_name(
                    kind,
                    bundle,
                    label,
                )
            ),
            "kind": kind,
            "label": label,
            "heads": list(
                bundle
            ),
            "K": int(
                k
            ),
            "kv_head": kv_head,
            "random_index": random_index,
        })

    # Prefix bundles: this is the main bundle-ablation curve.
    for k in range(
        1,
        top_k + 1,
    ):
        add(
            kind="prefix",
            heads=ranked_heads[
                :
                k
            ],
            label=f"top{k}",
            k=k,
        )

    # Shared KV family containing discovery Top1.
    top1 = int(
        ranked_heads[
            0
        ]
    )
    winning_kv = (
        kv_head_for_query(
            top1,
            n_query,
            n_kv,
        )
    )

    winning_group = (
        query_heads_for_kv(
            winning_kv,
            n_query,
            n_kv,
        )
    )

    add(
        kind="kv_group",
        heads=winning_group,
        label=f"top1_kv{winning_kv}",
        k=len(
            winning_group
        ),
        kv_head=winning_kv,
    )

    # All other complete KV groups are structural controls.
    for kv_head in range(
        n_kv
    ):
        group = (
            query_heads_for_kv(
                kv_head,
                n_query,
                n_kv,
            )
        )

        add(
            kind=(
                "kv_group"
                if kv_head
                == winning_kv
                else "kv_control"
            ),
            heads=group,
            label=f"kv{kv_head}",
            k=len(
                group
            ),
            kv_head=kv_head,
        )

    # Random same-size controls. Exclude ranked top-K heads so controls do not
    # accidentally contain the discovery-selected family.
    selected_set = set(
        ranked_heads[
            :
            top_k
        ]
    )

    pool = [
        head
        for head in range(
            n_query
        )
        if head
        not in selected_set
    ]

    rng = random.Random(
        seed
    )

    for k in range(
        1,
        top_k + 1,
    ):
        if k > len(
            pool
        ):
            continue

        for random_index in range(
            int(
                random_bundles_per_k
            )
        ):
            heads = rng.sample(
                pool,
                k,
            )

            add(
                kind="random",
                heads=heads,
                label=f"k{k}_r{random_index}",
                k=k,
                random_index=random_index,
            )

    return output


# =============================================================================
# Causal summary
# =============================================================================

def summarize_causal(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    conditions: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> Tuple[
    List[
        Dict[
            str,
            Any,
        ]
    ],
    List[
        Dict[
            str,
            Any,
        ]
    ],
]:
    grouped: Dict[
        str,
        List[
            Mapping[
                str,
                Any,
            ]
        ],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            str(
                row[
                    "condition"
                ]
            )
        ].append(
            row
        )

    summary: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    cohort_summary: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    condition_lookup = {
        str(
            condition[
                "condition"
            ]
        ): condition
        for condition in conditions
    }

    for condition_name, current in grouped.items():
        condition = condition_lookup[
            condition_name
        ]

        baseline_acc = safe_mean(
            float(
                row[
                    "baseline_correct"
                ]
            )
            for row in current
        )

        patched_acc = safe_mean(
            float(
                row[
                    "patched_correct"
                ]
            )
            for row in current
        )

        w2c = sum(
            bool(
                row[
                    "wrong_to_correct"
                ]
            )
            for row in current
        )

        c2w = sum(
            bool(
                row[
                    "correct_to_wrong"
                ]
            )
            for row in current
        )

        changed = sum(
            bool(
                row[
                    "generation_changed"
                ]
            )
            for row in current
        )

        summary.append({
            "condition": condition_name,
            "kind": condition[
                "kind"
            ],
            "label": condition[
                "label"
            ],
            "heads": ",".join(
                map(
                    str,
                    condition[
                        "heads"
                    ],
                )
            ),
            "K": condition[
                "K"
            ],
            "kv_head": condition.get(
                "kv_head"
            ),
            "random_index": condition.get(
                "random_index"
            ),
            "N": len(
                current
            ),
            "baseline_acc": baseline_acc,
            "patched_acc": patched_acc,
            "delta_acc": (
                patched_acc
                - baseline_acc
            ),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net_repairs": (
                w2c
                - c2w
            ),
            "generation_changed": changed,
            "generation_changed_rate": (
                changed
                / max(
                    len(
                        current
                    ),
                    1,
                )
            ),
            "mean_bundle_norm": safe_mean(
                row[
                    "bundle_norm"
                ]
                for row in current
            ),
        })

        cohorts = {
            "all": current,
            "eval_overwrite": [
                row
                for row in current
                if bool(
                    row[
                        "eval_overwrite_sample"
                    ]
                )
            ],
            "baseline_wrong": [
                row
                for row in current
                if not bool(
                    row[
                        "baseline_correct"
                    ]
                )
            ],
            "baseline_correct": [
                row
                for row in current
                if bool(
                    row[
                        "baseline_correct"
                    ]
                )
            ],
        }

        for cohort_name, subset in cohorts.items():
            if not subset:
                continue

            base_acc = safe_mean(
                float(
                    row[
                        "baseline_correct"
                    ]
                )
                for row in subset
            )

            patch_acc = safe_mean(
                float(
                    row[
                        "patched_correct"
                    ]
                )
                for row in subset
            )

            cohort_summary.append({
                "condition": condition_name,
                "kind": condition[
                    "kind"
                ],
                "label": condition[
                    "label"
                ],
                "K": condition[
                    "K"
                ],
                "cohort": cohort_name,
                "N": len(
                    subset
                ),
                "baseline_acc": base_acc,
                "patched_acc": patch_acc,
                "delta_acc": (
                    patch_acc
                    - base_acc
                ),
                "wrong_to_correct": sum(
                    bool(
                        row[
                            "wrong_to_correct"
                        ]
                    )
                    for row in subset
                ),
                "correct_to_wrong": sum(
                    bool(
                        row[
                            "correct_to_wrong"
                        ]
                    )
                    for row in subset
                ),
                "generation_changed_rate": safe_mean(
                    float(
                        row[
                            "generation_changed"
                        ]
                    )
                    for row in subset
                ),
            })

    summary.sort(
        key=lambda row: (
            (
                0
                if row[
                    "kind"
                ]
                == "prefix"
                else 1
            ),
            int(
                row[
                    "K"
                ]
            ),
            -float(
                row[
                    "delta_acc"
                ]
            ),
            str(
                row[
                    "condition"
                ]
            ),
        )
    )

    cohort_summary.sort(
        key=lambda row: (
            str(
                row[
                    "condition"
                ]
            ),
            str(
                row[
                    "cohort"
                ]
            ),
        )
    )

    return (
        summary,
        cohort_summary,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.model != "qwen-7b":
        raise ValueError(
            "v1 intentionally supports --model qwen-7b only."
        )

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

    records, audit = (
        base.load_records(
            args.dataset,
            Path(
                args.data_root
            ),
            None,
        )
    )

    selected = sample_records(
        records,
        n=args.num_samples,
        seed=args.seed,
    )

    discovery_records, eval_records = (
        stratified_split(
            selected,
            ratio=args.discovery_ratio,
            seed=args.seed
            + 1009,
        )
    )

    eval_records = (
        stratified_limit(
            eval_records,
            n=args.eval_max_samples,
            seed=args.seed
            + 2027,
        )
    )

    (
        output_dir
        / "selected_sids.json"
    ).write_text(
        json.dumps(
            {
                "seed": args.seed,
                "N_selected": len(
                    selected
                ),
                "N_discovery": len(
                    discovery_records
                ),
                "N_eval": len(
                    eval_records
                ),
                "discovery_sids": [
                    int(
                        record.sid
                    )
                    for record in discovery_records
                ],
                "eval_sids": [
                    int(
                        record.sid
                    )
                    for record in eval_records
                ],
            },
            ensure_ascii=False,
            indent=2,
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

        if len(
            decoder_layers
        ) != 28:
            raise RuntimeError(
                f"Expected Qwen-7B 28 decoder layers; "
                f"got {len(decoder_layers)} at {decoder_path}."
            )

        layer = int(
            args.layer
        )

        if layer != 27:
            print(
                f"[warning] mechanism interpretation was designed for L27; "
                f"running L{layer}.",
                flush=True,
            )

        if not (
            1
            <= layer
            < len(
                decoder_layers
            )
        ):
            raise ValueError(
                f"L{layer} outside valid range."
            )

        attention_module = (
            attention_helper.resolve_self_attention(
                decoder_layers[
                    layer
                ]
            )
        )

        n_query, n_kv = (
            infer_attention_counts(
                attention=attention_module,
                model=model,
            )
        )

        if (
            n_query != 28
            or n_kv != 4
        ):
            print(
                f"[warning] expected Qwen7B Hq=28,Hkv=4; "
                f"observed Hq={n_query},Hkv={n_kv}.",
                flush=True,
            )

        final_norm, final_norm_path = (
            resolve_final_norm(
                model,
                decoder_path,
            )
        )

        if final_norm is None:
            raise RuntimeError(
                "Could not resolve final norm."
            )

        token_map = relation_token_variants(
            processor.tokenizer
        )

        lens = RelationLogitLens(
            model=model,
            final_norm=final_norm,
            token_map=token_map,
        )

        trace_layers = [
            layer - 1,
            layer,
        ]

        print(
            "\n"
            + "=" * 180
        )
        print(
            "QWEN-7B L27 HARMFUL SELF-MESSAGE DISCOVERY + HELD-OUT BUNDLE ABLATION"
        )
        print(
            "=" * 180
        )
        print(
            "N selected/discovery/eval:",
            len(
                selected
            ),
            len(
                discovery_records
            ),
            len(
                eval_records
            ),
        )
        print(
            "layer:",
            layer,
        )
        print(
            "query heads / KV heads:",
            n_query,
            "/",
            n_kv,
        )
        print(
            "rank cohort:",
            args.rank_cohort,
        )
        print(
            "top K:",
            args.top_k,
        )
        print(
            "ablation scale:",
            args.ablation_scale,
        )
        print(
            "=" * 180,
            flush=True,
        )

        # =====================================================================
        # Phase 1: DISCOVERY natural logit-lens scan
        # =====================================================================

        discovery_sample_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        discovery_head_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        final_lens_native_match = 0
        final_lens_native_n = 0
        overwrite_count = 0

        for sample_index, record in enumerate(
            tqdm(
                discovery_records,
                desc="discover-self-heads",
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
                        f"Unsupported GT {record.relation!r}."
                    )

                question = (
                    args.prompt_template.format(
                        subject=record.subject,
                        reference=record.reference,
                    )
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

                raw_length = int(
                    batch[
                        "input_ids"
                    ].shape[
                        1
                    ]
                )
                prompt_last = (
                    raw_length
                    - 1
                )

                baseline, traces = (
                    attention_helper.run_and_trace(
                        model=model,
                        batch=batch,
                        token_map=token_map,
                        decoder_layers=decoder_layers,
                        layer_indices=trace_layers,
                        target_positions=[
                            prompt_last
                        ],
                    )
                )

                x = trace_block_state(
                    traces[
                        layer - 1
                    ],
                    prompt_last,
                )

                attention_output = (
                    trace_attention_state(
                        traces[
                            layer
                        ],
                        prompt_last,
                    )
                )

                r_attn = (
                    x
                    + attention_output
                ).astype(
                    np.float32
                )

                y = trace_block_state(
                    traces[
                        layer
                    ],
                    prompt_last,
                )

                self_messages, self_weights = (
                    per_head_self_messages(
                        trace=traces[
                            layer
                        ],
                        prompt_last=prompt_last,
                    )
                )

                if int(
                    self_messages.shape[
                        0
                    ]
                ) != n_query:
                    raise RuntimeError(
                        f"Expected {n_query} query-head self messages, "
                        f"got {self_messages.shape[0]}."
                    )

                x_scores = lens.scores(
                    x
                )
                rattn_scores = lens.scores(
                    r_attn
                )
                y_scores = lens.scores(
                    y
                )

                x_pred = lens_pred(
                    x_scores
                )
                rattn_pred = lens_pred(
                    rattn_scores
                )
                y_pred = lens_pred(
                    y_scores
                )

                x_margin = lens_margin(
                    x_scores,
                    gt,
                )
                rattn_margin = lens_margin(
                    rattn_scores,
                    gt,
                )
                y_margin = lens_margin(
                    y_scores,
                    gt,
                )

                overwrite_sample = (
                    x_pred == gt
                    and rattn_pred != gt
                )

                overwrite_count += int(
                    overwrite_sample
                )

                final_lens_native_n += 1
                final_lens_native_match += int(
                    y_pred
                    == str(
                        baseline[
                            "prediction"
                        ]
                    )
                )

                minus_states = (
                    r_attn[
                        None,
                        :,
                    ]
                    - self_messages
                )

                minus_scores = lens.scores(
                    minus_states
                )

                minus_margins = (
                    minus_scores[
                        :,
                        RID[
                            gt
                        ],
                    ]
                    - minus_scores[
                        :,
                        RID[
                            OPPOSITE[
                                gt
                            ]
                        ],
                    ]
                )

                removal_improvements = (
                    minus_margins
                    - rattn_margin
                )

                discovery_sample_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "native_first_step_pred": str(
                        baseline[
                            "prediction"
                        ]
                    ),
                    "x_pred": x_pred,
                    "x_correct": x_pred == gt,
                    "x_margin": x_margin,
                    "rattn_pred": rattn_pred,
                    "rattn_correct": rattn_pred == gt,
                    "rattn_margin": rattn_margin,
                    "attention_margin_gain": (
                        rattn_margin
                        - x_margin
                    ),
                    "y_pred": y_pred,
                    "y_correct": y_pred == gt,
                    "y_margin": y_margin,
                    "mlp_margin_gain": (
                        y_margin
                        - rattn_margin
                    ),
                    "overwrite_sample": overwrite_sample,
                })

                for head in range(
                    n_query
                ):
                    discovery_head_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "head": head,
                        "head_name": (
                            f"L{layer}H{head:02d}"
                        ),
                        "kv_head": (
                            kv_head_for_query(
                                head,
                                n_query,
                                n_kv,
                            )
                        ),
                        "overwrite_sample": overwrite_sample,
                        "self_attention_weight": float(
                            self_weights[
                                head
                            ]
                        ),
                        "self_message_norm": float(
                            np.linalg.norm(
                                self_messages[
                                    head
                                ]
                            )
                        ),
                        "removal_improvement": float(
                            removal_improvements[
                                head
                            ]
                        ),
                        "removal_improves_margin": bool(
                            removal_improvements[
                                head
                            ]
                            > 0
                        ),
                    })

                del traces

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "discovery",
                        "sid": int(
                            getattr(
                                record,
                                "sid",
                                -1,
                            )
                        ),
                        "error_type": (
                            type(
                                exc
                            ).__name__
                        ),
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if (
                    image
                    is not None
                ):
                    with contextlib.suppress(
                        Exception
                    ):
                        image.close()

                del batch

                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every
                    > 0
                    and sample_index
                    % args.empty_cache_every
                    == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(
            output_dir
            / "discovery_sample.csv",
            discovery_sample_rows,
        )

        discovery_summary = (
            summarize_discovery_heads(
                rows=discovery_head_rows,
                n_heads=n_query,
                n_query=n_query,
                n_kv=n_kv,
            )
        )

        effective_rank_cohort = (
            args.rank_cohort
        )

        if (
            args.rank_cohort
            == "overwrite"
            and overwrite_count
            < int(
                args.min_overwrite_n
            )
        ):
            print(
                f"[warning] overwrite discovery N={overwrite_count} < "
                f"{args.min_overwrite_n}; falling back to all-sample ranking.",
                flush=True,
            )
            effective_rank_cohort = (
                "all"
            )

        rank_metric = (
            "mean_removal_improvement_overwrite"
            if effective_rank_cohort
            == "overwrite"
            else "mean_removal_improvement_all"
        )

        positive_metric = (
            "positive_removal_rate_overwrite"
            if effective_rank_cohort
            == "overwrite"
            else "positive_removal_rate_all"
        )

        discovery_summary.sort(
            key=lambda row: (
                -float(
                    row[
                        rank_metric
                    ]
                )
                if math.isfinite(
                    float(
                        row[
                            rank_metric
                        ]
                    )
                )
                else float(
                    "inf"
                ),
                -float(
                    row[
                        positive_metric
                    ]
                )
                if math.isfinite(
                    float(
                        row[
                            positive_metric
                        ]
                    )
                )
                else float(
                    "inf"
                ),
                -float(
                    row[
                        "mean_removal_improvement_all"
                    ]
                ),
                int(
                    row[
                        "head"
                    ]
                ),
            )
        )

        for rank, row in enumerate(
            discovery_summary,
            start=1,
        ):
            row[
                "rank"
            ] = rank
            row[
                "rank_metric"
            ] = rank_metric
            row[
                "rank_value"
            ] = row[
                rank_metric
            ]

        write_csv(
            output_dir
            / "discovery_head_summary.csv",
            discovery_summary,
        )

        top_k = max(
            1,
            min(
                int(
                    args.top_k
                ),
                n_query,
            ),
        )

        ranked_top = (
            discovery_summary[
                :
                top_k
            ]
        )

        write_csv(
            output_dir
            / "discovery_ranked_harmful_self_heads.csv",
            ranked_top,
        )

        ranked_heads = [
            int(
                row[
                    "head"
                ]
            )
            for row in discovery_summary
        ]

        conditions = build_conditions(
            ranked_heads=ranked_heads,
            top_k=top_k,
            n_query=n_query,
            n_kv=n_kv,
            random_bundles_per_k=args.random_bundles_per_k,
            seed=args.seed
            + 9001,
        )

        (
            output_dir
            / "causal_conditions.json"
        ).write_text(
            json.dumps(
                conditions,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\n"
            + "=" * 180
        )
        print(
            "DISCOVERY: HARMFUL L27 SELF HEADS"
        )
        print(
            "=" * 180
        )
        print(
            f"Discovery N={len(discovery_sample_rows)} | "
            f"overwrite N={overwrite_count} | "
            f"rank cohort={effective_rank_cohort}"
        )
        print(
            f"Final-layer true lens vs native first-step: "
            f"{100*final_lens_native_match/max(final_lens_native_n,1):.2f}%"
        )
        print(
            f"  {'rank':>4s} {'head':<9s} {'KV':>3s} "
            f"{'rankImp':>10s} {'rank+':>8s} "
            f"{'allImp':>10s} {'all+':>8s} "
            f"{'selfW':>9s} {'norm':>9s}"
        )

        for rank, row in enumerate(
            discovery_summary[
                :
                max(
                    top_k,
                    15,
                )
            ],
            start=1,
        ):
            print(
                f"  {rank:4d} "
                f"{str(row['head_name']):<9s} "
                f"{int(row['kv_head']):3d} "
                f"{float(row[rank_metric]):+10.5f} "
                f"{100*float(row[positive_metric]):7.2f}% "
                f"{float(row['mean_removal_improvement_all']):+10.5f} "
                f"{100*float(row['positive_removal_rate_all']):7.2f}% "
                f"{float(row['mean_self_weight_all']):9.5f} "
                f"{float(row['mean_self_message_norm_all']):9.4f}"
            )

        print(
            "\nFrozen top bundle order:",
            [
                f"H{head:02d}"
                for head in ranked_heads[
                    :
                    top_k
                ]
            ],
        )

        top1_kv = kv_head_for_query(
            ranked_heads[
                0
            ],
            n_query,
            n_kv,
        )

        print(
            f"Top1 shared KV group = KVH{top1_kv}:",
            [
                f"H{head:02d}"
                for head in query_heads_for_kv(
                    top1_kv,
                    n_query,
                    n_kv,
                )
            ],
        )
        print(
            "=" * 180,
            flush=True,
        )

        # =====================================================================
        # Phase 2: held-out full-generation bundle ablation
        # =====================================================================

        causal_results_path = (
            output_dir
            / "causal_results.jsonl"
        )

        eval_baseline_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        causal_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        print(
            f"\nHeld-out causal evaluation: "
            f"N={len(eval_records)} x {len(conditions)} conditions "
            f"= {len(eval_records)*len(conditions)} patched generations",
            flush=True,
        )

        for sample_index, record in enumerate(
            tqdm(
                eval_records,
                desc="heldout-bundle-ablation",
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
                        f"Unsupported GT {record.relation!r}."
                    )

                question = (
                    args.prompt_template.format(
                        subject=record.subject,
                        reference=record.reference,
                    )
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

                prompt_length = int(
                    batch[
                        "input_ids"
                    ].shape[
                        1
                    ]
                )
                prompt_last = (
                    prompt_length
                    - 1
                )

                baseline_trace, traces = (
                    attention_helper.run_and_trace(
                        model=model,
                        batch=batch,
                        token_map=token_map,
                        decoder_layers=decoder_layers,
                        layer_indices=trace_layers,
                        target_positions=[
                            prompt_last
                        ],
                    )
                )

                x = trace_block_state(
                    traces[
                        layer - 1
                    ],
                    prompt_last,
                )

                attention_output = (
                    trace_attention_state(
                        traces[
                            layer
                        ],
                        prompt_last,
                    )
                )

                r_attn = (
                    x
                    + attention_output
                ).astype(
                    np.float32
                )

                y = trace_block_state(
                    traces[
                        layer
                    ],
                    prompt_last,
                )

                self_messages, self_weights = (
                    per_head_self_messages(
                        trace=traces[
                            layer
                        ],
                        prompt_last=prompt_last,
                    )
                )

                x_scores = lens.scores(
                    x
                )
                rattn_scores = lens.scores(
                    r_attn
                )
                y_scores = lens.scores(
                    y
                )

                x_pred = lens_pred(
                    x_scores
                )
                rattn_pred = lens_pred(
                    rattn_scores
                )
                y_pred = lens_pred(
                    y_scores
                )

                eval_overwrite = (
                    x_pred == gt
                    and rattn_pred != gt
                )

                baseline_pred, baseline_text = (
                    greedy_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                )

                baseline_correct = (
                    baseline_pred
                    == gt
                )

                eval_baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "native_first_step_pred": str(
                        baseline_trace[
                            "prediction"
                        ]
                    ),
                    "generation_pred": baseline_pred,
                    "generation_text": baseline_text,
                    "generation_correct": baseline_correct,
                    "x_pred": x_pred,
                    "x_correct": x_pred == gt,
                    "x_margin": lens_margin(
                        x_scores,
                        gt,
                    ),
                    "rattn_pred": rattn_pred,
                    "rattn_correct": rattn_pred == gt,
                    "rattn_margin": lens_margin(
                        rattn_scores,
                        gt,
                    ),
                    "y_pred": y_pred,
                    "y_correct": y_pred == gt,
                    "y_margin": lens_margin(
                        y_scores,
                        gt,
                    ),
                    "eval_overwrite_sample": eval_overwrite,
                })

                for condition in conditions:
                    heads = [
                        int(
                            head
                        )
                        for head in condition[
                            "heads"
                        ]
                    ]

                    bundle = (
                        self_messages[
                            heads
                        ]
                        .sum(
                            axis=0
                        )
                        .astype(
                            np.float32
                        )
                    )

                    delta = (
                        -float(
                            args.ablation_scale
                        )
                        * bundle
                    ).astype(
                        np.float32
                    )

                    patched_pred, patched_text = (
                        patched_generation(
                            model=model,
                            processor=processor,
                            batch=batch,
                            attention_module=attention_module,
                            prompt_length=prompt_length,
                            prompt_last=prompt_last,
                            delta=delta,
                            max_new_tokens=args.max_new_tokens,
                        )
                    )

                    patched_correct = (
                        patched_pred
                        == gt
                    )

                    row = {
                        "sid": sid,
                        "gt": gt,
                        "condition": condition[
                            "condition"
                        ],
                        "kind": condition[
                            "kind"
                        ],
                        "label": condition[
                            "label"
                        ],
                        "K": condition[
                            "K"
                        ],
                        "heads": ",".join(
                            map(
                                str,
                                heads,
                            )
                        ),
                        "kv_head": condition.get(
                            "kv_head"
                        ),
                        "random_index": condition.get(
                            "random_index"
                        ),
                        "ablation_scale": (
                            args.ablation_scale
                        ),
                        "eval_overwrite_sample": (
                            eval_overwrite
                        ),
                        "baseline_pred": baseline_pred,
                        "baseline_correct": baseline_correct,
                        "patched_pred": patched_pred,
                        "patched_text": patched_text,
                        "patched_correct": patched_correct,
                        "wrong_to_correct": (
                            (
                                not baseline_correct
                            )
                            and patched_correct
                        ),
                        "correct_to_wrong": (
                            baseline_correct
                            and (
                                not patched_correct
                            )
                        ),
                        "generation_changed": (
                            patched_pred
                            != baseline_pred
                        ),
                        "bundle_norm": float(
                            np.linalg.norm(
                                bundle
                            )
                        ),
                        "delta_norm": float(
                            np.linalg.norm(
                                delta
                            )
                        ),
                        "mean_selected_self_weight": safe_mean(
                            self_weights[
                                heads
                            ]
                        ),
                    }

                    causal_rows.append(
                        row
                    )

                    append_jsonl(
                        causal_results_path,
                        row,
                    )

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
                            type(
                                exc
                            ).__name__
                        ),
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    },
                )

                if args.fail_fast:
                    raise

            finally:
                if (
                    image
                    is not None
                ):
                    with contextlib.suppress(
                        Exception
                    ):
                        image.close()

                del batch

                gc.collect()

                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every
                    > 0
                    and sample_index
                    % args.empty_cache_every
                    == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(
            output_dir
            / "eval_baseline.csv",
            eval_baseline_rows,
        )

        causal_summary, cohort_summary = (
            summarize_causal(
                causal_rows,
                conditions,
            )
        )

        write_csv(
            output_dir
            / "causal_summary.csv",
            causal_summary,
        )
        write_csv(
            output_dir
            / "causal_summary_by_cohort.csv",
            cohort_summary,
        )

        baseline_acc = safe_mean(
            float(
                row[
                    "generation_correct"
                ]
            )
            for row in eval_baseline_rows
        )

        eval_overwrite_n = sum(
            bool(
                row[
                    "eval_overwrite_sample"
                ]
            )
            for row in eval_baseline_rows
        )

        print(
            "\n"
            + "=" * 190
        )
        print(
            "HELD-OUT FULL-GENERATION SELF-BUNDLE ABLATION"
        )
        print(
            "=" * 190
        )
        print(
            f"Baseline ACC={100*baseline_acc:.2f}% "
            f"| N={len(eval_baseline_rows)} "
            f"| eval overwrite N={eval_overwrite_n}"
        )
        print(
            f"  {'kind':<11s} {'label':<15s} {'K':>2s} "
            f"{'heads':<28s} {'ACC':>8s} {'delta':>9s} "
            f"{'W->C':>5s} {'C->W':>5s} {'net':>5s} {'chg':>8s}"
        )

        # Print prefixes first in K order, then structural/random controls.
        display_rows = sorted(
            causal_summary,
            key=lambda row: (
                0
                if row[
                    "kind"
                ]
                == "prefix"
                else (
                    1
                    if row[
                        "kind"
                    ]
                    in (
                        "kv_group",
                        "kv_control",
                    )
                    else 2
                ),
                int(
                    row[
                        "K"
                    ]
                ),
                str(
                    row[
                        "label"
                    ]
                ),
            )
        )

        for row in display_rows:
            print(
                f"  {str(row['kind']):<11s} "
                f"{str(row['label']):<15s} "
                f"{int(row['K']):2d} "
                f"{str(row['heads']):<28s} "
                f"{100*float(row['patched_acc']):7.2f}% "
                f"{100*float(row['delta_acc']):+8.2f} "
                f"{int(row['wrong_to_correct']):5d} "
                f"{int(row['correct_to_wrong']):5d} "
                f"{int(row['net_repairs']):+5d} "
                f"{100*float(row['generation_changed_rate']):7.2f}%"
            )

        print(
            "\nPREFIX BUNDLE CURVE"
        )

        prefix_rows = [
            row
            for row in causal_summary
            if row[
                "kind"
            ]
            == "prefix"
        ]

        prefix_rows.sort(
            key=lambda row: int(
                row[
                    "K"
                ]
            )
        )

        for row in prefix_rows:
            print(
                f"  Top{int(row['K'])}: "
                f"heads=[{row['heads']}] "
                f"ACC={100*float(row['patched_acc']):.2f}% "
                f"delta={100*float(row['delta_acc']):+.2f}pp "
                f"net={int(row['net_repairs']):+d}"
            )

        print(
            "=" * 190
        )

        # =====================================================================
        # Config/report
        # =====================================================================

        config = {
            "script_version": (
                SCRIPT_VERSION
            ),
            "model": args.model,
            "repo_id": (
                spec.repo_id
            ),
            "dataset": (
                args.dataset
            ),
            "N_selected": len(
                selected
            ),
            "N_discovery": len(
                discovery_records
            ),
            "N_eval": len(
                eval_records
            ),
            "discovery_ratio": (
                args.discovery_ratio
            ),
            "seed": (
                args.seed
            ),
            "layer": (
                layer
            ),
            "query_heads": (
                n_query
            ),
            "kv_heads": (
                n_kv
            ),
            "rank_cohort_requested": (
                args.rank_cohort
            ),
            "rank_cohort_effective": (
                effective_rank_cohort
            ),
            "rank_metric": (
                rank_metric
            ),
            "overwrite_discovery_N": (
                overwrite_count
            ),
            "top_k": (
                top_k
            ),
            "frozen_ranked_heads": (
                ranked_heads[
                    :
                    top_k
                ]
            ),
            "ablation_scale": (
                args.ablation_scale
            ),
            "random_bundles_per_k": (
                args.random_bundles_per_k
            ),
            "final_norm_path": (
                final_norm_path
            ),
            "self_message": (
                "W_O^h * A_h[last,last] * V_h[last]"
            ),
            "discovery_metric": (
                "margin(r_attn-c_h_self)-margin(r_attn)"
            ),
            "causal_intervention": (
                "attention_out_L27[last] -= scale * "
                "sum_{h in bundle} clean c_h_self; "
                "downstream computation recomputed online"
            ),
            "split_integrity": (
                "head ranking uses discovery only; causal generation uses held-out eval only"
            ),
            "audit": (
                audit
            ),
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

        report = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"N discovery/eval: {len(discovery_records)}/{len(eval_records)}",
            f"discovery overwrite N: {overwrite_count}",
            f"rank cohort: {effective_rank_cohort}",
            f"rank metric: {rank_metric}",
            f"frozen top heads: {ranked_heads[:top_k]}",
            f"ablation scale: {args.ablation_scale}",
            f"held-out baseline ACC: {100*baseline_acc:.2f}%",
            "",
            "DISCOVERY TOP HEADS",
        ]

        for rank, row in enumerate(
            ranked_top,
            start=1,
        ):
            report.append(
                f"{rank:02d} {row['head_name']} KVH{int(row['kv_head'])} "
                f"rankImp={float(row[rank_metric]):+.5f} "
                f"rankPos={100*float(row[positive_metric]):.2f}% "
                f"allImp={float(row['mean_removal_improvement_all']):+.5f}"
            )

        report += [
            "",
            "HELD-OUT PREFIX CURVE",
        ]

        for row in prefix_rows:
            report.append(
                f"Top{int(row['K'])} [{row['heads']}]: "
                f"ACC={100*float(row['patched_acc']):.2f}% "
                f"delta={100*float(row['delta_acc']):+.2f}pp "
                f"W->C={int(row['wrong_to_correct'])} "
                f"C->W={int(row['correct_to_wrong'])} "
                f"net={int(row['net_repairs']):+d}"
            )

        report += [
            "",
            "INTERPRETATION",
            (
                "A monotonic/saturating positive TopK curve with weak matched-random "
                "bundles supports a distributed harmful L27 self-message family."
            ),
            (
                "If the complete Top1 KV group is harmful while the other KV groups "
                "are not, inspect its shared K/V channel next."
            ),
            (
                "If only some query heads inside the same KV group are harmful, "
                "the difference is more likely in query routing and/or their W_O "
                "projection than in shared V content alone."
            ),
            (
                "If held-out bundle effects vanish, treat the discovery logit-lens "
                "ranking as descriptive rather than a generation-causal mechanism."
            ),
        ]

        (
            output_dir
            / "report.txt"
        ).write_text(
            "\n".join(
                report
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "\nSaved:"
        )

        for filename in (
            "selected_sids.json",
            "discovery_sample.csv",
            "discovery_head_summary.csv",
            "discovery_ranked_harmful_self_heads.csv",
            "causal_conditions.json",
            "eval_baseline.csv",
            "causal_results.jsonl",
            "causal_summary.csv",
            "causal_summary_by_cohort.csv",
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
