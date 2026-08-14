#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-7B L23 ATTENTION / MLP GT-ORACLE SCALE SWEEP.

Purpose
=======
We observed a large correct-vs-wrong trajectory bifurcation at L23.

This script asks a deliberately oracle question:

    If GT is allowed ONLY to choose how strongly L23 attention / MLP should
    contribute, how much first-step and full-generation ACC is recoverable?

No GT answer vector is injected.

Natural L23 block:
    x23
      -> attention contribution a23
      -> r23 = x23 + a23
      -> MLP contribution m23
      -> y23 = r23 + m23

Interventions
=============
ATTENTION SCALE:
    a23' = alpha * a23

implemented by scaling ONLY the prompt-last output vector of the L23
self-attention module after o_proj during prefill. The L23 MLP and all
downstream layers then recompute naturally.

MLP SCALE:
    m23' = beta * m23

implemented by scaling ONLY the prompt-last output vector of the L23 MLP
module during prefill. Downstream layers recompute naturally.

SEQUENTIAL ATTENTION -> MLP:
    1. choose alpha* by GT final first-step decision margin;
    2. freeze alpha*;
    3. under that attention-scaled forward, sweep beta and choose beta*;
    4. run full generation once with alpha*, beta*.

GT-ORACLE SELECTION
===================
For each candidate scale, the script performs a complete forward pass through
the model and measures FINAL first-step 4-way relation logits:

    M_decision
      = logit(GT) - max_{r != GT} logit(r)

Oracle chooses the candidate with maximum M_decision.
Ties prefer a scale closest to natural scale 1.0.

This means:
    * GT is used to SELECT scale only.
    * GT is NOT written into hidden states.
    * all layers after L23 recompute normally.

Full generation
===============
After selecting the oracle scale(s), the script performs ONE full
model.generate() per selected intervention mode and reports actual generation
ACC, W->C, C->W, etc.

Modes reported
==============
baseline
attention_oracle_all
attention_oracle_wrong_only
mlp_oracle_all
mlp_oracle_wrong_only
sequential_oracle_all
sequential_oracle_wrong_only

"wrong_only" means:
    if the clean native 4-way first-step prediction is already GT,
    keep alpha=1 / beta=1;
    only clean first-step-WRONG samples receive oracle scaling.

This is an oracle upper-bound diagnostic, NOT a deployable method.

Default grids
=============
alpha:
    -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2

beta:
    -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2

Interpretation
==============
If attention oracle strongly improves generation and alpha* for wrong samples
is mostly 0~0.75:
    L23 attention is often too strong on failures; learn a GT-free dynamic gate.

If alpha* is often negative:
    natural attention direction itself is often harmful; scaling alone implies
    sign reversal may help, but a learned repair direction is probably cleaner.

If MLP oracle improves strongly:
    L23 MLP gain is a candidate failure source.

If sequential >> either alone:
    the failure is coupled: attention changes the state on which the MLP acts.

If even GT oracle gives little gain:
    scalar magnitude control is insufficient; next try low-dimensional
    direction repair rather than more scaling.

Example
=======
CUDA_VISIBLE_DEVICES=0 python -u \
oracle_sweep_qwen7b_l23_attention_mlp_scale_v1.py \
  --model qwen-7b \
  --num-samples 0 \
  --device cuda:0 \
  --output-dir output/qwen7b_l23_attn_mlp_oracle_scale_v1 \
  --overwrite

Quick:
CUDA_VISIBLE_DEVICES=0 python -u \
oracle_sweep_qwen7b_l23_attention_mlp_scale_v1.py \
  --model qwen-7b \
  --num-samples 200 \
  --device cuda:0 \
  --output-dir output/qwen7b_l23_attn_mlp_oracle_scale_n200_v1 \
  --overwrite

Custom grids with negative first value:
    --alpha-grid=-1,-0.5,0,0.5,1,1.5,2
    --beta-grid=-1,-0.5,0,0.5,1,1.5,2

Outputs
=======
selected_sids.json
baseline.csv
oracle_candidate_scores.csv
oracle_selected_scales.csv
oracle_scale_histogram.csv
generation_results.jsonl
generation_summary.csv
summary_by_clean_firststep.csv
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
import shutil
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


SCRIPT_VERSION = "qwen7b-l23-attention-mlp-oracle-scale-v1"

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

DEFAULT_SCALE_GRID = (
    -1.0,
    -0.5,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
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
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--layer", type=int, default=23)

    p.add_argument(
        "--alpha-grid",
        default=",".join(str(x) for x in DEFAULT_SCALE_GRID),
    )
    p.add_argument(
        "--beta-grid",
        default=",".join(str(x) for x in DEFAULT_SCALE_GRID),
    )

    p.add_argument(
        "--run-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--run-mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--run-sequential",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager",),
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)

    p.add_argument(
        "--helper-module",
        default="analyze_qwen7b_l26_l27_attention_overwrite_v1",
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


def parse_float_grid(text: str) -> List[float]:
    values: List[float] = []

    for item in str(text).split(","):
        item = item.strip()

        if not item:
            continue

        values.append(
            float(
                item
            )
        )

    if not values:
        raise ValueError(
            "Empty scale grid."
        )

    # Deduplicate while preserving order.
    output: List[float] = []
    seen = set()

    for value in values:
        key = float(
            value
        )

        if key not in seen:
            output.append(
                key
            )
            seen.add(
                key
            )

    if 1.0 not in seen:
        output.append(
            1.0
        )

    return output


def safe_mean(values: Iterable[Any]) -> float:
    output: List[float] = []

    for value in values:
        try:
            x = float(
                value
            )
        except Exception:
            continue

        if math.isfinite(
            x
        ):
            output.append(
                x
            )

    return (
        float(
            np.mean(
                output
            )
        )
        if output
        else float(
            "nan"
        )
    )


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
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
                fields.append(
                    key
                )
                seen.add(
                    key
                )

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
                dict(
                    row
                )
            )


def append_jsonl(
    path: Path,
    row: Mapping[
        str,
        Any,
    ],
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
                dict(
                    row
                ),
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


# =============================================================================
# Module output scaling hooks
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
        "Could not find a 3D hidden-state tensor in module output."
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
        "Could not replace 3D hidden-state tensor."
    )


class PromptLastScaleHook:
    """
    Multiply one module's prompt-last output vector by scale on the FULL
    multimodal prefill only.

    For attention module:
        attention contribution a[last] -> alpha * a[last]

    For MLP module:
        MLP contribution m[last] -> beta * m[last]
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        prompt_length: int,
        prompt_last: int,
        scale: float,
        label: str,
    ) -> None:
        self.module = module
        self.prompt_length = int(
            prompt_length
        )
        self.prompt_last = int(
            prompt_last
        )
        self.scale = float(
            scale
        )
        self.label = str(
            label
        )

        self.handle = None
        self.applications = 0

    def __enter__(
        self,
    ) -> "PromptLastScaleHook":
        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> Any:
            hidden = first_3d(
                output
            )

            # Apply only on full prefill, not autoregressive cached q_len=1.
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
                    f"{self.label}: prompt_last={self.prompt_last}, "
                    f"q_len={hidden.shape[1]}."
                )

            modified = hidden.clone()

            modified[
                0,
                self.prompt_last,
            ] = (
                modified[
                    0,
                    self.prompt_last,
                ]
                * self.scale
            )

            self.applications += 1

            return replace_first_3d(
                output,
                modified,
            )

        self.handle = (
            self.module.register_forward_hook(
                hook
            )
        )

        return self

    def validate(
        self,
        *,
        expected: int = 1,
    ) -> None:
        if (
            self.applications
            != int(
                expected
            )
        ):
            raise RuntimeError(
                f"{self.label}: expected {expected} prefill patch application(s), "
                f"got {self.applications}."
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


# =============================================================================
# Relation scoring from ACTUAL final model logits
# =============================================================================

def build_relation_token_map(
    helper: Any,
    tokenizer: Any,
) -> Dict[
    str,
    List[int],
]:
    return helper.relation_token_variants(
        tokenizer
    )


def relation_scores_from_vocab_logits(
    vocab_logits: torch.Tensor,
    relation_token_map: Mapping[
        str,
        Sequence[int],
    ],
) -> np.ndarray:
    scores: List[float] = []

    for relation in RELATIONS:
        ids = torch.as_tensor(
            list(
                map(
                    int,
                    relation_token_map[
                        relation
                    ],
                )
            ),
            device=vocab_logits.device,
            dtype=torch.long,
        )

        values = vocab_logits.index_select(
            0,
            ids,
        )

        scores.append(
            float(
                values.max().item()
            )
        )

    return np.asarray(
        scores,
        dtype=np.float32,
    )


def relation_metrics(
    scores: np.ndarray,
    gt: str,
) -> Dict[
    str,
    Any,
]:
    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    gt_id = RID[
        gt
    ]

    wrong_ids = [
        index
        for index in range(
            len(
                RELATIONS
            )
        )
        if index
        != gt_id
    ]

    competitor_id = max(
        wrong_ids,
        key=lambda index: float(
            values[
                index
            ]
        ),
    )

    prediction_id = int(
        np.argmax(
            values
        )
    )

    shifted = (
        values
        - np.max(
            values
        )
    )

    probabilities = np.exp(
        shifted
    )

    probabilities = (
        probabilities
        / max(
            float(
                probabilities.sum()
            ),
            EPS,
        )
    )

    return {
        "prediction": RELATIONS[
            prediction_id
        ],
        "correct": (
            prediction_id
            == gt_id
        ),
        "decision_margin": float(
            values[
                gt_id
            ]
            - values[
                competitor_id
            ]
        ),
        "opposite_margin": float(
            values[
                gt_id
            ]
            - values[
                RID[
                    OPPOSITE[
                        gt
                    ]
                ]
            ]
        ),
        "p_gt_4way": float(
            probabilities[
                gt_id
            ]
        ),
        "top_competitor": RELATIONS[
            competitor_id
        ],
    }


@torch.inference_mode()
def forward_firststep_metrics(
    *,
    model: Any,
    batch: Any,
    relation_token_map: Mapping[
        str,
        Sequence[int],
    ],
    gt: str,
    attention_module: Optional[
        torch.nn.Module
    ] = None,
    alpha: float = 1.0,
    mlp_module: Optional[
        torch.nn.Module
    ] = None,
    beta: float = 1.0,
) -> Dict[
    str,
    Any,
]:
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

    managers: List[
        PromptLastScaleHook
    ] = []

    if (
        attention_module
        is not None
        and not math.isclose(
            float(
                alpha
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        managers.append(
            PromptLastScaleHook(
                module=attention_module,
                prompt_length=prompt_length,
                prompt_last=prompt_last,
                scale=float(
                    alpha
                ),
                label="attention",
            )
        )

    if (
        mlp_module
        is not None
        and not math.isclose(
            float(
                beta
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        managers.append(
            PromptLastScaleHook(
                module=mlp_module,
                prompt_length=prompt_length,
                prompt_last=prompt_last,
                scale=float(
                    beta
                ),
                label="mlp",
            )
        )

    with contextlib.ExitStack() as stack:
        active = [
            stack.enter_context(
                manager
            )
            for manager in managers
        ]

        outputs = model(
            **batch,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits

        if (
            logits.ndim != 3
            or logits.shape[
                0
            ] != 1
        ):
            raise RuntimeError(
                f"Unexpected logits shape: {tuple(logits.shape)}"
            )

        vocab_logits = logits[
            0,
            -1,
        ]

        relation_scores = (
            relation_scores_from_vocab_logits(
                vocab_logits,
                relation_token_map,
            )
        )

        metrics = relation_metrics(
            relation_scores,
            gt,
        )

        for manager in active:
            manager.validate(
                expected=1
            )

        del outputs

    metrics.update({
        "alpha": float(
            alpha
        ),
        "beta": float(
            beta
        ),
    })

    return metrics


@torch.inference_mode()
def greedy_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    helper: Any,
    max_new_tokens: int,
    attention_module: Optional[
        torch.nn.Module
    ] = None,
    alpha: float = 1.0,
    mlp_module: Optional[
        torch.nn.Module
    ] = None,
    beta: float = 1.0,
) -> Tuple[
    Optional[str],
    str,
]:
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

    managers: List[
        PromptLastScaleHook
    ] = []

    if (
        attention_module
        is not None
        and not math.isclose(
            float(
                alpha
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        managers.append(
            PromptLastScaleHook(
                module=attention_module,
                prompt_length=prompt_length,
                prompt_last=prompt_last,
                scale=float(
                    alpha
                ),
                label="attention-generation",
            )
        )

    if (
        mlp_module
        is not None
        and not math.isclose(
            float(
                beta
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        managers.append(
            PromptLastScaleHook(
                module=mlp_module,
                prompt_length=prompt_length,
                prompt_last=prompt_last,
                scale=float(
                    beta
                ),
                label="mlp-generation",
            )
        )

    with contextlib.ExitStack() as stack:
        active = [
            stack.enter_context(
                manager
            )
            for manager in managers
        ]

        generated = model.generate(
            **batch,
            max_new_tokens=int(
                max_new_tokens
            ),
            do_sample=False,
            use_cache=True,
        )

        for manager in active:
            manager.validate(
                expected=1
            )

    text = (
        processor.tokenizer.decode(
            generated[
                0,
                prompt_length:,
            ],
            skip_special_tokens=True,
        )
        .strip()
    )

    prediction = helper.normalize_relation(
        text
    )

    del generated

    return (
        prediction,
        text,
    )


# =============================================================================
# Oracle selection
# =============================================================================

def oracle_sort_key(
    row: Mapping[
        str,
        Any,
    ],
    *,
    scale_field: str,
) -> Tuple[
    float,
    float,
    float,
]:
    """
    Maximize:
        decision margin
        then p_GT
    Minimize:
        |scale - 1|
    """
    scale = float(
        row[
            scale_field
        ]
    )

    return (
        float(
            row[
                "decision_margin"
            ]
        ),
        float(
            row[
                "p_gt_4way"
            ]
        ),
        -abs(
            scale
            - 1.0
        ),
    )


def choose_best_single_scale(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    scale_field: str,
) -> Mapping[
    str,
    Any,
]:
    if not rows:
        raise RuntimeError(
            "No oracle candidates."
        )

    return max(
        rows,
        key=lambda row: oracle_sort_key(
            row,
            scale_field=scale_field,
        ),
    )


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation_modes(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> List[
    Dict[
        str,
        Any,
    ]
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
                    "mode"
                ]
            )
        ].append(
            row
        )

    output: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for mode, current in grouped.items():
        baseline_acc = safe_mean(
            float(
                row[
                    "baseline_generation_correct"
                ]
            )
            for row in current
        )

        patched_acc = safe_mean(
            float(
                row[
                    "patched_generation_correct"
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

        firststep_acc = safe_mean(
            float(
                row[
                    "selected_firststep_correct"
                ]
            )
            for row in current
        )

        output.append({
            "mode": mode,
            "N": len(
                current
            ),
            "baseline_generation_acc": baseline_acc,
            "patched_generation_acc": patched_acc,
            "generation_delta_acc": (
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
            "selected_firststep_acc": firststep_acc,
            "mean_selected_alpha": safe_mean(
                row[
                    "selected_alpha"
                ]
                for row in current
            ),
            "mean_selected_beta": safe_mean(
                row[
                    "selected_beta"
                ]
                for row in current
            ),
        })

    mode_order = {
        "attention_oracle_all": 0,
        "attention_oracle_wrong_only": 1,
        "mlp_oracle_all": 2,
        "mlp_oracle_wrong_only": 3,
        "sequential_oracle_all": 4,
        "sequential_oracle_wrong_only": 5,
    }

    output.sort(
        key=lambda row: mode_order.get(
            str(
                row[
                    "mode"
                ]
            ),
            99,
        )
    )

    return output


def build_scale_histogram(
    selected_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    grouped: Dict[
        Tuple[
            str,
            str,
            float,
        ],
        int,
    ] = Counter()

    totals: Counter = Counter()

    for row in selected_rows:
        mode = str(
            row[
                "oracle_mode"
            ]
        )
        clean_group = (
            "clean_firststep_correct"
            if bool(
                row[
                    "clean_firststep_correct"
                ]
            )
            else "clean_firststep_wrong"
        )

        if mode in (
            "attention",
            "sequential",
        ):
            alpha = float(
                row[
                    "selected_alpha"
                ]
            )

            grouped[
                (
                    mode
                    + "_alpha",
                    clean_group,
                    alpha,
                )
            ] += 1

            totals[
                (
                    mode
                    + "_alpha",
                    clean_group,
                )
            ] += 1

        if mode in (
            "mlp",
            "sequential",
        ):
            beta = float(
                row[
                    "selected_beta"
                ]
            )

            grouped[
                (
                    mode
                    + "_beta",
                    clean_group,
                    beta,
                )
            ] += 1

            totals[
                (
                    mode
                    + "_beta",
                    clean_group,
                )
            ] += 1

    output: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for (
        parameter,
        clean_group,
        scale,
    ), count in sorted(
        grouped.items(),
        key=lambda item: (
            item[
                0
            ][
                0
            ],
            item[
                0
            ][
                1
            ],
            item[
                0
            ][
                2
            ],
        ),
    ):
        total = totals[
            (
                parameter,
                clean_group,
            )
        ]

        output.append({
            "parameter": parameter,
            "clean_group": clean_group,
            "scale": scale,
            "N": count,
            "fraction": (
                count
                / max(
                    total,
                    1,
                )
            ),
        })

    return output


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

    alpha_grid = parse_float_grid(
        args.alpha_grid
    )
    beta_grid = parse_float_grid(
        args.beta_grid
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

    helper = importlib.import_module(
        args.helper_module
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

    selected_records = (
        helper.select_records(
            records,
            n=args.num_samples,
            seed=args.seed,
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
                    selected_records
                ),
                "sids": [
                    int(
                        record.sid
                    )
                    for record in selected_records
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

    model = None
    processor = None

    try:
        print(
            f"Loading {args.model}: {spec.repo_id}",
            flush=True,
        )

        model = model_class.from_pretrained(
            spec.repo_id,
            dtype=base.resolve_dtype(
                spec.dtype_name
            ),
            low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,
            device_map={
                "": args.device
            },
            attn_implementation=args.attn_impl,
        )

        model.eval()
        helper.clear_sampling_defaults(
            model
        )

        processor = (
            AutoProcessor.from_pretrained(
                spec.repo_id,
                trust_remote_code=spec.trust_remote_code,
            )
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
                f"Expected Qwen-7B 28 decoder layers; got "
                f"{len(decoder_layers)} at {decoder_path}."
            )

        layer = int(
            args.layer
        )

        if not (
            0
            <= layer
            < len(
                decoder_layers
            )
        ):
            raise ValueError(
                f"Bad layer L{layer}."
            )

        if layer != 23:
            print(
                f"[warning] experiment motivated by L23 but running L{layer}.",
                flush=True,
            )

        block = decoder_layers[
            layer
        ]

        attention_module = (
            attention_helper.resolve_self_attention(
                block
            )
        )

        mlp_module = getattr(
            block,
            "mlp",
            None,
        )

        if not isinstance(
            mlp_module,
            torch.nn.Module,
        ):
            raise RuntimeError(
                f"Could not resolve L{layer} MLP module via block.mlp."
            )

        relation_token_map = (
            build_relation_token_map(
                helper,
                processor.tokenizer,
            )
        )

        print(
            "\n"
            + "=" * 190
        )
        print(
            "QWEN-7B L23 ATTENTION / MLP GT-ORACLE SCALE SWEEP"
        )
        print(
            "=" * 190
        )
        print(
            "N               :",
            len(
                selected_records
            ),
        )
        print(
            "layer           :",
            layer,
        )
        print(
            "alpha grid      :",
            alpha_grid,
        )
        print(
            "beta grid       :",
            beta_grid,
        )
        print(
            "attention sweep :",
            args.run_attention,
        )
        print(
            "MLP sweep       :",
            args.run_mlp,
        )
        print(
            "sequential      :",
            args.run_sequential,
        )
        print(
            "=" * 190,
            flush=True,
        )

        baseline_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        candidate_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        selected_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        generation_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        for sample_index, record in enumerate(
            tqdm(
                selected_records,
                desc="L23-oracle-scale",
            ),
            start=1,
        ):
            image = None
            batch = None

            try:
                sid = int(
                    record.sid
                )

                gt = helper.normalize_relation(
                    record.relation
                )

                if gt not in RELATIONS:
                    raise RuntimeError(
                        f"Unsupported GT: {record.relation!r}"
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

                batch = helper.build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                # -------------------------------------------------------------
                # Clean actual final first-step + generation.
                # -------------------------------------------------------------

                clean_first = (
                    forward_firststep_metrics(
                        model=model,
                        batch=batch,
                        relation_token_map=relation_token_map,
                        gt=gt,
                        attention_module=attention_module,
                        alpha=1.0,
                        mlp_module=mlp_module,
                        beta=1.0,
                    )
                )

                (
                    clean_generation_pred,
                    clean_generation_text,
                ) = greedy_generation(
                    model=model,
                    processor=processor,
                    batch=batch,
                    helper=helper,
                    max_new_tokens=args.max_new_tokens,
                    attention_module=attention_module,
                    alpha=1.0,
                    mlp_module=mlp_module,
                    beta=1.0,
                )

                clean_generation_correct = (
                    clean_generation_pred
                    == gt
                )

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "clean_firststep_pred": clean_first[
                        "prediction"
                    ],
                    "clean_firststep_correct": clean_first[
                        "correct"
                    ],
                    "clean_firststep_decision_margin": clean_first[
                        "decision_margin"
                    ],
                    "clean_firststep_opposite_margin": clean_first[
                        "opposite_margin"
                    ],
                    "clean_firststep_p_gt_4way": clean_first[
                        "p_gt_4way"
                    ],
                    "clean_generation_pred": clean_generation_pred,
                    "clean_generation_text": clean_generation_text,
                    "clean_generation_correct": clean_generation_correct,
                })

                # -------------------------------------------------------------
                # Attention alpha oracle.
                # -------------------------------------------------------------

                attention_best = {
                    **clean_first,
                    "alpha": 1.0,
                }

                if (
                    args.run_attention
                    or args.run_sequential
                ):
                    current_candidates = []

                    for alpha in alpha_grid:
                        metrics = (
                            forward_firststep_metrics(
                                model=model,
                                batch=batch,
                                relation_token_map=relation_token_map,
                                gt=gt,
                                attention_module=attention_module,
                                alpha=alpha,
                                mlp_module=mlp_module,
                                beta=1.0,
                            )
                        )

                        row = {
                            "sid": sid,
                            "gt": gt,
                            "sweep": "attention",
                            "alpha": float(
                                alpha
                            ),
                            "beta": 1.0,
                            "prediction": metrics[
                                "prediction"
                            ],
                            "correct": metrics[
                                "correct"
                            ],
                            "decision_margin": metrics[
                                "decision_margin"
                            ],
                            "opposite_margin": metrics[
                                "opposite_margin"
                            ],
                            "p_gt_4way": metrics[
                                "p_gt_4way"
                            ],
                        }

                        candidate_rows.append(
                            row
                        )
                        current_candidates.append(
                            row
                        )

                    attention_best = dict(
                        choose_best_single_scale(
                            current_candidates,
                            scale_field="alpha",
                        )
                    )

                    selected_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "oracle_mode": "attention",
                        "clean_firststep_correct": clean_first[
                            "correct"
                        ],
                        "selected_alpha": float(
                            attention_best[
                                "alpha"
                            ]
                        ),
                        "selected_beta": 1.0,
                        "selected_firststep_pred": attention_best[
                            "prediction"
                        ],
                        "selected_firststep_correct": attention_best[
                            "correct"
                        ],
                        "selected_decision_margin": attention_best[
                            "decision_margin"
                        ],
                        "clean_decision_margin": clean_first[
                            "decision_margin"
                        ],
                        "decision_margin_gain": (
                            float(
                                attention_best[
                                    "decision_margin"
                                ]
                            )
                            - float(
                                clean_first[
                                    "decision_margin"
                                ]
                            )
                        ),
                        "any_grid_firststep_correct": any(
                            bool(
                                row[
                                    "correct"
                                ]
                            )
                            for row in current_candidates
                        ),
                    })

                # -------------------------------------------------------------
                # MLP beta oracle at natural attention alpha=1.
                # -------------------------------------------------------------

                mlp_best = {
                    **clean_first,
                    "beta": 1.0,
                }

                if args.run_mlp:
                    current_candidates = []

                    for beta in beta_grid:
                        metrics = (
                            forward_firststep_metrics(
                                model=model,
                                batch=batch,
                                relation_token_map=relation_token_map,
                                gt=gt,
                                attention_module=attention_module,
                                alpha=1.0,
                                mlp_module=mlp_module,
                                beta=beta,
                            )
                        )

                        row = {
                            "sid": sid,
                            "gt": gt,
                            "sweep": "mlp",
                            "alpha": 1.0,
                            "beta": float(
                                beta
                            ),
                            "prediction": metrics[
                                "prediction"
                            ],
                            "correct": metrics[
                                "correct"
                            ],
                            "decision_margin": metrics[
                                "decision_margin"
                            ],
                            "opposite_margin": metrics[
                                "opposite_margin"
                            ],
                            "p_gt_4way": metrics[
                                "p_gt_4way"
                            ],
                        }

                        candidate_rows.append(
                            row
                        )
                        current_candidates.append(
                            row
                        )

                    mlp_best = dict(
                        choose_best_single_scale(
                            current_candidates,
                            scale_field="beta",
                        )
                    )

                    selected_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "oracle_mode": "mlp",
                        "clean_firststep_correct": clean_first[
                            "correct"
                        ],
                        "selected_alpha": 1.0,
                        "selected_beta": float(
                            mlp_best[
                                "beta"
                            ]
                        ),
                        "selected_firststep_pred": mlp_best[
                            "prediction"
                        ],
                        "selected_firststep_correct": mlp_best[
                            "correct"
                        ],
                        "selected_decision_margin": mlp_best[
                            "decision_margin"
                        ],
                        "clean_decision_margin": clean_first[
                            "decision_margin"
                        ],
                        "decision_margin_gain": (
                            float(
                                mlp_best[
                                    "decision_margin"
                                ]
                            )
                            - float(
                                clean_first[
                                    "decision_margin"
                                ]
                            )
                        ),
                        "any_grid_firststep_correct": any(
                            bool(
                                row[
                                    "correct"
                                ]
                            )
                            for row in current_candidates
                        ),
                    })

                # -------------------------------------------------------------
                # Sequential oracle: alpha* first, then beta sweep under alpha*.
                # -------------------------------------------------------------

                sequential_best = None

                if args.run_sequential:
                    chosen_alpha = float(
                        attention_best[
                            "alpha"
                        ]
                    )

                    sequential_candidates = []

                    for beta in beta_grid:
                        metrics = (
                            forward_firststep_metrics(
                                model=model,
                                batch=batch,
                                relation_token_map=relation_token_map,
                                gt=gt,
                                attention_module=attention_module,
                                alpha=chosen_alpha,
                                mlp_module=mlp_module,
                                beta=beta,
                            )
                        )

                        row = {
                            "sid": sid,
                            "gt": gt,
                            "sweep": "sequential_beta_after_alpha",
                            "alpha": chosen_alpha,
                            "beta": float(
                                beta
                            ),
                            "prediction": metrics[
                                "prediction"
                            ],
                            "correct": metrics[
                                "correct"
                            ],
                            "decision_margin": metrics[
                                "decision_margin"
                            ],
                            "opposite_margin": metrics[
                                "opposite_margin"
                            ],
                            "p_gt_4way": metrics[
                                "p_gt_4way"
                            ],
                        }

                        candidate_rows.append(
                            row
                        )
                        sequential_candidates.append(
                            row
                        )

                    sequential_best = dict(
                        choose_best_single_scale(
                            sequential_candidates,
                            scale_field="beta",
                        )
                    )

                    selected_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "oracle_mode": "sequential",
                        "clean_firststep_correct": clean_first[
                            "correct"
                        ],
                        "selected_alpha": chosen_alpha,
                        "selected_beta": float(
                            sequential_best[
                                "beta"
                            ]
                        ),
                        "selected_firststep_pred": sequential_best[
                            "prediction"
                        ],
                        "selected_firststep_correct": sequential_best[
                            "correct"
                        ],
                        "selected_decision_margin": sequential_best[
                            "decision_margin"
                        ],
                        "clean_decision_margin": clean_first[
                            "decision_margin"
                        ],
                        "decision_margin_gain": (
                            float(
                                sequential_best[
                                    "decision_margin"
                                ]
                            )
                            - float(
                                clean_first[
                                    "decision_margin"
                                ]
                            )
                        ),
                        "any_grid_firststep_correct": any(
                            bool(
                                row[
                                    "correct"
                                ]
                            )
                            for row in sequential_candidates
                        ),
                    })

                # -------------------------------------------------------------
                # Full generation under oracle-selected scales.
                # -------------------------------------------------------------

                generation_modes = []

                if args.run_attention:
                    generation_modes += [
                        (
                            "attention_oracle_all",
                            float(
                                attention_best[
                                    "alpha"
                                ]
                            ),
                            1.0,
                        ),
                        (
                            "attention_oracle_wrong_only",
                            (
                                1.0
                                if clean_first[
                                    "correct"
                                ]
                                else float(
                                    attention_best[
                                        "alpha"
                                    ]
                                )
                            ),
                            1.0,
                        ),
                    ]

                if args.run_mlp:
                    generation_modes += [
                        (
                            "mlp_oracle_all",
                            1.0,
                            float(
                                mlp_best[
                                    "beta"
                                ]
                            ),
                        ),
                        (
                            "mlp_oracle_wrong_only",
                            1.0,
                            (
                                1.0
                                if clean_first[
                                    "correct"
                                ]
                                else float(
                                    mlp_best[
                                        "beta"
                                    ]
                                )
                            ),
                        ),
                    ]

                if args.run_sequential:
                    sequential_alpha = float(
                        sequential_best[
                            "alpha"
                        ]
                    )
                    sequential_beta = float(
                        sequential_best[
                            "beta"
                        ]
                    )

                    generation_modes += [
                        (
                            "sequential_oracle_all",
                            sequential_alpha,
                            sequential_beta,
                        ),
                        (
                            "sequential_oracle_wrong_only",
                            (
                                1.0
                                if clean_first[
                                    "correct"
                                ]
                                else sequential_alpha
                            ),
                            (
                                1.0
                                if clean_first[
                                    "correct"
                                ]
                                else sequential_beta
                            ),
                        ),
                    ]

                # Deduplicate exact mode alpha/beta? Keep modes separate because
                # summary semantics differ.
                for (
                    mode,
                    selected_alpha,
                    selected_beta,
                ) in generation_modes:
                    (
                        patched_pred,
                        patched_text,
                    ) = greedy_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        helper=helper,
                        max_new_tokens=args.max_new_tokens,
                        attention_module=attention_module,
                        alpha=selected_alpha,
                        mlp_module=mlp_module,
                        beta=selected_beta,
                    )

                    patched_correct = (
                        patched_pred
                        == gt
                    )

                    # Recompute final first-step metrics for the exact selected
                    # wrong-only/all configuration. This is cheap relative to
                    # generation and avoids assuming it equals the raw oracle row.
                    selected_first = (
                        forward_firststep_metrics(
                            model=model,
                            batch=batch,
                            relation_token_map=relation_token_map,
                            gt=gt,
                            attention_module=attention_module,
                            alpha=selected_alpha,
                            mlp_module=mlp_module,
                            beta=selected_beta,
                        )
                    )

                    row = {
                        "sid": sid,
                        "gt": gt,
                        "mode": mode,
                        "clean_firststep_pred": clean_first[
                            "prediction"
                        ],
                        "clean_firststep_correct": clean_first[
                            "correct"
                        ],
                        "baseline_generation_pred": clean_generation_pred,
                        "baseline_generation_correct": clean_generation_correct,
                        "selected_alpha": selected_alpha,
                        "selected_beta": selected_beta,
                        "selected_firststep_pred": selected_first[
                            "prediction"
                        ],
                        "selected_firststep_correct": selected_first[
                            "correct"
                        ],
                        "selected_firststep_decision_margin": selected_first[
                            "decision_margin"
                        ],
                        "patched_generation_pred": patched_pred,
                        "patched_generation_text": patched_text,
                        "patched_generation_correct": patched_correct,
                        "wrong_to_correct": (
                            (
                                not clean_generation_correct
                            )
                            and patched_correct
                        ),
                        "correct_to_wrong": (
                            clean_generation_correct
                            and (
                                not patched_correct
                            )
                        ),
                        "generation_changed": (
                            patched_pred
                            != clean_generation_pred
                        ),
                    }

                    generation_rows.append(
                        row
                    )

                    append_jsonl(
                        output_dir
                        / "generation_results.jsonl",
                        row,
                    )

            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "phase": "sample",
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

        # =====================================================================
        # Save raw tables
        # =====================================================================

        write_csv(
            output_dir
            / "baseline.csv",
            baseline_rows,
        )

        write_csv(
            output_dir
            / "oracle_candidate_scores.csv",
            candidate_rows,
        )

        write_csv(
            output_dir
            / "oracle_selected_scales.csv",
            selected_rows,
        )

        scale_histogram = (
            build_scale_histogram(
                selected_rows
            )
        )

        write_csv(
            output_dir
            / "oracle_scale_histogram.csv",
            scale_histogram,
        )

        generation_summary = (
            summarize_generation_modes(
                generation_rows
            )
        )

        write_csv(
            output_dir
            / "generation_summary.csv",
            generation_summary,
        )

        # =====================================================================
        # Secondary summary by clean first-step correctness
        # =====================================================================

        grouped_secondary: Dict[
            Tuple[
                str,
                str,
            ],
            List[
                Mapping[
                    str,
                    Any,
                ]
            ],
        ] = defaultdict(list)

        for row in generation_rows:
            clean_group = (
                "clean_firststep_correct"
                if bool(
                    row[
                        "clean_firststep_correct"
                    ]
                )
                else "clean_firststep_wrong"
            )

            grouped_secondary[
                (
                    str(
                        row[
                            "mode"
                        ]
                    ),
                    clean_group,
                )
            ].append(
                row
            )

        secondary_rows: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        for (
            mode,
            clean_group,
        ), rows in grouped_secondary.items():
            secondary_rows.append({
                "mode": mode,
                "clean_group": clean_group,
                "N": len(
                    rows
                ),
                "baseline_generation_acc": safe_mean(
                    float(
                        row[
                            "baseline_generation_correct"
                        ]
                    )
                    for row in rows
                ),
                "patched_generation_acc": safe_mean(
                    float(
                        row[
                            "patched_generation_correct"
                        ]
                    )
                    for row in rows
                ),
                "selected_firststep_acc": safe_mean(
                    float(
                        row[
                            "selected_firststep_correct"
                        ]
                    )
                    for row in rows
                ),
                "wrong_to_correct": sum(
                    bool(
                        row[
                            "wrong_to_correct"
                        ]
                    )
                    for row in rows
                ),
                "correct_to_wrong": sum(
                    bool(
                        row[
                            "correct_to_wrong"
                        ]
                    )
                    for row in rows
                ),
                "mean_alpha": safe_mean(
                    row[
                        "selected_alpha"
                    ]
                    for row in rows
                ),
                "mean_beta": safe_mean(
                    row[
                        "selected_beta"
                    ]
                    for row in rows
                ),
            })

        secondary_rows.sort(
            key=lambda row: (
                str(
                    row[
                        "mode"
                    ]
                ),
                str(
                    row[
                        "clean_group"
                    ]
                ),
            )
        )

        write_csv(
            output_dir
            / "summary_by_clean_firststep.csv",
            secondary_rows,
        )

        # =====================================================================
        # Console
        # =====================================================================

        baseline_firststep_acc = safe_mean(
            float(
                row[
                    "clean_firststep_correct"
                ]
            )
            for row in baseline_rows
        )

        baseline_generation_acc = safe_mean(
            float(
                row[
                    "clean_generation_correct"
                ]
            )
            for row in baseline_rows
        )

        print(
            "\n"
            + "=" * 190
        )
        print(
            "L23 ATTENTION / MLP GT-ORACLE SCALE RESULTS"
        )
        print(
            "=" * 190
        )
        print(
            f"N={len(baseline_rows)} | "
            f"baseline first-step ACC={100*baseline_firststep_acc:.2f}% | "
            f"baseline generation ACC={100*baseline_generation_acc:.2f}%"
        )

        print(
            f"\n  {'mode':<31s} {'1stACC':>8s} {'genACC':>8s} "
            f"{'delta':>9s} {'W->C':>5s} {'C->W':>5s} {'net':>5s} "
            f"{'alpha':>8s} {'beta':>8s}"
        )

        for row in generation_summary:
            print(
                f"  {str(row['mode']):<31s} "
                f"{100*float(row['selected_firststep_acc']):7.2f}% "
                f"{100*float(row['patched_generation_acc']):7.2f}% "
                f"{100*float(row['generation_delta_acc']):+8.2f} "
                f"{int(row['wrong_to_correct']):5d} "
                f"{int(row['correct_to_wrong']):5d} "
                f"{int(row['net_repairs']):+5d} "
                f"{float(row['mean_selected_alpha']):8.3f} "
                f"{float(row['mean_selected_beta']):8.3f}"
            )

        print(
            "\nORACLE SCALE DISTRIBUTION ON CLEAN FIRST-STEP WRONG SAMPLES"
        )

        wrong_hist = [
            row
            for row in scale_histogram
            if row[
                "clean_group"
            ]
            == "clean_firststep_wrong"
        ]

        histogram_groups: Dict[
            str,
            List[
                Mapping[
                    str,
                    Any,
                ]
            ],
        ] = defaultdict(list)

        for row in wrong_hist:
            histogram_groups[
                str(
                    row[
                        "parameter"
                    ]
                )
            ].append(
                row
            )

        for parameter, rows in histogram_groups.items():
            rows = sorted(
                rows,
                key=lambda row: float(
                    row[
                        "scale"
                    ]
                ),
            )

            text = " ".join(
                f"{float(row['scale']):g}:{int(row['N'])}"
                for row in rows
            )

            print(
                f"  {parameter:<22s} {text}"
            )

        print(
            "=" * 190
        )

        # =====================================================================
        # Config / report
        # =====================================================================

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "N": len(
                baseline_rows
            ),
            "seed": args.seed,
            "layer": layer,
            "alpha_grid": alpha_grid,
            "beta_grid": beta_grid,
            "run_attention": args.run_attention,
            "run_mlp": args.run_mlp,
            "run_sequential": args.run_sequential,
            "selection_criterion": (
                "GT final first-step decision margin = "
                "logit(GT)-max(other 3 relation logits)"
            ),
            "attention_intervention": (
                "scale L23 self-attention module output at prompt-last after o_proj "
                "during full prefill only"
            ),
            "mlp_intervention": (
                "scale L23 MLP module output at prompt-last during full prefill only"
            ),
            "sequential_intervention": (
                "choose alpha first; then sweep beta with alpha fixed; downstream recomputes"
            ),
            "baseline_firststep_acc": baseline_firststep_acc,
            "baseline_generation_acc": baseline_generation_acc,
            "oracle_warning": (
                "GT selects sample-specific scales; results are an upper-bound diagnostic, "
                "not a deployable GT-free repair."
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

        report = [
            f"script_version: {SCRIPT_VERSION}",
            f"N={len(baseline_rows)}",
            f"baseline first-step ACC={100*baseline_firststep_acc:.2f}%",
            f"baseline generation ACC={100*baseline_generation_acc:.2f}%",
            f"alpha_grid={alpha_grid}",
            f"beta_grid={beta_grid}",
            "",
            "GENERATION SUMMARY",
        ]

        for row in generation_summary:
            report.append(
                f"{row['mode']}: "
                f"firststep={100*float(row['selected_firststep_acc']):.2f}% "
                f"generation={100*float(row['patched_generation_acc']):.2f}% "
                f"delta={100*float(row['generation_delta_acc']):+.2f}pp "
                f"W->C={int(row['wrong_to_correct'])} "
                f"C->W={int(row['correct_to_wrong'])} "
                f"net={int(row['net_repairs']):+d} "
                f"mean_alpha={float(row['mean_selected_alpha']):.3f} "
                f"mean_beta={float(row['mean_selected_beta']):.3f}"
            )

        report += [
            "",
            "INTERPRETATION",
            (
                "Large attention-oracle gain with wrong-sample alpha mostly below 1 "
                "supports an L23 attention over-strength/gating failure."
            ),
            (
                "Large MLP-oracle gain supports an L23 MLP gain-control failure."
            ),
            (
                "Sequential substantially above both singles supports coupled "
                "attention-to-MLP miscalibration."
            ),
            (
                "Frequent negative selected scales indicates scalar sign reversal can "
                "help, suggesting direction rather than magnitude may be wrong."
            ),
            (
                "Little GT-oracle gain means scalar scaling is not the right repair; "
                "move to low-dimensional direction repair."
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
            "baseline.csv",
            "oracle_candidate_scores.csv",
            "oracle_selected_scales.csv",
            "oracle_scale_histogram.csv",
            "generation_results.jsonl",
            "generation_summary.csv",
            "summary_by_clean_firststep.csv",
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
