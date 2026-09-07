#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test whether the relation directions learned from the successfully enhanced
Controlled-A state C can improve the unenhanced baseline B.

B:
    eps=1e-6, weight=1.0, no AdaptVis.

Direction source:
    C-correct TRAIN samples from the saved v5 hidden-state cache.

For each selected layer:
    center = mean(h_C | C-correct TRAIN)
    v_r = mean(h_C | relation=r, C-correct TRAIN) - center

TEST intervention (oracle relation selection):
    h_B[last] <- h_B[last] + scale * v_r

This isolates cross-state causal transfer:
    enhanced C relation geometry -> unenhanced B residual stream.

Default:
    L18, scale=1.0, relation-stratified 30/70 TRAIN/TEST.

Example:
CUDA_VISIBLE_DEVICES=0 python eval_controlledA_B_plus_enhanced_state_direction_oracle_v1.py \
  --vectors output/llava_controlledA_lasttoken_reproduce_v5/lasttoken_vectors_ABC.npz \
  --inject-layers 18 \
  --state-scale 1.0 \
  --train-frac 0.30 \
  --seed 1 \
  --output-dir output/controlledA_B_plus_enhanced_state_direction_oracle_v1 \
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
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import analyze_llava_controlledA_lasttoken_reproduce_v5 as base
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_llava_controlledA_lasttoken_reproduce_v5.py. "
        "Put this script in the AdaptVis llava16 repo root together with v5.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )

from dataset_zoo import get_dataset
from misc import seed_all
from model_zoo import get_model


RELATIONS = ("left", "right", "on", "under")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--vectors",
        default=(
            "output/llava_controlledA_lasttoken_reproduce_v5/"
            "lasttoken_vectors_ABC.npz"
        ),
    )

    p.add_argument(
        "--inject-layers",
        default="18",
        help="Decoder block ids, e.g. 18 or 16-20 or 16,18,20.",
    )

    p.add_argument(
        "--state-scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.30,
    )

    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--min-train-per-class",
        type=int,
        default=2,
    )

    p.add_argument(
        "--base-rms-eps",
        type=float,
        default=1e-5,
        help="Only informational; C always uses enhanced-rms-eps.",
    )

    p.add_argument(
        "--enhanced-rms-eps",
        type=float,
        default=1e-6,
    )

    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.5)
    p.add_argument("--threshold", type=float, default=0.4)

    p.add_argument(
        "--adaptvis-max-layers",
        type=int,
        default=32,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument(
        "--run-state-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--run-delta-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = list(rows)

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
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))

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
        writer.writerows(rows)


def parse_layer_spec(
    text: str,
    available_layers: Sequence[int],
) -> List[int]:
    available = {
        int(x)
        for x in available_layers
    }

    vals: List[int] = []

    for part in str(text).split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a)
            b = int(b)

            step = 1 if b >= a else -1

            vals.extend(
                range(
                    a,
                    b + step,
                    step,
                )
            )

        else:
            vals.append(int(part))

    vals = list(dict.fromkeys(vals))

    missing = [
        x
        for x in vals
        if x not in available
    ]

    if missing:
        raise ValueError(
            f"Injection layers not present in vector cache: {missing}; "
            f"available={sorted(available)}"
        )

    if not vals:
        raise ValueError(
            "No injection layers."
        )

    return vals


def stratified_split_sids(
    labels: np.ndarray,
    sample_index: np.ndarray,
    train_frac: float,
    seed: int,
) -> Tuple[
    set[int],
    set[int],
]:
    rng = random.Random(int(seed))

    train: List[int] = []
    test: List[int] = []

    for relation in RELATIONS:
        positions = np.flatnonzero(
            labels == relation
        ).tolist()

        rng.shuffle(positions)

        if len(positions) < 2:
            raise RuntimeError(
                f"Need >=2 samples for relation={relation}"
            )

        n_train = int(
            round(
                len(positions)
                * float(train_frac)
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(positions) - 1,
            ),
        )

        train.extend(
            int(sample_index[p])
            for p in positions[:n_train]
        )

        test.extend(
            int(sample_index[p])
            for p in positions[n_train:]
        )

    return set(train), set(test)


# =============================================================================
# Fit RAW enhanced directions
# =============================================================================

def fit_raw_direction_bank(
    X: np.ndarray,
    labels: np.ndarray,
    sample_index: np.ndarray,
    eligible: np.ndarray,
    train_sids: set[int],
    vector_layers: np.ndarray,
    inject_layers: Sequence[int],
    *,
    min_train_per_class: int,
) -> Tuple[
    Dict[int, Dict[str, np.ndarray]],
    Dict[int, np.ndarray],
    Dict[str, int],
]:
    layer_to_pos = {
        int(layer): i
        for i, layer
        in enumerate(
            vector_layers.tolist()
        )
    }

    train_mask = np.asarray(
        [
            int(sid) in train_sids
            for sid in sample_index
        ],
        dtype=bool,
    )

    fit_mask = (
        train_mask
        & np.asarray(
            eligible,
            dtype=bool,
        )
    )

    counts = {
        relation: int(
            np.sum(
                fit_mask
                & (
                    labels
                    == relation
                )
            )
        )
        for relation
        in RELATIONS
    }

    missing = [
        relation
        for relation in RELATIONS
        if counts[
            relation
        ] < int(
            min_train_per_class
        )
    ]

    if missing:
        raise RuntimeError(
            f"Cannot fit enhanced direction bank; "
            f"insufficient C-correct TRAIN samples for {missing}. "
            f"counts={counts}"
        )

    bank: Dict[
        int,
        Dict[str, np.ndarray]
    ] = {}

    centers: Dict[
        int,
        np.ndarray
    ] = {}

    for layer in inject_layers:
        pos = layer_to_pos[
            int(layer)
        ]

        states = np.asarray(
            X[
                :,
                pos,
                :,
            ],
            dtype=np.float32,
        )

        # Exact same source population as the direction experiment:
        # only C-correct TRAIN samples.
        center = states[
            fit_mask
        ].mean(
            axis=0
        ).astype(
            np.float32
        )

        centers[
            int(layer)
        ] = center

        relation_bank = {}

        for relation in RELATIONS:
            relation_mask = (
                fit_mask
                & (
                    labels
                    == relation
                )
            )

            mu = states[
                relation_mask
            ].mean(
                axis=0
            ).astype(
                np.float32
            )

            # RAW centered mean direction.
            relation_bank[
                relation
            ] = (
                mu
                - center
            ).astype(
                np.float32
            )

        bank[
            int(layer)
        ] = relation_bank

    return bank, centers, counts


# =============================================================================
# Residual-stream last-token hook
# =============================================================================

class AddLastTokenDirections:
    """
    Add one relation-specific vector to the output residual stream of selected
    decoder blocks.

    The context is used ONLY around the full-prompt C forward and removed
    before cached continuation.

    v_l is fitted from hidden_states[layer+1], so it is injected into the
    output of decoder block `layer`.
    """

    def __init__(
        self,
        model: Any,
        bank: Mapping[
            int,
            Mapping[
                str,
                np.ndarray,
            ],
        ],
        relation: str,
        scale: float,
    ):
        self.handles = []

        if relation not in REL_TO_ID:
            raise ValueError(
                f"Bad relation={relation}"
            )

        layers = (
            model
            .language_model
            .model
            .layers
        )

        for layer_id, relation_bank in (
            bank.items()
        ):
            vector_np = np.asarray(
                relation_bank[
                    relation
                ],
                dtype=np.float32,
            )

            vector_cpu = torch.from_numpy(
                vector_np
            )

            def make_hook(
                vec_cpu: torch.Tensor,
                alpha: float,
                lid: int,
            ):
                def hook(
                    module,
                    inputs,
                    output,
                ):
                    # IMPORTANT:
                    # AdaptVis/model_zoo/llama/modeling_llama_add_attn.py
                    # returns a Python LIST from LLaMADecoderLayer.forward():
                    #
                    #     outputs = [hidden_states,]
                    #     ...
                    #     return outputs
                    #
                    # Standard HF implementations often return tuples.
                    # Preserve the original container type exactly.
                    if isinstance(output, list):
                        if len(output) == 0:
                            raise RuntimeError(
                                f"L{lid}: decoder returned an empty list"
                            )
                        hidden = output[0]
                        container_kind = "list"

                    elif isinstance(output, tuple):
                        if len(output) == 0:
                            raise RuntimeError(
                                f"L{lid}: decoder returned an empty tuple"
                            )
                        hidden = output[0]
                        container_kind = "tuple"

                    elif torch.is_tensor(output):
                        hidden = output
                        container_kind = "tensor"

                    else:
                        raise RuntimeError(
                            f"L{lid}: unsupported decoder output type="
                            f"{type(output)}"
                        )

                    if (
                        not torch.is_tensor(hidden)
                        or hidden.ndim != 3
                    ):
                        raise RuntimeError(
                            f"L{lid}: unexpected hidden state "
                            f"type/shape={type(hidden)} / "
                            f"{getattr(hidden, 'shape', None)}; "
                            f"outer output type={type(output)}"
                        )

                    # Full prompt has sequence length > 1.
                    # The steering context is removed before autoregressive
                    # continuation, but keep this guard for safety.
                    if int(hidden.shape[1]) <= 1:
                        return output

                    vec = vec_cpu.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )

                    if int(vec.numel()) != int(hidden.shape[-1]):
                        raise RuntimeError(
                            f"L{lid}: direction dim={int(vec.numel())} "
                            f"does not match hidden dim={int(hidden.shape[-1])}"
                        )

                    edited = hidden.clone()

                    # The direction was extracted from the decoder-block output
                    # hidden_states[layer+1], at the last prompt token.
                    edited[:, -1, :] = (
                        edited[:, -1, :]
                        + float(alpha) * vec.unsqueeze(0)
                    )

                    # Preserve the custom AdaptVis decoder's output structure.
                    if container_kind == "list":
                        result = list(output)
                        result[0] = edited
                        return result

                    if container_kind == "tuple":
                        return (edited,) + tuple(output[1:])

                    return edited

                return hook

            handle = (
                layers[
                    int(layer_id)
                ]
                .register_forward_hook(
                    make_hook(
                        vector_cpu,
                        float(scale),
                        int(layer_id),
                    )
                )
            )

            self.handles.append(
                handle
            )

    def close(self) -> None:
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

        self.handles = []

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.close()


# =============================================================================
# C generation
# =============================================================================

def selected_adaptvis_weight(
    B_out: Any,
    args: argparse.Namespace,
) -> Tuple[
    float,
    float,
]:
    first_logits = (
        B_out
        .logits[
            :,
            -1,
            :,
        ]
        .detach()
        .float()
    )

    confidence = float(
        torch.softmax(
            first_logits,
            dim=-1,
        )[0]
        .max()
        .item()
    )

    rounded = float(
        np.round(
            confidence,
            2,
        )
    )

    weight = (
        float(
            args.weight1
        )
        if rounded
        < float(
            args.threshold
        )
        else float(
            args.weight2
        )
    )

    return rounded, weight


def run_C_prompt(
    model: Any,
    batch: Mapping[str, Any],
    selected_weight: float,
):
    return base.full_prompt_forward(
        model,
        batch,
        weight=selected_weight,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
    )[0]


def continue_text(
    model: Any,
    tokenizer: Any,
    batch: Mapping[str, Any],
    prompt_output: Any,
    args: argparse.Namespace,
) -> str:
    return base.greedy_continue_from_prompt(
        model,
        tokenizer,
        batch,
        prompt_output,
        max_new_tokens=(
            args.max_new_tokens
        ),
    )


# =============================================================================
# Evaluation
# =============================================================================

def classify_transition(
    before_ok: bool,
    after_ok: bool,
) -> str:
    if (
        not before_ok
        and after_ok
    ):
        return "W2C"

    if (
        before_ok
        and not after_ok
    ):
        return "C2W"

    if (
        before_ok
        and after_ok
    ):
        return "C2C"

    return "W2W"


def method_summary(
    rows: Sequence[
        Mapping[str, Any]
    ],
    method: str,
) -> Dict[str, Any]:
    correct_key = (
        f"{method}_correct"
    )

    transition_key = (
        f"C_to_{method}_transition"
    )

    subset = [
        row
        for row in rows
        if correct_key in row
    ]

    W2C = int(
        sum(
            row[
                transition_key
            ]
            == "W2C"
            for row in subset
        )
    )

    C2W = int(
        sum(
            row[
                transition_key
            ]
            == "C2W"
            for row in subset
        )
    )

    C_wrong_rows = [
        row
        for row in subset
        if int(
            row[
                "C_correct"
            ]
        )
        == 0
    ]

    C_correct_rows = [
        row
        for row in subset
        if int(
            row[
                "C_correct"
            ]
        )
        == 1
    ]

    return {
        "method": method,
        "N": len(subset),

        "C_acc": safe_mean(
            row[
                "C_correct"
            ]
            for row in subset
        ),

        "edited_acc": safe_mean(
            row[
                correct_key
            ]
            for row in subset
        ),

        "gain": (
            safe_mean(
                row[
                    correct_key
                ]
                for row in subset
            )
            - safe_mean(
                row[
                    "C_correct"
                ]
                for row in subset
            )
        ),

        "W2C": W2C,
        "C2W": C2W,
        "net": W2C - C2W,

        "C_wrong_N": len(
            C_wrong_rows
        ),

        "repair_rate_on_C_wrong": (
            safe_mean(
                row[
                    correct_key
                ]
                for row
                in C_wrong_rows
            )
        ),

        "C_correct_N": len(
            C_correct_rows
        ),

        "preserve_rate_on_C_correct": (
            safe_mean(
                row[
                    correct_key
                ]
                for row
                in C_correct_rows
            )
        ),
    }


def per_relation_summary(
    rows: Sequence[
        Mapping[str, Any]
    ],
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    result = []

    for relation in RELATIONS:
        subset = [
            row
            for row in rows
            if row[
                "relation"
            ]
            == relation
        ]

        row: Dict[str, Any] = {
            "relation": relation,
            "N": len(subset),

            "C_acc": safe_mean(
                x[
                    "C_correct"
                ]
                for x in subset
            ),
        }

        for method in methods:
            key = (
                f"{method}_correct"
            )

            method_rows = [
                x
                for x in subset
                if key in x
            ]

            row[
                f"{method}_acc"
            ] = safe_mean(
                x[
                    key
                ]
                for x in method_rows
            )

            row[
                f"{method}_gain"
            ] = (
                row[
                    f"{method}_acc"
                ]
                - row[
                    "C_acc"
                ]
            )

        result.append(
            row
        )

    return result


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

    src = Path(
        args.vectors
    )

    if not src.exists():
        raise FileNotFoundError(
            f"Missing vector cache: {src}"
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

    seed_all(
        args.seed
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

    # ---------------------------------------------------------------------
    # Load successful v5 hidden states.
    # ---------------------------------------------------------------------
    with np.load(
        src,
        allow_pickle=True,
    ) as z:
        required = {
            "sample_index",
            "relation",
            "layers",
            "B_last",
            "C_last",
            "transition_group",
        }

        missing = sorted(
            required.difference(
                z.files
            )
        )

        if missing:
            raise KeyError(
                f"Vector NPZ missing={missing}; available={z.files}"
            )

        sample_index = np.asarray(
            z["sample_index"],
            dtype=np.int64,
        )

        labels = np.asarray(
            [
                str(x)
                .strip()
                .lower()
                for x
                in z[
                    "relation"
                ].tolist()
            ],
            dtype=object,
        )

        vector_layers = np.asarray(
            z["layers"],
            dtype=np.int64,
        )

        B = np.asarray(
            z["B_last"],
            dtype=np.float32,
        )

        C = np.asarray(
            z["C_last"],
            dtype=np.float32,
        )

        delta = (
            np.asarray(
                z[
                    "delta_adapt"
                ],
                dtype=np.float32,
            )
            if "delta_adapt"
            in z.files
            else (
                C - B
            ).astype(
                np.float32
            )
        )

        transition = np.asarray(
            z[
                "transition_group"
            ],
            dtype=object,
        )

    unexpected = sorted(
        set(
            labels.tolist()
        )
        - set(
            RELATIONS
        )
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected labels={unexpected}; expected={RELATIONS}"
        )

    inject_layers = (
        parse_layer_spec(
            args.inject_layers,
            vector_layers,
        )
    )

    C_correct = np.isin(
        transition,
        [
            "W2C",
            "C2C",
        ],
    )

    train_sids, test_sids = (
        stratified_split_sids(
            labels,
            sample_index,
            args.train_frac,
            args.seed,
        )
    )

    state_bank, _, state_counts = (
        fit_raw_direction_bank(
            C,
            labels,
            sample_index,
            C_correct,
            train_sids,
            vector_layers,
            inject_layers,
            min_train_per_class=(
                args.min_train_per_class
            ),
        )
    )

    delta_bank, _, delta_counts = (
        fit_raw_direction_bank(
            delta,
            labels,
            sample_index,
            C_correct,
            train_sids,
            vector_layers,
            inject_layers,
            min_train_per_class=(
                args.min_train_per_class
            ),
        )
    )

    print("\n" + "=" * 150)
    print("C + ENHANCED DIRECTION ORACLE CAUSAL TEST")
    print("=" * 150)

    print(
        f"vectors={src}"
    )

    print(
        f"TRAIN={len(train_sids)} | TEST={len(test_sids)} | "
        f"inject_layers={inject_layers}"
    )

    print(
        f"C-correct TRAIN counts="
        f"{state_counts}"
    )

    print(
        f"state_scale={args.state_scale} | "
        f"delta_scale={args.delta_scale}"
    )

    for layer in inject_layers:
        print(
            f"L{layer:02d} raw vector norms | "
            + " | ".join(
                f"{relation}="
                f"{np.linalg.norm(state_bank[layer][relation]):.4f}"
                for relation
                in RELATIONS
            )
        )

        print(
            f"L{layer:02d} delta vector norms | "
            + " | ".join(
                f"{relation}="
                f"{np.linalg.norm(delta_bank[layer][relation]):.4f}"
                for relation
                in RELATIONS
            )
        )

    print("=" * 150)

    # ---------------------------------------------------------------------
    # Load exact Controlled-A data/model using v5 path.
    # ---------------------------------------------------------------------
    prompts, answers = (
        base.load_prompts(
            "Controlled_Images_A",
            "four",
        )
    )

    dataset = get_dataset(
        "Controlled_Images_A",
        image_preprocess=None,
        download=False,
    )

    wrapper, _ = get_model(
        "llava1.5",
        args.device,
        method="adapt_vis",
        root_dir=args.root_dir,
    )

    model = (
        wrapper.model
    )

    tokenizer = (
        wrapper
        .processor
        .tokenizer
    )

    base.set_rms_eps(
        model,
        args.enhanced_rms_eps,
    )

    rows: List[
        Dict[str, Any]
    ] = []

    with base.RestrictAdaptVisLayers(
        model,
        args.adaptvis_max_layers,
    ):
        iterator = base.iter_samples(
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
            desc="TEST C vs C+direction",
        ):
            sid = int(
                sid
            )

            if sid not in test_sids:
                continue

            relation = (
                base.normalize_relation(
                    gold
                )
            )

            if relation not in REL_TO_ID:
                raise RuntimeError(
                    f"sid={sid}: unsupported relation={relation!r}"
                )

            batch = (
                base.build_input(
                    wrapper,
                    prompt,
                    image,
                )
            )

            # -------------------------------------------------------------
            # Neutral B prompt only to recover exact dynamic AdaptVis weight.
            # -------------------------------------------------------------
            base.set_rms_eps(
                model,
                args.enhanced_rms_eps,
            )

            B_out = (
                base.full_prompt_forward(
                    model,
                    batch,
                    weight=1.0,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=True,
                )[0]
            )

            (
                confidence,
                selected_weight,
            ) = selected_adaptvis_weight(
                B_out,
                args,
            )

            del B_out

            # -------------------------------------------------------------
            # Exact C baseline.
            # -------------------------------------------------------------
            C_out = run_C_prompt(
                model,
                batch,
                selected_weight,
            )

            C_text = continue_text(
                model,
                tokenizer,
                batch,
                C_out,
                args,
            )

            del C_out

            C_ok = bool(
                base._is_correct(
                    gold,
                    C_text,
                )
            )

            row: Dict[str, Any] = {
                "sid": sid,
                "relation": relation,
                "confidence": confidence,
                "selected_weight": (
                    selected_weight
                ),

                "C_text": C_text,
                "C_correct": int(
                    C_ok
                ),
            }

            # -------------------------------------------------------------
            # C + enhanced STATE relation direction, GT oracle selected.
            # -------------------------------------------------------------
            if args.run_state_direction:
                with AddLastTokenDirections(
                    model,
                    state_bank,
                    relation,
                    args.state_scale,
                ):
                    C_state_out = (
                        run_C_prompt(
                            model,
                            batch,
                            selected_weight,
                        )
                    )

                C_state_text = (
                    continue_text(
                        model,
                        tokenizer,
                        batch,
                        C_state_out,
                        args,
                    )
                )

                del C_state_out

                state_ok = bool(
                    base._is_correct(
                        gold,
                        C_state_text,
                    )
                )

                row.update({
                    "state_oracle_text": (
                        C_state_text
                    ),
                    "state_oracle_correct": int(
                        state_ok
                    ),
                    "C_to_state_oracle_transition": (
                        classify_transition(
                            C_ok,
                            state_ok,
                        )
                    ),
                })

            # -------------------------------------------------------------
            # C + AdaptVis DELTA relation direction, GT oracle selected.
            # -------------------------------------------------------------
            if args.run_delta_direction:
                with AddLastTokenDirections(
                    model,
                    delta_bank,
                    relation,
                    args.delta_scale,
                ):
                    C_delta_out = (
                        run_C_prompt(
                            model,
                            batch,
                            selected_weight,
                        )
                    )

                C_delta_text = (
                    continue_text(
                        model,
                        tokenizer,
                        batch,
                        C_delta_out,
                        args,
                    )
                )

                del C_delta_out

                delta_ok = bool(
                    base._is_correct(
                        gold,
                        C_delta_text,
                    )
                )

                row.update({
                    "delta_oracle_text": (
                        C_delta_text
                    ),
                    "delta_oracle_correct": int(
                        delta_ok
                    ),
                    "C_to_delta_oracle_transition": (
                        classify_transition(
                            C_ok,
                            delta_ok,
                        )
                    ),
                })

            rows.append(
                row
            )

            del batch
            cleanup()

    if not rows:
        raise RuntimeError(
            "No TEST rows evaluated."
        )

    methods = []

    if args.run_state_direction:
        methods.append(
            "state_oracle"
        )

    if args.run_delta_direction:
        methods.append(
            "delta_oracle"
        )

    summaries = [
        method_summary(
            rows,
            method,
        )
        for method in methods
    ]

    relation_rows = (
        per_relation_summary(
            rows,
            methods,
        )
    )

    print("\n" + "=" * 150)
    print("ACTUAL GREEDY GENERATION: C + ENHANCED DIRECTION")
    print("=" * 150)

    C_acc = safe_mean(
        row[
            "C_correct"
        ]
        for row in rows
    )

    print(
        f"N_TEST={len(rows)}"
    )

    print(
        f"C dynamic AdaptVis baseline : "
        f"{C_acc:.4f}"
    )

    for summary in summaries:
        print(
            f"{summary['method']:>14s} : "
            f"{summary['edited_acc']:.4f} "
            f"({summary['gain']:+.4f}) | "
            f"W2C={summary['W2C']} "
            f"C2W={summary['C2W']} "
            f"net={summary['net']:+d} | "
            f"repair(C-wrong)="
            f"{summary['repair_rate_on_C_wrong']:.4f} | "
            f"preserve(C-correct)="
            f"{summary['preserve_rate_on_C_correct']:.4f}"
        )

    print("\nPer relation:")

    for row in relation_rows:
        parts = [
            f"{row['relation']:>5s}",
            f"N={row['N']:3d}",
            f"C={row['C_acc']:.4f}",
        ]

        for method in methods:
            parts.append(
                f"{method}="
                f"{row[f'{method}_acc']:.4f} "
                f"({row[f'{method}_gain']:+.4f})"
            )

        print(
            " | ".join(
                parts
            )
        )

    print("=" * 150)

    write_csv(
        outdir
        / "generation_details.csv",
        rows,
    )

    write_csv(
        outdir
        / "summary.csv",
        summaries,
    )

    write_csv(
        outdir
        / "per_relation.csv",
        relation_rows,
    )

    # Save fitted raw vectors.
    arrays = {
        "relation_order": np.asarray(
            RELATIONS,
            dtype=object,
        ),

        "inject_layers": np.asarray(
            inject_layers,
            dtype=np.int64,
        ),
    }

    for layer in inject_layers:
        for relation in RELATIONS:
            arrays[
                f"state_L{layer}_{relation}"
            ] = state_bank[
                layer
            ][
                relation
            ]

            arrays[
                f"delta_L{layer}_{relation}"
            ] = delta_bank[
                layer
            ][
                relation
            ]

    np.savez_compressed(
        outdir
        / "fitted_direction_vectors.npz",
        **arrays,
    )

    metadata = {
        "source_vectors": str(
            src
        ),

        "dataset": (
            "Controlled_Images_A"
        ),

        "N_train": len(
            train_sids
        ),

        "N_test": len(
            rows
        ),

        "train_frac": (
            args.train_frac
        ),

        "seed": (
            args.seed
        ),

        "direction_fit": (
            "C-correct TRAIN only"
        ),

        "direction_mode": (
            "raw centered mean offset"
        ),

        "inject_layers": (
            inject_layers
        ),

        "state_scale": (
            args.state_scale
        ),

        "delta_scale": (
            args.delta_scale
        ),

        "routing": (
            "GT oracle on TEST; used only to test causal actuator quality"
        ),

        "C_condition": {
            "rms_eps": (
                args.enhanced_rms_eps
            ),
            "weight1": (
                args.weight1
            ),
            "weight2": (
                args.weight2
            ),
            "threshold": (
                args.threshold
            ),
            "adaptvis_max_layers": (
                args.adaptvis_max_layers
            ),
        },

        "state_fit_counts": (
            state_counts
        ),

        "delta_fit_counts": (
            delta_counts
        ),
    }

    (
        outdir
        / "config.json"
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
        f"{outdir / 'summary.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'generation_details.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'fitted_direction_vectors.npz'}"
    )



def main_B_plus_enhanced_state_direction() -> None:
    args = parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be in (0,1)")

    src = Path(args.vectors)
    if not src.exists():
        raise FileNotFoundError(f"Missing vector cache: {src}")

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Read saved A/B/C states. Directions are fitted ONLY from C-correct
    # TRAIN samples. TEST uses B with no AdaptVis.
    # ------------------------------------------------------------------
    with np.load(src, allow_pickle=True) as z:
        required = {
            "sample_index",
            "relation",
            "layers",
            "B_last",
            "C_last",
            "transition_group",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(
                f"Vector NPZ missing={missing}; available={z.files}"
            )

        sample_index = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray(
            [str(x).strip().lower() for x in z["relation"].tolist()],
            dtype=object,
        )
        vector_layers = np.asarray(z["layers"], dtype=np.int64)
        C = np.asarray(z["C_last"], dtype=np.float32)
        transition = np.asarray(z["transition_group"], dtype=object)

    unexpected = sorted(set(labels.tolist()) - set(RELATIONS))
    if unexpected:
        raise RuntimeError(
            f"Unexpected labels={unexpected}; expected={RELATIONS}"
        )

    inject_layers = parse_layer_spec(
        args.inject_layers,
        vector_layers,
    )

    C_correct = np.isin(
        transition,
        ["W2C", "C2C"],
    )

    train_sids, test_sids = stratified_split_sids(
        labels,
        sample_index,
        args.train_frac,
        args.seed,
    )

    state_bank, _, state_counts = fit_raw_direction_bank(
        C,
        labels,
        sample_index,
        C_correct,
        train_sids,
        vector_layers,
        inject_layers,
        min_train_per_class=args.min_train_per_class,
    )

    print("\n" + "=" * 150)
    print("B (NO ADAPTVIS) + ENHANCED C-STATE DIRECTION ORACLE TEST")
    print("=" * 150)
    print(f"vectors={src}")
    print(
        f"TRAIN={len(train_sids)} | TEST={len(test_sids)} | "
        f"inject_layers={inject_layers}"
    )
    print(f"C-correct TRAIN counts={state_counts}")
    print(f"state_scale={args.state_scale}")
    print(
        "TEST condition: eps=1e-6, weight=1.0, AdaptVis OFF"
    )

    for layer in inject_layers:
        print(
            f"L{layer:02d} enhanced-state vector norms | "
            + " | ".join(
                f"{r}={np.linalg.norm(state_bank[layer][r]):.4f}"
                for r in RELATIONS
            )
        )

    print("=" * 150)

    prompts, answers = base.load_prompts(
        "Controlled_Images_A",
        "four",
    )

    dataset = get_dataset(
        "Controlled_Images_A",
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
    tokenizer = wrapper.processor.tokenizer

    # B and C share eps=1e-6. The attention intervention is the only thing
    # removed here.
    base.set_rms_eps(
        model,
        args.enhanced_rms_eps,
    )

    rows = []

    # Keep the same repo model/path, but every TEST prompt is run with
    # weight=1.0, so AdaptVis is neutral/off.
    with base.RestrictAdaptVisLayers(
        model,
        args.adaptvis_max_layers,
    ):
        iterator = base.iter_samples(
            dataset,
            prompts,
            answers,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        for sid, image, prompt, gold in tqdm(
            iterator,
            desc="TEST B vs B+enhanced-state-direction",
        ):
            sid = int(sid)

            if sid not in test_sids:
                continue

            relation = base.normalize_relation(gold)

            if relation not in REL_TO_ID:
                raise RuntimeError(
                    f"sid={sid}: unsupported relation={relation!r}"
                )

            batch = base.build_input(
                wrapper,
                prompt,
                image,
            )

            # ----------------------------------------------------------
            # B baseline: eps=1e-6, neutral weight=1.0, no AdaptVis.
            # ----------------------------------------------------------
            B_out = base.full_prompt_forward(
                model,
                batch,
                weight=1.0,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
            )[0]

            B_text = continue_text(
                model,
                tokenizer,
                batch,
                B_out,
                args,
            )
            del B_out

            B_ok = bool(
                base._is_correct(
                    gold,
                    B_text,
                )
            )

            # ----------------------------------------------------------
            # B + direction learned from C-correct TRAIN states.
            # Still weight=1.0: AdaptVis remains OFF.
            # GT relation is used ONLY as an oracle actuator test.
            # ----------------------------------------------------------
            with AddLastTokenDirections(
                model,
                state_bank,
                relation,
                args.state_scale,
            ):
                edited_out = base.full_prompt_forward(
                    model,
                    batch,
                    weight=1.0,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=True,
                )[0]

            edited_text = continue_text(
                model,
                tokenizer,
                batch,
                edited_out,
                args,
            )
            del edited_out

            edited_ok = bool(
                base._is_correct(
                    gold,
                    edited_text,
                )
            )

            transition_name = classify_transition(
                B_ok,
                edited_ok,
            )

            rows.append({
                "sid": sid,
                "relation": relation,
                "B_text": B_text,
                "B_correct": int(B_ok),
                "edited_text": edited_text,
                "edited_correct": int(edited_ok),
                "transition": transition_name,
                "W2C": int(transition_name == "W2C"),
                "C2W": int(transition_name == "C2W"),
            })

            del batch
            cleanup()

    if not rows:
        raise RuntimeError("No TEST rows evaluated.")

    B_acc = safe_mean(
        row["B_correct"]
        for row in rows
    )

    edited_acc = safe_mean(
        row["edited_correct"]
        for row in rows
    )

    W2C = int(sum(row["W2C"] for row in rows))
    C2W = int(sum(row["C2W"] for row in rows))

    B_wrong = [
        row for row in rows
        if int(row["B_correct"]) == 0
    ]
    B_correct_rows = [
        row for row in rows
        if int(row["B_correct"]) == 1
    ]

    summary = {
        "N": len(rows),
        "B_acc": B_acc,
        "B_plus_enhanced_state_oracle_acc": edited_acc,
        "gain": edited_acc - B_acc,
        "W2C": W2C,
        "C2W": C2W,
        "net": W2C - C2W,
        "B_wrong_N": len(B_wrong),
        "repair_rate_on_B_wrong": safe_mean(
            row["edited_correct"]
            for row in B_wrong
        ),
        "B_correct_N": len(B_correct_rows),
        "preserve_rate_on_B_correct": safe_mean(
            row["edited_correct"]
            for row in B_correct_rows
        ),
    }

    relation_rows = []

    for relation in RELATIONS:
        subset = [
            row for row in rows
            if row["relation"] == relation
        ]

        base_acc = safe_mean(
            row["B_correct"]
            for row in subset
        )
        edit_acc = safe_mean(
            row["edited_correct"]
            for row in subset
        )

        relation_rows.append({
            "relation": relation,
            "N": len(subset),
            "B_acc": base_acc,
            "edited_acc": edit_acc,
            "gain": edit_acc - base_acc,
            "W2C": int(sum(row["W2C"] for row in subset)),
            "C2W": int(sum(row["C2W"] for row in subset)),
        })

    print("\n" + "=" * 150)
    print(
        "ACTUAL GREEDY GENERATION: "
        "B (NO ADAPTVIS) + ENHANCED C-STATE DIRECTION"
    )
    print("=" * 150)
    print(f"N_TEST={len(rows)}")
    print(f"B no-AdaptVis baseline            : {B_acc:.4f}")
    print(
        f"B + enhanced-state oracle        : "
        f"{edited_acc:.4f} ({edited_acc - B_acc:+.4f}) | "
        f"W2C={W2C} C2W={C2W} net={W2C-C2W:+d} | "
        f"repair(B-wrong)={summary['repair_rate_on_B_wrong']:.4f} | "
        f"preserve(B-correct)={summary['preserve_rate_on_B_correct']:.4f}"
    )

    print("\nPer relation:")
    for row in relation_rows:
        print(
            f"{row['relation']:>5s} | "
            f"N={row['N']:3d} | "
            f"B={row['B_acc']:.4f} | "
            f"+state={row['edited_acc']:.4f} "
            f"({row['gain']:+.4f}) | "
            f"W2C/C2W={row['W2C']}/{row['C2W']}"
        )

    print("=" * 150)

    write_csv(
        outdir / "generation_details.csv",
        rows,
    )
    write_csv(
        outdir / "summary.csv",
        [summary],
    )
    write_csv(
        outdir / "per_relation.csv",
        relation_rows,
    )

    arrays = {
        "relation_order": np.asarray(
            RELATIONS,
            dtype=object,
        ),
        "inject_layers": np.asarray(
            inject_layers,
            dtype=np.int64,
        ),
    }

    for layer in inject_layers:
        for relation in RELATIONS:
            arrays[
                f"enhanced_state_L{layer}_{relation}"
            ] = state_bank[layer][relation]

    np.savez_compressed(
        outdir / "enhanced_state_direction_vectors.npz",
        **arrays,
    )

    metadata = {
        "source_vectors": str(src),
        "dataset": "Controlled_Images_A",
        "N_train": len(train_sids),
        "N_test": len(rows),
        "train_frac": args.train_frac,
        "seed": args.seed,
        "direction_source": (
            "C-correct TRAIN last-token states"
        ),
        "direction_definition": (
            "raw centered relation mean offset"
        ),
        "inject_layers": inject_layers,
        "state_scale": args.state_scale,
        "test_condition": {
            "name": "B",
            "rms_eps": args.enhanced_rms_eps,
            "weight": 1.0,
            "adaptvis": False,
        },
        "routing": (
            "GT oracle on TEST, used only to test causal transfer "
            "of enhanced-state direction to baseline B"
        ),
        "state_fit_counts": state_counts,
    }

    (outdir / "config.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n[saved] {outdir / 'summary.csv'}")
    print(f"[saved] {outdir / 'generation_details.csv'}")
    print(f"[saved] {outdir / 'enhanced_state_direction_vectors.npz'}")


if __name__ == "__main__":
    main_B_plus_enhanced_state_direction()
