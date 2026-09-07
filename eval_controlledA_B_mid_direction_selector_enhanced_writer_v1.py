
# -*- coding: utf-8 -*-

"""
eval_controlledA_B_mid_direction_selector_enhanced_writer_v1.py

Non-oracle Controlled-A pipeline:

    B middle-layer last-token state
        -> middle direction-vector selector
        -> predicted relation r_hat
        -> enhanced C-state relation vector v^C_r_hat
        -> inject into B last token
        -> actual greedy generation

Definitions
===========
B:
    eps=1e-6, weight=1.0, NO AdaptVis.

C:
    eps=1e-6, successful dynamic AdaptVis.

SELECTOR
========
Fit relation directions from ALL B TRAIN samples, not only B-correct samples:

    center^B_l = mean(h^B_l | TRAIN)

    d^B_{r,l}
        = normalize(
            mean(h^B_l | relation=r, TRAIN)
            - center^B_l
          )

At TEST, from the unedited B last-token state at selector layer l:

    q = normalize(h^B_l - center^B_l)

    score(r) = cos(q, d^B_{r,l})

    r_hat = argmax_r score(r)

No TEST GT is used for routing.

Why ALL B TRAIN samples?
========================
The native generation is extremely poor on "on", so B-correct-only directions
cannot be fit. But previous probing showed that the B middle-layer last-token
state itself remains highly relation-decodable. The selector is therefore
trained from all labeled TRAIN examples and evaluated strictly on held-out TEST.

SELECTOR LAYER SELECTION
========================
Default:
    --selector-layer auto
    --selector-candidate-layers 8-18

The best selector layer is chosen using TRAIN-ONLY repeated internal
stratified fit/validation splits. Then the selector directions are refit using
ALL outer TRAIN samples at that frozen layer.

WRITER
======
Use the already validated enhanced-state direction learned from C-correct TRAIN:

    center^C_l = mean(h^C_l | C-correct TRAIN)

    v^C_{r,l}
        = mean(h^C_l | relation=r, C-correct TRAIN)
          - center^C_l

Default writer/injection layer:
    L18

TEST INTERVENTION
=================
During the B full-prompt forward:

1. At the selector layer, read the UNEDITED last-token hidden state.
2. Predict r_hat with the middle direction-vector selector.
3. At the writer layer, add:

       h_last <- h_last + scale * v^C_{r_hat}

If selector layer == writer layer, prediction is made from the original block
output first, then the selected vector is added to that same output.

The hook is removed before autoregressive continuation.

OUTPUT
======
Actual greedy generation:
    B baseline accuracy
    middle-direction selector accuracy       (reporting only; GT not routing)
    B + non-oracle selected direction accuracy
    optional oracle control
    W2C/C2W
    selector-correct / selector-wrong groups
    per-relation results
    selector confusion matrix

Example
=======
CUDA_VISIBLE_DEVICES=0 python eval_controlledA_B_mid_direction_selector_enhanced_writer_v1.py \
  --vectors output/llava_controlledA_lasttoken_reproduce_v5/lasttoken_vectors_ABC.npz \
  --selector-layer auto \
  --selector-candidate-layers 8-18 \
  --inject-layers 18 \
  --state-scale 1.0 \
  --train-frac 0.30 \
  --selector-cv-repeats 20 \
  --seed 1 \
  --output-dir output/controlledA_B_mid_direction_selector_enhanced_writer_v1 \
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
from collections import Counter
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
        "Run from the AdaptVis llava16 repository root.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )

from dataset_zoo import get_dataset
from misc import seed_all
from model_zoo import get_model


RELATIONS = ("left", "right", "on", "under")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for r, i in REL_TO_ID.items()}
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
        "--selector-layer",
        default="auto",
        help=(
            "'auto' selects the best layer using TRAIN-only internal CV; "
            "otherwise provide one decoder layer id."
        ),
    )

    p.add_argument(
        "--selector-candidate-layers",
        default="8-18",
        help=(
            "Candidate middle layers used only when --selector-layer=auto."
        ),
    )

    p.add_argument(
        "--selector-cv-repeats",
        type=int,
        default=20,
    )

    p.add_argument(
        "--selector-cv-fit-frac",
        type=float,
        default=0.70,
        help="Internal fit fraction inside the OUTER TRAIN split.",
    )

    p.add_argument(
        "--inject-layers",
        default="18",
        help="Writer layer(s), e.g. 18 or 18-20.",
    )

    p.add_argument(
        "--state-scale",
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
        "--enhanced-rms-eps",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--adaptvis-max-layers",
        type=int,
        default=32,
        help=(
            "Only used to preserve the repository model context. "
            "TEST B always uses weight=1.0, so AdaptVis is neutral."
        ),
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
        "--run-oracle-control",
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_layer_spec(
    text: str,
    available_layers: Sequence[int],
) -> List[int]:
    available = {int(x) for x in available_layers}
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

    values = list(dict.fromkeys(values))

    missing = [
        value
        for value in values
        if value not in available
    ]

    if missing:
        raise ValueError(
            f"Requested layers not in vector cache: {missing}; "
            f"available={sorted(available)}"
        )

    if not values:
        raise ValueError("No layers selected.")

    return values


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, EPS)


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


# =============================================================================
# Outer TRAIN / TEST split
# =============================================================================

def stratified_split_positions(
    labels: np.ndarray,
    indices: Sequence[int],
    fit_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(int(seed))
    indices = [int(x) for x in indices]

    fit: List[int] = []
    val: List[int] = []

    for relation in RELATIONS:
        ids = [
            i
            for i in indices
            if labels[i] == relation
        ]

        if len(ids) < 2:
            raise RuntimeError(
                f"Need >=2 samples for {relation}; got {len(ids)}"
            )

        rng.shuffle(ids)

        n_fit = int(round(len(ids) * float(fit_frac)))
        n_fit = max(1, min(n_fit, len(ids) - 1))

        fit.extend(ids[:n_fit])
        val.extend(ids[n_fit:])

    rng.shuffle(fit)
    rng.shuffle(val)

    return (
        np.asarray(fit, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
    )


def outer_stratified_split(
    labels: np.ndarray,
    sample_index: np.ndarray,
    train_frac: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    set[int],
    set[int],
]:
    all_positions = np.arange(len(labels), dtype=np.int64)

    train_pos, test_pos = stratified_split_positions(
        labels,
        all_positions,
        train_frac,
        seed,
    )

    train_sids = {
        int(sample_index[pos])
        for pos in train_pos
    }

    test_sids = {
        int(sample_index[pos])
        for pos in test_pos
    }

    return train_pos, test_pos, train_sids, test_sids


# =============================================================================
# Direction fitting
# =============================================================================

def fit_selector_directions(
    states: np.ndarray,
    labels: np.ndarray,
    fit_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Fit four normalized relation directions from ALL B fit samples.

      center = mean(B_fit)
      d_r = normalize(mean(B_fit[y=r]) - center)
    """
    fit_positions = np.asarray(fit_positions, dtype=np.int64)

    counts = {
        relation: int(
            np.sum(
                labels[fit_positions] == relation
            )
        )
        for relation in RELATIONS
    }

    if min(counts.values()) < 1:
        raise RuntimeError(
            f"Missing selector relation in fit data: {counts}"
        )

    X = np.asarray(
        states[fit_positions],
        dtype=np.float64,
    )

    center = X.mean(axis=0)
    directions = []

    for relation in RELATIONS:
        relation_positions = fit_positions[
            labels[fit_positions] == relation
        ]

        mu = np.asarray(
            states[relation_positions],
            dtype=np.float64,
        ).mean(axis=0)

        direction = mu - center
        norm = float(np.linalg.norm(direction))

        if norm <= EPS:
            raise RuntimeError(
                f"Near-zero selector direction for {relation}"
            )

        directions.append(direction / norm)

    return (
        center.astype(np.float32),
        np.stack(directions, axis=0).astype(np.float32),
        counts,
    )


def selector_scores(
    states: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    q = (
        np.asarray(states, dtype=np.float64)
        - np.asarray(center, dtype=np.float64)
    )

    q = normalize_rows(q)
    d = normalize_rows(directions)

    return q @ d.T


def evaluate_selector(
    states: np.ndarray,
    labels: np.ndarray,
    eval_positions: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> Dict[str, Any]:
    eval_positions = np.asarray(eval_positions, dtype=np.int64)

    scores = selector_scores(
        states[eval_positions],
        center,
        directions,
    )

    gt = np.asarray(
        [
            REL_TO_ID[str(x)]
            for x in labels[eval_positions]
        ],
        dtype=np.int64,
    )

    pred = np.argmax(scores, axis=1)

    rows = np.arange(len(gt), dtype=np.int64)
    gt_score = scores[rows, gt]

    masked = scores.copy()
    masked[rows, gt] = -np.inf

    margin = gt_score - np.max(masked, axis=1)

    result: Dict[str, Any] = {
        "accuracy": float(np.mean(pred == gt)),
        "gt_margin": float(np.mean(margin)),
    }

    for relation in RELATIONS:
        rid = REL_TO_ID[relation]
        mask = gt == rid

        result[f"{relation}_accuracy"] = (
            float(np.mean(pred[mask] == gt[mask]))
            if int(mask.sum()) > 0
            else float("nan")
        )

    return result


def choose_selector_layer_train_only(
    B: np.ndarray,
    labels: np.ndarray,
    train_positions: np.ndarray,
    vector_layers: np.ndarray,
    candidate_layers: Sequence[int],
    repeats: int,
    fit_frac: float,
    seed: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    layer_to_pos = {
        int(layer): i
        for i, layer in enumerate(vector_layers.tolist())
    }

    raw_rows: List[Dict[str, Any]] = []

    for repeat in range(repeats):
        fit_pos, val_pos = stratified_split_positions(
            labels,
            train_positions.tolist(),
            fit_frac,
            seed + 1000 + repeat,
        )

        for layer in candidate_layers:
            lp = layer_to_pos[int(layer)]

            center, directions, counts = fit_selector_directions(
                B[:, lp, :],
                labels,
                fit_pos,
            )

            metrics = evaluate_selector(
                B[:, lp, :],
                labels,
                val_pos,
                center,
                directions,
            )

            raw_rows.append({
                "repeat": repeat,
                "layer": int(layer),
                "accuracy": metrics["accuracy"],
                "gt_margin": metrics["gt_margin"],
                "left_accuracy": metrics["left_accuracy"],
                "right_accuracy": metrics["right_accuracy"],
                "on_accuracy": metrics["on_accuracy"],
                "under_accuracy": metrics["under_accuracy"],
                "fit_left": counts["left"],
                "fit_right": counts["right"],
                "fit_on": counts["on"],
                "fit_under": counts["under"],
            })

    summary_rows = []

    for layer in candidate_layers:
        rows = [
            row
            for row in raw_rows
            if int(row["layer"]) == int(layer)
        ]

        summary_rows.append({
            "layer": int(layer),
            "accuracy_mean": safe_mean(
                row["accuracy"]
                for row in rows
            ),
            "accuracy_std": safe_std(
                row["accuracy"]
                for row in rows
            ),
            "gt_margin_mean": safe_mean(
                row["gt_margin"]
                for row in rows
            ),
            "left_accuracy_mean": safe_mean(
                row["left_accuracy"]
                for row in rows
            ),
            "right_accuracy_mean": safe_mean(
                row["right_accuracy"]
                for row in rows
            ),
            "on_accuracy_mean": safe_mean(
                row["on_accuracy"]
                for row in rows
            ),
            "under_accuracy_mean": safe_mean(
                row["under_accuracy"]
                for row in rows
            ),
        })

    summary_rows.sort(
        key=lambda row: (
            -float(row["accuracy_mean"]),
            -float(row["gt_margin_mean"]),
            int(row["layer"]),
        )
    )

    best_layer = int(summary_rows[0]["layer"])

    print("\n" + "=" * 150)
    print("TRAIN-ONLY MIDDLE DIRECTION SELECTOR LAYER SELECTION")
    print("=" * 150)

    for row in summary_rows:
        print(
            f"L{row['layer']:02d} | "
            f"cv_acc={row['accuracy_mean']:.4f}"
            f"±{row['accuracy_std']:.4f} | "
            f"margin={row['gt_margin_mean']:+.4f} | "
            f"L/R/On/U="
            f"{row['left_accuracy_mean']:.3f}/"
            f"{row['right_accuracy_mean']:.3f}/"
            f"{row['on_accuracy_mean']:.3f}/"
            f"{row['under_accuracy_mean']:.3f}"
        )

    print(
        f"\nSELECTED selector layer = L{best_layer:02d} "
        f"(TRAIN-only CV)"
    )

    print("=" * 150)

    return best_layer, summary_rows


def fit_enhanced_writer_bank(
    C: np.ndarray,
    labels: np.ndarray,
    C_correct: np.ndarray,
    train_positions: np.ndarray,
    vector_layers: np.ndarray,
    inject_layers: Sequence[int],
    min_train_per_class: int,
) -> Tuple[
    Dict[int, Dict[str, np.ndarray]],
    Dict[str, int],
]:
    """
    Same enhanced-state writer as the successful oracle experiment:

      center_C = mean(C | C-correct TRAIN)
      v^C_r = mean(C | relation=r, C-correct TRAIN) - center_C

    Raw centered mean offsets: no unit normalization.
    """
    layer_to_pos = {
        int(layer): i
        for i, layer in enumerate(vector_layers.tolist())
    }

    train_mask = np.zeros(
        len(labels),
        dtype=bool,
    )

    train_mask[
        np.asarray(
            train_positions,
            dtype=np.int64,
        )
    ] = True

    fit_mask = (
        train_mask
        & np.asarray(C_correct, dtype=bool)
    )

    counts = {
        relation: int(
            np.sum(
                fit_mask
                & (labels == relation)
            )
        )
        for relation in RELATIONS
    }

    missing = [
        relation
        for relation in RELATIONS
        if counts[relation] < int(min_train_per_class)
    ]

    if missing:
        raise RuntimeError(
            f"Insufficient C-correct TRAIN samples for writer: "
            f"{missing}; counts={counts}"
        )

    bank: Dict[int, Dict[str, np.ndarray]] = {}

    for layer in inject_layers:
        lp = layer_to_pos[int(layer)]

        states = np.asarray(
            C[:, lp, :],
            dtype=np.float32,
        )

        center = states[fit_mask].mean(axis=0)

        relation_bank = {}

        for relation in RELATIONS:
            mask = (
                fit_mask
                & (labels == relation)
            )

            mu = states[mask].mean(axis=0)

            relation_bank[relation] = (
                mu - center
            ).astype(np.float32)

        bank[int(layer)] = relation_bank

    return bank, counts


# =============================================================================
# Custom AdaptVis decoder output handling
# =============================================================================

def unpack_decoder_output(
    output: Any,
    layer_id: int,
) -> Tuple[torch.Tensor, str]:
    if isinstance(output, list):
        if not output:
            raise RuntimeError(
                f"L{layer_id}: empty decoder list"
            )
        hidden = output[0]
        kind = "list"

    elif isinstance(output, tuple):
        if not output:
            raise RuntimeError(
                f"L{layer_id}: empty decoder tuple"
            )
        hidden = output[0]
        kind = "tuple"

    elif torch.is_tensor(output):
        hidden = output
        kind = "tensor"

    else:
        raise RuntimeError(
            f"L{layer_id}: unsupported decoder output type={type(output)}"
        )

    if (
        not torch.is_tensor(hidden)
        or hidden.ndim != 3
    ):
        raise RuntimeError(
            f"L{layer_id}: bad hidden type/shape="
            f"{type(hidden)} / {getattr(hidden, 'shape', None)}"
        )

    return hidden, kind


def repack_decoder_output(
    original: Any,
    edited_hidden: torch.Tensor,
    kind: str,
):
    if kind == "list":
        result = list(original)
        result[0] = edited_hidden
        return result

    if kind == "tuple":
        return (edited_hidden,) + tuple(original[1:])

    return edited_hidden


# =============================================================================
# Dynamic non-oracle selector -> enhanced writer hook
# =============================================================================

class MiddleDirectionSelectorEnhancedWriter:
    """
    One full-prompt forward:

      selector layer:
          read UNEDITED B last-token state
          classify by B middle directions
          store predicted relation

      writer layer(s):
          inject enhanced C-state vector selected by that prediction

    Supports selector_layer == writer_layer:
      prediction happens first, then the selected vector is added.
    """

    def __init__(
        self,
        model: Any,
        selector_layer: int,
        selector_center: np.ndarray,
        selector_directions: np.ndarray,
        writer_bank: Mapping[int, Mapping[str, np.ndarray]],
        scale: float,
    ):
        self.model = model
        self.selector_layer = int(selector_layer)
        self.selector_center_cpu = torch.from_numpy(
            np.asarray(selector_center, dtype=np.float32)
        )
        self.selector_dirs_cpu = torch.from_numpy(
            np.asarray(selector_directions, dtype=np.float32)
        )

        self.writer_bank = {
            int(layer): {
                relation: torch.from_numpy(
                    np.asarray(vector, dtype=np.float32)
                )
                for relation, vector in relation_bank.items()
            }
            for layer, relation_bank in writer_bank.items()
        }

        self.scale = float(scale)
        self.handles = []

        self.prediction: Optional[str] = None
        self.scores: Optional[np.ndarray] = None
        self.margin: Optional[float] = None

        writer_layers = sorted(self.writer_bank)

        if writer_layers and min(writer_layers) < self.selector_layer:
            raise ValueError(
                "Every writer layer must be >= selector layer for a "
                "single-forward non-oracle pipeline. "
                f"selector=L{self.selector_layer}, writer={writer_layers}"
            )

        layers = (
            model
            .language_model
            .model
            .layers
        )

        # Register exactly one hook per relevant layer.
        relevant = sorted(
            set(
                [self.selector_layer]
                + writer_layers
            )
        )

        for layer_id in relevant:
            handle = (
                layers[layer_id]
                .register_forward_hook(
                    self._make_hook(layer_id)
                )
            )
            self.handles.append(handle)

    def _predict_relation(
        self,
        hidden: torch.Tensor,
    ) -> str:
        # batch size is expected to be 1.
        x = (
            hidden[
                0,
                -1,
                :
            ]
            .detach()
            .float()
        )

        center = self.selector_center_cpu.to(
            device=x.device,
            dtype=torch.float32,
        )

        dirs = self.selector_dirs_cpu.to(
            device=x.device,
            dtype=torch.float32,
        )

        q = x - center

        q = q / q.norm(
            p=2
        ).clamp_min(
            1e-12
        )

        dirs = dirs / dirs.norm(
            p=2,
            dim=-1,
            keepdim=True,
        ).clamp_min(
            1e-12
        )

        scores = torch.mv(
            dirs,
            q,
        )

        pred_id = int(
            torch.argmax(
                scores
            ).item()
        )

        sorted_scores, _ = torch.sort(
            scores,
            descending=True,
        )

        margin = float(
            (
                sorted_scores[0]
                - sorted_scores[1]
            ).item()
        )

        self.scores = (
            scores.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        self.margin = margin

        return ID_TO_REL[
            pred_id
        ]

    def _make_hook(
        self,
        layer_id: int,
    ):
        def hook(
            module,
            inputs,
            output,
        ):
            hidden, kind = unpack_decoder_output(
                output,
                layer_id,
            )

            # Hook exists only around full prompt, but keep guard.
            if int(hidden.shape[1]) <= 1:
                return output

            edited = hidden

            # ----------------------------------------------------------
            # READ first, from the UNEDITED B state.
            # ----------------------------------------------------------
            if layer_id == self.selector_layer:
                self.prediction = self._predict_relation(
                    hidden
                )

            # ----------------------------------------------------------
            # WRITE after relation has been selected.
            # ----------------------------------------------------------
            if layer_id in self.writer_bank:
                if self.prediction is None:
                    raise RuntimeError(
                        f"L{layer_id}: writer reached before selector "
                        f"L{self.selector_layer}"
                    )

                vec = (
                    self.writer_bank[
                        layer_id
                    ][
                        self.prediction
                    ]
                    .to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                )

                if int(vec.numel()) != int(
                    hidden.shape[-1]
                ):
                    raise RuntimeError(
                        f"L{layer_id}: writer dim={vec.numel()} "
                        f"hidden dim={hidden.shape[-1]}"
                    )

                edited = hidden.clone()

                edited[
                    :,
                    -1,
                    :
                ] = (
                    edited[
                        :,
                        -1,
                        :
                    ]
                    + self.scale
                    * vec.unsqueeze(0)
                )

            if edited is hidden:
                return output

            return repack_decoder_output(
                output,
                edited,
                kind,
            )

        return hook

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


class OracleEnhancedWriter:
    """
    Same enhanced writer, but GT selected.
    Only used as an optional control.
    """

    def __init__(
        self,
        model: Any,
        writer_bank: Mapping[
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

        layers = (
            model
            .language_model
            .model
            .layers
        )

        for layer_id, relation_bank in writer_bank.items():
            vec_cpu = torch.from_numpy(
                np.asarray(
                    relation_bank[relation],
                    dtype=np.float32,
                )
            )

            def make_hook(
                lid: int,
                vec_cpu_local: torch.Tensor,
            ):
                def hook(
                    module,
                    inputs,
                    output,
                ):
                    hidden, kind = unpack_decoder_output(
                        output,
                        lid,
                    )

                    if int(hidden.shape[1]) <= 1:
                        return output

                    vec = vec_cpu_local.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )

                    edited = hidden.clone()

                    edited[
                        :,
                        -1,
                        :
                    ] = (
                        edited[
                            :,
                            -1,
                            :
                        ]
                        + float(scale)
                        * vec.unsqueeze(0)
                    )

                    return repack_decoder_output(
                        output,
                        edited,
                        kind,
                    )

                return hook

            self.handles.append(
                layers[
                    int(layer_id)
                ].register_forward_hook(
                    make_hook(
                        int(layer_id),
                        vec_cpu,
                    )
                )
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# =============================================================================
# Generation helpers
# =============================================================================

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
        max_new_tokens=args.max_new_tokens,
    )


# =============================================================================
# Summaries
# =============================================================================

def confusion_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result = []

    for gt in RELATIONS:
        for pred in RELATIONS:
            result.append({
                "gt": gt,
                "pred": pred,
                "count": int(
                    sum(
                        row["relation"] == gt
                        and row["selector_pred"] == pred
                        for row in rows
                    )
                ),
            })

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

    if not (
        0.0
        < args.selector_cv_fit_frac
        < 1.0
    ):
        raise ValueError(
            "--selector-cv-fit-frac must be in (0,1)"
        )

    src = Path(args.vectors)

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

    seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------------------------------------------------------
    # Load B/C cached last-token states.
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
                f"Vector cache missing={missing}; "
                f"available={z.files}"
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

        transition = np.asarray(
            z["transition_group"],
            dtype=object,
        )

    unexpected = sorted(
        set(labels.tolist())
        - set(RELATIONS)
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected relations={unexpected}; "
            f"expected={RELATIONS}"
        )

    layer_to_pos = {
        int(layer): i
        for i, layer
        in enumerate(
            vector_layers.tolist()
        )
    }

    (
        train_positions,
        test_positions,
        train_sids,
        test_sids,
    ) = outer_stratified_split(
        labels,
        sample_index,
        args.train_frac,
        args.seed,
    )

    C_correct = np.isin(
        transition,
        ["W2C", "C2C"],
    )

    # ---------------------------------------------------------------------
    # TRAIN-only selector layer choice.
    # ---------------------------------------------------------------------
    if (
        str(args.selector_layer)
        .strip()
        .lower()
        == "auto"
    ):
        candidate_layers = parse_layer_spec(
            args.selector_candidate_layers,
            vector_layers,
        )

        selector_layer, selector_cv_rows = (
            choose_selector_layer_train_only(
                B,
                labels,
                train_positions,
                vector_layers,
                candidate_layers,
                args.selector_cv_repeats,
                args.selector_cv_fit_frac,
                args.seed,
            )
        )

        write_csv(
            outdir
            / "selector_train_cv.csv",
            selector_cv_rows,
        )

    else:
        selector_layer = int(
            args.selector_layer
        )

        if selector_layer not in layer_to_pos:
            raise ValueError(
                f"selector L{selector_layer} "
                f"not in cached layers."
            )

        candidate_layers = [
            selector_layer
        ]

        selector_cv_rows = []

    inject_layers = parse_layer_spec(
        args.inject_layers,
        vector_layers,
    )

    if min(inject_layers) < selector_layer:
        raise ValueError(
            f"Writer must not precede selector in one forward: "
            f"selector=L{selector_layer}, "
            f"writer={inject_layers}"
        )

    # ---------------------------------------------------------------------
    # Refit selector directions on ALL outer TRAIN samples.
    # ---------------------------------------------------------------------
    selector_pos = layer_to_pos[
        selector_layer
    ]

    (
        selector_center,
        selector_directions,
        selector_train_counts,
    ) = fit_selector_directions(
        B[
            :,
            selector_pos,
            :,
        ],
        labels,
        train_positions,
    )

    selector_test_offline = (
        evaluate_selector(
            B[
                :,
                selector_pos,
                :,
            ],
            labels,
            test_positions,
            selector_center,
            selector_directions,
        )
    )

    # ---------------------------------------------------------------------
    # Fit writer from C-correct outer TRAIN only.
    # ---------------------------------------------------------------------
    writer_bank, writer_counts = (
        fit_enhanced_writer_bank(
            C,
            labels,
            C_correct,
            train_positions,
            vector_layers,
            inject_layers,
            args.min_train_per_class,
        )
    )

    print("\n" + "=" * 150)
    print("CONTROLLED-A NON-ORACLE: B MIDDLE DIRECTION SELECTOR -> ENHANCED C-STATE WRITER")
    print("=" * 150)

    print(
        f"TRAIN={len(train_positions)} | "
        f"TEST={len(test_positions)}"
    )

    print(
        f"selector=L{selector_layer} "
        f"(B all-TRAIN directions) | "
        f"writer={inject_layers} "
        f"(C-correct TRAIN enhanced directions)"
    )

    print(
        f"selector TRAIN counts="
        f"{selector_train_counts}"
    )

    print(
        f"writer C-correct TRAIN counts="
        f"{writer_counts}"
    )

    print(
        f"offline held-out selector check="
        f"{selector_test_offline['accuracy']:.4f} | "
        f"L/R/On/U="
        f"{selector_test_offline['left_accuracy']:.3f}/"
        f"{selector_test_offline['right_accuracy']:.3f}/"
        f"{selector_test_offline['on_accuracy']:.3f}/"
        f"{selector_test_offline['under_accuracy']:.3f}"
    )

    for layer in inject_layers:
        print(
            f"L{layer:02d} writer norms | "
            + " | ".join(
                f"{relation}="
                f"{np.linalg.norm(writer_bank[layer][relation]):.4f}"
                for relation in RELATIONS
            )
        )

    print("=" * 150)

    # Save fitted selector/writer vectors.
    save_arrays: Dict[str, Any] = {
        "relation_order": np.asarray(
            RELATIONS,
            dtype=object,
        ),
        "selector_layer": np.asarray(
            selector_layer,
            dtype=np.int64,
        ),
        "selector_center": (
            selector_center
        ),
        "selector_directions": (
            selector_directions
        ),
        "inject_layers": np.asarray(
            inject_layers,
            dtype=np.int64,
        ),
    }

    for layer in inject_layers:
        for relation in RELATIONS:
            save_arrays[
                f"writer_L{layer}_{relation}"
            ] = writer_bank[
                layer
            ][
                relation
            ]

    np.savez_compressed(
        outdir
        / "selector_and_writer_vectors.npz",
        **save_arrays,
    )

    # ---------------------------------------------------------------------
    # Actual model generation.
    # ---------------------------------------------------------------------
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

    base.set_rms_eps(
        model,
        args.enhanced_rms_eps,
    )

    rows: List[Dict[str, Any]] = []

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
            desc="TEST B middle-selector -> enhanced-writer",
        ):
            sid = int(sid)

            if sid not in test_sids:
                continue

            relation = (
                base.normalize_relation(
                    gold
                )
            )

            if relation not in REL_TO_ID:
                raise RuntimeError(
                    f"sid={sid}: bad relation={relation!r}"
                )

            batch = base.build_input(
                wrapper,
                prompt,
                image,
            )

            # -------------------------------------------------------------
            # B baseline: eps=1e-6, weight=1.0, no AdaptVis.
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # TRUE NON-ORACLE:
            # read B middle direction -> predict relation -> choose writer.
            # -------------------------------------------------------------
            with MiddleDirectionSelectorEnhancedWriter(
                model,
                selector_layer,
                selector_center,
                selector_directions,
                writer_bank,
                args.state_scale,
            ) as routed:
                edit_out = base.full_prompt_forward(
                    model,
                    batch,
                    weight=1.0,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=True,
                )[0]

            if routed.prediction is None:
                raise RuntimeError(
                    f"sid={sid}: selector did not produce a prediction"
                )

            selector_pred = routed.prediction

            edit_text = continue_text(
                model,
                tokenizer,
                batch,
                edit_out,
                args,
            )

            del edit_out

            edit_ok = bool(
                base._is_correct(
                    gold,
                    edit_text,
                )
            )

            selector_ok = (
                selector_pred
                == relation
            )

            transition_nonoracle = (
                classify_transition(
                    B_ok,
                    edit_ok,
                )
            )

            row: Dict[str, Any] = {
                "sid": sid,
                "relation": relation,

                "B_text": B_text,
                "B_correct": int(B_ok),

                "selector_pred": (
                    selector_pred
                ),
                "selector_correct": int(
                    selector_ok
                ),
                "selector_margin": float(
                    routed.margin
                    if routed.margin is not None
                    else float("nan")
                ),

                "selector_score_left": float(
                    routed.scores[
                        REL_TO_ID["left"]
                    ]
                ),
                "selector_score_right": float(
                    routed.scores[
                        REL_TO_ID["right"]
                    ]
                ),
                "selector_score_on": float(
                    routed.scores[
                        REL_TO_ID["on"]
                    ]
                ),
                "selector_score_under": float(
                    routed.scores[
                        REL_TO_ID["under"]
                    ]
                ),

                "nonoracle_text": (
                    edit_text
                ),
                "nonoracle_correct": int(
                    edit_ok
                ),

                "transition": (
                    transition_nonoracle
                ),

                "W2C": int(
                    transition_nonoracle
                    == "W2C"
                ),

                "C2W": int(
                    transition_nonoracle
                    == "C2W"
                ),
            }

            # -------------------------------------------------------------
            # Optional oracle control using exact same writer bank.
            # -------------------------------------------------------------
            if args.run_oracle_control:
                with OracleEnhancedWriter(
                    model,
                    writer_bank,
                    relation,
                    args.state_scale,
                ):
                    oracle_out = (
                        base.full_prompt_forward(
                            model,
                            batch,
                            weight=1.0,
                            output_attentions=False,
                            output_hidden_states=False,
                            use_cache=True,
                        )[0]
                    )

                oracle_text = continue_text(
                    model,
                    tokenizer,
                    batch,
                    oracle_out,
                    args,
                )

                del oracle_out

                oracle_ok = bool(
                    base._is_correct(
                        gold,
                        oracle_text,
                    )
                )

                row[
                    "oracle_text"
                ] = oracle_text

                row[
                    "oracle_correct"
                ] = int(
                    oracle_ok
                )

            rows.append(row)

            del batch
            cleanup()

    if not rows:
        raise RuntimeError(
            "No TEST samples evaluated."
        )

    # ---------------------------------------------------------------------
    # Final summaries.
    # ---------------------------------------------------------------------
    B_acc = safe_mean(
        row["B_correct"]
        for row in rows
    )

    selector_acc = safe_mean(
        row["selector_correct"]
        for row in rows
    )

    nonoracle_acc = safe_mean(
        row["nonoracle_correct"]
        for row in rows
    )

    W2C = int(
        sum(
            row["W2C"]
            for row in rows
        )
    )

    C2W = int(
        sum(
            row["C2W"]
            for row in rows
        )
    )

    selector_correct_rows = [
        row
        for row in rows
        if int(
            row["selector_correct"]
        ) == 1
    ]

    selector_wrong_rows = [
        row
        for row in rows
        if int(
            row["selector_correct"]
        ) == 0
    ]

    oracle_acc = (
        safe_mean(
            row["oracle_correct"]
            for row in rows
        )
        if args.run_oracle_control
        else float("nan")
    )

    summary = {
        "N": len(rows),
        "selector_layer": selector_layer,
        "writer_layers": ",".join(
            str(x)
            for x in inject_layers
        ),

        "B_baseline_acc": B_acc,
        "selector_acc": selector_acc,
        "nonoracle_acc": nonoracle_acc,
        "gain_vs_B": (
            nonoracle_acc
            - B_acc
        ),

        "oracle_acc": oracle_acc,

        "W2C": W2C,
        "C2W": C2W,
        "net": W2C - C2W,

        "selector_correct_N": len(
            selector_correct_rows
        ),

        "selector_wrong_N": len(
            selector_wrong_rows
        ),

        "B_acc_selector_correct": (
            safe_mean(
                row["B_correct"]
                for row
                in selector_correct_rows
            )
        ),

        "nonoracle_acc_selector_correct": (
            safe_mean(
                row["nonoracle_correct"]
                for row
                in selector_correct_rows
            )
        ),

        "B_acc_selector_wrong": (
            safe_mean(
                row["B_correct"]
                for row
                in selector_wrong_rows
            )
        ),

        "nonoracle_acc_selector_wrong": (
            safe_mean(
                row["nonoracle_correct"]
                for row
                in selector_wrong_rows
            )
        ),
    }

    relation_rows = []

    for relation in RELATIONS:
        subset = [
            row
            for row in rows
            if row["relation"] == relation
        ]

        relation_rows.append({
            "relation": relation,
            "N": len(subset),

            "B_acc": safe_mean(
                row["B_correct"]
                for row in subset
            ),

            "selector_acc": safe_mean(
                row["selector_correct"]
                for row in subset
            ),

            "nonoracle_acc": safe_mean(
                row["nonoracle_correct"]
                for row in subset
            ),

            "W2C": int(
                sum(
                    row["W2C"]
                    for row in subset
                )
            ),

            "C2W": int(
                sum(
                    row["C2W"]
                    for row in subset
                )
            ),
        })

    print("\n" + "=" * 150)
    print("ACTUAL GREEDY GENERATION: B MIDDLE-DIRECTION SELECTOR -> ENHANCED-STATE WRITER")
    print("=" * 150)

    print(
        f"N_TEST={len(rows)} | "
        f"selector=L{selector_layer} | "
        f"writer={','.join(str(x) for x in inject_layers)}"
    )

    print(
        f"B no-AdaptVis baseline        : "
        f"{B_acc:.4f}"
    )

    print(
        f"middle direction selector     : "
        f"{selector_acc:.4f}"
    )

    print(
        f"non-oracle selected steering  : "
        f"{nonoracle_acc:.4f} "
        f"({nonoracle_acc - B_acc:+.4f})"
    )

    if args.run_oracle_control:
        print(
            f"oracle writer control         : "
            f"{oracle_acc:.4f}"
        )

    print(
        f"B -> non-oracle steering      : "
        f"W2C={W2C} "
        f"C2W={C2W} "
        f"net={W2C-C2W:+d}"
    )

    print(
        f"selector CORRECT n={len(selector_correct_rows)}: "
        f"generation "
        f"{summary['B_acc_selector_correct']:.4f}"
        f" -> "
        f"{summary['nonoracle_acc_selector_correct']:.4f}"
    )

    print(
        f"selector WRONG   n={len(selector_wrong_rows)}: "
        f"generation "
        f"{summary['B_acc_selector_wrong']:.4f}"
        f" -> "
        f"{summary['nonoracle_acc_selector_wrong']:.4f}"
    )

    print("\nPer relation:")

    for row in relation_rows:
        print(
            f"{row['relation']:>5s} "
            f"N={row['N']:3d} | "
            f"B={row['B_acc']:.4f} | "
            f"selector={row['selector_acc']:.4f} | "
            f"steer={row['nonoracle_acc']:.4f} | "
            f"W2C/C2W={row['W2C']}/{row['C2W']}"
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
        [summary],
    )

    write_csv(
        outdir
        / "per_relation.csv",
        relation_rows,
    )

    write_csv(
        outdir
        / "selector_confusion.csv",
        confusion_rows(rows),
    )

    metadata = {
        "source_vectors": str(src),
        "dataset": "Controlled_Images_A",

        "outer_train_frac": (
            args.train_frac
        ),

        "N_train": len(
            train_positions
        ),

        "N_test": len(
            rows
        ),

        "selector": {
            "representation": (
                "B last-token hidden state"
            ),
            "fit_population": (
                "all outer TRAIN samples"
            ),
            "direction_definition": (
                "normalized centered relation mean"
            ),
            "layer": (
                selector_layer
            ),
            "candidate_layers": (
                candidate_layers
            ),
            "selection": (
                "TRAIN-only repeated internal fit/validation CV"
                if str(args.selector_layer).lower() == "auto"
                else "fixed by command line"
            ),
            "cv_repeats": (
                args.selector_cv_repeats
            ),
            "offline_test_accuracy": (
                selector_test_offline
            ),
        },

        "writer": {
            "representation": (
                "enhanced C last-token hidden state"
            ),
            "fit_population": (
                "C-correct outer TRAIN samples"
            ),
            "direction_definition": (
                "raw centered relation mean offset"
            ),
            "layers": (
                inject_layers
            ),
            "scale": (
                args.state_scale
            ),
        },

        "test_condition": {
            "name": "B",
            "rms_eps": (
                args.enhanced_rms_eps
            ),
            "weight": 1.0,
            "adaptvis": False,
        },

        "test_gt_used_for_routing": False,
        "oracle_control_enabled": (
            args.run_oracle_control
        ),
        "seed": (
            args.seed
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
        f"{outdir / 'selector_confusion.csv'}"
    )

    print(
        f"[saved] "
        f"{outdir / 'selector_and_writer_vectors.npz'}"
    )


if __name__ == "__main__":
    main()
