#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Left/right horizontal-flip counterfactual handoff patching for COCO-two spatial reasoning.

Purpose
-------
Test whether the causal carrier of spatial information shifts with depth:

    Visual tokens  ->  Object text tokens  ->  Prompt-last token

The SAME counterfactual experiment and the SAME decision-recovery metric are
used for all three carriers.

For each LEFT/RIGHT COCO-two example:
  1) Keep the prompt fixed.
  2) Create a counterfactual image by HORIZONTAL reflection so
     left <-> right while object identity and prompt stay fixed.
  3) Use a fixed random relation -> A/B/C/D mapping for BOTH images.
  4) Keep only clean pairs for which BOTH original and horizontally flipped
     inputs are correctly answered by actual greedy model.generate().
  5) At every decoder layer, patch the donor block-output residual into the
     counterfactual recipient at exactly one carrier:
         - all visual-token positions
         - subject + reference text-token positions
         - prompt-last position
  6) Let all later layers recompute naturally.
  7) Measure normalized donor-decision recovery:

        R = (m_patched - m_recipient) / (m_donor - m_recipient)

     where
        m = logit(donor_option) - logit(recipient_option)

Interpretation
--------------
R ~ 0 : patch carries little donor decision information at that layer/site.
R ~ 1 : patch approximately restores the full donor-vs-recipient decision gap.

If a three-stage causal handoff exists, the strongest recovery should move from
Visual -> Object Text -> Prompt Last as layer depth increases.

Outputs
-------
  causal_handoff_heatmap_raw.png
      raw mean normalized recovery, 3 x layers

  causal_handoff_heatmap_row_normalized.png
      each carrier normalized by its own layer-wise maximum.
      This is a timing/localization visualization only, not an effect-size plot.

  causal_handoff_lines.png
      raw mean recovery as three layer-wise curves

  handoff_summary.csv
  patch_results.jsonl
  baseline_pairs.jsonl
  config.json
  errors.jsonl

Dependencies in AdaptVis repo root
----------------------------------
  analyze_coco_centroid_generation_step1_v4.py
  eval_coco_flip_residual_patching_v1.py

This script intentionally reuses the old patching utilities rather than
reimplementing the decoder hooks.

Example quick run
-----------------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_coco_hflip_bothcorrect_threecarrier_handoff_v2.py \
  --model qwen-3b \
  --layers all \
  --max-samples 160 \
  --max-clean-pairs 40 \
  --output-dir output/qwen3b_coco_hflip_bothcorrect_threecarrier_quick \
  --overwrite

Full run
--------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_coco_hflip_bothcorrect_threecarrier_handoff_v2.py \
  --model qwen-3b \
  --layers all \
  --max-samples 0 \
  --max-clean-pairs 0 \
  --output-dir output/qwen3b_coco_hflip_bothcorrect_threecarrier_v2 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import json
import math
import random
import re
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

import eval_coco_flip_residual_patching_v1 as oldpatch


VERSION = "qwen3b-coco-hflip-bothcorrect-threecarrier-handoff-v2"

RELATIONS = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
CARRIERS = ("visual", "object_text", "prompt_last")
CARRIER_LABELS = {
    "visual": "Visual",
    "object_text": "Object Text",
    "prompt_last": "Prompt Last",
}
DIRECTIONS = ("orig_to_flip", "flip_to_orig")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--max-new-tokens", type=int, default=4, help="Greedy-generation length used only for the both-correct clean-pair filter.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--relations",
        default="left,right,above,below",
    )
    p.add_argument(
        "--layers",
        default="all",
        help="'all', 'auto:N', or comma-separated zero-based decoder block indices.",
    )
    p.add_argument(
        "--directions",
        default="orig_to_flip,flip_to_orig",
        help="orig_to_flip, flip_to_orig, or both.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all available dataset samples.",
    )
    p.add_argument(
        "--max-clean-pairs",
        type=int,
        default=0,
        help="0 = no clean-pair cap.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=19,
    )
    p.add_argument(
        "--min-denominator",
        type=float,
        default=1e-4,
        help="Skip direction/layer-independent pair if clean donor-recipient decision gap is too small.",
    )
    p.add_argument(
        "--clip-display",
        type=float,
        default=1.5,
        help="Heatmap display only: clip raw recovery to [-value,+value]. CSV remains unclipped.",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=5,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


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
    allowed_set = set(allowed)
    out: List[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported {label}: {item}; allowed={sorted(allowed_set)}"
            )
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
        raw = raw.strip().lower().replace("l", "")
        if not raw:
            continue
        layer = int(raw)
        if not (0 <= layer < n_layers):
            raise ValueError(
                f"Layer {layer} outside 0..{n_layers - 1}"
            )
        if layer not in out:
            out.append(layer)
    if not out:
        raise ValueError("No layers selected")
    return out


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
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


# -----------------------------------------------------------------------------
# Random A/B/C/D mapping
# -----------------------------------------------------------------------------


def relation_option_mapping(sid: int, seed: int) -> Dict[str, str]:
    """
    Deterministic sample-specific random relation -> A/B/C/D permutation.

    The exact same mapping is used for original and flipped images.
    """
    rng = random.Random(seed * 1000003 + int(sid) * 9176 + 73)
    rels = list(RELATIONS)
    rng.shuffle(rels)

    # LETTER -> relation
    letter_to_relation = {
        letter: relation
        for letter, relation in zip(LETTERS, rels)
    }

    # relation -> LETTER
    return {
        relation: letter
        for letter, relation in letter_to_relation.items()
    }


def mapping_string(mapping: Mapping[str, str]) -> str:
    return ",".join(
        f"{rel}->{mapping[rel]}"
        for rel in RELATIONS
    )


def build_random_option_question(
    subject: str,
    reference: str,
    mapping: Mapping[str, str],
) -> str:
    letter_to_relation = {
        letter: relation
        for relation, letter in mapping.items()
    }

    option_lines = "\n".join(
        f"{letter}. {letter_to_relation[letter]}"
        for letter in LETTERS
    )

    return (
        f"Determine the spatial relation of the {subject} to the {reference} "
        f"in the image.\n"
        f"{option_lines}\n"
        f"Answer with only A, B, C, or D."
    )


# -----------------------------------------------------------------------------
# A/B/C/D next-token scoring
# -----------------------------------------------------------------------------


def build_option_token_map(tokenizer: Any) -> Dict[str, List[int]]:
    """
    Collect single-token variants for A/B/C/D.

    We score the max logit over these variants. The baseline clean-pair filter
    therefore uses the same restricted A/B/C/D decision space as the patch metric.
    """
    out: Dict[str, List[int]] = {}

    for letter in LETTERS:
        candidates = [
            letter,
            " " + letter,
            "\n" + letter,
            "(" + letter + ")",
        ]
        ids: List[int] = []

        for text in candidates:
            enc = tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if len(enc) == 1:
                token_id = int(enc[0])
                if token_id not in ids:
                    ids.append(token_id)

        if not ids:
            enc = tokenizer.encode(
                letter,
                add_special_tokens=False,
            )
            if not enc:
                raise RuntimeError(
                    f"Tokenizer cannot encode option {letter}"
                )
            ids = [int(enc[0])]

        out[letter] = ids

    return out


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "logits",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "logits",
            None,
        ),
    ]

    for value in candidates:
        if (
            torch.is_tensor(value)
            and value.ndim == 3
        ):
            return value

    raise RuntimeError(
        "No language-model logits found"
    )


def score_options(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for letter in LETTERS:
        ids = [
            int(x)
            for x in token_map[letter]
            if 0 <= int(x) < logits.numel()
        ]
        if not ids:
            raise RuntimeError(
                f"No option-token variants for {letter}"
            )

        idx = torch.tensor(
            ids,
            device=logits.device,
            dtype=torch.long,
        )
        scores[letter] = float(
            logits
            .index_select(0, idx)
            .max()
            .detach()
            .cpu()
        )

    return scores


def run_option_forward(
    *,
    model: Any,
    batch: Mapping[str, Any],
    option_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    capture_layers: Sequence[int] = (),
) -> Tuple[List[Dict[str, Any]], Dict[int, torch.Tensor]]:
    """
    Forward pass with optional decoder-block-output capture.
    """
    with torch.inference_mode():
        if capture_layers:
            with oldpatch.CaptureBlockOutputs(
                decoder_layers,
                capture_layers,
            ) as capture:
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
        scores = score_options(
            logits[i],
            option_token_map,
        )
        prediction = max(
            LETTERS,
            key=lambda x: scores[x],
        )
        results.append(
            {
                "scores": scores,
                "prediction": prediction,
            }
        )

    del outputs, logits
    return results, captured



def generate_option(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    """
    Actual greedy generation used ONLY for selecting clean counterfactual pairs.

    The intervention metric below remains the continuous donor-vs-recipient
    A/B/C/D next-token margin, but a sample enters the causal analysis only if
    BOTH the original and horizontally flipped inputs are actually generated
    correctly by model.generate().
    """
    input_len = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        sequences = model.generate(
            **batch,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=False,
            use_cache=True,
        )

    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences

    new_ids = sequences[0, input_len:].detach().cpu().tolist()
    text = processor.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()

    # Prompt explicitly asks for only A/B/C/D.  Prefer a standalone option
    # letter, but keep parsing tolerant of strings such as "A." or "Answer: A".
    match = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper())
    pred = match.group(1) if match else None

    if pred is None and text:
        first = text.lstrip()[:1].upper()
        if first in LETTERS:
            pred = first

    return pred, text


# -----------------------------------------------------------------------------
# Counterfactual image + carrier positions
# -----------------------------------------------------------------------------


def flip_image(
    image: Image.Image,
    relation: str,
) -> Image.Image:
    """Horizontal reflection only; valid only for left/right examples."""
    if relation not in ("left", "right"):
        raise ValueError(
            f"This experiment is left/right only, got relation={relation!r}"
        )
    return image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )


def span_positions(
    span: Tuple[int, int],
) -> List[int]:
    return list(
        range(
            int(span[0]),
            int(span[1]) + 1,
        )
    )


def build_carrier_positions(
    *,
    visual_positions: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
) -> Tuple[List[str], List[List[int]]]:
    visual = sorted(
        set(map(int, visual_positions))
    )
    objects = sorted(
        set(
            map(
                int,
                list(subject_positions)
                + list(reference_positions),
            )
        )
    )
    last = [int(prompt_last)]

    if not visual:
        raise RuntimeError(
            "Visual position set is empty"
        )
    if not objects:
        raise RuntimeError(
            "Object-text position set is empty"
        )

    return (
        ["visual", "object_text", "prompt_last"],
        [visual, objects, last],
    )


def repeated_batch(
    *,
    processor: Any,
    rendered: str,
    image: Image.Image,
    repeats: int,
    device: torch.device,
    base: Any,
) -> Dict[str, Any]:
    """
    Re-process the same image/prompt repeats times.

    This is safer for multimodal processors than manually repeating tensors.
    """
    batch = processor(
        text=[rendered] * repeats,
        images=[image] * repeats,
        return_tensors="pt",
        padding=True,
    )
    return base.move_batch(
        batch,
        device,
    )


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_patch_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        Tuple[int, str],
        List[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                int(row["layer"]),
                str(row["carrier"]),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []

    for (layer, carrier), items in grouped.items():
        recoveries = finite_values(
            x.get("recovery")
            for x in items
        )
        shifts = finite_values(
            x.get("margin_shift")
            for x in items
        )

        donor_rate = float(
            np.mean(
                [
                    x["patched_prediction"]
                    == x["donor_option"]
                    for x in items
                ]
            )
        )

        recipient_rate = float(
            np.mean(
                [
                    x["patched_prediction"]
                    == x["recipient_option"]
                    for x in items
                ]
            )
        )

        out.append(
            {
                "layer": int(layer),
                "carrier": carrier,
                "carrier_label": CARRIER_LABELS[carrier],
                "N": len(items),
                "mean_recovery": (
                    float(np.mean(recoveries))
                    if recoveries
                    else None
                ),
                "median_recovery": (
                    float(np.median(recoveries))
                    if recoveries
                    else None
                ),
                "std_recovery": (
                    float(np.std(recoveries))
                    if recoveries
                    else None
                ),
                "mean_margin_shift": (
                    float(np.mean(shifts))
                    if shifts
                    else None
                ),
                "donor_option_rate": donor_rate,
                "recipient_option_rate": recipient_rate,
                "fraction_recovery_gt_0": (
                    float(
                        np.mean(
                            [
                                x > 0
                                for x in recoveries
                            ]
                        )
                    )
                    if recoveries
                    else None
                ),
                "fraction_recovery_gt_0_5": (
                    float(
                        np.mean(
                            [
                                x > 0.5
                                for x in recoveries
                            ]
                        )
                    )
                    if recoveries
                    else None
                ),
            }
        )

    return sorted(
        out,
        key=lambda x: (
            int(x["layer"]),
            CARRIERS.index(
                str(x["carrier"])
            ),
        ),
    )


def summary_matrix(
    rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    field: str = "mean_recovery",
) -> np.ndarray:
    lookup = {
        (
            int(row["layer"]),
            str(row["carrier"]),
        ): row.get(field)
        for row in rows
    }

    matrix = np.full(
        (len(CARRIERS), len(layers)),
        np.nan,
        dtype=np.float64,
    )

    for i, carrier in enumerate(CARRIERS):
        for j, layer in enumerate(layers):
            value = lookup.get(
                (int(layer), carrier)
            )
            if value is not None:
                try:
                    matrix[i, j] = float(value)
                except (TypeError, ValueError):
                    pass

    return matrix


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_heatmap_raw(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    out_path: Path,
    clip_display: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = summary_matrix(
        summary_rows,
        layers,
        field="mean_recovery",
    )

    display = matrix.copy()
    if (
        clip_display is not None
        and clip_display > 0
    ):
        display = np.clip(
            display,
            -float(clip_display),
            float(clip_display),
        )

    fig, ax = plt.subplots(
        figsize=(11.0, 3.6)
    )

    im = ax.imshow(
        display,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(
        np.arange(len(CARRIERS))
    )
    ax.set_yticklabels(
        [CARRIER_LABELS[x] for x in CARRIERS]
    )

    ax.set_xticks(
        np.arange(len(layers))
    )
    ax.set_xticklabels(
        [str(x) for x in layers],
        rotation=0,
    )

    ax.set_xlabel("Decoder Layer")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(
        "Counterfactual Recovery"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_heatmap_row_normalized(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    out_path: Path,
) -> None:
    """
    Presentation-only localization map.

    Each carrier is normalized by its own maximum positive mean recovery.
    This shows WHEN each carrier is maximally causal, not absolute effect size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = summary_matrix(
        summary_rows,
        layers,
        field="mean_recovery",
    )

    norm = np.full_like(
        raw,
        np.nan,
        dtype=np.float64,
    )

    for i in range(raw.shape[0]):
        row = raw[i].copy()
        finite = np.isfinite(row)
        if not finite.any():
            continue

        # For localization, negative recovery is not treated as positive evidence.
        positive = np.where(
            finite,
            np.maximum(row, 0.0),
            np.nan,
        )
        max_value = np.nanmax(positive)

        if (
            np.isfinite(max_value)
            and max_value > 1e-12
        ):
            norm[i] = positive / max_value
        else:
            norm[i] = positive

    fig, ax = plt.subplots(
        figsize=(11.0, 3.6)
    )

    im = ax.imshow(
        norm,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_yticks(
        np.arange(len(CARRIERS))
    )
    ax.set_yticklabels(
        [CARRIER_LABELS[x] for x in CARRIERS]
    )

    ax.set_xticks(
        np.arange(len(layers))
    )
    ax.set_xticklabels(
        [str(x) for x in layers],
        rotation=0,
    )

    ax.set_xlabel("Decoder Layer")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(
        "Within-Carrier Normalized Recovery"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_lines(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = summary_matrix(
        summary_rows,
        layers,
        field="mean_recovery",
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 5.0)
    )

    for i, carrier in enumerate(CARRIERS):
        ax.plot(
            layers,
            matrix[i],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=CARRIER_LABELS[carrier],
        )

    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )
    ax.set_xlabel(
        "Decoder Layer"
    )
    ax.set_ylabel(
        "Counterfactual Recovery"
    )
    ax.legend(
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Main causal-localization protocol:
    # use only left/right examples so the counterfactual is a natural
    # horizontal reflection.  Above/below + vertical reflection is
    # intentionally excluded to avoid the stronger distribution shift.
    relations = ["left", "right"]
    directions = parse_subset(
        args.directions,
        DIRECTIONS,
        "direction",
    )

    max_samples = (
        None
        if int(args.max_samples) <= 0
        else int(args.max_samples)
    )
    max_clean_pairs = (
        None
        if int(args.max_clean_pairs) <= 0
        else int(args.max_clean_pairs)
    )

    base = import_file(
        Path(args.base_script),
        "_handoff_base",
    )

    data_module = (
        base.import_two_object_module()
    )

    records, audit = (
        data_module.load_records(
            args.dataset,
            Path(args.data_root),
            max_samples,
        )
    )

    prompt_rows = (
        base.load_standard_prompts(
            Path(args.prompt_jsonl)
        )
    )

    specs = base.merged_model_specs(
        data_module
    )

    if args.model not in specs:
        raise ValueError(
            f"Unknown model {args.model}; "
            f"available={sorted(specs)}"
        )

    spec = specs[args.model]

    out_dir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and out_dir.exists()
    ):
        shutil.rmtree(
            out_dir
        )

    if (
        out_dir.exists()
        and any(out_dir.iterdir())
    ):
        raise RuntimeError(
            f"Output directory not empty: {out_dir}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_path = (
        out_dir
        / "baseline_pairs.jsonl"
    )
    patch_path = (
        out_dir
        / "patch_results.jsonl"
    )
    error_path = (
        out_dir
        / "errors.jsonl"
    )

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
    }

    if args.attn_impl != "none":
        kwargs[
            "attn_implementation"
        ] = args.attn_impl

    print(
        f"Version: {VERSION}",
        flush=True,
    )
    print(
        f"Loading {args.model}: "
        f"{spec.repo_id}",
        flush=True,
    )

    model = model_cls.from_pretrained(
        spec.repo_id,
        **kwargs,
    )
    model.eval()

    processor = (
        AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=(
                spec.trust_remote_code
            ),
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
        base.resolve_decoder_layers(
            model
        )
    )

    layers = parse_layers(
        args.layers,
        len(decoder_layers),
    )

    option_token_map = (
        build_option_token_map(
            processor.tokenizer
        )
    )

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(
            decoder_layers
        ),
        "layers": layers,
        "carriers": list(CARRIERS),
        "directions": directions,
        "relations": ["left", "right"],
        "counterfactual": "horizontal reflection only",
        "max_samples": max_samples,
        "max_clean_pairs": max_clean_pairs,
        "seed": args.seed,
        "audit": audit,
        "decision_space": "randomized A/B/C/D",
        "pair_filter": (
            "both original and horizontally flipped ACTUAL greedy generations correct"
        ),
        "patch_location": (
            "decoder block output residual"
        ),
        "recovery_metric": (
            "(patched donor-vs-recipient option margin - recipient margin) "
            "/ (donor margin - recipient margin)"
        ),
        "visual_patch": (
            "all visual-token residual positions"
        ),
        "object_text_patch": (
            "subject + reference text-token residual positions"
        ),
        "prompt_last_patch": (
            "single prompt-final residual position"
        ),
        "option_token_map": {
            k: list(map(int, v))
            for k, v
            in option_token_map.items()
        },
    }

    (
        out_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Decoder={decoder_path} | "
        f"n_layers={len(decoder_layers)}",
        flush=True,
    )
    print(
        f"scan layers={layers}",
        flush=True,
    )
    print(
        f"directions={directions}",
        flush=True,
    )
    print(
        f"option token map={option_token_map}",
        flush=True,
    )

    seen = 0
    clean = 0
    counts: Counter = Counter()
    start_time = time.time()

    try:
        for record in tqdm(
            records,
            desc=f"hflip-bothcorrect-threecarrier:{args.model}",
        ):
            if (
                max_clean_pairs is not None
                and clean >= max_clean_pairs
            ):
                break

            sid = int(record.sid)
            seen += 1

            original_image = None
            flipped_image = None
            original_batch = None
            flipped_batch = None

            try:
                row = prompt_rows[sid]

                subject = str(
                    row["subject"]
                )
                reference = str(
                    row["reference"]
                )

                original_relation = (
                    base.normalize_relation(
                        row["answer_raw"]
                    )
                )

                if (
                    original_relation
                    not in relations
                ):
                    continue

                flipped_relation = (
                    OPPOSITE[
                        original_relation
                    ]
                )

                axis = "horizontal"

                mapping = (
                    relation_option_mapping(
                        sid,
                        args.seed,
                    )
                )

                original_option = (
                    mapping[
                        original_relation
                    ]
                )
                flipped_option = (
                    mapping[
                        flipped_relation
                    ]
                )

                question = (
                    build_random_option_question(
                        subject,
                        reference,
                        mapping,
                    )
                )

                rendered = (
                    base.build_prompt(
                        processor,
                        question,
                    )
                )

                original_image = (
                    base.record_image(
                        record
                    )
                    .convert("RGB")
                )

                flipped_image = (
                    flip_image(
                        original_image,
                        original_relation,
                    )
                )

                original_batch = (
                    base.move_batch(
                        processor(
                            text=[rendered],
                            images=[
                                original_image
                            ],
                            return_tensors="pt",
                        ),
                        device,
                    )
                )

                flipped_batch = (
                    base.move_batch(
                        processor(
                            text=[rendered],
                            images=[
                                flipped_image
                            ],
                            return_tensors="pt",
                        ),
                        device,
                    )
                )

                original_ids = (
                    original_batch[
                        "input_ids"
                    ][0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                flipped_ids = (
                    flipped_batch[
                        "input_ids"
                    ][0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                # Actual generation is the clean-pair gate.
                # We deliberately do this before any activation patching.
                original_gen_pred, original_gen_text = generate_option(
                    model=model,
                    processor=processor,
                    batch=original_batch,
                    max_new_tokens=args.max_new_tokens,
                )
                flipped_gen_pred, flipped_gen_text = generate_option(
                    model=model,
                    processor=processor,
                    batch=flipped_batch,
                    max_new_tokens=args.max_new_tokens,
                )

                if (
                    original_ids
                    != flipped_ids
                ):
                    raise RuntimeError(
                        "Original/flip tokenization differs"
                    )

                (
                    subject_span,
                    reference_span,
                ) = (
                    base.locate_object_spans(
                        processor.tokenizer,
                        original_ids,
                        subject,
                        reference,
                    )
                )

                subject_positions = (
                    span_positions(
                        subject_span
                    )
                )
                reference_positions = (
                    span_positions(
                        reference_span
                    )
                )

                prompt_last = (
                    len(original_ids)
                    - 1
                )

                visual_indices = (
                    base.resolve_visual_indices(
                        model,
                        processor,
                        original_batch,
                        original_ids,
                    )
                )

                visual_positions = (
                    list(
                        map(
                            int,
                            visual_indices,
                        )
                    )
                )

                (
                    carriers,
                    position_sets,
                ) = build_carrier_positions(
                    visual_positions=(
                        visual_positions
                    ),
                    subject_positions=(
                        subject_positions
                    ),
                    reference_positions=(
                        reference_positions
                    ),
                    prompt_last=(
                        prompt_last
                    ),
                )

                (
                    original_results,
                    original_capture,
                ) = run_option_forward(
                    model=model,
                    batch=original_batch,
                    option_token_map=(
                        option_token_map
                    ),
                    decoder_layers=(
                        decoder_layers
                    ),
                    capture_layers=layers,
                )

                (
                    flipped_results,
                    flipped_capture,
                ) = run_option_forward(
                    model=model,
                    batch=flipped_batch,
                    option_token_map=(
                        option_token_map
                    ),
                    decoder_layers=(
                        decoder_layers
                    ),
                    capture_layers=layers,
                )

                original_result = (
                    original_results[0]
                )
                flipped_result = (
                    flipped_results[0]
                )

                original_pred = (
                    original_result[
                        "prediction"
                    ]
                )
                flipped_pred = (
                    flipped_result[
                        "prediction"
                    ]
                )

                # Restricted first-step predictions are retained for auditing and
                # for the continuous recovery metric.  Clean-pair eligibility,
                # however, is based on actual greedy generation.
                original_correct = (
                    original_gen_pred
                    == original_option
                )
                flipped_correct = (
                    flipped_gen_pred
                    == flipped_option
                )

                counts[
                    "eligible_relation_seen"
                ] += 1
                counts[
                    "original_correct"
                ] += int(
                    original_correct
                )
                counts[
                    "flip_correct"
                ] += int(
                    flipped_correct
                )
                counts[
                    "both_correct"
                ] += int(
                    original_correct
                    and flipped_correct
                )

                eligible = (
                    original_correct
                    and flipped_correct
                )

                append_jsonl(
                    pair_path,
                    {
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "axis": axis,
                        "mapping": (
                            mapping_string(
                                mapping
                            )
                        ),
                        "original_relation": (
                            original_relation
                        ),
                        "flipped_relation": (
                            flipped_relation
                        ),
                        "original_option": (
                            original_option
                        ),
                        "flipped_option": (
                            flipped_option
                        ),
                        "original_generation_prediction": (
                            original_gen_pred
                        ),
                        "flipped_generation_prediction": (
                            flipped_gen_pred
                        ),
                        "original_generation_text": (
                            original_gen_text
                        ),
                        "flipped_generation_text": (
                            flipped_gen_text
                        ),
                        "original_restricted_firststep_prediction": (
                            original_pred
                        ),
                        "flipped_restricted_firststep_prediction": (
                            flipped_pred
                        ),
                        "original_correct": (
                            original_correct
                        ),
                        "flipped_correct": (
                            flipped_correct
                        ),
                        "eligible": eligible,
                        "original_scores": (
                            original_result[
                                "scores"
                            ]
                        ),
                        "flipped_scores": (
                            flipped_result[
                                "scores"
                            ]
                        ),
                        "subject_span": (
                            list(
                                subject_span
                            )
                        ),
                        "reference_span": (
                            list(
                                reference_span
                            )
                        ),
                        "n_visual_positions": (
                            len(
                                visual_positions
                            )
                        ),
                        "n_object_positions": (
                            len(
                                set(
                                    subject_positions
                                    + reference_positions
                                )
                            )
                        ),
                        "prompt_last": (
                            prompt_last
                        ),
                    },
                )

                if not eligible:
                    continue

                clean += 1

                # -----------------------------------------------------
                # Counterfactual patching
                # -----------------------------------------------------
                for direction in directions:
                    if (
                        direction
                        == "orig_to_flip"
                    ):
                        donor_option = (
                            original_option
                        )
                        recipient_option = (
                            flipped_option
                        )
                        donor_result = (
                            original_result
                        )
                        recipient_result = (
                            flipped_result
                        )
                        donor_capture = (
                            original_capture
                        )
                        recipient_image = (
                            flipped_image
                        )
                    else:
                        donor_option = (
                            flipped_option
                        )
                        recipient_option = (
                            original_option
                        )
                        donor_result = (
                            flipped_result
                        )
                        recipient_result = (
                            original_result
                        )
                        donor_capture = (
                            flipped_capture
                        )
                        recipient_image = (
                            original_image
                        )

                    donor_margin = (
                        donor_result[
                            "scores"
                        ][donor_option]
                        - donor_result[
                            "scores"
                        ][recipient_option]
                    )

                    recipient_margin = (
                        recipient_result[
                            "scores"
                        ][donor_option]
                        - recipient_result[
                            "scores"
                        ][recipient_option]
                    )

                    denominator = (
                        donor_margin
                        - recipient_margin
                    )

                    if (
                        abs(denominator)
                        < args.min_denominator
                    ):
                        counts[
                            "small_denominator_skipped"
                        ] += 1
                        continue

                    # Three patch conditions are evaluated in one batched forward.
                    recipient_repeated = (
                        repeated_batch(
                            processor=processor,
                            rendered=rendered,
                            image=recipient_image,
                            repeats=len(
                                carriers
                            ),
                            device=device,
                            base=base,
                        )
                    )

                    repeated_ids = (
                        recipient_repeated[
                            "input_ids"
                        ]
                    )

                    if (
                        repeated_ids.shape[1]
                        != len(
                            original_ids
                        )
                    ):
                        raise RuntimeError(
                            "Repeated batch sequence length differs"
                        )

                    for layer in layers:
                        with (
                            oldpatch.PatchBlockOutput(
                                decoder_layers[
                                    layer
                                ],
                                donor_capture[
                                    layer
                                ],
                                position_sets,
                                len(
                                    original_ids
                                ),
                            )
                        ) as patcher:
                            (
                                patched_results,
                                _,
                            ) = (
                                run_option_forward(
                                    model=model,
                                    batch=(
                                        recipient_repeated
                                    ),
                                    option_token_map=(
                                        option_token_map
                                    ),
                                    decoder_layers=(
                                        decoder_layers
                                    ),
                                    capture_layers=(),
                                )
                            )

                        if (
                            patcher.events
                            != 1
                        ):
                            raise RuntimeError(
                                "Expected exactly one patch event, "
                                f"got {patcher.events}"
                            )

                        for (
                            carrier,
                            result,
                            positions,
                        ) in zip(
                            carriers,
                            patched_results,
                            position_sets,
                        ):
                            patched_margin = (
                                result[
                                    "scores"
                                ][donor_option]
                                - result[
                                    "scores"
                                ][recipient_option]
                            )

                            margin_shift = (
                                patched_margin
                                - recipient_margin
                            )

                            recovery = (
                                margin_shift
                                / denominator
                            )

                            append_jsonl(
                                patch_path,
                                {
                                    "sid": sid,
                                    "axis": axis,
                                    "direction": (
                                        direction
                                    ),
                                    "mapping": (
                                        mapping_string(
                                            mapping
                                        )
                                    ),
                                    "original_relation": (
                                        original_relation
                                    ),
                                    "flipped_relation": (
                                        flipped_relation
                                    ),
                                    "donor_option": (
                                        donor_option
                                    ),
                                    "recipient_option": (
                                        recipient_option
                                    ),
                                    "layer": (
                                        int(
                                            layer
                                        )
                                    ),
                                    "carrier": (
                                        carrier
                                    ),
                                    "n_positions": (
                                        len(
                                            positions
                                        )
                                    ),
                                    "donor_prediction": (
                                        donor_result[
                                            "prediction"
                                        ]
                                    ),
                                    "recipient_prediction": (
                                        recipient_result[
                                            "prediction"
                                        ]
                                    ),
                                    "patched_prediction": (
                                        result[
                                            "prediction"
                                        ]
                                    ),
                                    "donor_margin": (
                                        donor_margin
                                    ),
                                    "recipient_margin": (
                                        recipient_margin
                                    ),
                                    "patched_margin": (
                                        patched_margin
                                    ),
                                    "margin_shift": (
                                        margin_shift
                                    ),
                                    "recovery": (
                                        recovery
                                    ),
                                },
                            )

                    del recipient_repeated

                if (
                    args.print_every > 0
                    and clean
                    % args.print_every
                    == 0
                ):
                    tqdm.write(
                        f"\n[clean {clean}] "
                        f"sid={sid} "
                        f"{original_relation}->{flipped_relation} "
                        f"{original_option}->{flipped_option} "
                        f"| visual={len(visual_positions)} "
                        f"object={len(set(subject_positions + reference_positions))}"
                    )

                del (
                    original_capture,
                    flipped_capture,
                )

                if (
                    clean % 10
                    == 0
                ):
                    gc.collect()
                    if (
                        torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    },
                )
                tqdm.write(
                    f"\n[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                for image in (
                    original_image,
                    flipped_image,
                ):
                    if image is not None:
                        with contextlib.suppress(
                            Exception
                        ):
                            image.close()

                del (
                    original_batch,
                    flipped_batch,
                )

        # -------------------------------------------------------------
        # Aggregate + plot
        # -------------------------------------------------------------
        patch_rows = read_jsonl(
            patch_path
        )

        if not patch_rows:
            raise RuntimeError(
                "No patch rows were produced. "
                "Inspect baseline_pairs.jsonl and errors.jsonl."
            )

        summary_rows = (
            summarize_patch_rows(
                patch_rows
            )
        )

        write_csv(
            out_dir
            / "handoff_summary.csv",
            summary_rows,
        )

        plot_heatmap_raw(
            summary_rows=summary_rows,
            layers=layers,
            out_path=(
                out_dir
                / "causal_handoff_heatmap_raw.png"
            ),
            clip_display=(
                args.clip_display
            ),
        )

        plot_heatmap_row_normalized(
            summary_rows=summary_rows,
            layers=layers,
            out_path=(
                out_dir
                / "causal_handoff_heatmap_row_normalized.png"
            ),
        )

        plot_lines(
            summary_rows=summary_rows,
            layers=layers,
            out_path=(
                out_dir
                / "causal_handoff_lines.png"
            ),
        )

        # Peak layer per carrier, useful for a one-line sanity check.
        peaks: Dict[str, Dict[str, Any]] = {}

        for carrier in CARRIERS:
            candidates = [
                r
                for r in summary_rows
                if r["carrier"]
                == carrier
                and r["mean_recovery"]
                is not None
            ]

            if candidates:
                best = max(
                    candidates,
                    key=lambda r: float(
                        r[
                            "mean_recovery"
                        ]
                    ),
                )
                peaks[carrier] = {
                    "layer": int(
                        best["layer"]
                    ),
                    "mean_recovery": float(
                        best[
                            "mean_recovery"
                        ]
                    ),
                }

        summary_json = {
            "version": VERSION,
            "seen": seen,
            "clean_pairs": clean,
            "baseline_counts": (
                dict(counts)
            ),
            "n_patch_rows": (
                len(patch_rows)
            ),
            "peak_layer_by_carrier": (
                peaks
            ),
            "elapsed_minutes": (
                time.time()
                - start_time
            )
            / 60.0,
        }

        (
            out_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary_json,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\n"
            + "=" * 100
        )
        print(
            "THREE-CARRIER COUNTERFACTUAL HANDOFF"
        )
        print(
            "=" * 100
        )
        print(
            f"seen={seen} | "
            f"clean_pairs={clean}"
        )
        print(
            "baseline_counts="
            + json.dumps(
                dict(counts),
                ensure_ascii=False,
            )
        )

        print(
            "\nPeak layer by carrier:"
        )

        for carrier in CARRIERS:
            p = peaks.get(carrier)
            if p is None:
                print(
                    f"  {CARRIER_LABELS[carrier]:<12}: no valid rows"
                )
            else:
                print(
                    f"  {CARRIER_LABELS[carrier]:<12}: "
                    f"L{p['layer']:02d} "
                    f"meanR={p['mean_recovery']:+.4f}"
                )

        print(
            "\nSaved:"
        )
        for name in (
            "causal_handoff_heatmap_raw.png",
            "causal_handoff_heatmap_row_normalized.png",
            "causal_handoff_lines.png",
            "handoff_summary.csv",
            "summary.json",
            "baseline_pairs.jsonl",
            "patch_results.jsonl",
            "config.json",
        ):
            print(
                " ",
                out_dir / name,
            )

        if error_path.exists():
            print(
                " ",
                error_path,
            )

    finally:
        del model, processor
        gc.collect()
        if (
            torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
