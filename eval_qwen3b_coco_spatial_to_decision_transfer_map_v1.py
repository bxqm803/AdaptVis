#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-to-layer causal transfer from object-pair spatial states to the prompt-last
(decision-facing) state on COCO-two.

Why this experiment
-------------------
The paper's central mechanistic question is not merely whether a correct spatial
relation is decodable, but whether an upstream spatial state gains causal control
of the downstream decision.

This script uses a natural counterfactual pair:

    original image  <->  horizontal mirror image

for LEFT/RIGHT COCO-two examples.  The prompt and a deterministic sample-specific
random relation->A/B/C/D mapping are held FIXED across the pair.  Therefore the
pair differs in image spatial relation while the textual decision interface is
unchanged.

For every clean pair (both original and mirror are correctly answered by actual
greedy model.generate()):

  1) Capture donor and recipient residual states at selected decoder layers.
  2) At source layer s, patch ONLY the subject+reference text-token residuals
     from donor into recipient.
  3) Let all later layers recompute naturally.
  4) At every target layer t>s, read the prompt-last residual state.
  5) Measure how far the patched prompt-last state moves from the recipient
     toward the donor along the sample-specific donor-vs-recipient axis:

       u_t = h_last^donor(t) - h_last^recipient(t)

       R_state(s,t) =
           <h_last^patch(s->t)-h_last^recipient(t), u_t> / ||u_t||^2

     Interpretation:
       R_state ~ 0 : source object-pair patch does not recover donor decision state
       R_state ~ 1 : it recovers the donor prompt-last state along this axis

  6) Also measure FINAL A/B/C/D donor-option margin recovery:

       R_logit(s) =
         (m_patch - m_recipient) / (m_donor - m_recipient)

     where m = logit(donor option) - logit(recipient option).

A same-size random ordinary-text patch is included as a control by default.

Main outputs
------------
  spatial_to_decision_state_transfer_heatmap.png
      source layer x target layer mean R_state for object-pair patch

  spatial_to_decision_excess_vs_random_heatmap.png
      object-pair R_state minus matched random-text R_state

  source_to_final_option_recovery.png
      source layer -> final donor-option logit recovery

  state_transfer_rows.jsonl
  final_logit_rows.jsonl
  state_transfer_summary.csv
  final_logit_summary.csv
  baseline_pairs.jsonl
  config.json
  errors.jsonl

Dependencies in AdaptVis repo root
----------------------------------
  analyze_coco_centroid_generation_step1_v4.py
  eval_coco_flip_residual_patching_v1.py
  eval_qwen3b_coco_hflip_bothcorrect_threecarrier_handoff_v2.py

Recommended smoke test
----------------------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_coco_spatial_to_decision_transfer_map_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --target-layers all \
  --max-samples 160 \
  --max-clean-pairs 30 \
  --output-dir output/qwen3b_spatial_to_decision_transfer_quick \
  --overwrite

Full run
--------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_coco_spatial_to_decision_transfer_map_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --target-layers all \
  --max-samples 0 \
  --max-clean-pairs 0 \
  --output-dir output/qwen3b_spatial_to_decision_transfer_v1 \
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
import eval_qwen3b_coco_hflip_bothcorrect_threecarrier_handoff_v2 as handoff


VERSION = "qwen3b-coco-spatial-to-decision-transfer-map-v1"
RELATIONS = ("left", "right")
CONDITIONS = ("object_pair", "random_text")


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
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--source-layers",
        default="20,21,22,23,24,25,26",
        help="Source decoder blocks where donor object-pair states are patched.",
    )
    p.add_argument(
        "--target-layers",
        default="all",
        help="Target decoder blocks at which prompt-last residual is captured. Only t>s is used.",
    )
    p.add_argument(
        "--directions",
        default="orig_to_flip,flip_to_orig",
        help="Counterfactual directions to evaluate.",
    )
    p.add_argument(
        "--control",
        default="random_text",
        choices=["random_text", "none"],
        help="Matched-size ordinary-text patch control.",
    )
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all available samples.",
    )
    p.add_argument(
        "--max-clean-pairs",
        type=int,
        default=0,
        help="0 = no cap. Clean = both natural generations correct.",
    )
    p.add_argument("--seed", type=int, default=19)
    p.add_argument(
        "--min-logit-denominator",
        type=float,
        default=1e-6,
    )
    p.add_argument(
        "--min-state-axis-norm",
        type=float,
        default=1e-6,
        help="Skip state recovery when ||h_donor-h_recipient|| is too small.",
    )
    p.add_argument(
        "--clip-display",
        type=float,
        default=1.5,
        help="Only clips heatmap display; raw values are always saved.",
    )
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def span_positions(span: Tuple[int, int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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


def parse_directions(value: str) -> List[str]:
    return handoff.parse_subset(
        value,
        handoff.DIRECTIONS,
        "direction",
    )


def build_patch_conditions(
    *,
    input_ids: Sequence[int],
    visual_positions: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
    special_token_ids: Sequence[int],
    control: str,
    seed: int,
) -> Tuple[List[str], List[List[int]]]:
    """Object-pair patch + matched-size random ordinary-text control."""
    seq_len = len(input_ids)
    visual_set = set(map(int, visual_positions))
    special_ids = set(map(int, special_token_ids))

    object_pair = sorted(
        set(map(int, subject_positions))
        | set(map(int, reference_positions))
    )
    if not object_pair:
        raise RuntimeError("Object-pair token set is empty")

    excluded = set(object_pair) | visual_set | {int(prompt_last)}
    candidates = [
        i
        for i, tok in enumerate(input_ids)
        if i not in excluded
        and int(tok) not in special_ids
    ]

    names = ["object_pair"]
    sets = [object_pair]

    if control == "random_text":
        if len(candidates) < len(object_pair):
            raise RuntimeError(
                f"Need {len(object_pair)} random text positions but only "
                f"{len(candidates)} are available"
            )
        rng = random.Random(int(seed))
        random_pos = sorted(rng.sample(candidates, len(object_pair)))
        names.append("random_text")
        sets.append(random_pos)

    for positions in sets:
        if any(p < 0 or p >= seq_len for p in positions):
            raise RuntimeError("Patch position outside sequence")

    return names, sets


def prompt_last_state(
    capture: Mapping[int, torch.Tensor],
    layer: int,
    batch_index: int,
    prompt_last: int,
) -> torch.Tensor:
    hidden = capture[int(layer)]
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected [B,S,H], got {tuple(hidden.shape)}")
    return hidden[int(batch_index), int(prompt_last), :].detach().float().cpu()


def state_transfer_metrics(
    donor: torch.Tensor,
    recipient: torch.Tensor,
    patched: torch.Tensor,
    min_axis_norm: float,
) -> Optional[Dict[str, float]]:
    """Projection recovery of patched-recipient shift onto donor-recipient axis."""
    axis = donor - recipient
    shift = patched - recipient

    axis_norm = float(torch.linalg.vector_norm(axis))
    shift_norm = float(torch.linalg.vector_norm(shift))

    if axis_norm < float(min_axis_norm):
        return None

    denom = float(torch.dot(axis, axis))
    recovery = float(torch.dot(shift, axis) / denom)

    if shift_norm > 0.0:
        cosine = float(torch.dot(shift, axis) / (shift_norm * axis_norm))
    else:
        cosine = 0.0

    donor_distance = float(torch.linalg.vector_norm(patched - donor))
    recipient_to_donor = axis_norm
    distance_recovery = 1.0 - donor_distance / recipient_to_donor

    return {
        "state_recovery": recovery,
        "shift_axis_cosine": cosine,
        "shift_norm": shift_norm,
        "axis_norm": axis_norm,
        "shift_over_axis_norm": shift_norm / axis_norm,
        "distance_recovery": distance_recovery,
    }


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def summarize_state_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["source_layer"]),
                int(row["target_layer"]),
                str(row["condition"]),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (source, target, condition), items in grouped.items():
        recoveries = finite_values(x.get("state_recovery") for x in items)
        cosines = finite_values(x.get("shift_axis_cosine") for x in items)
        distance_recovery = finite_values(x.get("distance_recovery") for x in items)
        shift_ratios = finite_values(x.get("shift_over_axis_norm") for x in items)

        out.append(
            {
                "source_layer": source,
                "target_layer": target,
                "condition": condition,
                "N": len(items),
                "mean_state_recovery": float(np.mean(recoveries)) if recoveries else None,
                "median_state_recovery": float(np.median(recoveries)) if recoveries else None,
                "std_state_recovery": float(np.std(recoveries)) if recoveries else None,
                "mean_shift_axis_cosine": float(np.mean(cosines)) if cosines else None,
                "mean_distance_recovery": float(np.mean(distance_recovery)) if distance_recovery else None,
                "mean_shift_over_axis_norm": float(np.mean(shift_ratios)) if shift_ratios else None,
                "fraction_state_recovery_gt_0": (
                    float(np.mean([x > 0 for x in recoveries])) if recoveries else None
                ),
                "fraction_state_recovery_gt_0_5": (
                    float(np.mean([x > 0.5 for x in recoveries])) if recoveries else None
                ),
            }
        )

    lookup = {
        (int(r["source_layer"]), int(r["target_layer"]), str(r["condition"])): r
        for r in out
    }
    for row in out:
        if row["condition"] != "object_pair":
            continue
        ctrl = lookup.get(
            (int(row["source_layer"]), int(row["target_layer"]), "random_text")
        )
        if (
            ctrl
            and row["mean_state_recovery"] is not None
            and ctrl["mean_state_recovery"] is not None
        ):
            row["excess_recovery_vs_random"] = (
                float(row["mean_state_recovery"])
                - float(ctrl["mean_state_recovery"])
            )
        else:
            row["excess_recovery_vs_random"] = None

    return sorted(
        out,
        key=lambda r: (
            int(r["source_layer"]),
            int(r["target_layer"]),
            0 if r["condition"] == "object_pair" else 1,
        ),
    )


def summarize_logit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["source_layer"]), str(row["condition"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (source, condition), items in grouped.items():
        recoveries = finite_values(x.get("logit_recovery") for x in items)
        shifts = finite_values(x.get("margin_shift") for x in items)
        donor_rates = [
            str(x.get("patched_prediction")) == str(x.get("donor_option"))
            for x in items
        ]
        recipient_rates = [
            str(x.get("patched_prediction")) == str(x.get("recipient_option"))
            for x in items
        ]

        out.append(
            {
                "source_layer": source,
                "condition": condition,
                "N": len(items),
                "mean_logit_recovery": float(np.mean(recoveries)) if recoveries else None,
                "median_logit_recovery": float(np.median(recoveries)) if recoveries else None,
                "mean_margin_shift": float(np.mean(shifts)) if shifts else None,
                "donor_option_rate": float(np.mean(donor_rates)) if donor_rates else None,
                "recipient_option_rate": float(np.mean(recipient_rates)) if recipient_rates else None,
            }
        )

    lookup = {
        (int(r["source_layer"]), str(r["condition"])): r
        for r in out
    }
    for row in out:
        if row["condition"] != "object_pair":
            continue
        ctrl = lookup.get((int(row["source_layer"]), "random_text"))
        if (
            ctrl
            and row["mean_logit_recovery"] is not None
            and ctrl["mean_logit_recovery"] is not None
        ):
            row["excess_logit_recovery_vs_random"] = (
                float(row["mean_logit_recovery"])
                - float(ctrl["mean_logit_recovery"])
            )
        else:
            row["excess_logit_recovery_vs_random"] = None

    return sorted(
        out,
        key=lambda r: (
            int(r["source_layer"]),
            0 if r["condition"] == "object_pair" else 1,
        ),
    )


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def _matrix_from_summary(
    rows: Sequence[Mapping[str, Any]],
    source_layers: Sequence[int],
    target_layers: Sequence[int],
    field: str,
) -> np.ndarray:
    lookup = {
        (int(r["source_layer"]), int(r["target_layer"])): r.get(field)
        for r in rows
        if r["condition"] == "object_pair"
    }
    matrix = np.full((len(source_layers), len(target_layers)), np.nan, dtype=np.float64)
    for i, s in enumerate(source_layers):
        for j, t in enumerate(target_layers):
            if t <= s:
                continue
            value = lookup.get((int(s), int(t)))
            if value is not None:
                matrix[i, j] = float(value)
    return matrix


def plot_heatmap(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_layers: Sequence[int],
    target_layers: Sequence[int],
    field: str,
    title: str,
    cbar_label: str,
    out_path: Path,
    clip_display: float,
) -> None:
    import matplotlib.pyplot as plt

    matrix = _matrix_from_summary(rows, source_layers, target_layers, field)
    display = matrix.copy()
    if clip_display > 0:
        display = np.clip(display, -float(clip_display), float(clip_display))

    fig_w = max(7.0, 0.38 * len(target_layers) + 3.0)
    fig_h = max(4.5, 0.48 * len(source_layers) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(display, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(target_layers)))
    ax.set_xticklabels([str(x) for x in target_layers], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(source_layers)))
    ax.set_yticklabels([str(x) for x in source_layers])
    ax.set_xlabel("Target layer: prompt-last decision-facing state")
    ax.set_ylabel("Source layer: object-pair patch")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_logit(
    *,
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for condition in ("object_pair", "random_text"):
        items = sorted(
            [r for r in rows if r["condition"] == condition],
            key=lambda r: int(r["source_layer"]),
        )
        if not items:
            continue
        xs = [int(r["source_layer"]) for r in items]
        ys = [
            float(r["mean_logit_recovery"])
            if r["mean_logit_recovery"] is not None
            else np.nan
            for r in items
        ]
        ax.plot(xs, ys, marker="o", label=condition.replace("_", " "))

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Final donor-option logit recovery")
    ax.set_title("Object-pair spatial state -> final option decision")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)
    max_clean_pairs = (
        None if int(args.max_clean_pairs) <= 0 else int(args.max_clean_pairs)
    )
    directions = parse_directions(args.directions)

    base = handoff.import_file(Path(args.base_script), "_transfer_base")
    data_module = base.import_two_object_module()

    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        max_samples,
    )
    prompt_rows = base.load_standard_prompts(Path(args.prompt_jsonl))
    specs = base.merged_model_specs(data_module)

    if args.model not in specs:
        raise ValueError(f"Unknown model {args.model}; available={sorted(specs)}")
    spec = specs[args.model]

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_path = out_dir / "baseline_pairs.jsonl"
    state_path = out_dir / "state_transfer_rows.jsonl"
    logit_path = out_dir / "final_logit_rows.jsonl"
    error_path = out_dir / "errors.jsonl"

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers lacks {spec.model_class}")

    kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}", flush=True)
    print(f"Loading {args.model}: {spec.repo_id}", flush=True)

    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    device = torch.device(args.device)
    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    source_layers = handoff.parse_layers(args.source_layers, len(decoder_layers))
    target_layers = handoff.parse_layers(args.target_layers, len(decoder_layers))
    option_token_map = handoff.build_option_token_map(processor.tokenizer)

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "source_layers": source_layers,
        "target_layers": target_layers,
        "directions": directions,
        "relations": list(RELATIONS),
        "counterfactual": "horizontal reflection, left/right only",
        "decision_space": "sample-specific randomized A/B/C/D, fixed across mirror pair",
        "clean_pair_gate": "both original and mirror actual greedy generations correct",
        "source_intervention": "patch donor subject+reference block-output residual into recipient",
        "target_readout": "prompt-last block-output residual",
        "state_metric": "projection recovery along donor-recipient prompt-last axis",
        "final_metric": "donor-vs-recipient A/B/C/D option-margin recovery",
        "control": args.control,
        "max_samples": max_samples,
        "max_clean_pairs": max_clean_pairs,
        "seed": args.seed,
        "audit": audit,
        "option_token_map": {
            k: list(map(int, v)) for k, v in option_token_map.items()
        },
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Decoder={decoder_path} | n_layers={len(decoder_layers)}\n"
        f"source_layers={source_layers}\n"
        f"target_layers={target_layers}\n"
        f"directions={directions}",
        flush=True,
    )

    seen = 0
    clean = 0
    counts: Counter = Counter()
    start_time = time.time()

    try:
        for record in tqdm(records, desc=f"spatial->decision:{args.model}"):
            if max_clean_pairs is not None and clean >= max_clean_pairs:
                break

            sid = int(record.sid)
            seen += 1
            original_image: Optional[Image.Image] = None
            flipped_image: Optional[Image.Image] = None
            original_batch: Optional[Dict[str, Any]] = None
            flipped_batch: Optional[Dict[str, Any]] = None

            try:
                row = prompt_rows[sid]
                subject = str(row["subject"])
                reference = str(row["reference"])
                original_relation = base.normalize_relation(row["answer_raw"])
                if original_relation not in RELATIONS:
                    continue
                flipped_relation = handoff.OPPOSITE[original_relation]

                mapping = handoff.relation_option_mapping(sid, args.seed)
                original_option = mapping[original_relation]
                flipped_option = mapping[flipped_relation]
                question = handoff.build_random_option_question(
                    subject,
                    reference,
                    mapping,
                )
                rendered = base.build_prompt(processor, question)

                original_image = base.record_image(record).convert("RGB")
                flipped_image = handoff.flip_image(original_image, original_relation)

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

                original_ids = original_batch["input_ids"][0].detach().cpu().tolist()
                flipped_ids = flipped_batch["input_ids"][0].detach().cpu().tolist()
                if original_ids != flipped_ids:
                    raise RuntimeError("Original/flip tokenization differs")

                prompt_last = len(original_ids) - 1
                subject_span, reference_span = base.locate_object_spans(
                    processor.tokenizer,
                    original_ids,
                    subject,
                    reference,
                )
                subject_positions = span_positions(subject_span)
                reference_positions = span_positions(reference_span)

                visual_indices = base.resolve_visual_indices(
                    model,
                    processor,
                    original_batch,
                    original_ids,
                )
                visual_positions = list(map(int, visual_indices))

                condition_names, position_sets = build_patch_conditions(
                    input_ids=original_ids,
                    visual_positions=visual_positions,
                    subject_positions=subject_positions,
                    reference_positions=reference_positions,
                    prompt_last=prompt_last,
                    special_token_ids=getattr(
                        processor.tokenizer,
                        "all_special_ids",
                        [],
                    ),
                    control=args.control,
                    seed=args.seed * 1000003 + sid * 101,
                )

                # Actual generation is used only as the clean-pair gate.
                original_gen_pred, original_gen_text = handoff.generate_option(
                    model=model,
                    processor=processor,
                    batch=original_batch,
                    max_new_tokens=args.max_new_tokens,
                )
                flipped_gen_pred, flipped_gen_text = handoff.generate_option(
                    model=model,
                    processor=processor,
                    batch=flipped_batch,
                    max_new_tokens=args.max_new_tokens,
                )

                baseline_capture_layers = sorted(
                    set(source_layers) | set(target_layers)
                )
                original_results, original_capture = handoff.run_option_forward(
                    model=model,
                    batch=original_batch,
                    option_token_map=option_token_map,
                    decoder_layers=decoder_layers,
                    capture_layers=baseline_capture_layers,
                )
                flipped_results, flipped_capture = handoff.run_option_forward(
                    model=model,
                    batch=flipped_batch,
                    option_token_map=option_token_map,
                    decoder_layers=decoder_layers,
                    capture_layers=baseline_capture_layers,
                )
                original_result = original_results[0]
                flipped_result = flipped_results[0]

                original_correct = original_gen_pred == original_option
                flipped_correct = flipped_gen_pred == flipped_option
                eligible = original_correct and flipped_correct

                counts["eligible_relation_seen"] += 1
                counts["original_correct"] += int(original_correct)
                counts["flip_correct"] += int(flipped_correct)
                counts["both_correct"] += int(eligible)

                append_jsonl(
                    pair_path,
                    {
                        "sid": sid,
                        "subject": subject,
                        "reference": reference,
                        "mapping": handoff.mapping_string(mapping),
                        "original_relation": original_relation,
                        "flipped_relation": flipped_relation,
                        "original_option": original_option,
                        "flipped_option": flipped_option,
                        "original_generation_prediction": original_gen_pred,
                        "flipped_generation_prediction": flipped_gen_pred,
                        "original_generation_text": original_gen_text,
                        "flipped_generation_text": flipped_gen_text,
                        "original_firststep_prediction": original_result["prediction"],
                        "flipped_firststep_prediction": flipped_result["prediction"],
                        "original_correct": original_correct,
                        "flipped_correct": flipped_correct,
                        "eligible": eligible,
                        "subject_span": list(subject_span),
                        "reference_span": list(reference_span),
                        "object_pair_positions": sorted(
                            set(subject_positions) | set(reference_positions)
                        ),
                        "prompt_last": prompt_last,
                    },
                )

                if not eligible:
                    continue
                clean += 1

                # Build recipient repeated batches once per direction.
                repeats = len(condition_names)
                repeated: Dict[str, Dict[str, Any]] = {}
                if "orig_to_flip" in directions:
                    repeated["orig_to_flip"] = handoff.repeated_batch(
                        processor=processor,
                        rendered=rendered,
                        image=flipped_image,
                        repeats=repeats,
                        device=device,
                        base=base,
                    )
                if "flip_to_orig" in directions:
                    repeated["flip_to_orig"] = handoff.repeated_batch(
                        processor=processor,
                        rendered=rendered,
                        image=original_image,
                        repeats=repeats,
                        device=device,
                        base=base,
                    )

                for direction in directions:
                    if direction == "orig_to_flip":
                        donor_option = original_option
                        recipient_option = flipped_option
                        donor_result = original_result
                        recipient_result = flipped_result
                        donor_capture = original_capture
                        recipient_capture = flipped_capture
                    else:
                        donor_option = flipped_option
                        recipient_option = original_option
                        donor_result = flipped_result
                        recipient_result = original_result
                        donor_capture = flipped_capture
                        recipient_capture = original_capture

                    donor_margin = (
                        donor_result["scores"][donor_option]
                        - donor_result["scores"][recipient_option]
                    )
                    recipient_margin = (
                        recipient_result["scores"][donor_option]
                        - recipient_result["scores"][recipient_option]
                    )
                    logit_denom = donor_margin - recipient_margin

                    batch = repeated[direction]
                    if int(batch["input_ids"].shape[1]) != len(original_ids):
                        raise RuntimeError("Repeated batch sequence length differs")

                    for source_layer in source_layers:
                        valid_targets = [
                            t for t in target_layers if int(t) > int(source_layer)
                        ]
                        if not valid_targets:
                            continue

                        with oldpatch.PatchBlockOutput(
                            decoder_layers[source_layer],
                            donor_capture[source_layer],
                            position_sets,
                            len(original_ids),
                        ) as patcher:
                            patched_results, patched_capture = handoff.run_option_forward(
                                model=model,
                                batch=batch,
                                option_token_map=option_token_map,
                                decoder_layers=decoder_layers,
                                capture_layers=valid_targets,
                            )

                        if patcher.events != 1:
                            raise RuntimeError(
                                f"Expected exactly one patch event, got {patcher.events}"
                            )

                        # Final option effect of this source-layer intervention.
                        for condition_index, (condition, patched_result) in enumerate(
                            zip(condition_names, patched_results)
                        ):
                            patched_margin = (
                                patched_result["scores"][donor_option]
                                - patched_result["scores"][recipient_option]
                            )
                            margin_shift = patched_margin - recipient_margin
                            logit_recovery = (
                                margin_shift / logit_denom
                                if abs(logit_denom) >= args.min_logit_denominator
                                else None
                            )
                            append_jsonl(
                                logit_path,
                                {
                                    "sid": sid,
                                    "direction": direction,
                                    "mapping": handoff.mapping_string(mapping),
                                    "original_relation": original_relation,
                                    "flipped_relation": flipped_relation,
                                    "donor_option": donor_option,
                                    "recipient_option": recipient_option,
                                    "source_layer": int(source_layer),
                                    "condition": condition,
                                    "n_positions": len(position_sets[condition_index]),
                                    "donor_margin": donor_margin,
                                    "recipient_margin": recipient_margin,
                                    "patched_margin": patched_margin,
                                    "margin_shift": margin_shift,
                                    "logit_recovery": logit_recovery,
                                    "donor_prediction": donor_result["prediction"],
                                    "recipient_prediction": recipient_result["prediction"],
                                    "patched_prediction": patched_result["prediction"],
                                },
                            )

                        # Layer-to-layer prompt-last state transfer.
                        for target_layer in valid_targets:
                            donor_state = prompt_last_state(
                                donor_capture,
                                target_layer,
                                0,
                                prompt_last,
                            )
                            recipient_state = prompt_last_state(
                                recipient_capture,
                                target_layer,
                                0,
                                prompt_last,
                            )

                            for condition_index, condition in enumerate(condition_names):
                                patched_state = prompt_last_state(
                                    patched_capture,
                                    target_layer,
                                    condition_index,
                                    prompt_last,
                                )
                                metrics = state_transfer_metrics(
                                    donor_state,
                                    recipient_state,
                                    patched_state,
                                    args.min_state_axis_norm,
                                )
                                if metrics is None:
                                    counts["small_state_axis_skipped"] += 1
                                    continue

                                append_jsonl(
                                    state_path,
                                    {
                                        "sid": sid,
                                        "direction": direction,
                                        "mapping": handoff.mapping_string(mapping),
                                        "original_relation": original_relation,
                                        "flipped_relation": flipped_relation,
                                        "donor_option": donor_option,
                                        "recipient_option": recipient_option,
                                        "source_layer": int(source_layer),
                                        "target_layer": int(target_layer),
                                        "condition": condition,
                                        "n_positions": len(position_sets[condition_index]),
                                        **metrics,
                                    },
                                )

                        del patched_capture

                if args.print_every > 0 and clean % args.print_every == 0:
                    tqdm.write(
                        f"\n[clean {clean}] sid={sid} "
                        f"{original_relation}->{flipped_relation} "
                        f"{original_option}->{flipped_option} "
                        f"| object_tokens={len(position_sets[0])}"
                    )

                del original_capture, flipped_capture

                if clean % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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
                tqdm.write(f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}")

            finally:
                for image in (original_image, flipped_image):
                    if image is not None:
                        with contextlib.suppress(Exception):
                            image.close()
                del original_batch, flipped_batch

        # -----------------------------------------------------------------
        # Aggregate + figures
        # -----------------------------------------------------------------
        state_rows = read_jsonl(state_path)
        logit_rows = read_jsonl(logit_path)

        if not state_rows:
            raise RuntimeError(
                "No state-transfer rows were produced. Inspect baseline_pairs.jsonl and errors.jsonl."
            )

        state_summary = summarize_state_rows(state_rows)
        logit_summary = summarize_logit_rows(logit_rows)
        write_csv(out_dir / "state_transfer_summary.csv", state_summary)
        write_csv(out_dir / "final_logit_summary.csv", logit_summary)

        plot_heatmap(
            rows=state_summary,
            source_layers=source_layers,
            target_layers=target_layers,
            field="mean_state_recovery",
            title="Spatial-to-decision causal transfer",
            cbar_label="Mean prompt-last donor-state recovery",
            out_path=out_dir / "spatial_to_decision_state_transfer_heatmap.png",
            clip_display=args.clip_display,
        )

        if args.control == "random_text":
            plot_heatmap(
                rows=state_summary,
                source_layers=source_layers,
                target_layers=target_layers,
                field="excess_recovery_vs_random",
                title="Spatial-to-decision transfer beyond matched random text",
                cbar_label="Object-pair recovery - random-text recovery",
                out_path=out_dir / "spatial_to_decision_excess_vs_random_heatmap.png",
                clip_display=args.clip_display,
            )

        if logit_summary:
            plot_final_logit(
                rows=logit_summary,
                out_path=out_dir / "source_to_final_option_recovery.png",
            )

        elapsed = time.time() - start_time
        summary = {
            "version": VERSION,
            "model": args.model,
            "seen": seen,
            "clean_pairs": clean,
            "counts": dict(counts),
            "elapsed_sec": elapsed,
            "state_rows": len(state_rows),
            "logit_rows": len(logit_rows),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 96)
        print("SPATIAL -> DECISION LAYER-TO-LAYER CAUSAL TRANSFER")
        print("=" * 96)
        print(
            f"model={args.model} | seen={seen} | clean_pairs={clean} | "
            f"state_rows={len(state_rows)} | logit_rows={len(logit_rows)}"
        )

        object_final = [
            r for r in logit_summary if r["condition"] == "object_pair"
        ]
        object_final.sort(
            key=lambda r: -(
                float(r["mean_logit_recovery"])
                if r["mean_logit_recovery"] is not None
                else -999.0
            )
        )
        print("\nTop source layers by final donor-option recovery:")
        for row in object_final[:10]:
            print(
                f"  L{int(row['source_layer']):02d} "
                f"Rlogit={row['mean_logit_recovery']:.4f} "
                f"donor_rate={row['donor_option_rate']:.4f} "
                f"N={int(row['N'])}"
            )

        object_state = [
            r
            for r in state_summary
            if r["condition"] == "object_pair"
            and r["mean_state_recovery"] is not None
        ]
        object_state.sort(
            key=lambda r: -float(r["mean_state_recovery"])
        )
        print("\nTop source->target cells by prompt-last state recovery:")
        for row in object_state[:15]:
            excess = row.get("excess_recovery_vs_random")
            extra = f" excess={float(excess):.4f}" if excess is not None else ""
            print(
                f"  L{int(row['source_layer']):02d}->L{int(row['target_layer']):02d} "
                f"Rstate={float(row['mean_state_recovery']):.4f} "
                f"cos={float(row['mean_shift_axis_cosine']):.4f}"
                f"{extra} N={int(row['N'])}"
            )

        print(f"\nSaved to: {out_dir}")

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
