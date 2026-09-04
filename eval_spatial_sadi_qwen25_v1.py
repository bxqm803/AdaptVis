#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spatial_sadi_qwen25_v1.py

Spatial-SADI: relation-sensitive mask + sample-specific Real-Gray residual steering.

Motivation
----------
Plain SADI-style mask identification used:

    D_l = E_train[h_real - h_gray]

For a symmetric spatial task this can cancel the most useful dimensions:
    left  = +a
    right = -a
=> global mean ~ 0.

Spatial-SADI instead explicitly identifies dimensions that discriminate
the two spatial axes:

    Delta_i,l = h_real(i,l,last) - h_gray(i,l,last)

    mu_L,l = E[Delta | left]
    mu_R,l = E[Delta | right]
    mu_A,l = E[Delta | above]
    mu_B,l = E[Delta | below]

    d_x,l = mu_L,l - mu_R,l
    d_y,l = mu_A,l - mu_B,l

Main spatial importance:

    importance_l,j = sqrt(d_x,l,j^2 + d_y,l,j^2)

Then take top-K dimensions across the late trajectory:

    M_l,j in {0,1}

At inference there is NO 4-way selector.

For each sample i:

    q_i,l = Delta_i,l - mu_global,l

where:

    mu_global,l = 1/4 (mu_L + mu_R + mu_A + mu_B)

Then actual generate() is edited as:

    h'_i,l,last
      = h_i,l,last
        + alpha * (q_i,l * M_l)

Thus the direction/sign comes from the CURRENT SAMPLE'S OWN Real-Gray residual,
while TRAIN only identifies which dimensions are spatial-relation-sensitive.

No:
    - L/R/A/B classifier
    - cosine routing
    - oracle relation at TEST
    - discrete direction selection

Protocol
--------
Original TRAIN -> FIT + CAL
  FIT:
    - fit relation means
    - build spatial mask
  CAL:
    - select top-K and alpha using actual model.generate()
  FULL TRAIN:
    - refit mask with selected top-K
  TEST:
    - evaluate exactly once

The existing TEST baseline is reused, so TEST baseline should remain e.g. 0.6526
for the Qwen3B COCO split.

This script reuses utilities from:
    eval_cosine_confidence_selector_qwen25_v1.py

Example
-------
CUDA_VISIBLE_DEVICES=0 python eval_spatial_sadi_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --prior-output-dir output/qwen3b_last_causal_auto_v1 \
  --cache-deltas output/qwen3b_real_gray_increment_similarity_v1/real_gray_last_deltas.npz \
  --layers auto \
  --topk-counts 64,128,256,512 \
  --strengths 0.1,0.25,0.5,1.0 \
  --sample-mode centered \
  --mask-metric l2 \
  --mask-allocation global \
  --output-dir output/qwen3b_spatial_sadi_v1 \
  --overwrite

Recommended first run:
    --sample-mode centered
    --mask-metric l2
    --mask-allocation global

Useful diagnostics:
    --sample-mode raw
        inject raw Real-Gray residual instead of subtracting mu_global

    --mask-metric max
        importance = max(|d_x|, |d_y|)

    --mask-allocation balanced
        distribute top-K approximately equally over selected layers instead
        of selecting top-K globally across all layer x hidden coordinates.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import eval_cosine_confidence_selector_qwen25_v1 as base


RELS = base.RELS
RELSET = set(RELS)
EPS = 1e-12


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--model-id", required=True)
    p.add_argument("--prior-output-dir", required=True)

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--annotation-json",
        default="data/coco_qa_two_obj.json",
    )
    p.add_argument("--data-root", default="data")

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument(
        "--layers",
        default="auto",
        help=(
            "Layers to edit. auto uses known late actuator window: "
            "Qwen3B=L32-35, Qwen7B=L25-27."
        ),
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
        help=(
            "TRAIN filter used for relation means. "
            "The existing helper automatically relaxes the filter if a "
            "relation cell is empty."
        ),
    )

    p.add_argument(
        "--mask-metric",
        default="l2",
        choices=["l2", "max", "sumabs"],
        help=(
            "How to combine horizontal and vertical relation contrasts: "
            "l2=sqrt(dx^2+dy^2), max=max(|dx|,|dy|), "
            "sumabs=|dx|+|dy|."
        ),
    )

    p.add_argument(
        "--mask-allocation",
        default="global",
        choices=["global", "balanced"],
        help=(
            "global: choose top-K over all selected layer x hidden coordinates. "
            "balanced: allocate K approximately equally across selected layers."
        ),
    )

    p.add_argument(
        "--sample-mode",
        default="centered",
        choices=["centered", "raw"],
        help=(
            "centered [recommended]: q_i = Delta_i - mu_global. "
            "raw: q_i = Delta_i."
        ),
    )

    p.add_argument(
        "--topk-counts",
        default="64,128,256,512",
        help="Global number of spatial-sensitive coordinates scanned on CAL.",
    )
    p.add_argument(
        "--strengths",
        default="0.1,0.25,0.5,1.0",
        help="Intervention alpha scanned on CAL.",
    )

    p.add_argument(
        "--fixed-topk",
        type=int,
        default=0,
        help="If >0, use this top-K and skip top-K CAL search.",
    )
    p.add_argument(
        "--fixed-strength",
        type=float,
        default=float("nan"),
        help=(
            "Together with --fixed-topk, use this alpha and skip CAL search."
        ),
    )

    p.add_argument("--cal-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument(
        "--cache-deltas",
        default="",
        help=(
            "Optional existing Real-Gray last-token cache. "
            "It may contain a superset of layers (e.g. L18-35). "
            "For lowest compute, use the cache made by the previous "
            "Real-Gray increment diagnostic."
        ),
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def parse_ints(spec):
    return [
        int(x.strip())
        for x in str(spec).split(",")
        if x.strip()
    ]


def parse_floats(spec):
    return [
        float(x.strip())
        for x in str(spec).split(",")
        if x.strip()
    ]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_layers(args, decoder_layers):
    if str(args.layers).strip().lower() == "auto":
        selected = list(
            base.model_preset(args.model_id)["actuator_layers"]
        )
    else:
        selected = base.parse_layer_spec(args.layers)

    selected = sorted(set(selected))

    if not selected:
        raise ValueError("No selected layers.")

    bad = [
        layer
        for layer in selected
        if layer < 0 or layer >= len(decoder_layers)
    ]

    if bad:
        raise ValueError(
            f"Invalid layers={bad}; decoder blocks=0..{len(decoder_layers)-1}"
        )

    return selected


# =============================================================================
# Real-Gray delta cache
# =============================================================================

def load_cache_subset(
    cache_path,
    selected_layers,
    required_sids,
):
    if not cache_path:
        return None

    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        cache_sids = np.asarray(
            z["sample_index"],
            dtype=np.int64,
        )
        cache_layers = np.asarray(
            z["layers"],
            dtype=np.int64,
        ).tolist()
        cache_deltas = np.asarray(
            z["deltas"],
            dtype=np.float32,
        )

    layer_to_col = {
        int(layer): idx
        for idx, layer in enumerate(cache_layers)
    }

    missing_layers = [
        layer
        for layer in selected_layers
        if layer not in layer_to_col
    ]

    if missing_layers:
        raise RuntimeError(
            f"Cache lacks selected layers={missing_layers}; "
            f"available={cache_layers}"
        )

    sid_to_row = {
        int(sid): idx
        for idx, sid in enumerate(cache_sids.tolist())
    }

    missing_sids = [
        int(sid)
        for sid in required_sids
        if int(sid) not in sid_to_row
    ]

    if missing_sids:
        raise RuntimeError(
            f"Cache lacks {len(missing_sids)} required samples. "
            f"First missing={missing_sids[:10]}"
        )

    cols = [
        layer_to_col[layer]
        for layer in selected_layers
    ]

    result = {
        int(sid): cache_deltas[
            sid_to_row[int(sid)]
        ][cols].astype(np.float32)
        for sid in required_sids
    }

    print(
        f"[reuse] Real-Gray cache={path} | "
        f"cached_layers={cache_layers} -> selected={selected_layers} | "
        f"N={len(result)}"
    )

    return result


def get_delta_cache(
    model,
    processor,
    decoder_layers,
    records,
    selected_layers,
    args,
    outdir,
):
    required_sids = sorted(records)

    cached = load_cache_subset(
        args.cache_deltas,
        selected_layers,
        required_sids,
    )

    if cached is not None:
        return cached

    cache_path = (
        outdir
        / "all_samples_real_gray_last_deltas.npz"
    )

    return base.build_or_load_delta_cache(
        model=model,
        processor=processor,
        layers=decoder_layers,
        records=records,
        selected_layers=selected_layers,
        args=args,
        cache_path=cache_path,
    )


# =============================================================================
# Relation-sensitive mask
# =============================================================================

def fit_spatial_statistics(
    delta_cache,
    layer_index,
    records,
    fit_sids,
    selected_layers,
    train_generation,
    requested_filter,
):
    """
    Reuse the repository's relation-template fitter.

    It computes:
        global = balanced mean of four relation means
        shared[r] = mu_r - global

    Then:
        dx = mu_L - mu_R
           = shared[L] - shared[R]

        dy = mu_A - mu_B
           = shared[A] - shared[B]
    """

    templates, filter_used = base.fit_templates(
        delta_cache=delta_cache,
        layer_index=layer_index,
        records=records,
        sids=fit_sids,
        selected_layers=selected_layers,
        train_generation=train_generation,
        requested_filter=requested_filter,
    )

    stats = {}

    for layer in selected_layers:
        shared = templates[layer]["shared"]

        dx = (
            shared["left"].astype(np.float32)
            - shared["right"].astype(np.float32)
        )
        dy = (
            shared["above"].astype(np.float32)
            - shared["below"].astype(np.float32)
        )

        if args_global.mask_metric == "l2":
            importance = np.sqrt(
                dx.astype(np.float64) ** 2
                + dy.astype(np.float64) ** 2
            ).astype(np.float32)

        elif args_global.mask_metric == "max":
            importance = np.maximum(
                np.abs(dx),
                np.abs(dy),
            ).astype(np.float32)

        elif args_global.mask_metric == "sumabs":
            importance = (
                np.abs(dx)
                + np.abs(dy)
            ).astype(np.float32)

        else:
            raise ValueError(
                args_global.mask_metric
            )

        stats[layer] = {
            "global": templates[layer]["global"].astype(
                np.float32
            ),
            "dx": dx,
            "dy": dy,
            "importance": importance,
        }

    return stats, filter_used


def make_mask_global(
    stats,
    selected_layers,
    topk,
):
    matrix = np.stack(
        [
            stats[layer]["importance"]
            for layer in selected_layers
        ],
        axis=0,
    ).astype(np.float32)

    flat = matrix.reshape(-1)
    total = flat.size
    k = max(1, min(int(topk), total))

    chosen = np.argpartition(
        flat,
        total - k,
    )[total - k:]

    flat_mask = np.zeros(
        total,
        dtype=np.float32,
    )
    flat_mask[chosen] = 1.0

    mask_matrix = flat_mask.reshape(
        matrix.shape
    )

    masks = {
        layer: mask_matrix[i].astype(np.float32)
        for i, layer in enumerate(selected_layers)
    }

    counts = {
        layer: int(mask_matrix[i].sum())
        for i, layer in enumerate(selected_layers)
    }

    threshold = float(
        np.min(flat[chosen])
    )

    return masks, counts, threshold


def make_mask_balanced(
    stats,
    selected_layers,
    topk,
):
    """
    Approximately equal K allocation over layers.
    Remainder is assigned to earlier layers in selected_layers.
    """
    n_layers = len(selected_layers)

    base_k = int(topk) // n_layers
    remainder = int(topk) % n_layers

    masks = {}
    counts = {}
    thresholds = []

    for i, layer in enumerate(selected_layers):
        score = stats[layer]["importance"]
        dim = score.size

        layer_k = (
            base_k
            + (1 if i < remainder else 0)
        )
        layer_k = max(
            1,
            min(layer_k, dim),
        )

        chosen = np.argpartition(
            score,
            dim - layer_k,
        )[dim - layer_k:]

        mask = np.zeros(
            dim,
            dtype=np.float32,
        )
        mask[chosen] = 1.0

        masks[layer] = mask
        counts[layer] = int(mask.sum())
        thresholds.append(
            float(np.min(score[chosen]))
        )

    return (
        masks,
        counts,
        float(np.min(thresholds)),
    )


def make_spatial_mask(
    stats,
    selected_layers,
    topk,
    allocation,
):
    if allocation == "global":
        return make_mask_global(
            stats,
            selected_layers,
            topk,
        )

    if allocation == "balanced":
        return make_mask_balanced(
            stats,
            selected_layers,
            topk,
        )

    raise ValueError(allocation)


# =============================================================================
# Per-sample dynamic vector + hook
# =============================================================================

def sample_vectors(
    sid,
    delta_cache,
    layer_index,
    stats,
    selected_layers,
    masks,
    sample_mode,
):
    """
    Main:
        centered:
            q_i = Delta_i - mu_global

        raw:
            q_i = Delta_i

        v_i = q_i * mask
    """
    out = {}

    for layer in selected_layers:
        delta = delta_cache[sid][
            layer_index[layer]
        ].astype(np.float32)

        if sample_mode == "centered":
            q = (
                delta
                - stats[layer]["global"]
            ).astype(np.float32)

        elif sample_mode == "raw":
            q = delta

        else:
            raise ValueError(sample_mode)

        out[layer] = (
            q
            * masks[layer]
        ).astype(np.float32)

    return out


class SpatialSADIHook:
    """
    Add a PRECOMPUTED current-sample masked Real-Gray residual
    to the first/prefill last token at selected late layers.

        h' = h + alpha * v_i,l

    Only first call per layer is edited, matching the existing
    late-last-token intervention protocol.
    """

    def __init__(
        self,
        decoder_layers,
        selected_layers,
        vectors,
        strength,
    ):
        self.handles = []
        self.vectors = vectors
        self.strength = float(strength)

        self.done = {
            layer: False
            for layer in selected_layers
        }

        for layer in selected_layers:
            self.handles.append(
                decoder_layers[
                    layer
                ].register_forward_hook(
                    self._make_hook(layer)
                )
            )

    def _make_hook(self, layer):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[layer]:
                return output

            hidden, descriptor = (
                base.extract_hidden(output)
            )

            if hidden.ndim != 3:
                return output

            vector = torch.as_tensor(
                self.vectors[layer],
                device=hidden.device,
                dtype=hidden.dtype,
            )

            edited = hidden.clone()

            edited[:, -1, :] = (
                hidden[:, -1, :]
                + self.strength
                * vector[None, :]
            )

            self.done[layer] = True

            return base.replace_hidden(
                output,
                descriptor,
                edited,
            )

        return hook

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(
                Exception
            ):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Evaluation
# =============================================================================

def normalize_baseline_map(
    baseline_map,
    records,
    sids,
):
    out = {}

    for sid in sids:
        item = baseline_map[sid]
        pred = item.get("pred")

        correct = int(
            pred in RELSET
            and pred == records[sid]["gt"]
        )

        out[sid] = {
            "sid": sid,
            "pred": pred,
            "correct": correct,
        }

    return out


def run_spatial_sadi(
    model,
    processor,
    decoder_layers,
    delta_cache,
    layer_index,
    stats,
    masks,
    selected_layers,
    strength,
    sample_mode,
    records,
    sids,
    baseline_map,
    args,
    desc,
):
    rows = []

    for sid in tqdm(
        sids,
        desc=desc,
    ):
        rec = records[sid]
        image = None

        baseline_pred = baseline_map[
            sid
        ]["pred"]
        baseline_correct = int(
            baseline_map[sid]["correct"]
        )

        vectors = sample_vectors(
            sid=sid,
            delta_cache=delta_cache,
            layer_index=layer_index,
            stats=stats,
            selected_layers=selected_layers,
            masks=masks,
            sample_mode=sample_mode,
        )

        masked_norms = {
            layer: float(
                np.linalg.norm(
                    vectors[layer]
                )
            )
            for layer in selected_layers
        }

        try:
            image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            batch = base.build_batch(
                processor,
                image,
                rec,
                args,
            )

            with SpatialSADIHook(
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                vectors=vectors,
                strength=strength,
            ):
                text, pred = base.generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            edit_correct = int(
                pred == rec["gt"]
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "baseline_pred": baseline_pred or "",
                "edit_pred": pred or "",
                "baseline_correct": baseline_correct,
                "edit_correct": edit_correct,
                "W2C": int(
                    baseline_correct == 0
                    and edit_correct == 1
                ),
                "C2W": int(
                    baseline_correct == 1
                    and edit_correct == 0
                ),
                "changed": int(
                    (baseline_pred or "")
                    != (pred or "")
                ),
                "mean_masked_vector_norm": float(
                    np.mean(
                        list(
                            masked_norms.values()
                        )
                    )
                ),
                "max_masked_vector_norm": float(
                    np.max(
                        list(
                            masked_norms.values()
                        )
                    )
                ),
                "text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


def summarize(
    rows,
    topk,
    strength,
    condition,
    mask_counts,
):
    n = len(rows)

    base_acc = (
        sum(
            int(x["baseline_correct"])
            for x in rows
        )
        / n
        if n
        else float("nan")
    )

    edit_acc = (
        sum(
            int(x["edit_correct"])
            for x in rows
        )
        / n
        if n
        else float("nan")
    )

    w2c = sum(
        int(x["W2C"])
        for x in rows
    )
    c2w = sum(
        int(x["C2W"])
        for x in rows
    )

    return {
        "condition": condition,
        "N": n,
        "topk": int(topk),
        "strength": float(strength),
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
        "changed": sum(
            int(x["changed"])
            for x in rows
        ),
        "mean_masked_vector_norm": float(
            np.mean(
                [
                    x["mean_masked_vector_norm"]
                    for x in rows
                ]
            )
        )
        if rows
        else float("nan"),
        "mask_counts": ";".join(
            f"L{layer}:{count}"
            for layer, count
            in sorted(mask_counts.items())
        ),
    }


def choose_best(rows):
    """
    CAL selection:
      1. edited accuracy
      2. net W2C-C2W
      3. fewer C2W
      4. smaller K
      5. weaker intervention
    """
    return max(
        rows,
        key=lambda row: (
            float(row["edit_acc"]),
            int(row["net"]),
            -int(row["C2W"]),
            -int(row["topk"]),
            -abs(
                float(row["strength"])
            ),
        ),
    )


# Global only so fit_spatial_statistics can stay compact while
# still using CLI-selected mask metric.
args_global = None


def main():
    global args_global

    args = parse_args()
    args_global = args

    outdir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and outdir.exists()
    ):
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_test, baseline_path = (
        base.load_existing_test_baseline(
            args.prior_output_dir
        )
    )

    if existing_test is None:
        raise RuntimeError(
            "Could not reuse existing TEST baseline. "
            "Point --prior-output-dir to the earlier causal run."
        )

    train_generation, train_generation_path = (
        base.load_existing_train_generation(
            args.prior_output_dir
        )
    )

    records = base.load_all_records(
        args
    )

    train_sids, test_sids = (
        base.derive_train_test_ids(
            records,
            existing_test,
        )
    )

    model, processor = base.load_model(
        args
    )

    decoder_layers, decoder_path = (
        base.resolve_layers(model)
    )

    selected_layers = resolve_layers(
        args,
        decoder_layers,
    )

    layer_index = {
        layer: idx
        for idx, layer
        in enumerate(selected_layers)
    }

    print(
        "\n"
        + "=" * 140
    )
    print(
        "SPATIAL-SADI — RELATION-SENSITIVE MASK + "
        "SAMPLE REAL-GRAY RESIDUAL STEERING"
    )
    print(
        "=" * 140
    )
    print(
        f"model={args.model_id}"
    )
    print(
        f"decoder={decoder_path} | "
        f"blocks={len(decoder_layers)}"
    )
    print(
        f"layers={selected_layers}"
    )
    print(
        f"TRAIN={len(train_sids)} "
        f"TEST={len(test_sids)}"
    )
    print(
        f"mask_metric={args.mask_metric} | "
        f"allocation={args.mask_allocation} | "
        f"sample_mode={args.sample_mode}"
    )
    print(
        f"template_filter={args.template_filter}"
    )
    print(
        f"TEST baseline source={baseline_path}"
    )
    print(
        f"TRAIN Real/Gray labels={train_generation_path}"
    )
    print(
        "=" * 140
    )

    # All samples are cached because TEST inference itself needs each
    # sample's own Real-Gray residual. This uses no TEST GT.
    delta_cache = get_delta_cache(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        selected_layers=selected_layers,
        args=args,
        outdir=outdir,
    )

    fixed_mode = (
        args.fixed_topk > 0
        and math.isfinite(
            args.fixed_strength
        )
    )

    cal_summary_rows = []

    # =====================================================================
    # Fixed quick mode
    # =====================================================================
    if fixed_mode:
        selected_topk = int(
            args.fixed_topk
        )
        selected_strength = float(
            args.fixed_strength
        )

        final_stats, filter_used = (
            fit_spatial_statistics(
                delta_cache=delta_cache,
                layer_index=layer_index,
                records=records,
                fit_sids=train_sids,
                selected_layers=selected_layers,
                train_generation=train_generation,
                requested_filter=args.template_filter,
            )
        )

        final_masks, counts, threshold = (
            make_spatial_mask(
                stats=final_stats,
                selected_layers=selected_layers,
                topk=selected_topk,
                allocation=args.mask_allocation,
            )
        )

        print(
            f"\n[fixed] topk={selected_topk} | "
            f"strength={selected_strength:g} | "
            f"filter={filter_used}"
        )
        print(
            f"[fixed] mask counts={counts} | "
            f"threshold={threshold:.6g}"
        )

    # =====================================================================
    # FIT / CAL selection
    # =====================================================================
    else:
        fit_sids, cal_sids = (
            base.stratified_fit_cal_split(
                records=records,
                train_sids=train_sids,
                cal_frac=args.cal_frac,
                seed=args.seed,
            )
        )

        print(
            f"\n[FIT/CAL] FIT={len(fit_sids)} "
            f"CAL={len(cal_sids)}"
        )

        fit_stats, fit_filter = (
            fit_spatial_statistics(
                delta_cache=delta_cache,
                layer_index=layer_index,
                records=records,
                fit_sids=fit_sids,
                selected_layers=selected_layers,
                train_generation=train_generation,
                requested_filter=args.template_filter,
            )
        )

        print(
            f"[FIT spatial statistics] "
            f"filter_used={fit_filter}"
        )

        cal_baseline = (
            base.generate_baseline_for_sids(
                model=model,
                processor=processor,
                records=records,
                sids=cal_sids,
                args=args,
                desc="CAL baseline",
            )
        )

        topk_counts = sorted(
            set(
                parse_ints(
                    args.topk_counts
                )
            )
        )

        strengths = sorted(
            set(
                parse_floats(
                    args.strengths
                )
            )
        )

        if not topk_counts:
            raise ValueError(
                "No top-K values."
            )

        if not strengths:
            raise ValueError(
                "No strength values."
            )

        cal_detail_rows = []

        for topk in topk_counts:
            masks, counts, threshold = (
                make_spatial_mask(
                    stats=fit_stats,
                    selected_layers=selected_layers,
                    topk=topk,
                    allocation=args.mask_allocation,
                )
            )

            print(
                f"\n[mask] topk={topk} | "
                f"threshold={threshold:.6g} | "
                f"per_layer={counts}"
            )

            for strength in strengths:
                condition = (
                    f"cal_spatial_sadi_"
                    f"k{topk}_a{strength:g}"
                )

                rows = run_spatial_sadi(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    delta_cache=delta_cache,
                    layer_index=layer_index,
                    stats=fit_stats,
                    masks=masks,
                    selected_layers=selected_layers,
                    strength=strength,
                    sample_mode=args.sample_mode,
                    records=records,
                    sids=cal_sids,
                    baseline_map=cal_baseline,
                    args=args,
                    desc=condition,
                )

                summary = summarize(
                    rows=rows,
                    topk=topk,
                    strength=strength,
                    condition=condition,
                    mask_counts=counts,
                )

                summary[
                    "mask_threshold"
                ] = threshold

                cal_summary_rows.append(
                    summary
                )

                for row in rows:
                    row[
                        "condition"
                    ] = condition
                    row["topk"] = topk
                    row[
                        "strength"
                    ] = strength
                    cal_detail_rows.append(
                        row
                    )

                print(
                    f"{condition:34s} | "
                    f"{summary['base_acc']:.4f}"
                    f"->{summary['edit_acc']:.4f} "
                    f"{summary['gain']:+.4f} | "
                    f"W2C={summary['W2C']} "
                    f"C2W={summary['C2W']} "
                    f"net={summary['net']:+d} | "
                    f"changed={summary['changed']}"
                )

        write_csv(
            outdir
            / "cal_spatial_sadi_summary.csv",
            cal_summary_rows,
        )

        write_csv(
            outdir
            / "cal_spatial_sadi_details.csv",
            cal_detail_rows,
        )

        best = choose_best(
            cal_summary_rows
        )

        selected_topk = int(
            best["topk"]
        )
        selected_strength = float(
            best["strength"]
        )

        print(
            "\nSELECTED ON CAL"
        )
        print(
            "-" * 140
        )
        print(
            f"topk={selected_topk} | "
            f"strength={selected_strength:g} | "
            f"CAL {float(best['base_acc']):.4f}"
            f"->{float(best['edit_acc']):.4f} "
            f"{float(best['gain']):+.4f} | "
            f"W2C={int(best['W2C'])} "
            f"C2W={int(best['C2W'])} "
            f"net={int(best['net']):+d}"
        )

        # Refit spatial statistics on full original TRAIN.
        final_stats, final_filter = (
            fit_spatial_statistics(
                delta_cache=delta_cache,
                layer_index=layer_index,
                records=records,
                fit_sids=train_sids,
                selected_layers=selected_layers,
                train_generation=train_generation,
                requested_filter=args.template_filter,
            )
        )

        final_masks, counts, threshold = (
            make_spatial_mask(
                stats=final_stats,
                selected_layers=selected_layers,
                topk=selected_topk,
                allocation=args.mask_allocation,
            )
        )

        print(
            f"[FULL TRAIN refit] "
            f"filter_used={final_filter}"
        )
        print(
            f"[final mask] per_layer={counts} | "
            f"threshold={threshold:.6g}"
        )

    # =====================================================================
    # Save spatial mask/statistics
    # =====================================================================
    mask_rows = []

    for layer in selected_layers:
        indices = np.flatnonzero(
            final_masks[layer] > 0
        ).tolist()

        imp = final_stats[
            layer
        ]["importance"]

        selected_imp = (
            imp[indices]
            if indices
            else np.asarray(
                [],
                dtype=np.float32,
            )
        )

        mask_rows.append({
            "layer": layer,
            "selected_count": len(indices),
            "mean_selected_importance": (
                float(
                    selected_imp.mean()
                )
                if len(selected_imp)
                else float("nan")
            ),
            "min_selected_importance": (
                float(
                    selected_imp.min()
                )
                if len(selected_imp)
                else float("nan")
            ),
            "selected_indices": ",".join(
                map(str, indices)
            ),
        })

    write_csv(
        outdir
        / "final_spatial_sadi_mask.csv",
        mask_rows,
    )

    np.savez_compressed(
        outdir
        / "final_spatial_sadi_mask.npz",
        layers=np.asarray(
            selected_layers,
            dtype=np.int64,
        ),
        masks=np.stack(
            [
                final_masks[layer]
                for layer
                in selected_layers
            ],
            axis=0,
        ).astype(np.float32),
        global_mu=np.stack(
            [
                final_stats[
                    layer
                ]["global"]
                for layer
                in selected_layers
            ],
            axis=0,
        ).astype(np.float32),
        dx=np.stack(
            [
                final_stats[
                    layer
                ]["dx"]
                for layer
                in selected_layers
            ],
            axis=0,
        ).astype(np.float32),
        dy=np.stack(
            [
                final_stats[
                    layer
                ]["dy"]
                for layer
                in selected_layers
            ],
            axis=0,
        ).astype(np.float32),
        importance=np.stack(
            [
                final_stats[
                    layer
                ]["importance"]
                for layer
                in selected_layers
            ],
            axis=0,
        ).astype(np.float32),
        topk=np.asarray(
            [selected_topk],
            dtype=np.int64,
        ),
        strength=np.asarray(
            [selected_strength],
            dtype=np.float32,
        ),
    )

    # =====================================================================
    # TEST exactly once
    # =====================================================================
    test_baseline = (
        base.prepare_test_baseline(
            existing=existing_test,
            records=records,
            test_sids=test_sids,
        )
    )

    if test_baseline is None:
        test_baseline = (
            base.generate_baseline_for_sids(
                model=model,
                processor=processor,
                records=records,
                sids=test_sids,
                args=args,
                desc="TEST baseline fallback",
            )
        )

    test_baseline = normalize_baseline_map(
        test_baseline,
        records,
        test_sids,
    )

    test_rows = run_spatial_sadi(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        delta_cache=delta_cache,
        layer_index=layer_index,
        stats=final_stats,
        masks=final_masks,
        selected_layers=selected_layers,
        strength=selected_strength,
        sample_mode=args.sample_mode,
        records=records,
        sids=test_sids,
        baseline_map=test_baseline,
        args=args,
        desc="TEST Spatial-SADI",
    )

    test_summary = summarize(
        rows=test_rows,
        topk=selected_topk,
        strength=selected_strength,
        condition="test_spatial_sadi",
        mask_counts=counts,
    )

    test_summary.update({
        "layers": ",".join(
            map(
                str,
                selected_layers,
            )
        ),
        "mask_metric": args.mask_metric,
        "mask_allocation": args.mask_allocation,
        "sample_mode": args.sample_mode,
        "template_filter": args.template_filter,
    })

    write_csv(
        outdir
        / "test_spatial_sadi_summary.csv",
        [test_summary],
    )

    write_csv(
        outdir
        / "test_spatial_sadi_details.csv",
        test_rows,
    )

    print(
        "\n"
        + "=" * 140
    )
    print(
        "ACTUAL model.generate() — SPATIAL-SADI"
    )
    print(
        "=" * 140
    )
    print(
        f"layers={selected_layers} | "
        f"topk={selected_topk} | "
        f"strength={selected_strength:g} | "
        f"metric={args.mask_metric} | "
        f"allocation={args.mask_allocation} | "
        f"sample={args.sample_mode}"
    )
    print(
        f"acc "
        f"{test_summary['base_acc']:.4f}"
        f"->{test_summary['edit_acc']:.4f} "
        f"{test_summary['gain']:+.4f} | "
        f"W2C={test_summary['W2C']} "
        f"C2W={test_summary['C2W']} "
        f"net={test_summary['net']:+d} | "
        f"changed={test_summary['changed']}"
    )
    print(
        "=" * 140
    )

    print(
        f"[saved] "
        f"{outdir / 'cal_spatial_sadi_summary.csv'}"
    )
    print(
        f"[saved] "
        f"{outdir / 'test_spatial_sadi_summary.csv'}"
    )
    print(
        f"[saved] "
        f"{outdir / 'test_spatial_sadi_details.csv'}"
    )
    print(
        f"[saved] "
        f"{outdir / 'final_spatial_sadi_mask.npz'}"
    )


if __name__ == "__main__":
    main()
